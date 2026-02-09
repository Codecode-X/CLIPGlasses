import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from get_model import extract_img_features, extract_sentence_features, extract_objs_features, Clip_model
import numpy as np

class CCNegGtDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.csv_path = cfg['csv_path']
        self.data = []
        self.negatives_mapping_cc3m_to_coco = torch.load(cfg['negative_image_ft_mapping_path'], weights_only=False)
        self._preprocess_features()
        
    def _preprocess_features(self):
        cache_path = f"CCNegGtDataset_cache.pt"
        if os.path.exists(cache_path):
            self.data = torch.load(cache_path, weights_only=False)
            return
        
        df = pd.read_csv(self.csv_path)
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing features"):
            caption_p = eval(row['captions'])[0]
            caption_n = eval(row['n_captions'])[0]
            img_path = row['filepath']
            image_id = row['image_id']
            neg_object = eval(row['negative_objects'])[0]
            I = extract_img_features(image_path=img_path)
            hp, level_hp_list = extract_sentence_features(caption_p)
            hn, level_hn_list = extract_sentence_features(caption_n)
            neg_obj = extract_objs_features([neg_object])[0]
            I = torch.tensor(I, dtype=self.cfg['dtype'])
            hp = torch.tensor(hp, dtype=self.cfg['dtype'])
            hn = torch.tensor(hn, dtype=self.cfg['dtype'])
            level_hp_list = torch.stack([torch.tensor(l, dtype=self.cfg['dtype']) for l in level_hp_list])
            level_hn_list = torch.stack([torch.tensor(l, dtype=self.cfg['dtype']) for l in level_hn_list])
            neg_obj = torch.tensor(neg_obj, dtype=self.cfg['dtype'])
            self.data.append({'I': I, 'hp': hp, 'hn': hn, 'level_hp_list': level_hp_list, 'level_hn_list': level_hn_list, 'l_pos': hp, 'l_neg': hn, 'neg_obj': neg_obj, 'img_path': img_path, 'img_id': image_id})
        torch.save(self.data, cache_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        topk_indices = self.negatives_mapping_cc3m_to_coco[idx]
        top1_index = topk_indices[0]
        return {
            'Ip': self.data[idx]['I'],
            'In': self.data[top1_index]['I'],
            'hp': self.data[idx]['hp'],
            'hn': self.data[idx]['hn'],
            'level_hp_list': self.data[idx]['level_hp_list'],
            'level_hn_list': self.data[idx]['level_hn_list'],
            'l_pos': self.data[idx]['l_pos'],
            'l_neg': self.data[idx]['l_neg'],
            'neg_obj': self.data[idx]['neg_obj'],
            'img_path': self.data[idx]['img_path'],
            'img_id': self.data[idx]['img_id']
        }
        
def evaluate_model_CCNeg_etrieval_withGTNeg(model, data_loader, test_raw_clip=False, with_gt_neg=True, chunk_size=-1, device='cuda'):
    all_image_feats = []
    all_hp_feats = []
    all_hn_feats = []
    all_level_hp_feats = []
    all_level_hn_feats = []
    all_neg_obj_feats = []
    all_img_ids = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting features", total=len(data_loader)):
            image_feats = batch['Ip'].to(device)
            hp_feats = batch['hp'].to(device)
            hn_feats = batch['hn'].to(device)
            level_hp_list = batch['level_hp_list'].to(device)
            level_hn_list = batch['level_hn_list'].to(device)
            neg_obj_feats = batch['neg_obj'].to(device)
            img_ids = batch['img_id']
            all_image_feats.append(image_feats)
            all_hp_feats.append(hp_feats)
            all_hn_feats.append(hn_feats)
            all_level_hp_feats.append(level_hp_list)
            all_level_hn_feats.append(level_hn_list)
            all_neg_obj_feats.append(neg_obj_feats)
            all_img_ids.extend(img_ids.tolist())
    
        I = torch.cat(all_image_feats, dim=0)
        hp = torch.cat(all_hp_feats, dim=0)
        hn = torch.cat(all_hn_feats, dim=0)
        assert not torch.equal(hp, hn), print(f"hp and hn should not be equal, \nhp = {hp}, \nhn = {hn}")
        level_hp = torch.cat(all_level_hp_feats, dim=0)
        level_hn = torch.cat(all_level_hn_feats, dim=0)
        neg_obj = torch.cat(all_neg_obj_feats, dim=0)
        
        if test_raw_clip:
            I_norm = F.normalize(I, p=2, dim=-1)
            hp_norm = F.normalize(hp, p=2, dim=-1)
            hn_norm = F.normalize(hn, p=2, dim=-1)
            logit_scale = Clip_model.logit_scale.exp()
            scores_I_hp = logit_scale * (I_norm @ hp_norm.t())
            scores_I_hn = logit_scale * (I_norm @ hn_norm.t())  
        else:
            model.eval()
            if with_gt_neg:
                _, scores_I_hp = model(I, hp, level_hp, neg_obj, chunk_size=chunk_size)
                _, scores_I_hn = model(I, hn, level_hn, neg_obj, chunk_size=chunk_size) 
            else:
                _, scores_I_hp = model(I, hp, level_hp, chunk_size=chunk_size)
                _, scores_I_hn = model(I, hn, level_hn, chunk_size=chunk_size) 
        
        num_images = I.size(0)
        assert scores_I_hp.shape == (num_images, num_images), f"dimensions error:{scores_I_hp.shape}"
        assert scores_I_hn.shape == (num_images, num_images), f"dimensions error:{scores_I_hn.shape}"
        scores_combined = torch.cat([scores_I_hp, scores_I_hn], dim=1)
        scores_combined = scores_combined.to(device)
        correct_indices = torch.arange(num_images, device=device)
        _, top_indices = scores_combined.topk(k=1, dim=1)
        global_r1 = (top_indices.squeeze() == correct_indices).float().mean().item() * 100
        _, top_indices = scores_combined.topk(k=5, dim=1)
        global_r5 = (top_indices == correct_indices.unsqueeze(1)).float().sum(dim=1).mean().item() * 100
        _, top_indices = scores_combined.topk(k=10, dim=1)
        global_r10 = (top_indices == correct_indices.unsqueeze(1)).float().sum(dim=1).mean().item() * 100
        diag_scores_hp = torch.diag(scores_I_hp)
        diag_scores_hn = torch.diag(scores_I_hn)
        mean_diff = (diag_scores_hp - diag_scores_hn).mean().item()
        accuracy = (diag_scores_hp > diag_scores_hn).float().mean().item() * 100
        wrong_indices = (diag_scores_hp <= diag_scores_hn).nonzero(as_tuple=True)[0].tolist()
        wrong_scores_hp = diag_scores_hp[wrong_indices]
        wrong_scores_hn = diag_scores_hn[wrong_indices]
        diffs = wrong_scores_hp - wrong_scores_hn
        wrong_img_ids = [all_img_ids[i] for i in wrong_indices]
        print(f"Number of misclassified samples: {len(wrong_img_ids)}")
        print("Misclassified img_ids (first 50):", wrong_img_ids[:50])
        print(f"Mean Sp (hp score) for misclassified samples: {wrong_scores_hp.mean().item():.4f}")
        print(f"Mean Sn (hn score) for misclassified samples: {wrong_scores_hn.mean().item():.4f}")
        print(f"Mean Sp - Sn for misclassified samples: {diffs.mean().item():.4f}")
        print(f"Max Sp - Sn (misclassified): {diffs.max().item():.4f}, Min: {diffs.min().item():.4f}")
        print("="*50)
        print(f"global Retrieval Recall@1,5,10: {global_r1:.2f}% {global_r5:.2f}% {global_r10:.2f}%")
        print(f"Select Accuracy: {accuracy:.2f}%")
        print(f"Mean Similarity Difference (Sp-Sn): {mean_diff:.4f}")
        
        return {
            'global_R1': global_r1,
            'global_R5': global_r5,
            'global_R10': global_r10,
            'accuracy': accuracy,
            'mean_diff': mean_diff
        }
