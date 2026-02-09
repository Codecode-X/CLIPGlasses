import os
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from get_model import extract_img_features, extract_sentence_features, extract_objs_features, Clip_model


# Dataset class for training
class RetrievalNegGtDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.pos_csv_path = cfg['pos_csv_path']
        self.negpos_csv_path = cfg['negpos_csv_path'] 
        self.data = []
        self._preprocess_features()
        
    def _preprocess_features(self):
        cache_path = f"RetrievalNegGtDataset_cache.pt"
        if os.path.exists(cache_path):
            print(f"Loading Retrieval-gtNegObj dataset cache: {cache_path}...")
            self.data = torch.load(cache_path, weights_only=False)
            print(f"Loaded {len(self.data)} samples from cache")
            return
        
        df_np = pd.read_csv(self.negpos_csv_path)
        df_p = pd.read_csv(self.pos_csv_path)
        
        np_by_id = {}
        p_by_id = {}
        for _, row in df_np.iterrows():
            np_by_id[row['image_id']] = row
        for _, row in df_p.iterrows():
            p_by_id[row['image_id']] = row
        
        common_ids = set(np_by_id.keys()) & set(p_by_id.keys())
        
        for img_id in tqdm(common_ids, desc="Processing data"):
            np_row = np_by_id[img_id]
            p_row = p_by_id[img_id]
            
            np_captions = eval(np_row['captions'])
            p_captions = eval(p_row['captions'])
            
            neg_object_list = eval(np_row['negative_objects'])
            neg_objs_list = extract_objs_features(neg_object_list)
            for np_cap, p_cap in zip(np_captions, p_captions):
                h, level_h_list = extract_sentence_features(np_cap)
                h = torch.tensor(h, dtype=self.cfg['dtype'])
                l_pos, _ = extract_sentence_features(p_cap) # [embed_dim]
                img_path = np_row['filepath']
                I = extract_img_features(image_path=np_row['filepath'])
                I = torch.tensor(I, dtype=self.cfg['dtype']) # [embed_dim]
                
                biggest_sim = -float('inf')
                correct_neg_obj, correct_neg_obj_str = None, None
                for i, neg_obj in enumerate(neg_objs_list):
                    neg_obj = torch.tensor(neg_obj, dtype=self.cfg['dtype']) # [embed_dim]
                    sim = F.cosine_similarity(h, neg_obj, dim=-1)
                    if sim > biggest_sim:
                        biggest_sim = sim
                        correct_neg_obj = neg_obj
                        correct_neg_obj_str = neg_object_list[i] # Corresponding neg_object
                if correct_neg_obj is None: 
                    correct_neg_obj = torch.zeros_like(h) 
                self.data.append({'I': I, 'h': h, 'level_h_list': level_h_list, 'l_pos': l_pos, 'neg_obj': correct_neg_obj, 'img_path': img_path, 'img_id': img_id})
        
        print(f"Saving preprocessed features to cache: {cache_path}")  
        torch.save(self.data, cache_path)
        print(f"Preprocessed features saved to {cache_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'I': self.data[idx]['I'],
            'h': self.data[idx]['h'],
            'level_h_list': torch.stack([torch.tensor(l, dtype=self.cfg['dtype']) for l in self.data[idx]['level_h_list']]),  # EOS features from each layer of the CLIP text encoder
            'l_pos': torch.tensor(self.data[idx]['l_pos'], dtype=self.cfg['dtype']),
            'neg_obj': self.data[idx]['neg_obj'].to(dtype=self.cfg['dtype']),
            'img_path': self.data[idx]['img_path'],
            'img_id': self.data[idx]['img_id']
        }


def evaluate_model_retrieval_withGTNeg(model, data_loader, test_raw_clip=False, with_gt_neg=True, device='cuda'):
    txt2img_recalls = {1: 0, 5: 0, 10: 0}
    img2txt_recalls = {1: 0, 5: 0, 10: 0}
    all_image_feats = []
    all_text_feats  = [] 
    all_neg_feats = [] 
    all_level_text_feats = []
    caption_to_img  = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"TESTING", total=len(data_loader)):
            caption_feats = batch['h'].to(device) #
            level_H_list = batch['level_h_list'].to(device) 
            l_pos = batch['l_pos'].to(device) #
            l_neg = batch['neg_obj'].to(device) 
            image_feats = batch['I'].to(device)
            image_ids = batch['img_id'].to(device)
            caption_to_img.extend(image_ids.cpu().numpy())
            all_image_feats.extend(image_feats.cpu().numpy())
            all_text_feats.extend(caption_feats.cpu().numpy())
            all_neg_feats.extend(l_neg.cpu().numpy())
            all_level_text_feats.extend(level_H_list.cpu().numpy())
        
        caption_to_img = torch.tensor(caption_to_img, dtype=torch.long)
        unique_img_ids, remapped_ids = torch.unique(caption_to_img, sorted=True, return_inverse=True)
        caption_to_img = remapped_ids.cpu().numpy()
        N_imgs = len(unique_img_ids) 
        N_caps = len(caption_to_img)
        all_image_feats = [torch.from_numpy(f) for f in all_image_feats]
        all_text_feats = [torch.from_numpy(f) for f in all_text_feats]
        all_neg_feats = [torch.from_numpy(f) for f in all_neg_feats]
        all_level_text_feats = [torch.from_numpy(f) for f in all_level_text_feats]
        I = torch.stack(all_image_feats, dim=0).to(device)
        I_rep = I 
        h = torch.stack(all_text_feats, dim=0).to(device) 
        l_neg = torch.stack(all_neg_feats, dim=0).to(device)
        level_h = torch.stack(all_level_text_feats, dim=0).to(device)
      
        if test_raw_clip:
            I_norm = F.normalize(I_rep, p=2, dim=-1)
            h_norm = F.normalize(h, p=2, dim=-1)
            logit_scale = Clip_model.logit_scale.exp()
            scores_T2I = logit_scale * (h_norm @ I_norm.t())
            scores_I2T = scores_T2I.t()
        else:
            model.eval()
            if with_gt_neg:
                scores_T2I, scores_I2T = model(I_rep, h, level_h, l_neg) 
            else:
                scores_T2I, scores_I2T = model(I_rep, h, level_h) 
        cti = torch.tensor(caption_to_img, dtype=torch.long, device=device) 
        unique_vals = torch.unique(cti, sorted=True)
        first_idx = []
        for val in unique_vals:
            idx = (cti == val).nonzero(as_tuple=True)[0][0]
            first_idx.append(idx)
        first_idx = torch.stack(first_idx, dim=0) 
        scores_T2I = scores_T2I[:, first_idx] 
        scores_I2T = scores_T2I.t()
        txt2img_hits = {1:0, 5:0, 10:0}
        img2txt_hits = {1:0, 5:0, 10:0}
        for cap_idx, gt_img in enumerate(caption_to_img):
            scores = scores_T2I[cap_idx]
            neg_text_feats, topk = torch.topk(scores, k=10, largest=True)
            for k in txt2img_hits:
                if gt_img in topk[:k]:
                    txt2img_hits[k] += 1
        img2cap = [[] for _ in range(N_imgs)] 
        for cap_idx, img_idx in enumerate(caption_to_img):
            img2cap[img_idx].append(cap_idx)
        for img_idx in range(N_imgs):
            if not img2cap[img_idx]:
                continue
            scores = scores_I2T[img_idx]
            neg_text_feats, topk = torch.topk(scores, k=10, largest=True)
            for k in img2txt_hits:
                if any(cap in topk[:k] for cap in img2cap[img_idx]):
                    img2txt_hits[k] += 1
    total_caps = float(N_caps)
    total_imgs = float(N_imgs)
    txt2img_recalls = {k: txt2img_hits[k]/total_caps*100 for k in txt2img_hits}
    img2txt_recalls = {k: img2txt_hits[k]/total_imgs*100 for k in img2txt_hits}

    return {
        'txt2img': txt2img_recalls,
        'img2txt': img2txt_recalls,
        'mean':   {k:(txt2img_recalls[k]+img2txt_recalls[k])/2 for k in txt2img_recalls}
    }