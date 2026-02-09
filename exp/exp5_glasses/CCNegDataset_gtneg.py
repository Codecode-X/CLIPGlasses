import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from get_model import extract_img_features, extract_sentence_features, extract_objs_features, Clip_model
import numpy as np

# Dataset class for training
class CCNegGtDataset(Dataset):
    """ CCNegDataset + GT negative object + negative photo """
    def __init__(self, cfg):
        self.cfg = cfg
        self.csv_path = cfg['csv_path']  # COCO_val_retrieval.csv
        self.data = []  # Populated in _preprocess_features()
        
        self.negatives_mapping_cc3m_to_coco = torch.load(cfg['negative_image_ft_mapping_path'], weights_only=False)  # Negative image indices corresponding to negative text
        
        self._preprocess_features()
        
    def _preprocess_features(self):

        # Create cache file path based on CSV path
        cache_path = f"CCNegGtDataset_cache.pt"
        # Check if cache file exists
        if os.path.exists(cache_path):
            print(f"Loading CC-Neg dataset cache: {cache_path}...")
            self.data = torch.load(cache_path, weights_only=False)
            print(f"Loaded {len(self.data)} samples from cache")
            return
        
        # Read CSV files
        df = pd.read_csv(self.csv_path)
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing features"):
            # str
            caption_p = eval(row['captions'])[0]  # Positive text
            caption_n = eval(row['n_captions'])[0]  # Negative text
            img_path = row['filepath']
            image_id = row['image_id']
            neg_object = eval(row['negative_objects'])[0]
            # Extract features
            I = extract_img_features(image_path=img_path)  # [embed_dim]
            hp, level_hp_list = extract_sentence_features(caption_p)  # [embed_dim] # Positive text features
            hn, level_hn_list = extract_sentence_features(caption_n)  # [embed_dim] # Negative text features
            neg_obj = extract_objs_features([neg_object])[0]  # [num_objs, embed_dim]
            # Convert to tensor
            I = torch.tensor(I, dtype=self.cfg['dtype'])  # [embed_dim]
            hp = torch.tensor(hp, dtype=self.cfg['dtype'])  # [embed_dim]
            hn = torch.tensor(hn, dtype=self.cfg['dtype'])  # [embed_dim]
            level_hp_list = torch.stack([torch.tensor(l, dtype=self.cfg['dtype']) for l in level_hp_list])  # [num_layers, embed_dim]
            level_hn_list = torch.stack([torch.tensor(l, dtype=self.cfg['dtype']) for l in level_hn_list])  # [num_layers, embed_dim]
            neg_obj = torch.tensor(neg_obj, dtype=self.cfg['dtype'])  # [embed_dim]
            
            # Dimension validation
            assert I.shape == (512,), f"Image feature dimension error: {I.shape}"
            assert hp.shape == (512,), f"Positive text feature dimension error: {hp.shape}"
            assert hn.shape == (512,), f"Negative text feature dimension error: {hn.shape}"

            # Append to data list
            self.data.append({'I': I, 'hp': hp, 'hn': hn, 'level_hp_list': level_hp_list, 'level_hn_list': level_hn_list, 'l_pos': hp, 'l_neg': hn, 'neg_obj': neg_obj, 'img_path': img_path, 'img_id': image_id})
        
        # Save preprocessed features to cache
        print(f"Saving preprocessed features to cache: {cache_path}")  
        torch.save(self.data, cache_path)
        print(f"Preprocessed features saved to {cache_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get the negative image index corresponding to the negative text hn using negatives_mapping_cc3m_to_coco
        topk_indices = self.negatives_mapping_cc3m_to_coco[idx]
        top1_index = topk_indices[0]
        
        return {
            'Ip': self.data[idx]['I'],  # Image features [embed_dim]
            'In': self.data[top1_index]['I'],  # Negative image features [embed_dim]   
            'hp': self.data[idx]['hp'],  # Positive text features [embed_dim]
            'hn': self.data[idx]['hn'],  # Negative text features with interference [embed_dim]
            'level_hp_list': self.data[idx]['level_hp_list'],  # List of positive text features (per layer) [num_layers, embed_dim]
            'level_hn_list': self.data[idx]['level_hn_list'],  # List of negative text features with interference (per layer) [num_layers, embed_dim]
            'l_pos': self.data[idx]['l_pos'],  # Positive text features [embed_dim]
            'l_neg': self.data[idx]['l_neg'],  # Negative text features with interference [embed_dim]
            'neg_obj': self.data[idx]['neg_obj'],  # Negative object text features [num_objs, embed_dim]
            'img_path': self.data[idx]['img_path'],  # Image path
            'img_id': self.data[idx]['img_id']  # Image ID
        }
        
        
def evaluate_model_CCNeg_etrieval_withGTNeg(model, data_loader, test_raw_clip=False, with_gt_neg=True, device='cuda'):
    """
    Evaluate the model's retrieval task performance metrics on the CCNeg dataset:
        - Accuracy of image-to-text retrieval: Each image corresponds to one hp (correct matching text) and one hn (negative text with interference). Accuracy is the correct selection rate between hp and hn.
        - Mean difference between Sp (similarity of image and hp) and Sn (similarity of image and hn).
    """
    # Collect all features
    all_image_feats = []
    all_hp_feats = []  # Positive text features
    all_hn_feats = []  # Negative text features
    all_level_hp_feats = []  # Per-layer positive text features
    all_level_hn_feats = []  # Per-layer negative text features
    all_neg_obj_feats = []  # Negative object features
    
    with torch.no_grad():
        # Iterate through each batch to collect features
        for batch in tqdm(data_loader, desc="Extracting features", total=len(data_loader)):
            image_feats = batch['Ip'].to(device)  # [batch_size, embed_dim]
            hp_feats = batch['hp'].to(device)    # [batch_size, embed_dim]
            hn_feats = batch['hn'].to(device)    # [batch_size, embed_dim]
            level_hp_list = batch['level_hp_list'].to(device)  # [batch_size, num_layers, embed_dim]
            level_hn_list = batch['level_hn_list'].to(device)  # [batch_size, num_layers, embed_dim]
            neg_obj_feats = batch['neg_obj'].to(device)  # [batch_size, embed_dim]
            img_ids = batch['img_id']
            
            all_image_feats.append(image_feats)  # [batch_size, embed_dim] * batch_num
            all_hp_feats.append(hp_feats)
            all_hn_feats.append(hn_feats)
            all_level_hp_feats.append(level_hp_list)  # [batch_size, num_layers, embed_dim] * batch_num
            all_level_hn_feats.append(level_hn_list)
            all_neg_obj_feats.append(neg_obj_feats)
    
        I = torch.cat(all_image_feats, dim=0)  # [N, embed_dim]
        hp = torch.cat(all_hp_feats, dim=0)  # [N, embed_dim]
        hn = torch.cat(all_hn_feats, dim=0)  # [N, embed_dim]
        assert not torch.equal(hp, hn), print(f"hp and hn should not be equal, \nhp = {hp}, \nhn = {hn}")
        level_hp = torch.cat(all_level_hp_feats, dim=0)  # [N, num_layers, embed_dim]
        level_hn = torch.cat(all_level_hn_feats, dim=0)  # [N, num_layers, embed_dim]
        neg_obj = torch.cat(all_neg_obj_feats, dim=0)  # [N, embed_dim]
        
        # Compute similarity scores
        if test_raw_clip:
            # Use raw CLIP to compute similarity
            print("Testing raw CLIP model")
            I_norm = F.normalize(I, p=2, dim=-1)
            hp_norm = F.normalize(hp, p=2, dim=-1)
            hn_norm = F.normalize(hn, p=2, dim=-1)
            
            logit_scale = Clip_model.logit_scale.exp()
            scores_I_hp = logit_scale * (I_norm @ hp_norm.t())  # I2T [num_images=N, num_texts=N]
            scores_I_hn = logit_scale * (I_norm @ hn_norm.t())  
        else:
            # Use the provided model to compute similarity
            print("Testing Glasses model")
            model.eval()
            if with_gt_neg:
                # Use GT negative objects
                _, scores_I_hp = model(I, hp, level_hp, neg_obj, chunk_size=-1)  # I2T [num_images=N, num_texts=N]
                _, scores_I_hn = model(I, hn, level_hn, neg_obj, chunk_size=-1) 
            else:
                # Use model-predicted negative objects
                _, scores_I_hp = model(I, hp, level_hp, chunk_size=-1)  # I2T [num_images=N, num_texts=N]
                _, scores_I_hn = model(I, hn, level_hn, chunk_size=-1) 
        
        # Compute recall@1, recall@5, recall@10 for all positive and negative texts
        num_images = I.size(0)
        
        assert scores_I_hp.shape == (num_images, num_images), f"Dimension error: {scores_I_hp.shape}"
        assert scores_I_hn.shape == (num_images, num_images), f"Dimension error: {scores_I_hn.shape}"

        # Combine all text features and scores
        scores_combined = torch.cat([scores_I_hp, scores_I_hn], dim=1)  # I2T [num_images=N, num_texts=2N]
        scores_combined = scores_combined.to(device)
        
        # For each image, the correct matching text index should be i
        correct_indices = torch.arange(num_images, device=device)
        
        # Compute accuracy: The highest score for image i should be with positive text hp[i]
        _, top_indices = scores_combined.topk(k=1, dim=1)
        global_r1 = (top_indices.squeeze() == correct_indices).float().mean().item() * 100
        
        # Compute recall@5, recall@10 for all positive and negative texts
        _, top_indices = scores_combined.topk(k=5, dim=1)
        global_r5 = (top_indices == correct_indices.unsqueeze(1)).float().sum(dim=1).mean().item() * 100
        
        _, top_indices = scores_combined.topk(k=10, dim=1)
        global_r10 = (top_indices == correct_indices.unsqueeze(1)).float().sum(dim=1).mean().item() * 100
          
        # Compute the mean difference between Sp and Sn and prediction accuracy
        diag_scores_hp = torch.diag(scores_I_hp)  # [N]
        diag_scores_hn = torch.diag(scores_I_hn)  # [N]
        mean_diff = (diag_scores_hp - diag_scores_hn).mean().item()
        accuracy = (diag_scores_hp > diag_scores_hn).float().mean().item() * 100
        
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