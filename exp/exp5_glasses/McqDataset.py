import os
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from get_model import extract_img_features, extract_sentence_features


ANSWERTYPE_ID_MAP = {
    'hybrid': 0,
    'negative': 1,
    'positive': 2
}

class McqDataset(Dataset):
    def __init__(self, csv_path, transform=None, device='cuda'):
        self.device = device
        self.csv_path = csv_path
        try:
            self.data = pd.read_csv(csv_path, encoding='gbk')
        except UnicodeDecodeError:
            self.data = pd.read_csv(csv_path, encoding='utf-8')
        self.transform = transform
        self._preprocess_features()
        
    def _preprocess_features(self):
        csv_hash = hashlib.md5(self.data.to_json().encode()).hexdigest()[:10]
        cache_path = f"MCQ_features_cache_{csv_hash}.pt"
        if os.path.exists(cache_path):
            print(f"Loading MCQ dataset cache: {cache_path} of {self.csv_path} ...")
            cached_data = torch.load(cache_path, weights_only=False)
            self.image_features = cached_data['image_features']
            self.captions_feats = cached_data['captions_feats']
            self.level_H = cached_data['level_H_list']
            self.labels = cached_data['labels']
            self.types = cached_data['types']
            print(f"Loaded {len(self.image_features)} samples from cache")
            return
        
        self.image_features = []
        self.captions_feats = [] 
        self.level_H = []
        self.labels = []
        self.types = []
        for idx, row in tqdm(self.data.iterrows(), total=len(self.data)):
            img_path = row['image_path']
            img_feature = extract_img_features(img_path)
            self.image_features.append(img_feature)
            row_h = []
            row_level_h = []
            for i in range(4):
                caption = row[f'caption_{i}']
                h, level_h_list = extract_sentence_features(caption)
                row_h.append(h) 
                row_level_h.append(level_h_list)
            self.captions_feats.append(row_h)
            self.level_H.append(row_level_h)
            self.labels.append(row['correct_answer'])
            self.types.append(ANSWERTYPE_ID_MAP.get(row['correct_answer_template'], -1))
        print(f"Saving features to cache: {cache_path}")
        torch.save({
            'image_features': self.image_features,
            'captions_feats': self.captions_feats,
            'level_H_list': self.level_H,
            'labels': self.labels,
            'types': self.types
        }, cache_path)
        print(f"Cached {len(self.image_features)} samples.")
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_feature = torch.tensor(self.image_features[idx], dtype=torch.float32)
        caption_feat = torch.from_numpy(np.array(self.captions_feats[idx])).float() # [num_captions=4, embed_dim]
        level_h_list = torch.from_numpy(np.array(self.level_H[idx])).float() # [num_captions=4, num_layers, embed_dim]
        label = torch.tensor(self.labels[idx], dtype=torch.long) # [1]
        answer_type = torch.tensor(self.types[idx], dtype=torch.long) # [1]
        
        return img_feature, caption_feat, level_h_list, label, answer_type
    
    
def evaluate_model_mcq(model, data_loader, test_raw_clip=False, device='cuda'):
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for img_features, caption_feats, level_H_list, labels, answer_types in tqdm(data_loader, desc="Extract feats"):       
            img_features = img_features.to(device)
            caption_feats = caption_feats.to(device)
            level_H_list = level_H_list.to(device)
            labels = labels.to(device)
            answer_types = answer_types.to(device) # [batch_size]
            
            num_options = caption_feats.shape[1]
            all_scores = [None] * num_options 
            for i in range(num_options):
                choice_h = caption_feats[:, i] 
                level_h_list = level_H_list[:, i]
                scores, _ = model(img_features, choice_h, level_h_list) 
                all_scores[i] = scores
            logits = torch.stack([s.diag() for s in all_scores], dim=1)
            loss = F.cross_entropy(logits, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(logits, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_acc = 100. * val_correct / val_total
    return val_acc, val_loss, all_predictions, all_labels