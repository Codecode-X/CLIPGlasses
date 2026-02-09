import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)
sys.path.insert(0, parent_dir)
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
import hashlib
from get_model import extract_img_features, extract_sentence_features, Clip_model



class CLSDataset(Dataset):
    def __init__(self, cfg):
        """
        Args:
            csv_path: Path to the ImageNet dataset csv file
        """
        self.csv_path = cfg['csv_path']
        self.df = pd.read_csv(self.csv_path)
        
        # # Randomly select 5000 samples
        # if len(self.df) > 5000:
        #     self.df = self.df.sample(n=5000, random_state=3407).reset_index(drop=True)

        # Extract unique classes and create a mapping
        self.class_list = self.df['label'].unique().tolist()
        self.class_to_id = {cls: idx for idx, cls in enumerate(self.class_list)}

        # Preprocess and cache features
        self._preprocess_features()
        
    def _preprocess_features(self):
        """
        Preprocess and cache all image and text features
            - If preprocessed data file exists, load it directly
            - Otherwise extract features and save preprocessed data for future use
        """
        # Create cache file path based on CSV path
        csv_hash = hashlib.md5(self.df.to_json().encode()).hexdigest()[:10]
        cache_path = f"imagenet_cls_features_cache_{csv_hash}.pt"
        
        # Check if cache file exists
        if os.path.exists(cache_path):
            print(f"Loading preprocessed features from cache: {cache_path} of {self.csv_path} ...")
            cached_data = torch.load(cache_path, weights_only=False)
            with torch.no_grad():
                self.image_features = cached_data['image_features']
                self.labels = cached_data['labels']
                self.text_features = cached_data['text_features']
                self.level_H = cached_data['level_H_list']
            print(f"Loaded {len(self.labels)} samples from cache")
            return
        
        print(f"Preprocessing dataset features of {self.csv_path} ...")
        self.image_features = []
        self.text_features = []
        self.level_H = [] # Original CLIP image feature list for each layer
        self.labels = []
        
        # Extract image features for all images
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
            # Extract image features
            img_path = str(row['image_path'])
            img_features = extract_img_features(img_path)
            # Store image features and label
            self.image_features.append(img_features)
            self.labels.append(self.class_to_id[row['label']])
        
        # Extract text features for all classes
        print("Extracting text features for all classes...")
        for class_name in tqdm(self.class_list):
            # Create prompt for class
            prompt = f"A photo of a {class_name.replace('_', ' ')}."
            # Extract text features
            h, level_h  = extract_sentence_features(prompt)
            self.text_features.append(h)
            self.level_H.append(level_h)
        
        # Save preprocessed features to cache
        print(f"Saving preprocessed features to cache: {cache_path}")
        torch.save({
            'image_features': self.image_features,
            'labels': self.labels,
            'text_features': self.text_features, # [num_classes, embed_dim]
            'level_H_list': self.level_H # [num_classes, embed_dim]
        }, cache_path)
        print(f"Cached {len(self.labels)} samples with {len(self.class_list)} classes")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Returns:
            image_features: Image features [embed_dim]
            text_features: Text features [num_classes, embed_dim]
            level_H: Level H features [num_classes, num_layers, embed_dim]
            label: Class label (long)
        """
        img_features = torch.tensor(self.image_features[idx], dtype=torch.float32) # [embed_dim]
        text_features = torch.tensor(np.array(self.text_features), dtype=torch.float32) # [num_classes, embed_dim]
        level_H = torch.tensor(np.array(self.level_H), dtype=torch.float32) # [num_classes, num_layers, embed_dim]
        label = torch.tensor(self.labels[idx], dtype=torch.long) # long
        return img_features, text_features, level_H, label


def evaluate_model_CLS(model, data_loader, test_raw_clip=False, device='cuda', dtype=torch.float32):
    """
    Evaluate the CLIPGlassesFrame model on the validation set.
    
    Args:
        - model: The CLIPGlassesFrame model
        - data_loader: DataLoader for the validation set
        - test_raw_clip: Whether to test the raw CLIP model
        - device: Device to run the model on (CPU or GPU)
        
    Returns:
        - val_acc: Validation accuracy
        - val_loss: Validation loss
        - all_predictions: List of all predictions
        - all_labels: List of all true labels
    """
    val_loss = 0
    val_correct = 0
    val_total = 0
    all_predictions = []
    all_labels = []
    
    if test_raw_clip:
        print("Testing the classification ability of the raw CLIP model")
    else:
        print("Testing the classification ability of CLIPGlasses")
        model.eval()
    
    with torch.no_grad():
        for img_features, all_class_feats, level_H, labels in tqdm(data_loader, desc="Extract feats"):
            """
                img_features: Image features [batch_size, embed_dim]
                all_class_feats: Text features [batch_size, num_options, embed_dim]
                level_H: Level H features [batch_size, num_options, num_layers, embed_dim]
                labels: Class labels [batch_size]
            """            
            img_features = img_features.to(device, dtype=dtype) # [batch_size, embed_dim]
            all_class_feats = all_class_feats.to(device, dtype=dtype) # [batch_size, num_options, embed_dim]
            level_H = level_H.to(device, dtype=dtype) # [batch_size, num_options, num_layers, embed_dim]
            labels = labels.to(device) # [batch_size]
            
            # Get scores for all options
            num_cls = all_class_feats.shape[1]
            
            if test_raw_clip: # 53.68%
                img_features = F.normalize(img_features, p=2, dim=-1) # [num_imgs=B, embed_dim]
                all_class_feats = F.normalize(all_class_feats[0], p=2, dim=-1) # [num_options, embed_dim]
                logit_scale = Clip_model.logit_scale.exp()
                logits_per_image = logit_scale * (img_features @ all_class_feats.t()) # [num_imgs=B, num_options]
            else: # 53.20%
                all_scores = [None] * num_cls  # [batch_size, batch_size] * num_options
                for i in range(num_cls):
                    # Matching scores between each image in the mini-batch and the i-th option text feature
                    choice_h = all_class_feats[:, i] # [batch_size, embed_dim]
                    level_h_list = level_H[:, i] # [batch_size, num_layers, embed_dim]
                    _, scores_I2T = model(img_features, choice_h, level_h_list) # [batch_size, batch_size]
                    all_scores[i] = scores_I2T # [batch_size, batch_size]
                # # Incorrect method (directly using the full matrix)
                # logits = torch.stack(all_scores, dim=1)  # Get [B, C, B]
                # Correct method (take the diagonal)
                logits_per_image = torch.stack([s.diag() for s in all_scores], dim=1) # Only calculate scores for corresponding options [B, C]
        
            # Compute cross-entropy loss
            loss = F.cross_entropy(logits_per_image, labels)
            
            val_loss += loss.item()
            
            # Compute accuracy
            _, predicted = torch.max(logits_per_image, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    # Print metrics
    val_acc = 100. * val_correct / val_total
    print(f"predicted: {all_predictions}")
    print(f"labels: {all_labels}")
    print("="*50)
    print(f"Validation Accuracy: {val_acc:.2f}%")
    return val_acc, val_loss, all_predictions, all_labels
