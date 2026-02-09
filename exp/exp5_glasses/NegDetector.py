import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)
sys.path.insert(0, parent_dir)
from get_model import extract_all_sentence_features
from tqdm import tqdm


class NegationDetector(nn.Module):
    def __init__(self, embed_dim=512, hidden_dim=256):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, h):
        return torch.sigmoid(self.classifier(h))
    
    @staticmethod        
    def load_model(cfg):
        model = NegationDetector()
        if 'model_path' in cfg.keys() and cfg['model_path'] is not None:
            print(f"Loading NegationDetector model weights: {cfg['model_path']}")
            model.load_state_dict(torch.load(cfg['model_path'], weights_only=True))
        model = model.to(cfg['device'])
        model.eval()
        return model

class NegationDataset(Dataset):
    def __init__(self, csv_path, batch_size=64, device='cuda'):
        
        cache_path = os.path.basename(csv_path).split('.')[0] + "_features_cache.pt"
        
        if os.path.exists(cache_path): 
            print(f"Loading features from cache: {cache_path}")
            cached_data = torch.load(cache_path, weights_only=False)
            self.features = cached_data['features']
            self.labels = cached_data['labels']
        else:
            try:
                df = pd.read_csv(csv_path, encoding='utf-8', engine='python', on_bad_lines=lambda line: print(f"Skipping line: {line}"))
            except UnicodeDecodeError:
                df = pd.read_csv(csv_path, encoding='gbk', engine='python', on_bad_lines=lambda line: print(f"Skipping line: {line}"))
            self.texts = []
            self.labels = []
            
            positive_texts = df['positive'].dropna().tolist()
            self.texts.extend(positive_texts)
            self.labels.extend([0] * len(positive_texts))  # 0: positive sample

            negative_texts = df['negative'].dropna().tolist()
            self.texts.extend(negative_texts)
            self.labels.extend([1] * len(negative_texts))  # 1: negative sample
            
            print(f"Extracting features and creating cache: {cache_path}...")
            
            self.features = []
            with torch.no_grad():
                for i in tqdm(range(0, len(self.texts), batch_size), desc="Processing data"):
                    batch_texts = self.texts[i:i + batch_size]
                    # Extract features for the current batch
                    batch_features = extract_all_sentence_features(batch_texts)
                    self.features.append(torch.from_numpy(batch_features))

            self.features = torch.cat(self.features, dim=0)

            print(f"Saving features to cache: {cache_path}")
            torch.save({'features': self.features, 'labels': torch.tensor(self.labels, dtype=torch.long)}, cache_path)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx].float(), self.labels[idx].clone().detach().float()

def predict_negation(detector, texts):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    detector.eval()
    
    with torch.no_grad():
        features = extract_all_sentence_features(texts)
        features = torch.from_numpy(features).to(device, dtype=torch.float32)
        outputs = detector(features).squeeze()
        logits = outputs.float().cpu().numpy()
        preds = (outputs > 0.5).float().cpu().numpy()
        
    
    return logits, preds

if __name__ == "__main__":
    detector = NegationDetector.load_model({
        'model_path': '/root/X/exp/exp5_glasses/weights/best_NegDet_9404_9212.pth', 
        'device': 'cuda'})
    texts = [
        "a photo of a dog", 
        "a photo of a chair",
        "a photo of a human",
        "a photo of a woman",
        "there is a dog",
        '???',
        "there is no dog",
        "a photo of a none",
        "a photo of a nonesense object",
        "a photo of a no object",
        "a photo of a no object",
        "None",
        "a woman without glasses",
    ]
    logits, preds = predict_negation(detector, texts)
    import numpy as np
    np.set_printoptions(precision=4, suppress=True)
    print(logits)
    print(preds)
