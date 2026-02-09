from get_model import extract_sentence_features, extract_objs_features, extract_img_features
import torch
import tqdm
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset
import os

class GlassesDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.pos_csv_path = cfg['pos_csv_path'] 
        self.negpos_csv_path = cfg['negpos_csv_path'] 
        self.data = [] 
        self._preprocess_features()
        
    def _preprocess_features(self):
        cache_path = f"LensDataset_cache.pt"
        if os.path.exists(cache_path):
            print(f"Loading preprocessed features from cache: {cache_path}...")
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
        for img_id in tqdm.tqdm(common_ids, desc="Processing data"):
            np_row = np_by_id[img_id]
            p_row = p_by_id[img_id]
            np_captions = eval(np_row['captions'])
            p_captions = eval(p_row['captions'])
            neg_object_list = eval(np_row['negative_objects'])
            neg_objs_list = extract_objs_features(neg_object_list)

            for np_cap, p_cap in zip(np_captions, p_captions):
                h, level_h_list = extract_sentence_features(np_cap)
                h = torch.tensor(h, dtype=self.cfg['dtype'])
                l_pos, _ = extract_sentence_features(p_cap)
                img_path = np_row['filepath']
                I = extract_img_features(image_path=np_row['filepath'])
                I = torch.tensor(I, dtype=self.cfg['dtype'])
                biggest_sim = -float('inf')
                correct_neg_obj, correct_neg_obj_str = None, None
                for i, neg_obj in enumerate(neg_objs_list):
                    neg_obj = torch.tensor(neg_obj, dtype=self.cfg['dtype']) # [b, embed_dim]
                    sim = F.cosine_similarity(h, neg_obj, dim=-1)
                    if sim > biggest_sim:
                        biggest_sim = sim
                        correct_neg_obj = neg_obj
                        correct_neg_obj_str = neg_object_list[i]
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
            'level_h_list': torch.stack([torch.tensor(l, dtype=self.cfg['dtype']) for l in self.data[idx]['level_h_list']]),
            'l_pos': torch.tensor(self.data[idx]['l_pos'], dtype=self.cfg['dtype']),
            'neg_obj': self.data[idx]['neg_obj'].to(dtype=self.cfg['dtype']),
            'img_path': self.data[idx]['img_path'],
            'img_id': self.data[idx]['img_id']
        }
