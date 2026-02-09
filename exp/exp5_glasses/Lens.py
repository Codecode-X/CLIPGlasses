import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from get_model import Clip_model, extract_sentence_features
from utils import setup_logger, set_random_seed
setup_logger(os.path.join(current_dir, "log.txt")) # Redirect output to log.txt file
set_random_seed(3407)  # Set random seed
import torch
import tqdm
import numpy as np
import pandas as pd
import random
import torch.nn as nn
import torch.nn.functional as F
from GlassesDataset import GlassesDataset
from torch.utils.data import DataLoader
import torch.optim as optim

class CLIPGlassesLens(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_dim = 512
        self.res_gate = nn.Sequential(
            nn.Linear(self.embed_dim, 1),
            nn.Sigmoid()
        )
        self.syntax_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim, bias=False),
                nn.GELU(),
                nn.LayerNorm(self.embed_dim)
            ) for _ in range(3)
        ])
        self.semantic_k_proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim, bias=False),
            nn.GELU(),
            nn.LayerNorm(self.embed_dim)
        )
        self.semantic_v_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim, bias=False),
                nn.GELU(),
                nn.LayerNorm(self.embed_dim)
            ) for _ in range(3)
        ])
        attention_scale = torch.tensor([1.0, 0.8, 0.6])
        self.attention_log_scale = nn.Parameter(torch.log(attention_scale))
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, 2*self.embed_dim),
            nn.LayerNorm(2*self.embed_dim),  # Added layer normalization
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2*self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim)
        )
        self._init_weights()
        
    def _init_weights(self):
        nn.init.constant_(self.res_gate[0].bias, -2.0)  # Initial bias towards original features
        nn.init.normal_(self.res_gate[0].weight, mean=0.0, std=0.02)
        for proj in self.syntax_proj:
            nn.init.kaiming_normal_(
                proj[0].weight, 
                mode='fan_in',
                nonlinearity='relu'
            )
        for proj in self.semantic_v_proj:
            nn.init.kaiming_normal_(
                proj[0].weight, 
                mode='fan_in',
                nonlinearity='relu'
            )
         
            
    @staticmethod        
    def load_model(cfg):
        model = CLIPGlassesLens(cfg)
        if 'model_path' in cfg.keys() and cfg['model_path'] is not None:
            print(f"Loading CLIPGlassesLens model weights: {cfg['model_path']}")
            model.load_state_dict(torch.load(cfg['model_path'], weights_only=True))
        model = model.to(cfg['device'])
        model.eval()
        return model
        
    def forward(self, h, level_h_list):
        syntax_feats = []
        for i in range(3):
            layer_feat = level_h_list[:, i, :] 
            proj_feat = self.syntax_proj[i](layer_feat)  # [B, D]
            syntax_feats.append(proj_feat)
        K = self.semantic_k_proj(h)  # [B, D] | K (key)
        Vs = [proj(h) for proj in self.semantic_v_proj]
        c_neg = 0
        scale_factor = torch.sqrt(torch.tensor(self.embed_dim, dtype=torch.float32)) # Attention scaling factor
        for i in range(3):
            Q = syntax_feats[i] # [B, D] | Q1, Q2, Q3 (query)
            A = torch.bmm(Q.unsqueeze(1), K.unsqueeze(-1)).squeeze(-1) / scale_factor  # [B, 1]
            A = F.softmax(A * self.attention_log_scale[i], dim=0)  # Normalize attention weights
            c_neg += A * Vs[i]
        gate = torch.sigmoid(self.res_gate(c_neg)) 
        c_neg = gate * c_neg + (1 - gate) * h  # Add learnable residual gating
        h_neg = self.ffn(c_neg)
        return h_neg

    def calc_losses(self, h_neg, neg_obj, I, h=None):
        h_neg_n = F.normalize(h_neg, p=2, dim=-1)
        neg_obj_n = F.normalize(neg_obj, p=2, dim=-1)
        I_n = F.normalize(I, p=2, dim=-1)
        h_n = F.normalize(h, p=2, dim=-1)
        neg_sim = (h_neg_n * neg_obj_n).sum(dim=-1)  # [B] Higher is better
        neg_sim_loss = 1 - neg_sim.mean()  # Lower is better
        obj2i_sim = (neg_obj_n * I_n).sum(dim=-1)  # [B] Higher is better
        n2i_sim = (h_neg_n * I_n).sum(dim=-1)  # [B] Should be closer to obj2i_sim | Ensures alignment with image features
        n2i_sim_loss = abs(obj2i_sim - n2i_sim).mean()  # Lower is better
        total_loss = neg_sim_loss + n2i_sim_loss  # Lower is better
        return total_loss, {'neg_sim_loss': neg_sim_loss.item(), 'n2i_sim_loss': n2i_sim_loss.item()}

def train(cfg, model:CLIPGlassesLens, device='cuda'):
    if cfg:
        epochs = cfg['epochs']
        batch_size = cfg['batch_size']
        lr = cfg['lr']
        weight_decay = cfg['weight_decay']
        train_size, val_size, test_size = cfg['split']
        num_workers = cfg['num_workers']
        early_stop_patience = cfg['early_stop_patience']
    dataset = GlassesDataset(cfg)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
    for epoch in range(epochs):
        model.train()
        best_sim_loss = float('inf')
        patience_counter = 0 
        total_loss = 0
        losses = {'neg_sim_loss': 0, 'n2i_sim_loss': 0}
        for batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            I = batch['I'].to(device)
            h = batch['h'].to(device)
            level_h_list = batch['level_h_list'].to(device)
            neg_obj = batch['neg_obj'].to(device)
            h_neg = model(h, level_h_list)
            loss, loss_dict = model.calc_losses(h_neg, neg_obj, I, h)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            losses['neg_sim_loss'] += loss_dict['neg_sim_loss']
            losses['n2i_sim_loss'] += loss_dict['n2i_sim_loss']
        scheduler.step()
        batch_count = len(train_loader)
        batch_sim_loss = evaluate(cfg, model, val_loader)
        if batch_sim_loss < best_sim_loss:
            best_sim_loss = batch_sim_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(current_dir, cfg['save_path']))
        else:
            patience_counter += 1
            if early_stop_patience > 0 and patience_counter >= early_stop_patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
    return model

def evaluate(cfg, model, data_loader, device='cuda'):
    model.eval()
    model = model.to(device)
    p_sim_total = 0
    h_sim_total = 0
    neg_sim_total = 0
    
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, desc="Evaluating"):
            h = batch['h'].to(device)
            level_h_list = batch['level_h_list'].to(device)
            l_pos = batch['l_pos'].to(device)
            neg_obj = batch['neg_obj'].to(device)
            h_neg = model(h, level_h_list)
            h_neg_norm = h_neg / h_neg.norm(dim=-1, keepdim=True)
            l_pos_norm = l_pos / l_pos.norm(dim=-1, keepdim=True)
            h_norm = h / h.norm(dim=-1, keepdim=True)
            neg_obj_norm = neg_obj / neg_obj.norm(dim=-1, keepdim=True)  # [b,n,e]
            p_sim = F.cosine_similarity(l_pos_norm, h_neg_norm, dim=-1)
            h_sim = F.cosine_similarity(h_norm, h_neg_norm, dim=-1)
            neg_sim = F.cosine_similarity(neg_obj_norm, h_neg_norm, dim=-1)
            p_sim = torch.mean(p_sim).item()
            h_sim = torch.mean(h_sim).item()
            neg_sim = torch.mean(neg_sim).item()
            p_sim_total += p_sim
            h_sim_total += h_sim
            neg_sim_total += neg_sim
    batch_count = len(data_loader)
    p_sim_total = p_sim_total / batch_count
    h_sim_total = h_sim_total / batch_count
    neg_sim_total = neg_sim_total / batch_count
    sim_loss = (1-neg_sim_total)
    return sim_loss 
