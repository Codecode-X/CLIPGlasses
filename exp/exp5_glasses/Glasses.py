import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)
sys.path.insert(0, parent_dir)
from utils import setup_logger, set_random_seed
set_random_seed(3407)  # Set random seed
from Lens import CLIPGlassesLens
from Frame import CLIPGlassesFrame
from NegDetector import NegationDetector
from McqDataset import McqDataset, evaluate_model_mcq
from RetrievalDataset_gtneg import RetrievalNegGtDataset, evaluate_model_retrieval_withGTNeg
from RetrievalDataset import RetrievalDataset, evaluate_model_retrieval, retrieval_collate_fn
from CCNegDataset_gtneg import CCNegGtDataset, evaluate_model_CCNeg_etrieval_withGTNeg
from CLSDataset import CLSDataset, evaluate_model_CLS
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.optim as optim
from torch.nn import functional as F


class Glasses(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg['device']
        self.lens = CLIPGlassesLens.load_model(cfg['Lens'])
        self.frame = CLIPGlassesFrame.load_model(cfg['Frame'])
        self.negDetector = NegationDetector.load_model(cfg['NegationDetector'])
        self.neg_thr = cfg['NegationDetector']['neg_thr']
        self.dtype = cfg['dtype']
        for param in self.negDetector.parameters():
            param.requires_grad = False

    def forward(self, I, h, level_h_list, l_neg=None, chunk_size=-1):
        with torch.no_grad():
            neg_mask = self.negDetector(h).squeeze(-1) > self.neg_thr
        if l_neg is None:
            h_neg = self.lens(h, level_h_list)
        else:
            h_neg = l_neg
        assert I.size(0) == h_neg.size(0) == h.size(0)
        scores_T2I = self.frame(I, h, h_neg, neg_mask=neg_mask, chunk_size=chunk_size)
        scores_I2T = scores_T2I.T
        return scores_T2I, scores_I2T

    def calc_losses(self, scores_T2I, scores_I2T, caption_to_img):
        caption_to_img = torch.tensor(caption_to_img, device=self.device, dtype=torch.long)
        loss_txt2img = F.cross_entropy(scores_T2I, caption_to_img)
        exp_sim = scores_T2I.exp()
        all_exp = exp_sim.sum(dim=0)
        mask = torch.zeros_like(exp_sim)
        mask[torch.arange(exp_sim.size(0), device=self.device), caption_to_img] = 1
        pos_exp = (exp_sim * mask).sum(dim=0)
        loss_img2txt = - (pos_exp / all_exp).log().mean()
        contrastive_loss = 0.5 * (loss_txt2img + loss_img2txt)
        total_loss = contrastive_loss
        return total_loss, {'contrastive_loss': contrastive_loss.item()}

    def calc_ccneg_losses(self, scores_T2Ip, scores_Ip2T):
        batch_size = scores_Ip2T.size(0)
        device = scores_Ip2T.device
        labels_I2T = torch.arange(batch_size, device=device)
        labels_T2I = torch.cat([
            torch.arange(batch_size, device=device),
            -torch.ones(batch_size, device=device)
        ])
        loss_I2T = F.cross_entropy(scores_Ip2T, labels_I2T)
        valid_mask = (labels_T2I != -1)
        valid_scores = scores_T2Ip[valid_mask]
        valid_labels = labels_T2I[valid_mask].long()
        if valid_labels.numel() > 0:
            loss_T2I = F.cross_entropy(valid_scores, valid_labels)
        else:
            loss_T2I = torch.tensor(0.0, device=device)
        total_loss = (loss_I2T + loss_T2I) / 2
        return total_loss, {
            'loss_I2T': loss_I2T.item(),
            'loss_T2I': loss_T2I.item()
        }

    def calc_ccneg_4_losses(self, scores_Tpn2Ip, scores_Ip2Tpn, scores_In2Tpn, scores_Tpn2In):
        batch_size = scores_Ip2Tpn.size(0)
        device = scores_Ip2Tpn.device
        labels_Ip = torch.arange(batch_size, device=device)
        loss_Ip2Tpn = F.cross_entropy(scores_Ip2Tpn, labels_Ip)
        hp_scores_Tpn2Ip = scores_Tpn2Ip[:batch_size]
        loss_Tpn2Ip = F.cross_entropy(hp_scores_Tpn2Ip, labels_Ip)
        hn_scores_In2Tpn = scores_In2Tpn[:, batch_size:]
        hp_scores_In2Tpn = scores_In2Tpn[:, :batch_size]
        scores_In2Tnp = torch.cat([hn_scores_In2Tpn, hp_scores_In2Tpn], dim=1)
        loss_In2Tnp = F.cross_entropy(scores_In2Tnp, labels_Ip)
        loss_Tpn2In = torch.tensor(0.0, device=device)
        total_loss = (loss_Ip2Tpn + loss_Tpn2Ip + loss_In2Tnp + loss_Tpn2In) / 4
        return total_loss, {
            'loss_Ip2Tpn': loss_Ip2Tpn.item(),
            'loss_Tpn2Ip': loss_Tpn2Ip.item(),
            'loss_In2Tnp': loss_In2Tnp.item(),
            'loss_Tpn2In': loss_Tpn2In.item(),
        }

    @staticmethod
    def load_model(cfg):
        model = Glasses(cfg)
        model.negDetector.load_state_dict(torch.load(cfg['NegationDetector']['model_path'], weights_only=True))
        if 'pretrain' in cfg.keys() and cfg['pretrain'] and cfg['model_path'] is not None:
            full_ckpt = torch.load(os.path.join(current_dir, cfg['model_path']), map_location='cpu', weights_only=False)
            filtered_ckpt = {k: v for k, v in full_ckpt.items() if not k.startswith("negDetector.")}
            model.load_state_dict(filtered_ckpt, strict=False)
        if 'test' in cfg.keys() and cfg['test'] is True and cfg['model_path'] is not None:
            full_ckpt = torch.load(os.path.join(current_dir, cfg['model_path']), map_location='cpu', weights_only=False)
            filtered_ckpt = {k: v for k, v in full_ckpt.items() if not k.startswith("negDetector.")}
            model.load_state_dict(filtered_ckpt, strict=False)
        return model


def train_COCORetr_with_gtneg(cfg, model:Glasses, with_gt_neg=True):   
    device = cfg['device']
    epochs = cfg['epochs']
    clip_grad = True if cfg.get('clip_grad', False) else False
    batch_size = cfg['batch_size']
    early_stop_patience = cfg['early_stop_patience']
    lr = cfg['lr']
    num_workers = cfg['num_workers']
    train_rate, val_rate, test_rate = cfg['RetrievalWithGtNeg']['split']
    dataset = RetrievalNegGtDataset(cfg['RetrievalWithGtNeg'])
    train_size = int(len(dataset) * train_rate)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    if cfg.get('only_train_moudle', None) == 'lens':
        for param in model.frame.parameters():
            param.requires_grad = False
        optimizer = optim.AdamW(model.lens.parameters(), lr=lr, betas=(0.9, 0.98))
    elif cfg.get('only_train_moudle', None) == 'frame':
        for param in model.lens.parameters():
            param.requires_grad = False
        optimizer = optim.AdamW(model.frame.parameters(), lr=lr, betas=(0.9, 0.98))
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(
                lambda grad, name=name: print(f"Gradient norm for {name}: {grad.norm().item():.4f}")
                if grad.norm() > 5e2 else None
            )
    evaluate_model_retrieval_withGTNeg(model, val_loader, test_raw_clip=False, with_gt_neg=with_gt_neg)
    best_recall5 = 0
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        losses = {'contrastive_loss': 0}
        for batch in tqdm(train_loader, desc=f"Epoch{epoch+1}/{epochs}"):
            h = batch['h'].to(device)
            level_h = batch['level_h_list'].to(device)
            l_pos = batch['l_pos'].to(device)
            l_neg = batch['neg_obj'].to(device)
            I = batch['I'].to(device)
            image_ids = batch['img_id'].to(device)
            unique_img_ids, remapped_ids = torch.unique(image_ids, sorted=True, return_inverse=True)
            caption_to_img = remapped_ids.cpu().numpy()
            if with_gt_neg is True:
                scores_T2I, scores_I2T = model(I, h, level_h, l_neg)
            else:
                scores_T2I, scores_I2T = model(I, h, level_h)
            cti = torch.tensor(caption_to_img, dtype=torch.long, device=device)
            unique_vals = torch.unique(cti, sorted=True)
            first_idx = []
            for val in unique_vals:
                idx = (cti == val).nonzero(as_tuple=True)[0][0]
                first_idx.append(idx)
            first_idx = torch.stack(first_idx, dim=0)
            scores_T2I = scores_T2I[:, first_idx]
            scores_I2T = scores_T2I.t()
            loss, loss_dict = model.calc_losses(scores_T2I, scores_I2T, caption_to_img)
            epoch_loss += loss.item()
            losses['contrastive_loss'] += loss_dict['contrastive_loss']
            optimizer.zero_grad()
            loss.backward()
            if clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        batch_count = len(train_loader)
        print(f"Ep{epoch+1}/{epochs}  Loss: {epoch_loss/batch_count:.4f} contrastive_loss: {losses['contrastive_loss']/batch_count:.4f}")
        scheduler.step()
        val_recall5 = evaluate_model_retrieval_withGTNeg(model, val_loader, test_raw_clip=False, with_gt_neg=with_gt_neg)['mean'][5]
        if val_recall5 > best_recall5:
            best_recall5 = val_recall5
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(current_dir, cfg['save_path']))
            print(f"Best model saved at epoch {epoch} with recall@5: {best_recall5}")
        else:
            patience_counter += 1
            if early_stop_patience > 0 and patience_counter >= early_stop_patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
        if epoch % 5 == 0 or epoch == epochs - 1:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': epoch_loss,
                'recall5': val_recall5
            }
            torch.save(checkpoint, os.path.join(current_dir, f"checkpoint_epoch_{epoch}.pth"))
        print(f"Training completed. Best validation recall5: {best_recall5:.4f}")
    return model


def train_CCNeg_with_gtneg(cfg, model:Glasses, with_gt_neg=True):   
    device = cfg['device']
    epochs = cfg['epochs']
    clip_grad = True if cfg.get('clip_grad', False) else False
    batch_size = cfg['batch_size']
    early_stop_patience = cfg['early_stop_patience']
    lr = cfg['lr']
    num_workers = cfg['num_workers']
    dataset = CCNegGtDataset(cfg['CCNegGtDataset'])
    train_size, val_size = len(dataset)-40000, 40000
    train_dataset = Subset(dataset, list(range(train_size)))
    val_dataset = Subset(dataset, list(range(train_size, train_size+5000)))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    if cfg.get('only_train_moudle', None) == 'lens':
        for param in model.frame.parameters():
            param.requires_grad = False
        optimizer = optim.AdamW(model.lens.parameters(), lr=lr, betas=(0.9, 0.98))
    elif cfg.get('only_train_moudle', None) == 'frame':
        for param in model.lens.parameters():
            param.requires_grad = False
        optimizer = optim.AdamW(model.frame.parameters(), lr=lr, betas=(0.9, 0.98))
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(
                lambda grad, name=name: print(f"Gradient norm for {name}: {grad.norm().item():.4f}")
                if grad.norm() > 5e2 else None
            )
    evaluate_model_CCNeg_etrieval_withGTNeg(model, val_loader, test_raw_clip=False, with_gt_neg=with_gt_neg)
    best_acc = 0
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        losses = {'loss_Ip2Tpn': 0, 'loss_Tpn2Ip': 0, 'loss_In2Tnp': 0, 'loss_Tpn2In': 0}
        for batch in tqdm(train_loader, desc=f"Epoch{epoch+1}/{epochs}"):
            Ip = batch['Ip'].to(device)
            In = batch['In'].to(device)
            hp = batch['hp'].to(device)
            hn = batch['hn'].to(device)
            level_hp_list = batch['level_hp_list'].to(device)
            level_hn_list = batch['level_hn_list'].to(device)
            neg_obj = batch['neg_obj'].to(device)
            img_id = batch['img_id'].to(device)
            batch_size = Ip.size(0)
            if with_gt_neg is True:
                _, scores_Ip2Tp = model(Ip, hp, level_hp_list, neg_obj)
                _, scores_Ip2Tn = model(Ip, hn, level_hn_list, neg_obj)
                _, scores_In2Tp = model(In, hp, level_hp_list, neg_obj)
                _, scores_In2Tn = model(In, hn, level_hn_list, neg_obj)
            else:
                _, scores_Ip2Tp = model(Ip, hp, level_hp_list)
                _, scores_Ip2Tn = model(Ip, hn, level_hn_list)
                _, scores_In2Tp = model(In, hp, level_hp_list)
                _, scores_In2Tn = model(In, hn, level_hn_list)
            scores_Ip2T = torch.cat([scores_Ip2Tp, scores_Ip2Tn], dim=1)
            scores_T2Ip = scores_Ip2T.t()
            scores_In2T = torch.cat([scores_In2Tp, scores_In2Tn], dim=1)
            scores_T2In = scores_In2T.t()
            loss, loss_dict = model.calc_ccneg_4_losses(scores_T2Ip, scores_Ip2T, scores_In2T, scores_T2In)
            epoch_loss += loss.item()
            losses['loss_Ip2Tpn'] += loss_dict['loss_Ip2Tpn']
            losses['loss_Tpn2Ip'] += loss_dict['loss_Tpn2Ip']
            losses['loss_In2Tnp'] += loss_dict['loss_In2Tnp']
            losses['loss_Tpn2In'] += loss_dict['loss_Tpn2In']
            optimizer.zero_grad()
            loss.backward()
            if clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        batch_count = len(train_loader)
        print(f"Ep{epoch+1}/{epochs}  Loss: {epoch_loss/batch_count:.4f} \
              loss_Ip2Tpn: {losses['loss_Ip2Tpn']/batch_count:.4f} \
              loss_Tpn2Ip: {losses['loss_Tpn2Ip']/batch_count:.4f} \
              loss_In2Tnp: {losses['loss_In2Tnp']/batch_count:.4f} \
              loss_Tpn2In: {losses['loss_Tpn2In']/batch_count:.4f}")
        scheduler.step()
        val_results = evaluate_model_CCNeg_etrieval_withGTNeg(model, val_loader, test_raw_clip=False, with_gt_neg=with_gt_neg)
        val_acc = val_results['accuracy']
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(current_dir, cfg['save_path']))
            print(f"Best model saved at epoch {epoch} with ACC: {best_acc}")
        else:
            patience_counter += 1
            print(f"💔ACC drop from {best_acc:.4f} to {val_acc:.4f}, cur patience_counter add to {patience_counter}")
            if early_stop_patience > 0 and patience_counter >= early_stop_patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
        if epoch % 5 == 0 or epoch == epochs - 1:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': epoch_loss,
                'val_acc': val_acc
            }
            torch.save(checkpoint, os.path.join(current_dir, f"checkpoint_epoch_{epoch}.pth"))
    print(f"Training completed. Best validation ACC: {best_acc:.4f}")
    return model


if __name__ == "__main__":
    cfg = {
        'epochs': 30,
        'batch_size': 64,
        'lr': 1e-5,
        'num_workers': 4,
        'early_stop_patience': 5,
        'device': 'cuda',
        'dtype': torch.float32,
        'save_path': 'best_clip_Glasses.pth',
        'pretrain': False,
        'Lens': {
            'device': 'cuda',
            'dtype': torch.float32,
            'num_heads': 4,
            'dropout': 0.1,
            'model_path': '/root/X/exp/exp5_glasses/weights/v2_COCO/best_clip_lens_9922.pth'
        },
        'Frame': {
            'device': 'cuda',
            'dtype': torch.float32,
            'lambda_0': 1,
        },
        'NegationDetector': {
            'device': 'cuda',
            'model_path': '/root/X/exp/exp5_glasses/weights/best_NegDet_9404_9212.pth',
            'neg_thr': 0.5, 
        },
        'Mcq': {
            'batch_size': 64,
            'num_workers': 4,
            'num_options': 4,
            'split': [0.9, 0.1, 0.0],
            'train_dataset_path': '/root/NegBench/data/images/MCQ/COCO_val_mcq_llama3.1_rephrased.csv',
            'test_dataset_path': '/root/NegBench/data/images/MCQ/VOC2007_mcq_llama3.1_rephrased.csv',
        },
        'Retrieval': {
            'batch_size': 64,
            'num_workers': 4,
            'split': [0.9, 0.1, 0.0],
            'train_dataset_path': '/root/NegBench/data/images/Retrieval/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv',
            'test_dataset_path': '/root/NegBench/data/images/Retrieval/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv',
        },
        'RetrievalWithGtNeg': { 
            'batch_size': 64,
            'num_workers': 4, 
            'split': [0.9, 0.1, 0.0], 
            'pos_csv_path': "/root/NegBench/data/images/Retrieval/COCO_val_retrieval.csv",
            'negpos_csv_path': "/root/NegBench/data/images/Retrieval/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv",
            'dtype': torch.float32, 
        },
        'CCNegGtDataset': {
            'batch_size': 64,
            'num_workers': 4,
            'csv_path': '/root/NegBench/data/ccneg_converted.csv',
            'negative_image_ft_mapping_path': '/root/NegBench/data/distractor_image_mapping.pt', # Hard negative sample image indices
            'dtype': torch.float32, 
            
        },
        'ClsEvalDataset': { 
            'csv_path': '/root/NegBench/data/CLS_Imagenet/imagenet_train.csv',
            'batch_size': 64,
            'num_workers': 4,
        }
    }

    cfg['Lens']['model_path'] = 'weights/v2_COCO/best_clip_lens_9922.pth'
    cfg['lr'] = 1e-4 # 1e-3
    cfg['neg_thr'] = -1
    cfg['epochs'] = 10
    model = Glasses.load_model(cfg)
    model = train_COCORetr_with_gtneg(cfg, model, with_gt_neg=True)
    cfg['pretrain'] = True
    cfg['lr'] = 1e-3
    cfg['model_path'] = 'best_clip_Glasses.pth'
    cfg['neg_thr'] = -1
    cfg['clip_grad'] = True # Gradient clipping
    model = Glasses.load_model(cfg)
    model.lens = CLIPGlassesLens.load_model(cfg['Lens']) 
    model = train_COCORetr_with_gtneg(cfg, model, with_gt_neg=False) 
    cfg['test_raw_clip'] = False
    cfg['test'] = True
    cfg['model_path'] = 'weights/v2_COCO/best_clip_Glasses.pth'
    cfg['Lens']['model_path'], cfg['Frame']['model_path'] = None, None 
    cfg['NegationDetector']['model_path'] = '/root/X/exp/exp5_glasses/weights/best_NegDet_9404_9212.pth'
    test_ccneg_dataset = CCNegGtDataset(cfg['CCNegGtDataset'])
    train_size, val_size = len(test_ccneg_dataset)-40000, 40000
    val_dataset = Subset(test_ccneg_dataset, list(range(train_size, train_size+val_size))) # Validation set [-40000, -1)
    test_ccneg_dataloader = torch.utils.data.DataLoader(test_ccneg_dataset, batch_size=cfg['CCNegGtDataset']['batch_size'], shuffle=False, num_workers=cfg['CCNegGtDataset']['num_workers'])
    if cfg['test_raw_clip'] is True:
        evaluate_model_CCNeg_etrieval_withGTNeg(None, val_dataset, test_raw_clip=True, with_gt_neg=False)
    else:
        model = Glasses.load_model(cfg)
        evaluate_model_CCNeg_etrieval_withGTNeg(model, val_dataset, test_raw_clip=False, with_gt_neg=False)
    test_retrieval_dataset = RetrievalNegGtDataset(cfg['RetrievalWithGtNeg'])
    test_retrieval_dataloader = torch.utils.data.DataLoader(test_retrieval_dataset, batch_size=cfg['Retrieval']['batch_size'], shuffle=False, num_workers=cfg['Retrieval']['num_workers'])
    if cfg['test_raw_clip'] is True:
        evaluate_model_retrieval_withGTNeg(None, test_retrieval_dataloader, test_raw_clip=True, with_gt_neg=False)
    else:
        model = Glasses.load_model(cfg)
        evaluate_model_retrieval_withGTNeg(model, test_retrieval_dataloader, test_raw_clip=False, with_gt_neg=False) 
    cfg['Mcq']['test_dataset_path'] = '/root/NegBench/data/images/MCQ/VOC2007_mcq_llama3.1_rephrased.csv'
    test_retrieval_dataset = McqDataset(cfg['Mcq']['test_dataset_path'])
    test_retrieval_dataloader = torch.utils.data.DataLoader(test_retrieval_dataset, batch_size=cfg['Mcq']['batch_size'], shuffle=False, num_workers=cfg['Mcq']['num_workers'])
    if cfg['test_raw_clip'] is True:
        evaluate_model_mcq(None, test_retrieval_dataloader, test_raw_clip=True)
    else:
        model = Glasses.load_model(cfg)
        evaluate_model_mcq(model, test_retrieval_dataloader, test_raw_clip=False)
    cfg['Mcq']['test_dataset_path'] = '/root/NegBench/data/images/MCQ/COCO_val_mcq_llama3.1_rephrased.csv'
    test_retrieval_dataset = McqDataset(cfg['Mcq']['test_dataset_path'])
    test_retrieval_dataloader = torch.utils.data.DataLoader(test_retrieval_dataset, batch_size=cfg['Mcq']['batch_size'], shuffle=False, num_workers=cfg['Mcq']['num_workers'])
    if cfg['test_raw_clip'] is True:
        evaluate_model_mcq(None, test_retrieval_dataloader, test_raw_clip=True)
    else:
        model = Glasses.load_model(cfg)
        evaluate_model_mcq(model, test_retrieval_dataloader, test_raw_clip=False)
    cfg['ClsEvalDataset']['csv_path'] = '/root/NegBench/data/CLS/imagenet_val.csv'
    test_dataset = CLSDataset(cfg['ClsEvalDataset'])
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg['ClsEvalDataset']['batch_size'], shuffle=False, num_workers=cfg['ClsEvalDataset']['num_workers'])
    if cfg['test_raw_clip'] is True:
        evaluate_model_CLS(None, test_dataloader, test_raw_clip=True)
    else:
        model = Glasses.load_model(cfg)
        evaluate_model_CLS(model, test_dataloader, test_raw_clip=False)
    cfg['ClsEvalDataset']['csv_path'] = '/root/NegBench/data/CLS/caltech101.csv'
    test_dataset = CLSDataset(cfg['ClsEvalDataset'])
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg['ClsEvalDataset']['batch_size'], shuffle=False, num_workers=cfg['ClsEvalDataset']['num_workers'])
    if cfg['test_raw_clip'] is True:
        evaluate_model_CLS(None, test_dataloader, test_raw_clip=True)
    else:
        model = Glasses.load_model(cfg)
        evaluate_model_CLS(model, test_dataloader, test_raw_clip=False)
    cfg['ClsEvalDataset']['csv_path'] = '/root/NegBench/data/CLS/cifar100.csv'
    test_dataset = CLSDataset(cfg['ClsEvalDataset'])
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg['ClsEvalDataset']['batch_size'], shuffle=False, num_workers=cfg['ClsEvalDataset']['num_workers'])
    if cfg['test_raw_clip'] is True:
        evaluate_model_CLS(None, test_dataloader, test_raw_clip=True)
    else:
        model = Glasses.load_model(cfg)
        evaluate_model_CLS(model, test_dataloader, test_raw_clip=False)
    cfg['ClsEvalDataset']['csv_path'] = '/root/NegBench/data/CLS/cifar10.csv'
    test_dataset = CLSDataset(cfg['ClsEvalDataset'])
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg['ClsEvalDataset']['batch_size'], shuffle=False, num_workers=cfg['ClsEvalDataset']['num_workers'])
    if cfg['test_raw_clip'] is True:
        evaluate_model_CLS(None, test_dataloader, test_raw_clip=True)
    else:
        model = Glasses.load_model(cfg)
        evaluate_model_CLS(model, test_dataloader, test_raw_clip=False)
    for k, v in cfg.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for k1, v1 in v.items():
                print(f"  {k1}: {v1}")
        else:
            print(f"{k}: {v}")