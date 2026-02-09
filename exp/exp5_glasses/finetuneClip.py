import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import open_clip
import torch.nn.functional as F
import random
import numpy as np
import pdb

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

csv_path = '/root/NegBench/data/images/Retrieval/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_model = 'ViT-B-32'
pretrained_dataset = 'openai'
batch_size = 160
lr = 5e-6
num_epochs = 20
max_len = 77
save_path = "clip_text_encoder_negated_contrastive.pt"


class NegatedCOCODataset(Dataset):
    def __init__(self, csv_file, preprocess):
        self.df = pd.read_csv(csv_file)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = self.preprocess(Image.open(row["filepath"]).convert("RGB"))
        captions = eval(row["captions"])
        return image, captions


def custom_collate(batch):
    images = [item[0] for item in batch]
    captions = [item[1] for item in batch]
    return images, captions


def train_clip_text_encoder(model, dataset, tokenizer, num_epochs, lr, save_path):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for images, list_of_captions in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            model.train()
            B = len(images)

            # Step 1: Encode B unique images
            images = torch.stack(images).to(device)  # [B, C, H, W]
            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)  # [B, D]

            # Step 2: Flatten all captions and create mapping
            all_texts = []
            text_to_image_indices = []
            for img_idx, captions in enumerate(list_of_captions):
                for cap in captions:
                    all_texts.append(cap)
                    text_to_image_indices.append(img_idx)  # which image this text corresponds to

            text_tokens = tokenizer(all_texts, context_length=max_len).to(device)
            text_features = model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)  # [N_text, D]

            # Step 3: Compute similarity between each text and all images
            logit_scale = model.logit_scale.exp()
            sim_t2i = logit_scale * (text_features @ image_features.T)  # [N_text, B]
            sim_i2t = sim_t2i.T  # [B, N_text]
            
            labels_t2i = torch.tensor(text_to_image_indices, device=device) # [N_text]
            loss_t2i = F.cross_entropy(sim_t2i, labels_t2i)

            labels_i2t = torch.zeros_like(sim_i2t, device=device)
            for img_idx in range(B):
                for text_idx, img_img_idx in enumerate(text_to_image_indices):
                    if img_img_idx == img_idx:
                        labels_i2t[img_idx, text_idx] = 1
            loss_i2t = F.binary_cross_entropy_with_logits(sim_i2t, labels_i2t)
            
            loss = (loss_i2t + loss_t2i) / 2

            # Step 5: Compute recall@5
            topk_t2i = sim_t2i.topk(5, dim=1).indices # [N_text, 5]
            # pdb.set_trace()  # Debugging point
            recall_t2i = (topk_t2i == labels_t2i.unsqueeze(1)).any(dim=1).float().mean() # topk_t2i: [N_text, 5], labels_t2i: [N_text]

            print(f"Recall@5 (i2t): {recall_t2i.item():.4f}")
            # pdb.set_trace()  # Debugging point
            
            # Optimize
            loss = (loss_i2t + loss_t2i) / 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} - Loss: {total_loss / len(dataloader):.4f}")
        evaluate_COCORetrieval(model, dataset, tokenizer, k=5, max_len=max_len)

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


@torch.no_grad()
def evaluate_COCORetrieval(model, dataset, tokenizer, k=5, max_len=77):
    model.eval()
    print("Extracting features for evaluation...")

    image_features_all = []
    text_features_all = []
    text_gt_image_ids = []

    for idx in tqdm(range(len(dataset)), desc="Encoding dataset"):
        image, captions = dataset[idx]
        image_tensor = image.unsqueeze(0).to(device)
        image_feat = model.encode_image(image_tensor)
        image_feat = F.normalize(image_feat, dim=-1)
        image_features_all.append(image_feat.squeeze(0))

        for _ in captions:
            text_gt_image_ids.append(idx)

        tokens = tokenizer(captions, context_length=max_len).to(device)
        text_feat = model.encode_text(tokens)
        text_feat = F.normalize(text_feat, dim=-1)
        text_features_all.append(text_feat)

    image_features_all = torch.stack(image_features_all, dim=0)
    text_features_all = torch.cat(text_features_all, dim=0)

    print("Computing Recall@{}...".format(k))
    logit_scale = model.logit_scale.exp()
    sims = logit_scale * (text_features_all @ image_features_all.T)
    # sims = text_features_all @ image_features_all.T
    topk = sims.topk(k, dim=-1).indices

    correct = sum(gt in topk[i] for i, gt in enumerate(text_gt_image_ids))
    recall = correct / len(text_gt_image_ids)
    print(f"Recall@{k}: {recall * 100:.2f}%")
    return recall


@torch.no_grad()
def evaluate_mcq(model, preprocess, tokenizer, csv_path, device='cuda', max_len=77):
    model.eval()

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} MCQ samples.")

    correct = 0
    total = 0
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating MCQ"):
        image_path = row['image_path']
        captions = [row[f'caption_{j}'] for j in range(4)]
        correct_idx = int(row['correct_answer'])

        image = preprocess(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)
        image_feature = model.encode_image(image)
        image_feature = F.normalize(image_feature, dim=-1)  # [1, D]

        tokens = tokenizer(captions, context_length=max_len).to(device)
        text_features = model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=-1)  # [4, D]

        sims = (image_feature @ text_features.T).squeeze(0)  # [4]
        pred_idx = torch.argmax(sims).item()

        if pred_idx == correct_idx:
            correct += 1
        total += 1

    accuracy = correct / total * 100
    print(f"MCQ Accuracy: {accuracy:.2f}%")
    return accuracy


@torch.no_grad()
def evaluate_ccneg(model, preprocess, tokenizer, csv_path, device='cuda'):
    # Load data
    df = pd.read_csv(csv_path)
    df = df.iloc[-40000:]  # Select last 40,000 rows

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            img_path = row['filepath']
            try:
                image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue

            pos_caption = eval(row['captions'])[0]  # positive caption
            neg_caption = eval(row['n_captions'])[0]  # negative caption
            texts = [pos_caption, neg_caption]

            text_inputs = tokenizer(texts, context_length=max_len).to(device)
            
            image_features = model.encode_image(image)
            text_features = model.encode_text(text_inputs)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            sim = (image_features @ text_features.T).squeeze(0)  # shape: [2]
            pred = sim.argmax().item()  # 0 if positive is more similar

            if pred == 0:
                correct += 1
            total += 1

    acc = correct / total
    print(f"[CCNeg] Accuracy on last 40000 samples: {acc:.4f}")
    return acc



if __name__ == "__main__":
    model, _, preprocess = open_clip.create_model_and_transforms(pretrained_model, pretrained=pretrained_dataset)
    tokenizer = open_clip.get_tokenizer(pretrained_model)
    for p in model.visual.parameters():
        p.requires_grad = False
    model.to(device)

    model.load_state_dict(torch.load(save_path))
    model.to(device)
    evaluate_mcq(model, preprocess, tokenizer, '/root/NegBench/data/images/MCQ/COCO_val_mcq_llama3.1_rephrased.csv', device, max_len=max_len) # 30.61%
    evaluate_ccneg(model, preprocess, tokenizer, '/root/NegBench/data/ccneg_converted.csv', device) # 65.91%
