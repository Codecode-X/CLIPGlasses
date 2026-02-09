import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)
sys.path.insert(0, parent_dir)
from get_model import build_clip_model
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json
import numpy as np
import torch
from tqdm import tqdm
from sklearn.manifold import TSNE
from model.Clip import tokenize

config_path = "/root/X/config/CLS/CLS-Clip-VitB16-ep50-Caltech101-SGD.yaml"
output_dir = "/root/X/output"

Clip_model = build_clip_model(config_path=config_path)


def visualize_features(features, labels, method='tsne', title="Feature Visualization"):
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    else:
        raise ValueError("method must be 'tsne' or 'pca'")

    print(f"Using {method.upper()} for dimensionality reduction...")
    reduced = reducer.fit_transform(features)

    # Visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(
        reduced[labels == 0, 0], reduced[labels == 0, 1],
        c='blue', label='Positive (No Negation)', alpha=0.6
    )
    plt.scatter(
        reduced[labels == 1, 0], reduced[labels == 1, 1],
        c='red', label='Negative (With Negation)', alpha=0.6
    )
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    # plt.grid(True)
    plt.tight_layout()
    plt.show()

def read_negpos_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    neg_pos_pairs = [(item["negative"], item["positive"]) for item in data.get("negation_pairs", [])]
    print(f"Loaded {len(neg_pos_pairs)} negative-positive sentence pairs")
    return neg_pos_pairs


def extract_clip_features(sentences, levels):
    """Extract CLIP text features for sentences"""
    all_level_features = []
    
    for level in levels:
        
        level_features = []
        
        with torch.no_grad():  # Disable gradient calculation
            for sentence in tqdm(sentences):
                tokenized_texts = tokenize(sentence) # [num_classes, context_length]
                tokenized_texts = tokenized_texts.to(Clip_model.device) # [num_classes, context_length]
                text_features, level_text_features = Clip_model.encode_text(tokenized_texts) # [num_classes, embed_dim] # final output layer and all intermediate layer features
                # features.append(text_features.cpu().numpy()[0]) # use final layer
                level_features.append(level_text_features[level].cpu().numpy()[0]) # use intermediate layer
        
        level_features = np.array(level_features) # [num_classes, embed_dim]
        all_level_features.append(level_features) # [num_classes, embed_dim]
    
    return all_level_features

# Main function for visualization experiment
def run_visualization_experiment():
    neg_path = "/root/X/exp/exp1_neg_vanished/neg_pos_pairs_166.json"
    neg_pos_pairs = read_negpos_dataset(neg_path)
    n = len(neg_pos_pairs)
    n = min(n, 20)
    neg_sentences = [pair[0] for pair in neg_pos_pairs[-n:]]
    pos_sentences = [pair[1] for pair in neg_pos_pairs[-n:]]

    all_sentences = neg_sentences + pos_sentences
    print(f"Total number of sentences: {len(all_sentences)}")

    labels = (
        [1] * len(neg_sentences) +  # negative
        [0] * len(pos_sentences)  # positive
    )
    labels = np.array(labels)

    print("Extracting features for all sentences...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Clip_model.to(device)
    
    levels = range(12)
    all_features = extract_clip_features(all_sentences, levels=levels)

    # Visualization (call existing functions)
    # for method in methods:
    #     print(f"Visualizing features using {method.upper()}...")
    #     visualize_multiclass_features(all_features, labels, method=method, title=f"CLIP Feature Visualization - {method.upper()}")
    
    for level in levels:
        # visualize_pca_only(all_features[level], labels, title=f"CLIP Feature Visualization {level} - PCA")
        visualize_tsne_only(all_features[level], labels, title=f"CLIP Feature Visualization {level} - TSNE")
    


from matplotlib.cm import get_cmap
def visualize_pca_tsne_together(features, labels, title="Feature Embedding Comparison"):
    plt.rcParams.update({
        'font.family': 'sans-serif',
        # 'font.sans-serif': ['Arial', 'Helvetica'],
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'legend.fontsize': 10,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.titlesize': 16
    })

    # Check and create output directory
    os.makedirs(output_dir, exist_ok=True)
    n_samples = features.shape[0]
    assert n_samples % 2 == 0, "Number of samples should be twice the number of sentence pairs"
    n_pairs = n_samples // 2

    # Color map
    cmap = get_cmap('tab20')

    # Subplots: 1x2
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    methods = [('PCA', PCA(n_components=2, random_state=42)),
               ('T-SNE', TSNE(n_components=2, random_state=42,
                              perplexity=min(30, (n_samples-1)//3)))]

    for ax, (name, reducer) in zip(axes, methods):
        # Dimensionality reduction
        reduced = reducer.fit_transform(features)

        for i in range(n_pairs):
            color = cmap(i % 20)
            pos_idx = i
            neg_idx = i + n_pairs
            x1, y1 = reduced[pos_idx]
            x2, y2 = reduced[neg_idx]

            # Positive: o  Negative: s
            ax.scatter(x1, y1, c=[color], marker='o', edgecolors='k', s=120, alpha=0.9)
            ax.scatter(x2, y2, c=[color], marker='s', edgecolors='k', s=120, alpha=0.9)
            # Dashed line connection
            ax.plot([x1, x2], [y1, y2], linestyle='--', color=color, alpha=0.6, linewidth=1)

        ax.set_title(name)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        # ax.grid(True, linestyle=':', alpha=0.5)

    fig.suptitle(title)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = os.path.join(output_dir, f"{title}.png")
    plt.savefig(output_path, dpi=600, transparent=True)
    plt.close(fig)
    print(f"Saved high-res transparent visualization to: {output_path}")



def visualize_pca_only(features, labels, title="PCA Feature Visualization",
                       fixed_xlim=(-5, 5), fixed_ylim=(-5, 5)):

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.titlesize': 16
    })

    os.makedirs(output_dir, exist_ok=True)
    n_samples = features.shape[0]
    assert n_samples % 2 == 0, "Number of samples should be twice the number of sentence pairs"
    n_pairs = n_samples // 2

    # PCA dimensionality reduction
    reducer = PCA(n_components=2, random_state=42)
    reduced = reducer.fit_transform(features)

    cmap = get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(8, 6))

    for i in range(n_pairs):
        color = cmap(i % 20)
        pos_idx = i
        neg_idx = i + n_pairs
        x1, y1 = reduced[pos_idx]
        x2, y2 = reduced[neg_idx]

        ax.scatter(x1, y1, c=[color], marker='o', edgecolors='k', s=60, alpha=0.9)
        ax.scatter(x2, y2, c=[color], marker='s', edgecolors='k', s=60, alpha=0.9)
        ax.plot([x1, x2], [y1, y2], linestyle='--', color=color, alpha=0.6, linewidth=1)

    ax.set_title("PCA")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    if fixed_xlim is not None:
        ax.set_xlim(fixed_xlim)
    if fixed_ylim is not None:
        ax.set_ylim(fixed_ylim)

    ax.grid(False)
    ax.legend().set_visible(False)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{title}.png")
    plt.savefig(output_path, dpi=600, transparent=True)
    plt.close(fig)
    print(f"Saved PCA visualization to: {output_path}")


def visualize_tsne_only(features, labels, title="TSNE Feature Visualization",
                       fixed_xlim=(-30, 30), fixed_ylim=(-30, 30), outlier_percentile=98):

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.titlesize': 16
    })

    os.makedirs(output_dir, exist_ok=True)
    n_samples = features.shape[0]
    assert n_samples % 2 == 0, "Number of samples should be twice the number of sentence pairs"
    n_pairs = n_samples // 2

    reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, (n_samples-1)//3))
    reduced = reducer.fit_transform(features)
    
    pair_distances = []
    for i in range(n_pairs):
        pos_idx = i
        neg_idx = i + n_pairs
        x1, y1 = reduced[pos_idx]
        x2, y2 = reduced[neg_idx]
        dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        pair_distances.append(dist)

    pair_distances = np.array(pair_distances)
    threshold = np.percentile(pair_distances, outlier_percentile)
    print(f"Removing outlier pairs with distance above the {outlier_percentile}th percentile, threshold: {threshold:.2f}")

    valid_indices = [i for i, d in enumerate(pair_distances) if d <= threshold]

    cmap = get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(8, 6))

    for idx, i in enumerate(valid_indices):
        color = cmap(idx % 20)
        pos_idx = i
        neg_idx = i + n_pairs
        x1, y1 = reduced[pos_idx]
        x2, y2 = reduced[neg_idx]

        ax.scatter(x1, y1, c=[color], marker='o', edgecolors='k', s=180, alpha=0.9)
        ax.scatter(x2, y2, c=[color], marker='s', edgecolors='k', s=180, alpha=0.9)
        ax.plot([x1, x2], [y1, y2], linestyle='--', color=color, alpha=0.6, linewidth=1)

    ax.set_title("TSNE")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    # # Fix axis range (if not specified, auto-calculate)
    # if fixed_xlim is not None:
    #     ax.set_xlim(fixed_xlim)
    # if fixed_ylim is not None:
    #     ax.set_ylim(fixed_ylim)

    # No legend, no grid
    ax.grid(False)
    ax.legend().set_visible(False)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{title}.png")
    plt.savefig(output_path, dpi=600, transparent=True)
    plt.close(fig)
    print(f"Saved TSNE visualization to: {output_path}")

if __name__ == "__main__":
   run_visualization_experiment() 
           