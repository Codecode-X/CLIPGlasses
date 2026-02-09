import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from model import build_model
from utils import load_yaml_config, standard_image_transform
import torch
from model.Clip import tokenize  # Tokenizer
import numpy as np
from torchvision.transforms import Compose, ToTensor
from PIL import Image

def build_clip_model(config_path):
    """
    Build the CLIP model
    """
    cfg = load_yaml_config(config_path, save=None, modify_fn=None) 
    model = build_model(cfg)
    return model

def freeze_clip_model(model):
    """
    Freeze the parameters of the CLIP model
    """
    for param in model.parameters():
        param.requires_grad = False
    return model


config_path = "/root/X/config/CLS/CLS-Clip-VitB32-ep10-Caltech101-AdamW.yaml"

Clip_model = freeze_clip_model(build_clip_model(config_path=config_path)) # Load the CLIP model

def extract_sentence_features(sentence:str):
    """
    Extract CLIP text features for a single sentence
    
    Args:
        - sentence: Input sentence (str)
        
    Returns:
        - text_features (torch.Tensor): Text features from the CLIP text encoder (EOS feature) [embed_dim]
        - level_text_features_list (list<torch.Tensor>): EOS features from each layer of the CLIP text encoder [num_layers, embed_dim]
    """
    with torch.no_grad():  # Disable gradient computation
        tokenized_text = tokenize(sentence) # [num_classes=1, context_length]
        tokenized_text = tokenized_text.to(Clip_model.device) # [num_classes=1, context_length]
        text_features, level_text_features_list = Clip_model.encode_text(tokenized_text) # [num_classes=1, embed_dim]*num_layers
        text_features = text_features.cpu().numpy()[0]
        level_text_features_list = [level_text_features.cpu().numpy()[0] for level_text_features in level_text_features_list]
        return text_features, level_text_features_list # [embed_dim], [embed_dim]*num_layers

def extract_all_sentence_features(sentences:list):
    """
    Extract CLIP text features for multiple sentences in parallel
    
    Args:
        - sentences: List of input sentences (list<str>)
        
    Returns:
        - text_features (torch.Tensor): Text features from the CLIP text encoder (EOS feature) [embed_dim]
    """
    with torch.no_grad():  # Disable gradient computation
        tokenized_texts = [tokenize(sentence) for sentence in sentences]
        tokenized_texts = torch.cat(tokenized_texts, dim=0)
        tokenized_texts = tokenized_texts.to(Clip_model.device)
        text_features, _ = Clip_model.encode_text(tokenized_texts)
        text_features = text_features.cpu().numpy()
        return text_features



def extract_objs_features(objs:list):
    """
    Extract CLIP text features for a list of objects
    
    Args:
        - objs: List of objects (list<str>)
        
    Returns:
        - objs_features (list<torch.Tensor>): Text features from the CLIP text encoder (EOS feature) [embed_dim]*num_objs
    """
    objs_features = []
    for obj in objs:
        text = f"This image shows a {obj}."
        feature, _ = extract_sentence_features(text)
        # feature = torch.tensor(feature, dtype=Clip_model.dtype) # [embed_dim]
        objs_features.append(feature)
    # objs_features = torch.stack(objs_features) # [num_objs, embed_dim]
    return objs_features


def extract_img_features(image_path:str):
    """Extract CLIP image features for a single image"""
    # Load the image and convert it to a tensor [batch_size, 3, input_size, input_size]
    img = Image.open(image_path)  # Open the image
    transform = Compose([standard_image_transform(224, 'BICUBIC'), ToTensor()])  # Define transformations
    img = transform(img)  # Transform the image
    img = img.unsqueeze(0)  # Add batch dimension
    img = img.to(device=Clip_model.device)  # Move to the model's device
    with torch.no_grad():  # Disable gradient computation
        image_features = Clip_model.encode_image(img) # [num_classes, embed_dim]
        return image_features.cpu().numpy()[0] # [embed_dim]
