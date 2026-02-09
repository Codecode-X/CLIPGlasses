from .build import MODEL_REGISTRY
from .ModelBase import ModelBase
from collections import OrderedDict
from typing import Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import utils
import os
from packaging import version
import warnings
from typing import Union, List
from utils import SimpleTokenizer as Tokenizer

_tokenizer = Tokenizer()  # Initialize tokenizer

# Available pretrained models
_MODELS_URL = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}

@MODEL_REGISTRY.register()
class Clip(ModelBase):

    def __init__(self,
                 cfg,  # Configuration
                 embed_dim: int,
                 # Vision part
                 image_resolution: int,  # Input image resolution (e.g., 224)
                 vision_layers: Union[Tuple[int, int, int, int], int],  # Number of layers in the vision transformer
                 vision_width: int,  # Hidden layer width of the vision transformer
                 vision_patch_size: int,  # Patch size (ViT model)
                 # Text part
                 context_length: int,  # Maximum text length (sequence length)
                 vocab_size: int,  # Vocabulary size (input dimension of the embedding layer)
                 transformer_width: int,  # Hidden layer width of the transformer
                 transformer_heads: int,  # Number of attention heads in the transformer
                 transformer_layers: int,  # Number of layers in the transformer
                 ):

        """
        Initialization function for the CLIP model.

        Parameters:
            - cfg (dict): Configuration
            
            - embed_dim (int): Embedding dimension | Output feature dimension of the CLIP model
            - image_resolution (int): Input image resolution | 224
            - vision_layers (Union[Tuple[int, int, int, int], int]): Number of layers in the vision transformer | ResNet: tuple(3, 4, 6, 3) | ViT: int(12)
            - vision_width (int): Hidden layer width of the vision transformer | ResNet: 64 | ViT: 768
            - vision_patch_size (int): Patch size (ViT model) | 32
            
            - context_length (int): Maximum text length (sequence length) | 77
            - vocab_size (int): Vocabulary size (input dimension of the embedding layer) | 49408
            - transformer_width (int): Hidden layer width of the transformer | 512
            - transformer_heads (int): Number of attention heads in the transformer | 8
            - transformer_layers (int): Number of layers in the transformer | 12

        Main steps:
            1. Initialize the parent class
            2. Set the device
            3. Create the vision transformer (ResNet or ViT)
            4. Create the text transformer
            5. Create the token embedding layer and positional encoding
            6. Create the LayerNorm layer
            7. Create the projection layer (text_projection)
            8. Create the trainable softmax temperature parameter τ
            9. Initialize weights
        """
        super().__init__(cfg)
        self.device = 'cuda' if cfg.USE_CUDA else 'cpu'

        self.output_logits = None  # Record model output results
        self.output_featuer = None  # Record model output features

        self.context_length = context_length  # Text sequence length

        # Create the vision transformer
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(  # Based on ResNet
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64  # Calculate the number of transformer heads, usually set to width/64
            self.visual = VisionTransformer(  # Based on ViT
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )
        
        # Build the text transformer
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=build_attention_mask(self.context_length)
        )

        self.vocab_size = vocab_size  # Vocabulary size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)  # Token embedding layer
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))  # Positional encoding
        self.ln_final = LayerNorm(transformer_width)  # LayerNorm normalization layer

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))  # Project head to project text features to embed_dim dimensions

        # Trainable logit_scale = log(1 / τ) | CLIP uses a trainable logit_scale to let the model automatically adjust τ for different data and tasks.
        τ = 0.07  # Softmax temperature parameter τ during CLIP training. A smaller τ (larger logit_scale) makes the distribution steeper, highlighting samples with high similarity.
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / τ))  # During training, logit_scale is learned instead of τ to avoid gradient issues.

        self.initialize_parameters()  # Initialize weights

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype
    
    # Weight initialization
    def initialize_parameters(self):
        """
        Weight initialization 
        
        Main steps:
            1. Initialize the weights of token_embedding and positional_embedding
            2. Initialize the weights of the vision encoder (ResNet) | ViT version does not require initialization
            3. Initialize the weights of the text encoder (Transformer)
            4. Initialize the weights of text_projection

        Principle:
            - If the variance of W_Q, W_K, W_V initialization is too large, the Softmax in attention calculation may approach 0 or 1, leading to gradient vanishing.
            - Use std = d^{-0.5} to ensure reasonable mean and variance of QKV during initialization, avoiding extreme Softmax outputs that cause gradient issues.
            - d is the hidden layer width of the Transformer.
        """
        # Initialize the weights of token_embedding and positional_embedding with a normal distribution of mean 0 and std 0.02
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        # Initialize the weights of the vision encoder (ResNet) - Xavier initialization - for Softmax activation function | ViT version does not require initialization
        if isinstance(self.visual, ModifiedResNet):
            # Initialize the weights of the attention pooling layer in ResNet (similar to the CLS token method in ViT, aggregating global features)
            if self.visual.attnpool is not None: 
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)
            # Initialize the BatchNorm in ResNet residual blocks to 0
            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):  # Only initialize the last BatchNorm
                        nn.init.zeros_(param)

        # Initialize the weights of the text encoder (Transformer)
        # Calculate three types of standard deviations
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)  # ((2L)^{-0.5}) prevents residual block accumulation from causing value explosion.
        attn_std = self.transformer.width ** -0.5  # Xavier initialization - for Softmax activation function 
        fc_std = (2 * self.transformer.width) ** -0.5  # Kaiming He initialization - for ReLU activation function
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        # Initialize the text projection layer - Xavier initialization
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
    

    def encode_image(self, image):
        """ 
        Vision encoder to extract image features.

        Parameters:
            - image (Tensor): Image data | [batch_size, 3, input_size, input_size]

        Returns:
            - image_features (Tensor): Image features | [batch_size, embed_dim]
        """
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        """
        Text encoder to extract text features.

        Parameters:
            - text (Tensor): Text data | [num_classes, context_length (default 77 for CLIP)]

        Returns:
            - text_features (Tensor): Text features | [num_classes, embed_dim]
        
        Main steps:
            1. Convert input text to token embeddings
            2. Add trainable positional encoding
            3. Encode text using the Transformer
            4. Normalize data using the LayerNorm layer
            5. Use the EOT (End-of-Text) token's feature as the representation of the entire text sequence (similar to Bert's [cls] token)
            6. Perform a linear transformation using `text_projection` to obtain the final text features
        """
        # Convert input text to token embeddings
        x = self.token_embedding(text).type(self.dtype)  # [num_classes, context_length, transformer_width]
        # Add trainable positional encoding to retain sequence position information
        x = x + self.positional_embedding.type(self.dtype)
        
        # Encode text using the Transformer
        x = x.permute(1, 0, 2)  # Adjust dimensions to [context_length, num_classes, transformer_width] to fit the Transformer
        x, level_x_list = self.transformer(x)
        x = x.permute(1, 0, 2)  # Restore dimensions to [num_classes, context_length, transformer_width]
        level_x_list = [x.permute(1, 0, 2) for x in level_x_list]  # Restore dimensions for each layer's output
        
        # Normalize data using the LayerNorm layer
        x = self.ln_final(x).type(self.dtype)
        level_x_list = [self.ln_final(x).type(self.dtype) for x in level_x_list]  # Normalize each layer's output

        # Use the EOT (End-of-Text) token's feature as the representation of the entire text sequence (similar to Bert's [cls] token)
        EOT = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]  
        level_EOT_list = [x[torch.arange(x.shape[0]), text.argmax(dim=-1)] for x in level_x_list]  # Process each layer's output
        
        # Perform a linear transformation using `text_projection` to obtain the final text features
        text_features  = EOT @ self.text_projection  
        level_text_features_list = [EOT @ self.text_projection for EOT in level_EOT_list]  # Perform linear transformation for each layer's output

        return text_features, level_text_features_list  # [num_classes, embed_dim], [num_classes, transformer_width] * transformer_layers

    def init_text_features(self, label_texts:list):
        """
        Pre-extract text features for each class and save them to self.text_features.
        
        Parameters:
            - label_texts (list): List of class texts | [num_classes] | ['pagoda', 'panda', ..., 'stegosaurus', 'stop_sign']
        """
        print("Extracting text features for each class and saving them to self.text_features...")
        with torch.no_grad():  # Disable gradient computation
            # Tokenize text labels and convert to token embeddings
            tokenized_texts = tokenize(label_texts, self.context_length)  # [num_classes, context_length]
            # Move to device
            tokenized_texts = tokenized_texts.to(self.device)  # [num_classes, context_length]
            # Extract text features
            self.text_features = self.encode_text(tokenized_texts)  # [num_classes, embed_dim]
        print("Text feature extraction completed!")

    def forward(self, image, return_feature=False):
        """
        CLIP forward propagation.
        
        Parameters:
            - image(torch.Tensor): Input image, shape (batch_size, 3, height, width)
            - return_feature (bool): Whether to return features
        
        Returns:
            - output (dict): Output results | Contains 'logits_per_image' and 'logits_per_text' keys, corresponding to image and text similarity
            - feature (dict): Features | Contains 'image' and 'text' keys, corresponding to image and text features
        
        Main steps:
            1. Extract text and image features
            2. Normalize feature vectors
            3. Compute cosine similarity between image and text
        """
        
        # Extract image features
        image_features = self.encode_image(image)

        # Load preloaded text features
        text_features = self.text_features

        # Normalize feature vectors: compute the norm along the feature dimension and divide the feature vector by the norm
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # Compute cosine similarity between image and text
        logit_scale = self.logit_scale.exp()  # Inverse of the temperature parameter τ
        logits_per_image = logit_scale * image_features @ text_features.t()  # image->text similarity | [batch, num_classes]
        
        # Return results
        if return_feature: 
            return logits_per_image, image_features
        else:
            return logits_per_image


    @staticmethod
    def build_model(cfg: dict, jit=False):
        """
        Build the CLIP model from the state dictionary (state_dict / JIT) of the pretrained model.
        
        Parameters:
            - cfg (dict): Configuration

        Returns:
            - model (CLIP): The constructed CLIP model

        Main steps:
            1. Download the pretrained model parameters
            2. Load the pretrained model parameters and instantiate the CLIP model
                1. If loaded as a JIT model, return the JIT model directly
                2. If loaded as a regular model, construct the CLIP model based on the model parameters
            3. Return the CLIP model in eval mode
            
        """
        # ---Download the pretrained model parameters---
        # Read the pretrained model name and the download/save path for pretrained weights from the configuration
        pretrained_name = cfg.MODEL.pretrained  # e.g., 'RN50', 'ViT-B/32', etc.
        device = 'cuda' if cfg.USE_CUDA else 'cpu'
        
        download_root = cfg.MODEL.download_root \
            if hasattr(cfg.MODEL, 'download_root') else os.path.expanduser("~/.cache/clip")  # Path to save pretrained weights
        
        if pretrained_name in _MODELS_URL:
            model_path = utils.download_weight(_MODELS_URL[pretrained_name], download_root)
        elif os.path.isfile(pretrained_name):
            model_path = pretrained_name
        else:
            raise RuntimeError(f"Model {pretrained_name} not found; available models = {_MODELS_URL.keys()}")
        
        # ---Load the pretrained model parameters and instantiate the CLIP model---
        with open(model_path, 'rb') as opened_file:
            try: # First, try loading as a JIT model
                jit_model = torch.jit.load(opened_file, map_location="cpu").eval() # eval mode
                state_dict = None
            except RuntimeError:
                warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                # If loading fails, try loading as a regular model
                state_dict = torch.load(opened_file, map_location="cpu")

        if jit: 
            # ---Load as a JIT model---
            jit_model = utils.patch_jit_model(jit_model, device=device)  # Fix device and dtype information for the JIT model
            return jit_model 
        else:
            # ---Load as a regular model---
            state_dict = state_dict or jit_model.state_dict()  # If no state_dict, use the JIT model's state_dict
            vit = "visual.proj" in state_dict
            if vit: # ViT
                vision_width = state_dict["visual.conv1.weight"].shape[0]
                vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
                vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
                grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
                image_resolution = vision_patch_size * grid_size
            else: # ResNet
                counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
                vision_layers = tuple(counts)
                vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
                output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
                vision_patch_size = None
                assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
                image_resolution = output_width * 32

            embed_dim = state_dict["text_projection"].shape[1]
            context_length = state_dict["positional_embedding"].shape[0]
            vocab_size = state_dict["token_embedding.weight"].shape[0]
            transformer_width = state_dict["ln_final.weight"].shape[0]
            transformer_heads = transformer_width // 64
            transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

            # ---Instantiate the CLIP model---
            model = Clip(
                cfg,
                embed_dim,
                image_resolution, vision_layers, vision_width, vision_patch_size,
                context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
            )

            # ---Load the pretrained model parameters into the CLIP model---
            for key in ["input_resolution", "context_length", "vocab_size"]:
                if key in state_dict:
                    del state_dict[key]
            convert_weights(model) # CLIP defaults to using fp16 precision
            model.load_state_dict(state_dict)

            # Convert the model's precision based on the cfg requirements
            if cfg.TRAINER.PREC == 'fp32':
                model.float()
            elif cfg.TRAINER.PREC == 'fp16':
                pass
            else:
                raise NotImplementedError(f"CLIP does not support this precision conversion method: {cfg.TRAINER.PREC}")

            # Return the CLIP model in eval mode
            return model.eval()


# ------Below are the implementations of CLIP model submodules or helper functions------

def build_attention_mask(context_length):
    """
    Build a causal attention mask 
        - Upper triangular matrix
        - Only allows the Transformer to see the current token and its preceding tokens, preventing future information leakage.
    """
    mask = torch.empty(context_length, context_length)
    mask.fill_(float("-inf"))
    mask.triu_(1)  # zero out the lower diagonal
    return mask

def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Tokenize the input string or list of strings and return their token representations.

    Parameters:
    ----------
    texts : Union[str, List[str]]
        The text(s) to be tokenized, can be a single string or a list of strings.

    context_length : int
        The context length to be used (default is 77 for all CLIP models).

    Returns:
    -------
    torch.LongTensor
        A 2D tensor containing the tokenized results of the text, with shape (num_texts, context_length).
    """
    # If the input is a single string, convert it to a list to ensure uniform processing
    if isinstance(texts, str):
        texts = [texts]
    # Get special tokens: <|startoftext|> (SOT, start of text marker) and <|endoftext|> (EOT, end of text marker)
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    # Encode each text and add start and end markers | Text "Hello world" -> [sot_token, tokenized("Hello"), tokenized("world"), eot_token]
    all_tokens = []
    for text in texts:
        text_token = _tokenizer.encode(text) # Token list of the text
        token = [sot_token] + text_token + [eot_token]  # Concatenate lists
        all_tokens.append(token)
    
    # Create a tensor of shape (number of texts, context_length) initialized with 0 (for padding)
    if version.parse(torch.__version__) < version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long) # shape: (num_texts, context_length)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    # Iterate through all token sequences and pad them into the `result` tensor
    for i, tokens in enumerate(all_tokens):
        # If the number of tokens exceeds context_length, truncate or raise an error
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:          
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        # Fill tokens into result[i], with excess parts automatically padded with 0
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result # shape: (num_texts, context_length)

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # All conv layers have stride 1. An avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # Downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1.
    - The final pooling layer is a QKV attention instead of an average pool.
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    """QuickGELU: Approximates GELU using x * sigmoid(1.702 * x) for faster computation, suitable for large-scale Transformer tasks."""
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """Residual Attention Block."""
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, attention_maps=None):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        
        if attention_maps is None:  # Do not visualize attention heatmaps
            output = self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
            return output
        else:  # Visualize attention heatmaps
            output, attn_weights = self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)
        
        # ---Used for visualizing attention heatmaps---
        if attention_maps is not None:
            attention_maps.append(attn_weights)
        # ---------------------------------------------

        return output
    
    def forward(self, x: torch.Tensor, attention_maps=None):
        x = x + self.attention(self.ln_1(x), attention_maps)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """Residual Attention Network."""
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width  # Hidden layer dimension of the Transformer - determines the dimensions of query/key/value vectors.
        self.layers = layers  # Number of Transformer layers - number of stacked ResidualAttentionBlock layers.
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor, attention_maps=None):
        if attention_maps is None:
            level_list = []
            for block in self.resblocks:
                x = block(x)
                level_list.append(x.clone())
            return x, level_list
        else:
            level_list = []
            for block in self.resblocks:
                x = block(x, attention_maps)
                level_list.append(x.clone())
            return x, level_list


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        """
        Vision Transformer (ViT) structure, suitable for image classification tasks.

        Parameters:
        - input_resolution (int): Input image resolution (e.g., 224 for 224x224).
        - patch_size (int): Size of each patch (e.g., 16 for 16x16).
        - width (int): Hidden layer dimension of the Transformer (i.e., embedding dimension).
        - layers (int): Number of Transformer layers (number of encoder blocks).
        - heads (int): Number of heads in the multi-head attention mechanism.
        - output_dim (int): Final output feature dimension.
        """
        super().__init__()
        # Save input resolution and output dimension
        self.input_resolution = input_resolution  # Input image resolution, e.g., 224×224
        self.output_dim = output_dim  
        self.patch_size = patch_size  # Size of each patch, e.g., 16×16
        
        # **Convolutional Layer: Convert input image to patch embedding**
        # Here, a 2D convolutional layer is used, similar to ViT, to split the image into patches and perform embedding.
        self.conv1 = nn.Conv2d(
            in_channels=3,      # Number of input channels, RGB image has 3 channels.
            out_channels=width, # Number of output channels, i.e., embedding dimension, i.e., hidden layer dimension of the Transformer.
            kernel_size=patch_size, # Patch size, e.g., 16×16.
            stride=patch_size,   # Stride equal to patch size, effectively splitting into non-overlapping patches.
            bias=False          # No bias used.
        )

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, attention_maps=None):
        """
        Forward pass.

        Parameters:
        - x (Tensor): Input image tensor, shape (batch_size, 3, input_resolution, input_resolution).
        - attention_maps (List): Used to record attention heatmaps for visualization.

        Returns:
        - x (Tensor): Processed feature vector, shape (batch_size, output_dim).
        """
        # **1. Patch Embedding: Convert input image to patch-level tokens**
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        
        # **2. Reshape**
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        
        # **3. Concatenate x and class token**
        # Use broadcasting to expand to batchsize class tokens.
        batch_class_tokens = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], 
                                                                            dtype=x.dtype, device=x.device)
        x = torch.cat([batch_class_tokens, x], dim=1)  # Shape becomes (batch_size, patch_num + 1, width).
        
        # **4. Add positional encoding**
        x = x + self.positional_embedding.to(x.dtype)
        
        x = self.ln_pre(x)  # LayerNorm

        # **5. Pass through Transformer encoder**
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, width) -> (seq_len, batch_size, width)
        x, _ = self.transformer(x, attention_maps)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, width) -> (batch_size, seq_len, width)

        # **6. Extract CLS token as the feature representation of the entire sequence**
        x = self.ln_post(x[:, 0, :])  # LayerNorm

        # **7. Perform projection (if enabled)**
        if self.proj is not None:
            x = x @ self.proj  # Learnable projection layer.

        return x  # Return final features.
