from .Clip import Clip, tokenize
from utils import SimpleTokenizer as Tokenizer
from torch import nn
import torch.nn.functional as F
import torch
from .build import MODEL_REGISTRY
from .ModelBase import ModelBase

_tokenizer = Tokenizer()  # Initialize tokenizer

@MODEL_REGISTRY.register()
class CoOp(ModelBase):
    """ 
    CoOp Model: Contrastive learning of images and text based on prompt tuning.
    This model uses the CLIP model as the base and adds a Prompt Learner on top to generate optimal prompts for each class.
    
    Main steps:
    1. Encode image features
    2. Use the learnable Prompt Learner to generate text prompts for each class and encode text features
    3. Compute cosine similarity between images and text
    4. Return the class corresponding to the text prompt most similar to the image as the result
    """
    def __init__(self, cfg):
        """
        Initialize the CoOp model
        
        Args:
        cfg: Configuration file containing model hyperparameters and training settings
        """
        super().__init__(cfg)  # Call the parent class Clip constructor
        # Load the pre-trained CLIP model
        pretrained_clip = Clip.build_model(cfg)  # Call parent class build to get the pre-trained CLIP model (load pre-trained weights)
        if cfg.TRAINER.PREC == "fp32" or cfg.TRAINER.PREC == "amp":  # CLIP model defaults to float16 precision
            pretrained_clip.float()  # Convert the model to float32 precision
            
        self.image_encoder = pretrained_clip.visual  # Image encoder
        self.text_encoder = pretrained_clip.transformer  # Text encoder
        self.token_embedding = pretrained_clip.token_embedding  # Token embedding layer
        self.positional_embedding = pretrained_clip.positional_embedding  # Positional embedding layer
        self.ln_final = pretrained_clip.ln_final  # Final LayerNorm layer
        self.text_projection = pretrained_clip.text_projection  # Text projection layer
        self.logit_scale = pretrained_clip.logit_scale  # Inverse of temperature parameter τ
        self.device = pretrained_clip.device  # Device
        self.dtype = pretrained_clip.dtype  # Data type
        self.cfg = cfg  # Configuration file

        # Prompt learning - initialized through the trainer by calling init_promptLearner
        self._pptLearner = None  # Prompt Learner

    def register_promptLearner(self, pptLearner):
        """
        Register the Prompt Learner
        
        Args:
            pptLearner (PromptLearner): Prompt Learner object
        """
        self._pptLearner = pptLearner

    
    def batch_init_prompt_learner(self, batch_choices):
        """
        For MCQ tasks, batch initialize the Prompt Learner
        - Generate the suffix part of the prompt for the current batch of class names and store it in the Prompt Learner
        - Called by the external trainer during each forward() pass, using the current batch of class names to generate the suffix part of the prompt

        Args:
            - batch_choices (list): Batch class name list, shape [batch_size, num_choices]
        
        Returns:
            - None
        """
        # Flatten all options
        all_classes = [cls for choices in batch_choices for cls in choices]  # shape:(B*C=128, ) | 128 = 4(num_choices) * 32(batchsize)
        
        # Batch generate suffixes | suffixs:(batchsize*n_cls, target_length=72, dim); eos_indices:(batchsize*n_cls, )
        suffixs, eos_indices = self._pptLearner.batch_construct_suffix(all_classes, self)
        
        # Reshape to suffixs: [batch_size, n_cls, target_length, dim] and eos_indices: [batch_size, n_cls]
        num_choices = len(batch_choices[0])
        self._pptLearner.suffixs = suffixs.view(len(batch_choices), num_choices, *suffixs.shape[1:])  # [batch_size, num_choices, target_length=72, dim]
        self.eos_indices = eos_indices.view(len(batch_choices), num_choices)  # [batch_size, num_choices]

    def init_prompt_learner(self, cls_list):
        """
        For CLS tasks, initialize the Prompt Learner
        - Generate the suffix part of the prompt for all class names and store it in the buffer
        - Called by the external trainer during initialization, using all class names to generate the suffix part of the prompt
        
        Args:
            - cls_list (list): Class name list, shape [n_cls], e.g., ['pagoda', 'panda', ..., 'stegosaurus', 'stop_sign']
        
        Returns:
            - None

        Outputs:
            - suffixs (tensor): Embedding representation of the suffix | shape: (1, n_cls, target_length=72, dim)
            - eos_indices (tensor): Index of the EOS token in the full prompt | shape: (1, n_cls)
        """
        # Generate suffix | suffixs:(n_cls, target_length=72, dim); eos_indices:(n_cls, )
        suffixs, eos_indices = self._pptLearner.batch_construct_suffix(cls_list, self)
        
        # Add batch dimension
        suffixs = suffixs.unsqueeze(0)  # [1, n_cls, target_length=72, dim]
        eos_indices = eos_indices.unsqueeze(0)  # [1, n_cls]
        
        # Register suffix and EOS indices
        self._pptLearner.register_buffer('suffixs', suffixs)  # [1, n_cls, target_length=72, dim]
        self.register_buffer('eos_indices', eos_indices)  # [1, n_cls]

    @property
    def pptLearner(self):
        """
        Prompt Learner property
        """
        if self._pptLearner is None:
            raise ValueError("CoOp's Prompt Learner pptLearner is None!")
        return self._pptLearner  


    def forward(self, image):
        # --- Image encoding ---
        image_features = self.image_encoder(image.type(self.dtype))  # [B, D]
        
        # --- Text encoding ---
        suffixs = self._pptLearner.suffixs  # [B, C, suffix_len, D]
        B, num_choices = suffixs.shape[:2]
        
        # Expand prefix and ctx to match batch dimensions
        prefixs = self._pptLearner.prefixs.unsqueeze(0).expand(B, -1, -1, -1)  # [B, C, 1, D]
        ctx = self._pptLearner.ctx.unsqueeze(0).unsqueeze(1).expand(B, num_choices, -1, -1)  # [B, C, n_ctx, D]

        # Concatenate prompts
        prompts = torch.cat([prefixs, ctx, suffixs], dim=2)  # [B, C, full_len=77, D] | 1 + n_ctx(4) + suffix_len(72)  = 77
        prompts = prompts.view(B * num_choices, *prompts.shape[2:])  # [B*C, full_len=77, D]

        # Positional encoding
        t = prompts + self.positional_embedding.type(self.dtype)
        
        # Text encoding
        t = t.permute(1, 0, 2)  # [seq_len, B*C, D]
        t = self.text_encoder(t)
        t = t.permute(1, 0, 2)  # [B*C, seq_len, D]
        t = self.ln_final(t).type(self.dtype)
        
        # Extract EOS features
        eos_indices = self.eos_indices.view(-1)  # [B*C]
        batch_indices = torch.arange(t.shape[0], device=self.device)
        EOS = t[batch_indices, eos_indices]  # [B*C, D]
        
        # Projection and normalization
        text_features = EOS @ self.text_projection
        text_features = text_features.view(B, num_choices, -1)  # [B, C, D]
        text_features = F.normalize(text_features, p=2, dim=-1)
        
        # Compute similarity
        image_features = F.normalize(image_features, p=2, dim=-1)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features.unsqueeze(1) @ text_features.permute(0, 2, 1)  # [B, 1, D] @ [B, D, C] -> [B, 1, C]
        
        return logits.squeeze(1)  # [B, 1, C] -> [B, C]
        

class PromptLearner(nn.Module):
    """ 
    Prompt Learner: Learnable context words (ctx) for generating prompts for each class.

    Attributes:
        - init_ctx_text(str): Initial context text
        - n_ctx(int): Number of words in the initial context text
        - Prompt composition (prefixs + ctx + suffixs):
            - ctx: Learnable context vector ctx | torch.tensor | (n_ctx, dim)
            - prefixs: Prompt prefix: [SOS] | torch.tensor | (n_cls, 1, dim)
            - suffixs: Prompt suffix: " " + class name + "." + [EOS] + '...'  | torch.tensor | (n_cls, *, dim)
    """
    def __init__(self, cfg, clip_model, n_cls):
        """
        Initialize the Prompt Learner

        Args:
            - cfg: Configuration file containing model hyperparameters and training settings
            - classnames: List of class names for generating prompts | e.g., ['pagoda', 'panda', ..., 'stegosaurus', 'stop_sign']
            - clip_model: Instantiated CLIP model object
            - n_cls: Number of classes
        
        Configuration:
        - cfg.MODEL.init_ctx: Initial context text, e.g., "a photo of a"
        
        """
        super().__init__()
        # Bind the current instance to the CLIP model
        self.device = clip_model.device  # Device of the CLIP model
        self.dtype = clip_model.dtype  # Data type of the CLIP model
        clip_model.register_promptLearner(self)  # Bind the current instance to the CLIP model
        
        # Initialize parameters
        dtype = clip_model.dtype  # Data type of the CLIP model
        clip_imsize = clip_model.image_encoder.input_resolution  # Input image size of the CLIP model
        self.n_cls = n_cls  # Number of classes

        # Read configuration
        self.init_ctx_text = cfg.MODEL.init_ctx  # Preset context text (str) | e.g., "a photo of a" | Common context text for all classes
        cfg_imsize = cfg.INPUT.SIZE  # Input image size set in the configuration file
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal clip_imsize ({clip_imsize})"  # Ensure input sizes match
        
        # Convert initial context text init_ctx_text (str) to learnable context vector ctx (Torch.tensor)
        init_ctx_text = self.init_ctx_text.replace("_", " ")  # Replace underscores with spaces
        self.n_ctx = len(init_ctx_text.split(" "))  # Number of tokens in the context text | e.g., "a photo of a" -> 4
        init_ctx_token = _tokenizer.encode(init_ctx_text)  # Token list | list<int>
        init_ctx_token_tensor = torch.tensor(init_ctx_token, dtype=torch.long).unsqueeze(0).to(clip_model.device)  # shape: (1, n_ctx)
        with torch.no_grad():
            ctx_vectors = clip_model.token_embedding(init_ctx_token_tensor).type(dtype)  # shape: (1, n_ctx, dim)
        ctx_vectors = ctx_vectors.squeeze(0)  # Remove batch dimension, shape: (n_ctx, dim)
        self.ctx = nn.Parameter(ctx_vectors)  # (Torch.tensor) shape: (n_ctx, dim)

        # Prompt prefix self.prefixs -> <SOS> embedding vector sot_embedding (Torch.tensor)
        sos_token = _tokenizer.encoder["<|startoftext|>"]  # Token (int)
        sos_tensor = torch.tensor([sos_token], dtype=torch.long).to(clip_model.device)  # shape: (1,)
        with torch.no_grad():
            SOS = clip_model.token_embedding(sos_tensor).type(dtype)  # (Torch.tensor) shape: (1, dim)
        prefixs_buffer = SOS.unsqueeze(0).expand(self.n_cls, -1, -1).clone()  # [SOS] | torch.tensor | shape: (n_cls, 1, dim)
        self.register_buffer('prefixs', prefixs_buffer)  # Register prefixs as a buffer, representing fixed, non-trainable tensors to save memory
        
        # Prompt suffix suffixs | To be assigned or registered externally by calling construct_suffix
        # self.suffixs = None  # Class name + [EOS] token | torch.tensor | shape: CLS(n_cls, *, dim) or MCQ((B*n_cls), *, dim)| Directly use register_buffer for registration


    def batch_construct_suffix(self, class_list, clip_model):
        """
        Batch generate suffixes

        Args:
            - class_list (list): List of class names, e.g., ["sleeping_dog", "running_cat", ...]
            - clip_model: Instantiated CLIP model object

        Returns:
            - suffix_embeddings (torch.Tensor): Embedding representation of the suffix | shape: (batchsize*n_cls, target_length, dim) | where target_length = 77 - 1 - n_ctx
            - eos_indices (torch.Tensor): Index of the EOS token in the full prompt | shape: (batchsize*n_cls, )
        
        Main steps:
            1. Compute length constraints for the suffix
            2. Batch process all classes to generate suffix embeddings
            3. Return suffix embeddings and EOS token indices
        """
        # Compute length constraints (consistent with construct_suffix)
        context_length = 77  # Fixed context length in CLIP
        sos_length = 1       # [SOS] token
        target_length = context_length - sos_length - self.n_ctx  # Maximum allowed length for the suffix
        max_cls_length = target_length - 1  # Reserve one position for EOS

        # Batch process all classes
        tokens_list = []
        eos_indices = []
        
        for cls in class_list:
            # Preprocess class name, replace underscores with spaces, and add EOS
            cls_text = cls.replace("_", " ")
            cls_tokens = _tokenizer.encode(cls_text)[:max_cls_length]
            tokens = cls_tokens + [_tokenizer.encoder["<|endoftext|>"]]
            
            # Compute padding length
            pad_length = target_length - len(tokens)
            tokens += [0] * pad_length
            
            # Record EOS position (relative to the entire prompt)
            eos_in_suffix = len(cls_tokens)
            full_eos_pos = 1 + self.n_ctx + eos_in_suffix  # SOT(1) + ctx length + position in suffix
            eos_indices.append(full_eos_pos)
            
            tokens_list.append(tokens)

        # Convert to tensor
        token_tensor = torch.tensor(tokens_list, dtype=torch.long, device=self.device)  # [N, target_length]
        
        # Batch embedding
        with torch.no_grad():
            suffix_embeddings = clip_model.token_embedding(token_tensor).type(self.dtype)  # [N, target_length, D]
        
        eos_indices = torch.tensor(eos_indices, dtype=torch.long, device=self.device)
        
        return suffix_embeddings, eos_indices  # Suffix: torch.Size([128, 72, 512]), EOS indices: torch.Size([128])


    def forward(self):
        # Construct prompt | Class names are placed at the end (the paper shows "end" performs best)
        prompts = torch.cat(
            [
                self.prefixs,  # (n_cls, 1, dim) [SOS] | shape: (n_cls, 1, dim)
                self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1),  # (n_cls, n_ctx, dim) Context | shape: (n_cls, n_ctx, dim)
                self.suffixs,  # (n_cls, *, dim) Includes class name + [EOS] | shape: (n_cls, *, dim)
            ],
            dim=1,
        )
        return prompts  # [SOS] + Learnable context + (" "+class name+".") + [EOS] | Torch.tensor | shape: (n_cls, *, dim)