from .base_class.TrainerClsBase import TrainerBase
from model import build_model
from utils import count_num_param
from torch.cuda.amp import GradScaler
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from utils.metrics import compute_accuracy
from optimizer import build_optimizer
from lr_scheduler import build_lr_scheduler
from .build import TRAINER_REGISTRY

@TRAINER_REGISTRY.register()
class TrainerClip(TrainerBase):

    def check_cfg(self, cfg): # Check if the PREC field in the configuration file is valid
        """ (Override parent method) Check if the PREC field in the configuration file is valid. """
        assert cfg.TRAINER.PREC in ["fp16", "fp32", "amp"]

    def init_model(self, cfg):
        """
        (Override parent method) Initialize the model.
        -> Only train the image encoder as an example.

        Args:
            cfg (CfgNode): Configuration.

        Returns:
            model (nn.Module): Model.
            optim (Optimizer): Optimizer.
            sched (LRScheduler): Learning rate scheduler.

        Main steps:
        1. Build the model.
        2. Freeze the text encoder of the model, only train the image encoder.
        3. Move the model to the device.
        4. Adjust the model for mixed precision training.
        5. Deploy the model to multiple GPUs if applicable.
        6. Build the optimizer and scheduler, only optimize the image encoder, and register them.
        7. Return the model, optimizer, and scheduler.
        """
        # Build the model
        assert cfg.MODEL.NAME == "Clip", f"TrainerClip only supports Clip model, but cfg.MODEL.NAME = {cfg.MODEL.NAME}"
        self.clip_model = build_model(cfg) # Build the model (CLIP model provides pre-trained model loading here)
        print("Number of model parameters:", count_num_param(self.clip_model))

        # Freeze certain layers of the model -> Example: Freeze the text encoder of CLIP, only train the image encoder
        if cfg.TRAINER.FROZEN:
            for name, param in self.clip_model.named_parameters():
                if "visual" not in name:
                    param.requires_grad = False

        # Move the model to the device
        self.clip_model.to(self.device)

        # Set the text labels for the model, allowing it to pre-extract text features for each class
        sorted_labels = sorted(self.lab2cname.items(), key=lambda x: x[0]) # Sort text labels by label in ascending order for alignment with model predictions
        label_texts = [item[1] for item in sorted_labels]  # Text label tensor | [num_classes]
        print("Dataset text labels sorted in ascending order:", label_texts)
        self.clip_model.init_text_features(label_texts)

        # Adjust the model for mixed precision training to reduce memory usage (if configured for mixed precision training)
        self.scaler = GradScaler() if cfg.TRAINER.PREC == "amp" else None

        # Deploy the model to multiple GPUs if applicable (if multiple GPUs are available)
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.clip_model = nn.DataParallel(self.clip_model)

        # Build the optimizer and scheduler and register them -> Example: Only optimize the image encoder of CLIP
        image_encoder = self.clip_model.visual
        self.optim = build_optimizer(image_encoder, cfg)
        self.sched = build_lr_scheduler(cfg, self.optim)
        self.register_model("CLIP_image_encoder", image_encoder, self.optim, self.sched)

        return self.clip_model, self.optim, self.sched
    
    def forward_backward(self, batch): 
        """
        (Override parent method) Forward and backward propagation.
        """

        image, label = self.parse_batch_train(batch)  # Parse training batch data to get images and labels
        assert image is not None and label is not None, "In forward_backward(), parsed images and labels cannot be None"

        prec = self.cfg.TRAINER.PREC  # Configured precision
        if prec == "amp":  # Automatic mixed precision training
            with autocast():
                # CLIP requires both images and text (text features for each class are already loaded during model initialization).
                # Image: [batch, 3, 224, 224]
                output = self.clip_model(image) # Model prediction -> output: [batch, num_classes]
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:  # Default fp16
            output = self.clip_model(image) # Model prediction
            loss = F.cross_entropy(output, label)  # Compute loss  
            self.model_backward_and_update(loss)  # Backward propagation

        # Loss logs to be recorded
        loss_summary = {  
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        # Automatically update learning rate at the end of the phase
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary