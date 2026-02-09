from .base_class.TrainerClsBase import TrainerClsClip
from model import build_model
from utils import count_num_param, load_checkpoint
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.metrics import compute_accuracy
from optimizer import build_optimizer
from lr_scheduler import build_lr_scheduler
from .build import TRAINER_REGISTRY
import os.path as osp
from model.CoOp import PromptLearner


@TRAINER_REGISTRY.register()
class TrainerClsCoOp(TrainerClsClip):
    """Trainer class for handling classification tasks with CoOp."""
    def init_model(self, cfg):
        """
        (Override parent method) Initialize the model.
        -> Train only the Prompt Generator example.

        Args:
            cfg (CfgNode): Configuration.

        Returns:
            CoOp_model (nn.Module): CoOp model.
            optim (Optimizer): Optimizer.
            sched (LRScheduler): Learning rate scheduler.

        Class Attributes:
            - CoOp_model (nn.Module): CoOp model.
            - pptLearner (PromptLearner): Prompt learner.
            - optim (Optimizer): Optimizer.
            - sched (LRScheduler): Learning rate scheduler.

        Main Steps:
        1. Build the model.
        2. Freeze the text encoder and image encoder of CLIP, train only the PromptLearner of CoOp.
        3. Move the model to the device.
        4. Adjust the model for mixed precision training (if configured).
        5. Deploy the model to multiple GPUs in case of multi-GPU training (if multiple GPUs are available).
        6. Build the optimizer and scheduler, optimize only the PromptLearner of CLIP, and register them.
        7. Return the model, optimizer, and scheduler.
        """
        # Build the model
        assert cfg.MODEL.NAME == "CoOp", f"TrainerClsCoOp only supports CoOp model, but cfg.MODEL.NAME = {cfg.MODEL.NAME}"
        self.CoOp_model = build_model(cfg)  # Build the model (CLIP model provides pre-trained model loading here)
        print("Number of model parameters:", count_num_param(self.CoOp_model))

        # Freeze certain layers of the model -> Example: Freeze the text encoder and image encoder of CLIP, train only the PromptLearner of CoOp
        if cfg.TRAINER.FROZEN:
            for name, param in self.CoOp_model.named_parameters():
                if "prompt_learner" not in name:
                    param.requires_grad = False

        # Move the model to the device
        self.CoOp_model.to(self.device)

        # Set the model's text labels and initialize the prompt learner
        sorted_labels = sorted(self.dm.lab2cname.items(), key=lambda x: x[0])  # Sort text labels by label in ascending order for alignment with model predictions
        label_texts = [item[1] for item in sorted_labels]  # Text label tensor | [num_classes]
        print("Dataset text labels sorted in ascending order:", label_texts)
        self.pptLearner = PromptLearner(cfg, self.CoOp_model, n_cls=len(label_texts))  # Initialize a PromptLearner object and register it to CoOp_model for learning prompt information
        self.CoOp_model.init_prompt_learner(cls_list=label_texts)  # Initialize the prompt learner

        # Deploy the model to multiple GPUs in case of multi-GPU training (if multiple GPUs are available)
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.CoOp_model = nn.DataParallel(self.CoOp_model)

        # Build PromptLearner and register -> Example: Optimizer optimizes only the PromptLearner of CLIP
        self.optim = build_optimizer(self.pptLearner, cfg)
        self.sched = build_lr_scheduler(cfg, self.optim)
        self.register_model("CLIP_promptLearner", self.pptLearner, self.optim, self.sched)

        # Load pre-trained weights for the prompt learner (if configured) - Use pre-trained weights only for the same dataset, otherwise it is meaningless
        if hasattr(cfg.MODEL, "INIT_WEIGHTS_PATH") and cfg.MODEL.INIT_WEIGHTS_PATH: 
            pretarined_path = cfg.MODEL.INIT_WEIGHTS_PATH
            print(f"Loading pre-trained weights: {pretarined_path}")
            self.load_model(directory=pretarined_path)  # Load the best model

        return self.CoOp_model, self.optim, self.sched
    
    def forward_backward(self, batch): 
        """
        (Override parent method) Forward and backward propagation.
        """
        image, label = self.parse_batch_train(batch)  # Parse training batch data to get images and labels
        assert image is not None and label is not None, "Images and labels parsed by parse_batch_train in forward_backward() cannot be None"

        prec = self.cfg.TRAINER.PREC  # Configured precision # Default fp16
        if prec != "fp16": Warning.warn(f"TrainerClsCoOp only supports fp16 precision, but cfg.TRAINER.PREC = {prec}, using fp16 precision to match Clip")

        output = self.CoOp_model(image)  # Model prediction
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


    def load_model(self, directory, epoch=None):
        """
        Override parent method - Utility method: Load model files from the dictionary into the model.
        
        Solve the problem of mismatch between num_classes in the pre-trained model and the current model: Trim token_prefix and token_suffix.
        
        Args:
            * directory: Model directory
            * epoch: Epoch | If None, load the best model
        
        Returns:
            * epoch: Training epoch

        Load contents:
            * Model state dictionary
            * Epoch
            * Validation results
        """
        assert directory is not None, "The directory parameter of load_model() cannot be None"

        names = self.get_model_names()  # Get all model names

        model_file = "model-best.pth.tar"  # By default, load the best model
        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)  # If epoch is specified, load the model for the specified epoch

        # Iterate through all model names and load the model
        for name in names: 
            model_path = osp.join(directory, name, model_file)  # Model path

            if not osp.exists(model_path):  # If the model path does not exist, raise an exception
                raise FileNotFoundError(f"Model file not found at {model_path}!")

            checkpoint = load_checkpoint(model_path)  # Load checkpoint
            state_dict = checkpoint["state_dict"]  # Get state dictionary
            epoch = checkpoint["epoch"]  # Get epoch

            try:
                self._models[name].load_state_dict(state_dict)  # Load model state dictionary
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    print(f"Warning: {e}. Mismatch between num_classes in the pre-trained model and the current model, attempting to trim token_prefix and token_suffix")
                    # Trim token_prefix and token_suffix
                    state_dict["token_prefix"] = state_dict["token_prefix"][:self._models[name].token_prefix.shape[0]]  # Trim to [num_classes, 1, 512]
                    state_dict["token_suffix"] = state_dict["token_suffix"][:self._models[name].token_prefix.shape[0]]  # Trim to [num_classes, 72, 512]
                    self._models[name].load_state_dict(state_dict)  # Load model state dictionary
                else:
                    raise e
        
        return epoch  # Return training epoch
