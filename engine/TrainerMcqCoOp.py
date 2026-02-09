from .base_class.TrainerMcqBase import TrainerMcqBase
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
from tqdm import tqdm


@TRAINER_REGISTRY.register()
class TrainerMcqCoOp(TrainerMcqBase):
    """Trainer class for handling MCQ tasks with CoOp."""

    def init_model(self, cfg):
        """
        (Override parent method) Initialize the model.
        -> Only train the Prompt Generator as an example.

        Args:
            cfg (CfgNode): Configuration.

        Class Attributes:
            - CoOp_model (nn.Module): CoOp model.
            - pptLearner (PromptLearner): Prompt learner.
            - optim (Optimizer): Optimizer.
            - sched (LRScheduler): Learning rate scheduler.

        Returns:
            - model (nn.Module): Model.
            - optim (Optimizer): Optimizer.
            - sched (LRScheduler): Learning rate scheduler.

        Main Steps:
        1. Build the model.
        2. Freeze the text and image encoders of CLIP, only train the PromptLearner of CoOp.
        3. Move the model to the device.
        4. Adjust the model for mixed precision training (if configured).
        5. Deploy the model to multiple GPUs for parallel training (if multiple GPUs are available).
        6. Build the optimizer and scheduler, only optimize the PromptLearner of CLIP, and register them.
        7. Return the model, optimizer, and scheduler.
        """
        # Build the model
        assert cfg.MODEL.NAME == "CoOp", f"TrainerClsCoOp only supports CoOp model, but cfg.MODEL.NAME = {cfg.MODEL.NAME}"
        self.CoOp_model = build_model(cfg)  # Build the model (CLIP model with pre-trained weights)
        print("Number of model parameters:", count_num_param(self.CoOp_model))

        # Freeze certain layers -> Example: Freeze the text and image encoders of CLIP, only train the PromptLearner of CoOp
        if cfg.TRAINER.FROZEN:
            for name, param in self.CoOp_model.named_parameters():
                if "prompt_learner" not in name:
                    param.requires_grad = False

        # Move the model to the device
        self.CoOp_model.to(self.device)

        # Initialize the PromptLearner and register it to CoOp_model for learning prompt information
        self.pptLearner = PromptLearner(cfg, self.CoOp_model, self.num_choices)  # Prompt learner

        # Deploy the model to multiple GPUs for parallel training (if multiple GPUs are available)
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.CoOp_model = nn.DataParallel(self.CoOp_model)

        # Build the PromptLearner and register -> Example: Optimizer only optimizes the PromptLearner of CLIP
        promptLearner = self.CoOp_model.pptLearner
        self.optim = build_optimizer(promptLearner, cfg)
        self.sched = build_lr_scheduler(cfg, self.optim)
        self.register_model("CLIP_promptLearner", promptLearner, self.optim, self.sched)

        # Load pre-trained weights for the PromptLearner (if configured) - Only use pre-trained weights from the same dataset
        if hasattr(cfg.MODEL, "INIT_WEIGHTS_PATH") and cfg.MODEL.INIT_WEIGHTS_PATH: 
            pretarined_path = cfg.MODEL.INIT_WEIGHTS_PATH
            print(f"Loading pre-trained weights: {pretarined_path}")
            self.load_model(directory=pretarined_path)  # Load the best model

        return self.CoOp_model, self.optim, self.sched
    
    def forward_backward(self, batch): 
        """
        (Override parent method) Forward and backward propagation.
        """
        image, num_choices, choices, correct_answer, correct_answer_type = self.parse_batch_train(batch)  # Parse training batch data
        choices = list(zip(*choices))  # Convert (num_choices=4, batchsize=n) to (batch_size=n, num_choices=4)
        
        prec = self.cfg.TRAINER.PREC  # Precision configuration (default: fp16)
        if prec != "fp16": Warning.warn(f"TrainerClsCoOp only supports fp16 precision, but cfg.TRAINER.PREC = {prec}. Using fp16 precision to match CLIP.")
        
        # Batch initialize the PromptLearner
        self.CoOp_model.batch_init_prompt_learner(choices)
        
        # Prepare labels
        labels = torch.tensor(correct_answer, dtype=torch.long, device=self.device)
        
        # Batch forward computation
        logits = self.CoOp_model(image)  # Input the entire batch [B, C, H, W]
        
        # Compute loss
        loss = F.cross_entropy(logits, labels)
        
        # Backward propagation
        self.model_backward_and_update(loss)
        
        # Compute accuracy
        acc = compute_accuracy(logits, labels)[0].item()
        
        # Log results
        loss_summary = {"loss": loss.item(), "acc": acc}
        
        # Update learning rate
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        
        return loss_summary

    @torch.no_grad()
    def test(self, split=None):
        """
        Adapted batch processing test method.
        
        Main improvements:
        1. Batch initialize the PromptLearner.
        2. Support dynamic number of choices.
        3. Handle dimensions for different batch sizes.
        """
        # --------Initialization--------
        self.set_model_mode("eval")
        self.evaluator.reset()
        
        # --------Determine test set--------
        split = split or self.cfg.TEST.SPLIT
        data_loader = self.val_loader if split == "val" else self.test_loader
        print(f"Testing on *{split}* set")

        # --------Batch testing process--------
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            # Parse batch data (adapt to new dimensions)
            image, num_choices, choices, correct_answer, correct_answer_type = self.parse_batch_test(batch)
            
            # Convert choices format to [batch_size, num_choices]
            batch_choices = [list(per_sample) for per_sample in zip(*choices)] if isinstance(choices[0], (list, tuple)) else choices
            
            # Batch initialize the PromptLearner
            self.CoOp_model.batch_init_prompt_learner(batch_choices)
            
            # Model inference [batch_size, num_choices]
            logits = self.CoOp_model(image)  # [batch_size, num_choices]=[100, 4]
            
            # Evaluator processing (support batch statistics)
            self.evaluator.process(
                logits=logits,  # [batch_size]=[100, 4]
                correct_answer=correct_answer,  # [batch_size]=[100]
                correct_answer_type=correct_answer_type
            )

        # --------Result analysis and logging--------
        results = self.evaluator.evaluate()
        for metric, value in results.items():
            self.write_scalar(f"{split}/{metric}", value, self.epoch)
        
        return results["total_accuracy"]

    def load_model(self, directory, epoch=None):
        """
        Override parent method - Utility method: Load model files from a dictionary into the model.
        
        Solve the issue of mismatched num_classes between pre-trained weights and the current model: Trim token_prefix and token_suffix.
        
        Args:
            * directory: Model directory.
            * epoch: Epoch | If None, load the best model.
        
        Returns:
            * epoch: Training epoch.

        Load contents:
            * Model state dictionary.
            * Epoch.
            * Validation results.
        """
        assert directory is not None, "load_model() parameter 'directory' cannot be None."

        names = self.get_model_names()  # Get all model names.

        if epoch is not None:
            print(f"TrainerMcqCoOp.load_model is loading the model for specified epoch {epoch}...")
            model_file = "model.pth.tar-" + str(epoch)  # Load the model for the specified epoch.
        else:
            print("TrainerMcqCoOp.load_model is loading the best model model-best.pth.tar...")
            model_file = "model-best.pth.tar"  # By default, load the best model.

        # Iterate through all model names and load the model.
        for name in names: 
            model_path = osp.join(directory, name, model_file)  # Model path.

            if not osp.exists(model_path):  # If the model path does not exist, raise an exception.
                raise FileNotFoundError(f"Model file not found at {model_path}!")

            checkpoint = load_checkpoint(model_path)  # Load checkpoint.
            state_dict = checkpoint["state_dict"]  # Get state dictionary.
            epoch = checkpoint["epoch"]  # Get epoch.

            try:
                self._models[name].load_state_dict(state_dict)  # Load model state dictionary.
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    print(f"Warning: {e}. Pre-trained model and current model num_classes mismatch. Do not use pre-trained weights from other datasets!")
                else:
                    raise e
        
        return epoch  # Return training epoch.
