import os.path as osp
from collections import OrderedDict
import torch
from torch.utils.tensorboard import SummaryWriter
from utils import (tolist_if_not, load_checkpoint, save_checkpoint, resume_from_checkpoint)
import time
import numpy as np
import datetime
import torch
import torch.nn as nn
from tqdm import tqdm
from data_manager import DataManager
from torch.cuda.amp import GradScaler
from optimizer import build_optimizer
from lr_scheduler import build_lr_scheduler
from model import build_model
from utils import (count_num_param, mkdir_if_missing, load_pretrained_weights)
from evaluator import build_evaluator
from utils import (MetricMeter, AverageMeter)
from ..build import TRAINER_REGISTRY

@TRAINER_REGISTRY.register()
class TrainerBase:
    """
    Base class for iterative trainers.
    
    Contains methods:

    -------Utility Methods-------
    * init_writer: Initialize TensorBoard.
    * close_writer: Close TensorBoard.
    * write_scalar: Write scalar values to TensorBoard.

    * register_model: Register models, optimizers, and learning rate schedulers.
    * get_model_names: Get all registered model names.

    * save_model: Save the model, including model state, epoch, optimizer state, scheduler state, and validation results.
    * load_model: Load the model, including model state, epoch, and validation results.
    * resume_model_if_exist: Resume the model if a checkpoint exists, including model state, optimizer state, and scheduler state.

    * set_model_mode: Set the mode of the model (train/eval).

    * model_backward_and_update: Perform model backpropagation and update, including zeroing gradients, backpropagation, and updating model parameters.
    * update_lr: Call the step() method of the learning rate scheduler to update the learning rate of models in the names list.
    * get_current_lr: Get the current learning rate. 

    * train: General training loop, but the sub-methods inside (before_train, after_train, before_epoch,
             after_epoch, run_epoch (must be implemented)) need to be implemented by subclasses.
    
    -------Methods that can be overridden by subclasses (optional)-------
    * check_cfg: Check whether certain variables in the configuration are set correctly. (Not implemented)

    * before_train: Operations before training.
    * after_train: Operations after training.
    * before_epoch: Operations before each epoch. (Not implemented) 
    * after_epoch: Operations after each epoch.
    * run_epoch: Execute training for each epoch.
    * test: Testing method.
    * parse_batch_train: Parse training batches.
    * parse_batch_test: Parse testing batches.
    * model_inference: Model inference.

    -------Methods that must be overridden by subclasses (mandatory)-------
    * init_model: Initialize the model, such as freezing certain layers of the model, loading pre-trained weights, etc. (Not implemented - freezing certain layers)
    * forward_backward: Forward and backward propagation.
    """

    def __init__(self, cfg):
        """
        Initialize the trainer.
        Main tasks include:
        * Initialize relevant attributes and read configuration information
        * Build data loaders
        * Build and register models, optimizers, and learning rate schedulers; and initialize the model
        * Build the evaluator
        """
        
        assert isinstance(cfg, dict)  

        
        self.check_cfg(cfg)  

        
        self._models = OrderedDict()  
        self._optims = OrderedDict()  
        self._scheds = OrderedDict()  
        self._writer = None  
        self.best_result = -np.inf  
        self.start_epoch = self.epoch = 0
        
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR  
        self.cfg = cfg
        
        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        
        print("Building data loaders...")
        dm = DataManager(self.cfg)  
        self.dm = dm  

        self.train_loader_x = dm.train_loader  
        
        self.val_loader = dm.val_loader  
        self.test_loader = dm.test_loader  

        self.num_classes = dm.num_classes  
        self.lab2cname = dm.lab2cname  

        
        print("Building models, optimizers, and schedulers...")
        self.model, self.optim, self.sched = self.init_model(cfg)

        
        print("Building evaluator...")
        self.evaluator = build_evaluator(cfg, lab2cname=self.lab2cname)  
        

    def init_writer(self, log_dir):
        """
        Utility method: Initialize TensorBoard.
        
        Args:
            * log_dir: Log directory

        Returns:
            * None
        """
        if self.__dict__.get("_writer") is None or self._writer is None:
            print(f"Initialize tensorboard (log_dir={log_dir})")
            self._writer = SummaryWriter(log_dir=log_dir)  

    def close_writer(self):
        """Utility method: Close TensorBoard."""
        if self._writer is not None:
            self._writer.close()  

    def write_scalar(self, tag, scalar_value, global_step=None):
        """
        Utility method: Write scalar values to TensorBoard.
        
        Args:
            * tag: Tag
            * scalar_value: Scalar value
            * global_step: Global step
            
        Returns:
            * None
        """
        if self._writer is None:
            pass  
        else:
            self._writer.add_scalar(tag, scalar_value, global_step)  
    

    def register_model(self, name="model", model=None, optim=None, sched=None):
        """
        Utility method: Register models, optimizers, and learning rate schedulers.
        self._models[name] = model  
        self._optims[name] = optim  
        self._scheds[name] = sched  

        Args:
            * name: Model name
            * model: Model
            * optim: Optimizer
            * sched: Scheduler

        Returns:
            * None
        """
        
        
        if self.__dict__.get("_models") is None:
            raise AttributeError(
                "Cannot assign model before super().__init__() call"
            )
        if self.__dict__.get("_optims") is None:
            raise AttributeError(
                "Cannot assign optim before super().__init__() call"
            )
        if self.__dict__.get("_scheds") is None:
            raise AttributeError(
                "Cannot assign sched before super().__init__() call"
            )
        assert name not in self._models, "Found duplicate model names"  

        
        self._models[name] = model  
        self._optims[name] = optim  
        self._scheds[name] = sched  

    def get_model_names(self, names=None):
        """
        Utility method: Get all registered model names.
        self._models.keys()
        """
        names_real = list(self._models.keys())  
        if names is not None:
            names = tolist_if_not(names)  
            for name in names:
                assert name in names_real  
            return names
        else:
            return names_real
    
    def save_model(self, epoch, directory, is_best=False, val_result=None, model_name=""):
        """
        Utility method: Save the model.
        Parameters:
            * epoch: Current epoch
            * directory: Save directory
            * is_best: Whether it is the best model
            * val_result: Validation result
            * model_name: Model name

        Returns:
            * None

        Saved content:
            * Model state dictionary
            * Epoch
            * Optimizer state dictionary
            * Scheduler state dictionary
            * Validation result
        """
        names = self.get_model_names()  
        for name in names:
            model_dict = self._models[name].state_dict()  

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()  

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()  

            
            save_checkpoint(
                {
                    "state_dict": model_dict,
                    "epoch": epoch + 1,
                    "optimizer": optim_dict,
                    "scheduler": sched_dict,
                    "val_result": val_result
                },
                osp.join(directory, name),
                is_best=is_best,
                model_name=model_name,
            )

    def load_model(self, directory, epoch=None):
        """
        Utility method: Load the model file from the directory into the model.
        
        Parameters:
            * directory: Model directory
            * epoch: Epoch
        
        Returns:
            * None

        Loaded content:
            * Model state dictionary
            * Epoch
            * Validation result
        """
        assert directory is not None, "The parameter 'directory' of load_model() is None"

        names = self.get_model_names()  

        model_file = "model-best.pth.tar" 
        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch) 

        
        for name in names: 
            model_path = osp.join(directory, name, model_file) 

            if not osp.exists(model_path): 
                raise FileNotFoundError(f"No model at {model_path}")

            checkpoint = load_checkpoint(model_path) 
            state_dict = checkpoint["state_dict"] 
            epoch = checkpoint["epoch"] 
            val_result = checkpoint["val_result"] 
            print(
                f"Load {model_path} to {name} (epoch={epoch}, val_result={val_result:.1f})"
            )
            self._models[name].load_state_dict(state_dict) 


    def resume_model_if_exist(self, directory):
        """
        Utility method: Resume the model if a checkpoint exists.
        
        Parameters:
            * directory: Checkpoint directory
            
        Returns:
            * start_epoch: Starting epoch

        Resumed content:
            * Model state dictionary
            * Optimizer state dictionary
            * Learning rate scheduler state dictionary
        """
        names = self.get_model_names()  
        
        
        file_missing = False
        for name in names:
            path = osp.join(directory, name)
            if not osp.exists(path):
                file_missing = True 
                break
        if file_missing: 
            print("No checkpoint found, train from scratch")
            return 0
        print(f"Found checkpoint at {directory} (will resume training)")

        
        for name in names:
            path = osp.join(directory, name)
            start_epoch = resume_from_checkpoint( 
                path, self._models[name], self._optims[name], 
                self._scheds[name] 
            )
        return start_epoch 

    
    def set_model_mode(self, mode="train", names=None):
        """
        Utility method: Set the mode (train/eval) of the model.
        If names is None, set the mode for all models.

        Parameters:
            * mode: Mode
            * names: List of model names to set
        """
        names = self.get_model_names(names)  
        
        
        for name in names:
            if mode == "train":
                self._models[name].train()  
            elif mode in ["test", "eval"]:
                self._models[name].eval()  
            else:
                raise KeyError

    
    def model_backward_and_update(self, loss, names=None):
        """
        Utility method: Model backward propagation and update.
        
        Parameters:
            * loss: Loss
            * names: List of model names to update
            
        Returns:
            * None
            
        Process:
            1. Zero gradients (Iterate through the list of model names to update, call the optimizer's zero_grad() method to zero gradients)
            2. Backward propagation (Check if the loss is finite, if not, raise an exception)
            3. Update model parameters (Call the optimizer's step() method)
        """
        
        names = self.get_model_names(names)  
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].zero_grad()  

        
        
        if not torch.isfinite(loss).all(): 
            raise FloatingPointError("Loss is infinite or NaN!") 
        loss.backward()  
        
        
        names = self.get_model_names(names)  
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].step()  


    def update_lr(self, names=None):
        """
        Utility method: Call the step() method of the learning rate scheduler to update the learning rate of the models in the names list.
        
        Parameters:
            * names: List of model names whose learning rate needs to be updated
        Returns:
            * None

        Method:
            * Call the step() method of the learning rate scheduler   
        """
        names = self.get_model_names(names)  

        
        for name in names:
            if self._scheds[name] is not None:
                self._scheds[name].step()  

    
    def get_current_lr(self, names=None):
        """
        Utility method: Get the current learning rate.
        
        Parameters:
            * names: List of model names whose learning rate needs to be retrieved
            
        Returns:
            * The learning rate of the first model in the list of model names
        """
        names = self.get_model_names(names)
        name = names[0] 
        return self._optims[name].param_groups[0]["lr"]

    
    def train(self, start_epoch, max_epoch):
        """
        Utility method: General training loop.
            
        Process:
        1. Perform pre-training operations before_train()
        2. Start training
            * Perform operations before each epoch before_epoch()
                * Set output directory
                * If a checkpoint exists in the output directory, restore the checkpoint
                * Initialize summary writer
                * Record start time (used to calculate elapsed time)
            * Perform training for each epoch run_epoch()
            * Perform operations after each epoch after_epoch()
        3. Perform post-training operations after_train()
        """
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch

        
        self.before_train()
        
        
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch() 
            self.run_epoch() 
            self.after_epoch() 
        
        
        self.after_train() 


    def check_cfg(self, cfg):
        """
        Check whether certain variables in the configuration are set correctly (optional subclass implementation).   
        Not implemented
        """
        pass

    def before_train(self):
        """
        Pre-training operations. (Optional subclass implementation)
        
        Mainly includes:
        * Set output directory
        * If a checkpoint exists in the output directory, restore the checkpoint
        * Initialize summary writer
        * Record start time (used to calculate elapsed time)
        """
        
        if self.cfg.RESUME: 
            directory = self.cfg.RESUME 
            self.start_epoch = self.resume_model_if_exist(directory) 
        else: 
            directory =self.output_dir
            mkdir_if_missing(directory) 
        
        writer_dir = osp.join(self.output_dir, "tensorboard")
        mkdir_if_missing(writer_dir) 
        self.init_writer(writer_dir) 

        
        self.time_start = time.time()

    def after_train(self):
        """
        Post-training operations. (Optional subclass implementation)
        
        Mainly includes:
        * If testing is required after training, test and save the best model; otherwise, save the model of the last epoch
        * Print elapsed time
        * Close writer
        """
        
        print("Training completed")
        
        
        do_test = not self.cfg.TRAIN.NO_TEST 
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Testing the best validation performance model")
                self.load_model(self.output_dir)
            else:
                print("Testing the model of the last epoch")
            self.test()

        
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed time: {elapsed}")

        
        self.close_writer()


    def before_epoch(self):
        """
        Operations before each epoch. (Optional for subclasses to implement)
        Not implemented.
        """
        pass

    def after_epoch(self):
        """
        Operations after each epoch. (Optional for subclasses to implement)
        
        Mainly includes:
        * Determine model saving conditions: whether it is the last epoch, whether validation is needed, whether checkpoint saving frequency is met
        * Save the model based on conditions
        """
        
        is_last_epoch = (self.epoch + 1) == self.max_epoch 
        need_eval = self.cfg.TRAIN.DO_EVAL 
        meet_checkpoint_freq = (  
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        
        if need_eval:
            
            curr_result = self.test(split="val")  
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=curr_result,
                    model_name="model-best.pth.tar"
                )
        if meet_checkpoint_freq or is_last_epoch:
            
            self.save_model(self.epoch, self.output_dir)

    def run_epoch(self):
        """
        Execute training for each epoch. (Subclasses can override)
        This uses the standard labeled data training mode.

        Mainly includes:
        * Set the model to training mode
        * Initialize metrics: loss meter, batch time meter, data loading time meter
        * Start iteration
            — Iterate through the labeled dataset train_loader_x
            — Forward and backward propagation, calculate loss
            — Print logs (epoch, batch, time, data loading time, loss, learning rate, remaining time)
        """
        
        
        self.set_model_mode("train")  

        
        losses = MetricMeter()  
        batch_time = AverageMeter()  
        data_time = AverageMeter()  

        
        self.num_batches = len(self.train_loader_x)  
        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x): 
            data_time.update(time.time() - end)  
            
            
            loss_summary = self.forward_backward(batch)  
            
            
            batch_time.update(time.time() - end)  
            losses.update(loss_summary)  

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0  
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ  
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1  
                nb_remain += (self.max_epoch - self.epoch - 1) * self.num_batches  
                eta_seconds = batch_time.avg * nb_remain  
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                
                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))
            
            n_iter = self.epoch * self.num_batches + self.batch_idx  
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)  
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)  
            end = time.time()

    @torch.no_grad()
    def test(self, split=None):
        """
        Testing. (Subclasses can override)

        Mainly includes:
        * Set the model mode to eval, reset the evaluator
        * Determine the test set (val or test, default is test set)
        * Start testing
            - Iterate through the data loader
            - Parse the test batch to get input and labels - self.parse_batch_test(batch)
            - Model inference - self.model_inference(input)
            - Evaluator evaluates model output and labels - self.evaluator.process(output, label)
        * Use the evaluator to evaluate the results and record them in tensorboard
        * Return results (accuracy in this case)
        """
        
        self.set_model_mode("eval")
        self.evaluator.reset() 

        
        if split is None: 
            split = self.cfg.TEST.SPLIT 
        if split == "val" and self.val_loader is not None: 
            data_loader = self.val_loader
        else:
            split = "test"
            data_loader = self.test_loader
        print(f"Testing on *{split}* set")

        
        for batch_idx, batch in enumerate(tqdm(data_loader)): 
            input, label = self.parse_batch_test(batch) 
            output = self.model_inference(input) 
            self.evaluator.process(output, label) 

        
        results = self.evaluator.evaluate() 
        for k, v in results.items(): 
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0] 

    def parse_batch_train(self, batch):
        """
        Parse training batch. (Subclasses can override)
        Here, input images, class labels, and domain labels are directly retrieved from the batch dictionary.
        """
        input = batch["img"]  
        label = batch["label"]  

        input = input.to(self.device)  
        label = label.to(self.device)  

        return input, label  


    def parse_batch_test(self, batch):
        """
        Parse testing batch. (Subclasses can override)   
        Here, input and labels are directly retrieved from the batch dictionary.
        """
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
    
    def model_inference(self, input):
        """
        Model inference. (Subclasses can override)  
        Here, the model is directly called, and the model output is returned.
        """
        return self.model(input) 
    
    def init_model(self, cfg):
        """
        Initialize the model (subclasses need to override, only an example is provided).
        Main tasks include:
        * Build the model
        * Load pretrained weights
        * Freeze certain layers of the model
        * Move the model to the device
        * Adjust the model for mixed-precision training
        * Deploy the model on multiple GPUs
        * Build optimizers and learning rate schedulers for the entire model or specific modules, and register them
        
        Parameters:
            * cfg: Configuration

        Returns:
            * model: The model
            * optim: Optimizer
            * sched: Learning rate scheduler
        """
        
        self.model = build_model(cfg) 
        print("Number of model parameters:", count_num_param(self.model))
        
        if cfg.MODEL.INIT_WEIGHTS_PATH: 
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS_PATH)  
        pass  
        
        self.model.to(self.device)
        
        self.scaler = GradScaler() if cfg.TRAINER.PREC == "amp" else None
        
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), using all GPUs!")
            self.model = nn.DataParallel(self.model)

        self.optim = build_optimizer(self.model, cfg)  
        self.sched = build_lr_scheduler(cfg, self.optim)  
        self.register_model(cfg.MODEL.NAME, self.model, self.optim, self.sched) 

        return self.model, self.optim, self.sched

    def forward_backward(self, batch):
        """
        Forward and backward propagation. (Needs to be implemented by subclasses)
        Not implemented.
        """
        raise NotImplementedError