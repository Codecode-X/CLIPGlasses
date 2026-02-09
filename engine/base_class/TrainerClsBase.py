from ..build import TRAINER_REGISTRY
from .TrainerBase import TrainerBase
import torch
from tqdm import tqdm

@TRAINER_REGISTRY.register()
class TrainerClsBase(TrainerBase):
    """
    Base class for iterative trainers for classification tasks.
    
    Contains methods:

    -------Utility Methods-------
    Parent class TrainerClsBase:
        * init_writer: Initialize TensorBoard.
        * close_writer: Close TensorBoard.
        * write_scalar: Write scalar values to TensorBoard.

        * register_model: Register models, optimizers, and learning rate schedulers.
        * get_model_names: Get the names of all registered models.

        * save_model: Save the model, including model state, epoch, optimizer state, learning rate scheduler state, and validation results.
        * load_model: Load the model, including model state, epoch, and validation results.
        * resume_model_if_exist: Resume the model if a checkpoint exists, including model state, optimizer state, and learning rate scheduler state.

        * set_model_mode: Set the mode of the model (train/eval).

        * model_backward_and_update: Perform model backpropagation and update, including zeroing gradients, backpropagation, and updating model parameters.
        * update_lr: Call the step() method of the learning rate scheduler to update the learning rate of models in the names list.
        * get_current_lr: Get the current learning rate.

        * train: General training loop, but the sub-methods inside (before_train, after_train, before_epoch, after_epoch, run_epoch (must be implemented)) need to be implemented by subclasses.
    
    Current TrainerBase:
        * None
    
    -------Methods that can be overridden by subclasses (optional)-------
    Parent class TrainerClsBase:
        * check_cfg: Check whether certain variables in the configuration are set correctly. (Not implemented)

        * before_train: Operations before training.
        * after_train: Operations after training.
        * before_epoch: Operations before each epoch. (Not implemented) 
        * after_epoch: Operations after each epoch.
        * run_epoch: Execute training for each epoch.
        * test: Testing method. (Not implemented) 
        * model_inference: Model inference.
    
    Current TrainerBase:
        * parse_batch_train: Parse training batches for classification tasks. (Implements parent class method)
        * parse_batch_test: Parse testing batches for classification tasks. (Implements parent class method)
        * test: Testing method. (Implements parent class method)

    -------Methods that must be overridden by subclasses (mandatory)-------
    Parent class TrainerClsBase:
        * init_model: Initialize the model, such as freezing certain layers of the model, loading pre-trained weights, etc. (Not implemented - Freezing certain layers of the model)
        * forward_backward: Forward and backward propagation.
    
    Current TrainerBase:
        * None
    """

    def parse_batch_train(self, batch):
        """
        (Implements parent class method) Parse training batches for classification tasks. 
        Directly retrieves input images, class labels, and domain labels from the batch dictionary.

        Parameters:
            - batch (dict): Batch data dictionary containing input images and labels.

        Returns:
            - input (Tensor): Input images.
            - label (Tensor): Class labels.
        """
        input = batch["img"].to(self.device)  # Retrieve images
        label = batch["label"].to(self.device)  # Retrieve labels | shape: [batch, 1]
        return input, label  # Return images and labels


    def parse_batch_test(self, batch):
        """
        (Implements parent class method) Parse testing batches for classification tasks.
        Returns data parsed from the batch dictionary, such as input images and class labels for classification problems.
        
        Parameters:
            - batch (dict): Batch data dictionary containing input images and labels.

        Returns:
            - input (Tensor): Input images.
            - label (Tensor): Class labels.
        """
        input = batch["img"].to(self.device)  # Retrieve images
        label = batch["label"].to(self.device)  # Retrieve labels
        return input, label  # Return images and labels
    

    @torch.no_grad()
    def test(self, split=None):
        """
        Testing. (Subclasses can override)

        Main steps include:
        * Set model mode to eval, reset evaluator
        * Determine the test set (val or test, default is test set)
        * Start testing
            - Iterate through the data loader
            - Parse testing batches to get inputs and labels - self.parse_batch_test(batch)
            - Perform model inference - self.model_inference(input)
            - Evaluator evaluates model outputs and labels - self.evaluator.process(output, label)
        * Use evaluator to evaluate results and record them in TensorBoard
        * Return results (here, accuracy)
        """
        
        self.set_model_mode("eval")
        self.evaluator.reset() # Reset evaluator

        # Determine the test set (val or test, default is test set)
        if split is None: # If split is None, use the test set from the configuration
            split = self.cfg.TEST.SPLIT 
        if split == "val" and self.val_loader is not None: 
            data_loader = self.val_loader
        else:
            split = "test"
            data_loader = self.test_loader
        print(f"Testing on the *{split}* set")

        # Start testing
        for batch_idx, batch in enumerate(tqdm(data_loader)): # Iterate through the data loader
            input, label = self.parse_batch_test(batch) # Parse testing batches to get inputs and labels
            output = self.model_inference(input) # Perform model inference
            self.evaluator.process(output, label) # Evaluator evaluates model outputs and labels

        # Use evaluator to evaluate results and record them in TensorBoard
        results = self.evaluator.evaluate() 
        for k, v in results.items(): 
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0] # Return the first value: accuracy
