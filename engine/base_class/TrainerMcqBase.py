from ..build import TRAINER_REGISTRY
from .TrainerBase import TrainerBase
import torch

@TRAINER_REGISTRY.register()
class TrainerMcqBase(TrainerBase):
    """
    Base class for iterative trainers for classification tasks.
    
    Contains methods:

    -------Utility Methods-------
    Parent class TrainerMcqBase:
        * init_writer: Initialize TensorBoard.
        * close_writer: Close TensorBoard.
        * write_scalar: Write scalar values to TensorBoard.

        * register_model: Register model, optimizer, and learning rate scheduler.
        * get_model_names: Get all registered model names.

        * save_model: Save model, including model state, epoch, optimizer state, learning rate scheduler state, and validation results.
        * load_model: Load model, including model state, epoch, and validation results.
        * resume_model_if_exist: Resume model if a checkpoint exists, including model state, optimizer state, and learning rate scheduler state.

        * set_model_mode: Set the mode of the model (train/eval).

        * model_backward_and_update: Perform model backpropagation and update, including zeroing gradients, backpropagation, and updating model parameters.
        * update_lr: Call the step() method of the learning rate scheduler to update the learning rate of models in the names list.
        * get_current_lr: Get the current learning rate.

        * train: General training loop, but the sub-methods inside (before_train, after_train, before_epoch, after_epoch, run_epoch (must be implemented)) need to be implemented by subclasses.
    
    Current TrainerBase:
        * None
    
    -------Methods that can be overridden by subclasses (optional)-------
    Parent class TrainerMcqBase:
        * check_cfg: Check whether certain variables in the configuration are set correctly. (Not implemented)

        * before_train: Operations before training.
        * after_train: Operations after training.
        * before_epoch: Operations before each epoch. (Not implemented) 
        * after_epoch: Operations after each epoch.
        * run_epoch: Execute training for each epoch.
        * model_inference: Model inference.
    
    Current TrainerBase:
        * parse_batch_train: Parse training batches for classification tasks. (Implements parent class method)
        * parse_batch_test: Parse testing batches for classification tasks. (Implements parent class method)

    -------Methods that must be implemented by subclasses (mandatory)-------
    Parent class TrainerMcqBase:
        * init_model: Initialize the model, such as freezing certain layers or loading pre-trained weights. (Not implemented - Freeze certain layers)
        * forward_backward: Forward and backward propagation.
    
    Current TrainerBase:
        * test: Testing method.
    """

    def __init__(self, cfg):
        # Read additional configuration
        self.num_choices = int(cfg.DATASET.NUM_CHOICES)

        # Call parent class constructor
        super().__init__(cfg)  

    def parse_batch_train(self, batch):
        """
        (Implements parent class method) Parse training batches for classification tasks. 
        Directly retrieves input images, class labels, and domain labels from the batch dictionary.

        Parameters:
            - batch (dict): Batch data dictionary, containing:
                - Input images, number of answer options, answer options, correct answer index, correct answer type.
                
        Returns:
            - input (Tensor): Input images | [batch, 3, 224, 224].
            - num_choices (Tensor): Number of answer options | [batch].
            - choices (list<list<str>>): All answer option texts | (batch, num_choices).
            - correct_answer (Tensor): Index of the correct answer option | [batch].
            - correct_answer_type (list<str>): Type of the correct answer option | (batch). 
        """
        input = batch["img"].to(self.device)  # Retrieve a batch of images | [batch, 3, 224, 224]
        num_choices = batch["num_choices"].to(self.device) # Retrieve the number of answer options | [batch]
        correct_answer = batch["correct_answer"].to(self.device) # Retrieve the index of the correct answer option | [batch]
        
        choices = batch["choices"] # Retrieve all answer option texts | (batch, num_choices) | DataLoader does not automatically convert strings
        correct_answer_type = batch["correct_answer_type"] # Retrieve the type of the correct answer option | (batch)
        
        return input, num_choices, choices, correct_answer, correct_answer_type

    def parse_batch_test(self, batch):
        """
        (Implements parent class method) Parse testing batches for classification tasks.
        Returns parsed data from the batch dictionary, such as input images and class labels for classification problems.
        
        Parameters:
            - batch (dict): Batch data dictionary, containing input images and labels.
                - Input images, number of answer options, answer options, correct answer index, correct answer type.

        Returns:
            - input (Tensor): Input images | [batch, 3, 224, 224].
            - num_choices (Tensor): Number of answer options | [batch].
            - choices (list<list<str>>): All answer option texts | (batch, num_choices).
            - correct_answer (Tensor): Index of the correct answer option | [batch].
            - correct_answer_type (list<str>): Type of the correct answer option | (batch). 
        """
        input = batch["img"].to(self.device)  # Retrieve a batch of images | [batch, 3, 224, 224]
        num_choices = batch["num_choices"].to(self.device) # Retrieve the number of answer options | [batch]
        correct_answer = batch["correct_answer"].to(self.device) # Retrieve the index of the correct answer option | [batch]
        
        correct_answer_type = batch["correct_answer_type"] # Retrieve the type of the correct answer option | (batch)
        choices = batch["choices"]# Retrieve all answer option texts | (batch, num_choices) | DataLoader does not automatically convert strings
        
        return input, num_choices, choices, correct_answer, correct_answer_type

    

    @torch.no_grad()
    def test(self, split=None):
        """
        Testing. (Subclasses need to override this method based on task characteristics)

        Main steps include:
        * Set model mode to eval, reset evaluator
        * Determine the test set (val or test, default is test set)
        * Start testing
            - Iterate through the data loader
            - Parse test batches to retrieve inputs and ground truth (GT)
            - Perform model inference
            - Use the evaluator to evaluate model outputs and GT
        * Evaluate results using the evaluator and log results in TensorBoard
        * Return results (e.g., accuracy)
        """
        raise NotImplementedError("The test() method in TrainerMcqBase needs to be implemented by subclasses based on model characteristics")
