from torch.optim import Optimizer

class OptimizerBase(Optimizer):
    """
    Base class for optimizers.
    Inherits from torch.optim.Optimizer and provides a general structure for optimizers.

    Subclasses need to implement the following methods:
        - __init__(): Initialization method
        - step(): Perform parameter updates
    """

    def __init__(self, params, defaults):
        """
        Initialize the optimizer.

        Args:
            - params (iterable): List of parameter groups to optimize.
            Each parameter group is a dictionary containing a set of model parameters 
            and their corresponding hyperparameters (e.g., learning rate, weight decay, etc.).
            
            - defaults (dict): Default parameters for the optimizer.
            If a parameter group does not explicitly specify a hyperparameter, 
            the value from defaults will be used.
        """
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure=None):
        """
        Perform parameter updates. Subclasses must implement the specific step() logic
        (inherited from torch.optim.Optimizer).

        Args:
            - closure (callable, optional): An optional closure function to compute the loss 
            and perform backpropagation.
        
        Returns:
            - loss: The computed loss value
        """
        raise NotImplementedError
