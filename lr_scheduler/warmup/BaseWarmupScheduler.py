from torch.optim.lr_scheduler import _LRScheduler
from .build import WARMUP_REGISTRY

@WARMUP_REGISTRY.register()
class BaseWarmupScheduler(_LRScheduler):
    """
    Base class for learning rate warm-up wrapper.
    This class inherits from _LRScheduler and provides a general structure for learning rate warm-up wrappers.
    
    Subclasses need to implement the following methods:
        - __init__(): Initialization method
        - get_lr(): Compute the next learning rate
    """
    def __init__(self, cfg, successor, last_epoch=-1):
        """ Initialization method 
        Args:
            - optimizer (Optimizer): Optimizer 
            - successor (LRScheduler): Successor learning rate scheduler
            - last_epoch (int): Last epoch (current completed epoch)
                - When resuming training manually, you can pass the value of the last epoch to ensure the learning rate starts decaying from the correct position.
                - Default: -1, indicating training has not started yet.

        Config:
            - warmup_recount (bool): cfg.LR_SCHEDULER.WARMUP.WARMUP_RECOUNT: Whether to reset the epoch after warm-up ends
            - warmup_epoch (int): cfg.LR_SCHEDULER.WARMUP.EPOCHS: Warm-up epochs
            - verbose (bool): cfg.VERBOSE: Whether to print information
        """
        warmup_epoch = int(cfg.LR_SCHEDULER.WARMUP.EPOCHS)  # Warm-up epochs
        verbose = bool(cfg.VERBOSE)  # Whether to print logs
        self.warmup_recount = bool(cfg.LR_SCHEDULER.WARMUP.WARMUP_RECOUNT)  # Whether to reset epoch after warm-up ends

        self.successor = successor  # Successor learning rate scheduler
        self.warmup_epoch = warmup_epoch  # Warm-up epochs

        optimizer = successor.optimizer  # Optimizer

        super().__init__(optimizer, last_epoch, verbose)  # Call the parent class's initialization method

    def get_lr(self):
        """
        Logic for computing the next learning rate (implements _LRScheduler's get_lr())
        When customizing a scheduler, you need to implement _LRScheduler's get_lr() to define how the learning rate is calculated.

        Returns:
            - list: A list containing the computed learning rate for each parameter group.

        Other:
            - The difference between get_lr() and get_last_lr() is that get_last_lr() returns the learning rate of the last epoch, while get_lr() returns the learning rate of the next epoch.
        """
        raise NotImplementedError

    def step(self, epoch=None):
        """
        Override _LRScheduler's method to update the learning rate.

        Args:
            - epoch (int): Current epoch number
                - If training is resumed after interruption, passing the current epoch ensures the learning rate starts decaying from the correct position.
                - Default: None, indicating the current epoch is the last epoch + 1.
        """
        if self.last_epoch >= self.warmup_epoch:  # If the warm-up period has ended
            # If the epoch needs to be reset after warm-up ends
            if self.warmup_recount:
                print("Since warmup_recount is set and warm-up has ended, resetting epoch to -1")
                self.last_epoch = -1  # Reset epoch count

            self.successor.step(epoch)  # Use the successor learning rate scheduler to update the learning rate
            self._last_lr = self.successor.get_last_lr()  # Get the latest learning rate
        else:
            super().step(epoch)  # Otherwise, use the parent class's method to update the learning rate
