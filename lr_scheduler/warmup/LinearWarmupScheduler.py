from .BaseWarmupScheduler import BaseWarmupScheduler
from .build import WARMUP_REGISTRY

@WARMUP_REGISTRY.register()
class LinearWarmupScheduler(BaseWarmupScheduler):
    """
    Linear learning rate warmup wrapper.
    During the warmup phase, the learning rate changes linearly.
    After the warmup phase, the learning rate is controlled by the successor scheduler.

    """
    def __init__(self, cfg, successor, last_epoch=-1):
        """ Initialization method
        Args:
            cfg (CfgNode): Configuration
            successor (_LRScheduler): Successor learning rate scheduler
            last_epoch (int): Last epoch (the most recently completed epoch)
                - When resuming training manually, you can pass the value of the last epoch to ensure the learning rate decays from the correct position.
                - Default: -1, indicating training has not started yet.
        
        Current configuration:
            - min_lr (float): cfg.LR_SCHEDULER.WARMUP.MIN_LR: Minimum learning rate
        
        General configuration (retrieved in BaseWarmupScheduler):
            - warmup_recount (bool): cfg.LR_SCHEDULER.WARMUP.WARMUP_RECOUNT: Whether to reset the epoch count after warmup
            - warmup_epoch (int): cfg.LR_SCHEDULER.WARMUP.EPOCHS: Warmup epochs
            - verbose (bool): cfg.VERBOSE: Whether to print information
        """
        self.min_lr = float(cfg.LR_SCHEDULER.WARMUP.MIN_LR)  # Minimum learning rate
        super().__init__(cfg, successor, last_epoch)

    def get_lr(self):
        """(Implements the abstract method of the parent class) Method to get the learning rate"""
        if self.last_epoch >= self.warmup_epoch:  # If the current epoch is greater than or equal to (not in) the warmup phase
            # Here, last_epoch is a property of the parent class _LRScheduler, which is automatically updated during each step()
            return self.successor.get_last_lr()  # Return the learning rate from the successor scheduler
        
        # If the current epoch is still within the warmup phase
        if self.last_epoch == 0:  # If it is the first epoch
            return [self.min_lr for _ in self.base_lrs]  # Return the minimum learning rate
        return [
            lr * self.last_epoch/self.warmup_epoch for lr in self.base_lrs
        ]  # Otherwise, return the linearly changing learning rate