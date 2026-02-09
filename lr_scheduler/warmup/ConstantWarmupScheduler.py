from .BaseWarmupScheduler import BaseWarmupScheduler
from .build import WARMUP_REGISTRY

@WARMUP_REGISTRY.register()
class ConstantWarmupScheduler(BaseWarmupScheduler):
    """
    Constant learning rate warmup wrapper.
    During the warmup period, the learning rate remains constant, equal to cons_lr.
    After the warmup period ends, the learning rate is controlled by the successor scheduler.
    """
    # Initialization method
    def __init__(self, cfg, successor, last_epoch=-1):
        """ Initialization method
        Args:
            cfg (CfgNode): Configuration
            successor (_LRScheduler): Successor learning rate scheduler
            last_epoch (int): Last epoch (the current completed epoch)
                - When resuming training manually, you can pass the value of the last epoch to ensure the learning rate starts decaying from the correct position.
                - Default: -1, indicating training has not started yet.

        Current configuration:
            - cons_lr (float): cfg.LR_SCHEDULER.WARMUP.CONS_LR: Constant learning rate
        
        General configuration (retrieved in BaseWarmupScheduler):
            - warmup_recount (bool): cfg.LR_SCHEDULER.WARMUP.WARMUP_RECOUNT: Whether to reset the epoch count after warmup
            - warmup_epoch (int): cfg.LR_SCHEDULER.WARMUP.EPOCHS: Warmup epochs
            - verbose (bool): cfg.VERBOSE: Whether to print information
        """
        self.cons_lr = float(cfg.LR_SCHEDULER.WARMUP.CONS_LR)  # Constant learning rate
        super().__init__(cfg, successor, last_epoch)
        
    def get_lr(self):
        """
        Logic for calculating the next learning rate (implements _LRScheduler's get_lr()).
        When creating a custom scheduler, you need to implement _LRScheduler's get_lr() to define how the learning rate is calculated.

        Returns:
            - list: A list containing the calculated learning rate for each parameter group.
        """
        if self.last_epoch >= self.warmup_epoch:  # If the current epoch is greater than or equal to the warmup period
            return self.successor.get_last_lr()  # Return the learning rate from the successor scheduler
        else:  # If the current epoch is still within the warmup period
            return [self.cons_lr for _ in self.base_lrs]  # Return the constant learning rate
