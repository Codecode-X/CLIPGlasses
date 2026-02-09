from torch.optim.lr_scheduler import StepLR
from .build import LRSCHEDULER_REGISTRY

@LRSCHEDULER_REGISTRY.register()
class SingleStepLrScheduler(StepLR):
    """
    Single-step learning rate scheduler.
    SingleStepLrScheduler is a wrapper class for torch.optim.lr_scheduler.StepLR,
    using a registration mechanism to facilitate unified management of learning rate schedulers in the project.

    Parameters:
        - cfg (Config): Configuration object containing parameters related to the learning rate scheduler.
        - optimizer (torch.optim.Optimizer): Optimizer used during training.
    
    Related configuration items:
        - cfg.LR_SCHEDULER.STEP_SIZE: Step size, the number of epochs after which the learning rate is reduced.
        - cfg.LR_SCHEDULER.GAMMA: Learning rate decay factor.
    """
    def __init__(self, cfg, optimizer):

        step_size = int(cfg.LR_SCHEDULER.STEP_SIZE) # Number of epochs after which the learning rate is reduced
        assert step_size > 0, "Step size must be greater than 0"
            
        super().__init__(
            optimizer=optimizer,
            step_size=step_size, # Number of epochs after which the learning rate is reduced
            gamma=float(cfg.LR_SCHEDULER.GAMMA) # Decay factor
        )
