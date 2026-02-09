from torch.optim.lr_scheduler import CosineAnnealingLR
from .build import LRSCHEDULER_REGISTRY

@LRSCHEDULER_REGISTRY.register()
class CosineLrScheduler(CosineAnnealingLR):
    """
    Cosine learning rate scheduler.
    CosineLrScheduler is a wrapper class for torch.optim.lr_scheduler.CosineAnnealingLR,
    using a registration mechanism to facilitate unified management of learning rate schedulers in the project.

    Parameters:
        - cfg (Config): Configuration object containing parameters related to the learning rate scheduler.
        - optimizer (torch.optim.Optimizer): Optimizer used during training.

    Related configuration items:
        - cfg.TRAIN.MAX_EPOCH (int): Maximum number of training epochs.

    """
    def __init__(self, cfg, optimizer):
        T_max = int(cfg.TRAIN.MAX_EPOCH) # Maximum number of epochs
        assert isinstance(T_max, int), f"T_max must be an integer, but got {type(T_max)}"
        assert T_max > 0, "T_max must be greater than 0"
        
        super().__init__(
            optimizer=optimizer, # Optimizer
            T_max=T_max, # Maximum number of epochs
        )