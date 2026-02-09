from torch.optim.lr_scheduler import MultiStepLR
from .build import LRSCHEDULER_REGISTRY

@LRSCHEDULER_REGISTRY.register()
class MultiStepLrScheduler(MultiStepLR):
    """
    Multi-step learning rate scheduler.
    MultiStepLrScheduler is a wrapper class for torch.optim.lr_scheduler.MultiStepLR,
    using a registration mechanism to facilitate unified management of learning rate schedulers in the project.

    Parameters:
        - cfg (Config): Configuration object containing parameters related to the learning rate scheduler.
        - optimizer (torch.optim.Optimizer): Optimizer used during training.

    Related configuration items:
        - cfg.LR_SCHEDULER.MILESTONES(list<int>): Epochs at which the learning rate decreases.
        - cfg.LR_SCHEDULER.GAMMA(float): Learning rate decay factor.
    """
    def __init__(self, cfg, optimizer):

        milestones = list(map(int, cfg.LR_SCHEDULER.MILESTONES))  # Epochs at which the learning rate decreases
        assert isinstance(milestones, list), f"milestones must be a list, but got {type(milestones)}"
        assert len(milestones) > 0, "milestones list cannot be empty"   
        assert all(isinstance(i, int) for i in milestones), "milestones must be a list of integers"

        super().__init__(
            optimizer=optimizer,
            milestones=milestones, # Epochs at which the learning rate decreases
            gamma=float(cfg.LR_SCHEDULER.GAMMA) # Decay factor
        )