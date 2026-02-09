from .build import OPTIMIZER_REGISTRY
from torch.optim import SGD as TorchSGD

@OPTIMIZER_REGISTRY.register()
class SGD(TorchSGD):
    """ SGD Optimizer """
    def __init__(self, cfg, params=None):
        """
        Initialize the SGD optimizer

        Args:
            - cfg (CfgNode): Configuration
            - params (iterable): Model parameters

        Configuration:
            - Default optimizer parameters
                - OPTIMIZER.LR (float): Learning rate
                - OPTIMIZER.momentum (float): Momentum
                - OPTIMIZER.weight_decay (float): Weight decay
                - OPTIMIZER.dampening (float): Dampening
                - OPTIMIZER.nesterov (bool): Whether to use Nesterov momentum
        """
        
        # ---Read configuration---
        # Read default parameters for the optimizer
        lr = float(cfg.OPTIMIZER.LR)
        momentum = float(cfg.OPTIMIZER.momentum)
        weight_decay = float(cfg.OPTIMIZER.weight_decay)
        dampening = float(cfg.OPTIMIZER.dampening)
        nesterov = bool(cfg.OPTIMIZER.nesterov)

        # ---Check parameter validity---
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= dampening:
            raise ValueError("Invalid dampening value: {}".format(dampening))
        
        # ---Pass default parameters to the parent class---
        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=dampening,
            nesterov=nesterov  # Whether to use Nesterov momentum
        )