from .build import OPTIMIZER_REGISTRY
from torch.optim import RMSprop as TorchRMSprop

@OPTIMIZER_REGISTRY.register()
class RMSprop(TorchRMSprop):
    """ RMSprop optimizer """
    def __init__(self, cfg, params=None):
        """
        Initialize the RMSprop optimizer

        Args:
            - cfg (CfgNode): Configuration
            - params (iterable): Model parameters

        Configuration:
            - Default optimizer parameters
                - OPTIMIZER.LR (float): Learning rate
                - OPTIMIZER.alpha (float): Smoothing constant
                - OPTIMIZER.eps (float): Term added to the denominator to improve numerical stability
                - OPTIMIZER.weight_decay (float): Weight decay (L2 penalty)
                - OPTIMIZER.momentum (float): Momentum factor
                - OPTIMIZER.centered (bool): Whether to use centered RMSprop
        """
        
        # ---Read configuration---
        # Read default optimizer parameters
        lr = float(cfg.OPTIMIZER.LR)
        alpha = float(cfg.OPTIMIZER.alpha)
        eps = float(cfg.OPTIMIZER.eps)
        weight_decay = float(cfg.OPTIMIZER.weight_decay)
        momentum = float(cfg.OPTIMIZER.momentum)
        centered = bool(cfg.OPTIMIZER.centered)

        # ---Validate parameters---
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        
        # ---Pass default parameters to the parent class---
        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            alpha=alpha,
            eps=eps,
            centered=centered  # Whether to use centered RMSprop
        )