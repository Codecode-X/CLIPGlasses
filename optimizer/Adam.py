from .build import OPTIMIZER_REGISTRY
from torch.optim import Adam as TorchAdam


@OPTIMIZER_REGISTRY.register()
class Adam(TorchAdam):
    """ Adam optimizer """
    def __init__(self, cfg, params=None):
        """
        Initialize the Adam optimizer

        Args:
            - cfg (CfgNode): Configuration
            - params (iterable): Model parameters

        Configuration:
            - Default optimizer parameters
                - OPTIMIZER.LR (float): Learning rate
                - OPTIMIZER.betas (Tuple[float, float]): Beta parameters for Adam
                - OPTIMIZER.eps (float): Constant for numerical stability
                - OPTIMIZER.weight_decay (float): Weight decay
            - Other configurations
                - OPTIMIZER.amsgrad (bool): Whether to use AMSGrad

        Main steps:
            - Read configuration
            - Validate parameters
            - Pass default parameters to the parent class

        """
        
        # ---Read configuration---
        # Read default optimizer parameters
        lr = float(cfg.OPTIMIZER.LR)
        betas = tuple(map(float, cfg.OPTIMIZER.betas))
        eps = float(cfg.OPTIMIZER.eps)
        weight_decay = float(cfg.OPTIMIZER.weight_decay)

        # Related settings
        amsgrad = cfg.OPTIMIZER.get("amsgrad", False) # Whether to use AMSGrad

        # ---Validate parameters---
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        
        # ---Pass default parameters to the parent class---
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            amsgrad=amsgrad  # Whether to use AMSGrad
        )
