import math
import torch
from .OptimizerBase import OptimizerBase
from .build import OPTIMIZER_REGISTRY

@OPTIMIZER_REGISTRY.register()
class RAdam(OptimizerBase):
    """ RAdam Optimizer """
    def __init__(self, cfg, params=None):
        """ Initialization method 

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
                - OPTIMIZER.DEGENERATED_TO_SGD (bool): Whether to degenerate RAdam to SGD

        Main steps:
            - Initialize cache self.buffer = [[None, None, None] for _ in range(10)]
            - Read configuration
            - Validate parameters
            - Pass default optimizer parameters to the parent class
        """
        # ----Initialize cache-----
        self.buffer = [[None, None, None] for _ in range(10)]

        # ----Read configuration-----

        # Read default optimizer parameters
        lr = float(cfg.OPTIMIZER.LR)
        betas = list(map(float, cfg.OPTIMIZER.betas))
        eps = float(cfg.OPTIMIZER.eps)
        weight_decay = float(cfg.OPTIMIZER.weight_decay)

        # Related settings
        self.degenerated_to_sgd = bool(cfg.OPTIMIZER.degenerated_to_sgd)  # Whether to degenerate RAdam to SGD

        # ----Validate parameters-----
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        # ---Pass default optimizer parameters to the parent class---
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)  # Default values for optimizer options
        super(RAdam, self).__init__(params, defaults)


    def step(self, closure=None):
        """ 
        Perform a single optimization step 

        Args:
            - closure (callable, optional): A closure that reevaluates the model and returns the loss

        Returns:
            - loss: Computed loss value
        """
        loss = None
        if closure is not None:
            loss = closure()  # Compute loss

        # Iterate over parameter groups (base layer parameters and new layer parameters)
        # In some complex training scenarios, there may be more parameter groups
        for group in self.param_groups:
            for p in group["params"]:  # Iterate over parameters in the group
                
                if p.grad is None:
                    continue  # Skip parameters without gradients
                
                grad = p.grad.data.float()  # Get gradient and convert to float type
                if grad.is_sparse:  # Sparse gradients are not supported
                    raise RuntimeError("RAdam does not support sparse gradients")  

                # Retrieve parameter data and state
                p_data_fp32 = p.data.float()  # Get parameter data (weights or biases of the neural network)
                state = self.state[p]  # Get parameter state (auxiliary information maintained by the optimizer)

                # Initialize parameter state
                if len(state) == 0:
                    state["step"] = 0  # Initialize step counter
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)  # Initialize first moment estimate
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)  # Initialize second moment estimate
                else:
                    state["exp_avg"] = state["exp_avg"].type_as(p_data_fp32)  # Convert first moment estimate data type
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(p_data_fp32)  # Convert second moment estimate data type

                # Retrieve optimizer parameters
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                # Update first and second moment estimates
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)  # Update second moment estimate
                exp_avg.mul_(beta1).add_(1 - beta1, grad)  # Update first moment estimate

                # Update step counter
                state["step"] += 1  

                # Retrieve cache
                buffered = self.buffer[int(state["step"] % 10)]  # Get cache
                if state["step"] == buffered[0]:  # If current step matches cached step
                    N_sma, step_size = buffered[1], buffered[2]  # Retrieve N_sma and step_size from cache
                else:  # If current step does not match cached step, recalculate N_sma and step_size
                    buffered[0] = state["step"]
                    beta2_t = beta2**state["step"]
                    
                    # Calculate N_sma
                    N_sma_max = 2 / (1-beta2) - 1
                    N_sma = N_sma_max - 2 * state["step"] * beta2_t / (1-beta2_t)
                    buffered[1] = N_sma
                    
                    # Calculate step_size
                    if N_sma >= 5:
                        step_size = math.sqrt(
                            (1-beta2_t) * (N_sma-4) / (N_sma_max-4) *
                            (N_sma-2) / N_sma * N_sma_max / (N_sma_max-2)
                        ) / (1 - beta1**state["step"])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1**state["step"])
                    else:
                        step_size = -1
                    buffered[2] = step_size
                
                if N_sma >= 5:
                    if group["weight_decay"] != 0:
                        p_data_fp32.add_(-group["weight_decay"] * group["lr"], p_data_fp32)  # Weight decay
                    denom = exp_avg_sq.sqrt().add_(group["eps"])  # Calculate denominator
                    p_data_fp32.addcdiv_(-step_size * group["lr"], exp_avg, denom)  # Update parameters
                    p.data.copy_(p_data_fp32)  # Copy updated parameters back to original
                elif step_size > 0:
                    if group["weight_decay"] != 0:
                        p_data_fp32.add_(-group["weight_decay"] * group["lr"], p_data_fp32)  # Weight decay
                    p_data_fp32.add_(-step_size * group["lr"], exp_avg)  # Update parameters
                    p.data.copy_(p_data_fp32)  # Copy updated parameters back to original

        return loss  # Return loss value