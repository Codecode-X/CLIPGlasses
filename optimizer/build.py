from utils import Registry, check_availability
import warnings
import torch.nn as nn


OPTIMIZER_REGISTRY = Registry("OPTIMIZER")

def build_optimizer(model, cfg, param_groups=None):
    """Build optimizer.
    
    Args:
        - model (nn.Module): The model.
        - cfg (CfgNode): Configuration.
        - param_groups (Optional[List[Dict]]): Parameter groups | Default is None.
        
    Returns:
        - Instantiated optimizer object.
        
    Main steps:
        1. Process the model parameter, ensuring the model is an instance of nn.Module and not an instance of nn.DataParallel.
        2. Read configuration:
            - OPTIMIZER.NAME (str): Optimizer name.
            - OPTIMIZER.LR (float): Learning rate.
            - OPTIMIZER.STAGED_LR (bool): Whether to use staged learning rates.
                - OPTIMIZER.NEW_LAYERS (list): List of new layers.
                - OPTIMIZER.BASE_LR_MULT (float): Base layer learning rate scaling factor, usually set to less than 1.
        3. Create model parameter groups to optimize based on staged_lr configuration.
        4. Instantiate the optimizer.
        5. Return the optimizer.
    """
    # ---Process the model parameter, ensuring the model is an instance of nn.Module and not an instance of nn.DataParallel---
    assert isinstance(model, nn.Module), "The model passed to build_optimizer() must be an instance of nn.Module"
    if isinstance(model, nn.DataParallel): model = model.module

    # ---Read configuration---
    optimizer_name = cfg.OPTIMIZER.NAME  # Get optimizer name
    avai_optims = OPTIMIZER_REGISTRY.registered_names()  # Get all registered optimizers
    check_availability(optimizer_name, avai_optims)  # Check if the optimizer with the given name exists
    if cfg.VERBOSE: print("Loading optimizer: {}".format(optimizer_name))
    
    lr = float(cfg.OPTIMIZER.LR)  # Learning rate

    staged_lr = cfg.OPTIMIZER.STAGED_LR if hasattr(cfg.OPTIMIZER, "STAGED_LR") else False  # Whether to use staged learning rates
    if staged_lr: # If using staged learning rates
        if cfg.VERBOSE: print("Using staged_lr for optimizer.")
        base_lr_mult = float(cfg.OPTIMIZER.BASE_LR_MULT)
        new_layers = cfg.OPTIMIZER.NEW_LAYERS 
        if new_layers is None: warnings.warn("new_layers is None (staged_lr is ineffective)")
        if isinstance(new_layers, str): new_layers = [new_layers] # If new_layers is a string, convert it to a list    

    # ---Create model parameter groups to optimize based on staged_lr configuration---
    # If param_groups is provided, use param_groups directly for parameter configuration (staged_lr will be ignored)
    if param_groups is not None and staged_lr:  
        warnings.warn("Since param_groups is provided, staged_lr will be ignored. If you need staged_lr, please bind param_groups manually.")
    else: # If param_groups is not provided, create parameter groups based on staged_lr configuration
        
        if staged_lr: # If using staged learning rates
            base_params = []  # List of base layer parameters
            new_params = []  # List of new layer parameters
            for name, module in model.named_children():  # Iterate through the model's submodules
                if name in new_layers:  # If the submodule is in the new layer list, i.e., the module is a new layer, extract new parameters from the module
                    new_params += [p for p in module.parameters()]  # Add to the new layer parameter list
                else: # If the submodule is not in the new layer list
                    base_params += [p for p in module.parameters()]  # Add to the base layer parameter list
            # Create parameter groups
            param_groups = [{"params": base_params, "lr": lr * base_lr_mult},  # Parameters in the base layer and their learning rate (lr * base_lr_mult)
                            {"params": new_params, "lr": lr}]  # Learning rate for parameters in the new layer (lr)
               
        else: # If not using staged learning rates 
            param_groups = model.parameters()  # Directly use model parameters as parameter groups
                
    # ---Instantiate the optimizer---
    optimizer = OPTIMIZER_REGISTRY.get(optimizer_name)(cfg, param_groups)

    # ---Return the instantiated optimizer object---
    return optimizer