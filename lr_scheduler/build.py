from utils import Registry, check_availability
from .warmup import build_warmup

LRSCHEDULER_REGISTRY = Registry("LRSCHEDULER")

def build_lr_scheduler(cfg, optimizer):
    """Build the learning rate scheduler based on the name specified in the configuration (cfg.LR_SCHEDULER.NAME).
    
    Args:
        - cfg (CfgNode): Configuration.
        - optimizer (Optimizer): Optimizer.
    
    Returns:
        - The instantiated learning rate scheduler object.
    
    Main steps:
    1. Check if the learning rate scheduler is registered.
    2. Instantiate the learning rate scheduler.
    3. If the learning rate scheduler has a warmup scheduler, build the warmup scheduler and apply it to the learning rate scheduler.
    4. Return the learning rate scheduler object.
    """
    avai_lr_schedulers = LRSCHEDULER_REGISTRY.registered_names() # Get all registered learning rate schedulers
    lr_scheduler_name = cfg.LR_SCHEDULER.NAME # Get the learning rate scheduler name from the configuration

    check_availability(lr_scheduler_name, avai_lr_schedulers) # Check if the learning rate scheduler is registered
    if cfg.VERBOSE: # Whether to output information
        print("Loading lr_scheduler: {}".format(lr_scheduler_name))

    # Instantiate the learning rate scheduler
    lr_scheduler = LRSCHEDULER_REGISTRY.get(lr_scheduler_name)(cfg, optimizer)

    # If the learning rate scheduler has a warmup scheduler
    if hasattr(cfg.LR_SCHEDULER, "WARMUP") and cfg.LR_SCHEDULER.WARMUP is not None:
        lr_scheduler = build_warmup(cfg, lr_scheduler) # Build the warmup scheduler and apply it to the learning rate scheduler

    return lr_scheduler