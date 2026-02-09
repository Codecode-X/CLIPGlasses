from utils import Registry, check_availability

WARMUP_REGISTRY = Registry("WARMUP")

def build_warmup(cfg, successor):
    """
    Build the corresponding warmup scheduler based on the name in the configuration (cfg.LR_SCHEDULER.WARMUP.NAME).
    
    Args:
        - cfg (CfgNode): Configuration.
    
    Returns:
        - Instantiated warmup scheduler object.
    
    Main steps:
    1. Check if the warmup scheduler is registered.
    2. Instantiate the warmup scheduler.
    3. Return the warmup scheduler object.
    """
    avai_warmups = WARMUP_REGISTRY.registered_names() # Get all registered warmup schedulers
    check_availability(cfg.LR_SCHEDULER.WARMUP.NAME, avai_warmups) # Check if the warmup scheduler is registered
    if cfg.VERBOSE: # Whether to output information
        print("Loading warmup: {}".format(cfg.LR_SCHEDULER.WARMUP.NAME))

    # Instantiate the warmup scheduler
    warmup = WARMUP_REGISTRY.get(cfg.LR_SCHEDULER.WARMUP.NAME)(cfg, successor)
    return warmup