from utils import Registry, check_availability

EVALUATOR_REGISTRY = Registry("EVALUATOR")


def build_evaluator(cfg, dm):
    """
    Build the corresponding evaluator based on the evaluator name in the configuration (cfg.EVALUATOR.NAME).

    Args:
        - cfg (CfgNode): Configuration.
        - dm (Dataset): Dataset manager.
    Returns:
        - evaluator object.
    """
    avai_evaluators = EVALUATOR_REGISTRY.registered_names() # Get all registered evaluators
    check_availability(cfg.EVALUATOR.NAME, avai_evaluators) # Check if the evaluator with the specified name exists
    if cfg.VERBOSE: # Whether to output information
        print("Loading evaluator: {}".format(cfg.EVALUATOR.NAME))
    return EVALUATOR_REGISTRY.get(cfg.EVALUATOR.NAME)(cfg, dm) # Return the evaluator object with the specified name