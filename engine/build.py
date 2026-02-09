from utils import Registry, check_availability

TRAINER_REGISTRY = Registry("TRAINER")


def build_trainer(cfg):
    """Build the corresponding trainer based on the trainer name in the configuration (cfg.TRAINER.NAME)."""
    avai_trainers = TRAINER_REGISTRY.registered_names() # Get all registered trainers
    check_availability(cfg.TRAINER.NAME, avai_trainers) # Check if the trainer with the specified name exists
    if cfg.VERBOSE: # Whether to output information
        print("Loading trainer: {}".format(cfg.TRAINER.NAME))
    return TRAINER_REGISTRY.get(cfg.TRAINER.NAME)(cfg) # Return the trainer object with the specified name
