from utils import Registry, check_availability

SAMPLER_REGISTRY = Registry("SAMPLER")


def build_train_sampler(cfg, data_source):
    """ Build the corresponding training sampler according to the name of the training sampler in the configuration (cfg.SAMPLER.TRAIN_SP). """
    avai_samplers = SAMPLER_REGISTRY.registered_names()
    check_availability(cfg.SAMPLER.TRAIN_SP , avai_samplers)
    if cfg.VERBOSE:
        print("Loading sampler: {}".format(cfg.SAMPLER.TRAIN_SP))
    return SAMPLER_REGISTRY.get(cfg.SAMPLER.TRAIN_SP)(cfg, data_source)


def build_test_sampler(cfg, data_source):
    """ Build the corresponding test sampler according to the name of the test sampler in the configuration (cfg.SAMPLER.TEST_SP). """
    avai_samplers = SAMPLER_REGISTRY.registered_names()
    check_availability(cfg.SAMPLER.TEST_SP, avai_samplers)
    if cfg.VERBOSE:
        print("Loading sampler: {}".format(cfg.SAMPLER.TEST_SP))
    return SAMPLER_REGISTRY.get(cfg.SAMPLER.TEST_SP)(cfg, data_source)