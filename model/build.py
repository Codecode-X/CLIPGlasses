from utils import Registry, check_availability

MODEL_REGISTRY = Registry("MODEL")


def build_model(cfg):
    """
    Build the corresponding model based on the model name in the configuration (cfg.MODEL.NAME).
    
    Args:
        - cfg (CfgNode): Configuration.
    
    Returns:
        - Instantiated model object.
    
    Main steps:
    1. Check if the model is registered.
    2. Instantiate the model.
    3. Call the model's own build_model() method if necessary.
    4. Return the instantiated model, move it to the device, and set it to evaluation mode by default.
    """
    model_name = cfg.MODEL.NAME  # Get the model name
    avai_models = MODEL_REGISTRY.registered_names()  # Get all registered models
    check_availability(model_name, avai_models)  # Check if the model with the given name exists
    if cfg.VERBOSE:  # Whether to output information
        print("Loading model: {}".format(model_name))

    # Instantiate the model
    try:
        print("Calling the model constructor to build the model...")
        model = MODEL_REGISTRY.get(model_name)(cfg)  # Directly call the model constructor
    except TypeError as e:
        print("Direct call to the model constructor failed, trying the model's build_model method...")
        model_class = MODEL_REGISTRY.get(model_name)  # Get the model class
        if hasattr(model_class, "build_model") and callable(model_class.build_model):
            model = model_class.build_model(cfg)  # Call the model class's static method build_model to construct itself
        else:
            print("The model does not have a build_model method")
            raise e
    # Move to device and set the model to evaluation mode
    device = 'cuda' if cfg.USE_CUDA else 'cpu'
    model.to(device).eval()
    
    return model