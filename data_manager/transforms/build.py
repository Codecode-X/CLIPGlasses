from utils import Registry
from torchvision.transforms import (Resize, Compose, ToTensor, Normalize)
from torchvision.transforms.functional import InterpolationMode


TRANSFORM_REGISTRY = Registry("TRANSFORM")


def build_train_transform(cfg):
    """
    Build training data augmentation.
    Main steps:
    1. Check whether the data augmentation methods in the configuration exist/are valid
    2. Iterate through the selected data augmentation methods in the configuration, add them to the augmentation list, and Compose
        - Ensure the image size matches the model input size; if there is no cropping operation later, use Resize
        - Iterate through the data augmentation methods selected before ToTensor in the configuration, add them to the augmentation list
        - Add ToTensor() transformation
        - Iterate through the data augmentation methods selected after ToTensor in the configuration, add them to the augmentation list
        - Add normalization (optional)
    3. Return the augmentation list
    
    Note: Some augmentations require configuration parameters, which can be achieved through custom augmentation classes (by reading cfg in the class)
    """
    print("Building training data augmentation.....")
    
    avai_transforms = TRANSFORM_REGISTRY.registered_names()
    before_choices = cfg.INPUT.BEFORE_TOTENSOR_TRANSFORMS 
    after_choices = cfg.INPUT.AFTER_TOTENSOR_TRANSFORMS 
    _check_cfg(before_choices, avai_transforms)
    _check_cfg(after_choices, avai_transforms)

    
    tfm_train = [] 

    
    for choice in before_choices:
        if cfg.VERBOSE: print(f"Training data augmentation before ToTensor: {choice}")
        tfm_train.append(TRANSFORM_REGISTRY.get(choice)(cfg)) 
    tfm_train.append(ToTensor())
    for choice in after_choices:
        if cfg.VERBOSE: print(f"Training data augmentation after ToTensor: {choice}")
        tfm_train.append(TRANSFORM_REGISTRY.get(choice)(cfg))
    
    if cfg.INPUT.NORMALIZE:
        tfm_train.append(Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD))

    tfm_train = Compose(tfm_train)
    
    
    return tfm_train


def build_test_transform(cfg):
    """
    Build testing data augmentation.

    Main steps:
    1. Check configuration information
    2. Build the testing data augmentation list
        - Use standard image preprocessing without augmentation (resize + RGB)
        - Add ToTensor() transformation
        - Add normalization (optional)
    3. Return the testing data augmentation list
    """
    print("Building testing data augmentation.....")
    
    
    standardNoAugTransform = TRANSFORM_REGISTRY.get("StandardNoAugTransform")(cfg) 
    tfm_test = [standardNoAugTransform, ToTensor()] 
    if cfg.INPUT.NORMALIZE:
        tfm_test.append(Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)) 
    tfm_test = Compose(tfm_test)

    if cfg.VERBOSE:  
        print(f"Testing data augmentation:")
        print("  - Standard image preprocessing transformation: resize + RGB + toTensor + normalize")

    
    return tfm_test


def _check_cfg(choices, avai_transforms):
    if len(choices) == 0:
        return True
    for choice in choices:
        assert choice in avai_transforms, f"Augmentation method <{format(choice)}> is not in the available augmentation methods {avai_transforms}"
