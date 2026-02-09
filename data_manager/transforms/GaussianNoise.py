from data_manager.transforms import TRANSFORM_REGISTRY
from .base_class.TransformBase import TransformBase
import random
import torch

@TRANSFORM_REGISTRY.register()
class GaussianNoise(TransformBase):
    """
    Add Gaussian noise to the input image.
    
    Attributes:
        - mean (float): Mean of the Gaussian noise | Default is 0.
        - std (float): Standard deviation of the Gaussian noise | Default is 0.15.
        - p (float): Probability of applying Gaussian noise | Default is 0.5.
    
    Main functionality:
        - Add Gaussian noise to the input image.
    
    Main steps:
        1. Decide whether to add Gaussian noise based on probability p.
        2. If adding, generate Gaussian noise with the same size as the image.
        3. Add the generated Gaussian noise to the input image.
    """
    def __init__(self, cfg):
        self.mean = cfg.INPUT.GaussianNoise.mean \
            if hasattr(cfg.INPUT.GaussianNoise, 'mean') else 0
        self.std = cfg.INPUT.GaussianNoise.std \
            if hasattr(cfg.INPUT.GaussianNoise, 'std') else 0.15
        self.p = cfg.INPUT.GaussianNoise.p \
            if hasattr(cfg.INPUT.GaussianNoise, 'p') else 0.5

    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return img
        noise = torch.randn(img.size()) * self.std + self.mean
        return img + noise