from data_manager.transforms import TRANSFORM_REGISTRY
from .base_class.TransformBase import TransformBase
from torchvision.transforms import RandomHorizontalFlip as _RandomHorizontalFlip

@TRANSFORM_REGISTRY.register()
class RandomFlip(TransformBase):

    def __init__(self, cfg):
        self.transform = _RandomHorizontalFlip()

    def __call__(self, img):
        return self.transform(img)
