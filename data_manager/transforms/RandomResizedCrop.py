from data_manager.transforms import TRANSFORM_REGISTRY
from .base_class.TransformBase import TransformBase
from torchvision.transforms import RandomResizedCrop as _RandomResizedCrop
from torchvision.transforms import InterpolationMode

@TRANSFORM_REGISTRY.register()
class RandomResizedCrop(TransformBase):

    def __init__(self, cfg):
        self.target_size = cfg.INPUT.SIZE
        self.s_ = cfg.INPUT.RandomResizedCrop.scale
        interp_mode = getattr(InterpolationMode, cfg.INPUT.INTERPOLATION.upper(), InterpolationMode.BILINEAR)
        
        assert isinstance(self.target_size, int), "cfg.INPUT.SIZE must be a single integer"
        
        self.transform = _RandomResizedCrop(self.target_size, scale=self.s_, interpolation=interp_mode)

    def __call__(self, img):
        return self.transform(img)
