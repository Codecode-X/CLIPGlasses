from data_manager.transforms import TRANSFORM_REGISTRY
from .base_class.TransformBase import TransformBase
from utils import standard_image_transform

@TRANSFORM_REGISTRY.register()
class StandardNoAugTransform(TransformBase):
    """
    Standardized image transformation pipeline without augmentation
    (resize + center_crop + to_rgb + to_tensor + normalize)

    Main steps:
        - Scale the image to the specified size while maintaining the aspect ratio, then center crop to the target size
        - Convert the image to RGB format
    """

    def __init__(self, cfg):
        self.target_size = cfg.INPUT.SIZE
        assert isinstance(self.target_size, int), "cfg.INPUT.SIZE must be a single integer"
        
        # Transformation method
        self.transform = standard_image_transform(self.target_size, 
                                                  interp_mode=cfg.INPUT.INTERPOLATION)

    def __call__(self, img):
        return self.transform(img)
