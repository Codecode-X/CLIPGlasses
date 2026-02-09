from data_manager.transforms import TRANSFORM_REGISTRY
from .base_class.TransformBase import TransformBase
import random
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode


INTERPOLATION_MODES = {
    "bilinear": InterpolationMode.BILINEAR,  # Bilinear interpolation
    "bicubic": InterpolationMode.BICUBIC,  # Bicubic interpolation
    "nearest": InterpolationMode.NEAREST,  # Nearest neighbor interpolation
}

@TRANSFORM_REGISTRY.register()
class Random2DTranslation(TransformBase):
    """
    Adjusts the given image from (height, width) to (height*1.125, width*1.125), 
    then performs random cropping.
    Attributes:
        - height (int): Target image height.
        - width (int): Target image width.
        - p (float, optional): Probability of performing this operation | Default is 0.5.
        - interpolation (int, optional): Desired interpolation method | Default is
          ``torchvision.transforms.functional.InterpolationMode.BILINEAR``.
    Main functionality:
        - Performs random translation and cropping on the input image.
    Main steps:
        1. Decide whether to perform the operation based on probability p.
        2. If not performing the operation, directly resize the image to (height, width).
        3. If performing the operation, resize the image to (height*1.125, width*1.125).
        4. Randomly select a region on the resized image for cropping, 
           making its size (height, width).
    """
    def __init__(self, cfg):
        self.height, self.width = cfg.INPUT.SIZE, cfg.INPUT.SIZE
        self.p = cfg.INPUT.Random2DTranslation.p \
            if hasattr(cfg.INPUT.Random2DTranslation, 'p') else 0.5
        self.interpolation = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION] \
            if hasattr(cfg.INPUT, 'INTERPOLATION') else InterpolationMode.BILINEAR

    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return F.resize(
                img=img,
                size=[self.height, self.width],
                interpolation=self.interpolation
            )

        new_width = int(round(self.width * 1.125))
        new_height = int(round(self.height * 1.125))
        resized_img = F.resize(
            img=img,
            size=[new_height, new_width],
            interpolation=self.interpolation
        )
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = F.crop(
            img=resized_img,
            top=y1,
            left=x1,
            height=self.height,
            width=self.width
        )

        return croped_img