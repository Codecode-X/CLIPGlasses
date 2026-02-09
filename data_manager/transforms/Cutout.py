from data_manager.transforms import TRANSFORM_REGISTRY
from .base_class.TransformBase import TransformBase
import numpy as np
import torch

@TRANSFORM_REGISTRY.register()
class Cutout(TransformBase):
    """    
    Randomly masks one or more patches from the image.

    Attributes:
        - n_holes (int, optional): Number of patches to mask per image | Default is 1.
        - length (int, optional): Side length of each square patch (in pixels) | Default is 16.
    Main functionality:
        - Applies random masking patches to the input image.
    Main steps:
        1. Get the height and width of the image.
        2. Create a mask of the same size as the image, initialized to 1.
        3. Randomly select the center position of the patch and calculate the patch boundaries.
        4. Set the values of the mask corresponding to the patch location to 0.
        5. Expand the mask to the same dimensions as the image and multiply it with the image.
    """
    def __init__(self, cfg):
        self.n_holes = cfg.INPUT.Cutout.n_holes \
            if hasattr(cfg.INPUT.Cutout, 'n_holes') else 1
        self.length = cfg.INPUT.Cutout.length \
            if hasattr(cfg.INPUT.Cutout, 'length') else 16

    def __call__(self, img):
        """
        Args:
            img (Tensor): tensor image of size (C, H, W).

        Returns:
            Tensor: image with n_holes of dimension
                length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        return img * mask
