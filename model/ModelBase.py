import torch.nn as nn


class ModelBase(nn.Module):
    """
    Interface class for models.
    Inherits from torch.nn.Module and provides a common structure for models.

    Subclasses need to implement the following methods:
        - __init__(): Initialization method
        - forward(): Forward propagation
        - (Optional) build_model(): Build the model (e.g., load a pre-trained model)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
    
    def forward(self, x, return_feature=False):
        """
        Forward propagation.
        Args:
            x (torch.Tensor): Input data [batch, ...]
            return_feature (bool): Whether to return features
        Returns:
            torch.Tensor: Output results [batch, num_classes]
            (Optional) torch.Tensor: Features [batch, ...]
        """
        raise NotImplementedError
    
    def build_model(self):
        """Build the model. (Optional)"""
        pass