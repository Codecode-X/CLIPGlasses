import torch.nn as nn
from .build import MODEL_REGISTRY
from .ModelBase import ModelBase

@MODEL_REGISTRY.register()
class SimpleNet(ModelBase):
    """A simple neural network consisting of a CNN backbone and an optional head (e.g., an MLP for classification)."""

    def __init__(self, cfg, model_cfg, num_classes, **kwargs):
        self.num_classes = num_classes
        super().__init__()
        # Build the backbone network
        self.net = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            # Global max pooling
            nn.AdaptiveMaxPool2d((1, 1)), # Shape becomes [batch, 128, 1, 1]
            # Fully connected layer for 10 classes
            nn.Flatten(), # Shape becomes [batch, 128]
            nn.Linear(128, num_classes) # Output [batch, num_classes]
        )

    def forward(self, x, return_feature=False):
        """Forward pass."""
        # Forward pass through the backbone network
        y = self.net(x)
        # Return the output
        return y
    
    def build_model(self):
        """Build the model."""
        super().build_model() # Directly call the parent class method