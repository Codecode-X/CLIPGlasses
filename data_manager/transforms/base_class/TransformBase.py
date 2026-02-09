from data_manager.transforms import TRANSFORM_REGISTRY

@TRANSFORM_REGISTRY.register()
class TransformBase:
    """
    Base class for data transformation
    This class only implements an identity transformation, which does not apply any changes to the data.
    You can implement custom data transformations by inheriting from this class.

    Subclasses need to implement the following methods:
        - __init__(): Initialization method
        - __call__(): Method to apply the transformation to the data
    """

    def __init__(self, cfg):
        self.transform = lambda x: x # Identity transformation

    def __call__(self, img):
        return self.transform(img)