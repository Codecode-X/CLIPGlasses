"""
Modified from https://github.com/facebookresearch/fvcore
"""
__all__ = ["Registry"]


class Registry:
    """
    Registry class.
    Provides a registry that maps (name -> object) to support custom modules.

    To create a registry (e.g., a backbone network registry):
        BACKBONE_REGISTRY = Registry('BACKBONE')
    To register an object:
        @BACKBONE_REGISTRY.register()  # Decorator
        class MyBackbone(nn.Module):
    Or:
        BACKBONE_REGISTRY.register(MyBackbone)  # Function call
    """

    def __init__(self, name):
        # Initialize the registry, where `name` is the name of the registry
        self._name = name
        # _obj_map: A dictionary storing the mapping from names to objects
        self._obj_map = dict()

    def _do_register(self, name, obj, force=False):
        """ Register an object 
        If the name already exists in _obj_map and force is False, raise an exception.
        """
        if name in self._obj_map and not force:
            raise KeyError(
                'An object named "{}" was already '
                'registered in "{}" registry'.format(name, self._name)
            )
        # Add the name and object to _obj_map
        self._obj_map[name] = obj

    def register(self, obj=None, force=False):
        """ Registration decorator

        Args:
            - obj: The object to be registered (instantiated class or function)
            - force: If True, force registration even if the name already exists in _obj_map
        Usage:
            - obj is None: Used as a decorator
            - obj is not None: Used as a function call
        
        """
        # If obj is None, use as a decorator
        if obj is None:
            def wrapper(fn_or_class):
                # Get the name of the function or class
                name = fn_or_class.__name__
                # Register the name and object
                self._do_register(name, fn_or_class, force=force)
                return fn_or_class
            return wrapper
        else: # If obj is not None, use as a function call
            # Get the name of the object
            name = obj.__name__
            # Register the name and object
            self._do_register(name, obj, force=force)

    def get(self, name):
        """ Retrieve a registered object 
        Args:
            name (str): The name of the object
        Returns:
            object: The registered object corresponding to the name
        """
        # If the name is not in _obj_map, raise a KeyError exception
        if name not in self._obj_map: # _obj_map: A dictionary storing the mapping from names to objects
            raise KeyError(
                'Object name "{}" does not exist '
                'in "{}" registry'.format(name, self._name)
            )
        # Return the registered object corresponding to the name
        return self._obj_map[name]

    def registered_names(self):
        """Return a list of all registered names
        Returns:
            list: A list of all registered names (strings)
        """
        return list(self._obj_map.keys())
