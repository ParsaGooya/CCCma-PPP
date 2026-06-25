from __future__ import annotations
class Registery:
    """
    Registry for mapping string names to classes.
    """

    def __init__(self):
        """
        Initialize empty registry.

        Returns
        -------
        None
        """

        self._modules = {}

    def register(self, name):
        """
        Register a class under a given name.

        Parameters
        ----------
        name : str
            Identifier used to register the class.

        Returns
        -------
        Callable
            Decorator that registers the class.
        """

        def decorator(cls):
            self._modules[name] = cls
            return cls

        return decorator

    def get(self, name, config=None):
        """
        Retrieve and optionally instantiate a registered class.

        Parameters
        ----------
        name : str
            Name of the registered class.
        config : dict or None, optional
            Initialization arguments.

        Returns
        -------
        object
            Instantiated object if config is provided,
            otherwise the class itself.

        Raises
        ------
        ValueError
            If the name is not registered.
        """

        if name not in self._modules:
            raise ValueError(f"{name} not registered. should be in {self.available()}")

        cls = self._modules[name]
        if config is not None:
            return cls(**(config))
        else:
            return cls

    def available(self):
        """
        Available registered names.

        Returns
        -------
        list of str
            Registered identifiers.
        """

        return list(self._modules.keys())
