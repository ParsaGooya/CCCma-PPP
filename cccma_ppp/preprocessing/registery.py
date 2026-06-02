class Registery:
    """
    Simple registry for mapping names to classes and instantiating them.
    """

    def __init__(self):
        """
        Initialize an empty registry.

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
            Name used to register the class.

        Returns
        -------
        callable
            Decorator that registers the class.
        """

        def decorator(cls):
            self._modules[name] = cls
            return cls

        return decorator

    def get(self, name, config=None):
        """
        Retrieve and instantiate a registered class.

        Parameters
        ----------
        name : str
            Name of the registered class.
        config : dict, optional
            Keyword arguments for class initialization.

        Returns
        -------
        object
            Instantiated class or class reference if no config is provided.

        Raises
        ------
        ValueError
            If the requested name is not registered.
        TypeError
            If configuration is incompatible with class constructor.
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
        List all registered class names.

        Returns
        -------
        list of str
            Available registered names.
        """

        return list(self._modules.keys())
