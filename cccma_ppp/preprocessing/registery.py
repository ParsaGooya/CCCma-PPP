class Registery:

    """
    Simple registry for mapping string identifiers to classes and instantiating them.

    This registry enables dynamic construction of modules (e.g., preprocessing,
    loss functions, models) by associating string names with class definitions.

    Methods
    -------
    register(name)
        Register a class under a given name.
    get(name, config=None)
        Retrieve and optionally instantiate a registered class.
    available()
        Return the list of registered names.
    """

    def __init__(self):
        self._modules = {}

    def register(self, name):
        """
        Register a class under a specified name.

        Parameters
        ----------
        name : str
            Identifier used to register the class.

        Returns
        -------
        Callable
            Decorator that registers the class in the registry.

        Notes
        -----
        The decorated class will be stored in the registry under the provided name.
        If the same name is used multiple times, the latest registration overwrites
        the previous one.
        """
        
        def decorator(cls):
            self._modules[name] = cls
            return cls

        return decorator

    def get(self, name, config=None):
        if name not in self._modules:
            raise ValueError(f"{name} not registered. should be in {self.available()}")

        cls = self._modules[name]
        if config is not None:
            return cls(**(config))
        else:
            return cls

    def available(self):
        return list(self._modules.keys())
