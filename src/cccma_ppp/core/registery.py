import dacite
import dataclasses


class Registery:
    """
    Registry for mapping names to classes and instantiating them.
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
        config : dict or object, optional
            Configuration used to initialize the class.

        Returns
        -------
        object
            Instantiated class object.

        Raises
        ------
        ValueError
            If the requested name is not registered.
        TypeError
            If configuration is incompatible with class initialization.
        """

        if name not in self._modules:
            raise ValueError(f"{name} not registered. should be in {self.available()}")

        cls = self._modules[name]

        if isinstance(config, dict):
            if dataclasses.is_dataclass(cls):
                return dacite.from_dict(
                    data_class=cls,
                    data=config or {},
                    config=dacite.Config(strict=True),
                )
            else:
                return cls(**(config or {}))

        else:
            return cls(config)

    def available(self):
        """
        List all registered class names.

        Returns
        -------
        list of str
            Names of registered classes.
        """

        return list(self._modules.keys())
