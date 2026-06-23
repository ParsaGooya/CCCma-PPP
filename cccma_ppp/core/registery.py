import dacite
import dataclasses


class Registery:
    """
    Registry for mapping string identifiers to classes and instantiating them.

    Methods
    -------
    register(name)
        Register a class under a given name.
    get(name, config=None)
        Retrieve and instantiate a registered class.
    available()
        Return all registered names.
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
        If a name is registered more than once, the latest registration
        overwrites the previous one.
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
        config : dict or object or None, optional
            Configuration used to construct the class:
            - If dict and class is a dataclass, instantiates via `dacite.from_dict`.
            - If dict and class is not a dataclass, calls `cls(**config)`.
            - Otherwise, passes config directly to constructor as `cls(config)`.

        Returns
        -------
        object
            Instantiated object corresponding to the given name.

        Raises
        ------
        ValueError
            If the requested name is not registered.
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
        Return all registered class names.

        Returns
        -------
        list of str
            Registered identifiers.
        """

        return list(self._modules.keys())
