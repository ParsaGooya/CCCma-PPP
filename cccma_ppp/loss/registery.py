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
        config : dict or object or None, optional
            Configuration used for instantiation.

            - If dict and class is a dataclass, uses `dacite.from_dict`.
            - If dict and class is not a dataclass, uses `cls(**config)`.
            - Otherwise, instantiates using `cls(config)`.

        Returns
        -------
        object
            Instantiated object.

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
        Get available class names.

        Returns
        -------
        list of str
            Registered identifiers.
        """

        return list(self._modules.keys())
