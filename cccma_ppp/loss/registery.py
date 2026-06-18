import dacite
import dataclasses


class Registery:
    """
    Generic registry for mapping string identifiers to classes and instantiating them.

    Methods
    -------
    register(name)
        Register a class under a given name.
    get(name, config=None)
        Retrieve and instantiate a registered class.
    available()
        Return the list of registered names.
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
            Configuration used to initialize the class:
            - If dict and class is a dataclass, uses dacite for construction.
            - If dict and class is not a dataclass, calls class(**config).
            - Otherwise, passes config directly to constructor.

        Returns
        -------
        object
            Instantiated class corresponding to the given name.

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
        Return the list of registered names.

        Returns
        -------
        list of str
            Names of all registered classes.
        """
        return list(self._modules.keys())
