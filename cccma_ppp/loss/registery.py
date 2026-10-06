import dacite
import dataclasses


class Registery:
    """
    Registry for named classes.

    Registered classes can later be retrieved and instantiated from configuration
    data using their associated names.
    """

    def __init__(self):
        """
        Initialize an empty class registry.
        """
        self._modules = {}

    def register(self, name):
        """
        Create a decorator that registers a class under a name.

        Parameters
        ----------
        name : str
            Name used to identify the registered class.

        Returns
        -------
        Callable
            Decorator that registers a class and returns it unchanged.
        """

        def decorator(cls):
            """
            Register a class under the configured name.

            Parameters
            ----------
            cls : type
                Class to register.

            Returns
            -------
            type
                Registered class.
            """
            self._modules[name] = cls
            return cls

        return decorator

    def get(self, name, config=None):
        """
        Retrieve and instantiate a registered class.

        Dictionary configurations are unpacked into the registered class. Dataclass
        types are constructed using strict field matching.

        Parameters
        ----------
        name : str
            Name identifying the registered class.
        config : Any
            Configuration used to construct the registered class.

        Returns
        -------
        Any
            Instance of the registered class.

        Raises
        ------
        ValueError
            If ``name`` is not registered.
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
        Return the names of all registered classes.

        Returns
        -------
        list
            Names currently available in the registry.
        """
        return list(self._modules.keys())
