import dacite
import dataclasses


class Registery:
    """
    Registry for named classes.

    Registered classes can be retrieved and instantiated by name using optional
    configuration data.
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
            Decorator that registers and returns the supplied class.
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
                Registered class, returned unchanged.
            """
            self._modules[name] = cls
            return cls

        return decorator

    def get(self, name, config=None):
        """
        Retrieve and instantiate a registered class.

        Dictionary configurations are expanded into constructor arguments. Dataclass
        types are constructed using strict field matching, while other dictionary
        configurations are passed as keyword arguments. Non-dictionary configuration
        values are passed as a single positional argument.

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
        list[str]
            Names currently available in the registry.
        """
        return list(self._modules.keys())
