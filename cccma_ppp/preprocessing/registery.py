import inspect
from typing import Any


class Registery:
    """
    Registry for named classes.

    Classes can be registered under unique string identifiers and later
    instantiated from optional configuration dictionaries.
    """

    def __init__(self):
        """
        Initialize an empty class registry.
        """
        self._modules = {}

    def register(self, name: str):
        """
        Create a decorator that registers a class under a unique name.

        Parameters
        ----------
        name : str
            Name used to identify the registered class.

        Returns
        -------
        Callable
            Decorator that registers a class and returns it unchanged.

        Raises
        ------
        TypeError
            If ``name`` is not a string.
        ValueError
            If ``name`` is empty or has already been registered.
        """
        if not isinstance(name, str):
            raise TypeError(f"Name must be string, got {type(name)}.")

        if not name:
            raise ValueError("Name cannot be empty string.")

        if name in self._modules:
            raise ValueError(
                f"Name {name!r} already registered. Available: {self.available()}"
            )

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

            Raises
            ------
            TypeError
                If ``cls`` is not a class.
            """
            if not inspect.isclass(cls):
                raise TypeError(
                    f"Can only register classesGot {type(cls).__name__} for {name!r}."
                )

            self._modules[name] = cls
            return cls

        return decorator

    def get(self, name: str, config: dict | None = None) -> Any:
        """
        Instantiate a class registered under a name.

        When ``config`` is provided, its entries are passed as keyword arguments to
        the registered class. Without a configuration, the class is constructed
        without arguments.

        Parameters
        ----------
        name : str
            Name identifying the registered class.
        config : dict | None
            Keyword arguments used to construct the registered class.

        Returns
        -------
        Any
            Instance of the registered class.

        Raises
        ------
        ValueError
            If ``name`` is not registered.
        TypeError
            If ``config`` is neither a dictionary nor ``None``.
        """
        if name not in self._modules:
            raise ValueError(f"{name} not registered. should be in {self.available()}")

        cls = self._modules[name]

        if config is None:
            return cls()

        if not isinstance(config, dict):
            raise TypeError(
                f"Config must be dict or None, got {type(config).__name__}. "
                f"For class {cls.__name__}: {config}"
            )

        return cls(**(config))

    def available(self):
        """
        Return the names of all registered classes.

        Returns
        -------
        list[str]
            Names currently available in the registry.
        """
        return list(self._modules.keys())
