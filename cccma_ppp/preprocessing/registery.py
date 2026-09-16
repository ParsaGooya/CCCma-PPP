import inspect
from typing import Any
class Registery:
    """
    Document this class.
    """

    def __init__(self):
        """
        Document this function.
        """
        self._modules = {}

    def register(self, name: str):
        """
        Document this function.

        Parameters
        ----------
        name : Any
            Description not yet provided.

        Returns
        -------
        Any
            Description not yet provided.
        """
        if not isinstance(name, str):
            raise TypeError(f"Name must be string, got {type(name)}.")

        if not name:
            raise ValueError("Name cannot be empty string.")

        if name in self._modules:
            raise ValueError(
                f"Name {name!r} already registered. "
                f"Available: {self.available()}"
            )

        def decorator(cls):
            """
            Document this function.

            Returns
            -------
            Any
                Description not yet provided.
            """
            if not inspect.isclass(cls):
                raise TypeError(
                    f"Can only register classes"
                    f"Got {type(cls).__name__} for {name!r}."
                )

            self._modules[name] = cls
            return cls

        return decorator

    def get(self, name: str, config: dict | None = None) -> Any:
        """
        Document this function.

        Parameters
        ----------
        name : Any
            Description not yet provided.
        config : Any
            Description not yet provided.

        Returns
        -------
        Any
            Description not yet provided.

        Raises
        ------
        ValueError
            Description not yet provided.
        """
        if name.lower() not in self._modules:
            raise ValueError(f"{name} not registered. should be in {self.available()}")

        cls = self._modules[name.lower()]

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
        Document this function.

        Returns
        -------
        Any
            Description not yet provided.
        """
        return list(self._modules.keys())
