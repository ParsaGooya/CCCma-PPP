import dacite
import dataclasses


class Registery:
    """
    Document this class.
    """
    def __init__(self):
        """
        Document this function.
        """
        self._modules = {}

    def register(self, name):
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
        def decorator(cls):
            """
            Document this function.
            
            Returns
            -------
            Any
                Description not yet provided.
            """
            setattr(cls, "_type", name)
            self._modules[name] = cls
            return cls

        return decorator

    def get(self, name, config=None):
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
        Document this function.
        
        Returns
        -------
        Any
            Description not yet provided.
        """
        return list(self._modules.keys())
