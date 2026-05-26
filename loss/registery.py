import dacite
import dataclasses

class Registery:
    def __init__(self):
        self._modules = {}

    def register(self, name):
        def decorator(cls):
            self._modules[name] = cls
            return cls
        return decorator

    def get(self, name, config=None):
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
        return list(self._modules.keys())