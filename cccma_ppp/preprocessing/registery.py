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
        if config is not None:
            return cls(**(config ))
        else:
            return cls

    def available(self):
        return list(self._modules.keys())