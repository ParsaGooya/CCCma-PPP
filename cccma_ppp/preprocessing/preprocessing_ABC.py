import abc


class PreprocessModuleABC(abc.ABC):
    @abc.abstractmethod
    def fit(self, data):
        ...
    @abc.abstractmethod
    def transform(self, data, **kwargs):
        ...
    @abc.abstractmethod
    def inverse_transform(self, data, **kwargs):
        ...