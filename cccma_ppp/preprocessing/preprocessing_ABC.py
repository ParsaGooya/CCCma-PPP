import abc


class PreprocessModuleABC(abc.ABC):
    @abc.abstractmethod
    def fit(self, data):

        pass

    @abc.abstractmethod
    def transform(self, data, **kwargs):

        pass

    @abc.abstractmethod
    def inverse_transform(self, data, **kwargs):

        pass
