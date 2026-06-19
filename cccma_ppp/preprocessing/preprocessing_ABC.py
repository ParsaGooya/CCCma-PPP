import abc


class PreprocessModuleABC(abc.ABC):

    """
    Abstract base class defining the interface for preprocessing modules.

    Methods
    -------
    fit(data)
        Learn preprocessing parameters from data.
    transform(data, **kwargs)
        Apply preprocessing transformation.
    inverse_transform(data, **kwargs)
        Revert preprocessing transformation.
    """

    @abc.abstractmethod
    def fit(self, data):
        pass

    @abc.abstractmethod
    def transform(self, data, **kwargs):
        pass

    @abc.abstractmethod
    def inverse_transform(self, data, **kwargs):
        pass
