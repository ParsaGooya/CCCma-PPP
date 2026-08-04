import abc


class PreprocessModuleABC(abc.ABC):
    """
    Document this class.
    """

    @abc.abstractmethod
    def fit(self, data):
        """
        Document this function.

        Parameters
        ----------
        data : Any
            Description not yet provided.
        """
        pass

    @abc.abstractmethod
    def transform(self, data, **kwargs):
        """
        Document this function.

        Parameters
        ----------
        data : Any
            Description not yet provided.
        **kwargs : Any
            Description not yet provided.
        """
        pass

    @abc.abstractmethod
    def inverse_transform(self, data, **kwargs):
        """
        Document this function.

        Parameters
        ----------
        data : Any
            Description not yet provided.
        **kwargs : Any
            Description not yet provided.
        """
        pass
