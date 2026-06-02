import abc


class PreprocessModuleABC(abc.ABC):
    """
    Abstract base class for preprocessing modules.
    """

    @abc.abstractmethod
    def fit(self, data):
        """
        Fit preprocessing parameters to data.

        Parameters
        ----------
        data : object
            Input dataset.

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            If not implemented in subclass.
        """

    ...

    @abc.abstractmethod
    def transform(self, data, **kwargs):
        """
        Apply preprocessing transformation.

        Parameters
        ----------
        data : object
            Input dataset.
        **kwargs
            Additional transformation arguments.

        Returns
        -------
        object
            Transformed dataset.

        Raises
        ------
        NotImplementedError
            If not implemented in subclass.
        """

    ...

    @abc.abstractmethod
    def inverse_transform(self, data, **kwargs):
        """
        Reverse preprocessing transformation.

        Parameters
        ----------
        data : object
            Transformed dataset.
        **kwargs
            Additional arguments for inverse transformation.

        Returns
        -------
        object
            Original dataset.

        Raises
        ------
        NotImplementedError
            If not implemented in subclass.
        """
        ...
