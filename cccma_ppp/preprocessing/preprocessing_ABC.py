import abc


class PreprocessModuleABC(abc.ABC):
    """
    Abstract base class for preprocessing modules.
    """

    @abc.abstractmethod
    def fit(self, data: object) -> None:
        """
        Fit preprocessing parameters to data.

        Parameters
        ----------
        data : object
            Input data used to estimate preprocessing statistics.

        Returns
        -------
        None
        """

        pass

    @abc.abstractmethod
    def transform(self, data, **kwargs):
        """
        Apply preprocessing transformation.

        Parameters
        ----------
        data : object
            Input data to transform.
        **kwargs : dict
            Additional arguments for transformation.

        Returns
        -------
        object
            Transformed data.
        """

        pass

    @abc.abstractmethod
    def inverse_transform(self, data, **kwargs):
        """
        Reverse preprocessing transformation.

        Parameters
        ----------
        data : object
            Transformed data.
        **kwargs : dict
            Additional arguments for inverse transformation.

        Returns
        -------
        object
            Data in original representation.
        """
        pass
