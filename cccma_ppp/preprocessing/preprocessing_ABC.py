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
        """
        Fit preprocessing module to data.

        Parameters
        ----------
        data : object
            Input data used to compute preprocessing parameters. The type depends
            on the specific implementation (e.g., xarray.DataArray, numpy array).

        Returns
        -------
        PreprocessModuleABC
            Fitted preprocessing instance.


        """
        pass

    @abc.abstractmethod
    def transform(self, data, **kwargs):
        """
        Apply preprocessing transformation to data.

        Parameters
        ----------
        data : object
            Input data to be transformed.
        **kwargs
            Additional arguments controlling transformation behavior.

        Returns
        -------
        object
            Transformed data in the same structural format as input.


        """

        pass

    @abc.abstractmethod
    def inverse_transform(self, data, **kwargs):
        """
        Revert preprocessing transformation.

        Parameters
        ----------
        data : object
            Transformed data.
        **kwargs
            Additional arguments controlling inversion behavior.

        Returns
        -------
        object
            Data mapped back to original representation.

        """
        pass
