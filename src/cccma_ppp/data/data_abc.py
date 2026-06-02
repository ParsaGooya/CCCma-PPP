import abc


class XarrayDatasetConfigABC(abc.ABC):
    """
    Abstract base class for xarray dataset configuration.
    """

    @abc.abstractmethod
    def build(self):
        """
        Build and return a configured xarray dataset.

        Returns
        -------
        object
            Constructed dataset instance.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        ...

        ...


class XarrayDatasetABC(abc.ABC):
    """
    Abstract base class for xarray datasets.
    """

    @abc.abstractmethod
    def __getitem__(self, index):
        """
        Retrieve a dataset item by index.

        Parameters
        ----------
        index : int
            Index of the item to retrieve.

        Returns
        -------
        object
            Retrieved dataset item.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        ...

    @abc.abstractmethod
    def __len__(self):
        """
        Return the number of items in the dataset.

        Returns
        -------
        int
            Number of dataset items.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        ...
