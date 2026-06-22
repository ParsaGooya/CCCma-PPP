from __future__ import annotations
import abc


class XarrayDatasetConfigABC(abc.ABC):
    """
    Abstract base class for dataset configuration.
    """

    @abc.abstractmethod
    def build(self):
        """
        Construct dataset instance.

        Returns
        -------
        XarrayDatasetABC
            Instantiated dataset object.
        """
        pass


class XarrayDatasetABC(abc.ABC):
    """
    Abstract base class for dataset.

    Defines interface compatible with PyTorch-style datasets.
    """

    @abc.abstractmethod
    def __getitem__(self, index: int):
        """
        Retrieve data sample by index.

        Parameters
        ----------
        index : int
            Index of item.

        Returns
        -------
        object
            Data sample corresponding to the index.
        """

        pass

    @abc.abstractmethod
    def __len__(self):
        """
        Return dataset size.

        Returns
        -------
        int
            Total number of samples.
        """

        pass


class DataConfigABC(abc.ABC):
    """
    Abstract base class for dataset metadata configuration.
    """

    @abc.abstractmethod
    def _allowed_dims(self):
        """
        Define allowed dimensions.

        Returns
        -------
        tuple or list
            Allowed dimension names.
        """

        pass

    @abc.abstractmethod
    def _required_dims(self):
        """
        Define required dimensions.

        Returns
        -------
        tuple or list
            Required dimension names.
        """

        pass
