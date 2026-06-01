import abc


class XarrayDatasetConfigABC(abc.ABC):
    """Abstract base class for xarray dataset configuration."""
    @abc.abstractmethod
    def build(
        self):
        ...


class XarrayDatasetABC(abc.ABC):
    """Abstract base class for xarray datasets."""
    @abc.abstractmethod
    def __getitem__(self, index):
        ...

    @abc.abstractmethod
    def __len__(self):
        ...
