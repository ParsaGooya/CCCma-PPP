import abc


class XarrayDatasetConfigABC(abc.ABC):
    @abc.abstractmethod
    def build(self):
        pass


class XarrayDatasetABC(abc.ABC):
    @abc.abstractmethod
    def __getitem__(self, index):
        pass

    @abc.abstractmethod
    def __len__(self):
        pass


class DataConfig(abc.ABC):
    @abc.abstractmethod
    def _allowed_dims(cls):
        pass

    @abc.abstractmethod
    def _required_dims(cls):
        pass
