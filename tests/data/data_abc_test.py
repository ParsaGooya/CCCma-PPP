import pytest

from cccma_ppp.data.data_abc import XarrayDatasetABC, XarrayDatasetConfigABC


# ---------------------------
# Helpers: Concrete classes
# ---------------------------


class ValidDataset(XarrayDatasetABC):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ValidConfig(XarrayDatasetConfigABC):
    def __init__(self, data):
        self.data = data

    def build(self):
        return ValidDataset(self.data)


# ---------------------------
# Tests for XarrayDatasetABC
# ---------------------------


def test_dataset_cannot_instantiate_abstract():
    with pytest.raises(TypeError):
        XarrayDatasetABC()


def test_dataset_missing_getitem():
    class IncompleteDataset(XarrayDatasetABC):
        def __len__(self):
            return 1

    with pytest.raises(TypeError):
        IncompleteDataset()


def test_dataset_missing_len():
    class IncompleteDataset(XarrayDatasetABC):
        def __getitem__(self, index):
            return index

    with pytest.raises(TypeError):
        IncompleteDataset()


def test_valid_dataset_len():
    ds = ValidDataset([1, 2, 3])
    assert len(ds) == 3


def test_valid_dataset_getitem():
    ds = ValidDataset([10, 20, 30])
    assert ds[1] == 20


def test_dataset_index_out_of_bounds():
    ds = ValidDataset([1, 2, 3])
    with pytest.raises(IndexError):
        _ = ds[10]


# ---------------------------
# Tests for XarrayDatasetConfigABC
# ---------------------------


def test_config_cannot_instantiate_abstract():
    with pytest.raises(TypeError):
        XarrayDatasetConfigABC()


def test_config_missing_build():
    class IncompleteConfig(XarrayDatasetConfigABC):
        pass

    with pytest.raises(TypeError):
        IncompleteConfig()


def test_valid_config_build():
    cfg = ValidConfig([1, 2, 3])
    ds = cfg.build()
    assert isinstance(ds, ValidDataset)


def test_config_build_returns_dataset_like():
    cfg = ValidConfig([5, 6, 7])
    ds = cfg.build()

    assert hasattr(ds, "__getitem__")
    assert hasattr(ds, "__len__")


def test_config_dataset_integration():
    cfg = ValidConfig([100, 200])
    ds = cfg.build()

    assert len(ds) == 2
    assert ds[0] == 100
