import pytest
import torch
import numpy as np

from cccma_ppp.train.dataloader import (
    BatchData,
    TrainDataloaderConfig,
    collate_batch,
)


@pytest.mark.pruned
def test_batchdata_basic_nan_cleanup():
    x = torch.tensor([[1.0, float("nan")]])
    y = torch.tensor([[float("nan"), 2.0]])

    b = BatchData(x, y)

    assert not torch.isnan(b.input).any()
    assert not torch.isnan(b.target).any()


@pytest.mark.pruned
def test_batchdata_with_mask():
    x = torch.tensor([[1.0, float("nan")]])
    y = torch.tensor([[float("nan"), 2.0]])

    b = BatchData(x, y, return_spatial_mask=True)

    assert isinstance(b.input, tuple)
    assert isinstance(b.target, tuple)

    data, mask = b.input
    assert mask.shape == x.shape


@pytest.mark.pruned
def test_batchdata_reduce_mask():
    x = torch.tensor([[1.0, float("nan")], [1.0, 2.0]])
    y = torch.tensor([[1.0, 2.0], [float("nan"), 2.0]])

    b = BatchData(x, y, return_spatial_mask=True)

    assert isinstance(b.input, tuple)


@pytest.mark.pruned
def test_batchdata_to_device_no_mask():
    x = torch.ones((2, 2))
    y = torch.ones((2, 2))

    b = BatchData(x, y)
    b.to_device("cpu")

    assert isinstance(b.input, torch.Tensor)


def test_batchdata_to_device_with_mask():
    x = torch.ones((2, 2))
    y = torch.ones((2, 2))

    b = BatchData(x, y, return_spatial_mask=True)
    b.to_device("cpu")

    assert isinstance(b.input, tuple)


def test_batchdata_with_added_features():
    x = torch.ones((2, 2))
    y = torch.ones((2, 2))
    f = torch.ones((2, 3))

    b = BatchData(x, y, added_features=f)
    b.to_device("cpu")

    assert b.added_features.shape == (2, 3)


@pytest.mark.pruned
def test_collate_basic():
    batch = [
        {"input": torch.ones(2), "target": torch.zeros(2), "added_features": None},
        {"input": torch.ones(2), "target": torch.zeros(2), "added_features": None},
    ]

    out = collate_batch(batch)

    assert isinstance(out, BatchData)


def test_collate_with_features():
    batch = [
        {
            "input": torch.ones(2),
            "target": torch.zeros(2),
            "added_features": torch.ones(3),
        },
        {
            "input": torch.ones(2),
            "target": torch.zeros(2),
            "added_features": torch.ones(3),
        },
    ]

    out = collate_batch(batch)

    assert out.added_features is not None


@pytest.mark.pruned
def test_collate_with_mask_flags():
    batch = [
        {"input": torch.ones(2), "target": torch.zeros(2), "added_features": None},
        {"input": torch.ones(2), "target": torch.zeros(2), "added_features": None},
    ]

    out = collate_batch(batch, return_spatial_mask=True)

    assert isinstance(out.input, tuple)


class DummyDatasetConfig:
    def __init__(self):
        self.available_train_time = np.arange(2000, 2005)
        self.lead_months = np.arange(1, 13)

    def _fit_preprocessors(self, *args, **kwargs):
        pass

    def _load_fitted_preprocessors(self, *args, **kwargs):
        pass

    def build_dataset(self, **kwargs):
        class DummySet:
            def __len__(self):
                return 2

        return DummySet()

    @property
    def ds_operator(self):
        class O:
            def get_weights(self, *a):
                return "w"

            def get_input_var_metadata(self):
                return "i"

            def get_target_var_metadata(self):
                return "t"

        return O()


class DummyDistributed:
    rank = 0
    world_size = 1
    distributed = False

    def is_root(self):
        return True

    def barrier(self):
        pass


@pytest.mark.pruned
def test_config_default_train_years():
    cfg = TrainDataloaderConfig(
        dataset_config=DummyDatasetConfig(),
        batch_size=2,
    )

    assert len(cfg.train_years) > 0


@pytest.mark.pruned
def test_config_validation_split():
    cfg = TrainDataloaderConfig(
        dataset_config=DummyDatasetConfig(),
        batch_size=2,
        num_validation_years=2,
    )

    assert hasattr(cfg, "validation_years")


@pytest.mark.pruned
def test_config_workers_zero_sets_prefetch_none():
    cfg = TrainDataloaderConfig(
        dataset_config=DummyDatasetConfig(),
        batch_size=2,
        num_data_workers=0,
    )

    assert cfg.prefetch_factor is None


def test_config_invalid_train_years():
    cfg_data = DummyDatasetConfig()

    with pytest.raises(ValueError):
        TrainDataloaderConfig(
            dataset_config=cfg_data,
            batch_size=2,
            train_years=(1900, 1901),
        )


@pytest.mark.pruned
def test_setup_distributed_sets_flags(tmp_path):
    cfg = TrainDataloaderConfig(
        dataset_config=DummyDatasetConfig(),
        batch_size=2,
    )

    dist = DummyDistributed()

    cfg.setup_distributed(dist, save_path=tmp_path)

    assert cfg._setup
    assert cfg.rank == 0


@pytest.mark.pruned
def test_setup_distributed_with_ddp_load():
    cfg = TrainDataloaderConfig(
        dataset_config=DummyDatasetConfig(),
        batch_size=2,
    )

    dist = DummyDistributed()
    dist.distributed = True

    cfg.setup_distributed(dist, save_path="path")

    assert cfg._setup


def test_build_train_loader_requires_setup():
    cfg = TrainDataloaderConfig(
        dataset_config=DummyDatasetConfig(),
        batch_size=2,
    )

    with pytest.raises(RuntimeError):
        cfg.build_train_loader()


def test_build_train_loader_success(monkeypatch):
    cfg = TrainDataloaderConfig(
        dataset_config=DummyDatasetConfig(),
        batch_size=2,
    )

    dist = DummyDistributed()
    cfg.setup_distributed(dist)

    loader = cfg.build_train_loader()
    assert loader is not None


def test_build_validation_loader_none_warns(monkeypatch):
    cfg = TrainDataloaderConfig(
        dataset_config=DummyDatasetConfig(),
        batch_size=2,
        num_validation_years=0,
    )

    dist = DummyDistributed()
    cfg.setup_distributed(dist)

    with pytest.warns(UserWarning):
        out = cfg.build_validation_loader()

    assert out is None


def test_build_validation_loader_success():
    cfg = TrainDataloaderConfig(
        dataset_config=DummyDatasetConfig(),
        batch_size=2,
        num_validation_years=1,
    )

    dist = DummyDistributed()
    cfg.setup_distributed(dist)

    out = cfg.build_validation_loader()

    assert out is not None


@pytest.mark.pruned
def test_get_weights():
    cfg = TrainDataloaderConfig(
        dataset_config=DummyDatasetConfig(),
        batch_size=2,
    )

    assert cfg.get_weights() == "w"


@pytest.mark.pruned
def test_metadata_properties():
    cfg = TrainDataloaderConfig(
        dataset_config=DummyDatasetConfig(),
        batch_size=2,
    )

    assert cfg.input_var_metadata == "i"
    assert cfg.target_var_metadata == "t"


def test_batchdata_reduce_mask_branch(monkeypatch):
    x = torch.tensor([[1.0, float("nan")], [1.0, 2.0]])
    y = torch.tensor([[1.0, 2.0], [float("nan"), 2.0]])

    orig_mean = torch.Tensor.mean

    def safe_mean(self, *args, **kwargs):
        return orig_mean(self.float(), *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "mean", safe_mean)

    b = BatchData(
        x,
        y,
        return_spatial_mask=True,
        reduce_spatial_mask=True,
    )

    data, mask = b.input
    assert mask is not None


@pytest.mark.pruned
def test_batchdata_to_device_without_features():
    x = torch.ones((2, 2))
    y = torch.ones((2, 2))

    b = BatchData(x, y)
    b.to_device("cpu")

    assert b.added_features is None


@pytest.mark.pruned
def test_collate_with_metadata_and_no_features():
    batch = [
        (
            {"input": torch.ones(2), "target": torch.zeros(2), "added_features": None},
            {"a": 1},
        ),
        (
            {"input": torch.ones(2), "target": torch.zeros(2), "added_features": None},
            {"a": 2},
        ),
    ]

    data = collate_batch(batch)
    meta = getattr(data, "metadata", None)

    assert data.added_features is None
    assert meta is None


def test_config_custom_train_years_valid():
    cfg = TrainDataloaderConfig(
        dataset_config=DummyDatasetConfig(),
        batch_size=2,
        train_years=(2000, 2002),
    )

    assert list(cfg.train_years) == [2000, 2001, 2002]


def test_config_custom_train_and_validation_split():
    cfg = TrainDataloaderConfig(
        dataset_config=DummyDatasetConfig(),
        batch_size=2,
        train_years=(2000, 2002),
        num_validation_years=2,
    )

    assert hasattr(cfg, "validation_years")


class NonRootDistributed(DummyDistributed):
    def is_root(self):
        return False


def test_setup_distributed_non_root():
    cfg = TrainDataloaderConfig(
        dataset_config=DummyDatasetConfig(),
        batch_size=2,
    )

    dist = NonRootDistributed()

    cfg.setup_distributed(dist)

    assert cfg._setup


@pytest.mark.pruned
def test_get_weights_with_config():
    cfg = TrainDataloaderConfig(
        dataset_config=DummyDatasetConfig(),
        batch_size=2,
    )

    class DummyWeights:
        pass

    assert cfg.get_weights(DummyWeights()) == "w"


@pytest.mark.pruned
def test_build_validation_loader_with_mask_flags():
    cfg = TrainDataloaderConfig(
        dataset_config=DummyDatasetConfig(),
        batch_size=2,
        num_validation_years=1,
    )

    dist = DummyDistributed()
    cfg.setup_distributed(dist)

    out = cfg.build_validation_loader(
        return_spatial_mask=True,
        reduce_spatial_mask=False,
    )

    assert out is not None


@pytest.mark.pruned
def test_build_train_loader_with_mask_flags():
    cfg = TrainDataloaderConfig(
        dataset_config=DummyDatasetConfig(),
        batch_size=2,
    )

    dist = DummyDistributed()
    cfg.setup_distributed(dist)

    out = cfg.build_train_loader(return_spatial_mask=True)

    assert out is not None
