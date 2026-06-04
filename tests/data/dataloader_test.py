import pytest
import torch
import numpy as np

from cccma_ppp.data.dataloader import (
    BatchData,
    TrainDataloaderConfig,
    Dataloader,
    collate_batch,
)

# ----------------------------
# Mock / Stub Classes
# ----------------------------


class DummyDataset:
    def __init__(self, size=10):
        self.data = list(range(size))
        self.added_features_dim = 2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input": torch.tensor([float(idx)]),
            "target": torch.tensor([float(idx + 1)]),
            "added_features": torch.tensor([1.0, 2.0]),
        }

    def get_input_shape(self):
        return (1,)

    def get_target_shape(self):
        return (1,)


class DummyDatasetConfig:
    def __init__(self):
        self.available_train_time = np.arange(2000, 2005)
        self.num_lead_months = 12

    def build(self, **kwargs):
        return DummyDataset()

    def _fit_preprocessors(self, *args, **kwargs):
        pass

    def _load_fitted_preprocessors(self, *args, **kwargs):
        pass

    def _add_fitted_preprocessor(self, *args, **kwargs):
        pass

    def get_weights(self, config=None):
        return "weights"


class DummyDistributed:
    def __init__(self, distributed=False):
        self.rank = 0
        self.world_size = 1
        self.distributed = distributed

    def is_root(self):
        return True

    def barrier(self):
        pass


# ----------------------------
# BatchData Tests
# ----------------------------


def test_batchdata_basic():
    inp = torch.tensor([[1.0]])
    tgt = torch.tensor([[2.0]])

    batch = BatchData(inp, tgt)

    assert torch.equal(batch.input, inp)
    assert torch.equal(batch.target, tgt)


def test_batchdata_nan_handling():
    inp = torch.tensor([[float("nan")]])
    tgt = torch.tensor([[float("nan")]])

    batch = BatchData(inp, tgt)

    assert batch.input.item() == 0.0
    assert batch.target.item() == 0.0


def test_batchdata_masks():
    inp = torch.tensor([[1.0, float("nan")]])
    tgt = torch.tensor([[float("nan"), 2.0]])

    batch = BatchData(inp, tgt, return_spatial_mask=True)

    input_vals, input_mask = batch.input
    _, target_mask = batch.target

    assert input_mask.shape == inp.shape
    assert target_mask.shape == tgt.shape


def test_batchdata_to_device():
    inp = torch.tensor([[1.0]])
    tgt = torch.tensor([[2.0]])

    batch = BatchData(inp, tgt)
    batch = batch.to_device("cpu")

    assert batch.input.device.type == "cpu"


# ----------------------------
# collate_batch Tests
# ----------------------------


def test_collate_batch_basic():
    batch = [
        {
            "input": torch.tensor([1.0]),
            "target": torch.tensor([2.0]),
            "added_features": None,
        },
        {
            "input": torch.tensor([3.0]),
            "target": torch.tensor([4.0]),
            "added_features": None,
        },
    ]

    result = collate_batch(batch)

    assert result.input.shape[0] == 2
    assert result.target.shape[0] == 2


def test_collate_with_added_features():
    batch = [
        {
            "input": torch.tensor([1.0]),
            "target": torch.tensor([2.0]),
            "added_features": torch.tensor([1.0, 2.0]),
        },
        {
            "input": torch.tensor([3.0]),
            "target": torch.tensor([4.0]),
            "added_features": torch.tensor([3.0, 4.0]),
        },
    ]

    result = collate_batch(batch)

    assert result.added_features.shape[0] == 2


# ----------------------------
# TrainDataloaderConfig Tests
# ----------------------------


def test_config_default_years():
    cfg = TrainDataloaderConfig(
        dataset_config=DummyDatasetConfig(),
        batch_size=2,
    )

    assert len(cfg.train_years) > 0


def test_config_invalid_years():
    cfg_obj = DummyDatasetConfig()

    with pytest.raises(AssertionError):
        TrainDataloaderConfig(
            dataset_config=cfg_obj,
            batch_size=2,
            train_years=(1990, 1995),
        )


def test_config_prefetch_disabled():
    cfg = TrainDataloaderConfig(
        dataset_config=DummyDatasetConfig(),
        batch_size=2,
        num_data_workers=0,
    )

    assert cfg.prefetch_factor is None


def test_setup_distributed_sets_flag():
    cfg = TrainDataloaderConfig(
        dataset_config=DummyDatasetConfig(),
        batch_size=2,
    )

    dist = DummyDistributed()

    cfg.setup_distributed(dist)

    assert cfg._setup is True


def test_add_fitted_preprocessor_requires_fitted():
    cfg = TrainDataloaderConfig(
        dataset_config=DummyDatasetConfig(),
        batch_size=2,
    )

    class DummyPreprocessor:
        fitted = False

    with pytest.raises(AssertionError):
        cfg._add_fitted_preprocessor(DummyPreprocessor())


# ----------------------------
# Dataloader Tests
# ----------------------------


def setup_loader():
    cfg = TrainDataloaderConfig(
        dataset_config=DummyDatasetConfig(),
        batch_size=2,
    )

    dist = DummyDistributed()
    cfg.setup_distributed(dist)

    return cfg.build_train_loader()


def test_dataloader_iter():
    loader = setup_loader()

    batch = next(iter(loader))
    assert isinstance(batch, BatchData)


def test_dataloader_shapes():
    loader = setup_loader()

    assert loader.input_shape == (1,)
    assert loader.target_shape == (1,)


def test_dataloader_weights():
    loader = setup_loader()

    assert loader.get_weights() == "weights"


def test_dataloader_subset():
    loader = setup_loader()

    subset_iter = loader.subset_loader(start_batch=1)
    batch = next(subset_iter)

    assert isinstance(batch, BatchData)


def test_dataloader_set_epoch_no_sampler():
    loader = setup_loader()

    # Should not fail even if sampler is None
    loader.set_epoch(1)


# ----------------------------
# Validation Loader Tests
# ----------------------------


def test_validation_loader_none_if_disabled():
    cfg = TrainDataloaderConfig(
        dataset_config=DummyDatasetConfig(),
        batch_size=2,
        num_validation_years=0,
    )

    dist = DummyDistributed()
    cfg.setup_distributed(dist)

    val = cfg.build_validation_loader()
    assert val is None


def test_validation_loader_created():
    cfg = TrainDataloaderConfig(
        dataset_config=DummyDatasetConfig(),
        batch_size=2,
        num_validation_years=1,
    )

    dist = DummyDistributed()
    cfg.setup_distributed(dist)

    val_loader = cfg.build_validation_loader()

    assert val_loader is not None


# ----------------------------
# EXTRA BatchData Coverage
# ----------------------------


def test_batchdata_reduce_spatial_mask():
    inp = torch.tensor([[1.0, float("nan")], [1.0, 2.0]])
    tgt = torch.tensor([[1.0, float("nan")], [1.0, 2.0]])

    batch = BatchData(
        inp,
        tgt,
        return_spatial_mask=True,
        reduce_spatial_mask=True,
    )

    input_vals, input_mask = batch.input

    # reduced → no batch dimension
    assert input_mask.ndim == inp.ndim - 1


def test_batchdata_to_device_with_mask():
    inp = torch.tensor([[1.0, float("nan")]])
    tgt = torch.tensor([[2.0, 3.0]])

    batch = BatchData(inp, tgt, return_spatial_mask=True)
    batch.to_device("cpu")

    input_vals, input_mask = batch.input

    assert input_vals.device.type == "cpu"
    assert input_mask.device.type == "cpu"


# ----------------------------
# EXTRA collate_batch Coverage
# ----------------------------


def test_collate_with_masks_enabled():
    batch = [
        {
            "input": torch.tensor([1.0]),
            "target": torch.tensor([2.0]),
            "added_features": None,
        },
        {
            "input": torch.tensor([3.0]),
            "target": torch.tensor([4.0]),
            "added_features": None,
        },
    ]

    result = collate_batch(batch, return_spatial_mask=True)

    assert isinstance(result.input, tuple)
    assert isinstance(result.target, tuple)


# ----------------------------
# EXTRA Config Coverage
# ----------------------------


def test_config_train_year_range_expansion():
    cfg = TrainDataloaderConfig(
        dataset_config=DummyDatasetConfig(),
        batch_size=2,
        train_years=(2001, 2003),
    )

    # should expand into full range
    assert list(cfg.train_years) == [2001, 2002, 2003]


def test_add_fitted_preprocessor_success():
    cfg = TrainDataloaderConfig(
        dataset_config=DummyDatasetConfig(),
        batch_size=2,
    )

    class DummyPreprocessor:
        fitted = True

    # should not raise
    cfg._add_fitted_preprocessor(DummyPreprocessor())


def test_build_train_loader_without_setup():
    cfg = TrainDataloaderConfig(
        dataset_config=DummyDatasetConfig(),
        batch_size=2,
    )

    with pytest.raises(RuntimeError):
        cfg.build_train_loader()


# ----------------------------
# EXTRA Dataloader Coverage
# ----------------------------


def test_dataloader_len():
    loader = setup_loader()

    # requires you implement __len__
    assert len(loader) > 0


def test_dataloader_added_features_dim():
    loader = setup_loader()

    assert loader.added_features_dim == 2


def test_distributed_sampler_created():
    cfg = TrainDataloaderConfig(
        dataset_config=DummyDatasetConfig(),
        batch_size=2,
    )

    dist = DummyDistributed(distributed=True)
    dist.world_size = 2

    cfg.setup_distributed(dist)

    loader = cfg.build_train_loader()

    assert loader.sampler is not None


# ----------------------------
# EXTRA Validation Coverage
# ----------------------------


def test_validation_years_computation():
    cfg = TrainDataloaderConfig(
        dataset_config=DummyDatasetConfig(),
        batch_size=2,
        num_validation_years=2,
    )

    assert len(cfg.validation_years) == 2


def test_build_validation_loader_without_setup():
    cfg = TrainDataloaderConfig(
        dataset_config=DummyDatasetConfig(),
        batch_size=2,
        num_validation_years=1,
    )

    with pytest.raises(RuntimeError):
        cfg.build_validation_loader()


def test_sampler_none_when_single_process():
    loader = setup_loader()
    assert loader.sampler is None


def test_set_epoch_with_sampler(monkeypatch):
    cfg = TrainDataloaderConfig(
        dataset_config=DummyDatasetConfig(),
        batch_size=2,
    )

    dist = DummyDistributed(distributed=True)
    dist.world_size = 2

    cfg.setup_distributed(dist)
    loader = cfg.build_train_loader()

    called = {"flag": False}

    def fake_set_epoch(epoch):
        called["flag"] = True

    loader.sampler.set_epoch = fake_set_epoch

    loader.set_epoch(3)

    assert called["flag"] is True


def test_setup_distributed_loads_when_distributed(monkeypatch):
    cfg = TrainDataloaderConfig(
        dataset_config=DummyDatasetConfig(),
        batch_size=2,
    )

    called = {"flag": False}

    def fake_load(*args, **kwargs):
        called["flag"] = True

    cfg.dataset_config._load_fitted_preprocessors = fake_load

    dist = DummyDistributed(distributed=True)
    dist.world_size = 2

    cfg.setup_distributed(dist)

    assert called["flag"] is True


def test_available_train_years_excludes_validation():
    cfg = TrainDataloaderConfig(
        dataset_config=DummyDatasetConfig(),
        batch_size=2,
        num_validation_years=2,
    )

    # last 2 years excluded
    assert len(cfg.available_train_years) == 3


def test_batchdata_to_device_with_added_features():
    inp = torch.tensor([[1.0]])
    tgt = torch.tensor([[2.0]])
    feats = torch.tensor([[5.0]])

    batch = BatchData(inp, tgt, added_features=feats)
    batch.to_device("cpu")

    assert batch.added_features.device.type == "cpu"


def test_get_sampler_branch_direct():
    cfg = TrainDataloaderConfig(
        dataset_config=DummyDatasetConfig(),
        batch_size=2,
    )

    dist = DummyDistributed(distributed=True)
    dist.world_size = 2

    cfg.setup_distributed(dist)

    loader = cfg.build_train_loader()

    assert loader.sampler is not None


def test_subset_loader_empty():
    loader = setup_loader()

    it = loader.subset_loader(start_batch=100)
    with pytest.raises(StopIteration):
        next(it)


def test_collate_all_features_masks():
    batch = [
        {
            "input": torch.tensor([1.0]),
            "target": torch.tensor([2.0]),
            "added_features": torch.tensor([1.0]),
        },
        {
            "input": torch.tensor([3.0]),
            "target": torch.tensor([4.0]),
            "added_features": torch.tensor([2.0]),
        },
    ]

    result = collate_batch(
        batch,
        return_spatial_mask=True,
        reduce_spatial_mask=False,
    )

    assert isinstance(result.input, tuple)
    assert result.added_features is not None
