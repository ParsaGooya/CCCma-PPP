import pytest
import torch
import numpy as np

from cccma_ppp.data.dataloader import (
    BatchData,
    TrainDataloaderConfig,
    collate_batch,
)


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


def test_config_default_years():
    cfg = TrainDataloaderConfig(
        dataset_config=DummyDatasetConfig(),
        batch_size=2,
    )

    assert len(cfg.train_years) > 0


def test_config_invalid_years():
    cfg_obj = DummyDatasetConfig()

    with pytest.raises((AssertionError, ValueError, RuntimeError)):
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

    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        cfg._add_fitted_preprocessor(DummyPreprocessor())


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

    loader.set_epoch(1)


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


def test_batchdata_to_device_with_mask():
    inp = torch.tensor([[1.0, float("nan")]])
    tgt = torch.tensor([[2.0, 3.0]])

    batch = BatchData(inp, tgt, return_spatial_mask=True)
    batch.to_device("cpu")

    input_vals, input_mask = batch.input

    assert input_vals.device.type == "cpu"
    assert input_mask.device.type == "cpu"


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


def test_config_train_year_range_expansion():
    cfg = TrainDataloaderConfig(
        dataset_config=DummyDatasetConfig(),
        batch_size=2,
        train_years=(2001, 2003),
    )

    assert list(cfg.train_years) == [2001, 2002, 2003]


def test_add_fitted_preprocessor_success():
    cfg = TrainDataloaderConfig(
        dataset_config=DummyDatasetConfig(),
        batch_size=2,
    )

    class DummyPreprocessor:
        fitted = True

    cfg._add_fitted_preprocessor(DummyPreprocessor())


def test_build_train_loader_without_setup():
    cfg = TrainDataloaderConfig(
        dataset_config=DummyDatasetConfig(),
        batch_size=2,
    )

    with pytest.raises(RuntimeError):
        cfg.build_train_loader()


def test_dataloader_len():
    loader = setup_loader()

    assert len(loader) > 0


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


class RecordingDatasetConfig(DummyDatasetConfig):
    def __init__(self):
        super().__init__()
        self.fit_called = False
        self.load_called = False
        self.add_called = False
        self.build_calls = []
        self.weights_config_seen = None

    def build(self, **kwargs):
        self.build_calls.append(kwargs)
        return DummyDataset(size=6)

    def _fit_preprocessors(self, *args, **kwargs):
        self.fit_called = True
        self.fit_args = args
        self.fit_kwargs = kwargs

    def _load_fitted_preprocessors(self, *args, **kwargs):
        self.load_called = True
        self.load_args = args
        self.load_kwargs = kwargs

    def _add_fitted_preprocessor(self, *args, **kwargs):
        self.add_called = True
        self.add_args = args
        self.add_kwargs = kwargs

    def get_weights(self, config=None):
        self.weights_config_seen = config
        return "weights-with-config"


class NonRootDistributed(DummyDistributed):
    def __init__(self, distributed=False):
        super().__init__(distributed=distributed)
        self.rank = 1

    def is_root(self):
        return False


def test_batchdata_to_device_without_added_features_stays_none():
    batch = BatchData(
        torch.tensor([[1.0]]),
        torch.tensor([[2.0]]),
        added_features=None,
    )

    batch = batch.to_device("cpu")

    assert batch.added_features is None
    assert batch.input.device.type == "cpu"
    assert batch.target.device.type == "cpu"


def test_batchdata_mask_reduction_false_keeps_mask_shape():
    inp = torch.tensor([[[1.0, float("nan")], [2.0, 3.0]]])
    tgt = torch.tensor([[[float("nan"), 5.0], [6.0, 7.0]]])

    batch = BatchData(
        inp,
        tgt,
        return_spatial_mask=True,
        reduce_spatial_mask=False,
    )

    input_values, input_mask = batch.input
    target_values, target_mask = batch.target

    assert input_values.shape == inp.shape
    assert target_values.shape == tgt.shape
    assert input_mask.shape == inp.shape
    assert target_mask.shape == tgt.shape


def test_batchdata_nan_added_features_preserved_or_zeroed():
    feats = torch.tensor([[float("nan"), 2.0]])

    batch = BatchData(
        torch.tensor([[1.0]]),
        torch.tensor([[2.0]]),
        added_features=feats,
    )

    batch = batch.to_device("cpu")

    assert batch.added_features.device.type == "cpu"


def test_collate_batch_added_features_none_branch():
    batch = [
        {
            "input": torch.tensor([1.0]),
            "target": torch.tensor([2.0]),
            "added_features": None,
        }
    ]

    result = collate_batch(batch)

    assert result.added_features is None


def test_config_train_years_numpy_array_is_accepted():
    cfg = TrainDataloaderConfig(
        dataset_config=DummyDatasetConfig(),
        batch_size=2,
        train_years=np.array([2000, 2001]),
    )

    assert list(cfg.train_years) == [2000, 2001]


def test_setup_distributed_root_fits_preprocessors():
    dataset_config = RecordingDatasetConfig()

    cfg = TrainDataloaderConfig(
        dataset_config=dataset_config,
        batch_size=2,
    )

    dist = DummyDistributed(distributed=False)
    cfg.setup_distributed(dist)

    assert cfg._setup is True
    assert dataset_config.fit_called is True


def test_setup_distributed_non_root_loads_preprocessors():
    dataset_config = RecordingDatasetConfig()

    cfg = TrainDataloaderConfig(
        dataset_config=dataset_config,
        batch_size=2,
    )

    dist = NonRootDistributed(distributed=True)
    dist.world_size = 2

    cfg.setup_distributed(dist)

    assert cfg._setup is True
    assert dataset_config.load_called is True


def test_setup_distributed_barrier_called(monkeypatch):
    cfg = TrainDataloaderConfig(
        dataset_config=RecordingDatasetConfig(),
        batch_size=2,
    )

    dist = DummyDistributed(distributed=True)
    dist.world_size = 2

    called = {"barrier": False}

    def fake_barrier():
        called["barrier"] = True

    dist.barrier = fake_barrier

    cfg.setup_distributed(dist)

    assert called["barrier"] is True


def test_build_train_loader_passes_years_to_dataset_config():
    dataset_config = RecordingDatasetConfig()

    cfg = TrainDataloaderConfig(
        dataset_config=dataset_config,
        batch_size=2,
        train_years=[2000, 2001],
    )

    cfg.setup_distributed(DummyDistributed())
    loader = cfg.build_train_loader()

    assert loader is not None
    assert len(dataset_config.build_calls) >= 1
    assert "years" in dataset_config.build_calls[-1]


def test_build_validation_loader_passes_validation_years():
    dataset_config = RecordingDatasetConfig()

    cfg = TrainDataloaderConfig(
        dataset_config=dataset_config,
        batch_size=2,
        num_validation_years=1,
    )

    cfg.setup_distributed(DummyDistributed())
    val_loader = cfg.build_validation_loader()

    assert val_loader is not None
    assert len(dataset_config.build_calls) >= 1
    assert "years" in dataset_config.build_calls[-1]


def test_loader_get_weights_forwards_config_argument():
    dataset_config = RecordingDatasetConfig()

    cfg = TrainDataloaderConfig(
        dataset_config=dataset_config,
        batch_size=2,
    )

    cfg.setup_distributed(DummyDistributed())
    loader = cfg.build_train_loader()

    sentinel_config = object()
    result = loader.get_weights(config=sentinel_config)

    assert result == "weights-with-config"
    assert dataset_config.weights_config_seen is sentinel_config


def test_subset_loader_start_zero_returns_first_batch():
    loader = setup_loader()

    subset_iter = loader.subset_loader(start_batch=0)
    batch = next(subset_iter)

    assert isinstance(batch, BatchData)


def test_subset_loader_multiple_batches():
    loader = setup_loader()

    batches = list(loader.subset_loader(start_batch=0))

    assert len(batches) > 0
    assert all(isinstance(batch, BatchData) for batch in batches)


def test_loader_set_epoch_with_none_sampler_no_error():
    loader = setup_loader()

    loader.sampler = None
    loader.set_epoch(99)

    assert loader.sampler is None


def test_distributed_sampler_rank_world_size_branch():
    cfg = TrainDataloaderConfig(
        dataset_config=DummyDatasetConfig(),
        batch_size=2,
    )

    dist = DummyDistributed(distributed=True)
    dist.rank = 1
    dist.world_size = 4

    cfg.setup_distributed(dist)

    loader = cfg.build_train_loader()

    assert loader.sampler is not None


def test_validation_loader_with_distributed_sampler():
    cfg = TrainDataloaderConfig(
        dataset_config=DummyDatasetConfig(),
        batch_size=2,
        num_validation_years=1,
    )

    dist = DummyDistributed(distributed=True)
    dist.rank = 0
    dist.world_size = 2

    cfg.setup_distributed(dist)

    val_loader = cfg.build_validation_loader()

    assert val_loader is not None
    assert val_loader.sampler is not None


def test_prefetch_factor_kept_when_workers_positive():
    cfg = TrainDataloaderConfig(
        dataset_config=DummyDatasetConfig(),
        batch_size=2,
        num_data_workers=1,
        prefetch_factor=2,
    )

    assert cfg.prefetch_factor == 2


def test_drop_last_false_builds_loader():
    cfg = TrainDataloaderConfig(
        dataset_config=DummyDatasetConfig(),
        batch_size=2,
        drop_last=False,
    )

    cfg.setup_distributed(DummyDistributed())
    loader = cfg.build_train_loader()

    assert len(loader) > 0


def test_drop_last_true_builds_loader():
    cfg = TrainDataloaderConfig(
        dataset_config=DummyDatasetConfig(),
        batch_size=2,
        drop_last=True,
    )

    cfg.setup_distributed(DummyDistributed())
    loader = cfg.build_train_loader()

    assert len(loader) > 0
