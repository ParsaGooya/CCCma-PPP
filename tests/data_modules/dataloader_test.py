import pytest
import torch
import sys
import types
from cccma_ppp.data_modules.dataloader import (
    BatchDataABC,
    Dataloader,
    DataloaderConfigABC,
    Dataset,
)

fake_dataset_module = types.ModuleType("cccma_ppp.data_modules.dataset")


class DummyDatasetConfigABC:
    pass


fake_dataset_module.DatasetConfigABC = DummyDatasetConfigABC

sys.modules["cccma_ppp.data_modules.dataset"] = fake_dataset_module


class DummyBatch(BatchDataABC):
    def __init__(self):
        self.input = torch.ones(1)
        self.target = torch.ones(1)
        self.added_features = torch.ones(1)

    def to_device(self, device):
        self.input = self.input.to(device)
        return self


class DummyDataset(Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, idx):
        return idx

    def get_input_shape(self):
        return (3, 4)

    def get_target_shape(self):
        return (1,)

    def get_added_features_dim(self):
        return 5


class DummyConfig(DataloaderConfigABC):
    def __init__(self):
        self.batch_size = 2
        self.num_data_workers = 0
        self.prefetch_factor = None
        self.drop_last = False

        self.dataset_config = object()

    def setup_distributed(self):
        return self


def dummy_collate(
    batch,
    return_spatial_mask=False,
    reduce_spatial_mask=False,
):
    return {
        "batch": batch,
        "return_spatial_mask": return_spatial_mask,
        "reduce_spatial_mask": reduce_spatial_mask,
    }


@pytest.mark.pruned
def test_batch_to_device():
    batch = DummyBatch()

    result = batch.to_device("cpu")

    assert result.input.device.type == "cpu"


@pytest.mark.pruned
def test_dataloader_init():
    loader = Dataloader(
        config=DummyConfig(),
        dataset=DummyDataset(),
        collate_fn=dummy_collate,
    )

    assert loader is not None


@pytest.mark.pruned
def test_dataloader_sampler_none_single_world():
    loader = Dataloader(
        config=DummyConfig(),
        dataset=DummyDataset(),
        collate_fn=dummy_collate,
        world_size=1,
    )

    assert loader.sampler is None


@pytest.mark.pruned
def test_dataloader_sampler_distributed():
    loader = Dataloader(
        config=DummyConfig(),
        dataset=DummyDataset(),
        collate_fn=dummy_collate,
        world_size=2,
        rank=0,
    )

    assert loader.sampler is not None


@pytest.mark.pruned
def test_dataloader_return_spatial_mask():
    loader = Dataloader(
        config=DummyConfig(),
        dataset=DummyDataset(),
        collate_fn=dummy_collate,
        return_spatial_mask=True,
    )

    batch = next(iter(loader))

    assert batch["return_spatial_mask"] is True


@pytest.mark.pruned
def test_dataloader_reduce_spatial_mask():
    loader = Dataloader(
        config=DummyConfig(),
        dataset=DummyDataset(),
        collate_fn=dummy_collate,
        reduce_spatial_mask=True,
    )

    batch = next(iter(loader))

    assert batch["reduce_spatial_mask"] is True


@pytest.mark.pruned
def test_input_shape():
    loader = Dataloader(
        config=DummyConfig(),
        dataset=DummyDataset(),
        collate_fn=dummy_collate,
    )

    assert loader.input_shape == (3, 4)


@pytest.mark.pruned
def test_target_shape():
    loader = Dataloader(
        config=DummyConfig(),
        dataset=DummyDataset(),
        collate_fn=dummy_collate,
    )

    assert loader.target_shape == (1,)


@pytest.mark.pruned
def test_added_features_dim():
    loader = Dataloader(
        config=DummyConfig(),
        dataset=DummyDataset(),
        collate_fn=dummy_collate,
    )

    assert loader.added_features_dim == 5


@pytest.mark.pruned
def test_iter_returns_batches():
    loader = Dataloader(
        config=DummyConfig(),
        dataset=DummyDataset(),
        collate_fn=dummy_collate,
    )

    batch = next(iter(loader))

    assert "batch" in batch


@pytest.mark.pruned
def test_len():
    loader = Dataloader(
        config=DummyConfig(),
        dataset=DummyDataset(),
        collate_fn=dummy_collate,
    )

    assert len(loader) == 5


@pytest.mark.pruned
def test_len_drop_last_false():
    cfg = DummyConfig()

    cfg.batch_size = 3
    cfg.drop_last = False

    loader = Dataloader(
        config=cfg,
        dataset=DummyDataset(),
        collate_fn=dummy_collate,
    )

    assert len(loader) == 4


@pytest.mark.pruned
def test_get_dataloader_sampler_single_world():
    loader = Dataloader(
        config=DummyConfig(),
        dataset=DummyDataset(),
        collate_fn=dummy_collate,
        world_size=1,
    )

    sampler = loader._get_dataloader_sampler()

    assert sampler is None


@pytest.mark.pruned
def test_get_dataloader_sampler_distributed():
    loader = Dataloader(
        config=DummyConfig(),
        dataset=DummyDataset(),
        collate_fn=dummy_collate,
        world_size=2,
        rank=1,
    )

    sampler = loader._get_dataloader_sampler()

    assert sampler is not None


def test_set_epoch_without_sampler():
    loader = Dataloader(
        config=DummyConfig(),
        dataset=DummyDataset(),
        collate_fn=dummy_collate,
        world_size=1,
    )

    result = loader.set_epoch(1)

    assert result == loader


def test_set_epoch_with_sampler():
    loader = Dataloader(
        config=DummyConfig(),
        dataset=DummyDataset(),
        collate_fn=dummy_collate,
        world_size=2,
        rank=0,
    )

    result = loader.set_epoch(1)

    assert result == loader


@pytest.mark.pruned
def test_subset_loader():
    loader = Dataloader(
        config=DummyConfig(),
        dataset=DummyDataset(),
        collate_fn=dummy_collate,
    )

    subset = loader.subset_loader(start_batch=1)

    batch = next(subset)

    assert "batch" in batch


@pytest.mark.pruned
def test_dataloader_drop_last_true():
    cfg = DummyConfig()

    cfg.batch_size = 3
    cfg.drop_last = True

    loader = Dataloader(
        config=cfg,
        dataset=DummyDataset(),
        collate_fn=dummy_collate,
        world_size=2,
    )

    assert loader.sampler is not None


@pytest.mark.pruned
def test_subset_loader_zero_start():
    loader = Dataloader(
        config=DummyConfig(),
        dataset=DummyDataset(),
        collate_fn=dummy_collate,
    )

    subset = list(loader.subset_loader(0))

    assert len(subset) == len(loader)


@pytest.mark.pruned
def test_iter_multiple_batches():
    loader = Dataloader(
        config=DummyConfig(),
        dataset=DummyDataset(),
        collate_fn=dummy_collate,
    )

    batches = list(iter(loader))

    assert len(batches) > 1


@pytest.mark.pruned
def test_batch_target_exists():
    batch = DummyBatch()

    assert batch.target is not None


@pytest.mark.pruned
def test_batch_added_features_exists():
    batch = DummyBatch()

    assert batch.added_features is not None
