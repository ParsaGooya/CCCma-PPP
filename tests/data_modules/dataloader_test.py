from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import xarray as xr
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from cccma_ppp.data_modules.dataloader import (
    BatchDataABC,
    Dataloader,
    DataloaderConfigABC,
)


INIT_TIME_DIM = DataloaderConfigABC.init_time_dim
LEAD_TIME_DIM = DataloaderConfigABC.lead_time_dim


def make_available_times():
    values = np.array(
        [
            "2000-01-01",
            "2001-01-01",
            "2002-01-01",
            "2003-01-01",
        ],
        dtype="datetime64[ns]",
    )

    return xr.DataArray(
        values,
        dims=(INIT_TIME_DIM,),
        coords={
            INIT_TIME_DIM: values,
        },
        name=INIT_TIME_DIM,
    )


class ConcreteBatchData(BatchDataABC):
    def __init__(
        self,
        input,
        target=None,
        added_features=None,
        metadata=None,
    ):
        self.input = input
        self.target = target
        self.added_features = added_features
        self.metadata = metadata

    def to_device(
        self,
        device,
    ):
        self.input = self.input.to(device)

        if self.target is not None:
            self.target = self.target.to(device)

        if self.added_features is not None:
            self.added_features = self.added_features.to(device)

        return self


class ConcreteDataloaderConfig(DataloaderConfigABC):
    def __init__(
        self,
        *,
        available_times=None,
        num_data_workers=0,
        batch_size=2,
        prefetch_factor=2,
        pin_memory=True,
        reduce_spatial_mask=False,
        drop_last=False,
    ):
        self.dataset_config = MagicMock()
        self.num_data_workers = num_data_workers
        self.batch_size = batch_size
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.reduce_spatial_mask = reduce_spatial_mask
        self.drop_last = drop_last
        self.return_spatial_mask = False
        self.time_features = None

        self._available_times = (
            make_available_times() if available_times is None else available_times
        )

        super().__init__()

    @property
    def available_times(self):
        return self._available_times

    def setup_distributed(self):
        self._setup = True
        return self


class DummyDataset(Dataset):
    def __init__(
        self,
        size=5,
        input_shape=(2, 3, 4),
        target_shape=(1, 3, 4),
        added_features_dim=2,
    ):
        self.size = size
        self._input_shape = input_shape
        self._target_shape = target_shape
        self._added_features_dim = added_features_dim

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {
            "index": index,
            "input": torch.tensor(
                [float(index)],
                dtype=torch.float32,
            ),
            "target": torch.tensor(
                [float(index + 1)],
                dtype=torch.float32,
            ),
        }

    def get_input_shape(self):
        return self._input_shape

    def get_target_shape(self):
        return self._target_shape

    def get_added_features_dim(self):
        return self._added_features_dim


def collate_batch(
    batch,
    *,
    return_spatial_mask=False,
    reduce_spatial_mask=False,
):
    return {
        "indexes": [item["index"] for item in batch],
        "input": torch.stack([item["input"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "return_spatial_mask": return_spatial_mask,
        "reduce_spatial_mask": reduce_spatial_mask,
    }


class TestBatchDataABC:
    def test_to_device_moves_tensors(self):
        batch = ConcreteBatchData(
            input=torch.tensor(
                [
                    1.0,
                    2.0,
                ]
            ),
            target=torch.tensor(
                [
                    3.0,
                ]
            ),
            added_features=torch.tensor(
                [
                    4.0,
                ]
            ),
        )

        result = batch.to_device("cpu")

        assert result is batch
        assert batch.input.device.type == "cpu"
        assert batch.target.device.type == "cpu"
        assert batch.added_features.device.type == "cpu"

    def test_to_device_handles_optional_values(self):
        batch = ConcreteBatchData(
            input=torch.tensor(
                [
                    1.0,
                ]
            ),
            target=None,
            added_features=None,
        )

        result = batch.to_device("cpu")

        assert result is batch
        assert batch.target is None
        assert batch.added_features is None

    def test_metadata_is_preserved(self):
        metadata = [
            {
                "year": 2000,
            }
        ]

        batch = ConcreteBatchData(
            input=torch.tensor(
                [
                    1.0,
                ]
            ),
            metadata=metadata,
        )

        assert batch.metadata is metadata


class TestDataloaderConfigABC:
    def test_initial_state(self):
        config = ConcreteDataloaderConfig(
            num_data_workers=1,
            pin_memory=True,
        )

        assert config._setup is False

        assert config.pin_memory is False

    def test_zero_workers_disables_prefetch_factor(self):
        config = ConcreteDataloaderConfig(
            num_data_workers=0,
            prefetch_factor=4,
        )

        assert config.prefetch_factor is None

    def test_nonzero_workers_preserves_prefetch_factor(self):
        config = ConcreteDataloaderConfig(
            num_data_workers=2,
            prefetch_factor=4,
        )

        assert config.prefetch_factor == 4

    def test_setup_distributed(self):
        config = ConcreteDataloaderConfig()

        result = config.setup_distributed()

        assert result is config
        assert config._setup is True

    def test_select_requested_times_from_data_array(self):
        config = ConcreteDataloaderConfig()

        result = config.select_requested_times(
            slice(
                "2001",
                "2002",
            )
        )

        assert isinstance(
            result,
            xr.DataArray,
        )
        assert np.array_equal(
            result.values,
            np.array(
                [
                    "2001-01-01",
                    "2002-01-01",
                ],
                dtype="datetime64[ns]",
            ),
        )

    def test_select_requested_times_from_numpy_array(self):
        available = make_available_times().values[::-1]

        config = ConcreteDataloaderConfig(
            available_times=available,
        )

        result = config.select_requested_times(
            slice(
                "2000",
                "2001",
            )
        )

        assert np.array_equal(
            result.values,
            np.array(
                [
                    "2000-01-01",
                    "2001-01-01",
                ],
                dtype="datetime64[ns]",
            ),
        )

    def test_select_requested_times_sorts_available_times(self):
        available = make_available_times().isel(
            {
                INIT_TIME_DIM: [
                    3,
                    1,
                    2,
                    0,
                ]
            }
        )

        config = ConcreteDataloaderConfig(
            available_times=available,
        )

        result = config.select_requested_times(
            slice(
                "2000",
                "2003",
            )
        )

        assert np.array_equal(
            result.values,
            make_available_times().values,
        )

    def test_select_requested_times_rejects_early_start(self):
        config = ConcreteDataloaderConfig()

        with pytest.raises(
            ValueError,
            match="before the first available time",
        ):
            config.select_requested_times(
                slice(
                    "1999",
                    "2001",
                )
            )

    def test_select_requested_times_rejects_late_stop(self):
        config = ConcreteDataloaderConfig()

        with pytest.raises(
            ValueError,
            match="after the final available time",
        ):
            config.select_requested_times(
                slice(
                    "2001",
                    "2004",
                )
            )

    def test_select_requested_times_rejects_empty_selection(self):
        available = xr.DataArray(
            np.array(
                [
                    "2000-01-01",
                    "2002-01-01",
                ],
                dtype="datetime64[ns]",
            ),
            dims=(INIT_TIME_DIM,),
            coords={
                INIT_TIME_DIM: np.array(
                    [
                        "2000-01-01",
                        "2002-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
            },
        )

        config = ConcreteDataloaderConfig(
            available_times=available,
        )

        with pytest.raises(
            ValueError,
            match="No available times fall inside",
        ):
            config.select_requested_times(
                slice(
                    "2001",
                    "2001",
                )
            )

    def test_select_requested_times_full_available_range(self):
        config = ConcreteDataloaderConfig()

        result = config.select_requested_times(
            slice(
                "2000",
                "2003",
            )
        )

        assert np.array_equal(
            result.values,
            make_available_times().values,
        )

    def test_select_requested_times_single_year(self):
        config = ConcreteDataloaderConfig()

        result = config.select_requested_times(
            slice(
                "2002",
                "2002",
            )
        )

        assert result.size == 1
        assert result.values[0] == np.datetime64("2002-01-01")


class TestDataloader:
    def test_creates_torch_dataloader(self):
        config = ConcreteDataloaderConfig(
            num_data_workers=0,
            batch_size=2,
        )
        dataset = DummyDataset(
            size=5,
        )

        loader = Dataloader(
            config=config,
            dataset=dataset,
            collate_fn=collate_batch,
            shuffle=False,
        )

        assert loader._torch_loader.batch_size == 2
        assert loader._torch_loader.num_workers == 0
        assert loader._torch_loader.prefetch_factor is None
        assert loader._torch_loader.persistent_workers is False
        assert loader._torch_loader.timeout == 0

    def test_local_loader_has_no_sampler(self):
        config = ConcreteDataloaderConfig()
        dataset = DummyDataset()

        loader = Dataloader(
            config=config,
            dataset=dataset,
            collate_fn=collate_batch,
            world_size=1,
            shuffle=False,
        )

        assert loader.sampler is None

    def test_distributed_loader_uses_distributed_sampler(self):
        config = ConcreteDataloaderConfig(
            drop_last=True,
        )
        dataset = DummyDataset(
            size=8,
        )

        loader = Dataloader(
            config=config,
            dataset=dataset,
            collate_fn=collate_batch,
            rank=1,
            world_size=2,
            shuffle=False,
        )

        assert isinstance(
            loader.sampler,
            DistributedSampler,
        )
        assert loader.sampler.num_replicas == 2
        assert loader.sampler.rank == 1
        assert loader.sampler.shuffle is False
        assert loader.sampler.drop_last is True

    def test_distributed_sampler_defaults_to_shuffle(self):
        config = ConcreteDataloaderConfig()
        dataset = DummyDataset(
            size=8,
        )

        loader = Dataloader(
            config=config,
            dataset=dataset,
            collate_fn=collate_batch,
            rank=0,
            world_size=2,
            shuffle=None,
        )

        assert isinstance(
            loader.sampler,
            DistributedSampler,
        )
        assert loader.sampler.shuffle is True

    def test_explicit_shuffle_is_used_for_local_loader(self):
        config = ConcreteDataloaderConfig()
        dataset = DummyDataset()

        loader = Dataloader(
            config=config,
            dataset=dataset,
            collate_fn=collate_batch,
            shuffle=False,
        )

        assert loader._torch_loader.sampler is not None

        assert type(loader._torch_loader.sampler).__name__ == "SequentialSampler"

    def test_default_local_shuffle_is_true(self):
        config = ConcreteDataloaderConfig()
        dataset = DummyDataset()

        loader = Dataloader(
            config=config,
            dataset=dataset,
            collate_fn=collate_batch,
            world_size=1,
            shuffle=None,
        )

        assert type(loader._torch_loader.sampler).__name__ == "RandomSampler"

    def test_collate_receives_spatial_mask_arguments(self):
        config = ConcreteDataloaderConfig(
            batch_size=2,
            reduce_spatial_mask=True,
        )
        dataset = DummyDataset(
            size=2,
        )

        loader = Dataloader(
            config=config,
            dataset=dataset,
            collate_fn=collate_batch,
            shuffle=False,
            return_spatial_mask=True,
        )

        batch = next(iter(loader))

        assert batch["return_spatial_mask"] is True
        assert batch["reduce_spatial_mask"] is True

    def test_iteration(self):
        config = ConcreteDataloaderConfig(
            batch_size=2,
        )
        dataset = DummyDataset(
            size=4,
        )

        loader = Dataloader(
            config=config,
            dataset=dataset,
            collate_fn=collate_batch,
            shuffle=False,
        )

        batches = list(loader)

        assert len(batches) == 2
        assert batches[0]["indexes"] == [
            0,
            1,
        ]
        assert batches[1]["indexes"] == [
            2,
            3,
        ]

    def test_length_with_complete_batches(self):
        config = ConcreteDataloaderConfig(
            batch_size=2,
            drop_last=False,
        )
        dataset = DummyDataset(
            size=6,
        )

        loader = Dataloader(
            config=config,
            dataset=dataset,
            collate_fn=collate_batch,
            shuffle=False,
        )

        assert len(loader) == 3

    def test_length_includes_partial_batch(self):
        config = ConcreteDataloaderConfig(
            batch_size=2,
            drop_last=False,
        )
        dataset = DummyDataset(
            size=5,
        )

        loader = Dataloader(
            config=config,
            dataset=dataset,
            collate_fn=collate_batch,
            shuffle=False,
        )

        assert len(loader) == 3

    def test_input_shape_delegates_to_dataset(self):
        config = ConcreteDataloaderConfig()
        dataset = DummyDataset(
            input_shape=(
                4,
                16,
                32,
            )
        )

        loader = Dataloader(
            config=config,
            dataset=dataset,
            collate_fn=collate_batch,
            shuffle=False,
        )

        assert loader.input_shape == (
            4,
            16,
            32,
        )

    def test_target_shape_delegates_to_dataset(self):
        config = ConcreteDataloaderConfig()
        dataset = DummyDataset(
            target_shape=(
                2,
                16,
                32,
            )
        )

        loader = Dataloader(
            config=config,
            dataset=dataset,
            collate_fn=collate_batch,
            shuffle=False,
        )

        assert loader.target_shape == (
            2,
            16,
            32,
        )

    def test_added_features_dim_delegates_to_dataset(self):
        config = ConcreteDataloaderConfig()
        dataset = DummyDataset(
            added_features_dim=6,
        )

        loader = Dataloader(
            config=config,
            dataset=dataset,
            collate_fn=collate_batch,
            shuffle=False,
        )

        assert loader.added_features_dim == 6

    def test_set_epoch_for_distributed_sampler(self):
        config = ConcreteDataloaderConfig()
        dataset = DummyDataset(
            size=8,
        )

        loader = Dataloader(
            config=config,
            dataset=dataset,
            collate_fn=collate_batch,
            rank=0,
            world_size=2,
            shuffle=False,
        )

        result = loader.set_epoch(4)

        assert loader.sampler.epoch == 4
        assert result is loader

    def test_set_epoch_without_sampler(self):
        config = ConcreteDataloaderConfig()
        dataset = DummyDataset()

        loader = Dataloader(
            config=config,
            dataset=dataset,
            collate_fn=collate_batch,
            world_size=1,
            shuffle=False,
        )

        result = loader.set_epoch(4)

        assert result is loader

    def test_subset_loader_skips_batches(self):
        config = ConcreteDataloaderConfig(
            batch_size=2,
        )
        dataset = DummyDataset(
            size=6,
        )

        loader = Dataloader(
            config=config,
            dataset=dataset,
            collate_fn=collate_batch,
            shuffle=False,
        )

        remaining = list(
            loader.subset_loader(
                start_batch=1,
            )
        )

        assert len(remaining) == 2
        assert remaining[0]["indexes"] == [
            2,
            3,
        ]
        assert remaining[1]["indexes"] == [
            4,
            5,
        ]

    def test_subset_loader_from_zero_returns_every_batch(self):
        config = ConcreteDataloaderConfig(
            batch_size=2,
        )
        dataset = DummyDataset(
            size=4,
        )

        loader = Dataloader(
            config=config,
            dataset=dataset,
            collate_fn=collate_batch,
            shuffle=False,
        )

        assert (
            len(
                list(
                    loader.subset_loader(
                        start_batch=0,
                    )
                )
            )
            == 2
        )

    def test_subset_loader_past_end_is_empty(self):
        config = ConcreteDataloaderConfig(
            batch_size=2,
        )
        dataset = DummyDataset(
            size=4,
        )

        loader = Dataloader(
            config=config,
            dataset=dataset,
            collate_fn=collate_batch,
            shuffle=False,
        )

        assert (
            list(
                loader.subset_loader(
                    start_batch=10,
                )
            )
            == []
        )

    def test_get_sampler_accepts_additional_arguments(self):
        config = ConcreteDataloaderConfig()
        dataset = DummyDataset(
            size=8,
        )

        loader = Dataloader(
            config=config,
            dataset=dataset,
            collate_fn=collate_batch,
            rank=0,
            world_size=2,
            shuffle=False,
        )

        sampler = loader._get_dataloader_sampler(
            seed=42,
        )

        assert isinstance(
            sampler,
            DistributedSampler,
        )
        assert sampler.seed == 42
