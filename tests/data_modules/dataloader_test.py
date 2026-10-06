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
    @pytest.mark.pruned
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

    @pytest.mark.pruned
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

    @pytest.mark.pruned
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

    @pytest.mark.pruned
    def test_zero_workers_disables_prefetch_factor(self):
        config = ConcreteDataloaderConfig(
            num_data_workers=0,
            prefetch_factor=4,
        )

        assert config.prefetch_factor is None

    @pytest.mark.pruned
    def test_nonzero_workers_preserves_prefetch_factor(self):
        config = ConcreteDataloaderConfig(
            num_data_workers=2,
            prefetch_factor=4,
        )

        assert config.prefetch_factor == 4

    @pytest.mark.pruned
    def test_setup_distributed(self):
        config = ConcreteDataloaderConfig()

        result = config.setup_distributed()

        assert result is config
        assert config._setup is True

    @pytest.mark.pruned
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

    @pytest.mark.pruned
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

    @pytest.mark.pruned
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

    @pytest.mark.pruned
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

    @pytest.mark.pruned
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

    @pytest.mark.pruned
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
    @pytest.mark.pruned
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

    @pytest.mark.pruned
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

    @pytest.mark.pruned
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

    @pytest.mark.pruned
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

    @pytest.mark.pruned
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

    @pytest.mark.pruned
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

    @pytest.mark.pruned
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

    @pytest.mark.pruned
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

    @pytest.mark.pruned
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

    @pytest.mark.pruned
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

    @pytest.mark.pruned
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

    @pytest.mark.pruned
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

    @pytest.mark.pruned
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

    @pytest.mark.pruned
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

    @pytest.mark.pruned
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

    @pytest.mark.pruned
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

    @pytest.mark.pruned
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


from unittest.mock import Mock, patch


@pytest.mark.pruned
def test_batch_data_abc_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BatchDataABC()


@pytest.mark.pruned
def test_dataloader_config_abc_cannot_be_instantiated():
    with pytest.raises(TypeError):
        DataloaderConfigABC()


@pytest.mark.pruned
def test_batch_data_shared_masks_default_to_none():
    assert BatchDataABC._shared_input_mask is None
    assert BatchDataABC._shared_target_mask is None


@pytest.mark.pruned
def test_batch_data_shared_masks_are_visible_to_subclasses():
    input_mask = torch.ones(
        1,
        1,
        2,
        2,
    )
    target_mask = torch.zeros(
        1,
        1,
        2,
        2,
    )

    previous_input_mask = BatchDataABC._shared_input_mask
    previous_target_mask = BatchDataABC._shared_target_mask

    try:
        BatchDataABC._shared_input_mask = input_mask
        BatchDataABC._shared_target_mask = target_mask

        batch = ConcreteBatchData(
            input=torch.ones(1),
        )

        assert batch._shared_input_mask is input_mask
        assert batch._shared_target_mask is target_mask
    finally:
        BatchDataABC._shared_input_mask = previous_input_mask
        BatchDataABC._shared_target_mask = previous_target_mask


@pytest.mark.pruned
def test_config_class_dimension_names():
    assert DataloaderConfigABC.init_time_dim == INIT_TIME_DIM
    assert DataloaderConfigABC.lead_time_dim == LEAD_TIME_DIM


@pytest.mark.pruned
def test_config_initialization_always_disables_pin_memory():
    config = ConcreteDataloaderConfig(
        pin_memory=True,
    )

    assert config.pin_memory is False


@pytest.mark.pruned
def test_zero_workers_replace_none_prefetch_with_none():
    config = ConcreteDataloaderConfig(
        num_data_workers=0,
        prefetch_factor=None,
    )

    assert config.prefetch_factor is None


def test_nonzero_workers_allow_none_prefetch_factor():
    config = ConcreteDataloaderConfig(
        num_data_workers=2,
        prefetch_factor=None,
    )

    assert config.prefetch_factor is None


@pytest.mark.parametrize(
    "prefetch_factor",
    [
        0,
        -1,
        -10,
    ],
)
def test_nonzero_workers_reject_invalid_prefetch_factor(
    prefetch_factor,
):
    with pytest.raises(
        ValueError,
        match="must be at least 1",
    ):
        ConcreteDataloaderConfig(
            num_data_workers=1,
            prefetch_factor=prefetch_factor,
        )


@pytest.mark.pruned
@pytest.mark.parametrize(
    "prefetch_factor",
    [
        1,
        2,
        8,
    ],
)
def test_nonzero_workers_accept_positive_prefetch_factor(
    prefetch_factor,
):
    config = ConcreteDataloaderConfig(
        num_data_workers=1,
        prefetch_factor=prefetch_factor,
    )

    assert config.prefetch_factor == prefetch_factor


@pytest.mark.parametrize(
    (
        "bound",
        "expected",
    ),
    [
        (
            None,
            None,
        ),
        (
            2000,
            2000,
        ),
        (
            -1,
            -1,
        ),
        (
            "2001",
            2001,
        ),
        (
            "-10",
            -10,
        ),
        (
            "0007",
            7,
        ),
    ],
)
def test_parse_year_bound_accepts_supported_values(
    bound,
    expected,
):
    config = ConcreteDataloaderConfig()

    assert config._parse_year_bound(bound) == expected


@pytest.mark.pruned
@pytest.mark.parametrize(
    "bound",
    [
        "not-a-year",
        "2000.5",
        "",
        " ",
    ],
)
def test_parse_year_bound_rejects_invalid_strings(
    bound,
):
    config = ConcreteDataloaderConfig()

    with pytest.raises(
        ValueError,
        match="Invalid year bound",
    ):
        config._parse_year_bound(bound)


@pytest.mark.parametrize(
    "bound",
    [
        2000.0,
        [],
        {},
        object(),
    ],
)
def test_parse_year_bound_rejects_unsupported_types(
    bound,
):
    config = ConcreteDataloaderConfig()

    with pytest.raises(
        TypeError,
        match="Year bound must be int or str",
    ):
        config._parse_year_bound(bound)


@pytest.mark.pruned
def test_select_requested_times_accepts_integer_bounds():
    config = ConcreteDataloaderConfig()

    result = config.select_requested_times(
        slice(
            2001,
            2002,
        )
    )

    np.testing.assert_array_equal(
        result.values,
        np.asarray(
            [
                "2001-01-01",
                "2002-01-01",
            ],
            dtype="datetime64[ns]",
        ),
    )


@pytest.mark.pruned
def test_select_requested_times_preserves_dimension_name():
    config = ConcreteDataloaderConfig()

    result = config.select_requested_times(
        slice(
            "2000",
            "2003",
        )
    )

    assert result.dims == (INIT_TIME_DIM,)
    assert INIT_TIME_DIM in result.coords


@pytest.mark.pruned
def test_select_requested_times_does_not_mutate_original_order():
    available = make_available_times().isel(
        {
            INIT_TIME_DIM: [
                3,
                0,
                2,
                1,
            ]
        }
    )
    original = available.copy(
        deep=True,
    )

    config = ConcreteDataloaderConfig(
        available_times=available,
    )

    config.select_requested_times(
        slice(
            "2000",
            "2003",
        )
    )

    xr.testing.assert_identical(
        available,
        original,
    )


@pytest.mark.pruned
def test_select_requested_times_rejects_invalid_start_string():
    config = ConcreteDataloaderConfig()

    with pytest.raises(
        ValueError,
        match="Invalid year bound",
    ):
        config.select_requested_times(
            slice(
                "invalid",
                "2002",
            )
        )


@pytest.mark.pruned
def test_select_requested_times_rejects_invalid_stop_string():
    config = ConcreteDataloaderConfig()

    with pytest.raises(
        ValueError,
        match="Invalid year bound",
    ):
        config.select_requested_times(
            slice(
                "2001",
                "invalid",
            )
        )


@pytest.mark.pruned
def test_select_requested_times_rejects_invalid_start_type():
    config = ConcreteDataloaderConfig()

    with pytest.raises(
        TypeError,
        match="Year bound must be int or str",
    ):
        config.select_requested_times(
            slice(
                2000.5,
                "2002",
            )
        )


@pytest.mark.pruned
def test_select_requested_times_rejects_invalid_stop_type():
    config = ConcreteDataloaderConfig()

    with pytest.raises(
        TypeError,
        match="Year bound must be int or str",
    ):
        config.select_requested_times(
            slice(
                "2001",
                [],
            )
        )


@pytest.mark.pruned
def test_select_requested_times_checks_start_before_stop():
    config = ConcreteDataloaderConfig()

    with pytest.raises(
        ValueError,
        match="No available times fall inside",
    ):
        config.select_requested_times(
            slice(
                "2003",
                "2001",
            )
        )


@pytest.mark.pruned
def test_select_requested_times_error_reports_requested_start():
    config = ConcreteDataloaderConfig()

    with pytest.raises(
        ValueError,
        match="1998",
    ):
        config.select_requested_times(
            slice(
                "1998",
                "2001",
            )
        )


@pytest.mark.pruned
def test_select_requested_times_error_reports_requested_stop():
    config = ConcreteDataloaderConfig()

    with pytest.raises(
        ValueError,
        match="2005",
    ):
        config.select_requested_times(
            slice(
                "2001",
                "2005",
            )
        )


@pytest.mark.pruned
def test_local_sampler_method_returns_none():
    config = ConcreteDataloaderConfig()
    loader = Dataloader(
        config=config,
        dataset=DummyDataset(),
        collate_fn=collate_batch,
        rank=0,
        world_size=1,
        shuffle=False,
    )

    assert loader._get_dataloader_sampler() is None


@pytest.mark.parametrize(
    (
        "rank",
        "world_size",
    ),
    [
        (
            -1,
            2,
        ),
        (
            2,
            2,
        ),
        (
            3,
            2,
        ),
    ],
)
def test_distributed_loader_rejects_invalid_rank(
    rank,
    world_size,
):
    with pytest.raises(
        ValueError,
        match="0 <= rank < world_size",
    ):
        Dataloader(
            config=ConcreteDataloaderConfig(),
            dataset=DummyDataset(
                size=8,
            ),
            collate_fn=collate_batch,
            rank=rank,
            world_size=world_size,
            shuffle=False,
        )


@pytest.mark.pruned
def test_distributed_sampler_accepts_last_valid_rank():
    loader = Dataloader(
        config=ConcreteDataloaderConfig(),
        dataset=DummyDataset(
            size=8,
        ),
        collate_fn=collate_batch,
        rank=3,
        world_size=4,
        shuffle=False,
    )

    assert loader.sampler.rank == 3
    assert loader.sampler.num_replicas == 4


@pytest.mark.pruned
@pytest.mark.parametrize(
    "shuffle",
    [
        True,
        False,
    ],
)
def test_distributed_sampler_preserves_explicit_shuffle(
    shuffle,
):
    loader = Dataloader(
        config=ConcreteDataloaderConfig(),
        dataset=DummyDataset(
            size=8,
        ),
        collate_fn=collate_batch,
        rank=0,
        world_size=2,
        shuffle=shuffle,
    )

    assert loader.sampler.shuffle is shuffle


@pytest.mark.pruned
def test_distributed_torch_loader_disables_shuffle():
    loader = Dataloader(
        config=ConcreteDataloaderConfig(),
        dataset=DummyDataset(
            size=8,
        ),
        collate_fn=collate_batch,
        rank=0,
        world_size=2,
        shuffle=True,
    )

    assert isinstance(
        loader._torch_loader.sampler,
        DistributedSampler,
    )


@pytest.mark.pruned
def test_get_sampler_forwards_keyword_arguments():
    loader = Dataloader(
        config=ConcreteDataloaderConfig(),
        dataset=DummyDataset(
            size=8,
        ),
        collate_fn=collate_batch,
        rank=0,
        world_size=2,
        shuffle=False,
    )

    sampler = loader._get_dataloader_sampler(
        seed=123,
    )

    assert sampler.seed == 123
    assert sampler.num_replicas == 2
    assert sampler.rank == 0


@pytest.mark.pruned
def test_dataloader_forwards_zero_worker_options():
    config = ConcreteDataloaderConfig(
        num_data_workers=0,
        prefetch_factor=9,
        pin_memory=True,
    )

    with patch(
        "cccma_ppp.data_modules.dataloader.DataLoader",
    ) as constructor:
        Dataloader(
            config=config,
            dataset=DummyDataset(),
            collate_fn=collate_batch,
            shuffle=False,
        )

    kwargs = constructor.call_args.kwargs

    assert kwargs["num_workers"] == 0
    assert kwargs["prefetch_factor"] is None
    assert kwargs["persistent_workers"] is False
    assert kwargs["pin_memory"] is False
    assert kwargs["timeout"] == 0


@pytest.mark.pruned
def test_dataloader_forwards_nonzero_worker_options():
    config = ConcreteDataloaderConfig(
        num_data_workers=2,
        prefetch_factor=3,
        pin_memory=True,
    )

    with patch(
        "cccma_ppp.data_modules.dataloader.DataLoader",
    ) as constructor:
        Dataloader(
            config=config,
            dataset=DummyDataset(),
            collate_fn=collate_batch,
            shuffle=False,
        )

    kwargs = constructor.call_args.kwargs

    assert kwargs["num_workers"] == 2
    assert kwargs["prefetch_factor"] == 3
    assert kwargs["persistent_workers"] is True
    assert kwargs["pin_memory"] is False
    assert kwargs["timeout"] == 60


@pytest.mark.pruned
def test_dataloader_wraps_collate_function_with_mask_options():
    config = ConcreteDataloaderConfig(
        reduce_spatial_mask=True,
    )

    with patch(
        "cccma_ppp.data_modules.dataloader.DataLoader",
    ) as constructor:
        Dataloader(
            config=config,
            dataset=DummyDataset(),
            collate_fn=collate_batch,
            shuffle=False,
            return_spatial_mask=True,
        )

    wrapped_collate = constructor.call_args.kwargs["collate_fn"]

    result = wrapped_collate(
        [
            {
                "index": 0,
                "input": torch.tensor([0.0]),
                "target": torch.tensor([1.0]),
            }
        ]
    )

    assert result["return_spatial_mask"] is True
    assert result["reduce_spatial_mask"] is True


@pytest.mark.pruned
def test_dataloader_forwards_drop_last_to_distributed_sampler():
    loader = Dataloader(
        config=ConcreteDataloaderConfig(
            drop_last=True,
        ),
        dataset=DummyDataset(
            size=9,
        ),
        collate_fn=collate_batch,
        rank=0,
        world_size=3,
        shuffle=False,
    )

    assert loader.sampler.drop_last is True


@pytest.mark.pruned
def test_set_epoch_delegates_exact_epoch_to_sampler():
    loader = Dataloader(
        config=ConcreteDataloaderConfig(),
        dataset=DummyDataset(
            size=8,
        ),
        collate_fn=collate_batch,
        rank=0,
        world_size=2,
        shuffle=False,
    )
    loader.sampler.set_epoch = Mock()

    result = loader.set_epoch(
        17,
    )

    loader.sampler.set_epoch.assert_called_once_with(
        17,
    )
    assert result is loader


@pytest.mark.pruned
def test_set_epoch_local_loader_does_not_fail_for_none_epoch():
    loader = Dataloader(
        config=ConcreteDataloaderConfig(),
        dataset=DummyDataset(),
        collate_fn=collate_batch,
        world_size=1,
        shuffle=False,
    )

    assert loader.set_epoch(None) is loader


@pytest.mark.parametrize(
    "start_batch",
    [
        -1,
        -10,
    ],
)
def test_subset_loader_rejects_negative_start(
    start_batch,
):
    loader = Dataloader(
        config=ConcreteDataloaderConfig(),
        dataset=DummyDataset(),
        collate_fn=collate_batch,
        shuffle=False,
    )

    with pytest.raises(
        ValueError,
        match="must be >= 0",
    ):
        loader.subset_loader(
            start_batch=start_batch,
        )


@pytest.mark.parametrize(
    "start_batch",
    [
        1.0,
        "1",
        None,
        [],
    ],
)
def test_subset_loader_rejects_non_integer_start(
    start_batch,
):
    loader = Dataloader(
        config=ConcreteDataloaderConfig(),
        dataset=DummyDataset(),
        collate_fn=collate_batch,
        shuffle=False,
    )

    with pytest.raises(
        TypeError,
        match="start_batch must be int",
    ):
        loader.subset_loader(
            start_batch=start_batch,
        )


@pytest.mark.pruned
def test_subset_loader_returns_iterator():
    loader = Dataloader(
        config=ConcreteDataloaderConfig(
            batch_size=2,
        ),
        dataset=DummyDataset(
            size=4,
        ),
        collate_fn=collate_batch,
        shuffle=False,
    )

    subset = loader.subset_loader(
        start_batch=1,
    )

    assert iter(subset) is subset


@pytest.mark.pruned
def test_subset_loader_is_lazy():
    loader = Dataloader(
        config=ConcreteDataloaderConfig(
            batch_size=2,
        ),
        dataset=DummyDataset(
            size=4,
        ),
        collate_fn=collate_batch,
        shuffle=False,
    )
    loader._torch_loader = MagicMock()
    loader._torch_loader.__iter__.return_value = iter(
        [
            "first",
            "second",
        ]
    )

    subset = loader.subset_loader(
        start_batch=1,
    )

    assert next(subset) == "second"


@pytest.mark.pruned
def test_empty_dataset_has_zero_batches():
    loader = Dataloader(
        config=ConcreteDataloaderConfig(
            batch_size=2,
        ),
        dataset=DummyDataset(
            size=0,
        ),
        collate_fn=collate_batch,
        shuffle=False,
    )

    assert len(loader) == 0
    assert list(loader) == []


@pytest.mark.pruned
def test_partial_batch_contains_remaining_item():
    loader = Dataloader(
        config=ConcreteDataloaderConfig(
            batch_size=2,
            drop_last=False,
        ),
        dataset=DummyDataset(
            size=5,
        ),
        collate_fn=collate_batch,
        shuffle=False,
    )

    batches = list(loader)

    assert batches[-1]["indexes"] == [
        4,
    ]


@pytest.mark.pruned
def test_loader_properties_reflect_updated_dataset_values():
    dataset = DummyDataset()
    loader = Dataloader(
        config=ConcreteDataloaderConfig(),
        dataset=dataset,
        collate_fn=collate_batch,
        shuffle=False,
    )

    dataset._input_shape = (
        7,
        8,
        9,
    )
    dataset._target_shape = (
        3,
        8,
        9,
    )
    dataset._added_features_dim = 11

    assert loader.input_shape == (
        7,
        8,
        9,
    )
    assert loader.target_shape == (
        3,
        8,
        9,
    )
    assert loader.added_features_dim == 11
