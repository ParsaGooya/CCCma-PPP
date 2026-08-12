from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
import xarray as xr

import cccma_ppp.train.dataloader as module
from cccma_ppp.train.dataloader import (
    BatchData,
    TrainDataloaderConfig,
    collate_batch,
)


def make_times(*years):
    return xr.DataArray(
        np.asarray(
            [f"{year}-01-01" for year in years],
            dtype="datetime64[ns]",
        ),
        dims=("time",),
        coords={
            "time": np.asarray(
                [f"{year}-01-01" for year in years],
                dtype="datetime64[ns]",
            )
        },
    )


def make_dataset_config(
    *,
    years=(2000, 2001, 2002, 2003),
    input_lead_times=(1, 2),
):
    available_times = make_times(*years)

    return SimpleNamespace(
        lead_time_resolution="month",
        init_time_dim="time",
        lead_time_dim="lead_time",
        get_common_time=available_times,
        available_times=available_times,
        input_lead_times=np.asarray(input_lead_times),
        fit_preprocessors=Mock(),
        load_fitted_preprocessors=Mock(),
        get_input_times=Mock(side_effect=lambda times: times),
        build_dataset=Mock(
            return_value=SimpleNamespace(
                name="dataset",
            )
        ),
        ds_operator=SimpleNamespace(
            get_weights=Mock(
                return_value="weights",
            ),
            get_input_var_metadata=Mock(
                return_value={
                    "tas": {
                        "units": "K",
                    }
                }
            ),
            get_target_var_metadata=Mock(
                return_value={
                    "pr": {
                        "units": "kg m-2 s-1",
                    }
                }
            ),
        ),
    )


def make_config_without_post_init(
    *,
    dataset_config=None,
    batch_size=2,
    time_features=None,
    train_years_slice=None,
    num_validation_years=0,
    num_data_workers=0,
    prefetch_factor=2,
    drop_last=False,
    load=False,
    reduce_spatial_mask=False,
):
    if dataset_config is None:
        dataset_config = make_dataset_config()

    config = object.__new__(TrainDataloaderConfig)

    config.dataset_config = dataset_config
    config.batch_size = batch_size
    config.time_features = time_features
    config.train_years_slice = train_years_slice
    config.num_validation_years = num_validation_years
    config.num_data_workers = num_data_workers
    config.prefetch_factor = prefetch_factor
    config.drop_last = drop_last
    config.load = load
    config.reduce_spatial_mask = reduce_spatial_mask

    config._setup = False
    config.distributed = None
    config.rank = 0
    config.world_size = 1
    config.pin_memory = False

    return config


def make_initialized_config(**kwargs):
    config = make_config_without_post_init(**kwargs)

    with patch.object(
        module.DataloaderConfigABC,
        "__init__",
        return_value=None,
        create=True,
    ):
        config.__post_init__()

    return config


@pytest.fixture(autouse=True)
def reset_shared_batch_masks():
    original_input_mask = getattr(
        BatchData,
        "_shared_input_mask",
        None,
    )
    original_target_mask = getattr(
        BatchData,
        "_shared_target_mask",
        None,
    )

    BatchData._shared_input_mask = None
    BatchData._shared_target_mask = None

    yield

    BatchData._shared_input_mask = original_input_mask
    BatchData._shared_target_mask = original_target_mask


class TestBatchData:
    @pytest.mark.pruned
    def test_replaces_input_and_target_nan_values(self):
        batch = BatchData(
            input=torch.tensor(
                [
                    1.0,
                    float("nan"),
                ]
            ),
            target=torch.tensor(
                [
                    float("nan"),
                    2.0,
                ]
            ),
        )

        torch.testing.assert_close(
            batch.input,
            torch.tensor(
                [
                    1.0,
                    0.0,
                ]
            ),
        )
        torch.testing.assert_close(
            batch.target,
            torch.tensor(
                [
                    0.0,
                    2.0,
                ]
            ),
        )

    @pytest.mark.pruned
    def test_without_spatial_masks(self):
        batch = BatchData(
            input=torch.ones(
                2,
                3,
            ),
            target=torch.ones(
                2,
                1,
            ),
            return_spatial_mask=False,
        )

        assert batch.input_mask is None
        assert batch.target_mask is None

    @pytest.mark.pruned
    def test_reduced_spatial_masks(self):
        input_data = torch.tensor(
            [
                [
                    1.0,
                    float("nan"),
                    3.0,
                ],
                [
                    4.0,
                    5.0,
                    6.0,
                ],
            ]
        )
        target_data = torch.tensor(
            [
                [
                    1.0,
                    2.0,
                ],
                [
                    float("nan"),
                    4.0,
                ],
            ]
        )

        batch = BatchData(
            input=input_data,
            target=target_data,
            return_spatial_mask=True,
            reduce_spatial_mask=True,
        )

        torch.testing.assert_close(
            batch.input_mask,
            torch.tensor(
                [
                    True,
                    False,
                    True,
                ]
            ),
        )
        torch.testing.assert_close(
            batch.target_mask,
            torch.tensor(
                [
                    False,
                    True,
                ]
            ),
        )

    @pytest.mark.pruned
    def test_unreduced_spatial_masks(self):
        input_data = torch.tensor(
            [
                [
                    1.0,
                    float("nan"),
                ],
                [
                    float("nan"),
                    4.0,
                ],
            ]
        )
        target_data = torch.tensor(
            [
                [
                    float("nan"),
                    2.0,
                ],
                [
                    3.0,
                    4.0,
                ],
            ]
        )

        batch = BatchData(
            input=input_data,
            target=target_data,
            return_spatial_mask=True,
            reduce_spatial_mask=False,
        )

        torch.testing.assert_close(
            batch.input_mask,
            torch.tensor(
                [
                    [
                        True,
                        False,
                    ],
                    [
                        False,
                        True,
                    ],
                ]
            ),
        )
        torch.testing.assert_close(
            batch.target_mask,
            torch.tensor(
                [
                    [
                        False,
                        True,
                    ],
                    [
                        True,
                        True,
                    ],
                ]
            ),
        )

    def test_reduced_masks_are_shared(self):
        first = BatchData(
            input=torch.tensor(
                [
                    [
                        1.0,
                        float("nan"),
                    ],
                    [
                        2.0,
                        3.0,
                    ],
                ]
            ),
            target=torch.tensor(
                [
                    [
                        1.0,
                        2.0,
                    ],
                    [
                        3.0,
                        4.0,
                    ],
                ]
            ),
            return_spatial_mask=True,
            reduce_spatial_mask=True,
        )

        second = BatchData(
            input=torch.ones(
                2,
                2,
            ),
            target=torch.ones(
                2,
                2,
            ),
            return_spatial_mask=True,
            reduce_spatial_mask=True,
        )

        assert first.input_mask.data_ptr() == second.input_mask.data_ptr()
        assert first.target_mask.data_ptr() == second.target_mask.data_ptr()

        torch.testing.assert_close(
            second.input_mask,
            torch.tensor(
                [
                    True,
                    False,
                ]
            ),
        )

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            torch.device("cpu"),
        ],
    )
    def test_to_device_moves_all_tensors(
        self,
        device,
    ):
        batch = BatchData(
            input=torch.ones(
                2,
                3,
            ),
            target=torch.ones(
                2,
                1,
            ),
            added_features=torch.ones(
                2,
                4,
            ),
            return_spatial_mask=True,
            reduce_spatial_mask=False,
        )

        result = batch.to_device(device)

        assert result is batch
        assert batch.input.device.type == "cpu"
        assert batch.target.device.type == "cpu"
        assert batch.added_features.device.type == "cpu"
        assert batch.input_mask.device.type == "cpu"
        assert batch.target_mask.device.type == "cpu"

    def test_to_device_handles_optional_values(self):
        batch = BatchData(
            input=torch.ones(
                2,
                3,
            ),
            target=torch.ones(
                2,
                1,
            ),
            added_features=None,
            return_spatial_mask=False,
        )

        result = batch.to_device("cpu")

        assert result is batch
        assert batch.added_features is None
        assert batch.input_mask is None
        assert batch.target_mask is None

    @pytest.mark.pruned
    def test_metadata_is_preserved(self):
        metadata = [
            {
                "time": 2000,
            },
            {
                "time": 2001,
            },
        ]

        batch = BatchData(
            input=torch.ones(
                2,
                3,
            ),
            target=torch.ones(
                2,
                1,
            ),
            metadata=metadata,
        )

        assert batch.metadata is metadata


class TestCollateBatch:
    def make_sample(
        self,
        value,
        *,
        added_features=None,
    ):
        return {
            "input": torch.tensor(
                [
                    float(value),
                    float(value + 1),
                ]
            ),
            "target": torch.tensor(
                [
                    float(value + 2),
                ]
            ),
            "added_features": (
                None
                if added_features is None
                else torch.tensor(
                    added_features,
                    dtype=torch.float32,
                )
            ),
        }

    @pytest.mark.pruned
    def test_basic_collation(self):
        result = collate_batch(
            [
                self.make_sample(1),
                self.make_sample(4),
            ]
        )

        assert isinstance(
            result,
            BatchData,
        )
        assert result.input.shape == (
            2,
            2,
        )
        assert result.target.shape == (
            2,
            1,
        )
        assert result.added_features is None
        assert result.metadata is None

    def test_collates_added_features(self):
        result = collate_batch(
            [
                self.make_sample(
                    1,
                    added_features=[
                        10,
                        11,
                    ],
                ),
                self.make_sample(
                    4,
                    added_features=[
                        12,
                        13,
                    ],
                ),
            ]
        )

        torch.testing.assert_close(
            result.added_features,
            torch.tensor(
                [
                    [
                        10.0,
                        11.0,
                    ],
                    [
                        12.0,
                        13.0,
                    ],
                ]
            ),
        )

    def test_collates_metadata(self):
        samples = [
            (
                self.make_sample(1),
                {
                    "time": 2000,
                },
            ),
            (
                self.make_sample(4),
                {
                    "time": 2001,
                },
            ),
        ]

        result = collate_batch(samples)

        assert result.metadata == [
            {
                "time": 2000,
            },
            {
                "time": 2001,
            },
        ]

    @pytest.mark.pruned
    def test_forwards_spatial_mask_arguments(self):
        result = collate_batch(
            [
                self.make_sample(1),
                self.make_sample(4),
            ],
            return_spatial_mask=True,
            reduce_spatial_mask=False,
        )

        assert result.return_spatial_mask is True
        assert result.reduce_spatial_mask is False
        assert result.input_mask.shape == (
            2,
            2,
        )
        assert result.target_mask.shape == (
            2,
            1,
        )

    @pytest.mark.pruned
    def test_nan_values_are_cleaned_after_collation(self):
        sample = self.make_sample(1)
        sample["input"][0] = float("nan")
        sample["target"][0] = float("nan")

        result = collate_batch(
            [
                sample,
            ]
        )

        assert torch.isfinite(result.input).all()
        assert torch.isfinite(result.target).all()


class TestTrainDataloaderConfigPostInit:
    @pytest.mark.pruned
    def test_constructs_added_time_features(self):
        dataset_config = make_dataset_config()
        config = make_config_without_post_init(
            dataset_config=dataset_config,
            time_features=[
                "year",
                "lead_time",
            ],
        )
        time_features = object()

        with (
            patch.object(
                module.DataloaderConfigABC,
                "__init__",
                return_value=None,
                create=True,
            ) as parent_post_init,
            patch.object(
                module,
                "AddedTimeFeatures",
                return_value=time_features,
            ) as constructor,
        ):
            config.__post_init__()

        parent_post_init.assert_called_once_with()
        constructor.assert_called_once_with(
            dataset_config,
            [
                "year",
                "lead_time",
            ],
        )
        assert config.time_features is time_features

    @pytest.mark.pruned
    def test_default_train_times_use_available_times(self):
        config = make_initialized_config(
            train_years_slice=None,
        )

        xr.testing.assert_identical(
            config.train_times,
            config.dataset_config.available_times,
        )

    @pytest.mark.pruned
    def test_requested_train_slice_is_converted_to_strings(self):
        config = make_config_without_post_init(
            train_years_slice=[
                2001,
                2003,
            ]
        )
        selected = make_times(
            2001,
            2002,
            2003,
        )

        config.select_requested_times = Mock(return_value=selected)

        with patch.object(
            module.DataloaderConfigABC,
            "__init__",
            return_value=None,
            create=True,
        ):
            config.__post_init__()

        assert config.train_years_slice == slice(
            "2001",
            "2003",
        )
        config.select_requested_times.assert_called_once_with(
            requested_slice=slice(
                "2001",
                "2003",
            )
        )
        assert config.train_times is selected

    @pytest.mark.pruned
    def test_validation_times_are_split_from_end(self):
        config = make_initialized_config(
            dataset_config=make_dataset_config(
                years=(
                    2000,
                    2001,
                    2002,
                    2003,
                )
            ),
            num_validation_years=2,
        )

        np.testing.assert_array_equal(
            config.train_times.dt.year.values,
            np.asarray(
                [
                    2000,
                    2001,
                ]
            ),
        )
        np.testing.assert_array_equal(
            config.validation_times.dt.year.values,
            np.asarray(
                [
                    2002,
                    2003,
                ]
            ),
        )

    @pytest.mark.pruned
    def test_zero_validation_years_does_not_create_validation_times(self):
        config = make_initialized_config(
            num_validation_years=0,
        )

        assert not hasattr(
            config,
            "validation_times",
        )

    @pytest.mark.pruned
    def test_available_times_delegates_to_dataset_config(self):
        dataset_config = make_dataset_config()
        config = make_config_without_post_init(
            dataset_config=dataset_config,
        )

        assert config.available_times is dataset_config.available_times


class TestSetupDistributed:
    def make_distributed(
        self,
        *,
        root=True,
        distributed=False,
        rank=0,
        world_size=1,
    ):
        return SimpleNamespace(
            rank=rank,
            world_size=world_size,
            distributed=distributed,
            is_root=Mock(return_value=root),
            barrier=Mock(),
        )

    @pytest.mark.pruned
    def test_root_fits_preprocessors(self):
        config = make_initialized_config()
        distributed = self.make_distributed(
            root=True,
            distributed=False,
        )

        config.setup_distributed(distributed)

        config.dataset_config.fit_preprocessors.assert_called_once_with(
            config.train_times,
            save=True,
        )
        distributed.barrier.assert_called_once_with()
        assert config._setup is True

    def test_non_root_does_not_fit_preprocessors(self):
        config = make_initialized_config()
        distributed = self.make_distributed(
            root=False,
            distributed=False,
            rank=1,
        )

        config.setup_distributed(distributed)

        config.dataset_config.fit_preprocessors.assert_not_called()
        distributed.barrier.assert_called_once_with()

    def test_distributed_mode_loads_preprocessors(self):
        config = make_initialized_config()
        distributed = self.make_distributed(
            root=True,
            distributed=True,
            world_size=2,
        )

        config.setup_distributed(distributed)

        config.dataset_config.load_fitted_preprocessors.assert_called_once_with(
            load_dir=None
        )
        assert config.pin_memory is True

    def test_explicit_load_path_skips_fit_and_loads(self):
        config = make_initialized_config()
        distributed = self.make_distributed(
            root=True,
            distributed=False,
        )
        load_path = Path("/tmp/preprocessors")

        config.setup_distributed(
            distributed,
            load_path=load_path,
        )

        config.dataset_config.fit_preprocessors.assert_not_called()
        config.dataset_config.load_fitted_preprocessors.assert_called_once_with(
            load_dir=load_path
        )

    @pytest.mark.pruned
    def test_records_distributed_metadata(self):
        config = make_initialized_config()
        distributed = self.make_distributed(
            root=False,
            distributed=True,
            rank=3,
            world_size=8,
        )

        config.setup_distributed(distributed)

        assert config.distributed is distributed
        assert config.rank == 3
        assert config.world_size == 8


class TestBuildTrainLoader:
    def test_requires_setup(self):
        config = make_initialized_config()

        with pytest.raises(
            RuntimeError,
            match="setup for distributed training",
        ):
            config.build_train_loader()

    @pytest.mark.pruned
    def test_builds_dataset_and_dataloader(self):
        config = make_initialized_config(
            load=True,
        )
        config._setup = True
        config.rank = 0
        config.world_size = 1

        train_mask = object()
        loader = object()

        with (
            patch.object(
                module,
                "_create_train_mask",
                return_value=train_mask,
            ) as create_mask,
            patch.object(
                module,
                "Dataloader",
                return_value=loader,
            ) as constructor,
        ):
            result = config.build_train_loader(
                return_metadata=True,
                return_spatial_mask=True,
                shuffle=False,
            )

        assert result is loader

        config.dataset_config.get_input_times.assert_called_once_with(
            config.train_times
        )
        create_mask.assert_called_once_with(
            init_times=config.train_times,
            lead_times=config.dataset_config.input_lead_times,
        )

        config.dataset_config.build_dataset.assert_called_once_with(
            times=config.train_times,
            mask=train_mask,
            time_features=config.time_features,
            return_metadata=True,
            load=True,
        )

        constructor.assert_called_once_with(
            dataset=config.dataset_config.build_dataset.return_value,
            config=config,
            collate_fn=collate_batch,
            rank=0,
            shuffle=False,
            world_size=1,
            return_spatial_mask=True,
        )


class TestBuildValidationLoader:
    def test_requires_setup(self):
        config = make_initialized_config(
            num_validation_years=1,
        )

        with pytest.raises(
            RuntimeError,
            match="setup for distributed training",
        ):
            config.build_validation_loader()

    def test_warns_when_validation_is_disabled(self):
        config = make_initialized_config(
            num_validation_years=0,
        )
        config._setup = True

        with pytest.warns(
            UserWarning,
            match="Validation dataoader could not be built",
        ):
            result = config.build_validation_loader(supress_error=True)

        assert result is None

    @pytest.mark.pruned
    def test_raises_when_validation_is_disabled(self):
        config = make_initialized_config(
            num_validation_years=0,
        )
        config._setup = True

        with pytest.raises(
            RuntimeError,
            match="Validation dataoader could not be built",
        ):
            config.build_validation_loader(supress_error=False)

    def test_builds_validation_loader(self):
        config = make_initialized_config(
            dataset_config=make_dataset_config(
                years=(
                    2000,
                    2001,
                    2002,
                )
            ),
            num_validation_years=1,
            load=True,
        )
        config._setup = True
        config.rank = 0
        config.world_size = 1

        validation_mask = object()
        loader = object()

        with (
            patch.object(
                module,
                "_create_train_mask",
                return_value=validation_mask,
            ) as create_mask,
            patch.object(
                module,
                "Dataloader",
                return_value=loader,
            ) as constructor,
        ):
            result = config.build_validation_loader(
                return_metadata=True,
                return_spatial_mask=True,
                shuffle=False,
            )

        assert result is loader

        config.dataset_config.get_input_times.assert_called_once_with(
            config.validation_times
        )

        create_mask.assert_called_once_with(
            init_times=config.validation_times,
            lead_times=config.dataset_config.input_lead_times,
        )

        config.dataset_config.build_dataset.assert_called_once_with(
            times=config.validation_times,
            time_features=config.time_features,
            mask=validation_mask,
            return_metadata=True,
            load=True,
        )

        constructor.assert_called_once_with(
            dataset=config.dataset_config.build_dataset.return_value,
            config=config,
            collate_fn=collate_batch,
            rank=0,
            shuffle=False,
            world_size=1,
            return_spatial_mask=True,
        )


class TestWeightsAndMetadata:
    def make_distributed(
        self,
        *,
        root=True,
    ):
        return SimpleNamespace(
            is_root=Mock(return_value=root),
            barrier=Mock(),
        )

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "root",
        [
            True,
            False,
        ],
    )
    def test_get_weights(
        self,
        root,
    ):
        config = make_initialized_config()
        distributed = self.make_distributed(root=root)
        config.distributed = distributed

        weights_config = object()

        result = config.get_weights(weights_config)

        assert result == "weights"
        config.dataset_config.ds_operator.get_weights.assert_called_once_with(
            weights_config,
            save=root,
        )
        distributed.barrier.assert_called_once_with()

    @pytest.mark.pruned
    def test_get_weights_accepts_none(self):
        config = make_initialized_config()
        distributed = self.make_distributed()
        config.distributed = distributed

        config.get_weights()

        config.dataset_config.ds_operator.get_weights.assert_called_once_with(
            None,
            save=True,
        )

    @pytest.mark.pruned
    def test_input_variable_metadata(self):
        config = make_initialized_config()

        result = config.input_var_metadata

        assert result == {
            "tas": {
                "units": "K",
            }
        }
        config.dataset_config.ds_operator.get_input_var_metadata.assert_called_once_with()

    @pytest.mark.pruned
    def test_target_variable_metadata(self):
        config = make_initialized_config()

        result = config.target_var_metadata

        assert result == {
            "pr": {
                "units": "kg m-2 s-1",
            }
        }
        config.dataset_config.ds_operator.get_target_var_metadata.assert_called_once_with()
