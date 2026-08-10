from unittest.mock import Mock
import numpy as np
import pytest
import torch

from cccma_ppp.train.dataloader import (
    BatchData,
    TrainDataloaderConfig,
    collate_batch,
)


class DummySet:
    def __init__(self, length=2):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return {
            "input": torch.ones(2),
            "target": torch.zeros(2),
            "added_features": None,
        }


class DummyOperator:
    def __init__(self, config):
        self.config = config

    def fit_preprocessors(
        self,
        train_years,
        save=False,
        save_path=None,
        save_name=None,
    ):
        self.config.fit_calls.append(
            {
                "train_years": np.asarray(train_years),
                "save": save,
                "save_path": save_path,
                "save_name": save_name,
            }
        )
        self.config._fitted_preprocessors = True
        return self

    def load_fitted_preprocessors(
        self,
        load_dir=None,
    ):
        self.config.load_calls.append(load_dir)
        self.config._fitted_preprocessors = True
        return self

    def get_weights(self, *args, **kwargs):
        return "w"

    def get_input_var_metadata(self):
        return "i"

    def get_target_var_metadata(self):
        return "t"


class DummyDatasetConfig:
    def __init__(self):
        self.available_times = np.arange(
            2000,
            2005,
        )
        self.available_train_time = self.available_times

        self.input_lead_months = np.arange(
            1,
            13,
        )
        self.lead_months = np.arange(
            1,
            13,
        )

        self.fit_calls = []
        self.load_calls = []
        self.build_calls = []
        self._fitted_preprocessors = False

        self._operator = DummyOperator(self)

    def fit_preprocessors(
        self,
        years,
        save=False,
        save_path=None,
        save_name=None,
        **kwargs,
    ):
        self.fit_calls.append(
            {
                "train_years": np.asarray(years),
                "save": save,
                "save_path": save_path,
                "save_name": save_name,
                **kwargs,
            }
        )
        self._fitted_preprocessors = True
        return self

    def load_fitted_preprocessors(
        self,
        load_dir=None,
        **kwargs,
    ):
        self.load_calls.append(load_dir)
        self._fitted_preprocessors = True
        return self

    def build_dataset(self, **kwargs):
        self.build_calls.append(kwargs)
        return DummySet()

    @property
    def ds_operator(self):
        return self._operator


class DummyDistributed:
    rank = 0
    local_rank = 0
    world_size = 1
    distributed = False
    device = "cpu"

    def is_root(self):
        return True

    def barrier(self):
        return None


class DistributedContext(DummyDistributed):
    distributed = True
    world_size = 2


class NonRootDistributed(DummyDistributed):
    rank = 1
    local_rank = 1
    world_size = 2
    distributed = True

    def is_root(self):
        return False


def make_config(**kwargs):
    values = {
        "dataset_config": DummyDatasetConfig(),
        "batch_size": 2,
    }
    values.update(kwargs)
    return TrainDataloaderConfig(**values)


def setup_config(config=None, distributed=None):
    config = config or make_config()
    distributed = distributed or DummyDistributed()
    config.setup_distributed(distributed)
    return config


def make_batch(
    include_features=False,
    include_metadata=False,
):
    data = []

    for year in (2000, 2001):
        item = {
            "input": torch.ones(2),
            "target": torch.zeros(2),
            "added_features": (torch.ones(3) if include_features else None),
        }

        if include_metadata:
            data.append((item, {"year": year}))
        else:
            data.append(item)

    return data


def test_batchdata_basic_nan_cleanup():
    x = torch.tensor([[1.0, float("nan")]])
    y = torch.tensor([[float("nan"), 2.0]])

    batch = BatchData(x, y)

    assert not torch.isnan(batch.input).any()
    assert not torch.isnan(batch.target).any()
    torch.testing.assert_close(
        batch.input,
        torch.tensor([[1.0, 0.0]]),
    )
    torch.testing.assert_close(
        batch.target,
        torch.tensor([[0.0, 2.0]]),
    )


def test_batchdata_without_masks():
    batch = BatchData(
        torch.ones((2, 2)),
        torch.zeros((2, 2)),
    )

    assert batch.input_mask is None
    assert batch.target_mask is None


def test_batchdata_with_reduced_masks():
    x = torch.tensor(
        [
            [1.0, float("nan")],
            [1.0, 2.0],
        ]
    )
    y = torch.tensor(
        [
            [1.0, 2.0],
            [float("nan"), 2.0],
        ]
    )

    batch = BatchData(
        x,
        y,
        return_spatial_mask=True,
        reduce_spatial_mask=True,
    )

    assert isinstance(batch.input, torch.Tensor)
    assert isinstance(batch.target, torch.Tensor)
    assert isinstance(batch.input_mask, torch.Tensor)
    assert isinstance(batch.target_mask, torch.Tensor)

    torch.testing.assert_close(
        batch.input,
        torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 2.0],
            ]
        ),
    )
    torch.testing.assert_close(
        batch.target,
        torch.tensor(
            [
                [1.0, 2.0],
                [0.0, 2.0],
            ]
        ),
    )
    torch.testing.assert_close(
        batch.input_mask,
        torch.tensor([True, False]),
    )
    torch.testing.assert_close(
        batch.target_mask,
        torch.tensor([False, True]),
    )


def test_batchdata_with_unreduced_masks():
    x = torch.tensor(
        [
            [1.0, float("nan")],
            [1.0, 2.0],
        ]
    )
    y = torch.tensor(
        [
            [1.0, 2.0],
            [float("nan"), 2.0],
        ]
    )

    batch = BatchData(
        x,
        y,
        return_spatial_mask=True,
        reduce_spatial_mask=False,
    )

    torch.testing.assert_close(
        batch.input_mask,
        torch.tensor(
            [
                [True, False],
                [True, True],
            ]
        ),
    )
    torch.testing.assert_close(
        batch.target_mask,
        torch.tensor(
            [
                [True, True],
                [False, True],
            ]
        ),
    )


def test_batchdata_to_device_without_masks():
    batch = BatchData(
        torch.ones((2, 2)),
        torch.ones((2, 2)),
    )

    result = batch.to_device("cpu")

    assert result is batch
    assert batch.input.device.type == "cpu"
    assert batch.target.device.type == "cpu"
    assert batch.input_mask is None
    assert batch.target_mask is None


def test_batchdata_to_device_with_masks():
    batch = BatchData(
        torch.ones((2, 2)),
        torch.ones((2, 2)),
        return_spatial_mask=True,
    )

    result = batch.to_device("cpu")

    assert result is batch
    assert batch.input.device.type == "cpu"
    assert batch.target.device.type == "cpu"
    assert batch.input_mask.device.type == "cpu"
    assert batch.target_mask.device.type == "cpu"


def test_batchdata_to_device_with_features():
    batch = BatchData(
        torch.ones((2, 2)),
        torch.ones((2, 2)),
        added_features=torch.ones((2, 3)),
    )

    result = batch.to_device("cpu")

    assert result is batch
    assert batch.added_features.device.type == "cpu"
    assert batch.added_features.shape == (2, 3)


def test_batchdata_added_features_none():
    batch = BatchData(
        torch.ones((2, 2)),
        torch.ones((2, 2)),
    )

    assert batch.added_features is None


def test_batchdata_metadata_preserved():
    metadata = [{"year": 2000}]

    batch = BatchData(
        torch.ones((1, 2)),
        torch.ones((1, 2)),
        metadata=metadata,
    )

    assert batch.metadata is metadata


def test_collate_basic():
    result = collate_batch(make_batch())

    assert isinstance(result, BatchData)
    assert result.input.shape == (2, 2)
    assert result.target.shape == (2, 2)
    assert result.added_features is None
    assert result.metadata is None


def test_collate_with_features():
    result = collate_batch(make_batch(include_features=True))

    assert isinstance(result, BatchData)
    assert result.added_features is not None
    assert result.added_features.shape == (2, 3)


def test_collate_with_unreduced_masks():
    result = collate_batch(
        make_batch(),
        return_spatial_mask=True,
        reduce_spatial_mask=False,
    )

    assert result.input_mask.shape == (2, 2)
    assert result.target_mask.shape == (2, 2)
    assert torch.all(result.input_mask)
    assert torch.all(result.target_mask)


def test_collate_metadata_tuple_branch():
    result = collate_batch(make_batch(include_metadata=True))

    assert isinstance(result, BatchData)
    assert result.metadata == [
        {"year": 2000},
        {"year": 2001},
    ]


def test_collate_metadata_and_features():
    result = collate_batch(
        make_batch(
            include_features=True,
            include_metadata=True,
        )
    )

    assert result.added_features.shape == (2, 3)
    assert len(result.metadata) == 2


def test_config_default_train_years():
    config = make_config()

    np.testing.assert_array_equal(
        config.train_years,
        np.arange(2000, 2005),
    )


def test_config_validation_split():
    config = make_config(
        num_validation_years=2,
    )

    np.testing.assert_array_equal(
        config.train_years,
        np.array([2000, 2001, 2002]),
    )
    np.testing.assert_array_equal(
        config.validation_years,
        np.array([2003, 2004]),
    )


def test_config_workers_zero_sets_prefetch_none():
    config = make_config(
        num_data_workers=0,
        prefetch_factor=4,
    )

    assert config.prefetch_factor is None


def test_config_workers_positive_preserves_prefetch():
    config = make_config(
        num_data_workers=2,
        prefetch_factor=4,
    )

    assert config.prefetch_factor == 4


def test_config_invalid_train_years():
    with pytest.raises(
        ValueError,
        match="requested train years",
    ):
        make_config(
            train_years=(1900, 1901),
        )


def test_config_custom_train_years_valid():
    config = make_config(
        train_years=(2000, 2002),
    )

    np.testing.assert_array_equal(
        config.train_years,
        np.array([2000, 2001, 2002]),
    )


def test_config_custom_train_and_validation_split_rejects_overlap():
    with pytest.raises(
        ValueError,
        match="requested train years",
    ):
        make_config(
            train_years=(2000, 2003),
            num_validation_years=2,
        )


def test_config_custom_train_and_validation_split_valid():
    config = make_config(
        train_years=(2000, 2002),
        num_validation_years=2,
    )

    np.testing.assert_array_equal(
        config.train_years,
        np.array([2000, 2001, 2002]),
    )
    np.testing.assert_array_equal(
        config.validation_years,
        np.array([2003, 2004]),
    )


def test_train_years_subset():
    config = make_config(
        train_years=(2001, 2003),
    )

    np.testing.assert_array_equal(
        config.train_years,
        np.array([2001, 2002, 2003]),
    )


def test_available_times_without_validation():
    config = make_config()

    np.testing.assert_array_equal(
        config.available_times,
        np.arange(2000, 2005),
    )


def test_available_times_with_validation():
    config = make_config(
        num_validation_years=2,
    )

    np.testing.assert_array_equal(
        config.available_times,
        np.array([2000, 2001, 2002]),
    )


def test_setup_distributed_root():
    config = make_config()

    config.setup_distributed(DummyDistributed())

    assert config._setup is True
    assert config.rank == 0
    assert config.world_size == 1


def test_setup_distributed_non_root():
    config = make_config()

    config.setup_distributed(NonRootDistributed())

    assert config._setup is True
    assert config.rank == 1
    assert config.world_size == 2


def test_setup_distributed_distributed_mode():
    config = make_config()

    config.setup_distributed(DistributedContext())

    assert config._setup is True
    assert config.world_size == 2


def test_setup_distributed_fits_preprocessors():
    dataset_config = DummyDatasetConfig()
    config = make_config(
        dataset_config=dataset_config,
    )

    config.setup_distributed(DummyDistributed())

    assert dataset_config.fit_calls


def test_build_train_loader_requires_setup():
    config = make_config()

    with pytest.raises(RuntimeError):
        config.build_train_loader()


def test_build_validation_loader_requires_setup():
    config = make_config(
        num_validation_years=1,
    )

    with pytest.raises(RuntimeError):
        config.build_validation_loader()


def test_build_train_loader_success():
    config = setup_config()

    loader = config.build_train_loader()

    assert loader is not None
    assert len(loader.dataset) == 2


def test_build_train_loader_passes_years():
    config = setup_config()

    config.build_train_loader()

    call = config.dataset_config.build_calls[-1]

    np.testing.assert_array_equal(
        call["years"],
        config.train_years,
    )


def test_build_train_loader_sets_dataloader_length():
    config = setup_config()

    loader = config.build_train_loader()

    assert len(loader) == 1


def test_build_validation_loader_disabled():
    config = setup_config(
        make_config(
            num_validation_years=0,
        )
    )

    with pytest.warns(UserWarning):
        loader = config.build_validation_loader()

    assert loader is None


def test_build_validation_loader_success():
    config = setup_config(
        make_config(
            num_validation_years=1,
        )
    )

    loader = config.build_validation_loader()

    assert loader is not None
    assert len(loader.dataset) == 2


def test_build_validation_loader_passes_years():
    config = setup_config(
        make_config(
            num_validation_years=2,
        )
    )

    config.build_validation_loader()

    call = config.dataset_config.build_calls[-1]

    np.testing.assert_array_equal(
        call["years"],
        config.validation_years,
    )


def test_get_weights_default():
    config = make_config()
    config.distributed = Mock()
    config.distributed.is_root.return_value = True

    assert config.get_weights() == "w"


def test_get_weights_with_argument():
    config = make_config()
    config.distributed = Mock()
    config.distributed.is_root.return_value = True
    weights_config = object()

    assert config.get_weights(weights_config) == "w"


def test_input_var_metadata():
    config = make_config()

    assert config.input_var_metadata == "i"


def test_target_var_metadata():
    config = make_config()

    assert config.target_var_metadata == "t"
