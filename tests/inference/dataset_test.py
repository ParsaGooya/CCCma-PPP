from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import xarray as xr

import cccma_ppp.inference.dataset as module
from cccma_ppp.data_modules.dataset.dataset_abc import DatasetABC
from cccma_ppp.inference.dataset import (
    InferenceDataset,
    InferenceDatasetConfig,
    _from_train,
)


InferenceDatasetConfig.__abstractmethods__ = frozenset()


class DummyPipeline:
    def __init__(self, name="pipeline"):
        self.name = name


class DummyDataConfig:
    def __init__(
        self,
        name="data",
        *,
        years=(2000, 2001, 2002),
        ensemble_mean=False,
        ensemble_list=None,
    ):
        self.names = [name]
        self.year_range = np.asarray(years)
        self.ensemble_mean = ensemble_mean
        self.ensemble_list = ensemble_list
        self.preprocessing_pipeline = DummyPipeline(name)

        self.info = SimpleNamespace(
            coords={
                "year": xr.DataArray(
                    np.asarray(years),
                    dims=("year",),
                ),
                "lead_time": xr.DataArray(
                    np.arange(1, 13),
                    dims=("lead_time",),
                ),
                "ensembles": xr.DataArray(
                    np.asarray([0, 1]),
                    dims=("ensembles",),
                ),
                "lat": xr.DataArray(
                    np.asarray([0, 1]),
                    dims=("lat",),
                ),
                "lon": xr.DataArray(
                    np.asarray([10, 20, 30]),
                    dims=("lon",),
                ),
            },
            sizes={
                "year": len(years),
                "lead_time": 12,
                "ensembles": 2,
                "lat": 2,
                "lon": 3,
            },
        )

        self.list_paths = [f"{name}.nc"]
        self.paths = [f"{name}.nc"]
        self.concat_dim = None
        self.rename_dict = {}


class FakeTimeFeatures:
    def __init__(self, result=None):
        self.result = result
        self.calls = []

    def __call__(
        self,
        selection,
        input_data,
    ):
        self.calls.append(
            (
                selection,
                input_data,
            )
        )
        return self.result


def bare_config(
    *,
    model=None,
    condition=None,
    condition_method=None,
    lead_months=None,
):
    config = object.__new__(InferenceDatasetConfig)

    config.model = model
    config.condition = condition
    config.condition_method = condition_method
    config.lead_months = lead_months
    config._effective_condition = None
    config._fitted_preprocessors = True

    return config


def dataset_config_stub(
    *,
    model=None,
    condition=None,
    using_model_as_condition=False,
):
    return SimpleNamespace(
        model=model,
        condition=condition,
        _using_model_data_as_condition=using_model_as_condition,
    )


def bare_dataset(
    *,
    config=None,
    sample_coords=None,
    time_features=None,
    return_metadata=False,
    load=False,
):
    dataset = object.__new__(InferenceDataset)

    dataset.config = config or dataset_config_stub(
        model=DummyDataConfig("model"),
    )
    dataset.requested_years = np.asarray([2000])
    dataset.time_features = (
        time_features if time_features is not None else FakeTimeFeatures(None)
    )
    dataset.return_metadata = return_metadata
    dataset.load = load
    dataset.sample_coords = (
        sample_coords
        if sample_coords is not None
        else {
            "year": np.asarray([2000.0]),
            "lead_time": np.asarray([1.0]),
        }
    )

    return dataset


def data_array(
    values,
    channel_names=None,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    if channel_names is None:
        channel_names = [f"channel_{index}" for index in range(len(values))]

    return xr.DataArray(
        values,
        dims=("channels",),
        coords={
            "channels": channel_names,
        },
    )


def install_item_sources(
    dataset,
    *,
    model=None,
    condition=None,
):
    dataset._index_model_dataset = MagicMock(
        return_value=model,
    )
    dataset._index_condition_dataset = MagicMock(
        return_value=condition,
    )


def make_train_config(
    *,
    observation=None,
    effective_condition=None,
    using_model_as_condition=False,
    model=None,
    condition=None,
    condition_method="static",
    lead_months=None,
):
    return SimpleNamespace(
        observation=observation,
        effective_condition=effective_condition,
        _using_model_data_as_condition=using_model_as_condition,
        model=model,
        condition=condition,
        condition_method=condition_method,
        lead_months=(np.asarray([1, 2]) if lead_months is None else lead_months),
    )


def test_post_init_calls_dataset_abc_init():
    dataset = bare_dataset()

    with patch.object(
        DatasetABC,
        "__init__",
        return_value=None,
    ) as parent_init:
        dataset.__post_init__()

    parent_init.assert_called_once_with()


@pytest.mark.parametrize(
    (
        "model",
        "condition",
        "expected_name",
    ),
    [
        (
            DummyDataConfig("model"),
            DummyDataConfig("condition"),
            "model",
        ),
        (
            DummyDataConfig("model"),
            None,
            "model",
        ),
        (
            None,
            DummyDataConfig("condition"),
            "condition",
        ),
        (
            None,
            None,
            None,
        ),
    ],
)
def test_effective_input(
    model,
    condition,
    expected_name,
):
    config = bare_config(
        model=model,
        condition=condition,
    )

    result = config.effective_input

    if expected_name is None:
        assert result is None
    else:
        assert result.names == [expected_name]


def test_ds_operator_builds_operator():
    config = bare_config(
        model=DummyDataConfig("model"),
    )

    with patch.object(
        module,
        "DatasetOperator",
    ) as constructor:
        result = config.ds_operator

    constructor.assert_called_once_with(config)
    assert result is constructor.return_value


def test_available_times_model_only():
    config = bare_config(
        model=DummyDataConfig(
            "model",
            years=(2000, 2001, 2002),
        ),
    )

    np.testing.assert_array_equal(
        config.available_times,
        [2000, 2001, 2002],
    )


def test_available_times_condition_only():
    config = bare_config(
        condition=DummyDataConfig(
            "condition",
            years=(2001, 2002, 2003),
        ),
    )

    np.testing.assert_array_equal(
        config.available_times,
        [2001, 2002, 2003],
    )


def test_available_times_intersects_model_and_condition():
    config = bare_config(
        model=DummyDataConfig(
            "model",
            years=(2000, 2001, 2002),
        ),
        condition=DummyDataConfig(
            "condition",
            years=(2001, 2002, 2003),
        ),
    )

    np.testing.assert_array_equal(
        config.available_times,
        [2001, 2002],
    )


def test_available_times_empty_intersection():
    config = bare_config(
        model=DummyDataConfig(
            "model",
            years=(2000, 2001),
        ),
        condition=DummyDataConfig(
            "condition",
            years=(2010, 2011),
        ),
    )

    assert config.available_times.size == 0


def test_available_times_uses_coordinate_values():
    model = DummyDataConfig(
        "model",
        years=(2000, 2001, 2002),
    )
    condition = DummyDataConfig(
        "condition",
        years=(2001, 2002, 2003),
    )

    model.info.coords["year"] = xr.DataArray(
        np.asarray([2000, 2002]),
        dims=("year",),
    )

    config = bare_config(
        model=model,
        condition=condition,
    )

    np.testing.assert_array_equal(
        config.available_times,
        [2002],
    )


def test_load_fitted_preprocessors_delegates():
    config = bare_config()
    operator = MagicMock()

    with patch.object(
        module,
        "DatasetOperator",
        return_value=operator,
    ):
        config.load_fitted_preprocessors("load-directory")

    operator.load_fitted_preprocessors.assert_called_once_with("load-directory")


def test_load_fitted_preprocessors_default_argument():
    config = bare_config()
    operator = MagicMock()

    with patch.object(
        module,
        "DatasetOperator",
        return_value=operator,
    ):
        config.load_fitted_preprocessors()

    operator.load_fitted_preprocessors.assert_called_once_with(None)


def test_add_fitted_preprocessor_delegates():
    config = bare_config()
    operator = MagicMock()
    preprocessor = object()

    with patch.object(
        module,
        "DatasetOperator",
        return_value=operator,
    ):
        config.add_fitted_preprocessor(
            preprocessor,
            index=4,
        )

    operator.add_fitted_preprocessor.assert_called_once_with(
        preprocessor,
        4,
    )


def test_add_fitted_preprocessor_default_index():
    config = bare_config()
    operator = MagicMock()
    preprocessor = object()

    with patch.object(
        module,
        "DatasetOperator",
        return_value=operator,
    ):
        config.add_fitted_preprocessor(preprocessor)

    operator.add_fitted_preprocessor.assert_called_once_with(
        preprocessor,
        0,
    )


def test_build_dataset_all_arguments():
    config = bare_config(
        model=DummyDataConfig("model"),
    )
    features = object()
    expected = object()
    years = np.asarray([2000, 2001])

    with patch.object(
        module,
        "InferenceDataset",
        return_value=expected,
    ) as constructor:
        result = config.build_dataset(
            years=years,
            time_features=features,
            return_metadata=True,
            load=True,
        )

    assert result is expected

    constructor.assert_called_once_with(
        config=config,
        requested_years=years,
        time_features=features,
        return_metadata=True,
        load=True,
    )


def test_build_dataset_default_arguments():
    config = bare_config(
        model=DummyDataConfig("model"),
    )
    features = object()
    expected = object()
    years = np.asarray([2000])

    with patch.object(
        module,
        "InferenceDataset",
        return_value=expected,
    ) as constructor:
        result = config.build_dataset(
            years=years,
            time_features=features,
        )

    assert result is expected

    constructor.assert_called_once_with(
        config=config,
        requested_years=years,
        time_features=features,
        return_metadata=False,
        load=False,
    )


@pytest.mark.parametrize(
    (
        "using_model_as_condition",
        "model_present",
        "expected",
    ),
    [
        (False, True, True),
        (True, True, False),
        (False, False, False),
        (True, False, False),
    ],
)
def test_load_model_truth_table(
    using_model_as_condition,
    model_present,
    expected,
):
    dataset = bare_dataset(
        config=dataset_config_stub(
            model=(DummyDataConfig("model") if model_present else None),
            using_model_as_condition=using_model_as_condition,
        )
    )

    assert dataset._load_model is expected


@pytest.mark.parametrize(
    (
        "using_model_as_condition",
        "model_present",
        "expected",
    ),
    [
        (False, True, False),
        (True, True, True),
        (False, False, True),
        (True, False, True),
    ],
)
def test_write_condition_to_input_truth_table(
    using_model_as_condition,
    model_present,
    expected,
):
    dataset = bare_dataset(
        config=dataset_config_stub(
            model=(DummyDataConfig("model") if model_present else None),
            using_model_as_condition=using_model_as_condition,
        )
    )

    assert dataset._write_condition_to_input is expected


@pytest.mark.parametrize(
    (
        "using_model_as_condition",
        "model_present",
        "condition_present",
        "expected",
    ),
    [
        (False, True, True, True),
        (False, True, False, False),
        (True, True, True, False),
        (False, False, True, False),
        (True, False, True, False),
    ],
)
def test_concat_condition_to_input_truth_table(
    using_model_as_condition,
    model_present,
    condition_present,
    expected,
):
    dataset = bare_dataset(
        config=dataset_config_stub(
            model=(DummyDataConfig("model") if model_present else None),
            condition=(DummyDataConfig("condition") if condition_present else None),
            using_model_as_condition=using_model_as_condition,
        )
    )

    assert dataset._concat_condition_to_input is expected


def test_getitem_model_only():
    model = DummyDataConfig("model")

    dataset = bare_dataset(
        config=dataset_config_stub(
            model=model,
            condition=None,
            using_model_as_condition=False,
        )
    )

    model_data = data_array(
        [1.0, 2.0],
        ["model_a", "model_b"],
    )

    install_item_sources(
        dataset,
        model=model_data,
        condition=None,
    )

    with patch.object(
        module.dask,
        "compute",
        side_effect=lambda value: (np.asarray(value),),
    ) as compute:
        result = dataset[0]

    assert isinstance(result, dict)

    torch.testing.assert_close(
        result["input"],
        torch.tensor(
            [
                [
                    1.0,
                    2.0,
                ]
            ],
            dtype=torch.float32,
        ),
    )

    assert result["added_features"] is None

    dataset._index_model_dataset.assert_called_once_with(0)
    dataset._index_condition_dataset.assert_called_once_with(0)
    compute.assert_called_once()


def test_getitem_model_reused_as_condition():
    model = DummyDataConfig("model")

    dataset = bare_dataset(
        config=dataset_config_stub(
            model=model,
            condition=model,
            using_model_as_condition=True,
        )
    )

    condition_data = data_array(
        [3.0],
        ["condition"],
    )

    install_item_sources(
        dataset,
        model=None,
        condition=condition_data,
    )

    with patch.object(
        module.dask,
        "compute",
        side_effect=lambda value: (np.asarray(value),),
    ):
        result = dataset[0]

    torch.testing.assert_close(
        result["input"],
        torch.tensor(
            [
                [
                    3.0,
                ]
            ],
            dtype=torch.float32,
        ),
    )


def test_getitem_condition_only():
    condition = DummyDataConfig("condition")

    dataset = bare_dataset(
        config=dataset_config_stub(
            model=None,
            condition=condition,
            using_model_as_condition=False,
        )
    )

    condition_data = data_array(
        [4.0, 5.0],
        ["condition_a", "condition_b"],
    )

    install_item_sources(
        dataset,
        model=None,
        condition=condition_data,
    )

    with patch.object(
        module.dask,
        "compute",
        side_effect=lambda value: (np.asarray(value),),
    ):
        result = dataset[0]

    torch.testing.assert_close(
        result["input"],
        torch.tensor(
            [
                [
                    4.0,
                    5.0,
                ]
            ],
            dtype=torch.float32,
        ),
    )


def test_getitem_concatenates_model_and_condition():
    model = DummyDataConfig("model")
    condition = DummyDataConfig("condition")

    dataset = bare_dataset(
        config=dataset_config_stub(
            model=model,
            condition=condition,
            using_model_as_condition=False,
        )
    )

    model_data = data_array(
        [1.0, 2.0],
        ["model_a", "model_b"],
    )
    condition_data = data_array(
        [3.0, 4.0],
        ["condition_a", "condition_b"],
    )

    install_item_sources(
        dataset,
        model=model_data,
        condition=condition_data,
    )

    with patch.object(
        module.dask,
        "compute",
        side_effect=lambda value: (np.asarray(value),),
    ):
        result = dataset[0]

    torch.testing.assert_close(
        result["input"],
        torch.tensor(
            [
                [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                ]
            ],
            dtype=torch.float32,
        ),
    )


def test_getitem_time_features_receive_selection_and_input():
    model = DummyDataConfig("model")

    features = FakeTimeFeatures(
        result=np.asarray(
            [
                2000.0,
                1.0,
                0.5,
                -0.5,
            ]
        )
    )

    dataset = bare_dataset(
        config=dataset_config_stub(
            model=model,
        ),
        time_features=features,
    )

    model_data = data_array(
        [1.0],
        ["model"],
    )

    install_item_sources(
        dataset,
        model=model_data,
        condition=None,
    )

    with patch.object(
        module.dask,
        "compute",
        side_effect=lambda value: (np.asarray(value),),
    ):
        result = dataset[0]

    assert len(features.calls) == 1

    selection, feature_input = features.calls[0]

    assert selection == {
        "year": 2000.0,
        "lead_time": 1.0,
    }

    xr.testing.assert_equal(
        feature_input,
        model_data,
    )

    torch.testing.assert_close(
        result["added_features"],
        torch.tensor(
            [
                2000.0,
                1.0,
                0.5,
                -0.5,
            ],
            dtype=torch.float32,
        ),
    )


def test_getitem_without_time_features():
    model = DummyDataConfig("model")
    features = FakeTimeFeatures(result=None)

    dataset = bare_dataset(
        config=dataset_config_stub(
            model=model,
        ),
        time_features=features,
    )

    install_item_sources(
        dataset,
        model=data_array([1.0]),
        condition=None,
    )

    with patch.object(
        module.dask,
        "compute",
        side_effect=lambda value: (np.asarray(value),),
    ):
        result = dataset[0]

    assert result["added_features"] is None


def test_getitem_returns_float32_tensors():
    model = DummyDataConfig("model")

    features = FakeTimeFeatures(
        np.asarray(
            [2000.0],
            dtype=np.float64,
        )
    )

    dataset = bare_dataset(
        config=dataset_config_stub(
            model=model,
        ),
        time_features=features,
    )

    install_item_sources(
        dataset,
        model=xr.DataArray(
            np.asarray(
                [1.0],
                dtype=np.float64,
            ),
            dims=("channels",),
        ),
        condition=None,
    )

    with patch.object(
        module.dask,
        "compute",
        side_effect=lambda value: (np.asarray(value),),
    ):
        result = dataset[0]

    assert result["input"].dtype == torch.float32
    assert result["added_features"].dtype == torch.float32


def test_getitem_without_metadata_returns_dict():
    dataset = bare_dataset(
        return_metadata=False,
    )

    install_item_sources(
        dataset,
        model=data_array([1.0]),
        condition=None,
    )

    with patch.object(
        module.dask,
        "compute",
        side_effect=lambda value: (np.asarray(value),),
    ):
        result = dataset[0]

    assert isinstance(result, dict)
    assert set(result) == {
        "input",
        "added_features",
    }


def test_getitem_with_metadata_returns_tuple():
    dataset = bare_dataset(
        return_metadata=True,
        sample_coords={
            "year": np.asarray([2000.0]),
            "lead_time": np.asarray([1.0]),
            "ensembles": np.asarray([1]),
        },
    )

    install_item_sources(
        dataset,
        model=data_array([1.0]),
        condition=None,
    )

    with patch.object(
        module.dask,
        "compute",
        side_effect=lambda value: (np.asarray(value),),
    ):
        result, metadata = dataset[0]

    assert isinstance(result, dict)

    assert metadata == {
        "year": 2000.0,
        "lead_time": 1.0,
        "ensembles": 1,
    }


def test_getitem_uses_requested_index():
    dataset = bare_dataset(
        return_metadata=True,
        sample_coords={
            "year": np.asarray(
                [
                    2000.0,
                    2001.0,
                ]
            ),
            "lead_time": np.asarray(
                [
                    1.0,
                    13.0,
                ]
            ),
            "ensembles": np.asarray(
                [
                    0,
                    1,
                ]
            ),
        },
    )

    install_item_sources(
        dataset,
        model=data_array([2.0]),
        condition=None,
    )

    with patch.object(
        module.dask,
        "compute",
        side_effect=lambda value: (np.asarray(value),),
    ):
        _, metadata = dataset[1]

    assert metadata == {
        "year": 2001.0,
        "lead_time": 13.0,
        "ensembles": 1,
    }

    dataset._index_model_dataset.assert_called_once_with(1)
    dataset._index_condition_dataset.assert_called_once_with(1)


def test_getitem_computes_dask_data():
    dataset = bare_dataset()

    install_item_sources(
        dataset,
        model=data_array([1.0]),
        condition=None,
    )

    computed = np.asarray([9.0])

    with patch.object(
        module.dask,
        "compute",
        return_value=(computed,),
    ) as compute:
        result = dataset[0]

    compute.assert_called_once()

    torch.testing.assert_close(
        result["input"],
        torch.tensor(
            [
                [
                    9.0,
                ]
            ],
            dtype=torch.float32,
        ),
    )


def test_from_train_observation_without_condition():
    model = DummyDataConfig("model")

    train_config = make_train_config(
        observation=object(),
        effective_condition=None,
        using_model_as_condition=False,
        model=model,
        condition=None,
    )

    with patch.object(
        module,
        "InferenceDatasetConfig",
        side_effect=lambda **kwargs: kwargs,
    ):
        result = _from_train(train_config)

    assert result["model"] is not model
    assert result["model"].names == model.names
    assert "condition" not in result


def test_from_train_model_used_as_condition():
    model = DummyDataConfig("model")

    train_config = make_train_config(
        observation=None,
        effective_condition=model,
        using_model_as_condition=True,
        model=model,
        condition=None,
    )

    with patch.object(
        module,
        "InferenceDatasetConfig",
        side_effect=lambda **kwargs: kwargs,
    ):
        result = _from_train(train_config)

    assert result["model"] is not model
    assert result["model"].names == model.names
    assert "condition" not in result


def test_from_train_observation_and_independent_condition():
    model = DummyDataConfig("model")
    condition = DummyDataConfig("condition")

    train_config = make_train_config(
        observation=object(),
        effective_condition=condition,
        using_model_as_condition=False,
        model=model,
        condition=condition,
    )

    with patch.object(
        module,
        "InferenceDatasetConfig",
        side_effect=lambda **kwargs: kwargs,
    ):
        result = _from_train(train_config)

    assert result["model"] is not model
    assert result["condition"] is not condition
    assert result["model"].names == ["model"]
    assert result["condition"].names == ["condition"]


def test_from_train_condition_without_observation():
    model = DummyDataConfig("model")
    condition = DummyDataConfig("condition")

    train_config = make_train_config(
        observation=None,
        effective_condition=condition,
        using_model_as_condition=False,
        model=model,
        condition=condition,
    )

    with patch.object(
        module,
        "InferenceDatasetConfig",
        side_effect=lambda **kwargs: kwargs,
    ):
        result = _from_train(train_config)

    assert "model" not in result
    assert result["condition"] is not condition
    assert result["condition"].names == ["condition"]


def test_from_train_cannot_resolve_dataset():
    train_config = make_train_config(
        observation=None,
        effective_condition=None,
        using_model_as_condition=False,
        model=None,
        condition=None,
    )

    with pytest.raises(
        ValueError,
        match="Could not infer inference dataset config",
    ):
        _from_train(train_config)


def test_from_train_copies_condition_method():
    model = DummyDataConfig("model")

    train_config = make_train_config(
        observation=object(),
        model=model,
        condition_method="ensemble_mean",
    )

    with patch.object(
        module,
        "InferenceDatasetConfig",
        side_effect=lambda **kwargs: kwargs,
    ):
        result = _from_train(train_config)

    assert result["condition_method"] == "ensemble_mean"


def test_from_train_deepcopies_lead_months():
    model = DummyDataConfig("model")
    lead_months = np.asarray([1, 3, 6])

    train_config = make_train_config(
        observation=object(),
        model=model,
        lead_months=lead_months,
    )

    with patch.object(
        module,
        "InferenceDatasetConfig",
        side_effect=lambda **kwargs: kwargs,
    ):
        result = _from_train(train_config)

    np.testing.assert_array_equal(
        result["lead_months"],
        lead_months,
    )

    assert result["lead_months"] is not lead_months


def test_from_train_deepcopies_model():
    model = DummyDataConfig("model")

    train_config = make_train_config(
        observation=object(),
        model=model,
    )

    with patch.object(
        module,
        "InferenceDatasetConfig",
        side_effect=lambda **kwargs: kwargs,
    ):
        result = _from_train(train_config)

    copied_model = result["model"]

    assert copied_model is not model
    assert copied_model.names == model.names
    assert copied_model.preprocessing_pipeline is not (model.preprocessing_pipeline)


def test_from_train_deepcopies_condition():
    model = DummyDataConfig("model")
    condition = DummyDataConfig("condition")

    train_config = make_train_config(
        observation=object(),
        effective_condition=condition,
        model=model,
        condition=condition,
    )

    with patch.object(
        module,
        "InferenceDatasetConfig",
        side_effect=lambda **kwargs: kwargs,
    ):
        result = _from_train(train_config)

    copied_condition = result["condition"]

    assert copied_condition is not condition
    assert copied_condition.names == condition.names
    assert copied_condition.preprocessing_pipeline is not (
        condition.preprocessing_pipeline
    )


def test_from_train_does_not_copy_time_features():
    model = DummyDataConfig("model")

    train_config = make_train_config(
        observation=object(),
        model=model,
    )
    train_config.time_features = [
        "year",
        "lead_time",
    ]

    with patch.object(
        module,
        "InferenceDatasetConfig",
        side_effect=lambda **kwargs: kwargs,
    ):
        result = _from_train(train_config)

    assert "time_features" not in result


def test_from_train_does_not_mutate_training_config():
    model = DummyDataConfig("model")
    lead_months = np.asarray([1, 2])

    train_config = make_train_config(
        observation=object(),
        model=model,
        lead_months=lead_months,
    )

    original_model = train_config.model
    original_lead_months = train_config.lead_months.copy()

    with patch.object(
        module,
        "InferenceDatasetConfig",
        side_effect=lambda **kwargs: kwargs,
    ):
        _from_train(train_config)

    assert train_config.model is original_model

    np.testing.assert_array_equal(
        train_config.lead_months,
        original_lead_months,
    )


def test_from_train_constructs_inference_config():
    model = DummyDataConfig("model")

    train_config = make_train_config(
        observation=object(),
        model=model,
        condition_method=None,
        lead_months=np.asarray([1, 2]),
    )

    with patch.object(
        module,
        "InferenceDatasetConfig",
    ) as constructor:
        result = _from_train(train_config)

    assert result is constructor.return_value

    constructor.assert_called_once()

    kwargs = constructor.call_args.kwargs

    assert kwargs["condition_method"] is None

    np.testing.assert_array_equal(
        kwargs["lead_months"],
        [1, 2],
    )

    assert kwargs["model"] is not model
    assert "condition" not in kwargs
