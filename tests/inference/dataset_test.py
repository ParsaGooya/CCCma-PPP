from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import xarray as xr

import cccma_ppp.inference.dataset as module
from cccma_ppp.data_modules.dataset.config_abc import DatasetConfigABC
from cccma_ppp.data_modules.dataset.dataset_abc import DatasetABC
from cccma_ppp.inference.dataset import (
    InferenceDataset,
    InferenceDatasetConfig,
    _from_train,
)


class DummyPipeline:
    def __init__(self, name="pipeline"):
        self.name = name


class DummyDataConfig:
    def __init__(
        self,
        name="data",
        *,
        years=(2000, 2001, 2002),
        lead_time=12,
        ensemble_mean=False,
        ensemble_list=None,
        ensembles=(0, 1),
    ):
        self.names = [name]
        self.year_range = np.asarray(years)
        self.ensemble_mean = ensemble_mean
        self.ensemble_list = ensemble_list
        self.preprocessing_pipeline = DummyPipeline(name)

        coords = {
            "year": xr.DataArray(
                np.asarray(years),
                dims=("year",),
            ),
            "lead_time": xr.DataArray(
                np.arange(1, lead_time + 1),
                dims=("lead_time",),
            ),
            "lat": xr.DataArray(
                np.asarray([0, 1]),
                dims=("lat",),
            ),
            "lon": xr.DataArray(
                np.asarray([10, 20, 30]),
                dims=("lon",),
            ),
        }

        if ensembles == "missing":
            pass
        elif ensembles is None:
            coords["ensembles"] = None
        else:
            coords["ensembles"] = xr.DataArray(
                np.asarray(ensembles),
                dims=("ensembles",),
            )

        sizes = {
            "year": len(years),
            "lead_time": lead_time,
            "lat": 2,
            "lon": 3,
        }

        if ensembles not in (None, "missing"):
            sizes["ensembles"] = len(ensembles)

        self.info = SimpleNamespace(
            coords=coords,
            sizes=sizes,
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


InferenceDatasetConfig.__abstractmethods__ = frozenset()


def make_config(
    *,
    model=None,
    condition=None,
    condition_method=None,
    lead_months=None,
    effective_condition=None,
    using_model_as_condition=False,
):
    config = object.__new__(InferenceDatasetConfig)

    config.model = model
    config.condition = condition
    config.condition_method = condition_method
    config.lead_months = lead_months
    config._effective_condition = effective_condition
    config._using_model_data_as_condition_override = using_model_as_condition
    config._fitted_preprocessors = True

    if using_model_as_condition:
        config.condition = None
        config._using_model_data_as_condition_override = True
        config._effective_condition = model

    return config


def make_dataset_config(
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


def make_dataset(
    *,
    config=None,
    sample_coords=None,
    time_features=None,
    return_metadata=False,
    load=False,
):
    dataset = object.__new__(InferenceDataset)

    if config is None:
        config = make_dataset_config(
            model=DummyDataConfig("model"),
        )

    dataset.config = config
    dataset.requested_years = np.asarray([2000])
    dataset.time_features = (
        time_features if time_features is not None else FakeTimeFeatures(None)
    )
    dataset.return_metadata = return_metadata
    dataset.load = load
    dataset.mask = None

    dataset.sample_coords = (
        sample_coords
        if sample_coords is not None
        else {
            "year": np.asarray([2000.0]),
            "lead_time": np.asarray([1.0]),
        }
    )

    return dataset


def make_data_array(
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


def install_dataset_sources(
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


def test_effective_input_prefers_model():
    model = DummyDataConfig("model")
    condition = DummyDataConfig("condition")

    config = make_config(
        model=model,
        condition=condition,
    )

    assert config.effective_input is model


def test_effective_input_returns_model_without_condition():
    model = DummyDataConfig("model")

    config = make_config(
        model=model,
        condition=None,
    )

    assert config.effective_input is model


def test_effective_input_falls_back_to_condition():
    condition = DummyDataConfig("condition")

    config = make_config(
        model=None,
        condition=condition,
    )

    assert config.effective_input is condition


def test_effective_input_returns_none_without_sources():
    config = make_config(
        model=None,
        condition=None,
    )

    assert config.effective_input is None


def test_available_times_model_only():
    config = make_config(
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
    config = make_config(
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
    config = make_config(
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
    config = make_config(
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


def test_available_times_uses_info_coordinate_values():
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

    config = make_config(
        model=model,
        condition=condition,
    )

    np.testing.assert_array_equal(
        config.available_times,
        [2002],
    )


def test_available_times_preserves_single_source_order():
    model = DummyDataConfig(
        "model",
        years=(2002, 2000, 2001),
    )

    config = make_config(
        model=model,
    )

    np.testing.assert_array_equal(
        config.available_times,
        [2002, 2000, 2001],
    )


def test_available_times_intersection_is_sorted():
    model = DummyDataConfig(
        "model",
        years=(2002, 2000, 2001),
    )
    condition = DummyDataConfig(
        "condition",
        years=(2001, 2002),
    )

    config = make_config(
        model=model,
        condition=condition,
    )

    np.testing.assert_array_equal(
        config.available_times,
        [2001, 2002],
    )


def test_available_times_without_any_source_raises_index_error():
    config = make_config(
        model=None,
        condition=None,
    )

    with pytest.raises((IndexError, ValueError)):
        _ = config.available_times


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
    dataset = make_dataset(
        config=make_dataset_config(
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
    dataset = make_dataset(
        config=make_dataset_config(
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
        (False, False, False, False),
    ],
)
def test_concat_condition_to_input_truth_table(
    using_model_as_condition,
    model_present,
    condition_present,
    expected,
):
    dataset = make_dataset(
        config=make_dataset_config(
            model=(DummyDataConfig("model") if model_present else None),
            condition=(DummyDataConfig("condition") if condition_present else None),
            using_model_as_condition=using_model_as_condition,
        )
    )

    assert dataset._concat_condition_to_input is expected


def test_getitem_model_only():
    dataset = make_dataset(
        config=make_dataset_config(
            model=DummyDataConfig("model"),
            condition=None,
            using_model_as_condition=False,
        )
    )

    model_data = make_data_array(
        [1.0, 2.0],
        ["model_a", "model_b"],
    )

    install_dataset_sources(
        dataset,
        model=model_data,
        condition=None,
    )

    with patch.object(
        dataset,
        "_compute",
        side_effect=lambda value: np.asarray(value),
    ) as compute:
        result = dataset[0]

    torch.testing.assert_close(
        result["input"],
        torch.tensor(
            [
                1.0,
                2.0,
            ],
            dtype=torch.float32,
        ),
    )

    assert result["added_features"] is None

    dataset._index_model_dataset.assert_called_once_with(0)
    dataset._index_condition_dataset.assert_called_once_with(0)
    compute.assert_called_once()


def test_getitem_model_used_as_condition():
    model = DummyDataConfig("model")

    dataset = make_dataset(
        config=make_dataset_config(
            model=model,
            condition=model,
            using_model_as_condition=True,
        )
    )

    model_data = make_data_array(
        [1.0],
        ["model"],
    )
    condition_data = make_data_array(
        [3.0],
        ["condition"],
    )

    install_dataset_sources(
        dataset,
        model=model_data,
        condition=condition_data,
    )

    with patch.object(
        dataset,
        "_compute",
        side_effect=lambda value: np.asarray(value),
    ):
        result = dataset[0]

    torch.testing.assert_close(
        result["input"],
        torch.tensor(
            [
                3.0,
            ],
            dtype=torch.float32,
        ),
    )


def test_getitem_condition_only():
    dataset = make_dataset(
        config=make_dataset_config(
            model=None,
            condition=DummyDataConfig("condition"),
            using_model_as_condition=False,
        )
    )

    condition_data = make_data_array(
        [4.0, 5.0],
        ["condition_a", "condition_b"],
    )

    install_dataset_sources(
        dataset,
        model=None,
        condition=condition_data,
    )

    with patch.object(
        dataset,
        "_compute",
        side_effect=lambda value: np.asarray(value),
    ):
        result = dataset[0]

    torch.testing.assert_close(
        result["input"],
        torch.tensor(
            [
                4.0,
                5.0,
            ],
            dtype=torch.float32,
        ),
    )


def test_getitem_concatenates_model_and_condition():
    dataset = make_dataset(
        config=make_dataset_config(
            model=DummyDataConfig("model"),
            condition=DummyDataConfig("condition"),
            using_model_as_condition=False,
        )
    )

    model_data = make_data_array(
        [1.0, 2.0],
        ["model_a", "model_b"],
    )
    condition_data = make_data_array(
        [3.0, 4.0],
        ["condition_a", "condition_b"],
    )

    install_dataset_sources(
        dataset,
        model=model_data,
        condition=condition_data,
    )

    with patch.object(
        dataset,
        "_compute",
        side_effect=lambda value: np.asarray(value),
    ):
        result = dataset[0]

    torch.testing.assert_close(
        result["input"],
        torch.tensor(
            [
                1.0,
                2.0,
                3.0,
                4.0,
            ],
            dtype=torch.float32,
        ),
    )


def test_getitem_condition_replacement_precedes_concat():
    dataset = make_dataset(
        config=make_dataset_config(
            model=DummyDataConfig("model"),
            condition=DummyDataConfig("condition"),
            using_model_as_condition=True,
        )
    )

    install_dataset_sources(
        dataset,
        model=make_data_array([1.0]),
        condition=make_data_array([9.0]),
    )

    with patch.object(
        dataset,
        "_compute",
        side_effect=lambda value: np.asarray(value),
    ):
        result = dataset[0]

    torch.testing.assert_close(
        result["input"],
        torch.tensor(
            [
                9.0,
            ],
            dtype=torch.float32,
        ),
    )


def test_getitem_time_features_receive_final_model_input():
    features = FakeTimeFeatures(
        result=np.asarray(
            [
                2000.0,
                1.0,
            ]
        )
    )

    dataset = make_dataset(
        config=make_dataset_config(
            model=DummyDataConfig("model"),
        ),
        time_features=features,
    )

    model_data = make_data_array([1.0])

    install_dataset_sources(
        dataset,
        model=model_data,
        condition=None,
    )

    with patch.object(
        dataset,
        "_compute",
        side_effect=lambda value: np.asarray(value),
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
            ],
            dtype=torch.float32,
        ),
    )


def test_getitem_time_features_receive_condition_input():
    features = FakeTimeFeatures(result=np.asarray([2000.0]))

    dataset = make_dataset(
        config=make_dataset_config(
            model=None,
            condition=DummyDataConfig("condition"),
        ),
        time_features=features,
    )

    condition_data = make_data_array([3.0])

    install_dataset_sources(
        dataset,
        model=None,
        condition=condition_data,
    )

    with patch.object(
        dataset,
        "_compute",
        side_effect=lambda value: np.asarray(value),
    ):
        dataset[0]

    _, feature_input = features.calls[0]

    xr.testing.assert_equal(
        feature_input,
        condition_data,
    )


def test_getitem_time_features_receive_concatenated_input():
    features = FakeTimeFeatures(result=np.asarray([1.0]))

    dataset = make_dataset(
        config=make_dataset_config(
            model=DummyDataConfig("model"),
            condition=DummyDataConfig("condition"),
        ),
        time_features=features,
    )

    model_data = make_data_array(
        [1.0],
        ["model"],
    )
    condition_data = make_data_array(
        [2.0],
        ["condition"],
    )

    install_dataset_sources(
        dataset,
        model=model_data,
        condition=condition_data,
    )

    with patch.object(
        dataset,
        "_compute",
        side_effect=lambda value: np.asarray(value),
    ):
        dataset[0]

    _, feature_input = features.calls[0]

    assert list(feature_input.coords["channels"].values) == [
        "model",
        "condition",
    ]


def test_getitem_without_time_features_returns_none():
    dataset = make_dataset(
        time_features=FakeTimeFeatures(None),
    )

    install_dataset_sources(
        dataset,
        model=make_data_array([1.0]),
        condition=None,
    )

    with patch.object(
        dataset,
        "_compute",
        side_effect=lambda value: np.asarray(value),
    ):
        result = dataset[0]

    assert result["added_features"] is None


def test_getitem_converts_tensors_to_float32():
    dataset = make_dataset(
        time_features=FakeTimeFeatures(
            np.asarray(
                [2000.0],
                dtype=np.float64,
            )
        ),
    )

    install_dataset_sources(
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
        dataset,
        "_compute",
        side_effect=lambda value: np.asarray(value),
    ):
        result = dataset[0]

    assert result["input"].dtype == torch.float32
    assert result["added_features"].dtype == torch.float32


def test_getitem_without_metadata_returns_dictionary():
    dataset = make_dataset(
        return_metadata=False,
    )

    install_dataset_sources(
        dataset,
        model=make_data_array([1.0]),
        condition=None,
    )

    with patch.object(
        dataset,
        "_compute",
        side_effect=lambda value: np.asarray(value),
    ):
        result = dataset[0]

    assert isinstance(result, dict)

    assert set(result) == {
        "input",
        "added_features",
    }


def test_getitem_with_metadata_returns_tuple():
    dataset = make_dataset(
        return_metadata=True,
        sample_coords={
            "year": np.asarray([2000.0]),
            "lead_time": np.asarray([1.0]),
            "ensembles": np.asarray([1]),
        },
    )

    install_dataset_sources(
        dataset,
        model=make_data_array([1.0]),
        condition=None,
    )

    with patch.object(
        dataset,
        "_compute",
        side_effect=lambda value: np.asarray(value),
    ):
        result, metadata = dataset[0]

    assert isinstance(result, dict)

    assert metadata == {
        "year": 2000.0,
        "lead_time": 1.0,
        "ensembles": 1,
    }


def test_getitem_uses_requested_sample_index():
    dataset = make_dataset(
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

    install_dataset_sources(
        dataset,
        model=make_data_array([2.0]),
        condition=None,
    )

    with patch.object(
        dataset,
        "_compute",
        side_effect=lambda value: np.asarray(value),
    ):
        _, metadata = dataset[1]

    assert metadata == {
        "year": 2001.0,
        "lead_time": 13.0,
        "ensembles": 1,
    }

    dataset._index_model_dataset.assert_called_once_with(1)
    dataset._index_condition_dataset.assert_called_once_with(1)


def test_getitem_passes_input_data_to_dask_compute():
    dataset = make_dataset()

    model_data = make_data_array([1.0])

    install_dataset_sources(
        dataset,
        model=model_data,
        condition=None,
    )

    computed = np.asarray([9.0])

    with patch.object(
        dataset,
        "_compute",
        return_value=computed,
    ) as compute:
        result = dataset[0]

    compute.assert_called_once_with(model_data.data)

    torch.testing.assert_close(
        result["input"],
        torch.tensor(
            [
                9.0,
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


def test_from_train_model_condition_branch_precedes_condition_branch():
    model = DummyDataConfig("model")
    condition = DummyDataConfig("condition")

    train_config = make_train_config(
        observation=object(),
        effective_condition=condition,
        using_model_as_condition=True,
        model=model,
        condition=condition,
    )

    with patch.object(
        module,
        "InferenceDatasetConfig",
        side_effect=lambda **kwargs: kwargs,
    ):
        result = _from_train(train_config)

    assert "model" in result
    assert "condition" not in result
    assert result["model"].names == ["model"]


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


def test_from_train_unresolvable_configuration_raises():
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


def test_from_train_copies_none_condition_method():
    model = DummyDataConfig("model")

    train_config = make_train_config(
        observation=object(),
        model=model,
        condition_method=None,
    )

    with patch.object(
        module,
        "InferenceDatasetConfig",
        side_effect=lambda **kwargs: kwargs,
    ):
        result = _from_train(train_config)

    assert result["condition_method"] is None


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


def test_from_train_preserves_none_lead_months():
    model = DummyDataConfig("model")

    train_config = make_train_config(
        observation=object(),
        model=model,
    )
    train_config.lead_months = None

    with patch.object(
        module,
        "InferenceDatasetConfig",
        side_effect=lambda **kwargs: kwargs,
    ):
        result = _from_train(train_config)

    assert result["lead_months"] is None


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
