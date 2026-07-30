from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch
import warnings

import numpy as np
import pytest
import torch
import xarray as xr

import cccma_ppp.train.dataset as module
from cccma_ppp.data_modules.dataset.dataset_abc import (
    AddedTimeFeatures,
    DatasetABC,
)
from cccma_ppp.train.dataset import (
    TrainDataset,
    TrainDatasetConfig,
)


class Coordinate:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.size = self.values.size

    def equals(self, other):
        return np.array_equal(
            self.values,
            other.values,
        )

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]


class DummyPipeline:
    def __init__(
        self,
        fitted_preprocessors=None,
    ):
        self.fitted_preprocessors = list(fitted_preprocessors or [])
        self.transform_calls = []

    def transform(self, value):
        self.transform_calls.append(value)
        return value

    def get_preprocessors(self, name):
        return self._flattener


def make_data_config(
    *,
    paths=("data.nc",),
    names=("tas",),
    years=(2000, 2001, 2002),
    lead_times=(1, 2, 3),
    months=(0.5, 1.5, 2.5),
    ensembles=(0, 1),
    ensemble_mean=False,
    ensemble_list=None,
    lat=(0, 1),
    lon=(10, 20),
    pipeline=None,
):
    coords = {
        "year": Coordinate(years),
        "lead_time": Coordinate(lead_times),
        "month": Coordinate(months),
        "lat": Coordinate(lat),
        "lon": Coordinate(lon),
    }

    if ensembles is not None:
        coords["ensembles"] = Coordinate(ensembles)

    return SimpleNamespace(
        paths=list(paths),
        list_paths=list(paths),
        names=list(names),
        year_range=np.asarray(years),
        ensemble_mean=ensemble_mean,
        ensemble_list=ensemble_list,
        concat_dim="year",
        file_type="netcdf",
        rename_dict={},
        preprocessing_pipeline=(pipeline if pipeline is not None else DummyPipeline()),
        info=SimpleNamespace(
            coords=coords,
            sizes={name: coordinate.size for name, coordinate in coords.items()},
        ),
    )


def make_config_stub(
    *,
    model=None,
    observation=None,
    condition=None,
    effective_condition=None,
    condition_method="static",
    using_model_as_condition=False,
):
    if model is None:
        model = make_data_config(
            paths=("model.nc",),
        )

    return SimpleNamespace(
        model=model,
        observation=observation,
        condition=condition,
        effective_condition=(effective_condition),
        condition_method=condition_method,
        _using_model_data_as_condition=(using_model_as_condition),
        lead_months=np.asarray([1, 2, 3]),
        available_times=np.asarray([2000, 2001, 2002]),
        input_lead_months=np.asarray([1, 2, 3]),
        get_common_time=np.asarray([2000, 2001, 2002]),
        _fitted_preprocessors=True,
    )


class ConcreteTrainDatasetConfig(TrainDatasetConfig):
    def _check_model(self):
        return self

    def _check_condition(self):
        return self

    @property
    def num_input_lead_months(self):
        return self.model.info.sizes["lead_time"]


def bare_config(**kwargs):
    config = object.__new__(ConcreteTrainDatasetConfig)

    config.model = kwargs.get(
        "model",
        make_data_config(
            paths=("model.nc",),
        ),
    )
    config.observation = kwargs.get("observation")
    config.condition = kwargs.get("condition")
    config.condition_method = kwargs.get("condition_method")
    config.lead_months = kwargs.get("lead_months")
    config._effective_condition = kwargs.get("effective_condition")
    config._fitted_preprocessors = kwargs.get(
        "fitted_preprocessors",
        True,
    )

    return config


def bare_dataset(**kwargs):
    dataset = object.__new__(TrainDataset)

    for name, value in kwargs.items():
        setattr(dataset, name, value)

    return dataset


def make_loaded_dataset(
    *,
    ensembles=(0, 1),
):
    dims = [
        "ensembles",
        "year",
        "lead_time",
        "month",
        "channels",
    ]
    coords = {
        "ensembles": list(ensembles),
        "year": [2000, 2001, 2002],
        "lead_time": [1, 2, 3],
        "month": [0.5, 1.5, 2.5],
        "channels": ["tas"],
    }
    shape = [
        len(ensembles),
        3,
        3,
        3,
        1,
    ]

    return xr.DataArray(
        np.arange(
            np.prod(shape),
            dtype=float,
        ).reshape(shape),
        dims=dims,
        coords=coords,
        name="tas",
    )


def make_observation_dataset(
    *,
    ensembles=(0, 1),
):
    dims = [
        "year",
        "month",
        "channels",
    ]
    coords = {
        "year": [2000, 2001, 2002, 2003],
        "month": [
            0.5,
            1.5,
            2.5,
            3.5,
        ],
        "channels": ["tas"],
    }
    shape = [
        4,
        4,
        1,
    ]

    if ensembles is not None:
        dims.insert(0, "ensembles")
        coords["ensembles"] = list(ensembles)
        shape.insert(0, len(ensembles))

    return xr.DataArray(
        np.arange(
            np.prod(shape),
            dtype=float,
        ).reshape(shape),
        dims=dims,
        coords=coords,
        name="tas",
    )


def make_sample_coords(
    *,
    years=(2000,),
    lead_times=(1,),
    ensembles=(0,),
):
    result = {
        "year": np.asarray(years),
        "lead_time": np.asarray(lead_times),
    }

    if ensembles is not None:
        result["ensembles"] = np.asarray(ensembles)

    return result


def test_check_observation_none_requires_method():
    config = bare_config(
        observation=None,
        condition_method=None,
    )

    with pytest.raises(
        ValueError,
        match="No target observation",
    ):
        config._check_observation()


def test_check_observation_none_with_method():
    config = bare_config(
        observation=None,
        condition_method="static",
    )

    assert config._check_observation() is config


@pytest.mark.pruned
def test_check_observation_equal_coordinates():
    model = make_data_config()
    observation = make_data_config()

    config = bare_config(
        model=model,
        observation=observation,
        condition_method="static",
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = config._check_observation()

    assert result is config
    assert caught == []


def test_check_observation_coordinate_mismatch():
    model = make_data_config(
        lat=(0, 1),
    )
    observation = make_data_config(
        lat=(8, 9),
    )

    config = bare_config(
        model=model,
        observation=observation,
        condition_method="static",
    )

    with pytest.warns(
        UserWarning,
        match="lat",
    ):
        assert config._check_observation() is config


@pytest.mark.pruned
def test_check_observation_dimension_missing_from_model():
    model = make_data_config()
    observation = make_data_config()

    del model.info.coords["lat"]

    config = bare_config(
        model=model,
        observation=observation,
        condition_method="static",
    )

    with pytest.warns(
        UserWarning,
        match="not present in model",
    ):
        config._check_observation()


def test_common_time_without_observation():
    model = make_data_config(
        years=(2000, 2001, 2002),
    )

    config = bare_config(
        model=model,
        observation=None,
    )

    np.testing.assert_array_equal(
        config.get_common_time,
        [2000, 2001, 2002],
    )


def test_common_time_with_observation():
    model = make_data_config(
        years=(2000, 2001, 2002),
    )
    observation = make_data_config(
        years=(2001, 2002, 2003),
    )

    config = bare_config(
        model=model,
        observation=observation,
    )

    np.testing.assert_array_equal(
        config.get_common_time,
        [2001, 2002],
    )


@pytest.mark.pruned
def test_common_time_empty_intersection():
    model = make_data_config(
        years=(2000, 2001),
    )
    observation = make_data_config(
        years=(2010, 2011),
    )

    config = bare_config(
        model=model,
        observation=observation,
    )

    assert config.get_common_time.size == 0


@pytest.mark.pruned
def test_available_times_intersects_coordinates():
    model = make_data_config(
        years=(2000, 2001, 2002),
    )
    observation = make_data_config(
        years=(2001, 2002, 2003),
    )

    config = bare_config(
        model=model,
        observation=observation,
    )

    np.testing.assert_array_equal(
        config.available_times,
        [2001, 2002],
    )


@pytest.mark.parametrize(
    (
        "observation",
        "expected",
    ),
    [
        (None, True),
        (object(), False),
    ],
)
def test_autoencoding_model_data(
    observation,
    expected,
):
    dataset = bare_dataset(config=SimpleNamespace(observation=observation))

    assert dataset._autoencoding_model_data is expected


@pytest.mark.parametrize(
    (
        "observation",
        "using_model_condition",
        "expected",
    ),
    [
        (None, False, True),
        (None, True, True),
        (object(), False, True),
        (object(), True, False),
    ],
)
def test_load_model_truth_table(
    observation,
    using_model_condition,
    expected,
):
    dataset = bare_dataset(
        config=SimpleNamespace(
            observation=observation,
            _using_model_data_as_condition=(using_model_condition),
        )
    )

    assert dataset._load_model is expected


@pytest.mark.parametrize(
    (
        "observation",
        "using_model_condition",
        "expected",
    ),
    [
        (None, False, True),
        (None, True, True),
        (object(), False, False),
        (object(), True, True),
    ],
)
def test_write_condition_to_input_truth_table(
    observation,
    using_model_condition,
    expected,
):
    dataset = bare_dataset(
        config=SimpleNamespace(
            observation=observation,
            _using_model_data_as_condition=(using_model_condition),
        )
    )

    assert dataset._write_condition_to_input is expected


@pytest.mark.parametrize(
    (
        "observation",
        "using_model_condition",
        "effective_condition",
        "expected",
    ),
    [
        (
            object(),
            False,
            object(),
            True,
        ),
        (
            object(),
            False,
            None,
            False,
        ),
        (
            object(),
            True,
            object(),
            False,
        ),
        (
            None,
            False,
            object(),
            False,
        ),
    ],
)
def test_concat_condition_to_input_truth_table(
    observation,
    using_model_condition,
    effective_condition,
    expected,
):
    dataset = bare_dataset(
        config=SimpleNamespace(
            observation=observation,
            _using_model_data_as_condition=(using_model_condition),
            effective_condition=(effective_condition),
        )
    )

    assert dataset._concat_condition_to_input is expected


def test_post_init_without_observation():
    config = make_config_stub(
        observation=None,
    )
    features = AddedTimeFeatures(
        config,
        None,
    )

    dataset = object.__new__(TrainDataset)
    dataset.config = config
    dataset.requested_years = [2000]
    dataset.time_features = features
    dataset.mask = None
    dataset.return_metadata = False
    dataset.load = False

    sample_coords = make_sample_coords()

    with (
        patch.object(
            DatasetABC,
            "__init__",
            return_value=None,
        ),
        patch.object(
            TrainDataset,
            "get_obs_indexes",
            return_value=None,
        ) as get_indexes,
    ):
        dataset.sample_coords = sample_coords
        dataset.observation_dataset = None

        dataset.__post_init__()

    assert dataset.observation_dataset is None
    get_indexes.assert_called_once_with(sample_coords)


def test_post_init_loads_observation():
    observation = make_data_config(
        paths=("obs.nc",),
    )
    config = make_config_stub(
        observation=observation,
    )
    features = AddedTimeFeatures(
        config,
        None,
    )

    dataset = object.__new__(TrainDataset)
    dataset.config = config
    dataset.requested_years = [2000]
    dataset.time_features = features
    dataset.mask = None
    dataset.return_metadata = False
    dataset.load = True
    dataset.sample_coords = make_sample_coords()
    dataset.observation_dataset = None

    loaded = make_observation_dataset()

    with (
        patch.object(
            DatasetABC,
            "__init__",
            return_value=None,
        ),
        patch.object(
            TrainDataset,
            "_load_xarray_data",
            return_value=loaded,
        ) as loader,
        patch.object(
            TrainDataset,
            "get_obs_indexes",
            return_value={
                "year": np.asarray([0]),
                "month": np.asarray([0]),
            },
        ),
    ):
        dataset.__post_init__()

    assert dataset.observation_dataset is loaded

    loader.assert_called_once_with(
        observation,
        load=True,
    )


def test_get_obs_indexes_returns_none():
    dataset = bare_dataset(
        observation_dataset=None,
    )

    assert dataset.get_obs_indexes(make_sample_coords()) is None


def test_get_obs_indexes_missing_year():
    dataset = bare_dataset(
        observation_dataset=(
            make_observation_dataset(
                ensembles=None,
            )
        )
    )

    with pytest.raises(
        ValueError,
        match="observation coordinates",
    ):
        dataset.get_obs_indexes(
            make_sample_coords(
                years=(2099,),
                lead_times=(1,),
                ensembles=None,
            )
        )


@pytest.mark.pruned
def test_get_obs_indexes_missing_month():
    observation = make_observation_dataset(
        ensembles=None,
    ).sel(month=[0.5])
    dataset = bare_dataset(
        observation_dataset=observation,
    )

    with pytest.raises(
        ValueError,
        match="observation coordinates",
    ):
        dataset.get_obs_indexes(
            make_sample_coords(
                years=(2000,),
                lead_times=(2,),
                ensembles=None,
            )
        )


def test_get_target_shape_without_observation():
    dataset = bare_dataset(
        observation_dataset=None,
    )

    with patch.object(
        TrainDataset,
        "get_input_shape",
        return_value=(2, 3),
    ) as get_shape:
        result = dataset.get_target_shape()

    assert result == (2, 3)
    get_shape.assert_called_once_with()


def test_get_target_shape_observation_without_flattener(
    monkeypatch,
):
    class FakeFlatten:
        pass

    monkeypatch.setattr(
        ("cccma_ppp.preprocessing.utils_preprocessing.Flattennanremove"),
        FakeFlatten,
    )

    observation = make_data_config(
        names=("tas",),
        pipeline=DummyPipeline(),
    )

    dataset = bare_dataset(
        observation_dataset=object(),
        config=SimpleNamespace(
            observation=observation,
        ),
    )

    assert dataset.get_target_shape() == (
        1,
        2,
        2,
    )


def test_index_observation_returns_none():
    dataset = bare_dataset(
        observation_dataset=None,
    )

    assert dataset._index_observation_dataset(0) is None


def test_index_observation_without_ensembles():
    observation_dataset = make_observation_dataset(
        ensembles=None,
    )
    pipeline = DummyPipeline()

    dataset = bare_dataset(
        observation_dataset=(observation_dataset),
        obs_indexes={
            "year": np.asarray([0]),
            "month": np.asarray([0]),
        },
        config=SimpleNamespace(
            observation=SimpleNamespace(preprocessing_pipeline=(pipeline))
        ),
    )

    with patch.object(
        module,
        "_unwrap_data_variables",
        side_effect=lambda value: value,
    ):
        result = dataset._index_observation_dataset(0)

    assert result is not None
    assert "ensembles" not in result.dims
    assert len(pipeline.transform_calls) == 1


def test_index_observation_with_random_ensemble(
    monkeypatch,
):
    observation_dataset = make_observation_dataset()
    pipeline = DummyPipeline()

    dataset = bare_dataset(
        observation_dataset=(observation_dataset),
        obs_indexes={
            "year": np.asarray([0]),
            "month": np.asarray([0]),
        },
        config=SimpleNamespace(
            observation=SimpleNamespace(preprocessing_pipeline=(pipeline))
        ),
    )

    randint = MagicMock(return_value=1)

    monkeypatch.setattr(
        module.np.random,
        "randint",
        randint,
    )

    with patch.object(
        module,
        "_unwrap_data_variables",
        side_effect=lambda value: value,
    ):
        result = dataset._index_observation_dataset(0)

    randint.assert_called_once_with(2)
    np.testing.assert_array_equal(
        result.coords["ensembles"].values,
        [1],
    )


def make_getitem_dataset(
    *,
    autoencoding=False,
    write_condition=False,
    concat_condition=False,
    return_metadata=False,
    time_features=None,
):
    sample_coords = {
        "year": np.asarray([2000]),
        "lead_time": np.asarray([1]),
    }

    reference = make_config_stub()

    dataset = bare_dataset(
        config=reference,
        sample_coords=sample_coords,
        return_metadata=return_metadata,
        time_features=AddedTimeFeatures(
            reference,
            time_features,
        ),
    )

    dataset._index_condition_dataset = MagicMock(
        return_value=xr.DataArray(
            np.asarray([3.0]),
            dims=("channels",),
        )
    )
    dataset._index_observation_dataset = MagicMock(
        return_value=xr.DataArray(
            np.asarray([2.0]),
            dims=("channels",),
        )
    )
    dataset._index_model_dataset = MagicMock(
        return_value=xr.DataArray(
            np.asarray([1.0]),
            dims=("channels",),
        )
    )

    type(dataset)._autoencoding_model_data = property(lambda self: autoencoding)
    type(dataset)._write_condition_to_input = property(lambda self: write_condition)
    type(dataset)._concat_condition_to_input = property(lambda self: concat_condition)

    return dataset


@pytest.mark.pruned
def test_getitem_autoencoding_target():
    dataset = make_getitem_dataset(
        autoencoding=True,
    )

    result = dataset[0]

    torch.testing.assert_close(
        result["input"],
        result["target"],
    )


@pytest.mark.pruned
def test_getitem_condition_replaces_input():
    dataset = make_getitem_dataset(
        write_condition=True,
    )

    result = dataset[0]

    torch.testing.assert_close(
        result["input"],
        torch.tensor([3.0]),
    )


def test_getitem_concatenates_condition():
    dataset = make_getitem_dataset(
        concat_condition=True,
    )

    result = dataset[0]

    torch.testing.assert_close(
        result["input"],
        torch.tensor([1.0, 3.0]),
    )


@pytest.mark.pruned
def test_getitem_with_time_features():
    dataset = make_getitem_dataset(
        time_features=[
            "year",
            "lead_time",
            "month_sin",
            "month_cos",
        ],
    )

    result = dataset[0]

    assert result["added_features"] is not None
    assert result["added_features"].shape == (4,)


@pytest.mark.pruned
def test_getitem_without_metadata_returns_dict():
    dataset = make_getitem_dataset(
        return_metadata=False,
    )

    result = dataset[0]

    assert isinstance(result, dict)


def test_getitem_with_metadata_returns_tuple():
    dataset = make_getitem_dataset(
        return_metadata=True,
    )

    result, selection = dataset[0]

    assert isinstance(result, dict)
    assert selection == {
        "year": 2000,
        "lead_time": 1,
    }


@pytest.mark.parametrize(
    (
        "condition_method",
        "ensemble_mean",
        "expected_exception",
    ),
    [
        ("same_member", True, ValueError),
        ("same_member", False, None),
        ("ensemble_mean", True, None),
        ("cross_ensemble", False, None),
        ("static", False, None),
        (None, True, AttributeError),
    ],
)
def test_check_model_branches(
    condition_method,
    ensemble_mean,
    expected_exception,
):
    model = make_data_config(
        ensemble_mean=ensemble_mean,
    )

    config = bare_config(
        model=model,
        condition_method=condition_method,
    )

    if expected_exception is None:
        assert TrainDatasetConfig._check_model(config) is config
    else:
        with pytest.raises(
            expected_exception,
            match=None,
        ):
            TrainDatasetConfig._check_model(config)


@pytest.mark.parametrize(
    "condition_method",
    [
        "cross_ensemble",
        "same_member",
    ],
)
def test_check_condition_member_methods_reject_ensemble_mean(
    condition_method,
):
    condition = make_data_config(
        ensemble_mean=True,
    )

    config = bare_config(
        condition=condition,
        condition_method=condition_method,
        effective_condition=condition,
    )

    with pytest.raises(
        ValueError,
        match="ensemble_mean cannot be True",
    ):
        TrainDatasetConfig._check_condition(config)


@pytest.mark.parametrize(
    "condition_method",
    [
        "cross_ensemble",
        "same_member",
    ],
)
def test_check_condition_member_methods_accept_ensemble_data(
    condition_method,
):
    condition = make_data_config(
        ensembles=(0, 1),
        ensemble_mean=False,
    )

    config = bare_config(
        condition=condition,
        condition_method=condition_method,
        effective_condition=condition,
    )

    assert TrainDatasetConfig._check_condition(config) is config


@pytest.mark.pruned
def test_available_times_without_observation():
    model = make_data_config(
        years=(1999, 2000, 2001),
    )

    config = bare_config(
        model=model,
        observation=None,
    )

    np.testing.assert_array_equal(
        config.available_times,
        [1999, 2000, 2001],
    )


@pytest.mark.pruned
def test_available_times_respects_model_coordinate_values():
    model = make_data_config(
        years=(1999, 2000, 2001, 2002),
    )
    observation = make_data_config(
        years=(2000, 2001, 2002, 2003),
    )

    model.year_range = np.asarray([1999, 2000, 2001, 2002, 2003])

    config = bare_config(
        model=model,
        observation=observation,
    )

    np.testing.assert_array_equal(
        config.available_times,
        [2000, 2001, 2002],
    )


@pytest.mark.parametrize(
    (
        "dimension",
        "model_values",
        "observation_values",
    ),
    [
        ("lat", (0, 1), (8, 9)),
        ("lon", (10, 20), (30, 40)),
    ],
)
def test_check_observation_warns_for_each_coordinate_mismatch(
    dimension,
    model_values,
    observation_values,
):
    model_kwargs = {
        dimension: model_values,
    }
    observation_kwargs = {
        dimension: observation_values,
    }

    model = make_data_config(**model_kwargs)
    observation = make_data_config(**observation_kwargs)

    config = bare_config(
        model=model,
        observation=observation,
        condition_method="static",
    )

    with pytest.warns(
        UserWarning,
        match=dimension,
    ):
        assert config._check_observation() is config


@pytest.mark.parametrize(
    "dimension",
    [
        "lat",
        "lon",
    ],
)
def test_check_observation_warns_when_nn_dimension_missing(
    dimension,
):
    model = make_data_config()
    observation = make_data_config()

    del model.info.coords[dimension]

    config = bare_config(
        model=model,
        observation=observation,
        condition_method="static",
    )

    with pytest.warns(
        UserWarning,
        match=f"observation data has NN dim {dimension}",
    ):
        assert config._check_observation() is config


@pytest.mark.pruned
def test_check_observation_collects_multiple_warnings():
    model = make_data_config(
        lat=(0, 1),
        lon=(10, 20),
    )
    observation = make_data_config(
        lat=(8, 9),
        lon=(30, 40),
    )

    config = bare_config(
        model=model,
        observation=observation,
        condition_method="static",
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = config._check_observation()

    assert result is config
    assert len(caught) == 2
    assert any("lat" in str(item.message) for item in caught)
    assert any("lon" in str(item.message) for item in caught)


@pytest.mark.pruned
def test_get_obs_indexes_reports_missing_year_values():
    dataset = bare_dataset(
        observation_dataset=make_observation_dataset(
            ensembles=None,
        )
    )

    with pytest.raises(ValueError) as exc_info:
        dataset.get_obs_indexes(
            make_sample_coords(
                years=(2000, 2099),
                lead_times=(1, 1),
                ensembles=None,
            )
        )

    message = str(exc_info.value)

    assert "observation coordinates" in message
    assert "year" in message
    assert "2099" in message


@pytest.mark.pruned
def test_get_obs_indexes_reports_missing_month_values():
    observation = make_observation_dataset(
        ensembles=None,
    ).sel(
        month=[0.5],
    )

    dataset = bare_dataset(
        observation_dataset=observation,
    )

    with pytest.raises(ValueError) as exc_info:
        dataset.get_obs_indexes(
            make_sample_coords(
                years=(2000, 2000),
                lead_times=(1, 2),
                ensembles=None,
            )
        )

    message = str(exc_info.value)

    assert "observation coordinates" in message
    assert "month" in message


@pytest.mark.pruned
def test_get_obs_indexes_reports_multiple_missing_dimensions():
    observation = make_observation_dataset(
        ensembles=None,
    ).sel(
        year=[2000],
        month=[0.5],
    )

    dataset = bare_dataset(
        observation_dataset=observation,
    )

    with pytest.raises(ValueError) as exc_info:
        dataset.get_obs_indexes(
            make_sample_coords(
                years=(2099,),
                lead_times=(2,),
                ensembles=None,
            )
        )

    message = str(exc_info.value)

    assert "year" in message
    assert "month" in message


@pytest.mark.pruned
def test_get_target_shape_multiple_observation_variables_without_flattener(
    monkeypatch,
):
    class FakeFlatten:
        pass

    monkeypatch.setattr(
        "cccma_ppp.preprocessing.utils_preprocessing.Flattennanremove",
        FakeFlatten,
    )

    observation = make_data_config(
        names=("tas", "pr"),
        lat=(0, 1, 2),
        lon=(10, 20, 30, 40),
        pipeline=DummyPipeline(),
    )

    dataset = bare_dataset(
        observation_dataset=object(),
        config=SimpleNamespace(
            observation=observation,
        ),
    )

    assert dataset.get_target_shape() == (
        2,
        3,
        4,
    )


@pytest.mark.parametrize(
    (
        "features",
        "expected",
    ),
    [
        (None, 0),
        ([], 0),
        (["year"], 1),
        (["year", "lead_time"], 2),
        (
            [
                "year",
                "lead_time",
                "month_sin",
                "month_cos",
            ],
            4,
        ),
    ],
)
def test_get_added_features_dim(
    features,
    expected,
):
    config = make_config_stub()
    time_features = AddedTimeFeatures(
        config,
        features,
    )

    dataset = bare_dataset(
        time_features=time_features,
    )

    assert dataset.get_added_features_dim() == expected


@pytest.mark.pruned
def test_index_observation_random_ensemble_lower_bound(
    monkeypatch,
):
    observation_dataset = make_observation_dataset(
        ensembles=(5, 8, 13),
    )
    pipeline = DummyPipeline()

    dataset = bare_dataset(
        observation_dataset=observation_dataset,
        obs_indexes={
            "year": np.asarray([0]),
            "month": np.asarray([0]),
        },
        config=SimpleNamespace(
            observation=SimpleNamespace(
                preprocessing_pipeline=pipeline,
            )
        ),
    )

    randint = MagicMock(return_value=0)

    monkeypatch.setattr(
        module.np.random,
        "randint",
        randint,
    )

    with patch.object(
        module,
        "_unwrap_data_variables",
        side_effect=lambda value: value,
    ):
        result = dataset._index_observation_dataset(0)

    randint.assert_called_once_with(3)

    np.testing.assert_array_equal(
        result.coords["ensembles"].values,
        [5],
    )


@pytest.mark.pruned
def test_index_observation_random_ensemble_upper_bound(
    monkeypatch,
):
    observation_dataset = make_observation_dataset(
        ensembles=(5, 8, 13),
    )
    pipeline = DummyPipeline()

    dataset = bare_dataset(
        observation_dataset=observation_dataset,
        obs_indexes={
            "year": np.asarray([0]),
            "month": np.asarray([0]),
        },
        config=SimpleNamespace(
            observation=SimpleNamespace(
                preprocessing_pipeline=pipeline,
            )
        ),
    )

    randint = MagicMock(return_value=2)

    monkeypatch.setattr(
        module.np.random,
        "randint",
        randint,
    )

    with patch.object(
        module,
        "_unwrap_data_variables",
        side_effect=lambda value: value,
    ):
        result = dataset._index_observation_dataset(0)

    randint.assert_called_once_with(3)

    np.testing.assert_array_equal(
        result.coords["ensembles"].values,
        [13],
    )


@pytest.mark.pruned
def test_index_observation_calls_pipeline_before_unwrap():
    observation_dataset = make_observation_dataset(
        ensembles=None,
    )
    events = []

    class RecordingPipeline:
        def transform(self, value):
            events.append("transform")
            return value

    dataset = bare_dataset(
        observation_dataset=observation_dataset,
        obs_indexes={
            "year": np.asarray([0]),
            "month": np.asarray([0]),
        },
        config=SimpleNamespace(
            observation=SimpleNamespace(
                preprocessing_pipeline=RecordingPipeline(),
            )
        ),
    )

    def unwrap(value):
        events.append("unwrap")
        return value

    with patch.object(
        module,
        "_unwrap_data_variables",
        side_effect=unwrap,
    ):
        dataset._index_observation_dataset(0)

    assert events == [
        "transform",
        "unwrap",
    ]


@pytest.mark.pruned
def test_getitem_calls_all_index_helpers():
    dataset = make_getitem_dataset()

    dataset[0]

    dataset._index_condition_dataset.assert_called_once_with(0)
    dataset._index_observation_dataset.assert_called_once_with(0)
    dataset._index_model_dataset.assert_called_once_with(0)


@pytest.mark.pruned
def test_getitem_returns_float32_tensors():
    dataset = make_getitem_dataset(
        time_features=[
            "year",
            "lead_time",
        ],
    )

    result = dataset[0]

    assert result["input"].dtype == torch.float32
    assert result["target"].dtype == torch.float32
    assert result["added_features"].dtype == torch.float32


def test_getitem_autoencoding_happens_before_condition_replacement():
    dataset = make_getitem_dataset(
        autoencoding=True,
        write_condition=True,
    )

    result = dataset[0]

    torch.testing.assert_close(
        result["input"],
        torch.tensor([3.0]),
    )
    torch.testing.assert_close(
        result["target"],
        torch.tensor([1.0]),
    )


@pytest.mark.pruned
def test_getitem_condition_replacement_takes_precedence_over_concat():
    dataset = make_getitem_dataset(
        write_condition=True,
        concat_condition=True,
    )

    result = dataset[0]

    torch.testing.assert_close(
        result["input"],
        torch.tensor([3.0]),
    )


@pytest.mark.pruned
def test_getitem_concatenation_preserves_channel_order():
    dataset = make_getitem_dataset(
        concat_condition=True,
    )

    dataset._index_model_dataset.return_value = xr.DataArray(
        np.asarray(
            [
                1.0,
                2.0,
            ]
        ),
        dims=("channels",),
        coords={
            "channels": [
                "model_a",
                "model_b",
            ]
        },
    )

    dataset._index_condition_dataset.return_value = xr.DataArray(
        np.asarray(
            [
                3.0,
                4.0,
            ]
        ),
        dims=("channels",),
        coords={
            "channels": [
                "condition_a",
                "condition_b",
            ]
        },
    )

    result = dataset[0]

    torch.testing.assert_close(
        result["input"],
        torch.tensor(
            [
                1.0,
                2.0,
                3.0,
                4.0,
            ]
        ),
    )


@pytest.mark.parametrize(
    (
        "lead_time",
        "expected_count",
    ),
    [
        (1, 4),
        (2, 4),
        (12, 4),
        (13, 4),
    ],
)
def test_getitem_all_time_features_for_multiple_leads(
    lead_time,
    expected_count,
):
    dataset = make_getitem_dataset(
        time_features=[
            "year",
            "lead_time",
            "month_sin",
            "month_cos",
        ],
    )

    dataset.sample_coords = {
        "year": np.asarray([2000]),
        "lead_time": np.asarray([lead_time]),
    }

    result = dataset[0]

    assert result["added_features"] is not None
    assert result["added_features"].shape == (expected_count,)
    assert torch.isfinite(result["added_features"]).all()


@pytest.mark.pruned
def test_getitem_metadata_values_are_scalars():
    dataset = make_getitem_dataset(
        return_metadata=True,
    )

    _, selection = dataset[0]

    assert np.isscalar(selection["year"])
    assert np.isscalar(selection["lead_time"])


@pytest.mark.pruned
def test_getitem_selects_requested_sample_index():
    reference = make_config_stub()

    dataset = bare_dataset(
        config=reference,
        sample_coords={
            "year": np.asarray(
                [
                    2000,
                    2001,
                ]
            ),
            "lead_time": np.asarray(
                [
                    1,
                    13,
                ]
            ),
        },
        return_metadata=True,
        time_features=AddedTimeFeatures(
            reference,
            None,
        ),
    )

    dataset._index_condition_dataset = MagicMock(
        return_value=None,
    )
    dataset._index_observation_dataset = MagicMock(
        return_value=xr.DataArray(
            np.asarray([2.0]),
            dims=("channels",),
        )
    )
    dataset._index_model_dataset = MagicMock(
        return_value=xr.DataArray(
            np.asarray([1.0]),
            dims=("channels",),
        )
    )

    with (
        patch.object(
            TrainDataset,
            "_autoencoding_model_data",
            new_callable=PropertyMock,
            return_value=False,
        ),
        patch.object(
            TrainDataset,
            "_write_condition_to_input",
            new_callable=PropertyMock,
            return_value=False,
        ),
        patch.object(
            TrainDataset,
            "_concat_condition_to_input",
            new_callable=PropertyMock,
            return_value=False,
        ),
    ):
        _, selection = dataset[1]

    assert selection == {
        "year": 2001,
        "lead_time": 13,
    }

    dataset._index_condition_dataset.assert_called_once_with(1)
    dataset._index_observation_dataset.assert_called_once_with(1)
    dataset._index_model_dataset.assert_called_once_with(1)


@pytest.mark.pruned
def test_post_init_passes_load_false_to_observation_loader():
    observation = make_data_config(
        paths=("obs.nc",),
    )

    config = make_config_stub(
        observation=observation,
    )

    features = AddedTimeFeatures(
        config,
        None,
    )

    dataset = object.__new__(TrainDataset)
    dataset.config = config
    dataset.requested_years = [2000]
    dataset.time_features = features
    dataset.mask = None
    dataset.return_metadata = False
    dataset.load = False
    dataset.sample_coords = make_sample_coords()
    dataset.observation_dataset = None

    loaded = make_observation_dataset()

    with (
        patch.object(
            DatasetABC,
            "__init__",
            return_value=None,
        ),
        patch.object(
            TrainDataset,
            "_load_xarray_data",
            return_value=loaded,
        ) as loader,
        patch.object(
            TrainDataset,
            "get_obs_indexes",
            return_value={
                "year": np.asarray([0]),
                "month": np.asarray([0]),
            },
        ),
    ):
        dataset.__post_init__()

    loader.assert_called_once_with(
        observation,
        load=False,
    )


@pytest.mark.pruned
def test_post_init_sets_observation_indexes():
    observation = make_data_config(
        paths=("obs.nc",),
    )

    config = make_config_stub(
        observation=observation,
    )

    features = AddedTimeFeatures(
        config,
        None,
    )

    dataset = object.__new__(TrainDataset)
    dataset.config = config
    dataset.requested_years = [2000]
    dataset.time_features = features
    dataset.mask = None
    dataset.return_metadata = False
    dataset.load = False
    dataset.sample_coords = make_sample_coords()
    dataset.observation_dataset = None

    loaded = make_observation_dataset()
    expected_indexes = {
        "year": np.asarray([0]),
        "month": np.asarray([0]),
    }

    with (
        patch.object(
            DatasetABC,
            "__init__",
            return_value=None,
        ),
        patch.object(
            TrainDataset,
            "_load_xarray_data",
            return_value=loaded,
        ),
        patch.object(
            TrainDataset,
            "get_obs_indexes",
            return_value=expected_indexes,
        ) as get_indexes,
    ):
        dataset.__post_init__()

    assert dataset.obs_indexes is expected_indexes
    get_indexes.assert_called_once_with(dataset.sample_coords)

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