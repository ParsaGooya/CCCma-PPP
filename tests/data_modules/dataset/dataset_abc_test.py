from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import cccma_ppp.data_modules.dataset.dataset_abc as module
from cccma_ppp.data_modules.dataset.dataset_abc import (
    DatasetABC,
    DatasetConfigABC,
    lead_months_config,
)


class ConcreteConfig(DatasetConfigABC):
    def __init__(
        self,
        model=None,
        condition=None,
        condition_method=None,
        time_features=None,
        lead_months=None,
        num_input_lead_months=3,
        available_times=(2000, 2001),
        effective_input=None,
        observation=None,
        fitted_preprocessors=True,
    ):
        self.model = model
        self.condition = condition
        self.condition_method = condition_method
        self.time_features = time_features
        self.lead_months = lead_months
        self._num_input_lead_months = num_input_lead_months
        self._available_times = np.asarray(available_times)
        self._effective_input = effective_input or model or condition
        self.observation = observation
        self._fitted_preprocessors = fitted_preprocessors
        super().__init__()

    def _check_model(self):
        return self

    def _check_condition(self):
        return self

    @property
    def available_times(self):
        return self._available_times

    @property
    def ds_operator(self):
        return "operator"

    @property
    def num_input_lead_months(self):
        return self._num_input_lead_months

    @property
    def effective_input(self):
        return self._effective_input

    def build_dataset(self):
        return "dataset"


class ConcreteDataset(DatasetABC):
    def __init__(
        self,
        config,
        requested_years,
        mask,
        load_model=True,
        load=False,
        write_condition=False,
        concat_condition=False,
    ):
        self.config = config
        self.requested_years = requested_years
        self.mask = mask
        self.load = load
        self.return_metadata = False
        self._load_model_value = load_model
        self._write_condition_value = write_condition
        self._concat_condition_value = concat_condition
        super().__init__()

    @property
    def _load_model(self):
        return self._load_model_value

    @property
    def _write_condition_to_input(self):
        return self._write_condition_value

    @property
    def _concat_condition_to_input(self):
        return self._concat_condition_value


class BareDataset(DatasetABC):
    @property
    def _load_model(self):
        return getattr(self, "_load_model_value", True)

    @property
    def _write_condition_to_input(self):
        return getattr(self, "_write_condition_value", False)

    @property
    def _concat_condition_to_input(self):
        return getattr(self, "_concat_condition_value", False)


class Coordinate:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.size = self.values.size

    def equals(self, other):
        return np.array_equal(self.values, other.values)


def make_pipeline(fitted_preprocessors=None):
    pipeline = MagicMock()
    pipeline.fitted_preprocessors = fitted_preprocessors or []
    pipeline.transform.side_effect = lambda value: value
    return pipeline


def make_config_data(
    paths=("model.nc",),
    names=("tas",),
    ensembles=("r1", "r2"),
    ensemble_list=("r1", "r2"),
    ensemble_mean=False,
    times=(2000, 2001),
    lead_times=(1, 2, 3),
    extra_coords=None,
    pipeline=None,
):
    coords = {
        "time": Coordinate(times),
        "lead_time": Coordinate(lead_times),
    }

    if ensembles is not None:
        coords["ensembles"] = Coordinate(ensembles)

    if extra_coords:
        coords.update(extra_coords)

    sizes = {key: coordinate.size for key, coordinate in coords.items()}

    return SimpleNamespace(
        paths=list(paths),
        list_paths=list(paths),
        names=list(names),
        ensemble_list=(list(ensemble_list) if ensemble_list is not None else None),
        ensemble_mean=ensemble_mean,
        preprocessing_pipeline=pipeline or make_pipeline(),
        concat_dim="time",
        file_type="netcdf",
        rename_dict={"old": "new"},
        info=SimpleNamespace(
            coords=coords,
            sizes=sizes,
        ),
    )


def make_xarray(
    times=(2000, 2001),
    lead_times=(1, 2),
    ensembles=None,
):
    coords = {
        "time": list(times),
        "lead_time": list(lead_times),
    }

    dims = ["time", "lead_time"]
    shape = [len(times), len(lead_times)]

    if ensembles is not None:
        coords["ensembles"] = list(ensembles)
        dims.append("ensembles")
        shape.append(len(ensembles))

    return xr.DataArray(
        np.arange(np.prod(shape), dtype=float).reshape(shape),
        dims=dims,
        coords=coords,
        name="tas",
    )


def make_mask(
    times=(2000, 2001),
    lead_times=(1, 2),
    values=None,
):
    if values is None:
        values = np.zeros(
            (len(times), len(lead_times)),
            dtype=float,
        )

    return xr.DataArray(
        values,
        dims=("time", "lead_time"),
        coords={
            "time": list(times),
            "lead_time": list(lead_times),
        },
    )


def bare_config(**kwargs):
    config = object.__new__(ConcreteConfig)
    config.model = kwargs.get("model")
    config.condition = kwargs.get("condition")
    config.condition_method = kwargs.get("condition_method")
    config.time_features = kwargs.get("time_features")
    config.lead_months = kwargs.get("lead_months")
    config.observation = kwargs.get("observation")
    config._num_input_lead_months = kwargs.get(
        "num_input_lead_months",
        3,
    )
    config._available_times = np.asarray(kwargs.get("available_times", [2000, 2001]))
    config._effective_input = kwargs.get("effective_input")
    config._fitted_preprocessors = kwargs.get(
        "fitted_preprocessors",
        True,
    )
    config._effective_condition = kwargs.get("effective_condition")
    return config


def bare_dataset(**kwargs):
    dataset = object.__new__(BareDataset)

    for key, value in kwargs.items():
        setattr(dataset, key, value)

    return dataset


def test_lead_months_requires_list_or_end():
    with pytest.raises(ValueError, match="Provide a list"):
        lead_months_config()


def test_lead_months_explicit_list():
    config = lead_months_config(list_months=[1, 3])

    assert config.build_lead_months() == [1, 3]


def test_lead_months_range():
    config = lead_months_config(start=2, end=4)

    np.testing.assert_array_equal(
        config.build_lead_months(),
        [2, 3, 4],
    )


def test_dataset_config_is_abstract():
    with pytest.raises(TypeError):
        DatasetConfigABC()


def test_dataset_is_abstract():
    with pytest.raises(TypeError):
        DatasetABC()


def test_config_requires_input_source():
    with pytest.raises(
        ValueError,
        match="either model or condition",
    ):
        ConcreteConfig()


@pytest.mark.parametrize(
    "condition_method",
    [
        None,
        "ensemble_mean",
        "cross_ensemble",
        "same_member",
        "static",
    ],
)
def test_valid_condition_methods(condition_method):
    config = ConcreteConfig(
        model=make_config_data(),
        condition_method=condition_method,
    )

    assert config.condition_method == condition_method


def test_invalid_condition_method():
    with pytest.raises(
        ValueError,
        match="Invalid condition_method",
    ):
        ConcreteConfig(
            model=make_config_data(),
            condition_method="invalid",
        )


@pytest.mark.parametrize(
    "time_features",
    [
        None,
        [],
        ["year"],
        ["lead_time"],
        ["month_sin", "month_cos"],
        ["year", "lead_time", "month_sin", "month_cos"],
    ],
)
def test_valid_time_features(time_features):
    config = ConcreteConfig(
        model=make_config_data(),
        time_features=time_features,
    )

    assert config.time_features == time_features


def test_invalid_time_features():
    with pytest.raises(
        ValueError,
        match="Invalid time features",
    ):
        ConcreteConfig(
            model=make_config_data(),
            time_features=["day"],
        )


def test_default_lead_months():
    config = ConcreteConfig(
        model=make_config_data(),
        num_input_lead_months=4,
    )

    np.testing.assert_array_equal(
        config.lead_months,
        [1, 2, 3, 4],
    )


def test_resolves_lead_months_config():
    config = ConcreteConfig(
        model=make_config_data(),
        lead_months=lead_months_config(start=2, end=3),
    )

    np.testing.assert_array_equal(
        config.lead_months,
        [2, 3],
    )


def test_preserves_non_config_lead_months():
    lead_months = np.asarray([1, 2])

    config = ConcreteConfig(
        model=make_config_data(),
        lead_months=lead_months,
    )

    assert config.lead_months is lead_months


def test_rejects_unavailable_lead_month():
    with pytest.raises(
        ValueError,
        match="Maximum available lead months is 3",
    ):
        ConcreteConfig(
            model=make_config_data(),
            lead_months=[1, 4],
            num_input_lead_months=3,
        )


def test_condition_requires_sample_dimension(monkeypatch):
    monkeypatch.setattr(
        module,
        "required_sample_dimensions",
        ("time", "lead_time"),
    )

    model = make_config_data()
    condition = make_config_data()
    del condition.info.coords["lead_time"]

    with pytest.raises(
        ValueError,
        match="same dimestions",
    ):
        ConcreteConfig(
            model=model,
            condition=condition,
            condition_method="ensemble_mean",
        )


def test_condition_requires_matching_sample_coordinates(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "required_sample_dimensions",
        ("time", "lead_time"),
    )

    with pytest.raises(
        ValueError,
        match="same time coordinates",
    ):
        ConcreteConfig(
            model=make_config_data(times=(2000, 2001)),
            condition=make_config_data(times=(2000,)),
            condition_method="ensemble_mean",
        )


def test_condition_requires_matching_lead_coordinates(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "required_sample_dimensions",
        ("time", "lead_time"),
    )

    with pytest.raises(
        ValueError,
        match="same lead_time coordinates",
    ):
        ConcreteConfig(
            model=make_config_data(lead_times=(1, 2, 3)),
            condition=make_config_data(lead_times=(1, 2)),
            condition_method="ensemble_mean",
        )


def test_static_skips_sample_coordinate_validation(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "required_sample_dimensions",
        ("time", "lead_time"),
    )

    condition = make_config_data(
        times=(1990,),
        lead_times=(12,),
    )

    config = ConcreteConfig(
        model=make_config_data(),
        condition=condition,
        condition_method="static",
    )

    assert config.effective_condition is condition


def test_same_member_exposes_initialization_order_bug():
    model = make_config_data()
    condition = make_config_data(
        paths=("condition.nc",),
    )

    with pytest.raises(AttributeError):
        ConcreteConfig(
            model=model,
            condition=condition,
            condition_method="same_member",
        )


def test_same_member_requires_equal_ensembles():
    model = make_config_data(ensembles=("r1", "r2"))
    condition = make_config_data(
        paths=("condition.nc",),
        ensembles=("r1", "r3"),
    )
    config = bare_config(
        model=model,
        condition=condition,
        condition_method="same_member",
        effective_condition=condition,
    )

    with pytest.raises(
        ValueError,
        match="same ensemble members",
    ):
        config._check_model_vs_condition()


def test_same_member_requires_model_ensembles():
    model = make_config_data(ensembles=None)
    condition = make_config_data(
        paths=("condition.nc",),
    )
    config = bare_config(
        model=model,
        condition=condition,
        condition_method="same_member",
        effective_condition=condition,
    )

    with pytest.raises(
        ValueError,
        match="must have ensembles",
    ):
        config._check_model_vs_condition()


def test_same_member_requires_condition_ensembles():
    model = make_config_data()
    condition = make_config_data(
        paths=("condition.nc",),
        ensembles=None,
    )
    config = bare_config(
        model=model,
        condition=condition,
        condition_method="same_member",
        effective_condition=condition,
    )

    with pytest.raises(
        ValueError,
        match="must have ensembles",
    ):
        config._check_model_vs_condition()


def test_same_member_matching_ensembles_passes():
    model = make_config_data()
    condition = make_config_data(
        paths=("condition.nc",),
    )
    config = bare_config(
        model=model,
        condition=condition,
        condition_method="same_member",
        effective_condition=condition,
    )

    assert config._check_model_vs_condition() is None


def test_observation_missing_nn_dimension(monkeypatch):
    monkeypatch.setattr(
        module,
        "supported_NN_dimensions_sorted",
        ["lat"],
    )

    model = make_config_data(extra_coords={"lat": Coordinate([45, 46])})
    condition = make_config_data(
        paths=("condition.nc",),
    )

    with pytest.raises(TypeError):
        ConcreteConfig(
            model=model,
            condition=condition,
            condition_method="ensemble_mean",
            observation=object(),
        )


def test_observation_mismatched_nn_coordinates(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "supported_NN_dimensions_sorted",
        ["lat"],
    )

    model = make_config_data(extra_coords={"lat": Coordinate([45, 46])})
    condition = make_config_data(
        paths=("condition.nc",),
        extra_coords={"lat": Coordinate([40, 41])},
    )

    with pytest.raises(TypeError):
        ConcreteConfig(
            model=model,
            condition=condition,
            condition_method="ensemble_mean",
            observation=object(),
        )


def test_observation_matching_nn_coordinates(monkeypatch):
    monkeypatch.setattr(
        module,
        "supported_NN_dimensions_sorted",
        ["lat"],
    )

    model = make_config_data(extra_coords={"lat": Coordinate([45, 46])})
    condition = make_config_data(
        paths=("condition.nc",),
        extra_coords={"lat": Coordinate([45, 46])},
    )

    config = ConcreteConfig(
        model=model,
        condition=condition,
        condition_method="ensemble_mean",
        observation=object(),
    )

    assert config.effective_condition is condition


@pytest.mark.parametrize(
    "condition_method",
    ["ensemble_mean", "cross_ensemble", "same_member"],
)
def test_using_model_as_condition(condition_method):
    config = bare_config(
        model=make_config_data(),
        condition=None,
        condition_method=condition_method,
    )

    assert config._using_model_data_as_condition is True


@pytest.mark.parametrize(
    "condition_method",
    [None, "static"],
)
def test_not_using_model_as_condition(condition_method):
    config = bare_config(
        model=make_config_data(),
        condition=None,
        condition_method=condition_method,
    )

    assert config._using_model_data_as_condition is False


def test_same_condition_source_is_model_condition():
    model = make_config_data()
    condition = make_config_data()

    config = bare_config(
        model=model,
        condition=condition,
        condition_method=None,
    )

    assert config._using_model_data_as_condition is True


@pytest.mark.parametrize(
    ("paths", "names", "ensemble_list"),
    [
        (("other.nc",), ("tas",), ("r1", "r2")),
        (("model.nc",), ("pr",), ("r1", "r2")),
        (("model.nc",), ("tas",), ("r3",)),
    ],
)
def test_different_source_is_not_model_condition(
    paths,
    names,
    ensemble_list,
):
    config = bare_config(
        model=make_config_data(),
        condition=make_config_data(
            paths=paths,
            names=names,
            ensemble_list=ensemble_list,
        ),
        condition_method=None,
    )

    assert config._using_model_data_as_condition is False


def test_condition_without_model_is_not_model_condition():
    config = bare_config(
        model=None,
        condition=make_config_data(),
        condition_method="ensemble_mean",
    )

    assert config._using_model_data_as_condition is False


@pytest.mark.parametrize(
    ("condition_method", "ensemble_mean"),
    [
        ("ensemble_mean", True),
        ("cross_ensemble", False),
        ("same_member", False),
    ],
)
def test_model_as_condition(
    condition_method,
    ensemble_mean,
):
    model = make_config_data()
    config = bare_config(
        model=model,
        condition=None,
        condition_method=condition_method,
    )
    result = object()

    with patch.object(
        module,
        "ModelDataConfig",
        return_value=result,
    ) as constructor:
        assert config._model_as_condition() is result

    constructor.assert_called_once_with(
        paths=model.paths,
        names=model.names,
        preprocessing_pipeline=model.preprocessing_pipeline,
        ensemble_list=model.ensemble_list,
        concat_dim=model.concat_dim,
        file_type=model.file_type,
        ensemble_mean=ensemble_mean,
        rename_dict=model.rename_dict,
    )


def test_resolve_explicit_condition():
    condition = make_config_data()
    config = bare_config(
        model=make_config_data(),
        condition=condition,
        condition_method=None,
    )

    assert config._resolve_condition() is config
    assert config.effective_condition is condition


def test_resolve_model_condition():
    config = bare_config(
        model=make_config_data(),
        condition=None,
        condition_method="ensemble_mean",
    )
    result = object()

    with patch.object(
        config,
        "_model_as_condition",
        return_value=result,
    ):
        assert config._resolve_condition() is config

    assert config.effective_condition is result


def test_resolve_no_condition():
    config = bare_config(
        model=make_config_data(),
        condition=None,
        condition_method=None,
    )

    assert config._resolve_condition() is config
    assert config.effective_condition is None


def test_check_init_requires_fitted_preprocessors():
    dataset = bare_dataset(
        config=bare_config(
            fitted_preprocessors=False,
            available_times=[2000],
        ),
        requested_years=[2000],
    )

    with pytest.raises(
        RuntimeError,
        match="fit preprocessors",
    ):
        dataset._check_init()


def test_check_init_rejects_unavailable_years():
    dataset = bare_dataset(
        config=bare_config(
            fitted_preprocessors=True,
            available_times=[2000],
        ),
        requested_years=[2001],
    )

    with pytest.raises(
        ValueError,
        match="requested years",
    ):
        dataset._check_init()


def test_check_init_accepts_available_years():
    dataset = bare_dataset(
        config=bare_config(
            fitted_preprocessors=True,
            available_times=[2000, 2001],
        ),
        requested_years=[2000],
    )

    assert dataset._check_init() is None


def test_resolve_mask_creates_default_mask(monkeypatch):
    created = make_mask()

    monkeypatch.setattr(
        module,
        "required_sample_dimensions",
        ("time", "lead_time"),
    )

    creator = MagicMock(return_value=created)
    monkeypatch.setattr(
        module,
        "_create_train_mask",
        creator,
    )

    effective_input = make_config_data(
        times=(2000, 2001),
        lead_times=(1, 2),
    )

    dataset = bare_dataset(
        config=bare_config(
            available_times=[2000, 2001],
            effective_input=effective_input,
        ),
        mask=None,
    )

    assert dataset._resolve_mask() is None
    assert not bool(dataset.mask.any())

    creator.assert_called_once()
    np.testing.assert_array_equal(
        creator.call_args.kwargs["time"],
        [2000, 2001],
    )
    np.testing.assert_array_equal(
        creator.call_args.kwargs["lead_times"],
        [1, 2],
    )


def test_resolve_mask_requires_dimensions(monkeypatch):
    monkeypatch.setattr(
        module,
        "required_sample_dimensions",
        ("time", "lead_time"),
    )

    dataset = bare_dataset(
        config=bare_config(),
        mask=xr.DataArray(
            [False],
            dims=("time",),
            coords={"time": [2000]},
        ),
    )

    with pytest.raises(
        ValueError,
        match="mask must have",
    ):
        dataset._resolve_mask()


def test_sampling_selectors(monkeypatch):
    monkeypatch.setattr(
        module,
        "required_sample_dimensions",
        ("time", "lead_time"),
    )

    dataset = bare_dataset(
        config=SimpleNamespace(lead_months=[1, 2]),
        requested_years=[2000, 2001],
    )

    assert dataset._sampling_selectors == {
        "time": [2000, 2001],
        "lead_time": [1, 2],
    }


def test_prepare_sampling_mask_requires_selectors(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "required_sample_dimensions",
        ("time", "lead_time"),
    )

    dataset = bare_dataset(
        config=SimpleNamespace(),
        mask=make_mask(),
    )

    with pytest.raises(
        ValueError,
        match="No selectors provided",
    ):
        dataset._prepare_sampling_mask({"time": [2000]})


def test_prepare_sampling_mask_selects_values(monkeypatch):
    monkeypatch.setattr(
        module,
        "required_sample_dimensions",
        ("time", "lead_time"),
    )
    monkeypatch.setattr(
        module,
        "optional_sample_dimensions",
        (),
    )

    effective_input = make_config_data()
    dataset = bare_dataset(
        config=SimpleNamespace(
            effective_input=effective_input,
        ),
        mask=make_mask(
            times=(2000, 2001),
            lead_times=(1, 2),
        ),
    )

    result = dataset._prepare_sampling_mask(
        {
            "time": [2001],
            "lead_time": [2],
        }
    )

    assert result is dataset
    assert dataset.mask.sizes == {
        "time": 1,
        "lead_time": 1,
    }


def test_prepare_sampling_mask_expands_optional_dimension(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "required_sample_dimensions",
        ("time", "lead_time"),
    )
    monkeypatch.setattr(
        module,
        "optional_sample_dimensions",
        ("ensembles",),
    )

    effective_input = make_config_data(
        ensembles=("r1", "r2"),
    )

    dataset = bare_dataset(
        config=SimpleNamespace(
            effective_input=effective_input,
        ),
        mask=make_mask(),
    )

    dataset._prepare_sampling_mask(
        {
            "time": [2000],
            "lead_time": [1],
        }
    )

    assert dataset.mask.dims[0] == "ensembles"
    assert dataset.mask.sizes["ensembles"] == 2


def test_prepare_sampling_mask_skips_missing_optional_dimension(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "required_sample_dimensions",
        ("time", "lead_time"),
    )
    monkeypatch.setattr(
        module,
        "optional_sample_dimensions",
        ("members",),
    )

    effective_input = make_config_data()

    dataset = bare_dataset(
        config=SimpleNamespace(
            effective_input=effective_input,
        ),
        mask=make_mask(),
    )

    dataset._prepare_sampling_mask(
        {
            "time": [2000],
            "lead_time": [1],
        }
    )

    assert "members" not in dataset.mask.dims


def test_prepare_sampling_mask_skips_ensemble_mean(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "required_sample_dimensions",
        ("time", "lead_time"),
    )
    monkeypatch.setattr(
        module,
        "optional_sample_dimensions",
        ("ensembles",),
    )

    effective_input = make_config_data(
        ensemble_mean=True,
    )

    dataset = bare_dataset(
        config=SimpleNamespace(
            effective_input=effective_input,
        ),
        mask=make_mask(),
    )

    dataset._prepare_sampling_mask(
        {
            "time": [2000],
            "lead_time": [1],
        }
    )

    assert "ensembles" not in dataset.mask.dims


@pytest.mark.parametrize(
    "ensembles",
    [None, ("r1", "r2")],
)
@pytest.mark.parametrize(
    "load",
    [False, True],
)
def test_load_xarray_data(ensembles, load):
    config = make_config_data(ensembles=ensembles)
    dataset = bare_dataset()
    result = object()

    with patch.object(
        module,
        "_load_xarray_data",
        return_value=result,
    ) as loader:
        assert (
            dataset._load_xarray_data(
                config,
                load=load,
            )
            is result
        )

    loader.assert_called_once_with(
        config.list_paths,
        names=config.names,
        ensemble_mean=config.ensemble_mean,
        selection=(
            {"ensembles": config.info.coords["ensembles"]}
            if ensembles is not None
            else None
        ),
        concat_dim=config.concat_dim,
        rename_dict=config.rename_dict,
        load=load,
    )


def test_get_sampling_coords():
    dataset = bare_dataset(
        mask=make_mask(
            values=np.asarray(
                [
                    [np.nan, 0.0],
                    [1.0, np.nan],
                ]
            )
        )
    )

    result = dataset.get_sampling_coords()

    np.testing.assert_array_equal(
        result["time"],
        [2000, 2001],
    )
    np.testing.assert_array_equal(
        result["lead_time"],
        [2, 1],
    )


def test_get_model_indexes_returns_none():
    dataset = bare_dataset(
        _load_model_value=False,
    )

    assert dataset.get_model_indexes({"time": np.asarray([2000])}) is None


def test_get_model_indexes():
    dataset = bare_dataset(
        _load_model_value=True,
        model_dataset=make_xarray(),
    )

    result = dataset.get_model_indexes(
        {
            "time": np.asarray([2001]),
            "lead_time": np.asarray([2]),
        }
    )

    np.testing.assert_array_equal(result["time"], [1])
    np.testing.assert_array_equal(
        result["lead_time"],
        [1],
    )


def test_get_model_indexes_rejects_missing_coordinates():
    dataset = bare_dataset(
        _load_model_value=True,
        model_dataset=make_xarray(),
    )

    with pytest.raises(
        ValueError,
        match="not found in the model dataset",
    ):
        dataset.get_model_indexes({"time": np.asarray([1999])})


@pytest.mark.parametrize(
    "condition_method",
    [None, "static"],
)
def test_get_cond_indexes_returns_none(condition_method):
    dataset = bare_dataset(
        condition_dataset=(None if condition_method is None else make_xarray()),
        config=SimpleNamespace(condition_method=condition_method),
    )

    assert dataset.get_cond_indexes({"time": np.asarray([2000])}) is None


def test_get_cond_indexes_ignores_unavailable_dimensions():
    dataset = bare_dataset(
        condition_dataset=make_xarray(),
        config=SimpleNamespace(condition_method="ensemble_mean"),
    )

    result = dataset.get_cond_indexes(
        {
            "time": np.asarray([2000]),
            "lead_time": np.asarray([1]),
            "ensembles": np.asarray(["r1"]),
            "unknown": np.asarray([10]),
        }
    )

    assert set(result) == {"time", "lead_time"}


def test_get_cond_indexes_same_member_requires_ensembles():
    dataset = bare_dataset(
        condition_dataset=make_xarray(ensembles=("r1", "r2")),
        config=SimpleNamespace(condition_method="same_member"),
    )

    with pytest.raises(
        ValueError,
        match="requires ensemble coordinates",
    ):
        dataset.get_cond_indexes(
            {
                "time": np.asarray([2000]),
                "lead_time": np.asarray([1]),
            }
        )


def test_get_cond_indexes_same_member():
    dataset = bare_dataset(
        condition_dataset=make_xarray(ensembles=("r1", "r2")),
        config=SimpleNamespace(condition_method="same_member"),
    )

    result = dataset.get_cond_indexes(
        {
            "time": np.asarray([2000]),
            "lead_time": np.asarray([1]),
            "ensembles": np.asarray(["r2"]),
        }
    )

    np.testing.assert_array_equal(
        result["ensembles"],
        [1],
    )


def test_get_cond_indexes_rejects_missing_coordinates():
    dataset = bare_dataset(
        condition_dataset=make_xarray(),
        config=SimpleNamespace(condition_method="ensemble_mean"),
    )

    with pytest.raises(
        ValueError,
        match="conditioning coordinates were not found",
    ):
        dataset.get_cond_indexes({"time": np.asarray([1999])})


def test_get_input_shape_with_flattener(monkeypatch):
    class FakeFlattennanremove:
        pass

    flattener = FakeFlattennanremove()
    flattener.final_locations = np.zeros(5)

    pipeline = make_pipeline(fitted_preprocessors=[flattener])
    pipeline.get_preprocessors.return_value = flattener

    effective_input = make_config_data(
        names=("tas", "pr"),
        pipeline=pipeline,
    )
    effective_condition = make_config_data(
        names=("condition",),
    )

    fake_module = SimpleNamespace(Flattennanremove=FakeFlattennanremove)

    monkeypatch.setitem(
        __import__("sys").modules,
        "cccma_ppp.preprocessing.utils_preprocessing",
        fake_module,
    )

    dataset = bare_dataset(
        config=SimpleNamespace(
            effective_input=effective_input,
            effective_condition=effective_condition,
        ),
        _concat_condition_value=True,
    )

    assert dataset.get_input_shape() == (15,)


def test_get_input_shape_without_flattener(monkeypatch):
    class FakeFlattennanremove:
        pass

    fake_module = SimpleNamespace(Flattennanremove=FakeFlattennanremove)

    monkeypatch.setitem(
        __import__("sys").modules,
        "cccma_ppp.preprocessing.utils_preprocessing",
        fake_module,
    )
    monkeypatch.setattr(
        module,
        "supported_NN_dimensions_sorted",
        ["lat", "lon"],
    )

    effective_input = make_config_data(
        extra_coords={
            "lat": Coordinate([1, 2]),
            "lon": Coordinate([3, 4, 5]),
        }
    )

    dataset = bare_dataset(
        config=SimpleNamespace(
            effective_input=effective_input,
            effective_condition=None,
        ),
        _concat_condition_value=False,
    )

    assert dataset.get_input_shape() == (2, 3)


@pytest.mark.parametrize(
    ("time_features", "expected"),
    [
        (None, 0),
        ([], 0),
        (["year"], 1),
        (["year", "month_sin"], 2),
    ],
)
def test_get_added_features_dim(
    time_features,
    expected,
):
    dataset = bare_dataset(config=SimpleNamespace(time_features=time_features))

    assert dataset.get_added_features_dim() == expected


def test_index_condition_returns_none():
    dataset = bare_dataset(
        condition_dataset=None,
    )

    assert dataset._index_condition_dataset(0) is None


def test_index_static_condition():
    condition_dataset = make_xarray()
    pipeline = make_pipeline()
    effective_condition = make_config_data(pipeline=pipeline)
    dataset = bare_dataset(
        condition_dataset=condition_dataset,
        cond_indexes=None,
        config=SimpleNamespace(
            condition_method="static",
            effective_condition=effective_condition,
        ),
    )

    result = object()

    with patch.object(
        module,
        "_unwrap_data_variables",
        return_value=result,
    ) as unwrap:
        assert dataset._index_condition_dataset(0) is result

    pipeline.transform.assert_called_once()
    xr.testing.assert_identical(
        pipeline.transform.call_args.args[0],
        condition_dataset,
    )
    unwrap.assert_called_once()


def test_index_condition_with_indexes():
    condition_dataset = make_xarray()
    pipeline = make_pipeline()
    effective_condition = make_config_data(pipeline=pipeline)
    dataset = bare_dataset(
        condition_dataset=condition_dataset,
        cond_indexes={
            "time": np.asarray([1]),
            "lead_time": np.asarray([0]),
        },
        config=SimpleNamespace(
            condition_method="ensemble_mean",
            effective_condition=effective_condition,
        ),
    )

    result = dataset._index_condition_dataset(0)

    assert result.sizes["time"] == 1
    assert result.sizes["lead_time"] == 1
    assert result.time.item() == 2001
    assert result.lead_time.item() == 1


def test_index_cross_ensemble_condition():
    condition_dataset = make_xarray(ensembles=("r1", "r2"))
    pipeline = make_pipeline()
    effective_condition = make_config_data(pipeline=pipeline)
    dataset = bare_dataset(
        condition_dataset=condition_dataset,
        cond_indexes={
            "time": np.asarray([0]),
            "lead_time": np.asarray([0]),
        },
        config=SimpleNamespace(
            condition_method="cross_ensemble",
            effective_condition=effective_condition,
        ),
    )

    with patch.object(
        np.random,
        "randint",
        return_value=1,
    ) as randint:
        result = dataset._index_condition_dataset(0)

    randint.assert_called_once_with(2)
    assert result.ensembles.item() == "r2"


def test_index_model_returns_none():
    dataset = bare_dataset(
        _load_model_value=False,
    )

    assert dataset._index_model_dataset(0) is None


def test_index_model_dataset():
    model_dataset = make_xarray()
    pipeline = make_pipeline()
    model_config = make_config_data(pipeline=pipeline)
    dataset = bare_dataset(
        _load_model_value=True,
        model_dataset=model_dataset,
        model_indexes={
            "time": np.asarray([1]),
            "lead_time": np.asarray([0]),
        },
        config=SimpleNamespace(model=model_config),
    )

    result = dataset._index_model_dataset(0)

    assert result.time.item() == 2001
    assert result.lead_time.item() == 1
    pipeline.transform.assert_called_once()


def test_dataset_length():
    dataset = bare_dataset(
        sample_coords={
            "time": np.asarray([2000, 2001, 2002]),
            "lead_time": np.asarray([1, 1, 1]),
        }
    )

    assert len(dataset) == 3


def test_dataset_initialization(monkeypatch):
    monkeypatch.setattr(
        module,
        "required_sample_dimensions",
        ("time", "lead_time"),
    )
    monkeypatch.setattr(
        module,
        "optional_sample_dimensions",
        (),
    )

    model_config = make_config_data(
        times=(2000,),
        lead_times=(1,),
    )
    condition_config = make_config_data(
        paths=("condition.nc",),
        times=(2000,),
        lead_times=(1,),
    )

    config = bare_config(
        model=model_config,
        condition=condition_config,
        condition_method="ensemble_mean",
        lead_months=[1],
        available_times=[2000],
        effective_input=model_config,
        effective_condition=condition_config,
        fitted_preprocessors=True,
    )

    model_data = make_xarray(
        times=(2000,),
        lead_times=(1,),
    )
    condition_data = make_xarray(
        times=(2000,),
        lead_times=(1,),
    )

    with patch.object(
        DatasetABC,
        "_load_xarray_data",
        side_effect=[model_data, condition_data],
    ) as loader:
        dataset = ConcreteDataset(
            config=config,
            requested_years=[2000],
            mask=make_mask(
                times=(2000,),
                lead_times=(1,),
            ),
            load_model=True,
            load=True,
        )

    assert dataset.model_dataset is model_data
    assert dataset.condition_dataset is condition_data
    assert dataset.observation_dataset is None
    assert len(dataset) == 1
    assert loader.call_count == 2


def test_dataset_initialization_without_model_or_condition(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "required_sample_dimensions",
        ("time", "lead_time"),
    )
    monkeypatch.setattr(
        module,
        "optional_sample_dimensions",
        (),
    )

    effective_input = make_config_data(
        times=(2000,),
        lead_times=(1,),
    )

    config = bare_config(
        model=None,
        condition=None,
        condition_method=None,
        lead_months=[1],
        available_times=[2000],
        effective_input=effective_input,
        effective_condition=None,
        fitted_preprocessors=True,
    )

    dataset = ConcreteDataset(
        config=config,
        requested_years=[2000],
        mask=make_mask(
            times=(2000,),
            lead_times=(1,),
        ),
        load_model=False,
    )

    assert dataset.model_dataset is None
    assert dataset.condition_dataset is None
    assert dataset.model_indexes is None
    assert dataset.cond_indexes is None


def test_abstract_method_bodies():
    dummy = MagicMock()

    assert DatasetConfigABC._check_model(dummy) is None
    assert DatasetConfigABC._check_condition(dummy) is None
    assert DatasetConfigABC.available_times.fget(dummy) is None
    assert DatasetConfigABC.ds_operator.fget(dummy) is None
    assert DatasetConfigABC.num_input_lead_months.fget(dummy) is None
    assert DatasetConfigABC.effective_input.fget(dummy) is None
    assert DatasetConfigABC.build_dataset(dummy) is None
    assert DatasetABC._load_model.fget(dummy) is None
    assert DatasetABC._write_condition_to_input.fget(dummy) is None
    assert DatasetABC._concat_condition_to_input.fget(dummy) is None
