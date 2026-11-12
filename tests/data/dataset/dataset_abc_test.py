from __future__ import annotations

import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import S2D4P.data.dataset.dataset_abc as module
from S2D4P.data.dataset.dataset_abc import (
    AddedTimeFeatures,
    DatasetABC,
    DatasetConfigABC,
    lead_time_config,
)

INIT_TIME_DIM = DatasetConfigABC.init_time_dim
LEAD_TIME_DIM = DatasetConfigABC.lead_time_dim
REALIZATION_DIM = DatasetConfigABC.realization_dim

DATES_2 = ["2000-01-01", "2001-01-01"]


def make_coord(values, dim):
    values = np.asarray(values)
    return xr.DataArray(values, dims=(dim,), coords={dim: values})


def make_pipeline(flattener=None):
    return SimpleNamespace(
        has_flattener=flattener is not None,
        get_flattener=Mock(return_value=flattener),
    )


def make_data_config(
    name="tas",
    *,
    names=None,
    paths=None,
    times=None,
    lead_times=None,
    realizations=None,
    spatial_coords=None,
    ensemble_mean=False,
    realization_list=None,
    time_coords_type="datetime",
    preprocessing_pipeline=None,
):
    """A stand-in for ForecastDataConfig that exposes coords, indexes and info."""
    names = [name] if names is None else list(names)
    if paths is None:
        paths = [f"{name}.nc"]
    elif isinstance(paths, str):
        paths = [paths]
    if times is None:
        times = np.asarray(DATES_2, dtype="datetime64[ns]")
    if lead_times is None:
        lead_times = [1, 2, 3]

    coords = {
        INIT_TIME_DIM: make_coord(times, INIT_TIME_DIM),
        LEAD_TIME_DIM: make_coord(lead_times, LEAD_TIME_DIM),
    }
    if realizations is not None:
        coords[REALIZATION_DIM] = make_coord(realizations, REALIZATION_DIM)
    for dim, values in (spatial_coords or {}).items():
        coords[dim] = make_coord(values, dim)

    shape = tuple(coord.size for coord in coords.values())
    dims = tuple(coords)
    data = xr.Dataset(
        {variable: (dims, np.zeros(shape, dtype=np.float32)) for variable in names},
        coords={dim: coord.values for dim, coord in coords.items()},
    )

    return SimpleNamespace(
        name=name,
        names=names,
        paths=paths,
        coords=coords,
        indexes=data.indexes,
        dims=data.dims,
        sizes=data.sizes,
        info=SimpleNamespace(coords=coords, time_coords_type=time_coords_type),
        data=data,
        ensemble_mean=ensemble_mean,
        realization_list=realization_list,
        preprocessing_pipeline=(
            preprocessing_pipeline
            if preprocessing_pipeline is not None
            else make_pipeline()
        ),
        concat_dim=None,
        file_type="netcdf",
        rename_dict=None,
        open_xarray_data=Mock(),
        isel=data.isel,
    )


class ConcreteDatasetConfig(DatasetConfigABC):
    def __init__(
        self,
        *,
        forecast=None,
        condition=None,
        condition_method=None,
        lead_times=None,
        available_times=None,
        effective_input=None,
        run_parent=False,
    ):
        self.forecast = forecast
        self.condition = condition
        self.condition_method = condition_method
        self.lead_times = lead_times
        self._available_times = pd.DatetimeIndex(
            DATES_2 if available_times is None else available_times
        )
        self._effective_input_override = effective_input
        self._operator = object()

        if run_parent:
            super().__init__()
        else:
            self._fitted_preprocessors = True
            self._effective_condition = condition

    @property
    def available_times(self):
        return self._available_times

    @property
    def ds_operator(self):
        return self._operator

    @property
    def effective_input(self):
        if self._effective_input_override is not None:
            return self._effective_input_override
        return self.forecast if self.forecast is not None else self.condition

    def build_dataset(self):
        return "dataset"


class ConcreteDataset(DatasetABC):
    def __init__(
        self,
        *,
        config=None,
        requested_times=None,
        mask=None,
        time_features=None,
        load_forecast=True,
        concat_condition=False,
        run_parent=False,
    ):
        self.config = config
        if requested_times is None:
            requested_times = np.asarray(["2000-01-01"], dtype="datetime64[ns]")
        self.requested_times = xr.DataArray(
            requested_times,
            dims=(INIT_TIME_DIM,),
            coords={INIT_TIME_DIM: requested_times},
        )
        self.mask = mask
        self.time_features = time_features
        self.return_metadata = False
        self.load = False
        self._load_forecast_value = load_forecast
        self._concat_condition_value = concat_condition

        if run_parent:
            super().__init__()

    @property
    def _load_forecast(self):
        return self._load_forecast_value

    @property
    def _write_condition_to_input(self):
        return False

    @property
    def _concat_condition_to_input(self):
        return self._concat_condition_value

    def __getitem__(self, index):
        return index


class TimeFeatureReference:
    init_time_dim = INIT_TIME_DIM
    lead_time_dim = LEAD_TIME_DIM
    lead_time_resolution = "month"

    def __init__(self, *, lead_times=None, common_times=None):
        self.lead_times = np.asarray([1, 2, 4] if lead_times is None else lead_times)
        self.get_common_time = pd.DatetimeIndex(
            DATES_2 if common_times is None else common_times
        )


def config_with_condition(condition, method, forecast=None):
    config = ConcreteDatasetConfig(
        forecast=make_data_config() if forecast is None else forecast,
        condition=condition,
        condition_method=method,
    )
    config._effective_condition = condition
    return config


def patch_targets(targets):
    """Control add_lead_times so feature arithmetic can be checked directly."""
    return patch.object(
        module,
        "add_lead_times",
        return_value=np.asarray(targets, dtype="datetime64[ns]"),
    )


def coords_for(times, leads):
    return {
        INIT_TIME_DIM: np.asarray(times, dtype="datetime64[ns]"),
        LEAD_TIME_DIM: np.asarray(leads),
    }


def sample_coordinates():
    return coords_for(["2000-01-01"], [1])


def initialize_dataset(dataset, overrides=None):
    """Run DatasetABC.__init__ with the expensive stages replaced by mocks."""
    defaults = {
        "_check_init": Mock(),
        "_resolve_mask": Mock(),
        "_prepare_sampling_mask": Mock(),
        "get_sampling_coords": Mock(return_value=sample_coordinates()),
        "get_forecast_indexes": Mock(return_value=None),
        "get_cond_indexes": Mock(return_value=None),
    }
    defaults.update(overrides or {})
    with (
        patch.multiple(dataset, **defaults),
        patch.object(
            AddedTimeFeatures,
            "build_time_features",
            return_value=dataset.time_features,
        ),
    ):
        DatasetABC.__init__(dataset)


def make_init_dataset(
    *, forecast=None, condition=None, method=None, load_forecast=False
):
    config = ConcreteDatasetConfig(
        forecast=forecast,
        condition=condition,
        condition_method=method,
        lead_times=[1],
        available_times=pd.DatetimeIndex(["2000-01-01"]),
    )
    return ConcreteDataset(
        config=config,
        time_features=AddedTimeFeatures(
            TimeFeatureReference(lead_times=[1]), [LEAD_TIME_DIM]
        ),
        load_forecast=load_forecast,
    )


def make_mask(dims, coords_values, data=None):
    if data is None:
        data = np.zeros(tuple(len(coords_values[dim]) for dim in dims), dtype=bool)
    return xr.DataArray(
        data, dims=dims, coords={dim: coords_values[dim] for dim in dims}
    )


def check_pair(model, condition, method="ensemble_mean", reference=None):
    config = ConcreteDatasetConfig(
        forecast=model,
        condition=condition,
        condition_method=method,
    )
    config._effective_condition = condition
    if reference is not None:
        config.reference = reference
    return config._check_forecast_vs_condition()


@pytest.mark.parametrize(
    "time,expected",
    [
        (cftime.Datetime360Day(2000, 1, 1), 360),
        (cftime.DatetimeNoLeap(2000, 1, 1), 365),
        (cftime.DatetimeAllLeap(2001, 1, 1), 366),
        (np.datetime64("2000-01-01"), 366),
        (np.datetime64("2001-01-01"), 365),
        (np.datetime64("1900-01-01"), 365),
        (np.datetime64("2004-01-01"), 366),
        (datetime.datetime(2000, 1, 1), 366),
        (datetime.datetime(2001, 1, 1), 365),
    ],
)
def test_days_in_year_for_supported_calendars(time, expected):
    assert AddedTimeFeatures._days_in_year(time) == expected


@pytest.mark.pruned
def test_lead_time_requires_list_or_end():
    with pytest.raises(ValueError, match="Provide a list"):
        lead_time_config()


def test_lead_time_rejects_none_list_and_none_end():
    with pytest.raises(ValueError, match="Provide a list"):
        lead_time_config(list_lead_times=None, end=None)


@pytest.mark.pruned
def test_lead_time_explicit_list_is_returned_unchanged():
    configured = (1, 3, 5)

    assert lead_time_config(list_lead_times=configured).build_lead_times() is configured


@pytest.mark.pruned
def test_lead_time_inclusive_range():
    np.testing.assert_array_equal(
        lead_time_config(start=2, end=5).build_lead_times(), [2, 3, 4, 5]
    )


@pytest.mark.pruned
def test_lead_time_default_start_is_one():
    np.testing.assert_array_equal(lead_time_config(end=3).build_lead_times(), [1, 2, 3])


@pytest.mark.pruned
def test_lead_time_list_takes_precedence_over_range():
    assert lead_time_config(
        list_lead_times=[10, 20], start=1, end=3
    ).build_lead_times() == [10, 20]


def test_lead_time_explicit_empty_list_is_preserved():
    assert lead_time_config(list_lead_times=[]).build_lead_times() == []


@pytest.mark.pruned
def test_lead_time_descending_range_is_empty():
    assert lead_time_config(start=5, end=2).build_lead_times().size == 0


@pytest.mark.pruned
def test_lead_time_equal_start_and_end_gives_single_value():
    np.testing.assert_array_equal(
        lead_time_config(start=4, end=4).build_lead_times(), [4]
    )


@pytest.mark.pruned
def test_dataset_config_abc_cannot_be_instantiated():
    with pytest.raises(TypeError):
        DatasetConfigABC()


@pytest.mark.pruned
def test_dataset_config_abstract_members():
    assert DatasetConfigABC.available_times.__isabstractmethod__
    assert DatasetConfigABC.ds_operator.__isabstractmethod__
    assert DatasetConfigABC.effective_input.__isabstractmethod__
    assert DatasetConfigABC.build_dataset.__isabstractmethod__


@pytest.mark.pruned
def test_dataset_config_class_dimensions():
    assert DatasetConfigABC.init_time_dim == INIT_TIME_DIM
    assert DatasetConfigABC.lead_time_dim == LEAD_TIME_DIM
    assert DatasetConfigABC.realization_dim == REALIZATION_DIM


@pytest.mark.pruned
def test_valid_condition_methods_constant():
    assert DatasetConfigABC._VALID_CONDITION_METHODS == frozenset(
        {"ensemble_mean", "cross_ensemble", "same_member", "static"}
    )


@pytest.mark.pruned
def test_dataset_abc_cannot_be_instantiated():
    with pytest.raises(TypeError):
        DatasetABC()


@pytest.mark.pruned
def test_dataset_abc_abstract_members():
    assert DatasetABC._load_forecast.__isabstractmethod__
    assert DatasetABC._write_condition_to_input.__isabstractmethod__
    assert DatasetABC._concat_condition_to_input.__isabstractmethod__


@pytest.mark.pruned
def test_concrete_dataset_is_a_torch_dataset():
    from torch.utils.data import Dataset

    assert isinstance(
        ConcreteDataset(config=ConcreteDatasetConfig(forecast=make_data_config())),
        Dataset,
    )


@pytest.mark.parametrize(
    "method",
    [None, "ensemble_mean", "ENSEMBLE_MEAN", "Cross_Ensemble", "same_member", "STATIC"],
)
def test_valid_condition_methods_are_accepted(method):
    config = ConcreteDatasetConfig(forecast=make_data_config(), condition_method=method)

    assert config._check_condition_method() is config


@pytest.mark.pruned
@pytest.mark.parametrize("method", ["", "mean", "dynamic", "unsupported"])
def test_invalid_condition_methods_are_rejected(method):
    config = ConcreteDatasetConfig(forecast=make_data_config(), condition_method=method)

    with pytest.raises(ValueError, match="Invalid condition_method"):
        config._check_condition_method()


@pytest.mark.pruned
def test_invalid_condition_method_message_lists_valid_values():
    config = ConcreteDatasetConfig(
        forecast=make_data_config(), condition_method="bad_method"
    )

    with pytest.raises(ValueError) as error:
        config._check_condition_method()

    message = str(error.value)
    assert "bad_method" in message
    for valid in ("ensemble_mean", "cross_ensemble", "same_member", "static"):
        assert valid in message


@pytest.mark.pruned
@pytest.mark.parametrize(
    "method,expected",
    [
        ("ensemble_mean", "ensemble_mean"),
        ("ENSEMBLE_MEAN", "ensemble_mean"),
        ("Same_Member", "same_member"),
        (None, None),
    ],
)
def test_effective_condition_method_is_lowercase(method, expected):
    config = ConcreteDatasetConfig(forecast=make_data_config(), condition_method=method)

    assert config.effective_condition_method == expected


@pytest.mark.pruned
def test_no_input_source_is_rejected():
    config = ConcreteDatasetConfig(forecast=None, condition=None)

    with pytest.raises(ValueError) as error:
        config._check_required_input_source()

    message = str(error.value)
    assert "PPP dataset" in message
    assert "forecast" in message
    assert "condition" in message


@pytest.mark.pruned
@pytest.mark.parametrize(
    "forecast,condition",
    [
        (make_data_config(), None),
        (None, make_data_config(name="condition")),
        (make_data_config(), make_data_config(name="condition")),
    ],
)
def test_any_input_source_is_accepted(forecast, condition):
    config = ConcreteDatasetConfig(forecast=forecast, condition=condition)

    assert config._check_required_input_source() is config


@pytest.mark.pruned
@pytest.mark.parametrize(
    "method,expected",
    [
        ("ensemble_mean", True),
        ("ENSEMBLE_MEAN", True),
        ("cross_ensemble", True),
        ("same_member", True),
        ("static", False),
        ("STATIC", False),
        ("unsupported", False),
        (None, False),
    ],
)
def test_forecast_reused_as_condition_without_explicit_condition(method, expected):
    config = ConcreteDatasetConfig(
        forecast=make_data_config(), condition=None, condition_method=method
    )

    assert config._using_forecast_data_as_condition is expected


@pytest.mark.pruned
@pytest.mark.parametrize(
    "condition_paths,condition_names,condition_members,expected",
    [
        (["shared.nc"], ["tas"], [0, 1], True),
        (["other.nc"], ["tas"], [0, 1], False),
        (["shared.nc"], ["pr"], [0, 1], False),
        (["shared.nc"], ["tas"], [0, 2], False),
    ],
)
def test_explicit_condition_matches_forecast_only_when_all_fields_match(
    condition_paths, condition_names, condition_members, expected
):
    model = make_data_config(
        paths=["shared.nc"], names=["tas"], realization_list=[0, 1]
    )
    condition = make_data_config(
        name="condition",
        paths=condition_paths,
        names=condition_names,
        realization_list=condition_members,
    )
    config = ConcreteDatasetConfig(
        forecast=model, condition=condition, condition_method="same_member"
    )

    assert config._using_forecast_data_as_condition is expected


@pytest.mark.pruned
def test_condition_without_forecast_is_never_forecast_reuse():
    config = ConcreteDatasetConfig(
        forecast=None,
        condition=make_data_config(name="condition"),
        condition_method="static",
    )

    assert config._using_forecast_data_as_condition is False


@pytest.mark.pruned
def test_resolve_lead_times_builds_explicit_configuration():
    config = ConcreteDatasetConfig(
        forecast=make_data_config(),
        lead_times=lead_time_config(list_lead_times=[1, 3]),
    )

    config._resolve_lead_times()

    assert config.lead_times == [1, 3]


def test_resolve_lead_times_builds_range_configuration():
    config = ConcreteDatasetConfig(
        forecast=make_data_config(), lead_times=lead_time_config(start=2, end=4)
    )

    config._resolve_lead_times()

    np.testing.assert_array_equal(config.lead_times, [2, 3, 4])


@pytest.mark.pruned
def test_resolve_lead_times_leaves_plain_list_unchanged():
    lead_times = [1, 2]
    config = ConcreteDatasetConfig(forecast=make_data_config(), lead_times=lead_times)

    config._resolve_lead_times()

    assert config.lead_times is lead_times


@pytest.mark.pruned
def test_resolve_lead_times_leaves_array_unchanged():
    lead_times = np.asarray([1, 3])
    config = ConcreteDatasetConfig(forecast=make_data_config(), lead_times=lead_times)

    config._resolve_lead_times()

    assert config.lead_times is lead_times


@pytest.mark.pruned
def test_input_lead_times_come_from_effective_input():
    config = ConcreteDatasetConfig(forecast=make_data_config(lead_times=[1, 2, 4]))

    np.testing.assert_array_equal(config.input_lead_times, [1, 2, 4])


@pytest.mark.pruned
def test_input_lead_times_prefer_effective_input_override():
    config = ConcreteDatasetConfig(
        forecast=make_data_config(lead_times=[1, 2, 3]),
        effective_input=make_data_config(name="override", lead_times=[2, 4, 8]),
    )

    np.testing.assert_array_equal(config.input_lead_times, [2, 4, 8])


@pytest.mark.pruned
def test_explicit_condition_takes_precedence_over_forecast_reuse():
    condition = make_data_config(name="condition")
    config = ConcreteDatasetConfig(
        forecast=make_data_config(),
        condition=condition,
        condition_method="ensemble_mean",
    )
    config._effective_condition = None

    with patch.object(config, "_forecast_as_condition") as builder:
        result = config._resolve_condition()

    assert result is config
    assert config.effective_condition is condition
    builder.assert_not_called()


@pytest.mark.pruned
def test_forecast_is_converted_to_condition_when_reused():
    converted = make_data_config(name="converted", ensemble_mean=True)
    config = ConcreteDatasetConfig(
        forecast=make_data_config(), condition=None, condition_method="ensemble_mean"
    )

    with patch.object(
        config, "_forecast_as_condition", return_value=converted
    ) as builder:
        result = config._resolve_condition()

    assert result is config
    assert config.effective_condition is converted
    builder.assert_called_once_with()


@pytest.mark.pruned
def test_no_conditioning_resolves_to_none():
    config = ConcreteDatasetConfig(
        forecast=make_data_config(), condition=None, condition_method="static"
    )
    config._effective_condition = object()

    result = config._resolve_condition()

    assert result is config
    assert config.effective_condition is None


@pytest.mark.pruned
@pytest.mark.parametrize(
    "method,expected_ensemble_mean",
    [
        ("ensemble_mean", True),
        ("ENSEMBLE_MEAN", True),
        ("cross_ensemble", False),
        ("same_member", False),
    ],
)
def test_forecast_as_condition_forwards_settings(method, expected_ensemble_mean):
    pipeline = object()
    model = make_data_config(
        paths=["first.nc", "second.nc"],
        names=["tas", "pr"],
        realization_list=[0, 1],
        preprocessing_pipeline=pipeline,
    )
    model.concat_dim = "member"
    model.file_type = "netcdf"
    model.rename_dict = {"old": "new"}
    config = ConcreteDatasetConfig(forecast=model, condition_method=method)
    built = object()

    with patch.object(module, "ForecastDataConfig", return_value=built) as constructor:
        result = config._forecast_as_condition()

    assert result is built
    constructor.assert_called_once_with(
        paths=model.paths,
        names=model.names,
        preprocessing_pipeline=pipeline,
        realization_list=[0, 1],
        concat_dim="member",
        file_type="netcdf",
        ensemble_mean=expected_ensemble_mean,
        rename_dict={"old": "new"},
    )


@pytest.mark.pruned
def test_check_forecast_without_forecast_returns_self():
    config = ConcreteDatasetConfig(
        forecast=None,
        condition=make_data_config(name="condition"),
        condition_method="static",
    )

    assert config._check_forecast() is config


@pytest.mark.pruned
def test_same_member_rejects_ensemble_mean_forecast():
    config = ConcreteDatasetConfig(
        forecast=make_data_config(ensemble_mean=True), condition_method="same_member"
    )

    with pytest.raises(ValueError, match="should not be ensemble mean"):
        config._check_forecast()


def test_same_member_accepts_member_forecast():
    config = ConcreteDatasetConfig(
        forecast=make_data_config(ensemble_mean=False, realizations=[0, 1]),
        condition_method="same_member",
    )

    assert config._check_forecast() is config


@pytest.mark.pruned
@pytest.mark.parametrize("method", [None, "ensemble_mean", "cross_ensemble", "static"])
def test_other_methods_skip_forecast_ensemble_check(method):
    config = ConcreteDatasetConfig(
        forecast=make_data_config(ensemble_mean=True), condition_method=method
    )

    assert config._check_forecast() is config


def config_check(condition, method):
    return config_with_condition(condition, method)


def test_condition_requires_method_when_present():
    with pytest.raises(ValueError, match="specify condition_method"):
        config_check(make_data_config(name="condition"), None)._check_condition()


@pytest.mark.parametrize("method", ["cross_ensemble", "same_member", "CROSS_ENSEMBLE"])
def test_member_methods_reject_ensemble_mean_condition(method):
    condition = make_data_config(
        name="condition", ensemble_mean=True, realizations=[0, 1]
    )

    with pytest.raises(ValueError, match="ensemble_mean cannot be True"):
        config_check(condition, method)._check_condition()


@pytest.mark.parametrize("method", ["cross_ensemble", "same_member"])
def test_member_methods_require_realization_coordinate(method):
    condition = make_data_config(
        name="condition", ensemble_mean=False, realizations=None
    )

    with pytest.raises(ValueError, match="dim must exist"):
        config_check(condition, method)._check_condition()


@pytest.mark.pruned
@pytest.mark.parametrize("method", ["cross_ensemble", "same_member"])
def test_member_methods_accept_member_condition(method):
    condition = make_data_config(
        name="condition", ensemble_mean=False, realizations=[0, 1]
    )
    config = config_check(condition, method)

    assert config._check_condition() is config


@pytest.mark.parametrize("ensemble_mean", [False, None, 0])
def test_ensemble_mean_method_requires_literal_true(ensemble_mean):
    condition = make_data_config(name="condition", ensemble_mean=ensemble_mean)

    with pytest.raises(ValueError, match="Ensemble mean must be True"):
        config_check(condition, "ensemble_mean")._check_condition()


@pytest.mark.pruned
def test_ensemble_mean_method_accepts_true():
    condition = make_data_config(name="condition", ensemble_mean=True)
    config = config_check(condition, "ensemble_mean")

    assert config._check_condition() is config


def test_static_rejects_realization_list():
    condition = make_data_config(name="condition", realization_list=[0, 1])
    condition.coords = {"lat": make_coord([45.0], "lat")}

    with pytest.raises(ValueError, match="cannot specify realization list"):
        config_check(condition, "static")._check_condition()


@pytest.mark.parametrize(
    "sampling_dimension", [INIT_TIME_DIM, LEAD_TIME_DIM, REALIZATION_DIM]
)
def test_static_rejects_each_sampling_dimension(sampling_dimension):
    condition = make_data_config(name="condition")
    condition.realization_list = None
    condition.coords = {sampling_dimension: make_coord([0], sampling_dimension)}

    with pytest.raises(ValueError, match="sampling dimensions"):
        config_check(condition, "static")._check_condition()


@pytest.mark.pruned
def test_static_accepts_spatial_coordinates_only():
    condition = make_data_config(name="condition")
    condition.realization_list = None
    condition.coords = {
        "lat": make_coord([45.0], "lat"),
        "lon": make_coord([-123.0], "lon"),
    }
    config = config_check(condition, "static")

    assert config._check_condition() is config


def test_static_rejects_forecast_as_condition():
    model = make_data_config()
    config = config_with_condition(model, "static", forecast=model)

    with patch.object(
        ConcreteDatasetConfig,
        "_using_forecast_data_as_condition",
        new_callable=PropertyMock,
        return_value=True,
    ):
        with pytest.raises(ValueError, match="cannot point to the same forecast data"):
            config._check_condition()


@pytest.mark.pruned
def test_static_method_without_condition_is_rejected():
    config = ConcreteDatasetConfig(
        forecast=make_data_config(), condition=None, condition_method="static"
    )
    config._effective_condition = None

    with pytest.raises(ValueError, match="condition dataset must be specified"):
        config._check_condition()


def test_no_effective_condition_with_dynamic_method_passes():
    config = ConcreteDatasetConfig(
        forecast=make_data_config(), condition=None, condition_method="ensemble_mean"
    )
    config._effective_condition = None

    assert config._check_condition() is config


@pytest.mark.pruned
def test_compatibility_skipped_with_only_one_source():
    config = ConcreteDatasetConfig(forecast=make_data_config(), condition=None)

    assert config._check_forecast_vs_condition() is None


@pytest.mark.pruned
def test_compatibility_skipped_when_condition_is_forecast_reuse():
    with patch.object(
        ConcreteDatasetConfig,
        "_using_forecast_data_as_condition",
        new_callable=PropertyMock,
        return_value=True,
    ):
        result = check_pair(make_data_config(), make_data_config(name="condition"))

    assert result is None


def test_compatibility_skipped_for_static_condition():
    condition = make_data_config(name="condition")
    condition.coords = {}
    condition.realization_list = None

    assert check_pair(make_data_config(), condition, method="static") is None


@pytest.mark.parametrize("missing_dimension", [INIT_TIME_DIM, LEAD_TIME_DIM])
def test_dynamic_condition_requires_forecast_dimensions(missing_dimension):
    condition = make_data_config(name="condition")
    condition.coords.pop(missing_dimension)

    with pytest.raises(ValueError, match="same .* dimestions"):
        check_pair(make_data_config(), condition)


@pytest.mark.pruned
def test_condition_time_superset_is_accepted():
    model = make_data_config(times=np.asarray(DATES_2, dtype="datetime64[ns]"))
    condition = make_data_config(
        name="condition",
        paths=["condition.nc"],
        times=np.asarray(
            ["1999-01-01", "2000-01-01", "2001-01-01", "2002-01-01"],
            dtype="datetime64[ns]",
        ),
        ensemble_mean=True,
    )

    assert check_pair(model, condition) is None


@pytest.mark.pruned
def test_condition_lead_time_superset_is_accepted():
    model = make_data_config(lead_times=[1, 2])
    condition = make_data_config(
        name="condition",
        paths=["condition.nc"],
        lead_times=[1, 2, 3, 4],
        ensemble_mean=True,
    )

    assert check_pair(model, condition) is None


@pytest.mark.pruned
def test_missing_condition_time_value_is_rejected():
    model = make_data_config(times=np.asarray(DATES_2, dtype="datetime64[ns]"))
    condition = make_data_config(
        name="condition",
        paths=["condition.nc"],
        times=np.asarray(["2000-01-01"], dtype="datetime64[ns]"),
        ensemble_mean=True,
    )

    with pytest.raises(ValueError, match=f"same {INIT_TIME_DIM} coordinates"):
        check_pair(model, condition)


def test_missing_condition_lead_time_value_is_rejected():
    model = make_data_config(lead_times=[1, 2, 3])
    condition = make_data_config(
        name="condition", paths=["condition.nc"], lead_times=[1, 2], ensemble_mean=True
    )

    with pytest.raises(ValueError, match=f"same {LEAD_TIME_DIM} coordinates"):
        check_pair(model, condition)


@pytest.mark.parametrize(
    "model_type,condition_type", [("datetime", "cftime"), ("cftime", "datetime")]
)
def test_time_coordinate_type_mismatch_is_rejected(model_type, condition_type):
    model = make_data_config(time_coords_type=model_type)
    condition = make_data_config(
        name="condition",
        paths=["condition.nc"],
        time_coords_type=condition_type,
        ensemble_mean=True,
    )

    with pytest.raises(ValueError, match="same cftime/datetime type"):
        check_pair(model, condition)


@pytest.mark.parametrize(
    "model_members,condition_members", [(None, [0, 1]), ([0, 1], None)]
)
def test_same_member_requires_member_coordinates_on_both_sources(
    model_members, condition_members
):
    model = make_data_config(realizations=model_members, ensemble_mean=False)
    condition = make_data_config(
        name="condition",
        paths=["condition.nc"],
        realizations=condition_members,
        ensemble_mean=False,
    )

    with pytest.raises(ValueError, match="dims and coords"):
        check_pair(model, condition, method="same_member")


def test_same_member_rejects_reordered_members():
    model = make_data_config(realizations=[0, 1], ensemble_mean=False)
    condition = make_data_config(
        name="condition",
        paths=["condition.nc"],
        realizations=[1, 0],
        ensemble_mean=False,
    )

    with pytest.raises(ValueError, match="same ensemble members"):
        check_pair(model, condition, method="same_member")


def test_same_member_accepts_equal_members():
    model = make_data_config(realizations=[0, 1], ensemble_mean=False)
    condition = make_data_config(
        name="condition",
        paths=["condition.nc"],
        realizations=[0, 1],
        ensemble_mean=False,
    )

    assert check_pair(model, condition, method="same_member") is None


def test_reference_spatial_coordinates_must_match():
    model = make_data_config(spatial_coords={"lat": [45.0, 46.0]})
    condition = make_data_config(
        name="condition",
        paths=["condition.nc"],
        spatial_coords={"lat": [40.0, 41.0]},
        ensemble_mean=True,
    )

    with pytest.raises(ValueError, match="do not have the same lat"):
        check_pair(model, condition, reference=object())


def test_reference_requires_matching_spatial_dimensions_in_condition():
    model = make_data_config(spatial_coords={"lat": [45.0, 46.0]})
    condition = make_data_config(
        name="condition", paths=["condition.nc"], ensemble_mean=True
    )

    with pytest.raises(ValueError, match="same NN dims"):
        check_pair(model, condition, reference=object())


def test_get_input_times_rejects_unavailable_times():
    config = ConcreteDatasetConfig(
        forecast=make_data_config(
            times=np.asarray(["2000-01-01"], dtype="datetime64[ns]")
        ),
        available_times=pd.DatetimeIndex(["2000-01-01"]),
    )
    requested = xr.DataArray(
        np.asarray(["1990-01-01"], dtype="datetime64[ns]"), dims=(INIT_TIME_DIM,)
    )

    with pytest.raises(ValueError, match="requested_times are unavailable"):
        config.get_input_times(requested)


@pytest.mark.pruned
def test_get_input_times_intersects_with_effective_input():
    model_times = np.asarray(["2000-01-01", "2000-03-01"], dtype="datetime64[ns]")
    available = pd.DatetimeIndex(["2000-01-01", "2000-02-01", "2000-03-01"])
    config = ConcreteDatasetConfig(
        forecast=make_data_config(times=model_times), available_times=available
    )
    requested = xr.DataArray(
        available.values,
        dims=(INIT_TIME_DIM,),
        coords={INIT_TIME_DIM: available.values},
    )

    np.testing.assert_array_equal(config.get_input_times(requested).values, model_times)


def test_get_input_times_converts_plain_sequence():
    config = ConcreteDatasetConfig(
        forecast=make_data_config(
            times=np.asarray(["2000-01-01", "2000-03-01"], dtype="datetime64[ns]")
        ),
        available_times=pd.DatetimeIndex(["2000-01-01", "2000-02-01", "2000-03-01"]),
    )

    result = config.get_input_times(
        np.asarray(["2000-01-01", "2000-03-01"], dtype="datetime64[ns]")
    )

    assert isinstance(result, xr.DataArray)
    assert result.dims == (INIT_TIME_DIM,)


@pytest.mark.pruned
def test_get_input_times_removes_times_missing_from_input():
    config = ConcreteDatasetConfig(
        forecast=make_data_config(
            times=np.asarray(["2000-01-01"], dtype="datetime64[ns]")
        ),
        available_times=pd.DatetimeIndex(DATES_2),
    )

    result = config.get_input_times(np.asarray(DATES_2, dtype="datetime64[ns]"))

    np.testing.assert_array_equal(
        result.values, np.asarray(["2000-01-01"], dtype="datetime64[ns]")
    )


def test_constructor_rejects_missing_sources():
    with pytest.raises(
        ValueError, match="either forecast or condition data must be provided"
    ):
        ConcreteDatasetConfig(
            forecast=None,
            condition=None,
            condition_method="ensemble_mean",
            run_parent=True,
        )


def test_constructor_rejects_invalid_condition_method():
    with pytest.raises(ValueError, match="Invalid condition_method"):
        ConcreteDatasetConfig(
            forecast=make_data_config(), condition_method="unsupported", run_parent=True
        )


def test_constructor_builds_forecast_condition_when_reused():
    model = make_data_config(realization_list=[0, 1], realizations=[0, 1])
    converted = make_data_config(
        name="converted", ensemble_mean=True, realizations=[0, 1]
    )

    with (
        patch.object(
            ConcreteDatasetConfig,
            "_using_forecast_data_as_condition",
            new_callable=PropertyMock,
            return_value=True,
        ),
        patch.object(
            module, "ForecastDataConfig", return_value=converted
        ) as constructor,
    ):
        config = ConcreteDatasetConfig(
            forecast=model,
            condition_method="ensemble_mean",
            lead_times=[1],
            run_parent=True,
        )

    assert config.effective_condition is converted
    constructor.assert_called_once()


def test_constructor_accepts_static_condition_without_forecast():
    condition = make_data_config(name="condition", lead_times=[1, 2])
    condition.coords = {"lat": make_coord([45.0, 46.0], "lat")}
    condition.realization_list = None

    config = ConcreteDatasetConfig(
        forecast=None,
        condition=condition,
        condition_method="static",
        lead_times=[1],
        effective_input=make_data_config(name="effective", lead_times=[1, 2]),
        run_parent=True,
    )

    assert config.effective_condition is condition


@pytest.mark.pruned
def test_build_dataset_and_operator_accessors():
    config = ConcreteDatasetConfig(forecast=make_data_config())

    assert config.build_dataset() == "dataset"
    assert config.ds_operator is config._operator


@pytest.mark.pruned
def test_available_times_accessor():
    expected = pd.DatetimeIndex(DATES_2)
    config = ConcreteDatasetConfig(
        forecast=make_data_config(), available_times=expected
    )

    assert config.available_times.equals(expected)


@pytest.mark.pruned
def test_effective_input_prefers_override_then_forecast_then_condition():
    model = make_data_config(name="model")
    override = make_data_config(name="override")
    condition = make_data_config(name="condition")

    assert (
        ConcreteDatasetConfig(forecast=model, effective_input=override).effective_input
        is override
    )
    assert (
        ConcreteDatasetConfig(forecast=model, condition=condition).effective_input
        is model
    )
    assert (
        ConcreteDatasetConfig(forecast=None, condition=condition).effective_input
        is condition
    )


@pytest.mark.pruned
def test_effective_condition_returns_internal_value():
    config = ConcreteDatasetConfig(forecast=make_data_config())
    marker = object()
    config._effective_condition = marker

    assert config.effective_condition is marker


@pytest.mark.pruned
def test_initial_state_of_added_time_features():
    reference = TimeFeatureReference()
    features = AddedTimeFeatures(reference, None)

    assert features.reference_config is reference
    assert features.time_features == ()
    assert features.time_features_array is None
    assert len(features) == 0
    assert features.lead_time_resolution == reference.lead_time_resolution
    assert features.init_time_dim == INIT_TIME_DIM
    assert features.lead_time_dim == LEAD_TIME_DIM


@pytest.mark.pruned
def test_reference_min_max_and_span():
    features = AddedTimeFeatures(TimeFeatureReference(), None)

    assert features.min_time_ref == pd.Timestamp("2000-01-01")
    assert features.max_time_ref == pd.Timestamp("2001-01-01")
    assert features.time_span_ref == pd.Timedelta(days=366)


@pytest.mark.pruned
def test_feature_indices_are_stable():
    features = AddedTimeFeatures(TimeFeatureReference(), None)

    assert features.feature_indices == {
        INIT_TIME_DIM: 0,
        LEAD_TIME_DIM: 1,
        "month_sin": 2,
        "month_cos": 3,
        "day_sin": 4,
        "day_cos": 5,
    }


@pytest.mark.pruned
@pytest.mark.parametrize(
    "selected",
    [
        [INIT_TIME_DIM],
        [LEAD_TIME_DIM],
        ["month_sin"],
        ["month_cos"],
        ["day_sin"],
        ["day_cos"],
    ],
)
def test_single_feature_selection(selected):
    features = AddedTimeFeatures(TimeFeatureReference(), selected)

    assert features.time_features == tuple(selected)


@pytest.mark.pruned
def test_features_are_reordered_to_canonical_order_and_deduplicated():
    features = AddedTimeFeatures(
        TimeFeatureReference(),
        [
            "day_cos",
            "month_cos",
            LEAD_TIME_DIM,
            "month_sin",
            INIT_TIME_DIM,
            "day_sin",
            LEAD_TIME_DIM,
        ],
    )

    assert features.time_features == (
        INIT_TIME_DIM,
        LEAD_TIME_DIM,
        "month_sin",
        "month_cos",
        "day_sin",
        "day_cos",
    )


@pytest.mark.pruned
def test_none_and_empty_feature_requests_are_equivalent():
    assert AddedTimeFeatures(TimeFeatureReference(), None).time_features == ()
    assert AddedTimeFeatures(TimeFeatureReference(), []).time_features == ()


def test_unsupported_features_are_reported_by_name():
    with pytest.raises(ValueError) as error:
        AddedTimeFeatures(TimeFeatureReference(), ["hour_sin", "hour_cos"])

    message = str(error.value)
    assert "Unsupported time features" in message
    assert "hour_sin" in message
    assert "hour_cos" in message
    assert "month_sin" in message


@pytest.mark.pruned
def test_equal_configurations_compare_equal():
    first = AddedTimeFeatures(
        TimeFeatureReference(lead_times=[1, 2], common_times=DATES_2), [LEAD_TIME_DIM]
    )
    second = AddedTimeFeatures(
        TimeFeatureReference(lead_times=[1, 2], common_times=DATES_2), [LEAD_TIME_DIM]
    )

    assert first == second


@pytest.mark.parametrize(
    "first_kwargs,second_kwargs,first_features,second_features",
    [
        ({}, {}, [LEAD_TIME_DIM], ["month_cos"]),
        (
            {"lead_times": [1, 2]},
            {"lead_times": [1, 3]},
            [LEAD_TIME_DIM],
            [LEAD_TIME_DIM],
        ),
        (
            {"common_times": DATES_2},
            {"common_times": ["2000-01-01", "2002-01-01"]},
            [LEAD_TIME_DIM],
            [LEAD_TIME_DIM],
        ),
    ],
)
def test_different_configurations_compare_unequal(
    first_kwargs, second_kwargs, first_features, second_features
):
    first = AddedTimeFeatures(TimeFeatureReference(**first_kwargs), first_features)
    second = AddedTimeFeatures(TimeFeatureReference(**second_kwargs), second_features)

    assert first != second


@pytest.mark.pruned
def test_different_reference_types_compare_unequal():
    class OtherReference(TimeFeatureReference):
        pass

    first = AddedTimeFeatures(TimeFeatureReference(), [LEAD_TIME_DIM])
    second = AddedTimeFeatures(OtherReference(), [LEAD_TIME_DIM])

    assert first != second


def test_comparison_with_unrelated_object_is_not_implemented():
    features = AddedTimeFeatures(TimeFeatureReference(), ["month_sin"])

    assert features.__eq__(object()) is NotImplemented


@pytest.mark.pruned
def test_build_requires_both_dimensions_and_names_them():
    features = AddedTimeFeatures(TimeFeatureReference(), [LEAD_TIME_DIM])

    with pytest.raises(ValueError, match="missing required") as error:
        features.build_time_features({})

    assert INIT_TIME_DIM in str(error.value)
    assert LEAD_TIME_DIM in str(error.value)


@pytest.mark.parametrize("missing", [INIT_TIME_DIM, LEAD_TIME_DIM])
def test_build_rejects_each_missing_dimension(missing):
    coords = coords_for(DATES_2, [1, 2])
    coords.pop(missing)
    features = AddedTimeFeatures(TimeFeatureReference(), [LEAD_TIME_DIM])

    with pytest.raises(ValueError, match="missing required"):
        features.build_time_features(coords)


@pytest.mark.pruned
def test_build_without_features_returns_self_and_skips_lead_times():
    features = AddedTimeFeatures(TimeFeatureReference(), None)

    with patch.object(module, "add_lead_times") as add_times:
        result = features.build_time_features(coords_for(["2000-01-01"], [1]))

    assert result is features
    assert features.time_features_array is None
    add_times.assert_not_called()


def test_add_lead_times_receives_exact_arguments():
    coords = coords_for(DATES_2, [1, 2])
    features = AddedTimeFeatures(TimeFeatureReference(), ["month_cos"])

    with patch_targets(DATES_2) as add_times:
        features.build_time_features(coords)

    assert add_times.call_args.kwargs["init_times"] is coords[INIT_TIME_DIM]
    assert add_times.call_args.kwargs["lead_times"] is coords[LEAD_TIME_DIM]
    assert (
        add_times.call_args.kwargs["lead_time_resolution"]
        == features.lead_time_resolution
    )


def test_lead_time_feature_is_normalised_by_reference_maximum():
    features = AddedTimeFeatures(
        TimeFeatureReference(lead_times=[1, 5, 10]), [LEAD_TIME_DIM]
    )

    with patch_targets(["2000-01-01"] * 3):
        features.build_time_features(coords_for(["2000-01-01"] * 3, [1, 5, 10]))

    np.testing.assert_allclose(features.time_features_array[:, 0], [0.1, 0.5, 1.0])


def test_lead_time_maximum_zero_is_rejected():
    features = AddedTimeFeatures(TimeFeatureReference(lead_times=[0]), [LEAD_TIME_DIM])

    with patch_targets(["2000-01-01"]):
        with pytest.raises(ValueError, match="maximum is zero"):
            features.build_time_features(coords_for(["2000-01-01"], [0]))


@pytest.mark.pruned
def test_init_time_feature_is_normalised_against_reference_span():
    targets = ["2000-01-01", "2000-07-01"]
    features = AddedTimeFeatures(TimeFeatureReference(), [INIT_TIME_DIM])

    with patch_targets(targets):
        features.build_time_features(coords_for(targets, [1, 1]))

    np.testing.assert_allclose(
        features.time_features_array[:, 0], [0.0, 182 / 366], atol=1e-6
    )


def test_init_time_zero_span_is_rejected():
    features = AddedTimeFeatures(
        TimeFeatureReference(common_times=["2000-01-01", "2000-01-01"]),
        [INIT_TIME_DIM],
    )

    with patch_targets(["2000-01-01"]):
        with pytest.raises(ValueError, match="zero-length reference span"):
            features.build_time_features(coords_for(["2000-01-01"], [1]))


@pytest.mark.pruned
@pytest.mark.parametrize(
    "target,expected_sin,expected_cos",
    [
        ("2000-01-01", 0.0, 1.0),
        ("2000-04-01", 1.0, 0.0),
        ("2000-07-01", 0.0, -1.0),
        ("2000-10-01", -1.0, 0.0),
    ],
)
def test_month_features_follow_the_calendar_cycle(target, expected_sin, expected_cos):
    features = AddedTimeFeatures(TimeFeatureReference(), ["month_sin", "month_cos"])

    with patch_targets([target]):
        features.build_time_features(coords_for([target], [1]))

    np.testing.assert_allclose(
        features.time_features_array[0], [expected_sin, expected_cos], atol=1e-6
    )


@pytest.mark.pruned
def test_day_features_use_days_in_year():
    features = AddedTimeFeatures(TimeFeatureReference(), ["day_sin", "day_cos"])
    target = "2000-07-01"

    with patch_targets([target]):
        features.build_time_features(coords_for([target], [1]))

    angle = 2 * np.pi * 182 / 366
    np.testing.assert_allclose(
        features.time_features_array[0], [np.sin(angle), np.cos(angle)], atol=1e-6
    )


def test_all_features_have_expected_shape_and_dtype():
    features = AddedTimeFeatures(
        TimeFeatureReference(),
        [INIT_TIME_DIM, LEAD_TIME_DIM, "month_sin", "month_cos", "day_sin", "day_cos"],
    )
    targets = ["2000-01-01", "2000-02-01", "2000-03-01"]

    with patch_targets(targets):
        result = features.build_time_features(coords_for(targets, [1, 2, 4]))

    assert result is features
    assert features.time_features_array.shape == (3, 6)
    assert features.time_features_array.dtype == np.float32


@pytest.mark.pruned
def test_second_build_replaces_previous_array():
    features = AddedTimeFeatures(
        TimeFeatureReference(lead_times=[1, 2]), [LEAD_TIME_DIM]
    )

    with patch_targets(["2000-01-01"]):
        features.build_time_features(coords_for(["2000-01-01"], [1]))
    first_shape = features.time_features_array.shape

    with patch_targets(["2000-01-01", "2000-01-01"]):
        features.build_time_features(coords_for(["2000-01-01", "2000-01-01"], [1, 2]))

    assert first_shape == (1, 1)
    assert features.time_features_array.shape == (2, 1)


def make_called_features(values, features=(LEAD_TIME_DIM, "month_cos")):
    result = AddedTimeFeatures(TimeFeatureReference(), list(features))
    result.time_features_array = np.asarray(values, dtype=np.float32)
    return result


def test_call_without_features_returns_none():
    features = AddedTimeFeatures(TimeFeatureReference(), None)

    assert features(0, xr.DataArray(np.zeros((2, 3)))) is None


def test_call_before_building_raises():
    features = AddedTimeFeatures(TimeFeatureReference(), ["month_sin"])

    with pytest.raises(RuntimeError, match="must be built"):
        features(0, xr.DataArray(np.zeros((2, 3))))


@pytest.mark.parametrize("index", [-10, -1, 3, 10])
def test_call_rejects_out_of_bounds_index(index):
    features = make_called_features([[0.25, 1.0], [0.5, 0.0], [1.0, -1.0]])

    with pytest.raises(IndexError, match="out of bounds"):
        features(index, xr.DataArray([1.0]))


@pytest.mark.pruned
@pytest.mark.parametrize(
    "index,expected", [(0, [0.25, 1.0]), (1, [0.5, 0.0]), (2, [1.0, -1.0])]
)
def test_call_returns_feature_vector_for_each_sample(index, expected):
    features = make_called_features([[0.25, 1.0], [0.5, 0.0], [1.0, -1.0]])

    np.testing.assert_allclose(features(index, xr.DataArray([1.0])), expected)


@pytest.mark.parametrize(
    "shape,expected_shape",
    [
        ((2,), (2,)),
        ((2, 3), (2,)),
        ((2, 3, 4), (2, 3, 4)),
        ((2, 3, 4, 5), (2, 3, 4, 5)),
    ],
)
def test_call_broadcasts_features_over_spatial_dimensions(shape, expected_shape):
    features = make_called_features([[0.25, 1.0]])

    result = features(0, xr.DataArray(np.ones(shape)))

    assert result.shape == expected_shape


@pytest.mark.pruned
def test_call_broadcasts_feature_values_across_space():
    features = make_called_features([[0.25, 1.0]])
    data = xr.DataArray(np.ones((2, 3, 4)), dims=("channels", "lat", "lon"))

    result = features(0, data)

    np.testing.assert_allclose(result[0], 0.25)
    np.testing.assert_allclose(result[1], 1.0)


def test_broadcast_result_is_a_writable_copy():
    features = make_called_features([[0.25, 1.0]])

    result = features(0, xr.DataArray(np.ones((2, 2, 2))))
    result[0, 0, 0] = 99.0

    assert features.time_features_array[0, 0] == pytest.approx(0.25)


@pytest.mark.pruned
def test_len_matches_selected_features():
    assert len(make_called_features([[0.0, 0.0]])) == 2


def make_check_dataset(*, requested=None, available=None, fitted=True):
    config = ConcreteDatasetConfig(
        forecast=make_data_config(),
        available_times=pd.DatetimeIndex(
            ["2000-01-01"] if available is None else available
        ),
    )
    config._fitted_preprocessors = fitted
    return ConcreteDataset(
        config=config,
        requested_times=np.asarray(
            ["2000-01-01"] if requested is None else requested, dtype="datetime64[ns]"
        ),
    )


@pytest.mark.pruned
def test_check_init_validates_time_sequence_with_requested_times():
    dataset = make_check_dataset()

    with patch.object(module, "_validate_time_sequence") as validator:
        dataset._check_init()

    assert validator.call_args.args[0] is dataset.requested_times


@pytest.mark.pruned
def test_check_init_propagates_time_sequence_errors():
    dataset = make_check_dataset()

    with patch.object(
        module,
        "_validate_time_sequence",
        side_effect=TypeError("invalid time sequence"),
    ):
        with pytest.raises(TypeError, match="invalid time sequence"):
            dataset._check_init()


def test_check_init_rejects_unfitted_preprocessors():
    dataset = make_check_dataset(fitted=False)

    with pytest.raises(RuntimeError, match="fit preprocessors"):
        dataset._check_init()


def test_check_init_reports_all_missing_times():
    dataset = make_check_dataset(requested=["1990-01-01", "1991-01-01"])

    with pytest.raises(ValueError) as error:
        dataset._check_init()

    message = str(error.value)
    assert "initialization times" in message
    assert "1990" in message
    assert "1991" in message


def test_check_init_accepts_all_available_times():
    dataset = make_check_dataset(requested=DATES_2, available=DATES_2)

    assert dataset._check_init() is None


def make_mask_dataset(mask=None):
    config = ConcreteDatasetConfig(
        forecast=make_data_config(lead_times=[1, 2]),
        available_times=pd.DatetimeIndex(DATES_2),
    )
    return ConcreteDataset(config=config, mask=mask)


MASK_TEMPLATE_COORDS = {
    INIT_TIME_DIM: np.asarray(DATES_2, dtype="datetime64[ns]"),
    LEAD_TIME_DIM: [1, 2],
}


def mask_template():
    return make_mask(
        (INIT_TIME_DIM, LEAD_TIME_DIM),
        MASK_TEMPLATE_COORDS,
        np.ones((2, 2), dtype=bool),
    )


def test_missing_mask_is_created_from_train_mask_and_is_all_false():
    dataset = make_mask_dataset()

    with patch.object(
        module, "_create_train_mask", return_value=mask_template()
    ) as creator:
        dataset._resolve_mask()

    assert creator.call_args.kwargs["init_times"] is dataset.config.available_times
    assert creator.call_args.kwargs["lead_times"] is dataset.config.input_lead_times
    assert (
        creator.call_args.kwargs["lead_time_resolution"] == module.lead_time_resolution
    )
    assert dataset.mask.dtype == bool
    assert not bool(dataset.mask.any())


@pytest.mark.pruned
def test_existing_mask_is_preserved():
    existing = mask_template()
    dataset = make_mask_dataset(mask=existing)

    with patch.object(module, "_create_train_mask") as creator:
        dataset._resolve_mask()

    creator.assert_not_called()
    assert dataset.mask is existing


@pytest.mark.parametrize(
    "dims", [(), (INIT_TIME_DIM,), (LEAD_TIME_DIM,), ("lat",), ("lat", "lon")]
)
def test_mask_missing_time_dimensions_is_rejected(dims):
    mask = xr.DataArray(np.zeros(tuple(1 for _ in dims), dtype=bool), dims=dims)
    dataset = make_mask_dataset(mask=mask)

    with pytest.raises(ValueError, match="mask must have"):
        dataset._resolve_mask()


@pytest.mark.pruned
def test_mask_with_extra_dimensions_is_allowed():
    mask = xr.DataArray(
        np.zeros((1, 1, 2), dtype=bool),
        dims=(INIT_TIME_DIM, LEAD_TIME_DIM, "extra"),
        coords={
            INIT_TIME_DIM: [np.datetime64("2000-01-01")],
            LEAD_TIME_DIM: [1],
            "extra": [0, 1],
        },
    )
    dataset = make_mask_dataset(mask=mask)

    assert dataset._resolve_mask() is None


@pytest.mark.pruned
def test_sampling_selectors_use_input_times_and_configured_leads():
    config = ConcreteDatasetConfig(forecast=make_data_config(), lead_times=[1, 2])
    selected = xr.DataArray(
        np.asarray(["2000-01-01"], dtype="datetime64[ns]"), dims=(INIT_TIME_DIM,)
    )
    config.get_input_times = Mock(return_value=selected)
    dataset = ConcreteDataset(config=config)

    selectors = dataset._sampling_times_selectors

    assert config.get_input_times.call_args.args[0] is dataset.requested_times
    assert selectors[INIT_TIME_DIM] is selected
    assert selectors[LEAD_TIME_DIM] is config.lead_times


@pytest.mark.pruned
def test_sampling_selectors_pass_through_input_times_placeholder():
    config = ConcreteDatasetConfig(
        forecast=make_data_config(), lead_times=np.asarray([1, 3])
    )
    config.get_input_times = Mock(return_value="INPUT_TIMES")
    dataset = ConcreteDataset(config=config)

    selectors = dataset._sampling_times_selectors

    assert selectors[INIT_TIME_DIM] == "INPUT_TIMES"
    np.testing.assert_array_equal(selectors[LEAD_TIME_DIM], [1, 3])


def make_sampling_dataset(*, ensemble_mean=True, realizations=None):
    model = make_data_config(
        ensemble_mean=ensemble_mean, realizations=realizations, lead_times=[1, 2]
    )
    config = ConcreteDatasetConfig(forecast=model, lead_times=[1, 2])
    mask = make_mask(
        (INIT_TIME_DIM, LEAD_TIME_DIM),
        MASK_TEMPLATE_COORDS,
        np.asarray([[False, True], [False, False]]),
    )
    return ConcreteDataset(config=config, mask=mask)


def full_selectors(dataset):
    return {
        INIT_TIME_DIM: dataset.mask[INIT_TIME_DIM],
        LEAD_TIME_DIM: dataset.mask[LEAD_TIME_DIM],
    }


@pytest.mark.parametrize(
    "selectors",
    [{}, {INIT_TIME_DIM: [np.datetime64("2000-01-01")]}, {LEAD_TIME_DIM: [1]}],
)
def test_prepare_sampling_mask_requires_both_selectors(selectors):
    dataset = make_sampling_dataset()

    with pytest.raises(ValueError, match="No selectors provided"):
        dataset._prepare_sampling_mask(selectors)


@pytest.mark.pruned
def test_prepare_sampling_mask_returns_self():
    dataset = make_sampling_dataset()

    assert dataset._prepare_sampling_mask(full_selectors(dataset)) is dataset


@pytest.mark.pruned
def test_prepare_sampling_mask_selects_time_subset():
    dataset = make_sampling_dataset()

    dataset._prepare_sampling_mask(
        {INIT_TIME_DIM: [np.datetime64("2001-01-01")], LEAD_TIME_DIM: [1, 2]}
    )

    assert dataset.mask.sizes[INIT_TIME_DIM] == 1
    assert dataset.mask[INIT_TIME_DIM].values[0] == np.datetime64("2001-01-01", "ns")


@pytest.mark.pruned
def test_prepare_sampling_mask_selects_lead_subset():
    dataset = make_sampling_dataset()

    dataset._prepare_sampling_mask(
        {INIT_TIME_DIM: dataset.mask[INIT_TIME_DIM], LEAD_TIME_DIM: [2]}
    )

    assert dataset.mask.sizes[LEAD_TIME_DIM] == 1
    assert dataset.mask[LEAD_TIME_DIM].item() == 2


@pytest.mark.pruned
def test_prepare_sampling_mask_turns_masked_samples_into_nan():
    dataset = make_sampling_dataset()

    dataset._prepare_sampling_mask(full_selectors(dataset))

    assert np.isnan(
        dataset.mask.sel(
            {INIT_TIME_DIM: np.datetime64("2000-01-01"), LEAD_TIME_DIM: 2}
        ).item()
    )


def test_ensemble_mean_does_not_add_realization_dimension():
    dataset = make_sampling_dataset(ensemble_mean=True, realizations=[0, 1])

    dataset._prepare_sampling_mask(full_selectors(dataset))

    assert REALIZATION_DIM not in dataset.mask.dims


@pytest.mark.pruned
def test_member_forecast_without_realization_coordinate_is_not_expanded():
    dataset = make_sampling_dataset(ensemble_mean=False, realizations=None)

    dataset._prepare_sampling_mask(full_selectors(dataset))

    assert REALIZATION_DIM not in dataset.mask.dims


def test_member_forecast_expands_realization_as_first_dimension():
    dataset = make_sampling_dataset(ensemble_mean=False, realizations=[0, 1, 2])

    dataset._prepare_sampling_mask(full_selectors(dataset))

    assert dataset.mask.dims[0] == REALIZATION_DIM
    assert dataset.mask.sizes[REALIZATION_DIM] == 3
    np.testing.assert_array_equal(dataset.mask[REALIZATION_DIM], [0, 1, 2])


def make_coord_dataset(mask):
    return ConcreteDataset(
        config=ConcreteDatasetConfig(forecast=make_data_config()), mask=mask
    )


@pytest.mark.pruned
def test_all_valid_locations_are_returned():
    mask = make_mask(
        (INIT_TIME_DIM, LEAD_TIME_DIM),
        MASK_TEMPLATE_COORDS,
        np.zeros((2, 2), dtype=float),
    )

    result = make_coord_dataset(mask).get_sampling_coords()

    assert len(result[INIT_TIME_DIM]) == 4
    assert len(result[LEAD_TIME_DIM]) == 4


@pytest.mark.pruned
def test_nan_locations_are_removed():
    mask = make_mask(
        (INIT_TIME_DIM, LEAD_TIME_DIM),
        MASK_TEMPLATE_COORDS,
        np.asarray([[0.0, np.nan], [0.0, 0.0]]),
    )

    result = make_coord_dataset(mask).get_sampling_coords()

    assert len(result[INIT_TIME_DIM]) == 3
    assert len(result[LEAD_TIME_DIM]) == 3


@pytest.mark.pruned
def test_all_nan_mask_gives_empty_coordinates():
    mask = make_mask(
        (INIT_TIME_DIM, LEAD_TIME_DIM), MASK_TEMPLATE_COORDS, np.full((2, 2), np.nan)
    )

    result = make_coord_dataset(mask).get_sampling_coords()

    assert result[INIT_TIME_DIM].size == 0
    assert result[LEAD_TIME_DIM].size == 0


@pytest.mark.pruned
def test_sampling_coordinates_keep_realization_dimension_order():
    mask = make_mask(
        (REALIZATION_DIM, INIT_TIME_DIM, LEAD_TIME_DIM),
        {
            REALIZATION_DIM: np.asarray([0, 1]),
            INIT_TIME_DIM: np.asarray(["2000-01-01"], dtype="datetime64[ns]"),
            LEAD_TIME_DIM: np.asarray([1]),
        },
    )

    result = make_coord_dataset(mask).get_sampling_coords()

    assert tuple(result) == (REALIZATION_DIM, INIT_TIME_DIM, LEAD_TIME_DIM)
    np.testing.assert_array_equal(result[REALIZATION_DIM], [0, 1])


def make_forecast_dataset(*, load_forecast=True, realizations=None):
    model = make_data_config(
        times=np.asarray(DATES_2, dtype="datetime64[ns]"),
        lead_times=[1, 2],
        realizations=realizations,
    )
    return ConcreteDataset(
        config=ConcreteDatasetConfig(forecast=model), load_forecast=load_forecast
    )


def test_forecast_indexes_are_none_when_forecast_not_loaded():
    assert make_forecast_dataset(load_forecast=False).get_forecast_indexes({}) is None


@pytest.mark.pruned
def test_forecast_indexes_for_empty_mapping_are_empty():
    assert make_forecast_dataset().get_forecast_indexes({}) == {}


@pytest.mark.pruned
def test_forecast_time_indexes_follow_requested_order():
    result = make_forecast_dataset().get_forecast_indexes(
        {
            INIT_TIME_DIM: np.asarray(
                ["2001-01-01", "2000-01-01"], dtype="datetime64[ns]"
            )
        }
    )

    np.testing.assert_array_equal(result[INIT_TIME_DIM], [1, 0])


@pytest.mark.pruned
def test_forecast_lead_indexes():
    result = make_forecast_dataset().get_forecast_indexes(
        {LEAD_TIME_DIM: np.asarray([2, 1])}
    )

    np.testing.assert_array_equal(result[LEAD_TIME_DIM], [1, 0])


@pytest.mark.pruned
def test_forecast_realization_indexes():
    result = make_forecast_dataset(realizations=[0, 1, 2]).get_forecast_indexes(
        {REALIZATION_DIM: np.asarray([2, 0])}
    )

    np.testing.assert_array_equal(result[REALIZATION_DIM], [2, 0])


def test_forecast_indexes_cover_all_dimensions():
    result = make_forecast_dataset(realizations=[0, 1]).get_forecast_indexes(
        {
            INIT_TIME_DIM: np.asarray(["2001-01-01"], dtype="datetime64[ns]"),
            LEAD_TIME_DIM: np.asarray([2]),
            REALIZATION_DIM: np.asarray([1]),
        }
    )

    assert set(result) == {INIT_TIME_DIM, LEAD_TIME_DIM, REALIZATION_DIM}
    assert all(value.item() == 1 for value in result.values())


@pytest.mark.parametrize(
    "dimension,bad_value",
    [
        (INIT_TIME_DIM, np.datetime64("1990-01-01")),
        (LEAD_TIME_DIM, 99),
        (REALIZATION_DIM, 9),
    ],
)
def test_missing_forecast_coordinate_is_reported(dimension, bad_value):
    dataset = make_forecast_dataset(realizations=[0, 1])

    with pytest.raises(ValueError, match="not found in the forecast dataset"):
        dataset.get_forecast_indexes({dimension: np.asarray([bad_value])})


@pytest.mark.pruned
def test_multiple_missing_forecast_dimensions_are_all_reported():
    dataset = make_forecast_dataset(realizations=[0, 1])

    with pytest.raises(ValueError) as error:
        dataset.get_forecast_indexes(
            {
                INIT_TIME_DIM: np.asarray(["1990-01-01"], dtype="datetime64[ns]"),
                LEAD_TIME_DIM: np.asarray([99]),
                REALIZATION_DIM: np.asarray([9]),
            }
        )

    message = str(error.value)
    assert INIT_TIME_DIM in message
    assert LEAD_TIME_DIM in message
    assert REALIZATION_DIM in message


def make_condition_dataset(method, condition):
    config = ConcreteDatasetConfig(
        forecast=make_data_config(), condition=condition, condition_method=method
    )
    config._effective_condition = condition
    return ConcreteDataset(config=config)


@pytest.mark.pruned
def test_condition_indexes_are_none_without_condition():
    dataset = make_condition_dataset("ensemble_mean", None)

    assert dataset.get_cond_indexes({}) is None


@pytest.mark.parametrize("method", ["static", "STATIC"])
def test_condition_indexes_are_none_for_static_method(method):
    dataset = make_condition_dataset(method, make_data_config(name="condition"))

    assert dataset.get_cond_indexes({}) is None


@pytest.mark.pruned
def test_condition_indexes_for_empty_mapping_are_empty():
    dataset = make_condition_dataset(
        "ensemble_mean", make_data_config(name="condition", ensemble_mean=True)
    )

    assert dataset.get_cond_indexes({}) == {}


@pytest.mark.pruned
def test_unknown_dimensions_are_ignored_for_condition_indexes():
    dataset = make_condition_dataset(
        "ensemble_mean", make_data_config(name="condition", ensemble_mean=True)
    )

    result = dataset.get_cond_indexes({"unknown": np.asarray([1])})

    assert "unknown" not in result


@pytest.mark.pruned
def test_time_and_lead_condition_indexes():
    dataset = make_condition_dataset(
        "ensemble_mean", make_data_config(name="condition", ensemble_mean=True)
    )

    result = dataset.get_cond_indexes(
        {
            INIT_TIME_DIM: np.asarray(["2001-01-01"], dtype="datetime64[ns]"),
            LEAD_TIME_DIM: np.asarray([2]),
        }
    )

    np.testing.assert_array_equal(result[INIT_TIME_DIM], [1])
    np.testing.assert_array_equal(result[LEAD_TIME_DIM], [1])


@pytest.mark.pruned
def test_cross_ensemble_condition_indexes_exclude_realization():
    condition = make_data_config(
        name="condition", ensemble_mean=False, realizations=[0, 1]
    )
    dataset = make_condition_dataset("cross_ensemble", condition)

    result = dataset.get_cond_indexes(
        {
            INIT_TIME_DIM: np.asarray(["2000-01-01"], dtype="datetime64[ns]"),
            LEAD_TIME_DIM: np.asarray([1]),
            REALIZATION_DIM: np.asarray([1]),
        }
    )

    assert REALIZATION_DIM not in result


@pytest.mark.pruned
def test_same_member_condition_indexes_require_realization():
    condition = make_data_config(
        name="condition", ensemble_mean=False, realizations=[0, 1]
    )
    dataset = make_condition_dataset("same_member", condition)

    with pytest.raises(ValueError, match="requires.*coordinates"):
        dataset.get_cond_indexes(
            {
                INIT_TIME_DIM: np.asarray(["2000-01-01"], dtype="datetime64[ns]"),
                LEAD_TIME_DIM: np.asarray([1]),
            }
        )


@pytest.mark.parametrize("member,expected_index", [(0, 0), (1, 1)])
def test_same_member_realization_maps_to_position(member, expected_index):
    condition = make_data_config(
        name="condition", ensemble_mean=False, realizations=[0, 1]
    )
    dataset = make_condition_dataset("same_member", condition)

    result = dataset.get_cond_indexes({REALIZATION_DIM: np.asarray([member])})

    np.testing.assert_array_equal(result[REALIZATION_DIM], [expected_index])


@pytest.mark.parametrize(
    "method,sample_coords",
    [
        (
            "cross_ensemble",
            {INIT_TIME_DIM: np.asarray(["1990-01-01"], dtype="datetime64[ns]")},
        ),
        ("cross_ensemble", {LEAD_TIME_DIM: np.asarray([99])}),
        ("same_member", {REALIZATION_DIM: np.asarray([9])}),
    ],
)
def test_missing_condition_coordinates_are_reported(method, sample_coords):
    condition = make_data_config(
        name="condition", ensemble_mean=False, realizations=[0, 1]
    )
    dataset = make_condition_dataset(method, condition)

    with pytest.raises(ValueError, match="conditioning coordinates were not found"):
        dataset.get_cond_indexes(sample_coords)


@pytest.mark.pruned
def test_multiple_missing_condition_coordinates_are_all_reported():
    condition = make_data_config(
        name="condition", ensemble_mean=False, realizations=[0, 1]
    )
    dataset = make_condition_dataset("same_member", condition)

    with pytest.raises(ValueError) as error:
        dataset.get_cond_indexes(
            {
                INIT_TIME_DIM: np.asarray(["1990-01-01"], dtype="datetime64[ns]"),
                LEAD_TIME_DIM: np.asarray([99]),
                REALIZATION_DIM: np.asarray([9]),
            }
        )

    message = str(error.value)
    assert INIT_TIME_DIM in message
    assert LEAD_TIME_DIM in message
    assert REALIZATION_DIM in message


def model_dataset_with_values(values):
    data = xr.Dataset(
        {"tas": (INIT_TIME_DIM, values)},
        coords={INIT_TIME_DIM: np.asarray(DATES_2, dtype="datetime64[ns]")},
    )
    model = make_data_config()
    model.isel = Mock(side_effect=data.isel)
    dataset = ConcreteDataset(
        config=ConcreteDatasetConfig(forecast=model), load_forecast=True
    )
    return dataset, model


@pytest.mark.pruned
@pytest.mark.parametrize("sample_index,expected", [(0, 10.0), (1, 20.0)])
def test_forecast_sample_selects_matching_time(sample_index, expected):
    dataset, model = model_dataset_with_values([10.0, 20.0])
    dataset.forecast_indexes = {INIT_TIME_DIM: np.asarray([0, 1], dtype=np.int64)}

    with patch.object(
        module, "_unwrap_data_variables", side_effect=lambda value: value["tas"]
    ):
        result = dataset._index_forecast_dataset(sample_index)

    model.isel.assert_called_once_with(**{INIT_TIME_DIM: [sample_index]})
    assert result.item() == pytest.approx(expected)


def test_forecast_sample_is_none_when_forecast_not_loaded():
    dataset = ConcreteDataset(
        config=ConcreteDatasetConfig(forecast=make_data_config()), load_forecast=False
    )

    assert dataset._index_forecast_dataset(0) is None


def test_forecast_sample_converts_indexes_to_python_ints():
    model = make_data_config()
    model.isel = Mock(return_value=model.data)
    dataset = ConcreteDataset(
        config=ConcreteDatasetConfig(forecast=model), load_forecast=True
    )
    dataset.forecast_indexes = {
        INIT_TIME_DIM: np.asarray([np.int64(1)]),
        LEAD_TIME_DIM: np.asarray([np.int64(2)]),
    }

    with patch.object(module, "_unwrap_data_variables", return_value="MODEL"):
        result = dataset._index_forecast_dataset(0)

    model.isel.assert_called_once_with(**{INIT_TIME_DIM: [1], LEAD_TIME_DIM: [2]})
    assert result == "MODEL"


@pytest.mark.pruned
def test_forecast_sample_with_all_selection_dimensions():
    model = make_data_config(realizations=[0, 1])
    model.isel = Mock(return_value=model.data)
    dataset = ConcreteDataset(
        config=ConcreteDatasetConfig(forecast=model), load_forecast=True
    )
    dataset.forecast_indexes = {
        INIT_TIME_DIM: np.asarray([1]),
        LEAD_TIME_DIM: np.asarray([2]),
        REALIZATION_DIM: np.asarray([0]),
    }

    with patch.object(
        module, "_unwrap_data_variables", return_value=xr.DataArray([1.0])
    ):
        dataset._index_forecast_dataset(0)

    model.isel.assert_called_once_with(
        **{INIT_TIME_DIM: [1], LEAD_TIME_DIM: [2], REALIZATION_DIM: [0]}
    )


@pytest.mark.pruned
def test_forecast_sample_with_empty_mapping_calls_isel_without_arguments():
    model = make_data_config()
    model.isel = Mock(return_value=model.data)
    dataset = ConcreteDataset(
        config=ConcreteDatasetConfig(forecast=model), load_forecast=True
    )
    dataset.forecast_indexes = {}

    with patch.object(
        module, "_unwrap_data_variables", return_value=xr.DataArray([1.0])
    ):
        dataset._index_forecast_dataset(0)

    model.isel.assert_called_once_with()


def test_condition_sample_is_none_without_condition():
    dataset = ConcreteDataset(config=ConcreteDatasetConfig(forecast=make_data_config()))

    assert dataset._index_condition_dataset(0) is None


def test_static_condition_sample_uses_empty_selection():
    condition = make_data_config(name="condition")
    condition.isel = Mock(return_value=condition.data)
    dataset = make_condition_dataset("static", condition)

    with patch.object(
        module, "_unwrap_data_variables", return_value="CONDITION"
    ) as unwrap:
        result = dataset._index_condition_dataset(0)

    condition.isel.assert_called_once_with()
    unwrap.assert_called_once_with(condition.data)
    assert result == "CONDITION"


def test_condition_sample_selects_matching_time():
    data = xr.Dataset(
        {"condition": (INIT_TIME_DIM, [10.0, 20.0])},
        coords={INIT_TIME_DIM: np.asarray(DATES_2, dtype="datetime64[ns]")},
    )
    condition = make_data_config(name="condition", ensemble_mean=True)
    condition.isel = Mock(side_effect=data.isel)
    dataset = make_condition_dataset("ensemble_mean", condition)
    dataset.cond_indexes = {INIT_TIME_DIM: np.asarray([0, 1])}

    with patch.object(
        module, "_unwrap_data_variables", side_effect=lambda value: value["condition"]
    ):
        result = dataset._index_condition_dataset(1)

    condition.isel.assert_called_once_with(**{INIT_TIME_DIM: [1]})
    assert result.item() == pytest.approx(20.0)


def test_cross_ensemble_sample_draws_random_member():
    condition = make_data_config(name="condition", realizations=[0, 1, 2])
    condition.isel = Mock(return_value=condition.data)
    dataset = make_condition_dataset("cross_ensemble", condition)
    dataset.cond_indexes = {
        INIT_TIME_DIM: np.asarray([0]),
        LEAD_TIME_DIM: np.asarray([1]),
    }
    dataset._rng = Mock()
    dataset._rng.integers.return_value = 2

    with patch.object(module, "_unwrap_data_variables", return_value="CONDITION"):
        result = dataset._index_condition_dataset(0)

    dataset._rng.integers.assert_called_once_with(3)
    condition.isel.assert_called_once_with(
        **{INIT_TIME_DIM: [0], LEAD_TIME_DIM: [1], REALIZATION_DIM: [2]}
    )
    assert result == "CONDITION"


@pytest.mark.pruned
def test_input_shape_without_spatial_dimensions_counts_channels():
    model = make_data_config(
        names=["tas", "pr"], preprocessing_pipeline=make_pipeline()
    )
    dataset = ConcreteDataset(config=ConcreteDatasetConfig(forecast=model))

    assert dataset.get_input_shape() == (2,)


@pytest.mark.pruned
def test_input_shape_uses_spatial_dimensions():
    model = make_data_config(
        names=["tas", "pr"],
        spatial_coords={"lat": [45.0, 46.0], "lon": [-124.0, -123.0, -122.0]},
        preprocessing_pipeline=make_pipeline(),
    )
    dataset = ConcreteDataset(config=ConcreteDatasetConfig(forecast=model))

    assert dataset.get_input_shape() == (2, 2, 3)


def test_input_shape_adds_condition_channels_when_concatenating():
    model = make_data_config(
        names=["tas", "pr"], preprocessing_pipeline=make_pipeline()
    )
    condition = make_data_config(name="condition", names=["zg", "psl", "sic"])
    config = config_with_condition(condition, "ensemble_mean", forecast=model)
    dataset = ConcreteDataset(config=config, concat_condition=True)

    assert dataset.get_input_shape() == (5,)


def test_input_shape_uses_flattener_locations():
    flattener = SimpleNamespace(final_locations=np.zeros((7,)))
    pipeline = make_pipeline(flattener=flattener)
    model = make_data_config(names=["tas", "pr"], preprocessing_pipeline=pipeline)
    dataset = ConcreteDataset(config=ConcreteDatasetConfig(forecast=model))

    assert dataset.get_input_shape() == (2, 7)
    pipeline.get_flattener.assert_called_once_with()


@pytest.mark.pruned
def test_added_features_dimension_counts_selected_features():
    features = AddedTimeFeatures(
        TimeFeatureReference(), [LEAD_TIME_DIM, "month_sin", "month_cos"]
    )
    dataset = ConcreteDataset(time_features=features)

    assert dataset.get_added_features_dim() == 3


@pytest.mark.pruned
def test_added_features_dimension_is_zero_without_features():
    dataset = ConcreteDataset(
        config=ConcreteDatasetConfig(forecast=make_data_config()),
        time_features=AddedTimeFeatures(TimeFeatureReference(), None),
    )

    assert dataset.get_added_features_dim() == 0


@pytest.mark.pruned
def test_compute_with_no_arrays_returns_empty_tuple():
    assert ConcreteDataset._compute() == ()


@pytest.mark.pruned
def test_compute_preserves_argument_order_for_numpy_arrays():
    arrays = (np.asarray([1]), np.asarray([2]), np.asarray([3]))

    result = ConcreteDataset._compute(*arrays)

    assert isinstance(result, tuple)
    for actual, expected in zip(result, arrays, strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.pruned
def test_compute_uses_synchronous_scheduler_and_suppresses_stderr():
    with (
        patch.object(module, "suppress_stderr", return_value=MagicMock()) as suppress,
        patch.object(module.dask.config, "set", return_value=MagicMock()) as config_set,
        patch.object(
            module.dask, "compute", return_value=("FIRST", "SECOND")
        ) as compute,
    ):
        result = DatasetABC._compute(object(), object())

    suppress.assert_called_once_with()
    config_set.assert_called_once_with(scheduler="synchronous")
    assert compute.call_count == 1
    assert result == ("FIRST", "SECOND")


@pytest.mark.pruned
@pytest.mark.parametrize("length", [0, 1, 2, 10])
def test_len_uses_first_coordinate_length(length):
    dataset = ConcreteDataset(config=ConcreteDatasetConfig(forecast=make_data_config()))
    dataset.sample_coords = {"first": np.arange(length), "second": np.arange(length)}

    assert len(dataset) == length


@pytest.mark.pruned
def test_len_ignores_later_coordinate_lengths():
    dataset = ConcreteDataset(config=ConcreteDatasetConfig(forecast=make_data_config()))
    dataset.sample_coords = {"first": np.arange(3), "second": np.arange(100)}

    assert len(dataset) == 3


@pytest.mark.pruned
def test_len_with_empty_mapping_raises_stop_iteration():
    dataset = ConcreteDataset(config=ConcreteDatasetConfig(forecast=make_data_config()))
    dataset.sample_coords = {}

    with pytest.raises(StopIteration):
        len(dataset)


@pytest.mark.pruned
def test_getitem_returns_index():
    dataset = ConcreteDataset(config=ConcreteDatasetConfig(forecast=make_data_config()))

    assert dataset[10] == 10


def make_init_dataset_for_orchestration(
    *, forecast=None, condition=None, method=None, load_forecast=False
):
    return make_init_dataset(
        forecast=forecast,
        condition=condition,
        method=method,
        load_forecast=load_forecast,
    )


def test_initialisation_opens_forecast_with_time_features_enabled():
    model = make_data_config()
    dataset = make_init_dataset_for_orchestration(forecast=model, load_forecast=True)

    initialize_dataset(dataset)

    model.open_xarray_data.assert_called_once_with(
        load=False, add_time_auxiliary_coords=True
    )


@pytest.mark.pruned
def test_initialisation_skips_forecast_when_not_loaded():
    model = make_data_config()
    dataset = make_init_dataset_for_orchestration(forecast=model, load_forecast=False)

    initialize_dataset(dataset)

    model.open_xarray_data.assert_not_called()
    assert dataset.forecast_indexes is None


@pytest.mark.pruned
def test_initialisation_opens_condition_with_time_features_enabled():
    condition = make_data_config(name="condition", ensemble_mean=True)
    dataset = make_init_dataset_for_orchestration(
        forecast=make_data_config(),
        condition=condition,
        method="ensemble_mean",
    )

    initialize_dataset(dataset)

    condition.open_xarray_data.assert_called_once_with(
        load=False, add_time_auxiliary_coords=True
    )


def test_initialisation_disables_time_features_for_static_condition():
    condition = make_data_config(name="condition")
    condition.realization_list = None
    condition.coords = {"lat": make_coord([45.0], "lat")}
    dataset = make_init_dataset_for_orchestration(
        forecast=make_data_config(), condition=condition, method="static"
    )

    initialize_dataset(dataset)

    condition.open_xarray_data.assert_called_once_with(
        load=False, add_time_auxiliary_coords=False
    )


@pytest.mark.pruned
def test_initialisation_without_condition_leaves_condition_indexes_empty():
    dataset = make_init_dataset_for_orchestration(
        forecast=make_data_config(), condition=None, method=None
    )

    initialize_dataset(dataset)

    assert dataset.cond_indexes is None


@pytest.mark.pruned
def test_initialisation_copies_time_features():
    dataset = make_init_dataset_for_orchestration(forecast=make_data_config())
    original = dataset.time_features

    initialize_dataset(dataset)

    assert dataset.time_features is not original
    assert dataset.time_features == original


@pytest.mark.pruned
def test_initialisation_stores_sampling_coordinates():
    dataset = make_init_dataset_for_orchestration(forecast=make_data_config())
    expected = sample_coordinates()

    initialize_dataset(dataset, {"get_sampling_coords": Mock(return_value=expected)})

    assert dataset.sample_coords is expected


@pytest.mark.pruned
@pytest.mark.parametrize(
    "failing",
    [
        "_check_init",
        "_resolve_mask",
        "_prepare_sampling_mask",
        "get_sampling_coords",
        "get_forecast_indexes",
        "get_cond_indexes",
    ],
)
def test_initialisation_propagates_stage_errors(failing):
    dataset = make_init_dataset_for_orchestration(forecast=make_data_config())

    with pytest.raises(RuntimeError, match=failing):
        initialize_dataset(dataset, {failing: Mock(side_effect=RuntimeError(failing))})


@pytest.mark.pruned
def test_dataset_config_constructor_without_sources_fails_before_method_check():
    with pytest.raises(
        ValueError, match="either forecast or condition data must be provided"
    ):
        ConcreteDatasetConfig(
            forecast=None,
            condition=None,
            condition_method="unsupported",
            run_parent=True,
        )
