from unittest.mock import patch

import cftime
import numpy as np
import pytest
import xarray as xr

from S2D4P.configs import (
    condition_data_allowed_dimensions,
    condition_data_required_dimensions,
    forecast_data_allowed_dimensions,
    forecast_data_required_dimensions,
    lead_time_resolution,
    reference_data_allowed_dimensions,
    reference_data_required_dimensions,
    required_sample_dimensions,
)
from S2D4P.data.data_configs.data_configs import (
    ConditionDataConfig,
    ForecastDataConfig,
    RefDataConfig,
    build_time_range,
)
from S2D4P.data.data_configs.data_configs_abc import DataConfigABC

INIT_TIME_DIM, LEAD_TIME_DIM = required_sample_dimensions


def make_init_times(start="2000-01-01", end="2005-01-01"):
    return xr.DataArray(
        np.array([start, end], dtype="datetime64[ns]"),
        dims=(INIT_TIME_DIM,),
        name=INIT_TIME_DIM,
    )


def make_lead_times(values=None):
    if values is None:
        values = [1, 12]

    return xr.DataArray(values, dims=(LEAD_TIME_DIM,), name=LEAD_TIME_DIM)


class DummyInfo:
    def __init__(
        self,
        start_time=np.datetime64("2000-01-01"),
        final_time=np.datetime64("2005-01-01"),
        init_times=None,
        lead_times=None,
    ):
        init_times = make_init_times() if init_times is None else init_times
        lead_times = make_lead_times() if lead_times is None else lead_times

        self.start_time = start_time
        self.final_time = final_time
        self.sizes = {
            INIT_TIME_DIM: init_times.size,
            LEAD_TIME_DIM: lead_times.size,
        }
        self.coords = {
            INIT_TIME_DIM: init_times,
            LEAD_TIME_DIM: lead_times,
        }
        self.time_coords_type = None
        self.init_time_freq = lead_time_resolution


def patch_data_source(info=None):
    """Stub out data loading on the base class so configs can be built in isolation."""
    if info is None:
        info = DummyInfo()

    return patch.multiple(
        DataConfigABC,
        _resolve_data=lambda self: None,
        _get_ds_info=lambda self: info,
    )


def expected_monthly_range(start="2000-01-01", end="2005-12-01"):
    return xr.date_range(
        start=start,
        end=end,
        freq="MS",
        calendar="proleptic_gregorian",
        use_cftime=False,
    )


def build_forecast(info=None, **overrides):
    values = {"paths": "x", "names": ["var"]}
    values.update(overrides)
    with patch_data_source(info):
        return ForecastDataConfig(**values)


def build_reference(info=None, **overrides):
    values = {"paths": "x", "names": ["var"]}
    values.update(overrides)
    with patch_data_source(info):
        return RefDataConfig(**values)


def build_condition(info=None, **overrides):
    values = {"paths": "x", "names": ["var"]}
    values.update(overrides)
    with patch_data_source(info):
        return ConditionDataConfig(**values)


def test_forecast_basic_fields():
    cfg = build_forecast()

    assert cfg.paths == "x"
    assert cfg.names == ["var"]


def test_forecast_type():
    assert build_forecast().TYPE == "forecast"


def test_forecast_realization_list_defaults_to_none():
    cfg = build_forecast()

    assert cfg.realization_list is None


def test_forecast_realization_list_is_stored():
    cfg = build_forecast(realization_list=[0])

    assert cfg.realization_list == [0]


def test_forecast_default_fields():
    cfg = build_forecast(paths="input/*.nc", names=["tas"])

    assert cfg.realization_list is None
    assert cfg.ensemble_mean is True
    assert cfg.concat_dim is None
    assert cfg.file_type == "*.nc"
    assert cfg.rename_dict is None
    assert cfg.drop_vars_list is None


def test_forecast_time_range_uses_max_lead_time():
    info = DummyInfo(
        init_times=make_init_times("2000-01-01", "2005-01-01"),
        lead_times=make_lead_times([1, 12]),
    )

    cfg = build_forecast(info)

    assert np.array_equal(
        np.asarray(cfg.time_range),
        np.asarray(expected_monthly_range("2000-01-01", "2005-12-01")),
    )


def test_forecast_time_range_extends_with_longer_lead_time():
    info = DummyInfo(
        init_times=make_init_times("2000-01-01", "2005-01-01"),
        lead_times=make_lead_times([1, 24]),
    )

    cfg = build_forecast(info)

    assert cfg.time_range[-1] == np.datetime64("2006-12-01")


def test_forecast_passes_max_lead_time_to_builder():
    info = DummyInfo(
        init_times=make_init_times("2000-01-01", "2000-03-01"),
        lead_times=make_lead_times([1, 3, 6]),
    )

    with patch(
        "S2D4P.data.data_configs.build_time_range",
        return_value=np.array(["2000-01-01"], dtype="datetime64[ns]"),
    ) as mock_build:
        build_forecast(info)

    assert mock_build.call_args.kwargs["init_time"] is info.coords[INIT_TIME_DIM]
    assert mock_build.call_args.kwargs["n_lead_times"] == 6
    assert mock_build.call_args.kwargs["lead_time_resolution"] == lead_time_resolution


def test_forecast_passes_exact_time_range_arguments():
    info = DummyInfo(
        init_times=make_init_times("2001-01-01", "2002-01-01"),
        lead_times=make_lead_times([1, 4, 9]),
    )
    expected = object()

    with patch(
        "S2D4P.data.data_configs.build_time_range",
        return_value=expected,
    ) as build:
        cfg = build_forecast(info)

    assert cfg.time_range is expected
    build.assert_called_once_with(
        init_time=info.coords[INIT_TIME_DIM],
        n_lead_times=9,
        lead_time_resolution=lead_time_resolution,
    )


def test_forecast_resolve_and_info_are_called():
    called = {"resolve": False, "info": False}

    def fake_resolve(self):
        called["resolve"] = True

    def fake_info(self):
        called["info"] = True
        return DummyInfo()

    with patch.multiple(
        DataConfigABC,
        _resolve_data=fake_resolve,
        _get_ds_info=fake_info,
    ):
        ForecastDataConfig(paths="x", names=["var"])

    assert called == {"resolve": True, "info": True}


def test_forecast_instances_get_distinct_preprocessing_pipelines():
    first = build_forecast(paths="first", names=["tas"])
    second = build_forecast(paths="second", names=["pr"])

    assert first.preprocessing_pipeline is not second.preprocessing_pipeline


def test_forecast_dimension_sets_come_from_configs():
    assert ForecastDataConfig._allowed_dims() == forecast_data_allowed_dimensions
    assert ForecastDataConfig._required_dims() == forecast_data_required_dimensions


def test_reference_basic_fields():
    cfg = build_reference()

    assert cfg.paths == "x"
    assert cfg.names == ["var"]


def test_reference_type():
    assert build_reference().TYPE == "reference"


def test_reference_realization_list_is_stored():
    cfg = build_reference(realization_list=[0])

    assert cfg.realization_list == [0]


def test_reference_default_fields():
    cfg = build_reference(paths="observations/*.nc", names=["tas"])

    assert cfg.realization_list is None
    assert cfg.ensemble_mean is True
    assert cfg.concat_dim is None
    assert cfg.file_type == "*.nc"
    assert cfg.rename_dict is None


def test_reference_time_range_is_single_lead_step():
    info = DummyInfo(init_times=make_init_times("2000-01-01", "2005-01-01"))

    cfg = build_reference(info)

    assert np.array_equal(
        np.asarray(cfg.time_range),
        np.asarray(expected_monthly_range("2000-01-01", "2005-01-01")),
    )


def test_reference_passes_exact_time_range_arguments():
    info = DummyInfo(init_times=make_init_times("2001-01-01", "2002-01-01"))
    expected = object()

    with patch(
        "S2D4P.data.data_configs.build_time_range",
        return_value=expected,
    ) as build:
        cfg = build_reference(info)

    assert cfg.time_range is expected
    build.assert_called_once_with(
        init_time=info.coords[INIT_TIME_DIM],
        lead_time_resolution=lead_time_resolution,
    )


def test_reference_rejects_frequency_mismatch():
    info = DummyInfo()
    info.init_time_freq = "day"

    with pytest.raises(RuntimeError, match="same temporal frequency"):
        build_reference(info)


def test_reference_frequency_error_reports_both_values():
    info = DummyInfo()
    info.init_time_freq = "day"

    with pytest.raises(RuntimeError, match="day vs month"):
        build_reference(info)


def test_reference_dimension_sets_come_from_configs():
    assert RefDataConfig._allowed_dims() == reference_data_allowed_dimensions
    assert RefDataConfig._required_dims() == reference_data_required_dimensions


def test_condition_basic_fields():
    cfg = build_condition()

    assert cfg.paths == "x"
    assert cfg.names == ["var"]


def test_condition_type():
    assert build_condition().TYPE == "condition"


def test_condition_realization_list_is_stored():
    cfg = build_condition(realization_list=[0])

    assert cfg.realization_list == [0]


def test_condition_default_fields():
    cfg = build_condition(paths="conditions/*.nc", names=["zg"])

    assert cfg.realization_list is None
    assert cfg.ensemble_mean is True
    assert cfg.concat_dim is None
    assert cfg.file_type == "*.nc"
    assert cfg.rename_dict is None


def test_condition_time_range_uses_max_lead_time():
    info = DummyInfo(
        init_times=make_init_times("2000-01-01", "2005-01-01"),
        lead_times=make_lead_times([1, 12]),
    )

    cfg = build_condition(info)

    assert np.array_equal(
        np.asarray(cfg.time_range),
        np.asarray(expected_monthly_range("2000-01-01", "2005-12-01")),
    )


def test_condition_time_range_extends_with_longer_lead_time():
    info = DummyInfo(
        init_times=make_init_times("2000-01-01", "2005-01-01"),
        lead_times=make_lead_times([1, 24]),
    )

    cfg = build_condition(info)

    assert cfg.time_range[-1] == np.datetime64("2006-12-01")


def test_condition_passes_max_lead_time_to_builder():
    info = DummyInfo(lead_times=make_lead_times([1, 6, 12]))

    with patch(
        "S2D4P.data.data_configs.build_time_range",
        return_value=np.array(["2000-01-01"], dtype="datetime64[ns]"),
    ) as mock_build:
        build_condition(info)

    assert mock_build.call_args.kwargs["init_time"] is info.coords[INIT_TIME_DIM]
    assert mock_build.call_args.kwargs["n_lead_times"] == 12
    assert mock_build.call_args.kwargs["lead_time_resolution"] == lead_time_resolution


@pytest.mark.parametrize(
    ("start_time", "final_time"),
    [
        (None, np.datetime64("2005-01-01")),
        (np.datetime64("2000-01-01"), None),
        (None, None),
    ],
)
def test_condition_incomplete_bounds_set_time_range_to_none(start_time, final_time):
    info = DummyInfo(start_time=start_time, final_time=final_time)

    cfg = build_condition(info)

    assert cfg.time_range is None


@pytest.mark.parametrize(
    ("start_time", "final_time"),
    [
        (None, np.datetime64("2005-01-01")),
        (np.datetime64("2000-01-01"), None),
        (None, None),
    ],
)
def test_condition_incomplete_bounds_skip_time_range_builder(start_time, final_time):
    info = DummyInfo(start_time=start_time, final_time=final_time)

    with patch("S2D4P.data.data_configs.build_time_range") as build:
        build_condition(info)

    build.assert_not_called()


def test_condition_complete_bounds_build_time_range():
    info = DummyInfo(
        start_time=np.datetime64("2000-01-01"),
        final_time=np.datetime64("2005-01-01"),
        lead_times=make_lead_times([1, 6, 18]),
    )
    expected = object()

    with patch(
        "S2D4P.data.data_configs.build_time_range",
        return_value=expected,
    ) as build:
        cfg = build_condition(info)

    assert cfg.time_range is expected
    build.assert_called_once_with(
        init_time=info.coords[INIT_TIME_DIM],
        n_lead_times=18,
        lead_time_resolution=lead_time_resolution,
    )


def test_condition_resolve_and_info_are_called():
    called = {"resolve": False, "info": False}

    def fake_resolve(self):
        called["resolve"] = True

    def fake_info(self):
        called["info"] = True
        return DummyInfo()

    with patch.multiple(
        DataConfigABC,
        _resolve_data=fake_resolve,
        _get_ds_info=fake_info,
    ):
        ConditionDataConfig(paths="x", names=["var"])

    assert called == {"resolve": True, "info": True}


def test_condition_dimension_sets_come_from_configs():
    assert ConditionDataConfig._allowed_dims() == condition_data_allowed_dimensions
    assert ConditionDataConfig._required_dims() == condition_data_required_dimensions


def test_build_time_range_monthly_datetime64():
    init_time = xr.DataArray(
        np.array(["2000-01-01", "2000-03-01"], dtype="datetime64[ns]"),
        dims=(INIT_TIME_DIM,),
    )

    result = build_time_range(
        init_time=init_time,
        n_lead_times=3,
        lead_time_resolution="month",
    )

    expected = expected_monthly_range("2000-01-01", "2000-05-01")
    np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))


def test_build_time_range_daily_datetime64():
    init_time = xr.DataArray(
        np.array(["2000-01-01", "2000-01-03"], dtype="datetime64[ns]"),
        dims=(INIT_TIME_DIM,),
    )

    result = build_time_range(
        init_time=init_time,
        n_lead_times=3,
        lead_time_resolution="day",
    )

    expected = xr.date_range(
        start="2000-01-01",
        end="2000-01-05",
        freq="D",
        calendar="proleptic_gregorian",
        use_cftime=False,
    )
    np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))


def test_build_time_range_single_lead_time_ends_at_latest_init():
    init_time = xr.DataArray(
        np.array(["2000-01-01", "2000-03-01"], dtype="datetime64[ns]"),
        dims=(INIT_TIME_DIM,),
    )

    result = build_time_range(
        init_time=init_time,
        n_lead_times=1,
        lead_time_resolution="month",
    )

    expected = expected_monthly_range("2000-01-01", "2000-03-01")
    np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))


def test_build_time_range_monthly_result_is_inclusive():
    init_time = xr.DataArray(
        np.array(["2000-01-01", "2000-03-01"], dtype="datetime64[ns]"),
        dims=(INIT_TIME_DIM,),
    )

    result = build_time_range(
        init_time=init_time,
        n_lead_times=4,
        lead_time_resolution="month",
    )

    assert result[0] == np.datetime64("2000-01-01")
    assert result[-1] == np.datetime64("2000-06-01")
    assert len(result) == 6


def test_build_time_range_daily_result_is_inclusive():
    init_time = xr.DataArray(
        np.array(["2000-01-01", "2000-01-03"], dtype="datetime64[ns]"),
        dims=(INIT_TIME_DIM,),
    )

    result = build_time_range(
        init_time=init_time,
        n_lead_times=4,
        lead_time_resolution="day",
    )

    assert result[0] == np.datetime64("2000-01-01")
    assert result[-1] == np.datetime64("2000-01-06")
    assert len(result) == 6


def test_build_time_range_uses_min_and_max_of_unsorted_init_times():
    init_time = xr.DataArray(
        np.array(
            ["2000-05-01", "2000-01-01", "2000-03-01"],
            dtype="datetime64[ns]",
        ),
        dims=(INIT_TIME_DIM,),
    )

    result = build_time_range(
        init_time=init_time,
        n_lead_times=2,
        lead_time_resolution="month",
    )

    expected = expected_monthly_range("2000-01-01", "2000-06-01")
    np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))


def test_build_time_range_does_not_mutate_input_order():
    values = np.array(
        ["2000-03-01", "2000-01-01", "2000-02-01"],
        dtype="datetime64[ns]",
    )
    init_time = xr.DataArray(values.copy(), dims=(INIT_TIME_DIM,))

    build_time_range(
        init_time=init_time,
        n_lead_times=2,
        lead_time_resolution="month",
    )

    np.testing.assert_array_equal(init_time.values, values)


def test_build_time_range_single_datetime64_value():
    init_time = xr.DataArray(
        np.array(["2000-01-01"], dtype="datetime64[ns]"),
        dims=(INIT_TIME_DIM,),
    )

    result = build_time_range(
        init_time=init_time,
        n_lead_times=3,
        lead_time_resolution="month",
    )

    expected = np.array(
        ["2000-01-01", "2000-02-01", "2000-03-01"],
        dtype="datetime64[ns]",
    )
    np.testing.assert_array_equal(np.asarray(result), expected)


def test_build_time_range_cftime_monthly():
    init_time = xr.DataArray(
        [
            cftime.DatetimeNoLeap(2000, 1, 1),
            cftime.DatetimeNoLeap(2000, 3, 1),
        ],
        dims=(INIT_TIME_DIM,),
    )

    result = build_time_range(
        init_time=init_time,
        n_lead_times=3,
        lead_time_resolution="month",
    )

    assert isinstance(result, xr.CFTimeIndex)
    assert result[0] == cftime.DatetimeNoLeap(2000, 1, 1)
    assert result[-1] == cftime.DatetimeNoLeap(2000, 5, 1)


def test_build_time_range_cftime_daily():
    init_time = xr.DataArray(
        [
            cftime.DatetimeNoLeap(2001, 1, 1),
            cftime.DatetimeNoLeap(2001, 1, 3),
        ],
        dims=(INIT_TIME_DIM,),
    )

    result = build_time_range(
        init_time=init_time,
        n_lead_times=3,
        lead_time_resolution="day",
    )

    assert isinstance(result, xr.CFTimeIndex)
    assert result[0] == cftime.DatetimeNoLeap(2001, 1, 1)
    assert result[-1] == cftime.DatetimeNoLeap(2001, 1, 5)


def test_build_time_range_preserves_noleap_calendar():
    init_time = xr.DataArray(
        [
            cftime.DatetimeNoLeap(2000, 1, 1),
            cftime.DatetimeNoLeap(2000, 2, 1),
        ],
        dims=(INIT_TIME_DIM,),
    )

    result = build_time_range(
        init_time=init_time,
        n_lead_times=2,
        lead_time_resolution="month",
    )

    assert result.calendar == "noleap"
    assert all(isinstance(value, cftime.DatetimeNoLeap) for value in result)


def test_build_time_range_supports_360_day_calendar():
    init_time = xr.DataArray(
        [
            cftime.Datetime360Day(2000, 1, 1),
            cftime.Datetime360Day(2000, 2, 1),
        ],
        dims=(INIT_TIME_DIM,),
    )

    result = build_time_range(
        init_time=init_time,
        n_lead_times=3,
        lead_time_resolution="month",
    )

    assert result.calendar == "360_day"
    assert result[0] == cftime.Datetime360Day(2000, 1, 1)
    assert result[-1] == cftime.Datetime360Day(2000, 4, 1)


def test_build_time_range_single_cftime_value():
    init_time = xr.DataArray(
        [cftime.DatetimeNoLeap(2000, 1, 1)],
        dims=(INIT_TIME_DIM,),
    )

    result = build_time_range(
        init_time=init_time,
        n_lead_times=2,
        lead_time_resolution="month",
    )

    assert list(result) == [
        cftime.DatetimeNoLeap(2000, 1, 1),
        cftime.DatetimeNoLeap(2000, 2, 1),
    ]


def test_build_time_range_empty_init_time_raises():
    init_time = xr.DataArray(
        np.array([], dtype="datetime64[ns]"),
        dims=(INIT_TIME_DIM,),
    )

    with pytest.raises(ValueError, match="cannot be empty"):
        build_time_range(init_time=init_time)


@pytest.mark.parametrize("n_lead_times", [0, -1])
def test_build_time_range_rejects_non_positive_lead_times(n_lead_times):
    with pytest.raises(ValueError, match="at least 1"):
        build_time_range(
            init_time=make_init_times(),
            n_lead_times=n_lead_times,
        )


def test_build_time_range_rejects_two_dimensional_init_times():
    init_time = xr.DataArray(
        np.array([["2000-01-01", "2000-02-01"]], dtype="datetime64[ns]"),
        dims=("row", INIT_TIME_DIM),
    )

    with pytest.raises(ValueError, match="one-dimensional"):
        build_time_range(init_time=init_time)


def test_build_time_range_rejects_integer_init_times():
    init_time = xr.DataArray([2000, 2001], dims=(INIT_TIME_DIM,))

    with pytest.raises(TypeError, match="numpy.datetime64 or cftime.datetime"):
        build_time_range(init_time=init_time)


@pytest.mark.parametrize(
    "values",
    [
        [1.0, 2.0],
        ["2000-01-01", "2000-02-01"],
        [object(), object()],
    ],
)
def test_build_time_range_rejects_non_temporal_values(values):
    init_time = xr.DataArray(values, dims=(INIT_TIME_DIM,))

    with pytest.raises(TypeError, match="must contain either"):
        build_time_range(init_time=init_time)


@pytest.mark.parametrize(
    "resolution",
    ["hour", "year", "seconds", "", None],
)
def test_build_time_range_rejects_unsupported_resolution(resolution):
    with pytest.raises(ValueError, match="Unsupported lead-time resolution"):
        build_time_range(
            init_time=make_init_times(),
            lead_time_resolution=resolution,
        )


def test_build_time_range_resolution_error_lists_supported_values():
    with pytest.raises(ValueError, match=r"month.*day"):
        build_time_range(
            init_time=make_init_times(),
            lead_time_resolution="hour",
        )


def test_build_time_range_validates_shape_before_type():
    init_time = xr.DataArray(np.ones((2, 2)), dims=("row", INIT_TIME_DIM))

    with pytest.raises(ValueError, match="one-dimensional"):
        build_time_range(init_time=init_time)


def test_build_time_range_validates_lead_count_before_time_type():
    init_time = xr.DataArray([1, 2], dims=(INIT_TIME_DIM,))

    with pytest.raises(ValueError, match="at least 1"):
        build_time_range(init_time=init_time, n_lead_times=0)


def test_build_time_range_validates_time_type_before_resolution():
    init_time = xr.DataArray([1, 2], dims=(INIT_TIME_DIM,))

    with pytest.raises(TypeError, match="numpy.datetime64 or cftime.datetime"):
        build_time_range(init_time=init_time, lead_time_resolution="hour")


def test_build_time_range_calls_xarray_date_range_twice_for_monthly():
    init_time = xr.DataArray(
        np.array(["2000-01-01", "2000-03-01"], dtype="datetime64[ns]"),
        dims=(INIT_TIME_DIM,),
    )
    real_date_range = xr.date_range

    with patch(
        "S2D4P.data.data_configs.xr.date_range",
        side_effect=real_date_range,
    ) as date_range:
        build_time_range(
            init_time=init_time,
            n_lead_times=3,
            lead_time_resolution="month",
        )

    assert date_range.call_count == 2

    first_call = date_range.call_args_list[0]
    assert first_call.kwargs["periods"] == 3
    assert first_call.kwargs["freq"] == "MS"
    assert first_call.kwargs["calendar"] == "proleptic_gregorian"
    assert first_call.kwargs["use_cftime"] is False

    second_call = date_range.call_args_list[1]
    assert second_call.kwargs["freq"] == "MS"
    assert second_call.kwargs["calendar"] == "proleptic_gregorian"
    assert second_call.kwargs["use_cftime"] is False


def test_build_time_range_uses_daily_frequency_code():
    init_time = xr.DataArray(
        np.array(["2000-01-01", "2000-01-03"], dtype="datetime64[ns]"),
        dims=(INIT_TIME_DIM,),
    )
    real_date_range = xr.date_range

    with patch(
        "S2D4P.data.data_configs.xr.date_range",
        side_effect=real_date_range,
    ) as date_range:
        build_time_range(
            init_time=init_time,
            n_lead_times=2,
            lead_time_resolution="day",
        )

    assert all(call.kwargs["freq"] == "D" for call in date_range.call_args_list)


def test_build_time_range_uses_cftime_calendar_in_both_calls():
    init_time = xr.DataArray(
        [
            cftime.DatetimeNoLeap(2000, 1, 1),
            cftime.DatetimeNoLeap(2000, 3, 1),
        ],
        dims=(INIT_TIME_DIM,),
    )
    real_date_range = xr.date_range

    with patch(
        "S2D4P.data.data_configs.xr.date_range",
        side_effect=real_date_range,
    ) as date_range:
        build_time_range(
            init_time=init_time,
            n_lead_times=2,
            lead_time_resolution="month",
        )

    assert date_range.call_count == 2
    assert all(
        call.kwargs["calendar"] == "noleap" for call in date_range.call_args_list
    )
    assert all(call.kwargs["use_cftime"] is True for call in date_range.call_args_list)
