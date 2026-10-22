from unittest.mock import patch

import cftime
import numpy as np
import pytest
import xarray as xr

from cccma_ppp.configs import (
    lead_time_resolution,
    required_sample_dimensions,
)
from cccma_ppp.data_modules.data.data_configs import (
    ConditionDataConfig,
    ModelDataConfig,
    ObsDataConfig,
    build_time_range,
)

INIT_TIME_DIM, LEAD_TIME_DIM = required_sample_dimensions


def make_init_times(
    start="2000-01-01",
    end="2005-01-01",
):
    return xr.DataArray(
        np.array(
            [
                start,
                end,
            ],
            dtype="datetime64[ns]",
        ),
        dims=(INIT_TIME_DIM,),
        name=INIT_TIME_DIM,
    )


def make_lead_times(
    values=None,
):
    if values is None:
        values = [1, 12]

    return xr.DataArray(
        values,
        dims=(LEAD_TIME_DIM,),
        name=LEAD_TIME_DIM,
    )


class DummyInfo:
    def __init__(
        self,
        start_time=np.datetime64("2000-01-01"),
        final_time=np.datetime64("2005-01-01"),
        init_times=None,
        lead_times=None,
        sizes=None,
    ):
        if init_times is None:
            init_times = make_init_times()

        if lead_times is None:
            lead_times = make_lead_times()

        self.start_time = start_time
        self.final_time = final_time
        self.sizes = (
            {
                INIT_TIME_DIM: init_times.size,
                LEAD_TIME_DIM: lead_times.size,
            }
            if sizes is None
            else sizes
        )
        self.coords = {
            INIT_TIME_DIM: init_times,
            LEAD_TIME_DIM: lead_times,
        }
        self.time_coords_type = None
        self.init_time_freq = lead_time_resolution


def patch_common(
    info=None,
):
    if info is None:
        info = DummyInfo()

    return patch.multiple(
        "cccma_ppp.data_modules.data.data_abc",
        _resolve_data=lambda self: None,
        _get_ds_info=lambda self: info,
    )


def expected_monthly_range(
    start="2000-01-01",
    end="2005-12-01",
):
    return xr.date_range(
        start=start,
        end=end,
        freq="MS",
        calendar="proleptic_gregorian",
        use_cftime=False,
    )


@pytest.mark.pruned
def test_model_data_config_basic():
    with patch_common():
        cfg = ModelDataConfig(
            paths="x",
            names=["var"],
        )

    assert cfg.paths == "x"
    assert cfg.names == ["var"]


@pytest.mark.pruned
def test_model_data_config_type():
    with patch_common():
        cfg = ModelDataConfig(
            paths="x",
            names=["var"],
        )

    assert cfg.TYPE == "model"


@pytest.mark.pruned
def test_model_data_config_check_realization_false():
    with patch_common():
        cfg = ModelDataConfig(
            paths="x",
            names=["var"],
        )

    assert cfg._check_ensemble is False
    assert cfg.realization_list is None


@pytest.mark.pruned
def test_model_data_config_check_realization_true():
    with patch_common():
        cfg = ModelDataConfig(
            paths="x",
            names=["var"],
            realization_list=[0],
        )

    assert cfg._check_ensemble is True
    assert cfg.realization_list == [0]


@pytest.mark.pruned
def test_model_data_config_time_range():
    info = DummyInfo(
        init_times=make_init_times(
            "2000-01-01",
            "2005-01-01",
        ),
        lead_times=make_lead_times([1, 12]),
    )

    with patch_common(info):
        cfg = ModelDataConfig(
            paths="x",
            names=["var"],
        )

    assert np.array_equal(
        np.asarray(cfg.time_range),
        np.asarray(
            expected_monthly_range(
                start="2000-01-01",
                end="2005-12-01",
            )
        ),
    )


@pytest.mark.pruned
def test_model_data_config_time_range_extended():
    info = DummyInfo(
        init_times=make_init_times(
            "2000-01-01",
            "2005-01-01",
        ),
        lead_times=make_lead_times([1, 24]),
    )

    with patch_common(info):
        cfg = ModelDataConfig(
            paths="x",
            names=["var"],
        )

    assert cfg.time_range[-1] == np.datetime64("2006-12-01")


@pytest.mark.pruned
def test_model_data_config_uses_maximum_lead_time():
    info = DummyInfo(
        init_times=make_init_times(
            "2000-01-01",
            "2000-03-01",
        ),
        lead_times=make_lead_times(
            [
                1,
                3,
                6,
            ]
        ),
    )

    with patch(
        "cccma_ppp.data_modules.data.data_configs.build_time_range",
        return_value=np.array(
            ["2000-01-01"],
            dtype="datetime64[ns]",
        ),
    ) as mock_build:
        with patch_common(info):
            ModelDataConfig(
                paths="x",
                names=["var"],
            )

    assert mock_build.call_args.kwargs["init_time"] is (info.coords[INIT_TIME_DIM])
    assert mock_build.call_args.kwargs["n_lead_times"] == 6
    assert mock_build.call_args.kwargs["lead_time_resolution"] == lead_time_resolution


@pytest.mark.pruned
def test_model_data_config_resolve_called():
    called = {
        "resolve": False,
    }

    def fake_resolve(self):
        called["resolve"] = True

    with patch.multiple(
        "cccma_ppp.data_modules.data.data_abc",
        _resolve_data=fake_resolve,
        _get_ds_info=lambda self: DummyInfo(),
    ):
        ModelDataConfig(
            paths="x",
            names=["var"],
        )

    assert called["resolve"] is True


@pytest.mark.pruned
def test_model_data_config_get_info_called():
    called = {
        "info": False,
    }

    def fake_info(self):
        called["info"] = True
        return DummyInfo()

    with patch.multiple(
        "cccma_ppp.data_modules.data.data_abc",
        _resolve_data=lambda self: None,
        _get_ds_info=fake_info,
    ):
        ModelDataConfig(
            paths="x",
            names=["var"],
        )

    assert called["info"] is True


@pytest.mark.pruned
def test_obs_data_config_basic():
    with patch_common():
        cfg = ObsDataConfig(
            paths="x",
            names=["var"],
        )

    assert cfg.paths == "x"
    assert cfg.names == ["var"]


@pytest.mark.pruned
def test_obs_data_config_type():
    with patch_common():
        cfg = ObsDataConfig(
            paths="x",
            names=["var"],
        )

    assert cfg.TYPE == "observation"


@pytest.mark.pruned
def test_obs_data_config_check_realization_false():
    with patch_common():
        cfg = ObsDataConfig(
            paths="x",
            names=["var"],
        )

    assert cfg._check_ensemble is False
    assert cfg.realization_list is None


def test_obs_data_config_check_realization_true():
    with patch_common():
        cfg = ObsDataConfig(
            paths="x",
            names=["var"],
            realization_list=[0],
        )

    assert cfg._check_ensemble is True
    assert cfg.realization_list == [0]


@pytest.mark.pruned
def test_obs_data_config_time_range():
    info = DummyInfo(
        init_times=make_init_times(
            "2000-01-01",
            "2005-01-01",
        ),
    )

    with patch_common(info):
        cfg = ObsDataConfig(
            paths="x",
            names=["var"],
        )

    assert np.array_equal(
        np.asarray(cfg.time_range),
        np.asarray(
            expected_monthly_range(
                start="2000-01-01",
                end="2005-01-01",
            )
        ),
    )


def legacy_obs_data_config_uses_default_single_lead_time():
    info = DummyInfo()

    with patch(
        "cccma_ppp.data_modules.data.data_configs.build_time_range",
        return_value=np.array(
            ["2000-01-01"],
            dtype="datetime64[ns]",
        ),
    ) as mock_build:
        with patch_common(info):
            ObsDataConfig(
                paths="x",
                names=["var"],
            )

    mock_build.assert_called_once_with(
        info.coords[INIT_TIME_DIM],
    )


@pytest.mark.pruned
def test_obs_data_config_resolve_called():
    called = {
        "resolve": False,
    }

    def fake_resolve(self):
        called["resolve"] = True

    with patch.multiple(
        "cccma_ppp.data_modules.data.data_abc",
        _resolve_data=fake_resolve,
        _get_ds_info=lambda self: DummyInfo(),
    ):
        ObsDataConfig(
            paths="x",
            names=["var"],
        )

    assert called["resolve"] is True


@pytest.mark.pruned
def test_obs_data_config_get_info_called():
    called = {
        "info": False,
    }

    def fake_info(self):
        called["info"] = True
        return DummyInfo()

    with patch.multiple(
        "cccma_ppp.data_modules.data.data_abc",
        _resolve_data=lambda self: None,
        _get_ds_info=fake_info,
    ):
        ObsDataConfig(
            paths="x",
            names=["var"],
        )

    assert called["info"] is True


@pytest.mark.pruned
def test_condition_data_config_basic():
    with patch_common():
        cfg = ConditionDataConfig(
            paths="x",
            names=["var"],
        )

    assert cfg.paths == "x"
    assert cfg.names == ["var"]


@pytest.mark.pruned
def test_condition_data_config_type():
    with patch_common():
        cfg = ConditionDataConfig(
            paths="x",
            names=["var"],
        )

    assert cfg.TYPE == "condition"


@pytest.mark.pruned
def test_condition_data_config_check_realization_false():
    with patch_common():
        cfg = ConditionDataConfig(
            paths="x",
            names=["var"],
        )

    assert cfg._check_ensemble is False
    assert cfg.realization_list is None


def test_condition_data_config_check_realization_true():
    with patch_common():
        cfg = ConditionDataConfig(
            paths="x",
            names=["var"],
            realization_list=[0],
        )

    assert cfg._check_ensemble is True
    assert cfg.realization_list == [0]


@pytest.mark.pruned
def test_condition_data_config_time_range():
    info = DummyInfo(
        init_times=make_init_times(
            "2000-01-01",
            "2005-01-01",
        ),
        lead_times=make_lead_times([1, 12]),
    )

    with patch_common(info):
        cfg = ConditionDataConfig(
            paths="x",
            names=["var"],
        )

    assert np.array_equal(
        np.asarray(cfg.time_range),
        np.asarray(
            expected_monthly_range(
                start="2000-01-01",
                end="2005-12-01",
            )
        ),
    )


@pytest.mark.pruned
def test_condition_data_config_time_range_extended():
    info = DummyInfo(
        init_times=make_init_times(
            "2000-01-01",
            "2005-01-01",
        ),
        lead_times=make_lead_times([1, 24]),
    )

    with patch_common(info):
        cfg = ConditionDataConfig(
            paths="x",
            names=["var"],
        )

    assert cfg.time_range[-1] == np.datetime64("2006-12-01")


def legacy_condition_data_config_none_time_range():
    info = DummyInfo(
        start_time=None,
        final_time=None,
    )

    with patch_common(info):
        cfg = ConditionDataConfig(
            paths="x",
            names=["var"],
        )

    assert not hasattr(
        cfg,
        "time_range",
    )


@pytest.mark.parametrize(
    "start_time,final_time",
    [
        (
            None,
            np.datetime64("2005-01-01"),
        ),
        (
            np.datetime64("2000-01-01"),
            None,
        ),
        (
            None,
            None,
        ),
    ],
)
def legacy_condition_data_config_requires_both_time_bounds(
    start_time,
    final_time,
):
    info = DummyInfo(
        start_time=start_time,
        final_time=final_time,
    )

    with patch_common(info):
        cfg = ConditionDataConfig(
            paths="x",
            names=["var"],
        )

    assert not hasattr(
        cfg,
        "time_range",
    )


@pytest.mark.pruned
def test_condition_data_config_uses_maximum_lead_time():
    info = DummyInfo(
        lead_times=make_lead_times(
            [
                1,
                6,
                12,
            ]
        ),
    )

    with patch(
        "cccma_ppp.data_modules.data.data_configs.build_time_range",
        return_value=np.array(
            ["2000-01-01"],
            dtype="datetime64[ns]",
        ),
    ) as mock_build:
        with patch_common(info):
            ConditionDataConfig(
                paths="x",
                names=["var"],
            )

    assert mock_build.call_args.kwargs["init_time"] is (info.coords[INIT_TIME_DIM])
    assert mock_build.call_args.kwargs["n_lead_times"] == 12
    assert mock_build.call_args.kwargs["lead_time_resolution"] == lead_time_resolution


@pytest.mark.pruned
def test_condition_data_config_resolve_called():
    called = {
        "resolve": False,
    }

    def fake_resolve(self):
        called["resolve"] = True

    with patch.multiple(
        "cccma_ppp.data_modules.data.data_abc",
        _resolve_data=fake_resolve,
        _get_ds_info=lambda self: DummyInfo(),
    ):
        ConditionDataConfig(
            paths="x",
            names=["var"],
        )

    assert called["resolve"] is True


@pytest.mark.pruned
def test_condition_data_config_get_info_called():
    called = {
        "info": False,
    }

    def fake_info(self):
        called["info"] = True
        return DummyInfo()

    with patch.multiple(
        "cccma_ppp.data_modules.data.data_abc",
        _resolve_data=lambda self: None,
        _get_ds_info=fake_info,
    ):
        ConditionDataConfig(
            paths="x",
            names=["var"],
        )

    assert called["info"] is True


@pytest.mark.pruned
def test_build_time_range_monthly_datetime64():
    init_time = xr.DataArray(
        np.array(
            [
                "2000-01-01",
                "2000-03-01",
            ],
            dtype="datetime64[ns]",
        ),
        dims=(INIT_TIME_DIM,),
    )

    result = build_time_range(
        init_time=init_time,
        n_lead_times=3,
        lead_time_resolution="month",
    )

    expected = xr.date_range(
        start="2000-01-01",
        end="2000-05-01",
        freq="MS",
        calendar="proleptic_gregorian",
        use_cftime=False,
    )

    assert np.array_equal(
        np.asarray(result),
        np.asarray(expected),
    )


@pytest.mark.pruned
def test_build_time_range_daily_datetime64():
    init_time = xr.DataArray(
        np.array(
            [
                "2000-01-01",
                "2000-01-03",
            ],
            dtype="datetime64[ns]",
        ),
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

    assert np.array_equal(
        np.asarray(result),
        np.asarray(expected),
    )


@pytest.mark.pruned
def test_build_time_range_single_lead_time():
    init_time = xr.DataArray(
        np.array(
            [
                "2000-01-01",
                "2000-03-01",
            ],
            dtype="datetime64[ns]",
        ),
        dims=(INIT_TIME_DIM,),
    )

    result = build_time_range(
        init_time=init_time,
        n_lead_times=1,
        lead_time_resolution="month",
    )

    expected = xr.date_range(
        start="2000-01-01",
        end="2000-03-01",
        freq="MS",
        calendar="proleptic_gregorian",
        use_cftime=False,
    )

    assert np.array_equal(
        np.asarray(result),
        np.asarray(expected),
    )


@pytest.mark.pruned
def test_build_time_range_cftime():
    init_time = xr.DataArray(
        [
            cftime.DatetimeNoLeap(
                2000,
                1,
                1,
            ),
            cftime.DatetimeNoLeap(
                2000,
                3,
                1,
            ),
        ],
        dims=(INIT_TIME_DIM,),
    )

    result = build_time_range(
        init_time=init_time,
        n_lead_times=3,
        lead_time_resolution="month",
    )

    assert isinstance(
        result,
        xr.CFTimeIndex,
    )
    assert result[0] == cftime.DatetimeNoLeap(
        2000,
        1,
        1,
    )
    assert result[-1] == cftime.DatetimeNoLeap(
        2000,
        5,
        1,
    )


@pytest.mark.pruned
def test_build_time_range_empty_init_time():
    init_time = xr.DataArray(
        np.array(
            [],
            dtype="datetime64[ns]",
        ),
        dims=(INIT_TIME_DIM,),
    )

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        build_time_range(
            init_time=init_time,
        )


@pytest.mark.parametrize(
    "n_lead_times",
    [
        0,
        -1,
    ],
)
def test_build_time_range_invalid_lead_times(
    n_lead_times,
):
    init_time = make_init_times()

    with pytest.raises(
        ValueError,
        match="at least 1",
    ):
        build_time_range(
            init_time=init_time,
            n_lead_times=n_lead_times,
        )


def test_build_time_range_invalid_time_type():
    init_time = xr.DataArray(
        [
            2000,
            2001,
        ],
        dims=(INIT_TIME_DIM,),
    )

    with pytest.raises(
        TypeError,
        match="numpy.datetime64 or cftime.datetime",
    ):
        build_time_range(
            init_time=init_time,
        )


def legacy_build_time_range_invalid_resolution():
    init_time = make_init_times()

    with pytest.raises(KeyError):
        build_time_range(
            init_time=init_time,
            lead_time_resolution="hour",
        )


@pytest.mark.pruned
def test_model_data_config_allowed_dimensions():
    assert ModelDataConfig._allowed_dims() == (ModelDataConfig._allowed_dims())


@pytest.mark.pruned
def test_model_data_config_required_dimensions():
    assert ModelDataConfig._required_dims() == (ModelDataConfig._required_dims())


@pytest.mark.pruned
def test_obs_data_config_allowed_dimensions():
    assert ObsDataConfig._allowed_dims() == (ObsDataConfig._allowed_dims())


@pytest.mark.pruned
def test_obs_data_config_required_dimensions():
    assert ObsDataConfig._required_dims() == (ObsDataConfig._required_dims())


@pytest.mark.pruned
def test_condition_data_config_allowed_dimensions():
    assert ConditionDataConfig._allowed_dims() == (ConditionDataConfig._allowed_dims())


@pytest.mark.pruned
def test_condition_data_config_required_dimensions():
    assert ConditionDataConfig._required_dims() == (
        ConditionDataConfig._required_dims()
    )


@pytest.mark.pruned
def test_model_data_config_default_fields():
    with patch_common():
        config = ModelDataConfig(
            paths="input/*.nc",
            names=["tas"],
        )

    assert config.realization_list is None
    assert config.ensemble_mean is True
    assert config.concat_dim is None
    assert config.file_type == "*.nc"
    assert config.rename_dict is None


@pytest.mark.pruned
def test_obs_data_config_default_fields():
    with patch_common():
        config = ObsDataConfig(
            paths="observations/*.nc",
            names=["tas"],
        )

    assert config.realization_list is None
    assert config.ensemble_mean is True
    assert config.concat_dim is None
    assert config.file_type == "*.nc"
    assert config.rename_dict is None


@pytest.mark.pruned
def test_condition_data_config_default_fields():
    with patch_common():
        config = ConditionDataConfig(
            paths="conditions/*.nc",
            names=["zg"],
        )

    assert config.realization_list is None
    assert config.ensemble_mean is True
    assert config.concat_dim is None
    assert config.file_type == "*.nc"
    assert config.rename_dict is None


@pytest.mark.pruned
def test_data_config_instances_receive_distinct_preprocessing_pipelines():
    with patch_common():
        first = ModelDataConfig(
            paths="first",
            names=["tas"],
        )
        second = ModelDataConfig(
            paths="second",
            names=["pr"],
        )

    assert first.preprocessing_pipeline is not (second.preprocessing_pipeline)


@pytest.mark.pruned
def test_model_data_config_passes_exact_time_range_arguments():
    info = DummyInfo(
        init_times=make_init_times(
            "2001-01-01",
            "2002-01-01",
        ),
        lead_times=make_lead_times(
            [
                1,
                4,
                9,
            ]
        ),
    )
    expected = object()

    with (
        patch(
            "cccma_ppp.data_modules.data.data_configs.build_time_range",
            return_value=expected,
        ) as build,
        patch_common(info),
    ):
        config = ModelDataConfig(
            paths="x",
            names=["var"],
        )

    assert config.time_range is expected
    build.assert_called_once_with(
        init_time=info.coords[INIT_TIME_DIM],
        n_lead_times=9,
        lead_time_resolution=lead_time_resolution,
    )


@pytest.mark.pruned
def test_obs_data_config_passes_exact_time_range_arguments():
    info = DummyInfo(
        init_times=make_init_times(
            "2001-01-01",
            "2002-01-01",
        ),
    )
    expected = object()

    with (
        patch(
            "cccma_ppp.data_modules.data.data_configs.build_time_range",
            return_value=expected,
        ) as build,
        patch_common(info),
    ):
        config = ObsDataConfig(
            paths="x",
            names=["var"],
        )

    assert config.time_range is expected
    build.assert_called_once_with(
        init_time=info.coords[INIT_TIME_DIM],
        lead_time_resolution=lead_time_resolution,
    )


@pytest.mark.pruned
def test_obs_data_config_rejects_frequency_mismatch():
    info = DummyInfo()
    info.init_time_freq = "day"

    with (
        patch_common(info),
        pytest.raises(
            RuntimeError,
            match="same temporal frequency",
        ),
    ):
        ObsDataConfig(
            paths="x",
            names=["var"],
        )


def test_obs_data_config_frequency_error_reports_both_values():
    info = DummyInfo()
    info.init_time_freq = "day"

    with (
        patch_common(info),
        pytest.raises(
            RuntimeError,
            match="day vs month",
        ),
    ):
        ObsDataConfig(
            paths="x",
            names=["var"],
        )


@pytest.mark.parametrize(
    (
        "start_time",
        "final_time",
    ),
    [
        (
            None,
            np.datetime64("2005-01-01"),
        ),
        (
            np.datetime64("2000-01-01"),
            None,
        ),
        (
            None,
            None,
        ),
    ],
)
def test_condition_data_config_incomplete_bounds_set_time_range_to_none(
    start_time,
    final_time,
):
    info = DummyInfo(
        start_time=start_time,
        final_time=final_time,
    )

    with patch_common(info):
        config = ConditionDataConfig(
            paths="x",
            names=["var"],
        )

    assert config.time_range is None


@pytest.mark.pruned
@pytest.mark.parametrize(
    (
        "start_time",
        "final_time",
    ),
    [
        (
            None,
            np.datetime64("2005-01-01"),
        ),
        (
            np.datetime64("2000-01-01"),
            None,
        ),
        (
            None,
            None,
        ),
    ],
)
def test_condition_data_config_incomplete_bounds_skip_time_range_builder(
    start_time,
    final_time,
):
    info = DummyInfo(
        start_time=start_time,
        final_time=final_time,
    )

    with (
        patch(
            "cccma_ppp.data_modules.data.data_configs.build_time_range",
        ) as build,
        patch_common(info),
    ):
        ConditionDataConfig(
            paths="x",
            names=["var"],
        )

    build.assert_not_called()


@pytest.mark.pruned
def test_condition_data_config_complete_bounds_build_time_range():
    info = DummyInfo(
        start_time=np.datetime64("2000-01-01"),
        final_time=np.datetime64("2005-01-01"),
        lead_times=make_lead_times(
            [
                1,
                6,
                18,
            ]
        ),
    )
    expected = object()

    with (
        patch(
            "cccma_ppp.data_modules.data.data_configs.build_time_range",
            return_value=expected,
        ) as build,
        patch_common(info),
    ):
        config = ConditionDataConfig(
            paths="x",
            names=["var"],
        )

    assert config.time_range is expected
    build.assert_called_once_with(
        init_time=info.coords[INIT_TIME_DIM],
        n_lead_times=18,
        lead_time_resolution=lead_time_resolution,
    )


def test_build_time_range_rejects_two_dimensional_initialization_times():
    init_time = xr.DataArray(
        np.asarray(
            [
                [
                    "2000-01-01",
                    "2000-02-01",
                ]
            ],
            dtype="datetime64[ns]",
        ),
        dims=(
            "row",
            INIT_TIME_DIM,
        ),
    )

    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        build_time_range(
            init_time=init_time,
        )


@pytest.mark.parametrize(
    "lead_time_resolution_value",
    [
        "hour",
        "year",
        "seconds",
        "",
        None,
    ],
)
def test_build_time_range_rejects_unsupported_resolution(
    lead_time_resolution_value,
):
    with pytest.raises(
        ValueError,
        match="Unsupported lead-time resolution",
    ):
        build_time_range(
            init_time=make_init_times(),
            lead_time_resolution=lead_time_resolution_value,
        )


@pytest.mark.pruned
def test_build_time_range_resolution_error_lists_supported_values():
    with pytest.raises(
        ValueError,
        match=r"month.*day",
    ):
        build_time_range(
            init_time=make_init_times(),
            lead_time_resolution="hour",
        )


@pytest.mark.pruned
def test_build_time_range_uses_minimum_and_maximum_initialization_times():
    init_time = xr.DataArray(
        np.asarray(
            [
                "2000-05-01",
                "2000-01-01",
                "2000-03-01",
            ],
            dtype="datetime64[ns]",
        ),
        dims=(INIT_TIME_DIM,),
    )

    result = build_time_range(
        init_time=init_time,
        n_lead_times=2,
        lead_time_resolution="month",
    )

    expected = xr.date_range(
        start="2000-01-01",
        end="2000-06-01",
        freq="MS",
        calendar="proleptic_gregorian",
        use_cftime=False,
    )

    np.testing.assert_array_equal(
        np.asarray(result),
        np.asarray(expected),
    )


@pytest.mark.pruned
def test_build_time_range_does_not_mutate_input_order():
    values = np.asarray(
        [
            "2000-03-01",
            "2000-01-01",
            "2000-02-01",
        ],
        dtype="datetime64[ns]",
    )
    init_time = xr.DataArray(
        values.copy(),
        dims=(INIT_TIME_DIM,),
    )

    build_time_range(
        init_time=init_time,
        n_lead_times=2,
        lead_time_resolution="month",
    )

    np.testing.assert_array_equal(
        init_time.values,
        values,
    )


@pytest.mark.pruned
def test_build_time_range_monthly_result_is_inclusive():
    init_time = xr.DataArray(
        np.asarray(
            [
                "2000-01-01",
                "2000-03-01",
            ],
            dtype="datetime64[ns]",
        ),
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


@pytest.mark.pruned
def test_build_time_range_daily_result_is_inclusive():
    init_time = xr.DataArray(
        np.asarray(
            [
                "2000-01-01",
                "2000-01-03",
            ],
            dtype="datetime64[ns]",
        ),
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


@pytest.mark.pruned
def test_build_time_range_daily_cftime():
    init_time = xr.DataArray(
        [
            cftime.DatetimeNoLeap(
                2001,
                1,
                1,
            ),
            cftime.DatetimeNoLeap(
                2001,
                1,
                3,
            ),
        ],
        dims=(INIT_TIME_DIM,),
    )

    result = build_time_range(
        init_time=init_time,
        n_lead_times=3,
        lead_time_resolution="day",
    )

    assert isinstance(
        result,
        xr.CFTimeIndex,
    )
    assert result[0] == cftime.DatetimeNoLeap(
        2001,
        1,
        1,
    )
    assert result[-1] == cftime.DatetimeNoLeap(
        2001,
        1,
        5,
    )


@pytest.mark.pruned
def test_build_time_range_preserves_noleap_calendar():
    init_time = xr.DataArray(
        [
            cftime.DatetimeNoLeap(
                2000,
                1,
                1,
            ),
            cftime.DatetimeNoLeap(
                2000,
                2,
                1,
            ),
        ],
        dims=(INIT_TIME_DIM,),
    )

    result = build_time_range(
        init_time=init_time,
        n_lead_times=2,
        lead_time_resolution="month",
    )

    assert result.calendar == "noleap"
    assert all(
        isinstance(
            value,
            cftime.DatetimeNoLeap,
        )
        for value in result
    )


@pytest.mark.pruned
def test_build_time_range_supports_360_day_calendar():
    init_time = xr.DataArray(
        [
            cftime.Datetime360Day(
                2000,
                1,
                1,
            ),
            cftime.Datetime360Day(
                2000,
                2,
                1,
            ),
        ],
        dims=(INIT_TIME_DIM,),
    )

    result = build_time_range(
        init_time=init_time,
        n_lead_times=3,
        lead_time_resolution="month",
    )

    assert result.calendar == "360_day"
    assert result[0] == cftime.Datetime360Day(
        2000,
        1,
        1,
    )
    assert result[-1] == cftime.Datetime360Day(
        2000,
        4,
        1,
    )


@pytest.mark.pruned
def test_build_time_range_supports_single_datetime64_value():
    init_time = xr.DataArray(
        np.asarray(
            [
                "2000-01-01",
            ],
            dtype="datetime64[ns]",
        ),
        dims=(INIT_TIME_DIM,),
    )

    result = build_time_range(
        init_time=init_time,
        n_lead_times=3,
        lead_time_resolution="month",
    )

    expected = np.asarray(
        [
            "2000-01-01",
            "2000-02-01",
            "2000-03-01",
        ],
        dtype="datetime64[ns]",
    )

    np.testing.assert_array_equal(
        np.asarray(result),
        expected,
    )


@pytest.mark.pruned
def test_build_time_range_supports_single_cftime_value():
    init_time = xr.DataArray(
        [
            cftime.DatetimeNoLeap(
                2000,
                1,
                1,
            )
        ],
        dims=(INIT_TIME_DIM,),
    )

    result = build_time_range(
        init_time=init_time,
        n_lead_times=2,
        lead_time_resolution="month",
    )

    assert list(result) == [
        cftime.DatetimeNoLeap(
            2000,
            1,
            1,
        ),
        cftime.DatetimeNoLeap(
            2000,
            2,
            1,
        ),
    ]


@pytest.mark.pruned
@pytest.mark.parametrize(
    "values",
    [
        [
            1.0,
            2.0,
        ],
        [
            "2000-01-01",
            "2000-02-01",
        ],
        [
            object(),
            object(),
        ],
    ],
)
def test_build_time_range_rejects_non_temporal_values(
    values,
):
    init_time = xr.DataArray(
        values,
        dims=(INIT_TIME_DIM,),
    )

    with pytest.raises(
        TypeError,
        match="must contain either",
    ):
        build_time_range(
            init_time=init_time,
        )


@pytest.mark.pruned
def test_build_time_range_validates_shape_before_type():
    init_time = xr.DataArray(
        np.ones(
            (
                2,
                2,
            )
        ),
        dims=(
            "row",
            INIT_TIME_DIM,
        ),
    )

    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        build_time_range(
            init_time=init_time,
        )


@pytest.mark.pruned
def test_build_time_range_validates_lead_count_before_time_type():
    init_time = xr.DataArray(
        [
            1,
            2,
        ],
        dims=(INIT_TIME_DIM,),
    )

    with pytest.raises(
        ValueError,
        match="at least 1",
    ):
        build_time_range(
            init_time=init_time,
            n_lead_times=0,
        )


@pytest.mark.pruned
def test_build_time_range_validates_time_type_before_resolution():
    init_time = xr.DataArray(
        [
            1,
            2,
        ],
        dims=(INIT_TIME_DIM,),
    )

    with pytest.raises(
        TypeError,
        match="numpy.datetime64 or cftime.datetime",
    ):
        build_time_range(
            init_time=init_time,
            lead_time_resolution="hour",
        )


@pytest.mark.pruned
def test_build_time_range_calls_xarray_date_range_twice():
    init_time = xr.DataArray(
        np.asarray(
            [
                "2000-01-01",
                "2000-03-01",
            ],
            dtype="datetime64[ns]",
        ),
        dims=(INIT_TIME_DIM,),
    )

    real_date_range = xr.date_range

    with patch(
        "cccma_ppp.data_modules.data.data_configs.xr.date_range",
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
    assert first_call.kwargs["calendar"] == ("proleptic_gregorian")
    assert first_call.kwargs["use_cftime"] is False

    second_call = date_range.call_args_list[1]
    assert second_call.kwargs["freq"] == "MS"
    assert second_call.kwargs["calendar"] == ("proleptic_gregorian")
    assert second_call.kwargs["use_cftime"] is False


@pytest.mark.pruned
def test_build_time_range_uses_daily_frequency_code():
    init_time = xr.DataArray(
        np.asarray(
            [
                "2000-01-01",
                "2000-01-03",
            ],
            dtype="datetime64[ns]",
        ),
        dims=(INIT_TIME_DIM,),
    )

    real_date_range = xr.date_range

    with patch(
        "cccma_ppp.data_modules.data.data_configs.xr.date_range",
        side_effect=real_date_range,
    ) as date_range:
        build_time_range(
            init_time=init_time,
            n_lead_times=2,
            lead_time_resolution="day",
        )

    assert all(call.kwargs["freq"] == "D" for call in date_range.call_args_list)


@pytest.mark.pruned
def test_build_time_range_uses_cftime_calendar_in_both_calls():
    init_time = xr.DataArray(
        [
            cftime.DatetimeNoLeap(
                2000,
                1,
                1,
            ),
            cftime.DatetimeNoLeap(
                2000,
                3,
                1,
            ),
        ],
        dims=(INIT_TIME_DIM,),
    )

    real_date_range = xr.date_range

    with patch(
        "cccma_ppp.data_modules.data.data_configs.xr.date_range",
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
