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


@pytest.mark.pruned
def test_obs_data_config_uses_default_single_lead_time():
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


@pytest.mark.pruned
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


def test_condition_data_config_none_time_range():
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


@pytest.mark.pruned
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
def test_condition_data_config_requires_both_time_bounds(
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


@pytest.mark.pruned
def test_build_time_range_invalid_resolution():
    init_time = make_init_times()

    with pytest.raises(KeyError):
        build_time_range(
            init_time=init_time,
            lead_time_resolution="hour",
        )
