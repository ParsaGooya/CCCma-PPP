import pytest
import numpy as np

from unittest.mock import patch

from cccma_ppp.data_modules.data.data_configs import (
    ModelDataConfig,
    ObsDataConfig,
    ConditionDataConfig,
)


class DummyInfo:
    def __init__(
        self,
        start_year=2000,
        final_year=2005,
        sizes=None,
    ):
        self.start_year = start_year
        self.final_year = final_year
        self.sizes = sizes or {"lead_time": 12}


def patch_common():
    return patch.multiple(
        "cccma_ppp.data_modules.data.data_abc.DataConfigABC",
        _resolve_data=lambda self: None,
        _get_ds_info=lambda self: DummyInfo(),
    )


def test_model_data_config_basic():
    with patch_common():
        cfg = ModelDataConfig(
            paths="x",
            names=["var"],
        )

    assert cfg.paths == "x"


@pytest.mark.pruned
def test_model_data_config_type():
    with patch_common():
        cfg = ModelDataConfig(
            paths="x",
            names=["var"],
        )

    assert cfg.TYPE == "model"


@pytest.mark.pruned
# Remove test due to no coverage
def test_model_data_config_allowed_dims():
    dims = ModelDataConfig._allowed_dims()

    assert "year" in dims
    assert "lead_time" in dims


@pytest.mark.pruned
# Remove test due to no coverage
def test_model_data_config_required_dims():
    dims = ModelDataConfig._required_dims()

    assert "lead_time" in dims
    assert "lat" in dims


@pytest.mark.pruned
def test_model_data_config_check_ensemble_false():
    with patch_common():
        cfg = ModelDataConfig(
            paths="x",
            names=["var"],
        )

    assert cfg._check_ensemble is False


def test_model_data_config_check_ensemble_true():
    with patch_common():
        cfg = ModelDataConfig(
            paths="x",
            names=["var"],
            ensemble_list=[0],
        )

    assert cfg._check_ensemble is True


@pytest.mark.pruned
def test_model_data_config_year_range():
    with patch_common():
        cfg = ModelDataConfig(
            paths="x",
            names=["var"],
        )

    assert np.array_equal(
        cfg.year_range,
        np.array([2000, 2001, 2002, 2003, 2004, 2005]),
    )


@pytest.mark.pruned
def test_model_data_config_year_range_extended():
    info = DummyInfo(
        start_year=2000,
        final_year=2005,
        sizes={"lead_time": 24},
    )

    with patch.multiple(
        "cccma_ppp.data_modules.data.data_abc.DataConfigABC",
        _resolve_data=lambda self: None,
        _get_ds_info=lambda self: info,
    ):
        cfg = ModelDataConfig(
            paths="x",
            names=["var"],
        )

    assert cfg.year_range[-1] == 2006


@pytest.mark.pruned
def test_model_data_config_resolve_called():
    called = {"x": False}

    def fake_resolve(self):
        called["x"] = True

    with patch.multiple(
        "cccma_ppp.data_modules.data.data_abc.DataConfigABC",
        _resolve_data=fake_resolve,
        _get_ds_info=lambda self: DummyInfo(),
    ):
        ModelDataConfig(
            paths="x",
            names=["var"],
        )

    assert called["x"] is True


@pytest.mark.pruned
def test_model_data_config_get_info_called():
    called = {"x": False}

    def fake_info(self):
        called["x"] = True
        return DummyInfo()

    with patch.multiple(
        "cccma_ppp.data_modules.data.data_abc.DataConfigABC",
        _resolve_data=lambda self: None,
        _get_ds_info=fake_info,
    ):
        ModelDataConfig(
            paths="x",
            names=["var"],
        )

    assert called["x"] is True


@pytest.mark.pruned
def test_obs_data_config_basic():
    with patch_common():
        cfg = ObsDataConfig(
            paths="x",
            names=["var"],
        )

    assert cfg.paths == "x"


def test_obs_data_config_type():
    with patch_common():
        cfg = ObsDataConfig(
            paths="x",
            names=["var"],
        )

    assert cfg.TYPE == "observation"


@pytest.mark.pruned
# Remove test due to no coverage
def test_obs_data_config_allowed_dims():
    dims = ObsDataConfig._allowed_dims()

    assert "month" in dims


@pytest.mark.pruned
# Remove test due to no coverage
def test_obs_data_config_required_dims():
    dims = ObsDataConfig._required_dims()

    assert "month" in dims


@pytest.mark.pruned
def test_obs_data_config_check_ensemble_false():
    with patch_common():
        cfg = ObsDataConfig(
            paths="x",
            names=["var"],
        )

    assert cfg._check_ensemble is False


def test_obs_data_config_check_ensemble_true():
    with patch_common():
        cfg = ObsDataConfig(
            paths="x",
            names=["var"],
            ensemble_list=[0],
        )

    assert cfg._check_ensemble is True


@pytest.mark.pruned
def test_obs_data_config_year_range():
    with patch_common():
        cfg = ObsDataConfig(
            paths="x",
            names=["var"],
        )

    assert np.array_equal(
        cfg.year_range,
        np.array([2000, 2001, 2002, 2003, 2004, 2005]),
    )


@pytest.mark.pruned
def test_obs_data_config_resolve_called():
    called = {"x": False}

    def fake_resolve(self):
        called["x"] = True

    with patch.multiple(
        "cccma_ppp.data_modules.data.data_abc.DataConfigABC",
        _resolve_data=fake_resolve,
        _get_ds_info=lambda self: DummyInfo(),
    ):
        ObsDataConfig(
            paths="x",
            names=["var"],
        )

    assert called["x"] is True


@pytest.mark.pruned
def test_obs_data_config_get_info_called():
    called = {"x": False}

    def fake_info(self):
        called["x"] = True
        return DummyInfo()

    with patch.multiple(
        "cccma_ppp.data_modules.data.data_abc.DataConfigABC",
        _resolve_data=lambda self: None,
        _get_ds_info=fake_info,
    ):
        ObsDataConfig(
            paths="x",
            names=["var"],
        )

    assert called["x"] is True


@pytest.mark.pruned
def test_condition_data_config_basic():
    with patch_common():
        cfg = ConditionDataConfig(
            paths="x",
            names=["var"],
        )

    assert cfg.paths == "x"


@pytest.mark.pruned
def test_condition_data_config_type():
    with patch_common():
        cfg = ConditionDataConfig(
            paths="x",
            names=["var"],
        )

    assert cfg.TYPE == "condition"


@pytest.mark.pruned
# Remove test due to no coverage
def test_condition_data_config_allowed_dims():
    dims = ConditionDataConfig._allowed_dims()

    assert "lead_time" in dims


@pytest.mark.pruned
# Remove test due to no coverage
def test_condition_data_config_required_dims():
    dims = ConditionDataConfig._required_dims()

    assert "lat" in dims
    assert "lon" in dims


@pytest.mark.pruned
def test_condition_data_config_check_ensemble_false():
    with patch_common():
        cfg = ConditionDataConfig(
            paths="x",
            names=["var"],
        )

    assert cfg._check_ensemble is False


def test_condition_data_config_check_ensemble_true():
    with patch_common():
        cfg = ConditionDataConfig(
            paths="x",
            names=["var"],
            ensemble_list=[0],
        )

    assert cfg._check_ensemble is True


@pytest.mark.pruned
def test_condition_data_config_year_range():
    with patch_common():
        cfg = ConditionDataConfig(
            paths="x",
            names=["var"],
        )

    assert np.array_equal(
        cfg.year_range,
        np.array([2000, 2001, 2002, 2003, 2004, 2005]),
    )


@pytest.mark.pruned
def test_condition_data_config_year_range_extended():
    info = DummyInfo(
        start_year=2000,
        final_year=2005,
        sizes={"lead_time": 24},
    )

    with patch.multiple(
        "cccma_ppp.data_modules.data.data_abc.DataConfigABC",
        _resolve_data=lambda self: None,
        _get_ds_info=lambda self: info,
    ):
        cfg = ConditionDataConfig(
            paths="x",
            names=["var"],
        )

    assert cfg.year_range[-1] == 2006


def test_condition_data_config_none_year_range():
    info = DummyInfo(
        start_year=None,
        final_year=None,
    )

    with patch.multiple(
        "cccma_ppp.data_modules.data.data_abc.DataConfigABC",
        _resolve_data=lambda self: None,
        _get_ds_info=lambda self: info,
    ):
        cfg = ConditionDataConfig(
            paths="x",
            names=["var"],
        )

    assert not hasattr(cfg, "year_range")


@pytest.mark.pruned
def test_condition_data_config_resolve_called():
    called = {"x": False}

    def fake_resolve(self):
        called["x"] = True

    with patch.multiple(
        "cccma_ppp.data_modules.data.data_abc.DataConfigABC",
        _resolve_data=fake_resolve,
        _get_ds_info=lambda self: DummyInfo(),
    ):
        ConditionDataConfig(
            paths="x",
            names=["var"],
        )

    assert called["x"] is True


@pytest.mark.pruned
def test_condition_data_config_get_info_called():
    called = {"x": False}

    def fake_info(self):
        called["x"] = True
        return DummyInfo()

    with patch.multiple(
        "cccma_ppp.data_modules.data.data_abc.DataConfigABC",
        _resolve_data=lambda self: None,
        _get_ds_info=fake_info,
    ):
        ConditionDataConfig(
            paths="x",
            names=["var"],
        )

    assert called["x"] is True