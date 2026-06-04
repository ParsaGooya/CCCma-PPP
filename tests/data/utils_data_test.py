import pytest
import numpy as np
import xarray as xr
import os
import glob
from pathlib import Path
import importlib

from src.cccma_ppp.data import utils_data as mod


# ============================================================
# GLOBAL PATCH FOR SAFE TESTS
# ============================================================


@pytest.fixture(autouse=True)
def patch_io(monkeypatch):

    def fake_check_data(cfg):
        cfg.list_paths = ["fake.nc"]

    def fake_load(paths, **kwargs):
        return xr.Dataset(
            {
                "var": xr.DataArray(
                    np.random.rand(3, 12, 2, 2),
                    dims=("year", "lead_time", "lat", "lon"),
                    coords={
                        "year": [2000, 2001, 2002],
                        "lead_time": np.arange(12),
                        "lat": [0, 1],
                        "lon": [0, 1],
                    },
                )
            }
        )

    monkeypatch.setattr(mod, "_check_data", fake_check_data)
    monkeypatch.setattr(mod, "_load_xarray_data", fake_load)


# ============================================================
# HELPERS
# ============================================================


def coords():
    return {
        "lat": xr.DataArray([0, 1], dims=("lat",)),
        "lon": xr.DataArray([0, 1], dims=("lon",)),
    }


# ============================================================
# CONFIG CLASSES
# ============================================================


def test_model_obs_condition_configs():
    m = mod.ModelDataConfig("fake", ["var"])
    o = mod.ObsDataConfig("fake", ["var"])
    c = mod.ConditionDataConfig("fake", ["var"])

    assert m.info is not None
    assert o.year_range[-1] == 2002
    assert hasattr(c, "info")

    m2 = mod.ModelDataConfig("fake", ["var"], ensemble_list=[0])
    assert m2._check_ensemble is True


# ============================================================
# _get_ds_info
# ============================================================


def test_get_ds_info_all_branches():
    class Dummy:
        paths = "fake"
        names = ["var"]
        concat_dim = "year"
        rename_dict = {"var": "var"}
        ensemble_list = [0]
        file_type = "*.nc"

    info = mod._get_ds_info(Dummy())
    assert info.start_year == 2000


def test_get_ds_info_no_year(monkeypatch):

    def fake_load(*a, **k):
        return xr.Dataset({"var": xr.DataArray(np.ones((2, 2)), dims=("lat", "lon"))})

    monkeypatch.setattr(mod, "_load_xarray_data", fake_load)

    class Dummy:
        paths = "fake"
        names = ["var"]
        concat_dim = "year"
        rename_dict = None
        ensemble_list = None
        file_type = "*.nc"

    info = mod._get_ds_info(Dummy())
    assert info.start_year is None
    assert info.sizes is None or isinstance(info.sizes, dict)


# ============================================================
# TRAIN MASK
# ============================================================


def test_create_train_mask():
    m1 = mod._create_train_mask([1, 2], 12)
    m2 = mod._create_train_mask([1], np.arange(1, 13))

    assert m1.shape == (2, 12)
    assert m2.shape == (1, 12)


# ============================================================
# UNWRAP
# ============================================================


def test_unwrap_all():
    ds_multi = xr.Dataset(
        {"a": xr.DataArray(np.ones((2, 2))), "b": xr.DataArray(np.zeros((2, 2)))}
    )
    ds_single = xr.Dataset({"a": xr.DataArray(np.ones((2, 2)))})

    assert mod._unwrape_data_variables(ds_multi).shape[0] == 2
    assert mod._unwrape_data_variables(ds_single).shape[0] == 1

    bad = xr.Dataset(
        {"a": xr.DataArray(np.ones((2, 2, 2)), dims=("channels", "x", "y"))}
    )
    with pytest.raises(ValueError):
        mod._unwrape_data_variables(bad)


# ============================================================
# WEIGHTS — ALL BRANCHES
# ============================================================


def test_weights_standard_branches():
    w1 = mod.WeightsConfig("uniform").build_weights(coords(), save=False)
    w2 = mod.WeightsConfig("cosine_lat").build_weights(coords(), save=False)

    assert np.all(w1 == 1)
    assert w2.shape == (2, 2)

    w3 = mod.WeightsConfig(variable_weights={"a": 1, "b": 2}).build_weights(
        coords(), save=False
    )
    assert "channels" in w3.dims


def test_weights_ocean_branch():
    class Dummy:
        def transform(self, w):
            return w * 0

    w = mod.WeightsConfig().build_weights(coords(), oceannanremover=Dummy(), save=False)
    assert np.all(w == 0)


def test_weights_invalid():
    with pytest.raises(AssertionError):
        mod.WeightsConfig("bad")


def test_weights_save(monkeypatch, tmp_path):

    monkeypatch.setattr(xr.DataArray, "reset_index", lambda self, *a, **k: self)
    monkeypatch.setattr(xr.DataArray, "to_netcdf", lambda self, path: None)

    os.environ["GLOBAL_EXP_DIR"] = str(tmp_path)

    cfg = mod.WeightsConfig()
    cfg.build_weights(coords(), save=True)

    newdir = tmp_path / "subdir"
    cfg.build_weights(coords(), save=True, save_path=newdir)

    assert newdir.exists()


def test_weights_load_dir(monkeypatch):

    def fake_open(path):
        da = xr.DataArray(
            np.ones((2, 2)),
            dims=("lat", "lon"),
            coords={"lat": [0, 1], "lon": [0, 1]},
        )
        return xr.Dataset({"a": da})

    monkeypatch.setattr(xr, "open_dataset", fake_open)

    # IMPORTANT: bypass unwrap to keep coords consistent
    monkeypatch.setattr(
        mod,
        "_unwrape_data_variables",
        lambda ds: list(ds.data_vars.values())[0],
    )

    w = mod.WeightsConfig(load_dir="fake").build_weights(coords(), save=False)
    assert w is not None


def test_weights_load_assert(monkeypatch):

    def fake_open(path):
        da = xr.DataArray(
            np.ones((2, 2)),
            dims=("lat", "lon"),
            coords={"lat": [9, 9], "lon": [9, 9]},  # mismatch
        )
        return xr.Dataset({"a": da})

    monkeypatch.setattr(xr, "open_dataset", fake_open)
    monkeypatch.setattr(
        mod,
        "_unwrape_data_variables",
        lambda ds: list(ds.data_vars.values())[0],
    )

    with pytest.raises(AssertionError):
        mod.WeightsConfig(load_dir="fake").build_weights(coords(), save=False)


# ============================================================
# _load_xarray_data — FULL BRANCHES
# ============================================================


def test_load_xarray_data(monkeypatch):

    def fake_open_mfdataset(paths, combine, concat_dim):
        return xr.Dataset(
            {
                "a": xr.DataArray(
                    np.ones((2, 2, 2)),
                    dims=("ensembles", "lat", "lon"),
                )
            }
        )

    monkeypatch.setattr(xr, "open_mfdataset", fake_open_mfdataset)

    class DummyPre:
        def transform(self, ds):
            return ds * 2

    ds = mod._load_xarray_data(
        ["fake"],
        selection={"lat": 0},
        names=["a"],
        ensemble_mean=True,
        preprocessor=DummyPre(),
        rename_dict=None,
    )

    assert "a" in ds


# ============================================================
# _check_data — REAL BRANCH COVERAGE
# ============================================================


def test_check_data_real(monkeypatch):

    # force real _check_data
    monkeypatch.setattr(mod, "_check_data", mod._check_data)

    # ---- missing path branch ----
    monkeypatch.setattr(Path, "exists", lambda self: False)

    class Dummy:
        paths = "bad"
        file_type = "*.nc"
        names = ["var"]
        rename_dict = None
        TYPE = "model"
        ensemble_list = None
        _check_ensemble = False

        @staticmethod
        def _required_dims():
            return ["lat"]

        @staticmethod
        def _allowed_dims():
            return ["lat"]

    with pytest.raises(FileNotFoundError):
        mod._check_data(Dummy())

    # ---- empty glob branch ----
    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(glob, "glob", lambda x: [])

    with pytest.raises(FileNotFoundError):
        mod._check_data(Dummy())
