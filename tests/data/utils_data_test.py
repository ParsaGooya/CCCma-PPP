import glob
from pathlib import Path
import pandas as pd
import numpy as np
import pytest


def test_condition_config_with_ensemble_list_sets_check_ensemble(monkeypatch):
    def fake_load(paths, **kwargs):
        return xr.Dataset(
            {
                "var": xr.DataArray(
                    np.ones((3, 12, 2, 2, 2)),
                    dims=("year", "lead_time", "ensembles", "lat", "lon"),
                    coords={
                        "year": [2000, 2001, 2002],
                        "lead_time": np.arange(12),
                        "ensembles": [0, 1],
                        "lat": [0, 1],
                        "lon": [0, 1],
                    },
                )
            }
        )

    monkeypatch.setattr(mod, "_load_xarray_data", fake_load)

    cfg = mod.ConditionDataConfig("fake", ["var"], ensemble_list=[0])

    assert cfg._check_ensemble is True
    assert "ensembles" in cfg.info.coords


def test_condition_config_no_year_range_when_no_year(monkeypatch):
    def fake_load(paths, **kwargs):
        return condition_ds_no_year()

    monkeypatch.setattr(mod, "_load_xarray_data", fake_load)

    cfg = mod.ConditionDataConfig("fake", ["var"])

    assert not hasattr(cfg, "year_range")


def test_get_ds_info_uses_existing_list_paths():
    class Dummy:
        paths = "fake"
        list_paths = ["already_set.nc"]
        names = ["var"]
        concat_dim = "year"
        rename_dict = None
        ensemble_list = None
        file_type = "*.nc"

    info = mod._get_ds_info(Dummy())

    assert info.start_year == 2000
    assert info.final_year == 2002
    assert info.sizes["lead_time"] == 12


def test_get_ds_info_without_list_paths_uses_glob(monkeypatch):
    monkeypatch.setattr(glob, "glob", lambda pattern: ["from_glob.nc"])

    class Dummy:
        paths = "fake"
        names = ["var"]
        concat_dim = "year"
        rename_dict = None
        ensemble_list = None
        file_type = "*.nc"

    info = mod._get_ds_info(Dummy())

    assert info.start_year == 2000


def test_get_ds_info_with_ensemble_selection(monkeypatch):
    def fake_load(paths, **kwargs):
        return model_ds()

    monkeypatch.setattr(mod, "_load_xarray_data", fake_load)

    class Dummy:
        paths = "fake"
        list_paths = ["fake.nc"]
        names = ["var"]
        concat_dim = "year"
        rename_dict = None
        ensemble_list = [0]
        file_type = "*.nc"

    info = mod._get_ds_info(Dummy())

    assert "ensembles" in info.coords
    assert info.coords["ensembles"] is not None


def test_get_ds_info_no_year_no_sizes(monkeypatch):
    def fake_load(paths, **kwargs):
        return condition_ds_no_year()

    monkeypatch.setattr(mod, "_load_xarray_data", fake_load)

    class Dummy:
        paths = "fake"
        list_paths = ["fake.nc"]
        names = ["var"]
        concat_dim = "year"
        rename_dict = None
        ensemble_list = None
        file_type = "*.nc"

    info = mod._get_ds_info(Dummy())

    assert info.start_year is None
    assert info.final_year is None
    assert info.sizes is None


def test_create_train_mask_with_int_lead_times():
    mask = mod._create_train_mask([2000, 2001], 12)

    assert mask.shape == (2, 12)
    assert mask.dims == ("year", "lead_time")


def test_create_train_mask_with_array_lead_times():
    mask = mod._create_train_mask([2000], np.arange(1, 13))

    assert mask.shape == (1, 12)


def test_create_train_mask_with_exclude_idx():
    mask = mod._create_train_mask([2000, 2001, 2002], 24, exclude_idx=1)

    assert mask.shape == (3, 24)
    assert mask.dtype == bool


def test_unwrap_single_variable_dataset():
    ds = xr.Dataset(
        {
            "a": xr.DataArray(
                np.ones((2, 2)),
                dims=("lat", "lon"),
            )
        }
    )

    out = mod._unwrape_data_variables(ds)

    assert "channels" in out.dims
    assert out.shape[0] == 1


def test_unwrap_multiple_variable_dataset():
    ds = xr.Dataset(
        {
            "a": xr.DataArray(np.ones((2, 2)), dims=("lat", "lon")),
            "b": xr.DataArray(np.zeros((2, 2)), dims=("lat", "lon")),
        }
    )

    out = mod._unwrape_data_variables(ds)

    assert "channels" in out.dims
    assert out.shape[0] == 2


def test_weights_uniform():
    weights = mod.WeightsConfig("uniform").build_weights(coords(), save=False)

    assert np.all(weights == 1)


def test_weights_cosine_lat():
    weights = mod.WeightsConfig("cosine_lat").build_weights(coords(), save=False)

    assert weights.shape == (2, 2)


def test_weights_invalid_spatial_method():
    with pytest.raises(AssertionError):
        mod.WeightsConfig("bad")


def test_weights_variable_weights():
    weights = mod.WeightsConfig(variable_weights={"a": 1.0, "b": 2.0}).build_weights(
        coords(), save=False
    )

    assert "channels" in weights.dims
    assert list(weights.coords["channels"].values) == ["a", "b"]


def test_weights_with_oceannanremover_transform():
    class DummyOcean:
        def transform(self, weights):
            return weights * 0

    weights = mod.WeightsConfig().build_weights(
        coords(),
        oceannanremover=DummyOcean(),
        save=False,
    )

    assert np.all(weights == 0)


def test_weights_save_default_path(monkeypatch, tmp_path):
    monkeypatch.setenv("GLOBAL_EXP_DIR", str(tmp_path))
    monkeypatch.setattr(xr.DataArray, "reset_index", lambda self, *a, **k: self)
    monkeypatch.setattr(xr.DataArray, "to_netcdf", lambda self, path: None)

    mod.WeightsConfig().build_weights(coords(), save=True)

    assert tmp_path.exists()


def test_weights_save_custom_path_and_name(monkeypatch, tmp_path):
    monkeypatch.setattr(xr.DataArray, "reset_index", lambda self, *a, **k: self)
    monkeypatch.setattr(xr.DataArray, "to_netcdf", lambda self, path: None)

    save_path = tmp_path / "weights_dir"

    mod.WeightsConfig().build_weights(
        coords(),
        save=True,
        save_path=save_path,
        save_name="custom.nc",
    )

    assert save_path.exists()


def test_weights_load_dir_dataset(monkeypatch):
    def fake_open_dataset(path):
        return xr.Dataset(
            {
                "a": xr.DataArray(
                    np.ones((2, 2)),
                    dims=("lat", "lon"),
                    coords={"lat": [0, 1], "lon": [0, 1]},
                )
            }
        )

    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)

    weights = mod.WeightsConfig(load_dir="fake.nc").build_weights(
        coords(),
        save=False,
    )

    assert weights is not None


def test_weights_load_dir_lat_lon_mismatch(monkeypatch):
    def fake_open_dataset(path):
        return xr.Dataset(
            {
                "a": xr.DataArray(
                    np.ones((2, 2)),
                    dims=("lat", "lon"),
                    coords={"lat": [9, 10], "lon": [9, 10]},
                )
            }
        )

    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)

    with pytest.raises(AssertionError):
        mod.WeightsConfig(load_dir="fake.nc").build_weights(coords(), save=False)


def test_weights_load_dir_with_ref_and_oceannanremover(monkeypatch):
    ref = pd.MultiIndex.from_product([[0, 1], [0, 1]], names=("lat", "lon"))

    da = xr.DataArray(
        np.ones((4,)),
        dims=("ref",),
        coords={"ref": ref},
    )

    def fake_open_dataset(path):
        return xr.Dataset({"a": da})

    class DummyOcean:
        final_locations = da.coords["ref"]

    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)
    monkeypatch.setattr(
        mod,
        "_unwrape_data_variables",
        lambda ds: list(ds.data_vars.values())[0],
    )

    weights = mod.WeightsConfig(load_dir="fake.nc").build_weights(
        target_coords=None,
        oceannanremover=DummyOcean(),
        save=False,
    )

    assert "ref" in weights.dims


def test_weights_load_dir_ref_ocean_mismatch(monkeypatch):
    ref = pd.MultiIndex.from_product([[0, 1], [0, 1]], names=("lat", "lon"))
    other_ref = pd.MultiIndex.from_product([[9, 10], [9, 10]], names=("lat", "lon"))

    da = xr.DataArray(
        np.ones((4,)),
        dims=("ref",),
        coords={"ref": ref},
    )

    def fake_open_dataset(path):
        return xr.Dataset({"a": da})

    class DummyOcean:
        final_locations = xr.DataArray(
            np.arange(4),
            dims=("ref",),
            coords={"ref": other_ref},
        ).coords["ref"]

    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)
    monkeypatch.setattr(
        mod,
        "_unwrape_data_variables",
        lambda ds: list(ds.data_vars.values())[0],
    )

    with pytest.raises(AssertionError):
        mod.WeightsConfig(load_dir="fake.nc").build_weights(
            target_coords=None,
            oceannanremover=DummyOcean(),
            save=False,
        )


def test_load_xarray_data_basic(monkeypatch):
    monkeypatch.setattr(mod, "_load_xarray_data", ORIGINAL_LOAD_XARRAY_DATA)

    def fake_open_mfdataset(paths, *args, **kwargs):
        return xr.Dataset(
            {
                "a": xr.DataArray(
                    np.ones((2, 2)),
                    dims=("lat", "lon"),
                    coords={"lat": [0, 1], "lon": [0, 1]},
                )
            }
        )

    monkeypatch.setattr(xr, "open_mfdataset", fake_open_mfdataset)

    ds = mod._load_xarray_data(["fake.nc"], names=["a"])

    assert "a" in ds


def test_load_xarray_data_with_rename(monkeypatch):
    monkeypatch.setattr(mod, "_load_xarray_data", ORIGINAL_LOAD_XARRAY_DATA)

    def fake_open_mfdataset(paths, *args, **kwargs):
        return xr.Dataset(
            {
                "old": xr.DataArray(
                    np.ones((2, 2)),
                    dims=("lat", "lon"),
                    coords={"lat": [0, 1], "lon": [0, 1]},
                )
            }
        )

    monkeypatch.setattr(xr, "open_mfdataset", fake_open_mfdataset)

    ds = mod._load_xarray_data(
        ["fake.nc"],
        names=["new"],
        rename_dict={"old": "new"},
    )

    assert "new" in ds


def test_load_xarray_data_with_selection(monkeypatch):
    monkeypatch.setattr(mod, "_load_xarray_data", ORIGINAL_LOAD_XARRAY_DATA)

    def fake_open_mfdataset(paths, *args, **kwargs):
        return xr.Dataset(
            {
                "a": xr.DataArray(
                    np.ones((2, 2)),
                    dims=("lat", "lon"),
                    coords={"lat": [0, 1], "lon": [0, 1]},
                )
            }
        )

    monkeypatch.setattr(xr, "open_mfdataset", fake_open_mfdataset)

    ds = mod._load_xarray_data(
        ["fake.nc"],
        selection={"lat": 0},
        names=["a"],
    )

    assert ds["a"].shape == (2,)


def test_load_xarray_data_ensemble_mean(monkeypatch):
    monkeypatch.setattr(mod, "_load_xarray_data", ORIGINAL_LOAD_XARRAY_DATA)

    def fake_open_mfdataset(paths, *args, **kwargs):
        return xr.Dataset(
            {
                "a": xr.DataArray(
                    np.ones((2, 2, 2)),
                    dims=("ensembles", "lat", "lon"),
                    coords={
                        "ensembles": [0, 1],
                        "lat": [0, 1],
                        "lon": [0, 1],
                    },
                )
            }
        )

    monkeypatch.setattr(xr, "open_mfdataset", fake_open_mfdataset)

    ds = mod._load_xarray_data(
        ["fake.nc"],
        names=["a"],
        ensemble_mean=True,
    )

    assert "ensembles" not in ds.dims


def test_load_xarray_data_keep_ensembles(monkeypatch):
    monkeypatch.setattr(mod, "_load_xarray_data", ORIGINAL_LOAD_XARRAY_DATA)

    def fake_open_mfdataset(paths, *args, **kwargs):
        return xr.Dataset(
            {
                "a": xr.DataArray(
                    np.ones((2, 2, 2)),
                    dims=("ensembles", "lat", "lon"),
                    coords={
                        "ensembles": [0, 1],
                        "lat": [0, 1],
                        "lon": [0, 1],
                    },
                )
            }
        )

    monkeypatch.setattr(xr, "open_mfdataset", fake_open_mfdataset)

    ds = mod._load_xarray_data(
        ["fake.nc"],
        names=["a"],
        ensemble_mean=False,
    )

    assert "ensembles" in ds.dims


def test_load_xarray_data_with_preprocessor(monkeypatch):
    monkeypatch.setattr(mod, "_load_xarray_data", ORIGINAL_LOAD_XARRAY_DATA)

    def fake_open_mfdataset(paths, *args, **kwargs):
        return xr.Dataset(
            {
                "a": xr.DataArray(
                    np.ones((2, 2)),
                    dims=("lat", "lon"),
                    coords={"lat": [0, 1], "lon": [0, 1]},
                )
            }
        )

    class DummyPreprocessor:
        def transform(self, ds):
            return ds * 2

    monkeypatch.setattr(xr, "open_mfdataset", fake_open_mfdataset)

    ds = mod._load_xarray_data(
        ["fake.nc"],
        names=["a"],
        preprocessor=DummyPreprocessor(),
    )

    assert float(ds["a"].max()) == 2.0


def test_load_xarray_data_without_names(monkeypatch):
    monkeypatch.setattr(mod, "_load_xarray_data", ORIGINAL_LOAD_XARRAY_DATA)

    def fake_open_mfdataset(paths, *args, **kwargs):
        return xr.Dataset(
            {
                "a": xr.DataArray(
                    np.ones((2, 2)),
                    dims=("lat", "lon"),
                    coords={"lat": [0, 1], "lon": [0, 1]},
                ),
                "b": xr.DataArray(
                    np.ones((2, 2)),
                    dims=("lat", "lon"),
                    coords={"lat": [0, 1], "lon": [0, 1]},
                ),
            }
        )

    monkeypatch.setattr(xr, "open_mfdataset", fake_open_mfdataset)

    ds = mod._load_xarray_data(["fake.nc"], names=None)

    assert set(ds.data_vars) == {"a", "b"}


class CheckDummy:
    paths = "fake_dir"
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
        return ["lat", "lon", "ensembles"]


def test_check_data_missing_path(monkeypatch):
    monkeypatch.setattr(mod, "_check_data", ORIGINAL_CHECK_DATA)
    monkeypatch.setattr(Path, "exists", lambda self: False)

    with pytest.raises(FileNotFoundError):
        mod._check_data(CheckDummy())


def test_check_data_no_matching_files(monkeypatch):
    monkeypatch.setattr(mod, "_check_data", ORIGINAL_CHECK_DATA)
    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(glob, "glob", lambda pattern: [])

    with pytest.raises(FileNotFoundError):
        mod._check_data(CheckDummy())


def test_check_data_success(monkeypatch):
    monkeypatch.setattr(mod, "_check_data", ORIGINAL_CHECK_DATA)
    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(glob, "glob", lambda pattern: ["fake.nc"])

    def fake_open_dataset(path):
        return xr.Dataset(
            {
                "var": xr.DataArray(
                    np.ones((2, 2)),
                    dims=("lat", "lon"),
                    coords={"lat": [0, 1], "lon": [0, 1]},
                )
            }
        )

    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)

    cfg = CheckDummy()
    mod._check_data(cfg)

    assert cfg.list_paths == ["fake.nc"]


def test_check_data_rename_dict(monkeypatch):
    monkeypatch.setattr(mod, "_check_data", ORIGINAL_CHECK_DATA)
    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(glob, "glob", lambda pattern: ["fake.nc"])

    def fake_open_dataset(path):
        return xr.Dataset(
            {
                "old": xr.DataArray(
                    np.ones((2,)),
                    dims=("lat",),
                    coords={"lat": [0, 1]},
                )
            }
        )

    class Dummy(CheckDummy):
        names = ["new"]
        rename_dict = {"old": "new"}

    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)

    cfg = Dummy()
    mod._check_data(cfg)

    assert cfg.list_paths == ["fake.nc"]


def test_check_data_missing_required_dim(monkeypatch):
    monkeypatch.setattr(mod, "_check_data", ORIGINAL_CHECK_DATA)
    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(glob, "glob", lambda pattern: ["fake.nc"])

    def fake_open_dataset(path):
        return xr.Dataset(
            {
                "var": xr.DataArray(
                    np.ones((2,)),
                    dims=("lon",),
                    coords={"lon": [0, 1]},
                )
            }
        )

    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)

    with pytest.raises(AssertionError):
        mod._check_data(CheckDummy())


def test_check_data_missing_ensembles_when_requested(monkeypatch):
    monkeypatch.setattr(mod, "_check_data", ORIGINAL_CHECK_DATA)
    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(glob, "glob", lambda pattern: ["fake.nc"])

    def fake_open_dataset(path):
        return xr.Dataset(
            {
                "var": xr.DataArray(
                    np.ones((2,)),
                    dims=("lat",),
                    coords={"lat": [0, 1]},
                )
            }
        )

    class Dummy(CheckDummy):
        _check_ensemble = True

    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)

    with pytest.raises(AssertionError):
        mod._check_data(Dummy())


def test_check_data_invalid_dim(monkeypatch):
    monkeypatch.setattr(mod, "_check_data", ORIGINAL_CHECK_DATA)
    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(glob, "glob", lambda pattern: ["fake.nc"])

    def fake_open_dataset(path):
        return xr.Dataset(
            {
                "var": xr.DataArray(
                    np.ones((2, 2)),
                    dims=("lat", "bad_dim"),
                    coords={"lat": [0, 1], "bad_dim": [0, 1]},
                )
            }
        )

    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)

    with pytest.raises(AssertionError):
        mod._check_data(CheckDummy())


def test_check_data_missing_coord(monkeypatch):
    monkeypatch.setattr(mod, "_check_data", ORIGINAL_CHECK_DATA)
    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(glob, "glob", lambda pattern: ["fake.nc"])

    def fake_open_dataset(path):
        da = xr.DataArray(
            np.ones((2,)),
            dims=("lat",),
        )
        return xr.Dataset({"var": da})

    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)

    with pytest.raises(AssertionError):
        mod._check_data(CheckDummy())


def test_check_data_missing_variable(monkeypatch):
    monkeypatch.setattr(mod, "_check_data", ORIGINAL_CHECK_DATA)
    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(glob, "glob", lambda pattern: ["fake.nc"])

    def fake_open_dataset(path):
        return xr.Dataset(
            {
                "other": xr.DataArray(
                    np.ones((2,)),
                    dims=("lat",),
                    coords={"lat": [0, 1]},
                )
            }
        )

    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)

    with pytest.raises(ValueError):
        mod._check_data(CheckDummy())


import xarray as xr

from cccma_ppp.data import utils_data as mod


ORIGINAL_CHECK_DATA = mod._check_data
ORIGINAL_LOAD_XARRAY_DATA = mod._load_xarray_data


@pytest.fixture(autouse=True)
def patch_safe_io(monkeypatch):
    def fake_check_data(cfg):
        cfg.list_paths = ["fake.nc"]

    def fake_load_xarray_data(paths, **kwargs):
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
    monkeypatch.setattr(mod, "_load_xarray_data", fake_load_xarray_data)


def coords():
    return {
        "lat": xr.DataArray(
            [0, 1],
            dims=("lat",),
            coords={"lat": [0, 1]},
        ),
        "lon": xr.DataArray(
            [0, 1],
            dims=("lon",),
            coords={"lon": [0, 1]},
        ),
    }


def model_ds():
    return xr.Dataset(
        {
            "var": xr.DataArray(
                np.ones((2, 12, 2, 2, 2)),
                dims=("year", "lead_time", "ensembles", "lat", "lon"),
                coords={
                    "year": [2000, 2001],
                    "lead_time": np.arange(12),
                    "ensembles": [0, 1],
                    "lat": [0, 1],
                    "lon": [0, 1],
                },
            )
        }
    )


def obs_ds():
    return xr.Dataset(
        {
            "var": xr.DataArray(
                np.ones((2, 12, 2, 2)),
                dims=("year", "month", "lat", "lon"),
                coords={
                    "year": [2000, 2001],
                    "month": np.arange(12),
                    "lat": [0, 1],
                    "lon": [0, 1],
                },
            )
        }
    )


def condition_ds_no_year():
    return xr.Dataset(
        {
            "var": xr.DataArray(
                np.ones((2, 2)),
                dims=("lat", "lon"),
                coords={"lat": [0, 1], "lon": [0, 1]},
            )
        }
    )
