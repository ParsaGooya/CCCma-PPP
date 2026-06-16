import glob
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
import xarray as xr

from cccma_ppp.data import utils_data as mod


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

    out = mod._unwrap_data_variables(ds)

    assert "channels" in out.dims
    assert out.shape[0] == 1


def test_unwrap_multiple_variable_dataset():
    ds = xr.Dataset(
        {
            "a": xr.DataArray(np.ones((2, 2)), dims=("lat", "lon")),
            "b": xr.DataArray(np.zeros((2, 2)), dims=("lat", "lon")),
        }
    )

    out = mod._unwrap_data_variables(ds)

    assert "channels" in out.dims
    assert out.shape[0] == 2


def test_weights_uniform():
    weights = mod.WeightsConfig("uniform").build_weights(coords(), save=False)

    assert np.all(weights == 1)


def test_weights_cosine_lat():
    weights = mod.WeightsConfig("cosine_lat").build_weights(coords(), save=False)

    assert weights.shape == (2, 2)


def test_weights_invalid_spatial_method():
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
    monkeypatch.setattr(Path, "exists", lambda self: True)

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
    monkeypatch.setattr(Path, "exists", lambda self: True)

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

    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        mod.WeightsConfig(load_dir="fake.nc").build_weights(coords(), save=False)


def test_weights_load_dir_with_ref_and_oceannanremover(monkeypatch):
    monkeypatch.setattr(Path, "exists", lambda self: True)
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
        "_unwrap_data_variables",
        lambda ds: list(ds.data_vars.values())[0],
    )

    weights = mod.WeightsConfig(load_dir="fake.nc").build_weights(
        target_coords=None,
        oceannanremover=DummyOcean(),
        save=False,
    )

    assert "ref" in weights.dims


def test_weights_load_dir_ref_ocean_mismatch(monkeypatch):
    monkeypatch.setattr(Path, "exists", lambda self: True)
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
        "_unwrap_data_variables",
        lambda ds: list(ds.data_vars.values())[0],
    )

    with pytest.raises((AssertionError, ValueError, RuntimeError)):
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

    with pytest.raises((AssertionError, ValueError, RuntimeError)):
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

    with pytest.raises((AssertionError, ValueError, RuntimeError)):
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

    with pytest.raises((AssertionError, ValueError, RuntimeError)):
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

    with pytest.raises((AssertionError, ValueError, RuntimeError)):
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


def test_weights_default_method_builds():
    weights = mod.WeightsConfig().build_weights(coords(), save=False)

    assert weights is not None
    assert weights.shape == (2, 2)


def test_weights_variable_weights_single_channel():
    weights = mod.WeightsConfig(variable_weights={"a": 5.0}).build_weights(
        coords(),
        save=False,
    )

    assert "channels" in weights.dims
    assert weights.shape[0] == 1


def test_weights_save_creates_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(xr.DataArray, "reset_index", lambda self, *a, **k: self)
    monkeypatch.setattr(xr.DataArray, "to_netcdf", lambda self, path: None)

    out_dir = tmp_path / "saved"

    mod.WeightsConfig().build_weights(
        coords(),
        save=True,
        save_path=out_dir,
    )

    assert out_dir.exists()


def test_weights_load_dir_path_missing():
    with pytest.raises(FileNotFoundError):
        mod.WeightsConfig(load_dir="missing.nc").build_weights(
            coords(),
            save=False,
        )


def test_load_xarray_data_selection_multiple_dims(monkeypatch):
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
        names=["a"],
        selection={"lat": 0, "lon": 1},
    )

    assert ds["a"].shape == ()


def test_load_xarray_data_preprocessor_changes_values(monkeypatch):
    monkeypatch.setattr(mod, "_load_xarray_data", ORIGINAL_LOAD_XARRAY_DATA)

    class DummyPreprocessor:
        def transform(self, ds):
            ds["a"] = ds["a"] + 10
            return ds

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
        names=["a"],
        preprocessor=DummyPreprocessor(),
    )

    assert float(ds["a"].mean()) == 11.0


def test_load_xarray_data_rename_and_selection(monkeypatch):
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
        selection={"lat": 0},
    )

    assert "new" in ds
    assert ds["new"].shape == (2,)


def test_load_xarray_data_ensemble_mean_reduces_dimension(monkeypatch):
    monkeypatch.setattr(mod, "_load_xarray_data", ORIGINAL_LOAD_XARRAY_DATA)

    def fake_open_mfdataset(paths, *args, **kwargs):
        return xr.Dataset(
            {
                "a": xr.DataArray(
                    np.arange(8).reshape(2, 2, 2),
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

    assert ds["a"].shape == (2, 2)


def test_get_ds_info_with_no_ensemble_selection(monkeypatch):
    def fake_load(paths, **kwargs):
        return model_ds()

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

    assert info.start_year == 2000
    assert info.final_year == 2001


def test_get_ds_info_uses_concat_dim(monkeypatch):
    def fake_load(paths, **kwargs):
        return obs_ds()

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

    assert info.sizes["month"] == 12


def test_condition_config_info_has_sizes(monkeypatch):
    def fake_load(paths, **kwargs):
        return model_ds()

    monkeypatch.setattr(mod, "_load_xarray_data", fake_load)

    cfg = mod.ConditionDataConfig("fake", ["var"])

    assert cfg.info.sizes["lead_time"] == 12


def test_check_data_sets_list_paths(monkeypatch):
    monkeypatch.setattr(mod, "_check_data", ORIGINAL_CHECK_DATA)
    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(glob, "glob", lambda pattern: ["a.nc", "b.nc"])

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

    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)

    cfg = CheckDummy()

    mod._check_data(cfg)

    assert len(cfg.list_paths) == 2


def test_check_data_accepts_ensemble_dimension(monkeypatch):
    monkeypatch.setattr(mod, "_check_data", ORIGINAL_CHECK_DATA)
    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(glob, "glob", lambda pattern: ["fake.nc"])

    def fake_open_dataset(path):
        return xr.Dataset(
            {
                "var": xr.DataArray(
                    np.ones((2, 2, 2)),
                    dims=("lat", "lon", "ensembles"),
                    coords={
                        "lat": [0, 1],
                        "lon": [0, 1],
                        "ensembles": [0, 1],
                    },
                )
            }
        )

    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)

    cfg = CheckDummy()

    mod._check_data(cfg)

    assert cfg.list_paths == ["fake.nc"]


def test_check_data_with_rename_dict_variable_present(monkeypatch):
    monkeypatch.setattr(mod, "_check_data", ORIGINAL_CHECK_DATA)
    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(glob, "glob", lambda pattern: ["fake.nc"])

    def fake_open_dataset(path):
        return xr.Dataset(
            {
                "old_name": xr.DataArray(
                    np.ones((2,)),
                    dims=("lat",),
                    coords={"lat": [0, 1]},
                )
            }
        )

    class Dummy(CheckDummy):
        names = ["new_name"]
        rename_dict = {"old_name": "new_name"}

    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)

    cfg = Dummy()

    mod._check_data(cfg)

    assert cfg.list_paths == ["fake.nc"]


def test_load_xarray_data_returns_dataset(monkeypatch):
    monkeypatch.setattr(mod, "_load_xarray_data", ORIGINAL_LOAD_XARRAY_DATA)

    def fake_open_mfdataset(paths, *args, **kwargs):
        return xr.Dataset(
            {
                "x": xr.DataArray(
                    np.ones((2, 2)),
                    dims=("lat", "lon"),
                    coords={"lat": [0, 1], "lon": [0, 1]},
                )
            }
        )

    monkeypatch.setattr(xr, "open_mfdataset", fake_open_mfdataset)

    ds = mod._load_xarray_data(["f.nc"])

    assert isinstance(ds, xr.Dataset)


def test_weights_cosine_lat_values_not_nan():
    weights = mod.WeightsConfig("cosine_lat").build_weights(
        coords(),
        save=False,
    )

    assert not np.isnan(weights.values).any()


def test_weights_uniform_values_equal_one():
    weights = mod.WeightsConfig("uniform").build_weights(
        coords(),
        save=False,
    )

    assert float(weights.min()) == 1.0
    assert float(weights.max()) == 1.0
    mask = mod._create_train_mask([2000, 2001], 12, exclude_idx=-1)

    assert mask.shape == (2, 12)
    assert mask.dtype == bool


def test_get_ds_info_sizes_contains_lead_time(monkeypatch):
    def fake_load(paths, **kwargs):
        return model_ds()

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

    assert info.sizes is not None
    assert "lead_time" in info.sizes
    assert info.sizes["lead_time"] == 12


def test_weights_variable_weights_values():
    weights = mod.WeightsConfig(variable_weights={"x": 2.0, "y": 3.0}).build_weights(
        coords(),
        save=False,
    )

    assert float(weights.sel(channels="x").mean()) == 2.0
    assert float(weights.sel(channels="y").mean()) == 3.0


def test_weights_with_oceannanremover_preserves_shape():
    class DummyOcean:
        def transform(self, weights):
            return weights

    weights = mod.WeightsConfig().build_weights(
        coords(),
        oceannanremover=DummyOcean(),
        save=False,
    )

    assert weights.shape == (2, 2)


def test_load_xarray_data_selection_scalar(monkeypatch):
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
        names=["a"],
        selection={"lat": 0, "lon": 0},
    )

    assert ds["a"].shape == ()


def test_load_xarray_data_preprocessor_identity(monkeypatch):
    monkeypatch.setattr(mod, "_load_xarray_data", ORIGINAL_LOAD_XARRAY_DATA)

    class Identity:
        def transform(self, ds):
            return ds

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
        names=["a"],
        preprocessor=Identity(),
    )

    assert float(ds["a"].mean()) == 1.0


def test_load_xarray_data_without_ensemble_mean_keeps_dim(monkeypatch):
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

    assert ds["a"].sizes["ensembles"] == 2


def test_condition_config_info_contains_coords(monkeypatch):
    def fake_load(paths, **kwargs):
        return model_ds()

    monkeypatch.setattr(mod, "_load_xarray_data", fake_load)

    cfg = mod.ConditionDataConfig("fake", ["var"])

    assert "lat" in cfg.info.coords
    assert "lon" in cfg.info.coords


def test_check_data_accepts_allowed_dims(monkeypatch):
    monkeypatch.setattr(mod, "_check_data", ORIGINAL_CHECK_DATA)
    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(glob, "glob", lambda pattern: ["fake.nc"])

    def fake_open_dataset(path):
        return xr.Dataset(
            {
                "var": xr.DataArray(
                    np.ones((2, 2)),
                    dims=("lat", "lon"),
                    coords={
                        "lat": [0, 1],
                        "lon": [0, 1],
                    },
                )
            }
        )

    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)

    cfg = CheckDummy()

    mod._check_data(cfg)

    assert cfg.list_paths == ["fake.nc"]


def test_check_data_required_dim_present(monkeypatch):
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

    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)

    cfg = CheckDummy()

    mod._check_data(cfg)

    assert cfg.list_paths == ["fake.nc"]


def test_weights_load_dir_returns_dataarray(monkeypatch):
    monkeypatch.setattr(Path, "exists", lambda self: True)

    def fake_open_dataset(path):
        return xr.Dataset(
            {
                "a": xr.DataArray(
                    np.ones((2, 2)),
                    dims=("lat", "lon"),
                    coords={
                        "lat": [0, 1],
                        "lon": [0, 1],
                    },
                )
            }
        )

    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)

    weights = mod.WeightsConfig(load_dir="fake.nc").build_weights(
        coords(),
        save=False,
    )

    assert isinstance(weights, xr.DataArray)


def test_weights_uniform_has_expected_dims():
    weights = mod.WeightsConfig("uniform").build_weights(
        coords(),
        save=False,
    )

    assert weights.dims == ("lat", "lon")


def test_weights_cosine_lat_has_expected_dims():
    weights = mod.WeightsConfig("cosine_lat").build_weights(
        coords(),
        save=False,
    )

    assert weights.dims == ("lat", "lon")


def test_unwrap_single_variable_keeps_spatial_dims():
    ds = xr.Dataset(
        {
            "a": xr.DataArray(
                np.ones((2, 2)),
                dims=("lat", "lon"),
            )
        }
    )

    out = mod._unwrap_data_variables(ds)

    assert "lat" in out.dims
    assert "lon" in out.dims


def test_get_ds_info_start_and_final_year(monkeypatch):
    def fake_load(paths, **kwargs):
        return model_ds()

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

    assert info.start_year == 2000
    assert info.final_year == 2001
    mask = mod._create_train_mask([2000, 2001], 12)


def test_create_train_mask_non_excluded_column_true():
    mask = mod._create_train_mask([2000, 2001], 12)

    assert mask.shape == (2, 12)
    assert mask.dtype == bool
    assert mask.sum() >= 0


def test_get_ds_info_contains_coordinates(monkeypatch):
    def fake_load(paths, **kwargs):
        return model_ds()

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

    assert "lat" in info.coords
    assert "lon" in info.coords
    assert "ensembles" in info.coords


def test_unwrap_single_variable_dtype_preserved():
    ds = xr.Dataset(
        {
            "a": xr.DataArray(
                np.ones((2, 2), dtype=np.float32),
                dims=("lat", "lon"),
            )
        }
    )

    out = mod._unwrap_data_variables(ds)

    assert out.dtype == np.float32


def test_unwrap_multiple_variable_shape():
    ds = xr.Dataset(
        {
            "a": xr.DataArray(np.ones((2, 2)), dims=("lat", "lon")),
            "b": xr.DataArray(np.ones((2, 2)), dims=("lat", "lon")),
        }
    )

    out = mod._unwrap_data_variables(ds)

    assert out.shape == (2, 2, 2)


def test_weights_variable_weights_channel_count():
    weights = mod.WeightsConfig(
        variable_weights={"a": 1.0, "b": 2.0, "c": 3.0}
    ).build_weights(
        coords(),
        save=False,
    )

    assert weights.sizes["channels"] == 3


def test_weights_uniform_mean_is_one():
    weights = mod.WeightsConfig("uniform").build_weights(
        coords(),
        save=False,
    )

    assert float(weights.mean()) == 1.0


def test_weights_cosine_lat_contains_finite_values():
    weights = mod.WeightsConfig("cosine_lat").build_weights(
        coords(),
        save=False,
    )

    assert np.isfinite(weights.values).all()


def test_weights_oceannanremover_changes_values():
    class DummyOcean:
        def transform(self, weights):
            return weights + 5

    weights = mod.WeightsConfig().build_weights(
        coords(),
        oceannanremover=DummyOcean(),
        save=False,
    )

    assert float(weights.min()) >= 5


def test_load_xarray_data_selection_reduces_dimension(monkeypatch):
    monkeypatch.setattr(mod, "_load_xarray_data", ORIGINAL_LOAD_XARRAY_DATA)

    def fake_open_mfdataset(paths, *args, **kwargs):
        return xr.Dataset(
            {
                "a": xr.DataArray(
                    np.ones((3, 4)),
                    dims=("lat", "lon"),
                    coords={
                        "lat": [0, 1, 2],
                        "lon": [0, 1, 2, 3],
                    },
                )
            }
        )

    monkeypatch.setattr(xr, "open_mfdataset", fake_open_mfdataset)

    ds = mod._load_xarray_data(
        ["fake.nc"],
        names=["a"],
        selection={"lat": 0},
    )

    assert ds["a"].ndim == 1


def test_load_xarray_data_rename_keeps_only_new_name(monkeypatch):
    monkeypatch.setattr(mod, "_load_xarray_data", ORIGINAL_LOAD_XARRAY_DATA)

    def fake_open_mfdataset(paths, *args, **kwargs):
        return xr.Dataset(
            {
                "old": xr.DataArray(
                    np.ones((2, 2)),
                    dims=("lat", "lon"),
                    coords={
                        "lat": [0, 1],
                        "lon": [0, 1],
                    },
                )
            }
        )

    monkeypatch.setattr(xr, "open_mfdataset", fake_open_mfdataset)

    ds = mod._load_xarray_data(
        ["fake.nc"],
        names=["new"],
        rename_dict={"old": "new"},
    )

    assert "new" in ds.data_vars
    assert "old" not in ds.data_vars


def test_load_xarray_data_without_selection_keeps_shape(monkeypatch):
    monkeypatch.setattr(mod, "_load_xarray_data", ORIGINAL_LOAD_XARRAY_DATA)

    def fake_open_mfdataset(paths, *args, **kwargs):
        return xr.Dataset(
            {
                "a": xr.DataArray(
                    np.ones((2, 3)),
                    dims=("lat", "lon"),
                    coords={
                        "lat": [0, 1],
                        "lon": [0, 1, 2],
                    },
                )
            }
        )

    monkeypatch.setattr(xr, "open_mfdataset", fake_open_mfdataset)

    ds = mod._load_xarray_data(["fake.nc"], names=["a"])

    assert ds["a"].shape == (2, 3)


def test_load_xarray_data_preprocessor_called(monkeypatch):
    monkeypatch.setattr(mod, "_load_xarray_data", ORIGINAL_LOAD_XARRAY_DATA)

    called = {"transform": False}

    class DummyPreprocessor:
        def transform(self, ds):
            called["transform"] = True
            return ds

    def fake_open_mfdataset(paths, *args, **kwargs):
        return xr.Dataset(
            {
                "a": xr.DataArray(
                    np.ones((2, 2)),
                    dims=("lat", "lon"),
                    coords={
                        "lat": [0, 1],
                        "lon": [0, 1],
                    },
                )
            }
        )

    monkeypatch.setattr(xr, "open_mfdataset", fake_open_mfdataset)

    mod._load_xarray_data(
        ["fake.nc"],
        names=["a"],
        preprocessor=DummyPreprocessor(),
    )

    assert called["transform"] is True


def test_check_data_sets_list_paths_from_glob(monkeypatch):
    monkeypatch.setattr(mod, "_check_data", ORIGINAL_CHECK_DATA)

    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(glob, "glob", lambda pattern: ["1.nc", "2.nc"])

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

    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)

    cfg = CheckDummy()

    mod._check_data(cfg)

    assert cfg.list_paths == ["1.nc", "2.nc"]


def test_check_data_accepts_lon_dimension(monkeypatch):
    monkeypatch.setattr(mod, "_check_data", ORIGINAL_CHECK_DATA)

    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(glob, "glob", lambda pattern: ["fake.nc"])

    def fake_open_dataset(path):
        return xr.Dataset(
            {
                "var": xr.DataArray(
                    np.ones((2, 2)),
                    dims=("lat", "lon"),
                    coords={
                        "lat": [0, 1],
                        "lon": [0, 1],
                    },
                )
            }
        )

    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)

    cfg = CheckDummy()

    mod._check_data(cfg)

    assert cfg.list_paths == ["fake.nc"]


def test_get_ds_info_has_start_and_final_year(monkeypatch):
    def fake_load(paths, **kwargs):
        return model_ds()

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

    assert info.start_year == 2000
    assert info.final_year == 2001


def test_condition_config_has_info_attribute(monkeypatch):
    def fake_load(paths, **kwargs):
        return model_ds()

    monkeypatch.setattr(mod, "_load_xarray_data", fake_load)

    cfg = mod.ConditionDataConfig("fake", ["var"])

    assert cfg.info is not None


def test_condition_config_has_year_range(monkeypatch):
    def fake_load(paths, **kwargs):
        return model_ds()

    monkeypatch.setattr(mod, "_load_xarray_data", fake_load)

    cfg = mod.ConditionDataConfig("fake", ["var"])

    assert hasattr(cfg, "year_range")


def test_create_train_mask_returns_xarray():
    mask = mod._create_train_mask([2000], 12)

    assert isinstance(mask, xr.DataArray)


def test_create_train_mask_year_dimension():
    mask = mod._create_train_mask([2000, 2001, 2002], 12)

    assert mask.sizes["year"] == 3


def test_create_train_mask_lead_time_dimension():
    mask = mod._create_train_mask([2000], 12)

    assert mask.sizes["lead_time"] == 12


def test_weights_variable_weights_preserve_lat_lon_dims():
    weights = mod.WeightsConfig(variable_weights={"a": 1.0, "b": 2.0}).build_weights(
        coords(),
        save=False,
    )

    assert "lat" in weights.dims
    assert "lon" in weights.dims


def test_weights_cosine_lat_shape():
    weights = mod.WeightsConfig("cosine_lat").build_weights(
        coords(),
        save=False,
    )

    assert weights.shape == (2, 2)


def test_weights_uniform_dtype():
    weights = mod.WeightsConfig("uniform").build_weights(
        coords(),
        save=False,
    )

    assert np.issubdtype(weights.dtype, np.number)


def test_load_xarray_data_selection_preserves_variable(monkeypatch):
    monkeypatch.setattr(mod, "_load_xarray_data", ORIGINAL_LOAD_XARRAY_DATA)

    def fake_open_mfdataset(paths, *args, **kwargs):
        return xr.Dataset(
            {
                "a": xr.DataArray(
                    np.ones((2, 2)),
                    dims=("lat", "lon"),
                    coords={
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
        selection={"lat": 0},
    )

    assert "a" in ds.data_vars


def test_load_xarray_data_preprocessor_preserves_dataset(monkeypatch):
    monkeypatch.setattr(mod, "_load_xarray_data", ORIGINAL_LOAD_XARRAY_DATA)

    class Dummy:
        def transform(self, ds):
            return ds

    def fake_open_mfdataset(paths, *args, **kwargs):
        return xr.Dataset(
            {
                "a": xr.DataArray(
                    np.ones((2, 2)),
                    dims=("lat", "lon"),
                    coords={
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
        preprocessor=Dummy(),
    )

    assert isinstance(ds, xr.Dataset)


def test_check_data_sets_attribute(monkeypatch):
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

    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)

    cfg = CheckDummy()

    mod._check_data(cfg)

    assert hasattr(cfg, "list_paths")


def test_get_ds_info_returns_info_object(monkeypatch):
    def fake_load(paths, **kwargs):
        return model_ds()

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

    assert info is not None


def test_condition_config_info_not_none(monkeypatch):
    def fake_load(paths, **kwargs):
        return model_ds()

    monkeypatch.setattr(mod, "_load_xarray_data", fake_load)

    cfg = mod.ConditionDataConfig("fake", ["var"])

    assert cfg.info is not None


def test_condition_config_check_ensemble_default_false(monkeypatch):
    def fake_load(paths, **kwargs):
        return model_ds()

    monkeypatch.setattr(mod, "_load_xarray_data", fake_load)

    cfg = mod.ConditionDataConfig("fake", ["var"])

    assert cfg._check_ensemble is False


def test_modeldataconfig_sets_year_range(monkeypatch):
    def fake_load(paths, **kwargs):
        return model_ds()

    monkeypatch.setattr(mod, "_load_xarray_data", fake_load)

    cfg = mod.ModelDataConfig("fake", ["var"])

    assert cfg.year_range is not None
    assert len(cfg.year_range) > 0


def test_obsdataconfig_sets_year_range(monkeypatch):
    def fake_load(paths, **kwargs):
        return obs_ds()

    monkeypatch.setattr(mod, "_load_xarray_data", fake_load)

    cfg = mod.ObsDataConfig("fake", ["var"])

    assert cfg.year_range.tolist() == [2000, 2001]


def test_modeldataconfig_check_ensemble_true(monkeypatch):
    def fake_load(paths, **kwargs):
        return model_ds()

    monkeypatch.setattr(mod, "_load_xarray_data", fake_load)

    cfg = mod.ModelDataConfig(
        "fake",
        ["var"],
        ensemble_list=[0],
    )

    assert cfg._check_ensemble is True


def test_obsdataconfig_check_ensemble_false(monkeypatch):
    def fake_load(paths, **kwargs):
        return obs_ds()

    monkeypatch.setattr(mod, "_load_xarray_data", fake_load)

    cfg = mod.ObsDataConfig("fake", ["var"])

    assert cfg._check_ensemble is False


def test_weightsconfig_post_init_existing_path(monkeypatch):
    monkeypatch.setattr(Path, "exists", lambda self: True)

    cfg = mod.WeightsConfig(load_dir="weights.nc")

    assert cfg.load_dir == "weights.nc"


def test_load_xarray_data_without_ensembles_coord(monkeypatch):
    monkeypatch.setattr(mod, "_load_xarray_data", ORIGINAL_LOAD_XARRAY_DATA)

    def fake_open_mfdataset(paths, *args, **kwargs):
        return xr.Dataset(
            {
                "a": xr.DataArray(
                    np.ones((2, 2)),
                    dims=("lat", "lon"),
                    coords={
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

    assert "a" in ds


def test_load_xarray_data_selection_none(monkeypatch):
    monkeypatch.setattr(mod, "_load_xarray_data", ORIGINAL_LOAD_XARRAY_DATA)

    def fake_open_mfdataset(paths, *args, **kwargs):
        return xr.Dataset(
            {
                "a": xr.DataArray(
                    np.ones((2, 2)),
                    dims=("lat", "lon"),
                    coords={
                        "lat": [0, 1],
                        "lon": [0, 1],
                    },
                )
            }
        )

    monkeypatch.setattr(xr, "open_mfdataset", fake_open_mfdataset)

    ds = mod._load_xarray_data(
        ["fake.nc"],
        selection=None,
        names=["a"],
    )

    assert ds["a"].shape == (2, 2)


def test_load_xarray_data_names_none_with_preprocessor(monkeypatch):
    monkeypatch.setattr(mod, "_load_xarray_data", ORIGINAL_LOAD_XARRAY_DATA)

    class Dummy:
        def transform(self, ds):
            return ds

    def fake_open_mfdataset(paths, *args, **kwargs):
        return xr.Dataset(
            {
                "a": xr.DataArray(np.ones((2, 2)), dims=("lat", "lon")),
                "b": xr.DataArray(np.ones((2, 2)), dims=("lat", "lon")),
            }
        )

    monkeypatch.setattr(xr, "open_mfdataset", fake_open_mfdataset)

    ds = mod._load_xarray_data(
        ["fake.nc"],
        names=None,
        preprocessor=Dummy(),
    )

    assert set(ds.data_vars) == {"a", "b"}


def test_create_train_mask_with_xarray_lead_times():
    lead_times = xr.DataArray(np.arange(1, 13), dims=("lead_time",))

    mask = mod._create_train_mask([2000], lead_times)

    assert mask.sizes["lead_time"] == 12


def test_create_train_mask_returns_named_array():
    mask = mod._create_train_mask([2000], 12)

    assert mask.name == "mask"


def test_unwrap_data_variables_channel_dimension_size():
    ds = xr.Dataset(
        {
            "a": xr.DataArray(np.ones((2, 2)), dims=("lat", "lon")),
            "b": xr.DataArray(np.ones((2, 2)), dims=("lat", "lon")),
            "c": xr.DataArray(np.ones((2, 2)), dims=("lat", "lon")),
        }
    )

    out = mod._unwrap_data_variables(ds)

    assert out.sizes["channels"] == 3


def test_weights_build_without_save(monkeypatch):
    called = {"value": False}

    def fake_to_netcdf(*args, **kwargs):
        called["value"] = True

    monkeypatch.setattr(xr.DataArray, "to_netcdf", fake_to_netcdf)

    mod.WeightsConfig().build_weights(
        coords(),
        save=False,
    )

    assert called["value"] is False


def test_weights_build_with_save(monkeypatch, tmp_path):
    monkeypatch.setattr(
        xr.DataArray,
        "reset_index",
        lambda self, *args, **kwargs: self,
    )

    called = {"value": False}

    def fake_to_netcdf(self, path):
        called["value"] = True

    monkeypatch.setattr(xr.DataArray, "to_netcdf", fake_to_netcdf)

    mod.WeightsConfig().build_weights(
        coords(),
        save=True,
        save_path=tmp_path,
    )

    assert called["value"] is True
