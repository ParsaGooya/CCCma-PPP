import os
from unittest.mock import patch
import numpy as np
import pytest
import xarray as xr
from cccma_ppp.data_modules.utils import (
    WeightsConfig,
    _create_train_mask,
    _load_xarray_data,
)


class DummyFlatten:
    def __init__(self):
        self.final_locations = xr.DataArray(
            np.array([0, 1]),
            dims=("ref",),
            coords={"ref": [0, 1]},
        )

    def transform(self, weights):
        return weights.stack(ref=("lat", "lon"))


class DummyPreprocessor:
    def transform(self, ds):
        ds.attrs["transformed"] = True
        return ds


def make_coords():
    return {
        "lat": xr.DataArray(
            [0, 1],
            dims=("lat",),
            coords={"lat": [0, 1]},
        ),
        "lon": xr.DataArray(
            [10, 20],
            dims=("lon",),
            coords={"lon": [10, 20]},
        ),
    }


def make_dataset():
    return xr.Dataset(
        {
            "a": (
                ("year", "lead_time", "lat", "lon"),
                np.random.rand(2, 2, 2, 2),
            ),
            "b": (
                ("year", "lead_time", "lat", "lon"),
                np.random.rand(2, 2, 2, 2),
            ),
        },
        coords={
            "year": [2000, 2001],
            "lead_time": [1, 2],
            "lat": [0, 1],
            "lon": [10, 20],
        },
    )


def make_ensemble_dataset():
    return xr.Dataset(
        {
            "a": (
                ("ensembles", "year", "lead_time", "lat", "lon"),
                np.random.rand(2, 2, 2, 2, 2),
            )
        },
        coords={
            "ensembles": [0, 1],
            "year": [2000, 2001],
            "lead_time": [1, 2],
            "lat": [0, 1],
            "lon": [10, 20],
        },
    )


def test_weights_config_default():
    cfg = WeightsConfig()

    assert cfg.spatial_method == "uniform"


def test_weights_config_missing_load_dir(tmp_path):
    missing = tmp_path / "missing.nc"

    with pytest.raises(FileNotFoundError):
        WeightsConfig(load_dir=missing)


def test_build_weights_uniform():
    cfg = WeightsConfig(
        spatial_method="uniform",
    )

    weights = cfg.build_weights(
        target_coords=make_coords(),
        save=False,
    )

    assert np.allclose(weights.values, 1.0)


def test_build_weights_cosine_lat():
    cfg = WeightsConfig(
        spatial_method="cosine_lat",
    )

    weights = cfg.build_weights(
        target_coords=make_coords(),
        save=False,
    )

    assert not np.allclose(weights.values, 1.0)


def test_build_weights_variable_weights():
    cfg = WeightsConfig(
        variable_weights={
            "a": 1.0,
            "b": 2.0,
        }
    )

    weights = cfg.build_weights(
        target_coords=make_coords(),
        save=False,
    )

    assert "channels" in weights.dims


def test_build_weights_with_flattennanremover():
    cfg = WeightsConfig()

    weights = cfg.build_weights(
        target_coords=make_coords(),
        Flattennanremover=DummyFlatten(),
        save=False,
    )

    assert "ref" in weights.dims


def test_build_weights_creates_directory(tmp_path):
    cfg = WeightsConfig()

    outdir = tmp_path / "newdir"

    weights = xr.DataArray(
        np.ones((4,)),
        dims=("ref",),
        coords={"ref": [0, 1, 2, 3]},
    )

    with patch.object(
        xr.DataArray,
        "to_netcdf",
        lambda *args, **kwargs: None,
    ):
        with patch.object(
            xr.DataArray,
            "reset_index",
            return_value=weights,
        ):
            cfg.build_weights(
                target_coords=make_coords(),
                Flattennanremover=DummyFlatten(),
                save=True,
                save_path=outdir,
            )

    assert os.path.isdir(outdir)


def test_build_weights_load_dir(tmp_path):
    cfg = WeightsConfig(load_dir=tmp_path)

    weights_da = xr.DataArray(
        np.ones((2, 2)),
        dims=("lat", "lon"),
        coords={
            "lat": [0, 1],
            "lon": [10, 20],
        },
    )

    with patch(
        "xarray.open_dataset",
        return_value=weights_da,
    ):
        weights = cfg.build_weights(
            target_coords=make_coords(),
            save=False,
        )

    assert weights is not None


def test_build_weights_loaded_ref_coords_match(tmp_path):
    cfg = WeightsConfig(load_dir=tmp_path)

    flatten = DummyFlatten()

    weights_da = xr.DataArray(
        np.ones((2,)),
        dims=("ref",),
        coords={"ref": [0, 1]},
    )

    with patch(
        "xarray.open_dataset",
        return_value=weights_da,
    ):
        weights = cfg.build_weights(
            Flattennanremover=flatten,
            save=False,
        )

    assert weights is not None


def test_build_weights_loaded_ref_coords_mismatch(tmp_path):
    cfg = WeightsConfig(load_dir=tmp_path)

    flatten = DummyFlatten()

    weights_da = xr.DataArray(
        np.ones((2,)),
        dims=("ref",),
        coords={"ref": [9, 10]},
    )

    with patch(
        "xarray.open_dataset",
        return_value=weights_da,
    ):
        with pytest.raises(ValueError):
            cfg.build_weights(
                Flattennanremover=flatten,
                save=False,
            )


def test_build_weights_loaded_lat_mismatch(tmp_path):
    cfg = WeightsConfig(load_dir=tmp_path)

    weights_da = xr.DataArray(
        np.ones((2, 2)),
        dims=("lat", "lon"),
        coords={
            "lat": [99, 100],
            "lon": [10, 20],
        },
    )

    with patch(
        "xarray.open_dataset",
        return_value=weights_da,
    ):
        with pytest.raises(ValueError):
            cfg.build_weights(
                target_coords=make_coords(),
                save=False,
            )


def test_build_weights_loaded_lon_mismatch(tmp_path):
    cfg = WeightsConfig(load_dir=tmp_path)

    weights_da = xr.DataArray(
        np.ones((2, 2)),
        dims=("lat", "lon"),
        coords={
            "lat": [0, 1],
            "lon": [999, 1000],
        },
    )

    with patch(
        "xarray.open_dataset",
        return_value=weights_da,
    ):
        with pytest.raises(ValueError):
            cfg.build_weights(
                target_coords=make_coords(),
                save=False,
            )


def test_load_xarray_data_basic():
    ds = make_dataset()

    with patch(
        "xarray.open_mfdataset",
        return_value=ds,
    ):
        result = _load_xarray_data(
            paths=["x.nc"],
        )

    assert result is not None


def test_load_xarray_data_selection():
    ds = make_dataset()

    with patch(
        "xarray.open_mfdataset",
        return_value=ds,
    ):
        result = _load_xarray_data(
            paths=["x.nc"],
            selection={"year": 2000},
        )

    assert result.year.size == 1


def test_load_xarray_data_names():
    ds = make_dataset()

    with patch(
        "xarray.open_mfdataset",
        return_value=ds,
    ):
        result = _load_xarray_data(
            paths=["x.nc"],
            names=["a"],
        )

    assert list(result.data_vars) == ["a"]


def test_load_xarray_data_ensemble_mean():
    ds = make_ensemble_dataset()

    with patch(
        "xarray.open_mfdataset",
        return_value=ds,
    ):
        result = _load_xarray_data(
            paths=["x.nc"],
            ensemble_mean=True,
        )

    assert "ensembles" not in result.dims


def test_load_xarray_data_no_ensemble_mean():
    ds = make_ensemble_dataset()

    with patch(
        "xarray.open_mfdataset",
        return_value=ds,
    ):
        result = _load_xarray_data(
            paths=["x.nc"],
            ensemble_mean=False,
        )

    assert "ensembles" in result.dims


def test_load_xarray_data_preprocessor():
    ds = make_dataset()

    with patch(
        "xarray.open_mfdataset",
        return_value=ds,
    ):
        result = _load_xarray_data(
            paths=["x.nc"],
            preprocessor=DummyPreprocessor(),
        )

    assert result.attrs["transformed"] is True


def test_load_xarray_data_rename():
    ds = xr.Dataset(
        {
            "old": (
                ("year", "lead_time"),
                np.random.rand(2, 2),
            )
        },
        coords={
            "year": [2000, 2001],
            "lead_time": [1, 2],
        },
    )

    with patch(
        "xarray.open_mfdataset",
        return_value=ds,
    ):
        result = _load_xarray_data(
            paths=["x.nc"],
            rename_dict={"old": "new"},
        )

    assert "new" in result.data_vars


def test_create_train_mask_basic():
    mask = _create_train_mask(
        years=[2000, 2001],
        lead_times=np.arange(1, 13),
    )

    assert mask.shape == (2, 12)


def test_create_train_mask_int_lead_times():
    mask = _create_train_mask(
        years=[2000, 2001],
        lead_times=12,
    )

    assert mask.lead_time.size == 12


def test_create_train_mask_dims():
    mask = _create_train_mask(
        years=[2000],
        lead_times=np.arange(1, 13),
    )

    assert mask.dims == ("year", "lead_time")


def test_create_train_mask_name():
    mask = _create_train_mask(
        years=[2000],
        lead_times=np.arange(1, 13),
    )

    assert mask.name == "mask"


def test_create_train_mask_contains_true():
    mask = _create_train_mask(
        years=[2000, 2001],
        lead_times=24,
    )

    assert bool(mask.any())
