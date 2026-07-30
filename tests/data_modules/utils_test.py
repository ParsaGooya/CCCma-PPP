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


class DummyLoadedFlatten:
    def __init__(self):
        self.final_locations = xr.DataArray(
            np.array([0, 1]),
            dims=("ref",),
            coords={
                "ref": [0, 1],
            },
        )

    def transform(self, weights):
        return weights


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


@pytest.mark.pruned
def test_weights_config_default():
    cfg = WeightsConfig()

    assert cfg.spatial_method == "uniform"


def test_weights_config_missing_load_dir(tmp_path):
    missing = tmp_path / "missing.nc"

    with pytest.raises(FileNotFoundError):
        WeightsConfig(load_dir=missing)


@pytest.mark.pruned
def test_build_weights_uniform():
    cfg = WeightsConfig(
        spatial_method="uniform",
    )

    weights = cfg.build_weights(
        target_coords=make_coords(),
        save=False,
    )

    assert np.allclose(weights.values, 1.0)


@pytest.mark.pruned
def test_build_weights_cosine_lat():
    cfg = WeightsConfig(
        spatial_method="cosine_lat",
    )

    weights = cfg.build_weights(
        target_coords=make_coords(),
        save=False,
    )

    assert not np.allclose(weights.values, 1.0)


@pytest.mark.pruned
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


@pytest.mark.pruned
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


@pytest.mark.pruned
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

    flatten = DummyLoadedFlatten()

    weights_da = xr.DataArray(
        np.ones((2,)),
        dims=("ref",),
        coords={
            "ref": [0, 1],
        },
    )

    target_coords = {
        "ref": xr.DataArray(
            [0, 1],
            dims=("ref",),
            coords={
                "ref": [0, 1],
            },
        ),
    }

    with patch(
        "xarray.open_dataset",
        return_value=weights_da,
    ):
        weights = cfg.build_weights(
            target_coords=target_coords,
            Flattennanremover=flatten,
            save=False,
        )

    assert weights is not None
    assert np.array_equal(
        weights.coords["ref"].values,
        np.array([0, 1]),
    )


@pytest.mark.pruned
def test_build_weights_loaded_ref_coords_mismatch(tmp_path):
    cfg = WeightsConfig(load_dir=tmp_path)

    flatten = DummyFlatten()

    weights_da = xr.DataArray(
        np.ones((2,)),
        dims=("ref",),
        coords={
            "ref": [9, 10],
        },
    )

    target_coords = {
        "ref": xr.DataArray(
            [0, 1],
            dims=("ref",),
            coords={
                "ref": [0, 1],
            },
        ),
    }

    with patch(
        "xarray.open_dataset",
        return_value=weights_da,
    ):
        with pytest.raises(
            ValueError,
            match="must have coordinates that match",
        ):
            cfg.build_weights(
                target_coords=target_coords,
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


@pytest.mark.pruned
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


@pytest.mark.pruned
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


@pytest.mark.pruned
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


@pytest.mark.pruned
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


@pytest.mark.pruned
def test_create_train_mask_basic():
    mask = _create_train_mask(
        [2000, 2001],
        np.arange(1, 13),
    )

    assert isinstance(mask, xr.DataArray)
    assert mask.shape == (2, 12)


def test_create_train_mask_int_lead_times():
    mask = _create_train_mask(
        [2000, 2001],
        12,
    )

    assert isinstance(mask, xr.DataArray)
    assert mask.sizes["lead_time"] == 12


@pytest.mark.pruned
def test_create_train_mask_dims():
    mask = _create_train_mask(
        [2000],
        np.arange(1, 13),
    )

    assert mask.dims == (
        "year",
        "lead_time",
    )


@pytest.mark.pruned
def test_create_train_mask_name():
    mask = _create_train_mask(
        [2000],
        np.arange(1, 13),
    )

    assert mask.name == "mask"


@pytest.mark.pruned
def test_create_train_mask_contains_true():
    mask = _create_train_mask(
        [2000, 2001],
        24,
    )

    assert bool(mask.any())


@pytest.mark.pruned
def test_build_weights_uniform_shape():
    cfg = WeightsConfig(
        spatial_method="uniform",
    )

    weights = cfg.build_weights(
        target_coords=make_coords(),
        save=False,
    )

    assert weights.dims == (
        "lat",
        "lon",
    )
    assert weights.shape == (2, 2)


@pytest.mark.pruned
def test_build_weights_uniform_coordinates():
    cfg = WeightsConfig(
        spatial_method="uniform",
    )

    weights = cfg.build_weights(
        target_coords=make_coords(),
        save=False,
    )

    np.testing.assert_array_equal(
        weights.coords["lat"].values,
        np.array([0, 1]),
    )
    np.testing.assert_array_equal(
        weights.coords["lon"].values,
        np.array([10, 20]),
    )


@pytest.mark.pruned
def test_build_weights_cosine_lat_varies_by_latitude():
    coords = {
        "lat": xr.DataArray(
            [0.0, 60.0],
            dims=("lat",),
            coords={
                "lat": [0.0, 60.0],
            },
        ),
        "lon": xr.DataArray(
            [10.0, 20.0],
            dims=("lon",),
            coords={
                "lon": [10.0, 20.0],
            },
        ),
    }

    cfg = WeightsConfig(
        spatial_method="cosine_lat",
    )

    weights = cfg.build_weights(
        target_coords=coords,
        save=False,
    )

    assert not np.allclose(
        weights.sel(lat=0.0).values,
        weights.sel(lat=60.0).values,
    )


@pytest.mark.pruned
def test_build_weights_variable_weight_values():
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

    np.testing.assert_array_equal(
        weights.coords["channels"].values,
        np.array(["a", "b"]),
    )


@pytest.mark.pruned
def test_build_weights_variable_weight_dimension_size():
    cfg = WeightsConfig(
        variable_weights={
            "a": 1.0,
            "b": 2.0,
            "c": 3.0,
        }
    )

    weights = cfg.build_weights(
        target_coords=make_coords(),
        save=False,
    )

    assert weights.sizes["channels"] == 3


def test_build_weights_loaded_missing_coordinate(tmp_path):
    cfg = WeightsConfig(
        load_dir=tmp_path,
    )

    weights_da = xr.DataArray(
        np.ones((2,)),
        dims=("lat",),
        coords={
            "lat": [0, 1],
        },
    )

    with patch(
        "xarray.open_dataset",
        return_value=weights_da,
    ):
        with pytest.raises(
            ValueError,
            match="must have coordinates that match",
        ):
            cfg.build_weights(
                target_coords=make_coords(),
                save=False,
            )


@pytest.mark.pruned
def test_build_weights_loaded_values_preserved(tmp_path):
    cfg = WeightsConfig(
        load_dir=tmp_path,
    )

    values = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    weights_da = xr.DataArray(
        values,
        dims=(
            "lat",
            "lon",
        ),
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

    np.testing.assert_array_equal(
        weights.values,
        values,
    )


@pytest.mark.pruned
def test_build_weights_loaded_flatten_transform_called(tmp_path):
    cfg = WeightsConfig(
        load_dir=tmp_path,
    )

    weights_da = xr.DataArray(
        np.ones((2,)),
        dims=("ref",),
        coords={
            "ref": [0, 1],
        },
    )

    target_coords = {
        "ref": xr.DataArray(
            [0, 1],
            dims=("ref",),
            coords={
                "ref": [0, 1],
            },
        ),
    }

    class InspectFlatten:
        def __init__(self):
            self.called = False
            self.final_locations = target_coords["ref"]

        def transform(self, weights):
            self.called = True
            return weights

    flatten = InspectFlatten()

    with patch(
        "xarray.open_dataset",
        return_value=weights_da,
    ):
        cfg.build_weights(
            target_coords=target_coords,
            Flattennanremover=flatten,
            save=False,
        )

    assert flatten.called is True


@pytest.mark.pruned
def test_build_weights_save_false_does_not_write():
    cfg = WeightsConfig()

    with patch.object(
        xr.DataArray,
        "to_netcdf",
    ) as save_mock:
        cfg.build_weights(
            target_coords=make_coords(),
            save=False,
        )

    save_mock.assert_not_called()


def test_build_weights_save_true_writes_file(tmp_path):
    cfg = WeightsConfig()

    with patch.object(
        xr.DataArray,
        "to_netcdf",
    ) as save_mock:
        cfg.build_weights(
            target_coords=make_coords(),
            save=True,
            save_path=tmp_path,
            save_name="weights.nc",
        )

    save_mock.assert_called_once()


@pytest.mark.pruned
def test_build_weights_custom_save_name(tmp_path):
    cfg = WeightsConfig()

    captured = {}

    def fake_to_netcdf(self, path, *args, **kwargs):
        captured["path"] = path

    with patch.object(
        xr.DataArray,
        "to_netcdf",
        fake_to_netcdf,
    ):
        cfg.build_weights(
            target_coords=make_coords(),
            save=True,
            save_path=tmp_path,
            save_name="custom_weights.nc",
        )

    assert captured["path"].name == ("custom_weights.nc")


@pytest.mark.pruned
def test_load_xarray_data_empty_selection():
    ds = make_dataset()

    with patch(
        "xarray.open_mfdataset",
        return_value=ds,
    ):
        result = _load_xarray_data(
            paths=["x.nc"],
            selection={},
        )

    assert result.sizes["year"] == 2
    assert result.sizes["lead_time"] == 2


@pytest.mark.pruned
def test_load_xarray_data_multiple_names():
    ds = make_dataset()

    with patch(
        "xarray.open_mfdataset",
        return_value=ds,
    ):
        result = _load_xarray_data(
            paths=["x.nc"],
            names=["a", "b"],
        )

    assert set(result.data_vars) == {
        "a",
        "b",
    }


def test_load_xarray_data_load_true():
    ds = make_dataset()

    with (
        patch(
            "xarray.open_mfdataset",
            return_value=ds,
        ),
        patch.object(
            xr.Dataset,
            "load",
            return_value=ds,
        ) as load_mock,
    ):
        result = _load_xarray_data(
            paths=["x.nc"],
            load=True,
        )

    load_mock.assert_called_once()
    assert result is ds


@pytest.mark.pruned
def test_load_xarray_data_ensemble_mean_values():
    ds = make_ensemble_dataset()
    expected = ds.mean(
        dim="ensembles",
        keep_attrs=True,
    )

    with patch(
        "xarray.open_mfdataset",
        return_value=ds,
    ):
        result = _load_xarray_data(
            paths=["x.nc"],
            ensemble_mean=True,
        )

    xr.testing.assert_allclose(
        result,
        expected,
    )


@pytest.mark.pruned
def test_load_xarray_data_ensemble_mean_without_dimension():
    ds = make_dataset()

    with patch(
        "xarray.open_mfdataset",
        return_value=ds,
    ):
        result = _load_xarray_data(
            paths=["x.nc"],
            ensemble_mean=True,
        )

    assert "ensembles" not in result.dims
    assert result.sizes["year"] == 2


@pytest.mark.pruned
def test_load_xarray_data_rename_coordinate():
    ds = make_dataset()

    with patch(
        "xarray.open_mfdataset",
        return_value=ds,
    ):
        result = _load_xarray_data(
            paths=["x.nc"],
            rename_dict={
                "year": "time",
            },
        )

    assert "time" in result.dims
    assert "year" not in result.dims


@pytest.mark.pruned
def test_load_xarray_data_passes_paths():
    ds = make_dataset()

    with patch(
        "xarray.open_mfdataset",
        return_value=ds,
    ) as open_mock:
        _load_xarray_data(
            paths=[
                "a.nc",
                "b.nc",
            ],
        )

    assert open_mock.call_args.args[0] == [
        "a.nc",
        "b.nc",
    ]
