import pytest
import numpy as np
import xarray as xr
import joblib

from cccma_ppp.preprocessing.utils_preprocessing import PreprocessingStepSelector
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd

import cccma_ppp.preprocessing.utils_preprocessing as module


@pytest.fixture
def passthrough_stat_alignment(monkeypatch):
    monkeypatch.setattr(
        module,
        "align_stat_data_lead_time_inverse_transform",
        lambda data, stat: stat,
    )


from cccma_ppp.preprocessing.utils_preprocessing import (
    AnomaliesScaler,
    Flattennanremove,
    Normalizer,
    Standardizer,
    align_stat_data_lead_time_inverse_transform,
)

np.random.seed(0)


class FakeLoaded:
    def __init__(self, fitted=True):
        self.fitted = fitted
        self.reference_shape = (3, 3)
        self.final_locations = xr.DataArray(
            np.arange(8),
            dims=("ref",),
            coords={"ref": np.arange(8)},
        )
        self.common_to_input_and_target = False
        self.spatial_mask = xr.DataArray(
            np.ones((3, 3)),
            dims=("lat", "lon"),
            coords={"lat": [0, 1, 2], "lon": [0, 1, 2]},
        )


def make_data(shape=(2, 2, 2), dims=("time", "lat", "lon")):
    data = np.random.rand(*shape)
    coords = {dim: np.arange(size) for dim, size in zip(dims, shape)}
    return xr.DataArray(data, dims=dims, coords=coords)


def make_ensemble_data():
    data = np.random.rand(2, 2, 2, 2)
    return xr.DataArray(
        data,
        dims=("ensembles", "time", "lat", "lon"),
        coords={
            "ensembles": [0, 1],
            "time": [0, 1],
            "lat": [0, 1],
            "lon": [0, 1],
        },
    )


def make_mask(data):
    return xr.where(data > 0.5, 1.0, np.nan)


def make_geo_data():
    data = np.random.rand(2, 3, 3)
    data[:, 0, 0] = np.nan
    return xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={
            "time": [0, 1],
            "lat": [0, 1, 2],
            "lon": [0, 1, 2],
        },
    )


def assert_same_values(a, b, atol=1e-6):
    assert np.allclose(a.to_numpy(), b.to_numpy(), atol=atol, equal_nan=True)


@pytest.mark.pruned
def test_registry_case_insensitive():
    proc = PreprocessingStepSelector("NORMALIZER").get_preprocessor()
    assert proc is not None


@pytest.mark.pruned
def test_registry_invalid_name():
    with pytest.raises(Exception):
        PreprocessingStepSelector("not_registered").get_preprocessor()


@pytest.mark.pruned
def test_normalizer_basic(passthrough_stat_alignment):
    proc = PreprocessingStepSelector("normalizer").get_preprocessor()

    data = make_data()
    proc.fit(data)

    out = proc.transform(data)
    inv = proc.inverse_transform(out)

    assert proc.fitted
    assert_same_values(inv, data)


def test_normalizer_with_mask(passthrough_stat_alignment):
    proc = PreprocessingStepSelector("normalizer").get_preprocessor()

    data = make_data()
    mask = make_mask(data)

    proc.fit(data, mask=mask)
    out = proc.transform(data)
    inv = proc.inverse_transform(out)

    assert proc.fitted
    assert out.shape == data.shape
    assert inv.shape == data.shape


@pytest.mark.pruned
def test_normalizer_no_mask_explicit():
    proc = PreprocessingStepSelector("normalizer").get_preprocessor()

    data = make_data()
    proc.fit(data, mask=None)

    assert proc.fitted
    assert proc.min is not None
    assert proc.max is not None


@pytest.mark.pruned
def test_normalizer_no_dims():
    proc = PreprocessingStepSelector("normalizer").get_preprocessor()

    data = make_data()
    proc.fit(data)

    assert proc.min is not None
    assert proc.max is not None


@pytest.mark.pruned
def test_normalizer_dims_contains_ensembles():
    proc = PreprocessingStepSelector(
        "normalizer", {"dims": ["ensembles", "time"]}
    ).get_preprocessor()

    data = make_ensemble_data()
    proc.fit(data)

    assert proc.fitted


@pytest.mark.pruned
def test_normalizer_ensemble_branch():
    proc = PreprocessingStepSelector(
        "normalizer", {"dims": ["time"]}
    ).get_preprocessor()

    data = make_ensemble_data()
    proc.fit(data)

    assert hasattr(proc, "large_ensemble")
    assert proc.fitted


@pytest.mark.pruned
def test_normalizer_transform_before_fit():
    proc = PreprocessingStepSelector("normalizer").get_preprocessor()
    data = make_data()

    with pytest.raises(Exception):
        proc.transform(data)


@pytest.mark.pruned
def test_standardizer_basic(passthrough_stat_alignment):
    proc = PreprocessingStepSelector("standardizer").get_preprocessor()

    data = make_data()
    proc.fit(data)

    out = proc.transform(data)
    inv = proc.inverse_transform(out)

    assert proc.fitted
    assert_same_values(inv, data)


@pytest.mark.pruned
def test_standardizer_with_mask(passthrough_stat_alignment):
    proc = PreprocessingStepSelector("standardizer").get_preprocessor()

    data = make_data()
    mask = make_mask(data)

    proc.fit(data, mask=mask)
    out = proc.transform(data)
    inv = proc.inverse_transform(out)

    assert proc.fitted
    assert out.shape == data.shape
    assert inv.shape == data.shape


@pytest.mark.pruned
def test_standardizer_no_mask_explicit():
    proc = PreprocessingStepSelector("standardizer").get_preprocessor()

    data = make_data()
    proc.fit(data, mask=None)

    assert proc.fitted
    assert proc.mean is not None
    assert proc.std is not None


@pytest.mark.pruned
def test_standardizer_zero_std_branch():
    proc = PreprocessingStepSelector("standardizer").get_preprocessor()

    data = xr.DataArray(np.ones((2, 2)), dims=("x", "y"))
    proc.fit(data)

    assert proc.std is not None


@pytest.mark.pruned
def test_standardizer_std_filtering():
    proc = PreprocessingStepSelector("standardizer").get_preprocessor()

    data = xr.DataArray(np.zeros((3, 3)), dims=("x", "y"))
    proc.fit(data)

    assert proc.std is not None


@pytest.mark.pruned
def test_standardizer_mixed_std():
    proc = PreprocessingStepSelector("standardizer").get_preprocessor()

    data = xr.DataArray(np.array([[1, 1, 1], [1, 2, 3]]), dims=("x", "y"))
    proc.fit(data)

    assert proc.std is not None


@pytest.mark.pruned
def test_standardizer_ensemble_branch():
    proc = PreprocessingStepSelector(
        "standardizer", {"dims": ["time"]}
    ).get_preprocessor()

    data = make_ensemble_data()
    proc.fit(data)

    assert hasattr(proc, "large_ensemble")
    assert proc.fitted


@pytest.mark.pruned
def test_standardizer_dims_contains_ensembles():
    proc = PreprocessingStepSelector(
        "standardizer", {"dims": ["ensembles", "time"]}
    ).get_preprocessor()

    data = make_ensemble_data()
    proc.fit(data)

    assert proc.fitted


@pytest.mark.pruned
def test_standardizer_transform_before_fit():
    proc = PreprocessingStepSelector("standardizer").get_preprocessor()
    data = make_data()

    with pytest.raises(Exception):
        proc.transform(data)


@pytest.mark.pruned
def test_anomalies_basic(passthrough_stat_alignment):
    proc = PreprocessingStepSelector("anomalies").get_preprocessor()

    data = make_data()
    proc.fit(data)

    out = proc.transform(data)
    inv = proc.inverse_transform(out)

    assert proc.fitted
    assert_same_values(inv, data)


def test_anomalies_with_mask(passthrough_stat_alignment):
    proc = PreprocessingStepSelector("anomalies").get_preprocessor()

    data = make_data()
    mask = make_mask(data)

    proc.fit(data, mask=mask)

    out = proc.transform(data)
    inv = proc.inverse_transform(out)

    assert proc.fitted
    assert inv.shape == data.shape


@pytest.mark.pruned
def test_anomalies_ensemble_branch():
    proc = PreprocessingStepSelector("anomalies", {"dims": ["time"]}).get_preprocessor()

    data = make_ensemble_data()
    proc.fit(data)

    assert hasattr(proc, "large_ensemble")
    assert proc.fitted


@pytest.mark.pruned
def test_anomalies_no_expand(passthrough_stat_alignment):
    proc = PreprocessingStepSelector("anomalies").get_preprocessor()

    data = xr.DataArray(
        np.random.rand(12, 3, 3),
        dims=("lead", "lat", "lon"),
        coords={"lead": np.arange(12), "lat": [0, 1, 2], "lon": [0, 1, 2]},
    )
    proc.mean = data.mean("lead")

    out = proc.inverse_transform(data)

    assert out.shape == data.shape


@pytest.mark.pruned
def test_anomalies_equal_shape_branch(passthrough_stat_alignment):
    proc = PreprocessingStepSelector("anomalies").get_preprocessor()

    data = xr.DataArray(
        np.random.rand(6, 3, 3),
        dims=("lead", "lat", "lon"),
        coords={"lead": np.arange(6), "lat": [0, 1, 2], "lon": [0, 1, 2]},
    )
    proc.mean = xr.DataArray(
        np.random.rand(6, 3, 3),
        dims=("lead", "lat", "lon"),
        coords=data.coords,
    )

    out = proc.inverse_transform(data)

    assert out.shape == data.shape


@pytest.mark.pruned
def test_anomalies_transform_before_fit():
    proc = PreprocessingStepSelector("anomalies").get_preprocessor()
    data = make_data()

    with pytest.raises(Exception):
        proc.transform(data)


@pytest.mark.pruned
def test_normalizer_inverse_before_fit():
    proc = PreprocessingStepSelector("normalizer").get_preprocessor()
    data = make_data()

    with pytest.raises(Exception):
        proc.inverse_transform(data)


def test_standardizer_inverse_before_fit():
    proc = PreprocessingStepSelector("standardizer").get_preprocessor()
    data = make_data()

    with pytest.raises(Exception):
        proc.inverse_transform(data)


@pytest.mark.pruned
def test_anomalies_inverse_before_fit():
    proc = PreprocessingStepSelector("anomalies").get_preprocessor()
    data = make_data()

    with pytest.raises(Exception):
        proc.inverse_transform(data)


@pytest.mark.pruned
def test_normalizer_with_explicit_dims_lat_lon(passthrough_stat_alignment):
    proc = PreprocessingStepSelector(
        "normalizer", {"dims": ["lat", "lon"]}
    ).get_preprocessor()

    data = make_data()
    proc.fit(data)

    out = proc.transform(data)
    inv = proc.inverse_transform(out)

    assert proc.fitted
    assert np.allclose(inv, data, equal_nan=True)


@pytest.mark.pruned
def test_standardizer_with_explicit_dims_lat_lon(passthrough_stat_alignment):
    proc = PreprocessingStepSelector(
        "standardizer", {"dims": ["lat", "lon"]}
    ).get_preprocessor()

    data = make_data()
    proc.fit(data)

    out = proc.transform(data)
    inv = proc.inverse_transform(out)

    assert proc.fitted
    assert np.allclose(inv, data, atol=1e-6, equal_nan=True)


@pytest.mark.pruned
def test_anomalies_with_explicit_dims_lat_lon(passthrough_stat_alignment):
    proc = PreprocessingStepSelector(
        "anomalies", {"dims": ["lat", "lon"]}
    ).get_preprocessor()

    data = make_data()
    proc.fit(data)

    out = proc.transform(data)
    inv = proc.inverse_transform(out)

    assert proc.fitted
    assert np.allclose(inv, data, atol=1e-6, equal_nan=True)


@pytest.mark.pruned
def test_normalizer_dataset_input(passthrough_stat_alignment):
    proc = PreprocessingStepSelector("normalizer").get_preprocessor()

    da = make_data()
    ds = xr.Dataset({"a": da, "b": da + 1})

    proc.fit(ds)

    out = proc.transform(ds)
    inv = proc.inverse_transform(out)

    assert isinstance(out, xr.Dataset)
    assert isinstance(inv, xr.Dataset)
    assert set(inv.data_vars) == {"a", "b"}


@pytest.mark.pruned
def test_standardizer_dataset_input(passthrough_stat_alignment):
    proc = PreprocessingStepSelector("standardizer").get_preprocessor()

    da = make_data()
    ds = xr.Dataset({"a": da, "b": da + 1})

    proc.fit(ds)

    out = proc.transform(ds)
    inv = proc.inverse_transform(out)

    assert isinstance(out, xr.Dataset)
    assert isinstance(inv, xr.Dataset)
    assert set(inv.data_vars) == {"a", "b"}


@pytest.mark.pruned
def test_anomalies_dataset_input():
    proc = PreprocessingStepSelector("anomalies").get_preprocessor()

    da = make_data()
    ds = xr.Dataset({"a": da, "b": da + 1})

    proc.fit(ds)

    out = proc.transform(ds)

    assert isinstance(out, xr.Dataset)
    assert set(out.data_vars) == {"a", "b"}


@pytest.mark.pruned
def test_anomalies_no_mask_explicit():
    proc = PreprocessingStepSelector("anomalies").get_preprocessor()

    data = make_data()

    proc.fit(data, mask=None)

    assert proc.fitted


@pytest.mark.pruned
def test_standardizer_dataset_roundtrip():
    proc = PreprocessingStepSelector("standardizer").get_preprocessor()

    da = make_data()
    ds = xr.Dataset({"x": da})

    proc.fit(ds)

    out = proc.transform(ds)

    assert isinstance(out, xr.Dataset)


@pytest.mark.pruned
def test_normalizer_dims_tuple_conversion():
    proc = PreprocessingStepSelector(
        "normalizer", {"dims": ["time"]}
    ).get_preprocessor()
    assert isinstance(proc.dims, tuple)


@pytest.mark.pruned
def test_normalizer_min_equals_max():
    proc = PreprocessingStepSelector("normalizer").get_preprocessor()

    data = xr.DataArray(np.ones((2, 2, 2)), dims=("time", "lat", "lon"))
    proc.fit(data)

    out = proc.transform(data)
    assert np.isnan(out).any() or np.isfinite(out).all()


@pytest.mark.pruned
def test_standardizer_negative_std_filtered():
    proc = PreprocessingStepSelector("standardizer").get_preprocessor()

    data = xr.DataArray(np.ones((2, 2)), dims=("x", "y"))
    proc.fit(data)

    assert proc.std is not None


@pytest.mark.pruned
def test_flattener_reference_shape_created():
    proc = PreprocessingStepSelector("flattener").get_preprocessor()

    data = make_geo_data()
    proc.fit(data)

    assert hasattr(proc, "reference_shape")


@pytest.mark.pruned
def test_flattener_target_intersection_logic():
    proc = PreprocessingStepSelector("flattener").get_preprocessor()

    data = make_geo_data()
    target = make_geo_data()

    proc.fit(data, target=target)

    assert proc.common_to_input_and_target is True
    assert proc.final_locations is not None


def test_flattener_transform_without_latlon_dims():
    proc = PreprocessingStepSelector("flattener").get_preprocessor()

    data = make_geo_data()
    proc.fit(data)

    flat = proc.transform(data)

    out = proc.transform(flat)

    assert "ref" in out.dims


@pytest.mark.pruned
def test_flattener_save_with_explicit_path(tmp_path):
    proc = PreprocessingStepSelector("flattener").get_preprocessor()

    data = make_geo_data()
    proc.fit(
        data,
        save=True,
        save_path=tmp_path,
        save_name="abc",
    )

    assert (tmp_path / "abc.joblib").exists()


def test_flattener_load_success(tmp_path, monkeypatch):
    fake = FakeLoaded(fitted=True)

    path = tmp_path / "f.joblib"
    joblib.dump(fake, path)

    proc = PreprocessingStepSelector("flattener", {"load_dir": path}).get_preprocessor()
    proc.fit(make_geo_data())

    assert proc.fitted
    assert proc.final_locations is not None


@pytest.mark.pruned
def test_flattener_load_not_fitted(tmp_path):
    fake = FakeLoaded(fitted=False)

    path = tmp_path / "f.joblib"
    joblib.dump(fake, path)

    proc = PreprocessingStepSelector("flattener", {"load_dir": path}).get_preprocessor()

    with pytest.raises(RuntimeError):
        proc.fit(make_geo_data())


def test_flattener_inverse_structure():
    proc = PreprocessingStepSelector("flattener").get_preprocessor()

    data = make_geo_data()
    proc.fit(data)

    flat = proc.transform(data)
    restored = proc.inverse_transform(flat)

    assert hasattr(restored, "coords")


@pytest.mark.pruned
def test_normalizer_dims_tuple_none():
    proc = PreprocessingStepSelector("normalizer").get_preprocessor()

    assert proc.dims is None


@pytest.mark.pruned
def test_standardizer_dims_tuple_none():
    proc = PreprocessingStepSelector("standardizer").get_preprocessor()

    assert proc.dims is None


@pytest.mark.pruned
def test_anomalies_dims_tuple_none():
    proc = PreprocessingStepSelector("anomalies").get_preprocessor()

    assert proc.dims is None


@pytest.mark.pruned
def test_normalizer_fit_returns_self():
    proc = PreprocessingStepSelector("normalizer").get_preprocessor()

    result = proc.fit(make_data())

    assert result is proc


@pytest.mark.pruned
def test_standardizer_fit_returns_self():
    proc = PreprocessingStepSelector("standardizer").get_preprocessor()

    result = proc.fit(make_data())

    assert result is proc


@pytest.mark.pruned
def test_anomalies_fit_returns_self():
    proc = PreprocessingStepSelector("anomalies").get_preprocessor()

    result = proc.fit(make_data())

    assert result is proc


@pytest.mark.pruned
def test_normalizer_large_ensemble_not_created():
    proc = PreprocessingStepSelector(
        "normalizer",
        {"dims": ["ensembles", "time"]},
    ).get_preprocessor()

    proc.fit(make_ensemble_data())

    assert not hasattr(proc, "large_ensemble")


@pytest.mark.pruned
def test_standardizer_large_ensemble_not_created():
    proc = PreprocessingStepSelector(
        "standardizer",
        {"dims": ["ensembles", "time"]},
    ).get_preprocessor()

    proc.fit(make_ensemble_data())

    assert not hasattr(proc, "large_ensemble")


@pytest.mark.pruned
def test_anomalies_large_ensemble_not_created():
    proc = PreprocessingStepSelector(
        "anomalies",
        {"dims": ["ensembles", "time"]},
    ).get_preprocessor()

    proc.fit(make_ensemble_data())

    assert not hasattr(proc, "large_ensemble")


@pytest.mark.pruned
def test_normalizer_mask_branch_contains_nan():
    proc = PreprocessingStepSelector("normalizer").get_preprocessor()

    data = make_data()

    mask = xr.full_like(
        data,
        np.nan,
    )

    proc.fit(data, mask=mask)

    assert proc.fitted


@pytest.mark.pruned
def test_standardizer_mask_branch_contains_nan():
    proc = PreprocessingStepSelector("standardizer").get_preprocessor()

    data = make_data()

    mask = xr.full_like(
        data,
        np.nan,
    )

    proc.fit(data, mask=mask)

    assert proc.fitted


@pytest.mark.pruned
def test_anomalies_mask_branch_contains_nan():
    proc = PreprocessingStepSelector("anomalies").get_preprocessor()

    data = make_data()

    mask = xr.full_like(
        data,
        np.nan,
    )

    proc.fit(data, mask=mask)

    assert proc.fitted


@pytest.mark.pruned
def test_normalizer_dataset_fit_multiple_variables():
    proc = PreprocessingStepSelector("normalizer").get_preprocessor()

    da = make_data()

    ds = xr.Dataset(
        {
            "a": da,
            "b": da * 2,
        }
    )

    proc.fit(ds)

    assert proc.fitted


@pytest.mark.pruned
def test_standardizer_dataset_fit_multiple_variables():
    proc = PreprocessingStepSelector("standardizer").get_preprocessor()

    da = make_data()

    ds = xr.Dataset(
        {
            "a": da,
            "b": da * 2,
        }
    )

    proc.fit(ds)

    assert proc.fitted


@pytest.mark.pruned
def test_anomalies_dataset_fit_multiple_variables():
    proc = PreprocessingStepSelector("anomalies").get_preprocessor()

    da = make_data()

    ds = xr.Dataset(
        {
            "a": da,
            "b": da * 2,
        }
    )

    proc.fit(ds)

    assert proc.fitted


@pytest.mark.pruned
def test_anomalies_inverse_short_branch(passthrough_stat_alignment):
    proc = PreprocessingStepSelector("anomalies").get_preprocessor()

    proc.mean = xr.DataArray(
        np.ones((12, 2, 2)),
        dims=("lead", "lat", "lon"),
    )

    data = xr.DataArray(
        np.ones((12, 2, 2)),
        dims=("lead", "lat", "lon"),
    )

    out = proc.inverse_transform(data)

    assert out.shape == data.shape


@pytest.mark.pruned
def test_flattener_fit_returns_self():
    proc = PreprocessingStepSelector("flattener").get_preprocessor()

    result = proc.fit(make_geo_data())

    assert result is proc


@pytest.mark.pruned
def test_flattener_no_target_branch():
    proc = PreprocessingStepSelector("flattener").get_preprocessor()

    proc.fit(make_geo_data())

    assert proc.common_to_input_and_target is False


@pytest.mark.pruned
def test_flattener_target_branch():
    proc = PreprocessingStepSelector("flattener").get_preprocessor()

    data = make_geo_data()

    proc.fit(
        data,
        target=data.copy(),
    )

    assert proc.common_to_input_and_target is True


@pytest.mark.pruned
def test_flattener_inverse_missing_ref():
    proc = PreprocessingStepSelector("flattener").get_preprocessor()

    proc.fit(make_geo_data())

    bad = xr.DataArray(
        np.ones((2, 2)),
        dims=("lat", "lon"),
    )

    with pytest.raises(
        ValueError,
        match="ref",
    ):
        proc.inverse_transform(bad)


@pytest.mark.pruned
def test_flattener_transform_ref_branch():
    proc = PreprocessingStepSelector("flattener").get_preprocessor()

    proc.fit(make_geo_data())

    flattened = proc.transform(make_geo_data())

    again = proc.transform(flattened)

    assert "ref" in again.dims


@pytest.mark.pruned
def test_flattener_save_creates_nested_path(
    tmp_path,
):
    proc = PreprocessingStepSelector("flattener").get_preprocessor()

    save_dir = tmp_path / "a" / "b" / "c"

    proc.fit(
        make_geo_data(),
        save=True,
        save_path=save_dir,
    )

    assert save_dir.exists()


@pytest.mark.pruned
def test_flattener_load_from_memory_direct(
    tmp_path,
):
    path = tmp_path / "x.joblib"

    joblib.dump(
        FakeLoaded(fitted=True),
        path,
    )

    proc = PreprocessingStepSelector("flattener").get_preprocessor()

    proc._load_from_memory(path)

    assert proc.fitted


@pytest.mark.pruned
def test_flattener_load_from_memory_copies_attributes(
    tmp_path,
):
    path = tmp_path / "x.joblib"

    fake = FakeLoaded(fitted=True)

    fake.common_to_input_and_target = True

    joblib.dump(
        fake,
        path,
    )

    proc = PreprocessingStepSelector("flattener").get_preprocessor()

    proc._load_from_memory(path)

    assert proc.common_to_input_and_target


@pytest.mark.pruned
def test_flattener_fit_load_dir_branch(
    tmp_path,
):
    path = tmp_path / "x.joblib"

    joblib.dump(
        FakeLoaded(fitted=True),
        path,
    )

    proc = PreprocessingStepSelector(
        "flattener",
        {"load_dir": path},
    ).get_preprocessor()

    result = proc.fit(make_geo_data())

    assert result is proc
    assert proc.fitted


@pytest.mark.pruned
def test_flattener_missing_source_dimensions():
    proc = PreprocessingStepSelector("flattener").get_preprocessor()

    target = make_geo_data()

    data = xr.DataArray(
        np.ones((2,)),
        dims=("time",),
        coords={"time": [0, 1]},
    )

    with pytest.raises(
        RuntimeError,
        match="Missing from input data",
    ):
        proc.fit(
            data,
            target=target,
        )


@pytest.mark.pruned
def test_flattener_nn_dims_created():
    proc = PreprocessingStepSelector("flattener").get_preprocessor()

    proc.fit(make_geo_data())

    assert len(proc.NN_dims) > 0


@pytest.mark.pruned
def test_flattener_final_locations_created():
    proc = PreprocessingStepSelector("flattener").get_preprocessor()

    proc.fit(make_geo_data())

    assert proc.final_locations is not None


TIME_DIM, LEAD_TIME_DIM = module.required_sample_dimensions


def make_spatial_data():
    return xr.DataArray(
        np.asarray(
            [
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ],
                [
                    [5.0, 6.0],
                    [7.0, 8.0],
                ],
            ]
        ),
        dims=("samples", "lat", "lon"),
        coords={
            "samples": [0, 1],
            "lat": [45.0, 46.0],
            "lon": [-124.0, -123.0],
        },
        name="tas",
    )


def make_forecast_data(
    years=(2000, 2001),
    lead_times=(1, 2, 13),
):
    return xr.DataArray(
        np.zeros(
            (
                len(years),
                len(lead_times),
                1,
            )
        ),
        dims=(
            TIME_DIM,
            LEAD_TIME_DIM,
            "channels",
        ),
        coords={
            TIME_DIM: list(years),
            LEAD_TIME_DIM: list(lead_times),
            "channels": ["tas"],
        },
    )


@pytest.mark.pruned
def test_normalizer_defaults():
    scaler = Normalizer()

    assert scaler.min is None
    assert scaler.max is None
    assert scaler.dims is None
    assert scaler.fitted is False


@pytest.mark.pruned
def test_normalizer_converts_dims_to_tuple():
    scaler = Normalizer(
        dims=[
            "samples",
            "lat",
        ]
    )

    assert scaler.dims == (
        "samples",
        "lat",
    )


@pytest.mark.pruned
def test_normalizer_fit_computes_min_and_max():
    data = make_spatial_data()
    scaler = Normalizer(
        dims=["samples"],
    )

    scaler.fit(data)

    expected_min = data.min("samples")
    expected_max = data.max("samples")

    xr.testing.assert_allclose(
        scaler.min,
        expected_min,
    )
    xr.testing.assert_allclose(
        scaler.max,
        expected_max,
    )


@pytest.mark.pruned
def test_normalizer_fit_without_dims_uses_all_dimensions():
    data = make_spatial_data()
    scaler = Normalizer()

    scaler.fit(data)

    assert scaler.min.item() == pytest.approx(1.0)
    assert scaler.max.item() == pytest.approx(8.0)


@pytest.mark.pruned
def test_normalizer_fit_applies_mask():
    data = xr.DataArray(
        [1.0, 2.0, 100.0],
        dims=("samples",),
    )
    mask = xr.DataArray(
        [0.0, 0.0, np.nan],
        dims=("samples",),
    )

    scaler = Normalizer(
        dims=["samples"],
    )
    scaler.fit(
        data,
        mask=mask,
    )

    assert scaler.min.item() == pytest.approx(1.0)
    assert scaler.max.item() == pytest.approx(2.0)


def test_normalizer_adds_ensemble_dimension():
    data = xr.DataArray(
        np.arange(12, dtype=float).reshape(3, 4),
        dims=(
            "ensembles",
            "samples",
        ),
    )

    scaler = Normalizer(
        dims=["samples"],
    )
    scaler.fit(data)

    assert scaler.large_ensemble is True
    assert scaler.dims == (
        "ensembles",
        "samples",
    )


def test_normalizer_does_not_duplicate_ensemble_dimension():
    data = xr.DataArray(
        np.arange(12, dtype=float).reshape(3, 4),
        dims=(
            "ensembles",
            "samples",
        ),
    )

    scaler = Normalizer(
        dims=[
            "ensembles",
            "samples",
        ],
    )
    scaler.fit(data)

    assert scaler.dims == (
        "ensembles",
        "samples",
    )
    assert not hasattr(
        scaler,
        "large_ensemble",
    )


@pytest.mark.pruned
def test_normalizer_does_not_create_large_ensemble_without_dims():
    data = xr.DataArray(
        np.arange(12, dtype=float).reshape(3, 4),
        dims=(
            "ensembles",
            "samples",
        ),
    )

    scaler = Normalizer()
    scaler.fit(data)

    assert not hasattr(
        scaler,
        "large_ensemble",
    )


@pytest.mark.pruned
def test_normalizer_transform():
    data = xr.DataArray(
        [1.0, 2.0, 3.0],
        dims=("samples",),
    )

    scaler = Normalizer(
        dims=["samples"],
    ).fit(data)

    result = scaler.transform(data)

    xr.testing.assert_allclose(
        result,
        xr.DataArray(
            [0.0, 0.5, 1.0],
            dims=("samples",),
        ),
    )


@pytest.mark.pruned
def test_normalizer_inverse_transform(monkeypatch):
    scaler = Normalizer()
    scaler.min = xr.DataArray(2.0)
    scaler.max = xr.DataArray(6.0)

    align = Mock(
        side_effect=[
            xr.DataArray(6.0),
            xr.DataArray(2.0),
        ]
    )
    monkeypatch.setattr(
        module,
        "align_stat_data_lead_time_inverse_transform",
        align,
    )

    normalized = xr.DataArray(
        [0.0, 0.5, 1.0],
        dims=("samples",),
    )

    result = scaler.inverse_transform(normalized)

    xr.testing.assert_allclose(
        result,
        xr.DataArray(
            [2.0, 4.0, 6.0],
            dims=("samples",),
        ),
    )

    assert align.call_count == 2


@pytest.mark.pruned
def test_normalizer_roundtrip(monkeypatch):
    data = xr.DataArray(
        [2.0, 4.0, 6.0],
        dims=("samples",),
    )

    scaler = Normalizer(
        dims=["samples"],
    ).fit(data)

    monkeypatch.setattr(
        module,
        "align_stat_data_lead_time_inverse_transform",
        lambda _, stat: stat,
    )

    transformed = scaler.transform(data)
    restored = scaler.inverse_transform(transformed)

    xr.testing.assert_allclose(
        restored,
        data,
    )


@pytest.mark.pruned
def test_standardizer_defaults():
    scaler = Standardizer()

    assert scaler.mean is None
    assert scaler.std is None
    assert scaler.dims is None
    assert scaler.fitted is False


@pytest.mark.pruned
def test_standardizer_converts_dims_to_tuple():
    scaler = Standardizer(
        dims=[
            "samples",
            "lat",
        ]
    )

    assert scaler.dims == (
        "samples",
        "lat",
    )


@pytest.mark.pruned
def test_standardizer_fit_computes_mean_and_std():
    data = make_spatial_data()
    scaler = Standardizer(
        dims=["samples"],
    )

    scaler.fit(data)

    xr.testing.assert_allclose(
        scaler.mean,
        data.mean("samples"),
    )
    xr.testing.assert_allclose(
        scaler.std,
        data.std("samples"),
    )


def test_standardizer_fit_applies_mask():
    data = xr.DataArray(
        [1.0, 3.0, 100.0],
        dims=("samples",),
    )
    mask = xr.DataArray(
        [0.0, 0.0, np.nan],
        dims=("samples",),
    )

    scaler = Standardizer(
        dims=["samples"],
    )
    scaler.fit(
        data,
        mask=mask,
    )

    assert scaler.mean.item() == pytest.approx(2.0)
    assert scaler.std.item() == pytest.approx(1.0)


@pytest.mark.pruned
def test_standardizer_filters_zero_standard_deviation():
    data = xr.DataArray(
        [
            [1.0, 2.0],
            [1.0, 4.0],
        ],
        dims=(
            "samples",
            "channels",
        ),
    )

    scaler = Standardizer(
        dims=["samples"],
    )
    scaler.fit(data)

    assert np.isnan(scaler.std.sel(channels=0).item())
    assert scaler.std.sel(channels=1).item() == pytest.approx(1.0)


def test_standardizer_adds_ensemble_dimension():
    data = xr.DataArray(
        np.arange(12, dtype=float).reshape(3, 4),
        dims=(
            "ensembles",
            "samples",
        ),
    )

    scaler = Standardizer(
        dims=["samples"],
    )
    scaler.fit(data)

    assert scaler.large_ensemble is True
    assert scaler.dims == (
        "ensembles",
        "samples",
    )


@pytest.mark.pruned
def test_standardizer_transform():
    data = xr.DataArray(
        [1.0, 2.0, 3.0],
        dims=("samples",),
    )

    scaler = Standardizer(
        dims=["samples"],
    ).fit(data)

    result = scaler.transform(data)

    xr.testing.assert_allclose(
        result,
        (data - data.mean("samples")) / data.std("samples"),
    )


@pytest.mark.pruned
def test_standardizer_inverse_transform(monkeypatch):
    scaler = Standardizer()
    scaler.mean = xr.DataArray(4.0)
    scaler.std = xr.DataArray(2.0)

    align = Mock(
        side_effect=[
            xr.DataArray(2.0),
            xr.DataArray(4.0),
        ]
    )
    monkeypatch.setattr(
        module,
        "align_stat_data_lead_time_inverse_transform",
        align,
    )

    standardized = xr.DataArray(
        [-1.0, 0.0, 1.0],
        dims=("samples",),
    )

    result = scaler.inverse_transform(standardized)

    xr.testing.assert_allclose(
        result,
        xr.DataArray(
            [2.0, 4.0, 6.0],
            dims=("samples",),
        ),
    )


@pytest.mark.pruned
def test_standardizer_roundtrip(monkeypatch):
    data = xr.DataArray(
        [2.0, 4.0, 6.0],
        dims=("samples",),
    )

    scaler = Standardizer(
        dims=["samples"],
    ).fit(data)

    monkeypatch.setattr(
        module,
        "align_stat_data_lead_time_inverse_transform",
        lambda _, stat: stat,
    )

    transformed = scaler.transform(data)
    restored = scaler.inverse_transform(transformed)

    xr.testing.assert_allclose(
        restored,
        data,
    )


@pytest.mark.pruned
def test_anomalies_defaults():
    scaler = AnomaliesScaler()

    assert scaler.mean is None
    assert scaler.dims is None
    assert scaler.fitted is False


@pytest.mark.pruned
def test_anomalies_converts_dims_to_tuple():
    scaler = AnomaliesScaler(
        dims=[
            "samples",
            "lat",
        ]
    )

    assert scaler.dims == (
        "samples",
        "lat",
    )


@pytest.mark.pruned
def test_anomalies_fit_computes_mean():
    data = make_spatial_data()

    scaler = AnomaliesScaler(
        dims=["samples"],
    )
    scaler.fit(data)

    xr.testing.assert_allclose(
        scaler.mean,
        data.mean("samples"),
    )


@pytest.mark.pruned
def test_anomalies_fit_applies_mask():
    data = xr.DataArray(
        [1.0, 3.0, 100.0],
        dims=("samples",),
    )
    mask = xr.DataArray(
        [0.0, 0.0, np.nan],
        dims=("samples",),
    )

    scaler = AnomaliesScaler(
        dims=["samples"],
    )
    scaler.fit(
        data,
        mask=mask,
    )

    assert scaler.mean.item() == pytest.approx(2.0)


def test_anomalies_adds_ensemble_dimension():
    data = xr.DataArray(
        np.arange(12, dtype=float).reshape(3, 4),
        dims=(
            "ensembles",
            "samples",
        ),
    )

    scaler = AnomaliesScaler(
        dims=["samples"],
    )
    scaler.fit(data)

    assert scaler.large_ensemble is True
    assert scaler.dims == (
        "ensembles",
        "samples",
    )


@pytest.mark.pruned
def test_anomalies_transform():
    data = xr.DataArray(
        [1.0, 2.0, 3.0],
        dims=("samples",),
    )

    scaler = AnomaliesScaler(
        dims=["samples"],
    ).fit(data)

    result = scaler.transform(data)

    xr.testing.assert_allclose(
        result,
        xr.DataArray(
            [-1.0, 0.0, 1.0],
            dims=("samples",),
        ),
    )


@pytest.mark.pruned
def test_anomalies_inverse_transform(monkeypatch):
    scaler = AnomaliesScaler()
    scaler.mean = xr.DataArray(4.0)

    align = Mock(return_value=xr.DataArray(4.0))
    monkeypatch.setattr(
        module,
        "align_stat_data_lead_time_inverse_transform",
        align,
    )

    anomalies = xr.DataArray(
        [-2.0, 0.0, 2.0],
        dims=("samples",),
    )

    result = scaler.inverse_transform(anomalies)

    xr.testing.assert_allclose(
        result,
        xr.DataArray(
            [2.0, 4.0, 6.0],
            dims=("samples",),
        ),
    )

    align.assert_called_once_with(
        anomalies,
        scaler.mean,
    )


def make_flatten_data():
    return xr.DataArray(
        np.asarray(
            [
                [
                    [1.0, np.nan],
                    [3.0, 4.0],
                ],
                [
                    [2.0, np.nan],
                    [5.0, 6.0],
                ],
            ]
        ),
        dims=(
            "samples",
            "lat",
            "lon",
        ),
        coords={
            "samples": [0, 1],
            "lat": [45.0, 46.0],
            "lon": [-124.0, -123.0],
        },
        name="tas",
    )


                                
@pytest.mark.pruned
# Remove test due to no coverage
def test_flattener_defaults():
    flattener = Flattennanremove()

    assert flattener.load_dir is None
    assert flattener.fitted is False
    assert flattener.common_to_input_and_target is False
    assert flattener.NN_dims == []


@pytest.mark.pruned
def test_flattener_detects_nn_dimensions(monkeypatch):
    monkeypatch.setattr(
        module,
        "supported_NN_dimensions_sorted",
        [
            "lat",
            "lon",
        ],
    )

    flattener = Flattennanremove()
    flattener.fit(make_flatten_data())

    assert flattener.NN_dims == [
        "lat",
        "lon",
    ]


@pytest.mark.pruned
def test_flattener_creates_reference_shape(monkeypatch):
    monkeypatch.setattr(
        module,
        "supported_NN_dimensions_sorted",
        [
            "lat",
            "lon",
        ],
    )

    data = make_flatten_data()
    flattener = Flattennanremove()
    flattener.fit(data)

    assert set(flattener.reference_shape.coords) == {
        "lat",
        "lon",
    }

    xr.testing.assert_equal(
        flattener.reference_shape["lat"],
        data["lat"],
    )
    xr.testing.assert_equal(
        flattener.reference_shape["lon"],
        data["lon"],
    )


@pytest.mark.pruned
def test_flattener_removes_nan_locations(monkeypatch):
    monkeypatch.setattr(
        module,
        "supported_NN_dimensions_sorted",
        [
            "lat",
            "lon",
        ],
    )

    flattener = Flattennanremove()
    flattener.fit(make_flatten_data())

    assert flattener.final_locations.size == 3
    assert flattener.common_to_input_and_target is False


def test_flattener_rejects_missing_input_nn_dimensions(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "supported_NN_dimensions_sorted",
        [
            "lat",
            "lon",
        ],
    )

    data = xr.DataArray(
        np.ones((2, 2)),
        dims=(
            "samples",
            "lat",
        ),
        coords={
            "samples": [0, 1],
            "lat": [45.0, 46.0],
        },
    )

    target = xr.DataArray(
        np.ones((2, 2, 2)),
        dims=(
            "samples",
            "lat",
            "lon",
        ),
        coords={
            "samples": [0, 1],
            "lat": [45.0, 46.0],
            "lon": [-124.0, -123.0],
        },
    )

    with pytest.raises(
        RuntimeError,
        match="Missing from input data",
    ):
        Flattennanremove().fit(
            data,
            target=target,
        )


@pytest.mark.pruned
def test_flattener_transform_stacks_spatial_dims(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "supported_NN_dimensions_sorted",
        [
            "lat",
            "lon",
        ],
    )

    data = make_flatten_data()
    flattener = Flattennanremove()
    flattener.fit(data)

    result = flattener.transform(data)

    assert "ref" in result.dims
    assert result.dims[-1] == "ref"
    assert result.sizes["ref"] == 3


@pytest.mark.pruned
def test_flattener_transform_existing_ref_dimension(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "supported_NN_dimensions_sorted",
        [
            "lat",
            "lon",
        ],
    )

    data = make_flatten_data()
    flattener = Flattennanremove()
    flattener.fit(data)

    stacked = data.stack(ref=flattener.NN_dims)
    result = flattener.transform(stacked)

    xr.testing.assert_equal(
        result,
        stacked.sel(ref=flattener.final_locations),
    )


def test_flattener_inverse_requires_ref_dimension():
    flattener = Flattennanremove()

    with pytest.raises(
        ValueError,
        match="must contain the flattened 'ref' dimension",
    ):
        flattener.inverse_transform(
            xr.DataArray(
                np.ones((2, 3)),
                dims=(
                    "samples",
                    "channels",
                ),
            )
        )


@pytest.mark.pruned
def test_flattener_inverse_restores_spatial_layout(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "supported_NN_dimensions_sorted",
        [
            "lat",
            "lon",
        ],
    )

    data = make_flatten_data()
    flattener = Flattennanremove()
    flattener.fit(data)

    transformed = flattener.transform(data)
    result = flattener.inverse_transform(transformed)

    assert "lat" in result.dims
    assert "lon" in result.dims
    assert result.sizes["lat"] == 2
    assert result.sizes["lon"] == 2


def test_flattener_save_uses_default_name(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "supported_NN_dimensions_sorted",
        [
            "lat",
            "lon",
        ],
    )
    monkeypatch.setattr(
        module.RuntimeContext,
        "GLOBAL_EXP_DIR",
        tmp_path,
    )

    dump = Mock()
    monkeypatch.setattr(
        module.joblib,
        "dump",
        dump,
    )

    flattener = Flattennanremove()
    flattener.fit(
        make_flatten_data(),
        save=True,
    )

    dump.assert_called_once_with(
        flattener,
        tmp_path / "flattener.joblib",
    )


@pytest.mark.pruned
def test_flattener_check_nn_dims_accepts_none():
    flattener = Flattennanremove()
    flattener.NN_dims = [
        "lat",
        "lon",
    ]

    assert flattener._check_nn_dims(None) is None


@pytest.mark.pruned
def test_flattener_check_nn_dims_accepts_matching_data():
    flattener = Flattennanremove()
    flattener.NN_dims = [
        "lat",
        "lon",
    ]

    data = xr.DataArray(
        np.ones((2, 2)),
        dims=(
            "lat",
            "lon",
        ),
    )

    assert flattener._check_nn_dims(data) is None


def test_flattener_check_nn_dims_rejects_missing_dims():
    flattener = Flattennanremove()
    flattener.NN_dims = [
        "lat",
        "lon",
    ]

    data = xr.DataArray(
        np.ones(2),
        dims=("lat",),
    )

    with pytest.raises(
        ValueError,
        match="Missing dimensions.*lon",
    ):
        flattener._check_nn_dims(data)


def test_flattener_load_rejects_unfitted_object(
    tmp_path,
    monkeypatch,
):
    loaded = SimpleNamespace(
        fitted=False,
    )

    load = Mock(return_value=loaded)
    monkeypatch.setattr(
        module.joblib,
        "load",
        load,
    )

    flattener = Flattennanremove()

    with pytest.raises(
        RuntimeError,
        match="has to be fitted first",
    ):
        flattener._load_from_memory(tmp_path / "flattener.joblib")


@pytest.mark.pruned
def test_flattener_load_copies_state(
    tmp_path,
    monkeypatch,
):
    reference_shape = xr.Dataset(
        coords={
            "lat": [45.0, 46.0],
            "lon": [-124.0, -123.0],
        }
    )
    final_locations = xr.DataArray(
        pd.MultiIndex.from_tuples(
            [
                (45.0, -124.0),
                (46.0, -123.0),
            ],
            names=[
                "lat",
                "lon",
            ],
        ),
        dims=("ref",),
    )

    loaded = SimpleNamespace(
        reference_shape=reference_shape,
        final_locations=final_locations,
        common_to_input_and_target=True,
        fitted=True,
    )

    monkeypatch.setattr(
        module.joblib,
        "load",
        Mock(return_value=loaded),
    )

    flattener = Flattennanremove()
    flattener._load_from_memory(tmp_path / "flattener.joblib")

    assert flattener.reference_shape is reference_shape
    assert flattener.final_locations is final_locations
    assert flattener.common_to_input_and_target is True
    assert flattener.fitted is True


@pytest.mark.pruned
def test_align_rejects_missing_time_dimension():
    data = xr.DataArray(
        np.zeros((2, 1)),
        dims=(
            LEAD_TIME_DIM,
            "channels",
        ),
        coords={
            LEAD_TIME_DIM: [1, 2],
            "channels": ["tas"],
        },
    )

    with pytest.raises(
        ValueError,
        match="must contain the time dimension",
    ):
        align_stat_data_lead_time_inverse_transform(
            data,
            xr.DataArray(1.0),
        )


def test_align_rejects_missing_lead_time_dimension():
    data = xr.DataArray(
        np.zeros((2, 1)),
        dims=(
            TIME_DIM,
            "channels",
        ),
        coords={
            TIME_DIM: [2000, 2001],
            "channels": ["tas"],
        },
    )

    with pytest.raises(
        ValueError,
        match="must contain the lead-time dimension",
    ):
        align_stat_data_lead_time_inverse_transform(
            data,
            xr.DataArray(1.0),
        )


def test_align_returns_static_statistic_unchanged():
    data = make_forecast_data()
    stat = xr.DataArray(
        [1.0, 2.0],
        dims=("channels",),
        coords={
            "channels": [
                "tas",
                "pr",
            ]
        },
    )

    result = align_stat_data_lead_time_inverse_transform(
        data,
        stat,
    )

    assert result is stat


def test_align_returns_lead_time_statistic_unchanged():
    data = make_forecast_data(
        lead_times=(1, 2, 3),
    )
    stat = xr.DataArray(
        [10.0, 20.0, 30.0],
        dims=(LEAD_TIME_DIM,),
        coords={
            LEAD_TIME_DIM: [1, 2, 3],
        },
    )

    result = align_stat_data_lead_time_inverse_transform(
        data,
        stat,
    )

    assert result is stat


def test_align_preserves_time_dimension_when_stat_has_lead_time():
    data = make_forecast_data(
        years=(2000, 2001),
        lead_times=(1, 2),
    )
    stat = xr.DataArray(
        np.asarray(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        ),
        dims=(
            "year",
            LEAD_TIME_DIM,
        ),
        coords={
            "year": [
                2000,
                2001,
            ],
            LEAD_TIME_DIM: [1, 2],
        },
    )

    result = align_stat_data_lead_time_inverse_transform(
        data,
        stat,
    )

    assert TIME_DIM in result.dims
    assert LEAD_TIME_DIM in result.dims


@pytest.mark.pruned
def test_align_month_only_statistic():
    data = make_forecast_data(
        years=(2000,),
        lead_times=(
            1,
            2,
            12,
            13,
        ),
    )
    stat = xr.DataArray(
        np.arange(
            1,
            13,
            dtype=float,
        ),
        dims=("month",),
        coords={
            "month": np.arange(1, 13),
        },
    )

    result = align_stat_data_lead_time_inverse_transform(
        data,
        stat,
    )

    assert result.dims == (
        TIME_DIM,
        LEAD_TIME_DIM,
    )

    np.testing.assert_array_equal(
        result.values,
        np.asarray(
            [
                [
                    1.0,
                    2.0,
                    12.0,
                    1.0,
                ]
            ]
        ),
    )


def test_align_rejects_month_statistic_without_twelve_entries():
    data = make_forecast_data(
        years=(2000,),
        lead_times=(1, 2),
    )
    stat = xr.DataArray(
        np.arange(11),
        dims=("month",),
        coords={
            "month": np.arange(1, 12),
        },
    )

    with pytest.raises(
        ValueError,
        match="exactly 12 month entries",
    ):
        align_stat_data_lead_time_inverse_transform(
            data,
            stat,
        )


def test_align_year_only_statistic():
    data = make_forecast_data(
        years=(2000,),
        lead_times=(
            1,
            13,
            25,
        ),
    )
    stat = xr.DataArray(
        [
            10.0,
            20.0,
            30.0,
        ],
        dims=("year",),
        coords={
            "year": [
                2000,
                2001,
                2002,
            ],
        },
    )

    result = align_stat_data_lead_time_inverse_transform(
        data,
        stat,
    )

    np.testing.assert_array_equal(
        result.values,
        np.asarray(
            [
                [
                    10.0,
                    20.0,
                    30.0,
                ]
            ]
        ),
    )


@pytest.mark.pruned
def test_align_year_and_month_statistic():
    data = make_forecast_data(
        years=(2000,),
        lead_times=(
            1,
            12,
            13,
            14,
        ),
    )

    values = np.stack(
        [
            np.arange(1, 13),
            np.arange(101, 113),
        ]
    )

    stat = xr.DataArray(
        values,
        dims=(
            "year",
            "month",
        ),
        coords={
            "year": [
                2000,
                2001,
            ],
            "month": np.arange(1, 13),
        },
    )

    result = align_stat_data_lead_time_inverse_transform(
        data,
        stat,
    )

    np.testing.assert_array_equal(
        result.values,
        np.asarray(
            [
                [
                    1,
                    12,
                    101,
                    102,
                ]
            ]
        ),
    )


@pytest.mark.pruned
def test_align_rejects_missing_valid_years():
    data = make_forecast_data(
        years=(2000,),
        lead_times=(
            1,
            13,
        ),
    )
    stat = xr.DataArray(
        [10.0],
        dims=("year",),
        coords={
            "year": [2000],
        },
    )

    with pytest.raises(
        ValueError,
        match="Missing years.*2001",
    ):
        align_stat_data_lead_time_inverse_transform(
            data,
            stat,
        )


@pytest.mark.pruned
def test_align_preserves_non_temporal_dimensions():
    data = make_forecast_data(
        years=(2000,),
        lead_times=(1, 2),
    )
    stat = xr.DataArray(
        np.arange(
            24,
            dtype=float,
        ).reshape(12, 2),
        dims=(
            "month",
            "channels",
        ),
        coords={
            "month": np.arange(1, 13),
            "channels": [
                "tas",
                "pr",
            ],
        },
    )

    result = align_stat_data_lead_time_inverse_transform(
        data,
        stat,
    )

    assert TIME_DIM in result.dims
    assert LEAD_TIME_DIM in result.dims
    assert "channels" in result.dims
    assert result.sizes["channels"] == 2


@pytest.mark.pruned
def test_align_removes_auxiliary_coordinates():
    data = make_forecast_data(
        years=(2000,),
        lead_times=(1, 13),
    )
    stat = xr.DataArray(
        np.arange(24).reshape(2, 12),
        dims=(
            "year",
            "month",
        ),
        coords={
            "year": [
                2000,
                2001,
            ],
            "month": np.arange(1, 13),
        },
    )

    result = align_stat_data_lead_time_inverse_transform(
        data,
        stat,
    )

    assert "__stat_year" not in result.coords
    assert "__stat_month" not in result.coords


@pytest.mark.pruned
def test_align_broadcasts_across_initialization_years():
    data = make_forecast_data(
        years=(
            2000,
            2001,
        ),
        lead_times=(
            1,
            13,
        ),
    )
    stat = xr.DataArray(
        [
            10.0,
            20.0,
            30.0,
        ],
        dims=("year",),
        coords={
            "year": [
                2000,
                2001,
                2002,
            ],
        },
    )

    result = align_stat_data_lead_time_inverse_transform(
        data,
        stat,
    )

    np.testing.assert_array_equal(
        result.values,
        np.asarray(
            [
                [10.0, 20.0],
                [20.0, 30.0],
            ]
        ),
    )