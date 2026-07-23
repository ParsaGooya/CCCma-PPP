import pytest
import numpy as np
import xarray as xr
import joblib

from cccma_ppp.generic.runtime import RuntimeContext
from cccma_ppp.preprocessing.utils_preprocessing import PreprocessingStepSelector


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
def test_normalizer_basic():
    proc = PreprocessingStepSelector("normalizer").get_preprocessor()

    data = make_data()
    proc.fit(data)

    out = proc.transform(data)
    inv = proc.inverse_transform(out)

    assert proc.fitted
    assert_same_values(inv, data)


def test_normalizer_with_mask():
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
def test_standardizer_basic():
    proc = PreprocessingStepSelector("standardizer").get_preprocessor()

    data = make_data()
    proc.fit(data)

    out = proc.transform(data)
    inv = proc.inverse_transform(out)

    assert proc.fitted
    assert_same_values(inv, data)


def test_standardizer_with_mask():
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


def test_standardizer_ensemble_branch():
    proc = PreprocessingStepSelector(
        "standardizer", {"dims": ["time"]}
    ).get_preprocessor()

    data = make_ensemble_data()
    proc.fit(data)

    assert hasattr(proc, "large_ensemble")
    assert proc.fitted


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
def test_anomalies_basic():
    proc = PreprocessingStepSelector("anomalies").get_preprocessor()

    data = make_data()
    proc.fit(data)

    out = proc.transform(data)
    inv = proc.inverse_transform(out)

    assert proc.fitted
    assert_same_values(inv, data)


def test_anomalies_with_mask():
    proc = PreprocessingStepSelector("anomalies").get_preprocessor()

    data = make_data()
    mask = make_mask(data)

    proc.fit(data, mask=mask)

    out = proc.transform(data)
    inv = proc.inverse_transform(out)

    assert proc.fitted
    assert inv.shape == data.shape


def test_anomalies_ensemble_branch():
    proc = PreprocessingStepSelector("anomalies", {"dims": ["time"]}).get_preprocessor()

    data = make_ensemble_data()
    proc.fit(data)

    assert hasattr(proc, "large_ensemble")
    assert proc.fitted


@pytest.mark.pruned
def test_anomalies_no_expand():
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
def test_anomalies_equal_shape_branch():
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


@pytest.mark.pruned
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
def test_normalizer_with_explicit_dims_lat_lon():
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
def test_standardizer_with_explicit_dims_lat_lon():
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
def test_anomalies_with_explicit_dims_lat_lon():
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
def test_normalizer_dataset_input():
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
def test_standardizer_dataset_input():
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
def test_anomalies_inverse_short_branch():
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
def test_flattener_saved_default_name(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        RuntimeContext,
        "GLOBAL_EXP_DIR",
        str(tmp_path),
    )

    proc = PreprocessingStepSelector("flattener").get_preprocessor()

    proc.fit(
        make_geo_data(),
        save=True,
    )

    assert (tmp_path / "flattener.joblib").exists()


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