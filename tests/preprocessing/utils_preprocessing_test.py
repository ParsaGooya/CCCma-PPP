import pytest
import numpy as np
import xarray as xr
import joblib

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


def test_registry_integration():
    names = PreprocessingStepSelector.available()
    assert "normalizer" in names
    assert "standardizer" in names
    assert "anomalies" in names
    assert "oceannanremover" in names


def test_registry_case_insensitive():
    proc = PreprocessingStepSelector("NORMALIZER").get_preprocessor()
    assert proc is not None


def test_registry_invalid_name():
    with pytest.raises(Exception):
        PreprocessingStepSelector("not_registered").get_preprocessor()


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


def test_normalizer_no_mask_explicit():
    proc = PreprocessingStepSelector("normalizer").get_preprocessor()

    data = make_data()
    proc.fit(data, mask=None)

    assert proc.fitted
    assert proc.min is not None
    assert proc.max is not None


def test_normalizer_no_dims():
    proc = PreprocessingStepSelector("normalizer").get_preprocessor()

    data = make_data()
    proc.fit(data)

    assert proc.min is not None
    assert proc.max is not None


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


def test_normalizer_transform_before_fit():
    proc = PreprocessingStepSelector("normalizer").get_preprocessor()
    data = make_data()

    with pytest.raises(Exception):
        proc.transform(data)


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


def test_standardizer_no_mask_explicit():
    proc = PreprocessingStepSelector("standardizer").get_preprocessor()

    data = make_data()
    proc.fit(data, mask=None)

    assert proc.fitted
    assert proc.mean is not None
    assert proc.std is not None


def test_standardizer_zero_std_branch():
    proc = PreprocessingStepSelector("standardizer").get_preprocessor()

    data = xr.DataArray(np.ones((2, 2)), dims=("x", "y"))
    proc.fit(data)

    assert proc.std is not None


def test_standardizer_std_filtering():
    proc = PreprocessingStepSelector("standardizer").get_preprocessor()

    data = xr.DataArray(np.zeros((3, 3)), dims=("x", "y"))
    proc.fit(data)

    assert proc.std is not None


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


def test_standardizer_transform_before_fit():
    proc = PreprocessingStepSelector("standardizer").get_preprocessor()
    data = make_data()

    with pytest.raises(Exception):
        proc.transform(data)


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


def test_anomalies_transform_before_fit():
    proc = PreprocessingStepSelector("anomalies").get_preprocessor()
    data = make_data()

    with pytest.raises(Exception):
        proc.transform(data)


def test_oceannan_basic():
    proc = PreprocessingStepSelector("oceannanremover").get_preprocessor()

    data = make_geo_data()
    proc.fit(data)

    out = proc.transform(data)
    inv = proc.inverse_transform(out)

    assert proc.fitted
    assert "ref" in out.dims
    assert "ref" not in inv.dims


def test_oceannan_with_target():
    proc = PreprocessingStepSelector("oceannanremover").get_preprocessor()

    data = make_geo_data()
    target = make_geo_data()

    proc.fit(data, target=target)

    assert proc.fitted
    assert proc.common_to_input_and_target


def test_oceannan_flat_input_branch():
    proc = PreprocessingStepSelector("oceannanremover").get_preprocessor()

    data = make_geo_data()
    proc.fit(data)

    flat = data.stack(ref=["lat", "lon"]).dropna("ref")
    out = proc.transform(flat)

    assert "ref" in out.dims


def test_oceannan_no_latlon():
    proc = PreprocessingStepSelector("oceannanremover").get_preprocessor()

    data = make_geo_data()
    proc.fit(data)

    flat = data.stack(ref=["lat", "lon"])
    out = proc.transform(flat)

    assert "ref" in out.dims


def test_oceannan_partial_dims():
    proc = PreprocessingStepSelector("oceannanremover").get_preprocessor()

    data = make_geo_data()
    proc.fit(data)

    data_partial = data.drop_vars("lon", errors="ignore")
    out = proc.transform(data_partial)

    assert out is not None


def test_oceannan_inverse_full():
    proc = PreprocessingStepSelector("oceannanremover").get_preprocessor()

    data = make_geo_data()
    proc.fit(data)

    flat = proc.transform(data)
    out = proc.inverse_transform(flat)

    assert "lat" in out.dims
    assert "lon" in out.dims


def test_oceannan_transform_before_fit():
    proc = PreprocessingStepSelector("oceannanremover").get_preprocessor()
    data = make_geo_data()

    with pytest.raises(Exception):
        proc.transform(data)


def test_oceannan_save(tmp_path):
    proc = PreprocessingStepSelector("oceannanremover").get_preprocessor()

    data = make_geo_data()
    save_dir = tmp_path / "save_dir"
    save_dir.mkdir()

    proc.fit(data, save=True, save_path=save_dir, save_name="test")

    saved = save_dir / "test.joblib"
    assert saved.exists()


def test_oceannan_save_default_name(tmp_path):
    proc = PreprocessingStepSelector("oceannanremover").get_preprocessor()

    data = make_geo_data()
    save_dir = tmp_path / "save_dir"
    save_dir.mkdir()

    proc.fit(data, save=True, save_path=save_dir)

    assert any(path.suffix == ".joblib" for path in save_dir.iterdir())


def test_oceannan_load(tmp_path):
    file = tmp_path / "test.joblib"
    joblib.dump(FakeLoaded(), file)

    proc = PreprocessingStepSelector("oceannanremover").get_preprocessor()
    proc.load_dir = file

    proc._load_from_memory(file)

    assert proc.fitted


def test_oceannan_load_unfitted(tmp_path):
    file = tmp_path / "bad.joblib"
    joblib.dump(FakeLoaded(fitted=False), file)

    proc = PreprocessingStepSelector("oceannanremover").get_preprocessor()

    with pytest.raises(AssertionError):
        proc._load_from_memory(file)


def test_oceannan_load_missing_file(tmp_path):
    proc = PreprocessingStepSelector("oceannanremover").get_preprocessor()

    with pytest.raises(Exception):
        proc._load_from_memory(tmp_path / "missing.joblib")


def test_oceannan_fit_with_load_dir(tmp_path):
    file = tmp_path / "obj.joblib"
    joblib.dump(FakeLoaded(), file)

    proc = PreprocessingStepSelector("oceannanremover").get_preprocessor()
    proc.load_dir = file

    proc.fit(make_geo_data())

    assert proc.fitted


def test_oceannan_fit_with_target_and_save(tmp_path):
    proc = PreprocessingStepSelector("oceannanremover").get_preprocessor()

    data = make_geo_data()
    target = make_geo_data()
    save_dir = tmp_path / "save_target"
    save_dir.mkdir()

    proc.fit(
        data, target=target, save=True, save_path=save_dir, save_name="target_case"
    )

    assert proc.common_to_input_and_target
    assert (save_dir / "target_case.joblib").exists()


def test_oceannan_inverse_transform_unfitted():
    proc = PreprocessingStepSelector("oceannanremover").get_preprocessor()
    data = make_geo_data()

    with pytest.raises(Exception):
        proc.inverse_transform(data)


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


def test_anomalies_inverse_before_fit():
    proc = PreprocessingStepSelector("anomalies").get_preprocessor()
    data = make_data()

    with pytest.raises(Exception):
        proc.inverse_transform(data)


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


def test_anomalies_dataset_input():
    proc = PreprocessingStepSelector("anomalies").get_preprocessor()

    da = make_data()
    ds = xr.Dataset({"a": da, "b": da + 1})

    proc.fit(ds)

    out = proc.transform(ds)

    assert isinstance(out, xr.Dataset)
    assert set(out.data_vars) == {"a", "b"}


def test_oceannan_transform_target_like_ref_input():
    proc = PreprocessingStepSelector("oceannanremover").get_preprocessor()

    data = make_geo_data()
    target = make_geo_data()

    proc.fit(data, target=target)

    flat = target.stack(ref=["lat", "lon"]).dropna("ref")
    out = proc.transform(flat)

    assert "ref" in out.dims


def test_oceannan_inverse_transform_ref_only_input():
    proc = PreprocessingStepSelector("oceannanremover").get_preprocessor()

    data = make_geo_data()
    proc.fit(data)

    flat = proc.transform(data)
    out = proc.inverse_transform(flat)

    assert "lat" in out.dims
    assert "lon" in out.dims


def test_oceannan_transform_after_loaded_object(tmp_path):
    data = make_geo_data()

    proc = PreprocessingStepSelector("oceannanremover").get_preprocessor()
    proc.fit(data)

    file = tmp_path / "ocean.joblib"
    joblib.dump(proc, file)

    loaded = PreprocessingStepSelector("oceannanremover").get_preprocessor()
    loaded._load_from_memory(file)

    out = loaded.transform(data)

    assert loaded.fitted
    assert "ref" in out.dims


def test_oceannan_transform_ref_passthrough():
    proc = PreprocessingStepSelector("oceannanremover").get_preprocessor()

    data = make_geo_data()
    proc.fit(data)

    flat = data.stack(ref=["lat", "lon"])

    out = proc.transform(flat)

    assert "ref" in out.dims


def test_oceannan_inverse_non_ref_input():
    proc = PreprocessingStepSelector("oceannanremover").get_preprocessor()

    data = make_geo_data()
    proc.fit(data)

    out = proc.inverse_transform(data)

    assert out is not None


def test_anomalies_no_mask_explicit():
    proc = PreprocessingStepSelector("anomalies").get_preprocessor()

    data = make_data()

    proc.fit(data, mask=None)

    assert proc.fitted


def test_standardizer_dataset_roundtrip():
    proc = PreprocessingStepSelector("standardizer").get_preprocessor()

    da = make_data()
    ds = xr.Dataset({"x": da})

    proc.fit(ds)

    out = proc.transform(ds)

    assert isinstance(out, xr.Dataset)
