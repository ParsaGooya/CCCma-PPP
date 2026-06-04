import pytest
import numpy as np
import xarray as xr
import joblib
from pathlib import Path

# import classes (they register automatically)
from src.cccma_ppp.preprocessing.utils_preprocessing import PreprocessingStepSelector


class FakeLoaded:
    def __init__(self, fitted=True):
        self.fitted = fitted
        self.reference_shape = None
        self.final_locations = None


# ============================================================
# HELPERS
# ============================================================


def make_data(shape=(2, 2, 2), dims=("time", "lat", "lon")):
    data = np.random.rand(*shape)
    return xr.DataArray(data, dims=dims)


def make_ensemble_data():
    data = np.random.rand(2, 2, 2, 2)
    return xr.DataArray(data, dims=("ensembles", "time", "lat", "lon"))


def make_mask(data):
    return xr.where(data > 0.5, 1.0, np.nan)


# ============================================================
# NORMALIZER
# ============================================================


def test_normalizer_basic():
    proc = PreprocessingStepSelector("normalizer").get_preprocessor()

    data = make_data()
    proc.fit(data)

    out = proc.transform(data)
    inv = proc.inverse_transform(out)

    assert np.allclose(inv, data, equal_nan=True)


def test_normalizer_with_mask():
    proc = PreprocessingStepSelector("normalizer").get_preprocessor()

    data = make_data()
    mask = make_mask(data)

    proc.fit(data, mask=mask)

    assert proc.fitted


def test_normalizer_ensemble_branch():
    proc = PreprocessingStepSelector(
        "normalizer", {"dims": ["time"]}
    ).get_preprocessor()

    data = make_ensemble_data()
    proc.fit(data)

    assert hasattr(proc, "large_ensemble")


# ============================================================
# STANDARDIZER
# ============================================================


def test_standardizer_basic():
    proc = PreprocessingStepSelector("standardizer").get_preprocessor()

    data = make_data()
    proc.fit(data)

    out = proc.transform(data)
    inv = proc.inverse_transform(out)

    assert np.allclose(inv, data, atol=1e-6, equal_nan=True)


def test_standardizer_zero_std_branch():
    proc = PreprocessingStepSelector("standardizer").get_preprocessor()

    data = xr.DataArray(np.ones((2, 2)), dims=("x", "y"))
    proc.fit(data)

    assert proc.std is not None


def test_standardizer_ensemble_branch():
    proc = PreprocessingStepSelector(
        "standardizer", {"dims": ["time"]}
    ).get_preprocessor()

    data = make_ensemble_data()
    proc.fit(data)

    assert hasattr(proc, "large_ensemble")


# ============================================================
# ANOMALIES
# ============================================================


def test_anomalies_basic():
    proc = PreprocessingStepSelector("anomalies").get_preprocessor()

    data = make_data()
    proc.fit(data)

    out = proc.transform(data)
    inv = proc.inverse_transform(out)

    assert np.allclose(inv, data, equal_nan=True)


def test_anomalies_ensemble_branch():
    proc = PreprocessingStepSelector("anomalies", {"dims": ["time"]}).get_preprocessor()

    data = make_ensemble_data()
    proc.fit(data)

    assert hasattr(proc, "large_ensemble")


def test_anomalies_inverse_expand_branch():
    proc = PreprocessingStepSelector("anomalies").get_preprocessor()

    # force shape mismatch branch
    data = xr.DataArray(np.random.rand(24, 3, 3), dims=("lead", "lat", "lon"))
    proc.mean = xr.DataArray(np.random.rand(12, 3, 3), dims=("lead", "lat", "lon"))

    out = proc.inverse_transform(data)

    assert out.shape == data.shape


# ============================================================
# OCEAN NAN REMOVER
# ============================================================


def make_geo_data():
    data = np.random.rand(2, 3, 3)
    data[0, 0, 0] = np.nan
    return xr.DataArray(data, dims=("time", "lat", "lon"))


def test_oceannan_basic():
    proc = PreprocessingStepSelector("oceannanremover").get_preprocessor()

    data = make_geo_data()
    proc.fit(data)

    out = proc.transform(data)
    inv = proc.inverse_transform(out)

    assert "ref" not in inv.dims  # fully reconstructed


def test_oceannan_with_target():
    proc = PreprocessingStepSelector("oceannanremover").get_preprocessor()

    data = make_geo_data()
    target = make_geo_data()

    proc.fit(data, target=target)

    assert proc.common_to_input_and_target


def test_oceannan_flat_input_branch():
    proc = PreprocessingStepSelector("oceannanremover").get_preprocessor()

    data = make_geo_data()
    proc.fit(data)

    flat = data.stack(ref=["lat", "lon"]).dropna("ref")

    out = proc.transform(flat)

    assert "ref" in out.dims


# ============================================================
# SAVE + LOAD
# ============================================================


def test_oceannan_load(monkeypatch, tmp_path):
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


# ============================================================
# REGISTRY INTEGRATION
# ============================================================


def test_registry_integration():
    names = PreprocessingStepSelector.available()

    assert "normalizer" in names
    assert "standardizer" in names
    assert "anomalies" in names
    assert "oceannanremover" in names


def test_normalizer_no_dims():
    proc = PreprocessingStepSelector("normalizer").get_preprocessor()

    data = make_data()
    proc.fit(data)

    assert proc.min is not None


def test_normalizer_dims_contains_ensembles():
    proc = PreprocessingStepSelector(
        "normalizer", {"dims": ["ensembles", "time"]}
    ).get_preprocessor()

    data = make_ensemble_data()
    proc.fit(data)

    assert proc.fitted


def test_standardizer_std_filtering():
    proc = PreprocessingStepSelector("standardizer").get_preprocessor()

    data = xr.DataArray(np.zeros((3, 3)), dims=("x", "y"))
    proc.fit(data)

    # std=0 → should get NaNs filtered
    assert proc.std is not None


def test_anomalies_no_expand():
    proc = PreprocessingStepSelector("anomalies").get_preprocessor()

    data = xr.DataArray(np.random.rand(12, 3, 3), dims=("lead", "lat", "lon"))
    proc.mean = data.mean("lead")

    out = proc.inverse_transform(data)

    assert out.shape == data.shape


def test_anomalies_with_mask():
    proc = PreprocessingStepSelector("anomalies").get_preprocessor()

    data = make_data()
    mask = xr.where(data > 0.5, 1, np.nan)

    proc.fit(data, mask=mask)

    assert proc.fitted


def test_oceannan_save(monkeypatch, tmp_path):
    proc = PreprocessingStepSelector("oceannanremover").get_preprocessor()

    data = make_geo_data()

    file = tmp_path / "save_dir"
    file.mkdir()

    proc.fit(data, save=True, save_path=file, save_name="test")

    saved = file / "test.joblib"
    assert saved.exists()


def test_oceannan_inverse_full():
    proc = PreprocessingStepSelector("oceannanremover").get_preprocessor()

    data = make_geo_data()
    proc.fit(data)

    flat = proc.transform(data)
    out = proc.inverse_transform(flat)

    assert set(out.dims) == set(["lat", "lon"])


def test_oceannan_no_latlon():
    proc = PreprocessingStepSelector("oceannanremover").get_preprocessor()

    data = make_geo_data()
    proc.fit(data)

    flat = data.stack(ref=["lat", "lon"])

    out = proc.transform(flat)

    assert "ref" in out.dims


def test_registry_case_insensitive():
    proc = PreprocessingStepSelector("NORMALIZER").get_preprocessor()

    assert proc is not None


def test_standardizer_mixed_std():
    proc = PreprocessingStepSelector("standardizer").get_preprocessor()

    data = xr.DataArray(np.array([[1, 1, 1], [1, 2, 3]]), dims=("x", "y"))

    proc.fit(data)

    assert proc.std is not None


def test_anomalies_equal_shape_branch():
    proc = PreprocessingStepSelector("anomalies").get_preprocessor()

    data = xr.DataArray(np.random.rand(6, 3, 3), dims=("lead", "lat", "lon"))
    proc.mean = xr.DataArray(np.random.rand(6, 3, 3), dims=("lead", "lat", "lon"))

    out = proc.inverse_transform(data)

    assert out.shape == data.shape


def test_oceannan_fit_with_load_dir(tmp_path):
    file = tmp_path / "obj.joblib"

    joblib.dump(FakeLoaded(), file)

    proc = PreprocessingStepSelector("oceannanremover").get_preprocessor()
    proc.load_dir = file

    proc.fit(make_geo_data())

    assert proc.fitted


def test_oceannan_partial_dims():
    proc = PreprocessingStepSelector("oceannanremover").get_preprocessor()

    data = make_geo_data()
    proc.fit(data)

    # drop only lon
    data_partial = data.drop_vars("lon", errors="ignore")

    out = proc.transform(data_partial)

    assert out is not None


def test_normalizer_no_mask_explicit():
    proc = PreprocessingStepSelector("normalizer").get_preprocessor()

    data = make_data()
    proc.fit(data, mask=None)

    assert proc.fitted


def test_standardizer_no_mask_explicit():
    proc = PreprocessingStepSelector("standardizer").get_preprocessor()

    data = make_data()
    proc.fit(data, mask=None)

    assert proc.fitted
