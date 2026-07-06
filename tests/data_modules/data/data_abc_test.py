import pytest
import numpy as np
from unittest.mock import patch
import xarray as xr

from cccma_ppp.data_modules.data.data_abc import (
    DataConfigABC,
    _resolve_data,
    _get_ds_info,
)


def make_valid_ds():
    return xr.Dataset(
        {
            "var": (
                ("year", "lead_time", "lat", "lon"),
                np.random.rand(2, 2, 2, 2),
            )
        },
        coords={
            "year": [2000, 2001],
            "lead_time": [1, 2],
            "lat": [0, 1],
            "lon": [10, 20],
        },
    )


def make_ensemble_ds():
    return xr.Dataset(
        {
            "var": (
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


class DummyPipeline:
    def __init__(self):
        self.name = None
        self.fitted = True
        self.loaded = False
        self.fit_called = False

    def set_name(self, name):
        self.name = name

    def fit(self, **kwargs):
        self.fit_called = True

    def _load_from_memory(self, path):
        self.loaded = True


class DummyDataConfig(DataConfigABC):
    TYPE = "dummy"

    def __init__(
        self,
        paths,
        names=None,
        ensemble_list=None,
        rename_dict=None,
        file_type="*.nc",
    ):
        self.paths = paths
        self.names = names or ["var"]
        self.ensemble_list = ensemble_list
        self.rename_dict = rename_dict
        self.file_type = file_type

        self.concat_dim = "channels"
        self.ensemble_mean = False

        self.preprocessing_pipeline = DummyPipeline()

        self._check_ensemble = ensemble_list is not None

        super().__init__()

    @classmethod
    def _allowed_dims(cls):
        return frozenset(
            {
                "year",
                "lead_time",
                "lat",
                "lon",
                "ensembles",
            }
        )

    @classmethod
    def _required_dims(cls):
        return frozenset({"year", "lead_time"})


def test_missing_preprocessing_pipeline():
    class BadConfig(DataConfigABC):
        TYPE = "x"

        @classmethod
        def _allowed_dims(cls):
            return frozenset()

        @classmethod
        def _required_dims(cls):
            return frozenset()

    with pytest.raises(AttributeError):
        BadConfig()


@pytest.mark.pruned
def test_pipeline_name_set(tmp_path):
    cfg = DummyDataConfig(tmp_path)

    assert cfg.preprocessing_pipeline.name == "dummy"


def test_resolve_data_missing_path(tmp_path):
    cfg = DummyDataConfig(tmp_path / "missing")

    with pytest.raises(FileNotFoundError):
        _resolve_data(cfg)


def test_resolve_data_empty_directory(tmp_path):
    cfg = DummyDataConfig(tmp_path)

    with pytest.raises(FileNotFoundError):
        _resolve_data(cfg)


@pytest.mark.pruned
def test_resolve_data_valid(tmp_path):
    cfg = DummyDataConfig(tmp_path)

    with (
        patch("glob.glob", return_value=["x.nc"]),
        patch("xarray.open_dataset", return_value=make_valid_ds()),
    ):
        _resolve_data(cfg)

    assert len(cfg.list_paths) == 1


def test_resolve_data_missing_required_dims(tmp_path):
    ds = xr.Dataset(
        {"var": (("x",), [1, 2])},
        coords={"x": [0, 1]},
    )

    cfg = DummyDataConfig(tmp_path)

    with (
        patch("glob.glob", return_value=["x.nc"]),
        patch("xarray.open_dataset", return_value=ds),
    ):
        with pytest.raises(ValueError):
            _resolve_data(cfg)


def test_resolve_data_invalid_dims(tmp_path):
    ds = xr.Dataset(
        {
            "var": (
                ("year", "lead_time", "bad"),
                np.random.rand(2, 2, 2),
            )
        },
        coords={
            "year": [1, 2],
            "lead_time": [1, 2],
            "bad": [1, 2],
        },
    )

    cfg = DummyDataConfig(tmp_path)

    with (
        patch("glob.glob", return_value=["x.nc"]),
        patch("xarray.open_dataset", return_value=ds),
    ):
        with pytest.raises(ValueError):
            _resolve_data(cfg)


def test_resolve_data_missing_variable(tmp_path):
    ds = xr.Dataset(
        {
            "other": (
                ("year", "lead_time"),
                np.random.rand(2, 2),
            )
        },
        coords={
            "year": [1, 2],
            "lead_time": [1, 2],
        },
    )

    cfg = DummyDataConfig(tmp_path)

    with (
        patch("glob.glob", return_value=["x.nc"]),
        patch("xarray.open_dataset", return_value=ds),
    ):
        with pytest.raises(ValueError):
            _resolve_data(cfg)


def test_resolve_data_missing_coords(tmp_path):
    ds = xr.Dataset(
        {
            "var": (
                ("year", "lead_time"),
                np.random.rand(2, 2),
            )
        }
    )

    cfg = DummyDataConfig(tmp_path)

    with (
        patch("glob.glob", return_value=["x.nc"]),
        patch("xarray.open_dataset", return_value=ds),
    ):
        with pytest.raises(ValueError):
            _resolve_data(cfg)


def test_resolve_data_ensemble_required_missing(tmp_path):
    cfg = DummyDataConfig(
        tmp_path,
        ensemble_list=[0],
    )

    with (
        patch("glob.glob", return_value=["x.nc"]),
        patch("xarray.open_dataset", return_value=make_valid_ds()),
    ):
        with pytest.raises(ValueError):
            _resolve_data(cfg)


@pytest.mark.pruned
def test_resolve_data_ensemble_present(tmp_path):
    cfg = DummyDataConfig(
        tmp_path,
        ensemble_list=[0],
    )

    with (
        patch("glob.glob", return_value=["x.nc"]),
        patch("xarray.open_dataset", return_value=make_ensemble_ds()),
    ):
        _resolve_data(cfg)

    assert len(cfg.list_paths) == 1


def test_resolve_data_with_rename_dict(tmp_path):
    ds = xr.Dataset(
        {
            "old": (
                ("year", "lead_time"),
                np.random.rand(2, 2),
            )
        },
        coords={
            "year": [1, 2],
            "lead_time": [1, 2],
        },
    )

    cfg = DummyDataConfig(
        tmp_path,
        names=["var"],
        rename_dict={"old": "var"},
    )

    with (
        patch("glob.glob", return_value=["x.nc"]),
        patch("xarray.open_dataset", return_value=ds),
    ):
        _resolve_data(cfg)

    assert len(cfg.list_paths) == 1


@pytest.mark.pruned
def test_get_ds_info_basic(tmp_path):
    cfg = DummyDataConfig(tmp_path)

    with patch(
        "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
        return_value=make_valid_ds(),
    ):
        info = _get_ds_info(cfg)

    assert info.start_year == 2000


@pytest.mark.pruned
def test_get_ds_info_final_year(tmp_path):
    cfg = DummyDataConfig(tmp_path)

    with patch(
        "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
        return_value=make_valid_ds(),
    ):
        info = _get_ds_info(cfg)

    assert info.final_year == 2001


@pytest.mark.pruned
def test_get_ds_info_sizes(tmp_path):
    cfg = DummyDataConfig(tmp_path)

    with patch(
        "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
        return_value=make_valid_ds(),
    ):
        info = _get_ds_info(cfg)

    assert info.sizes["year"] == 2


@pytest.mark.pruned
def test_get_ds_info_coords(tmp_path):
    cfg = DummyDataConfig(tmp_path)

    with patch(
        "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
        return_value=make_valid_ds(),
    ):
        info = _get_ds_info(cfg)

    assert info.coords["lat"] is not None


def test_get_ds_info_without_year(tmp_path):
    ds = xr.Dataset(
        {
            "var": (
                ("lat", "lon"),
                np.random.rand(2, 2),
            )
        },
        coords={
            "lat": [0, 1],
            "lon": [10, 20],
        },
    )

    cfg = DummyDataConfig(tmp_path)

    with patch(
        "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
        return_value=ds,
    ):
        info = _get_ds_info(cfg)

    assert info.start_year is None


def test_get_ds_info_with_ensemble_selection(tmp_path):
    cfg = DummyDataConfig(
        tmp_path,
        ensemble_list=[0],
    )

    with patch(
        "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
        return_value=make_ensemble_ds(),
    ):
        info = _get_ds_info(cfg)

    assert info.coords["ensembles"] is not None


@pytest.mark.pruned
def test_get_ds_info_sizes_none(tmp_path):
    ds = xr.Dataset(
        {
            "var": (
                ("lat", "lon"),
                np.random.rand(2, 2),
            )
        },
        coords={
            "lat": [0, 1],
            "lon": [10, 20],
        },
    )

    cfg = DummyDataConfig(tmp_path)

    with patch(
        "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
        return_value=ds,
    ):
        info = _get_ds_info(cfg)

    assert info.sizes is None


def test_load_preprocessor_pipeline(tmp_path):
    cfg = DummyDataConfig(tmp_path)

    cfg._load_preprocessor_pipeline(tmp_path)

    assert cfg.preprocessing_pipeline.loaded is True


def test_load_preprocessor_pipeline_not_fitted(tmp_path):
    cfg = DummyDataConfig(tmp_path)

    cfg.preprocessing_pipeline.fitted = False

    with pytest.raises(RuntimeError):
        cfg._load_preprocessor_pipeline(tmp_path)