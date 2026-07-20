from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from cccma_ppp.data_modules.data.data_abc import (
    DataConfigABC,
    _get_ds_info,
    _resolve_data,
)
from cccma_ppp.generic.runtime import RuntimeContext


def make_valid_ds():
    return xr.Dataset(
        {
            "var": (
                (
                    "year",
                    "lead_time",
                    "lat",
                    "lon",
                ),
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
                (
                    "ensembles",
                    "year",
                    "lead_time",
                    "lat",
                    "lon",
                ),
                np.random.rand(
                    2,
                    2,
                    2,
                    2,
                    2,
                ),
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
        self.name = "dummy"
        self.fitted = True
        self.loaded = False
        self.loaded_path = None
        self.fit_called = False
        self.fit_args = None
        self.fit_kwargs = None
        self.save_called = False
        self.save_args = None
        self.save_kwargs = None

    def set_name(self, name):
        self.name = name
        return self

    def load_from_memory(self, path):
        self.loaded = True
        self.loaded_path = Path(path)
        return self

    def fit(self, *args, **kwargs):
        self.fit_called = True
        self.fit_args = args
        self.fit_kwargs = kwargs
        self.fitted = True
        return self

    def save(self, *args, **kwargs):
        self.save_called = True
        self.save_args = args
        self.save_kwargs = kwargs
        return self


class DummyDataConfig(DataConfigABC):
    @property
    def TYPE(self):
        return "dummy"

    @classmethod
    def _required_dims(cls):
        return {
            "year",
            "lead_time",
        }

    @classmethod
    def _allowed_dims(cls):
        return {
            "ensembles",
            "year",
            "lead_time",
            "channels",
            "lat",
            "lon",
        }

    def __init__(
        self,
        paths,
        names=None,
        ensemble_list=None,
        ensemble_mean=False,
        preprocessing_pipeline=None,
        rename_dict=None,
        concat_dim=None,
        file_type="*.nc",
    ):
        # The production code calls Path(self.paths), so this
        # must be one path rather than a list of paths.
        self.paths = Path(paths)

        # This is populated by _resolve_data after globbing.
        self.list_paths = None

        self.names = ["var"] if names is None else names
        self.ensemble_list = ensemble_list
        self.ensemble_mean = ensemble_mean
        self._check_ensemble = ensemble_list is not None
        self.rename_dict = {} if rename_dict is None else rename_dict
        self.concat_dim = concat_dim
        self.file_type = file_type

        self.preprocessing_pipeline = (
            DummyPipeline()
            if preprocessing_pipeline is None
            else preprocessing_pipeline
        )
        self.preprocessing_pipeline.name = self.TYPE


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


def test_pipeline_name_set(tmp_path):
    cfg = DummyDataConfig(tmp_path)

    assert cfg.preprocessing_pipeline.name == "dummy"


def test_resolve_data_missing_path(tmp_path):
    cfg = DummyDataConfig(
        tmp_path / "missing",
    )

    with pytest.raises(FileNotFoundError):
        _resolve_data(cfg)


def test_resolve_data_empty_directory(tmp_path):
    cfg = DummyDataConfig(tmp_path)

    with pytest.raises(FileNotFoundError):
        _resolve_data(cfg)


def test_resolve_data_valid(tmp_path):
    cfg = DummyDataConfig(tmp_path)

    with (
        patch(
            "glob.glob",
            return_value=["x.nc"],
        ),
        patch(
            "xarray.open_dataset",
            return_value=make_valid_ds(),
        ),
    ):
        _resolve_data(cfg)

    assert cfg.list_paths == ["x.nc"]


def test_resolve_data_missing_required_dims(tmp_path):
    ds = xr.Dataset(
        {
            "var": (
                ("x",),
                [1, 2],
            )
        },
        coords={
            "x": [0, 1],
        },
    )

    cfg = DummyDataConfig(tmp_path)

    with (
        patch(
            "glob.glob",
            return_value=["x.nc"],
        ),
        patch(
            "xarray.open_dataset",
            return_value=ds,
        ),
    ):
        with pytest.raises(ValueError):
            _resolve_data(cfg)


def test_resolve_data_invalid_dims(tmp_path):
    ds = xr.Dataset(
        {
            "var": (
                (
                    "year",
                    "lead_time",
                    "bad",
                ),
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
        patch(
            "glob.glob",
            return_value=["x.nc"],
        ),
        patch(
            "xarray.open_dataset",
            return_value=ds,
        ),
    ):
        with pytest.raises(ValueError):
            _resolve_data(cfg)


def test_resolve_data_missing_variable(tmp_path):
    ds = xr.Dataset(
        {
            "other": (
                (
                    "year",
                    "lead_time",
                ),
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
        patch(
            "glob.glob",
            return_value=["x.nc"],
        ),
        patch(
            "xarray.open_dataset",
            return_value=ds,
        ),
    ):
        with pytest.raises(ValueError):
            _resolve_data(cfg)


def test_resolve_data_missing_coords(tmp_path):
    ds = xr.Dataset(
        {
            "var": (
                (
                    "year",
                    "lead_time",
                ),
                np.random.rand(2, 2),
            )
        }
    )

    cfg = DummyDataConfig(tmp_path)

    with (
        patch(
            "glob.glob",
            return_value=["x.nc"],
        ),
        patch(
            "xarray.open_dataset",
            return_value=ds,
        ),
    ):
        with pytest.raises(ValueError):
            _resolve_data(cfg)


def test_resolve_data_ensemble_required_missing(tmp_path):
    cfg = DummyDataConfig(
        tmp_path,
        ensemble_list=[0],
    )

    with (
        patch(
            "glob.glob",
            return_value=["x.nc"],
        ),
        patch(
            "xarray.open_dataset",
            return_value=make_valid_ds(),
        ),
    ):
        with pytest.raises(ValueError):
            _resolve_data(cfg)


def test_resolve_data_ensemble_present(tmp_path):
    cfg = DummyDataConfig(
        tmp_path,
        ensemble_list=[0],
    )

    with (
        patch(
            "glob.glob",
            return_value=["x.nc"],
        ),
        patch(
            "xarray.open_dataset",
            return_value=make_ensemble_ds(),
        ),
    ):
        _resolve_data(cfg)

    assert cfg.list_paths == ["x.nc"]


def test_resolve_data_with_rename_dict(tmp_path):
    ds = xr.Dataset(
        {
            "old": (
                (
                    "year",
                    "lead_time",
                ),
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
        rename_dict={
            "old": "var",
        },
    )

    with (
        patch(
            "glob.glob",
            return_value=["x.nc"],
        ),
        patch(
            "xarray.open_dataset",
            return_value=ds,
        ),
    ):
        _resolve_data(cfg)

    assert cfg.list_paths == ["x.nc"]


def test_get_ds_info_basic(tmp_path):
    cfg = DummyDataConfig(tmp_path)

    with patch(
        "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
        return_value=make_valid_ds(),
    ):
        info = _get_ds_info(cfg)

    assert info.start_year == 2000


def test_get_ds_info_final_year(tmp_path):
    cfg = DummyDataConfig(tmp_path)

    with patch(
        "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
        return_value=make_valid_ds(),
    ):
        info = _get_ds_info(cfg)

    assert info.final_year == 2001


def test_get_ds_info_sizes(tmp_path):
    cfg = DummyDataConfig(tmp_path)

    with patch(
        "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
        return_value=make_valid_ds(),
    ):
        info = _get_ds_info(cfg)

    assert info.sizes["year"] == 2


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
                (
                    "lat",
                    "lon",
                ),
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


def test_get_ds_info_sizes_none(tmp_path):
    ds = xr.Dataset(
        {
            "var": (
                (
                    "lat",
                    "lon",
                ),
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

    cfg._load_preprocessor_pipeline(
        tmp_path,
    )

    assert cfg.preprocessing_pipeline.loaded is True
    assert cfg.preprocessing_pipeline.loaded_path == (
        tmp_path / "dummy_preprocessing_pipeline.joblib"
    )


def test_load_preprocessor_pipeline_not_fitted(tmp_path):
    cfg = DummyDataConfig(tmp_path)
    cfg.preprocessing_pipeline.fitted = False

    with pytest.raises(RuntimeError):
        cfg._load_preprocessor_pipeline(
            tmp_path,
        )


def test_method_wrapper_resolve_data(tmp_path):
    cfg = DummyDataConfig(tmp_path)

    with patch("cccma_ppp.data_modules.data.data_abc._resolve_data") as mock_resolve:
        cfg._resolve_data()

    mock_resolve.assert_called_once_with(cfg)


def test_method_wrapper_get_ds_info(tmp_path):
    cfg = DummyDataConfig(tmp_path)

    with patch(
        "cccma_ppp.data_modules.data.data_abc._get_ds_info",
        return_value="info",
    ) as mock_info:
        result = cfg._get_ds_info()

    assert result == "info"
    mock_info.assert_called_once_with(cfg)


def test_resolve_data_skip_checks_branch(tmp_path):
    cfg = DummyDataConfig(tmp_path)

    with patch(
        "glob.glob",
        return_value=["file1.nc"],
    ):
        _resolve_data(
            cfg,
            _do_checks=False,
        )

    assert cfg.list_paths == ["file1.nc"]


def test_resolve_data_no_supported_nn_dimensions(
    tmp_path,
):
    ds = xr.Dataset(
        {
            "var": (
                (
                    "year",
                    "lead_time",
                ),
                np.random.rand(2, 2),
            )
        },
        coords={
            "year": [2000, 2001],
            "lead_time": [1, 2],
        },
    )
    cfg = DummyDataConfig(tmp_path)

    with (
        patch(
            "glob.glob",
            return_value=["x.nc"],
        ),
        patch(
            "xarray.open_dataset",
            return_value=ds,
        ),
    ):
        with pytest.raises(
            ValueError,
            match="supported NN dimensions",
        ):
            _resolve_data(cfg)


def test_resolve_data_multiple_files(tmp_path):
    cfg = DummyDataConfig(tmp_path)

    with (
        patch(
            "glob.glob",
            return_value=[
                "a.nc",
                "b.nc",
            ],
        ),
        patch(
            "xarray.open_dataset",
            return_value=make_valid_ds(),
        ),
    ):
        _resolve_data(cfg)

    assert cfg.list_paths == [
        "a.nc",
        "b.nc",
    ]


def test_get_ds_info_uses_existing_list_paths(
    tmp_path,
):
    cfg = DummyDataConfig(tmp_path)
    cfg.list_paths = ["already_set.nc"]

    with (
        patch(
            "glob.glob",
        ) as mock_glob,
        patch(
            "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
            return_value=make_valid_ds(),
        ),
    ):
        _get_ds_info(cfg)

    mock_glob.assert_not_called()


def test_get_ds_info_without_ensemble_selection(
    tmp_path,
):
    cfg = DummyDataConfig(tmp_path)

    with patch(
        "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
        return_value=make_valid_ds(),
    ):
        info = _get_ds_info(cfg)

    assert "year" in info.coords


def test_get_ds_info_coord_contents(tmp_path):
    cfg = DummyDataConfig(tmp_path)

    with patch(
        "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
        return_value=make_valid_ds(),
    ):
        info = _get_ds_info(cfg)

    assert set(info.coords) >= {
        "year",
        "lead_time",
    }


def test_fit_preprocessor_pipeline_no_mask(
    tmp_path,
):
    cfg = DummyDataConfig(tmp_path)

    class DummyDS:
        year = np.array([2000, 2001])
        lead_time = np.array([1, 2])

        def load(self):
            return self

        def close(self):
            return None

    with patch(
        "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
        return_value=DummyDS(),
    ):
        cfg._fit_preprocessor_pipeline(
            selection={},
        )

    assert cfg.preprocessing_pipeline.fit_called is True


def test_fit_preprocessor_pipeline_with_mask(
    tmp_path,
):
    cfg = DummyDataConfig(tmp_path)

    class DummyDS:
        year = np.array([2000, 2001])
        lead_time = np.array([1, 2])

        def load(self):
            return self

        def close(self):
            return None

    with (
        patch(
            "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
            return_value=DummyDS(),
        ),
        patch(
            "cccma_ppp.data_modules.data.data_abc._create_train_mask",
            return_value="MASK",
        ) as mask_mock,
    ):
        cfg._fit_preprocessor_pipeline(
            selection={},
            mask=True,
        )

    mask_mock.assert_called_once()
    assert cfg.preprocessing_pipeline.fit_called is True


def test_fit_preprocessor_pipeline_save_args(
    tmp_path,
):
    cfg = DummyDataConfig(tmp_path)
    captured = {}

    class DummyDS:
        year = np.array([1, 2])
        lead_time = np.array([1, 2])

        def load(self):
            return self

        def close(self):
            return None

    def fake_fit(*args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)

    cfg.preprocessing_pipeline.fit = fake_fit

    with patch(
        "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
        return_value=DummyDS(),
    ):
        cfg._fit_preprocessor_pipeline(
            selection={},
            save=False,
            save_path="abc",
            save_name="xyz",
        )

    assert captured["save"] is False
    assert captured["save_path"] == "abc"
    assert captured["save_name"] == "xyz"


def test_load_preprocessor_pipeline_default_path(
    tmp_path,
):
    RuntimeContext.GLOBAL_EXP_DIR = str(tmp_path)

    cfg = DummyDataConfig(tmp_path)
    captured = {}

    def fake_load(path):
        captured["path"] = Path(path)

    cfg.preprocessing_pipeline.load_from_memory = fake_load

    cfg._load_preprocessor_pipeline()

    assert captured["path"].name == ("dummy_preprocessing_pipeline.joblib")
    assert captured["path"].parent == (tmp_path / "preprocessing_pipeline")


def test_load_preprocessor_pipeline_custom_path(
    tmp_path,
):
    cfg = DummyDataConfig(tmp_path)
    captured = {}

    def fake_load(path):
        captured["path"] = Path(path)

    cfg.preprocessing_pipeline.load_from_memory = fake_load

    cfg._load_preprocessor_pipeline(
        load_dir=tmp_path,
    )

    assert captured["path"] == (tmp_path / "dummy_preprocessing_pipeline.joblib")


def test_load_preprocessor_pipeline_fitted_success(
    tmp_path,
):
    cfg = DummyDataConfig(tmp_path)
    cfg.preprocessing_pipeline.fitted = True
    cfg.preprocessing_pipeline.load_from_memory = lambda path: None

    cfg._load_preprocessor_pipeline(tmp_path)
