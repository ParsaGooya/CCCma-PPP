from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import xarray as xr

from cccma_ppp.data_modules.data.data_abc import (
    DataConfigABC,
    _get_ds_info,
    _resolve_data,
    infoclass,
)
from cccma_ppp.generic.runtime import RuntimeContext

INIT_TIME_DIM = DataConfigABC.init_time_dim
LEAD_TIME_DIM = DataConfigABC.lead_time_dim
REALIZATION_DIM = DataConfigABC.realization_dim


NN_DIM = DataConfigABC.supported_NN_dimensions[0]


def make_valid_ds() -> xr.Dataset:
    return xr.Dataset(
        {
            "var": (
                (
                    INIT_TIME_DIM,
                    LEAD_TIME_DIM,
                    NN_DIM,
                ),
                np.arange(8, dtype=float).reshape(2, 2, 2),
            )
        },
        coords={
            INIT_TIME_DIM: np.array(
                [
                    "2000-01-01",
                    "2001-01-01",
                ],
                dtype="datetime64[ns]",
            ),
            LEAD_TIME_DIM: [1, 2],
            NN_DIM: [0, 1],
        },
    )


def make_ensemble_ds() -> xr.Dataset:
    return xr.Dataset(
        {
            "var": (
                (
                    REALIZATION_DIM,
                    INIT_TIME_DIM,
                    LEAD_TIME_DIM,
                    NN_DIM,
                ),
                np.arange(16, dtype=float).reshape(2, 2, 2, 2),
            )
        },
        coords={
            REALIZATION_DIM: [0, 1],
            INIT_TIME_DIM: np.array(
                [
                    "2000-01-01",
                    "2001-01-01",
                ],
                dtype="datetime64[ns]",
            ),
            LEAD_TIME_DIM: [1, 2],
            NN_DIM: [0, 1],
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
        return frozenset(
            {
                cls.init_time_dim,
                cls.lead_time_dim,
            }
        )

    @classmethod
    def _allowed_dims(cls):
        return frozenset(
            {
                cls.realization_dim,
                cls.init_time_dim,
                cls.lead_time_dim,
                *cls.supported_NN_dimensions,
            }
        )

    def __init__(
        self,
        paths,
        names=None,
        realization_list=None,
        ensemble_list=None,
        ensemble_mean=False,
        preprocessing_pipeline=None,
        rename_dict=None,
        concat_dim=None,
        file_type="*.nc",
    ):
        self.paths = Path(paths)
        self.list_paths = None
        self.names = ["var"] if names is None else names

        if realization_list is None:
            realization_list = ensemble_list

        self.realization_list = realization_list
        self.ensemble_mean = ensemble_mean
        self._check_ensemble = realization_list is not None
        self.rename_dict = {} if rename_dict is None else rename_dict
        self.concat_dim = concat_dim
        self.file_type = file_type

        self.preprocessing_pipeline = (
            DummyPipeline()
            if preprocessing_pipeline is None
            else preprocessing_pipeline
        )
        self.preprocessing_pipeline.set_name(self.TYPE)


def test_missing_preprocessing_pipeline():
    class BadConfig(DataConfigABC):
        @property
        def TYPE(self):
            return "x"

        @classmethod
        def _allowed_dims(cls):
            return frozenset()

        @classmethod
        def _required_dims(cls):
            return frozenset()

        def __init__(self):
            super().__init__()

    with pytest.raises(
        AttributeError,
        match="must define preprocessing_pipeline",
    ):
        BadConfig()


@pytest.mark.pruned
def test_data_config_class_dimensions():
    assert DummyDataConfig.init_time_dim == INIT_TIME_DIM
    assert DummyDataConfig.lead_time_dim == LEAD_TIME_DIM
    assert DummyDataConfig.realization_dim == REALIZATION_DIM


@pytest.mark.pruned
def test_resolve_data_missing_path(tmp_path):
    cfg = DummyDataConfig(tmp_path / "missing")

    with pytest.raises(FileNotFoundError):
        _resolve_data(cfg)


@pytest.mark.pruned
def test_resolve_data_empty_directory(tmp_path):
    cfg = DummyDataConfig(tmp_path)

    with patch(
        "cccma_ppp.data_modules.data.data_abc.glob.glob",
        return_value=[],
    ):
        with pytest.raises(FileNotFoundError):
            _resolve_data(cfg)


@pytest.mark.pruned
def test_resolve_data_valid(tmp_path):
    cfg = DummyDataConfig(tmp_path)

    with (
        patch(
            "cccma_ppp.data_modules.data.data_abc.glob.glob",
            return_value=["x.nc"],
        ),
        patch(
            "cccma_ppp.data_modules.data.data_abc.xr.open_dataset",
            return_value=make_valid_ds(),
        ),
    ):
        _resolve_data(cfg)

    assert cfg.list_paths == ["x.nc"]


def test_resolve_data_skip_checks_branch(tmp_path):
    cfg = DummyDataConfig(tmp_path)

    with (
        patch(
            "cccma_ppp.data_modules.data.data_abc.glob.glob",
            return_value=["file1.nc"],
        ),
        patch(
            "cccma_ppp.data_modules.data.data_abc.xr.open_dataset",
        ) as mock_open,
    ):
        _resolve_data(
            cfg,
            _do_checks=False,
        )

    assert cfg.list_paths == ["file1.nc"]
    mock_open.assert_not_called()


def test_resolve_data_missing_required_dims(tmp_path):
    ds = xr.Dataset(
        {
            "var": (
                (NN_DIM,),
                [1.0, 2.0],
            )
        },
        coords={
            NN_DIM: [0, 1],
        },
    )

    cfg = DummyDataConfig(tmp_path)

    with (
        patch(
            "cccma_ppp.data_modules.data.data_abc.glob.glob",
            return_value=["x.nc"],
        ),
        patch(
            "cccma_ppp.data_modules.data.data_abc.xr.open_dataset",
            return_value=ds,
        ),
    ):
        with pytest.raises(ValueError):
            _resolve_data(cfg)


def test_resolve_data_init_time_coordinate_is_accepted(tmp_path):
    ds = xr.Dataset(
        {
            "var": (
                (
                    LEAD_TIME_DIM,
                    NN_DIM,
                ),
                np.arange(4, dtype=float).reshape(2, 2),
            )
        },
        coords={
            INIT_TIME_DIM: np.datetime64("2000-01-01"),
            LEAD_TIME_DIM: [1, 2],
            NN_DIM: [0, 1],
        },
    )

    cfg = DummyDataConfig(tmp_path)

    with (
        patch(
            "cccma_ppp.data_modules.data.data_abc.glob.glob",
            return_value=["x.nc"],
        ),
        patch(
            "cccma_ppp.data_modules.data.data_abc.xr.open_dataset",
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
                    INIT_TIME_DIM,
                    LEAD_TIME_DIM,
                    NN_DIM,
                    "bad",
                ),
                np.zeros((2, 2, 2, 2)),
            )
        },
        coords={
            INIT_TIME_DIM: np.array(
                ["2000-01-01", "2001-01-01"],
                dtype="datetime64[ns]",
            ),
            LEAD_TIME_DIM: [1, 2],
            NN_DIM: [0, 1],
            "bad": [1, 2],
        },
    )

    cfg = DummyDataConfig(tmp_path)

    with (
        patch(
            "cccma_ppp.data_modules.data.data_abc.glob.glob",
            return_value=["x.nc"],
        ),
        patch(
            "cccma_ppp.data_modules.data.data_abc.xr.open_dataset",
            return_value=ds,
        ),
    ):
        with pytest.raises(
            ValueError,
            match="invalid data dimensions",
        ):
            _resolve_data(cfg)


@pytest.mark.pruned
def test_resolve_data_missing_variable(tmp_path):
    ds = make_valid_ds().rename_vars({"var": "other"})
    cfg = DummyDataConfig(tmp_path)

    with (
        patch(
            "cccma_ppp.data_modules.data.data_abc.glob.glob",
            return_value=["x.nc"],
        ),
        patch(
            "cccma_ppp.data_modules.data.data_abc.xr.open_dataset",
            return_value=ds,
        ),
    ):
        with pytest.raises(
            ValueError,
            match="missing variables",
        ):
            _resolve_data(cfg)


def test_resolve_data_missing_coords(tmp_path):
    ds = xr.Dataset(
        {
            "var": (
                (
                    INIT_TIME_DIM,
                    LEAD_TIME_DIM,
                    NN_DIM,
                ),
                np.zeros((2, 2, 2)),
            )
        }
    )

    cfg = DummyDataConfig(tmp_path)

    with (
        patch(
            "cccma_ppp.data_modules.data.data_abc.glob.glob",
            return_value=["x.nc"],
        ),
        patch(
            "cccma_ppp.data_modules.data.data_abc.xr.open_dataset",
            return_value=ds,
        ),
    ):
        with pytest.raises(ValueError):
            _resolve_data(cfg)


def test_resolve_data_realization_required_missing(tmp_path):
    cfg = DummyDataConfig(
        tmp_path,
        realization_list=[0],
    )

    with (
        patch(
            "cccma_ppp.data_modules.data.data_abc.glob.glob",
            return_value=["x.nc"],
        ),
        patch(
            "cccma_ppp.data_modules.data.data_abc.xr.open_dataset",
            return_value=make_valid_ds(),
        ),
    ):
        with pytest.raises(
            ValueError,
            match="Cannot select realization_list",
        ):
            _resolve_data(cfg)


def test_resolve_data_realization_present(tmp_path):
    cfg = DummyDataConfig(
        tmp_path,
        realization_list=[0],
    )

    with (
        patch(
            "cccma_ppp.data_modules.data.data_abc.glob.glob",
            return_value=["x.nc"],
        ),
        patch(
            "cccma_ppp.data_modules.data.data_abc.xr.open_dataset",
            return_value=make_ensemble_ds(),
        ),
    ):
        _resolve_data(cfg)

    assert cfg.list_paths == ["x.nc"]


@pytest.mark.pruned
def test_resolve_data_no_supported_nn_dimensions(tmp_path):
    ds = xr.Dataset(
        {
            "var": (
                (
                    INIT_TIME_DIM,
                    LEAD_TIME_DIM,
                ),
                np.zeros((2, 2)),
            )
        },
        coords={
            INIT_TIME_DIM: np.array(
                ["2000-01-01", "2001-01-01"],
                dtype="datetime64[ns]",
            ),
            LEAD_TIME_DIM: [1, 2],
        },
    )

    cfg = DummyDataConfig(tmp_path)

    with (
        patch(
            "cccma_ppp.data_modules.data.data_abc.glob.glob",
            return_value=["x.nc"],
        ),
        patch(
            "cccma_ppp.data_modules.data.data_abc.xr.open_dataset",
            return_value=ds,
        ),
    ):
        with pytest.raises(
            ValueError,
            match="supported NN dimensions",
        ):
            _resolve_data(cfg)


@pytest.mark.pruned
def test_resolve_data_invalid_time_type(tmp_path):
    ds = make_valid_ds().assign_coords(
        {
            INIT_TIME_DIM: [2000, 2001],
        }
    )
    cfg = DummyDataConfig(tmp_path)

    with (
        patch(
            "cccma_ppp.data_modules.data.data_abc.glob.glob",
            return_value=["x.nc"],
        ),
        patch(
            "cccma_ppp.data_modules.data.data_abc.xr.open_dataset",
            return_value=ds,
        ),
    ):
        with pytest.raises(
            TypeError,
            match="must contain",
        ):
            _resolve_data(cfg)


@pytest.mark.pruned
def test_resolve_data_multiple_files(tmp_path):
    cfg = DummyDataConfig(tmp_path)

    with (
        patch(
            "cccma_ppp.data_modules.data.data_abc.glob.glob",
            return_value=[
                "a.nc",
                "b.nc",
            ],
        ),
        patch(
            "cccma_ppp.data_modules.data.data_abc.xr.open_dataset",
            side_effect=[
                make_valid_ds(),
                make_valid_ds(),
            ],
        ),
    ):
        _resolve_data(cfg)

    assert cfg.list_paths == [
        "a.nc",
        "b.nc",
    ]


@pytest.mark.pruned
def test_resolve_data_applies_rename_dict(tmp_path):
    ds = make_valid_ds().rename(
        {
            INIT_TIME_DIM: "old_time",
        }
    )

    cfg = DummyDataConfig(
        tmp_path,
        rename_dict={
            "old_time": INIT_TIME_DIM,
        },
    )

    with (
        patch(
            "cccma_ppp.data_modules.data.data_abc.glob.glob",
            return_value=["x.nc"],
        ),
        patch(
            "cccma_ppp.data_modules.data.data_abc.xr.open_dataset",
            return_value=ds,
        ),
    ):
        _resolve_data(cfg)

    assert cfg.list_paths == ["x.nc"]


@pytest.mark.pruned
def test_get_ds_info_basic(tmp_path):
    cfg = DummyDataConfig(tmp_path)

    with patch(
        "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
        return_value=make_valid_ds(),
    ):
        info = _get_ds_info(cfg)

    assert isinstance(info, infoclass)
    assert info.start_time == np.datetime64("2000-01-01T00:00:00.000000000")


@pytest.mark.pruned
def test_get_ds_info_final_time(tmp_path):
    cfg = DummyDataConfig(tmp_path)

    with patch(
        "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
        return_value=make_valid_ds(),
    ):
        info = _get_ds_info(cfg)

    assert info.final_time == np.datetime64("2001-01-01T00:00:00.000000000")


@pytest.mark.pruned
def test_get_ds_info_sizes(tmp_path):
    cfg = DummyDataConfig(tmp_path)

    with patch(
        "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
        return_value=make_valid_ds(),
    ):
        info = _get_ds_info(cfg)

    assert info.sizes[INIT_TIME_DIM] == 2
    assert info.sizes[LEAD_TIME_DIM] == 2
    assert NN_DIM not in info.sizes


@pytest.mark.pruned
def test_get_ds_info_coords(tmp_path):
    cfg = DummyDataConfig(tmp_path)

    with patch(
        "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
        return_value=make_valid_ds(),
    ):
        info = _get_ds_info(cfg)

    assert info.coords[NN_DIM] is not None
    assert info.coords[INIT_TIME_DIM] is not None
    assert info.coords[LEAD_TIME_DIM] is not None


@pytest.mark.pruned
def test_get_ds_info_time_metadata(tmp_path):
    cfg = DummyDataConfig(tmp_path)

    with patch(
        "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
        return_value=make_valid_ds(),
    ):
        info = _get_ds_info(cfg)

    assert info.time_coords_type is not None
    assert info.init_time_freq is not None


@pytest.mark.pruned
def test_get_ds_info_time_helpers_are_used(tmp_path):
    cfg = DummyDataConfig(tmp_path)
    expected_type = object()
    expected_frequency = object()

    with (
        patch(
            "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
            return_value=make_valid_ds(),
        ),
        patch(
            "cccma_ppp.data_modules.data.data_abc.get_time_representation",
            return_value=expected_type,
        ) as mock_representation,
        patch(
            "cccma_ppp.data_modules.data.data_abc.infer_time_resolution",
            return_value=expected_frequency,
        ) as mock_resolution,
    ):
        info = _get_ds_info(cfg)

    assert info.time_coords_type is expected_type
    assert info.init_time_freq is expected_frequency
    mock_representation.assert_called_once()
    mock_resolution.assert_called_once()


def test_get_ds_info_requires_init_time_coordinate(tmp_path):
    ds = xr.Dataset(
        {
            "var": (
                (NN_DIM,),
                [1.0, 2.0],
            )
        },
        coords={
            NN_DIM: [0, 1],
        },
    )

    cfg = DummyDataConfig(tmp_path)

    with patch(
        "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
        return_value=ds,
    ):
        with pytest.raises(KeyError):
            _get_ds_info(cfg)


@pytest.mark.pruned
def test_get_ds_info_with_realization_selection(tmp_path):
    cfg = DummyDataConfig(
        tmp_path,
        realization_list=[0],
    )

    with patch(
        "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
        return_value=make_ensemble_ds(),
    ):
        info = _get_ds_info(cfg)

    assert info.sizes[REALIZATION_DIM] == 1
    assert info.coords[REALIZATION_DIM].values.tolist() == [0]


@pytest.mark.pruned
def test_get_ds_info_sizes_none(tmp_path):
    ds = (
        make_valid_ds()
        .isel(
            {
                INIT_TIME_DIM: 0,
                LEAD_TIME_DIM: 0,
            },
            drop=True,
        )
        .expand_dims(
            {
                INIT_TIME_DIM: np.array(
                    ["2000-01-01", "2001-01-01"],
                    dtype="datetime64[ns]",
                )
            }
        )
    )

    cfg = DummyDataConfig(tmp_path)

    with patch(
        "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
        return_value=ds,
    ):
        info = _get_ds_info(cfg)

    assert info.sizes == {
        INIT_TIME_DIM: 2,
    }


@pytest.mark.pruned
def test_get_ds_info_uses_existing_list_paths(tmp_path):
    cfg = DummyDataConfig(tmp_path)
    cfg.list_paths = ["already_set.nc"]

    with (
        patch(
            "cccma_ppp.data_modules.data.data_abc.glob.glob",
        ) as mock_glob,
        patch(
            "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
            return_value=make_valid_ds(),
        ) as mock_load,
    ):
        _get_ds_info(cfg)

    mock_glob.assert_not_called()
    assert mock_load.call_args.args[0] == ["already_set.nc"]


@pytest.mark.pruned
def test_get_ds_info_resolves_paths_when_not_set(tmp_path):
    cfg = DummyDataConfig(tmp_path)
    cfg.list_paths = None

    with (
        patch(
            "cccma_ppp.data_modules.data.data_abc.glob.glob",
            return_value=["resolved.nc"],
        ) as mock_glob,
        patch(
            "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
            return_value=make_valid_ds(),
        ) as mock_load,
    ):
        _get_ds_info(cfg)

    mock_glob.assert_called_once()
    assert mock_load.call_args.args[0] == ["resolved.nc"]


@pytest.mark.pruned
def test_get_ds_info_without_realization_selection(tmp_path):
    cfg = DummyDataConfig(tmp_path)

    with patch(
        "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
        return_value=make_valid_ds(),
    ):
        info = _get_ds_info(cfg)

    assert INIT_TIME_DIM in info.coords
    assert REALIZATION_DIM not in info.coords


@pytest.mark.pruned
def test_get_ds_info_coord_contents(tmp_path):
    cfg = DummyDataConfig(tmp_path)

    with patch(
        "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
        return_value=make_valid_ds(),
    ):
        info = _get_ds_info(cfg)

    assert set(info.coords) >= {
        INIT_TIME_DIM,
        LEAD_TIME_DIM,
        NN_DIM,
    }


@pytest.mark.pruned
def test_fit_preprocessor_pipeline(tmp_path):
    cfg = DummyDataConfig(tmp_path)
    cfg.list_paths = ["x.nc"]

    selection = {
        INIT_TIME_DIM: slice(
            np.datetime64("2000-01-01"),
            np.datetime64("2001-01-01"),
        )
    }

    with patch(
        "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
        return_value=make_valid_ds(),
    ) as mock_load:
        cfg.fit_preprocessor_pipeline(
            selection=selection,
            mask=False,
            save=False,
        )

    assert cfg.preprocessing_pipeline.fit_called is True
    assert cfg.preprocessing_pipeline.fit_kwargs["mask"] is None
    assert cfg.preprocessing_pipeline.fit_kwargs["save"] is False

    mock_load.assert_called_once_with(
        ["x.nc"],
        names=["var"],
        concat_dim=None,
        selection=selection,
        ensemble_mean=False,
        rename_dict={},
    )


@pytest.mark.pruned
def test_fit_preprocessor_pipeline_with_mask(tmp_path):
    cfg = DummyDataConfig(tmp_path)
    cfg.list_paths = ["x.nc"]

    expected_mask = xr.DataArray(
        np.ones((2, 2), dtype=bool),
        dims=(
            INIT_TIME_DIM,
            LEAD_TIME_DIM,
        ),
        coords={
            INIT_TIME_DIM: np.array(
                ["2000-01-01", "2001-01-01"],
                dtype="datetime64[ns]",
            ),
            LEAD_TIME_DIM: [1, 2],
        },
    )

    with (
        patch(
            "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
            return_value=make_valid_ds(),
        ),
        patch(
            "cccma_ppp.data_modules.data.data_abc._create_train_mask",
            return_value=expected_mask,
        ) as mock_mask,
    ):
        cfg.fit_preprocessor_pipeline(
            selection={},
            mask=True,
            save=True,
            save_path=tmp_path,
            save_name="pipeline.joblib",
        )

    assert cfg.preprocessing_pipeline.fit_called is True
    assert cfg.preprocessing_pipeline.fit_kwargs["mask"] is expected_mask
    assert cfg.preprocessing_pipeline.fit_kwargs["save"] is True
    assert cfg.preprocessing_pipeline.fit_kwargs["save_path"] == tmp_path
    assert cfg.preprocessing_pipeline.fit_kwargs["save_name"] == "pipeline.joblib"

    mock_mask.assert_called_once()


@pytest.mark.pruned
def test_load_preprocessor_pipeline(tmp_path):
    cfg = DummyDataConfig(tmp_path)

    cfg.load_preprocessor_pipeline(tmp_path)

    assert cfg.preprocessing_pipeline.loaded is True
    assert cfg.preprocessing_pipeline.loaded_path == (
        tmp_path / "dummy_preprocessing_pipeline.joblib"
    )


def test_load_preprocessor_pipeline_not_fitted(tmp_path):
    cfg = DummyDataConfig(tmp_path)
    cfg.preprocessing_pipeline.fitted = False

    with pytest.raises(
        RuntimeError,
        match="is not fitted",
    ):
        cfg.load_preprocessor_pipeline(tmp_path)


def test_load_preprocessor_pipeline_default_path(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        RuntimeContext,
        "GLOBAL_EXP_DIR",
        tmp_path,
        raising=False,
    )

    cfg = DummyDataConfig(tmp_path)
    captured = {}

    def fake_load(path):
        captured["path"] = Path(path)

    cfg.preprocessing_pipeline.load_from_memory = fake_load

    cfg.load_preprocessor_pipeline()

    assert captured["path"] == (
        tmp_path / "preprocessing_pipeline" / "dummy_preprocessing_pipeline.joblib"
    )


@pytest.mark.pruned
def test_load_preprocessor_pipeline_custom_path(tmp_path):
    cfg = DummyDataConfig(tmp_path)
    captured = {}

    def fake_load(path):
        captured["path"] = Path(path)

    cfg.preprocessing_pipeline.load_from_memory = fake_load

    cfg.load_preprocessor_pipeline(
        load_dir=tmp_path,
    )

    assert captured["path"] == (tmp_path / "dummy_preprocessing_pipeline.joblib")


@pytest.mark.pruned
def test_load_preprocessor_pipeline_fitted_success(tmp_path):
    cfg = DummyDataConfig(tmp_path)
    cfg.preprocessing_pipeline.fitted = True
    cfg.preprocessing_pipeline.load_from_memory = lambda path: None

    cfg.load_preprocessor_pipeline(tmp_path)


class TestDataConfigInitialization:
    @pytest.mark.pruned
    def test_base_initialization_sets_pipeline_name_and_state(
        self,
        tmp_path,
    ):
        pipeline = Mock()
        pipeline.set_name = Mock()

        class InitializedConfig(DummyDataConfig):
            def __init__(self):
                self.paths = tmp_path
                self.names = ["var"]
                self.preprocessing_pipeline = pipeline
                self.realization_list = None
                self.ensemble_mean = False
                self.concat_dim = None
                self.file_type = "*.nc"
                self.rename_dict = {
                    "old": "new",
                }

                DataConfigABC.__init__(self)

        expected_info = infoclass(
            sizes={},
            start_time=None,
            final_time=None,
            coords={},
            dims=(),
            time_coords_type=None,
            init_time_freq=None,
        )

        with (
            patch(
                "cccma_ppp.data_modules.data.data_abc._resolve_data",
            ) as resolve,
            patch(
                "cccma_ppp.data_modules.data.data_abc._get_ds_info",
                return_value=expected_info,
            ) as get_info,
        ):
            config = InitializedConfig()

        assert config.data is None
        assert config._check_ensemble is False
        assert config.info is expected_info

        pipeline.set_name.assert_called_once_with(
            "dummy",
            {
                "old": "new",
            },
        )
        resolve.assert_called_once_with(config)
        get_info.assert_called_once_with(config)

    @pytest.mark.pruned
    def test_base_initialization_enables_ensemble_check(
        self,
        tmp_path,
    ):
        pipeline = Mock()

        class EnsembleConfig(DummyDataConfig):
            def __init__(self):
                self.paths = tmp_path
                self.names = ["var"]
                self.preprocessing_pipeline = pipeline
                self.realization_list = [
                    1,
                    2,
                ]
                self.ensemble_mean = False
                self.concat_dim = None
                self.file_type = "*.nc"
                self.rename_dict = None

                DataConfigABC.__init__(self)

        with (
            patch(
                "cccma_ppp.data_modules.data.data_abc._resolve_data",
            ),
            patch(
                "cccma_ppp.data_modules.data.data_abc._get_ds_info",
                return_value=Mock(),
            ),
        ):
            config = EnsembleConfig()

        assert config._check_ensemble is True


class TestDataConfigProperties:
    @pytest.mark.pruned
    def test_coords_delegates_to_info(
        self,
        tmp_path,
    ):
        config = DummyDataConfig(tmp_path)
        expected = {
            INIT_TIME_DIM: make_valid_ds()[INIT_TIME_DIM],
        }
        config.info = Mock(
            coords=expected,
        )

        assert config.coords is expected

    @pytest.mark.pruned
    def test_sizes_delegates_to_info(
        self,
        tmp_path,
    ):
        config = DummyDataConfig(tmp_path)
        expected = {
            INIT_TIME_DIM: 2,
            LEAD_TIME_DIM: 3,
        }
        config.info = Mock(
            sizes=expected,
        )

        assert config.sizes is expected

    @pytest.mark.pruned
    def test_dims_delegates_to_info(
        self,
        tmp_path,
    ):
        config = DummyDataConfig(tmp_path)
        expected = (
            INIT_TIME_DIM,
            LEAD_TIME_DIM,
            NN_DIM,
        )
        config.info = Mock(
            dims=expected,
        )

        assert config.dims is expected

    @pytest.mark.pruned
    def test_init_time_frequency_delegates_to_info(
        self,
        tmp_path,
    ):
        config = DummyDataConfig(tmp_path)
        config.info = Mock(
            init_time_freq="year",
        )

        assert config.init_time_frequency == "year"


class TestOpenXarrayData:
    @pytest.mark.pruned
    def test_opens_data_without_realization_selection(
        self,
        tmp_path,
    ):
        config = DummyDataConfig(tmp_path)
        config.list_paths = [
            "first.nc",
            "second.nc",
        ]
        config.info = Mock(
            coords={
                INIT_TIME_DIM: make_valid_ds()[INIT_TIME_DIM],
            },
        )

        expected = make_valid_ds()

        with patch(
            "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
            return_value=expected,
        ) as load_data:
            result = config.open_xarray_data(
                load=True,
                add_time_auxiliary_coords=True,
            )

        assert result is config
        assert config.data is expected

        load_data.assert_called_once_with(
            [
                "first.nc",
                "second.nc",
            ],
            names=["var"],
            ensemble_mean=False,
            selection=None,
            concat_dim=None,
            rename_dict={},
            add_time_auxiliary_coords=True,
            load=True,
        )

    @pytest.mark.pruned
    def test_opens_data_with_realization_selection(
        self,
        tmp_path,
    ):
        config = DummyDataConfig(
            tmp_path,
            realization_list=[
                0,
                1,
            ],
        )
        config.list_paths = [
            "ensemble.nc",
        ]

        realization_coordinate = make_ensemble_ds()[REALIZATION_DIM]
        config.info = Mock(
            coords={
                REALIZATION_DIM: realization_coordinate,
            },
        )

        expected = make_ensemble_ds()

        with patch(
            "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
            return_value=expected,
        ) as load_data:
            result = config.open_xarray_data()

        assert result is config

        assert load_data.call_args.kwargs["selection"] == {
            REALIZATION_DIM: realization_coordinate,
        }

    @pytest.mark.pruned
    def test_open_uses_default_flags(
        self,
        tmp_path,
    ):
        config = DummyDataConfig(tmp_path)
        config.list_paths = [
            "data.nc",
        ]
        config.info = Mock(
            coords={},
        )

        with patch(
            "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
            return_value=make_valid_ds(),
        ) as load_data:
            config.open_xarray_data()

        assert load_data.call_args.kwargs["load"] is False
        assert load_data.call_args.kwargs["add_time_auxiliary_coords"] is False


class TestDataConfigIndexing:
    @pytest.mark.pruned
    def test_isel_requires_opened_data(
        self,
        tmp_path,
    ):
        config = DummyDataConfig(tmp_path)
        config.data = None

        with pytest.raises(
            ValueError,
            match="No data is currently opened",
        ):
            config.isel(
                {
                    INIT_TIME_DIM: 0,
                }
            )

    def test_isel_forwards_indexer_mapping(
        self,
        tmp_path,
    ):
        config = DummyDataConfig(tmp_path)
        config.data = make_valid_ds()

        transformed = object()
        config.preprocessing_pipeline.transform = Mock(
            return_value=transformed,
        )

        result = config.isel(
            {
                INIT_TIME_DIM: 1,
                LEAD_TIME_DIM: 0,
            }
        )

        expected_selection = config.data.isel(
            {
                INIT_TIME_DIM: 1,
                LEAD_TIME_DIM: 0,
            }
        )

        passed = config.preprocessing_pipeline.transform.call_args.args[0]

        xr.testing.assert_identical(
            passed,
            expected_selection,
        )
        assert result is transformed

    @pytest.mark.pruned
    def test_isel_forwards_keyword_indexers(
        self,
        tmp_path,
    ):
        config = DummyDataConfig(tmp_path)
        config.data = make_valid_ds()

        config.preprocessing_pipeline.transform = Mock(
            side_effect=lambda data: data,
        )

        result = config.isel(
            **{
                INIT_TIME_DIM: 0,
                LEAD_TIME_DIM: 1,
            }
        )

        expected = config.data.isel(
            {
                INIT_TIME_DIM: 0,
                LEAD_TIME_DIM: 1,
            }
        )

        xr.testing.assert_identical(
            result,
            expected,
        )

    def test_indexes_requires_opened_data(
        self,
        tmp_path,
    ):
        config = DummyDataConfig(tmp_path)
        config.data = None

        with pytest.raises(
            ValueError,
            match="No data is currently opened",
        ):
            _ = config.indexes


class TestDataConfigClosing:
    @pytest.mark.pruned
    def test_close_data_closes_dataset_and_clears_reference(
        self,
        tmp_path,
    ):
        config = DummyDataConfig(tmp_path)
        data = Mock()
        config.data = data

        result = config.close_data()

        assert result is None
        data.close.assert_called_once_with()
        assert config.data is None

    @pytest.mark.pruned
    def test_close_data_is_idempotent(
        self,
        tmp_path,
    ):
        config = DummyDataConfig(tmp_path)
        config.data = None

        assert config.close_data() is None
        assert config.data is None

    @pytest.mark.pruned
    def test_close_data_can_be_called_twice(
        self,
        tmp_path,
    ):
        config = DummyDataConfig(tmp_path)
        data = Mock()
        config.data = data

        config.close_data()
        config.close_data()

        data.close.assert_called_once_with()
        assert config.data is None

    @pytest.mark.pruned
    def test_destructor_closes_open_data(
        self,
        tmp_path,
    ):
        config = DummyDataConfig(tmp_path)
        data = Mock()
        config.data = data

        config.__del__()

        data.close.assert_called_once_with()
        assert config.data is None


class TestFitPreprocessorPipelineCleanup:
    @pytest.mark.pruned
    def test_fit_loads_base_data_before_passing_to_pipeline(
        self,
        tmp_path,
    ):
        config = DummyDataConfig(tmp_path)
        config.list_paths = [
            "data.nc",
        ]

        base = Mock()
        loaded_base = object()
        base.load.return_value = loaded_base

        with patch(
            "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
            return_value=base,
        ):
            config.fit_preprocessor_pipeline(
                selection={},
                save=False,
            )

        base.load.assert_called_once_with()

        assert config.preprocessing_pipeline.fit_kwargs["base_data"] is loaded_base

    @pytest.mark.pruned
    def test_fit_closes_base_dataset(
        self,
        tmp_path,
    ):
        config = DummyDataConfig(tmp_path)
        config.list_paths = [
            "data.nc",
        ]

        base = Mock()
        base.load.return_value = base

        with patch(
            "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
            return_value=base,
        ):
            config.fit_preprocessor_pipeline(
                selection={},
                save=False,
            )

        base.close.assert_called_once_with()

    @pytest.mark.pruned
    def test_fit_invokes_garbage_collection(
        self,
        tmp_path,
    ):
        config = DummyDataConfig(tmp_path)
        config.list_paths = [
            "data.nc",
        ]

        base = Mock()
        base.load.return_value = base

        with (
            patch(
                "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
                return_value=base,
            ),
            patch(
                "cccma_ppp.data_modules.data.data_abc.gc.collect",
            ) as collect,
        ):
            config.fit_preprocessor_pipeline(
                selection={},
                save=False,
            )

        collect.assert_called_once_with()

    @pytest.mark.pruned
    def test_mask_builder_receives_time_coordinates(
        self,
        tmp_path,
    ):
        config = DummyDataConfig(tmp_path)
        config.list_paths = [
            "data.nc",
        ]
        base = make_valid_ds()
        expected_mask = object()

        with (
            patch(
                "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
                return_value=base,
            ),
            patch(
                "cccma_ppp.data_modules.data.data_abc._create_train_mask",
                return_value=expected_mask,
            ) as create_mask,
        ):
            config.fit_preprocessor_pipeline(
                selection={},
                mask=True,
                save=False,
            )

        init_time_argument = create_mask.call_args.args[0]
        lead_time_argument = create_mask.call_args.args[1]

        xr.testing.assert_identical(
            init_time_argument,
            base[INIT_TIME_DIM],
        )
        xr.testing.assert_identical(
            lead_time_argument,
            base[LEAD_TIME_DIM],
        )

    @pytest.mark.pruned
    def test_mask_builder_is_not_called_when_mask_disabled(
        self,
        tmp_path,
    ):
        config = DummyDataConfig(tmp_path)
        config.list_paths = [
            "data.nc",
        ]

        with (
            patch(
                "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
                return_value=make_valid_ds(),
            ),
            patch(
                "cccma_ppp.data_modules.data.data_abc._create_train_mask",
            ) as create_mask,
        ):
            config.fit_preprocessor_pipeline(
                selection={},
                mask=False,
                save=False,
            )

        create_mask.assert_not_called()


class TestAdditionalResolveData:
    @pytest.mark.pruned
    def test_resolve_data_uses_configured_file_pattern(
        self,
        tmp_path,
    ):
        config = DummyDataConfig(
            tmp_path,
            file_type="*.zarr",
        )

        with (
            patch(
                "cccma_ppp.data_modules.data.data_abc.glob.glob",
                return_value=[
                    "data.zarr",
                ],
            ) as resolve_paths,
            patch(
                "cccma_ppp.data_modules.data.data_abc.xr.open_dataset",
                return_value=make_valid_ds(),
            ),
        ):
            _resolve_data(config)

        expected_pattern = str(tmp_path / "*.zarr")
        resolve_paths.assert_called_once_with(expected_pattern)

    @pytest.mark.pruned
    def test_resolve_data_opens_each_path_as_path_object(
        self,
        tmp_path,
    ):
        config = DummyDataConfig(tmp_path)

        with (
            patch(
                "cccma_ppp.data_modules.data.data_abc.glob.glob",
                return_value=[
                    "first.nc",
                    "second.nc",
                ],
            ),
            patch(
                "cccma_ppp.data_modules.data.data_abc.xr.open_dataset",
                side_effect=[
                    make_valid_ds(),
                    make_valid_ds(),
                ],
            ) as open_dataset,
        ):
            _resolve_data(config)

        assert open_dataset.call_args_list == [
            ((Path("first.nc"),),),
            ((Path("second.nc"),),),
        ]

    @pytest.mark.pruned
    def test_resolve_data_validates_time_sequence(
        self,
        tmp_path,
    ):
        config = DummyDataConfig(tmp_path)
        dataset = make_valid_ds()

        with (
            patch(
                "cccma_ppp.data_modules.data.data_abc.glob.glob",
                return_value=[
                    "data.nc",
                ],
            ),
            patch(
                "cccma_ppp.data_modules.data.data_abc.xr.open_dataset",
                return_value=dataset,
            ),
            patch(
                "cccma_ppp.data_modules.data.data_abc._validate_time_sequence",
            ) as validate,
        ):
            _resolve_data(config)

        validate.assert_called_once()

        xr.testing.assert_identical(
            validate.call_args.args[0],
            dataset.coords[INIT_TIME_DIM],
        )

    @pytest.mark.pruned
    def test_resolve_data_invokes_garbage_collection_per_file(
        self,
        tmp_path,
    ):
        config = DummyDataConfig(tmp_path)

        with (
            patch(
                "cccma_ppp.data_modules.data.data_abc.glob.glob",
                return_value=[
                    "first.nc",
                    "second.nc",
                ],
            ),
            patch(
                "cccma_ppp.data_modules.data.data_abc.xr.open_dataset",
                side_effect=[
                    make_valid_ds(),
                    make_valid_ds(),
                ],
            ),
            patch(
                "cccma_ppp.data_modules.data.data_abc.gc.collect",
            ) as collect,
        ):
            _resolve_data(config)

        assert collect.call_count == 2

    @pytest.mark.pruned
    def test_resolve_data_error_does_not_assign_list_paths(
        self,
        tmp_path,
    ):
        config = DummyDataConfig(
            tmp_path,
            names=[
                "missing",
            ],
        )

        with (
            patch(
                "cccma_ppp.data_modules.data.data_abc.glob.glob",
                return_value=[
                    "data.nc",
                ],
            ),
            patch(
                "cccma_ppp.data_modules.data.data_abc.xr.open_dataset",
                return_value=make_valid_ds(),
            ),
        ):
            with pytest.raises(
                ValueError,
                match="missing variables",
            ):
                _resolve_data(config)

        assert config.list_paths is None

    def test_resolve_data_error_reports_missing_path(
        self,
        tmp_path,
    ):
        missing = tmp_path / "missing"
        config = DummyDataConfig(missing)

        with pytest.raises(
            FileNotFoundError,
            match=str(missing),
        ):
            _resolve_data(config)


class TestAdditionalGetDatasetInfo:
    @pytest.mark.pruned
    def test_load_receives_exact_arguments_without_realizations(
        self,
        tmp_path,
    ):
        config = DummyDataConfig(tmp_path)
        config.list_paths = [
            "data.nc",
        ]

        with patch(
            "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
            return_value=make_valid_ds(),
        ) as load_data:
            _get_ds_info(config)

        load_data.assert_called_once_with(
            [
                "data.nc",
            ],
            names=[
                "var",
            ],
            selection=None,
            concat_dim=None,
            rename_dict={},
        )

    def test_load_receives_realization_selection(
        self,
        tmp_path,
    ):
        config = DummyDataConfig(
            tmp_path,
            realization_list=[
                1,
            ],
        )
        config.list_paths = [
            "ensemble.nc",
        ]

        with patch(
            "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
            return_value=make_ensemble_ds(),
        ) as load_data:
            _get_ds_info(config)

        assert load_data.call_args.kwargs["selection"] == {
            REALIZATION_DIM: [
                1,
            ],
        }

    @pytest.mark.pruned
    def test_info_dimensions_match_dataset_dimensions(
        self,
        tmp_path,
    ):
        config = DummyDataConfig(tmp_path)
        dataset = make_valid_ds()

        with patch(
            "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
            return_value=dataset,
        ):
            info = _get_ds_info(config)

        assert info.dims == dataset.dims

    @pytest.mark.pruned
    def test_info_coordinates_preserve_dataarrays(
        self,
        tmp_path,
    ):
        config = DummyDataConfig(tmp_path)
        dataset = make_valid_ds()

        with patch(
            "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
            return_value=dataset,
        ):
            info = _get_ds_info(config)

        for dimension in (
            INIT_TIME_DIM,
            LEAD_TIME_DIM,
            NN_DIM,
        ):
            xr.testing.assert_identical(
                info.coords[dimension],
                dataset.coords[dimension],
            )

    @pytest.mark.pruned
    def test_get_ds_info_start_and_end_use_minimum_and_maximum(
        self,
        tmp_path,
    ):
        dataset = make_valid_ds().assign_coords(
            {
                INIT_TIME_DIM: np.asarray(
                    [
                        "2002-01-01",
                        "2000-01-01",
                    ],
                    dtype="datetime64[ns]",
                )
            }
        )
        config = DummyDataConfig(tmp_path)

        with patch(
            "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
            return_value=dataset,
        ):
            info = _get_ds_info(config)

        assert info.start_time == np.datetime64("2000-01-01T00:00:00.000000000")
        assert info.final_time == np.datetime64("2002-01-01T00:00:00.000000000")

    @pytest.mark.pruned
    def test_get_ds_info_includes_realization_size_without_selection(
        self,
        tmp_path,
    ):
        config = DummyDataConfig(tmp_path)

        with patch(
            "cccma_ppp.data_modules.data.data_abc._load_xarray_data",
            return_value=make_ensemble_ds(),
        ):
            info = _get_ds_info(config)

        assert info.sizes == {
            REALIZATION_DIM: 2,
            INIT_TIME_DIM: 2,
            LEAD_TIME_DIM: 2,
        }
