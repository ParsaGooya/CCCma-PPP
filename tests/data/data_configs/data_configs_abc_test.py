import re
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import xarray as xr

from S2D4P.data.data_configs.data_configs_abc import (
    DataConfigABC,
    _get_ds_info,
    _resolve_data,
    infoclass,
)
from S2D4P.generic.runtime import RuntimeContext

ABC = "S2D4P.data.data_configs.data_configs_abc"

INIT_TIME_DIM = DataConfigABC.init_time_dim
LEAD_TIME_DIM = DataConfigABC.lead_time_dim
REALIZATION_DIM = DataConfigABC.realization_dim
NN_DIM = DataConfigABC.supported_NN_dimensions[0]


def make_valid_ds() -> xr.Dataset:
    return xr.Dataset(
        {
            "var": (
                (INIT_TIME_DIM, LEAD_TIME_DIM, NN_DIM),
                np.arange(8, dtype=float).reshape(2, 2, 2),
            )
        },
        coords={
            INIT_TIME_DIM: np.array(
                ["2000-01-01", "2001-01-01"], dtype="datetime64[ns]"
            ),
            LEAD_TIME_DIM: [1, 2],
            NN_DIM: [0, 1],
        },
    )


def make_ensemble_ds() -> xr.Dataset:
    return xr.Dataset(
        {
            "var": (
                (REALIZATION_DIM, INIT_TIME_DIM, LEAD_TIME_DIM, NN_DIM),
                np.arange(16, dtype=float).reshape(2, 2, 2, 2),
            )
        },
        coords={
            REALIZATION_DIM: [0, 1],
            INIT_TIME_DIM: np.array(
                ["2000-01-01", "2001-01-01"], dtype="datetime64[ns]"
            ),
            LEAD_TIME_DIM: [1, 2],
            NN_DIM: [0, 1],
        },
    )


def make_static_ds() -> xr.Dataset:
    """A dataset with spatial dimensions only and no time coordinate."""
    return xr.Dataset(
        {"var": ((NN_DIM,), [1.0, 2.0])},
        coords={NN_DIM: [0, 1]},
    )


class DummyPipeline:
    def __init__(self):
        self.name = "dummy"
        self.fitted = True
        self.loaded = False
        self.loaded_path = None
        self.fit_called = False
        self.fit_kwargs = None

    def set_name(self, name, *args):
        self.name = name
        return self

    def load_from_memory(self, path):
        self.loaded = True
        self.loaded_path = Path(path)
        return self

    def fit(self, *args, **kwargs):
        self.fit_called = True
        self.fit_kwargs = kwargs
        self.fitted = True
        return self


class DummyDataConfig(DataConfigABC):
    @property
    def TYPE(self):
        return "dummy"

    @classmethod
    def _required_dims(cls):
        return frozenset({cls.init_time_dim, cls.lead_time_dim})

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
        self.data = None
        self.names = ["var"] if names is None else names

        if realization_list is None:
            realization_list = ensemble_list

        self.realization_list = realization_list
        self.ensemble_mean = ensemble_mean
        self._check_ensemble = realization_list is not None
        self.rename_dict = {} if rename_dict is None else rename_dict
        self.concat_dim = concat_dim
        self.file_type = file_type
        self.drop_vars_list = None

        self.preprocessing_pipeline = (
            DummyPipeline()
            if preprocessing_pipeline is None
            else preprocessing_pipeline
        )
        self.preprocessing_pipeline.set_name(self.TYPE)


def patch_data_source(info=None):
    """Stub the module-level loaders so a config can be built without files."""
    if info is None:
        info = Mock()
    return patch.multiple(
        ABC,
        _resolve_data=lambda config, **kwargs: None,
        _get_ds_info=lambda config: info,
    )


def run_resolve(config, dataset, files=("x.nc",)):
    """Call `_resolve_data` with file discovery and xarray opening stubbed out."""
    open_kwargs = (
        {"side_effect": dataset}
        if isinstance(dataset, list)
        else {"return_value": dataset}
    )
    with (
        patch(f"{ABC}.glob.glob", return_value=list(files)),
        patch(f"{ABC}.xr.open_dataset", **open_kwargs),
    ):
        _resolve_data(config)


def test_missing_preprocessing_pipeline_raises():
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

    with pytest.raises(AttributeError, match="must define preprocessing_pipeline"):
        BadConfig()


@pytest.mark.pruned
def test_initialization_sets_pipeline_name_and_state():
    pipeline = Mock()
    expected_info = infoclass(
        sizes={},
        start_time=None,
        final_time=None,
        coords={},
        dims=(),
        time_coords_type=None,
        init_time_freq=None,
    )

    class InitializedConfig(DummyDataConfig):
        def __init__(self, paths):
            self.paths = paths
            self.names = ["var"]
            self.preprocessing_pipeline = pipeline
            self.realization_list = None
            self.ensemble_mean = False
            self.concat_dim = None
            self.file_type = "*.nc"
            self.rename_dict = {"old": "new"}
            self.drop_vars_list = None
            DataConfigABC.__init__(self)

    with (
        patch(f"{ABC}._resolve_data") as resolve,
        patch(f"{ABC}._get_ds_info", return_value=expected_info) as get_info,
    ):
        config = InitializedConfig(Path("x"))

    assert config.data is None
    assert config._check_ensemble is False
    assert config.info is expected_info
    pipeline.set_name.assert_called_once_with("dummy", {"old": "new"})
    resolve.assert_called_once_with(config)
    get_info.assert_called_once_with(config)


@pytest.mark.pruned
def test_initialization_enables_ensemble_check_when_realizations_given():
    class EnsembleConfig(DummyDataConfig):
        def __init__(self, paths):
            self.paths = paths
            self.names = ["var"]
            self.preprocessing_pipeline = Mock()
            self.realization_list = [1, 2]
            self.ensemble_mean = False
            self.concat_dim = None
            self.file_type = "*.nc"
            self.rename_dict = None
            self.drop_vars_list = None
            DataConfigABC.__init__(self)

    with (
        patch(f"{ABC}._resolve_data"),
        patch(f"{ABC}._get_ds_info", return_value=Mock()),
    ):
        config = EnsembleConfig(Path("x"))

    assert config._check_ensemble is True


@pytest.mark.pruned
def test_data_config_class_dimensions():
    assert DummyDataConfig.init_time_dim == INIT_TIME_DIM
    assert DummyDataConfig.lead_time_dim == LEAD_TIME_DIM
    assert DummyDataConfig.realization_dim == REALIZATION_DIM


@pytest.mark.pruned
def test_resolve_data_missing_path_raises(tmp_path):
    config = DummyDataConfig(tmp_path / "missing")

    with pytest.raises(FileNotFoundError):
        _resolve_data(config)


def test_resolve_data_missing_path_error_names_the_path(tmp_path):
    missing = tmp_path / "missing"
    config = DummyDataConfig(missing)

    with pytest.raises(FileNotFoundError, match=re.escape(str(missing))):
        _resolve_data(config)


@pytest.mark.pruned
def test_resolve_data_empty_directory_raises(tmp_path):
    config = DummyDataConfig(tmp_path)

    with patch(f"{ABC}.glob.glob", return_value=[]):
        with pytest.raises(FileNotFoundError):
            _resolve_data(config)


@pytest.mark.pruned
def test_resolve_data_valid_file_sets_list_paths(tmp_path):
    config = DummyDataConfig(tmp_path)

    run_resolve(config, make_valid_ds())

    assert config.list_paths == ["x.nc"]


def test_resolve_data_skip_checks_only_discovers_files(tmp_path):
    config = DummyDataConfig(tmp_path)

    with (
        patch(f"{ABC}.glob.glob", return_value=["file1.nc"]),
        patch(f"{ABC}.xr.open_dataset") as open_dataset,
    ):
        _resolve_data(config, _do_checks=False)

    assert config.list_paths == ["file1.nc"]
    open_dataset.assert_not_called()


@pytest.mark.pruned
def test_resolve_data_uses_configured_file_pattern(tmp_path):
    config = DummyDataConfig(tmp_path, file_type="*.zarr")

    with (
        patch(f"{ABC}.glob.glob", return_value=["data.zarr"]) as glob_mock,
        patch(f"{ABC}.xr.open_dataset", return_value=make_valid_ds()),
    ):
        _resolve_data(config)

    glob_mock.assert_called_once_with(str(tmp_path / "*.zarr"))


@pytest.mark.pruned
def test_resolve_data_opens_each_file_as_path(tmp_path):
    config = DummyDataConfig(tmp_path)

    with (
        patch(f"{ABC}.glob.glob", return_value=["first.nc", "second.nc"]),
        patch(
            f"{ABC}.xr.open_dataset",
            side_effect=[make_valid_ds(), make_valid_ds()],
        ) as open_dataset,
    ):
        _resolve_data(config)

    assert open_dataset.call_args_list == [
        ((Path("first.nc"),),),
        ((Path("second.nc"),),),
    ]


@pytest.mark.pruned
def test_resolve_data_accepts_multiple_valid_files(tmp_path):
    config = DummyDataConfig(tmp_path)

    run_resolve(
        config,
        [make_valid_ds(), make_valid_ds()],
        files=("a.nc", "b.nc"),
    )

    assert config.list_paths == ["a.nc", "b.nc"]


@pytest.mark.pruned
def test_resolve_data_rejects_missing_required_dims(tmp_path):
    config = DummyDataConfig(tmp_path)

    with pytest.raises(ValueError):
        run_resolve(config, make_static_ds())


@pytest.mark.pruned
def test_resolve_data_rejects_scalar_init_time_coordinate(tmp_path):
    # Init time as a scalar coordinate is not a dimension, so the required
    # init-time dimension is missing and validation must fail.
    ds = xr.Dataset(
        {"var": ((LEAD_TIME_DIM, NN_DIM), np.arange(4, dtype=float).reshape(2, 2))},
        coords={
            INIT_TIME_DIM: np.datetime64("2000-01-01"),
            LEAD_TIME_DIM: [1, 2],
            NN_DIM: [0, 1],
        },
    )
    config = DummyDataConfig(tmp_path)

    with pytest.raises(ValueError):
        run_resolve(config, ds)


def test_resolve_data_rejects_invalid_dims(tmp_path):
    ds = xr.Dataset(
        {
            "var": (
                (INIT_TIME_DIM, LEAD_TIME_DIM, NN_DIM, "bad"),
                np.zeros((2, 2, 2, 2)),
            )
        },
        coords={
            INIT_TIME_DIM: np.array(
                ["2000-01-01", "2001-01-01"], dtype="datetime64[ns]"
            ),
            LEAD_TIME_DIM: [1, 2],
            NN_DIM: [0, 1],
            "bad": [1, 2],
        },
    )
    config = DummyDataConfig(tmp_path)

    with pytest.raises(ValueError, match="invalid data dimensions"):
        run_resolve(config, ds)


@pytest.mark.pruned
def test_resolve_data_rejects_missing_variable(tmp_path):
    ds = make_valid_ds().rename_vars({"var": "other"})
    config = DummyDataConfig(tmp_path)

    with pytest.raises(ValueError, match="missing variables"):
        run_resolve(config, ds)


@pytest.mark.pruned
def test_resolve_data_reports_all_missing_variables(tmp_path):
    config = DummyDataConfig(tmp_path, names=["missing_a", "missing_b"])

    with pytest.raises(ValueError, match=r"missing_a.*missing_b"):
        run_resolve(config, make_valid_ds())


@pytest.mark.pruned
def test_resolve_data_rejects_missing_coordinates(tmp_path):
    ds = xr.Dataset(
        {"var": ((INIT_TIME_DIM, LEAD_TIME_DIM, NN_DIM), np.zeros((2, 2, 2)))}
    )
    config = DummyDataConfig(tmp_path)

    with pytest.raises(ValueError):
        run_resolve(config, ds)


@pytest.mark.pruned
def test_resolve_data_rejects_non_datetime_init_time(tmp_path):
    ds = make_valid_ds().assign_coords({INIT_TIME_DIM: [2000, 2001]})
    config = DummyDataConfig(tmp_path)

    with pytest.raises(TypeError, match="must contain"):
        run_resolve(config, ds)


def test_resolve_data_rejects_no_supported_nn_dimension(tmp_path):
    ds = xr.Dataset(
        {"var": ((INIT_TIME_DIM, LEAD_TIME_DIM), np.zeros((2, 2)))},
        coords={
            INIT_TIME_DIM: np.array(
                ["2000-01-01", "2001-01-01"], dtype="datetime64[ns]"
            ),
            LEAD_TIME_DIM: [1, 2],
        },
    )
    config = DummyDataConfig(tmp_path)

    with pytest.raises(ValueError, match="supported NN dimensions"):
        run_resolve(config, ds)


def test_resolve_data_rejects_realization_list_without_realization_dim(tmp_path):
    config = DummyDataConfig(tmp_path, realization_list=[0])

    with pytest.raises(ValueError, match="Cannot select realization_list"):
        run_resolve(config, make_valid_ds())


def test_resolve_data_accepts_realization_list_with_realization_dim(tmp_path):
    config = DummyDataConfig(tmp_path, realization_list=[0])

    run_resolve(config, make_ensemble_ds())

    assert config.list_paths == ["x.nc"]


@pytest.mark.pruned
def test_resolve_data_applies_rename_dict(tmp_path):
    ds = make_valid_ds().rename({INIT_TIME_DIM: "old_time"})
    config = DummyDataConfig(tmp_path, rename_dict={"old_time": INIT_TIME_DIM})

    run_resolve(config, ds)

    assert config.list_paths == ["x.nc"]


def test_resolve_data_applies_drop_vars_before_validation(tmp_path):
    dataset = make_valid_ds().assign(removable=((INIT_TIME_DIM,), [1.0, 2.0]))
    config = DummyDataConfig(tmp_path)
    config.drop_vars_list = ["removable"]

    with (
        patch(f"{ABC}.glob.glob", return_value=["data.nc"]),
        patch(f"{ABC}.xr.open_dataset", return_value=dataset),
        patch.object(
            xr.Dataset,
            "drop_vars",
            wraps=dataset.drop_vars,
        ) as drop_vars,
    ):
        _resolve_data(config)

    drop_vars.assert_called_once_with(["removable"], errors="ignore")
    assert config.list_paths == ["data.nc"]


@pytest.mark.pruned
def test_resolve_data_ignores_missing_drop_variable(tmp_path):
    config = DummyDataConfig(tmp_path)
    config.drop_vars_list = ["not_present"]

    run_resolve(config, make_valid_ds(), files=("data.nc",))

    assert config.list_paths == ["data.nc"]


def test_resolve_data_allows_missing_concat_dim_when_coordinate_exists(tmp_path):
    dataset = xr.Dataset(
        {"var": ((LEAD_TIME_DIM, NN_DIM), np.ones((2, 2)))},
        coords={
            INIT_TIME_DIM: np.datetime64("2000-01-01"),
            LEAD_TIME_DIM: [1, 2],
            NN_DIM: [0, 1],
        },
    )
    config = DummyDataConfig(tmp_path, concat_dim=INIT_TIME_DIM)

    run_resolve(config, dataset, files=("data.nc",))

    assert config.list_paths == ["data.nc"]


def test_resolve_data_rejects_missing_dim_that_is_not_the_concat_dim(tmp_path):
    dataset = xr.Dataset(
        {"var": ((LEAD_TIME_DIM, NN_DIM), np.ones((2, 2)))},
        coords={
            INIT_TIME_DIM: np.datetime64("2000-01-01"),
            LEAD_TIME_DIM: [1, 2],
            NN_DIM: [0, 1],
        },
    )
    config = DummyDataConfig(tmp_path, concat_dim=LEAD_TIME_DIM)

    with pytest.raises(ValueError, match="missing required dimensions"):
        run_resolve(config, dataset, files=("data.nc",))


@pytest.mark.pruned
def test_resolve_data_rejects_dimension_without_coordinate(tmp_path):
    dataset = xr.Dataset(
        {"var": ((INIT_TIME_DIM, LEAD_TIME_DIM, NN_DIM), np.ones((2, 2, 2)))},
        coords={
            INIT_TIME_DIM: np.asarray(
                ["2000-01-01", "2001-01-01"], dtype="datetime64[ns]"
            ),
            NN_DIM: [0, 1],
        },
    )
    config = DummyDataConfig(tmp_path)

    with pytest.raises(ValueError, match="coordinates"):
        run_resolve(config, dataset, files=("data.nc",))


def test_resolve_data_rejects_concat_dim_dimension_without_coordinate(tmp_path):
    dataset = xr.Dataset(
        {"var": ((INIT_TIME_DIM, LEAD_TIME_DIM, NN_DIM), np.ones((2, 2, 2)))},
        coords={
            INIT_TIME_DIM: np.asarray(
                ["2000-01-01", "2001-01-01"], dtype="datetime64[ns]"
            ),
            NN_DIM: [0, 1],
        },
    )
    config = DummyDataConfig(tmp_path, concat_dim=LEAD_TIME_DIM)

    with pytest.raises(ValueError, match="coordinates"):
        run_resolve(config, dataset, files=("data.nc",))


@pytest.mark.pruned
def test_resolve_data_skips_time_validation_without_time_coordinate(tmp_path):
    class StaticConfig(DummyDataConfig):
        @classmethod
        def _required_dims(cls):
            return frozenset()

    config = StaticConfig(tmp_path)

    with (
        patch(f"{ABC}.glob.glob", return_value=["static.nc"]),
        patch(f"{ABC}.xr.open_dataset", return_value=make_static_ds()),
        patch(f"{ABC}._validate_time_sequence") as validate,
    ):
        _resolve_data(config)

    validate.assert_not_called()
    assert config.list_paths == ["static.nc"]


@pytest.mark.pruned
def test_resolve_data_validates_time_sequence_of_init_time(tmp_path):
    dataset = make_valid_ds()
    config = DummyDataConfig(tmp_path)

    with (
        patch(f"{ABC}.glob.glob", return_value=["data.nc"]),
        patch(f"{ABC}.xr.open_dataset", return_value=dataset),
        patch(f"{ABC}._validate_time_sequence") as validate,
    ):
        _resolve_data(config)

    validate.assert_called_once()
    xr.testing.assert_identical(
        validate.call_args.args[0],
        dataset.coords[INIT_TIME_DIM],
    )


@pytest.mark.pruned
def test_resolve_data_does_not_run_garbage_collection(tmp_path):
    config = DummyDataConfig(tmp_path)

    with (
        patch(f"{ABC}.glob.glob", return_value=["first.nc", "second.nc"]),
        patch(
            f"{ABC}.xr.open_dataset",
            side_effect=[make_valid_ds(), make_valid_ds()],
        ),
        patch(f"{ABC}.gc.collect") as collect,
    ):
        _resolve_data(config)

    collect.assert_not_called()


@pytest.mark.pruned
def test_resolve_data_failure_does_not_assign_list_paths(tmp_path):
    config = DummyDataConfig(tmp_path, names=["missing"])

    with pytest.raises(ValueError, match="missing variables"):
        run_resolve(config, make_valid_ds(), files=("data.nc",))

    assert config.list_paths is None


@pytest.mark.pruned
def test_get_ds_info_returns_infoclass():
    config = DummyDataConfig(Path("."))

    with patch(f"{ABC}._load_xarray_data", return_value=make_valid_ds()):
        info = _get_ds_info(config)

    assert isinstance(info, infoclass)


@pytest.mark.pruned
def test_get_ds_info_start_time_is_minimum_init_time():
    config = DummyDataConfig(Path("."))

    with patch(f"{ABC}._load_xarray_data", return_value=make_valid_ds()):
        info = _get_ds_info(config)

    assert info.start_time == np.datetime64("2000-01-01T00:00:00.000000000")


@pytest.mark.pruned
def test_get_ds_info_final_time_is_maximum_init_time():
    config = DummyDataConfig(Path("."))

    with patch(f"{ABC}._load_xarray_data", return_value=make_valid_ds()):
        info = _get_ds_info(config)

    assert info.final_time == np.datetime64("2001-01-01T00:00:00.000000000")


@pytest.mark.pruned
def test_get_ds_info_start_and_end_use_min_and_max_of_unsorted_times():
    dataset = make_valid_ds().assign_coords(
        {
            INIT_TIME_DIM: np.asarray(
                ["2002-01-01", "2000-01-01"], dtype="datetime64[ns]"
            )
        }
    )
    config = DummyDataConfig(Path("."))

    with patch(f"{ABC}._load_xarray_data", return_value=dataset):
        info = _get_ds_info(config)

    assert info.start_time == np.datetime64("2000-01-01T00:00:00.000000000")
    assert info.final_time == np.datetime64("2002-01-01T00:00:00.000000000")


@pytest.mark.pruned
def test_get_ds_info_sizes_include_init_and_lead_dims_only():
    config = DummyDataConfig(Path("."))

    with patch(f"{ABC}._load_xarray_data", return_value=make_valid_ds()):
        info = _get_ds_info(config)

    assert info.sizes[INIT_TIME_DIM] == 2
    assert info.sizes[LEAD_TIME_DIM] == 2
    assert NN_DIM not in info.sizes


@pytest.mark.pruned
def test_get_ds_info_sizes_include_realization_without_selection():
    config = DummyDataConfig(Path("."))

    with patch(f"{ABC}._load_xarray_data", return_value=make_ensemble_ds()):
        info = _get_ds_info(config)

    assert info.sizes == {
        REALIZATION_DIM: 2,
        INIT_TIME_DIM: 2,
        LEAD_TIME_DIM: 2,
    }


def test_get_ds_info_sizes_none_when_no_relevant_dims():
    config = DummyDataConfig(Path("."))

    with patch(f"{ABC}._load_xarray_data", return_value=make_static_ds()):
        info = _get_ds_info(config)

    assert info.sizes is None


@pytest.mark.pruned
def test_get_ds_info_sizes_only_init_when_lead_dim_is_removed():
    ds = (
        make_valid_ds()
        .isel({INIT_TIME_DIM: 0, LEAD_TIME_DIM: 0}, drop=True)
        .expand_dims(
            {
                INIT_TIME_DIM: np.array(
                    ["2000-01-01", "2001-01-01"], dtype="datetime64[ns]"
                )
            }
        )
    )
    config = DummyDataConfig(Path("."))

    with patch(f"{ABC}._load_xarray_data", return_value=ds):
        info = _get_ds_info(config)

    assert info.sizes == {INIT_TIME_DIM: 2}


@pytest.mark.pruned
def test_get_ds_info_coords_contain_all_dataset_coords():
    config = DummyDataConfig(Path("."))

    with patch(f"{ABC}._load_xarray_data", return_value=make_valid_ds()):
        info = _get_ds_info(config)

    assert set(info.coords) >= {INIT_TIME_DIM, LEAD_TIME_DIM, NN_DIM}
    assert info.coords[NN_DIM] is not None


@pytest.mark.pruned
def test_get_ds_info_coords_preserve_dataarrays():
    dataset = make_valid_ds()
    config = DummyDataConfig(Path("."))

    with patch(f"{ABC}._load_xarray_data", return_value=dataset):
        info = _get_ds_info(config)

    for dim in (INIT_TIME_DIM, LEAD_TIME_DIM, NN_DIM):
        xr.testing.assert_identical(info.coords[dim], dataset.coords[dim])


@pytest.mark.pruned
def test_get_ds_info_dims_match_dataset_dims():
    dataset = make_valid_ds()
    config = DummyDataConfig(Path("."))

    with patch(f"{ABC}._load_xarray_data", return_value=dataset):
        info = _get_ds_info(config)

    assert info.dims == dataset.dims


@pytest.mark.pruned
def test_get_ds_info_time_metadata_comes_from_helpers():
    config = DummyDataConfig(Path("."))
    expected_type = object()
    expected_frequency = object()

    with (
        patch(f"{ABC}._load_xarray_data", return_value=make_valid_ds()),
        patch(
            f"{ABC}.get_time_representation",
            return_value=expected_type,
        ) as representation,
        patch(
            f"{ABC}.infer_time_resolution",
            return_value=expected_frequency,
        ) as resolution,
    ):
        info = _get_ds_info(config)

    assert info.time_coords_type is expected_type
    assert info.init_time_freq is expected_frequency
    representation.assert_called_once()
    resolution.assert_called_once()


@pytest.mark.pruned
def test_get_ds_info_static_dataset_has_no_temporal_metadata():
    config = DummyDataConfig(Path("."))

    with (
        patch(f"{ABC}._load_xarray_data", return_value=make_static_ds()),
        patch(f"{ABC}.get_time_representation") as representation,
        patch(f"{ABC}.infer_time_resolution") as resolution,
    ):
        info = _get_ds_info(config)

    assert info.start_time is None
    assert info.final_time is None
    assert info.time_coords_type is None
    assert info.init_time_freq is None
    representation.assert_not_called()
    resolution.assert_not_called()


@pytest.mark.pruned
def test_get_ds_info_applies_realization_selection():
    config = DummyDataConfig(Path("."), realization_list=[1])

    with patch(f"{ABC}._load_xarray_data", return_value=make_ensemble_ds()):
        info = _get_ds_info(config)

    assert info.sizes[REALIZATION_DIM] == 1
    assert info.coords[REALIZATION_DIM].values.tolist() == [1]


@pytest.mark.pruned
def test_get_ds_info_keeps_multiple_selected_realizations():
    config = DummyDataConfig(Path("."), realization_list=[0, 1])

    with patch(f"{ABC}._load_xarray_data", return_value=make_ensemble_ds()):
        info = _get_ds_info(config)

    assert info.sizes[REALIZATION_DIM] == 2
    assert info.coords[REALIZATION_DIM].values.tolist() == [0, 1]


def test_get_ds_info_passes_realization_selection_to_loader():
    config = DummyDataConfig(Path("."), realization_list=[1])
    config.list_paths = ["ensemble.nc"]

    with patch(
        f"{ABC}._load_xarray_data",
        return_value=make_ensemble_ds(),
    ) as load_data:
        _get_ds_info(config)

    assert load_data.call_args.kwargs["selection"] == {REALIZATION_DIM: [1]}


@pytest.mark.pruned
def test_get_ds_info_without_realization_passes_no_selection():
    config = DummyDataConfig(Path("."))
    config.list_paths = ["data.nc"]

    with patch(f"{ABC}._load_xarray_data", return_value=make_valid_ds()) as load_data:
        _get_ds_info(config)

    load_data.assert_called_once_with(
        ["data.nc"],
        names=["var"],
        selection=None,
        concat_dim=None,
        rename_dict={},
        drop_vars_list=None,
        resolution_to_standardize_time="month",
    )


@pytest.mark.pruned
def test_get_ds_info_uses_existing_list_paths():
    config = DummyDataConfig(Path("."))
    config.list_paths = ["already_set.nc"]

    with (
        patch(f"{ABC}.glob.glob") as glob_mock,
        patch(f"{ABC}._load_xarray_data", return_value=make_valid_ds()) as load_data,
    ):
        _get_ds_info(config)

    glob_mock.assert_not_called()
    assert load_data.call_args.args[0] == ["already_set.nc"]


@pytest.mark.pruned
def test_get_ds_info_discovers_paths_when_not_set():
    config = DummyDataConfig(Path("."))
    config.list_paths = None

    with (
        patch(f"{ABC}.glob.glob", return_value=["resolved.nc"]) as glob_mock,
        patch(f"{ABC}._load_xarray_data", return_value=make_valid_ds()) as load_data,
    ):
        _get_ds_info(config)

    glob_mock.assert_called_once()
    assert load_data.call_args.args[0] == ["resolved.nc"]


@pytest.mark.pruned
def test_fit_passes_selection_and_no_mask(tmp_path):
    config = DummyDataConfig(tmp_path)
    config.list_paths = ["x.nc"]
    selection = {
        INIT_TIME_DIM: slice(np.datetime64("2000-01-01"), np.datetime64("2001-01-01"))
    }

    with patch(f"{ABC}._load_xarray_data", return_value=make_valid_ds()) as load_data:
        config.fit_preprocessor_pipeline(selection=selection, mask=False, save=False)

    assert config.preprocessing_pipeline.fit_called is True
    assert config.preprocessing_pipeline.fit_kwargs["mask"] is None
    assert config.preprocessing_pipeline.fit_kwargs["save"] is False
    load_data.assert_called_once_with(
        ["x.nc"],
        names=["var"],
        concat_dim=None,
        selection=selection,
        ensemble_mean=False,
        rename_dict={},
        drop_vars_list=None,
        resolution_to_standardize_time="month",
    )


@pytest.mark.pruned
def test_fit_with_mask_forwards_mask_and_save_arguments(tmp_path):
    config = DummyDataConfig(tmp_path)
    config.list_paths = ["x.nc"]
    expected_mask = xr.DataArray(
        np.ones((2, 2), dtype=bool),
        dims=(INIT_TIME_DIM, LEAD_TIME_DIM),
        coords={
            INIT_TIME_DIM: np.array(
                ["2000-01-01", "2001-01-01"], dtype="datetime64[ns]"
            ),
            LEAD_TIME_DIM: [1, 2],
        },
    )

    with (
        patch(f"{ABC}._load_xarray_data", return_value=make_valid_ds()),
        patch(f"{ABC}._create_train_mask", return_value=expected_mask) as mock_mask,
    ):
        config.fit_preprocessor_pipeline(
            selection={},
            mask=True,
            save=True,
            save_path=tmp_path,
            save_name="pipeline.joblib",
        )

    kwargs = config.preprocessing_pipeline.fit_kwargs
    mock_mask.assert_called_once()
    assert kwargs["mask"] is expected_mask
    assert kwargs["save"] is True
    assert kwargs["save_path"] == tmp_path
    assert kwargs["save_name"] == "pipeline.joblib"


@pytest.mark.pruned
def test_fit_mask_builder_receives_time_coordinates(tmp_path):
    config = DummyDataConfig(tmp_path)
    config.list_paths = ["data.nc"]
    base = make_valid_ds()

    with (
        patch(f"{ABC}._load_xarray_data", return_value=base),
        patch(f"{ABC}._create_train_mask", return_value=object()) as create_mask,
    ):
        config.fit_preprocessor_pipeline(selection={}, mask=True, save=False)

    xr.testing.assert_identical(create_mask.call_args.args[0], base[INIT_TIME_DIM])
    xr.testing.assert_identical(create_mask.call_args.args[1], base[LEAD_TIME_DIM])


@pytest.mark.pruned
def test_fit_skips_mask_builder_when_mask_disabled(tmp_path):
    config = DummyDataConfig(tmp_path)
    config.list_paths = ["data.nc"]

    with (
        patch(f"{ABC}._load_xarray_data", return_value=make_valid_ds()),
        patch(f"{ABC}._create_train_mask") as create_mask,
    ):
        config.fit_preprocessor_pipeline(selection={}, mask=False, save=False)

    create_mask.assert_not_called()


@pytest.mark.pruned
def test_fit_skips_mask_when_init_dim_is_absent(tmp_path):
    config = DummyDataConfig(tmp_path)
    config.list_paths = ["static.nc"]

    with (
        patch(f"{ABC}._load_xarray_data", return_value=make_static_ds()),
        patch(f"{ABC}._create_train_mask") as create_mask,
    ):
        config.fit_preprocessor_pipeline(selection={}, mask=True, save=False)

    create_mask.assert_not_called()
    assert config.preprocessing_pipeline.fit_kwargs["mask"] is None


@pytest.mark.pruned
def test_fit_defaults_save_path_and_name_to_none(tmp_path):
    config = DummyDataConfig(tmp_path)
    config.list_paths = ["data.nc"]

    with patch(f"{ABC}._load_xarray_data", return_value=make_valid_ds()):
        config.fit_preprocessor_pipeline(selection={}, save=True)

    kwargs = config.preprocessing_pipeline.fit_kwargs
    assert kwargs["save"] is True
    assert kwargs["save_path"] is None
    assert kwargs["save_name"] is None


@pytest.mark.pruned
def test_fit_forwards_drop_variables_to_loader(tmp_path):
    config = DummyDataConfig(tmp_path)
    config.list_paths = ["data.nc"]
    config.drop_vars_list = ["unused"]

    with patch(f"{ABC}._load_xarray_data", return_value=make_valid_ds()) as load_data:
        config.fit_preprocessor_pipeline(selection={}, save=False)

    assert load_data.call_args.kwargs["drop_vars_list"] == ["unused"]


@pytest.mark.pruned
def test_fit_loads_base_data_before_passing_to_pipeline(tmp_path):
    config = DummyDataConfig(tmp_path)
    config.list_paths = ["data.nc"]
    base = Mock()
    loaded_base = object()
    base.load.return_value = loaded_base

    with patch(f"{ABC}._load_xarray_data", return_value=base):
        config.fit_preprocessor_pipeline(selection={}, save=False)

    base.load.assert_called_once_with()
    assert config.preprocessing_pipeline.fit_kwargs["base_data"] is loaded_base


@pytest.mark.pruned
def test_fit_closes_base_dataset(tmp_path):
    config = DummyDataConfig(tmp_path)
    config.list_paths = ["data.nc"]
    base = Mock()
    base.load.return_value = base

    with patch(f"{ABC}._load_xarray_data", return_value=base):
        config.fit_preprocessor_pipeline(selection={}, save=False)

    base.close.assert_called_once_with()


@pytest.mark.pruned
def test_fit_runs_garbage_collection(tmp_path):
    config = DummyDataConfig(tmp_path)
    config.list_paths = ["data.nc"]
    base = Mock()
    base.load.return_value = base

    with (
        patch(f"{ABC}._load_xarray_data", return_value=base),
        patch(f"{ABC}.gc.collect") as collect,
    ):
        config.fit_preprocessor_pipeline(selection={}, save=False)

    collect.assert_called_once_with()


@pytest.mark.pruned
def test_load_preprocessor_pipeline_uses_custom_directory(tmp_path):
    config = DummyDataConfig(tmp_path)

    config.load_preprocessor_pipeline(tmp_path)

    assert config.preprocessing_pipeline.loaded is True
    assert config.preprocessing_pipeline.loaded_path == (
        tmp_path / "dummy_preprocessing_pipeline.joblib"
    )


def test_load_preprocessor_pipeline_raises_when_not_fitted(tmp_path):
    config = DummyDataConfig(tmp_path)
    config.preprocessing_pipeline.fitted = False

    with pytest.raises(RuntimeError, match="is not fitted"):
        config.load_preprocessor_pipeline(tmp_path)


def test_load_preprocessor_pipeline_defaults_to_experiment_directory(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(RuntimeContext, "GLOBAL_EXP_DIR", tmp_path, raising=False)
    config = DummyDataConfig(tmp_path)
    captured = {}

    def fake_load(path):
        captured["path"] = Path(path)

    config.preprocessing_pipeline.load_from_memory = fake_load

    config.load_preprocessor_pipeline()

    assert captured["path"] == (
        tmp_path / "preprocessing_pipeline" / "dummy_preprocessing_pipeline.joblib"
    )


@pytest.mark.pruned
def test_load_preprocessor_pipeline_accepts_fitted_pipeline(tmp_path):
    config = DummyDataConfig(tmp_path)
    config.preprocessing_pipeline.fitted = True
    config.preprocessing_pipeline.load_from_memory = lambda path: None

    config.load_preprocessor_pipeline(tmp_path)


@pytest.mark.pruned
def test_coords_sizes_dims_delegate_to_info(tmp_path):
    config = DummyDataConfig(tmp_path)
    coords = {INIT_TIME_DIM: make_valid_ds()[INIT_TIME_DIM]}
    sizes = {INIT_TIME_DIM: 2, LEAD_TIME_DIM: 3}
    dims = (INIT_TIME_DIM, LEAD_TIME_DIM, NN_DIM)
    config.info = Mock(coords=coords, sizes=sizes, dims=dims)

    assert config.coords is coords
    assert config.sizes is sizes
    assert config.dims is dims


@pytest.mark.pruned
def test_init_time_frequency_delegates_to_info(tmp_path):
    config = DummyDataConfig(tmp_path)
    config.info = Mock(init_time_freq="year")

    assert config.init_time_frequency == "year"


@pytest.mark.pruned
def test_isel_requires_opened_data(tmp_path):
    config = DummyDataConfig(tmp_path)
    config.data = None

    with pytest.raises(ValueError, match="No data is currently opened"):
        config.isel({INIT_TIME_DIM: 0})


@pytest.mark.pruned
def test_isel_forwards_selection_to_pipeline(tmp_path):
    config = DummyDataConfig(tmp_path)
    config.data = make_valid_ds()
    transformed = object()
    config.preprocessing_pipeline.transform = Mock(return_value=transformed)

    result = config.isel({INIT_TIME_DIM: 1, LEAD_TIME_DIM: 0})

    passed = config.preprocessing_pipeline.transform.call_args.args[0]
    xr.testing.assert_identical(
        passed,
        config.data.isel({INIT_TIME_DIM: 1, LEAD_TIME_DIM: 0}),
    )
    assert result is transformed


@pytest.mark.pruned
def test_isel_accepts_keyword_indexers(tmp_path):
    config = DummyDataConfig(tmp_path)
    config.data = make_valid_ds()
    config.preprocessing_pipeline.transform = Mock(side_effect=lambda data: data)

    result = config.isel(**{INIT_TIME_DIM: 0, LEAD_TIME_DIM: 1})

    xr.testing.assert_identical(
        result,
        config.data.isel({INIT_TIME_DIM: 0, LEAD_TIME_DIM: 1}),
    )


@pytest.mark.pruned
def test_isel_accepts_none_mapping_with_keyword_indexer(tmp_path):
    config = DummyDataConfig(tmp_path)
    config.data = make_valid_ds()
    config.preprocessing_pipeline.transform = Mock(side_effect=lambda value: value)

    result = config.isel(None, **{INIT_TIME_DIM: 0})

    xr.testing.assert_identical(result, config.data.isel({INIT_TIME_DIM: 0}))


def test_indexes_requires_opened_data(tmp_path):
    config = DummyDataConfig(tmp_path)
    config.data = None

    with pytest.raises(ValueError, match="No data is currently opened"):
        _ = config.indexes


def test_check_opened_passes_for_open_dataset(tmp_path):
    config = DummyDataConfig(tmp_path)
    config.data = make_valid_ds()

    assert config._check_opened() is None


def test_open_xarray_data_forwards_all_configuration(tmp_path):
    config = DummyDataConfig(
        tmp_path,
        realization_list=[1],
        ensemble_mean=True,
        rename_dict={"old": "new"},
        concat_dim=INIT_TIME_DIM,
    )
    config.list_paths = ["data.nc"]
    config.drop_vars_list = ["unused"]
    config.info = Mock(
        coords={
            REALIZATION_DIM: xr.DataArray([1], dims=(REALIZATION_DIM,)),
        }
    )
    dataset = make_valid_ds()

    with (
        patch(f"{ABC}._load_xarray_data", return_value=dataset) as load_data,
        patch(f"{ABC}._validate_time_sequence") as validate,
    ):
        result = config.open_xarray_data(load=True, add_time_auxiliary_coords=True)

    assert result is config
    assert config.data is dataset
    load_data.assert_called_once_with(
        ["data.nc"],
        names=["var"],
        ensemble_mean=True,
        selection={REALIZATION_DIM: config.coords[REALIZATION_DIM]},
        concat_dim=INIT_TIME_DIM,
        rename_dict={"old": "new"},
        drop_vars_list=["unused"],
        resolution_to_standardize_time="month",
        add_time_auxiliary_coords=True,
        load=True,
    )
    validate.assert_called_once()
    xr.testing.assert_identical(
        validate.call_args.args[0],
        dataset.coords[INIT_TIME_DIM],
    )


@pytest.mark.pruned
def test_open_xarray_data_passes_no_selection_without_realization_coord(tmp_path):
    config = DummyDataConfig(tmp_path)
    config.list_paths = ["data.nc"]
    config.info = Mock(coords={})

    with patch(f"{ABC}._load_xarray_data", return_value=make_valid_ds()) as load_data:
        config.open_xarray_data()

    assert load_data.call_args.kwargs["selection"] is None


def test_open_xarray_data_skips_time_validation_for_static_data(tmp_path):
    config = DummyDataConfig(tmp_path)
    config.list_paths = ["static.nc"]
    config.info = Mock(coords={})

    with (
        patch(f"{ABC}._load_xarray_data", return_value=make_static_ds()),
        patch(f"{ABC}._validate_time_sequence") as validate,
    ):
        result = config.open_xarray_data()

    assert result is config
    validate.assert_not_called()


@pytest.mark.pruned
def test_reopening_replaces_dataset_without_closing_old_one(tmp_path):
    config = DummyDataConfig(tmp_path)
    config.list_paths = ["data.nc"]
    config.info = Mock(coords={})
    old_data = Mock()
    new_data = make_valid_ds()
    config.data = old_data

    with patch(f"{ABC}._load_xarray_data", return_value=new_data):
        config.open_xarray_data()

    assert config.data is new_data
    old_data.close.assert_not_called()


@pytest.mark.pruned
def test_close_data_closes_dataset_and_clears_reference(tmp_path):
    config = DummyDataConfig(tmp_path)
    data = Mock()
    config.data = data

    result = config.close_data()

    assert result is None
    data.close.assert_called_once_with()
    assert config.data is None


@pytest.mark.pruned
def test_close_data_is_safe_when_nothing_is_open(tmp_path):
    config = DummyDataConfig(tmp_path)
    config.data = None

    assert config.close_data() is None
    assert config.data is None


@pytest.mark.pruned
def test_close_data_can_be_called_twice(tmp_path):
    config = DummyDataConfig(tmp_path)
    data = Mock()
    config.data = data

    config.close_data()
    config.close_data()

    data.close.assert_called_once_with()
    assert config.data is None


@pytest.mark.pruned
def test_destructor_closes_open_data(tmp_path):
    config = DummyDataConfig(tmp_path)
    data = Mock()
    config.data = data

    config.__del__()

    data.close.assert_called_once_with()
    assert config.data is None


@pytest.mark.pruned
def test_infoclass_instances_with_equal_fields_are_equal():
    fields = dict(
        sizes={INIT_TIME_DIM: 2},
        start_time="2000-01-01",
        final_time="2001-01-01",
        coords={},
        dims=(INIT_TIME_DIM,),
        time_coords_type="datetime",
        init_time_freq="year",
    )

    assert infoclass(**fields) == infoclass(**fields)


@pytest.mark.pruned
def test_infoclass_accepts_missing_temporal_metadata():
    info = infoclass(
        sizes=None,
        start_time=None,
        final_time=None,
        coords={},
        dims=(),
        time_coords_type=None,
        init_time_freq=None,
    )

    assert info.sizes is None
    assert info.start_time is None
    assert info.final_time is None
    assert info.time_coords_type is None
    assert info.init_time_freq is None
