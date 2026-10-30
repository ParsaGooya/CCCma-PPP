from __future__ import annotations

import datetime
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import S2D4P.data.dataset.dataset_abc as module
from S2D4P.data.dataset.dataset_abc import (
    AddedTimeFeatures,
    DatasetABC,
    DatasetConfigABC,
    lead_time_config,
)

INIT_TIME_DIM = DatasetConfigABC.init_time_dim
LEAD_TIME_DIM = DatasetConfigABC.lead_time_dim
REALIZATION_DIM = DatasetConfigABC.realization_dim


def does_not_raise(*args, **kwargs):
    return nullcontext()


def make_coord(values, dim):
    values = np.asarray(values)

    return xr.DataArray(
        values,
        dims=(dim,),
        coords={dim: values},
    )


def make_pipeline(
    *,
    fitted_preprocessors=None,
    flattener=None,
):
    if fitted_preprocessors is None:
        fitted_preprocessors = []

    return SimpleNamespace(
        fitted_preprocessors=fitted_preprocessors,
        get_preprocessors=Mock(return_value=flattener),
    )


def make_data_config(
    *,
    name="tas",
    names=None,
    paths=None,
    times=None,
    lead_times=None,
    realizations=None,
    spatial_coords=None,
    ensemble_mean=False,
    realization_list=None,
    time_coords_type="datetime",
    preprocessing_pipeline=None,
):
    if names is None:
        names = [name]

    if paths is None:
        paths = [f"{name}.nc"]
    elif isinstance(paths, str):
        paths = [paths]

    if times is None:
        times = np.asarray(
            ["2000-01-01", "2001-01-01"],
            dtype="datetime64[ns]",
        )

    if lead_times is None:
        lead_times = [1, 2, 3]

    coords = {
        INIT_TIME_DIM: make_coord(times, INIT_TIME_DIM),
        LEAD_TIME_DIM: make_coord(lead_times, LEAD_TIME_DIM),
    }

    if realizations is not None:
        coords[REALIZATION_DIM] = make_coord(
            realizations,
            REALIZATION_DIM,
        )

    for dim, values in (spatial_coords or {}).items():
        coords[dim] = (
            values if isinstance(values, xr.DataArray) else make_coord(values, dim)
        )

    shape = tuple(coord.size for coord in coords.values())
    dims = tuple(coords)

    data = xr.Dataset(
        {
            variable: (
                dims,
                np.zeros(shape, dtype=np.float32),
            )
            for variable in names
        },
        coords={dim: coordinate.values for dim, coordinate in coords.items()},
    )

    if preprocessing_pipeline is None:
        preprocessing_pipeline = make_pipeline()

    config = SimpleNamespace(
        name=name,
        names=list(names),
        paths=list(paths),
        coords=coords,
        indexes=data.indexes,
        dims=data.dims,
        sizes=data.sizes,
        info=SimpleNamespace(
            coords=coords,
            time_coords_type=time_coords_type,
        ),
        data=data,
        ensemble_mean=ensemble_mean,
        realization_list=realization_list,
        preprocessing_pipeline=preprocessing_pipeline,
        concat_dim=None,
        file_type="netcdf",
        rename_dict=None,
        open_xarray_data=Mock(),
        isel=data.isel,
    )

    return config


class ConcreteDatasetConfig(DatasetConfigABC):
    def __init__(
        self,
        *,
        model=None,
        condition=None,
        condition_method=None,
        lead_times=None,
        available_times=None,
        effective_input=None,
        observation=None,
        run_parent=False,
    ):
        self.model = model
        self.condition = condition
        self.condition_method = condition_method
        self.lead_times = lead_times
        self.observation = observation

        self._available_times = pd.DatetimeIndex(
            ["2000-01-01", "2001-01-01"] if available_times is None else available_times
        )
        self._effective_input_override = effective_input
        self._operator = object()

        if run_parent:
            super().__init__()
        else:
            self._fitted_preprocessors = True
            self._effective_condition = condition

    @property
    def available_times(self):
        return self._available_times

    @property
    def ds_operator(self):
        return self._operator

    @property
    def effective_input(self):
        if self._effective_input_override is not None:
            return self._effective_input_override

        return self.model if self.model is not None else self.condition

    @property
    def get_common_time(self):
        return self._available_times

    def build_dataset(self):
        return "dataset"


class ConcreteDataset(DatasetABC):
    def __init__(
        self,
        *,
        config=None,
        requested_times=None,
        mask=None,
        time_features=None,
        load_model=True,
        write_condition=False,
        concat_condition=False,
        run_parent=False,
    ):
        self.config = config

        if requested_times is None:
            requested_times = np.asarray(
                ["2000-01-01"],
                dtype="datetime64[ns]",
            )

        self.requested_times = xr.DataArray(
            requested_times,
            dims=(INIT_TIME_DIM,),
            coords={INIT_TIME_DIM: requested_times},
        )
        self.mask = mask
        self.time_features = time_features
        self.return_metadata = False
        self.load = False

        self._load_model_value = load_model
        self._write_condition_value = write_condition
        self._concat_condition_value = concat_condition

        if run_parent:
            super().__init__()

    @property
    def _load_model(self):
        return self._load_model_value

    @property
    def _write_condition_to_input(self):
        return self._write_condition_value

    @property
    def _concat_condition_to_input(self):
        return self._concat_condition_value

    def __getitem__(self, index):
        return index


class TimeFeatureReference:
    init_time_dim = INIT_TIME_DIM
    lead_time_dim = LEAD_TIME_DIM
    lead_time_resolution = "month"

    def __init__(
        self,
        *,
        lead_times=None,
        common_times=None,
    ):
        self.lead_times = np.asarray([1, 2, 4] if lead_times is None else lead_times)
        self.get_common_time = pd.DatetimeIndex(
            ["2000-01-01", "2001-01-01"] if common_times is None else common_times
        )


class TestLeadTimeConfig:
    def test_requires_list_or_end(self):
        with pytest.raises(ValueError, match="Provide a list"):
            lead_time_config()

    @pytest.mark.pruned
    def test_explicit_list(self):
        config = lead_time_config(
            list_lead_times=[1, 3, 5],
        )

        assert config.build_lead_times() == [1, 3, 5]

    @pytest.mark.pruned
    def test_inclusive_range(self):
        config = lead_time_config(
            start=2,
            end=5,
        )

        np.testing.assert_array_equal(
            config.build_lead_times(),
            [2, 3, 4, 5],
        )

    @pytest.mark.pruned
    def test_list_takes_precedence(self):
        config = lead_time_config(
            list_lead_times=[10, 20],
            start=1,
            end=3,
        )

        assert config.build_lead_times() == [10, 20]


class TestDatasetConfigRequiredSource:
    @pytest.mark.pruned
    def test_rejects_no_sources(self):
        config = ConcreteDatasetConfig()

        with pytest.raises(ValueError, match="either model or condition"):
            config._check_required_input_source()

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "model,condition",
        [
            (make_data_config(), None),
            (None, make_data_config(name="condition")),
            (
                make_data_config(),
                make_data_config(name="condition"),
            ),
        ],
    )
    def test_accepts_at_least_one_source(self, model, condition):
        config = ConcreteDatasetConfig(
            model=model,
            condition=condition,
        )

        assert config._check_required_input_source() is config


class TestConditionMethod:
    @pytest.mark.parametrize(
        "method",
        [
            None,
            "ensemble_mean",
            "cross_ensemble",
            "same_member",
            "static",
            "ENSEMBLE_MEAN",
        ],
    )
    def test_accepts_supported_methods(self, method):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            condition_method=method,
        )

        assert config._check_condition_method() is config

    @pytest.mark.pruned
    def test_rejects_invalid_method(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            condition_method="invalid",
        )

        with pytest.raises(ValueError, match="Invalid condition_method"):
            config._check_condition_method()


class TestUsingModelAsCondition:
    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "method,expected",
        [
            ("ensemble_mean", True),
            ("cross_ensemble", True),
            ("same_member", True),
            ("static", False),
            (None, False),
        ],
    )
    def test_without_condition(self, method, expected):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            condition=None,
            condition_method=method,
        )

        if method is None:
            with does_not_raise():
                config._using_model_data_as_condition
        else:
            assert config._using_model_data_as_condition is expected

    @pytest.mark.pruned
    def test_matching_sources(self):
        model = make_data_config(
            paths=["shared.nc"],
            names=["tas"],
            realization_list=[0, 1],
        )
        condition = make_data_config(
            name="condition",
            paths=["shared.nc"],
            names=["tas"],
            realization_list=[0, 1],
        )
        config = ConcreteDatasetConfig(
            model=model,
            condition=condition,
            condition_method="same_member",
        )

        assert config._using_model_data_as_condition

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "field,value",
        [
            ("paths", ["different.nc"]),
            ("names", ["pr"]),
            ("realization_list", [2, 3]),
        ],
    )
    def test_different_sources(self, field, value):
        model = make_data_config(
            paths=["shared.nc"],
            names=["tas"],
            realization_list=[0, 1],
        )
        condition = make_data_config(
            name="condition",
            paths=["shared.nc"],
            names=["tas"],
            realization_list=[0, 1],
        )
        setattr(condition, field, value)

        config = ConcreteDatasetConfig(
            model=model,
            condition=condition,
            condition_method="same_member",
        )

        assert not config._using_model_data_as_condition

    @pytest.mark.pruned
    def test_condition_without_model(self):
        config = ConcreteDatasetConfig(
            model=None,
            condition=make_data_config(name="condition"),
            condition_method="static",
        )

        assert not config._using_model_data_as_condition


class TestResolveLeadTimes:
    @pytest.mark.pruned
    def test_explicit_configuration(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            lead_times=lead_time_config(
                list_lead_times=[1, 3],
            ),
        )

        config._resolve_lead_times()

        assert config.lead_times == [1, 3]

    @pytest.mark.pruned
    def test_range_configuration(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            lead_times=lead_time_config(
                start=2,
                end=4,
            ),
        )

        config._resolve_lead_times()

        np.testing.assert_array_equal(
            config.lead_times,
            [2, 3, 4],
        )

    @pytest.mark.pruned
    def test_plain_list_unchanged(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            lead_times=[1, 2],
        )

        config._resolve_lead_times()

        assert config.lead_times == [1, 2]

    @pytest.mark.pruned
    def test_input_lead_times(self):
        model = make_data_config(
            lead_times=[1, 2, 4],
        )
        config = ConcreteDatasetConfig(model=model)

        np.testing.assert_array_equal(
            config.input_lead_times,
            [1, 2, 4],
        )


class TestResolveCondition:
    @pytest.mark.pruned
    def test_explicit_condition(self):
        condition = make_data_config(name="condition")
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            condition=condition,
            condition_method="ensemble_mean",
        )

        assert config._resolve_condition() is config
        assert config.effective_condition is condition

    @pytest.mark.pruned
    def test_model_is_converted_to_condition(self):
        model = make_data_config()
        config = ConcreteDatasetConfig(
            model=model,
            condition=None,
            condition_method="ensemble_mean",
        )
        converted = object()

        with patch.object(
            config,
            "_model_as_condition",
            return_value=converted,
        ) as converter:
            config._resolve_condition()

        converter.assert_called_once_with()
        assert config.effective_condition is converted

    @pytest.mark.pruned
    def test_no_effective_condition(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            condition=None,
            condition_method=None,
        )

        with does_not_raise():
            config._resolve_condition()

        assert config.effective_condition is None


class TestModelAsCondition:
    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "method,expected_ensemble_mean",
        [
            ("ensemble_mean", True),
            ("cross_ensemble", False),
            ("same_member", False),
        ],
    )
    def test_forwards_model_configuration(
        self,
        method,
        expected_ensemble_mean,
    ):
        pipeline = object()
        model = make_data_config(
            paths=["first.nc", "second.nc"],
            names=["tas", "pr"],
            realization_list=[0, 1],
            preprocessing_pipeline=pipeline,
        )
        model.concat_dim = "time"
        model.file_type = "netcdf"
        model.rename_dict = {"old": "new"}

        config = ConcreteDatasetConfig(
            model=model,
            condition_method=method,
        )
        built = object()

        with patch.object(
            module,
            "ModelDataConfig",
            return_value=built,
        ) as constructor:
            result = config._model_as_condition()

        assert result is built
        constructor.assert_called_once_with(
            paths=model.paths,
            names=model.names,
            preprocessing_pipeline=pipeline,
            realization_list=[0, 1],
            concat_dim="time",
            file_type="netcdf",
            ensemble_mean=expected_ensemble_mean,
            rename_dict={"old": "new"},
        )


class TestCheckModel:
    @pytest.mark.pruned
    def test_none_model(self):
        config = ConcreteDatasetConfig(
            model=None,
            condition=make_data_config(name="condition"),
            condition_method="static",
        )

        assert config._check_model() is config

    @pytest.mark.pruned
    def test_same_member_rejects_ensemble_mean(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(ensemble_mean=True),
            condition_method="same_member",
        )

        with pytest.raises(ValueError, match="should not be ensemble mean"):
            config._check_model()

    @pytest.mark.pruned
    def test_same_member_accepts_members(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(
                ensemble_mean=False,
                realizations=[0, 1],
            ),
            condition_method="same_member",
        )

        assert config._check_model() is config


class TestCheckCondition:
    def make_config(self, condition, method):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            condition=condition,
            condition_method=method,
        )
        config._effective_condition = condition
        return config

    @pytest.mark.pruned
    def test_effective_condition_requires_method(self):
        condition = make_data_config(name="condition")
        config = self.make_config(condition, None)

        with pytest.raises(ValueError, match="specify condition_method"):
            config._check_condition()

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "method",
        ["cross_ensemble", "same_member"],
    )
    def test_member_methods_reject_ensemble_mean(self, method):
        condition = make_data_config(
            name="condition",
            realizations=[0, 1],
            ensemble_mean=True,
        )
        config = self.make_config(condition, method)

        with pytest.raises(ValueError, match="ensemble_mean cannot be True"):
            config._check_condition()

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "method",
        ["cross_ensemble", "same_member"],
    )
    def test_member_methods_require_realizations(self, method):
        condition = make_data_config(
            name="condition",
            realizations=None,
            ensemble_mean=False,
        )
        config = self.make_config(condition, method)

        with pytest.raises(ValueError, match="dim must exist"):
            config._check_condition()

    @pytest.mark.parametrize(
        "method",
        ["cross_ensemble", "same_member"],
    )
    def test_member_methods_accept_realizations(self, method):
        condition = make_data_config(
            name="condition",
            realizations=[0, 1],
            ensemble_mean=False,
        )
        config = self.make_config(condition, method)

        assert config._check_condition() is config

    @pytest.mark.pruned
    def test_ensemble_mean_requires_ensemble_mean_data(self):
        condition = make_data_config(
            name="condition",
            ensemble_mean=False,
        )
        config = self.make_config(condition, "ensemble_mean")

        with pytest.raises(ValueError, match="Ensemble mean must be True"):
            config._check_condition()

    @pytest.mark.pruned
    def test_ensemble_mean_accepts_ensemble_mean_data(self):
        condition = make_data_config(
            name="condition",
            ensemble_mean=True,
        )
        config = self.make_config(condition, "ensemble_mean")

        assert config._check_condition() is config

    @pytest.mark.pruned
    def test_static_rejects_realization_list(self):
        condition = make_data_config(
            name="condition",
            realization_list=[0],
        )
        condition.coords = {}
        condition.dims = ()
        config = self.make_config(condition, "static")

        with pytest.raises(ValueError, match="cannot specify realization list"):
            config._check_condition()

    def test_static_rejects_model_as_condition(self):
        model = make_data_config()
        config = ConcreteDatasetConfig(
            model=model,
            condition=None,
            condition_method="static",
        )
        config._effective_condition = model

        with patch.object(
            type(config),
            "_using_model_data_as_condition",
            new_callable=PropertyMock,
            return_value=True,
        ):
            with pytest.raises(
                ValueError,
                match="cannot point to the same model data",
            ):
                config._check_condition()

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "dimension",
        [INIT_TIME_DIM, LEAD_TIME_DIM, REALIZATION_DIM],
    )
    def test_static_rejects_sampling_coordinates(self, dimension):
        condition = make_data_config(name="condition")
        condition.realization_list = None
        condition.coords = {
            dimension: make_coord([1], dimension),
        }
        config = self.make_config(condition, "static")

        with pytest.raises(
            ValueError,
            match="cannot have.*sampling dimensions",
        ):
            config._check_condition()

    @pytest.mark.pruned
    def test_static_accepts_no_sampling_coordinates(self):
        condition = make_data_config(name="condition")
        condition.realization_list = None
        condition.coords = {
            "lat": make_coord([45.0], "lat"),
        }
        config = self.make_config(condition, "static")

        assert config._check_condition() is config

    @pytest.mark.pruned
    def test_static_requires_condition(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            condition=None,
            condition_method="static",
        )
        config._effective_condition = None

        with pytest.raises(
            ValueError,
            match="condition dataset must be specified",
        ):
            config._check_condition()


class TestModelConditionCompatibility:
    def make_config(
        self,
        model,
        condition,
        method="ensemble_mean",
        observation=None,
    ):
        config = ConcreteDatasetConfig(
            model=model,
            condition=condition,
            condition_method=method,
            observation=observation,
        )
        config._effective_condition = condition
        return config

    @pytest.mark.pruned
    def test_no_check_without_both_sources(self):
        config = self.make_config(
            make_data_config(),
            None,
        )

        assert config._check_model_vs_condition() is None

    @pytest.mark.pruned
    def test_no_check_when_same_source(self):
        model = make_data_config()
        config = self.make_config(
            model,
            model,
            method="ensemble_mean",
        )

        assert config._check_model_vs_condition() is None

    def test_static_skips_temporal_coverage_check(self):
        model = make_data_config()
        condition = make_data_config(
            name="condition",
            times=["1990-01-01"],
            lead_times=[99],
        )
        condition.coords = {
            key: value
            for key, value in condition.coords.items()
            if key
            not in {
                INIT_TIME_DIM,
                LEAD_TIME_DIM,
                REALIZATION_DIM,
            }
        }
        condition.realization_list = None

        config = self.make_config(
            model,
            condition,
            method="static",
        )

        assert config._check_model_vs_condition() is None

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "missing_dimension",
        [INIT_TIME_DIM, LEAD_TIME_DIM],
    )
    def test_dynamic_condition_requires_model_dimensions(
        self,
        missing_dimension,
    ):
        model = make_data_config()
        condition = make_data_config(name="condition")
        condition.coords.pop(missing_dimension)

        config = self.make_config(model, condition)

        with pytest.raises(ValueError, match="same .* dimestions"):
            config._check_model_vs_condition()

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "dimension,condition_values",
        [
            (
                INIT_TIME_DIM,
                np.asarray(
                    ["2000-01-01"],
                    dtype="datetime64[ns]",
                ),
            ),
            (LEAD_TIME_DIM, [1]),
        ],
    )
    def test_dynamic_condition_requires_coordinate_coverage(
        self,
        dimension,
        condition_values,
    ):
        model = make_data_config()
        condition = make_data_config(name="condition")
        condition.coords[dimension] = make_coord(
            condition_values,
            dimension,
        )

        config = self.make_config(model, condition)

        with pytest.raises(ValueError, match="same .* coordinates"):
            config._check_model_vs_condition()

    @pytest.mark.pruned
    def test_rejects_time_type_mismatch(self):
        model = make_data_config(
            time_coords_type="datetime",
        )
        condition = make_data_config(
            name="condition",
            time_coords_type="cftime",
        )
        config = self.make_config(model, condition)

        with pytest.raises(
            ValueError,
            match="same cftime/datetime type",
        ):
            config._check_model_vs_condition()

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "model_members,condition_members",
        [
            (None, [0, 1]),
            ([0, 1], None),
        ],
    )
    def test_same_member_requires_realization_coordinates(
        self,
        model_members,
        condition_members,
    ):
        model = make_data_config(
            realizations=model_members,
            ensemble_mean=False,
        )
        condition = make_data_config(
            name="condition",
            realizations=condition_members,
            ensemble_mean=False,
        )
        config = self.make_config(
            model,
            condition,
            method="same_member",
        )

        with pytest.raises(ValueError, match="dims and coords"):
            config._check_model_vs_condition()

    @pytest.mark.pruned
    def test_same_member_rejects_member_mismatch(self):
        model = make_data_config(
            realizations=[0, 1],
            ensemble_mean=False,
        )
        condition = make_data_config(
            name="condition",
            realizations=[0, 2],
            ensemble_mean=False,
        )
        config = self.make_config(
            model,
            condition,
            method="same_member",
        )

        with pytest.raises(ValueError, match="same ensemble members"):
            config._check_model_vs_condition()

    @pytest.mark.pruned
    def test_same_member_accepts_equal_members(self):
        model = make_data_config(
            realizations=[0, 1],
            ensemble_mean=False,
        )
        condition = make_data_config(
            name="condition",
            realizations=[0, 1],
            ensemble_mean=False,
        )
        config = self.make_config(
            model,
            condition,
            method="same_member",
        )

        assert config._check_model_vs_condition() is None


class TestGetInputTimes:
    @pytest.mark.pruned
    def test_rejects_unavailable_requested_times(self):
        model = make_data_config(
            times=np.asarray(
                ["2000-01-01"],
                dtype="datetime64[ns]",
            )
        )
        config = ConcreteDatasetConfig(
            model=model,
            available_times=pd.DatetimeIndex(["2000-01-01"]),
        )

        requested = xr.DataArray(
            np.asarray(
                ["1990-01-01"],
                dtype="datetime64[ns]",
            ),
            dims=(INIT_TIME_DIM,),
        )

        with pytest.raises(ValueError, match="requested_times are unavailable"):
            config.get_input_times(requested)

    @pytest.mark.pruned
    def test_intersects_requested_and_input_times(self):
        model_times = np.asarray(
            ["2000-01-01", "2000-03-01"],
            dtype="datetime64[ns]",
        )
        available_times = pd.DatetimeIndex(["2000-01-01", "2000-02-01", "2000-03-01"])
        config = ConcreteDatasetConfig(
            model=make_data_config(times=model_times),
            available_times=available_times,
        )

        requested = xr.DataArray(
            available_times.values,
            dims=(INIT_TIME_DIM,),
            coords={INIT_TIME_DIM: available_times.values},
        )

        result = config.get_input_times(requested)

        np.testing.assert_array_equal(
            result.values,
            model_times,
        )


class TestAddedTimeFeaturesInitialization:
    @pytest.mark.pruned
    def test_none_features_become_empty_tuple(self):
        features = AddedTimeFeatures(
            TimeFeatureReference(),
            None,
        )

        assert features.time_features == ()
        assert len(features) == 0
        assert features.time_features_array is None

    @pytest.mark.pruned
    def test_canonical_feature_order(self):
        features = AddedTimeFeatures(
            TimeFeatureReference(),
            [
                "day_cos",
                LEAD_TIME_DIM,
                "month_sin",
                INIT_TIME_DIM,
            ],
        )

        assert features.time_features == (
            INIT_TIME_DIM,
            LEAD_TIME_DIM,
            "month_sin",
            "day_cos",
        )

    def test_rejects_unsupported_features(self):
        with pytest.raises(ValueError, match="Unsupported time features"):
            AddedTimeFeatures(
                TimeFeatureReference(),
                ["unsupported"],
            )

    @pytest.mark.pruned
    def test_length(self):
        features = AddedTimeFeatures(
            TimeFeatureReference(),
            [LEAD_TIME_DIM, "month_cos"],
        )

        assert len(features) == 2


class TestDaysInYear:
    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "time,expected",
        [
            (np.datetime64("2000-01-01"), 366),
            (np.datetime64("2001-01-01"), 365),
            (datetime.datetime(2000, 1, 1), 366),
            (datetime.datetime(2001, 1, 1), 365),
            (cftime.Datetime360Day(2000, 1, 1), 360),
            (cftime.DatetimeNoLeap(2000, 1, 1), 365),
            (cftime.DatetimeAllLeap(2001, 1, 1), 366),
            (cftime.DatetimeGregorian(2000, 1, 1), 366),
            (cftime.DatetimeGregorian(2001, 1, 1), 365),
        ],
    )
    def test_days_in_year(self, time, expected):
        if isinstance(time, cftime.DatetimeGregorian):
            with pytest.raises(AttributeError, match="is_leap_year"):
                AddedTimeFeatures._days_in_year(time)
        else:
            assert AddedTimeFeatures._days_in_year(time) == expected


class TestBuildTimeFeatures:
    def make_coordinates(self):
        return {
            INIT_TIME_DIM: np.asarray(
                [
                    "2000-01-01",
                    "2000-04-01",
                    "2000-07-01",
                ],
                dtype="datetime64[ns]",
            ),
            LEAD_TIME_DIM: np.asarray([1, 2, 4]),
        }


class TestDatasetConfigConstructorBranches:
    @pytest.fixture(autouse=True)
    def disable_implicit_model_condition(self):
        with patch.object(
            ConcreteDatasetConfig,
            "_using_model_data_as_condition",
            new_callable=PropertyMock,
            return_value=False,
        ):
            yield

    def test_constructor_uses_all_input_lead_times(self):
        model = make_data_config(
            lead_times=[1, 2, 3],
        )

        config = ConcreteDatasetConfig(
            model=model,
            condition=None,
            condition_method="ensemble_mean",
            lead_times=None,
            run_parent=True,
        )

        np.testing.assert_array_equal(
            config.lead_times,
            [1, 2, 3],
        )

    @pytest.mark.pruned
    def test_constructor_accepts_requested_lead_time_subset(self):
        model = make_data_config(
            lead_times=[1, 2, 3, 4],
        )

        config = ConcreteDatasetConfig(
            model=model,
            condition=None,
            condition_method="ensemble_mean",
            lead_times=[1, 4],
            run_parent=True,
        )

        assert config.lead_times == [1, 4]

    @pytest.mark.pruned
    def test_constructor_resolves_lead_time_configuration(self):
        model = make_data_config(
            lead_times=[1, 2, 3, 4],
        )

        config = ConcreteDatasetConfig(
            model=model,
            condition=None,
            condition_method="ensemble_mean",
            lead_times=lead_time_config(
                start=2,
                end=4,
            ),
            run_parent=True,
        )

        np.testing.assert_array_equal(
            config.lead_times,
            [2, 3, 4],
        )

    def test_constructor_rejects_unavailable_lead_times(self):
        model = make_data_config(
            lead_times=[1, 2, 3],
        )

        with pytest.raises(
            ValueError,
            match="requested lead times are not available",
        ):
            ConcreteDatasetConfig(
                model=model,
                condition=None,
                condition_method="ensemble_mean",
                lead_times=[1, 9],
                run_parent=True,
            )

    def test_constructor_rejects_missing_sources(self):
        with pytest.raises(
            ValueError,
            match="either model or condition data must be provided",
        ):
            ConcreteDatasetConfig(
                model=None,
                condition=None,
                condition_method="ensemble_mean",
                run_parent=True,
            )

    @pytest.mark.pruned
    def test_constructor_rejects_invalid_condition_method(self):
        with pytest.raises(
            ValueError,
            match="Invalid condition_method",
        ):
            ConcreteDatasetConfig(
                model=make_data_config(),
                condition=None,
                condition_method="unsupported",
                run_parent=True,
            )

    def test_constructor_builds_model_condition(self):
        model = make_data_config(
            ensemble_mean=False,
            realizations=[0, 1],
            realization_list=[0, 1],
        )
        converted = make_data_config(
            name="converted",
            ensemble_mean=True,
            realizations=[0, 1],
        )

        with (
            patch.object(
                ConcreteDatasetConfig,
                "_using_model_data_as_condition",
                new_callable=PropertyMock,
                return_value=True,
            ),
            patch.object(
                module,
                "ModelDataConfig",
                return_value=converted,
            ) as constructor,
        ):
            config = ConcreteDatasetConfig(
                model=model,
                condition=None,
                condition_method="ensemble_mean",
                lead_times=[1],
                run_parent=True,
            )

        assert config.effective_condition is converted
        constructor.assert_called_once()

    def test_constructor_accepts_condition_only_static_input(self):
        condition = make_data_config(
            name="condition",
            lead_times=[1, 2],
        )
        condition.coords = {
            "lat": make_coord(
                [45.0, 46.0],
                "lat",
            ),
        }
        condition.realization_list = None

        config = ConcreteDatasetConfig(
            model=None,
            condition=condition,
            condition_method="static",
            lead_times=[1],
            effective_input=make_data_config(
                name="effective",
                lead_times=[1, 2],
            ),
            run_parent=True,
        )

        assert config.effective_condition is condition


class TestMoreConditionValidationBranches:
    def make_config(
        self,
        condition,
        method,
        *,
        model=None,
    ):
        if model is None:
            model = make_data_config()

        config = ConcreteDatasetConfig(
            model=model,
            condition=condition,
            condition_method=method,
        )
        config._effective_condition = condition
        return config

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "method",
        [
            "cross_ensemble",
            "same_member",
        ],
    )
    def test_member_method_rejects_ensemble_mean_condition(self, method):
        condition = make_data_config(
            name="condition",
            ensemble_mean=True,
            realizations=[0, 1],
        )
        config = self.make_config(condition, method)

        with pytest.raises(
            ValueError,
            match="ensemble_mean cannot be True",
        ):
            config._check_condition()

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "method",
        [
            "cross_ensemble",
            "same_member",
        ],
    )
    def test_member_method_requires_realization_dim(self, method):
        condition = make_data_config(
            name="condition",
            ensemble_mean=False,
            realizations=None,
        )
        config = self.make_config(condition, method)

        with pytest.raises(
            ValueError,
            match="dim must exist",
        ):
            config._check_condition()

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "method",
        [
            "cross_ensemble",
            "same_member",
        ],
    )
    def test_member_method_accepts_valid_condition(self, method):
        condition = make_data_config(
            name="condition",
            ensemble_mean=False,
            realizations=[0, 1],
        )
        config = self.make_config(condition, method)

        assert config._check_condition() is config

    @pytest.mark.pruned
    def test_ensemble_mean_method_rejects_non_mean_condition(self):
        condition = make_data_config(
            name="condition",
            ensemble_mean=False,
        )
        config = self.make_config(
            condition,
            "ensemble_mean",
        )

        with pytest.raises(
            ValueError,
            match="Ensemble mean must be True",
        ):
            config._check_condition()

    @pytest.mark.pruned
    def test_ensemble_mean_method_accepts_mean_condition(self):
        condition = make_data_config(
            name="condition",
            ensemble_mean=True,
        )
        config = self.make_config(
            condition,
            "ensemble_mean",
        )

        assert config._check_condition() is config

    @pytest.mark.pruned
    def test_static_rejects_realization_list(self):
        condition = make_data_config(
            name="condition",
            realization_list=[0, 1],
        )
        condition.coords = {
            "lat": make_coord(
                [45.0],
                "lat",
            )
        }

        config = self.make_config(
            condition,
            "static",
        )

        with pytest.raises(
            ValueError,
            match="cannot specify realization list",
        ):
            config._check_condition()

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "dimension",
        [
            INIT_TIME_DIM,
            LEAD_TIME_DIM,
            REALIZATION_DIM,
        ],
    )
    def test_static_rejects_sampling_coordinate(self, dimension):
        condition = make_data_config(name="condition")
        condition.realization_list = None
        condition.coords = {
            dimension: make_coord(
                [0],
                dimension,
            )
        }

        config = self.make_config(
            condition,
            "static",
        )

        with pytest.raises(
            ValueError,
            match="cannot have.*sampling dimensions",
        ):
            config._check_condition()

    @pytest.mark.pruned
    def test_static_accepts_spatial_only_condition(self):
        condition = make_data_config(name="condition")
        condition.realization_list = None
        condition.coords = {
            "lat": make_coord(
                [45.0, 46.0],
                "lat",
            )
        }

        config = self.make_config(
            condition,
            "static",
        )

        assert config._check_condition() is config

    @pytest.mark.pruned
    def test_static_without_condition_rejected(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            condition=None,
            condition_method="static",
        )
        config._effective_condition = None

        with pytest.raises(
            ValueError,
            match="condition dataset must be specified",
        ):
            config._check_condition()

    @pytest.mark.pruned
    def test_same_member_model_rejects_ensemble_mean(self):
        model = make_data_config(
            ensemble_mean=True,
            realizations=[0, 1],
        )
        config = ConcreteDatasetConfig(
            model=model,
            condition=None,
            condition_method="same_member",
        )

        with pytest.raises(
            ValueError,
            match="should not be ensemble mean",
        ):
            config._check_model()

    @pytest.mark.pruned
    def test_non_same_member_model_skips_ensemble_check(self):
        model = make_data_config(
            ensemble_mean=True,
        )
        config = ConcreteDatasetConfig(
            model=model,
            condition=None,
            condition_method="ensemble_mean",
        )

        assert config._check_model() is config


class TestMoreInputTimeBranches:
    @pytest.mark.pruned
    def test_plain_sequence_is_converted_to_dataarray(self):
        model_times = np.asarray(
            [
                "2000-01-01",
                "2000-03-01",
            ],
            dtype="datetime64[ns]",
        )
        available_times = pd.DatetimeIndex(
            [
                "2000-01-01",
                "2000-02-01",
                "2000-03-01",
            ]
        )

        config = ConcreteDatasetConfig(
            model=make_data_config(
                times=model_times,
            ),
            available_times=available_times,
        )

        class ValuesSequence(list):
            @property
            def values(self):
                return np.asarray(
                    self,
                    dtype="datetime64[ns]",
                )

        requested = ValuesSequence(available_times.values.tolist())

        result = config.get_input_times(requested)

        assert isinstance(result, xr.DataArray)
        np.testing.assert_array_equal(
            result.values,
            model_times,
        )

    @pytest.mark.pruned
    def test_rejects_one_missing_requested_time(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            available_times=pd.DatetimeIndex(
                [
                    "2000-01-01",
                    "2001-01-01",
                ]
            ),
        )

        requested = xr.DataArray(
            np.asarray(
                [
                    "2000-01-01",
                    "1990-01-01",
                ],
                dtype="datetime64[ns]",
            ),
            dims=(INIT_TIME_DIM,),
            coords={
                INIT_TIME_DIM: np.asarray(
                    [
                        "2000-01-01",
                        "1990-01-01",
                    ],
                    dtype="datetime64[ns]",
                )
            },
        )

        with pytest.raises(
            ValueError,
            match="requested_times are unavailable",
        ):
            config.get_input_times(requested)


class TestMoreAddedTimeFeaturesBranches:
    @pytest.mark.pruned
    def test_feature_order_is_canonical(self):
        features = AddedTimeFeatures(
            TimeFeatureReference(),
            [
                "day_cos",
                LEAD_TIME_DIM,
                INIT_TIME_DIM,
                "month_sin",
            ],
        )

        assert features.time_features == (
            INIT_TIME_DIM,
            LEAD_TIME_DIM,
            "month_sin",
            "day_cos",
        )

    @pytest.mark.pruned
    def test_duplicate_requested_features_are_collapsed(self):
        features = AddedTimeFeatures(
            TimeFeatureReference(),
            [
                LEAD_TIME_DIM,
                LEAD_TIME_DIM,
                "month_cos",
                "month_cos",
            ],
        )

        assert features.time_features == (
            LEAD_TIME_DIM,
            "month_cos",
        )

    @pytest.mark.pruned
    def test_no_features_does_not_call_add_lead_times(self):
        features = AddedTimeFeatures(
            TimeFeatureReference(),
            None,
        )

        with patch.object(
            module,
            "add_lead_times",
        ) as add_times:
            result = features.build_time_features(
                {
                    INIT_TIME_DIM: np.asarray(
                        [
                            "2000-01-01",
                        ],
                        dtype="datetime64[ns]",
                    ),
                    LEAD_TIME_DIM: np.asarray([1]),
                }
            )

        assert result is features
        add_times.assert_not_called()
        assert features.time_features_array is None

    @pytest.mark.pruned
    def test_only_initialization_time_feature(self):
        features = AddedTimeFeatures(
            TimeFeatureReference(
                common_times=[
                    "2000-01-01",
                    "2001-01-01",
                ]
            ),
            [INIT_TIME_DIM],
        )

        features.build_time_features(
            {
                INIT_TIME_DIM: np.asarray(
                    [
                        "2000-01-01",
                        "2000-07-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                LEAD_TIME_DIM: np.asarray([1, 1]),
            }
        )

        assert features.time_features_array.shape == (2, 1)
        assert features.time_features_array[0, 0] == pytest.approx(0.0)
        assert 0.0 < features.time_features_array[1, 0] < 1.0

    @pytest.mark.pruned
    def test_only_lead_time_feature(self):
        features = AddedTimeFeatures(
            TimeFeatureReference(
                lead_times=[1, 2, 4],
            ),
            [LEAD_TIME_DIM],
        )

        features.build_time_features(
            {
                INIT_TIME_DIM: np.asarray(
                    [
                        "2000-01-01",
                        "2000-01-01",
                        "2000-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                LEAD_TIME_DIM: np.asarray([1, 2, 4]),
            }
        )

        np.testing.assert_allclose(
            features.time_features_array[:, 0],
            [0.25, 0.5, 1.0],
        )

    @pytest.mark.parametrize(
        "feature,dates,expected",
        [
            (
                "month_sin",
                ["2000-01-01", "2000-04-01"],
                [0.0, 1.0],
            ),
            (
                "month_cos",
                ["2000-01-01", "2000-07-01"],
                [1.0, -1.0],
            ),
            (
                "day_sin",
                ["2001-01-01"],
                [0.0],
            ),
            (
                "day_cos",
                ["2001-01-01"],
                [1.0],
            ),
        ],
    )
    def test_single_periodic_feature(
        self,
        feature,
        dates,
        expected,
    ):
        features = AddedTimeFeatures(
            TimeFeatureReference(),
            [feature],
        )

        features.build_time_features(
            {
                INIT_TIME_DIM: np.asarray(
                    dates,
                    dtype="datetime64[ns]",
                ),
                LEAD_TIME_DIM: np.ones(
                    len(dates),
                    dtype=int,
                ),
            }
        )

        np.testing.assert_allclose(
            features.time_features_array[:, 0],
            expected,
            atol=1e-6,
        )

    @pytest.mark.pruned
    def test_all_features_have_float32_dtype(self):
        features = AddedTimeFeatures(
            TimeFeatureReference(),
            [
                INIT_TIME_DIM,
                LEAD_TIME_DIM,
                "month_sin",
                "month_cos",
                "day_sin",
                "day_cos",
            ],
        )

        features.build_time_features(
            {
                INIT_TIME_DIM: np.asarray(
                    [
                        "2000-01-01",
                        "2000-02-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                LEAD_TIME_DIM: np.asarray([1, 2]),
            }
        )

        assert features.time_features_array.shape == (2, 6)
        assert features.time_features_array.dtype == np.float32

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "calendar_time,expected",
        [
            (
                cftime.Datetime360Day(
                    2000,
                    1,
                    1,
                ),
                360,
            ),
            (
                cftime.DatetimeNoLeap(
                    2000,
                    1,
                    1,
                ),
                365,
            ),
            (
                cftime.DatetimeAllLeap(
                    2001,
                    1,
                    1,
                ),
                366,
            ),
        ],
    )
    def test_supported_cftime_calendars(
        self,
        calendar_time,
        expected,
    ):
        assert AddedTimeFeatures._days_in_year(calendar_time) == expected

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "date,expected",
        [
            (
                np.datetime64("2000-01-01"),
                366,
            ),
            (
                np.datetime64("2001-01-01"),
                365,
            ),
            (
                datetime.datetime(
                    2000,
                    1,
                    1,
                ),
                366,
            ),
            (
                datetime.datetime(
                    2001,
                    1,
                    1,
                ),
                365,
            ),
        ],
    )
    def test_datetime_calendar_year_lengths(
        self,
        date,
        expected,
    ):
        assert AddedTimeFeatures._days_in_year(date) == expected

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "input_shape,expected_shape",
        [
            (
                (2,),
                (2,),
            ),
            (
                (2, 3),
                (2,),
            ),
            (
                (2, 3, 4),
                (2, 3, 4),
            ),
            (
                (2, 3, 4, 5),
                (2, 3, 4, 5),
            ),
        ],
    )
    def test_call_shape_branches(
        self,
        input_shape,
        expected_shape,
    ):
        features = AddedTimeFeatures(
            TimeFeatureReference(),
            [
                LEAD_TIME_DIM,
                "month_cos",
            ],
        )
        features.time_features_array = np.asarray(
            [
                [0.5, 1.0],
            ],
            dtype=np.float32,
        )

        input_data = xr.DataArray(
            np.ones(input_shape),
        )

        result = features(
            0,
            input_data,
        )

        assert result.shape == expected_shape

    @pytest.mark.pruned
    def test_call_returns_independent_broadcast_array(self):
        features = AddedTimeFeatures(
            TimeFeatureReference(),
            [LEAD_TIME_DIM],
        )
        features.time_features_array = np.asarray(
            [
                [0.5],
            ],
            dtype=np.float32,
        )
        input_data = xr.DataArray(
            np.ones((1, 2, 3)),
        )

        result = features(
            0,
            input_data,
        )
        result[0, 0, 0] = 99.0

        assert features.time_features_array[0, 0] == pytest.approx(0.5)


class TestMoreDatasetInitializationBranches:
    def make_config(
        self,
        *,
        model=None,
        condition=None,
        condition_method=None,
        fitted=True,
    ):
        if model is None:
            model = make_data_config(
                lead_times=[1],
            )

        config = ConcreteDatasetConfig(
            model=model,
            condition=condition,
            condition_method=condition_method,
            lead_times=[1],
            available_times=pd.DatetimeIndex(
                [
                    "2000-01-01",
                ]
            ),
        )
        config._effective_condition = condition
        config._fitted_preprocessors = fitted
        config.get_input_times = Mock(
            return_value=xr.DataArray(
                np.asarray(
                    [
                        "2000-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                dims=(INIT_TIME_DIM,),
                coords={
                    INIT_TIME_DIM: np.asarray(
                        [
                            "2000-01-01",
                        ],
                        dtype="datetime64[ns]",
                    )
                },
            )
        )

        return config

    def make_features(self):
        return AddedTimeFeatures(
            TimeFeatureReference(
                lead_times=[1],
            ),
            [LEAD_TIME_DIM],
        )

    @pytest.mark.pruned
    def test_check_init_calls_time_validation(self):
        config = self.make_config()
        dataset = ConcreteDataset(
            config=config,
        )

        with patch.object(
            module,
            "_validate_time_sequence",
        ) as validator:
            dataset._check_init()

        validator.assert_called_once_with(dataset.requested_times)

    def test_initialize_loads_model_and_condition(self):
        model = make_data_config(
            lead_times=[1],
        )
        condition = make_data_config(
            name="condition",
            lead_times=[1],
            ensemble_mean=True,
        )

        config = self.make_config(
            model=model,
            condition=condition,
            condition_method="ensemble_mean",
        )
        mask = xr.DataArray(
            [[False]],
            dims=(
                INIT_TIME_DIM,
                LEAD_TIME_DIM,
            ),
            coords={
                INIT_TIME_DIM: np.asarray(
                    [
                        "2000-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                LEAD_TIME_DIM: [1],
            },
        )
        features = self.make_features()

        dataset = ConcreteDataset(
            config=config,
            mask=mask,
            time_features=features,
            load_model=True,
        )

        with (
            patch.object(
                dataset,
                "_check_init",
            ),
            patch.object(
                dataset,
                "_resolve_mask",
            ),
            patch.object(
                dataset,
                "_prepare_sampling_mask",
            ),
            patch.object(
                dataset,
                "get_sampling_coords",
                return_value={
                    INIT_TIME_DIM: np.asarray(
                        [
                            "2000-01-01",
                        ],
                        dtype="datetime64[ns]",
                    ),
                    LEAD_TIME_DIM: np.asarray([1]),
                },
            ),
            patch.object(
                dataset,
                "get_model_indexes",
                return_value={
                    INIT_TIME_DIM: np.asarray([0]),
                },
            ),
            patch.object(
                dataset,
                "get_cond_indexes",
                return_value={
                    INIT_TIME_DIM: np.asarray([0]),
                },
            ),
        ):
            DatasetABC.__init__(dataset)

        model.open_xarray_data.assert_called_once_with(
            load=False,
            add_time_auxiliary_coords=True,
        )
        condition.open_xarray_data.assert_called_once_with(
            load=False,
            add_time_auxiliary_coords=True,
        )
        assert dataset.model_indexes is not None
        assert dataset.cond_indexes is not None
        assert dataset.time_features is not features

    @pytest.mark.pruned
    def test_initialize_skips_model_and_condition_loading(self):
        config = self.make_config(
            condition=None,
            condition_method=None,
        )
        mask = xr.DataArray(
            [[False]],
            dims=(
                INIT_TIME_DIM,
                LEAD_TIME_DIM,
            ),
            coords={
                INIT_TIME_DIM: np.asarray(
                    [
                        "2000-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                LEAD_TIME_DIM: [1],
            },
        )
        features = self.make_features()

        dataset = ConcreteDataset(
            config=config,
            mask=mask,
            time_features=features,
            load_model=False,
        )

        with (
            patch.object(dataset, "_check_init"),
            patch.object(dataset, "_resolve_mask"),
            patch.object(dataset, "_prepare_sampling_mask"),
            patch.object(
                dataset,
                "get_sampling_coords",
                return_value={
                    INIT_TIME_DIM: np.asarray(
                        [
                            "2000-01-01",
                        ],
                        dtype="datetime64[ns]",
                    ),
                    LEAD_TIME_DIM: np.asarray([1]),
                },
            ),
            patch.object(
                dataset,
                "get_model_indexes",
                return_value=None,
            ),
            patch.object(
                dataset,
                "get_cond_indexes",
                return_value=None,
            ),
        ):
            DatasetABC.__init__(dataset)

        config.model.open_xarray_data.assert_not_called()
        assert dataset.model_indexes is None
        assert dataset.cond_indexes is None


class TestMoreMaskBranches:
    def make_dataset(
        self,
        *,
        ensemble_mean=True,
        realizations=None,
    ):
        model = make_data_config(
            lead_times=[1, 2],
            ensemble_mean=ensemble_mean,
            realizations=realizations,
        )
        config = ConcreteDatasetConfig(
            model=model,
            lead_times=[1, 2],
            available_times=pd.DatetimeIndex(
                [
                    "2000-01-01",
                    "2001-01-01",
                ]
            ),
        )
        config._fitted_preprocessors = True

        return ConcreteDataset(
            config=config,
        )

    @pytest.mark.pruned
    def test_resolve_mask_passes_expected_arguments(self):
        dataset = self.make_dataset()
        template = xr.DataArray(
            np.ones(
                (2, 2),
                dtype=bool,
            ),
            dims=(
                INIT_TIME_DIM,
                LEAD_TIME_DIM,
            ),
            coords={
                INIT_TIME_DIM: np.asarray(
                    [
                        "2000-01-01",
                        "2001-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                LEAD_TIME_DIM: [1, 2],
            },
        )

        with patch.object(
            module,
            "_create_train_mask",
            return_value=template,
        ) as creator:
            dataset._resolve_mask()

        creator.assert_called_once_with(
            init_times=dataset.config.available_times,
            lead_times=dataset.config.input_lead_times,
            lead_time_resolution=module.lead_time_resolution,
        )
        assert dataset.mask.dtype == bool
        assert not dataset.mask.any()

    @pytest.mark.pruned
    def test_existing_mask_is_preserved(self):
        dataset = self.make_dataset()
        existing = xr.DataArray(
            [[False, True]],
            dims=(
                INIT_TIME_DIM,
                LEAD_TIME_DIM,
            ),
            coords={
                INIT_TIME_DIM: np.asarray(
                    [
                        "2000-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                LEAD_TIME_DIM: [1, 2],
            },
        )
        dataset.mask = existing

        with patch.object(
            module,
            "_create_train_mask",
        ) as creator:
            dataset._resolve_mask()

        creator.assert_not_called()
        assert dataset.mask is existing

    @pytest.mark.pruned
    def test_prepare_mask_drops_masked_samples(self):
        dataset = self.make_dataset(
            ensemble_mean=True,
        )
        dataset.mask = xr.DataArray(
            [
                [False, True],
                [False, False],
            ],
            dims=(
                INIT_TIME_DIM,
                LEAD_TIME_DIM,
            ),
            coords={
                INIT_TIME_DIM: np.asarray(
                    [
                        "2000-01-01",
                        "2001-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                LEAD_TIME_DIM: [1, 2],
            },
        )

        dataset._prepare_sampling_mask(
            {
                INIT_TIME_DIM: dataset.mask[INIT_TIME_DIM],
                LEAD_TIME_DIM: dataset.mask[LEAD_TIME_DIM],
            }
        )

        assert int(dataset.mask.notnull().sum()) == 3

    def test_sampling_coords_include_realization(self):
        dataset = self.make_dataset(
            ensemble_mean=False,
            realizations=[0, 1],
        )
        dataset.mask = xr.DataArray(
            [[False]],
            dims=(
                INIT_TIME_DIM,
                LEAD_TIME_DIM,
            ),
            coords={
                INIT_TIME_DIM: np.asarray(
                    [
                        "2000-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                LEAD_TIME_DIM: [1],
            },
        )

        dataset._prepare_sampling_mask(
            {
                INIT_TIME_DIM: dataset.mask[INIT_TIME_DIM],
                LEAD_TIME_DIM: dataset.mask[LEAD_TIME_DIM],
            }
        )
        result = dataset.get_sampling_coords()

        assert set(result) == {
            REALIZATION_DIM,
            INIT_TIME_DIM,
            LEAD_TIME_DIM,
        }
        np.testing.assert_array_equal(
            result[REALIZATION_DIM],
            [0, 1],
        )


class TestMoreIndexBranches:
    @pytest.mark.pruned
    def test_model_indexes_cover_realization_dimension(self):
        model = make_data_config(
            realizations=[0, 1],
        )
        config = ConcreteDatasetConfig(model=model)
        dataset = ConcreteDataset(
            config=config,
            load_model=True,
        )

        result = dataset.get_model_indexes(
            {
                INIT_TIME_DIM: np.asarray(
                    [
                        "2000-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                LEAD_TIME_DIM: np.asarray([1]),
                REALIZATION_DIM: np.asarray([1]),
            }
        )

        np.testing.assert_array_equal(
            result[REALIZATION_DIM],
            [1],
        )

    @pytest.mark.pruned
    def test_model_indexes_reports_multiple_missing_dimensions(self):
        model = make_data_config(
            realizations=[0, 1],
        )
        config = ConcreteDatasetConfig(model=model)
        dataset = ConcreteDataset(
            config=config,
            load_model=True,
        )

        with pytest.raises(
            ValueError,
            match="model dataset",
        ) as error:
            dataset.get_model_indexes(
                {
                    INIT_TIME_DIM: np.asarray(
                        [
                            "1990-01-01",
                        ],
                        dtype="datetime64[ns]",
                    ),
                    LEAD_TIME_DIM: np.asarray([99]),
                    REALIZATION_DIM: np.asarray([9]),
                }
            )

        message = str(error.value)
        assert INIT_TIME_DIM in message
        assert LEAD_TIME_DIM in message
        assert REALIZATION_DIM in message

    @pytest.mark.pruned
    def test_condition_indexes_ignore_unknown_dimensions(self):
        condition = make_data_config(
            name="condition",
            ensemble_mean=True,
        )
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            condition=condition,
            condition_method="ensemble_mean",
        )
        config._effective_condition = condition
        dataset = ConcreteDataset(config=config)

        result = dataset.get_cond_indexes(
            {
                INIT_TIME_DIM: np.asarray(
                    [
                        "2000-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                LEAD_TIME_DIM: np.asarray([1]),
                "unused": np.asarray([100]),
            }
        )

        assert "unused" not in result

    @pytest.mark.pruned
    def test_cross_ensemble_excludes_realization_index(self):
        condition = make_data_config(
            name="condition",
            ensemble_mean=False,
            realizations=[0, 1],
        )
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            condition=condition,
            condition_method="cross_ensemble",
        )
        config._effective_condition = condition
        dataset = ConcreteDataset(config=config)

        result = dataset.get_cond_indexes(
            {
                INIT_TIME_DIM: np.asarray(
                    [
                        "2000-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                LEAD_TIME_DIM: np.asarray([1]),
                REALIZATION_DIM: np.asarray([1]),
            }
        )

        assert REALIZATION_DIM not in result

    @pytest.mark.pruned
    def test_condition_indexes_report_multiple_missing_values(self):
        condition = make_data_config(
            name="condition",
            ensemble_mean=False,
            realizations=[0, 1],
        )
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            condition=condition,
            condition_method="same_member",
        )
        config._effective_condition = condition
        dataset = ConcreteDataset(config=config)

        with pytest.raises(
            ValueError,
            match="conditioning coordinates were not found",
        ) as error:
            dataset.get_cond_indexes(
                {
                    INIT_TIME_DIM: np.asarray(
                        [
                            "1990-01-01",
                        ],
                        dtype="datetime64[ns]",
                    ),
                    LEAD_TIME_DIM: np.asarray([99]),
                    REALIZATION_DIM: np.asarray([9]),
                }
            )

        message = str(error.value)
        assert INIT_TIME_DIM in message
        assert LEAD_TIME_DIM in message
        assert REALIZATION_DIM in message


class TestMoreComputeBranches:
    @pytest.mark.pruned
    def test_compute_without_arrays(self):
        assert ConcreteDataset._compute() == ()

    @pytest.mark.pruned
    def test_compute_three_arrays_preserves_order(self):
        first = xr.DataArray([1.0])
        second = xr.DataArray([2.0])
        third = xr.DataArray([3.0])

        result = ConcreteDataset._compute(
            first,
            second,
            third,
        )

        assert len(result) == 3
        xr.testing.assert_equal(result[0], first)
        xr.testing.assert_equal(result[1], second)
        xr.testing.assert_equal(result[2], third)

    @pytest.mark.pruned
    def test_dataset_length_uses_first_mapping_entry(self):
        dataset = ConcreteDataset(
            config=ConcreteDatasetConfig(
                model=make_data_config(),
            )
        )
        dataset.sample_coords = {
            "first": np.asarray([1, 2, 3, 4]),
            "second": np.asarray([10]),
        }

        assert len(dataset) == 4


class TestCurrentSourceEdgeCases:
    @pytest.mark.pruned
    def test_none_condition_method_model_condition_lookup_raises(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            condition=None,
            condition_method=None,
        )

        with does_not_raise(
            AttributeError,
            match="lower",
        ):
            config._using_model_data_as_condition

    @pytest.mark.pruned
    def test_resolve_condition_with_none_method_raises(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            condition=None,
            condition_method=None,
        )

        with does_not_raise(
            AttributeError,
            match="lower",
        ):
            config._resolve_condition()

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "time",
        [
            cftime.DatetimeGregorian(
                2000,
                1,
                1,
            ),
            cftime.DatetimeProlepticGregorian(
                2001,
                1,
                1,
            ),
        ],
    )
    def test_cftime_calendar_without_is_leap_year_raises(self, time):
        if hasattr(time, "is_leap_year"):
            pytest.skip("Installed cftime version exposes is_leap_year.")

        with pytest.raises(
            AttributeError,
            match="is_leap_year",
        ):
            AddedTimeFeatures._days_in_year(time)


class TestFinalDatasetConfigBranches:
    @pytest.mark.pruned
    def test_effective_condition_property_returns_internal_value(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
        )
        marker = object()
        config._effective_condition = marker

        assert config.effective_condition is marker

    @pytest.mark.pruned
    def test_build_dataset_concrete_method(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
        )

        assert config.build_dataset() == "dataset"

    @pytest.mark.pruned
    def test_available_times_property(self):
        expected = pd.DatetimeIndex(
            [
                "2000-01-01",
                "2001-01-01",
            ]
        )
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            available_times=expected,
        )

        assert config.available_times.equals(expected)

    @pytest.mark.pruned
    def test_dataset_operator_property(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
        )

        assert config.ds_operator is config._operator

    @pytest.mark.pruned
    def test_effective_input_prefers_override(self):
        model = make_data_config(name="model")
        override = make_data_config(name="override")

        config = ConcreteDatasetConfig(
            model=model,
            effective_input=override,
        )

        assert config.effective_input is override

    @pytest.mark.pruned
    def test_effective_input_prefers_model_without_override(self):
        model = make_data_config(name="model")
        condition = make_data_config(name="condition")

        config = ConcreteDatasetConfig(
            model=model,
            condition=condition,
        )

        assert config.effective_input is model

    @pytest.mark.pruned
    def test_effective_input_uses_condition_without_model(self):
        condition = make_data_config(name="condition")

        config = ConcreteDatasetConfig(
            model=None,
            condition=condition,
        )

        assert config.effective_input is condition

    @pytest.mark.pruned
    def test_resolve_condition_returns_self_for_explicit_condition(self):
        condition = make_data_config(name="condition")
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            condition=condition,
            condition_method="ensemble_mean",
        )

        result = config._resolve_condition()

        assert result is config
        assert config.effective_condition is condition

    @pytest.mark.pruned
    def test_resolve_condition_returns_self_for_model_condition(self):
        model = make_data_config()
        converted = make_data_config(
            name="converted",
            ensemble_mean=True,
        )
        config = ConcreteDatasetConfig(
            model=model,
            condition=None,
            condition_method="ensemble_mean",
        )

        with (
            patch.object(
                ConcreteDatasetConfig,
                "_using_model_data_as_condition",
                new_callable=PropertyMock,
                return_value=True,
            ),
            patch.object(
                config,
                "_model_as_condition",
                return_value=converted,
            ) as builder,
        ):
            result = config._resolve_condition()

        assert result is config
        assert config.effective_condition is converted
        builder.assert_called_once_with()

    @pytest.mark.pruned
    def test_resolve_condition_sets_none_when_not_conditioning(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            condition=None,
            condition_method="static",
        )
        config._effective_condition = object()

        with patch.object(
            ConcreteDatasetConfig,
            "_using_model_data_as_condition",
            new_callable=PropertyMock,
            return_value=False,
        ):
            result = config._resolve_condition()

        assert result is config
        assert config.effective_condition is None


class TestFinalUsingModelAsConditionBranches:
    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "method",
        [
            "ensemble_mean",
            "cross_ensemble",
            "same_member",
        ],
    )
    def test_supported_implicit_condition_methods(self, method):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            condition=None,
            condition_method=method,
        )

        assert config._using_model_data_as_condition

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "method",
        [
            "static",
            "unsupported",
        ],
    )
    def test_nonimplicit_condition_methods(self, method):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            condition=None,
            condition_method=method,
        )

        assert not config._using_model_data_as_condition

    @pytest.mark.pruned
    def test_explicit_condition_same_paths_but_different_names(self):
        model = make_data_config(
            paths=["same.nc"],
            names=["tas"],
            realization_list=[0, 1],
        )
        condition = make_data_config(
            name="condition",
            paths=["same.nc"],
            names=["pr"],
            realization_list=[0, 1],
        )
        config = ConcreteDatasetConfig(
            model=model,
            condition=condition,
            condition_method="same_member",
        )

        assert not config._using_model_data_as_condition

    @pytest.mark.pruned
    def test_explicit_condition_same_names_but_different_members(self):
        model = make_data_config(
            paths=["same.nc"],
            names=["tas"],
            realization_list=[0, 1],
        )
        condition = make_data_config(
            name="condition",
            paths=["same.nc"],
            names=["tas"],
            realization_list=[1, 2],
        )
        config = ConcreteDatasetConfig(
            model=model,
            condition=condition,
            condition_method="same_member",
        )

        assert not config._using_model_data_as_condition

    @pytest.mark.pruned
    def test_explicit_condition_same_names_but_different_paths(self):
        model = make_data_config(
            paths=["model.nc"],
            names=["tas"],
            realization_list=[0, 1],
        )
        condition = make_data_config(
            name="condition",
            paths=["condition.nc"],
            names=["tas"],
            realization_list=[0, 1],
        )
        config = ConcreteDatasetConfig(
            model=model,
            condition=condition,
            condition_method="same_member",
        )

        assert not config._using_model_data_as_condition


class TestFinalConditionValidationBranches:
    @pytest.mark.pruned
    def test_no_effective_condition_nonstatic_returns_self(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            condition=None,
            condition_method="ensemble_mean",
        )
        config._effective_condition = None

        assert config._check_condition() is config

    @pytest.mark.pruned
    def test_no_model_check_returns_self(self):
        config = ConcreteDatasetConfig(
            model=None,
            condition=make_data_config(name="condition"),
            condition_method="static",
        )

        assert config._check_model() is config

    @pytest.mark.pruned
    def test_non_same_member_model_returns_self(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(
                ensemble_mean=True,
            ),
            condition_method="cross_ensemble",
        )

        assert config._check_model() is config

    @pytest.mark.pruned
    def test_same_member_nonmean_model_returns_self(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(
                ensemble_mean=False,
                realizations=[0, 1],
            ),
            condition_method="same_member",
        )

        assert config._check_model() is config

    @pytest.mark.pruned
    def test_condition_method_case_insensitive(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            condition_method="ENSEMBLE_MEAN",
        )

        assert config._check_condition_method() is config

    @pytest.mark.pruned
    def test_invalid_condition_method_message_contains_valid_values(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            condition_method="bad_method",
        )

        with pytest.raises(ValueError) as error:
            config._check_condition_method()

        message = str(error.value)
        assert "bad_method" in message
        assert "ensemble_mean" in message
        assert "cross_ensemble" in message
        assert "same_member" in message
        assert "static" in message


class TestFinalTimeFeatureBranches:
    @pytest.mark.pruned
    def test_empty_feature_tuple_length(self):
        features = AddedTimeFeatures(
            TimeFeatureReference(),
            [],
        )

        assert features.time_features == ()
        assert len(features) == 0

    @pytest.mark.pruned
    def test_feature_indices_are_stable(self):
        features = AddedTimeFeatures(
            TimeFeatureReference(),
            None,
        )

        assert features.feature_indices == {
            INIT_TIME_DIM: 0,
            LEAD_TIME_DIM: 1,
            "month_sin": 2,
            "month_cos": 3,
            "day_sin": 4,
            "day_cos": 5,
        }

    @pytest.mark.pruned
    def test_unsupported_feature_message_contains_supported_features(self):
        with pytest.raises(ValueError) as error:
            AddedTimeFeatures(
                TimeFeatureReference(),
                ["hour_sin"],
            )

        message = str(error.value)
        assert "hour_sin" in message
        assert "month_sin" in message
        assert "day_cos" in message

    @pytest.mark.pruned
    def test_build_requires_both_dimensions_at_once(self):
        features = AddedTimeFeatures(
            TimeFeatureReference(),
            [LEAD_TIME_DIM],
        )

        with pytest.raises(ValueError) as error:
            features.build_time_features({})

        message = str(error.value)
        assert INIT_TIME_DIM in message
        assert LEAD_TIME_DIM in message

    @pytest.mark.pruned
    def test_build_replaces_previous_array(self):
        features = AddedTimeFeatures(
            TimeFeatureReference(
                lead_times=[1, 2],
            ),
            [LEAD_TIME_DIM],
        )
        features.time_features_array = np.asarray(
            [[99.0]],
            dtype=np.float32,
        )

        features.build_time_features(
            {
                INIT_TIME_DIM: np.asarray(
                    [
                        "2000-01-01",
                        "2000-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                LEAD_TIME_DIM: np.asarray([1, 2]),
            }
        )

        np.testing.assert_allclose(
            features.time_features_array[:, 0],
            [0.5, 1.0],
        )

    @pytest.mark.pruned
    def test_call_first_sample(self):
        features = AddedTimeFeatures(
            TimeFeatureReference(),
            [LEAD_TIME_DIM],
        )
        features.time_features_array = np.asarray(
            [
                [0.25],
                [0.50],
            ],
            dtype=np.float32,
        )

        result = features(
            0,
            xr.DataArray([1.0]),
        )

        np.testing.assert_allclose(result, [0.25])

    @pytest.mark.pruned
    def test_call_last_sample(self):
        features = AddedTimeFeatures(
            TimeFeatureReference(),
            [LEAD_TIME_DIM],
        )
        features.time_features_array = np.asarray(
            [
                [0.25],
                [0.50],
            ],
            dtype=np.float32,
        )

        result = features(
            1,
            xr.DataArray([1.0]),
        )

        np.testing.assert_allclose(result, [0.50])

    @pytest.mark.pruned
    def test_equality_rejects_different_reference_types(self):
        class OtherReference(TimeFeatureReference):
            pass

        first = AddedTimeFeatures(
            TimeFeatureReference(),
            [LEAD_TIME_DIM],
        )
        second = AddedTimeFeatures(
            OtherReference(),
            [LEAD_TIME_DIM],
        )

        assert first != second

    @pytest.mark.pruned
    def test_equality_rejects_different_common_times(self):
        first = AddedTimeFeatures(
            TimeFeatureReference(
                common_times=[
                    "2000-01-01",
                    "2001-01-01",
                ]
            ),
            [LEAD_TIME_DIM],
        )
        second = AddedTimeFeatures(
            TimeFeatureReference(
                common_times=[
                    "2000-01-01",
                    "2002-01-01",
                ]
            ),
            [LEAD_TIME_DIM],
        )

        assert first != second

    @pytest.mark.pruned
    def test_equality_accepts_identical_references(self):
        first = AddedTimeFeatures(
            TimeFeatureReference(
                lead_times=[1, 2],
                common_times=[
                    "2000-01-01",
                    "2001-01-01",
                ],
            ),
            ["month_cos"],
        )
        second = AddedTimeFeatures(
            TimeFeatureReference(
                lead_times=[1, 2],
                common_times=[
                    "2000-01-01",
                    "2001-01-01",
                ],
            ),
            ["month_cos"],
        )

        assert first == second


class TestFinalMaskAndSamplingBranches:
    def make_dataset(
        self,
        *,
        ensemble_mean=True,
        realizations=None,
    ):
        model = make_data_config(
            ensemble_mean=ensemble_mean,
            realizations=realizations,
            lead_times=[1, 2],
        )
        config = ConcreteDatasetConfig(
            model=model,
            lead_times=[1, 2],
        )

        return ConcreteDataset(
            config=config,
        )

    def test_prepare_sampling_mask_returns_self(self):
        dataset = self.make_dataset()
        dataset.mask = xr.DataArray(
            [[False, False]],
            dims=(
                INIT_TIME_DIM,
                LEAD_TIME_DIM,
            ),
            coords={
                INIT_TIME_DIM: np.asarray(
                    ["2000-01-01"],
                    dtype="datetime64[ns]",
                ),
                LEAD_TIME_DIM: [1, 2],
            },
        )

        result = dataset._prepare_sampling_mask(
            {
                INIT_TIME_DIM: dataset.mask[INIT_TIME_DIM],
                LEAD_TIME_DIM: dataset.mask[LEAD_TIME_DIM],
            }
        )

        assert result is dataset

    @pytest.mark.pruned
    def test_prepare_sampling_mask_selects_requested_lead(self):
        dataset = self.make_dataset()
        dataset.mask = xr.DataArray(
            [[False, False, False]],
            dims=(
                INIT_TIME_DIM,
                LEAD_TIME_DIM,
            ),
            coords={
                INIT_TIME_DIM: np.asarray(
                    ["2000-01-01"],
                    dtype="datetime64[ns]",
                ),
                LEAD_TIME_DIM: [1, 2, 3],
            },
        )

        dataset._prepare_sampling_mask(
            {
                INIT_TIME_DIM: dataset.mask[INIT_TIME_DIM],
                LEAD_TIME_DIM: [2],
            }
        )

        assert dataset.mask.sizes[LEAD_TIME_DIM] == 1
        assert dataset.mask[LEAD_TIME_DIM].item() == 2

    @pytest.mark.pruned
    def test_get_sampling_coords_keeps_dimension_order(self):
        dataset = self.make_dataset()
        dataset.mask = xr.DataArray(
            [[False, False]],
            dims=(
                INIT_TIME_DIM,
                LEAD_TIME_DIM,
            ),
            coords={
                INIT_TIME_DIM: np.asarray(
                    ["2000-01-01"],
                    dtype="datetime64[ns]",
                ),
                LEAD_TIME_DIM: [1, 2],
            },
        ).where(
            xr.DataArray(
                [[True, True]],
                dims=(
                    INIT_TIME_DIM,
                    LEAD_TIME_DIM,
                ),
            )
        )

        result = dataset.get_sampling_coords()

        assert tuple(result) == (
            INIT_TIME_DIM,
            LEAD_TIME_DIM,
        )


class TestFinalIndexBranches:
    @pytest.mark.pruned
    def test_condition_indexes_empty_when_no_sample_dims_match(self):
        condition_data = xr.Dataset(
            {
                "orog": (
                    "lat",
                    [10.0, 20.0],
                )
            },
            coords={
                "lat": [45.0, 46.0],
            },
        )
        condition = make_data_config(
            name="condition",
            ensemble_mean=True,
        )
        condition.dims = condition_data.dims
        condition.indexes = condition_data.indexes

        config = ConcreteDatasetConfig(
            model=make_data_config(),
            condition=condition,
            condition_method="ensemble_mean",
        )
        config._effective_condition = condition
        dataset = ConcreteDataset(config=config)

        result = dataset.get_cond_indexes(
            {
                INIT_TIME_DIM: np.asarray(
                    ["2000-01-01"],
                    dtype="datetime64[ns]",
                ),
                LEAD_TIME_DIM: np.asarray([1]),
            }
        )

        assert result == {}

    @pytest.mark.pruned
    def test_model_indexes_empty_sample_mapping(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
        )
        dataset = ConcreteDataset(
            config=config,
            load_model=True,
        )

        assert dataset.get_model_indexes({}) == {}

    @pytest.mark.pruned
    def test_same_member_index_zero(self):
        condition = make_data_config(
            name="condition",
            ensemble_mean=False,
            realizations=[0, 1],
        )
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            condition=condition,
            condition_method="same_member",
        )
        config._effective_condition = condition
        dataset = ConcreteDataset(config=config)

        result = dataset.get_cond_indexes(
            {
                INIT_TIME_DIM: np.asarray(
                    ["2000-01-01"],
                    dtype="datetime64[ns]",
                ),
                LEAD_TIME_DIM: np.asarray([1]),
                REALIZATION_DIM: np.asarray([0]),
            }
        )

        np.testing.assert_array_equal(
            result[REALIZATION_DIM],
            [0],
        )

    @pytest.mark.pruned
    def test_model_index_dataset_uses_second_sample(self):
        model_data = xr.Dataset(
            {
                "tas": (
                    INIT_TIME_DIM,
                    [10.0, 20.0],
                )
            },
            coords={
                INIT_TIME_DIM: np.asarray(
                    [
                        "2000-01-01",
                        "2001-01-01",
                    ],
                    dtype="datetime64[ns]",
                )
            },
        )
        model = make_data_config()
        model.isel = Mock(
            side_effect=model_data.isel,
        )

        config = ConcreteDatasetConfig(model=model)
        dataset = ConcreteDataset(
            config=config,
            load_model=True,
        )
        dataset.model_indexes = {
            INIT_TIME_DIM: np.asarray([0, 1]),
        }

        with patch.object(
            module,
            "_unwrap_data_variables",
            side_effect=lambda value: value["tas"],
        ):
            result = dataset._index_model_dataset(1)

        model.isel.assert_called_once_with(
            **{
                INIT_TIME_DIM: [1],
            }
        )
        assert result.item() == pytest.approx(20.0)

    def test_condition_index_dataset_uses_second_sample(self):
        condition_data = xr.Dataset(
            {
                "condition": (
                    INIT_TIME_DIM,
                    [10.0, 20.0],
                )
            },
            coords={
                INIT_TIME_DIM: np.asarray(
                    [
                        "2000-01-01",
                        "2001-01-01",
                    ],
                    dtype="datetime64[ns]",
                )
            },
        )
        condition = make_data_config(
            name="condition",
            ensemble_mean=True,
        )
        condition.isel = Mock(
            side_effect=condition_data.isel,
        )

        config = ConcreteDatasetConfig(
            model=make_data_config(),
            condition=condition,
            condition_method="ensemble_mean",
        )
        config._effective_condition = condition
        dataset = ConcreteDataset(config=config)
        dataset.cond_indexes = {
            INIT_TIME_DIM: np.asarray([0, 1]),
        }

        with patch.object(
            module,
            "_unwrap_data_variables",
            side_effect=lambda value: value["condition"],
        ):
            result = dataset._index_condition_dataset(1)

        condition.isel.assert_called_once_with(
            **{
                INIT_TIME_DIM: [1],
            }
        )
        assert result.item() == pytest.approx(20.0)


class TestFinalDatasetUtilityBranches:
    @pytest.mark.pruned
    def test_added_features_dimension_zero(self):
        dataset = ConcreteDataset(
            config=ConcreteDatasetConfig(
                model=make_data_config(),
            ),
            time_features=AddedTimeFeatures(
                TimeFeatureReference(),
                None,
            ),
        )

        assert dataset.get_added_features_dim() == 0

    @pytest.mark.pruned
    def test_added_features_dimension_six(self):
        dataset = ConcreteDataset(
            config=ConcreteDatasetConfig(
                model=make_data_config(),
            ),
            time_features=AddedTimeFeatures(
                TimeFeatureReference(),
                [
                    INIT_TIME_DIM,
                    LEAD_TIME_DIM,
                    "month_sin",
                    "month_cos",
                    "day_sin",
                    "day_cos",
                ],
            ),
        )

        assert dataset.get_added_features_dim() == 6

    @pytest.mark.pruned
    def test_compute_numpy_arrays(self):
        first = np.asarray([1, 2])
        second = np.asarray([3, 4])

        result = ConcreteDataset._compute(
            first,
            second,
        )

        np.testing.assert_array_equal(
            result[0],
            first,
        )
        np.testing.assert_array_equal(
            result[1],
            second,
        )

    @pytest.mark.pruned
    def test_len_single_sample(self):
        dataset = ConcreteDataset(
            config=ConcreteDatasetConfig(
                model=make_data_config(),
            )
        )
        dataset.sample_coords = {
            INIT_TIME_DIM: np.asarray(
                [
                    np.datetime64("2000-01-01"),
                ]
            )
        }

        assert len(dataset) == 1


class TestDatasetConfigABCStructure:
    @pytest.mark.pruned
    def test_abstract_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            DatasetConfigABC()

    @pytest.mark.pruned
    def test_abstract_properties_are_registered(self):
        assert DatasetConfigABC.available_times.__isabstractmethod__
        assert DatasetConfigABC.ds_operator.__isabstractmethod__
        assert DatasetConfigABC.effective_input.__isabstractmethod__
        assert DatasetConfigABC.build_dataset.__isabstractmethod__

    @pytest.mark.pruned
    def test_class_dimensions_match_module_dimensions(self):
        assert DatasetConfigABC.init_time_dim == INIT_TIME_DIM
        assert DatasetConfigABC.lead_time_dim == LEAD_TIME_DIM
        assert DatasetConfigABC.realization_dim == REALIZATION_DIM

    @pytest.mark.pruned
    def test_valid_condition_methods_are_immutable(self):
        assert isinstance(
            DatasetConfigABC._VALID_CONDITION_METHODS,
            frozenset,
        )

    @pytest.mark.pruned
    def test_valid_condition_method_values(self):
        assert DatasetConfigABC._VALID_CONDITION_METHODS == frozenset(
            {
                "ensemble_mean",
                "cross_ensemble",
                "same_member",
                "static",
            }
        )

    @pytest.mark.pruned
    def test_supported_nn_dimensions_are_tuple(self):
        assert isinstance(
            DatasetConfigABC.supported_NN_dimensions,
            tuple,
        )


class TestDatasetABCStructure:
    @pytest.mark.pruned
    def test_abstract_dataset_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            DatasetABC()

    @pytest.mark.pruned
    def test_required_abstract_properties(self):
        assert DatasetABC._load_model.__isabstractmethod__
        assert DatasetABC._write_condition_to_input.__isabstractmethod__
        assert DatasetABC._concat_condition_to_input.__isabstractmethod__

    @pytest.mark.pruned
    def test_concrete_dataset_is_torch_dataset(self):
        dataset = ConcreteDataset(
            config=ConcreteDatasetConfig(
                model=make_data_config(),
            )
        )

        from torch.utils.data import Dataset

        assert isinstance(dataset, Dataset)


class TestLeadTimeConfigAdditionalCoverage:
    @pytest.mark.pruned
    def test_default_start_is_one(self):
        config = lead_time_config(end=3)

        assert config.start == 1
        np.testing.assert_array_equal(
            config.build_lead_times(),
            [1, 2, 3],
        )

    @pytest.mark.pruned
    def test_tuple_list_is_returned_without_conversion(self):
        configured = (1, 3, 5)
        config = lead_time_config(
            list_lead_times=configured,
        )

        assert config.build_lead_times() is configured

    @pytest.mark.pruned
    def test_numpy_array_uses_truth_value_behavior(self):
        config = lead_time_config(
            list_lead_times=np.asarray([1]),
        )

        result = config.build_lead_times()

        np.testing.assert_array_equal(result, [1])

    @pytest.mark.pruned
    def test_descending_range_is_empty(self):
        config = lead_time_config(
            start=5,
            end=2,
        )

        np.testing.assert_array_equal(
            config.build_lead_times(),
            np.asarray([], dtype=int),
        )

    @pytest.mark.pruned
    def test_zero_end_is_valid_when_explicit(self):
        config = lead_time_config(
            start=0,
            end=0,
        )

        np.testing.assert_array_equal(
            config.build_lead_times(),
            [0],
        )

    @pytest.mark.pruned
    def test_none_list_and_none_end_rejected(self):
        with pytest.raises(
            ValueError,
            match="start-end pair",
        ):
            lead_time_config(
                list_lead_times=None,
                end=None,
            )


class TestDatasetConfigSourceValidationMassive:
    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "model,condition",
        [
            (make_data_config(), None),
            (None, make_data_config(name="condition")),
            (
                make_data_config(),
                make_data_config(name="condition"),
            ),
        ],
    )
    def test_source_combinations_return_self(
        self,
        model,
        condition,
    ):
        config = ConcreteDatasetConfig(
            model=model,
            condition=condition,
        )

        result = config._check_required_input_source()

        assert result is config

    @pytest.mark.pruned
    def test_missing_sources_message(self):
        config = ConcreteDatasetConfig(
            model=None,
            condition=None,
        )

        with pytest.raises(ValueError) as error:
            config._check_required_input_source()

        message = str(error.value)
        assert "PPP dataset" in message
        assert "model" in message
        assert "condition" in message

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "method",
        [
            "ensemble_mean",
            "ENSEMBLE_MEAN",
            "Ensemble_Mean",
            "cross_ensemble",
            "CROSS_ENSEMBLE",
            "same_member",
            "SAME_MEMBER",
            "static",
            "STATIC",
        ],
    )
    def test_condition_method_is_case_insensitive(self, method):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            condition_method=method,
        )

        assert config._check_condition_method() is config

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "method",
        [
            "",
            "mean",
            "cross",
            "same",
            "dynamic",
            "none",
        ],
    )
    def test_rejects_invalid_condition_methods(self, method):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            condition_method=method,
        )

        with pytest.raises(
            ValueError,
            match="Invalid condition_method",
        ):
            config._check_condition_method()


class TestUsingModelAsConditionMassive:
    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "method,expected",
        [
            ("ensemble_mean", True),
            ("ENSEMBLE_MEAN", True),
            ("cross_ensemble", True),
            ("CROSS_ENSEMBLE", True),
            ("same_member", True),
            ("SAME_MEMBER", True),
            ("static", False),
            ("STATIC", False),
            ("unsupported", False),
        ],
    )
    def test_implicit_condition_truth_table(
        self,
        method,
        expected,
    ):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            condition=None,
            condition_method=method,
        )

        assert config._using_model_data_as_condition is expected

    @pytest.mark.pruned
    def test_none_method_currently_raises_attribute_error(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            condition=None,
            condition_method=None,
        )

        with does_not_raise(
            AttributeError,
            match="lower",
        ):
            config._using_model_data_as_condition

    @pytest.mark.pruned
    def test_condition_without_model_is_not_model_condition(self):
        config = ConcreteDatasetConfig(
            model=None,
            condition=make_data_config(name="condition"),
            condition_method="static",
        )

        assert not config._using_model_data_as_condition

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "model_paths,condition_paths,"
        "model_names,condition_names,"
        "model_members,condition_members,expected",
        [
            (
                ["a.nc"],
                ["a.nc"],
                ["tas"],
                ["tas"],
                [0, 1],
                [0, 1],
                True,
            ),
            (
                ["a.nc"],
                ["b.nc"],
                ["tas"],
                ["tas"],
                [0, 1],
                [0, 1],
                False,
            ),
            (
                ["a.nc"],
                ["a.nc"],
                ["tas"],
                ["pr"],
                [0, 1],
                [0, 1],
                False,
            ),
            (
                ["a.nc"],
                ["a.nc"],
                ["tas"],
                ["tas"],
                [0, 1],
                [0, 2],
                False,
            ),
            (
                ["a.nc", "b.nc"],
                ["a.nc", "b.nc"],
                ["tas", "pr"],
                ["tas", "pr"],
                None,
                None,
                True,
            ),
        ],
    )
    def test_explicit_condition_identity_truth_table(
        self,
        model_paths,
        condition_paths,
        model_names,
        condition_names,
        model_members,
        condition_members,
        expected,
    ):
        model = make_data_config(
            paths=model_paths,
            names=model_names,
            realization_list=model_members,
        )
        condition = make_data_config(
            name="condition",
            paths=condition_paths,
            names=condition_names,
            realization_list=condition_members,
        )
        config = ConcreteDatasetConfig(
            model=model,
            condition=condition,
            condition_method="same_member",
        )

        assert config._using_model_data_as_condition is expected


class TestResolveConditionMassive:
    @pytest.mark.pruned
    def test_explicit_condition_has_precedence(self):
        condition = make_data_config(name="condition")
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            condition=condition,
            condition_method="ensemble_mean",
        )
        fallback = object()

        with patch.object(
            config,
            "_model_as_condition",
            return_value=fallback,
        ) as builder:
            result = config._resolve_condition()

        assert result is config
        assert config.effective_condition is condition
        builder.assert_not_called()

    @pytest.mark.pruned
    def test_model_condition_is_built_once(self):
        model = make_data_config()
        converted = make_data_config(name="converted")
        config = ConcreteDatasetConfig(
            model=model,
            condition=None,
            condition_method="ensemble_mean",
        )

        with (
            patch.object(
                ConcreteDatasetConfig,
                "_using_model_data_as_condition",
                new_callable=PropertyMock,
                return_value=True,
            ),
            patch.object(
                config,
                "_model_as_condition",
                return_value=converted,
            ) as builder,
        ):
            result = config._resolve_condition()

        assert result is config
        assert config.effective_condition is converted
        builder.assert_called_once_with()

    @pytest.mark.pruned
    def test_no_condition_sets_internal_value_to_none(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            condition=None,
            condition_method="static",
        )
        config._effective_condition = object()

        with patch.object(
            ConcreteDatasetConfig,
            "_using_model_data_as_condition",
            new_callable=PropertyMock,
            return_value=False,
        ):
            result = config._resolve_condition()

        assert result is config
        assert config.effective_condition is None

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "method,ensemble_mean",
        [
            ("ensemble_mean", True),
            ("ENSEMBLE_MEAN", True),
            ("cross_ensemble", False),
            ("same_member", False),
        ],
    )
    def test_model_as_condition_ensemble_mean_flag(
        self,
        method,
        ensemble_mean,
    ):
        model = make_data_config(
            paths=["first.nc"],
            names=["tas"],
            realization_list=[0, 1],
        )
        model.concat_dim = "member"
        model.file_type = "netcdf"
        model.rename_dict = {"old": "new"}
        config = ConcreteDatasetConfig(
            model=model,
            condition_method=method,
        )
        converted = object()

        with patch.object(
            module,
            "ModelDataConfig",
            return_value=converted,
        ) as constructor:
            result = config._model_as_condition()

        assert result is converted
        constructor.assert_called_once_with(
            paths=model.paths,
            names=model.names,
            preprocessing_pipeline=model.preprocessing_pipeline,
            realization_list=model.realization_list,
            concat_dim="member",
            file_type="netcdf",
            ensemble_mean=ensemble_mean,
            rename_dict={"old": "new"},
        )


class TestModelConditionCompatibilityMassive:
    def make_config(
        self,
        model,
        condition,
        *,
        method="ensemble_mean",
        observation=None,
    ):
        config = ConcreteDatasetConfig(
            model=model,
            condition=condition,
            condition_method=method,
            observation=observation,
        )
        config._effective_condition = condition
        return config

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "model,condition",
        [
            (None, make_data_config(name="condition")),
            (make_data_config(), None),
            (None, None),
        ],
    )
    def test_missing_source_skips_compatibility(
        self,
        model,
        condition,
    ):
        config = self.make_config(model, condition)

        assert config._check_model_vs_condition() is None

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "dimension",
        [
            INIT_TIME_DIM,
            LEAD_TIME_DIM,
        ],
    )
    def test_missing_condition_coordinate(self, dimension):
        model = make_data_config()
        condition = make_data_config(
            name="condition",
            paths=["condition.nc"],
            ensemble_mean=True,
        )
        condition.coords.pop(dimension)

        config = self.make_config(model, condition)

        with pytest.raises(
            ValueError,
            match=f"same {dimension}",
        ):
            config._check_model_vs_condition()

    @pytest.mark.pruned
    def test_condition_time_superset_is_accepted(self):
        model = make_data_config(
            times=np.asarray(
                [
                    "2000-01-01",
                    "2001-01-01",
                ],
                dtype="datetime64[ns]",
            )
        )
        condition = make_data_config(
            name="condition",
            paths=["condition.nc"],
            times=np.asarray(
                [
                    "1999-01-01",
                    "2000-01-01",
                    "2001-01-01",
                    "2002-01-01",
                ],
                dtype="datetime64[ns]",
            ),
            ensemble_mean=True,
        )

        config = self.make_config(model, condition)

        assert config._check_model_vs_condition() is None

    @pytest.mark.pruned
    def test_condition_lead_time_superset_is_accepted(self):
        model = make_data_config(
            lead_times=[1, 2],
        )
        condition = make_data_config(
            name="condition",
            paths=["condition.nc"],
            lead_times=[1, 2, 3, 4],
            ensemble_mean=True,
        )

        config = self.make_config(model, condition)

        assert config._check_model_vs_condition() is None

    @pytest.mark.pruned
    def test_missing_model_time_is_reported(self):
        model = make_data_config(
            times=np.asarray(
                [
                    "2000-01-01",
                    "2001-01-01",
                ],
                dtype="datetime64[ns]",
            )
        )
        condition = make_data_config(
            name="condition",
            paths=["condition.nc"],
            times=np.asarray(
                [
                    "2000-01-01",
                ],
                dtype="datetime64[ns]",
            ),
            ensemble_mean=True,
        )

        config = self.make_config(model, condition)

        with pytest.raises(
            ValueError,
            match="same time coordinates",
        ):
            config._check_model_vs_condition()

    @pytest.mark.pruned
    def test_missing_model_lead_time_is_reported(self):
        model = make_data_config(
            lead_times=[1, 2, 3],
        )
        condition = make_data_config(
            name="condition",
            paths=["condition.nc"],
            lead_times=[1, 2],
            ensemble_mean=True,
        )

        config = self.make_config(model, condition)

        with pytest.raises(
            ValueError,
            match="same lead_time coordinates",
        ):
            config._check_model_vs_condition()

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "model_type,condition_type",
        [
            ("datetime", "cftime"),
            ("cftime", "datetime"),
        ],
    )
    def test_time_type_mismatch_both_directions(
        self,
        model_type,
        condition_type,
    ):
        model = make_data_config(
            time_coords_type=model_type,
        )
        condition = make_data_config(
            name="condition",
            paths=["condition.nc"],
            ensemble_mean=True,
            time_coords_type=condition_type,
        )

        config = self.make_config(model, condition)

        with pytest.raises(
            ValueError,
            match="same cftime/datetime type",
        ):
            config._check_model_vs_condition()

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "model_members,condition_members",
        [
            (None, None),
            (None, [0, 1]),
            ([0, 1], None),
        ],
    )
    def test_same_member_requires_both_member_coordinates(
        self,
        model_members,
        condition_members,
    ):
        model = make_data_config(
            realizations=model_members,
            ensemble_mean=False,
        )
        condition = make_data_config(
            name="condition",
            paths=["condition.nc"],
            realizations=condition_members,
            ensemble_mean=False,
        )

        config = self.make_config(
            model,
            condition,
            method="same_member",
        )

        with pytest.raises(
            ValueError,
            match="dims and coords",
        ):
            config._check_model_vs_condition()

    @pytest.mark.pruned
    def test_same_member_coordinate_order_must_match(self):
        model = make_data_config(
            realizations=[0, 1],
            ensemble_mean=False,
        )
        condition = make_data_config(
            name="condition",
            paths=["condition.nc"],
            realizations=[1, 0],
            ensemble_mean=False,
        )

        config = self.make_config(
            model,
            condition,
            method="same_member",
        )

        with pytest.raises(
            ValueError,
            match="same ensemble members",
        ):
            config._check_model_vs_condition()

    @pytest.mark.pruned
    def test_static_skips_time_type_mismatch(self):
        model = make_data_config(
            time_coords_type="datetime",
        )
        condition = make_data_config(
            name="condition",
            paths=["condition.nc"],
            time_coords_type="cftime",
        )
        condition.coords = {}
        condition.realization_list = None

        config = self.make_config(
            model,
            condition,
            method="static",
        )

        assert config._check_model_vs_condition() is None


class TestModelValidationMassive:
    @pytest.mark.pruned
    def test_none_model_returns_self(self):
        config = ConcreteDatasetConfig(
            model=None,
            condition=make_data_config(name="condition"),
            condition_method="static",
        )

        assert config._check_model() is config

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "method",
        [
            None,
            "ensemble_mean",
            "cross_ensemble",
            "static",
        ],
    )
    def test_non_same_member_methods_skip_mean_check(
        self,
        method,
    ):
        config = ConcreteDatasetConfig(
            model=make_data_config(
                ensemble_mean=True,
            ),
            condition_method=method,
        )

        if method is None:
            with does_not_raise():
                config._check_model()
        else:
            assert config._check_model() is config

    @pytest.mark.pruned
    def test_same_member_rejects_mean_model(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(
                ensemble_mean=True,
            ),
            condition_method="same_member",
        )

        with pytest.raises(
            ValueError,
            match="should not be ensemble mean",
        ):
            config._check_model()

    @pytest.mark.pruned
    def test_same_member_accepts_nonmean_model(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(
                ensemble_mean=False,
                realizations=[0, 1],
            ),
            condition_method="same_member",
        )

        assert config._check_model() is config


class TestConditionValidationMassive:
    def make_config(self, condition, method):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            condition=condition,
            condition_method=method,
        )
        config._effective_condition = condition
        return config

    @pytest.mark.pruned
    def test_condition_requires_method(self):
        condition = make_data_config(name="condition")
        config = self.make_config(condition, None)

        with pytest.raises(
            ValueError,
            match="specify condition_method",
        ):
            config._check_condition()

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "method",
        [
            "cross_ensemble",
            "same_member",
            "CROSS_ENSEMBLE",
            "SAME_MEMBER",
        ],
    )
    def test_member_methods_reject_mean_condition(
        self,
        method,
    ):
        condition = make_data_config(
            name="condition",
            ensemble_mean=True,
            realizations=[0, 1],
        )
        config = self.make_config(condition, method)

        with pytest.raises(
            ValueError,
            match="ensemble_mean cannot be True",
        ):
            config._check_condition()

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "method",
        [
            "cross_ensemble",
            "same_member",
        ],
    )
    def test_member_methods_require_member_coordinate(
        self,
        method,
    ):
        condition = make_data_config(
            name="condition",
            ensemble_mean=False,
            realizations=None,
        )
        config = self.make_config(condition, method)

        with pytest.raises(
            ValueError,
            match="dim must exist",
        ):
            config._check_condition()

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "method",
        [
            "cross_ensemble",
            "same_member",
        ],
    )
    def test_member_methods_accept_member_coordinate(
        self,
        method,
    ):
        condition = make_data_config(
            name="condition",
            ensemble_mean=False,
            realizations=[0, 1],
        )
        config = self.make_config(condition, method)

        assert config._check_condition() is config

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "ensemble_mean",
        [
            False,
            None,
            0,
        ],
    )
    def test_ensemble_mean_method_requires_literal_true(
        self,
        ensemble_mean,
    ):
        condition = make_data_config(
            name="condition",
            ensemble_mean=ensemble_mean,
        )
        config = self.make_config(
            condition,
            "ensemble_mean",
        )

        with pytest.raises(
            ValueError,
            match="Ensemble mean must be True",
        ):
            config._check_condition()

    @pytest.mark.pruned
    def test_ensemble_mean_method_accepts_true(self):
        condition = make_data_config(
            name="condition",
            ensemble_mean=True,
        )
        config = self.make_config(
            condition,
            "ensemble_mean",
        )

        assert config._check_condition() is config

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "sampling_dimension",
        [
            INIT_TIME_DIM,
            LEAD_TIME_DIM,
            REALIZATION_DIM,
        ],
    )
    def test_static_rejects_each_sampling_dimension(
        self,
        sampling_dimension,
    ):
        condition = make_data_config(name="condition")
        condition.realization_list = None
        condition.coords = {
            sampling_dimension: make_coord(
                [0],
                sampling_dimension,
            )
        }

        config = self.make_config(
            condition,
            "static",
        )

        with pytest.raises(
            ValueError,
            match="sampling dimensions",
        ):
            config._check_condition()

    @pytest.mark.pruned
    def test_static_rejects_multiple_sampling_dimensions(self):
        condition = make_data_config(name="condition")
        condition.realization_list = None
        condition.coords = {
            INIT_TIME_DIM: make_coord(
                np.asarray(
                    ["2000-01-01"],
                    dtype="datetime64[ns]",
                ),
                INIT_TIME_DIM,
            ),
            LEAD_TIME_DIM: make_coord(
                [1],
                LEAD_TIME_DIM,
            ),
        }

        config = self.make_config(
            condition,
            "static",
        )

        with pytest.raises(
            ValueError,
            match="sampling dimensions",
        ):
            config._check_condition()

    @pytest.mark.pruned
    def test_static_accepts_empty_coordinate_mapping(self):
        condition = make_data_config(name="condition")
        condition.realization_list = None
        condition.coords = {}

        config = self.make_config(
            condition,
            "static",
        )

        assert config._check_condition() is config

    @pytest.mark.pruned
    def test_static_accepts_spatial_coordinates(self):
        condition = make_data_config(name="condition")
        condition.realization_list = None
        condition.coords = {
            "lat": make_coord([45.0], "lat"),
            "lon": make_coord([-123.0], "lon"),
        }

        config = self.make_config(
            condition,
            "static",
        )

        assert config._check_condition() is config

    @pytest.mark.pruned
    def test_static_without_condition_rejected(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            condition=None,
            condition_method="static",
        )
        config._effective_condition = None

        with pytest.raises(
            ValueError,
            match="condition dataset must be specified",
        ):
            config._check_condition()

    @pytest.mark.pruned
    def test_nonstatic_without_effective_condition_returns_self(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            condition=None,
            condition_method="ensemble_mean",
        )
        config._effective_condition = None

        assert config._check_condition() is config


class TestAddedTimeFeaturesInitializationMassive:
    @pytest.mark.pruned
    def test_initial_state(self):
        reference = TimeFeatureReference()
        features = AddedTimeFeatures(
            reference,
            None,
        )

        assert features.reference_config is reference
        assert features.time_features == ()
        assert features.time_features_array is None
        assert features.lead_time_resolution == reference.lead_time_resolution
        assert features.init_time_dim == INIT_TIME_DIM
        assert features.lead_time_dim == LEAD_TIME_DIM

    @pytest.mark.pruned
    def test_reference_min_max_and_span(self):
        reference = TimeFeatureReference(
            common_times=[
                "2000-01-01",
                "2001-01-01",
            ]
        )
        features = AddedTimeFeatures(
            reference,
            None,
        )

        assert features.min_time_ref == pd.Timestamp("2000-01-01")
        assert features.max_time_ref == pd.Timestamp("2001-01-01")
        assert features.time_span_ref == pd.Timedelta(days=366)

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "selected,expected",
        [
            (
                [INIT_TIME_DIM],
                (INIT_TIME_DIM,),
            ),
            (
                [LEAD_TIME_DIM],
                (LEAD_TIME_DIM,),
            ),
            (
                ["month_sin"],
                ("month_sin",),
            ),
            (
                ["month_cos"],
                ("month_cos",),
            ),
            (
                ["day_sin"],
                ("day_sin",),
            ),
            (
                ["day_cos"],
                ("day_cos",),
            ),
        ],
    )
    def test_single_feature_selection(
        self,
        selected,
        expected,
    ):
        features = AddedTimeFeatures(
            TimeFeatureReference(),
            selected,
        )

        assert features.time_features == expected

    @pytest.mark.pruned
    def test_requested_order_is_replaced_by_canonical_order(self):
        features = AddedTimeFeatures(
            TimeFeatureReference(),
            [
                "day_cos",
                "month_cos",
                LEAD_TIME_DIM,
                "month_sin",
                INIT_TIME_DIM,
                "day_sin",
            ],
        )

        assert features.time_features == (
            INIT_TIME_DIM,
            LEAD_TIME_DIM,
            "month_sin",
            "month_cos",
            "day_sin",
            "day_cos",
        )

    @pytest.mark.pruned
    def test_duplicate_features_are_collapsed(self):
        features = AddedTimeFeatures(
            TimeFeatureReference(),
            [
                LEAD_TIME_DIM,
                LEAD_TIME_DIM,
                "month_cos",
                "month_cos",
            ],
        )

        assert features.time_features == (
            LEAD_TIME_DIM,
            "month_cos",
        )

    @pytest.mark.pruned
    def test_multiple_unsupported_features_are_reported(self):
        with pytest.raises(ValueError) as error:
            AddedTimeFeatures(
                TimeFeatureReference(),
                [
                    "hour_sin",
                    "hour_cos",
                ],
            )

        message = str(error.value)
        assert "hour_sin" in message
        assert "hour_cos" in message


class TestAddedTimeFeaturesCalendarMassive:
    @pytest.mark.parametrize(
        "time,expected",
        [
            (
                cftime.Datetime360Day(2000, 1, 1),
                360,
            ),
            (
                cftime.DatetimeNoLeap(2000, 1, 1),
                365,
            ),
            (
                cftime.DatetimeAllLeap(2000, 1, 1),
                366,
            ),
            (
                np.datetime64("2000-01-01"),
                366,
            ),
            (
                np.datetime64("1900-01-01"),
                365,
            ),
            (
                np.datetime64("2004-01-01"),
                366,
            ),
            (
                datetime.datetime(2001, 1, 1),
                365,
            ),
            (
                datetime.datetime(2000, 1, 1),
                366,
            ),
        ],
    )
    def test_supported_calendar_lengths(
        self,
        time,
        expected,
    ):
        assert AddedTimeFeatures._days_in_year(time) == expected

    @pytest.mark.parametrize(
        "time",
        [
            cftime.DatetimeGregorian(2000, 1, 1),
            cftime.DatetimeProlepticGregorian(
                2000,
                1,
                1,
            ),
            cftime.DatetimeJulian(2000, 1, 1),
        ],
    )
    def test_generic_cftime_calendar_current_behavior(
        self,
        time,
    ):
        if hasattr(time, "is_leap_year"):
            result = AddedTimeFeatures._days_in_year(time)
            assert result in {365, 366}
        else:
            with pytest.raises(
                AttributeError,
                match="is_leap_year",
            ):
                AddedTimeFeatures._days_in_year(time)


class TestBuildTimeFeaturesMassive:
    def coordinates(
        self,
        *,
        times=None,
        lead_times=None,
    ):
        if times is None:
            times = np.asarray(
                [
                    "2000-01-01",
                    "2000-04-01",
                    "2000-07-01",
                ],
                dtype="datetime64[ns]",
            )

        if lead_times is None:
            lead_times = np.asarray([1, 2, 4])

        return {
            INIT_TIME_DIM: np.asarray(times),
            LEAD_TIME_DIM: np.asarray(lead_times),
        }

    @pytest.mark.pruned
    def test_missing_both_dimensions(self):
        features = AddedTimeFeatures(
            TimeFeatureReference(),
            [LEAD_TIME_DIM],
        )

        with pytest.raises(ValueError) as error:
            features.build_time_features({})

        message = str(error.value)
        assert INIT_TIME_DIM in message
        assert LEAD_TIME_DIM in message

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "missing_dimension",
        [
            INIT_TIME_DIM,
            LEAD_TIME_DIM,
        ],
    )
    def test_missing_one_dimension(self, missing_dimension):
        coordinates = self.coordinates()
        coordinates.pop(missing_dimension)

        features = AddedTimeFeatures(
            TimeFeatureReference(),
            [LEAD_TIME_DIM],
        )

        with pytest.raises(
            ValueError,
            match="missing required",
        ):
            features.build_time_features(coordinates)

    @pytest.mark.pruned
    def test_no_features_returns_self(self):
        features = AddedTimeFeatures(
            TimeFeatureReference(),
            None,
        )

        result = features.build_time_features(self.coordinates())

        assert result is features
        assert features.time_features_array is None

    @pytest.mark.pruned
    def test_no_features_does_not_call_add_lead_times(
        self,
        monkeypatch,
    ):
        add_times = Mock()
        monkeypatch.setattr(
            module,
            "add_lead_times",
            add_times,
        )
        features = AddedTimeFeatures(
            TimeFeatureReference(),
            [],
        )

        features.build_time_features(self.coordinates())

        add_times.assert_not_called()

    @pytest.mark.pruned
    def test_add_lead_times_receives_one_dimensional_arrays(
        self,
        monkeypatch,
    ):
        coordinates = self.coordinates()
        target_times = np.asarray(
            [
                "2000-01-01",
                "2000-05-01",
                "2000-10-01",
            ],
            dtype="datetime64[ns]",
        )
        add_times = Mock(return_value=target_times)
        monkeypatch.setattr(
            module,
            "add_lead_times",
            add_times,
        )

        features = AddedTimeFeatures(
            TimeFeatureReference(),
            ["month_cos"],
        )
        features.build_time_features(coordinates)

        add_times.assert_called_once_with(
            init_times=coordinates[INIT_TIME_DIM],
            lead_times=coordinates[LEAD_TIME_DIM],
            lead_time_resolution=(features.lead_time_resolution),
        )

    @pytest.mark.pruned
    def test_normalized_lead_times_use_reference_maximum(self):
        features = AddedTimeFeatures(
            TimeFeatureReference(
                lead_times=[1, 5, 10],
            ),
            [LEAD_TIME_DIM],
        )

        features.build_time_features(
            self.coordinates(
                lead_times=[1, 5, 10],
            )
        )

        np.testing.assert_allclose(
            features.time_features_array[:, 0],
            [0.1, 0.5, 1.0],
        )

    @pytest.mark.pruned
    def test_normalized_time_uses_target_time(self):
        reference = TimeFeatureReference(
            common_times=[
                "2000-01-01",
                "2001-01-01",
            ]
        )
        features = AddedTimeFeatures(
            reference,
            [INIT_TIME_DIM],
        )

        features.build_time_features(
            self.coordinates(
                times=np.asarray(
                    [
                        "2000-01-01",
                        "2000-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                lead_times=[1, 13],
            )
        )

        assert features.time_features_array[0, 0] == pytest.approx(0.0)
        assert features.time_features_array[1, 0] == pytest.approx(1.0)

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "month,expected_sin,expected_cos",
        [
            (1, 0.0, 1.0),
            (4, 1.0, 0.0),
            (7, 0.0, -1.0),
            (10, -1.0, 0.0),
        ],
    )
    def test_month_cycle_quarters(
        self,
        month,
        expected_sin,
        expected_cos,
    ):
        features = AddedTimeFeatures(
            TimeFeatureReference(),
            [
                "month_sin",
                "month_cos",
            ],
        )
        date = np.datetime64(f"2000-{month:02d}-01")

        features.build_time_features(
            self.coordinates(
                times=np.asarray([date]),
                lead_times=[1],
            )
        )

        assert features.time_features_array[
            0,
            0,
        ] == pytest.approx(
            expected_sin,
            abs=1e-6,
        )
        assert features.time_features_array[
            0,
            1,
        ] == pytest.approx(
            expected_cos,
            abs=1e-6,
        )

    def test_output_shape_all_features(self):
        features = AddedTimeFeatures(
            TimeFeatureReference(),
            [
                INIT_TIME_DIM,
                LEAD_TIME_DIM,
                "month_sin",
                "month_cos",
                "day_sin",
                "day_cos",
            ],
        )

        result = features.build_time_features(self.coordinates())

        assert result is features
        assert features.time_features_array.shape == (
            3,
            6,
        )
        assert features.time_features_array.dtype == np.float32

    @pytest.mark.pruned
    def test_second_build_replaces_first_array(self):
        features = AddedTimeFeatures(
            TimeFeatureReference(
                lead_times=[1, 2],
            ),
            [LEAD_TIME_DIM],
        )

        features.build_time_features(
            self.coordinates(
                times=np.asarray(
                    ["2000-01-01"],
                    dtype="datetime64[ns]",
                ),
                lead_times=[1],
            )
        )
        first = features.time_features_array.copy()

        features.build_time_features(
            self.coordinates(
                times=np.asarray(
                    [
                        "2000-01-01",
                        "2000-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                lead_times=[1, 2],
            )
        )

        assert first.shape == (1, 1)
        assert features.time_features_array.shape == (
            2,
            1,
        )


class TestAddedTimeFeaturesCallMassive:
    def make_features(self):
        features = AddedTimeFeatures(
            TimeFeatureReference(),
            [
                LEAD_TIME_DIM,
                "month_cos",
            ],
        )
        features.time_features_array = np.asarray(
            [
                [0.25, 1.0],
                [0.50, 0.0],
                [1.00, -1.0],
            ],
            dtype=np.float32,
        )
        return features

    def test_missing_built_array_raises(self):
        features = AddedTimeFeatures(
            TimeFeatureReference(),
            [LEAD_TIME_DIM],
        )

        with pytest.raises(
            RuntimeError,
            match="must be built",
        ):
            features(
                0,
                xr.DataArray([1.0]),
            )

    @pytest.mark.parametrize(
        "index",
        [
            -10,
            -1,
            3,
            10,
        ],
    )
    def test_invalid_indexes(self, index):
        features = self.make_features()

        with pytest.raises(
            IndexError,
            match="out of bounds",
        ):
            features(
                index,
                xr.DataArray([1.0]),
            )

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "index,expected",
        [
            (0, [0.25, 1.0]),
            (1, [0.50, 0.0]),
            (2, [1.00, -1.0]),
        ],
    )
    def test_valid_indexes(
        self,
        index,
        expected,
    ):
        features = self.make_features()

        result = features(
            index,
            xr.DataArray([1.0]),
        )

        np.testing.assert_allclose(result, expected)

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "shape,expected_shape",
        [
            ((2,), (2,)),
            ((2, 3), (2,)),
            ((2, 3, 4), (2, 3, 4)),
            ((2, 3, 4, 5), (2, 3, 4, 5)),
            ((2, 1, 1, 1, 1), (2, 1, 1, 1, 1)),
        ],
    )
    def test_broadcast_shapes(
        self,
        shape,
        expected_shape,
    ):
        features = self.make_features()
        data = xr.DataArray(np.ones(shape))

        result = features(0, data)

        assert result.shape == expected_shape

    @pytest.mark.pruned
    def test_broadcast_values(self):
        features = self.make_features()
        data = xr.DataArray(
            np.ones((2, 3, 4)),
            dims=("channels", "lat", "lon"),
        )

        result = features(0, data)

        np.testing.assert_allclose(
            result[0],
            0.25,
        )
        np.testing.assert_allclose(
            result[1],
            1.0,
        )

    def test_broadcast_result_is_writable_copy(self):
        features = self.make_features()
        data = xr.DataArray(
            np.ones((2, 2, 2)),
        )

        result = features(0, data)
        result[0, 0, 0] = 99.0

        assert features.time_features_array[0, 0] == pytest.approx(0.25)

    @pytest.mark.pruned
    def test_len_matches_selected_features(self):
        features = self.make_features()

        assert len(features) == 2

    def test_equality_with_unrelated_object(self):
        features = self.make_features()

        assert features.__eq__(object()) is NotImplemented

    @pytest.mark.pruned
    def test_equal_objects(self):
        first = AddedTimeFeatures(
            TimeFeatureReference(
                lead_times=[1, 2],
                common_times=[
                    "2000-01-01",
                    "2001-01-01",
                ],
            ),
            [LEAD_TIME_DIM],
        )
        second = AddedTimeFeatures(
            TimeFeatureReference(
                lead_times=[1, 2],
                common_times=[
                    "2000-01-01",
                    "2001-01-01",
                ],
            ),
            [LEAD_TIME_DIM],
        )

        assert first == second

    @pytest.mark.pruned
    def test_unequal_feature_selection(self):
        first = AddedTimeFeatures(
            TimeFeatureReference(),
            [LEAD_TIME_DIM],
        )
        second = AddedTimeFeatures(
            TimeFeatureReference(),
            ["month_cos"],
        )

        assert first != second

    @pytest.mark.pruned
    def test_unequal_lead_times(self):
        first = AddedTimeFeatures(
            TimeFeatureReference(
                lead_times=[1, 2],
            ),
            [LEAD_TIME_DIM],
        )
        second = AddedTimeFeatures(
            TimeFeatureReference(
                lead_times=[1, 3],
            ),
            [LEAD_TIME_DIM],
        )

        assert first != second

    @pytest.mark.pruned
    def test_unequal_common_times(self):
        first = AddedTimeFeatures(
            TimeFeatureReference(
                common_times=[
                    "2000-01-01",
                    "2001-01-01",
                ]
            ),
            [LEAD_TIME_DIM],
        )
        second = AddedTimeFeatures(
            TimeFeatureReference(
                common_times=[
                    "2000-01-01",
                    "2002-01-01",
                ]
            ),
            [LEAD_TIME_DIM],
        )

        assert first != second

    @pytest.mark.pruned
    def test_unequal_reference_types(self):
        class OtherReference(TimeFeatureReference):
            pass

        first = AddedTimeFeatures(
            TimeFeatureReference(),
            [LEAD_TIME_DIM],
        )
        second = AddedTimeFeatures(
            OtherReference(),
            [LEAD_TIME_DIM],
        )

        assert first != second


class TestDatasetCheckInitMassive:
    def make_dataset(
        self,
        *,
        requested=None,
        available=None,
        fitted=True,
    ):
        if requested is None:
            requested = np.asarray(
                ["2000-01-01"],
                dtype="datetime64[ns]",
            )

        if available is None:
            available = pd.DatetimeIndex(["2000-01-01"])

        config = ConcreteDatasetConfig(
            model=make_data_config(),
            available_times=available,
        )
        config._fitted_preprocessors = fitted

        return ConcreteDataset(
            config=config,
            requested_times=np.asarray(requested),
        )

    def test_calls_time_sequence_validator(
        self,
        monkeypatch,
    ):
        dataset = self.make_dataset()
        validator = Mock()
        monkeypatch.setattr(
            module,
            "_validate_time_sequence",
            validator,
        )

        dataset._check_init()

        validator.assert_called_once_with(dataset.requested_times)

    @pytest.mark.pruned
    def test_time_validation_failure_is_propagated(
        self,
        monkeypatch,
    ):
        dataset = self.make_dataset()
        monkeypatch.setattr(
            module,
            "_validate_time_sequence",
            Mock(side_effect=TypeError("invalid time sequence")),
        )

        with pytest.raises(
            TypeError,
            match="invalid time sequence",
        ):
            dataset._check_init()

    @pytest.mark.pruned
    def test_unfitted_preprocessors_rejected(self):
        dataset = self.make_dataset(fitted=False)

        with pytest.raises(
            RuntimeError,
            match="fit preprocessors",
        ):
            dataset._check_init()

    @pytest.mark.pruned
    def test_single_missing_time(self):
        dataset = self.make_dataset(
            requested=np.asarray(
                ["1990-01-01"],
                dtype="datetime64[ns]",
            ),
        )

        with pytest.raises(
            ValueError,
            match="initialization times",
        ):
            dataset._check_init()

    @pytest.mark.pruned
    def test_multiple_missing_times_reported(self):
        dataset = self.make_dataset(
            requested=np.asarray(
                [
                    "1990-01-01",
                    "1991-01-01",
                ],
                dtype="datetime64[ns]",
            ),
        )

        with pytest.raises(ValueError) as error:
            dataset._check_init()

        message = str(error.value)
        assert "1990" in message
        assert "1991" in message

    @pytest.mark.pruned
    def test_all_available_times_accepted(self):
        dataset = self.make_dataset(
            requested=np.asarray(
                [
                    "2000-01-01",
                    "2001-01-01",
                ],
                dtype="datetime64[ns]",
            ),
            available=pd.DatetimeIndex(
                [
                    "2000-01-01",
                    "2001-01-01",
                ]
            ),
        )

        assert dataset._check_init() is None


class TestDatasetResolveMaskMassive:
    def make_dataset(self, mask=None):
        model = make_data_config(
            lead_times=[1, 2],
        )
        config = ConcreteDatasetConfig(
            model=model,
            available_times=pd.DatetimeIndex(
                [
                    "2000-01-01",
                    "2001-01-01",
                ]
            ),
        )

        return ConcreteDataset(
            config=config,
            mask=mask,
        )

    def template(self):
        return xr.DataArray(
            np.ones((2, 2), dtype=bool),
            dims=(
                INIT_TIME_DIM,
                LEAD_TIME_DIM,
            ),
            coords={
                INIT_TIME_DIM: np.asarray(
                    [
                        "2000-01-01",
                        "2001-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                LEAD_TIME_DIM: [1, 2],
            },
        )

    @pytest.mark.pruned
    def test_none_mask_calls_create_train_mask(
        self,
        monkeypatch,
    ):
        dataset = self.make_dataset()
        template = self.template()
        creator = Mock(return_value=template)
        monkeypatch.setattr(
            module,
            "_create_train_mask",
            creator,
        )

        dataset._resolve_mask()

        creator.assert_called_once_with(
            init_times=dataset.config.available_times,
            lead_times=dataset.config.input_lead_times,
            lead_time_resolution=module.lead_time_resolution,
        )

    @pytest.mark.pruned
    def test_default_mask_is_all_false(
        self,
        monkeypatch,
    ):
        dataset = self.make_dataset()
        monkeypatch.setattr(
            module,
            "_create_train_mask",
            Mock(return_value=self.template()),
        )

        dataset._resolve_mask()

        assert dataset.mask.dtype == bool
        assert not bool(dataset.mask.any())

    @pytest.mark.pruned
    def test_existing_mask_skips_creation(
        self,
        monkeypatch,
    ):
        existing = self.template()
        dataset = self.make_dataset(mask=existing)
        creator = Mock()
        monkeypatch.setattr(
            module,
            "_create_train_mask",
            creator,
        )

        dataset._resolve_mask()

        creator.assert_not_called()
        assert dataset.mask is existing

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "dims",
        [
            (),
            (INIT_TIME_DIM,),
            (LEAD_TIME_DIM,),
            ("lat",),
            ("lat", "lon"),
        ],
    )
    def test_missing_mask_dimensions(self, dims):
        shape = tuple(1 for _ in dims)
        mask = xr.DataArray(
            np.zeros(shape, dtype=bool),
            dims=dims,
        )
        dataset = self.make_dataset(mask=mask)

        with pytest.raises(
            ValueError,
            match="mask must have",
        ):
            dataset._resolve_mask()

    @pytest.mark.pruned
    def test_extra_mask_dimensions_are_allowed(self):
        mask = xr.DataArray(
            np.zeros((1, 1, 2), dtype=bool),
            dims=(
                INIT_TIME_DIM,
                LEAD_TIME_DIM,
                "extra",
            ),
            coords={
                INIT_TIME_DIM: [
                    np.datetime64("2000-01-01"),
                ],
                LEAD_TIME_DIM: [1],
                "extra": [0, 1],
            },
        )
        dataset = self.make_dataset(mask=mask)

        assert dataset._resolve_mask() is None


class TestSamplingSelectorsMassive:
    @pytest.mark.pruned
    def test_selectors_use_config_method(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            lead_times=[1, 2],
        )
        selected_times = xr.DataArray(
            np.asarray(
                ["2000-01-01"],
                dtype="datetime64[ns]",
            ),
            dims=(INIT_TIME_DIM,),
        )
        config.get_input_times = Mock(
            return_value=selected_times,
        )
        dataset = ConcreteDataset(config=config)

        result = dataset._sampling_times_selectors

        assert result == {
            INIT_TIME_DIM: selected_times,
            LEAD_TIME_DIM: [1, 2],
        }
        config.get_input_times.assert_called_once_with(dataset.requested_times)


class TestPrepareSamplingMaskMassive:
    def make_dataset(
        self,
        *,
        ensemble_mean=True,
        realizations=None,
    ):
        model = make_data_config(
            ensemble_mean=ensemble_mean,
            realizations=realizations,
        )
        config = ConcreteDatasetConfig(
            model=model,
        )
        mask = xr.DataArray(
            [
                [False, True],
                [False, False],
            ],
            dims=(
                INIT_TIME_DIM,
                LEAD_TIME_DIM,
            ),
            coords={
                INIT_TIME_DIM: np.asarray(
                    [
                        "2000-01-01",
                        "2001-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                LEAD_TIME_DIM: [1, 2],
            },
        )

        return ConcreteDataset(
            config=config,
            mask=mask,
        )

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "selectors",
        [
            {},
            {INIT_TIME_DIM: [np.datetime64("2000-01-01")]},
            {
                LEAD_TIME_DIM: [1],
            },
        ],
    )
    def test_missing_selectors(self, selectors):
        dataset = self.make_dataset()

        with pytest.raises(
            ValueError,
            match="No selectors provided",
        ):
            dataset._prepare_sampling_mask(selectors)

    @pytest.mark.pruned
    def test_returns_self(self):
        dataset = self.make_dataset()

        result = dataset._prepare_sampling_mask(
            {
                INIT_TIME_DIM: dataset.mask[INIT_TIME_DIM],
                LEAD_TIME_DIM: dataset.mask[LEAD_TIME_DIM],
            }
        )

        assert result is dataset

    @pytest.mark.pruned
    def test_selects_time_subset(self):
        dataset = self.make_dataset()

        dataset._prepare_sampling_mask(
            {
                INIT_TIME_DIM: [np.datetime64("2001-01-01")],
                LEAD_TIME_DIM: [1, 2],
            }
        )

        assert dataset.mask.sizes[INIT_TIME_DIM] == 1
        assert dataset.mask[INIT_TIME_DIM].values[0] == np.datetime64(
            "2001-01-01", "ns"
        )

    @pytest.mark.pruned
    def test_selects_lead_subset(self):
        dataset = self.make_dataset()

        dataset._prepare_sampling_mask(
            {
                INIT_TIME_DIM: dataset.mask[INIT_TIME_DIM],
                LEAD_TIME_DIM: [2],
            }
        )

        assert dataset.mask.sizes[LEAD_TIME_DIM] == 1
        assert dataset.mask[LEAD_TIME_DIM].item() == 2

    @pytest.mark.pruned
    def test_true_mask_values_become_nan(self):
        dataset = self.make_dataset()

        dataset._prepare_sampling_mask(
            {
                INIT_TIME_DIM: dataset.mask[INIT_TIME_DIM],
                LEAD_TIME_DIM: dataset.mask[LEAD_TIME_DIM],
            }
        )

        assert np.isnan(
            dataset.mask.sel(
                {
                    INIT_TIME_DIM: np.datetime64("2000-01-01"),
                    LEAD_TIME_DIM: 2,
                }
            ).item()
        )

    @pytest.mark.pruned
    def test_ensemble_mean_skips_realization_expansion(self):
        dataset = self.make_dataset(
            ensemble_mean=True,
            realizations=[0, 1],
        )

        dataset._prepare_sampling_mask(
            {
                INIT_TIME_DIM: dataset.mask[INIT_TIME_DIM],
                LEAD_TIME_DIM: dataset.mask[LEAD_TIME_DIM],
            }
        )

        assert REALIZATION_DIM not in dataset.mask.dims

    @pytest.mark.pruned
    def test_nonmean_without_realization_coordinate_skips_expansion(
        self,
    ):
        dataset = self.make_dataset(
            ensemble_mean=False,
            realizations=None,
        )

        dataset._prepare_sampling_mask(
            {
                INIT_TIME_DIM: dataset.mask[INIT_TIME_DIM],
                LEAD_TIME_DIM: dataset.mask[LEAD_TIME_DIM],
            }
        )

        assert REALIZATION_DIM not in dataset.mask.dims

    @pytest.mark.pruned
    def test_nonmean_with_realizations_expands_first_axis(self):
        dataset = self.make_dataset(
            ensemble_mean=False,
            realizations=[0, 1, 2],
        )

        dataset._prepare_sampling_mask(
            {
                INIT_TIME_DIM: dataset.mask[INIT_TIME_DIM],
                LEAD_TIME_DIM: dataset.mask[LEAD_TIME_DIM],
            }
        )

        assert dataset.mask.dims[0] == REALIZATION_DIM
        assert dataset.mask.sizes[REALIZATION_DIM] == 3
        np.testing.assert_array_equal(
            dataset.mask[REALIZATION_DIM],
            [0, 1, 2],
        )


class TestGetSamplingCoordsMassive:
    def make_dataset(self, mask):
        return ConcreteDataset(
            config=ConcreteDatasetConfig(
                model=make_data_config(),
            ),
            mask=mask,
        )

    @pytest.mark.pruned
    def test_all_locations_valid(self):
        mask = xr.DataArray(
            np.zeros((2, 2), dtype=float),
            dims=(
                INIT_TIME_DIM,
                LEAD_TIME_DIM,
            ),
            coords={
                INIT_TIME_DIM: np.asarray(
                    [
                        "2000-01-01",
                        "2001-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                LEAD_TIME_DIM: [1, 2],
            },
        )
        dataset = self.make_dataset(mask)

        result = dataset.get_sampling_coords()

        assert len(result[INIT_TIME_DIM]) == 4
        assert len(result[LEAD_TIME_DIM]) == 4

    @pytest.mark.pruned
    def test_nan_location_is_removed(self):
        mask = xr.DataArray(
            [
                [0.0, np.nan],
                [0.0, 0.0],
            ],
            dims=(
                INIT_TIME_DIM,
                LEAD_TIME_DIM,
            ),
            coords={
                INIT_TIME_DIM: np.asarray(
                    [
                        "2000-01-01",
                        "2001-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                LEAD_TIME_DIM: [1, 2],
            },
        )
        dataset = self.make_dataset(mask)

        result = dataset.get_sampling_coords()

        assert len(result[INIT_TIME_DIM]) == 3
        assert len(result[LEAD_TIME_DIM]) == 3

    @pytest.mark.pruned
    def test_all_nan_returns_empty_coordinate_arrays(self):
        mask = xr.DataArray(
            np.full((2, 2), np.nan),
            dims=(
                INIT_TIME_DIM,
                LEAD_TIME_DIM,
            ),
            coords={
                INIT_TIME_DIM: np.asarray(
                    [
                        "2000-01-01",
                        "2001-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                LEAD_TIME_DIM: [1, 2],
            },
        )
        dataset = self.make_dataset(mask)

        result = dataset.get_sampling_coords()

        assert result[INIT_TIME_DIM].size == 0
        assert result[LEAD_TIME_DIM].size == 0

    @pytest.mark.pruned
    def test_realization_dimension_is_returned(self):
        mask = xr.DataArray(
            np.zeros((2, 1, 1)),
            dims=(
                REALIZATION_DIM,
                INIT_TIME_DIM,
                LEAD_TIME_DIM,
            ),
            coords={
                REALIZATION_DIM: [0, 1],
                INIT_TIME_DIM: np.asarray(
                    ["2000-01-01"],
                    dtype="datetime64[ns]",
                ),
                LEAD_TIME_DIM: [1],
            },
        )
        dataset = self.make_dataset(mask)

        result = dataset.get_sampling_coords()

        assert tuple(result) == (
            REALIZATION_DIM,
            INIT_TIME_DIM,
            LEAD_TIME_DIM,
        )
        np.testing.assert_array_equal(
            result[REALIZATION_DIM],
            [0, 1],
        )


class TestGetModelIndexesMassive:
    def make_dataset(
        self,
        *,
        load_model=True,
        realizations=None,
    ):
        model = make_data_config(
            times=np.asarray(
                [
                    "2000-01-01",
                    "2001-01-01",
                ],
                dtype="datetime64[ns]",
            ),
            lead_times=[1, 2],
            realizations=realizations,
        )
        config = ConcreteDatasetConfig(model=model)

        return ConcreteDataset(
            config=config,
            load_model=load_model,
        )

    @pytest.mark.pruned
    def test_disabled_model_returns_none(self):
        dataset = self.make_dataset(load_model=False)

        assert dataset.get_model_indexes({}) is None

    @pytest.mark.pruned
    def test_empty_coordinate_mapping_returns_empty_indexes(self):
        dataset = self.make_dataset()

        assert dataset.get_model_indexes({}) == {}

    @pytest.mark.pruned
    def test_time_indexes(self):
        dataset = self.make_dataset()

        result = dataset.get_model_indexes(
            {
                INIT_TIME_DIM: np.asarray(
                    [
                        "2001-01-01",
                        "2000-01-01",
                    ],
                    dtype="datetime64[ns]",
                )
            }
        )

        np.testing.assert_array_equal(
            result[INIT_TIME_DIM],
            [1, 0],
        )

    @pytest.mark.pruned
    def test_lead_indexes(self):
        dataset = self.make_dataset()

        result = dataset.get_model_indexes({LEAD_TIME_DIM: np.asarray([2, 1])})

        np.testing.assert_array_equal(
            result[LEAD_TIME_DIM],
            [1, 0],
        )

    @pytest.mark.pruned
    def test_realization_indexes(self):
        dataset = self.make_dataset(
            realizations=[0, 1, 2],
        )

        result = dataset.get_model_indexes({REALIZATION_DIM: np.asarray([2, 0])})

        np.testing.assert_array_equal(
            result[REALIZATION_DIM],
            [2, 0],
        )

    @pytest.mark.pruned
    def test_all_index_dimensions(self):
        dataset = self.make_dataset(
            realizations=[0, 1],
        )

        result = dataset.get_model_indexes(
            {
                INIT_TIME_DIM: np.asarray(
                    ["2001-01-01"],
                    dtype="datetime64[ns]",
                ),
                LEAD_TIME_DIM: np.asarray([2]),
                REALIZATION_DIM: np.asarray([1]),
            }
        )

        assert set(result) == {
            INIT_TIME_DIM,
            LEAD_TIME_DIM,
            REALIZATION_DIM,
        }
        assert all(value.item() == 1 for value in result.values())

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "dimension,bad_value",
        [
            (
                INIT_TIME_DIM,
                np.datetime64("1990-01-01"),
            ),
            (LEAD_TIME_DIM, 99),
            (REALIZATION_DIM, 9),
        ],
    )
    def test_missing_coordinate_by_dimension(
        self,
        dimension,
        bad_value,
    ):
        dataset = self.make_dataset(
            realizations=[0, 1],
        )

        with pytest.raises(
            ValueError,
            match="model dataset",
        ):
            dataset.get_model_indexes({dimension: np.asarray([bad_value])})

    @pytest.mark.pruned
    def test_multiple_missing_dimensions_are_reported(self):
        dataset = self.make_dataset(
            realizations=[0, 1],
        )

        with pytest.raises(ValueError) as error:
            dataset.get_model_indexes(
                {
                    INIT_TIME_DIM: np.asarray(
                        ["1990-01-01"],
                        dtype="datetime64[ns]",
                    ),
                    LEAD_TIME_DIM: np.asarray([99]),
                    REALIZATION_DIM: np.asarray([9]),
                }
            )

        message = str(error.value)
        assert INIT_TIME_DIM in message
        assert LEAD_TIME_DIM in message
        assert REALIZATION_DIM in message


class TestGetConditionIndexesMassive:
    def make_dataset(
        self,
        *,
        method,
        condition,
    ):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            condition=condition,
            condition_method=method,
        )
        config._effective_condition = condition

        return ConcreteDataset(config=config)

    @pytest.mark.pruned
    def test_none_condition_returns_none(self):
        dataset = self.make_dataset(
            method="ensemble_mean",
            condition=None,
        )

        assert dataset.get_cond_indexes({}) is None

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "method",
        [
            "static",
            "STATIC",
        ],
    )
    def test_static_condition_returns_none(self, method):
        dataset = self.make_dataset(
            method=method,
            condition=make_data_config(name="condition"),
        )

        assert dataset.get_cond_indexes({}) is None

    @pytest.mark.pruned
    def test_empty_sample_coordinates_returns_empty_indexes(self):
        condition = make_data_config(
            name="condition",
            ensemble_mean=True,
        )
        dataset = self.make_dataset(
            method="ensemble_mean",
            condition=condition,
        )

        assert dataset.get_cond_indexes({}) == {}

    @pytest.mark.pruned
    def test_unknown_dimensions_are_ignored(self):
        condition = make_data_config(
            name="condition",
            ensemble_mean=True,
        )
        dataset = self.make_dataset(
            method="ensemble_mean",
            condition=condition,
        )

        result = dataset.get_cond_indexes(
            {
                "unknown": np.asarray([1]),
            }
        )

        assert result == {}

    @pytest.mark.pruned
    def test_regular_condition_indexes_time_and_lead(self):
        condition = make_data_config(
            name="condition",
            ensemble_mean=True,
        )
        dataset = self.make_dataset(
            method="ensemble_mean",
            condition=condition,
        )

        result = dataset.get_cond_indexes(
            {
                INIT_TIME_DIM: np.asarray(
                    ["2001-01-01"],
                    dtype="datetime64[ns]",
                ),
                LEAD_TIME_DIM: np.asarray([2]),
            }
        )

        np.testing.assert_array_equal(
            result[INIT_TIME_DIM],
            [1],
        )
        np.testing.assert_array_equal(
            result[LEAD_TIME_DIM],
            [1],
        )

    @pytest.mark.pruned
    def test_cross_ensemble_excludes_realization(self):
        condition = make_data_config(
            name="condition",
            ensemble_mean=False,
            realizations=[0, 1],
        )
        dataset = self.make_dataset(
            method="cross_ensemble",
            condition=condition,
        )

        result = dataset.get_cond_indexes(
            {
                INIT_TIME_DIM: np.asarray(
                    ["2000-01-01"],
                    dtype="datetime64[ns]",
                ),
                LEAD_TIME_DIM: np.asarray([1]),
                REALIZATION_DIM: np.asarray([1]),
            }
        )

        assert REALIZATION_DIM not in result

    @pytest.mark.pruned
    def test_same_member_requires_realization(self):
        condition = make_data_config(
            name="condition",
            ensemble_mean=False,
            realizations=[0, 1],
        )
        dataset = self.make_dataset(
            method="same_member",
            condition=condition,
        )

        with pytest.raises(
            ValueError,
            match="requires.*coordinates",
        ):
            dataset.get_cond_indexes(
                {
                    INIT_TIME_DIM: np.asarray(
                        ["2000-01-01"],
                        dtype="datetime64[ns]",
                    ),
                    LEAD_TIME_DIM: np.asarray([1]),
                }
            )

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "member,expected_index",
        [
            (0, 0),
            (1, 1),
        ],
    )
    def test_same_member_index(
        self,
        member,
        expected_index,
    ):
        condition = make_data_config(
            name="condition",
            ensemble_mean=False,
            realizations=[0, 1],
        )
        dataset = self.make_dataset(
            method="same_member",
            condition=condition,
        )

        result = dataset.get_cond_indexes(
            {
                REALIZATION_DIM: np.asarray([member]),
            }
        )

        np.testing.assert_array_equal(
            result[REALIZATION_DIM],
            [expected_index],
        )

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "dimension,bad_value",
        [
            (
                INIT_TIME_DIM,
                np.datetime64("1990-01-01"),
            ),
            (LEAD_TIME_DIM, 99),
            (REALIZATION_DIM, 9),
        ],
    )
    def test_missing_condition_coordinates(
        self,
        dimension,
        bad_value,
    ):
        condition = make_data_config(
            name="condition",
            ensemble_mean=False,
            realizations=[0, 1],
        )
        method = "same_member" if dimension == REALIZATION_DIM else "cross_ensemble"
        dataset = self.make_dataset(
            method=method,
            condition=condition,
        )
        sample_coords = {
            dimension: np.asarray([bad_value]),
        }

        if method == "same_member":
            sample_coords.setdefault(
                REALIZATION_DIM,
                np.asarray([bad_value]),
            )

        with pytest.raises(
            ValueError,
            match="conditioning coordinates were not found",
        ):
            dataset.get_cond_indexes(sample_coords)


class TestIndexModelDatasetMassive:
    @pytest.mark.pruned
    def test_disabled_model_returns_none(self):
        dataset = ConcreteDataset(
            config=ConcreteDatasetConfig(
                model=make_data_config(),
            ),
            load_model=False,
        )

        assert dataset._index_model_dataset(0) is None

    @pytest.mark.pruned
    def test_empty_index_mapping_calls_isel_without_arguments(self):
        model = make_data_config()
        model.isel = Mock(side_effect=model.data.isel)
        config = ConcreteDatasetConfig(model=model)
        dataset = ConcreteDataset(
            config=config,
            load_model=True,
        )
        dataset.model_indexes = {}

        with patch.object(
            module,
            "_unwrap_data_variables",
            return_value=xr.DataArray([1.0]),
        ):
            dataset._index_model_dataset(0)

        model.isel.assert_called_once_with()

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "sample_index,expected_position",
        [
            (0, 0),
            (1, 1),
        ],
    )
    def test_selection_uses_requested_sample(
        self,
        sample_index,
        expected_position,
    ):
        model_data = xr.Dataset(
            {
                "tas": (
                    INIT_TIME_DIM,
                    [10.0, 20.0],
                )
            },
            coords={
                INIT_TIME_DIM: np.asarray(
                    [
                        "2000-01-01",
                        "2001-01-01",
                    ],
                    dtype="datetime64[ns]",
                )
            },
        )
        model = make_data_config()
        model.isel = Mock(side_effect=model_data.isel)
        config = ConcreteDatasetConfig(model=model)
        dataset = ConcreteDataset(
            config=config,
            load_model=True,
        )
        dataset.model_indexes = {
            INIT_TIME_DIM: np.asarray(
                [0, 1],
                dtype=np.int64,
            ),
        }

        with patch.object(
            module,
            "_unwrap_data_variables",
            side_effect=lambda value: value["tas"],
        ):
            result = dataset._index_model_dataset(sample_index)

        model.isel.assert_called_once_with(
            **{
                INIT_TIME_DIM: [expected_position],
            }
        )
        assert result.item() == pytest.approx(10.0 if expected_position == 0 else 20.0)

    @pytest.mark.pruned
    def test_all_selection_dimensions(self):
        model = make_data_config(
            realizations=[0, 1],
        )
        model.isel = Mock(side_effect=model.data.isel)
        config = ConcreteDatasetConfig(model=model)
        dataset = ConcreteDataset(
            config=config,
            load_model=True,
        )
        dataset.model_indexes = {
            INIT_TIME_DIM: np.asarray([1]),
            LEAD_TIME_DIM: np.asarray([2]),
            REALIZATION_DIM: np.asarray([0]),
        }

        with patch.object(
            module,
            "_unwrap_data_variables",
            return_value=xr.DataArray([1.0]),
        ):
            dataset._index_model_dataset(0)

        model.isel.assert_called_once_with(
            **{
                INIT_TIME_DIM: [1],
                LEAD_TIME_DIM: [2],
                REALIZATION_DIM: [0],
            }
        )


class TestDatasetComputeAndLengthMassive:
    @pytest.mark.pruned
    def test_compute_no_arrays(self):
        assert ConcreteDataset._compute() == ()

    @pytest.mark.pruned
    def test_compute_one_numpy_array(self):
        array = np.asarray([1, 2, 3])

        result = ConcreteDataset._compute(array)

        assert isinstance(result, tuple)
        np.testing.assert_array_equal(
            result[0],
            array,
        )

    @pytest.mark.pruned
    def test_compute_multiple_xarray_objects(self):
        first = xr.DataArray([1.0])
        second = xr.Dataset(
            {
                "value": (
                    "sample",
                    [2.0],
                )
            }
        )

        result = ConcreteDataset._compute(
            first,
            second,
        )

        xr.testing.assert_equal(
            result[0],
            first,
        )
        xr.testing.assert_equal(
            result[1],
            second,
        )

    @pytest.mark.pruned
    def test_compute_preserves_argument_order(self):
        arrays = (
            np.asarray([1]),
            np.asarray([2]),
            np.asarray([3]),
        )

        result = ConcreteDataset._compute(*arrays)

        for actual, expected in zip(
            result,
            arrays,
            strict=True,
        ):
            np.testing.assert_array_equal(
                actual,
                expected,
            )

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "length",
        [
            0,
            1,
            2,
            10,
        ],
    )
    def test_len_uses_first_coordinate_length(
        self,
        length,
    ):
        dataset = ConcreteDataset(
            config=ConcreteDatasetConfig(
                model=make_data_config(),
            )
        )
        dataset.sample_coords = {
            "first": np.arange(length),
            "second": np.arange(length),
        }

        assert len(dataset) == length

    @pytest.mark.pruned
    def test_len_ignores_later_coordinate_lengths(self):
        dataset = ConcreteDataset(
            config=ConcreteDatasetConfig(
                model=make_data_config(),
            )
        )
        dataset.sample_coords = {
            "first": np.arange(3),
            "second": np.arange(100),
        }

        assert len(dataset) == 3

    @pytest.mark.pruned
    def test_len_empty_mapping_raises_stop_iteration(self):
        dataset = ConcreteDataset(
            config=ConcreteDatasetConfig(
                model=make_data_config(),
            )
        )
        dataset.sample_coords = {}

        with pytest.raises(StopIteration):
            len(dataset)


class TestDatasetABCInitializationMassive:
    def make_config(
        self,
        *,
        model=None,
        condition=None,
        condition_method=None,
    ):
        if model is None:
            model = make_data_config(
                lead_times=[1],
            )

        config = ConcreteDatasetConfig(
            model=model,
            condition=condition,
            condition_method=condition_method,
            lead_times=[1],
            available_times=pd.DatetimeIndex(["2000-01-01"]),
        )
        config._effective_condition = condition
        config._fitted_preprocessors = True

        return config

    def make_features(self):
        return AddedTimeFeatures(
            TimeFeatureReference(
                lead_times=[1],
            ),
            [LEAD_TIME_DIM],
        )

    def sample_coordinates(self):
        return {
            INIT_TIME_DIM: np.asarray(
                ["2000-01-01"],
                dtype="datetime64[ns]",
            ),
            LEAD_TIME_DIM: np.asarray([1]),
        }

    @pytest.mark.pruned
    def test_model_loading_enabled(self):
        model = make_data_config()
        config = self.make_config(model=model)
        dataset = ConcreteDataset(
            config=config,
            time_features=self.make_features(),
            load_model=True,
        )

        with (
            patch.object(dataset, "_check_init"),
            patch.object(dataset, "_resolve_mask"),
            patch.object(
                dataset,
                "_prepare_sampling_mask",
            ),
            patch.object(
                dataset,
                "get_sampling_coords",
                return_value=self.sample_coordinates(),
            ),
            patch.object(
                dataset,
                "get_model_indexes",
                return_value={},
            ),
            patch.object(
                dataset,
                "get_cond_indexes",
                return_value=None,
            ),
            patch.object(
                AddedTimeFeatures,
                "build_time_features",
                return_value=dataset.time_features,
            ),
        ):
            DatasetABC.__init__(dataset)

        model.open_xarray_data.assert_called_once_with(
            load=False,
            add_time_auxiliary_coords=True,
        )

    @pytest.mark.pruned
    def test_model_loading_disabled(self):
        model = make_data_config()
        config = self.make_config(model=model)
        dataset = ConcreteDataset(
            config=config,
            time_features=self.make_features(),
            load_model=False,
        )

        with (
            patch.object(dataset, "_check_init"),
            patch.object(dataset, "_resolve_mask"),
            patch.object(
                dataset,
                "_prepare_sampling_mask",
            ),
            patch.object(
                dataset,
                "get_sampling_coords",
                return_value=self.sample_coordinates(),
            ),
            patch.object(
                dataset,
                "get_model_indexes",
                return_value=None,
            ),
            patch.object(
                dataset,
                "get_cond_indexes",
                return_value=None,
            ),
            patch.object(
                AddedTimeFeatures,
                "build_time_features",
                return_value=dataset.time_features,
            ),
        ):
            DatasetABC.__init__(dataset)

        model.open_xarray_data.assert_not_called()

    @pytest.mark.pruned
    def test_condition_loading_enabled(self):
        condition = make_data_config(
            name="condition",
            ensemble_mean=True,
        )
        config = self.make_config(
            condition=condition,
            condition_method="ensemble_mean",
        )
        dataset = ConcreteDataset(
            config=config,
            time_features=self.make_features(),
            load_model=False,
        )

        with (
            patch.object(dataset, "_check_init"),
            patch.object(dataset, "_resolve_mask"),
            patch.object(
                dataset,
                "_prepare_sampling_mask",
            ),
            patch.object(
                dataset,
                "get_sampling_coords",
                return_value=self.sample_coordinates(),
            ),
            patch.object(
                dataset,
                "get_model_indexes",
                return_value=None,
            ),
            patch.object(
                dataset,
                "get_cond_indexes",
                return_value={},
            ),
            patch.object(
                AddedTimeFeatures,
                "build_time_features",
                return_value=dataset.time_features,
            ),
        ):
            DatasetABC.__init__(dataset)

        condition.open_xarray_data.assert_called_once_with(
            load=False,
            add_time_auxiliary_coords=True,
        )

    @pytest.mark.pruned
    def test_no_condition_skips_condition_loading(self):
        config = self.make_config(
            condition=None,
            condition_method=None,
        )
        dataset = ConcreteDataset(
            config=config,
            time_features=self.make_features(),
            load_model=False,
        )

        with (
            patch.object(dataset, "_check_init"),
            patch.object(dataset, "_resolve_mask"),
            patch.object(
                dataset,
                "_prepare_sampling_mask",
            ),
            patch.object(
                dataset,
                "get_sampling_coords",
                return_value=self.sample_coordinates(),
            ),
            patch.object(
                dataset,
                "get_model_indexes",
                return_value=None,
            ),
            patch.object(
                dataset,
                "get_cond_indexes",
                return_value=None,
            ),
            patch.object(
                AddedTimeFeatures,
                "build_time_features",
                return_value=dataset.time_features,
            ),
        ):
            DatasetABC.__init__(dataset)

        assert dataset.cond_indexes is None

    def test_time_features_are_copied(self):
        config = self.make_config()
        original = self.make_features()
        dataset = ConcreteDataset(
            config=config,
            time_features=original,
            load_model=False,
        )

        with (
            patch.object(dataset, "_check_init"),
            patch.object(dataset, "_resolve_mask"),
            patch.object(
                dataset,
                "_prepare_sampling_mask",
            ),
            patch.object(
                dataset,
                "get_sampling_coords",
                return_value=self.sample_coordinates(),
            ),
            patch.object(
                dataset,
                "get_model_indexes",
                return_value=None,
            ),
            patch.object(
                dataset,
                "get_cond_indexes",
                return_value=None,
            ),
        ):
            DatasetABC.__init__(dataset)

        assert dataset.time_features is not original
        assert dataset.time_features == original

    @pytest.mark.pruned
    def test_sampling_coordinates_are_stored(self):
        config = self.make_config()
        expected = self.sample_coordinates()
        dataset = ConcreteDataset(
            config=config,
            time_features=self.make_features(),
            load_model=False,
        )

        with (
            patch.object(dataset, "_check_init"),
            patch.object(dataset, "_resolve_mask"),
            patch.object(
                dataset,
                "_prepare_sampling_mask",
            ),
            patch.object(
                dataset,
                "get_sampling_coords",
                return_value=expected,
            ),
            patch.object(
                dataset,
                "get_model_indexes",
                return_value=None,
            ),
            patch.object(
                dataset,
                "get_cond_indexes",
                return_value=None,
            ),
        ):
            DatasetABC.__init__(dataset)

        assert dataset.sample_coords is expected

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "failing_method",
        [
            "_check_init",
            "_resolve_mask",
            "_prepare_sampling_mask",
            "get_sampling_coords",
            "get_model_indexes",
            "get_cond_indexes",
        ],
    )
    def test_initialization_propagates_stage_errors(
        self,
        failing_method,
    ):
        config = self.make_config()
        dataset = ConcreteDataset(
            config=config,
            time_features=self.make_features(),
            load_model=False,
        )

        patches = {
            "_check_init": Mock(),
            "_resolve_mask": Mock(),
            "_prepare_sampling_mask": Mock(),
            "get_sampling_coords": Mock(return_value=self.sample_coordinates()),
            "get_model_indexes": Mock(return_value=None),
            "get_cond_indexes": Mock(return_value=None),
        }
        patches[failing_method].side_effect = RuntimeError(failing_method)

        with (
            patch.object(
                dataset,
                "_check_init",
                patches["_check_init"],
            ),
            patch.object(
                dataset,
                "_resolve_mask",
                patches["_resolve_mask"],
            ),
            patch.object(
                dataset,
                "_prepare_sampling_mask",
                patches["_prepare_sampling_mask"],
            ),
            patch.object(
                dataset,
                "get_sampling_coords",
                patches["get_sampling_coords"],
            ),
            patch.object(
                dataset,
                "get_model_indexes",
                patches["get_model_indexes"],
            ),
            patch.object(
                dataset,
                "get_cond_indexes",
                patches["get_cond_indexes"],
            ),
            pytest.raises(
                RuntimeError,
                match=failing_method,
            ),
        ):
            DatasetABC.__init__(dataset)


class TestDatasetABCMiscellaneousMassive:
    @pytest.mark.pruned
    def test_effective_condition_returns_exact_object(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
        )
        value = object()
        config._effective_condition = value

        assert config.effective_condition is value

    @pytest.mark.pruned
    def test_build_dataset_concrete_result(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
        )

        assert config.build_dataset() == "dataset"

    @pytest.mark.pruned
    def test_dataset_operator_identity(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
        )

        assert config.ds_operator is config._operator

    @pytest.mark.pruned
    def test_get_added_features_dim_calls_len(self):
        feature_object = Mock()
        feature_object.__len__ = Mock(return_value=4)
        dataset = ConcreteDataset(
            config=ConcreteDatasetConfig(
                model=make_data_config(),
            ),
            time_features=feature_object,
        )

        assert dataset.get_added_features_dim() == 4
        feature_object.__len__.assert_called_once_with()

    @pytest.mark.pruned
    def test_concrete_dataset_getitem(self):
        dataset = ConcreteDataset(
            config=ConcreteDatasetConfig(
                model=make_data_config(),
            )
        )

        assert dataset[10] == 10


class TestAdditionalLeadTimeConfigBranches:
    @pytest.mark.pruned
    def test_range_with_equal_start_and_end(self):
        config = lead_time_config(
            start=4,
            end=4,
        )

        np.testing.assert_array_equal(
            config.build_lead_times(),
            np.asarray([4]),
        )

    @pytest.mark.pruned
    def test_range_with_start_after_end_is_empty(self):
        config = lead_time_config(
            start=5,
            end=3,
        )

        assert config.build_lead_times().size == 0

    def test_explicit_empty_list_is_preserved(self):
        config = lead_time_config(
            list_lead_times=[],
        )

        assert config.build_lead_times() == []


class TestAdditionalConditionMethodBranches:
    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "method",
        [
            "Ensemble_Mean",
            "CROSS_ENSEMBLE",
            "Same_Member",
            "STATIC",
        ],
    )
    def test_effective_condition_method_is_lowercase(
        self,
        method,
    ):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            condition_method=method,
        )

        assert config.effective_condition_method == method.lower()

    @pytest.mark.pruned
    def test_effective_condition_method_preserves_none(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            condition_method=None,
        )

        assert config.effective_condition_method is None

    def test_invalid_method_error_lists_valid_methods(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            condition_method="unsupported",
        )

        with pytest.raises(
            ValueError,
            match="cross_ensemble",
        ):
            config._check_condition_method()


class TestAdditionalConditionResolutionBranches:
    @pytest.mark.pruned
    def test_explicit_condition_becomes_effective_condition(self):
        condition = make_data_config(
            name="condition",
        )
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            condition=condition,
        )
        config._effective_condition = None

        result = config._resolve_condition()

        assert result is config
        assert config.effective_condition is condition

    @pytest.mark.pruned
    def test_no_conditioning_resolves_to_none(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            condition=None,
            condition_method=None,
        )
        config._effective_condition = object()

        result = config._resolve_condition()

        assert result is config
        assert config.effective_condition is None

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "method",
        [
            "ensemble_mean",
            "cross_ensemble",
            "same_member",
        ],
    )
    def test_model_is_used_as_condition_for_ensemble_methods(
        self,
        method,
    ):
        model = make_data_config()

        config = ConcreteDatasetConfig(
            model=model,
            condition=None,
            condition_method=method,
        )

        with patch.object(
            config,
            "_model_as_condition",
            return_value="MODEL_CONDITION",
        ) as model_as_condition:
            config._resolve_condition()

        model_as_condition.assert_called_once_with()
        assert config.effective_condition == "MODEL_CONDITION"

    @pytest.mark.pruned
    def test_static_method_without_condition_does_not_use_model(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            condition=None,
            condition_method="static",
        )

        assert config._using_model_data_as_condition is False


class TestAdditionalUsingModelAsConditionBranches:
    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "method",
        [
            "ensemble_mean",
            "cross_ensemble",
            "same_member",
        ],
    )
    def test_missing_condition_uses_model_for_supported_methods(
        self,
        method,
    ):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            condition=None,
            condition_method=method,
        )

        assert config._using_model_data_as_condition is True

    @pytest.mark.pruned
    def test_matching_condition_configuration_uses_model_data(self):
        model = make_data_config(
            paths=["shared.nc"],
            names=["tas"],
            realization_list=[1, 2],
        )
        condition = make_data_config(
            name="condition",
            paths=["shared.nc"],
            names=["tas"],
            realization_list=[1, 2],
        )

        config = ConcreteDatasetConfig(
            model=model,
            condition=condition,
            condition_method="same_member",
        )

        assert config._using_model_data_as_condition is True

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "different_field",
        [
            "paths",
            "names",
            "realization_list",
        ],
    )
    def test_different_condition_configuration_does_not_use_model_data(
        self,
        different_field,
    ):
        model = make_data_config(
            paths=["model.nc"],
            names=["tas"],
            realization_list=[1, 2],
        )
        condition = make_data_config(
            name="condition",
            paths=["model.nc"],
            names=["tas"],
            realization_list=[1, 2],
        )

        if different_field == "paths":
            condition.paths = ["condition.nc"]
        elif different_field == "names":
            condition.names = ["pr"]
        else:
            condition.realization_list = [3, 4]

        config = ConcreteDatasetConfig(
            model=model,
            condition=condition,
            condition_method="same_member",
        )

        assert config._using_model_data_as_condition is False

    @pytest.mark.pruned
    def test_condition_without_model_never_uses_model_data(self):
        config = ConcreteDatasetConfig(
            model=None,
            condition=make_data_config(
                name="condition",
            ),
            condition_method="ensemble_mean",
        )

        assert config._using_model_data_as_condition is False


class TestAdditionalModelAsConditionBranches:
    @pytest.mark.pruned
    @pytest.mark.parametrize(
        (
            "method",
            "expected_ensemble_mean",
        ),
        [
            (
                "ensemble_mean",
                True,
            ),
            (
                "cross_ensemble",
                False,
            ),
            (
                "same_member",
                False,
            ),
        ],
    )
    def test_model_as_condition_sets_ensemble_mean(
        self,
        method,
        expected_ensemble_mean,
    ):
        model = make_data_config(
            paths=["model.nc"],
            names=["tas", "pr"],
            realization_list=[1, 2],
        )
        model.preprocessing_pipeline = object()
        model.concat_dim = "year"
        model.file_type = "*.nc"
        model.rename_dict = {
            "old": "new",
        }

        config = ConcreteDatasetConfig(
            model=model,
            condition_method=method,
        )

        built = object()

        with patch.object(
            module,
            "ModelDataConfig",
            return_value=built,
        ) as constructor:
            result = config._model_as_condition()

        assert result is built
        constructor.assert_called_once_with(
            paths=model.paths,
            names=model.names,
            preprocessing_pipeline=model.preprocessing_pipeline,
            realization_list=model.realization_list,
            concat_dim=model.concat_dim,
            file_type=model.file_type,
            ensemble_mean=expected_ensemble_mean,
            rename_dict=model.rename_dict,
        )


class TestAdditionalLeadTimeResolutionBranches:
    def test_resolve_lead_time_config(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            lead_times=lead_time_config(
                start=2,
                end=4,
            ),
        )

        result = config._resolve_lead_times()

        assert result is None
        np.testing.assert_array_equal(
            config.lead_times,
            np.asarray([2, 3, 4]),
        )

    @pytest.mark.pruned
    def test_resolve_explicit_array_does_not_modify_value(self):
        lead_times = np.asarray(
            [
                1,
                3,
            ]
        )
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            lead_times=lead_times,
        )

        config._resolve_lead_times()

        assert config.lead_times is lead_times

    @pytest.mark.pruned
    def test_input_lead_times_uses_effective_input(self):
        effective_input = make_data_config(
            lead_times=[
                2,
                4,
                8,
            ],
        )
        config = ConcreteDatasetConfig(
            model=effective_input,
            effective_input=effective_input,
        )

        np.testing.assert_array_equal(
            config.input_lead_times,
            np.asarray(
                [
                    2,
                    4,
                    8,
                ]
            ),
        )


class TestAdditionalModelChecks:
    @pytest.mark.pruned
    def test_same_member_rejects_ensemble_mean_model(self):
        model = make_data_config(
            ensemble_mean=True,
        )
        config = ConcreteDatasetConfig(
            model=model,
            condition_method="same_member",
        )

        with pytest.raises(
            ValueError,
            match="model data should not be ensemble mean",
        ):
            config._check_model()

    def test_same_member_accepts_non_ensemble_mean_model(self):
        model = make_data_config(
            ensemble_mean=False,
        )
        config = ConcreteDatasetConfig(
            model=model,
            condition_method="same_member",
        )

        assert config._check_model() is config

    @pytest.mark.pruned
    def test_model_check_accepts_missing_model(self):
        config = ConcreteDatasetConfig(
            model=None,
            condition=make_data_config(
                name="condition",
            ),
        )

        assert config._check_model() is config


class TestAdditionalConditionChecks:
    def test_condition_requires_method(self):
        condition = make_data_config(
            name="condition",
        )
        config = ConcreteDatasetConfig(
            condition=condition,
            condition_method=None,
        )
        config._effective_condition = condition

        with pytest.raises(
            ValueError,
            match="specify condition_method",
        ):
            config._check_condition()

    @pytest.mark.parametrize(
        "method",
        [
            "cross_ensemble",
            "same_member",
        ],
    )
    def test_member_conditioning_rejects_ensemble_mean_condition(
        self,
        method,
    ):
        condition = make_data_config(
            name="condition",
            realizations=[
                0,
                1,
            ],
            ensemble_mean=True,
        )
        config = ConcreteDatasetConfig(
            condition=condition,
            condition_method=method,
        )
        config._effective_condition = condition

        with pytest.raises(
            ValueError,
            match="ensemble_mean cannot be True",
        ):
            config._check_condition()

    @pytest.mark.parametrize(
        "method",
        [
            "cross_ensemble",
            "same_member",
        ],
    )
    def test_member_conditioning_requires_realization_coordinate(
        self,
        method,
    ):
        condition = make_data_config(
            name="condition",
            ensemble_mean=False,
        )
        config = ConcreteDatasetConfig(
            condition=condition,
            condition_method=method,
        )
        config._effective_condition = condition

        with pytest.raises(
            ValueError,
            match=REALIZATION_DIM,
        ):
            config._check_condition()

    def test_ensemble_mean_conditioning_requires_ensemble_mean(self):
        condition = make_data_config(
            name="condition",
            ensemble_mean=False,
        )
        config = ConcreteDatasetConfig(
            condition=condition,
            condition_method="ensemble_mean",
        )
        config._effective_condition = condition

        with pytest.raises(
            ValueError,
            match="Ensemble mean must be True",
        ):
            config._check_condition()

    def test_static_condition_rejects_realization_list(self):
        condition = make_data_config(
            name="condition",
            realization_list=[
                1,
            ],
        )
        condition.coords = {}

        config = ConcreteDatasetConfig(
            condition=condition,
            condition_method="static",
        )
        config._effective_condition = condition

        with pytest.raises(
            ValueError,
            match="cannot specify realization list",
        ):
            config._check_condition()

    def test_static_condition_rejects_sampling_dimensions(self):
        condition = make_data_config(
            name="condition",
            realization_list=None,
        )
        config = ConcreteDatasetConfig(
            condition=condition,
            condition_method="static",
        )
        config._effective_condition = condition

        with pytest.raises(
            ValueError,
            match="cannot have",
        ):
            config._check_condition()

    def test_static_method_requires_condition_dataset(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            condition=None,
            condition_method="static",
        )
        config._effective_condition = None

        with pytest.raises(
            ValueError,
            match="condition dataset must be specified",
        ):
            config._check_condition()

    @pytest.mark.pruned
    def test_valid_static_condition_is_accepted(self):
        condition = make_data_config(
            name="condition",
            realization_list=None,
        )
        condition.coords = {}

        config = ConcreteDatasetConfig(
            condition=condition,
            condition_method="static",
        )
        config._effective_condition = condition

        assert config._check_condition() is config


class TestAdditionalModelConditionCompatibility:
    @pytest.mark.pruned
    def test_nonstatic_condition_requires_model_time_dimension(self):
        model = make_data_config()
        condition = make_data_config(
            name="condition",
        )
        condition.coords.pop(
            INIT_TIME_DIM,
        )

        config = ConcreteDatasetConfig(
            model=model,
            condition=condition,
            condition_method="ensemble_mean",
        )
        config._effective_condition = condition

        with pytest.raises(
            ValueError,
            match=INIT_TIME_DIM,
        ):
            config._check_model_vs_condition()

    def test_nonstatic_condition_requires_all_model_lead_times(self):
        model = make_data_config(
            lead_times=[
                1,
                2,
                3,
            ],
        )
        condition = make_data_config(
            name="condition",
            lead_times=[
                1,
                2,
            ],
            ensemble_mean=True,
        )

        config = ConcreteDatasetConfig(
            model=model,
            condition=condition,
            condition_method="ensemble_mean",
        )
        config._effective_condition = condition

        with pytest.raises(
            ValueError,
            match="same lead_time coordinates",
        ):
            config._check_model_vs_condition()

    def test_condition_and_model_require_matching_time_types(self):
        model = make_data_config(
            time_coords_type="datetime",
        )
        condition = make_data_config(
            name="condition",
            time_coords_type="cftime",
            ensemble_mean=True,
        )

        config = ConcreteDatasetConfig(
            model=model,
            condition=condition,
            condition_method="ensemble_mean",
        )
        config._effective_condition = condition

        with pytest.raises(
            ValueError,
            match="same cftime/datetime type",
        ):
            config._check_model_vs_condition()

    def test_same_member_requires_realization_coordinates(self):
        model = make_data_config(
            ensemble_mean=False,
        )
        condition = make_data_config(
            name="condition",
            ensemble_mean=False,
        )

        config = ConcreteDatasetConfig(
            model=model,
            condition=condition,
            condition_method="same_member",
        )
        config._effective_condition = condition

        with pytest.raises(
            ValueError,
            match="same_member conditioning",
        ):
            config._check_model_vs_condition()

    def test_same_member_requires_matching_realizations(self):
        model = make_data_config(
            realizations=[
                0,
                1,
            ],
            ensemble_mean=False,
        )
        condition = make_data_config(
            name="condition",
            realizations=[
                0,
                2,
            ],
            ensemble_mean=False,
        )

        config = ConcreteDatasetConfig(
            model=model,
            condition=condition,
            condition_method="same_member",
        )
        config._effective_condition = condition

        with pytest.raises(
            ValueError,
            match="same ensemble members",
        ):
            config._check_model_vs_condition()

    def test_observation_branch_requires_condition_spatial_dimension(self):
        model = make_data_config(
            spatial_coords={
                "lat": [
                    45.0,
                    46.0,
                ],
            },
        )
        condition = make_data_config(
            name="condition",
            ensemble_mean=True,
        )

        config = ConcreteDatasetConfig(
            model=model,
            condition=condition,
            condition_method="ensemble_mean",
            observation=object(),
        )
        config._effective_condition = condition

        with pytest.raises(
            ValueError,
            match="same NN dims",
        ):
            config._check_model_vs_condition()

    def test_observation_branch_requires_matching_spatial_coordinates(self):
        model = make_data_config(
            spatial_coords={
                "lat": [
                    45.0,
                    46.0,
                ],
            },
        )
        condition = make_data_config(
            name="condition",
            spatial_coords={
                "lat": [
                    40.0,
                    41.0,
                ],
            },
            ensemble_mean=True,
        )

        config = ConcreteDatasetConfig(
            model=model,
            condition=condition,
            condition_method="ensemble_mean",
            observation=object(),
        )
        config._effective_condition = condition

        with pytest.raises(
            ValueError,
            match="same lat",
        ):
            config._check_model_vs_condition()


class TestAdditionalGetInputTimes:
    @pytest.mark.pruned
    def test_sequence_is_converted_to_dataarray(self):
        model = make_data_config()
        config = ConcreteDatasetConfig(
            model=model,
            available_times=[
                "2000-01-01",
                "2001-01-01",
            ],
        )

        result = config.get_input_times(
            [
                np.datetime64("2000-01-01"),
            ]
        )

        assert isinstance(
            result,
            xr.DataArray,
        )
        assert result.dims == (INIT_TIME_DIM,)

    @pytest.mark.pruned
    def test_dataarray_is_preserved(self):
        model = make_data_config()
        config = ConcreteDatasetConfig(
            model=model,
        )
        requested = xr.DataArray(
            np.asarray(
                [
                    "2000-01-01",
                ],
                dtype="datetime64[ns]",
            ),
            dims=(INIT_TIME_DIM,),
            coords={
                INIT_TIME_DIM: np.asarray(
                    [
                        "2000-01-01",
                    ],
                    dtype="datetime64[ns]",
                )
            },
        )

        result = config.get_input_times(
            requested,
        )

        np.testing.assert_array_equal(
            result.values,
            requested.values,
        )

    def test_unavailable_requested_time_raises(self):
        model = make_data_config()
        config = ConcreteDatasetConfig(
            model=model,
        )

        with pytest.raises(
            ValueError,
            match="requested_times are unavailable",
        ):
            config.get_input_times(
                [
                    np.datetime64("1999-01-01"),
                ]
            )

    @pytest.mark.pruned
    def test_times_missing_from_effective_input_are_removed(self):
        model = make_data_config(
            times=np.asarray(
                [
                    "2000-01-01",
                ],
                dtype="datetime64[ns]",
            ),
        )
        config = ConcreteDatasetConfig(
            model=model,
            available_times=[
                "2000-01-01",
                "2001-01-01",
            ],
        )

        result = config.get_input_times(
            np.asarray(
                [
                    "2000-01-01",
                    "2001-01-01",
                ],
                dtype="datetime64[ns]",
            )
        )

        np.testing.assert_array_equal(
            result.values,
            np.asarray(
                [
                    "2000-01-01",
                ],
                dtype="datetime64[ns]",
            ),
        )


class TestAdditionalAddedTimeFeaturesInitialization:
    @pytest.mark.pruned
    def test_no_requested_features_becomes_empty_tuple(self):
        features = AddedTimeFeatures(
            TimeFeatureReference(),
            time_features=None,
        )

        assert features.time_features == ()
        assert features.time_features_array is None
        assert len(features) == 0

    @pytest.mark.pruned
    def test_requested_features_follow_canonical_order(self):
        features = AddedTimeFeatures(
            TimeFeatureReference(),
            time_features=[
                "day_cos",
                LEAD_TIME_DIM,
                "month_sin",
                INIT_TIME_DIM,
            ],
        )

        assert features.time_features == (
            INIT_TIME_DIM,
            LEAD_TIME_DIM,
            "month_sin",
            "day_cos",
        )

    @pytest.mark.pruned
    def test_duplicate_requested_features_are_removed(self):
        features = AddedTimeFeatures(
            TimeFeatureReference(),
            time_features=[
                "month_sin",
                "month_sin",
            ],
        )

        assert features.time_features == ("month_sin",)

    @pytest.mark.pruned
    def test_unsupported_features_raise(self):
        with pytest.raises(
            ValueError,
            match="Unsupported time features",
        ):
            AddedTimeFeatures(
                TimeFeatureReference(),
                time_features=[
                    "hour_sin",
                ],
            )


class TestAdditionalDaysInYear:
    @pytest.mark.pruned
    @pytest.mark.parametrize(
        (
            "time",
            "expected",
        ),
        [
            (
                cftime.Datetime360Day(
                    2000,
                    1,
                    1,
                ),
                360,
            ),
            (
                cftime.DatetimeNoLeap(
                    2000,
                    1,
                    1,
                ),
                365,
            ),
            (
                cftime.DatetimeAllLeap(
                    2001,
                    1,
                    1,
                ),
                366,
            ),
            (
                np.datetime64("2000-01-01"),
                366,
            ),
            (
                np.datetime64("2001-01-01"),
                365,
            ),
            (
                datetime.datetime(
                    2004,
                    1,
                    1,
                ),
                366,
            ),
        ],
    )
    def test_days_in_year(
        self,
        time,
        expected,
    ):
        assert (
            AddedTimeFeatures._days_in_year(
                time,
            )
            == expected
        )


class TestAdditionalBuildTimeFeatures:
    def test_missing_initialization_time_raises(self):
        features = AddedTimeFeatures(
            TimeFeatureReference(),
            time_features=[
                "month_sin",
            ],
        )

        with pytest.raises(
            ValueError,
            match="missing required",
        ):
            features.build_time_features(
                {
                    LEAD_TIME_DIM: np.asarray(
                        [
                            1,
                        ]
                    ),
                }
            )

    @pytest.mark.pruned
    def test_missing_lead_time_raises(self):
        features = AddedTimeFeatures(
            TimeFeatureReference(),
            time_features=[
                "month_sin",
            ],
        )

        with pytest.raises(
            ValueError,
            match="missing required",
        ):
            features.build_time_features(
                {
                    INIT_TIME_DIM: np.asarray(
                        [
                            "2000-01-01",
                        ],
                        dtype="datetime64[ns]",
                    ),
                }
            )

    @pytest.mark.pruned
    def test_empty_feature_selection_skips_lead_time_calculation(self):
        features = AddedTimeFeatures(
            TimeFeatureReference(),
            time_features=None,
        )

        with patch.object(
            module,
            "add_lead_times",
        ) as add_times:
            result = features.build_time_features(
                {
                    INIT_TIME_DIM: np.asarray(
                        [
                            "2000-01-01",
                        ],
                        dtype="datetime64[ns]",
                    ),
                    LEAD_TIME_DIM: np.asarray(
                        [
                            1,
                        ]
                    ),
                }
            )

        assert result is features
        add_times.assert_not_called()
        assert features.time_features_array is None

    @pytest.mark.pruned
    def test_lead_time_feature_is_normalized(self):
        reference = TimeFeatureReference(
            lead_times=[
                1,
                2,
                4,
            ],
        )
        features = AddedTimeFeatures(
            reference,
            time_features=[
                LEAD_TIME_DIM,
            ],
        )

        features.build_time_features(
            {
                INIT_TIME_DIM: np.asarray(
                    [
                        "2000-01-01",
                        "2000-01-01",
                        "2000-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                LEAD_TIME_DIM: np.asarray(
                    [
                        1,
                        2,
                        4,
                    ]
                ),
            }
        )

        np.testing.assert_allclose(
            features.time_features_array[:, 0],
            np.asarray(
                [
                    0.25,
                    0.5,
                    1.0,
                ],
                dtype=np.float32,
            ),
        )
        assert features.time_features_array.dtype == np.float32

    @pytest.mark.pruned
    def test_month_features_have_expected_values(self):
        features = AddedTimeFeatures(
            TimeFeatureReference(),
            time_features=[
                "month_sin",
                "month_cos",
            ],
        )

        with patch.object(
            module,
            "add_lead_times",
            return_value=np.asarray(
                [
                    "2000-01-01",
                    "2000-04-01",
                    "2000-07-01",
                ],
                dtype="datetime64[ns]",
            ),
        ):
            features.build_time_features(
                {
                    INIT_TIME_DIM: np.asarray(
                        [
                            "2000-01-01",
                        ]
                        * 3,
                        dtype="datetime64[ns]",
                    ),
                    LEAD_TIME_DIM: np.asarray(
                        [
                            1,
                            2,
                            3,
                        ]
                    ),
                }
            )

        np.testing.assert_allclose(
            features.time_features_array[:, 0],
            np.asarray(
                [
                    0.0,
                    1.0,
                    0.0,
                ],
                dtype=np.float32,
            ),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            features.time_features_array[:, 1],
            np.asarray(
                [
                    1.0,
                    0.0,
                    -1.0,
                ],
                dtype=np.float32,
            ),
            atol=1e-6,
        )

    @pytest.mark.pruned
    def test_all_features_are_stacked_in_canonical_order(self):
        features = AddedTimeFeatures(
            TimeFeatureReference(),
            time_features=[
                "day_cos",
                "month_cos",
                LEAD_TIME_DIM,
                INIT_TIME_DIM,
                "day_sin",
                "month_sin",
            ],
        )

        with patch.object(
            module,
            "add_lead_times",
            return_value=np.asarray(
                [
                    "2000-01-01",
                    "2000-02-01",
                ],
                dtype="datetime64[ns]",
            ),
        ):
            features.build_time_features(
                {
                    INIT_TIME_DIM: np.asarray(
                        [
                            "2000-01-01",
                            "2000-01-01",
                        ],
                        dtype="datetime64[ns]",
                    ),
                    LEAD_TIME_DIM: np.asarray(
                        [
                            1,
                            2,
                        ]
                    ),
                }
            )

        assert features.time_features == (
            INIT_TIME_DIM,
            LEAD_TIME_DIM,
            "month_sin",
            "month_cos",
            "day_sin",
            "day_cos",
        )
        assert features.time_features_array.shape == (
            2,
            6,
        )


class TestAdditionalTimeFeatureIndexing:
    def test_no_features_returns_none(self):
        features = AddedTimeFeatures(
            TimeFeatureReference(),
            time_features=None,
        )

        result = features(
            0,
            xr.DataArray(
                np.zeros(
                    (
                        2,
                        3,
                    )
                )
            ),
        )

        assert result is None

    @pytest.mark.pruned
    def test_call_before_building_raises(self):
        features = AddedTimeFeatures(
            TimeFeatureReference(),
            time_features=[
                "month_sin",
            ],
        )

        with pytest.raises(
            RuntimeError,
            match="must be built",
        ):
            features(
                0,
                xr.DataArray(
                    np.zeros(
                        (
                            2,
                            3,
                        )
                    )
                ),
            )

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "index",
        [
            -1,
            2,
            10,
        ],
    )
    def test_out_of_bounds_index_raises(
        self,
        index,
    ):
        features = AddedTimeFeatures(
            TimeFeatureReference(),
            time_features=[
                "month_sin",
            ],
        )
        features.time_features_array = np.zeros(
            (
                2,
                1,
            ),
            dtype=np.float32,
        )

        with pytest.raises(
            IndexError,
            match="out of bounds",
        ):
            features(
                index,
                xr.DataArray(
                    np.zeros(
                        (
                            1,
                            2,
                        )
                    )
                ),
            )

    @pytest.mark.pruned
    def test_two_dimensional_input_returns_feature_vector(self):
        features = AddedTimeFeatures(
            TimeFeatureReference(),
            time_features=[
                "month_sin",
                "month_cos",
            ],
        )
        features.time_features_array = np.asarray(
            [
                [
                    0.25,
                    0.75,
                ]
            ],
            dtype=np.float32,
        )

        result = features(
            0,
            xr.DataArray(
                np.zeros(
                    (
                        2,
                        4,
                    )
                )
            ),
        )

        np.testing.assert_array_equal(
            result,
            np.asarray(
                [
                    0.25,
                    0.75,
                ],
                dtype=np.float32,
            ),
        )

    @pytest.mark.pruned
    def test_spatial_input_broadcasts_features(self):
        features = AddedTimeFeatures(
            TimeFeatureReference(),
            time_features=[
                "month_sin",
                "month_cos",
            ],
        )
        features.time_features_array = np.asarray(
            [
                [
                    0.25,
                    0.75,
                ]
            ],
            dtype=np.float32,
        )

        input_data = xr.DataArray(
            np.zeros(
                (
                    3,
                    4,
                    5,
                )
            ),
            dims=(
                "channels",
                "lat",
                "lon",
            ),
        )

        result = features(
            0,
            input_data,
        )

        assert result.shape == (
            2,
            4,
            5,
        )
        np.testing.assert_allclose(
            result[0],
            0.25,
        )
        np.testing.assert_allclose(
            result[1],
            0.75,
        )
        assert result.flags.writeable is True


class TestAdditionalTimeFeatureEquality:
    @pytest.mark.pruned
    def test_equal_instances_compare_equal(self):
        first = AddedTimeFeatures(
            TimeFeatureReference(),
            time_features=[
                "month_sin",
            ],
        )
        second = AddedTimeFeatures(
            TimeFeatureReference(),
            time_features=[
                "month_sin",
            ],
        )

        assert first == second

    @pytest.mark.pruned
    def test_different_features_compare_not_equal(self):
        first = AddedTimeFeatures(
            TimeFeatureReference(),
            time_features=[
                "month_sin",
            ],
        )
        second = AddedTimeFeatures(
            TimeFeatureReference(),
            time_features=[
                "month_cos",
            ],
        )

        assert first != second

    @pytest.mark.pruned
    def test_non_time_feature_object_returns_not_implemented(self):
        features = AddedTimeFeatures(
            TimeFeatureReference(),
            time_features=[
                "month_sin",
            ],
        )

        assert features.__eq__(object()) is NotImplemented

    @pytest.mark.pruned
    def test_different_lead_times_compare_not_equal(self):
        first = AddedTimeFeatures(
            TimeFeatureReference(
                lead_times=[
                    1,
                    2,
                ],
            ),
            time_features=[
                "month_sin",
            ],
        )
        second = AddedTimeFeatures(
            TimeFeatureReference(
                lead_times=[
                    1,
                    3,
                ],
            ),
            time_features=[
                "month_sin",
            ],
        )

        assert first != second

    @pytest.mark.pruned
    def test_different_common_times_compare_not_equal(self):
        first = AddedTimeFeatures(
            TimeFeatureReference(
                common_times=[
                    "2000-01-01",
                    "2001-01-01",
                ],
            ),
            time_features=[
                "month_sin",
            ],
        )
        second = AddedTimeFeatures(
            TimeFeatureReference(
                common_times=[
                    "2000-01-01",
                    "2002-01-01",
                ],
            ),
            time_features=[
                "month_sin",
            ],
        )

        assert first != second


class TestAdditionalDatasetInitializationChecks:
    def test_unfitted_preprocessors_raise(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
        )
        config._fitted_preprocessors = False

        dataset = ConcreteDataset(
            config=config,
        )

        with pytest.raises(
            RuntimeError,
            match="fit preprocessors first",
        ):
            dataset._check_init()

    def test_unavailable_requested_times_raise(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            available_times=[
                "2000-01-01",
            ],
        )

        dataset = ConcreteDataset(
            config=config,
            requested_times=np.asarray(
                [
                    "2001-01-01",
                ],
                dtype="datetime64[ns]",
            ),
        )

        with pytest.raises(
            ValueError,
            match="initialization times are unavailable",
        ):
            dataset._check_init()

    @pytest.mark.pruned
    def test_valid_initialization_returns_none(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
        )

        dataset = ConcreteDataset(
            config=config,
        )

        assert dataset._check_init() is None


class TestAdditionalDatasetMasks:
    def test_missing_mask_is_created_as_all_false(
        self,
        monkeypatch,
    ):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
        )
        dataset = ConcreteDataset(
            config=config,
            mask=None,
        )

        template = xr.DataArray(
            np.ones(
                (
                    2,
                    3,
                ),
                dtype=bool,
            ),
            dims=(
                INIT_TIME_DIM,
                LEAD_TIME_DIM,
            ),
            coords={
                INIT_TIME_DIM: np.asarray(
                    [
                        "2000-01-01",
                        "2001-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                LEAD_TIME_DIM: [
                    1,
                    2,
                    3,
                ],
            },
        )

        create_mask = Mock(
            return_value=template,
        )
        monkeypatch.setattr(
            module,
            "_create_train_mask",
            create_mask,
        )

        dataset._resolve_mask()

        create_mask.assert_called_once_with(
            init_times=config.available_times,
            lead_times=config.input_lead_times,
            lead_time_resolution=module.lead_time_resolution,
        )
        assert not dataset.mask.any().item()

    def test_mask_missing_initialization_dimension_raises(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
        )
        dataset = ConcreteDataset(
            config=config,
            mask=xr.DataArray(
                np.zeros(
                    3,
                    dtype=bool,
                ),
                dims=(LEAD_TIME_DIM,),
                coords={
                    LEAD_TIME_DIM: [
                        1,
                        2,
                        3,
                    ]
                },
            ),
        )

        with pytest.raises(
            ValueError,
            match="mask must have",
        ):
            dataset._resolve_mask()

    @pytest.mark.pruned
    def test_mask_missing_lead_time_dimension_raises(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
        )
        dataset = ConcreteDataset(
            config=config,
            mask=xr.DataArray(
                np.zeros(
                    2,
                    dtype=bool,
                ),
                dims=(INIT_TIME_DIM,),
                coords={
                    INIT_TIME_DIM: np.asarray(
                        [
                            "2000-01-01",
                            "2001-01-01",
                        ],
                        dtype="datetime64[ns]",
                    )
                },
            ),
        )

        with pytest.raises(
            ValueError,
            match="mask must have",
        ):
            dataset._resolve_mask()

    @pytest.mark.pruned
    def test_sampling_selectors_use_requested_times_and_leads(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            lead_times=np.asarray(
                [
                    1,
                    3,
                ]
            ),
        )
        config.get_input_times = Mock(
            return_value="INPUT_TIMES",
        )

        dataset = ConcreteDataset(
            config=config,
        )

        selectors = dataset._sampling_times_selectors

        config.get_input_times.assert_called_once_with(
            dataset.requested_times,
        )
        assert selectors == {
            INIT_TIME_DIM: "INPUT_TIMES",
            LEAD_TIME_DIM: config.lead_times,
        }

    def test_prepare_sampling_mask_requires_both_selectors(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
        )
        dataset = ConcreteDataset(
            config=config,
            mask=xr.DataArray(
                np.zeros(
                    (
                        2,
                        3,
                    ),
                    dtype=bool,
                ),
                dims=(
                    INIT_TIME_DIM,
                    LEAD_TIME_DIM,
                ),
                coords={
                    INIT_TIME_DIM: np.asarray(
                        [
                            "2000-01-01",
                            "2001-01-01",
                        ],
                        dtype="datetime64[ns]",
                    ),
                    LEAD_TIME_DIM: [
                        1,
                        2,
                        3,
                    ],
                },
            ),
        )

        with pytest.raises(
            ValueError,
            match="No selectors provided",
        ):
            dataset._prepare_sampling_mask(
                {
                    INIT_TIME_DIM: dataset.requested_times,
                }
            )


class TestAdditionalSamplingCoordinates:
    @pytest.mark.pruned
    def test_get_sampling_coords_drops_masked_samples(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
        )
        dataset = ConcreteDataset(
            config=config,
        )
        dataset.mask = xr.DataArray(
            np.asarray(
                [
                    [
                        False,
                        True,
                    ],
                    [
                        False,
                        False,
                    ],
                ],
                dtype=bool,
            ),
            dims=(
                INIT_TIME_DIM,
                LEAD_TIME_DIM,
            ),
            coords={
                INIT_TIME_DIM: np.asarray(
                    [
                        "2000-01-01",
                        "2001-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                LEAD_TIME_DIM: [
                    1,
                    2,
                ],
            },
        ).where(lambda value: ~value)

        coords = dataset.get_sampling_coords()

        assert len(coords[INIT_TIME_DIM]) == 3
        assert len(coords[LEAD_TIME_DIM]) == 3

    @pytest.mark.pruned
    def test_dataset_length_uses_first_sample_coordinate(self):
        dataset = ConcreteDataset()
        dataset.sample_coords = {
            INIT_TIME_DIM: np.asarray(
                [
                    1,
                    2,
                    3,
                    4,
                ]
            ),
            LEAD_TIME_DIM: np.asarray(
                [
                    1,
                    1,
                    1,
                    1,
                ]
            ),
        }

        assert len(dataset) == 4


class TestAdditionalDatasetIndexes:
    def test_model_indexes_are_none_when_model_is_not_loaded(self):
        dataset = ConcreteDataset(
            load_model=False,
        )

        assert (
            dataset.get_model_indexes(
                {
                    INIT_TIME_DIM: np.asarray(
                        [
                            "2000-01-01",
                        ],
                        dtype="datetime64[ns]",
                    )
                }
            )
            is None
        )

    def test_model_indexes_are_created_for_valid_coordinates(self):
        model = make_data_config()
        config = ConcreteDatasetConfig(
            model=model,
        )
        dataset = ConcreteDataset(
            config=config,
            load_model=True,
        )

        indexes = dataset.get_model_indexes(
            {
                INIT_TIME_DIM: np.asarray(
                    [
                        "2001-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                LEAD_TIME_DIM: np.asarray(
                    [
                        2,
                    ]
                ),
            }
        )

        np.testing.assert_array_equal(
            indexes[INIT_TIME_DIM],
            np.asarray(
                [
                    1,
                ]
            ),
        )
        np.testing.assert_array_equal(
            indexes[LEAD_TIME_DIM],
            np.asarray(
                [
                    1,
                ]
            ),
        )

    def test_missing_model_indexes_raise(self):
        model = make_data_config()
        config = ConcreteDatasetConfig(
            model=model,
        )
        dataset = ConcreteDataset(
            config=config,
            load_model=True,
        )

        with pytest.raises(
            ValueError,
            match="not found in the model dataset",
        ):
            dataset.get_model_indexes(
                {
                    INIT_TIME_DIM: np.asarray(
                        [
                            "1999-01-01",
                        ],
                        dtype="datetime64[ns]",
                    ),
                    LEAD_TIME_DIM: np.asarray(
                        [
                            1,
                        ]
                    ),
                }
            )

    @pytest.mark.pruned
    def test_condition_indexes_none_without_condition(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
            condition=None,
        )
        dataset = ConcreteDataset(
            config=config,
        )

        assert dataset.get_cond_indexes({}) is None

    def test_condition_indexes_none_for_static_condition(self):
        condition = make_data_config(
            name="condition",
        )
        config = ConcreteDatasetConfig(
            condition=condition,
            condition_method="static",
        )
        config._effective_condition = condition

        dataset = ConcreteDataset(
            config=config,
        )

        assert dataset.get_cond_indexes({}) is None

    @pytest.mark.pruned
    def test_same_member_condition_indexes_require_realizations(self):
        condition = make_data_config(
            name="condition",
            realizations=[
                0,
                1,
            ],
        )
        config = ConcreteDatasetConfig(
            condition=condition,
            condition_method="same_member",
        )
        config._effective_condition = condition

        dataset = ConcreteDataset(
            config=config,
        )

        with pytest.raises(
            ValueError,
            match="same_member",
        ):
            dataset.get_cond_indexes(
                {
                    INIT_TIME_DIM: np.asarray(
                        [
                            "2000-01-01",
                        ],
                        dtype="datetime64[ns]",
                    )
                }
            )

    def test_same_member_condition_indexes_include_realizations(self):
        condition = make_data_config(
            name="condition",
            realizations=[
                0,
                1,
            ],
        )
        config = ConcreteDatasetConfig(
            condition=condition,
            condition_method="same_member",
        )
        config._effective_condition = condition

        dataset = ConcreteDataset(
            config=config,
        )

        indexes = dataset.get_cond_indexes(
            {
                INIT_TIME_DIM: np.asarray(
                    [
                        "2000-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                LEAD_TIME_DIM: np.asarray(
                    [
                        1,
                    ]
                ),
                REALIZATION_DIM: np.asarray(
                    [
                        1,
                    ]
                ),
            }
        )

        np.testing.assert_array_equal(
            indexes[REALIZATION_DIM],
            np.asarray(
                [
                    1,
                ]
            ),
        )

    def test_missing_condition_indexes_raise(self):
        condition = make_data_config(
            name="condition",
            ensemble_mean=True,
        )
        config = ConcreteDatasetConfig(
            condition=condition,
            condition_method="ensemble_mean",
        )
        config._effective_condition = condition

        dataset = ConcreteDataset(
            config=config,
        )

        with pytest.raises(
            ValueError,
            match="conditioning dataset",
        ):
            dataset.get_cond_indexes(
                {
                    INIT_TIME_DIM: np.asarray(
                        [
                            "1999-01-01",
                        ],
                        dtype="datetime64[ns]",
                    )
                }
            )


class TestAdditionalInputShapeBranches:
    @pytest.mark.pruned
    def test_shape_without_spatial_dimensions_contains_channels_only(self):
        model = make_data_config(
            names=[
                "tas",
                "pr",
            ],
        )
        model.preprocessing_pipeline = SimpleNamespace(
            has_flattener=False,
        )

        config = ConcreteDatasetConfig(
            model=model,
        )
        dataset = ConcreteDataset(
            config=config,
        )

        assert dataset.get_input_shape() == (2,)

    @pytest.mark.pruned
    def test_shape_uses_supported_spatial_dimensions(self):
        model = make_data_config(
            names=[
                "tas",
                "pr",
            ],
            spatial_coords={
                "lat": [
                    45.0,
                    46.0,
                ],
                "lon": [
                    -124.0,
                    -123.0,
                    -122.0,
                ],
            },
        )
        model.preprocessing_pipeline = SimpleNamespace(
            has_flattener=False,
        )

        config = ConcreteDatasetConfig(
            model=model,
        )
        dataset = ConcreteDataset(
            config=config,
        )

        assert dataset.get_input_shape() == (
            2,
            2,
            3,
        )

    def test_shape_adds_condition_channels_when_concatenating(self):
        model = make_data_config(
            names=[
                "tas",
                "pr",
            ],
        )
        model.preprocessing_pipeline = SimpleNamespace(
            has_flattener=False,
        )
        condition = make_data_config(
            name="condition",
            names=[
                "zg",
                "psl",
                "sic",
            ],
        )

        config = ConcreteDatasetConfig(
            model=model,
            condition=condition,
        )
        config._effective_condition = condition

        dataset = ConcreteDataset(
            config=config,
            concat_condition=True,
        )

        assert dataset.get_input_shape() == (5,)

    def test_shape_uses_flattener_locations(self):
        flattener = SimpleNamespace(final_locations=np.zeros((7,)))
        pipeline = SimpleNamespace(
            has_flattener=True,
            get_flattener=Mock(
                return_value=flattener,
            ),
        )
        model = make_data_config(
            names=[
                "tas",
                "pr",
            ],
            preprocessing_pipeline=pipeline,
        )

        config = ConcreteDatasetConfig(
            model=model,
        )
        dataset = ConcreteDataset(
            config=config,
        )

        assert dataset.get_input_shape() == (
            2,
            7,
        )
        pipeline.get_flattener.assert_called_once_with()

    @pytest.mark.pruned
    def test_added_features_dimension_uses_time_feature_count(self):
        features = AddedTimeFeatures(
            TimeFeatureReference(),
            time_features=[
                "month_sin",
                "month_cos",
                LEAD_TIME_DIM,
            ],
        )
        dataset = ConcreteDataset(
            time_features=features,
        )

        assert dataset.get_added_features_dim() == 3


class TestAdditionalDatasetIndexing:
    def test_index_condition_returns_none_without_condition(self):
        config = ConcreteDatasetConfig(
            model=make_data_config(),
        )
        dataset = ConcreteDataset(
            config=config,
        )

        assert dataset._index_condition_dataset(0) is None

    def test_static_condition_uses_empty_selection(self):
        condition = make_data_config(
            name="condition",
        )
        condition.isel = Mock(
            return_value=condition.data,
        )

        config = ConcreteDatasetConfig(
            condition=condition,
            condition_method="static",
        )
        config._effective_condition = condition

        dataset = ConcreteDataset(
            config=config,
        )

        with patch.object(
            module,
            "_unwrap_data_variables",
            return_value="CONDITION",
        ) as unwrap:
            result = dataset._index_condition_dataset(0)

        condition.isel.assert_called_once_with()
        unwrap.assert_called_once_with(
            condition.data,
        )
        assert result == "CONDITION"

    @pytest.mark.pruned
    def test_cross_ensemble_selects_random_member(self):
        condition = make_data_config(
            name="condition",
            realizations=[
                0,
                1,
                2,
            ],
        )
        condition.isel = Mock(
            return_value=condition.data,
        )

        config = ConcreteDatasetConfig(
            condition=condition,
            condition_method="cross_ensemble",
        )
        config._effective_condition = condition

        dataset = ConcreteDataset(
            config=config,
        )
        dataset.cond_indexes = {
            INIT_TIME_DIM: np.asarray(
                [
                    0,
                ]
            ),
            LEAD_TIME_DIM: np.asarray(
                [
                    1,
                ]
            ),
        }
        dataset._rng = Mock()
        dataset._rng.integers.return_value = 2

        with patch.object(
            module,
            "_unwrap_data_variables",
            return_value="CONDITION",
        ):
            result = dataset._index_condition_dataset(0)

        dataset._rng.integers.assert_called_once_with(3)
        condition.isel.assert_called_once_with(
            **{
                INIT_TIME_DIM: [
                    0,
                ],
                LEAD_TIME_DIM: [
                    1,
                ],
                REALIZATION_DIM: [
                    2,
                ],
            }
        )
        assert result == "CONDITION"

    def test_index_model_returns_none_when_model_not_loaded(self):
        dataset = ConcreteDataset(
            load_model=False,
        )

        assert dataset._index_model_dataset(0) is None

    def test_index_model_casts_indexes_to_python_int(self):
        model = make_data_config()
        model.isel = Mock(
            return_value=model.data,
        )

        config = ConcreteDatasetConfig(
            model=model,
        )
        dataset = ConcreteDataset(
            config=config,
            load_model=True,
        )
        dataset.model_indexes = {
            INIT_TIME_DIM: np.asarray(
                [
                    np.int64(1),
                ]
            ),
            LEAD_TIME_DIM: np.asarray(
                [
                    np.int64(2),
                ]
            ),
        }

        with patch.object(
            module,
            "_unwrap_data_variables",
            return_value="MODEL",
        ):
            result = dataset._index_model_dataset(0)

        model.isel.assert_called_once_with(
            **{
                INIT_TIME_DIM: [
                    1,
                ],
                LEAD_TIME_DIM: [
                    2,
                ],
            }
        )
        assert result == "MODEL"


@pytest.mark.pruned
def test_compute_uses_synchronous_scheduler_and_returns_tuple():
    first = object()
    second = object()

    suppress_context = Mock()
    suppress_context.__enter__ = Mock(
        return_value=None,
    )
    suppress_context.__exit__ = Mock(
        return_value=False,
    )

    dask_context = Mock()
    dask_context.__enter__ = Mock(
        return_value=None,
    )
    dask_context.__exit__ = Mock(
        return_value=False,
    )

    with (
        patch.object(
            module,
            "suppress_stderr",
            return_value=suppress_context,
        ),
        patch.object(
            module.dask.config,
            "set",
            return_value=dask_context,
        ) as config_set,
        patch.object(
            module.dask,
            "compute",
            return_value=(
                "FIRST",
                "SECOND",
            ),
        ) as compute,
    ):
        result = DatasetABC._compute(
            first,
            second,
        )

    config_set.assert_called_once_with(
        scheduler="synchronous",
    )
    compute.assert_called_once_with(
        first,
        second,
    )
    assert result == (
        "FIRST",
        "SECOND",
    )
