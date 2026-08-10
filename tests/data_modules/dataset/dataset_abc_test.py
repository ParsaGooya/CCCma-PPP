from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from cccma_ppp.data_modules.dataset.dataset_abc import (
    AddedTimeFeatures,
    DatasetABC,
    DatasetConfigABC,
    lead_time_config,
)


INIT_TIME_DIM = DatasetConfigABC.init_time_dim
LEAD_TIME_DIM = DatasetConfigABC.lead_time_dim
REALIZATION_DIM = DatasetConfigABC.realization_dim
init_time_dim = INIT_TIME_DIM
lead_time_dim = LEAD_TIME_DIM
realization_dim = REALIZATION_DIM


def make_times():
    return np.array(
        [
            "2000-01-01",
            "2000-02-01",
            "2000-03-01",
        ],
        dtype="datetime64[ns]",
    )


def make_time_coord():
    return xr.DataArray(
        make_times(),
        dims=(INIT_TIME_DIM,),
        coords={
            INIT_TIME_DIM: make_times(),
        },
        name=INIT_TIME_DIM,
    )


def make_lead_coord():
    values = np.array(
        [
            1,
            2,
            3,
        ]
    )

    return xr.DataArray(
        values,
        dims=(LEAD_TIME_DIM,),
        coords={
            LEAD_TIME_DIM: values,
        },
        name=LEAD_TIME_DIM,
    )


def make_info(
    *,
    times=None,
    lead_times=None,
    realizations=None,
    extra_coords=None,
    time_coords_type="datetime64",
):
    if times is None:
        times = make_times()

    if lead_times is None:
        lead_times = np.array([1, 2, 3])

    coords = {
        INIT_TIME_DIM: xr.DataArray(
            times,
            dims=(INIT_TIME_DIM,),
            coords={
                INIT_TIME_DIM: times,
            },
        ),
        LEAD_TIME_DIM: xr.DataArray(
            lead_times,
            dims=(LEAD_TIME_DIM,),
            coords={
                LEAD_TIME_DIM: lead_times,
            },
        ),
    }

    if realizations is not None:
        coords[REALIZATION_DIM] = xr.DataArray(
            realizations,
            dims=(REALIZATION_DIM,),
            coords={
                REALIZATION_DIM: realizations,
            },
        )

    if extra_coords is not None:
        coords.update(extra_coords)

    return SimpleNamespace(
        coords=coords,
        time_coords_type=time_coords_type,
    )


def make_data_config(
    *,
    paths="data",
    names=None,
    times=None,
    lead_times=None,
    realizations=None,
    ensemble_mean=False,
    realization_list=None,
    preprocessing_pipeline=None,
):
    if names is None:
        names = ["var"]

    if preprocessing_pipeline is None:
        preprocessing_pipeline = MagicMock()
        preprocessing_pipeline.fitted_preprocessors = []
        preprocessing_pipeline.transform.side_effect = lambda value: value

    return SimpleNamespace(
        paths=paths,
        names=names,
        realization_list=realization_list,
        ensemble_mean=ensemble_mean,
        concat_dim=INIT_TIME_DIM,
        file_type="*.nc",
        rename_dict=None,
        list_paths=["data.nc"],
        info=make_info(
            times=times,
            lead_times=lead_times,
            realizations=realizations,
        ),
        preprocessing_pipeline=preprocessing_pipeline,
    )


class ConcreteDatasetConfig(DatasetConfigABC):
    def _check_condition(self):
        if self.condition_method is None:
            return self
        return super()._check_condition()

    def _check_model(self):
        if self.condition_method is None:
            return self
        return super()._check_model()

    @property
    def input_lead_times(self):
        coords = self.effective_input.info.coords
        if self.lead_time_dim in coords:
            return coords[self.lead_time_dim].values
        return np.asarray(self.lead_times)

    @property
    def _using_model_data_as_condition(self):
        if self.condition is None and self.condition_method is None:
            return False
        return super()._using_model_data_as_condition

    def __init__(
        self,
        *,
        model=None,
        condition=None,
        condition_method=None,
        lead_times=None,
        available_times=None,
        effective_input=None,
    ):
        self.model = model
        self.condition = condition
        self.condition_method = condition_method
        self.lead_times = lead_times
        self._available_times = (
            make_times() if available_times is None else np.asarray(available_times)
        )
        self._effective_input_override = effective_input
        self._ds_operator = MagicMock()

        super().__init__()
        self._check_model_vs_condition()
        self._check_model_vs_condition()

    @property
    def available_times(self):
        return self._available_times

    @property
    def ds_operator(self):
        return self._ds_operator

    @property
    def effective_input(self):
        if self._effective_input_override is not None:
            return self._effective_input_override

        if self.model is not None:
            return self.model

        return self.effective_condition

    @property
    def get_common_time(self):
        return self.available_times

    def build_dataset(self):
        return MagicMock()


class ConcreteDataset(DatasetABC):
    @property
    def _load_model(self):
        return getattr(
            self,
            "load_model",
            True,
        )

    @property
    def _write_condition_to_input(self):
        return getattr(
            self,
            "write_condition_to_input",
            False,
        )

    @property
    def _concat_condition_to_input(self):
        return getattr(
            self,
            "concat_condition_to_input",
            False,
        )


def make_dataset_without_init():
    return ConcreteDataset.__new__(ConcreteDataset)


def make_reference_config(
    *,
    lead_times=None,
    common_times=None,
):
    if lead_times is None:
        lead_times = np.array(
            [
                1,
                2,
                3,
                4,
            ]
        )

    if common_times is None:
        common_times = np.array(
            [
                "2000-01-01",
                "2001-01-01",
            ],
            dtype="datetime64[ns]",
        )

    return SimpleNamespace(
        lead_time_resolution="month",
        init_time_dim=INIT_TIME_DIM,
        lead_time_dim=LEAD_TIME_DIM,
        lead_times=np.asarray(lead_times),
        get_common_time=np.asarray(common_times),
    )


def make_feature_input(
    shape=(1,),
):
    if len(shape) == 1:
        dims = ("channels",)
    elif len(shape) == 2:
        dims = (
            "channels",
            "latitude",
        )
    elif len(shape) == 3:
        dims = (
            "channels",
            "latitude",
            "longitude",
        )
    else:
        dims = tuple(f"dim_{index}" for index in range(len(shape)))

    return xr.DataArray(
        np.zeros(
            shape,
            dtype=np.float32,
        ),
        dims=dims,
    )


class TestLeadTimeConfig:
    @pytest.mark.pruned
    def test_requires_list_or_end(self):
        with pytest.raises(
            ValueError,
            match="Provide a list of lead_times",
        ):
            lead_time_config()

    @pytest.mark.pruned
    def test_build_explicit_lead_times(self):
        config = lead_time_config(
            list_lead_times=[
                1,
                3,
                6,
            ]
        )

        assert config.build_lead_times() == [
            1,
            3,
            6,
        ]

    @pytest.mark.pruned
    def test_build_lead_time_range(self):
        config = lead_time_config(
            start=2,
            end=5,
        )

        assert np.array_equal(
            config.build_lead_times(),
            np.array(
                [
                    2,
                    3,
                    4,
                    5,
                ]
            ),
        )

    @pytest.mark.pruned
    def test_explicit_list_takes_precedence(self):
        config = lead_time_config(
            list_lead_times=[
                2,
                4,
            ],
            start=1,
            end=10,
        )

        assert config.build_lead_times() == [
            2,
            4,
        ]


class TestDatasetConfigABC:
    def test_requires_model_or_condition(self):
        with pytest.raises(
            ValueError,
            match="either model or condition data must be provided",
        ):
            ConcreteDatasetConfig(
                model=None,
                condition=None,
                condition_method=None,
            )

    @pytest.mark.pruned
    def test_invalid_condition_method(self):
        model = make_data_config()

        with pytest.raises(
            ValueError,
            match="Invalid condition_method",
        ):
            ConcreteDatasetConfig(
                model=model,
                condition=None,
                condition_method="invalid",
            )

    def test_resolves_lead_time_config(self):
        model = make_data_config(
            lead_times=np.array(
                [
                    1,
                    2,
                    3,
                    4,
                ]
            )
        )

        config = ConcreteDatasetConfig(
            model=model,
            condition=None,
            condition_method=None,
            lead_times=lead_time_config(
                start=2,
                end=4,
            ),
        )

        assert np.array_equal(
            config.lead_times,
            np.array(
                [
                    2,
                    3,
                    4,
                ]
            ),
        )

    @pytest.mark.pruned
    def test_defaults_to_all_input_lead_times(self):
        model = make_data_config(
            lead_times=np.array(
                [
                    1,
                    2,
                    6,
                ]
            )
        )

        config = ConcreteDatasetConfig(
            model=model,
            condition=None,
            condition_method=None,
            lead_times=None,
        )

        assert np.array_equal(
            config.lead_times,
            np.array(
                [
                    1,
                    2,
                    6,
                ]
            ),
        )

    @pytest.mark.pruned
    def test_rejects_unavailable_lead_times(self):
        model = make_data_config(
            lead_times=np.array(
                [
                    1,
                    2,
                    3,
                ]
            )
        )

        with pytest.raises(
            ValueError,
            match="requested lead times are not available",
        ):
            ConcreteDatasetConfig(
                model=model,
                condition=None,
                condition_method=None,
                lead_times=[
                    1,
                    12,
                ],
            )

    @pytest.mark.pruned
    def test_condition_becomes_effective_condition(self):
        condition = make_data_config(ensemble_mean=True)

        config = ConcreteDatasetConfig(
            model=None,
            condition=condition,
            condition_method="ensemble_mean",
        )

        assert config.effective_condition is condition
        assert config.effective_input is condition

    @pytest.mark.pruned
    def test_no_condition_resolves_to_none(self):
        model = make_data_config()

        config = ConcreteDatasetConfig(
            model=model,
            condition=None,
            condition_method=None,
        )

        assert config.effective_condition is None

    @pytest.mark.parametrize(
        "condition_method,expected_ensemble_mean",
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
    def test_model_as_condition_constructor_arguments(
        self,
        condition_method,
        expected_ensemble_mean,
    ):
        model = make_data_config(
            paths="model-data",
            names=[
                "tas",
                "pr",
            ],
            realizations=[
                0,
                1,
            ],
            realization_list=[
                0,
                1,
            ],
            ensemble_mean=False,
        )

        with patch(
            "cccma_ppp.data_modules.dataset.dataset_abc.ModelDataConfig"
        ) as model_config:
            copied_model = MagicMock()
            copied_model.info = model.info
            copied_model.ensemble_mean = expected_ensemble_mean
            model_config.return_value = copied_model

            config = ConcreteDatasetConfig(
                model=model,
                condition=None,
                condition_method=condition_method,
            )

        assert config.effective_condition is copied_model

        model_config.assert_called_once_with(
            paths="model-data",
            names=[
                "tas",
                "pr",
            ],
            preprocessing_pipeline=model.preprocessing_pipeline,
            realization_list=[
                0,
                1,
            ],
            concat_dim=INIT_TIME_DIM,
            file_type="*.nc",
            ensemble_mean=expected_ensemble_mean,
            rename_dict=None,
        )

    @pytest.mark.pruned
    def test_using_same_model_source_as_condition(self):
        model = make_data_config(
            paths="same-path",
            names=["tas"],
            realization_list=[0],
        )
        condition = make_data_config(
            paths="same-path",
            names=["tas"],
            realization_list=[0],
        )

        config = ConcreteDatasetConfig.__new__(ConcreteDatasetConfig)
        config.model = model
        config.condition = condition
        config.condition_method = "same_member"

        assert config._using_model_data_as_condition is True

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "condition_paths,condition_names,condition_members",
        [
            (
                "different-path",
                ["tas"],
                [0],
            ),
            (
                "same-path",
                ["pr"],
                [0],
            ),
            (
                "same-path",
                ["tas"],
                [1],
            ),
        ],
    )
    def test_different_condition_is_not_model_source(
        self,
        condition_paths,
        condition_names,
        condition_members,
    ):
        model = make_data_config(
            paths="same-path",
            names=["tas"],
            realization_list=[0],
        )
        condition = make_data_config(
            paths=condition_paths,
            names=condition_names,
            realization_list=condition_members,
        )

        config = ConcreteDatasetConfig.__new__(ConcreteDatasetConfig)
        config.model = model
        config.condition = condition
        config.condition_method = "same_member"

        assert config._using_model_data_as_condition is False

    @pytest.mark.pruned
    def test_model_and_condition_time_coordinates_must_match(self):
        model = make_data_config(
            times=np.array(
                [
                    "2000-01-01",
                    "2000-02-01",
                ],
                dtype="datetime64[ns]",
            )
        )
        condition = make_data_config(
            paths="condition",
            ensemble_mean=True,
            times=np.array(
                [
                    "2000-01-01",
                ],
                dtype="datetime64[ns]",
            ),
        )

        with pytest.raises(
            ValueError,
            match="same.*coordinates",
        ):
            ConcreteDatasetConfig(
                model=model,
                condition=condition,
                condition_method="ensemble_mean",
            )

    @pytest.mark.pruned
    def test_condition_coordinate_superset_is_allowed(self):
        model = make_data_config(
            times=np.array(
                [
                    "2000-01-01",
                    "2000-02-01",
                ],
                dtype="datetime64[ns]",
            )
        )
        condition = make_data_config(
            ensemble_mean=True,
            times=np.array(
                [
                    "1999-12-01",
                    "2000-01-01",
                    "2000-02-01",
                    "2000-03-01",
                ],
                dtype="datetime64[ns]",
            ),
        )

        config = ConcreteDatasetConfig(
            model=model,
            condition=condition,
            condition_method="ensemble_mean",
        )

        assert config.effective_condition is condition

    @pytest.mark.pruned
    def test_model_and_condition_time_types_must_match(self):
        model = make_data_config()
        condition = make_data_config(ensemble_mean=True, paths="condition")

        model.info.time_coords_type = "datetime64"
        condition.info.time_coords_type = "cftime"

        with pytest.raises(
            ValueError,
            match="same cftime/datetime type",
        ):
            ConcreteDatasetConfig(
                model=model,
                condition=condition,
                condition_method="ensemble_mean",
            )

    @pytest.mark.pruned
    def test_same_member_requires_condition_realizations(self):
        model = make_data_config(
            realizations=[
                0,
                1,
            ],
            ensemble_mean=False,
        )
        condition = make_data_config(
            realizations=None,
            ensemble_mean=False,
        )

        with pytest.raises(
            ValueError,
            match="dim must exist in the condition",
        ):
            ConcreteDatasetConfig(
                model=model,
                condition=condition,
                condition_method="same_member",
            )

    @pytest.mark.pruned
    def test_same_member_accepts_equal_realizations(self):
        model = make_data_config(
            realizations=[
                0,
                1,
            ],
            ensemble_mean=False,
        )
        condition = make_data_config(
            realizations=[
                0,
                1,
            ],
            ensemble_mean=False,
        )

        config = ConcreteDatasetConfig(
            model=model,
            condition=condition,
            condition_method="same_member",
        )

        assert config.effective_condition is condition

    @pytest.mark.pruned
    def test_same_member_rejects_model_ensemble_mean(self):
        model = make_data_config(
            realizations=[
                0,
                1,
            ],
            ensemble_mean=True,
        )
        condition = make_data_config(
            realizations=[
                0,
                1,
            ],
            ensemble_mean=False,
        )

        with pytest.raises(
            ValueError,
            match="model data should not be ensemble mean",
        ):
            ConcreteDatasetConfig(
                model=model,
                condition=condition,
                condition_method="same_member",
            )

    @pytest.mark.parametrize(
        "condition_method",
        [
            "cross_ensemble",
            "same_member",
        ],
    )
    def test_member_conditioning_rejects_condition_ensemble_mean(
        self,
        condition_method,
    ):
        model = make_data_config(
            realizations=[
                0,
                1,
            ],
            ensemble_mean=False,
        )
        condition = make_data_config(
            realizations=[
                0,
                1,
            ],
            ensemble_mean=True,
        )

        with pytest.raises(
            ValueError,
            match="ensemble_mean cannot be True",
        ):
            ConcreteDatasetConfig(
                model=model,
                condition=condition,
                condition_method=condition_method,
            )

    @pytest.mark.pruned
    def test_ensemble_mean_conditioning_requires_ensemble_mean(self):
        condition = make_data_config(
            ensemble_mean=False,
        )

        with pytest.raises(
            ValueError,
            match="Ensemble mean must be True",
        ):
            ConcreteDatasetConfig(
                model=None,
                condition=condition,
                condition_method="ensemble_mean",
            )

    @pytest.mark.pruned
    def test_static_condition_rejects_realization_list(self):
        condition = make_data_config(
            realization_list=[0],
        )
        condition.info.coords = {}

        with pytest.raises(
            ValueError,
            match="cannot specify realization list",
        ):
            ConcreteDatasetConfig(
                model=None,
                condition=condition,
                condition_method="static",
            )

    @pytest.mark.pruned
    def test_static_condition_rejects_sampling_coordinates(self):
        condition = make_data_config()
        condition.info.coords = {
            INIT_TIME_DIM: make_time_coord(),
        }

        with pytest.raises(
            ValueError,
            match="cannot have.*sampling dimensions",
        ):
            ConcreteDatasetConfig(
                model=None,
                condition=condition,
                condition_method="static",
            )

    @pytest.mark.pruned
    def test_static_condition_accepts_no_sampling_coordinates(self):
        condition = make_data_config()
        condition.info.coords = {
            "channels": xr.DataArray(
                [
                    0,
                    1,
                ],
                dims=("channels",),
            ),
        }

        config = ConcreteDatasetConfig(
            model=None,
            condition=condition,
            condition_method="static",
            lead_times=[1],
        )

        assert config.effective_condition is condition

    @pytest.mark.pruned
    def test_get_input_times(self):
        available = make_times()
        model = make_data_config(
            times=available,
        )

        config = ConcreteDatasetConfig(
            model=model,
            condition=None,
            condition_method=None,
            available_times=available,
        )

        requested = xr.DataArray(
            available[:2],
            dims=(INIT_TIME_DIM,),
            coords={
                INIT_TIME_DIM: available[:2],
            },
        )

        result = config.get_input_times(requested)

        assert np.array_equal(
            result.values,
            available[:2],
        )

    @pytest.mark.pruned
    def test_get_input_times_rejects_unavailable_values(self):
        available = make_times()
        model = make_data_config(
            times=available,
        )

        config = ConcreteDatasetConfig(
            model=model,
            condition=None,
            condition_method=None,
            available_times=available,
        )

        requested_values = np.array(
            [
                "1990-01-01",
            ],
            dtype="datetime64[ns]",
        )
        requested = xr.DataArray(
            requested_values,
            dims=(INIT_TIME_DIM,),
            coords={
                INIT_TIME_DIM: requested_values,
            },
        )

        with pytest.raises(
            ValueError,
            match="requested_times are unavailable",
        ):
            config.get_input_times(requested)


class TestAddedTimeFeatures:
    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "features,expected",
        [
            (
                None,
                (),
            ),
            (
                [],
                (),
            ),
            (
                [INIT_TIME_DIM],
                (INIT_TIME_DIM,),
            ),
            (
                [LEAD_TIME_DIM],
                (LEAD_TIME_DIM,),
            ),
            (
                [
                    "month_sin",
                    "month_cos",
                ],
                (
                    "month_sin",
                    "month_cos",
                ),
            ),
            (
                [
                    "day_sin",
                    "day_cos",
                ],
                (
                    "day_sin",
                    "day_cos",
                ),
            ),
        ],
    )
    def test_valid_features(
        self,
        features,
        expected,
    ):
        added = AddedTimeFeatures(
            make_reference_config(),
            features,
        )

        assert added.time_features == expected

    @pytest.mark.pruned
    def test_features_are_stored_in_canonical_order(self):
        added = AddedTimeFeatures(
            make_reference_config(),
            [
                "day_cos",
                INIT_TIME_DIM,
                "month_sin",
                LEAD_TIME_DIM,
            ],
        )

        assert added.time_features == (
            INIT_TIME_DIM,
            LEAD_TIME_DIM,
            "month_sin",
            "day_cos",
        )

    def test_rejects_unsupported_features(self):
        with pytest.raises(
            ValueError,
            match="Unsupported time features",
        ):
            AddedTimeFeatures(
                make_reference_config(),
                [
                    "unsupported",
                ],
            )

    @pytest.mark.pruned
    def test_length(self):
        added = AddedTimeFeatures(
            make_reference_config(),
            [
                INIT_TIME_DIM,
                LEAD_TIME_DIM,
                "month_sin",
            ],
        )

        assert len(added) == 3

    @pytest.mark.pruned
    def test_days_in_normal_year(self):
        assert AddedTimeFeatures._days_in_year(np.datetime64("2001-06-01")) == 365

    @pytest.mark.pruned
    def test_days_in_leap_year(self):
        assert AddedTimeFeatures._days_in_year(np.datetime64("2000-06-01")) == 366

    def test_days_in_360_day_calendar(self):
        assert (
            AddedTimeFeatures._days_in_year(
                cftime.Datetime360Day(
                    2000,
                    1,
                    1,
                )
            )
            == 360
        )

    def test_days_in_noleap_calendar(self):
        assert (
            AddedTimeFeatures._days_in_year(
                cftime.DatetimeNoLeap(
                    2000,
                    1,
                    1,
                )
            )
            == 365
        )

    def test_days_in_all_leap_calendar(self):
        assert (
            AddedTimeFeatures._days_in_year(
                cftime.DatetimeAllLeap(
                    2001,
                    1,
                    1,
                )
            )
            == 366
        )

    def test_build_requires_initialization_dimension(self):
        added = AddedTimeFeatures(
            make_reference_config(),
            [
                INIT_TIME_DIM,
            ],
        )

        with pytest.raises(
            ValueError,
            match="missing required dimensions",
        ):
            added.build_time_features(
                {
                    LEAD_TIME_DIM: np.array([1]),
                }
            )

    @pytest.mark.pruned
    def test_build_requires_lead_time_dimension(self):
        added = AddedTimeFeatures(
            make_reference_config(),
            [
                INIT_TIME_DIM,
            ],
        )

        with pytest.raises(
            ValueError,
            match="missing required dimensions",
        ):
            added.build_time_features(
                {
                    INIT_TIME_DIM: np.array(
                        ["2000-01-01"],
                        dtype="datetime64[ns]",
                    ),
                }
            )

    def test_empty_features_do_not_call_add_lead_times(self):
        added = AddedTimeFeatures(
            make_reference_config(),
            None,
        )

        with patch(
            "cccma_ppp.data_modules.dataset.dataset_abc.add_lead_times"
        ) as mock_add:
            result = added.build_time_features(
                {
                    INIT_TIME_DIM: np.array(
                        ["2000-01-01"],
                        dtype="datetime64[ns]",
                    ),
                    LEAD_TIME_DIM: np.array([1]),
                }
            )

        assert result is added
        assert added.time_features_array is None
        mock_add.assert_not_called()

    @pytest.mark.pruned
    def test_build_lead_time_feature(self):
        reference = make_reference_config(
            lead_times=np.array(
                [
                    1,
                    2,
                    3,
                    4,
                ]
            )
        )
        added = AddedTimeFeatures(
            reference,
            [
                LEAD_TIME_DIM,
            ],
        )

        with patch(
            "cccma_ppp.data_modules.dataset.dataset_abc.add_lead_times",
            return_value=np.array(
                [
                    "2000-01-01",
                    "2000-02-01",
                ],
                dtype="datetime64[ns]",
            ),
        ):
            added.build_time_features(
                {
                    INIT_TIME_DIM: np.array(
                        [
                            "2000-01-01",
                            "2000-01-01",
                        ],
                        dtype="datetime64[ns]",
                    ),
                    LEAD_TIME_DIM: np.array(
                        [
                            1,
                            4,
                        ]
                    ),
                }
            )

        assert added.time_features_array.dtype == np.float32
        assert np.allclose(
            added.time_features_array[:, 0],
            np.array(
                [
                    0.25,
                    1.0,
                ],
                dtype=np.float32,
            ),
        )

    @pytest.mark.pruned
    def test_build_month_features(self):
        added = AddedTimeFeatures(
            make_reference_config(),
            [
                "month_sin",
                "month_cos",
            ],
        )

        target_times = np.array(
            [
                "2000-01-01",
                "2000-04-01",
            ],
            dtype="datetime64[ns]",
        )

        with patch(
            "cccma_ppp.data_modules.dataset.dataset_abc.add_lead_times",
            return_value=target_times,
        ):
            added.build_time_features(
                {
                    INIT_TIME_DIM: target_times,
                    LEAD_TIME_DIM: np.array(
                        [
                            1,
                            1,
                        ]
                    ),
                }
            )

        assert added.time_features_array.shape == (
            2,
            2,
        )
        assert np.allclose(
            added.time_features_array[0],
            np.array(
                [
                    0.0,
                    1.0,
                ],
                dtype=np.float32,
            ),
            atol=1e-6,
        )
        assert np.allclose(
            added.time_features_array[1],
            np.array(
                [
                    1.0,
                    0.0,
                ],
                dtype=np.float32,
            ),
            atol=1e-6,
        )

    @pytest.mark.pruned
    def test_build_day_features(self):
        added = AddedTimeFeatures(
            make_reference_config(),
            [
                "day_sin",
                "day_cos",
            ],
        )

        target_times = np.array(
            [
                "2001-01-01",
                "2001-07-02",
            ],
            dtype="datetime64[ns]",
        )

        with patch(
            "cccma_ppp.data_modules.dataset.dataset_abc.add_lead_times",
            return_value=target_times,
        ):
            added.build_time_features(
                {
                    INIT_TIME_DIM: target_times,
                    LEAD_TIME_DIM: np.array(
                        [
                            1,
                            1,
                        ]
                    ),
                }
            )

        assert added.time_features_array.shape == (
            2,
            2,
        )
        assert added.time_features_array.dtype == np.float32
        assert np.isfinite(added.time_features_array).all()

    def test_call_requires_built_features(self):
        added = AddedTimeFeatures(
            make_reference_config(),
            [
                LEAD_TIME_DIM,
            ],
        )

        with pytest.raises(
            RuntimeError,
            match="must be built before indexing",
        ):
            added(
                0,
                make_feature_input(),
            )

    @pytest.mark.pruned
    def test_call_rejects_negative_index(self):
        added = AddedTimeFeatures(
            make_reference_config(),
            [
                LEAD_TIME_DIM,
            ],
        )
        added.time_features_array = np.array(
            [
                [0.5],
            ],
            dtype=np.float32,
        )

        with pytest.raises(
            IndexError,
            match="out of bounds",
        ):
            added(
                -1,
                make_feature_input(),
            )

    def test_call_rejects_large_index(self):
        added = AddedTimeFeatures(
            make_reference_config(),
            [
                LEAD_TIME_DIM,
            ],
        )
        added.time_features_array = np.array(
            [
                [0.5],
            ],
            dtype=np.float32,
        )

        with pytest.raises(
            IndexError,
            match="out of bounds",
        ):
            added(
                1,
                make_feature_input(),
            )

    def test_call_without_spatial_broadcast(self):
        added = AddedTimeFeatures(
            make_reference_config(),
            [
                INIT_TIME_DIM,
                LEAD_TIME_DIM,
            ],
        )
        added.time_features_array = np.array(
            [
                [
                    0.25,
                    0.5,
                ],
            ],
            dtype=np.float32,
        )

        result = added(
            0,
            make_feature_input(
                shape=(2,),
            ),
        )

        assert result.shape == (2,)
        assert np.allclose(
            result,
            np.array(
                [
                    0.25,
                    0.5,
                ],
                dtype=np.float32,
            ),
        )

    def test_call_broadcasts_over_spatial_dimensions(self):
        added = AddedTimeFeatures(
            make_reference_config(),
            [
                INIT_TIME_DIM,
                LEAD_TIME_DIM,
            ],
        )
        added.time_features_array = np.array(
            [
                [
                    0.25,
                    0.5,
                ],
            ],
            dtype=np.float32,
        )

        result = added(
            0,
            make_feature_input(
                shape=(
                    1,
                    3,
                    4,
                ),
            ),
        )

        assert result.shape == (
            2,
            3,
            4,
        )
        assert np.allclose(
            result[0],
            0.25,
        )
        assert np.allclose(
            result[1],
            0.5,
        )

    def test_equal_instances(self):
        reference = make_reference_config()

        left = AddedTimeFeatures(
            reference,
            [
                INIT_TIME_DIM,
                "month_sin",
            ],
        )
        right = AddedTimeFeatures(
            reference,
            [
                INIT_TIME_DIM,
                "month_sin",
            ],
        )

        assert left == right

    @pytest.mark.pruned
    def test_not_equal_features(self):
        reference = make_reference_config()

        left = AddedTimeFeatures(
            reference,
            [
                INIT_TIME_DIM,
            ],
        )
        right = AddedTimeFeatures(
            reference,
            [
                LEAD_TIME_DIM,
            ],
        )

        assert left != right

    def test_equality_with_other_type_returns_not_implemented(self):
        added = AddedTimeFeatures(
            make_reference_config(),
            [
                INIT_TIME_DIM,
            ],
        )

        assert added.__eq__(object()) is NotImplemented


class TestDatasetABC:
    @pytest.mark.pruned
    def test_check_init_requires_fitted_preprocessors(self):
        dataset = make_dataset_without_init()
        dataset.requested_times = make_time_coord()
        dataset.config = SimpleNamespace(
            _fitted_preprocessors=False,
            available_times=make_times(),
        )

        with pytest.raises(
            RuntimeError,
            match="fit preprocessors first",
        ):
            dataset._check_init()

    def test_check_init_rejects_unavailable_times(self):
        dataset = make_dataset_without_init()
        requested = np.array(
            [
                "1990-01-01",
            ],
            dtype="datetime64[ns]",
        )
        dataset.requested_times = xr.DataArray(
            requested,
            dims=(INIT_TIME_DIM,),
        )
        dataset.config = SimpleNamespace(
            _fitted_preprocessors=True,
            available_times=make_times(),
        )

        with pytest.raises(
            ValueError,
            match="initialization times are unavailable",
        ):
            dataset._check_init()

    def test_check_init_accepts_available_times(self):
        dataset = make_dataset_without_init()
        dataset.requested_times = make_time_coord()
        dataset.config = SimpleNamespace(
            _fitted_preprocessors=True,
            available_times=make_times(),
        )

        dataset._check_init()

    def test_resolve_mask_creates_default(self):
        dataset = make_dataset_without_init()
        dataset.mask = None
        dataset.config = SimpleNamespace(
            available_times=make_times(),
            input_lead_times=np.array(
                [
                    1,
                    2,
                ]
            ),
            init_time_dim=INIT_TIME_DIM,
            lead_time_dim=LEAD_TIME_DIM,
        )

        mask = xr.DataArray(
            np.ones(
                (
                    3,
                    2,
                ),
                dtype=bool,
            ),
            dims=(
                INIT_TIME_DIM,
                LEAD_TIME_DIM,
            ),
            coords={
                INIT_TIME_DIM: make_times(),
                LEAD_TIME_DIM: [
                    1,
                    2,
                ],
            },
        )

        with patch(
            "cccma_ppp.data_modules.dataset.dataset_abc._create_train_mask",
            return_value=mask,
        ) as mock_create:
            dataset._resolve_mask()

        mock_create.assert_called_once()
        assert dataset.mask.dims == (
            INIT_TIME_DIM,
            LEAD_TIME_DIM,
        )
        assert not dataset.mask.values.any()

    @pytest.mark.pruned
    def test_resolve_mask_preserves_existing_mask(self):
        dataset = make_dataset_without_init()
        existing = xr.DataArray(
            np.zeros(
                (
                    2,
                    2,
                ),
                dtype=bool,
            ),
            dims=(
                INIT_TIME_DIM,
                LEAD_TIME_DIM,
            ),
        )
        dataset.mask = existing
        dataset.config = SimpleNamespace(
            init_time_dim=INIT_TIME_DIM,
            lead_time_dim=LEAD_TIME_DIM,
        )

        with patch(
            "cccma_ppp.data_modules.dataset.dataset_abc._create_train_mask"
        ) as mock_create:
            dataset._resolve_mask()

        assert dataset.mask is existing
        mock_create.assert_not_called()

    def test_resolve_mask_rejects_missing_dimension(self):
        dataset = make_dataset_without_init()
        dataset.mask = xr.DataArray(
            np.zeros(
                (
                    2,
                    2,
                ),
                dtype=bool,
            ),
            dims=(
                INIT_TIME_DIM,
                "wrong",
            ),
        )
        dataset.config = SimpleNamespace(
            init_time_dim=INIT_TIME_DIM,
            lead_time_dim=LEAD_TIME_DIM,
        )

        with pytest.raises(
            ValueError,
            match="mask must have",
        ):
            dataset._resolve_mask()

    @pytest.mark.pruned
    def test_sampling_times_selectors(self):
        dataset = make_dataset_without_init()
        requested = make_time_coord()
        selected_times = requested.isel(
            {
                INIT_TIME_DIM: slice(
                    0,
                    2,
                )
            }
        )

        config = MagicMock()
        config.init_time_dim = INIT_TIME_DIM
        config.lead_time_dim = LEAD_TIME_DIM
        config.lead_times = np.array(
            [
                1,
                2,
            ]
        )
        config.get_input_times.return_value = selected_times

        dataset.config = config
        dataset.requested_times = requested

        result = dataset._sampling_times_selectors

        assert result[INIT_TIME_DIM] is selected_times
        assert np.array_equal(
            result[LEAD_TIME_DIM],
            np.array(
                [
                    1,
                    2,
                ]
            ),
        )

    def test_prepare_sampling_mask_requires_all_selectors(self):
        dataset = make_dataset_without_init()
        dataset.config = SimpleNamespace(
            init_time_dim=INIT_TIME_DIM,
            lead_time_dim=LEAD_TIME_DIM,
        )

        with pytest.raises(
            ValueError,
            match="No selectors provided",
        ):
            dataset._prepare_sampling_mask(
                {
                    INIT_TIME_DIM: make_time_coord(),
                }
            )

    def test_prepare_sampling_mask(self):
        dataset = make_dataset_without_init()
        times = make_times()[:2]
        leads = np.array(
            [
                1,
                2,
            ]
        )

        dataset.mask = xr.DataArray(
            np.array(
                [
                    [
                        False,
                        True,
                    ],
                    [
                        False,
                        False,
                    ],
                ]
            ),
            dims=(
                INIT_TIME_DIM,
                LEAD_TIME_DIM,
            ),
            coords={
                INIT_TIME_DIM: times,
                LEAD_TIME_DIM: leads,
            },
        )

        effective_input = make_data_config(
            times=times,
            lead_times=leads,
            ensemble_mean=True,
        )

        dataset.config = SimpleNamespace(
            init_time_dim=INIT_TIME_DIM,
            lead_time_dim=LEAD_TIME_DIM,
            realization_dim=REALIZATION_DIM,
            effective_input=effective_input,
        )

        dataset._prepare_sampling_mask(
            {
                INIT_TIME_DIM: times,
                LEAD_TIME_DIM: leads,
            }
        )

        assert dataset.mask.values[0, 0] == 0.0
        assert np.isnan(dataset.mask.values[0, 1])
        dataset = make_dataset_without_init()
        times = make_times()[:2]
        leads = np.array(
            [
                1,
                2,
            ]
        )

        dataset.mask = xr.DataArray(
            np.ones(
                (
                    2,
                    2,
                ),
                dtype=bool,
            ),
            dims=(
                INIT_TIME_DIM,
                LEAD_TIME_DIM,
            ),
            coords={
                INIT_TIME_DIM: times,
                LEAD_TIME_DIM: leads,
            },
        )

        effective_input = make_data_config(
            times=times,
            lead_times=leads,
            realizations=[
                0,
                1,
            ],
            ensemble_mean=False,
        )

        dataset.config = SimpleNamespace(
            init_time_dim=INIT_TIME_DIM,
            lead_time_dim=LEAD_TIME_DIM,
            realization_dim=REALIZATION_DIM,
            effective_input=effective_input,
        )

        dataset._prepare_sampling_mask(
            {
                INIT_TIME_DIM: times,
                LEAD_TIME_DIM: leads,
            }
        )

        assert dataset.mask.dims == (
            REALIZATION_DIM,
            INIT_TIME_DIM,
            LEAD_TIME_DIM,
        )
        assert dataset.mask.sizes[REALIZATION_DIM] == 2

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "has_realizations",
        [
            False,
            True,
        ],
    )
    def test_load_xarray_data(
        self,
        has_realizations,
    ):
        dataset = make_dataset_without_init()

        realizations = (
            [
                0,
                1,
            ]
            if has_realizations
            else None
        )
        config = make_data_config(
            realizations=realizations,
        )
        config.realization_dim = REALIZATION_DIM

        with patch(
            "cccma_ppp.data_modules.dataset.dataset_abc._load_xarray_data",
            return_value=MagicMock(),
        ) as mock_load:
            dataset._load_xarray_data(
                config,
                load=True,
                add_time_auxiliary_coords=True,
            )

        expected_selection = (
            {REALIZATION_DIM: config.info.coords[REALIZATION_DIM]}
            if has_realizations
            else None
        )

        mock_load.assert_called_once_with(
            ["data.nc"],
            names=["var"],
            ensemble_mean=False,
            selection=expected_selection,
            concat_dim=INIT_TIME_DIM,
            rename_dict=None,
            add_time_auxiliary_coords=True,
            load=True,
        )

    @pytest.mark.pruned
    def test_get_sampling_coords(self):
        dataset = make_dataset_without_init()
        times = make_times()[:2]

        dataset.mask = xr.DataArray(
            np.array(
                [
                    [
                        1.0,
                        np.nan,
                    ],
                    [
                        1.0,
                        1.0,
                    ],
                ]
            ),
            dims=(
                INIT_TIME_DIM,
                LEAD_TIME_DIM,
            ),
            coords={
                INIT_TIME_DIM: times,
                LEAD_TIME_DIM: [
                    1,
                    2,
                ],
            },
        )

        result = dataset.get_sampling_coords()

        assert np.array_equal(
            result[INIT_TIME_DIM],
            np.array(
                [
                    times[0],
                    times[1],
                    times[1],
                ]
            ),
        )
        assert np.array_equal(
            result[LEAD_TIME_DIM],
            np.array(
                [
                    1,
                    1,
                    2,
                ]
            ),
        )

    def test_get_model_indexes_when_model_not_loaded(self):
        dataset = make_dataset_without_init()
        dataset.load_model = False

        assert (
            dataset.get_model_indexes(
                {
                    INIT_TIME_DIM: make_times(),
                }
            )
            is None
        )

    def test_get_model_indexes(self):
        dataset = make_dataset_without_init()
        dataset.load_model = True

        times = make_times()
        leads = np.array(
            [
                1,
                2,
                3,
            ]
        )

        dataset.model_dataset = xr.DataArray(
            np.zeros(
                (
                    3,
                    3,
                )
            ),
            dims=(
                INIT_TIME_DIM,
                LEAD_TIME_DIM,
            ),
            coords={
                INIT_TIME_DIM: times,
                LEAD_TIME_DIM: leads,
            },
        )

        result = dataset.get_model_indexes(
            {
                INIT_TIME_DIM: times[
                    [
                        2,
                        0,
                    ]
                ],
                LEAD_TIME_DIM: np.array(
                    [
                        3,
                        1,
                    ]
                ),
            }
        )

        assert np.array_equal(
            result[INIT_TIME_DIM],
            np.array(
                [
                    2,
                    0,
                ]
            ),
        )
        assert np.array_equal(
            result[LEAD_TIME_DIM],
            np.array(
                [
                    2,
                    0,
                ]
            ),
        )

    def test_get_model_indexes_rejects_missing_coordinates(self):
        dataset = make_dataset_without_init()
        dataset.load_model = True

        dataset.model_dataset = xr.DataArray(
            np.zeros(
                (
                    1,
                    1,
                )
            ),
            dims=(
                INIT_TIME_DIM,
                LEAD_TIME_DIM,
            ),
            coords={
                INIT_TIME_DIM: np.array(
                    [
                        "2000-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                LEAD_TIME_DIM: [
                    1,
                ],
            },
        )

        with pytest.raises(
            ValueError,
            match="not found in the model dataset",
        ):
            dataset.get_model_indexes(
                {
                    INIT_TIME_DIM: np.array(
                        [
                            "1990-01-01",
                        ],
                        dtype="datetime64[ns]",
                    ),
                    LEAD_TIME_DIM: np.array([1]),
                }
            )

    @pytest.mark.pruned
    def test_get_cond_indexes_without_condition(self):
        dataset = make_dataset_without_init()
        dataset.condition_dataset = None

        assert dataset.get_cond_indexes({}) is None

    def test_get_cond_indexes_static_condition(self):
        dataset = make_dataset_without_init()
        dataset.condition_dataset = xr.DataArray(
            [
                1.0,
            ],
            dims=("channels",),
        )
        dataset.config = SimpleNamespace(
            condition_method="static",
            realization_dim=REALIZATION_DIM,
        )

        assert dataset.get_cond_indexes({}) is None

    def test_get_cond_indexes_regular(self):
        dataset = make_dataset_without_init()

        times = make_times()
        leads = np.array(
            [
                1,
                2,
                3,
            ]
        )

        dataset.condition_dataset = xr.DataArray(
            np.zeros(
                (
                    3,
                    3,
                )
            ),
            dims=(
                INIT_TIME_DIM,
                LEAD_TIME_DIM,
            ),
            coords={
                INIT_TIME_DIM: times,
                LEAD_TIME_DIM: leads,
            },
        )
        dataset.config = SimpleNamespace(
            condition_method="ensemble_mean",
            realization_dim=REALIZATION_DIM,
        )

        result = dataset.get_cond_indexes(
            {
                INIT_TIME_DIM: times[
                    [
                        1,
                        2,
                    ]
                ],
                LEAD_TIME_DIM: np.array(
                    [
                        2,
                        3,
                    ]
                ),
                "unknown": np.array(
                    [
                        5,
                        6,
                    ]
                ),
            }
        )

        assert set(result) == {
            INIT_TIME_DIM,
            LEAD_TIME_DIM,
        }
        assert np.array_equal(
            result[INIT_TIME_DIM],
            np.array(
                [
                    1,
                    2,
                ]
            ),
        )

    def test_get_cond_indexes_same_member_requires_realizations(self):
        dataset = make_dataset_without_init()
        dataset.condition_dataset = xr.DataArray(
            np.zeros(
                (
                    1,
                    1,
                    1,
                )
            ),
            dims=(
                REALIZATION_DIM,
                INIT_TIME_DIM,
                LEAD_TIME_DIM,
            ),
            coords={
                REALIZATION_DIM: [0],
                INIT_TIME_DIM: np.array(
                    [
                        "2000-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                LEAD_TIME_DIM: [1],
            },
        )
        dataset.config = SimpleNamespace(
            condition_method="same_member",
            realization_dim=REALIZATION_DIM,
        )

        with pytest.raises(
            ValueError,
            match="requires.*coordinates",
        ):
            dataset.get_cond_indexes(
                {
                    INIT_TIME_DIM: np.array(
                        [
                            "2000-01-01",
                        ],
                        dtype="datetime64[ns]",
                    ),
                    LEAD_TIME_DIM: np.array([1]),
                }
            )

    @pytest.mark.pruned
    def test_get_cond_indexes_same_member(self):
        dataset = make_dataset_without_init()

        times = np.array(
            [
                "2000-01-01",
            ],
            dtype="datetime64[ns]",
        )

        dataset.condition_dataset = xr.DataArray(
            np.zeros(
                (
                    2,
                    1,
                    1,
                )
            ),
            dims=(
                REALIZATION_DIM,
                INIT_TIME_DIM,
                LEAD_TIME_DIM,
            ),
            coords={
                REALIZATION_DIM: [
                    0,
                    1,
                ],
                INIT_TIME_DIM: times,
                LEAD_TIME_DIM: [1],
            },
        )
        dataset.config = SimpleNamespace(
            condition_method="same_member",
            realization_dim=REALIZATION_DIM,
        )

        result = dataset.get_cond_indexes(
            {
                REALIZATION_DIM: np.array([1]),
                INIT_TIME_DIM: times,
                LEAD_TIME_DIM: np.array([1]),
            }
        )

        assert np.array_equal(
            result[REALIZATION_DIM],
            np.array([1]),
        )

    @pytest.mark.pruned
    def test_get_cond_indexes_rejects_missing_coordinates(self):
        dataset = make_dataset_without_init()

        dataset.condition_dataset = xr.DataArray(
            np.zeros(
                (
                    1,
                    1,
                )
            ),
            dims=(
                INIT_TIME_DIM,
                LEAD_TIME_DIM,
            ),
            coords={
                INIT_TIME_DIM: np.array(
                    [
                        "2000-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                LEAD_TIME_DIM: [1],
            },
        )
        dataset.config = SimpleNamespace(
            condition_method="ensemble_mean",
            realization_dim=REALIZATION_DIM,
        )

        with pytest.raises(
            ValueError,
            match="conditioning coordinates were not found",
        ):
            dataset.get_cond_indexes(
                {
                    INIT_TIME_DIM: np.array(
                        [
                            "1990-01-01",
                        ],
                        dtype="datetime64[ns]",
                    ),
                    LEAD_TIME_DIM: np.array([1]),
                }
            )

    def test_get_input_shape_without_flattener(self):
        dataset = make_dataset_without_init()

        pipeline = MagicMock()
        pipeline.fitted_preprocessors = []

        effective_input = make_data_config(
            names=[
                "tas",
                "pr",
            ],
            preprocessing_pipeline=pipeline,
        )
        effective_input.info.coords.update(
            {
                "latitude": xr.DataArray(
                    np.arange(3),
                    dims=("latitude",),
                ),
                "longitude": xr.DataArray(
                    np.arange(4),
                    dims=("longitude",),
                ),
            }
        )

        dataset.concat_condition_to_input = False
        dataset.config = SimpleNamespace(
            effective_input=effective_input,
            effective_condition=None,
            supported_NN_dimensions=(
                "latitude",
                "longitude",
            ),
        )

        assert dataset.get_input_shape() == (
            2,
            3,
            4,
        )

    def test_get_input_shape_includes_condition_names(self):
        dataset = make_dataset_without_init()

        pipeline = MagicMock()
        pipeline.fitted_preprocessors = []

        effective_input = make_data_config(
            names=["tas"],
            preprocessing_pipeline=pipeline,
        )
        effective_input.info.coords["channels"] = xr.DataArray(
            np.arange(5),
            dims=("channels",),
        )

        condition = make_data_config(
            names=[
                "pr",
                "psl",
            ]
        )

        dataset.concat_condition_to_input = True
        dataset.config = SimpleNamespace(
            effective_input=effective_input,
            effective_condition=condition,
            supported_NN_dimensions=("channels",),
        )

        assert dataset.get_input_shape() == (
            3,
            5,
        )

    @pytest.mark.pruned
    def test_get_added_features_dim(self):
        dataset = make_dataset_without_init()
        dataset.time_features = AddedTimeFeatures(
            make_reference_config(),
            [
                INIT_TIME_DIM,
                LEAD_TIME_DIM,
                "month_sin",
            ],
        )

        assert dataset.get_added_features_dim() == 3

    def test_index_condition_dataset_none(self):
        dataset = make_dataset_without_init()
        dataset.condition_dataset = None

        assert dataset._index_condition_dataset(0) is None

    def test_index_static_condition_dataset(self):
        dataset = make_dataset_without_init()

        condition = xr.Dataset(
            {
                "var": (
                    ("channels",),
                    [
                        1.0,
                        2.0,
                    ],
                )
            },
            coords={
                "channels": [
                    0,
                    1,
                ],
            },
        )

        pipeline = MagicMock()
        pipeline.transform.side_effect = lambda value: value

        dataset.condition_dataset = condition
        dataset.cond_indexes = None
        dataset.config = SimpleNamespace(
            condition_method="static",
            realization_dim=REALIZATION_DIM,
            effective_condition=SimpleNamespace(
                preprocessing_pipeline=pipeline,
            ),
        )

        with patch(
            "cccma_ppp.data_modules.dataset.dataset_abc._unwrap_data_variables",
            return_value=np.array(
                [
                    1.0,
                    2.0,
                ]
            ),
        ) as mock_unwrap:
            result = dataset._index_condition_dataset(0)

        pipeline.transform.assert_called_once()
        mock_unwrap.assert_called_once()
        assert np.array_equal(
            result,
            np.array(
                [
                    1.0,
                    2.0,
                ]
            ),
        )

    def test_index_cross_ensemble_condition(self):
        dataset = make_dataset_without_init()

        dataset.condition_dataset = xr.Dataset(
            {
                "var": (
                    (
                        REALIZATION_DIM,
                        INIT_TIME_DIM,
                        LEAD_TIME_DIM,
                    ),
                    np.zeros(
                        (
                            2,
                            1,
                            1,
                        )
                    ),
                )
            },
            coords={
                REALIZATION_DIM: [
                    0,
                    1,
                ],
                INIT_TIME_DIM: np.array(
                    [
                        "2000-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                LEAD_TIME_DIM: [1],
            },
        )

        pipeline = MagicMock()
        pipeline.transform.side_effect = lambda value: value

        dataset.cond_indexes = {
            INIT_TIME_DIM: np.array([0]),
            LEAD_TIME_DIM: np.array([0]),
        }
        dataset.config = SimpleNamespace(
            condition_method="cross_ensemble",
            realization_dim=REALIZATION_DIM,
            effective_condition=SimpleNamespace(
                preprocessing_pipeline=pipeline,
            ),
        )

        with (
            patch(
                "cccma_ppp.data_modules.dataset.dataset_abc.np.random.randint",
                return_value=1,
            ) as mock_randint,
            patch(
                "cccma_ppp.data_modules.dataset.dataset_abc._unwrap_data_variables",
                return_value=np.array([0.0]),
            ),
        ):
            dataset._index_condition_dataset(0)

        mock_randint.assert_called_once_with(2)
        pipeline.transform.assert_called_once()

    def test_index_model_dataset_none(self):
        dataset = make_dataset_without_init()
        dataset.load_model = False

        assert dataset._index_model_dataset(0) is None

    def test_index_model_dataset(self):
        dataset = make_dataset_without_init()
        dataset.load_model = True

        dataset.model_dataset = xr.Dataset(
            {
                "var": (
                    (
                        INIT_TIME_DIM,
                        LEAD_TIME_DIM,
                    ),
                    np.array(
                        [
                            [
                                1.0,
                            ]
                        ]
                    ),
                )
            },
            coords={
                INIT_TIME_DIM: np.array(
                    [
                        "2000-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                LEAD_TIME_DIM: [1],
            },
        )
        dataset.model_indexes = {
            INIT_TIME_DIM: np.array([0]),
            LEAD_TIME_DIM: np.array([0]),
        }

        pipeline = MagicMock()
        pipeline.transform.side_effect = lambda value: value

        dataset.config = SimpleNamespace(
            model=SimpleNamespace(
                preprocessing_pipeline=pipeline,
            ),
        )

        with patch(
            "cccma_ppp.data_modules.dataset.dataset_abc._unwrap_data_variables",
            return_value=np.array([1.0]),
        ) as mock_unwrap:
            result = dataset._index_model_dataset(0)

        pipeline.transform.assert_called_once()
        mock_unwrap.assert_called_once()
        assert np.array_equal(
            result,
            np.array([1.0]),
        )

    @pytest.mark.pruned
    def test_compute(self):
        first = np.array(
            [
                1,
                2,
            ]
        )
        second = np.array(
            [
                3,
                4,
            ]
        )

        result = ConcreteDataset._compute(
            first,
            second,
        )

        assert len(result) == 2
        assert np.array_equal(
            result[0],
            first,
        )
        assert np.array_equal(
            result[1],
            second,
        )

    @pytest.mark.pruned
    def test_length(self):
        dataset = make_dataset_without_init()
        dataset.sample_coords = {
            INIT_TIME_DIM: np.array(
                [
                    1,
                    2,
                    3,
                ]
            ),
            LEAD_TIME_DIM: np.array(
                [
                    1,
                    1,
                    2,
                ]
            ),
        }

        assert len(dataset) == 3


class TestDatasetConfigAdditionalBranches:
    @staticmethod
    def make_uninitialized_config(
        *,
        model=None,
        condition=None,
        condition_method=None,
        effective_condition=None,
    ):
        config = ConcreteDatasetConfig.__new__(ConcreteDatasetConfig)
        config.model = model
        config.condition = condition
        config.condition_method = condition_method
        config._effective_condition = effective_condition
        config.lead_times = None
        config._available_times = make_times()
        config._effective_input_override = None
        config._ds_operator = MagicMock()
        config._fitted_preprocessors = False

        return config

    @pytest.mark.pruned
    def test_check_required_input_source_returns_self(self):
        model = make_data_config()
        config = self.make_uninitialized_config(
            model=model,
        )

        assert config._check_required_input_source() is config

    @pytest.mark.pruned
    def test_check_required_input_source_rejects_missing_sources(self):
        config = self.make_uninitialized_config()

        with pytest.raises(
            ValueError,
            match="either model or condition data must be provided",
        ):
            config._check_required_input_source()

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "condition_method",
        [
            "ensemble_mean",
            "cross_ensemble",
            "same_member",
            "static",
            "ENSEMBLE_MEAN",
            "CROSS_ENSEMBLE",
            "SAME_MEMBER",
            "STATIC",
        ],
    )
    def test_check_condition_method_accepts_valid_values(
        self,
        condition_method,
    ):
        config = self.make_uninitialized_config(
            condition_method=condition_method,
        )

        assert config._check_condition_method() is config

    @pytest.mark.pruned
    def test_check_condition_method_accepts_none(self):
        config = self.make_uninitialized_config(
            condition_method=None,
        )

        assert config._check_condition_method() is config

    def test_check_condition_method_rejects_invalid_value(self):
        config = self.make_uninitialized_config(
            condition_method="invalid",
        )

        with pytest.raises(
            ValueError,
            match="Invalid condition_method",
        ):
            config._check_condition_method()

    @pytest.mark.pruned
    def test_resolve_lead_times_converts_config(self):
        config = self.make_uninitialized_config()
        config.lead_times = lead_time_config(
            start=2,
            end=4,
        )

        config._resolve_lead_times()

        assert np.array_equal(
            config.lead_times,
            np.array(
                [
                    2,
                    3,
                    4,
                ]
            ),
        )

    @pytest.mark.pruned
    def test_resolve_lead_times_preserves_array(self):
        lead_times = np.array(
            [
                1,
                3,
                6,
            ]
        )
        config = self.make_uninitialized_config()
        config.lead_times = lead_times

        config._resolve_lead_times()

        assert config.lead_times is lead_times

    @pytest.mark.pruned
    def test_resolve_lead_times_preserves_none(self):
        config = self.make_uninitialized_config()
        config.lead_times = None

        config._resolve_lead_times()

        assert config.lead_times is None

    @pytest.mark.pruned
    def test_resolve_condition_uses_explicit_condition(self):
        condition = make_data_config()
        config = self.make_uninitialized_config(
            condition=condition,
            condition_method="ensemble_mean",
        )

        assert config._resolve_condition() is config
        assert config.effective_condition is condition

    @pytest.mark.pruned
    def test_resolve_condition_without_condition(self):
        model = make_data_config()
        config = self.make_uninitialized_config(
            model=model,
            condition=None,
            condition_method=None,
        )

        config._resolve_condition()

        assert config.effective_condition is None

    @pytest.mark.pruned
    def test_resolve_condition_builds_model_condition(self):
        model = make_data_config(
            ensemble_mean=False,
        )
        generated_condition = make_data_config(
            ensemble_mean=True,
        )

        config = self.make_uninitialized_config(
            model=model,
            condition=None,
            condition_method="ensemble_mean",
        )

        config._model_as_condition = MagicMock(return_value=generated_condition)

        config._resolve_condition()

        config._model_as_condition.assert_called_once_with()
        assert config.effective_condition is generated_condition

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "condition_method",
        [
            "ensemble_mean",
            "cross_ensemble",
            "same_member",
        ],
    )
    def test_using_model_as_condition_for_supported_methods(
        self,
        condition_method,
    ):
        config = self.make_uninitialized_config(
            model=make_data_config(),
            condition=None,
            condition_method=condition_method,
        )

        assert config._using_model_data_as_condition is True

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "condition_method",
        [
            "static",
            "other",
        ],
    )
    def test_not_using_model_as_condition_for_other_methods(
        self,
        condition_method,
    ):
        config = self.make_uninitialized_config(
            model=make_data_config(),
            condition=None,
            condition_method=condition_method,
        )

        assert config._using_model_data_as_condition is False

    @pytest.mark.pruned
    def test_using_model_as_condition_same_source(self):
        model = make_data_config(
            paths="same",
            names=["tas"],
            realization_list=[0],
        )
        condition = make_data_config(
            paths="same",
            names=["tas"],
            realization_list=[0],
        )

        config = self.make_uninitialized_config(
            model=model,
            condition=condition,
            condition_method="same_member",
        )

        assert config._using_model_data_as_condition is True

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "condition_paths,condition_names,condition_members",
        [
            (
                "different",
                ["tas"],
                [0],
            ),
            (
                "same",
                ["pr"],
                [0],
            ),
            (
                "same",
                ["tas"],
                [1],
            ),
        ],
    )
    def test_not_using_model_as_condition_different_source(
        self,
        condition_paths,
        condition_names,
        condition_members,
    ):
        model = make_data_config(
            paths="same",
            names=["tas"],
            realization_list=[0],
        )
        condition = make_data_config(
            paths=condition_paths,
            names=condition_names,
            realization_list=condition_members,
        )

        config = self.make_uninitialized_config(
            model=model,
            condition=condition,
            condition_method="same_member",
        )

        assert config._using_model_data_as_condition is False

    @pytest.mark.pruned
    def test_using_model_as_condition_without_model_is_false(self):
        condition = make_data_config()
        config = self.make_uninitialized_config(
            model=None,
            condition=condition,
            condition_method="ensemble_mean",
        )

        assert config._using_model_data_as_condition is False

    @pytest.mark.pruned
    def test_check_model_without_model(self):
        config = self.make_uninitialized_config(
            model=None,
            condition_method="same_member",
        )

        assert config._check_model() is config

    @pytest.mark.pruned
    def test_check_model_accepts_non_same_member(self):
        model = make_data_config(
            ensemble_mean=True,
        )
        config = self.make_uninitialized_config(
            model=model,
            condition_method="ensemble_mean",
        )

        assert config._check_model() is config

    @pytest.mark.pruned
    def test_check_model_same_member_accepts_members(self):
        model = make_data_config(
            ensemble_mean=False,
        )
        config = self.make_uninitialized_config(
            model=model,
            condition_method="same_member",
        )

        assert config._check_model() is config

    def test_check_model_same_member_rejects_ensemble_mean(self):
        model = make_data_config(
            ensemble_mean=True,
        )
        config = self.make_uninitialized_config(
            model=model,
            condition_method="same_member",
        )

        with pytest.raises(
            ValueError,
            match="should not be ensemble mean",
        ):
            config._check_model()

    def test_check_condition_requires_method(self):
        condition = make_data_config()
        config = self.make_uninitialized_config(
            condition=condition,
            condition_method=None,
            effective_condition=condition,
        )

        with pytest.raises(
            ValueError,
            match="specify condition_method",
        ):
            DatasetConfigABC._check_condition(config)

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "condition_method",
        [
            "cross_ensemble",
            "same_member",
        ],
    )
    def test_check_condition_member_method_rejects_ensemble_mean(
        self,
        condition_method,
    ):
        condition = make_data_config(
            realizations=[
                0,
                1,
            ],
            ensemble_mean=True,
        )
        config = self.make_uninitialized_config(
            condition=condition,
            condition_method=condition_method,
            effective_condition=condition,
        )

        with pytest.raises(
            ValueError,
            match="ensemble_mean cannot be True",
        ):
            config._check_condition()

    @pytest.mark.parametrize(
        "condition_method",
        [
            "cross_ensemble",
            "same_member",
        ],
    )
    def test_check_condition_member_method_requires_realizations(
        self,
        condition_method,
    ):
        condition = make_data_config(
            realizations=None,
            ensemble_mean=False,
        )
        config = self.make_uninitialized_config(
            condition=condition,
            condition_method=condition_method,
            effective_condition=condition,
        )

        with pytest.raises(
            ValueError,
            match="dim must exist",
        ):
            config._check_condition()

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "condition_method",
        [
            "cross_ensemble",
            "same_member",
        ],
    )
    def test_check_condition_member_method_valid(
        self,
        condition_method,
    ):
        condition = make_data_config(
            realizations=[
                0,
                1,
            ],
            ensemble_mean=False,
        )
        config = self.make_uninitialized_config(
            condition=condition,
            condition_method=condition_method,
            effective_condition=condition,
        )

        assert config._check_condition() is config

    def test_check_condition_ensemble_mean_requires_true(self):
        condition = make_data_config(
            ensemble_mean=False,
        )
        config = self.make_uninitialized_config(
            condition=condition,
            condition_method="ensemble_mean",
            effective_condition=condition,
        )

        with pytest.raises(
            ValueError,
            match="Ensemble mean must be True",
        ):
            config._check_condition()

    def test_check_condition_ensemble_mean_valid(self):
        condition = make_data_config(
            ensemble_mean=True,
        )
        config = self.make_uninitialized_config(
            condition=condition,
            condition_method="ensemble_mean",
            effective_condition=condition,
        )

        assert config._check_condition() is config

    def test_check_static_condition_rejects_realization_list(self):
        condition = make_data_config(
            realization_list=[0],
        )
        condition.info.coords = {
            "channels": xr.DataArray(
                [
                    0,
                    1,
                ],
                dims=("channels",),
            )
        }

        config = self.make_uninitialized_config(
            condition=condition,
            condition_method="static",
            effective_condition=condition,
        )

        with pytest.raises(
            ValueError,
            match="cannot specify realization list",
        ):
            config._check_condition()

    def test_check_static_condition_rejects_model_source(self):
        model = make_data_config(
            paths="same",
            names=["tas"],
            realization_list=None,
        )
        condition = make_data_config(
            paths="same",
            names=["tas"],
            realization_list=None,
        )
        condition.info.coords = {
            "channels": xr.DataArray(
                [
                    0,
                    1,
                ],
                dims=("channels",),
            )
        }

        config = self.make_uninitialized_config(
            model=model,
            condition=condition,
            condition_method="static",
            effective_condition=condition,
        )

        with pytest.raises(
            ValueError,
            match="cannot point to the same model data",
        ):
            config._check_condition()

    @pytest.mark.parametrize(
        "sampling_dimension",
        [
            INIT_TIME_DIM,
            LEAD_TIME_DIM,
            REALIZATION_DIM,
        ],
    )
    def test_static_condition_rejects_sampling_dimension(
        self,
        sampling_dimension,
    ):
        condition = make_data_config(
            realization_list=None,
        )
        condition.info.coords = {
            sampling_dimension: xr.DataArray(
                [
                    0,
                    1,
                ],
                dims=(sampling_dimension,),
            )
        }

        config = self.make_uninitialized_config(
            condition=condition,
            condition_method="static",
            effective_condition=condition,
        )

        with pytest.raises(
            ValueError,
            match="cannot have",
        ):
            config._check_condition()

    @pytest.mark.pruned
    def test_static_condition_without_sampling_dimensions_valid(self):
        condition = make_data_config(
            realization_list=None,
        )
        condition.info.coords = {
            "channels": xr.DataArray(
                [
                    0,
                    1,
                ],
                dims=("channels",),
            )
        }

        config = self.make_uninitialized_config(
            condition=condition,
            condition_method="static",
            effective_condition=condition,
        )

        assert config._check_condition() is config

    def test_static_method_requires_condition(self):
        config = self.make_uninitialized_config(
            condition=None,
            condition_method="static",
            effective_condition=None,
        )

        with pytest.raises(
            ValueError,
            match="condition dataset must be specified",
        ):
            config._check_condition()

    def test_no_effective_condition_non_static_valid(self):
        config = self.make_uninitialized_config(
            condition=None,
            condition_method="ensemble_mean",
            effective_condition=None,
        )

        assert config._check_condition() is config

    def test_model_condition_rejects_missing_time_coordinate(self):
        model = make_data_config()
        condition = make_data_config(
            paths="condition",
            ensemble_mean=True,
        )
        condition.info.coords.pop(INIT_TIME_DIM)

        config = self.make_uninitialized_config(
            model=model,
            condition=condition,
            condition_method="ensemble_mean",
            effective_condition=condition,
        )

        with pytest.raises(
            ValueError,
            match="same.*dimestions",
        ):
            config._check_model_vs_condition()

    @pytest.mark.pruned
    def test_model_condition_rejects_missing_lead_coordinate(self):
        model = make_data_config()
        condition = make_data_config(
            paths="condition",
            ensemble_mean=True,
        )
        condition.info.coords.pop(LEAD_TIME_DIM)

        config = self.make_uninitialized_config(
            model=model,
            condition=condition,
            condition_method="ensemble_mean",
            effective_condition=condition,
        )

        with pytest.raises(
            ValueError,
            match="same.*dimestions",
        ):
            config._check_model_vs_condition()

    def test_model_condition_rejects_time_coordinate_mismatch(self):
        model = make_data_config(
            times=np.array(
                [
                    "2000-01-01",
                    "2000-02-01",
                ],
                dtype="datetime64[ns]",
            )
        )
        condition = make_data_config(
            paths="condition",
            times=np.array(
                [
                    "2000-01-01",
                ],
                dtype="datetime64[ns]",
            ),
            ensemble_mean=True,
        )

        config = self.make_uninitialized_config(
            model=model,
            condition=condition,
            condition_method="ensemble_mean",
            effective_condition=condition,
        )

        with pytest.raises(
            ValueError,
            match="same.*coordinates",
        ):
            config._check_model_vs_condition()

    @pytest.mark.pruned
    def test_model_condition_rejects_lead_coordinate_mismatch(self):
        model = make_data_config(
            lead_times=[
                1,
                2,
                3,
            ]
        )
        condition = make_data_config(
            paths="condition",
            lead_times=[
                1,
                2,
            ],
            ensemble_mean=True,
        )

        config = self.make_uninitialized_config(
            model=model,
            condition=condition,
            condition_method="ensemble_mean",
            effective_condition=condition,
        )

        with pytest.raises(
            ValueError,
            match="same.*coordinates",
        ):
            config._check_model_vs_condition()

    @pytest.mark.pruned
    def test_model_condition_accepts_coordinate_superset(self):
        model = make_data_config(
            times=np.array(
                [
                    "2000-01-01",
                    "2000-02-01",
                ],
                dtype="datetime64[ns]",
            ),
            lead_times=[
                1,
                2,
            ],
        )
        condition = make_data_config(
            paths="condition",
            times=np.array(
                [
                    "1999-12-01",
                    "2000-01-01",
                    "2000-02-01",
                    "2000-03-01",
                ],
                dtype="datetime64[ns]",
            ),
            lead_times=[
                1,
                2,
                3,
            ],
            ensemble_mean=True,
        )

        config = self.make_uninitialized_config(
            model=model,
            condition=condition,
            condition_method="ensemble_mean",
            effective_condition=condition,
        )

        config._check_model_vs_condition()

    def test_model_condition_rejects_time_representation_mismatch(self):
        model = make_data_config()
        condition = make_data_config(
            paths="condition",
            ensemble_mean=True,
        )

        model.info.time_coords_type = "datetime"
        condition.info.time_coords_type = "cftime"

        config = self.make_uninitialized_config(
            model=model,
            condition=condition,
            condition_method="ensemble_mean",
            effective_condition=condition,
        )

        with pytest.raises(
            ValueError,
            match="same cftime/datetime type",
        ):
            config._check_model_vs_condition()

    @pytest.mark.pruned
    def test_model_condition_same_member_requires_model_realizations(self):
        model = make_data_config(
            realizations=None,
            ensemble_mean=False,
        )
        condition = make_data_config(
            paths="condition",
            realizations=[
                0,
                1,
            ],
            ensemble_mean=False,
        )

        config = self.make_uninitialized_config(
            model=model,
            condition=condition,
            condition_method="same_member",
            effective_condition=condition,
        )

        with pytest.raises(
            ValueError,
            match="dims and coords",
        ):
            config._check_model_vs_condition()

    def test_model_condition_same_member_requires_condition_realizations(self):
        model = make_data_config(
            realizations=[
                0,
                1,
            ],
            ensemble_mean=False,
        )
        condition = make_data_config(
            paths="condition",
            realizations=None,
            ensemble_mean=False,
        )

        config = self.make_uninitialized_config(
            model=model,
            condition=condition,
            condition_method="same_member",
            effective_condition=condition,
        )

        with pytest.raises(
            ValueError,
            match="dims and coords",
        ):
            config._check_model_vs_condition()

    def test_model_condition_same_member_rejects_member_mismatch(self):
        model = make_data_config(
            realizations=[
                0,
                1,
            ],
            ensemble_mean=False,
        )
        condition = make_data_config(
            paths="condition",
            realizations=[
                0,
                2,
            ],
            ensemble_mean=False,
        )

        config = self.make_uninitialized_config(
            model=model,
            condition=condition,
            condition_method="same_member",
            effective_condition=condition,
        )

        with pytest.raises(
            ValueError,
            match="same ensemble members",
        ):
            config._check_model_vs_condition()

    def test_model_condition_same_member_accepts_equal_members(self):
        model = make_data_config(
            realizations=[
                0,
                1,
            ],
            ensemble_mean=False,
        )
        condition = make_data_config(
            paths="condition",
            realizations=[
                0,
                1,
            ],
            ensemble_mean=False,
        )

        config = self.make_uninitialized_config(
            model=model,
            condition=condition,
            condition_method="same_member",
            effective_condition=condition,
        )

        config._check_model_vs_condition()

    @pytest.mark.pruned
    def test_model_condition_skips_comparison_for_same_source(self):
        model = make_data_config(
            paths="same",
            names=["tas"],
            realization_list=[0],
        )
        condition = make_data_config(
            paths="same",
            names=["tas"],
            realization_list=[0],
        )

        condition.info.coords = {}

        config = self.make_uninitialized_config(
            model=model,
            condition=condition,
            condition_method="ensemble_mean",
            effective_condition=condition,
        )

        config._check_model_vs_condition()

    @pytest.mark.pruned
    def test_model_condition_skips_without_model(self):
        condition = make_data_config(
            ensemble_mean=True,
        )

        config = self.make_uninitialized_config(
            model=None,
            condition=condition,
            condition_method="ensemble_mean",
            effective_condition=condition,
        )

        config._check_model_vs_condition()

    @pytest.mark.pruned
    def test_model_condition_skips_without_condition(self):
        model = make_data_config()

        config = self.make_uninitialized_config(
            model=model,
            condition=None,
            condition_method=None,
            effective_condition=None,
        )

        config._check_model_vs_condition()

    def test_get_input_times_returns_intersection(self):
        model_times = np.array(
            [
                "2000-01-01",
                "2000-03-01",
            ],
            dtype="datetime64[ns]",
        )
        model = make_data_config(
            times=model_times,
        )

        config = self.make_uninitialized_config(
            model=model,
        )
        config._available_times = np.array(
            [
                "2000-01-01",
                "2000-02-01",
                "2000-03-01",
            ],
            dtype="datetime64[ns]",
        )

        requested = xr.DataArray(
            config._available_times,
            dims=(INIT_TIME_DIM,),
            coords={
                INIT_TIME_DIM: config._available_times,
            },
        )

        result = config.get_input_times(requested)

        assert np.array_equal(
            result.values,
            model_times,
        )

    def test_get_input_times_rejects_unavailable_times(self):
        model = make_data_config()
        config = self.make_uninitialized_config(
            model=model,
        )
        config._available_times = make_times()

        unavailable = np.array(
            [
                "1990-01-01",
            ],
            dtype="datetime64[ns]",
        )
        requested = xr.DataArray(
            unavailable,
            dims=(INIT_TIME_DIM,),
            coords={
                INIT_TIME_DIM: unavailable,
            },
        )

        with pytest.raises(
            ValueError,
            match="requested_times are unavailable",
        ):
            config.get_input_times(requested)


class TestAddedTimeFeaturesAdditionalBranches:
    def test_build_all_features_together(self):
        reference = make_reference_config(
            lead_times=np.array(
                [
                    1,
                    2,
                    3,
                    12,
                ]
            ),
            common_times=np.array(
                [
                    "2000-01-01",
                    "2001-01-01",
                ],
                dtype="datetime64[ns]",
            ),
        )

        added = AddedTimeFeatures(
            reference,
            [
                INIT_TIME_DIM,
                LEAD_TIME_DIM,
                "month_sin",
                "month_cos",
                "day_sin",
                "day_cos",
            ],
        )

        init_times = np.array(
            [
                "2000-01-01",
                "2000-06-15",
            ],
            dtype="datetime64[ns]",
        )
        lead_times = np.array(
            [
                1,
                2,
            ]
        )

        added.build_time_features(
            {
                INIT_TIME_DIM: init_times,
                LEAD_TIME_DIM: lead_times,
            }
        )

        assert added.time_features_array.shape == (
            2,
            6,
        )
        assert added.time_features_array.dtype == np.float32
        assert np.isfinite(added.time_features_array).all()

    @pytest.mark.pruned
    def test_build_preserves_feature_count_with_duplicate_request(self):
        added = AddedTimeFeatures(
            make_reference_config(),
            [
                "month_sin",
                "month_sin",
            ],
        )

        assert added.time_features == ("month_sin",)

    @pytest.mark.pruned
    def test_call_two_dimensional_input_does_not_broadcast(self):
        added = AddedTimeFeatures(
            make_reference_config(),
            [
                INIT_TIME_DIM,
                LEAD_TIME_DIM,
            ],
        )
        added.time_features_array = np.array(
            [
                [
                    0.25,
                    0.5,
                ],
            ],
            dtype=np.float32,
        )

        result = added(
            0,
            make_feature_input(
                shape=(
                    2,
                    3,
                )
            ),
        )

        assert result.shape == (2,)

    @pytest.mark.pruned
    def test_call_four_dimensional_input_broadcasts(self):
        added = AddedTimeFeatures(
            make_reference_config(),
            [
                INIT_TIME_DIM,
                LEAD_TIME_DIM,
            ],
        )
        added.time_features_array = np.array(
            [
                [
                    0.25,
                    0.5,
                ],
            ],
            dtype=np.float32,
        )

        result = added(
            0,
            make_feature_input(
                shape=(
                    1,
                    2,
                    3,
                    4,
                )
            ),
        )

        assert result.shape == (
            2,
            2,
            3,
            4,
        )

    @pytest.mark.pruned
    def test_not_equal_lead_times(self):
        left = AddedTimeFeatures(
            make_reference_config(
                lead_times=np.array(
                    [
                        1,
                        2,
                    ]
                )
            ),
            [
                INIT_TIME_DIM,
            ],
        )
        right = AddedTimeFeatures(
            make_reference_config(
                lead_times=np.array(
                    [
                        1,
                        3,
                    ]
                )
            ),
            [
                INIT_TIME_DIM,
            ],
        )

        assert left != right

    @pytest.mark.pruned
    def test_not_equal_common_times(self):
        left = AddedTimeFeatures(
            make_reference_config(
                common_times=np.array(
                    [
                        "2000-01-01",
                        "2001-01-01",
                    ],
                    dtype="datetime64[ns]",
                )
            ),
            [
                INIT_TIME_DIM,
            ],
        )
        right = AddedTimeFeatures(
            make_reference_config(
                common_times=np.array(
                    [
                        "2000-01-01",
                        "2002-01-01",
                    ],
                    dtype="datetime64[ns]",
                )
            ),
            [
                INIT_TIME_DIM,
            ],
        )

        assert left != right

    @pytest.mark.pruned
    def test_not_equal_reference_types(self):
        class OtherReference:
            lead_time_resolution = "month"
            init_time_dim = INIT_TIME_DIM
            lead_time_dim = LEAD_TIME_DIM
            lead_times = np.array(
                [
                    1,
                    2,
                ]
            )
            get_common_time = np.array(
                [
                    "2000-01-01",
                    "2001-01-01",
                ],
                dtype="datetime64[ns]",
            )

        left = AddedTimeFeatures(
            make_reference_config(
                lead_times=np.array(
                    [
                        1,
                        2,
                    ]
                )
            ),
            [
                INIT_TIME_DIM,
            ],
        )
        right = AddedTimeFeatures(
            OtherReference(),
            [
                INIT_TIME_DIM,
            ],
        )

        assert left != right


class TestDatasetABCAdditionalBranches:
    @pytest.mark.pruned
    def test_prepare_sampling_mask_ignores_realizations_for_ensemble_mean(
        self,
    ):
        dataset = make_dataset_without_init()

        times = make_times()[:2]
        leads = np.array(
            [
                1,
                2,
            ]
        )

        dataset.mask = xr.DataArray(
            np.ones(
                (
                    2,
                    2,
                ),
                dtype=bool,
            ),
            dims=(
                INIT_TIME_DIM,
                LEAD_TIME_DIM,
            ),
            coords={
                INIT_TIME_DIM: times,
                LEAD_TIME_DIM: leads,
            },
        )

        effective_input = make_data_config(
            times=times,
            lead_times=leads,
            realizations=[
                0,
                1,
            ],
            ensemble_mean=True,
        )

        dataset.config = SimpleNamespace(
            init_time_dim=INIT_TIME_DIM,
            lead_time_dim=LEAD_TIME_DIM,
            realization_dim=REALIZATION_DIM,
            effective_input=effective_input,
        )

        dataset._prepare_sampling_mask(
            {
                INIT_TIME_DIM: times,
                LEAD_TIME_DIM: leads,
            }
        )

        assert REALIZATION_DIM not in dataset.mask.dims

    @pytest.mark.pruned
    def test_get_sampling_coords_with_realization_dimension(self):
        dataset = make_dataset_without_init()

        dataset.mask = xr.DataArray(
            np.ones(
                (
                    2,
                    1,
                    1,
                )
            ),
            dims=(
                REALIZATION_DIM,
                INIT_TIME_DIM,
                LEAD_TIME_DIM,
            ),
            coords={
                REALIZATION_DIM: [
                    0,
                    1,
                ],
                INIT_TIME_DIM: np.array(
                    [
                        "2000-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                LEAD_TIME_DIM: [1],
            },
        )

        result = dataset.get_sampling_coords()

        assert np.array_equal(
            result[REALIZATION_DIM],
            np.array(
                [
                    0,
                    1,
                ]
            ),
        )

    @pytest.mark.pruned
    def test_get_cond_indexes_cross_ensemble_ignores_members(self):
        dataset = make_dataset_without_init()

        times = np.array(
            [
                "2000-01-01",
            ],
            dtype="datetime64[ns]",
        )

        dataset.condition_dataset = xr.DataArray(
            np.zeros(
                (
                    2,
                    1,
                    1,
                )
            ),
            dims=(
                REALIZATION_DIM,
                INIT_TIME_DIM,
                LEAD_TIME_DIM,
            ),
            coords={
                REALIZATION_DIM: [
                    0,
                    1,
                ],
                INIT_TIME_DIM: times,
                LEAD_TIME_DIM: [1],
            },
        )
        dataset.config = SimpleNamespace(
            condition_method="cross_ensemble",
            realization_dim=REALIZATION_DIM,
        )

        result = dataset.get_cond_indexes(
            {
                REALIZATION_DIM: np.array([1]),
                INIT_TIME_DIM: times,
                LEAD_TIME_DIM: np.array([1]),
            }
        )

        assert REALIZATION_DIM not in result

    def test_get_cond_indexes_same_member_reports_missing_member(self):
        dataset = make_dataset_without_init()

        times = np.array(
            [
                "2000-01-01",
            ],
            dtype="datetime64[ns]",
        )

        dataset.condition_dataset = xr.DataArray(
            np.zeros(
                (
                    2,
                    1,
                    1,
                )
            ),
            dims=(
                REALIZATION_DIM,
                INIT_TIME_DIM,
                LEAD_TIME_DIM,
            ),
            coords={
                REALIZATION_DIM: [
                    0,
                    1,
                ],
                INIT_TIME_DIM: times,
                LEAD_TIME_DIM: [1],
            },
        )
        dataset.config = SimpleNamespace(
            condition_method="same_member",
            realization_dim=REALIZATION_DIM,
        )

        with pytest.raises(
            ValueError,
            match="conditioning coordinates were not found",
        ):
            dataset.get_cond_indexes(
                {
                    REALIZATION_DIM: np.array([5]),
                    INIT_TIME_DIM: times,
                    LEAD_TIME_DIM: np.array([1]),
                }
            )

    @pytest.mark.pruned
    def test_get_input_shape_without_supported_dimensions(self):
        dataset = make_dataset_without_init()

        pipeline = MagicMock()
        pipeline.fitted_preprocessors = []

        effective_input = make_data_config(
            names=[
                "tas",
                "pr",
            ],
            preprocessing_pipeline=pipeline,
        )
        effective_input.info.coords = {
            INIT_TIME_DIM: make_time_coord(),
            LEAD_TIME_DIM: make_lead_coord(),
        }

        dataset.concat_condition_to_input = False
        dataset.config = SimpleNamespace(
            effective_input=effective_input,
            effective_condition=None,
            supported_NN_dimensions=(
                "latitude",
                "longitude",
            ),
        )

        assert dataset.get_input_shape() == (2,)

    def test_index_condition_regular_selection(self):
        dataset = make_dataset_without_init()

        times = np.array(
            [
                "2000-01-01",
                "2000-02-01",
            ],
            dtype="datetime64[ns]",
        )

        dataset.condition_dataset = xr.Dataset(
            {
                "var": (
                    (
                        INIT_TIME_DIM,
                        LEAD_TIME_DIM,
                    ),
                    np.array(
                        [
                            [
                                1.0,
                                2.0,
                            ],
                            [
                                3.0,
                                4.0,
                            ],
                        ]
                    ),
                )
            },
            coords={
                INIT_TIME_DIM: times,
                LEAD_TIME_DIM: [
                    1,
                    2,
                ],
            },
        )

        pipeline = MagicMock()
        pipeline.transform.side_effect = lambda value: value

        dataset.cond_indexes = {
            INIT_TIME_DIM: np.array([1]),
            LEAD_TIME_DIM: np.array([0]),
        }
        dataset.config = SimpleNamespace(
            condition_method="ensemble_mean",
            realization_dim=REALIZATION_DIM,
            effective_condition=SimpleNamespace(
                preprocessing_pipeline=pipeline,
            ),
        )

        with patch(
            "cccma_ppp.data_modules.dataset.dataset_abc._unwrap_data_variables",
            return_value=np.array([3.0]),
        ):
            result = dataset._index_condition_dataset(0)

        assert np.array_equal(
            result,
            np.array([3.0]),
        )
        pipeline.transform.assert_called_once()

    @pytest.mark.pruned
    def test_index_cross_ensemble_uses_member_upper_bound(self):
        dataset = make_dataset_without_init()

        dataset.condition_dataset = xr.Dataset(
            {
                "var": (
                    (
                        REALIZATION_DIM,
                        INIT_TIME_DIM,
                        LEAD_TIME_DIM,
                    ),
                    np.zeros(
                        (
                            3,
                            1,
                            1,
                        )
                    ),
                )
            },
            coords={
                REALIZATION_DIM: [
                    0,
                    1,
                    2,
                ],
                INIT_TIME_DIM: np.array(
                    [
                        "2000-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                LEAD_TIME_DIM: [1],
            },
        )

        pipeline = MagicMock()
        pipeline.transform.side_effect = lambda value: value

        dataset.cond_indexes = {
            INIT_TIME_DIM: np.array([0]),
            LEAD_TIME_DIM: np.array([0]),
        }
        dataset.config = SimpleNamespace(
            condition_method="cross_ensemble",
            realization_dim=REALIZATION_DIM,
            effective_condition=SimpleNamespace(
                preprocessing_pipeline=pipeline,
            ),
        )

        with (
            patch(
                "cccma_ppp.data_modules.dataset.dataset_abc.np.random.randint",
                return_value=2,
            ) as mock_random,
            patch(
                "cccma_ppp.data_modules.dataset.dataset_abc._unwrap_data_variables",
                return_value=np.array([0.0]),
            ),
        ):
            dataset._index_condition_dataset(0)

        mock_random.assert_called_once_with(3)

    @pytest.mark.pruned
    def test_get_added_features_dim_empty(self):
        dataset = make_dataset_without_init()
        dataset.time_features = AddedTimeFeatures(
            make_reference_config(),
            None,
        )

        assert dataset.get_added_features_dim() == 0


class ExtraDatasetConfig(DatasetConfigABC):
    def __init__(
        self,
        *,
        model=None,
        condition=None,
        condition_method="static",
        lead_times=None,
        available_times=None,
    ):
        self.model = model
        self.condition = condition
        self.condition_method = condition_method
        self.lead_times = lead_times
        self._available_times = available_times

    @property
    def available_times(self):
        return self._available_times

    @property
    def ds_operator(self):
        return None

    @property
    def effective_input(self):
        return self.model or self.condition

    def build_dataset(self):
        return None


class ExtraDataset(DatasetABC):
    def __init__(
        self,
        *,
        config=None,
        load_model=True,
        write_condition=False,
        concat_condition=False,
    ):
        self.config = config
        self._test_load_model = load_model
        self._test_write_condition = write_condition
        self._test_concat_condition = concat_condition

    @property
    def _load_model(self):
        return self._test_load_model

    @property
    def _write_condition_to_input(self):
        return self._test_write_condition

    @property
    def _concat_condition_to_input(self):
        return self._test_concat_condition

    def __getitem__(self, index):
        raise NotImplementedError


def make_extra_time_config(
    *,
    lead_times=(1, 2, 3),
    common_times=None,
    config_type=SimpleNamespace,
):
    if common_times is None:
        common_times = pd.DatetimeIndex(
            [
                "2000-01-01",
                "2000-02-01",
                "2000-03-01",
            ]
        )

    return


class BranchDatasetConfig(DatasetConfigABC):
    @property
    def available_times(self):
        return self._available_times

    @property
    def ds_operator(self):
        return None

    @property
    def effective_input(self):
        return self.model or self.condition

    def build_dataset(self):
        return None


class BranchDataset(DatasetABC):
    @property
    def _load_model(self):
        return self.load_model

    @property
    def _write_condition_to_input(self):
        return False

    @property
    def _concat_condition_to_input(self):
        return self.concat_condition

    def __getitem__(self, index):
        raise NotImplementedError


def make_branch_dataset(
    *,
    load_model=True,
    concat_condition=False,
):
    dataset = object.__new__(BranchDataset)
    dataset.load_model = load_model


from types import SimpleNamespace
from unittest.mock import Mock


import cccma_ppp.data_modules.dataset.dataset_abc as dataset_abc_module


class ExtraDatasetConfig(DatasetConfigABC):
    @property
    def available_times(self):
        return self._available_times

    @property
    def ds_operator(self):
        return None

    @property
    def effective_input(self):
        return self.model or self.condition

    def build_dataset(self):
        return None


class ExtraDataset(DatasetABC):
    @property
    def _load_model(self):
        return self.load_model

    @property
    def _write_condition_to_input(self):
        return False

    @property
    def _concat_condition_to_input(self):
        return self.concat_condition

    def __getitem__(self, index):
        raise NotImplementedError


def make_extra_dataset(
    *,
    load_model=True,
    concat_condition=False,
):
    dataset = object.__new__(ExtraDataset)
    dataset.load_model = load_model
    dataset.concat_condition = concat_condition
    dataset.model_dataset = None
    dataset.condition_dataset = None
    dataset.observation_dataset = None
    dataset.model_indexes = None
    dataset.cond_indexes = None
    dataset.sample_coords = {}

    return dataset


def make_extra_pipeline():
    return SimpleNamespace(
        fitted_preprocessors=[],
        transform=Mock(side_effect=lambda value: value),
        get_preprocessors=Mock(),
    )


def make_extra_data_config(
    *,
    paths=None,
    names=None,
    realization_list=None,
    ensemble_mean=False,
    coords=None,
    time_coords_type="datetime",
):
    paths = paths or ["data.nc"]
    names = names or ["tas"]
    coords = coords or {}

    return SimpleNamespace(
        paths=paths,
        list_paths=paths,
        names=names,
        realization_list=realization_list,
        ensemble_mean=ensemble_mean,
        info=SimpleNamespace(
            coords=coords,
            time_coords_type=time_coords_type,
        ),
        preprocessing_pipeline=make_extra_pipeline(),
        concat_dim=None,
        file_type="netcdf",
        rename_dict=None,
        realization_dim=realization_dim,
    )


def make_extra_time_feature_config(
    *,
    lead_times=(1, 2, 4),
    common_times=None,
):
    if common_times is None:
        common_times = pd.DatetimeIndex(
            [
                "2000-01-01",
                "2000-12-31",
            ]
        )

    return SimpleNamespace(
        init_time_dim=init_time_dim,
        lead_time_dim=lead_time_dim,
        lead_time_resolution="month",
        lead_times=np.asarray(lead_times),
        get_common_time=common_times,
    )


class TestExtraLeadTimeConfig:
    @pytest.mark.pruned
    def test_single_value_range(self):
        config = lead_time_config(
            start=3,
            end=3,
        )

        np.testing.assert_array_equal(
            config.build_lead_times(),
            [3],
        )

    @pytest.mark.pruned
    def test_explicit_list_overrides_range(self):
        config = lead_time_config(
            list_lead_times=[1, 6],
            start=2,
            end=4,
        )

        assert config.build_lead_times() == [
            1,
            6,
        ]

    def test_missing_list_and_end_raises(self):
        with pytest.raises(
            ValueError,
            match="Provide a list of lead_times",
        ):
            lead_time_config()

    def test_empty_list_uses_range(self):
        config = lead_time_config(
            list_lead_times=[],
            start=2,
            end=4,
        )

        np.testing.assert_array_equal(
            config.build_lead_times(),
            [2, 3, 4],
        )


class TestExtraDatasetConfig:
    @pytest.mark.pruned
    def test_required_input_source_accepts_model(self):
        config = object.__new__(ExtraDatasetConfig)
        config.model = object()
        config.condition = None

        assert config._check_required_input_source() is config

    @pytest.mark.pruned
    def test_required_input_source_accepts_condition(self):
        config = object.__new__(ExtraDatasetConfig)
        config.model = None
        config.condition = object()

        assert config._check_required_input_source() is config

    @pytest.mark.pruned
    def test_required_input_source_rejects_missing_sources(self):
        config = object.__new__(ExtraDatasetConfig)
        config.model = None
        config.condition = None

        with pytest.raises(
            ValueError,
            match="either model or condition",
        ):
            config._check_required_input_source()

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "method",
        [
            None,
            "ensemble_mean",
            "cross_ensemble",
            "same_member",
            "static",
            "STATIC",
        ],
    )
    def test_condition_method_accepts_supported_values(
        self,
        method,
    ):
        config = object.__new__(ExtraDatasetConfig)
        config.condition_method = method

        assert config._check_condition_method() is config

    @pytest.mark.pruned
    def test_condition_method_rejects_unknown_value(self):
        config = object.__new__(ExtraDatasetConfig)
        config.condition_method = "invalid"

        with pytest.raises(
            ValueError,
            match="Invalid condition_method",
        ):
            config._check_condition_method()

    @pytest.mark.pruned
    def test_resolve_lead_time_config(self):
        config = object.__new__(ExtraDatasetConfig)
        config.lead_times = lead_time_config(
            start=2,
            end=4,
        )

        config._resolve_lead_times()

        np.testing.assert_array_equal(
            config.lead_times,
            [2, 3, 4],
        )

    @pytest.mark.pruned
    def test_resolve_plain_lead_times_leaves_object_unchanged(self):
        lead_times = np.asarray([1, 3, 5])
        config = object.__new__(ExtraDatasetConfig)
        config.lead_times = lead_times

        config._resolve_lead_times()

        assert config.lead_times is lead_times

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "method,expected",
        [
            ("ensemble_mean", True),
            ("cross_ensemble", True),
            ("same_member", True),
            ("static", False),
        ],
    )
    def test_using_model_as_condition_without_condition(
        self,
        method,
        expected,
    ):
        config = object.__new__(ExtraDatasetConfig)
        config.model = object()
        config.condition = None
        config.condition_method = method

        assert config._using_model_data_as_condition is expected

    @pytest.mark.pruned
    def test_matching_model_and_condition_are_same_source(self):
        model = SimpleNamespace(
            paths=["same.nc"],
            names=["tas"],
            realization_list=[0, 1],
        )
        condition = SimpleNamespace(
            paths=["same.nc"],
            names=["tas"],
            realization_list=[0, 1],
        )

        config = object.__new__(ExtraDatasetConfig)
        config.model = model
        config.condition = condition
        config.condition_method = "same_member"

        assert config._using_model_data_as_condition is True

    @pytest.mark.pruned
    def test_different_condition_paths_are_not_same_source(self):
        model = SimpleNamespace(
            paths=["model.nc"],
            names=["tas"],
            realization_list=[0],
        )
        condition = SimpleNamespace(
            paths=["condition.nc"],
            names=["tas"],
            realization_list=[0],
        )

        config = object.__new__(ExtraDatasetConfig)
        config.model = model
        config.condition = condition
        config.condition_method = "same_member"

        assert config._using_model_data_as_condition is False

    @pytest.mark.pruned
    def test_condition_without_model_is_not_model_source(self):
        config = object.__new__(ExtraDatasetConfig)
        config.model = None
        config.condition = object()
        config.condition_method = "static"

        assert config._using_model_data_as_condition is False

    @pytest.mark.pruned
    def test_resolve_explicit_condition(self):
        condition = object()
        config = object.__new__(ExtraDatasetConfig)
        config.model = object()
        config.condition = condition
        config.condition_method = "static"
        config._effective_condition = None

        result = config._resolve_condition()

        assert result is config
        assert config.effective_condition is condition

    @pytest.mark.pruned
    def test_resolve_model_as_condition(
        self,
        monkeypatch,
    ):
        config = object.__new__(ExtraDatasetConfig)
        config.model = object()
        config.condition = None
        config.condition_method = "ensemble_mean"
        config._effective_condition = None

        resolved = object()
        resolver = Mock(return_value=resolved)
        monkeypatch.setattr(
            config,
            "_model_as_condition",
            resolver,
        )

        config._resolve_condition()

        assert config.effective_condition is resolved
        resolver.assert_called_once_with()

    @pytest.mark.pruned
    def test_resolve_condition_to_none(self):
        config = object.__new__(ExtraDatasetConfig)
        config.model = object()
        config.condition = None
        config.condition_method = "static"
        config._effective_condition = object()

        config._resolve_condition()

        assert config.effective_condition is None

    @pytest.mark.pruned
    def test_input_lead_times_from_effective_input(self):
        config = object.__new__(ExtraDatasetConfig)
        config.model = make_extra_data_config(
            coords={
                lead_time_dim: xr.DataArray(
                    [1, 2, 3],
                    dims=(lead_time_dim,),
                )
            }
        )
        config.condition = None

        np.testing.assert_array_equal(
            config.input_lead_times,
            [1, 2, 3],
        )

    @pytest.mark.pruned
    def test_same_member_rejects_ensemble_mean_model(self):
        config = object.__new__(ExtraDatasetConfig)
        config.model = SimpleNamespace(ensemble_mean=True)
        config.condition_method = "same_member"

        with pytest.raises(
            ValueError,
            match="should not be ensemble mean",
        ):
            config._check_model()

    @pytest.mark.pruned
    def test_model_check_accepts_non_same_member_method(self):
        config = object.__new__(ExtraDatasetConfig)
        config.model = SimpleNamespace(ensemble_mean=True)
        config.condition_method = "ensemble_mean"

        assert config._check_model() is config

    @pytest.mark.pruned
    def test_condition_requires_method(self):
        condition = make_extra_data_config()

        config = object.__new__(ExtraDatasetConfig)
        config.model = None
        config.condition = condition
        config.condition_method = None
        config._effective_condition = condition

        with pytest.raises(
            ValueError,
            match="specify condition_method",
        ):
            config._check_condition()

    @pytest.mark.pruned
    def test_cross_ensemble_rejects_ensemble_mean(self):
        condition = make_extra_data_config(
            ensemble_mean=True,
            coords={
                realization_dim: xr.DataArray(
                    [0, 1],
                    dims=(realization_dim,),
                )
            },
        )

        config = object.__new__(ExtraDatasetConfig)
        config.model = None
        config.condition = condition
        config.condition_method = "cross_ensemble"
        config._effective_condition = condition

        with pytest.raises(
            ValueError,
            match="ensemble_mean cannot be True",
        ):
            config._check_condition()

    @pytest.mark.pruned
    def test_same_member_requires_realization_coordinate(self):
        condition = make_extra_data_config(
            ensemble_mean=False,
            coords={},
        )

        config = object.__new__(ExtraDatasetConfig)
        config.model = None
        config.condition = condition
        config.condition_method = "same_member"
        config._effective_condition = condition

        with pytest.raises(
            ValueError,
            match="dim must exist",
        ):
            config._check_condition()

    @pytest.mark.pruned
    def test_ensemble_mean_method_requires_ensemble_mean_data(self):
        condition = make_extra_data_config(ensemble_mean=False)

        config = object.__new__(ExtraDatasetConfig)
        config.model = None
        config.condition = condition
        config.condition_method = "ensemble_mean"
        config._effective_condition = condition

        with pytest.raises(
            ValueError,
            match="Ensemble mean must be True",
        ):
            config._check_condition()

    @pytest.mark.pruned
    def test_static_rejects_realization_list(self):
        condition = make_extra_data_config(
            realization_list=[0],
            coords={},
        )

        config = object.__new__(ExtraDatasetConfig)
        config.model = None
        config.condition = condition
        config.condition_method = "static"
        config._effective_condition = condition

        with pytest.raises(
            ValueError,
            match="cannot specify realization list",
        ):
            config._check_condition()

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "sampling_dim",
        [
            init_time_dim,
            lead_time_dim,
            realization_dim,
        ],
    )
    def test_static_rejects_sampling_coordinates(
        self,
        sampling_dim,
    ):
        condition = make_extra_data_config(
            coords={
                sampling_dim: xr.DataArray(
                    [0],
                    dims=(sampling_dim,),
                )
            }
        )

        config = object.__new__(ExtraDatasetConfig)
        config.model = None
        config.condition = condition
        config.condition_method = "static"
        config._effective_condition = condition

        with pytest.raises(
            ValueError,
            match="cannot have.*sampling dimensions",
        ):
            config._check_condition()

    @pytest.mark.pruned
    def test_static_requires_condition_dataset(self):
        config = object.__new__(ExtraDatasetConfig)
        config.model = object()
        config.condition = None
        config.condition_method = "static"
        config._effective_condition = None

        with pytest.raises(
            ValueError,
            match="condition dataset must be specified",
        ):
            config._check_condition()

    @pytest.mark.pruned
    def test_get_input_times_rejects_unavailable_time(self):
        values = np.asarray(
            ["2001-01-01"],
            dtype="datetime64[ns]",
        )
        requested = xr.DataArray(
            values,
            dims=(init_time_dim,),
        )
        model_times = xr.DataArray(
            np.asarray(
                ["2000-01-01"],
                dtype="datetime64[ns]",
            ),
            dims=(init_time_dim,),
        )

        config = object.__new__(ExtraDatasetConfig)
        config.model = make_extra_data_config(coords={init_time_dim: model_times})
        config.condition = None
        config._available_times = pd.DatetimeIndex(["2000-01-01"])

        with pytest.raises(
            ValueError,
            match="requested_times are unavailable",
        ):
            config.get_input_times(requested)

    @pytest.mark.pruned
    def test_get_input_times_intersects_model_times(self):
        values = np.asarray(
            [
                "2000-01-01",
                "2001-01-01",
                "2002-01-01",
            ],
            dtype="datetime64[ns]",
        )
        requested = xr.DataArray(
            values,
            dims=(init_time_dim,),
            coords={init_time_dim: values},
        )
        model_values = values[[0, 2]]
        model_times = xr.DataArray(
            model_values,
            dims=(init_time_dim,),
            coords={init_time_dim: model_values},
        )

        config = object.__new__(ExtraDatasetConfig)
        config.model = make_extra_data_config(coords={init_time_dim: model_times})
        config.condition = None
        config._available_times = pd.DatetimeIndex(values)

        result = config.get_input_times(requested)

        np.testing.assert_array_equal(
            result.values,
            model_values,
        )


class TestExtraAddedTimeFeatures:
    @pytest.mark.pruned
    def test_rejects_unsupported_feature(self):
        with pytest.raises(
            ValueError,
            match="Unsupported time features",
        ):
            AddedTimeFeatures(
                make_extra_time_feature_config(),
                ["hour_sin"],
            )

    @pytest.mark.pruned
    def test_feature_order_is_canonical(self):
        features = AddedTimeFeatures(
            make_extra_time_feature_config(),
            [
                "day_cos",
                lead_time_dim,
                "month_sin",
            ],
        )

        assert features.time_features == (
            lead_time_dim,
            "month_sin",
            "day_cos",
        )

    @pytest.mark.pruned
    def test_duplicate_features_are_removed(self):
        features = AddedTimeFeatures(
            make_extra_time_feature_config(),
            [
                lead_time_dim,
                lead_time_dim,
                "month_cos",
                "month_cos",
            ],
        )

        assert features.time_features == (
            lead_time_dim,
            "month_cos",
        )

    @pytest.mark.pruned
    def test_missing_initialization_time_raises(self):
        features = AddedTimeFeatures(
            make_extra_time_feature_config(),
            [lead_time_dim],
        )

        with pytest.raises(
            ValueError,
            match="missing required dimensions",
        ):
            features.build_time_features({lead_time_dim: np.asarray([1])})

    @pytest.mark.pruned
    def test_missing_lead_time_raises(self):
        features = AddedTimeFeatures(
            make_extra_time_feature_config(),
            [init_time_dim],
        )

        with pytest.raises(
            ValueError,
            match="missing required dimensions",
        ):
            features.build_time_features(
                {
                    init_time_dim: np.asarray(
                        ["2000-01-01"],
                        dtype="datetime64[ns]",
                    )
                }
            )

    @pytest.mark.pruned
    def test_no_features_returns_without_building_array(
        self,
        monkeypatch,
    ):
        features = AddedTimeFeatures(
            make_extra_time_feature_config(),
            None,
        )
        add_times = Mock()
        monkeypatch.setattr(
            dataset_abc_module,
            "add_lead_times",
            add_times,
        )

        result = features.build_time_features(
            {
                init_time_dim: np.asarray(
                    ["2000-01-01"],
                    dtype="datetime64[ns]",
                ),
                lead_time_dim: np.asarray([1]),
            }
        )

        assert result is features
        assert features.time_features_array is None
        add_times.assert_not_called()

    @pytest.mark.pruned
    def test_builds_all_feature_branches(self):
        features = AddedTimeFeatures(
            make_extra_time_feature_config(),
            [
                init_time_dim,
                lead_time_dim,
                "month_sin",
                "month_cos",
                "day_sin",
                "day_cos",
            ],
        )

        result = features.build_time_features(
            {
                init_time_dim: np.asarray(
                    [
                        "2000-01-01",
                        "2000-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                lead_time_dim: np.asarray([1, 4]),
            }
        )

        assert result is features
        assert features.time_features_array.shape == (2, 6)
        assert features.time_features_array.dtype == np.float32
        assert np.isfinite(features.time_features_array).all()

    @pytest.mark.pruned
    def test_call_requires_built_features(self):
        features = AddedTimeFeatures(
            make_extra_time_feature_config(),
            [lead_time_dim],
        )

        with pytest.raises(
            RuntimeError,
            match="must be built before indexing",
        ):
            features(
                0,
                xr.DataArray(
                    [1.0],
                    dims=("channels",),
                ),
            )

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "index",
        [-1, 2],
    )
    def test_call_rejects_out_of_bounds_index(
        self,
        index,
    ):
        features = AddedTimeFeatures(
            make_extra_time_feature_config(),
            [lead_time_dim],
        )
        features.time_features_array = np.asarray(
            [
                [0.25],
                [0.50],
            ],
            dtype=np.float32,
        )

        with pytest.raises(
            IndexError,
            match="out of bounds",
        ):
            features(
                index,
                xr.DataArray(
                    [1.0],
                    dims=("channels",),
                ),
            )

    @pytest.mark.pruned
    def test_call_broadcasts_over_spatial_dimensions(self):
        features = AddedTimeFeatures(
            make_extra_time_feature_config(),
            [
                lead_time_dim,
                "month_cos",
            ],
        )
        features.time_features_array = np.asarray(
            [[0.5, 1.0]],
            dtype=np.float32,
        )
        input_data = xr.DataArray(
            np.ones((3, 2, 4)),
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
            2,
            4,
        )
        assert result.flags.writeable

    @pytest.mark.pruned
    def test_call_does_not_broadcast_two_dimensional_input(self):
        features = AddedTimeFeatures(
            make_extra_time_feature_config(),
            [
                lead_time_dim,
                "month_cos",
            ],
        )
        features.time_features_array = np.asarray(
            [[0.5, 1.0]],
            dtype=np.float32,
        )

        result = features(
            0,
            xr.DataArray(
                np.ones((2, 4)),
                dims=(
                    "channels",
                    "features",
                ),
            ),
        )

        assert result.shape == (2,)

    @pytest.mark.pruned
    def test_equality_other_type_returns_not_implemented(self):
        features = AddedTimeFeatures(
            make_extra_time_feature_config(),
            None,
        )

        assert features.__eq__(object()) is NotImplemented


class TestExtraDatasetABC:
    def test_check_init_rejects_unfitted_preprocessors(
        self,
        monkeypatch,
    ):
        dataset = make_extra_dataset()
        dataset.requested_times = xr.DataArray(
            np.asarray(
                ["2000-01-01"],
                dtype="datetime64[ns]",
            ),
            dims=(init_time_dim,),
        )
        dataset.config = SimpleNamespace(
            _fitted_preprocessors=False,
            available_times=pd.DatetimeIndex(["2000-01-01"]),
        )

        monkeypatch.setattr(
            dataset_abc_module,
            "_validate_time_sequence",
            Mock(),
        )

        with pytest.raises(
            RuntimeError,
            match="fit preprocessors first",
        ):
            dataset._check_init()

    @pytest.mark.pruned
    def test_check_init_rejects_unavailable_time(
        self,
        monkeypatch,
    ):
        dataset = make_extra_dataset()
        dataset.requested_times = xr.DataArray(
            np.asarray(
                ["2001-01-01"],
                dtype="datetime64[ns]",
            ),
            dims=(init_time_dim,),
        )
        dataset.config = SimpleNamespace(
            _fitted_preprocessors=True,
            available_times=pd.DatetimeIndex(["2000-01-01"]),
        )

        monkeypatch.setattr(
            dataset_abc_module,
            "_validate_time_sequence",
            Mock(),
        )

        with pytest.raises(
            ValueError,
            match="initialization times are unavailable",
        ):
            dataset._check_init()

    @pytest.mark.pruned
    def test_resolve_mask_builds_default_false_mask(
        self,
        monkeypatch,
    ):
        template = xr.DataArray(
            np.ones(
                (1, 2),
                dtype=bool,
            ),
            dims=(
                init_time_dim,
                lead_time_dim,
            ),
            coords={
                init_time_dim: np.asarray(
                    ["2000-01-01"],
                    dtype="datetime64[ns]",
                ),
                lead_time_dim: [1, 2],
            },
        )
        create_mask = Mock(return_value=template)
        monkeypatch.setattr(
            dataset_abc_module,
            "_create_train_mask",
            create_mask,
        )

        dataset = make_extra_dataset()
        dataset.mask = None
        dataset.config = SimpleNamespace(
            available_times=template[init_time_dim],
            input_lead_times=np.asarray([1, 2]),
            init_time_dim=init_time_dim,
            lead_time_dim=lead_time_dim,
        )

        dataset._resolve_mask()

        assert not dataset.mask.any()
        create_mask.assert_called_once()

    @pytest.mark.pruned
    def test_resolve_mask_rejects_missing_dimension(self):
        dataset = make_extra_dataset()
        dataset.mask = xr.DataArray(
            np.ones(2),
            dims=(init_time_dim,),
        )
        dataset.config = SimpleNamespace(
            init_time_dim=init_time_dim,
            lead_time_dim=lead_time_dim,
        )

        with pytest.raises(
            ValueError,
            match="mask must have",
        ):
            dataset._resolve_mask()

    @pytest.mark.pruned
    def test_sampling_selectors_delegate_to_config(self):
        dataset = make_extra_dataset()
        requested = xr.DataArray(
            np.asarray(
                ["2000-01-01"],
                dtype="datetime64[ns]",
            ),
            dims=(init_time_dim,),
        )
        selected = requested.copy()
        get_input_times = Mock(return_value=selected)

        dataset.requested_times = requested
        dataset.config = SimpleNamespace(
            init_time_dim=init_time_dim,
            lead_time_dim=lead_time_dim,
            lead_times=np.asarray([1, 2]),
            get_input_times=get_input_times,
        )

        result = dataset._sampling_times_selectors

        assert result[init_time_dim] is selected
        np.testing.assert_array_equal(
            result[lead_time_dim],
            [1, 2],
        )
        get_input_times.assert_called_once_with(requested)

    @pytest.mark.pruned
    def test_prepare_sampling_mask_requires_both_selectors(self):
        dataset = make_extra_dataset()
        dataset.config = SimpleNamespace(
            init_time_dim=init_time_dim,
            lead_time_dim=lead_time_dim,
        )

        with pytest.raises(
            ValueError,
            match="No selectors provided",
        ):
            dataset._prepare_sampling_mask(
                {
                    init_time_dim: [],
                }
            )

    @pytest.mark.pruned
    def test_prepare_sampling_mask_expands_realizations(self):
        times = np.asarray(
            ["2000-01-01"],
            dtype="datetime64[ns]",
        )
        dataset = make_extra_dataset()
        dataset.mask = xr.DataArray(
            [[False, True]],
            dims=(
                init_time_dim,
                lead_time_dim,
            ),
            coords={
                init_time_dim: times,
                lead_time_dim: [1, 2],
            },
        )
        dataset.config = SimpleNamespace(
            init_time_dim=init_time_dim,
            lead_time_dim=lead_time_dim,
            realization_dim=realization_dim,
            effective_input=SimpleNamespace(
                ensemble_mean=False,
                info=SimpleNamespace(
                    coords={
                        realization_dim: (
                            xr.DataArray(
                                [0, 1],
                                dims=(realization_dim,),
                            )
                        )
                    }
                ),
            ),
        )

        result = dataset._prepare_sampling_mask(
            {
                init_time_dim: times,
                lead_time_dim: [1, 2],
            }
        )

        assert result is dataset
        assert realization_dim in dataset.mask.dims
        assert dataset.mask.sizes[realization_dim] == 2
        assert np.isnan(
            dataset.mask.sel(
                {
                    realization_dim: 0,
                    lead_time_dim: 2,
                }
            ).item()
        )

    @pytest.mark.pruned
    def test_prepare_sampling_mask_skips_realization_for_ensemble_mean(self):
        times = np.asarray(
            ["2000-01-01"],
            dtype="datetime64[ns]",
        )
        dataset = make_extra_dataset()
        dataset.mask = xr.DataArray(
            [[False]],
            dims=(
                init_time_dim,
                lead_time_dim,
            ),
            coords={
                init_time_dim: times,
                lead_time_dim: [1],
            },
        )
        dataset.config = SimpleNamespace(
            init_time_dim=init_time_dim,
            lead_time_dim=lead_time_dim,
            realization_dim=realization_dim,
            effective_input=SimpleNamespace(
                ensemble_mean=True,
                info=SimpleNamespace(
                    coords={
                        realization_dim: (
                            xr.DataArray(
                                [0, 1],
                                dims=(realization_dim,),
                            )
                        )
                    }
                ),
            ),
        )

        dataset._prepare_sampling_mask(
            {
                init_time_dim: times,
                lead_time_dim: [1],
            }
        )

        assert realization_dim not in dataset.mask.dims

    @pytest.mark.pruned
    def test_get_sampling_coords_drops_nan_entries(self):
        dataset = make_extra_dataset()
        dataset.mask = xr.DataArray(
            [
                [0.0, np.nan],
                [0.0, 0.0],
            ],
            dims=(
                init_time_dim,
                lead_time_dim,
            ),
            coords={
                init_time_dim: np.asarray(
                    [
                        "2000-01-01",
                        "2001-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                lead_time_dim: [1, 2],
            },
        )

        result = dataset.get_sampling_coords()

        assert len(result[init_time_dim]) == 3
        assert len(result[lead_time_dim]) == 3

    @pytest.mark.pruned
    def test_get_model_indexes_returns_none_when_disabled(self):
        dataset = make_extra_dataset(load_model=False)

        assert dataset.get_model_indexes({}) is None

    @pytest.mark.pruned
    def test_get_model_indexes_success(self):
        times = np.asarray(
            [
                "2000-01-01",
                "2001-01-01",
            ],
            dtype="datetime64[ns]",
        )
        dataset = make_extra_dataset(load_model=True)
        dataset.model_dataset = xr.DataArray(
            np.ones(2),
            dims=(init_time_dim,),
            coords={init_time_dim: times},
        )

        result = dataset.get_model_indexes({init_time_dim: times[::-1]})

        np.testing.assert_array_equal(
            result[init_time_dim],
            [1, 0],
        )

    @pytest.mark.pruned
    def test_get_model_indexes_reports_missing_values(self):
        dataset = make_extra_dataset(load_model=True)
        dataset.model_dataset = xr.DataArray(
            np.ones(1),
            dims=(init_time_dim,),
            coords={
                init_time_dim: np.asarray(
                    ["2000-01-01"],
                    dtype="datetime64[ns]",
                )
            },
        )

        with pytest.raises(
            ValueError,
            match="model dataset",
        ):
            dataset.get_model_indexes(
                {
                    init_time_dim: np.asarray(
                        ["2001-01-01"],
                        dtype="datetime64[ns]",
                    )
                }
            )

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "condition_dataset,method",
        [
            (None, "same_member"),
            (xr.DataArray(1.0), "static"),
        ],
    )
    def test_condition_index_early_returns(
        self,
        condition_dataset,
        method,
    ):
        dataset = make_extra_dataset()
        dataset.condition_dataset = condition_dataset
        dataset.config = SimpleNamespace(
            condition_method=method,
            realization_dim=realization_dim,
        )

        assert dataset.get_cond_indexes({}) is None

    @pytest.mark.pruned
    def test_same_member_indexes_require_realization(self):
        dataset = make_extra_dataset()
        dataset.condition_dataset = xr.DataArray(
            np.ones(1),
            dims=(init_time_dim,),
            coords={
                init_time_dim: np.asarray(
                    ["2000-01-01"],
                    dtype="datetime64[ns]",
                )
            },
        )
        dataset.config = SimpleNamespace(
            condition_method="same_member",
            realization_dim=realization_dim,
        )

        with pytest.raises(
            ValueError,
            match="requires.*coordinates",
        ):
            dataset.get_cond_indexes(
                {
                    init_time_dim: (
                        np.asarray(
                            ["2000-01-01"],
                            dtype="datetime64[ns]",
                        )
                    )
                }
            )

    @pytest.mark.pruned
    def test_condition_indexes_report_missing_values(self):
        dataset = make_extra_dataset()
        dataset.condition_dataset = xr.DataArray(
            np.ones(1),
            dims=(init_time_dim,),
            coords={
                init_time_dim: np.asarray(
                    ["2000-01-01"],
                    dtype="datetime64[ns]",
                )
            },
        )
        dataset.config = SimpleNamespace(
            condition_method="ensemble_mean",
            realization_dim=realization_dim,
        )

        with pytest.raises(
            ValueError,
            match="conditioning dataset",
        ):
            dataset.get_cond_indexes(
                {
                    init_time_dim: (
                        np.asarray(
                            ["2001-01-01"],
                            dtype="datetime64[ns]",
                        )
                    )
                }
            )

    @pytest.mark.pruned
    def test_load_xarray_data_forwards_realization_selection(
        self,
        monkeypatch,
    ):
        members = xr.DataArray(
            [0, 2],
            dims=(realization_dim,),
        )
        config = SimpleNamespace(
            list_paths=["a.nc"],
            names=["tas"],
            ensemble_mean=False,
            realization_dim=realization_dim,
            info=SimpleNamespace(coords={realization_dim: members}),
            concat_dim="time",
            rename_dict={"old": "new"},
        )
        loader = Mock(return_value="loaded")
        monkeypatch.setattr(
            dataset_abc_module,
            "_load_xarray_data",
            loader,
        )

        dataset = make_extra_dataset()
        result = dataset._load_xarray_data(
            config,
            load=True,
            add_time_auxiliary_coords=True,
        )

        assert result == "loaded"
        loader.assert_called_once_with(
            ["a.nc"],
            names=["tas"],
            ensemble_mean=False,
            selection={realization_dim: members},
            concat_dim="time",
            rename_dict={"old": "new"},
            add_time_auxiliary_coords=True,
            load=True,
        )

    @pytest.mark.pruned
    def test_load_xarray_data_without_realization_uses_none_selection(
        self,
        monkeypatch,
    ):
        config = SimpleNamespace(
            list_paths=["a.nc"],
            names=["tas"],
            ensemble_mean=True,
            realization_dim=realization_dim,
            info=SimpleNamespace(coords={}),
            concat_dim=None,
            rename_dict=None,
        )
        loader = Mock(return_value="loaded")
        monkeypatch.setattr(
            dataset_abc_module,
            "_load_xarray_data",
            loader,
        )

        dataset = make_extra_dataset()
        dataset._load_xarray_data(config)

        assert loader.call_args.kwargs["selection"] is None

    @pytest.mark.pruned
    def test_index_static_condition(
        self,
        monkeypatch,
    ):
        transform = Mock(side_effect=lambda value: value)
        dataset = make_extra_dataset()
        dataset.condition_dataset = xr.Dataset(
            {
                "condition": (
                    "channels",
                    [5.0],
                )
            },
            coords={"channels": ["condition"]},
        )
        dataset.config = SimpleNamespace(
            condition_method="static",
            realization_dim=realization_dim,
            effective_condition=SimpleNamespace(
                preprocessing_pipeline=(SimpleNamespace(transform=transform))
            ),
        )

        monkeypatch.setattr(
            dataset_abc_module,
            "_unwrap_data_variables",
            lambda value: value["condition"],
        )

        result = dataset._index_condition_dataset(0)

        assert result.item() == pytest.approx(5.0)
        transform.assert_called_once()

    @pytest.mark.pruned
    def test_index_cross_ensemble_condition(
        self,
        monkeypatch,
    ):
        dataset = make_extra_dataset()
        dataset.condition_dataset = xr.Dataset(
            {
                "condition": (
                    (
                        init_time_dim,
                        realization_dim,
                    ),
                    [[1.0, 2.0, 3.0]],
                )
            },
            coords={
                init_time_dim: np.asarray(
                    ["2000-01-01"],
                    dtype="datetime64[ns]",
                ),
                realization_dim: [
                    0,
                    1,
                    2,
                ],
            },
        )
        dataset.cond_indexes = {init_time_dim: np.asarray([0])}
        dataset.config = SimpleNamespace(
            condition_method="cross_ensemble",
            realization_dim=realization_dim,
            effective_condition=SimpleNamespace(
                preprocessing_pipeline=(SimpleNamespace(transform=lambda value: value))
            ),
        )

        randint = Mock(return_value=2)
        monkeypatch.setattr(
            dataset_abc_module.np.random,
            "randint",
            randint,
        )
        monkeypatch.setattr(
            dataset_abc_module,
            "_unwrap_data_variables",
            lambda value: value["condition"],
        )

        result = dataset._index_condition_dataset(0)

        assert result.item() == pytest.approx(3.0)
        randint.assert_called_once_with(3)

    @pytest.mark.pruned
    def test_index_model_dataset(
        self,
        monkeypatch,
    ):
        transform = Mock(side_effect=lambda value: value)
        dataset = make_extra_dataset(load_model=True)
        dataset.model_dataset = xr.Dataset(
            {
                "tas": (
                    init_time_dim,
                    [1.0, 2.0],
                )
            },
            coords={
                init_time_dim: np.asarray(
                    [
                        "2000-01-01",
                        "2001-01-01",
                    ],
                    dtype="datetime64[ns]",
                )
            },
        )
        dataset.model_indexes = {init_time_dim: np.asarray([np.int64(1)])}
        dataset.config = SimpleNamespace(
            model=SimpleNamespace(
                preprocessing_pipeline=(SimpleNamespace(transform=transform))
            )
        )

        monkeypatch.setattr(
            dataset_abc_module,
            "_unwrap_data_variables",
            lambda value: value["tas"],
        )

        result = dataset._index_model_dataset(0)

        assert result.item() == pytest.approx(2.0)
        transform.assert_called_once()

    @pytest.mark.pruned
    def test_compute_uses_dask_compute(
        self,
        monkeypatch,
    ):
        first = np.asarray([1.0])
        second = np.asarray([2.0])
        compute = Mock(
            return_value=(
                first,
                second,
            )
        )
        monkeypatch.setattr(
            dataset_abc_module.dask,
            "compute",
            compute,
        )

        result = DatasetABC._compute(
            first,
            second,
        )

        assert result == (
            first,
            second,
        )
        compute.assert_called_once_with(
            first,
            second,
        )

    @pytest.mark.pruned
    def test_length_uses_first_sample_coordinate(self):
        dataset = make_extra_dataset()
        dataset.sample_coords = {
            init_time_dim: np.asarray([1, 2, 3]),
            lead_time_dim: np.asarray([1, 1, 1]),
        }

        assert len(dataset) == 3


from types import SimpleNamespace

import pytest
import xarray as xr

from cccma_ppp.configs import (
    realization_dim as branch_realization_dim,
    required_sample_dimensions as branch_required_sample_dimensions,
)
from cccma_ppp.data_modules.dataset.dataset_abc import (
    AddedTimeFeatures as BranchAddedTimeFeatures,
    DatasetABC as BranchDatasetABC,
    DatasetConfigABC as BranchDatasetConfigABC,
    lead_time_config as BranchLeadTimeConfig,
)

branch_init_time_dim, branch_lead_time_dim = branch_required_sample_dimensions


class BranchCoverageDatasetConfig(BranchDatasetConfigABC):
    def __init__(
        self,
        *,
        model=None,
        condition=None,
        condition_method=None,
        lead_times=None,
        available_times=None,
        effective_input_override=None,
        run_parent=False,
    ):
        self.model = model
        self.condition = condition
        self.condition_method = condition_method
        self.lead_times = lead_times
        self._branch_available_times = available_times
        self._branch_effective_input = effective_input_override

        if run_parent:
            super().__init__()
        else:
            self._fitted_preprocessors = False
            self._effective_condition = None

    @property
    def available_times(self):
        return self._branch_available_times

    @property
    def ds_operator(self):
        return None

    @property
    def effective_input(self):
        if self._branch_effective_input is not None:
            return self._branch_effective_input

        if self.model is not None:
            return self.model

        return self.condition

    def build_dataset(self):
        return None


class BranchCoverageDataset(BranchDatasetABC):
    @property
    def _load_model(self):
        return self.branch_load_model

    @property
    def _write_condition_to_input(self):
        return self.branch_write_condition

    @property
    def _concat_condition_to_input(self):
        return self.branch_concat_condition

    def __getitem__(self, index):
        raise NotImplementedError


def make_branch_coverage_dataset(
    *,
    load_model=True,
    write_condition=False,
    concat_condition=False,
):
    dataset = object.__new__(BranchCoverageDataset)
    dataset.branch_load_model = load_model
    dataset.branch_write_condition = write_condition
    dataset.branch_concat_condition = concat_condition
    dataset.model_dataset = None
    dataset.condition_dataset = None
    dataset.observation_dataset = None
    dataset.model_indexes = None
    dataset.cond_indexes = None
    dataset.sample_coords = {}

    return dataset


def make_branch_coverage_coord(
    values,
    dimension,
):
    return xr.DataArray(
        values,
        dims=(dimension,),
        coords={
            dimension: values,
        },
    )


def make_branch_coverage_pipeline():
    return SimpleNamespace(
        fitted_preprocessors=[],
        transform=Mock(side_effect=lambda value: value),
        get_preprocessors=Mock(),
    )


def make_branch_coverage_data_config(
    *,
    paths=None,
    names=None,
    realization_list=None,
    ensemble_mean=False,
    coords=None,
    time_coords_type="datetime",
    preprocessing_pipeline=None,
):
    if paths is None:
        paths = ["data.nc"]

    if names is None:
        names = ["tas"]

    if coords is None:
        coords = {}

    if preprocessing_pipeline is None:
        preprocessing_pipeline = make_branch_coverage_pipeline()

    return SimpleNamespace(
        paths=paths,
        list_paths=paths,
        names=names,
        realization_list=realization_list,
        ensemble_mean=ensemble_mean,
        info=SimpleNamespace(
            coords=coords,
            time_coords_type=time_coords_type,
        ),
        preprocessing_pipeline=(preprocessing_pipeline),
        concat_dim=None,
        file_type="netcdf",
        rename_dict=None,
        realization_dim=(branch_realization_dim),
    )


def make_branch_time_feature_reference(
    *,
    lead_times=(1, 2, 4),
    common_times=None,
    resolution="month",
):
    if common_times is None:
        common_times = pd.DatetimeIndex(
            [
                "2000-01-01",
                "2000-12-31",
            ]
        )

    return SimpleNamespace(
        init_time_dim=branch_init_time_dim,
        lead_time_dim=branch_lead_time_dim,
        lead_time_resolution=resolution,
        lead_times=np.asarray(lead_times),
        get_common_time=common_times,
    )


class TestMoreLeadTimeConfigBranches:
    @pytest.mark.pruned
    def test_explicit_tuple_is_returned(self):
        values = (1, 3, 6)
        config = BranchLeadTimeConfig(list_lead_times=values)

        assert config.build_lead_times() is values

    @pytest.mark.pruned
    def test_explicit_nonempty_list_ignores_invalid_range(self):
        config = BranchLeadTimeConfig(
            list_lead_times=[2, 5],
            start=10,
            end=1,
        )

        assert config.build_lead_times() == [
            2,
            5,
        ]

    @pytest.mark.pruned
    def test_zero_start_is_supported(self):
        config = BranchLeadTimeConfig(
            start=0,
            end=2,
        )

        np.testing.assert_array_equal(
            config.build_lead_times(),
            [0, 1, 2],
        )

    @pytest.mark.pruned
    def test_reversed_range_returns_empty_array(self):
        config = BranchLeadTimeConfig(
            start=4,
            end=2,
        )

        assert config.build_lead_times().size == 0


class TestMoreDatasetConfigConstructorBranches:
    @pytest.mark.pruned
    def test_constructor_uses_all_input_lead_times(self):
        condition = make_branch_coverage_data_config(coords={})
        effective_input = SimpleNamespace(
            info=SimpleNamespace(
                coords={
                    branch_lead_time_dim: make_branch_coverage_coord(
                        [1, 2, 3],
                        branch_lead_time_dim,
                    )
                }
            )
        )

        config = BranchCoverageDatasetConfig(
            model=None,
            condition=condition,
            condition_method="static",
            lead_times=None,
            available_times=pd.DatetimeIndex(["2000-01-01"]),
            effective_input_override=(effective_input),
            run_parent=True,
        )

        np.testing.assert_array_equal(
            config.lead_times,
            [1, 2, 3],
        )
        assert config._fitted_preprocessors is False

    @pytest.mark.pruned
    def test_constructor_accepts_requested_subset(self):
        condition = make_branch_coverage_data_config(coords={})
        effective_input = SimpleNamespace(
            info=SimpleNamespace(
                coords={
                    branch_lead_time_dim: make_branch_coverage_coord(
                        [1, 2, 3, 4],
                        branch_lead_time_dim,
                    )
                }
            )
        )

        config = BranchCoverageDatasetConfig(
            condition=condition,
            condition_method="static",
            lead_times=[1, 4],
            available_times=pd.DatetimeIndex(["2000-01-01"]),
            effective_input_override=(effective_input),
            run_parent=True,
        )

        assert config.lead_times == [1, 4]

    @pytest.mark.pruned
    def test_constructor_resolves_lead_time_config(self):
        condition = make_branch_coverage_data_config(coords={})
        effective_input = SimpleNamespace(
            info=SimpleNamespace(
                coords={
                    branch_lead_time_dim: make_branch_coverage_coord(
                        [1, 2, 3, 4],
                        branch_lead_time_dim,
                    )
                }
            )
        )

        config = BranchCoverageDatasetConfig(
            condition=condition,
            condition_method="static",
            lead_times=BranchLeadTimeConfig(
                start=2,
                end=4,
            ),
            available_times=pd.DatetimeIndex(["2000-01-01"]),
            effective_input_override=(effective_input),
            run_parent=True,
        )

        np.testing.assert_array_equal(
            config.lead_times,
            [2, 3, 4],
        )

    def test_constructor_rejects_unavailable_lead_time(self):
        condition = make_branch_coverage_data_config(coords={})
        effective_input = SimpleNamespace(
            info=SimpleNamespace(
                coords={
                    branch_lead_time_dim: make_branch_coverage_coord(
                        [1, 2, 3],
                        branch_lead_time_dim,
                    )
                }
            )
        )

        with pytest.raises(
            ValueError,
            match="requested lead times are not available",
        ):
            BranchCoverageDatasetConfig(
                condition=condition,
                condition_method="static",
                lead_times=[1, 9],
                available_times=(pd.DatetimeIndex(["2000-01-01"])),
                effective_input_override=(effective_input),
                run_parent=True,
            )

    @pytest.mark.pruned
    def test_constructor_rejects_missing_input_sources(self):
        with pytest.raises(
            ValueError,
            match="either model or condition",
        ):
            BranchCoverageDatasetConfig(
                model=None,
                condition=None,
                condition_method="static",
                lead_times=[1],
                available_times=(pd.DatetimeIndex(["2000-01-01"])),
                run_parent=True,
            )

    @pytest.mark.pruned
    def test_constructor_rejects_invalid_method_before_resolution(self):
        condition = make_branch_coverage_data_config()

        with pytest.raises(
            ValueError,
            match="Invalid condition_method",
        ):
            BranchCoverageDatasetConfig(
                condition=condition,
                condition_method="invalid",
                lead_times=[1],
                available_times=(pd.DatetimeIndex(["2000-01-01"])),
                run_parent=True,
            )


class TestMoreModelConditionCompatibilityBranches:
    @pytest.mark.pruned
    def test_compatibility_check_skips_without_model(self):
        config = object.__new__(BranchCoverageDatasetConfig)
        config.model = None
        config.condition = make_branch_coverage_data_config()
        config.condition_method = "static"

        assert config._check_model_vs_condition() is None

    @pytest.mark.pruned
    def test_compatibility_check_skips_without_condition(self):
        config = object.__new__(BranchCoverageDatasetConfig)
        config.model = make_branch_coverage_data_config()
        config.condition = None
        config.condition_method = "ensemble_mean"

        assert config._check_model_vs_condition() is None

    @pytest.mark.pruned
    def test_compatibility_check_skips_same_source(self):
        model = make_branch_coverage_data_config(
            paths=["same.nc"],
            names=["tas"],
            realization_list=[0],
        )
        condition = make_branch_coverage_data_config(
            paths=["same.nc"],
            names=["tas"],
            realization_list=[0],
        )

        config = object.__new__(BranchCoverageDatasetConfig)
        config.model = model
        config.condition = condition
        config.condition_method = "same_member"
        config._effective_condition = condition

        assert config._check_model_vs_condition() is None

    @pytest.mark.pruned
    def test_nonstatic_condition_requires_model_time_dimension(self):
        model = make_branch_coverage_data_config(
            coords={
                branch_init_time_dim: make_branch_coverage_coord(
                    np.asarray(
                        ["2000-01-01"],
                        dtype="datetime64[ns]",
                    ),
                    branch_init_time_dim,
                )
            }
        )
        condition = make_branch_coverage_data_config(
            paths=["condition.nc"],
            coords={},
        )

        config = object.__new__(BranchCoverageDatasetConfig)
        config.model = model
        config.condition = condition
        config.condition_method = "cross_ensemble"
        config._effective_condition = condition

        with pytest.raises(
            ValueError,
            match="same.*dimestions",
        ):
            config._check_model_vs_condition()

    @pytest.mark.pruned
    def test_nonstatic_condition_requires_all_model_times(self):
        model = make_branch_coverage_data_config(
            coords={
                branch_init_time_dim: make_branch_coverage_coord(
                    np.asarray(
                        [
                            "2000-01-01",
                            "2001-01-01",
                        ],
                        dtype="datetime64[ns]",
                    ),
                    branch_init_time_dim,
                )
            }
        )
        condition = make_branch_coverage_data_config(
            paths=["condition.nc"],
            coords={
                branch_init_time_dim: make_branch_coverage_coord(
                    np.asarray(
                        ["2000-01-01"],
                        dtype="datetime64[ns]",
                    ),
                    branch_init_time_dim,
                )
            },
        )

        config = object.__new__(BranchCoverageDatasetConfig)
        config.model = model
        config.condition = condition
        config.condition_method = "cross_ensemble"
        config._effective_condition = condition

        with pytest.raises(
            ValueError,
            match="same.*coordinates",
        ):
            config._check_model_vs_condition()

    @pytest.mark.pruned
    def test_nonstatic_condition_requires_matching_time_types(self):
        time_coord = make_branch_coverage_coord(
            np.asarray(
                ["2000-01-01"],
                dtype="datetime64[ns]",
            ),
            branch_init_time_dim,
        )
        model = make_branch_coverage_data_config(
            coords={branch_init_time_dim: time_coord},
            time_coords_type="datetime",
        )
        condition = make_branch_coverage_data_config(
            paths=["condition.nc"],
            coords={branch_init_time_dim: time_coord.copy()},
            time_coords_type="cftime",
        )

        config = object.__new__(BranchCoverageDatasetConfig)
        config.model = model
        config.condition = condition
        config.condition_method = "cross_ensemble"
        config._effective_condition = condition

        with pytest.raises(
            ValueError,
            match="same.*time coordinates",
        ):
            config._check_model_vs_condition()

    def test_static_condition_skips_time_compatibility_checks(self):
        model = make_branch_coverage_data_config(
            coords={
                branch_init_time_dim: make_branch_coverage_coord(
                    np.asarray(
                        ["2000-01-01"],
                        dtype="datetime64[ns]",
                    ),
                    branch_init_time_dim,
                )
            }
        )
        condition = make_branch_coverage_data_config(
            paths=["condition.nc"],
            coords={},
            time_coords_type="cftime",
        )

        config = object.__new__(BranchCoverageDatasetConfig)
        config.model = model
        config.condition = condition
        config.condition_method = "static"
        config._effective_condition = condition

        assert config._check_model_vs_condition() is None

    @pytest.mark.pruned
    def test_same_member_requires_model_realization(self):
        condition = make_branch_coverage_data_config(
            paths=["condition.nc"],
            coords={
                branch_realization_dim: make_branch_coverage_coord(
                    [0, 1],
                    branch_realization_dim,
                )
            },
        )
        model = make_branch_coverage_data_config(coords={})

        config = object.__new__(BranchCoverageDatasetConfig)
        config.model = model
        config.condition = condition
        config.condition_method = "same_member"
        config._effective_condition = condition

        with pytest.raises(
            ValueError,
            match="dims and coords",
        ):
            config._check_model_vs_condition()

    @pytest.mark.pruned
    def test_same_member_requires_condition_realization(self):
        model = make_branch_coverage_data_config(
            coords={
                branch_realization_dim: make_branch_coverage_coord(
                    [0, 1],
                    branch_realization_dim,
                )
            }
        )
        condition = make_branch_coverage_data_config(
            paths=["condition.nc"],
            coords={},
        )

        config = object.__new__(BranchCoverageDatasetConfig)
        config.model = model
        config.condition = condition
        config.condition_method = "same_member"
        config._effective_condition = condition

        with pytest.raises(
            ValueError,
            match="dims and coords",
        ):
            config._check_model_vs_condition()

    @pytest.mark.pruned
    def test_same_member_requires_matching_members(self):
        model = make_branch_coverage_data_config(
            coords={
                branch_realization_dim: make_branch_coverage_coord(
                    [0, 1],
                    branch_realization_dim,
                )
            }
        )
        condition = make_branch_coverage_data_config(
            paths=["condition.nc"],
            coords={
                branch_realization_dim: make_branch_coverage_coord(
                    [0, 2],
                    branch_realization_dim,
                )
            },
        )

        config = object.__new__(BranchCoverageDatasetConfig)
        config.model = model
        config.condition = condition
        config.condition_method = "same_member"
        config._effective_condition = condition

        with pytest.raises(
            ValueError,
            match="same ensemble members",
        ):
            config._check_model_vs_condition()

    @pytest.mark.pruned
    def test_same_member_accepts_matching_members(self):
        members = make_branch_coverage_coord(
            [0, 1],
            branch_realization_dim,
        )
        model = make_branch_coverage_data_config(
            coords={branch_realization_dim: members}
        )
        condition = make_branch_coverage_data_config(
            paths=["condition.nc"],
            coords={branch_realization_dim: members.copy()},
        )

        config = object.__new__(BranchCoverageDatasetConfig)
        config.model = model
        config.condition = condition
        config.condition_method = "same_member"
        config._effective_condition = condition

        assert config._check_model_vs_condition() is None


class TestMoreConditionValidationBranches:
    @pytest.mark.pruned
    def test_valid_cross_ensemble_condition(self):
        condition = make_branch_coverage_data_config(
            ensemble_mean=False,
            coords={
                branch_realization_dim: make_branch_coverage_coord(
                    [0, 1],
                    branch_realization_dim,
                )
            },
        )

        config = object.__new__(BranchCoverageDatasetConfig)
        config.model = None
        config.condition = condition
        config.condition_method = "cross_ensemble"
        config._effective_condition = condition

        assert config._check_condition() is config

    @pytest.mark.pruned
    def test_valid_same_member_condition(self):
        condition = make_branch_coverage_data_config(
            ensemble_mean=False,
            coords={
                branch_realization_dim: make_branch_coverage_coord(
                    [0, 1],
                    branch_realization_dim,
                )
            },
        )

        config = object.__new__(BranchCoverageDatasetConfig)
        config.model = None
        config.condition = condition
        config.condition_method = "same_member"
        config._effective_condition = condition

        assert config._check_condition() is config

    @pytest.mark.pruned
    def test_valid_ensemble_mean_condition(self):
        condition = make_branch_coverage_data_config(
            ensemble_mean=True,
            coords={},
        )

        config = object.__new__(BranchCoverageDatasetConfig)
        config.model = None
        config.condition = condition
        config.condition_method = "ensemble_mean"
        config._effective_condition = condition

        assert config._check_condition() is config

    @pytest.mark.pruned
    def test_valid_static_condition(self):
        condition = make_branch_coverage_data_config(
            ensemble_mean=False,
            realization_list=None,
            coords={
                "lat": make_branch_coverage_coord(
                    [45.0, 46.0],
                    "lat",
                )
            },
        )

        config = object.__new__(BranchCoverageDatasetConfig)
        config.model = None
        config.condition = condition
        config.condition_method = "static"
        config._effective_condition = condition

        assert config._check_condition() is config

    @pytest.mark.pruned
    def test_static_same_source_is_rejected(self):
        model = make_branch_coverage_data_config(
            paths=["same.nc"],
            names=["tas"],
            realization_list=None,
            coords={},
        )
        condition = make_branch_coverage_data_config(
            paths=["same.nc"],
            names=["tas"],
            realization_list=None,
            coords={},
        )

        config = object.__new__(BranchCoverageDatasetConfig)
        config.model = model
        config.condition = condition
        config.condition_method = "static"
        config._effective_condition = condition

        with pytest.raises(
            ValueError,
            match="cannot point to the same model data",
        ):
            config._check_condition()


class TestMoreModelAsConditionBranches:
    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "method,expected_ensemble_mean",
        [
            ("ensemble_mean", True),
            ("cross_ensemble", False),
            ("same_member", False),
        ],
    )
    def test_model_as_condition_forwards_configuration(
        self,
        method,
        expected_ensemble_mean,
        monkeypatch,
    ):
        pipeline = object()
        model = SimpleNamespace(
            paths=["first.nc", "second.nc"],
            names=["tas", "pr"],
            preprocessing_pipeline=pipeline,
            realization_list=[0, 1],
            concat_dim="time",
            file_type="netcdf",
            rename_dict={"old": "new"},
        )

        config = object.__new__(BranchCoverageDatasetConfig)
        config.model = model
        config.condition = None
        config.condition_method = method

        built = object()
        constructor = Mock(return_value=built)
        monkeypatch.setattr(
            dataset_abc_module,
            "ModelDataConfig",
            constructor,
        )

        result = config._model_as_condition()

        assert result is built
        constructor.assert_called_once_with(
            paths=model.paths,
            names=model.names,
            preprocessing_pipeline=pipeline,
            realization_list=[0, 1],
            concat_dim="time",
            file_type="netcdf",
            ensemble_mean=(expected_ensemble_mean),
            rename_dict={"old": "new"},
        )


class TestMoreAddedTimeFeatureBranches:
    @pytest.mark.pruned
    def test_only_lead_time_feature(self):
        features = BranchAddedTimeFeatures(
            make_branch_time_feature_reference(lead_times=(1, 2, 4)),
            [branch_lead_time_dim],
        )

        features.build_time_features(
            {
                branch_init_time_dim: np.asarray(
                    [
                        "2000-01-01",
                        "2000-01-01",
                        "2000-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                branch_lead_time_dim: np.asarray([1, 2, 4]),
            }
        )

        np.testing.assert_allclose(
            features.time_features_array[
                :,
                0,
            ],
            [0.25, 0.5, 1.0],
        )

    def test_only_month_sine_feature(self):
        features = BranchAddedTimeFeatures(
            make_branch_time_feature_reference(),
            ["month_sin"],
        )

        features.build_time_features(
            {
                branch_init_time_dim: np.asarray(
                    [
                        "2000-01-01",
                        "2000-04-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                branch_lead_time_dim: np.asarray([1, 1]),
            }
        )

        np.testing.assert_allclose(
            features.time_features_array[
                :,
                0,
            ],
            [0.0, 1.0],
            atol=1e-6,
        )

    @pytest.mark.pruned
    def test_only_month_cosine_feature(self):
        features = BranchAddedTimeFeatures(
            make_branch_time_feature_reference(),
            ["month_cos"],
        )

        features.build_time_features(
            {
                branch_init_time_dim: np.asarray(
                    [
                        "2000-01-01",
                        "2000-07-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                branch_lead_time_dim: np.asarray([1, 1]),
            }
        )

        np.testing.assert_allclose(
            features.time_features_array[
                :,
                0,
            ],
            [1.0, -1.0],
            atol=1e-6,
        )

    def test_only_day_sine_feature(self):
        features = BranchAddedTimeFeatures(
            make_branch_time_feature_reference(),
            ["day_sin"],
        )

        features.build_time_features(
            {
                branch_init_time_dim: np.asarray(
                    ["2001-01-01"],
                    dtype="datetime64[ns]",
                ),
                branch_lead_time_dim: np.asarray([1]),
            }
        )

        assert features.time_features_array[
            0,
            0,
        ] == pytest.approx(
            0.0,
            abs=1e-6,
        )

    @pytest.mark.pruned
    def test_only_day_cosine_feature(self):
        features = BranchAddedTimeFeatures(
            make_branch_time_feature_reference(),
            ["day_cos"],
        )

        features.build_time_features(
            {
                branch_init_time_dim: np.asarray(
                    ["2001-01-01"],
                    dtype="datetime64[ns]",
                ),
                branch_lead_time_dim: np.asarray([1]),
            }
        )

        assert features.time_features_array[
            0,
            0,
        ] == pytest.approx(
            1.0,
            abs=1e-6,
        )

    def test_add_lead_times_called_once(
        self,
        monkeypatch,
    ):
        target_times = np.asarray(
            [
                "2000-01-01",
                "2000-02-01",
            ],
            dtype="datetime64[ns]",
        )
        add_times = Mock(return_value=target_times)
        monkeypatch.setattr(
            dataset_abc_module,
            "add_lead_times",
            add_times,
        )

        features = BranchAddedTimeFeatures(
            make_branch_time_feature_reference(),
            [
                branch_lead_time_dim,
                "month_cos",
                "day_cos",
            ],
        )
        init_times = np.asarray(
            [
                "2000-01-01",
                "2000-01-01",
            ],
            dtype="datetime64[ns]",
        )
        lead_times = np.asarray([1, 2])

        features.build_time_features(
            {
                branch_init_time_dim: init_times,
                branch_lead_time_dim: lead_times,
            }
        )

        add_times.assert_called_once_with(
            init_times=init_times,
            lead_times=lead_times,
            lead_time_resolution="month",
        )

    def test_build_replaces_existing_feature_array(self):
        features = BranchAddedTimeFeatures(
            make_branch_time_feature_reference(),
            [branch_lead_time_dim],
        )
        features.time_features_array = np.asarray(
            [[99.0]],
            dtype=np.float32,
        )

        features.build_time_features(
            {
                branch_init_time_dim: np.asarray(
                    ["2000-01-01"],
                    dtype="datetime64[ns]",
                ),
                branch_lead_time_dim: np.asarray([1]),
            }
        )

        assert (
            features.time_features_array[
                0,
                0,
            ]
            != 99.0
        )

    @pytest.mark.pruned
    def test_call_returns_unbroadcast_vector_for_one_dimensional_input(self):
        features = BranchAddedTimeFeatures(
            make_branch_time_feature_reference(),
            [
                branch_lead_time_dim,
                "month_cos",
            ],
        )
        features.time_features_array = np.asarray(
            [[0.5, 1.0]],
            dtype=np.float32,
        )

        result = features(
            0,
            xr.DataArray(
                np.ones(3),
                dims=("channels",),
            ),
        )

        assert result.shape == (2,)

    @pytest.mark.pruned
    def test_broadcast_result_is_independent_copy(self):
        features = BranchAddedTimeFeatures(
            make_branch_time_feature_reference(),
            [branch_lead_time_dim],
        )
        features.time_features_array = np.asarray(
            [[0.5]],
            dtype=np.float32,
        )

        result = features(
            0,
            xr.DataArray(
                np.ones((2, 3, 4)),
                dims=(
                    "channels",
                    "lat",
                    "lon",
                ),
            ),
        )
        result[0, 0, 0] = 99.0

        assert features.time_features_array[
            0,
            0,
        ] == pytest.approx(0.5)

    @pytest.mark.pruned
    def test_equality_accepts_matching_configs(self):
        class MatchingReference:
            lead_times = np.asarray([1, 2])
            get_common_time = pd.DatetimeIndex(
                [
                    "2000-01-01",
                    "2000-02-01",
                ]
            )
            lead_time_resolution = "month"
            init_time_dim = branch_init_time_dim
            lead_time_dim = branch_lead_time_dim

        first = BranchAddedTimeFeatures(
            MatchingReference(),
            [branch_lead_time_dim],
        )
        second = BranchAddedTimeFeatures(
            MatchingReference(),
            [branch_lead_time_dim],
        )

        assert first == second

    @pytest.mark.pruned
    def test_equality_rejects_different_features(self):
        reference = make_branch_time_feature_reference()

        first = BranchAddedTimeFeatures(
            reference,
            [branch_lead_time_dim],
        )
        second = BranchAddedTimeFeatures(
            reference,
            ["month_cos"],
        )

        assert first != second

    @pytest.mark.pruned
    def test_equality_rejects_different_lead_times(self):
        first = BranchAddedTimeFeatures(
            make_branch_time_feature_reference(lead_times=(1, 2)),
            [branch_lead_time_dim],
        )
        second = BranchAddedTimeFeatures(
            make_branch_time_feature_reference(lead_times=(1, 3)),
            [branch_lead_time_dim],
        )

        assert first != second


class TestMoreDatasetMaskAndSamplingBranches:
    @pytest.mark.pruned
    def test_valid_existing_mask_is_preserved(self):
        mask = xr.DataArray(
            np.zeros((1, 1), dtype=bool),
            dims=(
                branch_init_time_dim,
                branch_lead_time_dim,
            ),
        )
        dataset = make_branch_coverage_dataset()
        dataset.mask = mask
        dataset.config = SimpleNamespace(
            init_time_dim=(branch_init_time_dim),
            lead_time_dim=(branch_lead_time_dim),
        )

        dataset._resolve_mask()

        assert dataset.mask is mask

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "retained_dimension",
        [
            branch_init_time_dim,
            branch_lead_time_dim,
        ],
    )
    def test_mask_reports_each_missing_dimension(
        self,
        retained_dimension,
    ):
        dataset = make_branch_coverage_dataset()
        dataset.mask = xr.DataArray(
            np.ones(2),
            dims=(retained_dimension,),
        )
        dataset.config = SimpleNamespace(
            init_time_dim=(branch_init_time_dim),
            lead_time_dim=(branch_lead_time_dim),
        )

        with pytest.raises(
            ValueError,
            match="mask must have",
        ):
            dataset._resolve_mask()

    @pytest.mark.pruned
    def test_sampling_mask_skips_realization_when_coordinate_missing(self):
        times = np.asarray(
            ["2000-01-01"],
            dtype="datetime64[ns]",
        )
        dataset = make_branch_coverage_dataset()
        dataset.mask = xr.DataArray(
            [[False]],
            dims=(
                branch_init_time_dim,
                branch_lead_time_dim,
            ),
            coords={
                branch_init_time_dim: times,
                branch_lead_time_dim: [1],
            },
        )
        dataset.config = SimpleNamespace(
            init_time_dim=(branch_init_time_dim),
            lead_time_dim=(branch_lead_time_dim),
            realization_dim=(branch_realization_dim),
            effective_input=SimpleNamespace(
                ensemble_mean=False,
                info=SimpleNamespace(coords={}),
            ),
        )

        dataset._prepare_sampling_mask(
            {
                branch_init_time_dim: times,
                branch_lead_time_dim: [1],
            }
        )

        assert branch_realization_dim not in dataset.mask.dims

    @pytest.mark.pruned
    def test_sampling_mask_expands_realization_when_available(self):
        times = np.asarray(
            ["2000-01-01"],
            dtype="datetime64[ns]",
        )
        dataset = make_branch_coverage_dataset()
        dataset.mask = xr.DataArray(
            [[False, True]],
            dims=(
                branch_init_time_dim,
                branch_lead_time_dim,
            ),
            coords={
                branch_init_time_dim: times,
                branch_lead_time_dim: [1, 2],
            },
        )
        dataset.config = SimpleNamespace(
            init_time_dim=(branch_init_time_dim),
            lead_time_dim=(branch_lead_time_dim),
            realization_dim=(branch_realization_dim),
            effective_input=SimpleNamespace(
                ensemble_mean=False,
                info=SimpleNamespace(
                    coords={
                        branch_realization_dim: make_branch_coverage_coord(
                            [0, 1],
                            branch_realization_dim,
                        )
                    }
                ),
            ),
        )

        dataset._prepare_sampling_mask(
            {
                branch_init_time_dim: times,
                branch_lead_time_dim: [1, 2],
            }
        )

        assert dataset.mask.dims == (
            branch_realization_dim,
            branch_init_time_dim,
            branch_lead_time_dim,
        )
        assert np.isnan(
            dataset.mask.sel(
                {
                    branch_realization_dim: 0,
                    branch_lead_time_dim: 2,
                }
            ).item()
        )

    @pytest.mark.pruned
    def test_empty_sampling_mask_returns_empty_coordinates(self):
        dataset = make_branch_coverage_dataset()
        dataset.mask = xr.DataArray(
            np.full((1, 1), np.nan),
            dims=(
                branch_init_time_dim,
                branch_lead_time_dim,
            ),
            coords={
                branch_init_time_dim: np.asarray(
                    ["2000-01-01"],
                    dtype="datetime64[ns]",
                ),
                branch_lead_time_dim: [1],
            },
        )

        result = dataset.get_sampling_coords()

        assert len(result[branch_init_time_dim]) == 0
        assert len(result[branch_lead_time_dim]) == 0

    @pytest.mark.pruned
    def test_sampling_coordinates_include_realization(self):
        dataset = make_branch_coverage_dataset()
        dataset.mask = xr.DataArray(
            np.zeros((2, 1, 1)),
            dims=(
                branch_realization_dim,
                branch_init_time_dim,
                branch_lead_time_dim,
            ),
            coords={
                branch_realization_dim: [0, 1],
                branch_init_time_dim: np.asarray(
                    ["2000-01-01"],
                    dtype="datetime64[ns]",
                ),
                branch_lead_time_dim: [1],
            },
        )

        result = dataset.get_sampling_coords()

        assert set(result) == {
            branch_realization_dim,
            branch_init_time_dim,
            branch_lead_time_dim,
        }
        np.testing.assert_array_equal(
            result[branch_realization_dim],
            [0, 1],
        )


class TestMoreIndexBranches:
    @pytest.mark.pruned
    def test_model_indexes_cover_multiple_dimensions(self):
        times = np.asarray(
            [
                "2000-01-01",
                "2001-01-01",
            ],
            dtype="datetime64[ns]",
        )
        dataset = make_branch_coverage_dataset(load_model=True)
        dataset.model_dataset = xr.DataArray(
            np.ones((2, 2)),
            dims=(
                branch_init_time_dim,
                branch_lead_time_dim,
            ),
            coords={
                branch_init_time_dim: times,
                branch_lead_time_dim: [1, 2],
            },
        )

        result = dataset.get_model_indexes(
            {
                branch_init_time_dim: times[::-1],
                branch_lead_time_dim: np.asarray([2, 1]),
            }
        )

        np.testing.assert_array_equal(
            result[branch_init_time_dim],
            [1, 0],
        )
        np.testing.assert_array_equal(
            result[branch_lead_time_dim],
            [1, 0],
        )

    @pytest.mark.pruned
    def test_condition_indexes_ignore_dimensions_not_present(self):
        times = np.asarray(
            ["2000-01-01"],
            dtype="datetime64[ns]",
        )
        dataset = make_branch_coverage_dataset()
        dataset.condition_dataset = xr.DataArray(
            np.ones(1),
            dims=(branch_init_time_dim,),
            coords={branch_init_time_dim: times},
        )
        dataset.config = SimpleNamespace(
            condition_method="ensemble_mean",
            realization_dim=(branch_realization_dim),
        )

        result = dataset.get_cond_indexes(
            {
                branch_init_time_dim: times,
                branch_lead_time_dim: np.asarray([99]),
                "unused": np.asarray([100]),
            }
        )

        assert set(result) == {branch_init_time_dim}

    @pytest.mark.pruned
    def test_cross_ensemble_indexes_exclude_realization(self):
        times = np.asarray(
            ["2000-01-01"],
            dtype="datetime64[ns]",
        )
        dataset = make_branch_coverage_dataset()
        dataset.condition_dataset = xr.DataArray(
            np.ones((1, 2)),
            dims=(
                branch_init_time_dim,
                branch_realization_dim,
            ),
            coords={
                branch_init_time_dim: times,
                branch_realization_dim: [0, 1],
            },
        )
        dataset.config = SimpleNamespace(
            condition_method="cross_ensemble",
            realization_dim=(branch_realization_dim),
        )

        result = dataset.get_cond_indexes(
            {
                branch_init_time_dim: times,
                branch_realization_dim: np.asarray([1]),
            }
        )

        assert branch_realization_dim not in result

    @pytest.mark.pruned
    def test_same_member_indexes_include_realization(self):
        times = np.asarray(
            ["2000-01-01"],
            dtype="datetime64[ns]",
        )
        dataset = make_branch_coverage_dataset()
        dataset.condition_dataset = xr.DataArray(
            np.ones((1, 2)),
            dims=(
                branch_init_time_dim,
                branch_realization_dim,
            ),
            coords={
                branch_init_time_dim: times,
                branch_realization_dim: [0, 1],
            },
        )
        dataset.config = SimpleNamespace(
            condition_method="same_member",
            realization_dim=(branch_realization_dim),
        )

        result = dataset.get_cond_indexes(
            {
                branch_init_time_dim: times,
                branch_realization_dim: np.asarray([1]),
            }
        )

        np.testing.assert_array_equal(
            result[branch_realization_dim],
            [1],
        )


class TestMoreShapeAndIndexingBranches:
    @pytest.mark.pruned
    def test_input_shape_without_flattener(self):
        effective_input = SimpleNamespace(
            names=["tas", "pr"],
            info=SimpleNamespace(
                coords={
                    "lat": make_branch_coverage_coord(
                        [45.0, 46.0],
                        "lat",
                    ),
                    "lon": make_branch_coverage_coord(
                        [-124.0, -123.0, -122.0],
                        "lon",
                    ),
                }
            ),
            preprocessing_pipeline=(make_branch_coverage_pipeline()),
        )
        dataset = make_branch_coverage_dataset()
        dataset.config = SimpleNamespace(
            effective_input=effective_input,
            effective_condition=None,
            supported_NN_dimensions=(
                "lat",
                "lon",
            ),
        )

        assert dataset.get_input_shape() == (
            2,
            2,
            3,
        )

    @pytest.mark.pruned
    def test_input_shape_includes_condition_channels(self):
        effective_input = SimpleNamespace(
            names=["tas", "pr"],
            info=SimpleNamespace(
                coords={
                    "lat": make_branch_coverage_coord(
                        [45.0, 46.0],
                        "lat",
                    )
                }
            ),
            preprocessing_pipeline=(make_branch_coverage_pipeline()),
        )
        effective_condition = SimpleNamespace(
            names=[
                "orog",
                "land_mask",
                "ice_mask",
            ]
        )
        dataset = make_branch_coverage_dataset(concat_condition=True)
        dataset.config = SimpleNamespace(
            effective_input=effective_input,
            effective_condition=(effective_condition),
            supported_NN_dimensions=(
                "lat",
                "lon",
            ),
        )

        assert dataset.get_input_shape() == (
            5,
            2,
        )

    @pytest.mark.pruned
    def test_input_shape_without_spatial_coordinates(self):
        effective_input = SimpleNamespace(
            names=["tas"],
            info=SimpleNamespace(coords={}),
            preprocessing_pipeline=(make_branch_coverage_pipeline()),
        )
        dataset = make_branch_coverage_dataset()
        dataset.config = SimpleNamespace(
            effective_input=effective_input,
            effective_condition=None,
            supported_NN_dimensions=(
                "lat",
                "lon",
            ),
        )

        assert dataset.get_input_shape() == (1,)

    @pytest.mark.pruned
    def test_index_condition_returns_none_without_dataset(self):
        dataset = make_branch_coverage_dataset()
        dataset.condition_dataset = None

        assert dataset._index_condition_dataset(0) is None

    @pytest.mark.pruned
    def test_index_model_returns_none_when_disabled(self):
        dataset = make_branch_coverage_dataset(load_model=False)

        assert dataset._index_model_dataset(0) is None

    @pytest.mark.pruned
    def test_index_nonstatic_condition_uses_indexes(
        self,
        monkeypatch,
    ):
        transform = Mock(side_effect=lambda value: value)
        dataset = make_branch_coverage_dataset()
        dataset.condition_dataset = xr.Dataset(
            {
                "condition": (
                    branch_init_time_dim,
                    [10.0, 20.0],
                )
            },
            coords={
                branch_init_time_dim: np.asarray(
                    [
                        "2000-01-01",
                        "2001-01-01",
                    ],
                    dtype="datetime64[ns]",
                )
            },
        )
        dataset.cond_indexes = {branch_init_time_dim: np.asarray([1])}
        dataset.config = SimpleNamespace(
            condition_method="ensemble_mean",
            realization_dim=(branch_realization_dim),
            effective_condition=SimpleNamespace(
                preprocessing_pipeline=(SimpleNamespace(transform=transform))
            ),
        )

        monkeypatch.setattr(
            dataset_abc_module,
            "_unwrap_data_variables",
            lambda value: value["condition"],
        )

        result = dataset._index_condition_dataset(0)

        assert result.item() == pytest.approx(20.0)
        transform.assert_called_once()


class TestMoreLoadAndComputeBranches:
    @pytest.mark.pruned
    def test_data_loader_defaults_are_forwarded(
        self,
        monkeypatch,
    ):
        config = SimpleNamespace(
            list_paths=["data.nc"],
            names=["tas"],
            ensemble_mean=True,
            realization_dim=(branch_realization_dim),
            info=SimpleNamespace(coords={}),
            concat_dim=None,
            rename_dict=None,
        )
        loader = Mock(return_value="loaded")
        monkeypatch.setattr(
            dataset_abc_module,
            "_load_xarray_data",
            loader,
        )

        dataset = make_branch_coverage_dataset()
        result = dataset._load_xarray_data(config)

        assert result == "loaded"
        loader.assert_called_once_with(
            ["data.nc"],
            names=["tas"],
            ensemble_mean=True,
            selection=None,
            concat_dim=None,
            rename_dict=None,
            add_time_auxiliary_coords=False,
            load=False,
        )

    @pytest.mark.pruned
    def test_compute_with_single_array(
        self,
        monkeypatch,
    ):
        array = np.asarray([1.0, 2.0])
        compute = Mock(return_value=(array,))
        monkeypatch.setattr(
            dataset_abc_module.dask,
            "compute",
            compute,
        )

        result = BranchDatasetABC._compute(array)

        assert result == (array,)
        compute.assert_called_once_with(array)

    @pytest.mark.pruned
    def test_compute_with_no_arrays(
        self,
        monkeypatch,
    ):
        compute = Mock(return_value=())
        monkeypatch.setattr(
            dataset_abc_module.dask,
            "compute",
            compute,
        )

        result = BranchDatasetABC._compute()

        assert result == ()
        compute.assert_called_once_with()

    @pytest.mark.pruned
    def test_zero_length_dataset(self):
        dataset = make_branch_coverage_dataset()
        dataset.sample_coords = {
            branch_init_time_dim: np.asarray([]),
            branch_lead_time_dim: np.asarray([]),
        }

        assert len(dataset) == 0

    @pytest.mark.pruned
    def test_length_uses_first_coordinate(self):
        dataset = make_branch_coverage_dataset()
        dataset.sample_coords = {
            "first": np.asarray([1, 2, 3, 4]),
            "second": np.asarray([1, 2]),
        }

        assert len(dataset) == 4
