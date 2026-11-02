from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from S2D4P.data.dataset.operator import (
    DatasetOperator,
    _build_chunks,
)
from S2D4P.preprocessing.preprocessing_ABC import PreprocessModuleABC

INIT_TIME_DIM = "time"
LEAD_TIME_DIM = "lead_time"
REALIZATION_DIM = "ensembles"

AVAILABLE_TIMES = np.array(
    [
        "2000-01-01",
        "2001-01-01",
    ],
    dtype="datetime64[ns]",
)


def make_train_times(
    values=None,
):
    if values is None:
        values = AVAILABLE_TIMES

    values = np.asarray(
        values,
        dtype="datetime64[ns]",
    )

    return xr.DataArray(
        values,
        dims=(INIT_TIME_DIM,),
        coords={
            INIT_TIME_DIM: values,
        },
    )


def make_coord(
    values,
    dim,
):
    values = np.asarray(values)

    return xr.DataArray(
        values,
        dims=(dim,),
        coords={
            dim: values,
        },
    )


def make_pipeline(
    *,
    pipeline=None,
    fitted_preprocessors=None,
):
    if pipeline is None:
        pipeline = [
            (
                "StandardScaler",
                MagicMock(),
            ),
        ]

    if fitted_preprocessors is None:
        fitted_preprocessors = []

    result = MagicMock()
    result.pipeline = pipeline
    result.fitted_preprocessors = fitted_preprocessors
    result.get_preprocessors.return_value = None

    return result


def make_data_config(
    *,
    names=None,
    times=None,
    lead_times=None,
    realizations=None,
    spatial_coords=None,
    ensemble_mean=False,
    pipeline=None,
):
    if names is None:
        names = ["tas"]

    if times is None:
        times = AVAILABLE_TIMES

    if lead_times is None:
        lead_times = [1, 2, 3]

    coords = {
        INIT_TIME_DIM: make_coord(
            times,
            INIT_TIME_DIM,
        ),
        LEAD_TIME_DIM: make_coord(
            lead_times,
            LEAD_TIME_DIM,
        ),
    }

    if realizations is not None:
        coords[REALIZATION_DIM] = make_coord(
            realizations,
            REALIZATION_DIM,
        )

    if spatial_coords:
        for dim, values in spatial_coords.items():
            coords[dim] = make_coord(
                values,
                dim,
            )

    return SimpleNamespace(
        names=list(names),
        coords=coords,
        info=SimpleNamespace(
            coords=coords,
        ),
        ensemble_mean=ensemble_mean,
        preprocessing_pipeline=(make_pipeline() if pipeline is None else pipeline),
        fit_preprocessor_pipeline=MagicMock(),
        load_preprocessor_pipeline=MagicMock(),
        init_time_dim=INIT_TIME_DIM,
        lead_time_dim=LEAD_TIME_DIM,
        realization_dim=REALIZATION_DIM,
    )


class DummyDatasetConfig:
    effective_condition_method = property(lambda self: self.condition_method)

    init_time_dim = INIT_TIME_DIM
    lead_time_dim = LEAD_TIME_DIM
    realization_dim = REALIZATION_DIM
    supported_NN_dimensions = (
        "latitude",
        "longitude",
    )

    def __init__(
        self,
        *,
        model=None,
        observation=None,
        condition=None,
        condition_method=None,
        using_model_as_condition=False,
    ):
        self.model = model
        self.observation = observation
        self.effective_condition = condition
        self.condition_method = condition_method
        self._using_model_data_as_condition = using_model_as_condition

        self.available_times = AVAILABLE_TIMES
        self._fitted_preprocessors = False

    def get_input_times(
        self,
        requested_times,
    ):
        return requested_times


class DummyPreprocessor(PreprocessModuleABC):
    def _reset_fit(self):
        self.fitted = False

    def __init__(
        self,
        fitted=True,
    ):

        self.fitted = fitted

    def fit(self, data, *args, **kwargs):
        self.fitted = True
        return self

    def transform(self, data, *args, **kwargs):
        return data

    def inverse_transform(self, data, *args, **kwargs):
        return data


class TestConfigObservation:
    @pytest.mark.pruned
    def test_returns_observation_when_present(self):
        observation = make_data_config()
        config = DummyDatasetConfig(
            observation=observation,
        )

        operator = DatasetOperator(config)

        assert operator.config_observation is observation

    def test_returns_none_when_attribute_is_absent(self):
        config = SimpleNamespace()
        operator = DatasetOperator(config)

        assert operator.config_observation is None

    @pytest.mark.pruned
    def test_returns_none_when_observation_is_none(self):
        config = DummyDatasetConfig(
            observation=None,
        )
        operator = DatasetOperator(config)

        assert operator.config_observation is None


class LegacyFitPreprocessors:
    def test_rejects_unavailable_train_times(self):
        config = DummyDatasetConfig(
            model=make_data_config(),
        )
        operator = DatasetOperator(config)

        train_times = make_train_times(
            [
                "1990-01-01",
            ]
        )

        with pytest.raises(
            ValueError,
            match="train_times are unavailable",
        ):
            operator.fit_preprocessors(train_times)

    def test_validates_time_sequence(self):
        config = DummyDatasetConfig(
            model=make_data_config(),
        )
        operator = DatasetOperator(config)
        train_times = make_train_times()

        with patch(
            "cccma_ppp.data_modules.dataset.operator._validate_time_sequence"
        ) as mock_validate:
            operator.fit_preprocessors(train_times)

        mock_validate.assert_called_once_with(train_times)

    def test_fits_model_preprocessor(self):
        model = make_data_config()
        config = DummyDatasetConfig(
            model=model,
        )
        operator = DatasetOperator(config)
        train_times = make_train_times()

        operator.fit_preprocessors(
            train_times,
            save=True,
            save_path=Path("/tmp/output"),
            save_name="model.joblib",
        )

        model.fit_preprocessor_pipeline.assert_called_once()

        kwargs = model.fit_preprocessor_pipeline.call_args.kwargs

        assert kwargs["mask"] is True
        assert kwargs["save"] is True
        assert kwargs["save_path"] == Path("/tmp/output")
        assert kwargs["save_name"] == "model.joblib"
        assert kwargs["selection"][INIT_TIME_DIM] is train_times
        assert kwargs["selection"][LEAD_TIME_DIM] is model.info.coords[LEAD_TIME_DIM]

    def test_model_selection_includes_realizations(self):
        model = make_data_config(
            realizations=[
                0,
                1,
            ]
        )
        config = DummyDatasetConfig(
            model=model,
        )
        operator = DatasetOperator(config)

        operator.fit_preprocessors(make_train_times())

        selection = model.fit_preprocessor_pipeline.call_args.kwargs["selection"]

        assert selection[REALIZATION_DIM] is (model.info.coords[REALIZATION_DIM])

    def test_model_selection_omits_realizations_when_absent(self):
        model = make_data_config(
            realizations=None,
        )
        config = DummyDatasetConfig(
            model=model,
        )
        operator = DatasetOperator(config)

        operator.fit_preprocessors(make_train_times())

        selection = model.fit_preprocessor_pipeline.call_args.kwargs["selection"]

        assert REALIZATION_DIM not in selection

    def test_fits_observation_preprocessor(self):
        observation = make_data_config()
        config = DummyDatasetConfig(
            observation=observation,
        )
        operator = DatasetOperator(config)
        train_times = make_train_times()

        operator.fit_preprocessors(
            train_times,
            save=True,
            save_path="/tmp/output",
            save_name="observation.joblib",
        )

        observation.fit_preprocessor_pipeline.assert_called_once_with(
            selection={
                INIT_TIME_DIM: train_times,
            },
            save=True,
            save_path="/tmp/output",
            save_name="observation.joblib",
        )

    def test_observation_selection_includes_realizations(self):
        observation = make_data_config(
            realizations=[
                0,
                1,
            ]
        )
        config = DummyDatasetConfig(
            observation=observation,
        )
        operator = DatasetOperator(config)
        train_times = make_train_times()

        operator.fit_preprocessors(train_times)

        selection = observation.fit_preprocessor_pipeline.call_args.kwargs["selection"]

        assert selection[INIT_TIME_DIM] is train_times
        assert selection[REALIZATION_DIM] is (observation.info.coords[REALIZATION_DIM])

    def test_fits_dynamic_condition_preprocessor(self):
        condition = make_data_config()
        config = DummyDatasetConfig(
            condition=condition,
            condition_method="ensemble_mean",
        )
        operator = DatasetOperator(config)
        train_times = make_train_times()

        operator.fit_preprocessors(
            train_times,
            save=False,
        )

        condition.fit_preprocessor_pipeline.assert_called_once()

        kwargs = condition.fit_preprocessor_pipeline.call_args.kwargs

        assert kwargs["mask"] is True
        assert kwargs["save"] is False
        assert kwargs["selection"][INIT_TIME_DIM] is train_times
        assert (
            kwargs["selection"][LEAD_TIME_DIM] is (condition.info.coords[LEAD_TIME_DIM])
        )

    def test_dynamic_condition_selection_includes_realizations(self):
        condition = make_data_config(
            realizations=[
                0,
                1,
            ]
        )
        config = DummyDatasetConfig(
            condition=condition,
            condition_method="cross_ensemble",
        )
        operator = DatasetOperator(config)

        operator.fit_preprocessors(make_train_times())

        selection = condition.fit_preprocessor_pipeline.call_args.kwargs["selection"]

        assert selection[REALIZATION_DIM] is (condition.info.coords[REALIZATION_DIM])

    def test_fits_static_condition_with_empty_selection(self):
        condition = make_data_config()
        config = DummyDatasetConfig(
            condition=condition,
            condition_method="static",
        )
        operator = DatasetOperator(config)

        operator.fit_preprocessors(
            make_train_times(),
            save=True,
            save_path="/tmp/output",
            save_name="condition.joblib",
        )

        condition.fit_preprocessor_pipeline.assert_called_once_with(
            selection={},
            mask=True,
            save=True,
            save_path="/tmp/output",
            save_name="condition.joblib",
        )

    def test_get_input_times_is_used_for_model_and_condition(self):
        model = make_data_config()
        condition = make_data_config()
        config = DummyDatasetConfig(
            model=model,
            condition=condition,
            condition_method="ensemble_mean",
        )

        selected_times = make_train_times(
            [
                "2000-01-01",
            ]
        )
        config.get_input_times = MagicMock(return_value=selected_times)

        operator = DatasetOperator(config)
        train_times = make_train_times()

        operator.fit_preprocessors(train_times)

        assert config.get_input_times.call_count == 2

        model_selection = model.fit_preprocessor_pipeline.call_args.kwargs["selection"]
        condition_selection = condition.fit_preprocessor_pipeline.call_args.kwargs[
            "selection"
        ]

        assert model_selection[INIT_TIME_DIM] is selected_times
        assert condition_selection[INIT_TIME_DIM] is selected_times

    def test_skips_missing_sources(self):
        config = DummyDatasetConfig(
            model=None,
            observation=None,
            condition=None,
        )
        operator = DatasetOperator(config)

        operator.fit_preprocessors(make_train_times())

        assert config._fitted_preprocessors is True

    def test_sets_fitted_preprocessors_flag(self):
        config = DummyDatasetConfig(
            model=make_data_config(),
        )
        operator = DatasetOperator(config)

        assert config._fitted_preprocessors is False

        operator.fit_preprocessors(make_train_times())

        assert config._fitted_preprocessors is True


class LegacyLoadFittedPreprocessors:
    def test_loads_all_available_preprocessors(self):
        model = make_data_config()
        observation = make_data_config()
        condition = make_data_config()

        config = DummyDatasetConfig(
            model=model,
            observation=observation,
            condition=condition,
            condition_method="ensemble_mean",
        )
        operator = DatasetOperator(config)
        load_dir = Path("/tmp/preprocessors")

        operator.load_fitted_preprocessors(load_dir)

        model.load_preprocessor_pipeline.assert_called_once_with(load_dir)
        observation.load_preprocessor_pipeline.assert_called_once_with(load_dir)
        condition.load_preprocessor_pipeline.assert_called_once_with(load_dir)
        assert config._fitted_preprocessors is True

    def test_skips_missing_sources(self):
        config = DummyDatasetConfig()
        operator = DatasetOperator(config)

        operator.load_fitted_preprocessors()

        assert config._fitted_preprocessors is True

    def test_forwards_none_load_directory(self):
        model = make_data_config()
        config = DummyDatasetConfig(
            model=model,
        )
        operator = DatasetOperator(config)

        operator.load_fitted_preprocessors()

        model.load_preprocessor_pipeline.assert_called_once_with(None)


class TestAddFittedPreprocessor:
    def test_rejects_wrong_type(self):
        config = DummyDatasetConfig(
            model=make_data_config(),
        )
        operator = DatasetOperator(config)

        with pytest.raises(
            TypeError,
            match="preprocessor must be an instance",
        ):
            operator.add_fitted_preprocessor(object())

    def test_rejects_unfitted_preprocessor(self):
        config = DummyDatasetConfig(
            model=make_data_config(),
        )
        operator = DatasetOperator(config)
        preprocessor = DummyPreprocessor(
            fitted=False,
        )

        with pytest.raises(
            ValueError,
            match="must be fitted",
        ):
            operator.add_fitted_preprocessor(preprocessor)

    @pytest.mark.pruned
    def test_adds_to_all_available_pipelines(self):
        model = make_data_config()
        observation = make_data_config()
        condition = make_data_config()

        config = DummyDatasetConfig(
            model=model,
            observation=observation,
            condition=condition,
            condition_method="ensemble_mean",
        )
        operator = DatasetOperator(config)
        preprocessor = DummyPreprocessor(
            fitted=True,
        )

        operator.add_fitted_preprocessor(
            preprocessor,
            index=2,
        )

        for source in (
            model,
            observation,
            condition,
        ):
            (
                source.preprocessing_pipeline.add_fitted_preprocessor.assert_called_once_with(
                    preprocessor,
                    index=2,
                )
            )

    def test_skips_missing_sources(self):
        model = make_data_config()
        config = DummyDatasetConfig(
            model=model,
        )
        operator = DatasetOperator(config)
        preprocessor = DummyPreprocessor(
            fitted=True,
        )

        operator.add_fitted_preprocessor(preprocessor)

        (
            model.preprocessing_pipeline.add_fitted_preprocessor.assert_called_once_with(
                preprocessor,
                index=0,
            )
        )


class TestGetWeights:
    def test_uses_observation_before_model(self):
        model = make_data_config(
            spatial_coords={
                "latitude": [
                    -45,
                    45,
                ],
            }
        )
        observation = make_data_config(
            spatial_coords={
                "latitude": [
                    -30,
                    30,
                ],
                "longitude": [
                    0,
                    90,
                ],
            }
        )

        config = DummyDatasetConfig(
            model=model,
            observation=observation,
        )
        operator = DatasetOperator(config)

        weights_config = MagicMock()
        expected = xr.DataArray(
            np.ones(
                (
                    2,
                    2,
                )
            ),
            dims=(
                "latitude",
                "longitude",
            ),
        )
        weights_config.build_weights.return_value = expected

        result = operator.get_weights(
            config=weights_config,
            save=False,
        )

        assert result is expected

        target_coords = weights_config.build_weights.call_args.args[0]

        assert target_coords == {
            "latitude": observation.info.coords["latitude"],
            "longitude": observation.info.coords["longitude"],
        }

    @pytest.mark.pruned
    def test_uses_model_when_observation_is_absent(self):
        model = make_data_config(
            spatial_coords={
                "latitude": [
                    -45,
                    45,
                ],
            }
        )

        config = DummyDatasetConfig(
            model=model,
            observation=None,
        )
        operator = DatasetOperator(config)

        weights_config = MagicMock()
        expected = xr.DataArray(
            np.ones(2),
            dims=("latitude",),
        )
        weights_config.build_weights.return_value = expected

        result = operator.get_weights(
            config=weights_config,
        )

        assert result is expected

        target_coords = weights_config.build_weights.call_args.args[0]

        assert target_coords == {
            "latitude": model.info.coords["latitude"],
        }

    def test_rejects_missing_model_and_observation(self):
        config = DummyDatasetConfig()
        operator = DatasetOperator(config)

        with pytest.raises(
            ValueError,
            match="No model or observation data",
        ):
            operator.get_weights(config=MagicMock())

    @pytest.mark.pruned
    def test_forwards_save_arguments(self):
        model = make_data_config(
            spatial_coords={
                "latitude": [
                    -45,
                    45,
                ],
            }
        )
        config = DummyDatasetConfig(
            model=model,
        )
        operator = DatasetOperator(config)

        weights_config = MagicMock()
        weights_config.build_weights.return_value = xr.DataArray(
            np.ones(2),
            dims=("latitude",),
        )

        operator.get_weights(
            config=weights_config,
            save=True,
            save_path=Path("/tmp/output"),
            save_name="weights.nc",
        )

        kwargs = weights_config.build_weights.call_args.kwargs

        assert kwargs["save"] is True
        assert kwargs["save_path"] == Path("/tmp/output")
        assert kwargs["save_name"] == "weights.nc"

    @pytest.mark.pruned
    def test_without_flattennanremove_passes_none(self):
        model = make_data_config(
            spatial_coords={
                "latitude": [
                    -45,
                    45,
                ],
            }
        )
        config = DummyDatasetConfig(
            model=model,
        )
        operator = DatasetOperator(config)
        model.preprocessing_pipeline.get_flattener.return_value = None

        weights_config = MagicMock()
        weights_config.build_weights.return_value = xr.DataArray(
            np.ones(2),
            dims=("latitude",),
        )

        with patch(
            "cccma_ppp.preprocessing.utils_preprocessing.Flattennanremove"
        ) as flatten_type:
            flatten_type.return_value = MagicMock()

            operator.get_weights(
                config=weights_config,
            )

        assert (
            weights_config.build_weights.call_args.kwargs["Flattennanremover"] is None
        )

    @pytest.mark.pruned
    def test_with_flattennanremove_passes_flattener(self):
        from S2D4P.preprocessing.utils_preprocessing import (
            Flattennanremove,
        )

        flattener = MagicMock(spec=Flattennanremove)

        pipeline = make_pipeline(
            fitted_preprocessors=[
                flattener,
            ]
        )
        pipeline.get_flattener.return_value = flattener

        model = make_data_config(
            pipeline=pipeline,
            spatial_coords={
                "latitude": [
                    -45,
                    45,
                ],
            },
        )
        config = DummyDatasetConfig(
            model=model,
        )
        operator = DatasetOperator(config)

        weights_config = MagicMock()
        weights_config.build_weights.return_value = xr.DataArray(
            np.ones(2),
            dims=("latitude",),
        )

        operator.get_weights(
            config=weights_config,
        )

        pipeline.get_flattener.assert_called_once_with()
        assert (
            weights_config.build_weights.call_args.kwargs["Flattennanremover"]
            is flattener
        )

    @pytest.mark.pruned
    def test_matching_channel_weights(self):
        model = make_data_config(
            names=[
                "tas",
                "pr",
            ],
            spatial_coords={
                "latitude": [
                    -45,
                    45,
                ],
            },
        )
        config = DummyDatasetConfig(
            model=model,
        )
        operator = DatasetOperator(config)
        operator.ref = model
        operator.ref = model

        weights_config = MagicMock()
        weights_config.build_weights.return_value = xr.DataArray(
            np.ones(
                (
                    2,
                    2,
                )
            ),
            dims=(
                "channels",
                "latitude",
            ),
            coords={
                "channels": [
                    "tas",
                    "pr",
                ],
                "latitude": [
                    -45,
                    45,
                ],
            },
        )

        result = operator.get_weights(
            config=weights_config,
        )

        assert result.channels.values.tolist() == [
            "tas",
            "pr",
        ]

    def test_inconsistent_channel_weights_raise(self):
        model = make_data_config(
            names=[
                "tas",
                "pr",
            ],
        )
        config = DummyDatasetConfig(
            model=model,
        )
        operator = DatasetOperator(config)
        operator.ref = model
        operator.ref = model

        weights_config = MagicMock()
        weights_config.build_weights.return_value = xr.DataArray(
            np.ones(2),
            dims=("channels",),
            coords={
                "channels": [
                    "tas",
                    "wrong",
                ],
            },
        )

        with pytest.raises(
            RuntimeError,
            match="inconsistent variable weights",
        ):
            operator.get_weights(
                config=weights_config,
            )


class TestInputMetadata:
    def test_model_only(self):
        model = make_data_config(
            names=[
                "tas",
                "pr",
            ],
            spatial_coords={
                "latitude": [
                    -45,
                    45,
                ],
                "longitude": [
                    0,
                    90,
                ],
            },
        )
        model.preprocessing_pipeline.pipeline = [
            (
                "StandardScaler",
                MagicMock(),
            ),
            (
                "LogTransform",
                MagicMock(),
            ),
        ]

        config = DummyDatasetConfig(
            model=model,
        )
        operator = DatasetOperator(config)

        metadata = operator.get_input_var_metadata()

        assert metadata == {
            "variables": [
                "tas",
                "pr",
            ],
            "preprocessors": [
                [
                    "standardscaler",
                    "logtransform",
                ],
                [
                    "standardscaler",
                    "logtransform",
                ],
            ],
            "NN_dims": [
                "channels",
                "latitude",
                "longitude",
            ],
        }

    @pytest.mark.pruned
    def test_model_and_condition(self):
        model = make_data_config(
            names=["tas"],
        )
        condition = make_data_config(
            names=[
                "pr",
                "psl",
            ],
            spatial_coords={
                "latitude": [
                    -45,
                    45,
                ],
            },
        )

        config = DummyDatasetConfig(
            model=model,
            condition=condition,
            condition_method="ensemble_mean",
            using_model_as_condition=False,
        )
        operator = DatasetOperator(config)

        metadata = operator.get_input_var_metadata()

        assert metadata["variables"] == [
            "tas",
            "pr",
            "psl",
        ]
        assert metadata["NN_dims"] == [
            "channels",
            "latitude",
        ]

    @pytest.mark.pruned
    def test_model_used_as_condition_is_not_duplicated(self):
        model = make_data_config(
            names=["tas"],
        )
        effective_condition = make_data_config(
            names=["tas"],
            spatial_coords={
                "longitude": [
                    0,
                    90,
                ],
            },
        )

        config = DummyDatasetConfig(
            model=model,
            condition=effective_condition,
            condition_method="ensemble_mean",
            using_model_as_condition=True,
        )
        operator = DatasetOperator(config)

        metadata = operator.get_input_var_metadata()

        assert metadata["variables"] == [
            "tas",
        ]
        assert metadata["NN_dims"] == [
            "channels",
            "longitude",
        ]

    @pytest.mark.pruned
    def test_preprocessor_names_are_lowercase(self):
        model = make_data_config(
            names=["tas"],
        )
        model.preprocessing_pipeline.pipeline = [
            (
                "StandardScaler",
                MagicMock(),
            ),
            (
                "FlattenNaNRemove",
                MagicMock(),
            ),
        ]

        config = DummyDatasetConfig(
            model=model,
        )
        operator = DatasetOperator(config)

        metadata = operator.get_input_var_metadata()

        assert metadata["preprocessors"] == [
            [
                "standardscaler",
                "flattennanremove",
            ]
        ]


class TestTargetMetadata:
    @pytest.mark.pruned
    def test_uses_observation_before_model(self):
        model = make_data_config(
            names=["model_var"],
        )
        observation = make_data_config(
            names=["obs_var"],
            spatial_coords={
                "latitude": [
                    -45,
                    45,
                ],
            },
        )

        config = DummyDatasetConfig(
            model=model,
            observation=observation,
        )
        operator = DatasetOperator(config)

        metadata = operator.get_target_var_metadata()

        assert metadata["variables"] == [
            "obs_var",
        ]
        assert metadata["NN_dims"] == [
            "channels",
            "latitude",
        ]

    def test_uses_model_without_observation(self):
        model = make_data_config(
            names=["model_var"],
            spatial_coords={
                "longitude": [
                    0,
                    90,
                ],
            },
        )

        config = DummyDatasetConfig(
            model=model,
            observation=None,
        )
        operator = DatasetOperator(config)

        metadata = operator.get_target_var_metadata()

        assert metadata["variables"] == [
            "model_var",
        ]
        assert metadata["NN_dims"] == [
            "channels",
            "longitude",
        ]

    @pytest.mark.pruned
    def test_rejects_missing_model_and_observation(self):
        config = DummyDatasetConfig()
        operator = DatasetOperator(config)

        with pytest.raises(
            ValueError,
            match="target variable metadata could not be generated",
        ):
            operator.get_target_var_metadata()


class TestUpdateMetadata:
    @pytest.mark.pruned
    def test_one_preprocessor_list_is_added_per_variable(self):
        source = make_data_config(
            names=[
                "tas",
                "pr",
                "psl",
            ]
        )
        source.preprocessing_pipeline.pipeline = [
            (
                "StandardScaler",
                MagicMock(),
            ),
        ]

        operator = DatasetOperator(
            DummyDatasetConfig(
                model=source,
            )
        )

        metadata = {
            "variables": [],
            "preprocessors": [],
        }

        result = operator._update_metadata_with_dataconfig_metadata(
            metadata,
            source,
        )

        assert result is metadata
        assert result["variables"] == [
            "tas",
            "pr",
            "psl",
        ]
        assert result["preprocessors"] == [
            ["standardscaler"],
            ["standardscaler"],
            ["standardscaler"],
        ]

    @pytest.mark.pruned
    def test_empty_pipeline(self):
        source = make_data_config(
            names=["tas"],
        )
        source.preprocessing_pipeline.pipeline = []

        operator = DatasetOperator(
            DummyDatasetConfig(
                model=source,
            )
        )

        result = operator._update_metadata_with_dataconfig_metadata(
            {
                "variables": [],
                "preprocessors": [],
            },
            source,
        )

        assert result == {
            "variables": [
                "tas",
            ],
            "preprocessors": [
                [],
            ],
        }


class TestBuildChunks:
    def test_none_config_returns_none(self):
        assert _build_chunks() is None

    @pytest.mark.pruned
    def test_builds_chunks_for_available_sample_dimensions(self):
        source = make_data_config(
            realizations=[
                0,
                1,
            ]
        )

        source.coords = source.info.coords
        source.coords = source.info.coords
        result = _build_chunks(source)

        assert result == {
            INIT_TIME_DIM: 1,
            LEAD_TIME_DIM: 1,
            REALIZATION_DIM: 1,
        }

    @pytest.mark.pruned
    def test_omits_missing_realization_dimension(self):
        source = make_data_config(
            realizations=None,
        )

        source.coords = source.info.coords
        source.coords = source.info.coords
        result = _build_chunks(source)

        assert result == {
            INIT_TIME_DIM: 1,
            LEAD_TIME_DIM: 1,
        }

    @pytest.mark.pruned
    def test_uses_configured_dimension_names(self):
        source = SimpleNamespace(
            init_time_dim="forecast_reference_time",
            lead_time_dim="step",
            realization_dim="member",
            info=SimpleNamespace(
                coords={
                    "forecast_reference_time": make_coord(
                        AVAILABLE_TIMES,
                        "forecast_reference_time",
                    ),
                    "step": make_coord(
                        [
                            1,
                            2,
                        ],
                        "step",
                    ),
                    "member": make_coord(
                        [
                            0,
                            1,
                        ],
                        "member",
                    ),
                }
            ),
        )

        source.coords = source.info.coords
        source.coords = source.info.coords
        assert _build_chunks(source) == {
            "forecast_reference_time": 1,
            "step": 1,
            "member": 1,
        }

    @pytest.mark.pruned
    def test_ignores_non_sample_dimensions(self):
        source = make_data_config(
            spatial_coords={
                "latitude": [
                    -45,
                    45,
                ],
                "longitude": [
                    0,
                    90,
                ],
            },
        )

        source.coords = source.info.coords
        source.coords = source.info.coords
        result = _build_chunks(source)

        assert "latitude" not in result
        assert "longitude" not in result


class TestAdditionalFitPreprocessors:
    def test_rejects_unavailable_training_time(self):
        model = make_data_config()
        config = DummyDatasetConfig(
            model=model,
        )
        operator = DatasetOperator(config)

        with pytest.raises(
            ValueError,
            match="train_times are unavailable",
        ):
            operator.fit_preprocessors(
                make_train_times(
                    [
                        "1999-01-01",
                    ]
                )
            )

        model.fit_preprocessor_pipeline.assert_not_called()

    @pytest.mark.pruned
    def test_validates_times_before_fitting_sources(self):
        model = make_data_config()
        config = DummyDatasetConfig(
            model=model,
        )
        operator = DatasetOperator(config)
        error = ValueError("invalid time sequence")

        with patch(
            "cccma_ppp.data_modules.dataset.operator._validate_time_sequence",
            side_effect=error,
        ) as validate:
            with pytest.raises(
                ValueError,
                match="invalid time sequence",
            ):
                operator.fit_preprocessors(make_train_times())

        validate.assert_called_once()
        model.fit_preprocessor_pipeline.assert_not_called()

    @pytest.mark.pruned
    def test_model_fit_forwards_default_save_arguments(self):
        model = make_data_config()
        config = DummyDatasetConfig(
            model=model,
        )
        operator = DatasetOperator(config)
        train_times = make_train_times()

        operator.fit_preprocessors(
            train_times,
        )

        kwargs = model.fit_preprocessor_pipeline.call_args.kwargs

        assert kwargs["mask"] is True
        assert kwargs["save"] is False
        assert kwargs["save_path"] is None
        assert kwargs["save_name"] is None
        assert kwargs["selection"][INIT_TIME_DIM] is train_times
        assert kwargs["selection"][LEAD_TIME_DIM] is model.coords[LEAD_TIME_DIM]

    @pytest.mark.pruned
    def test_model_fit_includes_realization_coordinate(self):
        model = make_data_config(
            realizations=[
                2,
                4,
            ],
        )
        config = DummyDatasetConfig(
            model=model,
        )
        operator = DatasetOperator(config)

        operator.fit_preprocessors(make_train_times())

        selection = model.fit_preprocessor_pipeline.call_args.kwargs["selection"]

        assert selection[REALIZATION_DIM] is model.coords[REALIZATION_DIM]

    @pytest.mark.pruned
    def test_observation_fit_does_not_request_mask(self):
        observation = make_data_config()
        config = DummyDatasetConfig(
            observation=observation,
        )
        operator = DatasetOperator(config)
        train_times = make_train_times()

        operator.fit_preprocessors(
            train_times,
        )

        kwargs = observation.fit_preprocessor_pipeline.call_args.kwargs

        assert "mask" not in kwargs
        assert kwargs["selection"] == {
            INIT_TIME_DIM: train_times,
        }

    def test_observation_fit_includes_realization_coordinate(self):
        observation = make_data_config(
            realizations=[
                0,
                1,
            ],
        )
        config = DummyDatasetConfig(
            observation=observation,
        )
        operator = DatasetOperator(config)
        train_times = make_train_times()

        operator.fit_preprocessors(
            train_times,
        )

        selection = observation.fit_preprocessor_pipeline.call_args.kwargs["selection"]

        assert selection[INIT_TIME_DIM] is train_times
        assert selection[REALIZATION_DIM] is observation.coords[REALIZATION_DIM]

    @pytest.mark.pruned
    def test_dynamic_condition_fit_omits_realization_when_absent(
        self,
    ):
        condition = make_data_config(
            realizations=None,
        )
        config = DummyDatasetConfig(
            condition=condition,
            condition_method="ensemble_mean",
        )
        operator = DatasetOperator(config)

        operator.fit_preprocessors(make_train_times())

        selection = condition.fit_preprocessor_pipeline.call_args.kwargs["selection"]

        assert REALIZATION_DIM not in selection

    @pytest.mark.pruned
    def test_dynamic_condition_fit_includes_realization(self):
        condition = make_data_config(
            realizations=[
                3,
                5,
            ],
        )
        config = DummyDatasetConfig(
            condition=condition,
            condition_method="cross_ensemble",
        )
        operator = DatasetOperator(config)

        operator.fit_preprocessors(make_train_times())

        selection = condition.fit_preprocessor_pipeline.call_args.kwargs["selection"]

        assert selection[REALIZATION_DIM] is condition.coords[REALIZATION_DIM]

    @pytest.mark.pruned
    def test_static_condition_does_not_call_get_input_times(self):
        condition = make_data_config()
        config = DummyDatasetConfig(
            condition=condition,
            condition_method="static",
        )
        config.get_input_times = MagicMock()
        operator = DatasetOperator(config)

        operator.fit_preprocessors(make_train_times())

        config.get_input_times.assert_not_called()
        condition.fit_preprocessor_pipeline.assert_called_once_with(
            selection={},
            mask=True,
            save=False,
            save_path=None,
            save_name=None,
        )

    def test_model_and_condition_each_use_input_time_selection(self):
        model = make_data_config()
        condition = make_data_config()
        config = DummyDatasetConfig(
            model=model,
            condition=condition,
            condition_method="ensemble_mean",
        )
        selected = make_train_times(
            [
                "2000-01-01",
            ]
        )
        config.get_input_times = MagicMock(
            return_value=selected,
        )
        operator = DatasetOperator(config)

        operator.fit_preprocessors(make_train_times())

        assert config.get_input_times.call_count == 2
        assert (
            model.fit_preprocessor_pipeline.call_args.kwargs["selection"][INIT_TIME_DIM]
            is selected
        )
        assert (
            condition.fit_preprocessor_pipeline.call_args.kwargs["selection"][
                INIT_TIME_DIM
            ]
            is selected
        )

    @pytest.mark.pruned
    def test_fitted_flag_is_false_when_no_pipelines_exist(self):
        config = DummyDatasetConfig()
        operator = DatasetOperator(config)

        operator.fit_preprocessors(make_train_times())

        assert config._fitted_preprocessors is False

    @pytest.mark.pruned
    def test_fitted_flag_reflects_pipeline_states(self):
        model_pipeline = make_pipeline()
        model_pipeline.fitted = True
        observation_pipeline = make_pipeline()
        observation_pipeline.fitted = False

        model = make_data_config(
            pipeline=model_pipeline,
        )
        observation = make_data_config(
            pipeline=observation_pipeline,
        )
        config = DummyDatasetConfig(
            model=model,
            observation=observation,
        )
        operator = DatasetOperator(config)

        operator.fit_preprocessors(make_train_times())

        assert config._fitted_preprocessors is False


class TestAdditionalLoadFittedPreprocessors:
    def test_loads_each_active_source(self):
        model = make_data_config()
        observation = make_data_config()
        condition = make_data_config()

        model.preprocessing_pipeline.fitted = True
        observation.preprocessing_pipeline.fitted = True
        condition.preprocessing_pipeline.fitted = True

        config = DummyDatasetConfig(
            model=model,
            observation=observation,
            condition=condition,
            condition_method="ensemble_mean",
        )
        operator = DatasetOperator(config)
        load_dir = Path("/tmp/pipelines")

        operator.load_fitted_preprocessors(
            load_dir,
        )

        model.load_preprocessor_pipeline.assert_called_once_with(load_dir)
        observation.load_preprocessor_pipeline.assert_called_once_with(load_dir)
        condition.load_preprocessor_pipeline.assert_called_once_with(load_dir)
        assert config._fitted_preprocessors is True

    @pytest.mark.pruned
    def test_load_forwards_none_directory(self):
        model = make_data_config()
        model.preprocessing_pipeline.fitted = True
        config = DummyDatasetConfig(
            model=model,
        )
        operator = DatasetOperator(config)

        operator.load_fitted_preprocessors()

        model.load_preprocessor_pipeline.assert_called_once_with(None)

    def test_load_with_no_sources_sets_fitted_flag_false(self):
        config = DummyDatasetConfig()
        operator = DatasetOperator(config)

        operator.load_fitted_preprocessors()

        assert config._fitted_preprocessors is False

    @pytest.mark.pruned
    def test_load_fitted_flag_is_false_when_one_pipeline_is_unfitted(
        self,
    ):
        model = make_data_config()
        condition = make_data_config()

        model.preprocessing_pipeline.fitted = True
        condition.preprocessing_pipeline.fitted = False

        config = DummyDatasetConfig(
            model=model,
            condition=condition,
            condition_method="ensemble_mean",
        )
        operator = DatasetOperator(config)

        operator.load_fitted_preprocessors("/tmp/pipelines")

        assert config._fitted_preprocessors is False


class TestAdditionalAddFittedPreprocessor:
    @pytest.mark.pruned
    def test_default_index_is_zero(self):
        model = make_data_config()
        config = DummyDatasetConfig(
            model=model,
        )
        operator = DatasetOperator(config)
        preprocessor = DummyPreprocessor(
            fitted=True,
        )

        operator.add_fitted_preprocessor(preprocessor)

        (
            model.preprocessing_pipeline.add_fitted_preprocessor.assert_called_once_with(
                preprocessor,
                index=0,
            )
        )

    def test_only_observation_pipeline_is_updated(self):
        observation = make_data_config()
        config = DummyDatasetConfig(
            observation=observation,
        )
        operator = DatasetOperator(config)
        preprocessor = DummyPreprocessor(
            fitted=True,
        )

        operator.add_fitted_preprocessor(
            preprocessor,
            index=3,
        )

        (
            observation.preprocessing_pipeline.add_fitted_preprocessor.assert_called_once_with(
                preprocessor,
                index=3,
            )
        )

    def test_only_condition_pipeline_is_updated(self):
        condition = make_data_config()
        config = DummyDatasetConfig(
            condition=condition,
            condition_method="static",
        )
        operator = DatasetOperator(config)
        preprocessor = DummyPreprocessor(
            fitted=True,
        )

        operator.add_fitted_preprocessor(
            preprocessor,
            index=1,
        )

        (
            condition.preprocessing_pipeline.add_fitted_preprocessor.assert_called_once_with(
                preprocessor,
                index=1,
            )
        )

    @pytest.mark.pruned
    def test_unfitted_preprocessor_is_not_added_anywhere(self):
        model = make_data_config()
        observation = make_data_config()
        condition = make_data_config()
        config = DummyDatasetConfig(
            model=model,
            observation=observation,
            condition=condition,
            condition_method="ensemble_mean",
        )
        operator = DatasetOperator(config)
        preprocessor = DummyPreprocessor(
            fitted=False,
        )

        with pytest.raises(
            ValueError,
            match="must be fitted",
        ):
            operator.add_fitted_preprocessor(preprocessor)

        for source in (
            model,
            observation,
            condition,
        ):
            (source.preprocessing_pipeline.add_fitted_preprocessor.assert_not_called())


class TestAdditionalGetWeights:
    def test_none_configuration_builds_default_weights_config(
        self,
        monkeypatch,
    ):
        model = make_data_config(
            spatial_coords={
                "latitude": [
                    -45,
                    45,
                ],
            },
        )
        model.preprocessing_pipeline.has_flattener = False

        config = DummyDatasetConfig(
            model=model,
        )
        operator = DatasetOperator(config)

        expected = xr.DataArray(
            np.ones(2),
            dims=("latitude",),
        )
        weights_config = MagicMock()
        weights_config.build_weights.return_value = expected
        constructor = MagicMock(
            return_value=weights_config,
        )

        monkeypatch.setattr(
            "cccma_ppp.data_modules.dataset.operator.WeightsConfig",
            constructor,
        )

        result = operator.get_weights(
            config=None,
            save=False,
        )

        constructor.assert_called_once_with()
        assert result is expected

    @pytest.mark.pruned
    def test_empty_spatial_coordinates_are_forwarded(self):
        model = make_data_config()
        model.preprocessing_pipeline.has_flattener = False

        config = DummyDatasetConfig(
            model=model,
        )
        operator = DatasetOperator(config)

        weights_config = MagicMock()
        expected = xr.DataArray(
            1.0,
        )
        weights_config.build_weights.return_value = expected

        result = operator.get_weights(
            config=weights_config,
            save=False,
        )

        assert result is expected
        assert weights_config.build_weights.call_args.args[0] == {}

    @pytest.mark.pruned
    def test_unsupported_coordinates_are_not_forwarded(self):
        model = make_data_config(
            spatial_coords={
                "latitude": [
                    -45,
                    45,
                ],
                "height": [
                    1000,
                    500,
                ],
            },
        )
        model.preprocessing_pipeline.has_flattener = False

        config = DummyDatasetConfig(
            model=model,
        )
        operator = DatasetOperator(config)

        weights_config = MagicMock()
        weights_config.build_weights.return_value = xr.DataArray(
            np.ones(2),
            dims=("latitude",),
        )

        operator.get_weights(
            config=weights_config,
        )

        target_coords = weights_config.build_weights.call_args.args[0]

        assert "latitude" in target_coords
        assert "height" not in target_coords

    @pytest.mark.pruned
    def test_flattener_is_not_requested_when_pipeline_has_none(
        self,
    ):
        model = make_data_config(
            spatial_coords={
                "latitude": [
                    -45,
                    45,
                ],
            },
        )
        model.preprocessing_pipeline.has_flattener = False
        model.preprocessing_pipeline.get_flattener.reset_mock()

        config = DummyDatasetConfig(
            model=model,
        )
        operator = DatasetOperator(config)

        weights_config = MagicMock()
        weights_config.build_weights.return_value = xr.DataArray(
            np.ones(2),
            dims=("latitude",),
        )

        operator.get_weights(
            config=weights_config,
        )

        (model.preprocessing_pipeline.get_flattener.assert_not_called())
        assert (
            weights_config.build_weights.call_args.kwargs["Flattennanremover"] is None
        )

    @pytest.mark.pruned
    def test_flattener_is_forwarded_when_present(self):
        flattener = object()
        pipeline = make_pipeline()
        pipeline.has_flattener = True
        pipeline.get_flattener.return_value = flattener

        model = make_data_config(
            pipeline=pipeline,
            spatial_coords={
                "latitude": [
                    -45,
                    45,
                ],
            },
        )
        config = DummyDatasetConfig(
            model=model,
        )
        operator = DatasetOperator(config)

        weights_config = MagicMock()
        weights_config.build_weights.return_value = xr.DataArray(
            np.ones(2),
            dims=("latitude",),
        )

        operator.get_weights(
            config=weights_config,
        )

        pipeline.get_flattener.assert_called_once_with()
        assert (
            weights_config.build_weights.call_args.kwargs["Flattennanremover"]
            is flattener
        )

    @pytest.mark.pruned
    def test_channel_order_mismatch_raises(self):
        model = make_data_config(
            names=[
                "tas",
                "pr",
            ],
        )
        model.preprocessing_pipeline.has_flattener = False
        config = DummyDatasetConfig(
            model=model,
        )
        operator = DatasetOperator(config)

        weights_config = MagicMock()
        weights_config.build_weights.return_value = xr.DataArray(
            np.ones(2),
            dims=("channels",),
            coords={
                "channels": [
                    "pr",
                    "tas",
                ],
            },
        )

        with pytest.raises(
            RuntimeError,
            match="inconsistent variable weights",
        ):
            operator.get_weights(
                config=weights_config,
            )

    @pytest.mark.pruned
    def test_weights_without_channel_dimension_are_accepted(self):
        model = make_data_config(
            names=[
                "tas",
                "pr",
            ],
            spatial_coords={
                "latitude": [
                    -45,
                    45,
                ],
            },
        )
        model.preprocessing_pipeline.has_flattener = False
        config = DummyDatasetConfig(
            model=model,
        )
        operator = DatasetOperator(config)

        expected = xr.DataArray(
            np.ones(2),
            dims=("latitude",),
        )
        weights_config = MagicMock()
        weights_config.build_weights.return_value = expected

        result = operator.get_weights(
            config=weights_config,
        )

        assert result is expected


class TestAdditionalInputMetadata:
    @pytest.mark.pruned
    def test_condition_only_with_model_as_condition(self):
        condition = make_data_config(
            names=[
                "zg",
                "psl",
            ],
            spatial_coords={
                "latitude": [
                    -45,
                    45,
                ],
            },
        )
        config = DummyDatasetConfig(
            model=None,
            condition=condition,
            condition_method="ensemble_mean",
            using_model_as_condition=True,
        )
        operator = DatasetOperator(config)

        metadata = operator.get_input_var_metadata()

        assert metadata["variables"] == [
            "zg",
            "psl",
        ]
        assert metadata["NN_dims"] == [
            "channels",
            "latitude",
        ]

    def test_condition_dimensions_determine_input_dimensions(self):
        model = make_data_config(
            spatial_coords={
                "longitude": [
                    0,
                    90,
                ],
            },
        )
        condition = make_data_config(
            names=[
                "condition",
            ],
            spatial_coords={
                "latitude": [
                    -45,
                    45,
                ],
            },
        )
        config = DummyDatasetConfig(
            model=model,
            condition=condition,
            condition_method="ensemble_mean",
            using_model_as_condition=False,
        )
        operator = DatasetOperator(config)

        metadata = operator.get_input_var_metadata()

        assert metadata["NN_dims"] == [
            "channels",
            "latitude",
        ]

    @pytest.mark.pruned
    def test_shared_preprocessor_list_is_recorded_per_variable(
        self,
    ):
        model = make_data_config(
            names=[
                "tas",
                "pr",
            ],
        )
        model.preprocessing_pipeline.pipeline = [
            (
                "Normalize",
                object(),
            )
        ]

        config = DummyDatasetConfig(
            model=model,
        )
        operator = DatasetOperator(config)

        metadata = operator.get_input_var_metadata()

        assert metadata["preprocessors"] == [
            [
                "normalize",
            ],
            [
                "normalize",
            ],
        ]


class TestAdditionalTargetMetadata:
    @pytest.mark.pruned
    def test_observation_preprocessors_are_used(self):
        observation = make_data_config(
            names=[
                "tas",
                "pr",
            ],
        )
        observation.preprocessing_pipeline.pipeline = [
            (
                "Standardizer",
                object(),
            ),
            (
                "Clip",
                object(),
            ),
        ]

        config = DummyDatasetConfig(
            model=make_data_config(
                names=[
                    "model",
                ],
            ),
            observation=observation,
        )
        operator = DatasetOperator(config)

        metadata = operator.get_target_var_metadata()

        assert metadata["variables"] == [
            "tas",
            "pr",
        ]
        assert metadata["preprocessors"] == [
            [
                "standardizer",
                "clip",
            ],
            [
                "standardizer",
                "clip",
            ],
        ]

    def test_target_dimensions_follow_supported_order(self):
        observation = make_data_config(
            spatial_coords={
                "longitude": [
                    0,
                    90,
                ],
                "latitude": [
                    -45,
                    45,
                ],
            },
        )
        config = DummyDatasetConfig(
            observation=observation,
        )
        operator = DatasetOperator(config)

        metadata = operator.get_target_var_metadata()

        assert metadata["NN_dims"] == [
            "channels",
            "latitude",
            "longitude",
        ]


class TestAdditionalMetadataUpdate:
    @pytest.mark.pruned
    def test_existing_metadata_entries_are_preserved(self):
        source = make_data_config(
            names=[
                "pr",
            ],
        )
        source.preprocessing_pipeline.pipeline = [
            (
                "LogTransform",
                object(),
            )
        ]

        metadata = {
            "variables": [
                "tas",
            ],
            "preprocessors": [
                [
                    "standardizer",
                ]
            ],
        }
        operator = DatasetOperator(
            DummyDatasetConfig(
                model=source,
            )
        )

        result = operator._update_metadata_with_dataconfig_metadata(
            metadata,
            source,
        )

        assert result is metadata
        assert result == {
            "variables": [
                "tas",
                "pr",
            ],
            "preprocessors": [
                [
                    "standardizer",
                ],
                [
                    "logtransform",
                ],
            ],
        }

    @pytest.mark.pruned
    def test_preprocessor_names_are_lowercased(self):
        source = make_data_config(
            names=[
                "tas",
            ],
        )
        source.preprocessing_pipeline.pipeline = [
            (
                "MixedCASEPreProcessor",
                object(),
            )
        ]
        operator = DatasetOperator(
            DummyDatasetConfig(
                model=source,
            )
        )

        metadata = {
            "variables": [],
            "preprocessors": [],
        }
        operator._update_metadata_with_dataconfig_metadata(
            metadata,
            source,
        )

        assert metadata["preprocessors"] == [
            [
                "mixedcasepreprocessor",
            ]
        ]


class TestAdditionalFittedPreprocessorsProperty:
    @pytest.mark.pruned
    def test_no_active_pipelines_returns_false(self):
        operator = DatasetOperator(DummyDatasetConfig())

        assert operator._fitted_preprocessors is False

    @pytest.mark.pruned
    def test_single_fitted_model_pipeline_returns_true(self):
        model = make_data_config()
        model.preprocessing_pipeline.fitted = True

        operator = DatasetOperator(
            DummyDatasetConfig(
                model=model,
            )
        )

        assert operator._fitted_preprocessors is True

    @pytest.mark.pruned
    def test_single_unfitted_model_pipeline_returns_false(self):
        model = make_data_config()
        model.preprocessing_pipeline.fitted = False

        operator = DatasetOperator(
            DummyDatasetConfig(
                model=model,
            )
        )

        assert operator._fitted_preprocessors is False

    @pytest.mark.pruned
    def test_all_active_pipelines_must_be_fitted(self):
        model = make_data_config()
        observation = make_data_config()
        condition = make_data_config()

        model.preprocessing_pipeline.fitted = True
        observation.preprocessing_pipeline.fitted = True
        condition.preprocessing_pipeline.fitted = False

        operator = DatasetOperator(
            DummyDatasetConfig(
                model=model,
                observation=observation,
                condition=condition,
                condition_method="ensemble_mean",
            )
        )

        assert operator._fitted_preprocessors is False

    @pytest.mark.pruned
    def test_all_fitted_active_pipelines_return_true(self):
        model = make_data_config()
        observation = make_data_config()
        condition = make_data_config()

        model.preprocessing_pipeline.fitted = True
        observation.preprocessing_pipeline.fitted = True
        condition.preprocessing_pipeline.fitted = True

        operator = DatasetOperator(
            DummyDatasetConfig(
                model=model,
                observation=observation,
                condition=condition,
                condition_method="ensemble_mean",
            )
        )

        assert operator._fitted_preprocessors is True


class TestAdditionalBuildChunks:
    @pytest.mark.pruned
    def test_only_initialization_dimension(self):
        source = SimpleNamespace(
            init_time_dim="initialization",
            lead_time_dim="forecast_step",
            realization_dim="member",
            coords={
                "initialization": make_coord(
                    AVAILABLE_TIMES,
                    "initialization",
                ),
                "latitude": make_coord(
                    [
                        -45,
                        45,
                    ],
                    "latitude",
                ),
            },
        )

        assert _build_chunks(source) == {
            "initialization": 1,
        }

    @pytest.mark.pruned
    def test_only_lead_time_dimension(self):
        source = SimpleNamespace(
            init_time_dim="initialization",
            lead_time_dim="forecast_step",
            realization_dim="member",
            coords={
                "forecast_step": make_coord(
                    [
                        1,
                        2,
                    ],
                    "forecast_step",
                ),
            },
        )

        assert _build_chunks(source) == {
            "forecast_step": 1,
        }

    @pytest.mark.pruned
    def test_only_realization_dimension(self):
        source = SimpleNamespace(
            init_time_dim="initialization",
            lead_time_dim="forecast_step",
            realization_dim="member",
            coords={
                "member": make_coord(
                    [
                        0,
                        1,
                    ],
                    "member",
                ),
            },
        )

        assert _build_chunks(source) == {
            "member": 1,
        }

    @pytest.mark.pruned
    def test_no_sample_dimensions_returns_empty_mapping(self):
        source = SimpleNamespace(
            init_time_dim="initialization",
            lead_time_dim="forecast_step",
            realization_dim="member",
            coords={
                "latitude": make_coord(
                    [
                        -45,
                        45,
                    ],
                    "latitude",
                ),
            },
        )

        assert _build_chunks(source) == {}

    def test_chunk_values_are_always_one(self):
        source = make_data_config(
            realizations=[
                0,
                1,
                2,
            ],
        )

        chunks = _build_chunks(source)

        assert chunks
        assert set(chunks.values()) == {
            1,
        }
