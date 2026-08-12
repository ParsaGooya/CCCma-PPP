from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from cccma_ppp.data_modules.dataset.operator import (
    DatasetOperator,
    _build_chunks,
)
from cccma_ppp.preprocessing.preprocessing_ABC import PreprocessModuleABC


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


class TestFitPreprocessors:
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

    @pytest.mark.pruned
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

    @pytest.mark.pruned
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

    @pytest.mark.pruned
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

    @pytest.mark.pruned
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

    @pytest.mark.pruned
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

    @pytest.mark.pruned
    def test_skips_missing_sources(self):
        config = DummyDatasetConfig(
            model=None,
            observation=None,
            condition=None,
        )
        operator = DatasetOperator(config)

        operator.fit_preprocessors(make_train_times())

        assert config._fitted_preprocessors is True

    @pytest.mark.pruned
    def test_sets_fitted_preprocessors_flag(self):
        config = DummyDatasetConfig(
            model=make_data_config(),
        )
        operator = DatasetOperator(config)

        assert config._fitted_preprocessors is False

        operator.fit_preprocessors(make_train_times())

        assert config._fitted_preprocessors is True


class TestLoadFittedPreprocessors:
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

    @pytest.mark.pruned
    def test_forwards_none_load_directory(self):
        model = make_data_config()
        config = DummyDatasetConfig(
            model=model,
        )
        operator = DatasetOperator(config)

        operator.load_fitted_preprocessors()

        model.load_preprocessor_pipeline.assert_called_once_with(None)


class TestAddFittedPreprocessor:
    @pytest.mark.pruned
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

    @pytest.mark.pruned
    def test_rejects_unfitted_preprocessor(self):
        config = DummyDatasetConfig(
            model=make_data_config(),
        )
        operator = DatasetOperator(config)
        preprocessor = DummyPreprocessor(
            fitted=False,
        )

        with pytest.raises(
            AssertionError,
            match="must be fitted",
        ):
            operator.add_fitted_preprocessor(preprocessor)

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
        from cccma_ppp.preprocessing.utils_preprocessing import (
            Flattennanremove,
        )

        flattener = MagicMock(spec=Flattennanremove)

        pipeline = make_pipeline(
            fitted_preprocessors=[
                flattener,
            ]
        )
        pipeline.get_preprocessors.return_value = flattener

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

        pipeline.get_preprocessors.assert_called_once_with("flattener")
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
