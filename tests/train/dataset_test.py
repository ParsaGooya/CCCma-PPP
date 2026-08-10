                             

from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch
import warnings

import cftime
import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

import cccma_ppp.train.dataset as module
from cccma_ppp.train.dataset import (
    TrainDataset,
    TrainDatasetConfig,
)


INIT_TIME_DIM = "time"
LEAD_TIME_DIM = "lead_time"
REALIZATION_DIM = "ensembles"


def make_datetime_index(*years):
    return pd.DatetimeIndex([f"{year}-01-01" for year in years])


def make_coord(
    values,
    *,
    dim,
):
    return xr.DataArray(
        values,
        dims=(dim,),
        coords={
            dim: values,
        },
    )


def make_data_config(
    *,
    name="model",
    times=None,
    time_range=None,
    init_time_freq="year",
    time_coords_type="datetime",
    spatial_coords=None,
    names=None,
    preprocessing_pipeline=None,
):
    if times is None:
        times = make_datetime_index(
            2000,
            2001,
            2002,
        )

    if time_range is None:
        time_range = times

    coords = {
        INIT_TIME_DIM: make_coord(
            times,
            dim=INIT_TIME_DIM,
        ),
    }

    if spatial_coords is not None:
        coords.update(spatial_coords)

    if names is None:
        names = [
            name,
        ]

    if preprocessing_pipeline is None:
        preprocessing_pipeline = SimpleNamespace(
            fitted_preprocessors=[],
            transform=Mock(side_effect=lambda value: value),
            get_preprocessors=Mock(),
        )

    return SimpleNamespace(
        name=name,
        names=names,
        info=SimpleNamespace(
            coords=coords,
            init_time_freq=init_time_freq,
            time_coords_type=time_coords_type,
        ),
        time_range=pd.DatetimeIndex(time_range),
        preprocessing_pipeline=preprocessing_pipeline,
    )


class StubTrainDatasetConfig(TrainDatasetConfig):

    @property
    def _using_model_data_as_condition(self):
        return self.__using_model_data_as_condition

    @_using_model_data_as_condition.setter
    def _using_model_data_as_condition(self, value):
        self.__using_model_data_as_condition = value

    @property
    def effective_condition(self):
        return self.__effective_condition

    @effective_condition.setter
    def effective_condition(self, value):
        self.__effective_condition = value



def make_config(
    *,
    model=None,
    observation=None,
    condition=None,
    condition_method="static",
    using_model_as_condition=False,
    effective_condition=None,
):
    if model is None:
        model = make_data_config()

    config = object.__new__(StubTrainDatasetConfig)

    config.model = model
    config.observation = observation
    config.condition = condition
    config.condition_method = condition_method
    config.lead_times = None

    config.init_time_dim = INIT_TIME_DIM
    config.lead_time_dim = LEAD_TIME_DIM
    config.realization_dim = REALIZATION_DIM
    config.supported_NN_dimensions = (
        "lat",
        "lon",
    )

    config._using_model_data_as_condition = using_model_as_condition
    config.effective_condition = effective_condition

    return config


def make_dataset(
    *,
    config=None,
    requested_times=None,
    time_features=None,
    return_metadata=False,
    load=False,
):
    if config is None:
        config = make_config()

    if requested_times is None:
        requested_times = np.asarray(
            [
                "2000-01-01",
            ],
            dtype="datetime64[ns]",
        )

    if time_features is None:
        time_features = Mock(return_value=None)

    dataset = object.__new__(TrainDataset)

    dataset.config = config
    dataset.requested_times = requested_times
    dataset.time_features = time_features
    dataset.mask = None
    dataset.return_metadata = return_metadata
    dataset.load = load

    dataset.observation_dataset = None
    dataset.model_dataset = None
    dataset.condition_dataset = None
    dataset.obs_indexes = None
    dataset.model_indexes = None
    dataset.cond_indexes = None
    dataset.sample_coords = {}

    return dataset


def make_indexed_data(
    values,
    *,
    times=None,
    lead_times=None,
    realizations=None,
    variable_name="tas",
):
    if times is None:
        times = np.asarray(
            [
                "2000-01-01",
                "2000-02-01",
            ],
            dtype="datetime64[ns]",
        )

    if lead_times is None:
        lead_times = [
            1,
            2,
        ]

    dims = [
        INIT_TIME_DIM,
        LEAD_TIME_DIM,
    ]
    coords = {
        INIT_TIME_DIM: times,
        LEAD_TIME_DIM: lead_times,
    }

    if realizations is not None:
        dims.append(REALIZATION_DIM)
        coords[REALIZATION_DIM] = realizations

    data = xr.DataArray(
        values,
        dims=tuple(dims),
        coords=coords,
        name=variable_name,
    )

    return data.to_dataset(name=variable_name)


class TestTrainDatasetConfigObservationChecks:
    def test_observation_none_requires_condition_method(self):
        config = make_config(
            observation=None,
            condition_method=None,
        )

        with pytest.raises(
            ValueError,
            match=("No target observation is specified. Specify condition_method"),
        ):
            config._check_observation()

    def test_observation_none_accepts_condition_method(self):
        config = make_config(
            observation=None,
            condition_method="static",
        )

        result = config._check_observation()

        assert result is config

    def test_matching_spatial_coordinates_are_accepted(self):
        lat = make_coord(
            [
                45.0,
                46.0,
            ],
            dim="lat",
        )
        lon = make_coord(
            [
                -124.0,
                -123.0,
            ],
            dim="lon",
        )

        model = make_data_config(
            spatial_coords={
                "lat": lat,
                "lon": lon,
            }
        )
        observation = make_data_config(
            name="observation",
            spatial_coords={
                "lat": lat.copy(),
                "lon": lon.copy(),
            },
        )

        config = make_config(
            model=model,
            observation=observation,
        )

        with warnings_capture() as caught:
            result = config._check_observation()

        assert result is config
        assert caught == []

    @pytest.mark.parametrize(
        "dimension",
        [
            "lat",
            "lon",
        ],
    )
    def test_coordinate_mismatch_warns(
        self,
        dimension,
    ):
        model_coord = make_coord(
            [
                0.0,
                1.0,
            ],
            dim=dimension,
        )
        observation_coord = make_coord(
            [
                0.0,
                2.0,
            ],
            dim=dimension,
        )

        model = make_data_config(
            spatial_coords={
                dimension: model_coord,
            }
        )
        observation = make_data_config(
            name="observation",
            spatial_coords={
                dimension: observation_coord,
            },
        )

        config = make_config(
            model=model,
            observation=observation,
        )

        with pytest.warns(
            UserWarning,
            match=(f"do not have the same {dimension}"),
        ):
            config._check_observation()

    @pytest.mark.parametrize(
        "dimension",
        [
            "lat",
            "lon",
        ],
    )
    def test_observation_dimension_missing_from_model_warns(
        self,
        dimension,
    ):
        model = make_data_config()
        observation = make_data_config(
            name="observation",
            spatial_coords={
                dimension: make_coord(
                    [
                        0.0,
                        1.0,
                    ],
                    dim=dimension,
                )
            },
        )

        config = make_config(
            model=model,
            observation=observation,
        )

        with pytest.warns(
            UserWarning,
            match=(f"NN dim {dimension}.*not present"),
        ):
            config._check_observation()

    def test_time_coordinate_type_mismatch_raises(self):
        model = make_data_config(
            time_coords_type="datetime",
            spatial_coords={
                "lat": make_coord(
                    [
                        45.0,
                    ],
                    dim="lat",
                )
            },
        )
        observation = make_data_config(
            name="observation",
            time_coords_type="cftime",
            spatial_coords={
                "lat": make_coord(
                    [
                        45.0,
                    ],
                    dim="lat",
                )
            },
        )

        config = make_config(
            model=model,
            observation=observation,
        )

        with pytest.raises(
            ValueError,
            match=("must have the same.*time coordinates"),
        ):
            config._check_observation()


class TestTrainDatasetConfigProperties:
    def test_effective_input_is_model(self):
        model = make_data_config()
        config = make_config(model=model)

        assert config.effective_input is model

    def test_dataset_operator_is_constructed(self):
        config = make_config()
        operator = object()

        with patch.object(
            module,
            "DatasetOperator",
            return_value=operator,
        ) as constructor:
            result = config.ds_operator

        assert result is operator
        constructor.assert_called_once_with(config)

    def test_common_time_without_observation(self):
        model = make_data_config(
            time_range=make_datetime_index(
                2000,
                2001,
                2002,
            )
        )
        config = make_config(
            model=model,
            observation=None,
        )

        result = config.get_common_time

        assert result.equals(model.time_range)

    def test_common_time_intersects_observation(self):
        model = make_data_config(
            time_range=make_datetime_index(
                2000,
                2001,
                2002,
            )
        )
        observation = make_data_config(
            name="observation",
            time_range=make_datetime_index(
                2001,
                2002,
                2003,
            ),
        )

        config = make_config(
            model=model,
            observation=observation,
        )

        result = config.get_common_time

        assert result.equals(
            make_datetime_index(
                2001,
                2002,
            )
        )

    def test_available_times_for_yearly_initializations(self):
        model = make_data_config(
            times=make_datetime_index(
                2001,
                2002,
            ),
            time_range=make_datetime_index(
                2000,
                2001,
                2002,
                2003,
            ),
            init_time_freq="year",
        )

        config = make_config(
            model=model,
            observation=None,
        )

        result = config.available_times

        assert result.equals(
            make_datetime_index(
                2001,
                2002,
            )
        )

    def test_available_times_for_non_yearly_initializations(self):
        model_times = pd.DatetimeIndex(
            [
                "2000-02-01",
                "2000-03-01",
            ]
        )
        common_times = pd.DatetimeIndex(
            [
                "2000-01-01",
                "2000-02-01",
                "2000-03-01",
                "2000-04-01",
            ]
        )

        model = make_data_config(
            times=model_times,
            time_range=common_times,
            init_time_freq="month",
        )
        config = make_config(
            model=model,
        )

        result = config.available_times

        assert result.equals(model_times)


class TestTrainDatasetConfigDelegation:
    def test_fit_preprocessors_forwards_arguments(self):
        config = make_config()
        operator = Mock()
        train_times = make_datetime_index(
            2000,
            2001,
        )

        with patch.object(
            TrainDatasetConfig,
            "ds_operator",
            new_callable=PropertyMock,
            return_value=operator,
        ):
            result = config.fit_preprocessors(
                train_times,
                save=True,
                save_path="/tmp/preprocessors",
                save_name="train",
            )

        assert result is None
        operator.fit_preprocessors.assert_called_once_with(
            train_times=train_times,
            save=True,
            save_path="/tmp/preprocessors",
            save_name="train",
        )

    def test_load_fitted_preprocessors_forwards_directory(self):
        config = make_config()
        operator = Mock()

        with patch.object(
            TrainDatasetConfig,
            "ds_operator",
            new_callable=PropertyMock,
            return_value=operator,
        ):
            result = config.load_fitted_preprocessors("/tmp/preprocessors")

        assert result is None
        operator.load_fitted_preprocessors.assert_called_once_with("/tmp/preprocessors")

    def test_add_fitted_preprocessor_forwards_arguments(self):
        config = make_config()
        operator = Mock()
        preprocessor = object()

        with patch.object(
            TrainDatasetConfig,
            "ds_operator",
            new_callable=PropertyMock,
            return_value=operator,
        ):
            result = config.add_fitted_preprocessor(
                preprocessor,
                index=2,
            )

        assert result is None
        operator.add_fitted_preprocessor.assert_called_once_with(
            preprocessor,
            2,
        )

    def test_build_dataset(self):
        config = make_config()
        times = make_datetime_index(
            2000,
            2001,
        )
        time_features = object()
        mask = object()
        dataset = object()

        with patch.object(
            module,
            "TrainDataset",
            return_value=dataset,
        ) as constructor:
            result = config.build_dataset(
                times=times,
                time_features=time_features,
                mask=mask,
                return_metadata=True,
                load=True,
            )

        assert result is dataset
        constructor.assert_called_once_with(
            config=config,
            requested_times=times,
            time_features=time_features,
            mask=mask,
            return_metadata=True,
            load=True,
        )


class TestTrainDatasetBehaviorProperties:
    @pytest.mark.parametrize(
        "observation,expected",
        [
            (
                None,
                True,
            ),
            (
                object(),
                False,
            ),
        ],
    )
    def test_autoencoding_model_data(
        self,
        observation,
        expected,
    ):
        config = make_config(
            observation=observation,
        )
        dataset = make_dataset(config=config)

        assert dataset._autoencoding_model_data is expected

    @pytest.mark.parametrize(
        (
            "observation",
            "using_model_as_condition",
            "expected",
        ),
        [
            (
                None,
                False,
                True,
            ),
            (
                None,
                True,
                True,
            ),
            (
                object(),
                False,
                True,
            ),
            (
                object(),
                True,
                False,
            ),
        ],
    )
    def test_load_model(
        self,
        observation,
        using_model_as_condition,
        expected,
    ):
        config = make_config(
            observation=observation,
            using_model_as_condition=(using_model_as_condition),
        )
        dataset = make_dataset(config=config)

        assert dataset._load_model is expected

    @pytest.mark.parametrize(
        (
            "observation",
            "using_model_as_condition",
            "expected",
        ),
        [
            (
                None,
                False,
                True,
            ),
            (
                None,
                True,
                True,
            ),
            (
                object(),
                False,
                False,
            ),
            (
                object(),
                True,
                True,
            ),
        ],
    )
    def test_write_condition_to_input(
        self,
        observation,
        using_model_as_condition,
        expected,
    ):
        config = make_config(
            observation=observation,
            using_model_as_condition=(using_model_as_condition),
        )
        dataset = make_dataset(config=config)

        assert dataset._write_condition_to_input is expected

    @pytest.mark.parametrize(
        (
            "observation",
            "using_model_as_condition",
            "effective_condition",
            "expected",
        ),
        [
            (
                object(),
                False,
                object(),
                True,
            ),
            (
                object(),
                False,
                None,
                False,
            ),
            (
                object(),
                True,
                object(),
                False,
            ),
            (
                None,
                False,
                object(),
                False,
            ),
        ],
    )
    def test_concat_condition_to_input(
        self,
        observation,
        using_model_as_condition,
        effective_condition,
        expected,
    ):
        config = make_config(
            observation=observation,
            using_model_as_condition=(using_model_as_condition),
            effective_condition=(effective_condition),
        )
        dataset = make_dataset(config=config)

        assert dataset._concat_condition_to_input is expected


class TestTrainDatasetPostInit:
    def test_loads_observation_when_configured(self):
        observation = make_data_config(name="observation")
        config = make_config(
            observation=observation,
        )
        dataset = make_dataset(
            config=config,
            load=True,
        )

        loaded_observation = object()

        with (
            patch.object(
                module.DatasetABC,
                "__init__",
                return_value=None,
                create=True,
            ),
            patch.object(
                dataset,
                "_load_xarray_data",
                return_value=loaded_observation,
            ) as loader,
            patch.object(
                dataset,
                "get_obs_indexes",
                return_value={
                    INIT_TIME_DIM: np.asarray(
                        [
                            0,
                        ]
                    )
                },
            ) as get_indexes,
        ):
            dataset.sample_coords = {
                INIT_TIME_DIM: np.asarray([np.datetime64("2000-01-01")]),
                LEAD_TIME_DIM: np.asarray(
                    [
                        1,
                    ]
                ),
            }
            dataset.__post_init__()

        assert dataset.observation_dataset is loaded_observation
        loader.assert_called_once_with(
            observation,
            load=True,
            add_time_auxiliary_coords=True,
        )
        get_indexes.assert_called_once_with(dataset.sample_coords)

    def test_skips_observation_loading_when_absent(self):
        config = make_config(
            observation=None,
        )
        dataset = make_dataset(config=config)

        with (
            patch.object(
                module.DatasetABC,
                "__init__",
                return_value=None,
                create=True,
            ),
            patch.object(
                dataset,
                "_load_xarray_data",
            ) as loader,
            patch.object(
                dataset,
                "get_obs_indexes",
                return_value=None,
            ),
        ):
            dataset.sample_coords = {}
            dataset.__post_init__()

        loader.assert_not_called()
        assert dataset.obs_indexes is None


class TestObservationIndexes:
    def test_returns_none_without_observation_dataset(self):
        dataset = make_dataset()
        dataset.observation_dataset = None

        result = dataset.get_obs_indexes(
            {
                INIT_TIME_DIM: np.asarray([np.datetime64("2000-01-01")]),
                LEAD_TIME_DIM: np.asarray(
                    [
                        1,
                    ]
                ),
            }
        )

        assert result is None

    def test_computes_observation_indexes(self):
        dataset = make_dataset()

        observation_times = np.asarray(
            [
                "2000-01-01",
                "2000-02-01",
            ],
            dtype="datetime64[ns]",
        )
        dataset.observation_dataset = xr.DataArray(
            np.ones(2),
            dims=(INIT_TIME_DIM,),
            coords={INIT_TIME_DIM: (observation_times)},
        )

        sample_coords = {
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

        result = dataset.get_obs_indexes(sample_coords)

        np.testing.assert_array_equal(
            result[INIT_TIME_DIM],
            np.asarray(
                [
                    0,
                    1,
                ]
            ),
        )

    def test_reports_missing_observation_times(self):
        dataset = make_dataset()

        dataset.observation_dataset = xr.DataArray(
            np.ones(1),
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

        sample_coords = {
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

        with pytest.raises(
            ValueError,
            match=("Some observation coordinates were not found"),
        ):
            dataset.get_obs_indexes(sample_coords)


class TestTargetShape:
    def test_without_observation_uses_input_shape(self):
        dataset = make_dataset(config=make_config(observation=None))

        with patch.object(
            dataset,
            "get_input_shape",
            return_value=(
                2,
                8,
                8,
            ),
        ) as get_input_shape:
            result = dataset.get_target_shape()

        assert result == (
            2,
            8,
            8,
        )
        get_input_shape.assert_called_once_with()

    def test_observation_shape_without_flattener(self):
        observation = make_data_config(
            name="observation",
            names=[
                "tas",
                "pr",
            ],
            spatial_coords={
                "lat": make_coord(
                    [
                        45.0,
                        46.0,
                    ],
                    dim="lat",
                ),
                "lon": make_coord(
                    [
                        -124.0,
                        -123.0,
                        -122.0,
                    ],
                    dim="lon",
                ),
            },
        )
        config = make_config(
            observation=observation,
        )
        dataset = make_dataset(config=config)
        dataset.observation_dataset = object()

        result = dataset.get_target_shape()

        assert result == (
            2,
            2,
            3,
        )

    def test_observation_shape_with_flattener(
        self,
        monkeypatch,
    ):
        class FakeFlattennanremove:
            pass

        flattener = FakeFlattennanremove()
        flattener.final_locations = np.zeros(5)

        pipeline = SimpleNamespace(
            fitted_preprocessors=[
                flattener,
            ],
            get_preprocessors=Mock(return_value=flattener),
        )
        observation = make_data_config(
            name="observation",
            names=[
                "tas",
                "pr",
            ],
            preprocessing_pipeline=pipeline,
        )
        config = make_config(
            observation=observation,
        )
        dataset = make_dataset(config=config)
        dataset.observation_dataset = object()

        monkeypatch.setattr(
            "cccma_ppp.preprocessing.utils_preprocessing.Flattennanremove",
            FakeFlattennanremove,
        )

        result = dataset.get_target_shape()

        assert result == (
            2,
            5,
        )
        pipeline.get_preprocessors.assert_called_once_with("flattener")


class TestIndexObservationDataset:
    def test_returns_none_without_observation_dataset(self):
        dataset = make_dataset()
        dataset.observation_dataset = None

        assert dataset._index_observation_dataset(0) is None

    def test_selects_observation_and_applies_preprocessor(
        self,
    ):
        pipeline = SimpleNamespace(
            fitted_preprocessors=[],
            transform=Mock(side_effect=lambda value: value),
        )
        observation = make_data_config(
            name="observation",
            preprocessing_pipeline=pipeline,
        )
        config = make_config(
            observation=observation,
        )
        dataset = make_dataset(config=config)

        times = np.asarray(
            [
                "2000-01-01",
                "2000-02-01",
            ],
            dtype="datetime64[ns]",
        )
        dataset.observation_dataset = xr.Dataset(
            {
                "tas": (
                    (
                        INIT_TIME_DIM,
                        "channels",
                    ),
                    np.asarray(
                        [
                            [
                                1.0,
                            ],
                            [
                                2.0,
                            ],
                        ]
                    ),
                )
            },
            coords={
                INIT_TIME_DIM: times,
                "channels": [
                    "tas",
                ],
            },
        )
        dataset.obs_indexes = {
            INIT_TIME_DIM: np.asarray(
                [
                    1,
                ]
            )
        }

        with patch.object(
            module,
            "_unwrap_data_variables",
            side_effect=lambda value: value["tas"],
        ) as unwrap:
            result = dataset._index_observation_dataset(0)

        pipeline.transform.assert_called_once()
        unwrap.assert_called_once()
        assert result.item() == pytest.approx(2.0)

    def test_randomly_selects_observation_realization(
        self,
    ):
        pipeline = SimpleNamespace(
            fitted_preprocessors=[],
            transform=Mock(side_effect=lambda value: value),
        )
        observation = make_data_config(
            name="observation",
            preprocessing_pipeline=pipeline,
        )
        config = make_config(
            observation=observation,
        )
        dataset = make_dataset(config=config)

        dataset.observation_dataset = xr.Dataset(
            {
                "tas": (
                    (
                        INIT_TIME_DIM,
                        REALIZATION_DIM,
                    ),
                    np.asarray(
                        [
                            [
                                1.0,
                                2.0,
                                3.0,
                            ]
                        ]
                    ),
                )
            },
            coords={
                INIT_TIME_DIM: np.asarray(
                    [
                        "2000-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                REALIZATION_DIM: [
                    0,
                    1,
                    2,
                ],
            },
        )
        dataset.obs_indexes = {
            INIT_TIME_DIM: np.asarray(
                [
                    0,
                ]
            )
        }

        with (
            patch.object(
                module.np.random,
                "randint",
                return_value=2,
            ) as randint,
            patch.object(
                module,
                "_unwrap_data_variables",
                side_effect=lambda value: value["tas"],
            ),
        ):
            result = dataset._index_observation_dataset(0)

        randint.assert_called_once_with(3)
        assert result.item() == pytest.approx(3.0)


class TestGetItem:
    def make_getitem_dataset(
        self,
        *,
        observation=object(),
        using_model_as_condition=False,
        effective_condition=None,
        return_metadata=False,
        time_features_result=None,
    ):
        config = make_config(
            observation=observation,
            using_model_as_condition=(using_model_as_condition),
            effective_condition=(effective_condition),
        )
        time_features = Mock(return_value=time_features_result)
        dataset = make_dataset(
            config=config,
            time_features=time_features,
            return_metadata=(return_metadata),
        )
        dataset.sample_coords = {
            INIT_TIME_DIM: np.asarray([np.datetime64("2000-01-01")]),
            LEAD_TIME_DIM: np.asarray(
                [
                    1,
                ]
            ),
        }

        dataset._index_condition_dataset = Mock(
            return_value=xr.DataArray(
                np.asarray(
                    [
                        10.0,
                    ]
                ),
                dims=("channels",),
                coords={
                    "channels": [
                        "condition",
                    ]
                },
            )
        )
        dataset._index_observation_dataset = Mock(
            return_value=xr.DataArray(
                np.asarray(
                    [
                        20.0,
                    ]
                ),
                dims=("channels",),
                coords={
                    "channels": [
                        "target",
                    ]
                },
            )
        )
        dataset._index_model_dataset = Mock(
            return_value=xr.DataArray(
                np.asarray(
                    [
                        30.0,
                    ]
                ),
                dims=("channels",),
                coords={
                    "channels": [
                        "model",
                    ]
                },
            )
        )
        dataset._compute = Mock(
            side_effect=lambda *values: tuple(np.asarray(value) for value in values)
        )

        return dataset

    def test_getitem_calls_all_index_helpers(self):
        dataset = self.make_getitem_dataset()

        dataset[0]

        dataset._index_condition_dataset.assert_called_once_with(0)
        dataset._index_observation_dataset.assert_called_once_with(0)
        dataset._index_model_dataset.assert_called_once_with(0)

    def test_autoencoding_uses_model_as_target(self):
        dataset = self.make_getitem_dataset(
            observation=None,
        )

        result = dataset[0]

        torch.testing.assert_close(
            result["input"],
            torch.tensor(
                [
                    10.0,
                ]
            ),
        )
        torch.testing.assert_close(
            result["target"],
            torch.tensor(
                [
                    30.0,
                ]
            ),
        )

    def test_model_condition_replaces_input(self):
        dataset = self.make_getitem_dataset(
            observation=object(),
            using_model_as_condition=True,
            effective_condition=object(),
        )

        result = dataset[0]

        torch.testing.assert_close(
            result["input"],
            torch.tensor(
                [
                    10.0,
                ]
            ),
        )

    def test_condition_is_concatenated_to_input(self):
        dataset = self.make_getitem_dataset(
            observation=object(),
            using_model_as_condition=False,
            effective_condition=object(),
        )

        result = dataset[0]

        torch.testing.assert_close(
            result["input"],
            torch.tensor(
                [
                    30.0,
                    10.0,
                ]
            ),
        )

    def test_returns_float32_tensors(self):
        dataset = self.make_getitem_dataset()

        result = dataset[0]

        assert result["input"].dtype == torch.float32
        assert result["target"].dtype == torch.float32

    def test_time_features_are_called_with_final_input(self):
        features = np.asarray(
            [
                2000.0,
                1.0,
            ]
        )
        dataset = self.make_getitem_dataset(
            using_model_as_condition=True,
            effective_condition=object(),
            time_features_result=features,
        )

        result = dataset[0]

        dataset.time_features.assert_called_once()
        feature_index, feature_input = dataset.time_features.call_args.args

        assert feature_index == 0
        torch.testing.assert_close(
            torch.as_tensor(feature_input.values, dtype=torch.float32),
            torch.tensor(
                [
                    10.0,
                ]
            ),
        )
        torch.testing.assert_close(
            result["added_features"],
            torch.tensor(
                [
                    2000.0,
                    1.0,
                ]
            ),
        )

    def test_none_time_features_are_preserved(self):
        dataset = self.make_getitem_dataset(
            time_features_result=None,
        )

        result = dataset[0]

        assert result["added_features"] is None

    def test_returns_datadict_without_metadata(self):
        dataset = self.make_getitem_dataset(return_metadata=False)

        result = dataset[0]

        assert isinstance(
            result,
            dict,
        )
        assert set(result) == {
            "input",
            "target",
            "added_features",
        }

    def test_returns_metadata_when_requested(self):
        dataset = self.make_getitem_dataset(return_metadata=True)

        data, metadata = dataset[0]

        assert isinstance(
            data,
            dict,
        )
        assert metadata == {
            INIT_TIME_DIM: np.datetime64("2000-01-01"),
            LEAD_TIME_DIM: 1,
        }


class TestCftimeObservationIndexes:
    def test_cftime_coordinates_are_supported(self):
        config = make_config()
        dataset = make_dataset(config=config)

        observation_times = [
            cftime.DatetimeNoLeap(
                2000,
                1,
                1,
            ),
            cftime.DatetimeNoLeap(
                2000,
                2,
                1,
            ),
        ]

        dataset.observation_dataset = xr.DataArray(
            np.ones(2),
            dims=(INIT_TIME_DIM,),
            coords={INIT_TIME_DIM: (observation_times)},
        )

        result = dataset.get_obs_indexes(
            {
                INIT_TIME_DIM: np.asarray(
                    [
                        cftime.DatetimeNoLeap(
                            2000,
                            1,
                            1,
                        ),
                        cftime.DatetimeNoLeap(
                            2000,
                            1,
                            1,
                        ),
                    ],
                    dtype=object,
                ),
                LEAD_TIME_DIM: np.asarray(
                    [
                        1,
                        2,
                    ]
                ),
            }
        )

        np.testing.assert_array_equal(
            result[INIT_TIME_DIM],
            np.asarray(
                [
                    0,
                    1,
                ]
            ),
        )


class warnings_capture:

    def __enter__(self):
        self._context = warnings.catch_warnings(record=True)
        captured = self._context.__enter__()
        warnings.simplefilter("always")
        self.captured = captured
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        return self._context.__exit__(
            exc_type,
            exc_value,
            traceback,
        )

    def __eq__(
        self,
        other,
    ):
        if other == []:
            return len(self.captured) == 0
        return NotImplemented
