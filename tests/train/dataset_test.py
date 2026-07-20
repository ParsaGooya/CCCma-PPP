import numpy as np
import pytest
import torch
import xarray as xr

from cccma_ppp.train.dataset import (
    TrainDataset,
    TrainDatasetConfig,
)


class DummyPipeline:
    def __init__(self):
        self.fitted_preprocessors = []

    def transform(self, value):
        return value

    def get_preprocessors(self, name):
        class Preprocessor:
            final_locations = np.arange(4)

        return Preprocessor()


class DummyInfo:
    def __init__(self):
        self.sizes = {
            "lead_time": 12,
        }
        self.coords = {
            "lat": xr.DataArray(
                [0, 1],
                dims="lat",
            ),
            "lon": xr.DataArray(
                [0, 1],
                dims="lon",
            ),
            "ensembles": xr.DataArray(
                [0, 1],
                dims="ensembles",
            ),
            "year": xr.DataArray(
                [2000, 2001, 2002],
                dims="year",
            ),
            "lead_time": xr.DataArray(
                np.arange(1, 13),
                dims="lead_time",
            ),
            "month": xr.DataArray(
                np.arange(1, 13),
                dims="month",
            ),
        }


class DummyConfig:
    def __init__(self, ensemble_mean=False):
        self.info = DummyInfo()
        self.ensemble_mean = ensemble_mean
        self.preprocessing_pipeline = DummyPipeline()
        self.year_range = np.array([2000, 2001, 2002])
        self.names = ["a"]
        self.list_paths = ["x"]
        self.concat_dim = None
        self.rename_dict = {}
        self.ensemble_list = None
        self.paths = ["x"]


def set_years(config, years):
    years = np.asarray(years)

    config.year_range = years
    config.info.coords["year"] = xr.DataArray(
        years,
        dims="year",
    )


@pytest.fixture
def fake_ds():
    return xr.Dataset(
        {
            "var": (
                (
                    "ensembles",
                    "year",
                    "lead_time",
                    "month",
                ),
                np.zeros((2, 3, 12, 12)),
            )
        },
        coords={
            "ensembles": [0, 1],
            "year": [2000, 2001, 2002],
            "lead_time": np.arange(1, 13),
            "month": np.arange(1, 13),
        },
    )


@pytest.fixture(autouse=True)
def patch_helpers(monkeypatch, fake_ds):
    monkeypatch.setattr(
        TrainDatasetConfig,
        "effective_condition",
        property(
            lambda self: getattr(
                self,
                "_effective_condition",
                None,
            )
        ),
    )

    monkeypatch.setattr(
        TrainDataset,
        "_load_xarray_data",
        lambda *args, **kwargs: fake_ds,
    )

    monkeypatch.setattr(
        "cccma_ppp.train.dataset._get_time_features",
        lambda *args, **kwargs: None,
    )


def make_config(
    model=None,
    observation=None,
    condition=None,
    condition_method="static",
    using_model_condition=False,
    time_features=None,
    lead_months=None,
):
    if model is None:
        model = DummyConfig()

    config = TrainDatasetConfig.__new__(TrainDatasetConfig)

    config.model = model
    config.observation = observation
    config.condition = condition
    config.condition_method = condition_method
    config.time_features = time_features
    config.lead_months = (
        np.arange(1, 13) if lead_months is None else np.asarray(lead_months)
    )
    config._fitted_preprocessors = True
    config._effective_condition = condition

    monkey_value = bool(using_model_condition)

    type(config)._using_model_data_as_condition = property(lambda self: monkey_value)

    return config


def make_dataset(
    config=None,
    years=None,
    mask=None,
    return_metadata=False,
):
    if config is None:
        config = make_config()

    if years is None:
        years = [2000]

    return TrainDataset(
        config=config,
        requested_years=years,
        mask=mask,
        return_metadata=return_metadata,
    )


def test_check_model_same_member_raises():
    config = make_config(
        model=DummyConfig(ensemble_mean=True),
        condition_method="same_member",
    )

    with pytest.raises(ValueError):
        TrainDatasetConfig._check_model(config)


def test_check_model_success():
    config = make_config()

    assert TrainDatasetConfig._check_model(config) is config


def test_check_model_non_same_member_passes():
    config = make_config(
        condition_method="cross_ensemble",
    )

    assert TrainDatasetConfig._check_model(config) is config


def test_check_observation_missing_target_raises():
    config = make_config(
        observation=None,
        condition=None,
        condition_method=None,
    )

    with pytest.raises(ValueError):
        TrainDatasetConfig._check_observation(config)


def test_check_observation_returns_self_without_observation():
    config = make_config(
        observation=None,
        condition_method="static",
    )

    assert TrainDatasetConfig._check_observation(config) is config


def test_check_observation_warns_different_coords():
    model = DummyConfig()
    observation = DummyConfig()

    observation.info.coords["lat"] = xr.DataArray(
        [99],
        dims="lat",
    )

    config = make_config(
        model=model,
        observation=observation,
    )

    with pytest.warns(UserWarning):
        TrainDatasetConfig._check_observation(config)


def test_check_observation_matching_coords():
    config = make_config(
        observation=DummyConfig(),
    )

    assert TrainDatasetConfig._check_observation(config) is config


def test_check_condition_requires_method():
    config = make_config(
        condition=DummyConfig(),
        condition_method=None,
    )

    with pytest.raises(ValueError):
        TrainDatasetConfig._check_condition(config)


def test_check_condition_cross_ensemble_rejects_mean():
    config = make_config(
        condition=DummyConfig(ensemble_mean=True),
        condition_method="cross_ensemble",
    )

    with pytest.raises(ValueError):
        TrainDatasetConfig._check_condition(config)


def test_check_condition_cross_ensemble_requires_ensembles():
    condition = DummyConfig()
    condition.info.coords.pop(
        "ensembles",
        None,
    )

    config = make_config(
        condition=condition,
        condition_method="cross_ensemble",
    )

    with pytest.raises(ValueError):
        TrainDatasetConfig._check_condition(config)


def test_check_condition_ensemble_mean_requires_mean():
    config = make_config(
        condition=DummyConfig(),
        condition_method="ensemble_mean",
    )

    with pytest.raises(ValueError):
        TrainDatasetConfig._check_condition(config)


def test_check_condition_static_rejects_ensemble_list():
    condition = DummyConfig()
    condition.ensemble_list = [1]

    config = make_config(
        condition=condition,
        condition_method="static",
    )

    with pytest.raises(ValueError):
        TrainDatasetConfig._check_condition(config)


def test_check_condition_static_rejects_model_condition():
    config = make_config(
        condition=DummyConfig(),
        condition_method="static",
        using_model_condition=True,
    )

    with pytest.raises(ValueError):
        TrainDatasetConfig._check_condition(config)


def test_check_condition_static_without_condition_raises():
    config = make_config(
        condition=None,
        condition_method="static",
    )

    with pytest.raises(ValueError):
        TrainDatasetConfig._check_condition(config)


@pytest.mark.parametrize(
    "method",
    [
        "cross_ensemble",
        "same_member",
    ],
)
def test_check_condition_valid_ensemble_methods(
    method,
):
    condition = DummyConfig()

    config = make_config(
        condition=condition,
        condition_method=method,
    )

    assert TrainDatasetConfig._check_condition(config) is config


def test_check_condition_valid_ensemble_mean():
    condition = DummyConfig(ensemble_mean=True)

    config = make_config(
        condition=condition,
        condition_method="ensemble_mean",
    )

    assert TrainDatasetConfig._check_condition(config) is config


def test_check_condition_without_condition_nonstatic():
    config = make_config(
        condition=None,
        condition_method="same_member",
    )

    assert TrainDatasetConfig._check_condition(config) is config


def test_num_input_lead_months():
    config = make_config()

    assert config.num_input_lead_months == 12


def test_num_input_lead_months_different_size():
    model = DummyConfig()
    model.info.sizes["lead_time"] = 24

    config = make_config(model=model)

    assert config.num_input_lead_months == 24


def test_common_time_without_observation():
    model = DummyConfig()
    config = make_config(model=model)

    np.testing.assert_array_equal(
        config.get_common_time,
        np.array([2000, 2001, 2002]),
    )


def test_common_time_with_observation():
    model = DummyConfig()
    observation = DummyConfig()

    set_years(model, [2000, 2001])
    set_years(observation, [2001, 2002])

    config = make_config(
        model=model,
        observation=observation,
    )

    np.testing.assert_array_equal(
        config.get_common_time,
        np.array([2001]),
    )


def test_common_time_identical_ranges():
    model = DummyConfig()
    observation = DummyConfig()

    set_years(
        model,
        [2000, 2001, 2002],
    )
    set_years(
        observation,
        [2000, 2001, 2002],
    )

    config = make_config(
        model=model,
        observation=observation,
    )

    np.testing.assert_array_equal(
        config.get_common_time,
        np.array([2000, 2001, 2002]),
    )


def test_common_time_single_value_overlap():
    model = DummyConfig()
    observation = DummyConfig()

    set_years(model, [1999, 2000])
    set_years(observation, [2000, 2001])

    config = make_config(
        model=model,
        observation=observation,
    )

    np.testing.assert_array_equal(
        config.get_common_time,
        np.array([2000]),
    )


def test_common_time_empty_intersection():
    model = DummyConfig()
    observation = DummyConfig()

    set_years(model, [2000])
    set_years(observation, [2001])

    config = make_config(
        model=model,
        observation=observation,
    )

    assert config.get_common_time.size == 0


def test_available_times_without_observation():
    config = make_config()

    np.testing.assert_array_equal(
        config.available_times,
        np.array([2000, 2001, 2002]),
    )


def test_available_times_with_observation():
    model = DummyConfig()
    observation = DummyConfig()

    set_years(model, [2000, 2001, 2002])
    set_years(observation, [2001, 2002])

    config = make_config(
        model=model,
        observation=observation,
    )

    np.testing.assert_array_equal(
        config.available_times,
        np.array([2001, 2002]),
    )


def test_available_times_empty():
    model = DummyConfig()
    observation = DummyConfig()

    set_years(model, [2000])
    set_years(observation, [2001])

    config = make_config(
        model=model,
        observation=observation,
    )

    assert config.available_times.size == 0


def test_dataset_requires_fitted_preprocessors():
    config = make_config()
    config._fitted_preprocessors = False

    with pytest.raises(RuntimeError):
        make_dataset(config)


def test_dataset_rejects_bad_year():
    config = make_config()

    with pytest.raises(ValueError):
        make_dataset(
            config,
            years=[9999],
        )


def test_dataset_basic_construction():
    dataset = make_dataset()

    assert dataset.model_dataset is not None
    assert dataset.mask is not None
    assert dataset.model_indexes is not None


def test_autoencoding_property():
    dataset = make_dataset()

    assert dataset._autoencoding_model_data is True


def test_write_condition_to_input_autoencoding():
    dataset = make_dataset()

    assert dataset._write_condition_to_input is True


def test_load_model_autoencoding():
    dataset = make_dataset()

    assert dataset._load_model is True


def test_concat_condition_to_input():
    config = make_config(
        observation=DummyConfig(),
        condition=DummyConfig(),
    )

    dataset = make_dataset(config)

    assert dataset._concat_condition_to_input is True


def test_concat_condition_false_without_condition():
    config = make_config(
        observation=DummyConfig(),
        condition=None,
    )

    dataset = make_dataset(config)

    assert dataset._concat_condition_to_input is False


def test_get_model_indexes_expected_keys():
    dataset = make_dataset()

    indexes = dataset.get_model_indexes(dataset.sample_coords)

    assert "year" in indexes
    assert "lead_time" in indexes
    assert "ensembles" in indexes


def test_get_model_indexes_without_ensemble():
    config = make_config()
    config.model.info.coords.pop(
        "ensembles",
        None,
    )

    dataset = make_dataset(config)
    indexes = dataset.get_model_indexes(dataset.sample_coords)

    assert "ensembles" not in indexes


def test_get_obs_indexes_none():
    dataset = make_dataset()

    assert dataset.get_obs_indexes(dataset.sample_coords) is None


def test_get_obs_indexes_ensemble_mean():
    observation = DummyConfig(ensemble_mean=True)

    config = make_config(
        observation=observation,
    )

    dataset = make_dataset(config)
    indexes = dataset.get_obs_indexes(dataset.sample_coords)

    assert indexes is not None
    assert "ensembles" not in indexes


def test_get_obs_indexes_without_ensemble_coord():
    observation = DummyConfig()
    observation.info.coords.pop(
        "ensembles",
        None,
    )

    config = make_config(
        observation=observation,
    )

    dataset = make_dataset(config)
    indexes = dataset.get_obs_indexes(dataset.sample_coords)

    assert "ensembles" not in indexes


def test_get_cond_indexes_static():
    config = make_config(
        condition=DummyConfig(),
        condition_method="static",
    )

    dataset = make_dataset(config)

    assert dataset.get_cond_indexes(dataset.sample_coords) is None


def test_get_cond_indexes_without_condition():
    dataset = make_dataset()
    dataset.condition_dataset = None

    assert dataset.get_cond_indexes(dataset.sample_coords) is None


def test_get_cond_indexes_same_member():
    config = make_config(
        condition=DummyConfig(),
        condition_method="same_member",
    )

    dataset = make_dataset(config)
    indexes = dataset.get_cond_indexes(dataset.sample_coords)

    np.testing.assert_array_equal(
        indexes["ensembles"],
        dataset.model_indexes["ensembles"],
    )


def test_index_model_dataset():
    dataset = make_dataset()

    assert dataset._index_model_dataset(0) is not None


def test_index_model_dataset_without_ensemble_key():
    dataset = make_dataset()
    dataset.model_indexes.pop(
        "ensembles",
        None,
    )

    assert dataset._index_model_dataset(0) is not None


def test_index_model_dataset_returns_none_when_not_loaded(
    monkeypatch,
):
    dataset = make_dataset()

    monkeypatch.setattr(
        TrainDataset,
        "_load_model",
        property(lambda self: False),
    )

    assert dataset._index_model_dataset(0) is None


def test_index_condition_dataset_static():
    config = make_config(
        condition=DummyConfig(),
        condition_method="static",
    )

    dataset = make_dataset(config)

    assert dataset._index_condition_dataset(0) is not None


def test_index_condition_dataset_none():
    dataset = make_dataset()
    dataset.condition_dataset = None

    assert dataset._index_condition_dataset(0) is None


@pytest.mark.parametrize(
    "method",
    [
        "cross_ensemble",
        "same_member",
    ],
)
def test_index_condition_dataset_ensemble_methods(
    method,
):
    config = make_config(
        condition=DummyConfig(),
        condition_method=method,
    )

    dataset = make_dataset(config)

    assert dataset._index_condition_dataset(0) is not None


def test_index_condition_without_ensemble_indexes():
    config = make_config(
        condition=DummyConfig(),
        condition_method="same_member",
    )

    dataset = make_dataset(config)
    dataset.cond_indexes.pop(
        "ensembles",
        None,
    )

    assert dataset._index_condition_dataset(0) is not None


def test_index_observation_dataset_none():
    dataset = make_dataset()
    dataset.observation_dataset = None

    assert dataset._index_observation_dataset(0) is None


def test_index_observation_dataset():
    config = make_config(
        observation=DummyConfig(),
    )

    dataset = make_dataset(config)

    assert dataset._index_observation_dataset(0) is not None


def test_index_observation_without_ensemble_indexes():
    config = make_config(
        observation=DummyConfig(),
    )

    dataset = make_dataset(config)
    dataset.obs_indexes.pop(
        "ensembles",
        None,
    )

    assert dataset._index_observation_dataset(0) is not None


def test_prepare_mask_requested_years():
    dataset = make_dataset(
        years=[2000],
    )

    np.testing.assert_array_equal(
        dataset.mask.coords["year"].values,
        np.array([2000]),
    )


def test_prepare_mask_with_ensemble_dimension():
    dataset = make_dataset()

    assert "ensembles" in dataset.mask.dims


def test_prepare_mask_without_ensemble_dimension():
    config = make_config()
    config.model.info.coords.pop(
        "ensembles",
        None,
    )

    dataset = make_dataset(config)

    assert "ensembles" not in dataset.mask.dims


def test_prepare_mask_ensemble_mean():
    config = make_config()
    config.model.ensemble_mean = True

    dataset = make_dataset(config)

    assert "ensembles" not in dataset.mask.dims


def test_get_added_features_dim_none():
    dataset = make_dataset()

    assert dataset.get_added_features_dim() == 0


def test_get_added_features_dim_empty():
    config = make_config(
        time_features=[],
    )

    dataset = make_dataset(config)

    assert dataset.get_added_features_dim() == 0


@pytest.mark.parametrize(
    ("features", "expected"),
    [
        (["year"], 1),
        (["a", "b"], 2),
        (["a", "b", "c"], 3),
    ],
)
def test_get_added_features_dim(
    features,
    expected,
):
    config = make_config(
        time_features=features,
    )

    dataset = make_dataset(config)

    assert dataset.get_added_features_dim() == expected


def test_get_input_shape():
    dataset = make_dataset()

    assert isinstance(
        dataset.get_input_shape(),
        tuple,
    )


def test_get_target_shape_without_observation():
    dataset = make_dataset()

    dataset.input_shape = dataset.get_input_shape()

    assert dataset.get_target_shape() == dataset.input_shape


def test_get_target_shape_with_observation():
    config = make_config(
        observation=DummyConfig(),
    )

    dataset = make_dataset(config)

    assert isinstance(
        dataset.get_target_shape(),
        tuple,
    )


def test_len():
    dataset = make_dataset()

    assert len(dataset) > 0


def test_len_matches_sample_coordinates():
    dataset = make_dataset()

    expected = len(next(iter(dataset.sample_coords.values())))

    assert len(dataset) == expected


def test_getitem_returns_dict(monkeypatch):
    dataset = make_dataset()

    value = xr.DataArray(np.array([1.0]))

    monkeypatch.setattr(
        dataset,
        "_index_model_dataset",
        lambda index: value,
    )
    monkeypatch.setattr(
        dataset,
        "_index_condition_dataset",
        lambda index: value,
    )

    result = dataset[0]

    assert isinstance(result, dict)
    assert "input" in result
    assert "target" in result
    assert "added_features" in result


def test_getitem_returns_metadata(monkeypatch):
    dataset = make_dataset(
        return_metadata=True,
    )

    value = xr.DataArray(np.array([1.0]))

    monkeypatch.setattr(
        dataset,
        "_index_model_dataset",
        lambda index: value,
    )
    monkeypatch.setattr(
        dataset,
        "_index_condition_dataset",
        lambda index: value,
    )

    result, metadata = dataset[0]

    assert isinstance(result, dict)
    assert metadata["year"] == 2000
    assert "lead_time" in metadata


def test_getitem_added_features(monkeypatch):
    dataset = make_dataset()

    value = xr.DataArray(np.array([1.0]))

    monkeypatch.setattr(
        dataset,
        "_index_model_dataset",
        lambda index: value,
    )
    monkeypatch.setattr(
        dataset,
        "_index_condition_dataset",
        lambda index: value,
    )
    monkeypatch.setattr(
        "cccma_ppp.train.dataset._get_time_features",
        lambda *args, **kwargs: np.array([1.0, 2.0]),
    )

    result = dataset[0]

    assert result["added_features"] is not None


def test_getitem_autoencoding(monkeypatch):
    dataset = make_dataset()

    value = xr.DataArray(np.array([1.0]))

    monkeypatch.setattr(
        dataset,
        "_index_model_dataset",
        lambda index: value,
    )
    monkeypatch.setattr(
        dataset,
        "_index_condition_dataset",
        lambda index: value,
    )
    monkeypatch.setattr(
        TrainDataset,
        "_write_condition_to_input",
        property(lambda self: False),
    )

    result = dataset[0]

    torch.testing.assert_close(
        result["target"],
        result["input"],
    )


def test_getitem_write_condition(monkeypatch):
    config = make_config(
        condition=DummyConfig(),
    )

    dataset = make_dataset(config)

    model_value = xr.DataArray(np.array([1.0]))
    condition_value = xr.DataArray(np.array([2.0]))

    monkeypatch.setattr(
        dataset,
        "_index_model_dataset",
        lambda index: model_value,
    )
    monkeypatch.setattr(
        dataset,
        "_index_condition_dataset",
        lambda index: condition_value,
    )

    result = dataset[0]

    assert result["input"].item() == 2.0


def test_getitem_concat_condition(monkeypatch):
    config = make_config(
        observation=DummyConfig(),
        condition=DummyConfig(),
    )

    dataset = make_dataset(config)

    model_value = xr.DataArray(
        np.array([1.0]),
        dims="channels",
    )
    condition_value = xr.DataArray(
        np.array([2.0]),
        dims="channels",
    )
    target_value = xr.DataArray(
        np.array([3.0]),
        dims="channels",
    )

    monkeypatch.setattr(
        dataset,
        "_index_model_dataset",
        lambda index: model_value,
    )
    monkeypatch.setattr(
        dataset,
        "_index_condition_dataset",
        lambda index: condition_value,
    )
    monkeypatch.setattr(
        dataset,
        "_index_observation_dataset",
        lambda index: target_value,
    )

    result = dataset[0]

    assert isinstance(
        result["input"],
        torch.Tensor,
    )

    assert result["input"].shape[0] == 2


def test_build_dataset(monkeypatch):
    config = make_config()

    monkeypatch.setattr(
        TrainDataset,
        "__post_init__",
        lambda self: None,
    )

    dataset = config.build_dataset(
        years=[2000],
        return_metadata=True,
    )

    assert isinstance(
        dataset,
        TrainDataset,
    )
    assert dataset.return_metadata is True


def test_build_dataset_mask(monkeypatch):
    config = make_config()

    monkeypatch.setattr(
        TrainDataset,
        "__post_init__",
        lambda self: None,
    )

    mask = object()

    dataset = config.build_dataset(
        years=[2000],
        mask=mask,
    )

    assert dataset.mask is mask


def test_fit_preprocessors_delegates(monkeypatch):
    config = make_config()
    called = {}

    class Operator:
        def _fit_preprocessors(
            self,
            **kwargs,
        ):
            called.update(kwargs)

    monkeypatch.setattr(
        TrainDatasetConfig,
        "ds_operator",
        property(lambda self: Operator()),
    )

    config._fit_preprocessors(
        train_years=[2000],
        save=True,
        save_path="a",
        save_name="b",
    )

    assert called == {
        "train_years": [2000],
        "save": True,
        "save_path": "a",
        "save_name": "b",
    }


def test_load_fitted_preprocessors_delegates(
    monkeypatch,
):
    config = make_config()
    called = {}

    class Operator:
        def _load_fitted_preprocessors(
            self,
            load_dir,
        ):
            called["load_dir"] = load_dir

    monkeypatch.setattr(
        TrainDatasetConfig,
        "ds_operator",
        property(lambda self: Operator()),
    )

    config._load_fitted_preprocessors("abc")

    assert called["load_dir"] == "abc"


def test_add_fitted_preprocessor_delegates(
    monkeypatch,
):
    config = make_config()
    called = {}

    class Operator:
        def _add_fitted_preprocessor(
            self,
            preprocessor,
            index,
        ):
            called["preprocessor"] = preprocessor
            called["index"] = index

    monkeypatch.setattr(
        TrainDatasetConfig,
        "ds_operator",
        property(lambda self: Operator()),
    )

    value = object()

    config._add_fitted_preprocessor(
        value,
        index=4,
    )

    assert called["preprocessor"] is value
    assert called["index"] == 4


def test_get_obs_indexes():
    config = make_config(
        observation=DummyConfig(),
    )

    dataset = make_dataset(config)

    indexes = dataset.get_obs_indexes(dataset.sample_coords)

    assert set(indexes) == {
        "year",
        "month",
    }
    assert len(indexes["year"]) == len(dataset)
    assert len(indexes["month"]) == len(dataset)


def test_get_cond_indexes_cross_ensemble():
    config = make_config(
        condition=DummyConfig(),
        condition_method="cross_ensemble",
    )

    dataset = make_dataset(config)

    indexes = dataset.get_cond_indexes(dataset.sample_coords)

    assert set(indexes) == {
        "year",
        "lead_time",
    }
    assert len(indexes["year"]) == len(dataset)
    assert len(indexes["lead_time"]) == len(dataset)


def test_prepare_existing_mask():
    config = make_config()
    config.model.ensemble_mean = True

    mask = xr.DataArray(
        np.zeros(
            (1, 12),
            dtype=bool,
        ),
        dims=(
            "year",
            "lead_time",
        ),
        coords={
            "year": [2000],
            "lead_time": np.arange(1, 13),
        },
    )

    dataset = make_dataset(
        config,
        mask=mask,
    )

    assert dataset.mask.dims == (
        "year",
        "lead_time",
    )
    assert not dataset.mask.any().item()
    assert len(dataset) == 12


def test_prepare_existing_mask_with_ensemble():
    config = make_config()

    mask = xr.DataArray(
        np.zeros(
            (1, 12),
            dtype=bool,
        ),
        dims=(
            "year",
            "lead_time",
        ),
        coords={
            "year": [2000],
            "lead_time": np.arange(1, 13),
        },
    )

    dataset = make_dataset(
        config,
        mask=mask,
    )

    assert "ensembles" in dataset.mask.dims
    assert dataset.mask.sizes["ensembles"] == 2
    assert dataset.mask.sizes["year"] == 1
    assert dataset.mask.sizes["lead_time"] == 12
    assert not dataset.mask.any().item()
    assert len(dataset) == 24


def test_get_obs_indexes_raises_for_missing_coordinates():
    observation = DummyConfig()

    config = make_config(
        observation=observation,
    )
    dataset = make_dataset(config)

    bad_coords = {
        "year": np.array([9999]),
        "lead_time": np.array([1]),
    }

    with pytest.raises(
        ValueError,
        match="observation coordinates were not found",
    ):
        dataset.get_obs_indexes(bad_coords)


def test_get_cond_indexes_raises_for_missing_coordinates():
    condition = DummyConfig()

    config = make_config(
        condition=condition,
        condition_method="cross_ensemble",
    )
    dataset = make_dataset(config)

    bad_coords = {
        "year": np.array([9999]),
        "lead_time": np.array([999]),
    }

    with pytest.raises(
        ValueError,
        match="conditioning coordinates were not found",
    ):
        dataset.get_cond_indexes(bad_coords)


def test_get_cond_indexes_same_member_requires_ensembles():
    condition = DummyConfig()

    config = make_config(
        condition=condition,
        condition_method="same_member",
    )
    dataset = make_dataset(config)

    sample_coords = {
        "year": np.array([2000]),
        "lead_time": np.array([1]),
    }

    with pytest.raises(
        ValueError,
        match="requires ensemble coordinates",
    ):
        dataset.get_cond_indexes(sample_coords)


def test_get_cond_indexes_ignores_unavailable_dimensions():
    condition = DummyConfig()

    config = make_config(
        condition=condition,
        condition_method="cross_ensemble",
    )
    dataset = make_dataset(config)

    sample_coords = {
        "year": np.array([2000]),
        "lead_time": np.array([1]),
        "ensembles": np.array([0]),
        "not_a_condition_dimension": np.array([7]),
    }

    indexes = dataset.get_cond_indexes(sample_coords)

    assert set(indexes) == {
        "year",
        "lead_time",
    }


def test_get_obs_indexes_zero_based_positions():
    observation = DummyConfig()

    config = make_config(
        observation=observation,
    )
    dataset = make_dataset(config)

    sample_coords = {
        "year": np.array([2000]),
        "lead_time": np.array([1]),
    }

    indexes = dataset.get_obs_indexes(sample_coords)

    np.testing.assert_array_equal(
        indexes["year"],
        np.array([0]),
    )
    np.testing.assert_array_equal(
        indexes["month"],
        np.array([0]),
    )


def test_get_obs_indexes_lead_time_crosses_year():
    observation = DummyConfig()

    config = make_config(
        observation=observation,
    )
    dataset = make_dataset(config)

    sample_coords = {
        "year": np.array([2000]),
        "lead_time": np.array([13]),
    }

    indexes = dataset.get_obs_indexes(sample_coords)

    np.testing.assert_array_equal(
        indexes["year"],
        np.array([1]),
    )
    np.testing.assert_array_equal(
        indexes["month"],
        np.array([0]),
    )


def test_get_cond_indexes_same_member_maps_ensemble_positions():
    condition = DummyConfig()

    config = make_config(
        condition=condition,
        condition_method="same_member",
    )
    dataset = make_dataset(config)

    sample_coords = {
        "year": np.array([2000, 2000]),
        "lead_time": np.array([1, 2]),
        "ensembles": np.array([0, 1]),
    }

    indexes = dataset.get_cond_indexes(sample_coords)

    np.testing.assert_array_equal(
        indexes["ensembles"],
        np.array([0, 1]),
    )


def test_get_target_shape_observation_without_flattener():
    observation = DummyConfig()
    observation.preprocessing_pipeline.fitted_preprocessors = []

    config = make_config(
        observation=observation,
    )
    dataset = make_dataset(config)

    shape = dataset.get_target_shape()

    assert isinstance(shape, tuple)
    assert shape == (2, 2)


def test_get_input_shape_without_flattener():
    config = make_config()
    config.model.preprocessing_pipeline.fitted_preprocessors = []

    dataset = make_dataset(config)

    shape = dataset.get_input_shape()

    assert isinstance(shape, tuple)
    assert shape == (2, 2)


def test_get_input_shape_with_flattener(
    monkeypatch,
):
    class FakeFlattennanremove:
        pass

    monkeypatch.setattr(
        "cccma_ppp.preprocessing.utils_preprocessing.Flattennanremove",
        FakeFlattennanremove,
    )

    config = make_config()
    config.model.preprocessing_pipeline.fitted_preprocessors = [FakeFlattennanremove()]

    dataset = make_dataset(config)

    assert dataset.get_input_shape() == (4,)


def test_get_target_shape_with_flattener(
    monkeypatch,
):
    class FakeFlattennanremove:
        pass

    monkeypatch.setattr(
        "cccma_ppp.preprocessing.utils_preprocessing.Flattennanremove",
        FakeFlattennanremove,
    )

    observation = DummyConfig()
    observation.preprocessing_pipeline.fitted_preprocessors = [FakeFlattennanremove()]

    config = make_config(
        observation=observation,
    )
    dataset = make_dataset(config)

    assert dataset.get_target_shape() == (4,)


def test_build_dataset_load_passthrough(
    monkeypatch,
):
    config = make_config()

    monkeypatch.setattr(
        TrainDataset,
        "__post_init__",
        lambda self: None,
    )

    dataset = config.build_dataset(
        years=[2000],
        load=True,
    )

    assert dataset.load is True


def test_build_dataset_load_default_false(
    monkeypatch,
):
    config = make_config()

    monkeypatch.setattr(
        TrainDataset,
        "__post_init__",
        lambda self: None,
    )

    dataset = config.build_dataset(
        years=[2000],
    )

    assert dataset.load is False
