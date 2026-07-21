from types import SimpleNamespace

import numpy as np
import pytest
import torch
import xarray as xr

from cccma_ppp.data_modules.dataset.dataset_abc import AddedTimeFeatures
from cccma_ppp.inference.dataset import (
    InferenceDataset,
    InferenceDatasetConfig,
    _from_train,
)


                                                                          
InferenceDatasetConfig.__abstractmethods__ = frozenset()


class DummyInfo:
    def __init__(
        self,
        lead_time=12,
        ensembles=True,
        years=None,
        lat_size=2,
        lon_size=3,
    ):
        years = [2000, 2001, 2002] if years is None else years

        self.sizes = {"lead_time": lead_time}
        self.coords = {
            "lat": xr.DataArray(np.arange(lat_size), dims="lat"),
            "lon": xr.DataArray(np.arange(lon_size), dims="lon"),
            "year": xr.DataArray(years, dims="year"),
            "lead_time": xr.DataArray(
                np.arange(1, lead_time + 1),
                dims="lead_time",
            ),
            "month": xr.DataArray(
                np.arange(1, 13),
                dims="month",
            ),
        }

        if ensembles:
            self.coords["ensembles"] = xr.DataArray(
                [0, 1],
                dims="ensembles",
            )


class DummyPipeline:
    def __init__(self, name="pipeline"):
        self.name = name
        self.fitted_preprocessors = []
        self.transform_calls = []

    def transform(self, value):
        self.transform_calls.append(value)
        return value

    def get_preprocessors(self, name):
        return SimpleNamespace(
            final_locations=np.arange(4),
        )


class DummyDataConfig:
    def __init__(
        self,
        name="data",
        ensemble_mean=False,
        ensembles=True,
        years=None,
        lead_time=12,
        names=None,
    ):
        years = [2000, 2001, 2002] if years is None else years

        self.names = names or [name]
        self.info = DummyInfo(
            lead_time=lead_time,
            ensembles=ensembles,
            years=years,
        )
        self.ensemble_mean = ensemble_mean
        self.ensemble_list = None
        self.year_range = np.asarray(years)
        self.preprocessing_pipeline = DummyPipeline(name)
        self.list_paths = [f"{name}.nc"]
        self.paths = [f"{name}.nc"]
        self.concat_dim = None
        self.rename_dict = {}


class DummyOperator:
    def __init__(self):
        self.load_calls = []
        self.add_calls = []

    def load_fitted_preprocessors(self, load_dir=None):
        self.load_calls.append(load_dir)

    def add_fitted_preprocessor(self, preprocessor, index=0):
        self.add_calls.append((preprocessor, index))


def make_config(
    model=None,
    condition=None,
    method=None,
    using_model_condition=False,
    effective_condition=None,
    lead_months=None,
    fitted=True,
):
    if effective_condition is None:
        if condition is not None:
            effective_condition = condition
        elif using_model_condition:
            effective_condition = model

    return SimpleNamespace(
        model=model,
        condition=condition,
        condition_method=method,
        lead_months=(
            np.arange(1, 13) if lead_months is None else np.asarray(lead_months)
        ),
        get_common_time=np.asarray([2000, 2001, 2002]),
        _fitted_preprocessors=fitted,
        _using_model_data_as_condition=using_model_condition,
        effective_condition=effective_condition,
        effective_input=model if model is not None else condition,
    )


def make_config_object(
    model=None,
    condition=None,
    method=None,
    effective_condition=None,
    using_model_condition=False,
):
    config = object.__new__(InferenceDatasetConfig)

    if using_model_condition and condition is None:
        condition = model

    config.model = model
    config.condition = condition
    config.condition_method = method
    config.lead_months = None
    config._effective_condition = effective_condition
    config._fitted_preprocessors = True

    return config


def make_dataset_object(
    config=None,
    return_metadata=False,
    include_ensembles=False,
    features=None,
):
    if config is None:
        config = make_config(model=DummyDataConfig())

    dataset = object.__new__(InferenceDataset)
    dataset.config = config
    dataset.return_metadata = return_metadata
    dataset.requested_years = np.asarray([2000])
    dataset.sample_coords = {
        "year": np.asarray([2000.0]),
        "lead_time": np.asarray([1.0]),
    }

    if include_ensembles:
        dataset.sample_coords["ensembles"] = np.asarray([0])

    dataset.time_features = AddedTimeFeatures(
        config,
        features,
    )

    return dataset


def call_check_model(config):
    return InferenceDatasetConfig._check_model(config)


def call_check_condition(config):
    return InferenceDatasetConfig._check_condition(config)


def get_num_input_lead_months(config):
    return InferenceDatasetConfig.num_input_lead_months.fget(config)


def test_effective_input_model():
    model = DummyDataConfig()
    condition = DummyDataConfig()

    config = make_config_object(
        model=model,
        condition=condition,
    )

    assert config.effective_input is model


def test_effective_input_condition():
    condition = DummyDataConfig()
    config = make_config_object(condition=condition)

    assert config.effective_input is condition


def test_available_times():
    model = DummyDataConfig(years=[2000, 2001, 2002])
    config = make_config_object(model=model)

    np.testing.assert_array_equal(
        config.available_times,
        [2000, 2001, 2002],
    )


@pytest.mark.pruned
                                
def test_load_fitted_preprocessors(monkeypatch):
    config = make_config_object()
    operator = DummyOperator()

    monkeypatch.setattr(
        InferenceDatasetConfig,
        "ds_operator",
        property(lambda self: operator),
    )

    config.load_fitted_preprocessors("load-dir")

    assert operator.load_calls == ["load-dir"]


@pytest.mark.pruned
                                
def test_add_fitted_preprocessor(monkeypatch):
    config = make_config_object()
    operator = DummyOperator()
    preprocessor = object()

    monkeypatch.setattr(
        InferenceDatasetConfig,
        "ds_operator",
        property(lambda self: operator),
    )

    config.add_fitted_preprocessor(
        preprocessor,
        index=4,
    )

    assert operator.add_calls == [(preprocessor, 4)]


@pytest.mark.pruned
def test_build_dataset(monkeypatch):
    config = make_config_object(model=DummyDataConfig())
    features = AddedTimeFeatures(
        make_config(model=DummyDataConfig()),
        None,
    )

    monkeypatch.setattr(
        InferenceDataset,
        "__post_init__",
        lambda self: None,
    )

    result = config.build_dataset(
        years=np.asarray([2000]),
        time_features=features,
        return_metadata=True,
        load=True,
    )

    assert isinstance(result, InferenceDataset)
    assert result.time_features is features
    np.testing.assert_array_equal(
        result.requested_years,
        [2000],
    )
    assert result.return_metadata is True
    assert result.load is True


@pytest.mark.parametrize(
    ("using_model", "model_present", "expected"),
    [
        (False, True, True),
        (True, True, False),
        (False, False, False),
        (True, False, False),
    ],
)
def test_load_model(
    using_model,
    model_present,
    expected,
):
    dataset = object.__new__(InferenceDataset)
    dataset.config = SimpleNamespace(
        _using_model_data_as_condition=using_model,
        model=DummyDataConfig() if model_present else None,
    )

    assert dataset._load_model is expected


@pytest.mark.parametrize(
    ("using_model", "model_present", "expected"),
    [
        (True, True, True),
        (False, False, True),
        (True, False, True),
        (False, True, False),
    ],
)
def test_write_condition_to_input(
    using_model,
    model_present,
    expected,
):
    dataset = object.__new__(InferenceDataset)
    dataset.config = SimpleNamespace(
        _using_model_data_as_condition=using_model,
        model=DummyDataConfig() if model_present else None,
    )

    assert dataset._write_condition_to_input is expected


@pytest.mark.parametrize(
    ("write_condition", "condition_present", "expected"),
    [
        (False, True, True),
        (False, False, False),
        (True, True, False),
        (True, False, False),
    ],
)
def test_concat_condition_to_input(
    monkeypatch,
    write_condition,
    condition_present,
    expected,
):
    dataset = object.__new__(InferenceDataset)
    dataset.config = SimpleNamespace(
        condition=DummyDataConfig() if condition_present else None
    )

    monkeypatch.setattr(
        InferenceDataset,
        "_write_condition_to_input",
        property(lambda self: write_condition),
    )

    assert dataset._concat_condition_to_input is expected


def patch_item_sources(
    monkeypatch,
    model=None,
    condition=None,
    write_condition=False,
    concat_condition=False,
):
    monkeypatch.setattr(
        InferenceDataset,
        "_index_model_dataset",
        lambda self, index: model,
    )
    monkeypatch.setattr(
        InferenceDataset,
        "_index_condition_dataset",
        lambda self, index: condition,
    )
    monkeypatch.setattr(
        InferenceDataset,
        "_write_condition_to_input",
        property(lambda self: write_condition),
    )
    monkeypatch.setattr(
        InferenceDataset,
        "_concat_condition_to_input",
        property(lambda self: concat_condition),
    )


def test_getitem_time_features(monkeypatch):
    dataset = make_dataset_object(
        features=["year", "lead_time"],
    )

    patch_item_sources(
        monkeypatch,
        model=xr.DataArray(np.asarray([1.0])),
    )

    result = dataset[0]

    assert result["added_features"] is not None
    assert result["added_features"].shape[0] == 2


def test_getitem_metadata(monkeypatch):
    dataset = make_dataset_object(
        return_metadata=True,
        include_ensembles=True,
        features=None,
    )

    patch_item_sources(
        monkeypatch,
        model=xr.DataArray(np.asarray([1.0])),
    )

    result, metadata = dataset[0]

    assert isinstance(result, dict)
    assert metadata == {
        "year": 2000.0,
        "lead_time": 1.0,
        "ensembles": 0,
    }


def make_train_config(
    observation=None,
    effective_condition=None,
    using_model_condition=False,
    model=None,
    condition=None,
):
    return SimpleNamespace(
        condition_method="static",
        lead_months=np.asarray([1, 2]),
        observation=observation,
        effective_condition=effective_condition,
        _using_model_data_as_condition=using_model_condition,
        model=model,
        condition=condition,
    )


def test_from_train_observation_only(monkeypatch):
    model = DummyDataConfig("model")
    train = make_train_config(
        observation=object(),
        model=model,
    )

    monkeypatch.setattr(
        "cccma_ppp.inference.dataset.InferenceDatasetConfig",
        lambda **kwargs: kwargs,
    )

    result = _from_train(train)

    assert result["model"] is not model
    assert result["model"].names == model.names
    assert "condition" not in result


def test_from_train_model_condition(monkeypatch):
    model = DummyDataConfig("model")
    train = make_train_config(
        effective_condition=model,
        using_model_condition=True,
        model=model,
    )

    monkeypatch.setattr(
        "cccma_ppp.inference.dataset.InferenceDatasetConfig",
        lambda **kwargs: kwargs,
    )

    result = _from_train(train)

    assert result["model"].names == model.names
    assert "condition" not in result


def test_from_train_condition_with_observation(monkeypatch):
    model = DummyDataConfig("model")
    condition = DummyDataConfig("condition")
    train = make_train_config(
        observation=object(),
        effective_condition=condition,
        model=model,
        condition=condition,
    )

    monkeypatch.setattr(
        "cccma_ppp.inference.dataset.InferenceDatasetConfig",
        lambda **kwargs: kwargs,
    )

    result = _from_train(train)

    assert result["model"].names == model.names
    assert result["condition"].names == condition.names


def test_from_train_condition_without_observation(monkeypatch):
    condition = DummyDataConfig("condition")
    train = make_train_config(
        effective_condition=condition,
        condition=condition,
    )

    monkeypatch.setattr(
        "cccma_ppp.inference.dataset.InferenceDatasetConfig",
        lambda **kwargs: kwargs,
    )

    result = _from_train(train)

    assert "model" not in result
    assert result["condition"].names == condition.names


def test_from_train_unresolvable():
    with pytest.raises(ValueError):
        _from_train(make_train_config())


@pytest.mark.pruned
def test_from_train_deepcopies_fields(monkeypatch):
    model = DummyDataConfig("model")
    lead_months = np.asarray([1, 2])

    train = SimpleNamespace(
        condition_method="static",
        lead_months=lead_months,
        observation=object(),
        effective_condition=None,
        _using_model_data_as_condition=False,
        model=model,
        condition=None,
    )

    monkeypatch.setattr(
        "cccma_ppp.inference.dataset.InferenceDatasetConfig",
        lambda **kwargs: kwargs,
    )

    result = _from_train(train)

    assert "time_features" not in result
    np.testing.assert_array_equal(
        result["lead_months"],
        lead_months,
    )
    assert result["lead_months"] is not lead_months


@pytest.mark.pruned
def test_effective_input_prefers_model():
    model = DummyDataConfig("model")
    condition = DummyDataConfig("condition")

    config = make_config_object(
        model=model,
        condition=condition,
    )

    assert config.effective_input is model


@pytest.mark.pruned
def test_effective_input_falls_back_to_condition():
    condition = DummyDataConfig("condition")

    config = make_config_object(
        model=None,
        condition=condition,
    )

    assert config.effective_input is condition


def test_available_times_filters_effective_input_coordinates():
    model = DummyDataConfig(
        "model",
        years=[2000, 2001, 2002],
    )
    condition = DummyDataConfig(
        "condition",
        years=[2001, 2002, 2003],
    )

    model.info.coords["year"] = xr.DataArray(
        [2000, 2002],
        dims="year",
    )

    config = make_config_object(
        model=model,
        condition=condition,
    )

    np.testing.assert_array_equal(
        config.available_times,
        [2002],
    )


def test_available_times_condition_only_filters_coordinates():
    condition = DummyDataConfig(
        "condition",
        years=[2000, 2001, 2002],
    )
    condition.info.coords["year"] = xr.DataArray(
        [2000, 2002],
        dims="year",
    )

    config = make_config_object(
        model=None,
        condition=condition,
    )

    np.testing.assert_array_equal(
        config.available_times,
        [2000, 2002],
    )


@pytest.mark.pruned
                                
def test_load_fitted_preprocessors_default_argument(
    monkeypatch,
):
    config = make_config_object()
    operator = DummyOperator()

    monkeypatch.setattr(
        InferenceDatasetConfig,
        "ds_operator",
        property(lambda self: operator),
    )

    config.load_fitted_preprocessors()

    assert operator.load_calls == [None]


@pytest.mark.pruned
                                
def test_add_fitted_preprocessor_default_index(
    monkeypatch,
):
    config = make_config_object()
    operator = DummyOperator()
    preprocessor = object()

    monkeypatch.setattr(
        InferenceDatasetConfig,
        "ds_operator",
        property(lambda self: operator),
    )

    config.add_fitted_preprocessor(preprocessor)

    assert operator.add_calls == [(preprocessor, 0)]


@pytest.mark.pruned
def test_build_dataset_default_arguments(
    monkeypatch,
):
    config = make_config_object(model=DummyDataConfig())
    features = AddedTimeFeatures(
        make_config(model=DummyDataConfig()),
        None,
    )

    monkeypatch.setattr(
        InferenceDataset,
        "__post_init__",
        lambda self: None,
    )

    dataset = config.build_dataset(
        years=np.asarray([2001]),
        time_features=features,
    )

    assert isinstance(
        dataset,
        InferenceDataset,
    )
    assert dataset.return_metadata is False
    assert dataset.load is False
    assert dataset.mask is None
    assert dataset.time_features is features

    np.testing.assert_array_equal(
        dataset.requested_years,
        [2001],
    )


@pytest.mark.pruned
                                
def test_load_model_false_when_model_is_reused_as_condition():
    dataset = object.__new__(InferenceDataset)
    dataset.config = SimpleNamespace(
        model=DummyDataConfig(),
        _using_model_data_as_condition=True,
    )

    assert dataset._load_model is False


@pytest.mark.pruned
                                
def test_load_model_true_for_independent_model():
    dataset = object.__new__(InferenceDataset)
    dataset.config = SimpleNamespace(
        model=DummyDataConfig(),
        _using_model_data_as_condition=False,
    )

    assert dataset._load_model is True


@pytest.mark.pruned
                                
def test_load_model_false_without_model():
    dataset = object.__new__(InferenceDataset)
    dataset.config = SimpleNamespace(
        model=None,
        _using_model_data_as_condition=False,
    )

    assert dataset._load_model is False


@pytest.mark.pruned
                                
def test_write_condition_true_when_model_reused():
    dataset = object.__new__(InferenceDataset)
    dataset.config = SimpleNamespace(
        model=DummyDataConfig(),
        _using_model_data_as_condition=True,
    )

    assert dataset._write_condition_to_input is True


@pytest.mark.pruned
                                
def test_write_condition_true_without_model():
    dataset = object.__new__(InferenceDataset)
    dataset.config = SimpleNamespace(
        model=None,
        _using_model_data_as_condition=False,
    )

    assert dataset._write_condition_to_input is True


@pytest.mark.pruned
                                
def test_write_condition_false_for_independent_model():
    dataset = object.__new__(InferenceDataset)
    dataset.config = SimpleNamespace(
        model=DummyDataConfig(),
        _using_model_data_as_condition=False,
    )

    assert dataset._write_condition_to_input is False


@pytest.mark.pruned
                                
def test_concat_condition_true_for_independent_condition():
    dataset = object.__new__(InferenceDataset)
    dataset.config = SimpleNamespace(
        model=DummyDataConfig(),
        condition=DummyDataConfig("condition"),
        _using_model_data_as_condition=False,
    )

    assert dataset._concat_condition_to_input is True


@pytest.mark.pruned
                                
def test_concat_condition_false_without_condition():
    dataset = object.__new__(InferenceDataset)
    dataset.config = SimpleNamespace(
        model=DummyDataConfig(),
        condition=None,
        _using_model_data_as_condition=False,
    )

    assert dataset._concat_condition_to_input is False


@pytest.mark.pruned
                                
def test_concat_condition_false_when_condition_overwrites_input():
    dataset = object.__new__(InferenceDataset)
    dataset.config = SimpleNamespace(
        model=None,
        condition=DummyDataConfig("condition"),
        _using_model_data_as_condition=False,
    )

    assert dataset._write_condition_to_input is True
    assert dataset._concat_condition_to_input is False


@pytest.mark.parametrize(
    "feature",
    [
        "year",
        "lead_time",
        "month_sin",
        "month_cos",
    ],
)
def test_getitem_individual_time_features(
    monkeypatch,
    feature,
):
    dataset = make_dataset_object(
        features=[feature],
    )

    patch_item_sources(
        monkeypatch,
        model=xr.DataArray(np.asarray([1.0])),
    )

    result = dataset[0]

    assert result["added_features"] is not None
    assert result["added_features"].shape[0] == 1


@pytest.mark.pruned
def test_getitem_all_time_features(
    monkeypatch,
):
    features = [
        "year",
        "lead_time",
        "month_sin",
        "month_cos",
    ]
    dataset = make_dataset_object(
        features=features,
    )

    patch_item_sources(
        monkeypatch,
        model=xr.DataArray(np.asarray([1.0])),
    )

    result = dataset[0]

    assert result["added_features"].shape == (4,)


@pytest.mark.pruned
def test_getitem_metadata_without_ensemble(
    monkeypatch,
):
    dataset = make_dataset_object(
        return_metadata=True,
        include_ensembles=False,
        features=None,
    )

    patch_item_sources(
        monkeypatch,
        model=xr.DataArray(np.asarray([1.0])),
    )

    result, metadata = dataset[0]

    assert isinstance(result, dict)
    assert metadata == {
        "year": 2000.0,
        "lead_time": 1.0,
    }


@pytest.mark.pruned
def test_getitem_metadata_with_multiple_coordinates(
    monkeypatch,
):
    dataset = make_dataset_object(
        return_metadata=True,
        include_ensembles=True,
        features=None,
    )
    dataset.sample_coords = {
        "year": np.asarray([2000.0, 2001.0]),
        "lead_time": np.asarray([1.0, 2.0]),
        "ensembles": np.asarray([0, 1]),
    }

    patch_item_sources(
        monkeypatch,
        model=xr.DataArray(np.asarray([2.0])),
    )

    _, metadata = dataset[1]

    assert metadata == {
        "year": 2001.0,
        "lead_time": 2.0,
        "ensembles": 1,
    }


@pytest.mark.pruned
def test_getitem_returns_float32(
    monkeypatch,
):
    dataset = make_dataset_object(
        features=["year"],
    )

    patch_item_sources(
        monkeypatch,
        model=xr.DataArray(
            np.asarray(
                [1.0],
                dtype=np.float64,
            )
        ),
    )

    result = dataset[0]

    assert result["input"].dtype == torch.float32
    assert result["added_features"].dtype == torch.float32


@pytest.mark.pruned
def test_from_train_does_not_copy_time_features(
    monkeypatch,
):
    model = DummyDataConfig("model")

    train = SimpleNamespace(
        condition_method="static",
        time_features=[
            "year",
            "lead_time",
        ],
        lead_months=np.asarray([1, 2]),
        observation=object(),
        effective_condition=None,
        _using_model_data_as_condition=False,
        model=model,
        condition=None,
    )

    monkeypatch.setattr(
        ("cccma_ppp.inference.dataset.InferenceDatasetConfig"),
        lambda **kwargs: kwargs,
    )

    result = _from_train(train)

    assert "time_features" not in result


@pytest.mark.pruned
def test_from_train_copies_condition_method(
    monkeypatch,
):
    model = DummyDataConfig("model")
    train = make_train_config(
        observation=object(),
        model=model,
    )
    train.condition_method = "ensemble_mean"

    monkeypatch.setattr(
        ("cccma_ppp.inference.dataset.InferenceDatasetConfig"),
        lambda **kwargs: kwargs,
    )

    result = _from_train(train)

    assert result["condition_method"] == "ensemble_mean"


@pytest.mark.pruned
def test_from_train_deepcopies_model(
    monkeypatch,
):
    model = DummyDataConfig("model")
    train = make_train_config(
        observation=object(),
        model=model,
    )

    monkeypatch.setattr(
        ("cccma_ppp.inference.dataset.InferenceDatasetConfig"),
        lambda **kwargs: kwargs,
    )

    result = _from_train(train)

    assert result["model"] is not model
    assert result["model"].paths == model.paths


@pytest.mark.pruned
def test_from_train_deepcopies_condition(
    monkeypatch,
):
    model = DummyDataConfig("model")
    condition = DummyDataConfig("condition")
    train = make_train_config(
        observation=object(),
        effective_condition=condition,
        model=model,
        condition=condition,
    )

    monkeypatch.setattr(
        ("cccma_ppp.inference.dataset.InferenceDatasetConfig"),
        lambda **kwargs: kwargs,
    )

    result = _from_train(train)

    assert result["condition"] is not condition
    assert result["condition"].paths == condition.paths


@pytest.mark.pruned
def test_from_train_deepcopies_lead_months(
    monkeypatch,
):
    model = DummyDataConfig("model")
    lead_months = np.asarray([1, 3, 6])
    train = make_train_config(
        observation=object(),
        model=model,
    )
    train.lead_months = lead_months

    monkeypatch.setattr(
        ("cccma_ppp.inference.dataset.InferenceDatasetConfig"),
        lambda **kwargs: kwargs,
    )

    result = _from_train(train)

    np.testing.assert_array_equal(
        result["lead_months"],
        lead_months,
    )
    assert result["lead_months"] is not lead_months