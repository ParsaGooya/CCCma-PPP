from types import SimpleNamespace

import numpy as np
import pytest
import torch
import xarray as xr

from cccma_ppp.inference.dataset import (
    InferenceDataset,
    InferenceDatasetConfig,
    _from_train,
)


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

        self.sizes = {
            "lead_time": lead_time,
        }
        self.coords = {
            "lat": xr.DataArray(
                np.arange(lat_size),
                dims="lat",
            ),
            "lon": xr.DataArray(
                np.arange(lon_size),
                dims="lon",
            ),
            "year": xr.DataArray(
                years,
                dims="year",
            ),
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

    def _load_fitted_preprocessors(
        self,
        load_dir=None,
    ):
        self.load_calls.append(load_dir)

    def _add_fitted_preprocessor(
        self,
        preprocessor,
        index=0,
    ):
        self.add_calls.append((preprocessor, index))


def make_config(
    model=None,
    condition=None,
    method=None,
    using_model_condition=False,
    effective_condition=None,
    time_features=None,
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
        time_features=time_features,
        lead_months=(
            np.arange(1, 13) if lead_months is None else np.asarray(lead_months)
        ),
        _fitted_preprocessors=fitted,
        _using_model_data_as_condition=(using_model_condition),
        effective_condition=effective_condition,
        effective_input=(model if model is not None else condition),
    )


def make_config_object(
    model=None,
    condition=None,
    method=None,
    effective_condition=None,
    using_model_condition=False,
):
    config = InferenceDatasetConfig.__new__(InferenceDatasetConfig)
    config.model = model
    config.condition = condition
    config.condition_method = method
    config._effective_condition = effective_condition

    type(config).effective_condition = property(lambda self: self._effective_condition)
    type(config)._using_model_data_as_condition = property(
        lambda self: using_model_condition
    )

    return config


def make_dataset_object(
    config=None,
    return_metadata=False,
    include_ensembles=False,
):
    dataset = object.__new__(InferenceDataset)
    dataset.config = config or make_config(model=DummyDataConfig())
    dataset.return_metadata = return_metadata
    dataset.requested_years = np.array([2000])
    dataset.sample_coords = {
        "year": np.array([2000.0]),
        "lead_time": np.array([1.0]),
    }

    if include_ensembles:
        dataset.sample_coords["ensembles"] = np.array([0])

    return dataset


def test_check_model_none():
    config = make_config_object(
        model=None,
        method="same_member",
    )

    assert config._check_model() is config


def test_check_model_non_same_member():
    config = make_config_object(
        model=DummyDataConfig(),
        method="cross_ensemble",
    )

    assert config._check_model() is config


def test_check_model_same_member_non_mean():
    config = make_config_object(
        model=DummyDataConfig(ensemble_mean=False),
        method="same_member",
    )

    assert config._check_model() is config


def test_check_model_same_member_mean():
    config = make_config_object(
        model=DummyDataConfig(ensemble_mean=True),
        method="same_member",
    )

    with pytest.raises(ValueError):
        config._check_model()


def test_check_condition_requires_method():
    condition = DummyDataConfig()
    config = make_config_object(
        condition=condition,
        effective_condition=condition,
    )

    with pytest.raises(ValueError):
        config._check_condition()


@pytest.mark.parametrize(
    "method",
    [
        "cross_ensemble",
        "same_member",
    ],
)
def test_check_condition_ensemble_method_rejects_mean(
    method,
):
    condition = DummyDataConfig(ensemble_mean=True)
    config = make_config_object(
        condition=condition,
        method=method,
        effective_condition=condition,
    )

    with pytest.raises(ValueError):
        config._check_condition()


@pytest.mark.parametrize(
    "method",
    [
        "cross_ensemble",
        "same_member",
    ],
)
def test_check_condition_ensemble_method_requires_ensembles(
    method,
):
    condition = DummyDataConfig(ensembles=False)
    config = make_config_object(
        condition=condition,
        method=method,
        effective_condition=condition,
    )

    with pytest.raises(ValueError):
        config._check_condition()


@pytest.mark.parametrize(
    "method",
    [
        "cross_ensemble",
        "same_member",
    ],
)
def test_check_condition_ensemble_method_valid(
    method,
):
    condition = DummyDataConfig()
    config = make_config_object(
        condition=condition,
        method=method,
        effective_condition=condition,
    )

    assert config._check_condition() is config


@pytest.mark.pruned
def test_check_condition_ensemble_mean_requires_mean():
    condition = DummyDataConfig()
    config = make_config_object(
        condition=condition,
        method="ensemble_mean",
        effective_condition=condition,
    )

    with pytest.raises(ValueError):
        config._check_condition()


def test_check_condition_ensemble_mean_valid():
    condition = DummyDataConfig(ensemble_mean=True)
    config = make_config_object(
        condition=condition,
        method="ensemble_mean",
        effective_condition=condition,
    )

    assert config._check_condition() is config


@pytest.mark.pruned
def test_check_condition_static_ensemble_list():
    condition = DummyDataConfig()
    condition.ensemble_list = [0, 1]

    config = make_config_object(
        condition=condition,
        method="static",
        effective_condition=condition,
    )

    with pytest.raises(ValueError):
        config._check_condition()


@pytest.mark.pruned
def test_check_condition_static_model_condition():
    model = DummyDataConfig()
    config = make_config_object(
        model=model,
        method="static",
        effective_condition=model,
        using_model_condition=True,
    )

    with pytest.raises(ValueError):
        config._check_condition()


def test_check_condition_static_valid():
    condition = DummyDataConfig()
    config = make_config_object(
        condition=condition,
        method="static",
        effective_condition=condition,
    )

    assert config._check_condition() is config


def test_check_condition_static_requires_dataset():
    config = make_config_object(
        method="static",
        effective_condition=None,
    )

    with pytest.raises(ValueError):
        config._check_condition()


def test_check_condition_none():
    config = make_config_object(
        method=None,
        effective_condition=None,
    )

    assert config._check_condition() is config


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

    config = make_config_object(
        condition=condition,
    )

    assert config.effective_input is condition


def test_num_input_lead_months_model():
    config = make_config_object(model=DummyDataConfig(lead_time=6))

    assert config.num_input_lead_months == 6


def test_num_input_lead_months_condition():
    config = make_config_object(condition=DummyDataConfig(lead_time=9))

    assert config.num_input_lead_months == 9


def test_common_time_condition_only():
    condition = DummyDataConfig(years=[2000, 2001])
    config = make_config_object(condition=condition)

    np.testing.assert_array_equal(
        config.get_common_time,
        np.array([2000, 2001]),
    )


def test_common_time_model_only():
    model = DummyDataConfig(years=[2001, 2002])
    config = make_config_object(model=model)

    np.testing.assert_array_equal(
        config.get_common_time,
        np.array([2001, 2002]),
    )


def test_common_time_intersection():
    model = DummyDataConfig(years=[2000, 2001, 2002])
    condition = DummyDataConfig(years=[2001, 2002, 2003])
    config = make_config_object(
        model=model,
        condition=condition,
    )

    np.testing.assert_array_equal(
        config.get_common_time,
        np.array([2001, 2002]),
    )


@pytest.mark.pruned
def test_available_times():
    model = DummyDataConfig(years=[2000, 2001, 2002])
    config = make_config_object(model=model)

    np.testing.assert_array_equal(
        config.available_times,
        np.array([2000, 2001, 2002]),
    )


@pytest.mark.pruned
# Remove test due to no coverage
def test_load_fitted_preprocessors(
    monkeypatch,
):
    config = make_config_object()
    operator = DummyOperator()

    monkeypatch.setattr(
        InferenceDatasetConfig,
        "ds_operator",
        property(lambda self: operator),
    )

    config._load_fitted_preprocessors("load-dir")

    assert operator.load_calls == ["load-dir"]


@pytest.mark.pruned
# Remove test due to no coverage
def test_add_fitted_preprocessor(
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

    config._add_fitted_preprocessor(
        preprocessor,
        index=4,
    )

    assert operator.add_calls == [(preprocessor, 4)]


@pytest.mark.pruned
# Remove test due to no coverage
def test_build_dataset(monkeypatch):
    config = make_config_object()

    monkeypatch.setattr(
        InferenceDataset,
        "__post_init__",
        lambda self: None,
    )

    result = config.build_dataset(
        years=np.array([2000]),
        return_metadata=True,
        load=True,
    )

    assert isinstance(
        result,
        InferenceDataset,
    )
    np.testing.assert_array_equal(
        result.requested_years,
        np.array([2000]),
    )
    assert result.return_metadata is True
    assert result.load is True


@pytest.mark.parametrize(
    "using_model,model_present,expected",
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
        model=(DummyDataConfig() if model_present else None),
    )

    assert dataset._load_model is expected


@pytest.mark.parametrize(
    "using_model,model_present,expected",
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
        model=(DummyDataConfig() if model_present else None),
    )

    assert dataset._write_condition_to_input is expected


@pytest.mark.parametrize(
    "write_condition,condition_present,expected",
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
        condition=(DummyDataConfig() if condition_present else None)
    )

    monkeypatch.setattr(
        InferenceDataset,
        "_write_condition_to_input",
        property(lambda self: write_condition),
    )

    assert dataset._concat_condition_to_input is expected


def test_getitem_condition_input(
    monkeypatch,
):
    dataset = make_dataset_object(
        config=make_config(
            model=None,
            condition=DummyDataConfig(),
        )
    )

    model = xr.DataArray(np.array([1.0]))
    condition = xr.DataArray(np.array([5.0]))

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
        property(lambda self: True),
    )
    monkeypatch.setattr(
        InferenceDataset,
        "_concat_condition_to_input",
        property(lambda self: False),
    )
    monkeypatch.setattr(
        "cccma_ppp.inference.dataset._get_time_features",
        lambda *args, **kwargs: None,
    )

    result = dataset[0]

    assert result["input"].item() == 5.0


@pytest.mark.pruned
def test_getitem_time_features(
    monkeypatch,
):
    dataset = make_dataset_object()

    model = xr.DataArray(np.array([1.0]))

    monkeypatch.setattr(
        InferenceDataset,
        "_index_model_dataset",
        lambda self, index: model,
    )
    monkeypatch.setattr(
        InferenceDataset,
        "_index_condition_dataset",
        lambda self, index: None,
    )
    monkeypatch.setattr(
        InferenceDataset,
        "_write_condition_to_input",
        property(lambda self: False),
    )
    monkeypatch.setattr(
        InferenceDataset,
        "_concat_condition_to_input",
        property(lambda self: False),
    )
    monkeypatch.setattr(
        "cccma_ppp.inference.dataset._get_time_features",
        lambda *args, **kwargs: np.array([2000.0, 1.0]),
    )

    result = dataset[0]

    torch.testing.assert_close(
        result["added_features"],
        torch.tensor([2000.0, 1.0]),
    )


def test_getitem_metadata(
    monkeypatch,
):
    dataset = make_dataset_object(
        return_metadata=True,
        include_ensembles=True,
    )

    model = xr.DataArray(np.array([1.0]))

    monkeypatch.setattr(
        InferenceDataset,
        "_index_model_dataset",
        lambda self, index: model,
    )
    monkeypatch.setattr(
        InferenceDataset,
        "_index_condition_dataset",
        lambda self, index: None,
    )
    monkeypatch.setattr(
        InferenceDataset,
        "_write_condition_to_input",
        property(lambda self: False),
    )
    monkeypatch.setattr(
        InferenceDataset,
        "_concat_condition_to_input",
        property(lambda self: False),
    )
    monkeypatch.setattr(
        "cccma_ppp.inference.dataset._get_time_features",
        lambda *args, **kwargs: None,
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
        time_features=["year"],
        lead_months=np.array([1, 2]),
        observation=observation,
        effective_condition=effective_condition,
        _using_model_data_as_condition=(using_model_condition),
        model=model,
        condition=condition,
    )


def test_from_train_observation_only(
    monkeypatch,
):
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


def test_from_train_model_condition(
    monkeypatch,
):
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


def test_from_train_condition_with_observation(
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
        "cccma_ppp.inference.dataset.InferenceDatasetConfig",
        lambda **kwargs: kwargs,
    )

    result = _from_train(train)

    assert result["model"].names == model.names
    assert result["condition"].names == condition.names


def test_from_train_condition_without_observation(
    monkeypatch,
):
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
    train = make_train_config()

    with pytest.raises(ValueError):
        _from_train(train)


@pytest.mark.pruned
def test_from_train_deepcopies_fields(
    monkeypatch,
):
    model = DummyDataConfig("model")
    time_features = ["year"]
    lead_months = np.array([1, 2])

    train = SimpleNamespace(
        condition_method="static",
        time_features=time_features,
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

    assert result["time_features"] == time_features
    assert result["time_features"] is not time_features
    np.testing.assert_array_equal(
        result["lead_months"],
        lead_months,
    )
    assert result["lead_months"] is not lead_months