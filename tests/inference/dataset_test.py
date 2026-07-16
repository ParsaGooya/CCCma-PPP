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
        lead_time=3,
        ensembles=True,
        lat_size=2,
        lon_size=3,
    ):
        self.sizes = {"lead_time": lead_time}
        self.coords = {
            "lat": xr.DataArray(
                np.arange(lat_size),
                dims="lat",
            ),
            "lon": xr.DataArray(
                np.arange(lon_size),
                dims="lon",
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
        self.loaded_dir = None
        self.added = []

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
        year_range=None,
        lead_time=3,
        names=None,
    ):
        self.names = names or [name]
        self.info = DummyInfo(
            lead_time=lead_time,
            ensembles=ensembles,
        )
        self.ensemble_mean = ensemble_mean
        self.ensemble_list = None
        self.year_range = (
            np.asarray(year_range)
            if year_range is not None
            else np.array([2000, 2001, 2002])
        )
        self.preprocessing_pipeline = DummyPipeline(name)
        self.list_paths = [f"{name}.nc"]
        self.concat_dim = None
        self.rename_dict = {}


class DummyOperator:
    def __init__(self):
        self.load_calls = []
        self.add_calls = []

    def _load_fitted_preprocessors(self, load_dir=None):
        self.load_calls.append(load_dir)

    def _add_fitted_preprocessor(self, preprocessor, index=0):
        self.add_calls.append((preprocessor, index))


def make_runtime_config(
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
            np.asarray(lead_months) if lead_months is not None else np.array([1, 2, 3])
        ),
        _fitted_preprocessors=fitted,
        _using_model_data_as_condition=using_model_condition,
        effective_condition=effective_condition,
        effective_input=model if model is not None else condition,
    )


def make_config_check_object(
    model=None,
    condition=None,
    method=None,
    effective_condition=None,
    using_model_condition=False,
):
    return SimpleNamespace(
        model=model,
        condition=condition,
        condition_method=method,
        effective_condition=effective_condition,
        _using_model_data_as_condition=using_model_condition,
    )


@pytest.fixture
def fake_loaded_data():
    return xr.DataArray(
        np.arange(
            2 * 3 * 3 * 2 * 3,
            dtype=np.float32,
        ).reshape(2, 3, 3, 2, 3),
        dims=(
            "ensembles",
            "year",
            "lead_time",
            "lat",
            "lon",
        ),
        coords={
            "ensembles": [0, 1],
            "year": [2000, 2001, 2002],
            "lead_time": [1, 2, 3],
            "lat": [0, 1],
            "lon": [0, 1, 2],
        },
    )


@pytest.fixture(autouse=True)
def patch_external_helpers(monkeypatch, fake_loaded_data):
    monkeypatch.setattr(
        "cccma_ppp.inference.dataset._load_xarray_data",
        lambda *args, **kwargs: fake_loaded_data.copy(),
    )

    monkeypatch.setattr(
        "cccma_ppp.inference.dataset._unwrap_data_variables",
        lambda value: value,
    )

    monkeypatch.setattr(
        "cccma_ppp.inference.dataset._create_train_mask",
        lambda years, lead_times: xr.DataArray(
            np.zeros(
                (
                    len(years),
                    len(lead_times),
                ),
                dtype=bool,
            ),
            dims=("year", "lead_time"),
            coords={
                "year": years,
                "lead_time": lead_times,
            },
        ),
    )

    monkeypatch.setattr(
        "cccma_ppp.inference.dataset._get_time_features",
        lambda *args, **kwargs: None,
    )


def test_check_model_none_valid():
    cfg = make_config_check_object(
        model=None,
        method="same_member",
    )

    assert InferenceDatasetConfig._check_model(cfg) is cfg


def test_check_model_non_same_member_valid():
    cfg = make_config_check_object(
        model=DummyDataConfig(),
        method="cross_ensemble",
    )

    assert InferenceDatasetConfig._check_model(cfg) is cfg


def test_check_model_same_member_non_mean_valid():
    cfg = make_config_check_object(
        model=DummyDataConfig(
            ensemble_mean=False,
        ),
        method="same_member",
    )

    assert InferenceDatasetConfig._check_model(cfg) is cfg


def test_check_model_same_member_mean_raises():
    cfg = make_config_check_object(
        model=DummyDataConfig(
            ensemble_mean=True,
        ),
        method="same_member",
    )

    with pytest.raises(ValueError):
        InferenceDatasetConfig._check_model(cfg)


def test_check_condition_requires_method():
    condition = DummyDataConfig()

    cfg = make_config_check_object(
        condition=condition,
        method=None,
        effective_condition=condition,
    )

    with pytest.raises(ValueError):
        InferenceDatasetConfig._check_condition(cfg)


@pytest.mark.parametrize(
    "method",
    [
        "cross_ensemble",
        "same_member",
    ],
)
def test_check_condition_ensemble_methods_reject_mean(method):
    condition = DummyDataConfig(
        ensemble_mean=True,
    )

    cfg = make_config_check_object(
        condition=condition,
        method=method,
        effective_condition=condition,
    )

    with pytest.raises(ValueError):
        InferenceDatasetConfig._check_condition(cfg)


@pytest.mark.parametrize(
    "method",
    [
        "cross_ensemble",
        "same_member",
    ],
)
def test_check_condition_ensemble_methods_require_ensembles(method):
    condition = DummyDataConfig(
        ensembles=False,
    )

    cfg = make_config_check_object(
        condition=condition,
        method=method,
        effective_condition=condition,
    )

    with pytest.raises(ValueError):
        InferenceDatasetConfig._check_condition(cfg)


@pytest.mark.parametrize(
    "method",
    [
        "cross_ensemble",
        "same_member",
    ],
)
def test_check_condition_ensemble_methods_valid(method):
    condition = DummyDataConfig(
        ensemble_mean=False,
        ensembles=True,
    )

    cfg = make_config_check_object(
        condition=condition,
        method=method,
        effective_condition=condition,
    )

    assert InferenceDatasetConfig._check_condition(cfg) is cfg


def test_check_condition_ensemble_mean_requires_mean():
    condition = DummyDataConfig(
        ensemble_mean=False,
    )

    cfg = make_config_check_object(
        condition=condition,
        method="ensemble_mean",
        effective_condition=condition,
    )

    with pytest.raises(ValueError):
        InferenceDatasetConfig._check_condition(cfg)


def test_check_condition_ensemble_mean_valid():
    condition = DummyDataConfig(
        ensemble_mean=True,
    )

    cfg = make_config_check_object(
        condition=condition,
        method="ensemble_mean",
        effective_condition=condition,
    )

    assert InferenceDatasetConfig._check_condition(cfg) is cfg


def test_check_condition_static_rejects_ensemble_list():
    condition = DummyDataConfig()
    condition.ensemble_list = [0, 1]

    cfg = make_config_check_object(
        condition=condition,
        method="static",
        effective_condition=condition,
    )

    with pytest.raises(ValueError):
        InferenceDatasetConfig._check_condition(cfg)


def test_check_condition_static_rejects_model_as_condition():
    model = DummyDataConfig()

    cfg = make_config_check_object(
        model=model,
        method="static",
        effective_condition=model,
        using_model_condition=True,
    )

    with pytest.raises(ValueError):
        InferenceDatasetConfig._check_condition(cfg)


def test_check_condition_static_valid():
    condition = DummyDataConfig()

    cfg = make_config_check_object(
        condition=condition,
        method="static",
        effective_condition=condition,
        using_model_condition=False,
    )

    assert InferenceDatasetConfig._check_condition(cfg) is cfg


def test_check_condition_static_requires_dataset():
    cfg = make_config_check_object(
        condition=None,
        method="static",
        effective_condition=None,
    )

    with pytest.raises(ValueError):
        InferenceDatasetConfig._check_condition(cfg)


def test_check_condition_none_nonstatic_valid():
    cfg = make_config_check_object(
        condition=None,
        method=None,
        effective_condition=None,
    )

    assert InferenceDatasetConfig._check_condition(cfg) is cfg


def test_effective_input_returns_model():
    model = DummyDataConfig()
    condition = DummyDataConfig()

    cfg = InferenceDatasetConfig.__new__(InferenceDatasetConfig)
    cfg.model = model
    cfg.condition = condition

    assert cfg.effective_input is model


def test_effective_input_returns_condition():
    condition = DummyDataConfig()

    cfg = InferenceDatasetConfig.__new__(InferenceDatasetConfig)
    cfg.model = None
    cfg.condition = condition

    assert cfg.effective_input is condition


def test_num_input_lead_months_from_model():
    model = DummyDataConfig(
        lead_time=6,
    )

    cfg = InferenceDatasetConfig.__new__(InferenceDatasetConfig)
    cfg.model = model
    cfg.condition = None

    assert cfg.num_input_lead_months == 6


def test_num_input_lead_months_from_condition():
    condition = DummyDataConfig(
        lead_time=9,
    )

    cfg = InferenceDatasetConfig.__new__(InferenceDatasetConfig)
    cfg.model = None
    cfg.condition = condition

    assert cfg.num_input_lead_months == 9


def test_common_time_condition_only():
    condition = DummyDataConfig(
        year_range=[2000, 2001],
    )

    cfg = InferenceDatasetConfig.__new__(InferenceDatasetConfig)
    cfg.model = None
    cfg.condition = condition

    assert np.array_equal(
        cfg.get_common_time,
        np.array([2000, 2001]),
    )


def test_common_time_model_only():
    model = DummyDataConfig(
        year_range=[2001, 2002],
    )

    cfg = InferenceDatasetConfig.__new__(InferenceDatasetConfig)
    cfg.model = model
    cfg.condition = None

    assert np.array_equal(
        cfg.get_common_time,
        np.array([2001, 2002]),
    )


def test_common_time_model_and_condition_intersection():
    model = DummyDataConfig(
        year_range=[2000, 2001, 2002],
    )
    condition = DummyDataConfig(
        year_range=[2001, 2002, 2003],
    )

    cfg = InferenceDatasetConfig.__new__(InferenceDatasetConfig)
    cfg.model = model
    cfg.condition = condition

    assert np.array_equal(
        cfg.get_common_time,
        np.array([2001, 2002]),
    )


def test_available_inference_years_zero_lead_years():
    model = DummyDataConfig(
        year_range=[2000, 2001, 2002],
    )

    cfg = InferenceDatasetConfig.__new__(InferenceDatasetConfig)
    cfg.model = model
    cfg.condition = None
    cfg.lead_months = np.array([1])

    assert np.array_equal(
        cfg.available_inference_years,
        np.array([2000, 2001, 2002, 2003]),
    )


def test_available_inference_years_lead_adjustment():
    model = DummyDataConfig(
        year_range=[2000, 2001, 2002, 2003],
    )

    cfg = InferenceDatasetConfig.__new__(InferenceDatasetConfig)
    cfg.model = model
    cfg.condition = None
    cfg.lead_months = np.array([24])

    assert np.array_equal(
        cfg.available_inference_years,
        np.array([2000, 2001, 2002]),
    )


def test_load_fitted_preprocessors_delegates(monkeypatch):
    cfg = InferenceDatasetConfig.__new__(InferenceDatasetConfig)
    operator = DummyOperator()

    monkeypatch.setattr(
        InferenceDatasetConfig,
        "ds_operator",
        property(lambda self: operator),
    )

    cfg._load_fitted_preprocessors("load-dir")

    assert operator.load_calls == ["load-dir"]


def test_add_fitted_preprocessor_delegates(monkeypatch):
    cfg = InferenceDatasetConfig.__new__(InferenceDatasetConfig)
    operator = DummyOperator()
    module = object()

    monkeypatch.setattr(
        InferenceDatasetConfig,
        "ds_operator",
        property(lambda self: operator),
    )

    cfg._add_fitted_preprocessor(
        module,
        index=4,
    )

    assert operator.add_calls == [(module, 4)]


def test_build_dataset_wrapper(monkeypatch):
    cfg = InferenceDatasetConfig.__new__(InferenceDatasetConfig)

    monkeypatch.setattr(
        InferenceDataset,
        "__post_init__",
        lambda self: None,
    )

    dataset = cfg.build_dataset(
        years=np.array([2000]),
        return_metadata=True,
    )

    assert isinstance(dataset, InferenceDataset)
    assert np.array_equal(
        dataset.requested_years,
        np.array([2000]),
    )
    assert dataset.return_metadata is True


def test_dataset_requires_fitted_preprocessors():
    config = make_runtime_config(
        model=DummyDataConfig(),
        fitted=False,
    )

    with pytest.raises(RuntimeError):
        InferenceDataset(
            config=config,
            requested_years=[2000],
        )


def test_dataset_rejects_invalid_years():
    model = DummyDataConfig(
        year_range=[2000, 2001],
    )

    config = make_runtime_config(
        model=model,
        fitted=True,
    )

    config.available_inference_years = np.array([2000, 2001])

    with pytest.raises(ValueError):
        InferenceDataset(
            config=config,
            requested_years=[1999],
        )


def test_dataset_loads_model_only():
    model = DummyDataConfig()

    config = make_runtime_config(
        model=model,
        fitted=True,
    )
    config.available_inference_years = np.array([2000, 2001, 2002])

    dataset = InferenceDataset(
        config=config,
        requested_years=[2000],
    )

    assert dataset.model_dataset is not None
    assert dataset.condition_dataset is None


def test_dataset_loads_condition_only():
    condition = DummyDataConfig()

    config = make_runtime_config(
        model=None,
        condition=condition,
        method="static",
        effective_condition=condition,
        fitted=True,
    )
    config.available_inference_years = np.array([2000, 2001, 2002])

    dataset = InferenceDataset(
        config=config,
        requested_years=[2000],
    )

    assert dataset.model_dataset is None
    assert dataset.condition_dataset is not None


def test_dataset_loads_model_and_condition():
    model = DummyDataConfig("model")
    condition = DummyDataConfig("condition")

    config = make_runtime_config(
        model=model,
        condition=condition,
        method="static",
        effective_condition=condition,
        fitted=True,
    )
    config.available_inference_years = np.array([2000, 2001, 2002])

    dataset = InferenceDataset(
        config=config,
        requested_years=[2000],
    )

    assert dataset.model_dataset is not None
    assert dataset.condition_dataset is not None


@pytest.mark.parametrize(
    "using_model,model_present,expected",
    [
        (False, True, True),
        (True, True, False),
        (False, False, False),
        (True, False, False),
    ],
)
def test_load_model_property_matrix(
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
def test_write_condition_to_input_matrix(
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
def test_concat_condition_to_input_matrix(
    monkeypatch,
    write_condition,
    condition_present,
    expected,
):
    dataset = object.__new__(InferenceDataset)

    dataset.config = SimpleNamespace(
        condition=(DummyDataConfig() if condition_present else None),
    )

    monkeypatch.setattr(
        InferenceDataset,
        "_write_condition_to_input",
        property(lambda self: write_condition),
    )

    assert dataset._concat_condition_to_input is expected


def test_prepare_mask_adds_ensemble_dimension():
    model = DummyDataConfig(
        ensemble_mean=False,
        ensembles=True,
    )

    config = make_runtime_config(
        model=model,
        lead_months=[1, 2],
    )

    dataset = object.__new__(InferenceDataset)
    dataset.config = config
    dataset.requested_years = np.array([2000])

    mask = dataset._prepare_mask()

    assert "ensembles" in mask.dims
    assert "year" in mask.dims
    assert "lead_time" in mask.dims


def test_prepare_mask_skips_ensemble_for_mean():
    model = DummyDataConfig(
        ensemble_mean=True,
        ensembles=True,
    )

    config = make_runtime_config(
        model=model,
        lead_months=[1, 2],
    )

    dataset = object.__new__(InferenceDataset)
    dataset.config = config
    dataset.requested_years = np.array([2000])

    mask = dataset._prepare_mask()

    assert "ensembles" not in mask.dims


def test_prepare_mask_skips_missing_ensemble_coords():
    model = DummyDataConfig(
        ensemble_mean=False,
        ensembles=False,
    )

    config = make_runtime_config(
        model=model,
        lead_months=[1, 2],
    )

    dataset = object.__new__(InferenceDataset)
    dataset.config = config
    dataset.requested_years = np.array([2000])

    mask = dataset._prepare_mask()

    assert "ensembles" not in mask.dims


def test_load_xarray_data_with_ensemble_selection(monkeypatch):
    model = DummyDataConfig(
        ensembles=True,
    )
    captured = {}

    def fake_load(*args, **kwargs):
        captured.update(kwargs)
        return "loaded"

    monkeypatch.setattr(
        "cccma_ppp.inference.dataset._load_xarray_data",
        fake_load,
    )

    dataset = object.__new__(InferenceDataset)

    result = dataset._load_xarray_data(model)

    assert result == "loaded"
    assert "ensembles" in captured["selection"]


def test_load_xarray_data_without_ensemble_selection(monkeypatch):
    model = DummyDataConfig(
        ensembles=False,
    )
    captured = {}

    def fake_load(*args, **kwargs):
        captured.update(kwargs)
        return "loaded"

    monkeypatch.setattr(
        "cccma_ppp.inference.dataset._load_xarray_data",
        fake_load,
    )

    dataset = object.__new__(InferenceDataset)

    result = dataset._load_xarray_data(model)

    assert result == "loaded"
    assert captured["selection"] is None


def test_get_model_indexes_with_ensembles():
    model = DummyDataConfig()

    config = make_runtime_config(
        model=model,
    )

    dataset = object.__new__(InferenceDataset)
    dataset.config = config
    dataset.requested_years = np.array([2000])
    dataset.mask = dataset._prepare_mask()

    indexes = dataset.get_model_indexes()

    assert "ensembles" in indexes
    assert "year" in indexes
    assert "lead_time" in indexes


def test_get_model_indexes_without_ensembles():
    model = DummyDataConfig(
        ensemble_mean=True,
    )

    config = make_runtime_config(
        model=model,
    )

    dataset = object.__new__(InferenceDataset)
    dataset.config = config
    dataset.requested_years = np.array([2000])
    dataset.mask = dataset._prepare_mask()

    indexes = dataset.get_model_indexes()

    assert "ensembles" not in indexes


def test_get_cond_indexes_none_without_condition_dataset():
    dataset = object.__new__(InferenceDataset)
    dataset.condition_dataset = None

    result = dataset.get_cond_indexes(
        {
            "year": np.array([2000]),
            "lead_time": np.array([1]),
        }
    )

    assert result is None


def test_get_cond_indexes_none_for_static():
    dataset = object.__new__(InferenceDataset)
    dataset.condition_dataset = xr.DataArray([1])
    dataset.config = SimpleNamespace(
        condition_method="static",
    )

    result = dataset.get_cond_indexes(
        {
            "year": np.array([2000]),
            "lead_time": np.array([1]),
        }
    )

    assert result is None


def test_get_cond_indexes_cross_ensemble():
    condition = DummyDataConfig()

    dataset = object.__new__(InferenceDataset)
    dataset.condition_dataset = xr.DataArray([1])
    dataset.config = SimpleNamespace(
        condition_method="cross_ensemble",
        effective_condition=condition,
    )

    indexes = dataset.get_cond_indexes(
        {
            "year": np.array([2000, 2001]),
            "lead_time": np.array([1, 2]),
        }
    )

    assert "year" in indexes
    assert "lead_time" in indexes
    assert "ensembles" in indexes
    assert len(indexes["ensembles"]) == 2


def test_get_cond_indexes_same_member():
    condition = DummyDataConfig()

    dataset = object.__new__(InferenceDataset)
    dataset.condition_dataset = xr.DataArray([1])
    dataset.config = SimpleNamespace(
        condition_method="same_member",
        effective_condition=condition,
    )

    model_indexes = {
        "year": np.array([2000, 2001]),
        "lead_time": np.array([1, 2]),
        "ensembles": np.array([0, 1]),
    }

    indexes = dataset.get_cond_indexes(model_indexes)

    assert np.array_equal(
        indexes["ensembles"],
        model_indexes["ensembles"],
    )


def test_get_cond_indexes_other_nonstatic_method():
    condition = DummyDataConfig()

    dataset = object.__new__(InferenceDataset)
    dataset.condition_dataset = xr.DataArray([1])
    dataset.config = SimpleNamespace(
        condition_method="other",
        effective_condition=condition,
    )

    indexes = dataset.get_cond_indexes(
        {
            "year": np.array([2000]),
            "lead_time": np.array([1]),
        }
    )

    assert "year" in indexes
    assert "lead_time" in indexes
    assert "ensembles" not in indexes


def test_get_input_shape_spatial():
    model = DummyDataConfig()

    config = make_runtime_config(
        model=model,
    )

    dataset = object.__new__(InferenceDataset)
    dataset.config = config

    assert dataset.get_input_shape() == (2, 3)


def test_get_input_shape_spatial_with_condition_concat(
    monkeypatch,
):
    model = DummyDataConfig(
        names=["tas"],
    )
    condition = DummyDataConfig(
        names=["psl", "pr"],
    )

    config = make_runtime_config(
        model=model,
        condition=condition,
        effective_condition=condition,
    )

    dataset = object.__new__(InferenceDataset)
    dataset.config = config

    monkeypatch.setattr(
        InferenceDataset,
        "_concat_condition_to_input",
        property(lambda self: True),
    )

    assert dataset.get_input_shape() == (2, 3)


def test_get_input_shape_flattened(monkeypatch):
    class FakeFlatten:
        pass

    monkeypatch.setattr(
        "cccma_ppp.preprocessing.utils_preprocessing.Flattennanremove",
        FakeFlatten,
    )

    model = DummyDataConfig(
        names=["tas"],
    )
    model.preprocessing_pipeline.fitted_preprocessors = [FakeFlatten()]

    config = make_runtime_config(
        model=model,
    )

    dataset = object.__new__(InferenceDataset)
    dataset.config = config

    assert dataset.get_input_shape() == (4,)


def test_get_input_shape_flattened_concat(monkeypatch):
    class FakeFlatten:
        pass

    monkeypatch.setattr(
        "cccma_ppp.preprocessing.utils_preprocessing.Flattennanremove",
        FakeFlatten,
    )

    model = DummyDataConfig(
        names=["tas"],
    )
    condition = DummyDataConfig(
        names=["psl", "pr"],
    )

    model.preprocessing_pipeline.fitted_preprocessors = [FakeFlatten()]

    config = make_runtime_config(
        model=model,
        condition=condition,
        effective_condition=condition,
    )

    dataset = object.__new__(InferenceDataset)
    dataset.config = config

    monkeypatch.setattr(
        InferenceDataset,
        "_concat_condition_to_input",
        property(lambda self: True),
    )

    assert dataset.get_input_shape() == (12,)


@pytest.mark.parametrize(
    "features,expected",
    [
        (None, 0),
        ([], 0),
        (["year"], 1),
        (["year", "month_sin", "month_cos"], 3),
    ],
)
def test_get_added_features_dim(
    features,
    expected,
):
    dataset = object.__new__(InferenceDataset)
    dataset.config = SimpleNamespace(
        time_features=features,
    )

    assert dataset.get_added_features_dim() == expected


def test_index_condition_dataset_none():
    dataset = object.__new__(InferenceDataset)
    dataset.condition_dataset = None

    assert dataset._index_condition_dataset(0) is None


def test_index_condition_dataset_static():
    condition = DummyDataConfig()

    dataset = object.__new__(InferenceDataset)
    dataset.config = SimpleNamespace(
        condition_method="static",
        effective_condition=condition,
    )
    dataset.condition_dataset = xr.DataArray(
        [1.0, 2.0],
        dims="x",
    )
    dataset.cond_indexes = None

    result = dataset._index_condition_dataset(0)

    assert result is not None
    assert len(condition.preprocessing_pipeline.transform_calls) == 1


def test_index_condition_dataset_cross_ensemble():
    condition = DummyDataConfig()

    dataset = object.__new__(InferenceDataset)
    dataset.config = SimpleNamespace(
        condition_method="cross_ensemble",
        effective_condition=condition,
    )
    dataset.condition_dataset = xr.DataArray(
        np.ones((2, 2, 2)),
        dims=(
            "ensembles",
            "year",
            "lead_time",
        ),
        coords={
            "ensembles": [0, 1],
            "year": [2000, 2001],
            "lead_time": [1, 2],
        },
    )
    dataset.cond_indexes = {
        "year": np.array([2000]),
        "lead_time": np.array([1]),
        "ensembles": np.array([0]),
    }

    result = dataset._index_condition_dataset(0)

    assert result is not None


def test_index_condition_dataset_nonstatic_without_ensembles():
    condition = DummyDataConfig()

    dataset = object.__new__(InferenceDataset)
    dataset.config = SimpleNamespace(
        condition_method="other",
        effective_condition=condition,
    )
    dataset.condition_dataset = xr.DataArray(
        np.ones((2, 2)),
        dims=("year", "lead_time"),
        coords={
            "year": [2000, 2001],
            "lead_time": [1, 2],
        },
    )
    dataset.cond_indexes = {
        "year": np.array([2000]),
        "lead_time": np.array([1]),
    }

    result = dataset._index_condition_dataset(0)

    assert result is not None


def test_index_model_dataset_returns_none_when_not_loaded(
    monkeypatch,
):
    dataset = object.__new__(InferenceDataset)

    monkeypatch.setattr(
        InferenceDataset,
        "_load_model",
        property(lambda self: False),
    )

    assert dataset._index_model_dataset(0) is None


def test_index_model_dataset_with_ensemble(monkeypatch):
    model = DummyDataConfig()

    dataset = object.__new__(InferenceDataset)
    dataset.config = SimpleNamespace(
        model=model,
    )
    dataset.model_dataset = xr.DataArray(
        np.ones((2, 2, 2)),
        dims=(
            "ensembles",
            "year",
            "lead_time",
        ),
        coords={
            "ensembles": [0, 1],
            "year": [2000, 2001],
            "lead_time": [1, 2],
        },
    )
    dataset.model_indexes = {
        "year": np.array([2000]),
        "lead_time": np.array([1]),
        "ensembles": np.array([0]),
    }

    monkeypatch.setattr(
        InferenceDataset,
        "_load_model",
        property(lambda self: True),
    )

    result = dataset._index_model_dataset(0)

    assert result is not None


def test_index_model_dataset_without_ensemble(monkeypatch):
    model = DummyDataConfig(
        ensembles=False,
    )

    dataset = object.__new__(InferenceDataset)
    dataset.config = SimpleNamespace(
        model=model,
    )
    dataset.model_dataset = xr.DataArray(
        np.ones((2, 2)),
        dims=("year", "lead_time"),
        coords={
            "year": [2000, 2001],
            "lead_time": [1, 2],
        },
    )
    dataset.model_indexes = {
        "year": np.array([2000]),
        "lead_time": np.array([1]),
    }

    monkeypatch.setattr(
        InferenceDataset,
        "_load_model",
        property(lambda self: True),
    )

    result = dataset._index_model_dataset(0)

    assert result is not None


def make_getitem_dataset(
    return_metadata=False,
    include_ensembles=False,
):
    dataset = object.__new__(InferenceDataset)
    dataset.return_metadata = return_metadata
    dataset.requested_years = np.array([2000])

    dataset.model_indexes = {
        "year": np.array([2000]),
        "lead_time": np.array([1]),
    }

    if include_ensembles:
        dataset.model_indexes["ensembles"] = np.array([0])

    dataset.config = SimpleNamespace(
        time_features=None,
    )

    return dataset


def test_getitem_model_input_without_metadata(monkeypatch):
    dataset = make_getitem_dataset()

    model = xr.DataArray(np.array([1.0, 2.0]))

    monkeypatch.setattr(
        InferenceDataset,
        "_index_model_dataset",
        lambda self, ind: model,
    )
    monkeypatch.setattr(
        InferenceDataset,
        "_index_condition_dataset",
        lambda self, ind: None,
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

    sample = dataset[0]

    assert isinstance(sample, dict)
    assert torch.equal(
        sample["input"],
        torch.tensor([1.0, 2.0]),
    )
    assert sample["added_features"] is None


def test_getitem_condition_replaces_input(monkeypatch):
    dataset = make_getitem_dataset()

    model = xr.DataArray(np.array([1.0]))
    condition = xr.DataArray(np.array([5.0]))

    monkeypatch.setattr(
        InferenceDataset,
        "_index_model_dataset",
        lambda self, ind: model,
    )
    monkeypatch.setattr(
        InferenceDataset,
        "_index_condition_dataset",
        lambda self, ind: condition,
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

    sample = dataset[0]

    assert sample["input"].item() == 5.0


def test_getitem_concatenates_condition(monkeypatch):
    dataset = make_getitem_dataset()

    model = xr.DataArray(
        np.array([1.0]),
        dims="feature",
    )
    condition = xr.DataArray(
        np.array([2.0]),
        dims="feature",
    )

    monkeypatch.setattr(
        InferenceDataset,
        "_index_model_dataset",
        lambda self, ind: model,
    )
    monkeypatch.setattr(
        InferenceDataset,
        "_index_condition_dataset",
        lambda self, ind: condition,
    )
    monkeypatch.setattr(
        InferenceDataset,
        "_write_condition_to_input",
        property(lambda self: False),
    )
    monkeypatch.setattr(
        InferenceDataset,
        "_concat_condition_to_input",
        property(lambda self: True),
    )

    sample = dataset[0]

    assert sample["input"].shape == (2, 1)


def test_getitem_with_time_features(monkeypatch):
    dataset = make_getitem_dataset()

    model = xr.DataArray(np.array([1.0]))

    monkeypatch.setattr(
        InferenceDataset,
        "_index_model_dataset",
        lambda self, ind: model,
    )
    monkeypatch.setattr(
        InferenceDataset,
        "_index_condition_dataset",
        lambda self, ind: None,
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

    sample = dataset[0]

    assert torch.equal(
        sample["added_features"],
        torch.tensor([2000.0, 1.0]),
    )


def test_getitem_returns_metadata_without_ensemble(monkeypatch):
    dataset = make_getitem_dataset(
        return_metadata=True,
        include_ensembles=False,
    )

    model = xr.DataArray(np.array([1.0]))

    monkeypatch.setattr(
        InferenceDataset,
        "_index_model_dataset",
        lambda self, ind: model,
    )
    monkeypatch.setattr(
        InferenceDataset,
        "_index_condition_dataset",
        lambda self, ind: None,
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

    sample, metadata = dataset[0]

    assert isinstance(sample, dict)
    assert metadata == {
        "year": 2000.0,
        "lead_time": 1.0,
    }


def test_getitem_returns_metadata_with_ensemble(monkeypatch):
    dataset = make_getitem_dataset(
        return_metadata=True,
        include_ensembles=True,
    )

    model = xr.DataArray(np.array([1.0]))

    monkeypatch.setattr(
        InferenceDataset,
        "_index_model_dataset",
        lambda self, ind: model,
    )
    monkeypatch.setattr(
        InferenceDataset,
        "_index_condition_dataset",
        lambda self, ind: None,
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

    _, metadata = dataset[0]

    assert metadata["ensemble_id"] == 0


def test_len_uses_first_model_index():
    dataset = object.__new__(InferenceDataset)
    dataset.model_indexes = {
        "year": np.array([2000, 2001, 2002]),
        "lead_time": np.array([1, 2, 3]),
    }

    assert len(dataset) == 3


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
        _using_model_data_as_condition=using_model_condition,
        model=model,
        condition=condition,
    )


def test_from_train_observation_without_condition(monkeypatch):
    model = DummyDataConfig("model")
    train = make_train_config(
        observation=object(),
        effective_condition=None,
        using_model_condition=False,
        model=model,
    )
    captured = {}

    monkeypatch.setattr(
        "cccma_ppp.inference.dataset.InferenceDatasetConfig",
        lambda **kwargs: captured.update(kwargs) or kwargs,
    )

    result = _from_train(train)

    assert result["model"] is not model
    assert result["model"].names == model.names
    assert "condition" not in result


def test_from_train_model_as_condition(monkeypatch):
    model = DummyDataConfig("model")
    train = make_train_config(
        observation=None,
        effective_condition=model,
        using_model_condition=True,
        model=model,
    )
    captured = {}

    monkeypatch.setattr(
        "cccma_ppp.inference.dataset.InferenceDatasetConfig",
        lambda **kwargs: captured.update(kwargs) or kwargs,
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
        using_model_condition=False,
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
        observation=None,
        effective_condition=condition,
        using_model_condition=False,
        model=None,
        condition=condition,
    )

    monkeypatch.setattr(
        "cccma_ppp.inference.dataset.InferenceDatasetConfig",
        lambda **kwargs: kwargs,
    )

    result = _from_train(train)

    assert "model" not in result
    assert result["condition"].names == condition.names


def test_from_train_unresolvable_raises():
    train = make_train_config(
        observation=None,
        effective_condition=None,
        using_model_condition=False,
        model=None,
        condition=None,
    )

    with pytest.raises(ValueError):
        _from_train(train)


def test_from_train_deepcopies_shared_fields(monkeypatch):
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
    assert np.array_equal(
        result["lead_months"],
        lead_months,
    )
    assert result["lead_months"] is not lead_months
