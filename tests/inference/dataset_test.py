import numpy as np
import pytest
from types import SimpleNamespace


import xarray as xr
import torch


from cccma_ppp.inference.dataset import (
    InferenceDataset,
    InferenceDatasetConfig,
    _from_train,
)


class DummyPipeline:
    def transform(self, x):
        return x


class DummyInfo:
    def __init__(self, ensembles=None):
        self.coords = {"ensembles": ensembles}
        self.sizes = {"lead_time": 12}


class DummyCondition:
    def __init__(
        self,
        ensemble_mean=False,
        ensemble_list=None,
        ensembles=None,
    ):
        self.ensemble_mean = ensemble_mean
        self.ensemble_list = ensemble_list
        self.info = DummyInfo(ensembles)


class DummyOperator:
    def __init__(self):
        self.loaded = False
        self.added = False

    def _load_fitted_preprocessors(
        self,
        load_dir=None,
    ):
        self.loaded = True

    def _add_fitted_preprocessor(
        self,
        preprocessor,
        index,
    ):
        self.added = True
        self.preprocessor = preprocessor
        self.index = index


class DummyDatasetConfig:
    def __init__(self, time_features=None):
        self.time_features = time_features


def test_condition_without_method_raises():
    cfg = object.__new__(InferenceDatasetConfig)

    cfg._effective_condition = DummyCondition(ensembles=[1])

    cfg.condition_method = None

    with pytest.raises(ValueError):
        cfg._check_condition()


def test_same_member_requires_ensemble_dimension():
    cfg = object.__new__(InferenceDatasetConfig)

    cfg.condition_method = "same_member"

    cfg._effective_condition = DummyCondition(ensembles=None)

    with pytest.raises(ValueError):
        cfg._check_condition()


def test_cross_ensemble_requires_ensemble_dimension():
    cfg = object.__new__(InferenceDatasetConfig)

    cfg.condition_method = "cross_ensemble"

    cfg._effective_condition = DummyCondition(ensembles=None)

    with pytest.raises(ValueError):
        cfg._check_condition()


def test_cross_ensemble_rejects_ensemble_mean():
    cfg = object.__new__(InferenceDatasetConfig)

    cfg.condition_method = "cross_ensemble"

    cfg._effective_condition = DummyCondition(
        ensemble_mean=True,
        ensembles=[1],
    )

    with pytest.raises(ValueError):
        cfg._check_condition()


def test_ensemble_mean_requires_ensemble_mean():
    cfg = object.__new__(InferenceDatasetConfig)

    cfg.condition_method = "ensemble_mean"

    cfg._effective_condition = DummyCondition(
        ensemble_mean=False,
        ensembles=[1],
    )

    with pytest.raises(ValueError):
        cfg._check_condition()


def test_static_requires_condition():
    cfg = object.__new__(InferenceDatasetConfig)

    cfg.condition_method = "static"
    cfg._effective_condition = None

    with pytest.raises(ValueError):
        cfg._check_condition()


def test_get_added_features_dim_none():
    ds = object.__new__(InferenceDataset)

    ds.config = DummyDatasetConfig(time_features=None)

    assert ds.get_added_features_dim() == 0


def test_get_added_features_dim_present():
    ds = object.__new__(InferenceDataset)

    ds.config = DummyDatasetConfig(
        time_features=[
            "year",
            "lead_time",
        ]
    )

    assert ds.get_added_features_dim() == 2


def test_get_cond_indexes_static():
    ds = object.__new__(InferenceDataset)

    ds.config = SimpleNamespace(condition_method="static")

    ds.condition_dataset = object()

    result = ds.get_cond_indexes(
        {
            "year": np.array([2000]),
            "lead_time": np.array([1]),
        }
    )

    assert result is None


def test_get_cond_indexes_same_member():
    ds = object.__new__(InferenceDataset)

    ds.config = SimpleNamespace(condition_method="same_member")

    ds.condition_dataset = object()

    result = ds.get_cond_indexes(
        {
            "year": np.array([2000]),
            "lead_time": np.array([1]),
            "ensembles": np.array([3]),
        }
    )

    assert result["ensembles"][0] == 3


def test_get_cond_indexes_no_condition_dataset():
    ds = object.__new__(InferenceDataset)

    ds.condition_dataset = None

    result = ds.get_cond_indexes(
        {
            "year": np.array([2000]),
        }
    )

    assert result is None


def test_len():
    ds = object.__new__(InferenceDataset)

    ds.model_indexes = {
        "year": np.array([2000, 2001, 2002]),
        "lead_time": np.array([1, 2, 3]),
    }

    assert len(ds) == 3


def test_load_fitted_preprocessors(monkeypatch):
    op = DummyOperator()

    monkeypatch.setattr(
        InferenceDatasetConfig,
        "ds_operator",
        property(lambda self: op),
    )

    cfg = object.__new__(InferenceDatasetConfig)

    cfg._load_fitted_preprocessors()

    assert op.loaded


def test_add_fitted_preprocessor(monkeypatch):
    op = DummyOperator()

    monkeypatch.setattr(
        InferenceDatasetConfig,
        "ds_operator",
        property(lambda self: op),
    )

    cfg = object.__new__(InferenceDatasetConfig)

    cfg._add_fitted_preprocessor(
        "dummy",
        5,
    )

    assert op.added
    assert op.preprocessor == "dummy"
    assert op.index == 5


def test_common_time_overlap():
    cfg = object.__new__(InferenceDatasetConfig)

    cfg.model = SimpleNamespace(year_range=np.array([2000, 2001, 2002]))

    cfg.condition = SimpleNamespace(year_range=np.array([2001, 2002, 2003]))

    result = cfg.get_common_time

    assert np.array_equal(
        result,
        np.array([2001, 2002]),
    )


def test_common_time_model_only():
    cfg = object.__new__(InferenceDatasetConfig)

    cfg.model = SimpleNamespace(year_range=np.array([2000, 2001, 2002]))

    cfg.condition = None

    result = cfg.get_common_time

    assert np.array_equal(
        result,
        np.array([2000, 2001, 2002]),
    )


def test_check_model_valid_configuration():
    cfg = object.__new__(InferenceDatasetConfig)

    cfg.model = SimpleNamespace(ensemble_mean=False)

    cfg.condition_method = "same_member"

    cfg._check_model()


def test_check_condition_valid_same_member():
    cfg = object.__new__(InferenceDatasetConfig)

    cfg.condition_method = "same_member"

    cfg._effective_condition = DummyCondition(
        ensemble_mean=False,
        ensembles=[1, 2, 3],
    )

    cfg._check_condition()


def test_check_condition_valid_cross_ensemble():
    cfg = object.__new__(InferenceDatasetConfig)

    cfg.condition_method = "cross_ensemble"

    cfg._effective_condition = DummyCondition(
        ensemble_mean=False,
        ensembles=[1, 2, 3],
    )

    cfg._check_condition()


def test_check_condition_valid_ensemble_mean():
    cfg = object.__new__(InferenceDatasetConfig)

    cfg.condition_method = "ensemble_mean"

    cfg._effective_condition = DummyCondition(
        ensemble_mean=True,
        ensembles=[1, 2, 3],
    )

    cfg._check_condition()


def test_common_time_with_condition():
    cfg = object.__new__(InferenceDatasetConfig)

    cfg.model = SimpleNamespace(year_range=np.array([2000, 2001, 2002]))

    cfg.condition = SimpleNamespace(year_range=np.array([2001, 2002, 2003]))

    result = cfg.get_common_time

    assert np.array_equal(
        result,
        np.array([2001, 2002]),
    )


def test_common_time_without_condition():
    cfg = object.__new__(InferenceDatasetConfig)

    cfg.model = SimpleNamespace(year_range=np.array([2000, 2001, 2002]))

    cfg.condition = None

    result = cfg.get_common_time

    assert np.array_equal(
        result,
        np.array([2000, 2001, 2002]),
    )


def test_num_input_lead_months():
    cfg = object.__new__(InferenceDatasetConfig)

    cfg.model = SimpleNamespace(info=SimpleNamespace(sizes={"lead_time": 12}))

    assert cfg.num_input_lead_months == 12


def test_available_inference_time_single_lead():
    cfg = object.__new__(InferenceDatasetConfig)

    cfg.lead_months = np.array([1])

    cfg.model = SimpleNamespace(year_range=np.array([2000, 2001, 2002]))

    cfg.condition = None

    result = cfg.available_inference_time

    assert len(result) > 0


def test_available_inference_time_multiple_leads():
    cfg = object.__new__(InferenceDatasetConfig)

    cfg.lead_months = np.array([1, 2, 3])

    cfg.model = SimpleNamespace(year_range=np.array([2000, 2001, 2002]))

    cfg.condition = None

    result = cfg.available_inference_time

    assert len(result) > 0


def test_build_dataset_returns_dataset(monkeypatch):
    created = {}

    class DummyInferenceDatasetCreated:
        def __init__(
            self,
            config,
            requested_years,
            return_metadata,
        ):
            created["config"] = config
            created["years"] = requested_years
            created["metadata"] = return_metadata

    monkeypatch.setattr(
        "cccma_ppp.inference.dataset.InferenceDataset",
        DummyInferenceDatasetCreated,
    )

    cfg = object.__new__(InferenceDatasetConfig)

    result = cfg.build_dataset(
        years=np.array([2000]),
        return_metadata=True,
    )

    assert result is not None
    assert created["metadata"] is True


def test_using_model_data_as_condition_true():
    cfg = object.__new__(InferenceDatasetConfig)

    cfg.model = SimpleNamespace()
    cfg.condition = None
    cfg.condition_method = "same_member"

    assert cfg._using_model_data_as_condition is True


def test_using_model_data_as_condition_false():
    model = SimpleNamespace(
        paths=["a"],
        names=["tas"],
        ensemble_list=[1],
    )

    condition = SimpleNamespace(
        paths=["b"],
        names=["tas"],
        ensemble_list=[1],
    )

    cfg = object.__new__(InferenceDatasetConfig)

    cfg.model = model
    cfg.condition = condition

    assert cfg._using_model_data_as_condition is False


def test_dataset_requires_preprocessors_loaded():
    cfg = SimpleNamespace(
        _fitted_preprocessors=False,
    )

    with pytest.raises(RuntimeError):
        InferenceDataset(
            config=cfg,
            requested_years=np.array([2000]),
        )


def test_dataset_invalid_years():
    cfg = SimpleNamespace(
        _fitted_preprocessors=True,
        available_train_time=np.array([2000, 2001]),
    )

    with pytest.raises(ValueError):
        InferenceDataset(
            config=cfg,
            requested_years=np.array([1990]),
        )


def test_load_model_property():
    ds = object.__new__(InferenceDataset)

    ds.config = SimpleNamespace(
        _using_model_data_as_condition=False,
    )

    assert ds._load_model is True


def test_load_model_property_false():
    ds = object.__new__(InferenceDataset)

    ds.config = SimpleNamespace(
        _using_model_data_as_condition=True,
    )

    assert ds._load_model is False


def test_write_condition_to_input_property():
    ds = object.__new__(InferenceDataset)

    ds.config = SimpleNamespace(
        _using_model_data_as_condition=True,
    )

    assert ds._write_condition_to_input is True


def test_write_condition_to_input_property_false():
    ds = object.__new__(InferenceDataset)

    ds.config = SimpleNamespace(
        _using_model_data_as_condition=False,
    )

    assert ds._write_condition_to_input is False


def test_concat_condition_to_input_property():
    ds = object.__new__(InferenceDataset)

    ds.config = SimpleNamespace(
        _using_model_data_as_condition=False,
        effective_condition=object(),
    )

    assert ds._concat_condition_to_input is True


def test_concat_condition_to_input_property_false():
    ds = object.__new__(InferenceDataset)

    ds.config = SimpleNamespace(
        _using_model_data_as_condition=False,
        effective_condition=None,
    )

    assert ds._concat_condition_to_input is False


class DummyFlatten:
    pass


def test_get_input_shape(monkeypatch):
    ds = object.__new__(InferenceDataset)

    ds.config = SimpleNamespace(
        model=object(),
        _using_model_data_as_condition=False,
        effective_condition=None,
    )

    ds.model_dataset = np.ones((10, 3, 4, 5))

    monkeypatch.setattr(
        InferenceDataset,
        "effective_input",
        property(
            lambda self: SimpleNamespace(
                names=["tas"],
                preprocessing_pipeline=SimpleNamespace(fitted_preprocessors=[]),
                info=SimpleNamespace(
                    coords={
                        "lat": np.arange(4),
                        "lon": np.arange(5),
                    }
                ),
            )
        ),
    )

    shape = ds.get_input_shape()

    assert shape is not None


def test_get_cond_indexes_same_member_copy():
    ds = object.__new__(InferenceDataset)

    ds.condition_dataset = object()

    ds.config = SimpleNamespace(condition_method="same_member")

    indexes = {
        "year": np.array([2000]),
        "lead_time": np.array([1]),
        "ensembles": np.array([5]),
    }

    result = ds.get_cond_indexes(indexes)

    assert np.array_equal(
        result["ensembles"],
        np.array([5]),
    )


def test_get_cond_indexes_cross_ensemble():
    ds = object.__new__(InferenceDataset)

    ds.condition_dataset = object()

    ds.config = SimpleNamespace(
        condition_method="cross_ensemble",
        effective_condition=SimpleNamespace(
            info=SimpleNamespace(coords={"ensembles": np.array([1, 2, 3])})
        ),
    )

    indexes = {
        "year": np.array([2000]),
        "lead_time": np.array([1]),
    }

    result = ds.get_cond_indexes(indexes)

    assert "ensembles" in result


def test_effective_input_model():
    ds = object.__new__(InferenceDataset)

    model = object()

    ds.config = SimpleNamespace(
        model=model,
    )

    assert ds.effective_input is model


def test_effective_input_condition():
    ds = object.__new__(InferenceDataset)

    condition = object()

    ds.config = SimpleNamespace(
        model=None,
        condition=condition,
    )

    assert ds.effective_input is condition


def test_concat_condition_true():
    ds = object.__new__(InferenceDataset)

    ds.config = SimpleNamespace(
        _using_model_data_as_condition=False,
        effective_condition=object(),
    )

    assert ds._concat_condition_to_input is True


def test_concat_condition_false():
    ds = object.__new__(InferenceDataset)

    ds.config = SimpleNamespace(
        _using_model_data_as_condition=True,
        effective_condition=object(),
    )

    assert ds._concat_condition_to_input is False


def test_len_lead_time_key():
    ds = object.__new__(InferenceDataset)

    ds.model_indexes = {
        "lead_time": np.arange(7),
    }

    assert len(ds) == 7


def test_get_cond_indexes_no_condition():
    ds = object.__new__(InferenceDataset)

    ds.condition_dataset = None

    assert (
        ds.get_cond_indexes(
            {
                "year": np.array([2000]),
                "lead_time": np.array([1]),
            }
        )
        is None
    )


def test_index_condition_dataset_none():
    ds = object.__new__(InferenceDataset)

    ds.condition_dataset = None

    assert ds._index_condition_dataset(0) is None


def _fake_da(value=1.0):
    return xr.DataArray(
        np.full((1, 1, 2, 2), value),
        dims=("channels", "lead_time", "lat", "lon"),
    )


def test_index_condition_dataset_static(monkeypatch):
    ds = object.__new__(InferenceDataset)

    class DummyPipeline:
        def transform(self, x):
            return x

    ds.condition_dataset = _fake_da()

    ds.config = SimpleNamespace(
        condition_method="static",
        effective_condition=SimpleNamespace(preprocessing_pipeline=DummyPipeline()),
    )

    monkeypatch.setattr(
        "cccma_ppp.inference.dataset._unwrap_data_variables",
        lambda x: x,
    )

    out = ds._index_condition_dataset(0)

    assert out is not None


def test_index_condition_dataset_dynamic(monkeypatch):
    ds = object.__new__(InferenceDataset)

    class DummyPipeline:
        def transform(self, x):
            return x

    ds.condition_dataset = xr.DataArray(
        np.ones((1, 1, 1, 2, 2)),
        dims=("ensembles", "year", "lead_time", "lat", "lon"),
        coords={
            "ensembles": [0],
            "year": [2000],
            "lead_time": [1],
        },
    )

    ds.cond_indexes = {
        "year": np.array([2000]),
        "lead_time": np.array([1]),
        "ensembles": np.array([0]),
    }

    ds.config = SimpleNamespace(
        condition_method="same_member",
        effective_condition=SimpleNamespace(preprocessing_pipeline=DummyPipeline()),
    )

    monkeypatch.setattr(
        "cccma_ppp.inference.dataset._unwrap_data_variables",
        lambda x: x,
    )

    out = ds._index_condition_dataset(0)

    assert out is not None


def test_getitem_write_condition(monkeypatch):
    ds = object.__new__(InferenceDataset)

    ds.return_metadata = False

    ds.model_indexes = {
        "year": np.array([2000]),
        "lead_time": np.array([1]),
    }

    ds.config = SimpleNamespace(
        _using_model_data_as_condition=True,
        effective_condition=None,
        time_features=None,
    )

    monkeypatch.setattr(
        InferenceDataset,
        "_index_condition_dataset",
        lambda self, ind: _fake_da(5),
    )

    monkeypatch.setattr(
        InferenceDataset,
        "_index_model_dataset",
        lambda self, ind: _fake_da(1),
    )

    out = ds[0]

    assert isinstance(out["input"], torch.Tensor)


def test_getitem_concat_condition(monkeypatch):
    ds = object.__new__(InferenceDataset)

    ds.return_metadata = False

    ds.model_indexes = {
        "year": np.array([2000]),
        "lead_time": np.array([1]),
    }

    ds.config = SimpleNamespace(
        _using_model_data_as_condition=False,
        effective_condition=object(),
        time_features=None,
    )

    monkeypatch.setattr(
        InferenceDataset,
        "_index_condition_dataset",
        lambda self, ind: _fake_da(2),
    )

    monkeypatch.setattr(
        InferenceDataset,
        "_index_model_dataset",
        lambda self, ind: _fake_da(1),
    )

    monkeypatch.setattr(
        "cccma_ppp.inference.dataset._get_time_features",
        lambda *args, **kwargs: None,
    )

    out = ds[0]

    assert isinstance(out["input"], torch.Tensor)


def test_getitem_return_metadata(monkeypatch):
    ds = object.__new__(InferenceDataset)

    ds.return_metadata = True

    ds.model_indexes = {
        "year": np.array([2000]),
        "lead_time": np.array([1]),
    }

    ds.config = SimpleNamespace(
        _using_model_data_as_condition=False,
        effective_condition=None,
        time_features=None,
    )

    monkeypatch.setattr(
        InferenceDataset,
        "_index_condition_dataset",
        lambda self, ind: None,
    )

    monkeypatch.setattr(
        InferenceDataset,
        "_index_model_dataset",
        lambda self, ind: _fake_da(),
    )

    monkeypatch.setattr(
        "cccma_ppp.inference.dataset._get_time_features",
        lambda *args, **kwargs: None,
    )

    data, meta = ds[0]

    assert meta["year"] == 2000.0
    assert meta["lead_time"] == 1.0


def test_getitem_added_features(monkeypatch):
    ds = object.__new__(InferenceDataset)

    ds.return_metadata = False

    ds.model_indexes = {
        "year": np.array([2000]),
        "lead_time": np.array([1]),
    }

    ds.config = SimpleNamespace(
        _using_model_data_as_condition=False,
        effective_condition=None,
        time_features=["month"],
    )

    monkeypatch.setattr(
        InferenceDataset,
        "_index_condition_dataset",
        lambda self, ind: None,
    )

    monkeypatch.setattr(
        InferenceDataset,
        "_index_model_dataset",
        lambda self, ind: _fake_da(),
    )

    monkeypatch.setattr(
        "cccma_ppp.inference.dataset._get_time_features",
        lambda *args, **kwargs: np.array([1.0]),
    )

    out = ds[0]

    assert out["added_features"] is not None


def test_get_input_shape_flatten_branch(monkeypatch):
    ds = object.__new__(InferenceDataset)

    class DummyFlatten:
        final_locations = np.arange(10)

    class DummyPipeline:
        fitted_preprocessors = [DummyFlatten()]

        def get_preprocessors(self, name):
            return DummyFlatten()

    monkeypatch.setattr(
        "cccma_ppp.preprocessing.utils_preprocessing.Flattennanremove",
        DummyFlatten,
    )

    ds.config = SimpleNamespace(
        _using_model_data_as_condition=False,
        effective_condition=None,
    )

    monkeypatch.setattr(
        InferenceDataset,
        "effective_input",
        property(
            lambda self: SimpleNamespace(
                names=["tas"],
                preprocessing_pipeline=DummyPipeline(),
            )
        ),
    )

    shape = ds.get_input_shape()

    assert shape == (10,)


def test_get_input_shape_concat_condition(monkeypatch):
    ds = object.__new__(InferenceDataset)

    class DummyPipeline:
        fitted_preprocessors = []

    ds.config = SimpleNamespace(
        _using_model_data_as_condition=False,
        effective_condition=SimpleNamespace(names=["cond1", "cond2"]),
    )

    monkeypatch.setattr(
        InferenceDataset,
        "effective_input",
        property(
            lambda self: SimpleNamespace(
                names=["tas"],
                info=SimpleNamespace(
                    coords={
                        "lat": np.arange(4),
                        "lon": np.arange(5),
                    }
                ),
                preprocessing_pipeline=DummyPipeline(),
            )
        ),
    )

    assert ds.get_input_shape() == (4, 5)


def test_get_cond_indexes_cross_ensemble_branch():
    ds = object.__new__(InferenceDataset)

    ds.condition_dataset = object()

    ds.config = SimpleNamespace(
        condition_method="cross_ensemble",
        effective_condition=SimpleNamespace(
            info=SimpleNamespace(coords={"ensembles": np.array([1, 2, 3])})
        ),
    )

    model_indexes = {
        "year": np.array([2000]),
        "lead_time": np.array([1]),
    }

    result = ds.get_cond_indexes(model_indexes)

    assert "ensembles" in result


def test_index_model_dataset(monkeypatch):
    ds = object.__new__(InferenceDataset)

    class DummyPipeline:
        def transform(self, x):
            return x

    ds.model_indexes = {
        "year": np.array([2000]),
        "lead_time": np.array([1]),
        "ensembles": np.array([0]),
    }

    ds.model_dataset = xr.DataArray(
        np.ones((1, 1, 1, 2, 2)),
        dims=(
            "ensembles",
            "year",
            "lead_time",
            "lat",
            "lon",
        ),
        coords={
            "ensembles": [0],
            "year": [2000],
            "lead_time": [1],
        },
    )

    ds.config = SimpleNamespace(
        _using_model_data_as_condition=False,
        model=SimpleNamespace(preprocessing_pipeline=DummyPipeline()),
    )

    monkeypatch.setattr(
        "cccma_ppp.inference.dataset._unwrap_data_variables",
        lambda x: x,
    )

    out = ds._index_model_dataset(0)

    assert out is not None


def test_post_init_requested_year_subset_check():
    cfg = SimpleNamespace(
        _fitted_preprocessors=True,
        available_train_time=np.array([2000, 2001, 2002]),
    )

    with pytest.raises(ValueError):
        InferenceDataset(
            config=cfg,
            requested_years=np.array([1999]),
        )


def test_from_train_invalid():
    train_cfg = SimpleNamespace(
        input_dataset=None,
        condition_dataset=None,
        observation_dataset=None,
    )

    with pytest.raises(Exception):
        InferenceDatasetConfig._from_train(train_cfg)


def test_getitem_static_condition(monkeypatch):
    ds = object.__new__(InferenceDataset)

    ds.return_metadata = False

    ds.model_indexes = {
        "year": np.array([2000]),
        "lead_time": np.array([1]),
    }

    ds.config = SimpleNamespace(
        _using_model_data_as_condition=False,
        effective_condition=object(),
        condition_method="static",
        time_features=None,
    )

    monkeypatch.setattr(
        InferenceDataset,
        "_index_model_dataset",
        lambda self, ind: _fake_da(),
    )

    monkeypatch.setattr(
        InferenceDataset,
        "_index_condition_dataset",
        lambda self, ind: _fake_da(2),
    )

    monkeypatch.setattr(
        "cccma_ppp.inference.dataset._get_time_features",
        lambda *a, **k: None,
    )

    sample = ds[0]

    assert "input" in sample


def test_getitem_without_condition(monkeypatch):
    ds = object.__new__(InferenceDataset)

    ds.return_metadata = False

    ds.model_indexes = {
        "year": np.array([2000]),
        "lead_time": np.array([1]),
    }

    ds.config = SimpleNamespace(
        _using_model_data_as_condition=False,
        effective_condition=None,
        time_features=None,
    )

    monkeypatch.setattr(
        InferenceDataset,
        "_index_model_dataset",
        lambda self, ind: _fake_da(),
    )

    monkeypatch.setattr(
        InferenceDataset,
        "_index_condition_dataset",
        lambda self, ind: None,
    )

    monkeypatch.setattr(
        "cccma_ppp.inference.dataset._get_time_features",
        lambda *a, **k: None,
    )

    sample = ds[0]

    assert "input" in sample


def test_using_model_data_as_condition_same_dataset():
    shared = SimpleNamespace(
        paths=["path"],
        names=["tas"],
        ensemble_list=[1],
    )

    cfg = object.__new__(InferenceDatasetConfig)

    cfg.model = shared
    cfg.condition = shared

    assert cfg._using_model_data_as_condition is True


def test_check_model_same_member_ensemble_mean_raises():
    cfg = object.__new__(InferenceDatasetConfig)

    cfg.model = SimpleNamespace(ensemble_mean=True)

    cfg.condition_method = "same_member"

    with pytest.raises(ValueError):
        cfg._check_model()


def test_get_added_features_dim_empty():
    ds = object.__new__(InferenceDataset)

    ds.config = DummyDatasetConfig(time_features=[])

    assert ds.get_added_features_dim() == 0


def test_len_year_only():
    ds = object.__new__(InferenceDataset)

    ds.model_indexes = {
        "year": np.arange(8),
    }

    assert len(ds) == 8


def test_get_cond_indexes_same_member_multiple():
    ds = object.__new__(InferenceDataset)

    ds.condition_dataset = object()

    ds.config = SimpleNamespace(condition_method="same_member")

    result = ds.get_cond_indexes(
        {
            "year": np.array([2000, 2001]),
            "lead_time": np.array([1, 2]),
            "ensembles": np.array([3, 4]),
        }
    )

    assert np.array_equal(
        result["ensembles"],
        np.array([3, 4]),
    )


def test_index_condition_dataset_dynamic_multiple(monkeypatch):
    ds = object.__new__(InferenceDataset)

    class DummyPipeline:
        def transform(self, x):
            return x

    ds.condition_dataset = xr.DataArray(
        np.ones((2, 2, 2, 2, 2)),
        dims=(
            "ensembles",
            "year",
            "lead_time",
            "lat",
            "lon",
        ),
        coords={
            "ensembles": [0, 1],
            "year": [2000, 2001],
            "lead_time": [1, 2],
        },
    )

    ds.cond_indexes = {
        "year": np.array([2000, 2001]),
        "lead_time": np.array([1, 2]),
        "ensembles": np.array([0, 1]),
    }

    ds.config = SimpleNamespace(
        condition_method="same_member",
        effective_condition=SimpleNamespace(preprocessing_pipeline=DummyPipeline()),
    )

    monkeypatch.setattr(
        "cccma_ppp.inference.dataset._unwrap_data_variables",
        lambda x: x,
    )

    result = ds._index_condition_dataset(0)

    assert result is not None


def test_index_model_dataset_no_ensemble(monkeypatch):
    ds = object.__new__(InferenceDataset)

    class DummyPipeline:
        def transform(self, x):
            return x

    ds.model_indexes = {
        "year": np.array([2000]),
        "lead_time": np.array([1]),
    }

    ds.model_dataset = xr.DataArray(
        np.ones((1, 1, 2, 2)),
        dims=(
            "year",
            "lead_time",
            "lat",
            "lon",
        ),
        coords={
            "year": [2000],
            "lead_time": [1],
        },
    )

    ds.config = SimpleNamespace(
        _using_model_data_as_condition=False,
        model=SimpleNamespace(preprocessing_pipeline=DummyPipeline()),
    )

    monkeypatch.setattr(
        "cccma_ppp.inference.dataset._unwrap_data_variables",
        lambda x: x,
    )

    result = ds._index_model_dataset(0)

    assert result is not None


def test_getitem_metadata_and_features(monkeypatch):
    ds = object.__new__(InferenceDataset)

    ds.return_metadata = True

    ds.model_indexes = {
        "year": np.array([2000]),
        "lead_time": np.array([1]),
    }

    ds.config = SimpleNamespace(
        _using_model_data_as_condition=False,
        effective_condition=None,
        time_features=["year"],
    )

    monkeypatch.setattr(
        InferenceDataset,
        "_index_model_dataset",
        lambda self, ind: _fake_da(),
    )

    monkeypatch.setattr(
        InferenceDataset,
        "_index_condition_dataset",
        lambda self, ind: None,
    )

    monkeypatch.setattr(
        "cccma_ppp.inference.dataset._get_time_features",
        lambda *a, **k: np.array([1.0]),
    )

    sample, meta = ds[0]

    assert "added_features" in sample
    assert meta is not None


def test_load_model_false_branch(monkeypatch):
    ds = object.__new__(InferenceDataset)

    ds.config = SimpleNamespace(_using_model_data_as_condition=True)

    assert ds._load_model is False


def test_load_model_true_branch(monkeypatch):
    ds = object.__new__(InferenceDataset)

    ds.config = SimpleNamespace(_using_model_data_as_condition=False)

    assert ds._load_model is True


def test_using_model_data_as_condition_match():
    obj = SimpleNamespace(
        paths=["x"],
        names=["tas"],
        ensemble_list=[1],
    )

    cfg = object.__new__(InferenceDatasetConfig)

    cfg.model = obj
    cfg.condition = obj

    assert cfg._using_model_data_as_condition


def test_len_alternate_key():
    ds = object.__new__(InferenceDataset)

    ds.model_indexes = {
        "foo": np.arange(4),
    }

    assert len(ds) == 4


def test_get_added_features_dim_three():
    ds = object.__new__(InferenceDataset)

    ds.config = DummyDatasetConfig(["year", "month", "lead"])

    assert ds.get_added_features_dim() == 3


def test_from_train_failure():
    train_cfg = SimpleNamespace(
        condition_method=None,
        time_features=None,
        lead_months=np.array([1]),
        observation=None,
        effective_condition=None,
        _using_model_data_as_condition=False,
        model=None,
        condition=None,
    )

    with pytest.raises(ValueError):
        _from_train(train_cfg)


def test_index_model_dataset_load_model_false():
    ds = object.__new__(InferenceDataset)

    ds.config = SimpleNamespace(_using_model_data_as_condition=True)

    assert ds._index_model_dataset(0) is None


def test_prepare_mask_with_ensemble(monkeypatch):
    ds = object.__new__(InferenceDataset)

    ds.requested_years = np.array([2000])

    ds.config = SimpleNamespace(lead_months=np.array([1]))

    effective_input = SimpleNamespace(
        year_range=np.array([2000]),
        ensemble_mean=False,
        info=SimpleNamespace(
            sizes={"lead_time": 1},
            coords={
                "ensembles": np.array([0, 1]),
            },
        ),
    )

    monkeypatch.setattr(
        InferenceDataset,
        "effective_input",
        property(lambda self: effective_input),
    )

    base_mask = xr.DataArray(
        np.zeros((1, 1), dtype=bool),
        dims=("year", "lead_time"),
        coords={
            "year": [2000],
            "lead_time": [1],
        },
    )

    monkeypatch.setattr(
        "cccma_ppp.inference.dataset._create_train_mask",
        lambda **kwargs: base_mask,
    )

    mask = ds._prepare_mask()

    assert "ensembles" in mask.dims


def test_get_model_indexes_real():
    ds = object.__new__(InferenceDataset)

    ds.mask = xr.DataArray(
        np.array([[[np.nan, 1.0]]]),
        dims=("ensembles", "year", "lead_time"),
        coords={
            "ensembles": [0],
            "year": [2000],
            "lead_time": [1, 2],
        },
    )

    result = ds.get_model_indexes()

    assert "ensembles" in result
    assert "year" in result
    assert "lead_time" in result


def test_check_model_present():
    cfg = object.__new__(InferenceDatasetConfig)

    cfg.model = object()

    cfg._check_model()


def test_using_model_data_as_condition_with_matching_objects():
    obj = SimpleNamespace(
        paths=["a"],
        names=["tas"],
        ensemble_list=[0],
    )

    cfg = object.__new__(InferenceDatasetConfig)

    cfg.model = obj
    cfg.condition = obj

    assert cfg._using_model_data_as_condition is True


def test_using_model_data_as_condition_non_matching():
    cfg = object.__new__(InferenceDatasetConfig)

    cfg.model = SimpleNamespace(
        paths=["a"],
        names=["tas"],
        ensemble_list=[0],
    )

    cfg.condition = SimpleNamespace(
        paths=["b"],
        names=["pr"],
        ensemble_list=[1],
    )

    assert cfg._using_model_data_as_condition is False


def test_get_cond_indexes_condition_none():
    ds = object.__new__(InferenceDataset)

    ds.condition_dataset = None

    result = ds.get_cond_indexes(
        {
            "year": np.array([2000]),
            "lead_time": np.array([1]),
        }
    )

    assert result is None


def test_load_model_property_true():
    ds = object.__new__(InferenceDataset)

    ds.config = SimpleNamespace(
        _using_model_data_as_condition=False,
    )

    assert ds._load_model is True


def test_get_added_features_dim_zero():
    ds = object.__new__(InferenceDataset)

    ds.config = SimpleNamespace(
        time_features=None,
    )

    assert ds.get_added_features_dim() == 0


def test_get_added_features_dim_many():
    ds = object.__new__(InferenceDataset)

    ds.config = SimpleNamespace(
        time_features=["year", "month", "lead_time"],
    )

    assert ds.get_added_features_dim() == 3


def test_prepare_mask_ensemble_mean(monkeypatch):
    ds = object.__new__(InferenceDataset)

    ds.requested_years = np.array([2000])

    ds.config = SimpleNamespace(
        lead_months=np.array([1]),
    )

    effective_input = SimpleNamespace(
        ensemble_mean=True,
        year_range=np.array([2000]),
        info=SimpleNamespace(
            sizes={"lead_time": 1},
            coords={
                "ensembles": np.array([0, 1]),
            },
        ),
    )

    monkeypatch.setattr(
        InferenceDataset,
        "effective_input",
        property(lambda self: effective_input),
    )

    mask = xr.DataArray(
        np.zeros((1, 1), dtype=bool),
        dims=("year", "lead_time"),
        coords={
            "year": [2000],
            "lead_time": [1],
        },
    )

    monkeypatch.setattr(
        "cccma_ppp.inference.dataset._create_train_mask",
        lambda **kwargs: mask,
    )

    result = ds._prepare_mask()

    assert result is not None


def test_index_condition_dataset_dynamic_no_ensemble(monkeypatch):
    ds = object.__new__(InferenceDataset)

    class DummyPipeline:
        def transform(self, x):
            return x

    ds.condition_dataset = xr.DataArray(
        np.ones((1, 1, 2, 2)),
        dims=("year", "lead_time", "lat", "lon"),
        coords={
            "year": [2000],
            "lead_time": [1],
        },
    )

    ds.cond_indexes = {
        "year": np.array([2000]),
        "lead_time": np.array([1]),
    }

    ds.config = SimpleNamespace(
        condition_method="same_member",
        effective_condition=SimpleNamespace(preprocessing_pipeline=DummyPipeline()),
    )

    monkeypatch.setattr(
        "cccma_ppp.inference.dataset._unwrap_data_variables",
        lambda x: x,
    )

    assert ds._index_condition_dataset(0) is not None


def test_get_cond_indexes_empty_ensemble_coord():
    ds = object.__new__(InferenceDataset)

    ds.condition_dataset = object()

    ds.config = SimpleNamespace(
        condition_method="cross_ensemble",
        effective_condition=SimpleNamespace(
            info=SimpleNamespace(
                coords={
                    "ensembles": np.array([5]),
                }
            )
        ),
    )

    result = ds.get_cond_indexes(
        {
            "year": np.array([2000]),
            "lead_time": np.array([1]),
        }
    )

    assert result["ensembles"][0] == 5


def test_from_train_all_paths(monkeypatch):
    created = {}

    def fake_init(**kwargs):
        created.update(kwargs)
        return kwargs

    monkeypatch.setattr(
        "cccma_ppp.inference.dataset.InferenceDatasetConfig",
        fake_init,
    )

    train_cfg = SimpleNamespace(
        model="MODEL",
        condition="COND",
        condition_method="static",
        time_features=["year"],
        lead_months="LEADS",
        observation=object(),
        effective_condition=object(),
        _using_model_data_as_condition=False,
    )

    result = _from_train(train_cfg)

    assert result["condition_method"] == "static"


def test_index_model_dataset_with_load(monkeypatch):
    ds = object.__new__(InferenceDataset)

    class DummyPipeline:
        def transform(self, x):
            return x

    ds.model_dataset = xr.DataArray(
        np.ones((1, 1, 2, 2)),
        dims=("year", "lead_time", "lat", "lon"),
        coords={
            "year": [2000],
            "lead_time": [1],
        },
    )

    ds.model_indexes = {
        "year": np.array([2000]),
        "lead_time": np.array([1]),
    }

    ds.config = SimpleNamespace(
        _using_model_data_as_condition=False,
        model=SimpleNamespace(preprocessing_pipeline=DummyPipeline()),
    )

    monkeypatch.setattr(
        "cccma_ppp.inference.dataset._unwrap_data_variables",
        lambda x: x,
    )

    assert ds._index_model_dataset(0) is not None


def test_from_train_branch_model_only(monkeypatch):
    captured = {}

    def fake_cfg(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(
        "cccma_ppp.inference.dataset.InferenceDatasetConfig",
        fake_cfg,
    )

    cfg = SimpleNamespace(
        model="MODEL",
        condition=None,
        condition_method=None,
        time_features=None,
        lead_months=None,
        observation=object(),
        effective_condition=None,
        _using_model_data_as_condition=False,
    )

    result = _from_train(cfg)

    assert result["model"] == "MODEL"


def test_from_train_branch_condition(monkeypatch):
    captured = {}

    def fake_cfg(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(
        "cccma_ppp.inference.dataset.InferenceDatasetConfig",
        fake_cfg,
    )

    cfg = SimpleNamespace(
        model="MODEL",
        condition="COND",
        condition_method="static",
        time_features=None,
        lead_months=None,
        observation=object(),
        effective_condition=object(),
        _using_model_data_as_condition=False,
    )

    result = _from_train(cfg)

    assert result["condition"] == "COND"


def test_from_train_branch_model_as_condition(monkeypatch):
    captured = {}

    def fake_cfg(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(
        "cccma_ppp.inference.dataset.InferenceDatasetConfig",
        fake_cfg,
    )

    cfg = SimpleNamespace(
        model="MODEL",
        condition=None,
        condition_method="same_member",
        time_features=None,
        lead_months=None,
        observation=None,
        effective_condition=object(),
        _using_model_data_as_condition=True,
    )

    result = _from_train(cfg)

    assert result["condition_method"] == "same_member"


def test_get_cond_indexes_cross_ensemble_single_member():
    ds = object.__new__(InferenceDataset)

    ds.condition_dataset = object()

    ds.config = SimpleNamespace(
        condition_method="cross_ensemble",
        effective_condition=SimpleNamespace(
            info=SimpleNamespace(
                coords={
                    "ensembles": np.array([99]),
                }
            )
        ),
    )

    result = ds.get_cond_indexes(
        {
            "year": np.array([2000]),
            "lead_time": np.array([1]),
        }
    )

    assert result["ensembles"][0] == 99


def test_check_model_same_member_with_ensemble_mean():
    cfg = object.__new__(InferenceDatasetConfig)

    cfg.model = SimpleNamespace(
        ensemble_mean=True,
    )
    cfg.condition_method = "same_member"

    with pytest.raises(
        ValueError,
        match="same member coniditioning",
    ):
        cfg._check_model()


def test_check_condition_requires_method():
    cfg = object.__new__(InferenceDatasetConfig)

    cfg._effective_condition = object()
    cfg.condition_method = None

    with pytest.raises(
        ValueError,
        match="specify condition_method",
    ):
        cfg._check_condition()


def test_check_condition_ensemble_mean_requires_mean():
    cfg = object.__new__(InferenceDatasetConfig)

    cfg.condition_method = "ensemble_mean"

    cfg._effective_condition = SimpleNamespace(
        ensemble_mean=False,
    )

    with pytest.raises(
        ValueError,
        match="Ensemble mean must be True",
    ):
        cfg._check_condition()


def test_check_condition_static_cannot_use_model_data(monkeypatch):
    monkeypatch.setattr(
        InferenceDatasetConfig,
        "_using_model_data_as_condition",
        property(lambda self: True),
    )

    cfg = object.__new__(InferenceDatasetConfig)

    cfg.condition_method = "static"

    cfg._effective_condition = SimpleNamespace(
        ensemble_list=None,
    )

    with pytest.raises(ValueError):
        cfg._check_condition()


def test_check_condition_static_requires_dataset():
    cfg = object.__new__(InferenceDatasetConfig)

    cfg._effective_condition = None
    cfg.condition_method = "static"

    with pytest.raises(
        ValueError,
        match="condition dataset must be specified",
    ):
        cfg._check_condition()


def test_num_input_lead_months_condition_only():
    cfg = object.__new__(InferenceDatasetConfig)

    cfg.model = None
    cfg.condition = SimpleNamespace(info=SimpleNamespace(sizes={"lead_time": 17}))

    assert cfg.num_input_lead_months == 17


def test_available_inference_time():
    cfg = object.__new__(InferenceDatasetConfig)

    cfg.lead_months = [24]

    cfg.condition = SimpleNamespace(year_range=np.arange(2000, 2006))

    cfg.model = None

    result = cfg.available_inference_time

    np.testing.assert_array_equal(result, np.arange(2000, 2005))


def test_get_input_shape_spatial(monkeypatch):
    monkeypatch.setattr(
        InferenceDataset,
        "_concat_condition_to_input",
        property(lambda self: False),
    )

    monkeypatch.setattr(
        InferenceDataset,
        "effective_input",
        property(
            lambda self: SimpleNamespace(
                names=["tas"],
                info=SimpleNamespace(
                    coords={
                        "lat": np.arange(10),
                        "lon": np.arange(20),
                    }
                ),
                preprocessing_pipeline=SimpleNamespace(fitted_preprocessors=[]),
            )
        ),
    )

    ds = object.__new__(InferenceDataset)

    assert ds.get_input_shape() == (10, 20)


def test_from_train_invalid_config():
    cfg = SimpleNamespace(
        condition_method=None,
        time_features=None,
        lead_months=None,
        observation=None,
        effective_condition=None,
        _using_model_data_as_condition=False,
    )

    with pytest.raises(
        ValueError,
        match="Could not infer",
    ):
        _from_train(cfg)


def test_check_model_same_member_ensemble_mean():
    cfg = object.__new__(InferenceDatasetConfig)

    cfg.model = SimpleNamespace(ensemble_mean=True)

    cfg.condition_method = "same_member"

    with pytest.raises(ValueError):
        cfg._check_model()


def test_check_condition_cross_ensemble_missing_ensembles():
    cfg = object.__new__(InferenceDatasetConfig)

    cfg.condition_method = "cross_ensemble"

    cfg._effective_condition = SimpleNamespace(
        ensemble_mean=False, info=SimpleNamespace(coords={"ensembles": None})
    )

    with pytest.raises(ValueError):
        cfg._check_condition()


def test_check_condition_ensemble_mean_false():
    cfg = object.__new__(InferenceDatasetConfig)

    cfg.condition_method = "ensemble_mean"

    cfg._effective_condition = SimpleNamespace(ensemble_mean=False)

    with pytest.raises(ValueError):
        cfg._check_condition()


def test_check_condition_requires_condition_method():
    cfg = object.__new__(InferenceDatasetConfig)

    cfg._effective_condition = SimpleNamespace()
    cfg.condition_method = None

    with pytest.raises(ValueError):
        cfg._check_condition()


def test_check_condition_cross_ensemble_requires_no_ensemble_mean():
    cfg = object.__new__(InferenceDatasetConfig)

    cfg.condition_method = "cross_ensemble"

    cfg._effective_condition = SimpleNamespace(
        ensemble_mean=True,
        info=SimpleNamespace(coords={"ensembles": np.array([1])}),
    )

    with pytest.raises(ValueError):
        cfg._check_condition()


def test_check_condition_cross_ensemble_requires_ensemble_coord():
    cfg = object.__new__(InferenceDatasetConfig)

    cfg.condition_method = "cross_ensemble"

    cfg._effective_condition = SimpleNamespace(
        ensemble_mean=False,
        info=SimpleNamespace(coords={"ensembles": None}),
    )

    with pytest.raises(ValueError):
        cfg._check_condition()


def test_check_condition_ensemble_mean_requires_true():
    cfg = object.__new__(InferenceDatasetConfig)

    cfg.condition_method = "ensemble_mean"

    cfg._effective_condition = SimpleNamespace(
        ensemble_mean=False,
    )

    with pytest.raises(ValueError):
        cfg._check_condition()


def test_check_condition_static_rejects_model_as_condition(monkeypatch):
    monkeypatch.setattr(
        InferenceDatasetConfig,
        "_using_model_data_as_condition",
        property(lambda self: True),
    )

    cfg = object.__new__(InferenceDatasetConfig)

    cfg.condition_method = "static"

    cfg._effective_condition = SimpleNamespace(
        ensemble_list=None,
    )

    with pytest.raises(ValueError):
        cfg._check_condition()


def test_check_condition_static_without_condition():
    cfg = object.__new__(InferenceDatasetConfig)

    cfg._effective_condition = None
    cfg.condition_method = "static"

    with pytest.raises(ValueError):
        cfg._check_condition()


def test_available_inference_time_with_condition_only():
    cfg = object.__new__(InferenceDatasetConfig)

    cfg.model = None

    cfg.condition = SimpleNamespace(year_range=np.arange(2000, 2006))

    cfg.lead_months = [24]

    result = cfg.available_inference_time

    np.testing.assert_array_equal(
        result,
        np.arange(2000, 2005),
    )


def test_post_init_success_with_condition(monkeypatch):

    monkeypatch.setattr(
        InferenceDataset,
        "_load_xarray_data",
        lambda self, cfg: "DATA",
    )

    monkeypatch.setattr(
        InferenceDataset,
        "_prepare_mask",
        lambda self: xr.DataArray(
            np.zeros((1, 1), dtype=bool),
            dims=("year", "lead_time"),
            coords={
                "year": [2000],
                "lead_time": [1],
            },
        ),
    )

    monkeypatch.setattr(
        InferenceDataset,
        "get_model_indexes",
        lambda self: {
            "year": np.array([2000]),
            "lead_time": np.array([1]),
        },
    )

    monkeypatch.setattr(
        InferenceDataset,
        "get_cond_indexes",
        lambda self, x: {},
    )

    cfg = SimpleNamespace(
        _fitted_preprocessors=True,
        available_train_time=np.array([2000]),
        model=None,
        effective_condition=object(),
    )

    ds = InferenceDataset(
        config=cfg,
        requested_years=np.array([2000]),
    )

    assert ds.condition_dataset == "DATA"


def test_get_input_shape_without_flattener(monkeypatch):

    monkeypatch.setattr(
        InferenceDataset,
        "_concat_condition_to_input",
        property(lambda self: False),
    )

    monkeypatch.setattr(
        InferenceDataset,
        "effective_input",
        property(
            lambda self: SimpleNamespace(
                names=["tas"],
                info=SimpleNamespace(
                    coords={
                        "lat": np.arange(10),
                        "lon": np.arange(20),
                    }
                ),
                preprocessing_pipeline=SimpleNamespace(
                    fitted_preprocessors=[],
                ),
            )
        ),
    )

    ds = object.__new__(InferenceDataset)

    assert ds.get_input_shape() == (10, 20)


def test_from_train_failure_branch():

    cfg = SimpleNamespace(
        condition_method=None,
        time_features=None,
        lead_months=None,
        observation=None,
        effective_condition=None,
        _using_model_data_as_condition=False,
    )

    with pytest.raises(ValueError):
        _from_train(cfg)
