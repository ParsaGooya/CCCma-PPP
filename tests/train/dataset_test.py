import numpy as np
import pytest
import torch
import xarray as xr

from cccma_ppp.train.dataset import TrainDatasetConfig, TrainDataset


@pytest.fixture(autouse=True)
def patch_dataset_config_props(monkeypatch):
    monkeypatch.setattr(
        TrainDatasetConfig,
        "effective_condition",
        property(lambda self: getattr(self, "_effective_condition", None)),
    )


class DummyPipeline:
    fitted_preprocessors = []

    def transform(self, x):
        return x

    def get_preprocessors(self, name):
        class P:
            final_locations = np.arange(4)

        return P()


class DummyInfo:
    def __init__(self):
        self.sizes = {"lead_time": 3}
        self.coords = {
            "lat": xr.DataArray([0, 1], dims="lat"),
            "lon": xr.DataArray([0, 1], dims="lon"),
            "ensembles": xr.DataArray([0, 1], dims="ensembles"),
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


@pytest.fixture
def fake_ds():
    return xr.DataArray(
        np.zeros((2, 3, 3)),
        dims=("ensembles", "year", "lead_time"),
        coords={
            "ensembles": [0, 1],
            "year": [2000, 2001, 2002],
            "lead_time": [1, 2, 3],
        },
    )


@pytest.fixture(autouse=True)
def patch_helpers(monkeypatch, fake_ds):
    monkeypatch.setattr(
        "cccma_ppp.train.dataset._load_xarray_data",
        lambda *a, **k: fake_ds,
    )

    monkeypatch.setattr(
        "cccma_ppp.train.dataset._unwrap_data_variables",
        lambda x: x,
    )

    monkeypatch.setattr(
        "cccma_ppp.train.dataset._get_time_features",
        lambda *a, **k: None,
    )

    monkeypatch.setattr(
        "cccma_ppp.train.dataset._create_train_mask",
        lambda years, lead_times: xr.DataArray(
            np.zeros((len(years), len(lead_times)), dtype=bool),
            dims=("year", "lead_time"),
            coords={"year": years, "lead_time": lead_times},
        ),
    )


def _make_cfg(
    model=None,
    obs=None,
    cond=None,
    method="static",
    using_model_condition=False,
):
    if model is None:
        model = DummyConfig()

    cfg = TrainDatasetConfig.__new__(TrainDatasetConfig)

    cfg.model = model
    cfg.observation = obs
    cfg.condition = cond
    cfg.condition_method = method
    cfg.time_features = None
    cfg.lead_months = np.array([1, 2])

    cfg._fitted_preprocessors = True
    cfg._effective_condition = cond

    if using_model_condition:
        type(cfg)._using_model_data_as_condition = property(lambda self: True)
    else:
        type(cfg)._using_model_data_as_condition = property(lambda self: False)

    type(cfg).effective_condition = property(
        lambda self: getattr(self, "_effective_condition", None)
    )

    return cfg


def test_check_model_same_member_raises():
    cfg = _make_cfg(model=DummyConfig(ensemble_mean=True), method="same_member")

    with pytest.raises(ValueError):
        TrainDatasetConfig._check_model(cfg)


def test_check_model_success():
    cfg = _make_cfg()

    assert TrainDatasetConfig._check_model(cfg) is cfg


def test_check_observation_missing_target_raises():
    cfg = _make_cfg(obs=None, cond=None, method=None)

    with pytest.raises(ValueError):
        TrainDatasetConfig._check_observation(cfg)


def test_check_observation_warns_different_coords():
    model = DummyConfig()
    obs = DummyConfig()

    obs.info.coords["lat"] = xr.DataArray([99], dims="lat")

    cfg = _make_cfg(model=model, obs=obs)

    with pytest.warns(UserWarning):
        TrainDatasetConfig._check_observation(cfg)


def test_check_observation_warns_extra_dim(monkeypatch):
    from cccma_ppp.train import dataset as mod

    model = DummyConfig()
    obs = DummyConfig()

    obs.info.coords["depth"] = xr.DataArray([1], dims="depth")

    monkeypatch.setattr(
        mod,
        "supported_NN_dimensions_sorted",
        ["depth", "lat", "lon"],
    )

    cfg = _make_cfg(model=model, obs=obs)

    with pytest.warns(UserWarning):
        TrainDatasetConfig._check_observation(cfg)


def test_check_condition_requires_method():
    cond = DummyConfig()

    cfg = _make_cfg(cond=cond, method=None)

    with pytest.raises(ValueError):
        TrainDatasetConfig._check_condition(cfg)


def test_check_condition_cross_ensemble_rejects_mean():
    cond = DummyConfig(ensemble_mean=True)

    cfg = _make_cfg(cond=cond, method="cross_ensemble")

    with pytest.raises(ValueError):
        TrainDatasetConfig._check_condition(cfg)


def test_check_condition_cross_ensemble_requires_ensembles():
    cond = DummyConfig()
    cond.info.coords["ensembles"] = None

    cfg = _make_cfg(cond=cond, method="cross_ensemble")

    with pytest.raises(ValueError):
        TrainDatasetConfig._check_condition(cfg)


def test_check_condition_ensemble_mean_requires_mean():
    cond = DummyConfig()

    cfg = _make_cfg(cond=cond, method="ensemble_mean")

    with pytest.raises(ValueError):
        TrainDatasetConfig._check_condition(cfg)


def test_check_condition_static_rejects_ensemble_list():
    cond = DummyConfig()
    cond.ensemble_list = [1]

    cfg = _make_cfg(cond=cond, method="static")

    with pytest.raises(ValueError):
        TrainDatasetConfig._check_condition(cfg)


def test_check_condition_static_rejects_model_condition():
    cond = DummyConfig()

    cfg = _make_cfg(
        cond=cond,
        using_model_condition=True,
    )

    with pytest.raises(ValueError):
        TrainDatasetConfig._check_condition(cfg)


def test_check_condition_static_without_condition_raises():
    cfg = _make_cfg(cond=None, method="static")

    with pytest.raises(ValueError):
        TrainDatasetConfig._check_condition(cfg)


def test_num_input_lead_months():
    cfg = _make_cfg()

    assert TrainDatasetConfig.num_input_lead_months.fget(cfg) == 3


def test_common_time_with_obs():
    model = DummyConfig()
    obs = DummyConfig()

    model.year_range = np.array([2000, 2001])
    obs.year_range = np.array([2001, 2002])

    cfg = _make_cfg(model=model, obs=obs)

    assert np.array_equal(cfg.get_common_time, np.array([2001]))


def test_common_time_no_obs():
    model = DummyConfig()

    cfg = _make_cfg(model=model)

    assert np.array_equal(cfg.get_common_time, model.year_range)


def test_available_train_time_no_obs():
    model = DummyConfig()
    model.year_range = np.array([2000, 2001, 2002])

    cfg = _make_cfg(model=model)
    cfg.lead_months = np.array([24])

    assert np.array_equal(
        cfg.available_train_time,
        np.array([2000, 2001]),
    )


def test_available_train_time_with_obs():
    model = DummyConfig()
    obs = DummyConfig()

    model.year_range = np.array([2000, 2001])
    obs.year_range = np.array([2001])

    cfg = _make_cfg(model=model, obs=obs)

    assert np.array_equal(cfg.available_train_time, np.array([2001]))


def test_dataset_requires_fitted_preprocessors():
    cfg = _make_cfg()
    cfg._fitted_preprocessors = False

    with pytest.raises(RuntimeError):
        TrainDataset(cfg, [2000])


def test_dataset_rejects_bad_year():
    cfg = _make_cfg()

    with pytest.raises(ValueError):
        TrainDataset(cfg, [9999])


def test_dataset_basic_construction():
    cfg = _make_cfg()
    ds = TrainDataset(cfg, [2000])

    assert ds.model_dataset is not None


def test_autoencoding_property():
    cfg = _make_cfg()
    ds = TrainDataset(cfg, [2000])

    assert ds._autoencoding_model_data is True


def test_write_condition_to_input_true_for_autoencoding():
    cfg = _make_cfg()
    ds = TrainDataset(cfg, [2000])

    assert ds._write_condition_to_input is True


def test_concat_condition_to_input():
    obs = DummyConfig()
    cond = DummyConfig()

    cfg = _make_cfg(obs=obs, cond=cond)
    ds = TrainDataset(cfg, [2000])

    assert ds._concat_condition_to_input is True


def test_get_obs_indexes_none():
    cfg = _make_cfg()
    ds = TrainDataset(cfg, [2000])

    assert ds.get_obs_indexes(ds.model_indexes) is None


def test_get_obs_indexes():
    obs = DummyConfig()

    cfg = _make_cfg(obs=obs)
    ds = TrainDataset(cfg, [2000])

    indexes = ds.get_obs_indexes(ds.model_indexes)

    assert "year" in indexes
    assert "month" in indexes


def test_get_cond_indexes_static():
    cond = DummyConfig()

    cfg = _make_cfg(cond=cond)
    ds = TrainDataset(cfg, [2000])

    assert ds.get_cond_indexes(ds.model_indexes) is None


def test_get_cond_indexes_cross_ensemble():
    cond = DummyConfig()

    cfg = _make_cfg(cond=cond, method="cross_ensemble")
    ds = TrainDataset(cfg, [2000])

    indexes = ds.get_cond_indexes(ds.model_indexes)

    assert "ensembles" in indexes


def test_get_cond_indexes_same_member():
    cond = DummyConfig()

    cfg = _make_cfg(cond=cond, method="same_member")
    ds = TrainDataset(cfg, [2000])

    indexes = ds.get_cond_indexes(ds.model_indexes)

    assert "ensembles" in indexes


def test_index_model_dataset():
    cfg = _make_cfg()
    ds = TrainDataset(cfg, [2000])

    assert ds._index_model_dataset(0) is not None


def test_index_condition_dataset_static():
    cond = DummyConfig()

    cfg = _make_cfg(cond=cond)
    ds = TrainDataset(cfg, [2000])

    assert ds._index_condition_dataset(0) is not None


def test_get_added_features_dim_none():
    cfg = _make_cfg()
    ds = TrainDataset(cfg, [2000])

    assert ds.get_added_features_dim() == 0


def test_get_added_features_dim_nonzero():
    cfg = _make_cfg()
    cfg.time_features = ["a", "b"]

    ds = TrainDataset(cfg, [2000])

    assert ds.get_added_features_dim() == 2


def test_len():
    cfg = _make_cfg()
    ds = TrainDataset(cfg, [2000])

    assert len(ds) > 0


def test_check_observation_returns_self_when_no_obs_and_method_set():
    cfg = _make_cfg(method="static")
    assert TrainDatasetConfig._check_observation(cfg) is cfg


def test_check_condition_returns_self_cross_ensemble_valid(monkeypatch):
    cond = DummyConfig()

    monkeypatch.setattr(
        TrainDatasetConfig,
        "effective_condition",
        property(lambda self: cond),
    )

    cfg = _make_cfg(cond=cond, method="cross_ensemble")

    assert TrainDatasetConfig._check_condition(cfg) is cfg


def test_check_condition_returns_self_same_member_valid(monkeypatch):
    cond = DummyConfig()

    monkeypatch.setattr(
        TrainDatasetConfig,
        "effective_condition",
        property(lambda self: cond),
    )

    cfg = _make_cfg(cond=cond, method="same_member")

    assert TrainDatasetConfig._check_condition(cfg) is cfg


def test_check_condition_returns_self_ensemble_mean_valid(monkeypatch):
    cond = DummyConfig(ensemble_mean=True)

    monkeypatch.setattr(
        TrainDatasetConfig,
        "effective_condition",
        property(lambda self: cond),
    )

    cfg = _make_cfg(cond=cond, method="ensemble_mean")

    assert TrainDatasetConfig._check_condition(cfg) is cfg


def test_load_model_true_when_autoencoding():
    cfg = _make_cfg()
    ds = TrainDataset(cfg, [2000])

    assert ds._load_model is True


def test_load_model_true_when_not_using_model_condition(monkeypatch):
    obs = DummyConfig()

    cfg = _make_cfg(obs=obs)

    monkeypatch.setattr(
        TrainDatasetConfig,
        "_using_model_data_as_condition",
        property(lambda self: False),
    )

    ds = TrainDataset(cfg, [2000])

    assert ds._load_model is True


def test_write_condition_to_input_false(monkeypatch):
    obs = DummyConfig()

    cfg = _make_cfg(obs=obs)

    monkeypatch.setattr(
        TrainDatasetConfig,
        "_using_model_data_as_condition",
        property(lambda self: False),
    )

    ds = TrainDataset(cfg, [2000])

    assert ds._write_condition_to_input is False


def test_concat_condition_to_input_false_without_condition(monkeypatch):
    obs = DummyConfig()

    cfg = _make_cfg(obs=obs)

    monkeypatch.setattr(
        TrainDatasetConfig,
        "_using_model_data_as_condition",
        property(lambda self: False),
    )

    ds = TrainDataset(cfg, [2000])

    assert ds._concat_condition_to_input is False


def test_get_cond_indexes_returns_none_without_condition_dataset():
    cfg = _make_cfg()
    ds = TrainDataset(cfg, [2000])

    ds.condition_dataset = None

    assert ds.get_cond_indexes(ds.model_indexes) is None


def test_index_condition_dataset_returns_none_without_condition():
    cfg = _make_cfg()
    ds = TrainDataset(cfg, [2000])

    ds.condition_dataset = None

    assert ds._index_condition_dataset(0) is None


def test_index_observation_dataset_returns_none_without_obs():
    cfg = _make_cfg()
    ds = TrainDataset(cfg, [2000])

    ds.observation_dataset = None

    assert ds._index_observation_dataset(0) is None


def test_index_model_dataset_returns_none_when_load_model_false(monkeypatch):
    cfg = _make_cfg()
    ds = TrainDataset(cfg, [2000])

    monkeypatch.setattr(
        TrainDataset,
        "_load_model",
        property(lambda self: False),
    )

    assert ds._index_model_dataset(0) is None


def test_getitem_returns_metadata_tuple(monkeypatch):
    cfg = _make_cfg()
    ds = TrainDataset(cfg, [2000])

    monkeypatch.setattr(
        ds,
        "_index_condition_dataset",
        lambda ind: xr.DataArray(np.array([1.0])),
    )

    monkeypatch.setattr(
        ds,
        "_index_model_dataset",
        lambda ind: xr.DataArray(np.array([1.0])),
    )

    ds.return_metadata = True

    result = ds[0]

    assert isinstance(result, tuple)
    assert len(result) == 2


def test_getitem_returns_dict_without_metadata(monkeypatch):
    cfg = _make_cfg()
    ds = TrainDataset(cfg, [2000])

    monkeypatch.setattr(
        ds,
        "_index_condition_dataset",
        lambda ind: xr.DataArray(np.array([1.0])),
    )

    monkeypatch.setattr(
        ds,
        "_index_model_dataset",
        lambda ind: xr.DataArray(np.array([1.0])),
    )

    ds.return_metadata = False

    result = ds[0]

    assert isinstance(result, dict)


def test_available_train_time_zero_lead_year_shift():
    cfg = _make_cfg()
    cfg.model.year_range = np.array([2000, 2001, 2002])
    cfg.lead_months = np.array([1])

    assert np.array_equal(
        cfg.available_train_time,
        np.array([2000, 2001, 2002, 2003]),
    )


def test_get_common_time_empty_intersection():
    model = DummyConfig()
    obs = DummyConfig()

    model.year_range = np.array([2000])
    obs.year_range = np.array([2001])

    cfg = _make_cfg(model=model, obs=obs)

    assert cfg.get_common_time.size == 0


def test_len_uses_first_index_key():
    cfg = _make_cfg()
    ds = TrainDataset(cfg, [2000])

    ds.model_indexes = {
        "year": np.array([1, 2, 3]),
        "lead_time": np.array([1]),
    }

    assert len(ds) == 3


def test_num_input_lead_months_different_size():
    model = DummyConfig()
    model.info.sizes["lead_time"] = 24

    cfg = _make_cfg(model=model)

    assert cfg.num_input_lead_months == 24


def test_get_obs_indexes_without_ensemble_dimension():
    obs = DummyConfig()
    obs.info.coords["ensembles"] = None

    cfg = _make_cfg(obs=obs)

    ds = TrainDataset(cfg, [2000])

    indexes = ds.get_obs_indexes(ds.model_indexes)

    assert "ensembles" not in indexes


def test_get_cond_indexes_contains_year_and_lead_time():
    cond = DummyConfig()

    cfg = _make_cfg(cond=cond, method="cross_ensemble")

    ds = TrainDataset(cfg, [2000])

    indexes = ds.get_cond_indexes(ds.model_indexes)

    assert "year" in indexes
    assert "lead_time" in indexes


def test_prepare_mask_preserves_requested_year_subset():
    cfg = _make_cfg()

    ds = TrainDataset(cfg, [2000])

    assert np.all(ds.mask.year.values == np.array([2000]))


def test_build_dataset_returns_train_dataset(monkeypatch):
    cfg = _make_cfg()

    monkeypatch.setattr(
        TrainDataset,
        "__post_init__",
        lambda self: None,
    )

    ds = cfg.build_dataset([2000])

    assert isinstance(ds, TrainDataset)


def test_fit_preprocessors_delegates(monkeypatch):
    cfg = _make_cfg()

    called = {}

    class DummyOperator:
        def _fit_preprocessors(self, **kwargs):
            called.update(kwargs)

    monkeypatch.setattr(
        TrainDatasetConfig,
        "ds_operator",
        property(lambda self: DummyOperator()),
    )

    cfg._fit_preprocessors(
        train_years=[2000],
        save=True,
        save_path="a",
        save_name="b",
    )

    assert called["train_years"] == [2000]
    assert called["save"] is True
    assert called["save_path"] == "a"
    assert called["save_name"] == "b"


def test_load_fitted_preprocessors_delegates(monkeypatch):
    cfg = _make_cfg()

    called = {}

    class DummyOperator:
        def _load_fitted_preprocessors(self, load_dir):
            called["load_dir"] = load_dir

    monkeypatch.setattr(
        TrainDatasetConfig,
        "ds_operator",
        property(lambda self: DummyOperator()),
    )

    cfg._load_fitted_preprocessors("abc")

    assert called["load_dir"] == "abc"


def test_add_fitted_preprocessor_delegates(monkeypatch):
    cfg = _make_cfg()

    called = {}

    class DummyOperator:
        def _add_fitted_preprocessor(self, preprocessor, index):
            called["preprocessor"] = preprocessor
            called["index"] = index

    monkeypatch.setattr(
        TrainDatasetConfig,
        "ds_operator",
        property(lambda self: DummyOperator()),
    )

    obj = object()

    cfg._add_fitted_preprocessor(obj, index=4)

    assert called["preprocessor"] is obj
    assert called["index"] == 4


def test_prepare_mask_no_ensemble_dimension_when_missing():
    cfg = _make_cfg()
    cfg.model.info.coords["ensembles"] = None

    ds = TrainDataset(cfg, [2000])

    assert "ensembles" not in ds.mask.dims


def test_prepare_mask_no_ensemble_dimension_when_ensemble_mean():
    cfg = _make_cfg()
    cfg.model.ensemble_mean = True

    ds = TrainDataset(cfg, [2000])

    assert "ensembles" not in ds.mask.dims


def test_get_model_indexes_contains_expected_keys():
    cfg = _make_cfg()

    ds = TrainDataset(cfg, [2000])

    assert "year" in ds.model_indexes
    assert "lead_time" in ds.model_indexes


def test_get_model_indexes_contains_ensemble_key():
    cfg = _make_cfg()

    ds = TrainDataset(cfg, [2000])

    assert "ensembles" in ds.model_indexes


def test_get_obs_indexes_adds_ensemble_indices():
    obs = DummyConfig()

    cfg = _make_cfg(obs=obs)

    ds = TrainDataset(cfg, [2000])

    indexes = ds.get_obs_indexes(ds.model_indexes)

    assert "ensembles" in indexes


def test_get_cond_indexes_cross_ensemble_length_matches():
    cond = DummyConfig()

    cfg = _make_cfg(cond=cond, method="cross_ensemble")

    ds = TrainDataset(cfg, [2000])

    indexes = ds.get_cond_indexes(ds.model_indexes)

    assert len(indexes["ensembles"]) == len(indexes["year"])


def test_get_cond_indexes_same_member_matches_model():
    cond = DummyConfig()

    cfg = _make_cfg(cond=cond, method="same_member")

    ds = TrainDataset(cfg, [2000])

    indexes = ds.get_cond_indexes(ds.model_indexes)

    assert np.array_equal(
        indexes["ensembles"],
        ds.model_indexes["ensembles"],
    )


def test_index_condition_dataset_cross_ensemble_branch():
    cond = DummyConfig()

    cfg = _make_cfg(cond=cond, method="cross_ensemble")

    ds = TrainDataset(cfg, [2000])

    assert ds._index_condition_dataset(0) is not None


def test_index_condition_dataset_same_member_branch():
    cond = DummyConfig()

    cfg = _make_cfg(cond=cond, method="same_member")

    ds = TrainDataset(cfg, [2000])

    assert ds._index_condition_dataset(0) is not None


def test_index_model_dataset_with_ensemble_selection():
    cfg = _make_cfg()

    ds = TrainDataset(cfg, [2000])

    result = ds._index_model_dataset(0)

    assert result is not None


def test_getitem_autoencoding_path(monkeypatch):
    cfg = _make_cfg()
    ds = TrainDataset(cfg, [2000])

    arr = xr.DataArray(np.array([1.0]))

    monkeypatch.setattr(ds, "_index_model_dataset", lambda ind: arr)
    monkeypatch.setattr(ds, "_index_condition_dataset", lambda ind: arr)

    monkeypatch.setattr(
        TrainDataset,
        "_write_condition_to_input",
        property(lambda self: False),
    )

    result = ds[0]

    assert torch.equal(result["input"], result["target"])


def test_getitem_write_condition_path(monkeypatch):
    cfg = _make_cfg(cond=DummyConfig())

    ds = TrainDataset(cfg, [2000])

    arr1 = xr.DataArray(np.array([1.0]))
    arr2 = xr.DataArray(np.array([2.0]))

    monkeypatch.setattr(ds, "_index_model_dataset", lambda ind: arr1)
    monkeypatch.setattr(ds, "_index_condition_dataset", lambda ind: arr2)

    result = ds[0]

    assert result["input"].item() == 2.0


def test_getitem_added_features_not_none(monkeypatch):
    cfg = _make_cfg()
    ds = TrainDataset(cfg, [2000])

    arr = xr.DataArray(np.array([1.0]))

    monkeypatch.setattr(ds, "_index_model_dataset", lambda ind: arr)
    monkeypatch.setattr(ds, "_index_condition_dataset", lambda ind: arr)

    monkeypatch.setattr(
        TrainDataset,
        "_write_condition_to_input",
        property(lambda self: False),
    )

    monkeypatch.setattr(
        "cccma_ppp.train.dataset._get_time_features",
        lambda *args, **kwargs: np.array([1.0, 2.0]),
    )

    result = ds[0]

    assert result["added_features"] is not None


def test_getitem_added_features_none(monkeypatch):
    cfg = _make_cfg()
    ds = TrainDataset(cfg, [2000])

    arr = xr.DataArray(np.array([1.0]))

    monkeypatch.setattr(ds, "_index_model_dataset", lambda ind: arr)
    monkeypatch.setattr(ds, "_index_condition_dataset", lambda ind: arr)

    monkeypatch.setattr(
        TrainDataset,
        "_write_condition_to_input",
        property(lambda self: False),
    )

    monkeypatch.setattr(
        "cccma_ppp.train.dataset._get_time_features",
        lambda *args, **kwargs: None,
    )

    result = ds[0]

    assert result["added_features"] is None


def test_getitem_metadata_contains_year(monkeypatch):
    cfg = _make_cfg()

    ds = TrainDataset(cfg, [2000], return_metadata=True)

    arr = xr.DataArray(np.array([1.0]))

    monkeypatch.setattr(ds, "_index_model_dataset", lambda ind: arr)
    monkeypatch.setattr(ds, "_index_condition_dataset", lambda ind: arr)

    monkeypatch.setattr(
        TrainDataset,
        "_write_condition_to_input",
        property(lambda self: False),
    )

    _, metadata = ds[0]

    assert "year" in metadata


def test_getitem_metadata_contains_lead_time(monkeypatch):
    cfg = _make_cfg()

    ds = TrainDataset(cfg, [2000], return_metadata=True)

    arr = xr.DataArray(np.array([1.0]))

    monkeypatch.setattr(ds, "_index_model_dataset", lambda ind: arr)
    monkeypatch.setattr(ds, "_index_condition_dataset", lambda ind: arr)

    monkeypatch.setattr(
        TrainDataset,
        "_write_condition_to_input",
        property(lambda self: False),
    )

    _, metadata = ds[0]

    assert "lead_time" in metadata


def test_get_target_shape_observation_branch(monkeypatch):
    obs = DummyConfig()

    cfg = _make_cfg(obs=obs)

    ds = TrainDataset(cfg, [2000])

    monkeypatch.setattr(
        obs.preprocessing_pipeline,
        "fitted_preprocessors",
        [],
    )

    result = ds.get_target_shape()

    assert isinstance(result, tuple)


def test_get_input_shape_non_flatten_branch():
    cfg = _make_cfg()

    ds = TrainDataset(cfg, [2000])

    result = ds.get_input_shape()

    assert isinstance(result, tuple)


def test_build_dataset_metadata_true(monkeypatch):
    cfg = _make_cfg()

    monkeypatch.setattr(
        TrainDataset,
        "__post_init__",
        lambda self: None,
    )

    ds = cfg.build_dataset(
        years=[2000],
        return_metadata=True,
    )

    assert ds.return_metadata is True


def test_build_dataset_mask_passthrough(monkeypatch):
    cfg = _make_cfg()

    monkeypatch.setattr(
        TrainDataset,
        "__post_init__",
        lambda self: None,
    )

    mask = object()

    ds = cfg.build_dataset(
        years=[2000],
        mask=mask,
    )

    assert ds.mask is mask


def test_check_condition_no_condition_non_static_passes():
    cfg = _make_cfg(cond=None, method="same_member")

    assert TrainDatasetConfig._check_condition(cfg) is cfg


def test_check_model_non_same_member_passes():
    cfg = _make_cfg(method="cross_ensemble")

    assert TrainDatasetConfig._check_model(cfg) is cfg


def test_check_observation_matching_coords_no_warning():
    model = DummyConfig()
    obs = DummyConfig()

    cfg = _make_cfg(model=model, obs=obs)

    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        TrainDatasetConfig._check_observation(cfg)

    assert len(w) == 0


def test_load_model_true_when_not_using_model_condition_branch(monkeypatch):
    obs = DummyConfig()

    cfg = _make_cfg(obs=obs)

    monkeypatch.setattr(
        TrainDatasetConfig,
        "_using_model_data_as_condition",
        property(lambda self: False),
    )

    ds = TrainDataset(cfg, [2000])

    assert ds._load_model


def test_concat_condition_requires_condition(monkeypatch):
    obs = DummyConfig()

    cfg = _make_cfg(obs=obs)

    monkeypatch.setattr(
        TrainDatasetConfig,
        "_using_model_data_as_condition",
        property(lambda self: False),
    )

    ds = TrainDataset(cfg, [2000])

    assert ds._concat_condition_to_input is False


def test_get_added_features_dim_single_feature():
    cfg = _make_cfg()
    cfg.time_features = ["year"]

    ds = TrainDataset(cfg, [2000])

    assert ds.get_added_features_dim() == 1


def test_get_added_features_dim_three_features():
    cfg = _make_cfg()
    cfg.time_features = ["a", "b", "c"]

    ds = TrainDataset(cfg, [2000])

    assert ds.get_added_features_dim() == 3


def test_len_equals_index_length_after_override():
    cfg = _make_cfg()

    ds = TrainDataset(cfg, [2000])

    ds.model_indexes = {
        "year": np.arange(10),
        "lead_time": np.arange(10),
    }

    assert len(ds) == 10


def test_get_common_time_identical_ranges():
    model = DummyConfig()
    obs = DummyConfig()

    cfg = _make_cfg(model=model, obs=obs)

    assert np.array_equal(
        cfg.get_common_time,
        model.year_range,
    )


def test_available_train_time_observation_branch_returns_common_time():
    model = DummyConfig()
    obs = DummyConfig()

    model.year_range = np.array([2000, 2001, 2002])
    obs.year_range = np.array([2001, 2002])

    cfg = _make_cfg(model=model, obs=obs)

    assert np.array_equal(
        cfg.available_train_time,
        np.array([2001, 2002]),
    )


def test_getitem_concat_condition_branch(monkeypatch):
    obs = DummyConfig()
    cond = DummyConfig()

    cfg = _make_cfg(obs=obs, cond=cond)

    monkeypatch.setattr(
        TrainDataset,
        "_autoencoding_model_data",
        property(lambda self: False),
    )

    monkeypatch.setattr(
        TrainDataset,
        "_write_condition_to_input",
        property(lambda self: False),
    )

    monkeypatch.setattr(
        TrainDataset,
        "_concat_condition_to_input",
        property(lambda self: True),
    )

    ds = TrainDataset(cfg, [2000])

    arr1 = xr.DataArray(np.array([1.0]), dims=("channels",))
    arr2 = xr.DataArray(np.array([2.0]), dims=("channels",))

    monkeypatch.setattr(ds, "_index_model_dataset", lambda i: arr1)
    monkeypatch.setattr(ds, "_index_condition_dataset", lambda i: arr2)
    monkeypatch.setattr(ds, "_index_observation_dataset", lambda i: arr1)

    result = ds[0]

    assert result["input"].shape[0] == 2


def test_getitem_write_condition_branch(monkeypatch):
    cond = DummyConfig()

    cfg = _make_cfg(cond=cond)

    monkeypatch.setattr(
        TrainDataset,
        "_autoencoding_model_data",
        property(lambda self: False),
    )

    monkeypatch.setattr(
        TrainDataset,
        "_write_condition_to_input",
        property(lambda self: True),
    )

    ds = TrainDataset(cfg, [2000])

    cond_arr = xr.DataArray(np.array([5.0]))

    monkeypatch.setattr(
        ds, "_index_model_dataset", lambda i: xr.DataArray(np.array([1.0]))
    )
    monkeypatch.setattr(ds, "_index_condition_dataset", lambda i: cond_arr)
    monkeypatch.setattr(
        ds, "_index_observation_dataset", lambda i: xr.DataArray(np.array([2.0]))
    )

    result = ds[0]

    assert result["input"].item() == 5.0


def test_getitem_return_metadata_true(monkeypatch):
    cfg = _make_cfg()

    ds = TrainDataset(
        cfg,
        [2000],
        return_metadata=True,
    )

    arr = xr.DataArray(np.array([1.0]))

    monkeypatch.setattr(ds, "_index_model_dataset", lambda i: arr)
    monkeypatch.setattr(ds, "_index_condition_dataset", lambda i: arr)

    monkeypatch.setattr(
        TrainDataset,
        "_write_condition_to_input",
        property(lambda self: False),
    )

    result, metadata = ds[0]

    assert isinstance(metadata, dict)


def test_index_condition_dataset_cross_ensemble_selection(monkeypatch):
    cond = DummyConfig()

    cfg = _make_cfg(
        cond=cond,
        method="cross_ensemble",
    )

    ds = TrainDataset(cfg, [2000])

    assert ds.cond_indexes is not None

    result = ds._index_condition_dataset(0)

    assert result is not None


def test_index_condition_dataset_same_member_selection(monkeypatch):
    cond = DummyConfig()

    cfg = _make_cfg(
        cond=cond,
        method="same_member",
    )

    ds = TrainDataset(cfg, [2000])

    result = ds._index_condition_dataset(0)

    assert result is not None


def test_index_observation_dataset_with_observation(monkeypatch):
    obs = DummyConfig()

    cfg = _make_cfg(obs=obs)

    ds = TrainDataset(cfg, [2000])

    result = ds._index_observation_dataset(0)

    assert result is not None


def test_get_target_shape_without_observation():
    cfg = _make_cfg()

    ds = TrainDataset(cfg, [2000])

    assert ds.get_target_shape() == ds.get_input_shape()


def test_index_model_dataset_without_ensemble_key():
    cfg = _make_cfg()

    ds = TrainDataset(cfg, [2000])

    ds.model_indexes.pop("ensembles", None)

    assert ds._index_model_dataset(0) is not None


def test_get_obs_indexes_with_ensemble_mean():
    obs = DummyConfig()
    obs.ensemble_mean = True

    cfg = _make_cfg(obs=obs)

    ds = TrainDataset(cfg, [2000])

    indexes = ds.get_obs_indexes(ds.model_indexes)

    if indexes is not None:
        assert "year" in indexes


def test_prepare_mask_with_existing_mask():
    cfg = _make_cfg()

    existing_mask = xr.DataArray(
        np.ones((1, 2), dtype=bool),
        dims=("year", "lead_time"),
        coords={
            "year": [2000],
            "lead_time": [1, 2],
        },
    )

    ds = TrainDataset(
        cfg,
        [2000],
        mask=existing_mask,
    )

    assert ds.mask is existing_mask


def test_get_cond_indexes_cross_ensemble_no_ensembles():
    cond = DummyConfig()
    cond.info.coords["ensembles"] = None

    cfg = _make_cfg(cond=cond, method="cross_ensemble")

    ds = TrainDataset(cfg, [2000])

    indexes = ds.get_cond_indexes(ds.model_indexes)

    assert indexes is not None


def test_get_obs_indexes_ensemble_mean_true():
    obs = DummyConfig()
    obs.ensemble_mean = True

    cfg = _make_cfg(obs=obs)

    ds = TrainDataset(cfg, [2000])

    indexes = ds.get_obs_indexes(ds.model_indexes)

    assert indexes is not None


def test_get_obs_indexes_no_ensemble_coord():
    obs = DummyConfig()
    obs.info.coords["ensembles"] = None

    cfg = _make_cfg(obs=obs)

    ds = TrainDataset(cfg, [2000])

    indexes = ds.get_obs_indexes(ds.model_indexes)

    assert indexes is not None


def test_index_model_dataset_missing_ensemble_indexes():
    cfg = _make_cfg()

    ds = TrainDataset(cfg, [2000])

    ds.model_indexes.pop("ensembles", None)

    result = ds._index_model_dataset(0)

    assert result is not None


def test_index_observation_dataset_missing_ensemble_indexes():
    obs = DummyConfig()

    cfg = _make_cfg(obs=obs)

    ds = TrainDataset(cfg, [2000])

    if ds.obs_indexes is not None:
        ds.obs_indexes.pop("ensembles", None)

    result = ds._index_observation_dataset(0)

    assert result is not None


def test_index_condition_dataset_missing_ensemble_indexes():
    cond = DummyConfig()

    cfg = _make_cfg(
        cond=cond,
        method="same_member",
    )

    ds = TrainDataset(cfg, [2000])

    if ds.cond_indexes is not None:
        ds.cond_indexes.pop("ensembles", None)

    result = ds._index_condition_dataset(0)

    assert result is not None


def test_prepare_mask_existing_mask_with_ensemble_coord():
    cfg = _make_cfg()

    mask = xr.DataArray(
        np.zeros((2, 1, 2), dtype=bool),
        dims=("ensembles", "year", "lead_time"),
        coords={
            "ensembles": [0, 1],
            "year": [2000],
            "lead_time": [1, 2],
        },
    )

    ds = TrainDataset(
        cfg,
        [2000],
        mask=mask,
    )

    assert ds.mask is mask


def test_prepare_mask_existing_mask_without_ensemble_coord():
    cfg = _make_cfg()

    mask = xr.DataArray(
        np.zeros((1, 2), dtype=bool),
        dims=("year", "lead_time"),
        coords={
            "year": [2000],
            "lead_time": [1, 2],
        },
    )

    ds = TrainDataset(
        cfg,
        [2000],
        mask=mask,
    )

    assert ds.mask is mask


def test_get_added_features_dim_empty_list():
    cfg = _make_cfg()

    cfg.time_features = []

    ds = TrainDataset(cfg, [2000])

    assert ds.get_added_features_dim() == 0


def test_build_dataset_with_mask_and_metadata(monkeypatch):
    cfg = _make_cfg()

    monkeypatch.setattr(
        TrainDataset,
        "__post_init__",
        lambda self: None,
    )

    mask = object()

    ds = cfg.build_dataset(
        years=[2000],
        mask=mask,
        return_metadata=True,
    )

    assert ds.mask is mask
    assert ds.return_metadata is True


def test_common_time_single_value_overlap():
    model = DummyConfig()
    obs = DummyConfig()

    model.year_range = np.array([1999, 2000])
    obs.year_range = np.array([2000, 2001])

    cfg = _make_cfg(model=model, obs=obs)

    assert np.array_equal(
        cfg.get_common_time,
        np.array([2000]),
    )


def test_available_train_time_empty():
    model = DummyConfig()
    obs = DummyConfig()

    model.year_range = np.array([2000])
    obs.year_range = np.array([2001])

    cfg = _make_cfg(model=model, obs=obs)

    assert cfg.available_train_time.size == 0
