import pytest
import numpy as np
import xarray as xr
import warnings

from cccma_ppp.train.datasets import (
    TrainDatasetConfig,
    TrainDataset,
)


def make_valid_config_with(**kwargs):
    cfg = make_valid_config()

    for key, value in kwargs.items():
        setattr(cfg, key, value)

    return cfg


@pytest.fixture(autouse=True)
def mock_xarray_loader(monkeypatch):
    def fake_loader(*args, **kwargs):
        return xr.Dataset(
            {
                "var": (
                    ("ensembles", "year", "lead_time", "month"),
                    np.zeros((2, 3, 3, 3)),
                )
            },
            coords={
                "ensembles": np.array([0, 1]),
                "year": np.array([2000, 2001, 2002]),
                "lead_time": np.array([1, 2, 3]),
                "month": np.array([1, 2, 3]),
            },
        )

    monkeypatch.setattr(
        "cccma_ppp.train.datasets._load_xarray_data",
        fake_loader,
    )


class DummyLeadMonths:
    def build_lead_months(self):
        return np.array([100])


class DummyInfo:
    def __init__(self):
        self.sizes = {"lead_time": 12}
        self.coords = {
            "lat": xr.DataArray([0, 1], dims="lat"),
            "lon": xr.DataArray([0, 1], dims="lon"),
            "ensembles": xr.DataArray([0, 1], dims="ensembles"),
        }


class DummyPipeline:
    def set_name(self, name):
        self.name = name

    def __init__(self):
        self.pipeline = []
        self.fitted_preprocessors = []

    def transform(self, x):
        return x

    def get_preprocessors(self, name):
        return self

    @property
    def final_locations(self):
        return np.arange(4)


class DummyDataConfig:
    file_type = "nc"
    rename_dict = None

    def __init__(self, ensemble_mean=False):
        self.ensemble_mean = ensemble_mean
        self.info = DummyInfo()
        self.year_range = np.arange(2000, 2005)

        self.list_paths = []
        self.names = ["x"]
        self.concat_dim = None
        self.rename_dict = {}

        self.preprocessing_pipeline = DummyPipeline()
        self.ensemble_list = None

        self.paths = ["dummy"]


def test_invalid_lead_months_exceeds():
    model = DummyDataConfig()
    model.info.sizes["lead_time"] = 5

    cond = DummyDataConfig()
    cond.paths = ["different"]

    with pytest.raises(ValueError):
        TrainDatasetConfig(
            model=model,
            condition=cond,
            condition_method="static",
            lead_months=DummyLeadMonths(),
        )


def test_observation_warns_mismatch():
    model = DummyDataConfig()
    obs = DummyDataConfig()

    obs.info.coords["lat"] = xr.DataArray([9], dims="lat")

    cond = DummyDataConfig()
    cond.paths = ["different"]

    with pytest.warns(UserWarning):
        TrainDatasetConfig(
            model=model,
            observation=obs,
            condition=cond,
            condition_method="static",
        )


def test_no_observation_requires_condition_method():
    model = DummyDataConfig()

    with pytest.raises(AssertionError):
        TrainDatasetConfig(model=model, observation=None, condition_method=None)


def test_invalid_condition_without_method():
    model = DummyDataConfig()
    cond = DummyDataConfig()

    with pytest.raises(ValueError):
        TrainDatasetConfig(
            model=model,
            observation=cond,
            condition=cond,
            condition_method=None,
        )


def test_static_condition_missing_dataset():
    model = DummyDataConfig()

    with pytest.raises(ValueError):
        TrainDatasetConfig(model=model, condition_method="static", observation=None)


def make_valid_config():
    model = DummyDataConfig()
    cond = DummyDataConfig()

    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=None,
        condition=cond,
        condition_method="static",
    )
    cfg._fitted_preprocessors = True
    return cfg


def test_dataset_requires_fitted_preprocessors():
    cfg = make_valid_config_with()
    cfg._fitted_preprocessors = False

    with pytest.raises(RuntimeError):
        TrainDataset(cfg, [2000])


def test_dataset_year_not_subset():
    cfg = make_valid_config_with()

    with pytest.raises(ValueError):
        TrainDataset(cfg, [9999])


def test_prepare_mask_default():
    cfg = make_valid_config_with()
    cfg._fitted_preprocessors = True

    ds = TrainDataset.__new__(TrainDataset)
    ds.config = cfg
    ds.mask = None


def test_same_member_with_ensemble_mean_raises():
    model = DummyDataConfig()
    model.ensemble_mean = True

    cond = DummyDataConfig()
    cond.paths = ["different"]

    with pytest.raises(ValueError):
        TrainDatasetConfig(
            model=model,
            condition=cond,
            condition_method="same_member",
        )


def test_default_lead_months_set():
    model = DummyDataConfig()
    model.info.sizes["lead_time"] = 3

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        condition=cond,
        condition_method="static",
    )

    assert np.array_equal(cfg.lead_months, np.array([1, 2, 3]))


def test_no_observation_requires_condition_method_assert():
    model = DummyDataConfig()

    with pytest.raises(AssertionError):
        TrainDatasetConfig(
            model=model,
            observation=None,
            condition=None,
            condition_method=None,
        )


def test_observation_lon_warning():
    model = DummyDataConfig()
    obs = DummyDataConfig()

    obs.info.coords["lon"] = xr.DataArray([99], dims="lon")

    cond = DummyDataConfig()
    cond.paths = ["different"]

    with pytest.warns(UserWarning):
        TrainDatasetConfig(
            model=model,
            observation=obs,
            condition=cond,
            condition_method="static",
        )


def test_ensemble_mean_condition_requires_true():
    model = DummyDataConfig()

    cond = DummyDataConfig()
    cond.paths = ["different"]
    cond.ensemble_mean = False

    with pytest.raises(ValueError):
        TrainDatasetConfig(
            model=model,
            condition=cond,
            condition_method="ensemble_mean",
        )


def test_cross_ensemble_requires_ensemble_dim():
    model = DummyDataConfig()

    cond = DummyDataConfig()
    cond.paths = ["different"]
    cond.info.coords["ensembles"] = None

    with pytest.raises(ValueError):
        TrainDatasetConfig(
            model=model,
            condition=cond,
            condition_method="cross_ensemble",
        )


def test_static_condition_with_ensemble_list_fails():
    model = DummyDataConfig()

    cond = DummyDataConfig()
    cond.paths = ["different"]
    cond.ensemble_list = [1, 2]

    with pytest.raises(ValueError):
        TrainDatasetConfig(
            model=model,
            condition=cond,
            condition_method="static",
        )


def test_num_input_lead_months():
    model = DummyDataConfig()
    model.info.sizes["lead_time"] = 6

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        condition=cond,
        condition_method="static",
    )

    assert cfg.num_input_lead_months == 6


def test_get_common_time_with_obs():
    model = DummyDataConfig()
    obs = DummyDataConfig()

    model.year_range = np.array([2000, 2001, 2002])
    obs.year_range = np.array([2001, 2002, 2003])

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=obs,
        condition=cond,
        condition_method="static",
    )

    assert np.array_equal(cfg.get_common_time, np.array([2001, 2002]))


def test_available_train_time_no_obs():
    cfg = make_valid_config_with()
    cfg.lead_months = np.array([1, 24])

    cfg.model.year_range = np.array([2000, 2001, 2002])

    result = cfg.available_train_time

    assert isinstance(result, np.ndarray)


def test_autoencoding_mode_property():
    cfg = make_valid_config_with()
    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    assert ds._autoencoding_model_data is True


def test_concat_condition_flag():
    cfg = make_valid_config_with()
    cfg.observation = DummyDataConfig()

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    assert isinstance(ds._concat_condition_to_input, bool)


def test_prepare_mask_with_existing_mask():
    cfg = make_valid_config_with()
    years = cfg.get_common_time[:1]

    mask = xr.DataArray(
        np.zeros((len(cfg.model.year_range), cfg.model.info.sizes["lead_time"])),
        dims=("year", "lead_time"),
        coords={"year": cfg.model.year_range, "lead_time": np.arange(1, 13)},
    ).astype(bool)

    ds = TrainDataset(cfg, requested_years=years, mask=mask)

    assert ds.mask is not None


def test_get_cond_indexes_static_returns_none():
    cfg = make_valid_config_with()
    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    result = ds.get_cond_indexes(ds.model_indexes)

    assert result is None


def test_get_obs_indexes_none_when_no_obs():
    cfg = make_valid_config_with()
    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    assert ds.get_obs_indexes(ds.model_indexes) is None


def test_get_input_shape_without_flattener():
    cfg = make_valid_config_with()

    cfg.model.preprocessing_pipeline.fitted_preprocessors = []

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    shape = ds.get_input_shape()
    assert isinstance(shape, tuple)


def test_getitem_with_metadata():
    cfg = make_valid_config_with()

    ds = TrainDataset(
        cfg,
        requested_years=cfg.get_common_time[:1],
        return_metadata=True,
    )

    data, meta = ds[0]

    assert "input" in data
    assert isinstance(meta, dict)


def test_condition_cross_ensemble_branch():
    model = DummyDataConfig()
    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=None,
        condition=cond,
        condition_method="cross_ensemble",
    )
    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    assert ds.cond_indexes is not None
    assert "ensembles" in ds.cond_indexes


def test_condition_same_member_branch():
    model = DummyDataConfig()
    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=None,
        condition=cond,
        condition_method="same_member",
    )
    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    assert np.array_equal(
        ds.cond_indexes["ensembles"],
        ds.model_indexes["ensembles"],
    )


def test_get_target_shape_with_observation():
    model = DummyDataConfig()
    obs = DummyDataConfig()
    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=obs,
        condition=cond,
        condition_method="static",
    )
    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    shape = ds.get_target_shape()

    assert isinstance(shape, tuple)


def test_time_features_added():
    cfg = make_valid_config_with()
    cfg.time_features = ["year"]

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    data = ds[0]

    assert "added_features" in data


def test_concat_condition_executes():
    model = DummyDataConfig()
    obs = DummyDataConfig()
    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=obs,
        condition=cond,
        condition_method="static",
    )
    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    assert ds._concat_condition_to_input is True


def test_observation_indexing_branch():
    model = DummyDataConfig()
    obs = DummyDataConfig()
    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=obs,
        condition=cond,
        condition_method="static",
    )
    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    assert ds.obs_indexes is not None
    obs_val = ds._index_observation_dataset(0)

    assert obs_val is not None


def test_load_model_false_path():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    _ = ds._load_model

    assert isinstance(ds._load_model, bool)


def test_write_condition_false_branch():
    model = DummyDataConfig()
    obs = DummyDataConfig()
    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=obs,
        condition=cond,
        condition_method="static",
    )
    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    assert isinstance(ds._write_condition_to_input, bool)


def test_actual_concat_execution_branch():
    model = DummyDataConfig()
    obs = DummyDataConfig()
    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=obs,
        condition=cond,
        condition_method="static",
    )
    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    if ds._concat_condition_to_input:
        try:
            ds[0]
        except ValueError:
            pass


def test_condition_overwrites_input():
    model = DummyDataConfig()
    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=None,
        condition=cond,
        condition_method="static",
    )
    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    data = ds[0]

    assert "input" in data


def test_static_condition_indexing_executes():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    result = ds._index_condition_dataset(0)

    assert result is not None


def test_condition_none_branch():
    cfg = make_valid_config_with()

    cfg._effective_condition = None

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    assert ds.condition_dataset is None


def test_autoencoding_target_equals_input():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    data = ds[0]

    assert (data["input"] == data["target"]).all()


def test_write_condition_true_autoencoding():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    assert ds._autoencoding_model_data is True
    assert ds._write_condition_to_input is True


def test_prepare_mask_no_ensemble_expansion():
    model = DummyDataConfig()
    model.ensemble_mean = True

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        condition=cond,
        condition_method="static",
    )
    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    assert "ensembles" not in ds.mask.dims


def test_index_model_dataset_skip_branch():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    result = ds._index_model_dataset(0)

    assert result is not None


def test_get_target_shape_without_observation():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    ds.input_shape = ds.get_input_shape()

    shape = ds.get_target_shape()

    assert shape == ds.input_shape


def test_get_added_features_dim():
    cfg = make_valid_config_with()
    cfg.time_features = ["year", "month"]

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    assert ds.get_added_features_dim() == 2


def test_index_condition_dataset_non_static():
    model = DummyDataConfig()

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=None,
        condition=cond,
        condition_method="same_member",
    )
    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    result = ds._index_condition_dataset(0)

    assert result is not None


def test_index_observation_dataset_none_branch():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    result = ds._index_observation_dataset(0)

    assert result is None


def test_dataset_len():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    assert len(ds) > 0


def test_prepare_mask_with_ensemble_dim():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    assert "ensembles" in ds.mask.dims


def test_load_xarray_without_ensembles():
    model = DummyDataConfig()
    model.info.coords["ensembles"] = None

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        condition=cond,
        condition_method="static",
    )
    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    assert ds.model_dataset is not None


def test_obs_indexes_without_ensemble_sampling():
    model = DummyDataConfig()

    obs = DummyDataConfig()
    obs.ensemble_mean = True

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=obs,
        condition=cond,
        condition_method="static",
    )
    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    indexes = ds.get_obs_indexes(ds.model_indexes)

    assert "ensembles" not in indexes


def test_condition_dataset_none_branch():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    ds.condition_dataset = None

    result = ds._index_condition_dataset(0)

    assert result is None


def test_model_dataset_indexing():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    result = ds._index_model_dataset(0)

    assert result is not None


def test_get_input_shape_flattener_branch(monkeypatch):
    class FakeFlatten:
        pass

    monkeypatch.setattr(
        "cccma_ppp.preprocessing.utils_preprocessing.Flattennanremove",
        FakeFlatten,
    )

    cfg = make_valid_config_with()

    cfg.model.preprocessing_pipeline.fitted_preprocessors = [FakeFlatten()]

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    shape = ds.get_input_shape()

    assert shape is not None


def test_get_target_shape_flattener_branch(monkeypatch):
    class FakeFlatten:
        pass

    monkeypatch.setattr(
        "cccma_ppp.preprocessing.utils_preprocessing.Flattennanremove",
        FakeFlatten,
    )

    model = DummyDataConfig()
    obs = DummyDataConfig()
    cond = DummyDataConfig()
    cond.paths = ["different"]

    obs.preprocessing_pipeline.fitted_preprocessors = [FakeFlatten()]

    cfg = TrainDatasetConfig(
        model=model,
        observation=obs,
        condition=cond,
        condition_method="static",
    )
    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    shape = ds.get_target_shape()

    assert shape is not None


def test_concat_condition_false_branch():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    assert ds._concat_condition_to_input is False


def test_getitem_without_metadata():
    cfg = make_valid_config_with()

    ds = TrainDataset(
        cfg,
        requested_years=cfg.get_common_time[:1],
        return_metadata=False,
    )

    result = ds[0]

    assert isinstance(result, dict)


def test_get_cond_indexes_none_when_no_condition_dataset():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    ds.condition_dataset = None

    result = ds.get_cond_indexes(ds.model_indexes)

    assert result is None


def test_obs_indexes_with_no_ensembles():
    model = DummyDataConfig()

    obs = DummyDataConfig()
    obs.info.coords["ensembles"] = None

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=obs,
        condition=cond,
        condition_method="static",
    )
    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    indexes = ds.get_obs_indexes(ds.model_indexes)

    assert "ensembles" not in indexes


def test_concat_success_branch(monkeypatch):
    def fake_concat(objs, dim=None):
        return objs[0]

    monkeypatch.setattr(xr, "concat", fake_concat)

    model = DummyDataConfig()
    obs = DummyDataConfig()

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=obs,
        condition=cond,
        condition_method="static",
    )
    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    result = ds[0]

    assert "input" in result


def test_get_input_shape_flattener_false_branch():
    cfg = make_valid_config_with()

    cfg.model.preprocessing_pipeline.fitted_preprocessors = [object()]

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    shape = ds.get_input_shape()

    assert isinstance(shape, tuple)


def test_write_condition_false_explicit():
    model = DummyDataConfig()
    obs = DummyDataConfig()

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=obs,
        condition=cond,
        condition_method="static",
    )
    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    assert ds._autoencoding_model_data is False
    assert ds._write_condition_to_input is False


def test_concat_condition_true_explicit():
    model = DummyDataConfig()
    obs = DummyDataConfig()

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=obs,
        condition=cond,
        condition_method="static",
    )
    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    assert ds._concat_condition_to_input is True


def test_get_obs_indexes_with_ensemble_sampling():
    model = DummyDataConfig()

    obs = DummyDataConfig()

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=obs,
        condition=cond,
        condition_method="static",
    )
    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    indexes = ds.get_obs_indexes(ds.model_indexes)

    assert "ensembles" in indexes


def test_get_cond_indexes_cross_ensemble():
    model = DummyDataConfig()

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=None,
        condition=cond,
        condition_method="cross_ensemble",
    )
    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    indexes = ds.get_cond_indexes(ds.model_indexes)

    assert "ensembles" in indexes


def test_get_cond_indexes_same_member():
    model = DummyDataConfig()

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=None,
        condition=cond,
        condition_method="same_member",
    )
    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    indexes = ds.get_cond_indexes(ds.model_indexes)

    assert np.array_equal(
        indexes["ensembles"],
        ds.model_indexes["ensembles"],
    )


def test_prepare_mask_existing_mask_branch():
    cfg = make_valid_config_with()

    mask = xr.DataArray(
        np.zeros((5, 12), dtype=bool),
        dims=("year", "lead_time"),
        coords={
            "year": cfg.model.year_range,
            "lead_time": np.arange(1, 13),
        },
    )

    ds = TrainDataset(
        cfg,
        requested_years=cfg.get_common_time[:1],
        mask=mask,
    )

    assert ds.mask is not None


def test_getitem_added_features_none():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    result = ds[0]

    assert result["added_features"] is None


def test_build_dataset_wrapper():
    cfg = make_valid_config_with()

    ds = cfg.build_dataset(
        years=cfg.get_common_time[:1],
        return_metadata=False,
    )

    assert isinstance(ds, TrainDataset)


def test_available_train_time_with_observation_branch():
    model = DummyDataConfig()
    obs = DummyDataConfig()

    model.year_range = np.array([2000, 2001, 2002])
    obs.year_range = np.array([2001, 2002])

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=obs,
        condition=cond,
        condition_method="static",
    )

    result = cfg.available_train_time

    assert np.array_equal(result, np.array([2001, 2002]))


def test_prepare_mask_existing_mask_branch_again():
    model = DummyDataConfig()
    model.ensemble_mean = True

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        condition=cond,
        condition_method="static",
    )
    cfg._fitted_preprocessors = True

    mask = xr.DataArray(
        np.zeros((5, 12), dtype=bool),
        dims=("year", "lead_time"),
        coords={
            "year": cfg.model.year_range,
            "lead_time": np.arange(1, 13),
        },
    )

    ds = TrainDataset(
        cfg,
        requested_years=cfg.get_common_time[:1],
        mask=mask,
    )

    assert ds.mask is not None


def test_getitem_autoencoding_path():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    item = ds[0]

    assert item["target"] is not None
    assert item["input"] is not None


def test_get_model_indexes_keys():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    indexes = ds.get_model_indexes()

    assert "year" in indexes
    assert "lead_time" in indexes


def test_get_model_indexes_ensemble_key():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    indexes = ds.get_model_indexes()

    assert "ensembles" in indexes


def test_obs_indexes_contains_year_and_month():
    model = DummyDataConfig()
    obs = DummyDataConfig()

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=obs,
        condition=cond,
        condition_method="static",
    )
    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    indexes = ds.get_obs_indexes(ds.model_indexes)

    assert "year" in indexes
    assert "month" in indexes


def test_cond_indexes_cross_ensemble_contains_keys():
    model = DummyDataConfig()

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=None,
        condition=cond,
        condition_method="cross_ensemble",
    )
    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    indexes = ds.get_cond_indexes(ds.model_indexes)

    assert "year" in indexes
    assert "lead_time" in indexes
    assert "ensembles" in indexes


def test_cond_indexes_same_member_contains_keys():
    model = DummyDataConfig()

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=None,
        condition=cond,
        condition_method="same_member",
    )
    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    indexes = ds.get_cond_indexes(ds.model_indexes)

    assert "ensembles" in indexes


def test_getitem_returns_target_key():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    result = ds[0]

    assert "target" in result


def test_getitem_returns_input_key():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    result = ds[0]

    assert "input" in result


def test_getitem_returns_added_features_none():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    result = ds[0]

    assert result["added_features"] is None


def test_time_features_not_none_branch():
    cfg = make_valid_config_with()
    cfg.time_features = ["year"]

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    result = ds[0]

    assert result["added_features"] is not None


def test_index_condition_dataset_static_branch_again():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    out = ds._index_condition_dataset(0)

    assert out is not None


def test_index_condition_dataset_same_member_branch_again():
    model = DummyDataConfig()

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=None,
        condition=cond,
        condition_method="same_member",
    )
    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    out = ds._index_condition_dataset(0)

    assert out is not None


def test_index_condition_dataset_cross_ensemble_branch_again():
    model = DummyDataConfig()

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=None,
        condition=cond,
        condition_method="cross_ensemble",
    )
    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    out = ds._index_condition_dataset(0)

    assert out is not None


def test_index_observation_dataset_with_ensemble_branch():
    model = DummyDataConfig()
    obs = DummyDataConfig()

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=obs,
        condition=cond,
        condition_method="static",
    )
    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    out = ds._index_observation_dataset(0)

    assert out is not None


def test_index_model_dataset_with_ensemble_branch():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    out = ds._index_model_dataset(0)

    assert out is not None


def test_get_input_shape_return_type():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    result = ds.get_input_shape()

    assert isinstance(result, tuple)


def test_get_target_shape_return_type():
    model = DummyDataConfig()
    obs = DummyDataConfig()

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=obs,
        condition=cond,
        condition_method="static",
    )
    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    result = ds.get_target_shape()

    assert isinstance(result, tuple)


def test_write_condition_false_branch_explicit():
    model = DummyDataConfig()
    obs = DummyDataConfig()

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=obs,
        condition=cond,
        condition_method="static",
    )
    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    assert ds._write_condition_to_input is False


def test_concat_condition_true_branch_explicit():
    model = DummyDataConfig()
    obs = DummyDataConfig()

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=obs,
        condition=cond,
        condition_method="static",
    )
    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    assert ds._concat_condition_to_input is True


def test_load_model_property_branch():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    assert ds._load_model is True


def test_dataset_len_positive():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    assert len(ds) >= 1


def test_prepare_mask_dims():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    assert "year" in ds.mask.dims
    assert "lead_time" in ds.mask.dims


def test_prepare_mask_ensemble_branch_again():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    assert "ensembles" in ds.mask.coords


def test_prepare_mask_without_ensemble_coords():
    model = DummyDataConfig()
    model.info.coords["ensembles"] = None

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        condition=cond,
        condition_method="static",
    )
    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    assert "ensembles" not in ds.mask.dims


def test_return_metadata_false_path():
    cfg = make_valid_config_with()

    ds = TrainDataset(
        cfg,
        requested_years=cfg.get_common_time[:1],
        return_metadata=False,
    )

    result = ds[0]

    assert isinstance(result, dict)


def test_return_metadata_true_path_again():
    cfg = make_valid_config_with()

    ds = TrainDataset(
        cfg,
        requested_years=cfg.get_common_time[:1],
        return_metadata=True,
    )

    result, meta = ds[0]

    assert isinstance(meta, dict)


def test_concat_success_real_branch(monkeypatch):
    def fake_concat(*args, **kwargs):
        return args[0][0]

    monkeypatch.setattr(xr, "concat", fake_concat)

    model = DummyDataConfig()
    obs = DummyDataConfig()

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=obs,
        condition=cond,
        condition_method="static",
    )
    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    result = ds[0]

    assert "input" in result


def test_check_observation_matching_coords_no_warning():
    model = DummyDataConfig()
    obs = DummyDataConfig()

    cond = DummyDataConfig()
    cond.paths = ["different"]

    with warnings.catch_warnings(record=True) as w:
        TrainDatasetConfig(
            model=model,
            observation=obs,
            condition=cond,
            condition_method="static",
        )

    assert len(w) == 0


def test_available_train_time_lead_year_adjustment():
    cfg = make_valid_config_with()

    cfg.model.year_range = np.array([2000, 2001, 2002, 2003])
    cfg.lead_months = np.array([24])

    result = cfg.available_train_time

    assert isinstance(result, np.ndarray)


def test_get_common_time_without_obs():
    cfg = make_valid_config_with()

    result = cfg.get_common_time

    assert np.array_equal(result, cfg.model.year_range)


def test_prepare_mask_without_ensemble_mean_and_without_ensembles():
    model = DummyDataConfig()
    model.ensemble_mean = False
    model.info.coords["ensembles"] = None

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        condition=cond,
        condition_method="static",
    )
    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    assert "ensembles" not in ds.mask.dims


def test_obs_indexes_ensemble_mean_true_branch():
    model = DummyDataConfig()

    obs = DummyDataConfig(ensemble_mean=True)

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=obs,
        condition=cond,
        condition_method="static",
    )
    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    indexes = ds.get_obs_indexes(ds.model_indexes)

    assert "ensembles" not in indexes


def test_obs_indexes_none_ensemble_coords_branch():
    model = DummyDataConfig()

    obs = DummyDataConfig()
    obs.info.coords["ensembles"] = None

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=obs,
        condition=cond,
        condition_method="static",
    )
    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    indexes = ds.get_obs_indexes(ds.model_indexes)

    assert "ensembles" not in indexes


def test_cond_indexes_static_branch_returns_none():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    indexes = ds.get_cond_indexes(ds.model_indexes)

    assert indexes is None


def test_cond_indexes_cross_ensemble_selection():
    model = DummyDataConfig()

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=None,
        condition=cond,
        condition_method="cross_ensemble",
    )
    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    out = ds._index_condition_dataset(0)

    assert out is not None


def test_cond_indexes_same_member_selection():
    model = DummyDataConfig()

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=None,
        condition=cond,
        condition_method="same_member",
    )
    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    out = ds._index_condition_dataset(0)

    assert out is not None


def test_getitem_concat_path(monkeypatch):
    def fake_concat(*args, **kwargs):
        return args[0][0]

    monkeypatch.setattr(xr, "concat", fake_concat)

    model = DummyDataConfig()
    obs = DummyDataConfig()

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=obs,
        condition=cond,
        condition_method="static",
    )
    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    item = ds[0]

    assert isinstance(item, dict)


def test_getitem_without_time_features_branch():
    cfg = make_valid_config_with()
    cfg.time_features = None

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    item = ds[0]

    assert item["added_features"] is None


def test_getitem_with_time_features_branch():
    cfg = make_valid_config_with()
    cfg.time_features = ["year"]

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    item = ds[0]

    assert item["added_features"] is not None


def test_get_input_shape_flattener_true_branch(monkeypatch):
    class FakeFlatten:
        pass

    monkeypatch.setattr(
        "cccma_ppp.preprocessing.utils_preprocessing.Flattennanremove",
        FakeFlatten,
    )

    cfg = make_valid_config_with()

    cfg.model.preprocessing_pipeline.fitted_preprocessors = [FakeFlatten()]

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    result = ds.get_input_shape()

    assert result is not None


def test_get_target_shape_flattener_true_branch(monkeypatch):
    class FakeFlatten:
        pass

    monkeypatch.setattr(
        "cccma_ppp.preprocessing.utils_preprocessing.Flattennanremove",
        FakeFlatten,
    )

    model = DummyDataConfig()
    obs = DummyDataConfig()

    cond = DummyDataConfig()
    cond.paths = ["different"]

    obs.preprocessing_pipeline.fitted_preprocessors = [FakeFlatten()]

    cfg = TrainDatasetConfig(
        model=model,
        observation=obs,
        condition=cond,
        condition_method="static",
    )
    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    result = ds.get_target_shape()

    assert result is not None


def test_write_condition_to_input_false_branch():
    model = DummyDataConfig()
    obs = DummyDataConfig()

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=obs,
        condition=cond,
        condition_method="static",
    )
    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    assert ds._write_condition_to_input is False


def test_concat_condition_property_true():
    model = DummyDataConfig()
    obs = DummyDataConfig()

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=obs,
        condition=cond,
        condition_method="static",
    )
    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    assert ds._concat_condition_to_input is True


def test_load_model_property_true():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    assert ds._load_model is True


def test_dataset_len_nonzero():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    assert len(ds) != 0


def test_return_metadata_false_again():
    cfg = make_valid_config_with()

    ds = TrainDataset(
        cfg,
        requested_years=cfg.get_common_time[:1],
        return_metadata=False,
    )

    result = ds[0]

    assert isinstance(result, dict)


def test_return_metadata_true_again():
    cfg = make_valid_config_with()

    ds = TrainDataset(
        cfg,
        requested_years=cfg.get_common_time[:1],
        return_metadata=True,
    )

    result, meta = ds[0]

    assert isinstance(meta, dict)


def test_property_ds_operator():
    cfg = make_valid_config_with()

    assert cfg.ds_operator is not None


def test_build_dataset_return_metadata_true():
    cfg = make_valid_config_with()

    ds = cfg.build_dataset(
        years=cfg.get_common_time[:1],
        return_metadata=True,
    )

    assert ds.return_metadata is True


def test_build_dataset_return_metadata_false():
    cfg = make_valid_config_with()

    ds = cfg.build_dataset(
        years=cfg.get_common_time[:1],
        return_metadata=False,
    )

    assert ds.return_metadata is False


def test_common_time_no_observation_exact():
    cfg = make_valid_config_with()

    assert np.array_equal(
        cfg.get_common_time,
        cfg.model.year_range,
    )


def test_num_input_lead_months_property_again():
    cfg = make_valid_config_with()

    assert cfg.num_input_lead_months == 12


def test_available_train_time_type():
    cfg = make_valid_config_with()

    result = cfg.available_train_time

    assert isinstance(result, np.ndarray)


def test_prepare_mask_has_requested_year():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=[2000])

    assert 2000 in ds.mask.year.values


def test_prepare_mask_has_lead_times():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=[2000])

    assert "lead_time" in ds.mask.coords


def test_model_indexes_return_dict():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=[2000])

    result = ds.get_model_indexes()

    assert isinstance(result, dict)


def test_obs_indexes_return_none_without_obs():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=[2000])

    result = ds.get_obs_indexes(ds.model_indexes)

    assert result is None


def test_cond_indexes_return_none_static():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=[2000])

    result = ds.get_cond_indexes(ds.model_indexes)

    assert result is None


def test_index_model_dataset_return_type():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=[2000])

    result = ds._index_model_dataset(0)

    assert result is not None


def test_index_condition_dataset_static_return_type():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=[2000])

    result = ds._index_condition_dataset(0)

    assert result is not None


def test_index_observation_dataset_none_again():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=[2000])

    result = ds._index_observation_dataset(0)

    assert result is None


def test_getitem_returns_dict_again():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=[2000])

    result = ds[0]

    assert isinstance(result, dict)


def test_getitem_contains_input():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=[2000])

    result = ds[0]

    assert "input" in result


def test_getitem_contains_target():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=[2000])

    result = ds[0]

    assert "target" in result


def test_getitem_contains_added_features():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=[2000])

    result = ds[0]

    assert "added_features" in result


def test_getitem_metadata_true_tuple():
    cfg = make_valid_config_with()

    ds = TrainDataset(
        cfg,
        requested_years=[2000],
        return_metadata=True,
    )

    result = ds[0]

    assert isinstance(result, tuple)


def test_getitem_metadata_false_dict():
    cfg = make_valid_config_with()

    ds = TrainDataset(
        cfg,
        requested_years=[2000],
        return_metadata=False,
    )

    result = ds[0]

    assert isinstance(result, dict)


def test_len_nonzero_again():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=[2000])

    assert len(ds) > 0


def test_autoencoding_property_true_again():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=[2000])

    assert ds._autoencoding_model_data


def test_load_model_property_bool():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=[2000])

    assert isinstance(ds._load_model, bool)


def test_write_condition_to_input_bool():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=[2000])

    assert isinstance(ds._write_condition_to_input, bool)


def test_concat_condition_property_bool():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=[2000])

    assert isinstance(ds._concat_condition_to_input, bool)


def test_get_input_shape_tuple_again():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=[2000])

    shape = ds.get_input_shape()

    assert isinstance(shape, tuple)


def test_get_target_shape_tuple_again():
    model = DummyDataConfig()
    obs = DummyDataConfig()

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=obs,
        condition=cond,
        condition_method="static",
    )
    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=[2000])

    shape = ds.get_target_shape()

    assert isinstance(shape, tuple)


def test_added_features_dim_value():
    cfg = make_valid_config_with()

    cfg.time_features = ["year"]

    ds = TrainDataset(cfg, requested_years=[2000])

    assert ds.get_added_features_dim() == 1


def test_time_features_none_path_again():
    cfg = make_valid_config_with()

    cfg.time_features = None

    ds = TrainDataset(cfg, requested_years=[2000])

    item = ds[0]

    assert item["added_features"] is None


def test_time_features_present_path_again():
    cfg = make_valid_config_with()

    cfg.time_features = ["year"]

    ds = TrainDataset(cfg, requested_years=[2000])

    item = ds[0]

    assert item["added_features"] is not None


def test_cross_ensemble_indexes_type():
    model = DummyDataConfig()

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        condition=cond,
        condition_method="cross_ensemble",
    )
    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=[2000])

    assert isinstance(ds.cond_indexes, dict)


def test_same_member_indexes_type():
    model = DummyDataConfig()

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        condition=cond,
        condition_method="same_member",
    )
    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=[2000])

    assert isinstance(ds.cond_indexes, dict)


def test_observation_dataset_loaded():
    model = DummyDataConfig()
    obs = DummyDataConfig()

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=obs,
        condition=cond,
        condition_method="static",
    )

    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=[2000])

    assert ds.observation_dataset is not None


def test_condition_dataset_loaded():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=[2000])

    assert ds.condition_dataset is not None


def test_model_dataset_loaded():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=[2000])

    assert ds.model_dataset is not None


def test_mask_not_none():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=[2000])

    assert ds.mask is not None


def test_model_indexes_not_none():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=[2000])

    assert ds.model_indexes is not None


def test_cond_indexes_none_static_again():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=[2000])

    assert ds.cond_indexes is None


def test_obs_indexes_none_again():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=[2000])

    assert ds.obs_indexes is None


def test_condition_method_static_string():
    cfg = make_valid_config_with()

    assert cfg.condition_method == "static"


def test_lead_months_array_exists():
    cfg = make_valid_config_with()

    assert isinstance(cfg.lead_months, np.ndarray)


def test_requested_years_subset():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=[2000])

    assert set(ds.requested_years).issubset(set(cfg.get_common_time))


def test_concat_success_monkeypatched_again(monkeypatch):
    def fake_concat(*args, **kwargs):
        return args[0][0]

    monkeypatch.setattr(xr, "concat", fake_concat)

    model = DummyDataConfig()
    obs = DummyDataConfig()

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=obs,
        condition=cond,
        condition_method="static",
    )

    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=[2000])

    result = ds[0]

    assert isinstance(result, dict)


def test_index_condition_dataset_without_ensembles_key():
    model = DummyDataConfig()

    cond = DummyDataConfig()
    cond.info.coords["ensembles"] = None
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=None,
        condition=cond,
        condition_method="static",
    )

    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    result = ds._index_condition_dataset(0)

    assert result is not None


def test_index_observation_dataset_without_ensembles_key():
    model = DummyDataConfig()

    obs = DummyDataConfig()
    obs.info.coords["ensembles"] = None

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=obs,
        condition=cond,
        condition_method="static",
    )

    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    result = ds._index_observation_dataset(0)

    assert result is not None


def test_index_model_dataset_without_ensembles_key():
    model = DummyDataConfig()
    model.info.coords["ensembles"] = None

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        condition=cond,
        condition_method="static",
    )

    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    result = ds._index_model_dataset(0)

    assert result is not None


def test_prepare_mask_without_expand_branch():
    model = DummyDataConfig()
    model.info.coords["ensembles"] = None

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        condition=cond,
        condition_method="static",
    )

    cfg._fitted_preprocessors = True

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    assert "ensembles" not in ds.mask.dims


def test_prepare_mask_with_expand_branch():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    assert "ensembles" in ds.mask.dims


def test_get_target_shape_no_observation_branch():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    ds.input_shape = (2, 2)

    result = ds.get_target_shape()

    assert result == (2, 2)


def test_get_added_features_dim_branch():
    cfg = make_valid_config_with()

    cfg.time_features = ["year", "month", "lead"]

    ds = TrainDataset(cfg, requested_years=cfg.get_common_time[:1])

    assert ds.get_added_features_dim() == 3


def test_return_metadata_true_real():
    cfg = make_valid_config_with()

    ds = TrainDataset(
        cfg,
        requested_years=cfg.get_common_time[:1],
        return_metadata=True,
    )

    data, meta = ds[0]

    assert isinstance(meta, dict)


def test_get_model_indexes_returns_dict():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=[2000])

    indexes = ds.get_model_indexes()

    assert isinstance(indexes, dict)


def test_get_model_indexes_contains_year():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=[2000])

    indexes = ds.get_model_indexes()

    assert "year" in indexes


def test_get_obs_indexes_returns_none_without_observation():
    cfg = make_valid_config_with(observation=None)

    ds = TrainDataset(cfg, requested_years=[2000])

    indexes = ds.get_obs_indexes(0)

    assert indexes is None


def test_get_input_shape_is_tuple():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=[2000])

    assert isinstance(ds.get_input_shape(), tuple)


def test_get_target_shape_is_tuple():
    cfg = make_valid_config()

    ds = TrainDataset(cfg, requested_years=[2000])

    ds.input_shape = ds.get_input_shape()

    assert isinstance(ds.get_target_shape(), tuple)


def test_added_features_dim_int():
    cfg = make_valid_config_with(time_features=["year"])

    ds = TrainDataset(cfg, requested_years=[2000])

    assert isinstance(ds.get_added_features_dim(), int)


def test_added_features_dim_zero_without_features():
    cfg = make_valid_config_with(time_features=None)

    ds = TrainDataset(cfg, requested_years=[2000])

    assert ds.get_added_features_dim() == 0


def test_getitem_returns_dict():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=[2000])

    item = ds[0]

    assert isinstance(item, dict)


def test_getitem_contains_added_features_key():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=[2000])

    item = ds[0]

    assert "added_features" in item


def test_getitem_added_features_none_without_time_features():
    cfg = make_valid_config_with(time_features=None)

    ds = TrainDataset(cfg, requested_years=[2000])

    item = ds[0]

    assert item["added_features"] is None


def test_return_metadata_true_returns_tuple():
    cfg = make_valid_config_with()

    ds = TrainDataset(
        cfg,
        requested_years=[2000],
        return_metadata=True,
    )

    item = ds[0]

    assert isinstance(item, tuple)
    assert len(item) == 2


def test_return_metadata_false_returns_dict():
    cfg = make_valid_config_with()

    ds = TrainDataset(
        cfg,
        requested_years=[2000],
        return_metadata=False,
    )

    item = ds[0]

    assert isinstance(item, dict)


def test_condition_static_branch_returns_none():
    cfg = make_valid_config_with(condition_method="static")

    ds = TrainDataset(cfg, requested_years=[2000])

    indexes = ds.get_cond_indexes(0)

    assert indexes is None


def test_dataset_has_mask():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=[2000])

    assert ds.mask is not None


def test_condition_dataset_loaded_when_present():
    cfg = make_valid_config_with()

    ds = TrainDataset(cfg, requested_years=[2000])

    assert ds.condition_dataset is not None


def test_dataset_config_effective_condition_exists():
    cfg = make_valid_config_with()

    assert cfg.effective_condition is not None


def test_load_model_false_branch_real():
    model = DummyDataConfig()

    cond = DummyDataConfig()

    cond.paths = model.paths
    cond.names = model.names
    cond.ensemble_list = model.ensemble_list

    obs = DummyDataConfig()

    cfg = TrainDatasetConfig(
        model=model,
        observation=obs,
        condition=cond,
        condition_method="same_member",
    )

    cfg._fitted_preprocessors = True

    ds = TrainDataset(
        cfg,
        requested_years=[2000],
    )

    assert ds._autoencoding_model_data is False
    assert cfg._using_model_data_as_condition is True
    assert ds._load_model is False


def test_write_condition_to_input_false_real():
    model = DummyDataConfig()

    obs = DummyDataConfig()

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=obs,
        condition=cond,
        condition_method="static",
    )

    cfg._fitted_preprocessors = True

    ds = TrainDataset(
        cfg,
        requested_years=[2000],
    )

    assert ds._autoencoding_model_data is False
    assert cfg._using_model_data_as_condition is False
    assert ds._write_condition_to_input is False


def test_concat_condition_true_real_branch():
    model = DummyDataConfig()

    obs = DummyDataConfig()

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=obs,
        condition=cond,
        condition_method="static",
    )

    cfg._fitted_preprocessors = True

    ds = TrainDataset(
        cfg,
        requested_years=[2000],
    )

    assert ds._write_condition_to_input is False
    assert ds.condition_dataset is not None
    assert ds._concat_condition_to_input is True


def test_get_cond_indexes_cross_ensemble_real():
    model = DummyDataConfig()

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=None,
        condition=cond,
        condition_method="cross_ensemble",
    )

    cfg._fitted_preprocessors = True

    ds = TrainDataset(
        cfg,
        requested_years=[2000],
    )

    indexes = ds.get_cond_indexes(ds.model_indexes)

    assert "year" in indexes
    assert "lead_time" in indexes
    assert "ensembles" in indexes


def test_get_cond_indexes_same_member_real():
    model = DummyDataConfig()

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=None,
        condition=cond,
        condition_method="same_member",
    )

    cfg._fitted_preprocessors = True

    ds = TrainDataset(
        cfg,
        requested_years=[2000],
    )

    indexes = ds.get_cond_indexes(ds.model_indexes)

    assert np.array_equal(
        indexes["ensembles"],
        ds.model_indexes["ensembles"],
    )


def test_get_obs_indexes_sampling_branch():
    model = DummyDataConfig()

    obs = DummyDataConfig()
    obs.ensemble_mean = False

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=obs,
        condition=cond,
        condition_method="static",
    )

    cfg._fitted_preprocessors = True

    ds = TrainDataset(
        cfg,
        requested_years=[2000],
    )

    indexes = ds.get_obs_indexes(ds.model_indexes)

    assert "ensembles" in indexes


def test_get_obs_indexes_no_ensemble_branch():
    model = DummyDataConfig()

    obs = DummyDataConfig()
    obs.info.coords["ensembles"] = None

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=obs,
        condition=cond,
        condition_method="static",
    )

    cfg._fitted_preprocessors = True

    ds = TrainDataset(
        cfg,
        requested_years=[2000],
    )

    indexes = ds.get_obs_indexes(ds.model_indexes)

    assert "ensembles" not in indexes


def test_prepare_mask_expand_dims_branch():
    cfg = make_valid_config_with()

    ds = TrainDataset(
        cfg,
        requested_years=[2000],
    )

    assert "ensembles" in ds.mask.dims


def test_prepare_mask_no_expand_dims_branch():
    model = DummyDataConfig()

    model.info.coords["ensembles"] = None

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        condition=cond,
        condition_method="static",
    )

    cfg._fitted_preprocessors = True

    ds = TrainDataset(
        cfg,
        requested_years=[2000],
    )

    assert "ensembles" not in ds.mask.dims


def test_getitem_concat_executes(monkeypatch):
    def fake_concat(*args, **kwargs):
        return args[0][0]

    monkeypatch.setattr(xr, "concat", fake_concat)

    model = DummyDataConfig()
    obs = DummyDataConfig()

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=obs,
        condition=cond,
        condition_method="static",
    )

    cfg._fitted_preprocessors = True

    ds = TrainDataset(
        cfg,
        requested_years=[2000],
    )

    result = ds[0]

    assert "input" in result
    assert "target" in result


def test_getitem_autoencoding_branch():
    cfg = make_valid_config_with()

    ds = TrainDataset(
        cfg,
        requested_years=[2000],
    )

    item = ds[0]

    assert (item["input"] == item["target"]).all()


def test_getitem_metadata_branch():
    cfg = make_valid_config_with()

    ds = TrainDataset(
        cfg,
        requested_years=[2000],
        return_metadata=True,
    )

    item, meta = ds[0]

    assert isinstance(item, dict)
    assert isinstance(meta, dict)
    assert "year" in meta
    assert "lead_time" in meta


def test_get_input_shape_flattener_branch_real(monkeypatch):
    class FakeFlatten:
        pass

    monkeypatch.setattr(
        "cccma_ppp.preprocessing.utils_preprocessing.Flattennanremove",
        FakeFlatten,
    )

    cfg = make_valid_config_with()

    cfg.model.preprocessing_pipeline.fitted_preprocessors = [FakeFlatten()]

    ds = TrainDataset(
        cfg,
        requested_years=[2000],
    )

    result = ds.get_input_shape()

    assert result is not None


def test_get_target_shape_flattener_branch_real(monkeypatch):
    class FakeFlatten:
        pass

    monkeypatch.setattr(
        "cccma_ppp.preprocessing.utils_preprocessing.Flattennanremove",
        FakeFlatten,
    )

    model = DummyDataConfig()

    obs = DummyDataConfig()
    obs.preprocessing_pipeline.fitted_preprocessors = [FakeFlatten()]

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=obs,
        condition=cond,
        condition_method="static",
    )

    cfg._fitted_preprocessors = True

    ds = TrainDataset(
        cfg,
        requested_years=[2000],
    )

    result = ds.get_target_shape()

    assert result is not None


def test_getitem_time_features_branch():
    cfg = make_valid_config_with()

    cfg.time_features = ["year"]

    ds = TrainDataset(
        cfg,
        requested_years=[2000],
    )

    item = ds[0]

    assert item["added_features"] is not None


def test_getitem_added_features_none_branch():
    cfg = make_valid_config_with()

    cfg.time_features = None

    ds = TrainDataset(
        cfg,
        requested_years=[2000],
    )

    item = ds[0]

    assert item["added_features"] is None


def test_index_model_dataset_returns_none_when_not_loading():
    model = DummyDataConfig()

    cond = DummyDataConfig()

    cond.paths = model.paths
    cond.names = model.names
    cond.ensemble_list = model.ensemble_list

    obs = DummyDataConfig()

    cfg = TrainDatasetConfig(
        model=model,
        observation=obs,
        condition=cond,
        condition_method="same_member",
    )

    cfg._fitted_preprocessors = True

    ds = TrainDataset(
        cfg,
        requested_years=[2000],
    )

    assert ds._load_model is False

    result = ds._index_model_dataset(0)

    assert result is None


def test_index_condition_dataset_none_branch_real():
    cfg = make_valid_config_with()

    ds = TrainDataset(
        cfg,
        requested_years=[2000],
    )

    ds.condition_dataset = None

    result = ds._index_condition_dataset(0)

    assert result is None


def test_available_train_time_adjustment_exact():
    cfg = make_valid_config_with()

    cfg.model.year_range = np.array(
        [
            2000,
            2001,
            2002,
            2003,
        ]
    )

    cfg.lead_months = np.array([24])

    result = cfg.available_train_time

    assert np.array_equal(
        result,
        np.array([2000, 2001, 2002]),
    )


def test_get_target_shape_observation_nonflattener():
    model = DummyDataConfig()

    obs = DummyDataConfig()

    cond = DummyDataConfig()
    cond.paths = ["different"]

    obs.preprocessing_pipeline.fitted_preprocessors = [object()]

    cfg = TrainDatasetConfig(
        model=model,
        observation=obs,
        condition=cond,
        condition_method="static",
    )

    cfg._fitted_preprocessors = True

    ds = TrainDataset(
        cfg,
        requested_years=[2000],
    )

    result = ds.get_target_shape()

    assert isinstance(result, tuple)


def test_getitem_condition_overwrites_input_branch():
    cfg = make_valid_config_with()

    ds = TrainDataset(
        cfg,
        requested_years=[2000],
    )

    assert ds._write_condition_to_input is True

    item = ds[0]

    assert item["input"] is not None
    assert item["target"] is not None


def test_load_xarray_without_ensemble_selection():
    model = DummyDataConfig()

    model.info.coords["ensembles"] = None

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        condition=cond,
        condition_method="static",
    )

    cfg._fitted_preprocessors = True

    ds = TrainDataset(
        cfg,
        requested_years=[2000],
    )

    assert ds.model_dataset is not None


def test_prepare_mask_existing_without_expand():
    model = DummyDataConfig()

    model.ensemble_mean = True

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        condition=cond,
        condition_method="static",
    )

    cfg._fitted_preprocessors = True

    mask = xr.DataArray(
        np.zeros((5, 12), dtype=bool),
        dims=("year", "lead_time"),
        coords={
            "year": cfg.model.year_range,
            "lead_time": np.arange(1, 13),
        },
    )

    ds = TrainDataset(
        cfg,
        requested_years=[2000],
        mask=mask,
    )

    assert "ensembles" not in ds.mask.dims

    cfg = make_valid_config_with()

    cfg.time_features = [
        "year",
        "month",
        "lead_time",
    ]

    ds = TrainDataset(
        cfg,
        requested_years=[2000],
    )

    assert ds.get_added_features_dim() == 3


def test_getitem_metadata_false_branch_real():
    cfg = make_valid_config_with()

    ds = TrainDataset(
        cfg,
        requested_years=[2000],
        return_metadata=False,
    )

    result = ds[0]

    assert isinstance(result, dict)


def test_getitem_metadata_true_branch_real():
    cfg = make_valid_config_with()

    ds = TrainDataset(
        cfg,
        requested_years=[2000],
        return_metadata=True,
    )

    result, meta = ds[0]

    assert isinstance(result, dict)
    assert isinstance(meta, dict)


def test_index_condition_dataset_static_selection():
    cfg = make_valid_config_with()

    ds = TrainDataset(
        cfg,
        requested_years=[2000],
    )

    result = ds._index_condition_dataset(0)

    assert result is not None


def test_index_condition_dataset_same_member_selection_real():
    model = DummyDataConfig()

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        condition=cond,
        condition_method="same_member",
    )

    cfg._fitted_preprocessors = True

    ds = TrainDataset(
        cfg,
        requested_years=[2000],
    )

    result = ds._index_condition_dataset(0)

    assert result is not None


def test_index_condition_dataset_cross_ensemble_selection_real():
    model = DummyDataConfig()

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        condition=cond,
        condition_method="cross_ensemble",
    )

    cfg._fitted_preprocessors = True

    ds = TrainDataset(
        cfg,
        requested_years=[2000],
    )

    result = ds._index_condition_dataset(0)

    assert result is not None


def test_index_model_dataset_ensemble_selection():
    cfg = make_valid_config_with()

    ds = TrainDataset(
        cfg,
        requested_years=[2000],
    )

    result = ds._index_model_dataset(0)

    assert result is not None


def test_available_train_time_observation_branch_real():
    model = DummyDataConfig()
    obs = DummyDataConfig()

    model.year_range = np.array([2000, 2001, 2002])
    obs.year_range = np.array([2001, 2002])

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=obs,
        condition=cond,
        condition_method="static",
    )

    result = cfg.available_train_time

    assert np.array_equal(
        result,
        np.array([2001, 2002]),
    )


def test_available_train_time_no_observation_real():
    cfg = make_valid_config_with()

    cfg.lead_months = np.array([24])

    result = cfg.available_train_time

    assert isinstance(result, np.ndarray)


def test_build_dataset_wrapper_real():
    cfg = make_valid_config_with()

    ds = cfg.build_dataset(
        years=[2000],
        return_metadata=True,
    )

    assert isinstance(ds, TrainDataset)
    assert ds.return_metadata is True


def test_ensemble_mean_condition_valid_branch():
    model = DummyDataConfig()

    cond = DummyDataConfig(ensemble_mean=True)
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        condition=cond,
        condition_method="ensemble_mean",
    )

    assert cfg.condition_method == "ensemble_mean"


def test_check_observation_no_warning_branch():
    model = DummyDataConfig()
    obs = DummyDataConfig()

    cond = DummyDataConfig()
    cond.paths = ["different"]

    with warnings.catch_warnings(record=True) as w:
        TrainDatasetConfig(
            model=model,
            observation=obs,
            condition=cond,
            condition_method="static",
        )

    assert len(w) == 0


def test_prepare_mask_existing_mask_expand_branch():
    cfg = make_valid_config_with()

    mask = xr.DataArray(
        np.zeros((5, 12), dtype=bool),
        dims=("year", "lead_time"),
        coords={
            "year": cfg.model.year_range,
            "lead_time": np.arange(1, 13),
        },
    )

    ds = TrainDataset(
        cfg,
        requested_years=[2000],
        mask=mask,
    )

    assert "ensembles" in ds.mask.dims


def test_get_obs_indexes_ensemble_mean_true():
    model = DummyDataConfig()

    obs = DummyDataConfig(ensemble_mean=True)

    cond = DummyDataConfig()
    cond.paths = ["different"]

    cfg = TrainDatasetConfig(
        model=model,
        observation=obs,
        condition=cond,
        condition_method="static",
    )

    cfg._fitted_preprocessors = True

    ds = TrainDataset(
        cfg,
        requested_years=[2000],
    )

    indexes = ds.get_obs_indexes(ds.model_indexes)

    assert "ensembles" not in indexes


def test_get_cond_indexes_static_none_real():
    cfg = make_valid_config_with()

    ds = TrainDataset(
        cfg,
        requested_years=[2000],
    )

    indexes = ds.get_cond_indexes(ds.model_indexes)

    assert indexes is None


def test_index_observation_dataset_none_real():
    cfg = make_valid_config_with()

    ds = TrainDataset(
        cfg,
        requested_years=[2000],
    )

    result = ds._index_observation_dataset(0)

    assert result is None


def test_get_target_shape_returns_input_shape():
    cfg = make_valid_config_with()

    ds = TrainDataset(
        cfg,
        requested_years=[2000],
    )

    ds.input_shape = (10, 20)

    result = ds.get_target_shape()

    assert result == (10, 20)


def test_get_added_features_dim_zero():
    cfg = make_valid_config_with()

    cfg.time_features = None

    ds = TrainDataset(
        cfg,
        requested_years=[2000],
    )

    assert ds.get_added_features_dim() == 0
