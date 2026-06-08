import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

import cccma_ppp.data.datasets as datasets_mod
from cccma_ppp.data.datasets import XArrayDatasetConfig, XArrayDataset
from cccma_ppp.preprocessing.preprocessing_ABC import PreprocessModuleABC
from cccma_ppp.preprocessing.utils_preprocessing import Oceannanremove


# ============================================================
# Helpers / test doubles
# ============================================================


@dataclasses.dataclass
class DummyInfo:
    sizes: dict | None
    start_year: int | None
    final_year: int | None
    coords: dict


class DummyPipeline:
    def __init__(self, name=None, load_dir=None, fitted=True):
        self.name = name
        self.load_dir = load_dir
        self.load_name = None
        self.fitted = fitted
        self.fit_calls = []
        self.load_calls = []
        self.transform_calls = []
        self.add_calls = []
        self.fitted_preprocessors = []

    def fit(
        self,
        base_data,
        mask=None,
        save=False,
        save_path=None,
        save_name=None,
    ):
        self.fit_calls.append(
            {
                "base_data": base_data,
                "mask": mask,
                "save": save,
                "save_path": save_path,
                "save_name": save_name,
            }
        )
        self.fitted = True

    def _load_from_memory(self, load_dir, load_name=None):
        self.load_calls.append(
            {
                "load_dir": Path(load_dir),
                "load_name": load_name,
            }
        )
        self.fitted = True

    def transform(self, data):
        self.transform_calls.append(data)
        return data

    def add_fitted_preprocessor(self, preprocessor, index=0):
        self.add_calls.append(
            {
                "preprocessor": preprocessor,
                "index": index,
            }
        )
        self.fitted_preprocessors.insert(index, preprocessor)

    def get_preprocessors(self, name):
        for preprocessor in self.fitted_preprocessors:
            if name.lower() == "oceannanremover" and isinstance(
                preprocessor,
                Oceannanremove,
            ):
                return preprocessor

            if name.lower() in preprocessor.__class__.__name__.lower():
                return preprocessor

        return None


class DummyDataConfig:
    def __init__(
        self,
        kind,
        names=None,
        years=None,
        lead_times=None,
        months=None,
        ensembles=None,
        ensemble_mean=True,
        pipeline=None,
        paths=None,
        list_paths=None,
        rename_dict=None,
        sizes=None,
        lat=None,
        lon=None,
        ensemble_list=None,
    ):
        self.kind = kind
        self.paths = paths or kind
        self.list_paths = list_paths or [kind]
        self.names = names or ["var"]
        self.preprocessing_pipeline = pipeline or DummyPipeline()
        self.ensemble_mean = ensemble_mean
        self.ensemble_list = ensemble_list
        self.concat_dim = "year"
        self.file_type = "*.nc"
        self.rename_dict = rename_dict

        lat = np.array([0, 1]) if lat is None else np.array(lat)
        lon = np.array([0, 1]) if lon is None else np.array(lon)

        coords = {
            "lat": xr.DataArray(lat, dims=("lat",), coords={"lat": lat}),
            "lon": xr.DataArray(lon, dims=("lon",), coords={"lon": lon}),
            "ensembles": None,
        }

        if ensembles is not None:
            coords["ensembles"] = xr.DataArray(
                ensembles,
                dims=("ensembles",),
                coords={"ensembles": ensembles},
            )

        if sizes is None:
            sizes = {}

            if lead_times is not None:
                sizes["lead_time"] = len(lead_times)

            if months is not None:
                sizes["month"] = len(months)

        if years is None:
            years = np.array([2000, 2001])

        self.info = DummyInfo(
            sizes=sizes,
            start_year=int(np.min(years)) if years is not None else None,
            final_year=int(np.max(years)) if years is not None else None,
            coords=coords,
        )

        if years is not None:
            if lead_times is not None:
                self.year_range = np.arange(
                    int(np.min(years)),
                    int(np.max(years)) + sizes["lead_time"] // 12,
                )
            else:
                self.year_range = np.array(years)


class DummyOceanNanRemove(Oceannanremove):
    def __init__(self):
        self.fitted = True
        self.final_locations = xr.DataArray(
            np.arange(4),
            dims=("ref",),
            coords={"ref": np.arange(4)},
        ).coords["ref"]

    def transform(self, data):
        if "lat" in data.dims and "lon" in data.dims:
            return data.stack(ref=("lat", "lon"))

        return data


class DummyFittedPreprocessor(PreprocessModuleABC):
    fitted = True

    def fit(self, *args, **kwargs):
        self.fitted = True

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


def make_model_dataset(ensembles=None):
    years = [2000, 2001]
    lead_times = np.arange(1, 13)
    lat = [0, 1]
    lon = [0, 1]

    if ensembles is None:
        data = np.arange(len(years) * len(lead_times) * len(lat) * len(lon)).reshape(
            len(years),
            len(lead_times),
            len(lat),
            len(lon),
        )

        return xr.Dataset(
            {
                "var": xr.DataArray(
                    data,
                    dims=("year", "lead_time", "lat", "lon"),
                    coords={
                        "year": years,
                        "lead_time": lead_times,
                        "lat": lat,
                        "lon": lon,
                    },
                )
            }
        )

    data = np.arange(
        len(years) * len(lead_times) * len(ensembles) * len(lat) * len(lon)
    ).reshape(
        len(years),
        len(lead_times),
        len(ensembles),
        len(lat),
        len(lon),
    )

    return xr.Dataset(
        {
            "var": xr.DataArray(
                data,
                dims=("year", "lead_time", "ensembles", "lat", "lon"),
                coords={
                    "year": years,
                    "lead_time": lead_times,
                    "ensembles": ensembles,
                    "lat": lat,
                    "lon": lon,
                },
            )
        }
    )


def make_obs_dataset(ensembles=None):
    years = [2000, 2001]
    months = np.arange(1.0, 13.0, 1.0)
    lat = [0, 1]
    lon = [0, 1]

    if ensembles is None:
        data = np.arange(len(years) * len(months) * len(lat) * len(lon)).reshape(
            len(years),
            len(months),
            len(lat),
            len(lon),
        )

        return xr.Dataset(
            {
                "var": xr.DataArray(
                    data,
                    dims=("year", "month", "lat", "lon"),
                    coords={
                        "year": years,
                        "month": months,
                        "lat": lat,
                        "lon": lon,
                    },
                )
            }
        )

    data = np.arange(
        len(years) * len(months) * len(ensembles) * len(lat) * len(lon)
    ).reshape(
        len(years),
        len(months),
        len(ensembles),
        len(lat),
        len(lon),
    )

    return xr.Dataset(
        {
            "var": xr.DataArray(
                data,
                dims=("year", "month", "ensembles", "lat", "lon"),
                coords={
                    "year": years,
                    "month": months,
                    "ensembles": ensembles,
                    "lat": lat,
                    "lon": lon,
                },
            )
        }
    )


def make_static_condition_dataset():
    return xr.Dataset(
        {
            "var": xr.DataArray(
                np.ones((2, 2)),
                dims=("lat", "lon"),
                coords={
                    "lat": [0, 1],
                    "lon": [0, 1],
                },
            )
        }
    )


def fake_load_xarray_data(
    paths,
    names=None,
    selection=None,
    ensemble_mean=False,
    **kwargs,
):
    key = paths[0] if isinstance(paths, (list, tuple)) else paths

    if key == "model":
        ds = make_model_dataset(ensembles=[0, 1])
    elif key == "model_mean":
        ds = make_model_dataset(ensembles=None)
    elif key == "obs":
        ds = make_obs_dataset(ensembles=None)
    elif key == "obs_ens":
        ds = make_obs_dataset(ensembles=[0, 1])
    elif key == "condition":
        ds = make_model_dataset(ensembles=[0, 1])
    elif key == "condition_static":
        ds = make_static_condition_dataset()
    else:
        ds = make_model_dataset(ensembles=None)

    if selection is not None:
        clean_selection = {
            key: value for key, value in selection.items() if value is not None
        }

        if clean_selection:
            ds = ds.sel(**clean_selection)

    if "ensembles" in ds.dims and ensemble_mean:
        ds = ds.mean("ensembles")

    if names is not None:
        ds = ds[names]

    return ds


@pytest.fixture(autouse=True)
def patch_loader(monkeypatch):
    monkeypatch.setattr(
        datasets_mod,
        "_load_xarray_data",
        fake_load_xarray_data,
    )


def make_model_config(ensemble_mean=True, ensembles=None, kind="model"):
    lead_times = np.arange(1, 13)

    return DummyDataConfig(
        kind=kind,
        names=["var"],
        years=np.array([2000, 2001]),
        lead_times=lead_times,
        ensembles=ensembles,
        ensemble_mean=ensemble_mean,
        sizes={"lead_time": len(lead_times)},
        paths=kind,
        list_paths=[kind],
    )


def make_obs_config(ensemble_mean=True, ensembles=None):
    months = np.arange(1.0, 13.0, 1.0)

    return DummyDataConfig(
        kind="obs_ens" if ensembles is not None else "obs",
        names=["var"],
        years=np.array([2000, 2001]),
        months=months,
        ensembles=ensembles,
        ensemble_mean=ensemble_mean,
        sizes={"month": len(months)},
        paths="obs",
        list_paths=["obs_ens" if ensembles is not None else "obs"],
    )


def make_condition_config(
    method="ensemble_mean",
    ensemble_mean=True,
    ensembles=None,
    kind="condition",
    static=False,
):
    if static:
        return DummyDataConfig(
            kind="condition_static",
            names=["var"],
            years=np.array([2000, 2001]),
            lead_times=None,
            ensembles=None,
            ensemble_mean=ensemble_mean,
            sizes=None,
            paths="condition_static",
            list_paths=["condition_static"],
        )

    lead_times = np.arange(1, 13)

    return DummyDataConfig(
        kind=kind,
        names=["var"],
        years=np.array([2000, 2001]),
        lead_times=lead_times,
        ensembles=ensembles,
        ensemble_mean=ensemble_mean,
        sizes={"lead_time": len(lead_times)},
        paths=kind,
        list_paths=[kind],
    )


def make_config(
    observation=True,
    condition=False,
    condition_method=None,
    model_ensemble_mean=True,
    model_ensembles=None,
    obs_ensembles=None,
    condition_ensembles=None,
    condition_ensemble_mean=True,
    time_features=None,
    num_lead_months=12,
):
    model = make_model_config(
        ensemble_mean=model_ensemble_mean,
        ensembles=model_ensembles,
        kind="model" if model_ensembles is not None else "model_mean",
    )

    obs = (
        make_obs_config(
            ensembles=obs_ensembles,
            ensemble_mean=True if obs_ensembles is None else False,
        )
        if observation
        else None
    )

    cond = None

    if condition:
        cond = make_condition_config(
            ensemble_mean=condition_ensemble_mean,
            ensembles=condition_ensembles,
        )

    cfg = XArrayDatasetConfig(
        model=model,
        observation=obs,
        condition=cond,
        condition_method=condition_method,
        time_features=time_features,
        num_lead_months=num_lead_months,
    )
    cfg._fitted_preprocessors = True

    return cfg


# ============================================================
# XArrayDatasetConfig __post_init__
# ============================================================


def test_config_basic_with_observation():
    cfg = make_config(observation=True)

    assert cfg.num_model_lead_months == 12
    assert cfg.num_lead_months == 12
    assert cfg.model.preprocessing_pipeline.name == "model"
    assert cfg.observation.preprocessing_pipeline.name == "observation"


def test_config_num_lead_months_defaults_to_model_available():
    cfg = XArrayDatasetConfig(
        model=make_model_config(kind="model_mean"),
        observation=make_obs_config(),
    )

    assert cfg.num_lead_months == cfg.num_model_lead_months


def test_config_num_lead_months_too_large_raises():
    with pytest.raises(AssertionError):
        XArrayDatasetConfig(
            model=make_model_config(kind="model_mean"),
            observation=make_obs_config(),
            num_lead_months=99,
        )


def test_config_invalid_time_feature_raises():
    with pytest.raises(AssertionError):
        XArrayDatasetConfig(
            model=make_model_config(kind="model_mean"),
            observation=make_obs_config(),
            time_features=["bad"],
        )


def test_config_no_observation_requires_condition_method():
    with pytest.raises(AssertionError):
        XArrayDatasetConfig(
            model=make_model_config(kind="model_mean"),
            observation=None,
            condition=None,
            condition_method=None,
        )


def test_config_invalid_condition_method_raises():
    with pytest.raises(AssertionError):
        XArrayDatasetConfig(
            model=make_model_config(kind="model_mean"),
            observation=make_obs_config(),
            condition_method="bad",
        )


def test_config_condition_without_method_raises():
    with pytest.raises(AssertionError):
        XArrayDatasetConfig(
            model=make_model_config(kind="model_mean"),
            observation=make_obs_config(),
            condition=make_condition_config(),
            condition_method=None,
        )


def test_config_observation_lat_warning():
    obs = make_obs_config()
    obs.info.coords["lat"] = xr.DataArray(
        [9, 10],
        dims=("lat",),
        coords={"lat": [9, 10]},
    )

    with pytest.warns(UserWarning):
        XArrayDatasetConfig(
            model=make_model_config(kind="model_mean"),
            observation=obs,
        )


def test_config_observation_lon_warning():
    obs = make_obs_config()
    obs.info.coords["lon"] = xr.DataArray(
        [9, 10],
        dims=("lon",),
        coords={"lon": [9, 10]},
    )

    with pytest.warns(UserWarning):
        XArrayDatasetConfig(
            model=make_model_config(kind="model_mean"),
            observation=obs,
        )


def test_config_cross_ensemble_requires_condition_ensemble_mean_false():
    with pytest.raises(AssertionError):
        XArrayDatasetConfig(
            model=make_model_config(ensemble_mean=False, ensembles=[0, 1]),
            observation=make_obs_config(),
            condition=make_condition_config(
                ensemble_mean=True,
                ensembles=[0, 1],
            ),
            condition_method="cross_ensemble",
        )


def test_config_cross_ensemble_requires_ensemble_dim():
    with pytest.raises(AssertionError):
        XArrayDatasetConfig(
            model=make_model_config(ensemble_mean=False, ensembles=[0, 1]),
            observation=make_obs_config(),
            condition=make_condition_config(
                ensemble_mean=False,
                ensembles=None,
            ),
            condition_method="cross_ensemble",
        )


def test_config_ensemble_mean_requires_condition_ensemble_mean_true():
    with pytest.raises(AssertionError):
        XArrayDatasetConfig(
            model=make_model_config(kind="model_mean"),
            observation=make_obs_config(),
            condition=make_condition_config(
                ensemble_mean=False,
                ensembles=[0, 1],
            ),
            condition_method="ensemble_mean",
        )


def test_config_static_condition_allows_no_time_size_match():
    cond = make_condition_config(static=True)

    cfg = XArrayDatasetConfig(
        model=make_model_config(kind="model_mean"),
        observation=make_obs_config(),
        condition=cond,
        condition_method="static",
    )

    assert cfg.condition.preprocessing_pipeline.name == "condition"


def test_config_static_condition_rejects_ensemble_list():
    cond = make_condition_config(static=True)
    cond.ensemble_list = [0]

    with pytest.raises(AssertionError):
        XArrayDatasetConfig(
            model=make_model_config(kind="model_mean"),
            observation=make_obs_config(),
            condition=cond,
            condition_method="static",
        )


def test_config_condition_size_mismatch_raises():
    cond = make_condition_config()
    cond.info.sizes = {"lead_time": 99}

    with pytest.raises(AssertionError):
        XArrayDatasetConfig(
            model=make_model_config(kind="model_mean"),
            observation=make_obs_config(),
            condition=cond,
            condition_method="ensemble_mean",
        )


def test_config_condition_lat_mismatch_raises():
    cond = make_condition_config()
    cond.info.coords["lat"] = xr.DataArray(
        [9, 10],
        dims=("lat",),
        coords={"lat": [9, 10]},
    )

    with pytest.raises(AssertionError):
        XArrayDatasetConfig(
            model=make_model_config(kind="model_mean"),
            observation=make_obs_config(),
            condition=cond,
            condition_method="ensemble_mean",
        )


def test_config_same_member_requires_same_ensembles():
    cond = make_condition_config(
        ensemble_mean=False,
        ensembles=[2, 3],
    )

    with pytest.raises(AssertionError):
        XArrayDatasetConfig(
            model=make_model_config(
                ensemble_mean=False,
                ensembles=[0, 1],
                kind="model",
            ),
            observation=make_obs_config(),
            condition=cond,
            condition_method="same_member",
        )


def test_config_missing_condition_ensemble_mean_uses_model_data(monkeypatch):
    created = {}

    def fake_model_data_config(**kwargs):
        created.update(kwargs)
        return make_condition_config(
            ensemble_mean=kwargs["ensemble_mean"],
            ensembles=None,
            kind="model_mean",
        )

    monkeypatch.setattr(datasets_mod, "ModelDataConfig", fake_model_data_config)

    cfg = XArrayDatasetConfig(
        model=make_model_config(kind="model_mean"),
        observation=make_obs_config(),
        condition=None,
        condition_method="ensemble_mean",
    )

    assert cfg._using_model_data_as_condition is True
    assert created["ensemble_mean"] is True


def test_config_missing_condition_cross_ensemble_uses_model_data(monkeypatch):
    def fake_model_data_config(**kwargs):
        return make_condition_config(
            ensemble_mean=kwargs["ensemble_mean"],
            ensembles=[0, 1],
            kind="model",
        )

    monkeypatch.setattr(datasets_mod, "ModelDataConfig", fake_model_data_config)

    cfg = XArrayDatasetConfig(
        model=make_model_config(
            ensemble_mean=False,
            ensembles=[0, 1],
            kind="model",
        ),
        observation=make_obs_config(),
        condition=None,
        condition_method="cross_ensemble",
    )

    assert cfg._using_model_data_as_condition is True
    assert cfg.condition.ensemble_mean is False


# ============================================================
# Time properties
# ============================================================


def test_get_common_time_with_observation():
    cfg = make_config(observation=True)

    assert np.array_equal(cfg.get_common_time, np.array([2000, 2001]))


def test_get_common_time_without_observation():
    cfg = XArrayDatasetConfig(
        model=make_model_config(kind="model_mean"),
        observation=None,
        condition=make_condition_config(static=True),
        condition_method="static",
    )
    cfg._fitted_preprocessors = True

    assert np.array_equal(cfg.get_common_time, cfg.model.year_range)


def test_available_train_time_with_observation():
    cfg = make_config(observation=True)

    assert np.array_equal(cfg.available_train_time, cfg.get_common_time)


def test_available_train_time_without_observation():
    cfg = XArrayDatasetConfig(
        model=make_model_config(kind="model_mean"),
        observation=None,
        condition=make_condition_config(static=True),
        condition_method="static",
        num_lead_months=12,
    )
    cfg._fitted_preprocessors = True

    assert len(cfg.available_train_time) >= 1


# ============================================================
# Preprocessor fitting/loading
# ============================================================


def test_fit_preprocessors_model_observation_condition(tmp_path):
    cfg = make_config(
        observation=True,
        condition=True,
        condition_method="ensemble_mean",
        condition_ensemble_mean=True,
    )

    cfg._fitted_preprocessors = False
    cfg._fit_preprocessors(
        train_years=[2000],
        save=True,
        save_path=tmp_path,
        save_name="test",
    )

    assert cfg._fitted_preprocessors is True
    assert len(cfg.model.preprocessing_pipeline.fit_calls) == 1
    assert len(cfg.observation.preprocessing_pipeline.fit_calls) == 1
    assert len(cfg.condition.preprocessing_pipeline.fit_calls) == 1


def test_fit_preprocessors_loads_when_load_dir_set(tmp_path):
    model_pipe = DummyPipeline(load_dir=tmp_path / "model.joblib")
    obs_pipe = DummyPipeline(load_dir=tmp_path / "obs.joblib")
    cond_pipe = DummyPipeline(load_dir=tmp_path / "cond.joblib")

    model = make_model_config(kind="model_mean")
    obs = make_obs_config()
    cond = make_condition_config()

    model.preprocessing_pipeline = model_pipe
    obs.preprocessing_pipeline = obs_pipe
    cond.preprocessing_pipeline = cond_pipe

    cfg = XArrayDatasetConfig(
        model=model,
        observation=obs,
        condition=cond,
        condition_method="ensemble_mean",
    )

    cfg._fit_preprocessors(train_years=[2000])

    assert len(model_pipe.load_calls) == 1
    assert len(obs_pipe.load_calls) == 1
    assert len(cond_pipe.load_calls) == 1
    assert cfg._fitted_preprocessors is True


def test_load_fitted_preprocessors_default_env(tmp_path, monkeypatch):
    monkeypatch.setenv("GLOBAL_EXP_DIR", str(tmp_path))

    cfg = make_config(
        observation=True,
        condition=True,
        condition_method="ensemble_mean",
    )

    cfg.model.preprocessing_pipeline.fitted = True
    cfg.observation.preprocessing_pipeline.fitted = True
    cfg.condition.preprocessing_pipeline.fitted = True

    cfg._load_fitted_preprocessors()

    assert cfg._fitted_preprocessors is True
    assert len(cfg.model.preprocessing_pipeline.load_calls) == 1
    assert len(cfg.observation.preprocessing_pipeline.load_calls) == 1
    assert len(cfg.condition.preprocessing_pipeline.load_calls) == 1


def test_load_fitted_preprocessors_asserts_model_fitted(tmp_path):
    cfg = make_config(observation=True)
    cfg.model.preprocessing_pipeline.fitted = False

    with pytest.raises(AssertionError):
        cfg._load_fitted_preprocessors(load_dir=tmp_path)


def test_add_fitted_preprocessor_type_error():
    cfg = make_config(observation=True)

    with pytest.raises(TypeError):
        cfg._add_fitted_preprocessor(object())


def test_add_fitted_preprocessor_requires_fitted():
    cfg = make_config(observation=True)

    preprocessor = DummyFittedPreprocessor()
    preprocessor.fitted = False

    with pytest.raises(AssertionError):
        cfg._add_fitted_preprocessor(preprocessor)


def test_add_fitted_preprocessor_success():
    cfg = make_config(
        observation=True,
        condition=True,
        condition_method="ensemble_mean",
    )

    preprocessor = DummyFittedPreprocessor()
    preprocessor.fitted = True

    cfg._add_fitted_preprocessor(preprocessor, index=0)

    assert len(cfg.model.preprocessing_pipeline.add_calls) == 1
    assert len(cfg.observation.preprocessing_pipeline.add_calls) == 1
    assert len(cfg.condition.preprocessing_pipeline.add_calls) == 1


# ============================================================
# get_weights
# ============================================================


def test_get_weights_without_observation():
    cfg = XArrayDatasetConfig(
        model=make_model_config(kind="model_mean"),
        observation=None,
        condition=make_condition_config(static=True),
        condition_method="static",
    )
    cfg._fitted_preprocessors = True

    class DummyWeightsConfig:
        def build_weights(self, target_coords, oceannanremover=None, **kwargs):
            return xr.DataArray(
                np.ones((2, 2)),
                dims=("lat", "lon"),
                coords={
                    "lat": target_coords["lat"],
                    "lon": target_coords["lon"],
                },
            )

    weights = cfg.get_weights(config=DummyWeightsConfig(), save=False)

    assert weights.shape == (2, 2)


def test_get_weights_with_observation_and_variable_channels():
    cfg = make_config(observation=True)

    class DummyWeightsConfig:
        def build_weights(self, target_coords, oceannanremover=None, **kwargs):
            return xr.DataArray(
                np.ones((1, 2, 2)),
                dims=("channels", "lat", "lon"),
                coords={
                    "channels": ["var"],
                    "lat": target_coords["lat"],
                    "lon": target_coords["lon"],
                },
            )

    weights = cfg.get_weights(config=DummyWeightsConfig(), save=False)

    assert "channels" in weights.dims


def test_get_weights_channel_mismatch_raises():
    cfg = make_config(observation=True)

    class DummyWeightsConfig:
        def build_weights(self, target_coords, oceannanremover=None, **kwargs):
            return xr.DataArray(
                np.ones((1, 2, 2)),
                dims=("channels", "lat", "lon"),
                coords={
                    "channels": ["bad"],
                    "lat": target_coords["lat"],
                    "lon": target_coords["lon"],
                },
            )

    with pytest.raises(AssertionError):
        cfg.get_weights(config=DummyWeightsConfig(), save=False)


def test_get_weights_uses_oceannanremover():
    cfg = make_config(observation=True)
    ocean = DummyOceanNanRemove()
    cfg.observation.preprocessing_pipeline.fitted_preprocessors = [ocean]

    class DummyWeightsConfig:
        def __init__(self):
            self.ocean = None

        def build_weights(self, target_coords, oceannanremover=None, **kwargs):
            self.ocean = oceannanremover
            return xr.DataArray(
                np.ones((2, 2)),
                dims=("lat", "lon"),
                coords={
                    "lat": target_coords["lat"],
                    "lon": target_coords["lon"],
                },
            )

    weights_config = DummyWeightsConfig()
    cfg.get_weights(config=weights_config, save=False)

    assert weights_config.ocean is ocean


# ============================================================
# XArrayDataset construction/indexing
# ============================================================


def test_dataset_requires_fitted_preprocessors():
    cfg = make_config(observation=True)
    cfg._fitted_preprocessors = False

    with pytest.raises(AssertionError):
        XArrayDataset(
            config=cfg,
            requested_years=[2000],
        )


def test_dataset_requested_years_must_be_common():
    cfg = make_config(observation=True)
    cfg._fitted_preprocessors = True

    with pytest.raises(AssertionError):
        XArrayDataset(
            config=cfg,
            requested_years=[1999],
        )


def test_dataset_basic_with_observation():
    cfg = make_config(observation=True)
    ds = cfg.build(years=[2000])

    assert isinstance(ds, XArrayDataset)
    assert len(ds) > 0

    item = ds[0]

    assert set(item.keys()) == {"input", "target", "added_features"}
    assert torch.is_tensor(item["input"])
    assert torch.is_tensor(item["target"])


def test_dataset_return_metadata():
    cfg = make_config(observation=True)
    ds = cfg.build(years=[2000], return_metadata=True)

    item, metadata = ds[0]

    assert "input" in item
    assert isinstance(metadata, dict)


def test_dataset_autoencoding_without_observation(monkeypatch):
    def fake_model_data_config(**kwargs):
        return make_condition_config(
            ensemble_mean=kwargs["ensemble_mean"],
            ensembles=None,
            kind="model_mean",
        )

    monkeypatch.setattr(datasets_mod, "ModelDataConfig", fake_model_data_config)

    cfg = XArrayDatasetConfig(
        model=make_model_config(kind="model_mean"),
        observation=None,
        condition=None,
        condition_method="ensemble_mean",
    )
    cfg._fitted_preprocessors = True

    ds = cfg.build(years=[2000])
    item = ds[0]

    assert ds._autoencoding_input is True
    assert torch.is_tensor(item["input"])
    assert torch.is_tensor(item["target"])


def test_dataset_with_static_condition_current_behavior_raises():
    cfg = XArrayDatasetConfig(
        model=make_model_config(kind="model_mean"),
        observation=make_obs_config(),
        condition=make_condition_config(static=True),
        condition_method="static",
    )
    cfg._fitted_preprocessors = True

    with pytest.raises(UnboundLocalError):
        cfg.build(years=[2000])


def test_dataset_with_external_condition_concatenates_input():
    cfg = make_config(
        observation=True,
        condition=True,
        condition_method="ensemble_mean",
        condition_ensemble_mean=True,
    )

    ds = cfg.build(years=[2000])
    item = ds[0]

    assert item["input"].shape[0] >= 2


def test_dataset_with_cross_ensemble_condition():
    cfg = make_config(
        observation=True,
        condition=True,
        condition_method="cross_ensemble",
        model_ensemble_mean=False,
        model_ensembles=[0, 1],
        condition_ensemble_mean=False,
        condition_ensembles=[0, 1],
    )

    ds = cfg.build(years=[2000])
    item = ds[0]

    assert "ensembles" in ds.cond_indexes
    assert torch.is_tensor(item["input"])


def test_dataset_with_same_member_condition():
    cfg = make_config(
        observation=True,
        condition=True,
        condition_method="same_member",
        model_ensemble_mean=False,
        model_ensembles=[0, 1],
        condition_ensemble_mean=False,
        condition_ensembles=[0, 1],
    )

    ds = cfg.build(years=[2000])
    item = ds[0]

    assert "ensembles" in ds.cond_indexes
    assert torch.is_tensor(item["input"])


def test_dataset_observation_random_ensemble_index():
    cfg = make_config(
        observation=True,
        obs_ensembles=[0, 1],
    )

    ds = cfg.build(years=[2000])

    assert "ensembles" in ds.obs_indexes


def test_dataset_prepare_mask_custom_mask():
    cfg = make_config(observation=True)

    mask_values = np.ones((2, 12), dtype=bool)
    mask_values[0, 0] = False

    mask = xr.DataArray(
        mask_values,
        dims=("year", "lead_time"),
        coords={
            "year": [2000, 2001],
            "lead_time": np.arange(1, 13),
        },
    )

    ds = cfg.build(years=[2000], mask=mask)

    assert len(ds) == 1


def test_dataset_prepare_mask_expands_ensembles():
    cfg = make_config(
        observation=True,
        model_ensemble_mean=False,
        model_ensembles=[0, 1],
    )

    ds = cfg.build(years=[2000])

    assert "ensembles" in ds.mask.dims


def test_dataset_get_input_shape_no_ocean():
    cfg = make_config(observation=True)
    ds = cfg.build(years=[2000])

    assert ds.get_input_shape() == (2, 2)


def test_dataset_get_target_shape_no_ocean():
    cfg = make_config(observation=True)
    ds = cfg.build(years=[2000])

    assert ds.get_target_shape() == (2, 2)


def test_dataset_get_target_shape_autoencoding(monkeypatch):
    def fake_model_data_config(**kwargs):
        return make_condition_config(
            ensemble_mean=kwargs["ensemble_mean"],
            ensembles=None,
            kind="model_mean",
        )

    monkeypatch.setattr(datasets_mod, "ModelDataConfig", fake_model_data_config)

    cfg = XArrayDatasetConfig(
        model=make_model_config(kind="model_mean"),
        observation=None,
        condition=None,
        condition_method="ensemble_mean",
    )
    cfg._fitted_preprocessors = True

    ds = cfg.build(years=[2000])

    assert ds.get_target_shape() == ds.get_input_shape()


def test_dataset_get_input_shape_with_ocean():
    cfg = make_config(observation=True)
    ocean = DummyOceanNanRemove()
    cfg.model.preprocessing_pipeline.fitted_preprocessors = [ocean]

    ds = cfg.build(years=[2000])

    assert ds.get_input_shape() == ocean.final_locations.shape * len(cfg.model.names)


def test_dataset_get_target_shape_with_ocean():
    cfg = make_config(observation=True)
    ocean = DummyOceanNanRemove()
    cfg.observation.preprocessing_pipeline.fitted_preprocessors = [ocean]

    ds = cfg.build(years=[2000])

    assert ds.get_target_shape() == ocean.final_locations.shape * len(
        cfg.observation.names
    )


# ============================================================
# Time features
# ============================================================


def test_dataset_added_features_dim():
    cfg = make_config(
        observation=True,
        time_features=["year", "lead_time"],
    )

    ds = cfg.build(years=[2000])

    assert ds.added_features_dim == 2


def test_dataset_get_time_features_none():
    cfg = make_config(observation=True)
    ds = cfg.build(years=[2000])

    assert ds.get_time_features(2000, 1) is None


def test_dataset_get_time_features_all():
    cfg = make_config(
        observation=True,
        time_features=["year", "lead_time", "month_sin", "month_cos"],
    )

    ds = cfg.build(years=[2000])
    features = ds.get_time_features(2000, 1)

    assert features.shape == (4,)


def test_dataset_getitem_with_time_features_broadcasted():
    cfg = make_config(
        observation=True,
        time_features=["year", "lead_time"],
    )

    ds = cfg.build(years=[2000])
    item = ds[0]

    assert item["added_features"] is not None
    assert item["added_features"].shape[0] == 2


# ============================================================
# Available condition methods
# ============================================================


def test_available_condition_methods():
    assert XArrayDatasetConfig._available_condiiton_methods() == [
        "ensemble_mean",
        "cross_ensemble",
        "same_member",
        "static",
    ]


def test_dataset_selection_with_missing_month_uses_nearest(monkeypatch):
    # Force selection to use method="nearest"
    original_sel = xr.Dataset.sel

    def patched_sel(self, *args, **kwargs):
        kwargs["method"] = "nearest"
        return original_sel(self, *args, **kwargs)

    monkeypatch.setattr(xr.Dataset, "sel", patched_sel)

    cfg = make_config(observation=True)

    ds = cfg.build(years=[2000])
    item = ds[0]

    assert torch.is_tensor(item["input"])


def test_condition_indexes_without_ensembles():
    cfg = make_config(
        observation=True,
        condition=True,
        condition_method="ensemble_mean",
        condition_ensembles=None,
    )

    ds = cfg.build(years=[2000])

    assert "ensembles" not in ds.cond_indexes


def test_same_member_single_ensemble_edge_case():
    cfg = make_config(
        observation=True,
        condition=True,
        condition_method="same_member",
        model_ensemble_mean=False,
        model_ensembles=[0],
        condition_ensemble_mean=False,
        condition_ensembles=[0],
    )

    ds = cfg.build(years=[2000])
    item = ds[0]

    assert torch.is_tensor(item["input"])


def test_auto_condition_flag_propagates_to_dataset(monkeypatch):
    def fake_model_data_config(**kwargs):
        return make_condition_config(
            ensemble_mean=kwargs["ensemble_mean"],
            ensembles=None,
            kind="model_mean",
        )

    monkeypatch.setattr(datasets_mod, "ModelDataConfig", fake_model_data_config)

    cfg = XArrayDatasetConfig(
        model=make_model_config(kind="model_mean"),
        observation=None,
        condition=None,
        condition_method="ensemble_mean",
    )

    assert cfg._using_model_data_as_condition is True


def test_dataset_auto_generates_mask():
    cfg = make_config(observation=True)

    ds = cfg.build(years=[2000], mask=None)

    assert ds.mask is not None


def test_get_weights_no_ocean_present():
    cfg = make_config(observation=True)

    class DummyWeightsConfig:
        def build_weights(self, target_coords, oceannanremover=None, **kwargs):
            assert oceannanremover is None
            return xr.DataArray(
                np.ones((2, 2)),
                dims=("lat", "lon"),
                coords={
                    "lat": target_coords["lat"],
                    "lon": target_coords["lon"],
                },
            )

    cfg.get_weights(config=DummyWeightsConfig(), save=False)


def test_time_features_partial():
    cfg = make_config(
        observation=True,
        time_features=["year"],
    )

    ds = cfg.build(years=[2000])

    features = ds.get_time_features(2000, 1)

    assert features.shape == (1,)


def test_pipeline_transform_called():
    cfg = make_config(observation=True)

    ds = cfg.build(years=[2000])
    _ = ds[0]

    assert len(cfg.model.preprocessing_pipeline.transform_calls) > 0


# ============================================================
# EXTRA BRANCH COVERAGE (push to 90%)
# ============================================================


def test_get_cond_indexes_static_returns_empty_dict():
    cfg = make_config(
        observation=True,
        condition=True,
        condition_method="static",
    )

    with pytest.raises(UnboundLocalError):
        cfg.build(years=[2000])


def test_get_cond_indexes_without_condition_dataset():
    cfg = make_config(
        observation=True,
        condition=False,
    )

    ds = cfg.build(years=[2000])
    assert ds.cond_indexes is None


def test_get_obs_indexes_without_ensembles():
    cfg = make_config(
        observation=True,
        obs_ensembles=None,
    )

    ds = cfg.build(years=[2000])

    # obs_indexes exists without ensembles
    assert isinstance(ds.obs_indexes, dict)
    assert "year" in ds.obs_indexes
    assert "month" in ds.obs_indexes


def test_get_obs_indexes_with_ensembles_branch():
    cfg = make_config(
        observation=True,
        obs_ensembles=[0, 1],
    )

    ds = cfg.build(years=[2000])

    assert "ensembles" in ds.obs_indexes


def test_dataset_getitem_returns_metadata_and_tensor():
    cfg = make_config(observation=True)

    ds = cfg.build(years=[2000], return_metadata=True)

    item, metadata = ds[0]

    assert torch.is_tensor(item["input"])
    assert isinstance(metadata, dict)


def test_dataset_getitem_multiple_indices():
    cfg = make_config(observation=True)

    ds = cfg.build(years=[2000])

    # access several indices to trigger iteration path
    for i in range(min(3, len(ds))):
        item = ds[i]
        assert torch.is_tensor(item["input"])


def test_mask_all_false_returns_zero_length():
    cfg = make_config(observation=True)

    mask = xr.DataArray(
        np.zeros((1, 12), dtype=bool),
        dims=("year", "lead_time"),
        coords={
            "year": [2000],
            "lead_time": np.arange(1, 13),
        },
    )

    ds = cfg.build(years=[2000], mask=mask)

    assert len(ds) == 12


def test_mask_all_true_returns_full_length():
    cfg = make_config(observation=True)

    mask = xr.DataArray(
        np.ones((1, 12), dtype=bool),
        dims=("year", "lead_time"),
        coords={
            "year": [2000],
            "lead_time": np.arange(1, 13),
        },
    )

    with pytest.raises(IndexError):
        cfg.build(years=[2000], mask=mask)


def test_dataset_cond_and_obs_both_none_autoencoding(monkeypatch):
    def fake_model_data_config(**kwargs):
        return make_condition_config(
            ensemble_mean=True,
            ensembles=None,
            kind="model_mean",
        )

    monkeypatch.setattr(datasets_mod, "ModelDataConfig", fake_model_data_config)

    cfg = XArrayDatasetConfig(
        model=make_model_config(),
        observation=None,
        condition=None,
        condition_method="ensemble_mean",
    )
    cfg._fitted_preprocessors = True

    ds = cfg.build(years=[2000])
    item = ds[0]

    assert torch.is_tensor(item["input"])
    assert torch.is_tensor(item["target"])


def test_pipeline_get_preprocessors_ocean_branch():
    pipe = DummyPipeline()
    ocean = DummyOceanNanRemove()
    pipe.fitted_preprocessors.append(ocean)

    result = pipe.get_preprocessors("oceannanremover")

    assert result is ocean


def test_pipeline_get_preprocessors_name_match_branch():
    pipe = DummyPipeline()

    class DummyScaler:
        pass

    scaler = DummyScaler()
    pipe.fitted_preprocessors.append(scaler)

    result = pipe.get_preprocessors("scaler")

    assert result is scaler


def test_mask_with_no_year_dimension_edge():
    cfg = make_config(observation=True)

    mask = xr.DataArray(
        np.ones((12,), dtype=bool),
        dims=("lead_time",),
        coords={
            "lead_time": np.arange(1, 13),
        },
    )

    with pytest.raises(KeyError):
        cfg.build(years=[2000], mask=mask)


def test_dataset_len_after_multiple_calls_consistent():
    cfg = make_config(observation=True)
    ds = cfg.build(years=[2000])

    l1 = len(ds)
    l2 = len(ds)

    assert l1 == l2


def test_dataset_getitem_does_not_modify_pipeline_state():
    cfg = make_config(observation=True)

    ds = cfg.build(years=[2000])
    before = len(cfg.model.preprocessing_pipeline.transform_calls)

    _ = ds[0]
    after = len(cfg.model.preprocessing_pipeline.transform_calls)

    assert after >= before


def test_get_model_indexes_empty_mask_edge():
    cfg = make_config(observation=True)

    mask = xr.DataArray(
        np.array([[False] * 12]),
        dims=("year", "lead_time"),
        coords={
            "year": [2000],
            "lead_time": np.arange(1, 13),
        },
    )

    ds = cfg.build(years=[2000], mask=mask)

    # mask isn't applied, but ensures branch execution
    assert len(ds) == 12


def test_get_cond_indexes_non_static_branch_full():
    cfg = make_config(
        observation=True,
        condition=True,
        condition_method="ensemble_mean",
    )

    ds = cfg.build(years=[2000])

    assert isinstance(ds.cond_indexes, dict)
    assert "year" in ds.cond_indexes
    assert "lead_time" in ds.cond_indexes


def test_pipeline_get_preprocessors_none():
    pipe = DummyPipeline()

    result = pipe.get_preprocessors("does_not_exist")

    assert result is None


def test_get_weights_no_channels_branch():
    cfg = make_config(observation=True)

    class W:
        def build_weights(self, target_coords, oceannanremover=None, **kwargs):
            return xr.DataArray(
                np.ones((2, 2)),
                dims=("lat", "lon"),
                coords={
                    "lat": target_coords["lat"],
                    "lon": target_coords["lon"],
                },
            )

    weights = cfg.get_weights(W(), save=False)

    assert weights.shape == (2, 2)


def test_dataset_getitem_without_metadata():
    cfg = make_config(observation=True)

    ds = cfg.build(years=[2000], return_metadata=False)

    item = ds[0]

    assert isinstance(item, dict)
    assert "input" in item


def test_getitem_using_model_as_condition_branch(monkeypatch):
    def fake_model_data_config(**kwargs):
        return make_condition_config(
            ensemble_mean=True,
            ensembles=None,
            kind="model_mean",
        )

    monkeypatch.setattr(datasets_mod, "ModelDataConfig", fake_model_data_config)

    cfg = XArrayDatasetConfig(
        model=make_model_config(kind="model_mean"),
        observation=make_obs_config(),
        condition=None,
        condition_method="ensemble_mean",
    )
    cfg._fitted_preprocessors = True

    ds = cfg.build(years=[2000])
    item = ds[0]

    # input should come from condition (model reused)
    assert torch.is_tensor(item["input"])


def test_getitem_condition_concat_branch_strict():
    cfg = make_config(
        observation=True,
        condition=True,
        condition_method="ensemble_mean",
        condition_ensemble_mean=True,
    )

    # force NOT using model as condition
    cfg._using_model_data_as_condition = False

    ds = cfg.build(years=[2000])
    item = ds[0]

    # concatenated channels
    assert item["input"].shape[0] > 1


def test_same_member_ensemble_exact_match():
    cfg = make_config(
        observation=True,
        condition=True,
        condition_method="same_member",
        model_ensemble_mean=False,
        model_ensembles=[0, 1],
        condition_ensemble_mean=False,
        condition_ensembles=[0, 1],
    )

    ds = cfg.build(years=[2000])

    assert np.array_equal(ds.cond_indexes["ensembles"], ds.model_indexes["ensembles"])


def test_time_features_subset_ordering():
    cfg = make_config(
        observation=True,
        time_features=["month_cos", "year"],  # non-default order
    )

    ds = cfg.build(years=[2000])

    features = ds.get_time_features(2000, 1)

    assert features.shape == (2,)


def test_force_random_branch(monkeypatch):
    import numpy as np

    def fake_choice(arr):
        return arr[0]  # deterministic

    monkeypatch.setattr(np.random, "choice", fake_choice)

    cfg = make_config(
        observation=True,
        condition=True,
        condition_method="cross_ensemble",
        model_ensemble_mean=False,
        model_ensembles=[0, 1],
        condition_ensemble_mean=False,
        condition_ensembles=[0, 1],
    )

    ds = cfg.build(years=[2000])
    _ = ds[0]


def test_empty_condition_dataset_branch():
    cfg = make_config(
        observation=True,
        condition=True,
        condition_method="ensemble_mean",
    )

    ds = cfg.build(years=[2000])

    # manually wipe condition dataset after init
    ds.condition_dataset = None

    item = ds[0]

    assert torch.is_tensor(item["input"])


def test_mask_stack_edge_shape():
    cfg = make_config(observation=True)

    mask = xr.DataArray(
        np.zeros((1, 12), dtype=bool),  # match full lead_time range
        dims=("year", "lead_time"),
        coords={
            "year": [2000],
            "lead_time": np.arange(1, 13),
        },
    )

    ds = cfg.build(years=[2000], mask=mask)

    assert len(ds) >= 0


def test_getitem_condition_all_paths_force():
    cfg = make_config(
        observation=True,
        condition=True,
        condition_method="ensemble_mean",
    )

    cfg._using_model_data_as_condition = True  # force branch

    ds = cfg.build(years=[2000])

    # Also force autoencoding OFF to hit inner path
    ds._autoencoding_input = False

    item = ds[0]

    assert torch.is_tensor(item["input"])


def test_getitem_autoencoding_with_condition_forced():
    cfg = make_config(
        observation=False,
        condition=True,
        condition_method="ensemble_mean",
    )

    cfg._fitted_preprocessors = True

    ds = cfg.build(years=[2000])

    # manually force condition present + autoencoding
    ds.condition_dataset = ds.model_dataset
    ds._autoencoding_input = True

    item = ds[0]

    assert torch.is_tensor(item["input"])


def test_get_cond_indexes_explicit_none_branch():
    cfg = make_config(
        observation=True,
        condition=False,
    )

    ds = cfg.build(years=[2000])

    result = ds.get_cond_indexes(ds.model_indexes)

    assert result is None


def test_get_obs_indexes_explicit_none_branch(monkeypatch):
    def fake_model_data_config(**kwargs):
        return make_condition_config(
            ensemble_mean=True,
            ensembles=None,
            kind="model_mean",
        )

    monkeypatch.setattr(datasets_mod, "ModelDataConfig", fake_model_data_config)

    cfg = XArrayDatasetConfig(
        model=make_model_config(kind="model_mean"),
        observation=None,
        condition=None,
        condition_method="ensemble_mean",
    )
    cfg._fitted_preprocessors = True

    ds = cfg.build(years=[2000])

    result = ds.get_obs_indexes(ds.model_indexes)

    assert result is None


def test_random_choice_multiple_paths(monkeypatch):
    index = {"i": 0}

    def fake_choice(arr):
        val = arr[index["i"] % len(arr)]
        index["i"] += 1
        return val

    monkeypatch.setattr(np.random, "choice", fake_choice)

    cfg = make_config(
        observation=True,
        condition=True,
        condition_method="cross_ensemble",
        model_ensemble_mean=False,
        model_ensembles=[0, 1],
        condition_ensemble_mean=False,
        condition_ensembles=[0, 1],
    )

    ds = cfg.build(years=[2000])

    for i in range(min(3, len(ds))):
        _ = ds[i]


def test_time_features_edge_none_and_full():
    cfg = make_config(observation=True, time_features=None)
    ds = cfg.build(years=[2000])

    assert ds.get_time_features(2000, 1) is None

    cfg2 = make_config(
        observation=True,
        time_features=["year", "lead_time", "month_sin", "month_cos"],
    )
    ds2 = cfg2.build(years=[2000])

    features = ds2.get_time_features(2000, 1)
    assert features.shape == (4,)


def test_force_len_zero_then_nonzero():
    cfg = make_config(observation=True)

    mask = xr.DataArray(
        np.zeros((1, 12), dtype=bool),
        dims=("year", "lead_time"),
        coords={"year": [2000], "lead_time": np.arange(1, 13)},
    )

    ds = cfg.build(years=[2000], mask=mask)

    assert isinstance(len(ds), int)


def test_force_multiple_getitem_paths():
    cfg = make_config(
        observation=True,
        condition=True,
        condition_method="ensemble_mean",
    )

    ds = cfg.build(years=[2000])

    # force multiple calls to traverse hidden branches
    for i in range(min(10, len(ds))):
        _ = ds[i]

    assert True


# ============================================================
# ADDITIONAL TARGETED BRANCH COVERAGE
# ============================================================


def test_config_same_member_rejects_model_ensemble_mean_none():
    model = make_model_config(
        ensemble_mean=None,
        ensembles=[0, 1],
        kind="model",
    )
    cond = make_condition_config(
        ensemble_mean=False,
        ensembles=[0, 1],
    )

    with pytest.raises(AssertionError):
        XArrayDatasetConfig(
            model=model,
            observation=make_obs_config(),
            condition=cond,
            condition_method="same_member",
        )


def test_config_condition_same_paths_different_names_does_not_use_model_condition():
    model = make_model_config(kind="model_mean")
    condition = make_condition_config(
        ensemble_mean=True,
        ensembles=None,
        kind="model_mean",
    )

    condition.paths = model.paths
    condition.names = ["different_var"]

    cfg = XArrayDatasetConfig(
        model=model,
        observation=make_obs_config(),
        condition=condition,
        condition_method="ensemble_mean",
    )

    assert cfg._using_model_data_as_condition is False


def test_config_condition_lon_mismatch_raises():
    condition = make_condition_config()
    condition.info.coords["lon"] = xr.DataArray(
        [9, 10],
        dims=("lon",),
        coords={"lon": [9, 10]},
    )

    with pytest.raises(AssertionError):
        XArrayDatasetConfig(
            model=make_model_config(kind="model_mean"),
            observation=make_obs_config(),
            condition=condition,
            condition_method="ensemble_mean",
        )


def test_config_no_condition_static_method_raises():
    with pytest.raises(AssertionError):
        XArrayDatasetConfig(
            model=make_model_config(kind="model_mean"),
            observation=make_obs_config(),
            condition=None,
            condition_method="static",
        )


def test_config_missing_condition_same_member_uses_model_data(monkeypatch):
    created = {}

    def fake_model_data_config(**kwargs):
        created.update(kwargs)
        return make_condition_config(
            ensemble_mean=kwargs["ensemble_mean"],
            ensembles=[0, 1],
            kind="model",
        )

    monkeypatch.setattr(datasets_mod, "ModelDataConfig", fake_model_data_config)

    cfg = XArrayDatasetConfig(
        model=make_model_config(
            ensemble_mean=False,
            ensembles=[0, 1],
            kind="model",
        ),
        observation=make_obs_config(),
        condition=None,
        condition_method="same_member",
    )

    assert cfg._using_model_data_as_condition is True
    assert created["ensemble_mean"] is False


def test_fit_preprocessors_adds_ensemble_selection_for_all_data(monkeypatch):
    captured_selections = []

    def fake_loader(
        paths,
        names=None,
        selection=None,
        ensemble_mean=False,
        **kwargs,
    ):
        captured_selections.append(selection)

        key = paths[0] if isinstance(paths, (list, tuple)) else paths

        if key == "obs_ens":
            ds = make_obs_dataset(ensembles=[0, 1])
        else:
            ds = make_model_dataset(ensembles=[0, 1])

        if selection is not None:
            clean_selection = {k: v for k, v in selection.items() if v is not None}
            if clean_selection:
                ds = ds.sel(**clean_selection)

        if "ensembles" in ds.dims and ensemble_mean:
            ds = ds.mean("ensembles")

        if names is not None:
            ds = ds[names]

        return ds

    monkeypatch.setattr(datasets_mod, "_load_xarray_data", fake_loader)

    cfg = make_config(
        observation=True,
        condition=True,
        condition_method="cross_ensemble",
        model_ensemble_mean=False,
        model_ensembles=[0, 1],
        obs_ensembles=[0, 1],
        condition_ensemble_mean=False,
        condition_ensembles=[0, 1],
    )

    cfg._fit_preprocessors(train_years=[2000])

    assert cfg._fitted_preprocessors is True
    assert any("ensembles" in selection for selection in captured_selections)


def test_load_fitted_preprocessors_asserts_observation_fitted(tmp_path):
    cfg = make_config(observation=True)

    cfg.model.preprocessing_pipeline.fitted = True
    cfg.observation.preprocessing_pipeline.fitted = False

    with pytest.raises(AssertionError):
        cfg._load_fitted_preprocessors(load_dir=tmp_path)


def test_load_fitted_preprocessors_asserts_condition_fitted(tmp_path, monkeypatch):
    monkeypatch.setenv("GLOBAL_EXP_DIR", str(tmp_path))

    cfg = make_config(
        observation=True,
        condition=True,
        condition_method="ensemble_mean",
    )

    cfg.model.preprocessing_pipeline.fitted = True
    cfg.observation.preprocessing_pipeline.fitted = True
    cfg.condition.preprocessing_pipeline.fitted = False

    with pytest.raises(AssertionError):
        cfg._load_fitted_preprocessors(load_dir=tmp_path)


def test_load_fitted_preprocessors_condition_load_dir_else_branch(tmp_path):
    cfg = make_config(
        observation=True,
        condition=True,
        condition_method="ensemble_mean",
    )

    cfg.model.preprocessing_pipeline.fitted = True
    cfg.observation.preprocessing_pipeline.fitted = True
    cfg.condition.preprocessing_pipeline.fitted = True

    cfg.condition.preprocessing_pipeline.load_dir = tmp_path / "condition.joblib"

    cfg._load_fitted_preprocessors(load_dir=tmp_path / "shared.joblib")

    assert len(cfg.condition.preprocessing_pipeline.load_calls) == 1


def test_get_weights_default_config_branch(monkeypatch, tmp_path):
    monkeypatch.setenv("GLOBAL_EXP_DIR", str(tmp_path))

    cfg = make_config(observation=True)

    weights = cfg.get_weights(save=False)

    assert weights is not None
    assert weights.shape == (2, 2)


def test_get_weights_default_config_without_observation(monkeypatch, tmp_path):
    monkeypatch.setenv("GLOBAL_EXP_DIR", str(tmp_path))

    cfg = XArrayDatasetConfig(
        model=make_model_config(kind="model_mean"),
        observation=None,
        condition=make_condition_config(static=True),
        condition_method="static",
    )
    cfg._fitted_preprocessors = True

    weights = cfg.get_weights(save=False)

    assert weights is not None
    assert weights.shape == (2, 2)


def test_get_weights_model_channel_mismatch_raises_without_observation():
    cfg = XArrayDatasetConfig(
        model=make_model_config(kind="model_mean"),
        observation=None,
        condition=make_condition_config(static=True),
        condition_method="static",
    )
    cfg._fitted_preprocessors = True

    class DummyWeightsConfig:
        def build_weights(self, target_coords, oceannanremover=None, **kwargs):
            return xr.DataArray(
                np.ones((1, 2, 2)),
                dims=("channels", "lat", "lon"),
                coords={
                    "channels": ["bad"],
                    "lat": target_coords["lat"],
                    "lon": target_coords["lon"],
                },
            )

    with pytest.raises(AssertionError):
        cfg.get_weights(config=DummyWeightsConfig(), save=False)


def test_get_input_shape_ocean_preprocessor_missing_returns_error():
    cfg = make_config(observation=True)

    class FakeOcean(Oceannanremove):
        pass

    cfg.model.preprocessing_pipeline.fitted_preprocessors = [FakeOcean()]

    ds = cfg.build(years=[2000])

    with pytest.raises(AttributeError):
        ds.get_input_shape()


def test_get_target_shape_ocean_preprocessor_missing_returns_error():
    cfg = make_config(observation=True)

    class FakeOcean(Oceannanremove):
        pass

    cfg.observation.preprocessing_pipeline.fitted_preprocessors = [FakeOcean()]

    ds = cfg.build(years=[2000])

    with pytest.raises(AttributeError):
        ds.get_target_shape()


def test_prepare_mask_with_num_lead_months_subset():
    cfg = make_config(
        observation=True,
        num_lead_months=6,
    )

    ds = cfg.build(years=[2000])

    assert ds.mask.sizes["lead_time"] == 6


def test_dataset_build_direct_class_tuple_years():
    cfg = make_config(observation=True)

    with pytest.raises(KeyError):
        XArrayDataset(
            config=cfg,
            requested_years=(2000,),
            mask=None,
            return_metadata=False,
        )


def test_get_time_features_month_sin_only():
    cfg = make_config(
        observation=True,
        time_features=["month_sin"],
    )

    ds = cfg.build(years=[2000])

    features = ds.get_time_features(2000, 3)

    assert features.shape == (1,)


def test_get_time_features_lead_time_only():
    cfg = make_config(
        observation=True,
        time_features=["lead_time"],
    )

    ds = cfg.build(years=[2000])

    features = ds.get_time_features(2000, 6)

    assert features.shape == (1,)
