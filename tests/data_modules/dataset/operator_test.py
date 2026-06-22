import pytest
import numpy as np
import xarray as xr
from unittest.mock import patch

from cccma_ppp.data_modules.dataset.operator import (
    DatasetOperator,
    _get_time_features,
)
from cccma_ppp.preprocessing.preprocessing_ABC import PreprocessModuleABC


class DummyFlatten:
    pass


class DummyPipeline:
    def __init__(self):
        self.pipeline = [("scale", object())]
        self.fitted_preprocessors = []
        self.added = None

    def add_fitted_preprocessor(self, preprocessor, index=0):
        self.added = (preprocessor, index)

    def get_preprocessors(self, name):
        return "flattener_obj"


class DummyInfo:
    def __init__(self, ensembles=None):
        self.coords = {
            "ensembles": ensembles,
            "lat": [0, 1],
            "lon": [10, 20],
        }


class DummyDataConfig:
    def __init__(self, names=None, ensembles=None):
        self.names = names or ["var"]

        self.info = DummyInfo(ensembles)

        self.preprocessing_pipeline = DummyPipeline()

        self.fit_called = None
        self.load_called = None

    def _fit_preprocessor_pipeline(self, **kwargs):
        self.fit_called = kwargs

    def _load_preprocessor_pipeline(self, load_dir):
        self.load_called = load_dir


class DummyDatasetConfig:
    def __init__(self):
        self.model = DummyDataConfig()
        self.observation = DummyDataConfig()
        self.effective_condition = DummyDataConfig()

        self.condition_method = "same_member"

        self._using_model_data_as_condition = False
        self._fitted_preprocessors = False

        self.lead_months = [1, 2, 3]
        self.time_features = None

        self.get_common_time = np.array([2000, 2001, 2002])


class DummyWeights:
    def __init__(self, dims=(), channels=None):
        self.dims = dims
        self.channels = xr.DataArray(
            np.array(channels or []),
            dims=("channels",),
        )


class DummyWeightsConfig:
    def build_weights(self, *args, **kwargs):
        return DummyWeights()


class DummyPreprocessor(PreprocessModuleABC):
    fitted = True

    def fit(self, *args, **kwargs):
        pass

    def transform(self, x):
        return x

    def inverse_transform(self, x):
        return x


def test_dataset_operator_init():
    cfg = DummyDatasetConfig()

    op = DatasetOperator(cfg)

    assert op.config == cfg


def test_config_observation_exists():
    cfg = DummyDatasetConfig()

    op = DatasetOperator(cfg)

    assert op.config_observation is not None


def test_config_observation_missing():
    cfg = DummyDatasetConfig()

    del cfg.observation

    op = DatasetOperator(cfg)

    assert op.config_observation is None


def test_fit_preprocessors_model_called():
    cfg = DummyDatasetConfig()

    op = DatasetOperator(cfg)

    op._fit_preprocessors([2000])

    assert cfg.model.fit_called is not None


def test_fit_preprocessors_observation_called():
    cfg = DummyDatasetConfig()

    op = DatasetOperator(cfg)

    op._fit_preprocessors([2000])

    assert cfg.observation.fit_called is not None


def test_fit_preprocessors_condition_called():
    cfg = DummyDatasetConfig()

    op = DatasetOperator(cfg)

    op._fit_preprocessors([2000])

    assert cfg.effective_condition.fit_called is not None


def test_fit_preprocessors_static_condition():
    cfg = DummyDatasetConfig()

    cfg.condition_method = "static"

    op = DatasetOperator(cfg)

    op._fit_preprocessors([2000])

    assert cfg.effective_condition.fit_called["selection"] == {}


def test_fit_preprocessors_with_ensemble_selection():
    cfg = DummyDatasetConfig()

    cfg.model.info.coords["ensembles"] = [0]

    op = DatasetOperator(cfg)

    op._fit_preprocessors([2000])

    assert "ensembles" in cfg.model.fit_called["selection"]


def test_fit_preprocessors_sets_flag():
    cfg = DummyDatasetConfig()

    op = DatasetOperator(cfg)

    op._fit_preprocessors([2000])

    assert cfg._fitted_preprocessors is True


def test_fit_preprocessors_without_model():
    cfg = DummyDatasetConfig()

    cfg.model = None

    op = DatasetOperator(cfg)

    op._fit_preprocessors([2000])

    assert cfg._fitted_preprocessors is True


def test_fit_preprocessors_without_observation():
    cfg = DummyDatasetConfig()

    cfg.observation = None

    op = DatasetOperator(cfg)

    op._fit_preprocessors([2000])

    assert cfg._fitted_preprocessors is True


def test_fit_preprocessors_without_condition():
    cfg = DummyDatasetConfig()

    cfg.effective_condition = None

    op = DatasetOperator(cfg)

    op._fit_preprocessors([2000])

    assert cfg._fitted_preprocessors is True


def test_load_fitted_preprocessors_model():
    cfg = DummyDatasetConfig()

    op = DatasetOperator(cfg)

    op._load_fitted_preprocessors("x")

    assert cfg.model.load_called == "x"


def test_load_fitted_preprocessors_observation():
    cfg = DummyDatasetConfig()

    op = DatasetOperator(cfg)

    op._load_fitted_preprocessors("x")

    assert cfg.observation.load_called == "x"


def test_load_fitted_preprocessors_condition():
    cfg = DummyDatasetConfig()

    op = DatasetOperator(cfg)

    op._load_fitted_preprocessors("x")

    assert cfg.effective_condition.load_called == "x"


def test_load_fitted_preprocessors_sets_flag():
    cfg = DummyDatasetConfig()

    op = DatasetOperator(cfg)

    op._load_fitted_preprocessors("x")

    assert cfg._fitted_preprocessors is True


def test_add_fitted_preprocessor_invalid_type():
    cfg = DummyDatasetConfig()

    op = DatasetOperator(cfg)

    with pytest.raises(TypeError):
        op._add_fitted_preprocessor(object())


def test_add_fitted_preprocessor_not_fitted():
    class BadPreprocessor(DummyPreprocessor):
        fitted = False

    cfg = DummyDatasetConfig()

    op = DatasetOperator(cfg)

    with pytest.raises(AssertionError):
        op._add_fitted_preprocessor(BadPreprocessor())


def test_add_fitted_preprocessor_model():
    cfg = DummyDatasetConfig()

    op = DatasetOperator(cfg)

    preprocessor = DummyPreprocessor()

    op._add_fitted_preprocessor(preprocessor)

    assert cfg.model.preprocessing_pipeline.added[0] == preprocessor


def test_add_fitted_preprocessor_observation():
    cfg = DummyDatasetConfig()

    op = DatasetOperator(cfg)

    preprocessor = DummyPreprocessor()

    op._add_fitted_preprocessor(preprocessor)

    assert cfg.observation.preprocessing_pipeline.added[0] == preprocessor


def test_add_fitted_preprocessor_condition():
    cfg = DummyDatasetConfig()

    op = DatasetOperator(cfg)

    preprocessor = DummyPreprocessor()

    op._add_fitted_preprocessor(preprocessor)

    assert cfg.effective_condition.preprocessing_pipeline.added[0] == preprocessor


def test_get_weights_with_config():
    cfg = DummyDatasetConfig()

    op = DatasetOperator(cfg)

    with patch(
        "cccma_ppp.preprocessing.utils_preprocessing.Flattennanremove",
        DummyFlatten,
    ):
        weights = op.get_weights(config=DummyWeightsConfig())

    assert weights is not None


def test_get_weights_model_only():
    cfg = DummyDatasetConfig()

    cfg.observation = None

    op = DatasetOperator(cfg)

    with patch(
        "cccma_ppp.preprocessing.utils_preprocessing.Flattennanremove",
        DummyFlatten,
    ):
        weights = op.get_weights(config=DummyWeightsConfig())

    assert weights is not None


def test_get_weights_no_model_or_observation():
    cfg = DummyDatasetConfig()

    cfg.model = None
    cfg.observation = None

    op = DatasetOperator(cfg)

    with pytest.raises(ValueError):
        op.get_weights(config=DummyWeightsConfig())


def test_get_weights_channels_match():
    cfg = DummyDatasetConfig()

    op = DatasetOperator(cfg)

    class Config:
        def build_weights(self, *args, **kwargs):
            return DummyWeights(
                dims=("channels",),
                channels=["var"],
            )

    with patch(
        "cccma_ppp.preprocessing.utils_preprocessing.Flattennanremove",
        DummyFlatten,
    ):
        weights = op.get_weights(config=Config())

    assert weights is not None


def test_get_weights_channels_mismatch():
    cfg = DummyDatasetConfig()

    op = DatasetOperator(cfg)

    class Config:
        def build_weights(self, *args, **kwargs):
            return DummyWeights(
                dims=("channels",),
                channels=["bad"],
            )

    with patch(
        "cccma_ppp.preprocessing.utils_preprocessing.Flattennanremove",
        DummyFlatten,
    ):
        with pytest.raises(RuntimeError):
            op.get_weights(config=Config())


def test_get_weights_with_flattennanremove():
    cfg = DummyDatasetConfig()

    cfg.observation.preprocessing_pipeline.fitted_preprocessors = [DummyFlatten()]

    op = DatasetOperator(cfg)

    captured = {}

    class Config:
        def build_weights(self, *args, **kwargs):
            captured["Flattennanremover"] = kwargs["Flattennanremover"]
            return DummyWeights()

    with patch(
        "cccma_ppp.preprocessing.utils_preprocessing.Flattennanremove",
        DummyFlatten,
    ):
        op.get_weights(config=Config())

    assert captured["Flattennanremover"] == "flattener_obj"


def test_get_input_var_metadata_model_only():
    cfg = DummyDatasetConfig()

    cfg.effective_condition = None

    op = DatasetOperator(cfg)

    metadata = op.get_input_var_metadata()

    assert metadata["variables"] == ["var"]


def test_get_input_var_metadata_model_and_condition():
    cfg = DummyDatasetConfig()

    op = DatasetOperator(cfg)

    metadata = op.get_input_var_metadata()

    assert len(metadata["variables"]) == 2


def test_get_input_var_metadata_using_model_as_condition():
    cfg = DummyDatasetConfig()

    cfg._using_model_data_as_condition = True

    op = DatasetOperator(cfg)

    metadata = op.get_input_var_metadata()

    assert len(metadata["variables"]) == 1


def test_get_target_var_metadata_observation():
    cfg = DummyDatasetConfig()

    op = DatasetOperator(cfg)

    metadata = op.get_target_var_metadata()

    assert metadata["variables"] == ["var"]


def test_get_target_var_metadata_model():
    cfg = DummyDatasetConfig()

    cfg.observation = None

    op = DatasetOperator(cfg)

    metadata = op.get_target_var_metadata()

    assert metadata["variables"] == ["var"]


def test_get_target_var_metadata_missing():
    cfg = DummyDatasetConfig()

    cfg.model = None
    cfg.observation = None

    op = DatasetOperator(cfg)

    with pytest.raises(ValueError):
        op.get_target_var_metadata()


def test_update_metadata():
    cfg = DummyDatasetConfig()

    op = DatasetOperator(cfg)

    metadata = {
        "variables": [],
        "preprocessors": [],
    }

    result = op._update_metadata_with_dataconfig_metadata(
        metadata,
        cfg.model,
    )

    assert result["variables"] == ["var"]


def test_get_time_features_none():
    cfg = DummyDatasetConfig()

    cfg.time_features = None

    result = _get_time_features(
        cfg,
        2000,
        1,
        xr.DataArray(np.random.rand(2, 2)),
    )

    assert result is None


def test_get_time_features_year():
    cfg = DummyDatasetConfig()

    cfg.time_features = ["year"]

    result = _get_time_features(
        cfg,
        2000,
        1,
        xr.DataArray(np.random.rand(2, 2)),
    )

    assert result.shape[0] == 1


def test_get_time_features_multiple():
    cfg = DummyDatasetConfig()

    cfg.time_features = [
        "year",
        "lead_time",
        "month_sin",
    ]

    result = _get_time_features(
        cfg,
        2000,
        1,
        xr.DataArray(np.random.rand(2, 2)),
    )

    assert result.shape[0] == 3


def test_get_time_features_broadcast():
    cfg = DummyDatasetConfig()

    cfg.time_features = ["year"]

    arr = xr.DataArray(np.random.rand(3, 4, 5))

    result = _get_time_features(
        cfg,
        2000,
        1,
        arr,
    )

    assert result.ndim > 1


def test_get_time_features_values():
    cfg = DummyDatasetConfig()

    cfg.time_features = [
        "month_sin",
        "month_cos",
    ]

    result = _get_time_features(
        cfg,
        2000,
        6,
        xr.DataArray(np.random.rand(2, 2)),
    )

    assert result.shape[0] == 2


def test_fit_preprocessors_condition_with_ensemble_selection():
    cfg = DummyDatasetConfig()

    cfg.effective_condition.info.coords["ensembles"] = [0]

    op = DatasetOperator(cfg)

    op._fit_preprocessors([2000])

    assert "ensembles" in cfg.effective_condition.fit_called["selection"]


def test_fit_preprocessors_observation_with_ensemble_selection():
    cfg = DummyDatasetConfig()

    cfg.observation.info.coords["ensembles"] = [0]

    op = DatasetOperator(cfg)

    op._fit_preprocessors([2000])

    assert "ensembles" in cfg.observation.fit_called["selection"]


def test_load_fitted_preprocessors_without_model():
    cfg = DummyDatasetConfig()

    cfg.model = None

    op = DatasetOperator(cfg)

    op._load_fitted_preprocessors("x")

    assert cfg._fitted_preprocessors is True


def test_load_fitted_preprocessors_without_observation():
    cfg = DummyDatasetConfig()

    cfg.observation = None

    op = DatasetOperator(cfg)

    op._load_fitted_preprocessors("x")

    assert cfg._fitted_preprocessors is True


def test_load_fitted_preprocessors_without_condition():
    cfg = DummyDatasetConfig()

    cfg.effective_condition = None

    op = DatasetOperator(cfg)

    op._load_fitted_preprocessors("x")

    assert cfg._fitted_preprocessors is True


def test_add_fitted_preprocessor_without_model():
    cfg = DummyDatasetConfig()

    cfg.model = None

    op = DatasetOperator(cfg)

    p = DummyPreprocessor()

    op._add_fitted_preprocessor(p)

    assert cfg.observation.preprocessing_pipeline.added[0] == p


def test_add_fitted_preprocessor_without_observation():
    cfg = DummyDatasetConfig()

    cfg.observation = None

    op = DatasetOperator(cfg)

    p = DummyPreprocessor()

    op._add_fitted_preprocessor(p)

    assert cfg.model.preprocessing_pipeline.added[0] == p


def test_add_fitted_preprocessor_without_condition():
    cfg = DummyDatasetConfig()

    cfg.effective_condition = None

    op = DatasetOperator(cfg)

    p = DummyPreprocessor()

    op._add_fitted_preprocessor(p)

    assert cfg.model.preprocessing_pipeline.added[0] == p


def test_get_weights_model_channel_mismatch():
    cfg = DummyDatasetConfig()

    cfg.observation = None

    op = DatasetOperator(cfg)

    class Config:
        def build_weights(self, *args, **kwargs):
            return DummyWeights(
                dims=("channels",),
                channels=["bad"],
            )

    with patch(
        "cccma_ppp.preprocessing.utils_preprocessing.Flattennanremove",
        DummyFlatten,
    ):
        with pytest.raises(RuntimeError):
            op.get_weights(config=Config())


def test_get_weights_without_flattennanremove():
    cfg = DummyDatasetConfig()

    captured = {}

    class Config:
        def build_weights(self, *args, **kwargs):
            captured["Flattennanremover"] = kwargs["Flattennanremover"]
            return DummyWeights()

    op = DatasetOperator(cfg)

    with patch(
        "cccma_ppp.preprocessing.utils_preprocessing.Flattennanremove",
        DummyFlatten,
    ):
        op.get_weights(config=Config())

    assert captured["Flattennanremover"] is None


def test_update_metadata_multiple_variables():
    cfg = DummyDatasetConfig()

    cfg.model.names = ["a", "b"]

    op = DatasetOperator(cfg)

    metadata = {
        "variables": [],
        "preprocessors": [],
    }

    result = op._update_metadata_with_dataconfig_metadata(
        metadata,
        cfg.model,
    )

    assert result["variables"] == ["a", "b"]


def test_get_time_features_lead_time_only():
    cfg = DummyDatasetConfig()

    cfg.time_features = ["lead_time"]

    result = _get_time_features(
        cfg,
        2000,
        3,
        xr.DataArray(np.random.rand(2, 2)),
    )

    assert result.shape[0] == 1


def test_get_time_features_month_cos_only():
    cfg = DummyDatasetConfig()

    cfg.time_features = ["month_cos"]

    result = _get_time_features(
        cfg,
        2000,
        6,
        xr.DataArray(np.random.rand(2, 2)),
    )

    assert result.shape[0] == 1


def test_get_time_features_full_feature_set():
    cfg = DummyDatasetConfig()

    cfg.time_features = [
        "year",
        "lead_time",
        "month_sin",
        "month_cos",
    ]

    result = _get_time_features(
        cfg,
        2000,
        6,
        xr.DataArray(np.random.rand(2, 2)),
    )

    assert result.shape[0] == 4


def test_get_time_features_no_broadcast():
    cfg = DummyDatasetConfig()

    cfg.time_features = ["year"]

    arr = xr.DataArray(np.random.rand(2))

    result = _get_time_features(
        cfg,
        2000,
        1,
        arr,
    )

    assert result.ndim == 1
