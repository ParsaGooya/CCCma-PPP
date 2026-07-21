import pytest
from cccma_ppp.data_modules.dataset.dataset_abc import AddedTimeFeatures
import numpy as np
import xarray as xr
from unittest.mock import patch

from cccma_ppp.data_modules.dataset.operator import (
    DatasetOperator,
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
            "lead_time": np.array([1, 2, 3]),
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

    def fit_preprocessor_pipeline(
        self,
        selection=None,
        mask=False,
        save=False,
        save_path=None,
        save_name=None,
    ):
        self.fit_called = {
            "selection": selection,
            "mask": mask,
            "save": save,
            "save_path": save_path,
            "save_name": save_name,
        }
        return self

    def load_preprocessor_pipeline(self, load_dir=None):
        self.load_called = load_dir
        return self


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


@pytest.mark.pruned
def test_config_observation_exists():
    cfg = DummyDatasetConfig()

    op = DatasetOperator(cfg)

    assert op.config_observation is not None


def test_config_observation_missing():
    cfg = DummyDatasetConfig()

    del cfg.observation

    op = DatasetOperator(cfg)

    assert op.config_observation is None


@pytest.mark.pruned
def test_fit_preprocessors_model_called():
    cfg = DummyDatasetConfig()

    op = DatasetOperator(cfg)

    op.fit_preprocessors([2000])

    assert cfg.model.fit_called is not None


@pytest.mark.pruned
def test_fit_preprocessors_observation_called():
    cfg = DummyDatasetConfig()

    op = DatasetOperator(cfg)

    op.fit_preprocessors([2000])

    assert cfg.observation.fit_called is not None


@pytest.mark.pruned
def test_fit_preprocessors_condition_called():
    cfg = DummyDatasetConfig()

    op = DatasetOperator(cfg)

    op.fit_preprocessors([2000])

    assert cfg.effective_condition.fit_called is not None


def test_fit_preprocessors_static_condition():
    cfg = DummyDatasetConfig()

    cfg.condition_method = "static"

    op = DatasetOperator(cfg)

    op.fit_preprocessors([2000])

    assert cfg.effective_condition.fit_called["selection"] == {}


def test_fit_preprocessors_with_ensemble_selection():
    cfg = DummyDatasetConfig()

    cfg.model.info.coords["ensembles"] = [0]

    op = DatasetOperator(cfg)

    op.fit_preprocessors([2000])

    assert "ensembles" in cfg.model.fit_called["selection"]


@pytest.mark.pruned
def test_fit_preprocessors_sets_flag():
    cfg = DummyDatasetConfig()

    op = DatasetOperator(cfg)

    op.fit_preprocessors([2000])

    assert cfg._fitted_preprocessors is True


def test_fit_preprocessors_without_model():
    cfg = DummyDatasetConfig()

    cfg.model = None

    op = DatasetOperator(cfg)

    op.fit_preprocessors([2000])

    assert cfg._fitted_preprocessors is True


def test_fit_preprocessors_without_observation():
    cfg = DummyDatasetConfig()

    cfg.observation = None

    op = DatasetOperator(cfg)

    op.fit_preprocessors([2000])

    assert cfg._fitted_preprocessors is True


@pytest.mark.pruned
def test_fit_preprocessors_without_condition():
    cfg = DummyDatasetConfig()

    cfg.effective_condition = None

    op = DatasetOperator(cfg)

    op.fit_preprocessors([2000])

    assert cfg._fitted_preprocessors is True


@pytest.mark.pruned
def test_load_fitted_preprocessors_model():
    cfg = DummyDatasetConfig()

    op = DatasetOperator(cfg)

    op.load_fitted_preprocessors("x")

    assert cfg.model.load_called == "x"


@pytest.mark.pruned
def test_load_fitted_preprocessors_observation():
    cfg = DummyDatasetConfig()

    op = DatasetOperator(cfg)

    op.load_fitted_preprocessors("x")

    assert cfg.observation.load_called == "x"


@pytest.mark.pruned
def test_load_fitted_preprocessors_condition():
    cfg = DummyDatasetConfig()

    op = DatasetOperator(cfg)

    op.load_fitted_preprocessors("x")

    assert cfg.effective_condition.load_called == "x"


@pytest.mark.pruned
def test_load_fitted_preprocessors_sets_flag():
    cfg = DummyDatasetConfig()

    op = DatasetOperator(cfg)

    op.load_fitted_preprocessors("x")

    assert cfg._fitted_preprocessors is True


def test_add_fitted_preprocessor_invalid_type():
    cfg = DummyDatasetConfig()

    op = DatasetOperator(cfg)

    with pytest.raises(TypeError):
        op.add_fitted_preprocessor(object())


@pytest.mark.pruned
def test_add_fitted_preprocessor_not_fitted():
    class BadPreprocessor(DummyPreprocessor):
        fitted = False

    cfg = DummyDatasetConfig()

    op = DatasetOperator(cfg)

    with pytest.raises(AssertionError):
        op.add_fitted_preprocessor(BadPreprocessor())


@pytest.mark.pruned
def test_add_fitted_preprocessor_model():
    cfg = DummyDatasetConfig()

    op = DatasetOperator(cfg)

    preprocessor = DummyPreprocessor()

    op.add_fitted_preprocessor(preprocessor)

    assert cfg.model.preprocessing_pipeline.added[0] == preprocessor


@pytest.mark.pruned
def test_add_fitted_preprocessor_observation():
    cfg = DummyDatasetConfig()

    op = DatasetOperator(cfg)

    preprocessor = DummyPreprocessor()

    op.add_fitted_preprocessor(preprocessor)

    assert cfg.observation.preprocessing_pipeline.added[0] == preprocessor


@pytest.mark.pruned
def test_add_fitted_preprocessor_condition():
    cfg = DummyDatasetConfig()

    op = DatasetOperator(cfg)

    preprocessor = DummyPreprocessor()

    op.add_fitted_preprocessor(preprocessor)

    assert cfg.effective_condition.preprocessing_pipeline.added[0] == preprocessor


@pytest.mark.pruned
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


@pytest.mark.pruned
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


@pytest.mark.pruned
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


@pytest.mark.pruned
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


def _get_time_features(config, selection, input):
    return AddedTimeFeatures(config, config.time_features)(selection, input)


def make_time_selection(
    year=2000,
    lead_time=6,
):
    return {
        "year": year,
        "lead_time": lead_time,
    }


def make_time_input(shape=(2,)):
    return xr.DataArray(
        np.ones(
            shape,
            dtype=np.float32,
        )
    )


@pytest.mark.parametrize(
    "features,expected_length",
    [
        (["year"], 1),
        (["lead_time"], 1),
        (["month_sin"], 1),
        (["month_cos"], 1),
        (
            [
                "year",
                "lead_time",
                "month_sin",
            ],
            3,
        ),
        (
            [
                "month_sin",
                "month_cos",
            ],
            2,
        ),
        (
            [
                "year",
                "lead_time",
                "month_sin",
                "month_cos",
            ],
            4,
        ),
    ],
)
def test_get_time_features_dimensions(
    features,
    expected_length,
):
    cfg = DummyDatasetConfig()
    cfg.time_features = features

    result = _get_time_features(
        cfg,
        make_time_selection(),
        make_time_input(),
    )

    assert result.shape[0] == expected_length


@pytest.mark.pruned
def test_get_time_features_year_value():
    cfg = DummyDatasetConfig()
    cfg.time_features = ["year"]

    result = _get_time_features(
        cfg,
        make_time_selection(
            year=2000,
            lead_time=1,
        ),
        make_time_input(),
    )

    assert result[0] == pytest.approx(0.5)


@pytest.mark.pruned
def test_get_time_features_lead_time_value():
    cfg = DummyDatasetConfig()
    cfg.time_features = ["lead_time"]

    result = _get_time_features(
        cfg,
        make_time_selection(
            year=2000,
            lead_time=3,
        ),
        make_time_input(),
    )

    assert result[0] == pytest.approx(1.0)


@pytest.mark.pruned
def test_get_time_features_month_values_are_finite():
    cfg = DummyDatasetConfig()
    cfg.time_features = [
        "month_sin",
        "month_cos",
    ]

    result = _get_time_features(
        cfg,
        make_time_selection(
            year=2000,
            lead_time=6,
        ),
        make_time_input(),
    )

    assert np.isfinite(result).all()


@pytest.mark.pruned
def test_get_time_features_broadcast():
    cfg = DummyDatasetConfig()
    cfg.time_features = ["year"]

    result = _get_time_features(
        cfg,
        make_time_selection(),
        make_time_input(
            shape=(3, 4, 5),
        ),
    )

    assert result.ndim > 1


@pytest.mark.pruned
def test_get_time_features_no_broadcast():
    cfg = DummyDatasetConfig()
    cfg.time_features = ["year"]

    result = _get_time_features(
        cfg,
        make_time_selection(),
        make_time_input(
            shape=(2,),
        ),
    )

    assert result.ndim == 1


@pytest.mark.parametrize(
    "selection",
    [
        {},
        {
            "year": 2000,
        },
        {
            "lead_time": 1,
        },
    ],
)
def test_get_time_features_missing_selection_keys(
    selection,
):
    cfg = DummyDatasetConfig()
    cfg.time_features = ["year"]

    with pytest.raises(
        ValueError,
        match="selection coords are not in required sample dimensions",
    ):
        _get_time_features(
            cfg,
            selection,
            make_time_input(),
        )


def test_fit_preprocessors_condition_with_ensemble_selection():
    cfg = DummyDatasetConfig()

    cfg.effective_condition.info.coords["ensembles"] = [0]

    op = DatasetOperator(cfg)

    op.fit_preprocessors([2000])

    assert "ensembles" in cfg.effective_condition.fit_called["selection"]


def test_fit_preprocessors_observation_with_ensemble_selection():
    cfg = DummyDatasetConfig()

    cfg.observation.info.coords["ensembles"] = [0]

    op = DatasetOperator(cfg)

    op.fit_preprocessors([2000])

    assert "ensembles" in cfg.observation.fit_called["selection"]


def test_load_fitted_preprocessors_without_model():
    cfg = DummyDatasetConfig()
    cfg.model = None

    op = DatasetOperator(cfg)

    op.load_fitted_preprocessors("x")

    assert cfg._fitted_preprocessors is True


def test_load_fitted_preprocessors_without_observation():
    cfg = DummyDatasetConfig()
    cfg.observation = None

    op = DatasetOperator(cfg)

    op.load_fitted_preprocessors("x")

    assert cfg._fitted_preprocessors is True


def test_load_fitted_preprocessors_without_condition():
    cfg = DummyDatasetConfig()
    cfg.effective_condition = None

    op = DatasetOperator(cfg)

    op.load_fitted_preprocessors("x")

    assert cfg._fitted_preprocessors is True


@pytest.mark.pruned
def test_add_fitted_preprocessor_without_model():
    cfg = DummyDatasetConfig()
    cfg.model = None

    op = DatasetOperator(cfg)
    preprocessor = DummyPreprocessor()

    op.add_fitted_preprocessor(preprocessor)

    assert cfg.observation.preprocessing_pipeline.added[0] is preprocessor


def test_add_fitted_preprocessor_without_observation():
    cfg = DummyDatasetConfig()
    cfg.observation = None

    op = DatasetOperator(cfg)
    preprocessor = DummyPreprocessor()

    op.add_fitted_preprocessor(preprocessor)

    assert cfg.model.preprocessing_pipeline.added[0] is preprocessor


def test_add_fitted_preprocessor_without_condition():
    cfg = DummyDatasetConfig()
    cfg.effective_condition = None

    op = DatasetOperator(cfg)
    preprocessor = DummyPreprocessor()

    op.add_fitted_preprocessor(preprocessor)

    assert cfg.model.preprocessing_pipeline.added[0] is preprocessor


@pytest.mark.pruned
def test_get_weights_model_channel_mismatch():
    cfg = DummyDatasetConfig()
    cfg.observation = None

    op = DatasetOperator(cfg)

    class Config:
        def build_weights(
            self,
            *args,
            **kwargs,
        ):
            return DummyWeights(
                dims=("channels",),
                channels=["bad"],
            )

    with patch(
        "cccma_ppp.preprocessing.utils_preprocessing.Flattennanremove",
        DummyFlatten,
    ):
        with pytest.raises(RuntimeError):
            op.get_weights(
                config=Config(),
            )


@pytest.mark.pruned
def test_get_weights_without_flattennanremove():
    cfg = DummyDatasetConfig()
    captured = {}

    class Config:
        def build_weights(
            self,
            *args,
            **kwargs,
        ):
            captured["Flattennanremover"] = kwargs["Flattennanremover"]
            return DummyWeights()

    op = DatasetOperator(cfg)

    with patch(
        "cccma_ppp.preprocessing.utils_preprocessing.Flattennanremove",
        DummyFlatten,
    ):
        op.get_weights(
            config=Config(),
        )

    assert captured["Flattennanremover"] is None


@pytest.mark.pruned
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

    assert result["variables"] == [
        "a",
        "b",
    ]