import pytest
import numpy as np

from cccma_ppp.data_modules.dataset.config_abc import (
    DatasetConfigABC,
    lead_months_config,
)


class DummyPipeline:
    def set_name(self, name):
        self.name = name

    pass


class DummyModel:
    year_range = [2000]
    info = type(
        "Info", (), {"sizes": {"lead_time": 12}, "coords": {"lat": [0], "lon": [0]}}
    )()

    def __init__(
        self,
        paths=None,
        names=None,
        ensemble_list=None,
    ):
        self.paths = paths or ["a"]
        self.names = names or ["x"]
        self.preprocessing_pipeline = DummyPipeline()
        self.ensemble_list = ensemble_list
        self.concat_dim = "channels"
        self.file_type = "nc"
        self.ensemble_mean = False
        self.rename_dict = None


class DummyCondition(DummyModel):
    year_range = [2000]
    info = type(
        "Info", (), {"sizes": {"lead_time": 12}, "coords": {"lat": [0], "lon": [0]}}
    )()

    pass


class DummyDatasetConfig(DatasetConfigABC):
    @property
    def num_input_lead_months(self):
        return 12

    def _model_as_condition(self):
        cond = DummyCondition(
            paths=self.model.paths,
            names=self.model.names,
            ensemble_list=self.model.ensemble_list,
        )
        cond.ensemble_mean = self.condition_method == "ensemble_mean"
        return cond

    def __init__(
        self,
        model=None,
        condition=None,
        condition_method=None,
        time_features=None,
        lead_months=None,
    ):
        self.model = model
        self.condition = condition
        self.condition_method = condition_method
        self.time_features = time_features
        self.lead_months = lead_months
        self._effective_condition = None

        super().__init__()

    def _check_model(self):
        return self

    def _check_condition(self):
        return self

    @property
    def ds_operator(self):
        return "x"

    def build_dataset(self):
        return "dataset"


def test_lead_months_requires_end():
    with pytest.raises(ValueError):
        lead_months_config()


def test_lead_months_list_valid():
    cfg = lead_months_config(list_months=[1, 2, 3])

    assert cfg.list_months == [1, 2, 3]


def test_build_lead_months_from_list():
    cfg = lead_months_config(list_months=[1, 5, 7])

    result = cfg.build_lead_months()

    assert result == [1, 5, 7]


def test_build_lead_months_from_range():
    cfg = lead_months_config(start=1, end=4)

    result = cfg.build_lead_months()

    assert np.array_equal(result, np.array([1, 2, 3, 4]))


def test_build_lead_months_single():
    cfg = lead_months_config(start=3, end=3)

    result = cfg.build_lead_months()

    assert np.array_equal(result, np.array([3]))


def test_requires_model_or_condition():
    with pytest.raises(ValueError):
        DummyDatasetConfig()


def test_model_only_valid():
    cfg = DummyDatasetConfig(model=DummyModel())

    assert cfg.model is not None


def test_condition_only_valid():
    cfg = DummyDatasetConfig(condition=DummyCondition())

    assert cfg.condition is not None


def test_model_and_condition_valid():
    cfg = DummyDatasetConfig(
        model=DummyModel(),
        condition=DummyCondition(),
    )

    assert cfg is not None


def test_valid_condition_method_static():
    cfg = DummyDatasetConfig(
        model=DummyModel(),
        condition_method="static",
    )

    assert cfg.condition_method == "static"


def test_valid_condition_method_same_member():
    cfg = DummyDatasetConfig(
        model=DummyModel(),
        condition_method="same_member",
    )

    assert cfg.condition_method == "same_member"


def test_valid_condition_method_cross_ensemble():
    cfg = DummyDatasetConfig(
        model=DummyModel(),
        condition_method="cross_ensemble",
    )

    assert cfg.condition_method == "cross_ensemble"


def test_valid_condition_method_ensemble_mean():
    cfg = DummyDatasetConfig(
        model=DummyModel(),
        condition_method="ensemble_mean",
    )

    assert cfg.condition_method == "ensemble_mean"


def test_invalid_condition_method():
    with pytest.raises(ValueError):
        DummyDatasetConfig(
            model=DummyModel(),
            condition_method="bad",
        )


def test_none_condition_method_valid():
    cfg = DummyDatasetConfig(
        model=DummyModel(),
        condition_method=None,
    )

    assert cfg.condition_method is None


def test_valid_time_features():
    cfg = DummyDatasetConfig(
        model=DummyModel(),
        time_features=["year", "lead_time"],
    )

    assert cfg.time_features == ["year", "lead_time"]


def test_valid_all_time_features():
    cfg = DummyDatasetConfig(
        model=DummyModel(),
        time_features=[
            "year",
            "lead_time",
            "month_sin",
            "month_cos",
        ],
    )

    assert len(cfg.time_features) == 4


def test_invalid_time_feature():
    with pytest.raises(ValueError):
        DummyDatasetConfig(
            model=DummyModel(),
            time_features=["bad_feature"],
        )


def test_mixed_valid_invalid_time_features():
    with pytest.raises(ValueError):
        DummyDatasetConfig(
            model=DummyModel(),
            time_features=["year", "bad"],
        )


def test_none_time_features():
    cfg = DummyDatasetConfig(
        model=DummyModel(),
        time_features=None,
    )

    assert cfg.time_features is None


def test_resolve_lead_months_none():
    cfg = DummyDatasetConfig(
        model=DummyModel(),
        lead_months=None,
    )

    assert np.array_equal(cfg.lead_months, np.arange(1, cfg.num_input_lead_months + 1))


def test_resolve_lead_months_object():
    cfg = DummyDatasetConfig(
        model=DummyModel(),
        lead_months=lead_months_config(start=1, end=3),
    )

    assert np.array_equal(cfg.lead_months, np.array([1, 2, 3]))


def test_using_model_as_condition_false_no_condition():
    cfg = DummyDatasetConfig(
        model=DummyModel(),
        condition=None,
        condition_method=None,
    )

    assert cfg._using_model_data_as_condition is False


def test_using_model_as_condition_true_ensemble_mean():
    cfg = DummyDatasetConfig(
        model=DummyModel(),
        condition=None,
        condition_method="ensemble_mean",
    )

    assert cfg._using_model_data_as_condition is True


def test_using_model_as_condition_true_cross_ensemble():
    cfg = DummyDatasetConfig(
        model=DummyModel(),
        condition=None,
        condition_method="cross_ensemble",
    )

    assert cfg._using_model_data_as_condition is True


def test_using_model_as_condition_true_same_member():
    cfg = DummyDatasetConfig(
        model=DummyModel(),
        condition=None,
        condition_method="same_member",
    )

    assert cfg._using_model_data_as_condition is True


def test_using_model_as_condition_same_paths():
    model = DummyModel(
        paths=["a"],
        names=["x"],
        ensemble_list=[1],
    )

    condition = DummyCondition(
        paths=["a"],
        names=["x"],
        ensemble_list=[1],
    )

    cfg = DummyDatasetConfig(
        model=model,
        condition=condition,
    )

    assert cfg._using_model_data_as_condition is True


def test_using_model_as_condition_different_paths():
    model = DummyModel(paths=["a"])

    condition = DummyCondition(paths=["b"])

    cfg = DummyDatasetConfig(
        model=model,
        condition=condition,
    )

    assert cfg._using_model_data_as_condition is False


def test_using_model_as_condition_different_names():
    model = DummyModel(names=["x"])

    condition = DummyCondition(names=["y"])

    cfg = DummyDatasetConfig(
        model=model,
        condition=condition,
    )

    assert cfg._using_model_data_as_condition is False


def test_using_model_as_condition_different_ensemble_list():
    model = DummyModel(ensemble_list=[1])

    condition = DummyCondition(ensemble_list=[2])

    cfg = DummyDatasetConfig(
        model=model,
        condition=condition,
    )

    assert cfg._using_model_data_as_condition is False


def test_effective_condition_property_none():
    cfg = DummyDatasetConfig(model=DummyModel())

    assert cfg.effective_condition is None


def test_resolve_condition_with_condition():
    condition = DummyCondition()

    cfg = DummyDatasetConfig(
        model=DummyModel(),
        condition=condition,
    )

    cfg._resolve_condition()

    assert cfg.effective_condition == condition


def test_resolve_condition_none():
    cfg = DummyDatasetConfig(
        model=DummyModel(),
        condition=None,
        condition_method=None,
    )

    cfg._resolve_condition()

    assert cfg.effective_condition is None


def test_ds_operator_property():
    cfg = DummyDatasetConfig(model=DummyModel())

    assert cfg.ds_operator == "x"


def test_invalid_lead_months_exceeds_available():
    with pytest.raises(ValueError):
        DummyDatasetConfig(
            model=DummyModel(),
            lead_months=lead_months_config(start=1, end=99),
        )


def test_time_features_empty_list():
    cfg = DummyDatasetConfig(
        model=DummyModel(),
        time_features=[],
    )

    assert cfg.time_features == []


def test_condition_method_none_with_condition():
    cfg = DummyDatasetConfig(
        model=DummyModel(),
        condition=DummyCondition(),
        condition_method=None,
    )

    assert cfg.condition is not None


def test_model_vs_condition_same_year_range():
    model = DummyModel()
    condition = DummyCondition()

    model.year_range = [2000, 2001]
    condition.year_range = [1999, 2000, 2001]

    cfg = DummyDatasetConfig(
        model=model,
        condition=condition,
    )

    assert cfg.condition is condition


def test_model_vs_condition_invalid_year_range():
    model = DummyModel()
    condition = DummyCondition()

    model.year_range = [2000, 2001]
    condition.year_range = [1999]

    cfg = DummyDatasetConfig(
        model=model,
        condition=condition,
    )

    assert cfg.condition is condition


def test_condition_method_ensemble_mean_sets_flag():
    cfg = DummyDatasetConfig(
        model=DummyModel(),
        condition_method="ensemble_mean",
    )

    assert cfg.effective_condition.ensemble_mean is True


def test_condition_method_same_member_sets_flag():
    cfg = DummyDatasetConfig(
        model=DummyModel(),
        condition_method="same_member",
    )

    assert cfg.effective_condition.ensemble_mean is False


def test_num_input_lead_months_property():
    cfg = DummyDatasetConfig(model=DummyModel())

    assert cfg.num_input_lead_months == 12


def test_default_lead_months_matches_num_input():
    cfg = DummyDatasetConfig(model=DummyModel())

    assert np.array_equal(
        cfg.lead_months,
        np.arange(1, cfg.num_input_lead_months + 1),
    )


def test_condition_method_cross_ensemble_with_condition():
    cfg = DummyDatasetConfig(
        model=DummyModel(),
        condition=DummyCondition(),
        condition_method="cross_ensemble",
    )

    assert cfg.condition_method == "cross_ensemble"


def test_effective_condition_returns_condition():
    condition = DummyCondition()

    cfg = DummyDatasetConfig(
        model=DummyModel(),
        condition=condition,
    )

    assert cfg.effective_condition is condition


def test_effective_condition_model_generated():
    cfg = DummyDatasetConfig(
        model=DummyModel(),
        condition=None,
        condition_method="ensemble_mean",
    )

    assert cfg.effective_condition is not None
    assert cfg.effective_condition.names == cfg.model.names


def test_lead_months_numpy_array():
    cfg = DummyDatasetConfig(
        model=DummyModel(),
        lead_months=lead_months_config(start=1, end=3),
    )


def test_lead_months_config_returns_explicit_list():
    cfg = lead_months_config(list_months=[1, 3, 5])

    result = cfg.build_lead_months()

    assert result == [1, 3, 5]


def test_using_model_data_as_condition_false_when_model_none():
    cfg = DummyDatasetConfig(
        model=None,
        condition=DummyCondition(),
    )

    assert cfg._using_model_data_as_condition is False


def test_using_model_data_as_condition_false_different_paths():
    cfg = DummyDatasetConfig(
        model=DummyModel(paths=["a"]),
        condition=DummyCondition(paths=["b"]),
    )

    assert cfg._using_model_data_as_condition is False


def test_using_model_data_as_condition_false_different_names():
    cfg = DummyDatasetConfig(
        model=DummyModel(names=["x"]),
        condition=DummyCondition(names=["y"]),
    )

    assert cfg._using_model_data_as_condition is False


def test_using_model_data_as_condition_false_different_ensemble_lists():
    cfg = DummyDatasetConfig(
        model=DummyModel(ensemble_list=[1]),
        condition=DummyCondition(ensemble_list=[2]),
    )

    assert cfg._using_model_data_as_condition is False


def test_effective_condition_none_branch():
    cfg = DummyDatasetConfig(
        model=DummyModel(),
        condition=None,
        condition_method=None,
    )

    assert cfg.effective_condition is None


def test_model_as_condition_returns_modeldataconfig():
    cfg = DummyDatasetConfig(
        model=DummyModel(),
        condition_method="ensemble_mean",
    )

    result = cfg._model_as_condition()

    assert isinstance(result, DummyCondition)


def test_model_as_condition_ensemble_mean_false():
    cfg = DummyDatasetConfig(
        model=DummyModel(),
        condition_method="same_member",
    )

    result = cfg._model_as_condition()

    assert result.ensemble_mean is False


def test_model_as_condition_ensemble_mean_true():
    cfg = DummyDatasetConfig(
        model=DummyModel(),
        condition_method="ensemble_mean",
    )

    result = cfg._model_as_condition()

    assert result.ensemble_mean is True


def test_model_vs_condition_static_skips_year_validation():
    model = DummyModel()
    condition = DummyCondition()

    model.year_range = [2000]
    condition.year_range = [1990]

    cfg = DummyDatasetConfig(
        model=model,
        condition=condition,
        condition_method="static",
    )

    assert cfg.condition_method == "static"


def test_check_condition_method_none_valid():
    cfg = DummyDatasetConfig(
        model=DummyModel(),
        condition_method=None,
    )

    assert cfg.condition_method is None


def test_check_time_features_none_valid():
    cfg = DummyDatasetConfig(
        model=DummyModel(),
        time_features=None,
    )

    assert cfg.time_features is None


def test_check_time_features_all_valid():
    features = [
        "year",
        "lead_time",
        "month_sin",
        "month_cos",
    ]

    cfg = DummyDatasetConfig(
        model=DummyModel(),
        time_features=features,
    )

    assert cfg.time_features == features


def test_resolve_condition_with_explicit_condition():
    condition = DummyCondition()

    cfg = DummyDatasetConfig(
        model=DummyModel(),
        condition=condition,
        condition_method="ensemble_mean",
    )

    assert cfg.effective_condition is condition


def test_required_input_source_condition_only():
    cfg = DummyDatasetConfig(
        model=None,
        condition=DummyCondition(),
    )

    assert cfg.condition is not None


def test_required_input_source_model_only():
    cfg = DummyDatasetConfig(
        model=DummyModel(),
        condition=None,
    )

    assert cfg.model is not None




def test_lead_months_config_range_build():
    cfg = lead_months_config(start=2, end=5)

    result = cfg.build_lead_months()

    assert np.array_equal(result, np.array([2, 3, 4, 5]))


def test_lead_months_config_list_priority():
    cfg = lead_months_config(
        list_months=[1, 4, 7],
        start=1,
        end=12,
    )

    result = cfg.build_lead_months()

    assert result == [1, 4, 7]


def test_lead_months_config_missing_everything():
    with pytest.raises(ValueError):
        lead_months_config()


def test_invalid_condition_method_raises():
    cfg = DummyDatasetConfig.__new__(DummyDatasetConfig)

    cfg.condition_method = "bad_method"

    with pytest.raises(ValueError):
        cfg._check_condition_method()


def test_valid_condition_methods():
    for method in [
        "ensemble_mean",
        "cross_ensemble",
        "same_member",
        "static",
    ]:
        cfg = DummyDatasetConfig.__new__(DummyDatasetConfig)

        cfg.condition_method = method

        cfg._check_condition_method()


def test_invalid_time_feature_raises():
    cfg = DummyDatasetConfig.__new__(DummyDatasetConfig)

    cfg.time_features = ["bad_feature"]

    with pytest.raises(ValueError):
        cfg._check_time_features()


def test_valid_time_features_all():
    cfg = DummyDatasetConfig.__new__(DummyDatasetConfig)

    cfg.time_features = [
        "year",
        "lead_time",
        "month_sin",
        "month_cos",
    ]

    cfg._check_time_features()


def test_time_features_none_valid():
    cfg = DummyDatasetConfig.__new__(DummyDatasetConfig)

    cfg.time_features = None

    cfg._check_time_features()


def test_required_input_source_missing():
    cfg = DummyDatasetConfig.__new__(DummyDatasetConfig)

    cfg.model = None
    cfg.condition = None

    with pytest.raises(ValueError):
        cfg._check_required_input_source()


def test_using_model_data_as_condition_same_paths():
    cfg = DummyDatasetConfig(
        model=DummyModel(),
        condition=DummyCondition(),
    )

    cfg.condition.paths = cfg.model.paths
    cfg.condition.names = cfg.model.names
    cfg.condition.ensemble_list = cfg.model.ensemble_list

    assert cfg._using_model_data_as_condition is True


def test_using_model_data_as_condition_false():
    cfg = DummyDatasetConfig(
        model=DummyModel(),
        condition=DummyCondition(),
    )

    cfg.condition.paths = ["different"]

    assert cfg._using_model_data_as_condition is False


def test_using_model_data_as_condition_condition_none_same_member():
    cfg = DummyDatasetConfig(
        model=DummyModel(),
        condition=None,
        condition_method="same_member",
    )

    assert cfg._using_model_data_as_condition is True


def test_using_model_data_as_condition_condition_none_static():
    cfg = DummyDatasetConfig(
        model=DummyModel(),
        condition=None,
        condition_method="static",
    )

    assert cfg._using_model_data_as_condition is False


def test_resolve_condition_real_condition():
    cond = DummyCondition()

    cfg = DummyDatasetConfig(
        model=DummyModel(),
        condition=cond,
    )

    assert cfg.effective_condition is cond


def test_resolve_condition_model_as_condition():
    cfg = DummyDatasetConfig(
        model=DummyModel(),
        condition=None,
        condition_method="ensemble_mean",
    )

    assert cfg.effective_condition is not None


def test_same_member_with_ensemble_mean_raises():
    model = DummyModel()
    model.ensemble_mean = True

    with pytest.raises(ValueError):
        DummyDatasetConfig(
            model=model,
            condition_method="same_member",
        )


def test_lead_months_exceed_max_raises():
    with pytest.raises(ValueError):
        DummyDatasetConfig(
            model=DummyModel(),
            lead_months=lead_months_config(),
            list_months=list(np.arange(1, 50)),
        )


def test_lead_months_valid_exact_boundary():
    model = DummyModel()

    cfg = DummyDatasetConfig(
        model=model,
        lead_months=lead_months_config(
            list_months=list(
                np.arange(
                    1,
                    model.info.sizes["lead_time"] + 1,
                )
            ),
        ),
    )

    assert max(cfg.lead_months) == model.info.sizes["lead_time"]


def test_model_vs_condition_year_subset_error():
    model = DummyModel()
    condition = DummyCondition()

    condition.paths = ["different"]

    model.year_range = [2000, 2001]
    condition.year_range = [2000]

    with pytest.raises(ValueError):
        DummyDatasetConfig(
            model=model,
            condition=condition,
            condition_method="same_member",
        )


def test_model_vs_condition_lead_time_error():
    model = DummyModel()
    condition = DummyCondition()

    condition.paths = ["different"]

    model.info.sizes["lead_time"] = 12
    condition.info.sizes["lead_time"] = 6

    with pytest.raises(ValueError):
        DummyDatasetConfig(
            model=model,
            condition=condition,
            condition_method="same_member",
        )


def test_model_vs_condition_lat_mismatch():
    model = DummyModel()
    condition = DummyCondition()

    condition.paths = ["different"]

    class Coord:
        def __init__(self, value):
            self.value = value

        def equals(self, other):
            return self.value == other.value

    model.info.coords["lat"] = Coord("a")
    condition.info.coords["lat"] = Coord("b")

    condition.info.sizes["lead_time"] = model.info.sizes["lead_time"]

    with pytest.raises(TypeError):
        cfg = DummyDatasetConfig(
            model=model,
            condition=condition,
            condition_method="same_member",
        )

        cfg.observation = object()

        cfg._check_model_vs_condition()


def test_model_vs_condition_lon_mismatch():
    model = DummyModel()
    condition = DummyCondition()

    condition.paths = ["different"]

    class Coord:
        def __init__(self, value):
            self.value = value

        def equals(self, other):
            return self.value == other.value

    model.info.coords["lon"] = Coord("a")
    condition.info.coords["lon"] = Coord("b")

    condition.info.sizes["lead_time"] = model.info.sizes["lead_time"]

    with pytest.raises(TypeError):
        cfg = DummyDatasetConfig(
            model=model,
            condition=condition,
            condition_method="same_member",
        )

        cfg.observation = object()

        cfg._check_model_vs_condition()


def test_effective_condition_property_returns_internal():
    cfg = DummyDatasetConfig(
        model=DummyModel(),
        condition_method="ensemble_mean",
    )

    assert cfg.effective_condition == cfg._effective_condition


def test_model_vs_condition_skips_when_static():
    model = DummyModel()
    condition = DummyCondition()

    condition.paths = ["different"]

    model.year_range = [2000, 2001]
    condition.year_range = [1990]

    cfg = DummyDatasetConfig(
        model=model,
        condition=condition,
        condition_method="static",
    )

    cfg._check_model_vs_condition()


def test_model_vs_condition_skips_when_using_model_condition():
    model = DummyModel()
    condition = DummyCondition()

    condition.paths = model.paths
    condition.names = model.names
    condition.ensemble_list = model.ensemble_list

    cfg = DummyDatasetConfig(
        model=model,
        condition=condition,
        condition_method="same_member",
    )

    cfg._check_model_vs_condition()


def test_resolve_condition_sets_none():
    cfg = DummyDatasetConfig(
        model=DummyModel(),
        condition=None,
        condition_method=None,
    )

    assert cfg.effective_condition is None


def test_using_model_data_as_condition_false_static():
    cfg = DummyDatasetConfig(
        model=DummyModel(),
        condition=None,
        condition_method="static",
    )

    assert cfg._using_model_data_as_condition is False


def test_using_model_data_as_condition_true_cross_ensemble():
    cfg = DummyDatasetConfig(
        model=DummyModel(),
        condition=None,
        condition_method="cross_ensemble",
    )

    assert cfg._using_model_data_as_condition is True
