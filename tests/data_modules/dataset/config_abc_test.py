import pytest
import numpy as np

from cccma_ppp.data_modules.dataset.config_abc import (
    DatasetConfigABC,
    lead_months_config,
)


class DummyPipeline:
    pass


class DummyModel:
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
    pass


class DummyDatasetConfig(DatasetConfigABC):
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

    assert cfg.lead_months is None


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
