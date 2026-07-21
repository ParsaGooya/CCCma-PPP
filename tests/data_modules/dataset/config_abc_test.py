import abc
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import cccma_ppp.data_modules.dataset.config_abc as module
from cccma_ppp.data_modules.dataset.config_abc import (
    DatasetConfigABC,
    lead_months_config,
)


class Coordinate:
    def __init__(self, values, equality=None):
        self.values = np.asarray(values)
        self.equality = equality

    def equals(self, other):
        if self.equality is not None:
            return self.equality
        return np.array_equal(self.values, other.values)


class LeadMonthsBuilder:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def build_lead_months(self):
        self.calls += 1
        return self.result


def make_data(
    years=(2000, 2001),
    leads=(1, 2, 3),
    ensembles=("r1", "r2"),
    paths=("model.nc",),
    names=("tas",),
    ensemble_list=("r1", "r2"),
    extra_coords=None,
):
    coords = {
        "lead_time": Coordinate(leads),
    }

    if ensembles is not None:
        coords["ensembles"] = Coordinate(ensembles)

    if extra_coords:
        coords.update(extra_coords)

    return SimpleNamespace(
        year_range=list(years),
        info=SimpleNamespace(coords=coords),
        paths=list(paths),
        names=list(names),
        ensemble_list=(list(ensemble_list) if ensemble_list is not None else None),
        preprocessing_pipeline="pipeline",
        concat_dim="time",
        file_type="netcdf",
        rename_dict={"old": "new"},
    )


def make_condition(**kwargs):
    kwargs.setdefault("paths", ("condition.nc",))
    return make_data(**kwargs)


class ConcreteConfig(DatasetConfigABC):
    def __init__(
        self,
        model=None,
        condition=None,
        condition_method=None,
        time_features=None,
        lead_months=None,
        num_input_lead_months=3,
        observation=None,
    ):
        self.model = model
        self.condition = condition
        self.condition_method = condition_method
        self.time_features = time_features
        self.lead_months = lead_months
        self._num_input_lead_months = num_input_lead_months
        self.observation = observation
        self.model_checked = False
        self.condition_checked = False
        super().__init__()

    def _check_model(self):
        self.model_checked = True
        return self

    def _check_condition(self):
        self.condition_checked = True
        return self

    @property
    def ds_operator(self):
        return "operator"

    @property
    def num_input_lead_months(self):
        return self._num_input_lead_months

    def build_dataset(self):
        return "dataset"


def bare_config(**kwargs):
    config = object.__new__(ConcreteConfig)
    config.model = kwargs.get("model")
    config.condition = kwargs.get("condition")
    config.condition_method = kwargs.get("condition_method")
    config.time_features = kwargs.get("time_features")
    config.lead_months = kwargs.get("lead_months")
    config.observation = kwargs.get("observation")
    config._num_input_lead_months = kwargs.get(
        "num_input_lead_months",
        3,
    )
    config._effective_condition = kwargs.get("effective_condition")
    return config


def test_lead_months_requires_list_or_end():
    with pytest.raises(ValueError, match="Provide a list"):
        lead_months_config()


def test_lead_months_explicit_list():
    value = lead_months_config(list_months=[1, 3, 5])

    assert value.build_lead_months() == [1, 3, 5]


def test_lead_months_range():
    value = lead_months_config(
        start=2,
        end=4,
    )

    np.testing.assert_array_equal(
        value.build_lead_months(),
        [2, 3, 4],
    )


def test_lead_months_single_value_range():
    value = lead_months_config(
        start=3,
        end=3,
    )

    np.testing.assert_array_equal(
        value.build_lead_months(),
        [3],
    )


def test_lead_months_empty_list_uses_range():
    value = lead_months_config(
        list_months=[],
        start=2,
        end=3,
    )

    np.testing.assert_array_equal(
        value.build_lead_months(),
        [2, 3],
    )


def test_abstract_class_cannot_be_instantiated():
    with pytest.raises(TypeError):
        DatasetConfigABC()


def test_requires_model_or_condition():
    with pytest.raises(
        ValueError,
        match="either model or condition",
    ):
        ConcreteConfig()


@pytest.mark.parametrize(
    "condition_method",
    [None, "static"],
)
def test_valid_condition_methods_without_derived_condition(
    condition_method,
):
    config = ConcreteConfig(
        model=make_data(),
        condition_method=condition_method,
    )

    assert config.condition_method == condition_method


@pytest.mark.parametrize(
    "condition_method",
    ["ensemble_mean", "cross_ensemble"],
)
def test_valid_condition_methods_with_explicit_condition(
    condition_method,
):
    condition = make_condition()

    config = ConcreteConfig(
        model=make_data(),
        condition=condition,
        condition_method=condition_method,
    )

    assert config.condition_method == condition_method
    assert config.effective_condition is condition


def test_valid_same_member_condition_method():
    condition = make_condition()

    config = ConcreteConfig(
        model=make_data(),
        condition=condition,
        condition_method="same_member",
    )

    assert config.condition_method == "same_member"
    assert config.effective_condition is condition


def test_invalid_condition_method():
    with pytest.raises(
        ValueError,
        match="Invalid condition_method",
    ):
        ConcreteConfig(
            model=make_data(),
            condition_method="invalid",
        )


@pytest.mark.parametrize(
    "time_features",
    [
        None,
        [],
        ["year"],
        ["lead_time"],
        ["month_sin"],
        ["month_cos"],
        ["month_sin", "month_cos"],
        [
            "year",
            "lead_time",
            "month_sin",
            "month_cos",
        ],
    ],
)
def test_valid_time_features(time_features):
    config = ConcreteConfig(
        model=make_data(),
        time_features=time_features,
    )

    assert config.time_features == time_features


@pytest.mark.parametrize(
    "time_features",
    [
        ["day"],
        ["year", "day"],
        ["month"],
        ["YEAR"],
    ],
)
def test_invalid_time_features(time_features):
    with pytest.raises(
        ValueError,
        match="Invalid time features",
    ):
        ConcreteConfig(
            model=make_data(),
            time_features=time_features,
        )


def test_default_lead_months():
    config = ConcreteConfig(
        model=make_data(),
        num_input_lead_months=4,
    )

    np.testing.assert_array_equal(
        config.lead_months,
        [1, 2, 3, 4],
    )


def test_lead_month_config_is_resolved():
    config = ConcreteConfig(
        model=make_data(),
        lead_months=lead_months_config(
            start=2,
            end=3,
        ),
    )

    np.testing.assert_array_equal(
        config.lead_months,
        [2, 3],
    )


def test_custom_lead_month_builder_is_resolved():
    builder = LeadMonthsBuilder([1, 3])

    config = ConcreteConfig(
        model=make_data(),
        lead_months=builder,
    )

    assert config.lead_months == [1, 3]
    assert builder.calls == 1


def test_existing_array_lead_months_is_preserved():
    values = np.asarray([1, 2])

    config = ConcreteConfig(
        model=make_data(),
        lead_months=values,
    )

    assert config.lead_months is values


def test_maximum_lead_month_validation():
    with pytest.raises(
        ValueError,
        match="Maximum available lead months is 3",
    ):
        ConcreteConfig(
            model=make_data(),
            lead_months=lead_months_config(list_months=[1, 4]),
            num_input_lead_months=3,
        )


def test_maximum_available_lead_month_is_accepted():
    config = ConcreteConfig(
        model=make_data(),
        lead_months=[3],
        num_input_lead_months=3,
    )

    assert config.lead_months == [3]


def test_condition_years_must_cover_model_years():
    with pytest.raises(
        ValueError,
        match="same time period",
    ):
        ConcreteConfig(
            model=make_data(years=(2000, 2001)),
            condition=make_condition(years=(2000,)),
            condition_method="ensemble_mean",
        )


def test_condition_may_cover_additional_years():
    condition = make_condition(years=(1999, 2000, 2001, 2002))

    config = ConcreteConfig(
        model=make_data(years=(2000, 2001)),
        condition=condition,
        condition_method="ensemble_mean",
    )

    assert config.effective_condition is condition


def test_condition_leads_must_cover_model_leads():
    with pytest.raises(
        ValueError,
        match="same lead_times",
    ):
        ConcreteConfig(
            model=make_data(leads=(1, 2, 3)),
            condition=make_condition(leads=(1, 2)),
            condition_method="ensemble_mean",
        )


def test_condition_may_cover_additional_leads():
    condition = make_condition(leads=(1, 2, 3, 4))

    config = ConcreteConfig(
        model=make_data(leads=(1, 2, 3)),
        condition=condition,
        condition_method="ensemble_mean",
    )

    assert config.effective_condition is condition


@pytest.mark.parametrize(
    (
        "model_ensembles",
        "condition_ensembles",
    ),
    [
        (None, ("r1", "r2")),
        (("r1", "r2"), None),
        (None, None),
    ],
)
def test_same_member_requires_ensemble_coordinates(
    model_ensembles,
    condition_ensembles,
):
    with pytest.raises(
        ValueError,
        match="same ensembles dims and coords",
    ):
        ConcreteConfig(
            model=make_data(ensembles=model_ensembles),
            condition=make_condition(ensembles=condition_ensembles),
            condition_method="same_member",
        )


def test_same_member_requires_equal_ensembles():
    with pytest.raises(
        ValueError,
        match="same ensemble members",
    ):
        ConcreteConfig(
            model=make_data(ensembles=("r1", "r2")),
            condition=make_condition(ensembles=("r1", "r3")),
            condition_method="same_member",
        )


def test_same_member_accepts_matching_ensembles():
    config = ConcreteConfig(
        model=make_data(ensembles=("r1", "r2")),
        condition=make_condition(ensembles=("r1", "r2")),
        condition_method="same_member",
    )

    assert config.effective_condition is config.condition


def test_static_condition_skips_year_and_lead_validation():
    condition = make_condition(
        years=(1990,),
        leads=(12,),
    )

    config = ConcreteConfig(
        model=make_data(
            years=(2000, 2001),
            leads=(1, 2, 3),
        ),
        condition=condition,
        condition_method="static",
    )

    assert config.effective_condition is condition


def test_matching_condition_passes_validation():
    condition = make_condition()

    config = ConcreteConfig(
        model=make_data(),
        condition=condition,
        condition_method="ensemble_mean",
    )

    assert config.effective_condition is condition


def test_same_files_skip_model_condition_comparison():
    model = make_data()

    condition = make_data(
        years=(1990,),
        leads=(99,),
        paths=model.paths,
        names=model.names,
        ensemble_list=model.ensemble_list,
    )

    config = ConcreteConfig(
        model=model,
        condition=condition,
        condition_method="ensemble_mean",
    )

    assert config._using_model_data_as_condition is True
    assert config.effective_condition is condition


def test_check_model_vs_condition_skips_without_condition():
    config = bare_config(
        model=make_data(),
        condition=None,
        condition_method=None,
    )

    assert config._check_model_vs_condition() is None


def test_check_model_vs_condition_skips_without_model():
    config = bare_config(
        model=None,
        condition=make_condition(),
        condition_method="ensemble_mean",
    )

    assert config._check_model_vs_condition() is None


@pytest.mark.parametrize(
    "condition_method",
    [
        "ensemble_mean",
        "cross_ensemble",
        "same_member",
    ],
)
def test_model_used_as_condition_for_supported_methods(
    condition_method,
):
    config = bare_config(
        model=make_data(),
        condition=None,
        condition_method=condition_method,
    )

    assert config._using_model_data_as_condition is True


@pytest.mark.parametrize(
    "condition_method",
    [None, "static"],
)
def test_model_not_used_as_condition_for_other_methods(
    condition_method,
):
    config = bare_config(
        model=make_data(),
        condition=None,
        condition_method=condition_method,
    )

    assert config._using_model_data_as_condition is False


def test_condition_without_model_is_not_model_condition():
    config = bare_config(
        model=None,
        condition=make_condition(),
        condition_method="ensemble_mean",
    )

    assert config._using_model_data_as_condition is False


@pytest.mark.parametrize(
    (
        "paths",
        "names",
        "ensemble_list",
    ),
    [
        (
            ["different.nc"],
            ["tas"],
            ["r1", "r2"],
        ),
        (
            ["model.nc"],
            ["pr"],
            ["r1", "r2"],
        ),
        (
            ["model.nc"],
            ["tas"],
            ["r3"],
        ),
    ],
)
def test_different_condition_source_is_not_model_condition(
    paths,
    names,
    ensemble_list,
):
    model = make_data()

    condition = make_data(
        paths=paths,
        names=names,
        ensemble_list=ensemble_list,
    )

    config = bare_config(
        model=model,
        condition=condition,
        condition_method="ensemble_mean",
    )

    assert config._using_model_data_as_condition is False


def test_identical_condition_source_is_model_condition():
    model = make_data()
    condition = make_data()

    config = bare_config(
        model=model,
        condition=condition,
        condition_method=None,
    )

    assert config._using_model_data_as_condition is True


def test_resolve_explicit_condition():
    condition = make_condition()

    config = bare_config(
        model=make_data(),
        condition=condition,
        condition_method="ensemble_mean",
    )

    with patch.object(
        config,
        "_model_as_condition",
    ) as resolver:
        result = config._resolve_condition()

    assert result is config
    assert config.effective_condition is condition
    resolver.assert_not_called()


def test_resolve_no_effective_condition():
    config = bare_config(
        model=make_data(),
        condition=None,
        condition_method=None,
    )

    with patch.object(
        config,
        "_model_as_condition",
    ) as resolver:
        result = config._resolve_condition()

    assert result is config
    assert config.effective_condition is None
    resolver.assert_not_called()


@pytest.mark.parametrize(
    (
        "condition_method",
        "expected_ensemble_mean",
    ),
    [
        ("ensemble_mean", True),
        ("cross_ensemble", False),
        ("same_member", False),
        ("static", False),
        (None, False),
    ],
)
def test_model_as_condition_construction(
    condition_method,
    expected_ensemble_mean,
):
    model = make_data()

    config = bare_config(
        model=model,
        condition=None,
        condition_method=condition_method,
    )

    constructed = object()

    with patch.object(
        module,
        "ModelDataConfig",
        return_value=constructed,
    ) as constructor:
        result = config._model_as_condition()

    assert result is constructed

    constructor.assert_called_once_with(
        paths=model.paths,
        names=model.names,
        preprocessing_pipeline=(model.preprocessing_pipeline),
        ensemble_list=model.ensemble_list,
        concat_dim=model.concat_dim,
        file_type=model.file_type,
        ensemble_mean=expected_ensemble_mean,
        rename_dict=model.rename_dict,
    )


def test_resolve_model_as_condition():
    model = make_data()

    config = bare_config(
        model=model,
        condition=None,
        condition_method="ensemble_mean",
    )

    effective = object()

    with patch.object(
        config,
        "_model_as_condition",
        return_value=effective,
    ) as resolver:
        result = config._resolve_condition()

    assert result is config
    assert config.effective_condition is effective
    resolver.assert_called_once_with()


def test_observation_requires_condition_spatial_coordinate(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "supported_NN_dimensions_sorted",
        ["lat"],
    )

    model = make_data(
        extra_coords={"lat": Coordinate([45.0, 46.0])},
    )

    condition = make_condition()

    with pytest.raises(TypeError):
        ConcreteConfig(
            model=model,
            condition=condition,
            condition_method="ensemble_mean",
            observation=object(),
        )


def test_observation_requires_matching_spatial_coordinate(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "supported_NN_dimensions_sorted",
        ["lat"],
    )

    model = make_data(
        extra_coords={"lat": Coordinate([45.0, 46.0])},
    )

    condition = make_condition(
        extra_coords={"lat": Coordinate([40.0, 41.0])},
    )

    with pytest.raises(TypeError):
        ConcreteConfig(
            model=model,
            condition=condition,
            condition_method="ensemble_mean",
            observation=object(),
        )


def test_observation_accepts_matching_spatial_coordinates(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "supported_NN_dimensions_sorted",
        ["lat", "lon"],
    )

    model = make_data(
        extra_coords={
            "lat": Coordinate([45.0, 46.0]),
            "lon": Coordinate([-125.0, -124.0]),
        },
    )

    condition = make_condition(
        extra_coords={
            "lat": Coordinate([45.0, 46.0]),
            "lon": Coordinate([-125.0, -124.0]),
        },
    )

    config = ConcreteConfig(
        model=model,
        condition=condition,
        condition_method="ensemble_mean",
        observation=object(),
    )

    assert config.effective_condition is condition


def test_observation_ignores_non_nn_coordinates(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "supported_NN_dimensions_sorted",
        ["lat"],
    )

    model = make_data(
        extra_coords={"unrelated": Coordinate([1, 2])},
    )

    condition = make_condition()

    config = ConcreteConfig(
        model=model,
        condition=condition,
        condition_method="ensemble_mean",
        observation=object(),
    )

    assert config.effective_condition is condition


def test_observation_validation_skipped_without_observation(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "supported_NN_dimensions_sorted",
        ["lat"],
    )

    model = make_data(
        extra_coords={"lat": Coordinate([45.0, 46.0])},
    )

    condition = make_condition()

    config = ConcreteConfig(
        model=model,
        condition=condition,
        condition_method="ensemble_mean",
        observation=None,
    )

    assert config.effective_condition is condition


def test_forced_coordinate_equality_true(monkeypatch):
    monkeypatch.setattr(
        module,
        "supported_NN_dimensions_sorted",
        ["lat"],
    )

    model = make_data(
        extra_coords={
            "lat": Coordinate(
                [45.0],
                equality=True,
            )
        },
    )

    condition = make_condition(
        extra_coords={"lat": Coordinate([99.0], equality=True)},
    )

    config = ConcreteConfig(
        model=model,
        condition=condition,
        condition_method="ensemble_mean",
        observation=object(),
    )

    assert config.effective_condition is condition


def test_check_methods_return_self():
    config = bare_config(
        model=make_data(),
        condition=None,
        condition_method=None,
        time_features=None,
    )

    assert config._check_required_input_source() is config
    assert config._check_condition_method() is config
    assert config._check_time_features() is config


def test_resolve_lead_months_returns_none():
    config = bare_config(
        model=make_data(),
        lead_months=lead_months_config(
            start=1,
            end=2,
        ),
    )

    result = config._resolve_lead_months()

    assert result is None

    np.testing.assert_array_equal(
        config.lead_months,
        [1, 2],
    )


def test_resolve_lead_months_none_is_unchanged():
    config = bare_config(
        model=make_data(),
        lead_months=None,
    )

    result = config._resolve_lead_months()

    assert result is None
    assert config.lead_months is None


def test_resolve_lead_months_plain_list_is_unchanged():
    values = [1, 2]

    config = bare_config(
        model=make_data(),
        lead_months=values,
    )

    result = config._resolve_lead_months()

    assert result is None
    assert config.lead_months is values


def test_concrete_abstract_implementations():
    config = ConcreteConfig(model=make_data())

    assert config._check_model() is config
    assert config._check_condition() is config
    assert config.model_checked is True
    assert config.condition_checked is True
    assert config.ds_operator == "operator"
    assert config.build_dataset() == "dataset"
    assert config.num_input_lead_months == 3


def test_abstract_method_bodies_are_callable():
    dummy = MagicMock()

    assert DatasetConfigABC._check_model(dummy) is None
    assert DatasetConfigABC._check_condition(dummy) is None
    assert DatasetConfigABC.ds_operator.fget(dummy) is None
    assert DatasetConfigABC.num_input_lead_months.fget(dummy) is None
    assert DatasetConfigABC.build_dataset(dummy) is None


def test_class_is_abstract():
    assert issubclass(
        DatasetConfigABC,
        abc.ABC,
    )

    assert {
        "_check_model",
        "_check_condition",
        "ds_operator",
        "num_input_lead_months",
        "build_dataset",
    }.issubset(DatasetConfigABC.__abstractmethods__)
