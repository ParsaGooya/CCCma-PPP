from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

import cccma_ppp.data_modules.dataset.dataset_abc as module
from cccma_ppp.data_modules.dataset.dataset_abc import (
    AddedTimeFeatures,
    DatasetABC,
    DatasetConfigABC,
    lead_months_config,
)


class Coordinate:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.size = self.values.size

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]

    def equals(self, other):
        return np.array_equal(
            self.values,
            other.values,
        )


class BareDataset(DatasetABC):
    @property
    def _load_model(self):
        return getattr(self, "_load_model_value", True)

    @property
    def _write_condition_to_input(self):
        return getattr(
            self,
            "_write_condition_value",
            False,
        )

    @property
    def _concat_condition_to_input(self):
        return getattr(
            self,
            "_concat_condition_value",
            False,
        )


def bare_dataset(**kwargs):
    dataset = object.__new__(BareDataset)

    for name, value in kwargs.items():
        setattr(dataset, name, value)

    return dataset


def make_pipeline(fitted_preprocessors=None):
    pipeline = MagicMock()
    pipeline.fitted_preprocessors = list(fitted_preprocessors or [])
    pipeline.transform.side_effect = lambda value: value
    return pipeline


def make_config_data(
    paths=("model.nc",),
    names=("tas",),
    ensembles=("r1", "r2"),
    ensemble_list=("r1", "r2"),
    ensemble_mean=False,
    times=(2000, 2001),
    lead_times=(1, 2, 3),
    extra_coords=None,
    pipeline=None,
):
    coords = {
        "year": Coordinate(times),
        "lead_time": Coordinate(lead_times),
    }

    if ensembles is not None:
        coords["ensembles"] = Coordinate(ensembles)

    if extra_coords:
        coords.update(extra_coords)

    return SimpleNamespace(
        paths=list(paths),
        list_paths=list(paths),
        names=list(names),
        ensemble_list=(list(ensemble_list) if ensemble_list is not None else None),
        ensemble_mean=ensemble_mean,
        preprocessing_pipeline=(pipeline if pipeline is not None else make_pipeline()),
        concat_dim="year",
        file_type="netcdf",
        rename_dict={"old": "new"},
        info=SimpleNamespace(
            coords=coords,
            sizes={name: coordinate.size for name, coordinate in coords.items()},
        ),
    )


def make_dataset_config(**kwargs):
    default_effective_input = make_config_data(
        ensembles=None,
        ensemble_list=None,
    )

    return SimpleNamespace(
        model=kwargs.get("model"),
        condition=kwargs.get("condition"),
        condition_method=kwargs.get("condition_method"),
        lead_months=kwargs.get(
            "lead_months",
            [1, 2],
        ),
        available_times=np.asarray(
            kwargs.get(
                "available_times",
                [2000, 2001],
            )
        ),
        input_lead_months=np.asarray(
            kwargs.get(
                "input_lead_months",
                [1, 2],
            )
        ),
        effective_input=kwargs.get(
            "effective_input",
            default_effective_input,
        ),
        effective_condition=kwargs.get("effective_condition"),
        _fitted_preprocessors=kwargs.get(
            "fitted_preprocessors",
            True,
        ),
    )


def make_reference_config(
    lead_months=(1, 2, 3, 6),
    common_time=(2000, 2001, 2002),
):
    return SimpleNamespace(
        lead_months=np.asarray(lead_months),
        get_common_time=np.asarray(common_time),
    )


def make_xarray(
    times=(2000, 2001),
    lead_times=(1, 2),
    ensembles=None,
):
    coords = {
        "year": list(times),
        "lead_time": list(lead_times),
    }
    dims = ["year", "lead_time"]
    shape = [len(times), len(lead_times)]

    if ensembles is not None:
        coords["ensembles"] = list(ensembles)
        dims.append("ensembles")
        shape.append(len(ensembles))

    return xr.DataArray(
        np.arange(np.prod(shape), dtype=float).reshape(shape),
        dims=dims,
        coords=coords,
        name="tas",
    )


def make_mask(
    times=(2000, 2001),
    lead_times=(1, 2),
    values=None,
):
    if values is None:
        values = np.zeros(
            (len(times), len(lead_times)),
            dtype=float,
        )

    return xr.DataArray(
        values,
        dims=("year", "lead_time"),
        coords={
            "year": list(times),
            "lead_time": list(lead_times),
        },
    )


def test_lead_months_requires_list_or_end():
    with pytest.raises(ValueError, match="Provide a list"):
        lead_months_config()


def test_lead_months_explicit_list():
    config = lead_months_config(list_months=[1, 3])

    assert config.build_lead_months() == [1, 3]


@pytest.mark.pruned
def test_lead_months_range():
    config = lead_months_config(start=2, end=4)

    np.testing.assert_array_equal(
        config.build_lead_months(),
        [2, 3, 4],
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
def test_added_time_features_valid(time_features):
    result = AddedTimeFeatures(
        make_reference_config(),
        time_features,
    )

    assert result.time_features == tuple(time_features or ())


def test_added_time_features_invalid():
    with pytest.raises(
        ValueError,
        match="Unsupported time features",
    ):
        AddedTimeFeatures(
            make_reference_config(),
            ["invalid"],
        )


@pytest.mark.parametrize(
    "selection",
    [
        {},
        {"year": 2000},
        {"lead_time": 1},
    ],
)
def test_added_time_features_requires_dimensions(
    selection,
):
    features = AddedTimeFeatures(
        make_reference_config(),
        ["year"],
    )

    with pytest.raises(
        ValueError,
        match="required sample dimensions",
    ):
        features(
            selection,
            xr.DataArray(np.ones((2,))),
        )


@pytest.mark.pruned
def test_added_time_features_values():
    features = AddedTimeFeatures(
        make_reference_config(),
        [
            "year",
            "lead_time",
            "month_sin",
            "month_cos",
        ],
    )

    result = features(
        {
            "year": 2000,
            "lead_time": 3,
        },
        xr.DataArray(np.ones((2,))),
    )

    assert result.shape == (4,)
    assert result[1] == pytest.approx(0.5)
    assert result[2] == pytest.approx(1.0)
    assert result[3] == pytest.approx(
        0.0,
        abs=1e-12,
    )


@pytest.mark.pruned
def test_added_time_features_no_broadcast():
    features = AddedTimeFeatures(
        make_reference_config(),
        ["year"],
    )

    result = features(
        {
            "year": 2000,
            "lead_time": 1,
        },
        xr.DataArray(np.ones((2,))),
    )

    assert result.shape == (1,)
    assert result.ndim == 1


def test_added_time_features_broadcast():
    features = AddedTimeFeatures(
        make_reference_config(),
        ["year", "month_sin"],
    )

    result = features(
        {
            "year": 2000,
            "lead_time": 1,
        },
        xr.DataArray(
            np.ones((3, 4, 5)),
            dims=("channels", "lat", "lon"),
        ),
    )

    assert result.shape == (2, 4, 5)


@pytest.mark.pruned
def test_added_time_features_length():
    features = AddedTimeFeatures(
        make_reference_config(),
        ["year", "lead_time", "month_sin"],
    )

    assert len(features) == 3


def test_added_time_features_equal():
    left = AddedTimeFeatures(
        make_reference_config(),
        ["year"],
    )
    right = AddedTimeFeatures(
        make_reference_config(),
        ["year"],
    )

    assert left == right


@pytest.mark.pruned
def test_added_time_features_not_equal_features():
    left = AddedTimeFeatures(
        make_reference_config(),
        ["year"],
    )
    right = AddedTimeFeatures(
        make_reference_config(),
        ["lead_time"],
    )

    assert left != right


@pytest.mark.pruned
def test_added_time_features_not_equal_lead_months():
    left = AddedTimeFeatures(
        make_reference_config(
            lead_months=(1, 2, 3),
        ),
        ["year"],
    )
    right = AddedTimeFeatures(
        make_reference_config(
            lead_months=(1, 2, 4),
        ),
        ["year"],
    )

    assert left != right


@pytest.mark.pruned
def test_added_time_features_not_equal_common_time():
    left = AddedTimeFeatures(
        make_reference_config(
            common_time=(2000, 2001),
        ),
        ["year"],
    )
    right = AddedTimeFeatures(
        make_reference_config(
            common_time=(1990, 1991),
        ),
        ["year"],
    )

    assert left != right


def test_added_time_features_other_type():
    features = AddedTimeFeatures(
        make_reference_config(),
        ["year"],
    )

    assert features.__eq__(object()) is NotImplemented


def test_check_init_requires_fitted_preprocessors():
    dataset = bare_dataset(
        config=make_dataset_config(
            fitted_preprocessors=False,
            available_times=[2000],
        ),
        requested_years=[2000],
    )

    with pytest.raises(
        RuntimeError,
        match="fit preprocessors",
    ):
        dataset._check_init()


def test_check_init_rejects_unavailable_years():
    dataset = bare_dataset(
        config=make_dataset_config(
            available_times=[2000],
        ),
        requested_years=[2001],
    )

    with pytest.raises(
        ValueError,
        match="requested years",
    ):
        dataset._check_init()


def test_check_init_accepts_available_years():
    dataset = bare_dataset(
        config=make_dataset_config(
            available_times=[2000, 2001],
        ),
        requested_years=[2000],
    )

    assert dataset._check_init() is None


def test_resolve_mask_creates_default(monkeypatch):
    created = make_mask()
    creator = MagicMock(return_value=created)

    monkeypatch.setattr(
        module,
        "_create_train_mask",
        creator,
    )

    dataset = bare_dataset(
        config=make_dataset_config(
            available_times=[2000, 2001],
            input_lead_months=[1, 2],
        ),
        mask=None,
    )

    assert dataset._resolve_mask() is None

    creator.assert_called_once()
    np.testing.assert_array_equal(
        creator.call_args.kwargs["time"],
        [2000, 2001],
    )
    np.testing.assert_array_equal(
        creator.call_args.kwargs["lead_times"],
        [1, 2],
    )
    assert not bool(dataset.mask.any())


@pytest.mark.pruned
def test_resolve_mask_preserves_existing_mask():
    mask = make_mask()

    dataset = bare_dataset(
        config=make_dataset_config(),
        mask=mask,
    )

    dataset._resolve_mask()

    xr.testing.assert_identical(
        dataset.mask,
        mask,
    )


def test_resolve_mask_rejects_missing_dimension():
    dataset = bare_dataset(
        config=make_dataset_config(),
        mask=xr.DataArray(
            [False],
            dims=("year",),
            coords={"year": [2000]},
        ),
    )

    with pytest.raises(
        ValueError,
        match="mask must have",
    ):
        dataset._resolve_mask()


def test_prepare_sampling_mask_requires_all_selectors():
    dataset = bare_dataset(
        config=make_dataset_config(),
        mask=make_mask(),
    )

    with pytest.raises(
        ValueError,
        match="No selectors provided",
    ):
        dataset._prepare_sampling_mask({"year": [2000]})


@pytest.mark.parametrize(
    "ensembles",
    [None, ("r1", "r2")],
)
@pytest.mark.parametrize(
    "load",
    [False, True],
)
def test_load_xarray_data(
    ensembles,
    load,
):
    config = make_config_data(
        ensembles=ensembles,
    )
    dataset = bare_dataset()
    expected = object()

    with patch.object(
        module,
        "_load_xarray_data",
        return_value=expected,
    ) as loader:
        result = dataset._load_xarray_data(
            config,
            load=load,
        )

    assert result is expected

    loader.assert_called_once_with(
        config.list_paths,
        names=config.names,
        ensemble_mean=config.ensemble_mean,
        selection=(
            {"ensembles": (config.info.coords["ensembles"])}
            if ensembles is not None
            else None
        ),
        concat_dim=config.concat_dim,
        rename_dict=config.rename_dict,
        load=load,
    )


def test_get_model_indexes_returns_none():
    dataset = bare_dataset(
        _load_model_value=False,
    )

    assert dataset.get_model_indexes({"year": np.asarray([2000])}) is None


def test_get_model_indexes_success():
    dataset = bare_dataset(
        _load_model_value=True,
        model_dataset=make_xarray(),
    )

    result = dataset.get_model_indexes(
        {
            "year": np.asarray([2001]),
            "lead_time": np.asarray([2]),
        }
    )

    np.testing.assert_array_equal(
        result["year"],
        [1],
    )
    np.testing.assert_array_equal(
        result["lead_time"],
        [1],
    )


@pytest.mark.pruned
def test_get_model_indexes_multiple_samples():
    dataset = bare_dataset(
        _load_model_value=True,
        model_dataset=make_xarray(),
    )

    result = dataset.get_model_indexes(
        {
            "year": np.asarray([2000, 2001]),
            "lead_time": np.asarray([2, 1]),
        }
    )

    np.testing.assert_array_equal(
        result["year"],
        [0, 1],
    )
    np.testing.assert_array_equal(
        result["lead_time"],
        [1, 0],
    )


def test_get_model_indexes_missing_coordinate():
    dataset = bare_dataset(
        _load_model_value=True,
        model_dataset=make_xarray(),
    )

    with pytest.raises(
        ValueError,
        match="not found in the model dataset",
    ):
        dataset.get_model_indexes({"year": np.asarray([1999])})


@pytest.mark.parametrize(
    "condition_method",
    [None, "static"],
)
def test_get_cond_indexes_returns_none(
    condition_method,
):
    dataset = bare_dataset(
        condition_dataset=(None if condition_method is None else make_xarray()),
        config=SimpleNamespace(
            condition_method=condition_method,
        ),
    )

    result = dataset.get_cond_indexes({"year": np.asarray([2000])})

    assert result is None


@pytest.mark.pruned
def test_get_cond_indexes_regular():
    dataset = bare_dataset(
        condition_dataset=make_xarray(),
        config=SimpleNamespace(
            condition_method="ensemble_mean",
        ),
    )

    result = dataset.get_cond_indexes(
        {
            "year": np.asarray([2001]),
            "lead_time": np.asarray([2]),
            "unknown": np.asarray([9]),
        }
    )

    assert set(result) == {
        "year",
        "lead_time",
    }
    np.testing.assert_array_equal(
        result["year"],
        [1],
    )
    np.testing.assert_array_equal(
        result["lead_time"],
        [1],
    )


def test_get_cond_indexes_same_member_requires_ensembles():
    dataset = bare_dataset(
        condition_dataset=make_xarray(
            ensembles=("r1", "r2"),
        ),
        config=SimpleNamespace(
            condition_method="same_member",
        ),
    )

    with pytest.raises(
        ValueError,
        match="requires ensemble coordinates",
    ):
        dataset.get_cond_indexes(
            {
                "year": np.asarray([2000]),
                "lead_time": np.asarray([1]),
            }
        )


def test_get_cond_indexes_same_member_success():
    dataset = bare_dataset(
        condition_dataset=make_xarray(
            ensembles=("r1", "r2"),
        ),
        config=SimpleNamespace(
            condition_method="same_member",
        ),
    )

    result = dataset.get_cond_indexes(
        {
            "year": np.asarray([2000]),
            "lead_time": np.asarray([1]),
            "ensembles": np.asarray(["r2"]),
        }
    )

    np.testing.assert_array_equal(
        result["ensembles"],
        [1],
    )


def test_get_cond_indexes_missing_coordinate():
    dataset = bare_dataset(
        condition_dataset=make_xarray(),
        config=SimpleNamespace(
            condition_method="ensemble_mean",
        ),
    )

    with pytest.raises(
        ValueError,
        match="conditioning coordinates were not found",
    ):
        dataset.get_cond_indexes({"year": np.asarray([1999])})


def test_get_input_shape_with_flattener(
    monkeypatch,
):
    class FakeFlattennanremove:
        pass

    flattener = FakeFlattennanremove()
    flattener.final_locations = np.zeros(5)

    pipeline = make_pipeline(
        fitted_preprocessors=[flattener],
    )
    pipeline.get_preprocessors.return_value = flattener

    effective_input = make_config_data(
        names=("tas", "pr"),
        pipeline=pipeline,
    )
    effective_condition = make_config_data(
        names=("condition",),
    )

    fake_module = SimpleNamespace(Flattennanremove=FakeFlattennanremove)

    monkeypatch.setitem(
        __import__("sys").modules,
        ("cccma_ppp.preprocessing.utils_preprocessing"),
        fake_module,
    )

    dataset = bare_dataset(
        config=SimpleNamespace(
            effective_input=effective_input,
            effective_condition=effective_condition,
        ),
        _concat_condition_value=True,
    )

    assert dataset.get_input_shape() == (3, 5)


def test_get_input_shape_without_flattener(
    monkeypatch,
):
    class FakeFlattennanremove:
        pass

    fake_module = SimpleNamespace(Flattennanremove=FakeFlattennanremove)

    monkeypatch.setitem(
        __import__("sys").modules,
        ("cccma_ppp.preprocessing.utils_preprocessing"),
        fake_module,
    )
    monkeypatch.setattr(
        module,
        "supported_NN_dimensions_sorted",
        ["lat", "lon"],
    )

    effective_input = make_config_data(
        extra_coords={
            "lat": Coordinate([1, 2]),
            "lon": Coordinate([3, 4, 5]),
        }
    )

    dataset = bare_dataset(
        config=SimpleNamespace(
            effective_input=effective_input,
            effective_condition=None,
        ),
        _concat_condition_value=False,
    )

    assert dataset.get_input_shape() == (1, 2, 3)


@pytest.mark.parametrize(
    ("time_features", "expected"),
    [
        (None, 0),
        ([], 0),
        (["year"], 1),
        (["year", "month_sin"], 2),
    ],
)
def test_get_added_features_dim(
    time_features,
    expected,
):
    dataset = bare_dataset(
        time_features=AddedTimeFeatures(
            make_reference_config(),
            time_features,
        )
    )

    assert dataset.get_added_features_dim() == expected


def test_index_condition_returns_none():
    dataset = bare_dataset(
        condition_dataset=None,
    )

    assert dataset._index_condition_dataset(0) is None


def test_index_static_condition():
    pipeline = make_pipeline()

    dataset = bare_dataset(
        condition_dataset=make_xarray(),
        cond_indexes=None,
        config=SimpleNamespace(
            condition_method="static",
            effective_condition=make_config_data(
                pipeline=pipeline,
            ),
        ),
    )

    expected = object()

    with patch.object(
        module,
        "_unwrap_data_variables",
        return_value=expected,
    ) as unwrap:
        result = dataset._index_condition_dataset(0)

    assert result is expected
    pipeline.transform.assert_called_once()
    unwrap.assert_called_once()


def test_index_regular_condition():
    pipeline = make_pipeline()

    dataset = bare_dataset(
        condition_dataset=make_xarray(),
        cond_indexes={
            "year": np.asarray([1]),
            "lead_time": np.asarray([0]),
        },
        config=SimpleNamespace(
            condition_method="ensemble_mean",
            effective_condition=make_config_data(
                pipeline=pipeline,
            ),
        ),
    )

    expected = object()

    with patch.object(
        module,
        "_unwrap_data_variables",
        return_value=expected,
    ):
        result = dataset._index_condition_dataset(0)

    assert result is expected
    pipeline.transform.assert_called_once()


def test_index_cross_ensemble_condition(
    monkeypatch,
):
    pipeline = make_pipeline()

    dataset = bare_dataset(
        condition_dataset=make_xarray(
            ensembles=("r1", "r2", "r3"),
        ),
        cond_indexes={
            "year": np.asarray([0]),
            "lead_time": np.asarray([1]),
        },
        config=SimpleNamespace(
            condition_method="cross_ensemble",
            effective_condition=make_config_data(
                ensembles=("r1", "r2", "r3"),
                pipeline=pipeline,
            ),
        ),
    )

    randint = MagicMock(return_value=2)

    monkeypatch.setattr(
        module.np.random,
        "randint",
        randint,
    )

    expected = object()

    with patch.object(
        module,
        "_unwrap_data_variables",
        return_value=expected,
    ):
        result = dataset._index_condition_dataset(0)

    assert result is expected
    randint.assert_called_once_with(3)

    indexed = pipeline.transform.call_args.args[0]

    assert indexed.sizes["ensembles"] == 1
    assert indexed.coords["ensembles"].values[0] == "r3"


def test_index_model_returns_none():
    dataset = bare_dataset(
        _load_model_value=False,
    )

    assert dataset._index_model_dataset(0) is None


def test_index_model_success():
    pipeline = make_pipeline()

    dataset = bare_dataset(
        _load_model_value=True,
        model_dataset=make_xarray(),
        model_indexes={
            "year": np.asarray([1]),
            "lead_time": np.asarray([0]),
        },
        config=SimpleNamespace(
            model=make_config_data(
                pipeline=pipeline,
            ),
        ),
    )

    expected = object()

    with patch.object(
        module,
        "_unwrap_data_variables",
        return_value=expected,
    ) as unwrap:
        result = dataset._index_model_dataset(0)

    assert result is expected
    pipeline.transform.assert_called_once()
    unwrap.assert_called_once()


class ConcreteDatasetConfig(DatasetConfigABC):
    def __init__(
        self,
        model=None,
        condition=None,
        condition_method=None,
        lead_months=None,
        effective_input=None,
        available_times=(2000, 2001),
        observation=None,
    ):
        self.model = model
        self.condition = condition
        self.condition_method = condition_method
        self.lead_months = lead_months
        self._effective_input_value = (
            effective_input
            if effective_input is not None
            else model
            if model is not None
            else condition
        )
        self._available_times_value = np.asarray(available_times)
        self.observation = observation

        super().__init__()

    @property
    def available_times(self):
        return self._available_times_value

    @property
    def ds_operator(self):
        return "operator"

    @property
    def effective_input(self):
        return self._effective_input_value

    def build_dataset(self):
        return "dataset"


def bare_abc_config(**kwargs):
    config = object.__new__(ConcreteDatasetConfig)
    config.model = kwargs.get("model")
    config.condition = kwargs.get("condition")
    config.condition_method = kwargs.get("condition_method")
    config.lead_months = kwargs.get("lead_months")
    config.observation = kwargs.get("observation")
    config._effective_input_value = kwargs.get("effective_input")
    config._effective_condition = kwargs.get("effective_condition")
    config._available_times_value = np.asarray(
        kwargs.get(
            "available_times",
            [2000, 2001],
        )
    )
    config._fitted_preprocessors = kwargs.get(
        "fitted_preprocessors",
        True,
    )
    return config


def test_config_requires_input_source():
    with pytest.raises(
        ValueError,
        match="either model or condition",
    ):
        ConcreteDatasetConfig()


@pytest.mark.parametrize(
    "condition_method",
    [
        None,
        "static",
        "cross_ensemble",
        "same_member",
    ],
)
def test_check_condition_method_valid(
    condition_method,
):
    config = bare_abc_config(
        condition_method=condition_method,
    )

    assert config._check_condition_method() is config


@pytest.mark.parametrize(
    "condition_method",
    [
        "",
        "invalid",
        "mean",
        "cross",
    ],
)
def test_check_condition_method_invalid(
    condition_method,
):
    config = bare_abc_config(
        condition_method=condition_method,
    )

    with pytest.raises(
        ValueError,
        match="Invalid condition_method",
    ):
        config._check_condition_method()


def test_check_required_input_source_model_only():
    config = bare_abc_config(
        model=make_config_data(),
        condition=None,
    )

    assert config._check_required_input_source() is config


@pytest.mark.pruned
def test_check_required_input_source_condition_only():
    config = bare_abc_config(
        model=None,
        condition=make_config_data(),
    )

    assert config._check_required_input_source() is config


@pytest.mark.pruned
def test_check_required_input_source_rejects_none():
    config = bare_abc_config(
        model=None,
        condition=None,
    )

    with pytest.raises(
        ValueError,
        match="either model or condition",
    ):
        config._check_required_input_source()


def test_resolve_lead_months_none():
    config = bare_abc_config(
        lead_months=None,
    )

    assert config._resolve_lead_months() is None
    assert config.lead_months is None


@pytest.mark.pruned
def test_resolve_lead_months_preserves_list():
    months = [1, 3]

    config = bare_abc_config(
        lead_months=months,
    )

    config._resolve_lead_months()

    assert config.lead_months is months


def test_resolve_lead_months_builds_configuration():
    config = bare_abc_config(
        lead_months=lead_months_config(
            start=2,
            end=4,
        ),
    )

    config._resolve_lead_months()

    np.testing.assert_array_equal(
        config.lead_months,
        [2, 3, 4],
    )


@pytest.mark.pruned
def test_using_model_as_condition_without_condition():
    config = bare_abc_config(
        model=make_config_data(),
        condition=None,
        condition_method="ensemble_mean",
    )

    assert config._using_model_data_as_condition is True


@pytest.mark.parametrize(
    "condition_method",
    [
        "cross_ensemble",
        "same_member",
    ],
)
def test_using_model_as_condition_supported_methods(
    condition_method,
):
    config = bare_abc_config(
        model=make_config_data(),
        condition=None,
        condition_method=condition_method,
    )

    assert config._using_model_data_as_condition is True


@pytest.mark.parametrize(
    "condition_method",
    [
        None,
        "static",
    ],
)
def test_not_using_model_as_condition_without_condition(
    condition_method,
):
    config = bare_abc_config(
        model=make_config_data(),
        condition=None,
        condition_method=condition_method or "static",
    )

    assert config._using_model_data_as_condition is False


@pytest.mark.pruned
def test_using_same_model_source_as_condition():
    model = make_config_data()
    condition = make_config_data()

    config = bare_abc_config(
        model=model,
        condition=condition,
        condition_method=None,
    )

    assert config._using_model_data_as_condition is True


@pytest.mark.parametrize(
    (
        "paths",
        "names",
        "ensemble_list",
    ),
    [
        (
            ("different.nc",),
            ("tas",),
            ("r1", "r2"),
        ),
        (
            ("model.nc",),
            ("pr",),
            ("r1", "r2"),
        ),
        (
            ("model.nc",),
            ("tas",),
            ("r3",),
        ),
    ],
)
def test_different_condition_is_not_model_source(
    paths,
    names,
    ensemble_list,
):
    config = bare_abc_config(
        model=make_config_data(),
        condition=make_config_data(
            paths=paths,
            names=names,
            ensemble_list=ensemble_list,
        ),
        condition_method=None,
    )

    assert config._using_model_data_as_condition is False


@pytest.mark.pruned
def test_condition_without_model_is_not_model_source():
    config = bare_abc_config(
        model=None,
        condition=make_config_data(),
        condition_method="ensemble_mean",
    )

    assert config._using_model_data_as_condition is False


def test_resolve_explicit_condition():
    condition = make_config_data(
        paths=("condition.nc",),
    )

    config = bare_abc_config(
        model=make_config_data(),
        condition=condition,
        condition_method="static",
    )

    assert config._resolve_condition() is config
    assert config.effective_condition is condition


def test_resolve_model_as_condition():
    expected = object()

    config = bare_abc_config(
        model=make_config_data(),
        condition=None,
        condition_method="ensemble_mean",
    )

    with patch.object(
        config,
        "_model_as_condition",
        return_value=expected,
    ) as builder:
        result = config._resolve_condition()

    assert result is config
    assert config.effective_condition is expected
    builder.assert_called_once_with()


@pytest.mark.parametrize(
    (
        "condition_method",
        "expected_ensemble_mean",
    ),
    [
        ("ensemble_mean", True),
        ("cross_ensemble", False),
        ("same_member", False),
    ],
)
def test_model_as_condition_constructor_arguments(
    condition_method,
    expected_ensemble_mean,
):
    model = make_config_data()

    config = bare_abc_config(
        model=model,
        condition=None,
        condition_method=condition_method,
    )

    expected = object()

    with patch.object(
        module,
        "ModelDataConfig",
        return_value=expected,
    ) as constructor:
        result = config._model_as_condition()

    assert result is expected

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


def test_check_model_without_model():
    config = bare_abc_config(
        model=None,
        condition_method="same_member",
    )

    assert config._check_model() is config


def test_check_model_non_same_member():
    config = bare_abc_config(
        model=make_config_data(
            ensemble_mean=True,
        ),
        condition_method="ensemble_mean",
    )

    assert config._check_model() is config


def test_check_model_same_member_valid():
    config = bare_abc_config(
        model=make_config_data(
            ensemble_mean=False,
        ),
        condition_method="same_member",
    )

    assert config._check_model() is config


def test_check_model_same_member_rejects_mean():
    config = bare_abc_config(
        model=make_config_data(
            ensemble_mean=True,
        ),
        condition_method="same_member",
    )

    with pytest.raises(
        ValueError,
        match="should not be ensemble mean",
    ):
        config._check_model()


def test_static_condition_requires_dataset():
    config = bare_abc_config(
        model=make_config_data(),
        condition=None,
        condition_method="static",
        effective_condition=None,
    )

    with pytest.raises(
        ValueError,
        match="condition dataset must be specified",
    ):
        config._check_condition()


def test_condition_requires_method():
    condition = make_config_data(
        paths=("condition.nc",),
    )

    config = bare_abc_config(
        model=make_config_data(),
        condition=condition,
        condition_method=None,
        effective_condition=condition,
    )

    with pytest.raises(
        ValueError,
        match="specify condition_method",
    ):
        config._check_condition()


@pytest.mark.parametrize(
    "condition_method",
    [
        "cross_ensemble",
        "same_member",
    ],
)
def test_ensemble_condition_rejects_mean(
    condition_method,
):
    condition = make_config_data(
        paths=("condition.nc",),
        ensemble_mean=True,
    )

    config = bare_abc_config(
        model=make_config_data(),
        condition=condition,
        condition_method=condition_method,
        effective_condition=condition,
    )

    with pytest.raises(
        ValueError,
        match="ensemble_mean cannot be True",
    ):
        config._check_condition()


@pytest.mark.parametrize(
    "condition_method",
    [
        "cross_ensemble",
        "same_member",
    ],
)
def test_ensemble_condition_requires_ensembles(
    condition_method,
):
    condition = make_config_data(
        paths=("condition.nc",),
        ensembles=None,
        ensemble_list=None,
        ensemble_mean=False,
    )

    config = bare_abc_config(
        model=make_config_data(),
        condition=condition,
        condition_method=condition_method,
        effective_condition=condition,
    )

    with pytest.raises(
        ValueError,
        match="ensembles dim must exist",
    ):
        config._check_condition()


@pytest.mark.parametrize(
    "condition_method",
    [
        "cross_ensemble",
        "same_member",
    ],
)
def test_ensemble_condition_valid(
    condition_method,
):
    condition = make_config_data(
        paths=("condition.nc",),
        ensemble_mean=False,
    )

    config = bare_abc_config(
        model=make_config_data(),
        condition=condition,
        condition_method=condition_method,
        effective_condition=condition,
    )

    assert config._check_condition() is config


def test_ensemble_mean_condition_requires_mean():
    condition = make_config_data(
        paths=("condition.nc",),
        ensemble_mean=False,
    )

    config = bare_abc_config(
        model=make_config_data(),
        condition=condition,
        condition_method="ensemble_mean",
        effective_condition=condition,
    )

    with pytest.raises(
        ValueError,
        match="Ensemble mean must be True",
    ):
        config._check_condition()


def test_ensemble_mean_condition_valid():
    condition = make_config_data(
        paths=("condition.nc",),
        ensemble_mean=True,
    )

    config = bare_abc_config(
        model=make_config_data(),
        condition=condition,
        condition_method="ensemble_mean",
        effective_condition=condition,
    )

    assert config._check_condition() is config


def test_static_condition_rejects_ensemble_list():
    condition = make_config_data(
        paths=("condition.nc",),
        ensemble_list=("r1",),
    )

    config = bare_abc_config(
        model=make_config_data(),
        condition=condition,
        condition_method="static",
        effective_condition=condition,
    )

    with pytest.raises(
        ValueError,
        match="cannot specify ensemble list",
    ):
        config._check_condition()


def test_static_condition_rejects_model_source():
    model = make_config_data(
        ensemble_list=None,
    )

    config = bare_abc_config(
        model=model,
        condition=model,
        condition_method="static",
        effective_condition=model,
    )

    with pytest.raises(
        ValueError,
        match="cannot point to the same model data",
    ):
        config._check_condition()


def test_static_condition_valid():
    model = make_config_data()
    condition = make_config_data(
        paths=("condition.nc",),
        ensemble_list=None,
    )

    condition.info.coords = {}
    config = bare_abc_config(
        model=model,
        condition=condition,
        condition_method="static",
        effective_condition=condition,
    )

    assert config._check_condition() is config


def test_model_condition_validation_skipped_without_model():
    config = bare_abc_config(
        model=None,
        condition=make_config_data(),
        condition_method="static",
    )

    assert config._check_model_vs_condition() is None


@pytest.mark.pruned
def test_model_condition_validation_skipped_same_source():
    model = make_config_data()

    config = bare_abc_config(
        model=model,
        condition=model,
        condition_method="ensemble_mean",
        effective_condition=model,
    )

    assert config._check_model_vs_condition() is None


@pytest.mark.pruned
def test_static_condition_skips_sample_coordinate_checks():
    model = make_config_data(
        paths=("model.nc",),
        times=(2000, 2001),
        lead_times=(1, 2),
    )
    condition = make_config_data(
        paths=("condition.nc",),
        times=(1900,),
        lead_times=(12,),
        ensemble_list=None,
    )

    config = bare_abc_config(
        model=model,
        condition=condition,
        condition_method="static",
        effective_condition=condition,
    )

    assert config._check_model_vs_condition() is None


@pytest.mark.pruned
def test_condition_missing_required_sample_dimension():
    model = make_config_data(
        paths=("model.nc",),
    )
    condition = make_config_data(
        paths=("condition.nc",),
    )

    del condition.info.coords["lead_time"]

    config = bare_abc_config(
        model=model,
        condition=condition,
        condition_method="cross_ensemble",
        effective_condition=condition,
    )

    with pytest.raises(
        ValueError,
        match="same dimestions",
    ):
        config._check_model_vs_condition()


@pytest.mark.pruned
def test_condition_missing_model_coordinate_values():
    model = make_config_data(
        paths=("model.nc",),
        lead_times=(1, 2, 3),
    )
    condition = make_config_data(
        paths=("condition.nc",),
        lead_times=(1, 2),
    )

    config = bare_abc_config(
        model=model,
        condition=condition,
        condition_method="cross_ensemble",
        effective_condition=condition,
    )

    with pytest.raises(
        ValueError,
        match="same lead_time coordinates",
    ):
        config._check_model_vs_condition()


@pytest.mark.pruned
def test_condition_with_superset_coordinates_passes():
    model = make_config_data(
        paths=("model.nc",),
        times=(2000, 2001),
        lead_times=(1, 2),
    )
    condition = make_config_data(
        paths=("condition.nc",),
        times=(1999, 2000, 2001, 2002),
        lead_times=(1, 2, 3),
    )

    config = bare_abc_config(
        model=model,
        condition=condition,
        condition_method="cross_ensemble",
        effective_condition=condition,
    )

    assert config._check_model_vs_condition() is None


def test_same_member_requires_model_ensembles():
    model = make_config_data(
        paths=("model.nc",),
        ensembles=None,
        ensemble_list=None,
    )
    condition = make_config_data(
        paths=("condition.nc",),
    )

    config = bare_abc_config(
        model=model,
        condition=condition,
        condition_method="same_member",
        effective_condition=condition,
    )

    with pytest.raises(
        ValueError,
        match="must have ensembles",
    ):
        config._check_model_vs_condition()


@pytest.mark.pruned
def test_same_member_requires_condition_ensembles():
    model = make_config_data(
        paths=("model.nc",),
    )
    condition = make_config_data(
        paths=("condition.nc",),
        ensembles=None,
        ensemble_list=None,
    )

    config = bare_abc_config(
        model=model,
        condition=condition,
        condition_method="same_member",
        effective_condition=condition,
    )

    with pytest.raises(
        ValueError,
        match="must have ensembles",
    ):
        config._check_model_vs_condition()


def test_same_member_rejects_different_ensembles():
    model = make_config_data(
        paths=("model.nc",),
        ensembles=("r1", "r2"),
    )
    condition = make_config_data(
        paths=("condition.nc",),
        ensembles=("r1", "r3"),
    )

    config = bare_abc_config(
        model=model,
        condition=condition,
        condition_method="same_member",
        effective_condition=condition,
    )

    with pytest.raises(
        ValueError,
        match="same ensemble members",
    ):
        config._check_model_vs_condition()


def test_same_member_accepts_equal_ensembles():
    model = make_config_data(
        paths=("model.nc",),
        ensembles=("r1", "r2"),
    )
    condition = make_config_data(
        paths=("condition.nc",),
        ensembles=("r1", "r2"),
    )

    config = bare_abc_config(
        model=model,
        condition=condition,
        condition_method="same_member",
        effective_condition=condition,
    )

    assert config._check_model_vs_condition() is None


@pytest.mark.parametrize(
    (
        "feature",
        "expected",
    ),
    [
        ("lead_time", 0.5),
        ("month_sin", 1.0),
        ("month_cos", 0.0),
    ],
)
def test_individual_time_feature_values(
    feature,
    expected,
):
    features = AddedTimeFeatures(
        make_reference_config(),
        [feature],
    )

    result = features(
        {
            "year": 2000,
            "lead_time": 3,
        },
        xr.DataArray(np.ones((2,))),
    )

    assert result.shape == (1,)
    assert result[0] == pytest.approx(
        expected,
        abs=1e-12,
    )


@pytest.mark.pruned
def test_year_feature_value():
    features = AddedTimeFeatures(
        make_reference_config(),
        ["year"],
    )

    result = features(
        {
            "year": 2000,
            "lead_time": 1,
        },
        xr.DataArray(np.ones((2,))),
    )

    assert np.isfinite(result[0])


@pytest.mark.pruned
def test_added_time_features_two_dimensional_input_not_broadcast():
    features = AddedTimeFeatures(
        make_reference_config(),
        ["year", "lead_time"],
    )

    result = features(
        {
            "year": 2000,
            "lead_time": 1,
        },
        xr.DataArray(
            np.ones((3, 4)),
            dims=("channels", "features"),
        ),
    )

    assert result.shape == (2,)


@pytest.mark.pruned
def test_added_time_features_four_dimensional_broadcast():
    features = AddedTimeFeatures(
        make_reference_config(),
        ["year", "lead_time"],
    )

    result = features(
        {
            "year": 2000,
            "lead_time": 1,
        },
        xr.DataArray(
            np.ones((2, 3, 4, 5)),
            dims=(
                "channels",
                "level",
                "lat",
                "lon",
            ),
        ),
    )

    assert result.shape == (2, 3, 4, 5)


class DifferentReference:
    def __init__(self):
        self.lead_months = np.asarray([1, 2, 3])
        self.get_common_time = np.asarray([2000, 2001, 2002])


@pytest.mark.pruned
def test_added_time_features_different_reference_type():
    left = AddedTimeFeatures(
        make_reference_config(
            lead_months=(1, 2, 3),
        ),
        ["year"],
    )
    right = AddedTimeFeatures(
        DifferentReference(),
        ["year"],
    )

    assert left != right


@pytest.mark.pruned
def test_added_time_features_equal_multiple_features():
    left = AddedTimeFeatures(
        make_reference_config(),
        ["year", "lead_time"],
    )
    right = AddedTimeFeatures(
        make_reference_config(),
        ["year", "lead_time"],
    )

    assert left == right


def test_prepare_sampling_mask_converts_false_to_non_nan():
    dataset = bare_dataset(
        config=make_dataset_config(),
        mask=make_mask(
            values=np.asarray(
                [
                    [False, True],
                    [False, False],
                ]
            )
        ),
    )

    dataset._prepare_sampling_mask(
        {
            "year": [2000, 2001],
            "lead_time": [1, 2],
        }
    )

    assert np.isnan(
        dataset.mask.sel(
            year=2000,
            lead_time=2,
        ).item()
    )
    assert not np.isnan(
        dataset.mask.sel(
            year=2000,
            lead_time=1,
        ).item()
    )


@pytest.mark.pruned
def test_get_model_indexes_multiple_missing_dimensions():
    dataset = bare_dataset(
        _load_model_value=True,
        model_dataset=make_xarray(),
    )

    with pytest.raises(
        ValueError,
        match="model dataset",
    ) as error:
        dataset.get_model_indexes(
            {
                "year": np.asarray([1999]),
                "lead_time": np.asarray([9]),
            }
        )

    message = str(error.value)

    assert "year" in message
    assert "lead_time" in message


@pytest.mark.pruned
def test_get_cond_indexes_ignores_ensemble_for_non_same_member():
    dataset = bare_dataset(
        condition_dataset=make_xarray(
            ensembles=("r1", "r2"),
        ),
        config=SimpleNamespace(
            condition_method="cross_ensemble",
        ),
    )

    result = dataset.get_cond_indexes(
        {
            "year": np.asarray([2000]),
            "lead_time": np.asarray([1]),
            "ensembles": np.asarray(["r2"]),
        }
    )

    assert "ensembles" not in result


@pytest.mark.pruned
def test_get_cond_indexes_multiple_missing_dimensions():
    dataset = bare_dataset(
        condition_dataset=make_xarray(),
        config=SimpleNamespace(
            condition_method="ensemble_mean",
        ),
    )

    with pytest.raises(
        ValueError,
        match="conditioning dataset",
    ) as error:
        dataset.get_cond_indexes(
            {
                "year": np.asarray([1999]),
                "lead_time": np.asarray([9]),
            }
        )

    message = str(error.value)

    assert "year" in message
    assert "lead_time" in message


@pytest.mark.pruned
def test_get_input_shape_without_supported_dimensions(
    monkeypatch,
):
    class FakeFlattennanremove:
        pass

    monkeypatch.setattr(
        module,
        "supported_NN_dimensions_sorted",
        ["lat", "lon"],
    )

    monkeypatch.setitem(
        __import__("sys").modules,
        ("cccma_ppp.preprocessing.utils_preprocessing"),
        SimpleNamespace(Flattennanremove=FakeFlattennanremove),
    )

    effective_input = make_config_data()

    dataset = bare_dataset(
        config=SimpleNamespace(
            effective_input=effective_input,
            effective_condition=None,
        ),
        _concat_condition_value=False,
    )

    assert dataset.get_input_shape() == (1,)


@pytest.mark.pruned
def test_get_input_shape_flattener_without_condition_concat(
    monkeypatch,
):
    class FakeFlattennanremove:
        pass

    flattener = FakeFlattennanremove()
    flattener.final_locations = np.zeros(4)

    pipeline = make_pipeline(
        fitted_preprocessors=[flattener],
    )
    pipeline.get_preprocessors.return_value = flattener

    monkeypatch.setitem(
        __import__("sys").modules,
        ("cccma_ppp.preprocessing.utils_preprocessing"),
        SimpleNamespace(Flattennanremove=FakeFlattennanremove),
    )

    dataset = bare_dataset(
        config=SimpleNamespace(
            effective_input=make_config_data(
                names=("tas", "pr"),
                pipeline=pipeline,
            ),
            effective_condition=None,
        ),
        _concat_condition_value=False,
    )

    assert dataset.get_input_shape() == (2, 4)


@pytest.mark.pruned
def test_index_condition_uses_requested_index():
    pipeline = make_pipeline()

    dataset = bare_dataset(
        condition_dataset=make_xarray(
            times=(2000, 2001),
            lead_times=(1, 2),
        ),
        cond_indexes={
            "year": np.asarray([0, 1]),
            "lead_time": np.asarray([1, 0]),
        },
        config=SimpleNamespace(
            condition_method="ensemble_mean",
            effective_condition=make_config_data(
                pipeline=pipeline,
            ),
        ),
    )

    with patch.object(
        module,
        "_unwrap_data_variables",
        side_effect=lambda value: value,
    ):
        result = dataset._index_condition_dataset(1)

    np.testing.assert_array_equal(
        result.year.values,
        [2001],
    )
    np.testing.assert_array_equal(
        result.lead_time.values,
        [1],
    )


@pytest.mark.pruned
def test_index_model_uses_requested_index():
    pipeline = make_pipeline()

    dataset = bare_dataset(
        _load_model_value=True,
        model_dataset=make_xarray(
            times=(2000, 2001),
            lead_times=(1, 2),
        ),
        model_indexes={
            "year": np.asarray([0, 1]),
            "lead_time": np.asarray([1, 0]),
        },
        config=SimpleNamespace(
            model=make_config_data(
                pipeline=pipeline,
            ),
        ),
    )

    with patch.object(
        module,
        "_unwrap_data_variables",
        side_effect=lambda value: value,
    ):
        result = dataset._index_model_dataset(1)

    np.testing.assert_array_equal(
        result.year.values,
        [2001],
    )
    np.testing.assert_array_equal(
        result.lead_time.values,
        [1],
    )


def test_dataset_init_loads_model_and_condition(
    monkeypatch,
):
    config = make_dataset_config(
        model=make_config_data(
            paths=("model.nc",),
        ),
        effective_condition=make_config_data(
            paths=("condition.nc",),
        ),
        condition_method="ensemble_mean",
    )

    dataset = object.__new__(BareDataset)
    dataset.config = config
    dataset.requested_years = [2000]
    dataset.mask = make_mask()
    dataset.load = True
    dataset._load_model_value = True

    loaded_model = make_xarray()
    loaded_condition = make_xarray()

    monkeypatch.setattr(
        dataset,
        "_check_init",
        MagicMock(),
    )
    monkeypatch.setattr(
        dataset,
        "_resolve_mask",
        MagicMock(),
    )
    monkeypatch.setattr(
        dataset,
        "_prepare_sampling_mask",
        MagicMock(),
    )
    monkeypatch.setattr(
        dataset,
        "_load_xarray_data",
        MagicMock(
            side_effect=[
                loaded_model,
                loaded_condition,
            ]
        ),
    )
    monkeypatch.setattr(
        dataset,
        "get_sampling_coords",
        MagicMock(
            return_value={
                "year": np.asarray([2000]),
                "lead_time": np.asarray([1]),
            }
        ),
    )
    monkeypatch.setattr(
        dataset,
        "get_model_indexes",
        MagicMock(
            return_value={
                "year": np.asarray([0]),
                "lead_time": np.asarray([0]),
            }
        ),
    )
    monkeypatch.setattr(
        dataset,
        "get_cond_indexes",
        MagicMock(
            return_value={
                "year": np.asarray([0]),
                "lead_time": np.asarray([0]),
            }
        ),
    )

    DatasetABC.__init__(dataset)

    assert dataset.model_dataset is loaded_model
    assert dataset.condition_dataset is loaded_condition
    assert dataset.observation_dataset is None

    assert dataset._load_xarray_data.call_count == 2


def test_dataset_init_skips_model_and_condition(
    monkeypatch,
):
    config = make_dataset_config(
        model=None,
        effective_condition=None,
        condition_method=None,
    )

    dataset = object.__new__(BareDataset)
    dataset.config = config
    dataset.requested_years = [2000]
    dataset.mask = make_mask()
    dataset.load = False
    dataset._load_model_value = False

    monkeypatch.setattr(
        dataset,
        "_check_init",
        MagicMock(),
    )
    monkeypatch.setattr(
        dataset,
        "_resolve_mask",
        MagicMock(),
    )
    monkeypatch.setattr(
        dataset,
        "_prepare_sampling_mask",
        MagicMock(),
    )
    monkeypatch.setattr(
        dataset,
        "_load_xarray_data",
        MagicMock(),
    )
    monkeypatch.setattr(
        dataset,
        "get_sampling_coords",
        MagicMock(
            return_value={
                "year": np.asarray([2000]),
                "lead_time": np.asarray([1]),
            }
        ),
    )
    monkeypatch.setattr(
        dataset,
        "get_model_indexes",
        MagicMock(return_value=None),
    )
    monkeypatch.setattr(
        dataset,
        "get_cond_indexes",
        MagicMock(return_value=None),
    )

    DatasetABC.__init__(dataset)

    assert dataset.model_dataset is None
    assert dataset.condition_dataset is None
    assert dataset.observation_dataset is None
    dataset._load_xarray_data.assert_not_called()


@pytest.mark.pruned
def test_dataset_init_loads_condition_without_model(
    monkeypatch,
):
    condition = make_config_data(
        paths=("condition.nc",),
    )

    config = make_dataset_config(
        model=None,
        effective_condition=condition,
        condition_method="static",
    )

    dataset = object.__new__(BareDataset)
    dataset.config = config
    dataset.requested_years = [2000]
    dataset.mask = make_mask()
    dataset.load = False
    dataset._load_model_value = False

    loaded_condition = make_xarray()

    monkeypatch.setattr(
        dataset,
        "_check_init",
        MagicMock(),
    )
    monkeypatch.setattr(
        dataset,
        "_resolve_mask",
        MagicMock(),
    )
    monkeypatch.setattr(
        dataset,
        "_prepare_sampling_mask",
        MagicMock(),
    )
    monkeypatch.setattr(
        dataset,
        "_load_xarray_data",
        MagicMock(return_value=loaded_condition),
    )
    monkeypatch.setattr(
        dataset,
        "get_sampling_coords",
        MagicMock(
            return_value={
                "year": np.asarray([2000]),
                "lead_time": np.asarray([1]),
            }
        ),
    )
    monkeypatch.setattr(
        dataset,
        "get_model_indexes",
        MagicMock(return_value=None),
    )
    monkeypatch.setattr(
        dataset,
        "get_cond_indexes",
        MagicMock(return_value=None),
    )

    DatasetABC.__init__(dataset)

    assert dataset.model_dataset is None
    assert dataset.condition_dataset is loaded_condition
    assert dataset._load_xarray_data.call_count == 1


@pytest.mark.pruned
def test_lead_months_explicit_list_takes_precedence_over_range():
    config = lead_months_config(
        list_months=[2, 6],
        start=1,
        end=12,
    )

    assert config.build_lead_months() == [2, 6]


@pytest.mark.pruned
def test_resolve_mask_rejects_missing_year_dimension():
    dataset = bare_dataset(
        config=make_dataset_config(),
        mask=xr.DataArray(
            [False],
            dims=("lead_time",),
            coords={"lead_time": [1]},
        ),
    )

    with pytest.raises(
        ValueError,
        match="mask must have",
    ):
        dataset._resolve_mask()


@pytest.mark.pruned
def test_resolve_mask_rejects_all_missing_dimensions():
    dataset = bare_dataset(
        config=make_dataset_config(),
        mask=xr.DataArray(False),
    )

    with pytest.raises(
        ValueError,
        match="mask must have",
    ) as error:
        dataset._resolve_mask()

    assert "year" in str(error.value)
    assert "lead_time" in str(error.value)


@pytest.mark.pruned
def test_prepare_sampling_mask_empty_year_selection():
    dataset = bare_dataset(
        config=make_dataset_config(),
        mask=make_mask(
            values=np.zeros((2, 2), dtype=bool),
        ),
    )

    dataset._prepare_sampling_mask(
        {
            "year": [],
            "lead_time": [1, 2],
        }
    )

    assert bool(dataset.mask.isnull().all())


@pytest.mark.pruned
def test_prepare_sampling_mask_empty_lead_time_selection():
    dataset = bare_dataset(
        config=make_dataset_config(),
        mask=make_mask(
            values=np.zeros((2, 2), dtype=bool),
        ),
    )

    dataset._prepare_sampling_mask(
        {
            "year": [2000, 2001],
            "lead_time": [],
        }
    )

    assert bool(dataset.mask.isnull().all())


@pytest.mark.pruned
def test_get_model_indexes_empty_coordinates():
    dataset = bare_dataset(
        _load_model_value=True,
        model_dataset=make_xarray(),
    )

    result = dataset.get_model_indexes(
        {
            "year": np.asarray([], dtype=int),
            "lead_time": np.asarray([], dtype=int),
        }
    )

    assert result["year"].size == 0
    assert result["lead_time"].size == 0


@pytest.mark.pruned
def test_get_cond_indexes_empty_coordinates():
    dataset = bare_dataset(
        condition_dataset=make_xarray(),
        config=SimpleNamespace(
            condition_method="ensemble_mean",
        ),
    )

    result = dataset.get_cond_indexes(
        {
            "year": np.asarray([], dtype=int),
            "lead_time": np.asarray([], dtype=int),
        }
    )

    assert result["year"].size == 0
    assert result["lead_time"].size == 0


@pytest.mark.pruned
def test_get_cond_indexes_same_member_missing_ensemble_value():
    dataset = bare_dataset(
        condition_dataset=make_xarray(
            ensembles=("r1", "r2"),
        ),
        config=SimpleNamespace(
            condition_method="same_member",
        ),
    )

    with pytest.raises(
        ValueError,
        match="conditioning coordinates were not found",
    ):
        dataset.get_cond_indexes(
            {
                "year": np.asarray([2000]),
                "lead_time": np.asarray([1]),
                "ensembles": np.asarray(["r9"]),
            }
        )


@pytest.mark.pruned
def test_get_cond_indexes_same_member_multiple_samples():
    dataset = bare_dataset(
        condition_dataset=make_xarray(
            ensembles=("r1", "r2"),
        ),
        config=SimpleNamespace(
            condition_method="same_member",
        ),
    )

    result = dataset.get_cond_indexes(
        {
            "year": np.asarray([2000, 2001]),
            "lead_time": np.asarray([2, 1]),
            "ensembles": np.asarray(["r2", "r1"]),
        }
    )

    np.testing.assert_array_equal(
        result["year"],
        [0, 1],
    )
    np.testing.assert_array_equal(
        result["lead_time"],
        [1, 0],
    )
    np.testing.assert_array_equal(
        result["ensembles"],
        [1, 0],
    )


@pytest.mark.pruned
def test_index_condition_same_member():
    pipeline = make_pipeline()

    dataset = bare_dataset(
        condition_dataset=make_xarray(
            ensembles=("r1", "r2"),
        ),
        cond_indexes={
            "year": np.asarray([0]),
            "lead_time": np.asarray([1]),
            "ensembles": np.asarray([1]),
        },
        config=SimpleNamespace(
            condition_method="same_member",
            effective_condition=make_config_data(
                ensembles=("r1", "r2"),
                pipeline=pipeline,
            ),
        ),
    )

    with patch.object(
        module,
        "_unwrap_data_variables",
        side_effect=lambda value: value,
    ):
        result = dataset._index_condition_dataset(0)

    assert result.sizes["year"] == 1
    assert result.sizes["lead_time"] == 1
    assert result.sizes["ensembles"] == 1
    assert result.coords["ensembles"].values[0] == "r2"


@pytest.mark.pruned
def test_index_cross_ensemble_condition_single_member(
    monkeypatch,
):
    pipeline = make_pipeline()

    dataset = bare_dataset(
        condition_dataset=make_xarray(
            ensembles=("r1",),
        ),
        cond_indexes={
            "year": np.asarray([0]),
            "lead_time": np.asarray([0]),
        },
        config=SimpleNamespace(
            condition_method="cross_ensemble",
            effective_condition=make_config_data(
                ensembles=("r1",),
                pipeline=pipeline,
            ),
        ),
    )

    randint = MagicMock(return_value=0)
    monkeypatch.setattr(
        module.np.random,
        "randint",
        randint,
    )

    with patch.object(
        module,
        "_unwrap_data_variables",
        side_effect=lambda value: value,
    ):
        result = dataset._index_condition_dataset(0)

    randint.assert_called_once_with(1)
    assert result.sizes["ensembles"] == 1
    assert result.coords["ensembles"].values[0] == "r1"


@pytest.mark.pruned
def test_index_model_with_ensemble_index():
    pipeline = make_pipeline()

    dataset = bare_dataset(
        _load_model_value=True,
        model_dataset=make_xarray(
            ensembles=("r1", "r2"),
        ),
        model_indexes={
            "year": np.asarray([0]),
            "lead_time": np.asarray([1]),
            "ensembles": np.asarray([1]),
        },
        config=SimpleNamespace(
            model=make_config_data(
                ensembles=("r1", "r2"),
                pipeline=pipeline,
            ),
        ),
    )

    with patch.object(
        module,
        "_unwrap_data_variables",
        side_effect=lambda value: value,
    ):
        result = dataset._index_model_dataset(0)

    assert result.sizes["year"] == 1
    assert result.sizes["lead_time"] == 1
    assert result.sizes["ensembles"] == 1
    assert result.coords["ensembles"].values[0] == "r2"


@pytest.mark.pruned
def test_check_init_accepts_empty_requested_years():
    dataset = bare_dataset(
        config=make_dataset_config(
            available_times=[2000, 2001],
        ),
        requested_years=[],
    )

    assert dataset._check_init() is None


@pytest.mark.pruned
def test_added_time_features_empty_returns_empty_array():
    features = AddedTimeFeatures(
        make_reference_config(),
        [],
    )

    result = features(
        {
            "year": 2000,
            "lead_time": 1,
        },
        xr.DataArray(np.ones((2,))),
    )

    assert result.shape == (0,)
    assert result.dtype.kind == "f"


@pytest.mark.pruned
def test_added_time_features_empty_with_spatial_input():
    features = AddedTimeFeatures(
        make_reference_config(),
        None,
    )

    result = features(
        {
            "year": 2000,
            "lead_time": 1,
        },
        xr.DataArray(
            np.ones((2, 3, 4)),
            dims=("channels", "lat", "lon"),
        ),
    )

    assert result.shape == (0, 3, 4)


@pytest.mark.pruned
def test_using_model_as_condition_different_rename_dict():
    model = make_config_data()
    condition = make_config_data()
    condition.rename_dict = {"other": "value"}

    config = bare_abc_config(
        model=model,
        condition=condition,
        condition_method=None,
    )

    assert config._using_model_data_as_condition is True


@pytest.mark.pruned
def test_model_condition_check_skips_static_before_coordinate_access():
    model = make_config_data(
        paths=("model.nc",),
    )
    condition = make_config_data(
        paths=("condition.nc",),
        ensemble_list=None,
    )
    condition.info = None

    config = bare_abc_config(
        model=model,
        condition=condition,
        condition_method="static",
        effective_condition=condition,
    )

    assert config._check_model_vs_condition() is None


@pytest.mark.pruned
def test_same_member_accepts_ensemble_coordinates_in_different_order():
    model = make_config_data(
        paths=("model.nc",),
        ensembles=("r1", "r2"),
    )
    condition = make_config_data(
        paths=("condition.nc",),
        ensembles=("r2", "r1"),
    )

    config = bare_abc_config(
        model=model,
        condition=condition,
        condition_method="same_member",
        effective_condition=condition,
    )

    with pytest.raises(
        ValueError,
        match="same ensemble members",
    ):
        config._check_model_vs_condition()


@pytest.mark.pruned
def test_dataset_init_forwards_false_load_flag(
    monkeypatch,
):
    config = make_dataset_config(
        model=make_config_data(
            paths=("model.nc",),
        ),
        effective_condition=None,
        condition_method=None,
    )

    dataset = object.__new__(BareDataset)
    dataset.config = config
    dataset.requested_years = [2000]
    dataset.mask = make_mask()
    dataset.load = False
    dataset._load_model_value = True

    loaded_model = make_xarray()

    monkeypatch.setattr(dataset, "_check_init", MagicMock())
    monkeypatch.setattr(dataset, "_resolve_mask", MagicMock())
    monkeypatch.setattr(dataset, "_prepare_sampling_mask", MagicMock())
    monkeypatch.setattr(
        dataset,
        "_load_xarray_data",
        MagicMock(return_value=loaded_model),
    )
    monkeypatch.setattr(
        dataset,
        "get_sampling_coords",
        MagicMock(
            return_value={
                "year": np.asarray([2000]),
                "lead_time": np.asarray([1]),
            }
        ),
    )
    monkeypatch.setattr(
        dataset,
        "get_model_indexes",
        MagicMock(return_value=None),
    )
    monkeypatch.setattr(
        dataset,
        "get_cond_indexes",
        MagicMock(return_value=None),
    )

    DatasetABC.__init__(dataset)

    dataset._load_xarray_data.assert_called_once_with(
        config.model,
        load=False,
    )
    assert dataset.model_dataset is loaded_model
    assert dataset.condition_dataset is None


@pytest.mark.pruned
def test_dataset_init_condition_only_forwards_true_load_flag(
    monkeypatch,
):
    condition = make_config_data(
        paths=("condition.nc",),
    )
    config = make_dataset_config(
        model=None,
        effective_condition=condition,
        condition_method="static",
    )

    dataset = object.__new__(BareDataset)
    dataset.config = config
    dataset.requested_years = [2000]
    dataset.mask = make_mask()
    dataset.load = True
    dataset._load_model_value = False

    loaded_condition = make_xarray()

    monkeypatch.setattr(dataset, "_check_init", MagicMock())
    monkeypatch.setattr(dataset, "_resolve_mask", MagicMock())
    monkeypatch.setattr(dataset, "_prepare_sampling_mask", MagicMock())
    monkeypatch.setattr(
        dataset,
        "_load_xarray_data",
        MagicMock(return_value=loaded_condition),
    )
    monkeypatch.setattr(
        dataset,
        "get_sampling_coords",
        MagicMock(
            return_value={
                "year": np.asarray([2000]),
                "lead_time": np.asarray([1]),
            }
        ),
    )
    monkeypatch.setattr(
        dataset,
        "get_model_indexes",
        MagicMock(return_value=None),
    )
    monkeypatch.setattr(
        dataset,
        "get_cond_indexes",
        MagicMock(return_value=None),
    )

    DatasetABC.__init__(dataset)

    dataset._load_xarray_data.assert_called_once_with(
        condition,
        load=True,
    )
    assert dataset.model_dataset is None
    assert dataset.condition_dataset is loaded_condition


@pytest.mark.parametrize(
    ("list_months", "start", "end", "expected"),
    [
        ([1], 1, None, [1]),
        ([2, 4, 8], 1, None, [2, 4, 8]),
        (None, 1, 1, [1]),
        (None, 0, 2, [0, 1, 2]),
        (None, -2, 0, [-2, -1, 0]),
        (None, 4, 2, []),
    ],
)
def test_lead_months_additional_build_cases(
    list_months,
    start,
    end,
    expected,
):
    config = lead_months_config(
        list_months=list_months,
        start=start,
        end=end,
    )

    np.testing.assert_array_equal(
        config.build_lead_months(),
        expected,
    )


@pytest.mark.pruned
def test_lead_months_list_takes_precedence_over_range():
    config = lead_months_config(
        list_months=[3, 9],
        start=1,
        end=12,
    )

    assert config.build_lead_months() == [3, 9]


@pytest.mark.pruned
def test_lead_months_empty_list_falls_back_to_range():
    config = lead_months_config(
        list_months=[],
        start=2,
        end=4,
    )

    np.testing.assert_array_equal(
        config.build_lead_months(),
        [2, 3, 4],
    )


@pytest.mark.pruned
def test_lead_months_empty_list_without_end_fails_when_built():
    config = lead_months_config(
        list_months=[],
        end=None,
    )

    with pytest.raises(TypeError):
        config.build_lead_months()


@pytest.mark.pruned
def test_check_condition_method_none_avoids_membership_check():
    config = bare_abc_config(
        condition_method=None,
    )

    assert config._check_condition_method() is config


@pytest.mark.parametrize(
    "condition_method",
    sorted(DatasetConfigABC._VALID_CONDITION_METHODS),
)
def test_check_condition_method_accepts_every_supported_value(
    condition_method,
):
    config = bare_abc_config(
        condition_method=condition_method,
    )

    assert config._check_condition_method() is config


@pytest.mark.parametrize(
    "condition_method",
    [
        "",
        "same-member",
        "unknown",
    ],
)
def test_check_condition_method_rejects_additional_invalid_values(
    condition_method,
):
    config = bare_abc_config(
        condition_method=condition_method,
    )

    with pytest.raises(
        ValueError,
        match="Invalid condition_method",
    ):
        config._check_condition_method()


@pytest.mark.parametrize(
    ("condition_method", "expected"),
    [
        (None, False),
        ("static", False),
        ("ensemble_mean", True),
        ("cross_ensemble", True),
        ("same_member", True),
        ("unsupported", False),
    ],
)
def test_using_model_as_condition_without_explicit_condition(
    condition_method,
    expected,
):
    config = bare_abc_config(
        model=make_config_data(),
        condition=None,
        condition_method=condition_method or "static",
    )

    assert config._using_model_data_as_condition is expected


@pytest.mark.pruned
def test_using_model_as_condition_condition_present_model_missing():
    config = bare_abc_config(
        model=None,
        condition=make_config_data(
            paths=("condition.nc",),
        ),
        condition_method="ensemble_mean",
    )

    assert config._using_model_data_as_condition is False


@pytest.mark.parametrize(
    ("same_paths", "same_names", "same_ensembles", "expected"),
    [
        (True, True, True, True),
        (False, True, True, False),
        (True, False, True, False),
        (True, True, False, False),
        (False, False, False, False),
    ],
)
def test_using_model_as_condition_comparison_matrix(
    same_paths,
    same_names,
    same_ensembles,
    expected,
):
    model = make_config_data(
        paths=("model.nc",),
        names=("tas",),
        ensemble_list=("r1", "r2"),
    )
    condition = make_config_data(
        paths=(("model.nc",) if same_paths else ("condition.nc",)),
        names=(("tas",) if same_names else ("pr",)),
        ensemble_list=(("r1", "r2") if same_ensembles else ("r3",)),
    )

    config = bare_abc_config(
        model=model,
        condition=condition,
        condition_method="ensemble_mean",
    )

    assert config._using_model_data_as_condition is expected


@pytest.mark.pruned
def test_resolve_condition_replaces_stale_value_with_explicit_condition():
    explicit_condition = make_config_data(
        paths=("condition.nc",),
    )
    config = bare_abc_config(
        model=make_config_data(),
        condition=explicit_condition,
        condition_method="static",
        effective_condition=object(),
    )

    result = config._resolve_condition()

    assert result is config
    assert config.effective_condition is explicit_condition


@pytest.mark.parametrize(
    ("condition_method", "expected_ensemble_mean"),
    [
        ("ensemble_mean", True),
        ("cross_ensemble", False),
        ("same_member", False),
    ],
)
def test_model_as_condition_ensemble_mean_branch(
    condition_method,
    expected_ensemble_mean,
):
    model = make_config_data()
    config = bare_abc_config(
        model=model,
        condition=None,
        condition_method=condition_method,
    )

    expected = object()

    with patch.object(
        module,
        "ModelDataConfig",
        return_value=expected,
    ) as constructor:
        result = config._model_as_condition()

    assert result is expected
    assert constructor.call_args.kwargs["ensemble_mean"] is (expected_ensemble_mean)


@pytest.mark.pruned
def test_check_model_none_skips_same_member_validation():
    config = bare_abc_config(
        model=None,
        condition_method="same_member",
    )

    assert config._check_model() is config


@pytest.mark.pruned
def test_check_model_non_same_member_skips_ensemble_mean_validation():
    config = bare_abc_config(
        model=make_config_data(
            ensemble_mean=True,
        ),
        condition_method="ensemble_mean",
    )

    assert config._check_model() is config


@pytest.mark.pruned
def test_check_model_same_member_accepts_non_mean_model():
    config = bare_abc_config(
        model=make_config_data(
            ensemble_mean=False,
        ),
        condition_method="same_member",
    )

    assert config._check_model() is config


@pytest.mark.pruned
def test_check_model_same_member_rejects_mean_model():
    config = bare_abc_config(
        model=make_config_data(
            ensemble_mean=True,
        ),
        condition_method="same_member",
    )

    with pytest.raises(
        ValueError,
        match="should not be ensemble mean",
    ):
        config._check_model()


@pytest.mark.parametrize(
    "condition_method",
    [
        None,
        "cross_ensemble",
        "same_member",
    ],
)
def test_check_condition_without_effective_condition_nonstatic(
    condition_method,
):
    config = bare_abc_config(
        model=make_config_data(),
        condition=None,
        condition_method=condition_method or "cross_ensemble",
        effective_condition=None,
    )

    assert config._check_condition() is config


@pytest.mark.pruned
def test_check_condition_static_without_effective_condition_raises():
    config = bare_abc_config(
        model=make_config_data(),
        condition=None,
        condition_method="static",
        effective_condition=None,
    )

    with pytest.raises(
        ValueError,
        match="condition dataset must be specified",
    ):
        config._check_condition()


@pytest.mark.parametrize(
    "condition_method",
    [
        "cross_ensemble",
        "same_member",
    ],
)
def test_check_condition_checks_mean_before_missing_ensemble_dimension(
    condition_method,
):
    condition = make_config_data(
        paths=("condition.nc",),
        ensembles=None,
        ensemble_list=None,
        ensemble_mean=True,
    )
    config = bare_abc_config(
        model=make_config_data(),
        condition=condition,
        condition_method=condition_method,
        effective_condition=condition,
    )

    with pytest.raises(
        ValueError,
        match="ensemble_mean cannot be True",
    ):
        config._check_condition()


@pytest.mark.pruned
def test_check_condition_static_checks_ensemble_list_before_same_source():
    model = make_config_data(
        ensemble_list=("r1",),
    )
    config = bare_abc_config(
        model=model,
        condition=model,
        condition_method="static",
        effective_condition=model,
    )

    with pytest.raises(
        ValueError,
        match="cannot specify ensemble list",
    ):
        config._check_condition()


@pytest.mark.pruned
def test_check_condition_unknown_method_uses_static_validation_branch():
    condition = make_config_data(
        paths=("condition.nc",),
        ensemble_list=None,
    )
    config = bare_abc_config(
        model=make_config_data(),
        condition=condition,
        condition_method="unknown",
        effective_condition=condition,
    )

    assert config._check_condition() is config


@pytest.mark.pruned
def test_model_vs_condition_model_none_short_circuits():
    condition = make_config_data(
        paths=("condition.nc",),
    )
    condition.info = None

    config = bare_abc_config(
        model=None,
        condition=condition,
        condition_method="static",
        effective_condition=condition,
    )

    assert config._check_model_vs_condition() is None


@pytest.mark.pruned
def test_model_vs_condition_same_source_short_circuits():
    model = make_config_data()

    config = bare_abc_config(
        model=model,
        condition=model,
        condition_method="same_member",
        effective_condition=model,
    )

    assert config._using_model_data_as_condition is True
    assert config._check_model_vs_condition() is None


@pytest.mark.pruned
def test_model_vs_condition_static_skips_sample_coordinate_validation():
    model = make_config_data(
        paths=("model.nc",),
    )
    condition = make_config_data(
        paths=("condition.nc",),
        times=(1900,),
        lead_times=(99,),
        ensemble_list=None,
    )

    config = bare_abc_config(
        model=model,
        condition=condition,
        condition_method="static",
        effective_condition=condition,
    )

    assert config._check_model_vs_condition() is None


@pytest.mark.pruned
def test_model_vs_condition_ignores_nonrequired_coordinates():
    model = make_config_data(
        paths=("model.nc",),
        extra_coords={
            "experiment": Coordinate(["historical"]),
        },
    )
    condition = make_config_data(
        paths=("condition.nc",),
    )

    config = bare_abc_config(
        model=model,
        condition=condition,
        condition_method="cross_ensemble",
        effective_condition=condition,
    )

    assert config._check_model_vs_condition() is None


@pytest.mark.parametrize(
    "missing_dimension",
    [
        "year",
        "lead_time",
    ],
)
def test_model_vs_condition_rejects_missing_required_dimension(
    missing_dimension,
):
    model = make_config_data(
        paths=("model.nc",),
    )
    condition = make_config_data(
        paths=("condition.nc",),
    )
    del condition.info.coords[missing_dimension]

    config = bare_abc_config(
        model=model,
        condition=condition,
        condition_method="cross_ensemble",
        effective_condition=condition,
    )

    with pytest.raises(
        ValueError,
        match="same dimestions",
    ):
        config._check_model_vs_condition()


@pytest.mark.parametrize(
    ("model_times", "condition_times", "model_leads", "condition_leads", "match"),
    [
        (
            (2000, 2001, 2002),
            (2000, 2001),
            (1, 2),
            (1, 2),
            "same year coordinates",
        ),
        (
            (2000, 2001),
            (2000, 2001),
            (1, 2, 3),
            (1, 2),
            "same lead_time coordinates",
        ),
    ],
)
def test_model_vs_condition_rejects_missing_coordinate_values(
    model_times,
    condition_times,
    model_leads,
    condition_leads,
    match,
):
    model = make_config_data(
        paths=("model.nc",),
        times=model_times,
        lead_times=model_leads,
    )
    condition = make_config_data(
        paths=("condition.nc",),
        times=condition_times,
        lead_times=condition_leads,
    )

    config = bare_abc_config(
        model=model,
        condition=condition,
        condition_method="cross_ensemble",
        effective_condition=condition,
    )

    with pytest.raises(
        ValueError,
        match=match,
    ):
        config._check_model_vs_condition()


@pytest.mark.pruned
def test_model_vs_condition_accepts_condition_coordinate_superset():
    model = make_config_data(
        paths=("model.nc",),
        times=(2000, 2001),
        lead_times=(1, 2),
    )
    condition = make_config_data(
        paths=("condition.nc",),
        times=(1999, 2000, 2001, 2002),
        lead_times=(0, 1, 2, 3),
    )

    config = bare_abc_config(
        model=model,
        condition=condition,
        condition_method="cross_ensemble",
        effective_condition=condition,
    )

    assert config._check_model_vs_condition() is None


@pytest.mark.pruned
def test_same_member_rejects_both_missing_ensemble_coordinates():
    model = make_config_data(
        paths=("model.nc",),
        ensembles=None,
        ensemble_list=None,
    )
    condition = make_config_data(
        paths=("condition.nc",),
        ensembles=None,
        ensemble_list=None,
    )

    config = bare_abc_config(
        model=model,
        condition=condition,
        condition_method="same_member",
        effective_condition=condition,
    )

    with pytest.raises(
        ValueError,
        match="must have ensembles",
    ):
        config._check_model_vs_condition()


@pytest.mark.pruned
def test_same_member_accepts_equal_empty_ensemble_coordinates():
    model = make_config_data(
        paths=("model.nc",),
        ensembles=(),
        ensemble_list=(),
    )
    condition = make_config_data(
        paths=("condition.nc",),
        ensembles=(),
        ensemble_list=(),
    )

    config = bare_abc_config(
        model=model,
        condition=condition,
        condition_method="same_member",
        effective_condition=condition,
    )

    assert config._check_model_vs_condition() is None


@pytest.mark.pruned
def test_same_member_rejects_same_members_in_different_order():
    model = make_config_data(
        paths=("model.nc",),
        ensembles=("r1", "r2"),
    )
    condition = make_config_data(
        paths=("condition.nc",),
        ensembles=("r2", "r1"),
    )

    config = bare_abc_config(
        model=model,
        condition=condition,
        condition_method="same_member",
        effective_condition=condition,
    )

    with pytest.raises(
        ValueError,
        match="same ensemble members",
    ):
        config._check_model_vs_condition()


@pytest.mark.pruned
def test_model_vs_condition_observation_without_spatial_dims():
    model = make_config_data(
        paths=("model.nc",),
    )
    condition = make_config_data(
        paths=("condition.nc",),
    )

    config = bare_abc_config(
        model=model,
        condition=condition,
        condition_method="cross_ensemble",
        effective_condition=condition,
        observation=object(),
    )

    assert config._check_model_vs_condition() is None


def test_model_vs_condition_observation_matching_spatial_coordinates(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "supported_NN_dimensions_sorted",
        ["lat", "lon"],
    )

    model = make_config_data(
        paths=("model.nc",),
        extra_coords={
            "lat": Coordinate([45.0, 46.0]),
            "lon": Coordinate([-125.0, -124.0]),
        },
    )
    condition = make_config_data(
        paths=("condition.nc",),
        extra_coords={
            "lat": Coordinate([45.0, 46.0]),
            "lon": Coordinate([-125.0, -124.0]),
        },
    )

    config = bare_abc_config(
        model=model,
        condition=condition,
        condition_method="cross_ensemble",
        effective_condition=condition,
        observation=object(),
    )

    assert config._check_model_vs_condition() is None


def test_model_vs_condition_observation_missing_spatial_coordinate(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "supported_NN_dimensions_sorted",
        ["lat", "lon"],
    )

    model = make_config_data(
        paths=("model.nc",),
        extra_coords={
            "lat": Coordinate([45.0, 46.0]),
        },
    )
    condition = make_config_data(
        paths=("condition.nc",),
    )

    config = bare_abc_config(
        model=model,
        condition=condition,
        condition_method="cross_ensemble",
        effective_condition=condition,
        observation=object(),
    )

    with pytest.raises(TypeError):
        config._check_model_vs_condition()


@pytest.mark.pruned
def test_model_vs_condition_observation_mismatched_spatial_coordinate(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "supported_NN_dimensions_sorted",
        ["lat", "lon"],
    )

    model = make_config_data(
        paths=("model.nc",),
        extra_coords={
            "lat": Coordinate([45.0, 46.0]),
        },
    )
    condition = make_config_data(
        paths=("condition.nc",),
        extra_coords={
            "lat": Coordinate([50.0, 51.0]),
        },
    )

    config = bare_abc_config(
        model=model,
        condition=condition,
        condition_method="cross_ensemble",
        effective_condition=condition,
        observation=object(),
    )

    with pytest.raises(TypeError):
        config._check_model_vs_condition()


@pytest.mark.parametrize(
    ("time_features", "expected_length"),
    [
        (None, 0),
        ([], 0),
        (["year"], 1),
        (["lead_time"], 1),
        (["month_sin"], 1),
        (["month_cos"], 1),
        (["year", "lead_time"], 2),
        (["month_sin", "month_cos"], 2),
        (
            ["year", "lead_time", "month_sin", "month_cos"],
            4,
        ),
    ],
)
def test_added_time_features_length_matrix(
    time_features,
    expected_length,
):
    features = AddedTimeFeatures(
        make_reference_config(),
        time_features,
    )

    assert len(features) == expected_length


@pytest.mark.pruned
def test_added_time_features_multiple_unsupported_values():
    with pytest.raises(
        ValueError,
        match="Unsupported time features",
    ) as error:
        AddedTimeFeatures(
            make_reference_config(),
            ["bad_one", "bad_two"],
        )

    assert "bad_one" in str(error.value)
    assert "bad_two" in str(error.value)


@pytest.mark.pruned
def test_added_time_features_empty_selection_reports_both_dimensions():
    features = AddedTimeFeatures(
        make_reference_config(),
        ["year"],
    )

    with pytest.raises(
        ValueError,
        match="required sample dimensions",
    ) as error:
        features(
            {},
            xr.DataArray(np.ones((2,))),
        )

    assert "year" in str(error.value)
    assert "lead_time" in str(error.value)


@pytest.mark.pruned
def test_added_time_features_ignores_extra_selection_coordinates():
    features = AddedTimeFeatures(
        make_reference_config(),
        ["year", "lead_time"],
    )

    result = features(
        {
            "year": 2000,
            "lead_time": 2,
            "ensembles": "r1",
            "unused": 999,
        },
        xr.DataArray(np.ones((3,))),
    )

    assert result.shape == (2,)


@pytest.mark.parametrize(
    ("shape", "dims", "expected_shape"),
    [
        ((), (), (2,)),
        ((5,), ("features",), (2,)),
        ((3, 4), ("channels", "features"), (2,)),
        (
            (3, 4, 5),
            ("channels", "lat", "lon"),
            (2, 4, 5),
        ),
        (
            (2, 3, 4, 5),
            ("channels", "level", "lat", "lon"),
            (2, 3, 4, 5),
        ),
        (
            (2, 3, 4, 5, 6),
            ("channels", "level", "member", "lat", "lon"),
            (2, 3, 4, 5, 6),
        ),
    ],
)
def test_added_time_features_broadcast_matrix(
    shape,
    dims,
    expected_shape,
):
    features = AddedTimeFeatures(
        make_reference_config(),
        ["year", "lead_time"],
    )

    values = np.asarray(1.0) if shape == () else np.ones(shape)
    input_array = xr.DataArray(
        values,
        dims=dims,
    )

    result = features(
        {
            "year": 2000,
            "lead_time": 2,
        },
        input_array,
    )

    assert result.shape == expected_shape


@pytest.mark.pruned
def test_added_time_features_empty_tuple_returns_empty_vector():
    features = AddedTimeFeatures(
        make_reference_config(),
        [],
    )

    result = features(
        {
            "year": 2000,
            "lead_time": 1,
        },
        xr.DataArray(np.ones((2,))),
    )

    assert result.shape == (0,)


@pytest.mark.pruned
def test_added_time_features_empty_tuple_broadcasts_spatially():
    features = AddedTimeFeatures(
        make_reference_config(),
        [],
    )

    result = features(
        {
            "year": 2000,
            "lead_time": 1,
        },
        xr.DataArray(
            np.ones((2, 3, 4)),
            dims=("channels", "lat", "lon"),
        ),
    )

    assert result.shape == (0, 3, 4)


@pytest.mark.pruned
def test_added_time_features_preserves_requested_order():
    features = AddedTimeFeatures(
        make_reference_config(),
        [
            "month_cos",
            "month_sin",
            "lead_time",
        ],
    )

    result = features(
        {
            "year": 2000,
            "lead_time": 3,
        },
        xr.DataArray(np.ones((2,))),
    )

    assert result[0] == pytest.approx(
        0.0,
        abs=1e-12,
    )
    assert result[1] == pytest.approx(
        1.0,
        abs=1e-12,
    )
    assert result[2] == pytest.approx(0.5)


@pytest.mark.pruned
def test_added_time_features_preserves_duplicates():
    features = AddedTimeFeatures(
        make_reference_config(),
        ["year", "year", "lead_time"],
    )

    result = features(
        {
            "year": 2001,
            "lead_time": 3,
        },
        xr.DataArray(np.ones((2,))),
    )

    assert result.shape == (3,)
    assert result[0] == result[1]


@pytest.mark.pruned
def test_added_time_features_equality_same_object():
    features = AddedTimeFeatures(
        make_reference_config(),
        ["year"],
    )

    assert features == features


@pytest.mark.pruned
def test_added_time_features_not_equal_feature_order():
    left = AddedTimeFeatures(
        make_reference_config(),
        ["year", "lead_time"],
    )
    right = AddedTimeFeatures(
        make_reference_config(),
        ["lead_time", "year"],
    )

    assert left != right


@pytest.mark.pruned
def test_added_time_features_not_equal_feature_count():
    left = AddedTimeFeatures(
        make_reference_config(),
        ["year"],
    )
    right = AddedTimeFeatures(
        make_reference_config(),
        ["year", "lead_time"],
    )

    assert left != right


@pytest.mark.pruned
def test_added_time_features_not_equal_reference_type():
    class AlternateReference:
        def __init__(self):
            self.lead_months = np.asarray([1, 2, 3, 6])
            self.get_common_time = np.asarray([2000, 2001, 2002])

    left = AddedTimeFeatures(
        make_reference_config(),
        ["year"],
    )
    right = AddedTimeFeatures(
        AlternateReference(),
        ["year"],
    )

    assert left != right


@pytest.mark.pruned
def test_check_init_preprocessor_error_precedes_year_error():
    dataset = bare_dataset(
        config=make_dataset_config(
            fitted_preprocessors=False,
            available_times=[2000],
        ),
        requested_years=[9999],
    )

    with pytest.raises(
        RuntimeError,
        match="fit preprocessors",
    ):
        dataset._check_init()


@pytest.mark.pruned
def test_check_init_accepts_duplicate_available_years():
    dataset = bare_dataset(
        config=make_dataset_config(
            available_times=[2000, 2000, 2001],
        ),
        requested_years=[2000],
    )

    assert dataset._check_init() is None


@pytest.mark.pruned
def test_resolve_mask_accepts_reverse_dimension_order():
    mask = xr.DataArray(
        np.zeros((2, 2), dtype=bool),
        dims=("lead_time", "year"),
        coords={
            "lead_time": [1, 2],
            "year": [2000, 2001],
        },
    )
    dataset = bare_dataset(
        config=make_dataset_config(),
        mask=mask,
    )

    dataset._resolve_mask()

    xr.testing.assert_identical(
        dataset.mask,
        mask,
    )


@pytest.mark.pruned
def test_resolve_mask_accepts_extra_dimension():
    mask = xr.DataArray(
        np.zeros((2, 2, 2), dtype=bool),
        dims=("year", "lead_time", "member"),
        coords={
            "year": [2000, 2001],
            "lead_time": [1, 2],
            "member": ["a", "b"],
        },
    )
    dataset = bare_dataset(
        config=make_dataset_config(),
        mask=mask,
    )

    dataset._resolve_mask()

    xr.testing.assert_identical(
        dataset.mask,
        mask,
    )


@pytest.mark.parametrize(
    ("dims", "coords", "expected_missing"),
    [
        (
            ("lead_time",),
            {"lead_time": [1, 2]},
            "year",
        ),
        (
            ("year",),
            {"year": [2000, 2001]},
            "lead_time",
        ),
    ],
)
def test_resolve_mask_rejects_each_missing_required_dimension(
    dims,
    coords,
    expected_missing,
):
    dataset = bare_dataset(
        config=make_dataset_config(),
        mask=xr.DataArray(
            [False, False],
            dims=dims,
            coords=coords,
        ),
    )

    with pytest.raises(
        ValueError,
        match="mask must have",
    ) as error:
        dataset._resolve_mask()

    assert expected_missing in str(error.value)


@pytest.mark.pruned
def test_resolve_mask_rejects_scalar_mask():
    dataset = bare_dataset(
        config=make_dataset_config(),
        mask=xr.DataArray(False),
    )

    with pytest.raises(
        ValueError,
        match="mask must have",
    ) as error:
        dataset._resolve_mask()

    assert "year" in str(error.value)
    assert "lead_time" in str(error.value)


@pytest.mark.parametrize(
    ("selectors", "missing_dimension"),
    [
        ({}, None),
        ({"year": [2000]}, "lead_time"),
        ({"lead_time": [1]}, "year"),
    ],
)
def test_prepare_sampling_mask_rejects_missing_selectors(
    selectors,
    missing_dimension,
):
    dataset = bare_dataset(
        config=make_dataset_config(),
        mask=make_mask(),
    )

    with pytest.raises(
        ValueError,
        match="No selectors provided",
    ) as error:
        dataset._prepare_sampling_mask(selectors)

    if missing_dimension is not None:
        assert missing_dimension in str(error.value)


@pytest.mark.parametrize(
    ("ensemble_mean", "ensembles", "expected_selection"),
    [
        (False, None, False),
        (True, None, False),
        (False, ("r1", "r2"), True),
        (True, ("r1", "r2"), True),
    ],
)
def test_load_xarray_data_selection_matrix(
    ensemble_mean,
    ensembles,
    expected_selection,
):
    config = make_config_data(
        ensembles=ensembles,
        ensemble_mean=ensemble_mean,
    )
    dataset = bare_dataset()

    with patch.object(
        module,
        "_load_xarray_data",
        return_value="loaded",
    ) as loader:
        result = dataset._load_xarray_data(
            config,
            load=False,
        )

    assert result == "loaded"

    selection = loader.call_args.kwargs["selection"]

    if expected_selection:
        assert set(selection) == {"ensembles"}
        assert selection["ensembles"] is (config.info.coords["ensembles"])
    else:
        assert selection is None


@pytest.mark.pruned
def test_get_model_indexes_returns_empty_arrays():
    dataset = bare_dataset(
        _load_model_value=True,
        model_dataset=make_xarray(),
    )

    result = dataset.get_model_indexes(
        {
            "year": np.asarray([], dtype=int),
            "lead_time": np.asarray([], dtype=int),
        }
    )

    assert result["year"].size == 0
    assert result["lead_time"].size == 0


@pytest.mark.pruned
def test_get_model_indexes_accepts_single_dimension():
    dataset = bare_dataset(
        _load_model_value=True,
        model_dataset=make_xarray(),
    )

    result = dataset.get_model_indexes(
        {
            "year": np.asarray([2001]),
        }
    )

    assert set(result) == {"year"}
    np.testing.assert_array_equal(
        result["year"],
        [1],
    )


@pytest.mark.pruned
def test_get_model_indexes_accepts_ensemble_dimension():
    dataset = bare_dataset(
        _load_model_value=True,
        model_dataset=make_xarray(
            ensembles=("r1", "r2"),
        ),
    )

    result = dataset.get_model_indexes(
        {
            "year": np.asarray([2000, 2001]),
            "lead_time": np.asarray([1, 2]),
            "ensembles": np.asarray(["r2", "r1"]),
        }
    )

    np.testing.assert_array_equal(
        result["ensembles"],
        [1, 0],
    )


@pytest.mark.pruned
def test_get_model_indexes_reports_missing_ensemble():
    dataset = bare_dataset(
        _load_model_value=True,
        model_dataset=make_xarray(
            ensembles=("r1", "r2"),
        ),
    )

    with pytest.raises(
        ValueError,
        match="model dataset",
    ) as error:
        dataset.get_model_indexes(
            {
                "ensembles": np.asarray(["r9"]),
            }
        )

    assert "r9" in str(error.value)


@pytest.mark.pruned
def test_get_cond_indexes_none_dataset_short_circuits():
    dataset = bare_dataset(
        condition_dataset=None,
        config=SimpleNamespace(
            condition_method="ensemble_mean",
        ),
    )

    assert dataset.get_cond_indexes({}) is None


@pytest.mark.pruned
def test_get_cond_indexes_static_short_circuits_invalid_values():
    dataset = bare_dataset(
        condition_dataset=make_xarray(),
        config=SimpleNamespace(
            condition_method="static",
        ),
    )

    assert (
        dataset.get_cond_indexes(
            {
                "year": np.asarray([9999]),
            }
        )
        is None
    )


@pytest.mark.pruned
def test_get_cond_indexes_ignores_unknown_dimension():
    dataset = bare_dataset(
        condition_dataset=make_xarray(),
        config=SimpleNamespace(
            condition_method="ensemble_mean",
        ),
    )

    result = dataset.get_cond_indexes(
        {
            "year": np.asarray([2000]),
            "unknown": np.asarray([123]),
        }
    )

    assert set(result) == {"year"}


@pytest.mark.pruned
def test_get_cond_indexes_cross_ensemble_ignores_member_values():
    dataset = bare_dataset(
        condition_dataset=make_xarray(
            ensembles=("r1", "r2"),
        ),
        config=SimpleNamespace(
            condition_method="cross_ensemble",
        ),
    )

    result = dataset.get_cond_indexes(
        {
            "year": np.asarray([2000]),
            "lead_time": np.asarray([1]),
            "ensembles": np.asarray(["not-in-dataset"]),
        }
    )

    assert set(result) == {
        "year",
        "lead_time",
    }


@pytest.mark.pruned
def test_get_cond_indexes_same_member_reports_missing_member():
    dataset = bare_dataset(
        condition_dataset=make_xarray(
            ensembles=("r1", "r2"),
        ),
        config=SimpleNamespace(
            condition_method="same_member",
        ),
    )

    with pytest.raises(
        ValueError,
        match="conditioning dataset",
    ) as error:
        dataset.get_cond_indexes(
            {
                "year": np.asarray([2000]),
                "lead_time": np.asarray([1]),
                "ensembles": np.asarray(["r9"]),
            }
        )

    assert "ensembles" in str(error.value)
    assert "r9" in str(error.value)


def install_fake_flattener_module(
    monkeypatch,
    flattener_class,
):
    monkeypatch.setitem(
        __import__("sys").modules,
        "cccma_ppp.preprocessing.utils_preprocessing",
        SimpleNamespace(
            Flattennanremove=flattener_class,
        ),
    )


@pytest.mark.pruned
def test_get_input_shape_empty_preprocessor_list(
    monkeypatch,
):
    class FakeFlattennanremove:
        pass

    install_fake_flattener_module(
        monkeypatch,
        FakeFlattennanremove,
    )
    monkeypatch.setattr(
        module,
        "supported_NN_dimensions_sorted",
        ["lat"],
    )

    effective_input = make_config_data(
        extra_coords={
            "lat": Coordinate([1, 2, 3, 4]),
        },
        pipeline=make_pipeline(
            fitted_preprocessors=[],
        ),
    )

    dataset = bare_dataset(
        config=SimpleNamespace(
            effective_input=effective_input,
            effective_condition=None,
        ),
        _concat_condition_value=False,
    )

    assert dataset.get_input_shape() == (1, 4)


@pytest.mark.pruned
def test_get_input_shape_uses_supported_dimension_order(
    monkeypatch,
):
    class FakeFlattennanremove:
        pass

    install_fake_flattener_module(
        monkeypatch,
        FakeFlattennanremove,
    )
    monkeypatch.setattr(
        module,
        "supported_NN_dimensions_sorted",
        ["lon", "lat"],
    )

    effective_input = make_config_data(
        extra_coords={
            "lat": Coordinate([1, 2]),
            "lon": Coordinate([3, 4, 5]),
        },
    )

    dataset = bare_dataset(
        config=SimpleNamespace(
            effective_input=effective_input,
            effective_condition=None,
        ),
        _concat_condition_value=False,
    )

    assert dataset.get_input_shape() == (1, 3, 2)


@pytest.mark.pruned
def test_get_input_shape_flattener_with_multiple_preprocessors(
    monkeypatch,
):
    class FakeFlattennanremove:
        pass

    class OtherPreprocessor:
        pass

    flattener = FakeFlattennanremove()
    flattener.final_locations = np.zeros(7)

    pipeline = make_pipeline(
        fitted_preprocessors=[
            OtherPreprocessor(),
            flattener,
        ],
    )
    pipeline.get_preprocessors.return_value = flattener

    install_fake_flattener_module(
        monkeypatch,
        FakeFlattennanremove,
    )

    dataset = bare_dataset(
        config=SimpleNamespace(
            effective_input=make_config_data(
                names=("tas", "pr"),
                pipeline=pipeline,
            ),
            effective_condition=None,
        ),
        _concat_condition_value=False,
    )

    assert dataset.get_input_shape() == (2, 7)


@pytest.mark.pruned
def test_get_input_shape_concat_condition_counts_condition_variables(
    monkeypatch,
):
    class FakeFlattennanremove:
        pass

    flattener = FakeFlattennanremove()
    flattener.final_locations = np.zeros(5)

    pipeline = make_pipeline(
        fitted_preprocessors=[flattener],
    )
    pipeline.get_preprocessors.return_value = flattener

    install_fake_flattener_module(
        monkeypatch,
        FakeFlattennanremove,
    )

    dataset = bare_dataset(
        config=SimpleNamespace(
            effective_input=make_config_data(
                names=("tas", "pr"),
                pipeline=pipeline,
            ),
            effective_condition=make_config_data(
                names=("psl", "zg"),
            ),
        ),
        _concat_condition_value=True,
    )

    assert dataset.get_input_shape() == (4, 5)


@pytest.mark.pruned
def test_index_condition_none_short_circuits_pipeline():
    pipeline = make_pipeline()

    dataset = bare_dataset(
        condition_dataset=None,
        config=SimpleNamespace(
            condition_method="ensemble_mean",
            effective_condition=make_config_data(
                pipeline=pipeline,
            ),
        ),
    )

    assert dataset._index_condition_dataset(0) is None
    pipeline.transform.assert_not_called()


@pytest.mark.pruned
def test_index_condition_same_member_uses_member_index():
    pipeline = make_pipeline()

    dataset = bare_dataset(
        condition_dataset=make_xarray(
            ensembles=("r1", "r2"),
        ),
        cond_indexes={
            "year": np.asarray([0]),
            "lead_time": np.asarray([1]),
            "ensembles": np.asarray([1]),
        },
        config=SimpleNamespace(
            condition_method="same_member",
            effective_condition=make_config_data(
                ensembles=("r1", "r2"),
                pipeline=pipeline,
            ),
        ),
    )

    with patch.object(
        module,
        "_unwrap_data_variables",
        side_effect=lambda value: value,
    ):
        result = dataset._index_condition_dataset(0)

    assert result.sizes["year"] == 1
    assert result.sizes["lead_time"] == 1
    assert result.sizes["ensembles"] == 1
    assert result.coords["ensembles"].item() == "r2"


@pytest.mark.pruned
def test_index_cross_ensemble_overrides_indexed_member(
    monkeypatch,
):
    pipeline = make_pipeline()

    dataset = bare_dataset(
        condition_dataset=make_xarray(
            ensembles=("r1", "r2", "r3"),
        ),
        cond_indexes={
            "year": np.asarray([0]),
            "lead_time": np.asarray([0]),
            "ensembles": np.asarray([0]),
        },
        config=SimpleNamespace(
            condition_method="cross_ensemble",
            effective_condition=make_config_data(
                ensembles=("r1", "r2", "r3"),
                pipeline=pipeline,
            ),
        ),
    )

    randint = MagicMock(
        return_value=2,
    )
    monkeypatch.setattr(
        module.np.random,
        "randint",
        randint,
    )

    with patch.object(
        module,
        "_unwrap_data_variables",
        side_effect=lambda value: value,
    ):
        result = dataset._index_condition_dataset(0)

    randint.assert_called_once_with(3)
    assert result.coords["ensembles"].item() == "r3"


@pytest.mark.pruned
def test_index_model_false_short_circuits_pipeline():
    pipeline = make_pipeline()

    dataset = bare_dataset(
        _load_model_value=False,
        config=SimpleNamespace(
            model=make_config_data(
                pipeline=pipeline,
            ),
        ),
    )

    assert dataset._index_model_dataset(0) is None
    pipeline.transform.assert_not_called()


@pytest.mark.pruned
def test_index_model_with_every_available_dimension():
    pipeline = make_pipeline()

    dataset = bare_dataset(
        _load_model_value=True,
        model_dataset=make_xarray(
            ensembles=("r1", "r2"),
        ),
        model_indexes={
            "year": np.asarray([1]),
            "lead_time": np.asarray([0]),
            "ensembles": np.asarray([1]),
        },
        config=SimpleNamespace(
            model=make_config_data(
                ensembles=("r1", "r2"),
                pipeline=pipeline,
            ),
        ),
    )

    with patch.object(
        module,
        "_unwrap_data_variables",
        side_effect=lambda value: value,
    ):
        result = dataset._index_model_dataset(0)

    assert result.coords["year"].item() == 2001
    assert result.coords["lead_time"].item() == 1
    assert result.coords["ensembles"].item() == "r2"


def configure_dataset_init_mocks(
    monkeypatch,
    dataset,
    *,
    sample_coords=None,
    model_indexes=None,
    cond_indexes=None,
):
    if sample_coords is None:
        sample_coords = {
            "year": np.asarray([2000]),
            "lead_time": np.asarray([1]),
        }

    monkeypatch.setattr(
        dataset,
        "_check_init",
        MagicMock(),
    )
    monkeypatch.setattr(
        dataset,
        "_resolve_mask",
        MagicMock(),
    )
    monkeypatch.setattr(
        dataset,
        "_prepare_sampling_mask",
        MagicMock(),
    )
    monkeypatch.setattr(
        dataset,
        "get_sampling_coords",
        MagicMock(
            return_value=sample_coords,
        ),
    )
    monkeypatch.setattr(
        dataset,
        "get_model_indexes",
        MagicMock(
            return_value=model_indexes,
        ),
    )
    monkeypatch.setattr(
        dataset,
        "get_cond_indexes",
        MagicMock(
            return_value=cond_indexes,
        ),
    )