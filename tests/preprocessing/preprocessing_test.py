from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import xarray as xr

import cccma_ppp.preprocessing.preprocessing as module
from cccma_ppp.preprocessing.preprocessing import (
    PreprocessingPipeline,
)


class DummyPreprocessor:
    def __init__(
        self,
        *,
        fitted=False,
        transform_offset=1.0,
    ):
        self.fitted = fitted
        self.transform_offset = transform_offset
        self.fit_calls = []
        self.transform_calls = []
        self.inverse_calls = []

    def fit(self, data, mask=None):
        self.fit_calls.append(
            {
                "data": data,
                "mask": mask,
            }
        )
        self.fitted = True
        return self

    def transform(self, data, **kwargs):
        self.transform_calls.append(
            {
                "data": data,
                "kwargs": kwargs,
            }
        )

        scale = kwargs.get("scale", 1.0)

        return data * scale + self.transform_offset

    def inverse_transform(self, data, **kwargs):
        self.inverse_calls.append(
            {
                "data": data,
                "kwargs": kwargs,
            }
        )

        scale = kwargs.get("scale", 1.0)

        return (data - self.transform_offset) / scale


class DummySelector:
    def __init__(
        self,
        name,
        preprocessor,
    ):
        self.name = name
        self.preprocessor = preprocessor
        self.get_calls = 0

    def get_preprocessor(self):
        self.get_calls += 1
        return self.preprocessor


def make_dataset():
    values = np.arange(
        16,
        dtype=np.float32,
    ).reshape(2, 2, 2, 2)

    return xr.Dataset(
        {
            "tas": (
                (
                    "year",
                    "channels",
                    "lat",
                    "lon",
                ),
                values,
            ),
            "pr": (
                (
                    "year",
                    "channels",
                    "lat",
                    "lon",
                ),
                values + 10,
            ),
        },
        coords={
            "year": [
                2000,
                2001,
            ],
            "channels": [
                "member_1",
                "member_2",
            ],
            "lat": [
                45.0,
                46.0,
            ],
            "lon": [
                -124.0,
                -123.0,
            ],
        },
    )


def make_output_array(
    *,
    channels=("tas", "pr"),
):
    return xr.DataArray(
        np.ones(
            (
                len(channels),
                2,
                2,
            ),
            dtype=np.float32,
        ),
        dims=(
            "channels",
            "output_dim_0",
            "output_dim_1",
        ),
        coords={
            "channels": list(channels),
            "output_dim_0": [
                0,
                1,
            ],
            "output_dim_1": [
                0,
                1,
            ],
        },
    )


@pytest.fixture(autouse=True)
def reset_pipeline_counter():
    original = PreprocessingPipeline.num_instances
    PreprocessingPipeline.num_instances = 0

    yield

    PreprocessingPipeline.num_instances = original


@pytest.mark.pruned
def test_pipeline_defaults():
    pipeline = PreprocessingPipeline()

    assert pipeline.preprocessors_list == []
    assert pipeline.load_dir is None
    assert pipeline.fitted is False
    assert pipeline.reference_coords is None
    assert pipeline.reference_var is None
    assert pipeline.name == "instance_1"
    assert pipeline.pipeline == []


@pytest.mark.pruned
def test_pipeline_assigns_unique_default_names():
    first = PreprocessingPipeline()
    second = PreprocessingPipeline()
    third = PreprocessingPipeline()

    assert first.name == "instance_1"
    assert second.name == "instance_1"
    assert third.name == "instance_1"


@pytest.mark.pruned
def test_pipeline_constructs_preprocessors():
    first_preprocessor = DummyPreprocessor()
    second_preprocessor = DummyPreprocessor()

    first_selector = DummySelector(
        "NORMALIZER",
        first_preprocessor,
    )
    second_selector = DummySelector(
        "Standardizer",
        second_preprocessor,
    )

    pipeline = PreprocessingPipeline(
        preprocessors_list=[
            first_selector,
            second_selector,
        ]
    )

    assert pipeline.pipeline == [
        (
            "normalizer",
            first_preprocessor,
        ),
        (
            "standardizer",
            second_preprocessor,
        ),
    ]
    assert first_selector.get_calls == 1
    assert second_selector.get_calls == 1


@pytest.mark.pruned
def test_pipeline_with_load_dir_does_not_construct_steps(
    tmp_path,
):
    selector = DummySelector(
        "normalizer",
        DummyPreprocessor(),
    )

    pipeline = PreprocessingPipeline(
        preprocessors_list=[selector],
        load_dir=tmp_path / "pipeline.joblib",
    )

    assert pipeline.load_dir == (tmp_path / "pipeline.joblib")
    assert pipeline.fitted is False
    assert selector.get_calls == 0
    assert not hasattr(pipeline, "pipeline")
    assert not hasattr(pipeline, "name")


@pytest.mark.pruned
def test_set_name():
    pipeline = PreprocessingPipeline()

    result = pipeline.set_name("model")

    assert result is None
    assert pipeline.name == "model"


@pytest.mark.pruned
def test_fit_empty_pipeline_returns_self():
    pipeline = PreprocessingPipeline()
    data = make_dataset()

    result = pipeline.fit(
        base_data=data,
        save=False,
    )

    assert result is pipeline
    assert pipeline.fitted is True
    assert pipeline.steps == []
    assert pipeline.fitted_preprocessors == []


@pytest.mark.pruned
def test_fit_records_fitted_years():
    pipeline = PreprocessingPipeline()
    data = make_dataset()

    pipeline.fit(
        base_data=data,
        save=False,
    )

    np.testing.assert_array_equal(
        pipeline.fitted_based_year,
        np.asarray(
            [
                2000,
                2001,
            ]
        ),
    )


@pytest.mark.pruned
def test_fit_calls_preprocessors_in_order():
    first = DummyPreprocessor(
        transform_offset=1.0,
    )
    second = DummyPreprocessor(
        transform_offset=2.0,
    )

    pipeline = PreprocessingPipeline(
        preprocessors_list=[
            DummySelector(
                "first",
                first,
            ),
            DummySelector(
                "second",
                second,
            ),
        ]
    )

    data = make_dataset()

    pipeline.fit(
        base_data=data,
        save=False,
    )

    assert pipeline.steps == [
        "first",
        "second",
    ]
    assert pipeline.fitted_preprocessors == [
        first,
        second,
    ]

    assert len(first.fit_calls) == 1
    assert len(second.fit_calls) == 1

    xr.testing.assert_equal(
        first.fit_calls[0]["data"],
        data,
    )

    expected_second_input = data + 1.0

    xr.testing.assert_equal(
        second.fit_calls[0]["data"],
        expected_second_input,
    )


def test_fit_passes_mask_to_each_preprocessor():
    first = DummyPreprocessor()
    second = DummyPreprocessor()

    pipeline = PreprocessingPipeline(
        preprocessors_list=[
            DummySelector(
                "first",
                first,
            ),
            DummySelector(
                "second",
                second,
            ),
        ]
    )

    data = make_dataset()
    mask = xr.DataArray(
        np.ones((2, 2)),
        dims=(
            "lat",
            "lon",
        ),
    )

    pipeline.fit(
        base_data=data,
        mask=mask,
        save=False,
    )

    assert first.fit_calls[0]["mask"] is mask
    assert second.fit_calls[0]["mask"] is mask


@pytest.mark.pruned
def test_fit_extracts_reference_metadata(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "supported_NN_dimensions_sorted",
        [
            "lat",
            "lon",
        ],
    )

    pipeline = PreprocessingPipeline()
    data = make_dataset()

    pipeline.fit(
        base_data=data,
        save=False,
    )

    assert list(pipeline.reference_coords) == [
        "lat",
        "lon",
    ]
    assert pipeline.reference_var == [
        "tas",
        "pr",
    ]


@pytest.mark.pruned
def test_fit_saves_pipeline_with_default_path(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        module.RuntimeContext,
        "GLOBAL_EXP_DIR",
        tmp_path,
    )

    dump_mock = Mock()
    monkeypatch.setattr(
        module.joblib,
        "dump",
        dump_mock,
    )

    pipeline = PreprocessingPipeline()
    data = make_dataset()

    pipeline.fit(
        base_data=data,
        save=True,
    )

    expected_directory = tmp_path / "preprocessing_pipeline"

    assert expected_directory.exists()

    dump_mock.assert_called_once_with(
        pipeline,
        expected_directory / "instance_1_preprocessing_pipeline.joblib",
    )


def test_fit_saves_pipeline_with_custom_path_and_name(
    tmp_path,
    monkeypatch,
):
    dump_mock = Mock()

    monkeypatch.setattr(
        module.joblib,
        "dump",
        dump_mock,
    )

    pipeline = PreprocessingPipeline()
    pipeline.set_name("ignored-by-explicit-name")

    save_path = tmp_path / "nested" / "pipelines"

    pipeline.fit(
        base_data=make_dataset(),
        save=True,
        save_name="custom.joblib",
        save_path=save_path,
    )

    assert save_path.exists()

    dump_mock.assert_called_once_with(
        pipeline,
        save_path / "custom.joblib",
    )


@pytest.mark.pruned
def test_fit_preserves_existing_save_directory(
    tmp_path,
    monkeypatch,
):
    save_path = tmp_path / "pipelines"
    save_path.mkdir()

    dump_mock = Mock()

    monkeypatch.setattr(
        module.joblib,
        "dump",
        dump_mock,
    )

    pipeline = PreprocessingPipeline()

    pipeline.fit(
        base_data=make_dataset(),
        save=True,
        save_path=save_path,
    )

    assert save_path.is_dir()
    dump_mock.assert_called_once()


@pytest.mark.pruned
def test_fit_does_not_save_when_disabled(
    monkeypatch,
):
    dump_mock = Mock()

    monkeypatch.setattr(
        module.joblib,
        "dump",
        dump_mock,
    )

    pipeline = PreprocessingPipeline()

    pipeline.fit(
        base_data=make_dataset(),
        save=False,
    )

    dump_mock.assert_not_called()


def test_fit_load_branch_calls_private_loader(
    tmp_path,
):
    pipeline = PreprocessingPipeline(
        load_dir=tmp_path / "saved.joblib",
    )
    pipeline._load_from_memory = Mock()

    result = pipeline.fit()

    assert result is pipeline

    pipeline._load_from_memory.assert_called_once_with(tmp_path / "saved.joblib")


def make_fitted_pipeline(
    preprocessors,
    steps=None,
):
    pipeline = PreprocessingPipeline()
    pipeline.fitted = True
    pipeline.fitted_preprocessors = list(preprocessors)
    pipeline.steps = (
        list(steps)
        if steps is not None
        else [f"step_{index}" for index in range(len(preprocessors))]
    )
    return pipeline


def test_transform_empty_pipeline_returns_input():
    pipeline = make_fitted_pipeline([])
    data = make_dataset()

    result = pipeline.transform(data)

    assert result is data


@pytest.mark.pruned
def test_transform_applies_steps_in_order():
    first = DummyPreprocessor(
        transform_offset=1.0,
    )
    second = DummyPreprocessor(
        transform_offset=2.0,
    )

    pipeline = make_fitted_pipeline(
        [
            first,
            second,
        ],
        steps=[
            "first",
            "second",
        ],
    )

    data = make_dataset()
    result = pipeline.transform(data)

    expected = data + 3.0

    xr.testing.assert_equal(
        result,
        expected,
    )

    xr.testing.assert_equal(
        first.transform_calls[0]["data"],
        data,
    )
    xr.testing.assert_equal(
        second.transform_calls[0]["data"],
        data + 1.0,
    )


@pytest.mark.pruned
def test_transform_passes_step_arguments():
    first = DummyPreprocessor(
        transform_offset=1.0,
    )
    second = DummyPreprocessor(
        transform_offset=2.0,
    )

    pipeline = make_fitted_pipeline(
        [
            first,
            second,
        ],
        steps=[
            "first",
            "second",
        ],
    )

    data = make_dataset()

    result = pipeline.transform(
        data,
        step_arguments={
            "first": {
                "scale": 2.0,
            },
            "second": {
                "scale": 3.0,
            },
        },
    )

    expected = (data * 2.0 + 1.0) * 3.0 + 2.0

    xr.testing.assert_equal(
        result,
        expected,
    )

    assert first.transform_calls[0]["kwargs"] == {
        "scale": 2.0,
    }
    assert second.transform_calls[0]["kwargs"] == {
        "scale": 3.0,
    }


@pytest.mark.pruned
def test_transform_uses_empty_arguments_for_unspecified_step():
    preprocessor = DummyPreprocessor()

    pipeline = make_fitted_pipeline(
        [preprocessor],
        steps=["normalizer"],
    )

    pipeline.transform(
        make_dataset(),
        step_arguments={},
    )

    assert preprocessor.transform_calls[0]["kwargs"] == {}


def test_transform_rejects_unknown_step_arguments():
    pipeline = make_fitted_pipeline(
        [DummyPreprocessor()],
        steps=["normalizer"],
    )

    with pytest.raises(
        ValueError,
        match="unknown not in preprocessing steps",
    ):
        pipeline.transform(
            make_dataset(),
            step_arguments={
                "unknown": {},
            },
        )


def test_inverse_transform_empty_pipeline_returns_input():
    pipeline = make_fitted_pipeline([])
    data = make_dataset()

    result = pipeline.inverse_transform(data)

    assert result is data


@pytest.mark.pruned
def test_inverse_transform_applies_steps_in_reverse_order():
    first = DummyPreprocessor(
        transform_offset=1.0,
    )
    second = DummyPreprocessor(
        transform_offset=2.0,
    )

    pipeline = make_fitted_pipeline(
        [
            first,
            second,
        ],
        steps=[
            "first",
            "second",
        ],
    )

    original = make_dataset()
    transformed = (original + 1.0) + 2.0

    result = pipeline.inverse_transform(transformed)

    xr.testing.assert_equal(
        result,
        original,
    )

    xr.testing.assert_equal(
        second.inverse_calls[0]["data"],
        transformed,
    )
    xr.testing.assert_equal(
        first.inverse_calls[0]["data"],
        transformed - 2.0,
    )


def test_inverse_transform_passes_step_arguments():
    first = DummyPreprocessor(
        transform_offset=1.0,
    )
    second = DummyPreprocessor(
        transform_offset=2.0,
    )

    pipeline = make_fitted_pipeline(
        [
            first,
            second,
        ],
        steps=[
            "first",
            "second",
        ],
    )

    original = make_dataset()

    transformed = pipeline.transform(
        original,
        step_arguments={
            "first": {
                "scale": 2.0,
            },
            "second": {
                "scale": 3.0,
            },
        },
    )

    result = pipeline.inverse_transform(
        transformed,
        step_arguments={
            "first": {
                "scale": 2.0,
            },
            "second": {
                "scale": 3.0,
            },
        },
    )

    xr.testing.assert_allclose(
        result,
        original,
    )

    assert second.inverse_calls[0]["kwargs"] == {
        "scale": 3.0,
    }
    assert first.inverse_calls[0]["kwargs"] == {
        "scale": 2.0,
    }


def test_inverse_transform_rejects_unknown_step_arguments():
    pipeline = make_fitted_pipeline(
        [DummyPreprocessor()],
        steps=["normalizer"],
    )

    with pytest.raises(
        ValueError,
        match="unknown not in preprocessing steps",
    ):
        pipeline.inverse_transform(
            make_dataset(),
            step_arguments={
                "unknown": {},
            },
        )


def test_get_preprocessors_requires_fitted_pipeline():
    pipeline = PreprocessingPipeline()

    with pytest.raises(
        RuntimeError,
        match="Pipeline needs to be fitted first",
    ):
        pipeline.get_preprocessors()


def test_get_preprocessors_returns_all():
    first = DummyPreprocessor(
        fitted=True,
    )
    second = DummyPreprocessor(
        fitted=True,
    )

    pipeline = make_fitted_pipeline(
        [
            first,
            second,
        ],
        steps=[
            "first",
            "second",
        ],
    )

    result = pipeline.get_preprocessors()

    assert result is pipeline.fitted_preprocessors
    assert result == [
        first,
        second,
    ]


def test_get_preprocessors_returns_first_duplicate():
    first = DummyPreprocessor(
        fitted=True,
    )
    second = DummyPreprocessor(
        fitted=True,
    )

    pipeline = make_fitted_pipeline(
        [
            first,
            second,
        ],
        steps=[
            "duplicate",
            "duplicate",
        ],
    )

    with pytest.raises(
        (TypeError, ValueError),
    ):
        pipeline.get_preprocessors("duplicate")


def test_add_fitted_preprocessor_appends():
    existing = DummyPreprocessor(
        fitted=True,
    )
    added = DummyPreprocessor(
        fitted=True,
    )

    pipeline = make_fitted_pipeline(
        [existing],
        steps=["existing"],
    )

    result = pipeline.add_fitted_preprocessor(
        added,
        "added",
    )

    assert result is None
    assert pipeline.steps == [
        "existing",
        "added",
    ]
    assert pipeline.fitted_preprocessors == [
        existing,
        added,
    ]


def test_add_fitted_preprocessor_inserts_at_index():
    first = DummyPreprocessor(
        fitted=True,
    )
    second = DummyPreprocessor(
        fitted=True,
    )
    inserted = DummyPreprocessor(
        fitted=True,
    )

    pipeline = make_fitted_pipeline(
        [
            first,
            second,
        ],
        steps=[
            "first",
            "second",
        ],
    )

    pipeline.add_fitted_preprocessor(
        inserted,
        "inserted",
        index=1,
    )

    assert pipeline.steps == [
        "first",
        "inserted",
        "second",
    ]
    assert pipeline.fitted_preprocessors == [
        first,
        inserted,
        second,
    ]


@pytest.mark.pruned
def test_add_fitted_preprocessor_requires_fitted_object():
    pipeline = make_fitted_pipeline([])

    with pytest.raises(
        AssertionError,
        match="preprocessor must be fitted",
    ):
        pipeline.add_fitted_preprocessor(
            DummyPreprocessor(
                fitted=False,
            ),
            "invalid",
        )


def test_extract_output_coords_requires_fitted_pipeline():
    pipeline = PreprocessingPipeline()

    with pytest.raises(
        ValueError,
        match=("Spatial coords can only be extracted for a fitted pipeline"),
    ):
        pipeline.extract_output_coords_vars(make_dataset())


@pytest.mark.pruned
def test_extract_output_coords_uses_supported_dimensions(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "supported_NN_dimensions_sorted",
        [
            "lat",
            "lon",
            "height",
            "width",
        ],
    )

    pipeline = PreprocessingPipeline()
    pipeline.fitted = True

    data = make_dataset()

    pipeline.extract_output_coords_vars(data)

    assert list(pipeline.reference_coords) == [
        "lat",
        "lon",
    ]

    xr.testing.assert_equal(
        pipeline.reference_coords["lat"],
        data["lat"],
    )
    xr.testing.assert_equal(
        pipeline.reference_coords["lon"],
        data["lon"],
    )


@pytest.mark.pruned
def test_extract_output_coords_records_dataset_variables():
    pipeline = PreprocessingPipeline()
    pipeline.fitted = True

    pipeline.extract_output_coords_vars(make_dataset())

    assert pipeline.reference_var == [
        "tas",
        "pr",
    ]


def test_to_dataset_rejects_channel_mismatch():
    pipeline = make_fitted_pipeline([])
    pipeline.reference_var = [
        "tas",
        "pr",
    ]
    pipeline.reference_coords = {
        "lat": xr.DataArray(
            [45.0, 46.0],
            dims=("lat",),
        ),
        "lon": xr.DataArray(
            [-124.0, -123.0],
            dims=("lon",),
        ),
    }

    data = make_output_array(
        channels=("tas",),
    )

    with pytest.raises(
        ValueError,
        match=("does not match the preprocessing pipeline"),
    ):
        pipeline.to_dataset(data)


def test_to_dataset_without_flattener(
    monkeypatch,
):
    class FakeFlattener:
        pass

    monkeypatch.setattr(
        "cccma_ppp.preprocessing.utils_preprocessing.Flattennanremove",
        FakeFlattener,
    )

    pipeline = make_fitted_pipeline(
        [DummyPreprocessor(fitted=True)],
        steps=["normalizer"],
    )
    pipeline.reference_var = [
        "tas",
        "pr",
    ]
    pipeline.reference_coords = {
        "lat": xr.DataArray(
            [45.0, 46.0],
            dims=("lat",),
        ),
        "lon": xr.DataArray(
            [-124.0, -123.0],
            dims=("lon",),
        ),
    }

    data = make_output_array()

    result = pipeline.to_dataset(data)

    assert isinstance(result, xr.Dataset)
    assert set(result.data_vars) == {
        "tas",
        "pr",
    }
    assert "lat" in result.dims
    assert "lon" in result.dims
    assert "output_dim_0" not in result.dims
    assert "output_dim_1" not in result.dims


def make_saved_pipeline():
    preprocessor = DummyPreprocessor(
        fitted=True,
    )

    return SimpleNamespace(
        preprocessors_list=["selector"],
        pipeline=[
            (
                "normalizer",
                preprocessor,
            )
        ],
        steps=["normalizer"],
        fitted_preprocessors=[preprocessor],
        fitted=True,
        fitted_based_year=np.asarray(
            [
                2000,
                2001,
            ]
        ),
        reference_coords={
            "lat": xr.DataArray(
                [45.0, 46.0],
                dims=("lat",),
            ),
        },
        reference_var=[
            "tas",
            "pr",
        ],
        name="loaded",
    )


def test_load_from_memory_rejects_unfitted_pipeline(
    tmp_path,
    monkeypatch,
):
    loaded = SimpleNamespace(
        fitted=False,
    )

    load_mock = Mock(
        return_value=loaded,
    )

    monkeypatch.setattr(
        module.joblib,
        "load",
        load_mock,
    )

    pipeline = PreprocessingPipeline()

    with pytest.raises(
        ValueError,
        match="has to be fitted first",
    ):
        pipeline.load_from_memory(tmp_path / "pipeline.joblib")

    load_mock.assert_called_once_with(tmp_path / "pipeline.joblib")


def test_load_from_memory_copies_pipeline_state(
    tmp_path,
    monkeypatch,
):
    loaded = make_saved_pipeline()

    monkeypatch.setattr(
        module.joblib,
        "load",
        Mock(return_value=loaded),
    )

    pipeline = PreprocessingPipeline()

    result = pipeline.load_from_memory(tmp_path / "pipeline.joblib")

    assert result is pipeline
    assert pipeline.preprocessors_list == ["selector"]
    assert pipeline.pipeline == (loaded.pipeline)
    assert pipeline.steps == ["normalizer"]
    assert pipeline.fitted_preprocessors == loaded.fitted_preprocessors
    assert pipeline.fitted is True

    np.testing.assert_array_equal(
        pipeline.fitted_based_year,
        np.asarray(
            [
                2000,
                2001,
            ]
        ),
    )

    assert pipeline.reference_coords == loaded.reference_coords
    assert pipeline.reference_var == [
        "tas",
        "pr",
    ]
    assert pipeline.name == "loaded"


@pytest.mark.pruned
def test_load_from_memory_accepts_string_path(
    tmp_path,
    monkeypatch,
):
    loaded = make_saved_pipeline()
    load_mock = Mock(
        return_value=loaded,
    )

    monkeypatch.setattr(
        module.joblib,
        "load",
        load_mock,
    )

    pipeline = PreprocessingPipeline()

    pipeline.load_from_memory(str(tmp_path / "pipeline.joblib"))

    load_mock.assert_called_once_with(tmp_path / "pipeline.joblib")