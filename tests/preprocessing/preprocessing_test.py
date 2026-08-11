from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import xarray as xr

import cccma_ppp.preprocessing.preprocessing as module
from cccma_ppp.preprocessing.preprocessing import PreprocessingPipeline
from cccma_ppp.generic.runtime import RuntimeContext


class AddPreprocessor:
    def __init__(
        self,
        value=1.0,
        *,
        fitted=False,
    ):
        self.value = value
        self.fitted = fitted
        self.fit_calls = []
        self.transform_calls = []
        self.inverse_calls = []

    def fit(
        self,
        data,
        mask=None,
    ):
        self.fit_calls.append(
            {
                "data": data,
                "mask": mask,
            }
        )
        self.fitted = True
        return self

    def transform(
        self,
        data,
        **kwargs,
    ):
        self.transform_calls.append(
            {
                "data": data,
                "kwargs": kwargs,
            }
        )

        value = kwargs.get(
            "value",
            self.value,
        )
        return data + value

    def inverse_transform(
        self,
        data,
        **kwargs,
    ):
        self.inverse_calls.append(
            {
                "data": data,
                "kwargs": kwargs,
            }
        )

        value = kwargs.get(
            "value",
            self.value,
        )
        return data - value


class MultiplyPreprocessor:
    def __init__(
        self,
        value=2.0,
        *,
        fitted=False,
    ):
        self.value = value
        self.fitted = fitted
        self.fit_calls = []
        self.transform_calls = []
        self.inverse_calls = []

    def fit(
        self,
        data,
        mask=None,
    ):
        self.fit_calls.append(
            {
                "data": data,
                "mask": mask,
            }
        )
        self.fitted = True
        return self

    def transform(
        self,
        data,
        **kwargs,
    ):
        self.transform_calls.append(
            {
                "data": data,
                "kwargs": kwargs,
            }
        )

        value = kwargs.get(
            "value",
            self.value,
        )
        return data * value

    def inverse_transform(
        self,
        data,
        **kwargs,
    ):
        self.inverse_calls.append(
            {
                "data": data,
                "kwargs": kwargs,
            }
        )

        value = kwargs.get(
            "value",
            self.value,
        )
        return data / value


class DummySelector:
    def __init__(
        self,
        name,
        preprocessor,
    ):
        self.name = name
        self.preprocessor = preprocessor
        self.calls = 0

    def get_preprocessor(self):
        self.calls += 1
        return self.preprocessor


def make_base_data():
    times = np.asarray(
        [
            "2000-01-01",
            "2001-01-01",
        ],
        dtype="datetime64[ns]",
    )

    return xr.Dataset(
        {
            "tas": (
                (
                    PreprocessingPipeline.init_time_time,
                    "lat",
                    "lon",
                ),
                np.arange(
                    8,
                    dtype=float,
                ).reshape(
                    2,
                    2,
                    2,
                ),
            ),
            "pr": (
                (
                    PreprocessingPipeline.init_time_time,
                    "lat",
                    "lon",
                ),
                np.arange(
                    8,
                    16,
                    dtype=float,
                ).reshape(
                    2,
                    2,
                    2,
                ),
            ),
        },
        coords={
            PreprocessingPipeline.init_time_time: times,
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


@pytest.fixture(autouse=True)
def reset_pipeline_counter():
    original = PreprocessingPipeline.num_instances
    PreprocessingPipeline.num_instances = 0

    yield

    PreprocessingPipeline.num_instances = original


class TestInitialization:
    def test_default_values(self):
        pipeline = PreprocessingPipeline()

        assert pipeline.preprocessors_list == []
        assert pipeline.load_dir is None
        assert pipeline.fitted is False
        assert pipeline.reference_coords is None
        assert pipeline.reference_var is None
        assert pipeline.name == "instance_1"
        assert pipeline.pipeline == []

    def test_assigns_unique_default_names(self):
        first = PreprocessingPipeline()
        second = PreprocessingPipeline()
        third = PreprocessingPipeline()

        assert first.name == "instance_1"
        assert second.name == "instance_1"
        assert third.name == "instance_1"

    def test_constructs_preprocessors(self):
        first_preprocessor = AddPreprocessor()
        second_preprocessor = MultiplyPreprocessor()

        first_selector = DummySelector(
            "NORMALIZER",
            first_preprocessor,
        )
        second_selector = DummySelector(
            "STANDARDIZER",
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
        assert first_selector.calls == 1
        assert second_selector.calls == 1

    def test_load_directory_does_not_construct_pipeline(self):
        selector = DummySelector(
            "normalizer",
            AddPreprocessor(),
        )

        pipeline = PreprocessingPipeline(
            preprocessors_list=[
                selector,
            ],
            load_dir="/tmp/pipeline.joblib",
        )

        assert selector.calls == 0
        assert not hasattr(
            pipeline,
            "pipeline",
        )
        assert not hasattr(
            pipeline,
            "name",
        )

    def test_configuration_dimension_names(self):
        pipeline = PreprocessingPipeline()

        assert isinstance(
            pipeline.init_time_time,
            str,
        )
        assert isinstance(
            pipeline.lead_time_time,
            str,
        )
        assert isinstance(
            pipeline.supported_NN_dimensions,
            tuple,
        )

    def test_set_name(self):
        pipeline = PreprocessingPipeline()

        result = pipeline.set_name("tas_pipeline", {})

        assert result is None
        assert pipeline.name == "tas_pipeline"


class TestFit:
    def test_empty_pipeline_returns_self(
        self,
        monkeypatch,
    ):
        pipeline = PreprocessingPipeline()
        data = make_base_data()

        monkeypatch.setattr(
            pipeline,
            "extract_output_coords_vars",
            Mock(),
        )

        result = pipeline.fit(
            data,
            save=False,
        )

        assert result is pipeline
        assert pipeline.fitted is True
        assert pipeline.steps == []
        assert pipeline.fitted_preprocessors == []

    def test_records_fitted_times(
        self,
        monkeypatch,
    ):
        pipeline = PreprocessingPipeline()
        data = make_base_data()

        monkeypatch.setattr(
            pipeline,
            "extract_output_coords_vars",
            Mock(),
        )

        pipeline.fit(
            data,
            save=False,
        )

        np.testing.assert_array_equal(
            pipeline.fitted_based_time,
            data[pipeline.init_time_time].values,
        )

    def test_calls_preprocessors_in_order(
        self,
        monkeypatch,
    ):
        first = AddPreprocessor(value=1.0)
        second = MultiplyPreprocessor(value=2.0)

        pipeline = PreprocessingPipeline(
            preprocessors_list=[
                DummySelector(
                    "add",
                    first,
                ),
                DummySelector(
                    "multiply",
                    second,
                ),
            ]
        )

        data = make_base_data()

        monkeypatch.setattr(
            pipeline,
            "extract_output_coords_vars",
            Mock(),
        )

        pipeline.fit(
            data,
            save=False,
        )

        assert len(first.fit_calls) == 1
        assert len(second.fit_calls) == 1

        xr.testing.assert_identical(
            first.fit_calls[0]["data"],
            data,
        )
        xr.testing.assert_identical(
            second.fit_calls[0]["data"],
            data + 1.0,
        )

        assert pipeline.steps == [
            "add",
            "multiply",
        ]
        assert pipeline.fitted_preprocessors == [
            first,
            second,
        ]

    def test_passes_mask_to_each_preprocessor(
        self,
        monkeypatch,
    ):
        first = AddPreprocessor()
        second = MultiplyPreprocessor()

        pipeline = PreprocessingPipeline(
            preprocessors_list=[
                DummySelector(
                    "add",
                    first,
                ),
                DummySelector(
                    "multiply",
                    second,
                ),
            ]
        )

        data = make_base_data()
        mask = xr.DataArray(
            np.ones(
                (
                    2,
                    2,
                )
            ),
            dims=(
                "lat",
                "lon",
            ),
        )

        monkeypatch.setattr(
            pipeline,
            "extract_output_coords_vars",
            Mock(),
        )

        pipeline.fit(
            data,
            mask=mask,
            save=False,
        )

        assert first.fit_calls[0]["mask"] is mask
        assert second.fit_calls[0]["mask"] is mask

    def test_extracts_reference_metadata(
        self,
        monkeypatch,
    ):
        pipeline = PreprocessingPipeline()
        data = make_base_data()

        extraction = Mock()
        monkeypatch.setattr(
            pipeline,
            "extract_output_coords_vars",
            extraction,
        )

        pipeline.fit(
            data,
            save=False,
        )

        extraction.assert_called_once_with(data)

    def test_saves_with_default_path(
        self,
        tmp_path,
        monkeypatch,
    ):
        pipeline = PreprocessingPipeline()
        data = make_base_data()

        monkeypatch.setattr(
            RuntimeContext,
            "GLOBAL_EXP_DIR",
            tmp_path,
        )
        monkeypatch.setattr(
            pipeline,
            "extract_output_coords_vars",
            Mock(),
        )

        dump = Mock()
        monkeypatch.setattr(
            module.joblib,
            "dump",
            dump,
        )

        pipeline.fit(
            data,
            save=True,
        )

        expected_path = (
            tmp_path
            / "preprocessing_pipeline"
            / "instance_1_preprocessing_pipeline.joblib"
        )

        dump.assert_called_once_with(
            pipeline,
            expected_path,
        )
        assert expected_path.parent.is_dir()

    def test_saves_with_custom_path_and_name(
        self,
        tmp_path,
        monkeypatch,
    ):
        pipeline = PreprocessingPipeline()
        data = make_base_data()
        target = tmp_path / "custom" / "nested"

        monkeypatch.setattr(
            pipeline,
            "extract_output_coords_vars",
            Mock(),
        )

        dump = Mock()
        monkeypatch.setattr(
            module.joblib,
            "dump",
            dump,
        )

        pipeline.fit(
            data,
            save=True,
            save_name="pipeline.joblib",
            save_path=target,
        )

        dump.assert_called_once_with(
            pipeline,
            target / "pipeline.joblib",
        )
        assert target.is_dir()

    def test_preserves_existing_save_directory(
        self,
        tmp_path,
        monkeypatch,
    ):
        target = tmp_path / "existing"
        target.mkdir()

        marker = target / "marker.txt"
        marker.write_text("keep")

        pipeline = PreprocessingPipeline()
        data = make_base_data()

        monkeypatch.setattr(
            pipeline,
            "extract_output_coords_vars",
            Mock(),
        )
        monkeypatch.setattr(
            module.joblib,
            "dump",
            Mock(),
        )

        pipeline.fit(
            data,
            save=True,
            save_path=target,
        )

        assert marker.read_text() == "keep"

    def test_does_not_save_when_disabled(
        self,
        monkeypatch,
    ):
        pipeline = PreprocessingPipeline()
        data = make_base_data()

        monkeypatch.setattr(
            pipeline,
            "extract_output_coords_vars",
            Mock(),
        )

        dump = Mock()
        monkeypatch.setattr(
            module.joblib,
            "dump",
            dump,
        )

        pipeline.fit(
            data,
            save=False,
        )

        dump.assert_not_called()

    def test_load_branch_calls_loader(
        self,
        tmp_path,
        monkeypatch,
    ):
        load_path = tmp_path / "pipeline.joblib"

        pipeline = PreprocessingPipeline(
            load_dir=load_path,
        )

        loader = Mock(return_value=pipeline)
        monkeypatch.setattr(
            pipeline,
            "_load_from_memory",
            loader,
            raising=False,
        )

        result = pipeline.fit()

        loader.assert_called_once_with(load_path)
        assert result is pipeline


class TestTransform:
    def make_fitted_pipeline(self):
        first = AddPreprocessor(
            value=1.0,
            fitted=True,
        )
        second = MultiplyPreprocessor(
            value=2.0,
            fitted=True,
        )

        pipeline = PreprocessingPipeline()
        pipeline.fitted = True
        pipeline.steps = [
            "add",
            "multiply",
        ]
        pipeline.fitted_preprocessors = [
            first,
            second,
        ]

        return pipeline, first, second

    def test_empty_pipeline_returns_input(self):
        pipeline = PreprocessingPipeline()
        pipeline.fitted = True
        pipeline.steps = []
        pipeline.fitted_preprocessors = []

        data = xr.DataArray(
            [
                1.0,
                2.0,
            ],
            dims=("samples",),
        )

        result = pipeline.transform(data)

        assert result is data

    def test_applies_steps_in_order(self):
        pipeline, first, second = self.make_fitted_pipeline()

        data = xr.DataArray(
            [
                1.0,
                2.0,
            ],
            dims=("samples",),
        )

        result = pipeline.transform(data)

        xr.testing.assert_identical(
            result,
            (data + 1.0) * 2.0,
        )

        xr.testing.assert_identical(
            first.transform_calls[0]["data"],
            data,
        )
        xr.testing.assert_identical(
            second.transform_calls[0]["data"],
            data + 1.0,
        )

    def test_passes_step_arguments(self):
        pipeline, first, second = self.make_fitted_pipeline()

        data = xr.DataArray(
            [
                1.0,
                2.0,
            ],
            dims=("samples",),
        )

        result = pipeline.transform(
            data,
            step_arguments={
                "add": {
                    "value": 3.0,
                },
                "multiply": {
                    "value": 4.0,
                },
            },
        )

        xr.testing.assert_identical(
            result,
            (data + 3.0) * 4.0,
        )
        assert first.transform_calls[0]["kwargs"] == {
            "value": 3.0,
        }
        assert second.transform_calls[0]["kwargs"] == {
            "value": 4.0,
        }

    def test_unspecified_step_receives_empty_arguments(self):
        pipeline, first, second = self.make_fitted_pipeline()

        data = xr.DataArray(
            [
                1.0,
                2.0,
            ],
            dims=("samples",),
        )

        pipeline.transform(
            data,
            step_arguments={
                "add": {
                    "value": 3.0,
                }
            },
        )

        assert first.transform_calls[0]["kwargs"] == {
            "value": 3.0,
        }
        assert second.transform_calls[0]["kwargs"] == {}

    def test_rejects_unknown_step_arguments(self):
        pipeline, _, _ = self.make_fitted_pipeline()

        with pytest.raises(
            ValueError,
            match="unknown not in preprocessing steps",
        ):
            pipeline.transform(
                xr.DataArray(
                    [
                        1.0,
                    ],
                    dims=("samples",),
                ),
                step_arguments={
                    "unknown": {},
                },
            )


class TestInverseTransform:
    def make_fitted_pipeline(self):
        first = AddPreprocessor(
            value=1.0,
            fitted=True,
        )
        second = MultiplyPreprocessor(
            value=2.0,
            fitted=True,
        )

        pipeline = PreprocessingPipeline()
        pipeline.fitted = True
        pipeline.steps = [
            "add",
            "multiply",
        ]
        pipeline.fitted_preprocessors = [
            first,
            second,
        ]

        return pipeline, first, second

    def test_empty_pipeline_returns_input(self):
        pipeline = PreprocessingPipeline()
        pipeline.fitted = True
        pipeline.steps = []
        pipeline.fitted_preprocessors = []

        data = xr.DataArray(
            [
                1.0,
                2.0,
            ],
            dims=("samples",),
        )

        result = pipeline.inverse_transform(data)

        assert result is data

    def test_applies_steps_in_reverse_order(self):
        pipeline, first, second = self.make_fitted_pipeline()

        original = xr.DataArray(
            [
                1.0,
                2.0,
            ],
            dims=("samples",),
        )
        transformed = (original + 1.0) * 2.0

        result = pipeline.inverse_transform(transformed)

        xr.testing.assert_identical(
            result,
            original,
        )

        xr.testing.assert_identical(
            second.inverse_calls[0]["data"],
            transformed,
        )
        xr.testing.assert_identical(
            first.inverse_calls[0]["data"],
            transformed / 2.0,
        )

    def test_passes_step_arguments(self):
        pipeline, first, second = self.make_fitted_pipeline()

        original = xr.DataArray(
            [
                1.0,
                2.0,
            ],
            dims=("samples",),
        )
        transformed = (original + 3.0) * 4.0

        result = pipeline.inverse_transform(
            transformed,
            step_arguments={
                "add": {
                    "value": 3.0,
                },
                "multiply": {
                    "value": 4.0,
                },
            },
        )

        xr.testing.assert_identical(
            result,
            original,
        )
        assert first.inverse_calls[0]["kwargs"] == {
            "value": 3.0,
        }
        assert second.inverse_calls[0]["kwargs"] == {
            "value": 4.0,
        }

    def test_rejects_unknown_step_arguments(self):
        pipeline, _, _ = self.make_fitted_pipeline()

        with pytest.raises(
            ValueError,
            match="unknown not in preprocessing steps",
        ):
            pipeline.inverse_transform(
                xr.DataArray(
                    [
                        1.0,
                    ],
                    dims=("samples",),
                ),
                step_arguments={
                    "unknown": {},
                },
            )


class TestGetPreprocessors:
    def make_pipeline(self):
        first = AddPreprocessor(fitted=True)
        second = MultiplyPreprocessor(fitted=True)

        pipeline = PreprocessingPipeline()
        pipeline.fitted = True
        pipeline.steps = [
            "add",
            "multiply",
        ]
        pipeline.fitted_preprocessors = [
            first,
            second,
        ]

        return pipeline, first, second

    def test_requires_fitted_pipeline(self):
        pipeline = PreprocessingPipeline()

        with pytest.raises(
            RuntimeError,
            match="Pipeline needs to be fitted first",
        ):
            pipeline.get_preprocessors()

    def test_returns_all_preprocessors(self):
        pipeline, first, second = self.make_pipeline()

        result = pipeline.get_preprocessors()

        assert result == [
            first,
            second,
        ]

    def test_returns_requested_preprocessor(self):
        pipeline, first, _ = self.make_pipeline()

        result = pipeline.get_preprocessors("add")

        assert result is first

    def test_rejects_missing_name(self):
        pipeline, _, _ = self.make_pipeline()

        with pytest.raises(
            ValueError,
            match="'missing' not in preprocessing steps",
        ):
            pipeline.get_preprocessors("missing")

    def test_rejects_duplicate_names(self):
        pipeline, first, second = self.make_pipeline()
        pipeline.steps = [
            "same",
            "same",
        ]
        pipeline.fitted_preprocessors = [
            first,
            second,
        ]

        with pytest.raises(
            ValueError,
            match="Expected exactly one preprocessor",
        ):
            pipeline.get_preprocessors("same")


class TestAddFittedPreprocessor:
    def make_pipeline(self):
        pipeline = PreprocessingPipeline()
        pipeline.fitted = True
        pipeline.steps = []
        pipeline.fitted_preprocessors = []
        return pipeline

    def test_rejects_unfitted_preprocessor(self):
        pipeline = self.make_pipeline()
        preprocessor = AddPreprocessor(fitted=False)

        with pytest.raises(
            AssertionError,
            match="must be fitted",
        ):
            pipeline.add_fitted_preprocessor(
                preprocessor,
                "add",
            )

    def test_appends_preprocessor(self):
        pipeline = self.make_pipeline()
        preprocessor = AddPreprocessor(fitted=True)

        result = pipeline.add_fitted_preprocessor(
            preprocessor,
            "add",
        )

        assert result is None
        assert pipeline.steps == [
            "add",
        ]
        assert pipeline.fitted_preprocessors == [
            preprocessor,
        ]

    def test_inserts_preprocessor_at_index(self):
        pipeline = self.make_pipeline()

        existing = MultiplyPreprocessor(fitted=True)
        inserted = AddPreprocessor(fitted=True)

        pipeline.steps = [
            "multiply",
        ]
        pipeline.fitted_preprocessors = [
            existing,
        ]

        pipeline.add_fitted_preprocessor(
            inserted,
            "add",
            index=0,
        )

        assert pipeline.steps == [
            "add",
            "multiply",
        ]
        assert pipeline.fitted_preprocessors == [
            inserted,
            existing,
        ]


class TestExtractOutputCoordinates:
    def test_requires_fitted_pipeline(self):
        pipeline = PreprocessingPipeline()

        with pytest.raises(
            ValueError,
            match="fitted pipeline",
        ):
            pipeline.extract_output_coords_vars(make_base_data())

    def test_extracts_supported_dimensions(self):
        pipeline = PreprocessingPipeline()
        pipeline.fitted = True
        pipeline.supported_NN_dimensions = (
            "lat",
            "lon",
            "missing",
        )

        data = make_base_data()

        pipeline.extract_output_coords_vars(data)

        assert set(pipeline.reference_coords) == {
            "lat",
            "lon",
        }
        xr.testing.assert_identical(
            pipeline.reference_coords["lat"],
            data["lat"],
        )
        xr.testing.assert_identical(
            pipeline.reference_coords["lon"],
            data["lon"],
        )

    def test_extracts_dataset_variable_names(self):
        pipeline = PreprocessingPipeline()
        pipeline.fitted = True

        data = make_base_data()

        pipeline.extract_output_coords_vars(data)

        assert pipeline.reference_var == [
            "tas",
            "pr",
        ]

    def test_dataarray_has_no_dataset_variables(self):
        pipeline = PreprocessingPipeline()
        pipeline.fitted = True

        data = xr.DataArray(
            np.ones(
                (
                    2,
                    2,
                )
            ),
            dims=(
                "lat",
                "lon",
            ),
            coords={
                "lat": [
                    45.0,
                    46.0,
                ],
                "lon": [
                    -124.0,
                    -123.0,
                ],
            },
            name="tas",
        )

        with pytest.raises(AttributeError):
            pipeline.extract_output_coords_vars(data)



class TestLoadFromMemory:
    def make_loaded_pipeline(self):
        loaded = PreprocessingPipeline()
        loaded.preprocessors_list = [
            "selector",
        ]
        loaded.pipeline = [
            (
                "add",
                "preprocessor",
            )
        ]
        loaded.steps = [
            "add",
        ]
        loaded.fitted_preprocessors = [
            "fitted-preprocessor",
        ]
        loaded.fitted = True
        loaded.fitted_based_time = np.asarray(
            [
                "2000-01-01",
            ],
            dtype="datetime64[ns]",
        )
        loaded.reference_coords = {
            "lat": xr.DataArray(
                [
                    45.0,
                ],
                dims=("lat",),
            )
        }
        loaded.reference_var = [
            "tas",
        ]
        loaded.name = "loaded"
        return loaded

    def test_rejects_unfitted_pipeline(
        self,
        tmp_path,
        monkeypatch,
    ):
        loaded = self.make_loaded_pipeline()
        loaded.fitted = False

        monkeypatch.setattr(
            module.joblib,
            "load",
            Mock(return_value=loaded),
        )

        pipeline = PreprocessingPipeline()

        with pytest.raises(
            ValueError,
            match="has to be fitted first",
        ):
            pipeline.load_from_memory(tmp_path / "pipeline.joblib")

    def test_copies_loaded_state(
        self,
        tmp_path,
        monkeypatch,
    ):
        loaded = self.make_loaded_pipeline()

        load = Mock(return_value=loaded)
        monkeypatch.setattr(
            module.joblib,
            "load",
            load,
        )

        pipeline = PreprocessingPipeline()

        result = pipeline.load_from_memory(tmp_path / "pipeline.joblib")

        assert result is pipeline
        assert pipeline.preprocessors_list == [
            "selector",
        ]
        assert pipeline.pipeline == [
            (
                "add",
                "preprocessor",
            )
        ]
        assert pipeline.steps == [
            "add",
        ]
        assert pipeline.fitted_preprocessors == [
            "fitted-preprocessor",
        ]
        assert pipeline.fitted is True
        assert pipeline.name == "loaded"
        assert pipeline.reference_var == [
            "tas",
        ]

    def test_accepts_string_path(
        self,
        monkeypatch,
    ):
        loaded = self.make_loaded_pipeline()

        load = Mock(return_value=loaded)
        monkeypatch.setattr(
            module.joblib,
            "load",
            load,
        )

        pipeline = PreprocessingPipeline()

        pipeline.load_from_memory("/tmp/pipeline.joblib")

        load.assert_called_once_with(Path("/tmp/pipeline.joblib"))


class TestToDataset:
    def make_pipeline(self):
        pipeline = PreprocessingPipeline()
        pipeline.fitted = True
        pipeline.steps = []
        pipeline.fitted_preprocessors = []
        pipeline.reference_var = [
            "tas",
            "pr",
        ]
        pipeline.reference_coords = {
            "lat": xr.DataArray(
                [
                    45.0,
                    46.0,
                ],
                dims=("lat",),
            ),
            "lon": xr.DataArray(
                [
                    -124.0,
                    -123.0,
                ],
                dims=("lon",),
            ),
        }
        return pipeline

    def test_rejects_channel_mismatch(self):
        pipeline = self.make_pipeline()

        data = xr.DataArray(
            np.ones(
                (
                    1,
                    2,
                    2,
                )
            ),
            dims=(
                "channels",
                "output_dim_0",
                "output_dim_1",
            ),
            coords={
                "channels": [
                    0,
                ],
            },
        )

        with pytest.raises(
            ValueError,
            match="does not match the preprocessing pipeline",
        ):
            pipeline.to_dataset(data)

    def test_reconstructs_dataset_without_flattener(self):
        pipeline = self.make_pipeline()

        data = xr.DataArray(
            np.arange(
                8,
                dtype=float,
            ).reshape(
                2,
                2,
                2,
            ),
            dims=(
                "channels",
                "output_dim_0",
                "output_dim_1",
            ),
            coords={
                "channels": [
                    0,
                    1,
                ],
            },
        )

        result = pipeline.to_dataset(data)

        assert isinstance(
            result,
            xr.Dataset,
        )
        assert set(result.data_vars) == {
            "tas",
            "pr",
        }
        assert result["tas"].dims == (
            "lat",
            "lon",
        )
        np.testing.assert_array_equal(
            result["lat"].values,
            [
                45.0,
                46.0,
            ],
        )
        np.testing.assert_array_equal(
            result["lon"].values,
            [
                -124.0,
                -123.0,
            ],
        )

    def test_reconstructs_dataset_with_flattener(
        self,
        monkeypatch,
    ):
        pipeline = self.make_pipeline()

        class FakeFlattennanremove:
            def __init__(self):
                self.fitted = True
                self.final_locations = xr.DataArray(
                    [
                        10,
                        20,
                        30,
                    ],
                    dims=("ref",),
                )

        flattener = FakeFlattennanremove()
        pipeline.steps = [
            "flattener",
        ]
        pipeline.fitted_preprocessors = [
            flattener,
        ]

        monkeypatch.setattr(
            "cccma_ppp.preprocessing.utils_preprocessing.Flattennanremove",
            FakeFlattennanremove,
        )

        data = xr.DataArray(
            np.arange(
                6,
                dtype=float,
            ).reshape(
                2,
                3,
            ),
            dims=(
                "channels",
                "output_dim_0",
            ),
            coords={
                "channels": [
                    0,
                    1,
                ],
            },
        )

        result = pipeline.to_dataset(data)

        assert set(result.data_vars) == {
            "tas",
            "pr",
        }
        assert result["tas"].dims == ("ref",)
        np.testing.assert_array_equal(
            result["ref"].values,
            [
                10,
                20,
                30,
            ],
        )
