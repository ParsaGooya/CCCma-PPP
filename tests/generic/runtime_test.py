from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from cccma_ppp.generic.runtime import RuntimeContext


class TestRuntimeContextDefaults:
    # Remove test due to no coverage

    @pytest.mark.pruned
    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(RuntimeContext)

    @pytest.mark.pruned
    def test_default_experiment_directory_is_current_directory(
        self,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.chdir(tmp_path)

        context = RuntimeContext()

        assert context.GLOBAL_EXP_DIR == tmp_path
        assert isinstance(
            context.GLOBAL_EXP_DIR,
            Path,
        )

    @pytest.mark.pruned
    def test_default_checkpoint_directory(
        self,
        tmp_path,
    ):
        context = RuntimeContext(
            GLOBAL_EXP_DIR=tmp_path,
        )

        assert context.GLOBAL_CHECKPOINT_DIR == (tmp_path / "checkpoints")

    @pytest.mark.pruned
    def test_default_figures_directory(
        self,
        tmp_path,
    ):
        context = RuntimeContext(
            GLOBAL_EXP_DIR=tmp_path,
        )

        assert context.GLOBAL_FIGURES_DIR == (tmp_path / "figures")

    @pytest.mark.pruned
    def test_default_log_directory(
        self,
        tmp_path,
    ):
        context = RuntimeContext(
            GLOBAL_EXP_DIR=tmp_path,
        )

        assert context.GLOBAL_LOG_DIR == (tmp_path / "logs")

    @pytest.mark.pruned
    def test_default_monitoring_directory(
        self,
        tmp_path,
    ):
        context = RuntimeContext(
            GLOBAL_EXP_DIR=tmp_path,
        )

        assert context.GLOBAL_MONITORING_DIR == (tmp_path / "resource_monitoring")

    @pytest.mark.pruned
    def test_default_output_directory(
        self,
        tmp_path,
    ):
        context = RuntimeContext(
            GLOBAL_EXP_DIR=tmp_path,
        )

        assert context.GLOBAL_OUTPUT_DIR == (tmp_path / "inference")

    @pytest.mark.pruned
    def test_all_default_subdirectories_are_derived_from_experiment_directory(
        self,
        tmp_path,
    ):
        experiment_directory = tmp_path / "experiment"

        context = RuntimeContext(
            GLOBAL_EXP_DIR=experiment_directory,
        )

        assert context.GLOBAL_CHECKPOINT_DIR == (experiment_directory / "checkpoints")
        assert context.GLOBAL_FIGURES_DIR == (experiment_directory / "figures")
        assert context.GLOBAL_LOG_DIR == (experiment_directory / "logs")
        assert context.GLOBAL_MONITORING_DIR == (
            experiment_directory / "resource_monitoring"
        )
        assert context.GLOBAL_OUTPUT_DIR == (experiment_directory / "inference")

    @pytest.mark.pruned
    def test_default_metadata_is_none(
        self,
        tmp_path,
    ):
        context = RuntimeContext(
            GLOBAL_EXP_DIR=tmp_path,
        )

        assert context.INPUT_VAR_METADATA is None
        assert context.TARGET_VAR_METADATA is None


class TestRuntimeContextCustomDirectories:
    def test_preserves_custom_checkpoint_directory(
        self,
        tmp_path,
    ):
        custom = tmp_path / "custom-checkpoints"

        context = RuntimeContext(
            GLOBAL_EXP_DIR=tmp_path,
            GLOBAL_CHECKPOINT_DIR=custom,
        )

        assert context.GLOBAL_CHECKPOINT_DIR is custom

    @pytest.mark.pruned
    def test_preserves_custom_figures_directory(
        self,
        tmp_path,
    ):
        custom = tmp_path / "custom-figures"

        context = RuntimeContext(
            GLOBAL_EXP_DIR=tmp_path,
            GLOBAL_FIGURES_DIR=custom,
        )

        assert context.GLOBAL_FIGURES_DIR is custom

    @pytest.mark.pruned
    def test_preserves_custom_log_directory(
        self,
        tmp_path,
    ):
        custom = tmp_path / "custom-logs"

        context = RuntimeContext(
            GLOBAL_EXP_DIR=tmp_path,
            GLOBAL_LOG_DIR=custom,
        )

        assert context.GLOBAL_LOG_DIR is custom

    @pytest.mark.pruned
    def test_preserves_custom_monitoring_directory(
        self,
        tmp_path,
    ):
        custom = tmp_path / "custom-monitoring"

        context = RuntimeContext(
            GLOBAL_EXP_DIR=tmp_path,
            GLOBAL_MONITORING_DIR=custom,
        )

        assert context.GLOBAL_MONITORING_DIR is custom

    @pytest.mark.pruned
    def test_preserves_custom_output_directory(
        self,
        tmp_path,
    ):
        custom = tmp_path / "custom-output"

        context = RuntimeContext(
            GLOBAL_EXP_DIR=tmp_path,
            GLOBAL_OUTPUT_DIR=custom,
        )

        assert context.GLOBAL_OUTPUT_DIR is custom

    @pytest.mark.pruned
    def test_preserves_all_custom_directories(
        self,
        tmp_path,
    ):
        checkpoint_directory = tmp_path / "checkpoints-custom"
        figures_directory = tmp_path / "figures-custom"
        log_directory = tmp_path / "logs-custom"
        monitoring_directory = tmp_path / "monitoring-custom"
        output_directory = tmp_path / "output-custom"

        context = RuntimeContext(
            GLOBAL_EXP_DIR=tmp_path,
            GLOBAL_CHECKPOINT_DIR=checkpoint_directory,
            GLOBAL_FIGURES_DIR=figures_directory,
            GLOBAL_LOG_DIR=log_directory,
            GLOBAL_MONITORING_DIR=monitoring_directory,
            GLOBAL_OUTPUT_DIR=output_directory,
        )

        assert context.GLOBAL_CHECKPOINT_DIR is checkpoint_directory
        assert context.GLOBAL_FIGURES_DIR is figures_directory
        assert context.GLOBAL_LOG_DIR is log_directory
        assert context.GLOBAL_MONITORING_DIR is monitoring_directory
        assert context.GLOBAL_OUTPUT_DIR is output_directory

    def test_custom_string_subdirectories_are_preserved(
        self,
        tmp_path,
    ):
        context = RuntimeContext(
            GLOBAL_EXP_DIR=tmp_path,
            GLOBAL_CHECKPOINT_DIR="checkpoints",
            GLOBAL_FIGURES_DIR="figures",
            GLOBAL_LOG_DIR="logs",
            GLOBAL_MONITORING_DIR="monitoring",
            GLOBAL_OUTPUT_DIR="output",
        )

        assert context.GLOBAL_CHECKPOINT_DIR == "checkpoints"
        assert context.GLOBAL_FIGURES_DIR == "figures"
        assert context.GLOBAL_LOG_DIR == "logs"
        assert context.GLOBAL_MONITORING_DIR == "monitoring"
        assert context.GLOBAL_OUTPUT_DIR == "output"


class TestRuntimeContextMetadata:
    @pytest.mark.pruned
    def test_preserves_input_metadata(
        self,
        tmp_path,
    ):
        metadata = {
            "variables": [
                "tas",
                "pr",
            ],
            "NN_dims": [
                "channels",
                "lat",
                "lon",
            ],
        }

        context = RuntimeContext(
            GLOBAL_EXP_DIR=tmp_path,
            INPUT_VAR_METADATA=metadata,
        )

        assert context.INPUT_VAR_METADATA is metadata

    @pytest.mark.pruned
    def test_preserves_target_metadata(
        self,
        tmp_path,
    ):
        metadata = {
            "variables": [
                "tas",
            ],
            "preprocessors": [
                [
                    "standardscaler",
                ]
            ],
        }

        context = RuntimeContext(
            GLOBAL_EXP_DIR=tmp_path,
            TARGET_VAR_METADATA=metadata,
        )

        assert context.TARGET_VAR_METADATA is metadata

    @pytest.mark.pruned
    def test_input_and_target_metadata_are_independent(
        self,
        tmp_path,
    ):
        input_metadata = {
            "variables": [
                "model",
            ]
        }
        target_metadata = {
            "variables": [
                "observation",
            ]
        }

        context = RuntimeContext(
            GLOBAL_EXP_DIR=tmp_path,
            INPUT_VAR_METADATA=input_metadata,
            TARGET_VAR_METADATA=target_metadata,
        )

        assert context.INPUT_VAR_METADATA is input_metadata
        assert context.TARGET_VAR_METADATA is target_metadata
        assert context.INPUT_VAR_METADATA is not context.TARGET_VAR_METADATA


class TestRuntimeContextInitializationBehavior:
    @pytest.mark.pruned
    def test_does_not_create_directories(
        self,
        tmp_path,
    ):
        experiment_directory = tmp_path / "experiment"

        context = RuntimeContext(
            GLOBAL_EXP_DIR=experiment_directory,
        )

        assert context.GLOBAL_CHECKPOINT_DIR == experiment_directory / "checkpoints"
        assert not experiment_directory.exists()
        assert not (experiment_directory / "checkpoints").exists()
        assert not (experiment_directory / "figures").exists()
        assert not (experiment_directory / "logs").exists()
        assert not (experiment_directory / "resource_monitoring").exists()
        assert not (experiment_directory / "inference").exists()

    @pytest.mark.pruned
    def test_instances_are_independent(
        self,
        tmp_path,
    ):
        first = RuntimeContext(
            GLOBAL_EXP_DIR=tmp_path / "first",
        )
        second = RuntimeContext(
            GLOBAL_EXP_DIR=tmp_path / "second",
        )

        assert first.GLOBAL_EXP_DIR != (second.GLOBAL_EXP_DIR)
        assert first.GLOBAL_LOG_DIR != (second.GLOBAL_LOG_DIR)
        assert first.GLOBAL_OUTPUT_DIR != (second.GLOBAL_OUTPUT_DIR)

    @pytest.mark.pruned
    def test_explicit_custom_directory_does_not_affect_other_defaults(
        self,
        tmp_path,
    ):
        experiment_directory = tmp_path / "experiment"
        custom_log_directory = tmp_path / "external-logs"

        context = RuntimeContext(
            GLOBAL_EXP_DIR=experiment_directory,
            GLOBAL_LOG_DIR=custom_log_directory,
        )

        assert context.GLOBAL_LOG_DIR is custom_log_directory
        assert context.GLOBAL_CHECKPOINT_DIR == (experiment_directory / "checkpoints")
        assert context.GLOBAL_FIGURES_DIR == (experiment_directory / "figures")
        assert context.GLOBAL_MONITORING_DIR == (
            experiment_directory / "resource_monitoring"
        )
        assert context.GLOBAL_OUTPUT_DIR == (experiment_directory / "inference")

    @pytest.mark.pruned
    def test_explicit_none_subdirectories_trigger_defaults(
        self,
        tmp_path,
    ):
        context = RuntimeContext(
            GLOBAL_EXP_DIR=tmp_path,
            GLOBAL_CHECKPOINT_DIR=None,
            GLOBAL_FIGURES_DIR=None,
            GLOBAL_LOG_DIR=None,
            GLOBAL_MONITORING_DIR=None,
            GLOBAL_OUTPUT_DIR=None,
        )

        assert context.GLOBAL_CHECKPOINT_DIR == (tmp_path / "checkpoints")
        assert context.GLOBAL_FIGURES_DIR == (tmp_path / "figures")
        assert context.GLOBAL_LOG_DIR == (tmp_path / "logs")
        assert context.GLOBAL_MONITORING_DIR == (tmp_path / "resource_monitoring")
        assert context.GLOBAL_OUTPUT_DIR == (tmp_path / "inference")


class TestRuntimeContextStringExperimentDirectory:
    @pytest.mark.pruned
    def test_string_experiment_directory_currently_raises_type_error(
        self,
    ):
        with pytest.raises(
            TypeError,
            match="unsupported operand type",
        ):
            RuntimeContext(
                GLOBAL_EXP_DIR="/tmp/experiment",
            )

    @pytest.mark.pruned
    def test_string_experiment_directory_works_when_all_subdirectories_are_explicit(
        self,
    ):
        context = RuntimeContext(
            GLOBAL_EXP_DIR="/tmp/experiment",
            GLOBAL_CHECKPOINT_DIR="/tmp/checkpoints",
            GLOBAL_FIGURES_DIR="/tmp/figures",
            GLOBAL_LOG_DIR="/tmp/logs",
            GLOBAL_MONITORING_DIR="/tmp/monitoring",
            GLOBAL_OUTPUT_DIR="/tmp/output",
        )

        assert context.GLOBAL_EXP_DIR == "/tmp/experiment"
        assert context.GLOBAL_CHECKPOINT_DIR == "/tmp/checkpoints"
        assert context.GLOBAL_FIGURES_DIR == "/tmp/figures"
        assert context.GLOBAL_LOG_DIR == "/tmp/logs"
        assert context.GLOBAL_MONITORING_DIR == "/tmp/monitoring"
        assert context.GLOBAL_OUTPUT_DIR == "/tmp/output"


class TestRuntimeContextDataclassBehavior:
    # Remove test due to no coverage

    @pytest.mark.pruned
    def test_declared_field_order(self):
        field_names = [field.name for field in dataclasses.fields(RuntimeContext)]

        assert field_names == [
            "GLOBAL_EXP_DIR",
            "GLOBAL_CHECKPOINT_DIR",
            "GLOBAL_FIGURES_DIR",
            "GLOBAL_LOG_DIR",
            "GLOBAL_MONITORING_DIR",
            "GLOBAL_OUTPUT_DIR",
            "INPUT_VAR_METADATA",
            "TARGET_VAR_METADATA",
        ]

    @pytest.mark.pruned
    def test_equal_contexts_compare_equal(
        self,
        tmp_path,
    ):
        first = RuntimeContext(
            GLOBAL_EXP_DIR=tmp_path,
        )
        second = RuntimeContext(
            GLOBAL_EXP_DIR=tmp_path,
        )

        assert first == second

    @pytest.mark.pruned
    def test_different_contexts_compare_not_equal(
        self,
        tmp_path,
    ):
        first = RuntimeContext(
            GLOBAL_EXP_DIR=tmp_path / "first",
        )
        second = RuntimeContext(
            GLOBAL_EXP_DIR=tmp_path / "second",
        )

        assert first != second

    @pytest.mark.pruned
    def test_repr_contains_derived_directories(
        self,
        tmp_path,
    ):
        context = RuntimeContext(
            GLOBAL_EXP_DIR=tmp_path,
        )

        representation = repr(context)

        assert "GLOBAL_EXP_DIR" in representation
        assert "GLOBAL_CHECKPOINT_DIR" in representation
        assert "GLOBAL_OUTPUT_DIR" in representation
        assert str(tmp_path / "checkpoints") in representation
