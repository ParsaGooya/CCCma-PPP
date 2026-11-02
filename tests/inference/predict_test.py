from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

import S2D4P.inference.predict as predict_mod
from S2D4P.inference.predict import get_parser, main


class DummyDistributed:
    local_rank = 0
    world_size = 1

    def __init__(
        self,
        *,
        root=True,
        rank=0,
    ):
        self._root = root
        self.rank = rank
        self.cleanup_called = False

    def is_root(self):
        return self._root

    def cleanup(self):
        self.cleanup_called = True


class DummyLogger:
    def __init__(self):
        self.messages = []

    def info(
        self,
        message,
        *args,
        **kwargs,
    ):
        self.messages.append(message)


class DummyInferenceConfig:
    def __init__(self):
        self.log_dir = "logs"

        self.seed_calls = []
        self.prepare_directory_calls = []

    def set_random_seed(
        self,
        rank,
    ):
        self.seed_calls.append(rank)

    def prepare_directory(
        self,
        distributed,
    ):
        self.prepare_directory_calls.append(distributed)


class DummyWriter:
    def __init__(self):
        self.setup_calls = []
        self.predict_called = False

    def setup_distributed(
        self,
        *,
        distributed,
        logger,
    ):
        self.setup_calls.append(
            {
                "distributed": distributed,
                "logger": logger,
            }
        )

    def predict(self):
        self.predict_called = True


def patch_successful_main_dependencies(
    monkeypatch,
    *,
    distributed=None,
    config=None,
    logger=None,
    writer=None,
    config_data=None,
):
    distributed = distributed if distributed is not None else DummyDistributed()
    config = config if config is not None else DummyInferenceConfig()
    logger = logger if logger is not None else DummyLogger()
    writer = writer if writer is not None else DummyWriter()
    config_data = (
        config_data
        if config_data is not None
        else {
            "experiment_dir": "test",
        }
    )

    get_instance = Mock(
        return_value=distributed,
    )
    prepare_config = Mock(
        return_value=config_data,
    )
    from_dict = Mock(
        return_value=config,
    )
    setup_logger = Mock(
        return_value=logger,
    )
    build_writer = Mock(
        return_value=writer,
    )

    monkeypatch.setattr(
        predict_mod.Distributed,
        "get_instance",
        get_instance,
    )
    monkeypatch.setattr(
        predict_mod,
        "prepare_config",
        prepare_config,
    )
    monkeypatch.setattr(
        predict_mod.dacite,
        "from_dict",
        from_dict,
    )
    monkeypatch.setattr(
        predict_mod,
        "setup_logger",
        setup_logger,
    )
    monkeypatch.setattr(
        predict_mod,
        "build_writer",
        build_writer,
    )

    return {
        "distributed": distributed,
        "config": config,
        "logger": logger,
        "writer": writer,
        "config_data": config_data,
        "get_instance": get_instance,
        "prepare_config": prepare_config,
        "from_dict": from_dict,
        "setup_logger": setup_logger,
        "build_writer": build_writer,
    }


class TestGetParser:
    @pytest.mark.pruned
    def test_returns_argument_parser(self):
        parser = get_parser()

        assert parser.prog
        assert parser.description == "Run inference from a configuration file"

    @pytest.mark.pruned
    def test_parses_config_path(self):
        parser = get_parser()

        arguments = parser.parse_args(
            [
                "inference.yaml",
            ]
        )

        assert arguments.config == "inference.yaml"

    @pytest.mark.pruned
    def test_config_argument_is_string(self):
        parser = get_parser()

        arguments = parser.parse_args(
            [
                str(Path("inference.yaml")),
            ]
        )

        assert arguments.config == "inference.yaml"
        assert isinstance(
            arguments.config,
            str,
        )

    @pytest.mark.pruned
    def test_accepts_relative_config_path(self):
        parser = get_parser()

        arguments = parser.parse_args(
            [
                "configs/inference.yaml",
            ]
        )

        assert arguments.config == "configs/inference.yaml"

    @pytest.mark.pruned
    def test_accepts_absolute_config_path(
        self,
        tmp_path,
    ):
        parser = get_parser()
        config_path = tmp_path / "inference.yaml"

        arguments = parser.parse_args(
            [
                str(config_path),
            ]
        )

        assert arguments.config == str(config_path)

    @pytest.mark.pruned
    def test_requires_config_argument(self):
        parser = get_parser()

        with pytest.raises(
            SystemExit,
        ) as error:
            parser.parse_args([])

        assert error.value.code == 2

    @pytest.mark.pruned
    def test_rejects_unknown_arguments(self):
        parser = get_parser()

        with pytest.raises(
            SystemExit,
        ) as error:
            parser.parse_args(
                [
                    "inference.yaml",
                    "--unknown",
                ]
            )

        assert error.value.code == 2

    @pytest.mark.pruned
    def test_help_exits_successfully(
        self,
        capsys,
    ):
        parser = get_parser()

        with pytest.raises(
            SystemExit,
        ) as error:
            parser.parse_args(
                [
                    "--help",
                ]
            )

        assert error.value.code == 0

        output = capsys.readouterr().out

        assert "Run inference from a configuration file" in output
        assert "Path to the YAML config file" in output
        assert "config" in output


class TestMainHappyPath:
    @pytest.mark.pruned
    def test_root_happy_path(
        self,
        monkeypatch,
        tmp_path,
    ):
        yaml_path = tmp_path / "inference.yaml"

        dependencies = patch_successful_main_dependencies(
            monkeypatch,
        )

        main(str(yaml_path))

        distributed = dependencies["distributed"]
        config = dependencies["config"]
        logger = dependencies["logger"]
        writer = dependencies["writer"]

        assert config.seed_calls == [
            distributed.rank,
        ]
        assert config.prepare_directory_calls == [
            distributed,
        ]
        assert writer.setup_calls == [
            {
                "distributed": distributed,
                "logger": logger,
            }
        ]
        assert writer.predict_called is True
        assert distributed.cleanup_called is True

        assert "Setting up directories ..." in logger.messages
        assert "Building objects:" in logger.messages

    def test_non_root_skips_root_logging(
        self,
        monkeypatch,
    ):
        distributed = DummyDistributed(
            root=False,
        )
        logger = DummyLogger()

        dependencies = patch_successful_main_dependencies(
            monkeypatch,
            distributed=distributed,
            logger=logger,
        )

        main("inference.yaml")

        assert logger.messages == []
        assert dependencies["writer"].predict_called is True
        assert distributed.cleanup_called is True

    @pytest.mark.pruned
    def test_returns_none(
        self,
        monkeypatch,
    ):
        patch_successful_main_dependencies(
            monkeypatch,
        )

        result = main("inference.yaml")

        assert result is None


class TestMainDependencyArguments:
    @pytest.mark.pruned
    def test_gets_distributed_instance_once(
        self,
        monkeypatch,
    ):
        dependencies = patch_successful_main_dependencies(
            monkeypatch,
        )

        main("inference.yaml")

        dependencies["get_instance"].assert_called_once_with()

    @pytest.mark.pruned
    def test_passes_yaml_path_to_prepare_config(
        self,
        monkeypatch,
        tmp_path,
    ):
        yaml_path = tmp_path / "inference.yaml"

        dependencies = patch_successful_main_dependencies(
            monkeypatch,
        )

        main(str(yaml_path))

        dependencies["prepare_config"].assert_called_once_with(str(yaml_path))

    @pytest.mark.pruned
    def test_passes_yaml_data_to_dacite(
        self,
        monkeypatch,
    ):
        config_data = {
            "experiment_dir": "experiment",
            "num_output_sampling": 4,
        }

        dependencies = patch_successful_main_dependencies(
            monkeypatch,
            config_data=config_data,
        )

        main("inference.yaml")

        call = dependencies["from_dict"].call_args

        assert call.kwargs["data_class"] is predict_mod.InferenceConfig
        assert call.kwargs["data"] is config_data

    @pytest.mark.pruned
    def test_uses_strict_dacite_configuration(
        self,
        monkeypatch,
    ):
        dependencies = patch_successful_main_dependencies(
            monkeypatch,
        )

        main("inference.yaml")

        dacite_config = dependencies["from_dict"].call_args.kwargs["config"]

        assert isinstance(
            dacite_config,
            predict_mod.dacite.Config,
        )
        assert dacite_config.strict is True

    @pytest.mark.pruned
    def test_sets_seed_with_distributed_rank(
        self,
        monkeypatch,
    ):
        distributed = DummyDistributed(
            rank=7,
        )
        config = DummyInferenceConfig()
        config.set_random_seed = Mock()

        patch_successful_main_dependencies(
            monkeypatch,
            distributed=distributed,
            config=config,
        )

        main("inference.yaml")

        config.set_random_seed.assert_called_once_with(7)

    @pytest.mark.pruned
    def test_setup_logger_receives_exact_arguments(
        self,
        monkeypatch,
    ):
        config = DummyInferenceConfig()
        config.log_dir = "/tmp/inference-logs"

        dependencies = patch_successful_main_dependencies(
            monkeypatch,
            config=config,
        )

        main("inference.yaml")

        dependencies["setup_logger"].assert_called_once_with(
            name="inference",
            log_dir=config.log_dir,
        )

    @pytest.mark.pruned
    def test_prepare_directory_receives_distributed(
        self,
        monkeypatch,
    ):
        config = DummyInferenceConfig()
        config.prepare_directory = Mock()

        dependencies = patch_successful_main_dependencies(
            monkeypatch,
            config=config,
        )

        main("inference.yaml")

        config.prepare_directory.assert_called_once_with(dependencies["distributed"])

    @pytest.mark.pruned
    def test_build_writer_receives_exact_arguments(
        self,
        monkeypatch,
    ):
        dependencies = patch_successful_main_dependencies(
            monkeypatch,
        )

        main("inference.yaml")

        dependencies["build_writer"].assert_called_once_with(
            dependencies["config"],
            dependencies["distributed"],
            dependencies["logger"],
        )

    @pytest.mark.pruned
    def test_writer_setup_receives_exact_arguments(
        self,
        monkeypatch,
    ):
        writer = DummyWriter()
        writer.setup_distributed = Mock()

        dependencies = patch_successful_main_dependencies(
            monkeypatch,
            writer=writer,
        )

        main("inference.yaml")

        writer.setup_distributed.assert_called_once_with(
            distributed=dependencies["distributed"],
            logger=dependencies["logger"],
        )

    @pytest.mark.pruned
    def test_writer_predict_is_called_once(
        self,
        monkeypatch,
    ):
        writer = DummyWriter()
        writer.predict = Mock()

        patch_successful_main_dependencies(
            monkeypatch,
            writer=writer,
        )

        main("inference.yaml")

        writer.predict.assert_called_once_with()


class TestMainExecutionOrder:
    @pytest.mark.pruned
    def test_seed_precedes_logger_setup(
        self,
        monkeypatch,
    ):
        events = []
        config = DummyInferenceConfig()
        logger = DummyLogger()

        config.set_random_seed = Mock(
            side_effect=lambda rank: events.append("seed"),
        )

        dependencies = patch_successful_main_dependencies(
            monkeypatch,
            config=config,
            logger=logger,
        )

        dependencies["setup_logger"].side_effect = lambda **kwargs: (
            events.append("setup_logger"),
            logger,
        )[1]

        main("inference.yaml")

        assert events == [
            "seed",
            "setup_logger",
        ]

    @pytest.mark.pruned
    def test_directory_preparation_precedes_writer_build(
        self,
        monkeypatch,
    ):
        events = []
        config = DummyInferenceConfig()
        writer = DummyWriter()

        config.prepare_directory = Mock(
            side_effect=lambda distributed: events.append("prepare_directory"),
        )

        dependencies = patch_successful_main_dependencies(
            monkeypatch,
            config=config,
            writer=writer,
        )

        dependencies["build_writer"].side_effect = lambda *args: (
            events.append("build_writer"),
            writer,
        )[1]

        main("inference.yaml")

        assert events == [
            "prepare_directory",
            "build_writer",
        ]

    @pytest.mark.pruned
    def test_writer_setup_precedes_prediction(
        self,
        monkeypatch,
    ):
        events = []
        writer = DummyWriter()

        writer.setup_distributed = Mock(
            side_effect=lambda **kwargs: events.append("setup"),
        )
        writer.predict = Mock(
            side_effect=lambda: events.append("predict"),
        )

        patch_successful_main_dependencies(
            monkeypatch,
            writer=writer,
        )

        main("inference.yaml")

        assert events == [
            "setup",
            "predict",
        ]

    @pytest.mark.pruned
    def test_cleanup_occurs_after_prediction(
        self,
        monkeypatch,
    ):
        events = []
        distributed = DummyDistributed()
        writer = DummyWriter()

        writer.predict = Mock(
            side_effect=lambda: events.append("predict"),
        )
        distributed.cleanup = Mock(
            side_effect=lambda: events.append("cleanup"),
        )

        patch_successful_main_dependencies(
            monkeypatch,
            distributed=distributed,
            writer=writer,
        )

        main("inference.yaml")

        assert events == [
            "predict",
            "cleanup",
        ]
        distributed.cleanup.assert_called_once_with()

    @pytest.mark.pruned
    def test_root_log_messages_surround_directory_preparation(
        self,
        monkeypatch,
    ):
        events = []
        logger = DummyLogger()
        config = DummyInferenceConfig()

        logger.info = Mock(
            side_effect=lambda message: events.append(message),
        )
        config.prepare_directory = Mock(
            side_effect=lambda distributed: events.append("prepare_directory"),
        )

        patch_successful_main_dependencies(
            monkeypatch,
            logger=logger,
            config=config,
        )

        main("inference.yaml")

        assert events == [
            "Setting up directories ...",
            "prepare_directory",
            "Building objects:",
        ]


class TestMainCleanupOnFailure:
    @pytest.mark.pruned
    def test_cleanup_runs_when_prepare_config_fails(
        self,
        monkeypatch,
    ):
        distributed = DummyDistributed()

        monkeypatch.setattr(
            predict_mod.Distributed,
            "get_instance",
            Mock(
                return_value=distributed,
            ),
        )
        monkeypatch.setattr(
            predict_mod,
            "prepare_config",
            Mock(
                side_effect=RuntimeError("configuration failed"),
            ),
        )

        with pytest.raises(
            RuntimeError,
            match="configuration failed",
        ):
            main("inference.yaml")

        assert distributed.cleanup_called is True

    @pytest.mark.pruned
    def test_cleanup_runs_when_dacite_conversion_fails(
        self,
        monkeypatch,
    ):
        dependencies = patch_successful_main_dependencies(
            monkeypatch,
        )

        dependencies["from_dict"].side_effect = RuntimeError("conversion failed")

        with pytest.raises(
            RuntimeError,
            match="conversion failed",
        ):
            main("inference.yaml")

        assert dependencies["distributed"].cleanup_called is True

    @pytest.mark.pruned
    def test_cleanup_runs_when_seed_setup_fails(
        self,
        monkeypatch,
    ):
        config = DummyInferenceConfig()
        config.set_random_seed = Mock(
            side_effect=RuntimeError("seed failed"),
        )

        dependencies = patch_successful_main_dependencies(
            monkeypatch,
            config=config,
        )

        with pytest.raises(
            RuntimeError,
            match="seed failed",
        ):
            main("inference.yaml")

        assert dependencies["distributed"].cleanup_called is True
        dependencies["setup_logger"].assert_not_called()

    @pytest.mark.pruned
    def test_cleanup_runs_when_logger_setup_fails(
        self,
        monkeypatch,
    ):
        dependencies = patch_successful_main_dependencies(
            monkeypatch,
        )

        dependencies["setup_logger"].side_effect = RuntimeError("logger failed")

        with pytest.raises(
            RuntimeError,
            match="logger failed",
        ):
            main("inference.yaml")

        assert dependencies["distributed"].cleanup_called is True
        dependencies["build_writer"].assert_not_called()

    @pytest.mark.pruned
    def test_cleanup_runs_when_directory_preparation_fails(
        self,
        monkeypatch,
    ):
        config = DummyInferenceConfig()
        config.prepare_directory = Mock(
            side_effect=RuntimeError("directory failed"),
        )

        dependencies = patch_successful_main_dependencies(
            monkeypatch,
            config=config,
        )

        with pytest.raises(
            RuntimeError,
            match="directory failed",
        ):
            main("inference.yaml")

        assert dependencies["distributed"].cleanup_called is True
        dependencies["build_writer"].assert_not_called()

    @pytest.mark.pruned
    def test_cleanup_runs_when_writer_build_fails(
        self,
        monkeypatch,
    ):
        dependencies = patch_successful_main_dependencies(
            monkeypatch,
        )

        dependencies["build_writer"].side_effect = RuntimeError("writer build failed")

        with pytest.raises(
            RuntimeError,
            match="writer build failed",
        ):
            main("inference.yaml")

        assert dependencies["distributed"].cleanup_called is True

    @pytest.mark.pruned
    def test_cleanup_runs_when_writer_setup_fails(
        self,
        monkeypatch,
    ):
        writer = DummyWriter()
        writer.setup_distributed = Mock(
            side_effect=RuntimeError("writer setup failed"),
        )
        writer.predict = Mock()

        dependencies = patch_successful_main_dependencies(
            monkeypatch,
            writer=writer,
        )

        with pytest.raises(
            RuntimeError,
            match="writer setup failed",
        ):
            main("inference.yaml")

        assert dependencies["distributed"].cleanup_called is True
        writer.predict.assert_not_called()

    @pytest.mark.pruned
    def test_cleanup_runs_when_prediction_fails(
        self,
        monkeypatch,
    ):
        writer = DummyWriter()
        writer.predict = Mock(
            side_effect=RuntimeError("prediction failed"),
        )

        dependencies = patch_successful_main_dependencies(
            monkeypatch,
            writer=writer,
        )

        with pytest.raises(
            RuntimeError,
            match="prediction failed",
        ):
            main("inference.yaml")

        assert dependencies["distributed"].cleanup_called is True

    def test_cleanup_error_propagates_after_success(
        self,
        monkeypatch,
    ):
        distributed = DummyDistributed()
        distributed.cleanup = Mock(
            side_effect=RuntimeError("cleanup failed"),
        )

        patch_successful_main_dependencies(
            monkeypatch,
            distributed=distributed,
        )

        with pytest.raises(
            RuntimeError,
            match="cleanup failed",
        ):
            main("inference.yaml")

        distributed.cleanup.assert_called_once_with()

    @pytest.mark.pruned
    def test_distributed_initialization_failure_does_not_cleanup(
        self,
        monkeypatch,
    ):
        get_instance = Mock(
            side_effect=RuntimeError("distributed initialization failed"),
        )

        monkeypatch.setattr(
            predict_mod.Distributed,
            "get_instance",
            get_instance,
        )

        with pytest.raises(
            RuntimeError,
            match="distributed initialization failed",
        ):
            main("inference.yaml")

        get_instance.assert_called_once_with()
