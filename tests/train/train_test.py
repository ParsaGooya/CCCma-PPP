from pathlib import Path
import pytest
import cccma_ppp.train.train as train_mod
from cccma_ppp.train.train import main
from contextlib import contextmanager
from unittest.mock import Mock, call
from cccma_ppp.train.train import get_parser
import runpy
import sys


@pytest.mark.pruned
def test_main_root_happy_path(monkeypatch, tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("experiment_dir: test\n", encoding="utf-8")

    distributed = DummyDistributed(root=True)
    logger = DummyLogger()
    config_obj = DummyTrainConfig()
    trainer = DummyTrainer()

    monkeypatch.setattr(
        train_mod.Distributed,
        "get_instance",
        lambda: distributed,
    )

    monkeypatch.setattr(
        train_mod,
        "prepare_config",
        lambda path: {"experiment_dir": "test"},
    )

    monkeypatch.setattr(
        train_mod.dacite,
        "from_dict",
        lambda data_class, data, config: config_obj,
    )

    monkeypatch.setattr(
        train_mod,
        "setup_logger",
        lambda name, log_dir: logger,
    )

    monkeypatch.setattr(
        train_mod,
        "build_trainer",
        lambda config, distributed, logger: trainer,
    )

    main(str(yaml_path))

    assert config_obj.seed_called is True
    assert config_obj.prepare_called is True
    assert config_obj.prepare_args["distributed"] is distributed
    assert config_obj.prepare_args["yaml_config"] == str(yaml_path)

    assert trainer.setup_called is True
    assert trainer.setup_args["distributed"] is distributed
    assert trainer.setup_args["logger"] is logger
    assert trainer.setup_args["log_every_n_epochs"] == config_obj.log_every_n_epochs
    assert trainer.setup_args["save_checkpoint"] == config_obj.save_checkpoint

    assert trainer.train_called is True
    assert distributed.cleanup_called is True

    assert "Setting up directories ..." in logger.messages
    assert "Building objects:" in logger.messages


def test_main_non_root_skips_root_logging(monkeypatch, tmp_path):

    distributed = DummyDistributed(root=False)
    logger = DummyLogger()
    config_obj = DummyTrainConfig()
    trainer = DummyTrainer()

    monkeypatch.setattr(
        train_mod.Distributed,
        "get_instance",
        lambda: distributed,
    )

    monkeypatch.setattr(
        train_mod,
        "prepare_config",
        lambda path: {"experiment_dir": "test"},
    )

    monkeypatch.setattr(
        train_mod.dacite,
        "from_dict",
        lambda data_class, data, config: config_obj,
    )

    monkeypatch.setattr(
        train_mod,
        "setup_logger",
        lambda name, log_dir: logger,
    )

    monkeypatch.setattr(
        train_mod,
        "build_trainer",
        lambda config, distributed, logger: trainer,
    )
    yaml_path = tmp_path / "config.yaml"

    main(str(yaml_path))

    assert config_obj.seed_called is True
    assert config_obj.prepare_called is True
    assert trainer.setup_called is True
    assert trainer.train_called is True
    assert distributed.cleanup_called is True

    assert "Setting up directories ..." not in logger.messages
    assert "Building objects:" not in logger.messages


@pytest.mark.pruned
def test_main_passes_yaml_data_to_dacite(monkeypatch, tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("experiment_dir: test\n", encoding="utf-8")

    distributed = DummyDistributed(root=True)
    logger = DummyLogger()
    config_obj = DummyTrainConfig()
    trainer = DummyTrainer()

    captured = {}

    monkeypatch.setattr(
        train_mod.Distributed,
        "get_instance",
        lambda: distributed,
    )

    monkeypatch.setattr(
        train_mod,
        "prepare_config",
        lambda path: {"experiment_dir": "abc", "epochs": 5},
    )

    def fake_from_dict(data_class, data, config):
        captured["data_class"] = data_class
        captured["data"] = data
        captured["config"] = config
        return config_obj

    monkeypatch.setattr(train_mod.dacite, "from_dict", fake_from_dict)

    monkeypatch.setattr(
        train_mod,
        "setup_logger",
        lambda name, log_dir: logger,
    )

    monkeypatch.setattr(
        train_mod,
        "build_trainer",
        lambda config, distributed, logger: trainer,
    )

    main(str(yaml_path))

    assert captured["data_class"] is train_mod.TrainConfig
    assert captured["data"] == {"experiment_dir": "abc", "epochs": 5}
    assert captured["config"].strict is True


@pytest.mark.pruned
def test_main_setup_logger_called_with_training_name_and_log_dir(monkeypatch, tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("experiment_dir: test\n", encoding="utf-8")

    distributed = DummyDistributed(root=True)
    logger = DummyLogger()
    config_obj = DummyTrainConfig()
    trainer = DummyTrainer()

    captured = {}

    monkeypatch.setattr(
        train_mod.Distributed,
        "get_instance",
        lambda: distributed,
    )

    monkeypatch.setattr(
        train_mod,
        "prepare_config",
        lambda path: {"experiment_dir": "test"},
    )

    monkeypatch.setattr(
        train_mod.dacite,
        "from_dict",
        lambda data_class, data, config: config_obj,
    )

    def fake_setup_logger(name, log_dir):
        captured["name"] = name
        captured["log_dir"] = log_dir
        return logger

    monkeypatch.setattr(train_mod, "setup_logger", fake_setup_logger)

    monkeypatch.setattr(
        train_mod,
        "build_trainer",
        lambda config, distributed, logger: trainer,
    )

    main(str(yaml_path))

    assert captured["name"] == "training"
    assert captured["log_dir"] == config_obj.log_dir


def test_main_build_trainer_receives_config_distributed_logger(monkeypatch, tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("experiment_dir: test\n", encoding="utf-8")

    distributed = DummyDistributed(root=True)
    logger = DummyLogger()
    config_obj = DummyTrainConfig()
    trainer = DummyTrainer()

    captured = {}

    monkeypatch.setattr(
        train_mod.Distributed,
        "get_instance",
        lambda: distributed,
    )

    monkeypatch.setattr(
        train_mod,
        "prepare_config",
        lambda path: {"experiment_dir": "test"},
    )

    monkeypatch.setattr(
        train_mod.dacite,
        "from_dict",
        lambda data_class, data, config: config_obj,
    )

    monkeypatch.setattr(
        train_mod,
        "setup_logger",
        lambda name, log_dir: logger,
    )

    def fake_build_trainer(config, distributed, logger):
        captured["config"] = config
        captured["distributed"] = distributed
        captured["logger"] = logger
        return trainer

    monkeypatch.setattr(train_mod, "build_trainer", fake_build_trainer)

    main(str(yaml_path))

    assert captured["config"] is config_obj
    assert captured["distributed"] is distributed
    assert captured["logger"] is logger


def test_main_propagates_train_error_and_cleanup(monkeypatch, tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("experiment_dir: test\n", encoding="utf-8")

    distributed = DummyDistributed(root=True)
    logger = DummyLogger()
    config_obj = DummyTrainConfig()

    class FailingTrainer(DummyTrainer):
        def train(self):
            raise RuntimeError("training failed")

    trainer = FailingTrainer()

    monkeypatch.setattr(
        train_mod.Distributed,
        "get_instance",
        lambda: distributed,
    )

    monkeypatch.setattr(
        train_mod,
        "prepare_config",
        lambda path: {"experiment_dir": "test"},
    )

    monkeypatch.setattr(
        train_mod.dacite,
        "from_dict",
        lambda data_class, data, config: config_obj,
    )

    monkeypatch.setattr(
        train_mod,
        "setup_logger",
        lambda name, log_dir: logger,
    )

    monkeypatch.setattr(
        train_mod,
        "build_trainer",
        lambda config, distributed, logger: trainer,
    )

    with pytest.raises(RuntimeError, match="training failed"):
        main(str(yaml_path))

    assert distributed.cleanup_called is True


class DummyDistributed:
    def barrier(self):
        return None

    local_rank = 0
    world_size = 1

    def __init__(self, root=True):
        self._root = root
        self.rank = 0
        self.cleanup_called = False

    def is_root(self):
        return self._root

    def cleanup(self):
        self.cleanup_called = True


class DummyLogger:
    def __init__(self):
        self.messages = []

    def info(self, msg, *args, **kwargs):
        self.messages.append(msg)


class DummyTrainConfig:
    monitoring_dir = Path("/tmp")

    def __init__(self):
        self.seed_called = False
        self.prepare_called = False
        self.prepare_args = None

        self.log_dir = "logs"
        self.log_every_n_epochs = 3
        self.save_checkpoint = True

    def set_random_seed(self, rank=None):
        self.seed_called = True

    def prepare_directory(self, distributed, yaml_config):
        self.prepare_called = True
        self.prepare_args = {
            "distributed": distributed,
            "yaml_config": yaml_config,
        }


class DummyTrainer:
    def __init__(self):
        self.setup_called = False
        self.setup_args = None
        self.train_called = False

    def setup_distributed(
        self,
        distributed,
        logger,
        log_every_n_epochs=1,
        save_checkpoint=True,
    ):
        self.setup_called = True
        self.setup_args = {
            "distributed": distributed,
            "logger": logger,
            "log_every_n_epochs": log_every_n_epochs,
            "save_checkpoint": save_checkpoint,
        }

    def train(self):
        self.train_called = True


class RecordingMonitor:
    def __init__(self):
        self.spans = []
        self.active_spans = []
        self.checkpoints = []

    @contextmanager
    def span(self, name):
        self.spans.append(
            (
                "start",
                name,
            )
        )
        self.active_spans.append(name)

        try:
            yield
        finally:
            self.active_spans.pop()
            self.spans.append(
                (
                    "end",
                    name,
                )
            )

    def checkpoint(self, name):
        self.checkpoints.append(name)


def patch_successful_main_dependencies(
    monkeypatch,
    *,
    distributed=None,
    config=None,
    logger=None,
    trainer=None,
    monitor=None,
    config_data=None,
):
    distributed = distributed if distributed is not None else DummyDistributed()
    config = config if config is not None else DummyTrainConfig()
    logger = logger if logger is not None else DummyLogger()
    trainer = trainer if trainer is not None else DummyTrainer()
    monitor = monitor if monitor is not None else RecordingMonitor()
    config_data = (
        config_data
        if config_data is not None
        else {
            "experiment_dir": "test",
        }
    )

    monkeypatch.setattr(
        train_mod.Distributed,
        "get_instance",
        Mock(
            return_value=distributed,
        ),
    )
    monkeypatch.setattr(
        train_mod,
        "prepare_config",
        Mock(
            return_value=config_data,
        ),
    )
    monkeypatch.setattr(
        train_mod.dacite,
        "from_dict",
        Mock(
            return_value=config,
        ),
    )
    monkeypatch.setattr(
        train_mod,
        "setup_logger",
        Mock(
            return_value=logger,
        ),
    )
    monkeypatch.setattr(
        train_mod,
        "build_trainer",
        Mock(
            return_value=trainer,
        ),
    )

    @contextmanager
    def fake_distributed_monitoring(
        distributed_argument,
        output_dir,
    ):
        assert distributed_argument is distributed
        assert output_dir == config.monitoring_dir
        yield monitor

    monkeypatch.setattr(
        train_mod,
        "distributed_monitoring",
        fake_distributed_monitoring,
    )

    return {
        "distributed": distributed,
        "config": config,
        "logger": logger,
        "trainer": trainer,
        "monitor": monitor,
        "config_data": config_data,
    }


@pytest.mark.pruned
def test_get_parser_returns_argument_parser():
    parser = get_parser()

    assert parser.prog
    assert parser.description == "Train model from config file"


@pytest.mark.pruned
def test_get_parser_parses_config_path():
    parser = get_parser()

    arguments = parser.parse_args(
        [
            "training.yaml",
        ]
    )

    assert arguments.config == "training.yaml"


@pytest.mark.pruned
def test_get_parser_requires_config_argument():
    parser = get_parser()

    with pytest.raises(SystemExit) as error:
        parser.parse_args([])

    assert error.value.code == 2


@pytest.mark.pruned
def test_get_parser_rejects_unknown_arguments():
    parser = get_parser()

    with pytest.raises(SystemExit) as error:
        parser.parse_args(
            [
                "training.yaml",
                "--unknown",
            ]
        )

    assert error.value.code == 2


@pytest.mark.pruned
def test_main_gets_distributed_singleton(
    monkeypatch,
    tmp_path,
):
    dependencies = patch_successful_main_dependencies(monkeypatch)

    yaml_path = tmp_path / "config.yaml"

    main(str(yaml_path))

    train_mod.Distributed.get_instance.assert_called_once_with()

    assert dependencies["distributed"].cleanup_called is True


@pytest.mark.pruned
def test_main_passes_yaml_path_to_prepare_config(
    monkeypatch,
    tmp_path,
):
    patch_successful_main_dependencies(monkeypatch)

    yaml_path = tmp_path / "config.yaml"

    main(str(yaml_path))

    train_mod.prepare_config.assert_called_once_with(str(yaml_path))


@pytest.mark.pruned
def test_main_constructs_strict_dacite_config(
    monkeypatch,
    tmp_path,
):
    config_data = {
        "experiment_dir": "experiment",
        "epochs": 4,
    }

    patch_successful_main_dependencies(
        monkeypatch,
        config_data=config_data,
    )

    yaml_path = tmp_path / "config.yaml"

    main(str(yaml_path))

    call_arguments = train_mod.dacite.from_dict.call_args

    assert call_arguments.kwargs["data_class"] is train_mod.TrainConfig
    assert call_arguments.kwargs["data"] is config_data
    assert call_arguments.kwargs["config"].strict is True


@pytest.mark.pruned
def test_main_sets_seed_from_distributed_rank(
    monkeypatch,
    tmp_path,
):
    distributed = DummyDistributed()
    distributed.rank = 7

    config = DummyTrainConfig()
    config.set_random_seed = Mock()

    patch_successful_main_dependencies(
        monkeypatch,
        distributed=distributed,
        config=config,
    )

    main(str(tmp_path / "config.yaml"))

    config.set_random_seed.assert_called_once_with(7)


@pytest.mark.pruned
def test_main_passes_monitoring_directory(
    monkeypatch,
    tmp_path,
):
    config = DummyTrainConfig()
    config.monitoring_dir = tmp_path / "monitoring"

    calls = []
    monitor = RecordingMonitor()

    dependencies = patch_successful_main_dependencies(
        monkeypatch,
        config=config,
        monitor=monitor,
    )

    @contextmanager
    def fake_monitoring(
        distributed,
        output_dir,
    ):
        calls.append(
            (
                distributed,
                output_dir,
            )
        )
        yield monitor

    monkeypatch.setattr(
        train_mod,
        "distributed_monitoring",
        fake_monitoring,
    )

    main(str(tmp_path / "config.yaml"))

    assert calls == [
        (
            dependencies["distributed"],
            config.monitoring_dir,
        )
    ]


@pytest.mark.pruned
def test_main_records_monitoring_spans_in_order(
    monkeypatch,
    tmp_path,
):
    monitor = RecordingMonitor()

    patch_successful_main_dependencies(
        monkeypatch,
        monitor=monitor,
    )

    main(str(tmp_path / "config.yaml"))

    assert monitor.spans == [
        (
            "start",
            "build_trainer",
        ),
        (
            "end",
            "build_trainer",
        ),
        (
            "start",
            "setup_distributed",
        ),
        (
            "end",
            "setup_distributed",
        ),
        (
            "start",
            "train",
        ),
        (
            "end",
            "train",
        ),
    ]


@pytest.mark.pruned
def test_main_records_trainer_ready_checkpoint(
    monkeypatch,
    tmp_path,
):
    monitor = RecordingMonitor()

    patch_successful_main_dependencies(
        monkeypatch,
        monitor=monitor,
    )

    main(str(tmp_path / "config.yaml"))

    assert monitor.checkpoints == [
        "trainer_ready",
    ]


@pytest.mark.pruned
def test_build_trainer_runs_inside_monitoring_span(
    monkeypatch,
    tmp_path,
):
    monitor = RecordingMonitor()

    patch_successful_main_dependencies(
        monkeypatch,
        monitor=monitor,
    )

    def build_trainer(
        config,
        distributed,
        logger,
    ):
        assert monitor.active_spans == [
            "build_trainer",
        ]

        return DummyTrainer()

    monkeypatch.setattr(
        train_mod,
        "build_trainer",
        build_trainer,
    )

    main(str(tmp_path / "config.yaml"))


@pytest.mark.pruned
def test_setup_distributed_runs_inside_monitoring_span(
    monkeypatch,
    tmp_path,
):
    monitor = RecordingMonitor()
    trainer = DummyTrainer()

    def setup_distributed(
        distributed,
        logger,
        log_every_n_epochs,
        save_checkpoint,
    ):
        assert monitor.active_spans == [
            "setup_distributed",
        ]

        trainer.setup_called = True

    trainer.setup_distributed = setup_distributed

    patch_successful_main_dependencies(
        monkeypatch,
        trainer=trainer,
        monitor=monitor,
    )

    main(str(tmp_path / "config.yaml"))

    assert trainer.setup_called is True


@pytest.mark.pruned
def test_train_runs_inside_monitoring_span(
    monkeypatch,
    tmp_path,
):
    monitor = RecordingMonitor()
    trainer = DummyTrainer()

    def train():
        assert monitor.active_spans == [
            "train",
        ]

        trainer.train_called = True

    trainer.train = train

    patch_successful_main_dependencies(
        monkeypatch,
        trainer=trainer,
        monitor=monitor,
    )

    main(str(tmp_path / "config.yaml"))

    assert trainer.train_called is True


@pytest.mark.pruned
def test_checkpoint_occurs_between_setup_and_train(
    monkeypatch,
    tmp_path,
):
    events = []

    class EventMonitor:
        @contextmanager
        def span(self, name):
            events.append(f"start:{name}")

            try:
                yield
            finally:
                events.append(f"end:{name}")

        def checkpoint(self, name):
            events.append(f"checkpoint:{name}")

    patch_successful_main_dependencies(
        monkeypatch,
        monitor=EventMonitor(),
    )

    main(str(tmp_path / "config.yaml"))

    assert events.index("end:setup_distributed") < events.index(
        "checkpoint:trainer_ready"
    )

    assert events.index("checkpoint:trainer_ready") < events.index("start:train")


@pytest.mark.pruned
@pytest.mark.parametrize(
    (
        "failing_stage",
        "expected_message",
    ),
    [
        (
            "prepare_config",
            "prepare failed",
        ),
        (
            "from_dict",
            "dacite failed",
        ),
        (
            "set_random_seed",
            "seed failed",
        ),
        (
            "setup_logger",
            "logger failed",
        ),
        (
            "prepare_directory",
            "directory failed",
        ),
        (
            "monitoring",
            "monitoring failed",
        ),
        (
            "build_trainer",
            "build failed",
        ),
        (
            "setup_distributed",
            "setup failed",
        ),
        (
            "checkpoint",
            "checkpoint failed",
        ),
        (
            "train",
            "train failed",
        ),
    ],
)
def test_main_always_cleans_up_after_stage_failure(
    monkeypatch,
    tmp_path,
    failing_stage,
    expected_message,
):
    distributed = DummyDistributed()
    config = DummyTrainConfig()
    logger = DummyLogger()
    trainer = DummyTrainer()
    monitor = RecordingMonitor()

    dependencies = patch_successful_main_dependencies(
        monkeypatch,
        distributed=distributed,
        config=config,
        logger=logger,
        trainer=trainer,
        monitor=monitor,
    )

    if failing_stage == "prepare_config":
        train_mod.prepare_config.side_effect = RuntimeError(expected_message)

    elif failing_stage == "from_dict":
        train_mod.dacite.from_dict.side_effect = RuntimeError(expected_message)

    elif failing_stage == "set_random_seed":
        config.set_random_seed = Mock(side_effect=RuntimeError(expected_message))

    elif failing_stage == "setup_logger":
        train_mod.setup_logger.side_effect = RuntimeError(expected_message)

    elif failing_stage == "prepare_directory":
        config.prepare_directory = Mock(side_effect=RuntimeError(expected_message))

    elif failing_stage == "monitoring":

        @contextmanager
        def failed_monitoring(
            distributed_argument,
            output_dir,
        ):
            raise RuntimeError(expected_message)
            yield

        monkeypatch.setattr(
            train_mod,
            "distributed_monitoring",
            failed_monitoring,
        )

    elif failing_stage == "build_trainer":
        train_mod.build_trainer.side_effect = RuntimeError(expected_message)

    elif failing_stage == "setup_distributed":
        trainer.setup_distributed = Mock(side_effect=RuntimeError(expected_message))

    elif failing_stage == "checkpoint":
        monitor.checkpoint = Mock(side_effect=RuntimeError(expected_message))

    elif failing_stage == "train":
        trainer.train = Mock(side_effect=RuntimeError(expected_message))

    with pytest.raises(
        RuntimeError,
        match=expected_message,
    ):
        main(str(tmp_path / "config.yaml"))

    assert distributed.cleanup_called is True


@pytest.mark.pruned
def test_cleanup_runs_once_after_success(
    monkeypatch,
    tmp_path,
):
    distributed = DummyDistributed()
    distributed.cleanup = Mock()

    patch_successful_main_dependencies(
        monkeypatch,
        distributed=distributed,
    )

    main(str(tmp_path / "config.yaml"))

    distributed.cleanup.assert_called_once_with()


@pytest.mark.pruned
def test_cleanup_runs_once_after_failure(
    monkeypatch,
    tmp_path,
):
    distributed = DummyDistributed()
    distributed.cleanup = Mock()

    patch_successful_main_dependencies(
        monkeypatch,
        distributed=distributed,
    )

    train_mod.prepare_config.side_effect = RuntimeError("failure")

    with pytest.raises(
        RuntimeError,
        match="failure",
    ):
        main(str(tmp_path / "config.yaml"))

    distributed.cleanup.assert_called_once_with()


@pytest.mark.pruned
def test_cleanup_error_replaces_successful_result(
    monkeypatch,
    tmp_path,
):
    distributed = DummyDistributed()
    distributed.cleanup = Mock(side_effect=RuntimeError("cleanup failed"))

    patch_successful_main_dependencies(
        monkeypatch,
        distributed=distributed,
    )

    with pytest.raises(
        RuntimeError,
        match="cleanup failed",
    ):
        main(str(tmp_path / "config.yaml"))


@pytest.mark.pruned
def test_cleanup_error_replaces_original_error(
    monkeypatch,
    tmp_path,
):
    distributed = DummyDistributed()
    distributed.cleanup = Mock(side_effect=RuntimeError("cleanup failed"))

    patch_successful_main_dependencies(
        monkeypatch,
        distributed=distributed,
    )

    train_mod.prepare_config.side_effect = ValueError("configuration failed")

    with pytest.raises(
        RuntimeError,
        match="cleanup failed",
    ):
        main(str(tmp_path / "config.yaml"))


@pytest.mark.pruned
def test_non_root_still_prepares_directory(
    monkeypatch,
    tmp_path,
):
    distributed = DummyDistributed(root=False)
    config = DummyTrainConfig()
    config.prepare_directory = Mock()

    patch_successful_main_dependencies(
        monkeypatch,
        distributed=distributed,
        config=config,
    )

    yaml_path = tmp_path / "config.yaml"

    main(str(yaml_path))

    config.prepare_directory.assert_called_once_with(
        distributed,
        str(yaml_path),
    )


@pytest.mark.pruned
def test_root_logger_messages_are_emitted_once(
    monkeypatch,
    tmp_path,
):
    distributed = DummyDistributed(root=True)
    logger = DummyLogger()
    logger.info = Mock()

    patch_successful_main_dependencies(
        monkeypatch,
        distributed=distributed,
        logger=logger,
    )

    main(str(tmp_path / "config.yaml"))

    assert logger.info.call_args_list == [
        call("Setting up directories ..."),
        call("Building objects:"),
    ]


@pytest.mark.pruned
def test_non_root_logger_info_is_not_called(
    monkeypatch,
    tmp_path,
):
    distributed = DummyDistributed(root=False)
    logger = DummyLogger()
    logger.info = Mock()

    patch_successful_main_dependencies(
        monkeypatch,
        distributed=distributed,
        logger=logger,
    )

    main(str(tmp_path / "config.yaml"))

    logger.info.assert_not_called()


@pytest.mark.pruned
@pytest.mark.parametrize(
    (
        "log_every_n_epochs",
        "save_checkpoint",
    ),
    [
        (
            1,
            True,
        ),
        (
            1,
            False,
        ),
        (
            7,
            True,
        ),
        (
            7,
            False,
        ),
    ],
)
def test_main_forwards_setup_configuration(
    monkeypatch,
    tmp_path,
    log_every_n_epochs,
    save_checkpoint,
):
    config = DummyTrainConfig()
    config.log_every_n_epochs = log_every_n_epochs
    config.save_checkpoint = save_checkpoint

    trainer = DummyTrainer()
    trainer.setup_distributed = Mock()

    dependencies = patch_successful_main_dependencies(
        monkeypatch,
        config=config,
        trainer=trainer,
    )

    main(str(tmp_path / "config.yaml"))

    trainer.setup_distributed.assert_called_once_with(
        distributed=dependencies["distributed"],
        logger=dependencies["logger"],
        log_every_n_epochs=(log_every_n_epochs),
        save_checkpoint=(save_checkpoint),
    )


def test_train_module_main_guard(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cccma_ppp.train.train",
        ],
    )

    with pytest.raises(
        SystemExit,
    ) as error:
        runpy.run_module(
            "cccma_ppp.train.train",
            run_name="__main__",
        )

    assert error.value.code == 2


@pytest.mark.pruned
def test_get_parser_accepts_config_path():
    parser = train_mod.get_parser()

    arguments = parser.parse_args(
        [
            "train_config.yaml",
        ]
    )

    assert arguments.config == "train_config.yaml"


@pytest.mark.pruned
def test_get_parser_requires_config_path():
    parser = train_mod.get_parser()

    with pytest.raises(
        SystemExit,
    ) as error:
        parser.parse_args([])

    assert error.value.code == 2
