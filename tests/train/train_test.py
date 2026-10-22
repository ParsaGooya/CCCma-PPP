from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock

import pytest

import cccma_ppp.train.train as train_mod
from cccma_ppp.train.train import get_parser, main


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


@pytest.mark.pruned
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


@pytest.mark.pruned
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
def test_main_passes_yaml_path_to_prepare_config(
    monkeypatch,
    tmp_path,
):
    yaml_path = tmp_path / "training.yaml"
    dependencies = patch_successful_main_dependencies(
        monkeypatch,
    )

    main(str(yaml_path))

    train_mod.prepare_config.assert_called_once_with(
        str(yaml_path),
    )
    assert dependencies["distributed"].cleanup_called is True


@pytest.mark.pruned
def test_main_sets_random_seed_with_distributed_rank(
    monkeypatch,
    tmp_path,
):
    distributed = DummyDistributed()
    distributed.rank = 4
    config = DummyTrainConfig()
    config.set_random_seed = Mock()

    patch_successful_main_dependencies(
        monkeypatch,
        distributed=distributed,
        config=config,
    )

    main(str(tmp_path / "training.yaml"))

    config.set_random_seed.assert_called_once_with(4)


@pytest.mark.pruned
def test_main_uses_strict_dacite_configuration(
    monkeypatch,
    tmp_path,
):
    dependencies = patch_successful_main_dependencies(
        monkeypatch,
        config_data={
            "epochs": 10,
        },
    )

    main(str(tmp_path / "training.yaml"))

    train_mod.dacite.from_dict.assert_called_once()

    call = train_mod.dacite.from_dict.call_args

    assert call.kwargs["data_class"] is train_mod.TrainConfig
    assert call.kwargs["data"] is dependencies["config_data"]
    assert isinstance(
        call.kwargs["config"],
        train_mod.dacite.Config,
    )
    assert call.kwargs["config"].strict is True


@pytest.mark.pruned
def test_main_sets_up_logger_after_random_seed(
    monkeypatch,
    tmp_path,
):
    events = []
    config = DummyTrainConfig()
    logger = DummyLogger()

    config.set_random_seed = Mock(
        side_effect=lambda rank: events.append("seed"),
    )

    dependencies = patch_successful_main_dependencies(
        monkeypatch,
        config=config,
        logger=logger,
    )

    train_mod.setup_logger.side_effect = lambda **kwargs: (
        events.append("logger"),
        logger,
    )[1]

    main(str(tmp_path / "training.yaml"))

    assert events == [
        "seed",
        "logger",
    ]
    assert dependencies["distributed"].cleanup_called is True


@pytest.mark.pruned
def test_main_prepares_directory_before_building_trainer(
    monkeypatch,
    tmp_path,
):
    events = []
    config = DummyTrainConfig()
    trainer = DummyTrainer()

    config.prepare_directory = Mock(
        side_effect=lambda *args: events.append("prepare_directory"),
    )

    dependencies = patch_successful_main_dependencies(
        monkeypatch,
        config=config,
        trainer=trainer,
    )

    train_mod.build_trainer.side_effect = lambda *args: (
        events.append("build_trainer"),
        trainer,
    )[1]

    main(str(tmp_path / "training.yaml"))

    assert events == [
        "prepare_directory",
        "build_trainer",
    ]
    assert dependencies["distributed"].cleanup_called is True


@pytest.mark.pruned
def test_main_passes_exact_directory_preparation_arguments(
    monkeypatch,
    tmp_path,
):
    yaml_path = tmp_path / "training.yaml"
    config = DummyTrainConfig()
    config.prepare_directory = Mock()

    dependencies = patch_successful_main_dependencies(
        monkeypatch,
        config=config,
    )

    main(str(yaml_path))

    config.prepare_directory.assert_called_once_with(
        dependencies["distributed"],
        str(yaml_path),
    )


@pytest.mark.pruned
def test_main_passes_exact_logger_arguments(
    monkeypatch,
    tmp_path,
):
    config = DummyTrainConfig()

    patch_successful_main_dependencies(
        monkeypatch,
        config=config,
    )

    main(str(tmp_path / "training.yaml"))

    train_mod.setup_logger.assert_called_once_with(
        name="training",
        log_dir=config.log_dir,
    )


@pytest.mark.pruned
def test_main_passes_exact_build_trainer_arguments(
    monkeypatch,
    tmp_path,
):
    dependencies = patch_successful_main_dependencies(
        monkeypatch,
    )

    main(str(tmp_path / "training.yaml"))

    train_mod.build_trainer.assert_called_once_with(
        dependencies["config"],
        dependencies["distributed"],
        dependencies["logger"],
    )


@pytest.mark.pruned
def test_main_passes_exact_trainer_setup_arguments(
    monkeypatch,
    tmp_path,
):
    trainer = DummyTrainer()
    trainer.setup_distributed = Mock()

    dependencies = patch_successful_main_dependencies(
        monkeypatch,
        trainer=trainer,
    )

    main(str(tmp_path / "training.yaml"))

    trainer.setup_distributed.assert_called_once_with(
        distributed=dependencies["distributed"],
        logger=dependencies["logger"],
        log_every_n_epochs=(dependencies["config"].log_every_n_epochs),
        save_checkpoint=(dependencies["config"].save_checkpoint),
    )


@pytest.mark.pruned
def test_main_calls_train_once(
    monkeypatch,
    tmp_path,
):
    trainer = DummyTrainer()
    trainer.train = Mock()

    patch_successful_main_dependencies(
        monkeypatch,
        trainer=trainer,
    )

    main(str(tmp_path / "training.yaml"))

    trainer.train.assert_called_once_with()


@pytest.mark.pruned
def test_main_uses_expected_monitoring_directory(
    monkeypatch,
    tmp_path,
):
    config = DummyTrainConfig()
    config.monitoring_dir = tmp_path / "monitoring"
    captured = {}

    dependencies = patch_successful_main_dependencies(
        monkeypatch,
        config=config,
    )

    @contextmanager
    def monitoring(
        distributed,
        output_dir,
    ):
        captured["distributed"] = distributed
        captured["output_dir"] = output_dir
        yield dependencies["monitor"]

    monkeypatch.setattr(
        train_mod,
        "distributed_monitoring",
        monitoring,
    )

    main(str(tmp_path / "training.yaml"))

    assert captured["distributed"] is dependencies["distributed"]
    assert captured["output_dir"] == config.monitoring_dir


@pytest.mark.pruned
def test_main_records_expected_monitor_spans(
    monkeypatch,
    tmp_path,
):
    monitor = RecordingMonitor()

    patch_successful_main_dependencies(
        monkeypatch,
        monitor=monitor,
    )

    main(str(tmp_path / "training.yaml"))

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

    main(str(tmp_path / "training.yaml"))

    assert monitor.checkpoints == [
        "trainer_ready",
    ]


@pytest.mark.pruned
def test_main_checkpoint_occurs_after_setup_and_before_training(
    monkeypatch,
    tmp_path,
):
    events = []
    trainer = DummyTrainer()

    trainer.setup_distributed = Mock(
        side_effect=lambda **kwargs: events.append("setup"),
    )
    trainer.train = Mock(
        side_effect=lambda: events.append("train"),
    )

    class Monitor:
        @contextmanager
        def span(self, name):
            yield

        def checkpoint(self, name):
            events.append(name)

    patch_successful_main_dependencies(
        monkeypatch,
        trainer=trainer,
        monitor=Monitor(),
    )

    main(str(tmp_path / "training.yaml"))

    assert events == [
        "setup",
        "trainer_ready",
        "train",
    ]


@pytest.mark.pruned
def test_main_operations_run_inside_expected_spans(
    monkeypatch,
    tmp_path,
):
    monitor = RecordingMonitor()
    trainer = DummyTrainer()
    captured = {}

    def build_trainer(
        config,
        distributed,
        logger,
    ):
        captured["build"] = list(monitor.active_spans)
        return trainer

    def setup_distributed(**kwargs):
        captured["setup"] = list(monitor.active_spans)

    def train():
        captured["train"] = list(monitor.active_spans)

    trainer.setup_distributed = setup_distributed
    trainer.train = train

    patch_successful_main_dependencies(
        monkeypatch,
        trainer=trainer,
        monitor=monitor,
    )
    monkeypatch.setattr(
        train_mod,
        "build_trainer",
        build_trainer,
    )

    main(str(tmp_path / "training.yaml"))

    assert captured == {
        "build": [
            "build_trainer",
        ],
        "setup": [
            "setup_distributed",
        ],
        "train": [
            "train",
        ],
    }


@pytest.mark.pruned
def test_main_root_logs_before_directory_preparation(
    monkeypatch,
    tmp_path,
):
    events = []
    logger = DummyLogger()
    config = DummyTrainConfig()

    logger.info = Mock(
        side_effect=lambda message: events.append(message),
    )
    config.prepare_directory = Mock(
        side_effect=lambda *args: events.append("prepare_directory"),
    )

    patch_successful_main_dependencies(
        monkeypatch,
        logger=logger,
        config=config,
    )

    main(str(tmp_path / "training.yaml"))

    assert events[:2] == [
        "Setting up directories ...",
        "prepare_directory",
    ]


@pytest.mark.pruned
def test_main_root_logs_build_message_after_directory_preparation(
    monkeypatch,
    tmp_path,
):
    events = []
    logger = DummyLogger()
    config = DummyTrainConfig()

    logger.info = Mock(
        side_effect=lambda message: events.append(message),
    )
    config.prepare_directory = Mock(
        side_effect=lambda *args: events.append("prepare_directory"),
    )

    patch_successful_main_dependencies(
        monkeypatch,
        logger=logger,
        config=config,
    )

    main(str(tmp_path / "training.yaml"))

    assert events == [
        "Setting up directories ...",
        "prepare_directory",
        "Building objects:",
    ]


@pytest.mark.pruned
def test_main_non_root_never_calls_logger_info(
    monkeypatch,
    tmp_path,
):
    distributed = DummyDistributed(
        root=False,
    )
    logger = DummyLogger()
    logger.info = Mock()

    patch_successful_main_dependencies(
        monkeypatch,
        distributed=distributed,
        logger=logger,
    )

    main(str(tmp_path / "training.yaml"))

    logger.info.assert_not_called()
    assert distributed.cleanup_called is True


@pytest.mark.pruned
@pytest.mark.parametrize(
    (
        "attribute",
        "value",
    ),
    [
        (
            "log_every_n_epochs",
            7,
        ),
        (
            "save_checkpoint",
            False,
        ),
    ],
)
def test_main_forwards_trainer_setup_configuration(
    monkeypatch,
    tmp_path,
    attribute,
    value,
):
    config = DummyTrainConfig()
    setattr(
        config,
        attribute,
        value,
    )

    trainer = DummyTrainer()
    trainer.setup_distributed = Mock()

    patch_successful_main_dependencies(
        monkeypatch,
        config=config,
        trainer=trainer,
    )

    main(str(tmp_path / "training.yaml"))

    assert trainer.setup_distributed.call_args.kwargs[attribute] == value


@pytest.mark.pruned
def test_main_cleanup_occurs_after_successful_training(
    monkeypatch,
    tmp_path,
):
    events = []
    distributed = DummyDistributed()
    trainer = DummyTrainer()

    trainer.train = Mock(
        side_effect=lambda: events.append("train"),
    )
    distributed.cleanup = Mock(
        side_effect=lambda: events.append("cleanup"),
    )

    patch_successful_main_dependencies(
        monkeypatch,
        distributed=distributed,
        trainer=trainer,
    )

    main(str(tmp_path / "training.yaml"))

    assert events == [
        "train",
        "cleanup",
    ]
    distributed.cleanup.assert_called_once_with()


@pytest.mark.pruned
def test_main_cleanup_runs_when_prepare_config_fails(
    monkeypatch,
):
    distributed = DummyDistributed()

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
            side_effect=RuntimeError("configuration failed"),
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="configuration failed",
    ):
        main("training.yaml")

    assert distributed.cleanup_called is True


@pytest.mark.pruned
def test_main_cleanup_runs_when_dacite_conversion_fails(
    monkeypatch,
):
    distributed = DummyDistributed()

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
            return_value={
                "invalid": True,
            },
        ),
    )
    monkeypatch.setattr(
        train_mod.dacite,
        "from_dict",
        Mock(
            side_effect=RuntimeError("conversion failed"),
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="conversion failed",
    ):
        main("training.yaml")

    assert distributed.cleanup_called is True


@pytest.mark.pruned
def test_main_cleanup_runs_when_seed_setup_fails(
    monkeypatch,
):
    distributed = DummyDistributed()
    config = DummyTrainConfig()
    config.set_random_seed = Mock(
        side_effect=RuntimeError("seed failed"),
    )

    patch_successful_main_dependencies(
        monkeypatch,
        distributed=distributed,
        config=config,
    )

    with pytest.raises(
        RuntimeError,
        match="seed failed",
    ):
        main("training.yaml")

    assert distributed.cleanup_called is True
    train_mod.setup_logger.assert_not_called()


@pytest.mark.pruned
def test_main_cleanup_runs_when_logger_setup_fails(
    monkeypatch,
):
    distributed = DummyDistributed()

    patch_successful_main_dependencies(
        monkeypatch,
        distributed=distributed,
    )
    train_mod.setup_logger.side_effect = RuntimeError("logger failed")

    with pytest.raises(
        RuntimeError,
        match="logger failed",
    ):
        main("training.yaml")

    assert distributed.cleanup_called is True
    train_mod.build_trainer.assert_not_called()


@pytest.mark.pruned
def test_main_cleanup_runs_when_directory_preparation_fails(
    monkeypatch,
):
    distributed = DummyDistributed()
    config = DummyTrainConfig()
    config.prepare_directory = Mock(
        side_effect=RuntimeError("directory failed"),
    )

    patch_successful_main_dependencies(
        monkeypatch,
        distributed=distributed,
        config=config,
    )

    with pytest.raises(
        RuntimeError,
        match="directory failed",
    ):
        main("training.yaml")

    assert distributed.cleanup_called is True
    train_mod.build_trainer.assert_not_called()


@pytest.mark.pruned
def test_main_cleanup_runs_when_monitoring_context_entry_fails(
    monkeypatch,
):
    distributed = DummyDistributed()

    patch_successful_main_dependencies(
        monkeypatch,
        distributed=distributed,
    )

    @contextmanager
    def failing_monitoring(
        distributed,
        output_dir,
    ):
        raise RuntimeError("monitoring failed")
        yield

    monkeypatch.setattr(
        train_mod,
        "distributed_monitoring",
        failing_monitoring,
    )

    with pytest.raises(
        RuntimeError,
        match="monitoring failed",
    ):
        main("training.yaml")

    assert distributed.cleanup_called is True
    train_mod.build_trainer.assert_not_called()


@pytest.mark.pruned
def test_main_cleanup_runs_when_build_trainer_fails(
    monkeypatch,
):
    distributed = DummyDistributed()
    monitor = RecordingMonitor()

    patch_successful_main_dependencies(
        monkeypatch,
        distributed=distributed,
        monitor=monitor,
    )
    train_mod.build_trainer.side_effect = RuntimeError("build failed")

    with pytest.raises(
        RuntimeError,
        match="build failed",
    ):
        main("training.yaml")

    assert distributed.cleanup_called is True
    assert monitor.spans == [
        (
            "start",
            "build_trainer",
        ),
        (
            "end",
            "build_trainer",
        ),
    ]
    assert monitor.checkpoints == []


@pytest.mark.pruned
def test_main_cleanup_runs_when_trainer_setup_fails(
    monkeypatch,
):
    distributed = DummyDistributed()
    trainer = DummyTrainer()
    trainer.setup_distributed = Mock(
        side_effect=RuntimeError("setup failed"),
    )
    trainer.train = Mock()
    monitor = RecordingMonitor()

    patch_successful_main_dependencies(
        monkeypatch,
        distributed=distributed,
        trainer=trainer,
        monitor=monitor,
    )

    with pytest.raises(
        RuntimeError,
        match="setup failed",
    ):
        main("training.yaml")

    assert distributed.cleanup_called is True
    trainer.train.assert_not_called()
    assert monitor.checkpoints == []


@pytest.mark.pruned
def test_main_cleanup_runs_when_checkpoint_fails(
    monkeypatch,
):
    distributed = DummyDistributed()
    trainer = DummyTrainer()
    trainer.train = Mock()

    class FailingMonitor(RecordingMonitor):
        def checkpoint(self, name):
            super().checkpoint(name)
            raise RuntimeError("checkpoint failed")

    monitor = FailingMonitor()

    patch_successful_main_dependencies(
        monkeypatch,
        distributed=distributed,
        trainer=trainer,
        monitor=monitor,
    )

    with pytest.raises(
        RuntimeError,
        match="checkpoint failed",
    ):
        main("training.yaml")

    assert distributed.cleanup_called is True
    trainer.train.assert_not_called()
    assert monitor.checkpoints == [
        "trainer_ready",
    ]


@pytest.mark.pruned
def test_main_training_error_closes_train_span(
    monkeypatch,
):
    distributed = DummyDistributed()
    trainer = DummyTrainer()
    trainer.train = Mock(
        side_effect=RuntimeError("training failed"),
    )
    monitor = RecordingMonitor()

    patch_successful_main_dependencies(
        monkeypatch,
        distributed=distributed,
        trainer=trainer,
        monitor=monitor,
    )

    with pytest.raises(
        RuntimeError,
        match="training failed",
    ):
        main("training.yaml")

    assert distributed.cleanup_called is True
    assert monitor.spans[-2:] == [
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
def test_main_cleanup_error_propagates_after_success(
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
        main("training.yaml")

    distributed.cleanup.assert_called_once_with()


@pytest.mark.pruned
def test_get_instance_failure_does_not_attempt_cleanup(
    monkeypatch,
):
    get_instance = Mock(
        side_effect=RuntimeError("distributed initialization failed"),
    )
    monkeypatch.setattr(
        train_mod.Distributed,
        "get_instance",
        get_instance,
    )

    with pytest.raises(
        RuntimeError,
        match="distributed initialization failed",
    ):
        main("training.yaml")

    get_instance.assert_called_once_with()


@pytest.mark.pruned
def test_get_parser_config_argument_is_string():
    parser = get_parser()

    arguments = parser.parse_args(
        [
            str(Path("training.yaml")),
        ]
    )

    assert arguments.config == "training.yaml"
    assert isinstance(
        arguments.config,
        str,
    )


@pytest.mark.pruned
def test_get_parser_accepts_relative_config_path():
    parser = get_parser()

    arguments = parser.parse_args(
        [
            "configs/training.yaml",
        ]
    )

    assert arguments.config == "configs/training.yaml"


@pytest.mark.pruned
def test_get_parser_accepts_absolute_config_path(
    tmp_path,
):
    parser = get_parser()
    config_path = tmp_path / "training.yaml"

    arguments = parser.parse_args(
        [
            str(config_path),
        ]
    )

    assert arguments.config == str(config_path)


@pytest.mark.pruned
def test_get_parser_help_exits_successfully(
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

    assert "Train model from config file" in output
    assert "Path to the YAML config file" in output
    assert "config" in output


def test_module_main_guard_executes_cli(
    monkeypatch,
    tmp_path,
):
    import argparse
    import runpy

    import cccma_ppp.generic.logger as logger_module
    import cccma_ppp.generic.monitoring as monitoring_module
    import cccma_ppp.train.train_configs as train_configs_module

    yaml_path = tmp_path / "training.yaml"

    distributed = DummyDistributed()
    config = DummyTrainConfig()
    logger = DummyLogger()
    trainer = DummyTrainer()
    monitor = RecordingMonitor()

    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        Mock(
            return_value=argparse.Namespace(
                config=str(yaml_path),
            )
        ),
    )
    monkeypatch.setattr(
        train_mod.Distributed,
        "get_instance",
        Mock(
            return_value=distributed,
        ),
    )
    monkeypatch.setattr(
        train_configs_module,
        "prepare_config",
        Mock(
            return_value={
                "experiment_dir": "test",
            }
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
        logger_module,
        "setup_logger",
        Mock(
            return_value=logger,
        ),
    )
    monkeypatch.setattr(
        train_configs_module,
        "build_trainer",
        Mock(
            return_value=trainer,
        ),
    )

    @contextmanager
    def monitoring(
        distributed_argument,
        output_dir,
    ):
        assert distributed_argument is distributed
        assert output_dir == config.monitoring_dir
        yield monitor

    monkeypatch.setattr(
        monitoring_module,
        "distributed_monitoring",
        monitoring,
    )

    runpy.run_path(
        train_mod.__file__,
        run_name="__main__",
    )

    assert config.prepare_args["yaml_config"] == str(yaml_path)
    assert trainer.setup_called is True
    assert trainer.train_called is True
    assert distributed.cleanup_called is True
