import pytest
import cccma_ppp.train.train as train_mod
from cccma_ppp.train.train import main


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
