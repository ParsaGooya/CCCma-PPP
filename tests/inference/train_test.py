import pytest

from cccma_ppp.inference.train import (
    main,
)


def test_main_root_rank(monkeypatch):

    calls = []

    class DummyDistributed:
        def is_root(self):
            return True

        def cleanup(self):
            calls.append("cleanup")

    distributed = DummyDistributed()

    class DummyConfig:
        log_dir = "logs"
        log_every_n_epochs = 1
        save_checkpoint = True

        def set_random_seed(self):
            calls.append("seed")

        def prepare_directory(self, distributed, yaml):
            calls.append(("prepare_directory", yaml))

    class DummyLogger:
        def info(self, msg):
            calls.append(("info", msg))

    class DummyTrainer:
        def setup_distributed(
            self,
            distributed,
            logger,
            log_every_n_epochs,
            save_checkpoint,
        ):
            calls.append("setup_distributed")

        def train(self):
            calls.append("train")

    monkeypatch.setattr(
        "cccma_ppp.inference.train.Distributed.get_instance",
        lambda: distributed,
    )

    monkeypatch.setattr(
        "cccma_ppp.inference.train.prepare_config",
        lambda yaml: {"some": "config"},
    )

    monkeypatch.setattr(
        "cccma_ppp.inference.train.dacite.from_dict",
        lambda **kwargs: DummyConfig(),
    )

    monkeypatch.setattr(
        "cccma_ppp.inference.train.setup_logger",
        lambda **kwargs: DummyLogger(),
    )

    monkeypatch.setattr(
        "cccma_ppp.inference.train.build_trainer",
        lambda config, distributed, logger: DummyTrainer(),
    )

    main("config.yaml")

    assert "seed" in calls
    assert "setup_distributed" in calls
    assert "train" in calls
    assert "cleanup" in calls


def test_main_non_root_rank(monkeypatch):

    calls = []

    class DummyDistributed:
        def is_root(self):
            return False

        def cleanup(self):
            calls.append("cleanup")

    distributed = DummyDistributed()

    class DummyConfig:
        log_dir = "logs"
        log_every_n_epochs = 1
        save_checkpoint = False

        def set_random_seed(self):
            calls.append("seed")

        def prepare_directory(self, distributed, yaml):
            calls.append("prepare_directory")

    class DummyLogger:
        def info(self, msg):
            calls.append(msg)

    class DummyTrainer:
        def setup_distributed(
            self,
            distributed,
            logger,
            log_every_n_epochs,
            save_checkpoint,
        ):
            calls.append("setup_distributed")

        def train(self):
            calls.append("train")

    monkeypatch.setattr(
        "cccma_ppp.inference.train.Distributed.get_instance",
        lambda: distributed,
    )

    monkeypatch.setattr(
        "cccma_ppp.inference.train.prepare_config",
        lambda yaml: {},
    )

    monkeypatch.setattr(
        "cccma_ppp.inference.train.dacite.from_dict",
        lambda **kwargs: DummyConfig(),
    )

    monkeypatch.setattr(
        "cccma_ppp.inference.train.setup_logger",
        lambda **kwargs: DummyLogger(),
    )

    monkeypatch.setattr(
        "cccma_ppp.inference.train.build_trainer",
        lambda config, distributed, logger: DummyTrainer(),
    )

    main("config.yaml")

    assert "train" in calls
    assert "cleanup" in calls

    assert not any(isinstance(x, tuple) and x[0] == "info" for x in calls)


def test_main_trainer_failure(monkeypatch):

    class DummyDistributed:
        def is_root(self):
            return True

        def cleanup(self):
            pass

    class DummyConfig:
        log_dir = "logs"
        log_every_n_epochs = 1
        save_checkpoint = True

        def set_random_seed(self):
            pass

        def prepare_directory(self, distributed, yaml):
            pass

    class DummyLogger:
        def info(self, msg):
            pass

    class DummyTrainer:
        def setup_distributed(
            self,
            distributed,
            logger,
            log_every_n_epochs,
            save_checkpoint,
        ):
            pass

        def train(self):
            raise RuntimeError("training failed")

    monkeypatch.setattr(
        "cccma_ppp.inference.train.Distributed.get_instance",
        lambda: DummyDistributed(),
    )

    monkeypatch.setattr(
        "cccma_ppp.inference.train.prepare_config",
        lambda yaml: {},
    )

    monkeypatch.setattr(
        "cccma_ppp.inference.train.dacite.from_dict",
        lambda **kwargs: DummyConfig(),
    )

    monkeypatch.setattr(
        "cccma_ppp.inference.train.setup_logger",
        lambda **kwargs: DummyLogger(),
    )

    monkeypatch.setattr(
        "cccma_ppp.inference.train.build_trainer",
        lambda config, distributed, logger: DummyTrainer(),
    )

    with pytest.raises(RuntimeError):
        main("config.yaml")
