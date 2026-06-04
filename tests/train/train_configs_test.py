import pytest
import warnings
import os
from pathlib import Path

from src.cccma_ppp.train.train_configs import TrainConfig


# ============================================================
# HELPERS
# ============================================================


class DummyPipeline:
    def __init__(self, pipeline=None):
        self.pipeline = pipeline or []


class DummyDatasetConfig:
    def __init__(self, observation=None, condition_type=None, pipeline=None):
        self.condition_type = condition_type

        obs_pipeline = DummyPipeline(pipeline)
        model_pipeline = DummyPipeline(pipeline)

        if observation is not None:
            self.observation = type(
                "Obs", (), {"preprocessing_pipeline": obs_pipeline}
            )()
        else:
            self.observation = None

        self.model = type("Model", (), {"preprocessing_pipeline": model_pipeline})()


class DummyTrainLoader:
    def __init__(self, dataset_config):
        self.dataset_config = dataset_config


class DummyModule:
    def __init__(self, type_="deterministic", generator=False, num_out=1):
        self.type = type_

        class Model:
            GENERATOR = generator
            NUM_OUTPUT_DIMS = num_out

        self._module_config = type("Inner", (), {"model": Model()})


class DummyTrainer:
    def __init__(self, beta_finder=None):
        self.beta_finder = beta_finder


class DummyLoss:
    def __init__(self, loss_types):
        self.loss_pipeline = type("LP", (), {"loss_types": loss_types})


# helper to inject Oceannanremove
def ocean_pipeline():
    from src.cccma_ppp.preprocessing.utils_preprocessing import Oceannanremove

    return [(None, Oceannanremove())]


# ============================================================
# CORE BRANCHES
# ============================================================


def test_epochs_assertion():
    with pytest.raises(AssertionError):
        TrainConfig(
            "x",
            -1,
            DummyTrainLoader(DummyDatasetConfig()),
            DummyModule(),
            DummyLoss(["mse"]),
            DummyTrainer(),
        )


# ---- observation + deterministic/default branch ----


def test_no_observation_deterministic():
    with pytest.raises(ValueError):
        TrainConfig(
            "x",
            1,
            DummyTrainLoader(DummyDatasetConfig()),
            DummyModule("deterministic"),
            DummyLoss(["mse"]),
            DummyTrainer(),
        )


def test_no_observation_default():
    with pytest.raises(ValueError):
        TrainConfig(
            "x",
            1,
            DummyTrainLoader(DummyDatasetConfig()),
            DummyModule("default"),
            DummyLoss(["mse"]),
            DummyTrainer(),
        )


def test_no_observation_other_type():
    TrainConfig(
        "x",
        1,
        DummyTrainLoader(DummyDatasetConfig()),
        DummyModule("other"),
        DummyLoss(["mse"]),
        DummyTrainer(),
    )


# ---- cvae branch ----


def test_cvae_beta_required():
    with pytest.raises(ValueError):
        TrainConfig(
            "x",
            1,
            DummyTrainLoader(DummyDatasetConfig(condition_type="cond")),
            DummyModule("cvae", num_out=2),
            DummyLoss(["mse"]),
            DummyTrainer(None),
        )


def test_cvae_condition_required():
    with pytest.raises(AssertionError):
        TrainConfig(
            "x",
            1,
            DummyTrainLoader(DummyDatasetConfig(condition_type=None)),
            DummyModule("cvae", num_out=2),
            DummyLoss(["mse"]),
            DummyTrainer(True),
        )


def test_cvae_valid():
    TrainConfig(
        "x",
        1,
        DummyTrainLoader(DummyDatasetConfig(condition_type="cond")),
        DummyModule("cvae", num_out=2),
        DummyLoss(["mse"]),
        DummyTrainer(True),
    )


# ---- generator branch ----


def test_generator_requires_crps():
    with pytest.raises(RuntimeError):
        TrainConfig(
            "x",
            1,
            DummyTrainLoader(DummyDatasetConfig(observation="obs")),
            DummyModule(generator=True, num_out=2),
            DummyLoss(["mse"]),
            DummyTrainer(),
        )


def test_generator_valid():
    TrainConfig(
        "x",
        1,
        DummyTrainLoader(DummyDatasetConfig(observation="obs")),
        DummyModule(generator=True, num_out=2),
        DummyLoss(["crps"]),
        DummyTrainer(),
    )


def test_non_generator_skip():
    TrainConfig(
        "x",
        1,
        DummyTrainLoader(DummyDatasetConfig(observation="obs")),
        DummyModule(generator=False, num_out=2),
        DummyLoss(["mse"]),
        DummyTrainer(),
    )


# ---- warning branch ----


def test_deterministic_warning():
    with warnings.catch_warnings(record=True) as w:
        TrainConfig(
            "x",
            1,
            DummyTrainLoader(
                DummyDatasetConfig(observation="obs", pipeline=ocean_pipeline())
            ),
            DummyModule("deterministic"),
            DummyLoss(["mse"]),
            DummyTrainer(None),
        )
        assert len(w) > 0


def test_deterministic_no_warning():
    with warnings.catch_warnings(record=True) as w:
        TrainConfig(
            "x",
            1,
            DummyTrainLoader(
                DummyDatasetConfig(observation="obs", pipeline=ocean_pipeline())
            ),
            DummyModule("deterministic"),
            DummyLoss(["mse"]),
            DummyTrainer(True),
        )
        assert len(w) == 0


# ---- oceannan branch ----


def test_ocean_required_fail_observation():
    with pytest.raises(RuntimeError):
        TrainConfig(
            "x",
            1,
            DummyTrainLoader(DummyDatasetConfig(observation="obs")),
            DummyModule(num_out=1),
            DummyLoss(["mse"]),
            DummyTrainer(),
        )


def test_ocean_required_pass_observation():
    TrainConfig(
        "x",
        1,
        DummyTrainLoader(
            DummyDatasetConfig(observation="obs", pipeline=ocean_pipeline())
        ),
        DummyModule(num_out=1),
        DummyLoss(["mse"]),
        DummyTrainer(),
    )


def test_ocean_required_model_pipeline():
    TrainConfig(
        "x",
        1,
        DummyTrainLoader(
            DummyDatasetConfig(condition_type="cond", pipeline=ocean_pipeline())
        ),
        DummyModule("cvae", num_out=1),
        DummyLoss(["mse"]),
        DummyTrainer(True),
    )


def test_ocean_required_model_fail():
    with pytest.raises(RuntimeError):
        TrainConfig(
            "x",
            1,
            DummyTrainLoader(DummyDatasetConfig(condition_type="cond")),
            DummyModule("cvae", num_out=1),
            DummyLoss(["mse"]),
            DummyTrainer(True),
        )


# ============================================================
# PREPARE DIRECTORY (ALL BRANCHES)
# ============================================================


class Root:
    def is_root(self):
        return True

    def barrier(self):
        pass


class NonRoot:
    def is_root(self):
        return False

    def barrier(self):
        pass


def test_prepare_root(tmp_path):
    cfg = TrainConfig(
        str(tmp_path),
        1,
        DummyTrainLoader(
            DummyDatasetConfig(observation="obs", pipeline=ocean_pipeline())
        ),
        DummyModule(num_out=2),
        DummyLoss(["mse"]),
        DummyTrainer(),
    )
    cfg.prepare_directory(Root())
    assert os.path.isdir(cfg.log_dir)


def test_prepare_root_with_yaml(tmp_path):
    yaml = tmp_path / "cfg.yaml"
    yaml.write_text("x")

    cfg = TrainConfig(
        str(tmp_path),
        1,
        DummyTrainLoader(
            DummyDatasetConfig(observation="obs", pipeline=ocean_pipeline())
        ),
        DummyModule(num_out=2),
        DummyLoss(["mse"]),
        DummyTrainer(),
    )

    cfg.prepare_directory(Root(), yaml)
    assert os.path.exists(Path(cfg.experiment_dir) / "config.yaml")


def test_prepare_non_root(tmp_path):
    cfg = TrainConfig(
        str(tmp_path),
        1,
        DummyTrainLoader(
            DummyDatasetConfig(observation="obs", pipeline=ocean_pipeline())
        ),
        DummyModule(num_out=2),
        DummyLoss(["mse"]),
        DummyTrainer(),
    )

    cfg.prepare_directory(NonRoot())
    assert cfg.log_dir


def test_prepare_non_root_yaml(tmp_path):
    yaml = tmp_path / "cfg.yaml"
    yaml.write_text("x")

    cfg = TrainConfig(
        str(tmp_path),
        1,
        DummyTrainLoader(
            DummyDatasetConfig(observation="obs", pipeline=ocean_pipeline())
        ),
        DummyModule(num_out=2),
        DummyLoss(["mse"]),
        DummyTrainer(),
    )

    cfg.prepare_directory(NonRoot(), yaml)
    assert cfg.checkpoint_dir
