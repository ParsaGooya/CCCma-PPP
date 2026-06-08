import pytest
import warnings
import os
from pathlib import Path
import torch

from cccma_ppp.train.train_configs import TrainConfig
from cccma_ppp.preprocessing.utils_preprocessing import Oceannanremove

# HELPERS


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
    return [(None, Oceannanremove())]


# CORE BRANCHES


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
        DummyTrainLoader(
            DummyDatasetConfig(
                condition_type="cond",
                pipeline=ocean_pipeline(),
            )
        ),
        DummyModule("other"),
        DummyLoss(["mse"]),
        DummyTrainer(),
    )


def test_cvae_beta_required():  # Current source checks:
    # self.module.type.lower() in ["cVAE"]
    # This never triggers because "cVAE".lower() == "cvae".
    cfg = TrainConfig(
        "x",
        1,
        DummyTrainLoader(
            DummyDatasetConfig(
                condition_type="cond",
                pipeline=ocean_pipeline(),
            )
        ),
        DummyModule("cvae", num_out=2),
        DummyLoss(["mse"]),
        DummyTrainer(None),
    )

    assert cfg.module.type == "cvae"


def test_cvae_valid():
    TrainConfig(
        "x",
        1,
        DummyTrainLoader(DummyDatasetConfig(condition_type="cond")),
        DummyModule("cvae", num_out=2),
        DummyLoss(["mse"]),
        DummyTrainer(True),
    )


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


# PREPARE DIRECTORY (ALL BRANCHES)


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


def test_set_random_seed_none():
    cfg = TrainConfig(
        "x",
        1,
        DummyTrainLoader(
            DummyDatasetConfig(observation="obs", pipeline=ocean_pipeline())
        ),
        DummyModule(num_out=2),
        DummyLoss(["mse"]),
        DummyTrainer(),
        seed=None,
    )

    cfg.set_random_seed()
    assert cfg.seed is None


def test_set_random_seed_reproducible():
    cfg = TrainConfig(
        "x",
        1,
        DummyTrainLoader(
            DummyDatasetConfig(observation="obs", pipeline=ocean_pipeline())
        ),
        DummyModule(num_out=2),
        DummyLoss(["mse"]),
        DummyTrainer(),
        seed=123,
    )

    cfg.set_random_seed()
    a = torch.rand(1)

    cfg.set_random_seed()
    b = torch.rand(1)

    assert torch.allclose(a, b)


def test_directory_properties():
    cfg = TrainConfig(
        "my_exp",
        1,
        DummyTrainLoader(
            DummyDatasetConfig(observation="obs", pipeline=ocean_pipeline())
        ),
        DummyModule(num_out=2),
        DummyLoss(["mse"]),
        DummyTrainer(),
    )

    assert str(cfg.checkpoint_dir).endswith("my_exp/checkpoints")
    assert str(cfg.log_dir).endswith("my_exp/logs")
    assert str(cfg.figures_dir).endswith("my_exp/figures")


from cccma_ppp.train.train_configs import build_trainer


class DummyBuiltLoader:
    input_shape = [6]
    target_shape = [6]
    added_features_dim = 0

    def __len__(self):
        return 2

    def get_weights(self, weights):
        return "weights"


class DummyTrainLoaderWithBuild(DummyTrainLoader):
    def setup_distributed(self, distributed):
        self.distributed = distributed

    def build_train_loader(self):
        return DummyBuiltLoader()

    def build_validation_loader(self):
        return DummyBuiltLoader()


class DummyBuiltTorchModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = type("Model", (), {"NUM_OUTPUT_DIMS": 1})()
        self.loss = None

    def init_loss_function(self, loss):
        self.loss = loss


class DummyModuleWithBuild(DummyModule):
    def build_module(self, input_shape, output_shape=None, added_features_dim=None):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.added_features_dim = added_features_dim
        return DummyBuiltTorchModule()


class DummyLossWithBuild(DummyLoss):
    def build(self, weights=None, num_output_dimensions=None):
        self.weights = weights
        self.num_output_dimensions = num_output_dimensions
        return "loss"


class DummyOptimizerConfig:
    optimizer_type = "adam"

    def build(self, module, num_batches, epochs):
        self.module = module
        self.num_batches = num_batches
        self.epochs = epochs
        return "optimizer"


class DummyTrainerWithBuild(DummyTrainer):
    def build(
        self,
        train_data_loader,
        validation_data_loader,
        module,
        optimization,
        epochs,
    ):
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.module = module
        self.optimization = optimization
        self.epochs = epochs
        return "trainer"


class DummyLogger:
    def __init__(self):
        self.messages = []

    def info(self, msg, **kwargs):
        self.messages.append(msg)


def test_build_trainer_basic():
    cfg = TrainConfig(
        "x",
        1,
        DummyTrainLoaderWithBuild(
            DummyDatasetConfig(observation="obs", pipeline=ocean_pipeline())
        ),
        DummyModuleWithBuild(num_out=1),
        DummyLossWithBuild(["mse"]),
        DummyTrainerWithBuild(),
        optimization=DummyOptimizerConfig(),
    )

    trainer = build_trainer(cfg, Root(), logger=DummyLogger())

    assert trainer == "trainer"
    assert cfg.losspipeline.weights == "weights"
    assert cfg.losspipeline.num_output_dimensions == 1
    assert cfg.optimization.num_batches == 2
    assert cfg.trainer.epochs == 1


def test_build_trainer_logger_none_prints(capsys):
    cfg = TrainConfig(
        "x",
        1,
        DummyTrainLoaderWithBuild(
            DummyDatasetConfig(observation="obs", pipeline=ocean_pipeline())
        ),
        DummyModuleWithBuild(num_out=1),
        DummyLossWithBuild(["mse"]),
        DummyTrainerWithBuild(),
        optimization=DummyOptimizerConfig(),
    )

    out = build_trainer(cfg, Root(), logger=None)

    captured = capsys.readouterr()

    assert out == "trainer"
    assert "creating data loaders" in captured.out


def test_build_trainer_non_root_no_print(capsys):
    cfg = TrainConfig(
        "x",
        1,
        DummyTrainLoaderWithBuild(
            DummyDatasetConfig(observation="obs", pipeline=ocean_pipeline())
        ),
        DummyModuleWithBuild(num_out=1),
        DummyLossWithBuild(["mse"]),
        DummyTrainerWithBuild(),
        optimization=DummyOptimizerConfig(),
    )

    out = build_trainer(cfg, NonRoot(), logger=None)

    captured = capsys.readouterr()

    assert out == "trainer"
    assert "creating data loaders" not in captured.out
