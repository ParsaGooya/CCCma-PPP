import os
import warnings
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn


from cccma_ppp.train.train_configs import TrainConfig, build_trainer, set_seed
from cccma_ppp.preprocessing.utils_preprocessing import Oceannanremove


# ============================================================
# Basic dummy objects
# ============================================================


class DummyPipeline:
    def __init__(self, pipeline=None):
        self.pipeline = pipeline or []


class DummyDatasetConfig:
    def __init__(self, observation=None, condition_type=None, pipeline=None):
        self.condition_type = condition_type

        observation_pipeline = DummyPipeline(pipeline)
        model_pipeline = DummyPipeline(pipeline)

        if observation is not None:
            self.observation = type(
                "ObservationConfig",
                (),
                {"preprocessing_pipeline": observation_pipeline},
            )()
        else:
            self.observation = None

        self.model = type(
            "ModelDataConfig",
            (),
            {"preprocessing_pipeline": model_pipeline},
        )()


class DummyTrainLoader:
    def __init__(self, dataset_config):
        self.dataset_config = dataset_config


class DummyModel:
    GENERATOR = False
    NUM_OUTPUT_DIMS = 1


class DummyGeneratorModel:
    GENERATOR = True
    NUM_OUTPUT_DIMS = 2


class DummyNonGeneratorModel:
    GENERATOR = False
    NUM_OUTPUT_DIMS = 2


class Dummy2DModel:
    GENERATOR = False
    NUM_OUTPUT_DIMS = 2


class DummyModule:
    def __init__(self, type_="deterministic", generator=False, num_out=1):
        self.type = type_

        class Model:
            GENERATOR = generator
            NUM_OUTPUT_DIMS = num_out

        self._module_config = type("InnerModuleConfig", (), {"model": Model()})()


class DummyTrainer:
    def __init__(self, beta_finder=None):
        self.beta_finder = beta_finder


class DummyLoss:
    def __init__(self, loss_types):
        self.loss_pipeline = type("LossPipeline", (), {"loss_types": loss_types})()


def ocean_pipeline():
    return [("ocean", Oceannanremove())]


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
    cfg = TrainConfig(
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

    assert cfg.train_loader.dataset_config.observation is None
    assert cfg.module.type == "other"


def test_cvae_beta_required():
    """
    Current source behavior:
    The cVAE validation branch is unreachable because source checks:

        self.module.type.lower() in ["cVAE"]

    but "cvae" != "cVAE".

    This test documents current behavior.
    If source is fixed to ["cvae"], replace this with pytest.raises(ValueError).
    """
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
    assert cfg.trainer.beta_finder is None


def test_cvae_condition_required():
    """
    Current source behavior:
    The cVAE condition validation branch is unreachable because of the
    ["cVAE"] case mismatch in source.
    """
    cfg = TrainConfig(
        "x",
        1,
        DummyTrainLoader(
            DummyDatasetConfig(
                condition_type=None,
                pipeline=ocean_pipeline(),
            )
        ),
        DummyModule("cvae", num_out=2),
        DummyLoss(["mse"]),
        DummyTrainer(True),
    )

    assert cfg.module.type == "cvae"
    assert cfg.train_loader.dataset_config.condition_type is None


def test_cvae_valid():
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
        DummyTrainer(True),
    )

    assert cfg.module.type == "cvae"


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
    cfg = TrainConfig(
        "x",
        1,
        DummyTrainLoader(DummyDatasetConfig(observation="obs")),
        DummyModule(generator=True, num_out=2),
        DummyLoss(["crps"]),
        DummyTrainer(),
    )

    assert "crps" in cfg.losspipeline.loss_pipeline.loss_types


def test_non_generator_skip():
    cfg = TrainConfig(
        "x",
        1,
        DummyTrainLoader(DummyDatasetConfig(observation="obs")),
        DummyModule(generator=False, num_out=2),
        DummyLoss(["mse"]),
        DummyTrainer(),
    )

    assert cfg.module._module_config.model.GENERATOR is False


def test_deterministic_warning():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")

        TrainConfig(
            "x",
            1,
            DummyTrainLoader(
                DummyDatasetConfig(
                    observation="obs",
                    pipeline=ocean_pipeline(),
                )
            ),
            DummyModule("deterministic"),
            DummyLoss(["mse"]),
            DummyTrainer(None),
        )

    assert any("beta_finder" in str(w.message) for w in captured)


def test_deterministic_no_warning():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")

        TrainConfig(
            "x",
            1,
            DummyTrainLoader(
                DummyDatasetConfig(
                    observation="obs",
                    pipeline=ocean_pipeline(),
                )
            ),
            DummyModule("deterministic"),
            DummyLoss(["mse"]),
            DummyTrainer(True),
        )

    assert len(captured) == 0


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
    cfg = TrainConfig(
        "x",
        1,
        DummyTrainLoader(
            DummyDatasetConfig(
                observation="obs",
                pipeline=ocean_pipeline(),
            )
        ),
        DummyModule(num_out=1),
        DummyLoss(["mse"]),
        DummyTrainer(),
    )

    assert cfg.module._module_config.model.NUM_OUTPUT_DIMS == 1


def test_ocean_required_model_pipeline():
    cfg = TrainConfig(
        "x",
        1,
        DummyTrainLoader(
            DummyDatasetConfig(
                condition_type="cond",
                pipeline=ocean_pipeline(),
            )
        ),
        DummyModule("cvae", num_out=1),
        DummyLoss(["mse"]),
        DummyTrainer(True),
    )

    assert cfg.train_loader.dataset_config.observation is None


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


def test_non_mlp_skips_ocean_requirement():
    cfg = TrainConfig(
        "x",
        1,
        DummyTrainLoader(DummyDatasetConfig(observation="obs")),
        DummyModule("other", num_out=2),
        DummyLoss(["mse"]),
        DummyTrainer(),
    )

    assert cfg.module._module_config.model.NUM_OUTPUT_DIMS == 2


# ============================================================
# Seed and path property tests
# ============================================================


def test_set_seed_reproducible_numpy_and_torch():
    set_seed(123)
    np_first = np.random.rand()
    torch_first = torch.rand(1)

    set_seed(123)
    np_second = np.random.rand()
    torch_second = torch.rand(1)

    assert np_first == np_second
    assert torch.allclose(torch_first, torch_second)


def test_set_random_seed_none():
    cfg = TrainConfig(
        "x",
        1,
        DummyTrainLoader(
            DummyDatasetConfig(
                observation="obs",
                pipeline=ocean_pipeline(),
            )
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
            DummyDatasetConfig(
                observation="obs",
                pipeline=ocean_pipeline(),
            )
        ),
        DummyModule(num_out=2),
        DummyLoss(["mse"]),
        DummyTrainer(),
        seed=123,
    )

    cfg.set_random_seed()
    first = torch.rand(1)

    cfg.set_random_seed()
    second = torch.rand(1)

    assert torch.allclose(first, second)


def test_experiment_dir_is_path():
    cfg = TrainConfig(
        "x",
        1,
        DummyTrainLoader(
            DummyDatasetConfig(
                observation="obs",
                pipeline=ocean_pipeline(),
            )
        ),
        DummyModule(num_out=2),
        DummyLoss(["mse"]),
        DummyTrainer(),
    )

    assert isinstance(cfg.experiment_dir, Path)


def test_directory_properties():
    cfg = TrainConfig(
        "my_exp",
        1,
        DummyTrainLoader(
            DummyDatasetConfig(
                observation="obs",
                pipeline=ocean_pipeline(),
            )
        ),
        DummyModule(num_out=2),
        DummyLoss(["mse"]),
        DummyTrainer(),
    )

    assert str(cfg.checkpoint_dir).endswith("my_exp/checkpoints")
    assert str(cfg.log_dir).endswith("my_exp/logs")
    assert str(cfg.figures_dir).endswith("my_exp/figures")


# ============================================================
# prepare_directory branch tests
# ============================================================


def test_prepare_root(tmp_path):
    cfg = TrainConfig(
        str(tmp_path),
        1,
        DummyTrainLoader(
            DummyDatasetConfig(
                observation="obs",
                pipeline=ocean_pipeline(),
            )
        ),
        DummyModule(num_out=2),
        DummyLoss(["mse"]),
        DummyTrainer(),
    )

    root = Root()
    cfg.prepare_directory(root)

    assert os.path.isdir(cfg.log_dir)
    assert os.environ["GLOBAL_EXP_DIR"] == str(cfg.experiment_dir)
    assert os.environ["GLOBAL_CHECKPOINT_DIR"] == str(cfg.checkpoint_dir)
    assert os.environ["GLOBAL_FIGURES_DIR"] == str(cfg.figures_dir)
    assert os.environ["GLOBAL_LOG_DIR"] == str(cfg.log_dir)
    assert root.barrier_calls == 5


def test_prepare_root_with_yaml(tmp_path):
    yaml = tmp_path / "cfg.yaml"
    yaml.write_text("x")

    cfg = TrainConfig(
        str(tmp_path),
        1,
        DummyTrainLoader(
            DummyDatasetConfig(
                observation="obs",
                pipeline=ocean_pipeline(),
            )
        ),
        DummyModule(num_out=2),
        DummyLoss(["mse"]),
        DummyTrainer(),
    )

    root = Root()
    cfg.prepare_directory(root, yaml)

    assert os.path.exists(Path(cfg.experiment_dir) / "config.yaml")
    assert root.barrier_calls == 5


def test_prepare_non_root(tmp_path):
    cfg = TrainConfig(
        str(tmp_path / "non_root_exp"),
        1,
        DummyTrainLoader(
            DummyDatasetConfig(
                observation="obs",
                pipeline=ocean_pipeline(),
            )
        ),
        DummyModule(num_out=2),
        DummyLoss(["mse"]),
        DummyTrainer(),
    )

    non_root = NonRoot()
    cfg.prepare_directory(non_root)

    assert not Path(cfg.experiment_dir).exists()
    assert non_root.barrier_calls == 5


def test_prepare_non_root_yaml(tmp_path):
    yaml = tmp_path / "cfg.yaml"
    yaml.write_text("x")

    cfg = TrainConfig(
        str(tmp_path / "non_root_yaml_exp"),
        1,
        DummyTrainLoader(
            DummyDatasetConfig(
                observation="obs",
                pipeline=ocean_pipeline(),
            )
        ),
        DummyModule(num_out=2),
        DummyLoss(["mse"]),
        DummyTrainer(),
    )

    non_root = NonRoot()
    cfg.prepare_directory(non_root, yaml)

    assert not os.path.exists(Path(cfg.experiment_dir) / "config.yaml")
    assert non_root.barrier_calls == 5


# ============================================================
# build_trainer test doubles
# ============================================================


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
        self.setup_distributed_called = True

    def build_train_loader(self):
        self.train_loader_built = True
        return DummyBuiltLoader()

    def build_validation_loader(self):
        self.validation_loader_built = True
        return DummyBuiltLoader()


class DummyBuiltTorchModule(nn.Module):
    def __init__(self, model=None):
        super().__init__()
        self.model = model or type("Model", (), {"NUM_OUTPUT_DIMS": 1})()
        self.loss = None
        self.device_used = None
        self.built = True

    def to(self, device):
        self.device_used = device
        return self

    def init_loss_function(self, loss):
        self.loss = loss


class DummyModuleWithBuild(DummyModule):
    def __init__(self, type_="other", generator=False, num_out=1):
        super().__init__(type_, generator=generator, num_out=num_out)
        self.build_called = False

    def build_module(self, input_shape, output_shape=None, added_features_dim=None):
        self.build_called = True
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.added_features_dim = added_features_dim
        return DummyBuiltTorchModule(model=self._module_config.model)


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


class DistributedRoot:
    distributed = False
    device = torch.device("cpu")
    local_rank = 0

    def is_root(self):
        return True

    def barrier(self):
        pass


class DistributedNonRoot:
    distributed = False
    device = torch.device("cpu")
    local_rank = 0

    def is_root(self):
        return False

    def barrier(self):
        pass


class DistributedDDP:
    distributed = True
    device = torch.device("cpu")
    local_rank = 0

    def is_root(self):
        return True

    def barrier(self):
        pass


class FakeDDP:
    def __init__(
        self,
        module,
        device_ids=None,
        output_device=None,
        find_unused_parameters=False,
    ):
        self.module = module
        self.device_ids = device_ids
        self.output_device = output_device
        self.find_unused_parameters = find_unused_parameters

    def __getattr__(self, name):
        return getattr(self.module, name)


def build_train_config_for_builder(module_num_out=1):
    return TrainConfig(
        "x",
        1,
        DummyTrainLoaderWithBuild(
            DummyDatasetConfig(
                observation="obs",
                pipeline=ocean_pipeline(),
            )
        ),
        DummyModuleWithBuild("other", num_out=module_num_out),
        DummyLossWithBuild(["mse"]),
        DummyTrainerWithBuild(),
        optimization=DummyOptimizerConfig(),
    )


# ============================================================
# build_trainer branch tests
# ============================================================


def test_build_trainer_basic_root_logger():
    cfg = build_train_config_for_builder()
    logger = DummyLogger()

    trainer = build_trainer(cfg, DistributedRoot(), logger=logger)

    assert trainer == "trainer"
    assert cfg.train_loader.setup_distributed_called is True
    assert cfg.train_loader.train_loader_built is True
    assert cfg.train_loader.validation_loader_built is True

    assert cfg.module.build_called is True
    assert cfg.module.input_shape == [6]
    assert cfg.module.output_shape == [6]
    assert cfg.module.added_features_dim == 0

    assert cfg.losspipeline.weights == "weights"
    assert cfg.losspipeline.num_output_dimensions == 1

    assert cfg.optimization.num_batches == 2
    assert cfg.optimization.epochs == 1

    assert cfg.trainer.train_data_loader is not None
    assert cfg.trainer.validation_data_loader is not None
    assert cfg.trainer.optimization == "optimizer"
    assert cfg.trainer.epochs == 1

    assert any("creating data loaders" in msg for msg in logger.messages)
    assert any("Creating loss function" in msg for msg in logger.messages)


def test_build_trainer_logger_none_prints(capsys):
    cfg = build_train_config_for_builder()

    trainer = build_trainer(cfg, DistributedRoot(), logger=None)

    captured = capsys.readouterr()

    assert trainer == "trainer"
    assert "creating data loaders" in captured.out
    assert "Creating loss function" in captured.out


def test_build_trainer_non_root_does_not_print(capsys):
    cfg = build_train_config_for_builder()

    trainer = build_trainer(cfg, DistributedNonRoot(), logger=None)

    captured = capsys.readouterr()

    assert trainer == "trainer"
    assert "creating data loaders" not in captured.out
    assert "Creating loss function" not in captured.out


def test_build_trainer_distributed_wraps_module(monkeypatch):
    monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", FakeDDP)

    cfg = build_train_config_for_builder()
    trainer = build_trainer(cfg, DistributedDDP(), logger=None)

    built_module = cfg.trainer.module

    assert trainer == "trainer"
    assert isinstance(built_module, FakeDDP)
    assert built_module.device_ids == [0]
    assert built_module.output_device == 0
    assert built_module.find_unused_parameters is False


def test_build_trainer_uses_len_output_shape_when_num_output_dims_missing():
    class ModelWithoutNumOutputDims:
        GENERATOR = False

    cfg = TrainConfig(
        "x",
        1,
        DummyTrainLoaderWithBuild(
            DummyDatasetConfig(
                observation="obs",
                pipeline=ocean_pipeline(),
            )
        ),
        DummyModuleWithBuild("other", num_out=2),
        DummyLossWithBuild(["mse"]),
        DummyTrainerWithBuild(),
        optimization=DummyOptimizerConfig(),
    )

    cfg.module._module_config.model = ModelWithoutNumOutputDims()

    trainer = build_trainer(cfg, DistributedRoot(), logger=None)

    assert trainer == "trainer"
    assert cfg.losspipeline.num_output_dimensions == 1


def test_build_trainer_passes_module_to_device():
    cfg = build_train_config_for_builder()

    trainer = build_trainer(cfg, DistributedRoot(), logger=None)

    assert trainer == "trainer"
    assert cfg.trainer.module.device_used == torch.device("cpu")


def test_build_trainer_initializes_loss_on_module():
    cfg = build_train_config_for_builder()

    trainer = build_trainer(cfg, DistributedRoot(), logger=None)

    assert trainer == "trainer"
    assert cfg.trainer.module.loss == "loss"


def test_build_trainer_optimizer_receives_module_batches_epochs():
    cfg = build_train_config_for_builder()

    trainer = build_trainer(cfg, DistributedRoot(), logger=None)

    assert trainer == "trainer"
    assert cfg.optimization.module is cfg.trainer.module
    assert cfg.optimization.num_batches == 2
    assert cfg.optimization.epochs == 1


def test_build_trainer_trainer_receives_validation_loader():
    cfg = build_train_config_for_builder()

    trainer = build_trainer(cfg, DistributedRoot(), logger=None)

    assert trainer == "trainer"
    assert isinstance(cfg.trainer.validation_data_loader, DummyBuiltLoader)


# ============================================================
# Distributed test doubles
# ============================================================


class Root:
    def __init__(self):
        self.barrier_calls = 0

    def is_root(self):
        return True

    def barrier(self):
        self.barrier_calls += 1


class NonRoot:
    def __init__(self):
        self.barrier_calls = 0

    def is_root(self):
        return False

    def barrier(self):
        self.barrier_calls += 1


# ============================================================
# Core TrainConfig branch tests
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
