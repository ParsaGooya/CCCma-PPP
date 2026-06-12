import os
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from cccma_ppp.generic.runtime import RuntimeContext
from cccma_ppp.preprocessing.utils_preprocessing import Oceannanremove
from cccma_ppp.train.train_configs import (
    TrainConfig,
    build_trainer,
    set_seed,
)


def ocean_pipeline():
    return [("ocean", Oceannanremove())]


def _active_pipeline(loader):
    if loader.dataset_config.observation is not None:
        return loader.dataset_config.observation.preprocessing_pipeline.pipeline
    return loader.dataset_config.model.preprocessing_pipeline.pipeline


def make_cfg(*args, **kwargs):
    loader = args[2]
    module = args[3]
    pipeline = _active_pipeline(loader)
    num_out = module._module_config.model_config.NUM_OUTPUT_DIMS

    if num_out != 1:
        pipeline[:] = [
            step for step in pipeline if not isinstance(step[1], Oceannanremove)
        ]

    return TrainConfig(*args, **kwargs)


class DummyPipeline:
    def __init__(self, pipeline=None, name=None, load_dir=None, fitted=True):
        self.name = name
        self.load_dir = load_dir
        self.load_name = None
        self.fitted = fitted
        self.pipeline = list(pipeline or [])
        self.fit_calls = []
        self.load_calls = []
        self.transform_calls = []
        self.add_calls = []
        self.fitted_preprocessors = []

    def set_name(self, name):
        self.name = name
        return self

    def fit(
        self,
        base_data,
        mask=None,
        save=False,
        save_path=None,
        save_name=None,
    ):
        self.fit_calls.append(
            {
                "base_data": base_data,
                "mask": mask,
                "save": save,
                "save_path": save_path,
                "save_name": save_name,
            }
        )
        self.fitted = True

    def _load_from_memory(self, load_dir, load_name=None):
        self.load_calls.append(
            {
                "load_dir": Path(load_dir),
                "load_name": load_name,
            }
        )
        self.fitted = True

    def transform(self, data):
        self.transform_calls.append(data)
        return data

    def add_fitted_preprocessor(self, preprocessor, index=0):
        self.add_calls.append(
            {
                "preprocessor": preprocessor,
                "index": index,
            }
        )
        self.fitted_preprocessors.insert(index, preprocessor)

    def get_preprocessors(self, name):
        for preprocessor in self.fitted_preprocessors:
            if name.lower() == "oceannanremover" and isinstance(
                preprocessor,
                Oceannanremove,
            ):
                return preprocessor

            if name.lower() in preprocessor.__class__.__name__.lower():
                return preprocessor

        return None


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
        self.input_var_metadata = {}
        self.target_var_metadata = {}

    def sanitize(self, module):
        if not getattr(self, "_force_sanitize", False):
            return

        model = module._module_config.model_config
        pipeline = _active_pipeline(self)

        if model.NUM_OUTPUT_DIMS == 1:
            if not any(isinstance(step[1], Oceannanremove) for step in pipeline):
                pipeline.append(("auto_ocean", Oceannanremove()))
        else:
            pipeline[:] = [
                step for step in pipeline if not isinstance(step[1], Oceannanremove)
            ]


class DummyModule:
    def __init__(self, type_="deterministic", generator=False, num_out=1):
        self.type = type_

        class Model:
            GENERATOR = generator
            NUM_OUTPUT_DIMS = num_out

        model = Model()

        self._module_config = type(
            "InnerModuleConfig",
            (),
            {
                "model": model,
                "model_config": model,
            },
        )()


class DummyTrainer:
    def __init__(self, beta_finder=None):
        self.beta_finder = beta_finder


class DummyLoss:
    def __init__(self, loss_types):
        self.loss_pipeline = type("LossPipeline", (), {"loss_types": loss_types})()


def test_no_observation_default_raises():
    with pytest.raises(ValueError):
        make_cfg(
            "x",
            1,
            DummyTrainLoader(DummyDatasetConfig()),
            DummyModule("default"),
            DummyLoss(["mse"]),
            DummyTrainer(),
        )


def test_no_observation_deterministic_raises():
    with pytest.raises(ValueError):
        make_cfg(
            "x",
            1,
            DummyTrainLoader(DummyDatasetConfig()),
            DummyModule("deterministic"),
            DummyLoss(["mse"]),
            DummyTrainer(),
        )


def test_no_observation_other_type_valid():
    cfg = make_cfg(
        "x",
        1,
        DummyTrainLoader(
            DummyDatasetConfig(
                condition_type="cond",
                pipeline=ocean_pipeline(),
            )
        ),
        DummyModule("other", num_out=2),
        DummyLoss(["mse"]),
        DummyTrainer(),
    )

    assert cfg.train_loader.dataset_config.observation is None
    assert cfg.module.type == "other"


def test_cvae_lowercase_branch_current_source_behavior_beta_not_required():
    cfg = make_cfg(
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


def test_cvae_lowercase_branch_current_source_behavior_condition_not_required():
    cfg = make_cfg(
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


def test_cvae_uppercase_branch_currently_unreachable():
    cfg = make_cfg(
        "x",
        1,
        DummyTrainLoader(
            DummyDatasetConfig(
                condition_type="cond",
                pipeline=[],
            )
        ),
        DummyModule("cVAE", num_out=2),
        DummyLoss(["mse"]),
        DummyTrainer(None),
    )

    assert cfg.module.type == "cVAE"
    assert cfg.trainer.beta_finder is None


def test_generator_requires_crps():
    with pytest.raises(RuntimeError, match="crps"):
        make_cfg(
            "x",
            1,
            DummyTrainLoader(DummyDatasetConfig(observation="obs")),
            DummyModule(generator=True, num_out=2),
            DummyLoss(["mse"]),
            DummyTrainer(),
        )


def test_generator_valid_with_crps():
    cfg = make_cfg(
        "x",
        1,
        DummyTrainLoader(DummyDatasetConfig(observation="obs")),
        DummyModule(generator=True, num_out=2),
        DummyLoss(["mse", "crps"]),
        DummyTrainer(),
    )

    assert "crps" in cfg.losspipeline.loss_pipeline.loss_types


def test_non_generator_with_crps_is_valid():
    cfg = make_cfg(
        "x",
        1,
        DummyTrainLoader(DummyDatasetConfig(observation="obs")),
        DummyModule(generator=False, num_out=2),
        DummyLoss(["crps"]),
        DummyTrainer(),
    )

    assert cfg.module._module_config.model_config.GENERATOR is False


def test_deterministic_warning_when_beta_finder_none():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")

        make_cfg(
            "x",
            1,
            DummyTrainLoader(
                DummyDatasetConfig(
                    observation="obs",
                    pipeline=ocean_pipeline(),
                )
            ),
            DummyModule("deterministic", num_out=1),
            DummyLoss(["mse"]),
            DummyTrainer(None),
        )

    assert any("beta_finder" in str(w.message) for w in captured)


def test_deterministic_no_warning_when_beta_finder_present():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")

        make_cfg(
            "x",
            1,
            DummyTrainLoader(
                DummyDatasetConfig(
                    observation="obs",
                    pipeline=ocean_pipeline(),
                )
            ),
            DummyModule("deterministic", num_out=1),
            DummyLoss(["mse"]),
            DummyTrainer(True),
        )

    assert len(captured) == 0


def test_default_model_warning_branch():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")

        make_cfg(
            "x",
            1,
            DummyTrainLoader(
                DummyDatasetConfig(
                    observation="obs",
                    pipeline=ocean_pipeline(),
                )
            ),
            DummyModule("default", num_out=1),
            DummyLoss(["mse"]),
            DummyTrainer(None),
        )

    assert any("beta_finder" in str(w.message) for w in captured)


def test_ocean_required_fail_observation_for_mlp():
    with pytest.raises(RuntimeError, match="MLP"):
        TrainConfig(
            "x",
            1,
            DummyTrainLoader(DummyDatasetConfig(observation="obs")),
            DummyModule(num_out=1),
            DummyLoss(["mse"]),
            DummyTrainer(),
        )


def test_ocean_required_pass_observation_for_mlp():
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

    assert cfg.module._module_config.model_config.NUM_OUTPUT_DIMS == 1


def test_ocean_required_model_pipeline_for_mlp():
    cfg = TrainConfig(
        "x",
        1,
        DummyTrainLoader(
            DummyDatasetConfig(
                observation=None,
                condition_type="cond",
                pipeline=ocean_pipeline(),
            )
        ),
        DummyModule("other", num_out=1),
        DummyLoss(["mse"]),
        DummyTrainer(),
    )

    assert cfg.train_loader.dataset_config.observation is None


def test_ocean_required_model_fail_for_mlp():
    with pytest.raises(RuntimeError, match="MLP"):
        TrainConfig(
            "x",
            1,
            DummyTrainLoader(DummyDatasetConfig(condition_type="cond")),
            DummyModule("other", num_out=1),
            DummyLoss(["mse"]),
            DummyTrainer(),
        )


def test_non_mlp_observation_with_ocean_raises_without_sanitize():
    with pytest.raises(RuntimeError, match="non-MLP"):
        TrainConfig(
            "x",
            1,
            DummyTrainLoader(
                DummyDatasetConfig(
                    observation="obs",
                    pipeline=ocean_pipeline(),
                )
            ),
            DummyModule("other", num_out=2),
            DummyLoss(["mse"]),
            DummyTrainer(),
        )


def test_non_mlp_model_pipeline_with_ocean_raises_without_sanitize():
    with pytest.raises(RuntimeError, match="non-MLP"):
        TrainConfig(
            "x",
            1,
            DummyTrainLoader(
                DummyDatasetConfig(
                    observation=None,
                    condition_type="cond",
                    pipeline=ocean_pipeline(),
                )
            ),
            DummyModule("other", num_out=2),
            DummyLoss(["mse"]),
            DummyTrainer(),
        )


def test_non_mlp_without_ocean_is_valid():
    cfg = make_cfg(
        "x",
        1,
        DummyTrainLoader(DummyDatasetConfig(observation="obs")),
        DummyModule("other", num_out=2),
        DummyLoss(["mse"]),
        DummyTrainer(),
    )

    assert cfg.module._module_config.model_config.NUM_OUTPUT_DIMS == 2


def test_model_pipeline_used_when_observation_missing_non_mlp_valid():
    cfg = make_cfg(
        "x",
        1,
        DummyTrainLoader(
            DummyDatasetConfig(
                observation=None,
                condition_type="cond",
                pipeline=[],
            )
        ),
        DummyModule("other", num_out=2),
        DummyLoss(["mse"]),
        DummyTrainer(),
    )

    assert cfg.train_loader.dataset_config.observation is None
    assert cfg.train_loader.dataset_config.model.preprocessing_pipeline.pipeline == []


def test_max_epochs_none_becomes_inf():
    cfg = make_cfg(
        "x",
        None,
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

    assert cfg.max_epochs == float("inf")


def test_zero_epochs_is_allowed():
    cfg = make_cfg(
        "x",
        0,
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

    assert cfg.max_epochs == 0


def test_negative_epochs_assertion():
    with pytest.raises(AssertionError):
        make_cfg(
            "x",
            -1,
            DummyTrainLoader(DummyDatasetConfig()),
            DummyModule(),
            DummyLoss(["mse"]),
            DummyTrainer(),
        )


def test_set_seed_reproducible_numpy_and_torch():
    set_seed(123)
    np_first = np.random.rand()
    torch_first = torch.rand(1)

    set_seed(123)
    np_second = np.random.rand()
    torch_second = torch.rand(1)

    assert np_first == np_second
    assert torch.allclose(torch_first, torch_second)


def test_set_seed_reproducible_multiple_values():
    set_seed(999)
    np_values_1 = np.random.rand(3)
    torch_values_1 = torch.rand(3)

    set_seed(999)
    np_values_2 = np.random.rand(3)
    torch_values_2 = torch.rand(3)

    assert np.allclose(np_values_1, np_values_2)
    assert torch.allclose(torch_values_1, torch_values_2)


def test_set_random_seed_none():
    cfg = make_cfg(
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
    cfg = make_cfg(
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


def test_set_random_seed_zero_is_valid():
    cfg = make_cfg(
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
        seed=0,
    )

    cfg.set_random_seed()
    first = torch.rand(1)

    cfg.set_random_seed()
    second = torch.rand(1)

    assert torch.allclose(first, second)


class Root:
    def __init__(self):
        self.barrier_calls = 0

    def is_root(self):
        return True

    def barrier(self):
        self.barrier_calls += 1

    def set_env(self, cfg):
        os.environ["GLOBAL_EXP_DIR"] = str(cfg.experiment_dir)
        os.environ["GLOBAL_CHECKPOINT_DIR"] = str(cfg.checkpoint_dir)
        os.environ["GLOBAL_FIGURES_DIR"] = str(cfg.figures_dir)
        os.environ["GLOBAL_LOG_DIR"] = str(cfg.log_dir)


class NonRoot:
    def __init__(self):
        self.barrier_calls = 0

    def is_root(self):
        return False

    def barrier(self):
        self.barrier_calls += 1


def valid_directory_cfg(path, num_out=2, loader=None):
    loader = loader or DummyTrainLoader(
        DummyDatasetConfig(
            observation="obs",
            pipeline=ocean_pipeline(),
        )
    )

    return make_cfg(
        str(path),
        1,
        loader,
        DummyModule(num_out=num_out),
        DummyLoss(["mse"]),
        DummyTrainer(),
    )


def test_experiment_dir_is_path():
    cfg = valid_directory_cfg("x")

    assert isinstance(cfg.experiment_dir, Path)


def test_directory_properties():
    cfg = valid_directory_cfg("my_exp")

    assert str(cfg.checkpoint_dir).endswith("my_exp/checkpoints")
    assert str(cfg.log_dir).endswith("my_exp/logs")
    assert str(cfg.figures_dir).endswith("my_exp/figures")


def test_prepare_root_sets_dirs_runtime_context_and_env(tmp_path):
    cfg = valid_directory_cfg(tmp_path)

    root = Root()
    cfg.prepare_directory(root)
    root.set_env(cfg)

    assert Path(cfg.experiment_dir).exists()
    assert Path(cfg.checkpoint_dir).exists()
    assert Path(cfg.log_dir).exists()
    assert Path(cfg.figures_dir).exists()

    assert RuntimeContext.GLOBAL_EXP_DIR == str(cfg.experiment_dir)
    assert RuntimeContext.GLOBAL_CHECKPOINT_DIR == str(cfg.checkpoint_dir)
    assert RuntimeContext.GLOBAL_FIGURES_DIR == str(cfg.figures_dir)
    assert RuntimeContext.GLOBAL_LOG_DIR == str(cfg.log_dir)

    assert os.environ["GLOBAL_EXP_DIR"] == str(cfg.experiment_dir)
    assert os.environ["GLOBAL_CHECKPOINT_DIR"] == str(cfg.checkpoint_dir)
    assert os.environ["GLOBAL_FIGURES_DIR"] == str(cfg.figures_dir)
    assert os.environ["GLOBAL_LOG_DIR"] == str(cfg.log_dir)

    assert root.barrier_calls == 2


def test_prepare_root_with_yaml_path(tmp_path):
    yaml = tmp_path / "cfg.yaml"
    yaml.write_text("x")

    cfg = valid_directory_cfg(tmp_path / "exp")

    root = Root()
    cfg.prepare_directory(root, yaml)

    assert (Path(cfg.experiment_dir) / "config.yaml").exists()
    assert root.barrier_calls == 2


def test_prepare_root_with_yaml_string_path(tmp_path):
    yaml = tmp_path / "cfg.yaml"
    yaml.write_text("x")

    cfg = valid_directory_cfg(tmp_path / "exp")

    root = Root()
    cfg.prepare_directory(root, str(yaml))

    assert (Path(cfg.experiment_dir) / "config.yaml").exists()
    assert root.barrier_calls == 2


def test_prepare_root_with_existing_config_yaml_overwrites(tmp_path):
    source_yaml = tmp_path / "source.yaml"
    source_yaml.write_text("new config")

    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    existing_yaml = exp_dir / "config.yaml"
    existing_yaml.write_text("old config")

    cfg = valid_directory_cfg(exp_dir)

    root = Root()
    cfg.prepare_directory(root, source_yaml)

    assert existing_yaml.exists()
    assert existing_yaml.read_text() == "new config"
    assert root.barrier_calls == 2


def test_prepare_root_existing_dirs(tmp_path):
    cfg = valid_directory_cfg(tmp_path / "exp")

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.figures_dir).mkdir(parents=True, exist_ok=True)

    root = Root()
    cfg.prepare_directory(root)

    assert Path(cfg.checkpoint_dir).exists()
    assert Path(cfg.log_dir).exists()
    assert Path(cfg.figures_dir).exists()
    assert root.barrier_calls == 2


def test_prepare_root_without_yaml_does_not_create_config(tmp_path):
    cfg = valid_directory_cfg(tmp_path / "exp_no_yaml")

    root = Root()
    cfg.prepare_directory(root)

    assert Path(cfg.experiment_dir).exists()
    assert not (Path(cfg.experiment_dir) / "config.yaml").exists()
    assert root.barrier_calls == 2


def test_prepare_non_root_does_not_create_dirs(tmp_path):
    cfg = valid_directory_cfg(tmp_path / "non_root_exp")

    non_root = NonRoot()
    cfg.prepare_directory(non_root)

    assert not Path(cfg.experiment_dir).exists()
    assert non_root.barrier_calls == 2


def test_prepare_non_root_yaml_does_not_copy_yaml(tmp_path):
    yaml = tmp_path / "cfg.yaml"
    yaml.write_text("x")

    cfg = valid_directory_cfg(tmp_path / "non_root_yaml_exp")

    non_root = NonRoot()
    cfg.prepare_directory(non_root, yaml)

    assert not (Path(cfg.experiment_dir) / "config.yaml").exists()
    assert non_root.barrier_calls == 2


def test_prepare_non_root_with_max_epochs_none(tmp_path):
    cfg = make_cfg(
        str(tmp_path / "non_root_inf"),
        None,
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

    assert cfg.max_epochs == float("inf")
    assert not cfg.experiment_dir.exists()
    assert non_root.barrier_calls == 2


def test_prepare_directory_sets_runtime_context_metadata(tmp_path):
    loader = DummyTrainLoader(
        DummyDatasetConfig(
            observation="obs",
            pipeline=ocean_pipeline(),
        )
    )
    loader.input_var_metadata = {"tas": {"units": "K"}}
    loader.target_var_metadata = {"pr": {"units": "mm/day"}}

    cfg = valid_directory_cfg(tmp_path / "metadata_exp", num_out=1, loader=loader)

    root = Root()
    cfg.prepare_directory(root)

    assert RuntimeContext.INPUT_VAR_METADATA == {"tas": {"units": "K"}}
    assert RuntimeContext.TARGET_VAR_METADATA == {"pr": {"units": "mm/day"}}


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
        max_epochs=None,
        epochs=None,
    ):
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.module = module
        self.optimization = optimization
        self.epochs = max_epochs if max_epochs is not None else epochs
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


def build_train_config_for_builder(module_num_out=1, loader=None, module=None):
    loader = loader or DummyTrainLoaderWithBuild(
        DummyDatasetConfig(
            observation="obs",
            pipeline=ocean_pipeline(),
        )
    )
    module = module or DummyModuleWithBuild("other", num_out=module_num_out)

    loader._force_sanitize = True
    loader.sanitize(module)

    return make_cfg(
        "x",
        1,
        loader,
        module,
        DummyLossWithBuild(["mse"]),
        DummyTrainerWithBuild(),
        optimization=DummyOptimizerConfig(),
    )


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

    assert cfg.optimization.module is cfg.trainer.module
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


def test_build_trainer_non_root_logger_does_not_log():
    cfg = build_train_config_for_builder()
    logger = DummyLogger()

    trainer = build_trainer(cfg, DistributedNonRoot(), logger=logger)

    assert trainer == "trainer"
    assert logger.messages == []


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


def test_build_trainer_distributed_with_logger(monkeypatch):
    monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", FakeDDP)

    cfg = build_train_config_for_builder()
    logger = DummyLogger()

    trainer = build_trainer(cfg, DistributedDDP(), logger=logger)

    assert trainer == "trainer"
    assert isinstance(cfg.trainer.module, FakeDDP)
    assert any("Creating trainer" in msg for msg in logger.messages)


def test_build_trainer_uses_len_output_shape_when_num_output_dims_missing():
    class ModelWithoutNumOutputDims:
        GENERATOR = False

    cfg = make_cfg(
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

    model = ModelWithoutNumOutputDims()
    cfg.module._module_config.model = model
    cfg.module._module_config.model_config = model

    trainer = build_trainer(cfg, DistributedRoot(), logger=None)

    assert trainer == "trainer"
    assert cfg.losspipeline.num_output_dimensions == 1


def test_build_trainer_validation_loader_none():
    class DummyTrainLoaderNoValidation(DummyTrainLoaderWithBuild):
        def build_validation_loader(self):
            self.validation_loader_built = True
            return None

    loader = DummyTrainLoaderNoValidation(
        DummyDatasetConfig(
            observation="obs",
            pipeline=ocean_pipeline(),
        )
    )

    cfg = build_train_config_for_builder(
        loader=loader,
        module=DummyModuleWithBuild("other", num_out=1),
    )

    trainer = build_trainer(cfg, DistributedRoot(), logger=None)

    assert trainer == "trainer"
    assert cfg.trainer.validation_data_loader is None
    assert cfg.train_loader.validation_loader_built is True


def test_build_trainer_loss_receives_none_weights():
    class DummyBuiltLoaderNoWeights(DummyBuiltLoader):
        def get_weights(self, weights):
            return None

    class DummyTrainLoaderWithBuildNoWeights(DummyTrainLoaderWithBuild):
        def build_train_loader(self):
            self.train_loader_built = True
            return DummyBuiltLoaderNoWeights()

        def build_validation_loader(self):
            self.validation_loader_built = True
            return DummyBuiltLoaderNoWeights()

    loader = DummyTrainLoaderWithBuildNoWeights(
        DummyDatasetConfig(
            observation="obs",
            pipeline=ocean_pipeline(),
        )
    )

    cfg = build_train_config_for_builder(
        loader=loader,
        module=DummyModuleWithBuild("other", num_out=1),
    )

    trainer = build_trainer(cfg, DistributedRoot(), logger=None)

    assert trainer == "trainer"
    assert cfg.losspipeline.weights is None
    assert cfg.trainer.module.loss == "loss"


def test_build_trainer_passes_none_added_features_dim():
    class DummyBuiltLoaderNoAddedFeatures:
        input_shape = [6]
        target_shape = [6]
        added_features_dim = None

        def __len__(self):
            return 2

        def get_weights(self, weights):
            return "weights"

    class DummyTrainLoaderNoAddedFeatures(DummyTrainLoaderWithBuild):
        def build_train_loader(self):
            self.train_loader_built = True
            return DummyBuiltLoaderNoAddedFeatures()

        def build_validation_loader(self):
            self.validation_loader_built = True
            return DummyBuiltLoaderNoAddedFeatures()

    loader = DummyTrainLoaderNoAddedFeatures(
        DummyDatasetConfig(
            observation="obs",
            pipeline=ocean_pipeline(),
        )
    )

    cfg = build_train_config_for_builder(
        loader=loader,
        module=DummyModuleWithBuild("other", num_out=1),
    )

    trainer = build_trainer(cfg, DistributedRoot(), logger=None)

    assert trainer == "trainer"
    assert cfg.module.added_features_dim is None


def test_sanitize_mlp_adds_ocean_when_forced():
    loader = DummyTrainLoader(DummyDatasetConfig(observation="obs", pipeline=[]))
    module = DummyModule(num_out=1)

    loader._force_sanitize = True
    loader.sanitize(module)

    pipeline = loader.dataset_config.observation.preprocessing_pipeline.pipeline

    assert any(isinstance(step[1], Oceannanremove) for step in pipeline)


def test_sanitize_no_force_returns_without_changes():
    loader = DummyTrainLoader(DummyDatasetConfig(observation="obs", pipeline=[]))
    module = DummyModule(num_out=1)

    loader.sanitize(module)

    assert loader.dataset_config.observation.preprocessing_pipeline.pipeline == []


def test_sanitize_model_pipeline_non_mlp_removes_ocean_when_forced():
    loader = DummyTrainLoader(
        DummyDatasetConfig(
            observation=None,
            condition_type="cond",
            pipeline=ocean_pipeline(),
        )
    )
    module = DummyModule("other", num_out=2)

    loader._force_sanitize = True
    loader.sanitize(module)

    pipeline = loader.dataset_config.model.preprocessing_pipeline.pipeline

    assert not any(isinstance(step[1], Oceannanremove) for step in pipeline)


def test_make_cfg_sanitizes_model_pipeline_for_non_mlp():
    loader = DummyTrainLoader(
        DummyDatasetConfig(
            observation=None,
            condition_type="cond",
            pipeline=ocean_pipeline(),
        )
    )
    module = DummyModule("other", num_out=2)

    cfg = make_cfg(
        "x",
        1,
        loader,
        module,
        DummyLoss(["mse"]),
        DummyTrainer(),
    )

    pipeline = cfg.train_loader.dataset_config.model.preprocessing_pipeline.pipeline

    assert pipeline == []


def test_make_cfg_keeps_ocean_for_mlp():
    loader = DummyTrainLoader(
        DummyDatasetConfig(
            observation="obs",
            pipeline=ocean_pipeline(),
        )
    )
    module = DummyModule(num_out=1)

    cfg = make_cfg(
        "x",
        1,
        loader,
        module,
        DummyLoss(["mse"]),
        DummyTrainer(),
    )

    pipeline = (
        cfg.train_loader.dataset_config.observation.preprocessing_pipeline.pipeline
    )

    assert any(isinstance(step[1], Oceannanremove) for step in pipeline)


class CvaeReachableType:
    def lower(self):
        return "cVAE"

    def __str__(self):
        return "cVAE"


def test_cvae_reachable_branch_valid():
    cfg = make_cfg(
        "x",
        1,
        DummyTrainLoader(
            DummyDatasetConfig(
                observation=None,
                condition_type="cond",
                pipeline=[],
            )
        ),
        DummyModule(CvaeReachableType(), num_out=2),
        DummyLoss(["mse"]),
        DummyTrainer(True),
    )

    assert cfg.train_loader.dataset_config.condition_type == "cond"
    assert cfg.trainer.beta_finder is True


def test_model_config_without_generator_attribute_is_valid():
    class ModelWithoutGenerator:
        NUM_OUTPUT_DIMS = 2

    module = DummyModule("other", num_out=2)
    model = ModelWithoutGenerator()
    module._module_config.model = model
    module._module_config.model_config = model

    cfg = make_cfg(
        "x",
        1,
        DummyTrainLoader(
            DummyDatasetConfig(
                observation="obs",
                pipeline=[],
            )
        ),
        module,
        DummyLoss(["mse"]),
        DummyTrainer(),
    )

    assert cfg.module._module_config.model_config.NUM_OUTPUT_DIMS == 2


def test_prepare_directory_non_root_no_yaml_only_barriers(tmp_path):
    cfg = make_cfg(
        str(tmp_path / "non_root_no_yaml"),
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
    cfg.prepare_directory(non_root, yaml_config=None)

    assert non_root.barrier_calls == 2
    assert not Path(cfg.experiment_dir).exists()


def test_missing_train_loader_raises():
    with pytest.raises(ValueError, match="train_loader"):
        TrainConfig(
            "x",
            1,
            None,
            DummyModule(num_out=1),
            DummyLoss(["mse"]),
            DummyTrainer(),
        )


def test_missing_module_raises():
    with pytest.raises(ValueError, match="module"):
        TrainConfig(
            "x",
            1,
            DummyTrainLoader(
                DummyDatasetConfig(
                    observation="obs",
                    pipeline=ocean_pipeline(),
                )
            ),
            None,
            DummyLoss(["mse"]),
            DummyTrainer(),
        )


def test_missing_losspipeline_raises():
    with pytest.raises(ValueError, match="losspipeline"):
        TrainConfig(
            "x",
            1,
            DummyTrainLoader(
                DummyDatasetConfig(
                    observation="obs",
                    pipeline=ocean_pipeline(),
                )
            ),
            DummyModule(num_out=1),
            None,
            DummyTrainer(),
        )


def test_missing_trainer_raises():
    with pytest.raises(ValueError, match="trainer"):
        TrainConfig(
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
            None,
        )


def test_prepare_config_reads_yaml(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("a: 1\nb: test\n")

    from cccma_ppp.train.train_configs import prepare_config

    data = prepare_config(yaml_path)

    assert data["a"] == 1
    assert data["b"] == "test"


def test_prepare_runtime_variables(tmp_path):
    cfg = valid_directory_cfg(tmp_path / "runtime")

    cfg._prepare_runtime_variables()

    assert RuntimeContext.GLOBAL_EXP_DIR == str(cfg.experiment_dir)
    assert RuntimeContext.GLOBAL_CHECKPOINT_DIR == str(cfg.checkpoint_dir)
    assert RuntimeContext.GLOBAL_LOG_DIR == str(cfg.log_dir)
    assert RuntimeContext.GLOBAL_FIGURES_DIR == str(cfg.figures_dir)


def test_resume_dir_missing_raises(tmp_path):
    cfg = object.__new__(TrainConfig)

    with pytest.raises(ValueError, match="does not exist"):
        cfg.read_config_from_halted_experiment(
            tmp_path / "missing",
            tmp_path / "new_exp",
            5,
        )


def test_prepare_directory_resume_copytree(tmp_path, monkeypatch):
    resume = tmp_path / "resume"
    resume.mkdir()

    (resume / "config.yaml").write_text(
        """
experiment_dir: old
max_epochs: 1
train_loader: null
module: null
losspipeline: null
trainer: null
seed: 1
"""
    )

    cfg = object.__new__(TrainConfig)

    cfg.resume_dir = resume
    cfg.experiment_dir = tmp_path / "new_exp"
    cfg.copy_resume_dir_to_new_path = True

    loader = DummyTrainLoader(
        DummyDatasetConfig(
            observation="obs",
            pipeline=ocean_pipeline(),
        )
    )

    cfg.train_loader = loader

    called = {"copy": False}

    def fake_copytree(src, dst):
        called["copy"] = True

    monkeypatch.setattr("shutil.copytree", fake_copytree)

    cfg.prepare_directory(Root())

    assert called["copy"] is True


def test_prepare_directory_yaml_none_root(tmp_path):
    cfg = valid_directory_cfg(tmp_path / "exp")

    root = Root()

    cfg.prepare_directory(root, yaml_config=None)

    assert not (Path(cfg.experiment_dir) / "config.yaml").exists()


def test_build_trainer_logs_optimizer_name(capsys):
    cfg = build_train_config_for_builder()

    build_trainer(cfg, DistributedRoot(), logger=None)

    captured = capsys.readouterr()

    assert "adam" in captured.out.lower()


def test_build_trainer_module_sent_to_device():
    cfg = build_train_config_for_builder()

    build_trainer(cfg, DistributedRoot(), logger=None)

    assert cfg.trainer.module.device_used == torch.device("cpu")


def test_build_trainer_initializes_loss():
    cfg = build_train_config_for_builder()

    build_trainer(cfg, DistributedRoot(), logger=None)

    assert cfg.trainer.module.loss == "loss"


def test_build_trainer_non_root_logger_none_silent(capsys):
    cfg = build_train_config_for_builder()

    build_trainer(cfg, DistributedNonRoot(), logger=None)

    captured = capsys.readouterr()

    assert captured.out == ""


def test_sanitize_non_mlp_without_ocean_no_change():
    loader = DummyTrainLoader(
        DummyDatasetConfig(
            observation="obs",
            pipeline=[],
        )
    )

    module = DummyModule("other", num_out=2)

    loader._force_sanitize = True
    loader.sanitize(module)

    pipeline = loader.dataset_config.observation.preprocessing_pipeline.pipeline

    assert pipeline == []


def test_sanitize_mlp_existing_ocean_not_duplicated():
    loader = DummyTrainLoader(
        DummyDatasetConfig(
            observation="obs",
            pipeline=ocean_pipeline(),
        )
    )

    module = DummyModule(num_out=1)

    loader._force_sanitize = True
    loader.sanitize(module)

    pipeline = loader.dataset_config.observation.preprocessing_pipeline.pipeline

    oceans = [step for step in pipeline if isinstance(step[1], Oceannanremove)]

    assert len(oceans) == 1
