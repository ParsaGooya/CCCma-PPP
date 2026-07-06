from pathlib import Path
from types import SimpleNamespace
import torch
import pytest

from cccma_ppp.generic.runtime import RuntimeContext
from cccma_ppp.inference.inference_configs import (
    InferenceConfig,
    build_writer,
    prepare_config,
)


def test_check_ensemble_generation_deterministic():

    cfg = object.__new__(InferenceConfig)

    cfg.output_ensemble_size = 5
    cfg.output_sampler = None

    cfg.train_config = {"module": {"type": "deterministic"}}

    with pytest.raises(ValueError):
        cfg._check_esnsemble_generation()


@pytest.mark.pruned
def test_check_ensemble_generation_default():

    cfg = object.__new__(InferenceConfig)

    cfg.output_ensemble_size = 5
    cfg.output_sampler = None

    cfg.train_config = {"module": {"type": "default"}}

    with pytest.raises(ValueError):
        cfg._check_esnsemble_generation()


def test_check_ensemble_generation_allowed_sampler():

    cfg = object.__new__(InferenceConfig)

    cfg.output_ensemble_size = 5
    cfg.output_sampler = object()

    cfg.train_config = {"module": {"type": "deterministic"}}

    cfg._check_esnsemble_generation()


@pytest.mark.pruned
def test_check_ensemble_generation_single_member():

    cfg = object.__new__(InferenceConfig)

    cfg.output_ensemble_size = 1
    cfg.output_sampler = None

    cfg.train_config = {"module": {"type": "deterministic"}}

    cfg._check_esnsemble_generation()


def test_check_inference_dataset_metadata_mismatch():

    cfg = object.__new__(InferenceConfig)

    cfg.experiment_dir = Path("/tmp/exp")

    cfg.train_config = {"module": {"type": "deterministic"}}

    cfg.train_loader = SimpleNamespace(input_var_metadata=["train"])

    cfg.inference_loader = SimpleNamespace(
        dataset_config=SimpleNamespace(condition_method="static"),
        input_var_metadata=["infer"],
    )

    with pytest.raises(RuntimeError):
        cfg._check_inference_dataset()


def test_check_inference_dataset_success():

    cfg = object.__new__(InferenceConfig)

    cfg.train_config = {"module": {"type": "deterministic"}}

    cfg.train_loader = SimpleNamespace(input_var_metadata=["same"])

    cfg.inference_loader = SimpleNamespace(
        dataset_config=SimpleNamespace(condition_method="static"),
        input_var_metadata=["same"],
    )

    cfg._check_inference_dataset()


def test_resolve_dataset_config_calls_check(monkeypatch):

    called = {"flag": False}

    cfg = object.__new__(InferenceConfig)

    cfg.inference_loader = SimpleNamespace(dataset_config=object())

    monkeypatch.setattr(
        InferenceConfig,
        "_check_inference_dataset",
        lambda self: called.__setitem__("flag", True),
    )

    cfg._resolve_inference_dataset_config()

    assert called["flag"] is True


def test_output_preprocessor_dir_observation():

    cfg = object.__new__(InferenceConfig)

    cfg.experiment_dir = Path("/tmp/exp")

    pipeline = SimpleNamespace(name="obs")

    cfg.train_config = {
        "train_loader": {
            "dataset_config": {"observation": {"preprocessing_pipeline": pipeline}}
        }
    }

    expected = (
        Path("/tmp/exp")
        / "preprocessing_pipeline"
        / "obs_preprocessing_pipeline.joblib"
    )

    assert cfg.output_preprocessor_dir == expected


def test_output_preprocessor_dir_model():

    cfg = object.__new__(InferenceConfig)

    cfg.experiment_dir = Path("/tmp/exp")

    pipeline = SimpleNamespace(name="model")

    cfg.train_config = {
        "train_loader": {
            "dataset_config": {"model": {"preprocessing_pipeline": pipeline}}
        }
    }

    expected = (
        Path("/tmp/exp")
        / "preprocessing_pipeline"
        / "model_preprocessing_pipeline.joblib"
    )

    assert cfg.output_preprocessor_dir == expected


def test_prepare_directory_root(monkeypatch):

    cfg = object.__new__(InferenceConfig)

    cfg.output_dir = "/tmp/out"

    monkeypatch.setattr(
        InferenceConfig,
        "_prepare_runtime_variables",
        lambda self: None,
    )

    calls = {}

    monkeypatch.setattr(
        "os.makedirs",
        lambda path, exist_ok: calls.setdefault("mkdir", path),
    )

    distributed = SimpleNamespace(
        is_root=lambda: True,
        barrier=lambda: None,
    )

    cfg.prepare_directory(distributed)

    assert calls["mkdir"] == "/tmp/out"


def test_prepare_directory_non_root(monkeypatch):

    cfg = object.__new__(InferenceConfig)

    cfg.output_dir = "/tmp/out"

    monkeypatch.setattr(
        InferenceConfig,
        "_prepare_runtime_variables",
        lambda self: None,
    )

    called = {"mkdir": False}

    monkeypatch.setattr(
        "os.makedirs",
        lambda *args, **kwargs: called.__setitem__("mkdir", True),
    )

    distributed = SimpleNamespace(
        is_root=lambda: False,
        barrier=lambda: None,
    )

    cfg.prepare_directory(distributed)

    assert called["mkdir"] is False


@pytest.mark.pruned
def test_output_preprocessor_dir_model_branch():

    cfg = object.__new__(InferenceConfig)

    cfg.experiment_dir = Path("/tmp/exp")

    pipeline = SimpleNamespace(name="modelpipe")

    cfg.train_config = {
        "train_loader": {
            "dataset_config": {"model": {"preprocessing_pipeline": pipeline}}
        }
    }

    result = cfg.output_preprocessor_dir

    assert result.name == "modelpipe_preprocessing_pipeline.joblib"


def test_build_writer(monkeypatch):

    class FakeLoader:
        input_shape = (10,)
        target_shape = (5,)
        added_features_dim = 2

        def __len__(self):
            return 4

        def get_weights(self, _):
            return None

    class FakeTrainLoaderConfig:
        def setup_distributed(self, distributed):
            pass

        def build_train_loader(self):
            return FakeLoader()

        def build_validation_loader(self):
            return FakeLoader()

        def get_weights(self, weights):
            return None

    class FakeModule:
        model = SimpleNamespace()

        def to(self, device):
            return self

        def init_loss_function(self, loss):
            self.loss = loss

    class FakeModuleSelector:
        type = "deterministic"

        def build_module(self, **kwargs):
            return FakeModule()

    class FakeLoss:
        pass

    class FakeLossPipeline:
        def build(self, **kwargs):
            return FakeLoss()

    class FakeOptimizer:
        pass

    class FakeOptimization:
        optimizer_type = "adam"

        def build(self, module, batches, epochs):
            return FakeOptimizer()

    class FakeTrainer:
        pass

    class FakeTrainerConfig:
        def build(self, **kwargs):
            return FakeTrainer()

    distributed = SimpleNamespace(
        device=torch.device("cpu"),
        distributed=False,
        local_rank=0,
        is_root=lambda: True,
    )

    cfg = SimpleNamespace(
        train_loader=FakeTrainLoaderConfig(),
        weights=None,
        module=FakeModuleSelector(),
        losspipeline=FakeLossPipeline(),
        optimization=FakeOptimization(),
        trainer=FakeTrainerConfig(),
        max_epochs=10,
    )

    trainer = build_writer(
        cfg,
        distributed,
        logger=None,
    )

    assert isinstance(trainer, FakeTrainer)


def test_build_writer_logger_path(monkeypatch):

    logger_calls = []

    class FakeLogger:
        def info(self, msg, **kwargs):
            logger_calls.append(msg)

    class FakeLoader:
        input_shape = (1,)
        target_shape = (1,)
        added_features_dim = 0

        def __len__(self):
            return 2

        def get_weights(self, weights):
            return None

    class FakeTrainLoader:
        def setup_distributed(self, distributed):
            pass

        def build_train_loader(self):
            return FakeLoader()

        def build_validation_loader(self):
            return FakeLoader()

        def get_weights(self, weights):
            return None

    class FakeModule:
        model = type("M", (), {})()

        def to(self, device):
            return self

        def init_loss_function(self, loss):
            pass

    class FakeModuleCfg:
        type = "deterministic"

        def build_module(self, **kwargs):
            return FakeModule()

    class FakeLossPipeline:
        def build(self, **kwargs):
            return object()

    class FakeOptimization:
        optimizer_type = "adam"

        def build(self, *args):
            return object()

    class FakeTrainer:
        pass

    class FakeTrainerConfig:
        def build(self, **kwargs):
            return FakeTrainer()

    distributed = SimpleNamespace(
        device=torch.device("cpu"),
        distributed=False,
        local_rank=0,
        is_root=lambda: True,
    )

    cfg = SimpleNamespace(
        train_loader=FakeTrainLoader(),
        weights=None,
        module=FakeModuleCfg(),
        losspipeline=FakeLossPipeline(),
        optimization=FakeOptimization(),
        trainer=FakeTrainerConfig(),
        max_epochs=5,
    )

    build_writer(cfg, distributed, FakeLogger())

    assert logger_calls


def test_build_writer_distributed(monkeypatch):

    wrapped = {}

    class FakeDDP:
        def __init__(
            self,
            module,
            device_ids=None,
            output_device=None,
            find_unused_parameters=None,
        ):
            self._module = module
            wrapped["module"] = module

        def __getattr__(self, name):
            return getattr(self._module, name)

    monkeypatch.setattr(
        torch.nn.parallel,
        "DistributedDataParallel",
        FakeDDP,
    )

    class FakeLoader:
        input_shape = (1,)
        target_shape = (1,)
        added_features_dim = 0

        def __len__(self):
            return 1

        def get_weights(self, weights):
            return None

    class FakeTrainLoader:
        def setup_distributed(self, distributed):
            pass

        def build_train_loader(self):
            return FakeLoader()

        def build_validation_loader(self):
            return FakeLoader()

        def get_weights(self, weights):
            return None

    class FakeModule:
        model = type("M", (), {})()

        def to(self, device):
            return self

        def init_loss_function(self, loss):
            pass

    class FakeModuleCfg:
        type = "deterministic"

        def build_module(self, **kwargs):
            return FakeModule()

    class FakeLossPipeline:
        def build(self, **kwargs):
            return object()

    class FakeOptimization:
        optimizer_type = "adam"

        def build(self, *args):
            return object()

    class FakeTrainer:
        pass

    class FakeTrainerConfig:
        def build(self, **kwargs):
            return FakeTrainer()

    distributed = SimpleNamespace(
        device=torch.device("cpu"),
        distributed=True,
        local_rank=0,
        is_root=lambda: False,
    )

    cfg = SimpleNamespace(
        train_loader=FakeTrainLoader(),
        weights=None,
        module=FakeModuleCfg(),
        losspipeline=FakeLossPipeline(),
        optimization=FakeOptimization(),
        trainer=FakeTrainerConfig(),
        max_epochs=1,
    )

    build_writer(cfg, distributed)

    assert "module" in wrapped


def test_resolve_dataset_config_none_branch():

    cfg = object.__new__(InferenceConfig)

    cfg.train_loader = SimpleNamespace(dataset_config=object())

    cfg.inference_loader = SimpleNamespace(dataset_config=None)

    with pytest.raises(AttributeError):
        cfg._resolve_inference_dataset_config()


class FakeType(str):
    def lower(self):
        return "cvae"


def test_cvae_requires_condition_method():

    cfg = object.__new__(InferenceConfig)

    cfg.experiment_dir = Path("/tmp/exp")

    cfg.train_config = {"module": {"type": FakeType("ignored")}}

    cfg.train_loader = SimpleNamespace(input_var_metadata=["same"])

    cfg.inference_loader = SimpleNamespace(
        dataset_config=SimpleNamespace(
            condition_method=None,
        ),
        input_var_metadata=["same"],
    )

    with pytest.raises(ValueError):
        cfg._check_inference_dataset()


@pytest.mark.pruned
# Remove test due to no coverage
def test_save_dir_default():

    cfg = object.__new__(InferenceConfig)

    cfg.experiment_dir = "/tmp/exp"
    cfg.output_path = None

    assert cfg.save_dir == "/tmp/exp/inference"


@pytest.mark.pruned
# Remove test due to no coverage
def test_save_dir_explicit():

    cfg = object.__new__(InferenceConfig)

    cfg.output_path = "/custom/out"

    assert cfg.save_dir == "/custom/out"


@pytest.mark.pruned
# Remove test due to no coverage
def test_prepare_runtime_variables():

    cfg = object.__new__(InferenceConfig)

    cfg.experiment_dir = Path("/tmp/exp")
    cfg.output_dir = "/tmp/out"

    cfg.inference_loader = SimpleNamespace(
        input_var_metadata=["a"],
        target_var_metadata=["b"],
    )

    cfg._prepare_runtime_variables()

    assert RuntimeContext.GLOBAL_EXP_DIR == "/tmp/exp"
    assert RuntimeContext.GLOBAL_OUTPUT_DIR == "/tmp/out"
    assert RuntimeContext.INPUT_VAR_METADATA == ["a"]
    assert RuntimeContext.TARGET_VAR_METADATA == ["b"]


@pytest.mark.pruned
# Remove test due to no coverage
def test_prepare_config(tmp_path):

    path = tmp_path / "cfg.yaml"

    path.write_text("a: 1\nb: test\n")

    result = prepare_config(path)

    assert result["a"] == 1
    assert result["b"] == "test"


@pytest.mark.pruned
def test_resolve_dataset_config_none_branch_current_behavior():

    cfg = object.__new__(InferenceConfig)

    cfg.train_loader = SimpleNamespace(dataset_config=object())

    cfg.inference_loader = SimpleNamespace(dataset_config=None)

    with pytest.raises(AttributeError):
        cfg._resolve_inference_dataset_config()