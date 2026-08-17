from types import SimpleNamespace

import pytest
import os
from pathlib import Path
import logging
import torch
from cccma_ppp.preprocessing.utils_preprocessing import Flattennanremove
from cccma_ppp.train.train_configs import (
    TrainConfig,
    _check_IO,
    build_trainer,
    RuntimeContext,
)


class DummyObservation:
    def __init__(self):
        self.preprocessing_pipeline = type(
            "P",
            (),
            {"pipeline": [("flatten", Flattennanremove())]},
        )()


class DummyDatasetConfig:
    def __init__(self):
        self.observation = None
        self.condition_method = "x"
        self.preprocessing_pipeline = type("P", (), {"pipeline": []})
        self.model = type(
            "M", (), {"preprocessing_pipeline": type("P", (), {"pipeline": []})}
        )


class DummyTrainLoader:
    def __init__(self):
        self.dataset_config = type("DS", (), {})()

        self.dataset_config.observation = DummyObservation()
        self.dataset_config.model = DummyObservation()
        self.dataset_config.condition_method = "x"

        self.input_var_metadata = {
            "NN_dims": [10, 10, 10, 10],
            "preprocessors": [],
        }

        self.target_var_metadata = {
            "NN_dims": [10],
            "preprocessors": [],
        }

    def setup_distributed(self, d):
        pass

    def build_train_loader(self, return_spatial_mask=False, **kwargs):
        class L:
            def __len__(self):
                return 2

            input_shape = (2,)
            target_shape = (2,)
            added_features_dim = None

        return L()

    def build_validation_loader(self, return_spatial_mask=False, **kwargs):
        return None

    def get_weights(self, w):
        return None


class DummyModuleSelector:
    EXPECTS_MASK = False
    type = "cvae"
    NUM_INPUT_DIMS = 4
    NUM_OUTPUT_DIMS = 1
    GENERATOR = None

    def build_module(self, **kwargs):
        class M:
            model_config = SimpleNamespace(GENERATOR=None)
            model = SimpleNamespace(
                generative_modeling=False, config=SimpleNamespace(NUM_OUTPUT_DIMS=1)
            )

            def __init__(self):
                self.model_config = SimpleNamespace(
                    GENERATOR=None,
                )
                self.model = SimpleNamespace(
                    generative_modeling=False,
                    config=SimpleNamespace(),
                )

            def to(self, device):
                self.device = device
                return self

            def init_loss_function(self, loss):
                self.loss = loss

        return M()


class DummyLossPipeline:
    loss_types = []

    def build(self, **kwargs):
        return "loss"


class DummyTrainer:
    gradient_accumulation_steps = 1

    def __init__(self):
        self.beta_finder = True

    def build(self, **kwargs):
        return "trainer"


class DummyOptimization:
    optimizer_type = "adam"

    def build(self, *args, **kwargs):
        return "opt"


class DummyDistributed:
    distributed = False
    device = "cpu"
    local_rank = 0

    def is_root(self):
        return True

    def barrier(self):
        pass


def make_valid_config_with(tmp_path):
    return TrainConfig(
        experiment_dir=tmp_path,
        max_epochs=1,
        train_loader=DummyTrainLoader(),
        module=DummyModuleSelector(),
        losspipeline=DummyLossPipeline(),
        trainer=DummyTrainer(),
        optimization=DummyOptimization(),
    )


@pytest.mark.pruned
def test_basic_init(tmp_path):
    cfg = make_valid_config_with(tmp_path)
    assert cfg.max_epochs == 1
    assert isinstance(cfg.experiment_dir, Path)


def test_max_epochs_none_becomes_inf(tmp_path):
    cfg = TrainConfig(
        tmp_path,
        None,
        DummyTrainLoader(),
        DummyModuleSelector(),
        DummyLossPipeline(),
        DummyTrainer(),
    )
    assert cfg.max_epochs == float("inf")


@pytest.mark.pruned
def test_missing_required_inputs_raises(tmp_path):
    with pytest.raises(ValueError):
        TrainConfig(tmp_path, 1, None, None, None, None)


@pytest.mark.pruned
def test_negative_epochs_assert(tmp_path):
    with pytest.raises(AssertionError):
        TrainConfig(
            tmp_path,
            -1,
            DummyTrainLoader(),
            DummyModuleSelector(),
            DummyLossPipeline(),
            DummyTrainer(),
        )


@pytest.mark.pruned
def test_deterministic_observation_required(tmp_path):
    loader = DummyTrainLoader()
    loader.dataset_config.observation = None

    module = DummyModuleSelector()
    module.type = "deterministic"

    with pytest.raises(ValueError):
        TrainConfig(
            tmp_path,
            1,
            loader,
            module,
            DummyLossPipeline(),
            DummyTrainer(),
        )


@pytest.mark.pruned
def test_generator_requires_crps(tmp_path):
    loader = DummyTrainLoader()

    module = DummyModuleSelector()
    module.GENERATOR = True

    loss = DummyLossPipeline()
    loss.loss_types = ["mse"]

    with pytest.raises(RuntimeError):
        TrainConfig(tmp_path, 1, loader, module, loss, DummyTrainer())


def test_set_random_seed(tmp_path):
    cfg = make_valid_config_with(tmp_path)
    cfg.seed = 42
    cfg.set_random_seed(0)


@pytest.mark.pruned
def test_prepare_directory_creates_dirs(tmp_path):
    cfg = make_valid_config_with(tmp_path)
    d = DummyDistributed()

    cfg.prepare_directory(d)

    assert os.path.exists(cfg.checkpoint_dir)


@pytest.mark.pruned
def test_prepare_directory_yaml_copy(tmp_path):
    cfg = make_valid_config_with(tmp_path)
    d = DummyDistributed()

    yaml_file = tmp_path / "cfg.yaml"
    yaml_file.write_text("x: 1")

    cfg.prepare_directory(d, yaml_config=yaml_file)

    assert (Path(cfg.experiment_dir) / "config.yaml").exists()


@pytest.mark.pruned
def test_resolve_resuming_same_path(tmp_path):
    cfg = object.__new__(TrainConfig)
    cfg.resume_dir = tmp_path
    cfg.experiment_dir = tmp_path
    cfg.max_epochs = 1

    def fake_read(**kwargs):
        return make_valid_config_with(tmp_path)

    cfg.read_config_from_halted_experiment = fake_read

    cfg._resolve_resuming()

    assert cfg.copy_resume_dir_to_new_path is False


@pytest.mark.pruned
def test_resolve_resuming_different_path(tmp_path):
    cfg = object.__new__(TrainConfig)
    cfg.resume_dir = tmp_path
    cfg.experiment_dir = tmp_path / "new"
    cfg.max_epochs = 1

    def fake_read(**kwargs):
        return make_valid_config_with(tmp_path)

    cfg.read_config_from_halted_experiment = fake_read

    cfg._resolve_resuming()

    assert cfg.copy_resume_dir_to_new_path is True


def test_build_trainer_basic(tmp_path):
    cfg = make_valid_config_with(tmp_path)
    d = DummyDistributed()

    trainer = build_trainer(cfg, d)

    assert trainer == "trainer"


@pytest.mark.pruned
def test_build_trainer_with_logger(tmp_path):
    cfg = make_valid_config_with(tmp_path)
    d = DummyDistributed()

    logger = logging.getLogger("x")

    build_trainer(cfg, d, logger=logger)


@pytest.mark.pruned
def test_prepare_directory_without_yaml(tmp_path):
    cfg = make_valid_config_with(tmp_path)
    d = DummyDistributed()

    cfg.prepare_directory(d, yaml_config=None)

    assert os.path.exists(cfg.experiment_dir)


@pytest.mark.pruned
def test_resolve_resuming_updates_config(tmp_path):
    cfg = object.__new__(TrainConfig)

    cfg.resume_dir = tmp_path
    cfg.experiment_dir = tmp_path
    cfg.max_epochs = 1

    halted = make_valid_config_with(tmp_path)

    def fake_read(**kwargs):
        return halted

    cfg.read_config_from_halted_experiment = fake_read

    cfg._resolve_resuming()

    assert cfg.max_epochs == halted.max_epochs


def test_set_random_seed_none(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    cfg.seed = None

    cfg.set_random_seed(0)


@pytest.mark.pruned
def test_build_trainer_non_distributed(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    d = DummyDistributed()
    d.distributed = False

    trainer = build_trainer(cfg, d)

    assert trainer == "trainer"


@pytest.mark.pruned
def test_build_trainer_with_validation_loader(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    class Loader(DummyTrainLoader):
        def build_validation_loader(self, return_spatial_mask=False):
            return [1]

    cfg.train_loader = Loader()

    d = DummyDistributed()

    trainer = build_trainer(cfg, d)

    assert trainer == "trainer"


@pytest.mark.pruned
def test_build_trainer_without_validation_loader(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    class Loader(DummyTrainLoader):
        def build_validation_loader(self, return_spatial_mask=False):
            return None

    cfg.train_loader = Loader()

    d = DummyDistributed()

    trainer = build_trainer(cfg, d)

    assert trainer == "trainer"


@pytest.mark.pruned
def test_build_trainer_beta_finder_false(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    class T(DummyTrainer):
        beta_finder = False

    cfg.trainer = T()

    d = DummyDistributed()

    trainer = build_trainer(cfg, d)

    assert trainer == "trainer"


@pytest.mark.pruned
def test_build_trainer_beta_finder_true(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    class T(DummyTrainer):
        beta_finder = True

    cfg.trainer = T()

    d = DummyDistributed()

    trainer = build_trainer(cfg, d)

    assert trainer == "trainer"


@pytest.mark.pruned
def test_generator_with_crps_valid(tmp_path):
    loader = DummyTrainLoader()

    module = DummyModuleSelector()
    module.GENERATOR = True

    loss = DummyLossPipeline()
    loss.loss_types = ["crps"]

    cfg = TrainConfig(
        tmp_path,
        1,
        loader,
        module,
        loss,
        DummyTrainer(),
    )

    assert cfg is not None


def test_deterministic_with_observation_valid(tmp_path):
    loader = DummyTrainLoader()
    loader.dataset_config.observation = DummyObservation()

    module = DummyModuleSelector()
    module.type = "deterministic"

    cfg = TrainConfig(
        tmp_path,
        1,
        loader,
        module,
        DummyLossPipeline(),
        DummyTrainer(),
    )

    assert cfg is not None


@pytest.mark.pruned
def test_prepare_directory_existing_dir(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    os.makedirs(cfg.experiment_dir, exist_ok=True)

    d = DummyDistributed()

    cfg.prepare_directory(d)

    assert os.path.exists(cfg.experiment_dir)


@pytest.mark.pruned
def test_train_loader_setup_distributed_called(tmp_path):
    called = {"x": False}

    class Loader(DummyTrainLoader):
        def setup_distributed(self, d):
            called["x"] = True

    cfg = make_valid_config_with(tmp_path)
    cfg.train_loader = Loader()

    d = DummyDistributed()

    build_trainer(cfg, d)

    assert called["x"] is True


@pytest.mark.pruned
def test_build_trainer_with_weights(tmp_path):
    class Loader(DummyTrainLoader):
        def get_weights(self, w):
            return [1.0]

    cfg = make_valid_config_with(tmp_path)
    cfg.train_loader = Loader()

    d = DummyDistributed()

    trainer = build_trainer(cfg, d)

    assert trainer == "trainer"


@pytest.mark.pruned
def test_build_trainer_without_weights(tmp_path):
    class Loader(DummyTrainLoader):
        def get_weights(self, w):
            return None

    cfg = make_valid_config_with(tmp_path)
    cfg.train_loader = Loader()

    d = DummyDistributed()

    trainer = build_trainer(cfg, d)

    assert trainer == "trainer"


@pytest.mark.pruned
def test_module_build_called(tmp_path):
    called = {"x": False}

    class Module(DummyModuleSelector):
        def build_module(self, **kwargs):
            called["x"] = True
            return super().build_module(**kwargs)

    cfg = make_valid_config_with(tmp_path)
    cfg.module = Module()

    d = DummyDistributed()

    build_trainer(cfg, d)

    assert called["x"] is True


@pytest.mark.pruned
def test_loss_build_called(tmp_path):
    called = {"x": False}

    class Loss(DummyLossPipeline):
        def build(self, **kwargs):
            called["x"] = True
            return "loss"

    cfg = make_valid_config_with(tmp_path)
    cfg.losspipeline = Loss()

    d = DummyDistributed()

    build_trainer(cfg, d)

    assert called["x"] is True


@pytest.mark.pruned
def test_optimizer_build_called(tmp_path):
    called = {"x": False}

    class Opt(DummyOptimization):
        def build(self, *args, **kwargs):
            called["x"] = True
            return "opt"

    cfg = make_valid_config_with(tmp_path)
    cfg.optimization = Opt()

    d = DummyDistributed()

    build_trainer(cfg, d)

    assert called["x"] is True


@pytest.mark.pruned
def test_trainer_build_called(tmp_path):
    called = {"x": False}

    class T(DummyTrainer):
        def build(self, **kwargs):
            called["x"] = True
            return "trainer"

    cfg = make_valid_config_with(tmp_path)
    cfg.trainer = T()

    d = DummyDistributed()

    build_trainer(cfg, d)

    assert called["x"] is True


@pytest.mark.pruned
def test_build_trainer_logger_none(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    d = DummyDistributed()

    trainer = build_trainer(cfg, d, logger=None)

    assert trainer == "trainer"


@pytest.mark.pruned
def test_build_trainer_device_cpu(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    d = DummyDistributed()
    d.device = "cpu"

    trainer = build_trainer(cfg, d)

    assert trainer == "trainer"


@pytest.mark.pruned
def test_build_trainer_local_rank_zero(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    d = DummyDistributed()
    d.local_rank = 0

    trainer = build_trainer(cfg, d)

    assert trainer == "trainer"


@pytest.mark.pruned
def test_prepare_directory_checkpoint_exists(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    d = DummyDistributed()

    cfg.prepare_directory(d)

    assert Path(cfg.checkpoint_dir).exists()


@pytest.mark.pruned
def test_prepare_directory_logs_exists(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    d = DummyDistributed()

    cfg.prepare_directory(d)

    assert Path(cfg.log_dir).exists()


@pytest.mark.pruned
def test_max_epochs_positive(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    assert cfg.max_epochs > 0


@pytest.mark.pruned
def test_train_loader_exists(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    assert cfg.train_loader is not None


@pytest.mark.pruned
def test_module_exists(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    assert cfg.module is not None


@pytest.mark.pruned
def test_loss_pipeline_exists(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    assert cfg.losspipeline is not None


@pytest.mark.pruned
def test_trainer_exists(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    assert cfg.trainer is not None


@pytest.mark.pruned
def test_optimization_exists(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    assert cfg.optimization is not None


@pytest.mark.pruned
def test_deterministic_beta_finder_warning(tmp_path):
    loader = DummyTrainLoader()
    loader.dataset_config.observation = DummyObservation()

    trainer = DummyTrainer()
    trainer.beta_finder = None

    module = DummyModuleSelector()
    module.type = "deterministic"

    with pytest.warns(UserWarning):
        TrainConfig(
            experiment_dir=tmp_path,
            max_epochs=1,
            train_loader=loader,
            module=module,
            losspipeline=DummyLossPipeline(),
            trainer=trainer,
        )


@pytest.mark.pruned
def test_non_mlp_with_flattener_raises(tmp_path):
    loader = DummyTrainLoader()

    module = DummyModuleSelector()
    module.NUM_OUTPUT_DIMS = 3

    with pytest.raises(RuntimeError):
        TrainConfig(
            experiment_dir=tmp_path,
            max_epochs=1,
            train_loader=loader,
            module=module,
            losspipeline=DummyLossPipeline(),
            trainer=DummyTrainer(),
        )


@pytest.mark.pruned
def test_non_mlp_without_flattener_valid(tmp_path):
    loader = DummyTrainLoader()

    loader.input_var_metadata["preprocessors"] = []
    loader.target_var_metadata["preprocessors"] = []
    loader.input_var_metadata["NN_dims"] = [10, 10]
    loader.target_var_metadata["NN_dims"] = [10, 10]

    module = DummyModuleSelector()
    module.NUM_INPUT_DIMS = 2
    module.NUM_OUTPUT_DIMS = 2

    cfg = TrainConfig(
        experiment_dir=tmp_path,
        max_epochs=1,
        train_loader=loader,
        module=module,
        losspipeline=DummyLossPipeline(),
        trainer=DummyTrainer(),
    )

    assert cfg is not None


@pytest.mark.pruned
def test_prepare_runtime_variables(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    cfg._prepare_runtime_variables()

    assert RuntimeContext.GLOBAL_EXP_DIR is not None


@pytest.mark.pruned
def test_checkpoint_dir_property(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    assert "checkpoints" in str(cfg.checkpoint_dir)


@pytest.mark.pruned
def test_log_dir_property(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    assert "logs" in str(cfg.log_dir)


@pytest.mark.pruned
def test_figures_dir_property(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    assert "figures" in str(cfg.figures_dir)


@pytest.mark.pruned
def test_prepare_directory_copy_resume_branch(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"

    src.mkdir()

    cfg = make_valid_config_with(dst)

    cfg.resume_dir = src
    cfg.copy_resume_dir_to_new_path = True

    d = DummyDistributed()

    cfg.prepare_directory(d)

    assert dst.exists()


@pytest.mark.pruned
def test_prepare_directory_without_copy_resume(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    cfg.copy_resume_dir_to_new_path = False

    d = DummyDistributed()

    cfg.prepare_directory(d)

    assert Path(cfg.log_dir).exists()


@pytest.mark.pruned
def test_prepare_directory_yaml_none(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    d = DummyDistributed()

    cfg.prepare_directory(d, yaml_config=None)

    assert Path(cfg.experiment_dir).exists()


@pytest.mark.pruned
def test_read_config_missing_resume_dir(tmp_path):
    cfg = object.__new__(TrainConfig)

    with pytest.raises(ValueError):
        cfg.read_config_from_halted_experiment(
            resume_dir=tmp_path / "missing",
            experiment_dir=tmp_path,
            max_epochs=1,
        )


@pytest.mark.pruned
def test_build_trainer_print_logger_branch(tmp_path, capsys):
    cfg = make_valid_config_with(tmp_path)

    d = DummyDistributed()

    build_trainer(cfg, d, logger=None)

    captured = capsys.readouterr()

    assert "creating data loaders" in captured.out.lower()


def test_build_trainer_logger_info_branch(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    d = DummyDistributed()

    class FakeLogger:
        def __init__(self):
            self.called = False

        def info(self, *args, **kwargs):
            self.called = True

    logger = FakeLogger()

    build_trainer(cfg, d, logger=logger)

    assert logger.called is True


@pytest.mark.pruned
def test_build_trainer_distributed_false(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    d = DummyDistributed()
    d.distributed = False

    trainer = build_trainer(cfg, d)

    assert trainer == "trainer"


@pytest.mark.pruned
def test_module_to_called(tmp_path):
    called = {"x": False}

    class M:
        model_config = SimpleNamespace(GENERATOR=None)
        model = type(
            "X",
            (),
            {"config": type("C", (), {"NUM_OUTPUT_DIMS": 1})()},
        )

        def to(self, d):
            called["x"] = True
            return self

        def init_loss_function(self, x):
            pass

    class Module(DummyModuleSelector):
        def build_module(self, **kwargs):
            return M()

    cfg = make_valid_config_with(tmp_path)
    cfg.module = Module()

    d = DummyDistributed()

    build_trainer(cfg, d)

    assert called["x"] is True


@pytest.mark.pruned
def test_module_init_loss_called(tmp_path):
    called = {"x": False}

    class M:
        model_config = SimpleNamespace(GENERATOR=None)
        model = type(
            "X",
            (),
            {"config": type("C", (), {"NUM_OUTPUT_DIMS": 1})()},
        )

        def to(self, d):
            return self

        def init_loss_function(self, x):
            called["x"] = True

    class Module(DummyModuleSelector):
        def build_module(self, **kwargs):
            return M()

    cfg = make_valid_config_with(tmp_path)
    cfg.module = Module()

    d = DummyDistributed()

    build_trainer(cfg, d)

    assert called["x"] is True


@pytest.mark.pruned
def test_build_trainer_num_output_dims_fallback(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    class M:
        model_config = SimpleNamespace(GENERATOR=None)
        model = type("X", (), {"config": type("C", (), {})()})

        def to(self, d):
            return self

        def init_loss_function(self, x):
            pass

    class Module(DummyModuleSelector):
        def build_module(self, **kwargs):
            return M()

    cfg.module = Module()

    d = DummyDistributed()

    trainer = build_trainer(cfg, d)

    assert trainer == "trainer"


@pytest.mark.pruned
def test_set_random_seed_executes(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    cfg.seed = 123

    cfg.set_random_seed(0)


@pytest.mark.pruned
def test_resolve_resuming_updates_resume_dir(tmp_path):
    cfg = object.__new__(TrainConfig)

    cfg.resume_dir = tmp_path
    cfg.experiment_dir = tmp_path
    cfg.max_epochs = 5

    resumed = make_valid_config_with(tmp_path)

    def fake_read(**kwargs):
        return resumed

    cfg.read_config_from_halted_experiment = fake_read

    cfg._resolve_resuming()

    assert cfg.resume_dir == tmp_path


@pytest.mark.pruned
def test_resolve_resuming_sets_experiment_dir_path(tmp_path):
    cfg = object.__new__(TrainConfig)

    cfg.resume_dir = tmp_path
    cfg.experiment_dir = tmp_path
    cfg.max_epochs = 5

    resumed = make_valid_config_with(tmp_path)

    def fake_read(**kwargs):
        return resumed

    cfg.read_config_from_halted_experiment = fake_read

    cfg._resolve_resuming()

    assert isinstance(cfg.experiment_dir, Path)


def test_missing_flattener_for_mlp(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    cfg.train_loader.dataset_config.observation.preprocessing_pipeline.pipeline = []
    cfg.train_loader.input_var_metadata["preprocessors"] = []
    cfg.train_loader.input_var_metadata["NN_dims"] = [10, 10, 10]

    with pytest.raises(RuntimeError):
        cfg.__post_init__()


@pytest.mark.pruned
def test_resolve_resuming_none_branch(tmp_path):
    cfg = object.__new__(TrainConfig)

    cfg.resume_dir = None
    cfg.experiment_dir = tmp_path
    cfg.max_epochs = 1

    cfg.train_loader = DummyTrainLoader()
    cfg.module = DummyModuleSelector()
    cfg.losspipeline = DummyLossPipeline()
    cfg.trainer = DummyTrainer()

    cfg._resolve_resuming()

    assert cfg.resume_dir is None


@pytest.mark.pruned
def test_build_trainer_distributed_true_monkeypatch(tmp_path, monkeypatch):
    cfg = make_valid_config_with(tmp_path)

    class FakeDDP:
        def __init__(self, module, **kwargs):
            self.model = module.model

        def init_loss_function(self, x):
            pass

    monkeypatch.setattr(
        torch.nn.parallel,
        "DistributedDataParallel",
        FakeDDP,
    )

    d = DummyDistributed()
    d.distributed = True

    trainer = build_trainer(cfg, d)

    assert trainer == "trainer"


@pytest.mark.pruned
def test_prepare_runtime_sets_checkpoint_dir(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    cfg._prepare_runtime_variables()

    assert RuntimeContext.GLOBAL_CHECKPOINT_DIR is not None


@pytest.mark.pruned
def test_prepare_runtime_sets_log_dir(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    cfg._prepare_runtime_variables()

    assert RuntimeContext.GLOBAL_LOG_DIR is not None


@pytest.mark.pruned
def test_prepare_runtime_sets_figures_dir(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    cfg._prepare_runtime_variables()

    assert RuntimeContext.GLOBAL_FIGURES_DIR is not None


@pytest.mark.pruned
def test_prepare_runtime_sets_input_metadata(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    cfg._prepare_runtime_variables()

    assert RuntimeContext.INPUT_VAR_METADATA == cfg.train_loader.input_var_metadata


@pytest.mark.pruned
def test_prepare_runtime_sets_target_metadata(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    cfg._prepare_runtime_variables()

    assert RuntimeContext.TARGET_VAR_METADATA == cfg.train_loader.target_var_metadata


def test_checkpoint_dir_exists_after_prepare(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    d = DummyDistributed()

    cfg.prepare_directory(d)

    assert os.path.isdir(cfg.checkpoint_dir)


@pytest.mark.pruned
def test_log_dir_exists_after_prepare(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    d = DummyDistributed()

    cfg.prepare_directory(d)

    assert os.path.isdir(cfg.log_dir)


@pytest.mark.pruned
def test_figures_dir_exists_after_prepare(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    d = DummyDistributed()

    cfg.prepare_directory(d)

    assert os.path.isdir(cfg.figures_dir)


@pytest.mark.pruned
def test_prepare_directory_root_false(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    class D(DummyDistributed):
        def is_root(self):
            return False

    d = D()

    cfg.prepare_directory(d)


@pytest.mark.pruned
def test_build_trainer_train_loader_len_used(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    d = DummyDistributed()

    trainer = build_trainer(cfg, d)

    assert trainer == "trainer"


@pytest.mark.pruned
def test_build_trainer_output_shape_exists(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    loader = cfg.train_loader.build_train_loader()

    assert loader.target_shape is not None


@pytest.mark.pruned
def test_build_trainer_input_shape_exists(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    loader = cfg.train_loader.build_train_loader()

    assert loader.input_shape is not None


@pytest.mark.pruned
def test_build_trainer_added_features_none(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    loader = cfg.train_loader.build_train_loader()

    assert loader.added_features_dim is None


@pytest.mark.pruned
def test_prepare_directory_with_existing_checkpoint_dir(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    d = DummyDistributed()

    cfg.prepare_directory(d)

    assert Path(cfg.checkpoint_dir).exists()


@pytest.mark.pruned
def test_prepare_directory_with_existing_logs_dir(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    os.makedirs(cfg.log_dir, exist_ok=True)

    d = DummyDistributed()

    cfg.prepare_directory(d)

    assert Path(cfg.log_dir).exists()


@pytest.mark.pruned
def test_prepare_directory_with_existing_figures_dir(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    os.makedirs(cfg.figures_dir, exist_ok=True)

    d = DummyDistributed()

    cfg.prepare_directory(d)

    assert Path(cfg.figures_dir).exists()


@pytest.mark.pruned
def test_prepare_directory_copy_resume_false_branch(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    cfg.copy_resume_dir_to_new_path = False

    d = DummyDistributed()

    cfg.prepare_directory(d)

    assert os.path.isdir(cfg.experiment_dir)


@pytest.mark.pruned
def test_prepare_directory_copy_resume_true_branch(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"

    src.mkdir()

    (src / "x.txt").write_text("hello")

    cfg = make_valid_config_with(dst)

    cfg.copy_resume_dir_to_new_path = True
    cfg.resume_dir = src

    d = DummyDistributed()

    cfg.prepare_directory(d)

    assert (dst / "x.txt").exists()


@pytest.mark.pruned
def test_prepare_directory_yaml_copy_again(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    yaml_path = tmp_path / "a.yaml"
    yaml_path.write_text("x: 1")

    d = DummyDistributed()

    cfg.prepare_directory(d, yaml_config=yaml_path)

    assert (Path(cfg.experiment_dir) / "config.yaml").exists()


@pytest.mark.pruned
def test_prepare_directory_non_root_yaml_branch(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    class D(DummyDistributed):
        def is_root(self):
            return False

    yaml_path = tmp_path / "a.yaml"
    yaml_path.write_text("x: 1")

    d = D()

    cfg.prepare_directory(d, yaml_config=yaml_path)


@pytest.mark.pruned
def test_checkpoint_dir_type(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    assert isinstance(cfg.checkpoint_dir, str)


@pytest.mark.pruned
def test_log_dir_type(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    assert isinstance(cfg.log_dir, str)


@pytest.mark.pruned
def test_figures_dir_type(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    assert isinstance(cfg.figures_dir, str)


@pytest.mark.pruned
def test_prepare_runtime_variables_checkpoint(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    cfg._prepare_runtime_variables()

    assert RuntimeContext.GLOBAL_CHECKPOINT_DIR.endswith("checkpoints")


@pytest.mark.pruned
def test_prepare_runtime_variables_logs(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    cfg._prepare_runtime_variables()

    assert RuntimeContext.GLOBAL_LOG_DIR.endswith("logs")


@pytest.mark.pruned
def test_prepare_runtime_variables_figures(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    cfg._prepare_runtime_variables()

    assert RuntimeContext.GLOBAL_FIGURES_DIR.endswith("figures")


@pytest.mark.pruned
def test_prepare_runtime_variables_input_metadata(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    cfg._prepare_runtime_variables()

    assert RuntimeContext.INPUT_VAR_METADATA == cfg.train_loader.input_var_metadata


@pytest.mark.pruned
def test_prepare_runtime_variables_target_metadata(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    cfg._prepare_runtime_variables()

    assert RuntimeContext.TARGET_VAR_METADATA == cfg.train_loader.target_var_metadata


@pytest.mark.pruned
def test_set_random_seed_with_none(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    cfg.seed = None

    cfg.set_random_seed(0)


@pytest.mark.pruned
def test_set_random_seed_with_integer(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    cfg.seed = 999

    cfg.set_random_seed(0)


@pytest.mark.pruned
def test_build_train_loader_len(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    loader = cfg.train_loader.build_train_loader()

    assert len(loader) == 2


@pytest.mark.pruned
def test_build_train_loader_input_shape(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    loader = cfg.train_loader.build_train_loader()

    assert loader.input_shape == (2,)


@pytest.mark.pruned
def test_build_train_loader_target_shape(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    loader = cfg.train_loader.build_train_loader()

    assert loader.target_shape == (2,)


@pytest.mark.pruned
def test_build_train_loader_added_features_none(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    loader = cfg.train_loader.build_train_loader()

    assert loader.added_features_dim is None


@pytest.mark.pruned
def test_build_trainer_returns_trainer_again(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    d = DummyDistributed()

    result = build_trainer(cfg, d)

    assert result == "trainer"


@pytest.mark.pruned
def test_build_trainer_logger_object(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    class L:
        called = False

        def info(self, *args, **kwargs):
            self.called = True

    logger = L()

    d = DummyDistributed()

    build_trainer(cfg, d, logger=logger)

    assert logger.called is True


@pytest.mark.pruned
def test_build_trainer_no_logger_stdout(tmp_path, capsys):
    cfg = make_valid_config_with(tmp_path)

    d = DummyDistributed()

    build_trainer(cfg, d)

    out = capsys.readouterr().out

    assert "creating data loaders" in out.lower()


@pytest.mark.pruned
def test_build_trainer_module_build(tmp_path):
    called = {"x": False}

    class Module(DummyModuleSelector):
        def build_module(self, **kwargs):
            called["x"] = True
            return super().build_module(**kwargs)

    cfg = make_valid_config_with(tmp_path)

    cfg.module = Module()

    d = DummyDistributed()

    build_trainer(cfg, d)

    assert called["x"]


@pytest.mark.pruned
def test_build_trainer_loss_build(tmp_path):
    called = {"x": False}

    class Loss(DummyLossPipeline):
        def build(self, **kwargs):
            called["x"] = True
            return "loss"

    cfg = make_valid_config_with(tmp_path)

    cfg.losspipeline = Loss()

    d = DummyDistributed()

    build_trainer(cfg, d)

    assert called["x"]


@pytest.mark.pruned
def test_build_trainer_optimizer_build(tmp_path):
    called = {"x": False}

    class Opt(DummyOptimization):
        def build(self, *args, **kwargs):
            called["x"] = True
            return "opt"

    cfg = make_valid_config_with(tmp_path)

    cfg.optimization = Opt()

    d = DummyDistributed()

    build_trainer(cfg, d)

    assert called["x"]


@pytest.mark.pruned
def test_build_trainer_trainer_build(tmp_path):
    called = {"x": False}

    class T(DummyTrainer):
        def build(self, **kwargs):
            called["x"] = True
            return "trainer"

    cfg = make_valid_config_with(tmp_path)

    cfg.trainer = T()

    d = DummyDistributed()

    build_trainer(cfg, d)

    assert called["x"]


@pytest.mark.pruned
def test_module_to_executes(tmp_path):
    called = {"x": False}

    class M:
        model_config = SimpleNamespace(GENERATOR=None)
        model = type(
            "X",
            (),
            {"config": type("C", (), {"NUM_OUTPUT_DIMS": 1})()},
        )

        def to(self, device):
            called["x"] = True
            return self

        def init_loss_function(self, x):
            pass

    class Module(DummyModuleSelector):
        def build_module(self, **kwargs):
            return M()

    cfg = make_valid_config_with(tmp_path)

    cfg.module = Module()

    d = DummyDistributed()

    build_trainer(cfg, d)

    assert called["x"]


@pytest.mark.pruned
def test_init_loss_function_executes(tmp_path):
    called = {"x": False}

    class M:
        model_config = SimpleNamespace(GENERATOR=None)
        model = type(
            "X",
            (),
            {"config": type("C", (), {"NUM_OUTPUT_DIMS": 1})()},
        )

        def to(self, device):
            return self

        def init_loss_function(self, x):
            called["x"] = True

    class Module(DummyModuleSelector):
        def build_module(self, **kwargs):
            return M()

    cfg = make_valid_config_with(tmp_path)

    cfg.module = Module()

    d = DummyDistributed()

    build_trainer(cfg, d)

    assert called["x"]


@pytest.mark.pruned
def test_build_trainer_distributed_false_again(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    d = DummyDistributed()
    d.distributed = False

    result = build_trainer(cfg, d)

    assert result == "trainer"


@pytest.mark.pruned
def test_build_trainer_validation_loader_none(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    class Loader(DummyTrainLoader):
        def build_validation_loader(self, return_spatial_mask=False):
            return None

    cfg.train_loader = Loader()

    d = DummyDistributed()

    result = build_trainer(cfg, d)

    assert result == "trainer"


@pytest.mark.pruned
def test_build_trainer_validation_loader_present(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    class Loader(DummyTrainLoader):
        def build_validation_loader(self, return_spatial_mask=False):
            return [1]

    cfg.train_loader = Loader()

    d = DummyDistributed()

    result = build_trainer(cfg, d)

    assert result == "trainer"


@pytest.mark.pruned
def test_prepare_directory_copytree_branch_real(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"

    src.mkdir()
    (src / "hello.txt").write_text("hello")

    cfg = make_valid_config_with(dst)

    cfg.copy_resume_dir_to_new_path = True
    cfg.resume_dir = src

    d = DummyDistributed()

    cfg.prepare_directory(d)

    assert (dst / "hello.txt").exists()


@pytest.mark.pruned
def test_non_mlp_without_flatten_branch(tmp_path):
    loader = DummyTrainLoader()

    loader.input_var_metadata["preprocessors"] = []
    loader.target_var_metadata["preprocessors"] = []
    loader.input_var_metadata["NN_dims"] = [10, 10]
    loader.target_var_metadata["NN_dims"] = [10, 10]

    module = DummyModuleSelector()
    module.NUM_INPUT_DIMS = 2
    module.NUM_OUTPUT_DIMS = 2

    cfg = TrainConfig(
        experiment_dir=tmp_path,
        max_epochs=1,
        train_loader=loader,
        module=module,
        losspipeline=DummyLossPipeline(),
        trainer=DummyTrainer(),
    )

    assert cfg is not None


def test_non_mlp_with_flatten_branch(tmp_path):
    loader = DummyTrainLoader()

    module = DummyModuleSelector()
    module.NUM_OUTPUT_DIMS = 2

    with pytest.raises(RuntimeError):
        TrainConfig(
            experiment_dir=tmp_path,
            max_epochs=1,
            train_loader=loader,
            module=module,
            losspipeline=DummyLossPipeline(),
            trainer=DummyTrainer(),
        )


@pytest.mark.pruned
def test_generator_false_branch(tmp_path):
    module = DummyModuleSelector()
    module.GENERATOR = None

    cfg = TrainConfig(
        experiment_dir=tmp_path,
        max_epochs=1,
        train_loader=DummyTrainLoader(),
        module=module,
        losspipeline=DummyLossPipeline(),
        trainer=DummyTrainer(),
    )

    assert cfg is not None


@pytest.mark.pruned
def test_build_trainer_distributed_real_completion(tmp_path, monkeypatch):
    cfg = make_valid_config_with(tmp_path)

    class FakeDDP:
        def __init__(self, module, **kwargs):
            self.model = module.model

        def init_loss_function(self, loss):
            pass

    monkeypatch.setattr(
        torch.nn.parallel,
        "DistributedDataParallel",
        FakeDDP,
    )

    d = DummyDistributed()
    d.distributed = True

    result = build_trainer(cfg, d)

    assert result == "trainer"


@pytest.mark.pruned
def test_prepare_directory_root_false_with_copy_resume(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"

    src.mkdir()

    cfg = make_valid_config_with(dst)

    cfg.resume_dir = src
    cfg.copy_resume_dir_to_new_path = True

    class D(DummyDistributed):
        def is_root(self):
            return False

    d = D()

    cfg.prepare_directory(d)

    assert not dst.exists()


@pytest.mark.pruned
def test_prepare_directory_root_true_without_copy_resume(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    cfg.copy_resume_dir_to_new_path = False

    d = DummyDistributed()

    cfg.prepare_directory(d)

    assert Path(cfg.figures_dir).exists()


@pytest.mark.pruned
def test_prepare_runtime_variables_all_fields(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    cfg._prepare_runtime_variables()

    assert RuntimeContext.GLOBAL_EXP_DIR
    assert RuntimeContext.GLOBAL_CHECKPOINT_DIR
    assert RuntimeContext.GLOBAL_LOG_DIR
    assert RuntimeContext.GLOBAL_FIGURES_DIR


def test_resolve_resuming_copy_flag_true(tmp_path):
    cfg = object.__new__(TrainConfig)

    cfg.resume_dir = tmp_path / "old"
    cfg.experiment_dir = tmp_path / "new"
    cfg.max_epochs = 1

    resumed = make_valid_config_with(tmp_path)

    def fake_read(**kwargs):
        return resumed

    cfg.read_config_from_halted_experiment = fake_read

    cfg._resolve_resuming()

    assert cfg.copy_resume_dir_to_new_path is True


def test_resolve_resuming_copy_flag_false(tmp_path):
    cfg = object.__new__(TrainConfig)

    cfg.resume_dir = tmp_path
    cfg.experiment_dir = tmp_path
    cfg.max_epochs = 1

    resumed = make_valid_config_with(tmp_path)

    def fake_read(**kwargs):
        return resumed

    cfg.read_config_from_halted_experiment = fake_read

    cfg._resolve_resuming()

    assert cfg.copy_resume_dir_to_new_path is False


@pytest.mark.pruned
def test_logger_not_called_when_not_root(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    class D(DummyDistributed):
        def is_root(self):
            return False

    class Logger:
        called = False

        def info(self, *args, **kwargs):
            self.called = True

    logger = Logger()

    d = D()

    build_trainer(cfg, d, logger=logger)

    assert logger.called is False


@pytest.mark.pruned
def test_prepare_directory_root_false_yaml_none(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    class D(DummyDistributed):
        def is_root(self):
            return False

    d = D()

    cfg.prepare_directory(d, yaml_config=None)


@pytest.mark.pruned
def test_prepare_directory_root_true_yaml_present(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("x: 1")

    d = DummyDistributed()

    cfg.prepare_directory(d, yaml_config=yaml_path)

    assert (Path(cfg.experiment_dir) / "config.yaml").exists()


@pytest.mark.pruned
def test_prepare_directory_root_false_yaml_present(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    class D(DummyDistributed):
        def is_root(self):
            return False

    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("x: 1")

    d = D()

    cfg.prepare_directory(d, yaml_config=yaml_path)

    assert not (Path(cfg.experiment_dir) / "config.yaml").exists()


def test_prepare_directory_copy_resume_root_false(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"

    src.mkdir()

    cfg = make_valid_config_with(dst)

    cfg.resume_dir = src
    cfg.copy_resume_dir_to_new_path = True

    class D(DummyDistributed):
        def is_root(self):
            return False

    d = D()

    cfg.prepare_directory(d)

    assert not dst.exists()


@pytest.mark.pruned
def test_prepare_directory_copy_resume_root_true(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"

    src.mkdir()
    (src / "a.txt").write_text("hello")

    cfg = make_valid_config_with(dst)

    cfg.resume_dir = src
    cfg.copy_resume_dir_to_new_path = True

    d = DummyDistributed()

    cfg.prepare_directory(d)

    assert (dst / "a.txt").exists()


@pytest.mark.pruned
def test_prepare_directory_no_copy_root_true(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    cfg.copy_resume_dir_to_new_path = False

    d = DummyDistributed()

    cfg.prepare_directory(d)

    assert Path(cfg.checkpoint_dir).exists()


@pytest.mark.pruned
def test_prepare_directory_no_copy_root_false(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    cfg.copy_resume_dir_to_new_path = False

    class D(DummyDistributed):
        def is_root(self):
            return False

    d = D()

    cfg.prepare_directory(d)


def test_build_trainer_logger_none_non_root(tmp_path, capsys):
    cfg = make_valid_config_with(tmp_path)

    class D(DummyDistributed):
        def is_root(self):
            return False

    d = D()

    build_trainer(cfg, d)

    out = capsys.readouterr().out

    assert out == ""


@pytest.mark.pruned
def test_build_trainer_logger_present_non_root(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    class D(DummyDistributed):
        def is_root(self):
            return False

    class Logger:
        called = False

        def info(self, *args, **kwargs):
            self.called = True

    logger = Logger()

    d = D()

    build_trainer(cfg, d, logger=logger)

    assert logger.called is False


@pytest.mark.pruned
def test_build_trainer_logger_present_root(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    class Logger:
        called = False

        def info(self, *args, **kwargs):
            self.called = True

    logger = Logger()

    d = DummyDistributed()

    build_trainer(cfg, d, logger=logger)

    assert logger.called is True


@pytest.mark.pruned
def test_build_trainer_distributed_true_full(tmp_path, monkeypatch):
    cfg = make_valid_config_with(tmp_path)

    class FakeDDP:
        def __init__(self, module, **kwargs):
            self.model = module.model

        def init_loss_function(self, x):
            pass

    monkeypatch.setattr(
        torch.nn.parallel,
        "DistributedDataParallel",
        FakeDDP,
    )

    d = DummyDistributed()
    d.distributed = True

    result = build_trainer(cfg, d)

    assert result == "trainer"


@pytest.mark.pruned
def test_build_trainer_distributed_false_full(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    d = DummyDistributed()
    d.distributed = False

    result = build_trainer(cfg, d)

    assert result == "trainer"


@pytest.mark.pruned
def test_build_trainer_num_output_dims_from_model(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    called = {"value": None}

    class Loss(DummyLossPipeline):
        def build(self, **kwargs):
            called["value"] = kwargs["num_output_dimensions"]
            return "loss"

    cfg.losspipeline = Loss()

    d = DummyDistributed()

    build_trainer(cfg, d)

    assert called["value"] == 1


@pytest.mark.pruned
def test_build_trainer_num_output_dims_fallback_len(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    class M:
        model_config = SimpleNamespace(
            GENERATOR=None,
        )
        model = SimpleNamespace(
            generative_modeling=False,
            config=SimpleNamespace(),
        )

        def to(self, device):
            return self

        def init_loss_function(self, loss):
            pass

    class Module(DummyModuleSelector):
        def build_module(self, **kwargs):
            return M()

    called = {"value": None}

    class Loss(DummyLossPipeline):
        def build(self, **kwargs):
            called["value"] = kwargs["num_output_dimensions"]
            return "loss"

    cfg.module = Module()
    cfg.losspipeline = Loss()

    d = DummyDistributed()

    build_trainer(cfg, d)

    assert called["value"] == 1


def test_read_config_missing_dir_branch(tmp_path):
    cfg = object.__new__(TrainConfig)

    with pytest.raises(ValueError):
        cfg.read_config_from_halted_experiment(
            resume_dir=tmp_path / "does_not_exist",
            experiment_dir=tmp_path,
            max_epochs=1,
        )


@pytest.mark.pruned
def test_resolve_resuming_same_dir_flag(tmp_path):
    cfg = object.__new__(TrainConfig)

    cfg.resume_dir = tmp_path
    cfg.experiment_dir = tmp_path
    cfg.max_epochs = 1

    resumed = make_valid_config_with(tmp_path)

    cfg.read_config_from_halted_experiment = lambda **kwargs: resumed

    cfg._resolve_resuming()

    assert cfg.copy_resume_dir_to_new_path is False


@pytest.mark.pruned
def test_resolve_resuming_different_dir_flag(tmp_path):
    cfg = object.__new__(TrainConfig)

    cfg.resume_dir = tmp_path / "old"
    cfg.experiment_dir = tmp_path / "new"
    cfg.max_epochs = 1

    resumed = make_valid_config_with(tmp_path)

    cfg.read_config_from_halted_experiment = lambda **kwargs: resumed

    cfg._resolve_resuming()

    assert cfg.copy_resume_dir_to_new_path is True


@pytest.mark.pruned
def test_resolve_resuming_restores_resume_dir(tmp_path):
    cfg = object.__new__(TrainConfig)

    cfg.resume_dir = tmp_path
    cfg.experiment_dir = tmp_path
    cfg.max_epochs = 1

    resumed = make_valid_config_with(tmp_path)

    cfg.read_config_from_halted_experiment = lambda **kwargs: resumed

    cfg._resolve_resuming()

    assert cfg.resume_dir == tmp_path


@pytest.mark.pruned
def test_resolve_resuming_restores_experiment_dir(tmp_path):
    cfg = object.__new__(TrainConfig)

    cfg.resume_dir = tmp_path
    cfg.experiment_dir = tmp_path
    cfg.max_epochs = 1

    resumed = make_valid_config_with(tmp_path)

    cfg.read_config_from_halted_experiment = lambda **kwargs: resumed

    cfg._resolve_resuming()

    assert isinstance(cfg.experiment_dir, Path)


@pytest.mark.pruned
def test_required_inputs_train_loader_missing(tmp_path):
    with pytest.raises(ValueError):
        TrainConfig(
            experiment_dir=tmp_path,
            max_epochs=1,
            train_loader=None,
            module=DummyModuleSelector(),
            losspipeline=DummyLossPipeline(),
            trainer=DummyTrainer(),
        )


@pytest.mark.pruned
def test_required_inputs_module_missing(tmp_path):
    with pytest.raises(ValueError):
        TrainConfig(
            experiment_dir=tmp_path,
            max_epochs=1,
            train_loader=DummyTrainLoader(),
            module=None,
            losspipeline=DummyLossPipeline(),
            trainer=DummyTrainer(),
        )


def test_required_inputs_loss_missing(tmp_path):
    with pytest.raises(ValueError):
        TrainConfig(
            experiment_dir=tmp_path,
            max_epochs=1,
            train_loader=DummyTrainLoader(),
            module=DummyModuleSelector(),
            losspipeline=None,
            trainer=DummyTrainer(),
        )


@pytest.mark.pruned
def test_required_inputs_trainer_missing(tmp_path):
    with pytest.raises(ValueError):
        TrainConfig(
            experiment_dir=tmp_path,
            max_epochs=1,
            train_loader=DummyTrainLoader(),
            module=DummyModuleSelector(),
            losspipeline=DummyLossPipeline(),
            trainer=None,
        )


@pytest.mark.pruned
def test_generator_true_with_crps_valid(tmp_path):
    module = DummyModuleSelector()
    module.GENERATOR = True

    loss = DummyLossPipeline()
    loss.loss_types = ["crps"]

    cfg = TrainConfig(
        experiment_dir=tmp_path,
        max_epochs=1,
        train_loader=DummyTrainLoader(),
        module=module,
        losspipeline=loss,
        trainer=DummyTrainer(),
    )

    assert cfg is not None


@pytest.mark.pruned
def test_generator_false_valid(tmp_path):
    module = DummyModuleSelector()
    module.GENERATOR = None

    cfg = TrainConfig(
        experiment_dir=tmp_path,
        max_epochs=1,
        train_loader=DummyTrainLoader(),
        module=module,
        losspipeline=DummyLossPipeline(),
        trainer=DummyTrainer(),
    )

    assert cfg is not None


@pytest.mark.pruned
def test_prepare_directory_root_false_copy_resume_true_yaml_none(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"

    src.mkdir()

    cfg = make_valid_config_with(dst)
    cfg.resume_dir = src
    cfg.copy_resume_dir_to_new_path = True

    class D(DummyDistributed):
        def is_root(self):
            return False

    d = D()

    cfg.prepare_directory(d, yaml_config=None)


@pytest.mark.pruned
def test_prepare_directory_root_true_copy_resume_false_yaml_none(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    cfg.copy_resume_dir_to_new_path = False

    d = DummyDistributed()

    cfg.prepare_directory(d, yaml_config=None)

    assert Path(cfg.figures_dir).exists()


@pytest.mark.pruned
def test_prepare_directory_root_true_copy_resume_false_yaml_present(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    cfg.copy_resume_dir_to_new_path = False

    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("x: 1")

    d = DummyDistributed()

    cfg.prepare_directory(d, yaml_config=yaml_path)

    assert (Path(cfg.experiment_dir) / "config.yaml").exists()


@pytest.mark.pruned
def test_prepare_directory_root_false_copy_resume_false_yaml_present(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    cfg.copy_resume_dir_to_new_path = False

    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("x: 1")

    class D(DummyDistributed):
        def is_root(self):
            return False

    d = D()

    cfg.prepare_directory(d, yaml_config=yaml_path)


def test_prepare_directory_root_true_copy_resume_true_yaml_present(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"

    src.mkdir()
    (src / "f.txt").write_text("hello")

    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("x: 1")

    cfg = make_valid_config_with(dst)

    cfg.resume_dir = src
    cfg.copy_resume_dir_to_new_path = True

    d = DummyDistributed()

    cfg.prepare_directory(d, yaml_config=yaml_path)

    assert (dst / "f.txt").exists()
    assert (dst / "config.yaml").exists()


@pytest.mark.pruned
def test_build_trainer_logger_none_non_root_branch(tmp_path, capsys):
    cfg = make_valid_config_with(tmp_path)

    class D(DummyDistributed):
        def is_root(self):
            return False

    d = D()

    build_trainer(cfg, d, logger=None)

    captured = capsys.readouterr()

    assert captured.out == ""


@pytest.mark.pruned
def test_build_trainer_logger_present_non_root_branch(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    class Logger:
        called = False

        def info(self, *args, **kwargs):
            self.called = True

    logger = Logger()

    class D(DummyDistributed):
        def is_root(self):
            return False

    d = D()

    build_trainer(cfg, d, logger=logger)

    assert logger.called is False


@pytest.mark.pruned
def test_build_trainer_logger_present_root_branch(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    class Logger:
        called = False

        def info(self, *args, **kwargs):
            self.called = True

    logger = Logger()

    d = DummyDistributed()

    build_trainer(cfg, d, logger=logger)

    assert logger.called is True


@pytest.mark.pruned
def test_num_output_dims_equals_one_branch(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    cfg.module.NUM_OUTPUT_DIMS = 1

    assert cfg.module.NUM_OUTPUT_DIMS == 1


@pytest.mark.pruned
def test_num_output_dims_not_one_branch(tmp_path):
    loader = DummyTrainLoader()

    loader.dataset_config.observation.preprocessing_pipeline.pipeline = []

    module = DummyModuleSelector()
    module.NUM_INPUT_DIMS = 2
    module.NUM_OUTPUT_DIMS = 2
    loader.input_var_metadata["NN_dims"] = [10, 10]
    loader.target_var_metadata["NN_dims"] = [10, 10]
    loader.input_var_metadata["preprocessors"] = []
    loader.target_var_metadata["preprocessors"] = []

    cfg = TrainConfig(
        experiment_dir=tmp_path,
        max_epochs=1,
        train_loader=loader,
        module=module,
        losspipeline=DummyLossPipeline(),
        trainer=DummyTrainer(),
    )

    assert cfg.module.NUM_OUTPUT_DIMS == 2


@pytest.mark.pruned
def test_generator_branch_false(tmp_path):
    module = DummyModuleSelector()

    module.GENERATOR = None

    cfg = TrainConfig(
        experiment_dir=tmp_path,
        max_epochs=1,
        train_loader=DummyTrainLoader(),
        module=module,
        losspipeline=DummyLossPipeline(),
        trainer=DummyTrainer(),
    )

    assert cfg is not None


def test_generator_branch_true_valid(tmp_path):
    module = DummyModuleSelector()

    module.GENERATOR = True

    loss = DummyLossPipeline()
    loss.loss_types = ["crps"]

    cfg = TrainConfig(
        experiment_dir=tmp_path,
        max_epochs=1,
        train_loader=DummyTrainLoader(),
        module=module,
        losspipeline=loss,
        trainer=DummyTrainer(),
    )

    assert cfg is not None


def test_cvae_requires_beta_finder(tmp_path):
    trainer = DummyTrainer()
    trainer.beta_finder = None

    module = DummyModuleSelector()
    module.type = "cVAE"

    with pytest.raises(ValueError):
        TrainConfig(
            experiment_dir=tmp_path,
            max_epochs=1,
            train_loader=DummyTrainLoader(),
            module=module,
            losspipeline=DummyLossPipeline(),
            trainer=trainer,
        )


def test_default_model_without_observation_raises(tmp_path):
    loader = DummyTrainLoader()
    loader.dataset_config.observation = None

    module = DummyModuleSelector()
    module.type = "default"

    with pytest.raises(ValueError):
        TrainConfig(
            experiment_dir=tmp_path,
            max_epochs=1,
            train_loader=loader,
            module=module,
            losspipeline=DummyLossPipeline(),
            trainer=DummyTrainer(),
        )


def test_default_model_beta_warning(tmp_path):
    loader = DummyTrainLoader()
    loader.dataset_config.observation = DummyObservation()

    trainer = DummyTrainer()
    trainer.beta_finder = None

    module = DummyModuleSelector()
    module.type = "default"

    with pytest.warns(UserWarning):
        TrainConfig(
            experiment_dir=tmp_path,
            max_epochs=1,
            train_loader=loader,
            module=module,
            losspipeline=DummyLossPipeline(),
            trainer=trainer,
        )


@pytest.mark.pruned
def test_mlp_with_flattener_valid(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    cfg.module.NUM_OUTPUT_DIMS = 1

    assert cfg is not None


@pytest.mark.pruned
def test_non_mlp_without_flattener_valid_again(tmp_path):
    loader = DummyTrainLoader()

    loader.input_var_metadata["preprocessors"] = []
    loader.target_var_metadata["preprocessors"] = []
    loader.input_var_metadata["NN_dims"] = [10, 10]
    loader.target_var_metadata["NN_dims"] = [10, 10]

    module = DummyModuleSelector()
    module.NUM_INPUT_DIMS = 2
    module.NUM_OUTPUT_DIMS = 2

    cfg = TrainConfig(
        experiment_dir=tmp_path,
        max_epochs=1,
        train_loader=loader,
        module=module,
        losspipeline=DummyLossPipeline(),
        trainer=DummyTrainer(),
    )

    assert cfg is not None


@pytest.mark.pruned
def test_generator_false_edge(tmp_path):
    module = DummyModuleSelector()
    module.GENERATOR = None

    cfg = TrainConfig(
        experiment_dir=tmp_path,
        max_epochs=1,
        train_loader=DummyTrainLoader(),
        module=module,
        losspipeline=DummyLossPipeline(),
        trainer=DummyTrainer(),
    )

    assert cfg is not None


@pytest.mark.pruned
def test_generator_true_edge(tmp_path):
    module = DummyModuleSelector()
    module.GENERATOR = True

    loss = DummyLossPipeline()
    loss.loss_types = ["crps"]

    cfg = TrainConfig(
        experiment_dir=tmp_path,
        max_epochs=1,
        train_loader=DummyTrainLoader(),
        module=module,
        losspipeline=loss,
        trainer=DummyTrainer(),
    )

    assert cfg is not None


def test_check_io_invalid_which():
    with pytest.raises(
        ValueError,
        match="only checks IO",
    ):
        _check_IO(
            {"NN_dims": [1], "preprocessors": []},
            1,
            "bad",
        )


@pytest.mark.pruned
def test_check_io_1d_with_single_dim_no_flattener():
    _check_IO(
        {
            "NN_dims": [10],
            "preprocessors": [],
        },
        1,
        "input",
    )


@pytest.mark.pruned
def test_check_io_multid_with_flattener_wrong_dims():
    with pytest.raises(
        RuntimeError,
        match="do not add Flattennanremove",
    ):
        _check_IO(
            {
                "NN_dims": [10, 10],
                "preprocessors": ["flattener"],
            },
            3,
            "input",
        )


def test_check_io_multid_with_flattener_correct_dims():
    with pytest.raises(
        RuntimeError,
        match="do not add Flattennanremove",
    ):
        _check_IO(
            {
                "NN_dims": [10, 10, 10],
                "preprocessors": ["flattener"],
            },
            3,
            "input",
        )


@pytest.mark.pruned
def test_prepare_runtime_variables_string_conversion(tmp_path):
    cfg = make_valid_config_with(tmp_path)

    cfg._prepare_runtime_variables()

    assert isinstance(
        RuntimeContext.GLOBAL_EXP_DIR,
        str,
    )

    assert isinstance(
        RuntimeContext.GLOBAL_CHECKPOINT_DIR,
        str,
    )


@pytest.mark.pruned
def test_build_trainer_passes_max_epochs(tmp_path):
    received = {"epochs": None}

    class T(DummyTrainer):
        def build(self, **kwargs):
            received["epochs"] = kwargs["max_epochs"]
            return "trainer"

    cfg = make_valid_config_with(tmp_path)
    cfg.trainer = T()

    build_trainer(
        cfg,
        DummyDistributed(),
    )

    assert received["epochs"] == 1


@pytest.mark.pruned
def test_build_trainer_passes_validation_loader(tmp_path):
    received = {"loader": None}

    class Loader(DummyTrainLoader):
        def build_validation_loader(self, return_spatial_mask=False):
            return ["val"]

    class T(DummyTrainer):
        def build(self, **kwargs):
            received["loader"] = kwargs["validation_data_loader"]
            return "trainer"

    cfg = make_valid_config_with(tmp_path)
    cfg.train_loader = Loader()
    cfg.trainer = T()

    build_trainer(
        cfg,
        DummyDistributed(),
    )

    assert received["loader"] == ["val"]


@pytest.mark.pruned
def test_build_trainer_passes_train_loader(tmp_path):
    received = {"loader": None}

    class T(DummyTrainer):
        def build(self, **kwargs):
            received["loader"] = kwargs["train_data_loader"]
            return "trainer"

    cfg = make_valid_config_with(tmp_path)
    cfg.trainer = T()

    build_trainer(
        cfg,
        DummyDistributed(),
    )

    assert received["loader"] is not None


@pytest.mark.pruned
def test_build_trainer_passes_optimizer(tmp_path):
    received = {"optimizer": None}

    class T(DummyTrainer):
        def build(self, **kwargs):
            received["optimizer"] = kwargs["optimization"]
            return "trainer"

    cfg = make_valid_config_with(tmp_path)
    cfg.trainer = T()

    build_trainer(
        cfg,
        DummyDistributed(),
    )

    assert received["optimizer"] == "opt"


@pytest.mark.pruned
def test_build_trainer_passes_module(tmp_path):
    received = {"module": None}

    class T(DummyTrainer):
        def build(self, **kwargs):
            received["module"] = kwargs["module"]
            return "trainer"

    cfg = make_valid_config_with(tmp_path)
    cfg.trainer = T()

    build_trainer(
        cfg,
        DummyDistributed(),
    )

    assert received["module"] is not None


@pytest.mark.pruned
def test_build_trainer_uses_len_output_shape_fallback(tmp_path):
    class Module(DummyModuleSelector):
        NUM_OUTPUT_DIMS = 0

    captured = {"value": None}

    class Loss(DummyLossPipeline):
        def build(self, **kwargs):
            captured["value"] = kwargs["num_output_dimensions"]
            return "loss"

    cfg = make_valid_config_with(tmp_path)
    cfg.module = Module()
    cfg.losspipeline = Loss()

    build_trainer(
        cfg,
        DummyDistributed(),
    )

    assert captured["value"] == 1


@pytest.mark.pruned
def test_resolve_resuming_restores_requested_resume_dir(tmp_path):
    cfg = object.__new__(TrainConfig)

    cfg.resume_dir = tmp_path / "resume"
    cfg.experiment_dir = tmp_path
    cfg.max_epochs = 1

    resumed = make_valid_config_with(tmp_path)

    cfg.read_config_from_halted_experiment = lambda **kwargs: resumed

    cfg._resolve_resuming()

    assert cfg.resume_dir == tmp_path / "resume"


@pytest.mark.pruned
def test_prepare_directory_barrier_called(tmp_path):
    calls = {"n": 0}

    class D(DummyDistributed):
        def barrier(self):
            calls["n"] += 1

    cfg = make_valid_config_with(tmp_path)

    cfg.prepare_directory(
        D(),
        yaml_config=None,
    )

    assert calls["n"] == 2


@pytest.mark.pruned
def test_prepare_directory_barrier_called_with_yaml(tmp_path):
    calls = {"n": 0}

    class D(DummyDistributed):
        def barrier(self):
            calls["n"] += 1

    cfg = make_valid_config_with(tmp_path)

    yaml_file = tmp_path / "cfg.yaml"
    yaml_file.write_text("x: 1")

    cfg.prepare_directory(
        D(),
        yaml_config=yaml_file,
    )

    assert calls["n"] == 2
