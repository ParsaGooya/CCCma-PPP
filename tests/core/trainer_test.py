from types import SimpleNamespace
import logging
import dataclasses
import pytest
import torch
import torch.nn as nn

import cccma_ppp.core.trainer as trainer_mod
from cccma_ppp.core.trainer import Trainer, TrainerConfig


@dataclasses.dataclass
class DummyModuleConfig:
    dummy: bool = True


class DummyBatch:
    def __init__(self):
        self.moved_to = None
        self.x = torch.ones(2, 2)

    def to_device(self, device):
        self.moved_to = device
        return self


class DummyLoader:
    def __init__(self, n=2):
        self.n = n
        self.epoch = None
        self.batches = [DummyBatch() for _ in range(n)]
        self.input_shape = [2]
        self.target_shape = [1]
        self.added_features_dim = None

    def __len__(self):
        return self.n

    def __iter__(self):
        return iter(self.batches)

    def set_epoch(self, epoch):
        self.epoch = epoch


class DummyModule(nn.Module):
    def __init__(self, built=True):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.built = built
        self.config = DummyModuleConfig()
        self.input_shape = [2]
        self.output_shape = [1]
        self.train_called = False
        self.eval_called = False
        self.loss_calls = []

    def _get_device(self):
        return next(self.parameters()).device

    def train(self, mode=True):
        self.train_called = True
        return super().train(mode)

    def eval(self):
        self.eval_called = True
        return super().eval()

    def _compute_loss(self, data, beta=None):
        self.loss_calls.append({"data": data, "beta": beta})
        out = self.linear(data.x).sum()
        loss = out * 0 + torch.tensor(1.0, requires_grad=True)
        if beta is not None:
            return loss, {"total_loss": 1.0, "beta": float(beta)}
        return loss, {"total_loss": 1.0}

    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=strict)


class DummyCVAE(DummyModule):
    pass


class DummyOptimizer:
    learning_rate = 0.001

    def __init__(self, module):
        self.optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        self.zero_grad_calls = []
        self.scheduler_steps = 0
        self.loaded_state = None

        self.lr_scheduler = SimpleNamespace(num_steps=0)
        self.learning_rate = 0.0

    def zero_grad(self, set_to_none=True, **kwargs):
        self.zero_grad_calls.append(set_to_none)
        self.optimizer.zero_grad(set_to_none=set_to_none, **kwargs)

    def scheduler_step(self):
        self.scheduler_steps += 1

    def state_dict(self):
        return {
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": None,
        }

    def load_state_dict(self, state_dict):
        self.loaded_state = state_dict
        self.optimizer.load_state_dict(state_dict["optimizer"])


class DummyDistributed:
    def __init__(self, distributed=False, root=True, device=None):
        self.device = device or torch.device("cpu")
        self.rank = 0 if root else 1
        self.local_rank = 0
        self.world_size = 1
        self.distributed = distributed
        self.barrier_calls = 0
        self.broadcast_calls = 0

    def is_root(self):
        return self.rank == 0

    def barrier(self):
        self.barrier_calls += 1

    def broadcast(self, tensor, src=0):
        self.broadcast_calls += 1
        return tensor


class DummyLogger:
    def __init__(self):
        self.records = []

    def log(self, level, msg, *args):
        self.records.append((level, msg, args))


class DummyBetaFinder:
    def __init__(self):
        self.built_with = None
        self.calls = []

    def build(self, num_batches):
        self.built_with = num_batches

    def __call__(self, step):
        self.calls.append(step)
        return 0.5


class FakeAggregator:
    plot_calls = []

    def __init__(self, distributed, name=""):
        self.distributed = distributed
        self.name = name
        self.records = []
        self.epochs = []
        self.loaded_state = None
        self.remove_second_last_called = False

    def record(self, loss_dict, current_lr=None, kwargs=None):
        self.records.append(loss_dict)

    def _dist_compute(self):
        if self.records:
            return self.records[-1]
        return {"total_loss": 1.0}

    def record_epoch(self, logs, time_elapsed=None, index=None):
        self.epochs.append({"logs": logs, "time_elapsed": time_elapsed, "index": index})

    def state_dict(self):
        return {"records": self.records, "epochs": self.epochs, "name": self.name}

    def load_state_dict(self, state):
        self.loaded_state = state

    def remove_second_last_epoch(self):
        self.remove_second_last_called = True
        if len(self.epochs) >= 2:
            self.epochs.pop(-2)

    @staticmethod
    def plot(aggregators, plot_dir):
        FakeAggregator.plot_calls.append((aggregators, plot_dir))


@pytest.fixture(autouse=True)
def patch_aggregator(monkeypatch):
    FakeAggregator.plot_calls.clear()
    monkeypatch.setattr(trainer_mod, "MetricsAggregator", FakeAggregator)


@pytest.fixture
def env_dirs(tmp_path, monkeypatch):
    ckpt = tmp_path / "checkpoints"
    figs = tmp_path / "figures"

    monkeypatch.setenv("GLOBAL_CHECKPOINT_DIR", str(ckpt))
    monkeypatch.setenv("GLOBAL_FIGURES_DIR", str(figs))

    monkeypatch.setattr(trainer_mod.RuntimeContext, "GLOBAL_CHECKPOINT_DIR", str(ckpt))
    monkeypatch.setattr(trainer_mod.RuntimeContext, "GLOBAL_FIGURES_DIR", str(figs))
    monkeypatch.setattr(trainer_mod.RuntimeContext, "INPUT_VAR_METADATA", {})
    monkeypatch.setattr(trainer_mod.RuntimeContext, "TARGET_VAR_METADATA", {})

    return ckpt, figs


def make_trainer(validation=True, mixed_precision=False, grad_clip=None):
    module = DummyModule(built=True)
    optimizer = DummyOptimizer(module)
    train_loader = DummyLoader(n=2)
    validation_loader = DummyLoader(n=1) if validation else None

    config = TrainerConfig(
        mixed_precision=mixed_precision,
        grad_clip=grad_clip,
        gradient_accumulation_steps=1,
    )

    trainer = config.build(
        train_data_loader=train_loader,
        validation_data_loader=validation_loader,
        optimization=optimizer,
        module=module,
        max_epochs=2,
    )

    return trainer, module, optimizer, train_loader, validation_loader


@pytest.mark.pruned
def test_trainer_config_defaults():
    cfg = TrainerConfig()

    assert cfg.beta_finder is None
    assert cfg.earlystoppingbuffer == float("inf")
    assert cfg.gradient_accumulation_steps == 1
    assert cfg.mixed_precision is True
    assert cfg.grad_clip is None


@pytest.mark.pruned
def test_trainer_config_invalid_grad_clip():
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        TrainerConfig(grad_clip=0)


@pytest.mark.pruned
def test_trainer_config_build_sets_batch_counts():
    module = DummyModule(built=True)
    optimizer = DummyOptimizer(module)
    train_loader = DummyLoader(n=3)
    validation_loader = DummyLoader(n=2)

    cfg = TrainerConfig(mixed_precision=False)
    trainer = cfg.build(
        train_data_loader=train_loader,
        validation_data_loader=validation_loader,
        optimization=optimizer,
        module=module,
        max_epochs=5,
    )

    assert isinstance(trainer, Trainer)
    assert cfg.num_train_batches == 3
    assert cfg.num_validation_batches == 2


@pytest.mark.pruned
def test_trainer_config_build_without_validation_loader():
    module = DummyModule(built=True)
    optimizer = DummyOptimizer(module)
    train_loader = DummyLoader(n=3)

    cfg = TrainerConfig(mixed_precision=False)
    trainer = cfg.build(
        train_data_loader=train_loader,
        validation_data_loader=None,
        optimization=optimizer,
        module=module,
        max_epochs=5,
    )

    assert isinstance(trainer, Trainer)
    assert cfg.num_train_batches == 3
    assert not hasattr(cfg, "num_validation_batches")


@pytest.mark.pruned
def test_trainer_config_build_accepts_unbuilt_module_current_behavior():
    module = DummyModule(built=False)
    optimizer = DummyOptimizer(module)

    cfg = TrainerConfig(mixed_precision=False)

    trainer = cfg.build(
        train_data_loader=DummyLoader(),
        validation_data_loader=None,
        optimization=optimizer,
        module=module,
        max_epochs=1,
    )

    assert isinstance(trainer, Trainer)


def test_trainer_config_cvae_requires_beta_finder(monkeypatch):
    monkeypatch.setattr(trainer_mod, "cVAE", DummyCVAE)

    module = DummyCVAE(built=True)
    optimizer = DummyOptimizer(module)
    cfg = TrainerConfig(beta_finder=None, mixed_precision=False)

    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        cfg.build(
            train_data_loader=DummyLoader(),
            validation_data_loader=None,
            optimization=optimizer,
            module=module,
            max_epochs=1,
        )


def test_trainer_config_cvae_builds_beta_finder(monkeypatch):
    monkeypatch.setattr(trainer_mod, "cVAE", DummyCVAE)

    beta = DummyBetaFinder()
    module = DummyCVAE(built=True)
    optimizer = DummyOptimizer(module)
    train_loader = DummyLoader(n=4)

    cfg = TrainerConfig(beta_finder=beta, mixed_precision=False)
    trainer = cfg.build(
        train_data_loader=train_loader,
        validation_data_loader=None,
        optimization=optimizer,
        module=module,
        max_epochs=1,
    )

    assert isinstance(trainer, Trainer)
    assert beta.built_with == 4


@pytest.mark.pruned
def test_setup_distributed_basic(env_dirs):
    trainer, module, _, _, _ = make_trainer(validation=True)
    dist = DummyDistributed(distributed=False)
    logger = DummyLogger()

    trainer.setup_distributed(
        distributed=dist,
        logger=logger,
        log_every_n_epochs=2,
        save_checkpoint=True,
    )

    assert trainer._setup is True
    assert trainer.device == torch.device("cpu")
    assert isinstance(trainer.train_aggregator, FakeAggregator)
    assert isinstance(trainer.validation_aggregator, FakeAggregator)
    assert trainer.log_every_n_epochs == 2
    assert trainer.save_checkpoint is True
    assert env_dirs[0].exists()
    assert env_dirs[1].exists()
    assert any("Trainer setup complete" in rec[1] for rec in logger.records)


@pytest.mark.pruned
def test_setup_distributed_without_validation(env_dirs):
    trainer, _, _, _, _ = make_trainer(validation=False)
    dist = DummyDistributed(distributed=False)

    trainer.setup_distributed(dist, logger=DummyLogger())

    assert trainer.validation_aggregator is None


@pytest.mark.pruned
def test_setup_distributed_logger_none_prints(env_dirs, capsys):
    trainer, _, _, _, _ = make_trainer(validation=False)

    trainer.setup_distributed(DummyDistributed(), logger=None)

    captured = capsys.readouterr()
    assert "Logger is None" in captured.out


@pytest.mark.pruned
def test_setup_distributed_save_checkpoint_false_logs_warning(env_dirs):
    trainer, _, _, _, _ = make_trainer(validation=False)
    logger = DummyLogger()

    trainer.setup_distributed(
        DummyDistributed(),
        logger=logger,
        save_checkpoint=False,
    )

    assert any("no checkpoints" in rec[1] for rec in logger.records)


def test_setup_distributed_device_mismatch_raises(env_dirs):
    trainer, module, _, _, _ = make_trainer(validation=False)

    module._get_device = lambda: torch.device("meta")

    with pytest.raises(RuntimeError):
        trainer.setup_distributed(
            DummyDistributed(device=torch.device("cpu")), DummyLogger()
        )


@pytest.mark.pruned
def test_setup_distributed_calls_barrier_when_distributed(env_dirs, monkeypatch):
    class FakeDDP:
        def __init__(
            self,
            module,
            device_ids=None,
            output_device=None,
            find_unused_parameters=None,
        ):
            self.module = module

        def __getattr__(self, name):
            return getattr(self.module, name)

    monkeypatch.setattr(
        torch.nn.parallel,
        "DistributedDataParallel",
        FakeDDP,
    )

    trainer, _, _, _, _ = make_trainer(validation=False)
    dist = DummyDistributed(distributed=True)

    trainer.setup_distributed(dist, logger=DummyLogger())

    assert dist.barrier_calls >= 1


@pytest.mark.pruned
def test_log_root_uses_logger(env_dirs):
    trainer, _, _, _, _ = make_trainer(validation=False)
    logger = DummyLogger()
    trainer.setup_distributed(DummyDistributed(root=True), logger)

    trainer.log_root(logging.INFO, "hello")

    assert any(rec[1] == "hello" for rec in logger.records)


@pytest.mark.pruned
def test_log_root_prints_when_logger_none(env_dirs, capsys):
    trainer, _, _, _, _ = make_trainer(validation=False)
    trainer.setup_distributed(DummyDistributed(root=True), None)

    trainer.log_root(logging.INFO, "hello print")

    assert "hello print" in capsys.readouterr().out


@pytest.mark.pruned
def test_log_root_noop_when_not_root(env_dirs):
    trainer, _, _, _, _ = make_trainer(validation=False)
    logger = DummyLogger()
    trainer.setup_distributed(DummyDistributed(root=False), logger)

    trainer.log_root(logging.INFO, "hidden")

    assert not any(rec[1] == "hidden" for rec in logger.records)


@pytest.mark.pruned
def test_is_improved_first_validation_is_true():
    trainer, _, _, _, _ = make_trainer(validation=True)

    assert trainer._is_improved(10.0) is True


def test_is_improved_tensor_input():
    trainer, _, _, _, _ = make_trainer(validation=True)
    trainer._best_validation_loss = 10.0

    assert trainer._is_improved(torch.tensor(5.0)) is True


@pytest.mark.pruned
def test_is_improved_requires_minimum_percentage():
    trainer, _, _, _, _ = make_trainer(validation=True)
    trainer._best_validation_loss = 100.0
    trainer.config.minimum_validation_improvement_percentage = 0.02

    assert trainer._is_improved(99.0) is False
    assert trainer._is_improved(97.0) is True


@pytest.mark.pruned
def test_should_stop_early_no_validation():
    trainer, _, _, _, _ = make_trainer(validation=False)
    trainer.earlystopping_counter = 999

    assert trainer._should_stop_early() is False


def test_should_stop_early_none_buffer():
    trainer, _, _, _, _ = make_trainer(validation=True)
    trainer.config.earlystoppingbuffer = None
    trainer.earlystopping_counter = 999

    assert trainer._should_stop_early() is False


@pytest.mark.pruned
def test_should_stop_early_inf_buffer():
    trainer, _, _, _, _ = make_trainer(validation=True)
    trainer.config.earlystoppingbuffer = float("inf")
    trainer.earlystopping_counter = 999

    assert trainer._should_stop_early() is False


@pytest.mark.pruned
def test_should_stop_early_true():
    trainer, _, _, _, _ = make_trainer(validation=True)
    trainer.config.earlystoppingbuffer = 2
    trainer.earlystopping_counter = 2

    assert trainer._should_stop_early() is True


def test_train_requires_setup():
    trainer, _, _, _, _ = make_trainer(validation=False)

    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        trainer.train()


@pytest.mark.pruned
def test_train_on_batch_basic(env_dirs):
    trainer, module, optimizer, _, _ = make_trainer(validation=False)
    trainer.setup_distributed(DummyDistributed(), DummyLogger())

    batch = DummyBatch()
    logs, _ = trainer._train_on_batch(batch, accumulation_size=1)

    assert batch.moved_to == torch.device("cpu")
    assert logs["total_loss"] == 1.0
    assert trainer.batch_step == 1
    assert trainer.global_step == 1
    assert optimizer.scheduler_steps == 1


def test_train_on_batch_with_beta(env_dirs):
    beta = DummyBetaFinder()
    module = DummyCVAE(built=True)
    optimizer = DummyOptimizer(module)

    config = TrainerConfig(beta_finder=beta, mixed_precision=False)
    trainer = Trainer(
        config=config,
        train_data_loader=DummyLoader(),
        validation_data_loader=None,
        module=module,
        optimizer=optimizer,
        max_epochs=1,
    )
    trainer.setup_distributed(DummyDistributed(), DummyLogger())

    logs, _ = trainer._train_on_batch(DummyBatch(), accumulation_size=1)

    assert logs["beta"] == 0.5
    assert beta.calls == [0]


@pytest.mark.pruned
def test_train_on_batch_gradient_accumulation_delays_optimizer(env_dirs):
    module = DummyModule()
    optimizer = DummyOptimizer(module)

    config = TrainerConfig(
        mixed_precision=False,
        gradient_accumulation_steps=2,
    )

    trainer = Trainer(
        config=config,
        train_data_loader=DummyLoader(),
        validation_data_loader=None,
        module=module,
        optimizer=optimizer,
        max_epochs=1,
    )
    trainer.setup_distributed(DummyDistributed(), DummyLogger())

    trainer._train_on_batch(DummyBatch(), accumulation_size=1)
    assert trainer.global_step == 0

    trainer._train_on_batch(DummyBatch(), accumulation_size=1)
    assert trainer.global_step == 1


@pytest.mark.pruned
def test_train_on_epoch(env_dirs):
    trainer, module, _, train_loader, _ = make_trainer(validation=False)
    trainer.setup_distributed(DummyDistributed(), DummyLogger())

    elapsed = trainer._train_on_epoch()

    assert elapsed >= 0
    assert train_loader.epoch == 0
    assert module.train_called is True
    assert trainer._epochs_trained == 1
    assert len(trainer.train_aggregator.records) == len(train_loader)


def test_validate_on_epoch_requires_loader(env_dirs):
    trainer, _, _, _, _ = make_trainer(validation=False)
    trainer.setup_distributed(DummyDistributed(), DummyLogger())

    with pytest.raises(RuntimeError):
        trainer._validate_on_epoch()


@pytest.mark.pruned
def test_validate_on_batch_basic(env_dirs):
    trainer, module, _, _, _ = make_trainer(validation=True)
    trainer.setup_distributed(DummyDistributed(), DummyLogger())

    batch = DummyBatch()
    logs = trainer._validate_on_batch(batch)

    assert batch.moved_to == torch.device("cpu")
    assert logs["total_loss"] == 1.0


@pytest.mark.pruned
def test_validate_on_epoch(env_dirs):
    trainer, module, _, _, validation_loader = make_trainer(validation=True)
    trainer.setup_distributed(DummyDistributed(), DummyLogger())

    trainer._validate_on_epoch()

    assert module.eval_called is True
    assert len(trainer.validation_aggregator.records) == len(validation_loader)


def test_optimizer_step_with_grad_clip(env_dirs):
    trainer, module, optimizer, _, _ = make_trainer(validation=False, grad_clip=1.0)
    trainer.setup_distributed(DummyDistributed(), DummyLogger())

    loss = module.linear(torch.ones(1, 2)).sum()
    loss.backward()

    trainer._optimizer_step()

    assert trainer.global_step == 1
    assert optimizer.scheduler_steps == 1


@pytest.mark.pruned
def test_clear_memory_cpu(monkeypatch):
    called = {"gc": False}

    monkeypatch.setattr(
        trainer_mod.gc,
        "collect",
        lambda: called.__setitem__("gc", True),
    )
    monkeypatch.setattr(
        trainer_mod.torch.cuda,
        "is_available",
        lambda: False,
    )

    trainer_mod.clear_memory()

    assert called["gc"] is True


@pytest.mark.pruned
def test_save_checkpoint_without_validation(env_dirs):
    trainer, _, _, _, _ = make_trainer(validation=False)
    trainer.setup_distributed(DummyDistributed(), DummyLogger())

    trainer._save_checkpoint(
        name="best",
        train_logs={"total_loss": 1.0},
        validation_logs=None,
    )

    path = env_dirs[0] / "best.pt"
    assert path.exists()


@pytest.mark.pruned
def test_save_checkpoint_with_validation(env_dirs):
    trainer, _, _, _, _ = make_trainer(validation=True)
    trainer.setup_distributed(DummyDistributed(), DummyLogger())

    trainer._save_checkpoint(
        name="best",
        train_logs={"total_loss": 1.0},
        validation_logs={"total_loss": 0.5},
    )

    path = env_dirs[0] / "best.pt"
    assert path.exists()


@pytest.mark.pruned
def test_save_checkpoint_distributed_barrier(env_dirs, monkeypatch):

    class FakeDDP:
        def __init__(
            self,
            module,
            device_ids=None,
            output_device=None,
            find_unused_parameters=None,
        ):
            self.module = module

        def __getattr__(self, name):
            return getattr(self.module, name)

    monkeypatch.setattr(
        torch.nn.parallel,
        "DistributedDataParallel",
        FakeDDP,
    )
    trainer, _, _, _, _ = make_trainer(validation=False)
    dist = DummyDistributed(distributed=True)
    trainer.setup_distributed(dist, DummyLogger())

    before = dist.barrier_calls
    trainer._save_checkpoint(
        name="best",
        train_logs={"total_loss": 1.0},
        validation_logs=None,
    )

    assert dist.barrier_calls == before


def test_load_checkpoint_missing_file(env_dirs):
    trainer, _, _, _, _ = make_trainer(validation=False)
    trainer.setup_distributed(DummyDistributed(), DummyLogger())

    with pytest.raises(FileNotFoundError):
        trainer._load_checkpoint(env_dirs[0] / "missing.pt")


@pytest.mark.pruned
def test_load_checkpoint_success(env_dirs):
    trainer, _, _, _, _ = make_trainer(validation=True)
    trainer.setup_distributed(DummyDistributed(), DummyLogger())

    trainer._save_checkpoint(
        name="best",
        train_logs={"total_loss": 1.0},
        validation_logs={"total_loss": 0.5},
    )

    checkpoint = trainer._load_checkpoint(env_dirs[0] / "best.pt")

    assert checkpoint["epoch"] == trainer._epochs_trained
    assert trainer.train_aggregator.loaded_state is not None
    assert trainer.validation_aggregator.loaded_state is not None


def test_load_checkpoint_default_path(env_dirs):
    trainer, _, _, _, _ = make_trainer(validation=False)
    trainer.setup_distributed(DummyDistributed(), DummyLogger())

    trainer._save_checkpoint(
        name="best",
        train_logs={"total_loss": 1.0},
        validation_logs=None,
    )

    checkpoint = trainer._load_checkpoint()

    assert "module" in checkpoint


def test_load_checkpoint_without_scaler_or_histories(env_dirs):
    trainer, _, _, _, _ = make_trainer(validation=False)
    trainer.setup_distributed(DummyDistributed(), DummyLogger())

    path = env_dirs[0] / "minimal.pt"
    torch.save(
        {
            "module": trainer.raw_module.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "epoch": 3,
            "global_step": 4,
            "batch_step": 5,
        },
        path,
    )

    trainer._load_checkpoint(path)

    assert trainer._epochs_trained == 3
    assert trainer._start_epoch == 3
    assert trainer.global_step == 4
    assert trainer.batch_step == 5


@pytest.mark.pruned
def test_log_epoch_with_validation(env_dirs):
    trainer, _, _, _, _ = make_trainer(validation=True)
    logger = DummyLogger()
    trainer.setup_distributed(DummyDistributed(), logger)

    trainer.start_time_train = 0
    trainer._epochs_trained = 1
    trainer._log_epoch(
        train_logs={"total_loss": 1.0},
        validation_logs={"total_loss": 0.5},
    )

    assert any("validation loss" in rec[1] for rec in logger.records)


@pytest.mark.pruned
def test_log_epoch_without_validation(env_dirs):
    trainer, _, _, _, _ = make_trainer(validation=False)
    logger = DummyLogger()
    trainer.setup_distributed(DummyDistributed(), logger)

    trainer.start_time_train = 0
    trainer._epochs_trained = 1
    trainer._log_epoch(train_logs={"total_loss": 1.0}, validation_logs=None)

    assert any("train loss" in rec[1] for rec in logger.records)


@pytest.mark.pruned
def test_train_loop_without_validation(env_dirs):
    trainer, _, _, _, _ = make_trainer(validation=False)
    trainer.max_epochs = 1
    trainer.setup_distributed(DummyDistributed(), DummyLogger())

    trainer.train()

    assert trainer._epochs_trained == 1
    assert (env_dirs[0] / "best.pt").exists()
    assert FakeAggregator.plot_calls


@pytest.mark.pruned
def test_train_loop_with_validation_improvement(env_dirs):
    trainer, _, _, _, _ = make_trainer(validation=True)
    trainer.max_epochs = 1
    trainer.setup_distributed(DummyDistributed(), DummyLogger())

    trainer.train()

    assert trainer._best_validation_loss == 1.0
    assert trainer.earlystopping_counter == 0
    assert (env_dirs[0] / "best.pt").exists()


@pytest.mark.pruned
def test_train_loop_with_validation_no_improvement(env_dirs):
    trainer, _, _, _, _ = make_trainer(validation=True)
    trainer.max_epochs = 1
    trainer._best_validation_loss = 0.1
    trainer.setup_distributed(DummyDistributed(), DummyLogger())

    trainer.train()

    assert trainer.earlystopping_counter == 1


@pytest.mark.pruned
def test_train_loop_early_stopping(env_dirs):
    trainer, _, _, _, _ = make_trainer(validation=True)
    trainer.max_epochs = 5
    trainer._best_validation_loss = 0.1
    trainer.config.earlystoppingbuffer = 1
    trainer.setup_distributed(DummyDistributed(), DummyLogger())

    trainer.train()

    assert trainer.earlystopping_counter >= 1
    assert trainer._epochs_trained < 5


def test_train_loop_distributed_stop_broadcast(env_dirs, monkeypatch):
    class FakeDDP:
        def __init__(
            self,
            module,
            device_ids=None,
            output_device=None,
            find_unused_parameters=None,
        ):
            self.module = module

        def __getattr__(self, name):
            return getattr(self.module, name)

    monkeypatch.setattr(
        torch.nn.parallel,
        "DistributedDataParallel",
        FakeDDP,
    )
    trainer, _, _, _, _ = make_trainer(validation=True)
    trainer.max_epochs = 1
    trainer.config.earlystoppingbuffer = 0
    trainer.setup_distributed(DummyDistributed(distributed=True), DummyLogger())

    trainer.train()

    assert trainer.distributed.broadcast_calls >= 1


def test_train_loop_final_leftover_optimizer_step_no_validation(env_dirs):
    module = DummyModule()
    optimizer = DummyOptimizer(module)
    config = TrainerConfig(
        mixed_precision=False,
        gradient_accumulation_steps=3,
    )

    trainer = Trainer(
        config=config,
        train_data_loader=DummyLoader(n=1),
        validation_data_loader=None,
        module=module,
        optimizer=optimizer,
        max_epochs=1,
    )

    trainer.setup_distributed(DummyDistributed(), DummyLogger())
    trainer.train()

    assert trainer.global_step >= 1


def test_train_loop_final_leftover_validation_improved(env_dirs):
    module = DummyModule()
    optimizer = DummyOptimizer(module)
    config = TrainerConfig(
        mixed_precision=False,
        gradient_accumulation_steps=3,
    )

    trainer = Trainer(
        config=config,
        train_data_loader=DummyLoader(n=1),
        validation_data_loader=DummyLoader(n=1),
        module=module,
        optimizer=optimizer,
        max_epochs=1,
    )

    trainer.setup_distributed(DummyDistributed(), DummyLogger())
    trainer.train()

    assert trainer.validation_aggregator.epochs


@pytest.mark.pruned
def test_raw_module_ddp_branch(monkeypatch):
    class FakeDDP:
        def __init__(self, module):
            self.module = module

    real_module = DummyModule()
    trainer = Trainer(
        config=TrainerConfig(mixed_precision=False),
        train_data_loader=DummyLoader(),
        validation_data_loader=None,
        module=FakeDDP(real_module),
        optimizer=DummyOptimizer(real_module),
        max_epochs=1,
    )

    monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", FakeDDP)

    assert trainer.raw_module is real_module


def test_setup_distributed_resume_branch(monkeypatch, env_dirs):
    trainer, _, _, _, _ = make_trainer(validation=False)

    called = {"loaded": False}

    def fake_load_checkpoint(path):
        called["loaded"] = True

    monkeypatch.setattr(trainer_mod.os.path, "isfile", lambda path: True)
    monkeypatch.setattr(trainer, "_load_checkpoint", fake_load_checkpoint)

    trainer.setup_distributed(DummyDistributed(), DummyLogger())

    assert called["loaded"] is True
    assert trainer._setup is True


def test_optimizer_step_amp_skipped_does_not_increment(monkeypatch, env_dirs):
    trainer, module, optimizer, _, _ = make_trainer(validation=False)
    trainer.setup_distributed(DummyDistributed(), DummyLogger())

    class FakeScaler:
        def __init__(self):
            self.calls = 0

        def unscale_(self, optimizer):
            pass

        def get_scale(self):
            self.calls += 1

            return 2.0 if self.calls == 1 else 1.0

        def step(self, optimizer):
            pass

        def update(self):
            pass

        def state_dict(self):
            return {}

        def load_state_dict(self, state):
            pass

        def scale(self, loss):
            return loss

        def is_enabled(self):
            return False

    trainer.scaler = FakeScaler()

    trainer._optimizer_step()

    assert trainer.global_step == 0
    assert optimizer.scheduler_steps == 0


@pytest.mark.pruned
def test_load_checkpoint_without_train_history_key(env_dirs):
    trainer, _, _, _, _ = make_trainer(validation=False)
    trainer.setup_distributed(DummyDistributed(), DummyLogger())

    path = env_dirs[0] / "no_history.pt"
    torch.save(
        {
            "module": trainer.raw_module.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "epoch": 2,
            "global_step": 3,
            "batch_step": 4,
        },
        path,
    )

    trainer._load_checkpoint(path)

    assert trainer._epochs_trained == 2
    assert trainer.global_step == 3
    assert trainer.batch_step == 4


@pytest.mark.pruned
def test_train_loop_without_validation_no_checkpoint_when_disabled(env_dirs):
    trainer, _, _, _, _ = make_trainer(validation=False)
    trainer.max_epochs = 1
    trainer.setup_distributed(
        DummyDistributed(),
        DummyLogger(),
        save_checkpoint=False,
    )

    trainer.train()

    assert trainer._epochs_trained == 1
    assert not (env_dirs[0] / "best.pt").exists()


@pytest.mark.pruned
def test_train_loop_validation_no_improvement_checkpoint_disabled(env_dirs):
    trainer, _, _, _, _ = make_trainer(validation=True)
    trainer.max_epochs = 1
    trainer._best_validation_loss = 0.1

    trainer.setup_distributed(
        DummyDistributed(),
        DummyLogger(),
        save_checkpoint=False,
    )

    trainer.train()

    assert trainer.earlystopping_counter == 1
    assert not (env_dirs[0] / "best.pt").exists()


def test_train_loop_leftover_skipped_due_to_early_stop(env_dirs):
    module = DummyModule()
    optimizer = DummyOptimizer(module)

    config = TrainerConfig(
        mixed_precision=False,
        gradient_accumulation_steps=3,
        earlystoppingbuffer=0,
    )

    trainer = Trainer(
        config=config,
        train_data_loader=DummyLoader(n=1),
        validation_data_loader=DummyLoader(n=1),
        module=module,
        optimizer=optimizer,
        max_epochs=1,
    )

    trainer._best_validation_loss = 0.1
    trainer.setup_distributed(
        DummyDistributed(),
        DummyLogger(),
        save_checkpoint=False,
    )

    trainer.train()

    assert trainer.batch_step % trainer.config.gradient_accumulation_steps != 0
    assert trainer._should_stop_early() is True


@pytest.mark.pruned
def test_log_root_not_root_logger_none(env_dirs, capsys):
    trainer, _, _, _, _ = make_trainer(validation=False)
    trainer.setup_distributed(DummyDistributed(root=False), logger=None)

    trainer.log_root(logging.INFO, "should not print")

    assert "should not print" not in capsys.readouterr().out


@pytest.mark.pruned
def test_setup_distributed_non_root_does_not_create_checkpoint_dir(env_dirs):
    trainer, _, _, _, _ = make_trainer(validation=False)

    ckpt_dir, fig_dir = env_dirs
    trainer.setup_distributed(
        DummyDistributed(root=False),
        DummyLogger(),
        save_checkpoint=True,
    )

    assert trainer.is_on_root is False


@pytest.mark.pruned
def test_setup_distributed_existing_dirs_no_crash(env_dirs):
    trainer, _, _, _, _ = make_trainer(validation=False)

    ckpt_dir, fig_dir = env_dirs
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    trainer.setup_distributed(
        DummyDistributed(),
        DummyLogger(),
    )

    assert ckpt_dir.exists()
    assert fig_dir.exists()


@pytest.mark.pruned
def test_setup_distributed_non_root_logger_none(env_dirs, capsys):
    trainer, _, _, _, _ = make_trainer(validation=False)

    trainer.setup_distributed(
        DummyDistributed(root=False),
        logger=None,
    )

    captured = capsys.readouterr()

    assert "Logger is None" in captured.out


@pytest.mark.pruned
def test_is_improved_equal_loss_not_improved():
    trainer, _, _, _, _ = make_trainer(validation=True)

    trainer._best_validation_loss = 1.0

    assert trainer._is_improved(1.0) is False


@pytest.mark.pruned
def test_should_stop_early_false_when_counter_below_buffer():
    trainer, _, _, _, _ = make_trainer(validation=True)

    trainer.config.earlystoppingbuffer = 3
    trainer.earlystopping_counter = 2

    assert trainer._should_stop_early() is False


@pytest.mark.pruned
def test_train_on_batch_batch_step_increments_only(env_dirs):
    trainer, _, _, _, _ = make_trainer(validation=False)

    trainer.setup_distributed(DummyDistributed(), DummyLogger())

    trainer._train_on_batch(DummyBatch(), accumulation_size=1)

    assert trainer.batch_step == 1


@pytest.mark.pruned
def test_validate_on_batch_keeps_module_eval(env_dirs):
    trainer, module, _, _, _ = make_trainer(validation=True)

    trainer.setup_distributed(DummyDistributed(), DummyLogger())

    trainer._validate_on_epoch()

    assert module.training is False


@pytest.mark.pruned
def test_save_checkpoint_non_root_missing_dir_raises(env_dirs):
    trainer, _, _, _, _ = make_trainer(validation=False)

    trainer.setup_distributed(
        DummyDistributed(root=False),
        DummyLogger(),
    )

    with pytest.raises(RuntimeError):
        trainer._save_checkpoint(
            name="best",
            train_logs={"total_loss": 1.0},
            validation_logs=None,
        )


@pytest.mark.pruned
def test_load_checkpoint_restores_scaler_state(env_dirs):
    trainer, _, _, _, _ = make_trainer(validation=False)

    trainer.setup_distributed(DummyDistributed(), DummyLogger())

    path = env_dirs[0] / "scaler.pt"

    checkpoint = {
        "module": trainer.raw_module.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "epoch": 1,
        "global_step": 2,
        "batch_step": 3,
        "scaler": trainer.scaler.state_dict(),
    }

    torch.save(checkpoint, path)

    trainer._load_checkpoint(path)

    assert trainer.global_step == 2
    assert trainer.batch_step == 3


@pytest.mark.pruned
def test_optimizer_step_without_grad_clip(env_dirs):
    trainer, module, optimizer, _, _ = make_trainer(
        validation=False,
        grad_clip=None,
    )

    trainer.setup_distributed(DummyDistributed(), DummyLogger())

    loss = module.linear(torch.ones(1, 2)).sum()
    loss.backward()

    trainer._optimizer_step()

    assert trainer.global_step == 1


@pytest.mark.pruned
def test_train_on_epoch_sets_module_train_mode(env_dirs):
    trainer, module, _, _, _ = make_trainer(validation=False)

    trainer.setup_distributed(DummyDistributed(), DummyLogger())

    trainer._train_on_epoch()

    assert module.training is True


@pytest.mark.pruned
def test_load_checkpoint_validation_aggregator_none(env_dirs):
    trainer, _, _, _, _ = make_trainer(validation=False)

    trainer.setup_distributed(DummyDistributed(), DummyLogger())

    trainer._save_checkpoint(
        name="basic",
        train_logs={"total_loss": 1.0},
        validation_logs=None,
    )

    checkpoint = trainer._load_checkpoint(env_dirs[0] / "basic.pt")

    assert checkpoint is not None
    assert trainer.validation_aggregator is None


@pytest.mark.pruned
def test_setup_distributed_stores_distributed_reference(env_dirs):
    trainer, _, _, _, _ = make_trainer(validation=False)

    dist = DummyDistributed()

    trainer.setup_distributed(dist, DummyLogger())

    assert trainer.distributed is dist


@pytest.mark.pruned
def test_log_root_accepts_format_args(env_dirs):
    trainer, _, _, _, _ = make_trainer(validation=False)

    logger = DummyLogger()

    trainer.setup_distributed(
        DummyDistributed(root=True),
        logger,
    )

    trainer.log_root(logging.INFO, "value %s", 123)

    assert any(rec[2] == (123,) for rec in logger.records)


@pytest.mark.pruned
def test_setup_distributed_save_checkpoint_false_no_warning_non_root(env_dirs):
    trainer, _, _, _, _ = make_trainer(validation=False)

    logger = DummyLogger()

    trainer.setup_distributed(
        DummyDistributed(root=False),
        logger,
        save_checkpoint=False,
    )

    assert not any("no checkpoints" in rec[1] for rec in logger.records)


@pytest.mark.pruned
def test_setup_distributed_distributed_existing_dirs(env_dirs, monkeypatch):
    class FakeDDP:
        def __init__(
            self,
            module,
            device_ids=None,
            output_device=None,
            find_unused_parameters=None,
        ):
            self.module = module

        def __getattr__(self, name):
            return getattr(self.module, name)

    monkeypatch.setattr(
        torch.nn.parallel,
        "DistributedDataParallel",
        FakeDDP,
    )
    trainer, _, _, _, _ = make_trainer(validation=False)

    dist = DummyDistributed(distributed=True)

    trainer.setup_distributed(
        dist,
        DummyLogger(),
    )

    assert dist.barrier_calls >= 1


@pytest.mark.pruned
def test_optimizer_step_amp_enabled_branch(env_dirs):
    trainer, module, optimizer, _, _ = make_trainer(validation=False)

    trainer.setup_distributed(
        DummyDistributed(),
        DummyLogger(),
    )

    class FakeScaler:
        def scale(self, loss):
            return loss

        def step(self, optimizer):
            pass

        def update(self):
            pass

        def unscale_(self, optimizer):
            pass

        def get_scale(self):
            return 1.0

        def is_enabled(self):
            return True

        def state_dict(self):
            return {}

    trainer.scaler = FakeScaler()

    loss = module.linear(torch.ones(1, 2)).sum()
    loss.backward()

    trainer._optimizer_step()

    assert trainer.global_step == 1


def test_log_epoch_root_without_logger(env_dirs, capsys):
    trainer, _, _, _, _ = make_trainer(validation=False)

    trainer.setup_distributed(
        DummyDistributed(root=True),
        logger=None,
    )

    trainer.start_time_train = 0
    trainer._epochs_trained = 1

    trainer._log_epoch(
        train_logs={"total_loss": 1.0},
        validation_logs=None,
    )

    assert "train loss" in capsys.readouterr().out.lower()


def test_train_loop_no_validation_no_plot_when_non_root(env_dirs):
    trainer, _, _, _, _ = make_trainer(validation=False)

    trainer.max_epochs = 1

    trainer.setup_distributed(
        DummyDistributed(root=False),
        DummyLogger(),
    )

    FakeAggregator.plot_calls.clear()

    trainer.train()

    assert trainer._epochs_trained == 1


def test_load_checkpoint_restores_histories(env_dirs):
    trainer, _, _, _, _ = make_trainer(validation=True)

    trainer.setup_distributed(
        DummyDistributed(),
        DummyLogger(),
    )

    trainer.train_aggregator.records.append({"a": 1})
    trainer.validation_aggregator.records.append({"b": 2})

    trainer._save_checkpoint(
        name="history",
        train_logs={"total_loss": 1.0},
        validation_logs={"total_loss": 1.0},
    )

    trainer._load_checkpoint(env_dirs[0] / "history.pt")

    assert trainer.train_aggregator.loaded_state is not None
    assert trainer.validation_aggregator.loaded_state is not None


@pytest.mark.pruned
def test_should_stop_early_exact_buffer():
    trainer, _, _, _, _ = make_trainer(validation=True)

    trainer.config.earlystoppingbuffer = 3
    trainer.earlystopping_counter = 3

    assert trainer._should_stop_early() is True


@pytest.mark.pruned
def test_is_improved_worse_loss():
    trainer, _, _, _, _ = make_trainer(validation=True)

    trainer._best_validation_loss = 1.0

    assert trainer._is_improved(2.0) is False
