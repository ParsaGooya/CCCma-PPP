import pytest
import torch
import torch.nn as nn
import numpy as np

from cccma_ppp.core.core_abc import moduleABC
from cccma_ppp.core.optimization import (
    LRSchedulerConfig,
    OptimizerConfig,
    OptimizerWrapper,
    CosineAnnealingLRScheduler,
)


class DummyModule(moduleABC):
    def __init__(self, built=True):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.built = built

    def build(
        self, input_shape: np.ndarray, output_shape=None, added_features_dim=None
    ):
        self.built = True
        return self

    def init_loss_function(self, reconstruction_loss, **kwargs):
        self.criterion = reconstruction_loss

    def _compute_loss(self):
        return torch.tensor(1.0)

    def forward(self, x=None):
        if x is None:
            x = torch.ones(1, 2)
        return self.linear(x)

    def predict(self):
        return self.forward()


def make_module(built=True):
    return DummyModule(built=built)


def test_lr_scheduler_config_valid_defaults():
    cfg = LRSchedulerConfig()
    assert cfg.min_lr == 0.0
    assert cfg.warmup_epochs == 0


def test_lr_scheduler_config_invalid_min_lr():
    with pytest.raises(AssertionError):
        LRSchedulerConfig(min_lr=-1.0)


def test_lr_scheduler_config_invalid_warmup():
    with pytest.raises(AssertionError):
        LRSchedulerConfig(warmup_epochs=-1)


def test_lr_scheduler_build_invalid_total_epochs():
    module = make_module()
    opt = torch.optim.Adam(module.parameters(), lr=0.01)
    cfg = LRSchedulerConfig()

    with pytest.raises(AssertionError):
        cfg.build(opt, num_batches=1, total_epochs=0)


def test_lr_scheduler_build_invalid_num_batches():
    module = make_module()
    opt = torch.optim.Adam(module.parameters(), lr=0.01)
    cfg = LRSchedulerConfig()

    with pytest.raises(AssertionError):
        cfg.build(opt, num_batches=0, total_epochs=2)


def test_lr_scheduler_build_warmup_equals_total_epochs_invalid():
    module = make_module()
    opt = torch.optim.Adam(module.parameters(), lr=0.01)
    cfg = LRSchedulerConfig(warmup_epochs=2)

    with pytest.raises(AssertionError):
        cfg.build(opt, num_batches=1, total_epochs=2)


def test_lr_scheduler_build_sets_steps():
    module = make_module()
    opt = torch.optim.Adam(module.parameters(), lr=0.01)
    cfg = LRSchedulerConfig(warmup_epochs=1)

    scheduler = cfg.build(opt, num_batches=3, total_epochs=4)

    assert isinstance(scheduler, CosineAnnealingLRScheduler)
    assert cfg.total_steps == 12
    assert cfg.warmup_steps == 3


def test_cosine_scheduler_without_warmup():
    module = make_module()
    opt = torch.optim.Adam(module.parameters(), lr=0.01)

    cfg = LRSchedulerConfig(min_lr=0.001, warmup_epochs=0)
    cfg.total_steps = 10
    cfg.warmup_steps = 0

    scheduler = CosineAnnealingLRScheduler(cfg, opt)

    assert scheduler.scheduler is not None
    scheduler.step()
    state = scheduler.state_dict()
    assert isinstance(state, dict)


def test_cosine_scheduler_with_warmup():
    module = make_module()
    opt = torch.optim.Adam(module.parameters(), lr=0.01)

    cfg = LRSchedulerConfig(min_lr=0.001, warmup_epochs=1)
    cfg.total_steps = 10
    cfg.warmup_steps = 2

    scheduler = CosineAnnealingLRScheduler(cfg, opt)

    assert scheduler.scheduler is not None
    scheduler.step()
    state = scheduler.state_dict()
    assert isinstance(state, dict)


def test_cosine_scheduler_load_state_dict():
    module = make_module()
    opt = torch.optim.Adam(module.parameters(), lr=0.01)

    cfg = LRSchedulerConfig(min_lr=0.001, warmup_epochs=0)
    cfg.total_steps = 10
    cfg.warmup_steps = 0

    scheduler = CosineAnnealingLRScheduler(cfg, opt)
    scheduler.step()

    state = scheduler.state_dict()

    scheduler2 = CosineAnnealingLRScheduler(cfg, opt)
    scheduler2.load_state_dict(state)

    assert scheduler2.state_dict() is not None


def test_cosine_scheduler_step_without_scheduler_attr():
    module = make_module()
    opt = torch.optim.Adam(module.parameters(), lr=0.01)

    cfg = LRSchedulerConfig()
    cfg.total_steps = 10
    cfg.warmup_steps = 0

    scheduler = CosineAnnealingLRScheduler(cfg, opt)
    del scheduler.scheduler

    with pytest.raises(RuntimeError):
        scheduler.step()


def test_cosine_scheduler_state_dict_without_scheduler_attr():
    module = make_module()
    opt = torch.optim.Adam(module.parameters(), lr=0.01)

    cfg = LRSchedulerConfig()
    cfg.total_steps = 10
    cfg.warmup_steps = 0

    scheduler = CosineAnnealingLRScheduler(cfg, opt)
    del scheduler.scheduler

    with pytest.raises(RuntimeError):
        scheduler.state_dict()


def test_cosine_scheduler_load_state_dict_without_scheduler_attr():
    module = make_module()
    opt = torch.optim.Adam(module.parameters(), lr=0.01)

    cfg = LRSchedulerConfig()
    cfg.total_steps = 10
    cfg.warmup_steps = 0

    scheduler = CosineAnnealingLRScheduler(cfg, opt)
    state = scheduler.state_dict()
    del scheduler.scheduler

    with pytest.raises(RuntimeError):
        scheduler.load_state_dict(state)


def test_optimizer_config_defaults():
    cfg = OptimizerConfig()

    assert cfg.lr == 0.0001
    assert cfg.weight_decay == 0
    assert cfg.optimizer_type == "adam"
    assert cfg.optimizer is None


def test_optimizer_config_invalid_weight_decay():
    with pytest.raises(AssertionError):
        OptimizerConfig(weight_decay=-1)


def test_optimizer_build_requires_built_module():
    module = make_module(built=False)
    cfg = OptimizerConfig()

    with pytest.raises(AssertionError):
        cfg.build(module)


def test_optimizer_build_without_scheduler():
    module = make_module()
    cfg = OptimizerConfig(lr=0.01, optimizer_type="adam")

    wrapper = cfg.build(module)

    assert isinstance(wrapper, OptimizerWrapper)
    assert isinstance(wrapper.optimizer, torch.optim.Adam)


def test_optimizer_build_adamw_case_insensitive():
    module = make_module()
    cfg = OptimizerConfig(lr=0.01, optimizer_type="ADAMW")

    wrapper = cfg.build(module)

    assert isinstance(wrapper.optimizer, torch.optim.AdamW)


def test_optimizer_build_invalid_optimizer_type():
    module = make_module()
    cfg = OptimizerConfig(optimizer_type="not_real")

    with pytest.raises(TypeError):
        cfg.build(module)


def test_optimizer_build_scheduler_requires_num_batches_and_epochs_config_level():
    module = make_module()
    cfg = OptimizerConfig(lr_scheduler_config=LRSchedulerConfig())

    with pytest.raises(ValueError):
        cfg.build(module)


def test_optimizer_build_scheduler_requires_num_batches_wrapper_level():
    module = make_module()
    cfg = OptimizerConfig(lr_scheduler_config=LRSchedulerConfig())

    with pytest.raises(ValueError):
        OptimizerWrapper(cfg, module, num_batches=None, total_epochs=2)


def test_optimizer_build_with_scheduler():
    module = make_module()
    cfg = OptimizerConfig(
        lr=0.01,
        lr_scheduler_config=LRSchedulerConfig(min_lr=0.001, warmup_epochs=0),
    )

    wrapper = cfg.build(module, num_batches=2, total_epochs=3)

    assert isinstance(wrapper.optimizer, torch.optim.Adam)
    assert wrapper.lr_scheduler is not None


def test_optimizer_build_with_warmup_scheduler():
    module = make_module()
    cfg = OptimizerConfig(
        lr=0.01,
        lr_scheduler_config=LRSchedulerConfig(min_lr=0.001, warmup_epochs=1),
    )

    wrapper = cfg.build(module, num_batches=2, total_epochs=3)

    assert wrapper.lr_scheduler is not None
    assert cfg.lr_scheduler_config.total_steps == 6
    assert cfg.lr_scheduler_config.warmup_steps == 2


def test_optimizer_step():
    module = make_module()
    cfg = OptimizerConfig(lr=0.01)
    wrapper = cfg.build(module)

    x = torch.ones(1, 2)
    y = module(x).sum()
    y.backward()

    wrapper.step()

    assert wrapper.optimizer is not None


def test_optimizer_zero_grad_default():
    module = make_module()
    cfg = OptimizerConfig(lr=0.01)
    wrapper = cfg.build(module)

    x = torch.ones(1, 2)
    y = module(x).sum()
    y.backward()

    wrapper.zero_grad()

    for p in module.parameters():
        assert p.grad is None


def test_optimizer_zero_grad_set_to_none_false():
    module = make_module()
    cfg = OptimizerConfig(lr=0.01)
    wrapper = cfg.build(module)

    x = torch.ones(1, 2)
    y = module(x).sum()
    y.backward()

    wrapper.zero_grad(set_to_none=False)

    for p in module.parameters():
        assert p.grad is not None
        assert torch.allclose(p.grad, torch.zeros_like(p.grad))


def test_optimizer_scheduler_step_with_scheduler():
    module = make_module()
    cfg = OptimizerConfig(
        lr=0.01,
        lr_scheduler_config=LRSchedulerConfig(min_lr=0.001, warmup_epochs=0),
    )
    wrapper = cfg.build(module, num_batches=2, total_epochs=3)

    wrapper.scheduler_step()

    assert wrapper.lr_scheduler is not None


def test_optimizer_scheduler_step_without_scheduler_current_behavior():
    module = make_module()
    cfg = OptimizerConfig(lr=0.01)
    wrapper = cfg.build(module)

    with pytest.raises(AttributeError):
        wrapper.scheduler_step()


def test_optimizer_state_dict_with_scheduler():
    module = make_module()
    cfg = OptimizerConfig(
        lr=0.01,
        lr_scheduler_config=LRSchedulerConfig(min_lr=0.001, warmup_epochs=0),
    )
    wrapper = cfg.build(module, num_batches=2, total_epochs=3)

    state = wrapper.state_dict()

    assert "optimizer" in state
    assert "lr_scheduler" in state
    assert state["lr_scheduler"] is not None


def test_optimizer_state_dict_without_scheduler_current_behavior():
    module = make_module()
    cfg = OptimizerConfig(lr=0.01)
    wrapper = cfg.build(module)

    with pytest.raises(AttributeError):
        wrapper.state_dict()


def test_optimizer_load_state_dict_with_scheduler():
    module = make_module()
    cfg = OptimizerConfig(
        lr=0.01,
        lr_scheduler_config=LRSchedulerConfig(min_lr=0.001, warmup_epochs=0),
    )
    wrapper = cfg.build(module, num_batches=2, total_epochs=3)

    state = wrapper.state_dict()

    module2 = make_module()
    cfg2 = OptimizerConfig(
        lr=0.01,
        lr_scheduler_config=LRSchedulerConfig(min_lr=0.001, warmup_epochs=0),
    )
    wrapper2 = cfg2.build(module2, num_batches=2, total_epochs=3)

    wrapper2.load_state_dict(state)

    assert wrapper2.state_dict() is not None


def test_optimizer_load_state_dict_without_scheduler_state():
    module = make_module()
    cfg = OptimizerConfig(
        lr=0.01,
        lr_scheduler_config=LRSchedulerConfig(min_lr=0.001, warmup_epochs=0),
    )
    wrapper = cfg.build(module, num_batches=2, total_epochs=3)

    state = wrapper.state_dict()
    state["lr_scheduler"] = None

    wrapper.load_state_dict(state)

    assert wrapper.optimizer is not None


def test_optimizer_step_without_optimizer_attr():
    module = make_module()
    cfg = OptimizerConfig(lr=0.01)
    wrapper = cfg.build(module)

    wrapper.optimizer = None

    with pytest.raises(RuntimeError):
        wrapper.step()


def test_optimizer_zero_grad_without_optimizer_attr():
    module = make_module()
    cfg = OptimizerConfig(lr=0.01)
    wrapper = cfg.build(module)

    wrapper.optimizer = None

    with pytest.raises(RuntimeError):
        wrapper.zero_grad()


def test_optimizer_state_dict_without_optimizer_attr():
    module = make_module()
    cfg = OptimizerConfig(lr=0.01)
    wrapper = cfg.build(module)

    wrapper.optimizer = None

    with pytest.raises(RuntimeError):
        wrapper.state_dict()


def test_optimizer_load_state_dict_without_optimizer_attr():
    module = make_module()
    cfg = OptimizerConfig(lr=0.01)
    wrapper = cfg.build(module)

    wrapper.optimizer = None

    with pytest.raises(RuntimeError):
        wrapper.load_state_dict({"optimizer": {}, "lr_scheduler": None})
