import unittest

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


def make_optimizer(module=None, lr=0.01):
    if module is None:
        module = make_module()
    return torch.optim.Adam(module.parameters(), lr=lr)


def test_lr_scheduler_build_warmup_equals_total_epochs_invalid():
    module = make_module()
    opt = make_optimizer(module)
    cfg = LRSchedulerConfig(warmup_epochs=2, total_epochs=2)

    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        cfg.build(opt, num_batches=1)


@pytest.mark.pruned
def test_lr_scheduler_build_sets_steps():
    module = make_module()
    opt = make_optimizer(module)
    cfg = LRSchedulerConfig(warmup_epochs=1, total_epochs=4)

    scheduler = cfg.build(opt, num_batches=3)

    assert isinstance(scheduler, CosineAnnealingLRScheduler)
    assert cfg.total_steps == 12
    assert cfg.warmup_steps == 3


@pytest.mark.pruned
def test_cosine_scheduler_without_warmup():
    module = make_module()
    opt = make_optimizer(module)

    cfg = LRSchedulerConfig(min_lr=0.001, warmup_epochs=0, total_epochs=10)
    cfg.total_steps = 10
    cfg.warmup_steps = 0

    scheduler = CosineAnnealingLRScheduler(cfg, opt)

    assert scheduler.scheduler is not None

    scheduler.step()
    state = scheduler.state_dict()

    assert isinstance(state, dict)


@pytest.mark.pruned
def test_cosine_scheduler_with_warmup():
    module = make_module()
    opt = make_optimizer(module)

    cfg = LRSchedulerConfig(min_lr=0.001, warmup_epochs=1, total_epochs=10)
    cfg.total_steps = 10
    cfg.warmup_steps = 2

    scheduler = CosineAnnealingLRScheduler(cfg, opt)

    assert scheduler.scheduler is not None

    scheduler.step()
    state = scheduler.state_dict()

    assert isinstance(state, dict)


@pytest.mark.pruned
def test_cosine_scheduler_load_state_dict():
    module = make_module()
    opt = make_optimizer(module)

    cfg = LRSchedulerConfig(min_lr=0.001, warmup_epochs=0, total_epochs=10)
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
    opt = make_optimizer(module)

    cfg = LRSchedulerConfig(total_epochs=10)
    cfg.total_steps = 10
    cfg.warmup_steps = 0

    scheduler = CosineAnnealingLRScheduler(cfg, opt)
    del scheduler.scheduler

    with pytest.raises(RuntimeError):
        scheduler.step()


def test_cosine_scheduler_state_dict_without_scheduler_attr():
    module = make_module()
    opt = make_optimizer(module)

    cfg = LRSchedulerConfig(total_epochs=10)
    cfg.total_steps = 10
    cfg.warmup_steps = 0

    scheduler = CosineAnnealingLRScheduler(cfg, opt)
    del scheduler.scheduler

    with pytest.raises(RuntimeError):
        scheduler.state_dict()


def test_cosine_scheduler_load_state_dict_without_scheduler_attr():
    module = make_module()
    opt = make_optimizer(module)

    cfg = LRSchedulerConfig(total_epochs=10)
    cfg.total_steps = 10
    cfg.warmup_steps = 0

    scheduler = CosineAnnealingLRScheduler(cfg, opt)
    state = scheduler.state_dict()
    del scheduler.scheduler

    with pytest.raises(RuntimeError):
        scheduler.load_state_dict(state)


@pytest.mark.pruned
def test_optimizer_config_defaults():
    cfg = OptimizerConfig()

    assert cfg.lr == 0.0001
    assert cfg.weight_decay == 0
    assert cfg.optimizer_type == "adam"
    assert cfg.optimizer is None


def test_optimizer_config_invalid_weight_decay():
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        OptimizerConfig(weight_decay=-1)


@pytest.mark.pruned
def test_optimizer_build_accepts_unbuilt_module_current_behavior():
    module = make_module(built=False)
    cfg = OptimizerConfig()

    wrapper = cfg.build(module)

    assert isinstance(wrapper, OptimizerWrapper)
    assert wrapper.optimizer is not None


@pytest.mark.pruned
def test_optimizer_build_without_scheduler():
    module = make_module()
    cfg = OptimizerConfig(lr=0.01, optimizer_type="adam")

    wrapper = cfg.build(module)

    assert isinstance(wrapper, OptimizerWrapper)
    assert isinstance(wrapper.optimizer, torch.optim.Adam)


@pytest.mark.pruned
def test_optimizer_build_adamw_case_insensitive():
    module = make_module()
    cfg = OptimizerConfig(lr=0.01, optimizer_type="ADAMW")

    wrapper = cfg.build(module)

    assert isinstance(wrapper, OptimizerWrapper)
    assert isinstance(wrapper.optimizer, torch.optim.AdamW)


@pytest.mark.pruned
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
    cfg = OptimizerConfig(lr_scheduler_config=LRSchedulerConfig(total_epochs=2))

    with pytest.raises(ValueError):
        OptimizerWrapper(cfg, module, num_batches=None, max_epochs=2)


def test_optimizer_build_scheduler_requires_epochs_wrapper_level():
    module = make_module()
    cfg = OptimizerConfig(lr_scheduler_config=LRSchedulerConfig())

    with pytest.raises(ValueError):
        OptimizerWrapper(cfg, module, num_batches=2, max_epochs=None)


@pytest.mark.pruned
def test_optimizer_build_with_scheduler():
    module = make_module()
    cfg = OptimizerConfig(
        lr=0.01,
        lr_scheduler_config=LRSchedulerConfig(min_lr=0.001, warmup_epochs=0),
    )

    wrapper = cfg.build(module, num_batches=2, max_epochs=3)

    assert isinstance(wrapper, OptimizerWrapper)
    assert isinstance(wrapper.optimizer, torch.optim.Adam)
    assert wrapper.lr_scheduler is not None
    assert cfg.lr_scheduler_config.total_epochs == 3
    assert cfg.lr_scheduler_config.total_steps == 6
    assert cfg.lr_scheduler_config.warmup_steps == 0


@pytest.mark.pruned
def test_optimizer_build_with_scheduler_total_epochs_preconfigured():
    module = make_module()
    cfg = OptimizerConfig(
        lr=0.01,
        lr_scheduler_config=LRSchedulerConfig(
            min_lr=0.001,
            warmup_epochs=0,
            total_epochs=4,
        ),
    )

    wrapper = cfg.build(module, num_batches=2)

    assert isinstance(wrapper, OptimizerWrapper)
    assert wrapper.lr_scheduler is not None
    assert cfg.lr_scheduler_config.total_epochs == 4
    assert cfg.lr_scheduler_config.total_steps == 8
    assert cfg.lr_scheduler_config.warmup_steps == 0


def test_optimizer_build_with_warmup_scheduler():
    module = make_module()
    cfg = OptimizerConfig(
        lr=0.01,
        lr_scheduler_config=LRSchedulerConfig(min_lr=0.001, warmup_epochs=1),
    )

    wrapper = cfg.build(module, num_batches=2, max_epochs=3)

    assert isinstance(wrapper, OptimizerWrapper)
    assert wrapper.lr_scheduler is not None
    assert cfg.lr_scheduler_config.total_epochs == 3
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


@pytest.mark.pruned
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


@pytest.mark.pruned
def test_optimizer_scheduler_step_with_scheduler():
    module = make_module()
    cfg = OptimizerConfig(
        lr=0.01,
        lr_scheduler_config=LRSchedulerConfig(min_lr=0.001, warmup_epochs=0),
    )
    wrapper = cfg.build(module, num_batches=2, max_epochs=3)

    wrapper.scheduler_step()

    assert wrapper.lr_scheduler is not None


@pytest.mark.pruned
def test_optimizer_scheduler_step_without_scheduler_current_behavior():
    module = make_module()
    cfg = OptimizerConfig(lr=0.01)
    wrapper = cfg.build(module)

    assert wrapper.scheduler_step() is None


@pytest.mark.pruned
def test_optimizer_state_dict_with_scheduler():
    module = make_module()
    cfg = OptimizerConfig(
        lr=0.01,
        lr_scheduler_config=LRSchedulerConfig(min_lr=0.001, warmup_epochs=0),
    )
    wrapper = cfg.build(module, num_batches=2, max_epochs=3)

    state = wrapper.state_dict()

    assert "optimizer" in state
    assert "lr_scheduler" in state
    assert state["lr_scheduler"] is not None


@pytest.mark.pruned
def test_optimizer_state_dict_without_scheduler_current_behavior():
    module = make_module()
    cfg = OptimizerConfig(lr=0.01)
    wrapper = cfg.build(module)

    state = wrapper.state_dict()
    assert "optimizer" in state
    assert "scheduler" not in state or state["scheduler"] is None


def test_optimizer_load_state_dict_with_scheduler():
    module = make_module()
    cfg = OptimizerConfig(
        lr=0.01,
        lr_scheduler_config=LRSchedulerConfig(min_lr=0.001, warmup_epochs=0),
    )
    wrapper = cfg.build(module, num_batches=2, max_epochs=3)

    state = wrapper.state_dict()

    module2 = make_module()
    cfg2 = OptimizerConfig(
        lr=0.01,
        lr_scheduler_config=LRSchedulerConfig(min_lr=0.001, warmup_epochs=0),
    )
    wrapper2 = cfg2.build(module2, num_batches=2, max_epochs=3)

    wrapper2.load_state_dict(state)

    assert wrapper2.state_dict() is not None


@pytest.mark.pruned
def test_optimizer_load_state_dict_without_scheduler_state():
    module = make_module()
    cfg = OptimizerConfig(
        lr=0.01,
        lr_scheduler_config=LRSchedulerConfig(min_lr=0.001, warmup_epochs=0),
    )
    wrapper = cfg.build(module, num_batches=2, max_epochs=3)

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


@pytest.mark.pruned
def test_optimizer_zero_grad_without_optimizer_attr():
    module = make_module()
    cfg = OptimizerConfig(lr=0.01)
    wrapper = cfg.build(module)

    wrapper.optimizer = None

    with pytest.raises(RuntimeError):
        wrapper.zero_grad()


@pytest.mark.pruned
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


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_lr": -0.1}, None),
        ({"warmup_epochs": -1}, None),
    ],
)
def test_lr_scheduler_config_rejects_negative_values(
    kwargs,
    message,
):
    with pytest.raises(AssertionError, match=message):
        LRSchedulerConfig(**kwargs)


@pytest.mark.parametrize(
    "total_epochs",
    [
        None,
        0,
        -1,
    ],
)
def test_lr_scheduler_build_rejects_nonpositive_total_epochs(
    total_epochs,
):
    module = make_module()
    optimizer = make_optimizer(module)
    config = LRSchedulerConfig(
        total_epochs=total_epochs,
    )

    with pytest.raises(
        ValueError,
        match="total_epochs must be positive",
    ):
        config.build(
            optimizer,
            num_batches=2,
        )


@pytest.mark.parametrize(
    "num_batches",
    [
        0,
        -1,
    ],
)
def test_lr_scheduler_build_rejects_nonpositive_num_batches(
    num_batches,
):
    module = make_module()
    optimizer = make_optimizer(module)
    config = LRSchedulerConfig(
        total_epochs=2,
    )

    with pytest.raises(
        ValueError,
        match="num_batches must be positive",
    ):
        config.build(
            optimizer,
            num_batches=num_batches,
        )


@pytest.mark.parametrize(
    "gradient_accumulation_steps",
    [
        0,
        -1,
    ],
)
def test_lr_scheduler_build_rejects_nonpositive_accumulation_steps(
    gradient_accumulation_steps,
):
    module = make_module()
    optimizer = make_optimizer(module)
    config = LRSchedulerConfig(
        total_epochs=2,
    )

    with pytest.raises(
        ValueError,
        match="gradient_accumulation_steps must be positive",
    ):
        config.build(
            optimizer,
            num_batches=4,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )


@pytest.mark.pruned
def test_lr_scheduler_build_accounts_for_gradient_accumulation():
    module = make_module()
    optimizer = make_optimizer(module)
    config = LRSchedulerConfig(
        warmup_epochs=2,
        total_epochs=5,
    )

    scheduler = config.build(
        optimizer,
        num_batches=10,
        gradient_accumulation_steps=4,
    )

    assert isinstance(
        scheduler,
        CosineAnnealingLRScheduler,
    )
    assert config.total_steps == 15
    assert config.warmup_steps == 6


@pytest.mark.pruned
def test_lr_scheduler_build_rounds_partial_accumulation_batch_up():
    module = make_module()
    optimizer = make_optimizer(module)
    config = LRSchedulerConfig(
        warmup_epochs=1,
        total_epochs=3,
    )

    config.build(
        optimizer,
        num_batches=5,
        gradient_accumulation_steps=2,
    )

    assert config.total_steps == 9
    assert config.warmup_steps == 3


@pytest.mark.pruned
def test_optimizer_build_forwards_gradient_accumulation_steps():
    module = make_module()
    scheduler_config = LRSchedulerConfig(
        warmup_epochs=1,
    )
    config = OptimizerConfig(
        lr_scheduler_config=scheduler_config,
    )

    wrapper = config.build(
        module,
        num_batches=5,
        max_epochs=3,
        gradient_accumulation_steps=2,
    )

    assert wrapper.lr_scheduler is not None
    assert scheduler_config.total_steps == 9
    assert scheduler_config.warmup_steps == 3


@pytest.mark.pruned
def test_optimizer_build_preserves_preconfigured_total_epochs():
    module = make_module()
    scheduler_config = LRSchedulerConfig(
        total_epochs=4,
    )
    config = OptimizerConfig(
        lr_scheduler_config=scheduler_config,
    )

    config.build(
        module,
        num_batches=2,
        max_epochs=10,
    )

    assert scheduler_config.total_epochs == 4
    assert scheduler_config.total_steps == 8


@pytest.mark.parametrize(
    ("optimizer_type", "expected_type"),
    [
        ("adam", torch.optim.Adam),
        ("ADAM", torch.optim.Adam),
        ("AdamW", torch.optim.AdamW),
        ("ADAMW", torch.optim.AdamW),
    ],
)
def test_optimizer_registry_is_case_insensitive(
    optimizer_type,
    expected_type,
):
    module = make_module()
    config = OptimizerConfig(
        optimizer_type=optimizer_type,
    )

    wrapper = config.build(module)

    assert isinstance(
        wrapper.optimizer,
        expected_type,
    )


@pytest.mark.pruned
def test_optimizer_wrapper_optimizes_only_trainable_parameters():
    module = make_module()
    module.linear.bias.requires_grad = False

    wrapper = OptimizerConfig().build(module)

    optimized_parameters = {
        id(parameter)
        for group in wrapper.optimizer.param_groups
        for parameter in group["params"]
    }

    assert id(module.linear.weight) in optimized_parameters
    assert id(module.linear.bias) not in optimized_parameters


@pytest.mark.pruned
def test_optimizer_wrapper_rejects_module_without_trainable_parameters():
    module = make_module()

    for parameter in module.parameters():
        parameter.requires_grad = False

    with pytest.raises(ValueError):
        OptimizerConfig().build(module)


@pytest.mark.pruned
def test_optimizer_learning_rate_property():
    module = make_module()
    config = OptimizerConfig(
        lr=0.0125,
    )

    wrapper = config.build(module)

    assert wrapper.learning_rate == pytest.approx(0.0125)


@pytest.mark.pruned
def test_optimizer_learning_rate_reflects_param_group_change():
    module = make_module()
    wrapper = OptimizerConfig(
        lr=0.01,
    ).build(module)

    wrapper.optimizer.param_groups[0]["lr"] = 0.004

    assert wrapper.learning_rate == pytest.approx(0.004)


@pytest.mark.pruned
def test_scheduler_step_advances_before_total_steps():
    module = make_module()
    optimizer = make_optimizer(module)
    config = LRSchedulerConfig(
        total_epochs=2,
        hold_min_lr=True,
    )
    config.total_steps = 2
    config.warmup_steps = 0

    scheduler = CosineAnnealingLRScheduler(
        config,
        optimizer,
    )

    scheduler.step()

    assert scheduler.num_steps == 1


@pytest.mark.pruned
def test_scheduler_step_reaches_configured_total_steps():
    module = make_module()
    optimizer = make_optimizer(module)
    config = LRSchedulerConfig(
        total_epochs=2,
        hold_min_lr=True,
    )
    config.total_steps = 2
    config.warmup_steps = 0

    scheduler = CosineAnnealingLRScheduler(
        config,
        optimizer,
    )

    scheduler.step()
    scheduler.step()

    assert scheduler.num_steps == config.total_steps


def test_scheduler_holds_minimum_lr_after_final_step():
    module = make_module()
    optimizer = make_optimizer(
        module,
        lr=0.01,
    )
    config = LRSchedulerConfig(
        min_lr=0.002,
        total_epochs=1,
        hold_min_lr=True,
    )
    config.total_steps = 1
    config.warmup_steps = 0

    scheduler = CosineAnnealingLRScheduler(
        config,
        optimizer,
    )

    scheduler.step()
    scheduler.step()

    assert scheduler.num_steps == 1

    for param_group in optimizer.param_groups:
        assert param_group["lr"] == pytest.approx(0.002)


@pytest.mark.pruned
def test_scheduler_does_not_increment_after_final_step_when_holding_minimum():
    module = make_module()
    optimizer = make_optimizer(module)
    config = LRSchedulerConfig(
        total_epochs=1,
        hold_min_lr=True,
    )
    config.total_steps = 1
    config.warmup_steps = 0

    scheduler = CosineAnnealingLRScheduler(
        config,
        optimizer,
    )

    scheduler.step()
    scheduler.step()
    scheduler.step()

    assert scheduler.num_steps == 1


def test_scheduler_continues_stepping_when_minimum_is_not_held(
    monkeypatch,
):
    module = make_module()
    optimizer = make_optimizer(module)
    config = LRSchedulerConfig(
        total_epochs=1,
        hold_min_lr=False,
    )
    config.total_steps = 1
    config.warmup_steps = 0

    scheduler = CosineAnnealingLRScheduler(
        config,
        optimizer,
    )
    scheduler.num_steps = config.total_steps

    step_mock = unittest.mock.Mock()
    monkeypatch.setattr(
        scheduler.scheduler,
        "step",
        step_mock,
    )

    scheduler.step()

    step_mock.assert_called_once_with()
    assert scheduler.num_steps == config.total_steps + 1


@pytest.mark.pruned
def test_scheduler_hold_minimum_updates_every_parameter_group():
    first = nn.Parameter(torch.tensor([1.0]))
    second = nn.Parameter(torch.tensor([2.0]))

    optimizer = torch.optim.Adam(
        [
            {
                "params": [first],
                "lr": 0.01,
            },
            {
                "params": [second],
                "lr": 0.02,
            },
        ]
    )

    config = LRSchedulerConfig(
        min_lr=0.001,
        total_epochs=1,
        hold_min_lr=True,
    )
    config.total_steps = 1
    config.warmup_steps = 0

    scheduler = CosineAnnealingLRScheduler(
        config,
        optimizer,
    )
    scheduler.num_steps = config.total_steps

    scheduler.step()

    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [
            0.001,
            0.001,
        ]
    )


@pytest.mark.pruned
def test_optimizer_scheduler_step_delegates_to_scheduler(
    monkeypatch,
):
    module = make_module()
    config = OptimizerConfig(
        lr_scheduler_config=LRSchedulerConfig(
            total_epochs=2,
        ),
    )
    wrapper = config.build(
        module,
        num_batches=2,
    )

    step_mock = unittest.mock.Mock()
    monkeypatch.setattr(
        wrapper.lr_scheduler,
        "step",
        step_mock,
    )

    wrapper.scheduler_step()

    step_mock.assert_called_once_with()


@pytest.mark.pruned
def test_optimizer_step_delegates_to_optimizer(
    monkeypatch,
):
    module = make_module()
    wrapper = OptimizerConfig().build(module)

    step_mock = unittest.mock.Mock()
    monkeypatch.setattr(
        wrapper.optimizer,
        "step",
        step_mock,
    )

    wrapper.step()

    step_mock.assert_called_once_with()


@pytest.mark.pruned
def test_optimizer_zero_grad_forwards_keyword_arguments(
    monkeypatch,
):
    module = make_module()
    wrapper = OptimizerConfig().build(module)

    zero_grad_mock = unittest.mock.Mock()
    monkeypatch.setattr(
        wrapper.optimizer,
        "zero_grad",
        zero_grad_mock,
    )

    wrapper.zero_grad(
        set_to_none=False,
    )

    zero_grad_mock.assert_called_once_with(
        set_to_none=False,
    )


@pytest.mark.pruned
def test_optimizer_state_dict_without_scheduler_uses_none():
    module = make_module()
    wrapper = OptimizerConfig().build(module)

    state = wrapper.state_dict()

    assert state["lr_scheduler"] is None


@pytest.mark.pruned
def test_optimizer_load_state_dict_does_not_load_absent_scheduler_state(
    monkeypatch,
):
    module = make_module()
    config = OptimizerConfig(
        lr_scheduler_config=LRSchedulerConfig(
            total_epochs=2,
        ),
    )
    wrapper = config.build(
        module,
        num_batches=2,
    )

    scheduler_load_mock = unittest.mock.Mock()
    monkeypatch.setattr(
        wrapper.lr_scheduler,
        "load_state_dict",
        scheduler_load_mock,
    )

    state = wrapper.state_dict()
    state["lr_scheduler"] = None

    wrapper.load_state_dict(state)

    scheduler_load_mock.assert_not_called()


@pytest.mark.pruned
def test_optimizer_without_scheduler_ignores_scheduler_state(
    monkeypatch,
):
    module = make_module()
    wrapper = OptimizerConfig().build(module)

    optimizer_state = wrapper.optimizer.state_dict()
    state = {
        "optimizer": optimizer_state,
        "lr_scheduler": {
            "unused": True,
        },
    }

    wrapper.load_state_dict(state)

    assert wrapper.lr_scheduler is None


@pytest.mark.pruned
def test_optimizer_load_state_dict_requires_optimizer_key():
    module = make_module()
    wrapper = OptimizerConfig().build(module)

    with pytest.raises(KeyError):
        wrapper.load_state_dict(
            {
                "lr_scheduler": None,
            }
        )


def test_optimizer_load_state_dict_requires_scheduler_key():
    module = make_module()
    wrapper = OptimizerConfig(
        lr_scheduler_config=LRSchedulerConfig(
            total_epochs=2,
        ),
    ).build(
        module,
        num_batches=2,
    )

    with pytest.raises(KeyError):
        wrapper.load_state_dict(
            {
                "optimizer": wrapper.optimizer.state_dict(),
            }
        )


@pytest.mark.pruned
def test_scheduler_state_dict_round_trip_preserves_scheduler_epoch():
    module = make_module()
    optimizer = make_optimizer(module)
    config = LRSchedulerConfig(
        total_epochs=4,
    )
    config.total_steps = 4
    config.warmup_steps = 0

    scheduler = CosineAnnealingLRScheduler(
        config,
        optimizer,
    )
    scheduler.step()
    scheduler.step()

    state = scheduler.state_dict()

    second_optimizer = make_optimizer(
        make_module(),
    )
    second_scheduler = CosineAnnealingLRScheduler(
        config,
        second_optimizer,
    )
    second_scheduler.load_state_dict(state)

    assert second_scheduler.scheduler.last_epoch == scheduler.scheduler.last_epoch