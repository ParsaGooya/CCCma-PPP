# test_abc_models.py

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from cccma_ppp.models.models_abc import (
    CheckpointConfig,
    modelABC,
    modelConfigABC,
    weights_init,
)


# ------------------------------------------------------------------
# Test helpers
# ------------------------------------------------------------------


class DummyConfig(modelConfigABC):
    def build(self):
        return "built"


class DummyModel(modelABC):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def build(self, *args, **kwargs):
        return self

    def forward(self, x):
        return self.linear(x)


class BufferOnlyModel(modelABC):
    def __init__(self):
        super().__init__()
        self.register_buffer("buf", torch.zeros(1))

    def parameters(self):
        return iter(())

    def build(self, *args, **kwargs):
        return self

    def forward(self, x):
        return x


class EmptyModel(modelABC):
    def build(self, *args, **kwargs):
        return self

    def forward(self, x):
        return x

    def parameters(self):
        return iter(())

    def buffers(self):
        return iter(())


# ------------------------------------------------------------------
# modelConfigABC
# ------------------------------------------------------------------


def test_add_checkpoint_config():
    cfg = DummyConfig()

    checkpoint_cfg = CheckpointConfig(
        load_path="foo.pt",
        checkpoint_input_shape=np.array([1]),
        checkpoint_output_shape=np.array([1]),
    )

    cfg._add_checkpoint_config(checkpoint_cfg)

    assert cfg.checkpoint_config is checkpoint_cfg


# ------------------------------------------------------------------
# _get_device
# ------------------------------------------------------------------


def test_get_device_from_parameter():
    model = DummyModel()

    assert model._get_device() == model.linear.weight.device


def test_get_device_from_buffer():
    model = BufferOnlyModel()

    assert model._get_device() == model.buf.device


def test_get_device_cpu_fallback():
    model = EmptyModel()

    assert model._get_device() == torch.device("cpu")


# ------------------------------------------------------------------
# weights_init
# ------------------------------------------------------------------


def test_weights_init_non_module_returns():
    relu = nn.ReLU()

    # should simply return
    weights_init(relu)


def test_weights_init_xavier():
    layer = nn.Linear(4, 4)

    weights_init(layer, method="xavier")

    assert torch.all(layer.bias == 0)


def test_weights_init_trunc_normal():
    layer = nn.Linear(4, 4)

    weights_init(layer, method="trunc_normal")

    assert torch.all(layer.bias == 0)


def test_weights_init_invalid_method():
    layer = nn.Linear(4, 4)

    with pytest.raises(NotImplementedError):
        weights_init(layer, method="invalid")


def test_weights_init_frozen_weight_and_bias():
    layer = nn.Linear(4, 4)

    layer.weight.requires_grad = False
    layer.bias.requires_grad = False

    old_weight = layer.weight.clone()
    old_bias = layer.bias.clone()

    weights_init(layer)

    assert torch.equal(old_weight, layer.weight)
    assert torch.equal(old_bias, layer.bias)


# ------------------------------------------------------------------
# _initialize_weights
# ------------------------------------------------------------------


def test_initialize_weights():
    model = DummyModel()

    model._initialize_weights()

    assert model.linear.bias is not None


# ------------------------------------------------------------------
# _load_state_dict
# ------------------------------------------------------------------


def test_load_state_dict_missing_checkpoint(tmp_path):
    model = DummyModel()

    cfg = CheckpointConfig(
        load_path=tmp_path / "missing.pt",
        checkpoint_input_shape=np.array([2]),
        checkpoint_output_shape=np.array([2]),
    )

    with pytest.raises(FileNotFoundError):
        model._load_state_dict(cfg)


def test_load_state_dict_success(tmp_path):
    model = DummyModel()

    checkpoint = {
        "module": {
            "model.linear.weight": torch.ones_like(model.linear.weight),
            "model.linear.bias": torch.ones_like(model.linear.bias),
        }
    }

    ckpt_path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, ckpt_path)

    cfg = CheckpointConfig(
        load_path=ckpt_path,
        checkpoint_input_shape=np.array([2]),
        checkpoint_output_shape=np.array([2]),
        strict=True,
    )

    model._load_state_dict(cfg)

    assert torch.all(model.linear.weight == 1)
    assert torch.all(model.linear.bias == 1)


def test_load_state_dict_freezes_weights(tmp_path):
    model = DummyModel()

    checkpoint = {
        "module": {
            "model.linear.weight": torch.ones_like(model.linear.weight),
            "model.linear.bias": torch.ones_like(model.linear.bias),
        }
    }

    ckpt_path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, ckpt_path)

    cfg = CheckpointConfig(
        load_path=ckpt_path,
        checkpoint_input_shape=np.array([2]),
        checkpoint_output_shape=np.array([2]),
        freeze_weights=True,
    )

    model._load_state_dict(cfg)

    assert all(not p.requires_grad for p in model.parameters())
