import pytest
import torch
import torch.nn as nn
import numpy as np

from cccma_ppp.models.models_abc import (
    CheckpointConfig,
    modelABC,
    modelConfigABC,
    cVAEmodelConfigABC,
    weights_init,
)


class DummyModel(modelABC):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)


class DummyConfig(modelConfigABC):
    NUM_OUTPUT_DIMS = 2
    GENERATOR = False

    def build(self, input_shape, output_shape=None, added_features_dim=None, **kwargs):
        return DummyModel()


class DummyCvaeConfig(cVAEmodelConfigABC):
    NUM_OUTPUT_DIMS = 2
    GENERATOR = True

    def __init__(self):
        self.latent_size = 4
        self.condition_embedding_size = 4
        self.condition_dependant_latent = True

    def build(self, *args, **kwargs):
        return DummyModel()


@pytest.mark.pruned
def test_get_device_from_parameter():
    model = DummyModel()
    device = model._get_device()
    assert isinstance(device, torch.device)


def test_get_device_cpu_when_empty():
    class EmptyModel(modelABC):
        def forward(self, x):
            return x

    m = EmptyModel()
    device = m._get_device()
    assert device.type == "cpu"


def test_load_state_dict_missing_file(tmp_path):
    model = DummyModel()

    cfg = CheckpointConfig(
        load_path=tmp_path / "missing.pt",
        checkpoint_input_shape=np.array([1]),
        checkpoint_output_shape=np.array([1]),
        checkpoint_input_var_metadata={},
        checkpoint_output_var_metadata={},
    )

    with pytest.raises(FileNotFoundError):
        model._load_state_dict(cfg)


def test_load_state_dict_success(tmp_path):
    model = DummyModel()

    new_model = DummyModel()
    for p in new_model.parameters():
        nn.init.constant_(p, 0.5)

    checkpoint = {
        "module": {f"model.{k}": v.clone() for k, v in new_model.state_dict().items()}
    }

    path = tmp_path / "ckpt.pt"
    torch.save(checkpoint, path)

    cfg = CheckpointConfig(
        load_path=path,
        checkpoint_input_shape=np.array([1]),
        checkpoint_output_shape=np.array([1]),
        checkpoint_input_var_metadata={},
        checkpoint_output_var_metadata={},
    )

    model._load_state_dict(cfg)

    for p in model.parameters():
        assert torch.allclose(p, torch.full_like(p, 0.5))


def test_freeze_weights(tmp_path):
    model = DummyModel()

    checkpoint = {
        "module": {f"model.{k}": v.clone() for k, v in model.state_dict().items()}
    }

    path = tmp_path / "ckpt.pt"
    torch.save(checkpoint, path)

    cfg = CheckpointConfig(
        load_path=path,
        checkpoint_input_shape=np.array([1]),
        checkpoint_output_shape=np.array([1]),
        checkpoint_input_var_metadata={},
        checkpoint_output_var_metadata={},
        freeze_weights=True,
    )

    model._load_state_dict(cfg)

    for p in model.parameters():
        assert p.requires_grad is False


@pytest.mark.pruned
def test_cvae_resolve_flow_success():
    cfg = DummyCvaeConfig()
    cfg._resolve_flow_settings(condition_dependant_flow=False)


def test_cvae_resolve_flow_error():
    cfg = DummyCvaeConfig()
    cfg.condition_embedding_size = 3

    with pytest.raises(ValueError):
        cfg._resolve_flow_settings(condition_dependant_flow=False)


@pytest.mark.pruned
def test_weights_init_xavier():
    layer = nn.Linear(10, 5)
    weights_init(layer, method="xavier")

    assert layer.weight is not None


@pytest.mark.pruned
def test_weights_init_trunc_normal():
    layer = nn.Linear(10, 5)
    weights_init(layer, method="trunc_normal")

    assert layer.weight is not None


@pytest.mark.pruned
def test_weights_init_unsupported_module():
    class Dummy:
        pass

    obj = Dummy()

    weights_init(obj)


def test_weights_init_invalid_method():
    layer = nn.Linear(2, 2)

    with pytest.raises(NotImplementedError):
        weights_init(layer, method="invalid")


def test_cvae_resolve_flow_with_flow_enabled_mismatch_allowed():
    cfg = DummyCvaeConfig()
    cfg.condition_embedding_size = 1
    cfg.latent_size = 4

    cfg._resolve_flow_settings(condition_dependant_flow=True)


def test_cvae_resolve_flow_condition_independent():
    cfg = DummyCvaeConfig()
    cfg.condition_dependant_latent = False

    cfg._resolve_flow_settings(condition_dependant_flow=False)


def test_get_device_from_buffer():
    class BufferModel(modelABC):
        def __init__(self):
            super().__init__()
            self.register_buffer("buf", torch.zeros(1))

        def forward(self, x):
            return x

    m = BufferModel()
    assert m._get_device().type == "cpu"


@pytest.mark.pruned
def test_load_state_dict_ignores_non_model_keys(tmp_path):
    model = DummyModel()

    checkpoint = {
        "module": {
            "model.linear.weight": torch.ones_like(model.linear.weight),
            "model.linear.bias": torch.zeros_like(model.linear.bias),
            "other.weight": torch.randn(1),
        }
    }

    path = tmp_path / "ckpt.pt"
    torch.save(checkpoint, path)

    cfg = CheckpointConfig(
        load_path=path,
        checkpoint_input_shape=np.array([1]),
        checkpoint_output_shape=np.array([1]),
        checkpoint_input_var_metadata={},
        checkpoint_output_var_metadata={},
        strict=True,
    )

    model._load_state_dict(cfg)


@pytest.mark.pruned
def test_load_state_dict_strict_false_missing_keys(tmp_path):
    model = DummyModel()

    checkpoint = {
        "module": {
            "model.linear.bias": torch.zeros_like(model.linear.bias),
        }
    }

    path = tmp_path / "ckpt.pt"
    torch.save(checkpoint, path)

    cfg = CheckpointConfig(
        load_path=path,
        checkpoint_input_shape=np.array([1]),
        checkpoint_output_shape=np.array([1]),
        checkpoint_input_var_metadata={},
        checkpoint_output_var_metadata={},
        strict=False,
    )

    model._load_state_dict(cfg)


@pytest.mark.pruned
def test_load_state_dict_no_model_prefix_keys(tmp_path):
    model = DummyModel()

    checkpoint = {"module": {"something": torch.tensor(1)}}

    path = tmp_path / "ckpt.pt"
    torch.save(checkpoint, path)

    cfg = CheckpointConfig(
        load_path=path,
        checkpoint_input_shape=np.array([1]),
        checkpoint_output_shape=np.array([1]),
        checkpoint_input_var_metadata={},
        checkpoint_output_var_metadata={},
        strict=False,
    )

    model._load_state_dict(cfg)


def test_weights_init_skips_frozen_params():
    layer = nn.Linear(3, 3)
    layer.weight.requires_grad = False
    layer.bias.requires_grad = False

    w_before = layer.weight.clone()

    weights_init(layer, method="xavier")

    assert torch.allclose(layer.weight, w_before)


@pytest.mark.pruned
def test_weights_init_bias_only():
    class BiasOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.ones(3))

    m = BiasOnly()

    weights_init(m, method="xavier")


@pytest.mark.pruned
def test_weights_init_changes_weights():
    layer = nn.Linear(10, 10)
    w_before = layer.weight.clone()

    weights_init(layer, method="trunc_normal")

    assert not torch.allclose(layer.weight, w_before)