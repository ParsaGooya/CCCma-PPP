import numpy as np
import pytest
import torch
import torch.nn as nn

from cccma_ppp.models.models_abc import (
    CheckpointConfig,
    cVAEmodelConfigABC,
    cVAEmodelsABC,
    deterministicmodelsABC,
    flowABC,
    modelABC,
    modelConfigABC,
    weights_init,
)


class ConcreteFlow(flowABC):
    def forward(self, x, condition=None):
        return x if condition is None else x + condition

    def inverse(self, z, condition=None):
        return z if condition is None else z - condition


class ConcreteModelConfig(modelConfigABC):
    NUM_OUTPUT_DIMS = 3
    GENERATOR = True

    def build(self, input_shape, output_shape=None, added_features_dim=None, **kwargs):
        return ConcreteModel(self)


class ConcreteDeterministicConfig(modelConfigABC):
    NUM_OUTPUT_DIMS = 2
    GENERATOR = False

    def build(self, input_shape, output_shape=None, added_features_dim=None, **kwargs):
        return ConcreteDeterministicModel(self)


class ConcreteCVAEConfig(cVAEmodelConfigABC):
    NUM_OUTPUT_DIMS = 2
    GENERATOR = True

    def build(self, input_shape, output_shape=None, added_features_dim=None, **kwargs):
        return ConcreteCvaeModel(self)


class ConcreteModel(modelABC):
    def __init__(self, config):
        super().__init__(config)
        self.linear = nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)


class EmptyModel(modelABC):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, x):
        return x


class BufferOnlyModel(modelABC):
    def __init__(self, config):
        super().__init__(config)
        self.register_buffer("buf", torch.ones(1))

    def forward(self, x):
        return x


class ConcreteDeterministicModel(deterministicmodelsABC):
    def __init__(self, config):
        super().__init__(config)
        self.linear = nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)


class ConcreteCvaeModel(cVAEmodelsABC):
    def __init__(self, config):
        super().__init__(config)
        self.linear = nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)

    def predict(self, x):
        return self.forward(x)

    def _recognition(self):
        return (torch.tensor(1.0),)

    def _condition(self):
        return (torch.tensor(2.0),)

    def _generate(self):
        return torch.tensor(3.0)


def test_cvae_resolve_flow_settings_condition_independent():
    cfg = ConcreteCVAEConfig(
        latent_size=4,
        condition_dependant_latent=False,
        condition_embedding_size=None,
    )

    result = cfg._resolve_flow_settings(condition_dependant_flow=False)

    assert result is cfg
    assert cfg.condition_dependant_flow is False


def test_cvae_resolve_flow_settings_condition_dependent_flow_on_allows_size_mismatch():
    cfg = ConcreteCVAEConfig(
        latent_size=4,
        condition_dependant_latent=True,
        condition_embedding_size=8,
    )

    result = cfg._resolve_flow_settings(condition_dependant_flow=True)

    assert result is cfg
    assert cfg.condition_dependant_flow is True


def test_cvae_resolve_flow_settings_condition_dependent_latent_without_flow_requires_size_match():
    cfg = ConcreteCVAEConfig(
        latent_size=4,
        condition_dependant_latent=True,
        condition_embedding_size=8,
    )

    with pytest.raises(ValueError, match="condition embedding size"):
        cfg._resolve_flow_settings(condition_dependant_flow=False)


def test_cvae_resolve_flow_settings_condition_dependent_latent_without_flow_size_match():
    cfg = ConcreteCVAEConfig(
        latent_size=4,
        condition_dependant_latent=True,
        condition_embedding_size=4,
    )

    result = cfg._resolve_flow_settings(condition_dependant_flow=False)

    assert result is cfg
    assert cfg.condition_dependant_flow is False


def test_get_device_from_parameter():
    model = ConcreteModel(ConcreteModelConfig())

    assert model._get_device() == next(model.parameters()).device


def test_get_device_from_buffer_when_no_parameters():
    model = BufferOnlyModel(ConcreteModelConfig())

    assert model._get_device() == model.buf.device


def test_get_device_cpu_when_no_parameters_or_buffers():
    model = EmptyModel(ConcreteModelConfig())

    assert model._get_device() == torch.device("cpu")


@pytest.mark.parametrize(
    "module",
    [
        nn.Linear(2, 2),
        nn.Conv1d(1, 1, kernel_size=1),
        nn.Conv2d(1, 1, kernel_size=1),
        nn.Conv3d(1, 1, kernel_size=1),
    ],
)
def test_weights_init_xavier_supported_modules(module):
    before_weight = module.weight.detach().clone()

    weights_init(module, method="xavier")

    assert not torch.allclose(module.weight.detach(), before_weight)
    if module.bias is not None:
        assert torch.allclose(module.bias.detach(), torch.zeros_like(module.bias))


@pytest.mark.parametrize(
    "module",
    [
        nn.Linear(2, 2),
        nn.Conv1d(1, 1, kernel_size=1),
        nn.Conv2d(1, 1, kernel_size=1),
        nn.Conv3d(1, 1, kernel_size=1),
    ],
)
def test_weights_init_trunc_normal_supported_modules(module):
    before_weight = module.weight.detach().clone()

    weights_init(module, method="trunc_normal")

    assert not torch.allclose(module.weight.detach(), before_weight)
    if module.bias is not None:
        assert torch.allclose(module.bias.detach(), torch.zeros_like(module.bias))


def test_weights_init_unsupported_module_noop():
    module = nn.ReLU()

    weights_init(module, method="not_a_real_method")


def test_weights_init_unknown_method_raises_for_supported_module():
    module = nn.Linear(2, 2)

    with pytest.raises(NotImplementedError, match="trunc_normal"):
        weights_init(module, method="bad")


def test_weights_init_skips_frozen_weight_and_bias():
    module = nn.Linear(2, 2)
    module.weight.requires_grad = False
    module.bias.requires_grad = False

    before_weight = module.weight.detach().clone()
    before_bias = module.bias.detach().clone()

    weights_init(module, method="xavier")

    assert torch.allclose(module.weight.detach(), before_weight)
    assert torch.allclose(module.bias.detach(), before_bias)


def test_initialize_weights_applies_to_submodules():
    model = ConcreteModel(ConcreteModelConfig())
    before_weight = model.linear.weight.detach().clone()

    model.init_method = "xavier"
    model._initialize_weights()

    assert not torch.allclose(model.linear.weight.detach(), before_weight)
    assert torch.allclose(
        model.linear.bias.detach(), torch.zeros_like(model.linear.bias)
    )


def test_initialize_weights_raises_for_bad_method():
    model = ConcreteModel(ConcreteModelConfig())
    model.init_method = "bad"

    with pytest.raises(NotImplementedError):
        model._initialize_weights()


def test_load_state_dict_missing_file_raises(tmp_path):
    model = ConcreteModel(ConcreteModelConfig())

    cfg = CheckpointConfig(
        load_path=tmp_path / "missing.pt",
        checkpoint_input_shape=np.array([2]),
        checkpoint_output_shape=np.array([2]),
        checkpoint_input_var_metadata={},
        checkpoint_output_var_metadata={},
    )

    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        model._load_state_dict(cfg)


def test_load_state_dict_loads_model_prefixed_keys(tmp_path):
    model = ConcreteModel(ConcreteModelConfig())

    new_model = ConcreteModel(ConcreteModelConfig())
    for param in new_model.parameters():
        nn.init.constant_(param, 0.25)

    checkpoint = {
        "module": {
            f"model.{key}": value.detach().clone()
            for key, value in new_model.state_dict().items()
        }
    }

    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)

    cfg = CheckpointConfig(
        load_path=checkpoint_path,
        checkpoint_input_shape=np.array([2]),
        checkpoint_output_shape=np.array([2]),
        checkpoint_input_var_metadata={},
        checkpoint_output_var_metadata={},
        strict=True,
        freeze_weights=False,
    )

    model._load_state_dict(cfg)

    for key, value in model.state_dict().items():
        assert torch.allclose(value, new_model.state_dict()[key])


def test_load_state_dict_ignores_non_model_prefixed_keys_with_strict_false(tmp_path):
    model = ConcreteModel(ConcreteModelConfig())

    checkpoint = {
        "module": {
            "model.linear.weight": torch.ones_like(model.linear.weight),
            "model.linear.bias": torch.zeros_like(model.linear.bias),
            "other.weight": torch.randn(1),
        }
    }

    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)

    cfg = CheckpointConfig(
        load_path=checkpoint_path,
        checkpoint_input_shape=np.array([2]),
        checkpoint_output_shape=np.array([2]),
        checkpoint_input_var_metadata={},
        checkpoint_output_var_metadata={},
        strict=True,
        freeze_weights=False,
    )

    model._load_state_dict(cfg)

    assert torch.allclose(model.linear.weight, torch.ones_like(model.linear.weight))
    assert torch.allclose(model.linear.bias, torch.zeros_like(model.linear.bias))


def test_load_state_dict_freezes_weights(tmp_path):
    model = ConcreteModel(ConcreteModelConfig())

    checkpoint = {
        "module": {
            f"model.{key}": value.detach().clone()
            for key, value in model.state_dict().items()
        }
    }

    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)

    cfg = CheckpointConfig(
        load_path=checkpoint_path,
        checkpoint_input_shape=np.array([2]),
        checkpoint_output_shape=np.array([2]),
        checkpoint_input_var_metadata={},
        checkpoint_output_var_metadata={},
        strict=True,
        freeze_weights=True,
    )

    assert all(param.requires_grad for param in model.parameters())

    model._load_state_dict(cfg)

    assert all(not param.requires_grad for param in model.parameters())


def test_load_state_dict_strict_false_allows_missing_keys(tmp_path):
    model = ConcreteModel(ConcreteModelConfig())

    checkpoint = {
        "module": {
            "model.linear.bias": torch.zeros_like(model.linear.bias),
        }
    }

    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)

    cfg = CheckpointConfig(
        load_path=checkpoint_path,
        checkpoint_input_shape=np.array([2]),
        checkpoint_output_shape=np.array([2]),
        checkpoint_input_var_metadata={},
        checkpoint_output_var_metadata={},
        strict=False,
        freeze_weights=False,
    )

    model._load_state_dict(cfg)


def test_load_state_dict_strict_true_missing_keys_raises(tmp_path):
    model = ConcreteModel(ConcreteModelConfig())

    checkpoint = {
        "module": {
            "model.linear.bias": torch.zeros_like(model.linear.bias),
        }
    }

    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)

    cfg = CheckpointConfig(
        load_path=checkpoint_path,
        checkpoint_input_shape=np.array([2]),
        checkpoint_output_shape=np.array([2]),
        checkpoint_input_var_metadata={},
        checkpoint_output_var_metadata={},
        strict=True,
        freeze_weights=False,
    )

    with pytest.raises(RuntimeError):
        model._load_state_dict(cfg)
