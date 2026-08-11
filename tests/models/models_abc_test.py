from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

import cccma_ppp.architectures.models_abc as module
from cccma_ppp.generic.runtime import RuntimeContext
from cccma_ppp.architectures.models_abc import (
    CheckpointConfig,
    GENERATORConfig,
    cVAEmodelConfigABC,
    cVAEmodelsABC,
    deterministicmodelsABC,
    flowABC,
    modelABC,
    modelConfigABC,
    weights_init,
)


@pytest.fixture(autouse=True)
def reset_runtime_context():
    original_input_metadata = RuntimeContext.INPUT_VAR_METADATA
    original_target_metadata = RuntimeContext.TARGET_VAR_METADATA

    RuntimeContext.INPUT_VAR_METADATA = {}
    RuntimeContext.TARGET_VAR_METADATA = {}

    yield

    RuntimeContext.INPUT_VAR_METADATA = original_input_metadata
    RuntimeContext.TARGET_VAR_METADATA = original_target_metadata


def make_checkpoint_config(
    load_path,
    *,
    input_shape=(2, 3),
    output_shape=(1, 3),
    input_metadata=None,
    output_metadata=None,
    strict=True,
    freeze_weights=False,
):
    return CheckpointConfig(
        load_path=load_path,
        checkpoint_input_shape=np.asarray(input_shape),
        checkpoint_output_shape=np.asarray(output_shape),
        checkpoint_input_var_metadata=(
            {} if input_metadata is None else input_metadata
        ),
        checkpoint_output_var_metadata=(
            {} if output_metadata is None else output_metadata
        ),
        strict=strict,
        freeze_weights=freeze_weights,
    )


class DummyConfig(modelConfigABC):
    activation = "relu"
    NUM_INPUT_DIMS = 2
    NUM_OUTPUT_DIMS = 2
    GENERATOR = None

    def build(
        self,
        input_shape,
        output_shape=None,
        added_features_dim=None,
        **kwargs,
    ):
        return DummyModel(config=self)


class SecondDummyConfig(modelConfigABC):
    activation = "relu"
    NUM_INPUT_DIMS = 2
    NUM_OUTPUT_DIMS = 2
    GENERATOR = None

    def build(
        self,
        input_shape,
        output_shape=None,
        added_features_dim=None,
        **kwargs,
    ):
        return DummyModel(config=self)


class DummyModel(modelABC):
    def __init__(self, config=None):
        super().__init__()
        self.config = DummyConfig() if config is None else config
        self.linear = nn.Linear(2, 2)

    def forward(self, value):
        return self.linear(value)


class EmptyModel(modelABC):
    def __init__(self):
        super().__init__()
        self.config = DummyConfig()

    def forward(self, value):
        return value


class BufferModel(modelABC):
    def __init__(self):
        super().__init__()
        self.config = DummyConfig()
        self.register_buffer(
            "buffer_value",
            torch.zeros(1),
        )

    def forward(self, value):
        return value


class DummyDeterministicModel(deterministicmodelsABC):
    def __init__(self):
        super().__init__()
        self.config = DummyConfig()
        self.linear = nn.Linear(2, 2)

    def forward(self, request):
        return SimpleNamespace(
            output=self.linear(request.input),
        )


class DummyCvaeConfig(cVAEmodelConfigABC):
    activation = "relu"
    NUM_INPUT_DIMS = 2
    NUM_OUTPUT_DIMS = 2
    GENERATOR = None

    def __init__(
        self,
        *,
        latent_size=4,
        condition_embedding_size=4,
        condition_dependant_latent=True,
    ):
        self.latent_size = latent_size
        self.condition_embedding_size = condition_embedding_size
        self.condition_dependant_latent = condition_dependant_latent
        self.condition_embedding_dims = [4]
        self.condemb_to_decoder = True

    def build(
        self,
        input_shape,
        output_shape=None,
        added_features_dim=None,
        **kwargs,
    ):
        return DummyCvaeModel(config=self)


class DummyCvaeModel(cVAEmodelsABC):
    def __init__(self, config=None):
        super().__init__()
        self.config = DummyCvaeConfig() if config is None else config
        self.linear = nn.Linear(2, 2)

    def forward(self, request):
        return SimpleNamespace(
            output=self.linear(request.target),
        )

    def predict(self, request):
        return SimpleNamespace(
            output=self.linear(request.condition),
        )

    def _recognition(self, value=None):
        return value, value

    def _condition(self, value=None):
        return value, None

    def _generate(self, value=None):
        return value


class DummyFlow(flowABC):
    def forward(self, value, condition=None):
        if condition is None:
            return value
        return value + condition

    def inverse(self, value, condition=None):
        if condition is None:
            return value
        return value - condition


def test_generator_config_defaults():
    config = GENERATORConfig()

    assert config.noise_level == "full"
    assert config.num_training_noise_samples == 10
    assert config.num_validation_noise_samples == 10


@pytest.mark.pruned
def test_generator_config_uses_training_samples_as_validation_default():
    config = GENERATORConfig(
        num_training_noise_samples=7,
        num_validation_noise_samples=None,
    )

    assert config.num_training_noise_samples == 7
    assert config.num_validation_noise_samples == 7


def test_generator_config_preserves_explicit_validation_samples():
    config = GENERATORConfig(
        num_training_noise_samples=7,
        num_validation_noise_samples=3,
    )

    assert config.num_training_noise_samples == 7
    assert config.num_validation_noise_samples == 3


@pytest.mark.parametrize(
    "noise_level",
    [
        "full",
        "medium",
        "low",
    ],
)
def test_generator_config_preserves_noise_level(noise_level):
    config = GENERATORConfig(
        noise_level=noise_level,
    )

    assert config.noise_level == noise_level


@pytest.mark.pruned
def test_validate_checkpoint_compatibility_without_checkpoint():
    model = DummyModel()
    model.config.checkpoint_config = None

    result = model._validate_checkpoint_compatibility(
        input_shape=np.asarray([2, 3]),
        output_shape=np.asarray([1, 3]),
    )

    assert result is None


def test_validate_checkpoint_compatibility_matching_values():
    RuntimeContext.INPUT_VAR_METADATA = {
        "input": "metadata",
    }
    RuntimeContext.TARGET_VAR_METADATA = {
        "target": "metadata",
    }

    model = DummyModel()
    model.config.checkpoint_config = make_checkpoint_config(
        "checkpoint.pt",
        input_metadata={
            "input": "metadata",
        },
        output_metadata={
            "target": "metadata",
        },
    )

    result = model._validate_checkpoint_compatibility(
        input_shape=np.asarray([2, 3]),
        output_shape=np.asarray([1, 3]),
    )

    assert result is None


@pytest.mark.pruned
def test_validate_checkpoint_compatibility_accepts_tuple_shapes():
    model = DummyModel()
    model.config.checkpoint_config = make_checkpoint_config(
        "checkpoint.pt",
    )

    result = model._validate_checkpoint_compatibility(
        input_shape=(2, 3),
        output_shape=(1, 3),
    )

    assert result is None


def test_validate_checkpoint_compatibility_rejects_input_shape():
    model = DummyModel()
    model.config.checkpoint_config = make_checkpoint_config(
        "checkpoint.pt",
    )

    with pytest.raises(
        RuntimeError,
        match="Requested input shape.*does not match checkpoint input shape",
    ):
        model._validate_checkpoint_compatibility(
            input_shape=np.asarray([9, 9]),
            output_shape=np.asarray([1, 3]),
        )


def test_validate_checkpoint_compatibility_rejects_output_shape():
    model = DummyModel()
    model.config.checkpoint_config = make_checkpoint_config(
        "checkpoint.pt",
    )

    with pytest.raises(
        RuntimeError,
        match="Requested output shape.*does not match checkpoint output shape",
    ):
        model._validate_checkpoint_compatibility(
            input_shape=np.asarray([2, 3]),
            output_shape=np.asarray([9, 9]),
        )


@pytest.mark.pruned
def test_validate_checkpoint_compatibility_rejects_input_metadata():
    RuntimeContext.INPUT_VAR_METADATA = {
        "current": "input",
    }

    model = DummyModel()
    model.config.checkpoint_config = make_checkpoint_config(
        "checkpoint.pt",
        input_metadata={
            "checkpoint": "input",
        },
    )

    with pytest.raises(
        RuntimeError,
        match="Checkpoint input metadata is incompatible",
    ):
        model._validate_checkpoint_compatibility(
            input_shape=np.asarray([2, 3]),
            output_shape=np.asarray([1, 3]),
        )


@pytest.mark.pruned
def test_validate_checkpoint_compatibility_rejects_target_metadata():
    RuntimeContext.TARGET_VAR_METADATA = {
        "current": "target",
    }

    model = DummyModel()
    model.config.checkpoint_config = make_checkpoint_config(
        "checkpoint.pt",
        output_metadata={
            "checkpoint": "target",
        },
    )

    with pytest.raises(
        RuntimeError,
        match="Checkpoint target metadata is incompatible",
    ):
        model._validate_checkpoint_compatibility(
            input_shape=np.asarray([2, 3]),
            output_shape=np.asarray([1, 3]),
        )


@pytest.mark.pruned
def test_input_shape_validation_precedes_other_validation():
    RuntimeContext.INPUT_VAR_METADATA = {
        "current": "input",
    }
    RuntimeContext.TARGET_VAR_METADATA = {
        "current": "target",
    }

    model = DummyModel()
    model.config.checkpoint_config = make_checkpoint_config(
        "checkpoint.pt",
        output_shape=(9, 9),
        input_metadata={
            "checkpoint": "input",
        },
        output_metadata={
            "checkpoint": "target",
        },
    )

    with pytest.raises(
        RuntimeError,
        match="Requested input shape",
    ):
        model._validate_checkpoint_compatibility(
            input_shape=np.asarray([7, 7]),
            output_shape=np.asarray([8, 8]),
        )


@pytest.mark.pruned
def test_output_shape_validation_precedes_metadata_validation():
    RuntimeContext.INPUT_VAR_METADATA = {
        "current": "input",
    }
    RuntimeContext.TARGET_VAR_METADATA = {
        "current": "target",
    }

    model = DummyModel()
    model.config.checkpoint_config = make_checkpoint_config(
        "checkpoint.pt",
        output_shape=(9, 9),
        input_metadata={
            "checkpoint": "input",
        },
        output_metadata={
            "checkpoint": "target",
        },
    )

    with pytest.raises(
        RuntimeError,
        match="Requested output shape",
    ):
        model._validate_checkpoint_compatibility(
            input_shape=np.asarray([2, 3]),
            output_shape=np.asarray([8, 8]),
        )


@pytest.mark.pruned
def test_input_metadata_validation_precedes_target_metadata():
    RuntimeContext.INPUT_VAR_METADATA = {
        "current": "input",
    }
    RuntimeContext.TARGET_VAR_METADATA = {
        "current": "target",
    }

    model = DummyModel()
    model.config.checkpoint_config = make_checkpoint_config(
        "checkpoint.pt",
        input_metadata={
            "checkpoint": "input",
        },
        output_metadata={
            "checkpoint": "target",
        },
    )

    with pytest.raises(
        RuntimeError,
        match="Checkpoint input metadata",
    ):
        model._validate_checkpoint_compatibility(
            input_shape=np.asarray([2, 3]),
            output_shape=np.asarray([1, 3]),
        )


@pytest.mark.pruned
def test_get_device_from_parameter():
    model = DummyModel()

    assert model._get_device() == model.linear.weight.device


def test_get_device_from_buffer():
    model = BufferModel()

    assert model._get_device() == model.buffer_value.device


def test_get_device_cpu_when_model_has_no_parameters_or_buffers():
    model = EmptyModel()

    assert model._get_device() == torch.device("cpu")


@pytest.mark.pruned
def test_get_device_prefers_parameter_over_buffer():
    model = DummyModel()
    model.register_buffer(
        "extra_buffer",
        torch.zeros(1),
    )

    assert model._get_device() == model.linear.weight.device


@pytest.mark.pruned
def test_initialize_weights_xavier():
    model = DummyModel()

    with torch.no_grad():
        model.linear.weight.zero_()
        model.linear.bias.fill_(3.0)

    result = model._initialize_weights("xavier")

    assert result is None
    assert not torch.all(model.linear.weight == 0)
    torch.testing.assert_close(
        model.linear.bias,
        torch.zeros_like(model.linear.bias),
    )


@pytest.mark.pruned
def test_initialize_weights_truncated_normal():
    model = DummyModel()

    with torch.no_grad():
        model.linear.weight.zero_()
        model.linear.bias.fill_(3.0)

    model._initialize_weights("trunc_normal")

    assert not torch.all(model.linear.weight == 0)
    torch.testing.assert_close(
        model.linear.bias,
        torch.zeros_like(model.linear.bias),
    )


def test_initialize_weights_invalid_method():
    model = DummyModel()

    with pytest.raises(
        NotImplementedError,
        match="trunc_normal.*xavier",
    ):
        model._initialize_weights("invalid")


def test_load_state_dict_missing_file(tmp_path):
    model = DummyModel()
    config = make_checkpoint_config(
        tmp_path / "missing.pt",
    )

    with pytest.raises(
        FileNotFoundError,
        match="Checkpoint not found",
    ):
        model._load_state_dict(config)


def test_load_state_dict_success(tmp_path):
    model = DummyModel()
    expected_model = DummyModel()

    with torch.no_grad():
        expected_model.linear.weight.fill_(0.5)
        expected_model.linear.bias.fill_(0.25)

    checkpoint = {
        "module": {
            f"model.{key}": value.clone()
            for key, value in expected_model.state_dict().items()
        }
    }

    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)

    model._load_state_dict(make_checkpoint_config(path))

    torch.testing.assert_close(
        model.linear.weight,
        expected_model.linear.weight,
    )
    torch.testing.assert_close(
        model.linear.bias,
        expected_model.linear.bias,
    )


@pytest.mark.pruned
def test_load_state_dict_strips_model_prefix(tmp_path):
    model = DummyModel()

    checkpoint = {
        "module": {
            "model.linear.weight": torch.full_like(
                model.linear.weight,
                2.0,
            ),
            "model.linear.bias": torch.full_like(
                model.linear.bias,
                3.0,
            ),
        }
    }

    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)

    model._load_state_dict(make_checkpoint_config(path))

    torch.testing.assert_close(
        model.linear.weight,
        torch.full_like(model.linear.weight, 2.0),
    )
    torch.testing.assert_close(
        model.linear.bias,
        torch.full_like(model.linear.bias, 3.0),
    )


@pytest.mark.pruned
def test_load_state_dict_ignores_non_model_keys(tmp_path):
    model = DummyModel()

    checkpoint = {
        "module": {
            "model.linear.weight": torch.ones_like(
                model.linear.weight,
            ),
            "model.linear.bias": torch.zeros_like(
                model.linear.bias,
            ),
            "optimizer.state": torch.tensor(1),
            "unrelated": torch.tensor(2),
        }
    }

    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)

    model._load_state_dict(make_checkpoint_config(path))

    torch.testing.assert_close(
        model.linear.weight,
        torch.ones_like(model.linear.weight),
    )


@pytest.mark.pruned
def test_load_state_dict_strict_false_allows_missing_keys(tmp_path):
    model = DummyModel()

    checkpoint = {
        "module": {
            "model.linear.bias": torch.zeros_like(
                model.linear.bias,
            ),
        }
    }

    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)

    model._load_state_dict(
        make_checkpoint_config(
            path,
            strict=False,
        )
    )


@pytest.mark.pruned
def test_load_state_dict_strict_true_rejects_missing_keys(tmp_path):
    model = DummyModel()

    checkpoint = {
        "module": {
            "model.linear.bias": torch.zeros_like(
                model.linear.bias,
            ),
        }
    }

    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)

    with pytest.raises(RuntimeError):
        model._load_state_dict(
            make_checkpoint_config(
                path,
                strict=True,
            )
        )


@pytest.mark.pruned
def test_load_state_dict_without_model_keys_in_non_strict_mode(tmp_path):
    model = DummyModel()

    checkpoint = {
        "module": {
            "optimizer.state": torch.tensor(1),
        }
    }

    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)

    model._load_state_dict(
        make_checkpoint_config(
            path,
            strict=False,
        )
    )


@pytest.mark.pruned
def test_load_state_dict_uses_device_and_weights_only(
    monkeypatch,
    tmp_path,
):
    model = DummyModel()
    path = tmp_path / "checkpoint.pt"
    path.touch()

    captured = {}

    def fake_load(
        load_path,
        map_location,
        weights_only,
    ):
        captured["load_path"] = load_path
        captured["map_location"] = map_location
        captured["weights_only"] = weights_only

        return {
            "module": {
                f"model.{key}": value.clone()
                for key, value in model.state_dict().items()
            }
        }

    monkeypatch.setattr(
        torch,
        "load",
        fake_load,
    )

    model._load_state_dict(make_checkpoint_config(path))

    assert captured["load_path"] == path
    assert captured["map_location"] == model._get_device()
    assert captured["weights_only"] is True


@pytest.mark.pruned
def test_load_state_dict_passes_strict_value(
    monkeypatch,
    tmp_path,
):
    model = DummyModel()
    path = tmp_path / "checkpoint.pt"
    path.touch()

    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: {
            "module": {
                f"model.{key}": value.clone()
                for key, value in model.state_dict().items()
            }
        },
    )

    captured = {}

    def fake_load_state_dict(
        state_dict,
        strict,
    ):
        captured["state_dict"] = state_dict
        captured["strict"] = strict
        return SimpleNamespace(
            missing_keys=[],
            unexpected_keys=[],
        )

    monkeypatch.setattr(
        model,
        "load_state_dict",
        fake_load_state_dict,
    )

    model._load_state_dict(
        make_checkpoint_config(
            path,
            strict=False,
        )
    )

    assert captured["strict"] is False
    assert set(captured["state_dict"]) == {
        "linear.weight",
        "linear.bias",
    }


@pytest.mark.pruned
def test_load_state_dict_calls_gc_collect(
    monkeypatch,
    tmp_path,
):
    model = DummyModel()

    checkpoint = {
        "module": {
            f"model.{key}": value.clone() for key, value in model.state_dict().items()
        }
    }

    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)

    collect = MagicMock()
    monkeypatch.setattr(
        module.gc,
        "collect",
        collect,
    )

    model._load_state_dict(make_checkpoint_config(path))

    collect.assert_called_once_with()


def test_load_state_dict_freezes_weights(tmp_path):
    model = DummyModel()

    checkpoint = {
        "module": {
            f"model.{key}": value.clone() for key, value in model.state_dict().items()
        }
    }

    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)

    model._load_state_dict(
        make_checkpoint_config(
            path,
            freeze_weights=True,
        )
    )

    assert all(parameter.requires_grad is False for parameter in model.parameters())


@pytest.mark.pruned
def test_load_state_dict_does_not_freeze_weights_by_default(tmp_path):
    model = DummyModel()

    checkpoint = {
        "module": {
            f"model.{key}": value.clone() for key, value in model.state_dict().items()
        }
    }

    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)

    model._load_state_dict(
        make_checkpoint_config(
            path,
            freeze_weights=False,
        )
    )

    assert all(parameter.requires_grad is True for parameter in model.parameters())


def test_resolve_flow_settings_matching_latent_size():
    config = DummyCvaeConfig(
        latent_size=4,
        condition_embedding_size=4,
        condition_dependant_latent=True,
    )

    result = config._resolve_flow_settings(
        condition_dependant_flow=False,
    )

    assert result is config
    assert config.condition_dependant_flow is False


def test_resolve_flow_settings_rejects_mismatched_latent_size_without_flow():
    config = DummyCvaeConfig(
        latent_size=4,
        condition_embedding_size=3,
        condition_dependant_latent=True,
    )

    with pytest.raises(
        ValueError,
        match="condition embedding size.*must equal latent size",
    ):
        config._resolve_flow_settings(
            condition_dependant_flow=False,
        )


def test_resolve_flow_settings_allows_mismatch_with_conditioned_flow():
    config = DummyCvaeConfig(
        latent_size=4,
        condition_embedding_size=3,
        condition_dependant_latent=True,
    )

    result = config._resolve_flow_settings(
        condition_dependant_flow=True,
    )

    assert result is config
    assert config.condition_dependant_flow is True


def test_resolve_flow_settings_condition_independent_latent():
    config = DummyCvaeConfig(
        latent_size=4,
        condition_embedding_size=3,
        condition_dependant_latent=False,
    )

    result = config._resolve_flow_settings(
        condition_dependant_flow=False,
    )

    assert result is config
    assert config.condition_dependant_flow is False


@pytest.mark.parametrize(
    "module_factory",
    [
        lambda: nn.Linear(4, 3),
        lambda: nn.Conv1d(2, 3, kernel_size=3),
        lambda: nn.Conv2d(2, 3, kernel_size=3),
        lambda: nn.Conv3d(2, 3, kernel_size=3),
    ],
)
def test_weights_init_xavier_supported_modules(module_factory):
    layer = module_factory()

    with torch.no_grad():
        layer.weight.zero_()
        if layer.bias is not None:
            layer.bias.fill_(5.0)

    result = weights_init(
        layer,
        method="xavier",
    )

    assert result is None
    assert not torch.all(layer.weight == 0)

    if layer.bias is not None:
        torch.testing.assert_close(
            layer.bias,
            torch.zeros_like(layer.bias),
        )


@pytest.mark.parametrize(
    "module_factory",
    [
        lambda: nn.Linear(4, 3),
        lambda: nn.Conv1d(2, 3, kernel_size=3),
        lambda: nn.Conv2d(2, 3, kernel_size=3),
        lambda: nn.Conv3d(2, 3, kernel_size=3),
    ],
)
def test_weights_init_truncated_normal_supported_modules(
    module_factory,
):
    layer = module_factory()

    with torch.no_grad():
        layer.weight.zero_()

    weights_init(
        layer,
        method="trunc_normal",
    )

    assert not torch.all(layer.weight == 0)


@pytest.mark.pruned
def test_weights_init_xavier_calls_initializer(
    monkeypatch,
):
    layer = nn.Linear(4, 3)
    initializer = MagicMock()

    monkeypatch.setattr(
        nn.init,
        "xavier_uniform_",
        initializer,
    )

    weights_init(
        layer,
        method="xavier",
    )

    initializer.assert_called_once_with(layer.weight)


@pytest.mark.pruned
def test_weights_init_truncated_normal_calls_initializer(
    monkeypatch,
):
    layer = nn.Linear(4, 3)
    initializer = MagicMock()

    monkeypatch.setattr(
        module,
        "trunc_normal_",
        initializer,
    )

    weights_init(
        layer,
        method="trunc_normal",
    )

    initializer.assert_called_once_with(
        layer.weight,
        std=0.02,
    )


@pytest.mark.pruned
def test_weights_init_ignores_unsupported_module():
    layer = nn.BatchNorm1d(4)

    weight_before = layer.weight.detach().clone()
    bias_before = layer.bias.detach().clone()

    result = weights_init(
        layer,
        method="invalid",
    )

    assert result is None

    torch.testing.assert_close(
        layer.weight,
        weight_before,
    )
    torch.testing.assert_close(
        layer.bias,
        bias_before,
    )


@pytest.mark.pruned
def test_weights_init_rejects_invalid_method():
    layer = nn.Linear(2, 2)

    with pytest.raises(
        NotImplementedError,
        match="trunc_normal.*xavier",
    ):
        weights_init(
            layer,
            method="invalid",
        )


@pytest.mark.pruned
def test_weights_init_skips_frozen_weight():
    layer = nn.Linear(3, 3)
    layer.weight.requires_grad = False

    weight_before = layer.weight.detach().clone()

    weights_init(
        layer,
        method="xavier",
    )

    torch.testing.assert_close(
        layer.weight,
        weight_before,
    )


@pytest.mark.pruned
def test_weights_init_skips_frozen_bias():
    layer = nn.Linear(3, 3)
    layer.bias.requires_grad = False

    with torch.no_grad():
        layer.bias.fill_(4.0)

    bias_before = layer.bias.detach().clone()

    weights_init(
        layer,
        method="xavier",
    )

    torch.testing.assert_close(
        layer.bias,
        bias_before,
    )


@pytest.mark.pruned
def test_weights_init_zeroes_trainable_bias():
    layer = nn.Linear(3, 3)

    with torch.no_grad():
        layer.bias.fill_(4.0)

    weights_init(
        layer,
        method="xavier",
    )

    torch.testing.assert_close(
        layer.bias,
        torch.zeros_like(layer.bias),
    )


@pytest.mark.pruned
def test_weights_init_layer_without_bias():
    layer = nn.Linear(
        3,
        3,
        bias=False,
    )

    weights_init(
        layer,
        method="xavier",
    )

    assert layer.bias is None


@pytest.mark.pruned
def test_weights_init_frozen_weight_still_initializes_bias():
    layer = nn.Linear(3, 3)
    layer.weight.requires_grad = False

    with torch.no_grad():
        layer.bias.fill_(4.0)

    weight_before = layer.weight.detach().clone()

    weights_init(
        layer,
        method="xavier",
    )

    torch.testing.assert_close(
        layer.weight,
        weight_before,
    )
    torch.testing.assert_close(
        layer.bias,
        torch.zeros_like(layer.bias),
    )


class ConfigCompatibilityHarness(modelConfigABC):
    activation = "relu"
    NUM_INPUT_DIMS = 2
    NUM_OUTPUT_DIMS = 2
    GENERATOR = None

    def __init__(self, checkpoint_config=None):
        self.config = SimpleNamespace(
            checkpoint_config=checkpoint_config,
        )

    def build(
        self,
        input_shape,
        output_shape=None,
        added_features_dim=None,
        **kwargs,
    ):
        return None


def test_config_level_checkpoint_validation_without_checkpoint():
    config = ConfigCompatibilityHarness(
        checkpoint_config=None,
    )

    result = config._validate_checkpoint_compatibility(
        input_shape=np.asarray([2, 3]),
        output_shape=np.asarray([1, 3]),
    )

    assert result is None


@pytest.mark.pruned
def test_config_level_checkpoint_validation_matching_values():
    RuntimeContext.INPUT_VAR_METADATA = {
        "input": "metadata",
    }
    RuntimeContext.TARGET_VAR_METADATA = {
        "target": "metadata",
    }

    checkpoint = make_checkpoint_config(
        "checkpoint.pt",
        input_metadata={
            "input": "metadata",
        },
        output_metadata={
            "target": "metadata",
        },
    )
    config = ConfigCompatibilityHarness(checkpoint)

    result = config._validate_checkpoint_compatibility(
        input_shape=(2, 3),
        output_shape=(1, 3),
    )

    assert result is None


def test_config_level_checkpoint_validation_rejects_input_shape():
    config = ConfigCompatibilityHarness(make_checkpoint_config("checkpoint.pt"))

    with pytest.raises(
        RuntimeError,
        match="Requested input shape.*checkpoint input shape",
    ):
        config._validate_checkpoint_compatibility(
            input_shape=(9, 9),
            output_shape=(1, 3),
        )


def test_config_level_checkpoint_validation_rejects_output_shape():
    config = ConfigCompatibilityHarness(make_checkpoint_config("checkpoint.pt"))

    with pytest.raises(
        RuntimeError,
        match="Requested output shape.*checkpoint output shape",
    ):
        config._validate_checkpoint_compatibility(
            input_shape=(2, 3),
            output_shape=(9, 9),
        )


@pytest.mark.pruned
def test_config_level_checkpoint_validation_rejects_input_metadata():
    RuntimeContext.INPUT_VAR_METADATA = {
        "current": "input",
    }

    config = ConfigCompatibilityHarness(
        make_checkpoint_config(
            "checkpoint.pt",
            input_metadata={
                "checkpoint": "input",
            },
        )
    )

    with pytest.raises(
        RuntimeError,
        match="Checkpoint input metadata is incompatible",
    ):
        config._validate_checkpoint_compatibility(
            input_shape=(2, 3),
            output_shape=(1, 3),
        )


def test_config_level_checkpoint_validation_rejects_target_metadata():
    RuntimeContext.TARGET_VAR_METADATA = {
        "current": "target",
    }

    config = ConfigCompatibilityHarness(
        make_checkpoint_config(
            "checkpoint.pt",
            output_metadata={
                "checkpoint": "target",
            },
        )
    )

    with pytest.raises(
        RuntimeError,
        match="Checkpoint target metadata is incompatible",
    ):
        config._validate_checkpoint_compatibility(
            input_shape=(2, 3),
            output_shape=(1, 3),
        )


@pytest.mark.pruned
def test_config_level_input_shape_check_precedes_output_shape_check():
    config = ConfigCompatibilityHarness(
        make_checkpoint_config(
            "checkpoint.pt",
            input_shape=(2, 3),
            output_shape=(1, 3),
        )
    )

    with pytest.raises(
        RuntimeError,
        match="Requested input shape",
    ):
        config._validate_checkpoint_compatibility(
            input_shape=(8, 8),
            output_shape=(9, 9),
        )


@pytest.mark.pruned
def test_config_level_output_shape_check_precedes_metadata_checks():
    RuntimeContext.INPUT_VAR_METADATA = {
        "current": "input",
    }
    RuntimeContext.TARGET_VAR_METADATA = {
        "current": "target",
    }

    config = ConfigCompatibilityHarness(
        make_checkpoint_config(
            "checkpoint.pt",
            input_metadata={
                "checkpoint": "input",
            },
            output_metadata={
                "checkpoint": "target",
            },
        )
    )

    with pytest.raises(
        RuntimeError,
        match="Requested output shape",
    ):
        config._validate_checkpoint_compatibility(
            input_shape=(2, 3),
            output_shape=(9, 9),
        )


@pytest.mark.pruned
def test_config_level_input_metadata_check_precedes_target_metadata():
    RuntimeContext.INPUT_VAR_METADATA = {
        "current": "input",
    }
    RuntimeContext.TARGET_VAR_METADATA = {
        "current": "target",
    }

    config = ConfigCompatibilityHarness(
        make_checkpoint_config(
            "checkpoint.pt",
            input_metadata={
                "checkpoint": "input",
            },
            output_metadata={
                "checkpoint": "target",
            },
        )
    )

    with pytest.raises(
        RuntimeError,
        match="Checkpoint input metadata",
    ):
        config._validate_checkpoint_compatibility(
            input_shape=(2, 3),
            output_shape=(1, 3),
        )


@pytest.mark.pruned
def test_get_device_parameter_branch_does_not_inspect_buffers(
    monkeypatch,
):
    model = DummyModel()

    def fail_buffers(*args, **kwargs):
        raise AssertionError("Buffers should not be inspected.")

    monkeypatch.setattr(
        model,
        "buffers",
        fail_buffers,
    )

    assert model._get_device() == model.linear.weight.device


@pytest.mark.pruned
def test_get_device_buffer_branch_after_empty_parameter_iterator(
    monkeypatch,
):
    model = BufferModel()
    parameter_calls = []

    def empty_parameters(*args, **kwargs):
        parameter_calls.append(True)
        return iter(())

    monkeypatch.setattr(
        model,
        "parameters",
        empty_parameters,
    )

    assert model._get_device() == model.buffer_value.device
    assert parameter_calls == [True]


@pytest.mark.pruned
def test_get_device_cpu_branch_after_empty_iterators(
    monkeypatch,
):
    model = EmptyModel()

    monkeypatch.setattr(
        model,
        "parameters",
        lambda *args, **kwargs: iter(()),
    )
    monkeypatch.setattr(
        model,
        "buffers",
        lambda *args, **kwargs: iter(()),
    )

    assert model._get_device() == torch.device("cpu")


@pytest.mark.pruned
def test_load_state_dict_accepts_string_checkpoint_path(tmp_path):
    model = DummyModel()
    path = tmp_path / "checkpoint.pt"

    torch.save(
        {
            "module": {
                f"model.{key}": value.clone()
                for key, value in model.state_dict().items()
            }
        },
        path,
    )

    config = make_checkpoint_config(str(path))

    model._load_state_dict(config)

    assert isinstance(config.load_path, str)


@pytest.mark.pruned
def test_load_state_dict_filters_mixed_prefixed_keys(
    monkeypatch,
    tmp_path,
):
    model = DummyModel()
    path = tmp_path / "checkpoint.pt"
    path.touch()

    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: {
            "module": {
                "model.linear.weight": torch.ones_like(model.linear.weight),
                "model.linear.bias": torch.zeros_like(model.linear.bias),
                "linear.weight": torch.full_like(
                    model.linear.weight,
                    9.0,
                ),
                "optimizer.state": torch.tensor(1),
            }
        },
    )

    captured = {}

    def fake_load_state_dict(state_dict, strict):
        captured["state_dict"] = state_dict
        captured["strict"] = strict
        return SimpleNamespace(
            missing_keys=[],
            unexpected_keys=[],
        )

    monkeypatch.setattr(
        model,
        "load_state_dict",
        fake_load_state_dict,
    )

    model._load_state_dict(make_checkpoint_config(path))

    assert captured["strict"] is True
    assert set(captured["state_dict"]) == {
        "linear.weight",
        "linear.bias",
    }
    torch.testing.assert_close(
        captured["state_dict"]["linear.weight"],
        torch.ones_like(model.linear.weight),
    )


@pytest.mark.pruned
def test_load_state_dict_freezes_every_parameter(
    monkeypatch,
    tmp_path,
):
    class MultiParameterModel(modelABC):
        def __init__(self):
            super().__init__()
            self.config = DummyConfig()
            self.first = nn.Linear(2, 3)
            self.second = nn.Linear(3, 1)

        def forward(self, value):
            return self.second(self.first(value))

    model = MultiParameterModel()
    path = tmp_path / "checkpoint.pt"

    torch.save(
        {
            "module": {
                f"model.{key}": value.clone()
                for key, value in model.state_dict().items()
            }
        },
        path,
    )

    model._load_state_dict(
        make_checkpoint_config(
            path,
            freeze_weights=True,
        )
    )

    assert len(list(model.parameters())) == 4
    assert all(parameter.requires_grad is False for parameter in model.parameters())


@pytest.mark.pruned
def test_load_state_dict_freeze_branch_handles_model_without_parameters(
    monkeypatch,
    tmp_path,
):
    model = EmptyModel()
    path = tmp_path / "checkpoint.pt"
    path.touch()

    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: {
            "module": {},
        },
    )

    model._load_state_dict(
        make_checkpoint_config(
            path,
            freeze_weights=True,
        )
    )


@pytest.mark.pruned
def test_load_state_dict_propagates_missing_module_key(tmp_path):
    model = DummyModel()
    path = tmp_path / "checkpoint.pt"

    torch.save(
        {
            "not_module": {},
        },
        path,
    )

    with pytest.raises(KeyError):
        model._load_state_dict(make_checkpoint_config(path))


@pytest.mark.pruned
def test_load_state_dict_propagates_torch_load_error(
    monkeypatch,
    tmp_path,
):
    model = DummyModel()
    path = tmp_path / "checkpoint.pt"
    path.touch()

    monkeypatch.setattr(
        torch,
        "load",
        MagicMock(side_effect=RuntimeError("checkpoint load failed")),
    )

    with pytest.raises(
        RuntimeError,
        match="checkpoint load failed",
    ):
        model._load_state_dict(make_checkpoint_config(path))


@pytest.mark.pruned
def test_load_state_dict_does_not_collect_after_load_failure(
    monkeypatch,
    tmp_path,
):
    model = DummyModel()
    path = tmp_path / "checkpoint.pt"
    path.touch()

    monkeypatch.setattr(
        torch,
        "load",
        MagicMock(side_effect=RuntimeError("load failed")),
    )
    collect = MagicMock()
    monkeypatch.setattr(
        module.gc,
        "collect",
        collect,
    )

    with pytest.raises(
        RuntimeError,
        match="load failed",
    ):
        model._load_state_dict(make_checkpoint_config(path))

    collect.assert_not_called()


@pytest.mark.pruned
def test_load_state_dict_does_not_freeze_before_successful_load(
    monkeypatch,
    tmp_path,
):
    model = DummyModel()
    path = tmp_path / "checkpoint.pt"
    path.touch()

    monkeypatch.setattr(
        torch,
        "load",
        MagicMock(side_effect=RuntimeError("load failed")),
    )

    with pytest.raises(
        RuntimeError,
        match="load failed",
    ):
        model._load_state_dict(
            make_checkpoint_config(
                path,
                freeze_weights=True,
            )
        )

    assert all(parameter.requires_grad is True for parameter in model.parameters())


@pytest.mark.pruned
def test_resolve_flow_settings_default_argument():
    config = DummyCvaeConfig(
        latent_size=4,
        condition_embedding_size=4,
        condition_dependant_latent=True,
    )

    result = config._resolve_flow_settings()

    assert result is config
    assert config.condition_dependant_flow is False


@pytest.mark.pruned
def test_resolve_flow_settings_condition_independent_with_flow():
    config = DummyCvaeConfig(
        latent_size=4,
        condition_embedding_size=1,
        condition_dependant_latent=False,
    )

    result = config._resolve_flow_settings(
        condition_dependant_flow=True,
    )

    assert result is config
    assert config.condition_dependant_flow is True


@pytest.mark.pruned
def test_weights_init_default_method_is_xavier(
    monkeypatch,
):
    layer = nn.Linear(3, 2)
    initializer = MagicMock()

    monkeypatch.setattr(
        nn.init,
        "xavier_uniform_",
        initializer,
    )

    weights_init(layer)

    initializer.assert_called_once_with(layer.weight)


@pytest.mark.pruned
def test_weights_init_unsupported_module_does_not_validate_method():
    module_value = nn.ReLU()

    result = weights_init(
        module_value,
        method="definitely-invalid",
    )

    assert result is None


@pytest.mark.pruned
def test_weights_init_supported_module_with_none_weight():
    class NoWeightLinear(nn.Linear):
        def __init__(self):
            super().__init__(2, 2)
            self.register_parameter(
                "weight",
                None,
            )

    layer = NoWeightLinear()

    with torch.no_grad():
        layer.bias.fill_(3.0)

    weights_init(
        layer,
        method="xavier",
    )

    assert layer.weight is None
    torch.testing.assert_close(
        layer.bias,
        torch.zeros_like(layer.bias),
    )


@pytest.mark.pruned
def test_weights_init_supported_module_with_none_bias():
    layer = nn.Linear(
        2,
        2,
        bias=False,
    )

    before = layer.weight.detach().clone()

    weights_init(
        layer,
        method="xavier",
    )

    assert layer.bias is None
    assert not torch.equal(
        layer.weight,
        before,
    )


@pytest.mark.pruned
def test_weights_init_frozen_weight_and_trainable_bias(
    monkeypatch,
):
    layer = nn.Linear(3, 3)
    layer.weight.requires_grad = False

    with torch.no_grad():
        layer.bias.fill_(5.0)

    initializer = MagicMock()
    monkeypatch.setattr(
        nn.init,
        "xavier_uniform_",
        initializer,
    )

    weights_init(
        layer,
        method="xavier",
    )

    initializer.assert_not_called()
    torch.testing.assert_close(
        layer.bias,
        torch.zeros_like(layer.bias),
    )


@pytest.mark.pruned
def test_weights_init_trainable_weight_and_frozen_bias(
    monkeypatch,
):
    layer = nn.Linear(3, 3)
    layer.bias.requires_grad = False

    with torch.no_grad():
        layer.bias.fill_(5.0)

    initializer = MagicMock()
    monkeypatch.setattr(
        nn.init,
        "xavier_uniform_",
        initializer,
    )

    weights_init(
        layer,
        method="xavier",
    )

    initializer.assert_called_once_with(layer.weight)
    torch.testing.assert_close(
        layer.bias,
        torch.full_like(layer.bias, 5.0),
    )


def test_weights_init_frozen_weight_and_bias_call_no_initializers(
    monkeypatch,
):
    layer = nn.Linear(3, 3)
    layer.weight.requires_grad = False
    layer.bias.requires_grad = False

    xavier = MagicMock()
    constant = MagicMock()

    monkeypatch.setattr(
        nn.init,
        "xavier_uniform_",
        xavier,
    )
    monkeypatch.setattr(
        nn.init,
        "constant_",
        constant,
    )

    weights_init(
        layer,
        method="xavier",
    )

    xavier.assert_not_called()
    constant.assert_not_called()


@pytest.mark.pruned
def test_weights_init_trainable_bias_calls_constant_initializer(
    monkeypatch,
):
    layer = nn.Linear(3, 3)
    constant = MagicMock()

    monkeypatch.setattr(
        nn.init,
        "constant_",
        constant,
    )

    weights_init(
        layer,
        method="xavier",
    )

    constant.assert_called_once_with(
        layer.bias,
        0,
    )


@pytest.mark.pruned
def test_weights_init_no_bias_does_not_call_constant_initializer(
    monkeypatch,
):
    layer = nn.Linear(
        3,
        3,
        bias=False,
    )
    constant = MagicMock()

    monkeypatch.setattr(
        nn.init,
        "constant_",
        constant,
    )

    weights_init(
        layer,
        method="xavier",
    )

    constant.assert_not_called()
