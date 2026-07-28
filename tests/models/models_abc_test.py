import dataclasses
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

import cccma_ppp.models.models_abc as module
from cccma_ppp.generic.runtime import RuntimeContext
from cccma_ppp.models.models_abc import (
    CheckpointConfig,
    DeterministicRequest,
    GENERATORConfig,
    cVAEForwardRequest,
    cVAEmodelConfigABC,
    cVAEmodelsABC,
    cVAEPredictRequest,
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


# ---------------------------------------------------------------------------
# GENERATORConfig
# ---------------------------------------------------------------------------


def test_generator_config_defaults():
    config = GENERATORConfig()

    assert config.noise_level == "full"
    assert config.num_training_noise_samples == 10
    assert config.num_validation_noise_samples == 10


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


# ---------------------------------------------------------------------------
# CheckpointConfig
# ---------------------------------------------------------------------------


def test_checkpoint_config_defaults(tmp_path):
    path = tmp_path / "checkpoint.pt"

    config = make_checkpoint_config(path)

    assert config.load_path == path
    assert np.array_equal(
        config.checkpoint_input_shape,
        np.asarray([2, 3]),
    )
    assert np.array_equal(
        config.checkpoint_output_shape,
        np.asarray([1, 3]),
    )
    assert config.checkpoint_input_var_metadata == {}
    assert config.checkpoint_output_var_metadata == {}
    assert config.strict is True
    assert config.freeze_weights is False


def test_checkpoint_config_accepts_string_path():
    config = make_checkpoint_config(
        "checkpoint.pt",
    )

    assert config.load_path == "checkpoint.pt"


def test_checkpoint_config_preserves_options(tmp_path):
    config = make_checkpoint_config(
        tmp_path / "checkpoint.pt",
        strict=False,
        freeze_weights=True,
        input_metadata={"input": "metadata"},
        output_metadata={"target": "metadata"},
    )

    assert config.strict is False
    assert config.freeze_weights is True
    assert config.checkpoint_input_var_metadata == {
        "input": "metadata",
    }
    assert config.checkpoint_output_var_metadata == {
        "target": "metadata",
    }


# ---------------------------------------------------------------------------
# Abstract classes
# ---------------------------------------------------------------------------


def test_flow_abc_cannot_be_instantiated():
    with pytest.raises(TypeError):
        flowABC()


def test_model_config_abc_cannot_be_instantiated():
    with pytest.raises(TypeError):
        modelConfigABC()


def test_model_abc_cannot_be_instantiated():
    with pytest.raises(TypeError):
        modelABC()


def test_deterministic_model_abc_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        deterministicmodelsABC()


def test_cvae_model_abc_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        cVAEmodelsABC()


def test_dummy_flow_forward_without_condition():
    flow = DummyFlow()
    value = torch.ones(2, 3)

    result = flow.forward(value)

    assert result is value


def test_dummy_flow_forward_with_condition():
    flow = DummyFlow()
    value = torch.ones(2, 3)
    condition = torch.full((2, 3), 2.0)

    result = flow.forward(
        value,
        condition,
    )

    torch.testing.assert_close(
        result,
        torch.full((2, 3), 3.0),
    )


def test_dummy_flow_inverse_with_condition():
    flow = DummyFlow()
    value = torch.full((2, 3), 3.0)
    condition = torch.full((2, 3), 2.0)

    result = flow.inverse(
        value,
        condition,
    )

    torch.testing.assert_close(
        result,
        torch.ones(2, 3),
    )


# ---------------------------------------------------------------------------
# modelConfigABC
# ---------------------------------------------------------------------------


def test_model_config_subclass_gets_checkpoint_attribute():
    assert hasattr(DummyConfig, "checkpoint_config")
    assert DummyConfig.checkpoint_config is None


def test_each_model_config_subclass_gets_checkpoint_attribute():
    assert DummyConfig.checkpoint_config is None
    assert SecondDummyConfig.checkpoint_config is None


def test_add_checkpoint_config():
    config = DummyConfig()
    checkpoint = make_checkpoint_config(
        "checkpoint.pt",
    )

    result = config._add_checkpoint_config(checkpoint)

    assert result is None
    assert config.checkpoint_config is checkpoint


def test_dummy_config_build_returns_model():
    config = DummyConfig()

    model = config.build(
        input_shape=np.asarray([2, 3]),
        output_shape=np.asarray([1, 3]),
    )

    assert isinstance(model, DummyModel)
    assert model.config is config


# ---------------------------------------------------------------------------
# Checkpoint compatibility
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


def test_get_device_from_parameter():
    model = DummyModel()

    assert model._get_device() == model.linear.weight.device


def test_get_device_from_buffer():
    model = BufferModel()

    assert model._get_device() == model.buffer_value.device


def test_get_device_cpu_when_model_has_no_parameters_or_buffers():
    model = EmptyModel()

    assert model._get_device() == torch.device("cpu")


def test_get_device_prefers_parameter_over_buffer():
    model = DummyModel()
    model.register_buffer(
        "extra_buffer",
        torch.zeros(1),
    )

    assert model._get_device() == model.linear.weight.device


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_get_device_from_cuda_parameter():
    model = DummyModel().cuda()

    assert model._get_device().type == "cuda"


# ---------------------------------------------------------------------------
# Weight initialization through modelABC
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# State-dict loading
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Request data classes
# ---------------------------------------------------------------------------


def test_deterministic_request_defaults():
    input_value = torch.randn(2, 3)

    request = DeterministicRequest(
        input=input_value,
    )

    assert request.input is input_value
    assert request.input_mask is None
    assert request.added_features is None
    assert request.output_sample_size == 1


def test_deterministic_request_preserves_values():
    input_value = torch.randn(2, 3)
    input_mask = torch.ones_like(input_value)
    features = torch.randn(2, 4)

    request = DeterministicRequest(
        input=input_value,
        input_mask=input_mask,
        added_features=features,
        output_sample_size=5,
    )

    assert request.input is input_value
    assert request.input_mask is input_mask
    assert request.added_features is features
    assert request.output_sample_size == 5


def test_cvae_forward_request_defaults():
    target = torch.randn(2, 3)
    condition = torch.randn(2, 4)

    request = cVAEForwardRequest(
        target=target,
        condition=condition,
    )

    assert request.target is target
    assert request.condition is condition
    assert request.target_mask is None
    assert request.condition_mask is None
    assert request.added_features is None
    assert request.sample_size == 1
    assert request.output_sample_size == 1
    assert request.min_posterior_variance is None


def test_cvae_forward_request_preserves_values():
    target = torch.randn(2, 3)
    condition = torch.randn(2, 4)
    target_mask = torch.ones_like(target)
    condition_mask = torch.ones_like(condition)
    features = torch.randn(2, 5)
    minimum = torch.tensor(-1.0)

    request = cVAEForwardRequest(
        target=target,
        condition=condition,
        target_mask=target_mask,
        condition_mask=condition_mask,
        added_features=features,
        sample_size=3,
        output_sample_size=4,
        min_posterior_variance=minimum,
    )

    assert request.target_mask is target_mask
    assert request.condition_mask is condition_mask
    assert request.added_features is features
    assert request.sample_size == 3
    assert request.output_sample_size == 4
    assert request.min_posterior_variance is minimum


def test_cvae_predict_request_defaults():
    condition = torch.randn(2, 4)

    request = cVAEPredictRequest(
        condition=condition,
    )

    assert request.condition is condition
    assert request.condition_mask is None
    assert request.added_features is None
    assert request.prior_flow is None
    assert request.latent_samples is None
    assert request.nstds == 1
    assert request.sample_size == 1
    assert request.output_sample_size == 1


def test_cvae_predict_request_preserves_values():
    condition = torch.randn(2, 4)
    condition_mask = torch.ones_like(condition)
    features = torch.randn(2, 5)
    flow = DummyFlow()
    latent = torch.randn(3, 2, 4)

    request = cVAEPredictRequest(
        condition=condition,
        condition_mask=condition_mask,
        added_features=features,
        prior_flow=flow,
        latent_samples=latent,
        nstds=2,
        sample_size=3,
        output_sample_size=4,
    )

    assert request.condition_mask is condition_mask
    assert request.added_features is features
    assert request.prior_flow is flow
    assert request.latent_samples is latent
    assert request.nstds == 2
    assert request.sample_size == 3
    assert request.output_sample_size == 4


# ---------------------------------------------------------------------------
# Deterministic and cVAE model base classes
# ---------------------------------------------------------------------------


def test_deterministic_model_sets_generative_modeling_false():
    model = DummyDeterministicModel()

    assert model.generative_modeling is False


def test_cvae_model_sets_generative_modeling_true():
    model = DummyCvaeModel()

    assert model.generative_modeling is True


def test_dummy_deterministic_model_forward():
    model = DummyDeterministicModel()
    request = DeterministicRequest(
        input=torch.randn(3, 2),
    )

    result = model.forward(request)

    assert result.output.shape == (3, 2)


def test_dummy_cvae_model_forward():
    model = DummyCvaeModel()
    request = cVAEForwardRequest(
        target=torch.randn(3, 2),
        condition=torch.randn(3, 2),
    )

    result = model.forward(request)

    assert result.output.shape == (3, 2)


def test_dummy_cvae_model_predict():
    model = DummyCvaeModel()
    request = cVAEPredictRequest(
        condition=torch.randn(3, 2),
    )

    result = model.predict(request)

    assert result.output.shape == (3, 2)


# ---------------------------------------------------------------------------
# cVAEmodelConfigABC flow settings
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# cVAEmodelsABC sampling
# ---------------------------------------------------------------------------


def test_cvae_sample_shape():
    model = DummyCvaeModel()
    mu = torch.zeros(2, 4)
    log_var = torch.zeros(2, 4)

    samples = model._sample(
        mu,
        log_var,
        sample_size=3,
    )

    assert samples.shape == (3, 2, 4)


def test_cvae_sample_calls_shared_sample(
    monkeypatch,
):
    model = DummyCvaeModel()
    mu = torch.ones(2, 4)
    log_var = torch.zeros(2, 4)
    expected = torch.full((5, 2, 4), 7.0)
    captured = {}

    def fake_sample(
        sample_mu,
        variance,
        sample_size,
        std,
    ):
        captured["mu"] = sample_mu
        captured["variance"] = variance
        captured["sample_size"] = sample_size
        captured["std"] = std
        return expected

    monkeypatch.setattr(
        module,
        "_sample",
        fake_sample,
    )

    result = model._sample(
        mu,
        log_var,
        sample_size=5,
        std=2.5,
    )

    assert result is expected
    assert captured["mu"] is mu
    assert captured["sample_size"] == 5
    assert captured["std"] == pytest.approx(2.5)

    torch.testing.assert_close(
        captured["variance"],
        torch.full_like(log_var, 1.0001),
    )


def test_cvae_sample_uses_exponential_log_variance(
    monkeypatch,
):
    model = DummyCvaeModel()
    mu = torch.zeros(1, 2)
    log_var = torch.log(
        torch.tensor(
            [
                [2.0, 4.0],
            ]
        )
    )
    captured = {}

    def fake_sample(
        sample_mu,
        variance,
        sample_size,
        std,
    ):
        captured["variance"] = variance
        return torch.zeros(sample_size, 1, 2)

    monkeypatch.setattr(
        module,
        "_sample",
        fake_sample,
    )

    model._sample(
        mu,
        log_var,
        sample_size=2,
    )

    torch.testing.assert_close(
        captured["variance"],
        torch.tensor(
            [
                [2.0001, 4.0001],
            ]
        ),
    )


# ---------------------------------------------------------------------------
# weights_init
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Additional branch coverage
# ---------------------------------------------------------------------------


def test_dummy_flow_inverse_without_condition():
    flow = DummyFlow()
    value = torch.ones(2, 3)

    result = flow.inverse(value)

    assert result is value


def test_model_config_checkpoint_attribute_is_independent_per_subclass():
    first = DummyConfig()
    second = SecondDummyConfig()
    checkpoint = make_checkpoint_config("checkpoint.pt")

    first._add_checkpoint_config(checkpoint)

    assert first.checkpoint_config is checkpoint
    assert second.checkpoint_config is None
    assert SecondDummyConfig.checkpoint_config is None


def test_new_model_config_subclass_resets_inherited_checkpoint_config():
    checkpoint = make_checkpoint_config("checkpoint.pt")
    DummyConfig.checkpoint_config = checkpoint

    class ChildConfig(DummyConfig):
        pass

    try:
        assert ChildConfig.checkpoint_config is None
    finally:
        DummyConfig.checkpoint_config = None


def test_add_checkpoint_config_replaces_existing_checkpoint():
    config = DummyConfig()
    first = make_checkpoint_config("first.pt")
    second = make_checkpoint_config("second.pt")

    config._add_checkpoint_config(first)
    config._add_checkpoint_config(second)

    assert config.checkpoint_config is second


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


def test_resolve_flow_settings_default_argument():
    config = DummyCvaeConfig(
        latent_size=4,
        condition_embedding_size=4,
        condition_dependant_latent=True,
    )

    result = config._resolve_flow_settings()

    assert result is config
    assert config.condition_dependant_flow is False


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


def test_cvae_sample_default_arguments(
    monkeypatch,
):
    model = DummyCvaeModel()
    mu = torch.zeros(2, 4)
    log_var = torch.zeros(2, 4)
    captured = {}

    def fake_sample(
        sample_mu,
        variance,
        sample_size,
        std,
    ):
        captured["mu"] = sample_mu
        captured["variance"] = variance
        captured["sample_size"] = sample_size
        captured["std"] = std
        return torch.zeros(1, 2, 4)

    monkeypatch.setattr(
        module,
        "_sample",
        fake_sample,
    )

    result = model._sample(
        mu,
        log_var,
    )

    assert result.shape == (1, 2, 4)
    assert captured["mu"] is mu
    assert captured["sample_size"] == 1
    assert captured["std"] == 1


def test_cvae_sample_preserves_dtype(
    monkeypatch,
):
    model = DummyCvaeModel()
    mu = torch.zeros(
        2,
        4,
        dtype=torch.float64,
    )
    log_var = torch.zeros_like(mu)
    captured = {}

    def fake_sample(
        sample_mu,
        variance,
        sample_size,
        std,
    ):
        captured["variance"] = variance
        return torch.zeros(
            sample_size,
            2,
            4,
            dtype=variance.dtype,
        )

    monkeypatch.setattr(
        module,
        "_sample",
        fake_sample,
    )

    result = model._sample(
        mu,
        log_var,
        sample_size=2,
    )

    assert captured["variance"].dtype == torch.float64
    assert result.dtype == torch.float64


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


def test_weights_init_unsupported_module_does_not_validate_method():
    module_value = nn.ReLU()

    result = weights_init(
        module_value,
        method="definitely-invalid",
    )

    assert result is None


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
