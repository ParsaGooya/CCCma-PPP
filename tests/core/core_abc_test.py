import numpy as np
import pytest
import torch
import torch.nn as nn

from cccma_ppp.core.core_abc import (
    GenerativeContext,
    moduleABC,
    moduleConfigABC,
)


class ConcreteModule(moduleABC):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.loss_function = None

    def init_loss_function(self, reconstruction_loss, **kwargs):
        self.loss_function = reconstruction_loss
        self.loss_kwargs = kwargs

    def _compute_loss(self):
        return "loss"

    def forward(self, x=None):
        if x is None:
            x = torch.ones(1, 2)
        return self.linear(x)

    def predict(self):
        return "predict"


class EmptyConcreteModule(moduleABC):
    def init_loss_function(self, reconstruction_loss, **kwargs):
        self.loss_function = reconstruction_loss

    def _compute_loss(self):
        return "loss"

    def forward(self):
        return "forward"

    def predict(self):
        return "predict"


class BufferOnlyConcreteModule(moduleABC):
    def __init__(self):
        super().__init__()
        self.register_buffer("buffer_value", torch.ones(1))

    def init_loss_function(self, reconstruction_loss, **kwargs):
        self.loss_function = reconstruction_loss

    def _compute_loss(self):
        return "loss"

    def forward(self):
        return "forward"

    def predict(self):
        return "predict"


class ConcreteModuleConfig(moduleConfigABC):
    def __init__(self):
        super().__init__()
        self.loaded_from_checkpoint = False

    def build(
        self,
        input_shape: np.ndarray,
        output_shape: np.ndarray | None = None,
        added_features_dim: int = None,
    ):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.added_features_dim = added_features_dim
        return ConcreteModule()

    def _load_from_checkpoint(self):
        self.loaded_from_checkpoint = True
        return "loaded"


@pytest.mark.pruned
def test_get_device_returns_parameter_device():
    module = ConcreteModule()

    expected_device = next(module.parameters()).device

    assert module._get_device() == expected_device


@pytest.mark.pruned
def test_get_device_returns_buffer_device_when_no_parameters():
    module = BufferOnlyConcreteModule()

    assert module._get_device() == module.buffer_value.device


@pytest.mark.pruned
def test_get_device_returns_cpu_without_parameters_or_buffers():
    module = EmptyConcreteModule()

    assert module._get_device() == torch.device("cpu")


def test_load_state_dict_missing_file_raises(tmp_path):
    module = ConcreteModule()

    missing_path = tmp_path / "missing.pt"

    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        module._load_state_dict(missing_path)


@pytest.mark.pruned
def test_load_state_dict_loads_checkpoint_strict_true(tmp_path):
    source = ConcreteModule()
    target = ConcreteModule()

    for parameter in source.parameters():
        nn.init.constant_(parameter, 0.25)

    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"module": source.state_dict()}, checkpoint_path)

    target._load_state_dict(checkpoint_path, strict=True)

    for key, value in target.state_dict().items():
        assert torch.allclose(value, source.state_dict()[key])


def test_load_state_dict_accepts_string_path(tmp_path):
    source = ConcreteModule()
    target = ConcreteModule()

    for parameter in source.parameters():
        nn.init.constant_(parameter, 0.5)

    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"module": source.state_dict()}, checkpoint_path)

    target._load_state_dict(str(checkpoint_path), strict=True)

    for key, value in target.state_dict().items():
        assert torch.allclose(value, source.state_dict()[key])


@pytest.mark.pruned
def test_load_state_dict_strict_false_allows_missing_keys(tmp_path):
    module = ConcreteModule()

    checkpoint_path = tmp_path / "partial_checkpoint.pt"
    torch.save(
        {
            "module": {
                "linear.bias": torch.zeros_like(module.linear.bias),
            }
        },
        checkpoint_path,
    )

    module._load_state_dict(checkpoint_path, strict=False)

    assert torch.allclose(module.linear.bias, torch.zeros_like(module.linear.bias))


@pytest.mark.pruned
def test_load_state_dict_strict_true_raises_on_missing_keys(tmp_path):
    module = ConcreteModule()

    checkpoint_path = tmp_path / "partial_checkpoint.pt"
    torch.save(
        {
            "module": {
                "linear.bias": torch.zeros_like(module.linear.bias),
            }
        },
        checkpoint_path,
    )

    with pytest.raises(RuntimeError):
        module._load_state_dict(checkpoint_path, strict=True)


@pytest.mark.pruned
def test_load_state_dict_strict_true_raises_on_unexpected_keys(tmp_path):
    module = ConcreteModule()

    checkpoint_path = tmp_path / "bad_checkpoint.pt"
    torch.save(
        {
            "module": {
                **module.state_dict(),
                "unexpected.weight": torch.ones(1),
            }
        },
        checkpoint_path,
    )

    with pytest.raises(RuntimeError):
        module._load_state_dict(checkpoint_path, strict=True)


@pytest.mark.pruned
def test_load_state_dict_strict_false_allows_unexpected_keys(tmp_path):
    module = ConcreteModule()

    checkpoint_path = tmp_path / "bad_checkpoint.pt"
    torch.save(
        {
            "module": {
                **module.state_dict(),
                "unexpected.weight": torch.ones(1),
            }
        },
        checkpoint_path,
    )

    module._load_state_dict(checkpoint_path, strict=False)


@pytest.mark.pruned
def test_load_state_dict_uses_module_key(tmp_path):
    module = ConcreteModule()

    checkpoint_path = tmp_path / "invalid_checkpoint.pt"
    torch.save({"not_module": module.state_dict()}, checkpoint_path)

    with pytest.raises(KeyError):
        module._load_state_dict(checkpoint_path)


@pytest.mark.pruned
def test_load_state_dict_preserves_loaded_values(tmp_path):
    module = ConcreteModule()

    checkpoint_state = {
        "linear.weight": torch.full_like(module.linear.weight, 2.0),
        "linear.bias": torch.full_like(module.linear.bias, -1.0),
    }

    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"module": checkpoint_state}, checkpoint_path)

    module._load_state_dict(checkpoint_path)

    assert torch.allclose(
        module.linear.weight, torch.full_like(module.linear.weight, 2.0)
    )
    assert torch.allclose(module.linear.bias, torch.full_like(module.linear.bias, -1.0))


def test_check_registered_raises_when_unregistered():
    class NotRegistered(moduleConfigABC):
        _type = None

        def build(self, *args, **kwargs):
            pass

        def _load_from_checkpoint(self):
            pass

    with pytest.raises(RuntimeError, match="has not been registered"):
        NotRegistered.check_registered()


def test_check_registered_success():
    class Registered(moduleConfigABC):
        _type = "registered"

        def build(self, *args, **kwargs):
            pass

        def _load_from_checkpoint(self):
            pass

    Registered.check_registered()


@pytest.mark.pruned
def test_check_registered_inherited_type():
    class Registered(moduleConfigABC):
        _type = "registered"

        def build(self, *args, **kwargs):
            pass

        def _load_from_checkpoint(self):
            pass

    Registered.check_registered()


@pytest.mark.pruned
def test_generative_context_without_module_uses_false_defaults():
    context = GenerativeContext()

    assert context.generator is False
    assert context.generative_modeling is False


@pytest.mark.pruned
def test_generative_context_detects_generator():
    class ModelConfig:
        GENERATOR = object()

    class Model:
        generative_modeling = False

    class Module:
        model_config = ModelConfig()
        model = Model()

    context = GenerativeContext(Module())

    assert context.generator is True
    assert context.generative_modeling is False


@pytest.mark.pruned
def test_generative_context_without_generator():
    class ModelConfig:
        GENERATOR = None

    class Model:
        generative_modeling = True

    class Module:
        model_config = ModelConfig()
        model = Model()

    context = GenerativeContext(Module())

    assert context.generator is False
    assert context.generative_modeling is True


@pytest.mark.pruned
def test_generative_context_defaults_missing_generator_to_none():
    class ModelConfig:
        pass

    class Model:
        generative_modeling = True

    class Module:
        model_config = ModelConfig()
        model = Model()

    context = GenerativeContext(Module())

    assert context.generator is False
    assert context.generative_modeling is True


@pytest.mark.pruned
def test_generative_context_defaults_missing_generative_modeling_to_false():
    class ModelConfig:
        GENERATOR = object()

    class Model:
        pass

    class Module:
        model_config = ModelConfig()
        model = Model()

    context = GenerativeContext(Module())

    assert context.generator is True
    assert context.generative_modeling is False


@pytest.mark.pruned
def test_generative_context_treats_false_generator_as_present():
    class ModelConfig:
        GENERATOR = False

    class Model:
        generative_modeling = False

    class Module:
        model_config = ModelConfig()
        model = Model()

    context = GenerativeContext(Module())

    assert context.generator is True
    assert context.generative_modeling is False


@pytest.mark.pruned
def test_generative_context_preserves_truthy_generative_modeling_value():
    marker = object()

    class ModelConfig:
        GENERATOR = None

    class Model:
        generative_modeling = marker

    class Module:
        model_config = ModelConfig()
        model = Model()

    context = GenerativeContext(Module())

    assert context.generator is False
    assert context.generative_modeling is marker


@pytest.mark.pruned
def test_load_state_dict_calls_torch_load_with_expected_arguments(
    tmp_path,
    monkeypatch,
):
    module = ConcreteModule()
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.touch()

    checkpoint = {
        "module": module.state_dict(),
    }
    captured = {}

    def fake_load(path, map_location, weights_only):
        captured["path"] = path
        captured["map_location"] = map_location
        captured["weights_only"] = weights_only
        return checkpoint

    monkeypatch.setattr(torch, "load", fake_load)

    module._load_state_dict(checkpoint_path)

    assert captured["path"] == checkpoint_path
    assert captured["map_location"] == module._get_device()
    assert captured["weights_only"] is True


@pytest.mark.pruned
def test_load_state_dict_calls_gc_collect(
    tmp_path,
    monkeypatch,
):
    module = ConcreteModule()
    checkpoint_path = tmp_path / "checkpoint.pt"

    torch.save(
        {"module": module.state_dict()},
        checkpoint_path,
    )

    calls = []

    monkeypatch.setattr(
        "cccma_ppp.core.core_abc.gc.collect",
        lambda: calls.append(True),
    )

    module._load_state_dict(checkpoint_path)

    assert calls == [True]