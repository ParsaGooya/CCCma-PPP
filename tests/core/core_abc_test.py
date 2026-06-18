import numpy as np
import pytest
import torch
import torch.nn as nn

from cccma_ppp.core.core_abc import moduleABC, moduleConfigABC


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


def test_get_device_returns_parameter_device():
    module = ConcreteModule()

    expected_device = next(module.parameters()).device

    assert module._get_device() == expected_device


def test_get_device_returns_buffer_device_when_no_parameters():
    module = BufferOnlyConcreteModule()

    assert module._get_device() == module.buffer_value.device


def test_get_device_returns_cpu_without_parameters_or_buffers():
    module = EmptyConcreteModule()

    assert module._get_device() == torch.device("cpu")


def test_load_state_dict_missing_file_raises(tmp_path):
    module = ConcreteModule()

    missing_path = tmp_path / "missing.pt"

    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        module._load_state_dict(missing_path)


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


def test_load_state_dict_uses_module_key(tmp_path):
    module = ConcreteModule()

    checkpoint_path = tmp_path / "invalid_checkpoint.pt"
    torch.save({"not_module": module.state_dict()}, checkpoint_path)

    with pytest.raises(KeyError):
        module._load_state_dict(checkpoint_path)


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
