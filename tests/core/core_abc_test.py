import pytest
import torch
import torch.nn as nn
import numpy as np

from cccma_ppp.core.core_abc import moduleABC, moduleConfigABC


class DummyModule(moduleABC):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def build(self, input_shape, output_shape=None, added_features_dim=None):
        return True

    def init_loss_function(self, reconstruction_loss, **kwargs):
        self.loss = reconstruction_loss

    def _compute_loss(self):
        return torch.tensor(1.0)

    def forward(self, x=None):
        return self.linear(torch.ones(1, 2))

    def predict(self):
        return torch.tensor(0.0)


class DummyConfig(moduleConfigABC):
    def build(self, input_shape, output_shape=None, added_features_dim=None):
        return "built"

    def _load_from_checkpoint(self):
        return "loaded"


def test_module_cannot_instantiate_abstract():
    with pytest.raises(TypeError):
        moduleABC()


def test_config_cannot_instantiate_abstract():
    with pytest.raises(TypeError):
        moduleConfigABC()


def test_dummy_module_basic():
    m = DummyModule()

    assert m.build(np.array([1, 2]))
    assert m.forward().shape == (1, 2)
    assert m.predict() is not None
    assert m._compute_loss() >= 0


def test_dummy_config_basic():
    c = DummyConfig()
    assert c.build(np.array([1])) == "built"
    assert c._load_from_checkpoint() == "loaded"


def test_get_device_from_parameters():
    m = DummyModule()
    device = m._get_device()
    assert isinstance(device, torch.device)


def test_get_device_from_buffer():
    class BufferOnly(moduleABC):
        def __init__(self):
            super().__init__()
            self.register_buffer("buf", torch.ones(1))

        def build(self, *a, **k):
            pass

        def init_loss_function(self, *a, **k):
            pass

        def _compute_loss(self):
            pass

        def forward(self):
            pass

        def predict(self):
            pass

    m = BufferOnly()
    assert m._get_device() == m.buf.device


def test_get_device_cpu_fallback():
    class NoParams(moduleABC):
        def build(self, *a, **k):
            pass

        def init_loss_function(self, *a, **k):
            pass

        def _compute_loss(self):
            pass

        def forward(self):
            pass

        def predict(self):
            pass

    m = NoParams()
    assert m._get_device().type == "cpu"


def test_load_state_dict_file_not_found(tmp_path):
    m = DummyModule()
    fake_path = tmp_path / "missing.pt"

    with pytest.raises(FileNotFoundError):
        m._load_state_dict(fake_path)


def test_load_state_dict_success(tmp_path):
    m = DummyModule()

    checkpoint_path = tmp_path / "ckpt.pt"
    torch.save({"module": m.state_dict()}, checkpoint_path)

    new_model = DummyModule()
    new_model._load_state_dict(checkpoint_path)

    for k, v in m.state_dict().items():
        assert torch.equal(v, new_model.state_dict()[k])


def test_load_state_dict_non_strict(tmp_path):
    m = DummyModule()

    checkpoint_path = tmp_path / "ckpt.pt"
    torch.save({"module": m.state_dict()}, checkpoint_path)

    new_model = DummyModule()

    state = m.state_dict()
    state.pop(list(state.keys())[0])
    torch.save({"module": state}, checkpoint_path)

    new_model._load_state_dict(checkpoint_path, strict=False)


def test_load_on_specific_device(tmp_path):
    m = DummyModule()

    checkpoint_path = tmp_path / "ckpt.pt"
    torch.save({"module": m.state_dict()}, checkpoint_path)

    m.to(torch.device("cpu"))

    m._load_state_dict(checkpoint_path)
    assert True


def test_load_triggers_gc(tmp_path):
    m = DummyModule()

    checkpoint_path = tmp_path / "ckpt.pt"
    torch.save({"module": m.state_dict()}, checkpoint_path)

    m._load_state_dict(checkpoint_path)
