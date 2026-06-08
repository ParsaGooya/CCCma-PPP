import pytest
import torch
import numpy as np

from cccma_ppp.core.deterministic_module import (
    deterministic,
    deterministicConfig,
    deterministicOutput,
)


class ConcreteDeterministic(deterministic):
    def predict(self, *args, **kwargs):
        return super().predict(*args, **kwargs)


class DummyModel:
    def __init__(self):
        self.was_built = False
        self.last_build_kwargs = None
        self.last_call_kwargs = None

    def build(self, **kwargs):
        self.was_built = True
        self.last_build_kwargs = kwargs
        return self

    def __call__(self, x=None, added_features=None):
        self.last_call_kwargs = {
            "x": x,
            "added_features": added_features,
        }
        return deterministicOutput(output=torch.ones_like(x))


class DummySelector:
    def get_model_config(self):
        return self

    def build(self):
        return DummyModel()


class DummyBatch:
    def __init__(self):
        self.input = torch.ones(2, 1, 3, 4)
        self.target = torch.zeros(2, 1, 3, 4)
        self.added_features = None


class DummyLoss:
    def __call__(self, output, target, target_mask=None, print_loss=False):
        return torch.tensor(1.0), {"mse": 1.0}


class MultiLoss:
    def __call__(self, output, target, target_mask=None, print_loss=False):
        return torch.tensor(2.0), {"mse": 1.0, "mae": 1.0}


class EmptyLoss:
    def __call__(self, output, target, target_mask=None, print_loss=False):
        return torch.tensor(1.0), {}


def make_config():
    return deterministicConfig(ModelConfig=DummySelector())


def make_module():
    return ConcreteDeterministic(make_config())


def test_output_dataclass():
    out = deterministicOutput(output=torch.ones(1))
    assert torch.is_tensor(out.output)


def test_raw_deterministic_is_abstract():
    cfg = make_config()

    module = deterministic(cfg)
    assert isinstance(module, deterministic)


def test_config_requires_model_or_load():
    with pytest.raises(AssertionError):
        deterministicConfig(ModelConfig=None, load_dir=None)


def test_config_basic_initialization():
    cfg = make_config()

    assert cfg.ModelConfig is not None
    assert cfg.model_config is not None
    assert cfg.model is not None


def test_config_build_returns_built_module(monkeypatch):
    import cccma_ppp.core.deterministic_module as mod

    monkeypatch.setattr(mod, "deterministic", ConcreteDeterministic)

    cfg = make_config()
    module = cfg.build(input_shape=np.array([1]))

    assert isinstance(module, ConcreteDeterministic)
    assert module.built is True


def test_load_checkpoint_missing_file():
    cfg = make_config()

    with pytest.raises(FileNotFoundError):
        cfg._load_from_checkpoint("missing_checkpoint.pt")


def test_load_checkpoint_success(monkeypatch, tmp_path):
    import cccma_ppp.core.deterministic_module as mod

    ckpt_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "module_config": {"ModelConfig": {}},
            "model_input_shape": np.array([1]),
            "model_output_shape": np.array([2]),
        },
        ckpt_path,
    )

    monkeypatch.setattr(
        mod.dacite,
        "from_dict",
        lambda **kwargs: DummySelector(),
    )

    cfg = make_config()
    out = cfg._load_from_checkpoint(ckpt_path)

    assert out is cfg
    assert isinstance(cfg.ModelConfig, DummySelector)
    assert (cfg._checkpoint_input_shape == np.array([1])).all()
    assert (cfg._checkpoint_output_shape == np.array([2])).all()


def test_load_checkpoint_missing_module_config(monkeypatch, tmp_path):

    ckpt_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_input_shape": np.array([1]),
            "model_output_shape": np.array([1]),
        },
        ckpt_path,
    )

    cfg = make_config()

    with pytest.raises(AttributeError):
        cfg._load_from_checkpoint(ckpt_path)


def test_config_load_dir_warns_and_uses_loaded_config(monkeypatch, tmp_path):
    ckpt_path = tmp_path / "checkpoint.pt"
    ckpt_path.write_bytes(b"placeholder")

    def fake_load_from_checkpoint(self, load_path):
        self.ModelConfig = DummySelector()
        self._checkpoint_input_shape = np.array([1])
        self._checkpoint_output_shape = np.array([1])
        return self

    monkeypatch.setattr(
        deterministicConfig,
        "_load_from_checkpoint",
        fake_load_from_checkpoint,
    )

    with pytest.warns(UserWarning):
        cfg = deterministicConfig(load_dir=str(ckpt_path))

    assert isinstance(cfg.ModelConfig, DummySelector)
    assert cfg.model is not None


def test_module_initial_state():
    module = make_module()

    assert module.built is False
    assert module.criterion is None
    assert module.model is not None


def test_build_basic():
    module = make_module()

    out = module.build(input_shape=np.array([1]))

    assert out is module
    assert module.built is True
    assert (module.input_shape == np.array([1])).all()
    assert (module.output_shape == np.array([1])).all()
    assert module.model.was_built is True


def test_build_default_output_shape_copies_input_shape():
    module = make_module()

    input_shape = np.array([1, 2, 3])
    module.build(input_shape=input_shape)

    assert (module.output_shape == input_shape).all()
    assert module.output_shape is not input_shape


def test_build_explicit_output_shape():
    module = make_module()

    module.build(
        input_shape=np.array([1]),
        output_shape=np.array([2]),
    )

    assert (module.input_shape == np.array([1])).all()
    assert (module.output_shape == np.array([2])).all()


def test_build_passes_added_features_dim_to_model():
    module = make_module()

    module.build(
        input_shape=np.array([1]),
        output_shape=np.array([1]),
        added_features_dim=5,
    )

    assert module.model.last_build_kwargs["added_features_dim"] == 5


def test_build_load_dir_input_shape_mismatch():
    cfg = make_config()
    cfg.load_dir = "fake_checkpoint.pt"
    cfg._checkpoint_input_shape = np.array([1])
    cfg._checkpoint_output_shape = np.array([1])

    module = ConcreteDeterministic(cfg)

    with pytest.raises(AssertionError):
        module.build(input_shape=np.array([2]))


def test_build_load_dir_output_shape_mismatch():
    cfg = make_config()
    cfg.load_dir = "fake_checkpoint.pt"
    cfg._checkpoint_input_shape = np.array([1])
    cfg._checkpoint_output_shape = np.array([1])

    module = ConcreteDeterministic(cfg)

    with pytest.raises(AssertionError):
        module.build(
            input_shape=np.array([1]),
            output_shape=np.array([2]),
        )


def test_build_load_dir_success_calls_load_state_dict(monkeypatch):
    cfg = make_config()
    cfg.load_dir = "fake_checkpoint.pt"
    cfg._checkpoint_input_shape = np.array([1])
    cfg._checkpoint_output_shape = np.array([1])

    called = {"load": False}

    def fake_load_state_dict(self, load_path):
        called["load"] = True
        assert load_path == "fake_checkpoint.pt"

    monkeypatch.setattr(
        ConcreteDeterministic,
        "_load_state_dict",
        fake_load_state_dict,
    )

    module = ConcreteDeterministic(cfg)
    module.build(input_shape=np.array([1]))

    assert module.built is True
    assert called["load"] is True


def test_build_load_dir_success_with_explicit_output_shape(monkeypatch):
    cfg = make_config()
    cfg.load_dir = "fake_checkpoint.pt"
    cfg._checkpoint_input_shape = np.array([1])
    cfg._checkpoint_output_shape = np.array([2])

    monkeypatch.setattr(
        ConcreteDeterministic,
        "_load_state_dict",
        lambda self, load_path: None,
    )

    module = ConcreteDeterministic(cfg)
    module.build(input_shape=np.array([1]), output_shape=np.array([2]))

    assert module.built is True


def test_init_loss_function_sets_criterion():
    module = make_module()
    loss = DummyLoss()

    module.init_loss_function(loss)

    assert module.criterion is loss


def test_forward_returns_deterministic_output():
    module = make_module()
    module.build(input_shape=np.array([1]))

    batch = DummyBatch()
    out = module.forward(batch)

    assert isinstance(out, deterministicOutput)
    assert out.output.shape == batch.input.shape


def test_forward_passes_added_features():
    module = make_module()
    module.build(input_shape=np.array([1]))

    batch = DummyBatch()
    batch.added_features = torch.ones(2, 3)

    out = module.forward(batch)

    assert isinstance(out, deterministicOutput)
    assert module.model.last_call_kwargs["added_features"] is batch.added_features


def test_predict_calls_forward():
    module = make_module()
    module.build(input_shape=np.array([1]))

    batch = DummyBatch()
    out = module.predict(batch)

    assert isinstance(out, deterministicOutput)


def test_predict_alias_calls_predict():
    module = make_module()
    module.build(input_shape=np.array([1]))

    batch = DummyBatch()
    out = module.predict(batch)

    assert isinstance(out, deterministicOutput)


def test_compute_loss_requires_criterion():
    module = make_module()
    module.build(input_shape=np.array([1]))

    with pytest.raises(AssertionError):
        module._compute_loss(DummyBatch())


def test_compute_loss_plain_target():
    module = make_module()
    module.build(input_shape=np.array([1]))
    module.init_loss_function(DummyLoss())

    total, losses = module._compute_loss(DummyBatch())

    assert torch.is_tensor(total)
    assert total.item() == 1.0
    assert losses["total_loss"] == 1.0
    assert losses["mse"] == 1.0


def test_compute_loss_tuple_target_with_mask():
    module = make_module()
    module.build(input_shape=np.array([1]))
    module.init_loss_function(DummyLoss())

    batch = DummyBatch()
    mask = torch.ones_like(batch.target)
    batch.target = (batch.target, mask)

    total, losses = module._compute_loss(batch)

    assert total.item() == 1.0
    assert losses["total_loss"] == 1.0


def test_compute_loss_list_target_with_mask():
    module = make_module()
    module.build(input_shape=np.array([1]))
    module.init_loss_function(DummyLoss())

    batch = DummyBatch()
    mask = torch.ones_like(batch.target)
    batch.target = [batch.target, mask]

    total, losses = module._compute_loss(batch)

    assert total.item() == 1.0
    assert losses["total_loss"] == 1.0


def test_compute_loss_merges_multiple_individual_losses():
    module = make_module()
    module.build(input_shape=np.array([1]))
    module.init_loss_function(MultiLoss())

    total, losses = module._compute_loss(DummyBatch())

    assert total.item() == 2.0
    assert losses["total_loss"] == 2.0
    assert losses["mse"] == 1.0
    assert losses["mae"] == 1.0


def test_compute_loss_empty_individual_losses():
    module = make_module()
    module.build(input_shape=np.array([1]))
    module.init_loss_function(EmptyLoss())

    total, losses = module._compute_loss(DummyBatch())

    assert total.item() == 1.0
    assert losses == {"total_loss": 1.0}


def test_compute_loss_passes_mask_to_criterion():
    class InspectLoss:
        def __init__(self):
            self.seen_mask = None

        def __call__(self, output, target, target_mask=None, print_loss=False):
            self.seen_mask = target_mask
            return torch.tensor(1.0), {}

    module = make_module()
    module.build(input_shape=np.array([1]))

    loss = InspectLoss()
    module.init_loss_function(loss)

    batch = DummyBatch()
    mask = torch.ones_like(batch.target)
    batch.target = (batch.target, mask)

    module._compute_loss(batch)

    assert loss.seen_mask is mask


def test_compute_loss_passes_print_loss_false():
    class InspectLoss:
        def __init__(self):
            self.print_loss = None

        def __call__(self, output, target, target_mask=None, print_loss=False):
            self.print_loss = print_loss
            return torch.tensor(1.0), {}

    module = make_module()
    module.build(input_shape=np.array([1]))

    loss = InspectLoss()
    module.init_loss_function(loss)

    module._compute_loss(DummyBatch())

    assert loss.print_loss is False


def test_compute_loss_target_tuple_with_none_mask():
    module = make_module()
    module.build(input_shape=np.array([1]))
    module.init_loss_function(DummyLoss())

    batch = DummyBatch()
    batch.target = (batch.target, None)

    total, losses = module._compute_loss(batch)

    assert total.item() == 1.0
    assert "total_loss" in losses
