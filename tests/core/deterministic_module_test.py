import pytest
import torch
import numpy as np

from cccma_ppp.core.modules.deterministic import (
    deterministicOutput,
    deterministic,
    deterministicConfig,
)


def make_config():
    return deterministicConfig(ModelConfig=DummySelector())


def make_module(
    input_shape=np.array([1]),
    output_shape=None,
    added_features_dim=None,
    cfg=None,
):
    if cfg is None:
        cfg = make_config()

    return ConcreteDeterministic(
        cfg,
        input_shape=input_shape,
        output_shape=output_shape,
        added_features_dim=added_features_dim,
    )


@pytest.mark.pruned
def test_raw_deterministic_can_be_constructed():
    cfg = make_config()

    module = deterministic(cfg, input_shape=np.array([1]))

    assert isinstance(module, deterministic)
    assert module.model is not None


def test_config_requires_model_or_load():
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        deterministicConfig(ModelConfig=None, load_dir=None)


@pytest.mark.pruned
def test_config_basic_initialization():
    cfg = make_config()

    assert cfg.ModelConfig is not None
    assert cfg.model_config is not None


@pytest.mark.pruned
def test_config_build_returns_built_module(monkeypatch):
    import cccma_ppp.core.modules.deterministic as mod

    setattr(mod, "deterministic", ConcreteDeterministic)

    cfg = make_config()
    module = cfg.build(input_shape=np.array([1]))

    assert isinstance(module, ConcreteDeterministic)
    assert module.model is not None
    assert module.model.was_built is True
    assert (module.model.last_build_kwargs["input_shape"] == np.array([1])).all()
    assert (module.model.last_build_kwargs["output_shape"] == np.array([1])).all()


@pytest.mark.pruned
def test_load_checkpoint_missing_file():
    cfg = make_config()

    with pytest.raises(FileNotFoundError):
        cfg._load_from_checkpoint("missing_checkpoint.pt")


@pytest.mark.pruned
def test_load_checkpoint_success(monkeypatch, tmp_path):
    import cccma_ppp.core.modules.deterministic as mod

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
    assert cfg.model_config is not None


@pytest.mark.pruned
def test_load_checkpoint_missing_module_config(tmp_path):
    ckpt_path = tmp_path / "checkpoint.pt"

    torch.save(
        {
            "input_shape": np.array([1]),
            "output_shape": np.array([1]),
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
        self.model_config = self.ModelConfig.get_model_config()
        self.checkpoint_config = DummyCheckpointConfig(
            input_shape=np.array([1]),
            output_shape=np.array([1]),
        )
        return self

    monkeypatch.setattr(
        deterministicConfig,
        "_load_from_checkpoint",
        fake_load_from_checkpoint,
    )

    with pytest.warns(UserWarning):
        cfg = deterministicConfig(load_dir=str(ckpt_path))

    assert isinstance(cfg.ModelConfig, DummySelector)
    assert cfg.model_config is not None
    assert cfg.checkpoint_config is not None


@pytest.mark.pruned
def test_module_initial_state():
    module = make_module()

    assert module.criterion is None
    assert module.model is not None
    assert module.model.was_built is True


@pytest.mark.pruned
def test_constructor_builds_basic():
    module = make_module(input_shape=np.array([1]))

    assert module.model is not None
    assert module.model.was_built is True
    assert (module.model.last_build_kwargs["input_shape"] == np.array([1])).all()
    assert (module.model.last_build_kwargs["output_shape"] == np.array([1])).all()


@pytest.mark.pruned
def test_constructor_default_output_shape_copies_input_shape():
    input_shape = np.array([1, 2, 3])

    module = make_module(input_shape=input_shape)

    output_shape = module.model.last_build_kwargs["output_shape"]

    assert (output_shape == input_shape).all()
    assert output_shape is not input_shape


@pytest.mark.pruned
def test_constructor_explicit_output_shape():
    module = make_module(
        input_shape=np.array([1]),
        output_shape=np.array([2]),
    )

    assert (module.model.last_build_kwargs["input_shape"] == np.array([1])).all()
    assert (module.model.last_build_kwargs["output_shape"] == np.array([2])).all()


@pytest.mark.pruned
def test_constructor_passes_added_features_dim_to_model():
    module = make_module(
        input_shape=np.array([1]),
        output_shape=np.array([1]),
        added_features_dim=5,
    )

    assert module.model.last_build_kwargs["added_features_dim"] == 5


@pytest.mark.pruned
def test_constructor_passes_shapes_to_model():
    module = make_module(
        input_shape=np.array([1]),
        output_shape=np.array([2]),
        added_features_dim=3,
    )

    assert (module.model.last_build_kwargs["input_shape"] == np.array([1])).all()
    assert (module.model.last_build_kwargs["output_shape"] == np.array([2])).all()
    assert module.model.last_build_kwargs["added_features_dim"] == 3


def test_load_dir_input_metadata_mismatch(monkeypatch):
    import cccma_ppp.core.modules.deterministic as mod

    monkeypatch.setattr(mod.RuntimeContext, "INPUT_VAR_METADATA", {"current": "input"})
    monkeypatch.setattr(
        mod.RuntimeContext, "TARGET_VAR_METADATA", {"current": "target"}
    )

    cfg = make_config()
    cfg.load_dir = "fake_checkpoint.pt"
    cfg.checkpoint_config = DummyCheckpointConfig(
        input_shape=np.array([1]),
        output_shape=np.array([1]),
        input_metadata={"old": "input"},
        output_metadata={"current": "target"},
    )

    with pytest.raises(RuntimeError):
        ConcreteDeterministic(cfg, input_shape=np.array([1]))


def test_load_dir_output_metadata_mismatch(monkeypatch):
    import cccma_ppp.core.modules.deterministic as mod

    monkeypatch.setattr(mod.RuntimeContext, "INPUT_VAR_METADATA", {"current": "input"})
    monkeypatch.setattr(
        mod.RuntimeContext, "TARGET_VAR_METADATA", {"current": "target"}
    )

    cfg = make_config()
    cfg.load_dir = "fake_checkpoint.pt"
    cfg.checkpoint_config = DummyCheckpointConfig(
        input_shape=np.array([1]),
        output_shape=np.array([1]),
        input_metadata={"current": "input"},
        output_metadata={"old": "target"},
    )

    with pytest.raises(RuntimeError):
        ConcreteDeterministic(cfg, input_shape=np.array([1]))


def test_load_dir_input_shape_mismatch(monkeypatch):
    import cccma_ppp.core.modules.deterministic as mod

    monkeypatch.setattr(mod.RuntimeContext, "INPUT_VAR_METADATA", None)
    monkeypatch.setattr(mod.RuntimeContext, "TARGET_VAR_METADATA", None)

    cfg = make_config()
    cfg.load_dir = "fake_checkpoint.pt"
    cfg.checkpoint_config = DummyCheckpointConfig(
        input_shape=np.array([1]),
        output_shape=np.array([1]),
    )

    with pytest.raises(RuntimeError):
        ConcreteDeterministic(cfg, input_shape=np.array([2]))


def test_load_dir_output_shape_mismatch(monkeypatch):
    import cccma_ppp.core.modules.deterministic as mod

    monkeypatch.setattr(mod.RuntimeContext, "INPUT_VAR_METADATA", None)
    monkeypatch.setattr(mod.RuntimeContext, "TARGET_VAR_METADATA", None)

    cfg = make_config()
    cfg.load_dir = "fake_checkpoint.pt"
    cfg.checkpoint_config = DummyCheckpointConfig(
        input_shape=np.array([1]),
        output_shape=np.array([1]),
    )

    with pytest.raises(RuntimeError):
        ConcreteDeterministic(
            cfg,
            input_shape=np.array([1]),
            output_shape=np.array([2]),
        )


def test_load_dir_success_calls_load_state_dict(monkeypatch):
    import cccma_ppp.core.modules.deterministic as mod

    monkeypatch.setattr(mod.RuntimeContext, "INPUT_VAR_METADATA", None)
    monkeypatch.setattr(mod.RuntimeContext, "TARGET_VAR_METADATA", None)

    cfg = make_config()
    cfg.load_dir = "fake_checkpoint.pt"
    cfg.checkpoint_config = DummyCheckpointConfig(
        input_shape=np.array([1]),
        output_shape=np.array([1]),
    )

    called = {"load": False}

    def fake_load_state_dict(self, load_path):
        called["load"] = True
        assert load_path == "fake_checkpoint.pt"

    monkeypatch.setattr(
        ConcreteDeterministic,
        "_load_state_dict",
        fake_load_state_dict,
    )

    module = ConcreteDeterministic(cfg, input_shape=np.array([1]))

    assert module.model is not None
    assert module.model.was_built is True
    assert called["load"] is True


@pytest.mark.pruned
def test_load_dir_success_with_explicit_output_shape(monkeypatch):
    import cccma_ppp.core.modules.deterministic as mod

    monkeypatch.setattr(mod.RuntimeContext, "INPUT_VAR_METADATA", None)
    monkeypatch.setattr(mod.RuntimeContext, "TARGET_VAR_METADATA", None)

    cfg = make_config()
    cfg.load_dir = "fake_checkpoint.pt"
    cfg.checkpoint_config = DummyCheckpointConfig(
        input_shape=np.array([1]),
        output_shape=np.array([2]),
    )

    monkeypatch.setattr(
        ConcreteDeterministic,
        "_load_state_dict",
        lambda self, load_path: None,
    )

    module = ConcreteDeterministic(
        cfg,
        input_shape=np.array([1]),
        output_shape=np.array([2]),
    )

    assert module.model is not None
    assert module.model.was_built is True


@pytest.mark.pruned
def test_init_loss_function_sets_criterion():
    module = make_module()
    loss = DummyLoss()

    module.init_loss_function(loss)

    assert module.criterion is loss


@pytest.mark.pruned
def test_forward_returns_deterministic_output():
    module = make_module(input_shape=np.array([1]))

    batch = DummyBatch()
    out = module.forward(batch)

    assert isinstance(out, deterministicOutput)
    assert out.output.shape == batch.input.shape


@pytest.mark.pruned
def test_forward_passes_added_features():
    module = make_module(input_shape=np.array([1]))

    batch = DummyBatch()
    batch.added_features = torch.ones(2, 3)

    out = module.forward(batch)

    assert isinstance(out, deterministicOutput)
    assert module.model.last_call_kwargs["added_features"] is batch.added_features


@pytest.mark.pruned
def test_predict_calls_forward():
    module = make_module(input_shape=np.array([1]))

    batch = DummyBatch()
    out = module.predict(batch)

    assert isinstance(out, deterministicOutput)


@pytest.mark.pruned
def test_predict_alias_calls_predict():
    module = make_module(input_shape=np.array([1]))

    batch = DummyBatch()
    out = module.predict(batch)

    assert isinstance(out, deterministicOutput)


@pytest.mark.pruned
def test_compute_loss_requires_criterion():
    module = make_module(input_shape=np.array([1]))

    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        module._compute_loss(DummyBatch())


@pytest.mark.pruned
def test_compute_loss_plain_target():
    module = make_module(input_shape=np.array([1]))
    module.init_loss_function(DummyLoss())

    total, losses = module._compute_loss(DummyBatch())

    assert torch.is_tensor(total)
    assert total.item() == 1.0
    assert losses["total_loss"] == 1.0
    assert losses["mse"] == 1.0


@pytest.mark.pruned
def test_compute_loss_tuple_target_with_mask():
    module = make_module(input_shape=np.array([1]))
    module.init_loss_function(DummyLoss())

    batch = DummyBatch()
    batch.target_mask = torch.ones_like(batch.target)

    total, losses = module._compute_loss(batch)

    assert total.item() == 1.0
    assert losses["total_loss"] == 1.0


def test_compute_loss_list_target_with_mask():
    module = make_module(input_shape=np.array([1]))
    module.init_loss_function(DummyLoss())

    batch = DummyBatch()
    batch.target_mask = torch.ones_like(batch.target)

    total, losses = module._compute_loss(batch)

    assert total.item() == 1.0
    assert losses["total_loss"] == 1.0


@pytest.mark.pruned
def test_compute_loss_merges_multiple_individual_losses():
    module = make_module(input_shape=np.array([1]))
    module.init_loss_function(MultiLoss())

    total, losses = module._compute_loss(DummyBatch())

    assert total.item() == 2.0
    assert losses["total_loss"] == 2.0
    assert losses["mse"] == 1.0
    assert losses["mae"] == 1.0


@pytest.mark.pruned
def test_compute_loss_empty_individual_losses():
    module = make_module(input_shape=np.array([1]))
    module.init_loss_function(EmptyLoss())

    total, losses = module._compute_loss(DummyBatch())

    assert total.item() == 1.0
    assert losses == {"total_loss": 1.0}


@pytest.mark.pruned
def test_compute_loss_passes_print_loss_false():
    class InspectLoss:
        def __init__(self):
            self.print_loss = None

        def to(self, device):
            return self

        def __call__(
            self,
            output,
            target,
            target_mask=None,
            print_loss=False,
        ):
            self.print_loss = print_loss
            return torch.tensor(1.0), {}

    module = make_module(input_shape=np.array([1]))

    loss = InspectLoss()
    module.init_loss_function(loss)

    module._compute_loss(DummyBatch())

    assert loss.print_loss is False


@pytest.mark.pruned
def test_compute_loss_target_tuple_with_none_mask():
    module = make_module(input_shape=np.array([1]))
    module.init_loss_function(DummyLoss())

    batch = DummyBatch()
    batch.target_mask = None

    total, losses = module._compute_loss(batch)

    assert total.item() == 1.0
    assert "total_loss" in losses


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

    def __call__(self, request):
        self.last_request = request
        self.last_call_kwargs = {
            "x": request.input,
            "x_mask": request.input_mask,
            "added_features": request.added_features,
            "output_sample_size": request.output_sample_size,
        }

        return deterministicOutput(
            output=torch.ones_like(request.input),
        )


class DummySelector:
    GENERATOR = None

    def __init__(self):
        self.last_build_kwargs = None

    def get_model_config(self):
        return self

    def build(self, **kwargs):
        self.last_build_kwargs = kwargs
        model = DummyModel()
        model.build(**kwargs)
        return model


class DummyCheckpointConfig:
    def __init__(
        self,
        input_shape,
        output_shape,
        input_metadata=None,
        output_metadata=None,
    ):
        self.checkpoint_input_shape = input_shape
        self.checkpoint_output_shape = output_shape
        self.checkpoint_input_var_metadata = input_metadata
        self.checkpoint_output_var_metadata = output_metadata


class DummyBatch:
    def __init__(
        self,
        input=None,
        target=None,
        input_mask=None,
        target_mask=None,
        added_features=None,
        metadata=None,
    ):
        self.input = torch.ones(2, 1) if input is None else input
        self.target = torch.ones(2, 1) if target is None else target
        self.input_mask = input_mask
        self.target_mask = target_mask
        self.added_features = added_features
        self.metadata = metadata


class DummyLoss:
    def __call__(self, output, target, target_mask=None, print_loss=False):
        return torch.tensor(1.0), {"mse": 1.0}

    def to(self, device):
        return self


class MultiLoss:
    def __call__(self, output, target, target_mask=None, print_loss=False):
        return torch.tensor(2.0), {"mse": 1.0, "mae": 1.0}

    def to(self, device):
        return self


class EmptyLoss:
    def __call__(self, output, target, target_mask=None, print_loss=False):
        return torch.tensor(1.0), {}

    def to(self, device):
        return self
