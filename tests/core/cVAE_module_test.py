import dataclasses

import numpy as np
import pytest
import torch

import cccma_ppp.core.cVAE_module as mod
from cccma_ppp.core.cVAE_module import cVAE, cVAEConfig, cVAEOutput
from cccma_ppp.generic.runtime import RuntimeContext


@pytest.fixture(autouse=True)
def reset_runtime_context_metadata():
    RuntimeContext.INPUT_VAR_METADATA = {}
    RuntimeContext.TARGET_VAR_METADATA = {}
    yield


@dataclasses.dataclass
class DummyCheckpointConfig:
    checkpoint_input_shape: np.ndarray
    checkpoint_output_shape: np.ndarray
    checkpoint_input_var_metadata: dict = dataclasses.field(default_factory=dict)
    checkpoint_output_var_metadata: dict = dataclasses.field(default_factory=dict)
    strict: bool = True
    freeze_weights: bool = False
    load_path: str = "fake_checkpoint.pt"


def make_module(
    cfg=None,
    input_shape=np.array([1]),
    output_shape=None,
    added_features_dim=None,
):
    cfg = cfg or cVAEConfig(ModelConfig=DummySelector())
    return cfg.build(
        input_shape=input_shape,
        output_shape=output_shape,
        added_features_dim=added_features_dim,
    )


class DummyFlow:
    def __init__(self):
        self.flow_sample_size = 2
        self.latent_size = None
        self.condition_size = None

    def build(self, latent_size, condition_size):
        self.latent_size = latent_size
        self.condition_size = condition_size
        return self

    class FlowOutput:
        def __init__(self, x):
            self.e_samples = x
            self.log_det = torch.zeros(x.shape[0])

    def __call__(self, x, condition=None):
        return DummyFlow.FlowOutput(x)


class FullFlow(DummyFlow):
    class FlowOutput:
        def __init__(self, x):
            self.e_samples = x
            self.log_det = torch.ones(x.shape[0])

    def __call__(self, x, condition=None):
        return FullFlow.FlowOutput(x)


class DummyModel:
    GENERATOR = False

    def __init__(self):
        self.latent_size = 4
        self.condition_dependant_latent = False
        self.condition_dependant_flow = False
        self.condition_embedding_size = 3
        self.build_args = None
        self.build_kwargs = None

    def _resolve_flow_settings(self, condition_dependant_flow=False):
        self.condition_dependant_flow = condition_dependant_flow
        return self

    def build(self, *args, **kwargs):
        self.build_args = args
        self.build_kwargs = kwargs
        return self

    def __call__(self, **kwargs):
        return cVAEOutput(
            output=torch.ones(1, 2, 1, 3, 4),
            mu=torch.zeros(1, 4),
            log_var=torch.zeros(1, 4),
        )

    def predict(self, **kwargs):
        return cVAEOutput(
            output=torch.ones(1, 2, 1, 3, 4),
            mu=None,
            log_var=None,
        )


class DummySelector:
    def get_model_config(self):
        return DummyModel()


class DummyBatch:
    def __init__(self):
        self.target = torch.zeros(2, 1, 3, 4)
        self.input = torch.zeros(2, 1, 3, 4)
        self.added_features = None


class DummyLoss:
    reduction = "mean"

    def __call__(self, output, target, **kwargs):
        return torch.tensor(1.0), {"recon": 1.0}


class SumLoss:
    reduction = "sum"

    def __call__(self, *args, **kwargs):
        return torch.tensor(1.0), {}


def test_config_requires_model_or_load():
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        cVAEConfig(ModelConfig=None, load_dir=None)


@pytest.mark.pruned
def test_config_default_weight():
    cfg = cVAEConfig(ModelConfig=DummySelector())

    assert cfg.combined_CGCN_weight == 0


@pytest.mark.pruned
def test_config_weight_bounds():
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        cVAEConfig(ModelConfig=DummySelector(), combined_CGCN_weight=2)


@pytest.mark.pruned
def test_config_negative_cgcn_weight():
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        cVAEConfig(ModelConfig=DummySelector(), combined_CGCN_weight=-0.1)


@pytest.mark.pruned
def test_config_build():
    cfg = cVAEConfig(ModelConfig=DummySelector())

    assert isinstance(cfg.build(np.array([1])), cVAE)


@pytest.mark.pruned
def test_module_build_basic():
    m = make_module()

    assert isinstance(m, cVAE)
    assert m.model is not None


@pytest.mark.pruned
def test_build_without_output_shape():
    m = make_module(input_shape=np.array([1]))

    assert np.array_equal(m.model.build_kwargs["input_shape"], np.array([1]))
    assert np.array_equal(m.model.build_kwargs["output_shape"], np.array([1]))


@pytest.mark.pruned
def test_build_with_explicit_output_shape():
    cfg = cVAEConfig(ModelConfig=DummySelector())

    m = make_module(
        cfg,
        input_shape=np.array([1]),
        output_shape=np.array([2]),
    )

    assert np.array_equal(m.model.build_kwargs["input_shape"], np.array([1]))
    assert np.array_equal(m.model.build_kwargs["output_shape"], np.array([2]))


@pytest.mark.pruned
def test_build_shape_mismatch(monkeypatch):
    cfg = cVAEConfig(ModelConfig=DummySelector())
    cfg.load_dir = "fake_checkpoint.pt"
    cfg.checkpoint_config = DummyCheckpointConfig(
        checkpoint_input_shape=np.array([1]),
        checkpoint_output_shape=np.array([1]),
    )

    monkeypatch.setattr(cVAE, "_load_state_dict", lambda self, load_path: None)

    with pytest.raises((AssertionError, RuntimeError)):
        make_module(cfg, input_shape=np.array([2]), output_shape=np.array([1]))


class ConditionalModel(DummyModel):
    GENERATOR = False

    def __init__(self):
        super().__init__()
        self.condition_dependant_latent = True
        self.condition_embedding_size = 5


class ConditionalSelector:
    def get_model_config(self):
        return ConditionalModel()


@pytest.mark.pruned
def test_conditional_flow_branch():
    cfg = cVAEConfig(ModelConfig=ConditionalSelector(), prior_flow_config=DummyFlow())

    m = make_module(cfg)

    assert m.flow_condition_size == 5
    assert m.prior_flow is not None
    assert m.prior_flow.condition_size == 5


@pytest.mark.pruned
def test_prior_flow_build():
    cfg = cVAEConfig(ModelConfig=DummySelector(), prior_flow_config=DummyFlow())

    m = make_module(cfg)

    assert isinstance(m.prior_flow, DummyFlow)
    assert m.prior_flow.latent_size == 4
    assert m.prior_flow.condition_size is None


@pytest.mark.pruned
def test_flow_without_condition():
    cfg = cVAEConfig(ModelConfig=NoCondSelector(), prior_flow_config=DummyFlow())

    m = make_module(cfg)

    assert m.flow_condition_size is None
    assert m.prior_flow is not None


def test_build_with_prior_flow_and_min_variance_and_added_features():
    cfg = cVAEConfig(
        ModelConfig=DummySelector(),
        prior_flow_config=DummyFlow(),
        min_posterior_variance=0.2,
    )

    m = make_module(
        cfg,
        input_shape=np.array([1]),
        output_shape=np.array([1]),
        added_features_dim=3,
    )

    assert m.prior_flow is not None
    assert torch.is_tensor(m.min_posterior_variance)
    assert m.model.build_kwargs["added_features_dim"] == 3


@pytest.mark.pruned
def test_init_loss_function_basic():
    m = make_module()

    m.init_loss_function(DummyLoss())

    assert m.criterion is not None


def test_init_loss_with_flow_requires_sum():
    cfg = cVAEConfig(ModelConfig=DummySelector(), prior_flow_config=DummyFlow())
    m = make_module(cfg)

    with pytest.raises(RuntimeError):
        m.init_loss_function(DummyLoss())


@pytest.mark.pruned
def test_flow_loss_valid_sum():
    cfg = cVAEConfig(ModelConfig=DummySelector(), prior_flow_config=DummyFlow())
    m = make_module(cfg)

    m.init_loss_function(SumLoss())

    assert m.criterion is not None


@pytest.mark.pruned
def test_forward_pass():
    m = make_module()

    assert isinstance(m.forward(DummyBatch()), cVAEOutput)


@pytest.mark.pruned
def test_predict_pass():
    m = make_module()

    assert isinstance(m.predict(DummyBatch()), cVAEOutput)


@pytest.mark.pruned
def test_forward_with_added_features():
    m = make_module()

    batch = DummyBatch()
    batch.added_features = torch.ones(2, 3)

    assert isinstance(m.forward(batch), cVAEOutput)


@pytest.mark.pruned
def test_predict_with_sample_size_and_prior_flow():
    cfg = cVAEConfig(ModelConfig=DummySelector(), prior_flow_config=DummyFlow())
    m = make_module(cfg)

    out = m.predict(DummyBatch(), sample_size=3)

    assert isinstance(out, cVAEOutput)


@pytest.mark.pruned
def test_forward_with_sample_size_explicit():
    cfg = cVAEConfig(ModelConfig=DummySelector())
    m = make_module(cfg)

    out = m.forward(DummyBatch(), sample_size=5)

    assert isinstance(out, cVAEOutput)


@pytest.mark.pruned
def test_compute_loss_requires_init():
    m = make_module()

    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        m._compute_loss(1.0, DummyBatch())


@pytest.mark.pruned
def test_compute_loss_basic():
    m = make_module()
    m.init_loss_function(DummyLoss())

    total, losses = m._compute_loss(1.0, DummyBatch())

    assert total >= 0
    assert "total_loss" in losses


@pytest.mark.pruned
def test_compute_loss_plain_target():
    m = make_module()
    m.init_loss_function(DummyLoss())

    batch = DummyBatch()
    batch.target = torch.zeros(2, 1, 3, 4)

    total, _ = m._compute_loss(1.0, batch)

    assert total >= 0


@pytest.mark.pruned
def test_compute_loss_with_mask():
    m = make_module()
    m.init_loss_function(DummyLoss())

    batch = DummyBatch()
    batch.target = (batch.target, torch.ones_like(batch.target))

    total, _ = m._compute_loss(1.0, batch)

    assert total >= 0


@pytest.mark.pruned
def test_mask_not_expanded_branch():
    m = make_module()
    m.init_loss_function(DummyLoss())

    batch = DummyBatch()
    batch.target = (batch.target, torch.ones(1, 1, 3, 4))

    total, _ = m._compute_loss(1.0, batch)

    assert total >= 0


@pytest.mark.pruned
def test_compute_loss_mask_none_explicit():
    m = make_module()
    m.init_loss_function(DummyLoss())

    batch = DummyBatch()
    batch.target = (batch.target, None)

    total, _ = m._compute_loss(1.0, batch)

    assert total >= 0


@pytest.mark.pruned
def test_compute_loss_target_as_list():
    cfg = cVAEConfig(ModelConfig=DummySelector())
    m = make_module(cfg)
    m.init_loss_function(DummyLoss())

    batch = DummyBatch()
    batch.target = [batch.target, torch.ones_like(batch.target)]

    total, losses = m._compute_loss(1.0, batch)

    assert total >= 0
    assert "total_loss" in losses


@pytest.mark.pruned
def test_compute_loss_individual_losses_merge_multiple_keys():
    class MultiLoss:
        reduction = "mean"

        def __call__(self, output, target, **kwargs):
            return torch.tensor(1.0), {"a": 1.0, "b": 2.0}

    cfg = cVAEConfig(ModelConfig=DummySelector())
    m = make_module(cfg)
    m.init_loss_function(MultiLoss())

    total, losses = m._compute_loss(1.0, DummyBatch())

    assert total >= 0
    assert losses["a"] == 1.0
    assert losses["b"] == 2.0


@pytest.mark.pruned
def test_compute_loss_beta_zero():
    cfg = cVAEConfig(ModelConfig=DummySelector())
    m = make_module(cfg)
    m.init_loss_function(DummyLoss())

    total, losses = m._compute_loss(0.0, DummyBatch())

    assert total >= 0
    assert "kld" in losses


@pytest.mark.pruned
def test_compute_loss_negative_beta():
    cfg = cVAEConfig(ModelConfig=DummySelector())
    m = make_module(cfg)
    m.init_loss_function(DummyLoss())

    total, losses = m._compute_loss(-1.0, DummyBatch())

    assert torch.is_tensor(total)
    assert "kld" in losses


@pytest.mark.pruned
def test_cgcn_weight_zero_branch():
    cfg = cVAEConfig(ModelConfig=DummySelector(), combined_CGCN_weight=0)
    m = make_module(cfg)
    m.init_loss_function(DummyLoss())

    _, losses = m._compute_loss(1.0, DummyBatch())

    assert "total_loss_CGCN" not in losses


def test_cgcn_weight_active():
    cfg = cVAEConfig(ModelConfig=DummySelector(), combined_CGCN_weight=0.5)
    m = make_module(cfg)
    m.init_loss_function(DummyLoss())

    _, losses = m._compute_loss(1.0, DummyBatch())

    assert "total_loss_CGCN" in losses


@pytest.mark.pruned
def test_cgcn_with_flow():
    cfg = cVAEConfig(
        ModelConfig=DummySelector(),
        prior_flow_config=DummyFlow(),
        combined_CGCN_weight=0.5,
    )

    m = make_module(cfg)
    m.init_loss_function(SumLoss())

    _, losses = m._compute_loss(1.0, DummyBatch())

    assert "total_loss_CGCN" in losses


@pytest.mark.pruned
def test_predict_called_in_cgcn():
    cfg = cVAEConfig(ModelConfig=DummySelector(), combined_CGCN_weight=0.8)

    m = make_module(cfg)
    m.init_loss_function(DummyLoss())

    total, _ = m._compute_loss(1.0, DummyBatch())

    assert total >= 0


def test_kld_with_condition():
    class CondModel(DummyModel):
        GENERATOR = False

        def __call__(self, **kwargs):
            return cVAEOutput(
                output=torch.ones(1, 2, 1, 3, 4),
                mu=torch.zeros(1, 4),
                log_var=torch.zeros(1, 4),
                cond_mu=torch.ones(1, 4),
                cond_log_var=torch.ones(1, 4),
            )

    class Selector:
        def get_model_config(self):
            return CondModel()

    m = make_module(cVAEConfig(ModelConfig=Selector()))
    m.init_loss_function(DummyLoss())

    total, _ = m._compute_loss(1.0, DummyBatch())

    assert total >= 0


@pytest.mark.pruned
def test_kld_base_distribution_path():
    class NoCondModel(DummyModel):
        def __call__(self, **kwargs):
            return cVAEOutput(
                output=torch.ones(1, 2, 1, 3, 4),
                mu=torch.zeros(1, 4),
                log_var=torch.zeros(1, 4),
                cond_mu=None,
                cond_log_var=None,
            )

    class Sel:
        def get_model_config(self):
            return NoCondModel()

    m = make_module(cVAEConfig(ModelConfig=Sel()))
    m.init_loss_function(DummyLoss())

    total, _ = m._compute_loss(1.0, DummyBatch())

    assert total >= 0


@pytest.mark.pruned
def test_kld_cond_shape_mismatch():
    class WeirdModel(DummyModel):
        GENERATOR = False

        def __call__(self, **kwargs):
            return cVAEOutput(
                output=torch.ones(1, 2, 1, 3, 4),
                mu=torch.zeros(1, 4),
                log_var=torch.zeros(1, 4),
                cond_mu=torch.ones(1, 4),
                cond_log_var=torch.ones(1, 5),
            )

    class Sel:
        def get_model_config(self):
            return WeirdModel()

    m = make_module(cVAEConfig(ModelConfig=Sel()))
    m.init_loss_function(DummyLoss())

    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        m._compute_loss(1.0, DummyBatch())


@pytest.mark.pruned
def test_kld_with_prior_flow():
    cfg = cVAEConfig(ModelConfig=DummySelector(), prior_flow_config=DummyFlow())
    m = make_module(cfg)
    m.init_loss_function(SumLoss())

    total, _ = m._compute_loss(1.0, DummyBatch())

    assert total >= 0


@pytest.mark.pruned
def test_full_flow_logdet_branch():
    cfg = cVAEConfig(ModelConfig=DummySelector(), prior_flow_config=FullFlow())
    m = make_module(cfg)
    m.init_loss_function(SumLoss())

    total, _ = m._compute_loss(1.0, DummyBatch())

    assert total >= 0


def test_kld_condition_expansion():
    class CondModel(DummyModel):
        GENERATOR = False

        def __call__(self, **kwargs):
            return cVAEOutput(
                output=torch.ones(1, 2, 1, 3, 4),
                mu=torch.zeros(1, 4),
                log_var=torch.zeros(1, 4),
                cond_mu=torch.ones(1, 4),
                cond_log_var=torch.ones(1, 4),
            )

    class Sel:
        def get_model_config(self):
            return CondModel()

    cfg = cVAEConfig(ModelConfig=Sel(), prior_flow_config=DummyFlow())
    m = make_module(cfg)
    m.init_loss_function(SumLoss())

    total, _ = m._compute_loss(1.0, DummyBatch())

    assert total >= 0


@pytest.mark.pruned
def test_load_checkpoint_missing():
    cfg = cVAEConfig(ModelConfig=DummySelector())

    with pytest.raises(FileNotFoundError):
        cfg._load_from_checkpoint("missing.pt")


@pytest.mark.pruned
def test_config_load_dir_branch_success(monkeypatch):
    def fake_load_from_checkpoint(self, load_path):
        self.ModelConfig = DummySelector()
        self.checkpoint_config = DummyCheckpointConfig(
            checkpoint_input_shape=np.array([1]),
            checkpoint_output_shape=np.array([1]),
        )
        self.prior_flow_config = None
        self.min_posterior_variance = 0.25
        self.combined_CGCN_weight = 0.4
        return self

    monkeypatch.setattr(cVAEConfig, "_load_from_checkpoint", fake_load_from_checkpoint)

    with pytest.warns(UserWarning):
        cfg = cVAEConfig(load_dir="fake_checkpoint.pt")

    assert isinstance(cfg.ModelConfig, DummySelector)
    assert cfg.min_posterior_variance == 0.25
    assert cfg.combined_CGCN_weight == 0.4
    assert cfg.model_config is not None


def test_config_load_dir_branch_with_none_combined_sets_default(monkeypatch):
    def fake_load_from_checkpoint(self, load_path):
        self.ModelConfig = DummySelector()
        self.checkpoint_config = DummyCheckpointConfig(
            checkpoint_input_shape=np.array([1]),
            checkpoint_output_shape=np.array([1]),
        )
        self.prior_flow_config = None
        self.min_posterior_variance = None
        self.combined_CGCN_weight = None
        return self

    monkeypatch.setattr(cVAEConfig, "_load_from_checkpoint", fake_load_from_checkpoint)

    with pytest.warns(UserWarning):
        cfg = cVAEConfig(load_dir="fake_checkpoint.pt")

    assert cfg.combined_CGCN_weight == 0


def test_load_from_checkpoint_full_success_with_prior_flow(monkeypatch, tmp_path):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"placeholder")

    fake_checkpoint = {
        "module_config": {
            "ModelConfig": {"dummy": True},
            "prior_flow_config": {"dummy": True},
            "min_posterior_variance": 0.33,
            "combined_CGCN_weight": 0.66,
        },
        "model_input_shape": np.array([1]),
        "model_output_shape": np.array([2]),
    }

    def fake_torch_load(*args, **kwargs):
        return fake_checkpoint

    def fake_from_dict(data_class, data, config):
        if data_class.__name__ == "cVAEModelSelector":
            return DummySelector()
        if data_class.__name__ == "NormalizedFlowConfig":
            return DummyFlow()
        raise AssertionError(f"Unexpected data_class: {data_class}")

    monkeypatch.setattr(mod.torch, "load", fake_torch_load)
    monkeypatch.setattr(mod.dacite, "from_dict", fake_from_dict)

    cfg = cVAEConfig(ModelConfig=DummySelector())
    cfg.min_posterior_variance = None
    cfg.combined_CGCN_weight = None

    out = cfg._load_from_checkpoint(path)

    assert out is cfg
    assert isinstance(cfg.ModelConfig, DummySelector)
    assert isinstance(cfg.prior_flow_config, DummyFlow)
    assert cfg.min_posterior_variance == 0.33
    assert cfg.combined_CGCN_weight == 0.66

    assert hasattr(cfg, "checkpoint_config")
    assert cfg.checkpoint_config.load_path == path
    assert cfg.checkpoint_config.strict is True
    assert cfg.checkpoint_config.freeze_weights is False


def test_load_from_checkpoint_does_not_override_existing_values(monkeypatch, tmp_path):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"placeholder")

    fake_checkpoint = {
        "module_config": {
            "ModelConfig": {"dummy": True},
            "prior_flow_config": {"dummy": True},
            "min_posterior_variance": 0.33,
            "combined_CGCN_weight": 0.66,
        },
        "model_input_shape": np.array([1]),
        "model_output_shape": np.array([2]),
    }

    def fake_torch_load(*args, **kwargs):
        return fake_checkpoint

    def fake_from_dict(data_class, data, config):
        if data_class.__name__ == "cVAEModelSelector":
            return DummySelector()
        if data_class.__name__ == "NormalizedFlowConfig":
            return DummyFlow()
        raise AssertionError(f"Unexpected data_class: {data_class}")

    monkeypatch.setattr(mod.torch, "load", fake_torch_load)
    monkeypatch.setattr(mod.dacite, "from_dict", fake_from_dict)

    cfg = cVAEConfig(
        ModelConfig=DummySelector(),
        min_posterior_variance=0.9,
        combined_CGCN_weight=0.1,
    )

    cfg._load_from_checkpoint(path)

    assert cfg.min_posterior_variance == 0.9
    assert cfg.combined_CGCN_weight == 0.1


@pytest.mark.pruned
def test_load_from_checkpoint_missing_module_config(monkeypatch, tmp_path):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"placeholder")

    def fake_torch_load(*args, **kwargs):
        return {
            "model_input_shape": np.array([1]),
            "model_output_shape": np.array([1]),
        }

    monkeypatch.setattr(mod.torch, "load", fake_torch_load)

    cfg = cVAEConfig(ModelConfig=DummySelector())

    with pytest.raises((AttributeError, KeyError)):
        cfg._load_from_checkpoint(path)


def test_build_load_dir_success_path(monkeypatch):
    cfg = cVAEConfig(ModelConfig=DummySelector())
    cfg.load_dir = "fake_checkpoint.pt"
    cfg.checkpoint_config = DummyCheckpointConfig(
        checkpoint_input_shape=np.array([1]),
        checkpoint_output_shape=np.array([1]),
    )

    called = {"loaded": False}

    def fake_load_state_dict(self, load_path):
        called["loaded"] = True

    monkeypatch.setattr(cVAE, "_load_state_dict", fake_load_state_dict)

    m = make_module(cfg, input_shape=np.array([1]), output_shape=np.array([1]))

    assert isinstance(m, cVAE)
    assert called["loaded"] is True


@pytest.mark.pruned
def test_build_load_dir_output_shape_mismatch(monkeypatch):
    cfg = cVAEConfig(ModelConfig=DummySelector())
    cfg.load_dir = "fake_checkpoint.pt"
    cfg.checkpoint_config = DummyCheckpointConfig(
        checkpoint_input_shape=np.array([1]),
        checkpoint_output_shape=np.array([1]),
    )

    monkeypatch.setattr(cVAE, "_load_state_dict", lambda self, load_path: None)

    with pytest.raises(RuntimeError, match="output shape"):
        make_module(
            cfg,
            input_shape=np.array([1]),
            output_shape=np.array([2]),
        )


@pytest.mark.pruned
def test_build_load_dir_input_shape_success_output_shape_success(monkeypatch):
    cfg = cVAEConfig(ModelConfig=DummySelector())
    cfg.load_dir = "fake_checkpoint.pt"
    cfg.checkpoint_config = DummyCheckpointConfig(
        checkpoint_input_shape=np.array([1]),
        checkpoint_output_shape=np.array([2]),
    )

    monkeypatch.setattr(cVAE, "_load_state_dict", lambda self, load_path: None)

    m = make_module(
        cfg,
        input_shape=np.array([1]),
        output_shape=np.array([2]),
    )

    assert isinstance(m, cVAE)


class NoCondFlowModel(DummyModel):
    def __init__(self):
        super().__init__()
        self.condition_dependant_latent = False


class NoCondSelector:
    def get_model_config(self):
        return NoCondFlowModel()


@pytest.mark.pruned
def test_forward_uses_default_sample_size():
    m = make_module()

    out = m.forward(DummyBatch())

    assert isinstance(out, cVAEOutput)
    assert out.output.shape[0] == 1


@pytest.mark.pruned
def test_predict_uses_default_sample_size():
    m = make_module()

    out = m.predict(DummyBatch())

    assert isinstance(out, cVAEOutput)
    assert out.mu is None
    assert out.log_var is None


@pytest.mark.pruned
def test_prior_flow_receives_condition_when_enabled():
    cfg = cVAEConfig(
        ModelConfig=ConditionalSelector(),
        prior_flow_config=DummyFlow(),
    )

    m = make_module(cfg)

    assert m.prior_flow.condition_size == 5
    assert m.flow_condition_size == 5


@pytest.mark.pruned
def test_prior_flow_without_condition_path():
    cfg = cVAEConfig(
        ModelConfig=NoCondSelector(),
        prior_flow_config=DummyFlow(),
    )

    m = make_module(cfg)

    assert m.prior_flow.condition_size is None


@pytest.mark.pruned
def test_compute_loss_returns_tensor_total():
    m = make_module()

    m.init_loss_function(DummyLoss())

    total, _ = m._compute_loss(1.0, DummyBatch())

    assert torch.is_tensor(total)


@pytest.mark.pruned
def test_compute_loss_contains_reconstruction_key():
    m = make_module()

    m.init_loss_function(DummyLoss())

    _, losses = m._compute_loss(1.0, DummyBatch())

    assert "recon" in losses


@pytest.mark.pruned
def test_compute_loss_with_zero_mask():
    m = make_module()

    m.init_loss_function(DummyLoss())

    batch = DummyBatch()

    batch.target = (
        batch.target,
        torch.zeros_like(batch.target),
    )

    total, losses = m._compute_loss(1.0, batch)

    assert total >= 0
    assert "total_loss" in losses


@pytest.mark.pruned
def test_compute_loss_with_singleton_mask_dimension():
    m = make_module()

    m.init_loss_function(DummyLoss())

    batch = DummyBatch()

    batch.target = (
        batch.target,
        torch.ones(1, 1, 1, 1),
    )

    total, _ = m._compute_loss(1.0, batch)

    assert total >= 0


@pytest.mark.pruned
def test_compute_loss_with_large_beta():
    m = make_module()

    m.init_loss_function(DummyLoss())

    total, losses = m._compute_loss(100.0, DummyBatch())

    assert torch.is_tensor(total)
    assert "kld" in losses


@pytest.mark.pruned
def test_compute_loss_with_prior_flow_and_zero_beta():
    cfg = cVAEConfig(
        ModelConfig=DummySelector(),
        prior_flow_config=DummyFlow(),
    )

    m = make_module(cfg)

    m.init_loss_function(SumLoss())

    total, losses = m._compute_loss(0.0, DummyBatch())

    assert total >= 0
    assert "kld" in losses


@pytest.mark.pruned
def test_compute_loss_with_full_flow_and_negative_beta():
    cfg = cVAEConfig(
        ModelConfig=DummySelector(),
        prior_flow_config=FullFlow(),
    )

    m = make_module(cfg)

    m.init_loss_function(SumLoss())

    total, _ = m._compute_loss(-1.0, DummyBatch())

    assert torch.is_tensor(total)


@pytest.mark.pruned
def test_build_preserves_added_features_dim():
    cfg = cVAEConfig(ModelConfig=DummySelector())

    m = make_module(
        cfg,
        input_shape=np.array([1]),
        output_shape=np.array([1]),
        added_features_dim=7,
    )

    assert m.model.build_kwargs["added_features_dim"] == 7


@pytest.mark.pruned
def test_model_build_receives_correct_shapes():
    cfg = cVAEConfig(ModelConfig=DummySelector())

    input_shape = np.array([3])
    output_shape = np.array([5])

    m = make_module(
        cfg,
        input_shape=input_shape,
        output_shape=output_shape,
    )

    assert np.array_equal(
        m.model.build_kwargs["input_shape"],
        input_shape,
    )

    assert np.array_equal(
        m.model.build_kwargs["output_shape"],
        output_shape,
    )


@pytest.mark.pruned
def test_build_load_dir_calls_load_state_dict_once(monkeypatch):
    cfg = cVAEConfig(ModelConfig=DummySelector())

    cfg.load_dir = "fake_checkpoint.pt"

    cfg.checkpoint_config = DummyCheckpointConfig(
        checkpoint_input_shape=np.array([1]),
        checkpoint_output_shape=np.array([1]),
    )

    calls = {"count": 0}

    def fake_load_state_dict(self, load_path):
        calls["count"] += 1

    monkeypatch.setattr(
        cVAE,
        "_load_state_dict",
        fake_load_state_dict,
    )

    make_module(
        cfg,
        input_shape=np.array([1]),
        output_shape=np.array([1]),
    )

    assert calls["count"] == 1


@pytest.mark.pruned
def test_compute_loss_with_flow_and_cgcn_and_negative_beta():
    cfg = cVAEConfig(
        ModelConfig=DummySelector(),
        prior_flow_config=DummyFlow(),
        combined_CGCN_weight=0.3,
    )

    m = make_module(cfg)

    m.init_loss_function(SumLoss())

    total, losses = m._compute_loss(-1.0, DummyBatch())

    assert torch.is_tensor(total)
    assert "total_loss_CGCN" in losses


def test_compute_loss_with_conditional_flow_and_mask():
    cfg = cVAEConfig(
        ModelConfig=ConditionalSelector(),
        prior_flow_config=DummyFlow(),
    )

    m = make_module(cfg)

    m.init_loss_function(SumLoss())

    batch = DummyBatch()

    batch.target = (
        batch.target,
        torch.ones_like(batch.target),
    )

    total, _ = m._compute_loss(1.0, batch)

    assert total >= 0