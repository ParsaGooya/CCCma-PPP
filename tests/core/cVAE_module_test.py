import dataclasses

import numpy as np
import pytest
import torch

import cccma_ppp.core.models.cvae as mod
from cccma_ppp.core.models.cvae import cVAE, cVAEConfig, cVAEOutput
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
    GENERATOR = None

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

    def __call__(self, request):
        return cVAEOutput(
            output=torch.ones(1, 2, 1, 3, 4),
            mu=torch.zeros(1, 4),
            log_var=torch.zeros(1, 4),
        )

    def predict(self, *args, **kwargs):
        return cVAEOutput(
            output=torch.ones(1, 2, 1, 3, 4),
            mu=None,
            log_var=None,
        )


class DummySelector:
    def get_model_config(self):
        return DummyModel()


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
    reduction = "mean"

    def to(self, device):
        self.device = device
        return self

    def __call__(self, output, target, **kwargs):
        return torch.tensor(1.0), {"recon": 1.0}


class SumLoss:
    reduction = "sum"

    def to(self, device):
        self.device = device
        return self

    def __call__(self, *args, **kwargs):
        return torch.tensor(1.0), {}


def test_config_requires_model_or_load():
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        cVAEConfig(ModelConfig=None, load_dir=None)


def test_config_default_weight():
    cfg = cVAEConfig(ModelConfig=DummySelector())

    assert cfg.combined_CGCN_weight == 0


def test_config_weight_bounds():
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        cVAEConfig(ModelConfig=DummySelector(), combined_CGCN_weight=2)


def test_config_negative_cgcn_weight():
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        cVAEConfig(ModelConfig=DummySelector(), combined_CGCN_weight=-0.1)


def test_config_build():
    cfg = cVAEConfig(ModelConfig=DummySelector())

    assert isinstance(cfg.build(np.array([1])), cVAE)


def test_module_build_basic():
    m = make_module()

    assert isinstance(m, cVAE)
    assert m.model is not None


def test_build_without_output_shape():
    m = make_module(input_shape=np.array([1]))

    assert np.array_equal(m.model.build_kwargs["input_shape"], np.array([1]))
    assert np.array_equal(m.model.build_kwargs["output_shape"], np.array([1]))


def test_build_with_explicit_output_shape():
    cfg = cVAEConfig(ModelConfig=DummySelector())

    m = make_module(
        cfg,
        input_shape=np.array([1]),
        output_shape=np.array([2]),
    )

    assert np.array_equal(m.model.build_kwargs["input_shape"], np.array([1]))
    assert np.array_equal(m.model.build_kwargs["output_shape"], np.array([2]))


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
    GENERATOR = None

    def __init__(self):
        super().__init__()
        self.condition_dependant_latent = True
        self.condition_embedding_size = 5


class ConditionalSelector:
    def get_model_config(self):
        return ConditionalModel()


def test_conditional_flow_branch():
    cfg = cVAEConfig(ModelConfig=ConditionalSelector(), prior_flow_config=DummyFlow())

    m = make_module(cfg)

    assert m.flow_condition_size == 5
    assert m.prior_flow is not None
    assert m.prior_flow.condition_size == 5


def test_prior_flow_build():
    cfg = cVAEConfig(ModelConfig=DummySelector(), prior_flow_config=DummyFlow())

    m = make_module(cfg)

    assert isinstance(m.prior_flow, DummyFlow)
    assert m.prior_flow.latent_size == 4
    assert m.prior_flow.condition_size is None


def test_flow_without_condition():
    cfg = cVAEConfig(ModelConfig=NoCondSelector(), prior_flow_config=DummyFlow())

    m = make_module(cfg)

    assert m.flow_condition_size is None
    assert m.prior_flow is not None


def test_init_loss_function_basic():
    m = make_module()

    m.init_loss_function(DummyLoss())

    assert m.criterion is not None


def test_init_loss_with_flow_requires_sum():
    cfg = cVAEConfig(ModelConfig=DummySelector(), prior_flow_config=DummyFlow())
    m = make_module(cfg)

    with pytest.raises(RuntimeError):
        m.init_loss_function(DummyLoss())


def test_flow_loss_valid_sum():
    cfg = cVAEConfig(ModelConfig=DummySelector(), prior_flow_config=DummyFlow())
    m = make_module(cfg)

    m.init_loss_function(SumLoss())

    assert m.criterion is not None


def test_forward_pass():
    m = make_module()

    assert isinstance(m.forward(DummyBatch()), cVAEOutput)


def test_predict_pass():
    m = make_module()

    assert isinstance(m.predict(DummyBatch()), cVAEOutput)


def test_forward_with_added_features():
    m = make_module()

    batch = DummyBatch()
    batch.added_features = torch.ones(2, 3)

    assert isinstance(m.forward(batch), cVAEOutput)


def test_predict_with_sample_size_and_prior_flow():
    cfg = cVAEConfig(ModelConfig=DummySelector(), prior_flow_config=DummyFlow())
    m = make_module(cfg)

    out = m.predict(DummyBatch(), 3)

    assert isinstance(out, cVAEOutput)


def test_forward_with_sample_size_explicit():
    cfg = cVAEConfig(ModelConfig=DummySelector())
    m = make_module(cfg)

    out = m.forward(DummyBatch(), 5)

    assert isinstance(out, cVAEOutput)


def test_compute_loss_requires_init():
    m = make_module()

    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        m._compute_loss(1.0, DummyBatch())


def test_compute_loss_plain_target():
    m = make_module()
    m.init_loss_function(DummyLoss())

    batch = DummyBatch()
    batch.target = torch.zeros(2, 1, 3, 4)

    total, _ = m._compute_loss(1.0, batch)

    assert total >= 0


def test_kld_cond_shape_mismatch():
    class WeirdModel(DummyModel):
        GENERATOR = None

        def __call__(self, request):
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

    module = make_module(
        cVAEConfig(ModelConfig=Sel()),
    )
    module.init_loss_function(DummyLoss())

    with pytest.raises(
        (AssertionError, ValueError, RuntimeError),
    ):
        module._compute_loss(
            1.0,
            DummyBatch(),
        )


def test_load_checkpoint_missing():
    cfg = cVAEConfig(ModelConfig=DummySelector())

    with pytest.raises(FileNotFoundError):
        cfg._load_from_checkpoint("missing.pt")


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
            "posterior_variance_limits": (0.33, None),
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
    cfg.posterior_variance_limits = None
    cfg.combined_CGCN_weight = None

    out = cfg._load_from_checkpoint(path)

    assert out is cfg
    assert isinstance(cfg.ModelConfig, DummySelector)
    assert isinstance(cfg.prior_flow_config, DummyFlow)
    assert cfg.posterior_variance_limits == (0.33, None)
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
            "posterior_variance_limits": (0.33, None),
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
        posterior_variance_limits=(0.9, None),
        combined_CGCN_weight=0.1,
    )

    cfg._load_from_checkpoint(path)

    assert cfg.posterior_variance_limits == (0.9, None)
    assert cfg.combined_CGCN_weight == 0.1


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


def test_forward_uses_default_sample_size():
    m = make_module()

    out = m.forward(DummyBatch())

    assert isinstance(out, cVAEOutput)
    assert out.output.shape[0] == 1


def test_predict_uses_default_sample_size():
    m = make_module()

    out = m.predict(DummyBatch())

    assert isinstance(out, cVAEOutput)
    assert out.mu is None
    assert out.log_var is None


def test_prior_flow_receives_condition_when_enabled():
    cfg = cVAEConfig(
        ModelConfig=ConditionalSelector(),
        prior_flow_config=DummyFlow(),
    )

    m = make_module(cfg)

    assert m.prior_flow.condition_size == 5
    assert m.flow_condition_size == 5


def test_prior_flow_without_condition_path():
    cfg = cVAEConfig(
        ModelConfig=NoCondSelector(),
        prior_flow_config=DummyFlow(),
    )

    m = make_module(cfg)

    assert m.prior_flow.condition_size is None


def test_build_preserves_added_features_dim():
    cfg = cVAEConfig(ModelConfig=DummySelector())

    m = make_module(
        cfg,
        input_shape=np.array([1]),
        output_shape=np.array([1]),
        added_features_dim=7,
    )

    assert m.model.build_kwargs["added_features_dim"] == 7


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


class RecordingModel(DummyModel):
    def __init__(self):
        super().__init__()
        self.forward_request = None
        self.predict_request = None

    def __call__(self, request):
        self.forward_request = request

        return cVAEOutput(
            output=torch.ones(1, 2, 1, 3, 4),
            mu=torch.zeros(1, 4),
            log_var=torch.zeros(1, 4),
        )

    def predict(self, request):
        self.predict_request = request

        return cVAEOutput(
            output=torch.ones(1, 2, 1, 3, 4),
            mu=None,
            log_var=None,
        )


class RecordingSelector:
    def __init__(self):
        self.model_config = RecordingModel()

    def get_model_config(self):
        return self.model_config


class DummyGenerator:
    num_training_noise_samples = 7
    num_validation_noise_samples = 11


class GeneratorModel(RecordingModel):
    GENERATOR = DummyGenerator()


class GeneratorSelector:
    def __init__(self):
        self.model_config = GeneratorModel()

    def get_model_config(self):
        return self.model_config


class RecordingLoss:
    reduction = "mean"

    def __init__(self, value=1.0):
        self.value = value
        self.calls = []

    def to(self, device):
        self.device = device
        return self

    def __call__(
        self,
        output,
        target,
        target_mask=None,
        print_loss=False,
    ):
        self.calls.append(
            {
                "output": output,
                "target": target,
                "target_mask": target_mask,
                "print_loss": print_loss,
            }
        )

        return torch.tensor(self.value), {"recon": self.value}


class ConstantKLD(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.calls = []

    def forward(
        self,
        mu,
        log_var,
        cond_mu,
        cond_log_var,
        prior_flow=None,
        print_loss=False,
    ):
        self.calls.append(
            {
                "mu": mu,
                "log_var": log_var,
                "cond_mu": cond_mu,
                "cond_log_var": cond_log_var,
                "prior_flow": prior_flow,
                "print_loss": print_loss,
            }
        )

        return torch.tensor(self.value)


def test_forward_eval_uses_validation_noise_sample_count():
    selector = GeneratorSelector()
    module = make_module(
        cVAEConfig(ModelConfig=selector),
    )
    module.eval()

    module.forward(DummyBatch())

    request = selector.model_config.forward_request

    assert request.output_sample_size == 11


def test_predict_training_uses_training_noise_sample_count():
    selector = GeneratorSelector()
    module = make_module(
        cVAEConfig(ModelConfig=selector),
    )
    module.train()

    module.predict(
        DummyBatch(),
        output_sample_size=3,
    )

    request = selector.model_config.predict_request

    assert request.output_sample_size == 7


def test_predict_eval_preserves_explicit_output_sample_size():
    selector = GeneratorSelector()
    module = make_module(
        cVAEConfig(ModelConfig=selector),
    )
    module.eval()

    module.predict(
        DummyBatch(),
        output_sample_size=13,
    )

    request = selector.model_config.predict_request

    assert request.output_sample_size == 13


def test_compute_loss_expands_matching_target_mask():
    module = make_module()
    criterion = RecordingLoss()
    module.init_loss_function(criterion)
    module.KLD = ConstantKLD(2.0)

    target = torch.zeros(2, 1, 3, 4)
    target_mask = torch.ones_like(target)
    batch = DummyBatch(
        target=target,
        target_mask=target_mask,
    )

    total, losses = module._compute_loss(
        beta=0.5,
        data=batch,
    )

    call = criterion.calls[0]

    assert call["target"].shape == (1, 2, 1, 3, 4)
    assert call["target_mask"].shape == (1, 2, 1, 3, 4)
    assert call["print_loss"] is False
    assert torch.allclose(total, torch.tensor(2.0))
    assert losses["total_loss"] == pytest.approx(2.0)
    assert losses["kld"] == pytest.approx(2.0)
    assert losses["recon"] == pytest.approx(1.0)


def test_compute_loss_accepts_none_target_mask():
    module = make_module()
    criterion = RecordingLoss()
    module.init_loss_function(criterion)
    module.KLD = ConstantKLD(0.0)

    batch = DummyBatch(
        target=torch.zeros(2, 1, 3, 4),
        target_mask=None,
    )

    module._compute_loss(
        beta=1.0,
        data=batch,
    )

    assert criterion.calls[0]["target_mask"] is None


def test_compute_loss_passes_prior_flow_to_kld():
    cfg = cVAEConfig(
        ModelConfig=DummySelector(),
        prior_flow_config=DummyFlow(),
    )
    module = make_module(cfg)

    criterion = SumLoss()
    module.init_loss_function(criterion)

    kld = ConstantKLD(2.0)
    module.KLD = kld

    batch = DummyBatch(
        target=torch.zeros(2, 1, 3, 4),
    )

    module._compute_loss(
        beta=0.25,
        data=batch,
    )

    assert kld.calls[0]["prior_flow"] is module.prior_flow
    assert kld.calls[0]["print_loss"] is False


def test_compute_loss_applies_beta_to_kld():
    module = make_module()
    criterion = RecordingLoss(value=3.0)
    module.init_loss_function(criterion)
    module.KLD = ConstantKLD(4.0)

    batch = DummyBatch(
        target=torch.zeros(2, 1, 3, 4),
    )

    total, losses = module._compute_loss(
        beta=0.5,
        data=batch,
    )

    assert torch.allclose(
        total,
        torch.tensor(5.0),
    )
    assert losses["total_loss"] == pytest.approx(5.0)
    assert losses["kld"] == pytest.approx(4.0)


def test_compute_loss_combines_cgcn_reconstruction_loss(monkeypatch):
    cfg = cVAEConfig(
        ModelConfig=DummySelector(),
        combined_CGCN_weight=0.25,
    )
    module = make_module(cfg)

    criterion = RecordingLoss(value=2.0)
    module.init_loss_function(criterion)
    module.KLD = ConstantKLD(4.0)

    predicted = cVAEOutput(
        output=torch.ones(1, 2, 1, 3, 4),
        mu=None,
        log_var=None,
    )

    predict_calls = []

    def fake_predict(data):
        predict_calls.append(data)
        return predicted

    monkeypatch.setattr(
        module,
        "predict",
        fake_predict,
    )

    batch = DummyBatch(
        target=torch.zeros(2, 1, 3, 4),
    )

    total, losses = module._compute_loss(
        beta=0.5,
        data=batch,
    )

    assert torch.allclose(
        total,
        torch.tensor(3.5),
    )
    assert predict_calls == [batch]
    assert len(criterion.calls) == 2
    assert criterion.calls[1]["target_mask"] is None
    assert criterion.calls[1]["print_loss"] is False
    assert losses["total_loss_CGCN"] == pytest.approx(2.0)
    assert losses["total_loss"] == pytest.approx(3.5)


def test_compute_loss_skips_cgcn_prediction_at_zero_weight(
    monkeypatch,
):
    module = make_module()
    criterion = RecordingLoss()
    module.init_loss_function(criterion)
    module.KLD = ConstantKLD(0.0)

    def unexpected_predict(data):
        raise AssertionError("predict() must not be called when CGCN weight is zero.")

    monkeypatch.setattr(
        module,
        "predict",
        unexpected_predict,
    )

    batch = DummyBatch(
        target=torch.zeros(2, 1, 3, 4),
    )

    module._compute_loss(
        beta=1.0,
        data=batch,
    )


def test_init_loss_function_moves_losses_to_module_device():
    module = make_module()

    criterion = RecordingLoss()
    module.init_loss_function(criterion)

    assert criterion.device == module._get_device()
    assert module.KLD.reduction == criterion.reduction


def test_build_load_dir_rejects_input_metadata_mismatch(
    monkeypatch,
):
    RuntimeContext.INPUT_VAR_METADATA = {
        "variable": "current",
    }
    RuntimeContext.TARGET_VAR_METADATA = {
        "variable": "target",
    }

    cfg = cVAEConfig(ModelConfig=DummySelector())
    cfg.load_dir = "fake_checkpoint.pt"
    cfg.checkpoint_config = DummyCheckpointConfig(
        checkpoint_input_shape=np.array([1]),
        checkpoint_output_shape=np.array([1]),
        checkpoint_input_var_metadata={
            "variable": "different",
        },
        checkpoint_output_var_metadata={
            "variable": "target",
        },
    )

    monkeypatch.setattr(
        cVAE,
        "_load_state_dict",
        lambda self, load_path: None,
    )

    with pytest.raises(
        RuntimeError,
        match="input variables",
    ):
        make_module(
            cfg,
            input_shape=np.array([1]),
            output_shape=np.array([1]),
        )


def test_build_load_dir_rejects_output_metadata_mismatch(
    monkeypatch,
):
    RuntimeContext.INPUT_VAR_METADATA = {
        "variable": "input",
    }
    RuntimeContext.TARGET_VAR_METADATA = {
        "variable": "current",
    }

    cfg = cVAEConfig(ModelConfig=DummySelector())
    cfg.load_dir = "fake_checkpoint.pt"
    cfg.checkpoint_config = DummyCheckpointConfig(
        checkpoint_input_shape=np.array([1]),
        checkpoint_output_shape=np.array([1]),
        checkpoint_input_var_metadata={
            "variable": "input",
        },
        checkpoint_output_var_metadata={
            "variable": "different",
        },
    )

    monkeypatch.setattr(
        cVAE,
        "_load_state_dict",
        lambda self, load_path: None,
    )

    with pytest.raises(
        RuntimeError,
        match="output variables",
    ):
        make_module(
            cfg,
            input_shape=np.array([1]),
            output_shape=np.array([1]),
        )


def test_build_load_dir_accepts_matching_metadata(
    monkeypatch,
):
    input_metadata = {
        "variable": "input",
    }
    output_metadata = {
        "variable": "output",
    }

    RuntimeContext.INPUT_VAR_METADATA = input_metadata
    RuntimeContext.TARGET_VAR_METADATA = output_metadata

    cfg = cVAEConfig(ModelConfig=DummySelector())
    cfg.load_dir = "fake_checkpoint.pt"
    cfg.checkpoint_config = DummyCheckpointConfig(
        checkpoint_input_shape=np.array([1]),
        checkpoint_output_shape=np.array([1]),
        checkpoint_input_var_metadata=input_metadata,
        checkpoint_output_var_metadata=output_metadata,
    )

    calls = []

    monkeypatch.setattr(
        cVAE,
        "_load_state_dict",
        lambda self, load_path: calls.append(load_path),
    )

    module = make_module(
        cfg,
        input_shape=np.array([1]),
        output_shape=np.array([1]),
    )

    assert isinstance(module, cVAE)
    assert calls == ["fake_checkpoint.pt"]


def test_conditional_flow_resolves_model_flow_settings():
    selector = ConditionalSelector()

    cfg = cVAEConfig(
        ModelConfig=selector,
        prior_flow_config=DummyFlow(),
    )

    assert cfg.condition_dependant_flow is True
    assert cfg.model_config.condition_dependant_flow is True


def test_nonconditional_flow_does_not_set_condition_size():
    cfg = cVAEConfig(
        ModelConfig=NoCondSelector(),
        prior_flow_config=DummyFlow(),
    )

    module = make_module(cfg)

    assert cfg.condition_dependant_flow is False
    assert module.flow_condition_size is None

    output_tensor = torch.ones(1)
    samples = torch.ones(2, 4)
    cond_mu = torch.zeros(2, 4)
    cond_log_var = torch.ones(2, 4)

    output = cVAEOutput(
        output=output_tensor,
        mu=torch.zeros(2, 4),
        log_var=torch.zeros(2, 4),
        samples=samples,
        cond_mu=cond_mu,
        cond_log_var=cond_log_var,
    )

    assert output.output is output_tensor
    assert output.samples is samples
    assert output.cond_mu is cond_mu
    assert output.cond_log_var is cond_log_var
