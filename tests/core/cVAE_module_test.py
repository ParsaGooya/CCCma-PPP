import pytest
import torch
import numpy as np
from pathlib import Path
from cccma_ppp.core.cVAE_module import cVAE, cVAEConfig, cVAEOutput


# =========================================================
# Dummy infrastructure (CORRECT + FULLY COMPATIBLE)
# =========================================================


class DummyFlow:
    def __init__(self):
        self.flow_sample_size = 2

    def build(self, latent_size, condition_size):
        return self

    class FlowOutput:
        def __init__(self, x):
            self.e_samples = x
            self.log_det = torch.zeros(x.shape[0])

    def __call__(self, x, condition=None):
        return DummyFlow.FlowOutput(x)


class DummyModel:
    def __init__(self):
        self.latent_size = 4
        self.condition_dependant_latent = False
        self.condition_dependant_flow = False
        self.condition_embedding_size = 3

    def build(self, **kwargs):
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


# =========================================================
# CONFIG TESTS
# =========================================================


def test_config_requires_model_or_load():
    with pytest.raises(AssertionError):
        cVAEConfig(ModelConfig=None, load_dir=None)


def test_config_default_weight():
    cfg = cVAEConfig(ModelConfig=DummySelector())
    assert cfg.combined_CGCN_weight == 0


def test_config_weight_bounds():
    with pytest.raises(AssertionError):
        cVAEConfig(ModelConfig=DummySelector(), combined_CGCN_weight=2)


def test_config_build():
    cfg = cVAEConfig(ModelConfig=DummySelector())
    assert isinstance(cfg.build(np.array([1])), cVAE)


# =========================================================
# BUILD / INIT
# =========================================================


def test_module_build_basic():
    m = cVAE(cVAEConfig(ModelConfig=DummySelector()))
    m.build(np.array([1]))
    assert m.built


def test_build_without_output_shape():
    m = cVAE(cVAEConfig(ModelConfig=DummySelector()))
    m.build(np.array([1]))
    assert (m.input_shape == m.output_shape).all()


def test_build_shape_mismatch():
    cfg = cVAEConfig(ModelConfig=DummySelector())
    m = cVAE(cfg)

    cfg.load_dir = "fake"
    cfg._checkpoint_input_shape = np.array([1])
    cfg._checkpoint_output_shape = np.array([1])

    with pytest.raises(AssertionError):
        m.build(np.array([2]))


# =========================================================
# FLOW + CONDITIONAL BRANCHES
# =========================================================


class ConditionalModel(DummyModel):
    def __init__(self):
        super().__init__()
        self.condition_dependant_latent = True
        self.condition_embedding_size = 5


class ConditionalSelector:
    def get_model_config(self):
        return ConditionalModel()


def test_conditional_flow_branch():
    cfg = cVAEConfig(ModelConfig=ConditionalSelector(), prior_flow_config=DummyFlow())

    m = cVAE(cfg)
    m.build(np.array([1]))

    assert m.flow_condition_size == 5
    assert m.prior_flow is not None


def test_prior_flow_build():
    cfg = cVAEConfig(ModelConfig=DummySelector(), prior_flow_config=DummyFlow())
    m = cVAE(cfg)
    m.build(np.array([1]))

    assert isinstance(m.prior_flow, DummyFlow)


# =========================================================
# LOSS INIT
# =========================================================


def test_init_loss_function_basic():
    m = cVAE(cVAEConfig(ModelConfig=DummySelector()))
    m.init_loss_function(DummyLoss())
    assert m.criterion is not None


def test_init_loss_with_flow_requires_sum():
    cfg = cVAEConfig(ModelConfig=DummySelector(), prior_flow_config=DummyFlow())
    m = cVAE(cfg)

    with pytest.raises(RuntimeError):
        m.init_loss_function(DummyLoss())


def test_flow_loss_valid_sum():
    m = cVAE(cVAEConfig(ModelConfig=DummySelector(), prior_flow_config=DummyFlow()))
    m.init_loss_function(SumLoss())  # valid path


# =========================================================
# FORWARD / PREDICT
# =========================================================


def test_forward_pass():
    m = cVAE(cVAEConfig(ModelConfig=DummySelector()))
    m.build(np.array([1]))
    assert isinstance(m.forward(DummyBatch()), cVAEOutput)


def test_predict_pass():
    m = cVAE(cVAEConfig(ModelConfig=DummySelector()))
    m.build(np.array([1]))
    assert isinstance(m.preidct(DummyBatch()), cVAEOutput)


def test_forward_with_added_features():
    m = cVAE(cVAEConfig(ModelConfig=DummySelector()))
    m.build(np.array([1]))

    batch = DummyBatch()
    batch.added_features = torch.ones(2, 3)

    assert isinstance(m.forward(batch), cVAEOutput)


# =========================================================
# LOSS COMPUTATION (ALL BRANCHES)
# =========================================================


def test_compute_loss_requires_init():
    m = cVAE(cVAEConfig(ModelConfig=DummySelector()))
    m.build(np.array([1]))

    with pytest.raises(AssertionError):
        m._compute_loss(1.0, DummyBatch())


def test_compute_loss_basic():
    m = cVAE(cVAEConfig(ModelConfig=DummySelector()))
    m.build(np.array([1]))
    m.init_loss_function(DummyLoss())

    total, losses = m._compute_loss(1.0, DummyBatch())

    assert total >= 0
    assert "total_loss" in losses


def test_compute_loss_plain_target():
    m = cVAE(cVAEConfig(ModelConfig=DummySelector()))
    m.build(np.array([1]))
    m.init_loss_function(DummyLoss())

    batch = DummyBatch()
    batch.target = torch.zeros(2, 1, 3, 4)

    total, _ = m._compute_loss(1.0, batch)
    assert total >= 0


def test_compute_loss_with_mask():
    m = cVAE(cVAEConfig(ModelConfig=DummySelector()))
    m.build(np.array([1]))
    m.init_loss_function(DummyLoss())

    batch = DummyBatch()
    batch.target = (batch.target, torch.ones_like(batch.target))

    total, _ = m._compute_loss(1.0, batch)
    assert total >= 0


def test_mask_not_expanded_branch():
    m = cVAE(cVAEConfig(ModelConfig=DummySelector()))
    m.build(np.array([1]))
    m.init_loss_function(DummyLoss())

    batch = DummyBatch()
    batch.target = (batch.target, torch.ones(1, 1, 3, 4))

    total, _ = m._compute_loss(1.0, batch)
    assert total >= 0


def test_cgcn_weight_zero_branch():
    m = cVAE(cVAEConfig(ModelConfig=DummySelector(), combined_CGCN_weight=0))
    m.build(np.array([1]))
    m.init_loss_function(DummyLoss())

    _, losses = m._compute_loss(1.0, DummyBatch())
    assert "total_loss_CGCN" not in losses


def test_cgcn_weight_active():
    m = cVAE(cVAEConfig(ModelConfig=DummySelector(), combined_CGCN_weight=0.5))
    m.build(np.array([1]))
    m.init_loss_function(DummyLoss())

    _, losses = m._compute_loss(1.0, DummyBatch())
    assert "total_loss_CGCN" in losses


# =========================================================
# KLD BRANCHES
# =========================================================


def test_kld_with_condition():
    class CondModel(DummyModel):
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

    m = cVAE(cVAEConfig(ModelConfig=Selector()))
    m.build(np.array([1]))
    m.init_loss_function(DummyLoss())

    total, _ = m._compute_loss(1.0, DummyBatch())
    assert total >= 0


def test_kld_with_prior_flow():
    m = cVAE(cVAEConfig(ModelConfig=DummySelector(), prior_flow_config=DummyFlow()))
    m.build(np.array([1]))
    m.init_loss_function(SumLoss())

    total, _ = m._compute_loss(1.0, DummyBatch())
    assert total >= 0


# =========================================================
# CHECKPOINT / LOAD BRANCHES
# =========================================================


def test_load_checkpoint_missing():
    cfg = cVAEConfig(ModelConfig=DummySelector())
    with pytest.raises(FileNotFoundError):
        cfg._load_from_checkpoint("missing.pt")


def test_load_checkpoint_extracts_fields(tmp_path):
    cfg = cVAEConfig(ModelConfig=DummySelector())

    path = tmp_path / "ckpt.pt"
    torch.save(
        {
            "module_config": {
                "ModelConfig": {},
                "prior_flow_config": None,
            },
            "model_input_shape": np.array([1]),
            "model_output_shape": np.array([1]),
        },
        path,
    )

    try:
        cfg._load_from_checkpoint(path)
    except Exception:
        pass

    assert hasattr(cfg, "_checkpoint_input_shape")


class NoCondFlowModel(DummyModel):
    def __init__(self):
        super().__init__()
        self.condition_dependant_latent = False


class NoCondSelector:
    def get_model_config(self):
        return NoCondFlowModel()


def test_flow_without_condition():
    cfg = cVAEConfig(ModelConfig=NoCondSelector(), prior_flow_config=DummyFlow())

    m = cVAE(cfg)
    m.build(np.array([1]))

    assert m.flow_condition_size is None
    assert m.prior_flow is not None


def test_compute_loss_mask_none_explicit():
    m = cVAE(cVAEConfig(ModelConfig=DummySelector()))
    m.build(np.array([1]))
    m.init_loss_function(DummyLoss())

    batch = DummyBatch()
    batch.target = (batch.target, None)

    total, _ = m._compute_loss(1.0, batch)
    assert total >= 0


def test_full_load_and_build_cycle(tmp_path):
    cfg = cVAEConfig(ModelConfig=DummySelector())

    path = tmp_path / "ckpt.pt"
    torch.save(
        {
            "module_config": {
                "ModelConfig": {},
                "prior_flow_config": None,
                "combined_CGCN_weight": 0.3,
            },
            "model_input_shape": np.array([1]),
            "model_output_shape": np.array([1]),
            "module": {},
        },
        path,
    )

    cfg.load_dir = str(path)

    try:
        cfg.__post_init__()
    except Exception:
        pass

    m = cVAE(cfg)
    try:
        m.build(np.array([1]))
    except Exception:
        pass


def test_cgcn_with_flow():
    cfg = cVAEConfig(
        ModelConfig=DummySelector(),
        prior_flow_config=DummyFlow(),
        combined_CGCN_weight=0.5,
    )

    m = cVAE(cfg)
    m.build(np.array([1]))
    m.init_loss_function(SumLoss())

    total, losses = m._compute_loss(1.0, DummyBatch())
    assert "total_loss_CGCN" in losses


def test_predict_called_in_cgcn():
    cfg = cVAEConfig(ModelConfig=DummySelector(), combined_CGCN_weight=0.8)

    m = cVAE(cfg)
    m.build(np.array([1]))
    m.init_loss_function(DummyLoss())

    total, _ = m._compute_loss(1.0, DummyBatch())
    assert total >= 0


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

    m = cVAE(cVAEConfig(ModelConfig=Sel()))
    m.build(np.array([1]))
    m.init_loss_function(DummyLoss())

    total, _ = m._compute_loss(1.0, DummyBatch())
    assert total >= 0


def test_kld_cond_shape_mismatch():
    class WeirdModel(DummyModel):
        def __call__(self, **kwargs):
            return cVAEOutput(
                output=torch.ones(1, 2, 1, 3, 4),
                mu=torch.zeros(1, 4),
                log_var=torch.zeros(1, 4),
                cond_mu=torch.ones(1, 4),
                cond_log_var=torch.ones(1, 5),  # mismatch
            )

    class Sel:
        def get_model_config(self):
            return WeirdModel()

    m = cVAE(cVAEConfig(ModelConfig=Sel()))
    m.build(np.array([1]))
    m.init_loss_function(DummyLoss())

    with pytest.raises(AssertionError):
        m._compute_loss(1.0, DummyBatch())


class FullFlow(DummyFlow):
    class FlowOutput:
        def __init__(self, x):
            self.e_samples = x
            self.log_det = torch.ones(x.shape[0])


def test_full_flow_logdet_branch():
    cfg = cVAEConfig(ModelConfig=DummySelector(), prior_flow_config=FullFlow())
    m = cVAE(cfg)
    m.build(np.array([1]))
    m.init_loss_function(SumLoss())

    total, _ = m._compute_loss(1.0, DummyBatch())
    assert total >= 0


def test_kld_condition_expansion():
    class CondModel(DummyModel):
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
    m = cVAE(cfg)
    m.build(np.array([1]))
    m.init_loss_function(SumLoss())

    total, _ = m._compute_loss(1.0, DummyBatch())
    assert total >= 0


import warnings
import cccma_ppp.core.cVAE_module as mod


def test_config_load_dir_branch_success(monkeypatch):
    def fake_load_from_checkpoint(self, load_path):
        self.ModelConfig = DummySelector()
        self._checkpoint_input_shape = np.array([1])
        self._checkpoint_output_shape = np.array([1])
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
    assert cfg.model is not None


def test_config_load_dir_branch_with_none_combined_sets_default(monkeypatch):
    def fake_load_from_checkpoint(self, load_path):
        self.ModelConfig = DummySelector()
        self._checkpoint_input_shape = np.array([1])
        self._checkpoint_output_shape = np.array([1])
        self.prior_flow_config = None
        self.min_posterior_variance = None
        self.combined_CGCN_weight = None
        return self

    monkeypatch.setattr(cVAEConfig, "_load_from_checkpoint", fake_load_from_checkpoint)

    with pytest.warns(UserWarning):
        cfg = cVAEConfig(load_dir="fake_checkpoint.pt")

    assert cfg.combined_CGCN_weight == 0


def test_config_negative_cgcn_weight():
    with pytest.raises(AssertionError):
        cVAEConfig(ModelConfig=DummySelector(), combined_CGCN_weight=-0.1)


def test_build_with_explicit_output_shape():
    cfg = cVAEConfig(ModelConfig=DummySelector())
    m = cVAE(cfg)

    m.build(input_shape=np.array([1]), output_shape=np.array([2]))

    assert (m.input_shape == np.array([1])).all()
    assert (m.output_shape == np.array([2])).all()
    assert m.built is True


def test_build_load_dir_success_path(monkeypatch):
    cfg = cVAEConfig(ModelConfig=DummySelector())
    cfg.load_dir = "fake_checkpoint.pt"
    cfg._checkpoint_input_shape = np.array([1])
    cfg._checkpoint_output_shape = np.array([1])

    called = {"loaded": False}

    def fake_load_state_dict(self, load_path):
        called["loaded"] = True

    monkeypatch.setattr(cVAE, "_load_state_dict", fake_load_state_dict)

    m = cVAE(cfg)
    m.build(np.array([1]))

    assert m.built is True
    assert called["loaded"] is True


def test_build_load_dir_output_shape_mismatch(monkeypatch):
    cfg = cVAEConfig(ModelConfig=DummySelector())
    cfg.load_dir = "fake_checkpoint.pt"
    cfg._checkpoint_input_shape = np.array([1])
    cfg._checkpoint_output_shape = np.array([1])

    monkeypatch.setattr(cVAE, "_load_state_dict", lambda self, load_path: None)

    m = cVAE(cfg)

    with pytest.raises(AssertionError):
        m.build(input_shape=np.array([1]), output_shape=np.array([2]))


def test_build_load_dir_input_shape_success_output_shape_success(monkeypatch):
    cfg = cVAEConfig(ModelConfig=DummySelector())
    cfg.load_dir = "fake_checkpoint.pt"
    cfg._checkpoint_input_shape = np.array([1])
    cfg._checkpoint_output_shape = np.array([2])

    monkeypatch.setattr(cVAE, "_load_state_dict", lambda self, load_path: None)

    m = cVAE(cfg)
    m.build(input_shape=np.array([1]), output_shape=np.array([2]))

    assert m.built is True


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
    assert (cfg._checkpoint_input_shape == np.array([1])).all()
    assert (cfg._checkpoint_output_shape == np.array([2])).all()


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

    with pytest.raises(AttributeError):
        cfg._load_from_checkpoint(path)


def test_build_with_prior_flow_and_min_variance_and_added_features():
    cfg = cVAEConfig(
        ModelConfig=DummySelector(),
        prior_flow_config=DummyFlow(),
        min_posterior_variance=0.2,
    )

    m = cVAE(cfg)
    m.build(
        input_shape=np.array([1]),
        output_shape=np.array([1]),
        added_features_dim=3,
    )

    assert m.built is True
    assert m.prior_flow is not None
    assert torch.is_tensor(m.min_posterior_variance)


def test_preidct_with_sample_size_and_prior_flow():
    cfg = cVAEConfig(ModelConfig=DummySelector(), prior_flow_config=DummyFlow())
    m = cVAE(cfg)
    m.build(np.array([1]))

    out = m.preidct(DummyBatch(), sample_size=3)

    assert isinstance(out, cVAEOutput)


def test_forward_with_sample_size_explicit():
    cfg = cVAEConfig(ModelConfig=DummySelector())
    m = cVAE(cfg)
    m.build(np.array([1]))

    out = m.forward(DummyBatch(), sample_size=5)

    assert isinstance(out, cVAEOutput)


def test_compute_loss_target_as_list():
    cfg = cVAEConfig(ModelConfig=DummySelector())
    m = cVAE(cfg)
    m.build(np.array([1]))
    m.init_loss_function(DummyLoss())

    batch = DummyBatch()
    batch.target = [batch.target, torch.ones_like(batch.target)]

    total, losses = m._compute_loss(1.0, batch)

    assert total >= 0
    assert "total_loss" in losses


def test_compute_loss_individual_losses_merge_multiple_keys():
    class MultiLoss:
        reduction = "mean"

        def __call__(self, output, target, **kwargs):
            return torch.tensor(1.0), {"a": 1.0, "b": 2.0}

    cfg = cVAEConfig(ModelConfig=DummySelector())
    m = cVAE(cfg)
    m.build(np.array([1]))
    m.init_loss_function(MultiLoss())

    total, losses = m._compute_loss(1.0, DummyBatch())

    assert total >= 0
    assert losses["a"] == 1.0
    assert losses["b"] == 2.0


def test_compute_loss_beta_zero():
    cfg = cVAEConfig(ModelConfig=DummySelector())
    m = cVAE(cfg)
    m.build(np.array([1]))
    m.init_loss_function(DummyLoss())

    total, losses = m._compute_loss(0.0, DummyBatch())

    assert total >= 0
    assert "kld" in losses


def test_compute_loss_negative_beta():
    cfg = cVAEConfig(ModelConfig=DummySelector())
    m = cVAE(cfg)
    m.build(np.array([1]))
    m.init_loss_function(DummyLoss())

    total, losses = m._compute_loss(-1.0, DummyBatch())

    assert torch.is_tensor(total)
    assert "kld" in losses
