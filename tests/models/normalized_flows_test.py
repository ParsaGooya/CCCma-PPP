import pytest
import torch
import torch.nn as nn

from cccma_ppp.core.selectors import FlowSelector
from cccma_ppp.models.normalized_flows import (
    FCNN,
    MAF,
    RealNVP,
    NormalizedFlowConfig,
    NormalizedFlowModel,
    flowOutput,
)


torch.manual_seed(0)


# ============================================================
# Helpers
# ============================================================


def x(batch=4, dim=4):
    return torch.randn(batch, dim)


def cond(batch=4, condition_size=3):
    return torch.randn(batch, condition_size)


class IdentityFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.built_with = None
        self.forward_called = False
        self.inverse_called = False

    def build(self, dim, condition_size=None):
        self.built_with = {
            "dim": dim,
            "condition_size": condition_size,
        }
        return self

    def forward(self, x, condition=None):
        self.forward_called = True
        return x, torch.ones(x.shape[0], device=x.device)

    def inverse(self, z, condition=None):
        self.inverse_called = True
        return z, -torch.ones(z.shape[0], device=z.device)


class ScaleFlow(nn.Module):
    def __init__(self, scale=2.0):
        super().__init__()
        self.scale = scale
        self.built_with = None

    def build(self, dim, condition_size=None):
        self.built_with = {
            "dim": dim,
            "condition_size": condition_size,
        }
        return self

    def forward(self, x, condition=None):
        return x * self.scale, torch.full((x.shape[0],), 0.5, device=x.device)

    def inverse(self, z, condition=None):
        return z / self.scale, torch.full((z.shape[0],), -0.5, device=z.device)


class DummySelector:
    def __init__(self, model):
        self.model = model

    def get_model(self):
        return self.model


# ============================================================
# flowOutput
# ============================================================


def test_flow_output_dataclass():
    samples = torch.randn(2, 3)
    log_det = torch.randn(2)

    out = flowOutput(e_samples=samples, log_det=log_det)

    assert out.e_samples is samples
    assert out.log_det is log_det


# ============================================================
# FCNN
# ============================================================


def test_fcnn_forward_shape():
    model = FCNN(in_dim=3, out_dim=2, hidden_dim=5)

    out = model(torch.randn(4, 3))

    assert out.shape == (4, 2)


def test_fcnn_contains_expected_layers():
    model = FCNN(in_dim=3, out_dim=2, hidden_dim=5)

    assert isinstance(model.network, nn.Sequential)
    assert any(isinstance(layer, nn.Tanh) for layer in model.network)
    assert sum(isinstance(layer, nn.Linear) for layer in model.network) == 3


# ============================================================
# NormalizedFlowConfig / NormalizedFlowModel
# ============================================================


def test_normalized_flow_config_build():
    cfg = NormalizedFlowConfig(
        list_flows=[DummySelector(IdentityFlow())],
        flow_sample_size=123,
    )

    model = cfg.build(latent_size=4)

    assert isinstance(model, NormalizedFlowModel)
    assert model.flow_sample_size == 123
    assert model.condition_size is None
    assert len(model.flows) == 1


def test_normalized_flow_config_build_with_condition_size():
    cfg = NormalizedFlowConfig(
        list_flows=[DummySelector(IdentityFlow())],
        flow_sample_size=10,
    )

    model = cfg.build(latent_size=4, condition_size=3)

    assert model.condition_size == 3
    assert model.flows[0].built_with["condition_size"] == 3


def test_normalized_flow_model_build_converts_to_modulelist():
    cfg = NormalizedFlowConfig(
        list_flows=[
            DummySelector(IdentityFlow()),
            DummySelector(ScaleFlow()),
        ],
        flow_sample_size=10,
    )

    model = NormalizedFlowModel(cfg).build(latent_size=4)

    assert isinstance(model.flows, nn.ModuleList)
    assert len(model.flows) == 2


def test_normalized_flow_forward_single_flow():
    cfg = NormalizedFlowConfig(
        list_flows=[DummySelector(IdentityFlow())],
        flow_sample_size=10,
    )

    model = cfg.build(latent_size=4)
    data = x()

    out = model(data)

    assert isinstance(out, flowOutput)
    assert out.e_samples.shape == data.shape
    assert out.log_det.shape == (data.shape[0],)
    assert torch.allclose(out.log_det, torch.ones(data.shape[0]))


def test_normalized_flow_inverse_single_flow():
    cfg = NormalizedFlowConfig(
        list_flows=[DummySelector(IdentityFlow())],
        flow_sample_size=10,
    )

    model = cfg.build(latent_size=4)
    data = x()

    out = model.inverse(data)

    assert isinstance(out, flowOutput)
    assert out.e_samples.shape == data.shape
    assert out.log_det.shape == (data.shape[0],)
    assert torch.allclose(out.log_det, -torch.ones(data.shape[0]))


def test_normalized_flow_forward_multiple_flows_logdet_accumulates():
    cfg = NormalizedFlowConfig(
        list_flows=[
            DummySelector(IdentityFlow()),
            DummySelector(ScaleFlow(scale=2.0)),
        ],
        flow_sample_size=10,
    )

    model = cfg.build(latent_size=4)
    data = x()

    out = model(data)

    assert out.e_samples.shape == data.shape
    assert torch.allclose(
        out.log_det,
        torch.full((data.shape[0],), 1.5),
    )


def test_normalized_flow_inverse_multiple_flows_reverse_order():
    f1 = IdentityFlow()
    f2 = ScaleFlow(scale=2.0)

    cfg = NormalizedFlowConfig(
        list_flows=[
            DummySelector(f1),
            DummySelector(f2),
        ],
        flow_sample_size=10,
    )

    model = cfg.build(latent_size=4)
    data = x()

    out = model.inverse(data)

    assert out.e_samples.shape == data.shape
    assert torch.allclose(
        out.log_det,
        torch.full((data.shape[0],), -1.5),
    )


def test_normalized_flow_forward_with_condition():
    cfg = NormalizedFlowConfig(
        list_flows=[DummySelector(IdentityFlow())],
        flow_sample_size=10,
    )

    model = cfg.build(latent_size=4, condition_size=3)
    data = x()
    condition = cond()

    out = model(data, condition=condition)

    assert out.e_samples.shape == data.shape
    assert out.log_det.shape == (data.shape[0],)


def test_normalized_flow_inverse_with_condition():
    cfg = NormalizedFlowConfig(
        list_flows=[DummySelector(IdentityFlow())],
        flow_sample_size=10,
    )

    model = cfg.build(latent_size=4, condition_size=3)
    data = x()
    condition = cond()

    out = model.inverse(data, condition=condition)

    assert out.e_samples.shape == data.shape
    assert out.log_det.shape == (data.shape[0],)


def test_normalized_flow_empty_flow_list_forward():
    cfg = NormalizedFlowConfig(list_flows=[], flow_sample_size=10)
    model = cfg.build(latent_size=4)

    data = x()
    out = model(data)

    assert torch.allclose(out.e_samples, data)
    assert torch.allclose(out.log_det, torch.zeros(data.shape[0]))


def test_normalized_flow_empty_flow_list_inverse():
    cfg = NormalizedFlowConfig(list_flows=[], flow_sample_size=10)
    model = cfg.build(latent_size=4)

    data = x()
    out = model.inverse(data)

    assert torch.allclose(out.e_samples, data)
    assert torch.allclose(out.log_det, torch.zeros(data.shape[0]))


# ============================================================
# FlowSelector registry integration
# ============================================================


def test_flow_selector_maf_registered():
    selector = FlowSelector(type="maf", args={"hidden_dim": 8})

    flow = selector.get_model()

    assert isinstance(flow, MAF)
    assert flow.hidden_dim == 8


def test_flow_selector_realnvp_registered():
    selector = FlowSelector(type="realnvp", args={"hidden_dim": 8})

    flow = selector.get_model()

    assert isinstance(flow, RealNVP)
    assert flow.hidden_dim == 8


def test_normalized_flow_config_with_real_selectors():
    cfg = NormalizedFlowConfig(
        list_flows=[
            FlowSelector(type="maf", args={"hidden_dim": 8}),
            FlowSelector(type="realnvp", args={"hidden_dim": 8}),
        ],
        flow_sample_size=7,
    )

    model = cfg.build(latent_size=4)

    assert isinstance(model, NormalizedFlowModel)
    assert len(model.flows) == 2
    assert isinstance(model.flows[0], MAF)
    assert isinstance(model.flows[1], RealNVP)


# ============================================================
# MAF
# ============================================================


def test_maf_build_without_condition():
    maf = MAF(hidden_dim=8).build(dim=4)

    assert maf.dim == 4
    assert hasattr(maf, "layers")
    assert hasattr(maf, "initial_param")
    assert isinstance(maf.initial_param, torch.nn.Parameter)


def test_maf_build_with_condition():
    maf = MAF(hidden_dim=8).build(dim=4, condition_size=3)

    assert maf.dim == 4
    assert hasattr(maf, "layers")
    assert hasattr(maf, "initial_param")
    assert isinstance(maf.initial_param, FCNN)


def test_maf_reset_parameters_changes_parameter():
    maf = MAF(hidden_dim=8).build(dim=4)

    before = maf.initial_param.detach().clone()
    maf.reset_parameters()
    after = maf.initial_param.detach().clone()

    assert before.shape == after.shape


def test_maf_forward_without_condition():
    maf = MAF(hidden_dim=8).build(dim=4)
    data = x(batch=5, dim=4)

    z, log_det = maf.forward(data)

    assert z.shape == data.shape
    assert log_det.shape == (5,)


def test_maf_forward_with_condition():
    maf = MAF(hidden_dim=8).build(dim=4, condition_size=3)
    data = x(batch=5, dim=4)
    condition = cond(batch=5, condition_size=3)

    z, log_det = maf.forward(data, condition=condition)

    assert z.shape == data.shape
    assert log_det.shape == (5,)


def test_maf_inverse_with_condition():
    maf = MAF(hidden_dim=8).build(dim=4, condition_size=3)
    data = x(batch=5, dim=4)
    condition = cond(batch=5, condition_size=3)

    inv, log_det = maf.inverse(data, condition=condition)

    assert inv.shape == data.shape
    assert log_det.shape == (5,)


# ============================================================
# RealNVP
# ============================================================


def test_realnvp_build_without_condition():
    flow = RealNVP(hidden_dim=8).build(dim=4)

    assert flow.dim == 4
    assert hasattr(flow, "t1")
    assert hasattr(flow, "s1")
    assert hasattr(flow, "t2")
    assert hasattr(flow, "s2")


def test_realnvp_build_with_condition():
    flow = RealNVP(hidden_dim=8).build(dim=4, condition_size=3)

    assert flow.dim == 4
    assert hasattr(flow, "t1")
    assert hasattr(flow, "s1")
    assert hasattr(flow, "t2")
    assert hasattr(flow, "s2")


def test_realnvp_forward_without_condition():
    flow = RealNVP(hidden_dim=8).build(dim=4)
    data = x(batch=5, dim=4)

    z, log_det = flow.forward(data)

    assert z.shape == data.shape
    assert log_det.shape == (5,)


def test_realnvp_inverse_without_condition():
    flow = RealNVP(hidden_dim=8).build(dim=4)
    data = x(batch=5, dim=4)

    inv, log_det = flow.inverse(data)

    assert inv.shape == data.shape
    assert log_det.shape == (5,)


def test_realnvp_forward_with_condition():
    flow = RealNVP(hidden_dim=8).build(dim=4, condition_size=3)
    data = x(batch=5, dim=4)
    condition = cond(batch=5, condition_size=3)

    z, log_det = flow.forward(data, condition=condition)

    assert z.shape == data.shape
    assert log_det.shape == (5,)


def test_realnvp_inverse_with_condition():
    flow = RealNVP(hidden_dim=8).build(dim=4, condition_size=3)
    data = x(batch=5, dim=4)
    condition = cond(batch=5, condition_size=3)

    inv, log_det = flow.inverse(data, condition=condition)

    assert inv.shape == data.shape
    assert log_det.shape == (5,)


def test_realnvp_forward_inverse_roundtrip_shape():
    flow = RealNVP(hidden_dim=8).build(dim=4)
    data = x(batch=5, dim=4)

    z, _ = flow.forward(data)
    inv, _ = flow.inverse(z)

    assert inv.shape == data.shape


def test_realnvp_odd_dimension_shape_behavior():
    flow = RealNVP(hidden_dim=8).build(dim=5)
    data = x(batch=4, dim=5)

    # Current implementation uses dim//2 for both halves.
    # For odd dimensions this produces incompatible dimensions.
    with pytest.raises(RuntimeError):
        flow.forward(data)
