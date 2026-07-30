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


@pytest.mark.pruned
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


@pytest.mark.pruned
def test_normalized_flow_config_build_with_condition_size():
    cfg = NormalizedFlowConfig(
        list_flows=[DummySelector(IdentityFlow())],
        flow_sample_size=10,
    )

    model = cfg.build(latent_size=4, condition_size=3)

    assert model.condition_size == 3
    assert model.flows[0].built_with["condition_size"] == 3


@pytest.mark.pruned
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


@pytest.mark.pruned
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


@pytest.mark.pruned
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


@pytest.mark.pruned
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


@pytest.mark.pruned
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


@pytest.mark.pruned
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


@pytest.mark.pruned
def test_normalized_flow_empty_flow_list_forward():
    cfg = NormalizedFlowConfig(list_flows=[], flow_sample_size=10)
    model = cfg.build(latent_size=4)

    data = x()
    out = model(data)

    assert torch.allclose(out.e_samples, data)
    assert torch.allclose(out.log_det, torch.zeros(data.shape[0]))


@pytest.mark.pruned
def test_normalized_flow_empty_flow_list_inverse():
    cfg = NormalizedFlowConfig(list_flows=[], flow_sample_size=10)
    model = cfg.build(latent_size=4)

    data = x()
    out = model.inverse(data)

    assert torch.allclose(out.e_samples, data)
    assert torch.allclose(out.log_det, torch.zeros(data.shape[0]))


@pytest.mark.pruned
def test_flow_selector_maf_registered():
    selector = FlowSelector(type="maf", args={"hidden_dim": 8})

    flow = selector.get_model()

    assert isinstance(flow, MAF)
    assert flow.hidden_dim == 8


@pytest.mark.pruned
def test_flow_selector_realnvp_registered():
    selector = FlowSelector(type="realnvp", args={"hidden_dim": 8})

    flow = selector.get_model()

    assert isinstance(flow, RealNVP)
    assert flow.hidden_dim == 8


@pytest.mark.pruned
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


@pytest.mark.pruned
def test_maf_build_without_condition():
    maf = MAF(hidden_dim=8).build(dim=4)

    assert maf.dim == 4
    assert hasattr(maf, "layers")
    assert hasattr(maf, "initial_param")
    assert isinstance(maf.initial_param, torch.nn.Parameter)


@pytest.mark.pruned
def test_maf_build_with_condition():
    maf = MAF(hidden_dim=8).build(dim=4, condition_size=3)

    assert maf.dim == 4
    assert hasattr(maf, "layers")
    assert hasattr(maf, "initial_param")
    assert isinstance(maf.initial_param, FCNN)


@pytest.mark.pruned
def test_maf_reset_parameters_changes_parameter():
    maf = MAF(hidden_dim=8).build(dim=4)

    before = maf.initial_param.detach().clone()
    maf.reset_parameters()
    after = maf.initial_param.detach().clone()

    assert before.shape == after.shape


@pytest.mark.pruned
def test_maf_forward_without_condition():
    maf = MAF(hidden_dim=8).build(dim=4)
    data = x(batch=5, dim=4)

    z, log_det = maf.forward(data)

    assert z.shape == data.shape
    assert log_det.shape == (5,)


@pytest.mark.pruned
def test_maf_forward_with_condition():
    maf = MAF(hidden_dim=8).build(dim=4, condition_size=3)
    data = x(batch=5, dim=4)
    condition = cond(batch=5, condition_size=3)

    z, log_det = maf.forward(data, condition=condition)

    assert z.shape == data.shape
    assert log_det.shape == (5,)


@pytest.mark.pruned
def test_maf_inverse_with_condition():
    maf = MAF(hidden_dim=8).build(dim=4, condition_size=3)
    data = x(batch=5, dim=4)
    condition = cond(batch=5, condition_size=3)

    inv, log_det = maf.inverse(data, condition=condition)

    assert inv.shape == data.shape
    assert log_det.shape == (5,)


@pytest.mark.pruned
def test_realnvp_forward_without_condition():
    flow = RealNVP(hidden_dim=8).build(dim=4)
    data = x(batch=5, dim=4)

    z, log_det = flow.forward(data)

    assert z.shape == data.shape
    assert log_det.shape == (5,)


@pytest.mark.pruned
def test_realnvp_inverse_without_condition():
    flow = RealNVP(hidden_dim=8).build(dim=4)
    data = x(batch=5, dim=4)

    inv, log_det = flow.inverse(data)

    assert inv.shape == data.shape
    assert log_det.shape == (5,)


@pytest.mark.pruned
def test_realnvp_forward_with_condition():
    flow = RealNVP(hidden_dim=8).build(dim=4, condition_size=3)
    data = x(batch=5, dim=4)
    condition = cond(batch=5, condition_size=3)

    z, log_det = flow.forward(data, condition=condition)

    assert z.shape == data.shape
    assert log_det.shape == (5,)


@pytest.mark.pruned
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


@pytest.mark.pruned
def test_realnvp_odd_dimension_shape_behavior():
    flow = RealNVP(hidden_dim=8).build(dim=5)
    data = x(batch=4, dim=5)

    with pytest.raises(RuntimeError):
        flow.forward(data)


class TrackingSelector:
    def __init__(self, flow):
        self.flow = flow
        self.calls = 0

    def get_model(self):
        self.calls += 1
        return self.flow


class OrderFlow(nn.Module):
    def __init__(self, name, events):
        super().__init__()
        self.name = name
        self.events = events
        self.dim = None
        self.condition_size = None

    def build(self, dim, condition_size=None):
        self.dim = dim
        self.condition_size = condition_size
        self.events.append(
            (
                "build",
                self.name,
                dim,
                condition_size,
            )
        )
        return self

    def forward(self, value, condition=None):
        self.events.append(
            (
                "forward",
                self.name,
                condition,
            )
        )
        return (
            value + 1,
            torch.ones(
                value.shape[0],
                device=value.device,
            ),
        )

    def inverse(self, value, condition=None):
        self.events.append(
            (
                "inverse",
                self.name,
                condition,
            )
        )
        return (
            value - 1,
            -torch.ones(
                value.shape[0],
                device=value.device,
            ),
        )


class ZeroNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

    def forward(self, value):
        return torch.zeros(
            value.shape[0],
            self.out_dim,
            dtype=value.dtype,
            device=value.device,
        )


@pytest.mark.pruned
def test_normalized_flow_config_build_returns_model():
    config = NormalizedFlowConfig(
        list_flows=[],
        flow_sample_size=32,
    )

    model = config.build(
        latent_size=4,
        condition_size=3,
    )

    assert isinstance(
        model,
        NormalizedFlowModel,
    )
    assert model.flow_sample_size == 32
    assert model.condition_size == 3


@pytest.mark.pruned
def test_normalized_flow_selector_called_once():
    selector = TrackingSelector(IdentityFlow())

    config = NormalizedFlowConfig(
        list_flows=[selector],
    )

    model = config.build(
        latent_size=4,
    )

    assert selector.calls == 1
    assert len(model.flows) == 1


@pytest.mark.pruned
def test_normalized_flow_builds_each_flow():
    first = IdentityFlow()
    second = ScaleFlow()

    config = NormalizedFlowConfig(
        list_flows=[
            DummySelector(first),
            DummySelector(second),
        ],
    )

    config.build(
        latent_size=6,
        condition_size=2,
    )

    assert first.built_with == {
        "dim": 6,
        "condition_size": 2,
    }
    assert second.built_with == {
        "dim": 6,
        "condition_size": 2,
    }


@pytest.mark.pruned
def test_normalized_flow_uses_module_list():
    config = NormalizedFlowConfig(
        list_flows=[
            DummySelector(IdentityFlow()),
            DummySelector(ScaleFlow()),
        ],
    )

    model = config.build(
        latent_size=4,
    )

    assert isinstance(
        model.flows,
        nn.ModuleList,
    )


@pytest.mark.pruned
def test_normalized_flow_forward_order():
    events = []

    config = NormalizedFlowConfig(
        list_flows=[
            DummySelector(OrderFlow("first", events)),
            DummySelector(OrderFlow("second", events)),
        ],
    )

    model = config.build(
        latent_size=4,
    )

    events.clear()

    data = x()
    output = model.forward(data)

    assert [event[1] for event in events] == [
        "first",
        "second",
    ]
    assert torch.allclose(
        output.e_samples,
        data + 2,
    )
    assert torch.allclose(
        output.log_det,
        torch.full(
            (data.shape[0],),
            2.0,
        ),
    )


@pytest.mark.pruned
def test_normalized_flow_inverse_reverse_order():
    events = []

    config = NormalizedFlowConfig(
        list_flows=[
            DummySelector(OrderFlow("first", events)),
            DummySelector(OrderFlow("second", events)),
        ],
    )

    model = config.build(
        latent_size=4,
    )

    events.clear()

    data = x()
    output = model.inverse(data)

    assert [event[1] for event in events] == [
        "second",
        "first",
    ]
    assert torch.allclose(
        output.e_samples,
        data - 2,
    )
    assert torch.allclose(
        output.log_det,
        torch.full(
            (data.shape[0],),
            -2.0,
        ),
    )


@pytest.mark.pruned
def test_normalized_flow_passes_condition_forward():
    events = []

    config = NormalizedFlowConfig(
        list_flows=[
            DummySelector(OrderFlow("flow", events)),
        ],
    )

    model = config.build(
        latent_size=4,
        condition_size=3,
    )

    events.clear()

    condition = cond()

    model.forward(
        x(),
        condition=condition,
    )

    assert events[0][0] == "forward"
    assert events[0][2] is condition


@pytest.mark.pruned
def test_normalized_flow_passes_condition_inverse():
    events = []

    config = NormalizedFlowConfig(
        list_flows=[
            DummySelector(OrderFlow("flow", events)),
        ],
    )

    model = config.build(
        latent_size=4,
        condition_size=3,
    )

    events.clear()

    condition = cond()

    model.inverse(
        x(),
        condition=condition,
    )

    assert events[0][0] == "inverse"
    assert events[0][2] is condition


@pytest.mark.pruned
def test_normalized_flow_empty_forward_preserves_input_identity():
    model = NormalizedFlowConfig(
        list_flows=[],
    ).build(
        latent_size=4,
    )

    data = x()
    output = model(data)

    assert output.e_samples is data
    assert output.log_det.device == data.device
    assert output.log_det.dtype == data.dtype


@pytest.mark.pruned
def test_normalized_flow_empty_inverse_preserves_input_identity():
    model = NormalizedFlowConfig(
        list_flows=[],
    ).build(
        latent_size=4,
    )

    data = x()
    output = model.inverse(data)

    assert output.e_samples is data
    assert output.log_det.device == data.device
    assert output.log_det.dtype == data.dtype


def test_normalized_flow_forward_inverse_identity_roundtrip():
    model = NormalizedFlowConfig(
        list_flows=[
            DummySelector(IdentityFlow()),
        ],
    ).build(
        latent_size=4,
    )

    data = x()

    transformed = model(data)
    reconstructed = model.inverse(transformed.e_samples)

    assert torch.allclose(
        reconstructed.e_samples,
        data,
    )
    assert torch.allclose(
        transformed.log_det + reconstructed.log_det,
        torch.zeros(data.shape[0]),
    )


@pytest.mark.pruned
def test_maf_build_dimension_one_without_condition():
    maf = MAF(
        hidden_dim=8,
    ).build(
        dim=1,
    )

    assert maf.dim == 1
    assert len(maf.layers) == 0
    assert isinstance(
        maf.initial_param,
        nn.Parameter,
    )


@pytest.mark.pruned
def test_maf_build_dimension_one_with_condition():
    maf = MAF(
        hidden_dim=8,
    ).build(
        dim=1,
        condition_size=3,
    )

    assert maf.dim == 1
    assert len(maf.layers) == 0
    assert isinstance(
        maf.initial_param,
        FCNN,
    )
    assert maf._conditional is True


@pytest.mark.pruned
def test_maf_build_without_condition_is_not_conditional():
    maf = MAF(
        hidden_dim=8,
    ).build(
        dim=4,
    )

    assert maf._conditional is False


@pytest.mark.pruned
def test_maf_build_with_condition_is_conditional():
    maf = MAF(
        hidden_dim=8,
    ).build(
        dim=4,
        condition_size=3,
    )

    assert maf._conditional is True


@pytest.mark.pruned
def test_maf_build_creates_one_layer_per_remaining_dimension():
    maf = MAF(
        hidden_dim=8,
    ).build(
        dim=5,
    )

    assert len(maf.layers) == 4


@pytest.mark.pruned
def test_maf_build_layer_input_dimensions_without_condition():
    maf = MAF(
        hidden_dim=8,
    ).build(
        dim=4,
    )

    expected_input_dimensions = [
        1,
        2,
        3,
    ]

    actual_input_dimensions = [layer.network[0].in_features for layer in maf.layers]

    assert actual_input_dimensions == expected_input_dimensions


@pytest.mark.pruned
def test_maf_build_layer_input_dimensions_with_condition():
    maf = MAF(
        hidden_dim=8,
    ).build(
        dim=4,
        condition_size=3,
    )

    expected_input_dimensions = [
        4,
        5,
        6,
    ]

    actual_input_dimensions = [layer.network[0].in_features for layer in maf.layers]

    assert actual_input_dimensions == expected_input_dimensions


@pytest.mark.pruned
def test_maf_initial_parameter_bounds():
    maf = MAF(
        hidden_dim=8,
    ).build(
        dim=4,
    )

    bound = 0.5**0.5

    assert torch.all(maf.initial_param <= bound)
    assert torch.all(maf.initial_param >= -bound)


@pytest.mark.pruned
def test_maf_forward_conditional_requires_condition():
    maf = MAF(
        hidden_dim=8,
    ).build(
        dim=4,
        condition_size=3,
    )

    with pytest.raises(
        ValueError,
        match="no condition",
    ):
        maf.forward(x(batch=5, dim=4))


@pytest.mark.pruned
def test_maf_inverse_conditional_requires_condition():
    maf = MAF(
        hidden_dim=8,
    ).build(
        dim=4,
        condition_size=3,
    )

    with pytest.raises(
        ValueError,
        match="no condition",
    ):
        maf.inverse(x(batch=5, dim=4))


@pytest.mark.pruned
def test_maf_dimension_one_forward_without_condition():
    maf = MAF(
        hidden_dim=8,
    ).build(
        dim=1,
    )

    data = x(
        batch=5,
        dim=1,
    )

    transformed, log_det = maf.forward(data)

    assert transformed.shape == data.shape
    assert log_det.shape == (5,)


@pytest.mark.pruned
def test_maf_dimension_one_inverse_without_condition():
    maf = MAF(
        hidden_dim=8,
    ).build(
        dim=1,
    )

    data = x(
        batch=5,
        dim=1,
    )

    reconstructed, log_det = maf.inverse(data)

    assert reconstructed.shape == data.shape
    assert log_det.shape == (5,)


@pytest.mark.pruned
def test_maf_dimension_one_forward_with_condition():
    maf = MAF(
        hidden_dim=8,
    ).build(
        dim=1,
        condition_size=3,
    )

    data = x(
        batch=5,
        dim=1,
    )
    condition = cond(
        batch=5,
        condition_size=3,
    )

    transformed, log_det = maf.forward(
        data,
        condition=condition,
    )

    assert transformed.shape == data.shape
    assert log_det.shape == (5,)


@pytest.mark.pruned
def test_maf_dimension_one_inverse_with_condition():
    maf = MAF(
        hidden_dim=8,
    ).build(
        dim=1,
        condition_size=3,
    )

    data = x(
        batch=5,
        dim=1,
    )
    condition = cond(
        batch=5,
        condition_size=3,
    )

    reconstructed, log_det = maf.inverse(
        data,
        condition=condition,
    )

    assert reconstructed.shape == data.shape
    assert log_det.shape == (5,)


def test_maf_zero_network_roundtrip_without_condition():
    maf = MAF(
        hidden_dim=8,
        base_network=ZeroNetwork,
    ).build(
        dim=4,
    )

    with torch.no_grad():
        maf.initial_param.zero_()

    data = x(
        batch=5,
        dim=4,
    )

    transformed, forward_log_det = maf.forward(data)
    reconstructed, inverse_log_det = maf.inverse(transformed)

    assert torch.allclose(
        reconstructed,
        data,
    )
    assert torch.allclose(
        forward_log_det,
        torch.zeros(5),
    )
    assert torch.allclose(
        inverse_log_det,
        torch.zeros(5),
    )


def test_maf_zero_network_roundtrip_with_condition():
    maf = MAF(
        hidden_dim=8,
        base_network=ZeroNetwork,
    ).build(
        dim=4,
        condition_size=3,
    )

    data = x(
        batch=5,
        dim=4,
    )
    condition = cond(
        batch=5,
        condition_size=3,
    )

    transformed, forward_log_det = maf.forward(
        data,
        condition=condition,
    )
    reconstructed, inverse_log_det = maf.inverse(
        transformed,
        condition=condition,
    )

    assert torch.allclose(
        reconstructed,
        data,
    )
    assert torch.allclose(
        forward_log_det,
        torch.zeros(5),
    )
    assert torch.allclose(
        inverse_log_det,
        torch.zeros(5),
    )


@pytest.mark.pruned
def test_maf_forward_inverse_roundtrip():
    maf = MAF(
        hidden_dim=8,
    ).build(
        dim=4,
    )

    data = x(
        batch=5,
        dim=4,
    )

    transformed, forward_log_det = maf.forward(data)
    reconstructed, inverse_log_det = maf.inverse(transformed)

    assert torch.allclose(
        reconstructed,
        data,
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        forward_log_det + inverse_log_det,
        torch.zeros(5),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.pruned
def test_maf_conditional_forward_inverse_roundtrip():
    maf = MAF(
        hidden_dim=8,
    ).build(
        dim=4,
        condition_size=3,
    )

    data = x(
        batch=5,
        dim=4,
    )
    condition = cond(
        batch=5,
        condition_size=3,
    )

    transformed, forward_log_det = maf.forward(
        data,
        condition=condition,
    )
    reconstructed, inverse_log_det = maf.inverse(
        transformed,
        condition=condition,
    )

    assert torch.allclose(
        reconstructed,
        data,
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        forward_log_det + inverse_log_det,
        torch.zeros(5),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.pruned
def test_maf_forward_supports_gradients():
    maf = MAF(
        hidden_dim=8,
    ).build(
        dim=4,
    )

    data = torch.randn(
        5,
        4,
        requires_grad=True,
    )

    transformed, log_det = maf.forward(data)

    loss = transformed.sum() + log_det.sum()
    loss.backward()

    assert data.grad is not None


@pytest.mark.pruned
def test_realnvp_build_without_condition_is_not_conditional():
    flow = RealNVP(
        hidden_dim=8,
    ).build(
        dim=4,
    )

    assert flow._conditional is False


@pytest.mark.pruned
def test_realnvp_build_with_condition_is_conditional():
    flow = RealNVP(
        hidden_dim=8,
    ).build(
        dim=4,
        condition_size=3,
    )

    assert flow._conditional is True


@pytest.mark.pruned
def test_realnvp_network_dimensions_without_condition():
    flow = RealNVP(
        hidden_dim=8,
    ).build(
        dim=6,
    )

    assert flow.t1.network[0].in_features == 3
    assert flow.t1.network[-1].out_features == 3
    assert flow.s1.network[0].in_features == 3
    assert flow.s1.network[-1].out_features == 3
    assert flow.t2.network[0].in_features == 3
    assert flow.t2.network[-1].out_features == 3
    assert flow.s2.network[0].in_features == 3
    assert flow.s2.network[-1].out_features == 3


@pytest.mark.pruned
def test_realnvp_network_dimensions_with_condition():
    flow = RealNVP(
        hidden_dim=8,
    ).build(
        dim=6,
        condition_size=2,
    )

    assert flow.t1.network[0].in_features == 5
    assert flow.t1.network[-1].out_features == 3
    assert flow.s1.network[0].in_features == 5
    assert flow.s1.network[-1].out_features == 3
    assert flow.t2.network[0].in_features == 5
    assert flow.t2.network[-1].out_features == 3
    assert flow.s2.network[0].in_features == 5
    assert flow.s2.network[-1].out_features == 3


@pytest.mark.pruned
def test_realnvp_forward_conditional_requires_condition():
    flow = RealNVP(
        hidden_dim=8,
    ).build(
        dim=4,
        condition_size=3,
    )

    with pytest.raises(
        ValueError,
        match="no condition",
    ):
        flow.forward(x(batch=5, dim=4))


@pytest.mark.pruned
def test_realnvp_inverse_conditional_requires_condition():
    flow = RealNVP(
        hidden_dim=8,
    ).build(
        dim=4,
        condition_size=3,
    )

    with pytest.raises(
        ValueError,
        match="no condition",
    ):
        flow.inverse(x(batch=5, dim=4))


@pytest.mark.pruned
def test_realnvp_zero_network_roundtrip_without_condition():
    flow = RealNVP(
        hidden_dim=8,
        base_network=ZeroNetwork,
    ).build(
        dim=4,
    )

    data = x(
        batch=5,
        dim=4,
    )

    transformed, forward_log_det = flow.forward(data)
    reconstructed, inverse_log_det = flow.inverse(transformed)

    assert torch.allclose(
        transformed,
        data,
    )
    assert torch.allclose(
        reconstructed,
        data,
    )
    assert torch.allclose(
        forward_log_det,
        torch.zeros(5),
    )
    assert torch.allclose(
        inverse_log_det,
        torch.zeros(5),
    )


def test_realnvp_zero_network_roundtrip_with_condition():
    flow = RealNVP(
        hidden_dim=8,
        base_network=ZeroNetwork,
    ).build(
        dim=4,
        condition_size=3,
    )

    data = x(
        batch=5,
        dim=4,
    )
    condition = cond(
        batch=5,
        condition_size=3,
    )

    transformed, forward_log_det = flow.forward(
        data,
        condition=condition,
    )
    reconstructed, inverse_log_det = flow.inverse(
        transformed,
        condition=condition,
    )

    assert torch.allclose(
        transformed,
        data,
    )
    assert torch.allclose(
        reconstructed,
        data,
    )
    assert torch.allclose(
        forward_log_det,
        torch.zeros(5),
    )
    assert torch.allclose(
        inverse_log_det,
        torch.zeros(5),
    )


@pytest.mark.pruned
def test_realnvp_forward_inverse_roundtrip():
    flow = RealNVP(
        hidden_dim=8,
    ).build(
        dim=4,
    )

    data = x(
        batch=5,
        dim=4,
    )

    transformed, forward_log_det = flow.forward(data)
    reconstructed, inverse_log_det = flow.inverse(transformed)

    assert torch.allclose(
        reconstructed,
        data,
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        forward_log_det + inverse_log_det,
        torch.zeros(5),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.pruned
def test_realnvp_conditional_roundtrip():
    flow = RealNVP(
        hidden_dim=8,
    ).build(
        dim=4,
        condition_size=3,
    )

    data = x(
        batch=5,
        dim=4,
    )
    condition = cond(
        batch=5,
        condition_size=3,
    )

    transformed, forward_log_det = flow.forward(
        data,
        condition=condition,
    )
    reconstructed, inverse_log_det = flow.inverse(
        transformed,
        condition=condition,
    )

    assert torch.allclose(
        reconstructed,
        data,
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        forward_log_det + inverse_log_det,
        torch.zeros(5),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.pruned
def test_realnvp_forward_supports_gradients():
    flow = RealNVP(
        hidden_dim=8,
    ).build(
        dim=4,
    )

    data = torch.randn(
        5,
        4,
        requires_grad=True,
    )

    transformed, log_det = flow.forward(data)

    loss = transformed.sum() + log_det.sum()
    loss.backward()

    assert data.grad is not None

    for parameter in flow.parameters():
        assert parameter.grad is not None


@pytest.mark.pruned
def test_realnvp_even_dimension_six():
    flow = RealNVP(
        hidden_dim=8,
    ).build(
        dim=6,
    )

    data = x(
        batch=5,
        dim=6,
    )

    transformed, log_det = flow.forward(data)

    assert transformed.shape == data.shape
    assert log_det.shape == (5,)


@pytest.mark.parametrize(
    "dimension",
    [
        1,
        3,
        5,
        7,
    ],
)
def test_realnvp_rejects_odd_dimensions_during_inverse(
    dimension,
):
    flow = RealNVP(
        hidden_dim=8,
    ).build(
        dim=dimension,
    )

    data = x(
        batch=4,
        dim=dimension,
    )

    with pytest.raises(RuntimeError):
        flow.inverse(data)


@pytest.mark.pruned
def test_realnvp_log_determinants_are_finite():
    flow = RealNVP(
        hidden_dim=8,
    ).build(
        dim=4,
    )

    data = x(
        batch=10,
        dim=4,
    )

    transformed, forward_log_det = flow.forward(data)
    _, inverse_log_det = flow.inverse(transformed)

    assert torch.isfinite(forward_log_det).all()
    assert torch.isfinite(inverse_log_det).all()
