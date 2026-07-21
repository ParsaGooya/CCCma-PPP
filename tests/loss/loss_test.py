import pytest
import torch
import xarray as xr
import numpy as np

from cccma_ppp.loss.loss import LossStepConfig, LosspipelineConfig, Losspipeline


class DummyLoss(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, data, target, target_mask=None, **kwargs):
        base = ((data - target) ** 2).mean()
        if kwargs.get("flip", False):
            base = -base
        return base


@pytest.fixture(autouse=True)
def patch_registry(monkeypatch):
    class DummyRegistry:
        def get(self, name, args):
            return DummyLoss(**args)

    monkeypatch.setattr(Losspipeline, "registery", DummyRegistry())


def make_weights():
    return xr.DataArray(
        np.ones((2, 3)),
        coords={"lat": np.arange(2), "lon": np.arange(3)},
        dims=("lat", "lon"),
    )


def make_data():
    return torch.ones(2, 1, 2, 3)


def make_target():
    return torch.zeros(2, 1, 2, 3)


@pytest.mark.pruned
def test_config_default_weights():
    cfg = LosspipelineConfig(loss_pipeline=[LossStepConfig(name="a")])
    assert cfg.loss_weights == [1.0]


@pytest.mark.pruned
def test_config_multiple_auto_weights():
    cfg = LosspipelineConfig(
        loss_pipeline=[LossStepConfig(name="a"), LossStepConfig(name="b")]
    )
    assert sum(cfg.loss_weights) == pytest.approx(1.0)


@pytest.mark.pruned
def test_config_weights_valid_branch():
    cfg = LosspipelineConfig(
        loss_pipeline=[LossStepConfig(name="a"), LossStepConfig(name="b")],
        loss_weights=[0.5, 0.5],
    )
    assert cfg.loss_weights == [0.5, 0.5]


@pytest.mark.pruned
def test_config_invalid_weight_length():
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        LosspipelineConfig(
            loss_pipeline=[LossStepConfig(name="a"), LossStepConfig(name="b")],
            loss_weights=[1.0],
        )


def test_config_invalid_weight_sum():
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        LosspipelineConfig(
            loss_pipeline=[LossStepConfig(name="a"), LossStepConfig(name="b")],
            loss_weights=[0.6, 0.3],
        )


@pytest.mark.pruned
def test_config_invalid_reduction():
    LosspipelineConfig(
        loss_pipeline=[LossStepConfig(name="a")],
        reduction="bad",
    )


def test_config_invalid_args_forbidden_keys():
    with pytest.raises(ValueError):
        LosspipelineConfig(
            loss_pipeline=[LossStepConfig(name="a", args={"weights": 1})]
        )


@pytest.mark.pruned
def test_config_collect_loss_types():
    cfg = LosspipelineConfig(
        loss_pipeline=[LossStepConfig(name="a"), LossStepConfig(name="b")]
    )
    assert "a" in cfg.loss_types and "b" in cfg.loss_types


@pytest.mark.pruned
def test_build_pipeline_basic():
    cfg = LosspipelineConfig(loss_pipeline=[LossStepConfig(name="a")])
    pipe = cfg.build(make_weights())
    assert len(pipe.pipeline) == 1


def test_build_pipeline_duplicate_names():
    cfg = LosspipelineConfig(
        loss_pipeline=[LossStepConfig(name="a"), LossStepConfig(name="a")]
    )
    pipe = cfg.build(make_weights())
    assert len(set(pipe.steps)) == 2


def test_build_pipeline_low_res_name_branch():
    cfg = LosspipelineConfig(
        loss_pipeline=[LossStepConfig(name="a", args={"low_ress_kernel_size": 2})]
    )
    pipe = cfg.build(make_weights())
    assert "low_ress" in pipe.steps[0]


@pytest.mark.pruned
def test_forward_single_loss():
    cfg = LosspipelineConfig(loss_pipeline=[LossStepConfig(name="a")])
    pipe = cfg.build(make_weights())
    loss, indiv = pipe(make_data(), make_target())
    assert isinstance(loss, torch.Tensor)
    assert len(indiv) == 1


def test_forward_multiple_losses_weighted():
    cfg = LosspipelineConfig(
        loss_pipeline=[LossStepConfig(name="a"), LossStepConfig(name="b")],
        loss_weights=[0.7, 0.3],
    )
    pipe = cfg.build(make_weights())
    loss, indiv = pipe(make_data(), make_target())
    assert len(indiv) == 2


@pytest.mark.pruned
def test_forward_dimension_check_pass():
    cfg = LosspipelineConfig(loss_pipeline=[LossStepConfig(name="a")])
    pipe = cfg.build(make_weights(), num_output_dimensions=2)
    pipe(make_data(), make_target())


@pytest.mark.pruned
def test_forward_dimension_check_fail():
    cfg = LosspipelineConfig(loss_pipeline=[LossStepConfig(name="a")])
    pipe = cfg.build(make_weights(), num_output_dimensions=2)
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        pipe(make_data(), torch.zeros(2, 1, 10))


def test_forward_print_loss_flag():
    cfg = LosspipelineConfig(loss_pipeline=[LossStepConfig(name="a")])
    pipe = cfg.build(make_weights())
    loss, indiv = pipe(make_data(), make_target(), print_loss=True)
    assert loss is not None


def test_forward_step_arguments_passthrough():
    cfg = LosspipelineConfig(loss_pipeline=[LossStepConfig(name="a")])
    pipe = cfg.build(make_weights())
    loss, _ = pipe(make_data(), make_target(), step_arguments={"flip": True})
    assert loss < 0


@pytest.mark.pruned
def test_forward_total_loss_branch():
    cfg = LosspipelineConfig(
        loss_pipeline=[LossStepConfig(name="a"), LossStepConfig(name="b")]
    )
    pipe = cfg.build(make_weights())
    loss, _ = pipe(make_data(), make_target())
    assert isinstance(loss, torch.Tensor)


def test_checked_dimensionality_cache():
    cfg = LosspipelineConfig(loss_pipeline=[LossStepConfig(name="a")])
    pipe = cfg.build(make_weights())
    pipe(make_data(), make_target())
    pipe(make_data(), make_target())
    assert pipe._checked_dimensionality is True