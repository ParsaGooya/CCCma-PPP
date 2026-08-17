import pytest
import torch
import xarray as xr
import numpy as np

from cccma_ppp.loss.loss import LossStepConfig, LosspipelineConfig, Losspipeline
from unittest.mock import Mock

from cccma_ppp.core.core_abc import GenerativeContext


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


def test_config_default_weights():
    cfg = LosspipelineConfig(loss_pipeline=[LossStepConfig(name="a")])
    assert cfg.loss_weights == [1.0]


def test_config_multiple_auto_weights():
    cfg = LosspipelineConfig(
        loss_pipeline=[LossStepConfig(name="a"), LossStepConfig(name="b")]
    )
    assert sum(cfg.loss_weights) == pytest.approx(1.0)


def test_config_weights_valid_branch():
    cfg = LosspipelineConfig(
        loss_pipeline=[LossStepConfig(name="a"), LossStepConfig(name="b")],
        loss_weights=[0.5, 0.5],
    )
    assert cfg.loss_weights == [0.5, 0.5]


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


def test_config_collect_loss_types():
    cfg = LosspipelineConfig(
        loss_pipeline=[LossStepConfig(name="a"), LossStepConfig(name="b")]
    )
    assert "a" in cfg.loss_types and "b" in cfg.loss_types


def test_build_pipeline_basic():
    cfg = LosspipelineConfig(loss_pipeline=[LossStepConfig(name="a")])
    pipe = cfg.build(make_weights(), num_output_dimensions=3)
    assert len(pipe.pipeline) == 1


def test_build_pipeline_duplicate_names():
    cfg = LosspipelineConfig(
        loss_pipeline=[LossStepConfig(name="a"), LossStepConfig(name="a")]
    )
    pipe = cfg.build(make_weights(), num_output_dimensions=3)
    assert len(set(pipe.steps)) == 2


def test_build_pipeline_low_res_name_branch():
    cfg = LosspipelineConfig(
        loss_pipeline=[LossStepConfig(name="a", args={"low_ress_kernel_size": 2})]
    )
    pipe = cfg.build(make_weights(), num_output_dimensions=3)
    assert "low_ress" in pipe.steps[0]


def test_forward_single_loss():
    cfg = LosspipelineConfig(loss_pipeline=[LossStepConfig(name="a")])
    pipe = cfg.build(make_weights(), num_output_dimensions=3)
    loss, indiv = pipe(make_data(), make_target())
    assert isinstance(loss, torch.Tensor)
    assert len(indiv) == 1


def test_forward_multiple_losses_weighted():
    cfg = LosspipelineConfig(
        loss_pipeline=[LossStepConfig(name="a"), LossStepConfig(name="b")],
        loss_weights=[0.7, 0.3],
    )
    pipe = cfg.build(make_weights(), num_output_dimensions=3)
    loss, indiv = pipe(make_data(), make_target())
    assert len(indiv) == 2


def test_forward_dimension_check_pass():
    cfg = LosspipelineConfig(loss_pipeline=[LossStepConfig(name="a")])
    pipe = cfg.build(make_weights(), num_output_dimensions=3)
    pipe(make_data(), make_target())


def test_forward_dimension_check_fail():
    cfg = LosspipelineConfig(loss_pipeline=[LossStepConfig(name="a")])
    pipe = cfg.build(make_weights(), num_output_dimensions=3)
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        pipe(make_data(), torch.zeros(2, 1, 10))


def test_forward_print_loss_flag():
    cfg = LosspipelineConfig(loss_pipeline=[LossStepConfig(name="a")])
    pipe = cfg.build(make_weights(), num_output_dimensions=3)
    loss, indiv = pipe(make_data(), make_target(), print_loss=True)
    assert loss is not None


def test_forward_step_arguments_passthrough():
    cfg = LosspipelineConfig(loss_pipeline=[LossStepConfig(name="a")])
    pipe = cfg.build(make_weights(), num_output_dimensions=3)
    loss, _ = pipe(make_data(), make_target(), step_arguments={"flip": True})
    assert loss < 0


def test_forward_total_loss_branch():
    cfg = LosspipelineConfig(
        loss_pipeline=[LossStepConfig(name="a"), LossStepConfig(name="b")]
    )
    pipe = cfg.build(make_weights(), num_output_dimensions=3)
    loss, _ = pipe(make_data(), make_target())
    assert isinstance(loss, torch.Tensor)


def test_checked_dimensionality_cache():
    cfg = LosspipelineConfig(loss_pipeline=[LossStepConfig(name="a")])
    pipe = cfg.build(make_weights(), num_output_dimensions=3)
    pipe(make_data(), make_target())
    pipe(make_data(), make_target())
    assert pipe._checked_dimensionality is True


@pytest.mark.parametrize(
    "forbidden_key",
    [
        "reduction",
        "weights",
        "num_output_dimensions",
        "generative_context",
    ],
)
def test_config_rejects_each_forbidden_step_argument(
    forbidden_key,
):
    with pytest.raises(
        ValueError,
        match="Do not specify",
    ):
        LosspipelineConfig(
            loss_pipeline=[
                LossStepConfig(
                    name="a",
                    args={forbidden_key: object()},
                )
            ]
        )


def test_config_allows_nonreserved_step_arguments():
    config = LosspipelineConfig(
        loss_pipeline=[
            LossStepConfig(
                name="a",
                args={
                    "alpha": 0.5,
                    "custom_option": True,
                },
            )
        ]
    )

    assert config.loss_pipeline[0].args == {
        "alpha": 0.5,
        "custom_option": True,
    }


def test_config_rejects_empty_pipeline():
    with pytest.raises(
        ValueError,
        match="at least one loss term",
    ):
        LosspipelineConfig(
            loss_pipeline=[],
        )


def test_config_loss_types_collapses_duplicate_names():
    config = LosspipelineConfig(
        loss_pipeline=[
            LossStepConfig(name="a"),
            LossStepConfig(name="a"),
            LossStepConfig(name="b"),
        ]
    )

    assert config.loss_types == {"a", "b"}


def test_config_auto_weights_three_steps():
    config = LosspipelineConfig(
        loss_pipeline=[
            LossStepConfig(name="a"),
            LossStepConfig(name="b"),
            LossStepConfig(name="c"),
        ]
    )

    assert config.loss_weights == pytest.approx(
        [
            1 / 3,
            1 / 3,
            1 / 3,
        ]
    )


def test_config_accepts_single_explicit_weight():
    config = LosspipelineConfig(
        loss_pipeline=[
            LossStepConfig(name="a"),
        ],
        loss_weights=[1.0],
    )

    assert config.loss_weights == [1.0]


def test_config_nonlist_weights_are_replaced_with_defaults():
    config = LosspipelineConfig(
        loss_pipeline=[
            LossStepConfig(name="a"),
            LossStepConfig(name="b"),
        ],
        loss_weights=None,
    )

    assert config.loss_weights == [0.5, 0.5]


def test_config_build_forwards_all_arguments(
    monkeypatch,
):
    captured = {}

    class FakePipeline:
        def __init__(
            self,
            config,
            weights,
            num_output_dimensions,
            generative_context=None,
        ):
            captured["config"] = config
            captured["weights"] = weights
            captured["num_output_dimensions"] = num_output_dimensions
            captured["generative_context"] = generative_context

    monkeypatch.setattr(
        "cccma_ppp.loss.loss.Losspipeline",
        FakePipeline,
    )

    config = LosspipelineConfig(
        loss_pipeline=[
            LossStepConfig(name="a"),
        ]
    )
    weights = make_weights()
    context = object()

    result = config.build(
        weights,
        num_output_dimensions=5,
        generative_context=context,
    )

    assert isinstance(result, FakePipeline)
    assert captured["config"] is config
    assert captured["weights"] is weights
    assert captured["num_output_dimensions"] == 5
    assert captured["generative_context"] is context


class RecordingLoss(torch.nn.Module):
    def __init__(
        self,
        value=1.0,
        **kwargs,
    ):
        super().__init__()
        self.value = value
        self.kwargs = kwargs
        self.calls = []

    def forward(
        self,
        data,
        target,
        target_mask=None,
        **kwargs,
    ):
        self.calls.append(
            {
                "data": data,
                "target": target,
                "target_mask": target_mask,
                "kwargs": kwargs,
            }
        )

        return torch.as_tensor(
            self.value,
            dtype=data.dtype,
            device=data.device,
        )


class CapturingRegistry:
    def __init__(self):
        self.calls = []
        self.modules = []

    def get(
        self,
        name,
        args,
    ):
        module = RecordingLoss(**args)

        self.calls.append(
            {
                "name": name,
                "args": args,
            }
        )
        self.modules.append(module)

        return module


def test_loss_step_config_defaults_to_independent_empty_dicts():
    first = LossStepConfig(name="a")
    second = LossStepConfig(name="b")

    first.args["alpha"] = 1

    assert first.args == {
        "alpha": 1,
    }
    assert second.args == {}


@pytest.mark.parametrize(
    "reserved_arguments",
    [
        {
            "reduction": "sum",
            "weights": object(),
        },
        {
            "num_output_dimensions": 3,
            "generative_context": object(),
        },
    ],
)
def test_config_reports_multiple_reserved_arguments(
    reserved_arguments,
):
    expected_names = sorted(reserved_arguments)

    with pytest.raises(
        ValueError,
        match="Do not specify",
    ) as error:
        LosspipelineConfig(
            loss_pipeline=[
                LossStepConfig(
                    name="a",
                    args=reserved_arguments,
                )
            ]
        )

    for name in expected_names:
        assert name in str(error.value)


@pytest.mark.parametrize(
    "loss_weights",
    [
        [
            1.0 + 1e-9,
        ],
        [
            0.500000001,
            0.499999999,
        ],
    ],
)
def test_config_accepts_weights_within_tolerance(
    loss_weights,
):
    config = LosspipelineConfig(
        loss_pipeline=[
            LossStepConfig(name=f"loss_{index}") for index in range(len(loss_weights))
        ],
        loss_weights=loss_weights,
    )

    assert config.loss_weights is loss_weights


@pytest.mark.parametrize(
    "loss_weights",
    [
        [
            0.5,
            0.400001,
        ],
        [
            1.01,
        ],
        [
            -0.1,
            1.0,
        ],
    ],
)
def test_config_rejects_weight_sums_outside_tolerance(
    loss_weights,
):
    pipeline = [
        LossStepConfig(name=f"loss_{index}") for index in range(len(loss_weights))
    ]

    with pytest.raises(
        ValueError,
        match="Sum of loss term weights",
    ):
        LosspipelineConfig(
            loss_pipeline=pipeline,
            loss_weights=loss_weights,
        )


def test_no_saved_output_mask_resolves_to_none():
    config = LosspipelineConfig(
        loss_pipeline=[
            LossStepConfig(name="a"),
        ],
        saved_output_mask_dir=None,
    )

    assert config.output_mask is None


def test_saved_output_mask_missing_file_raises(
    tmp_path,
):
    mask_path = tmp_path / "missing_mask.pt"

    with pytest.raises(
        FileNotFoundError,
        match="Saved mask file does not exist",
    ):
        LosspipelineConfig(
            loss_pipeline=[
                LossStepConfig(name="a"),
            ],
            saved_output_mask_dir=str(mask_path),
        )


def test_saved_output_mask_rejects_directory(
    tmp_path,
):
    with pytest.raises(
        FileNotFoundError,
        match="Saved mask file does not exist",
    ):
        LosspipelineConfig(
            loss_pipeline=[
                LossStepConfig(name="a"),
            ],
            saved_output_mask_dir=str(tmp_path),
        )


def test_saved_output_mask_rejects_unknown_extension(
    tmp_path,
):
    mask_path = tmp_path / "mask.npy"
    mask_path.write_bytes(b"data")

    with pytest.raises(
        TypeError,
        match=r"NetCDF \(.nc\) or PyTorch \(.pt\)",
    ):
        LosspipelineConfig(
            loss_pipeline=[
                LossStepConfig(name="a"),
            ],
            saved_output_mask_dir=str(mask_path),
        )


def test_loads_tensor_mask_from_pt(
    tmp_path,
):
    mask_path = tmp_path / "mask.pt"

    source = torch.tensor(
        [
            [
                0,
                1,
            ],
            [
                1,
                0,
            ],
        ],
        dtype=torch.int64,
    )
    torch.save(
        source,
        mask_path,
    )

    config = LosspipelineConfig(
        loss_pipeline=[
            LossStepConfig(name="a"),
        ],
        saved_output_mask_dir=str(mask_path),
    )

    assert config.output_mask.dtype == (torch.float32)
    torch.testing.assert_close(
        config.output_mask,
        source.float(),
    )


def test_pt_mask_must_contain_tensor(
    tmp_path,
):
    mask_path = tmp_path / "mask.pt"
    torch.save(
        {
            "mask": [
                1,
                0,
            ]
        },
        mask_path,
    )

    with pytest.raises(
        TypeError,
        match="must contain a torch.Tensor",
    ):
        LosspipelineConfig(
            loss_pipeline=[
                LossStepConfig(name="a"),
            ],
            saved_output_mask_dir=str(mask_path),
        )


def test_loads_mask_from_netcdf(
    tmp_path,
):
    mask_path = tmp_path / "mask.nc"

    source = xr.DataArray(
        np.asarray(
            [
                [
                    1.0,
                    0.0,
                    1.0,
                ],
                [
                    0.0,
                    1.0,
                    0.0,
                ],
            ],
            dtype=np.float64,
        ),
        dims=(
            "lat",
            "lon",
        ),
        name="mask",
    )
    source.to_dataset().to_netcdf(mask_path)

    config = LosspipelineConfig(
        loss_pipeline=[
            LossStepConfig(name="a"),
        ],
        saved_output_mask_dir=str(mask_path),
    )

    assert config.output_mask.dtype == (torch.float32)
    assert tuple(config.output_mask.shape) == (
        1,
        2,
        3,
    )

    torch.testing.assert_close(
        config.output_mask,
        torch.as_tensor(
            source.values,
            dtype=torch.float32,
        ).unsqueeze(0),
    )


def test_build_with_saved_mask_requires_output_shape():
    config = LosspipelineConfig(
        loss_pipeline=[
            LossStepConfig(name="a"),
        ]
    )
    config.output_mask = torch.ones(
        2,
        3,
    )

    with pytest.raises(
        RuntimeError,
        match="output_shape must be provided",
    ):
        config.build(
            make_weights(),
            output_shape=None,
        )


def test_resolve_output_mask_adds_channel_dimension():
    config = LosspipelineConfig(
        loss_pipeline=[
            LossStepConfig(name="a"),
        ]
    )
    config.output_mask = torch.ones(
        2,
        3,
    )

    config._resove_output_mask(
        (
            4,
            2,
            3,
        )
    )

    assert tuple(config.output_mask.shape) == (
        1,
        2,
        3,
    )


def test_resolve_output_mask_preserves_matching_channel_mask():
    config = LosspipelineConfig(
        loss_pipeline=[
            LossStepConfig(name="a"),
        ]
    )

    original = torch.ones(
        4,
        2,
        3,
    )
    config.output_mask = original

    config._resove_output_mask(
        (
            4,
            2,
            3,
        )
    )

    assert config.output_mask is original


def test_resolve_output_mask_rejects_dimension_count():
    config = LosspipelineConfig(
        loss_pipeline=[
            LossStepConfig(name="a"),
        ]
    )
    config.output_mask = torch.ones(
        1,
        1,
        2,
        3,
    )

    with pytest.raises(
        RuntimeError,
        match="same number of dimensions",
    ):
        config._resove_output_mask(
            (
                2,
                2,
                3,
            )
        )


def test_resolve_output_mask_rejects_spatial_mismatch():
    config = LosspipelineConfig(
        loss_pipeline=[
            LossStepConfig(name="a"),
        ]
    )
    config.output_mask = torch.ones(
        1,
        4,
        5,
    )

    with pytest.raises(
        RuntimeError,
        match="non-channel dimensions",
    ):
        config._resove_output_mask(
            (
                2,
                2,
                3,
            )
        )


@pytest.mark.parametrize(
    "mask_channels",
    [
        0,
        2,
        4,
    ],
)
def test_resolve_output_mask_rejects_invalid_channel_count(
    mask_channels,
):
    config = LosspipelineConfig(
        loss_pipeline=[
            LossStepConfig(name="a"),
        ]
    )
    config.output_mask = torch.ones(
        mask_channels,
        2,
        3,
    )

    with pytest.raises(
        RuntimeError,
        match="channel dimension",
    ):
        config._resove_output_mask(
            (
                3,
                2,
                3,
            )
        )


@pytest.mark.parametrize(
    "mask_channels",
    [
        1,
        3,
    ],
)
def test_resolve_output_mask_accepts_valid_channel_counts(
    mask_channels,
):
    config = LosspipelineConfig(
        loss_pipeline=[
            LossStepConfig(name="a"),
        ]
    )
    config.output_mask = torch.ones(
        mask_channels,
        2,
        3,
    )

    config._resove_output_mask(
        (
            3,
            2,
            3,
        )
    )

    assert config.output_mask.shape[0] == (mask_channels)


def test_build_resolves_mask_before_constructing_pipeline(
    monkeypatch,
):
    config = LosspipelineConfig(
        loss_pipeline=[
            LossStepConfig(name="a"),
        ]
    )
    config.output_mask = torch.ones(
        2,
        3,
    )

    constructor = Mock(return_value=object())
    monkeypatch.setattr(
        "cccma_ppp.loss.loss.Losspipeline",
        constructor,
    )

    config.build(
        make_weights(),
        output_shape=(
            1,
            2,
            3,
        ),
    )

    assert tuple(config.output_mask.shape) == (
        1,
        2,
        3,
    )

    constructor.assert_called_once()


def test_pipeline_uses_default_generative_context():
    config = LosspipelineConfig(
        loss_pipeline=[
            LossStepConfig(name="a"),
        ]
    )

    pipeline = config.build(
        make_weights(),
        num_output_dimensions=3,
    )

    assert isinstance(
        pipeline.generative_context,
        GenerativeContext,
    )


def test_pipeline_preserves_explicit_generative_context():
    context = GenerativeContext()
    config = LosspipelineConfig(
        loss_pipeline=[
            LossStepConfig(name="a"),
        ]
    )

    pipeline = config.build(
        make_weights(),
        generative_context=context,
    )

    assert pipeline.generative_context is context


def test_pipeline_forwards_shared_arguments_to_registry(
    monkeypatch,
):
    registry = CapturingRegistry()
    monkeypatch.setattr(
        Losspipeline,
        "registery",
        registry,
    )

    weights = make_weights()
    context = GenerativeContext()

    config = LosspipelineConfig(
        loss_pipeline=[
            LossStepConfig(
                name="CustomLoss",
                args={
                    "alpha": 0.25,
                },
            )
        ],
        reduction="sum",
    )

    Losspipeline(
        config,
        weights,
        num_output_dimensions=4,
        generative_context=context,
    )

    assert len(registry.calls) == 1

    call = registry.calls[0]

    assert call["name"] == "customloss"
    assert call["args"]["alpha"] == (pytest.approx(0.25))
    assert call["args"]["weights"] is (weights)
    assert call["args"]["reduction"] == ("sum")
    assert call["args"]["num_output_dimensions"] == 4
    assert call["args"]["generative_context"] is context


def test_pipeline_does_not_mutate_step_arguments(
    monkeypatch,
):
    registry = CapturingRegistry()
    monkeypatch.setattr(
        Losspipeline,
        "registery",
        registry,
    )

    arguments = {
        "alpha": 0.25,
    }
    step = LossStepConfig(
        name="a",
        args=arguments,
    )

    LosspipelineConfig(
        loss_pipeline=[
            step,
        ]
    ).build(make_weights())

    assert step.args == {
        "alpha": 0.25,
    }
    assert arguments == {
        "alpha": 0.25,
    }


def test_pipeline_is_module_list():
    config = LosspipelineConfig(
        loss_pipeline=[
            LossStepConfig(name="a"),
            LossStepConfig(name="b"),
        ]
    )

    pipeline = config.build(make_weights())

    assert isinstance(
        pipeline.pipeline,
        torch.nn.ModuleList,
    )
    assert len(list(pipeline.parameters())) == 0


def test_register_lowercases_name(
    monkeypatch,
):
    registered = {}

    class FakeRegistry:
        def register(
            self,
            name,
        ):
            registered["name"] = name

            return lambda value: value

    monkeypatch.setattr(
        Losspipeline,
        "registery",
        FakeRegistry(),
    )

    decorator = Losspipeline.register("MyCustomLoss")

    assert callable(decorator)
    assert registered["name"] == ("mycustomloss")


def test_duplicate_step_names_are_numbered():
    config = LosspipelineConfig(
        loss_pipeline=[
            LossStepConfig(name="a"),
            LossStepConfig(name="a"),
        ]
    )

    pipeline = config.build(make_weights())

    assert pipeline.steps == [
        "a",
        "a_2",
    ]


def test_three_duplicate_step_names_are_unique():
    config = LosspipelineConfig(
        loss_pipeline=[
            LossStepConfig(name="a"),
            LossStepConfig(name="a"),
            LossStepConfig(name="a"),
        ]
    )

    pipeline = config.build(make_weights())

    assert len(pipeline.steps) == 3
    assert len(set(pipeline.steps)) == 2


def test_low_resolution_name_uses_kernel_size():
    config = LosspipelineConfig(
        loss_pipeline=[
            LossStepConfig(
                name="a",
                args={
                    "low_ress_kernel_size": 7,
                },
            )
        ]
    )

    pipeline = config.build(make_weights())

    assert pipeline.steps == [
        "a_low_ress_7",
    ]


def test_duplicate_low_resolution_names_are_numbered():
    config = LosspipelineConfig(
        loss_pipeline=[
            LossStepConfig(
                name="a",
                args={
                    "low_ress_kernel_size": 3,
                },
            ),
            LossStepConfig(
                name="a",
                args={
                    "low_ress_kernel_size": 3,
                },
            ),
        ]
    )

    pipeline = config.build(make_weights())

    assert pipeline.steps == [
        "a_low_ress_3",
        "a_low_ress_3_2",
    ]


def test_masked_pipeline_registers_output_mask_as_buffer():
    config = LosspipelineConfig(
        loss_pipeline=[
            LossStepConfig(name="a"),
        ],
        masked_loss_calculation=True,
    )
    config.output_mask = torch.ones(
        1,
        2,
        3,
    )

    pipeline = Losspipeline(
        config,
        make_weights(),
        num_output_dimensions=3,
    )

    buffers = dict(pipeline.named_buffers())

    assert "output_mask" in buffers
    assert buffers["output_mask"] is pipeline.output_mask


def test_unmasked_pipeline_does_not_use_configured_output_mask():
    config = LosspipelineConfig(
        loss_pipeline=[
            LossStepConfig(name="a"),
        ],
        masked_loss_calculation=False,
    )
    config.output_mask = torch.ones(
        1,
        2,
        3,
    )

    pipeline = Losspipeline(
        config,
        make_weights(),
        num_output_dimensions=3,
    )

    assert pipeline.output_mask is None


def test_output_mask_is_used_when_target_mask_is_none(
    monkeypatch,
):
    registry = CapturingRegistry()
    monkeypatch.setattr(
        Losspipeline,
        "registery",
        registry,
    )

    config = LosspipelineConfig(
        loss_pipeline=[
            LossStepConfig(name="a"),
        ],
        masked_loss_calculation=True,
    )
    config.output_mask = torch.tensor(
        [
            [
                [
                    1.0,
                    0.0,
                    1.0,
                ],
                [
                    0.0,
                    1.0,
                    0.0,
                ],
            ]
        ]
    )

    pipeline = Losspipeline(
        config,
        make_weights(),
        num_output_dimensions=3,
    )

    pipeline(
        make_data(),
        make_target(),
        target_mask=None,
    )

    received = registry.modules[0].calls[0]["target_mask"]

    assert received is pipeline.output_mask


def test_output_mask_is_multiplied_with_target_mask(
    monkeypatch,
):
    registry = CapturingRegistry()
    monkeypatch.setattr(
        Losspipeline,
        "registery",
        registry,
    )

    config = LosspipelineConfig(
        loss_pipeline=[
            LossStepConfig(name="a"),
        ],
        masked_loss_calculation=True,
    )
    config.output_mask = torch.tensor(
        [
            [
                [
                    1.0,
                    0.0,
                    1.0,
                ],
                [
                    0.0,
                    1.0,
                    0.0,
                ],
            ]
        ]
    )

    pipeline = Losspipeline(
        config,
        make_weights(),
        num_output_dimensions=3,
    )

    target_mask = torch.full(
        (
            2,
            1,
            2,
            3,
        ),
        0.5,
    )

    pipeline(
        make_data(),
        make_target(),
        target_mask=target_mask,
    )

    received = registry.modules[0].calls[0]["target_mask"]

    expected = target_mask * pipeline.output_mask

    torch.testing.assert_close(
        received,
        expected,
    )


def test_masked_loss_disabled_clears_target_mask(
    monkeypatch,
):
    registry = CapturingRegistry()
    monkeypatch.setattr(
        Losspipeline,
        "registery",
        registry,
    )

    config = LosspipelineConfig(
        loss_pipeline=[
            LossStepConfig(name="a"),
        ],
        masked_loss_calculation=False,
    )
    config.output_mask = torch.ones(
        1,
        2,
        3,
    )

    pipeline = Losspipeline(
        config,
        make_weights(),
        num_output_dimensions=3,
    )

    pipeline(
        make_data(),
        make_target(),
        target_mask=torch.ones_like(make_target()),
    )

    assert registry.modules[0].calls[0]["target_mask"] is None


def test_forward_passes_print_loss_to_every_step(
    monkeypatch,
):
    registry = CapturingRegistry()
    monkeypatch.setattr(
        Losspipeline,
        "registery",
        registry,
    )

    config = LosspipelineConfig(
        loss_pipeline=[
            LossStepConfig(name="a"),
            LossStepConfig(name="b"),
        ]
    )

    pipeline = Losspipeline(
        config,
        make_weights(),
        num_output_dimensions=3,
    )

    pipeline(
        make_data(),
        make_target(),
        print_loss=True,
    )

    assert all(
        module.calls[0]["kwargs"]["print_loss"] is True for module in registry.modules
    )


def test_forward_does_not_add_print_loss_when_disabled(
    monkeypatch,
):
    registry = CapturingRegistry()
    monkeypatch.setattr(
        Losspipeline,
        "registery",
        registry,
    )

    config = LosspipelineConfig(
        loss_pipeline=[
            LossStepConfig(name="a"),
        ]
    )

    pipeline = Losspipeline(
        config,
        make_weights(),
        num_output_dimensions=3,
    )

    pipeline(
        make_data(),
        make_target(),
        print_loss=False,
    )

    assert "print_loss" not in registry.modules[0].calls[0]["kwargs"]


def test_forward_does_not_mutate_step_arguments(
    monkeypatch,
):
    registry = CapturingRegistry()
    monkeypatch.setattr(
        Losspipeline,
        "registery",
        registry,
    )

    pipeline = LosspipelineConfig(
        loss_pipeline=[
            LossStepConfig(name="a"),
        ]
    ).build(
        make_weights(),
        num_output_dimensions=3,
    )

    step_arguments = {
        "flip": True,
    }

    pipeline(
        make_data(),
        make_target(),
        print_loss=True,
        step_arguments=step_arguments,
    )

    assert step_arguments == {
        "flip": True,
    }

    assert registry.modules[0].calls[0]["kwargs"] == {
        "flip": True,
        "print_loss": True,
    }


def test_forward_individual_losses_are_detached(
    monkeypatch,
):
    parameter = torch.nn.Parameter(torch.tensor(2.0))

    class GradientLoss(torch.nn.Module):
        def forward(
            self,
            data,
            target,
            target_mask=None,
            **kwargs,
        ):
            return data.mean() * parameter

    class GradientRegistry:
        def get(
            self,
            name,
            args,
        ):
            return GradientLoss()

    monkeypatch.setattr(
        Losspipeline,
        "registery",
        GradientRegistry(),
    )

    pipeline = LosspipelineConfig(
        loss_pipeline=[
            LossStepConfig(name="a"),
        ]
    ).build(
        make_weights(),
        num_output_dimensions=3,
    )

    data = make_data().requires_grad_()

    total_loss, individual = pipeline(
        data,
        make_target(),
    )

    assert total_loss.requires_grad is True
    assert individual["a"].requires_grad is (False)

    total_loss.backward()

    assert data.grad is not None
    assert parameter.grad is not None


def test_weighted_total_loss_uses_configured_weights(
    monkeypatch,
):
    registry = CapturingRegistry()
    monkeypatch.setattr(
        Losspipeline,
        "registery",
        registry,
    )

    config = LosspipelineConfig(
        loss_pipeline=[
            LossStepConfig(
                name="first",
                args={
                    "value": 2.0,
                },
            ),
            LossStepConfig(
                name="second",
                args={
                    "value": 6.0,
                },
            ),
        ],
        loss_weights=[
            0.25,
            0.75,
        ],
    )

    pipeline = Losspipeline(
        config,
        make_weights(),
        num_output_dimensions=3,
    )

    total_loss, individual = pipeline(
        make_data(),
        make_target(),
    )

    assert total_loss.item() == (pytest.approx(5.0))
    assert individual["first"].item() == (pytest.approx(2.0))
    assert individual["second"].item() == (pytest.approx(6.0))


def test_dimension_check_runs_only_once(
    monkeypatch,
):
    registry = CapturingRegistry()
    monkeypatch.setattr(
        Losspipeline,
        "registery",
        registry,
    )

    pipeline = LosspipelineConfig(
        loss_pipeline=[
            LossStepConfig(name="a"),
        ]
    ).build(
        make_weights(),
        num_output_dimensions=3,
    )

    pipeline(
        make_data(),
        make_target(),
    )

    assert pipeline._checked_dimensionality is (True)

    incorrect_target = torch.zeros(
        2,
        1,
        6,
    )

    pipeline(
        make_data(),
        incorrect_target,
    )


def test_generative_context_requires_extra_target_dimension():
    context = GenerativeContext()
    context.generative_modeling = True

    pipeline = LosspipelineConfig(
        loss_pipeline=[
            LossStepConfig(name="a"),
        ]
    ).build(
        make_weights(),
        num_output_dimensions=3,
        generative_context=context,
    )

    generative_target = torch.zeros(
        4,
        2,
        1,
        2,
        3,
    )
    generative_data = torch.ones_like(generative_target)

    loss, individual = pipeline(
        generative_data,
        generative_target,
    )

    assert torch.is_tensor(loss)
    assert "a" in individual


def test_generative_context_rejects_nongenerative_target_shape():
    context = GenerativeContext()
    context.generative_modeling = True

    pipeline = LosspipelineConfig(
        loss_pipeline=[
            LossStepConfig(name="a"),
        ]
    ).build(
        make_weights(),
        num_output_dimensions=3,
        generative_context=context,
    )

    with pytest.raises(
        AssertionError,
        match="Expected target to have 5 dims",
    ):
        pipeline(
            make_data(),
            make_target(),
        )


def test_non_generative_context_rejects_extra_sample_dimension():
    context = GenerativeContext()
    context.generative_modeling = False

    pipeline = LosspipelineConfig(
        loss_pipeline=[
            LossStepConfig(name="a"),
        ]
    ).build(
        make_weights(),
        num_output_dimensions=3,
        generative_context=context,
    )

    target = torch.zeros(
        3,
        2,
        1,
        2,
        3,
    )

    with pytest.raises(
        AssertionError,
        match="Expected target to have 4 dims",
    ):
        pipeline(
            torch.ones_like(target),
            target,
        )
