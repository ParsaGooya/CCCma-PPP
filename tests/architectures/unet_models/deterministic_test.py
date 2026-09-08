from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

import cccma_ppp.architectures.unet.deterministic as module
from cccma_ppp.architectures.layers.conv import (
    ConvBlockConfig,
    ConvNeXtBlockConfig,
    PartialConvBlockConfig,
    TensorMask,
)
from cccma_ppp.architectures.layers.unet import (
    UNetOutputSIC,
)
from cccma_ppp.architectures.unet.deterministic import (
    UNet,
    UNetConfig,
    UNetSIC,
    UNetSICConfig,
)


def make_block_config(**kwargs):
    defaults = {
        "name": "standard_conv",
        "num_convolutions": 1,
        "kernel_size": 3,
        "normalization": "none",
        "padding_method": "zeros",
        "activation": "relu",
        "dropout_rate": None,
        "bias": False,
        "group_norm_groups": 1,
    }
    defaults.update(kwargs)
    defaults.pop("bottleneck_dim", None)
    defaults.pop("process_skip", None)
    return ConvBlockConfig(**defaults)


def make_partial_block_config(**kwargs):
    defaults = {
        "name": "partial_conv",
        "num_convolutions": 1,
        "kernel_size": 3,
        "normalization": "none",
        "padding_method": "zeros",
        "activation": "relu",
        "dropout_rate": None,
        "bias": False,
        "group_norm_groups": 1,
    }
    defaults.update(kwargs)
    defaults.pop("bottleneck_dim", None)
    defaults.pop("process_skip", None)
    return PartialConvBlockConfig(**defaults)


def make_convnext_block_config(**kwargs):
    defaults = {
        "name": "convnext",
        "num_blocks": 1,
        "kernel_size": 3,
        "expansion_ratio": 2,
        "padding_method": "zeros",
        "layer_scale_init": 1e-6,
        "dropout_rate": 0.0,
        "drop_path_rate": 0.0,
        "use_partial_conv": False,
    }
    defaults.update(kwargs)
    defaults.pop("bottleneck_dim", None)
    defaults.pop("process_skip", None)
    return ConvNeXtBlockConfig(**defaults)


def make_generator(
    *,
    noise_level="full",
    num_training_noise_samples=2,
):
    return SimpleNamespace(
        noise_level=noise_level,
        num_training_noise_samples=num_training_noise_samples,
    )


def make_config(**kwargs):
    defaults = {
        "channels": [4, 8, 16],
        "block_config": make_block_config(),
        "upsampling_method": "bilinear",
        "skip_alignment_method": "padd",
        "transpose_kernel_sizes": 3,
        "mask_pooling": "any",
        "mask_fraction_threshold": 0.5,
        "output_activation": "identity",
        "output_block_hidden_channels": None,
        "init_method": "trunc_normal",
        "GENERATOR": None,
    }
    defaults.update(kwargs)
    defaults.pop("bottleneck_dim", None)
    defaults.pop("process_skip", None)

    with patch.object(
        module,
        "_unet_config_checks",
        return_value=None,
    ):
        return UNetConfig(**defaults)


def make_model(
    *,
    config=None,
    input_shape=(2, 16, 16),
    output_shape=(1, 16, 16),
    added_features_dim=None,
):
    if config is None:
        config = make_config()

    return UNet(
        config=config,
        input_shape=input_shape,
        output_shape=output_shape,
        added_features_dim=added_features_dim,
    )


def make_request(
    *,
    input=None,
    input_mask=None,
    added_features=None,
    output_sample_size=None,
):
    if input is None:
        input = torch.randn(2, 2, 16, 16)

    return SimpleNamespace(
        input=input,
        input_mask=input_mask,
        added_features=added_features,
        output_sample_size=output_sample_size,
        chunk_output_samples=None,
    )


def test_config_defaults():
    config = make_config()

    assert config.channels == [4, 8, 16]
    assert config.upsampling_method == "bilinear"
    assert config.skip_alignment_method == "padd"
    assert config.transpose_kernel_sizes == [3, 3]
    assert config.mask_pooling == "any"
    assert config.mask_fraction_threshold == pytest.approx(0.5)
    assert config.output_activation == "identity"
    assert config.output_block_hidden_channels is None
    assert config.init_method == "trunc_normal"
    assert config.GENERATOR is None


def test_config_calls_shared_validation():
    with patch.object(
        module,
        "_unet_config_checks",
        return_value=None,
    ) as checks:
        config = UNetConfig(
            channels=[4, 8, 16],
            block_config=make_block_config(),
        )

    checks.assert_called_once_with(config)


def test_config_propagates_shared_validation_error():
    with (
        patch.object(
            module,
            "_unet_config_checks",
            side_effect=ValueError("invalid UNet configuration"),
        ),
        pytest.raises(
            ValueError,
            match="invalid UNet configuration",
        ),
    ):
        UNetConfig(
            channels=[4, 8, 16],
            block_config=make_block_config(),
        )


def test_shared_validation_runs_before_kernel_normalization():
    captured = {}

    def fake_checks(config):
        captured["kernel_sizes"] = config.transpose_kernel_sizes

    with patch.object(
        module,
        "_unet_config_checks",
        side_effect=fake_checks,
    ):
        config = UNetConfig(
            channels=[4, 8, 16],
            block_config=make_block_config(),
            transpose_kernel_sizes=5,
        )

    assert captured["kernel_sizes"] == 5
    assert config.transpose_kernel_sizes == [5, 5]


@pytest.mark.parametrize(
    ("channels", "expected"),
    [
        ([4], None),
        ([4, 8], None),
        ([4, 8, 16], None),
        ([4, 8, 16, 32], None),
    ],
)
def test_none_kernel_sizes_create_one_per_up_block(
    channels,
    expected,
):
    config = make_config(
        channels=channels,
        transpose_kernel_sizes=None,
    )

    assert config.transpose_kernel_sizes == expected


@pytest.mark.parametrize(
    "kernel_size",
    [
        1,
        2,
        3,
        5,
    ],
)
def test_integer_kernel_size_is_repeated(kernel_size):
    config = make_config(
        channels=[4, 8, 16, 32],
        transpose_kernel_sizes=kernel_size,
    )

    assert config.transpose_kernel_sizes == [
        kernel_size,
        kernel_size,
        kernel_size,
    ]


def test_explicit_kernel_list_is_preserved():
    kernel_sizes = [2, 3]

    config = make_config(
        channels=[4, 8, 16],
        transpose_kernel_sizes=kernel_sizes,
    )

    assert config.transpose_kernel_sizes is kernel_sizes


@pytest.mark.parametrize(
    "method",
    [
        "bilinear",
        "transpose_conv",
    ],
)
def test_config_preserves_upsampling_method(method):
    config = make_config(
        upsampling_method=method,
    )

    assert config.upsampling_method == method


@pytest.mark.parametrize(
    "method",
    [
        "resize",
        "padd",
        "strict",
    ],
)
def test_config_preserves_alignment_method(method):
    config = make_config(
        skip_alignment_method=method,
    )

    assert config.skip_alignment_method == method


@pytest.mark.parametrize(
    "method",
    [
        "any",
        "all",
        "fraction",
    ],
)
def test_config_preserves_mask_pooling(method):
    config = make_config(
        mask_pooling=method,
    )

    assert config.mask_pooling == method


@pytest.mark.parametrize(
    "activation",
    [
        "identity",
        "sigmoid",
        "tanh",
    ],
)
def test_config_preserves_output_activation(activation):
    config = make_config(
        output_activation=activation,
    )

    assert config.output_activation == activation


@pytest.mark.parametrize(
    "hidden_channels",
    [
        None,
        1,
        4,
        16,
    ],
)
def test_config_preserves_output_hidden_channels(hidden_channels):
    config = make_config(
        output_block_hidden_channels=hidden_channels,
    )

    assert config.output_block_hidden_channels == hidden_channels


def test_config_accepts_partial_conv_block():
    block_config = make_partial_block_config()

    config = make_config(
        block_config=block_config,
    )

    assert config.block_config is block_config


def test_config_accepts_convnext_block():
    block_config = make_convnext_block_config()

    config = make_config(
        block_config=block_config,
    )

    assert config.block_config is block_config


def test_config_build_constructs_unet():
    config = make_config()
    expected = object()
    input_shape = np.asarray([2, 16, 16])
    output_shape = np.asarray([1, 16, 16])

    with patch.object(
        module,
        "UNet",
        return_value=expected,
    ) as constructor:
        result = config.build(
            input_shape=input_shape,
            output_shape=output_shape,
            added_features_dim=3,
        )

    assert result is expected

    constructor.assert_called_once_with(
        config=config,
        input_shape=input_shape,
        output_shape=output_shape,
        added_features_dim=3,
    )


def test_config_build_passes_none_values():
    config = make_config()
    expected = object()

    with patch.object(
        module,
        "UNet",
        return_value=expected,
    ) as constructor:
        result = config.build(
            input_shape=(2, 16, 16),
            output_shape=None,
            added_features_dim=None,
        )

    assert result is expected

    constructor.assert_called_once_with(
        config=config,
        input_shape=(2, 16, 16),
        output_shape=None,
        added_features_dim=None,
    )


def test_model_defaults_output_shape_to_input_shape():
    model = make_model(
        input_shape=(2, 16, 16),
        output_shape=None,
    )

    assert model.output_shape == (2, 16, 16)


@pytest.mark.parametrize(
    "input_shape",
    [
        (16, 16),
        (1, 2, 16, 16),
    ],
)
def test_model_rejects_invalid_input_rank(input_shape):
    with pytest.raises(
        RuntimeError,
        match="UNet expects 3D input shapes",
    ):
        make_model(
            input_shape=input_shape,
        )


@pytest.mark.parametrize(
    "output_shape",
    [
        (16, 16),
        (1, 2, 16, 16),
    ],
)
def test_model_rejects_invalid_output_rank(output_shape):
    with pytest.raises(
        RuntimeError,
        match="UNet expects 3D output shapes",
    ):
        make_model(
            output_shape=output_shape,
        )


def test_model_rejects_different_spatial_shapes_without_hidden_output():
    config = make_config(
        output_block_hidden_channels=None,
    )

    with pytest.raises(
        RuntimeError,
        match="does not match output spatial shape",
    ):
        make_model(
            config=config,
            input_shape=(2, 16, 16),
            output_shape=(1, 8, 8),
        )


def test_model_accepts_different_spatial_shapes_with_hidden_output():
    config = make_config(
        output_block_hidden_channels=4,
    )

    model = make_model(
        config=config,
        input_shape=(2, 16, 16),
        output_shape=(1, 8, 8),
    )

    assert model.output_shape == (1, 8, 8)


@pytest.mark.parametrize(
    "input_shape",
    [
        (2, 0, 16),
        (2, 16, 0),
        (2, -1, 16),
    ],
)
def test_model_rejects_nonpositive_spatial_dimensions(input_shape):
    with pytest.raises(
        ValueError,
        match="spatial dimensions must be positive",
    ):
        make_model(
            input_shape=input_shape,
            output_shape=input_shape,
        )


def test_model_rejects_depth_too_large_for_input():
    config = make_config(
        channels=[4, 8, 16, 32, 64],
        transpose_kernel_sizes=3,
    )

    with pytest.raises(
        ValueError,
        match="depth is too large",
    ):
        make_model(
            config=config,
            input_shape=(2, 8, 8),
            output_shape=(1, 8, 8),
        )


def test_depth_error_reports_allowed_levels():
    config = make_config(
        channels=[4, 8, 16, 32, 64],
        transpose_kernel_sizes=3,
    )

    with pytest.raises(
        ValueError,
        match="no more than 4 channel levels",
    ):
        make_model(
            config=config,
            input_shape=(2, 8, 8),
            output_shape=(1, 8, 8),
        )


def test_model_accepts_maximum_valid_depth():
    config = make_config(
        channels=[4, 8, 16, 32],
        transpose_kernel_sizes=3,
    )

    model = make_model(
        config=config,
        input_shape=(2, 8, 8),
        output_shape=(1, 8, 8),
    )

    assert len(model.down_blocks) == 3


def test_model_stores_configuration():
    config = make_config()

    model = make_model(
        config=config,
        added_features_dim=2,
    )

    assert model.config is config
    assert model.init_method == config.init_method
    assert model.added_features_dim == 2


def test_none_added_features_dimension_becomes_zero():
    model = make_model(
        added_features_dim=None,
    )

    assert model.added_features_dim == 0


def test_zero_added_features_dimension_remains_zero():
    model = make_model(
        added_features_dim=0,
    )

    assert model.added_features_dim == 0


def test_initial_mapping_includes_added_feature_channels():
    model = make_model(
        input_shape=(2, 16, 16),
        added_features_dim=3,
    )

    assert model.initial_mapping.stages[0].conv.in_channels == 5
    assert model.initial_mapping.stages[0].conv.out_channels == 4


def test_model_builds_one_down_block_per_channel_transition():
    config = make_config(
        channels=[4, 8, 16, 32],
        transpose_kernel_sizes=3,
    )

    model = make_model(config=config)

    assert len(model.down_blocks) == 3


def test_model_builds_one_up_block_per_channel_transition():
    config = make_config(
        channels=[4, 8, 16, 32],
        transpose_kernel_sizes=3,
    )

    model = make_model(config=config)

    assert len(model.up_blocks) == 3


def test_bottleneck_uses_last_channel_width():
    config = make_config(
        channels=[4, 8, 16],
        bottleneck_dim=None,
    )

    model = make_model(config=config)

    assert model.bottleneck.out_channels == 16


def test_output_uses_requested_channel_count():
    model = make_model(
        output_shape=(3, 16, 16),
    )

    final_conv = [
        layer for layer in model.output.modules() if isinstance(layer, nn.Conv2d)
    ][-1]

    assert final_conv.out_channels == 3


def test_down_blocks_without_skip_processing():
    config = make_config(
        process_skip=False,
    )

    model = make_model(config=config)

    assert all(block.skip_processor is None for block in model.down_blocks)


def test_up_blocks_use_kernel_sizes_in_decoder_order():
    config = make_config(
        channels=[4, 8, 16],
        upsampling_method="transpose_conv",
        transpose_kernel_sizes=[2, 3],
    )

    model = make_model(config=config)

    assert model.up_blocks[0].upsample.kernel_size == (2, 2)
    assert model.up_blocks[1].upsample.kernel_size == (3, 3)


def test_no_generator_disables_noise_in_all_up_blocks():
    model = make_model(
        config=make_config(
            GENERATOR=None,
        )
    )

    assert all(not block.inject_noise for block in model.up_blocks)


def test_full_noise_enables_noise_in_all_up_blocks():
    model = make_model(
        config=make_config(
            GENERATOR=make_generator(
                noise_level="full",
            ),
        )
    )

    assert all(block.inject_noise for block in model.up_blocks)
    assert all(block.inject_noise_in_block for block in model.up_blocks)


def test_low_noise_enables_upsampling_noise_but_not_block_noise():
    model = make_model(
        config=make_config(
            GENERATOR=make_generator(
                noise_level="low",
            ),
        )
    )

    assert all(block.inject_noise for block in model.up_blocks)
    assert all(block.inject_noise_in_block is False for block in model.up_blocks)


def test_prepare_input_without_mask_or_features():
    model = make_model()
    tensor = torch.randn(2, 2, 16, 16)

    result = model._prepare_input(
        x=tensor,
        x_mask=None,
        added_features=None,
    )

    assert isinstance(result, TensorMask)
    assert result.tensor is tensor
    assert result.mask is None


def test_prepare_input_broadcasts_mask(
    monkeypatch,
):
    model = make_model()
    tensor = torch.randn(2, 2, 16, 16)
    mask = torch.ones(16, 16)
    expected = torch.ones(2, 1, 16, 16)

    with patch.object(
        module,
        "_broadcast_mask",
        return_value=expected,
    ) as broadcast:
        result = model._prepare_input(
            x=tensor,
            x_mask=mask,
            added_features=None,
        )

    broadcast.assert_called_once_with(mask, tensor)
    assert result.mask is expected


def test_prepare_input_concatenates_added_features():
    model = make_model(
        added_features_dim=3,
    )

    tensor = torch.zeros(2, 2, 16, 16)
    features = torch.ones(2, 3, 16, 16)

    result = model._prepare_input(
        x=tensor,
        x_mask=None,
        added_features=features,
    )

    assert result.tensor.shape == (2, 5, 16, 16)
    torch.testing.assert_close(
        result.tensor[:, :2],
        tensor,
    )
    torch.testing.assert_close(
        result.tensor[:, 2:],
        features,
    )
    assert result.mask is None


def test_prepare_input_adds_valid_feature_mask():
    model = make_model(
        added_features_dim=3,
    )

    tensor = torch.zeros(2, 2, 16, 16)
    mask = torch.zeros(2, 2, 16, 16)
    features = torch.ones(2, 3, 16, 16)

    result = model._prepare_input(
        x=tensor,
        x_mask=mask,
        added_features=features,
    )

    assert result.mask.shape == (2, 5, 16, 16)

    torch.testing.assert_close(
        result.mask[:, :2],
        mask,
    )
    torch.testing.assert_close(
        result.mask[:, 2:],
        torch.ones_like(features),
    )


def test_prepare_input_does_not_create_feature_mask_without_input_mask():
    model = make_model(
        added_features_dim=3,
    )

    result = model._prepare_input(
        x=torch.zeros(2, 2, 16, 16),
        x_mask=None,
        added_features=torch.ones(2, 3, 16, 16),
    )

    assert result.mask is None


def test_forward_without_generator():
    model = make_model()
    model.eval()

    result = model(
        make_request(
            output_sample_size=None,
        )
    )

    assert result.output.shape == (2, 1, 16, 16)


def test_forward_without_generator_ignores_output_sample_size():
    model = make_model()
    model.eval()

    result = model(
        make_request(
            output_sample_size=5,
        )
    )

    assert result.output.shape == (2, 1, 16, 16)


def test_forward_with_added_features():
    model = make_model(
        added_features_dim=2,
    )
    model.eval()

    result = model(
        make_request(
            added_features=torch.randn(2, 2, 16, 16),
        )
    )

    assert result.output.shape == (2, 1, 16, 16)


def test_forward_resizes_to_output_shape(
    monkeypatch,
):
    config = make_config(
        output_block_hidden_channels=4,
    )
    model = make_model(
        config=config,
        input_shape=(2, 16, 16),
        output_shape=(1, 12, 10),
    )
    model.eval()

    expected = torch.randn(2, model.up_blocks[-1].out_channels, 12, 10)

    with patch.object(
        module,
        "_resize_tensor",
        return_value=expected,
    ) as resize:
        result = model(make_request())

    resize.assert_called_once()
    assert resize.call_args.args[1] == (12, 10)
    assert result.output.shape == (2, 1, 12, 10)


def test_forward_output_is_finite():
    model = make_model()
    model.eval()

    result = model(make_request())

    assert torch.isfinite(result.output).all()


def test_generator_evaluation_repeats_requested_samples():
    config = make_config(
        GENERATOR=make_generator(
            noise_level="full",
            num_training_noise_samples=3,
        ),
    )
    model = make_model(config=config)
    model.eval()

    result = model(
        make_request(
            output_sample_size=4,
        )
    )

    assert result.output.shape == (
        4,
        2,
        1,
        16,
        16,
    )


def test_generator_training_uses_configured_sample_count():
    config = make_config(
        GENERATOR=make_generator(
            noise_level="full",
            num_training_noise_samples=3,
        ),
    )
    model = make_model(config=config)
    model.train()

    result = model(
        make_request(
            output_sample_size=3,
        )
    )

    assert result.output.shape == (
        3,
        2,
        1,
        16,
        16,
    )


def test_generator_training_uses_configured_samples_when_request_is_none():
    config = make_config(
        GENERATOR=make_generator(
            noise_level="full",
            num_training_noise_samples=2,
        ),
    )
    model = make_model(config=config)
    model.train()

    result = model(
        make_request(
            output_sample_size=2,
        )
    )

    assert result.output.shape == (
        2,
        2,
        1,
        16,
        16,
    )


def test_generator_repeats_bottleneck_and_skips(
    monkeypatch,
):
    config = make_config(
        GENERATOR=make_generator(
            noise_level="full",
            num_training_noise_samples=2,
        ),
    )
    model = make_model(config=config)
    model.eval()

    calls = []

    original_repeat = module._repeat_tensor_mask

    def wrapped_repeat(value, repeats):
        calls.append(
            (
                value.tensor.shape,
                repeats,
            )
        )
        return original_repeat(
            value,
            repeats=repeats,
        )

    monkeypatch.setattr(
        module,
        "_repeat_tensor_mask",
        wrapped_repeat,
    )

    result = model(
        make_request(
            output_sample_size=3,
        )
    )

    assert len(calls) == 1 + len(model.down_blocks)
    assert all(repeats == 3 for _, repeats in calls)
    assert result.output.shape[0] == 3


def test_generator_does_not_repeat_when_sample_count_is_none(
    monkeypatch,
):
    config = make_config(
        GENERATOR=make_generator(
            noise_level="full",
        ),
    )
    model = make_model(config=config)
    model.eval()

    with patch.object(
        module,
        "_repeat_tensor_mask",
    ) as repeat:
        with pytest.raises(TypeError):
            model(
                make_request(
                    output_sample_size=None,
                )
            )

    repeat.assert_not_called()


class IdentityTensorMaskBlock(nn.Module):
    def forward(self, value):
        return value


class FakeDownBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, value):
        self.calls.append(value)

        downsampled = TensorMask(
            tensor=value.tensor[:, :, ::2, ::2],
            mask=(value.mask[:, :, ::2, ::2] if value.mask is not None else None),
        )

        return downsampled, value


class FakeUpBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, value, skip):
        self.calls.append(
            (
                value,
                skip,
            )
        )
        return skip


def test_forward_passes_skips_in_reverse_order(
    monkeypatch,
):
    config = make_config(
        channels=[4, 8, 16],
    )
    model = make_model(config=config)

    skip_one = TensorMask(
        tensor=torch.full((2, 4, 16, 16), 1.0),
        mask=None,
    )
    skip_two = TensorMask(
        tensor=torch.full((2, 8, 8, 8), 2.0),
        mask=None,
    )

    class FirstDown(nn.Module):
        def forward(self, value):
            return (
                TensorMask(
                    tensor=torch.zeros(2, 4, 8, 8),
                    mask=None,
                ),
                skip_one,
            )

    class SecondDown(nn.Module):
        def forward(self, value):
            return (
                TensorMask(
                    tensor=torch.zeros(2, 8, 4, 4),
                    mask=None,
                ),
                skip_two,
            )

    received = []

    class CaptureUp(nn.Module):
        def forward(self, value, skip):
            received.append(skip)

            return TensorMask(
                tensor=torch.zeros(
                    skip.tensor.shape[0],
                    skip.tensor.shape[1],
                    skip.tensor.shape[2],
                    skip.tensor.shape[3],
                ),
                mask=None,
            )

    model.initial_mapping = IdentityTensorMaskBlock()
    model.down_blocks = nn.ModuleList(
        [
            FirstDown(),
            SecondDown(),
        ]
    )
    model.bottleneck = IdentityTensorMaskBlock()
    model


def make_sic_config(**kwargs):
    defaults = {
        "channels": [4, 8, 16],
        "block_config": make_block_config(),
        "upsampling_method": "bilinear",
        "skip_alignment_method": "padd",
        "transpose_kernel_sizes": 3,
        "add_skip_connections": True,
        "process_skip_connections": False,
        "mask_pooling": "any",
        "mask_fraction_threshold": 0.5,
        "output_activation": "identity",
        "output_block_hidden_channels": None,
        "init_method": "trunc_normal",
        "GENERATOR": None,
        "clip_output": False,
    }
    defaults.update(kwargs)

    with patch.object(
        module,
        "_unet_config_checks",
        return_value=None,
    ):
        config = UNetSICConfig(**defaults)

    if not hasattr(
        config,
        "checkpoint_config",
    ):
        config.checkpoint_config = None

    return config


def make_sic_model(
    *,
    config=None,
    input_shape=(2, 16, 16),
    output_shape=(1, 16, 16),
    added_features_dim=None,
):
    if config is None:
        config = make_sic_config()

    return UNetSIC(
        config=config,
        input_shape=input_shape,
        output_shape=output_shape,
        added_features_dim=added_features_dim,
    )


class CaptureInitialMapping(nn.Module):
    def __init__(self):
        super().__init__()
        self.received = None

    def forward(self, value):
        self.received = value
        return value


class NoSkipDownBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.received = []

    def forward(self, value):
        self.received.append(value)

        return TensorMask(
            tensor=value.tensor[
                :,
                :,
                ::2,
                ::2,
            ],
            mask=(
                value.mask[
                    :,
                    :,
                    ::2,
                    ::2,
                ]
                if value.mask is not None
                else None
            ),
        )


class CaptureResizeUpBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(
        self,
        value,
        skip=None,
        resize_shape=None,
    ):
        self.calls.append(
            {
                "value": value,
                "skip": skip,
                "resize_shape": resize_shape,
            }
        )

        return TensorMask(
            tensor=torch.nn.functional.interpolate(
                value.tensor,
                size=resize_shape,
            ),
            mask=None,
        )


class CaptureSkipUpBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(
        self,
        value,
        skip=None,
        resize_shape=None,
    ):
        self.calls.append(
            {
                "value": value,
                "skip": skip,
                "resize_shape": resize_shape,
            }
        )

        return skip


def test_config_defaults_include_skip_settings():
    config = make_config()

    assert config.add_skip_connections is True
    assert config.process_skip_connections is False


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
    ],
)
def test_config_preserves_add_skip_connections(value):
    config = make_config(
        add_skip_connections=value,
    )

    assert config.add_skip_connections is value


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
    ],
)
def test_config_preserves_process_skip_connections(value):
    config = make_config(
        process_skip_connections=value,
    )

    assert config.process_skip_connections is value


def test_config_single_channel_level_has_no_kernel_sizes():
    config = make_config(
        channels=[4],
        transpose_kernel_sizes=3,
    )

    assert config.transpose_kernel_sizes == []


def test_config_expects_mask_for_convnext_partial_convolution():
    config = make_config(
        block_config=make_convnext_block_config(
            use_partial_conv=True,
        ),
    )

    assert config.EXPECTS_MASK is True


def test_config_does_not_expect_mask_for_convnext_standard_convolution():
    config = make_config(
        block_config=make_convnext_block_config(
            use_partial_conv=False,
        ),
    )

    assert config.EXPECTS_MASK is False


def test_down_blocks_receive_skip_connection_configuration():
    config = make_config(
        add_skip_connections=True,
        process_skip_connections=True,
    )

    model = make_model(
        config=config,
    )

    assert all(block.return_skip is True for block in model.down_blocks)
    assert all(block.skip_processor is not None for block in model.down_blocks)


def test_down_blocks_disable_skips_when_requested():
    config = make_config(
        add_skip_connections=False,
        process_skip_connections=True,
    )

    model = make_model(
        config=config,
    )

    assert model.add_skip_connections is False
    assert all(block.return_skip is False for block in model.down_blocks)
    assert all(block.skip_processor is None for block in model.down_blocks)


def test_up_blocks_have_no_skip_channels_when_skips_disabled():
    model = make_model(
        config=make_config(
            add_skip_connections=False,
        )
    )

    assert all(block.skip_channels is None for block in model.up_blocks)


def test_forward_decoder_training_uses_generator_sample_count(
    monkeypatch,
):
    model = make_model(
        config=make_config(
            GENERATOR=make_generator(
                noise_level="full",
                num_training_noise_samples=4,
            ),
        )
    )
    model.train()

    repeat_counts = []
    original_repeat = module._repeat_tensor_mask

    def capturing_repeat(
        value,
        repeats,
    ):
        repeat_counts.append(repeats)

        return original_repeat(
            value,
            repeats=repeats,
        )

    monkeypatch.setattr(
        module,
        "_repeat_tensor_mask",
        capturing_repeat,
    )

    model.forward_decoder(
        make_request(
            output_sample_size=2,
        )
    )

    assert repeat_counts
    assert all(count == 4 for count in repeat_counts)


def test_single_level_model_has_no_down_or_up_blocks():
    config = make_config(
        channels=[4],
        transpose_kernel_sizes=3,
    )

    model = make_model(
        config=config,
    )

    assert model.spatial_shapes == [
        (
            16,
            16,
        )
    ]
    assert len(model.down_blocks) == 0
    assert len(model.up_blocks) == 0
    assert model.bottleneck.out_channels == 4


def test_single_level_model_forward():
    config = make_config(
        channels=[4],
        transpose_kernel_sizes=3,
    )
    model = make_model(
        config=config,
    )
    model.eval()

    result = model(make_request())

    assert result.output.shape == (
        2,
        1,
        16,
        16,
    )


def test_config_preserves_add_skip_connections_true():
    config = make_config(
        add_skip_connections=True,
    )

    assert config.add_skip_connections is True


def test_config_preserves_add_skip_connections_false():
    config = make_config(
        add_skip_connections=False,
    )

    assert config.add_skip_connections is False


def test_config_preserves_process_skip_connections_true():
    config = make_config(
        process_skip_connections=True,
    )

    assert config.process_skip_connections is True


def test_config_preserves_process_skip_connections_false():
    config = make_config(
        process_skip_connections=False,
    )

    assert config.process_skip_connections is False


def test_down_blocks_receive_skip_processing_setting():
    model = make_model(
        config=make_config(
            add_skip_connections=True,
            process_skip_connections=True,
        )
    )

    assert all(block.return_skip is True for block in model.down_blocks)
    assert all(block.skip_processor is not None for block in model.down_blocks)


def test_skip_processors_are_not_created_when_skips_disabled():
    model = make_model(
        config=make_config(
            add_skip_connections=False,
            process_skip_connections=True,
        )
    )

    assert all(block.return_skip is False for block in model.down_blocks)
    assert all(block.skip_processor is None for block in model.down_blocks)


def test_up_blocks_receive_skip_channels_when_enabled():
    model = make_model(
        config=make_config(
            channels=[
                4,
                8,
                16,
            ],
            add_skip_connections=True,
        )
    )

    assert [block.skip_channels for block in model.up_blocks] == [
        8,
        4,
    ]


def test_up_blocks_receive_no_skip_channels_when_disabled():
    model = make_model(
        config=make_config(
            channels=[
                4,
                8,
                16,
            ],
            add_skip_connections=False,
        )
    )

    assert [block.skip_channels for block in model.up_blocks] == [
        None,
        None,
    ]


def test_odd_spatial_shapes_are_recorded_correctly():
    model = make_model(
        config=make_config(
            channels=[
                4,
                8,
                16,
            ],
        ),
        input_shape=(
            2,
            17,
            19,
        ),
        output_shape=(
            1,
            17,
            19,
        ),
    )

    assert model.spatial_shapes == [
        (
            17,
            19,
        ),
        (
            9,
            10,
        ),
        (
            5,
            5,
        ),
    ]


def test_rectangular_spatial_shapes_are_recorded_correctly():
    model = make_model(
        config=make_config(
            channels=[
                4,
                8,
                16,
                32,
            ],
        ),
        input_shape=(
            2,
            9,
            17,
        ),
        output_shape=(
            1,
            9,
            17,
        ),
    )

    assert model.spatial_shapes == [
        (
            9,
            17,
        ),
        (
            5,
            9,
        ),
        (
            3,
            5,
        ),
        (
            2,
            3,
        ),
    ]


def test_prepare_input_calls_broadcast_mask_with_none():
    model = make_model()
    tensor = torch.randn(
        2,
        2,
        16,
        16,
    )

    with patch.object(
        module,
        "_broadcast_mask",
        return_value=None,
    ) as broadcast:
        result = model._prepare_input(
            x=tensor,
            x_mask=None,
            added_features=None,
        )

    broadcast.assert_called_once_with(
        None,
        tensor,
    )
    assert result.tensor is tensor
    assert result.mask is None


def test_prepare_input_preserves_channel_order():
    model = make_model(
        added_features_dim=2,
    )

    input_tensor = torch.full(
        (
            1,
            2,
            4,
            4,
        ),
        1.0,
    )
    added_features = torch.full(
        (
            1,
            2,
            4,
            4,
        ),
        2.0,
    )

    result = model._prepare_input(
        x=input_tensor,
        x_mask=None,
        added_features=added_features,
    )

    torch.testing.assert_close(
        result.tensor[:, :2],
        input_tensor,
    )
    torch.testing.assert_close(
        result.tensor[:, 2:],
        added_features,
    )


def test_prepare_input_uses_broadcast_mask_before_adding_feature_mask():
    model = make_model(
        added_features_dim=2,
    )

    input_tensor = torch.zeros(
        2,
        2,
        16,
        16,
    )
    added_features = torch.ones(
        2,
        2,
        16,
        16,
    )
    broadcast_mask = torch.zeros(
        2,
        2,
        16,
        16,
    )

    with patch.object(
        module,
        "_broadcast_mask",
        return_value=broadcast_mask,
    ):
        result = model._prepare_input(
            x=input_tensor,
            x_mask=torch.ones(
                16,
                16,
            ),
            added_features=added_features,
        )

    torch.testing.assert_close(
        result.mask[:, :2],
        broadcast_mask,
    )
    torch.testing.assert_close(
        result.mask[:, 2:],
        torch.ones_like(added_features),
    )


def test_forward_decoder_without_skips_uses_resize_shapes_in_reverse_order():
    model = make_model(
        config=make_config(
            channels=[
                4,
                8,
                16,
            ],
            add_skip_connections=False,
        )
    )

    class InitialMapping(nn.Module):
        def forward(
            self,
            value,
        ):
            return TensorMask(
                tensor=torch.zeros(
                    2,
                    4,
                    16,
                    16,
                ),
                mask=None,
            )

    class FirstDown(nn.Module):
        def forward(
            self,
            value,
        ):
            return TensorMask(
                tensor=torch.zeros(
                    2,
                    8,
                    8,
                    8,
                ),
                mask=None,
            )

    class SecondDown(nn.Module):
        def forward(
            self,
            value,
        ):
            return TensorMask(
                tensor=torch.zeros(
                    2,
                    16,
                    4,
                    4,
                ),
                mask=None,
            )

    class Bottleneck(nn.Module):
        def forward(
            self,
            value,
        ):
            return value

    calls = []

    class CaptureUp(nn.Module):
        def __init__(
            self,
            out_channels,
        ):
            super().__init__()
            self.out_channels = out_channels

        def forward(
            self,
            value,
            skip=None,
            resize_shape=None,
        ):
            calls.append(
                (
                    skip,
                    resize_shape,
                )
            )

            return TensorMask(
                tensor=torch.zeros(
                    value.tensor.shape[0],
                    self.out_channels,
                    *resize_shape,
                ),
                mask=None,
            )

    model.initial_mapping = InitialMapping()
    model.down_blocks = nn.ModuleList(
        [
            FirstDown(),
            SecondDown(),
        ]
    )
    model.bottleneck = Bottleneck()
    model.up_blocks = nn.ModuleList(
        [
            CaptureUp(8),
            CaptureUp(4),
        ]
    )

    result = model.forward_decoder(make_request())

    assert calls == [
        (
            None,
            (
                8,
                8,
            ),
        ),
        (
            None,
            (
                16,
                16,
            ),
        ),
    ]
    assert result.shape == (
        2,
        4,
        16,
        16,
    )


def test_forward_decoder_with_skips_does_not_pass_resize_shape():
    model = make_model(
        config=make_config(
            channels=[
                4,
                8,
            ],
            add_skip_connections=True,
        )
    )

    skip = TensorMask(
        tensor=torch.ones(
            2,
            4,
            16,
            16,
        ),
        mask=None,
    )

    class InitialMapping(nn.Module):
        def forward(
            self,
            value,
        ):
            return skip

    class Down(nn.Module):
        def forward(
            self,
            value,
        ):
            return (
                TensorMask(
                    tensor=torch.zeros(
                        2,
                        8,
                        8,
                        8,
                    ),
                    mask=None,
                ),
                skip,
            )

    class Identity(nn.Module):
        def forward(
            self,
            value,
        ):
            return value

    captured = {}

    class Up(nn.Module):
        def forward(
            self,
            value,
            skip=None,
            resize_shape=None,
        ):
            captured["skip"] = skip
            captured["resize_shape"] = resize_shape
            return skip

    model.initial_mapping = InitialMapping()
    model.down_blocks = nn.ModuleList(
        [
            Down(),
        ]
    )
    model.bottleneck = Identity()
    model.up_blocks = nn.ModuleList(
        [
            Up(),
        ]
    )

    model.forward_decoder(make_request())

    assert captured["skip"] is skip
    assert captured["resize_shape"] is None


def test_forward_decoder_always_resizes_decoder_output(
    monkeypatch,
):
    model = make_model()

    expected = torch.randn(
        2,
        4,
        16,
        16,
    )
    resize = Mock(
        return_value=expected,
    )

    monkeypatch.setattr(
        module,
        "_resize_tensor",
        resize,
    )

    result = model.forward_decoder(make_request())

    resize.assert_called_once()
    assert resize.call_args.args[1] == (
        16,
        16,
    )
    assert result is expected


def test_forward_uses_output_block_property():
    model = make_model()
    model.eval()

    decoder_output = torch.randn(
        2,
        4,
        16,
        16,
    )
    projected_output = torch.randn(
        2,
        1,
        16,
        16,
    )

    model.forward_decoder = Mock(
        return_value=decoder_output,
    )

    output = Mock(
        return_value=projected_output,
    )
    model._modules["output"] = output

    result = model.forward(make_request())

    output.assert_called_once_with(decoder_output)
    assert result.output is projected_output


def test_forward_returns_deterministic_output():
    model = make_model()
    model.eval()

    result = model.forward(make_request())

    assert result.__class__.__name__ == ("deterministicOutput")
    assert torch.is_tensor(result.output)


def test_generator_evaluation_with_zero_samples_does_not_repeat():
    model = make_model(
        config=make_config(
            GENERATOR=make_generator(
                noise_level="full",
            ),
        )
    )
    model.eval()

    with patch.object(
        module,
        "_repeat_tensor_mask",
    ) as repeat:
        result = model(
            make_request(
                output_sample_size=0,
            )
        )

    repeat.assert_not_called()
    assert result.output.shape == (
        2,
        1,
        16,
        16,
    )


def test_generator_medium_noise_only_enables_final_upsampling_noise():
    model = make_model(
        config=make_config(
            channels=[
                4,
                8,
                16,
            ],
            GENERATOR=make_generator(
                noise_level="medium",
            ),
        )
    )

    assert model.up_blocks[0].inject_noise is False
    assert model.up_blocks[1].inject_noise is True
    assert all(block.inject_noise_in_block for block in model.up_blocks)


def test_generator_low_noise_enables_every_upsampling_stage():
    model = make_model(
        config=make_config(
            channels=[
                4,
                8,
                16,
                32,
            ],
            GENERATOR=make_generator(
                noise_level="low",
            ),
        )
    )

    assert all(block.inject_noise for block in model.up_blocks)
    assert all(not block.inject_noise_in_block for block in model.up_blocks)


def test_generator_full_noise_enables_every_noise_location():
    model = make_model(
        config=make_config(
            channels=[
                4,
                8,
                16,
                32,
            ],
            GENERATOR=make_generator(
                noise_level="full",
            ),
        )
    )

    assert all(block.inject_noise for block in model.up_blocks)
    assert all(block.inject_noise_in_block for block in model.up_blocks)


def test_forward_decoder_training_overrides_zero_sample_request(
    monkeypatch,
):
    model = make_model(
        config=make_config(
            GENERATOR=make_generator(
                noise_level="full",
                num_training_noise_samples=3,
            ),
        )
    )
    model.train()

    repeat_counts = []
    original_repeat = module._repeat_tensor_mask

    def capturing_repeat(
        value,
        repeats,
    ):
        repeat_counts.append(repeats)

        return original_repeat(
            value,
            repeats=repeats,
        )

    monkeypatch.setattr(
        module,
        "_repeat_tensor_mask",
        capturing_repeat,
    )

    output = model.forward_decoder(
        make_request(
            output_sample_size=0,
        )
    )

    assert repeat_counts
    assert all(count == 3 for count in repeat_counts)
    assert output.shape[0] == 6


def test_forward_training_uses_configured_generator_samples():
    model = make_model(
        config=make_config(
            GENERATOR=make_generator(
                noise_level="full",
                num_training_noise_samples=3,
            ),
        )
    )
    model.train()

    result = model.forward(
        make_request(
            output_sample_size=1,
        )
    )

    assert result.output.shape == (
        3,
        2,
        1,
        16,
        16,
    )


def test_forward_evaluation_uses_requested_generator_samples():
    model = make_model(
        config=make_config(
            GENERATOR=make_generator(
                noise_level="full",
                num_training_noise_samples=7,
            ),
        )
    )
    model.eval()

    result = model.forward(
        make_request(
            output_sample_size=3,
        )
    )

    assert result.output.shape == (
        3,
        2,
        1,
        16,
        16,
    )


def test_forward_generator_reshapes_batch_before_transpose():
    model = make_model(
        config=make_config(
            GENERATOR=make_generator(
                noise_level="full",
            ),
        )
    )
    model.eval()

    decoder_output = torch.arange(
        6,
        dtype=torch.float32,
    ).reshape(
        6,
        1,
        1,
        1,
    )

    model.forward_decoder = Mock(
        return_value=decoder_output,
    )
    model._modules["output"] = nn.Identity()

    result = model.forward(
        make_request(
            output_sample_size=3,
        )
    )

    assert result.output.shape == (
        3,
        2,
        1,
        1,
        1,
    )

    torch.testing.assert_close(
        result.output[:, 0, 0, 0, 0],
        torch.tensor(
            [
                0.0,
                1.0,
                2.0,
            ]
        ),
    )
    torch.testing.assert_close(
        result.output[:, 1, 0, 0, 0],
        torch.tensor(
            [
                3.0,
                4.0,
                5.0,
            ]
        ),
    )


def test_checkpoint_configuration_calls_load_state_dict():
    config = make_config()
    checkpoint = SimpleNamespace(
        checkpoint_input_shape=np.asarray(
            [
                2,
                16,
                16,
            ]
        ),
        checkpoint_output_shape=np.asarray(
            [
                1,
                16,
                16,
            ]
        ),
    )
    config.checkpoint_config = checkpoint

    with (
        patch.object(
            UNet,
            "_validate_checkpoint_compatibility",
        ),
        patch.object(
            UNet,
            "_load_state_dict",
        ) as load_state,
        patch.object(
            UNet,
            "_initialize_weights",
        ) as initialize,
    ):
        make_model(
            config=config,
        )

    load_state.assert_called_once_with(checkpoint)
    initialize.assert_not_called()


def test_without_checkpoint_calls_initialize_weights():
    config = make_config(
        init_method="xavier",
    )
    config.checkpoint_config = None

    with patch.object(
        UNet,
        "_initialize_weights",
    ) as initialize:
        make_model(
            config=config,
        )

    initialize.assert_called_once_with("xavier")


def test_checkpoint_compatibility_validation_receives_shapes():
    with patch.object(
        UNet,
        "_validate_checkpoint_compatibility",
    ) as validate:
        make_model(
            input_shape=(
                2,
                16,
                16,
            ),
            output_shape=(
                3,
                16,
                16,
            ),
        )

    validate.assert_called_once_with(
        input_shape=(
            2,
            16,
            16,
        ),
        output_shape=(
            3,
            16,
            16,
        ),
    )


def test_build_output_forwards_configuration(
    monkeypatch,
):
    config = make_config(
        output_activation="tanh",
        output_block_hidden_channels=7,
    )
    output = Mock(
        return_value=nn.Identity(),
    )

    monkeypatch.setattr(
        module,
        "UNetOutput",
        output,
    )

    model = make_model(
        config=config,
        output_shape=(
            3,
            16,
            16,
        ),
    )

    output.assert_called_once_with(
        in_channels=4,
        out_channels=3,
        hidden_channels=7,
        activation="tanh",
    )
    assert isinstance(
        model.output,
        nn.Identity,
    )


def test_output_block_property_tracks_replaced_output():
    model = make_model()
    replacement = nn.Identity()

    model.output = replacement

    assert model.output_block is replacement


def test_sic_config_defaults_clip_output_to_false():
    config = make_sic_config()

    assert config.clip_output is False


def test_sic_config_preserves_true_clip_output():
    config = make_sic_config(
        clip_output=True,
    )

    assert config.clip_output is True


def test_sic_config_build_constructs_sic_model():
    config = make_sic_config()
    expected = object()

    with patch.object(
        module,
        "UNetSIC",
        return_value=expected,
    ) as constructor:
        result = config.build(
            input_shape=np.asarray(
                [
                    2,
                    16,
                    16,
                ]
            ),
            output_shape=np.asarray(
                [
                    1,
                    16,
                    16,
                ]
            ),
            added_features_dim=3,
        )

    assert result is expected
    constructor.assert_called_once()

    assert constructor.call_args.kwargs["config"] is config
    assert constructor.call_args.kwargs["added_features_dim"] == 3


def test_sic_model_uses_unet_output_sic():
    model = make_sic_model(
        config=make_sic_config(
            clip_output=True,
        )
    )

    assert isinstance(
        model.output,
        UNetOutputSIC,
    )
    assert model.output.clip_output is True


def test_sic_build_output_forwards_all_configuration(
    monkeypatch,
):
    config = make_sic_config(
        output_activation="sigmoid",
        output_block_hidden_channels=7,
        clip_output=True,
    )

    output = Mock(
        return_value=nn.Identity(),
    )

    monkeypatch.setattr(
        module,
        "UNetOutputSIC",
        output,
    )

    model = make_sic_model(
        config=config,
        output_shape=(
            3,
            16,
            16,
        ),
    )

    output.assert_called_once_with(
        in_channels=4,
        out_channels=3,
        hidden_channels=7,
        activation="sigmoid",
        clip_output=True,
    )
    assert isinstance(
        model.output,
        nn.Identity,
    )


def test_sic_clipped_output_stays_within_unit_interval():
    model = make_sic_model(
        config=make_sic_config(
            output_activation="identity",
            output_block_hidden_channels=None,
            clip_output=True,
        )
    )

    final_conv = model.output.layers[-1]

    with torch.no_grad():
        final_conv.weight.fill_(10.0)
        final_conv.bias.fill_(10.0)

    result = model.output(
        torch.ones(
            2,
            4,
            16,
            16,
        )
    )

    assert torch.all(result >= 0)
    assert torch.all(result <= 1)


def test_sic_unclipped_output_can_exceed_unit_interval():
    model = make_sic_model(
        config=make_sic_config(
            output_activation="identity",
            output_block_hidden_channels=None,
            clip_output=False,
        )
    )

    final_conv = model.output.layers[-1]

    with torch.no_grad():
        final_conv.weight.fill_(10.0)
        final_conv.bias.fill_(10.0)

    result = model.output(
        torch.ones(
            2,
            4,
            16,
            16,
        )
    )

    assert torch.any(result > 1)


@pytest.mark.parametrize(
    (
        "activation",
        "clip_output",
    ),
    [
        (
            "identity",
            False,
        ),
        (
            "identity",
            True,
        ),
        (
            "sigmoid",
            False,
        ),
        (
            "sigmoid",
            True,
        ),
        (
            "tanh",
            False,
        ),
        (
            "tanh",
            True,
        ),
    ],
)
def test_sic_model_forward_shapes(
    activation,
    clip_output,
):
    model = make_sic_model(
        config=make_sic_config(
            output_activation=activation,
            clip_output=clip_output,
        )
    )
    model.eval()

    result = model.forward(make_request())

    assert result.output.shape == (
        2,
        1,
        16,
        16,
    )


@pytest.mark.parametrize(
    "block_config",
    [
        pytest.param(
            make_partial_block_config(),
            id="partial-convolution",
        ),
        pytest.param(
            make_convnext_block_config(
                use_partial_conv=True,
            ),
            id="convnext-partial",
        ),
    ],
)
def test_partial_convolution_models_expect_masks(
    block_config,
):
    config = make_config(
        block_config=block_config,
    )

    assert config.EXPECTS_MASK is True


@pytest.mark.parametrize(
    "block_config",
    [
        pytest.param(
            make_block_config(),
            id="standard-convolution",
        ),
        pytest.param(
            make_convnext_block_config(
                use_partial_conv=False,
            ),
            id="convnext-standard",
        ),
    ],
)
def test_standard_convolution_models_do_not_expect_masks(
    block_config,
):
    config = make_config(
        block_config=block_config,
    )

    assert config.EXPECTS_MASK is False


def test_no_skip_model_forward():
    model = make_model(
        config=make_config(
            add_skip_connections=False,
        )
    )
    model.eval()

    result = model.forward(make_request())

    assert result.output.shape == (
        2,
        1,
        16,
        16,
    )


def test_processed_skip_model_forward():
    model = make_model(
        config=make_config(
            add_skip_connections=True,
            process_skip_connections=True,
        )
    )
    model.eval()

    result = model.forward(make_request())

    assert result.output.shape == (
        2,
        1,
        16,
        16,
    )


class ChunkIdentityBlock(nn.Module):
    def forward(self, value):
        return value


class ChunkDownBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, value):
        self.calls.append(value)

        downsampled = TensorMask(
            tensor=value.tensor[
                :,
                :,
                ::2,
                ::2,
            ],
            mask=None,
        )

        return downsampled, value


class ChunkUpBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(
        self,
        value,
        skip=None,
        resize_shape=None,
    ):
        self.calls.append(
            {
                "value": value,
                "skip": skip,
                "resize_shape": resize_shape,
            }
        )

        if skip is not None:
            spatial_shape = skip.tensor.shape[-2:]
        else:
            spatial_shape = resize_shape

        return TensorMask(
            tensor=torch.nn.functional.interpolate(
                value.tensor,
                size=spatial_shape,
                mode="nearest",
            ),
            mask=None,
        )


def make_chunk_request(
    *,
    input=None,
    input_mask=None,
    added_features=None,
    output_sample_size=3,
    chunk_output_samples=True,
):
    if input is None:
        input = torch.randn(
            2,
            2,
            16,
            16,
        )

    return SimpleNamespace(
        input=input,
        input_mask=input_mask,
        added_features=added_features,
        output_sample_size=output_sample_size,
        chunk_output_samples=chunk_output_samples,
    )


def test_forward_decoder_chunking_runs_decoder_once_per_output_sample(
    monkeypatch,
):
    model = make_model(
        config=make_config(
            channels=[
                4,
                8,
            ],
            GENERATOR=make_generator(
                noise_level="full",
            ),
        )
    )
    model.eval()

    initial = TensorMask(
        tensor=torch.ones(
            2,
            4,
            16,
            16,
        ),
        mask=None,
    )
    bottleneck = TensorMask(
        tensor=torch.ones(
            2,
            8,
            8,
            8,
        ),
        mask=None,
    )

    class InitialMapping(nn.Module):
        def forward(self, value):
            return initial

    class Down(nn.Module):
        def forward(self, value):
            return bottleneck, initial

    up = ChunkUpBlock()

    model.initial_mapping = InitialMapping()
    model.down_blocks = nn.ModuleList(
        [
            Down(),
        ]
    )
    model.bottleneck = ChunkIdentityBlock()
    model.up_blocks = nn.ModuleList(
        [
            up,
        ]
    )

    resize = Mock(
        side_effect=lambda value, shape: value,
    )
    monkeypatch.setattr(
        module,
        "_resize_tensor",
        resize,
    )

    result = model.forward_decoder(
        make_chunk_request(
            output_sample_size=4,
        )
    )

    assert len(up.calls) == 4
    assert resize.call_count == 4
    assert result.shape == (
        8,
        8,
        16,
        16,
    )


def test_forward_decoder_chunking_does_not_repeat_bottleneck_or_skips(
    monkeypatch,
):
    model = make_model(
        config=make_config(
            GENERATOR=make_generator(
                noise_level="full",
            ),
        )
    )
    model.eval()

    repeat = Mock(
        side_effect=AssertionError("Chunked decoding must not repeat tensors.")
    )
    monkeypatch.setattr(
        module,
        "_repeat_tensor_mask",
        repeat,
    )

    result = model.forward_decoder(
        make_chunk_request(
            output_sample_size=3,
            chunk_output_samples=True,
        )
    )

    repeat.assert_not_called()
    assert result.shape[0] == 6


def test_forward_decoder_without_chunking_repeats_bottleneck_and_skips(
    monkeypatch,
):
    model = make_model(
        config=make_config(
            GENERATOR=make_generator(
                noise_level="full",
            ),
        )
    )
    model.eval()

    original_repeat = module._repeat_tensor_mask
    repeat = Mock(
        side_effect=original_repeat,
    )
    monkeypatch.setattr(
        module,
        "_repeat_tensor_mask",
        repeat,
    )

    result = model.forward_decoder(
        make_chunk_request(
            output_sample_size=3,
            chunk_output_samples=False,
        )
    )

    assert repeat.call_count == 1 + len(model.down_blocks)
    assert all(call.kwargs["repeats"] == 3 for call in repeat.call_args_list)
    assert result.shape[0] == 6


@pytest.mark.parametrize(
    "chunk_output_samples",
    [
        None,
        False,
        0,
    ],
)
def test_false_chunk_setting_uses_vectorized_decoder_path(
    monkeypatch,
    chunk_output_samples,
):
    model = make_model(
        config=make_config(
            GENERATOR=make_generator(
                noise_level="full",
            ),
        )
    )
    model.eval()

    original_repeat = module._repeat_tensor_mask
    repeat = Mock(
        side_effect=original_repeat,
    )
    monkeypatch.setattr(
        module,
        "_repeat_tensor_mask",
        repeat,
    )

    model.forward_decoder(
        make_chunk_request(
            output_sample_size=3,
            chunk_output_samples=chunk_output_samples,
        )
    )

    assert repeat.called


def test_chunk_setting_is_ignored_without_generator(
    monkeypatch,
):
    model = make_model(
        config=make_config(
            GENERATOR=None,
        )
    )
    model.eval()

    repeat = Mock(
        side_effect=AssertionError("Tensors must not be repeated without a generator.")
    )
    monkeypatch.setattr(
        module,
        "_repeat_tensor_mask",
        repeat,
    )

    result = model.forward_decoder(
        make_chunk_request(
            output_sample_size=4,
            chunk_output_samples=True,
        )
    )

    repeat.assert_not_called()
    assert result.shape == (
        2,
        4,
        16,
        16,
    )


@pytest.mark.parametrize(
    "output_sample_size",
    [
        0,
        1,
    ],
)
def test_chunk_setting_is_ignored_for_at_most_one_output_sample(
    monkeypatch,
    output_sample_size,
):
    model = make_model(
        config=make_config(
            GENERATOR=make_generator(
                noise_level="full",
            ),
        )
    )
    model.eval()

    repeat = Mock(
        side_effect=AssertionError(
            "Tensors must not be repeated for zero or one sample."
        )
    )
    monkeypatch.setattr(
        module,
        "_repeat_tensor_mask",
        repeat,
    )

    result = model.forward_decoder(
        make_chunk_request(
            output_sample_size=output_sample_size,
            chunk_output_samples=True,
        )
    )

    repeat.assert_not_called()
    assert result.shape == (
        2,
        4,
        16,
        16,
    )


def test_chunked_decoder_uses_generator_training_sample_count(
    monkeypatch,
):
    model = make_model(
        config=make_config(
            GENERATOR=make_generator(
                noise_level="full",
                num_training_noise_samples=4,
            ),
        )
    )
    model.train()

    initial = TensorMask(
        tensor=torch.ones(
            2,
            4,
            16,
            16,
        ),
        mask=None,
    )
    bottleneck = TensorMask(
        tensor=torch.ones(
            2,
            8,
            8,
            8,
        ),
        mask=None,
    )

    class InitialMapping(nn.Module):
        def forward(self, value):
            return initial

    class Down(nn.Module):
        def forward(self, value):
            return bottleneck, initial

    up = ChunkUpBlock()

    model.initial_mapping = InitialMapping()
    model.down_blocks = nn.ModuleList(
        [
            Down(),
        ]
    )
    model.bottleneck = ChunkIdentityBlock()
    model.up_blocks = nn.ModuleList(
        [
            up,
        ]
    )

    monkeypatch.setattr(
        module,
        "_resize_tensor",
        lambda value, shape: value,
    )

    result = model.forward_decoder(
        make_chunk_request(
            output_sample_size=2,
            chunk_output_samples=True,
        )
    )

    assert len(up.calls) == 4
    assert result.shape[0] == 8


def test_chunked_decoder_passes_skips_in_reverse_order(
    monkeypatch,
):
    model = make_model(
        config=make_config(
            channels=[
                4,
                8,
                16,
            ],
            add_skip_connections=True,
            GENERATOR=make_generator(
                noise_level="full",
            ),
        )
    )
    model.eval()

    first_skip = TensorMask(
        tensor=torch.full(
            (
                2,
                4,
                16,
                16,
            ),
            1.0,
        ),
        mask=None,
    )
    second_skip = TensorMask(
        tensor=torch.full(
            (
                2,
                8,
                8,
                8,
            ),
            2.0,
        ),
        mask=None,
    )
    bottleneck = TensorMask(
        tensor=torch.zeros(
            2,
            16,
            4,
            4,
        ),
        mask=None,
    )

    class InitialMapping(nn.Module):
        def forward(self, value):
            return first_skip

    class FirstDown(nn.Module):
        def forward(self, value):
            return (
                TensorMask(
                    tensor=torch.zeros(
                        2,
                        8,
                        8,
                        8,
                    ),
                    mask=None,
                ),
                first_skip,
            )

    class SecondDown(nn.Module):
        def forward(self, value):
            return bottleneck, second_skip

    first_up = ChunkUpBlock()
    second_up = ChunkUpBlock()

    model.initial_mapping = InitialMapping()
    model.down_blocks = nn.ModuleList(
        [
            FirstDown(),
            SecondDown(),
        ]
    )
    model.bottleneck = ChunkIdentityBlock()
    model.up_blocks = nn.ModuleList(
        [
            first_up,
            second_up,
        ]
    )

    monkeypatch.setattr(
        module,
        "_resize_tensor",
        lambda value, shape: value,
    )

    model.forward_decoder(
        make_chunk_request(
            output_sample_size=3,
            chunk_output_samples=True,
        )
    )

    assert len(first_up.calls) == 3
    assert len(second_up.calls) == 3
    assert all(call["skip"] is second_skip for call in first_up.calls)
    assert all(call["skip"] is first_skip for call in second_up.calls)
    assert all(
        call["resize_shape"] is None for call in first_up.calls + second_up.calls
    )


def test_chunked_decoder_without_skips_uses_reverse_spatial_shapes(
    monkeypatch,
):
    model = make_model(
        config=make_config(
            channels=[
                4,
                8,
                16,
            ],
            add_skip_connections=False,
            GENERATOR=make_generator(
                noise_level="full",
            ),
        )
    )
    model.eval()

    model.initial_mapping = CaptureInitialMapping()
    model.initial_mapping_output = TensorMask(
        tensor=torch.zeros(
            2,
            4,
            16,
            16,
        ),
        mask=None,
    )

    class InitialMapping(nn.Module):
        def forward(self, value):
            return TensorMask(
                tensor=torch.zeros(
                    2,
                    4,
                    16,
                    16,
                ),
                mask=None,
            )

    class FirstDown(nn.Module):
        def forward(self, value):
            return TensorMask(
                tensor=torch.zeros(
                    2,
                    8,
                    8,
                    8,
                ),
                mask=None,
            )

    class SecondDown(nn.Module):
        def forward(self, value):
            return TensorMask(
                tensor=torch.zeros(
                    2,
                    16,
                    4,
                    4,
                ),
                mask=None,
            )

    first_up = CaptureResizeUpBlock()
    second_up = CaptureResizeUpBlock()

    model.initial_mapping = InitialMapping()
    model.down_blocks = nn.ModuleList(
        [
            FirstDown(),
            SecondDown(),
        ]
    )
    model.bottleneck = ChunkIdentityBlock()
    model.up_blocks = nn.ModuleList(
        [
            first_up,
            second_up,
        ]
    )
    model.spatial_shapes = [
        (
            16,
            16,
        ),
        (
            8,
            8,
        ),
        (
            4,
            4,
        ),
    ]

    monkeypatch.setattr(
        module,
        "_resize_tensor",
        lambda value, shape: value,
    )

    model.forward_decoder(
        make_chunk_request(
            output_sample_size=3,
            chunk_output_samples=True,
        )
    )

    assert [call["resize_shape"] for call in first_up.calls] == [
        (
            8,
            8,
        ),
    ] * 3
    assert [call["resize_shape"] for call in second_up.calls] == [
        (
            16,
            16,
        ),
    ] * 3
    assert all(call["skip"] is None for call in first_up.calls + second_up.calls)


def test_chunked_decoder_resizes_each_sample_separately(
    monkeypatch,
):
    model = make_model(
        config=make_config(
            GENERATOR=make_generator(
                noise_level="full",
            ),
        )
    )
    model.eval()

    resize = Mock(
        side_effect=lambda value, shape: value,
    )
    monkeypatch.setattr(
        module,
        "_resize_tensor",
        resize,
    )

    model.forward_decoder(
        make_chunk_request(
            output_sample_size=5,
            chunk_output_samples=True,
        )
    )

    assert resize.call_count == 5
    assert all(
        call.args[1]
        == (
            16,
            16,
        )
        for call in resize.call_args_list
    )


def test_vectorized_decoder_resizes_once(
    monkeypatch,
):
    model = make_model(
        config=make_config(
            GENERATOR=make_generator(
                noise_level="full",
            ),
        )
    )
    model.eval()

    resize = Mock(
        side_effect=lambda value, shape: value,
    )
    monkeypatch.setattr(
        module,
        "_resize_tensor",
        resize,
    )

    model.forward_decoder(
        make_chunk_request(
            output_sample_size=5,
            chunk_output_samples=False,
        )
    )

    resize.assert_called_once()


def test_chunked_forward_returns_requested_sample_shape():
    model = make_model(
        config=make_config(
            GENERATOR=make_generator(
                noise_level="full",
            ),
        )
    )
    model.eval()

    result = model.forward(
        make_chunk_request(
            output_sample_size=3,
            chunk_output_samples=True,
        )
    )

    assert result.output.shape == (
        3,
        2,
        1,
        16,
        16,
    )


def test_chunked_forward_training_returns_configured_sample_shape():
    model = make_model(
        config=make_config(
            GENERATOR=make_generator(
                noise_level="full",
                num_training_noise_samples=4,
            ),
        )
    )
    model.train()

    result = model.forward(
        make_chunk_request(
            output_sample_size=2,
            chunk_output_samples=True,
        )
    )

    assert result.output.shape == (
        4,
        2,
        1,
        16,
        16,
    )


def test_chunked_forward_applies_output_block_to_concatenated_decoder_output():
    model = make_model(
        config=make_config(
            GENERATOR=make_generator(
                noise_level="full",
            ),
        )
    )
    model.eval()

    decoder_output = torch.randn(
        6,
        4,
        16,
        16,
    )
    projected_output = torch.randn(
        6,
        1,
        16,
        16,
    )

    model.forward_decoder = Mock(
        return_value=decoder_output,
    )
    output = Mock(
        return_value=projected_output,
    )
    model._modules["output"] = output

    result = model.forward(
        make_chunk_request(
            output_sample_size=3,
            chunk_output_samples=True,
        )
    )

    output.assert_called_once_with(decoder_output)
    assert result.output.shape == (
        3,
        2,
        1,
        16,
        16,
    )


def test_chunked_and_vectorized_decoder_shapes_match():
    model = make_model(
        config=make_config(
            GENERATOR=make_generator(
                noise_level="full",
            ),
        )
    )
    model.eval()

    with torch.no_grad():
        chunked = model.forward_decoder(
            make_chunk_request(
                output_sample_size=3,
                chunk_output_samples=True,
            )
        )
        vectorized = model.forward_decoder(
            make_chunk_request(
                output_sample_size=3,
                chunk_output_samples=False,
            )
        )

    assert chunked.shape == vectorized.shape


def test_chunked_decoder_output_is_finite():
    model = make_model(
        config=make_config(
            GENERATOR=make_generator(
                noise_level="full",
            ),
        )
    )
    model.eval()

    with torch.no_grad():
        result = model.forward_decoder(
            make_chunk_request(
                output_sample_size=3,
                chunk_output_samples=True,
            )
        )

    assert torch.isfinite(result).all()


def test_chunked_decoder_preserves_input_dtype():
    model = make_model(
        config=make_config(
            GENERATOR=make_generator(
                noise_level="full",
            ),
        )
    ).double()
    model.eval()

    with torch.no_grad():
        result = model.forward_decoder(
            make_chunk_request(
                input=torch.randn(
                    2,
                    2,
                    16,
                    16,
                    dtype=torch.float64,
                ),
                output_sample_size=3,
                chunk_output_samples=True,
            )
        )

    assert result.dtype == torch.float64


def test_chunked_decoder_forwards_added_features():
    model = make_model(
        config=make_config(
            GENERATOR=make_generator(
                noise_level="full",
            ),
        ),
        added_features_dim=2,
    )
    model.eval()

    prepare_input = Mock(
        wraps=model._prepare_input,
    )
    model._prepare_input = prepare_input

    input_tensor = torch.randn(
        2,
        2,
        16,
        16,
    )
    input_mask = torch.ones(
        2,
        1,
        16,
        16,
    )
    added_features = torch.randn(
        2,
        2,
        16,
        16,
    )

    model.forward_decoder(
        make_chunk_request(
            input=input_tensor,
            input_mask=input_mask,
            added_features=added_features,
            output_sample_size=3,
            chunk_output_samples=True,
        )
    )

    prepare_input.assert_called_once_with(
        x=input_tensor,
        x_mask=input_mask,
        added_features=added_features,
    )


def test_chunked_decoder_uses_fresh_decoder_reference_for_each_sample():
    model = make_model(
        config=make_config(
            channels=[
                4,
                8,
            ],
            GENERATOR=make_generator(
                noise_level="full",
            ),
        )
    )
    model.eval()

    bottleneck_tensor = torch.randn(
        2,
        8,
        8,
        8,
    )
    bottleneck = TensorMask(
        tensor=bottleneck_tensor,
        mask=None,
    )
    skip = TensorMask(
        tensor=torch.randn(
            2,
            4,
            16,
            16,
        ),
        mask=None,
    )

    class InitialMapping(nn.Module):
        def forward(self, value):
            return skip

    class Down(nn.Module):
        def forward(self, value):
            return bottleneck, skip

    received = []

    class CaptureUp(nn.Module):
        def forward(
            self,
            value,
            skip=None,
            resize_shape=None,
        ):
            received.append(value)

            return TensorMask(
                tensor=torch.nn.functional.interpolate(
                    value.tensor,
                    size=skip.tensor.shape[-2:],
                    mode="nearest",
                ),
                mask=None,
            )

    model.initial_mapping = InitialMapping()
    model.down_blocks = nn.ModuleList(
        [
            Down(),
        ]
    )
    model.bottleneck = ChunkIdentityBlock()
    model.up_blocks = nn.ModuleList(
        [
            CaptureUp(),
        ]
    )

    model.forward_decoder(
        make_chunk_request(
            output_sample_size=3,
            chunk_output_samples=True,
        )
    )

    assert len(received) == 3
    assert all(value is bottleneck for value in received)


def test_chunking_with_no_decoder_levels_concatenates_bottleneck_outputs():
    model = make_model(
        config=make_config(
            channels=[
                4,
            ],
            transpose_kernel_sizes=3,
            GENERATOR=make_generator(
                noise_level="full",
            ),
        )
    )
    model.eval()

    result = model.forward_decoder(
        make_chunk_request(
            output_sample_size=3,
            chunk_output_samples=True,
        )
    )

    assert result.shape == (
        6,
        4,
        16,
        16,
    )


def test_chunking_with_no_decoder_levels_forward_shape():
    model = make_model(
        config=make_config(
            channels=[
                4,
            ],
            transpose_kernel_sizes=3,
            GENERATOR=make_generator(
                noise_level="full",
            ),
        )
    )
    model.eval()

    result = model.forward(
        make_chunk_request(
            output_sample_size=3,
            chunk_output_samples=True,
        )
    )

    assert result.output.shape == (
        3,
        2,
        1,
        16,
        16,
    )


@pytest.mark.parametrize(
    (
        "activation",
        "clip_output",
    ),
    [
        (
            "identity",
            False,
        ),
        (
            "identity",
            True,
        ),
        (
            "sigmoid",
            False,
        ),
        (
            "sigmoid",
            True,
        ),
        (
            "tanh",
            False,
        ),
        (
            "tanh",
            True,
        ),
    ],
)
def test_sic_chunked_forward_shapes(
    activation,
    clip_output,
):
    model = make_sic_model(
        config=make_sic_config(
            output_activation=activation,
            clip_output=clip_output,
            GENERATOR=make_generator(
                noise_level="full",
            ),
        )
    )
    model.eval()

    result = model.forward(
        make_chunk_request(
            output_sample_size=3,
            chunk_output_samples=True,
        )
    )

    assert result.output.shape == (
        3,
        2,
        1,
        16,
        16,
    )


def test_sic_chunked_clipped_output_stays_in_unit_interval():
    model = make_sic_model(
        config=make_sic_config(
            output_activation="identity",
            output_block_hidden_channels=None,
            clip_output=True,
            GENERATOR=make_generator(
                noise_level="full",
            ),
        )
    )
    model.eval()

    final_conv = model.output.layers[-1]

    with torch.no_grad():
        final_conv.weight.fill_(10.0)
        final_conv.bias.fill_(10.0)

        result = model.forward(
            make_chunk_request(
                output_sample_size=3,
                chunk_output_samples=True,
            )
        )

    assert torch.all(result.output >= 0)
    assert torch.all(result.output <= 1)
