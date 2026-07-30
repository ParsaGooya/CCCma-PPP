from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn

import cccma_ppp.models.unet_models.deterministic as module
from cccma_ppp.models.layers.conv import (
    ConvBlockConfig,
    ConvNeXtBlockConfig,
    PartialConvBlockConfig,
    TensorMask,
)
from cccma_ppp.models.unet_models.deterministic import (
    UNet,
    UNetConfig,
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
        "bottleneck_dim": 32,
        "block_config": make_block_config(),
        "upsampling_method": "bilinear",
        "skip_alignment_method": "padd",
        "transpose_kernel_sizes": 3,
        "process_skip": False,
        "mask_pooling": "any",
        "mask_fraction_threshold": 0.5,
        "output_activation": "identity",
        "output_block_hidden_channels": None,
        "init_method": "trunc_normal",
        "GENERATOR": None,
    }
    defaults.update(kwargs)

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
    )


@pytest.mark.pruned
def test_config_defaults():
    config = make_config()

    assert config.channels == [4, 8, 16]
    assert config.bottleneck_dim == 32
    assert config.upsampling_method == "bilinear"
    assert config.skip_alignment_method == "padd"
    assert config.transpose_kernel_sizes == [3, 3]
    assert config.process_skip is False
    assert config.mask_pooling == "any"
    assert config.mask_fraction_threshold == pytest.approx(0.5)
    assert config.output_activation == "identity"
    assert config.output_block_hidden_channels is None
    assert config.init_method == "trunc_normal"
    assert config.GENERATOR is None


@pytest.mark.pruned
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


@pytest.mark.pruned
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


@pytest.mark.pruned
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


@pytest.mark.pruned
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


@pytest.mark.parametrize(
    "process_skip",
    [
        True,
        False,
    ],
)
def test_config_preserves_process_skip(process_skip):
    config = make_config(
        process_skip=process_skip,
    )

    assert config.process_skip is process_skip


@pytest.mark.pruned
def test_config_accepts_partial_conv_block():
    block_config = make_partial_block_config()

    config = make_config(
        block_config=block_config,
    )

    assert config.block_config is block_config


@pytest.mark.pruned
def test_config_accepts_convnext_block():
    block_config = make_convnext_block_config()

    config = make_config(
        block_config=block_config,
    )

    assert config.block_config is block_config


@pytest.mark.pruned
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


@pytest.mark.pruned
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


@pytest.mark.pruned
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


@pytest.mark.pruned
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


@pytest.mark.pruned
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


@pytest.mark.pruned
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


@pytest.mark.pruned
def test_model_stores_configuration():
    config = make_config()

    model = make_model(
        config=config,
        added_features_dim=2,
    )

    assert model.config is config
    assert model.init_method == config.init_method
    assert model.added_features_dim == 2


@pytest.mark.pruned
def test_none_added_features_dimension_becomes_zero():
    model = make_model(
        added_features_dim=None,
    )

    assert model.added_features_dim == 0


@pytest.mark.pruned
def test_zero_added_features_dimension_remains_zero():
    model = make_model(
        added_features_dim=0,
    )

    assert model.added_features_dim == 0


@pytest.mark.pruned
def test_initial_mapping_includes_added_feature_channels():
    model = make_model(
        input_shape=(2, 16, 16),
        added_features_dim=3,
    )

    assert model.initial_mapping.stages[0].conv.in_channels == 5
    assert model.initial_mapping.stages[0].conv.out_channels == 4


@pytest.mark.pruned
def test_model_builds_one_down_block_per_channel_transition():
    config = make_config(
        channels=[4, 8, 16, 32],
        transpose_kernel_sizes=3,
    )

    model = make_model(config=config)

    assert len(model.down_blocks) == 3


@pytest.mark.pruned
def test_model_builds_one_up_block_per_channel_transition():
    config = make_config(
        channels=[4, 8, 16, 32],
        transpose_kernel_sizes=3,
    )

    model = make_model(config=config)

    assert len(model.up_blocks) == 3


@pytest.mark.pruned
def test_default_bottleneck_is_twice_last_channel_width():
    config = make_config(
        channels=[4, 8, 16],
        bottleneck_dim=None,
    )

    model = make_model(config=config)

    assert model.bottleneck.out_channels == 32


@pytest.mark.pruned
def test_explicit_bottleneck_dimension():
    config = make_config(
        bottleneck_dim=24,
    )

    model = make_model(config=config)

    assert model.bottleneck.out_channels == 24


@pytest.mark.pruned
def test_output_uses_requested_channel_count():
    model = make_model(
        output_shape=(3, 16, 16),
    )

    final_conv = [
        layer for layer in model.output.modules() if isinstance(layer, nn.Conv2d)
    ][-1]

    assert final_conv.out_channels == 3


@pytest.mark.pruned
def test_down_blocks_receive_skip_processing_flag():
    config = make_config(
        process_skip=True,
    )

    model = make_model(config=config)

    assert all(block.skip_processor is not None for block in model.down_blocks)


@pytest.mark.pruned
def test_down_blocks_without_skip_processing():
    config = make_config(
        process_skip=False,
    )

    model = make_model(config=config)

    assert all(block.skip_processor is None for block in model.down_blocks)


@pytest.mark.pruned
def test_up_blocks_use_kernel_sizes_in_decoder_order():
    config = make_config(
        channels=[4, 8, 16],
        upsampling_method="transpose_conv",
        transpose_kernel_sizes=[2, 3],
    )

    model = make_model(config=config)

    assert model.up_blocks[0].upsample.kernel_size == (2, 2)
    assert model.up_blocks[1].upsample.kernel_size == (3, 3)


@pytest.mark.pruned
def test_no_generator_disables_noise_in_all_up_blocks():
    model = make_model(
        config=make_config(
            GENERATOR=None,
        )
    )

    assert all(not block.inject_noise for block in model.up_blocks)


@pytest.mark.pruned
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


@pytest.mark.pruned
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


@pytest.mark.pruned
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


@pytest.mark.pruned
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


@pytest.mark.pruned
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


@pytest.mark.pruned
def test_forward_without_generator():
    model = make_model()
    model.eval()

    result = model(
        make_request(
            output_sample_size=None,
        )
    )

    assert result.output.shape == (2, 1, 16, 16)


@pytest.mark.pruned
def test_forward_without_generator_ignores_output_sample_size():
    model = make_model()
    model.eval()

    result = model(
        make_request(
            output_sample_size=5,
        )
    )

    assert result.output.shape == (2, 1, 16, 16)


@pytest.mark.pruned
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


@pytest.mark.pruned
def test_forward_output_is_finite():
    model = make_model()
    model.eval()

    result = model(make_request())

    assert torch.isfinite(result.output).all()


@pytest.mark.pruned
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
            output_sample_size=7,
        )
    )

    assert result.output.shape == (
        3,
        2,
        1,
        16,
        16,
    )


@pytest.mark.pruned
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
            output_sample_size=None,
        )
    )

    assert result.output.shape == (
        2,
        2,
        1,
        16,
        16,
    )


@pytest.mark.pruned
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


@pytest.mark.pruned
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


@pytest.mark.pruned
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