from copy import copy
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torch.nn as nn

from cccma_ppp.architectures.layers.unet import UNetOutputSIC
from cccma_ppp.architectures.layers.generic import LayerNorm2d
import cccma_ppp.architectures.layers.unet as module
from cccma_ppp.architectures.layers.conv import (
    ConvBlock,
    ConvBlockConfig,
    ConvNeXtBlock,
    ConvNeXtBlockConfig,
    LatentBlock,
    PartialConvBlock,
    PartialConvBlockConfig,
    TensorMask,
)
from cccma_ppp.architectures.layers.partialconv2d import PartialConv2d
from cccma_ppp.architectures.layers.unet import (
    DownBlock,
    UNetOutput,
    UpBlock,
    build_conv_block,
)


def make_conv_config(**overrides):
    values = {
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
    values.update(overrides)
    return ConvBlockConfig(**values)


def make_partial_config(**overrides):
    values = {
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
    values.update(overrides)
    return PartialConvBlockConfig(**values)


def make_convnext_config(**overrides):
    values = {
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
    values.update(overrides)
    return ConvNeXtBlockConfig(**values)


def make_tensor_mask(
    *,
    batch_size=2,
    channels=3,
    height=8,
    width=8,
    with_mask=False,
    fill_value=None,
):
    if fill_value is None:
        tensor = torch.randn(
            batch_size,
            channels,
            height,
            width,
        )
    else:
        tensor = torch.full(
            (
                batch_size,
                channels,
                height,
                width,
            ),
            fill_value,
        )

    mask = None

    if with_mask:
        mask = torch.ones(
            batch_size,
            1,
            height,
            width,
        )

    return TensorMask(
        tensor=tensor,
        mask=mask,
    )


class FixedTensorMaskModule(nn.Module):
    def __init__(self, output):
        super().__init__()
        self.output = output
        self.received = []

    def forward(self, value):
        self.received.append(value)
        return self.output


class IdentityTensorMaskModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.received = []

    def forward(self, value):
        self.received.append(value)
        return value


class AddOneTensorMaskModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.received = []

    def forward(self, value):
        self.received.append(value)

        return TensorMask(
            tensor=value.tensor + 1,
            mask=value.mask,
        )


class CaptureTensorMaskModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.received = []

    def forward(self, value):
        self.received.append(value)
        return value


class UnsupportedConfig:
    padding_method = "zeros"

    def setup_generator(self, inject_noise=False):
        result = copy(self)
        result.inject_noise = inject_noise
        return result


@pytest.mark.pruned
@pytest.mark.parametrize(
    ("config", "expected_type"),
    [
        pytest.param(
            make_conv_config(),
            ConvBlock,
            id="standard",
        ),
        pytest.param(
            make_partial_config(),
            PartialConvBlock,
            id="partial",
        ),
        pytest.param(
            make_convnext_config(),
            ConvNeXtBlock,
            id="convnext",
        ),
    ],
)
def test_build_conv_block_supported_types(
    config,
    expected_type,
):
    block = build_conv_block(
        in_channels=3,
        out_channels=5,
        config=config,
    )

    assert isinstance(block, expected_type)
    assert block.out_channels == 5


@pytest.mark.pruned
@pytest.mark.parametrize(
    "config",
    [
        make_conv_config(),
        make_partial_config(),
        make_convnext_config(),
    ],
)
def test_build_conv_block_does_not_mutate_config(
    config,
):
    original = copy(config)

    build_conv_block(
        in_channels=3,
        out_channels=5,
        config=config,
        inject_noise=True,
    )

    assert config == original


@pytest.mark.pruned
@pytest.mark.parametrize(
    ("config", "collection_name"),
    [
        (
            make_conv_config(),
            "stages",
        ),
        (
            make_partial_config(),
            "stages",
        ),
        (
            make_convnext_config(),
            "blocks",
        ),
    ],
)
def test_build_conv_block_passes_noise_setting(
    config,
    collection_name,
):
    block = build_conv_block(
        in_channels=3,
        out_channels=5,
        config=config,
        inject_noise=True,
    )

    collection = getattr(
        block,
        collection_name,
    )

    assert collection[0].inject_noise is True


@pytest.mark.pruned
@pytest.mark.parametrize(
    ("config", "collection_name"),
    [
        (
            make_conv_config(),
            "stages",
        ),
        (
            make_partial_config(),
            "stages",
        ),
        (
            make_convnext_config(),
            "blocks",
        ),
    ],
)
def test_build_conv_block_disables_noise_by_default(
    config,
    collection_name,
):
    block = build_conv_block(
        in_channels=3,
        out_channels=5,
        config=config,
    )

    collection = getattr(
        block,
        collection_name,
    )

    assert collection[0].inject_noise is False


@pytest.mark.pruned
def test_build_conv_block_rejects_unsupported_config():
    with pytest.raises(
        TypeError,
        match="Unsupported UNet block configuration",
    ):
        build_conv_block(
            in_channels=3,
            out_channels=5,
            config=UnsupportedConfig(),
        )


def test_build_conv_block_error_reports_config_type():
    with pytest.raises(
        TypeError,
        match="UnsupportedConfig",
    ):
        build_conv_block(
            in_channels=3,
            out_channels=5,
            config=UnsupportedConfig(),
        )


def test_build_conv_block_requires_latent_output_shape():
    with pytest.raises(
        ValueError,
        match="block_output_shape must be available",
    ):
        build_conv_block(
            in_channels=3,
            out_channels=5,
            config=make_conv_config(),
            latent_size=8,
            block_output_shape=None,
        )


@pytest.mark.pruned
def test_build_conv_block_rejects_latent_channel_mismatch():
    with pytest.raises(
        ValueError,
        match="starts with 7 channels",
    ):
        build_conv_block(
            in_channels=3,
            out_channels=5,
            config=make_conv_config(),
            latent_size=8,
            block_output_shape=(7, 4, 4),
        )


@pytest.mark.pruned
@pytest.mark.parametrize(
    (
        "config",
        "get_log_var",
    ),
    [
        (
            make_conv_config(),
            True,
        ),
        (
            make_conv_config(),
            False,
        ),
        (
            make_partial_config(),
            True,
        ),
        (
            make_convnext_config(),
            True,
        ),
    ],
)
def test_build_conv_block_constructs_latent_block(
    config,
    get_log_var,
):
    block = build_conv_block(
        in_channels=3,
        out_channels=5,
        config=config,
        latent_size=8,
        block_output_shape=(5, 4, 4),
        get_log_var=get_log_var,
        latent_normalization="layer",
    )

    assert isinstance(block, LatentBlock)


def make_down_block(
    *,
    return_skip=True,
    process_skip=False,
    mask_pooling="any",
    mask_fraction_threshold=0.5,
):
    return DownBlock(
        in_channels=3,
        out_channels=5,
        block_config=make_conv_config(),
        mask_pooling=mask_pooling,
        mask_fraction_threshold=mask_fraction_threshold,
        return_skip=return_skip,
        process_skip=process_skip,
    )


def make_up_block(
    *,
    block_config=None,
    skip_channels=3,
    upsampling_method="bilinear",
    skip_alignment_method="strict",
    transpose_kernel_size=2,
    inject_noise=False,
    inject_noise_in_block=False,
):
    if block_config is None:
        block_config = make_conv_config()

    return UpBlock(
        input_channels=4,
        skip_channels=skip_channels,
        out_channels=2,
        block_config=block_config,
        upsampling_method=upsampling_method,
        skip_alignment_method=(skip_alignment_method),
        transpose_kernel_size=(transpose_kernel_size),
        inject_noise=inject_noise,
        inject_noise_in_block=(inject_noise_in_block),
    )


@pytest.mark.pruned
def test_up_block_transpose_convolution():
    block = make_up_block(
        upsampling_method="transpose_conv",
    )

    assert isinstance(
        block.upsample,
        nn.ConvTranspose2d,
    )
    assert isinstance(
        block.channel_projection,
        nn.Identity,
    )
    assert block.upsample.in_channels == 4
    assert block.upsample.out_channels == 2
    assert block.upsample.kernel_size == (2, 2)
    assert block.upsample.stride == (2, 2)


@pytest.mark.pruned
def test_up_block_transpose_noise_adds_input_channel():
    block = make_up_block(
        upsampling_method="transpose_conv",
        inject_noise=True,
    )

    assert block.upsample.in_channels == 5


@pytest.mark.pruned
def test_up_block_bilinear_standard_projection():
    block = make_up_block(
        upsampling_method="bilinear",
    )

    assert isinstance(
        block.upsample,
        nn.Upsample,
    )
    assert isinstance(
        block.channel_projection,
        nn.Conv2d,
    )
    assert not isinstance(
        block.channel_projection,
        PartialConv2d,
    )
    assert block.channel_projection.in_channels == 4
    assert block.channel_projection.out_channels == 2


@pytest.mark.pruned
def test_up_block_bilinear_noise_adds_projection_channel():
    block = make_up_block(
        upsampling_method="bilinear",
        inject_noise=True,
    )

    assert block.channel_projection.in_channels == 5


@pytest.mark.pruned
def test_up_block_bilinear_partial_projection():
    block = make_up_block(
        block_config=make_partial_config(),
    )

    assert isinstance(
        block.channel_projection,
        PartialConv2d,
    )
    assert block.channel_projection.multi_channel is False
    assert block.channel_projection.return_mask is False


@pytest.mark.pruned
def test_up_block_convnext_partial_projection():
    block = make_up_block(
        block_config=make_convnext_config(
            use_partial_conv=True,
        )
    )

    assert isinstance(
        block.channel_projection,
        PartialConv2d,
    )


@pytest.mark.pruned
def test_up_block_convnext_standard_projection():
    block = make_up_block(
        block_config=make_convnext_config(
            use_partial_conv=False,
        )
    )

    assert isinstance(
        block.channel_projection,
        nn.Conv2d,
    )
    assert not isinstance(
        block.channel_projection,
        PartialConv2d,
    )


def test_up_block_rejects_unsupported_upsampling_method():
    with pytest.raises(
        ValueError,
        match="Unsupported upsampling mode",
    ):
        make_up_block(
            upsampling_method="nearest",
        )


@pytest.mark.pruned
def test_up_block_stores_configuration():
    block = make_up_block(
        block_config=make_conv_config(
            padding_method="reflect",
        ),
        skip_alignment_method="interpolation",
        inject_noise=True,
        inject_noise_in_block=True,
    )

    assert block.input_channels == 4
    assert block.skip_channels == 3
    assert block.out_channels == 2
    assert block.upsampling_method == "bilinear"
    assert block.skip_alignment_method == "interpolation"
    assert block.skip_padding_method == "reflect"
    assert block.inject_noise is True
    assert block.inject_noise_in_block is True


@pytest.mark.pruned
def test_up_block_merged_block_input_channels_with_skip():
    block = make_up_block(
        skip_channels=3,
    )

    assert block._block.stages[0].conv.in_channels == 5


@pytest.mark.pruned
def test_up_block_merged_block_input_channels_without_skip():
    block = make_up_block(
        skip_channels=None,
    )

    assert block._block.stages[0].conv.in_channels == 2


@pytest.mark.pruned
def test_up_block_does_not_mutate_partial_config():
    config = make_partial_config()

    block = make_up_block(
        block_config=config,
    )

    assert config.multi_channel is True
    assert config.return_mask is True

    assert block._block.stages[0].multi_channel is False
    assert block._block.stages[0].return_mask is False


@pytest.mark.pruned
def test_up_block_enables_noise_inside_block_when_both_flags_true():
    block = make_up_block(
        inject_noise=True,
        inject_noise_in_block=True,
    )

    assert block._block.stages[0].inject_noise is True


@pytest.mark.pruned
@pytest.mark.parametrize(
    (
        "inject_noise",
        "inject_noise_in_block",
    ),
    [
        (
            False,
            False,
        ),
        (
            False,
            True,
        ),
        (
            True,
            False,
        ),
    ],
)
def test_up_block_disables_internal_noise_unless_both_flags_true(
    inject_noise,
    inject_noise_in_block,
):
    block = make_up_block(
        inject_noise=inject_noise,
        inject_noise_in_block=(inject_noise_in_block),
    )

    assert block._block.stages[0].inject_noise is False


def test_up_block_requires_skip_when_skip_channels_configured():
    block = make_up_block(
        skip_channels=3,
    )

    with pytest.raises(
        ValueError,
        match="skip tensor is required",
    ):
        block(
            make_tensor_mask(
                channels=4,
                height=4,
                width=4,
            ),
            skip=None,
        )


@pytest.mark.pruned
def test_up_block_transpose_forward():
    block = make_up_block(
        upsampling_method="transpose_conv",
    )

    result = block(
        make_tensor_mask(
            channels=4,
            height=4,
            width=4,
        ),
        make_tensor_mask(
            channels=3,
            height=8,
            width=8,
        ),
    )

    assert result.tensor.shape == (
        2,
        2,
        8,
        8,
    )
    assert result.mask is None


@pytest.mark.pruned
def test_up_block_bilinear_forward():
    block = make_up_block(
        upsampling_method="bilinear",
    )

    result = block(
        make_tensor_mask(
            channels=4,
            height=4,
            width=4,
        ),
        make_tensor_mask(
            channels=3,
            height=8,
            width=8,
        ),
    )

    assert result.tensor.shape == (
        2,
        2,
        8,
        8,
    )
    assert result.mask is None


@pytest.mark.pruned
def test_up_block_discards_input_and_skip_masks():
    block = make_up_block()

    result = block(
        make_tensor_mask(
            channels=4,
            height=4,
            width=4,
            with_mask=True,
        ),
        make_tensor_mask(
            channels=3,
            height=8,
            width=8,
            with_mask=True,
        ),
    )

    assert result.mask is None


@pytest.mark.pruned
def test_up_block_concatenates_skip_before_upsampled_tensor():
    block = UpBlock(
        input_channels=1,
        skip_channels=1,
        out_channels=1,
        block_config=make_conv_config(),
        upsampling_method="bilinear",
        skip_alignment_method="strict",
        transpose_kernel_size=2,
    )

    capture = CaptureTensorMaskModule()

    block._block = capture
    block.channel_projection = nn.Identity()

    input_value = make_tensor_mask(
        batch_size=1,
        channels=1,
        height=2,
        width=2,
        fill_value=2.0,
    )
    skip_value = make_tensor_mask(
        batch_size=1,
        channels=1,
        height=4,
        width=4,
        fill_value=1.0,
    )

    result = block(
        input_value,
        skip_value,
    )

    received = capture.received[0]

    assert received.tensor.shape == (
        1,
        2,
        4,
        4,
    )

    torch.testing.assert_close(
        received.tensor[:, :1],
        skip_value.tensor,
    )

    torch.testing.assert_close(
        received.tensor[:, 1:],
        torch.full_like(
            skip_value.tensor,
            2.0,
        ),
    )

    assert result is received


@pytest.mark.pruned
def test_up_block_calls_align_to_skip_for_skip_tensor(
    monkeypatch,
):
    block = make_up_block(
        block_config=make_conv_config(
            padding_method="reflect",
        ),
        skip_alignment_method="interpolation",
    )

    aligned = torch.zeros(
        2,
        2,
        9,
        9,
    )

    align = Mock(return_value=aligned)

    monkeypatch.setattr(
        module,
        "align_to_skip",
        align,
    )

    skip = make_tensor_mask(
        channels=3,
        height=9,
        width=9,
    )

    result = block(
        make_tensor_mask(
            channels=4,
            height=4,
            width=4,
        ),
        skip,
    )

    align.assert_called_once()

    assert align.call_args.args[1] == (
        9,
        9,
    )
    assert align.call_args.args[2] == "interpolation"
    assert align.call_args.args[3] == "reflect"

    assert result.tensor.shape == (
        2,
        2,
        9,
        9,
    )


@pytest.mark.pruned
def test_up_block_without_skip_aligns_to_resize_shape(
    monkeypatch,
):
    block = make_up_block(
        skip_channels=None,
        skip_alignment_method="interpolation",
    )

    aligned = torch.zeros(
        2,
        2,
        7,
        9,
    )

    align = Mock(return_value=aligned)

    monkeypatch.setattr(
        module,
        "align_to_skip",
        align,
    )

    result = block(
        make_tensor_mask(
            channels=4,
            height=4,
            width=4,
        ),
        skip=None,
        resize_shape=(7, 9),
    )

    align.assert_called_once()

    assert align.call_args.args[1] == (7, 9)

    assert result.tensor.shape == (
        2,
        2,
        7,
        9,
    )


def test_up_block_transpose_noise_before_upsampling(
    monkeypatch,
):
    block = make_up_block(
        upsampling_method="transpose_conv",
        inject_noise=True,
    )

    calls = []

    def fake_noise(tensor):
        calls.append(tensor.shape)

        noise = torch.zeros(
            tensor.shape[0],
            1,
            tensor.shape[2],
            tensor.shape[3],
        )

        return torch.cat(
            [
                tensor,
                noise,
            ],
            dim=1,
        )

    monkeypatch.setattr(
        module,
        "_noise_injection",
        fake_noise,
    )

    result = block(
        make_tensor_mask(
            channels=4,
            height=4,
            width=4,
        ),
        make_tensor_mask(
            channels=3,
            height=8,
            width=8,
        ),
    )

    assert calls == [
        torch.Size(
            [
                2,
                4,
                4,
                4,
            ]
        )
    ]

    assert result.tensor.shape == (
        2,
        2,
        8,
        8,
    )


@pytest.mark.pruned
def test_up_block_bilinear_noise_after_upsampling(
    monkeypatch,
):
    block = make_up_block(
        upsampling_method="bilinear",
        inject_noise=True,
    )

    calls = []

    def fake_noise(tensor):
        calls.append(tensor.shape)

        noise = torch.zeros(
            tensor.shape[0],
            1,
            tensor.shape[2],
            tensor.shape[3],
        )

        return torch.cat(
            [
                tensor,
                noise,
            ],
            dim=1,
        )

    monkeypatch.setattr(
        module,
        "_noise_injection",
        fake_noise,
    )

    result = block(
        make_tensor_mask(
            channels=4,
            height=4,
            width=4,
        ),
        make_tensor_mask(
            channels=3,
            height=8,
            width=8,
        ),
    )

    assert calls == [
        torch.Size(
            [
                2,
                4,
                8,
                8,
            ]
        )
    ]

    assert result.tensor.shape == (
        2,
        2,
        8,
        8,
    )


@pytest.mark.pruned
def test_up_block_without_noise_does_not_call_noise_injection(
    monkeypatch,
):
    block = make_up_block(
        inject_noise=False,
    )

    noise = Mock(side_effect=AssertionError("Noise injection should not be called."))

    monkeypatch.setattr(
        module,
        "_noise_injection",
        noise,
    )

    result = block(
        make_tensor_mask(
            channels=4,
            height=4,
            width=4,
        ),
        make_tensor_mask(
            channels=3,
            height=8,
            width=8,
        ),
    )

    noise.assert_not_called()

    assert result.tensor.shape == (
        2,
        2,
        8,
        8,
    )


@pytest.mark.pruned
def test_unet_output_identity_direct_projection():
    layer = UNetOutput(
        in_channels=4,
        out_channels=2,
        hidden_channels=None,
        activation="identity",
    )

    assert len(layer.layers) == 1
    assert isinstance(
        layer.layers[0],
        nn.Conv2d,
    )
    assert layer.layers[0].in_channels == 4
    assert layer.layers[0].out_channels == 2
    assert layer.layers[0].kernel_size == (
        1,
        1,
    )


@pytest.mark.pruned
def test_unet_output_sigmoid_direct_projection():
    layer = UNetOutput(
        in_channels=4,
        out_channels=2,
        hidden_channels=None,
        activation="sigmoid",
    )

    assert len(layer.layers) == 2
    assert isinstance(
        layer.layers[-1],
        nn.Sigmoid,
    )


@pytest.mark.pruned
def test_unet_output_tanh_direct_projection():
    layer = UNetOutput(
        in_channels=4,
        out_channels=2,
        hidden_channels=None,
        activation="tanh",
    )

    assert len(layer.layers) == 2
    assert isinstance(
        layer.layers[-1],
        nn.Tanh,
    )


@pytest.mark.pruned
def test_unet_output_rejects_unsupported_activation():
    with pytest.raises(
        ValueError,
        match="Unsupported output activation",
    ):
        UNetOutput(
            in_channels=4,
            out_channels=2,
            hidden_channels=None,
            activation="relu",
        )


@pytest.mark.pruned
def test_unet_output_hidden_identity_structure():
    layer = UNetOutput(
        in_channels=4,
        out_channels=2,
        hidden_channels=6,
        activation="identity",
    )

    assert len(layer.layers) == 4

    assert isinstance(
        layer.layers[0],
        PartialConv2d,
    )
    assert layer.layers[0].in_channels == 4
    assert layer.layers[0].out_channels == 6
    assert layer.layers[0].kernel_size == (
        3,
        3,
    )
    assert layer.layers[0].padding == (
        1,
        1,
    )
    assert layer.layers[0].bias is None
    assert layer.layers[0].multi_channel is False
    assert layer.layers[0].return_mask is False

    assert isinstance(
        layer.layers[1],
        nn.BatchNorm2d,
    )
    assert layer.layers[1].num_features == 6

    assert isinstance(
        layer.layers[2],
        nn.ReLU,
    )
    assert layer.layers[2].inplace is True

    assert isinstance(
        layer.layers[3],
        nn.Conv2d,
    )
    assert layer.layers[3].in_channels == 6
    assert layer.layers[3].out_channels == 2


@pytest.mark.pruned
@pytest.mark.parametrize(
    ("activation", "expected_type"),
    [
        (
            "sigmoid",
            nn.Sigmoid,
        ),
        (
            "tanh",
            nn.Tanh,
        ),
    ],
)
def test_unet_output_hidden_activation_structure(
    activation,
    expected_type,
):
    layer = UNetOutput(
        in_channels=4,
        out_channels=2,
        hidden_channels=6,
        activation=activation,
    )

    assert len(layer.layers) == 5
    assert isinstance(
        layer.layers[-1],
        expected_type,
    )


@pytest.mark.pruned
@pytest.mark.parametrize(
    (
        "hidden_channels",
        "activation",
    ),
    [
        (
            None,
            "identity",
        ),
        (
            None,
            "sigmoid",
        ),
        (
            None,
            "tanh",
        ),
        (
            6,
            "identity",
        ),
        (
            6,
            "sigmoid",
        ),
        (
            6,
            "tanh",
        ),
    ],
)
def test_unet_output_forward_shape(
    hidden_channels,
    activation,
):
    layer = UNetOutput(
        in_channels=4,
        out_channels=2,
        hidden_channels=hidden_channels,
        activation=activation,
    )

    result = layer(
        torch.randn(
            2,
            4,
            8,
            8,
        )
    )

    assert result.shape == (
        2,
        2,
        8,
        8,
    )


@pytest.mark.pruned
def test_unet_output_sigmoid_range():
    layer = UNetOutput(
        in_channels=4,
        out_channels=2,
        hidden_channels=None,
        activation="sigmoid",
    )

    result = layer(
        torch.randn(
            2,
            4,
            8,
            8,
        )
    )

    assert torch.all(result >= 0)
    assert torch.all(result <= 1)


@pytest.mark.pruned
def test_unet_output_tanh_range():
    layer = UNetOutput(
        in_channels=4,
        out_channels=2,
        hidden_channels=None,
        activation="tanh",
    )

    result = layer(
        torch.randn(
            2,
            4,
            8,
            8,
        )
    )

    assert torch.all(result >= -1)
    assert torch.all(result <= 1)


@pytest.mark.pruned
def test_unet_output_identity_allows_negative_values():
    layer = UNetOutput(
        in_channels=1,
        out_channels=1,
        hidden_channels=None,
        activation="identity",
    )

    with torch.no_grad():
        layer.layers[0].weight.fill_(-1.0)
        layer.layers[0].bias.zero_()

    result = layer(
        torch.ones(
            1,
            1,
            2,
            2,
        )
    )

    torch.testing.assert_close(
        result,
        -torch.ones_like(result),
    )


@pytest.mark.pruned
def test_unet_output_sigmoid_zero_logit_is_half():
    layer = UNetOutput(
        in_channels=1,
        out_channels=1,
        hidden_channels=None,
        activation="sigmoid",
    )

    with torch.no_grad():
        layer.layers[0].weight.zero_()
        layer.layers[0].bias.zero_()

    result = layer(
        torch.ones(
            1,
            1,
            2,
            2,
        )
    )

    torch.testing.assert_close(
        result,
        torch.full_like(
            result,
            0.5,
        ),
    )


@pytest.mark.pruned
def test_unet_output_tanh_zero_logit_is_zero():
    layer = UNetOutput(
        in_channels=1,
        out_channels=1,
        hidden_channels=None,
        activation="tanh",
    )

    with torch.no_grad():
        layer.layers[0].weight.zero_()
        layer.layers[0].bias.zero_()

    result = layer(
        torch.ones(
            1,
            1,
            2,
            2,
        )
    )

    torch.testing.assert_close(
        result,
        torch.zeros_like(result),
    )


@pytest.mark.pruned
def test_unet_output_supports_backward():
    layer = UNetOutput(
        in_channels=4,
        out_channels=2,
        hidden_channels=6,
        activation="identity",
    )

    tensor = torch.randn(
        2,
        4,
        8,
        8,
        requires_grad=True,
    )

    result = layer(tensor)
    result.sum().backward()

    assert tensor.grad is not None

    for parameter in layer.parameters():
        assert parameter.grad is not None


@pytest.mark.pruned
def test_unet_output_preserves_float64():
    layer = UNetOutput(
        in_channels=4,
        out_channels=2,
        hidden_channels=None,
        activation="identity",
    ).double()

    tensor = torch.randn(
        2,
        4,
        8,
        8,
        dtype=torch.float64,
    )

    result = layer(tensor)

    assert result.dtype == torch.float64


@pytest.mark.pruned
@pytest.mark.parametrize(
    "config",
    [
        make_partial_config(),
        make_convnext_config(
            use_partial_conv=True,
        ),
    ],
)
def test_down_block_uses_partial_convolution_downsampling(
    config,
):
    block = DownBlock(
        in_channels=3,
        out_channels=5,
        block_config=config,
        mask_pooling="any",
        mask_fraction_threshold=0.5,
    )

    assert block.use_partial_conv_downsample is True
    assert isinstance(
        block.tensor_downsample,
        PartialConv2d,
    )
    assert block.tensor_downsample.in_channels == 3
    assert block.tensor_downsample.out_channels == 5
    assert block.tensor_downsample.kernel_size == (
        3,
        3,
    )
    assert block.tensor_downsample.stride == (
        2,
        2,
    )
    assert block.tensor_downsample.padding == (
        1,
        1,
    )
    assert block.tensor_downsample.multi_channel is True
    assert block.tensor_downsample.return_mask is True


@pytest.mark.pruned
@pytest.mark.parametrize(
    "config",
    [
        make_conv_config(),
        make_convnext_config(
            use_partial_conv=False,
        ),
    ],
)
def test_down_block_uses_standard_convolution_downsampling(
    config,
):
    block = DownBlock(
        in_channels=3,
        out_channels=5,
        block_config=config,
        mask_pooling="any",
        mask_fraction_threshold=0.5,
    )

    assert block.use_partial_conv_downsample is False
    assert isinstance(
        block.tensor_downsample,
        nn.Conv2d,
    )
    assert not isinstance(
        block.tensor_downsample,
        PartialConv2d,
    )
    assert block.tensor_downsample.in_channels == 3
    assert block.tensor_downsample.out_channels == 5


@pytest.mark.pruned
def test_down_block_standard_forward_returns_downsampled_and_skip():
    block = DownBlock(
        in_channels=3,
        out_channels=5,
        block_config=make_conv_config(),
        mask_pooling="any",
        mask_fraction_threshold=0.5,
        return_skip=True,
    )

    transformed = TensorMask(
        tensor=torch.ones(
            2,
            3,
            8,
            8,
        ),
        mask=torch.ones(
            2,
            1,
            8,
            8,
        ),
    )
    block._block = FixedTensorMaskModule(
        transformed,
    )

    downsampled, skip = block(
        make_tensor_mask(
            channels=3,
        )
    )

    assert skip is transformed
    assert downsampled.tensor.shape == (
        2,
        5,
        4,
        4,
    )
    assert downsampled.mask is None


@pytest.mark.pruned
def test_down_block_standard_forward_without_skip():
    block = DownBlock(
        in_channels=3,
        out_channels=5,
        block_config=make_conv_config(),
        mask_pooling="any",
        mask_fraction_threshold=0.5,
        return_skip=False,
    )

    transformed = TensorMask(
        tensor=torch.ones(
            2,
            3,
            8,
            8,
        ),
        mask=None,
    )
    block._block = FixedTensorMaskModule(
        transformed,
    )

    result = block(
        make_tensor_mask(
            channels=3,
        )
    )

    assert isinstance(
        result,
        TensorMask,
    )
    assert result.tensor.shape == (
        2,
        5,
        4,
        4,
    )
    assert result.mask is None


@pytest.mark.pruned
@pytest.mark.parametrize(
    ("input_shape", "expected"),
    [
        (
            (8, 10),
            (4, 5),
        ),
        (
            (9, 11),
            (5, 6),
        ),
        (
            (1, 1),
            (1, 1),
        ),
        (
            (2, 3),
            (1, 2),
        ),
        (
            np.asarray([7, 8]),
            (4, 4),
        ),
    ],
)
def test_down_block_output_shape_matches_strided_convolution(
    input_shape,
    expected,
):
    block = DownBlock(
        in_channels=3,
        out_channels=5,
        block_config=make_conv_config(),
        mask_pooling="any",
        mask_fraction_threshold=0.5,
    )

    assert (
        block.output_shape(
            input_shape,
        )
        == expected
    )


@pytest.mark.pruned
def test_down_block_mask_pooling_arguments_are_currently_unused():
    first = DownBlock(
        in_channels=3,
        out_channels=5,
        block_config=make_conv_config(),
        mask_pooling="any",
        mask_fraction_threshold=0.0,
    )
    second = DownBlock(
        in_channels=3,
        out_channels=5,
        block_config=make_conv_config(),
        mask_pooling="fraction",
        mask_fraction_threshold=1.0,
    )

    assert not hasattr(
        first,
        "mask_pool",
    )
    assert not hasattr(
        second,
        "mask_pool",
    )


@pytest.mark.pruned
def test_build_conv_block_forwards_latent_arguments(
    monkeypatch,
):
    captured = {}

    class FakeLatentBlock(nn.Module):
        def __init__(
            self,
            **kwargs,
        ):
            super().__init__()
            captured.update(kwargs)

    monkeypatch.setattr(
        module,
        "LatentBlock",
        FakeLatentBlock,
    )

    result = build_conv_block(
        in_channels=3,
        out_channels=5,
        config=make_conv_config(),
        latent_size=7,
        block_output_shape=(
            5,
            4,
            4,
        ),
        get_log_var=False,
        latent_normalization="group",
    )

    assert isinstance(
        result,
        FakeLatentBlock,
    )
    assert isinstance(
        captured["conv_block"],
        ConvBlock,
    )
    assert captured["input_shape"] == (
        5,
        4,
        4,
    )
    assert captured["latent_size"] == 7
    assert captured["get_log_var"] is False
    assert captured["latent_normalization"] == "group"


@pytest.mark.pruned
def test_up_block_partial_effective_config_disables_mask_output():
    config = make_partial_config()

    block = UpBlock(
        input_channels=4,
        skip_channels=3,
        out_channels=2,
        block_config=config,
        upsampling_method="bilinear",
        skip_alignment_method="strict",
        transpose_kernel_size=2,
    )

    assert config.multi_channel is True
    assert config.return_mask is True
    assert block._block.config.multi_channel is False
    assert block._block.config.return_mask is False


@pytest.mark.pruned
def test_up_block_convnext_partial_effective_config_disables_mask_output():
    config = make_convnext_config(
        use_partial_conv=True,
    )

    block = UpBlock(
        input_channels=4,
        skip_channels=3,
        out_channels=2,
        block_config=config,
        upsampling_method="bilinear",
        skip_alignment_method="strict",
        transpose_kernel_size=2,
    )

    assert config.multi_channel is True
    assert config.return_mask is True
    assert block._block.config.multi_channel is False
    assert block._block.config.return_mask is False


def test_up_block_without_skip_or_resize_shape_currently_raises():
    block = make_up_block(
        skip_channels=None,
    )

    with pytest.raises(
        UnboundLocalError,
    ):
        block(
            make_tensor_mask(
                channels=4,
                height=4,
                width=4,
            ),
            skip=None,
            resize_shape=None,
        )


@pytest.mark.pruned
def test_up_block_ignores_resize_shape_when_skip_is_configured(
    monkeypatch,
):
    block = make_up_block(
        skip_channels=3,
    )

    align_mock = Mock(side_effect=lambda value, shape, *args: value)
    monkeypatch.setattr(
        module,
        "align_to_skip",
        align_mock,
    )

    block(
        make_tensor_mask(
            channels=4,
            height=4,
            width=4,
        ),
        skip=make_tensor_mask(
            channels=3,
            height=8,
            width=8,
        ),
        resize_shape=(
            99,
            99,
        ),
    )

    assert align_mock.call_args.args[1] == (
        8,
        8,
    )


@pytest.mark.pruned
def test_up_block_bilinear_projects_after_upsampling(
    monkeypatch,
):
    block = make_up_block(
        upsampling_method="bilinear",
    )

    events = []

    class RecordingUpsample(nn.Module):
        def forward(
            self,
            value,
        ):
            events.append(
                (
                    "upsample",
                    value.shape,
                )
            )
            return torch.zeros(
                value.shape[0],
                value.shape[1],
                value.shape[2] * 2,
                value.shape[3] * 2,
            )

    class RecordingProjection(nn.Module):
        def forward(
            self,
            value,
        ):
            events.append(
                (
                    "projection",
                    value.shape,
                )
            )
            return torch.zeros(
                value.shape[0],
                2,
                value.shape[2],
                value.shape[3],
            )

    block.upsample = RecordingUpsample()
    block.channel_projection = RecordingProjection()

    block(
        make_tensor_mask(
            channels=4,
            height=4,
            width=4,
        ),
        make_tensor_mask(
            channels=3,
            height=8,
            width=8,
        ),
    )

    assert events == [
        (
            "upsample",
            torch.Size(
                [
                    2,
                    4,
                    4,
                    4,
                ]
            ),
        ),
        (
            "projection",
            torch.Size(
                [
                    2,
                    4,
                    8,
                    8,
                ]
            ),
        ),
    ]


@pytest.mark.pruned
def test_up_block_output_mask_is_cleared_before_block():
    block = make_up_block(
        skip_channels=3,
    )

    capture = CaptureTensorMaskModule()
    block._block = capture

    block(
        make_tensor_mask(
            channels=4,
            height=4,
            width=4,
            with_mask=True,
        ),
        make_tensor_mask(
            channels=3,
            height=8,
            width=8,
            with_mask=True,
        ),
    )

    assert capture.received[0].mask is None


@pytest.mark.pruned
def test_up_block_resize_path_forwards_aligned_tensor_directly(
    monkeypatch,
):
    block = make_up_block(
        skip_channels=None,
    )

    aligned = torch.randn(
        2,
        2,
        7,
        9,
    )
    monkeypatch.setattr(
        module,
        "align_to_skip",
        Mock(return_value=aligned),
    )

    capture = CaptureTensorMaskModule()
    block._block = capture

    result = block(
        make_tensor_mask(
            channels=4,
            height=4,
            width=4,
        ),
        resize_shape=(
            7,
            9,
        ),
    )

    assert capture.received[0].tensor is aligned
    assert capture.received[0].mask is None
    assert result is capture.received[0]


@pytest.mark.pruned
def test_unet_output_hidden_sigmoid_range():
    layer = UNetOutput(
        in_channels=4,
        out_channels=2,
        hidden_channels=6,
        activation="sigmoid",
    )

    result = layer(
        torch.randn(
            2,
            4,
            8,
            8,
        )
    )

    assert torch.all(result >= 0)
    assert torch.all(result <= 1)


@pytest.mark.pruned
def test_unet_output_hidden_tanh_range():
    layer = UNetOutput(
        in_channels=4,
        out_channels=2,
        hidden_channels=6,
        activation="tanh",
    )

    result = layer(
        torch.randn(
            2,
            4,
            8,
            8,
        )
    )

    assert torch.all(result >= -1)
    assert torch.all(result <= 1)


@pytest.mark.pruned
def test_unet_output_hidden_path_preserves_batch_and_spatial_dimensions():
    layer = UNetOutput(
        in_channels=3,
        out_channels=7,
        hidden_channels=5,
        activation="identity",
    )

    result = layer(
        torch.randn(
            4,
            3,
            11,
            13,
        )
    )

    assert result.shape == (
        4,
        7,
        11,
        13,
    )


@pytest.mark.pruned
def test_down_block_creates_skip_processor_when_requested():
    block = DownBlock(
        in_channels=3,
        out_channels=5,
        block_config=make_conv_config(),
        mask_pooling="any",
        mask_fraction_threshold=0.5,
        return_skip=True,
        process_skip_connections=True,
    )

    assert block.skip_processor is not None
    assert isinstance(
        block.skip_processor,
        ConvBlock,
    )


@pytest.mark.pruned
def test_down_block_does_not_create_unused_skip_processor():
    block = DownBlock(
        in_channels=3,
        out_channels=5,
        block_config=make_conv_config(),
        mask_pooling="any",
        mask_fraction_threshold=0.5,
        return_skip=False,
        process_skip_connections=True,
    )

    assert block.skip_processor is None


@pytest.mark.pruned
def test_down_block_does_not_create_skip_processor_by_default():
    block = DownBlock(
        in_channels=3,
        out_channels=5,
        block_config=make_conv_config(),
        mask_pooling="any",
        mask_fraction_threshold=0.5,
        return_skip=True,
    )

    assert block.skip_processor is None


@pytest.mark.pruned
def test_down_block_processes_skip_after_main_block():
    block = DownBlock(
        in_channels=3,
        out_channels=5,
        block_config=make_conv_config(),
        mask_pooling="any",
        mask_fraction_threshold=0.5,
        return_skip=True,
        process_skip_connections=True,
    )

    transformed = TensorMask(
        tensor=torch.ones(
            2,
            3,
            8,
            8,
        ),
        mask=torch.ones(
            2,
            1,
            8,
            8,
        ),
    )

    main_block = FixedTensorMaskModule(transformed)
    skip_processor = AddOneTensorMaskModule()

    block._block = main_block
    block.skip_processor = skip_processor

    downsampled, skip = block(
        make_tensor_mask(
            channels=3,
        )
    )

    assert main_block.received
    assert skip_processor.received == [transformed]

    torch.testing.assert_close(
        skip.tensor,
        transformed.tensor + 1,
    )
    assert skip.mask is transformed.mask

    assert downsampled.tensor.shape == (
        2,
        5,
        4,
        4,
    )


@pytest.mark.pruned
def test_down_block_partial_forward_preserves_returned_mask(
    monkeypatch,
):
    block = DownBlock(
        in_channels=3,
        out_channels=5,
        block_config=make_partial_config(),
        mask_pooling="fraction",
        mask_fraction_threshold=0.25,
        return_skip=True,
    )

    transformed = TensorMask(
        tensor=torch.ones(
            2,
            3,
            8,
            8,
        ),
        mask=torch.ones(
            2,
            1,
            8,
            8,
        ),
    )
    block._block = FixedTensorMaskModule(transformed)

    expected_tensor = torch.randn(
        2,
        5,
        4,
        4,
    )
    expected_mask = torch.zeros(
        2,
        1,
        4,
        4,
    )

    downsample = Mock(
        return_value=(
            expected_tensor,
            expected_mask,
        )
    )
    block._modules["tensor_downsample"] = downsample

    downsampled, skip = block(
        make_tensor_mask(
            channels=3,
            with_mask=True,
        )
    )

    downsample.assert_called_once_with(
        transformed.tensor,
        transformed.mask,
    )

    assert downsampled.tensor is expected_tensor
    assert downsampled.mask is expected_mask
    assert skip is transformed


def test_down_block_partial_forward_accepts_none_mask():
    block = DownBlock(
        in_channels=3,
        out_channels=5,
        block_config=make_partial_config(),
        mask_pooling="any",
        mask_fraction_threshold=0.5,
        return_skip=False,
    )

    result = block(
        make_tensor_mask(
            channels=3,
            with_mask=False,
        )
    )

    assert isinstance(
        result,
        TensorMask,
    )
    assert result.tensor.shape == (
        2,
        5,
        4,
        4,
    )

    if result.mask is not None:
        assert result.mask.shape[-2:] == (
            4,
            4,
        )


@pytest.mark.pruned
def test_down_block_standard_downsample_uses_configured_padding():
    block = DownBlock(
        in_channels=3,
        out_channels=5,
        block_config=make_conv_config(
            padding_method="reflect",
        ),
        mask_pooling="any",
        mask_fraction_threshold=0.5,
    )

    assert isinstance(
        block.tensor_downsample,
        nn.Conv2d,
    )
    assert block.tensor_downsample.padding_mode == ("reflect")


@pytest.mark.pruned
def test_down_block_partial_downsample_uses_configured_padding():
    block = DownBlock(
        in_channels=3,
        out_channels=5,
        block_config=make_partial_config(
            padding_method="reflect",
        ),
        mask_pooling="any",
        mask_fraction_threshold=0.5,
    )

    assert isinstance(
        block.tensor_downsample,
        PartialConv2d,
    )
    assert block.tensor_downsample.padding_mode == ("reflect")


@pytest.mark.pruned
@pytest.mark.parametrize(
    (
        "input_shape",
        "expected",
    ),
    [
        (
            (),
            (),
        ),
        (
            (0, 0),
            (0, 0),
        ),
        (
            np.asarray(
                [
                    1,
                    2,
                    3,
                    4,
                ]
            ),
            (
                1,
                1,
                2,
                2,
            ),
        ),
    ],
)
def test_down_block_output_shape_additional_cases(
    input_shape,
    expected,
):
    block = DownBlock(
        in_channels=3,
        out_channels=5,
        block_config=make_conv_config(),
        mask_pooling="any",
        mask_fraction_threshold=0.5,
    )

    assert block.output_shape(input_shape) == expected


@pytest.mark.pruned
def test_up_block_transpose_without_skip_resizes_output(
    monkeypatch,
):
    block = make_up_block(
        skip_channels=None,
        upsampling_method="transpose_conv",
        skip_alignment_method="interpolation",
    )

    aligned = torch.randn(
        2,
        2,
        9,
        7,
    )
    align = Mock(return_value=aligned)

    monkeypatch.setattr(
        module,
        "align_to_skip",
        align,
    )

    capture = CaptureTensorMaskModule()
    block._block = capture

    result = block(
        make_tensor_mask(
            channels=4,
            height=4,
            width=4,
        ),
        resize_shape=(
            9,
            7,
        ),
    )

    align.assert_called_once()
    assert align.call_args.args[1] == (
        9,
        7,
    )
    assert capture.received[0].tensor is aligned
    assert result is capture.received[0]


@pytest.mark.pruned
def test_up_block_resize_path_uses_padding_configuration(
    monkeypatch,
):
    block = make_up_block(
        block_config=make_conv_config(
            padding_method="replicate",
        ),
        skip_channels=None,
        skip_alignment_method="padd",
    )

    align = Mock(
        return_value=torch.zeros(
            2,
            2,
            9,
            9,
        )
    )
    monkeypatch.setattr(
        module,
        "align_to_skip",
        align,
    )

    block(
        make_tensor_mask(
            channels=4,
            height=4,
            width=4,
        ),
        resize_shape=(
            9,
            9,
        ),
    )

    assert align.call_args.args[1] == (
        9,
        9,
    )
    assert align.call_args.args[2] == "padd"
    assert align.call_args.args[3] == "replicate"


@pytest.mark.pruned
def test_up_block_partial_projection_forward():
    block = make_up_block(
        block_config=make_partial_config(),
        skip_channels=3,
    )

    result = block(
        make_tensor_mask(
            channels=4,
            height=4,
            width=4,
            with_mask=True,
        ),
        make_tensor_mask(
            channels=3,
            height=8,
            width=8,
            with_mask=True,
        ),
    )

    assert result.tensor.shape == (
        2,
        2,
        8,
        8,
    )
    assert result.mask is None


def test_up_block_convnext_partial_projection_forward():
    block = make_up_block(
        block_config=make_convnext_config(
            use_partial_conv=True,
        ),
        skip_channels=3,
    )

    result = block(
        make_tensor_mask(
            channels=4,
            height=4,
            width=4,
        ),
        make_tensor_mask(
            channels=3,
            height=8,
            width=8,
        ),
    )

    assert result.tensor.shape == (
        2,
        2,
        8,
        8,
    )


@pytest.mark.pruned
def test_up_block_transpose_channel_projection_is_called():
    block = make_up_block(
        upsampling_method="transpose_conv",
    )

    projection = Mock(wraps=block.channel_projection)
    block._modules["channel_projection"] = projection

    block(
        make_tensor_mask(
            channels=4,
            height=4,
            width=4,
        ),
        make_tensor_mask(
            channels=3,
            height=8,
            width=8,
        ),
    )

    projection.assert_called_once()


@pytest.mark.pruned
def test_up_block_skip_alignment_receives_upsampled_tensor(
    monkeypatch,
):
    block = make_up_block(
        skip_channels=3,
    )

    captured = {}

    def fake_align(
        value,
        shape,
        method,
        padding_method,
    ):
        captured["value"] = value
        captured["shape"] = shape
        captured["method"] = method
        captured["padding_method"] = padding_method
        return value

    monkeypatch.setattr(
        module,
        "align_to_skip",
        fake_align,
    )

    block(
        make_tensor_mask(
            channels=4,
            height=4,
            width=4,
        ),
        make_tensor_mask(
            channels=3,
            height=8,
            width=8,
        ),
    )

    assert captured["value"].shape == (
        2,
        2,
        8,
        8,
    )
    assert captured["shape"] == (
        8,
        8,
    )


@pytest.mark.pruned
def test_unet_output_sic_direct_structure():
    layer = UNetOutputSIC(
        in_channels=4,
        out_channels=2,
        hidden_channels=None,
        activation="identity",
    )

    assert len(layer.layers) == 3
    assert isinstance(
        layer.layers[0],
        LayerNorm2d,
    )
    assert isinstance(
        layer.layers[1],
        nn.ReLU,
    )
    assert isinstance(
        layer.layers[2],
        nn.Conv2d,
    )

    assert layer.layers[2].in_channels == 4
    assert layer.layers[2].out_channels == 2


@pytest.mark.pruned
def test_unet_output_sic_hidden_structure():
    layer = UNetOutputSIC(
        in_channels=4,
        out_channels=2,
        hidden_channels=6,
        activation="identity",
    )

    assert len(layer.layers) == 4

    assert isinstance(
        layer.layers[0],
        PartialConv2d,
    )
    assert layer.layers[0].in_channels == 4
    assert layer.layers[0].out_channels == 6
    assert layer.layers[0].bias is None
    assert layer.layers[0].multi_channel is False
    assert layer.layers[0].return_mask is False

    assert isinstance(
        layer.layers[1],
        LayerNorm2d,
    )
    assert isinstance(
        layer.layers[2],
        nn.ReLU,
    )
    assert isinstance(
        layer.layers[3],
        nn.Conv2d,
    )


@pytest.mark.pruned
@pytest.mark.parametrize(
    (
        "hidden_channels",
        "activation",
        "expected_activation",
    ),
    [
        (
            None,
            "sigmoid",
            nn.Sigmoid,
        ),
        (
            None,
            "tanh",
            nn.Tanh,
        ),
        (
            6,
            "sigmoid",
            nn.Sigmoid,
        ),
        (
            6,
            "tanh",
            nn.Tanh,
        ),
    ],
)
def test_unet_output_sic_activation_structure(
    hidden_channels,
    activation,
    expected_activation,
):
    layer = UNetOutputSIC(
        in_channels=4,
        out_channels=2,
        hidden_channels=hidden_channels,
        activation=activation,
    )

    assert isinstance(
        layer.layers[-1],
        expected_activation,
    )


@pytest.mark.parametrize(
    "hidden_channels",
    [
        None,
        6,
    ],
)
@pytest.mark.parametrize(
    "activation",
    [
        "identity",
        "sigmoid",
        "tanh",
    ],
)
def test_unet_output_sic_forward_shape(
    hidden_channels,
    activation,
):
    layer = UNetOutputSIC(
        in_channels=4,
        out_channels=2,
        hidden_channels=hidden_channels,
        activation=activation,
    )

    result = layer(
        torch.randn(
            3,
            4,
            9,
            11,
        )
    )

    assert result.shape == (
        3,
        2,
        9,
        11,
    )


@pytest.mark.pruned
def test_unet_output_sic_clip_output():
    layer = UNetOutputSIC(
        in_channels=1,
        out_channels=1,
        hidden_channels=None,
        activation="identity",
        clip_output=True,
    )

    with torch.no_grad():
        layer.layers[-1].weight.fill_(10.0)
        layer.layers[-1].bias.fill_(-5.0)

    negative = layer(
        -torch.ones(
            1,
            1,
            2,
            2,
        )
    )
    positive = layer(
        torch.ones(
            1,
            1,
            2,
            2,
        )
    )

    assert torch.all(negative >= 0)
    assert torch.all(negative <= 1)
    assert torch.all(positive >= 0)
    assert torch.all(positive <= 1)


@pytest.mark.pruned
def test_unet_output_sic_without_clipping_preserves_unbounded_output():
    layer = UNetOutputSIC(
        in_channels=1,
        out_channels=1,
        hidden_channels=None,
        activation="identity",
        clip_output=False,
    )

    with torch.no_grad():
        layer.layers[-1].weight.fill_(10.0)
        layer.layers[-1].bias.fill_(10.0)

    result = layer(
        torch.ones(
            1,
            1,
            2,
            2,
        )
    )

    assert torch.any(result > 1)


def test_unet_output_sic_rejects_unsupported_activation():
    with pytest.raises(
        ValueError,
        match="Unsupported output activation",
    ):
        UNetOutputSIC(
            in_channels=4,
            out_channels=2,
            hidden_channels=None,
            activation="relu",
        )


@pytest.mark.pruned
@pytest.mark.parametrize(
    "hidden_channels",
    [
        None,
        6,
    ],
)
def test_unet_output_sic_supports_backward(
    hidden_channels,
):
    layer = UNetOutputSIC(
        in_channels=4,
        out_channels=2,
        hidden_channels=hidden_channels,
        activation="identity",
    )

    tensor = torch.randn(
        2,
        4,
        8,
        8,
        requires_grad=True,
    )

    result = layer(tensor)
    result.sum().backward()

    assert tensor.grad is not None

    for parameter in layer.parameters():
        assert parameter.grad is not None


@pytest.mark.pruned
@pytest.mark.parametrize(
    "hidden_channels",
    [
        None,
        6,
    ],
)
def test_unet_output_sic_preserves_float64(
    hidden_channels,
):
    layer = UNetOutputSIC(
        in_channels=4,
        out_channels=2,
        hidden_channels=hidden_channels,
        activation="identity",
    ).double()

    result = layer(
        torch.randn(
            2,
            4,
            8,
            8,
            dtype=torch.float64,
        )
    )

    assert result.dtype == torch.float64
