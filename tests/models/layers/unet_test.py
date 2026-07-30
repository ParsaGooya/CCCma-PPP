from copy import copy
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

import cccma_ppp.models.layers.unet as module
from cccma_ppp.models.layers.conv import (
    ConvBlock,
    ConvBlockConfig,
    ConvNeXtBlock,
    ConvNeXtBlockConfig,
    LatentBlock,
    PartialConvBlock,
    PartialConvBlockConfig,
    TensorMask,
)
from cccma_ppp.models.layers.partialconv2d import PartialConv2d
from cccma_ppp.models.layers.unet import (
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


@pytest.mark.pruned
def test_down_block_without_skip_processor():
    block = make_down_block(
        return_skip=True,
        process_skip=False,
    )

    assert block.return_skip is True
    assert isinstance(block._block, ConvBlock)
    assert block.skip_processor is None
    assert isinstance(block.tensor_pool, nn.MaxPool2d)


@pytest.mark.pruned
def test_down_block_with_skip_processor():
    block = make_down_block(
        return_skip=True,
        process_skip=True,
    )

    assert isinstance(block.skip_processor, ConvBlock)


@pytest.mark.pruned
def test_down_block_does_not_create_unused_skip_processor():
    block = make_down_block(
        return_skip=False,
        process_skip=True,
    )

    assert block.return_skip is False
    assert block.skip_processor is None


@pytest.mark.parametrize(
    "method",
    [
        "any",
        "all",
        "fraction",
    ],
)
def test_down_block_mask_pooling_configuration(
    method,
):
    block = make_down_block(
        mask_pooling=method,
        mask_fraction_threshold=0.75,
    )

    assert block.mask_pool.method == method
    assert block.mask_pool.fraction_threshold == pytest.approx(0.75)


@pytest.mark.parametrize(
    ("input_shape", "expected"),
    [
        (
            (8, 10),
            (4, 5),
        ),
        (
            (9, 11),
            (4, 5),
        ),
        (
            (1, 1),
            (0, 0),
        ),
    ],
)
def test_down_block_output_shape(
    input_shape,
    expected,
):
    block = make_down_block()

    assert block.output_shape(input_shape) == expected


@pytest.mark.pruned
def test_down_block_returns_downsampled_and_skip():
    block = make_down_block(
        return_skip=True,
    )

    convolved = TensorMask(
        tensor=torch.ones(2, 5, 8, 8),
        mask=None,
    )

    block._block = FixedTensorMaskModule(convolved)

    value = make_tensor_mask(
        channels=3,
    )

    downsampled, skip = block(value)

    assert skip is convolved
    assert downsampled.tensor.shape == (
        2,
        5,
        4,
        4,
    )
    assert downsampled.mask is None


def test_down_block_returns_only_downsampled_when_skip_disabled():
    block = make_down_block(
        return_skip=False,
    )

    convolved = TensorMask(
        tensor=torch.ones(2, 5, 8, 8),
        mask=None,
    )

    block._block = FixedTensorMaskModule(convolved)

    result = block(
        make_tensor_mask(
            channels=3,
        )
    )

    assert isinstance(result, TensorMask)
    assert result.tensor.shape == (
        2,
        5,
        4,
        4,
    )


@pytest.mark.pruned
def test_down_block_processes_skip_when_enabled():
    block = make_down_block(
        return_skip=True,
        process_skip=True,
    )

    block._block = AddOneTensorMaskModule()
    block.skip_processor = AddOneTensorMaskModule()

    value = TensorMask(
        tensor=torch.zeros(2, 3, 8, 8),
        mask=None,
    )

    downsampled, skip = block(value)

    torch.testing.assert_close(
        skip.tensor,
        torch.full_like(
            skip.tensor,
            2.0,
        ),
    )

    torch.testing.assert_close(
        downsampled.tensor,
        torch.ones_like(
            downsampled.tensor,
        ),
    )


@pytest.mark.pruned
def test_down_block_skips_mask_pooling_when_mask_is_none(
    monkeypatch,
):
    block = make_down_block()

    block._block = IdentityTensorMaskModule()

    mask_pool = Mock(side_effect=AssertionError("Mask pool should not be called."))

    monkeypatch.setattr(
        block.mask_pool,
        "forward",
        mask_pool,
    )

    downsampled, skip = block(
        make_tensor_mask(
            channels=3,
            with_mask=False,
        )
    )

    assert downsampled.mask is None
    assert skip.mask is None
    mask_pool.assert_not_called()


@pytest.mark.parametrize(
    ("method", "mask", "expected"),
    [
        (
            "any",
            torch.tensor(
                [
                    [
                        [
                            [0.0, 0.0],
                            [0.0, 1.0],
                        ]
                    ]
                ]
            ),
            1.0,
        ),
        (
            "all",
            torch.tensor(
                [
                    [
                        [
                            [1.0, 1.0],
                            [1.0, 0.0],
                        ]
                    ]
                ]
            ),
            0.0,
        ),
        (
            "fraction",
            torch.tensor(
                [
                    [
                        [
                            [1.0, 1.0],
                            [1.0, 0.0],
                        ]
                    ]
                ]
            ),
            1.0,
        ),
    ],
)
def test_down_block_pools_mask(
    method,
    mask,
    expected,
):
    block = DownBlock(
        in_channels=1,
        out_channels=1,
        block_config=make_conv_config(),
        mask_pooling=method,
        mask_fraction_threshold=0.5,
        return_skip=True,
    )

    block._block = IdentityTensorMaskModule()

    downsampled, skip = block(
        TensorMask(
            tensor=torch.ones(1, 1, 2, 2),
            mask=mask,
        )
    )

    assert downsampled.mask.item() == pytest.approx(expected)
    assert skip.mask is mask


@pytest.mark.pruned
def test_down_block_uses_max_pool_for_tensor():
    block = DownBlock(
        in_channels=1,
        out_channels=1,
        block_config=make_conv_config(),
        mask_pooling="any",
        mask_fraction_threshold=0.5,
        return_skip=True,
    )

    block._block = IdentityTensorMaskModule()

    tensor = torch.tensor(
        [
            [
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ]
            ]
        ]
    )

    downsampled, skip = block(
        TensorMask(
            tensor=tensor,
            mask=None,
        )
    )

    assert skip.tensor is tensor
    assert downsampled.tensor.item() == pytest.approx(4.0)


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