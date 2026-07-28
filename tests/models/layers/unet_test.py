import pytest
import torch
import torch.nn as nn

import cccma_ppp.models.layers.unet as module
from cccma_ppp.models.layers.conv import (
    ConvBlock,
    ConvBlockConfig,
    ConvNeXtBlock,
    ConvNeXtBlockConfig,
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


def make_conv_config(**kwargs):
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


def make_partial_config(**kwargs):
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


def make_convnext_config(**kwargs):
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


def make_tensor(
    channels=3,
    height=8,
    width=8,
    batch_size=2,
):
    return torch.randn(
        batch_size,
        channels,
        height,
        width,
    )


def make_mask(
    channels=1,
    height=8,
    width=8,
    batch_size=2,
):
    return torch.ones(
        batch_size,
        channels,
        height,
        width,
    )


# ---------------------------------------------------------------------------
# build_conv_block
# ---------------------------------------------------------------------------


def test_build_conv_block_standard():
    config = make_conv_config()

    block = build_conv_block(
        3,
        5,
        config,
    )

    assert isinstance(block, ConvBlock)
    assert block.out_channels == 5


def test_build_conv_block_partial():
    config = make_partial_config()

    block = build_conv_block(
        3,
        5,
        config,
    )

    assert isinstance(block, PartialConvBlock)
    assert block.out_channels == 5


def test_build_conv_block_convnext():
    config = make_convnext_config()

    block = build_conv_block(
        3,
        5,
        config,
    )

    assert isinstance(block, ConvNeXtBlock)
    assert block.out_channels == 5


@pytest.mark.parametrize(
    (
        "config_factory",
        "expected_type",
    ),
    [
        (make_conv_config, ConvBlock),
        (make_partial_config, PartialConvBlock),
        (make_convnext_config, ConvNeXtBlock),
    ],
)
def test_build_conv_block_all_supported_types(
    config_factory,
    expected_type,
):
    block = build_conv_block(
        2,
        4,
        config_factory(),
    )

    assert isinstance(block, expected_type)


def test_build_conv_block_does_not_mutate_original_config():
    config = make_conv_config()

    block = build_conv_block(
        3,
        5,
        config,
        latent_size=8,
        inject_noise=True,
    )

    assert isinstance(block, ConvBlock)
    assert config.latent_size is None
    assert config.inject_noise is False


def test_build_conv_block_passes_generative_settings():
    config = make_conv_config()

    block = build_conv_block(
        3,
        5,
        config,
        latent_size=8,
        inject_noise=True,
    )

    assert block.stages[0].inject_noise is True
    assert block.stages[0].conv.in_channels == 4


def test_build_conv_block_partial_passes_noise_setting():
    config = make_partial_config()

    block = build_conv_block(
        3,
        5,
        config,
        inject_noise=True,
    )

    assert block.stages[0].inject_noise is True
    assert block.stages[0].conv.in_channels == 4


def test_build_conv_block_convnext_passes_noise_setting():
    config = make_convnext_config()

    block = build_conv_block(
        3,
        5,
        config,
        inject_noise=True,
    )

    assert block.blocks[0].inject_noise is True


def test_build_conv_block_rejects_unsupported_config():
    class UnsupportedConfig:
        latent_size = None
        inject_noise = False

        def setup_generative(
            self,
            latent_size=None,
            inject_noise=False,
        ):
            self.latent_size = latent_size
            self.inject_noise = inject_noise
            return self

    with pytest.raises(
        TypeError,
        match="Unsupported UNet block configuration",
    ):
        build_conv_block(
            3,
            5,
            UnsupportedConfig(),
        )


def test_build_conv_block_error_reports_config_name():
    class UnsupportedConfig:
        def setup_generative(
            self,
            latent_size=None,
            inject_noise=False,
        ):
            return self

    with pytest.raises(
        TypeError,
        match="UnsupportedConfig",
    ):
        build_conv_block(
            3,
            5,
            UnsupportedConfig(),
        )


# ---------------------------------------------------------------------------
# DownBlock initialization
# ---------------------------------------------------------------------------


def test_down_block_without_skip_processor():
    block = DownBlock(
        3,
        5,
        block_config=make_conv_config(),
        process_skip=False,
        mask_pooling="any",
        mask_fraction_threshold=0.5,
    )

    assert isinstance(block._block, ConvBlock)
    assert block.skip_processor is None
    assert isinstance(block.tensor_pool, nn.MaxPool2d)


def test_down_block_with_skip_processor():
    block = DownBlock(
        3,
        5,
        block_config=make_conv_config(),
        process_skip=True,
        mask_pooling="any",
        mask_fraction_threshold=0.5,
    )

    assert isinstance(block._block, ConvBlock)
    assert isinstance(block.skip_processor, ConvBlock)


def test_down_block_skip_processor_uses_output_channels():
    block = DownBlock(
        3,
        5,
        block_config=make_conv_config(),
        process_skip=True,
        mask_pooling="any",
        mask_fraction_threshold=0.5,
    )

    assert block.skip_processor.stages[0].conv.in_channels == 5
    assert block.skip_processor.stages[0].conv.out_channels == 5


@pytest.mark.parametrize(
    "method",
    [
        "any",
        "all",
        "fraction",
    ],
)
def test_down_block_passes_mask_pooling_method(method):
    block = DownBlock(
        3,
        5,
        block_config=make_conv_config(),
        process_skip=False,
        mask_pooling=method,
        mask_fraction_threshold=0.75,
    )

    assert block.mask_pool.method == method
    assert block.mask_pool.fraction_threshold == pytest.approx(0.75)


def test_down_block_rejects_invalid_fraction_threshold():
    with pytest.raises(
        ValueError,
        match="fraction_threshold must be between 0 and 1",
    ):
        DownBlock(
            3,
            5,
            block_config=make_conv_config(),
            process_skip=False,
            mask_pooling="fraction",
            mask_fraction_threshold=1.5,
        )


# ---------------------------------------------------------------------------
# DownBlock forward
# ---------------------------------------------------------------------------


def test_down_block_forward_without_mask():
    block = DownBlock(
        3,
        5,
        block_config=make_conv_config(),
        process_skip=False,
        mask_pooling="any",
        mask_fraction_threshold=0.5,
    )

    downsampled, skip = block(
        TensorMask(
            tensor=make_tensor(channels=3),
            mask=None,
        )
    )

    assert skip.tensor.shape == (2, 5, 8, 8)
    assert skip.mask is None
    assert downsampled.tensor.shape == (2, 5, 4, 4)
    assert downsampled.mask is None


def test_down_block_processes_skip_when_enabled(
    monkeypatch,
):
    block = DownBlock(
        3,
        5,
        block_config=make_conv_config(),
        process_skip=True,
        mask_pooling="any",
        mask_fraction_threshold=0.5,
    )

    calls = []

    class FakeBlock(nn.Module):
        def forward(self, value):
            calls.append(value)
            return TensorMask(
                tensor=value.tensor + 1,
                mask=value.mask,
            )

    block._block = FakeBlock()
    block.skip_processor = FakeBlock()

    input_value = TensorMask(
        tensor=torch.zeros(2, 3, 8, 8),
        mask=None,
    )

    downsampled, skip = block(input_value)

    assert len(calls) == 2
    torch.testing.assert_close(
        skip.tensor,
        torch.full_like(skip.tensor, 2.0),
    )
    torch.testing.assert_close(
        downsampled.tensor,
        torch.full_like(downsampled.tensor, 2.0),
    )


def test_down_block_does_not_process_skip_when_disabled():
    block = DownBlock(
        3,
        5,
        block_config=make_conv_config(),
        process_skip=False,
        mask_pooling="any",
        mask_fraction_threshold=0.5,
    )

    assert block.skip_processor is None


def test_down_block_applies_max_pool_to_tensor():
    block = DownBlock(
        1,
        1,
        block_config=make_conv_config(
            kernel_size=1,
            bias=False,
        ),
        process_skip=False,
        mask_pooling="any",
        mask_fraction_threshold=0.5,
    )

    block._block = nn.Identity()

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


def test_down_block_pools_mask_when_present():
    block = DownBlock(
        1,
        1,
        block_config=make_conv_config(),
        process_skip=False,
        mask_pooling="any",
        mask_fraction_threshold=0.5,
    )

    block._block = nn.Identity()

    mask = torch.tensor(
        [
            [
                [
                    [0.0, 0.0],
                    [0.0, 1.0],
                ]
            ]
        ]
    )

    downsampled, _ = block(
        TensorMask(
            tensor=torch.ones(1, 1, 2, 2),
            mask=mask,
        )
    )

    assert downsampled.mask.item() == pytest.approx(1.0)


def test_down_block_skips_mask_pooling_when_mask_is_none(
    monkeypatch,
):
    block = DownBlock(
        1,
        1,
        block_config=make_conv_config(),
        process_skip=False,
        mask_pooling="any",
        mask_fraction_threshold=0.5,
    )

    block._block = nn.Identity()

    def fail_pool(mask):
        raise AssertionError("Mask pooling should not be called.")

    monkeypatch.setattr(
        block.mask_pool,
        "forward",
        fail_pool,
    )

    downsampled, skip = block(
        TensorMask(
            tensor=torch.ones(1, 1, 4, 4),
            mask=None,
        )
    )

    assert downsampled.mask is None
    assert skip.mask is None


# ---------------------------------------------------------------------------
# UpBlock initialization
# ---------------------------------------------------------------------------


def test_up_block_transpose_convolution():
    block = UpBlock(
        input_channels=4,
        skip_channels=3,
        out_channels=2,
        block_config=make_conv_config(),
        upsampling_method="transpose_conv",
        skip_alignment_method="strict",
        transpose_kernel_size=2,
    )

    assert isinstance(block.upsample, nn.ConvTranspose2d)
    assert isinstance(block.channel_projection, nn.Identity)
    assert block.upsample.in_channels == 4
    assert block.upsample.out_channels == 2
    assert block.upsample.kernel_size == (2, 2)
    assert block.upsample.stride == (2, 2)


def test_up_block_bilinear_standard_projection():
    block = UpBlock(
        input_channels=4,
        skip_channels=3,
        out_channels=2,
        block_config=make_conv_config(),
        upsampling_method="bilinear",
        skip_alignment_method="strict",
        transpose_kernel_size=2,
    )

    assert isinstance(block.upsample, nn.Upsample)
    assert isinstance(block.channel_projection, nn.Conv2d)
    assert not isinstance(block.channel_projection, PartialConv2d)
    assert block.channel_projection.in_channels == 4
    assert block.channel_projection.out_channels == 2


def test_up_block_bilinear_partial_projection():
    block = UpBlock(
        input_channels=4,
        skip_channels=3,
        out_channels=2,
        block_config=make_partial_config(),
        upsampling_method="bilinear",
        skip_alignment_method="strict",
        transpose_kernel_size=2,
    )

    assert isinstance(block.channel_projection, PartialConv2d)
    assert block.channel_projection.multi_channel is False
    assert block.channel_projection.return_mask is False


def test_up_block_bilinear_convnext_partial_projection():
    block = UpBlock(
        input_channels=4,
        skip_channels=3,
        out_channels=2,
        block_config=make_convnext_config(
            use_partial_conv=True,
        ),
        upsampling_method="bilinear",
        skip_alignment_method="strict",
        transpose_kernel_size=2,
    )

    assert isinstance(block.channel_projection, PartialConv2d)


def test_up_block_bilinear_convnext_standard_projection():
    block = UpBlock(
        input_channels=4,
        skip_channels=3,
        out_channels=2,
        block_config=make_convnext_config(
            use_partial_conv=False,
        ),
        upsampling_method="bilinear",
        skip_alignment_method="strict",
        transpose_kernel_size=2,
    )

    assert isinstance(block.channel_projection, nn.Conv2d)
    assert not isinstance(block.channel_projection, PartialConv2d)


def test_up_block_rejects_unsupported_upsampling_method():
    with pytest.raises(
        ValueError,
        match="Unsupported upsampling mode",
    ):
        UpBlock(
            input_channels=4,
            skip_channels=3,
            out_channels=2,
            block_config=make_conv_config(),
            upsampling_method="nearest",
            skip_alignment_method="strict",
            transpose_kernel_size=2,
        )


def test_up_block_stores_configuration():
    block = UpBlock(
        input_channels=4,
        skip_channels=3,
        out_channels=2,
        block_config=make_conv_config(
            padding_method="reflect",
        ),
        upsampling_method="bilinear",
        skip_alignment_method="interpolation",
        transpose_kernel_size=2,
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


def test_up_block_noise_adds_transpose_input_channel():
    block = UpBlock(
        input_channels=4,
        skip_channels=3,
        out_channels=2,
        block_config=make_conv_config(),
        upsampling_method="transpose_conv",
        skip_alignment_method="strict",
        transpose_kernel_size=2,
        inject_noise=True,
    )

    assert block.upsample.in_channels == 5


def test_up_block_noise_adds_bilinear_projection_channel():
    block = UpBlock(
        input_channels=4,
        skip_channels=3,
        out_channels=2,
        block_config=make_conv_config(),
        upsampling_method="bilinear",
        skip_alignment_method="strict",
        transpose_kernel_size=2,
        inject_noise=True,
    )

    assert block.channel_projection.in_channels == 5


def test_up_block_merged_block_input_channels():
    block = UpBlock(
        input_channels=4,
        skip_channels=3,
        out_channels=2,
        block_config=make_conv_config(),
        upsampling_method="bilinear",
        skip_alignment_method="strict",
        transpose_kernel_size=2,
    )

    assert block._block.stages[0].conv.in_channels == 5
    assert block._block.stages[0].conv.out_channels == 2


def test_up_block_does_not_mutate_partial_config():
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
    assert block._block.stages[0].multi_channel is False
    assert block._block.stages[0].return_mask is False


def test_up_block_disables_partial_mask_return_in_merged_block():
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

    assert block._block.stages[0].multi_channel is False
    assert block._block.stages[0].return_mask is False


def test_up_block_noise_in_block_enabled_only_when_both_flags_true():
    block = UpBlock(
        input_channels=4,
        skip_channels=3,
        out_channels=2,
        block_config=make_conv_config(),
        upsampling_method="bilinear",
        skip_alignment_method="strict",
        transpose_kernel_size=2,
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
        (False, False),
        (False, True),
        (True, False),
    ],
)
def test_up_block_noise_in_block_disabled_unless_both_flags_true(
    inject_noise,
    inject_noise_in_block,
):
    block = UpBlock(
        input_channels=4,
        skip_channels=3,
        out_channels=2,
        block_config=make_conv_config(),
        upsampling_method="bilinear",
        skip_alignment_method="strict",
        transpose_kernel_size=2,
        inject_noise=inject_noise,
        inject_noise_in_block=inject_noise_in_block,
    )

    assert block._block.stages[0].inject_noise is False


# ---------------------------------------------------------------------------
# UpBlock forward
# ---------------------------------------------------------------------------


def test_up_block_transpose_forward_shape():
    block = UpBlock(
        input_channels=4,
        skip_channels=3,
        out_channels=2,
        block_config=make_conv_config(),
        upsampling_method="transpose_conv",
        skip_alignment_method="strict",
        transpose_kernel_size=2,
    )

    result = block(
        TensorMask(
            tensor=make_tensor(
                channels=4,
                height=4,
                width=4,
            ),
            mask=None,
        ),
        TensorMask(
            tensor=make_tensor(
                channels=3,
                height=8,
                width=8,
            ),
            mask=make_mask(
                height=8,
                width=8,
            ),
        ),
    )

    assert result.tensor.shape == (2, 2, 8, 8)
    assert result.mask is None


def test_up_block_bilinear_forward_shape():
    block = UpBlock(
        input_channels=4,
        skip_channels=3,
        out_channels=2,
        block_config=make_conv_config(),
        upsampling_method="bilinear",
        skip_alignment_method="strict",
        transpose_kernel_size=2,
    )

    result = block(
        TensorMask(
            tensor=make_tensor(
                channels=4,
                height=4,
                width=4,
            ),
            mask=None,
        ),
        TensorMask(
            tensor=make_tensor(
                channels=3,
                height=8,
                width=8,
            ),
            mask=None,
        ),
    )

    assert result.tensor.shape == (2, 2, 8, 8)
    assert result.mask is None


def test_up_block_ignores_input_and_skip_masks():
    block = UpBlock(
        input_channels=4,
        skip_channels=3,
        out_channels=2,
        block_config=make_conv_config(),
        upsampling_method="bilinear",
        skip_alignment_method="strict",
        transpose_kernel_size=2,
    )

    result = block(
        TensorMask(
            tensor=make_tensor(
                channels=4,
                height=4,
                width=4,
            ),
            mask=make_mask(
                channels=4,
                height=4,
                width=4,
            ),
        ),
        TensorMask(
            tensor=make_tensor(
                channels=3,
                height=8,
                width=8,
            ),
            mask=make_mask(
                channels=3,
                height=8,
                width=8,
            ),
        ),
    )

    assert result.mask is None


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

    class CaptureBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.value = None

        def forward(self, value):
            self.value = value
            return value

    capture = CaptureBlock()
    block._block = capture
    block.channel_projection = nn.Identity()

    input_tensor = torch.full(
        (1, 1, 2, 2),
        2.0,
    )
    skip_tensor = torch.full(
        (1, 1, 4, 4),
        1.0,
    )

    result = block(
        TensorMask(
            tensor=input_tensor,
            mask=None,
        ),
        TensorMask(
            tensor=skip_tensor,
            mask=None,
        ),
    )

    assert capture.value.tensor.shape == (1, 2, 4, 4)
    torch.testing.assert_close(
        capture.value.tensor[:, :1],
        skip_tensor,
    )
    torch.testing.assert_close(
        capture.value.tensor[:, 1:],
        torch.full_like(skip_tensor, 2.0),
    )
    assert result is capture.value


def test_up_block_calls_align_to_skip(
    monkeypatch,
):
    block = UpBlock(
        input_channels=4,
        skip_channels=3,
        out_channels=2,
        block_config=make_conv_config(
            padding_method="reflect",
        ),
        upsampling_method="bilinear",
        skip_alignment_method="interpolation",
        transpose_kernel_size=2,
    )

    calls = []

    def fake_align(
        tensor,
        skip,
        method,
        padding_method,
    ):
        calls.append(
            (
                tensor,
                skip,
                method,
                padding_method,
            )
        )
        return torch.zeros(
            skip.shape[0],
            2,
            skip.shape[2],
            skip.shape[3],
        )

    monkeypatch.setattr(
        module,
        "align_to_skip",
        fake_align,
    )

    input_value = TensorMask(
        tensor=make_tensor(
            channels=4,
            height=4,
            width=4,
        ),
        mask=None,
    )
    skip_value = TensorMask(
        tensor=make_tensor(
            channels=3,
            height=9,
            width=9,
        ),
        mask=None,
    )

    result = block(
        input_value,
        skip_value,
    )

    assert len(calls) == 1
    assert calls[0][1] is skip_value.tensor
    assert calls[0][2] == "interpolation"
    assert calls[0][3] == "reflect"
    assert result.tensor.shape == (2, 2, 9, 9)


def test_up_block_transpose_noise_injected_before_upsampling(
    monkeypatch,
):
    block = UpBlock(
        input_channels=4,
        skip_channels=3,
        out_channels=2,
        block_config=make_conv_config(),
        upsampling_method="transpose_conv",
        skip_alignment_method="strict",
        transpose_kernel_size=2,
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
        TensorMask(
            tensor=make_tensor(
                channels=4,
                height=4,
                width=4,
            ),
            mask=None,
        ),
        TensorMask(
            tensor=make_tensor(
                channels=3,
                height=8,
                width=8,
            ),
            mask=None,
        ),
    )

    assert calls == [
        torch.Size([2, 4, 4, 4]),
    ]
    assert result.tensor.shape == (2, 2, 8, 8)


def test_up_block_bilinear_noise_injected_after_upsampling(
    monkeypatch,
):
    block = UpBlock(
        input_channels=4,
        skip_channels=3,
        out_channels=2,
        block_config=make_conv_config(),
        upsampling_method="bilinear",
        skip_alignment_method="strict",
        transpose_kernel_size=2,
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
        TensorMask(
            tensor=make_tensor(
                channels=4,
                height=4,
                width=4,
            ),
            mask=None,
        ),
        TensorMask(
            tensor=make_tensor(
                channels=3,
                height=8,
                width=8,
            ),
            mask=None,
        ),
    )

    assert calls == [
        torch.Size([2, 4, 8, 8]),
    ]
    assert result.tensor.shape == (2, 2, 8, 8)


def test_up_block_without_noise_does_not_call_noise_injection(
    monkeypatch,
):
    block = UpBlock(
        input_channels=4,
        skip_channels=3,
        out_channels=2,
        block_config=make_conv_config(),
        upsampling_method="bilinear",
        skip_alignment_method="strict",
        transpose_kernel_size=2,
        inject_noise=False,
    )

    def fail_noise(tensor):
        raise AssertionError("Noise injection should not be called.")

    monkeypatch.setattr(
        module,
        "_noise_injection",
        fail_noise,
    )

    result = block(
        TensorMask(
            tensor=make_tensor(
                channels=4,
                height=4,
                width=4,
            ),
            mask=None,
        ),
        TensorMask(
            tensor=make_tensor(
                channels=3,
                height=8,
                width=8,
            ),
            mask=None,
        ),
    )

    assert result.tensor.shape == (2, 2, 8, 8)


# ---------------------------------------------------------------------------
# UNetOutput initialization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "activation",
    [
        "identity",
        "sigmoid",
        "tanh",
    ],
)
def test_unet_output_direct_projection(activation):
    layer = UNetOutput(
        4,
        2,
        hidden_channels=None,
        activation=activation,
    )

    assert isinstance(layer.layers[0], nn.Conv2d)
    assert layer.layers[0].in_channels == 4
    assert layer.layers[0].out_channels == 2
    assert layer.layers[0].kernel_size == (1, 1)


def test_unet_output_identity_has_only_projection():
    layer = UNetOutput(
        4,
        2,
        hidden_channels=None,
        activation="identity",
    )

    assert len(layer.layers) == 1
    assert isinstance(layer.layers[0], nn.Conv2d)


def test_unet_output_sigmoid_activation():
    layer = UNetOutput(
        4,
        2,
        hidden_channels=None,
        activation="sigmoid",
    )

    assert len(layer.layers) == 2
    assert isinstance(layer.layers[-1], nn.Sigmoid)


def test_unet_output_tanh_activation():
    layer = UNetOutput(
        4,
        2,
        hidden_channels=None,
        activation="tanh",
    )

    assert len(layer.layers) == 2
    assert isinstance(layer.layers[-1], nn.Tanh)


def test_unet_output_rejects_unsupported_activation():
    with pytest.raises(
        ValueError,
        match="Unsupported output activation",
    ):
        UNetOutput(
            4,
            2,
            hidden_channels=None,
            activation="relu",
        )


def test_unet_output_hidden_structure():
    layer = UNetOutput(
        4,
        2,
        hidden_channels=6,
        activation="identity",
    )

    assert len(layer.layers) == 4

    assert isinstance(layer.layers[0], PartialConv2d)
    assert layer.layers[0].in_channels == 4
    assert layer.layers[0].out_channels == 6
    assert layer.layers[0].kernel_size == (3, 3)
    assert layer.layers[0].padding == (1, 1)
    assert layer.layers[0].bias is None
    assert layer.layers[0].multi_channel is False
    assert layer.layers[0].return_mask is False

    assert isinstance(layer.layers[1], nn.BatchNorm2d)
    assert layer.layers[1].num_features == 6

    assert isinstance(layer.layers[2], nn.ReLU)
    assert layer.layers[2].inplace is True

    assert isinstance(layer.layers[3], nn.Conv2d)
    assert layer.layers[3].in_channels == 6
    assert layer.layers[3].out_channels == 2


def test_unet_output_hidden_sigmoid_structure():
    layer = UNetOutput(
        4,
        2,
        hidden_channels=6,
        activation="sigmoid",
    )

    assert len(layer.layers) == 5
    assert isinstance(layer.layers[-1], nn.Sigmoid)


def test_unet_output_hidden_tanh_structure():
    layer = UNetOutput(
        4,
        2,
        hidden_channels=6,
        activation="tanh",
    )

    assert len(layer.layers) == 5
    assert isinstance(layer.layers[-1], nn.Tanh)


# ---------------------------------------------------------------------------
# UNetOutput forward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    (
        "hidden_channels",
        "activation",
    ),
    [
        (None, "identity"),
        (None, "sigmoid"),
        (None, "tanh"),
        (6, "identity"),
        (6, "sigmoid"),
        (6, "tanh"),
    ],
)
def test_unet_output_forward_shape(
    hidden_channels,
    activation,
):
    layer = UNetOutput(
        4,
        2,
        hidden_channels=hidden_channels,
        activation=activation,
    )

    result = layer(make_tensor(channels=4))

    assert result.shape == (2, 2, 8, 8)


def test_unet_output_sigmoid_range():
    layer = UNetOutput(
        4,
        2,
        hidden_channels=None,
        activation="sigmoid",
    )

    result = layer(make_tensor(channels=4))

    assert torch.all(result >= 0)
    assert torch.all(result <= 1)


def test_unet_output_tanh_range():
    layer = UNetOutput(
        4,
        2,
        hidden_channels=None,
        activation="tanh",
    )

    result = layer(make_tensor(channels=4))

    assert torch.all(result >= -1)
    assert torch.all(result <= 1)


def test_unet_output_identity_preserves_negative_values():
    layer = UNetOutput(
        1,
        1,
        hidden_channels=None,
        activation="identity",
    )

    with torch.no_grad():
        layer.layers[0].weight.fill_(-1.0)
        layer.layers[0].bias.zero_()

    result = layer(torch.ones(1, 1, 2, 2))

    torch.testing.assert_close(
        result,
        -torch.ones_like(result),
    )


def test_unet_output_sigmoid_exact_zero_input():
    layer = UNetOutput(
        1,
        1,
        hidden_channels=None,
        activation="sigmoid",
    )

    with torch.no_grad():
        layer.layers[0].weight.zero_()
        layer.layers[0].bias.zero_()

    result = layer(torch.ones(1, 1, 2, 2))

    torch.testing.assert_close(
        result,
        torch.full_like(result, 0.5),
    )


def test_unet_output_tanh_exact_zero_input():
    layer = UNetOutput(
        1,
        1,
        hidden_channels=None,
        activation="tanh",
    )

    with torch.no_grad():
        layer.layers[0].weight.zero_()
        layer.layers[0].bias.zero_()

    result = layer(torch.ones(1, 1, 2, 2))

    torch.testing.assert_close(
        result,
        torch.zeros_like(result),
    )


def test_unet_output_supports_backward():
    layer = UNetOutput(
        4,
        2,
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


def test_unet_output_preserves_float64():
    layer = UNetOutput(
        4,
        2,
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
