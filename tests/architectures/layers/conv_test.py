from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

import cccma_ppp.architectures.layers.conv as module
from cccma_ppp.architectures.layers.conv import (
    ConvBlock,
    ConvBlockConfig,
    ConvNeXtBlock,
    ConvNeXtBlockConfig,
    ConvNeXtSingle,
    ConvSingle,
    LatentBlock,
    LatentLayer,
    LatentVector,
    MaskPool2d,
    PartialConvBlock,
    PartialConvBlockConfig,
    PartialConvSingle,
    TensorMask,
)


def make_conv_config(**kwargs):
    defaults = {
        "name": "standard_conv",
        "num_convolutions": 2,
        "kernel_size": 3,
        "normalization": "batch",
        "padding_method": "circular",
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
        "num_convolutions": 2,
        "kernel_size": 3,
        "normalization": "batch",
        "padding_method": "circular",
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
        "num_blocks": 2,
        "kernel_size": 3,
        "expansion_ratio": 2,
        "padding_method": "circular",
        "layer_scale_init": 1e-6,
        "dropout_rate": 0.0,
        "drop_path_rate": 0.0,
        "use_partial_conv": True,
    }
    defaults.update(kwargs)
    return ConvNeXtBlockConfig(**defaults)


def make_tensor(
    batch_size=2,
    channels=3,
    height=8,
    width=8,
):
    return torch.randn(
        batch_size,
        channels,
        height,
        width,
    )


def make_mask(
    batch_size=2,
    channels=1,
    height=8,
    width=8,
):
    return torch.ones(
        batch_size,
        channels,
        height,
        width,
    )


@pytest.mark.parametrize(
    "num_convolutions",
    [
        0,
        -1,
        -10,
    ],
)
def test_conv_config_rejects_invalid_num_convolutions(
    num_convolutions,
):
    with pytest.raises(
        ValueError,
        match="num_convolutions must be at least 1",
    ):
        make_conv_config(
            num_convolutions=num_convolutions,
        )


@pytest.mark.pruned
@pytest.mark.parametrize(
    "kernel_size",
    [
        2,
        4,
        6,
    ],
)
def test_conv_config_rejects_even_kernel_size(
    kernel_size,
):
    with pytest.raises(ValueError):
        make_conv_config(
            kernel_size=kernel_size,
        )


@pytest.mark.pruned
@pytest.mark.parametrize(
    "dropout_rate",
    [
        -0.1,
        1.1,
        2.0,
    ],
)
def test_conv_config_rejects_invalid_dropout(
    dropout_rate,
):
    with pytest.raises(ValueError):
        make_conv_config(
            dropout_rate=dropout_rate,
        )


@pytest.mark.pruned
@pytest.mark.parametrize(
    "dropout_rate",
    [
        None,
        0.0,
        0.25,
        1.0,
    ],
)
def test_conv_config_accepts_valid_dropout(
    dropout_rate,
):
    config = make_conv_config(
        dropout_rate=dropout_rate,
    )

    assert config.dropout_rate == dropout_rate


@pytest.mark.parametrize(
    "num_convolutions",
    [
        0,
        -1,
        -10,
    ],
)
def test_partial_conv_config_rejects_invalid_num_convolutions(
    num_convolutions,
):
    with pytest.raises(
        ValueError,
        match="num_convolutions must be at least 1",
    ):
        make_partial_config(
            num_convolutions=num_convolutions,
        )


@pytest.mark.pruned
@pytest.mark.parametrize(
    "kernel_size",
    [
        2,
        4,
        8,
    ],
)
def test_partial_conv_config_rejects_even_kernel_size(
    kernel_size,
):
    with pytest.raises(ValueError):
        make_partial_config(
            kernel_size=kernel_size,
        )


@pytest.mark.pruned
@pytest.mark.parametrize(
    "dropout_rate",
    [
        -0.1,
        1.1,
        10.0,
    ],
)
def test_partial_conv_config_rejects_invalid_dropout(
    dropout_rate,
):
    with pytest.raises(ValueError):
        make_partial_config(
            dropout_rate=dropout_rate,
        )


@pytest.mark.pruned
def test_convnext_config_defaults():
    config = ConvNeXtBlockConfig(name="convnext")

    assert config.name == "convnext"
    assert config.num_blocks == 2
    assert config.kernel_size == 7
    assert config.expansion_ratio == 4
    assert config.padding_method == "zeros"
    assert config.layer_scale_init == pytest.approx(1e-6)
    assert config.dropout_rate == 0.0
    assert config.drop_path_rate == 0.0
    assert config.use_partial_conv is True
    assert config.multi_channel is True
    assert config.return_mask is True


@pytest.mark.parametrize(
    "num_blocks",
    [
        0,
        -1,
        -5,
    ],
)
def test_convnext_config_rejects_invalid_num_blocks(
    num_blocks,
):
    with pytest.raises(
        ValueError,
        match="num_blocks must be at least 1",
    ):
        make_convnext_config(
            num_blocks=num_blocks,
        )


@pytest.mark.pruned
@pytest.mark.parametrize(
    "expansion_ratio",
    [
        0,
        -1,
        -10,
    ],
)
def test_convnext_config_rejects_invalid_expansion_ratio(
    expansion_ratio,
):
    with pytest.raises(
        ValueError,
        match="expansion_ratio must be at least 1",
    ):
        make_convnext_config(
            expansion_ratio=expansion_ratio,
        )


@pytest.mark.pruned
@pytest.mark.parametrize(
    "kernel_size",
    [
        2,
        4,
        6,
    ],
)
def test_convnext_config_rejects_even_kernel_size(
    kernel_size,
):
    with pytest.raises(ValueError):
        make_convnext_config(
            kernel_size=kernel_size,
        )


@pytest.mark.pruned
@pytest.mark.parametrize(
    "dropout_rate",
    [
        -0.1,
        1.1,
        2.0,
    ],
)
def test_convnext_config_rejects_invalid_dropout(
    dropout_rate,
):
    with pytest.raises(ValueError):
        make_convnext_config(
            dropout_rate=dropout_rate,
        )


@pytest.mark.pruned
@pytest.mark.parametrize(
    "drop_path_rate",
    [
        -0.1,
        1.1,
        2.0,
    ],
)
def test_convnext_config_rejects_invalid_drop_path_rate(
    drop_path_rate,
):
    with pytest.raises(ValueError):
        make_convnext_config(
            drop_path_rate=drop_path_rate,
        )


@pytest.mark.pruned
def test_conv_single_forward_shape():
    config = make_conv_config(
        normalization="batch",
        activation="relu",
    )
    layer = ConvSingle(
        in_channels=3,
        out_channels=5,
        config=config,
    )

    x = make_tensor(channels=3)
    result = layer(x)

    assert result.shape == (2, 5, 8, 8)


@pytest.mark.pruned
def test_conv_single_without_dropout_uses_identity():
    layer = ConvSingle(
        in_channels=3,
        out_channels=4,
        config=make_conv_config(
            dropout_rate=None,
        ),
    )

    assert isinstance(layer.dropout, nn.Identity)


@pytest.mark.pruned
def test_conv_single_zero_dropout_uses_identity():
    layer = ConvSingle(
        in_channels=3,
        out_channels=4,
        config=make_conv_config(
            dropout_rate=0.0,
        ),
    )

    assert isinstance(layer.dropout, nn.Identity)


@pytest.mark.pruned
def test_conv_single_positive_dropout_uses_dropout2d():
    layer = ConvSingle(
        in_channels=3,
        out_channels=4,
        config=make_conv_config(
            dropout_rate=0.5,
        ),
    )

    assert isinstance(layer.dropout, nn.Dropout2d)


@pytest.mark.pruned
def test_conv_single_uses_requested_bias():
    layer = ConvSingle(
        in_channels=3,
        out_channels=4,
        config=make_conv_config(
            bias=True,
        ),
    )

    assert layer.conv.bias is not None


@pytest.mark.pruned
def test_conv_single_without_bias():
    layer = ConvSingle(
        in_channels=3,
        out_channels=4,
        config=make_conv_config(
            bias=False,
        ),
    )

    assert layer.conv.bias is None


@pytest.mark.pruned
def test_conv_single_without_noise_does_not_call_noise_injection(
    monkeypatch,
):
    layer = ConvSingle(
        in_channels=3,
        out_channels=4,
        config=make_conv_config(),
    )

    def fail_noise_injection(x):
        raise AssertionError("Noise injection should not be called.")

    monkeypatch.setattr(
        module,
        "_noise_injection",
        fail_noise_injection,
    )

    result = layer(make_tensor(channels=3))

    assert result.shape == (2, 4, 8, 8)


@pytest.mark.pruned
def test_partial_conv_single_forward_with_mask():
    layer = PartialConvSingle(
        in_channels=3,
        out_channels=4,
        config=make_partial_config(),
    )

    x = make_tensor(channels=3)
    mask = make_mask(channels=3)

    result, result_mask = layer(
        x,
        mask,
    )

    assert result.shape == (2, 4, 8, 8)
    assert result_mask is not None
    assert result_mask.shape == result.shape


@pytest.mark.pruned
def test_partial_conv_single_forward_without_mask():
    layer = PartialConvSingle(
        in_channels=3,
        out_channels=4,
        config=make_partial_config(),
    )

    result, result_mask = layer(
        make_tensor(channels=3),
        None,
    )

    assert result.shape == (2, 4, 8, 8)
    assert result_mask is not None


@pytest.mark.pruned
def test_partial_conv_single_without_dropout_uses_identity():
    layer = PartialConvSingle(
        in_channels=3,
        out_channels=4,
        config=make_partial_config(
            dropout_rate=None,
        ),
    )

    assert isinstance(layer.dropout, nn.Identity)


@pytest.mark.pruned
def test_partial_conv_single_positive_dropout_uses_dropout2d():
    layer = PartialConvSingle(
        in_channels=3,
        out_channels=4,
        config=make_partial_config(
            dropout_rate=0.4,
        ),
    )

    assert isinstance(layer.dropout, nn.Dropout2d)


@pytest.mark.pruned
def test_conv_block_builds_requested_number_of_stages():
    block = ConvBlock(
        in_channels=3,
        out_channels=5,
        config=make_conv_config(
            num_convolutions=4,
        ),
    )

    assert len(block.stages) == 4
    assert block.out_channels == 5


@pytest.mark.pruned
def test_conv_block_first_stage_uses_input_channels():
    block = ConvBlock(
        in_channels=3,
        out_channels=5,
        config=make_conv_config(
            num_convolutions=3,
        ),
    )

    assert block.stages[0].conv.in_channels == 3
    assert block.stages[0].conv.out_channels == 5


@pytest.mark.pruned
def test_conv_block_later_stages_use_output_channels():
    block = ConvBlock(
        in_channels=3,
        out_channels=5,
        config=make_conv_config(
            num_convolutions=3,
        ),
    )

    assert block.stages[1].conv.in_channels == 5
    assert block.stages[2].conv.in_channels == 5


@pytest.mark.pruned
def test_conv_block_forward_preserves_mask_identity():
    block = ConvBlock(
        in_channels=3,
        out_channels=5,
        config=make_conv_config(),
    )

    mask = make_mask()
    result = block(
        TensorMask(
            tensor=make_tensor(channels=3),
            mask=mask,
        )
    )

    assert result.tensor.shape == (2, 5, 8, 8)
    assert result.mask is mask


@pytest.mark.pruned
def test_conv_block_forward_with_none_mask():
    block = ConvBlock(
        in_channels=3,
        out_channels=5,
        config=make_conv_config(),
    )

    result = block(
        TensorMask(
            tensor=make_tensor(channels=3),
        )
    )

    assert result.tensor.shape == (2, 5, 8, 8)
    assert result.mask is None


@pytest.mark.pruned
def test_partial_conv_block_builds_requested_number_of_stages():
    block = PartialConvBlock(
        in_channels=3,
        out_channels=5,
        config=make_partial_config(
            num_convolutions=3,
        ),
    )

    assert len(block.stages) == 3
    assert block.out_channels == 5


@pytest.mark.pruned
def test_partial_conv_block_channel_progression():
    block = PartialConvBlock(
        in_channels=3,
        out_channels=5,
        config=make_partial_config(
            num_convolutions=3,
        ),
    )

    assert block.stages[0].conv.in_channels == 3
    assert block.stages[1].conv.in_channels == 5
    assert block.stages[2].conv.in_channels == 5


@pytest.mark.pruned
def test_partial_conv_block_forward_without_mask():
    block = PartialConvBlock(
        in_channels=3,
        out_channels=5,
        config=make_partial_config(),
    )

    result = block(
        TensorMask(
            tensor=make_tensor(channels=3),
            mask=None,
        )
    )

    assert result.tensor.shape == (2, 5, 8, 8)
    assert result.mask is not None


@pytest.mark.pruned
def test_partial_conv_block_calls_broadcast_mask(
    monkeypatch,
):
    block = PartialConvBlock(
        in_channels=3,
        out_channels=5,
        config=make_partial_config(
            num_convolutions=1,
        ),
    )

    tensor = make_tensor(channels=3)
    mask = make_mask(channels=1)
    expanded = torch.ones_like(tensor)
    calls = []

    def fake_broadcast_mask(mask_value, tensor_value):
        calls.append(
            (
                mask_value,
                tensor_value,
            )
        )
        return expanded

    monkeypatch.setattr(
        module,
        "_broadcast_mask",
        fake_broadcast_mask,
    )

    result = block(
        TensorMask(
            tensor=tensor,
            mask=mask,
        )
    )

    assert calls == [
        (
            mask,
            tensor,
        )
    ]
    assert result.tensor.shape == (2, 5, 8, 8)


@pytest.mark.pruned
def test_convnext_single_uses_partial_depthwise_convolution():
    layer = ConvNeXtSingle(
        channels=4,
        config=make_convnext_config(
            use_partial_conv=True,
        ),
        drop_path_rate=0.0,
    )

    assert layer.use_partial_conv is True
    assert isinstance(
        layer.depthwise,
        module.PartialConv2d,
    )


@pytest.mark.pruned
def test_convnext_single_uses_standard_depthwise_convolution():
    layer = ConvNeXtSingle(
        channels=4,
        config=make_convnext_config(
            use_partial_conv=False,
        ),
        drop_path_rate=0.0,
    )

    assert layer.use_partial_conv is False
    assert isinstance(
        layer.depthwise,
        nn.Conv2d,
    )
    assert layer.depthwise.groups == 4


@pytest.mark.pruned
def test_convnext_single_hidden_channel_expansion():
    layer = ConvNeXtSingle(
        channels=4,
        config=make_convnext_config(
            expansion_ratio=3,
        ),
        drop_path_rate=0.0,
    )

    assert layer.pointwise_1.out_channels == 12
    assert layer.pointwise_2.in_channels == 12


@pytest.mark.pruned
def test_convnext_single_forward_partial_conv():
    layer = ConvNeXtSingle(
        channels=4,
        config=make_convnext_config(
            use_partial_conv=True,
        ),
        drop_path_rate=0.0,
    )

    x = make_tensor(channels=4)
    mask = make_mask(channels=4)

    result, result_mask = layer(
        x,
        mask,
    )

    assert result.shape == x.shape
    assert result_mask is not None
    assert result_mask.shape == x.shape


@pytest.mark.pruned
def test_convnext_single_forward_standard_conv_preserves_mask():
    layer = ConvNeXtSingle(
        channels=4,
        config=make_convnext_config(
            use_partial_conv=False,
        ),
        drop_path_rate=0.0,
    )

    x = make_tensor(channels=4)
    mask = make_mask(channels=1)

    result, result_mask = layer(
        x,
        mask,
    )

    assert result.shape == x.shape
    assert result_mask is mask


@pytest.mark.pruned
def test_convnext_single_positive_dropout_uses_dropout2d():
    layer = ConvNeXtSingle(
        channels=4,
        config=make_convnext_config(
            dropout_rate=0.25,
        ),
        drop_path_rate=0.0,
    )

    assert isinstance(layer.dropout, nn.Dropout2d)


@pytest.mark.pruned
def test_convnext_single_zero_dropout_uses_identity():
    layer = ConvNeXtSingle(
        channels=4,
        config=make_convnext_config(
            dropout_rate=0.0,
        ),
        drop_path_rate=0.0,
    )

    assert isinstance(layer.dropout, nn.Identity)


@pytest.mark.pruned
def test_convnext_single_positive_layer_scale_creates_parameter():
    layer = ConvNeXtSingle(
        channels=4,
        config=make_convnext_config(
            layer_scale_init=0.5,
        ),
        drop_path_rate=0.0,
    )

    assert isinstance(
        layer.layer_scale,
        nn.Parameter,
    )
    torch.testing.assert_close(
        layer.layer_scale,
        torch.full(
            (4,),
            0.5,
        ),
    )


@pytest.mark.pruned
@pytest.mark.parametrize(
    "layer_scale_init",
    [
        0.0,
        -1.0,
    ],
)
def test_convnext_single_nonpositive_layer_scale_disables_parameter(
    layer_scale_init,
):
    layer = ConvNeXtSingle(
        channels=4,
        config=make_convnext_config(
            layer_scale_init=layer_scale_init,
        ),
        drop_path_rate=0.0,
    )

    assert layer.layer_scale is None


@pytest.mark.pruned
def test_convnext_single_without_layer_scale_forward():
    layer = ConvNeXtSingle(
        channels=4,
        config=make_convnext_config(
            use_partial_conv=False,
            layer_scale_init=0.0,
        ),
        drop_path_rate=0.0,
    )

    x = make_tensor(channels=4)
    result, result_mask = layer(
        x,
        None,
    )

    assert result.shape == x.shape
    assert result_mask is None


def test_convnext_single_residual_connection():
    config = make_convnext_config(
        use_partial_conv=False,
        layer_scale_init=0.0,
    )
    layer = ConvNeXtSingle(
        channels=4,
        config=config,
        drop_path_rate=0.0,
    )

    class ZeroDepthwise(nn.Module):
        def forward(self, x):
            return torch.zeros_like(x)

    class ZeroPointwise(nn.Module):
        def forward(self, x):
            return torch.zeros(
                x.shape[0],
                4,
                x.shape[2],
                x.shape[3],
                dtype=x.dtype,
                device=x.device,
            )

    layer.depthwise = ZeroDepthwise()
    layer.normalization = nn.Identity()
    layer.pointwise_1 = nn.Identity()
    layer.activation = nn.Identity()
    layer.dropout = nn.Identity()
    layer.pointwise_2 = ZeroPointwise()

    x = make_tensor(channels=4)
    result, _ = layer(
        x,
        None,
    )

    torch.testing.assert_close(
        result,
        x,
    )


@pytest.mark.pruned
def test_convnext_block_requires_projection_when_channels_differ():
    block = ConvNeXtBlock(
        in_channels=3,
        out_channels=5,
        config=make_convnext_config(),
    )

    assert block.requires_projection is True
    assert block.projection_conv is not None


@pytest.mark.pruned
def test_convnext_block_skips_projection_when_channels_match():
    block = ConvNeXtBlock(
        in_channels=4,
        out_channels=4,
        config=make_convnext_config(),
    )

    assert block.requires_projection is False
    assert block.projection_conv is None
    assert isinstance(
        block.projection_norm,
        nn.Identity,
    )


@pytest.mark.pruned
def test_convnext_block_partial_projection():
    block = ConvNeXtBlock(
        in_channels=3,
        out_channels=5,
        config=make_convnext_config(
            use_partial_conv=True,
        ),
    )

    assert isinstance(
        block.projection_conv,
        module.PartialConv2d,
    )


@pytest.mark.pruned
def test_convnext_block_standard_projection():
    block = ConvNeXtBlock(
        in_channels=3,
        out_channels=5,
        config=make_convnext_config(
            use_partial_conv=False,
        ),
    )

    assert isinstance(
        block.projection_conv,
        nn.Conv2d,
    )


def test_convnext_block_partial_projection_forward():
    block = ConvNeXtBlock(
        in_channels=3,
        out_channels=5,
        config=make_convnext_config(
            use_partial_conv=True,
        ),
    )

    result = block(
        TensorMask(
            tensor=make_tensor(channels=3),
            mask=make_mask(channels=3),
        )
    )

    assert result.tensor.shape == (2, 5, 8, 8)
    assert result.mask is not None
    assert result.mask.shape == result.tensor.shape


def test_convnext_block_standard_projection_forward():
    mask = make_mask(channels=1)
    block = ConvNeXtBlock(
        in_channels=3,
        out_channels=5,
        config=make_convnext_config(
            use_partial_conv=False,
        ),
    )

    result = block(
        TensorMask(
            tensor=make_tensor(channels=3),
            mask=mask,
        )
    )

    assert result.tensor.shape == (2, 5, 8, 8)
    assert result.mask is mask


def test_convnext_block_without_projection_forward():
    block = ConvNeXtBlock(
        in_channels=4,
        out_channels=4,
        config=make_convnext_config(
            use_partial_conv=False,
        ),
    )

    result = block(
        TensorMask(
            tensor=make_tensor(channels=4),
            mask=None,
        )
    )

    assert result.tensor.shape == (2, 4, 8, 8)
    assert result.mask is None


@pytest.mark.pruned
def test_convnext_block_reports_output_channels():
    block = ConvNeXtBlock(
        in_channels=3,
        out_channels=7,
        config=make_convnext_config(),
    )

    assert block.out_channels == 7


@pytest.mark.pruned
@pytest.mark.parametrize(
    "fraction_threshold",
    [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ],
)
def test_mask_pool_accepts_valid_fraction_threshold(
    fraction_threshold,
):
    pooling = MaskPool2d(
        method="fraction",
        fraction_threshold=fraction_threshold,
    )

    assert pooling.fraction_threshold == fraction_threshold


@pytest.mark.parametrize(
    "fraction_threshold",
    [
        -0.1,
        -1.0,
        1.1,
        2.0,
    ],
)
def test_mask_pool_rejects_invalid_fraction_threshold(
    fraction_threshold,
):
    with pytest.raises(
        ValueError,
        match="fraction_threshold must be between 0 and 1",
    ):
        MaskPool2d(
            fraction_threshold=fraction_threshold,
        )


@pytest.mark.pruned
def test_mask_pool_defaults():
    pooling = MaskPool2d()

    assert pooling.method == "any"
    assert pooling.fraction_threshold == pytest.approx(0.5)


@pytest.mark.pruned
def test_mask_pool_any_marks_output_valid_when_any_input_is_valid():
    pooling = MaskPool2d(method="any")

    mask = torch.tensor(
        [
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ]
        ]
    )

    result = pooling(mask)

    expected = torch.tensor(
        [
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            ]
        ]
    )

    torch.testing.assert_close(
        result,
        expected,
    )


@pytest.mark.pruned
def test_mask_pool_any_all_zeros():
    pooling = MaskPool2d(method="any")
    mask = torch.zeros(2, 3, 8, 8)

    result = pooling(mask)

    torch.testing.assert_close(
        result,
        torch.zeros(2, 3, 4, 4),
    )


@pytest.mark.pruned
def test_mask_pool_any_all_ones():
    pooling = MaskPool2d(method="any")
    mask = torch.ones(2, 3, 8, 8)

    result = pooling(mask)

    torch.testing.assert_close(
        result,
        torch.ones(2, 3, 4, 4),
    )


@pytest.mark.pruned
def test_mask_pool_all_requires_every_input_to_be_valid():
    pooling = MaskPool2d(method="all")

    mask = torch.tensor(
        [
            [
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0],
                ]
            ]
        ]
    )

    result = pooling(mask)

    expected = torch.tensor(
        [
            [
                [
                    [1.0, 0.0],
                    [1.0, 0.0],
                ]
            ]
        ]
    )

    torch.testing.assert_close(
        result,
        expected,
    )


@pytest.mark.pruned
def test_mask_pool_all_all_ones():
    pooling = MaskPool2d(method="all")
    mask = torch.ones(2, 3, 8, 8)

    result = pooling(mask)

    torch.testing.assert_close(
        result,
        torch.ones(2, 3, 4, 4),
    )


@pytest.mark.pruned
def test_mask_pool_all_single_invalid_value():
    pooling = MaskPool2d(method="all")
    mask = torch.ones(1, 1, 2, 2)
    mask[0, 0, 0, 0] = 0

    result = pooling(mask)

    torch.testing.assert_close(
        result,
        torch.zeros(1, 1, 1, 1),
    )


@pytest.mark.pruned
@pytest.mark.parametrize(
    (
        "number_valid",
        "threshold",
        "expected",
    ),
    [
        (0, 0.5, 0.0),
        (1, 0.5, 0.0),
        (2, 0.5, 1.0),
        (3, 0.5, 1.0),
        (4, 0.5, 1.0),
        (3, 0.75, 1.0),
        (2, 0.75, 0.0),
        (4, 1.0, 1.0),
        (3, 1.0, 0.0),
        (0, 0.0, 1.0),
    ],
)
def test_mask_pool_fraction_threshold_cases(
    number_valid,
    threshold,
    expected,
):
    pooling = MaskPool2d(
        method="fraction",
        fraction_threshold=threshold,
    )

    values = torch.zeros(4)
    values[:number_valid] = 1
    mask = values.reshape(1, 1, 2, 2)

    result = pooling(mask)

    assert result.item() == pytest.approx(expected)


@pytest.mark.pruned
def test_mask_pool_fraction_preserves_dtype():
    pooling = MaskPool2d(
        method="fraction",
        fraction_threshold=0.5,
    )

    mask = torch.ones(
        1,
        1,
        4,
        4,
        dtype=torch.float64,
    )

    result = pooling(mask)

    assert result.dtype == torch.float64


@pytest.mark.pruned
def test_mask_pool_fraction_preserves_channels():
    pooling = MaskPool2d(
        method="fraction",
        fraction_threshold=0.5,
    )

    mask = torch.ones(2, 5, 8, 8)
    result = pooling(mask)

    assert result.shape == (2, 5, 4, 4)


@pytest.mark.pruned
def test_mask_pool_rejects_unsupported_method():
    pooling = MaskPool2d(method="unsupported")

    with pytest.raises(
        ValueError,
        match="Unsupported mask pooling method",
    ):
        pooling(torch.ones(1, 1, 4, 4))


@pytest.mark.pruned
def test_conv_block_config_setup_generator_enables_noise():
    config = make_conv_config()

    result = config.setup_generator(
        inject_noise=True,
    )

    assert result is config
    assert config.inject_noise is True


@pytest.mark.pruned
def test_conv_block_config_setup_generator_defaults_to_disabled():
    config = make_conv_config().setup_generator()

    assert config.inject_noise is False


@pytest.mark.pruned
def test_partial_conv_config_setup_generator_enables_noise():
    config = make_partial_config()

    result = config.setup_generator(
        inject_noise=True,
    )

    assert result is config
    assert config.inject_noise is True


@pytest.mark.pruned
def test_convnext_config_setup_generator_enables_noise():
    config = make_convnext_config()

    result = config.setup_generator(
        inject_noise=True,
    )

    assert result is config
    assert config.inject_noise is True


@pytest.mark.pruned
def test_conv_single_noise_adds_input_channel():
    config = make_conv_config().setup_generator(
        inject_noise=True,
    )

    layer = ConvSingle(
        in_channels=3,
        out_channels=5,
        config=config,
    )

    assert layer.inject_noise is True
    assert layer.conv.in_channels == 4


@pytest.mark.pruned
def test_conv_single_noise_injection_called(
    monkeypatch,
):
    config = make_conv_config().setup_generator(
        inject_noise=True,
    )
    layer = ConvSingle(
        in_channels=3,
        out_channels=4,
        config=config,
    )

    calls = []

    def fake_noise_injection(value):
        calls.append(value)
        noise = torch.zeros(
            value.shape[0],
            1,
            value.shape[2],
            value.shape[3],
            dtype=value.dtype,
            device=value.device,
        )
        return torch.cat(
            [
                value,
                noise,
            ],
            dim=1,
        )

    monkeypatch.setattr(
        module,
        "_noise_injection",
        fake_noise_injection,
    )

    value = make_tensor(
        channels=3,
    )
    result = layer(value)

    assert calls == [value]
    assert result.shape == (
        2,
        4,
        8,
        8,
    )


@pytest.mark.pruned
def test_conv_single_forward_applies_layers_in_order():
    layer = ConvSingle(
        in_channels=3,
        out_channels=3,
        config=make_conv_config(),
    )

    calls = []

    class RecordingLayer(nn.Module):
        def __init__(
            self,
            name,
            increment,
        ):
            super().__init__()
            self.name = name
            self.increment = increment

        def forward(
            self,
            value,
        ):
            calls.append(self.name)
            return value + self.increment

    layer.conv = RecordingLayer(
        "conv",
        1,
    )
    layer.normalization = RecordingLayer(
        "normalization",
        2,
    )
    layer.activation = RecordingLayer(
        "activation",
        3,
    )
    layer.dropout = RecordingLayer(
        "dropout",
        4,
    )

    value = torch.zeros(
        1,
        3,
        2,
        2,
    )
    result = layer(value)

    assert calls == [
        "conv",
        "normalization",
        "activation",
        "dropout",
    ]
    torch.testing.assert_close(
        result,
        torch.full_like(
            value,
            10,
        ),
    )


@pytest.mark.pruned
def test_partial_conv_single_noise_adds_input_channel():
    config = make_partial_config().setup_generator(
        inject_noise=True,
    )

    layer = PartialConvSingle(
        in_channels=3,
        out_channels=4,
        config=config,
    )

    assert layer.inject_noise is True
    assert layer.conv.in_channels == 4


def test_partial_conv_single_noise_expands_mask(
    monkeypatch,
):
    config = make_partial_config().setup_generator(
        inject_noise=True,
    )
    layer = PartialConvSingle(
        in_channels=3,
        out_channels=4,
        config=config,
    )

    value = make_tensor(
        channels=3,
    )
    mask = make_mask(
        channels=3,
    )

    injected = torch.cat(
        [
            value,
            torch.zeros(
                value.shape[0],
                1,
                value.shape[2],
                value.shape[3],
            ),
        ],
        dim=1,
    )
    expanded_mask = torch.ones_like(injected)

    noise_mock = Mock(
        return_value=injected,
    )
    expand_mock = Mock(
        return_value=expanded_mask,
    )

    monkeypatch.setattr(
        module,
        "_noise_injection",
        noise_mock,
    )
    monkeypatch.setattr(
        module,
        "_expand_mask",
        expand_mock,
    )

    result, result_mask = layer(
        value,
        mask,
    )

    noise_mock.assert_called_once_with(value)
    expand_mock.assert_called_once_with(
        injected,
        mask,
    )
    assert result.shape == (
        2,
        4,
        8,
        8,
    )
    assert result_mask is not None


@pytest.mark.pruned
def test_partial_conv_single_noise_does_not_expand_single_channel_mask(
    monkeypatch,
):
    config = make_partial_config()
    config.multi_channel = False
    config.setup_generator(
        inject_noise=True,
    )

    layer = PartialConvSingle(
        in_channels=3,
        out_channels=4,
        config=config,
    )

    value = make_tensor(
        channels=3,
    )
    injected = torch.cat(
        [
            value,
            torch.zeros(
                value.shape[0],
                1,
                value.shape[2],
                value.shape[3],
            ),
        ],
        dim=1,
    )

    monkeypatch.setattr(
        module,
        "_noise_injection",
        Mock(return_value=injected),
    )

    expand_mock = Mock()
    monkeypatch.setattr(
        module,
        "_expand_mask",
        expand_mock,
    )

    result, _ = layer(
        value,
        make_mask(channels=1),
    )

    assert result.shape == (
        2,
        4,
        8,
        8,
    )
    expand_mock.assert_not_called()


@pytest.mark.pruned
def test_convnext_single_noise_expands_pointwise_channels():
    config = make_convnext_config(
        expansion_ratio=3,
    ).setup_generator(
        inject_noise=True,
    )

    layer = ConvNeXtSingle(
        channels=4,
        config=config,
        drop_path_rate=0.0,
    )

    assert layer.pointwise_1.in_channels == 5
    assert layer.pointwise_1.out_channels == 12
    assert layer.pointwise_2.in_channels == 13
    assert layer.pointwise_2.out_channels == 4


def test_convnext_single_noise_injected_twice(
    monkeypatch,
):
    config = make_convnext_config(
        use_partial_conv=False,
    ).setup_generator(
        inject_noise=True,
    )

    layer = ConvNeXtSingle(
        channels=4,
        config=config,
        drop_path_rate=0.0,
    )

    calls = []

    def fake_noise(value):
        calls.append(value.shape)
        noise = torch.zeros(
            value.shape[0],
            1,
            value.shape[2],
            value.shape[3],
            dtype=value.dtype,
            device=value.device,
        )
        return torch.cat(
            [
                value,
                noise,
            ],
            dim=1,
        )

    monkeypatch.setattr(
        module,
        "_noise_injection",
        fake_noise,
    )

    value = make_tensor(
        channels=4,
    )

    result, result_mask = layer(
        value,
        None,
    )

    assert calls == [
        torch.Size([2, 4, 8, 8]),
        torch.Size([2, 8, 8, 8]),
    ]
    assert result.shape == value.shape
    assert result_mask is None


@pytest.mark.pruned
def test_convnext_single_layer_scale_is_applied():
    config = make_convnext_config(
        use_partial_conv=False,
        expansion_ratio=1,
        layer_scale_init=2.0,
    )

    layer = ConvNeXtSingle(
        channels=2,
        config=config,
        drop_path_rate=0.0,
    )

    layer.depthwise = nn.Identity()
    layer.normalization = nn.Identity()
    layer.pointwise_1 = nn.Identity()
    layer.activation = nn.Identity()
    layer.dropout = nn.Identity()
    layer.pointwise_2 = nn.Identity()
    layer.drop_path = nn.Identity()

    value = torch.ones(
        1,
        2,
        2,
        2,
    )

    result, _ = layer(
        value,
        None,
    )

    torch.testing.assert_close(
        result,
        torch.full_like(
            value,
            3.0,
        ),
    )


@pytest.mark.pruned
def test_convnext_block_passes_output_through_every_block():
    config = make_convnext_config(
        use_partial_conv=False,
        num_blocks=3,
    )

    block = ConvNeXtBlock(
        in_channels=4,
        out_channels=4,
        config=config,
    )

    calls = []

    class AddBlock(nn.Module):
        def __init__(
            self,
            index,
        ):
            super().__init__()
            self.index = index

        def forward(
            self,
            value,
            mask,
        ):
            calls.append(self.index)
            return value + 1, mask

    block.blocks = nn.ModuleList(
        [
            AddBlock(0),
            AddBlock(1),
            AddBlock(2),
        ]
    )

    value = torch.zeros(
        1,
        4,
        2,
        2,
    )
    mask = torch.ones(
        1,
        1,
        2,
        2,
    )

    result = block(
        TensorMask(
            tensor=value,
            mask=mask,
        )
    )

    assert calls == [
        0,
        1,
        2,
    ]
    torch.testing.assert_close(
        result.tensor,
        torch.full_like(
            value,
            3,
        ),
    )
    assert result.mask is mask


@pytest.mark.pruned
def test_latent_layer_with_normalization_builds_sequence():
    layer = LatentLayer(
        input_shape=(
            3,
            4,
            4,
        ),
        latent_size=5,
        latent_normalization="layer",
    )

    assert isinstance(
        layer.normalization,
        nn.Sequential,
    )
    assert isinstance(
        layer.normalization[-1],
        nn.ReLU,
    )


@pytest.mark.pruned
def test_latent_block_uses_conv_block_group_norm_groups():
    config = make_conv_config(
        group_norm_groups=2,
    )
    conv_block = ConvBlock(
        in_channels=4,
        out_channels=4,
        config=config,
    )

    latent_block = LatentBlock(
        conv_block=conv_block,
        input_shape=(
            4,
            2,
            2,
        ),
        latent_size=3,
        latent_normalization="group",
    )

    normalization = latent_block.latent_head.normalization[0]

    assert isinstance(
        normalization,
        nn.GroupNorm,
    )
    assert normalization.num_groups == 2


@pytest.mark.pruned
def test_latent_block_defaults_group_norm_groups_when_missing():
    class Config:
        pass

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = Config()

        def forward(
            self,
            value,
        ):
            return value

    latent_block = LatentBlock(
        conv_block=Block(),
        input_shape=(
            8,
            2,
            2,
        ),
        latent_size=3,
        latent_normalization="group",
    )

    normalization = latent_block.latent_head.normalization[0]

    assert isinstance(
        normalization,
        nn.GroupNorm,
    )
    assert normalization.num_groups == 8


@pytest.mark.pruned
def test_latent_block_delegates_tensor_to_latent_head():
    class FakeConvBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = make_conv_config()

        def forward(
            self,
            value,
        ):
            return TensorMask(
                tensor=value.tensor + 1,
                mask=value.mask,
            )

    conv_block = FakeConvBlock()

    latent_block = LatentBlock(
        conv_block=conv_block,
        input_shape=(
            2,
            2,
            2,
        ),
        latent_size=3,
        latent_normalization=None,
    )

    captured = {}

    class RecordingHead(nn.Module):
        def forward(
            self,
            value,
        ):
            captured["value"] = value
            return LatentVector(
                mu=torch.zeros(
                    value.shape[0],
                    3,
                ),
                log_var=None,
            )

    latent_block.latent_head = RecordingHead()

    input_tensor = torch.zeros(
        4,
        2,
        2,
        2,
    )

    result = latent_block(
        TensorMask(
            tensor=input_tensor,
            mask=None,
        )
    )

    torch.testing.assert_close(
        captured["value"],
        input_tensor + 1,
    )
    assert result.mu.shape == (
        4,
        3,
    )
    assert result.log_var is None


@pytest.mark.pruned
@pytest.mark.parametrize(
    "method",
    [
        "any",
        "all",
        "fraction",
    ],
)
def test_mask_pool_output_shape_for_odd_dimensions(
    method,
):
    pooling = MaskPool2d(
        method=method,
        fraction_threshold=0.5,
    )

    result = pooling(
        torch.ones(
            2,
            3,
            9,
            11,
        )
    )

    assert result.shape == (
        2,
        3,
        5,
        6,
    )


def test_mask_pool_fraction_boundary_is_inclusive():
    pooling = MaskPool2d(
        method="fraction",
        fraction_threshold=0.5,
    )

    mask = torch.tensor(
        [
            [
                [
                    [1.0, 1.0],
                    [0.0, 0.0],
                ]
            ]
        ]
    )

    result = pooling(mask)

    assert result.item() == pytest.approx(1.0)


@pytest.mark.pruned
def test_mask_pool_fraction_uses_actual_border_count():
    pooling = MaskPool2d(
        method="fraction",
        fraction_threshold=1.0,
    )

    mask = torch.ones(
        1,
        1,
        2,
        2,
    )

    result = pooling(mask)

    assert result.item() == pytest.approx(1.0)
