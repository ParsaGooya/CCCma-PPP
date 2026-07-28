from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

import cccma_ppp.models.unet_models.cvae as module
from cccma_ppp.models.layers.conv import (
    ConvBlockConfig,
    ConvNeXtBlockConfig,
    PartialConvBlockConfig,
)
from cccma_ppp.models.unet_models.cvae import (
    cVAEUNetConfig,
)


def make_block_config():
    return ConvBlockConfig(
        name="standard_conv",
        num_convolutions=1,
        kernel_size=3,
        normalization="none",
        padding_method="zeros",
        activation="relu",
        dropout_rate=None,
        bias=False,
        group_norm_groups=1,
    )


def make_config(**kwargs):
    defaults = {
        "channels": [8, 16, 32],
        "latent_size": 4,
        "condition_embedding_channels": [4, 8],
        "condition_embedding_size": 4,
        "condition_dependant_latent": False,
        "condemb_to_decoder": True,
        "block_config": make_block_config(),
        "upsampling_method": "bilinear",
        "transpose_kernel_sizes": 3,
        "add_skip_latent": False,
        "mask_pooling": "any",
        "mask_fraction_threshold": 0.5,
        "output_activation": "identity",
        "output_hidden_channels": None,
        "init_method": "trunc_normal",
        "GENERATOR": None,
    }
    defaults.update(kwargs)

    with patch.object(
        module,
        "_unet_config_checks",
        return_value=None,
    ):
        return cVAEUNetConfig(**defaults)


def test_config_default_generator_is_none():
    config = make_config()

    assert config.GENERATOR is None


def test_config_preserves_channels():
    channels = [8, 16, 32, 64]

    config = make_config(
        channels=channels,
    )

    assert config.channels is channels


def test_config_preserves_latent_size():
    config = make_config(
        latent_size=12,
    )

    assert config.latent_size == 12


def test_config_accepts_none_latent_size():
    config = make_config(
        latent_size=None,
    )

    assert config.latent_size is None


def test_config_preserves_condition_embedding_channels():
    embedding_channels = [4, 8, 16]

    config = make_config(
        condition_embedding_channels=embedding_channels,
    )

    assert config.condition_embedding_channels is embedding_channels


def test_config_accepts_none_condition_embedding_channels():
    config = make_config(
        condition_embedding_channels=None,
    )

    assert config.condition_embedding_channels is None


def test_config_preserves_condition_embedding_size():
    config = make_config(
        condition_embedding_size=7,
    )

    assert config.condition_embedding_size == 7


@pytest.mark.parametrize(
    "condition_dependant_latent",
    [
        True,
        False,
    ],
)
def test_config_preserves_condition_dependant_latent(
    condition_dependant_latent,
):
    config = make_config(
        condition_dependant_latent=condition_dependant_latent,
    )

    assert config.condition_dependant_latent is condition_dependant_latent


@pytest.mark.parametrize(
    "condemb_to_decoder",
    [
        True,
        False,
    ],
)
def test_config_preserves_condemb_to_decoder(
    condemb_to_decoder,
):
    config = make_config(
        condemb_to_decoder=condemb_to_decoder,
    )

    assert config.condemb_to_decoder is condemb_to_decoder


@pytest.mark.parametrize(
    "add_skip_latent",
    [
        True,
        False,
    ],
)
def test_config_preserves_add_skip_latent(
    add_skip_latent,
):
    config = make_config(
        add_skip_latent=add_skip_latent,
    )

    assert config.add_skip_latent is add_skip_latent


def test_config_preserves_block_config():
    block_config = make_block_config()

    config = make_config(
        block_config=block_config,
    )

    assert config.block_config is block_config


def test_post_init_calls_unet_config_checks():
    with patch.object(
        module,
        "_unet_config_checks",
        return_value=None,
    ) as checks:
        config = cVAEUNetConfig(
            channels=[8, 16, 32],
            latent_size=4,
            condition_embedding_channels=[4, 8],
            condition_embedding_size=4,
            condition_dependant_latent=False,
            block_config=make_block_config(),
            transpose_kernel_sizes=3,
        )

    checks.assert_called_once_with(config)


def test_post_init_propagates_unet_config_check_error():
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
        cVAEUNetConfig(
            channels=[8, 16, 32],
            latent_size=4,
            condition_embedding_channels=[4, 8],
            condition_embedding_size=4,
            condition_dependant_latent=False,
            block_config=make_block_config(),
            transpose_kernel_sizes=3,
        )


def test_transpose_kernel_normalization_occurs_after_shared_validation():
    observed = {}

    def fake_checks(config):
        observed["kernel_sizes"] = config.transpose_kernel_sizes

    with patch.object(
        module,
        "_unet_config_checks",
        side_effect=fake_checks,
    ):
        config = cVAEUNetConfig(
            channels=[8, 16, 32],
            latent_size=4,
            condition_embedding_channels=[4, 8],
            condition_embedding_size=4,
            condition_dependant_latent=False,
            block_config=make_block_config(),
            transpose_kernel_sizes=5,
        )

    assert observed["kernel_sizes"] == 5
    assert config.transpose_kernel_sizes == [5, 5]


@pytest.mark.parametrize(
    ("channels", "expected_stages"),
    [
        ([8], 0),
        ([8, 16], 1),
        ([8, 16, 32], 2),
        ([8, 16, 32, 64], 3),
        ([8, 16, 32, 64, 128], 4),
    ],
)
def test_number_of_upsampling_stages(
    channels,
    expected_stages,
):
    config = make_config(
        channels=channels,
        transpose_kernel_sizes=None,
    )

    assert len(config.transpose_kernel_sizes) == expected_stages


def test_none_transpose_kernel_sizes_uses_three_per_stage():
    config = make_config(
        channels=[8, 16, 32, 64],
        transpose_kernel_sizes=None,
    )

    assert config.transpose_kernel_sizes == [
        3,
        3,
        3,
    ]


@pytest.mark.parametrize(
    "kernel_size",
    [
        1,
        2,
        3,
        5,
        7,
    ],
)
def test_integer_transpose_kernel_size_is_repeated(
    kernel_size,
):
    config = make_config(
        channels=[8, 16, 32, 64],
        transpose_kernel_sizes=kernel_size,
    )

    assert config.transpose_kernel_sizes == [
        kernel_size,
        kernel_size,
        kernel_size,
    ]


def test_integer_transpose_kernel_size_for_single_stage():
    config = make_config(
        channels=[8, 16],
        transpose_kernel_sizes=4,
    )

    assert config.transpose_kernel_sizes == [4]


def test_integer_transpose_kernel_size_for_zero_stages():
    config = make_config(
        channels=[8],
        transpose_kernel_sizes=3,
    )

    assert config.transpose_kernel_sizes == []


def test_explicit_kernel_list_is_preserved():
    kernel_sizes = [2, 3, 4]

    config = make_config(
        channels=[8, 16, 32, 64],
        transpose_kernel_sizes=kernel_sizes,
    )

    assert config.transpose_kernel_sizes is kernel_sizes


def test_explicit_kernel_tuple_sequence_is_preserved():
    kernel_sizes = [
        (2, 2),
        (3, 3),
    ]

    with patch.object(
        module,
        "_unet_config_checks",
        return_value=None,
    ):
        with pytest.raises(TypeError):
            cVAEUNetConfig(
                channels=[8, 16, 32],
                latent_size=4,
                condition_embedding_channels=[4, 8],
                condition_embedding_size=4,
                condition_dependant_latent=False,
                block_config=make_block_config(),
                transpose_kernel_sizes=kernel_sizes,
            )


@pytest.mark.parametrize(
    ("channels", "kernel_sizes", "expected", "actual"),
    [
        ([8, 16, 32], [3], 2, 1),
        ([8, 16, 32], [3, 3, 3], 2, 3),
        ([8, 16], [], 1, 0),
        ([8], [3], 0, 1),
    ],
)
def test_rejects_wrong_number_of_kernel_sizes(
    channels,
    kernel_sizes,
    expected,
    actual,
):
    with pytest.raises(
        ValueError,
        match=(rf"Expected {expected}, got {actual}"),
    ):
        make_config(
            channels=channels,
            transpose_kernel_sizes=kernel_sizes,
        )


@pytest.mark.parametrize(
    "kernel_sizes",
    [
        [0, 3],
        [-1, 3],
        [3, 0],
        [3, -5],
    ],
)
def test_rejects_nonpositive_integer_kernel_sizes(
    kernel_sizes,
):
    with pytest.raises(
        ValueError,
        match=("All transpose-convolution kernel sizes must be positive integers"),
    ):
        make_config(
            channels=[8, 16, 32],
            transpose_kernel_sizes=kernel_sizes,
        )


@pytest.mark.parametrize(
    "kernel_sizes",
    [
        [3.0, 3],
        ["3", 3],
        [None, 3],
        [object(), 3],
        [[3, 3], 3],
    ],
)
def test_rejects_invalid_kernel_size_types(
    kernel_sizes,
):
    with pytest.raises(
        ValueError,
        match=("All transpose-convolution kernel sizes must be positive integers"),
    ):
        make_config(
            channels=[8, 16, 32],
            transpose_kernel_sizes=kernel_sizes,
        )


def test_kernel_length_validation_precedes_value_validation():
    with pytest.raises(
        ValueError,
        match="Expected 2, got 1",
    ):
        make_config(
            channels=[8, 16, 32],
            transpose_kernel_sizes=[0],
        )


@pytest.mark.parametrize(
    "upsampling_method",
    [
        "bilinear",
        "transpose_conv",
    ],
)
def test_preserves_upsampling_method(
    upsampling_method,
):
    config = make_config(
        upsampling_method=upsampling_method,
    )

    assert config.upsampling_method == upsampling_method


@pytest.mark.parametrize(
    "mask_pooling",
    [
        "any",
        "all",
        "fraction",
    ],
)
def test_preserves_mask_pooling(
    mask_pooling,
):
    config = make_config(
        mask_pooling=mask_pooling,
    )

    assert config.mask_pooling == mask_pooling


@pytest.mark.parametrize(
    "threshold",
    [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ],
)
def test_preserves_mask_fraction_threshold(
    threshold,
):
    config = make_config(
        mask_fraction_threshold=threshold,
    )

    assert config.mask_fraction_threshold == threshold


@pytest.mark.parametrize(
    "activation",
    [
        "identity",
        "sigmoid",
        "tanh",
    ],
)
def test_preserves_output_activation(
    activation,
):
    config = make_config(
        output_activation=activation,
    )

    assert config.output_activation == activation


@pytest.mark.parametrize(
    "hidden_channels",
    [
        None,
        1,
        8,
        32,
    ],
)
def test_preserves_output_hidden_channels(
    hidden_channels,
):
    config = make_config(
        output_hidden_channels=hidden_channels,
    )

    assert config.output_hidden_channels == hidden_channels


@pytest.mark.parametrize(
    "init_method",
    [
        "trunc_normal",
        "xavier",
    ],
)
def test_preserves_init_method(
    init_method,
):
    config = make_config(
        init_method=init_method,
    )

    assert config.init_method == init_method


def test_preserves_generator_configuration():
    generator = SimpleNamespace(
        type="generator",
    )

    config = make_config(
        GENERATOR=generator,
    )

    assert config.GENERATOR is generator


def test_accepts_standard_conv_block_config():
    block_config = make_block_config()

    config = make_config(
        block_config=block_config,
    )

    assert config.block_config is block_config


def test_accepts_partial_conv_block_config():
    block_config = PartialConvBlockConfig(
        name="partial_conv",
        num_convolutions=1,
        kernel_size=3,
        normalization="none",
        padding_method="zeros",
        activation="relu",
        dropout_rate=None,
        bias=False,
        group_norm_groups=1,
    )

    config = make_config(
        block_config=block_config,
    )

    assert config.block_config is block_config


def test_accepts_convnext_block_config():
    block_config = ConvNeXtBlockConfig(
        name="convnext",
        num_blocks=1,
        kernel_size=3,
        expansion_ratio=2,
        padding_method="zeros",
        layer_scale_init=1e-6,
        dropout_rate=0.0,
        drop_path_rate=0.0,
        use_partial_conv=True,
    )

    config = make_config(
        block_config=block_config,
    )

    assert config.block_config is block_config


@pytest.mark.xfail(reason="cVAEUNet is no longer exposed by this module")
def test_build_constructs_cvae_unet():
    config = make_config()
    expected = object()
    input_shape = np.asarray([2, 8, 8])
    output_shape = np.asarray([1, 8, 8])

    with patch.object(
        module,
        "cVAEUNet",
        return_value=expected,
    ) as constructor:
        result = config.build(
            input_shape=input_shape,
            output_shape=output_shape,
            added_features_dim=4,
        )

    assert result is expected

    constructor.assert_called_once_with(
        config=config,
        input_shape=input_shape,
        output_shape=output_shape,
        added_features_dim=4,
    )


@pytest.mark.xfail(reason="cVAEUNet is no longer exposed by this module")
def test_build_passes_none_output_shape():
    config = make_config()
    expected = object()
    input_shape = np.asarray([2, 8, 8])

    with patch.object(
        module,
        "cVAEUNet",
        return_value=expected,
    ) as constructor:
        result = config.build(
            input_shape=input_shape,
            output_shape=None,
            added_features_dim=None,
        )

    assert result is expected

    constructor.assert_called_once_with(
        config=config,
        input_shape=input_shape,
        output_shape=None,
        added_features_dim=None,
    )


@pytest.mark.xfail(reason="cVAEUNet is no longer exposed by this module")
def test_build_accepts_tuple_shapes():
    config = make_config()
    expected = object()

    with patch.object(
        module,
        "cVAEUNet",
        return_value=expected,
    ) as constructor:
        result = config.build(
            input_shape=(2, 8, 8),
            output_shape=(1, 8, 8),
            added_features_dim=3,
        )

    assert result is expected

    constructor.assert_called_once_with(
        config=config,
        input_shape=(2, 8, 8),
        output_shape=(1, 8, 8),
        added_features_dim=3,
    )
