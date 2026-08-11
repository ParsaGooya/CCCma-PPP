from types import SimpleNamespace

import pytest

from cccma_ppp.architectures.unet.utils import (
    _unet_config_checks,
)


def make_config(**overrides):
    values = {
        "channels": [4, 8],
        "condition_embedding_channels": [4, 8],
        "transpose_kernel_sizes": [3],
        "bottleneck_dim": 16,
        "latent_size": 4,
        "condition_embedding_size": 4,
        "condition_dependant_latent": True,
        "condemb_to_decoder": True,
        "deterministic_guess_config": None,
        "mask_fraction_threshold": 0.5,
        "output_block_hidden_channels": 32,
        "GENERATOR": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_unet_config_checks_accepts_valid_minimal_config():
    config = make_config()

    assert _unet_config_checks(config) is None


def test_unet_config_checks_accepts_single_channel_level():
    config = make_config(
        channels=[4],
        condition_embedding_channels=[4],
        transpose_kernel_sizes=[],
    )

    assert _unet_config_checks(config) is None


def test_unet_config_checks_accepts_multiple_channel_levels():
    config = make_config(
        channels=[4, 8, 16, 32],
        condition_embedding_channels=[2, 4, 8],
        transpose_kernel_sizes=[
            3,
            (2, 4),
            5,
        ],
    )

    assert _unet_config_checks(config) is None


@pytest.mark.parametrize(
    "transpose_kernel_sizes",
    [
        1,
        2,
        3,
        5,
        7,
    ],
)
def test_unet_config_checks_accepts_positive_integer_kernel(
    transpose_kernel_sizes,
):
    config = make_config(
        transpose_kernel_sizes=transpose_kernel_sizes,
    )

    assert _unet_config_checks(config) is None


@pytest.mark.parametrize(
    "kernel",
    [
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 4),
        (5, 7),
    ],
)
def test_unet_config_checks_accepts_tuple_kernels_in_list(
    kernel,
):
    config = make_config(
        transpose_kernel_sizes=[kernel],
    )

    assert _unet_config_checks(config) is None


def test_unet_config_checks_accepts_mixed_kernel_list():
    config = make_config(
        channels=[4, 8, 16, 32],
        condition_embedding_channels=[4, 8, 16, 32],
        transpose_kernel_sizes=[
            3,
            (2, 4),
            5,
        ],
    )

    assert _unet_config_checks(config) is None


def test_unet_config_checks_accepts_none_condition_channels():
    config = make_config(
        condition_embedding_channels=None,
    )

    assert _unet_config_checks(config) is None


def test_unet_config_checks_accepts_none_dimensions():
    config = make_config(
        bottleneck_dim=None,
        latent_size=None,
        condition_embedding_size=None,
    )

    assert _unet_config_checks(config) is None


@pytest.mark.parametrize(
    "mask_fraction_threshold",
    [
        0.0,
        0.1,
        0.5,
        0.9,
        1.0,
    ],
)
def test_unet_config_checks_accepts_valid_mask_thresholds(
    mask_fraction_threshold,
):
    config = make_config(
        mask_fraction_threshold=mask_fraction_threshold,
    )

    assert _unet_config_checks(config) is None


@pytest.mark.parametrize(
    "output_block_hidden_channels",
    [
        None,
        1,
        8,
        64,
    ],
)
def test_unet_config_checks_accepts_valid_output_hidden_channels(
    output_block_hidden_channels,
):
    config = make_config(
        output_block_hidden_channels=(output_block_hidden_channels),
    )

    assert _unet_config_checks(config) is None


@pytest.mark.parametrize(
    "num_training_noise_samples",
    [
        1,
        2,
        8,
        32,
    ],
)
def test_unet_config_checks_accepts_valid_generator_sample_counts(
    num_training_noise_samples,
):
    config = make_config(
        GENERATOR=SimpleNamespace(
            num_training_noise_samples=(num_training_noise_samples),
        ),
    )

    assert _unet_config_checks(config) is None


def test_unet_config_checks_accepts_generator_none():
    config = make_config(
        GENERATOR=None,
    )

    assert _unet_config_checks(config) is None


def test_unet_config_checks_accepts_condition_independent_latent_with_decoder():
    config = make_config(
        condition_dependant_latent=False,
        condemb_to_decoder=True,
    )

    assert _unet_config_checks(config) is None


def test_unet_config_checks_accepts_condition_dependent_latent_without_decoder():
    config = make_config(
        condition_dependant_latent=True,
        condemb_to_decoder=False,
    )

    assert _unet_config_checks(config) is None


def test_unet_config_checks_skips_condition_rule_when_attribute_is_none():
    config = make_config(
        condition_dependant_latent=None,
        condemb_to_decoder=False,
    )

    assert _unet_config_checks(config) is None


def test_unet_config_checks_does_not_mutate_config():
    config = make_config(
        channels=[4, 8, 16],
        condition_embedding_channels=[2, 4],
        transpose_kernel_sizes=[
            3,
            (2, 4),
        ],
    )

    original_channels = list(config.channels)
    original_condition_channels = list(config.condition_embedding_channels)
    original_kernels = list(config.transpose_kernel_sizes)

    _unet_config_checks(config)

    assert config.channels == original_channels
    assert config.condition_embedding_channels == original_condition_channels
    assert config.transpose_kernel_sizes == original_kernels


@pytest.mark.parametrize(
    (
        "channels",
        "transpose_kernel_sizes",
        "expected",
        "actual",
    ),
    [
        ([4, 8, 16], [3], 2, 1),
        ([4, 8, 16], [3, 3, 3], 2, 3),
        ([4, 8], [], 1, 0),
        ([4], [3], 0, 1),
    ],
)
def test_unet_config_checks_rejects_wrong_kernel_count(
    channels,
    transpose_kernel_sizes,
    expected,
    actual,
):
    config = make_config(
        channels=channels,
        condition_embedding_channels=channels,
        transpose_kernel_sizes=transpose_kernel_sizes,
    )

    with pytest.raises(
        ValueError,
        match=("transpose_kernel_sizes must contain one value per upsampling stage"),
    ) as exc_info:
        _unet_config_checks(config)

    message = str(exc_info.value)

    assert f"Expected {expected}" in message
    assert f"got {actual}" in message


@pytest.mark.parametrize(
    "transpose_kernel_sizes",
    [
        0,
        -1,
        -10,
    ],
)
def test_unet_config_checks_rejects_nonpositive_scalar_kernel(
    transpose_kernel_sizes,
):
    config = make_config(
        transpose_kernel_sizes=transpose_kernel_sizes,
    )

    with pytest.raises(
        ValueError,
        match=(
            "kernel size must be either a positive integer "
            "or a tuple of two positive integers"
        ),
    ):
        _unet_config_checks(config)


@pytest.mark.parametrize(
    "kernel",
    [
        0,
        -1,
        -10,
        (0, 1),
        (1, 0),
        (-1, 2),
        (2, -1),
        (0, 0),
        (1,),
        (1, 2, 3),
        ("1", 2),
        (1, "2"),
        (1.0, 2),
        (1, 2.0),
        ("a", "b"),
    ],
)
def test_unet_config_checks_rejects_invalid_list_kernel(
    kernel,
):
    config = make_config(
        transpose_kernel_sizes=[kernel],
    )

    with pytest.raises(
        ValueError,
        match=(
            "kernel size must be either a positive integer "
            "or a tuple of two positive integers"
        ),
    ):
        _unet_config_checks(config)


@pytest.mark.parametrize(
    "kernel",
    [
        None,
        "3",
        {"height": 3, "width": 3},
    ],
)
def test_unet_config_checks_rejects_or_errors_for_unsupported_scalar_kernel(
    kernel,
):
    config = make_config(
        transpose_kernel_sizes=kernel,
    )

    with pytest.raises(
        (TypeError, ValueError),
    ):
        _unet_config_checks(config)


def test_kernel_count_validation_precedes_kernel_value_validation():
    config = make_config(
        channels=[4, 8, 16],
        condition_embedding_channels=[4, 8, 16],
        transpose_kernel_sizes=[0],
    )

    with pytest.raises(
        ValueError,
        match="Expected 2, got 1",
    ):
        _unet_config_checks(config)


def test_unet_config_checks_rejects_empty_channels():
    config = make_config(
        channels=[],
        condition_embedding_channels=[4],
        transpose_kernel_sizes=[],
    )

    with pytest.raises(
        ValueError,
        match="Expected -1, got 0",
    ):
        _unet_config_checks(config)


@pytest.mark.parametrize(
    "channels",
    [
        [0],
        [-1],
        [4, 0],
        [4, -1],
        [4, 8, 0],
    ],
)
def test_unet_config_checks_rejects_nonpositive_channels(
    channels,
):
    config = make_config(
        channels=channels,
        condition_embedding_channels=None,
        transpose_kernel_sizes=[3 for _ in range(max(len(channels) - 1, 0))],
    )

    with pytest.raises(
        ValueError,
        match="All channels sizes must be positive integers",
    ):
        _unet_config_checks(config)


def test_unet_config_checks_rejects_empty_condition_channels():
    config = make_config(
        condition_embedding_channels=[],
    )

    with pytest.raises(
        ValueError,
        match=("condition_embedding_channels must contain at least one encoder width"),
    ):
        _unet_config_checks(config)


@pytest.mark.parametrize(
    "condition_embedding_channels",
    [
        [0],
        [-1],
        [4, 0],
        [4, -8],
        [4, 8, 0],
    ],
)
def test_unet_config_checks_rejects_nonpositive_condition_channels(
    condition_embedding_channels,
):
    config = make_config(
        condition_embedding_channels=(condition_embedding_channels),
    )

    with pytest.raises(
        ValueError,
        match=("All condition_embedding_channels sizes must be positive integers"),
    ):
        _unet_config_checks(config)


def test_channel_validation_precedes_dimension_validation():
    config = make_config(
        channels=[],
        transpose_kernel_sizes=[],
        bottleneck_dim=0,
    )

    with pytest.raises(
        ValueError,
        match="Expected -1, got 0",
    ):
        _unet_config_checks(config)


@pytest.mark.parametrize(
    "bottleneck_dim",
    [
        0,
        -1,
        -10,
    ],
)
def test_unet_config_checks_rejects_nonpositive_bottleneck(
    bottleneck_dim,
):
    config = make_config(
        bottleneck_dim=bottleneck_dim,
    )

    with pytest.raises(
        ValueError,
        match="bottleneck_dim must be a positive integer",
    ):
        _unet_config_checks(config)


@pytest.mark.parametrize(
    "latent_size",
    [
        0,
        -1,
        -10,
    ],
)
def test_unet_config_checks_rejects_nonpositive_latent_size(
    latent_size,
):
    config = make_config(
        latent_size=latent_size,
    )

    with pytest.raises(
        ValueError,
        match="latent_size must be a positive integer",
    ):
        _unet_config_checks(config)


@pytest.mark.parametrize(
    "condition_embedding_size",
    [
        0,
        -1,
        -10,
    ],
)
def test_unet_config_checks_rejects_nonpositive_condition_embedding_size(
    condition_embedding_size,
):
    config = make_config(
        condition_embedding_size=condition_embedding_size,
    )

    with pytest.raises(
        ValueError,
        match=("condition_embedding_size must be a positive integer"),
    ):
        _unet_config_checks(config)


def test_dimension_validation_order_starts_with_bottleneck():
    config = make_config(
        bottleneck_dim=0,
        latent_size=0,
        condition_embedding_size=0,
    )

    with pytest.raises(
        ValueError,
        match="bottleneck_dim must be a positive integer",
    ):
        _unet_config_checks(config)


def test_latent_size_checked_before_condition_embedding_size():
    config = make_config(
        bottleneck_dim=16,
        latent_size=0,
        condition_embedding_size=0,
    )

    with pytest.raises(
        ValueError,
        match="latent_size must be a positive integer",
    ):
        _unet_config_checks(config)


def test_unet_config_checks_rejects_independent_latent_without_decoder_condition():
    config = make_config(
        condition_dependant_latent=False,
        condemb_to_decoder=False,
    )

    with pytest.raises(
        ValueError,
        match=("condition embedding has to be passed to decoder"),
    ):
        _unet_config_checks(config)


def test_condition_rule_precedes_mask_threshold_validation():
    config = make_config(
        condition_dependant_latent=False,
        condemb_to_decoder=False,
        mask_fraction_threshold=2.0,
    )

    with pytest.raises(
        ValueError,
        match=("condition embedding has to be passed to decoder"),
    ):
        _unet_config_checks(config)


@pytest.mark.parametrize(
    "mask_fraction_threshold",
    [
        -10.0,
        -1.0,
        -0.01,
        1.01,
        2.0,
        100.0,
    ],
)
def test_unet_config_checks_rejects_invalid_mask_threshold(
    mask_fraction_threshold,
):
    config = make_config(
        mask_fraction_threshold=mask_fraction_threshold,
    )

    with pytest.raises(
        ValueError,
        match=("mask_fraction_threshold must be between 0 and 1"),
    ):
        _unet_config_checks(config)


def test_mask_threshold_checked_before_output_hidden_channels():
    config = make_config(
        mask_fraction_threshold=-1.0,
        output_block_hidden_channels=0,
    )

    with pytest.raises(
        ValueError,
        match=("mask_fraction_threshold must be between 0 and 1"),
    ):
        _unet_config_checks(config)


@pytest.mark.parametrize(
    "output_block_hidden_channels",
    [
        0,
        -1,
        -10,
    ],
)
def test_unet_config_checks_rejects_nonpositive_output_hidden_channels(
    output_block_hidden_channels,
):
    config = make_config(
        output_block_hidden_channels=(output_block_hidden_channels),
    )

    with pytest.raises(
        ValueError,
        match="output_block_hidden_channels must be positive",
    ):
        _unet_config_checks(config)


def test_output_hidden_channels_checked_before_generator():
    config = make_config(
        output_block_hidden_channels=0,
        GENERATOR=SimpleNamespace(
            num_training_noise_samples=0,
        ),
    )

    with pytest.raises(
        ValueError,
        match="output_block_hidden_channels must be positive",
    ):
        _unet_config_checks(config)


@pytest.mark.parametrize(
    "num_training_noise_samples",
    [
        0,
        -1,
        -10,
    ],
)
def test_unet_config_checks_rejects_nonpositive_training_noise_samples(
    num_training_noise_samples,
):
    config = make_config(
        GENERATOR=SimpleNamespace(
            num_training_noise_samples=(num_training_noise_samples),
        ),
    )

    with pytest.raises(
        ValueError,
        match=("num_training_noise_samples.*larger than 0"),
    ):
        _unet_config_checks(config)


def test_generator_error_reports_requirement():
    config = make_config(
        GENERATOR=SimpleNamespace(
            num_training_noise_samples=0,
        ),
    )

    with pytest.raises(ValueError) as exc_info:
        _unet_config_checks(config)

    message = str(exc_info.value)

    assert "Generator" in message
    assert "num_training_noise_samples" in message
    assert "larger than 0" in message
