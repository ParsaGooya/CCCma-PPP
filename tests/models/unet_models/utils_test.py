from types import SimpleNamespace

import pytest
import torch

from cccma_ppp.models.layers.conv import TensorMask
from cccma_ppp.models.unet_models.utils import (
    _repeat_tensor_mask,
    _unet_config_checks,
)


def make_config(
    *,
    channels=None,
    bottleneck_dim=None,
    mask_fraction_threshold=0.5,
    output_block_hidden_channels=None,
    generator=None,
):
    if channels is None:
        channels = [8, 16, 32]

    return SimpleNamespace(
        channels=channels,
        bottleneck_dim=bottleneck_dim,
        mask_fraction_threshold=mask_fraction_threshold,
        output_block_hidden_channels=output_block_hidden_channels,
        GENERATOR=generator,
    )


def make_generator(
    num_training_noise_samples=1,
):
    return SimpleNamespace(
        num_training_noise_samples=num_training_noise_samples,
    )


# ---------------------------------------------------------------------------
# _unet_config_checks: valid configurations
# ---------------------------------------------------------------------------


def test_unet_config_checks_valid_minimal_config():
    config = make_config()

    assert _unet_config_checks(config) is None


def test_unet_config_checks_accepts_single_channel_level():
    config = make_config(
        channels=[8],
    )

    assert _unet_config_checks(config) is None


def test_unet_config_checks_accepts_multiple_channel_levels():
    config = make_config(
        channels=[4, 8, 16, 32, 64],
    )

    assert _unet_config_checks(config) is None


def test_unet_config_checks_accepts_positive_bottleneck_dimension():
    config = make_config(
        bottleneck_dim=64,
    )

    assert _unet_config_checks(config) is None


def test_unet_config_checks_accepts_none_bottleneck_dimension():
    config = make_config(
        bottleneck_dim=None,
    )

    assert _unet_config_checks(config) is None


@pytest.mark.parametrize(
    "threshold",
    [
        0.0,
        0.1,
        0.5,
        0.9,
        1.0,
    ],
)
def test_unet_config_checks_accepts_valid_mask_thresholds(
    threshold,
):
    config = make_config(
        mask_fraction_threshold=threshold,
    )

    assert _unet_config_checks(config) is None


@pytest.mark.parametrize(
    "hidden_channels",
    [
        None,
        1,
        8,
        64,
    ],
)
def test_unet_config_checks_accepts_valid_output_hidden_channels(
    hidden_channels,
):
    config = make_config(
        output_block_hidden_channels=hidden_channels,
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
        generator=make_generator(
            num_training_noise_samples=num_training_noise_samples,
        ),
    )

    assert _unet_config_checks(config) is None


def test_unet_config_checks_accepts_generator_none():
    config = make_config(
        generator=None,
    )

    assert _unet_config_checks(config) is None


def test_unet_config_checks_does_not_mutate_config():
    generator = make_generator(
        num_training_noise_samples=3,
    )
    channels = [8, 16, 32]

    config = make_config(
        channels=channels,
        bottleneck_dim=64,
        mask_fraction_threshold=0.75,
        output_block_hidden_channels=16,
        generator=generator,
    )

    _unet_config_checks(config)

    assert config.channels is channels
    assert config.channels == [8, 16, 32]
    assert config.bottleneck_dim == 64
    assert config.mask_fraction_threshold == pytest.approx(0.75)
    assert config.output_block_hidden_channels == 16
    assert config.GENERATOR is generator
    assert config.GENERATOR.num_training_noise_samples == 3


# ---------------------------------------------------------------------------
# _unet_config_checks: channels
# ---------------------------------------------------------------------------


def test_unet_config_checks_rejects_empty_channels():
    config = make_config(
        channels=[],
    )

    with pytest.raises(
        ValueError,
        match="channels must contain at least one encoder width",
    ):
        _unet_config_checks(config)


@pytest.mark.parametrize(
    "channels",
    [
        [0],
        [-1],
        [-10],
        [8, 0, 16],
        [8, -1, 16],
    ],
)
def test_unet_config_checks_rejects_nonpositive_channels(
    channels,
):
    config = make_config(
        channels=channels,
    )

    with pytest.raises(
        ValueError,
        match="All channel sizes must be positive integers",
    ):
        _unet_config_checks(config)


def test_unet_config_checks_channel_validation_precedes_other_checks():
    config = make_config(
        channels=[],
        bottleneck_dim=-1,
        mask_fraction_threshold=-1,
        output_block_hidden_channels=-1,
        generator=make_generator(
            num_training_noise_samples=0,
        ),
    )

    with pytest.raises(
        ValueError,
        match="channels must contain at least one encoder width",
    ):
        _unet_config_checks(config)


def test_unet_config_checks_channel_values_checked_before_bottleneck():
    config = make_config(
        channels=[8, 0],
        bottleneck_dim=-1,
    )

    with pytest.raises(
        ValueError,
        match="All channel sizes must be positive integers",
    ):
        _unet_config_checks(config)


# ---------------------------------------------------------------------------
# _unet_config_checks: bottleneck
# ---------------------------------------------------------------------------


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
        match="bottleneck_dim must be positive",
    ):
        _unet_config_checks(config)


def test_unet_config_checks_bottleneck_checked_before_mask_threshold():
    config = make_config(
        bottleneck_dim=0,
        mask_fraction_threshold=-1,
    )

    with pytest.raises(
        ValueError,
        match="bottleneck_dim must be positive",
    ):
        _unet_config_checks(config)


# ---------------------------------------------------------------------------
# _unet_config_checks: mask fraction threshold
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "threshold",
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
    threshold,
):
    config = make_config(
        mask_fraction_threshold=threshold,
    )

    with pytest.raises(
        ValueError,
        match="mask_fraction_threshold must be between 0 and 1",
    ):
        _unet_config_checks(config)


def test_unet_config_checks_mask_threshold_checked_before_output_hidden_channels():
    config = make_config(
        mask_fraction_threshold=-1,
        output_block_hidden_channels=0,
    )

    with pytest.raises(
        ValueError,
        match="mask_fraction_threshold must be between 0 and 1",
    ):
        _unet_config_checks(config)


# ---------------------------------------------------------------------------
# _unet_config_checks: output hidden channels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hidden_channels",
    [
        0,
        -1,
        -10,
    ],
)
def test_unet_config_checks_rejects_nonpositive_output_hidden_channels(
    hidden_channels,
):
    config = make_config(
        output_block_hidden_channels=hidden_channels,
    )

    with pytest.raises(
        ValueError,
        match="output_block_hidden_channels must be positive",
    ):
        _unet_config_checks(config)


def test_unet_config_checks_output_hidden_channels_checked_before_generator():
    config = make_config(
        output_block_hidden_channels=0,
        generator=make_generator(
            num_training_noise_samples=0,
        ),
    )

    with pytest.raises(
        ValueError,
        match="output_block_hidden_channels must be positive",
    ):
        _unet_config_checks(config)


# ---------------------------------------------------------------------------
# _unet_config_checks: generator
# ---------------------------------------------------------------------------


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
        generator=make_generator(
            num_training_noise_samples=num_training_noise_samples,
        ),
    )

    with pytest.raises(
        ValueError,
        match="num_training_noise_samples.*larger than 0",
    ):
        _unet_config_checks(config)


def test_unet_config_checks_generator_error_message():
    config = make_config(
        generator=make_generator(
            num_training_noise_samples=0,
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "For Generator to work a num_training_noise_samples "
            "must be chosen that is larger than 0"
        ),
    ):
        _unet_config_checks(config)


# ---------------------------------------------------------------------------
# _repeat_tensor_mask: tensor only
# ---------------------------------------------------------------------------


def test_repeat_tensor_mask_returns_tensor_mask():
    value = TensorMask(
        tensor=torch.randn(2, 3, 4, 5),
        mask=None,
    )

    result = _repeat_tensor_mask(
        value,
        repeats=2,
    )

    assert isinstance(result, TensorMask)


def test_repeat_tensor_mask_repeats_tensor_batch_dimension():
    tensor = torch.randn(2, 3, 4, 5)

    result = _repeat_tensor_mask(
        TensorMask(
            tensor=tensor,
            mask=None,
        ),
        repeats=3,
    )

    assert result.tensor.shape == (6, 3, 4, 5)
    assert result.mask is None


def test_repeat_tensor_mask_repeats_samples_consecutively():
    tensor = torch.tensor(
        [
            [[[1.0]]],
            [[[2.0]]],
            [[[3.0]]],
        ]
    )

    result = _repeat_tensor_mask(
        TensorMask(
            tensor=tensor,
            mask=None,
        ),
        repeats=2,
    )

    expected = torch.tensor(
        [
            [[[1.0]]],
            [[[1.0]]],
            [[[2.0]]],
            [[[2.0]]],
            [[[3.0]]],
            [[[3.0]]],
        ]
    )

    torch.testing.assert_close(
        result.tensor,
        expected,
    )


def test_repeat_tensor_mask_once_returns_equal_tensor():
    tensor = torch.randn(2, 3, 4, 5)

    result = _repeat_tensor_mask(
        TensorMask(
            tensor=tensor,
            mask=None,
        ),
        repeats=1,
    )

    torch.testing.assert_close(
        result.tensor,
        tensor,
    )
    assert result.mask is None


def test_repeat_tensor_mask_zero_repeats_returns_empty_batch():
    tensor = torch.randn(2, 3, 4, 5)

    result = _repeat_tensor_mask(
        TensorMask(
            tensor=tensor,
            mask=None,
        ),
        repeats=0,
    )

    assert result.tensor.shape == (0, 3, 4, 5)
    assert result.mask is None


def test_repeat_tensor_mask_preserves_tensor_dtype():
    tensor = torch.randn(
        2,
        3,
        4,
        5,
        dtype=torch.float64,
    )

    result = _repeat_tensor_mask(
        TensorMask(
            tensor=tensor,
            mask=None,
        ),
        repeats=3,
    )

    assert result.tensor.dtype == torch.float64


def test_repeat_tensor_mask_preserves_tensor_device():
    tensor = torch.randn(2, 3, 4, 5)

    result = _repeat_tensor_mask(
        TensorMask(
            tensor=tensor,
            mask=None,
        ),
        repeats=3,
    )

    assert result.tensor.device == tensor.device


def test_repeat_tensor_mask_returns_new_tensor_mask_instance():
    value = TensorMask(
        tensor=torch.randn(2, 3, 4, 5),
        mask=None,
    )

    result = _repeat_tensor_mask(
        value,
        repeats=1,
    )

    assert result is not value


# ---------------------------------------------------------------------------
# _repeat_tensor_mask: tensor and mask
# ---------------------------------------------------------------------------


def test_repeat_tensor_mask_repeats_mask():
    tensor = torch.randn(2, 3, 4, 5)
    mask = torch.ones(2, 1, 4, 5)

    result = _repeat_tensor_mask(
        TensorMask(
            tensor=tensor,
            mask=mask,
        ),
        repeats=3,
    )

    assert result.tensor.shape == (6, 3, 4, 5)
    assert result.mask.shape == (6, 1, 4, 5)


def test_repeat_tensor_mask_repeats_mask_consecutively():
    tensor = torch.randn(2, 3, 1, 1)
    mask = torch.tensor(
        [
            [[[0.0]]],
            [[[1.0]]],
        ]
    )

    result = _repeat_tensor_mask(
        TensorMask(
            tensor=tensor,
            mask=mask,
        ),
        repeats=3,
    )

    expected = torch.tensor(
        [
            [[[0.0]]],
            [[[0.0]]],
            [[[0.0]]],
            [[[1.0]]],
            [[[1.0]]],
            [[[1.0]]],
        ]
    )

    torch.testing.assert_close(
        result.mask,
        expected,
    )


def test_repeat_tensor_mask_tensor_and_mask_order_match():
    tensor = torch.tensor(
        [
            [[[1.0]]],
            [[[2.0]]],
        ]
    )
    mask = torch.tensor(
        [
            [[[10.0]]],
            [[[20.0]]],
        ]
    )

    result = _repeat_tensor_mask(
        TensorMask(
            tensor=tensor,
            mask=mask,
        ),
        repeats=2,
    )

    expected_tensor = torch.tensor(
        [
            [[[1.0]]],
            [[[1.0]]],
            [[[2.0]]],
            [[[2.0]]],
        ]
    )
    expected_mask = torch.tensor(
        [
            [[[10.0]]],
            [[[10.0]]],
            [[[20.0]]],
            [[[20.0]]],
        ]
    )

    torch.testing.assert_close(
        result.tensor,
        expected_tensor,
    )
    torch.testing.assert_close(
        result.mask,
        expected_mask,
    )


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float64,
        torch.int32,
        torch.int64,
        torch.bool,
    ],
)
def test_repeat_tensor_mask_preserves_mask_dtype(dtype):
    tensor = torch.randn(2, 3, 4, 5)
    mask = torch.ones(
        2,
        1,
        4,
        5,
        dtype=dtype,
    )

    result = _repeat_tensor_mask(
        TensorMask(
            tensor=tensor,
            mask=mask,
        ),
        repeats=2,
    )

    assert result.mask.dtype == dtype


def test_repeat_tensor_mask_preserves_mask_device():
    tensor = torch.randn(2, 3, 4, 5)
    mask = torch.ones(2, 1, 4, 5)

    result = _repeat_tensor_mask(
        TensorMask(
            tensor=tensor,
            mask=mask,
        ),
        repeats=2,
    )

    assert result.mask.device == mask.device


def test_repeat_tensor_mask_preserves_gradients():
    tensor = torch.randn(
        2,
        3,
        4,
        5,
        requires_grad=True,
    )

    result = _repeat_tensor_mask(
        TensorMask(
            tensor=tensor,
            mask=None,
        ),
        repeats=3,
    )

    result.tensor.sum().backward()

    assert tensor.grad is not None
    torch.testing.assert_close(
        tensor.grad,
        torch.full_like(tensor, 3.0),
    )


def test_repeat_tensor_mask_preserves_mask_gradients():
    tensor = torch.randn(2, 3, 4, 5)
    mask = torch.randn(
        2,
        1,
        4,
        5,
        requires_grad=True,
    )

    result = _repeat_tensor_mask(
        TensorMask(
            tensor=tensor,
            mask=mask,
        ),
        repeats=2,
    )

    result.mask.sum().backward()

    assert mask.grad is not None
    torch.testing.assert_close(
        mask.grad,
        torch.full_like(mask, 2.0),
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_repeat_tensor_mask_on_cuda():
    tensor = torch.randn(
        2,
        3,
        4,
        5,
        device="cuda",
    )
    mask = torch.ones(
        2,
        1,
        4,
        5,
        device="cuda",
    )

    result = _repeat_tensor_mask(
        TensorMask(
            tensor=tensor,
            mask=mask,
        ),
        repeats=2,
    )

    assert result.tensor.device.type == "cuda"
    assert result.mask.device.type == "cuda"
    assert result.tensor.shape == (4, 3, 4, 5)
    assert result.mask.shape == (4, 1, 4, 5)
