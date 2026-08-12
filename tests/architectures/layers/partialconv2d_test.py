import pytest
import torch
import torch.nn as nn

from cccma_ppp.architectures.layers.partialconv2d import PartialConv2d


def make_input(
    batch_size=2,
    channels=3,
    height=8,
    width=8,
    *,
    value=None,
):
    if value is None:
        return torch.randn(
            batch_size,
            channels,
            height,
            width,
        )

    return torch.full(
        (
            batch_size,
            channels,
            height,
            width,
        ),
        value,
        dtype=torch.float32,
    )


@pytest.mark.pruned
def test_partial_conv_is_conv2d():
    layer = PartialConv2d(
        3,
        4,
        kernel_size=3,
    )

    assert isinstance(layer, torch.nn.Conv2d)


@pytest.mark.pruned
def test_partial_conv_default_options():
    layer = PartialConv2d(
        3,
        4,
        kernel_size=3,
    )

    assert layer.multi_channel is False
    assert layer.return_mask is False


@pytest.mark.pruned
def test_partial_conv_custom_options():
    layer = PartialConv2d(
        3,
        4,
        kernel_size=3,
        multi_channel=True,
        return_mask=True,
    )

    assert layer.multi_channel is True
    assert layer.return_mask is True


@pytest.mark.pruned
def test_single_channel_mask_updater_shape():
    layer = PartialConv2d(
        3,
        4,
        kernel_size=3,
        multi_channel=False,
    )

    assert layer.weight_maskUpdater.shape == (
        1,
        1,
        3,
        3,
    )


@pytest.mark.pruned
def test_multi_channel_mask_updater_shape():
    layer = PartialConv2d(
        3,
        4,
        kernel_size=3,
        multi_channel=True,
    )

    assert layer.weight_maskUpdater.shape == (
        4,
        3,
        3,
        3,
    )


@pytest.mark.pruned
def test_mask_updater_contains_ones():
    layer = PartialConv2d(
        3,
        4,
        kernel_size=3,
        multi_channel=True,
    )

    torch.testing.assert_close(
        layer.weight_maskUpdater,
        torch.ones(4, 3, 3, 3),
    )


@pytest.mark.pruned
def test_mask_updater_registered_as_buffer():
    layer = PartialConv2d(
        3,
        4,
        kernel_size=3,
    )

    buffers = dict(layer.named_buffers())

    assert "weight_maskUpdater" in buffers
    assert buffers["weight_maskUpdater"] is layer.weight_maskUpdater


@pytest.mark.pruned
def test_mask_updater_is_nonpersistent():
    layer = PartialConv2d(
        3,
        4,
        kernel_size=3,
    )

    assert "weight_maskUpdater" not in layer.state_dict()


@pytest.mark.pruned
def test_single_channel_slide_window_size():
    layer = PartialConv2d(
        3,
        4,
        kernel_size=3,
        multi_channel=False,
    )

    assert layer.slide_winsize == 9


@pytest.mark.pruned
def test_multi_channel_slide_window_size():
    layer = PartialConv2d(
        3,
        4,
        kernel_size=3,
        multi_channel=True,
    )

    assert layer.slide_winsize == 27


@pytest.mark.pruned
@pytest.mark.parametrize(
    ("kernel_size", "expected"),
    [
        (1, 1),
        (3, 9),
        (5, 25),
    ],
)
def test_single_channel_slide_window_size_for_kernels(
    kernel_size,
    expected,
):
    layer = PartialConv2d(
        2,
        4,
        kernel_size=kernel_size,
        multi_channel=False,
    )

    assert layer.slide_winsize == expected


@pytest.mark.pruned
@pytest.mark.parametrize(
    ("in_channels", "kernel_size", "expected"),
    [
        (1, 1, 1),
        (2, 3, 18),
        (3, 3, 27),
        (4, 5, 100),
    ],
)
def test_multi_channel_slide_window_size_for_kernels(
    in_channels,
    kernel_size,
    expected,
):
    layer = PartialConv2d(
        in_channels,
        5,
        kernel_size=kernel_size,
        multi_channel=True,
    )

    assert layer.slide_winsize == expected


@pytest.mark.pruned
def test_initial_cache_state():
    layer = PartialConv2d(
        3,
        4,
        kernel_size=3,
    )

    assert layer.last_size == (
        None,
        None,
        None,
        None,
    )
    assert layer.update_mask is None
    assert layer.mask_ratio is None


@pytest.mark.pruned
@pytest.mark.parametrize(
    "shape",
    [
        (3,),
        (2, 3),
        (2, 3, 8),
        (1, 2, 3, 8, 8),
    ],
)
def test_forward_requires_four_dimensional_input(shape):
    layer = PartialConv2d(
        3,
        4,
        kernel_size=3,
    )

    with pytest.raises(AssertionError):
        layer(torch.randn(*shape))


@pytest.mark.pruned
def test_forward_without_mask_shape():
    layer = PartialConv2d(
        3,
        4,
        kernel_size=3,
        padding=1,
    )

    input_tensor = make_input()
    result = layer(input_tensor)

    assert result.shape == (
        2,
        4,
        8,
        8,
    )


@pytest.mark.pruned
def test_forward_without_mask_updates_last_size():
    layer = PartialConv2d(
        3,
        4,
        kernel_size=3,
        padding=1,
    )

    input_tensor = make_input()
    layer(input_tensor)

    assert layer.last_size == tuple(input_tensor.shape)


@pytest.mark.pruned
def test_forward_without_mask_initializes_update_mask():
    layer = PartialConv2d(
        3,
        4,
        kernel_size=3,
        padding=1,
    )

    layer(make_input())

    assert layer.update_mask is not None
    assert layer.mask_ratio is not None


@pytest.mark.pruned
def test_forward_without_mask_return_mask_false_returns_tensor():
    layer = PartialConv2d(
        3,
        4,
        kernel_size=3,
        padding=1,
        return_mask=False,
    )

    result = layer(make_input())

    assert isinstance(result, torch.Tensor)


@pytest.mark.pruned
def test_forward_without_mask_return_mask_true_returns_tuple():
    layer = PartialConv2d(
        3,
        4,
        kernel_size=3,
        padding=1,
        return_mask=True,
    )

    result = layer(make_input())

    assert isinstance(result, tuple)
    assert len(result) == 2


@pytest.mark.pruned
def test_forward_without_mask_returns_output_and_mask_shapes():
    layer = PartialConv2d(
        3,
        4,
        kernel_size=3,
        padding=1,
        return_mask=True,
    )

    output, mask = layer(make_input())

    assert output.shape == (
        2,
        4,
        8,
        8,
    )
    assert mask.shape == (
        1,
        1,
        8,
        8,
    )


@pytest.mark.pruned
def test_multi_channel_forward_without_mask_returns_batch_mask():
    layer = PartialConv2d(
        3,
        4,
        kernel_size=3,
        padding=1,
        multi_channel=True,
        return_mask=True,
    )

    output, mask = layer(make_input())

    assert output.shape == (
        2,
        4,
        8,
        8,
    )
    assert mask.shape == (
        2,
        4,
        8,
        8,
    )


@pytest.mark.pruned
def test_forward_without_mask_preserves_dtype():
    layer = PartialConv2d(
        3,
        4,
        kernel_size=3,
        padding=1,
    ).double()

    input_tensor = torch.randn(
        2,
        3,
        8,
        8,
        dtype=torch.float64,
    )

    result = layer(input_tensor)

    assert result.dtype == torch.float64
    assert layer.update_mask.dtype == torch.float64
    assert layer.mask_ratio.dtype == torch.float64


@pytest.mark.pruned
def test_single_channel_forward_with_mask():
    layer = PartialConv2d(
        3,
        4,
        kernel_size=3,
        padding=1,
        multi_channel=False,
        return_mask=True,
    )

    input_tensor = make_input()
    mask = torch.ones(1, 1, 8, 8)

    output, output_mask = layer(
        input_tensor,
        mask,
    )

    assert output.shape == (
        2,
        4,
        8,
        8,
    )
    assert output_mask.shape == (
        1,
        1,
        8,
        8,
    )


@pytest.mark.pruned
def test_multi_channel_forward_with_mask():
    layer = PartialConv2d(
        3,
        4,
        kernel_size=3,
        padding=1,
        multi_channel=True,
        return_mask=True,
    )

    input_tensor = make_input()
    mask = torch.ones_like(input_tensor)

    output, output_mask = layer(
        input_tensor,
        mask,
    )

    assert output.shape == (
        2,
        4,
        8,
        8,
    )
    assert output_mask.shape == (
        2,
        4,
        8,
        8,
    )


@pytest.mark.pruned
def test_explicit_mask_forces_cache_recomputation():
    layer = PartialConv2d(
        3,
        4,
        kernel_size=3,
        padding=1,
        multi_channel=True,
        return_mask=True,
    )

    input_tensor = make_input()
    first_mask = torch.ones_like(input_tensor)
    second_mask = torch.zeros_like(input_tensor)

    _, first_update = layer(
        input_tensor,
        first_mask,
    )
    _, second_update = layer(
        input_tensor,
        second_mask,
    )

    assert torch.any(first_update > 0)
    assert torch.all(second_update == 0)


@pytest.mark.pruned
def test_different_input_shape_forces_cache_recomputation():
    layer = PartialConv2d(
        3,
        4,
        kernel_size=3,
        padding=1,
        return_mask=True,
    )

    first = make_input(
        height=8,
        width=8,
    )
    second = make_input(
        height=6,
        width=10,
    )

    layer(first)
    second_output, second_mask = layer(second)

    assert layer.last_size == tuple(second.shape)
    assert second_output.shape == (
        2,
        4,
        6,
        10,
    )
    assert second_mask.shape == (
        1,
        1,
        6,
        10,
    )


@pytest.mark.pruned
def test_same_shape_without_mask_reuses_cached_mask():
    layer = PartialConv2d(
        3,
        4,
        kernel_size=3,
        padding=1,
        return_mask=True,
    )

    first_input = make_input()
    second_input = make_input()

    layer(first_input)

    first_update_mask = layer.update_mask
    first_mask_ratio = layer.mask_ratio

    layer(second_input)

    assert layer.update_mask is first_update_mask
    assert layer.mask_ratio is first_mask_ratio


@pytest.mark.pruned
def test_zero_mask_produces_zero_update_mask():
    layer = PartialConv2d(
        3,
        4,
        kernel_size=3,
        padding=1,
        multi_channel=True,
        return_mask=True,
    )

    input_tensor = make_input()
    mask = torch.zeros_like(input_tensor)

    _, update_mask = layer(
        input_tensor,
        mask,
    )

    assert torch.count_nonzero(update_mask) == 0


@pytest.mark.pruned
def test_zero_mask_produces_zero_output_without_bias():
    layer = PartialConv2d(
        3,
        4,
        kernel_size=3,
        padding=1,
        bias=False,
        multi_channel=True,
    )

    input_tensor = make_input()
    mask = torch.zeros_like(input_tensor)

    output = layer(
        input_tensor,
        mask,
    )

    torch.testing.assert_close(
        output,
        torch.zeros_like(output),
    )


@pytest.mark.pruned
def test_zero_mask_produces_zero_output_with_bias():
    layer = PartialConv2d(
        3,
        4,
        kernel_size=3,
        padding=1,
        bias=True,
        multi_channel=True,
    )

    with torch.no_grad():
        layer.bias.fill_(5.0)

    input_tensor = make_input()
    mask = torch.zeros_like(input_tensor)

    output = layer(
        input_tensor,
        mask,
    )

    torch.testing.assert_close(
        output,
        torch.zeros_like(output),
    )


@pytest.mark.pruned
def test_update_mask_is_clamped_to_binary_range():
    layer = PartialConv2d(
        3,
        4,
        kernel_size=3,
        padding=1,
        multi_channel=True,
    )

    mask = torch.ones(2, 3, 8, 8)
    layer(
        make_input(),
        mask,
    )

    assert layer.update_mask.min().item() >= 0.0
    assert layer.update_mask.max().item() <= 1.0


@pytest.mark.pruned
def test_update_mask_contains_only_zero_or_one():
    layer = PartialConv2d(
        1,
        1,
        kernel_size=3,
        padding=1,
        multi_channel=True,
    )

    mask = torch.zeros(1, 1, 4, 4)
    mask[:, :, 1, 1] = 1

    layer(
        torch.ones(1, 1, 4, 4),
        mask,
    )

    unique_values = set(layer.update_mask.unique().tolist())

    assert unique_values.issubset(
        {
            0.0,
            1.0,
        }
    )


@pytest.mark.pruned
def test_mask_ratio_zero_where_update_mask_is_zero():
    layer = PartialConv2d(
        1,
        1,
        kernel_size=3,
        padding=1,
        multi_channel=True,
    )

    input_tensor = torch.ones(1, 1, 4, 4)
    mask = torch.zeros_like(input_tensor)
    mask[:, :, 0, 0] = 1

    layer(
        input_tensor,
        mask,
    )

    invalid = layer.update_mask == 0

    assert torch.all(layer.mask_ratio[invalid] == 0)


@pytest.mark.pruned
def test_full_mask_center_ratio_is_one():
    layer = PartialConv2d(
        1,
        1,
        kernel_size=3,
        padding=1,
        bias=False,
        multi_channel=True,
    )

    input_tensor = torch.ones(1, 1, 5, 5)
    mask = torch.ones_like(input_tensor)

    layer(
        input_tensor,
        mask,
    )

    assert layer.mask_ratio[0, 0, 2, 2].item() == pytest.approx(
        1.0,
    )


@pytest.mark.pruned
def test_missing_values_increase_mask_ratio():
    layer = PartialConv2d(
        1,
        1,
        kernel_size=3,
        padding=1,
        bias=False,
        multi_channel=True,
    )

    input_tensor = torch.ones(1, 1, 5, 5)
    mask = torch.ones_like(input_tensor)
    mask[:, :, 2, 2] = 0

    layer(
        input_tensor,
        mask,
    )

    assert layer.mask_ratio[0, 0, 2, 2].item() == pytest.approx(
        9.0 / 8.0,
    )


@pytest.mark.pruned
def test_single_valid_value_has_window_size_ratio():
    layer = PartialConv2d(
        1,
        1,
        kernel_size=3,
        padding=0,
        bias=False,
        multi_channel=True,
    )

    input_tensor = torch.ones(1, 1, 3, 3)
    mask = torch.zeros_like(input_tensor)
    mask[:, :, 1, 1] = 1

    layer(
        input_tensor,
        mask,
    )

    assert layer.update_mask.item() == pytest.approx(1.0)
    assert layer.mask_ratio.item() == pytest.approx(9.0)


@pytest.mark.pruned
def test_full_mask_matches_standard_conv_without_padding():
    partial = PartialConv2d(
        1,
        1,
        kernel_size=3,
        padding=0,
        bias=False,
        multi_channel=True,
    )
    standard = nn.Conv2d(
        1,
        1,
        kernel_size=3,
        padding=0,
        bias=False,
    )

    with torch.no_grad():
        partial.weight.fill_(2.0)
        standard.weight.copy_(partial.weight)

    input_tensor = torch.randn(2, 1, 5, 5)
    mask = torch.ones_like(input_tensor)

    partial_output = partial(
        input_tensor,
        mask,
    )
    standard_output = standard(input_tensor)

    torch.testing.assert_close(
        partial_output,
        standard_output,
    )


@pytest.mark.pruned
def test_forward_multiplies_input_by_mask():
    layer = PartialConv2d(
        1,
        1,
        kernel_size=1,
        bias=False,
        multi_channel=True,
    )

    with torch.no_grad():
        layer.weight.fill_(1.0)

    input_tensor = torch.tensor(
        [
            [
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ]
            ]
        ]
    )
    mask = torch.tensor(
        [
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            ]
        ]
    )

    output = layer(
        input_tensor,
        mask,
    )

    expected = torch.tensor(
        [
            [
                [
                    [1.0, 0.0],
                    [0.0, 4.0],
                ]
            ]
        ]
    )

    torch.testing.assert_close(
        output,
        expected,
    )


@pytest.mark.pruned
def test_forward_without_mask_does_not_multiply_input():
    layer = PartialConv2d(
        1,
        1,
        kernel_size=1,
        bias=False,
        multi_channel=True,
    )

    with torch.no_grad():
        layer.weight.fill_(1.0)

    input_tensor = torch.tensor(
        [
            [
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ]
            ]
        ]
    )

    output = layer(input_tensor)

    torch.testing.assert_close(
        output,
        input_tensor,
    )


@pytest.mark.pruned
def test_partial_mask_rescales_valid_values():
    layer = PartialConv2d(
        1,
        1,
        kernel_size=2,
        bias=False,
        multi_channel=True,
    )

    with torch.no_grad():
        layer.weight.fill_(1.0)

    input_tensor = torch.ones(1, 1, 2, 2)
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

    output = layer(
        input_tensor,
        mask,
    )

    assert output.item() == pytest.approx(4.0)


@pytest.mark.pruned
def test_bias_is_preserved_after_mask_rescaling():
    layer = PartialConv2d(
        1,
        1,
        kernel_size=2,
        bias=True,
        multi_channel=True,
    )

    with torch.no_grad():
        layer.weight.fill_(1.0)
        layer.bias.fill_(3.0)

    input_tensor = torch.ones(1, 1, 2, 2)
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

    output = layer(
        input_tensor,
        mask,
    )

    assert output.item() == pytest.approx(7.0)


@pytest.mark.pruned
def test_bias_only_output_with_valid_mask():
    layer = PartialConv2d(
        1,
        1,
        kernel_size=1,
        bias=True,
        multi_channel=True,
    )

    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.fill_(3.0)

    input_tensor = torch.ones(1, 1, 2, 2)
    mask = torch.ones_like(input_tensor)

    output = layer(
        input_tensor,
        mask,
    )

    torch.testing.assert_close(
        output,
        torch.full_like(output, 3.0),
    )


@pytest.mark.pruned
def test_dilation_changes_output_shape():
    layer = PartialConv2d(
        1,
        2,
        kernel_size=3,
        dilation=2,
        padding=2,
        multi_channel=True,
        return_mask=True,
    )

    input_tensor = torch.randn(2, 1, 8, 8)
    mask = torch.ones_like(input_tensor)

    output, output_mask = layer(
        input_tensor,
        mask,
    )

    assert output.shape == (
        2,
        2,
        8,
        8,
    )
    assert output_mask.shape == output.shape


@pytest.mark.pruned
def test_non_square_kernel_mask_updater_shape():
    layer = PartialConv2d(
        2,
        3,
        kernel_size=(3, 5),
        multi_channel=True,
    )

    assert layer.weight_maskUpdater.shape == (
        3,
        2,
        3,
        5,
    )
    assert layer.slide_winsize == 30


@pytest.mark.pruned
def test_non_square_kernel_forward():
    layer = PartialConv2d(
        2,
        3,
        kernel_size=(3, 5),
        padding=(1, 2),
        multi_channel=True,
        return_mask=True,
    )

    input_tensor = torch.randn(2, 2, 8, 10)
    mask = torch.ones_like(input_tensor)

    output, output_mask = layer(
        input_tensor,
        mask,
    )

    assert output.shape == (
        2,
        3,
        8,
        10,
    )
    assert output_mask.shape == output.shape


@pytest.mark.pruned
def test_multiple_output_channels_have_independent_masks():
    layer = PartialConv2d(
        2,
        3,
        kernel_size=1,
        multi_channel=True,
        return_mask=True,
    )

    input_tensor = torch.ones(1, 2, 2, 2)
    mask = torch.ones_like(input_tensor)

    _, output_mask = layer(
        input_tensor,
        mask,
    )

    assert output_mask.shape[1] == 3
    torch.testing.assert_close(
        output_mask[:, 0],
        output_mask[:, 1],
    )
    torch.testing.assert_close(
        output_mask[:, 1],
        output_mask[:, 2],
    )


@pytest.mark.pruned
def test_forward_supports_backward():
    layer = PartialConv2d(
        3,
        4,
        kernel_size=3,
        padding=1,
        multi_channel=True,
    )

    input_tensor = torch.randn(
        2,
        3,
        8,
        8,
        requires_grad=True,
    )
    mask = torch.ones_like(input_tensor)

    output = layer(
        input_tensor,
        mask,
    )
    output.sum().backward()

    assert input_tensor.grad is not None
    assert layer.weight.grad is not None


@pytest.mark.pruned
def test_mask_does_not_require_gradients():
    layer = PartialConv2d(
        3,
        4,
        kernel_size=3,
        padding=1,
        multi_channel=True,
    )

    input_tensor = torch.randn(
        2,
        3,
        8,
        8,
        requires_grad=True,
    )
    mask = torch.ones(
        2,
        3,
        8,
        8,
        requires_grad=True,
    )

    output = layer(
        input_tensor,
        mask,
    )
    output.sum().backward()

    assert input_tensor.grad is not None
    assert mask.grad is not None
    assert layer.update_mask.requires_grad is False
    assert layer.mask_ratio.requires_grad is False


@pytest.mark.pruned
def test_return_mask_false_returns_only_output():
    layer = PartialConv2d(
        1,
        2,
        kernel_size=3,
        padding=1,
        multi_channel=True,
        return_mask=False,
    )

    output = layer(
        torch.ones(1, 1, 4, 4),
        torch.ones(1, 1, 4, 4),
    )

    assert isinstance(output, torch.Tensor)
    assert output.shape == (
        1,
        2,
        4,
        4,
    )


@pytest.mark.pruned
def test_return_mask_true_returns_cached_update_mask():
    layer = PartialConv2d(
        1,
        2,
        kernel_size=3,
        padding=1,
        multi_channel=True,
        return_mask=True,
    )

    output, returned_mask = layer(
        torch.ones(1, 1, 4, 4),
        torch.ones(1, 1, 4, 4),
    )

    assert output.shape == (
        1,
        2,
        4,
        4,
    )
    assert returned_mask is layer.update_mask
