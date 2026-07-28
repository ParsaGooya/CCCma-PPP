import pytest
import torch
import torch.nn.functional as F

import cccma_ppp.models.layers.utils as module
from cccma_ppp.models.layers.utils import (
    _broadcast_mask,
    _expand_mask,
    _get_normal,
    _merge_masks,
    _noise_injection,
    _resize_mask,
    _resize_tensor,
    _same_padding,
    _sample,
    align_to_skip,
    padd,
)


# ---------------------------------------------------------------------------
# _same_padding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kernel_size", "expected"),
    [
        (1, 0),
        (3, 1),
        (5, 2),
        (7, 3),
        (9, 4),
    ],
)
def test_same_padding_valid_kernel_sizes(
    kernel_size,
    expected,
):
    assert _same_padding(kernel_size) == expected


@pytest.mark.parametrize(
    "kernel_size",
    [
        -5,
        -2,
        -1,
        0,
        2,
        4,
        6,
    ],
)
def test_same_padding_rejects_invalid_kernel_sizes(
    kernel_size,
):
    with pytest.raises(
        ValueError,
        match="positive odd integers",
    ):
        _same_padding(kernel_size)


# ---------------------------------------------------------------------------
# align_to_skip
# ---------------------------------------------------------------------------


def test_align_to_skip_returns_same_tensor_when_shapes_match():
    tensor = torch.randn(2, 3, 8, 10)
    skip = torch.randn(2, 4, 8, 10)

    result = align_to_skip(
        tensor,
        skip,
        mode="strict",
    )

    assert result is tensor


def test_align_to_skip_resize():
    tensor = torch.randn(2, 3, 4, 5)
    skip = torch.randn(2, 4, 8, 10)

    result = align_to_skip(
        tensor,
        skip,
        mode="resize",
    )

    expected = F.interpolate(
        tensor,
        size=(8, 10),
        mode="bilinear",
        align_corners=False,
    )

    assert result.shape == (2, 3, 8, 10)
    torch.testing.assert_close(result, expected)


def test_align_to_skip_resize_calls_interpolate(
    monkeypatch,
):
    tensor = torch.randn(2, 3, 4, 5)
    skip = torch.randn(2, 4, 8, 10)
    expected = torch.randn(2, 3, 8, 10)
    captured = {}

    def fake_interpolate(
        value,
        size,
        mode,
        align_corners,
    ):
        captured["value"] = value
        captured["size"] = size
        captured["mode"] = mode
        captured["align_corners"] = align_corners
        return expected

    monkeypatch.setattr(
        module.F,
        "interpolate",
        fake_interpolate,
    )

    result = align_to_skip(
        tensor,
        skip,
        mode="resize",
    )

    assert result is expected
    assert captured == {
        "value": tensor,
        "size": torch.Size([8, 10]),
        "mode": "bilinear",
        "align_corners": False,
    }


@pytest.mark.parametrize(
    "padding_mode",
    [
        "constant",
        "reflect",
        "replicate",
        "circular",
    ],
)
def test_align_to_skip_padd(
    padding_mode,
):
    tensor = torch.randn(1, 2, 4, 4)
    skip = torch.randn(1, 3, 6, 6)

    result = align_to_skip(
        tensor,
        skip,
        mode="padd",
        padding_mode=padding_mode,
    )

    assert result.shape == (1, 2, 6, 6)


def test_align_to_skip_padd_calls_helper(
    monkeypatch,
):
    tensor = torch.randn(1, 2, 4, 4)
    skip = torch.randn(1, 3, 6, 8)
    expected = torch.randn(1, 2, 6, 8)
    captured = {}

    def fake_padd(
        value,
        target_size,
        padding_mode,
    ):
        captured["value"] = value
        captured["target_size"] = target_size
        captured["padding_mode"] = padding_mode
        return expected

    monkeypatch.setattr(
        module,
        "padd",
        fake_padd,
    )

    result = align_to_skip(
        tensor,
        skip,
        mode="padd",
        padding_mode="reflect",
    )

    assert result is expected
    assert captured == {
        "value": tensor,
        "target_size": torch.Size([6, 8]),
        "padding_mode": "reflect",
    }


def test_align_to_skip_strict_raises_for_mismatched_shapes():
    tensor = torch.randn(2, 3, 4, 5)
    skip = torch.randn(2, 4, 8, 10)

    with pytest.raises(
        RuntimeError,
        match="incompatible spatial shapes",
    ):
        align_to_skip(
            tensor,
            skip,
            mode="strict",
        )


def test_align_to_skip_strict_error_contains_shapes():
    tensor = torch.randn(2, 3, 4, 5)
    skip = torch.randn(2, 4, 8, 10)

    with pytest.raises(
        RuntimeError,
        match=r"\[4, 5\].*\[8, 10\]",
    ):
        align_to_skip(
            tensor,
            skip,
            mode="strict",
        )


@pytest.mark.parametrize(
    "mode",
    [
        "invalid",
        "",
        "interpolation",
        "crop",
    ],
)
def test_align_to_skip_rejects_unknown_mode(mode):
    tensor = torch.randn(2, 3, 4, 5)
    skip = torch.randn(2, 4, 8, 10)

    with pytest.raises(
        ValueError,
        match="Unknown alignment mode",
    ):
        align_to_skip(
            tensor,
            skip,
            mode=mode,
        )


def test_align_to_skip_does_not_validate_mode_when_shapes_match():
    tensor = torch.randn(2, 3, 8, 10)
    skip = torch.randn(2, 4, 8, 10)

    result = align_to_skip(
        tensor,
        skip,
        mode="invalid",
    )

    assert result is tensor


# ---------------------------------------------------------------------------
# padd
# ---------------------------------------------------------------------------


def test_padd_symmetric_even_difference():
    tensor = torch.ones(1, 1, 2, 2)

    result = padd(
        tensor,
        target_size=(4, 6),
        padding_mode="constant",
    )

    expected = F.pad(
        tensor,
        [2, 2, 1, 1],
        mode="constant",
    )

    assert result.shape == (1, 1, 4, 6)
    torch.testing.assert_close(result, expected)


def test_padd_asymmetric_odd_difference():
    tensor = torch.ones(1, 1, 2, 3)

    result = padd(
        tensor,
        target_size=(5, 8),
        padding_mode="constant",
    )

    expected = F.pad(
        tensor,
        [2, 3, 1, 2],
        mode="constant",
    )

    assert result.shape == (1, 1, 5, 8)
    torch.testing.assert_close(result, expected)


def test_padd_places_original_tensor_in_center():
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

    result = padd(
        tensor,
        target_size=(4, 4),
        padding_mode="constant",
    )

    torch.testing.assert_close(
        result[:, :, 1:3, 1:3],
        tensor,
    )


@pytest.mark.parametrize(
    "padding_mode",
    [
        "reflect",
        "replicate",
        "circular",
    ],
)
def test_padd_supports_nonconstant_modes(
    padding_mode,
):
    tensor = torch.arange(
        16,
        dtype=torch.float32,
    ).reshape(1, 1, 4, 4)

    result = padd(
        tensor,
        target_size=(6, 6),
        padding_mode=padding_mode,
    )

    assert result.shape == (1, 1, 6, 6)


def test_padd_same_size_returns_equal_tensor():
    tensor = torch.randn(2, 3, 4, 5)

    result = padd(
        tensor,
        target_size=(4, 5),
        padding_mode="constant",
    )

    torch.testing.assert_close(result, tensor)


def test_padd_can_crop_with_negative_padding():
    tensor = torch.arange(
        36,
        dtype=torch.float32,
    ).reshape(1, 1, 6, 6)

    result = padd(
        tensor,
        target_size=(4, 4),
        padding_mode="constant",
    )

    assert result.shape == (1, 1, 4, 4)
    torch.testing.assert_close(
        result,
        tensor[:, :, 1:5, 1:5],
    )


def test_padd_mixed_crop_and_padding():
    tensor = torch.randn(1, 2, 6, 4)

    result = padd(
        tensor,
        target_size=(4, 8),
        padding_mode="constant",
    )

    assert result.shape == (1, 2, 4, 8)


def test_padd_preserves_dtype():
    tensor = torch.randn(
        1,
        2,
        4,
        4,
        dtype=torch.float64,
    )

    result = padd(
        tensor,
        target_size=(6, 6),
        padding_mode="constant",
    )

    assert result.dtype == torch.float64


# ---------------------------------------------------------------------------
# _resize_mask
# ---------------------------------------------------------------------------


def test_resize_mask_none_returns_none():
    assert _resize_mask(None, (8, 8)) is None


def test_resize_mask_matching_size_returns_same_object():
    mask = torch.ones(2, 1, 8, 8)

    result = _resize_mask(
        mask,
        (8, 8),
    )

    assert result is mask


def test_resize_mask_uses_nearest_interpolation():
    mask = torch.tensor(
        [
            [
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                ]
            ]
        ]
    )

    result = _resize_mask(
        mask,
        (4, 4),
    )

    expected = F.interpolate(
        mask,
        size=(4, 4),
        mode="nearest",
    )

    assert result.shape == (1, 1, 4, 4)
    torch.testing.assert_close(result, expected)


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
def test_resize_mask_preserves_dtype(dtype):
    mask = torch.tensor(
        [
            [
                [
                    [0, 1],
                    [1, 0],
                ]
            ]
        ],
        dtype=dtype,
    )

    result = _resize_mask(
        mask,
        (4, 4),
    )

    assert result.dtype == dtype


def test_resize_mask_preserves_binary_values():
    mask = torch.tensor(
        [
            [
                [
                    [False, True],
                    [True, False],
                ]
            ]
        ]
    )

    result = _resize_mask(
        mask,
        (5, 7),
    )

    assert set(result.unique().tolist()).issubset(
        {
            False,
            True,
        }
    )


def test_resize_mask_downsamples():
    mask = torch.ones(2, 3, 8, 10)

    result = _resize_mask(
        mask,
        (4, 5),
    )

    assert result.shape == (2, 3, 4, 5)


# ---------------------------------------------------------------------------
# _broadcast_mask
# ---------------------------------------------------------------------------


def test_broadcast_mask_none_returns_none():
    reference = torch.randn(2, 3, 8, 8)

    assert _broadcast_mask(None, reference) is None


def test_broadcast_mask_2d_adds_batch_and_channel_dimensions():
    mask = torch.ones(8, 8)
    reference = torch.randn(1, 3, 8, 8)

    result = _broadcast_mask(
        mask,
        reference,
    )

    assert result.shape == (1, 1, 8, 8)


def test_broadcast_mask_2d_expands_batch():
    mask = torch.ones(8, 8)
    reference = torch.randn(4, 3, 8, 8)

    result = _broadcast_mask(
        mask,
        reference,
    )

    assert result.shape == (4, 1, 8, 8)


def test_broadcast_mask_3d_interpreted_as_channels_height_width():
    mask = torch.ones(3, 8, 8)
    reference = torch.randn(1, 3, 8, 8)

    result = _broadcast_mask(
        mask,
        reference,
    )

    assert result.shape == (1, 3, 8, 8)


def test_broadcast_mask_3d_expands_batch():
    mask = torch.ones(3, 8, 8)
    reference = torch.randn(2, 3, 8, 8)

    result = _broadcast_mask(
        mask,
        reference,
    )

    assert result.shape == (2, 3, 8, 8)


def test_broadcast_mask_4d_matching_shape():
    mask = torch.ones(2, 3, 8, 8)
    reference = torch.randn(2, 3, 8, 8)

    result = _broadcast_mask(
        mask,
        reference,
    )

    assert result.shape == mask.shape
    torch.testing.assert_close(result, mask)


@pytest.mark.parametrize(
    "shape",
    [
        (1,),
        (1, 2, 3, 4, 5),
        (1, 2, 3, 4, 5, 6),
    ],
)
def test_broadcast_mask_rejects_invalid_rank(shape):
    mask = torch.ones(shape)
    reference = torch.randn(2, 3, 8, 8)

    with pytest.raises(
        ValueError,
        match="Expected a 2D, 3D, or 4D mask",
    ):
        _broadcast_mask(
            mask,
            reference,
        )


def test_broadcast_mask_rejects_batch_mismatch():
    mask = torch.ones(3, 1, 8, 8)
    reference = torch.randn(2, 3, 8, 8)

    with pytest.raises(
        ValueError,
        match="Mask batch dimension does not match",
    ):
        _broadcast_mask(
            mask,
            reference,
        )


def test_broadcast_mask_batch_error_contains_sizes():
    mask = torch.ones(3, 1, 8, 8)
    reference = torch.randn(2, 3, 8, 8)

    with pytest.raises(
        ValueError,
        match="3 != 2",
    ):
        _broadcast_mask(
            mask,
            reference,
        )


def test_broadcast_mask_resizes_spatial_dimensions():
    mask = torch.ones(2, 1, 4, 5)
    reference = torch.randn(2, 3, 8, 10)

    result = _broadcast_mask(
        mask,
        reference,
    )

    assert result.shape == (2, 1, 8, 10)


def test_broadcast_mask_calls_resize_helper(
    monkeypatch,
):
    mask = torch.ones(2, 1, 4, 5)
    reference = torch.randn(2, 3, 8, 10)
    expected = torch.ones(2, 1, 8, 10)

    captured = {}

    def fake_resize(
        value,
        size,
    ):
        captured["value"] = value
        captured["size"] = size
        return expected

    monkeypatch.setattr(
        module,
        "_resize_mask",
        fake_resize,
    )

    result = _broadcast_mask(
        mask,
        reference,
    )

    assert captured["value"] is mask
    assert captured["size"] == torch.Size([8, 10])
    assert result is expected


def test_broadcast_mask_converts_dtype():
    mask = torch.ones(
        1,
        1,
        8,
        8,
        dtype=torch.bool,
    )
    reference = torch.randn(
        2,
        3,
        8,
        8,
        dtype=torch.float64,
    )

    result = _broadcast_mask(
        mask,
        reference,
    )

    assert result.dtype == torch.float64


def test_broadcast_mask_expanded_batch_values_match():
    mask = torch.tensor(
        [
            [
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                ]
            ]
        ]
    )
    reference = torch.randn(3, 2, 2, 2)

    result = _broadcast_mask(
        mask,
        reference,
    )

    for batch_index in range(3):
        torch.testing.assert_close(
            result[batch_index],
            mask[0],
        )


# ---------------------------------------------------------------------------
# _merge_masks
# ---------------------------------------------------------------------------


def test_merge_masks_both_none_returns_none():
    reference = torch.randn(2, 5, 8, 8)

    result = _merge_masks(
        None,
        None,
        out_channels=2,
        skip_channels=3,
        spatial_size=(8, 8),
        reference=reference,
    )

    assert result is None


def test_merge_masks_both_present():
    input_mask = torch.zeros(2, 2, 8, 8)
    skip_mask = torch.ones(2, 3, 8, 8)
    reference = torch.randn(2, 5, 8, 8)

    result = _merge_masks(
        input_mask,
        skip_mask,
        out_channels=2,
        skip_channels=3,
        spatial_size=(8, 8),
        reference=reference,
    )

    assert result.shape == (2, 5, 8, 8)
    torch.testing.assert_close(
        result[:, :3],
        skip_mask,
    )
    torch.testing.assert_close(
        result[:, 3:],
        input_mask,
    )


def test_merge_masks_input_none_creates_valid_input_mask():
    skip_mask = torch.zeros(2, 3, 8, 8)
    reference = torch.randn(2, 5, 8, 8)

    result = _merge_masks(
        None,
        skip_mask,
        out_channels=2,
        skip_channels=3,
        spatial_size=(8, 8),
        reference=reference,
    )

    assert result.shape == (2, 5, 8, 8)
    torch.testing.assert_close(
        result[:, :3],
        skip_mask,
    )
    torch.testing.assert_close(
        result[:, 3:],
        torch.ones(2, 2, 8, 8),
    )


def test_merge_masks_skip_none_creates_valid_skip_mask():
    input_mask = torch.zeros(2, 2, 8, 8)
    reference = torch.randn(2, 5, 8, 8)

    result = _merge_masks(
        input_mask,
        None,
        out_channels=2,
        skip_channels=3,
        spatial_size=(8, 8),
        reference=reference,
    )

    assert result.shape == (2, 5, 8, 8)
    torch.testing.assert_close(
        result[:, :3],
        torch.ones(2, 3, 8, 8),
    )
    torch.testing.assert_close(
        result[:, 3:],
        input_mask,
    )


def test_merge_masks_resizes_both_masks():
    input_mask = torch.zeros(2, 2, 4, 4)
    skip_mask = torch.ones(2, 3, 16, 16)
    reference = torch.randn(2, 5, 8, 8)

    result = _merge_masks(
        input_mask,
        skip_mask,
        out_channels=2,
        skip_channels=3,
        spatial_size=(8, 8),
        reference=reference,
    )

    assert result.shape == (2, 5, 8, 8)


def test_merge_masks_concatenates_skip_before_input():
    input_mask = torch.full(
        (1, 2, 4, 4),
        2.0,
    )
    skip_mask = torch.full(
        (1, 3, 4, 4),
        3.0,
    )
    reference = torch.randn(1, 5, 4, 4)

    result = _merge_masks(
        input_mask,
        skip_mask,
        out_channels=2,
        skip_channels=3,
        spatial_size=(4, 4),
        reference=reference,
    )

    torch.testing.assert_close(
        result[:, :3],
        torch.full((1, 3, 4, 4), 3.0),
    )
    torch.testing.assert_close(
        result[:, 3:],
        torch.full((1, 2, 4, 4), 2.0),
    )


def test_merge_masks_generated_mask_uses_reference_dtype():
    skip_mask = torch.zeros(
        2,
        3,
        8,
        8,
        dtype=torch.float64,
    )
    reference = torch.randn(
        2,
        5,
        8,
        8,
        dtype=torch.float64,
    )

    result = _merge_masks(
        None,
        skip_mask,
        out_channels=2,
        skip_channels=3,
        spatial_size=(8, 8),
        reference=reference,
    )

    assert result.dtype == torch.float64


# ---------------------------------------------------------------------------
# _resize_tensor
# ---------------------------------------------------------------------------


def test_resize_tensor_matching_size_returns_same_object():
    tensor = torch.randn(2, 3, 8, 8)

    result = _resize_tensor(
        tensor,
        (8, 8),
    )

    assert result is tensor


@pytest.mark.parametrize(
    "mode",
    [
        "bilinear",
        "bicubic",
    ],
)
def test_resize_tensor_continuous_modes(mode):
    tensor = torch.randn(2, 3, 4, 5)

    result = _resize_tensor(
        tensor,
        (8, 10),
        mode=mode,
    )

    expected = F.interpolate(
        tensor,
        size=(8, 10),
        mode=mode,
        align_corners=False,
    )

    assert result.shape == (2, 3, 8, 10)
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize(
    "mode",
    [
        "nearest",
        "nearest-exact",
        "area",
    ],
)
def test_resize_tensor_modes_without_align_corners(mode):
    tensor = torch.randn(2, 3, 8, 10)

    result = _resize_tensor(
        tensor,
        (4, 5),
        mode=mode,
    )

    expected = F.interpolate(
        tensor,
        size=(4, 5),
        mode=mode,
    )

    assert result.shape == (2, 3, 4, 5)
    torch.testing.assert_close(result, expected)


def test_resize_tensor_default_mode_is_bilinear():
    tensor = torch.randn(2, 3, 4, 5)

    result = _resize_tensor(
        tensor,
        (8, 10),
    )

    expected = F.interpolate(
        tensor,
        size=(8, 10),
        mode="bilinear",
        align_corners=False,
    )

    torch.testing.assert_close(result, expected)


def test_resize_tensor_preserves_dtype():
    tensor = torch.randn(
        2,
        3,
        4,
        5,
        dtype=torch.float64,
    )

    result = _resize_tensor(
        tensor,
        (8, 10),
    )

    assert result.dtype == torch.float64


# ---------------------------------------------------------------------------
# _get_normal
# ---------------------------------------------------------------------------


def test_get_normal_returns_normal_distribution():
    reference = torch.randn(2, 3)

    distribution = _get_normal(reference)

    assert isinstance(
        distribution,
        torch.distributions.Normal,
    )


def test_get_normal_default_parameters():
    reference = torch.randn(2, 3)

    distribution = _get_normal(reference)

    torch.testing.assert_close(
        distribution.loc,
        torch.zeros_like(reference),
    )
    torch.testing.assert_close(
        distribution.scale,
        torch.ones_like(reference),
    )


@pytest.mark.parametrize(
    "std",
    [
        0.1,
        0.5,
        1.0,
        2.0,
        10.0,
    ],
)
def test_get_normal_uses_requested_standard_deviation(std):
    reference = torch.randn(2, 3)

    distribution = _get_normal(
        reference,
        std=std,
    )

    torch.testing.assert_close(
        distribution.scale,
        torch.full_like(reference, std),
    )


def test_get_normal_preserves_dtype():
    reference = torch.randn(
        2,
        3,
        dtype=torch.float64,
    )

    distribution = _get_normal(reference)

    assert distribution.loc.dtype == torch.float64
    assert distribution.scale.dtype == torch.float64


@pytest.mark.parametrize(
    "std",
    [
        0.0,
        -1.0,
    ],
)
def test_get_normal_rejects_nonpositive_standard_deviation(std):
    reference = torch.randn(2, 3)

    with pytest.raises(ValueError):
        _get_normal(
            reference,
            std=std,
        )


# ---------------------------------------------------------------------------
# _sample
# ---------------------------------------------------------------------------


def test_sample_default_shape():
    mu = torch.zeros(2, 3)
    var = torch.ones(2, 3)

    result = _sample(
        mu,
        var,
    )

    assert result.shape == (1, 2, 3)


@pytest.mark.parametrize(
    "sample_size",
    [
        1,
        2,
        5,
        10,
    ],
)
def test_sample_requested_shape(sample_size):
    mu = torch.zeros(2, 3)
    var = torch.ones(2, 3)

    result = _sample(
        mu,
        var,
        sample_size=sample_size,
    )

    assert result.shape == (
        sample_size,
        2,
        3,
    )


def test_sample_zero_variance_returns_mean():
    mu = torch.randn(2, 3)
    var = torch.zeros(2, 3)

    result = _sample(
        mu,
        var,
        sample_size=4,
    )

    expected = mu.unsqueeze(0).expand(
        4,
        -1,
        -1,
    )

    torch.testing.assert_close(result, expected)


def test_sample_calls_get_normal(
    monkeypatch,
):
    mu = torch.ones(2, 3)
    var = torch.full((2, 3), 4.0)
    captured = {}

    class FakeDistribution:
        def sample(self, shape):
            captured["sample_shape"] = shape
            return torch.full(
                (*shape, 2, 3),
                3.0,
            )

    def fake_get_normal(
        reference,
        std,
    ):
        captured["reference"] = reference
        captured["std"] = std
        return FakeDistribution()

    monkeypatch.setattr(
        module,
        "_get_normal",
        fake_get_normal,
    )

    result = _sample(
        mu,
        var,
        sample_size=5,
        std=2.5,
    )

    assert captured["reference"] is var
    assert captured["std"] == pytest.approx(2.5)
    assert captured["sample_shape"] == (5,)

    expected = torch.full(
        (5, 2, 3),
        7.0,
    )
    torch.testing.assert_close(result, expected)


def test_sample_preserves_dtype():
    mu = torch.zeros(
        2,
        3,
        dtype=torch.float64,
    )
    var = torch.ones_like(mu)

    result = _sample(
        mu,
        var,
        sample_size=2,
    )

    assert result.dtype == torch.float64


def test_sample_mean_is_approximately_mu():
    torch.manual_seed(0)

    mu = torch.full(
        (2, 3),
        5.0,
    )
    var = torch.ones(2, 3)

    result = _sample(
        mu,
        var,
        sample_size=10000,
    )

    torch.testing.assert_close(
        result.mean(dim=0),
        mu,
        atol=0.05,
        rtol=0.01,
    )


def test_sample_std_scales_noise():
    torch.manual_seed(1)

    mu = torch.zeros(1)
    var = torch.ones(1)

    result = _sample(
        mu,
        var,
        sample_size=20000,
        std=3.0,
    )

    assert result.std().item() == pytest.approx(
        3.0,
        abs=0.08,
    )


# ---------------------------------------------------------------------------
# _noise_injection
# ---------------------------------------------------------------------------


def test_noise_injection_adds_one_channel():
    reference = torch.randn(2, 3, 8, 10)

    result = _noise_injection(reference)

    assert result.shape == (2, 4, 8, 10)


def test_noise_injection_preserves_original_channels():
    reference = torch.randn(2, 3, 8, 10)

    result = _noise_injection(reference)

    torch.testing.assert_close(
        result[:, :3],
        reference,
    )


def test_noise_injection_uses_first_channel_shape():
    reference = torch.randn(4, 7, 5, 6)

    result = _noise_injection(reference)

    assert result[:, -1:].shape == (
        4,
        1,
        5,
        6,
    )


def test_noise_injection_calls_sample(
    monkeypatch,
):
    reference = torch.randn(2, 3, 4, 5)
    captured = {}

    def fake_sample(
        mu,
        var,
        sample_size=1,
        std=1,
    ):
        captured["mu"] = mu
        captured["var"] = var
        captured["sample_size"] = sample_size
        captured["std"] = std

        return torch.full(
            (1, 2, 1, 4, 5),
            7.0,
        )

    monkeypatch.setattr(
        module,
        "_sample",
        fake_sample,
    )

    result = _noise_injection(reference)

    torch.testing.assert_close(
        captured["mu"],
        torch.zeros(2, 1, 4, 5),
    )
    torch.testing.assert_close(
        captured["var"],
        torch.ones(2, 1, 4, 5),
    )
    assert captured["sample_size"] == 1
    assert captured["std"] == 1

    torch.testing.assert_close(
        result[:, -1:],
        torch.full(
            (2, 1, 4, 5),
            7.0,
        ),
    )


def test_noise_injection_preserves_dtype():
    reference = torch.randn(
        2,
        3,
        4,
        5,
        dtype=torch.float64,
    )

    result = _noise_injection(reference)

    assert result.dtype == torch.float64


def test_noise_injection_noise_is_not_constant():
    torch.manual_seed(0)
    reference = torch.zeros(8, 3, 16, 16)

    result = _noise_injection(reference)
    noise = result[:, -1:]

    assert noise.std().item() > 0


def test_noise_injection_supports_backward_for_original_tensor():
    reference = torch.randn(
        2,
        3,
        4,
        5,
        requires_grad=True,
    )

    result = _noise_injection(reference)
    result.sum().backward()

    assert reference.grad is not None
    assert reference.grad.shape == reference.shape


# ---------------------------------------------------------------------------
# _expand_mask
# ---------------------------------------------------------------------------


def test_expand_mask_adds_missing_channels():
    tensor = torch.randn(2, 4, 8, 8)
    mask = torch.zeros(2, 3, 8, 8)

    result = _expand_mask(
        tensor,
        mask,
    )

    assert result.shape == (2, 4, 8, 8)


def test_expand_mask_preserves_original_channels():
    tensor = torch.randn(2, 5, 8, 8)
    mask = torch.randn(2, 3, 8, 8)

    result = _expand_mask(
        tensor,
        mask,
    )

    torch.testing.assert_close(
        result[:, :3],
        mask,
    )


def test_expand_mask_new_channels_are_valid():
    tensor = torch.randn(2, 5, 8, 8)
    mask = torch.zeros(2, 3, 8, 8)

    result = _expand_mask(
        tensor,
        mask,
    )

    torch.testing.assert_close(
        result[:, 3:],
        torch.ones(2, 2, 8, 8),
    )


def test_expand_mask_matching_channels_returns_equal_mask():
    tensor = torch.randn(2, 3, 8, 8)
    mask = torch.randn(2, 3, 8, 8)

    result = _expand_mask(
        tensor,
        mask,
    )

    assert result.shape == mask.shape
    torch.testing.assert_close(result, mask)


def test_expand_mask_adds_single_noise_channel():
    tensor = torch.randn(2, 4, 8, 8)
    mask = torch.zeros(2, 3, 8, 8)

    result = _expand_mask(
        tensor,
        mask,
    )

    assert result.shape[1] == 4
    assert torch.all(result[:, -1] == 1)


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
def test_expand_mask_preserves_dtype(dtype):
    tensor = torch.randn(2, 4, 8, 8)
    mask = torch.zeros(
        2,
        3,
        8,
        8,
        dtype=dtype,
    )

    result = _expand_mask(
        tensor,
        mask,
    )

    assert result.dtype == dtype


def test_expand_mask_preserves_batch_and_spatial_dimensions():
    tensor = torch.randn(4, 7, 5, 9)
    mask = torch.zeros(4, 2, 5, 9)

    result = _expand_mask(
        tensor,
        mask,
    )

    assert result.shape == (4, 7, 5, 9)


def test_expand_mask_with_zero_existing_channels():
    tensor = torch.randn(2, 3, 4, 5)
    mask = torch.empty(2, 0, 4, 5)

    result = _expand_mask(
        tensor,
        mask,
    )

    torch.testing.assert_close(
        result,
        torch.ones(2, 3, 4, 5),
    )
