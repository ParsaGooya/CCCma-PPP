import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from cccma_ppp.models.layers.generic import (
    DropPath,
    LayerNorm2d,
    _build_activation,
    _build_normalization,
    _validate_dropout,
)


@pytest.mark.parametrize(
    "value",
    [
        None,
        0.0,
        0.25,
        0.5,
        1.0,
    ],
)
def test_validate_dropout_accepts_valid_values(value):
    assert _validate_dropout(value) is None


@pytest.mark.parametrize(
    "value",
    [
        -1.0,
        -0.1,
        1.1,
        2.0,
    ],
)
def test_validate_dropout_rejects_invalid_values(value):
    with pytest.raises(
        ValueError,
        match="Dropout rates must be between 0 and 1",
    ):
        _validate_dropout(value)


@pytest.mark.pruned
def test_build_activation_relu():
    activation = _build_activation("relu")

    assert isinstance(activation, nn.ReLU)
    assert activation.inplace is True


@pytest.mark.pruned
def test_build_activation_gelu():
    activation = _build_activation("gelu")

    assert isinstance(activation, nn.GELU)


def test_build_activation_silu():
    activation = _build_activation("silu")

    assert isinstance(activation, nn.SiLU)
    assert activation.inplace is True


@pytest.mark.parametrize(
    "name",
    [
        "invalid",
        "",
        "tanh",
        "sigmoid",
    ],
)
def test_build_activation_rejects_unsupported_name(name):
    with pytest.raises(
        ValueError,
        match="Unsupported activation",
    ):
        _build_activation(name)


@pytest.mark.parametrize(
    "name",
    [
        "relu",
        "gelu",
        "silu",
    ],
)
def test_build_activation_preserves_shape(name):
    activation = _build_activation(name)
    tensor = torch.randn(2, 3, 4, 5)

    result = activation(tensor)

    assert result.shape == tensor.shape


@pytest.mark.pruned
def test_relu_clamps_negative_values():
    activation = _build_activation("relu")
    tensor = torch.tensor(
        [
            -2.0,
            -1.0,
            0.0,
            1.0,
            2.0,
        ]
    )

    result = activation(tensor)

    torch.testing.assert_close(
        result,
        torch.tensor(
            [
                0.0,
                0.0,
                0.0,
                1.0,
                2.0,
            ]
        ),
    )


@pytest.mark.pruned
def test_silu_matches_functional_implementation():
    activation = _build_activation("silu")
    tensor = torch.randn(2, 3, 4, 5)
    expected = F.silu(tensor.clone())

    result = activation(tensor)

    torch.testing.assert_close(
        result,
        expected,
    )


@pytest.mark.pruned
def test_gelu_matches_functional_implementation():
    activation = _build_activation("gelu")
    tensor = torch.randn(2, 3, 4, 5)
    expected = F.gelu(tensor)

    result = activation(tensor)

    torch.testing.assert_close(
        result,
        expected,
    )


@pytest.mark.pruned
def test_build_normalization_batch():
    normalization = _build_normalization(
        "batch",
        channels=4,
    )

    assert isinstance(normalization, nn.BatchNorm2d)
    assert normalization.num_features == 4


@pytest.mark.pruned
def test_build_normalization_group():
    normalization = _build_normalization(
        "group",
        channels=8,
        group_norm_groups=4,
    )

    assert isinstance(normalization, nn.GroupNorm)
    assert normalization.num_groups == 4
    assert normalization.num_channels == 8


def test_build_normalization_layer():
    normalization = _build_normalization(
        "layer",
        channels=4,
    )

    assert isinstance(normalization, LayerNorm2d)
    assert normalization.weight.shape == (4,)
    assert normalization.bias.shape == (4,)


@pytest.mark.pruned
def test_build_normalization_none():
    normalization = _build_normalization(
        "none",
        channels=4,
    )

    assert isinstance(normalization, nn.Identity)


@pytest.mark.parametrize(
    "name",
    [
        "invalid",
        "",
        "instance",
        "rms",
    ],
)
def test_build_normalization_rejects_unsupported_name(name):
    with pytest.raises(
        ValueError,
        match="Unsupported normalization",
    ):
        _build_normalization(
            name,
            channels=4,
        )


@pytest.mark.pruned
def test_group_normalization_limits_groups_to_channels():
    normalization = _build_normalization(
        "group",
        channels=3,
        group_norm_groups=8,
    )

    assert isinstance(normalization, nn.GroupNorm)
    assert normalization.num_groups == 3


@pytest.mark.parametrize(
    ("channels", "requested_groups", "expected_groups"),
    [
        (8, 8, 8),
        (8, 6, 4),
        (8, 5, 4),
        (8, 3, 2),
        (8, 1, 1),
        (10, 8, 5),
        (10, 6, 5),
        (10, 4, 2),
        (7, 6, 1),
        (3, 8, 3),
        (1, 8, 1),
    ],
)
def test_group_normalization_selects_largest_valid_group_count(
    channels,
    requested_groups,
    expected_groups,
):
    normalization = _build_normalization(
        "group",
        channels=channels,
        group_norm_groups=requested_groups,
    )

    assert normalization.num_groups == expected_groups
    assert channels % normalization.num_groups == 0


@pytest.mark.parametrize(
    "name",
    [
        "batch",
        "group",
        "layer",
        "none",
    ],
)
def test_normalization_preserves_shape(name):
    normalization = _build_normalization(
        name,
        channels=4,
        group_norm_groups=2,
    )
    tensor = torch.randn(2, 4, 6, 5)

    result = normalization(tensor)

    assert result.shape == tensor.shape


@pytest.mark.pruned
def test_identity_normalization_returns_same_object():
    normalization = _build_normalization(
        "none",
        channels=4,
    )
    tensor = torch.randn(2, 4, 6, 5)

    result = normalization(tensor)

    assert result is tensor


@pytest.mark.parametrize(
    ("batch_size", "channels", "height", "width"),
    [
        (1, 1, 1, 1),
        (1, 3, 4, 5),
        (2, 4, 1, 1),
        (3, 8, 6, 7),
    ],
)
def test_layer_norm_2d_handles_multiple_shapes(
    batch_size,
    channels,
    height,
    width,
):
    normalization = LayerNorm2d(channels)
    tensor = torch.randn(
        batch_size,
        channels,
        height,
        width,
    )

    result = normalization(tensor)

    assert result.shape == tensor.shape


@pytest.mark.parametrize(
    "drop_probability",
    [
        0.0,
        0.1,
        0.5,
        0.9,
        1.0,
    ],
)
def test_drop_path_stores_probability(drop_probability):
    drop_path = DropPath(drop_probability)

    assert drop_path.drop_probability == pytest.approx(drop_probability)


@pytest.mark.pruned
def test_drop_path_zero_probability_returns_same_object_in_training():
    drop_path = DropPath(0.0)
    drop_path.train()

    tensor = torch.randn(3, 4, 5, 6)
    result = drop_path(tensor)

    assert result is tensor


@pytest.mark.pruned
def test_drop_path_zero_probability_returns_same_object_in_evaluation():
    drop_path = DropPath(0.0)
    drop_path.eval()

    tensor = torch.randn(3, 4, 5, 6)
    result = drop_path(tensor)

    assert result is tensor


@pytest.mark.parametrize(
    "drop_probability",
    [
        0.1,
        0.5,
        0.9,
    ],
)
def test_drop_path_evaluation_returns_same_object(
    drop_probability,
):
    drop_path = DropPath(drop_probability)
    drop_path.eval()

    tensor = torch.randn(3, 4, 5, 6)
    result = drop_path(tensor)

    assert result is tensor


def test_drop_path_training_preserves_shape():
    drop_path = DropPath(0.5)
    drop_path.train()

    torch.manual_seed(0)
    tensor = torch.ones(8, 3, 4, 5)

    result = drop_path(tensor)

    assert result.shape == tensor.shape


@pytest.mark.pruned
def test_drop_path_uses_per_sample_mask():
    drop_path = DropPath(0.5)
    drop_path.train()

    torch.manual_seed(0)
    tensor = torch.ones(16, 3, 4, 5)

    result = drop_path(tensor)

    for sample in result:
        unique_values = torch.unique(sample)

        assert unique_values.numel() == 1
        assert unique_values.item() in {
            0.0,
            2.0,
        }


@pytest.mark.pruned
def test_drop_path_scales_kept_samples():
    drop_path = DropPath(0.5)
    drop_path.train()

    torch.manual_seed(1)
    tensor = torch.ones(64, 2, 3, 4)

    result = drop_path(tensor)

    unique_values = set(result.unique().tolist())

    assert unique_values.issubset(
        {
            0.0,
            2.0,
        }
    )
    assert 2.0 in unique_values


@pytest.mark.pruned
def test_drop_path_drops_some_samples():
    drop_path = DropPath(0.5)
    drop_path.train()

    torch.manual_seed(2)
    tensor = torch.ones(128, 1, 1, 1)

    result = drop_path(tensor)

    assert torch.any(result == 0)
    assert torch.any(result == 2)


@pytest.mark.pruned
def test_drop_path_preserves_expected_mean_approximately():
    drop_path = DropPath(0.25)
    drop_path.train()

    torch.manual_seed(3)
    tensor = torch.ones(10000, 1, 1, 1)

    result = drop_path(tensor)

    assert result.mean().item() == pytest.approx(
        1.0,
        abs=0.03,
    )


@pytest.mark.parametrize(
    "shape",
    [
        (32,),
        (32, 4),
        (32, 4, 5),
        (32, 4, 5, 6),
        (32, 2, 3, 4, 5),
    ],
)
def test_drop_path_supports_multiple_tensor_ranks(shape):
    drop_path = DropPath(0.5)
    drop_path.train()

    torch.manual_seed(4)
    tensor = torch.ones(shape)

    result = drop_path(tensor)

    assert result.shape == tensor.shape


@pytest.mark.pruned
def test_drop_path_preserves_dtype():
    drop_path = DropPath(0.5)
    drop_path.train()

    torch.manual_seed(5)
    tensor = torch.ones(
        32,
        3,
        4,
        5,
        dtype=torch.float64,
    )

    result = drop_path(tensor)

    assert result.dtype == torch.float64


@pytest.mark.pruned
def test_drop_path_preserves_device():
    drop_path = DropPath(0.5)
    drop_path.train()

    tensor = torch.ones(32, 3, 4, 5)
    result = drop_path(tensor)

    assert result.device == tensor.device


@pytest.mark.pruned
def test_drop_path_supports_backward():
    drop_path = DropPath(0.5)
    drop_path.train()

    torch.manual_seed(6)
    tensor = torch.ones(
        32,
        3,
        4,
        5,
        requires_grad=True,
    )

    result = drop_path(tensor)
    result.sum().backward()

    assert tensor.grad is not None
    assert tensor.grad.shape == tensor.shape


@pytest.mark.pruned
def test_drop_path_random_tensor_shape(
    monkeypatch,
):
    captured = {}

    def fake_rand(
        shape,
        dtype,
        device,
    ):
        captured["shape"] = shape
        captured["dtype"] = dtype
        captured["device"] = device

        return torch.zeros(
            shape,
            dtype=dtype,
            device=device,
        )

    monkeypatch.setattr(
        torch,
        "rand",
        fake_rand,
    )

    drop_path = DropPath(0.5)
    drop_path.train()

    tensor = torch.ones(
        7,
        3,
        4,
        5,
        dtype=torch.float64,
    )

    result = drop_path(tensor)

    assert captured == {
        "shape": (7, 1, 1, 1),
        "dtype": torch.float64,
        "device": tensor.device,
    }
    torch.testing.assert_close(
        result,
        torch.zeros_like(tensor),
    )


@pytest.mark.pruned
def test_drop_path_random_values_above_threshold_are_kept(
    monkeypatch,
):
    def fake_rand(
        shape,
        dtype,
        device,
    ):
        return torch.full(
            shape,
            0.75,
            dtype=dtype,
            device=device,
        )

    monkeypatch.setattr(
        torch,
        "rand",
        fake_rand,
    )

    drop_path = DropPath(0.5)
    drop_path.train()

    tensor = torch.ones(2, 3, 4, 5)
    result = drop_path(tensor)

    torch.testing.assert_close(
        result,
        torch.full_like(
            tensor,
            2.0,
        ),
    )


@pytest.mark.pruned
def test_drop_path_random_values_below_threshold_are_dropped(
    monkeypatch,
):
    def fake_rand(
        shape,
        dtype,
        device,
    ):
        return torch.full(
            shape,
            0.25,
            dtype=dtype,
            device=device,
        )

    monkeypatch.setattr(
        torch,
        "rand",
        fake_rand,
    )

    drop_path = DropPath(0.5)
    drop_path.train()

    tensor = torch.ones(2, 3, 4, 5)
    result = drop_path(tensor)

    torch.testing.assert_close(
        result,
        torch.zeros_like(tensor),
    )