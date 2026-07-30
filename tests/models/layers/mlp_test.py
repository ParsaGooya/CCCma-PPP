import pytest
import torch
import torch.nn as nn

from cccma_ppp.models.layers.mlp import build_mlp


@pytest.mark.pruned
def test_build_mlp_returns_sequential():
    model = build_mlp([4, 3, 2])

    assert isinstance(model, nn.Sequential)


@pytest.mark.pruned
def test_build_mlp_single_linear_layer():
    model = build_mlp([4, 2])

    assert len(model) == 1
    assert isinstance(model[0], nn.Linear)
    assert model[0].in_features == 4
    assert model[0].out_features == 2


@pytest.mark.pruned
def test_build_mlp_multiple_linear_layers():
    model = build_mlp([4, 8, 6, 2])

    linear_layers = [layer for layer in model if isinstance(layer, nn.Linear)]

    assert len(linear_layers) == 3
    assert [
        (
            layer.in_features,
            layer.out_features,
        )
        for layer in linear_layers
    ] == [
        (4, 8),
        (8, 6),
        (6, 2),
    ]


@pytest.mark.pruned
def test_build_mlp_empty_dimensions_returns_empty_sequential():
    model = build_mlp([])

    assert isinstance(model, nn.Sequential)
    assert len(model) == 0


@pytest.mark.pruned
def test_build_mlp_single_dimension_returns_empty_sequential():
    model = build_mlp([4])

    assert isinstance(model, nn.Sequential)
    assert len(model) == 0


@pytest.mark.pruned
def test_build_mlp_two_dimensions_has_no_default_final_activation():
    model = build_mlp([4, 2])

    assert len(model) == 1
    assert isinstance(model[0], nn.Linear)


@pytest.mark.pruned
def test_build_mlp_default_structure():
    model = build_mlp([4, 8, 2])

    assert len(model) == 3
    assert isinstance(model[0], nn.Linear)
    assert isinstance(model[1], nn.ReLU)
    assert isinstance(model[2], nn.Linear)


@pytest.mark.pruned
def test_build_mlp_default_relu_is_inplace():
    model = build_mlp([4, 8, 2])

    assert model[1].inplace is True


@pytest.mark.pruned
def test_build_mlp_default_has_no_dropout():
    model = build_mlp([4, 8, 2])

    assert not any(isinstance(layer, nn.Dropout) for layer in model)


@pytest.mark.pruned
def test_build_mlp_default_has_no_batch_normalization():
    model = build_mlp([4, 8, 2])

    assert not any(isinstance(layer, nn.BatchNorm1d) for layer in model)


@pytest.mark.parametrize(
    (
        "activation",
        "expected_type",
    ),
    [
        ("relu", nn.ReLU),
        ("gelu", nn.GELU),
        ("silu", nn.SiLU),
    ],
)
def test_build_mlp_activation_types(
    activation,
    expected_type,
):
    model = build_mlp(
        [4, 8, 2],
        activation=activation,
    )

    assert isinstance(
        model[1],
        expected_type,
    )


@pytest.mark.parametrize(
    "activation",
    [
        "relu",
        "gelu",
        "silu",
    ],
)
def test_build_mlp_activation_on_every_hidden_layer(
    activation,
):
    model = build_mlp(
        [4, 8, 6, 2],
        activation=activation,
    )

    activation_types = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
    }

    activations = [
        layer
        for layer in model
        if isinstance(
            layer,
            activation_types[activation],
        )
    ]

    assert len(activations) == 2


@pytest.mark.pruned
def test_build_mlp_rejects_unsupported_activation():
    with pytest.raises(
        ValueError,
        match="Unsupported activation",
    ):
        build_mlp(
            [4, 8, 2],
            activation="invalid",
        )


@pytest.mark.pruned
def test_build_mlp_invalid_activation_not_checked_without_activated_layers():
    model = build_mlp(
        [4, 2],
        activation="invalid",
        activate_final=False,
    )

    assert len(model) == 1
    assert isinstance(model[0], nn.Linear)


@pytest.mark.pruned
def test_build_mlp_invalid_activation_checked_when_final_is_activated():
    with pytest.raises(
        ValueError,
        match="Unsupported activation",
    ):
        build_mlp(
            [4, 2],
            activation="invalid",
            activate_final=True,
        )


@pytest.mark.pruned
def test_build_mlp_does_not_activate_final_by_default():
    model = build_mlp([4, 8, 2])

    assert isinstance(model[-1], nn.Linear)


@pytest.mark.parametrize(
    (
        "activation",
        "expected_type",
    ),
    [
        ("relu", nn.ReLU),
        ("gelu", nn.GELU),
        ("silu", nn.SiLU),
    ],
)
def test_build_mlp_activates_final_when_requested(
    activation,
    expected_type,
):
    model = build_mlp(
        [4, 8, 2],
        activation=activation,
        activate_final=True,
    )

    assert isinstance(
        model[-1],
        expected_type,
    )


@pytest.mark.pruned
def test_build_mlp_single_linear_layer_can_activate_final():
    model = build_mlp(
        [4, 2],
        activate_final=True,
    )

    assert len(model) == 2
    assert isinstance(model[0], nn.Linear)
    assert isinstance(model[1], nn.ReLU)


@pytest.mark.pruned
def test_build_mlp_activate_final_adds_activation_to_every_layer():
    model = build_mlp(
        [4, 8, 6, 2],
        activate_final=True,
    )

    relu_layers = [layer for layer in model if isinstance(layer, nn.ReLU)]

    assert len(relu_layers) == 3


@pytest.mark.pruned
def test_build_mlp_adds_dropout_to_hidden_layers():
    model = build_mlp(
        [4, 8, 2],
        dropout_rate=0.25,
    )

    dropout_layers = [layer for layer in model if isinstance(layer, nn.Dropout)]

    assert len(dropout_layers) == 1
    assert dropout_layers[0].p == pytest.approx(0.25)


@pytest.mark.pruned
def test_build_mlp_adds_dropout_after_each_hidden_activation():
    model = build_mlp(
        [4, 8, 6, 2],
        dropout_rate=0.4,
    )

    dropout_layers = [layer for layer in model if isinstance(layer, nn.Dropout)]

    assert len(dropout_layers) == 2

    for layer in dropout_layers:
        assert layer.p == pytest.approx(0.4)


@pytest.mark.pruned
def test_build_mlp_does_not_add_dropout_to_final_layer_by_default():
    model = build_mlp(
        [4, 8, 2],
        dropout_rate=0.5,
    )

    assert isinstance(model[-1], nn.Linear)


@pytest.mark.pruned
def test_build_mlp_adds_dropout_to_final_layer_when_activated():
    model = build_mlp(
        [4, 8, 2],
        dropout_rate=0.5,
        activate_final=True,
    )

    assert isinstance(model[-1], nn.Dropout)
    assert model[-1].p == pytest.approx(0.5)


@pytest.mark.pruned
def test_build_mlp_dropout_zero_still_adds_dropout_layer():
    model = build_mlp(
        [4, 8, 2],
        dropout_rate=0.0,
    )

    dropout_layers = [layer for layer in model if isinstance(layer, nn.Dropout)]

    assert len(dropout_layers) == 1
    assert dropout_layers[0].p == pytest.approx(0.0)


@pytest.mark.pruned
def test_build_mlp_dropout_one_is_constructed():
    model = build_mlp(
        [4, 8, 2],
        dropout_rate=1.0,
    )

    dropout_layers = [layer for layer in model if isinstance(layer, nn.Dropout)]

    assert len(dropout_layers) == 1
    assert dropout_layers[0].p == pytest.approx(1.0)


@pytest.mark.parametrize(
    "dropout_rate",
    [
        -0.1,
        1.1,
        2.0,
    ],
)
def test_build_mlp_invalid_dropout_raises_when_layer_is_constructed(
    dropout_rate,
):
    with pytest.raises(ValueError):
        build_mlp(
            [4, 8, 2],
            dropout_rate=dropout_rate,
        )


@pytest.mark.pruned
def test_build_mlp_invalid_dropout_not_checked_without_activated_layers():
    model = build_mlp(
        [4, 2],
        dropout_rate=2.0,
        activate_final=False,
    )

    assert len(model) == 1


@pytest.mark.pruned
def test_build_mlp_adds_batch_normalization_to_hidden_layers():
    model = build_mlp(
        [4, 8, 2],
        batch_normalization=True,
    )

    batch_norm_layers = [layer for layer in model if isinstance(layer, nn.BatchNorm1d)]

    assert len(batch_norm_layers) == 1
    assert batch_norm_layers[0].num_features == 8


@pytest.mark.pruned
def test_build_mlp_adds_batch_normalization_after_each_hidden_layer():
    model = build_mlp(
        [4, 8, 6, 2],
        batch_normalization=True,
    )

    batch_norm_layers = [layer for layer in model if isinstance(layer, nn.BatchNorm1d)]

    assert len(batch_norm_layers) == 2
    assert [layer.num_features for layer in batch_norm_layers] == [
        8,
        6,
    ]


@pytest.mark.pruned
def test_build_mlp_does_not_normalize_final_layer_by_default():
    model = build_mlp(
        [4, 8, 2],
        batch_normalization=True,
    )

    assert isinstance(model[-1], nn.Linear)


@pytest.mark.pruned
def test_build_mlp_normalizes_final_layer_when_activated():
    model = build_mlp(
        [4, 8, 2],
        batch_normalization=True,
        activate_final=True,
    )

    assert isinstance(model[-1], nn.BatchNorm1d)
    assert model[-1].num_features == 2


@pytest.mark.pruned
def test_build_mlp_single_linear_layer_with_final_batch_normalization():
    model = build_mlp(
        [4, 2],
        batch_normalization=True,
        activate_final=True,
    )

    assert len(model) == 3
    assert isinstance(model[0], nn.Linear)
    assert isinstance(model[1], nn.ReLU)
    assert isinstance(model[2], nn.BatchNorm1d)


@pytest.mark.pruned
def test_build_mlp_batch_normalization_flag_has_no_effect_without_activation():
    model = build_mlp(
        [4, 2],
        batch_normalization=True,
        activate_final=False,
    )

    assert len(model) == 1
    assert isinstance(model[0], nn.Linear)


@pytest.mark.pruned
def test_build_mlp_hidden_layer_order():
    model = build_mlp(
        [4, 8, 2],
        activation="gelu",
        dropout_rate=0.25,
        batch_normalization=True,
    )

    assert len(model) == 5
    assert isinstance(model[0], nn.Linear)
    assert isinstance(model[1], nn.GELU)
    assert isinstance(model[2], nn.Dropout)
    assert isinstance(model[3], nn.BatchNorm1d)
    assert isinstance(model[4], nn.Linear)


def test_build_mlp_all_options_final_layer_order():
    model = build_mlp(
        [4, 8, 2],
        activation="silu",
        dropout_rate=0.3,
        batch_normalization=True,
        activate_final=True,
    )

    assert len(model) == 8

    assert isinstance(model[0], nn.Linear)
    assert isinstance(model[1], nn.SiLU)
    assert isinstance(model[2], nn.Dropout)
    assert isinstance(model[3], nn.BatchNorm1d)

    assert isinstance(model[4], nn.Linear)
    assert isinstance(model[5], nn.SiLU)
    assert isinstance(model[6], nn.Dropout)
    assert isinstance(model[7], nn.BatchNorm1d)


@pytest.mark.pruned
def test_build_mlp_three_layers_all_options_order():
    model = build_mlp(
        [3, 5, 7, 2],
        activation="relu",
        dropout_rate=0.1,
        batch_normalization=True,
        activate_final=False,
    )

    expected_types = [
        nn.Linear,
        nn.ReLU,
        nn.Dropout,
        nn.BatchNorm1d,
        nn.Linear,
        nn.ReLU,
        nn.Dropout,
        nn.BatchNorm1d,
        nn.Linear,
    ]

    assert len(model) == len(expected_types)

    for layer, expected_type in zip(
        model,
        expected_types,
    ):
        assert isinstance(layer, expected_type)


@pytest.mark.parametrize(
    (
        "dims",
        "batch_size",
    ),
    [
        ([4, 2], 1),
        ([4, 8, 2], 3),
        ([10, 20, 15, 5], 7),
        ([1, 1], 4),
    ],
)
def test_build_mlp_forward_shape(
    dims,
    batch_size,
):
    model = build_mlp(dims)
    tensor = torch.randn(
        batch_size,
        dims[0],
    )

    result = model(tensor)

    assert result.shape == (
        batch_size,
        dims[-1],
    )


@pytest.mark.pruned
def test_build_mlp_forward_with_all_options():
    model = build_mlp(
        [4, 8, 6, 2],
        activation="gelu",
        dropout_rate=0.2,
        batch_normalization=True,
        activate_final=True,
    )
    model.eval()

    tensor = torch.randn(5, 4)
    result = model(tensor)

    assert result.shape == (5, 2)
    assert torch.isfinite(result).all()


@pytest.mark.pruned
def test_build_mlp_preserves_leading_dimensions_without_batch_norm():
    model = build_mlp(
        [4, 8, 2],
        batch_normalization=False,
    )

    tensor = torch.randn(3, 5, 4)
    result = model(tensor)

    assert result.shape == (3, 5, 2)


@pytest.mark.pruned
def test_build_mlp_identity_when_dimensions_are_empty():
    model = build_mlp([])
    tensor = torch.randn(2, 3)

    result = model(tensor)

    assert result is tensor


@pytest.mark.pruned
def test_build_mlp_identity_with_single_dimension():
    model = build_mlp([3])
    tensor = torch.randn(2, 3)

    result = model(tensor)

    assert result is tensor


@pytest.mark.pruned
def test_build_mlp_output_is_finite():
    model = build_mlp(
        [4, 16, 8, 2],
        activation="silu",
    )

    tensor = torch.randn(32, 4)
    result = model(tensor)

    assert torch.isfinite(result).all()


@pytest.mark.pruned
def test_build_mlp_supports_float64():
    model = build_mlp(
        [4, 8, 2],
        activation="gelu",
    ).double()

    tensor = torch.randn(
        3,
        4,
        dtype=torch.float64,
    )
    result = model(tensor)

    assert result.dtype == torch.float64
    assert result.shape == (3, 2)


@pytest.mark.pruned
def test_build_mlp_supports_backward():
    model = build_mlp(
        [4, 8, 2],
        activation="relu",
    )

    tensor = torch.randn(
        5,
        4,
        requires_grad=True,
    )
    result = model(tensor)
    result.sum().backward()

    assert tensor.grad is not None

    linear_layers = [layer for layer in model if isinstance(layer, nn.Linear)]

    for layer in linear_layers:
        assert layer.weight.grad is not None
        assert layer.bias.grad is not None


@pytest.mark.pruned
def test_build_mlp_single_linear_exact_value():
    model = build_mlp([2, 1])

    with torch.no_grad():
        model[0].weight.copy_(
            torch.tensor(
                [
                    [2.0, 3.0],
                ]
            )
        )
        model[0].bias.copy_(torch.tensor([1.0]))

    tensor = torch.tensor(
        [
            [4.0, 5.0],
        ]
    )

    result = model(tensor)

    torch.testing.assert_close(
        result,
        torch.tensor(
            [
                [24.0],
            ]
        ),
    )


@pytest.mark.pruned
def test_build_mlp_hidden_activation_affects_output():
    model = build_mlp(
        [1, 1, 1],
        activation="relu",
    )

    with torch.no_grad():
        model[0].weight.fill_(-1.0)
        model[0].bias.zero_()
        model[2].weight.fill_(1.0)
        model[2].bias.zero_()

    result = model(
        torch.tensor(
            [
                [1.0],
            ]
        )
    )

    torch.testing.assert_close(
        result,
        torch.zeros(1, 1),
    )


@pytest.mark.pruned
def test_build_mlp_final_activation_affects_output():
    model = build_mlp(
        [1, 1],
        activation="relu",
        activate_final=True,
    )

    with torch.no_grad():
        model[0].weight.fill_(-1.0)
        model[0].bias.zero_()

    result = model(
        torch.tensor(
            [
                [1.0],
            ]
        )
    )

    torch.testing.assert_close(
        result,
        torch.zeros(1, 1),
    )


@pytest.mark.pruned
def test_build_mlp_without_final_activation_preserves_negative_output():
    model = build_mlp(
        [1, 1],
        activation="relu",
        activate_final=False,
    )

    with torch.no_grad():
        model[0].weight.fill_(-1.0)
        model[0].bias.zero_()

    result = model(
        torch.tensor(
            [
                [1.0],
            ]
        )
    )

    torch.testing.assert_close(
        result,
        torch.tensor(
            [
                [-1.0],
            ]
        ),
    )


@pytest.mark.pruned
def test_build_mlp_linear_layers_have_independent_parameters():
    model = build_mlp([4, 4, 4])

    first_linear = model[0]
    second_linear = model[2]

    assert first_linear.weight is not second_linear.weight
    assert first_linear.bias is not second_linear.bias


@pytest.mark.pruned
def test_build_mlp_registers_all_parameters():
    model = build_mlp([4, 8, 6, 2])

    parameter_names = set(name for name, _ in model.named_parameters())

    assert parameter_names == {
        "0.weight",
        "0.bias",
        "2.weight",
        "2.bias",
        "4.weight",
        "4.bias",
    }