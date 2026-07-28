from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from cccma_ppp.models.mlp_models.deterministic import (
    Autoencoder,
    AutoencoderConfig,
)


def make_config(**kwargs):
    defaults = {
        "encoder_hidden_dims": [8, 4],
        "decoder_hidden_dims": [8],
        "batch_normalization": False,
        "dropout_rate": None,
        "append_mode": 1,
        "init_method": "trunc_normal",
        "activation": "relu",
    }
    defaults.update(kwargs)
    return AutoencoderConfig(**defaults)


def make_model(
    *,
    config=None,
    input_shape=(1, 4),
    output_shape=(1, 4),
    added_features_dim=None,
):
    if config is None:
        config = make_config()

    return Autoencoder(
        config=config,
        input_shape=input_shape,
        output_shape=output_shape,
        added_features_dim=added_features_dim,
    )


def make_request(
    *,
    input=None,
    input_mask=None,
    added_features=None,
):
    if input is None:
        input = torch.randn(3, 1, 4)

    return SimpleNamespace(
        input=input,
        input_mask=input_mask,
        added_features=added_features,
    )


# ---------------------------------------------------------------------------
# AutoencoderConfig
# ---------------------------------------------------------------------------


def test_config_class_constants():
    assert AutoencoderConfig.NUM_INPUT_DIMS == 2
    assert AutoencoderConfig.NUM_OUTPUT_DIMS == 2
    assert AutoencoderConfig.GENERATOR is None


def test_config_defaults():
    config = AutoencoderConfig(
        encoder_hidden_dims=[8, 4],
    )

    assert config.encoder_hidden_dims == [8, 4]
    assert config.decoder_hidden_dims == [8]
    assert config.batch_normalization is False
    assert config.dropout_rate is None
    assert config.append_mode == 1
    assert config.init_method == "trunc_normal"
    assert config.activation == "relu"


def test_config_preserves_explicit_decoder_dimensions():
    config = make_config(
        decoder_hidden_dims=[6, 10],
    )

    assert config.decoder_hidden_dims == [6, 10]


def test_config_infers_decoder_dimensions():
    config = AutoencoderConfig(
        encoder_hidden_dims=[16, 8, 4],
        decoder_hidden_dims=None,
    )

    assert config.decoder_hidden_dims == [8, 16]


def test_config_infers_empty_decoder_for_single_encoder_dimension():
    config = AutoencoderConfig(
        encoder_hidden_dims=[4],
        decoder_hidden_dims=None,
    )

    assert config.decoder_hidden_dims == []


def test_config_infers_empty_decoder_for_empty_encoder_dimensions():
    config = AutoencoderConfig(
        encoder_hidden_dims=[],
        decoder_hidden_dims=None,
    )

    assert config.decoder_hidden_dims == []


@pytest.mark.parametrize(
    "dropout_rate",
    [
        None,
        0.0,
        0.25,
        0.5,
        1.0,
    ],
)
def test_config_accepts_valid_dropout(dropout_rate):
    config = make_config(
        dropout_rate=dropout_rate,
    )

    assert config.dropout_rate == dropout_rate


@pytest.mark.parametrize(
    "dropout_rate",
    [
        -1.0,
        -0.1,
        1.1,
        2.0,
    ],
)
def test_config_rejects_invalid_dropout(dropout_rate):
    with pytest.raises(
        ValueError,
        match="Dropout rates must be between 0 and 1",
    ):
        make_config(
            dropout_rate=dropout_rate,
        )


@pytest.mark.parametrize(
    "append_mode",
    [
        1,
        2,
        3,
    ],
)
def test_config_accepts_append_modes(append_mode):
    config = make_config(
        append_mode=append_mode,
    )

    assert config.append_mode == append_mode


@pytest.mark.parametrize(
    "activation",
    [
        "relu",
        "gelu",
        "silu",
    ],
)
def test_config_accepts_activation(activation):
    config = make_config(
        activation=activation,
    )

    assert config.activation == activation


def test_config_build_returns_autoencoder():
    config = make_config()

    model = config.build(
        input_shape=np.asarray([1, 4]),
        output_shape=np.asarray([1, 4]),
        added_features_dim=2,
    )

    assert isinstance(model, Autoencoder)
    assert model.config is config
    assert model.added_features_dim == 2


def test_config_build_defaults_output_shape():
    config = make_config()

    model = config.build(
        input_shape=np.asarray([1, 4]),
        output_shape=None,
    )

    assert model.input_shape == 4
    assert model.output_shape == 4


# ---------------------------------------------------------------------------
# Autoencoder initialization
# ---------------------------------------------------------------------------


def test_model_initialization_basic():
    config = make_config()

    model = make_model(config=config)

    assert model.config is config
    assert model.encoder_hidden_dims == [8, 4]
    assert model.decoder_hidden_dims == [8]
    assert model.latent_size == 4
    assert model.input_shape == 4
    assert model.output_shape == 4
    assert model.added_features_dim == 0
    assert model.append_mode == 1


def test_model_defaults_output_shape_to_input_shape():
    model = Autoencoder(
        config=make_config(),
        input_shape=np.asarray([2, 3]),
        output_shape=None,
    )

    assert model.input_shape == 6
    assert model.output_shape == 6


def test_model_accepts_tuple_shapes():
    model = make_model(
        input_shape=(2, 3),
        output_shape=(1, 5),
    )

    assert model.input_shape == 6
    assert model.output_shape == 5


def test_model_accepts_numpy_shapes():
    model = make_model(
        input_shape=np.asarray([2, 3]),
        output_shape=np.asarray([1, 5]),
    )

    assert model.input_shape == 6
    assert model.output_shape == 5


@pytest.mark.parametrize(
    "output_shape",
    [
        (4,),
        (1, 2, 3),
        (1, 2, 3, 4),
    ],
)
def test_model_rejects_invalid_output_rank(output_shape):
    with pytest.raises(
        RuntimeError,
        match="MLP models should create 2D outputs",
    ):
        make_model(
            output_shape=output_shape,
        )


def test_model_requires_nonempty_encoder_dimensions():
    config = make_config(
        encoder_hidden_dims=[],
        decoder_hidden_dims=[],
    )

    with pytest.raises(IndexError):
        make_model(config=config)


def test_model_converts_none_added_features_to_zero():
    model = make_model(
        added_features_dim=None,
    )

    assert model.added_features_dim == 0


@pytest.mark.parametrize(
    "added_features_dim",
    [
        0,
        1,
        3,
        10,
    ],
)
def test_model_preserves_added_features_dimension(added_features_dim):
    model = make_model(
        added_features_dim=added_features_dim,
    )

    assert model.added_features_dim == added_features_dim


def test_model_builds_encoder_and_decoder():
    model = make_model()

    assert isinstance(model.encoder, nn.Sequential)
    assert isinstance(model.decoder, nn.Sequential)


def test_model_uses_requested_activation():
    model = make_model(
        config=make_config(
            activation="gelu",
        )
    )

    assert any(isinstance(layer, nn.GELU) for layer in model.encoder)
    assert any(isinstance(layer, nn.GELU) for layer in model.decoder)


def test_model_uses_requested_dropout():
    model = make_model(
        config=make_config(
            dropout_rate=0.25,
        )
    )

    encoder_dropout = [
        layer for layer in model.encoder if isinstance(layer, nn.Dropout)
    ]
    decoder_dropout = [
        layer for layer in model.decoder if isinstance(layer, nn.Dropout)
    ]

    assert encoder_dropout
    assert decoder_dropout
    assert encoder_dropout[0].p == pytest.approx(0.25)
    assert decoder_dropout[0].p == pytest.approx(0.25)


def test_model_uses_batch_normalization():
    model = make_model(
        config=make_config(
            batch_normalization=True,
        )
    )

    assert any(isinstance(layer, nn.BatchNorm1d) for layer in model.encoder)
    assert any(isinstance(layer, nn.BatchNorm1d) for layer in model.decoder)


def test_decoder_final_layer_is_not_activated():
    model = make_model()

    assert isinstance(model.decoder[-1], nn.Linear)


# ---------------------------------------------------------------------------
# Append-mode architecture
# ---------------------------------------------------------------------------


def test_append_mode_one_adds_features_to_encoder():
    model = make_model(
        config=make_config(
            append_mode=1,
        ),
        added_features_dim=3,
    )

    first_encoder_linear = next(
        layer for layer in model.encoder if isinstance(layer, nn.Linear)
    )
    first_decoder_linear = next(
        layer for layer in model.decoder if isinstance(layer, nn.Linear)
    )

    assert first_encoder_linear.in_features == 7
    assert first_decoder_linear.in_features == 4


def test_append_mode_two_adds_features_to_decoder():
    model = make_model(
        config=make_config(
            append_mode=2,
        ),
        added_features_dim=3,
    )

    first_encoder_linear = next(
        layer for layer in model.encoder if isinstance(layer, nn.Linear)
    )
    first_decoder_linear = next(
        layer for layer in model.decoder if isinstance(layer, nn.Linear)
    )

    assert first_encoder_linear.in_features == 4
    assert first_decoder_linear.in_features == 7


def test_append_mode_three_adds_features_to_both_networks():
    model = make_model(
        config=make_config(
            append_mode=3,
        ),
        added_features_dim=3,
    )

    first_encoder_linear = next(
        layer for layer in model.encoder if isinstance(layer, nn.Linear)
    )
    first_decoder_linear = next(
        layer for layer in model.decoder if isinstance(layer, nn.Linear)
    )

    assert first_encoder_linear.in_features == 7
    assert first_decoder_linear.in_features == 7


@pytest.mark.parametrize(
    "append_mode",
    [
        1,
        2,
        3,
    ],
)
def test_append_modes_without_added_feature_dimension(append_mode):
    model = make_model(
        config=make_config(
            append_mode=append_mode,
        ),
        added_features_dim=None,
    )

    first_encoder_linear = next(
        layer for layer in model.encoder if isinstance(layer, nn.Linear)
    )
    first_decoder_linear = next(
        layer for layer in model.decoder if isinstance(layer, nn.Linear)
    )

    assert first_encoder_linear.in_features == 4
    assert first_decoder_linear.in_features == 4


# ---------------------------------------------------------------------------
# Forward without added features
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "append_mode",
    [
        1,
        2,
        3,
    ],
)
def test_forward_without_added_features(append_mode):
    model = make_model(
        config=make_config(
            append_mode=append_mode,
        ),
        added_features_dim=2,
    )

    first_encoder = next(
        layer for layer in model.encoder if isinstance(layer, nn.Linear)
    )
    first_decoder = next(
        layer for layer in model.decoder if isinstance(layer, nn.Linear)
    )

    first_encoder.in_features = 4
    first_decoder.in_features = 4

    model = make_model(
        config=make_config(
            append_mode=append_mode,
        ),
        added_features_dim=0,
    )

    result = model.forward(
        make_request(
            added_features=None,
        )
    )

    assert result.output.shape == (3, 1, 4)


def test_forward_returns_deterministic_output():
    model = make_model()

    result = model.forward(make_request())

    assert hasattr(result, "output")
    assert isinstance(result.output, torch.Tensor)


def test_forward_preserves_batch_and_channel_dimensions():
    model = make_model(
        input_shape=(2, 3),
        output_shape=(2, 5),
    )

    result = model.forward(
        make_request(
            input=torch.randn(4, 2, 3),
        )
    )

    assert result.output.shape == (4, 2, 5)


def test_forward_flattens_input_before_encoder():
    model = make_model(
        input_shape=(2, 3),
        output_shape=(2, 3),
    )

    captured = {}

    class CaptureEncoder(nn.Module):
        def forward(self, value):
            captured["shape"] = value.shape
            return torch.zeros(value.shape[0], 4)

    model.encoder = CaptureEncoder()
    model.decoder = nn.Linear(4, 6)

    result = model.forward(
        make_request(
            input=torch.randn(5, 2, 3),
        )
    )

    assert captured["shape"] == (5, 6)
    assert result.output.shape == (5, 2, 3)


def test_forward_applies_input_mask():
    model = make_model()
    captured = {}

    class CaptureEncoder(nn.Module):
        def forward(self, value):
            captured["value"] = value
            return torch.zeros(value.shape[0], 4)

    model.encoder = CaptureEncoder()
    model.decoder = nn.Linear(4, 4)

    input_tensor = torch.ones(2, 1, 4)
    input_mask = torch.tensor(
        [
            [[1.0, 0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0, 1.0]],
        ]
    )

    model.forward(
        make_request(
            input=input_tensor,
            input_mask=input_mask,
        )
    )

    torch.testing.assert_close(
        captured["value"],
        input_mask.flatten(start_dim=1),
    )


def test_forward_without_mask_preserves_input_values():
    model = make_model()
    captured = {}

    class CaptureEncoder(nn.Module):
        def forward(self, value):
            captured["value"] = value
            return torch.zeros(value.shape[0], 4)

    model.encoder = CaptureEncoder()
    model.decoder = nn.Linear(4, 4)

    input_tensor = torch.randn(2, 1, 4)

    model.forward(
        make_request(
            input=input_tensor,
            input_mask=None,
        )
    )

    torch.testing.assert_close(
        captured["value"],
        input_tensor.flatten(start_dim=1),
    )


# ---------------------------------------------------------------------------
# Forward with append mode 1
# ---------------------------------------------------------------------------


def test_forward_append_mode_one_concatenates_features_to_encoder():
    model = make_model(
        config=make_config(
            append_mode=1,
        ),
        added_features_dim=2,
    )

    captured = {}

    class CaptureEncoder(nn.Module):
        def forward(self, value):
            captured["encoder"] = value
            return torch.zeros(value.shape[0], 4)

    class CaptureDecoder(nn.Module):
        def forward(self, value):
            captured["decoder"] = value
            return torch.zeros(value.shape[0], 4)

    model.encoder = CaptureEncoder()
    model.decoder = CaptureDecoder()

    input_tensor = torch.full((2, 1, 4), 1.0)
    features = torch.full((2, 1, 2), 2.0)

    model.forward(
        make_request(
            input=input_tensor,
            added_features=features,
        )
    )

    torch.testing.assert_close(
        captured["encoder"],
        torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0, 2.0, 2.0],
                [1.0, 1.0, 1.0, 1.0, 2.0, 2.0],
            ]
        ),
    )
    assert captured["decoder"].shape == (2, 4)


def test_forward_append_mode_one_does_not_append_to_decoder():
    model = make_model(
        config=make_config(
            append_mode=1,
        ),
        added_features_dim=2,
    )

    captured = {}

    class FakeEncoder(nn.Module):
        def forward(self, value):
            return torch.zeros(value.shape[0], 4)

    class CaptureDecoder(nn.Module):
        def forward(self, value):
            captured["value"] = value
            return torch.zeros(value.shape[0], 4)

    model.encoder = FakeEncoder()
    model.decoder = CaptureDecoder()

    model.forward(
        make_request(
            added_features=torch.randn(3, 1, 2),
        )
    )

    assert captured["value"].shape == (3, 4)


# ---------------------------------------------------------------------------
# Forward with append mode 2
# ---------------------------------------------------------------------------


def test_forward_append_mode_two_concatenates_features_to_decoder():
    model = make_model(
        config=make_config(
            append_mode=2,
        ),
        added_features_dim=2,
    )

    captured = {}

    class CaptureEncoder(nn.Module):
        def forward(self, value):
            captured["encoder"] = value
            return torch.full((value.shape[0], 4), 3.0)

    class CaptureDecoder(nn.Module):
        def forward(self, value):
            captured["decoder"] = value
            return torch.zeros(value.shape[0], 4)

    model.encoder = CaptureEncoder()
    model.decoder = CaptureDecoder()

    input_tensor = torch.full((2, 1, 4), 1.0)
    features = torch.full((2, 1, 2), 2.0)

    model.forward(
        make_request(
            input=input_tensor,
            added_features=features,
        )
    )

    assert captured["encoder"].shape == (2, 4)

    torch.testing.assert_close(
        captured["decoder"],
        torch.tensor(
            [
                [3.0, 3.0, 3.0, 3.0, 2.0, 2.0],
                [3.0, 3.0, 3.0, 3.0, 2.0, 2.0],
            ]
        ),
    )


def test_forward_append_mode_two_does_not_append_to_encoder():
    model = make_model(
        config=make_config(
            append_mode=2,
        ),
        added_features_dim=2,
    )

    captured = {}

    class CaptureEncoder(nn.Module):
        def forward(self, value):
            captured["value"] = value
            return torch.zeros(value.shape[0], 4)

    class FakeDecoder(nn.Module):
        def forward(self, value):
            return torch.zeros(value.shape[0], 4)

    model.encoder = CaptureEncoder()
    model.decoder = FakeDecoder()

    model.forward(
        make_request(
            added_features=torch.randn(3, 1, 2),
        )
    )

    assert captured["value"].shape == (3, 4)


# ---------------------------------------------------------------------------
# Forward with append mode 3
# ---------------------------------------------------------------------------


def test_forward_append_mode_three_concatenates_features_to_both():
    model = make_model(
        config=make_config(
            append_mode=3,
        ),
        added_features_dim=2,
    )

    captured = {}

    class CaptureEncoder(nn.Module):
        def forward(self, value):
            captured["encoder"] = value
            return torch.full((value.shape[0], 4), 3.0)

    class CaptureDecoder(nn.Module):
        def forward(self, value):
            captured["decoder"] = value
            return torch.zeros(value.shape[0], 4)

    model.encoder = CaptureEncoder()
    model.decoder = CaptureDecoder()

    input_tensor = torch.full((2, 1, 4), 1.0)
    features = torch.full((2, 1, 2), 2.0)

    model.forward(
        make_request(
            input=input_tensor,
            added_features=features,
        )
    )

    torch.testing.assert_close(
        captured["encoder"],
        torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0, 2.0, 2.0],
                [1.0, 1.0, 1.0, 1.0, 2.0, 2.0],
            ]
        ),
    )
    torch.testing.assert_close(
        captured["decoder"],
        torch.tensor(
            [
                [3.0, 3.0, 3.0, 3.0, 2.0, 2.0],
                [3.0, 3.0, 3.0, 3.0, 2.0, 2.0],
            ]
        ),
    )


# ---------------------------------------------------------------------------
# Added-feature handling
# ---------------------------------------------------------------------------


def test_forward_flattens_added_features():
    model = make_model(
        config=make_config(
            append_mode=1,
        ),
        added_features_dim=4,
    )

    captured = {}

    class CaptureEncoder(nn.Module):
        def forward(self, value):
            captured["shape"] = value.shape
            return torch.zeros(value.shape[0], 4)

    model.encoder = CaptureEncoder()
    model.decoder = nn.Linear(4, 4)

    model.forward(
        make_request(
            input=torch.randn(2, 1, 4),
            added_features=torch.randn(2, 2, 2),
        )
    )

    assert captured["shape"] == (2, 8)


def test_forward_invalid_append_mode_with_features_raises():
    config = make_config(
        append_mode=99,
    )
    model = make_model(
        config=config,
        added_features_dim=2,
    )

    with pytest.raises(UnboundLocalError):
        model.forward(
            make_request(
                added_features=torch.randn(3, 1, 2),
            )
        )


def test_forward_invalid_append_mode_without_features_uses_default_path():
    model = make_model(
        config=make_config(
            append_mode=99,
        ),
        added_features_dim=0,
    )

    result = model.forward(
        make_request(
            added_features=None,
        )
    )

    assert result.output.shape == (3, 1, 4)


# ---------------------------------------------------------------------------
# Exact forward behavior
# ---------------------------------------------------------------------------


def test_forward_exact_identity_path():
    config = make_config(
        encoder_hidden_dims=[4],
        decoder_hidden_dims=[],
        append_mode=1,
    )
    model = make_model(config=config)

    encoder_linear = next(
        layer for layer in model.encoder if isinstance(layer, nn.Linear)
    )
    decoder_linear = next(
        layer for layer in model.decoder if isinstance(layer, nn.Linear)
    )

    with torch.no_grad():
        encoder_linear.weight.copy_(torch.eye(4))
        encoder_linear.bias.zero_()
        decoder_linear.weight.copy_(torch.eye(4))
        decoder_linear.bias.zero_()

    input_tensor = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0]],
        ]
    )

    result = model.forward(
        make_request(
            input=input_tensor,
        )
    )

    torch.testing.assert_close(
        result.output,
        input_tensor,
    )


def test_forward_zero_mask_with_zero_bias_produces_zero():
    config = make_config(
        encoder_hidden_dims=[4],
        decoder_hidden_dims=[],
    )
    model = make_model(config=config)

    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            with torch.no_grad():
                layer.bias.zero_()

    input_tensor = torch.randn(2, 1, 4)

    result = model.forward(
        make_request(
            input=input_tensor,
            input_mask=torch.zeros_like(input_tensor),
        )
    )

    torch.testing.assert_close(
        result.output,
        torch.zeros_like(result.output),
    )


# ---------------------------------------------------------------------------
# Output reshaping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    (
        "input_shape",
        "output_shape",
        "batch_size",
        "expected_shape",
    ),
    [
        ((1, 4), (1, 4), 2, (2, 1, 4)),
        ((2, 3), (2, 5), 4, (4, 2, 5)),
        ((3, 2), (3, 7), 1, (1, 3, 7)),
    ],
)
def test_forward_output_shapes(
    input_shape,
    output_shape,
    batch_size,
    expected_shape,
):
    model = make_model(
        input_shape=input_shape,
        output_shape=output_shape,
    )

    result = model.forward(
        make_request(
            input=torch.randn(
                batch_size,
                *input_shape,
            )
        )
    )

    assert result.output.shape == expected_shape


def test_forward_uses_input_channel_count_for_output_view():
    model = make_model(
        input_shape=(2, 3),
        output_shape=(2, 5),
    )

    result = model.forward(
        make_request(
            input=torch.randn(4, 2, 3),
        )
    )

    assert result.output.shape[1] == 2
    assert result.output.shape[2] == 5


# ---------------------------------------------------------------------------
# Training behavior
# ---------------------------------------------------------------------------


def test_forward_supports_backward():
    model = make_model()

    input_tensor = torch.randn(
        3,
        1,
        4,
        requires_grad=True,
    )

    result = model.forward(
        make_request(
            input=input_tensor,
        )
    )
    result.output.sum().backward()

    assert input_tensor.grad is not None

    linear_layers = [layer for layer in model.modules() if isinstance(layer, nn.Linear)]

    for layer in linear_layers:
        assert layer.weight.grad is not None
        assert layer.bias.grad is not None


def test_forward_with_added_features_supports_backward():
    model = make_model(
        config=make_config(
            append_mode=3,
        ),
        added_features_dim=2,
    )

    input_tensor = torch.randn(
        3,
        1,
        4,
        requires_grad=True,
    )
    features = torch.randn(
        3,
        1,
        2,
        requires_grad=True,
    )

    result = model.forward(
        make_request(
            input=input_tensor,
            added_features=features,
        )
    )
    result.output.sum().backward()

    assert input_tensor.grad is not None
    assert features.grad is not None


def test_forward_output_is_finite():
    model = make_model(
        config=make_config(
            activation="silu",
            dropout_rate=0.2,
        )
    )
    model.eval()

    result = model.forward(
        make_request(
            input=torch.randn(32, 1, 4),
        )
    )

    assert torch.isfinite(result.output).all()


def test_model_supports_float64():
    model = make_model().double()

    result = model.forward(
        make_request(
            input=torch.randn(
                3,
                1,
                4,
                dtype=torch.float64,
            )
        )
    )

    assert result.output.dtype == torch.float64
