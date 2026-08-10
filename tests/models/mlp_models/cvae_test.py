from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

from cccma_ppp.models.mlp_models.cvae import (
    cVAE_MLP,
    cVAE_MLPConfig,
)


def make_config(**kwargs):
    defaults = {
        "encoder_hidden_dims": [8, 6],
        "latent_size": 3,
        "decoder_hidden_dims": [6],
        "condition_embedding_dims": None,
        "condition_embedding_size": None,
        "condition_dependant_latent": False,
        "condemb_to_decoder": True,
        "batch_normalization": False,
        "dropout_rate": None,
        "init_method": "trunc_normal",
        "activation": "relu",
    }
    defaults.update(kwargs)
    return cVAE_MLPConfig(**defaults)


def make_condition_config(**kwargs):
    defaults = {
        "encoder_hidden_dims": [8, 6],
        "latent_size": 3,
        "decoder_hidden_dims": [6],
        "condition_embedding_dims": [5],
        "condition_embedding_size": 4,
        "condition_dependant_latent": False,
        "condemb_to_decoder": True,
        "batch_normalization": False,
        "dropout_rate": None,
        "activation": "relu",
    }
    defaults.update(kwargs)
    return make_config(**defaults)


def make_condition_latent_config(**kwargs):
    defaults = {
        "encoder_hidden_dims": [8, 6],
        "latent_size": 3,
        "decoder_hidden_dims": [6],
        "condition_embedding_dims": [5],
        "condition_embedding_size": 3,
        "condition_dependant_latent": True,
        "condemb_to_decoder": True,
        "batch_normalization": False,
        "dropout_rate": None,
        "activation": "relu",
    }
    defaults.update(kwargs)
    return make_config(**defaults)


def make_model(
    *,
    config=None,
    input_shape=(1, 4),
    output_shape=(1, 4),
    added_features_dim=None,
):
    if config is None:
        config = make_config()

    return cVAE_MLP(
        config=config,
        input_shape=input_shape,
        output_shape=output_shape,
        added_features_dim=added_features_dim,
    )


def make_forward_request(
    *,
    target=None,
    target_mask=None,
    condition=None,
    condition_mask=None,
    added_features=None,
    latent_sample_size=2,
    posterior_variance_limits=None,
    sample_size=None,
    min_posterior_variance=None,
):
    if target is None:
        target = torch.randn(3, 1, 4)

    if condition is None:
        condition = torch.randn(3, 1, 4)

    if sample_size is not None:
        latent_sample_size = sample_size

    if min_posterior_variance is not None:
        posterior_variance_limits = (
            min_posterior_variance,
            torch.tensor(float("inf")),
        )
    if latent_sample_size is None:
        latent_sample_size = 2
    if min_posterior_variance is not None:
        posterior_variance_limits = (
            min_posterior_variance,
            torch.tensor(float("inf")),
        )
    return SimpleNamespace(
        target=target,
        target_mask=target_mask,
        condition=condition,
        condition_mask=condition_mask,
        added_features=added_features,
        latent_sample_size=latent_sample_size,
        posterior_variance_limits=posterior_variance_limits,
    )


def make_predict_request(
    *,
    condition=None,
    condition_mask=None,
    added_features=None,
    prior_flow=None,
    latent_samples=None,
    nstds=1.0,
    latent_sample_size=2,
    sample_size=None,
):
    if condition is None:
        condition = torch.randn(3, 1, 4)

    if sample_size is not None:
        latent_sample_size = sample_size
    return SimpleNamespace(
        condition=condition,
        condition_mask=condition_mask,
        added_features=added_features,
        prior_flow=prior_flow,
        latent_samples=latent_samples,
        nstds=nstds,
        latent_sample_size=sample_size,
        sample_size=None,
    )


@pytest.mark.pruned
def test_config_preserves_explicit_decoder_dimensions():
    config = make_config(
        encoder_hidden_dims=[10, 8, 6],
        decoder_hidden_dims=[7, 9],
    )

    assert config.decoder_hidden_dims == [7, 9]


def test_config_builds_default_decoder_dimensions():
    config = make_config(
        encoder_hidden_dims=[10, 8, 6],
        decoder_hidden_dims=None,
    )

    assert config.decoder_hidden_dims == [8, 10]


@pytest.mark.pruned
def test_config_default_decoder_for_single_encoder_layer():
    config = make_config(
        encoder_hidden_dims=[8],
        decoder_hidden_dims=None,
    )

    assert config.decoder_hidden_dims == []


def test_config_default_decoder_for_empty_encoder():
    config = make_config(
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


@pytest.mark.pruned
def test_independent_latent_requires_embedding_passed_to_decoder():
    with pytest.raises(
        ValueError,
        match="condition embedding has to be passed to decoder",
    ):
        make_config(
            condition_embedding_dims=[5],
            condition_embedding_size=4,
            condition_dependant_latent=False,
            condemb_to_decoder=False,
        )


@pytest.mark.pruned
def test_config_build_returns_model():
    config = make_config()

    model = config.build(
        input_shape=np.asarray([1, 4]),
        output_shape=np.asarray([1, 4]),
        added_features_dim=None,
    )

    assert isinstance(model, cVAE_MLP)
    assert model.config is config


@pytest.mark.pruned
def test_model_defaults_output_shape_to_input_shape():
    config = make_config()

    model = cVAE_MLP(
        config=config,
        input_shape=np.asarray([1, 4]),
        output_shape=None,
    )

    assert model.output_shape == 4


def test_model_rejects_invalid_output_rank():
    with pytest.raises(
        RuntimeError,
        match="MLP models should create 2D outputs",
    ):
        make_model(
            output_shape=(1, 2, 3),
        )


@pytest.mark.pruned
def test_model_converts_added_features_none_to_zero():
    model = make_model(
        added_features_dim=None,
    )

    assert model.added_features_dim == 0


@pytest.mark.pruned
def test_model_preserves_added_features_dimension():
    model = make_model(
        added_features_dim=2,
    )

    assert model.added_features_dim == 2


@pytest.mark.pruned
def test_model_builds_encoder():
    model = make_model()

    assert isinstance(model.encoder, nn.Sequential)
    assert isinstance(model.mu, nn.Linear)
    assert isinstance(model.log_var, nn.Linear)
    assert model.mu.out_features == 3
    assert model.log_var.out_features == 3


@pytest.mark.pruned
def test_model_builds_condition_embedding():
    model = make_model(
        config=make_condition_config(),
    )

    assert isinstance(model.embedding, nn.Sequential)
    assert model.condemb_to_decoder is True
    assert model.add_condition_size == 4


@pytest.mark.pruned
def test_condition_independent_latent_appends_embedding_projection():
    model = make_model(
        config=make_condition_config(),
    )

    assert isinstance(model.embedding[-1], nn.Linear)
    assert model.embedding[-1].out_features == 4
    assert not hasattr(model, "condition_mu")
    assert not hasattr(model, "condition_log_var")


@pytest.mark.pruned
def test_condition_dependent_latent_builds_distribution_layers():
    model = make_model(
        config=make_condition_latent_config(),
    )

    assert isinstance(model.condition_mu, nn.Linear)
    assert isinstance(model.condition_log_var, nn.Linear)
    assert model.condition_mu.out_features == 3
    assert model.condition_log_var.out_features == 3


@pytest.mark.pruned
def test_condition_dependent_flow_uses_embedding_projection():
    config = make_condition_latent_config()
    config.condition_dependant_flow = True

    model = make_model(config=config)

    assert model.condition_dependant_flow is True
    assert isinstance(model.embedding[-1], nn.Linear)
    assert not hasattr(model, "condition_mu")
    assert not hasattr(model, "condition_log_var")


@pytest.mark.pruned
def test_model_uses_requested_activation():
    model = make_model(
        config=make_config(
            activation="gelu",
        )
    )

    assert any(isinstance(layer, nn.GELU) for layer in model.encoder)


@pytest.mark.pruned
def test_model_uses_requested_dropout():
    model = make_model(
        config=make_config(
            dropout_rate=0.25,
        )
    )

    dropout_layers = [layer for layer in model.encoder if isinstance(layer, nn.Dropout)]

    assert dropout_layers
    assert dropout_layers[0].p == pytest.approx(0.25)


@pytest.mark.pruned
def test_model_uses_batch_normalization():
    model = make_model(
        config=make_config(
            batch_normalization=True,
        )
    )

    assert any(isinstance(layer, nn.BatchNorm1d) for layer in model.encoder)


@pytest.mark.pruned
def test_recognition_applies_target_mask():
    model = make_model()

    target = torch.ones(2, 1, 4)
    mask = torch.zeros_like(target)

    captured = {}

    class CaptureEncoder(nn.Module):
        def forward(self, value):
            captured["value"] = value
            return torch.zeros(value.shape[0], 6)

    model.encoder = CaptureEncoder()

    model._recognition(
        x=target,
        x_mask=mask,
    )

    torch.testing.assert_close(
        captured["value"],
        torch.zeros(2, 4),
    )


@pytest.mark.pruned
def test_recognition_flattens_target():
    model = make_model()
    captured = {}

    class CaptureEncoder(nn.Module):
        def forward(self, value):
            captured["shape"] = value.shape
            return torch.zeros(value.shape[0], 6)

    model.encoder = CaptureEncoder()

    model._recognition(
        x=torch.randn(3, 1, 4),
        x_mask=None,
    )

    assert captured["shape"] == (3, 4)


@pytest.mark.pruned
def test_recognition_concatenates_condition():
    model = make_model(
        config=make_config(
            condition_embedding_dims=None,
        )
    )
    captured = {}

    model.encoder = nn.Identity()
    model.mu = nn.Identity()
    model.log_var = nn.Identity()

    target = torch.randn(2, 1, 4)
    condition = torch.randn(2, 3)

    mu, log_var = model._recognition(
        x=target,
        x_mask=None,
        condition=condition,
    )

    assert mu.shape == (2, 7)
    torch.testing.assert_close(mu, log_var)


@pytest.mark.pruned
def test_recognition_concatenates_added_features():
    model = make_model(
        added_features_dim=2,
    )

    model.encoder = nn.Identity()
    model.mu = nn.Identity()
    model.log_var = nn.Identity()

    target = torch.randn(2, 1, 4)
    added_features = torch.randn(2, 1, 2)

    mu, log_var = model._recognition(
        x=target,
        x_mask=None,
        added_features=added_features,
    )

    assert mu.shape == (2, 6)
    torch.testing.assert_close(mu, log_var)


@pytest.mark.pruned
def test_recognition_concatenates_condition_before_features():
    model = make_model(
        added_features_dim=2,
    )

    model.encoder = nn.Identity()
    model.mu = nn.Identity()
    model.log_var = nn.Identity()

    target = torch.full((1, 1, 4), 1.0)
    condition = torch.full((1, 3), 2.0)
    features = torch.full((1, 1, 2), 3.0)

    mu, _ = model._recognition(
        x=target,
        x_mask=None,
        condition=condition,
        added_features=features,
    )

    torch.testing.assert_close(
        mu,
        torch.tensor([[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0]]),
    )


@pytest.mark.pruned
def test_condition_embedding_returns_embedding():
    model = make_model(
        config=make_condition_config(),
    )

    cond_mu, cond_log_var = model._condition(
        condition=torch.randn(2, 1, 4),
    )

    assert cond_mu.shape == (2, 4)
    assert cond_log_var is None


@pytest.mark.pruned
def test_condition_dependent_latent_returns_distribution():
    model = make_model(
        config=make_condition_latent_config(),
    )

    cond_mu, cond_log_var = model._condition(
        condition=torch.randn(2, 1, 4),
    )

    assert cond_mu.shape == (2, 3)
    assert cond_log_var.shape == (2, 3)


@pytest.mark.pruned
def test_condition_applies_mask():
    model = make_model(
        config=make_condition_config(
            condition_embedding_dims=[],
        )
    )

    captured = {}

    class CaptureEmbedding(nn.Module):
        def forward(self, value):
            captured["value"] = value
            return torch.zeros(value.shape[0], 4)

    model.embedding = CaptureEmbedding()

    condition = torch.ones(2, 1, 4)
    condition_mask = torch.zeros_like(condition)

    model._condition(
        condition=condition,
        condition_mask=condition_mask,
    )

    torch.testing.assert_close(
        captured["value"],
        torch.zeros(2, 4),
    )


@pytest.mark.pruned
def test_condition_flattens_input():
    model = make_model(
        config=make_condition_config(),
    )

    captured = {}

    class CaptureEmbedding(nn.Module):
        def forward(self, value):
            captured["shape"] = value.shape
            return torch.zeros(value.shape[0], 4)

    model.embedding = CaptureEmbedding()

    model._condition(
        condition=torch.randn(2, 1, 4),
    )

    assert captured["shape"] == (2, 4)


@pytest.mark.pruned
def test_condition_concatenates_added_features():
    model = make_model(
        config=make_condition_config(),
        added_features_dim=2,
    )

    captured = {}

    class CaptureEmbedding(nn.Module):
        def forward(self, value):
            captured["value"] = value
            return torch.zeros(value.shape[0], 4)

    model.embedding = CaptureEmbedding()

    condition = torch.full((1, 1, 4), 1.0)
    features = torch.full((1, 1, 2), 2.0)

    model._condition(
        condition=condition,
        added_features=features,
    )

    torch.testing.assert_close(
        captured["value"],
        torch.tensor([[1.0, 1.0, 1.0, 1.0, 2.0, 2.0]]),
    )


@pytest.mark.pruned
def test_generate_concatenates_added_features():
    model = make_model(
        added_features_dim=2,
    )

    captured = {}

    class CaptureDecoder(nn.Module):
        def forward(self, value):
            captured["value"] = value
            return torch.zeros(value.shape[0], 4)

    model.decoder = CaptureDecoder()

    latent = torch.zeros(2, 3, 3)
    features = torch.full((3, 1, 2), 4.0)

    result = model._generate(
        latent_samples=latent,
        added_features=features,
    )

    assert result.shape == (2, 3, 4)
    assert captured["value"].shape == (6, 5)

    torch.testing.assert_close(
        captured["value"][:, -2:],
        torch.full((6, 2), 4.0),
    )


@pytest.mark.pruned
def test_generate_concatenates_condition_when_enabled():
    model = make_model(
        config=make_condition_config(),
    )

    captured = {}

    class CaptureDecoder(nn.Module):
        def forward(self, value):
            captured["value"] = value
            return torch.zeros(value.shape[0], 4)

    model.decoder = CaptureDecoder()

    latent = torch.zeros(2, 3, 3)
    condition = torch.full((3, 4), 5.0)

    model._generate(
        latent_samples=latent,
        condition=condition,
    )

    assert captured["value"].shape == (6, 7)
    torch.testing.assert_close(
        captured["value"][:, -4:],
        torch.full((6, 4), 5.0),
    )


def test_generate_does_not_concatenate_condition_when_disabled():
    config = make_condition_latent_config(
        condemb_to_decoder=False,
    )
    model = make_model(config=config)

    captured = {}

    class CaptureDecoder(nn.Module):
        def forward(self, value):
            captured["value"] = value
            return torch.zeros(value.shape[0], 4)

    model.decoder = CaptureDecoder()

    latent = torch.zeros(2, 3, 3)
    condition = torch.full((3, 3), 5.0)

    model._generate(
        latent_samples=latent,
        condition=condition,
    )

    assert captured["value"].shape == (6, 3)


@pytest.mark.pruned
def test_generate_ignores_none_condition():
    model = make_model(
        config=make_condition_config(),
    )

    model.decoder = nn.Identity()

    latent = torch.randn(2, 3, 3)
    result = model._generate(
        latent_samples=latent,
        condition=None,
    )

    assert result.shape == (2, 3, 3)


@pytest.mark.pruned
def test_forward_with_condition_embedding():
    model = make_model(
        config=make_condition_config(),
    )

    result = model.forward(
        make_forward_request(
            sample_size=2,
        )
    )

    assert result.output.shape == (2, 3, 1, 4)
    assert result.cond_mu.shape == (3, 4)
    assert result.cond_log_var is None


@pytest.mark.pruned
def test_forward_with_condition_dependent_latent():
    model = make_model(
        config=make_condition_latent_config(),
    )

    result = model.forward(
        make_forward_request(
            sample_size=2,
        )
    )

    assert result.output.shape == (2, 3, 1, 4)
    assert result.cond_mu.shape == (3, 3)
    assert result.cond_log_var.shape == (3, 3)


def test_forward_with_added_features():
    model = make_model(
        added_features_dim=2,
    )

    result = model.forward(
        make_forward_request(
            added_features=torch.randn(3, 1, 2),
            sample_size=2,
        )
    )

    assert result.output.shape == (2, 3, 1, 4)


def test_forward_with_masks():
    model = make_model(
        config=make_condition_config(),
    )

    target = torch.randn(3, 1, 4)
    condition = torch.randn(3, 1, 4)

    result = model.forward(
        make_forward_request(
            target=target,
            target_mask=torch.ones_like(target),
            condition=condition,
            condition_mask=torch.ones_like(condition),
        )
    )

    assert result.output.shape == (2, 3, 1, 4)


def test_forward_clamps_minimum_posterior_variance(
    monkeypatch,
):
    model = make_model()

    minimum = torch.tensor(-0.5)

    monkeypatch.setattr(
        model,
        "_recognition",
        lambda **kwargs: (
            torch.zeros(3, 3),
            torch.full((3, 3), -10.0),
        ),
    )
    monkeypatch.setattr(
        model,
        "_sample",
        lambda mu, log_var, sample_size: torch.zeros(
            sample_size,
            mu.shape[0],
            mu.shape[1],
        ),
    )

    result = model.forward(
        make_forward_request(
            min_posterior_variance=minimum,
        )
    )

    torch.testing.assert_close(
        result.log_var,
        torch.full((3, 3), -0.5),
    )


@pytest.mark.pruned
def test_forward_without_minimum_variance_does_not_clamp(
    monkeypatch,
):
    model = make_model()

    original_log_var = torch.full((3, 3), -10.0)

    monkeypatch.setattr(
        model,
        "_recognition",
        lambda **kwargs: (
            torch.zeros(3, 3),
            original_log_var,
        ),
    )
    monkeypatch.setattr(
        model,
        "_sample",
        lambda mu, log_var, sample_size: torch.zeros(
            sample_size,
            mu.shape[0],
            mu.shape[1],
        ),
    )

    result = model.forward(
        make_forward_request(
            min_posterior_variance=None,
        )
    )

    assert result.log_var is original_log_var


@pytest.mark.pruned
def test_predict_uses_condition_dependent_prior(
    monkeypatch,
):
    model = make_model(
        config=make_condition_latent_config(),
    )

    cond_mu = torch.zeros(3, 3)
    cond_log_var = torch.ones(3, 3)
    latent = torch.full((4, 3, 3), 2.0)

    monkeypatch.setattr(
        model,
        "_condition",
        lambda **kwargs: (
            cond_mu,
            cond_log_var,
        ),
    )

    sample = MagicMock(return_value=latent)
    monkeypatch.setattr(
        model,
        "_sample",
        sample,
    )

    result = model.predict(
        make_predict_request(
            sample_size=4,
            nstds=2.5,
        )
    )

    sample.assert_called_once_with(
        cond_mu,
        cond_log_var,
        4,
        std=2.5,
    )
    assert result.output.shape == (4, 3, 1, 4)


@pytest.mark.pruned
def test_predict_accepts_user_latent_samples():
    model = make_model()
    latent = torch.randn(4, 3, 3)

    result = model.predict(
        make_predict_request(
            sample_size=4,
            latent_samples=latent,
        )
    )

    assert result.output.shape == (4, 3, 1, 4)


@pytest.mark.parametrize(
    "shape",
    [
        (3, 3),
        (1, 3, 3),
        (4, 2, 3),
        (4, 3, 2),
    ],
)
def test_predict_rejects_invalid_user_latent_shape(shape):
    model = make_model()

    with pytest.raises(
        ValueError,
        match="latent_samples",
    ):
        model.predict(
            make_predict_request(
                sample_size=4,
                latent_samples=torch.randn(*shape),
            )
        )


class DummyFlowOutput:
    def __init__(self, samples):
        self.e_samples = samples


class DummyPriorFlow:
    def __init__(self, condition_size=None):
        self.condition_size = condition_size
        self.calls = []

    def inverse(
        self,
        samples,
        condition,
    ):
        self.calls.append(
            (
                samples,
                condition,
            )
        )
        return DummyFlowOutput(samples + 1)


@pytest.mark.pruned
def test_predict_applies_unconditional_prior_flow():
    model = make_model()
    flow = DummyPriorFlow(condition_size=None)

    result = model.predict(
        make_predict_request(
            prior_flow=flow,
            sample_size=4,
        )
    )

    assert len(flow.calls) == 1

    samples, condition = flow.calls[0]

    assert samples.shape == (12, 3)
    assert condition is None
    assert result.output.shape == (4, 3, 1, 4)


@pytest.mark.pruned
def test_predict_applies_conditioned_prior_flow():
    config = make_condition_config()
    model = make_model(config=config)

    flow = DummyPriorFlow(condition_size=4)

    result = model.predict(
        make_predict_request(
            prior_flow=flow,
            sample_size=4,
        )
    )

    assert len(flow.calls) == 1

    samples, condition = flow.calls[0]

    assert samples.shape == (12, 3)
    assert condition.shape == (12, 4)
    assert result.output.shape == (4, 3, 1, 4)


@pytest.mark.pruned
def test_predict_does_not_apply_flow_to_user_latent_samples():
    model = make_model()
    flow = DummyPriorFlow(condition_size=None)
    latent = torch.randn(4, 3, 3)

    model.predict(
        make_predict_request(
            prior_flow=flow,
            sample_size=4,
            latent_samples=latent,
        )
    )

    assert flow.calls == []


def test_predict_returns_condition_statistics():
    model = make_model(
        config=make_condition_latent_config(),
    )

    result = model.predict(
        make_predict_request(
            sample_size=2,
        )
    )

    assert result.cond_mu.shape == (3, 3)
    assert result.cond_log_var.shape == (3, 3)


@pytest.mark.pruned
def test_forward_supports_backward():
    model = make_model()

    target = torch.randn(
        3,
        1,
        4,
        requires_grad=True,
    )

    result = model.forward(
        make_forward_request(
            target=target,
            sample_size=2,
        )
    )

    result.output.sum().backward()

    assert target.grad is not None

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]

    assert any(parameter.grad is not None for parameter in trainable_parameters)


@pytest.mark.pruned
def test_predict_output_is_finite():
    model = make_model(
        config=make_condition_config(),
    )

    result = model.predict(
        make_predict_request(
            sample_size=10,
        )
    )

    assert torch.isfinite(result.output).all()