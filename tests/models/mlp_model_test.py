from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

import cccma_ppp.models.mlp_model as mlp_module
from cccma_ppp.core.deterministic_module import deterministicOutput
from cccma_ppp.generic.runtime import RuntimeContext
from cccma_ppp.models.mlp_model import (
    Autoencoder,
    AutoencoderConfig,
    cVAE_MLP,
    cVAE_MLPConfig,
)


torch.manual_seed(0)
np.random.seed(0)


@pytest.fixture(autouse=True)
def reset_runtime_metadata():
    original_input = RuntimeContext.INPUT_VAR_METADATA
    original_target = RuntimeContext.TARGET_VAR_METADATA

    RuntimeContext.INPUT_VAR_METADATA = {}
    RuntimeContext.TARGET_VAR_METADATA = {}

    yield

    RuntimeContext.INPUT_VAR_METADATA = original_input
    RuntimeContext.TARGET_VAR_METADATA = original_target


class DummyFlowOutput:
    def __init__(self, samples):
        self.e_samples = samples


class DummyPriorFlow:
    def __init__(self, condition_size=None):
        self.condition_size = condition_size
        self.calls = []

    def inverse(self, latent_samples, condition=None):
        self.calls.append((latent_samples, condition))
        return DummyFlowOutput(latent_samples + 1.0)


def make_checkpoint(
    *,
    input_shape=(4,),
    output_shape=(4,),
    input_metadata=None,
    output_metadata=None,
):
    return SimpleNamespace(
        checkpoint_input_shape=np.asarray(input_shape),
        checkpoint_output_shape=np.asarray(output_shape),
        checkpoint_input_var_metadata=(
            {} if input_metadata is None else input_metadata
        ),
        checkpoint_output_var_metadata=(
            {} if output_metadata is None else output_metadata
        ),
    )


def make_cvae_config(**kwargs):
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
        "init_method": "trunc_normal",
    }
    defaults.update(kwargs)
    return cVAE_MLPConfig(**defaults)


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
        "init_method": "trunc_normal",
    }
    defaults.update(kwargs)
    return cVAE_MLPConfig(**defaults)


def make_cvae_model(
    *,
    config=None,
    input_shape=None,
    output_shape=None,
    added_features_dim=None,
):
    if config is None:
        config = make_cvae_config()

    if input_shape is None:
        input_shape = np.asarray([4])

    if output_shape is None:
        output_shape = np.asarray([4])

    return cVAE_MLP(
        config=config,
        input_shape=input_shape,
        output_shape=output_shape,
        added_features_dim=added_features_dim,
    )


def make_predict_request(
    *,
    condition=None,
    condition_mask=None,
    added_features=None,
    prior_flow=None,
    latent_samples=None,
    nstds=1.0,
    sample_size=2,
):
    if condition is None:
        condition = torch.randn(3, 1, 4)

    return SimpleNamespace(
        condition=condition,
        condition_mask=condition_mask,
        added_features=added_features,
        prior_flow=prior_flow,
        latent_samples=latent_samples,
        nstds=nstds,
        sample_size=sample_size,
    )


def make_autoencoder_config(**kwargs):
    defaults = {
        "encoder_hidden_dims": [8, 4],
        "decoder_hidden_dims": [8],
        "batch_normalization": False,
        "dropout_rate": None,
        "append_mode": 1,
        "init_method": "trunc_normal",
    }
    defaults.update(kwargs)
    return AutoencoderConfig(**defaults)


def make_autoencoder(
    *,
    config=None,
    input_shape=None,
    output_shape=None,
    added_features_dim=None,
):
    if config is None:
        config = make_autoencoder_config()

    if input_shape is None:
        input_shape = np.asarray([4])

    if output_shape is None:
        output_shape = np.asarray([4])

    return Autoencoder(
        config=config,
        input_shape=input_shape,
        output_shape=output_shape,
        added_features_dim=added_features_dim,
    )


@pytest.mark.pruned
def test_cvae_config_disables_decoder_condition_without_embedding():
    config = make_cvae_config(
        condition_embedding_dims=None,
        condemb_to_decoder=True,
    )

    assert config.condemb_to_decoder is False


@pytest.mark.pruned
def test_cvae_config_preserves_decoder_condition_with_embedding():
    config = make_condition_config(
        condemb_to_decoder=True,
    )

    assert config.condemb_to_decoder is True


def test_cvae_config_empty_encoder_defaults_empty_decoder():
    config = cVAE_MLPConfig(
        encoder_hidden_dims=[],
        latent_size=3,
        decoder_hidden_dims=None,
    )

    assert config.decoder_hidden_dims == []


@pytest.mark.pruned
def test_cvae_config_single_encoder_defaults_empty_decoder():
    config = cVAE_MLPConfig(
        encoder_hidden_dims=[8],
        latent_size=3,
        decoder_hidden_dims=None,
    )

    assert config.decoder_hidden_dims == []


@pytest.mark.pruned
def test_cvae_config_mirrors_encoder_for_decoder():
    config = cVAE_MLPConfig(
        encoder_hidden_dims=[16, 8, 4],
        latent_size=3,
        decoder_hidden_dims=None,
    )

    assert config.decoder_hidden_dims == [8, 16]


@pytest.mark.pruned
def test_cvae_config_preserves_explicit_decoder():
    config = make_cvae_config(
        decoder_hidden_dims=[7, 5],
    )

    assert config.decoder_hidden_dims == [7, 5]


@pytest.mark.parametrize(
    "dropout_rate",
    [
        None,
        0.0,
        0.25,
        1.0,
    ],
)
def test_cvae_config_accepts_configured_dropout(dropout_rate):
    config = make_cvae_config(
        dropout_rate=dropout_rate,
    )

    assert config.dropout_rate == dropout_rate


@pytest.mark.pruned
def test_cvae_config_condition_dependent_latent_can_skip_decoder_condition():
    config = make_condition_latent_config(
        condemb_to_decoder=False,
    )

    assert config.condition_dependant_latent is True
    assert config.condemb_to_decoder is False


@pytest.mark.pruned
def test_cvae_config_independent_latent_requires_decoder_condition():
    with pytest.raises(
        ValueError,
        match="condition embedding has to be passed to decoder",
    ):
        make_condition_config(
            condition_dependant_latent=False,
            condemb_to_decoder=False,
        )


@pytest.mark.pruned
def test_cvae_config_build_constructs_model():
    config = make_cvae_config()

    model = config.build(
        input_shape=np.asarray([4]),
        output_shape=np.asarray([4]),
        added_features_dim=2,
    )

    assert isinstance(model, cVAE_MLP)
    assert model.config is config
    assert model.added_features_dim == 2


@pytest.mark.pruned
def test_cvae_config_build_passes_arguments(monkeypatch):
    config = make_cvae_config()
    expected = object()
    captured = {}

    def fake_model(
        *,
        config,
        input_shape,
        output_shape,
        added_features_dim,
    ):
        captured.update(
            config=config,
            input_shape=input_shape,
            output_shape=output_shape,
            added_features_dim=added_features_dim,
        )
        return expected

    monkeypatch.setattr(
        mlp_module,
        "cVAE_MLP",
        fake_model,
    )

    input_shape = np.asarray([4])
    output_shape = np.asarray([5])

    result = config.build(
        input_shape=input_shape,
        output_shape=output_shape,
        added_features_dim=2,
    )

    assert result is expected
    assert captured == {
        "config": config,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "added_features_dim": 2,
    }


@pytest.mark.pruned
def test_cvae_model_basic_state():
    model = make_cvae_model()

    assert model.generative_modeling is True
    assert model.input_shape == 4
    assert model.output_shape == 4
    assert model.latent_size == 3
    assert model.added_features_dim == 0
    assert model.add_condition_size == 0
    assert isinstance(model.encoder, nn.Sequential)
    assert isinstance(model.decoder, nn.Sequential)
    assert isinstance(model.mu, nn.Linear)
    assert isinstance(model.log_var, nn.Linear)


@pytest.mark.pruned
def test_cvae_model_preserves_added_feature_dimension():
    model = make_cvae_model(
        added_features_dim=2,
    )

    assert model.added_features_dim == 2


@pytest.mark.pruned
def test_cvae_model_rejects_wrong_output_rank():
    with pytest.raises(
        RuntimeError,
        match="MLP models should create 1D outputs",
    ):
        make_cvae_model(
            output_shape=np.asarray([2, 2]),
        )


@pytest.mark.pruned
def test_cvae_model_uses_condition_in_decoder_dimensions():
    model = make_cvae_model(
        config=make_condition_config(),
    )

    first_decoder = next(
        layer for layer in model.decoder if isinstance(layer, nn.Linear)
    )

    assert model.add_condition_size == 4
    assert first_decoder.in_features == 7


@pytest.mark.pruned
def test_cvae_model_uses_added_features_in_decoder_dimensions():
    model = make_cvae_model(
        added_features_dim=2,
    )

    first_decoder = next(
        layer for layer in model.decoder if isinstance(layer, nn.Linear)
    )

    assert first_decoder.in_features == 5


@pytest.mark.pruned
def test_cvae_model_builds_condition_embedding():
    model = make_cvae_model(
        config=make_condition_config(),
    )

    assert isinstance(model.embedding, nn.Sequential)
    assert isinstance(model.embedding[-1], nn.Linear)
    assert model.embedding[-1].out_features == 4


@pytest.mark.pruned
def test_cvae_model_builds_condition_latent_distribution():
    model = make_cvae_model(
        config=make_condition_latent_config(),
    )

    assert isinstance(model.condition_mu, nn.Linear)
    assert isinstance(model.condition_log_var, nn.Linear)
    assert model.condition_mu.out_features == 3
    assert model.condition_log_var.out_features == 3


@pytest.mark.pruned
def test_cvae_model_condition_flow_uses_embedding_projection():
    config = make_condition_latent_config()
    config.condition_dependant_flow = True

    model = make_cvae_model(
        config=config,
    )

    assert model.condition_dependant_flow is True
    assert isinstance(model.embedding[-1], nn.Linear)
    assert not hasattr(model, "condition_mu")
    assert not hasattr(model, "condition_log_var")


@pytest.mark.pruned
def test_cvae_model_builds_dropout_layers():
    model = make_cvae_model(
        config=make_cvae_config(
            dropout_rate=0.25,
        )
    )

    assert any(isinstance(layer, nn.Dropout) for layer in model.encoder)
    assert any(isinstance(layer, nn.Dropout) for layer in model.decoder)


@pytest.mark.pruned
def test_cvae_model_builds_batch_normalization_layers():
    model = make_cvae_model(
        config=make_cvae_config(
            batch_normalization=True,
        )
    )

    assert any(isinstance(layer, nn.BatchNorm1d) for layer in model.encoder)
    assert any(isinstance(layer, nn.BatchNorm1d) for layer in model.decoder)


def test_cvae_condition_embedding_builds_dropout_and_batchnorm():
    model = make_cvae_model(
        config=make_condition_config(
            dropout_rate=0.25,
            batch_normalization=True,
        )
    )

    assert any(isinstance(layer, nn.Dropout) for layer in model.embedding)
    assert any(isinstance(layer, nn.BatchNorm1d) for layer in model.embedding)


@pytest.mark.pruned
def test_cvae_decoder_final_layer_has_no_activation():
    model = make_cvae_model()

    assert isinstance(model.decoder[-1], nn.Linear)


@pytest.mark.pruned
def test_cvae_initialization_calls_initialize_weights(monkeypatch):
    called = {}

    def fake_initialize(self, method):
        called["method"] = method

    monkeypatch.setattr(
        cVAE_MLP,
        "_initialize_weights",
        fake_initialize,
    )

    make_cvae_model(
        config=make_cvae_config(
            init_method="xavier",
        )
    )

    assert called["method"] == "xavier"


def test_cvae_checkpoint_calls_load_state_dict(monkeypatch):
    config = make_cvae_config()
    checkpoint = make_checkpoint()
    config.checkpoint_config = checkpoint
    called = {}

    def fake_load(self, value):
        called["checkpoint"] = value

    monkeypatch.setattr(
        cVAE_MLP,
        "_load_state_dict",
        fake_load,
    )

    model = make_cvae_model(
        config=config,
    )

    assert isinstance(model, cVAE_MLP)
    assert called["checkpoint"] is checkpoint


def test_cvae_checkpoint_input_shape_mismatch():
    config = make_cvae_config()
    config.checkpoint_config = make_checkpoint(
        input_shape=(5,),
    )

    with pytest.raises(
        RuntimeError,
        match="requested input shape",
    ):
        make_cvae_model(
            config=config,
        )


@pytest.mark.pruned
def test_cvae_checkpoint_output_shape_mismatch():
    config = make_cvae_config()
    config.checkpoint_config = make_checkpoint(
        output_shape=(5,),
    )

    with pytest.raises(
        RuntimeError,
        match="requested output shape",
    ):
        make_cvae_model(
            config=config,
        )


@pytest.mark.pruned
def test_cvae_checkpoint_input_metadata_mismatch():
    RuntimeContext.INPUT_VAR_METADATA = {
        "current": "input",
    }

    config = make_cvae_config()
    config.checkpoint_config = make_checkpoint(
        input_metadata={
            "checkpoint": "input",
        },
    )

    with pytest.raises(
        RuntimeError,
        match="consistent input variables",
    ):
        make_cvae_model(
            config=config,
        )


@pytest.mark.pruned
def test_cvae_checkpoint_output_metadata_mismatch():
    RuntimeContext.TARGET_VAR_METADATA = {
        "current": "target",
    }

    config = make_cvae_config()
    config.checkpoint_config = make_checkpoint(
        output_metadata={
            "checkpoint": "target",
        },
    )

    with pytest.raises(
        RuntimeError,
        match="consistent output variables",
    ):
        make_cvae_model(
            config=config,
        )


@pytest.mark.pruned
def test_cvae_recognition_without_optional_values():
    model = make_cvae_model()

    mu, log_var = model._recognition(
        x=torch.randn(3, 1, 4),
        x_mask=None,
    )

    assert mu.shape == (3, 3)
    assert log_var.shape == (3, 3)


@pytest.mark.pruned
def test_cvae_recognition_applies_mask():
    model = make_cvae_model()
    captured = {}

    class CaptureEncoder(nn.Module):
        def forward(self, value):
            captured["value"] = value
            return torch.zeros(value.shape[0], 6)

    model.encoder = CaptureEncoder()

    x = torch.ones(2, 1, 4)
    mask = torch.zeros_like(x)

    model._recognition(
        x=x,
        x_mask=mask,
    )

    torch.testing.assert_close(
        captured["value"],
        torch.zeros(2, 4),
    )


@pytest.mark.pruned
def test_cvae_recognition_concatenates_condition():
    model = make_cvae_model()
    model.encoder = nn.Identity()
    model.mu = nn.Identity()
    model.log_var = nn.Identity()

    mu, log_var = model._recognition(
        x=torch.ones(1, 1, 4),
        x_mask=None,
        condition=torch.full((1, 3), 2.0),
    )

    expected = torch.tensor(
        [
            [
                1.0,
                1.0,
                1.0,
                1.0,
                2.0,
                2.0,
                2.0,
            ]
        ]
    )

    torch.testing.assert_close(mu, expected)
    torch.testing.assert_close(log_var, expected)


@pytest.mark.pruned
def test_cvae_recognition_concatenates_features():
    model = make_cvae_model(
        added_features_dim=2,
    )
    model.encoder = nn.Identity()
    model.mu = nn.Identity()
    model.log_var = nn.Identity()

    mu, _ = model._recognition(
        x=torch.ones(1, 1, 4),
        x_mask=None,
        added_features=torch.full((1, 1, 2), 3.0),
    )

    torch.testing.assert_close(
        mu,
        torch.tensor(
            [
                [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    3.0,
                    3.0,
                ]
            ]
        ),
    )


@pytest.mark.pruned
def test_cvae_recognition_concatenates_condition_then_features():
    model = make_cvae_model(
        added_features_dim=2,
    )
    model.encoder = nn.Identity()
    model.mu = nn.Identity()
    model.log_var = nn.Identity()

    mu, _ = model._recognition(
        x=torch.ones(1, 1, 4),
        x_mask=None,
        condition=torch.full((1, 3), 2.0),
        added_features=torch.full((1, 1, 2), 3.0),
    )

    torch.testing.assert_close(
        mu,
        torch.tensor(
            [
                [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    2.0,
                    2.0,
                    2.0,
                    3.0,
                    3.0,
                ]
            ]
        ),
    )


@pytest.mark.pruned
def test_cvae_condition_without_embedding_returns_none():
    model = make_cvae_model()

    cond_mu, cond_log_var = model._condition(
        condition=None,
    )

    assert cond_mu is None
    assert cond_log_var is None


@pytest.mark.pruned
def test_cvae_condition_embedding_without_mask_or_features():
    model = make_cvae_model(
        config=make_condition_config(),
    )

    cond_mu, cond_log_var = model._condition(
        condition=torch.randn(2, 1, 4),
    )

    assert cond_mu.shape == (2, 4)
    assert cond_log_var is None


@pytest.mark.pruned
def test_cvae_condition_applies_mask():
    model = make_cvae_model(
        config=make_condition_config(),
    )
    captured = {}

    class CaptureEmbedding(nn.Module):
        def forward(self, value):
            captured["value"] = value
            return torch.zeros(value.shape[0], 4)

    model.embedding = CaptureEmbedding()

    condition = torch.ones(2, 1, 4)
    mask = torch.zeros_like(condition)

    model._condition(
        condition=condition,
        condition_mask=mask,
    )

    torch.testing.assert_close(
        captured["value"],
        torch.zeros(2, 4),
    )


def test_cvae_condition_concatenates_features():
    model = make_cvae_model(
        config=make_condition_config(),
        added_features_dim=2,
    )
    captured = {}

    class CaptureEmbedding(nn.Module):
        def forward(self, value):
            captured["value"] = value
            return torch.zeros(value.shape[0], 4)

    model.embedding = CaptureEmbedding()

    model._condition(
        condition=torch.ones(1, 1, 4),
        added_features=torch.full((1, 1, 2), 2.0),
    )

    torch.testing.assert_close(
        captured["value"],
        torch.tensor(
            [
                [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    2.0,
                    2.0,
                ]
            ]
        ),
    )


@pytest.mark.pruned
def test_cvae_condition_returns_latent_distribution():
    model = make_cvae_model(
        config=make_condition_latent_config(),
    )

    cond_mu, cond_log_var = model._condition(
        condition=torch.randn(2, 1, 4),
    )

    assert cond_mu.shape == (2, 3)
    assert cond_log_var.shape == (2, 3)


@pytest.mark.pruned
def test_cvae_condition_flow_returns_embedding_without_variance():
    config = make_condition_latent_config()
    config.condition_dependant_flow = True

    model = make_cvae_model(
        config=config,
    )

    cond_mu, cond_log_var = model._condition(
        condition=torch.randn(2, 1, 4),
    )

    assert cond_mu.shape == (2, 3)
    assert cond_log_var is None


@pytest.mark.pruned
def test_cvae_generate_without_optional_values():
    model = make_cvae_model()

    output = model._generate(
        latent_samples=torch.randn(3, 2, 3),
    )

    assert output.shape == (3, 2, 4)


@pytest.mark.pruned
def test_cvae_generate_with_features():
    model = make_cvae_model(
        added_features_dim=2,
    )
    captured = {}

    class CaptureDecoder(nn.Module):
        def forward(self, value):
            captured["value"] = value
            return torch.zeros(value.shape[0], 4)

    model.decoder = CaptureDecoder()

    output = model._generate(
        latent_samples=torch.zeros(2, 3, 3),
        added_features=torch.full((3, 1, 2), 4.0),
    )

    assert output.shape == (2, 3, 4)
    assert captured["value"].shape == (6, 5)


@pytest.mark.pruned
def test_cvae_generate_with_decoder_condition():
    model = make_cvae_model(
        config=make_condition_config(),
    )
    captured = {}

    class CaptureDecoder(nn.Module):
        def forward(self, value):
            captured["value"] = value
            return torch.zeros(value.shape[0], 4)

    model.decoder = CaptureDecoder()

    model._generate(
        latent_samples=torch.zeros(2, 3, 3),
        condition=torch.full((3, 4), 5.0),
    )

    assert captured["value"].shape == (6, 7)


@pytest.mark.pruned
def test_cvae_generate_skips_condition_when_decoder_flag_false():
    model = make_cvae_model(
        config=make_condition_latent_config(
            condemb_to_decoder=False,
        )
    )
    captured = {}

    class CaptureDecoder(nn.Module):
        def forward(self, value):
            captured["value"] = value
            return torch.zeros(value.shape[0], 4)

    model.decoder = CaptureDecoder()

    model._generate(
        latent_samples=torch.zeros(2, 3, 3),
        condition=torch.full((3, 3), 5.0),
    )

    assert captured["value"].shape == (6, 3)


@pytest.mark.pruned
def test_cvae_generate_with_features_and_condition():
    model = make_cvae_model(
        config=make_condition_config(),
        added_features_dim=2,
    )
    captured = {}

    class CaptureDecoder(nn.Module):
        def forward(self, value):
            captured["value"] = value
            return torch.zeros(value.shape[0], 4)

    model.decoder = CaptureDecoder()

    model._generate(
        latent_samples=torch.zeros(2, 3, 3),
        condition=torch.full((3, 4), 5.0),
        added_features=torch.full((3, 1, 2), 4.0),
    )

    assert captured["value"].shape == (6, 9)


@pytest.mark.pruned
def test_cvae_sample_shape():
    model = make_cvae_model()

    samples = model._sample(
        mu=torch.zeros(2, 3),
        log_var=torch.zeros(2, 3),
        sample_size=4,
    )

    assert samples.shape == (4, 2, 3)


@pytest.mark.pruned
def test_cvae_sample_uses_requested_std(monkeypatch):
    model = make_cvae_model()
    captured = {}

    class FakeDistribution:
        def sample(self, shape):
            captured["shape"] = shape
            return torch.zeros(*shape, 2, 3)

    def fake_get_normal(reference, std):
        captured["reference"] = reference
        captured["std"] = std
        return FakeDistribution()

    monkeypatch.setattr(
        model,
        "_get_normal",
        fake_get_normal,
    )

    result = model._sample(
        mu=torch.ones(2, 3),
        log_var=torch.zeros(2, 3),
        sample_size=5,
        std=2.5,
    )

    assert result.shape == (5, 2, 3)
    assert captured["shape"] == (5,)
    assert captured["std"] == pytest.approx(2.5)


@pytest.mark.pruned
def test_cvae_get_normal_values():
    model = make_cvae_model()
    reference = torch.zeros(
        2,
        3,
        dtype=torch.float64,
    )

    distribution = model._get_normal(
        reference,
        std=2.0,
    )

    torch.testing.assert_close(
        distribution.loc,
        torch.zeros_like(reference),
    )
    torch.testing.assert_close(
        distribution.scale,
        torch.full_like(reference, 2.0),
    )


@pytest.mark.pruned
def test_cvae_forward_with_mask():
    model = make_cvae_model()
    x = torch.randn(3, 1, 4)

    result = model.forward(
        x=x,
        x_mask=torch.ones_like(x),
        sample_size=2,
    )

    assert result.output.shape == (2, 3, 1, 4)


@pytest.mark.pruned
def test_cvae_forward_with_condition():
    model = make_cvae_model(
        config=make_condition_config(),
    )

    result = model.forward(
        x=torch.randn(3, 1, 4),
        condition=torch.randn(3, 1, 4),
        sample_size=2,
    )

    assert result.cond_mu.shape == (3, 4)
    assert result.cond_log_var is None


def test_cvae_forward_with_condition_latent():
    model = make_cvae_model(
        config=make_condition_latent_config(),
    )

    result = model.forward(
        x=torch.randn(3, 1, 4),
        condition=torch.randn(3, 1, 4),
        sample_size=2,
    )

    assert result.cond_mu.shape == (3, 3)
    assert result.cond_log_var.shape == (3, 3)


def test_cvae_forward_with_features():
    model = make_cvae_model(
        added_features_dim=2,
    )

    result = model.forward(
        x=torch.randn(3, 1, 4),
        added_features=torch.randn(3, 1, 2),
        sample_size=2,
    )

    assert result.output.shape == (2, 3, 1, 4)


def test_cvae_forward_clamps_log_variance(monkeypatch):
    model = make_cvae_model()

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
            3,
            3,
        ),
    )

    result = model.forward(
        x=torch.randn(3, 1, 4),
        min_posterior_variance=torch.tensor(-0.5),
    )

    torch.testing.assert_close(
        result.log_var,
        torch.full((3, 3), -0.5),
    )


@pytest.mark.pruned
def test_cvae_forward_without_clamp_preserves_log_variance(monkeypatch):
    model = make_cvae_model()
    expected = torch.full((3, 3), -10.0)

    monkeypatch.setattr(
        model,
        "_recognition",
        lambda **kwargs: (
            torch.zeros(3, 3),
            expected,
        ),
    )
    monkeypatch.setattr(
        model,
        "_sample",
        lambda mu, log_var, sample_size: torch.zeros(
            sample_size,
            3,
            3,
        ),
    )

    result = model.forward(
        x=torch.randn(3, 1, 4),
        min_posterior_variance=None,
    )

    assert result.log_var is expected


@pytest.mark.pruned
def test_cvae_predict_standard_prior():
    model = make_cvae_model()

    result = model.predict(
        make_predict_request(
            sample_size=4,
        )
    )

    assert result.output.shape == (4, 3, 1, 4)
    assert result.mu is None
    assert result.log_var is None
    assert result.samples is None
    assert result.cond_mu is None
    assert result.cond_log_var is None


def test_cvae_predict_condition_prior(monkeypatch):
    model = make_cvae_model(
        config=make_condition_latent_config(),
    )

    cond_mu = torch.zeros(3, 3)
    cond_log_var = torch.ones(3, 3)
    latent_samples = torch.zeros(4, 3, 3)

    monkeypatch.setattr(
        model,
        "_condition",
        lambda **kwargs: (
            cond_mu,
            cond_log_var,
        ),
    )

    sample = MagicMock(
        return_value=latent_samples,
    )
    monkeypatch.setattr(
        model,
        "_sample",
        sample,
    )

    result = model.predict(
        make_predict_request(
            sample_size=4,
            nstds=2.0,
        )
    )

    sample.assert_called_once_with(
        cond_mu,
        cond_log_var,
        4,
        std=2.0,
    )
    assert result.output.shape == (4, 3, 1, 4)


@pytest.mark.pruned
def test_cvae_predict_condition_flow_uses_standard_prior(monkeypatch):
    config = make_condition_latent_config()
    config.condition_dependant_flow = True

    model = make_cvae_model(
        config=config,
    )

    distribution = MagicMock()
    distribution.sample.return_value = torch.zeros(4, 3, 3)
    normal = MagicMock(
        return_value=distribution,
    )

    monkeypatch.setattr(
        model,
        "_get_normal",
        normal,
    )

    model.predict(
        make_predict_request(
            sample_size=4,
        )
    )

    normal.assert_called_once()
    distribution.sample.assert_called_once_with((4,))


def test_cvae_predict_accepts_user_latent_samples():
    model = make_cvae_model()
    latent_samples = torch.randn(4, 3, 3)

    result = model.predict(
        make_predict_request(
            sample_size=4,
            latent_samples=latent_samples,
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
def test_cvae_predict_rejects_invalid_latent_shape(shape):
    model = make_cvae_model()

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


@pytest.mark.pruned
def test_cvae_predict_applies_unconditional_flow():
    model = make_cvae_model()
    flow = DummyPriorFlow(
        condition_size=None,
    )

    result = model.predict(
        make_predict_request(
            sample_size=4,
            prior_flow=flow,
        )
    )

    assert len(flow.calls) == 1

    samples, condition = flow.calls[0]

    assert samples.shape == (12, 3)
    assert condition is None
    assert result.output.shape == (4, 3, 1, 4)


def test_cvae_predict_applies_conditioned_flow():
    model = make_cvae_model(
        config=make_condition_config(),
    )
    flow = DummyPriorFlow(
        condition_size=4,
    )

    result = model.predict(
        make_predict_request(
            sample_size=4,
            prior_flow=flow,
        )
    )

    assert len(flow.calls) == 1

    samples, condition = flow.calls[0]

    assert samples.shape == (12, 3)
    assert condition.shape == (12, 4)
    assert result.output.shape == (4, 3, 1, 4)


@pytest.mark.pruned
def test_cvae_predict_skips_flow_for_user_latent_samples():
    model = make_cvae_model()
    flow = DummyPriorFlow()

    model.predict(
        make_predict_request(
            sample_size=4,
            latent_samples=torch.randn(4, 3, 3),
            prior_flow=flow,
        )
    )

    assert flow.calls == []


def test_autoencoder_config_single_encoder_defaults_empty_decoder():
    config = AutoencoderConfig(
        encoder_hidden_dims=[4],
        decoder_hidden_dims=None,
    )

    assert config.decoder_hidden_dims == []


def test_autoencoder_config_mirrors_encoder():
    config = AutoencoderConfig(
        encoder_hidden_dims=[16, 8, 4],
        decoder_hidden_dims=None,
    )

    assert config.decoder_hidden_dims == [8, 16]


@pytest.mark.pruned
def test_autoencoder_config_preserves_decoder_dimensions():
    config = make_autoencoder_config(
        decoder_hidden_dims=[7, 5],
    )

    assert config.decoder_hidden_dims == [7, 5]


@pytest.mark.pruned
def test_autoencoder_config_build_constructs_model():
    config = make_autoencoder_config()

    model = config.build(
        input_shape=np.asarray([4]),
        output_shape=np.asarray([4]),
        added_features_dim=2,
    )

    assert isinstance(model, Autoencoder)
    assert model.config is config
    assert model.added_features_dim == 2


@pytest.mark.pruned
def test_autoencoder_basic_state():
    model = make_autoencoder()

    assert model.generative_modeling is False
    assert model.input_shape == 4
    assert model.output_shape == 4
    assert model.latent_size == 4
    assert model.added_features_dim == 0
    assert isinstance(model.encoder, nn.Sequential)
    assert isinstance(model.decoder, nn.Sequential)


def test_autoencoder_rejects_wrong_output_rank():
    with pytest.raises(
        RuntimeError,
        match="MLP models should create 1D outputs",
    ):
        make_autoencoder(
            output_shape=np.asarray([2, 2]),
        )


@pytest.mark.parametrize(
    ("append_mode", "encoder_input", "decoder_input"),
    [
        (1, 6, 4),
        (2, 4, 6),
        (3, 6, 6),
        (99, 4, 4),
    ],
)
def test_autoencoder_append_mode_architecture(
    append_mode,
    encoder_input,
    decoder_input,
):
    model = make_autoencoder(
        config=make_autoencoder_config(
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

    assert first_encoder.in_features == encoder_input
    assert first_decoder.in_features == decoder_input


def test_autoencoder_builds_dropout_layers():
    model = make_autoencoder(
        config=make_autoencoder_config(
            dropout_rate=0.25,
        )
    )

    assert any(isinstance(layer, nn.Dropout) for layer in model.encoder)
    assert any(isinstance(layer, nn.Dropout) for layer in model.decoder)


def test_autoencoder_builds_batchnorm_layers():
    model = make_autoencoder(
        config=make_autoencoder_config(
            batch_normalization=True,
        )
    )

    assert any(isinstance(layer, nn.BatchNorm1d) for layer in model.encoder)
    assert any(isinstance(layer, nn.BatchNorm1d) for layer in model.decoder)


@pytest.mark.pruned
def test_autoencoder_decoder_final_layer_has_no_activation():
    model = make_autoencoder()

    assert isinstance(model.decoder[-1], nn.Linear)


@pytest.mark.pruned
def test_autoencoder_initialization_calls_initialize_weights(monkeypatch):
    captured = {}

    def fake_initialize(self, method):
        captured["method"] = method

    monkeypatch.setattr(
        Autoencoder,
        "_initialize_weights",
        fake_initialize,
    )

    make_autoencoder(
        config=make_autoencoder_config(
            init_method="xavier",
        )
    )

    assert captured["method"] == "xavier"


def test_autoencoder_checkpoint_calls_load_state_dict(monkeypatch):
    config = make_autoencoder_config()
    checkpoint = make_checkpoint()
    config.checkpoint_config = checkpoint
    captured = {}

    def fake_load(self, value):
        captured["checkpoint"] = value

    monkeypatch.setattr(
        Autoencoder,
        "_load_state_dict",
        fake_load,
    )

    model = make_autoencoder(
        config=config,
    )

    assert isinstance(model, Autoencoder)
    assert captured["checkpoint"] is checkpoint


def test_autoencoder_checkpoint_input_shape_mismatch():
    config = make_autoencoder_config()
    config.checkpoint_config = make_checkpoint(
        input_shape=(5,),
    )

    with pytest.raises(
        RuntimeError,
        match="requested input shape",
    ):
        make_autoencoder(
            config=config,
        )


@pytest.mark.pruned
def test_autoencoder_checkpoint_output_shape_mismatch():
    config = make_autoencoder_config()
    config.checkpoint_config = make_checkpoint(
        output_shape=(5,),
    )

    with pytest.raises(
        RuntimeError,
        match="requested output shape",
    ):
        make_autoencoder(
            config=config,
        )


def test_autoencoder_checkpoint_input_metadata_mismatch():
    RuntimeContext.INPUT_VAR_METADATA = {
        "current": "input",
    }

    config = make_autoencoder_config()
    config.checkpoint_config = make_checkpoint(
        input_metadata={
            "checkpoint": "input",
        },
    )

    with pytest.raises(
        RuntimeError,
        match="consistent input variables",
    ):
        make_autoencoder(
            config=config,
        )


def test_autoencoder_checkpoint_output_metadata_mismatch():
    RuntimeContext.TARGET_VAR_METADATA = {
        "current": "target",
    }

    config = make_autoencoder_config()
    config.checkpoint_config = make_checkpoint(
        output_metadata={
            "checkpoint": "target",
        },
    )

    with pytest.raises(
        RuntimeError,
        match="consistent output variables",
    ):
        make_autoencoder(
            config=config,
        )


@pytest.mark.pruned
def test_autoencoder_forward_without_mask_or_features():
    model = make_autoencoder()
    x = torch.randn(3, 1, 4)

    result = model.forward(
        x=x,
        x_mask=None,
    )

    assert isinstance(result, deterministicOutput)
    assert result.output.shape == x.shape


def test_autoencoder_forward_applies_mask():
    model = make_autoencoder()
    captured = {}

    class CaptureEncoder(nn.Module):
        def forward(self, value):
            captured["value"] = value
            return torch.zeros(value.shape[0], 4)

    model.encoder = CaptureEncoder()
    model.decoder = nn.Linear(4, 4)

    x = torch.ones(2, 1, 4)
    mask = torch.tensor(
        [
            [[1.0, 0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0, 1.0]],
        ]
    )

    model.forward(
        x=x,
        x_mask=mask,
    )

    torch.testing.assert_close(
        captured["value"],
        mask.flatten(start_dim=1),
    )


def test_autoencoder_append_mode_one_forward():
    model = make_autoencoder(
        config=make_autoencoder_config(
            append_mode=1,
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

    model.forward(
        x=torch.ones(2, 1, 4),
        x_mask=None,
        added_features=torch.full((2, 1, 2), 2.0),
    )

    assert captured["encoder"].shape == (2, 6)
    assert captured["decoder"].shape == (2, 4)


def test_autoencoder_append_mode_two_forward():
    model = make_autoencoder(
        config=make_autoencoder_config(
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

    model.forward(
        x=torch.ones(2, 1, 4),
        x_mask=None,
        added_features=torch.full((2, 1, 2), 2.0),
    )

    assert captured["encoder"].shape == (2, 4)
    assert captured["decoder"].shape == (2, 6)


def test_autoencoder_append_mode_three_forward():
    model = make_autoencoder(
        config=make_autoencoder_config(
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

    model.forward(
        x=torch.ones(2, 1, 4),
        x_mask=None,
        added_features=torch.full((2, 1, 2), 2.0),
    )

    assert captured["encoder"].shape == (2, 6)
    assert captured["decoder"].shape == (2, 6)


@pytest.mark.pruned
def test_autoencoder_invalid_append_mode_with_features_raises():
    model = make_autoencoder(
        config=make_autoencoder_config(
            append_mode=99,
        ),
        added_features_dim=0,
    )

    with pytest.raises(UnboundLocalError):
        model.forward(
            x=torch.randn(2, 1, 4),
            x_mask=None,
            added_features=torch.empty(2, 0),
        )


@pytest.mark.pruned
def test_autoencoder_invalid_append_mode_without_features_uses_default_path():
    model = make_autoencoder(
        config=make_autoencoder_config(
            append_mode=99,
        ),
        added_features_dim=0,
    )

    result = model.forward(
        x=torch.randn(2, 1, 4),
        x_mask=None,
        added_features=None,
    )

    assert result.output.shape == (2, 1, 4)


@pytest.mark.pruned
def test_autoencoder_flattens_features():
    model = make_autoencoder(
        config=make_autoencoder_config(
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
        x=torch.randn(2, 1, 4),
        x_mask=None,
        added_features=torch.randn(2, 2, 2),
    )

    assert captured["shape"] == (2, 8)


@pytest.mark.pruned
def test_autoencoder_output_uses_input_batch_and_channel_dimensions():
    model = make_autoencoder(
        input_shape=np.asarray([6]),
        output_shape=np.asarray([10]),
    )

    result = model.forward(
        x=torch.randn(3, 2, 3),
        x_mask=None,
    )

    assert result.output.shape == (3, 2, 5)


@pytest.mark.pruned
def test_autoencoder_forward_supports_backward():
    model = make_autoencoder()
    x = torch.randn(
        3,
        1,
        4,
        requires_grad=True,
    )

    result = model.forward(
        x=x,
        x_mask=None,
    )
    result.output.sum().backward()

    assert x.grad is not None