import dataclasses

import numpy as np
import pytest
import torch

from cccma_ppp.core.cVAE_module import cVAEOutput
from cccma_ppp.models.mlp_models import (
    AutoencoderConfig,
    cVAE_MLPConfig,
    cVAE_MLP,
    Autoencoder,
)
from cccma_ppp.core.deterministic_module import deterministicOutput
from cccma_ppp.core.selectors import cVAEModelSelector, deterministicModelSelector
from cccma_ppp.generic.runtime import RuntimeContext

torch.manual_seed(0)
np.random.seed(0)


@pytest.fixture(autouse=True)
def reset_runtime_context_metadata():
    RuntimeContext.INPUT_VAR_METADATA = {}
    RuntimeContext.TARGET_VAR_METADATA = {}
    yield


@dataclasses.dataclass
class DummyCheckpointConfig:
    checkpoint_input_shape: np.ndarray
    checkpoint_output_shape: np.ndarray
    checkpoint_input_var_metadata: dict = dataclasses.field(default_factory=dict)
    checkpoint_output_var_metadata: dict = dataclasses.field(default_factory=dict)
    strict: bool = True
    freeze_weights: bool = False
    load_path: str = "dummy.pt"


class DummyFlowOutput:
    def __init__(self, e_samples):
        self.e_samples = e_samples


class DummyPriorFlow:
    def __init__(self, condition_size=None):
        self.condition_size = condition_size
        self.called = False
        self.last_condition = None

    def inverse(self, latent_samples, cond=None):
        self.called = True
        self.last_condition = cond
        return DummyFlowOutput(latent_samples)


def x(batch=2, channels=1, features=6):
    return torch.randn(batch, channels, features)


def condition(batch=2, channels=1, features=6):
    return torch.randn(batch, channels, features)


def mask_like(tensor):
    return torch.ones_like(tensor)


def added_features(batch=2, features=3):
    return torch.randn(batch, features)


def make_cvae_config(
    encoder_hidden_dims=None,
    latent_size=4,
    decoder_hidden_dims=None,
    condition_embedding_dims=None,
    condition_dependant_latent=False,
    condemb_to_decoder=True,
    batch_normalization=False,
    dropout_rate=None,
    init_method="trunc_normal",
):
    if encoder_hidden_dims is None:
        encoder_hidden_dims = [8]

    cfg = cVAE_MLPConfig(
        encoder_hidden_dims=encoder_hidden_dims,
        latent_size=latent_size,
        decoder_hidden_dims=decoder_hidden_dims,
        condition_embedding_dims=condition_embedding_dims,
        condition_dependant_latent=condition_dependant_latent,
        condemb_to_decoder=condemb_to_decoder,
        batch_normalization=batch_normalization,
        dropout_rate=dropout_rate,
        init_method=init_method,
    )

    if condition_embedding_dims is not None:
        cfg.condition_embedding_size = condition_embedding_dims[-1]
        cfg.condition_embedding_dims = condition_embedding_dims[:-1]
    else:
        cfg.condition_embedding_size = 0
        cfg.condition_embedding_dims = None
        cfg.condemb_to_decoder = False

    if not hasattr(cfg, "condition_dependant_flow"):
        cfg.condition_dependant_flow = False

    return cfg


def build_cvae(
    *,
    encoder_hidden_dims=None,
    latent_size=4,
    decoder_hidden_dims=None,
    condition_embedding_dims=None,
    condition_dependant_latent=False,
    condemb_to_decoder=True,
    batch_normalization=False,
    dropout_rate=None,
    input_shape=np.array([6]),
    output_shape=np.array([6]),
    added_features_dim=None,
):
    cfg = make_cvae_config(
        encoder_hidden_dims=encoder_hidden_dims,
        latent_size=latent_size,
        decoder_hidden_dims=decoder_hidden_dims,
        condition_embedding_dims=condition_embedding_dims,
        condition_dependant_latent=condition_dependant_latent,
        condemb_to_decoder=condemb_to_decoder,
        batch_normalization=batch_normalization,
        dropout_rate=dropout_rate,
    )

    return cfg.build(
        input_shape=input_shape,
        output_shape=output_shape,
        added_features_dim=added_features_dim,
    )


def make_autoencoder_config(
    encoder_hidden_dims=None,
    decoder_hidden_dims=None,
    batch_normalization=False,
    dropout_rate=None,
    append_mode=1,
    init_method="trunc_normal",
):
    if encoder_hidden_dims is None:
        encoder_hidden_dims = [4]

    cfg = AutoencoderConfig(
        encoder_hidden_dims=encoder_hidden_dims,
        decoder_hidden_dims=decoder_hidden_dims,
        batch_normalization=batch_normalization,
        dropout_rate=dropout_rate,
        init_method=init_method,
    )
    cfg.append_mode = append_mode
    return cfg


def build_autoencoder(
    *,
    encoder_hidden_dims=None,
    decoder_hidden_dims=None,
    batch_normalization=False,
    dropout_rate=None,
    append_mode=1,
    input_shape=np.array([6]),
    output_shape=np.array([6]),
    added_features_dim=None,
):
    cfg = make_autoencoder_config(
        encoder_hidden_dims=encoder_hidden_dims,
        decoder_hidden_dims=decoder_hidden_dims,
        batch_normalization=batch_normalization,
        dropout_rate=dropout_rate,
        append_mode=append_mode,
    )

    return cfg.build(
        input_shape=input_shape,
        output_shape=output_shape,
        added_features_dim=added_features_dim,
    )


def test_cvae_mlp_registered():
    selector = cVAEModelSelector(
        type="mlp",
        config={
            "encoder_hidden_dims": [8],
            "latent_size": 4,
        },
    )

    cfg = selector.get_model_config()

    assert isinstance(cfg, cVAE_MLPConfig)


def test_autoencoder_registered():
    selector = deterministicModelSelector(
        type="mlp",
        config={
            "encoder_hidden_dims": [4],
        },
    )

    cfg = selector.get_model_config()

    assert isinstance(cfg, AutoencoderConfig)


def test_cvae_config_build_returns_model():
    cfg = make_cvae_config()

    model = cfg.build(
        input_shape=np.array([6]),
        output_shape=np.array([6]),
    )

    assert isinstance(model, cVAE_MLP)
    assert model.latent_size == 4
    assert model.generative_modeling is True


def test_cvae_config_sets_condition_dependant_flow_false():
    cfg = make_cvae_config()

    assert getattr(cfg, "condition_dependant_flow", False) is False


def test_cvae_invalid_dropout_low():
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        cfg = make_cvae_config(dropout_rate=-0.1)
        cfg.build(
            input_shape=np.array([6]),
            output_shape=np.array([6]),
        )


def test_cvae_invalid_dropout_high():
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        cfg = make_cvae_config(dropout_rate=1.1)
        cfg.build(
            input_shape=np.array([6]),
            output_shape=np.array([6]),
        )


def test_cvae_decoder_hidden_default_empty_when_encoder_empty():
    cfg = make_cvae_config(encoder_hidden_dims=[])

    model = cfg.build(
        input_shape=np.array([6]),
        output_shape=np.array([6]),
    )

    assert model.decoder_hidden_dims == []


def test_cvae_decoder_hidden_default_reverse_branch():
    cfg = make_cvae_config(
        encoder_hidden_dims=[16, 8, 4],
        latent_size=2,
    )

    model = cfg.build(
        input_shape=np.array([6]),
        output_shape=np.array([6]),
    )

    assert model.decoder_hidden_dims == [8, 16]


def test_cvae_explicit_decoder_hidden_dims():
    cfg = make_cvae_config(
        encoder_hidden_dims=[8],
        decoder_hidden_dims=[7, 6],
    )

    model = cfg.build(
        input_shape=np.array([6]),
        output_shape=np.array([6]),
    )

    assert model.decoder_hidden_dims == [7, 6]


def test_cvae_condition_embedding_none_disables_decoder_condition():
    cfg = make_cvae_config(
        condition_embedding_dims=None,
        condemb_to_decoder=True,
    )

    model = cfg.build(
        input_shape=np.array([6]),
        output_shape=np.array([6]),
    )

    assert model.condemb_to_decoder is False


def test_cvae_condition_embedding_sets_condition_size():
    cfg = make_cvae_config(condition_embedding_dims=[5, 4])

    model = cfg.build(
        input_shape=np.array([6]),
        output_shape=np.array([6]),
    )

    assert model.condition_embedding_size == 4
    assert model.condition_embedding_dims == [5]


def test_cvae_condition_dependant_latent_valid():
    cfg = make_cvae_config(
        latent_size=4,
        condition_embedding_dims=[5, 4],
        condition_dependant_latent=True,
    )

    model = cfg.build(
        input_shape=np.array([6]),
        output_shape=np.array([6]),
    )

    assert model.condition_dependant_latent is True


def test_cvae_condition_dependant_flow_skips_latent_size_assert():
    cfg = make_cvae_config(
        latent_size=3,
        condition_embedding_dims=[5, 4],
        condition_dependant_latent=True,
    )
    cfg.condition_dependant_flow = True

    model = cfg.build(
        input_shape=np.array([6]),
        output_shape=np.array([6]),
    )

    assert model.condition_dependant_flow is True


def test_cvae_build_basic():
    model = build_cvae()

    assert hasattr(model, "encoder")
    assert hasattr(model, "decoder")
    assert hasattr(model, "mu")
    assert hasattr(model, "log_var")
    assert model.input_shape == 6
    assert model.output_shape == 6


def test_cvae_build_with_added_features():
    model = build_cvae(
        encoder_hidden_dims=[8],
        added_features_dim=3,
    )

    assert model.added_features_dim == 3


def test_cvae_build_added_features_none_defaults_zero():
    model = build_cvae()

    assert model.added_features_dim == 0


def test_cvae_build_wrong_output_dims_raises():
    cfg = make_cvae_config()

    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        cfg.build(
            input_shape=np.array([6]),
            output_shape=np.array([2, 3]),
        )


def test_cvae_build_with_dropout_and_batchnorm():
    model = build_cvae(
        encoder_hidden_dims=[8],
        dropout_rate=0.2,
        batch_normalization=True,
    )

    assert any(isinstance(layer, torch.nn.Dropout) for layer in model.encoder)
    assert any(isinstance(layer, torch.nn.BatchNorm1d) for layer in model.encoder)


def test_cvae_build_with_condition_embedding():
    model = build_cvae(
        encoder_hidden_dims=[8],
        condition_embedding_dims=[7, 4],
        added_features_dim=1,
    )

    assert hasattr(model, "embedding")


def test_cvae_build_condition_dependant_latent_layers():
    model = build_cvae(
        encoder_hidden_dims=[8],
        latent_size=4,
        condition_embedding_dims=[7, 4],
        condition_dependant_latent=True,
        added_features_dim=1,
    )

    assert hasattr(model, "condition_mu")
    assert hasattr(model, "condition_log_var")


def test_cvae_build_checkpoint_input_shape_mismatch():
    cfg = make_cvae_config()
    cfg._add_checkpoint_config(
        DummyCheckpointConfig(
            checkpoint_input_shape=np.array([5]),
            checkpoint_output_shape=np.array([6]),
        )
    )

    with pytest.raises(RuntimeError, match="input shape"):
        cfg.build(
            input_shape=np.array([6]),
            output_shape=np.array([6]),
        )


def test_cvae_build_checkpoint_output_shape_mismatch():
    cfg = make_cvae_config()
    cfg._add_checkpoint_config(
        DummyCheckpointConfig(
            checkpoint_input_shape=np.array([6]),
            checkpoint_output_shape=np.array([5]),
        )
    )

    with pytest.raises(RuntimeError, match="output shape"):
        cfg.build(
            input_shape=np.array([6]),
            output_shape=np.array([6]),
        )


def test_cvae_build_checkpoint_success_calls_load(monkeypatch):
    cfg = make_cvae_config()
    cfg._add_checkpoint_config(
        DummyCheckpointConfig(
            checkpoint_input_shape=np.array([6]),
            checkpoint_output_shape=np.array([6]),
        )
    )

    called = {"load": False}

    def fake_load(self, checkpoint_config):
        called["load"] = True

    monkeypatch.setattr(cVAE_MLP, "_load_state_dict", fake_load)

    model = cfg.build(
        input_shape=np.array([6]),
        output_shape=np.array([6]),
    )

    assert isinstance(model, cVAE_MLP)
    assert called["load"] is True


def test_cvae_recognition_plain():
    model = build_cvae()

    mu, log_var = model._recognition(x())

    assert mu.shape == (2, 4)
    assert log_var.shape == (2, 4)


def test_cvae_recognition_with_mask():
    model = build_cvae()

    data = x()
    m = mask_like(data)

    mu, log_var = model._recognition((data, m))

    assert mu.shape == (2, 4)
    assert log_var.shape == (2, 4)


def test_cvae_recognition_with_added_features():
    model = build_cvae(
        encoder_hidden_dims=[8],
        added_features_dim=3,
    )

    mu, log_var = model._recognition(
        x=x(),
        added_features=added_features(),
    )

    assert mu.shape == (2, 4)


def test_cvae_recognition_with_condition():
    model = build_cvae(
        encoder_hidden_dims=[8],
        condition_embedding_dims=[6, 4],
    )

    cond_mu, _ = model._condition(condition=condition())

    mu, log_var = model._recognition(
        x=x(),
        condition=cond_mu,
    )

    assert mu.shape == (2, 4)
    assert log_var.shape == (2, 4)


def test_cvae_condition_none():
    model = build_cvae()

    cond_mu, cond_log_var = model._condition()

    assert cond_mu is None
    assert cond_log_var is None


def test_cvae_condition_plain():
    model = build_cvae(
        encoder_hidden_dims=[8],
        condition_embedding_dims=[5, 4],
    )

    cond_mu, cond_log_var = model._condition(condition=condition())

    assert cond_mu.shape == (2, 4)
    assert cond_log_var is None


def test_cvae_condition_with_mask():
    model = build_cvae(
        encoder_hidden_dims=[8],
        condition_embedding_dims=[5, 4],
    )

    cond = condition()
    cond_mask = torch.ones_like(cond)

    cond_mu, _ = model._condition(condition=(cond, cond_mask))

    assert cond_mu.shape == (2, 4)


def test_cvae_condition_with_added_features():
    model = build_cvae(
        encoder_hidden_dims=[8],
        condition_embedding_dims=[7, 4],
        added_features_dim=1,
    )

    cond_mu, _ = model._condition(
        condition=condition(),
        added_features=torch.randn(2, 1),
    )

    assert cond_mu.shape == (2, 4)


def test_cvae_condition_dependant_latent_outputs_cond_mu_log_var():
    model = build_cvae(
        encoder_hidden_dims=[8],
        latent_size=4,
        condition_embedding_dims=[5, 4],
        condition_dependant_latent=True,
    )

    cond_mu, cond_log_var = model._condition(condition=condition())

    assert cond_mu.shape == (2, 4)
    assert cond_log_var.shape == (2, 4)


def test_cvae_sample_shape():
    model = build_cvae()

    mu = torch.zeros(2, 4)
    log_var = torch.zeros(2, 4)

    samples = model._sample(mu, log_var, sample_size=3)

    assert samples.shape == (3, 2, 4)


def test_cvae_get_normal_std():
    model = build_cvae()

    ref = torch.zeros(2, 4)
    dist = model._get_normal(ref, std=2)

    assert dist.sample().shape == ref.shape


def test_cvae_generate_plain():
    model = build_cvae()

    latent = torch.randn(3, 2, 4)
    out = model._generate(latent)

    assert out.shape == (3, 2, 6)


def test_cvae_generate_with_added_features():
    model = build_cvae(
        encoder_hidden_dims=[8],
        added_features_dim=3,
    )

    latent = torch.randn(3, 2, 4)
    out = model._generate(
        latent,
        added_features=added_features(),
    )

    assert out.shape == (3, 2, 6)


def test_cvae_generate_with_condition():
    model = build_cvae(
        encoder_hidden_dims=[8],
        condition_embedding_dims=[5, 4],
    )

    cond_mu, _ = model._condition(condition=condition())

    latent = torch.randn(3, 2, 4)
    out = model._generate(latent, condition=cond_mu)

    assert out.shape == (3, 2, 6)


def test_cvae_generate_with_condition_but_decoder_flag_false():
    model = build_cvae(
        encoder_hidden_dims=[8],
        condition_embedding_dims=[5, 4],
        condition_dependant_latent=True,
        condemb_to_decoder=False,
    )

    cond_mu, _ = model._condition(condition=condition())

    latent = torch.randn(3, 2, 4)
    out = model._generate(latent, condition=cond_mu)

    assert out.shape == (3, 2, 6)


def test_cvae_forward_basic():
    model = build_cvae()

    out = model(x=x())

    assert isinstance(out, cVAEOutput)
    assert out.output.shape == (1, 2, 1, 6)
    assert out.mu.shape == (2, 4)
    assert out.log_var.shape == (2, 4)


def test_cvae_forward_sample_size():
    model = build_cvae()

    out = model(x=x(), sample_size=3)

    assert out.output.shape == (3, 2, 1, 6)


def test_cvae_forward_with_tuple_x_and_min_variance():
    model = build_cvae()

    data = x()
    m = mask_like(data)

    out = model(
        x=(data, m),
        sample_size=2,
        min_posterior_variance=torch.tensor(-0.1),
    )

    assert out.output.shape == (2, 2, 1, 6)


def test_cvae_forward_with_condition_and_added_features():
    model = build_cvae(
        encoder_hidden_dims=[8],
        condition_embedding_dims=[7, 4],
        added_features_dim=1,
    )

    out = model(
        x=x(),
        condition=condition(),
        added_features=torch.randn(2, 1),
        sample_size=2,
    )

    assert out.output.shape == (2, 2, 1, 6)
    assert out.cond_mu.shape == (2, 4)


def test_cvae_forward_condition_dependant_latent():
    model = build_cvae(
        encoder_hidden_dims=[8],
        latent_size=4,
        condition_embedding_dims=[5, 4],
        condition_dependant_latent=True,
    )

    out = model(
        x=x(),
        condition=condition(),
        sample_size=2,
    )

    assert out.cond_mu.shape == (2, 4)
    assert out.cond_log_var.shape == (2, 4)


def test_cvae_predict_basic():
    model = build_cvae()

    out = model.predict(condition=condition(), sample_size=2)

    assert isinstance(out, cVAEOutput)
    assert out.output.shape == (2, 2, 1, 6)


def test_cvae_predict_condition_as_tuple():
    model = build_cvae(
        encoder_hidden_dims=[8],
        condition_embedding_dims=[5, 4],
    )

    cond = condition()
    cond_mask = torch.ones_like(cond)

    out = model.predict(condition=(cond, cond_mask), sample_size=2)

    assert out.output.shape == (2, 2, 1, 6)


def test_cvae_predict_condition_dependant_latent():
    model = build_cvae(
        encoder_hidden_dims=[8],
        latent_size=4,
        condition_embedding_dims=[5, 4],
        condition_dependant_latent=True,
    )

    out = model.predict(condition=condition(), sample_size=2)

    assert out.output.shape == (2, 2, 1, 6)


def test_cvae_predict_with_prior_flow_no_condition():
    model = build_cvae()

    flow = DummyPriorFlow(condition_size=None)

    out = model.predict(
        condition=condition(),
        prior_flow=flow,
        sample_size=2,
    )

    assert flow.called is True
    assert flow.last_condition is None
    assert out.output.shape == (2, 2, 1, 6)


def test_cvae_predict_with_prior_flow_conditioned():
    model = build_cvae(
        encoder_hidden_dims=[8],
        condition_embedding_dims=[5, 4],
    )

    flow = DummyPriorFlow(condition_size=4)

    out = model.predict(
        condition=condition(),
        prior_flow=flow,
        sample_size=2,
    )

    assert flow.called is True
    assert flow.last_condition is not None
    assert out.output.shape == (2, 2, 1, 6)


def test_autoencoder_config_build_returns_model():
    cfg = AutoencoderConfig(encoder_hidden_dims=[4])

    model = cfg.build(
        input_shape=np.array([6]),
        output_shape=np.array([6]),
    )

    assert isinstance(model, Autoencoder)


def test_autoencoder_decoder_hidden_default_empty_for_single_encoder_dim():
    cfg = AutoencoderConfig(encoder_hidden_dims=[4])

    model = cfg.build(
        input_shape=np.array([6]),
        output_shape=np.array([6]),
    )

    assert model.decoder_hidden_dims == []


def test_autoencoder_decoder_hidden_default_reverse_branch():
    cfg = AutoencoderConfig(encoder_hidden_dims=[16, 8, 4])

    model = cfg.build(
        input_shape=np.array([6]),
        output_shape=np.array([6]),
    )

    assert model.decoder_hidden_dims == [8, 16]


def test_autoencoder_explicit_decoder_hidden_dims():
    cfg = AutoencoderConfig(
        encoder_hidden_dims=[8, 4],
        decoder_hidden_dims=[5],
    )

    model = cfg.build(
        input_shape=np.array([6]),
        output_shape=np.array([6]),
    )

    assert model.config.decoder_hidden_dims == [5]


@pytest.mark.parametrize("append_mode", [1, 2, 3])
def test_autoencoder_build_append_modes(append_mode):
    model = build_autoencoder(
        encoder_hidden_dims=[4],
        append_mode=append_mode,
        added_features_dim=3,
    )

    assert hasattr(model, "encoder")
    assert hasattr(model, "decoder")
    assert model.append_mode == append_mode


def test_autoencoder_build_append_mode_other():
    model = build_autoencoder(
        encoder_hidden_dims=[4],
        append_mode=0,
        added_features_dim=3,
    )

    assert model.append_mode == 0


def test_autoencoder_build_no_added_features():
    model = build_autoencoder()

    assert model.added_features_dim == 0


def test_autoencoder_build_wrong_output_dims_raises():
    cfg = AutoencoderConfig(encoder_hidden_dims=[4])

    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        cfg.build(
            input_shape=np.array([6]),
            output_shape=np.array([2, 3]),
        )


def test_autoencoder_build_dropout_and_batchnorm():
    model = build_autoencoder(
        encoder_hidden_dims=[8, 4],
        dropout_rate=0.2,
        batch_normalization=True,
    )

    assert any(isinstance(layer, torch.nn.Dropout) for layer in model.encoder)
    assert any(isinstance(layer, torch.nn.BatchNorm1d) for layer in model.encoder)


def test_autoencoder_build_checkpoint_input_shape_mismatch():
    cfg = AutoencoderConfig(encoder_hidden_dims=[4])
    cfg._add_checkpoint_config(
        DummyCheckpointConfig(
            checkpoint_input_shape=np.array([5]),
            checkpoint_output_shape=np.array([6]),
        )
    )

    with pytest.raises(RuntimeError, match="input shape"):
        cfg.build(
            input_shape=np.array([6]),
            output_shape=np.array([6]),
        )


def test_autoencoder_build_checkpoint_output_shape_mismatch():
    cfg = AutoencoderConfig(encoder_hidden_dims=[4])
    cfg._add_checkpoint_config(
        DummyCheckpointConfig(
            checkpoint_input_shape=np.array([6]),
            checkpoint_output_shape=np.array([5]),
        )
    )

    with pytest.raises(RuntimeError, match="output shape"):
        cfg.build(
            input_shape=np.array([6]),
            output_shape=np.array([6]),
        )


def test_autoencoder_build_checkpoint_success_calls_load(monkeypatch):
    cfg = AutoencoderConfig(encoder_hidden_dims=[4])
    cfg._add_checkpoint_config(
        DummyCheckpointConfig(
            checkpoint_input_shape=np.array([6]),
            checkpoint_output_shape=np.array([6]),
        )
    )

    called = {"load": False}

    def fake_load(self, checkpoint_config):
        called["load"] = True

    monkeypatch.setattr(Autoencoder, "_load_state_dict", fake_load)

    model = cfg.build(
        input_shape=np.array([6]),
        output_shape=np.array([6]),
    )

    assert isinstance(model, Autoencoder)
    assert called["load"] is True


def test_autoencoder_forward_plain():
    model = build_autoencoder()

    out = model(x())

    assert isinstance(out, deterministicOutput)
    assert out.output.shape == (2, 1, 6)


def test_autoencoder_forward_tuple_mask_append_mode_1():
    model = build_autoencoder(
        encoder_hidden_dims=[4],
        append_mode=1,
        added_features_dim=3,
    )

    data = x()
    m = mask_like(data)
    feats = added_features()

    out = model((data, m), added_features=feats)

    assert out.output.shape == (2, 1, 6)


def test_autoencoder_forward_tuple_mask_append_mode_2():
    model = build_autoencoder(
        encoder_hidden_dims=[4],
        append_mode=2,
        added_features_dim=3,
    )

    data = x()
    m = mask_like(data)
    feats = added_features()

    out = model((data, m), added_features=feats)

    assert out.output.shape == (2, 1, 6)


def test_autoencoder_forward_tuple_mask_append_mode_3():
    model = build_autoencoder(
        encoder_hidden_dims=[4],
        append_mode=3,
        added_features_dim=3,
    )

    data = x()
    m = mask_like(data)
    feats = added_features()

    out = model((data, m), added_features=feats)

    assert out.output.shape == (2, 1, 6)


def test_autoencoder_forward_list_mask_append_mode_1():
    model = build_autoencoder(
        encoder_hidden_dims=[4],
        append_mode=1,
        added_features_dim=3,
    )

    data = x()
    m = mask_like(data)
    feats = added_features()

    out = model([data, m], added_features=feats)

    assert out.output.shape == (2, 1, 6)
