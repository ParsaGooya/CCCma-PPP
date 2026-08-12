from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from cccma_ppp.architectures.mlp.cvae import (
    cVAE_MLP,
    cVAE_MLPConfig,
)


def make_config(**overrides):
    defaults = {
        "encoder_hidden_dims": [8, 4],
        "latent_size": 3,
        "condition_embedding_dims": [6, 4],
        "condition_embedding_size": 3,
        "decoder_hidden_dims": [4, 8],
        "condition_dependant_latent": False,
        "condemb_to_decoder": True,
        "batch_normalization": False,
        "dropout_rate": None,
        "init_method": "trunc_normal",
        "activation": "relu",
    }
    defaults.update(overrides)

    config = cVAE_MLPConfig(**defaults)

    config.condition_dependant_flow = False

    if not hasattr(config, "checkpoint_config"):
        config.checkpoint_config = None

    return config


def make_model(
    *,
    input_shape=(2, 3),
    output_shape=(1, 3),
    added_features_dim=None,
    **config_overrides,
):
    config = make_config(**config_overrides)

    return cVAE_MLP(
        config=config,
        input_shape=input_shape,
        output_shape=output_shape,
        added_features_dim=added_features_dim,
    )


class IdentityWithArguments(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_input = None

    def forward(self, value):
        self.last_input = value
        return value


class FixedOutputModule(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.last_input = None

    def forward(self, value):
        self.last_input = value
        return torch.zeros(
            (*value.shape[:-1], self.output_size),
            dtype=value.dtype,
            device=value.device,
        )


class TestCVaEMLPConfig:
    @pytest.mark.pruned
    def test_basic_initialization(self):
        config = make_config()

        assert config.encoder_hidden_dims == [8, 4]
        assert config.latent_size == 3
        assert config.condition_embedding_dims == [6, 4]
        assert config.condition_embedding_size == 3
        assert config.decoder_hidden_dims == [4, 8]
        assert config.condition_dependant_latent is False
        assert config.condemb_to_decoder is True
        assert config.batch_normalization is False
        assert config.dropout_rate is None
        assert config.activation == "relu"

    @pytest.mark.pruned
    def test_expected_input_and_output_dimensions(self):
        config = make_config()

        assert config.NUM_INPUT_DIMS == 2
        assert config.NUM_OUTPUT_DIMS == 2

    @pytest.mark.pruned
    def test_generator_is_none(self):
        assert cVAE_MLPConfig.GENERATOR is None

    @pytest.mark.pruned
    def test_expects_mask_is_false(self):
        config = make_config()

        assert config.EXPECTS_MASK is False

    @pytest.mark.pruned
    def test_decoder_defaults_from_encoder(self):
        config = make_config(
            encoder_hidden_dims=[16, 8, 4],
            decoder_hidden_dims=None,
        )

        assert config.decoder_hidden_dims == [8, 16]

    def test_empty_encoder_creates_empty_decoder(self):
        config = make_config(
            encoder_hidden_dims=[],
            decoder_hidden_dims=None,
            condition_embedding_dims=[],
        )

        assert config.decoder_hidden_dims == []

    @pytest.mark.pruned
    def test_explicit_decoder_is_preserved(self):
        config = make_config(
            decoder_hidden_dims=[7, 9],
        )

        assert config.decoder_hidden_dims == [7, 9]

    def test_condition_embedding_dims_default_to_encoder_dims(self):
        config = make_config(
            encoder_hidden_dims=[12, 6],
            condition_embedding_dims=None,
        )

        assert config.condition_embedding_dims == [12, 6]

    def test_condition_embedding_size_defaults_to_latent_size(self):
        config = make_config(
            latent_size=7,
            condition_embedding_size=None,
        )

        assert config.condition_embedding_size == 7

    def test_independent_latent_requires_condition_in_decoder(self):
        with pytest.raises(
            ValueError,
            match="condition embedding has to be passed to decoder",
        ):
            make_config(
                condition_dependant_latent=False,
                condemb_to_decoder=False,
            )

    @pytest.mark.pruned
    def test_dependent_latent_does_not_require_condition_in_decoder(self):
        config = make_config(
            condition_dependant_latent=True,
            condemb_to_decoder=False,
        )

        assert config.condemb_to_decoder is False

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "dropout_rate",
        [
            -0.1,
            1.1,
        ],
    )
    def test_invalid_dropout_is_rejected(self, dropout_rate):
        with pytest.raises((ValueError, AssertionError)):
            make_config(
                dropout_rate=dropout_rate,
            )

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "dropout_rate",
        [
            None,
            0.0,
            0.25,
            1.0,
        ],
    )
    def test_valid_dropout_is_accepted(self, dropout_rate):
        config = make_config(
            dropout_rate=dropout_rate,
        )

        assert config.dropout_rate == dropout_rate

    @pytest.mark.pruned
    def test_build_returns_model(self):
        config = make_config()

        model = config.build(
            input_shape=np.array([2, 3]),
            output_shape=np.array([1, 3]),
            added_features_dim=2,
        )

        assert isinstance(model, cVAE_MLP)
        assert model.added_features_dim == 2


class TestCVaEMLPInitialization:
    def test_defaults_output_shape_to_input_shape(self):
        model = cVAE_MLP(
            config=make_config(),
            input_shape=np.array([2, 3]),
        )

        assert model.input_shape == 6
        assert model.output_shape == 6

    @pytest.mark.parametrize(
        "output_shape",
        [
            (6,),
            (1, 2, 3),
        ],
    )
    def test_rejects_invalid_output_rank(self, output_shape):
        with pytest.raises(
            RuntimeError,
            match="MLP models should create 2D outputs",
        ):
            cVAE_MLP(
                config=make_config(),
                input_shape=(2, 3),
                output_shape=output_shape,
            )

    @pytest.mark.pruned
    def test_converts_added_features_none_to_zero(self):
        model = make_model(
            added_features_dim=None,
        )

        assert model.added_features_dim == 0

    @pytest.mark.pruned
    def test_preserves_added_features_dimension(self):
        model = make_model(
            added_features_dim=4,
        )

        assert model.added_features_dim == 4

    @pytest.mark.pruned
    def test_flattened_input_and_output_sizes(self):
        model = make_model(
            input_shape=(2, 5),
            output_shape=(3, 4),
        )

        assert model.input_shape == 10
        assert model.output_shape == 12

    @pytest.mark.pruned
    def test_condition_is_added_to_decoder_when_enabled(self):
        model = make_model(
            condition_embedding_size=5,
            condemb_to_decoder=True,
        )

        assert model.add_condition_size == 5

    @pytest.mark.pruned
    def test_condition_is_not_initially_added_when_disabled(self):
        config = make_config(
            condition_dependant_latent=True,
            condemb_to_decoder=False,
        )

        with patch(
            "cccma_ppp.architectures.mlp.cvae.build_mlp",
            return_value=nn.Sequential(nn.Identity()),
        ):
            model = cVAE_MLP(
                config=config,
                input_shape=(2, 3),
                output_shape=(1, 3),
            )

        assert model.condemb_to_decoder is False

    @pytest.mark.pruned
    def test_builds_condition_distribution_for_dependent_latent(self):
        model = make_model(
            condition_dependant_latent=True,
        )

        assert isinstance(model.condition_mu, nn.Linear)
        assert isinstance(model.condition_log_var, nn.Linear)

    @pytest.mark.pruned
    def test_independent_latent_appends_embedding_projection(self):
        model = make_model(
            condition_dependant_latent=False,
        )

        assert isinstance(model.embedding[-1], nn.Linear)

    @pytest.mark.pruned
    def test_condition_flow_skips_condition_distribution(self):
        config = make_config(
            condition_dependant_latent=True,
        )
        config.condition_dependant_flow = True

        model = cVAE_MLP(
            config=config,
            input_shape=(2, 3),
            output_shape=(1, 3),
        )

        assert not hasattr(model, "condition_mu")
        assert not hasattr(model, "condition_log_var")
        assert isinstance(model.embedding[-1], nn.Linear)

    @pytest.mark.pruned
    def test_weights_are_initialized_without_checkpoint(self):
        config = make_config(
            init_method="trunc_normal",
        )
        config.checkpoint_config = None

        with (
            patch.object(
                cVAE_MLP,
                "_initialize_weights",
            ) as initialize,
            patch.object(
                cVAE_MLP,
                "_load_state_dict",
            ) as load_state,
        ):
            cVAE_MLP(
                config=config,
                input_shape=(2, 3),
                output_shape=(1, 3),
            )

        initialize.assert_called_once_with("trunc_normal")
        load_state.assert_not_called()

    @pytest.mark.pruned
    def test_validates_checkpoint_compatibility(self):
        config = make_config()

        with patch.object(
            cVAE_MLP,
            "_validate_checkpoint_compatibility",
        ) as validate:
            cVAE_MLP(
                config=config,
                input_shape=(2, 3),
                output_shape=(1, 3),
            )

        validate.assert_called_once()

        call = validate.call_args.kwargs
        assert call["input_shape"] == (2, 3)
        assert call["output_shape"] == (1, 3)


class TestRecognition:
    def make_stubbed_model(self):
        model = make_model()

        model.encoder = IdentityWithArguments()
        model.mu = nn.Identity()
        model.log_var = nn.Identity()

        return model

    def test_applies_target_mask(self):
        model = self.make_stubbed_model()

        target = torch.tensor(
            [
                [
                    1.0,
                    2.0,
                    3.0,
                ]
            ]
        )
        mask = torch.tensor(
            [
                [
                    1.0,
                    0.0,
                    1.0,
                ]
            ]
        )

        mu, log_var = model._recognition(
            x=target,
            x_mask=mask,
        )

        expected = torch.tensor(
            [
                [
                    1.0,
                    0.0,
                    3.0,
                ]
            ]
        )

        torch.testing.assert_close(mu, expected)
        torch.testing.assert_close(log_var, expected)

    @pytest.mark.pruned
    def test_flattens_target(self):
        model = self.make_stubbed_model()

        target = torch.arange(
            12,
            dtype=torch.float32,
        ).reshape(2, 2, 3)

        model._recognition(
            x=target,
            x_mask=None,
        )

        assert model.encoder.last_input.shape == (2, 6)

    @pytest.mark.pruned
    def test_concatenates_condition(self):
        model = self.make_stubbed_model()

        target = torch.ones(2, 1, 3)
        condition = torch.full(
            (2, 2),
            4.0,
        )

        model._recognition(
            x=target,
            x_mask=None,
            condition=condition,
        )

        assert model.encoder.last_input.shape == (2, 5)
        torch.testing.assert_close(
            model.encoder.last_input[:, -2:],
            condition,
        )

    def test_concatenates_added_features(self):
        model = self.make_stubbed_model()

        target = torch.ones(2, 1, 3)
        features = torch.tensor(
            [
                [
                    5.0,
                    6.0,
                ],
                [
                    7.0,
                    8.0,
                ],
            ]
        )

        model._recognition(
            x=target,
            x_mask=None,
            added_features=features,
        )

        assert model.encoder.last_input.shape == (2, 5)
        torch.testing.assert_close(
            model.encoder.last_input[:, -2:],
            features,
        )

    @pytest.mark.pruned
    def test_condition_precedes_added_features(self):
        model = self.make_stubbed_model()

        target = torch.ones(1, 1, 2)
        condition = torch.tensor(
            [
                [
                    3.0,
                    4.0,
                ]
            ]
        )
        features = torch.tensor(
            [
                [
                    5.0,
                    6.0,
                ]
            ]
        )

        model._recognition(
            x=target,
            x_mask=None,
            condition=condition,
            added_features=features,
        )

        expected = torch.tensor(
            [
                [
                    1.0,
                    1.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                ]
            ]
        )

        torch.testing.assert_close(
            model.encoder.last_input,
            expected,
        )


class TestCondition:
    def test_applies_condition_mask(self):
        model = make_model()
        model.embedding = nn.Identity()

        condition = torch.tensor(
            [
                [
                    1.0,
                    2.0,
                    3.0,
                ]
            ]
        )
        mask = torch.tensor(
            [
                [
                    1.0,
                    0.0,
                    1.0,
                ]
            ]
        )

        cond_mu, cond_log_var = model._condition(
            condition=condition,
            condition_mask=mask,
        )

        torch.testing.assert_close(
            cond_mu,
            torch.tensor(
                [
                    [
                        1.0,
                        0.0,
                        3.0,
                    ]
                ]
            ),
        )
        assert cond_log_var is None

    @pytest.mark.pruned
    def test_flattens_condition(self):
        model = make_model()
        model.embedding = nn.Identity()

        condition = torch.arange(
            12,
            dtype=torch.float32,
        ).reshape(2, 2, 3)

        cond_mu, _ = model._condition(
            condition=condition,
        )

        assert cond_mu.shape == (2, 6)

    def test_concatenates_added_features(self):
        model = make_model()
        model.embedding = IdentityWithArguments()

        condition = torch.ones(2, 1, 3)
        features = torch.tensor(
            [
                [
                    4.0,
                    5.0,
                ],
                [
                    6.0,
                    7.0,
                ],
            ]
        )

        model._condition(
            condition=condition,
            added_features=features,
        )

        assert model.embedding.last_input.shape == (2, 5)
        torch.testing.assert_close(
            model.embedding.last_input[:, -2:],
            features,
        )

    def test_dependent_latent_returns_distribution(self):
        model = make_model(
            condition_dependant_latent=True,
        )

        model.embedding = nn.Identity()
        model.condition_mu = nn.Linear(
            3,
            2,
            bias=False,
        )
        model.condition_log_var = nn.Linear(
            3,
            2,
            bias=False,
        )

        with torch.no_grad():
            model.condition_mu.weight.fill_(1.0)
            model.condition_log_var.weight.fill_(2.0)

        condition = torch.ones(2, 1, 3)

        cond_mu, cond_log_var = model._condition(
            condition=condition,
        )

        assert cond_mu.shape == (2, 2)
        assert cond_log_var.shape == (2, 2)
        torch.testing.assert_close(
            cond_mu,
            torch.full(
                (2, 2),
                3.0,
            ),
        )
        torch.testing.assert_close(
            cond_log_var,
            torch.full(
                (2, 2),
                6.0,
            ),
        )


class TestGenerate:
    def make_stubbed_model(
        self,
        *,
        condemb_to_decoder=True,
    ):
        model = make_model(
            condemb_to_decoder=condemb_to_decoder,
            condition_dependant_latent=not condemb_to_decoder,
        )
        model.decoder = IdentityWithArguments()

        return model

    @pytest.mark.pruned
    def test_preserves_sample_and_batch_dimensions(self):
        model = self.make_stubbed_model(
            condemb_to_decoder=False,
        )

        latent = torch.ones(3, 2, 4)

        result = model._generate(
            latent_samples=latent,
        )

        assert result.shape == (3, 2, 4)

    def test_concatenates_added_features(self):
        model = self.make_stubbed_model(
            condemb_to_decoder=False,
        )

        latent = torch.ones(3, 2, 4)
        features = torch.tensor(
            [
                [
                    5.0,
                    6.0,
                ],
                [
                    7.0,
                    8.0,
                ],
            ]
        )

        result = model._generate(
            latent_samples=latent,
            added_features=features,
        )

        assert result.shape == (3, 2, 6)

        expected_features = features.unsqueeze(0).expand(
            3,
            2,
            2,
        )
        torch.testing.assert_close(
            result[..., -2:],
            expected_features,
        )

    @pytest.mark.pruned
    def test_concatenates_condition_when_enabled(self):
        model = self.make_stubbed_model(
            condemb_to_decoder=True,
        )

        latent = torch.ones(3, 2, 4)
        condition = torch.tensor(
            [
                [
                    5.0,
                    6.0,
                ],
                [
                    7.0,
                    8.0,
                ],
            ]
        )

        result = model._generate(
            latent_samples=latent,
            condition=condition,
        )

        assert result.shape == (3, 2, 6)

        expected_condition = condition.unsqueeze(0).expand(
            3,
            2,
            2,
        )
        torch.testing.assert_close(
            result[..., -2:],
            expected_condition,
        )

    @pytest.mark.pruned
    def test_does_not_concatenate_condition_when_disabled(self):
        model = self.make_stubbed_model(
            condemb_to_decoder=False,
        )

        latent = torch.ones(3, 2, 4)
        condition = torch.ones(2, 2)

        result = model._generate(
            latent_samples=latent,
            condition=condition,
        )

        assert result.shape == (3, 2, 4)

    @pytest.mark.pruned
    def test_ignores_none_condition(self):
        model = self.make_stubbed_model(
            condemb_to_decoder=True,
        )

        latent = torch.ones(3, 2, 4)

        result = model._generate(
            latent_samples=latent,
            condition=None,
        )

        assert result.shape == (3, 2, 4)


class TestForward:
    def make_forward_model(self):
        model = make_model()

        model._condition = Mock(
            return_value=(
                torch.ones(2, 3),
                None,
            )
        )
        model._recognition = Mock(
            return_value=(
                torch.zeros(2, 3),
                torch.zeros(2, 3),
            )
        )
        model._sample = Mock(return_value=torch.zeros(4, 2, 3))
        model._generate = Mock(return_value=torch.zeros(4, 2, 3))

        return model

    @pytest.mark.pruned
    def test_forward_uses_request_values(self):
        model = self.make_forward_model()

        target = torch.ones(2, 1, 3)
        target_mask = torch.ones_like(target)
        condition = torch.ones(2, 1, 3)
        condition_mask = torch.ones_like(condition)
        features = torch.ones(2, 2)

        request = SimpleNamespace(
            target=target,
            target_mask=target_mask,
            condition=condition,
            condition_mask=condition_mask,
            added_features=features,
            latent_sample_size=4,
            posterior_variance_limits=None,
        )

        result = model.forward(request)

        model._condition.assert_called_once_with(
            condition=condition,
            condition_mask=condition_mask,
            added_features=features,
        )
        model._recognition.assert_called_once()
        model._sample.assert_called_once()
        model._generate.assert_called_once()

        assert result.output.shape == (4, 2, 1, 3)

    @pytest.mark.pruned
    def test_forward_clamps_posterior_variance(self):
        model = self.make_forward_model()

        model._recognition.return_value = (
            torch.zeros(2, 3),
            torch.tensor(
                [
                    [
                        -10.0,
                        0.0,
                        10.0,
                    ],
                    [
                        -5.0,
                        1.0,
                        5.0,
                    ],
                ]
            ),
        )

        request = SimpleNamespace(
            target=torch.ones(2, 1, 3),
            target_mask=None,
            condition=torch.ones(2, 1, 3),
            condition_mask=None,
            added_features=None,
            latent_sample_size=4,
            posterior_variance_limits=(
                torch.tensor(-2.0),
                torch.tensor(2.0),
            ),
        )

        result = model.forward(request)

        assert torch.all(result.log_var >= -2.0)
        assert torch.all(result.log_var <= 2.0)

    @pytest.mark.pruned
    def test_forward_without_limits_preserves_log_variance(self):
        model = self.make_forward_model()

        expected = torch.tensor(
            [
                [
                    -10.0,
                    0.0,
                    10.0,
                ],
                [
                    -5.0,
                    1.0,
                    5.0,
                ],
            ]
        )
        model._recognition.return_value = (
            torch.zeros(2, 3),
            expected,
        )

        request = SimpleNamespace(
            target=torch.ones(2, 1, 3),
            target_mask=None,
            condition=torch.ones(2, 1, 3),
            condition_mask=None,
            added_features=None,
            latent_sample_size=4,
            posterior_variance_limits=None,
        )

        result = model.forward(request)

        torch.testing.assert_close(
            result.log_var,
            expected,
        )


class TestPredict:
    def make_predict_model(self):
        model = make_model()

        model._sample_prior = Mock(
            return_value=(
                torch.zeros(4, 2, 3),
                torch.ones(2, 3),
                torch.full(
                    (2, 3),
                    2.0,
                ),
            )
        )
        model._generate = Mock(return_value=torch.zeros(4, 2, 3))

        return model

    @pytest.mark.pruned
    def test_predict_samples_prior(self):
        model = self.make_predict_model()

        condition = torch.ones(2, 1, 3)
        features = torch.ones(2, 2)

        request = SimpleNamespace(
            condition=condition,
            condition_mask=None,
            added_features=features,
            latent_sample_size=4,
            latent_samples=None,
        )

        result = model.predict(request)

        model._sample_prior.assert_called_once_with(request)
        model._generate.assert_called_once()

        assert result.output.shape == (4, 2, 1, 3)
        assert result.mu is None
        assert result.log_var is None
        assert result.samples is None
        assert result.cond_mu.shape == (2, 3)
        assert result.cond_log_var.shape == (2, 3)

    @pytest.mark.pruned
    def test_predict_forwards_condition_and_features(self):
        model = self.make_predict_model()

        condition = torch.ones(2, 1, 3)
        features = torch.ones(2, 2)

        request = SimpleNamespace(
            condition=condition,
            condition_mask=None,
            added_features=features,
            latent_sample_size=4,
            latent_samples=None,
        )

        model.predict(request)

        _, kwargs = model._generate.call_args

        torch.testing.assert_close(
            kwargs["condition"],
            torch.ones(2, 3),
        )
        assert kwargs["added_features"] is features


class TestCVaEMLPIntegration:
    def test_forward_supports_backward(self):
        torch.manual_seed(0)

        model = make_model(
            input_shape=(1, 3),
            output_shape=(1, 3),
            encoder_hidden_dims=[6],
            decoder_hidden_dims=[6],
            condition_embedding_dims=[6],
            condition_embedding_size=3,
            latent_size=3,
        )

        request = SimpleNamespace(
            target=torch.randn(
                2,
                1,
                3,
                requires_grad=True,
            ),
            target_mask=None,
            condition=torch.randn(
                2,
                1,
                3,
            ),
            condition_mask=None,
            added_features=None,
            latent_sample_size=2,
            posterior_variance_limits=None,
        )

        result = model.forward(request)
        loss = result.output.mean()
        loss.backward()

        assert any(
            parameter.grad is not None
            for parameter in model.parameters()
            if parameter.requires_grad
        )
