from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from cccma_ppp.models.layers.conv import (
    ConvBlockConfig,
    LatentVector,
    PartialConvBlockConfig,
    TensorMask,
)
from cccma_ppp.models.unet_models.cvae import (
    Generation,
    Recognition,
    cVAEUNet,
    cVAEUNetConfig,
)


def make_config(**overrides):
    defaults = {
        "channels": [4, 8],
        "latent_size": 3,
        "condition_embedding_channels": [4, 8],
        "condition_embedding_size": 3,
        "latent_normalization": "layer",
        "condition_dependant_latent": False,
        "condemb_to_decoder": True,
        "deterministic_guess_config": None,
        "block_config": ConvBlockConfig(name="conv"),
        "upsampling_method": "bilinear",
        "upsampling_alignment_method": "padd",
        "transpose_kernel_sizes": 3,
        "add_skip_latent": False,
        "mask_pooling": "any",
        "mask_fraction_threshold": 0.5,
        "output_activation": "identity",
        "output_block_hidden_channels": 32,
        "init_method": "trunc_normal",
        "GENERATOR": None,
    }
    defaults.update(overrides)

    config = cVAEUNetConfig(**defaults)

    config.condition_dependant_flow = False

    if not hasattr(config, "checkpoint_config"):
        config.checkpoint_config = None

    return config


def make_model(
    *,
    input_shape=(2, 8, 8),
    output_shape=(1, 8, 8),
    added_features_dim=None,
    **config_overrides,
):
    return cVAEUNet(
        config=make_config(**config_overrides),
        input_shape=input_shape,
        output_shape=output_shape,
        added_features_dim=added_features_dim,
    )


class IdentityTensorMask(nn.Module):
    def forward(self, value):
        return value


class FixedRecognition(nn.Module):
    def __init__(
        self,
        mu,
        log_var,
    ):
        super().__init__()
        self.mu = mu
        self.log_var = log_var
        self.last_input = None

    def forward(self, value):
        self.last_input = value
        return LatentVector(
            mu=self.mu,
            log_var=self.log_var,
        )


class FixedGeneration(nn.Module):
    def __init__(self, output):
        super().__init__()
        self.output = output
        self.calls = []

    def forward(
        self,
        latent_samples,
        num_output_samples=0,
    ):
        self.calls.append(
            (
                latent_samples,
                num_output_samples,
            )
        )
        return self.output


class RecordingOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_input = None

    def forward(self, value):
        self.last_input = value
        return value


class TestCVAEUNetConfig:
    def test_basic_initialization(self):
        config = make_config()

        assert config.channels == [4, 8]
        assert config.latent_size == 3
        assert config.condition_embedding_channels == [4, 8]
        assert config.condition_embedding_size == 3
        assert config.NUM_INPUT_DIMS == 3
        assert config.NUM_OUTPUT_DIMS == 3

    def test_transpose_kernel_integer_is_expanded(self):
        config = make_config(
            channels=[
                4,
                8,
                16,
            ],
            condition_embedding_channels=[
                4,
                8,
                16,
            ],
            transpose_kernel_sizes=5,
        )

        assert config.transpose_kernel_sizes == [
            5,
            5,
        ]

    def test_transpose_kernel_list_is_preserved(self):
        config = make_config(
            channels=[
                4,
                8,
                16,
            ],
            condition_embedding_channels=[
                4,
                8,
                16,
            ],
            transpose_kernel_sizes=[
                3,
                5,
            ],
        )

        assert config.transpose_kernel_sizes == [
            3,
            5,
        ]

    def test_condition_channels_default_to_model_channels(self):
        config = make_config(
            channels=[
                4,
                8,
                16,
            ],
            condition_embedding_channels=None,
        )

        assert config.condition_embedding_channels == [
            4,
            8,
            16,
        ]

    def test_condition_embedding_size_defaults_to_latent_size(self):
        config = make_config(
            latent_size=7,
            condition_embedding_size=None,
        )

        assert config.condition_embedding_size == 7

    def test_without_deterministic_guess_disables_shared_output(self):
        config = make_config(
            deterministic_guess_config=None,
        )

        assert config.share_output_block is False

    def test_expects_mask_with_regular_convolution(self):
        config = make_config(
            block_config=ConvBlockConfig(name="conv"),
        )

        assert config.EXPECTS_MASK is False

    def test_expects_mask_with_partial_convolution(self):
        config = make_config(
            block_config=PartialConvBlockConfig(name="conv"),
        )

        assert config.EXPECTS_MASK is True

    def test_expects_mask_with_partial_conv_attribute(self):
        block_config = SimpleNamespace(
            use_partial_conv=True,
        )

        config = object.__new__(cVAEUNetConfig)
        config.block_config = block_config

        assert config.EXPECTS_MASK is True

    def test_build_returns_model(self):
        config = make_config()

        model = config.build(
            input_shape=np.array(
                [
                    2,
                    8,
                    8,
                ]
            ),
            output_shape=np.array(
                [
                    1,
                    8,
                    8,
                ]
            ),
            added_features_dim=2,
        )

        assert isinstance(
            model,
            cVAEUNet,
        )
        assert model.added_features_dim == 2

    def test_deterministic_guess_channel_mismatch_is_rejected(self):
        resolved = SimpleNamespace(
            channels=[
                5,
                8,
            ],
            GENERATOR=None,
            output_activation="identity",
            output_block_hidden_channels=32,
        )
        selector = SimpleNamespace(
            share_output_block=False,
            get_model_config=Mock(
                return_value=resolved,
            ),
        )

        with pytest.raises(
            ValueError,
            match="same number of channels",
        ):
            make_config(
                deterministic_guess_config=selector,
            )

    def test_deterministic_guess_generator_is_rejected(self):
        resolved = SimpleNamespace(
            channels=[
                4,
                8,
            ],
            GENERATOR=object(),
            output_activation="identity",
            output_block_hidden_channels=32,
        )
        selector = SimpleNamespace(
            share_output_block=False,
            get_model_config=Mock(
                return_value=resolved,
            ),
        )

        with pytest.raises(
            ValueError,
            match="cannot have GENERATOR on",
        ):
            make_config(
                deterministic_guess_config=selector,
            )

    @pytest.mark.parametrize(
        "activation,hidden_channels",
        [
            (
                "tanh",
                32,
            ),
            (
                "identity",
                16,
            ),
        ],
    )
    def test_shared_output_requires_matching_configuration(
        self,
        activation,
        hidden_channels,
    ):
        resolved = SimpleNamespace(
            channels=[
                4,
                8,
            ],
            GENERATOR=None,
            output_activation=activation,
            output_block_hidden_channels=hidden_channels,
        )
        selector = SimpleNamespace(
            share_output_block=True,
            get_model_config=Mock(
                return_value=resolved,
            ),
        )

        with pytest.raises(
            ValueError,
            match="share_output_block",
        ):
            make_config(
                deterministic_guess_config=selector,
            )

    def test_valid_shared_output_configuration(self):
        resolved = SimpleNamespace(
            channels=[
                4,
                8,
            ],
            GENERATOR=None,
            output_activation="identity",
            output_block_hidden_channels=32,
        )
        selector = SimpleNamespace(
            share_output_block=True,
            get_model_config=Mock(
                return_value=resolved,
            ),
        )

        config = make_config(
            deterministic_guess_config=selector,
        )

        assert config.share_output_block is True
        assert config.deterministic_guess_config is resolved


class TestCVAEUNetInitialization:
    @pytest.mark.parametrize(
        "input_shape",
        [
            (
                2,
                8,
            ),
            (
                2,
                8,
                8,
                1,
            ),
        ],
    )
    def test_rejects_invalid_input_rank(
        self,
        input_shape,
    ):
        with pytest.raises(
            RuntimeError,
            match="expects 3D input shapes",
        ):
            make_model(
                input_shape=input_shape,
            )

    @pytest.mark.parametrize(
        "output_shape",
        [
            (
                1,
                8,
            ),
            (
                1,
                8,
                8,
                1,
            ),
        ],
    )
    def test_rejects_invalid_output_rank(
        self,
        output_shape,
    ):
        with pytest.raises(
            RuntimeError,
            match="expects 3D output shapes",
        ):
            make_model(
                output_shape=output_shape,
            )

    @pytest.mark.parametrize(
        "input_shape",
        [
            (
                2,
                0,
                8,
            ),
            (
                2,
                8,
                0,
            ),
            (
                2,
                -1,
                8,
            ),
        ],
    )
    def test_rejects_nonpositive_spatial_dimensions(
        self,
        input_shape,
    ):
        with pytest.raises(
            ValueError,
            match="spatial dimensions must be positive",
        ):
            make_model(
                input_shape=input_shape,
                output_shape=(
                    1,
                    8,
                    8,
                ),
            )

    def test_rejects_depth_larger_than_input(self):
        with pytest.raises(
            ValueError,
            match="depth is too large",
        ):
            make_model(
                input_shape=(
                    2,
                    4,
                    4,
                ),
                output_shape=(
                    1,
                    4,
                    4,
                ),
                channels=[
                    4,
                    8,
                    16,
                    32,
                ],
                condition_embedding_channels=[
                    4,
                    8,
                    16,
                    32,
                ],
                transpose_kernel_sizes=[
                    3,
                    3,
                    3,
                ],
            )

    def test_defaults_output_shape_to_input(self):
        model = cVAEUNet(
            config=make_config(),
            input_shape=(
                2,
                8,
                8,
            ),
        )

        assert model.output_shape == (
            2,
            8,
            8,
        )

    def test_preserves_explicit_output_shape(self):
        model = make_model(
            output_shape=(
                3,
                8,
                8,
            ),
        )

        assert model.output_shape == (
            3,
            8,
            8,
        )

    def test_defaults_added_features_to_zero(self):
        model = make_model(
            added_features_dim=None,
        )

        assert model.added_features_dim == 0

    def test_preserves_added_features_dimension(self):
        model = make_model(
            added_features_dim=3,
        )

        assert model.added_features_dim == 3

    def test_recognition_input_channels(self):
        model = make_model(
            input_shape=(
                2,
                8,
                8,
            ),
            output_shape=(
                1,
                8,
                8,
            ),
            added_features_dim=3,
        )

        assert model.recognition.initial_mapping is not None

    def test_condition_requests_log_variance_for_dependent_latent(self):
        with patch(
            "cccma_ppp.models.unet_models.cvae.Recognition",
            autospec=True,
        ) as recognition:
            recognition.side_effect = [
                SimpleNamespace(
                    spatial_shapes=[
                        (
                            8,
                            8,
                        ),
                        (
                            4,
                            4,
                        ),
                    ]
                ),
                Mock(),
            ]

            with (
                patch(
                    "cccma_ppp.models.unet_models.cvae.Generation",
                ),
                patch(
                    "cccma_ppp.models.unet_models.cvae.UNetOutput",
                ),
                patch.object(
                    cVAEUNet,
                    "_initialize_weights",
                ),
            ):
                cVAEUNet(
                    config=make_config(
                        condition_dependant_latent=True,
                    ),
                    input_shape=(
                        2,
                        8,
                        8,
                    ),
                    output_shape=(
                        1,
                        8,
                        8,
                    ),
                )

        condition_call = recognition.call_args_list[1]
        assert condition_call.kwargs["get_log_var"] is True

    def test_condition_skips_log_variance_for_independent_latent(self):
        with patch(
            "cccma_ppp.models.unet_models.cvae.Recognition",
            autospec=True,
        ) as recognition:
            recognition.side_effect = [
                SimpleNamespace(
                    spatial_shapes=[
                        (
                            8,
                            8,
                        ),
                        (
                            4,
                            4,
                        ),
                    ]
                ),
                Mock(),
            ]

            with (
                patch(
                    "cccma_ppp.models.unet_models.cvae.Generation",
                ),
                patch(
                    "cccma_ppp.models.unet_models.cvae.UNetOutput",
                ),
                patch.object(
                    cVAEUNet,
                    "_initialize_weights",
                ),
            ):
                cVAEUNet(
                    config=make_config(
                        condition_dependant_latent=False,
                    ),
                    input_shape=(
                        2,
                        8,
                        8,
                    ),
                    output_shape=(
                        1,
                        8,
                        8,
                    ),
                )

        condition_call = recognition.call_args_list[1]
        assert condition_call.kwargs["get_log_var"] is False

    def test_condition_flow_skips_log_variance(self):
        config = make_config(
            condition_dependant_latent=True,
        )
        config.condition_dependant_flow = True

        with patch(
            "cccma_ppp.models.unet_models.cvae.Recognition",
            autospec=True,
        ) as recognition:
            recognition.side_effect = [
                SimpleNamespace(
                    spatial_shapes=[
                        (
                            8,
                            8,
                        ),
                        (
                            4,
                            4,
                        ),
                    ]
                ),
                Mock(),
            ]

            with (
                patch(
                    "cccma_ppp.models.unet_models.cvae.Generation",
                ),
                patch(
                    "cccma_ppp.models.unet_models.cvae.UNetOutput",
                ),
                patch.object(
                    cVAEUNet,
                    "_initialize_weights",
                ),
            ):
                cVAEUNet(
                    config=config,
                    input_shape=(
                        2,
                        8,
                        8,
                    ),
                    output_shape=(
                        1,
                        8,
                        8,
                    ),
                )

        condition_call = recognition.call_args_list[1]
        assert condition_call.kwargs["get_log_var"] is False

    def test_initializes_weights_without_checkpoint(self):
        config = make_config()
        config.checkpoint_config = None

        with patch.object(
            cVAEUNet,
            "_initialize_weights",
        ) as initialize:
            model = cVAEUNet(
                config=config,
                input_shape=(
                    2,
                    8,
                    8,
                ),
                output_shape=(
                    1,
                    8,
                    8,
                ),
            )

        initialize.assert_called_once_with(
            "trunc_normal",
            exclude=(model.deterministic_guess,),
        )

    def test_validates_checkpoint_compatibility(self):
        with patch.object(
            cVAEUNet,
            "_validate_checkpoint_compatibility",
        ) as validate:
            cVAEUNet(
                config=make_config(),
                input_shape=(
                    2,
                    8,
                    8,
                ),
                output_shape=(
                    1,
                    8,
                    8,
                ),
            )

        validate.assert_called_once_with(
            input_shape=(
                2,
                8,
                8,
            ),
            output_shape=(
                1,
                8,
                8,
            ),
        )


class TestPrepareInput:
    def test_returns_tensor_mask(self):
        model = make_model()
        x = torch.randn(
            2,
            1,
            8,
            8,
        )

        result = model._prepare_input(
            x=x,
            x_mask=None,
        )

        assert isinstance(
            result,
            TensorMask,
        )
        assert result.tensor is x
        assert result.mask is None

    def test_resizes_and_concatenates_condition(self):
        model = make_model()

        x = torch.ones(
            2,
            1,
            8,
            8,
        )
        condition = torch.zeros(
            2,
            2,
            4,
            4,
        )

        result = model._prepare_input(
            x=x,
            x_mask=None,
            condition=condition,
        )

        assert result.tensor.shape == (
            2,
            3,
            8,
            8,
        )

    def test_creates_condition_mask_when_missing(self):
        model = make_model()

        x = torch.ones(
            2,
            1,
            8,
            8,
        )
        condition = torch.ones(
            2,
            2,
            8,
            8,
        )

        result = model._prepare_input(
            x=x,
            x_mask=torch.ones_like(x),
            condition=condition,
            condition_mask=None,
        )

        assert torch.all(result.mask == 1)

    def test_resizes_and_concatenates_added_features(self):
        model = make_model()

        x = torch.ones(
            2,
            1,
            8,
            8,
        )
        features = torch.zeros(
            2,
            3,
            1,
            1,
        )

        result = model._prepare_input(
            x=x,
            x_mask=None,
            added_features=features,
        )

        assert result.tensor.shape == (
            2,
            4,
            8,
            8,
        )

    def test_adds_feature_mask_when_input_is_masked(self):
        model = make_model()

        x = torch.ones(
            2,
            1,
            8,
            8,
        )
        features = torch.ones(
            2,
            2,
            1,
            1,
        )

        result = model._prepare_input(
            x=x,
            x_mask=torch.ones_like(x),
            added_features=features,
        )

        assert result.mask.shape == (
            2,
            3,
            8,
            8,
        )
        assert torch.all(
            result.mask[
                :,
                1:,
            ]
            == 1
        )


class TestRecognitionAndCondition:
    def test_recognition_returns_mu_and_log_var(self):
        model = make_model()

        expected_mu = torch.ones(
            2,
            3,
        )
        expected_log_var = torch.zeros(
            2,
            3,
        )
        model.recognition = FixedRecognition(
            expected_mu,
            expected_log_var,
        )

        mu, log_var = model._recognition(
            x=torch.ones(
                2,
                1,
                8,
                8,
            ),
            x_mask=None,
            condition=torch.ones(
                2,
                2,
                8,
                8,
            ),
        )

        assert mu is expected_mu
        assert log_var is expected_log_var
        assert isinstance(
            model.recognition.last_input,
            TensorMask,
        )

    def test_condition_returns_embedding_statistics(self):
        model = make_model()

        expected_mu = torch.ones(
            2,
            3,
        )
        expected_log_var = torch.zeros(
            2,
            3,
        )
        model.condition = FixedRecognition(
            expected_mu,
            expected_log_var,
        )

        cond_mu, cond_log_var = model._condition(
            condition=torch.ones(
                2,
                2,
                8,
                8,
            )
        )

        assert cond_mu is expected_mu
        assert cond_log_var is expected_log_var


class TestGenerate:
    def test_condition_embedding_is_concatenated(self):
        model = make_model(
            condemb_to_decoder=True,
        )

        model.generation = FixedGeneration(
            torch.zeros(
                6,
                4,
                8,
                8,
            )
        )

        latent = torch.ones(
            3,
            2,
            5,
        )
        condition = torch.ones(
            2,
            4,
        )

        result = model._generate(
            latent_samples=latent,
            condition_embedding=condition,
        )

        passed_latent = model.generation.calls[0][0]

        assert passed_latent.shape == (
            6,
            9,
        )
        assert result.shape == (
            3,
            2,
            4,
            8,
            8,
        )

    def test_condition_embedding_is_ignored_when_disabled(self):
        model = make_model(
            condition_dependant_latent=True,
            condemb_to_decoder=False,
        )

        model.generation = FixedGeneration(
            torch.zeros(
                6,
                4,
                8,
                8,
            )
        )

        model._generate(
            latent_samples=torch.ones(
                3,
                2,
                5,
            ),
            condition_embedding=torch.ones(
                2,
                4,
            ),
        )

        passed_latent = model.generation.calls[0][0]
        assert passed_latent.shape == (
            6,
            5,
        )


class TestDeterministicGuess:
    def test_returns_none_without_deterministic_model(self):
        model = make_model()
        model.deterministic_guess = None

        result = model._deterministic_guess(
            input=torch.ones(
                2,
                1,
                8,
                8,
            )
        )

        assert result is None

    def test_calls_deterministic_decoder(self):
        model = make_model()

        deterministic = SimpleNamespace(
            forward_decoder=Mock(
                return_value=torch.ones(
                    2,
                    1,
                    8,
                    8,
                )
            )
        )
        model.deterministic_guess = deterministic

        result = model._deterministic_guess(
            input=torch.ones(
                2,
                2,
                8,
                8,
            ),
            input_mask=torch.ones(
                2,
                1,
                8,
                8,
            ),
            added_features=torch.ones(
                2,
                1,
                8,
                8,
            ),
        )

        deterministic.forward_decoder.assert_called_once()
        assert result.shape == (
            2,
            1,
            8,
            8,
        )


class TestOutputBlock:
    def test_flattens_and_restores_sample_dimensions(self):
        model = make_model()
        output = RecordingOutput()
        model.output = output

        value = torch.ones(
            4,
            3,
            2,
            5,
            8,
            8,
        )

        result = model._output_block(
            input=value,
            sample_sizes=(
                4,
                3,
                2,
            ),
        )

        assert output.last_input.shape == (
            24,
            5,
            8,
            8,
        )
        assert result.shape == (
            4,
            3,
            2,
            5,
            8,
            8,
        )


class TestForward:
    def make_stubbed_model(self):
        model = make_model()

        model._condition = Mock(
            return_value=(
                torch.ones(
                    2,
                    3,
                ),
                torch.zeros(
                    2,
                    3,
                ),
            )
        )
        model._recognition = Mock(
            return_value=(
                torch.zeros(
                    2,
                    3,
                ),
                torch.zeros(
                    2,
                    3,
                ),
            )
        )
        model._sample = Mock(
            return_value=torch.zeros(
                4,
                2,
                3,
            )
        )
        model._generate = Mock(
            return_value=torch.zeros(
                4,
                2,
                4,
                8,
                8,
            )
        )
        model._deterministic_guess = Mock(
            return_value=None,
        )
        model._output_block = Mock(
            return_value=torch.zeros(
                4,
                2,
                1,
                8,
                8,
            )
        )

        return model

    def make_request(
        self,
        **overrides,
    ):
        values = {
            "target": torch.ones(
                2,
                1,
                8,
                8,
            ),
            "target_mask": None,
            "condition": torch.ones(
                2,
                2,
                8,
                8,
            ),
            "condition_mask": None,
            "added_features": None,
            "latent_sample_size": 4,
            "posterior_variance_limits": None,
            "output_sample_size": 0,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_forward_returns_cvae_output(self):
        model = self.make_stubbed_model()

        result = model.forward(self.make_request())

        assert result.output.shape == (
            4,
            2,
            1,
            8,
            8,
        )
        assert result.mu.shape == (
            2,
            3,
        )
        assert result.log_var.shape == (
            2,
            3,
        )
        assert result.samples.shape == (
            4,
            2,
            3,
        )

    def test_forward_clamps_posterior_variance(self):
        model = self.make_stubbed_model()

        model._recognition.return_value = (
            torch.zeros(
                2,
                3,
            ),
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

        request = self.make_request(
            posterior_variance_limits=(
                torch.tensor(-2.0),
                torch.tensor(2.0),
            )
        )

        result = model.forward(request)

        assert torch.all(result.log_var >= -2)
        assert torch.all(result.log_var <= 2)

    def test_forward_uses_generator_training_sample_count(self):
        model = self.make_stubbed_model()
        model.config.GENERATOR = SimpleNamespace(
            num_training_noise_samples=5,
        )
        model.train()

        model._generate.return_value = torch.zeros(
            5,
            4,
            2,
            4,
            8,
            8,
        )
        model._output_block.return_value = torch.zeros(
            5,
            4,
            2,
            1,
            8,
            8,
        )

        model.forward(
            self.make_request(
                output_sample_size=2,
            )
        )

        assert model._generate.call_args.kwargs["num_output_samples"] == 5
        model._output_block.assert_called_once_with(
            model._generate.return_value,
            (
                5,
                4,
                2,
            ),
        )

    def test_forward_adds_deterministic_guess(self):
        model = self.make_stubbed_model()

        generated = torch.ones(
            4,
            2,
            4,
            8,
            8,
        )
        deterministic = torch.full(
            (
                4,
                2,
                4,
                8,
                8,
            ),
            2.0,
        )

        model._generate.return_value = generated
        model._deterministic_guess.return_value = deterministic

        model.forward(self.make_request())

        passed = model._output_block.call_args.args[0]
        torch.testing.assert_close(
            passed,
            torch.full_like(
                passed,
                3.0,
            ),
        )


class TestPredict:
    def make_stubbed_model(self):
        model = make_model()

        model._sample_prior = Mock(
            return_value=(
                torch.zeros(
                    4,
                    2,
                    3,
                ),
                torch.ones(
                    2,
                    3,
                ),
                torch.zeros(
                    2,
                    3,
                ),
            )
        )
        model._generate = Mock(
            return_value=torch.zeros(
                4,
                2,
                4,
                8,
                8,
            )
        )
        model._deterministic_guess = Mock(
            return_value=None,
        )
        model._output_block = Mock(
            return_value=torch.zeros(
                4,
                2,
                1,
                8,
                8,
            )
        )

        return model

    def make_request(
        self,
        **overrides,
    ):
        values = {
            "condition": torch.ones(
                2,
                2,
                8,
                8,
            ),
            "condition_mask": None,
            "added_features": None,
            "latent_sample_size": 4,
            "output_sample_size": 0,
            "latent_samples": None,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_predict_returns_cvae_output(self):
        model = self.make_stubbed_model()
        request = self.make_request()

        result = model.predict(request)

        model._sample_prior.assert_called_once_with(request)
        assert result.output.shape == (
            4,
            2,
            1,
            8,
            8,
        )
        assert result.mu is None
        assert result.log_var is None
        assert result.samples is None
        assert result.cond_mu.shape == (
            2,
            3,
        )

    def test_predict_restores_output_sample_dimension(self):
        model = self.make_stubbed_model()

        model._generate.return_value = torch.zeros(
            5,
            4,
            2,
            4,
            8,
            8,
        )
        model._output_block.return_value = torch.zeros(
            5,
            4,
            2,
            1,
            8,
            8,
        )

        model.predict(
            self.make_request(
                output_sample_size=5,
            )
        )

        model._output_block.assert_called_once_with(
            model._generate.return_value,
            (
                5,
                4,
                2,
            ),
        )

    def test_predict_adds_deterministic_guess(self):
        model = self.make_stubbed_model()

        generated = torch.ones(
            4,
            2,
            4,
            8,
            8,
        )
        deterministic = torch.full_like(
            generated,
            2.0,
        )

        model._generate.return_value = generated
        model._deterministic_guess.return_value = deterministic

        model.predict(self.make_request())

        passed = model._output_block.call_args.args[0]
        torch.testing.assert_close(
            passed,
            torch.full_like(
                passed,
                3.0,
            ),
        )


class TestRecognitionModule:
    def test_spatial_shapes_include_each_downsampling_level(self):
        recognition = Recognition(
            input_channels=2,
            input_spatial_shape=(
                16,
                16,
            ),
            channels=[
                4,
                8,
                16,
            ],
            latent_size=3,
            config=make_config(
                channels=[
                    4,
                    8,
                    16,
                ],
                condition_embedding_channels=[
                    4,
                    8,
                    16,
                ],
                transpose_kernel_sizes=[
                    3,
                    3,
                ],
            ),
        )

        assert len(recognition.spatial_shapes) == 3
        assert recognition.spatial_shapes[0] == (
            16,
            16,
        )

    def test_forward_calls_mapping_down_blocks_and_bottleneck(self):
        recognition = object.__new__(Recognition)
        nn.Module.__init__(recognition)

        calls = []

        class Stage(nn.Module):
            def __init__(
                self,
                name,
                output,
            ):
                super().__init__()
                self.name = name
                self.output = output

            def forward(
                self,
                value,
            ):
                calls.append(self.name)
                return self.output

        initial = TensorMask(
            tensor=torch.ones(
                2,
                4,
                8,
                8,
            ),
            mask=None,
        )
        down = TensorMask(
            tensor=torch.ones(
                2,
                8,
                4,
                4,
            ),
            mask=None,
        )
        latent = LatentVector(
            mu=torch.ones(
                2,
                3,
            ),
            log_var=torch.zeros(
                2,
                3,
            ),
        )

        recognition.initial_mapping = Stage(
            "initial",
            initial,
        )
        recognition.down_blocks = nn.ModuleList(
            [
                Stage(
                    "down",
                    down,
                )
            ]
        )
        recognition.bottleneck = Stage(
            "bottleneck",
            latent,
        )

        result = recognition(
            TensorMask(
                tensor=torch.ones(
                    2,
                    2,
                    8,
                    8,
                ),
                mask=None,
            )
        )

        assert calls == [
            "initial",
            "down",
            "bottleneck",
        ]
        assert result is latent


class TestGenerationModule:
    def make_generation(
        self,
        *,
        generator=None,
    ):
        config = make_config(
            GENERATOR=generator,
        )

        return Generation(
            latent_size=3,
            channels=[
                8,
                4,
            ],
            resize_shapes=[
                (
                    4,
                    4,
                ),
                (
                    8,
                    8,
                ),
            ],
            config=config,
        )

    def test_initializes_bottleneck_metadata(self):
        generation = self.make_generation()

        assert generation.bottleneck_dim == 8
        assert generation.bottleneck_shape == (
            4,
            4,
        )
        assert generation.resize_shapes == [
            (
                8,
                8,
            )
        ]

    def test_builds_one_up_block_per_resize_shape(self):
        generation = self.make_generation()

        assert len(generation.up_blocks) == 1

    def test_forward_without_generator(self):
        generation = self.make_generation()

        class PassThroughUpBlock(nn.Module):
            def forward(
                self,
                input,
                skip,
                resize_shape,
            ):
                return TensorMask(
                    tensor=torch.nn.functional.interpolate(
                        input.tensor,
                        size=resize_shape,
                    ),
                    mask=None,
                )

        generation.up_blocks = nn.ModuleList([PassThroughUpBlock()])

        result = generation(
            torch.ones(
                2,
                3,
            )
        )

        assert result.shape == (
            2,
            8,
            8,
            8,
        )


import cccma_ppp.models.unet_models.cvae as cvae_module


class TestAdditionalCVAEUNetConfig:
    def test_build_forwards_all_arguments(self, monkeypatch):
        config = make_config()
        built = object()
        constructor = Mock(return_value=built)
        monkeypatch.setattr(
            cvae_module,
            "cVAEUNet",
            constructor,
        )

        input_shape = np.asarray([2, 8, 8])
        output_shape = np.asarray([1, 8, 8])

        result = config.build(
            input_shape=input_shape,
            output_shape=output_shape,
            added_features_dim=3,
        )

        assert result is built
        constructor.assert_called_once_with(
            config=config,
            input_shape=input_shape,
            output_shape=output_shape,
            added_features_dim=3,
        )

    def test_false_partial_conv_attribute_does_not_expect_mask(self):
        config = object.__new__(cVAEUNetConfig)
        config.block_config = SimpleNamespace(
            use_partial_conv=False,
        )

        assert config.EXPECTS_MASK is False

    def test_unshared_guess_allows_different_output_settings(self):
        resolved = SimpleNamespace(
            channels=[4, 8],
            GENERATOR=None,
            output_activation="tanh",
            output_block_hidden_channels=7,
        )
        selector = SimpleNamespace(
            share_output_block=False,
            get_model_config=Mock(
                return_value=resolved,
            ),
        )

        config = make_config(
            deterministic_guess_config=selector,
        )

        assert config.share_output_block is False
        assert config.deterministic_guess_config is resolved


class TestAdditionalCVAEUNetInitialization:
    def test_condition_embedding_increases_generation_latent_size(self):
        with (
            patch.object(
                cVAEUNet,
                "_validate_checkpoint_compatibility",
            ),
            patch.object(
                cVAEUNet,
                "_initialize_weights",
            ),
            patch.object(
                cvae_module,
                "Recognition",
            ) as recognition,
            patch.object(
                cvae_module,
                "Generation",
            ) as generation,
            patch.object(
                cvae_module,
                "UNetOutput",
            ),
        ):
            recognition.side_effect = [
                SimpleNamespace(
                    spatial_shapes=[
                        (8, 8),
                        (4, 4),
                    ]
                ),
                Mock(),
            ]

            cVAEUNet(
                config=make_config(
                    latent_size=5,
                    condition_embedding_size=3,
                    condemb_to_decoder=True,
                ),
                input_shape=(2, 8, 8),
                output_shape=(1, 8, 8),
            )

        assert generation.call_args.kwargs["latent_size"] == 8

    def test_recognition_and_condition_channel_counts(self):
        with (
            patch.object(
                cVAEUNet,
                "_validate_checkpoint_compatibility",
            ),
            patch.object(
                cVAEUNet,
                "_initialize_weights",
            ),
            patch.object(
                cvae_module,
                "Recognition",
            ) as recognition,
            patch.object(
                cvae_module,
                "Generation",
            ),
            patch.object(
                cvae_module,
                "UNetOutput",
            ),
        ):
            recognition.side_effect = [
                SimpleNamespace(
                    spatial_shapes=[
                        (8, 8),
                        (4, 4),
                    ]
                ),
                Mock(),
            ]

            cVAEUNet(
                config=make_config(),
                input_shape=(2, 8, 8),
                output_shape=(3, 8, 8),
                added_features_dim=4,
            )

        assert recognition.call_args_list[0].kwargs["input_channels"] == 9
        assert recognition.call_args_list[1].kwargs["input_channels"] == 6

    def test_depth_equal_to_spatial_limit_is_allowed(self):
        model = make_model(
            input_shape=(2, 4, 4),
            output_shape=(1, 4, 4),
            channels=[4, 8, 16],
            condition_embedding_channels=[
                4,
                8,
                16,
            ],
            transpose_kernel_sizes=[3, 3],
        )

        assert model.input_shape == (2, 4, 4)


class TestAdditionalPrepareInput:
    def test_condition_mask_is_ignored_without_input_mask(self):
        model = make_model()

        result = model._prepare_input(
            x=torch.ones(2, 1, 8, 8),
            x_mask=None,
            condition=torch.ones(
                2,
                2,
                8,
                8,
            ),
            condition_mask=torch.zeros(
                2,
                2,
                8,
                8,
            ),
        )

        assert result.mask is None

    def test_added_features_use_nearest_resize(self, monkeypatch):
        model = make_model()
        features = torch.ones(
            2,
            2,
            1,
            1,
        )
        resize = Mock(
            return_value=torch.ones(
                2,
                2,
                8,
                8,
            )
        )

        monkeypatch.setattr(
            cvae_module,
            "_resize_tensor",
            resize,
        )

        result = model._prepare_input(
            x=torch.ones(
                2,
                1,
                8,
                8,
            ),
            x_mask=None,
            added_features=features,
        )

        resize.assert_called_once_with(
            features,
            (8, 8),
            mode="nearest",
        )
        assert result.tensor.shape == (
            2,
            3,
            8,
            8,
        )

    def test_condition_then_features_channel_order(self):
        model = make_model()

        result = model._prepare_input(
            x=torch.full(
                (1, 1, 4, 4),
                1.0,
            ),
            x_mask=None,
            condition=torch.full(
                (1, 1, 4, 4),
                2.0,
            ),
            added_features=torch.full(
                (1, 1, 4, 4),
                3.0,
            ),
        )

        assert torch.all(result.tensor[:, 0] == 1)
        assert torch.all(result.tensor[:, 1] == 2)
        assert torch.all(result.tensor[:, 2] == 3)


class TestAdditionalGenerate:
    def test_generate_without_condition_embedding(self):
        model = make_model(
            condemb_to_decoder=True,
        )
        model.generation = FixedGeneration(
            torch.zeros(
                6,
                4,
                8,
                8,
            )
        )

        result = model._generate(
            latent_samples=torch.ones(
                3,
                2,
                5,
            ),
            condition_embedding=None,
        )

        assert model.generation.calls[0][0].shape == (6, 5)
        assert result.shape == (
            3,
            2,
            4,
            8,
            8,
        )


class TestAdditionalForwardPredict:
    def test_forward_eval_keeps_requested_output_samples(self):
        helper = TestForward()
        model = helper.make_stubbed_model()
        model.config.GENERATOR = SimpleNamespace(
            num_training_noise_samples=7,
        )
        model.eval()

        model._generate.return_value = torch.zeros(
            2,
            4,
            2,
            4,
            8,
            8,
        )
        model._output_block.return_value = torch.zeros(
            2,
            4,
            2,
            1,
            8,
            8,
        )

        model.forward(
            helper.make_request(
                output_sample_size=2,
            )
        )

        assert model._generate.call_args.kwargs["num_output_samples"] == 2
        model._output_block.assert_called_once_with(
            model._generate.return_value,
            (2, 4, 2),
        )

    def test_predict_passes_condition_to_deterministic_guess(self):
        helper = TestPredict()
        model = helper.make_stubbed_model()

        condition = torch.ones(
            2,
            2,
            8,
            8,
        )
        condition_mask = torch.ones(
            2,
            1,
            8,
            8,
        )
        features = torch.ones(
            2,
            1,
            8,
            8,
        )

        model.predict(
            helper.make_request(
                condition=condition,
                condition_mask=condition_mask,
                added_features=features,
            )
        )

        model._deterministic_guess.assert_called_once_with(
            input=condition,
            input_mask=condition_mask,
            added_features=features,
        )


class TestAdditionalOutputBlock:
    def test_output_block_can_change_channels(self):
        model = make_model()

        class ChangeChannels(nn.Module):
            def forward(self, value):
                return value[:, :2]

        model.output = ChangeChannels()

        result = model._output_block(
            torch.ones(
                3,
                2,
                5,
                4,
                4,
            ),
            sample_sizes=(3, 2),
        )

        assert result.shape == (
            3,
            2,
            2,
            4,
            4,
        )


class TestAdditionalRecognition:
    def test_get_spatial_shapes_calls_each_block(self):
        recognition = object.__new__(Recognition)
        nn.Module.__init__(recognition)

        first = Mock(output_shape=Mock(return_value=(8, 8)))
        second = Mock(output_shape=Mock(return_value=(4, 4)))
        recognition.down_blocks = [
            first,
            second,
        ]

        result = recognition._get_spatial_shapes((16, 16))

        assert result == [
            (16, 16),
            (8, 8),
            (4, 4),
        ]
        first.output_shape.assert_called_once_with((16, 16))
        second.output_shape.assert_called_once_with((8, 8))
