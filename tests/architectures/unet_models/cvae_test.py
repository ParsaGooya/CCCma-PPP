from types import SimpleNamespace
from unittest.mock import Mock, patch
import warnings

import numpy as np
import pytest
from cccma_ppp.architectures.models_abc import DeterministicRequest
from cccma_ppp.architectures.unet.deterministic import UNetConfig
import torch
import torch.nn as nn
import cccma_ppp.architectures.unet.cvae as cvae_module
from cccma_ppp.architectures.layers.unet import (
    UNetOutputSIC,
)

from cccma_ppp.architectures.unet.cvae import (
    Generation,
    Recognition,
    cVAEUNet,
    cVAEUNetConfig,
    cVAEUNetSIC,
    cVAEUNetSICEConfig,
)
from cccma_ppp.architectures.layers.conv import (
    ConvBlockConfig,
    ConvNeXtBlockConfig,
    LatentVector,
    PartialConvBlockConfig,
    TensorMask,
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
    @pytest.mark.pruned
    def test_basic_initialization(self):
        config = make_config()

        assert config.channels == [4, 8]
        assert config.latent_size == 3
        assert config.condition_embedding_channels == [4, 8]
        assert config.condition_embedding_size == 3
        assert config.NUM_INPUT_DIMS == 3
        assert config.NUM_OUTPUT_DIMS == 3

    @pytest.mark.pruned
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

    @pytest.mark.pruned
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

    @pytest.mark.pruned
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

    @pytest.mark.pruned
    def test_without_deterministic_guess_disables_shared_output(self):
        config = make_config(
            deterministic_guess_config=None,
        )

        assert config.share_output_block is False

    @pytest.mark.pruned
    def test_expects_mask_with_regular_convolution(self):
        config = make_config(
            block_config=ConvBlockConfig(name="conv"),
        )

        assert config.EXPECTS_MASK is False

    @pytest.mark.pruned
    def test_expects_mask_with_partial_convolution(self):
        config = make_config(
            block_config=PartialConvBlockConfig(name="conv"),
        )

        assert config.EXPECTS_MASK is True

    @pytest.mark.pruned
    def test_expects_mask_with_partial_conv_attribute(self):
        block_config = SimpleNamespace(
            use_partial_conv=True,
        )

        config = object.__new__(cVAEUNetConfig)
        config.block_config = block_config

        assert config.EXPECTS_MASK is True

    @pytest.mark.pruned
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

    @pytest.mark.pruned
    def test_deterministic_guess_channel_mismatch_is_rejected(self):
        resolved = Mock(
            spec=UNetConfig,
            channels=[
                5,
                8,
            ],
            GENERATOR=None,
            output_activation="identity",
            output_block_hidden_channels=32,
        )
        selector = SimpleNamespace(
            checkpoint_config=SimpleNamespace(freeze_weights=False),
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
        resolved = Mock(
            spec=UNetConfig,
            channels=[
                4,
                8,
            ],
            GENERATOR=object(),
            output_activation="identity",
            output_block_hidden_channels=32,
        )
        selector = SimpleNamespace(
            checkpoint_config=SimpleNamespace(freeze_weights=False),
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

    @pytest.mark.pruned
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
        resolved = Mock(
            spec=UNetConfig,
            channels=[
                4,
                8,
            ],
            GENERATOR=None,
            output_activation=activation,
            output_block_hidden_channels=hidden_channels,
        )
        selector = SimpleNamespace(
            checkpoint_config=SimpleNamespace(freeze_weights=False),
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

    @pytest.mark.pruned
    def test_valid_shared_output_configuration(self):
        resolved = Mock(
            spec=UNetConfig,
            channels=[
                4,
                8,
            ],
            GENERATOR=None,
            output_activation="identity",
            output_block_hidden_channels=32,
        )
        selector = SimpleNamespace(
            checkpoint_config=SimpleNamespace(freeze_weights=False),
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
    @pytest.mark.pruned
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

    @pytest.mark.pruned
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

    @pytest.mark.pruned
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

    @pytest.mark.pruned
    def test_defaults_added_features_to_zero(self):
        model = make_model(
            added_features_dim=None,
        )

        assert model.added_features_dim == 0

    @pytest.mark.pruned
    def test_preserves_added_features_dimension(self):
        model = make_model(
            added_features_dim=3,
        )

        assert model.added_features_dim == 3

    @pytest.mark.pruned
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

    @pytest.mark.pruned
    def test_condition_requests_log_variance_for_dependent_latent(self):
        with patch(
            "cccma_ppp.architectures.unet.cvae.Recognition",
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
                    "cccma_ppp.architectures.unet.cvae.Generation",
                ),
                patch(
                    "cccma_ppp.architectures.unet.cvae.UNetOutput",
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

    @pytest.mark.pruned
    def test_condition_skips_log_variance_for_independent_latent(self):
        with patch(
            "cccma_ppp.architectures.unet.cvae.Recognition",
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
                    "cccma_ppp.architectures.unet.cvae.Generation",
                ),
                patch(
                    "cccma_ppp.architectures.unet.cvae.UNetOutput",
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

    @pytest.mark.pruned
    def test_condition_flow_skips_log_variance(self):
        config = make_config(
            condition_dependant_latent=True,
        )
        config.condition_dependant_flow = True

        with patch(
            "cccma_ppp.architectures.unet.cvae.Recognition",
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
                    "cccma_ppp.architectures.unet.cvae.Generation",
                ),
                patch(
                    "cccma_ppp.architectures.unet.cvae.UNetOutput",
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

    @pytest.mark.pruned
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

    @pytest.mark.pruned
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
    @pytest.mark.pruned
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

    @pytest.mark.pruned
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

    @pytest.mark.pruned
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

    @pytest.mark.pruned
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
    @pytest.mark.pruned
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

    @pytest.mark.pruned
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
    @pytest.mark.pruned
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

    @pytest.mark.pruned
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
    @pytest.mark.pruned
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

    @pytest.mark.pruned
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
    @pytest.mark.pruned
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

    @pytest.mark.pruned
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

    @pytest.mark.pruned
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

    @pytest.mark.pruned
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
    @pytest.mark.pruned
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

    @pytest.mark.pruned
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

    @pytest.mark.pruned
    def test_builds_one_up_block_per_resize_shape(self):
        generation = self.make_generation()

        assert len(generation.up_blocks) == 1

    @pytest.mark.pruned
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


class TestAdditionalCVAEUNetConfig:
    @pytest.mark.pruned
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

    @pytest.mark.pruned
    def test_false_partial_conv_attribute_does_not_expect_mask(self):
        config = object.__new__(cVAEUNetConfig)
        config.block_config = SimpleNamespace(
            use_partial_conv=False,
        )

        assert config.EXPECTS_MASK is False

    def test_unshared_guess_allows_different_output_settings(self):
        resolved = Mock(
            spec=UNetConfig,
            channels=[4, 8],
            GENERATOR=None,
            output_activation="tanh",
            output_block_hidden_channels=7,
        )
        selector = SimpleNamespace(
            checkpoint_config=SimpleNamespace(freeze_weights=False),
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
    @pytest.mark.pruned
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

    @pytest.mark.pruned
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

    @pytest.mark.pruned
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
    @pytest.mark.pruned
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

    @pytest.mark.pruned
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
    @pytest.mark.pruned
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
    @pytest.mark.pruned
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

    @pytest.mark.pruned
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
    @pytest.mark.pruned
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
    @pytest.mark.pruned
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


def make_real_unet_config(
    *,
    channels=None,
    generator=None,
    output_activation="identity",
    output_block_hidden_channels=32,
    clip_output=None,
):
    if channels is None:
        channels = [4, 8]

    config = UNetConfig(
        channels=channels,
        block_config=ConvBlockConfig(
            name="conv",
        ),
        transpose_kernel_sizes=3,
        output_activation=output_activation,
        output_block_hidden_channels=output_block_hidden_channels,
        GENERATOR=generator,
    )

    if clip_output is not None:
        config.clip_output = clip_output

    return config


def make_deterministic_selector(
    resolved,
    *,
    share_output_block=False,
    freeze_weights=False,
):
    return SimpleNamespace(
        checkpoint_config=SimpleNamespace(
            freeze_weights=freeze_weights,
        ),
        share_output_block=share_output_block,
        get_model_config=Mock(
            return_value=resolved,
        ),
    )


class TestAdditionalDeterministicGuessConfiguration:
    @pytest.mark.pruned
    def test_selector_defaults(self):
        selector = cvae_module.UNetSelector(config={})

        assert selector.type == "unet"
        assert selector.share_output_block is True

    @pytest.mark.pruned
    def test_no_deterministic_guess_disables_freezing(self):
        config = make_config(
            deterministic_guess_config=None,
        )

        assert config.share_output_block is False
        assert config.freeze_deterministic is False

    @pytest.mark.pruned
    def test_resolved_deterministic_configuration_is_stored(self):
        resolved = make_real_unet_config()

        selector = make_deterministic_selector(
            resolved,
            share_output_block=False,
            freeze_weights=True,
        )

        config = make_config(
            deterministic_guess_config=selector,
        )

        assert config.deterministic_guess_config is resolved
        assert config.share_output_block is False
        assert config.freeze_deterministic is True

    def test_resolved_configuration_must_be_unet_config(self):
        selector = make_deterministic_selector(
            SimpleNamespace(
                channels=[4, 8],
            )
        )

        with pytest.raises(
            TypeError,
            match="resolve to a UNetConfig",
        ):
            make_config(
                deterministic_guess_config=selector,
            )

    def test_shared_output_rejects_clip_output_mismatch(self):
        resolved = make_real_unet_config()
        resolved.clip_output = True

        selector = make_deterministic_selector(
            resolved,
            share_output_block=True,
        )

        with pytest.raises(
            ValueError,
            match="compatible output-block settings",
        ):
            make_config(
                deterministic_guess_config=selector,
            )

    @pytest.mark.pruned
    def test_shared_output_accepts_matching_clip_output(self):
        resolved = make_real_unet_config()
        resolved.clip_output = None

        selector = make_deterministic_selector(
            resolved,
            share_output_block=True,
        )

        config = make_config(
            deterministic_guess_config=selector,
        )

        assert config.share_output_block is True
        assert config.deterministic_guess_config is resolved

    @pytest.mark.pruned
    def test_unshared_output_ignores_clip_output_difference(self):
        resolved = make_real_unet_config(
            output_activation="tanh",
            output_block_hidden_channels=7,
        )
        resolved.clip_output = True

        selector = make_deterministic_selector(
            resolved,
            share_output_block=False,
        )

        config = make_config(
            deterministic_guess_config=selector,
        )

        assert config.share_output_block is False
        assert config.deterministic_guess_config is resolved


class TestAdditionalConfigProperties:
    @pytest.mark.pruned
    def test_expects_mask_for_convnext_partial_convolution(self):
        block_config = Mock(
            spec=ConvNeXtBlockConfig,
        )
        block_config.use_partial_conv = True

        config = object.__new__(cVAEUNetConfig)
        config.block_config = block_config

        assert config.EXPECTS_MASK is True

    @pytest.mark.pruned
    def test_expects_no_mask_for_convnext_standard_convolution(self):
        block_config = Mock(
            spec=ConvNeXtBlockConfig,
        )
        block_config.use_partial_conv = False

        config = object.__new__(cVAEUNetConfig)
        config.block_config = block_config

        assert config.EXPECTS_MASK is False

    @pytest.mark.pruned
    def test_single_channel_level_expands_to_no_kernels(self):
        config = make_config(
            channels=[4],
            condition_embedding_channels=[4],
            transpose_kernel_sizes=3,
        )

        assert config.transpose_kernel_sizes == []

    def test_condition_channels_reference_model_channels(self):
        channels = [4, 8, 16]

        config = make_config(
            channels=channels,
            condition_embedding_channels=None,
            transpose_kernel_sizes=[3, 3],
        )

        assert config.condition_embedding_channels is channels


class TestAdditionalInitializationBranches:
    @pytest.mark.pruned
    def test_condition_embedding_not_added_to_generation_when_disabled(self):
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
                    condition_dependant_latent=True,
                    condemb_to_decoder=False,
                ),
                input_shape=(2, 8, 8),
                output_shape=(1, 8, 8),
            )

        assert generation.call_args.kwargs["latent_size"] == 5

    def test_checkpoint_configuration_loads_state(self):
        config = make_config()
        checkpoint_config = object()
        config.checkpoint_config = checkpoint_config

        with (
            patch.object(cVAEUNet, "_validate_checkpoint_compatibility"),
            patch.object(cVAEUNet, "_validate_checkpoint_compatibility"),
            patch.object(
                cVAEUNet,
                "_load_state_dict",
            ) as load_state,
            patch.object(
                cVAEUNet,
                "_initialize_weights",
            ) as initialize,
        ):
            cVAEUNet(
                config=config,
                input_shape=(2, 8, 8),
                output_shape=(1, 8, 8),
            )

        load_state.assert_called_once_with(checkpoint_config)
        initialize.assert_not_called()

    @pytest.mark.pruned
    def test_deterministic_guess_is_built(self):
        deterministic_config = Mock(
            spec=UNetConfig,
        )
        deterministic_model = Mock()
        deterministic_config.build.return_value = deterministic_model

        config = make_config()
        config.deterministic_guess_config = deterministic_config
        config.share_output_block = False
        config.freeze_deterministic = False

        model = cVAEUNet(
            config=config,
            input_shape=(2, 8, 8),
            output_shape=(1, 8, 8),
            added_features_dim=3,
        )

        deterministic_config.build.assert_called_once_with(
            input_shape=(2, 8, 8),
            output_shape=(1, 8, 8),
            added_features_dim=3,
        )
        assert model.deterministic_guess is deterministic_model

    def test_shared_output_uses_deterministic_output_block(
        self,
    ):
        shared_output = nn.Identity()
        deterministic_model = SimpleNamespace(
            output_block=shared_output,
        )
        deterministic_config = Mock(
            spec=UNetConfig,
        )
        deterministic_config.build.return_value = deterministic_model

        config = make_config()
        config.deterministic_guess_config = deterministic_config
        config.share_output_block = True
        config.freeze_deterministic = False

        model = cVAEUNet(
            config=config,
            input_shape=(2, 8, 8),
            output_shape=(1, 8, 8),
        )

        assert model.output is shared_output

    def test_frozen_deterministic_branch_uses_its_output_block(
        self,
    ):
        output = nn.Conv2d(
            4,
            1,
            kernel_size=1,
        )

        for parameter in output.parameters():
            parameter.requires_grad = False

        deterministic_model = SimpleNamespace(
            output_block=output,
        )
        deterministic_config = Mock(
            spec=UNetConfig,
        )
        deterministic_config.build.return_value = deterministic_model

        config = make_config()
        config.deterministic_guess_config = deterministic_config
        config.share_output_block = False
        config.freeze_deterministic = True

        model = cVAEUNet(
            config=config,
            input_shape=(2, 8, 8),
            output_shape=(1, 8, 8),
        )

        assert model.output is output
        assert all(parameter.requires_grad for parameter in output.parameters())

    @pytest.mark.pruned
    def test_unfrozen_unshared_branch_builds_new_output(
        self,
    ):
        deterministic_model = SimpleNamespace(
            output_block=nn.Identity(),
        )
        deterministic_config = Mock(
            spec=UNetConfig,
        )
        deterministic_config.build.return_value = deterministic_model

        config = make_config()
        config.deterministic_guess_config = deterministic_config
        config.share_output_block = False
        config.freeze_deterministic = False

        with patch.object(
            cVAEUNet,
            "_build_output",
            return_value=nn.Identity(),
        ) as build_output:
            model = cVAEUNet(
                config=config,
                input_shape=(2, 8, 8),
                output_shape=(1, 8, 8),
            )

        build_output.assert_called_once_with(
            in_channels=4,
            out_channels=1,
        )
        assert model.output is build_output.return_value


class TestAdditionalPrepareInputBranches:
    def test_condition_mask_is_broadcast(self):
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
        condition_mask = torch.ones(
            2,
            1,
            8,
            8,
        )

        result = model._prepare_input(
            x=x,
            x_mask=torch.ones_like(x),
            condition=condition,
            condition_mask=condition_mask,
        )

        assert result.mask.shape == (
            2,
            2,
            8,
            8,
        )
        assert torch.all(result.mask == 1)

    @pytest.mark.pruned
    def test_condition_is_resized_before_mask_broadcast(
        self,
        monkeypatch,
    ):
        model = make_model()

        resized_condition = torch.ones(
            2,
            2,
            8,
            8,
        )
        resize = Mock(return_value=resized_condition)
        broadcast = Mock(
            side_effect=lambda mask, value: (
                torch.ones_like(value) if mask is not None else None
            )
        )

        monkeypatch.setattr(
            cvae_module,
            "_resize_tensor",
            resize,
        )
        monkeypatch.setattr(
            cvae_module,
            "_broadcast_mask",
            broadcast,
        )

        condition = torch.ones(
            2,
            2,
            4,
            4,
        )

        model._prepare_input(
            x=torch.ones(
                2,
                1,
                8,
                8,
            ),
            x_mask=torch.ones(
                2,
                1,
                8,
                8,
            ),
            condition=condition,
            condition_mask=torch.ones(
                2,
                1,
                4,
                4,
            ),
        )

        resize.assert_called_once_with(
            condition,
            (8, 8),
        )
        assert broadcast.call_args_list[1].args[1] is resized_condition

    @pytest.mark.pruned
    def test_feature_mask_is_not_created_without_input_mask(
        self,
    ):
        model = make_model()

        result = model._prepare_input(
            x=torch.ones(
                2,
                1,
                8,
                8,
            ),
            x_mask=None,
            added_features=torch.ones(
                2,
                2,
                1,
                1,
            ),
        )

        assert result.mask is None


class TestAdditionalGenerateBranches:
    def test_generate_with_output_samples_restores_all_dimensions(
        self,
    ):
        model = make_model(
            condition_dependant_latent=True,
            condemb_to_decoder=False,
        )

        model.generation = FixedGeneration(
            torch.zeros(
                5,
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
            num_output_samples=5,
        )

        assert model.generation.calls[0][1] == 5
        assert result.shape == (
            5,
            3,
            2,
            4,
            8,
            8,
        )

    def test_condition_embedding_values_are_repeated_for_each_latent(
        self,
    ):
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

        latent = torch.zeros(
            3,
            2,
            5,
        )
        condition = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        )

        model._generate(
            latent_samples=latent,
            condition_embedding=condition,
        )

        passed = model.generation.calls[0][0].reshape(
            3,
            2,
            7,
        )

        expected_condition = condition.unsqueeze(0).expand(
            3,
            2,
            2,
        )

        torch.testing.assert_close(
            passed[..., -2:],
            expected_condition,
        )


class TestAdditionalForwardBranches:
    @pytest.mark.pruned
    def test_forward_output_samples_use_three_sample_dimensions(
        self,
    ):
        helper = TestForward()
        model = helper.make_stubbed_model()

        model._generate.return_value = torch.zeros(
            3,
            4,
            2,
            4,
            8,
            8,
        )
        model._output_block.return_value = torch.zeros(
            3,
            4,
            2,
            1,
            8,
            8,
        )

        model.forward(
            helper.make_request(
                output_sample_size=3,
            )
        )

        model._output_block.assert_called_once_with(
            model._generate.return_value,
            (
                3,
                4,
                2,
            ),
        )

    @pytest.mark.pruned
    def test_forward_passes_all_inputs_to_condition_and_recognition(
        self,
    ):
        helper = TestForward()
        model = helper.make_stubbed_model()

        target = torch.ones(
            2,
            1,
            8,
            8,
        )
        target_mask = torch.ones_like(target)
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
            3,
            8,
            8,
        )

        model.forward(
            helper.make_request(
                target=target,
                target_mask=target_mask,
                condition=condition,
                condition_mask=condition_mask,
                added_features=features,
            )
        )

        model._condition.assert_called_once_with(
            condition=condition,
            condition_mask=condition_mask,
            added_features=features,
        )
        model._recognition.assert_called_once_with(
            x=target,
            x_mask=target_mask,
            condition=condition,
            condition_mask=condition_mask,
            added_features=features,
        )

    @pytest.mark.pruned
    def test_forward_passes_requested_latent_sample_size(
        self,
    ):
        helper = TestForward()
        model = helper.make_stubbed_model()

        model.forward(
            helper.make_request(
                latent_sample_size=7,
            )
        )

        mu, log_var, sample_size = model._sample.call_args.args

        assert mu is model._recognition.return_value[0]
        assert log_var is model._recognition.return_value[1]
        assert sample_size == 7


class TestAdditionalPredictBranches:
    @pytest.mark.pruned
    def test_deterministic_only_replaces_generated_output(
        self,
    ):
        helper = TestPredict()
        model = helper.make_stubbed_model()

        generated = torch.full(
            (
                4,
                2,
                1,
                8,
                8,
            ),
            -5.0,
        )
        deterministic = torch.full(
            (
                2,
                1,
                8,
                8,
            ),
            2.0,
        )

        model._generate.return_value = generated
        model._deterministic_guess.return_value = deterministic

        with pytest.warns(
            UserWarning,
            match="DETERMINISTIC GUESS",
        ):
            result = model.predict(
                helper.make_request(
                    output_sample_size=5,
                ),
                deterministic_guess_only=True,
            )

        passed = model._output_block.call_args.args[0]

        torch.testing.assert_close(
            passed,
            deterministic.unsqueeze(0),
        )
        assert model._output_block.call_args.args[1] == (
            1,
            2,
        )
        assert result.deterministic_guess is True

    def test_deterministic_only_warns_once(self):
        helper = TestPredict()
        model = helper.make_stubbed_model()
        model._deterministic_guess.return_value = torch.ones(
            2,
            1,
            8,
            8,
        )

        with pytest.warns(
            UserWarning,
            match="DETERMINISTIC GUESS",
        ):
            model.predict(
                helper.make_request(),
                deterministic_guess_only=True,
            )

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")

            model.predict(
                helper.make_request(),
                deterministic_guess_only=True,
            )

        assert not captured

    @pytest.mark.pruned
    def test_deterministic_only_without_guess_keeps_generated_output(
        self,
    ):
        helper = TestPredict()
        model = helper.make_stubbed_model()
        model._deterministic_guess.return_value = None

        generated = model._generate.return_value

        result = model.predict(
            helper.make_request(),
            deterministic_guess_only=True,
        )

        assert model._output_block.call_args.args[0] is generated
        assert result.deterministic_guess is True

    @pytest.mark.pruned
    def test_predict_sets_predict_called(self):
        helper = TestPredict()
        model = helper.make_stubbed_model()

        assert model._predict_called is False

        model.predict(helper.make_request())

        assert model._predict_called is True


class TestAdditionalDeterministicGuessRequest:
    @pytest.mark.pruned
    def test_request_contains_all_inputs(self):
        model = make_model()

        decoder = Mock(
            return_value=torch.ones(
                2,
                1,
                8,
                8,
            )
        )
        model.deterministic_guess = SimpleNamespace(
            forward_decoder=decoder,
        )

        input_tensor = torch.ones(
            2,
            2,
            8,
            8,
        )
        input_mask = torch.ones(
            2,
            1,
            8,
            8,
        )
        features = torch.ones(
            2,
            3,
            8,
            8,
        )

        model._deterministic_guess(
            input=input_tensor,
            input_mask=input_mask,
            added_features=features,
        )

        request = decoder.call_args.args[0]

        assert isinstance(
            request,
            DeterministicRequest,
        )
        assert request.input is input_tensor
        assert request.input_mask is input_mask
        assert request.added_features is features


class TestAdditionalRecognitionBranches:
    @pytest.mark.pruned
    def test_recognition_single_channel_has_no_down_blocks(
        self,
    ):
        config = make_config(
            channels=[4],
            condition_embedding_channels=[4],
            transpose_kernel_sizes=[],
        )

        recognition = Recognition(
            input_channels=2,
            input_spatial_shape=(8, 8),
            channels=[4],
            latent_size=3,
            config=config,
        )

        assert len(recognition.down_blocks) == 0
        assert recognition.spatial_shapes == [
            (
                8,
                8,
            )
        ]

    @pytest.mark.pruned
    def test_bottleneck_receives_final_spatial_shape(
        self,
        monkeypatch,
    ):
        captured = {}

        original = cvae_module.build_conv_block

        def wrapped_build(
            in_channels,
            out_channels,
            config,
            **kwargs,
        ):
            if kwargs.get("latent_size") is not None:
                captured.update(kwargs)

            return original(
                in_channels,
                out_channels,
                config,
                **kwargs,
            )

        monkeypatch.setattr(
            cvae_module,
            "build_conv_block",
            wrapped_build,
        )

        Recognition(
            input_channels=2,
            input_spatial_shape=(9, 11),
            channels=[4, 8],
            latent_size=3,
            config=make_config(),
        )

        assert captured["block_output_shape"] == (
            8,
            5,
            6,
        )


class TestAdditionalGenerationConfiguration:
    @pytest.mark.pruned
    @pytest.mark.parametrize(
        (
            "noise_level",
            "expected_noise",
            "expected_block_noise",
        ),
        [
            (
                "low",
                True,
                False,
            ),
            (
                "medium",
                True,
                True,
            ),
            (
                "full",
                True,
                True,
            ),
        ],
    )
    def test_single_up_block_noise_configuration(
        self,
        noise_level,
        expected_noise,
        expected_block_noise,
    ):
        generator = SimpleNamespace(
            noise_level=noise_level,
            num_training_noise_samples=2,
        )

        generation = Generation(
            latent_size=3,
            channels=[8, 4],
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
            config=make_config(
                GENERATOR=generator,
            ),
        )

        block = generation.up_blocks[0]

        assert block.inject_noise is expected_noise
        assert block.inject_noise_in_block is expected_block_noise

    @pytest.mark.pruned
    def test_medium_noise_only_injects_upsampling_noise_in_final_block(
        self,
    ):
        generator = SimpleNamespace(
            noise_level="medium",
            num_training_noise_samples=2,
        )

        generation = Generation(
            latent_size=3,
            channels=[
                16,
                8,
                4,
            ],
            resize_shapes=[
                (
                    2,
                    2,
                ),
                (
                    4,
                    4,
                ),
                (
                    8,
                    8,
                ),
            ],
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
                GENERATOR=generator,
            ),
        )

        assert generation.up_blocks[0].inject_noise is False
        assert generation.up_blocks[1].inject_noise is True

    @pytest.mark.pruned
    def test_low_noise_disables_internal_block_noise(
        self,
    ):
        generator = SimpleNamespace(
            noise_level="low",
            num_training_noise_samples=2,
        )

        generation = Generation(
            latent_size=3,
            channels=[
                16,
                8,
                4,
            ],
            resize_shapes=[
                (
                    2,
                    2,
                ),
                (
                    4,
                    4,
                ),
                (
                    8,
                    8,
                ),
            ],
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
                GENERATOR=generator,
            ),
        )

        assert all(
            block.inject_noise_in_block is False for block in generation.up_blocks
        )

    @pytest.mark.pruned
    def test_full_noise_enables_noise_in_all_blocks(
        self,
    ):
        generator = SimpleNamespace(
            noise_level="full",
            num_training_noise_samples=2,
        )

        generation = Generation(
            latent_size=3,
            channels=[
                16,
                8,
                4,
            ],
            resize_shapes=[
                (
                    2,
                    2,
                ),
                (
                    4,
                    4,
                ),
                (
                    8,
                    8,
                ),
            ],
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
                GENERATOR=generator,
            ),
        )

        assert all(block.inject_noise for block in generation.up_blocks)
        assert all(block.inject_noise_in_block for block in generation.up_blocks)


class TestAdditionalGenerationForward:
    def test_generator_repeats_tensor_mask(
        self,
        monkeypatch,
    ):
        generator = SimpleNamespace(
            noise_level="full",
            num_training_noise_samples=3,
        )

        generation = Generation(
            latent_size=3,
            channels=[8, 4],
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
            config=make_config(
                GENERATOR=generator,
            ),
        )

        repeated = TensorMask(
            tensor=torch.zeros(
                6,
                8,
                4,
                4,
            ),
            mask=None,
        )
        repeat = Mock(return_value=repeated)

        monkeypatch.setattr(
            cvae_module,
            "_repeat_tensor_mask",
            repeat,
        )

        class CaptureUpBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.input = None

            def forward(
                self,
                input,
                skip,
                resize_shape,
            ):
                self.input = input

                return TensorMask(
                    tensor=torch.zeros(
                        6,
                        4,
                        8,
                        8,
                    ),
                    mask=None,
                )

        block = CaptureUpBlock()
        generation.up_blocks = nn.ModuleList(
            [
                block,
            ]
        )

        result = generation(
            torch.ones(
                2,
                3,
            ),
            num_output_samples=3,
        )

        repeat.assert_called_once()
        assert repeat.call_args.kwargs["repeats"] == 3
        assert block.input is repeated
        assert result.shape == (
            3,
            2,
            4,
            8,
            8,
        )

    @pytest.mark.pruned
    def test_generator_does_not_repeat_for_zero_samples(
        self,
        monkeypatch,
    ):
        generator = SimpleNamespace(
            noise_level="full",
            num_training_noise_samples=3,
        )

        generation = Generation(
            latent_size=3,
            channels=[8, 4],
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
            config=make_config(
                GENERATOR=generator,
            ),
        )

        repeat = Mock(side_effect=AssertionError("repeat should not be called"))
        monkeypatch.setattr(
            cvae_module,
            "_repeat_tensor_mask",
            repeat,
        )

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

        generation.up_blocks = nn.ModuleList(
            [
                PassThroughUpBlock(),
            ]
        )

        result = generation(
            torch.ones(
                2,
                3,
            ),
            num_output_samples=0,
        )

        repeat.assert_not_called()
        assert result.shape == (
            2,
            8,
            8,
            8,
        )

    def test_generation_passes_resize_shapes_in_order(
        self,
    ):
        generation = object.__new__(Generation)
        nn.Module.__init__(generation)

        generation.bottleneck_dim = 8
        generation.bottleneck_shape = (
            2,
            2,
        )
        generation.resize_shapes = [
            (
                4,
                4,
            ),
            (
                8,
                8,
            ),
        ]
        generation.config = SimpleNamespace(
            GENERATOR=None,
        )
        generation.combine_latent = nn.Linear(
            3,
            8 * 2 * 2,
        )

        calls = []

        class ShapeBlock(nn.Module):
            def forward(
                self,
                input,
                skip,
                resize_shape,
            ):
                calls.append(resize_shape)

                return TensorMask(
                    tensor=torch.nn.functional.interpolate(
                        input.tensor,
                        size=resize_shape,
                    ),
                    mask=None,
                )

        generation.up_blocks = nn.ModuleList(
            [
                ShapeBlock(),
                ShapeBlock(),
            ]
        )

        result = generation(
            torch.ones(
                2,
                3,
            )
        )

        assert calls == [
            (
                4,
                4,
            ),
            (
                8,
                8,
            ),
        ]
        assert result.shape == (
            2,
            8,
            8,
            8,
        )


class TestSICConfiguration:
    def make_sic_config(
        self,
        **overrides,
    ):
        defaults = {
            "channels": [4, 8],
            "latent_size": 3,
            "condition_embedding_channels": [
                4,
                8,
            ],
            "condition_embedding_size": 3,
            "block_config": ConvBlockConfig(name="conv"),
            "transpose_kernel_sizes": 3,
            "output_activation": "identity",
            "output_block_hidden_channels": 6,
            "clip_output": False,
        }
        defaults.update(overrides)

        config = cVAEUNetSICEConfig(**defaults)
        config.condition_dependant_flow = False

        if not hasattr(
            config,
            "checkpoint_config",
        ):
            config.checkpoint_config = None

        return config

    @pytest.mark.pruned
    def test_default_clip_output_is_false(self):
        config = self.make_sic_config()

        assert config.clip_output is False

    @pytest.mark.pruned
    def test_clip_output_is_preserved(self):
        config = self.make_sic_config(
            clip_output=True,
        )

        assert config.clip_output is True

    @pytest.mark.pruned
    def test_build_returns_sic_model(self):
        config = self.make_sic_config(
            clip_output=True,
        )

        model = config.build(
            input_shape=np.asarray(
                [
                    2,
                    8,
                    8,
                ]
            ),
            output_shape=np.asarray(
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
            cVAEUNetSIC,
        )
        assert isinstance(
            model.output,
            UNetOutputSIC,
        )
        assert model.added_features_dim == 2

    @pytest.mark.pruned
    def test_build_forwards_all_arguments(
        self,
        monkeypatch,
    ):
        config = self.make_sic_config()
        built = object()
        constructor = Mock(return_value=built)

        monkeypatch.setattr(
            cvae_module,
            "cVAEUNetSIC",
            constructor,
        )

        input_shape = np.asarray(
            [
                2,
                8,
                8,
            ]
        )
        output_shape = np.asarray(
            [
                1,
                8,
                8,
            ]
        )

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


class TestSICModel:
    def make_sic_model(
        self,
        **overrides,
    ):
        config_helper = TestSICConfiguration()
        config = config_helper.make_sic_config(**overrides)

        return cVAEUNetSIC(
            config=config,
            input_shape=(2, 8, 8),
            output_shape=(1, 8, 8),
        )

    @pytest.mark.pruned
    def test_build_output_uses_sic_output_class(self):
        model = self.make_sic_model(
            clip_output=True,
        )

        assert isinstance(
            model.output,
            UNetOutputSIC,
        )
        assert model.output.clip_output is True

    @pytest.mark.pruned
    def test_sic_output_receives_configuration(
        self,
        monkeypatch,
    ):
        config_helper = TestSICConfiguration()
        config = config_helper.make_sic_config(
            output_activation="sigmoid",
            output_block_hidden_channels=9,
            clip_output=True,
        )

        output = Mock(return_value=nn.Identity())
        monkeypatch.setattr(
            cvae_module,
            "UNetOutputSIC",
            output,
        )

        cVAEUNetSIC(
            config=config,
            input_shape=(2, 8, 8),
            output_shape=(3, 8, 8),
        )

        output.assert_called_once_with(
            in_channels=4,
            out_channels=3,
            hidden_channels=9,
            activation="sigmoid",
            clip_output=True,
        )

    @pytest.mark.pruned
    def test_sic_model_clips_output(self):
        model = self.make_sic_model(
            clip_output=True,
            output_activation="identity",
            output_block_hidden_channels=None,
        )

        with torch.no_grad():
            final_layer = model.output.layers[-1]
            final_layer.weight.fill_(10.0)
            final_layer.bias.fill_(10.0)

        value = torch.ones(
            2,
            4,
            8,
            8,
        )

        result = model.output(value)

        assert torch.all(result >= 0)
        assert torch.all(result <= 1)

    def test_sic_model_without_clipping_can_exceed_one(
        self,
    ):
        model = self.make_sic_model(
            clip_output=False,
            output_activation="identity",
            output_block_hidden_channels=None,
        )

        with torch.no_grad():
            final_layer = model.output.layers[-1]
            final_layer.weight.fill_(10.0)
            final_layer.bias.fill_(10.0)

        result = model.output(
            torch.ones(
                2,
                4,
                8,
                8,
            )
        )

        assert torch.any(result > 1)
