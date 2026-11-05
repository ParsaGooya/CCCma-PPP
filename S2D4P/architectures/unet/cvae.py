import dataclasses
from dataclasses import field
from typing import ClassVar
import numpy as np
import math
import warnings
import torch
import torch.nn as nn

from S2D4P.core.selectors import cVAEArchSelector, deterministicArchSelector
from S2D4P.core.models.cvae import cVAEOutput

from S2D4P.configs import save_deterministic_guess_only

from S2D4P.architectures.architecture_abc import (
    cVAEArchConfigABC,
    cVAEArchABC,
    cVAEForwardRequest,
    cVAEPredictRequest,
    DeterministicRequest,
    GENERATORConfig,
)

from S2D4P.architectures.layers.generic import (
    InitMethod,
    UpsamplingMethod,
    MaskPoolingMethod,
    OutputActivation,
    NormalizationMethod,
    AlignmentMethod,
)
from S2D4P.architectures.layers.utils import (
    _broadcast_mask,
    _resize_tensor,
)

from S2D4P.architectures.layers.unet import (
    build_conv_block,
    UpBlock,
    DownBlock,
    UNetOutput,
    UNetOutputSIC,
)


from S2D4P.architectures.layers.conv import (
    ConvBlockConfig,
    PartialConvBlockConfig,
    ConvNeXtBlockConfig,
    TensorMask,
    LatentVector,
)

from S2D4P.architectures.unet.utils import _unet_config_checks, _repeat_tensor_mask
from S2D4P.architectures.unet.deterministic import UNetConfig


@dataclasses.dataclass
class UNetSelector(deterministicArchSelector):
    """
    Selector for a deterministic U-Net architecture used as a cVAE guess branch.

    Parameters
    ----------
    type : str
        Registered deterministic architecture type to select.
    share_output_block : bool
        Whether the cVAE and deterministic guess branches share the same output
        block.
    """

    type: str = "unet"
    share_output_block: bool = True


@cVAEArchSelector.register("unet")
@dataclasses.dataclass
class cVAEUNetConfig(cVAEArchConfigABC):
    """
    Configuration for a U-Net conditional variational autoencoder.

    Parameters
    ----------
    channels : list[int]
        Channel sizes used across the U-Net resolution levels.
    latent_size : int
        Size of the posterior latent representation.
    condition_embedding_channels : list | None
        Channel sizes used by the conditioning encoder. If ``None``, ``channels``
        is used.
    condition_embedding_size : int | None
        Size of the conditioning embedding. If ``None``, ``latent_size`` is used.
    latent_normalization : NormalizationMethod | None
        Normalization applied before projecting bottleneck features to latent
        parameters.
    condition_dependant_latent : bool
        Whether the latent prior depends on the conditioning input.
    condemb_to_decoder : bool
        Whether the conditioning embedding is concatenated with latent samples
        before generation.
    deterministic_guess_config : UNetSelector | None
        Optional deterministic U-Net configuration used to provide a baseline
        prediction that is added to the cVAE output.
    block_config : ConvBlockConfig | PartialConvBlockConfig | ConvNeXtBlockConfig
        Configuration used to construct convolutional blocks.
    upsampling_method : UpsamplingMethod
        Method used to increase decoder spatial resolution.
    upsampling_alignment_method : AlignmentMethod
        Method used to align decoder features with requested spatial shapes.
    transpose_kernel_sizes : list[int | tuple[int, int]] | int
        Kernel sizes used by successive upsampling blocks. A single integer is
        repeated across all upsampling levels.
    add_skip_latent : bool
        Whether latent information is configured to include skip-related
        behavior.
    mask_pooling : MaskPoolingMethod
        Method used for validity-mask pooling.
    mask_fraction_threshold : float
        Valid-fraction threshold used by fraction-based mask pooling.
    output_activation : OutputActivation
        Activation applied by the output block.
    output_block_hidden_channels : int | None
        Number of intermediate channels in the output block.
    init_method : InitMethod
        Parameter initialization method used when no checkpoint is loaded.
    GENERATOR : GENERATORConfig | None
        Optional generator configuration controlling stochastic output sampling
        and noise injection.
    """

    channels: list[int]
    latent_size: int
    condition_embedding_channels: list | None = None
    condition_embedding_size: int | None = None

    latent_normalization: NormalizationMethod | None = "layer"
    condition_dependant_latent: bool = False
    condemb_to_decoder: bool = True
    deterministic_guess_config: UNetSelector | None = None

    block_config: ConvBlockConfig | PartialConvBlockConfig | ConvNeXtBlockConfig = (
        field(default_factory=ConvBlockConfig)
    )

    upsampling_method: UpsamplingMethod = "bilinear"
    upsampling_alignment_method: AlignmentMethod = "padd"
    transpose_kernel_sizes: list[int | tuple[int, int]] | int = 3

    add_skip_latent: bool = False

    mask_pooling: MaskPoolingMethod = "any"
    mask_fraction_threshold: float = 0.5

    output_activation: OutputActivation = "identity"
    output_block_hidden_channels: int | None = None

    init_method: InitMethod = "default"
    GENERATOR: GENERATORConfig | None = None

    NUM_INPUT_DIMS: ClassVar[int] = 3
    NUM_OUTPUT_DIMS: ClassVar[int] = 3

    def __post_init__(self) -> None:
        """
        Resolve and validate the cVAE U-Net configuration.

        Common U-Net settings are validated, scalar upsampling kernel sizes are
        expanded across decoder levels, default conditioning encoder settings are
        resolved, and the optional deterministic guess configuration is prepared.
        """
        _unet_config_checks(self)
        n_up_blocks = len(self.channels) - 1

        if isinstance(self.transpose_kernel_sizes, int):
            self.transpose_kernel_sizes = [self.transpose_kernel_sizes] * n_up_blocks

        if self.condition_embedding_channels is None:
            self.condition_embedding_channels = self.channels

        if self.condition_embedding_size is None:
            self.condition_embedding_size = self.latent_size

        self._resolve_deterministic_guess()

    def _resolve_deterministic_guess(self):
        """
        Resolve and validate the optional deterministic guess architecture.

        The selected deterministic architecture is resolved to a U-Net
        configuration. Channel compatibility, generator settings, and shared
        output-block settings are then validated.

        Raises
        ------
        TypeError
            If the deterministic guess selector does not resolve to a ``UNetConfig``
            or one of its subclasses.
        ValueError
            If the deterministic guess has an incompatible output channel level,
            enables generator behavior, or has incompatible output-block settings
            when output-block sharing is enabled.
        """
        if self.deterministic_guess_config is not None:
            self.share_output_block = self.deterministic_guess_config.share_output_block
            self.freeze_deterministic = (
                self.deterministic_guess_config.checkpoint_config.freeze_weights
            )

            self.deterministic_guess_config = (
                self.deterministic_guess_config.get_architecture_config()
            )

            if not isinstance(self.deterministic_guess_config, UNetConfig):
                raise TypeError(
                    "cVAEUNet requires deterministic_guess_config to resolve to a "
                    "UNetConfig or one of its subclasses, got "
                    f"{type(self.deterministic_guess_config).__name__}."
                )

            if self.deterministic_guess_config.channels[0] != self.channels[0]:
                raise ValueError(
                    "The cVAE UNet model and the configured deterministic guess "
                    "model must have the same number of channels before output block "
                    f"for summation of the deterministic guess at that level. Expected : {self.channels[0]} "
                    f"got {self.deterministic_guess_config.channels[0]}"
                )

            if self.deterministic_guess_config.GENERATOR is not None:
                raise ValueError(
                    "The deterministic guess UNet model cannot have GENERATOR on. "
                )

            if self.share_output_block:
                if any(
                    [
                        (
                            self.output_activation
                            != self.deterministic_guess_config.output_activation
                        ),
                        (
                            self.output_block_hidden_channels
                            != self.deterministic_guess_config.output_block_hidden_channels
                        ),
                        (
                            getattr(self, "clip_output", None)
                            != getattr(
                                self.deterministic_guess_config, "clip_output", None
                            )
                        ),
                    ]
                ):
                    raise ValueError(
                        "With share_output_block=True, the cVAE and deterministic "
                        "guess must use compatible output-block settings "
                        "(activation, hidden channels, and clip_output)."
                    )

        else:
            self.share_output_block = False
            self.freeze_deterministic = False

    @property
    def EXPECTS_MASK(self) -> bool:
        """
        Indicate whether the architecture expects spatial validity masks.

        Returns
        -------
        bool
            ``True`` when partial convolution is used directly or through the
            configured ConvNeXt block, otherwise ``False``.
        """
        return isinstance(self.block_config, PartialConvBlockConfig) or getattr(
            self.block_config, "use_partial_conv", False
        )

    def build(
        self,
        input_shape: np.ndarray,
        output_shape: np.ndarray | None = None,
        added_features_dim: int | None = None,
    ):
        """
        Build a U-Net conditional variational autoencoder.

        Parameters
        ----------
        input_shape : np.ndarray
            Input shape in channel, height, and width order.
        output_shape : np.ndarray | None
            Output shape in channel, height, and width order. If ``None``, the input
            shape is used.
        added_features_dim : int | None
            Number of additional feature channels supplied with each sample.

        Returns
        -------
        cVAEUNet
            Constructed U-Net conditional variational autoencoder.
        """
        return cVAEUNet(
            config=self,
            input_shape=input_shape,
            output_shape=output_shape,
            added_features_dim=added_features_dim,
        )


class cVAEUNet(cVAEArchABC):
    """
    U-Net conditional variational autoencoder architecture.

    Uses separate recognition and conditioning encoders to construct latent
    representations, a decoder to generate spatial features, and an optional
    deterministic U-Net branch that provides a baseline prediction.

    Parameters
    ----------
    config : cVAEUNetConfig
        Configuration defining the cVAE U-Net architecture.
    input_shape : np.ndarray | tuple
        Input shape in channel, height, and width order.
    output_shape : np.ndarray | tuple | None
        Output shape in channel, height, and width order. If ``None``, the input
        shape is used.
    added_features_dim : int | None
        Number of additional feature channels supplied with each sample.
    """

    def __init__(
        self,
        config: cVAEUNetConfig,
        input_shape: np.ndarray | tuple,
        output_shape: np.ndarray | tuple | None = None,
        added_features_dim: int | None = None,
    ):
        """
        Initialize the U-Net conditional variational autoencoder.

        Constructs recognition and conditioning encoders, the latent generation
        decoder, the optional deterministic guess branch, and the output block.

        Parameters
        ----------
        config : cVAEUNetConfig
            Configuration defining the architecture.
        input_shape : np.ndarray | tuple
            Input shape in channel, height, and width order.
        output_shape : np.ndarray | tuple | None
            Output shape in channel, height, and width order. If ``None``, the input
            shape is used.
        added_features_dim : int | None
            Number of additional feature channels supplied with each sample.

        Raises
        ------
        RuntimeError
            If the input or output shape does not have the three dimensions required
            by the U-Net architecture.
        ValueError
            If a spatial dimension is nonpositive or the requested U-Net depth
            exceeds that supported by the input spatial resolution.
        """
        super().__init__(config)

        self.init_method = config.init_method
        self.added_features_dim = added_features_dim or 0
        self.latent_size = config.latent_size
        self.condition_embedding_channels = config.condition_embedding_channels
        self.condition_embedding_size = config.condition_embedding_size
        self.condition_dependant_latent = config.condition_dependant_latent
        self.condemb_to_decoder = config.condemb_to_decoder
        self.deterministic_guess_config = config.deterministic_guess_config
        self.share_output_block = config.share_output_block
        self.freeze_deterministic = config.freeze_deterministic
        self.deterministic_guess = None

        if output_shape is None:
            output_shape = input_shape

        if len(input_shape) != config.NUM_INPUT_DIMS:
            raise RuntimeError(
                f"UNet expects {config.NUM_INPUT_DIMS}D input shapes "
                f"(channel, height, width), got {input_shape}."
            )

        if len(output_shape) != config.NUM_OUTPUT_DIMS:
            raise RuntimeError(
                f"UNet expects {config.NUM_OUTPUT_DIMS}D output shapes "
                f"(channel, height, width), got {output_shape}."
            )

        min_spatial_size = int(min(input_shape[-2:]))

        if min_spatial_size < 1:
            raise ValueError(
                "UNet spatial dimensions must be positive, "
                f"got {tuple(input_shape[-2:])}."
            )

        n_down_blocks = len(config.channels) - 1
        max_down_blocks = math.floor(math.log2(min_spatial_size))

        if n_down_blocks > max_down_blocks:
            raise ValueError(
                "The requested UNet depth is too large for the input spatial "
                f"shape {tuple(input_shape[-2:])}. The configuration defines "
                f"{n_down_blocks} downsampling blocks, but the minimum spatial "
                f"dimension permits at most {max_down_blocks}. "
                f"Use no more than {max_down_blocks + 1} channel levels."
            )

        self._validate_checkpoint_compatibility(
            input_shape=input_shape,
            output_shape=output_shape,
        )

        self.input_shape = input_shape
        self.output_shape = output_shape

        output_channels = output_shape[0]
        output_spatial_shape = output_shape[1:]
        channels = config.channels

        recognition_input_channels = (
            output_shape[0] + input_shape[0] + self.added_features_dim
        )

        self.recognition = Recognition(
            input_channels=recognition_input_channels,
            input_spatial_shape=tuple(output_shape[-2:]),
            channels=channels,
            latent_size=self.latent_size,
            config=config,
        )

        condition_input_channels = input_shape[0] + self.added_features_dim
        get_log_var = (
            self.condition_dependant_latent and not self.condition_dependant_flow
        )

        self.condition = Recognition(
            input_channels=condition_input_channels,
            input_spatial_shape=tuple(input_shape[-2:]),
            channels=self.condition_embedding_channels,
            latent_size=self.condition_embedding_size,
            config=config,
            get_log_var=get_log_var,
        )

        self.upsampling_shapes = list(reversed(self.recognition.spatial_shapes))

        reversed_channels = list(reversed(channels))
        if self.condemb_to_decoder:
            self.add_condition_size = self.condition_embedding_size
        else:
            self.add_condition_size = 0
        generation_latent_size = self.latent_size + self.add_condition_size

        self.generation = Generation(
            latent_size=generation_latent_size,
            channels=reversed_channels,
            resize_shapes=self.upsampling_shapes,
            config=config,
        )

        if self.deterministic_guess_config is not None:
            self.deterministic_guess = self.deterministic_guess_config.build(
                input_shape=input_shape,
                output_shape=output_shape,
                added_features_dim=added_features_dim,
            )

        if self.share_output_block:
            self.output = self.deterministic_guess.output_block

        elif self.freeze_deterministic:
            for param in self.deterministic_guess.output_block.parameters():
                param.requires_grad = True

            self.output = self.deterministic_guess.output_block

        else:
            self.output = self._build_output(
                in_channels=reversed_channels[-1],
                out_channels=output_channels,
            )

        if config.checkpoint_config is not None:
            self._load_state_dict(config.checkpoint_config)
        else:
            self._initialize_weights(
                config.init_method, exclude=(self.deterministic_guess,)
            )

        self._predict_called: bool = False

    def _build_output(
        self,
        in_channels: int,
        out_channels: int,
    ) -> nn.Module:
        """
        Build the final U-Net output block.

        Parameters
        ----------
        in_channels : int
            Number of feature channels entering the output block.
        out_channels : int
            Number of prediction channels produced by the output block.

        Returns
        -------
        nn.Module
            Configured U-Net output projection.
        """
        return UNetOutput(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=self.config.output_block_hidden_channels,
            activation=self.config.output_activation,
        )

    def _prepare_input(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor | None,
        condition: torch.Tensor | None = None,
        condition_mask: torch.Tensor | None = None,
        added_features: torch.Tensor | None = None,
    ) -> TensorMask:
        """
        Prepare tensors and masks for a recognition or conditioning encoder.

        The primary mask is aligned with the input tensor. Optional conditioning data
        and additional features are resized to the input spatial resolution and
        concatenated along the channel dimension together with corresponding masks.

        Parameters
        ----------
        x : torch.Tensor
            Primary input tensor.
        x_mask : torch.Tensor | None
            Optional validity mask for the primary input.
        condition : torch.Tensor | None
            Optional conditioning tensor concatenated with the primary input.
        condition_mask : torch.Tensor | None
            Optional validity mask for ``condition``.
        added_features : torch.Tensor | None
            Optional additional feature maps concatenated with the input.

        Returns
        -------
        TensorMask
            Combined input tensor and corresponding validity mask.
        """
        x_mask = _broadcast_mask(x_mask, x)
        if condition is not None:
            condition = _resize_tensor(
                condition,
                x.shape[-2:],
            )

            x = torch.cat([x, condition], dim=1)

            if x_mask is not None:
                if condition_mask is not None:
                    condition_mask = _broadcast_mask(condition_mask, condition)
                else:
                    condition_mask = torch.ones_like(condition)

                x_mask = torch.cat(
                    [x_mask, condition_mask],
                    dim=1,
                )

        if added_features is not None:
            added_features = _resize_tensor(
                added_features,
                x.shape[-2:],
                mode="nearest",
            )

            feature_mask = (
                torch.ones_like(added_features) if x_mask is not None else None
            )

            x = torch.cat([x, added_features], dim=1)

            if x_mask is not None:
                x_mask = torch.cat(
                    [x_mask, feature_mask],
                    dim=1,
                )

        return TensorMask(tensor=x, mask=x_mask)

    def forward(self, request: cVAEForwardRequest) -> cVAEOutput:
        """
        Run the cVAE U-Net training forward pass.

        Posterior and conditioning latent parameters are inferred, posterior samples
        are drawn, decoder features are generated, and an optional deterministic
        guess is added before the output projection.

        Parameters
        ----------
        request : cVAEForwardRequest
            Forward request containing target and conditioning tensors, masks,
            additional features, latent and output sample counts, and optional
            posterior variance limits.

        Returns
        -------
        cVAEOutput
            Predictions together with posterior parameters, latent samples,
            conditional latent parameters, and deterministic-branch information.
        """
        x = request.target
        x_mask = request.target_mask
        condition = request.condition
        condition_mask = request.condition_mask
        added_features = request.added_features
        latent_sample_size = request.latent_sample_size
        posterior_variance_limits = request.posterior_variance_limits
        num_output_samples = request.output_sample_size

        batch_size = x.shape[0]

        if self.training and self.config.GENERATOR is not None:
            num_output_samples = self.config.GENERATOR.num_training_noise_samples

        cond_mu, cond_log_var = self._condition(
            condition=condition,
            condition_mask=condition_mask,
            added_features=added_features,
        )

        mu, log_var = self._recognition(
            x=x,
            x_mask=x_mask,
            condition=condition,
            condition_mask=condition_mask,
            added_features=added_features,
        )

        if posterior_variance_limits is not None:
            log_var = torch.clamp(
                log_var,
                min=posterior_variance_limits[0].type_as(mu),
                max=posterior_variance_limits[1].type_as(mu),
            )

        latent_samples = self._sample(mu, log_var, latent_sample_size)

        out = self._generate(
            latent_samples=latent_samples,
            condition_embedding=cond_mu,
            num_output_samples=num_output_samples,
        )

        if num_output_samples > 1:
            sample_sizes = (num_output_samples, latent_sample_size, batch_size)

        else:
            sample_sizes = (latent_sample_size, batch_size)

        deterministic_guess = self._deterministic_guess(
            input=condition,
            input_mask=condition_mask,
            added_features=added_features,
        )

        if deterministic_guess is not None:
            out = out + deterministic_guess

        output = self._output_block(out, sample_sizes)

        return cVAEOutput(
            output=output,
            mu=mu,
            log_var=log_var,
            samples=latent_samples,
            cond_mu=cond_mu,
            cond_log_var=cond_log_var,
        )

    def predict(
        self,
        request: cVAEPredictRequest,
        deterministic_guess_only: bool = save_deterministic_guess_only,
    ) -> cVAEOutput:
        """
        Generate predictions from the cVAE U-Net latent prior.

        Prior latent samples are decoded in optional chunks. An available
        deterministic guess is either added to the stochastic output or returned by
        itself when deterministic-only prediction is requested.

        Parameters
        ----------
        request : cVAEPredictRequest
            Prediction request containing conditioning data, masks, additional
            features, latent sampling settings, output sample count, and chunk size.
        deterministic_guess_only : bool
            Whether to return only the deterministic guess branch instead of adding it
            to the generated cVAE output.

        Returns
        -------
        cVAEOutput
            Generated predictions and conditional latent parameters.

        Warns
        -----
        UserWarning
            When deterministic-only prediction is enabled, indicating that only the
            deterministic guess branch is being saved.
        """
        num_output_samples = request.output_sample_size
        latent_sample_size = request.latent_sample_size
        chunk_size = request.chunk_size

        latent_samples, cond_mu, cond_log_var = self._sample_prior(request)
        batch_size = cond_mu.shape[0]

        out = self._generate(
            latent_samples,
            condition_embedding=cond_mu,
            num_output_samples=num_output_samples,
            chunk_size=chunk_size,
        )

        if deterministic_guess_only:
            latent_sample_size = 1
            num_output_samples = 1

        if num_output_samples > 1:
            sample_sizes = (num_output_samples, latent_sample_size, batch_size)

        else:
            sample_sizes = (latent_sample_size, batch_size)

        deterministic_guess = self._deterministic_guess(
            input=request.condition,
            input_mask=request.condition_mask,
            added_features=request.added_features,
        )

        if deterministic_guess is not None:
            if deterministic_guess_only:
                if not self._predict_called:
                    warnings.warn(
                        "======================================================\n"
                        "YOU ARE SAVING DETERMINISTIC GUESS BRANCH OUTPUT ONLY! \n"
                        "======================================================"
                    )
                out = deterministic_guess.unsqueeze(0)
            else:
                out = out + deterministic_guess

        output = self._output_block(out, sample_sizes)
        self._predict_called = True
        return cVAEOutput(
            output=output,
            mu=None,
            log_var=None,
            samples=None,
            cond_mu=cond_mu,
            cond_log_var=cond_log_var,
            deterministic_guess=deterministic_guess_only,
        )

    def _recognition(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor | None,
        condition: torch.Tensor = None,
        condition_mask: torch.Tensor = None,
        added_features: torch.Tensor = None,
    ) -> tuple[torch.Tensor]:
        """
        Infer posterior latent parameters from target and conditioning data.

        Parameters
        ----------
        x : torch.Tensor
            Target tensor used by the recognition encoder.
        x_mask : torch.Tensor | None
            Optional validity mask for the target.
        condition : torch.Tensor | None
            Optional conditioning tensor concatenated with the target.
        condition_mask : torch.Tensor | None
            Optional validity mask for the conditioning tensor.
        added_features : torch.Tensor | None
            Optional additional feature maps.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Posterior latent mean and log-variance.
        """
        input = self._prepare_input(
            x=x,
            x_mask=x_mask,
            condition=condition,
            condition_mask=condition_mask,
            added_features=added_features,
        )

        latent = self.recognition(input)

        return latent.mu, latent.log_var

    def _condition(
        self,
        condition: torch.Tensor,
        condition_mask: torch.Tensor = None,
        added_features: torch.Tensor = None,
    ) -> tuple[torch.Tensor]:
        """
        Encode conditioning data into latent conditioning parameters.

        Parameters
        ----------
        condition : torch.Tensor
            Conditioning tensor supplied to the conditioning encoder.
        condition_mask : torch.Tensor | None
            Optional validity mask for the conditioning tensor.
        added_features : torch.Tensor | None
            Optional additional feature maps.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None]
            Conditioning latent mean and optional log-variance.
        """
        input = self._prepare_input(
            x=condition,
            x_mask=condition_mask,
            added_features=added_features,
        )

        condition_embedding = self.condition(input)

        cond_mu = condition_embedding.mu
        cond_log_var = condition_embedding.log_var

        return cond_mu, cond_log_var

    def _generate(
        self,
        latent_samples: torch.Tensor,
        condition_embedding: torch.Tensor | None = None,
        num_output_samples: int = 1,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """
        Decode latent samples into spatial feature maps.

        The conditioning embedding is concatenated with latent samples when
        configured. Latent samples can be decoded in chunks to limit memory use, and
        multiple output samples can be produced for generator-enabled models.

        Parameters
        ----------
        latent_samples : torch.Tensor
            Latent samples with leading latent-sample and batch dimensions.
        condition_embedding : torch.Tensor | None
            Conditioning embedding optionally concatenated with each latent sample.
        num_output_samples : int
            Number of stochastic output samples generated for each latent sample.
        chunk_size : int | None
            Number of latent samples decoded per chunk. If ``None``, all latent
            samples are processed together.

        Returns
        -------
        torch.Tensor
            Generated spatial feature maps with latent-sample, batch, and optionally
            output-sample dimensions.
        """

        latent_sample_size, batch_size = latent_samples.shape[:-1]

        if all([condition_embedding is not None, self.condemb_to_decoder]):
            latent_samples = torch.cat(
                [
                    latent_samples,
                    condition_embedding.unsqueeze(0).expand(
                        latent_sample_size, *condition_embedding.shape
                    ),
                ],
                dim=-1,
            )

        feature_size = latent_samples.shape[-1]

        if chunk_size is None:
            chunk_size = latent_sample_size

        outputs = []

        for latent_chunk in latent_samples.split(chunk_size, dim=0):
            current_chunk_size = latent_chunk.shape[0]

            latent_chunk = latent_chunk.reshape(
                current_chunk_size * batch_size,
                feature_size,
            )

            out = self.generation(
                latent_chunk,
                num_output_samples,
            )

            if num_output_samples > 1:
                out = out.reshape(
                    num_output_samples,
                    current_chunk_size,
                    batch_size,
                    *out.shape[2:],
                )
            else:
                out = out.reshape(
                    current_chunk_size,
                    batch_size,
                    *out.shape[1:],
                )

            outputs.append(out)

        if num_output_samples > 1:
            return torch.cat(outputs, dim=1)

        return torch.cat(outputs, dim=0)

    def _deterministic_guess(
        self,
        input: torch.Tensor,
        input_mask: torch.Tensor | None = None,
        added_features: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """
        Generate the optional deterministic baseline prediction.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor supplied to the deterministic branch.
        input_mask : torch.Tensor | None
            Optional validity mask for the deterministic input.
        added_features : torch.Tensor | None
            Optional additional feature maps supplied to the deterministic branch.

        Returns
        -------
        torch.Tensor | None
            Deterministic decoder output, or ``None`` when no deterministic guess
            architecture is configured.
        """
        if self.deterministic_guess is None:
            return

        request = DeterministicRequest(
            input,
            input_mask,
            added_features,
        )

        return self.deterministic_guess.forward_decoder(request)

    def _output_block(
        self,
        input: torch.Tensor,
        sample_sizes: tuple[int, ...],
    ) -> torch.Tensor:
        """
        Apply the output block while preserving sample dimensions.

        Leading sample dimensions are flattened into the batch dimension before the
        output projection and restored afterward.

        Parameters
        ----------
        input : torch.Tensor
            Decoder features to project to model outputs.
        sample_sizes : tuple[int, ...]
            Leading sample dimensions to preserve across the output projection.

        Returns
        -------
        torch.Tensor
            Projected output with the supplied sample dimensions restored.
        """
        input = input.reshape(
            math.prod(sample_sizes),
            *input.shape[len(sample_sizes) :],
        )

        output = self.output(input)

        return output.reshape(
            *sample_sizes,
            *output.shape[1:],
        )


class Recognition(nn.Module):
    """
    Encode spatial inputs into latent distribution parameters.

    Features are mapped to the first channel level, progressively downsampled,
    and finally projected from the bottleneck representation to latent
    parameters.

    Parameters
    ----------
    input_channels : int
        Number of channels entering the encoder.
    input_spatial_shape : tuple[int, int]
        Height and width of the encoder input.
    channels : list[int]
        Channel sizes used across encoder resolution levels.
    latent_size : int
        Size of the latent representation.
    config : cVAEUNetConfig
        Configuration controlling convolutional blocks and latent projection.
    get_log_var : bool
        Whether the bottleneck produces latent log-variance in addition to the
        latent mean.
    """

    def __init__(
        self,
        input_channels: int,
        input_spatial_shape: tuple[int, int],
        channels: list[int],
        latent_size: int,
        config: cVAEUNetConfig,
        *,
        get_log_var: bool = True,
    ):
        """
        Initialize a spatial recognition encoder.

        Parameters
        ----------
        input_channels : int
            Number of channels entering the encoder.
        input_spatial_shape : tuple[int, int]
            Height and width of the input.
        channels : list[int]
            Channel sizes used across encoder levels.
        latent_size : int
            Size of the latent representation.
        config : cVAEUNetConfig
            Configuration controlling convolutional and latent blocks.
        get_log_var : bool
            Whether latent log-variance is produced.
        """
        super().__init__()

        self.initial_mapping = build_conv_block(
            input_channels,
            channels[0],
            config.block_config,
        )

        self.down_blocks = nn.ModuleList(
            [
                DownBlock(
                    channels[index],
                    channels[index + 1],
                    block_config=config.block_config,
                    return_skip=False,
                    mask_pooling=config.mask_pooling,
                    mask_fraction_threshold=config.mask_fraction_threshold,
                )
                for index in range(len(channels) - 1)
            ]
        )

        self.spatial_shapes = self._get_spatial_shapes(input_spatial_shape)

        bottleneck_spatial_shape = self.spatial_shapes[-1]
        bottleneck_output_shape = (
            channels[-1],
            *bottleneck_spatial_shape,
        )

        self.bottleneck = build_conv_block(
            channels[-1],
            channels[-1],
            config.block_config,
            latent_size=latent_size,
            block_output_shape=bottleneck_output_shape,
            get_log_var=get_log_var,
            latent_normalization=config.latent_normalization,
        )

    def _get_spatial_shapes(
        self,
        input_shape: tuple[int, int],
    ) -> list[tuple[int, int]]:
        """
        Calculate spatial shapes across encoder resolution levels.

        Parameters
        ----------
        input_shape : tuple[int, int]
            Initial input height and width.

        Returns
        -------
        list[tuple[int, int]]
            Spatial shape at the input level and after each downsampling block.
        """
        shape = tuple(input_shape)
        shapes = [shape]

        for block in self.down_blocks:
            shape = block.output_shape(shape)
            shapes.append(shape)

        return shapes

    def forward(
        self,
        input: TensorMask,
    ) -> LatentVector:
        """
        Encode an input into latent parameters.

        Parameters
        ----------
        input : TensorMask
            Input feature tensor and optional validity mask.

        Returns
        -------
        LatentVector
            Latent mean and optional log-variance produced at the encoder bottleneck.
        """
        input = self.initial_mapping(input)

        for down_block in self.down_blocks:
            input = down_block(input)

        return self.bottleneck(input)


class Generation(nn.Module):
    """
    Decode latent vectors into spatial feature maps.

    Latent vectors are projected to a spatial bottleneck and passed through
    successive upsampling blocks. Generator-enabled configurations can replicate
    the bottleneck to produce multiple stochastic outputs.

    Parameters
    ----------
    latent_size : int
        Number of latent features entering the decoder.
    channels : list[int]
        Decoder channel sizes from bottleneck to highest spatial resolution.
    resize_shapes : list[tuple]
        Spatial shapes targeted across successive decoder levels.
    config : cVAEUNetConfig
        Configuration controlling upsampling and generator noise behavior.
    """

    def __init__(
        self,
        latent_size: int,
        channels: list[int],
        resize_shapes: list[tuple],
        config: cVAEUNetConfig,
    ):
        """
        Initialize the latent generation decoder.

        Parameters
        ----------
        latent_size : int
            Number of latent features entering the decoder.
        channels : list[int]
            Channel sizes across decoder resolution levels.
        resize_shapes : list[tuple]
            Spatial shapes targeted by the bottleneck and subsequent upsampling
            blocks.
        config : cVAEUNetConfig
            Configuration controlling upsampling and generator noise injection.
        """
        super().__init__()
        self.bottleneck_dim = channels[0]
        self.bottleneck_shape = resize_shapes[0]
        self.resize_shapes = resize_shapes[1:]
        self.config = config

        self.combine_latent = nn.Linear(
            latent_size, self.bottleneck_dim * np.prod(self.bottleneck_shape)
        )

        input_channels = self.bottleneck_dim
        generator_enabled = config.GENERATOR is not None
        inject_noise_in_block = (
            generator_enabled and config.GENERATOR.noise_level != "low"
        )

        up_blocks: list[nn.Module] = []
        for index, out_channels in enumerate(channels[1:]):
            inject_noise = generator_enabled and (
                config.GENERATOR.noise_level != "medium"
                or index == len(self.resize_shapes) - 1
            )

            up_blocks.append(
                UpBlock(
                    input_channels=input_channels,
                    skip_channels=None,
                    out_channels=out_channels,
                    block_config=config.block_config,
                    upsampling_method=config.upsampling_method,
                    skip_alignment_method=config.upsampling_alignment_method,
                    upsampling_kernel_size=config.transpose_kernel_sizes[index],
                    inject_noise=inject_noise,
                    inject_noise_in_block=inject_noise_in_block,
                )
            )
            input_channels = out_channels

        self.up_blocks = nn.ModuleList(up_blocks)

    def forward(
        self,
        latent_samples: torch.Tensor,
        num_output_samples: int = 1,
    ) -> torch.Tensor:
        """
        Decode latent vectors into spatial feature maps.

        Parameters
        ----------
        latent_samples : torch.Tensor
            Latent vectors with the batch represented by the leading dimension.
        num_output_samples : int
            Number of stochastic output samples generated for each latent input when
            generator behavior is enabled.

        Returns
        -------
        torch.Tensor
            Decoded spatial features, with a leading output-sample dimension when
            multiple generator samples are requested.
        """
        batch_size = latent_samples.shape[0]
        x = self.combine_latent(latent_samples)
        x = x.reshape(-1, self.bottleneck_dim, *self.bottleneck_shape)

        input = TensorMask(tensor=x, mask=None)

        if self.config.GENERATOR is not None and num_output_samples > 1:
            input = _repeat_tensor_mask(
                input,
                repeats=num_output_samples,
            )

        for up_block, resize_shape in zip(
            self.up_blocks,
            self.resize_shapes,
            strict=True,
        ):
            input = up_block(input, skip=None, resize_shape=resize_shape)

        output = input.tensor
        if self.config.GENERATOR is not None and num_output_samples > 1:
            output = output.reshape(
                batch_size,
                num_output_samples,
                *output.shape[1:],
            ).transpose(0, 1)

        return output


@cVAEArchSelector.register("unetsic")
@dataclasses.dataclass
class cVAEUNetSICEConfig(cVAEUNetConfig):
    """
    Configuration for a sea-ice-concentration cVAE U-Net.

    Extends the standard cVAE U-Net configuration with optional clipping of the
    final output to the valid sea-ice-concentration range.

    Parameters
    ----------
    clip_output : bool
        Whether final predictions are clipped to the interval [0, 1].
    """

    clip_output: bool = False

    def build(
        self,
        input_shape: np.ndarray,
        output_shape: np.ndarray | None = None,
        added_features_dim: int | None = None,
    ):
        """
        Build a sea-ice-concentration cVAE U-Net.

        Parameters
        ----------
        input_shape : np.ndarray
            Input shape in channel, height, and width order.
        output_shape : np.ndarray | None
            Output shape in channel, height, and width order. If ``None``, the input
            shape is used.
        added_features_dim : int | None
            Number of additional feature channels supplied with each sample.

        Returns
        -------
        cVAEUNetSIC
            Constructed sea-ice-concentration cVAE U-Net.
        """
        return cVAEUNetSIC(
            config=self,
            input_shape=input_shape,
            output_shape=output_shape,
            added_features_dim=added_features_dim,
        )


class cVAEUNetSIC(cVAEUNet):
    """
    Sea-ice-concentration specialization of the cVAE U-Net.

    Uses the standard cVAE U-Net architecture with a sea-ice-specific output block
    that can constrain predictions to the interval [0, 1].
    """

    def _build_output(
        self,
        in_channels: int,
        out_channels: int,
    ) -> nn.Module:
        """
        Build the sea-ice-concentration output block.

        Parameters
        ----------
        in_channels : int
            Number of decoder feature channels entering the output block.
        out_channels : int
            Number of prediction channels produced by the output block.

        Returns
        -------
        nn.Module
            Sea-ice-concentration output projection with the configured activation
            and optional output clipping.
        """
        return UNetOutputSIC(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=self.config.output_block_hidden_channels,
            activation=self.config.output_activation,
            clip_output=self.config.clip_output,
        )
