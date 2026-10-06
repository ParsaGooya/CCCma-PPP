import dataclasses
from dataclasses import field
from typing import ClassVar
import numpy as np
import math
import warnings
import torch
import torch.nn as nn

from cccma_ppp.core.selectors import cVAEModelSelector, deterministicModelSelector
from cccma_ppp.core.models.cvae import cVAEOutput

from cccma_ppp.configs import save_deterministic_guess_only

from cccma_ppp.architectures.models_abc import (
    cVAEmodelConfigABC,
    cVAEmodelsABC,
    cVAEForwardRequest,
    cVAEPredictRequest,
    DeterministicRequest,
    GENERATORConfig,
)

from cccma_ppp.architectures.layers.generic import (
    InitMethod,
    UpsamplingMethod,
    MaskPoolingMethod,
    OutputActivation,
    NormalizationMethod,
    AlignmentMethod,
)
from cccma_ppp.architectures.layers.utils import (
    _broadcast_mask,
    _resize_tensor,
)

from cccma_ppp.architectures.layers.unet import (
    build_conv_block,
    UpBlock,
    DownBlock,
    UNetOutput,
    UNetOutputSIC,
)


from cccma_ppp.architectures.layers.conv import (
    ConvBlockConfig,
    PartialConvBlockConfig,
    ConvNeXtBlockConfig,
    TensorMask,
    LatentVector,
)

from cccma_ppp.architectures.unet.utils import _unet_config_checks, _repeat_tensor_mask
from cccma_ppp.architectures.unet.deterministic import UNetConfig


@dataclasses.dataclass
class UNetSelector(deterministicModelSelector):
    """
    Selector configuration for a deterministic U-Net used as a cVAE baseline.

    Parameters
    ----------
    type : str
        Model type used to resolve the deterministic U-Net configuration.
    share_output_block : bool
        Whether the cVAE and deterministic model share the same output block.
    """

    type: str = "unet"
    share_output_block: bool = True


@cVAEModelSelector.register("unet")
@dataclasses.dataclass
class cVAEUNetConfig(cVAEmodelConfigABC):
    """
    Configuration for a convolutional U-Net conditional variational autoencoder.

    Parameters
    ----------
    channels : list[int]
        Number of feature channels at each level of the recognition U-Net.
    latent_size : int
        Size of the latent representation.
    condition_embedding_channels : list | None
        Feature channels used by the condition encoder. If ``None``, ``channels``
        is used.
    condition_embedding_size : int | None
        Size of the encoded conditioning representation. If ``None``,
        ``latent_size`` is used.
    latent_normalization : NormalizationMethod | None
        Normalization applied before projecting bottleneck features into latent
        space. If ``None``, latent normalization is disabled.
    condition_dependant_latent : bool
        Whether the latent prior depends on the conditioning input.
    condemb_to_decoder : bool
        Whether to concatenate the condition embedding with latent samples before
        generation.
    deterministic_guess_config : UNetSelector | None
        Configuration for an optional deterministic U-Net whose output is added
        to the generated cVAE output.
    block_config : ConvBlockConfig | PartialConvBlockConfig | ConvNeXtBlockConfig
        Configuration defining the convolutional blocks used throughout the U-Net.
    upsampling_method : UpsamplingMethod
        Method used to increase spatial resolution in the generation network.
    upsampling_alignment_method : AlignmentMethod
        Method used to align generated feature maps with target spatial shapes.
    transpose_kernel_sizes : list[int | tuple[int, int]] | int
        Kernel sizes used by successive upsampling blocks. A single integer is
        repeated for all upsampling levels.
    add_skip_latent : bool
        Whether latent skip information is enabled in the U-Net configuration.
    mask_pooling : MaskPoolingMethod
        Method configured for validity-mask downsampling.
    mask_fraction_threshold : float
        Minimum valid fraction used when fractional mask pooling is selected.
    output_activation : OutputActivation
        Activation applied by the final output block.
    output_block_hidden_channels : int | None
        Number of hidden channels in the output block. If ``None``, no hidden
        output stage is used.
    init_method : InitMethod
        Weight initialization method used when no checkpoint is loaded.
    GENERATOR : GENERATORConfig | None
        Optional configuration controlling generator noise injection and output
        sampling.
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
        Validate and complete the cVAE U-Net configuration.

        A scalar upsampling kernel size is expanded across all upsampling blocks,
        missing condition-encoder settings are inferred from the main U-Net
        configuration, and the optional deterministic guess configuration is
        resolved.
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
        Resolve and validate the optional deterministic guess configuration.

        The deterministic model configuration is resolved to a U-Net configuration
        and checked for compatibility with the cVAE, including channel counts,
        generator settings, and shared output-block settings.

        Raises
        ------
        TypeError
            If the deterministic guess configuration does not resolve to a
            ``UNetConfig`` or one of its subclasses.
        ValueError
            If the deterministic and cVAE U-Nets have incompatible channel counts,
            if the deterministic model enables generator behavior, or if a shared
            output block is requested with incompatible output settings.
        """

        if self.deterministic_guess_config is not None:
            self.share_output_block = self.deterministic_guess_config.share_output_block
            self.freeze_deterministic = (
                self.deterministic_guess_config.checkpoint_config.freeze_weights
            )

            self.deterministic_guess_config = (
                self.deterministic_guess_config.get_model_config()
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
        Indicate whether the configured U-Net uses validity masks.

        Returns
        -------
        bool
            ``True`` when partial convolution is enabled by the block configuration,
            otherwise ``False``.
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
        Build a cVAE U-Net from this configuration.

        Parameters
        ----------
        input_shape : np.ndarray
            Shape ``(C, H, W)`` of the conditioning input.
        output_shape : np.ndarray | None
            Shape ``(C, H, W)`` of the model output. If ``None``, ``input_shape`` is
            used.
        added_features_dim : int | None
            Number of additional feature channels supplied to the model.

        Returns
        -------
        cVAEUNet
            Constructed conditional variational U-Net.
        """
        return cVAEUNet(
            config=self,
            input_shape=input_shape,
            output_shape=output_shape,
            added_features_dim=added_features_dim,
        )


class cVAEUNet(cVAEmodelsABC):
    """
    Conditional variational autoencoder with convolutional U-Net components.

    Separate recognition networks encode the posterior and conditioning inputs
    into latent representations. Latent samples are decoded through an
    upsampling network and may optionally be combined with a deterministic U-Net
    prediction.

    Parameters
    ----------
    config : cVAEUNetConfig
        Configuration defining the U-Net, latent representation, generation
        network, and optional deterministic guess.
    input_shape : np.ndarray | tuple
        Shape ``(C, H, W)`` of the conditioning input.
    output_shape : np.ndarray | tuple | None
        Shape ``(C, H, W)`` of the model output. If ``None``, ``input_shape`` is
        used.
    added_features_dim : int | None
        Number of additional feature channels supplied to the model.
    """

    def __init__(
        self,
        config: cVAEUNetConfig,
        input_shape: np.ndarray | tuple,
        output_shape: np.ndarray | tuple | None = None,
        added_features_dim: int | None = None,
    ):
        """
        Initialize a conditional variational U-Net.

        Parameters
        ----------
        config : cVAEUNetConfig
            Configuration defining the recognition, conditioning, generation, output,
            and optional deterministic components.
        input_shape : np.ndarray | tuple
            Shape ``(C, H, W)`` of the conditioning input.
        output_shape : np.ndarray | tuple | None
            Shape ``(C, H, W)`` of the model output. If ``None``, ``input_shape`` is
            used.
        added_features_dim : int | None
            Number of additional feature channels supplied to the model.

        Raises
        ------
        RuntimeError
            If the input or output shape does not have the required three dimensions.
        ValueError
            If a spatial input dimension is nonpositive or the configured U-Net depth
            exceeds that supported by the input spatial dimensions.
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
        Build the final U-Net output projection.

        Parameters
        ----------
        in_channels : int
            Number of channels in the generated feature tensor.
        out_channels : int
            Number of channels required in the model output.

        Returns
        -------
        nn.Module
            Output projection configured with the requested hidden channels and
            activation.
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
        Prepare tensors and masks for a recognition network.

        The primary tensor can be combined with a spatially resized conditioning
        tensor and additional feature channels. Associated masks are broadcast,
        resized, and extended with fully valid channels as needed.

        Parameters
        ----------
        x : torch.Tensor
            Primary input tensor.
        x_mask : torch.Tensor | None
            Validity mask associated with the primary input.
        condition : torch.Tensor | None
            Optional conditioning tensor to resize and concatenate with the input.
        condition_mask : torch.Tensor | None
            Optional validity mask associated with the conditioning tensor.
        added_features : torch.Tensor | None
            Optional feature tensor to resize and concatenate with the input.

        Returns
        -------
        TensorMask
            Combined input tensor and its corresponding validity mask.
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
        Perform a training forward pass through the conditional variational U-Net.

        The conditioning and target inputs are encoded, posterior latent samples are
        drawn, and the samples are decoded into output features. An optional
        deterministic prediction is added before the final output projection.

        Parameters
        ----------
        request : cVAEForwardRequest
            Forward-pass request containing targets, conditioning inputs, masks,
            optional additional features, and sampling parameters.

        Returns
        -------
        cVAEOutput
            Generated output together with posterior latent parameters, latent
            samples, and condition-distribution parameters.
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
        Generate predictions from samples drawn from the conditional prior.

        Generation can be performed in chunks and may produce multiple stochastic
        outputs for each latent sample. When configured, a deterministic U-Net
        prediction is either added to the stochastic output or returned by itself.

        Parameters
        ----------
        request : cVAEPredictRequest
            Prediction request containing conditioning data, masks, optional
            additional features, and sampling parameters.
        deterministic_guess_only : bool
            Whether to return only the deterministic guess when a deterministic
            branch is configured.

        Returns
        -------
        cVAEOutput
            Generated predictions and condition-distribution parameters.

        Warns
        -----
        UserWarning
            If deterministic-only output is requested and the deterministic branch
            is used, on the first prediction call.
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

        The target, conditioning input, masks, and optional additional features are
        prepared and passed through the posterior recognition network.

        Parameters
        ----------
        x : torch.Tensor
            Target tensor used to infer the posterior distribution.
        x_mask : torch.Tensor | None
            Validity mask associated with the target tensor.
        condition : torch.Tensor | None
            Conditioning tensor concatenated with the target before encoding.
        condition_mask : torch.Tensor | None
            Validity mask associated with the conditioning tensor.
        added_features : torch.Tensor | None
            Additional feature channels concatenated with the recognition input.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None]
            Posterior mean and log-variance tensors.
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
        Encode conditioning data into a latent conditioning representation.

        Parameters
        ----------
        condition : torch.Tensor
            Conditioning tensor to encode.
        condition_mask : torch.Tensor | None
            Validity mask associated with the conditioning tensor.
        added_features : torch.Tensor | None
            Additional feature channels concatenated with the conditioning input.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None]
            Condition mean and, when configured, condition log-variance tensors.
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

        When configured, the condition embedding is concatenated with each latent
        sample before generation. Latent samples may be processed in chunks to limit
        the size of individual generation operations.

        Parameters
        ----------
        latent_samples : torch.Tensor
            Latent samples with leading dimensions corresponding to latent sample and
            batch.
        condition_embedding : torch.Tensor | None
            Encoded conditioning representation to concatenate with latent samples
            when enabled.
        num_output_samples : int
            Number of stochastic generation outputs to produce for each latent sample.
        chunk_size : int | None
            Maximum number of latent samples processed in each generation chunk. If
            ``None``, all latent samples are processed together.

        Returns
        -------
        torch.Tensor
            Generated feature maps organized by latent sample and batch, with an
            additional leading output-sample dimension when ``num_output_samples`` is
            greater than one.
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
        Generate the optional deterministic U-Net prediction.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor supplied to the deterministic model.
        input_mask : torch.Tensor | None
            Validity mask associated with the input tensor.
        added_features : torch.Tensor | None
            Additional features supplied to the deterministic model.

        Returns
        -------
        torch.Tensor | None
            Deterministic decoder output, or ``None`` if no deterministic guess model
            is configured.
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
        Apply the output block while preserving sampling dimensions.

        Leading sample dimensions are temporarily merged into the batch dimension
        before applying the output projection and restored afterward.

        Parameters
        ----------
        input : torch.Tensor
            Generated feature tensor.
        sample_sizes : tuple[int, ...]
            Sizes of the leading sample and batch dimensions to preserve.

        Returns
        -------
        torch.Tensor
            Projected output with the leading sampling dimensions restored.
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
    Encode spatial features into a latent representation.

    The recognition network maps the input to the first feature level, repeatedly
    downsamples it through convolutional blocks, and projects the bottleneck
    features into latent parameters.

    Parameters
    ----------
    input_channels : int
        Number of channels in the recognition input.
    input_spatial_shape : tuple[int, int]
        Spatial shape ``(H, W)`` of the recognition input.
    channels : list[int]
        Number of feature channels at successive resolution levels.
    latent_size : int
        Size of the resulting latent representation.
    config : cVAEUNetConfig
        Configuration defining convolutional blocks, mask handling, and latent
        normalization.
    get_log_var : bool
        Whether to produce latent log-variance values in addition to means.
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
        Initialize a U-Net recognition network.

        Parameters
        ----------
        input_channels : int
            Number of channels in the recognition input.
        input_spatial_shape : tuple[int, int]
            Spatial shape ``(H, W)`` of the recognition input.
        channels : list[int]
            Number of feature channels at successive resolution levels.
        latent_size : int
            Size of the resulting latent representation.
        config : cVAEUNetConfig
            Configuration used to construct convolutional, downsampling, and latent
            projection blocks.
        get_log_var : bool
            Whether the bottleneck produces latent log-variance values in addition
            to means.
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
        Compute the spatial shape at each recognition-network level.

        Parameters
        ----------
        input_shape : tuple[int, int]
            Initial spatial shape ``(H, W)``.

        Returns
        -------
        list[tuple[int, int]]
            Spatial shapes from the input resolution through each successive
            downsampling level.
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
        Encode an input tensor and mask into latent parameters.

        Parameters
        ----------
        input : TensorMask
            Input tensor and its associated validity mask.

        Returns
        -------
        LatentVector
            Latent mean and, when enabled, log-variance tensors.
        """
        input = self.initial_mapping(input)

        for down_block in self.down_blocks:
            input = down_block(input)

        return self.bottleneck(input)


class Generation(nn.Module):
    """
    Decode latent representations into spatial feature maps.

    Latent vectors are projected to bottleneck feature maps and progressively
    upsampled to the requested spatial resolutions. Optional generator settings
    control noise injection and repeated stochastic outputs.

    Parameters
    ----------
    latent_size : int
        Number of features in each latent representation.
    channels : list[int]
        Number of feature channels from the bottleneck through successive
        upsampling levels.
    resize_shapes : list[tuple]
        Spatial shapes used at the bottleneck and successive upsampling levels.
    config : cVAEUNetConfig
        Configuration defining upsampling, convolutional processing, and optional
        generator noise injection.
    """

    def __init__(
        self,
        latent_size: int,
        channels: list[int],
        resize_shapes: list[tuple],
        config: cVAEUNetConfig,
    ):
        """
        Initialize a latent-to-spatial generation network.

        Parameters
        ----------
        latent_size : int
            Number of features in each latent representation.
        channels : list[int]
            Number of feature channels from the bottleneck through successive
            upsampling levels.
        resize_shapes : list[tuple]
            Spatial shapes used for the bottleneck and successive generated feature
            maps.
        config : cVAEUNetConfig
            Configuration defining upsampling behavior, convolutional blocks, and
            optional generator noise injection.
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
        Generate spatial feature maps from latent representations.

        Latent vectors are projected to the bottleneck resolution and passed through
        successive upsampling blocks. When generator sampling is enabled, bottleneck
        features can be repeated to produce multiple stochastic outputs.

        Parameters
        ----------
        latent_samples : torch.Tensor
            Latent representations with shape ``(N, latent_size)``.
        num_output_samples : int
            Number of stochastic outputs to generate for each latent representation.

        Returns
        -------
        torch.Tensor
            Generated spatial feature maps. When multiple generator outputs are
            requested, the first dimension indexes output samples.
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


@cVAEModelSelector.register("unetsic")
@dataclasses.dataclass
class cVAEUNetSICEConfig(cVAEUNetConfig):
    """
    Configuration for the sea ice concentration cVAE U-Net.

    Extends the standard cVAE U-Net configuration with optional clipping of the
    final output.

    Parameters
    ----------
    clip_output : bool
        Whether to clamp final output values to the interval [0, 1].
    """

    clip_output: bool = False

    def build(
        self,
        input_shape: np.ndarray,
        output_shape: np.ndarray | None = None,
        added_features_dim: int | None = None,
    ):
        """
        Build a sea ice concentration cVAE U-Net from this configuration.

        Parameters
        ----------
        input_shape : np.ndarray
            Shape ``(C, H, W)`` of the conditioning input.
        output_shape : np.ndarray | None
            Shape ``(C, H, W)`` of the model output. If ``None``, ``input_shape`` is
            used.
        added_features_dim : int | None
            Number of additional feature channels supplied to the model.

        Returns
        -------
        cVAEUNetSIC
            Constructed sea ice concentration conditional variational U-Net.
        """
        return cVAEUNetSIC(
            config=self,
            input_shape=input_shape,
            output_shape=output_shape,
            added_features_dim=added_features_dim,
        )


class cVAEUNetSIC(cVAEUNet):
    """
    Conditional variational U-Net specialized for sea ice concentration output.

    Uses the standard cVAE U-Net architecture with a sea ice concentration output
    block that supports optional clipping to the interval [0, 1].
    """

    def _build_output(
        self,
        in_channels: int,
        out_channels: int,
    ) -> nn.Module:
        """
        Build the sea ice concentration output projection.

        Parameters
        ----------
        in_channels : int
            Number of channels in the generated feature tensor.
        out_channels : int
            Number of channels required in the model output.

        Returns
        -------
        nn.Module
            Sea ice concentration output block configured with the requested hidden
            channels, activation, and clipping behavior.
        """
        return UNetOutputSIC(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=self.config.output_block_hidden_channels,
            activation=self.config.output_activation,
            clip_output=self.config.clip_output,
        )
