import dataclasses
from dataclasses import field
from typing import ClassVar
import numpy as np
import math
import torch
import torch.nn as nn


from cccma_ppp.core.selectors import deterministicModelSelector
from cccma_ppp.core.models.deterministic import deterministicOutput

from cccma_ppp.architectures.models_abc import (
    deterministicmodelsABC,
    modelConfigABC,
    DeterministicRequest,
    GENERATORConfig,
)

from cccma_ppp.architectures.layers.generic import (
    InitMethod,
    UpsamplingMethod,
    MaskPoolingMethod,
    OutputActivation,
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
)


from cccma_ppp.architectures.unet.utils import _unet_config_checks, _repeat_tensor_mask


@deterministicModelSelector.register("unet")
@dataclasses.dataclass
class UNetConfig(modelConfigABC):
    """
    Configuration for a deterministic convolutional U-Net.

    Parameters
    ----------
    channels : list[int]
        Number of feature channels at each resolution level of the U-Net.
    block_config : ConvBlockConfig | PartialConvBlockConfig | ConvNeXtBlockConfig
        Configuration defining the convolutional blocks used throughout the U-Net.
    upsampling_method : UpsamplingMethod
        Method used to increase spatial resolution in the decoder.
    skip_alignment_method : AlignmentMethod
        Method used to spatially align decoder features with skip connections or
        target feature-map sizes.
    transpose_kernel_sizes : list[int | tuple[int, int]] | int
        Kernel sizes used by successive upsampling blocks. A single integer is
        repeated for all upsampling levels.
    add_skip_connections : bool
        Whether encoder features are passed to corresponding decoder levels as
        skip connections.
    process_skip_connections : bool
        Whether skip connections are processed by an additional convolutional
        block before being passed to the decoder.
    mask_pooling : MaskPoolingMethod
        Mask pooling method associated with downsampling.
    mask_fraction_threshold : float
        Minimum valid fraction associated with fractional mask pooling.
    output_activation : OutputActivation
        Activation applied by the final output block.
    output_block_hidden_channels : int | None
        Number of hidden channels in the output block. If ``None``, the final
        feature map is projected directly to the output channels.
    init_method : InitMethod
        Weight initialization method used when no checkpoint is loaded.
    GENERATOR : GENERATORConfig | None
        Optional configuration controlling noise injection and stochastic output
        sampling.
    """

    channels: list[int]
    block_config: ConvBlockConfig | PartialConvBlockConfig | ConvNeXtBlockConfig = (
        field(default_factory=ConvBlockConfig)
    )

    upsampling_method: UpsamplingMethod = "bilinear"
    skip_alignment_method: AlignmentMethod = "padd"
    transpose_kernel_sizes: list[int | tuple[int, int]] | int = 3

    add_skip_connections: bool = True
    process_skip_connections: bool = False

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
        Validate and complete the U-Net configuration.

        A scalar upsampling kernel size is expanded to provide one kernel size for
        each upsampling block.
        """
        _unet_config_checks(self)

        n_up_blocks = len(self.channels) - 1

        if isinstance(self.transpose_kernel_sizes, int):
            self.transpose_kernel_sizes = [self.transpose_kernel_sizes] * n_up_blocks

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
        Build a deterministic U-Net from this configuration.

        Parameters
        ----------
        input_shape : np.ndarray
            Shape ``(C, H, W)`` of the model input.
        output_shape : np.ndarray | None
            Shape ``(C, H, W)`` of the model output. If ``None``, ``input_shape`` is
            used.
        added_features_dim : int | None
            Number of additional feature channels supplied to the model.

        Returns
        -------
        UNet
            Constructed deterministic U-Net.
        """
        return UNet(
            config=self,
            input_shape=input_shape,
            output_shape=output_shape,
            added_features_dim=added_features_dim,
        )


class UNet(deterministicmodelsABC):
    """
    Deterministic convolutional U-Net.

    The model progressively downsamples an input into bottleneck features and
    then reconstructs spatial features through a sequence of upsampling blocks.
    Encoder features can optionally be passed to the decoder through skip
    connections.

    Parameters
    ----------
    config : UNetConfig
        Configuration defining convolutional blocks, downsampling, upsampling,
        skip connections, output processing, and optional generator behavior.
    input_shape : np.ndarray | tuple
        Shape ``(C, H, W)`` of the model input.
    output_shape : np.ndarray | tuple | None
        Shape ``(C, H, W)`` of the model output. If ``None``, ``input_shape`` is
        used.
    added_features_dim : int | None
        Number of additional feature channels supplied to the model.
    """

    def __init__(
        self,
        config: UNetConfig,
        input_shape: np.ndarray | tuple,
        output_shape: np.ndarray | tuple | None = None,
        added_features_dim: int | None = None,
    ):
        """
        Initialize a deterministic convolutional U-Net.

        Parameters
        ----------
        config : UNetConfig
            Configuration defining the encoder, bottleneck, decoder, output block,
            and optional generator behavior.
        input_shape : np.ndarray | tuple
            Shape ``(C, H, W)`` of the model input.
        output_shape : np.ndarray | tuple | None
            Shape ``(C, H, W)`` of the model output. If ``None``, ``input_shape`` is
            used.
        added_features_dim : int | None
            Number of additional feature channels concatenated with the model input.

        Raises
        ------
        RuntimeError
            If the input or output shape does not have the required three dimensions,
            or if the input and output spatial dimensions differ without a hidden
            output block configured for post-interpolation processing.
        ValueError
            If a spatial input dimension is nonpositive or the configured U-Net depth
            exceeds that supported by the input spatial dimensions.
        """
        super().__init__(config)

        self.spatial_shapes = None
        self.add_skip_connections = config.add_skip_connections
        self.init_method = config.init_method
        self.added_features_dim = added_features_dim or 0

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

        if not np.array_equal(input_shape[-2:], output_shape[-2:]):
            if self.config.output_block_hidden_channels is None:
                raise RuntimeError(
                    f"Input spatial shape {input_shape[-2:]} does not match "
                    f"output spatial shape {output_shape[-2:]}. Output block "
                    "needs output_block_hidden_channels to process output after "
                    "interpolation."
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

        input_channels = input_shape[0] + self.added_features_dim
        output_channels = output_shape[0]

        channels = config.channels

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
                    return_skip=config.add_skip_connections,
                    process_skip_connections=config.process_skip_connections,
                    mask_pooling=config.mask_pooling,
                    mask_fraction_threshold=config.mask_fraction_threshold,
                )
                for index in range(len(channels) - 1)
            ]
        )

        shape = tuple(input_shape[-2:])
        self.spatial_shapes = [shape]

        for block in self.down_blocks:
            shape = block.output_shape(shape)
            self.spatial_shapes.append(shape)

        self.bottleneck = build_conv_block(
            channels[-1],
            channels[-1],
            config.block_config,
        )

        reversed_skips = list(reversed(channels[:-1]))
        input_channels = channels[-1]
        generator_enabled = config.GENERATOR is not None
        inject_noise_in_block = (
            generator_enabled and config.GENERATOR.noise_level != "low"
        )

        up_blocks: list[nn.Module] = []
        for index, skip_channels in enumerate(reversed_skips):
            inject_noise = generator_enabled and (
                config.GENERATOR.noise_level != "medium"
                or index == len(reversed_skips) - 1
            )

            out_channels = skip_channels

            up_blocks.append(
                UpBlock(
                    input_channels=input_channels,
                    skip_channels=skip_channels
                    if config.add_skip_connections
                    else None,
                    out_channels=out_channels,
                    block_config=config.block_config,
                    upsampling_method=config.upsampling_method,
                    skip_alignment_method=config.skip_alignment_method,
                    upsampling_kernel_size=config.transpose_kernel_sizes[index],
                    inject_noise=inject_noise,
                    inject_noise_in_block=inject_noise_in_block,
                )
            )
            input_channels = out_channels

        self.up_blocks = nn.ModuleList(up_blocks)

        self.output = self._build_output(
            in_channels=input_channels,
            out_channels=output_channels,
        )

        if config.checkpoint_config is not None:
            self._load_state_dict(config.checkpoint_config)

        else:
            self._initialize_weights(config.init_method)

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
            Number of channels in the decoder feature tensor.
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
        added_features: torch.Tensor | None,
    ) -> TensorMask:
        """
        Prepare the model input and its validity mask.

        The input mask is broadcast to match the input tensor. Additional feature
        channels are concatenated to the input and treated as fully valid when a mask
        is present.

        Parameters
        ----------
        x : torch.Tensor
            Primary model input tensor.
        x_mask : torch.Tensor | None
            Validity mask associated with the input tensor.
        added_features : torch.Tensor | None
            Additional feature channels to concatenate with the input.

        Returns
        -------
        TensorMask
            Combined input tensor and its corresponding validity mask.
        """
        x_mask = _broadcast_mask(x_mask, x)

        if added_features is not None:
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

    def forward(self, request: DeterministicRequest) -> deterministicOutput:
        """
        Perform a forward pass through the deterministic U-Net.

        The decoder features are generated first and then passed through the final
        output block. When generator sampling is enabled, multiple stochastic outputs
        are restored to separate sample and batch dimensions.

        Parameters
        ----------
        request : DeterministicRequest
            Request containing the model input, optional mask, additional features,
            and output-sampling parameters.

        Returns
        -------
        deterministicOutput
            Model output containing the final projected prediction.
        """
        num_output_samples = request.output_sample_size
        batch_size = request.input.shape[0]

        if self.training and self.config.GENERATOR is not None:
            num_output_samples = self.config.GENERATOR.num_training_noise_samples

        unet_output_tensor = self.forward_decoder(request)
        output = self.output_block(unet_output_tensor)

        if self.config.GENERATOR is not None and num_output_samples > 1:
            output = output.reshape(
                batch_size,
                num_output_samples,
                *output.shape[1:],
            ).transpose(0, 1)

        return deterministicOutput(output=output)

    def forward_decoder(self, request: DeterministicRequest) -> torch.Tensor:
        """
        Encode an input and reconstruct decoder feature maps.

        The input is passed through the encoder and bottleneck before being
        progressively upsampled. Encoder features are merged into the decoder when
        skip connections are enabled. Generator configurations can produce multiple
        stochastic decoder outputs either together or one at a time.

        Parameters
        ----------
        request : DeterministicRequest
            Request containing the model input, optional mask, additional features,
            output sample size, and output-sample chunking preference.

        Returns
        -------
        torch.Tensor
            Decoder feature tensor resized to the configured output spatial shape.
            When multiple generator samples are requested, samples are concatenated
            along the leading batch dimension.
        """
        x = request.input
        x_mask = request.input_mask
        added_features = request.added_features
        num_output_samples = request.output_sample_size
        chunk_output_samples = request.chunk_output_samples

        if self.training and self.config.GENERATOR is not None:
            num_output_samples = self.config.GENERATOR.num_training_noise_samples

        input = self._prepare_input(
            x=x,
            x_mask=x_mask,
            added_features=added_features,
        )

        input = self.initial_mapping(input)

        if self.add_skip_connections:
            down_features: list[TensorMask] = []

            for down_block in self.down_blocks:
                input, skip = down_block(input)
                down_features.append(skip)

        else:
            down_features = self.spatial_shapes[:-1]

            for down_block in self.down_blocks:
                input = down_block(input)

        input = self.bottleneck(input)

        if (
            chunk_output_samples
            and self.config.GENERATOR is not None
            and num_output_samples > 1
        ):
            outputs = []

            for _ in range(num_output_samples):
                decoder_input = input

                for up_block, feature in zip(
                    self.up_blocks,
                    reversed(down_features),
                    strict=True,
                ):
                    if self.add_skip_connections:
                        decoder_input = up_block(
                            decoder_input,
                            skip=feature,
                        )
                    else:
                        decoder_input = up_block(
                            decoder_input,
                            resize_shape=feature,
                        )

                outputs.append(
                    _resize_tensor(
                        decoder_input.tensor,
                        self.output_shape[-2:],
                    )
                )

            return torch.cat(outputs, dim=0)

        if self.config.GENERATOR is not None and num_output_samples > 1:
            input = _repeat_tensor_mask(
                input,
                repeats=num_output_samples,
            )

            if self.add_skip_connections:
                down_features = [
                    _repeat_tensor_mask(
                        skip,
                        repeats=num_output_samples,
                    )
                    for skip in down_features
                ]

        for up_block, feature in zip(
            self.up_blocks,
            reversed(down_features),
            strict=True,
        ):
            if self.add_skip_connections:
                input = up_block(
                    input,
                    skip=feature,
                )
            else:
                input = up_block(
                    input,
                    resize_shape=feature,
                )

        return _resize_tensor(
            input.tensor,
            self.output_shape[-2:],
        )

    @property
    def output_block(self) -> UNetOutput:
        """
        Return the final output projection module.

        Returns
        -------
        UNetOutput
            Output block used to map decoder features to model predictions.
        """
        return self.output


@deterministicModelSelector.register("unetsic")
@dataclasses.dataclass
class UNetSICConfig(UNetConfig):
    """
    Configuration for a deterministic sea ice concentration U-Net.

    Extends the standard U-Net configuration with optional clipping of the final
    output.

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
        Build a sea ice concentration U-Net from this configuration.

        Parameters
        ----------
        input_shape : np.ndarray
            Shape ``(C, H, W)`` of the model input.
        output_shape : np.ndarray | None
            Shape ``(C, H, W)`` of the model output. If ``None``, ``input_shape`` is
            used.
        added_features_dim : int | None
            Number of additional feature channels supplied to the model.

        Returns
        -------
        UNetSIC
            Constructed deterministic sea ice concentration U-Net.
        """
        return UNetSIC(
            config=self,
            input_shape=input_shape,
            output_shape=output_shape,
            added_features_dim=added_features_dim,
        )


class UNetSIC(UNet):
    """
    Deterministic U-Net specialized for sea ice concentration output.

    Uses the standard deterministic U-Net architecture with a sea ice
    concentration output block that supports optional clipping to the interval
    [0, 1].
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
            Number of channels in the decoder feature tensor.
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
