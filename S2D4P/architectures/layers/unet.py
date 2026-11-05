import torch
import torch.nn as nn
import numpy as np
import copy

from S2D4P.architectures.layers.generic import (
    LayerNorm2d,
    MaskPoolingMethod,
    UpsamplingMethod,
    OutputActivation,
    AlignmentMethod,
    NormalizationMethod,
)


from S2D4P.architectures.layers.utils import (
    align_to_skip,
    _noise_injection,
    _same_padding,
)

from S2D4P.architectures.layers.conv import (
    ConvBlockConfig,
    ConvBlock,
    PartialConvBlockConfig,
    PartialConvBlock,
    ConvNeXtBlockConfig,
    ConvNeXtBlock,
    LatentBlock,
    TensorMask,
)

from S2D4P.architectures.layers.partialconv2d import PartialConv2d


def build_conv_block(
    in_channels: int,
    out_channels: int,
    config: ConvBlockConfig | PartialConvBlockConfig | ConvNeXtBlockConfig,
    *,
    latent_size: int = None,
    block_output_shape: tuple[int, int, int] | None = None,
    get_log_var: bool = True,
    latent_normalization: NormalizationMethod | None = "layar",
    inject_noise: bool = False,
) -> nn.Module:
    """
    Build a configured convolutional block.

    The block type is determined from ``config``. Generator noise injection is
    configured on a copy of the supplied configuration. When ``latent_size`` is
    provided, the convolutional block is wrapped in a latent projection block.

    Parameters
    ----------
    in_channels : int
        Number of channels entering the convolutional block.
    out_channels : int
        Number of channels produced by the convolutional block.
    config : ConvBlockConfig | PartialConvBlockConfig | ConvNeXtBlockConfig
        Configuration defining the convolutional block type and behavior.
    latent_size : int | None
        Size of the latent representation. If ``None``, a standard convolutional
        block is returned.
    block_output_shape : tuple[int, int, int] | None
        Output shape of the convolutional block used to construct the latent
        projection layer.
    get_log_var : bool
        Whether a latent block produces log-variance values in addition to the
        latent mean.
    latent_normalization : NormalizationMethod | None
        Normalization applied before latent projection.
    inject_noise : bool
        Whether generator noise is injected within the convolutional block.

    Returns
    -------
    nn.Module
        Configured convolutional block or latent block.

    Raises
    ------
    TypeError
        If ``config`` is not a supported convolutional block configuration.
    ValueError
        If latent projection is requested without ``block_output_shape`` or if
        the output shape channel count does not match ``out_channels``.
    """
    effective_config = copy.copy(config)

    effective_config = effective_config.setup_generator(inject_noise=inject_noise)

    if isinstance(config, ConvBlockConfig):
        block = ConvBlock(
            in_channels,
            out_channels,
            effective_config,
        )

    elif isinstance(config, PartialConvBlockConfig):
        block = PartialConvBlock(
            in_channels,
            out_channels,
            effective_config,
        )

    elif isinstance(config, ConvNeXtBlockConfig):
        block = ConvNeXtBlock(
            in_channels,
            out_channels,
            effective_config,
        )

    else:
        raise TypeError(
            f"Unsupported UNet block configuration: {type(effective_config).__name__}."
        )

    if latent_size is not None:
        if block_output_shape is None:
            raise ValueError(
                "When latent_size is specified, block_output_shape must be available"
            )

        if block_output_shape[0] != out_channels:
            raise ValueError(
                f"Latent block output shape starts with "
                f"{block_output_shape[0]} channels, but the block outputs "
                f"{out_channels} channels."
            )

        return LatentBlock(
            conv_block=block,
            input_shape=block_output_shape,
            latent_size=latent_size,
            get_log_var=get_log_var,
            latent_normalization=latent_normalization,
        )

    return block


class DownBlock(nn.Module):
    """
    Encode and downsample features within a U-Net.

    The input is processed by a convolutional block before spatial downsampling.
    The pre-downsampling features can optionally be returned as a skip connection
    and processed by an additional convolutional block.

    Parameters
    ----------
    in_channels : int
        Number of channels entering the block.
    out_channels : int
        Number of channels produced by downsampling.
    block_config : ConvBlockConfig | PartialConvBlockConfig | ConvNeXtBlockConfig
        Configuration defining convolutional processing.
    mask_pooling : MaskPoolingMethod
        Mask pooling method associated with downsampling.
    mask_fraction_threshold : float
        Minimum valid fraction associated with fraction-based mask pooling.
    return_skip : bool
        Whether to return pre-downsampling features as a skip connection.
    process_skip_connections : bool
        Whether returned skip features are processed by an additional
        convolutional block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        block_config: ConvBlockConfig | PartialConvBlockConfig | ConvNeXtBlockConfig,
        mask_pooling: MaskPoolingMethod,
        mask_fraction_threshold: float,
        return_skip: bool = True,
        process_skip_connections: bool = False,
    ):
        """
        Initialize a U-Net downsampling block.

        Parameters
        ----------
        in_channels : int
            Number of channels entering the block.
        out_channels : int
            Number of channels produced by downsampling.
        block_config : ConvBlockConfig | PartialConvBlockConfig | ConvNeXtBlockConfig
            Configuration defining convolutional processing and downsampling behavior.
        mask_pooling : MaskPoolingMethod
            Mask pooling method associated with downsampling.
        mask_fraction_threshold : float
            Minimum valid fraction associated with fraction-based mask pooling.
        return_skip : bool
            Whether to return pre-downsampling features as a skip connection.
        process_skip_connections : bool
            Whether returned skip features are processed by an additional
            convolutional block.
        """
        super().__init__()
        self.return_skip = return_skip

        self._block = build_conv_block(
            in_channels,
            in_channels,
            block_config,
        )
        self.skip_processor = (
            build_conv_block(
                in_channels,
                in_channels,
                block_config,
            )
            if (process_skip_connections and return_skip)
            else None
        )

        self.use_partial_conv_downsample = isinstance(
            block_config, PartialConvBlockConfig
        ) or getattr(block_config, "use_partial_conv", False)

        if self.use_partial_conv_downsample:
            self.tensor_downsample = PartialConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                padding_mode=block_config.padding_method,
                multi_channel=True,
                return_mask=True,
            )
        else:
            self.tensor_downsample = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                padding_mode=block_config.padding_method,
            )

    def forward(
        self,
        input: TensorMask,
    ) -> TensorMask | tuple[TensorMask, TensorMask]:
        """
        Process and downsample an encoded feature tensor.

        Partial convolution is used for downsampling when required by the block
        configuration, allowing the validity mask to be propagated.

        Parameters
        ----------
        input : TensorMask
            Input feature tensor and optional validity mask.

        Returns
        -------
        TensorMask | tuple[TensorMask, TensorMask]
            Downsampled features, optionally together with the pre-downsampling skip
            features when ``return_skip`` is enabled.
        """
        skip = self._block(input)

        if self.use_partial_conv_downsample:
            downsampled_tensor, downsampled_mask = self.tensor_downsample(
                skip.tensor, skip.mask
            )

        else:
            downsampled_tensor = self.tensor_downsample(skip.tensor)
            downsampled_mask = None

        downsampled = TensorMask(
            tensor=downsampled_tensor,
            mask=downsampled_mask,
        )

        if self.skip_processor is not None:
            skip = self.skip_processor(skip)

        if self.return_skip:
            return downsampled, skip

        else:
            return downsampled

    def output_shape(self, input_shape: np.ndarray | tuple):
        """
        Calculate the spatial output shape after downsampling.

        Parameters
        ----------
        input_shape : np.ndarray | tuple
            Input spatial dimensions.

        Returns
        -------
        tuple
            Spatial dimensions after stride-two downsampling.
        """
        return tuple((shape + 1) // 2 for shape in input_shape)


class UpBlock(nn.Module):
    """
    Upsample and decode features within a U-Net.

    Supports transposed convolution, bilinear interpolation, and pixel-shuffle
    upsampling. Upsampled features can be aligned and concatenated with a skip
    connection before convolutional processing.

    Parameters
    ----------
    input_channels : int
        Number of channels entering the upsampling block.
    skip_channels : int | None
        Number of channels supplied by the skip connection. If ``None``, no skip
        connection is required.
    out_channels : int
        Number of channels produced by the block.
    block_config : ConvBlockConfig | PartialConvBlockConfig | ConvNeXtBlockConfig
        Configuration defining post-upsampling convolutional processing.
    upsampling_method : UpsamplingMethod
        Method used to increase spatial resolution.
    skip_alignment_method : AlignmentMethod
        Method used to align upsampled features with skip features or a requested
        output shape.
    upsampling_kernel_size : int
        Kernel size used by convolutional upsampling or channel projection.
    inject_noise : bool
        Whether generator noise is injected during upsampling.
    inject_noise_in_block : bool
        Whether noise injection is also enabled in the convolutional block after
        upsampling.
    """

    def __init__(
        self,
        input_channels: int,
        skip_channels: int | None,
        out_channels: int,
        *,
        block_config: ConvBlockConfig | PartialConvBlockConfig | ConvNeXtBlockConfig,
        upsampling_method: UpsamplingMethod,
        skip_alignment_method: AlignmentMethod,
        upsampling_kernel_size: int,
        inject_noise: bool = False,
        inject_noise_in_block: bool = False,
    ):
        """
        Initialize a U-Net upsampling block.

        Parameters
        ----------
        input_channels : int
            Number of channels entering the block.
        skip_channels : int | None
            Number of channels in the skip connection.
        out_channels : int
            Number of channels produced after upsampling and processing.
        block_config : ConvBlockConfig | PartialConvBlockConfig | ConvNeXtBlockConfig
            Configuration defining convolutional processing.
        upsampling_method : UpsamplingMethod
            Upsampling method to use.
        skip_alignment_method : AlignmentMethod
            Method used to align spatial dimensions.
        upsampling_kernel_size : int
            Kernel size used during upsampling or channel projection.
        inject_noise : bool
            Whether generator noise is injected during upsampling.
        inject_noise_in_block : bool
            Whether generator noise is additionally injected within the subsequent
            convolutional block.

        Raises
        ------
        ValueError
            If ``upsampling_method`` is unsupported.
        """
        super().__init__()

        self.input_channels = input_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        self.upsampling_method = upsampling_method
        self.skip_alignment_method = skip_alignment_method
        self.skip_padding_method = block_config.padding_method
        self.inject_noise = inject_noise
        self.inject_noise_in_block = inject_noise_in_block

        added_upsampling_channels = 0
        if self.inject_noise:
            added_upsampling_channels += 1

        if upsampling_method == "transpose_conv":
            self.upsample = nn.ConvTranspose2d(
                input_channels + added_upsampling_channels,
                out_channels,
                kernel_size=upsampling_kernel_size,
                stride=2,
            )
            self.channel_projection = nn.Identity()

        elif upsampling_method == "bilinear":
            self.upsample = nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            )

            if isinstance(block_config, PartialConvBlockConfig) or getattr(
                block_config, "use_partial_conv", False
            ):
                self.channel_projection = PartialConv2d(
                    input_channels + added_upsampling_channels,
                    out_channels,
                    kernel_size=upsampling_kernel_size,
                    multi_channel=False,
                    return_mask=False,
                    padding=_same_padding(upsampling_kernel_size),
                )

            else:
                self.channel_projection = nn.Conv2d(
                    input_channels + added_upsampling_channels,
                    out_channels,
                    kernel_size=upsampling_kernel_size,
                    padding=_same_padding(upsampling_kernel_size),
                    padding_mode=block_config.padding_method,
                )

        elif upsampling_method == "pixelshuffle":
            if isinstance(block_config, PartialConvBlockConfig) or getattr(
                block_config, "use_partial_conv", False
            ):
                self.upsample = PartialConv2d(
                    input_channels + added_upsampling_channels,
                    out_channels * 4,
                    kernel_size=upsampling_kernel_size,
                    multi_channel=False,
                    return_mask=False,
                    padding=_same_padding(upsampling_kernel_size),
                )
            else:
                self.upsample = nn.Conv2d(
                    input_channels + added_upsampling_channels,
                    out_channels * 4,
                    kernel_size=upsampling_kernel_size,
                    padding=_same_padding(upsampling_kernel_size),
                )

            self.channel_projection = nn.PixelShuffle(upscale_factor=2)
        else:
            raise ValueError(f"Unsupported upsampling mode: {upsampling_method!r}.")

        effective_block_config = copy.copy(block_config)
        if getattr(effective_block_config, "multi_channel", False) and getattr(
            effective_block_config, "return_mask", False
        ):
            effective_block_config.multi_channel = False
            effective_block_config.return_mask = False

        skip_channels = skip_channels or 0

        self._block = build_conv_block(
            skip_channels + out_channels,
            out_channels,
            effective_block_config,
            inject_noise=(self.inject_noise and self.inject_noise_in_block),
        )

    def forward(
        self,
        input: TensorMask,
        skip: TensorMask | None = None,
        resize_shape: tuple | None = None,
    ) -> TensorMask:
        """
        Upsample and process a decoder feature tensor.

        Optional generator noise is injected at a location determined by the
        upsampling method. When a skip connection is configured, the upsampled tensor
        is aligned with and concatenated to the skip tensor before convolutional
        processing.

        Parameters
        ----------
        input : TensorMask
            Input feature tensor and optional validity mask.
        skip : TensorMask | None
            Optional encoder skip features.
        resize_shape : tuple | None
            Spatial shape to which the upsampled tensor is aligned when no skip
            connection is configured.

        Returns
        -------
        TensorMask
            Upsampled and processed feature tensor.

        Raises
        ------
        ValueError
            If a skip connection is configured but ``skip`` is not provided.
        """
        if self.inject_noise:
            if self.upsampling_method == "transpose_conv":
                x = _noise_injection(input.tensor)
                x = self.upsample(x)
                x = self.channel_projection(x)

            elif self.upsampling_method == "bilinear":
                x = self.upsample(input.tensor)
                x = _noise_injection(x)
                x = self.channel_projection(x)

            else:
                x = _noise_injection(input.tensor)
                x = self.upsample(x)
                x = self.channel_projection(x)

        else:
            x = self.upsample(input.tensor)
            x = self.channel_projection(x)

        if self.skip_channels is not None:
            if skip is None:
                raise ValueError(
                    "A skip tensor is required when skip_channels is configured."
                )

            x = align_to_skip(
                x,
                skip.tensor.shape[-2:],
                self.skip_alignment_method,
                self.skip_padding_method,
            )

            merged_tensor = torch.cat([skip.tensor, x], dim=1)

        elif resize_shape is not None:
            merged_tensor = align_to_skip(
                x, resize_shape, self.skip_alignment_method, self.skip_padding_method
            )

        else:
            merged_tensor = x

        return self._block(
            TensorMask(
                tensor=merged_tensor,
                mask=None,
            )
        )


class UNetOutput(nn.Module):
    """
    Project U-Net features to output channels.

    The output head uses either a direct pointwise convolution or an intermediate
    partial-convolution stage before the final projection.

    Parameters
    ----------
    in_channels : int
        Number of input feature channels.
    out_channels : int
        Number of output channels.
    hidden_channels : int | None
        Number of intermediate channels. If ``None``, features are projected
        directly to the output channels.
    activation : OutputActivation
        Activation applied to the final output.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        hidden_channels: int | None,
        activation: OutputActivation,
    ):
        """
        Initialize a U-Net output projection.

        Parameters
        ----------
        in_channels : int
            Number of input feature channels.
        out_channels : int
            Number of output channels.
        hidden_channels : int | None
            Number of intermediate channels, or ``None`` for direct projection.
        activation : OutputActivation
            Final output activation.

        Raises
        ------
        ValueError
            If ``activation`` is unsupported.
        """
        super().__init__()

        if hidden_channels is None:
            layers: list[nn.Module] = [
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            ]
        else:
            layers = [
                PartialConv2d(
                    in_channels,
                    hidden_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    multi_channel=False,
                    return_mask=False,
                ),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
            ]

        if activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif activation == "tanh":
            layers.append(nn.Tanh())
        elif activation != "identity":
            raise ValueError(f"Unsupported output activation: {activation!r}.")

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project feature maps to model outputs.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor.

        Returns
        -------
        torch.Tensor
            Projected output tensor after the configured output activation.
        """
        return self.layers(x)


class UNetOutputSIC(nn.Module):
    """
    Project U-Net features to sea-ice-concentration outputs.

    The output head applies normalization and activation before the final
    projection and can optionally constrain predictions to the interval [0, 1].

    Parameters
    ----------
    in_channels : int
        Number of input feature channels.
    out_channels : int
        Number of output channels.
    hidden_channels : int | None
        Number of intermediate channels. If ``None``, no intermediate partial
        convolution is used.
    activation : OutputActivation
        Activation applied to the final output.
    clip_output : bool
        Whether final predictions are clipped to the interval [0, 1].
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        hidden_channels: int | None,
        activation: OutputActivation,
        clip_output: bool = False,
    ):
        """
        Initialize a sea-ice-concentration output projection.

        Parameters
        ----------
        in_channels : int
            Number of input feature channels.
        out_channels : int
            Number of output channels.
        hidden_channels : int | None
            Number of intermediate channels, or ``None`` for direct projection after
            input normalization and activation.
        activation : OutputActivation
            Final output activation.
        clip_output : bool
            Whether final predictions are clipped to the interval [0, 1].

        Raises
        ------
        ValueError
            If ``activation`` is unsupported.
        """
        super().__init__()

        self.clip_output = clip_output

        if hidden_channels is None:
            layers: list[nn.Module] = [
                LayerNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
            ]
        else:
            layers = [
                PartialConv2d(
                    in_channels,
                    hidden_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    multi_channel=False,
                    return_mask=False,
                ),
                LayerNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
            ]

        if activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif activation == "tanh":
            layers.append(nn.Tanh())
        elif activation != "identity":
            raise ValueError(f"Unsupported output activation: {activation!r}.")

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project feature maps to sea-ice-concentration outputs.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor.

        Returns
        -------
        torch.Tensor
            Projected output tensor, optionally clipped to the interval [0, 1].
        """
        out = self.layers(x)
        if self.clip_output:
            out = torch.clamp(out, 0, 1)
        return out
