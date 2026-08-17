import torch
import torch.nn as nn
import numpy as np
import copy

from cccma_ppp.architectures.layers.generic import (
    LayerNorm2d,
    MaskPoolingMethod,
    UpsamplingMethod,
    OutputActivation,
    AlignmentMethod,
    NormalizationMethod,
)


from cccma_ppp.architectures.layers.utils import align_to_skip, _noise_injection

from cccma_ppp.architectures.layers.conv import (
    ConvBlockConfig,
    ConvBlock,
    PartialConvBlockConfig,
    PartialConvBlock,
    ConvNeXtBlockConfig,
    ConvNeXtBlock,
    LatentBlock,
    TensorMask,
)

from cccma_ppp.architectures.layers.partialconv2d import PartialConv2d


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
    Document this function.

    Parameters
    ----------
    in_channels : int
        Description not yet provided.
    out_channels : int
        Description not yet provided.
    config : ConvBlockConfig | PartialConvBlockConfig | ConvNeXtBlockConfig
        Description not yet provided.
    latent_size : int
        Description not yet provided.
    block_output_shape : tuple[int, int, int] | None
        Description not yet provided.
    get_log_var : bool
        Description not yet provided.
    latent_normalization : NormalizationMethod | None
        Description not yet provided.
    inject_noise : bool
        Description not yet provided.

    Returns
    -------
    nn.Module
        Description not yet provided.

    Raises
    ------
    TypeError
        Description not yet provided.
    ValueError
        Description not yet provided.
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
    Document this class.

    Parameters
    ----------
    in_channels : int
        Description not yet provided.
    out_channels : int
        Description not yet provided.
    block_config : ConvBlockConfig | PartialConvBlockConfig | ConvNeXtBlockConfig
        Description not yet provided.
    mask_pooling : MaskPoolingMethod
        Description not yet provided.
    mask_fraction_threshold : float
        Description not yet provided.
    return_skip : bool
        Description not yet provided.
    process_skip_connections : bool
        Description not yet provided.
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
        Document this function.

        Parameters
        ----------
        in_channels : int
            Description not yet provided.
        out_channels : int
            Description not yet provided.
        block_config : ConvBlockConfig | PartialConvBlockConfig | ConvNeXtBlockConfig
            Description not yet provided.
        mask_pooling : MaskPoolingMethod
            Description not yet provided.
        mask_fraction_threshold : float
            Description not yet provided.
        return_skip : bool
            Description not yet provided.
        process_skip_connections : bool
            Description not yet provided.
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
        Document this function.

        Parameters
        ----------
        input : TensorMask
            Description not yet provided.

        Returns
        -------
        TensorMask | tuple[TensorMask, TensorMask]
            Description not yet provided.
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
        Document this function.

        Parameters
        ----------
        input_shape : np.ndarray | tuple
            Description not yet provided.

        Returns
        -------
        Any
            Description not yet provided.
        """
        return tuple((shape + 1) // 2 for shape in input_shape)


class UpBlock(nn.Module):
    """
    Document this class.

    Parameters
    ----------
    input_channels : int
        Description not yet provided.
    skip_channels : int | None
        Description not yet provided.
    out_channels : int
        Description not yet provided.
    block_config : ConvBlockConfig | PartialConvBlockConfig | ConvNeXtBlockConfig
        Description not yet provided.
    upsampling_method : UpsamplingMethod
        Description not yet provided.
    skip_alignment_method : AlignmentMethod
        Description not yet provided.
    transpose_kernel_size : int
        Description not yet provided.
    inject_noise : bool
        Description not yet provided.
    inject_noise_in_block : bool
        Description not yet provided.
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
        transpose_kernel_size: int,
        inject_noise: bool = False,
        inject_noise_in_block: bool = False,
    ):
        """
        Document this function.

        Parameters
        ----------
        input_channels : int
            Description not yet provided.
        skip_channels : int | None
            Description not yet provided.
        out_channels : int
            Description not yet provided.
        block_config : ConvBlockConfig | PartialConvBlockConfig | ConvNeXtBlockConfig
            Description not yet provided.
        upsampling_method : UpsamplingMethod
            Description not yet provided.
        skip_alignment_method : AlignmentMethod
            Description not yet provided.
        transpose_kernel_size : int
            Description not yet provided.
        inject_noise : bool
            Description not yet provided.
        inject_noise_in_block : bool
            Description not yet provided.

        Raises
        ------
        ValueError
            Description not yet provided.
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
                kernel_size=transpose_kernel_size,
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
                    kernel_size=3,
                    multi_channel=False,
                    return_mask=False,
                    padding=1,
                )

            else:
                self.channel_projection = nn.Conv2d(
                    input_channels + added_upsampling_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    padding_mode=block_config.padding_method,
                )

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
        Document this function.

        Parameters
        ----------
        input : TensorMask
            Description not yet provided.
        skip : TensorMask | None
            Description not yet provided.
        resize_shape : tuple | None
            Description not yet provided.

        Returns
        -------
        TensorMask
            Description not yet provided.

        Raises
        ------
        ValueError
            Description not yet provided.
        """
        if self.inject_noise:
            if self.upsampling_method == "transpose_conv":
                x = _noise_injection(input.tensor)
                x = self.upsample(x)
                x = self.channel_projection(x)

            else:
                x = self.upsample(input.tensor)
                x = _noise_injection(x)
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

        return self._block(
            TensorMask(
                tensor=merged_tensor,
                mask=None,
            )
        )


class UNetOutput(nn.Module):
    """
    Document this class.

    Parameters
    ----------
    in_channels : int
        Description not yet provided.
    out_channels : int
        Description not yet provided.
    hidden_channels : int | None
        Description not yet provided.
    activation : OutputActivation
        Description not yet provided.
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
        Document this function.

        Parameters
        ----------
        in_channels : int
            Description not yet provided.
        out_channels : int
            Description not yet provided.
        hidden_channels : int | None
            Description not yet provided.
        activation : OutputActivation
            Description not yet provided.

        Raises
        ------
        ValueError
            Description not yet provided.
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
        Document this function.

        Parameters
        ----------
        x : torch.Tensor
            Description not yet provided.

        Returns
        -------
        torch.Tensor
            Description not yet provided.
        """
        return self.layers(x)


class UNetOutputSIC(nn.Module):
    """
    Document this class.

    Parameters
    ----------
    in_channels : int
        Description not yet provided.
    out_channels : int
        Description not yet provided.
    hidden_channels : int | None
        Description not yet provided.
    activation : OutputActivation
        Description not yet provided.
    clip_output : bool
        Description not yet provided.
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
        Document this function.

        Parameters
        ----------
        in_channels : int
            Description not yet provided.
        out_channels : int
            Description not yet provided.
        hidden_channels : int | None
            Description not yet provided.
        activation : OutputActivation
            Description not yet provided.
        clip_output : bool
            Description not yet provided.

        Raises
        ------
        ValueError
            Description not yet provided.
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
        Document this function.

        Parameters
        ----------
        x : torch.Tensor
            Description not yet provided.

        Returns
        -------
        torch.Tensor
            Description not yet provided.
        """
        out = self.layers(x)
        if self.clip_output:
            out = torch.clamp(out, 0, 1)
        return out
