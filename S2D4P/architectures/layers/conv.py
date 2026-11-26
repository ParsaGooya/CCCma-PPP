import dataclasses
from typing import Literal, final
import abc

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from S2D4P.architectures.layers.partialconv2d import PartialConv2d
from S2D4P.architectures.layers.generic import (
    MaskPoolingMethod,
    NormalizationMethod,
    ActivationName,
    PaddingMethod,
    _build_normalization,
    _build_activation,
    _validate_dropout,
    LayerNorm2d,
    DropPath,
)
from S2D4P.architectures.layers.utils import (
    _same_padding,
    _broadcast_mask,
)

from S2D4P.architectures.layers.utils import _noise_injection, _expand_mask


@dataclasses.dataclass
class TensorMask:
    """
    Container for a tensor and its optional validity mask.

    Parameters
    ----------
    tensor : torch.Tensor
        Data tensor.
    mask : torch.Tensor | None
        Optional validity mask associated with the tensor.
    """

    tensor: torch.Tensor
    mask: torch.Tensor | None = None


@dataclasses.dataclass
class LatentVector:
    """
    Container for latent distribution parameters.

    Parameters
    ----------
    mu : torch.Tensor
        Mean of the latent representation.
    log_var : torch.Tensor | None
        Log-variance of the latent representation, or ``None`` when variance is
        not modeled.
    """

    mu: torch.Tensor
    log_var: torch.Tensor | None


class ConvBlockConfigABC(abc.ABC):
    """
    Abstract base configuration for convolutional blocks.

    Provides common configuration for optional generator noise injection.
    """

    def __init__(self):
        """
        Abstract base configuration for convolutional blocks.

        Provides common configuration for optional generator noise injection.
        """
        self.inject_noise: bool = False

    @final
    def setup_generator(
        self,
        inject_noise: bool = False,
    ):
        """
        Configure generator-specific noise injection.

        Parameters
        ----------
        inject_noise : bool
            Whether convolutional stages inject an additional random noise channel.

        Returns
        -------
        ConvBlockConfigABC
            This configuration instance with generator settings applied.
        """
        self.inject_noise = inject_noise

        return self


@dataclasses.dataclass
class ConvBlockConfig(ConvBlockConfigABC):
    """
    Configuration for a standard convolutional block.

    Parameters
    ----------
    name : Literal["standard_conv"]
        Identifier for the standard convolutional block type.
    num_convolutions : int
        Number of convolutional stages in the block.
    kernel_size : int
        Spatial kernel size used by each convolution.
    normalization : NormalizationMethod
        Normalization method applied after each convolution.
    padding_method : PaddingMethod
        Padding mode used by each convolution.
    activation : ActivationName
        Activation function applied after normalization.
    dropout_rate : float | None
        Dropout probability applied after activation. If ``None`` or zero,
        dropout is disabled.
    bias : bool
        Whether convolutional layers include a learnable bias.
    group_norm_groups : int
        Number of groups used when group normalization is selected.
    """

    name: Literal["standard_conv"]
    num_convolutions: int = 2
    kernel_size: int = 3
    normalization: NormalizationMethod = "batch"
    padding_method: PaddingMethod = "zeros"
    activation: ActivationName = "relu"
    dropout_rate: float | None = None
    bias: bool = False
    group_norm_groups: int = 8

    def __post_init__(self) -> None:
        """
        Validate the standard convolutional block configuration.

        Raises
        ------
        ValueError
            If ``num_convolutions`` is less than one, the kernel size is incompatible
            with same padding, or the dropout rate is invalid.
        """
        super().__init__()

        if self.num_convolutions < 1:
            raise ValueError("num_convolutions must be at least 1.")
        _same_padding(self.kernel_size)
        _validate_dropout(self.dropout_rate)


@dataclasses.dataclass
class PartialConvBlockConfig(ConvBlockConfigABC):
    """
    Configuration for a partial-convolution block.

    Parameters
    ----------
    name : Literal["partial_conv"]
        Identifier for the partial-convolution block type.
    num_convolutions : int
        Number of partial-convolution stages in the block.
    kernel_size : int
        Spatial kernel size used by each partial convolution.
    normalization : NormalizationMethod
        Normalization method applied after each convolution.
    padding_method : PaddingMethod
        Padding mode used by each convolution.
    activation : ActivationName
        Activation function applied after normalization.
    dropout_rate : float | None
        Dropout probability applied after activation. If ``None`` or zero,
        dropout is disabled.
    bias : bool
        Whether convolutional layers include a learnable bias.
    group_norm_groups : int
        Number of groups used when group normalization is selected.
    """

    name: Literal["partial_conv"]
    num_convolutions: int = 2
    kernel_size: int = 3
    normalization: NormalizationMethod = "batch"
    padding_method: PaddingMethod = "zeros"
    activation: ActivationName = "relu"
    dropout_rate: float | None = None
    bias: bool = False
    group_norm_groups: int = 8

    multi_channel: bool = dataclasses.field(init=False, default=True)
    return_mask: bool = dataclasses.field(init=False, default=True)

    def __post_init__(self) -> None:
        """
        Validate the partial-convolution block configuration.

        Raises
        ------
        ValueError
            If ``num_convolutions`` is less than one, the kernel size is incompatible
            with same padding, or the dropout rate is invalid.
        """
        super().__init__()

        if self.num_convolutions < 1:
            raise ValueError("num_convolutions must be at least 1.")
        _same_padding(self.kernel_size)
        _validate_dropout(self.dropout_rate)


@dataclasses.dataclass
class ConvNeXtBlockConfig(ConvBlockConfigABC):
    """
    Configuration for a ConvNeXt-style convolutional block.

    Parameters
    ----------
    name : Literal["convnext"]
        Identifier for the ConvNeXt block type.
    num_blocks : int
        Number of residual ConvNeXt stages in the block.
    kernel_size : int
        Spatial kernel size used by the depthwise convolution.
    expansion_ratio : int
        Factor by which the first pointwise convolution expands the channel
        dimension.
    padding_method : PaddingMethod
        Padding mode used by convolutional layers.
    layer_scale_init : float
        Initial value of the learnable per-channel residual scaling parameter.
    dropout_rate : float
        Dropout probability applied within each ConvNeXt stage.
    drop_path_rate : float
        Maximum stochastic-depth probability applied across the block.
    use_partial_conv : bool
        Whether depthwise and projection convolutions use partial convolutions.
    """

    name: Literal["convnext"]
    num_blocks: int = 2
    kernel_size: int = 7
    expansion_ratio: int = 4
    padding_method: PaddingMethod = "zeros"
    layer_scale_init: float = 1e-6
    dropout_rate: float = 0.0
    drop_path_rate: float = 0.0
    use_partial_conv: bool = True

    multi_channel: bool = dataclasses.field(init=False, default=True)
    return_mask: bool = dataclasses.field(init=False, default=True)

    def __post_init__(self) -> None:
        """
        Validate the ConvNeXt block configuration.

        Raises
        ------
        ValueError
            If ``num_blocks`` or ``expansion_ratio`` is less than one, the kernel
            size is incompatible with same padding, or either dropout probability is
            invalid.
        """
        super().__init__()

        if self.num_blocks < 1:
            raise ValueError("num_blocks must be at least 1.")
        if self.expansion_ratio < 1:
            raise ValueError("expansion_ratio must be at least 1.")
        _same_padding(self.kernel_size)
        _validate_dropout(self.dropout_rate)
        _validate_dropout(self.drop_path_rate)


class ConvSingle(nn.Module):
    """
    Single standard convolutional processing stage.

    The stage applies optional noise injection followed by convolution,
    normalization, activation, and dropout.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    config : ConvBlockConfig
        Configuration controlling convolution, normalization, activation,
        dropout, and noise injection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: ConvBlockConfig,
    ):
        """
        Single standard convolutional processing stage.

        The stage applies optional noise injection followed by convolution,
        normalization, activation, and dropout.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        config : ConvBlockConfig
            Configuration controlling convolution, normalization, activation,
            dropout, and noise injection.
        """

        super().__init__()

        self.inject_noise = config.inject_noise
        added_noise_channel = 0
        if self.inject_noise:
            added_noise_channel += 1

        self.conv = nn.Conv2d(
            in_channels + added_noise_channel,
            out_channels,
            kernel_size=config.kernel_size,
            padding=_same_padding(config.kernel_size),
            bias=config.bias,
            padding_mode=config.padding_method,
        )

        self.normalization = _build_normalization(
            config.normalization,
            out_channels,
            group_norm_groups=config.group_norm_groups,
        )
        self.activation = _build_activation(config.activation)
        self.dropout = (
            nn.Dropout2d(config.dropout_rate)
            if config.dropout_rate is not None and config.dropout_rate > 0
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the convolutional stage.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor.

        Returns
        -------
        torch.Tensor
            Output after optional noise injection, convolution, normalization,
            activation, and dropout.
        """
        if self.inject_noise:
            x = _noise_injection(x)

        x = self.conv(x)
        x = self.normalization(x)
        x = self.activation(x)
        return self.dropout(x)


class PartialConvSingle(nn.Module):
    """
    Single partial-convolution processing stage.

    The stage propagates an optional validity mask while applying partial
    convolution, normalization, activation, dropout, and optional noise injection.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    config : PartialConvBlockConfig
        Configuration controlling partial convolution and subsequent processing.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: PartialConvBlockConfig,
    ):
        """
        Initialize a partial-convolution stage.

        Parameters
        ----------
        in_channels : int
            Number of input channels before optional noise injection.
        out_channels : int
            Number of output channels.
        config : PartialConvBlockConfig
            Configuration controlling partial convolution, normalization, activation,
            dropout, masking, and noise injection.
        """
        super().__init__()
        self.multi_channel = config.multi_channel
        self.return_mask = config.return_mask
        self.inject_noise = config.inject_noise

        added_noise_channel = 0
        if self.inject_noise:
            added_noise_channel += 1

        self.conv = PartialConv2d(
            in_channels + added_noise_channel,
            out_channels,
            kernel_size=config.kernel_size,
            padding=_same_padding(config.kernel_size),
            bias=config.bias,
            multi_channel=config.multi_channel,
            return_mask=config.return_mask,
            padding_mode=config.padding_method,
        )
        self.normalization = _build_normalization(
            config.normalization,
            out_channels,
            group_norm_groups=config.group_norm_groups,
        )
        self.activation = _build_activation(config.activation)
        self.dropout = (
            nn.Dropout2d(config.dropout_rate)
            if config.dropout_rate is not None and config.dropout_rate > 0
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Apply the partial-convolution stage.

        When noise injection is enabled for multi-channel masks, the mask is expanded
        to remain compatible with the additional noise channel.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor.
        mask : torch.Tensor | None
            Optional validity mask associated with the input tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None]
            Processed feature tensor and updated validity mask. The returned mask is
            ``None`` when mask propagation is disabled.
        """
        if self.inject_noise:
            x = _noise_injection(x)
            if self.multi_channel:
                mask = _expand_mask(x, mask)

        if self.return_mask:
            x, mask = self.conv(x, mask)
        else:
            x = self.conv(x, mask)
            mask = None

        x = self.normalization(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x, mask


class ConvNeXtSingle(nn.Module):
    """
    Single residual ConvNeXt processing stage.

    Applies a depthwise convolution, normalization, pointwise channel expansion
    and contraction, optional noise injection, layer scaling, dropout, and
    stochastic depth before adding the residual connection.

    Parameters
    ----------
    channels : int
        Number of input and output feature channels.
    config : ConvNeXtBlockConfig
        Configuration controlling the ConvNeXt stage.
    drop_path_rate : float
        Stochastic-depth probability used for the residual branch.
    """

    def __init__(
        self,
        channels: int,
        config: ConvNeXtBlockConfig,
        drop_path_rate: float,
    ):
        """
        Initialize a residual ConvNeXt stage.

        Parameters
        ----------
        channels : int
            Number of input and output feature channels.
        config : ConvNeXtBlockConfig
            Configuration controlling convolution, channel expansion, dropout, layer
            scaling, masking, and noise injection.
        drop_path_rate : float
            Stochastic-depth probability applied to the residual branch.
        """
        super().__init__()
        hidden_channels = channels * config.expansion_ratio
        self.use_partial_conv = config.use_partial_conv
        self.inject_noise = config.inject_noise
        self.multi_channel = config.multi_channel
        self.return_mask = config.return_mask

        added_noise_channel = 0
        if self.inject_noise:
            added_noise_channel += 1

        if config.use_partial_conv:
            self.depthwise = PartialConv2d(
                channels,
                channels,
                kernel_size=config.kernel_size,
                padding=_same_padding(config.kernel_size),
                groups=channels,
                multi_channel=config.multi_channel,
                return_mask=config.return_mask,
                padding_mode=config.padding_method,
            )

        else:
            self.depthwise = nn.Conv2d(
                channels,
                channels,
                kernel_size=config.kernel_size,
                padding=_same_padding(config.kernel_size),
                groups=channels,
                padding_mode=config.padding_method,
            )

        self.normalization = LayerNorm2d(channels)
        self.pointwise_1 = nn.Conv2d(
            channels + added_noise_channel, hidden_channels, kernel_size=1
        )
        self.activation = nn.GELU()
        self.dropout = (
            nn.Dropout2d(config.dropout_rate)
            if config.dropout_rate > 0
            else nn.Identity()
        )
        self.pointwise_2 = nn.Conv2d(
            hidden_channels + added_noise_channel, channels, kernel_size=1
        )

        if config.layer_scale_init > 0:
            self.layer_scale = nn.Parameter(
                config.layer_scale_init * torch.ones(channels)
            )
        else:
            self.layer_scale = None

        self.drop_path = DropPath(drop_path_rate)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Apply the residual ConvNeXt stage.

        The validity mask is propagated when partial convolution is enabled. Optional
        noise channels are injected before each pointwise convolution.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor.
        mask : torch.Tensor | None
            Optional validity mask associated with the input tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None]
            Residual output tensor and the propagated validity mask.
        """
        residual = x

        if self.use_partial_conv:
            if self.return_mask:
                x, mask = self.depthwise(x, mask)
            else:
                x = self.depthwise(x, mask)
                mask = None
        else:
            x = self.depthwise(x)

        x = self.normalization(x)

        if self.inject_noise:
            x = _noise_injection(x)
        x = self.pointwise_1(x)
        x = self.activation(x)
        x = self.dropout(x)

        if self.inject_noise:
            x = _noise_injection(x)
        x = self.pointwise_2(x)

        if self.layer_scale is not None:
            x = x * self.layer_scale.view(1, -1, 1, 1)

        return residual + self.drop_path(x), mask


class ConvBlock(nn.Module):
    """
    Sequence of standard convolutional stages.

    Parameters
    ----------
    in_channels : int
        Number of channels entering the block.
    out_channels : int
        Number of channels produced by each convolutional stage.
    config : ConvBlockConfig
        Configuration defining the stages in the block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: ConvBlockConfig,
    ):
        """
        Initialize a standard convolutional block.

        Parameters
        ----------
        in_channels : int
            Number of channels entering the first stage.
        out_channels : int
            Number of channels produced by each stage.
        config : ConvBlockConfig
            Configuration defining the number and behavior of convolutional stages.
        """
        super().__init__()
        self.config = config
        self.out_channels = out_channels

        stages = []
        current_channels = in_channels

        for _ in range(config.num_convolutions):
            stages.append(
                ConvSingle(
                    current_channels,
                    out_channels,
                    config,
                )
            )
            current_channels = out_channels

        self.stages = nn.ModuleList(stages)

    def forward(self, input: TensorMask) -> TensorMask:
        """
        Apply all standard convolutional stages.

        The input validity mask is passed through unchanged because standard
        convolutions do not update masks.

        Parameters
        ----------
        input : TensorMask
            Input tensor and optional validity mask.

        Returns
        -------
        TensorMask
            Processed tensor together with the original validity mask.
        """
        x = input.tensor

        for stage in self.stages:
            x = stage(x)

        return TensorMask(
            tensor=x,
            mask=input.mask,
        )


class PartialConvBlock(nn.Module):
    """
    Sequence of partial-convolution stages.

    Parameters
    ----------
    in_channels : int
        Number of channels entering the block.
    out_channels : int
        Number of channels produced by each partial-convolution stage.
    config : PartialConvBlockConfig
        Configuration defining the stages in the block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: PartialConvBlockConfig,
    ):
        """
        Initialize a partial-convolution block.

        Parameters
        ----------
        in_channels : int
            Number of channels entering the first stage.
        out_channels : int
            Number of channels produced by each stage.
        config : PartialConvBlockConfig
            Configuration defining the number and behavior of partial-convolution
            stages.
        """
        super().__init__()
        self.config = config
        self.out_channels = out_channels

        stages: list[nn.Module] = []
        current_channels = in_channels

        for _ in range(config.num_convolutions):
            stages.append(
                PartialConvSingle(
                    current_channels,
                    out_channels,
                    config,
                )
            )
            current_channels = out_channels

        self.stages = nn.ModuleList(stages)

    def forward(self, input: TensorMask) -> TensorMask:
        """
        Apply all partial-convolution stages.

        The input mask is first broadcast to the input tensor and subsequently updated
        by each partial-convolution stage.

        Parameters
        ----------
        input : TensorMask
            Input tensor and optional validity mask.

        Returns
        -------
        TensorMask
            Processed tensor and propagated validity mask.
        """
        mask = _broadcast_mask(input.mask, input.tensor)

        x = input.tensor
        for stage in self.stages:
            x, mask = stage(x, mask)

        return TensorMask(tensor=x, mask=mask)


class ConvNeXtBlock(nn.Module):
    """
    Sequence of residual ConvNeXt stages.

    The block optionally projects the input to the requested channel count before
    applying a sequence of ConvNeXt stages with progressively increasing
    stochastic-depth rates.

    Parameters
    ----------
    in_channels : int
        Number of channels entering the block.
    out_channels : int
        Number of channels used by the ConvNeXt stages.
    config : ConvNeXtBlockConfig
        Configuration defining projection and ConvNeXt stage behavior.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: ConvNeXtBlockConfig,
    ):
        """
        Initialize a ConvNeXt block.

        Parameters
        ----------
        in_channels : int
            Number of channels entering the block.
        out_channels : int
            Number of channels used by the ConvNeXt stages.
        config : ConvNeXtBlockConfig
            Configuration defining projection, masking, stage count, and
            stochastic-depth behavior.
        """
        super().__init__()

        self.config = config
        self.out_channels = out_channels
        self.use_partial_conv = config.use_partial_conv
        self.requires_projection = in_channels != out_channels
        self.return_mask = config.return_mask

        if self.requires_projection:
            if self.use_partial_conv:
                self.projection_conv = PartialConv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    multi_channel=config.multi_channel,
                    return_mask=config.return_mask,
                )
            else:
                self.projection_conv = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                )

            self.projection_norm = LayerNorm2d(out_channels)

        else:
            self.projection_conv = None
            self.projection_norm = nn.Identity()

        if config.num_blocks == 1:
            drop_rates = [config.drop_path_rate]
        else:
            drop_rates = torch.linspace(
                0.0,
                config.drop_path_rate,
                config.num_blocks,
            ).tolist()

        self.blocks = nn.ModuleList(
            [
                ConvNeXtSingle(
                    out_channels,
                    config,
                    drop_path_rate=float(drop_rate),
                )
                for drop_rate in drop_rates
            ]
        )

    def forward(self, input: TensorMask) -> TensorMask:
        """
        Apply channel projection and the configured ConvNeXt stages.

        A projection is performed when input and output channel counts differ. Masks
        are propagated through partial convolutions when enabled.

        Parameters
        ----------
        input : TensorMask
            Input tensor and optional validity mask.

        Returns
        -------
        TensorMask
            Processed tensor and propagated validity mask.
        """
        x = input.tensor
        mask = input.mask

        if self.requires_projection:
            if self.use_partial_conv:
                if self.return_mask:
                    x, mask = self.projection_conv(x, mask)
                else:
                    x = self.projection_conv(x, mask)
                    mask = None
            else:
                x = self.projection_conv(x)

            x = self.projection_norm(x)

        for block in self.blocks:
            x, mask = block(x, mask)

        return TensorMask(
            tensor=x,
            mask=mask,
        )


class LatentBlock(nn.Module):
    """
    Convert convolutional features into latent distribution parameters.

    A convolutional feature block is applied first, followed by a latent head
    that produces the latent mean and optional log-variance.

    Parameters
    ----------
    conv_block : ConvNeXtBlock | PartialConvBlock | ConvBlock
        Convolutional block used to extract latent features.
    input_shape : tuple[int, int, int]
        Shape of the convolutional features supplied to the latent head.
    latent_size : int
        Number of latent features produced by the latent head.
    get_log_var : bool
        Whether to produce latent log-variance values in addition to the mean.
    latent_normalization : NormalizationMethod | None
        Normalization applied before flattening latent features. If ``None``, no
        latent normalization is applied.
    """

    def __init__(
        self,
        conv_block: ConvNeXtBlock | PartialConvBlock | ConvBlock,
        input_shape: tuple[int, int, int],
        latent_size: int,
        *,
        get_log_var: bool = True,
        latent_normalization: NormalizationMethod | None = "layer",
    ):
        """
        Initialize a convolutional latent block.

        Parameters
        ----------
        conv_block : ConvNeXtBlock | PartialConvBlock | ConvBlock
            Convolutional block used to extract latent features.
        input_shape : tuple[int, int, int]
            Shape of the convolutional features supplied to the latent head.
        latent_size : int
            Number of latent features produced by the latent head.
        get_log_var : bool
            Whether the latent head produces log-variance values.
        latent_normalization : NormalizationMethod | None
            Normalization applied before latent features are flattened.
        """
        super().__init__()

        self.conv_block = conv_block

        self.latent_head = LatentLayer(
            input_shape=input_shape,
            latent_size=latent_size,
            get_log_var=get_log_var,
            latent_normalization=latent_normalization,
            group_norm_groups=getattr(conv_block.config, "group_norm_groups", 8),
        )

    def forward(self, input: TensorMask) -> LatentVector:
        """
        Encode convolutional features into latent parameters.

        Parameters
        ----------
        input : TensorMask
            Input tensor and optional validity mask.

        Returns
        -------
        LatentVector
            Latent mean and optional log-variance.
        """
        features = self.conv_block(input)
        return self.latent_head(features.tensor)


class MaskPool2d(nn.Module):
    """
    Downsample two-dimensional validity masks.

    Supports retaining a pooled location when any input is valid, when all inputs
    are valid, or when a configurable fraction of inputs are valid.

    Parameters
    ----------
    method : MaskPoolingMethod
        Method used to determine pooled mask validity.
    fraction_threshold : float
        Minimum valid fraction required when ``method`` is ``"fraction"``.
    """

    def __init__(
        self,
        method: MaskPoolingMethod = "any",
        fraction_threshold: float = 0.5,
    ):
        """
        Initialize a two-dimensional mask pooling layer.

        Parameters
        ----------
        method : MaskPoolingMethod
            Mask pooling method to apply.
        fraction_threshold : float
            Minimum valid fraction required for fraction-based pooling.

        Raises
        ------
        ValueError
            If ``fraction_threshold`` is outside the interval [0, 1].
        """
        super().__init__()
        self.method = method
        self.fraction_threshold = fraction_threshold

        if not 0 <= fraction_threshold <= 1:
            raise ValueError("fraction_threshold must be between 0 and 1.")

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Downsample a validity mask.

        Parameters
        ----------
        mask : torch.Tensor
            Input validity mask.

        Returns
        -------
        torch.Tensor
            Downsampled validity mask.

        Raises
        ------
        ValueError
            If the configured mask pooling method is unsupported.
        """
        if self.method == "any":
            return F.max_pool2d(mask, kernel_size=3, stride=2, padding=1)

        if self.method == "all":
            invalid = F.max_pool2d(1.0 - mask, kernel_size=3, stride=2, padding=1)
            return 1.0 - invalid

        if self.method == "fraction":
            sum_pool = (
                F.avg_pool2d(
                    mask,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
                * 9
            )

            count_pool = (
                F.avg_pool2d(
                    torch.ones_like(mask),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
                * 9
            )

            fraction = sum_pool / count_pool
            return (fraction >= self.fraction_threshold).to(mask.dtype)

        raise ValueError(f"Unsupported mask pooling method: {self.method!r}")


class LatentLayer(nn.Module):
    """
    Map convolutional features to latent distribution parameters.

    Features are optionally normalized and activated before flattening. Linear
    heads then produce the latent mean and, optionally, log-variance.

    Parameters
    ----------
    input_shape : tuple[int, int, int]
        Shape of each convolutional feature sample.
    latent_size : int
        Number of latent features produced.
    latent_normalization : NormalizationMethod | None
        Normalization applied to convolutional features before flattening. If
        ``None``, normalization is disabled.
    get_log_var : bool
        Whether to construct a separate latent log-variance head.
    group_norm_groups : int
        Number of groups used when group normalization is selected.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        latent_size: int,
        latent_normalization: NormalizationMethod | None = "layer",
        *,
        get_log_var: bool = True,
        group_norm_groups: int = 8,
    ):
        """
        Initialize the latent projection layer.

        Parameters
        ----------
        input_shape : tuple[int, int, int]
            Shape of each convolutional feature sample.
        latent_size : int
            Number of latent features produced.
        latent_normalization : NormalizationMethod | None
            Normalization applied before flattening.
        get_log_var : bool
            Whether to construct a latent log-variance projection.
        group_norm_groups : int
            Number of groups used when group normalization is selected.
        """
        super().__init__()

        input_channels = input_shape[0]
        self.normalization = (
            nn.Sequential(
                _build_normalization(
                    latent_normalization,
                    input_channels,
                    group_norm_groups=group_norm_groups,
                ),
                nn.ReLU(inplace=True),
            )
            if latent_normalization is not None
            else nn.Identity()
        )

        flattened_size = math.prod(input_shape)
        self.mu = nn.Linear(flattened_size, latent_size)

        self.log_var = nn.Linear(flattened_size, latent_size) if get_log_var else None

    def forward(self, x: torch.Tensor) -> LatentVector:
        """
        Project convolutional features into latent parameters.

        Parameters
        ----------
        x : torch.Tensor
            Convolutional feature tensor.

        Returns
        -------
        LatentVector
            Latent mean and optional log-variance.
        """
        x = self.normalization(x)

        x = x.flatten(start_dim=1)

        mu = self.mu(x)
        log_var = self.log_var(x) if self.log_var is not None else None

        return LatentVector(mu=mu, log_var=log_var)
