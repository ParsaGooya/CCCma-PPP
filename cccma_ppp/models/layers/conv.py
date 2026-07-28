import dataclasses
from typing import Literal, final
import abc

import torch
import torch.nn as nn
import torch.nn.functional as F


from cccma_ppp.models.layers.partialconv2d import PartialConv2d
from cccma_ppp.models.layers import (
    MaskPoolingMethod,
    NormalizationMethod,
    ActivationName,
    PaddingMethod,
    _build_normalization,
    _build_activation,
    _same_padding,
    _broadcast_mask,
    _validate_dropout,
    LayerNorm2d,
    DropPath,
)

from cccma_ppp.models.layers.utils import _noise_injection, _expand_mask


@dataclasses.dataclass
class TensorMask:
    """Tensor and optional validity mask propagated through the conv models."""

    tensor: torch.Tensor
    mask: torch.Tensor | None = None


class ConvBlockConfigABC(abc.ABC):
    def __init__(self):
        self.latent_size: int | None = None
        self.inject_noise: bool = False

    @final
    def setup_generative(
        self, latent_size: int | None = None, inject_noise: bool = False
    ):

        self.latent_size = latent_size
        self.inject_noise = inject_noise

        return self


@dataclasses.dataclass
class ConvBlockConfig(ConvBlockConfigABC):
    """Configuration for a conventional repeated-convolution block."""

    name: Literal["standard_conv"]
    num_convolutions: int = 2
    kernel_size: int = 3
    normalization: NormalizationMethod = "batch"
    padding_method: PaddingMethod = "circular"
    activation: ActivationName = "relu"
    dropout_rate: float | None = None
    bias: bool = False
    group_norm_groups: int = 8

    def __post_init__(self) -> None:
        super().__init__()

        if self.num_convolutions < 1:
            raise ValueError("num_convolutions must be at least 1.")
        _same_padding(self.kernel_size)
        _validate_dropout(self.dropout_rate)


@dataclasses.dataclass
class PartialConvBlockConfig(ConvBlockConfigABC):
    """Configuration for a repeated partial-convolution block."""

    name: Literal["partial_conv"]
    num_convolutions: int = 2
    kernel_size: int = 3
    normalization: NormalizationMethod = "batch"
    padding_method: PaddingMethod = "circular"
    activation: ActivationName = "relu"
    dropout_rate: float | None = None
    bias: bool = False
    group_norm_groups: int = 8

    multi_channel: bool = dataclasses.field(init=False, default=True)
    return_mask: bool = dataclasses.field(init=False, default=True)

    def __post_init__(self) -> None:
        super().__init__()

        if self.num_convolutions < 1:
            raise ValueError("num_convolutions must be at least 1.")
        _same_padding(self.kernel_size)
        _validate_dropout(self.dropout_rate)


@dataclasses.dataclass
class ConvNeXtBlockConfig(ConvBlockConfigABC):
    """Configuration for a repeated ConvNeXt-style residual block."""

    name: Literal["convnext"]
    num_blocks: int = 2
    kernel_size: int = 7
    expansion_ratio: int = 4
    padding_method: PaddingMethod = "circular"
    layer_scale_init: float = 1e-6
    dropout_rate: float = 0.0
    drop_path_rate: float = 0.0
    use_partial_conv: bool = True

    multi_channel: bool = dataclasses.field(init=False, default=True)
    return_mask: bool = dataclasses.field(init=False, default=True)

    def __post_init__(self) -> None:
        super().__init__()

        if self.num_blocks < 1:
            raise ValueError("num_blocks must be at least 1.")
        if self.expansion_ratio < 1:
            raise ValueError("expansion_ratio must be at least 1.")
        _same_padding(self.kernel_size)
        _validate_dropout(self.dropout_rate)
        _validate_dropout(self.drop_path_rate)


class ConvSingle(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: ConvBlockConfig,
    ):
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
        if self.inject_noise:
            x = _noise_injection(x)

        x = self.conv(x)
        x = self.normalization(x)
        x = self.activation(x)
        return self.dropout(x)


class PartialConvSingle(nn.Module):
    """One PartialConv2d + normalization + activation stage."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: PartialConvBlockConfig,
    ):
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
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if self.inject_noise:
            x = _noise_injection(x)
            if self.multi_channel:
                mask = _expand_mask(x, mask)

        if self.return_mask:
            x, mask = self.conv(x, mask)
        else:
            x = self.conv(x, mask)

        x = self.normalization(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x, mask


class ConvNeXtSingle(nn.Module):
    """A compact NCHW ConvNeXt-style block."""

    def __init__(
        self,
        channels: int,
        config: ConvNeXtBlockConfig,
        drop_path_rate: float,
    ):
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
    ) -> tuple[torch.Tensor, torch.Tensor]:

        residual = x

        if self.use_partial_conv:
            if self.return_mask:
                x, mask = self.depthwise(x, mask)
            else:
                x = self.depthwise(x, mask)
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
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: ConvBlockConfig,
    ):
        super().__init__()
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
        x = input.tensor

        for stage in self.stages:
            x = stage(x)

        return TensorMask(
            tensor=x,
            mask=input.mask,
        )


class PartialConvBlock(nn.Module):
    """Repeated PartialConv2d block that updates the validity mask."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: PartialConvBlockConfig,
    ):
        super().__init__()
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
        mask = _broadcast_mask(input.mask, input.tensor)

        x = input.tensor
        for stage in self.stages:
            x, mask = stage(x, mask)

        return TensorMask(tensor=x, mask=mask)


class ConvNeXtBlock(nn.Module):
    """Channel projection followed by one or more ConvNeXt blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: ConvNeXtBlockConfig,
    ):
        super().__init__()

        self.out_channels = out_channels
        self.use_partial_conv = config.use_partial_conv
        self.requires_projection = in_channels != out_channels

        if self.requires_projection:
            if self.use_partial_conv:
                self.projection_conv = PartialConv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    multi_channel=True,
                    return_mask=True,
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
        x = input.tensor
        mask = input.mask

        if self.requires_projection:
            if self.use_partial_conv:
                x, mask = self.projection_conv(x, mask)
            else:
                x = self.projection_conv(x)

            x = self.projection_norm(x)

        for block in self.blocks:
            x, mask = block(x, mask)

        return TensorMask(
            tensor=x,
            mask=mask,
        )


class MaskPool2d(nn.Module):
    """Pool masks according to configurable validity semantics."""

    def __init__(
        self,
        method: MaskPoolingMethod = "any",
        fraction_threshold: float = 0.5,
    ):
        super().__init__()
        self.method = method
        self.fraction_threshold = fraction_threshold

        if not 0 <= fraction_threshold <= 1:
            raise ValueError("fraction_threshold must be between 0 and 1.")

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        if self.method == "any":
            return F.max_pool2d(mask, kernel_size=2, stride=2)

        if self.method == "all":
            invalid = F.max_pool2d(
                1.0 - mask,
                kernel_size=2,
                stride=2,
            )
            return 1.0 - invalid

        if self.method == "fraction":
            fraction = F.avg_pool2d(mask, kernel_size=2, stride=2)
            return (fraction >= self.fraction_threshold).to(mask.dtype)

        raise ValueError(f"Unsupported mask pooling method: {self.method!r}")
