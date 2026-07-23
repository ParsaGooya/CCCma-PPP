import dataclasses
from dataclasses import field
from typing import ClassVar, Literal, Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from cccma_ppp.models.layers.partialconv2d import PartialConv2d
from cccma_ppp.models.layers import (MaskPoolingMode,
                                     NormalizationMethod,
                                     ActivationName,
                                     _build_normalization,
                                     _build_activation,
                                     _same_padding,
                                     _broadcast_mask,
                                     _validate_dropout,
                                     LayerNorm2d,
                                     DropPath,)



@dataclasses.dataclass
class TensorMask:
    """Tensor and optional validity mask propagated through the conv models."""

    tensor: torch.Tensor
    mask: torch.Tensor | None = None




@dataclasses.dataclass
class ConvBlockConfig:
    """Configuration for a conventional repeated-convolution block."""

    name: Literal["standard_conv"] 
    num_convolutions: int = 2
    kernel_size: int = 3
    normalization: NormalizationMethod = "batch"
    activation: ActivationName = "relu"
    dropout_rate: float | None = None
    bias: bool = False
    group_norm_groups: int = 8

    def __post_init__(self) -> None:
        if self.num_convolutions < 1:
            raise ValueError("num_convolutions must be at least 1.")
        _same_padding(self.kernel_size)
        _validate_dropout(self.dropout_rate)


@dataclasses.dataclass
class PartialConvBlockConfig:
    """Configuration for a repeated partial-convolution block."""

    name: Literal["partial_conv"] 
    num_convolutions: int = 2
    kernel_size: int = 3
    normalization: NormalizationMethod = "batch"
    activation: ActivationName = "relu"
    dropout_rate: float | None = None
    bias: bool = False
    group_norm_groups: int = 8

    def __post_init__(self) -> None:
        if self.num_convolutions < 1:
            raise ValueError("num_convolutions must be at least 1.")
        _same_padding(self.kernel_size)
        _validate_dropout(self.dropout_rate)


@dataclasses.dataclass
class ConvNeXtBlockConfig:
    """Configuration for a repeated ConvNeXt-style residual block."""

    name: Literal["convnext"] 
    num_blocks: int = 2
    kernel_size: int = 7
    expansion_ratio: int = 4
    layer_scale_init: float = 1e-6
    dropout_rate: float = 0.0
    drop_path_rate: float = 0.0
    use_partial_conv: bool = True

    def __post_init__(self) -> None:
        if self.num_blocks < 1:
            raise ValueError("num_blocks must be at least 1.")
        if self.expansion_ratio < 1:
            raise ValueError("expansion_ratio must be at least 1.")
        _same_padding(self.kernel_size)
        _validate_dropout(self.dropout_rate)
        _validate_dropout(self.drop_path_rate)



class PartialConvSingle(nn.Module):
    """One PartialConv2d + normalization + activation stage."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: PartialConvBlockConfig,
    ):
        super().__init__()
        self.conv = PartialConv2d(
            in_channels,
            out_channels,
            kernel_size=config.kernel_size,
            padding=_same_padding(config.kernel_size),
            bias=config.bias,
            multi_channel=True,
            return_mask=True,
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
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, mask = self.conv(x, mask)
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

        if config.use_partial_conv:
            self.depthwise = PartialConv2d(
                            channels,
                            channels,
                            kernel_size=config.kernel_size,
                            padding=_same_padding(config.kernel_size),
                            groups=channels,
                            multi_channel=True,
                            return_mask=True,
            )

        else:
            self.depthwise = nn.Conv2d(
                channels,
                channels,
                kernel_size=config.kernel_size,
                padding=_same_padding(config.kernel_size),
                groups=channels,
            )

        self.normalization = LayerNorm2d(channels)
        self.pointwise_1 = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.activation = nn.GELU()
        self.dropout = (
            nn.Dropout2d(config.dropout_rate)
            if config.dropout_rate > 0
            else nn.Identity()
        )
        self.pointwise_2 = nn.Conv2d(hidden_channels, channels, kernel_size=1)

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
            x, mask = self.depthwise(x, mask)
        else:
            x = self.depthwise(x)

        x = self.normalization(x)
        x = self.pointwise_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.pointwise_2(x)

        if self.layer_scale is not None:
            x = x * self.layer_scale.view(1, -1, 1, 1)
        
        return residual + self.drop_path(x), mask

    


class ConvBlock(nn.Module):
    """Conventional convolution block with optional normalization/dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: ConvBlockConfig,
    ):
        super().__init__()
        self.out_channels = out_channels

        layers: list[nn.Module] = []
        current_channels = in_channels
        padding = _same_padding(config.kernel_size)

        for _ in range(config.num_convolutions):
            layers.append(
                nn.Conv2d(
                    current_channels,
                    out_channels,
                    kernel_size=config.kernel_size,
                    padding=padding,
                    bias=config.bias,
                )
            )
            layers.append(
                _build_normalization(
                    config.normalization,
                    out_channels,
                    group_norm_groups=config.group_norm_groups,
                )
            )
            layers.append(_build_activation(config.activation))

            if config.dropout_rate is not None and config.dropout_rate > 0:
                layers.append(nn.Dropout2d(config.dropout_rate))

            current_channels = out_channels

        self.layers = nn.Sequential(*layers)

    def forward(self, input: TensorMask) -> TensorMask:
        return TensorMask(
            tensor=self.layers(input.tensor),
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

        if mask is None:
            mask = torch.ones_like(input.tensor)

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

    def forward(self, input: TensorMask ) -> TensorMask:
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
        mode: MaskPoolingMode = "any",
        fraction_threshold: float = 0.5,
    ):
        super().__init__()
        self.mode = mode
        self.fraction_threshold = fraction_threshold

        if not 0 <= fraction_threshold <= 1:
            raise ValueError(
                "fraction_threshold must be between 0 and 1."
            )

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        if self.mode == "any":
            return F.max_pool2d(mask, kernel_size=2, stride=2)

        if self.mode == "all":
            invalid = F.max_pool2d(
                1.0 - mask,
                kernel_size=2,
                stride=2,
            )
            return 1.0 - invalid

        if self.mode == "fraction":
            fraction = F.avg_pool2d(mask, kernel_size=2, stride=2)
            return (fraction >= self.fraction_threshold).to(mask.dtype)

        raise ValueError(f"Unsupported mask pooling mode: {self.mode!r}")