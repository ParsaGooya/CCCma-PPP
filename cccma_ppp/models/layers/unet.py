import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from cccma_ppp.models.layers.generic import (
    MaskPoolingMethod,
    UpsamplingMethod,
    OutputActivation,
    AlignmentMethod,
)


from cccma_ppp.models.layers.utils import align_to_skip, _noise_injection

from cccma_ppp.models.layers.conv import (
    ConvBlockConfig,
    ConvBlock,
    PartialConvBlockConfig,
    PartialConvBlock,
    ConvNeXtBlockConfig,
    ConvNeXtBlock,
    MaskPool2d,
    TensorMask,
)

from cccma_ppp.models.layers.partialconv2d import PartialConv2d


def build_conv_block(
    in_channels: int,
    out_channels: int,
    config: ConvBlockConfig | PartialConvBlockConfig | ConvNeXtBlockConfig,
    *,
    latent_size: int = None,
    inject_noise: bool = False,
) -> nn.Module:

    effective_config = copy.copy(config)

    effective_config = effective_config.setup_generative(
        latent_size=latent_size, inject_noise=inject_noise
    )

    if isinstance(config, ConvBlockConfig):
        return ConvBlock(
            in_channels,
            out_channels,
            effective_config,
        )

    if isinstance(config, PartialConvBlockConfig):
        return PartialConvBlock(
            in_channels,
            out_channels,
            effective_config,
        )

    if isinstance(config, ConvNeXtBlockConfig):
        return ConvNeXtBlock(
            in_channels,
            out_channels,
            effective_config,
        )

    raise TypeError(
        f"Unsupported UNet block configuration: {type(effective_config).__name__}."
    )


class DownBlock(nn.Module):
    """Feature transformation followed by spatial downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        block_config: ConvBlockConfig | PartialConvBlockConfig | ConvNeXtBlockConfig,
        process_skip: bool,
        mask_pooling: MaskPoolingMethod,
        mask_fraction_threshold: float,
    ):
        super().__init__()

        self._block = build_conv_block(
            in_channels,
            out_channels,
            block_config,
        )
        self.skip_processor = (
            build_conv_block(
                out_channels,
                out_channels,
                block_config,
            )
            if process_skip
            else None
        )

        self.tensor_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mask_pool = MaskPool2d(
            method=mask_pooling,
            fraction_threshold=mask_fraction_threshold,
        )

    def forward(
        self,
        input: TensorMask,
    ) -> tuple[TensorMask, TensorMask]:
        skip = self._block(input)

        if self.skip_processor is not None:
            skip = self.skip_processor(skip)

        pooled_mask = self.mask_pool(skip.mask) if skip.mask is not None else None

        downsampled = TensorMask(
            tensor=self.tensor_pool(skip.tensor),
            mask=pooled_mask,
        )

        return downsampled, skip


class UpBlock(nn.Module):
    """Upsample, concatenate with a skip feature, and transform."""

    def __init__(
        self,
        input_channels: int,
        skip_channels: int,
        out_channels: int,
        *,
        block_config: ConvBlockConfig | PartialConvBlockConfig | ConvNeXtBlockConfig,
        upsampling_method: UpsamplingMethod,
        skip_alignment_method: AlignmentMethod,
        transpose_kernel_size: int,
        inject_noise: bool = False,
        inject_noise_in_block: bool = False,
    ):
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

        self._block = build_conv_block(
            skip_channels + out_channels,
            out_channels,
            effective_block_config,
            inject_noise=(self.inject_noise and self.inject_noise_in_block),
        )

    def forward(
        self,
        input: TensorMask,
        skip: TensorMask,
    ) -> TensorMask:
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

        x = align_to_skip(
            x, skip.tensor, self.skip_alignment_method, self.skip_padding_method
        )

        merged_tensor = torch.cat([skip.tensor, x], dim=1)

        return self._block(
            TensorMask(
                tensor=merged_tensor,
                mask=None,
            )
        )


class UNetOutput(nn.Module):
    """Final channel projection and optional output activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        hidden_channels: int | None,
        activation: OutputActivation,
    ):
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
        return self.layers(x)
