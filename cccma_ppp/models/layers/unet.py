import torch
import torch.nn as nn
import torch.nn.functional as F

from cccma_ppp.models.layers import (MaskPoolingMode,
                                    UpsamplingMethod,
                                    OutputActivation,
                                    AlignmentMode,
                                    PaddingMode)


from cccma_ppp.models.layers.utils import (align_to_skip,
                                           _resize_mask,
                                           _merge_masks)

from cccma_ppp.models.layers.conv import (ConvBlockConfig,
                                          ConvBlock,
                                          PartialConvBlockConfig,
                                          PartialConvBlock,
                                          ConvNeXtBlockConfig,
                                          ConvNeXtBlock,
                                          MaskPool2d,
                                          TensorMask)



def build_conv_block(
    in_channels: int,
    out_channels: int,
    config: ConvBlockConfig | PartialConvBlockConfig | ConvNeXtBlockConfig,
) -> nn.Module:
    if isinstance(config, ConvBlockConfig):
        return ConvBlock(
            in_channels,
            out_channels,
            config,
        )

    if isinstance(config, PartialConvBlockConfig):
        return PartialConvBlock(
            in_channels,
            out_channels,
            config,
        )

    if isinstance(config, ConvNeXtBlockConfig):
        return ConvNeXtBlock(
            in_channels,
            out_channels,
            config,
        )

    raise TypeError(
        f"Unsupported UNet block configuration: {type(config).__name__}."
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
        mask_pooling: MaskPoolingMode,
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
            mode=mask_pooling,
            fraction_threshold=mask_fraction_threshold,
        )

    def forward(
        self,
        input: TensorMask,
    ) -> tuple[TensorMask, TensorMask]:
        skip = self._block(input)

        if self.skip_processor is not None:
            skip = self.skip_processor(skip)

        pooled_mask = (
            self.mask_pool(skip.mask)
            if skip.mask is not None
            else None
        )

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
        skip_alignment_mode: AlignmentMode,
        padding_mode: PaddingMode,
        transpose_kernel_size: int,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        self.upsampling_method = upsampling_method
        self.skip_alignment_mode = skip_alignment_mode
        self.padding_mode = padding_mode

        if upsampling_method == "transpose_conv":
            self.upsample = nn.ConvTranspose2d(
                input_channels,
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
            self.channel_projection = nn.Conv2d(
                input_channels,
                out_channels,
                kernel_size=3,
            )

        else:
            raise ValueError(
                f"Unsupported upsampling mode: {upsampling_method!r}."
            )

        self._block = build_conv_block(
            skip_channels + out_channels,
            out_channels,
            block_config,
        )

    def forward(
        self,
        input: TensorMask,
        skip: TensorMask,
    ) -> TensorMask:
        x = self.upsample(input.tensor)
        x = self.channel_projection(x)

        target_size = skip.tensor.shape[-2:]
        x = align_to_skip(x, 
                          skip.tensor,
                          self.skip_alignment_mode,
                          self.padding_mode)

        input_mask = _resize_mask(input.mask, target_size)


        merged_tensor = torch.cat([skip.tensor, x], dim=1)
        merged_mask = _merge_masks(
            input_mask,
            skip.mask,
            out_channels=self.out_channels,
            skip_channels=self.skip_channels,
            spatial_size=target_size,
            reference=merged_tensor,
        )

        return self._block(
            TensorMask(
                tensor=merged_tensor,
                mask=merged_mask,
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
                nn.Conv2d(
                    in_channels,
                    hidden_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
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
            raise ValueError(
                f"Unsupported output activation: {activation!r}."
            )

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

