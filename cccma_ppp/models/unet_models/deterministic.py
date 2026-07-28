import dataclasses
from dataclasses import field
from typing import ClassVar
import numpy as np
import math
import torch
import torch.nn as nn


from cccma_ppp.core.selectors import deterministicModelSelector
from cccma_ppp.core.deterministic_module import deterministicOutput

from cccma_ppp.models.models_abc import (
    deterministicmodelsABC,
    modelConfigABC,
    DeterministicRequest,
    GENERATORConfig,
)

from cccma_ppp.models.layers.utils import (
    _broadcast_mask,
    _resize_tensor,
)
from cccma_ppp.models.layers.generic import (
    InitMethod,
    UpsamplingMethod,
    MaskPoolingMethod,
    OutputActivation,
    AlignmentMethod,
    NoiseLevel,
)

from cccma_ppp.models.layers.unet import (
    build_conv_block,
    UpBlock,
    DownBlock,
    UNetOutput,
)


from cccma_ppp.models.layers.conv import (
    ConvBlockConfig,
    PartialConvBlockConfig,
    ConvNeXtBlockConfig,
    TensorMask,
)


from cccma_ppp.models.unet_models.utils import _unet_config_checks, _repeat_tensor_mask


@deterministicModelSelector.register("unet")
@dataclasses.dataclass
class UNetConfig(modelConfigABC):
    """
    Configuration for a flexible deterministic UNet supporting Generator for decoder.

    The number of downsampling stages is determined by ``len(channels)``.
    For example, ``channels=[16, 32, 64, 128, 256]`` reproduces the
    encoder widths of the former fixed-depth UNet, with a separate
    bottleneck configured through ``bottleneck_dim``.
    """

    channels: list[int]
    bottleneck_dim: int | None = None
    block_config: ConvBlockConfig | PartialConvBlockConfig | ConvNeXtBlockConfig = (
        field(default_factory=ConvBlockConfig)
    )

    upsampling_method: UpsamplingMethod = "bilinear"
    skip_alignment_method: AlignmentMethod = "padd"
    transpose_kernel_sizes: list[int | tuple[int, int]] | int | None = 3

    process_skip: bool = False

    mask_pooling: MaskPoolingMethod = "any"
    mask_fraction_threshold: float = 0.5

    output_activation: OutputActivation = "identity"
    output_block_hidden_channels: int | None = None

    init_method: InitMethod = "trunc_normal"
    GENERATOR: GENERATORConfig | None = None

    NUM_INPUT_DIMS: ClassVar[int] = 3
    NUM_OUTPUT_DIMS: ClassVar[int] = 3

    def __post_init__(self) -> None:

        _unet_config_checks(self)

        n_up_blocks = len(self.channels) - 1

        if self.transpose_kernel_sizes is None:
            self.transpose_kernel_sizes = [3] * n_up_blocks

        elif isinstance(self.transpose_kernel_sizes, int):
            self.transpose_kernel_sizes = [self.transpose_kernel_sizes] * n_up_blocks

        if len(self.transpose_kernel_sizes) != n_up_blocks:
            raise ValueError(
                "transpose_kernel_sizes must contain one value per "
                f"upsampling stage. Expected {n_up_blocks}, got "
                f"{len(self.transpose_kernel_sizes)}."
            )

        if any(
            not isinstance(kernel, int | tuple) or kernel <= 0
            for kernel in self.transpose_kernel_sizes
        ):
            raise ValueError(
                "All transpose-convolution kernel sizes must be positive integers."
            )

    def build(
        self,
        input_shape: np.ndarray,
        output_shape: np.ndarray | None = None,
        added_features_dim: int | None = None,
    ):
        return UNet(
            config=self,
            input_shape=input_shape,
            output_shape=output_shape,
            added_features_dim=added_features_dim,
        )


class UNet(deterministicmodelsABC):
    def __init__(
        self,
        config: UNetConfig,
        input_shape: np.ndarray | tuple,
        output_shape: np.ndarray | tuple | None = None,
        added_features_dim: int | None = None,
    ):
        super().__init__()

        self.config = config
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
            #     "This UNet implementation preserves spatial resolution. "
            #     f"Input spatial shape {input_shape[-2:]} does not match "
            #     f"output spatial shape {output_shape[-2:]}."
            # )

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
        bottleneck_dim = (
            config.bottleneck_dim
            if config.bottleneck_dim is not None
            else channels[-1] * 2
        )

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
                    process_skip=config.process_skip,
                    mask_pooling=config.mask_pooling,
                    mask_fraction_threshold=config.mask_fraction_threshold,
                )
                for index in range(len(channels) - 1)
            ]
        )

        self.bottleneck = build_conv_block(
            channels[-1],
            bottleneck_dim,
            config.block_config,
        )

        reversed_skips = list(reversed(channels[1:]))
        input_channels = bottleneck_dim
        inject_noise = config.GENERATOR or None
        inject_noise_in_block = True
        if inject_noise:
            inject_noise_in_block = config.GENERATOR.noise_level != "low"

        up_blocks: list[nn.Module] = []
        for index, skip_channels in enumerate(reversed_skips):
            if inject_noise:
                inject_noise = (
                    config.GENERATOR.noise_level != "medium"
                    or index == len(reversed_skips) - 1
                )

            out_channels = skip_channels

            up_blocks.append(
                UpBlock(
                    input_channels=input_channels,
                    skip_channels=skip_channels,
                    out_channels=out_channels,
                    block_config=config.block_config,
                    upsampling_method=config.upsampling_method,
                    skip_alignment_method=config.skip_alignment_method,
                    transpose_kernel_size=config.transpose_kernel_sizes[index],
                    inject_noise=inject_noise,
                    inject_noise_in_block=inject_noise_in_block,
                )
            )
            input_channels = out_channels

        self.up_blocks = nn.ModuleList(up_blocks)

        self.output = UNetOutput(
            in_channels=input_channels,
            out_channels=output_channels,
            hidden_channels=config.output_block_hidden_channels,
            activation=config.output_activation,
        )

        if config.checkpoint_config is not None:
            self._load_state_dict(config.checkpoint_config)
        else:
            self._initialize_weights(config.init_method)

    def _prepare_input(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor | None,
        added_features: torch.Tensor | None,
    ) -> TensorMask:

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

        x = request.input
        x_mask = request.input_mask
        added_features = request.added_features
        num_output_samples = request.output_sample_size

        if self.training and self.config.GENERATOR is not None:
            num_output_samples = self.config.GENERATOR.num_training_noise_samples

        input = self._prepare_input(
            x=x,
            x_mask=x_mask,
            added_features=added_features,
        )

        input = self.initial_mapping(input)
        skips: list[TensorMask] = []

        for down_block in self.down_blocks:
            input, skip = down_block(input)
            skips.append(skip)

        input = self.bottleneck(input)
        batch_size = input.tensor.shape[0]

        if self.config.GENERATOR is not None and num_output_samples is not None:
            input = _repeat_tensor_mask(
                input,
                repeats=num_output_samples,
            )

            skips = [
                _repeat_tensor_mask(skip, repeats=num_output_samples) for skip in skips
            ]

        for up_block, skip in zip(
            self.up_blocks,
            reversed(skips),
            strict=True,
        ):
            input = up_block(input, skip)

        output_tensor = _resize_tensor(
            input.tensor,
            self.output_shape[-2:],
        )

        output = self.output(output_tensor)

        if self.config.GENERATOR is not None:
            output = output.reshape(
                batch_size,
                num_output_samples,
                *output.shape[1:],
            ).transpose(0, 1)

        return deterministicOutput(output=output)
