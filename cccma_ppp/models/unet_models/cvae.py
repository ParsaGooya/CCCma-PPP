import dataclasses
from dataclasses import field
from typing import ClassVar
import numpy as np
import math
import numpy as np
import torch
import torch.nn as nn


from cccma_ppp.core.selectors import cVAEModelSelector
from cccma_ppp.core.cVAE_module import cVAEOutput
from cccma_ppp.models.models_abc import (
    cVAEmodelConfigABC,
    cVAEmodelsABC,
)

from cccma_ppp.models.layers import (
    _broadcast_mask,
    _resize_tensor,
    InitMethod,
    UpsamplingMethod,
    MaskPoolingMode,
    OutputActivation,
    AlignmentMode,
    PaddingMode,
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

from cccma_ppp.models.unet_models.utils import _unet_config_checks


@cVAEModelSelector.register("unet")
@dataclasses.dataclass
class cVAEUNetConfig(cVAEmodelConfigABC):
    """
    Configuration for a flexible cVAE UNet.

    The number of downsampling stages is determined by ``len(channels)``.
    For example, ``channels=[16, 32, 64, 128, 256]`` reproduces the
    encoder widths of the former fixed-depth UNet, with a separate
    bottleneck configured through ``bottleneck_dim``.
    """

    channels: list[int]
    latent_size: int | None = None

    condition_embedding_channels: list = None
    condition_embedding_size: int
    condition_dependant_latent: bool
    condemb_to_decoder: bool = True

    block_config: ConvBlockConfig | PartialConvBlockConfig | ConvNeXtBlockConfig = (
        field(default_factory=ConvBlockConfig)
    )

    upsampling_method: UpsamplingMethod = "bilinear"
    transpose_kernel_sizes: list[int | tuple[int, int]] | int | None = 3

    add_skip_latent: bool = False

    mask_pooling: MaskPoolingMode = "any"
    mask_fraction_threshold: float = 0.5

    output_activation: OutputActivation = "identity"
    output_hidden_channels: int | None = None

    init_method: InitMethod = "trunc_normal"

    NUM_INPUT_DIMS: ClassVar[int] = 3
    NUM_OUTPUT_DIMS: ClassVar[int] = 3
    GENERATOR: ClassVar[bool] = False

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
        return cVAEUNet(
            config=self,
            input_shape=input_shape,
            output_shape=output_shape,
            added_features_dim=added_features_dim,
        )
