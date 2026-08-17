from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F


InitMethod = Literal["default", "kaiming", "xavier", "trunc_normal"]
ActivationName = Literal["relu", "gelu", "silu"]
NormalizationMethod = Literal["batch", "group", "layer", "none"]
UpsamplingMethod = Literal["transpose_conv", "bilinear"]
OutputActivation = Literal["identity", "sigmoid", "tanh"]
MaskPoolingMethod = Literal["any", "all", "fraction"]
AlignmentMethod = Literal["interpolation", "padd", "strict"]
PaddingMethod = Literal["zeros", "reflect", "replicate", "circular"]
NoiseLevel = Literal["full", "medium", "low"]


def _validate_dropout(value: float | None) -> None:
    if value is not None and not 0 <= value <= 1:
        raise ValueError("Dropout rates must be between 0 and 1.")


def _build_activation(name: ActivationName) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name!r}")


def _build_normalization(
    name: NormalizationMethod,
    channels: int,
    *,
    group_norm_groups: int = 8,
) -> nn.Module:
    if name == "batch":
        return nn.BatchNorm2d(channels)

    if name == "group":
        groups = min(group_norm_groups, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, channels)

    if name == "layer":
        return LayerNorm2d(channels)

    if name == "none":
        return nn.Identity()

    raise ValueError(f"Unsupported normalization: {name!r}")


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for NCHW tensors."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.permute(0, 2, 3, 1),
            (x.shape[1],),
            self.weight,
            self.bias,
            self.eps,
        ).permute(0, 3, 1, 2)


class DropPath(nn.Module):
    """Per-sample stochastic depth."""

    def __init__(self, drop_probability: float = 0.0):
        super().__init__()
        self.drop_probability = drop_probability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_probability == 0.0 or not self.training:
            return x

        keep_probability = 1.0 - self.drop_probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_probability + torch.rand(
            shape,
            dtype=x.dtype,
            device=x.device,
        )
        random_tensor.floor_()
        return x * random_tensor / keep_probability
