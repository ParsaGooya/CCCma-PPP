from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F


InitMethod = Literal["default", "kaiming", "xavier", "trunc_normal"]
ActivationName = Literal["relu", "gelu", "silu"]
NormalizationMethod = Literal["batch", "group", "layer", "none"]
UpsamplingMethod = Literal["transpose_conv", "bilinear", "pixelshuffle"]
OutputActivation = Literal["identity", "sigmoid", "tanh"]
MaskPoolingMethod = Literal["any", "all", "fraction"]
AlignmentMethod = Literal["interpolation", "padd", "strict"]
PaddingMethod = Literal["zeros", "reflect", "replicate", "circular"]
NoiseLevel = Literal["full", "medium", "low"]


def _validate_dropout(value: float | None) -> None:
    """
    Validate a dropout probability.

    Parameters
    ----------
    value : float | None
        Dropout probability to validate. ``None`` indicates that dropout is
        disabled.

    Raises
    ------
    ValueError
        If ``value`` is not ``None`` and is outside the interval [0, 1].
    """
    if value is not None and not 0 <= value <= 1:
        raise ValueError("Dropout rates must be between 0 and 1.")


def _build_activation(name: ActivationName) -> nn.Module:
    """
    Build an activation layer from its configured name.

    Parameters
    ----------
    name : ActivationName
        Name of the activation function to construct.

    Returns
    -------
    nn.Module
        Activation layer corresponding to ``name``.

    Raises
    ------
    ValueError
        If ``name`` is not a supported activation.
    """
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
    """
    Build a normalization layer for two-dimensional feature maps.

    Parameters
    ----------
    name : NormalizationMethod
        Name of the normalization method to construct.
    channels : int
        Number of feature channels to normalize.
    group_norm_groups : int
        Maximum number of groups to use for group normalization. The number of
        groups is reduced when necessary so that it divides ``channels``.

    Returns
    -------
    nn.Module
        Normalization layer corresponding to ``name``.

    Raises
    ------
    ValueError
        If ``name`` is not a supported normalization method.
    """
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
    """
    Apply layer normalization to channel-first two-dimensional feature maps.

    Normalization is performed over the channel dimension at each spatial
    location while preserving the input shape.

    Parameters
    ----------
    num_channels : int
        Number of channels in the input feature maps.
    eps : float
        Small value added for numerical stability.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        """
        Initialize a channel-wise layer normalization module.

        Parameters
        ----------
        num_channels : int
            Number of channels in the input feature maps.
        eps : float
            Small value added for numerical stability.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization to a channel-first feature map.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape ``(N, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Normalized tensor with the same shape as the input.
        """

        return F.layer_norm(
            x.permute(0, 2, 3, 1),
            (x.shape[1],),
            self.weight,
            self.bias,
            self.eps,
        ).permute(0, 3, 1, 2)


class DropPath(nn.Module):
    """
    Apply stochastic depth independently to each sample.

    During training, the input is randomly dropped on a per-sample basis and
    surviving samples are rescaled to preserve their expected magnitude. During
    evaluation, the input is returned unchanged.

    Parameters
    ----------
    drop_probability : float
        Probability of dropping each sample.
    """

    def __init__(self, drop_probability: float = 0.0):
        """
        Initialize a stochastic depth module.

        Parameters
        ----------
        drop_probability : float
            Probability of dropping each sample during training.
        """
        super().__init__()
        self.drop_probability = drop_probability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply stochastic depth to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor whose first dimension represents the batch.

        Returns
        -------
        torch.Tensor
            Tensor with randomly dropped and rescaled samples during training, or
            the unchanged input during evaluation.
        """
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
