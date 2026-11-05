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
        Dropout probability to validate. If ``None``, dropout is considered
        disabled.

    Raises
    ------
    ValueError
        If ``value`` is outside the interval [0, 1].
    """
    if value is not None and not 0 <= value <= 1:
        raise ValueError("Dropout rates must be between 0 and 1.")


def _build_activation(name: ActivationName) -> nn.Module:
    """
    Build an activation module from its configured name.

    Parameters
    ----------
    name : ActivationName
        Activation function to construct.

    Returns
    -------
    nn.Module
        Configured activation module.

    Raises
    ------
    ValueError
        If ``name`` is not a supported activation function.
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
    Build a normalization module.

    For group normalization, the requested number of groups is reduced when
    necessary until it evenly divides the number of channels.

    Parameters
    ----------
    name : NormalizationMethod
        Normalization method to construct.
    channels : int
        Number of feature channels to normalize.
    group_norm_groups : int
        Maximum number of groups used for group normalization.

    Returns
    -------
    nn.Module
        Configured normalization module.

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

    Normalization is performed over the channel dimension independently at each
    spatial location.

    Parameters
    ----------
    num_channels : int
        Number of feature channels.
    eps : float
        Small value added for numerical stability during normalization.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        """
        Initialize two-dimensional layer normalization.

        Parameters
        ----------
        num_channels : int
            Number of feature channels.
        eps : float
            Small value added for numerical stability during normalization.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization to a feature tensor.

        Parameters
        ----------
        x : torch.Tensor
            Channel-first tensor with shape ``(batch, channels, height, width)``.

        Returns
        -------
        torch.Tensor
            Layer-normalized tensor with the same shape as the input.
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
    Apply stochastic depth to a residual branch.

    During training, complete residual paths are independently dropped for each
    batch element and retained paths are rescaled to preserve their expected
    magnitude.

    Parameters
    ----------
    drop_probability : float
        Probability of dropping a residual path.
    """

    def __init__(self, drop_probability: float = 0.0):
        """
        Initialize stochastic depth.

        Parameters
        ----------
        drop_probability : float
            Probability of dropping a residual path during training.
        """
        super().__init__()
        self.drop_probability = drop_probability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply stochastic depth to an input tensor.

        During evaluation, or when the drop probability is zero, the input is
        returned unchanged.

        Parameters
        ----------
        x : torch.Tensor
            Input residual tensor.

        Returns
        -------
        torch.Tensor
            Tensor with stochastic depth applied during training.
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
