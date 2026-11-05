import torch
import torch.nn.functional as F

from S2D4P.architectures.layers.generic import AlignmentMethod, PaddingMethod


def _same_padding(kernel_size: int) -> int:
    """
    Calculate padding that preserves spatial size for an odd kernel.

    Parameters
    ----------
    kernel_size : int
        Convolution kernel size.

    Returns
    -------
    int
        Symmetric padding size required to preserve the spatial dimensions.

    Raises
    ------
    ValueError
        If ``kernel_size`` is nonpositive or even.
    """
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("Feature-block kernel sizes must be positive odd integers.")
    return kernel_size // 2


def align_to_skip(
    x: torch.Tensor,
    skip_shape: torch.Tensor,
    mode: AlignmentMethod = "padd",
    padding_mode: PaddingMethod = "zeros",
) -> torch.Tensor:
    """
    Align a tensor with a target spatial shape.

    The tensor can be aligned by interpolation or padding, or strict shape
    matching can be enforced.

    Parameters
    ----------
    x : torch.Tensor
        Tensor whose spatial dimensions are aligned.
    skip_shape : tuple[int, int]
        Target height and width.
    mode : AlignmentMethod
        Method used to align the spatial dimensions.
    padding_mode : PaddingMethod
        Padding mode used when ``mode`` is ``"padd"``.

    Returns
    -------
    torch.Tensor
        Tensor aligned to ``skip_shape``.

    Raises
    ------
    RuntimeError
        If strict alignment is requested and the spatial dimensions do not match.
    ValueError
        If ``mode`` is unsupported.
    """
    if x.shape[-2:] == skip_shape:
        return x

    if mode == "resize":
        return F.interpolate(
            x,
            size=skip_shape,
            mode="bilinear",
            align_corners=False,
        )

    if mode == "padd":
        return padd(x, skip_shape, padding_mode)

    if mode == "strict":
        raise RuntimeError(
            "Decoder and skip features have incompatible spatial "
            f"shapes: {x.shape[-2:]} and {skip_shape}."
        )

    raise ValueError(f"Unknown alignment mode: {mode}")


def padd(
    x: torch.Tensor, target_size: tuple[int, int], padding_mode: PaddingMethod = "zeros"
) -> torch.Tensor:
    """
    Pad a tensor symmetrically to a target spatial size.

    Any odd amount of required padding places the additional element on the
    right or bottom side.

    Parameters
    ----------
    x : torch.Tensor
        Tensor to pad.
    target_size : tuple[int, int]
        Target height and width.
    padding_mode : PaddingMethod
        Padding mode to apply. ``"zeros"`` uses constant zero padding.

    Returns
    -------
    torch.Tensor
        Tensor padded to the requested spatial size.
    """
    mode = "constant" if padding_mode == "zeros" else padding_mode
    target_h, target_w = target_size
    current_h, current_w = x.shape[-2:]

    pad_h = target_h - current_h
    pad_w = target_w - current_w

    return F.pad(
        x,
        [
            pad_w // 2,
            pad_w - pad_w // 2,
            pad_h // 2,
            pad_h - pad_h // 2,
        ],
        mode=mode,
    )


def _resize_mask(
    mask: torch.Tensor | None,
    size: tuple[int, int],
) -> torch.Tensor | None:
    """
    Resize a validity mask using nearest-neighbor interpolation.

    Parameters
    ----------
    mask : torch.Tensor | None
        Validity mask to resize. If ``None``, no resizing is performed.
    size : tuple[int, int]
        Target height and width.

    Returns
    -------
    torch.Tensor | None
        Resized mask with its original data type restored, or ``None`` when no
        mask is provided.
    """
    if mask is None:
        return None

    if mask.shape[-2:] == size:
        return mask

    original_dtype = mask.dtype
    resized = F.interpolate(
        mask.float(),
        size=size,
        mode="nearest",
    )
    return resized.to(original_dtype)


def _broadcast_mask(
    mask: torch.Tensor | None,
    reference: torch.Tensor,
) -> torch.Tensor | None:
    """
    Align a validity mask with a reference tensor.

    Two- and three-dimensional masks are expanded to four dimensions, singleton
    batch dimensions are broadcast when necessary, and spatial dimensions are
    resized to match the reference tensor.

    Parameters
    ----------
    mask : torch.Tensor | None
        Validity mask to align with the reference tensor.
    reference : torch.Tensor
        Reference tensor defining the required batch size, spatial dimensions,
        device, and data type.

    Returns
    -------
    torch.Tensor | None
        Mask aligned with the reference tensor, or ``None`` when no mask is
        provided.

    Raises
    ------
    ValueError
        If the mask is not two-, three-, or four-dimensional, or its batch
        dimension cannot be matched to the reference tensor.
    """
    if mask is None:
        return None

    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)

    elif mask.ndim == 3:
        mask = mask.unsqueeze(0)

    if mask.ndim != 4:
        raise ValueError(
            f"Expected a 2D, 3D, or 4D mask, got shape {tuple(mask.shape)}."
        )

    if mask.shape[0] == 1 and reference.shape[0] > 1:
        mask = mask.expand(reference.shape[0], -1, -1, -1)

    if mask.shape[0] != reference.shape[0]:
        raise ValueError(
            "Mask batch dimension does not match tensor batch dimension: "
            f"{mask.shape[0]} != {reference.shape[0]}."
        )

    if mask.shape[-2:] != reference.shape[-2:]:
        mask = _resize_mask(mask, reference.shape[-2:])

    return mask.to(device=reference.device, dtype=reference.dtype)


def _merge_masks(
    input_mask: torch.Tensor | None,
    skip_mask: torch.Tensor | None,
    *,
    out_channels: int,
    skip_channels: int,
    spatial_size: tuple[int, int],
    reference: torch.Tensor,
) -> torch.Tensor | None:
    """
    Merge input and skip-connection validity masks.

    Both masks are resized to the requested spatial dimensions. A missing mask is
    replaced with an all-valid mask having the corresponding number of channels.

    Parameters
    ----------
    input_mask : torch.Tensor | None
        Validity mask associated with the decoder input.
    skip_mask : torch.Tensor | None
        Validity mask associated with the skip connection.
    out_channels : int
        Number of decoder-output channels represented by ``input_mask``.
    skip_channels : int
        Number of skip-connection channels represented by ``skip_mask``.
    spatial_size : tuple[int, int]
        Target height and width for both masks.
    reference : torch.Tensor
        Reference tensor defining the batch size, device, and data type used when
        constructing missing masks.

    Returns
    -------
    torch.Tensor | None
        Channel-wise concatenation of the skip and input masks, or ``None`` when
        both masks are absent.
    """
    if input_mask is None and skip_mask is None:
        return None

    input_mask = _resize_mask(input_mask, spatial_size)
    skip_mask = _resize_mask(skip_mask, spatial_size)

    if input_mask is None:
        input_mask = torch.ones(
            reference.shape[0],
            out_channels,
            *spatial_size,
            device=reference.device,
            dtype=reference.dtype,
        )

    if skip_mask is None:
        skip_mask = torch.ones(
            reference.shape[0],
            skip_channels,
            *spatial_size,
            device=reference.device,
            dtype=reference.dtype,
        )

    return torch.cat([skip_mask, input_mask], dim=1)


def _resize_tensor(
    x: torch.Tensor,
    size: tuple[int, int],
    *,
    mode: str = "bilinear",
) -> torch.Tensor:
    """
    Resize a tensor to a target spatial size.

    Parameters
    ----------
    x : torch.Tensor
        Tensor to resize.
    size : tuple[int, int]
        Target height and width.
    mode : str
        Interpolation mode. Bilinear and bicubic interpolation disable corner
        alignment.

    Returns
    -------
    torch.Tensor
        Tensor with the requested spatial dimensions.
    """
    if x.shape[-2:] == size:
        return x

    if mode in {"bilinear", "bicubic"}:
        return F.interpolate(
            x,
            size=size,
            mode=mode,
            align_corners=False,
        )

    return F.interpolate(x, size=size, mode=mode)


def _sample(mu, var, sample_size=1, std=1):
    """
    Draw samples around a mean using element-wise variance.

    Parameters
    ----------
    mu : torch.Tensor
        Mean of the sampling distribution.
    var : torch.Tensor
        Element-wise variance of the sampling distribution.
    sample_size : int
        Number of samples to draw.
    std : float
        Standard deviation of the standard-normal noise before scaling by
        ``sqrt(var)``.

    Returns
    -------
    torch.Tensor
        Sampled tensor with a leading sample dimension.
    """
    out = mu + torch.sqrt(var) * _get_normal(var, std).sample((sample_size,))

    return out


def _get_normal(ref_tensor, std=1):
    """
    Construct a zero-mean normal distribution matching a reference tensor.

    Parameters
    ----------
    ref_tensor : torch.Tensor
        Reference tensor defining the shape, data type, and device of the
        distribution parameters.
    std : float
        Standard deviation of the normal distribution.

    Returns
    -------
    torch.distributions.Normal
        Element-wise zero-mean normal distribution.
    """
    return torch.distributions.Normal(
        torch.zeros_like(ref_tensor), torch.ones_like(ref_tensor) * std
    )


def _noise_injection(ref_tensor: torch.Tensor):
    """
    Append a random noise channel to a feature tensor.

    One independent standard-normal noise channel is generated with the same
    batch and spatial dimensions as the reference tensor and concatenated along
    the channel dimension.

    Parameters
    ----------
    ref_tensor : torch.Tensor
        Feature tensor receiving the additional noise channel.

    Returns
    -------
    torch.Tensor
        Feature tensor with one additional noise channel.
    """
    noise_ref_tensor = ref_tensor[:, [0], ...]

    noise = _sample(
        mu=torch.zeros_like(noise_ref_tensor), var=torch.ones_like(noise_ref_tensor)
    )[0]

    return torch.cat([ref_tensor, noise], dim=1)


def _expand_mask(x: torch.Tensor, mask: torch.Tensor):
    """
    Expand a channel-wise mask for newly added feature channels.

    New channels are assigned an all-valid mask and appended to the existing
    mask.

    Parameters
    ----------
    x : torch.Tensor
        Feature tensor defining the required number of channels.
    mask : torch.Tensor
        Existing channel-wise validity mask.

    Returns
    -------
    torch.Tensor
        Validity mask with enough all-valid channels to match ``x``.
    """
    n = x.shape[1] - mask.shape[1]

    noise_mask = torch.ones(
        mask.shape[0],
        n,
        *mask.shape[2:],
        dtype=mask.dtype,
        device=mask.device,
    )

    return torch.cat([mask, noise_mask], dim=1)
