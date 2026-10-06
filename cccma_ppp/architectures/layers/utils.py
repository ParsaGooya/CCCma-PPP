import torch
import torch.nn.functional as F

from cccma_ppp.architectures.layers.generic import AlignmentMethod, PaddingMethod


def _same_padding(kernel_size: int) -> int:
    """
    Compute symmetric padding that preserves spatial dimensions.

    Parameters
    ----------
    kernel_size : int
        Size of the convolutional kernel.

    Returns
    -------
    int
        Padding size required on each side of a spatial dimension.

    Raises
    ------
    ValueError
        If ``kernel_size`` is not a positive odd integer.
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
    Align a tensor to the spatial shape of a skip connection.

    Parameters
    ----------
    x : torch.Tensor
        Tensor to spatially align.
    skip_shape : tuple[int, int]
        Target spatial shape ``(H, W)``.
    mode : AlignmentMethod
        Method used to align mismatched spatial dimensions.
    padding_mode : PaddingMethod
        Padding mode used when ``mode`` is ``"padd"``.

    Returns
    -------
    torch.Tensor
        Tensor aligned to ``skip_shape``.

    Raises
    ------
    RuntimeError
        If the spatial dimensions differ and ``mode`` is ``"strict"``.
    ValueError
        If ``mode`` is not a supported alignment method.
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

    When an odd amount of padding is required, the additional element is added
    to the right or bottom side.

    Parameters
    ----------
    x : torch.Tensor
        Tensor to pad.
    target_size : tuple[int, int]
        Target spatial shape ``(H, W)``.
    padding_mode : PaddingMethod
        Padding mode to use. ``"zeros"`` applies constant zero padding.

    Returns
    -------
    torch.Tensor
        Padded tensor with the requested spatial dimensions.
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
    Resize a validity mask to a target spatial size.

    Nearest-neighbor interpolation is used to preserve discrete mask values and
    the original data type is restored after resizing.

    Parameters
    ----------
    mask : torch.Tensor | None
        Validity mask to resize. If ``None``, no resizing is performed.
    size : tuple[int, int]
        Target spatial shape ``(H, W)``.

    Returns
    -------
    torch.Tensor | None
        Resized mask with its original data type, or ``None`` if no mask was
        provided.
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
    Adapt a validity mask to the batch and spatial dimensions of a tensor.

    Two- and three-dimensional masks are expanded to four dimensions, singleton
    batch dimensions are broadcast when necessary, and mismatched spatial
    dimensions are resized using nearest-neighbor interpolation.

    Parameters
    ----------
    mask : torch.Tensor | None
        Validity mask to adapt. If ``None``, no mask is returned.
    reference : torch.Tensor
        Reference tensor whose batch size, spatial dimensions, device, and data
        type are used for the output mask.

    Returns
    -------
    torch.Tensor | None
        Adapted mask on the same device and with the same data type as
        ``reference``, or ``None`` if no mask was provided.

    Raises
    ------
    ValueError
        If ``mask`` is not two-, three-, or four-dimensional, or if its batch
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
    Merge input and skip-connection validity masks along the channel dimension.

    Missing masks are replaced with fully valid masks for their corresponding
    channels before the masks are concatenated.

    Parameters
    ----------
    input_mask : torch.Tensor | None
        Validity mask associated with the input features.
    skip_mask : torch.Tensor | None
        Validity mask associated with the skip-connection features.
    out_channels : int
        Number of channels represented by ``input_mask`` when it is absent.
    skip_channels : int
        Number of channels represented by ``skip_mask`` when it is absent.
    spatial_size : tuple[int, int]
        Spatial shape ``(H, W)`` to which both masks are resized.
    reference : torch.Tensor
        Reference tensor providing the batch size, device, and data type for
        generated masks.

    Returns
    -------
    torch.Tensor | None
        Concatenated skip and input masks, or ``None`` if both masks are absent.
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
        Target spatial shape ``(H, W)``.
    mode : str
        Interpolation mode. Bilinear and bicubic interpolation are performed
        with ``align_corners=False``.

    Returns
    -------
    torch.Tensor
        Resized tensor, or the original tensor if it already has the requested
        spatial dimensions.
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
    Draw samples from a Gaussian distribution with element-wise parameters.

    Parameters
    ----------
    mu : torch.Tensor
        Mean of the Gaussian distribution.
    var : torch.Tensor
        Variance of the Gaussian distribution.
    sample_size : int
        Number of samples to draw.
    std : float
        Standard deviation of the standard normal noise used for sampling.

    Returns
    -------
    torch.Tensor
        Samples obtained by adding scaled Gaussian noise to ``mu``.
    """
    out = mu + torch.sqrt(var) * _get_normal(var, std).sample((sample_size,))

    return out


def _get_normal(ref_tensor, std=1):
    """
    Construct a zero-mean normal distribution matching a reference tensor.

    Parameters
    ----------
    ref_tensor : torch.Tensor
        Reference tensor defining the shape, device, and data type of the
        distribution parameters.
    std : float
        Standard deviation of the normal distribution.

    Returns
    -------
    torch.distributions.Normal
        Normal distribution with zero mean and the specified standard deviation.
    """
    return torch.distributions.Normal(
        torch.zeros_like(ref_tensor), torch.ones_like(ref_tensor) * std
    )


def _noise_injection(ref_tensor: torch.Tensor):
    """
    Append a Gaussian noise channel to a tensor.

    A spatial noise channel is sampled independently for each batch element and
    concatenated to the input along the channel dimension.

    Parameters
    ----------
    ref_tensor : torch.Tensor
        Reference tensor with shape ``(N, C, H, W)``.

    Returns
    -------
    torch.Tensor
        Tensor containing the original channels followed by one Gaussian noise
        channel.
    """
    noise_ref_tensor = ref_tensor[:, [0], ...]

    noise = _sample(
        mu=torch.zeros_like(noise_ref_tensor), var=torch.ones_like(noise_ref_tensor)
    )[0]

    return torch.cat([ref_tensor, noise], dim=1)


def _expand_mask(x: torch.Tensor, mask: torch.Tensor):
    """
    Expand a validity mask to cover additional tensor channels.

    Fully valid mask channels are appended until the mask channel count matches
    that of the reference tensor.

    Parameters
    ----------
    x : torch.Tensor
        Reference tensor whose channel count determines the expanded mask size.
    mask : torch.Tensor
        Validity mask to expand.

    Returns
    -------
    torch.Tensor
        Mask with additional fully valid channels appended along the channel
        dimension.
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
