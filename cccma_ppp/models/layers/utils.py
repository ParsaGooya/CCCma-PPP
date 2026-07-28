import torch
import torch.nn as nn
import torch.nn.functional as F

from cccma_ppp.models.layers.generic import (AlignmentMethod,
                                     PaddingMethod)


def _same_padding(kernel_size: int) -> int:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(
            "Feature-block kernel sizes must be positive odd integers."
        )
    return kernel_size // 2



def align_to_skip(
    x: torch.Tensor,
    skip: torch.Tensor,
    mode: AlignmentMethod = 'padd',
    padding_mode: PaddingMethod = "zeros"
) -> torch.Tensor:
    target_size = skip.shape[-2:]

    if x.shape[-2:] == target_size:
        return x

    if mode == "resize":
        return F.interpolate(
            x,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

    if mode == "padd":
        return padd(
            x,
            target_size,
            padding_mode
        )

    if mode == "strict":
        raise RuntimeError(
            "Decoder and skip features have incompatible spatial "
            f"shapes: {x.shape[-2:]} and {target_size}."
        )

    raise ValueError(f"Unknown alignment mode: {mode}")



def padd(
    x: torch.Tensor,
    target_size: tuple[int, int],
    padding_mode: PaddingMethod = "zeros"
) -> torch.Tensor:
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
        mode = padding_mode
    )



def _resize_mask(
    mask: torch.Tensor | None,
    size: tuple[int, int],
) -> torch.Tensor | None:
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
    """Normalize common mask layouts to NCHW and broadcast channels."""

    if mask is None:
        return None

    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)

    elif mask.ndim == 3:
        # Interpret as either [C, H, W].
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
    """Create the mask matching a channel-wise concatenated feature tensor."""

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
    Sample latent variables from Gaussian distribution.

    Parameters
    ----------
    mu : torch.Tensor
    var : torch.Tensor
    sample_size : int, optional
    std : float, optional

    Returns
    -------
    torch.Tensor
        Sampled latent variables.
    """

    out = mu + torch.sqrt(var) * _get_normal(var, std).sample((sample_size,))

    return out

def _get_normal(ref_tensor, std=1):
    """
    Create standard normal distribution.

    Parameters
    ----------
    ref_tensor : torch.Tensor
    std : float, optional

    Returns
    -------
    torch.distributions.Normal
    """

    return torch.distributions.Normal(
        torch.zeros_like(ref_tensor), torch.ones_like(ref_tensor) * std
    )


def _noise_injection(ref_tensor: torch.Tensor):
    noise_ref_tensor = ref_tensor[:,[0],...]

    noise =  _sample(
    mu = torch.zeros_like(noise_ref_tensor),
    var = torch.ones_like(noise_ref_tensor)
            )[0]

    return torch.cat([ref_tensor, noise], dim = 1)


def _expand_mask(x: torch.Tensor,
                 mask: torch.Tensor
                 ):
    
    n = x.shape[1] - mask.shape[1]

    noise_mask = torch.ones(
        mask.shape[0],
        n,
        *mask.shape[2:],
        dtype=mask.dtype,
        device=mask.device,
    )

    return torch.cat([mask, noise_mask], dim=1)