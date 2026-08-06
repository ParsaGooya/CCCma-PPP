from cccma_ppp.models.layers.conv import TensorMask


def _unet_config_checks(config):
    """
    Document this function.

    Parameters
    ----------
    config : Any
        Description not yet provided.

    Raises
    ------
    ValueError
        Description not yet provided.
    """
    channels = config.channels
    transpose_kernel_sizes = config.transpose_kernel_sizes

    if isinstance(transpose_kernel_sizes, list):
        if len(transpose_kernel_sizes) != len(channels) - 1:
            raise ValueError(
                "transpose_kernel_sizes must contain one value per "
                f"upsampling stage. Expected {len(channels) - 1}, got "
                f"{len(transpose_kernel_sizes)}."
            )

        for kernel in config.transpose_kernel_sizes:
            if isinstance(kernel, int):
                check = kernel > 0

            else:
                check = len(kernel) == 2 and all(
                    isinstance(value, int) and value > 0 for value in kernel
                )

            if not check:
                raise ValueError(
                    "Each transpose-convolution kernel size must be either "
                    "a positive integer or a tuple of two positive integers."
                )
    else:
        if transpose_kernel_sizes <= 0:
            raise ValueError(
                "Each transpose-convolution kernel size must be either "
                "a positive integer or a tuple of two positive integers."
            )

    channel_lists = {
        "channels": channels,
        "condition_embedding_channels": getattr(
            config,
            "condition_embedding_channels",
            None,
        ),
    }

    for key, channels_list in channel_lists.items():
        if channels_list is None:
            continue

        if len(channels_list) < 1:
            raise ValueError(f"{key} must contain at least one encoder width.")

        if any(channel <= 0 for channel in channels_list):
            raise ValueError(f"All {key} sizes must be positive integers.")

    dimensions = {
        "bottleneck_dim": getattr(config, "bottleneck_dim", None),
        "latent_size": getattr(config, "latent_size", None),
        "condition_embedding_size": getattr(
            config,
            "condition_embedding_size",
            None,
        ),
    }

    for key, dimension in dimensions.items():
        if dimension is None:
            continue

        if dimension <= 0:
            raise ValueError(f"{key} must be a positive integer.")

    if getattr(config, "condition_dependant_latent", None) is not None:
        if (
            not config.condition_dependant_latent
            and config.deterministic_guess_config is None
            and not config.condemb_to_decoder
        ):
            raise ValueError(
                "condition embedding has to be passed to decoder for cVAE when latent is not condition dependant "
                "and a deterministic guess does not exist."
            )

    if not 0 <= config.mask_fraction_threshold <= 1:
        raise ValueError("mask_fraction_threshold must be between 0 and 1.")

    if config.output_block_hidden_channels is not None:
        if config.output_block_hidden_channels <= 0:
            raise ValueError("output_block_hidden_channels must be positive.")

    if config.GENERATOR is not None:
        if config.GENERATOR.num_training_noise_samples <= 0:
            raise ValueError(
                "For Generator to work a num_training_noise_samples "
                "must be chosen that is larger than 0."
            )


def _repeat_tensor_mask(
    tensor_mask: TensorMask,
    repeats: int,
) -> TensorMask:
    """
    Document this function.

    Parameters
    ----------
    tensor_mask : TensorMask
        Description not yet provided.
    repeats : int
        Description not yet provided.

    Returns
    -------
    TensorMask
        Description not yet provided.
    """
    return TensorMask(
        tensor=tensor_mask.tensor.repeat_interleave(repeats, dim=0),
        mask=(
            tensor_mask.mask.repeat_interleave(repeats, dim=0)
            if tensor_mask.mask is not None
            else None
        ),
    )
