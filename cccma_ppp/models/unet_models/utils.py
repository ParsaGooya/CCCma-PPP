from cccma_ppp.models.layers.conv import TensorMask

def _unet_config_checks(config):

        if len(config.channels) < 1:
            raise ValueError(
                "channels must contain at least one encoder width."
            )

        if any(
            not isinstance(channel, int) or channel <= 0
            for channel in config.channels
        ):
            raise ValueError(
                "All channel sizes must be positive integers."
            )

        if (
            config.bottleneck_dim is not None
            and config.bottleneck_dim <= 0
        ):
            raise ValueError(
                "bottleneck_dim must be positive."
            )

        if not 0 <= config.mask_fraction_threshold <= 1:
            raise ValueError(
                "mask_fraction_threshold must be between 0 and 1."
            )

        if config.output_block_hidden_channels is not None:
            if config.output_block_hidden_channels <= 0:
                raise ValueError(
                    "output_block_hidden_channels must be positive."
                )
        
        if config.GENERATOR is not None:
             if config.GENERATOR.num_training_noise_samples <= 0:
                  raise ValueError(
                       "For Generator to work a num_training_noise_samples " \
                       "must be chosen that is larger than 0."
                  )


def _repeat_tensor_mask(
    tensor_mask: TensorMask,
    repeats: int,
) -> TensorMask:
    return TensorMask(
        tensor=tensor_mask.tensor.repeat_interleave(repeats, dim=0),
        mask=(
            tensor_mask.mask.repeat_interleave(repeats, dim=0)
            if tensor_mask.mask is not None
            else None
        ),
    )