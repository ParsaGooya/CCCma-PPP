
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

        if config.output_hidden_channels is not None:
            if config.output_hidden_channels <= 0:
                raise ValueError(
                    "output_hidden_channels must be positive."
                )

