from cccma_ppp.inference.inference_configs import (
    InferenceConfig,
    build_writer,
    prepare_config,
)
from cccma_ppp.generic.distributed import Distributed
from cccma_ppp.generic.logger import setup_logger
import argparse
import dacite


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run inference from a configuration file"
    )

    parser.add_argument(
        "config",
        type=str,
        help="Path to the YAML config file.",
    )

    return parser


def main(yaml_config: str):

    distributed = Distributed.get_instance()

    try:
        config_data = prepare_config(yaml_config)

        # to-do
        # config.apply_overrides(args.override)

        config = dacite.from_dict(
            data_class=InferenceConfig,
            data=config_data,
            config=dacite.Config(strict=True),
        )
        config.set_random_seed(distributed.rank)

        logger = setup_logger(name="inference", log_dir=config.log_dir)

        if distributed.is_root():
            logger.info("Setting up directories ...")

        config.prepare_directory(distributed)

        if distributed.is_root():
            logger.info("Building objects:")

        writer = build_writer(config, distributed, logger)

        writer.setup_distributed(distributed=distributed, logger=logger)

        writer.predict()

    finally:
        distributed.cleanup()


# if __name__ == "__main__":
#     parser = get_parser()
#     args = parser.parse_args()
#     main(args.config)
