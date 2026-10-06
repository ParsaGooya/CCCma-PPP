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
    """
    Build the command-line argument parser for inference.

    Returns
    -------
    argparse.ArgumentParser
        Parser configured to accept the path to an inference YAML configuration
        file.
    """

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
    """
    Run inference from a YAML configuration file.

    Initializes the distributed runtime, loads and constructs the inference
    configuration, sets a rank-specific random seed, configures logging and
    runtime directories, builds and initializes the inference writer, and runs
    prediction. Distributed resources are cleaned up when execution completes or
    an exception is raised.

    Parameters
    ----------
    yaml_config : str
        Path to the YAML file containing the inference configuration.
    """
    distributed = Distributed.get_instance()

    try:
        config_data = prepare_config(yaml_config)

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
