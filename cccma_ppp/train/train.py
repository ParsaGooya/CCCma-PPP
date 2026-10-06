from __future__ import annotations

import argparse

import dacite

from cccma_ppp.generic.distributed import Distributed
from cccma_ppp.generic.logger import setup_logger
from cccma_ppp.generic.monitoring import distributed_monitoring
from cccma_ppp.train.train_configs import (
    TrainConfig,
    build_trainer,
    prepare_config,
)


def get_parser() -> argparse.ArgumentParser:
    """
    Build the command-line argument parser for model training.

    Returns
    -------
    argparse.ArgumentParser
        Parser configured to accept the path to a training YAML configuration
        file.
    """
    parser = argparse.ArgumentParser(description="Train model from config file")

    parser.add_argument(
        "config",
        type=str,
        help="Path to the YAML config file.",
    )

    return parser


def main(yaml_config: str) -> None:
    """
    Run the model training workflow from a YAML configuration file.

    Initializes the distributed runtime, loads and constructs the training
    configuration, sets a rank-specific random seed, configures logging and
    runtime directories, and builds the trainer. Resource monitoring records
    trainer construction, distributed setup, and training execution. Distributed
    resources are cleaned up when execution completes or an exception is raised.

    Parameters
    ----------
    yaml_config : str
        Path to the YAML file containing the training configuration.
    """
    distributed = Distributed.get_instance()

    try:
        config_data = prepare_config(yaml_config)

        config = dacite.from_dict(
            data_class=TrainConfig,
            data=config_data,
            config=dacite.Config(strict=True),
        )

        config.set_random_seed(distributed.rank)

        logger = setup_logger(
            name="training",
            log_dir=config.log_dir,
        )

        if distributed.is_root():
            logger.info("Setting up directories ...")

        config.prepare_directory(
            distributed,
            yaml_config,
        )

        if distributed.is_root():
            logger.info("Building objects:")

        with distributed_monitoring(
            distributed,
            config.monitoring_dir,
        ) as resource_monitor:
            with resource_monitor.span("build_trainer"):
                trainer = build_trainer(
                    config,
                    distributed,
                    logger,
                )

            with resource_monitor.span("setup_distributed"):
                trainer.setup_distributed(
                    distributed=distributed,
                    logger=logger,
                    log_every_n_epochs=(config.log_every_n_epochs),
                    save_checkpoint=(config.save_checkpoint),
                )

            resource_monitor.checkpoint("trainer_ready")

            with resource_monitor.span("train"):
                trainer.train()

    finally:
        distributed.cleanup()


if __name__ == "__main__":
    arguments = get_parser().parse_args()
    main(arguments.config)
