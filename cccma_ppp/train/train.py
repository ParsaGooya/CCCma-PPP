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
    Document this function.

    Returns
    -------
    argparse.ArgumentParser
        Description not yet provided.
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
    Document this function.

    Parameters
    ----------
    yaml_config : str
        Description not yet provided.
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
