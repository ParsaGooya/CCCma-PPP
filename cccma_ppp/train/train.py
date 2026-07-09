from cccma_ppp.train.train_configs import TrainConfig, build_trainer, prepare_config
from cccma_ppp.generic.distributed import Distributed
from cccma_ppp.generic.logger import setup_logger

import argparse
import dacite


def get_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for training script.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser expecting a YAML configuration path.
    """

    parser = argparse.ArgumentParser(description="Train model from config file")

    parser.add_argument(
        "config",
        type=str,
        help="Path to the YAML config file.",
    )

    return parser


def main(yaml_config: str):
    """
    Run training from configuration file.

    Parameters
    ----------
    yaml_config : str
        Path to YAML configuration file.

    Returns
    -------
    None
    """

    distributed = Distributed.get_instance()

    config_data = prepare_config(yaml_config)

    config = dacite.from_dict(
        data_class=TrainConfig, data=config_data, config=dacite.Config(strict=True)
    )
    config.set_random_seed()

    logger = setup_logger(name="training", log_dir=config.log_dir)

    if distributed.is_root():
        logger.info("Setting up directories ...")

    config.prepare_directory(distributed, yaml_config)

    if distributed.is_root():
        logger.info("Building objects:")

    trainer = build_trainer(config, distributed, logger)

    trainer.setup_distributed(
        distributed=distributed,
        logger=logger,
        log_every_n_epochs=config.log_every_n_epochs,
        save_checkpoint=config.save_checkpoint,
    )

    trainer.train()

    distributed.cleanup()
