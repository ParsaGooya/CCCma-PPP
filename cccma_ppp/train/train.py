from cccma_ppp.train.train_configs import TrainConfig, build_trainer, prepare_config
from cccma_ppp.generic.distributed import Distributed
from cccma_ppp.generic.logger import setup_logger

import argparse
import dacite


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


def main(yaml_config: str):
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
            data_class=TrainConfig, data=config_data, config=dacite.Config(strict=True)
        )
        config.set_random_seed(distributed.rank)

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

    finally:
        distributed.cleanup()
