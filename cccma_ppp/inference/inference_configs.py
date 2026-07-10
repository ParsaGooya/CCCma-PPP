import dataclasses
import os
import numpy as np
import torch
import warnings
import logging
from pathlib import Path
import yaml
import dacite


from cccma_ppp.inference.dataloader import InferenceDataloaderConfig

from cccma_ppp.core.writer import WriterConfig, Writer
from cccma_ppp.generic.distributed import Distributed
from cccma_ppp.generic.runtime import RuntimeContext

from cccma_ppp.train.dataloader import TrainDataloaderConfig
from cccma_ppp.train.train_configs import set_seed


@dataclasses.dataclass
class InferenceConfig:

    experiment_dir: str 
    writer: WriterConfig
    inference_loader: InferenceDataloaderConfig = dataclasses.field(default_factory=InferenceDataloaderConfig) 
    save_path: str = None
    seed: int | None = None

    def __post_init__(self):
        self.experiment_dir = Path(self.experiment_dir)
        self.train_config = self.load_train_config()
        self.train_loader = self.load_train_dataloader_config()

        # self._resolve_esnsemble_generation()
        self._resolve_inference_dataset_config()

        self.writer

    def _resolve_inference_dataset_config(self):
        if self.inference_loader.dataset_config is None: ### method needs to be implemented!
            self.inference_loader.read_datasetConfig_from_train(self.train_loader.dataset_config)
        else:
            self._check_inference_dataset()

    def _check_inference_dataset(self):

        if (self.inference_loader.input_var_metadata !=
                        self.train_loader.input_var_metadata):
                raise RuntimeError(
                    'Input variables or preprocessing steps are not consistent'
                    f'with the trained model at : {self.experiment_dir}' 
                )

    @property
    def output_preprocessor_dir(self):

        if 'observation' in self.train_config['train_loader']['dataset_config']:
            output_data = self.train_config['train_loader']['dataset_config']['observation']
        else:
            output_data = self.train_config['train_loader']['dataset_config']['model']

        output_preprocessor =  output_data.get('preprocessing_pipeline')

        location = self.experiment_dir / "preprocessing_pipeline"
        return location / f"{output_preprocessor.name}_preprocessing_pipeline.joblib"

    @property
    def output_dir(self) -> str:
        """
        The directory where checkpoints are saved.
        """
        return self.save_path or os.path.join(self.experiment_dir, "inference")

    @property
    def log_dir(self) -> str:
        """
        Path to logging directory.

        Returns
        -------
        str
        """

        return os.path.join(self.experiment_dir, "logs")   
    
    def _prepare_runtime_variables(self):

        RuntimeContext.GLOBAL_EXP_DIR = str(self.experiment_dir)
        RuntimeContext.GLOBAL_OUTPUT_DIR = str(self.output_dir)
        RuntimeContext.GLOBAL_LOG_DIR = str(self.log_dir)
        RuntimeContext.INPUT_VAR_METADATA = self.inference_loader.input_var_metadata


    def prepare_directory(self, distributed : Distributed):

        """
        Create output (sub)directories.
        """

        self._prepare_runtime_variables()

        if distributed.is_root():

            os.makedirs(self.output_dir, exist_ok=True)

        distributed.barrier()

    def set_random_seed(self):
        """
        Apply configured random seed.

        Returns
        -------
        None
        """

        if self.seed is not None:
            set_seed(self.seed)

    def load_train_config(self):
                
        return prepare_config(self.experiment_dir / 'config.yaml')  
    
    def load_train_dataloader_config(self):
        return dacite.from_dict(
            data_class=TrainDataloaderConfig,
            data=self.train_config.get('train_loader'),
            config=dacite.Config(strict=False),
        )


def prepare_config(path: Path | str) -> dict:
    """Get config and update with possible dotlist override."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return data



def build_writer(config : InferenceConfig, distributed : Distributed, logger : logging.Logger | None = None):
    def log(msg, **kwargs):
        if distributed.is_root():
            if logger is not None:
                logger.info(msg, **kwargs)
            else:
                print(msg)


    log(f"creating data loader ...")


    config.train_loader.setup_distributed(distributed)

    train_loader = config.train_loader.build_train_loader()
    validation_loader =  config.train_loader.build_validation_loader()

    num_train_batches = len(train_loader)
    input_shape = train_loader.input_shape
    output_shape = train_loader.target_shape
    added_features_dim = train_loader.added_features_dim

    log(f"Creating {config.module.type} module ...")

    module = config.module.build_module(  input_shape = input_shape,
                                        output_shape = output_shape,
                                        added_features_dim = added_features_dim)


    module  = module.to(distributed.device)

    if distributed.distributed:
        module = torch.nn.parallel.DistributedDataParallel(module, device_ids=[distributed.local_rank], output_device=distributed.local_rank, find_unused_parameters=False)

    log(f"Creating writer ...")

    writer = config.writer.build(
            inference_data_loader=inference_data_loader,
            train_dataloader_config=train_dataloader_config,
            module=module,
            output_dir = config.output_dir
        )

    return writer


