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
from cccma_ppp.core.writer import WriterConfig
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

    # def _resolve_esnsemble_generation(self):
    #     if self.writer.output_ensemble_size > 1:
            
    #         if self._input_ensemble_exist:
    #             warnings.warn(
    #                 "========================================================================\n"
    #                 "Number of output ensemble will be overwritten by the input ensemble size\n"
    #                 "========================================================================"
    #             )
    #             self.writer.output_ensemble_size = 1
            
    #         if self.train_config.get("module").get("type").lower() in ["deterministic", "default"]:
    #             if (self.writer.output_sampler is None and 
    #                  not self._input_ensemble_exist):
    #                 raise RuntimeError(
    #                     "with determisitic models output ensemble cannot be generated "
    #                     "unless input has multiple ensemble members or"
    #                     "output_sampler configuration is provided."
    #                 )
    #             elif self.writer.output_sampler is not None:
    #                 self.writer.output_sampler.num_samples_per_output = self.writer.output_ensemble_size
                
    @property
    def _input_ensemble_exist(self):
        if (self.inference_loader.dataset_config.effective_input.info.coords.get('ensembles') is not None and
            self.inference_loader.dataset_config.condition_method not in ['ensemble_mean']):
            return len(self.inference_loader.dataset_config.effective_input.info.coords['ensembles']) > 1
        return False

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

    weights = train_loader.get_weights(config.weights)

    log(f"Creating {config.module.type} module ...")

    module = config.module.build_module(  input_shape = input_shape,
                                        output_shape = output_shape,
                                        added_features_dim = added_features_dim)


    module  = module.to(distributed.device)

    if distributed.distributed:
        module = torch.nn.parallel.DistributedDataParallel(module, device_ids=[distributed.local_rank], output_device=distributed.local_rank, find_unused_parameters=False)


    log(f"Creating loss function ...")

    reconstruction_loss = config.losspipeline.build(weights=weights, num_output_dimensions= getattr(module.model, 'NUM_OUTPUT_DIMS', None ) or len(output_shape))

    module.init_loss_function(reconstruction_loss)


    log(f"Creating {config.optimization.optimizer_type} optimizer ...")

    optimizer = config.optimization.build(module, num_train_batches, config.max_epochs)


    log(f"Creating trainer ...")

    trainer = config.trainer.build(
              train_data_loader  = train_loader ,
              validation_data_loader = validation_loader,
              module = module,
              optimization = optimizer,
              max_epochs = config.max_epochs)

    return trainer


