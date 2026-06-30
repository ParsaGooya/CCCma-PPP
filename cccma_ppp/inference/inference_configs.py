from __future__ import annotations
import dataclasses
import os
import torch
import logging
from pathlib import Path
import yaml
import dacite


from cccma_ppp.inference.dataloader import InferenceDataloaderConfig
from cccma_ppp.generic.distributed import Distributed
from cccma_ppp.generic.runtime import RuntimeContext

from cccma_ppp.train.dataloader import TrainDataloaderConfig


@dataclasses.dataclass
class InferenceConfig:
    """
    Configuration for inference execution.

    Parameters
    ----------
    experiment_dir : str
        Path to trained experiment directory.
    inference_loader : InferenceDataloaderConfig, optional
        Inference dataloader configuration.
    output_path : str or None, optional
        Location where inference outputs are saved.
    output_ensemble_size : int, optional
        Number of output members to generate.
    output_sampler : OutputsamplerConfig or None, optional
        Sampler used to generate output ensembles.
    """

    experiment_dir: str
    inference_loader: InferenceDataloaderConfig = dataclasses.field(
        default_factory=InferenceDataloaderConfig
    )
    output_path: str = None
    output_ensemble_size: int = 1
    output_sampler: OutputsamplerConfig | None = None

    def __post_init__(self):
        """
        Initialize inference configuration.

        Returns
        -------
        None
        """
        self.experiment_dir = Path(self.experiment_dir)
        self.train_config = self.load_train_config()
        self.train_loader = self.load_train_dataloader_config()

        self._check_esnsemble_generation()
        self._resolve_inference_dataset_config()

    def _resolve_inference_dataset_config(self):
        """
        Resolve inference dataset configuration.

        Returns
        -------
        None
        """
        if (
            self.inference_loader.dataset_config is None
        ):  ### method needs to be implemented!
            self.inference_loader.dataset_config.read_from_train(
                self.train_loader.dataset_config
            )
        else:
            self._check_inference_dataset()

    def _check_inference_dataset(self):
        """
        Validate inference dataset compatibility.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If a conditioning method is required but not specified.
        RuntimeError
            If inference inputs are incompatible with the trained model.
        """

        if self.train_config.get("module").get("type").lower() in ["cvae"]:
            if self.inference_loader.dataset_config.condition_method is None:
                raise ValueError("with cVAE you must specify condition method!")

        if (
            self.inference_loader.input_var_metadata
            != self.train_loader.input_var_metadata
        ):
            raise RuntimeError(
                "Input variables or preprocessing steps are not consistent"
                f"with the trained model at : {self.experiment_dir}"
            )

    def _check_esnsemble_generation(self):
        """
        Validate ensemble generation settings.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ensemble generation settings are invalid.
        """
        if self.output_ensemble_size > 1 and self.output_sampler is None:
            if self.train_config.get("module").get("type").lower() in [
                "deterministic",
                "default",
            ]:
                raise ValueError(
                    "with determisitic models output ensemble cannot be generated "
                    "unless output_sampler configuration is provided."
                )

    def load_train_config(self):
        """
        Load training configuration from experiment directory.

        Returns
        -------
        dict
            Parsed training configuration.
        """

        return prepare_config(self.experiment_dir / "config.yaml")

    def load_train_dataloader_config(self):
        """
        Reconstruct training dataloader configuration.

        Returns
        -------
        TrainDataloaderConfig
        """
        return dacite.from_dict(
            data_class=TrainDataloaderConfig,
            data=self.train_config.get("train_loader"),
            config=dacite.Config(strict=False),
        )

    @property
    def output_preprocessor_dir(self):
        """
        Path to output preprocessing pipeline.

        Returns
        -------
        pathlib.Path
            Location of fitted preprocessing pipeline used to
            reconstruct model outputs.
        """

        if "observation" in self.train_config["train_loader"]["dataset_config"]:
            output_data = self.train_config["train_loader"]["dataset_config"][
                "observation"
            ]
        else:
            output_data = self.train_config["train_loader"]["dataset_config"]["model"]

        output_preprocessor = output_data.get("preprocessing_pipeline")

        location = self.experiment_dir / "preprocessing_pipeline"
        return location / f"{output_preprocessor.name}_preprocessing_pipeline.joblib"

    @property
    def save_dir(self) -> str:
        """
        Directory used for inference outputs.

        Returns
        -------
        str
        """
        return self.output_path or os.path.join(self.experiment_dir, "inference")

    def _prepare_runtime_variables(self):
        """
        Populate runtime context for inference.

        Returns
        -------
        None
        """

        RuntimeContext.GLOBAL_EXP_DIR = str(self.experiment_dir)
        RuntimeContext.GLOBAL_OUTPUT_DIR = str(self.output_dir)
        RuntimeContext.INPUT_VAR_METADATA = self.inference_loader.input_var_metadata
        RuntimeContext.TARGET_VAR_METADATA = self.inference_loader.target_var_metadata

    def prepare_directory(self, distributed: Distributed):
        """
        Create inference output directory structure.

        Parameters
        ----------
        distributed : Distributed
            Distributed execution context.

        Returns
        -------
        None
        """

        self._prepare_runtime_variables()

        if distributed.is_root():
            os.makedirs(self.output_dir, exist_ok=True)

        distributed.barrier()


def prepare_config(path: Path | str) -> dict:
    """
    Load configuration from YAML file.

    Parameters
    ----------
    path : pathlib.Path or str
        Configuration file path.

    Returns
    -------
    dict
        Parsed YAML configuration.
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    return data


def build_writer(
    config: InferenceConfig,
    distributed: Distributed,
    logger: logging.Logger | None = None,
):
    """
    Construct inference writer pipeline.

    Parameters
    ----------
    config : InferenceConfig
        Inference configuration.
    distributed : Distributed
        Distributed execution context.
    logger : logging.Logger or None, optional
        Logger instance.

    Returns
    -------
    object
        Initialized inference writer.

    Notes
    -----
    Builds all components required for inference, including
    dataloaders, model, optimization utilities, and execution
    pipeline.
    """

    def log(msg, **kwargs):
        if distributed.is_root():
            if logger is not None:
                logger.info(msg, **kwargs)
            else:
                print(msg)

    log("creating data loader ...")

    config.train_loader.setup_distributed(distributed)

    train_loader = config.train_loader.build_train_loader()
    validation_loader = config.train_loader.build_validation_loader()

    num_train_batches = len(train_loader)
    input_shape = train_loader.input_shape
    output_shape = train_loader.target_shape
    added_features_dim = train_loader.added_features_dim

    weights = train_loader.get_weights(config.weights)

    log(f"Creating {config.module.type} module ...")

    module = config.module.build_module(
        input_shape=input_shape,
        output_shape=output_shape,
        added_features_dim=added_features_dim,
    )

    module = module.to(distributed.device)

    if distributed.distributed:
        module = torch.nn.parallel.DistributedDataParallel(
            module,
            device_ids=[distributed.local_rank],
            output_device=distributed.local_rank,
            find_unused_parameters=False,
        )

    log("Creating loss function ...")

    reconstruction_loss = config.losspipeline.build(
        weights=weights,
        num_output_dimensions=getattr(module.model, "NUM_OUTPUT_DIMS", None)
        or len(output_shape),
    )

    module.init_loss_function(reconstruction_loss)

    log(f"Creating {config.optimization.optimizer_type} optimizer ...")

    optimizer = config.optimization.build(module, num_train_batches, config.max_epochs)

    log("Creating trainer ...")

    trainer = config.trainer.build(
        train_data_loader=train_loader,
        validation_data_loader=validation_loader,
        module=module,
        optimization=optimizer,
        max_epochs=config.max_epochs,
    )

    return trainer
