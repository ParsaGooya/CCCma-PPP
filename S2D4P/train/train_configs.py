import dataclasses
import os
import numpy as np
import torch
import warnings
import logging
from pathlib import Path
import shutil
import yaml
import dacite

from S2D4P.loss.loss import LosspipelineConfig

from S2D4P.train.dataloader import TrainDataloaderConfig
from S2D4P.data.weights import WeightsConfig

from S2D4P.generic.distributed import Distributed
from S2D4P.generic.runtime import RuntimeContext

from S2D4P.core.selectors import ModuleSelector
from S2D4P.core.trainer import TrainerConfig
from S2D4P.core.optimization import OptimizerConfig
from S2D4P.core.core_abc import GenerativeContext


def set_seed(seed):
    """
    Set NumPy and PyTorch random seeds.

    Parameters
    ----------
    seed : int
        Seed used to initialize NumPy and PyTorch random-number generators.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclasses.dataclass
class TrainConfig:
    """
    Configuration for a model training experiment.

    Coordinates data loading, model construction, loss configuration,
    optimization, trainer behavior, experiment directories, checkpoint
    resumption, and reproducibility settings.

    Parameters
    ----------
    experiment_dir : str
        Directory in which experiment outputs, checkpoints, logs, figures, and
        configuration files are stored.
    max_epochs : int
        Maximum number of training epochs.
    train_loader : TrainDataloaderConfig | None
        Configuration used to construct training and validation data loaders.
    module : ModuleSelector | None
        Model module and architecture configuration.
    losspipeline : LosspipelineConfig | None
        Configuration defining the reconstruction loss pipeline.
    trainer : TrainerConfig | None
        Configuration controlling the training loop.
    optimization : OptimizerConfig | None
        Optimizer and learning-rate scheduler configuration.
    weights : WeightsConfig | None
        Configuration used to construct spatial and variable loss weights.
    log_every_n_epochs : int
        Number of epochs between metric logging and plot updates.
    save_checkpoint : bool
        Whether training checkpoints are saved.
    seed : int | None
        Base random seed used for reproducible training.
    resume_dir : str | None
        Directory of a halted experiment whose configuration and checkpoint state
        should be resumed.
    """

    experiment_dir: str
    max_epochs: int
    train_loader: TrainDataloaderConfig | None
    module: ModuleSelector | None
    losspipeline: LosspipelineConfig | None
    trainer: TrainerConfig | None
    optimization: OptimizerConfig | None = dataclasses.field(
        default_factory=OptimizerConfig
    )
    weights: WeightsConfig | None = dataclasses.field(default_factory=WeightsConfig)
    log_every_n_epochs: int = 5
    save_checkpoint: bool = True
    seed: int | None = None
    resume_dir: str | None = None

    def __post_init__(self):
        """
        Resolve and validate the training configuration.

        Resumption settings are resolved first, followed by compatibility checks
        between the data pipeline, model architecture, loss configuration, and model
        input and output dimensionality.

        Raises
        ------
        AssertionError
            If ``max_epochs`` is negative.
        """
        self._resolve_resuming()
        self._check_module_pipeline_compatability()
        self._check_IO_consistency()

        if self.max_epochs is None:
            self.max_epochs = float("inf")

        assert self.max_epochs >= 0

        self.experiment_dir = Path(self.experiment_dir)

    def _check_module_pipeline_compatability(self):
        """
        Validate compatibility between the module, data, loss, and trainer settings.

        Generator-enabled architectures require CRPS in the reconstruction loss.
        Deterministic models require reference targets, while cVAE models require
        beta annealing. MLP architectures using batch normalization require a batch
        size greater than one.

        Raises
        ------
        RuntimeError
            If a generator-enabled architecture does not include CRPS in the loss
            pipeline.
        ValueError
            If required reference or conditioning data are missing, cVAE beta
            annealing is not configured, or an MLP uses batch normalization with a
            batch size of one.

        Warns
        -----
        UserWarning
            If beta annealing is configured for a deterministic model, where it is
            ignored.
        """

        if self.module.GENERATOR is not None:
            if "crps" not in self.losspipeline.loss_types:
                raise RuntimeError(
                    "For models with generators crps has to be in the loss function."
                )

        if self.module.type.lower() in ["deterministic", "default"]:
            if self.train_loader.dataset_config.reference is None:
                raise ValueError(
                    "with determisitic models reference data must be specified."
                )
            if self.trainer.beta_finder is None:
                warnings.warn(
                    "TrainerConfig.beta_finder setup will be ignored with deterministic models ..."
                )
        else:
            if (
                self.train_loader.dataset_config.condition_method is None
                and self.train_loader.dataset_config.reference is None
            ):
                raise ValueError(
                    "with generative models you must specify condition method if not bias correcting to a target!"
                )

        if self.module.type.lower() in ["cvae"]:
            if self.trainer.beta_finder is None:
                raise ValueError(
                    "with cVAE model TrainerConfig.beta_finder must be set up."
                )

        if self.module.arch_type.lower() in ["mlp"]:
            if self.module.architecture_config.batch_normalization:
                if self.train_loader.batch_size == 1:
                    raise ValueError(
                        f"With {self.module.arch_type} architecture, if batch_normalization is True, "
                        "batch_size must be greater thatn one duting training."
                    )

    def _check_IO_consistency(self):
        """
        Validate data dimensionality against model input and output requirements.

        Input and target metadata are checked against the dimensionality expected by
        the selected architecture, including compatibility with flattening
        preprocessing.
        """
        input_metadata = self.train_loader.input_var_metadata
        output_metadata = self.train_loader.target_var_metadata
        _check_IO(input_metadata, self.module.NUM_INPUT_DIMS, "input")
        _check_IO(output_metadata, self.module.NUM_OUTPUT_DIMS, "output")

    def set_random_seed(self, rank: int):
        """
        Set the process-specific training random seed.

        The configured base seed is offset by process rank so distributed processes
        use distinct deterministic seeds.

        Parameters
        ----------
        rank : int
            Distributed process rank added to the configured base seed.
        """
        if self.seed is not None:
            set_seed(self.seed + rank)

    @property
    def checkpoint_dir(self) -> str:
        """
        Return the experiment checkpoint directory.

        Returns
        -------
        str
            Path to the ``checkpoints`` directory under the experiment directory.
        """
        return os.path.join(self.experiment_dir, "checkpoints")

    @property
    def log_dir(self) -> str:
        """
        Return the experiment logging directory.

        Returns
        -------
        str
            Path to the ``logs`` directory under the experiment directory.
        """
        return os.path.join(self.experiment_dir, "logs")

    @property
    def monitoring_dir(self) -> str:
        """
        Return the experiment resource-monitoring directory.

        Returns
        -------
        str
            Path to the ``resource_monitoring`` directory under the experiment
            directory.
        """
        return os.path.join(self.experiment_dir, "resource_monitoring")

    @property
    def figures_dir(self) -> str:
        """
        Return the experiment resource-monitoring directory.

        Returns
        -------
        str
            Path to the ``resource_monitoring`` directory under the experiment
            directory.
        """
        return os.path.join(self.experiment_dir, "figures")

    def _prepare_runtime_variables(self):
        """
        Populate global runtime settings for training.

        Updates experiment, checkpoint, figure, logging, and monitoring directories
        together with input and target variable metadata derived from the training
        data configuration.
        """
        RuntimeContext.GLOBAL_EXP_DIR = str(self.experiment_dir)
        RuntimeContext.GLOBAL_CHECKPOINT_DIR = str(self.checkpoint_dir)
        RuntimeContext.GLOBAL_FIGURES_DIR = str(self.figures_dir)
        RuntimeContext.GLOBAL_LOG_DIR = str(self.log_dir)
        RuntimeContext.GLOBAL_MONITORING_DIR = str(self.monitoring_dir)
        RuntimeContext.INPUT_VAR_METADATA = self.train_loader.input_var_metadata
        RuntimeContext.TARGET_VAR_METADATA = self.train_loader.target_var_metadata

    def prepare_directory(self, distributed: Distributed, yaml_config: str = None):
        """
        Prepare experiment directories for training.

        For a resumed experiment targeting a new location, the previous experiment
        directory is copied to the new path. Otherwise, required experiment
        directories are created. The supplied YAML configuration is copied into the
        experiment directory by the root process.

        Parameters
        ----------
        distributed : Distributed
            Distributed runtime used for root-process detection and synchronization.
        yaml_config : str | None
            Path to the source training configuration file to copy into the experiment
            directory.
        """
        self._prepare_runtime_variables()

        if distributed.is_root():
            if getattr(self, "copy_resume_dir_to_new_path", False):
                shutil.copytree(self.resume_dir, self.experiment_dir)

            else:
                for path in (
                    self.experiment_dir,
                    self.checkpoint_dir,
                    self.figures_dir,
                    self.log_dir,
                ):
                    os.makedirs(path, exist_ok=True)

        distributed.barrier()

        if yaml_config is not None and distributed.is_root():
            shutil.copyfile(
                yaml_config,
                Path(self.experiment_dir) / "config.yaml",
            )

        distributed.barrier()

    def _resolve_resuming(self):
        """
        Resolve configuration for new or resumed training.

        When ``resume_dir`` is configured, the saved training configuration is loaded
        and updated with the requested experiment directory and maximum epoch count.
        For a new experiment, all required training configurations must be provided.

        Raises
        ------
        ValueError
            If a required configuration is missing for a new experiment or the
            requested resume configuration cannot be loaded.
        """
        if self.resume_dir is not None:
            requested_experiment_dir = self.experiment_dir
            requested_max_epochs = self.max_epochs
            requested_resume_dir = self.resume_dir

            resumed = self.read_config_from_halted_experiment(
                resume_dir=requested_resume_dir,
                experiment_dir=requested_experiment_dir,
                max_epochs=requested_max_epochs,
            )

            self.__dict__.update(resumed.__dict__)

            self.resume_dir = requested_resume_dir
            self.experiment_dir = Path(requested_experiment_dir)

            if Path(requested_experiment_dir) != Path(requested_resume_dir):
                self.copy_resume_dir_to_new_path = True
            else:
                self.copy_resume_dir_to_new_path = False

        else:
            required_inputs = [
                self.train_loader,
                self.module,
                self.losspipeline,
                self.trainer,
            ]
            for name, config in zip(
                ["train_loader", "module", "losspipeline", "trainer"], required_inputs
            ):
                if config is None:
                    raise ValueError(f"{name} must be specified for a new experiment.")

    def read_config_from_halted_experiment(
        self, resume_dir: str | Path, experiment_dir: str | Path, max_epochs: int
    ) -> "TrainConfig":
        """
        Load configuration from a halted training experiment.

        The saved configuration is updated with the newly requested experiment
        directory and maximum epoch count before being reconstructed as a training
        configuration.

        Parameters
        ----------
        resume_dir : str | Path
            Directory containing the halted experiment and its saved configuration.
        experiment_dir : str | Path
            Experiment directory to use for resumed training.
        max_epochs : int
            Maximum number of epochs for resumed training.

        Returns
        -------
        TrainConfig
            Reconstructed configuration for the resumed experiment.

        Raises
        ------
        ValueError
            If ``resume_dir`` does not exist as a directory.
        """
        resume_dir = Path(resume_dir)

        if not resume_dir.is_dir():
            raise ValueError(f"The directory {resume_dir} does not exist.")

        config_data = prepare_config(resume_dir / "config.yaml")
        config_data["experiment_dir"] = str(experiment_dir)
        config_data["max_epochs"] = max_epochs
        config_data["resume_dir"] = None

        return dacite.from_dict(
            data_class=TrainConfig, data=config_data, config=dacite.Config(strict=True)
        )


def prepare_config(path: Path | str) -> dict:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    path : Path | str
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration values.
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    return data


def build_trainer(
    config: TrainConfig, distributed: Distributed, logger: logging.Logger | None = None
):
    """
    Build the complete training workflow.

    Constructs training and validation data loaders, loss weights, the configured
    model module, reconstruction loss pipeline, optimizer, and trainer. Spatial
    masks are requested automatically when required by the selected architecture.

    Parameters
    ----------
    config : TrainConfig
        Configuration defining the training experiment.
    distributed : Distributed
        Distributed runtime used for data loading and model device placement.
    logger : logging.Logger | None
        Logger used for root-process setup messages. If ``None``, messages are
        printed instead.

    Returns
    -------
    Trainer
        Fully configured trainer ready for distributed setup and training.
    """

    def log(msg, **kwargs):
        """
        Log a training setup message from the root process.

        Parameters
        ----------
        msg : Any
            Message to log or print.
        kwargs : Any
            Additional keyword arguments passed to the configured logger.
        """
        if distributed.is_root():
            if logger is not None:
                logger.info(msg, **kwargs)
            else:
                print(msg)

    log("creating data loaders (this might take a few minutes)...")

    config.train_loader.setup_distributed(distributed)
    return_spatial_mask = config.module.EXPECTS_MASK

    train_loader = config.train_loader.build_train_loader(
        return_spatial_mask=return_spatial_mask
    )
    validation_loader = config.train_loader.build_validation_loader(
        return_spatial_mask=return_spatial_mask
    )
    weights = config.train_loader.get_weights(config.weights)

    num_train_batches = len(train_loader)
    input_shape = train_loader.input_shape
    output_shape = train_loader.target_shape
    added_features_dim = train_loader.added_features_dim

    log(f"Creating {config.module.type} module ...")

    module = config.module.build_module(
        input_shape=input_shape,
        output_shape=output_shape,
        added_features_dim=added_features_dim,
    )

    log("Creating loss function ...")

    generative_context = GenerativeContext(module)

    reconstruction_loss = config.losspipeline.build(
        weights=weights,
        num_output_dimensions=config.module.NUM_OUTPUT_DIMS or len(output_shape),
        generative_context=generative_context,
        output_shape=output_shape,
    )

    module.init_loss_function(reconstruction_loss)
    module = module.to(distributed.device)

    log(f"Creating {config.optimization.optimizer_type} optimizer ...")

    optimizer = config.optimization.build(
        module,
        num_train_batches,
        config.max_epochs,
        config.trainer.gradient_accumulation_steps,
    )

    log("Creating trainer ...")

    trainer = config.trainer.build(
        train_data_loader=train_loader,
        validation_data_loader=validation_loader,
        module=module,
        optimization=optimizer,
        max_epochs=config.max_epochs,
    )

    return trainer


def _check_IO(metadata: dict, model_dims: int, which: str = "input"):
    """
    Validate data preprocessing against model dimensionality.

    Two-dimensional model interfaces accept either two neural-network dimensions
    or flattened data. Higher-dimensional interfaces must not use flattening and
    must expose exactly the number of neural-network dimensions expected by the
    model.

    Parameters
    ----------
    metadata : dict
        Variable metadata containing ``NN_dims`` and preprocessing steps.
    model_dims : int
        Number of input or output dimensions expected by the model architecture.
    which : str
        Interface being validated. Must be ``"input"`` or ``"output"``.

    Raises
    ------
    RuntimeError
        If data dimensionality or flattening preprocessing is incompatible with
        the model interface.
    ValueError
        If ``which`` is neither ``"input"`` nor ``"output"``.
    """
    if which not in ["input", "output"]:
        raise ValueError("only checks IO in data vs module.")

    if model_dims == 2:
        if len(metadata.get("NN_dims")) != 2:
            if not all(
                ["flattener" in pipeline for pipeline in metadata.get("preprocessors")]
            ):
                raise RuntimeError(
                    f"The selected model supports 2D {which} but the input or target data has {metadata.get('NN_dims')} "
                    f"{which} NN dims. add Flattennanremove as a preprocessing step to flatten all data sources. "
                    "This includes all model, condition and reference data that are not flat."
                )
    else:
        if any(["flattener" in pipeline for pipeline in metadata.get("preprocessors")]):
            raise RuntimeError(
                f"For {model_dims}D {which} models, do not add Flattennanremove "
                "as a preprocessing step to any of the data sources as it flattens the data."
            )

        if model_dims != len(metadata.get("NN_dims")):
            raise RuntimeError(
                f"The selected model supports {model_dims}D {which} but the data "
                f"has {metadata.get('NN_dims')} {which} NN dims."
            )
