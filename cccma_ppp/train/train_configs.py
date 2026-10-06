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

from cccma_ppp.loss.loss import LosspipelineConfig

from cccma_ppp.train.dataloader import TrainDataloaderConfig
from cccma_ppp.data_modules.weights import WeightsConfig

from cccma_ppp.generic.distributed import Distributed
from cccma_ppp.generic.runtime import RuntimeContext

from cccma_ppp.core.selectors import ModuleSelector
from cccma_ppp.core.trainer import TrainerConfig
from cccma_ppp.core.optimization import OptimizerConfig
from cccma_ppp.core.core_abc import GenerativeContext


def set_seed(seed):
    """
    Set random seeds used by NumPy and PyTorch.

    Parameters
    ----------
    seed : int
        Random seed applied to NumPy and PyTorch.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclasses.dataclass
class TrainConfig:
    """
    Configuration for a model training experiment.

    Defines the training data, model, loss pipeline, trainer, optimization,
    weighting, checkpointing, logging, reproducibility, and experiment-resumption
    settings.

    Parameters
    ----------
    experiment_dir : str
        Directory in which experiment outputs are stored.
    max_epochs : int
        Maximum number of training epochs.
    train_loader : TrainDataloaderConfig | None
        Configuration used to construct training and validation data loaders.
    module : ModuleSelector | None
        Selector defining the model module to train.
    losspipeline : LosspipelineConfig | None
        Configuration defining the reconstruction loss pipeline.
    trainer : TrainerConfig | None
        Configuration controlling the training procedure.
    optimization : OptimizerConfig | None
        Configuration defining the optimizer and optional learning-rate
        scheduler.
    weights : WeightsConfig | None
        Configuration defining spatial and variable loss weights.
    log_every_n_epochs : int
        Number of epochs between training log and plot updates.
    save_checkpoint : bool
        Whether checkpoints are saved during training.
    seed : int | None
        Base random seed used for training. If ``None``, no seed is explicitly
        configured.
    resume_dir : str | None
        Directory of a halted experiment from which training configuration and
        state are resumed.
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

        Resuming settings are resolved first, after which model, loss, data, and
        input-output compatibility are validated. The experiment directory is
        converted to a path after validation.

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
        Validate compatibility between the module, loss pipeline, and training data.

        Generator-enabled models require CRPS loss. Deterministic models require
        observation targets, while cVAE models require beta annealing. Additional
        checks enforce conditioning requirements and valid batch sizes for MLP
        architectures using batch normalization.

        Raises
        ------
        RuntimeError
            If a generator-enabled model does not include CRPS in the loss pipeline.
        ValueError
            If deterministic training has no observation target, generative training
            has neither an observation target nor conditioning method, a cVAE has no
            beta annealing configuration, or MLP batch normalization is used with a
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
            if self.train_loader.dataset_config.observation is None:
                raise ValueError(
                    "with determisitic models target observation must be specified."
                )
            if self.trainer.beta_finder is None:
                warnings.warn(
                    "TrainerConfig.beta_finder setup will be ignored with deterministic models ..."
                )
        else:
            if (
                self.train_loader.dataset_config.condition_method is None
                and self.train_loader.dataset_config.observation is None
            ):
                raise ValueError(
                    "with generative models you must specify condition method if not bias correcting to a target!"
                )

        if self.module.type.lower() in ["cvae"]:
            if self.trainer.beta_finder is None:
                raise ValueError(
                    "with cVAE model TrainerConfig.beta_finder must be set up."
                )

        if self.module.model_type.lower() in ["mlp"]:
            if self.module.model_config.batch_normalization:
                if self.train_loader.batch_size == 1:
                    raise ValueError(
                        f"With {self.module.model_type} architecture, if batch_normalization is True, "
                        "batch_size must be greater thatn one duting training."
                    )

    def _check_IO_consistency(self):
        """
        Validate model dimensionality against input and target data.

        Input and target variable metadata are compared with the numbers of dimensions
        supported by the selected model architecture.

        Raises
        ------
        RuntimeError
            If input or target dimensionality is incompatible with the selected model
            architecture or preprocessing pipeline.
        """
        input_metadata = self.train_loader.input_var_metadata
        output_metadata = self.train_loader.target_var_metadata
        _check_IO(input_metadata, self.module.NUM_INPUT_DIMS, "input")
        _check_IO(output_metadata, self.module.NUM_OUTPUT_DIMS, "output")

    def set_random_seed(self, rank: int):
        """
        Set the process-specific training random seed.

        The configured base seed is offset by the process rank so distributed
        processes use distinct deterministic seeds.

        Parameters
        ----------
        rank : int
            Process rank added to the configured base seed.
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
        Return the experiment figures directory.

        Returns
        -------
        str
            Path to the ``figures`` directory under the experiment directory.
        """
        return os.path.join(self.experiment_dir, "figures")

    def _prepare_runtime_variables(self):
        """
        Populate global runtime settings for training.

        Updates experiment, checkpoint, figure, logging, and monitoring directories
        together with input and target variable metadata.
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
        directory is copied to the new location. Otherwise, required experiment
        directories are created. The training YAML configuration is optionally copied
        into the experiment directory.

        Parameters
        ----------
        distributed : Distributed
            Distributed runtime used for root-process detection and synchronization.
        yaml_config : str | None
            Path to the training YAML configuration to copy into the experiment
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

        When ``resume_dir`` is provided, the saved training configuration is loaded
        from the halted experiment while preserving the newly requested experiment
        directory and maximum epoch count. For a new experiment, all required
        training configurations must be provided.

        Raises
        ------
        ValueError
            If a new experiment is missing its training data-loader, module, loss
            pipeline, or trainer configuration.
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
        Load training configuration from a halted experiment.

        The saved configuration is updated with the requested experiment directory
        and maximum epoch count before being reconstructed as a new training
        configuration.

        Parameters
        ----------
        resume_dir : str | Path
            Directory containing the halted experiment.
        experiment_dir : str | Path
            Experiment directory to use for resumed training.
        max_epochs : int
            Maximum number of epochs for resumed training.

        Returns
        -------
        TrainConfig
            Training configuration reconstructed from the halted experiment.

        Raises
        ------
        ValueError
            If ``resume_dir`` does not exist or is not a directory.
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
    Build all components required for model training.

    Training and validation data loaders are constructed first, followed by loss
    weights, the selected model module, reconstruction loss, optimizer, and
    trainer. Model masking requirements and generative behavior are propagated to
    the relevant components.

    Parameters
    ----------
    config : TrainConfig
        Configuration defining the complete training experiment.
    distributed : Distributed
        Distributed runtime used for data loading and model device placement.
    logger : logging.Logger | None
        Logger used for root-process setup messages. If ``None``, messages are
        printed instead.

    Returns
    -------
    Trainer
        Fully constructed trainer ready for distributed setup and training.
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
    Validate data dimensionality against model input or output requirements.

    Two-dimensional model interfaces can accept data that are already
    two-dimensional or have been flattened with ``Flattennanremove``. Models
    expecting higher dimensionality require an exact match and cannot use the
    flattening preprocessor.

    Parameters
    ----------
    metadata : dict
        Variable metadata containing neural-network dimensions and preprocessing
        information.
    model_dims : int
        Number of input or output dimensions supported by the model.
    which : str
        Interface being validated. Must be ``"input"`` or ``"output"``.

    Raises
    ------
    RuntimeError
        If data dimensionality or flattening behavior is incompatible with the
        model dimensionality.
    ValueError
        If ``which`` is neither ``"input"`` nor ``"output"``.
    """
    if which not in ["input", "output"]:
        raise ValueError("only checks IO in data vs module.")

    if model_dims == 2:
        if len(metadata.get("NN_dims")) != 2:
            if not any(
                ["flattener" in pipeline for pipeline in metadata.get("preprocessors")]
            ):
                raise RuntimeError(
                    f"The selected model supports 2D {which} but the data has {metadata.get('NN_dims')} "
                    f"{which} NN dims. add Flattennanremove as a preprocessing step to flatten the data."
                )
    else:
        if any(["flattener" in pipeline for pipeline in metadata.get("preprocessors")]):
            raise RuntimeError(
                f"For {model_dims}D {which} models, do not add Flattennanremove "
                "as a preprocessing step as it flattens the data."
            )

        if model_dims != len(metadata.get("NN_dims")):
            raise RuntimeError(
                f"The selected model supports {model_dims}D {which} but the data "
                f"has {metadata.get('NN_dims')} {which} NN dims."
            )
