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

from cccma_ppp.data.dataloader import TrainDataloaderConfig
from cccma_ppp.data.utils_data import WeightsConfig

from cccma_ppp.generic.distributed import Distributed
from cccma_ppp.generic.runtime import RuntimeContext

from cccma_ppp.core.selectors import ModuleSelector
from cccma_ppp.core.trainer import TrainerConfig
from cccma_ppp.core.optimization import OptimizerConfig

from cccma_ppp.preprocessing.utils_preprocessing import Oceannanremove


def set_seed(seed):
    """
    Set random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed value.

    Returns
    -------
    None
    """

    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclasses.dataclass
class TrainConfig:
    """
    Configuration for training experiments.

    Parameters
    ----------
    experiment_dir : str
        Root directory for experiment outputs.
    max_epochs : int
        Maximum number of training epochs.
    train_loader : TrainDataloaderConfig or None
        Data loader configuration.
    module : ModuleSelector or None
        Model selector.
    losspipeline : LosspipelineConfig or None
        Loss pipeline configuration.
    trainer : TrainerConfig or None
        Trainer configuration.
    optimization : OptimizerConfig, optional
        Optimization configuration.
    weights : WeightsConfig, optional
        Weight configuration for loss.
    log_every_n_epochs : int, optional
        Logging frequency.
    save_checkpoint : bool, optional
        Whether to save checkpoints.
    seed : int or None, optional
        Random seed.
    resume_dir : str or None, optional
        Directory of a previous experiment to resume from.
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
        Validate and initialize training configuration.

        Handles resume logic, validates inputs, sets defaults,
        and enforces consistency across model, data, and loss.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If required configuration fields are missing or inconsistent.
        RuntimeError
            If incompatible preprocessing or loss setup is detected.
        AssertionError
            If max_epochs is invalid.
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

        if self.max_epochs is None:
            self.max_epochs = float("inf")

        assert self.max_epochs >= 0

        self.experiment_dir = Path(self.experiment_dir)

        if self.train_loader.dataset_config.observation is None:
            if self.module.type.lower() in ["deterministic", "default"]:
                raise ValueError(
                    "with determisitic models target observation must be specified."
                )

        if self.module.type.lower() in ["cVAE"]:
            if self.trainer.beta_finder is None:
                raise ValueError(
                    "with cVAE model TrainerConfig.beta_finder must be set up."
                )
            if not self.train_loader.dataset_config.condition_type is not None:
                raise ValueError("with cVAE you must specify condition type!")

        if getattr(self.module._module_config.model_config, "GENERATOR", False):
            if "crps" not in self.losspipeline.loss_pipeline.loss_types:
                raise RuntimeError(
                    "For models with generators crps has to be in the loss function."
                )

        if self.module.type.lower() in ["deterministic", "default"]:
            if self.trainer.beta_finder is None:
                warnings.warn(
                    "TrainerConfig.beta_finder setup will be ignored with deterministic models ..."
                )

        if self.train_loader.dataset_config.observation is not None:
            pipeline = self.train_loader.dataset_config.observation.preprocessing_pipeline.pipeline
        else:
            pipeline = (
                self.train_loader.dataset_config.model.preprocessing_pipeline.pipeline
            )

        if self.module._module_config.model_config.NUM_OUTPUT_DIMS == 1:
            if not any([isinstance(step[1], Oceannanremove) for step in pipeline]):
                raise RuntimeError(
                    "for MLP models, add Oceannanremove as a preprocessing step to flatten the maps."
                )
        else:
            if any([isinstance(step[1], Oceannanremove) for step in pipeline]):
                raise RuntimeError(
                    "for non-MLP models, do add Oceannanremove as a preprocessing step because it flattens the maps."
                )

    def read_config_from_halted_experiment(
        self, resume_dir: str | Path, experiment_dir: str | Path, max_epochs: int
    ) -> "TrainConfig":
        """
        Load configuration from a previous experiment.

        Parameters
        ----------
        resume_dir : str or pathlib.Path
            Path to existing experiment.
        experiment_dir : str or pathlib.Path
            New experiment directory.
        max_epochs : int
            Updated number of epochs.

        Returns
        -------
        TrainConfig

        Raises
        ------
        ValueError
            If resume directory does not exist.
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

    def set_random_seed(self):
        """
        Apply configured random seed.

        Returns
        -------
        None
        """

        if self.seed is not None:
            set_seed(self.seed)

    @property
    def checkpoint_dir(self) -> str:
        """
        Path to checkpoint directory.

        Returns
        -------
        str
        """
        return os.path.join(self.experiment_dir, "checkpoints")

    @property
    def log_dir(self) -> str:
        """
        Path to logging directory.

        Returns
        -------
        str
        """
        return os.path.join(self.experiment_dir, "logs")

    @property
    def figures_dir(self) -> str:
        """
        Path to figures directory.

        Returns
        -------
        str
        """
        return os.path.join(self.experiment_dir, "figures")

    def _prepare_runtime_variables(self):
        """
        Populate global runtime context variables.

        Returns
        -------
        None
        """

        RuntimeContext.GLOBAL_EXP_DIR = str(self.experiment_dir)
        RuntimeContext.GLOBAL_CHECKPOINT_DIR = str(self.checkpoint_dir)
        RuntimeContext.GLOBAL_FIGURES_DIR = str(self.figures_dir)
        RuntimeContext.GLOBAL_LOG_DIR = str(self.log_dir)
        RuntimeContext.INPUT_VAR_METADATA = self.train_loader.input_var_metadata
        RuntimeContext.TARGET_VAR_METADATA = self.train_loader.target_var_metadata

    def prepare_directory(
        self,
        distributed: Distributed,
        yaml_config: str | None = None,
    ):
        """
        Prepare experiment directory structure.

        Parameters
        ----------
        distributed : Distributed
            Distributed context used for coordinating directory creation.
        yaml_config : str or None, optional
            Path to configuration file to copy into the experiment directory.

        Returns
        -------
        None

        Notes
        -----
        - Creates directories for logs, checkpoints, and figures.
        - Handles resume directory copying if needed.
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


def prepare_config(path: Path | str) -> dict:
    """
    Load configuration from YAML file.

    Parameters
    ----------
    path : pathlib.Path or str

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    return data


def build_trainer(
    config: TrainConfig,
    distributed: Distributed,
    logger: logging.Logger | None = None,
):
    """
    Construct training pipeline.

    Builds data loaders, model, loss function, optimizer,
    and trainer object.

    Parameters
    ----------
    config : TrainConfig
        Training configuration.
    distributed : Distributed
        Distributed training context.
    logger : logging.Logger or None, optional
        Logger for output.

    Returns
    -------
    Trainer
        Initialized trainer instance.
    """

    def log(msg, **kwargs):
        if distributed.is_root():
            if logger is not None:
                logger.info(msg, **kwargs)
            else:
                print(msg)

    log("creating data loaders ...")

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
