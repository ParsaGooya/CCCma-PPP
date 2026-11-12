import dataclasses
import os
import torch
import logging
from pathlib import Path
import yaml
import dacite
import gc

from S2D4P.core.selectors import ModuleSelector
from S2D4P.inference.dataloader import InferenceDataloaderConfig
from S2D4P.preprocessing.preprocessing import PreprocessingPipeline
from S2D4P.core.writer import WriterConfig
from S2D4P.generic.distributed import Distributed
from S2D4P.generic.runtime import RuntimeContext
from S2D4P.train.dataloader import TrainDataloaderConfig
from S2D4P.train.train_configs import set_seed

from S2D4P.generic import registry_imports
@dataclasses.dataclass
class InferenceConfig:
    """
    Configuration for an inference workflow.

    Loads the training configuration associated with an experiment, reconstructs
    the training data-loader settings, and resolves compatible inference dataset
    and preprocessing settings.

    Parameters
    ----------
    experiment_dir : str
        Directory containing the trained experiment configuration, checkpoints,
        and fitted preprocessing pipelines.
    writer : WriterConfig
        Configuration controlling prediction and output writing.
    inference_loader : InferenceDataloaderConfig
        Configuration used to construct the inference data loader.
    save_path : str | None
        Directory in which inference outputs are saved. If ``None``, the
        ``inference`` directory under the experiment directory is used.
    seed : int | None
        Base random seed used during inference. If ``None``, no seed is explicitly
        configured.
    checkpoint_name : str | None
        Checkpoint filename to load. If ``None``, ``best.pt`` is used.
    """

    experiment_dir: str
    writer: WriterConfig = dataclasses.field(default_factory=WriterConfig)
    inference_loader: InferenceDataloaderConfig = dataclasses.field(
        default_factory=InferenceDataloaderConfig
    )
    save_path: str | None = None
    seed: int | None = None
    checkpoint_name: str | None = None

    def __post_init__(self):
        """
        Initialize the inference configuration.

        Converts configured directories to paths, loads the training configuration
        and data-loader configuration, and resolves inference settings from the
        training dataset.
        """
        self.experiment_dir = Path(self.experiment_dir)

        if self.save_path is not None:
            self.save_path = Path(self.save_path)

        self.train_config = self.load_train_config()
        self.train_loader = self.load_train_dataloader_config()

        self._resolve_inference_dataset_config()

    def _resolve_inference_dataset_config(self):
        """
        Resolve inference dataset settings from the training configuration.

        Training-derived temporal features and dataset settings are applied to the
        inference loader before input compatibility is validated.
        """
        self.inference_loader.read_configs_from_train(self.train_loader)
        self._check_inference_dataset()

    def _check_inference_dataset(self):
        """
        Validate inference inputs against the trained model configuration.

        Checks that inference variable metadata, preprocessing operations, and
        temporal features match those used during training.

        Raises
        ------
        RuntimeError
            If inference input metadata or temporal features differ from the training
            configuration.
        """
        if (
            self.inference_loader.input_var_metadata
            != self.train_loader.input_var_metadata
        ):
            raise RuntimeError(
                "Input variables or preprocessing steps are not consistent"
                f"with the trained model at : {self.experiment_dir}"
            )

        if not (self.inference_loader.time_features == self.train_loader.time_features):
            raise RuntimeError(
                "Input added time features are not consistent"
                f"with the trained model at : {self.experiment_dir}"
            )

    @property
    def output_preprocessor_dir(self):
        """
        Return the fitted output preprocessing pipeline path.

        Reference preprocessing is used when the training dataset contains reference
        targets. Otherwise, the forecast preprocessing pipeline is used.

        Returns
        -------
        Path
            Path to the fitted output preprocessing pipeline.
        """
        if "reference" in self.train_config["train_loader"]["dataset"]:
            preprocessor_name = "reference"
        else:
            preprocessor_name = "forecast"

        location = self.experiment_dir / "preprocessing_pipeline"
        return location / f"{preprocessor_name}_preprocessing_pipeline.joblib"

    @property
    def output_dir(self) -> Path:
        """
        Return the directory used for inference outputs.

        Returns
        -------
        Path
            Explicitly configured save path, or the ``inference`` directory under the
            experiment directory.
        """
        return (
            self.save_path
            if self.save_path is not None
            else self.experiment_dir / "inference"
        )

    @property
    def log_dir(self) -> Path:
        """
        Return the experiment logging directory.

        Returns
        -------
        Path
            Path to the ``logs`` directory under the experiment directory.
        """
        return self.experiment_dir / "logs"

    def _prepare_runtime_variables(self):
        """
        Populate global runtime settings for inference.

        Updates experiment, output, and logging directories together with input and
        target variable metadata derived from the inference and training
        configurations.
        """
        RuntimeContext.GLOBAL_EXP_DIR = str(self.experiment_dir)
        RuntimeContext.GLOBAL_OUTPUT_DIR = str(self.output_dir)
        RuntimeContext.GLOBAL_LOG_DIR = str(self.log_dir)
        RuntimeContext.INPUT_VAR_METADATA = self.inference_loader.input_var_metadata
        RuntimeContext.TARGET_VAR_METADATA = self.inference_loader.target_var_metadata

    def prepare_directory(self, distributed: Distributed):
        """
        Prepare runtime state and the inference output directory.

        The root process creates the output directory and all distributed processes
        are synchronized afterward.

        Parameters
        ----------
        distributed : Distributed
            Distributed runtime used for root-process detection and synchronization.
        """
        self._prepare_runtime_variables()

        if distributed.is_root():
            os.makedirs(self.output_dir, exist_ok=True)

        distributed.barrier()

    def set_random_seed(self, rank: int):
        """
        Set the process-specific inference random seed.

        The configured base seed is offset by process rank so distributed processes
        use distinct deterministic seeds.

        Parameters
        ----------
        rank : int
            Process rank added to the configured base seed.
        """
        if self.seed is not None:
            set_seed(self.seed + rank)

    def load_train_config(self):
        """
        Load the training configuration associated with the experiment.

        Returns
        -------
        dict
            Parsed contents of the experiment's ``config.yaml`` file.
        """
        return prepare_config(self.experiment_dir / "config.yaml")

    def load_train_dataloader_config(self):
        """
        Reconstruct the training data-loader configuration.

        Returns
        -------
        TrainDataloaderConfig
            Training data-loader configuration reconstructed from the saved experiment
            configuration.
        """
        return dacite.from_dict(
            data_class=TrainDataloaderConfig,
            data=self.train_config.get("train_loader"),
            config=dacite.Config(strict=False),
        )

    def load_module(self, strict: bool = False):
        """
        Load the trained model module from a checkpoint.

        The module architecture is reconstructed using the saved training
        configuration and checkpoint input and output dimensions before checkpoint
        parameters are restored.

        Parameters
        ----------
        strict : bool
            Whether checkpoint parameters must exactly match the reconstructed module
            state dictionary.

        Returns
        -------
        moduleABC
            Reconstructed model module with checkpoint parameters loaded.

        Raises
        ------
        FileNotFoundError
            If the requested checkpoint file does not exist.
        KeyError
            If the checkpoint does not contain the input shape, output shape,
            additional-feature dimension, or module state required for reconstruction.
        """
        path = Path(self.experiment_dir) / "checkpoints"

        if self.checkpoint_name is not None:
            path = path / self.checkpoint_name
        else:
            path = path / "best.pt"

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        required_keys = {
            "input_shape",
            "output_shape",
            "added_features_dim",
            "module",
        }

        missing = required_keys - checkpoint.keys()

        if missing:
            raise KeyError(f"Checkpoint {path} is missing keys: {sorted(missing)}")

        input_shape = checkpoint["input_shape"]
        output_shape = checkpoint["output_shape"]
        added_features_dim = checkpoint["added_features_dim"]

        selector = dacite.from_dict(
            data_class=ModuleSelector,
            data=self.train_config.get("module"),
            config=dacite.Config(strict=False),
        )

        module = selector.build_module(
            input_shape=input_shape,
            output_shape=output_shape,
            added_features_dim=added_features_dim,
        )

        module.load_state_dict(checkpoint["module"], strict=strict)

        del checkpoint
        gc.collect()

        return module


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


def build_writer(
    config: InferenceConfig,
    distributed: Distributed,
    logger: logging.Logger | None = None,
):
    """
    Build the inference writer and its required components.

    Loads the trained module, constructs the inference data loader, verifies model
    input compatibility, restores the fitted output preprocessing pipeline, and
    builds the configured writer.

    Parameters
    ----------
    config : InferenceConfig
        Configuration defining the inference workflow.
    distributed : Distributed
        Distributed runtime used for data loading, device placement, and process
        coordination.
    logger : logging.Logger | None
        Logger used for root-process setup messages. If ``None``, messages are
        printed instead.

    Returns
    -------
    Writer
        Fully configured inference writer.

    Raises
    ------
    RuntimeError
        If inference input dimensions or additional-feature dimensions are
        incompatible with the trained model.
    """

    def log(msg, **kwargs):
        """
        Log an inference setup message from the root process.

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

    log("Loading saved module ...")

    module = config.load_module()
    module = module.to(distributed.device)

    log("creating data loader ...")

    config.inference_loader.setup_distributed(config.train_loader, distributed)

    return_spatial_mask = module.architecture_config.EXPECTS_MASK
    inference_loader = config.inference_loader.build_inference_loader(
        return_spatial_mask=return_spatial_mask
    )

    log("Checking module dataloader compatability ...")

    if not all(
        [
            module.input_shape == inference_loader.input_shape,
            module.added_features_dim == inference_loader.added_features_dim,
        ]
    ):
        raise RuntimeError("Data and NN model IO dimensions do not match!")

    log("Loading postprocessor ...")
    post_processor = PreprocessingPipeline().load_from_memory(
        config.output_preprocessor_dir
    )

    log("Creating writer ...")

    writer = config.writer.build(
        inference_data_loader=inference_loader,
        train_dataloader_config=config.train_loader,
        module=module,
        post_processor=post_processor,
        output_dir=config.output_dir,
    )

    return writer
