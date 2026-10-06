import dataclasses
import os
import torch
import logging
from pathlib import Path
import yaml
import dacite
import gc

from cccma_ppp.core.selectors import ModuleSelector
from cccma_ppp.inference.dataloader import InferenceDataloaderConfig
from cccma_ppp.preprocessing.preprocessing import PreprocessingPipeline
from cccma_ppp.core.writer import WriterConfig
from cccma_ppp.generic.distributed import Distributed
from cccma_ppp.generic.runtime import RuntimeContext
from cccma_ppp.train.dataloader import TrainDataloaderConfig
from cccma_ppp.train.train_configs import set_seed


@dataclasses.dataclass
class InferenceConfig:
    """
    Configuration for running inference from a trained experiment.

    Parameters
    ----------
    experiment_dir : str
        Directory containing the trained experiment configuration, checkpoints,
        and preprocessing pipelines.
    writer : WriterConfig
        Configuration controlling prediction generation and output writing.
    inference_loader : InferenceDataloaderConfig
        Configuration controlling inference data loading and sample selection.
    save_path : str | None
        Directory in which inference outputs are saved. If ``None``, the
        ``inference`` directory under ``experiment_dir`` is used.
    seed : int | None
        Base random seed used for inference. If ``None``, no seed is explicitly
        configured.
    checkpoint_name : str | None
        Name of the checkpoint to load. If ``None``, ``best.pt`` is used.
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

        Resolves experiment and output paths, loads the saved training configuration
        and training data-loader configuration, and derives and validates the
        inference dataset configuration.
        """
        self.experiment_dir = Path(self.experiment_dir)

        if self.save_path is not None:
            self.save_path = Path(self.save_path)

        self.train_config = self.load_train_config()
        self.train_loader = self.load_train_dataloader_config()

        self._resolve_inference_dataset_config()

    def _resolve_inference_dataset_config(self):
        """
        Resolve and validate inference dataset settings from the training configuration.

        Missing inference dataset and temporal feature settings are inherited from
        the training data-loader configuration before compatibility checks are
        performed.
        """
        self.inference_loader.read_configs_from_train(self.train_loader)
        self._check_inference_dataset()

    def _check_inference_dataset(self):
        """
        Validate inference data against the trained model configuration.

        Raises
        ------
        RuntimeError
            If inference input-variable metadata or temporal features differ from
            those used to train the model.
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
        Return the path to the preprocessing pipeline used for model outputs.

        Observation preprocessing is selected when the training dataset contains
        observation data; otherwise, the model preprocessing pipeline is used.

        Returns
        -------
        Path
            Path to the saved output preprocessing pipeline.
        """
        if "observation" in self.train_config["train_loader"]["dataset_config"]:
            preprocessor_name = "observation"
        else:
            preprocessor_name = "model"

        location = self.experiment_dir / "preprocessing_pipeline"
        return location / f"{preprocessor_name}_preprocessing_pipeline.joblib"

    @property
    def output_dir(self) -> Path:
        """
        Return the directory used for inference outputs.

        Returns
        -------
        Path
            Explicitly configured output directory, or the ``inference`` directory
            under the experiment directory when no custom path is provided.
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
            ``logs`` directory under the experiment directory.
        """
        return self.experiment_dir / "logs"

    def _prepare_runtime_variables(self):
        """
        Populate global runtime settings for inference.

        Updates experiment, output, and logging directories together with the input
        and target variable metadata associated with the inference configuration.
        """
        RuntimeContext.GLOBAL_EXP_DIR = str(self.experiment_dir)
        RuntimeContext.GLOBAL_OUTPUT_DIR = str(self.output_dir)
        RuntimeContext.GLOBAL_LOG_DIR = str(self.log_dir)
        RuntimeContext.INPUT_VAR_METADATA = self.inference_loader.input_var_metadata
        RuntimeContext.TARGET_VAR_METADATA = self.inference_loader.target_var_metadata

    def prepare_directory(self, distributed: Distributed):
        """
        Prepare runtime state and the inference output directory.

        Runtime variables are configured on every process. The root process creates
        the output directory before all distributed processes are synchronized.

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

        The global seed is offset by the process rank so distributed processes use
        distinct deterministic seeds.

        Parameters
        ----------
        rank : int
            Process rank added to the configured base seed.
        """
        if self.seed is not None:
            set_seed(self.seed + rank)

    def load_train_config(self):
        """
        Load the training configuration for the experiment.

        Returns
        -------
        dict
            Configuration loaded from ``config.yaml`` in the experiment directory.
        """
        return prepare_config(self.experiment_dir / "config.yaml")

    def load_train_dataloader_config(self):
        """
        Reconstruct the training data-loader configuration.

        Returns
        -------
        TrainDataloaderConfig
            Training data-loader configuration reconstructed from the saved
            experiment configuration.
        """
        return dacite.from_dict(
            data_class=TrainDataloaderConfig,
            data=self.train_config.get("train_loader"),
            config=dacite.Config(strict=False),
        )

    def load_module(self, strict: bool = False):
        """
        Load the trained module from a checkpoint.

        The module architecture is reconstructed from the saved training
        configuration using the input, output, and additional-feature dimensions
        stored in the checkpoint. The saved model state is then restored.

        Parameters
        ----------
        strict : bool
            Whether checkpoint parameters must exactly match the reconstructed module
            state dictionary.

        Returns
        -------
        moduleABC
            Reconstructed module with checkpoint parameters loaded.

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
    Build an inference writer for a trained experiment.

    The saved module is reconstructed and moved to the distributed device, the
    inference data loader is prepared using the training configuration, model and
    data dimensions are checked for compatibility, and the saved output
    preprocessing pipeline is loaded before constructing the writer.

    Parameters
    ----------
    config : InferenceConfig
        Configuration defining the trained experiment and inference workflow.
    distributed : Distributed
        Distributed runtime used for device placement, data loading, and process
        coordination.
    logger : logging.Logger | None
        Logger used for root-process status messages. If ``None``, messages are
        printed instead.

    Returns
    -------
    Writer
        Configured inference writer.

    Raises
    ------
    RuntimeError
        If the inference data input shape or additional-feature dimension does
        not match the loaded module.
    """

    def log(msg, **kwargs):
        """
        Log an inference setup message from the root process.

        Parameters
        ----------
        msg : Any
            Message to log or print.
        **kwargs : Any
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

    return_spatial_mask = module.model_config.EXPECTS_MASK
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
        raise RuntimeError("Data and model IO dimensions do not match!")

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
