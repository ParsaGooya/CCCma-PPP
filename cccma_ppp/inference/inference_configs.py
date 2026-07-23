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
from cccma_ppp.data_modules.dataloader import Dataloader
from cccma_ppp.train.dataloader import TrainDataloaderConfig
from cccma_ppp.train.train_configs import set_seed


@dataclasses.dataclass
class InferenceConfig:
    """
    Configuration for running inference from a trained experiment.

    Parameters
    ----------
    experiment_dir : str
        Directory containing the trained experiment configuration,
        preprocessing pipelines, and checkpoints.
    writer : WriterConfig
        Configuration used to construct the inference writer.
    inference_loader : InferenceDataloaderConfig, optional
        Configuration used to construct the inference data loader.
    save_path : str or None, optional
        Directory in which inference outputs are saved. If ``None``, outputs
        are saved in the experiment's ``inference`` directory.
    seed : int or None, optional
        Base random seed used during inference.
    checkpoint_name : str or None, optional
        Name of the checkpoint file to load. If ``None``, ``best.pt`` is used.

    """

    experiment_dir: str
    writer: WriterConfig
    inference_loader: InferenceDataloaderConfig = dataclasses.field(
        default_factory=InferenceDataloaderConfig
    )
    save_path: str | None = None
    seed: int | None = None
    checkpoint_name: str | None = None

    def __post_init__(self):
        """
        Initialize paths and load the training configuration.

        """
        self.experiment_dir = Path(self.experiment_dir)

        if self.save_path is not None:
            self.save_path = Path(self.save_path)

        self.train_config = self.load_train_config()
        self.train_loader = self.load_train_dataloader_config()

        self._resolve_inference_dataset_config()

    def _resolve_inference_dataset_config(self):
        """
        Resolve and validate the inference dataset configuration.

        Raises
        ------
        RuntimeError
            If the inference input metadata or temporal features are inconsistent
            with the training configuration.

        """

        self.inference_loader.read_configs_from_train(self.train_loader)
        self._check_inference_dataset()

    def _check_inference_dataset(self):
        """
        Validate the inference dataset against the training dataset.

        Raises
        ------
        RuntimeError
            If the inference input variables or preprocessing metadata differ from
            those used during training.
        RuntimeError
            If the inference temporal features differ from those used during
            training.

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
        Return the path to the fitted output preprocessing pipeline.

        Returns
        -------
        pathlib.Path
            Path to the observation or model preprocessing pipeline used to
            post-process inference outputs.

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
        Return the inference output directory.

        Returns
        -------
        pathlib.Path
            Explicit save path when configured, otherwise the experiment's
            ``inference`` directory.

        """
        return (
            self.save_path
            if self.save_path is not None
            else self.experiment_dir / "inference"
        )

    @property
    def log_dir(self) -> Path:
        """
        Return the experiment log directory.

        Returns
        -------
        pathlib.Path
            Path to the experiment's ``logs`` directory.

        """
        return self.experiment_dir / "logs"

    def _prepare_runtime_variables(self):
        """
        Set global runtime paths and input-variable metadata.

        """

        RuntimeContext.GLOBAL_EXP_DIR = str(self.experiment_dir)
        RuntimeContext.GLOBAL_OUTPUT_DIR = str(self.output_dir)
        RuntimeContext.GLOBAL_LOG_DIR = str(self.log_dir)
        RuntimeContext.INPUT_VAR_METADATA = self.inference_loader.input_var_metadata

    def prepare_directory(self, distributed: Distributed):
        """
        Prepare runtime variables and create the inference output directory.

        Parameters
        ----------
        distributed : Distributed
            Distributed execution context. The output directory is created by the
            root process before all processes are synchronized.

        """

        self._prepare_runtime_variables()

        if distributed.is_root():
            os.makedirs(self.output_dir, exist_ok=True)

        distributed.barrier()

    def set_random_seed(self, rank: int):
        """
        Apply the configured process-specific random seed.

        Parameters
        ----------
        rank : int
            Process rank added to the configured base seed.

        """

        if self.seed is not None:
            set_seed(self.seed + rank)

    def load_train_config(self):
        """
        Load the training configuration.

        Returns
        -------
        dict
            Training configuration loaded from the experiment's ``config.yaml``
            file.

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

    def load_module(
        self,
        inference_loader: Dataloader | None = None,
        strict: bool = False,
    ):
        """
        Load the trained module from a checkpoint.

        Parameters
        ----------
        inference_loader : Dataloader or None, optional
            Inference data loader whose input shape and additional-feature
            dimension are validated against the checkpoint.
        strict : bool, optional
            Whether checkpoint parameters must exactly match the reconstructed
            module state.

        Returns
        -------
        moduleABC
            Reconstructed module with its trained parameters loaded.

        Raises
        ------
        FileNotFoundError
            If the requested checkpoint does not exist.
        KeyError
            If the checkpoint does not contain the required shape, feature, or
            module-state entries.
        RuntimeError
            If the inference data dimensions do not match the checkpoint.
        RuntimeError
            If checkpoint parameters do not match the reconstructed module when
            ``strict`` is ``True``.

        """

        path = Path(self.experiment_dir) / "checkpoints"

        if self.checkpoint_name is not None:
            path = path / self.checkpoint_name
        else:
            path = path / "best.pt"

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)  ### ERROR

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

        if inference_loader is not None:
            if not all(
                [
                    input_shape == inference_loader.input_shape,
                    added_features_dim == inference_loader.added_features_dim,
                ]
            ):
                raise RuntimeError("Data and model IO dimensions do not match!")

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
    path : pathlib.Path or str
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.

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
    Construct a writer for distributed inference.

    Parameters
    ----------
    config : InferenceConfig
        Inference configuration.
    distributed : Distributed
        Distributed execution context.
    logger : logging.Logger or None, optional
        Logger used to report setup progress. If ``None``, messages are printed
        by the root process.

    Returns
    -------
    Writer
        Configured inference writer.

    """

    def log(msg, **kwargs):
        if distributed.is_root():
            if logger is not None:
                logger.info(msg, **kwargs)
            else:
                print(msg)

    log("creating data loader ...")

    config.inference_loader.setup_distributed(config.train_loader, distributed)

    inference_loader = config.inference_loader.build_inference_loader()

    log("Loading saved module ...")

    module = config.load_module(inference_loader)
    module = module.to(distributed.device)

    # if distributed.distributed:
    #     module = torch.nn.parallel.DistributedDataParallel(module, device_ids=[distributed.local_rank], output_device=distributed.local_rank, find_unused_parameters=False)

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
