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
import cccma_ppp.generic.registry_imports


@dataclasses.dataclass
class InferenceConfig:
    experiment_dir: str
    writer: WriterConfig = dataclasses.field(
        default_factory=WriterConfig
    )
    inference_loader: InferenceDataloaderConfig = dataclasses.field(
        default_factory=InferenceDataloaderConfig
    )
    save_path: str | None = None
    seed: int | None = None
    checkpoint_name: str | None = None

    def __post_init__(self):
        self.experiment_dir = Path(self.experiment_dir)

        if self.save_path is not None:
            self.save_path = Path(self.save_path)

        self.train_config = self.load_train_config()
        self.train_loader = self.load_train_dataloader_config()

        self._resolve_inference_dataset_config()

    def _resolve_inference_dataset_config(self):

        self.inference_loader.read_configs_from_train(self.train_loader)
        self._check_inference_dataset()

    def _check_inference_dataset(self):

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

        if "observation" in self.train_config["train_loader"]["dataset_config"]:
            preprocessor_name = "observation"
        else:
            preprocessor_name = "model"

        location = self.experiment_dir / "preprocessing_pipeline"
        return location / f"{preprocessor_name}_preprocessing_pipeline.joblib"

    @property
    def output_dir(self) -> Path:
        return (
            self.save_path
            if self.save_path is not None
            else self.experiment_dir / "inference"
        )

    @property
    def log_dir(self) -> Path:
        return self.experiment_dir / "logs"

    def _prepare_runtime_variables(self):

        RuntimeContext.GLOBAL_EXP_DIR = str(self.experiment_dir)
        RuntimeContext.GLOBAL_OUTPUT_DIR = str(self.output_dir)
        RuntimeContext.GLOBAL_LOG_DIR = str(self.log_dir)
        RuntimeContext.INPUT_VAR_METADATA = self.inference_loader.input_var_metadata
        RuntimeContext.TARGET_VAR_METADATA = self.inference_loader.target_var_metadata

    def prepare_directory(self, distributed: Distributed):
        """
        Create output (sub)directories.
        """

        self._prepare_runtime_variables()

        if distributed.is_root():
            os.makedirs(self.output_dir, exist_ok=True)

        distributed.barrier()

    def set_random_seed(self, rank: int):
        """
        Apply configured random seed.

        Returns
        -------
        None
        """

        if self.seed is not None:
            set_seed(self.seed + rank)

    def load_train_config(self):

        return prepare_config(self.experiment_dir / "config.yaml")

    def load_train_dataloader_config(self):
        return dacite.from_dict(
            data_class=TrainDataloaderConfig,
            data=self.train_config.get("train_loader"),
            config=dacite.Config(strict=False),
        )

    def load_module(
        self, strict: bool = False
    ):

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
    """Get config and update with possible dotlist override."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return data


def build_writer(
    config: InferenceConfig,
    distributed: Distributed,
    logger: logging.Logger | None = None,
):
    def log(msg, **kwargs):
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

    return_spatial_mask= module.model_config.EXPECTS_MASK
    inference_loader = config.inference_loader.build_inference_loader(return_spatial_mask=return_spatial_mask)

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
