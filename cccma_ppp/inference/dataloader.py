import numpy as np
import dataclasses
import torch
from pathlib import Path

from cccma_ppp.train.dataset import TrainDatasetConfig
from cccma_ppp.train.dataloader import TrainDataloaderConfig
from cccma_ppp.inference.dataset import InferenceDatasetConfig, _from_train
from cccma_ppp.data_modules.dataloader import (
    Dataloader,
    DataloaderConfigABC,
    BatchDataABC,
)
from cccma_ppp.generic.distributed import Distributed
from cccma_ppp.generic.runtime import RuntimeContext


@dataclasses.dataclass
class BatchData(BatchDataABC):
    input: torch.Tensor
    added_features: torch.Tensor = None
    metadata: list[dict] | None = None
    return_spatial_mask: bool = False
    reduce_spatial_mask: bool = False

    def __post_init__(self):
        if self.return_spatial_mask:
            self.input_mask = (~torch.isnan(self.input)).to(torch.int)
            if self.reduce_spatial_mask:
                self.input_mask = self.input_mask.mean(0)
                self.input_mask = (self.input_mask == 1).float()

        self.input = torch.nan_to_num(self.input, nan=0.0)

        if self.return_spatial_mask:
            self.input = (self.input, self.input_mask)

    def to_device(self, device: torch.device | str):

        if self.return_spatial_mask:
            self.input = (self.input[0].to(device), self.input[1].to(device))
        else:
            self.input = self.input.to(device)

        if self.added_features is not None:
            self.added_features = self.added_features.to(device)

        return self


@dataclasses.dataclass
class InferenceDataloaderConfig(DataloaderConfigABC):
    dataset_config: InferenceDatasetConfig | None = None
    batch_size: int = 1
    inference_years: tuple | list = None
    num_data_workers: int = 0
    prefetch_factor: int | None = None
    drop_last: bool = False

    def __post_init__(self):
        self._setup = False
        self.train_dataset_config = None

        if self.num_data_workers == 0:
            self.prefetch_factor = None

        if self.dataset_config is not None:
            _ = self._inference_years

    def _check_dataset_config(self):
        if self.dataset_config is None:
            raise RuntimeError(
                "dataset_config must be provided or read froim train configs"
                "via  read_datasetconfig_from_train method."
            )

    def read_datasetConfig_from_train(self, train_dataset_config: TrainDatasetConfig):
        if self.dataset_config is None:
            self.dataset_config = _from_train(train_dataset_config)
            _ = self._inference_years
            self.train_dataset_config = train_dataset_config

    @property
    def _inference_years(self):
        if self.inference_years is None:
            return self.available_inference_years
        else:
            inference_years = np.arange(
                self.inference_years[0], self.inference_years[1] + 1
            )

            if not set(inference_years).issubset(set(self.available_inference_years)):
                raise ValueError(
                    f"the requested inference years are not available:"
                    f"available years: [{self.available_inference_years.min()},{self.available_inference_years.max()}]"
                )

            return inference_years

    @property
    def available_inference_years(self):
        self._check_dataset_config()
        return self.dataset_config.available_inference_years

    def _input_preprocessor_exists(self, load_dir: Path | str = None):
        self._check_dataset_config()
        preprocessor_to_check = []
        exists = []

        if load_dir is None:
            load_dir = Path(RuntimeContext.GLOBAL_EXP_DIR) / "preprocessing_pipeline"
        else:
            load_dir = Path(load_dir)

        if self.dataset_config.model is not None:
            preprocessor_name = self.dataset_config.model.preprocessing_pipeline.name
            preprocessor_to_check.append(
                load_dir / f"{preprocessor_name}_preprocessing_pipeline.joblib"
            )

        if self.dataset_config.condition is not None:
            preprocessor_name = (
                self.dataset_config.condition.preprocessing_pipeline.name
            )
            preprocessor_to_check.append(
                load_dir / f"{preprocessor_name}_preprocessing_pipeline.joblib"
            )

        for path in preprocessor_to_check:
            exists.append(path.exists())

        return all(exists)

    def setup_distributed(
        self,
        train_loader_config: TrainDataloaderConfig,
        distributed: Distributed,
        load_path: Path | str | None = None,
    ):

        self._check_dataset_config()
        self.rank = distributed.rank
        self.world_size = distributed.world_size

        if not self._input_preprocessor_exists(load_path):
            if distributed.is_root():
                train_loader_config.dataset_config._fit_preprocessors(
                    train_loader_config.dataset_config.train_years,
                    save=True,
                    save_path=load_path,
                )

        distributed.barrier()

        self.dataset_config._load_fitted_preprocessors(load_dir=load_path)

        self._setup = True

    def build_inference_loader(
        self, return_spatial_mask=False, reduce_spatial_mask=False
    ):
        if not self._setup:
            raise RuntimeError(
                "Dataloader has to be setup for distributed training first by calling .setup_distributed()"
            )

        inference_dataset = self.dataset_config.build_dataset(
            years=self._inference_years, return_metadata=True
        )

        return Dataloader(
            dataset=inference_dataset,
            config=self,
            collate_fn=collate_batch,
            rank=self.rank,
            world_size=self.world_size,
            shuffle=False,
            return_spatial_mask=return_spatial_mask,
            reduce_spatial_mask=reduce_spatial_mask,
        )

    @property
    def input_var_metadata(self):
        self._check_dataset_config()
        return self.dataset_config.ds_operator.get_input_var_metadata()

    @property
    def target_var_metadata(self):
        if self.train_dataset_config is None:
            raise RuntimeError(
                "output variables metadata cannot be read unless train dataloader "
                "is available. Hint: run setup_distributed(TrainDatasetConfig, ...)."
            )
        return self.train_dataset_config.ds_operator.get_target_var_metadata()


def collate_batch(
    batch,
    return_spatial_mask: bool = False,
    reduce_spatial_mask: bool = False,
):
    metadata = None

    if isinstance(batch[0], tuple):
        batch, metadata = zip(*batch)

    inputs = torch.stack([b["input"] for b in batch])

    added_features = None
    if batch[0]["added_features"] is not None:
        added_features = torch.stack([b["added_features"] for b in batch])

    return BatchData(
        input=inputs,
        added_features=added_features,
        metadata=list(metadata) if metadata is not None else None,
        return_spatial_mask=return_spatial_mask,
        reduce_spatial_mask=reduce_spatial_mask,
    )
