import numpy as np
import dataclasses
import warnings
import torch
from pathlib import Path


from cccma_ppp.train.datasets import TrainDatasetConfig
from cccma_ppp.data_modules import _create_train_mask, WeightsConfig
from cccma_ppp.data_modules.dataloader import Dataloader, DataloaderConfigABC, BatchDataABC
from cccma_ppp.generic import Distributed


@dataclasses.dataclass
class BatchData(BatchDataABC):
    input: torch.Tensor
    target: torch.Tensor
    added_features: torch.Tensor = None
    return_spatial_mask: bool = False
    reduce_spatial_mask: bool = False

    def __post_init__(self):
        if self.return_spatial_mask:
            self.input_mask = (~torch.isnan(self.input)).to(torch.int)
            self.target_mask = (~torch.isnan(self.target)).to(torch.int)
            if self.reduce_spatial_mask:
                self.input_mask = self.input_mask.mean(0)
                self.input_mask = (self.input_mask == 1).float()
                self.target_mask = self.target_mask.mean(0)
                self.target_mask = (self.target_mask == 1).float()

        self.input = torch.nan_to_num(self.input, nan=0.0)
        self.target = torch.nan_to_num(self.target, nan=0.0)

        if self.return_spatial_mask:
            self.input = (self.input, self.input_mask)
            self.target = (self.target, self.target_mask)

    def to_device(self, device):

        if self.return_spatial_mask:
            self.input = (self.input[0].to(device), self.input[1].to(device))
            self.target = (self.target[0].to(device), self.target[1].to(device))
        else:
            self.input = self.input.to(device)
            self.target = self.target.to(device)

        if self.added_features is not None:
            self.added_features = self.added_features.to(device)

        return self


@dataclasses.dataclass
class TrainDataloaderConfig(DataloaderConfigABC):
    dataset_config: TrainDatasetConfig
    batch_size: int
    train_years: tuple | list = None
    num_validation_years: int = 0
    num_data_workers: int = 0
    prefetch_factor: int = 2
    drop_last: bool = False

    def __post_init__(self):
        self._setup = False
        self.available_train_years = self.dataset_config.available_train_time

        if self.num_validation_years > 0:
            self.available_train_years = self.dataset_config.available_train_time[
                : -self.num_validation_years
            ]

        if self.num_data_workers == 0:
            self.prefetch_factor = None

        if self.train_years is None:
            self.train_years = self.available_train_years
            if self.num_validation_years > 0:
                self.validation_years = self.dataset_config.available_train_time[
                    -self.num_validation_years :
                ]
        else:
            self.train_years = np.arange(self.train_years[0], self.train_years[1] + 1)

            if not set(self.train_years).issubset(set(self.available_train_years)):
                raise ValueError(
                    f"the requested train years are not available: available years: [{self.available_train_years.min()},{self.available_train_years.max()}]"
                )

            if self.num_validation_years > 0:
                self.validation_years = np.arange(
                    self.train_years[-1] + 1,
                    self.train_years[-1] + 1 + self.num_validation_years,
                )

    def setup_distributed(
        self, distributed: Distributed, save_path: Path | str | None = None
    ):

        self.rank = distributed.rank
        self.world_size = distributed.world_size
        

        if distributed.is_root():
            self.dataset_config._fit_preprocessors(
                self.train_years, save=True, save_path=save_path
            )

        distributed.barrier()

        if distributed.distributed:
            self.dataset_config._load_fitted_preprocessors(load_dir=save_path)

        self._setup = True


    def build_train_loader(self, return_spatial_mask=False, reduce_spatial_mask=False):
        if not self._setup:
            raise RuntimeError(
                "Dataloader has to be setup for distributed training first by calling .setup_distributed()"
            )

        train_mask = _create_train_mask(
            years=self.train_years,
            lead_times=self.dataset_config.lead_months,
        )

        train_dataset = self.dataset_config.build_dataset(
            years=self.train_years, mask=train_mask, return_metadata=False
        )

        return Dataloader(
            dataset=train_dataset,
            config=self,
            collate_fn=collate_batch,
            rank=self.rank,
            world_size=self.world_size,
            return_spatial_mask=return_spatial_mask,
            reduce_spatial_mask=reduce_spatial_mask,
        )

    def build_validation_loader(
        self, return_spatial_mask=False, reduce_spatial_mask=False
    ):
        if not self._setup:
            raise RuntimeError(
                "Dataloader has to be setup for distributed training first by calling .setup_distributed()"
            )

        if self.num_validation_years > 0:
            validation_mask = _create_train_mask(
                years=self.validation_years,
                lead_times=self.dataset_config.lead_months,
            )
            validation_dataset = self.dataset_config.build_dataset(
                years=self.validation_years, mask=validation_mask, return_metadata=False
            )
            return Dataloader(
                dataset=validation_dataset,
                config=self,
                collate_fn=collate_batch,
                rank=self.rank,
                world_size=self.world_size,
                return_spatial_mask=return_spatial_mask,
                reduce_spatial_mask=reduce_spatial_mask,
            )

        else:
            warnings.warn(
                f"Validation dataoader could not be built for num_validation_years = {self.num_validation_years} "
            )
            return None
        
    def get_weights(self, config: WeightsConfig | None = None):
        return self.dataset_config.ds_operator.get_weights(config)

    @property
    def input_var_metadata(self):
        return self.dataset_config.ds_operator.get_input_var_metadata()

    @property
    def target_var_metadata(self):
        return self.dataset_config.ds_operator.get_target_var_metadata()




def collate_batch(
    batch,
    return_spatial_mask: bool = False,
    reduce_spatial_mask: bool = False,
):
    metadata = None

    if isinstance(batch[0], tuple):
        batch, metadata = zip(*batch)

    inputs = torch.stack([b["input"] for b in batch])
    targets = torch.stack([b["target"] for b in batch])

    added_features = None
    if batch[0]["added_features"] is not None:
        added_features = torch.stack([b["added_features"] for b in batch])

    batch_data = BatchData(
        input=inputs,
        target=targets,
        added_features=added_features,
        return_spatial_mask=return_spatial_mask,
        reduce_spatial_mask=reduce_spatial_mask,
    )

    if metadata is not None:
        return batch_data, list(metadata)

    return batch_data