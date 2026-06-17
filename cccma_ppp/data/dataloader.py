import numpy as np
import dataclasses
import torch
import warnings
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from functools import partial
from pathlib import Path

from typing import final
from collections.abc import Callable, Iterator
from itertools import islice

from cccma_ppp.data.datasets import TrainDataset, TrainDatasetConfig
from cccma_ppp.data.utils_data import _create_train_mask, WeightsConfig
from cccma_ppp.preprocessing.preprocessing_ABC import PreprocessModuleABC
from cccma_ppp.generic.distributed import Distributed


@dataclasses.dataclass
class BatchData:
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
class TrainDataloaderConfig:
    dataset_config: TrainDatasetConfig
    batch_size: int
    train_years: tuple | list = None
    num_validation_years: int = 0
    num_data_workers: int = 0
    prefetch_factor: int = 2
    drop_last: bool = False

    def __post_init__(self):
        self._setup = False
        self.dataset_processor = self.dataset_config.build_operator()
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
            self.dataset_processor._fit_preprocessors(
                self.train_years, save=True, save_path=save_path
            )

        distributed.barrier()

        if distributed.distributed:
            self.dataset_processor._load_fitted_preprocessors(load_dir=save_path)

        self._setup = True

    def _add_fitted_preprocessor(self, preprocessor: PreprocessModuleABC, index=0):
        assert preprocessor.fitted, "The preprocessor must be fitted"
        self.dataset_processor._add_fitted_preprocessor(preprocessor, index)

    def build_train_loader(self, return_spatial_mask=False, reduce_spatial_mask=False):
        if not self._setup:
            raise RuntimeError(
                "Dataloader has to be setup for distributed training first by calling .setup_distributed()"
            )

        train_mask = _create_train_mask(
            years=self.train_years,
            lead_times=np.arange(1, self.dataset_config.num_lead_months + 1),
        )

        train_dataset = self.dataset_processor.build_dataset(
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
                lead_times=np.arange(1, self.dataset_config.num_lead_months + 1),
            )
            validation_dataset = self.dataset_processor.build_dataset(
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

    @property
    def input_var_metadata(self):
        return self.dataset_processor.get_input_var_metadata()

    @property
    def target_var_metadata(self):
        return self.dataset_processor.get_target_var_metadata()


@dataclasses.dataclass
class Dataloader:
    config: TrainDataloaderConfig
    dataset: TrainDataset
    collate_fn: Callable
    rank: int = 0
    world_size: int = 1
    return_spatial_mask: bool = False
    reduce_spatial_mask: bool = False

    def __post_init__(self):

        self.sampler = self._get_dataloader_sampler()
        shuffle = self.world_size == 1
        self._torch_loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            sampler=self.sampler,
            collate_fn=partial(
                self.collate_fn,
                return_spatial_mask=self.return_spatial_mask,
                reduce_spatial_mask=self.reduce_spatial_mask,
            ),
            num_workers=self.config.num_data_workers,
            prefetch_factor=self.config.prefetch_factor,
        )

    def get_weights(self, config: WeightsConfig | None = None):

        return self.config.dataset_processor.get_weights(config)

    @property
    def input_shape(self):
        return self.dataset.get_input_shape()

    @property
    def target_shape(self):
        return self.dataset.get_target_shape()

    @property
    def added_features_dim(self):
        return self.dataset.get_added_features_dim()

    @final
    def __iter__(self) -> Iterator[BatchData]:
        return iter(self._torch_loader)

    @final
    def _get_dataloader_sampler(self, **kwargs):

        if self.world_size > 1:
            return DistributedSampler(
                self.dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=self.config.drop_last,
                **kwargs,
            )

        return None

    @final
    def __len__(self) -> int:
        return len(self._torch_loader)

    @final
    def set_epoch(self, epoch):
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)
        return self

    @final
    def subset_loader(self, start_batch=0):

        return islice(iter(self), start_batch, None)


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