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

from src.cccma_ppp.data.datasets import XArrayDataset, XArrayDatasetConfig
from src.cccma_ppp.data.utils_data import _create_train_mask, WeightsConfig
from src.cccma_ppp.preprocessing.preprocessing_ABC import PreprocessModuleABC
from src.cccma_ppp.generic.distributed import Distributed


@dataclasses.dataclass
class BatchData:
    """
    Container for batched input, target, and optional feature tensors.
    """

    input: torch.Tensor
    target: torch.Tensor
    added_features: torch.Tensor = None
    return_spatial_mask: bool = False
    reduce_spatial_mask: bool = False  # creates static mask

    def __post_init__(self):
        """
        Initialize batch data and optionally compute spatial masks.

        Returns
        -------
        None
        """

        if self.return_spatial_mask:
            self.input_mask = (~torch.isnan(self.input)).to(torch.float)
            self.target_mask = (~torch.isnan(self.target)).to(torch.float)
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
        """
        Move batch data to a specified device.

        Parameters
        ----------
        device : torch.device or str
            Target device.

        Returns
        -------
        BatchData
            Batch moved to the specified device.
        """

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
    """
    Configuration for constructing training and validation dataloaders.
    """

    dataset_config: XArrayDatasetConfig
    batch_size: int
    train_years: tuple | list = None
    num_validation_years: int = 0
    num_data_workers: int = 0
    prefetch_factor: int = 2
    drop_last: bool = False

    def __post_init__(self):
        """
        Initialize train/validation splits and validate configuration.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If requested training years are invalid.
        """

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

            assert set(self.train_years).issubset(set(self.available_train_years)), (
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
        """
        Set up preprocessing for distributed training.

        Parameters
        ----------
        distributed : Distributed
            Distributed training controller.
        save_path : Path or str, optional
            Path for saving or loading preprocessors.

        Returns
        -------
        None
        """

        self.rank = distributed.rank
        self.world_size = distributed.world_size

        if distributed.is_root():
            self.dataset_config._fit_preprocessors(
                self.train_years, save=True, save_path=save_path
            )

        distributed.barrier()

        if distributed.distributed:
            self.dataset_config._load_fitted_preprocessors(
                load_dir=save_path
            )  # all ranks load from disk after the barrier, including rank 0, to guarantee everyone uses the exact saved state

        self._setup = True

    def _add_fitted_preprocessor(self, preprocessor: PreprocessModuleABC, index=0):
        """
        Add a fitted preprocessor to the dataset configuration.

        Parameters
        ----------
        preprocessor : PreprocessModuleABC
            Preprocessor to add.
        index : int, optional
            Insertion index.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If preprocessor is not fitted.
        """

        assert preprocessor.fitted, "The preprocessor must be fitted"
        self.dataset_config._add_fitted_preprocessor(preprocessor, index)

    def build_train_loader(self, return_spatial_mask=False, reduce_spatial_mask=False):
        """
        Build the training dataloader.

        Parameters
        ----------
        return_spatial_mask : bool, optional
            Whether to include spatial masks.
        reduce_spatial_mask : bool, optional
            Whether to reduce masks.

        Returns
        -------
        Dataloader
            Training dataloader instance.

        Raises
        ------
        RuntimeError
            If distributed setup is not completed.
        """

        if not self._setup:
            raise RuntimeError(
                "Dataloader has to be setup for distributed training first by calling .setup_distributed()"
            )

        train_mask = _create_train_mask(
            years=self.train_years,
            lead_times=np.arange(1, self.dataset_config.num_lead_months + 1),
        )

        train_dataset = self.dataset_config.build(
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
        """
        Build the validation dataloader.

        Parameters
        ----------
        return_spatial_mask : bool, optional
            Whether to include spatial masks.
        reduce_spatial_mask : bool, optional
            Whether to reduce masks.

        Returns
        -------
        Dataloader or None
            Validation dataloader or None if unavailable.

        Raises
        ------
        RuntimeError
            If distributed setup is not completed.
        """

        if not self._setup:
            raise RuntimeError(
                "Dataloader has to be setup for distributed training first by calling .setup_distributed()"
            )

        if self.num_validation_years > 0:
            validation_mask = _create_train_mask(
                years=self.validation_years,
                lead_times=np.arange(1, self.dataset_config.num_lead_months + 1),
            )
            validation_dataset = self.dataset_config.build(
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
            # raise ValueError(f'Validation dataoader could not be built for num_validation_years = {self.num_validation_years} ')
            warnings.warn(
                f"Validation dataoader could not be built for num_validation_years = {self.num_validation_years} "
            )
            return None


@dataclasses.dataclass
class Dataloader:
    """
    Wrapper around PyTorch DataLoader with distributed support.
    """

    config: TrainDataloaderConfig  # | InferenceDataloaderConfig
    dataset: XArrayDataset
    collate_fn: Callable
    rank: int = 0
    world_size: int = 1
    return_spatial_mask: bool = False  ##later to be used for partial convs
    reduce_spatial_mask: bool = False  ##later to be used for partial convs

    def __post_init__(self):
        """
        Initialize underlying PyTorch DataLoader and sampler.

        Returns
        -------
        None
        """

        self.sampler = self._get_dataloader_sampler()
        shuffle = self.world_size == 1
        self._torch_loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            sampler=self.sampler,
            collate_fn=partial(  ###Safer alternative for lambda to avoid break on some multiprocessing backends because lambdas are not pickleable.
                self.collate_fn,
                return_spatial_mask=self.return_spatial_mask,
                reduce_spatial_mask=self.reduce_spatial_mask,
            ),
            num_workers=self.config.num_data_workers,
            prefetch_factor=self.config.prefetch_factor,
        )

    def get_weights(self, config: WeightsConfig | None = None):
        """
        Retrieve dataset weights.

        Parameters
        ----------
        config : WeightsConfig, optional
            Weight configuration.

        Returns
        -------
        object
            Computed weights.
        """

        return self.config.dataset_config.get_weights(config)

    @property
    def input_shape(self):
        """
        Get shape of input data.

        Returns
        -------
        tuple
            Input tensor shape.
        """
        return self.dataset.get_input_shape()

    @property
    def target_shape(self):
        """
        Get shape of target data.

        Returns
        -------
        tuple
            Target tensor shape.
        """
        return self.dataset.get_target_shape()

    @property
    def added_features_dim(self):
        """
        Get dimensionality of additional features.

        Returns
        -------
        int
            Number of additional feature dimensions.
        """
        return self.dataset.added_features_dim

    @final
    def __iter__(self) -> Iterator[BatchData]:
        """
        Create iterator over data batches.

        Returns
        -------
        Iterator of BatchData
            Batch iterator.
        """

        return iter(self._torch_loader)

    @final
    def _get_dataloader_sampler(self, **kwargs):
        """
        Create distributed sampler if required.

        Returns
        -------
        DistributedSampler or None
            Sampler instance.
        """

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
        """
        Return number of batches.

        Returns
        -------
        int
            Number of batches.
        """
        return len(self._torch_loader)

    @final
    def set_epoch(self, epoch):
        """
        Set epoch for distributed sampling.

        Parameters
        ----------
        epoch : int
            Epoch index.

        Returns
        -------
        Dataloader
            Updated dataloader.
        """

        if self.sampler is not None:
            self.sampler.set_epoch(epoch)
        return self

    @final
    def subset_loader(self, start_batch=0):
        """
        Create iterator starting from a given batch index.

        Parameters
        ----------
        start_batch : int, optional
            Starting batch index.

        Returns
        -------
        iterator
            Subset iterator.
        """

        return islice(iter(self), start_batch, None)


def collate_batch(
    batch, return_spatial_mask: bool = False, reduce_spatial_mask: bool = False
):
    """
    Collate dataset samples into a batched BatchData object.

    Parameters
    ----------
    batch : list
        List of samples.
    return_spatial_mask : bool, optional
        Whether to compute spatial masks.
    reduce_spatial_mask : bool, optional
        Whether to reduce masks.

    Returns
    -------
    BatchData
        Batched data container.
    """

    inputs = torch.stack([b["input"] for b in batch])
    targets = torch.stack([b["target"] for b in batch])

    added_features = None
    if batch[0]["added_features"] is not None:
        added_features = torch.stack([b["added_features"] for b in batch])

    return BatchData(
        input=inputs,
        target=targets,
        added_features=added_features,
        return_spatial_mask=return_spatial_mask,
        reduce_spatial_mask=reduce_spatial_mask,
    )
