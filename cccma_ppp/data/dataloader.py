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

from cccma_ppp.data.datasets import XArrayDataset, XArrayDatasetConfig
from cccma_ppp.data.utils_data import _create_train_mask, WeightsConfig
from cccma_ppp.preprocessing.preprocessing_ABC import PreprocessModuleABC
from cccma_ppp.generic.distributed import Distributed


@dataclasses.dataclass
class BatchData:
    """
    Container for batched data and optional masks.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor.
    target : torch.Tensor
        Target tensor.
    added_features : torch.Tensor, optional
        Additional input features.
    return_spatial_mask : bool, optional
        Whether to compute spatial masks.
    reduce_spatial_mask : bool, optional
        Whether to reduce masks across batch dimension.
    """

    input: torch.Tensor
    target: torch.Tensor
    added_features: torch.Tensor = None
    return_spatial_mask: bool = False
    reduce_spatial_mask: bool = False

    def __post_init__(self):
        """
        Initialize masks and replace NaNs.

        Returns
        -------
        None
        """

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
        """
        Move batch data to specified device.

        Parameters
        ----------
        device : torch.device

        Returns
        -------
        BatchData
            Updated batch on target device.
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
    Configuration for training and validation dataloaders.

    Parameters
    ----------
    dataset_config : XArrayDatasetConfig
        Dataset configuration.
    batch_size : int
        Batch size.
    train_years : tuple or list, optional
        Range of training years.
    num_validation_years : int, optional
        Number of validation years.
    num_data_workers : int, optional
        Number of worker processes.
    prefetch_factor : int, optional
        Prefetch factor for DataLoader.
    drop_last : bool, optional
        Whether to drop last batch.
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
        Initialize training/validation splits.

        Raises
        ------
        ValueError
            If requested years are not available.
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
        self,
        distributed: Distributed,
        save_path: Path | str | None = None,
    ):
        """
        Setup multiprocessing preprocessing and synchronization.

        Parameters
        ----------
        distributed : Distributed
            Distributed configuration or manager object.
        save_path : Path or str or None, optional
            Optional path for saving distributed artifacts.

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
            self.dataset_config._load_fitted_preprocessors(load_dir=save_path)

        self._setup = True

    def _add_fitted_preprocessor(self, preprocessor: PreprocessModuleABC, index=0):
        """
        Add fitted preprocessor to pipeline.

        Parameters
        ----------
        preprocessor : PreprocessModuleABC
        index : int, optional

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
        Build training dataloader.

        Parameters
        ----------
        return_spatial_mask : bool, optional
        reduce_spatial_mask : bool, optional

        Returns
        -------
        Dataloader

        Raises
        ------
        RuntimeError
            If distributed setup not completed.
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
        Build validation dataloader.

        Parameters
        ----------
        return_spatial_mask : bool, optional
        reduce_spatial_mask : bool, optional

        Returns
        -------
        Dataloader or None

        Warns
        -----
        If validation dataset is unavailable.
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
            warnings.warn(
                f"Validation dataoader could not be built for num_validation_years = {self.num_validation_years} "
            )
            return None

    @property
    def input_var_metadata(self):
        """
        Return input variable metadata.

        Returns
        -------
        dict
        """
        return self.dataset_config.get_input_var_metadata()

    @property
    def target_var_metadata(self):
        """
        Return target variable metadata.

        Returns
        -------
        dict
        """
        return self.dataset_config.get_target_var_metadata()


@dataclasses.dataclass
class Dataloader:
    """
    Wrapper around PyTorch DataLoader with distributed support.

    Parameters
    ----------
    config : TrainDataloaderConfig
    dataset : XArrayDataset
    collate_fn : Callable
    rank : int, optional
    world_size : int, optional
    return_spatial_mask : bool, optional
    reduce_spatial_mask : bool, optional
    """

    config: TrainDataloaderConfig
    dataset: XArrayDataset
    collate_fn: Callable
    rank: int = 0
    world_size: int = 1
    return_spatial_mask: bool = False
    reduce_spatial_mask: bool = False

    def __post_init__(self):
        """
        Initialize PyTorch DataLoader and sampler.

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
            collate_fn=partial(
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
        config : WeightsConfig or None

        Returns
        -------
        torch.Tensor or xr.DataArray
        """

        return self.config.dataset_config.get_weights(config)

    @property
    def input_shape(self):
        """
        Input tensor shape.

        Returns
        -------
        np.ndarray
        """
        return self.dataset.get_input_shape()

    @property
    def target_shape(self):
        """
        Target tensor shape.

        Returns
        -------
        np.ndarray
        """
        return self.dataset.get_target_shape()

    @property
    def added_features_dim(self):
        """
        Dimension of additional features.

        Returns
        -------
        int
        """
        return self.dataset.get_added_features_dim()

    @final
    def __iter__(self) -> Iterator[BatchData]:
        """
        Iterate over batches.

        Returns
        -------
        Iterator of BatchData
        """
        return iter(self._torch_loader)

    @final
    def _get_dataloader_sampler(self, **kwargs):
        """
        Create distributed sampler if needed.

        Returns
        -------
        DistributedSampler or None
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
        Number of batches.

        Returns
        -------
        int
        """
        return len(self._torch_loader)

    @final
    def set_epoch(self, epoch):
        """
        Set epoch for distributed sampler.

        Parameters
        ----------
        epoch : int

        Returns
        -------
        self
        """
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)
        return self

    @final
    def subset_loader(self, start_batch=0):
        """
        Return iterator starting from specific batch.

        Parameters
        ----------
        start_batch : int, optional

        Returns
        -------
        Iterator
        """

        return islice(iter(self), start_batch, None)


def collate_batch(
    batch, return_spatial_mask: bool = False, reduce_spatial_mask: bool = False
):
    """
    Collate dataset samples into BatchData.

    Parameters
    ----------
    batch : list of dict
        List of dataset samples.
    return_spatial_mask : bool, optional
    reduce_spatial_mask : bool, optional

    Returns
    -------
    BatchData
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
