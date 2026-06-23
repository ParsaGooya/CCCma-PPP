import dataclasses
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from functools import partial
import abc
from typing import final
from collections.abc import Callable, Iterator
from itertools import islice

from cccma_ppp.data_modules.dataset import DatasetConfigABC


class BatchDataABC(abc.ABC):
    """
    Abstract base class for batched data containers.

    Attributes
    ----------
    input : torch.Tensor
        Input batch.
    target : torch.Tensor or None
        Target batch.
    added_features : torch.Tensor or None
        Additional features.
    """

    input: torch.Tensor
    target: torch.Tensor | None
    added_features: torch.Tensor | None

    @abc.abstractmethod
    def to_device(self, device: torch.device | str):
        """
        Move batch to specified device.

        Parameters
        ----------
        device : torch.device or str
            Target device to move the batch to.

        Returns
        -------
        BatchDataABC
            Batch moved to the target device.
        """
        pass


class DataloaderConfigABC(abc.ABC):
    """
    Abstract base class for dataloader configuration.

    Attributes
    ----------
    dataset_config : DatasetConfigABC
        Dataset configuration.
    """

    dataset_config: DatasetConfigABC

    @abc.abstractmethod
    def setup_distributed(self):
        """
        Prepare dataloader for distributed execution.

        Returns
        -------
        None
        """
        pass


@dataclasses.dataclass
class Dataloader:
    """
    Wrapper around PyTorch DataLoader with distributed support.

    Parameters
    ----------
    config : DataloaderConfigABC
        Dataloader configuration.
    dataset : torch.utils.data.Dataset
        Dataset instance.
    rank : int, optional
        Process rank in distributed setup.
    world_size : int, optional
        Total number of processes.
    return_spatial_mask : bool, optional
        Whether to include spatial masks.
    reduce_spatial_mask : bool, optional
        Whether to reduce masks across batch dimension.
    """

    config: DataloaderConfigABC
    dataset: Dataset
    collate_fn: Callable
    rank: int = 0
    world_size: int = 1
    return_spatial_mask: bool = False
    reduce_spatial_mask: bool = False

    def __post_init__(self):
        """
        Initialize PyTorch DataLoader.

        Sets up batching, distributed sampling, and collate function.

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

    @property
    def input_shape(self) -> tuple:
        """
        Input data shape.

        Returns
        -------
        tuple
            Shape of the input data.
        """
        return self.dataset.get_input_shape()

    @property
    def target_shape(self):
        """
        Target data shape.

        Returns
        -------
        tuple
        """
        return self.dataset.get_target_shape()

    @property
    def added_features_dim(self):
        """
        Additional feature dimension.

        Returns
        -------
        int
        """

        return self.dataset.get_added_features_dim()

    @final
    def __iter__(self) -> Iterator[BatchDataABC]:
        """
        Iterate over batches.

        Returns
        -------
        Iterator of BatchDataABC
        """
        return iter(self._torch_loader)

    @final
    def _get_dataloader_sampler(self, **kwargs) -> torch.utils.data.Sampler | None:
        """
        Create distributed sampler if needed.

        Returns
        -------
        torch.utils.data.Sampler or None
            Sampler for distributed training, or None if not required.
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
        Number of batches per epoch.

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
        Dataloader
        """
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)
        return self

    @final
    def subset_loader(self, start_batch=0):
        """
        Create iterator over subset of batches.

        Parameters
        ----------
        start_batch : int, optional
            Starting batch index.

        Returns
        -------
        iterator
        """

        return islice(iter(self), start_batch, None)
