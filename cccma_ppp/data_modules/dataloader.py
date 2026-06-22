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
    input: torch.Tensor
    target: torch.Tensor | None
    added_features: torch.Tensor | None

    @abc.abstractmethod
    def to_device(self, device: torch.device | str):
        pass


class DataloaderConfigABC(abc.ABC):
    dataset_config: DatasetConfigABC

    @abc.abstractmethod
    def setup_distributed(self):
        pass


@dataclasses.dataclass
class Dataloader:
    config: DataloaderConfigABC
    dataset: Dataset
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
    def __iter__(self) -> Iterator[BatchDataABC]:
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
