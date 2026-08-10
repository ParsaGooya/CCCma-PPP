import dataclasses
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from functools import partial
import abc
from typing import final, ClassVar
from collections.abc import Callable, Iterator
from itertools import islice


from cccma_ppp.data_modules.dataset.dataset_abc import (
    DatasetConfigABC,
    AddedTimeFeatures,
)

from cccma_ppp.configs import required_sample_dimensions

init_time_dim, lead_time_dim = required_sample_dimensions


class BatchDataABC(abc.ABC):
    """
    Document this class.

    Attributes
    ----------
    input : torch.Tensor
        Description not yet provided.
    target : torch.Tensor | None
        Description not yet provided.
    added_features : torch.Tensor | None
        Description not yet provided.
    metadata : list[dict] | None
        Description not yet provided.
    """

    input: torch.Tensor
    target: torch.Tensor | None
    added_features: torch.Tensor | None
    metadata: list[dict] | None = None

    _shared_input_mask: ClassVar[torch.Tensor | None] = None
    _shared_target_mask: ClassVar[torch.Tensor | None] = None

    @abc.abstractmethod
    def to_device(self, device: torch.device | str):
        """
        Document this function.

        Parameters
        ----------
        device : torch.device | str
            Description not yet provided.
        """
        pass


class DataloaderConfigABC(abc.ABC):
    """
    Document this class.

    Attributes
    ----------
    dataset_config : DatasetConfigABC
        Description not yet provided.
    pin_memory : bool
        Description not yet provided.
    time_features : AddedTimeFeatures | list[str] | None
        Description not yet provided.
    prefetch_factor : int | None
        Description not yet provided.
    return_spatial_mask : bool
        Description not yet provided.
    reduce_spatial_mask : bool
        Description not yet provided.
    """

    dataset_config: DatasetConfigABC
    pin_memory: bool
    time_features: AddedTimeFeatures | list[str] | None
    prefetch_factor: int | None
    return_spatial_mask: bool
    reduce_spatial_mask: bool

    init_time_dim: ClassVar[str] = init_time_dim
    lead_time_dim: ClassVar[str] = lead_time_dim

    def __init__(self):
        """
        Document this function.
        """
        self._setup = False
        self.pin_memory = False

        if self.num_data_workers == 0:
            self.prefetch_factor = None

    @abc.abstractmethod
    def setup_distributed(self):
        """
        Document this function.
        """
        pass

    @property
    @abc.abstractmethod
    def available_times(self):
        """
        Document this function.
        """
        pass

    @final
    def select_requested_times(
        self,
        requested_slice: slice,
    ) -> xr.DataArray:
        """
        Document this function.

        Parameters
        ----------
        requested_slice : slice
            Description not yet provided.

        Returns
        -------
        xr.DataArray
            Description not yet provided.

        Raises
        ------
        ValueError
            Description not yet provided.
        """
        if isinstance(self.available_times, xr.DataArray):
            times = self.available_times
        else:
            times = xr.DataArray(
                self.available_times,
                dims=(self.init_time_dim,),
                coords={self.init_time_dim: self.available_times},
            )

        times = times.sortby(self.init_time_dim)

        available_start = times.dt.year.min().item()
        available_stop = times.dt.year.max().item()

        if (
            eval(requested_slice.start) is not None
            and eval(requested_slice.start) < available_start
        ):
            raise ValueError(
                f"Requested start time {int(requested_slice.start)} is before "
                f"the first available time {available_start}."
            )

        if (
            eval(requested_slice.stop) is not None
            and eval(requested_slice.stop) > available_stop
        ):
            raise ValueError(
                f"Requested stop time {eval(requested_slice.stop)} is after "
                f"the final available time {available_stop}."
            )

        selected = times.sel({self.init_time_dim: requested_slice})

        if selected.size == 0:
            raise ValueError(
                "No available times fall inside the requested range "
                f"[{requested_slice.start}, {requested_slice.stop}]."
            )

        return selected


@dataclasses.dataclass
class Dataloader:
    """
    Document this class.

    Parameters
    ----------
    config : DataloaderConfigABC
        Description not yet provided.
    dataset : Dataset
        Description not yet provided.
    collate_fn : Callable
        Description not yet provided.
    rank : int
        Description not yet provided.
    world_size : int
        Description not yet provided.
    shuffle : bool | None
        Description not yet provided.
    return_spatial_mask : bool
        Description not yet provided.
    """

    config: DataloaderConfigABC
    dataset: Dataset
    collate_fn: Callable
    rank: int = 0
    world_size: int = 1
    shuffle: bool | None = None
    return_spatial_mask: bool = False

    def __post_init__(self):
        """
        Document this function.
        """
        self.sampler = self._get_dataloader_sampler()
        shuffle = self.world_size == 1 if self.shuffle is None else self.shuffle
        num_workers = self.config.num_data_workers

        self._torch_loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            sampler=self.sampler,
            collate_fn=partial(
                self.collate_fn,
                return_spatial_mask=self.return_spatial_mask,
                reduce_spatial_mask=self.config.reduce_spatial_mask,
            ),
            num_workers=num_workers,
            prefetch_factor=self.config.prefetch_factor,
            persistent_workers=num_workers > 0,
            pin_memory=self.config.pin_memory,
            timeout=60 if num_workers > 0 else 0,
        )

    @property
    def input_shape(self) -> tuple:
        """
        Document this function.

        Returns
        -------
        tuple
            Description not yet provided.
        """
        return self.dataset.get_input_shape()

    @property
    def target_shape(self):
        """
        Document this function.

        Returns
        -------
        Any
            Description not yet provided.
        """
        return self.dataset.get_target_shape()

    @property
    def added_features_dim(self):
        """
        Document this function.

        Returns
        -------
        Any
            Description not yet provided.
        """
        return self.dataset.get_added_features_dim()

    @final
    def __iter__(self) -> Iterator[BatchDataABC]:
        """
        Document this function.

        Returns
        -------
        Iterator[BatchDataABC]
            Description not yet provided.
        """
        return iter(self._torch_loader)

    @final
    def _get_dataloader_sampler(self, **kwargs) -> torch.utils.data.Sampler | None:
        """
        Document this function.

        Parameters
        ----------
        **kwargs : Any
            Description not yet provided.

        Returns
        -------
        torch.utils.data.Sampler | None
            Description not yet provided.
        """
        if self.world_size > 1:
            shuffle = self.world_size > 1 if self.shuffle is None else self.shuffle
            return DistributedSampler(
                self.dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle,
                drop_last=self.config.drop_last,
                **kwargs,
            )

        return None

    @final
    def __len__(self) -> int:
        """
        Document this function.

        Returns
        -------
        int
            Description not yet provided.
        """
        return len(self._torch_loader)

    @final
    def set_epoch(self, epoch):
        """
        Document this function.

        Parameters
        ----------
        epoch : Any
            Description not yet provided.

        Returns
        -------
        Any
            Description not yet provided.
        """
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)
        return self

    @final
    def subset_loader(self, start_batch=0):
        """
        Document this function.

        Parameters
        ----------
        start_batch : Any
            Description not yet provided.

        Returns
        -------
        Any
            Description not yet provided.
        """
        return islice(iter(self), start_batch, None)
