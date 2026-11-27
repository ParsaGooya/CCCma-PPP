import dataclasses
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from functools import partial
import abc
from typing import final, ClassVar, Sequence
from collections.abc import Callable, Iterator
from itertools import islice


from S2D4P.data.dataset.dataset_abc import (
    DatasetConfigABC,
    AddedTimeFeatures,
)

from S2D4P.configs import required_sample_dimensions

init_time_dim, lead_time_dim = required_sample_dimensions


class BatchDataABC(abc.ABC):
    """
    Abstract base class for batched dataset samples.

    Defines the common interface for storing batched model inputs, targets,
    additional features, metadata, and shared spatial validity masks.

    Attributes
    ----------
    input : torch.Tensor
        Batched model input tensor.
    target : torch.Tensor | None
        Batched target tensor, or ``None`` when targets are unavailable.
    added_features : torch.Tensor | None
        Optional additional features associated with the batch.
    metadata : list[dict] | None
        Optional per-sample metadata.
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
        Move batch data to a device.

        Parameters
        ----------
        device : torch.device | str
            Device to which the batch tensors are moved.

        Returns
        -------
        BatchDataABC
            Batch instance with applicable tensors moved to the requested device.
        """
        pass


class DataloaderConfigABC(abc.ABC):
    """
    Abstract base configuration for data loaders.

    Defines common dataset, worker, memory, temporal-feature, and spatial-mask
    settings used to construct training and inference data loaders.

    Attributes
    ----------
    dataset : DatasetConfigABC
        Configuration used to construct the underlying dataset.
    pin_memory : bool
        Whether data-loader batches are allocated in pinned host memory.
    time_features : AddedTimeFeatures | list[str] | None
        Temporal features generated for dataset samples.
    prefetch_factor : int | None
        Number of batches prefetched by each worker process.
    return_spatial_mask : bool
        Whether batches include spatial validity masks.
    share_spatial_mask : bool
        Whether spatial validity masks are reduced and shared across samples.
    """

    dataset: DatasetConfigABC
    pin_memory: bool
    time_features: AddedTimeFeatures | list[str] | None
    prefetch_factor: int | None
    return_spatial_mask: bool
    share_spatial_mask: bool

    init_time_dim: ClassVar[str] = init_time_dim
    lead_time_dim: ClassVar[str] = lead_time_dim

    def __init__(self):
        """
        Initialize common data-loader settings.

        Marks distributed setup as incomplete, disables pinned memory initially, and
        configures worker prefetching. Prefetching is disabled when no worker
        processes are used.

        Raises
        ------
        ValueError
            If ``prefetch_factor`` is less than one when data-loading workers are
            enabled.
        """
        self._setup = False
        self.pin_memory = False
        self.dataset_config = self.dataset

        if self.num_data_workers == 0:
            self.prefetch_factor = None
        elif self.prefetch_factor is not None:
            if self.prefetch_factor < 1:
                raise ValueError(
                    f"'prefetch_factor' must be at least 1 when workers > 0, "
                    f"got {self.prefetch_factor}."
                )

    @abc.abstractmethod
    def setup_distributed(self):
        """
        Configure the data loader for distributed execution.

        Subclasses must implement distributed rank, world-size, preprocessing, and
        device-specific data-loader setup.
        """
        pass

    @property
    @abc.abstractmethod
    def available_times(self):
        """
        Return initialization times available to the data loader.

        Returns
        -------
        Any
            Initialization times available from the configured dataset.
        """
        pass

    @final
    def _to_year_slice(years: Sequence):
        if years is None:
            return None
        if len(years) != 2:
            raise ValueError(f"inference_years_slice must have 2 items, got {len(years)}.")
        start, stop = (None if y is None else int(y) for y in years)
        return slice(start, stop)

    @final
    def select_requested_times(
        self,
        requested_slice: slice,
    ) -> xr.DataArray:
        """
        Select available initialization times within a requested year range.

        Available times are sorted before selection. Requested start and stop years
        are validated against the temporal coverage of the dataset.

        Parameters
        ----------
        requested_slice : slice
            Inclusive year range used to select initialization times. Slice bounds may
            be integers, integer strings, or ``None``.

        Returns
        -------
        xr.DataArray
            Available initialization times falling within the requested year range.

        Raises
        ------
        ValueError
            If the requested range extends beyond available years, contains no
            available times, or contains an invalid year string.
        TypeError
            If a year bound is neither an integer, string, nor ``None``.
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

        requested_start = self._parse_year_bound(requested_slice.start)
        requested_stop = self._parse_year_bound(requested_slice.stop)

        if requested_start is not None and requested_start < available_start:
            raise ValueError(
                f"Requested start time {requested_start} is before "
                f"the first available time {available_start}."
            )

        if requested_stop is not None and requested_stop > available_stop:
            raise ValueError(
                f"Requested stop time {requested_stop} is after "
                f"the final available time {available_stop}."
            )

        selected = times.sel(
            {self.init_time_dim: slice(f"{requested_start}", f"{requested_stop}")}
        )

        if selected.size == 0:
            raise ValueError(
                "No available times fall inside the requested range "
                f"[{requested_start}, {requested_stop}]."
            )

        return selected

    def _parse_year_bound(self, bound):
        """
        Parse and validate a year bound.

        Parameters
        ----------
        bound : int | str | None
            Year bound to parse.

        Returns
        -------
        int | None
            Parsed integer year, or ``None`` when no bound is provided.

        Raises
        ------
        ValueError
            If a string cannot be converted to an integer year.
        TypeError
            If ``bound`` is neither an integer, string, nor ``None``.
        """
        if bound is None:
            return None
        if isinstance(bound, int):
            return bound
        if isinstance(bound, str):
            try:
                return int(bound)
            except ValueError:
                raise ValueError(
                    f"Invalid year bound: {bound}. Must be integer or string of integer."
                )
        raise TypeError(f"Year bound must be int or str, got {type(bound)}")


@dataclasses.dataclass
class Dataloader:
    """
    Wrapper around a PyTorch data loader with distributed sampling support.

    Constructs a PyTorch ``DataLoader`` from the dataset and loader configuration,
    using a distributed sampler when multiple processes participate.

    Parameters
    ----------
    config : DataloaderConfigABC
        Configuration defining batching, workers, memory, and sampling behavior.
    dataset : Dataset
        Dataset providing individual samples.
    collate_fn : Callable
        Function used to combine individual samples into batches.
    rank : int
        Rank of the current distributed process.
    world_size : int
        Number of distributed processes participating in data loading.
    shuffle : bool | None
        Whether samples are shuffled. If ``None``, an appropriate default is
        selected from the distributed configuration.
    return_spatial_mask : bool
        Whether the collation function constructs spatial validity masks.
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
        Construct the underlying PyTorch data loader.

        A distributed sampler is created when ``world_size`` is greater than one.
        Worker prefetching, persistent workers, pinned memory, spatial-mask handling,
        and batch collation are configured from the data-loader settings.
        """
        self.sampler = self._get_dataloader_sampler()
        if self.sampler is not None:
            shuffle = False
        else:
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
                share_spatial_mask=self.config.share_spatial_mask,
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
        Return the shape of an input sample.

        Returns
        -------
        tuple
            Model input shape excluding the batch dimension.
        """
        return self.dataset.get_input_shape()

    @property
    def target_shape(self):
        """
        Return the shape of a target sample.

        Returns
        -------
        tuple
            Model target shape excluding the batch dimension.
        """
        return self.dataset.get_target_shape()

    @property
    def added_features_dim(self):
        """
        Return the number of additional features associated with each sample.

        Returns
        -------
        int
            Number of additional features.
        """
        return self.dataset.get_added_features_dim()

    @final
    def __iter__(self) -> Iterator[BatchDataABC]:
        """
        Return an iterator over data-loader batches.

        Returns
        -------
        Iterator[BatchDataABC]
            Iterator yielding collated batches.
        """
        return iter(self._torch_loader)

    @final
    def _get_dataloader_sampler(self, **kwargs) -> torch.utils.data.Sampler | None:
        """
        Build the distributed sampler when required.

        Parameters
        ----------
        kwargs : Any
            Additional keyword arguments passed to ``DistributedSampler``.

        Returns
        -------
        torch.utils.data.Sampler | None
            Distributed sampler when ``world_size`` is greater than one, otherwise
            ``None``.

        Raises
        ------
        ValueError
            If ``rank`` is outside the valid range for the configured world size.
        """

        if self.world_size > 1:
            if not 0 <= self.rank < self.world_size:
                raise ValueError(
                    f"'rank' must satisfy 0 <= rank < world_size, "
                    f"got rank={self.rank}, world_size={self.world_size}."
                )

            shuffle = True if self.shuffle is None else self.shuffle
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
        Return the number of batches produced by the data loader.

        Returns
        -------
        int
            Number of batches in the underlying PyTorch data loader.
        """
        return len(self._torch_loader)

    @final
    def set_epoch(self, epoch):
        """
        Set the current epoch for distributed sampling.

        Updating the distributed sampler epoch allows deterministic reshuffling to
        change between training epochs.

        Parameters
        ----------
        epoch : int
            Epoch index assigned to the distributed sampler.

        Returns
        -------
        Dataloader
            This data-loader instance.
        """
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)
        return self

    @final
    def subset_loader(self, start_batch: int = 0) -> Iterator[BatchDataABC]:
        """
        Return an iterator beginning at a specified batch.

        Parameters
        ----------
        start_batch : int
            Zero-based index of the first batch to yield.

        Returns
        -------
        Iterator[BatchDataABC]
            Iterator over batches beginning at ``start_batch``.

        Raises
        ------
        TypeError
            If ``start_batch`` is not an integer.
        ValueError
            If ``start_batch`` is negative.
        """
        if not isinstance(start_batch, int):
            raise TypeError(f"start_batch must be int, got {type(start_batch)}.")
        if start_batch < 0:
            raise ValueError(f"start_batch must be >= 0, got {start_batch}.")

        return islice(iter(self), start_batch, None)
