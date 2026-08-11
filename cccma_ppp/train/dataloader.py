import numpy as np
import dataclasses
import warnings
import torch
from pathlib import Path


from cccma_ppp.train.dataset import TrainDatasetConfig
from cccma_ppp.data_modules.utils import _create_train_mask
from cccma_ppp.data_modules.weights import WeightsConfig
from cccma_ppp.data_modules.dataset.dataset_abc import AddedTimeFeatures
from cccma_ppp.data_modules.dataloader import (
    Dataloader,
    DataloaderConfigABC,
    BatchDataABC,
)
from cccma_ppp.generic.distributed import Distributed


@dataclasses.dataclass
class BatchData(BatchDataABC):
    """
    Container for batched training data.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor.
    target : torch.Tensor
        Target tensor.
    added_features : torch.Tensor or None, optional
        Additional input features.
    return_spatial_mask : bool, optional
        Whether to compute and include spatial masks.
    reduce_spatial_mask : bool, optional
        Whether to reduce masks across batch dimension.
    """

    input: torch.Tensor
    target: torch.Tensor
    added_features: torch.Tensor = None
    metadata: list[dict] | None = None
    return_spatial_mask: bool = False
    reduce_spatial_mask: bool = True
    input_mask: torch.Tensor | None = dataclasses.field(
        init=False,
        default=None,
    )
    target_mask: torch.Tensor | None = dataclasses.field(
        init=False,
        default=None,
    )

    def __post_init__(self):
        """
        Prepare batch data.

        Returns
        -------
        None
        """

        if self.return_spatial_mask:
            if self.reduce_spatial_mask:
                if type(self)._shared_input_mask is None:
                    type(self)._shared_input_mask = (~torch.isnan(self.input)).all(
                        dim=0
                    )

                if type(self)._shared_target_mask is None:
                    type(self)._shared_target_mask = (~torch.isnan(self.target)).all(
                        dim=0
                    )

                self.input_mask = type(self)._shared_input_mask
                self.target_mask = type(self)._shared_target_mask

            else:
                self.input_mask = ~torch.isnan(self.input)
                self.target_mask = ~torch.isnan(self.target)

        self.input.nan_to_num_(nan=0.0)
        self.target.nan_to_num_(nan=0.0)

    def to_device(self, device: torch.device | str):
        """
        Move batch data to specified device.

        Parameters
        ----------
        device : torch.device or str

        Returns
        -------
        BatchData
            Updated instance on target device.
        """
        self.input = self.input.to(device)
        self.target = self.target.to(device)

        if self.input_mask is not None:
            self.input_mask = self.input_mask.to(device)

        if self.target_mask is not None:
            self.target_mask = self.target_mask.to(device)

        if self.added_features is not None:
            self.added_features = self.added_features.to(device)

        return self


@dataclasses.dataclass
class TrainDataloaderConfig(DataloaderConfigABC):
    """
    Configuration for training and validation data loaders.

    Parameters
    ----------
    dataset_config : DatasetConfig
        Dataset configuration.
    batch_size : int
        Number of samples per batch.
    train_years_slice : tuple or list or None, optional
        Slice arguments for training years.
    num_validation_years : int, optional
        Number of years reserved for validation.
    num_data_workers : int, optional
        Number of parallel data workers.
    prefetch_factor : int, optional
        Data prefetching factor.
    drop_last : bool, optional
        Whether to drop incomplete batches.
    """

    dataset_config: TrainDatasetConfig
    batch_size: int
    time_features: list | None = None
    train_years_slice: tuple | list = None
    num_validation_years: int = 0
    num_data_workers: int = 0
    prefetch_factor: int = 2
    drop_last: bool = False
    load: bool = False
    reduce_spatial_mask: bool = False

    def __post_init__(self):
        """
        Initialize dataloader configuration.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If requested training years are invalid.
        """
        super().__init__()

        self.time_features = AddedTimeFeatures(self.dataset_config, self.time_features)

        if self.train_years_slice is None:
            self.train_times = self.available_times
        else:

            self.train_years_slice = slice(
                *[str(item) for item in self.train_years_slice]
            )

            self.train_times = self.select_requested_times(
                requested_slice=self.train_years_slice,
            )

        if self.num_validation_years > 0:

            last_train_year = self.train_times[-1].dt.year

            validation_mask = np.asarray(
                [t.dt.year > last_train_year - self.num_validation_years
                for t in self.train_times]
            )

            self.validation_times = self.train_times[validation_mask]
            self.train_times = self.train_times[~validation_mask]    

    @property
    def available_times(self):
        """
        Available training times given the dataset configs.

        Returns
        -------
        np.ndarray
            Array of times available for training.
            The items are cftime or datetime objects.
        """

        return self.dataset_config.available_times

    def setup_distributed(
        self, distributed: Distributed, load_path: Path | str | None = None
    ):
        """
        Prepare dataloader for distributed training.

        Parameters
        ----------
        distributed : Distributed
            Distributed training context.
        save_path : pathlib.Path or str or None, optional
            Path to save fitted preprocessors.

        Returns
        -------
        None
        """
        self.distributed = distributed
        self.rank = distributed.rank
        self.world_size = distributed.world_size

        if distributed.is_root():
            if load_path is None:
                self.dataset_config.fit_preprocessors(self.train_times, save=True)

        distributed.barrier()

        if distributed.distributed or load_path is not None:
            self.dataset_config.load_fitted_preprocessors(load_dir=load_path)

        if distributed.distributed:
            self.pin_memory = True

        self._setup = True

    def build_train_loader(
        self,
        return_metadata: bool = False,
        return_spatial_mask: bool = False,
        shuffle: bool | None = None,
    ):
        """
        Construct training dataloader.

        Parameters
        ----------
        return_metadata : bool, optional
            Whether to return the metadata for selection (coords).
        shuffle : bool, optional
            Whether to shuffle the dataset chunk.

        Returns
        -------
        Dataloader
            Training dataloader.

        Raises
        ------
        RuntimeError
            If setup has not been called.
        """
        if not self._setup:
            raise RuntimeError(
                "Dataloader has to be setup for distributed training first by calling .setup_distributed()"
            )

        train_mask = _create_train_mask(
            init_times=self.dataset_config.get_input_times(self.train_times),
            lead_times=self.dataset_config.input_lead_times,
        )

        train_dataset = self.dataset_config.build_dataset(
            times=self.train_times,
            mask=train_mask,
            time_features=self.time_features,
            return_metadata=return_metadata,
            load=self.load,
        )

        return Dataloader(
            dataset=train_dataset,
            config=self,
            collate_fn=collate_batch,
            rank=self.rank,
            shuffle=shuffle,
            world_size=self.world_size,
            return_spatial_mask=return_spatial_mask
        )

    def build_validation_loader(
        self,
        return_metadata: bool = False,
        return_spatial_mask: bool = False,
        shuffle: bool | None = None,
        supress_error: bool = True,
    ):
        """
        Construct validation dataloader.

        Parameters
        ----------
        return_metadata : bool, optional
            Whether to return the metadata for selection (coords).
        shuffle : bool, optional
            Whether to shuffle the dataset chunk.
        supress_error : bool, optional
            Whether to raise error if validation could not be built.

        Returns
        -------
        Dataloader or None
            Validation dataloader, or None if no number of validation years are specified.

        Raises
        ------
        RuntimeError
            If setup has not been called.
        """
        if not self._setup:
            raise RuntimeError(
                "Dataloader has to be setup for distributed training first by calling .setup_distributed()"
            )

        if self.num_validation_years > 0:
            validation_mask = _create_train_mask(
                init_times=self.dataset_config.get_input_times(self.validation_times),
                lead_times=self.dataset_config.input_lead_times,
            )
            validation_dataset = self.dataset_config.build_dataset(
                times=self.validation_times,
                time_features=self.time_features,
                mask=validation_mask,
                return_metadata=return_metadata,
                load=self.load,
            )

            return Dataloader(
                dataset=validation_dataset,
                config=self,
                collate_fn=collate_batch,
                rank=self.rank,
                shuffle=shuffle,
                world_size=self.world_size,
                return_spatial_mask=return_spatial_mask
            )

        else:
            msg = f"Validation dataoader could not be built for num_validation_years = {self.num_validation_years} "

            if supress_error:
                warnings.warn(msg)
                return None
            else:
                raise RuntimeError(msg)

    def get_weights(self, config: WeightsConfig | None = None):
        """
        Retrieve weights for loss computation.

        Parameters
        ----------
        config : WeightsConfig or None, optional
            Configuration used to compute or retrieve weights.

        Returns
        -------
        xr.DataArray
            Weights for loss computation.
        """
        save = self.distributed.is_root()
        weights = self.dataset_config.ds_operator.get_weights(config, save=save)
        self.distributed.barrier()
        return weights

    @property
    def input_var_metadata(self):
        """
        Retrieve input variable metadata.

        Returns
        -------
        dict
        """
        return self.dataset_config.ds_operator.get_input_var_metadata()

    @property
    def target_var_metadata(self):
        """
        Retrieve target variable metadata.

        Returns
        -------
        dict
        """
        return self.dataset_config.ds_operator.get_target_var_metadata()


def collate_batch(
    batch,
    return_spatial_mask: bool = False,
    reduce_spatial_mask: bool = False,
):
    """
    Collate dataset samples into a batch.

    Parameters
    ----------
    batch : list
        List of dataset samples.
    return_spatial_mask : bool, optional
        Whether to include spatial masks.
    reduce_spatial_mask : bool, optional
        Whether to reduce masks across the batch.

    Returns
    -------
    BatchData or tuple
        Batched data, optionally paired with metadata.
    """
    metadata = None

    if isinstance(batch[0], tuple):
        batch, metadata = zip(*batch)

    inputs = torch.stack([b["input"] for b in batch])
    targets = torch.stack([b["target"] for b in batch])

    added_features = None
    if batch[0]["added_features"] is not None:
        added_features = torch.stack([b["added_features"] for b in batch])

    return BatchData(
        input=inputs,
        target=targets,
        added_features=added_features,
        metadata=list(metadata) if metadata is not None else None,
        return_spatial_mask=return_spatial_mask,
        reduce_spatial_mask=reduce_spatial_mask,
    )
