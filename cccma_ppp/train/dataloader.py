import numpy as np
import dataclasses
import warnings
import torch
from pathlib import Path


from cccma_ppp.train.dataset import TrainDatasetConfig
from cccma_ppp.data_modules import _create_train_mask, WeightsConfig
from cccma_ppp.data_modules.dataloader import (
    Dataloader,
    DataloaderConfigABC,
    BatchDataABC,
)
from cccma_ppp.generic import Distributed



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
    reduce_spatial_mask: bool = False

    def __post_init__(self):
        """
        Prepare batch data.

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
    """
    Configuration for training and validation data loaders.

    Parameters
    ----------
    dataset_config : DatasetConfig
        Dataset configuration.
    batch_size : int
        Number of samples per batch.
    train_years : tuple or list or None, optional
        Range of training years.
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
    train_years: tuple | list = None
    num_validation_years: int = 0
    num_data_workers: int = 0
    prefetch_factor: int = 2
    drop_last: bool = False

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
        self._setup = False

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

    @property
    def available_train_years(self):
        """
        Training years excluding validation period.

        Returns
        -------
        np.ndarray
            Array of years used for training.
        """

        if self.num_validation_years > 0:
            return self.dataset_config.available_train_time[
                : -self.num_validation_years
            ]
        return self.dataset_config.available_train_time

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
        self.rank = distributed.rank
        self.world_size = distributed.world_size

        if distributed.is_root():
            if load_path is None:
                self.dataset_config._fit_preprocessors(
                    self.train_years, save=True
                )

        distributed.barrier()

        if (distributed.distributed or 
            self.load_path is not None):
            self.dataset_config._load_fitted_preprocessors(load_dir=load_path)

        self._setup = True

    def build_train_loader(
        self,
        return_spatial_mask: bool = False,
        reduce_spatial_mask: bool = False,
    ):
        """
        Construct training dataloader.

        Parameters
        ----------
        return_spatial_mask : bool, optional
            Whether to return the spatial mask along with the dataloader output.
        reduce_spatial_mask : bool, optional
            Whether to reduce the spatial mask before returning it.

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
        self,
        return_spatial_mask: bool = False,
        reduce_spatial_mask: bool = False,
        supress_error: bool = True,
    ):
        """
        Construct validation dataloader.

        Parameters
        ----------
        return_spatial_mask : bool, optional
            Whether to return the spatial mask along with the dataloader output.
        reduce_spatial_mask : bool, optional
            Whether to reduce the spatial mask before returning it.

        Returns
        -------
        Dataloader or None
            Validation dataloader, or None if no validation years are specified.

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
            msg = f"Validation dataoader could not be built for num_validation_years = {self.num_validation_years} "

            if  supress_error:
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
        return self.dataset_config.ds_operator.get_weights(config)

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
