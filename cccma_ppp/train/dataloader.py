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
    Document this class.

    Parameters
    ----------
    input : torch.Tensor
        Description not yet provided.
    target : torch.Tensor
        Description not yet provided.
    added_features : torch.Tensor
        Description not yet provided.
    metadata : list[dict] | None
        Description not yet provided.
    return_spatial_mask : bool
        Description not yet provided.
    reduce_spatial_mask : bool
        Description not yet provided.
    input_mask : torch.Tensor | None
        Description not yet provided.
    target_mask : torch.Tensor | None
        Description not yet provided.
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
        Document this function.
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
        Document this function.

        Parameters
        ----------
        device : torch.device | str
            Description not yet provided.

        Returns
        -------
        Any
            Description not yet provided.
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
    Document this class.

    Parameters
    ----------
    dataset_config : TrainDatasetConfig
        Description not yet provided.
    batch_size : int
        Description not yet provided.
    time_features : list | None
        Description not yet provided.
    train_years_slice : tuple | list
        Description not yet provided.
    num_validation_years : int
        Description not yet provided.
    num_data_workers : int
        Description not yet provided.
    prefetch_factor : int
        Description not yet provided.
    drop_last : bool
        Description not yet provided.
    load : bool
        Description not yet provided.
    reduce_spatial_mask : bool
        Description not yet provided.
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
        Document this function.
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
                [
                    t.dt.year > last_train_year - self.num_validation_years
                    for t in self.train_times
                ]
            )

            self.validation_times = self.train_times[validation_mask]
            self.train_times = self.train_times[~validation_mask]

    @property
    def available_times(self):
        """
        Document this function.

        Returns
        -------
        Any
            Description not yet provided.
        """
        return self.dataset_config.available_times

    def setup_distributed(
        self, distributed: Distributed, load_path: Path | str | None = None
    ):
        """
        Document this function.

        Parameters
        ----------
        distributed : Distributed
            Description not yet provided.
        load_path : Path | str | None
            Description not yet provided.
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
        Document this function.

        Parameters
        ----------
        return_metadata : bool
            Description not yet provided.
        return_spatial_mask : bool
            Description not yet provided.
        shuffle : bool | None
            Description not yet provided.

        Returns
        -------
        Any
            Description not yet provided.

        Raises
        ------
        RuntimeError
            Description not yet provided.
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
            return_spatial_mask=return_spatial_mask,
        )

    def build_validation_loader(
        self,
        return_metadata: bool = False,
        return_spatial_mask: bool = False,
        shuffle: bool | None = None,
        supress_error: bool = True,
    ):
        """
        Document this function.

        Parameters
        ----------
        return_metadata : bool
            Description not yet provided.
        return_spatial_mask : bool
            Description not yet provided.
        shuffle : bool | None
            Description not yet provided.
        supress_error : bool
            Description not yet provided.

        Returns
        -------
        Any
            Description not yet provided.

        Raises
        ------
        RuntimeError
            Description not yet provided.

        Warns
        -----
        UserWarning
            Description not yet provided.
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
                return_spatial_mask=return_spatial_mask,
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
        Document this function.

        Parameters
        ----------
        config : WeightsConfig | None
            Description not yet provided.

        Returns
        -------
        Any
            Description not yet provided.
        """
        save = self.distributed.is_root()
        weights = self.dataset_config.ds_operator.get_weights(config, save=save)
        self.distributed.barrier()
        return weights

    @property
    def input_var_metadata(self):
        """
        Document this function.

        Returns
        -------
        Any
            Description not yet provided.
        """
        return self.dataset_config.ds_operator.get_input_var_metadata()

    @property
    def target_var_metadata(self):
        """
        Document this function.

        Returns
        -------
        Any
            Description not yet provided.
        """
        return self.dataset_config.ds_operator.get_target_var_metadata()


def collate_batch(
    batch,
    return_spatial_mask: bool = False,
    reduce_spatial_mask: bool = False,
):
    """
    Document this function.

    Parameters
    ----------
    batch : Any
        Description not yet provided.
    return_spatial_mask : bool
        Description not yet provided.
    reduce_spatial_mask : bool
        Description not yet provided.

    Returns
    -------
    Any
        Description not yet provided.
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
