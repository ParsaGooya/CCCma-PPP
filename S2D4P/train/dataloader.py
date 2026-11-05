import numpy as np
import dataclasses
import warnings
import torch
from pathlib import Path


from S2D4P.train.dataset import TrainDatasetConfig
from S2D4P.data.utils import _create_train_mask
from S2D4P.data.weights import WeightsConfig
from S2D4P.data.dataset.dataset_abc import AddedTimeFeatures
from S2D4P.data.dataloader import (
    Dataloader,
    DataloaderConfigABC,
    BatchDataABC,
)
from S2D4P.generic.distributed import Distributed


@dataclasses.dataclass
class BatchData(BatchDataABC):
    """
    Container for a batch of training data.

    Input and target NaN values are replaced with zeros after optional spatial
    validity masks are constructed.

    Parameters
    ----------
    input : torch.Tensor
        Batched model input tensor.
    target : torch.Tensor
        Batched target tensor.
    added_features : torch.Tensor | None
        Optional additional features associated with the batch.
    metadata : list[dict] | None
        Optional per-sample metadata.
    return_spatial_mask : bool
        Whether validity masks are constructed from non-NaN input and target
        values.
    share_spatial_mask : bool
        Whether validity masks are reduced across the batch dimension and shared
        between batch instances.
    input_mask : torch.Tensor | None
        Validity mask derived from the input tensor.
    target_mask : torch.Tensor | None
        Validity mask derived from the target tensor.
    """

    input: torch.Tensor
    target: torch.Tensor
    added_features: torch.Tensor = None
    metadata: list[dict] | None = None
    return_spatial_mask: bool = False
    share_spatial_mask: bool = True
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
        Initialize spatial validity masks and replace missing values.

        When spatial masking is enabled, masks are constructed from non-NaN input and
        target values. The masks can optionally be reduced across the batch dimension
        and shared between batch instances. Input and target NaN values are then
        replaced with zeros.
        """
        if self.return_spatial_mask:
            if self.share_spatial_mask:
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
        Move batch tensors to a device.

        Parameters
        ----------
        device : torch.device | str
            Device to which the input, target, masks, and additional features are
            moved.

        Returns
        -------
        BatchData
            This batch instance with its applicable tensors moved to the requested
            device.
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

    Defines batching, temporal feature generation, train-validation temporal
    splitting, worker settings, spatial-mask behavior, and distributed data
    loading.

    Parameters
    ----------
    dataset : TrainDatasetConfig
        Configuration used to construct training and validation datasets.
    batch_size : int
        Number of samples in each batch.
    time_features : list | None
        Temporal features added to each dataset sample.
    train_years_slice : tuple | list | None
        Optional start and stop years restricting the data used for training and
        validation.
    num_validation_years : int
        Number of final years in the selected period reserved for validation.
    num_data_workers : int
        Number of worker processes used for data loading.
    prefetch_factor : int | None
        Number of batches prefetched by each data-loading worker.
    drop_last : bool
        Whether the final incomplete batch is discarded.
    load : bool
        Whether underlying xarray datasets are loaded into memory when opened.
    share_spatial_mask : bool
        Whether spatial validity masks are reduced across the batch dimension and
        shared between batches.
    """

    dataset: TrainDatasetConfig
    batch_size: int
    time_features: list | None = None
    train_years_slice: tuple | list = None
    num_validation_years: int = 0
    num_data_workers: int = 0
    prefetch_factor: int = 2
    drop_last: bool = False
    load: bool = False
    share_spatial_mask: bool = False

    def __post_init__(self):
        """
        Initialize the training data-loader configuration.

        Builds the temporal feature configuration, resolves the requested training
        period, and optionally reserves the final configured number of years for
        validation.
        """
        super().__init__()
        self.dataset_config = self.dataset
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
        Return initialization times available for training.

        Returns
        -------
        Any
            Initialization times available from the training dataset configuration.
        """
        return self.dataset_config.available_times

    def setup_distributed(
        self, distributed: Distributed, load_path: Path | str | None = None
    ):
        """
        Configure training data loading for distributed execution.

        The root process fits and saves preprocessing pipelines when necessary.
        Processes are synchronized before fitted preprocessing state is loaded, and
        pinned memory is enabled for distributed execution.

        Parameters
        ----------
        distributed : Distributed
            Distributed runtime containing rank and world-size information.
        load_path : Path | str | None
            Directory containing existing fitted preprocessing pipelines. If ``None``,
            preprocessors are fitted from the configured training data.
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
        Build the training data loader.

        A temporal mask prevents samples whose target times extend beyond the common
        data coverage. The resulting dataset is constructed from the selected
        training times and configured temporal features.

        Parameters
        ----------
        return_metadata : bool
            Whether samples include their sampling-coordinate metadata.
        return_spatial_mask : bool
            Whether batches contain spatial validity masks.
        shuffle : bool | None
            Whether training samples are shuffled. If ``None``, the data loader
            selects the appropriate default.

        Returns
        -------
        Dataloader
            Configured training data loader.

        Raises
        ------
        RuntimeError
            If distributed data-loader setup has not been completed.
        """
        if not self._setup:
            raise RuntimeError(
                "Dataloader has to be setup for distributed training first by calling .setup_distributed()"
            )

        train_mask = _create_train_mask(
            init_times=self.dataset_config.get_input_times(self.train_times),
            lead_times=self.dataset_config.input_lead_times,
            max_allowed_time=self.dataset_config.get_common_time.max(),
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
        Build the validation data loader.

        Validation samples are restricted to the configured validation years and
        masked when their target times extend beyond the common data coverage.

        Parameters
        ----------
        return_metadata : bool
            Whether samples include their sampling-coordinate metadata.
        return_spatial_mask : bool
            Whether batches contain spatial validity masks.
        shuffle : bool | None
            Whether validation samples are shuffled.
        supress_error : bool
            Whether absence of configured validation years produces a warning instead
            of raising an exception.

        Returns
        -------
        Dataloader | None
            Configured validation data loader, or ``None`` when validation is disabled
            and ``supress_error`` is ``True``.

        Raises
        ------
        RuntimeError
            If distributed setup has not been completed, or validation is unavailable
            and ``supress_error`` is ``False``.

        Warns
        -----
        UserWarning
            If no validation years are configured and ``supress_error`` is ``True``.
        """
        if not self._setup:
            raise RuntimeError(
                "Dataloader has to be setup for distributed training first by calling .setup_distributed()"
            )

        if self.num_validation_years > 0:
            validation_mask = _create_train_mask(
                init_times=self.dataset_config.get_input_times(self.validation_times),
                lead_times=self.dataset_config.input_lead_times,
                max_allowed_time=self.dataset_config.get_common_time.max(),
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
        Build loss weights for the training target.

        Only the root process saves newly generated weights. Distributed processes are
        synchronized before the weights are returned.

        Parameters
        ----------
        config : WeightsConfig | None
            Configuration controlling spatial and variable weighting. If ``None``,
            default weight settings are used.

        Returns
        -------
        xr.DataArray
            Loss weights aligned with the training target.
        """
        save = self.distributed.is_root()
        weights = self.dataset_config.ds_operator.get_weights(config, save=save)
        self.distributed.barrier()
        return weights

    @property
    def input_var_metadata(self):
        """
        Return metadata describing model input variables.

        Returns
        -------
        dict
            Input-variable names, preprocessing operations, and neural-network
            dimensions.
        """
        return self.dataset_config.ds_operator.get_input_var_metadata()

    @property
    def target_var_metadata(self):
        """
        Return metadata describing model target variables.

        Returns
        -------
        dict
            Target-variable names, preprocessing operations, and neural-network
            dimensions.
        """
        return self.dataset_config.ds_operator.get_target_var_metadata()


def collate_batch(
    batch,
    return_spatial_mask: bool = False,
    share_spatial_mask: bool = False,
):
    """
    Collate training samples into a batch.

    Input and target tensors are stacked along the batch dimension together with
    optional additional features. Per-sample metadata are retained when supplied
    by the dataset.

    Parameters
    ----------
    batch : Any
        Sequence of training samples, optionally paired with per-sample metadata.
    return_spatial_mask : bool
        Whether input and target validity masks are constructed from missing
        values.
    share_spatial_mask : bool
        Whether validity masks are reduced across the batch dimension and shared.

    Returns
    -------
    BatchData
        Collated training batch containing inputs, targets, optional additional
        features, metadata, and optional spatial validity masks.
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
        share_spatial_mask=share_spatial_mask,
    )
