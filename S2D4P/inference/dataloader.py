import dataclasses
import torch
from pathlib import Path
import copy

from S2D4P.train.dataloader import TrainDataloaderConfig
from S2D4P.inference.dataset import InferenceDatasetConfig, _from_train
from S2D4P.data.dataloader import (
    Dataloader,
    DataloaderConfigABC,
    BatchDataABC,
)
from S2D4P.data.dataset.dataset_abc import AddedTimeFeatures
from S2D4P.generic.distributed import Distributed
from S2D4P.generic.runtime import RuntimeContext


@dataclasses.dataclass
class BatchData(BatchDataABC):
    """
    Container for a batch of inference data.

    Input NaN values are replaced with zeros after optional spatial validity masks
    are constructed.

    Parameters
    ----------
    input : torch.Tensor
        Batched model input tensor.
    added_features : torch.Tensor | None
        Optional additional features associated with the batch.
    metadata : list[dict] | None
        Per-sample metadata associated with the batch.
    return_spatial_mask : bool
        Whether to construct a validity mask from non-NaN input values.
    share_spatial_mask : bool
        Whether to reduce the validity mask across the batch dimension and reuse
        it as a shared mask.
    input_mask : torch.Tensor | None
        Validity mask derived from the input tensor when spatial masking is
        requested.
    """

    input: torch.Tensor
    added_features: torch.Tensor = None
    metadata: list[dict] | None = None
    return_spatial_mask: bool = False
    share_spatial_mask: bool = False
    input_mask: torch.Tensor | None = dataclasses.field(
        init=False,
        default=None,
    )

    def __post_init__(self):
        """
        Initialize the spatial validity mask and replace missing input values.

        When spatial masking is enabled, a mask is constructed from non-NaN input
        values. The mask can optionally be reduced across the batch dimension and
        shared between batch instances. Input NaN values are then replaced with
        zeros.
        """
        if self.return_spatial_mask:
            if self.share_spatial_mask:
                if type(self)._shared_input_mask is None:
                    type(self)._shared_input_mask = (~torch.isnan(self.input)).all(
                        dim=0
                    )

                self.input_mask = type(self)._shared_input_mask

            else:
                self.input_mask = ~torch.isnan(self.input)

        self.input.nan_to_num_(nan=0.0)

    def to_device(self, device: torch.device | str) -> "BatchData":
        """
        Move batch tensors to a device.

        Parameters
        ----------
        device : torch.device | str
            Device to which the input, validity mask, and additional features are
            moved.

        Returns
        -------
        BatchData
            This batch instance with its applicable tensors moved to the requested
            device.
        """
        self.input = self.input.to(device)

        if self.input_mask is not None:
            self.input_mask = self.input_mask.to(device)

        if self.added_features is not None:
            self.added_features = self.added_features.to(device)

        return self


@dataclasses.dataclass
class InferenceDataloaderConfig(DataloaderConfigABC):
    """
    Configuration for an inference data loader.

    Parameters
    ----------
    dataset : InferenceDatasetConfig | None
        Configuration used to construct the inference dataset. If ``None``, it
        can be derived from a training data-loader configuration.
    batch_size : int
        Number of inference samples in each batch.
    inference_years_slice : tuple | list | None
        Optional start and stop years restricting the inference period.
    num_data_workers : int
        Number of worker processes used for data loading.
    prefetch_factor : int | None
        Number of batches prefetched by each data-loading worker.
    drop_last : bool
        Whether to discard the final incomplete batch.
    load : bool
        Whether underlying dataset contents are loaded into memory when opened.
    share_spatial_mask : bool
        Whether spatial validity masks are reduced across the batch dimension and
        shared between batches.
    time_features : AddedTimeFeatures | None
        Temporal feature configuration inherited from the training data loader.
    """

    dataset: InferenceDatasetConfig | None = None
    batch_size: int = 1
    inference_years_slice: tuple | list = None
    num_data_workers: int = 0
    prefetch_factor: int | None = None
    drop_last: bool = False
    load: bool = False
    share_spatial_mask: bool = False

    time_features: AddedTimeFeatures | None = dataclasses.field(
        init=False,
        default=None,
    )

    def __post_init__(self):
        """
        Initialize the inference data-loader configuration.

        Initializes common data-loader state, validates an explicitly supplied dataset
        configuration, and converts the optional inference year range to a slice.
        Training-derived dataset and temporal feature configurations can be attached
        later with ``read_configs_from_train``.
        """
        super().__init__()
        self.train_dataset_config = None
        self.dataset_config = self.dataset

        if self.dataset_config is not None:
            _ = self._inference_times

        self.inference_years_slice = (
            slice(*[str(item) for item in self.inference_years_slice])
            if self.inference_years_slice is not None
            else None
        )

    def _check_config(self):
        """
        Validate required inference configuration state.

        Raises
        ------
        RuntimeError
            If the inference dataset configuration or training-derived temporal
            feature configuration is unavailable.
        """
        if self.dataset_config is None:
            raise RuntimeError(
                "dataset config must be provided or read from train configs "
                "via read_configs_from_train method."
            )

        if self.time_features is None:
            raise RuntimeError(
                "time_features must be read from train configs "
                "via read_configs_from_train method."
            )

    def read_configs_from_train(self, train_dataloader_config: TrainDataloaderConfig):
        """
        Populate inference settings from a training data-loader configuration.

        Temporal features are copied from training and an inference dataset
        configuration is derived from the training dataset when these settings have
        not already been supplied.

        Parameters
        ----------
        train_dataloader_config : TrainDataloaderConfig
            Training data-loader configuration from which dataset and temporal feature
            settings are inherited.
        """
        if self.time_features is None:
            self.time_features = copy.deepcopy(train_dataloader_config.time_features)

        if self.dataset_config is None:
            self.dataset_config = _from_train(train_dataloader_config.dataset_config)

        _ = self._inference_times
        self.train_dataset_config = train_dataloader_config.dataset_config

    @property
    def _inference_times(self):
        """
        Return initialization times selected for inference.

        Returns
        -------
        xr.DataArray
            All available inference times or those falling within the configured
            inference year range.
        """
        if self.inference_years_slice is None:
            return self.available_times
        else:
            return self.select_requested_times(
                requested_slice=self.inference_years_slice,
            )

    @property
    def available_times(self):
        """
        Return initialization times available for inference.

        Returns
        -------
        Any
            Initialization times available from the inference dataset configuration.
        """
        self._check_config()
        return self.dataset_config.available_times

    def _input_preprocessor_exists(self, load_dir: Path | str = None):
        """
        Check whether fitted input preprocessing pipelines exist on disk.

        Forecast and explicitly configured conditioning preprocessing pipelines are
        checked in the requested directory.

        Parameters
        ----------
        load_dir : Path | str | None
            Directory containing fitted preprocessing pipelines. If ``None``, the
            experiment preprocessing directory is used.

        Returns
        -------
        bool
            ``True`` when all required preprocessing pipeline files exist.
        """
        self._check_config()
        preprocessor_to_check = []
        exists = []

        if load_dir is None:
            load_dir = Path(RuntimeContext.GLOBAL_EXP_DIR) / "preprocessing_pipeline"
        else:
            load_dir = Path(load_dir)

        if self.dataset_config.forecast is not None:
            preprocessor_name = self.dataset_config.forecast.preprocessing_pipeline.name
            preprocessor_to_check.append(
                load_dir / f"{preprocessor_name}_preprocessing_pipeline.joblib"
            )

        if self.dataset_config.condition is not None:
            preprocessor_name = (
                self.dataset_config.condition.preprocessing_pipeline.name
            )
            preprocessor_to_check.append(
                load_dir / f"{preprocessor_name}_preprocessing_pipeline.joblib"
            )

        for path in preprocessor_to_check:
            exists.append(path.exists())

        return all(exists)

    def setup_distributed(
        self,
        train_loader_config: TrainDataloaderConfig,
        distributed: Distributed,
        load_path: Path | str | None = None,
    ):
        """
        Configure inference data loading for distributed execution.

        Missing preprocessing pipelines are fitted and saved by the root process
        using the training configuration. Processes are synchronized before the
        fitted pipelines are loaded for inference.

        Parameters
        ----------
        train_loader_config : TrainDataloaderConfig
            Training data-loader configuration used to fit missing preprocessing
            pipelines.
        distributed : Distributed
            Distributed runtime containing process rank and world-size information.
        load_path : Path | str | None
            Directory containing fitted preprocessing pipelines. If ``None``, the
            default experiment preprocessing directory is used.
        """
        self._check_config()
        self.distributed = distributed
        self.rank = distributed.rank
        self.world_size = distributed.world_size

        if not self._input_preprocessor_exists(load_path):
            if distributed.is_root():
                train_loader_config.dataset_config.fit_preprocessors(
                    train_loader_config.train_times,
                    save=True,
                    save_path=load_path,
                )

        distributed.barrier()

        self.dataset_config.load_fitted_preprocessors(load_dir=load_path)

        if distributed.distributed:
            self.pin_memory = True

        self._setup = True

    def build_inference_loader(self, return_spatial_mask: bool = False):
        """
        Build the inference data loader.

        The configured inference times and training-derived temporal features are
        used to construct the inference dataset. Sampling metadata are always
        returned so predictions can later be reconstructed into NetCDF coordinates.

        Parameters
        ----------
        return_spatial_mask : bool
            Whether inference batches include validity masks derived from missing
            spatial values.

        Returns
        -------
        Dataloader
            Configured inference data loader.

        Raises
        ------
        RuntimeError
            If distributed data-loader setup has not been completed.
        """
        if not self._setup:
            raise RuntimeError(
                "Dataloader has to be setup for distributed training first by calling .setup_distributed()"
            )

        inference_dataset = self.dataset_config.build_dataset(
            times=self._inference_times,
            time_features=self.time_features,
            return_metadata=True,
            load=self.load,
        )

        return Dataloader(
            dataset=inference_dataset,
            config=self,
            collate_fn=collate_batch,
            rank=self.rank,
            world_size=self.world_size,
            shuffle=False,
            return_spatial_mask=return_spatial_mask,
        )

    @property
    def input_var_metadata(self):
        """
        Return metadata describing inference input variables.

        Returns
        -------
        dict
            Input-variable metadata including variable names, preprocessing
            operations, and neural-network dimensions.
        """
        self._check_config()
        return self.dataset_config.ds_operator.get_input_var_metadata()

    @property
    def target_var_metadata(self):
        """
        Return metadata describing the model target variables used during training.

        Target metadata are obtained from the retained training dataset configuration
        because an inference dataset may not contain target data.

        Returns
        -------
        dict
            Target-variable metadata including variable names, preprocessing
            operations, and neural-network dimensions.

        Raises
        ------
        RuntimeError
            If the training dataset configuration has not been supplied through
            ``read_configs_from_train``.
        """
        if self.train_dataset_config is None:
            raise RuntimeError(
                "output variables metadata cannot be read unless train dataloader "
                "is available. Hint: run read_configs_from_train(TrainDatasetConfig, ...)."
            )
        return self.train_dataset_config.ds_operator.get_target_var_metadata()


def collate_batch(
    batch,
    return_spatial_mask: bool = False,
    share_spatial_mask: bool = False,
):
    """
    Collate inference samples into a batch.

    Input tensors and optional additional features are stacked along the batch
    dimension. Per-sample metadata are retained when provided by the dataset.

    Parameters
    ----------
    batch : Any
        Sequence of inference samples, optionally paired with per-sample metadata.
    return_spatial_mask : bool
        Whether to construct input validity masks from missing values.
    share_spatial_mask : bool
        Whether validity masks are reduced across the batch dimension and shared.

    Returns
    -------
    BatchData
        Collated inference batch containing inputs, optional additional features,
        metadata, and an optional spatial validity mask.
    """
    metadata = None

    if isinstance(batch[0], tuple):
        batch, metadata = zip(*batch)

    inputs = torch.stack([b["input"] for b in batch])

    added_features = None
    if batch[0]["added_features"] is not None:
        added_features = torch.stack([b["added_features"] for b in batch])

    return BatchData(
        input=inputs,
        added_features=added_features,
        metadata=list(metadata) if metadata is not None else None,
        return_spatial_mask=return_spatial_mask,
        share_spatial_mask=share_spatial_mask,
    )
