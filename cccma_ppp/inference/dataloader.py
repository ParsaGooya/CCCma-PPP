import dataclasses
import torch
from pathlib import Path
import copy

from cccma_ppp.train.dataloader import TrainDataloaderConfig
from cccma_ppp.inference.dataset import InferenceDatasetConfig, _from_train
from cccma_ppp.data_modules.dataloader import (
    Dataloader,
    DataloaderConfigABC,
    BatchDataABC,
)
from cccma_ppp.data_modules.dataset.dataset_abc import AddedTimeFeatures
from cccma_ppp.generic.distributed import Distributed
from cccma_ppp.generic.runtime import RuntimeContext


@dataclasses.dataclass
class BatchData(BatchDataABC):
    """
    Container for a batch of inference data.

    Input NaN values are replaced with zeros after an optional validity mask is
    constructed from their locations.

    Parameters
    ----------
    input : torch.Tensor
        Batched model input tensor.
    added_features : torch.Tensor | None
        Optional additional features associated with the input batch.
    metadata : list[dict] | None
        Per-sample metadata associated with the batch.
    return_spatial_mask : bool
        Whether to construct a validity mask from non-NaN input values.
    reduce_spatial_mask : bool
        Whether to reduce the validity mask across the batch dimension and reuse
        it as a shared mask.
    input_mask : torch.Tensor | None
        Validity mask derived from the input tensor when spatial masks are
        requested.
    """

    input: torch.Tensor
    added_features: torch.Tensor = None
    metadata: list[dict] | None = None
    return_spatial_mask: bool = False
    reduce_spatial_mask: bool = False
    input_mask: torch.Tensor | None = dataclasses.field(
        init=False,
        default=None,
    )

    def __post_init__(self):
        """
        Initialize spatial masking and replace missing input values.

        When spatial masking is enabled, a validity mask is constructed from the
        non-NaN input values. The mask can optionally be reduced across the batch and
        shared between instances. Input NaN values are then replaced with zeros.
        """

        if self.return_spatial_mask:
            if self.reduce_spatial_mask:
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
    dataset_config : InferenceDatasetConfig | None
        Configuration used to construct the inference dataset. If ``None``, the
        configuration must be derived from a training data-loader configuration.
    batch_size : int
        Number of samples in each inference batch.
    inference_years_slice : tuple | list | None
        Optional start and stop years used to restrict the initialization times
        included in inference.
    num_data_workers : int
        Number of worker processes used for data loading.
    prefetch_factor : int | None
        Number of batches prefetched by each data-loading worker.
    drop_last : bool
        Whether to discard the final incomplete batch.
    load : bool
        Whether dataset contents are loaded into memory when opened.
    reduce_spatial_mask : bool
        Whether spatial validity masks are reduced across the batch dimension.
    time_features : AddedTimeFeatures | None
        Temporal feature configuration inherited from the training data loader.
    """

    dataset_config: InferenceDatasetConfig | None = None
    batch_size: int = 1
    inference_years_slice: tuple | list = None
    num_data_workers: int = 0
    prefetch_factor: int | None = None
    drop_last: bool = False
    load: bool = False
    reduce_spatial_mask: bool = False

    time_features: AddedTimeFeatures | None = dataclasses.field(
        init=False,
        default=None,
    )

    def __post_init__(self):
        """
        Initialize the inference data-loader configuration.

        Initializes common data-loader settings, validates inference times when a
        dataset configuration is already available, and converts configured
        inference-year bounds to a slice.
        """
        super().__init__()
        self.train_dataset_config = None

        if self.dataset_config is not None:
            _ = self._inference_times

        self.inference_years_slice = (
            slice(*[str(item) for item in self.inference_years_slice])
            if self.inference_years_slice is not None
            else None
        )

    def _check_config(self):
        """
        Verify that the required inference configuration has been resolved.

        Raises
        ------
        RuntimeError
            If the inference dataset configuration or temporal feature configuration
            is unavailable.
        """
        if self.dataset_config is None:
            raise RuntimeError(
                "dataset_config must be provided or read from train configs "
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

        Temporal feature settings are copied from the training configuration when
        not already available, and an inference dataset configuration is derived
        from the training dataset configuration when necessary.

        Parameters
        ----------
        train_dataloader_config : TrainDataloaderConfig
            Training data-loader configuration from which dataset and temporal
            feature settings are inherited.
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
            Available initialization times, optionally restricted to the configured
            inference-year range.
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
        Return initialization times available from the inference dataset.

        Returns
        -------
        xr.DataArray
            Initialization times available for inference.
        """
        self._check_config()
        return self.dataset_config.available_times

    def _input_preprocessor_exists(self, load_dir: Path | str = None):
        """
        Check whether the required input preprocessing pipelines have been saved.

        Model and conditioning preprocessing pipelines are checked when their
        corresponding data configurations are present.

        Parameters
        ----------
        load_dir : Path | str | None
            Directory containing saved preprocessing pipelines. If ``None``, the
            preprocessing directory of the current experiment is used.

        Returns
        -------
        bool
            ``True`` if all required preprocessing pipeline files exist, otherwise
            ``False``.
        """
        self._check_config()
        preprocessor_to_check = []
        exists = []

        if load_dir is None:
            load_dir = Path(RuntimeContext.GLOBAL_EXP_DIR) / "preprocessing_pipeline"
        else:
            load_dir = Path(load_dir)

        if self.dataset_config.model is not None:
            preprocessor_name = self.dataset_config.model.preprocessing_pipeline.name
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

        If required preprocessing pipelines are missing, the root process fits and
        saves them using the training data-loader configuration. Processes are then
        synchronized before the fitted pipelines are loaded for inference.

        Parameters
        ----------
        train_loader_config : TrainDataloaderConfig
            Training data-loader configuration used to fit missing preprocessing
            pipelines.
        distributed : Distributed
            Distributed runtime containing device, rank, and world-size information.
        load_path : Path | str | None
            Directory from which fitted preprocessing pipelines are loaded and, when
            necessary, saved.
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
        Build the configured inference data loader.

        The inference dataset is constructed for the selected initialization times
        with temporal features and sample metadata enabled.

        Parameters
        ----------
        return_spatial_mask : bool
            Whether inference batches include validity masks derived from missing
            spatial input values.

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
        Return target-variable metadata from the training dataset configuration.

        Returns
        -------
        dict
            Target-variable metadata including variable names, preprocessing
            operations, and neural-network dimensions.

        Raises
        ------
        RuntimeError
            If a training dataset configuration has not been supplied through
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
    reduce_spatial_mask: bool = False,
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
        Whether to construct an input validity mask from missing values.
    reduce_spatial_mask : bool
        Whether to reduce the validity mask across the batch dimension.

    Returns
    -------
    BatchData
        Collated inference batch containing inputs, optional additional features,
        metadata, and optional spatial validity information.
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
        reduce_spatial_mask=reduce_spatial_mask,
    )
