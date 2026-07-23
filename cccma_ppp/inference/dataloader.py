import numpy as np
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

    Parameters
    ----------
    input : torch.Tensor
        Batched model input.
    added_features : torch.Tensor or None, optional
        Additional features associated with the input samples.
    metadata : list of dict or None, optional
        Metadata associated with each sample in the batch.
    return_spatial_mask : bool, optional
        Whether to compute and return a mask of valid spatial input values.
    reduce_spatial_mask : bool, optional
        Whether to reduce the spatial mask across the batch dimension and
        reuse the resulting shared mask.
    input_mask : torch.Tensor or None, optional
        Mask identifying valid input values. This field is initialized
        automatically.

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
        Prepare the batched input and optional spatial mask.

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
        Move the batch tensors to a device.

        Parameters
        ----------
        device : torch.device or str
            Device to which the input, spatial mask, and additional features are
            moved.

        Returns
        -------
        BatchData
            Current batch instance with its tensors moved to the requested device.

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
    dataset_config : InferenceDatasetConfig or None, optional
        Configuration used to construct the inference dataset.
    batch_size : int, optional
        Number of samples in each batch.
    inference_years : tuple or list or None, optional
        Inclusive start and end years used for inference. If ``None``, all
        available years are used.
    num_data_workers : int, optional
        Number of worker processes used to load data.
    prefetch_factor : int or None, optional
        Number of batches loaded in advance by each worker.
    drop_last : bool, optional
        Whether to discard the final incomplete batch.
    load : bool, optional
        Whether to load the underlying xarray data eagerly into memory.
    time_features : AddedTimeFeatures or None, optional
        Temporal-feature configuration copied from the training data-loader
        configuration. This field is initialized automatically.

    """

    dataset_config: InferenceDatasetConfig | None = None
    batch_size: int = 1
    inference_years: tuple | list = None
    num_data_workers: int = 0
    prefetch_factor: int | None = None
    drop_last: bool = False
    load: bool = False

    time_features: AddedTimeFeatures | None = dataclasses.field(
        init=False,
        default=None,
    )

    def __post_init__(self):
        """
        Initialize and validate the inference data-loader configuration.

        """

        super().__init__()
        self.train_dataset_config = None

        if self.dataset_config is not None:
            _ = self._inference_years

    def _check_config(self):
        """
        Validate the resolved inference configuration.

        Raises
        ------
        RuntimeError
            If the inference dataset configuration is unavailable.
        RuntimeError
            If the temporal-feature configuration has not been read from the
            training data-loader configuration.

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

    def read_configs_from_train(
        self,
        train_dataloader_config: TrainDataloaderConfig,
    ):
        """
        Derive inference settings from a training data-loader configuration.

        Parameters
        ----------
        train_dataloader_config : TrainDataloaderConfig
            Training data-loader configuration from which dataset and temporal
            feature settings are read.

        Raises
        ------
        ValueError
            If the requested inference years are not available.

        """
        if self.time_features is None:
            self.time_features = copy.deepcopy(train_dataloader_config.time_features)

        if self.dataset_config is None:
            self.dataset_config = _from_train(train_dataloader_config.dataset_config)

        _ = self._inference_years
        self.train_dataset_config = train_dataloader_config.dataset_config

    @property
    def _inference_years(self):
        """
        Return the years selected for inference.

        Returns
        -------
        numpy.ndarray
            Available years or the requested inclusive range of inference years.

        Raises
        ------
        ValueError
            If one or more requested inference years are unavailable.

        """
        if self.inference_years is None:
            return self.available_times
        else:
            inference_years = np.arange(
                self.inference_years[0], self.inference_years[1] + 1
            )

            if not set(inference_years).issubset(set(self.available_times)):
                raise ValueError(
                    f"the requested inference years are not available:"
                    f"available years: [{self.available_times.min()},{self.available_times.max()}]"
                )

            return inference_years

    @property
    def available_times(self):
        """
        Return the times available for inference.

        Returns
        -------
        numpy.ndarray
            Time-coordinate values available from the inference dataset.

        Raises
        ------
        RuntimeError
            If the inference configuration is incomplete.

        """
        self._check_config()
        return self.dataset_config.available_times

    def _input_preprocessor_exists(
        self,
        load_dir: Path | str = None,
    ):
        """
        Determine whether the required fitted preprocessing pipelines exist.

        Parameters
        ----------
        load_dir : pathlib.Path, str, or None, optional
            Directory containing the fitted preprocessing pipelines. If ``None``,
            the preprocessing directory of the current experiment is used.

        Returns
        -------
        bool
            ``True`` if every required preprocessing pipeline exists, otherwise
            ``False``.

        Raises
        ------
        RuntimeError
            If the inference configuration is incomplete.

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
        Configure the inference data loader for distributed execution.

        Parameters
        ----------
        train_loader_config : TrainDataloaderConfig
            Training data-loader configuration used to fit missing preprocessing
            pipelines.
        distributed : Distributed
            Distributed execution context containing rank and process information.
        load_path : pathlib.Path, str, or None, optional
            Directory from which fitted preprocessing pipelines are loaded and to
            which newly fitted pipelines are saved.

        Raises
        ------
        RuntimeError
            If the inference configuration is incomplete.

        """

        self._check_config()
        self.rank = distributed.rank
        self.world_size = distributed.world_size

        if not self._input_preprocessor_exists(load_path):
            if distributed.is_root():
                train_loader_config.dataset_config.fit_preprocessors(
                    train_loader_config.dataset_config.train_years,
                    save=True,
                    save_path=load_path,
                )

        distributed.barrier()

        self.dataset_config.load_fitted_preprocessors(load_dir=load_path)

        if distributed.distributed:
            self.pin_memory = True

        self._setup = True

    def build_inference_loader(
        self,
        return_spatial_mask: bool = False,
        reduce_spatial_mask: bool = False,
    ):
        """
        Construct the inference data loader.

        Parameters
        ----------
        return_spatial_mask : bool, optional
            Whether each batch should include a mask identifying valid spatial
            input values.
        reduce_spatial_mask : bool, optional
            Whether the spatial mask should be reduced across the batch dimension
            and reused as a shared mask.

        Returns
        -------
        Dataloader
            Configured inference data loader.

        Raises
        ------
        RuntimeError
            If distributed setup has not been completed.

        """
        if not self._setup:
            raise RuntimeError(
                "Dataloader has to be setup for distributed training first by calling .setup_distributed()"
            )

        inference_dataset = self.dataset_config.build_dataset(
            years=self._inference_years,
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
            reduce_spatial_mask=reduce_spatial_mask,
        )

    @property
    def input_var_metadata(self):
        """
        Return metadata for the inference input variables.

        Returns
        -------
        object
            Input-variable metadata provided by the inference dataset operator.

        Raises
        ------
        RuntimeError
            If the inference configuration is incomplete.

        """
        self._check_config()
        return self.dataset_config.ds_operator.get_input_var_metadata()

    @property
    def target_var_metadata(self):
        """
        Return metadata for the training target variables.

        Returns
        -------
        object
            Target-variable metadata provided by the training dataset operator.

        Raises
        ------
        RuntimeError
            If the training dataset configuration is unavailable.

        """
        if self.train_dataset_config is None:
            raise RuntimeError(
                "output variables metadata cannot be read unless train dataloader "
                "is available. Hint: run setup_distributed(TrainDatasetConfig, ...)."
            )
        return self.train_dataset_config.ds_operator.get_target_var_metadata()


def collate_batch(
    batch,
    return_spatial_mask: bool = False,
    reduce_spatial_mask: bool = False,
):
    """
    Collate inference samples into a batch.

    Parameters
    ----------
    batch : sequence
        Sequence of sample dictionaries or ``(sample, metadata)`` pairs.
    return_spatial_mask : bool, optional
        Whether to compute a mask identifying valid spatial input values.
    reduce_spatial_mask : bool, optional
        Whether to reduce the spatial mask across the batch dimension and
        reuse it as a shared mask.

    Returns
    -------
    BatchData
        Collated inference batch.

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
