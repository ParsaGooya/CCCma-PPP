import numpy as np
import dataclasses
import torch
from pathlib import Path

from cccma_ppp.train.dataloader import TrainDataloaderConfig
from cccma_ppp.inference.dataset import InferenceDatasetConfig
from cccma_ppp.data_modules.dataloader import (
    Dataloader,
    DataloaderConfigABC,
    BatchDataABC,
)
from cccma_ppp.generic.distributed import Distributed
from cccma_ppp.generic.runtime import RuntimeContext


@dataclasses.dataclass
class BatchData(BatchDataABC):
    """
    Container for batched inference data.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor.
    added_features : torch.Tensor or None, optional
        Additional input features.
    metadata : list of dict or None, optional
        Per-sample metadata returned by the dataset.
    return_spatial_mask : bool, optional
        Whether to generate spatial masks.
    reduce_spatial_mask : bool, optional
        Whether to reduce masks across the batch dimension.
    """

    input: torch.Tensor
    added_features: torch.Tensor = None
    metadata: list[dict] | None = None
    return_spatial_mask: bool = False
    reduce_spatial_mask: bool = False

    def __post_init__(self):
        """
        Prepare batch data for inference.

        Returns
        -------
        None
        """
        if self.return_spatial_mask:
            self.input_mask = (~torch.isnan(self.input)).to(torch.int)
            if self.reduce_spatial_mask:
                self.input_mask = self.input_mask.mean(0)
                self.input_mask = (self.input_mask == 1).float()

        self.input = torch.nan_to_num(self.input, nan=0.0)

        if self.return_spatial_mask:
            self.input = (self.input, self.input_mask)

    def to_device(self, device: torch.device | str):
        """
        Move batch data to a device.

        Parameters
        ----------
        device : torch.device or str
            Target device.

        Returns
        -------
        BatchData
            Updated batch instance.
        """

        if self.return_spatial_mask:
            self.input = (self.input[0].to(device), self.input[1].to(device))
        else:
            self.input = self.input.to(device)

        if self.added_features is not None:
            self.added_features = self.added_features.to(device)

        return self


@dataclasses.dataclass
class InferenceDataloaderConfig(DataloaderConfigABC):
    """
    Configuration for inference dataloaders.

    Parameters
    ----------
    dataset_config : InferenceDatasetConfig or None, optional
        Inference dataset configuration.
    batch_size : int, optional
        Batch size.
    inference_years : tuple or list or None, optional
        Range of years to run inference for.
    num_data_workers : int, optional
        Number of worker processes.
    prefetch_factor : int or None, optional
        Number of batches prefetched by each worker.
    drop_last : bool, optional
        Whether to drop incomplete batches.
    """

    dataset_config: InferenceDatasetConfig | None = None
    batch_size: int = 1
    inference_years: tuple | list = None
    num_data_workers: int = 0
    prefetch_factor: int | None = None
    drop_last: bool = False

    def __post_init__(self):
        """
        Initialize dataloader configuration.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If dataset configuration has not been provided.
        """
        self._setup = False

        if self.num_data_workers == 0:
            self.prefetch_factor = None

        if self.dataset_config is not None:
            _ = self._inference_years

        else:
            raise RuntimeError(
                "Inference dataset_config must be resolved before setup."
            )

    @property
    def _inference_years(self):
        """
        Years selected for inference.

        Returns
        -------
        np.ndarray
            Array of inference years.

        Raises
        ------
        ValueError
            If requested inference years are unavailable.
        """
        if self.inference_years is None:
            return self.available_inference_years
        else:
            inference_years = np.arange(
                self.inference_years[0], self.inference_years[1] + 1
            )

            if not set(inference_years).issubset(set(self.available_inference_time)):
                raise ValueError(
                    f"the requested inference years are not available:"
                    f"available years: [{self.available_inference_time.min()},{self.available_inference_time.max()}]"
                )

            return inference_years

    @property
    def available_inference_years(self):
        """
        Available years for inference.

        Returns
        -------
        np.ndarray
            Years available in the inference dataset.
        """
        return self.dataset_config.available_inference_time

    def _input_preprocessor_exists(self, load_dir: Path | str = None):
        """
        Check whether required preprocessing pipelines exist.

        Parameters
        ----------
        load_dir : pathlib.Path or str or None, optional
            Directory containing saved preprocessing pipelines.

        Returns
        -------
        bool
            True if all required pipelines exist.
        """
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
        save_path: Path | str | None = None,
    ):
        """
        Prepare inference dataloader for distributed execution.

        Parameters
        ----------
        train_loader_config : TrainDataloaderConfig
            Training dataloader configuration.
        distributed : Distributed
            Distributed execution context.
        save_path : pathlib.Path or str or None, optional
            Directory containing fitted preprocessors.

        Returns
        -------
        None
        """

        self.train_dataset_config = train_loader_config.dataset_config
        self.rank = distributed.rank
        self.world_size = distributed.world_size

        if not self._input_preprocessor_exists(save_path):
            if distributed.is_root():
                self.train_dataset_config._fit_preprocessors(
                    train_loader_config.train_years, save=True, save_path=save_path
                )

        distributed.barrier()

        self.dataset_config._load_fitted_preprocessors(load_dir=save_path)

        self._setup = True

    def build_inference_loader(
        self,
        return_spatial_mask=False,
        reduce_spatial_mask=False,
    ):
        """
        Construct inference dataloader.

        Parameters
        ----------
        return_spatial_mask : bool, optional
            Whether to return spatial masks.
        reduce_spatial_mask : bool, optional
            Whether to reduce masks across the batch dimension.

        Returns
        -------
        Dataloader

        Raises
        ------
        RuntimeError
            If setup has not been completed.
        """
        if not self._setup:
            raise RuntimeError(
                "Dataloader has to be setup for distributed training first by calling .setup_distributed()"
            )

        inference_dataset = self.dataset_config.build_dataset(
            years=self._inference_years, return_metadata=True
        )

        return Dataloader(
            dataset=inference_dataset,
            config=self,
            collate_fn=collate_batch,
            rank=self.rank,
            world_size=self.world_size,
            return_spatial_mask=return_spatial_mask,
            reduce_spatial_mask=reduce_spatial_mask,
        )

    @property
    def input_var_metadata(self):
        """
        Metadata describing model inputs.

        Returns
        -------
        dict
        """
        return self.dataset_config.ds_operator.get_input_var_metadata()

    @property
    def target_var_metadata(self):
        """
        Metadata describing model outputs.

        Returns
        -------
        dict

        Raises
        ------
        RuntimeError
            If training dataset metadata is unavailable.
        """
        if not hasattr(self, "train_dataset_config"):
            raise RuntimeError(
                "output variables metadata cannot be read unless train dataloader"
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
    batch : list
        Dataset samples.
    return_spatial_mask : bool, optional
        Whether to return spatial masks.
    reduce_spatial_mask : bool, optional
        Whether to reduce masks across the batch dimension.

    Returns
    -------
    BatchData
        Batched inference data.
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
