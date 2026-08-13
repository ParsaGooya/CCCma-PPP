import torch
from pathlib import Path
import logging
import dataclasses
import gc
import os
import time
from tqdm import tqdm
import numpy as np
import xarray as xr

from cccma_ppp.core.trainer import clear_memory

from cccma_ppp.data_modules.dataloader import Dataloader
from cccma_ppp.generic.distributed import Distributed
from cccma_ppp.generic.runtime import RuntimeContext
from cccma_ppp.generic.aggregator import RunningCovariance
from cccma_ppp.train.dataloader import TrainDataloaderConfig
from cccma_ppp.core.core_abc import moduleABC

from cccma_ppp.preprocessing.preprocessing import PreprocessingPipeline

from cccma_ppp.inference.predictors.deterministic import DeterministicPredictorConfig
from cccma_ppp.inference.predictors.cvae import cVAEPredictorConfig

from cccma_ppp.configs import required_sample_dimensions, realization_dim

init_time_dim, lead_time_dim = required_sample_dimensions

@dataclasses.dataclass
class WriterConfig:
    predictor: DeterministicPredictorConfig | cVAEPredictorConfig = dataclasses.field(
        default_factory=DeterministicPredictorConfig
    )

    num_output_sampling: int = 0
    get_trained_model_stats_from_validation: bool = False

    def __post_init__(self):

        if self.num_output_sampling < 0:
            raise ValueError("num_output_sampling cannot be negative.")

    def build(
        self,
        inference_data_loader: Dataloader,
        train_dataloader_config: TrainDataloaderConfig,
        module: moduleABC,
        post_processor: PreprocessingPipeline,
        output_dir: Path | str,
    ):

        if self.predictor._type != module.config._type.lower():
            raise RuntimeError(
                f"The provided selector config matches {self.predictor._type}"
                f" selector but the module is {module.config._type.lower()}"
            )

        return Writer(
            config=self,
            inference_data_loader=inference_data_loader,
            train_dataloader_config=train_dataloader_config,
            module=module,
            post_processor=post_processor,
            output_dir=output_dir,
        )


class Writer:
    def __init__(
        self,
        config: WriterConfig,
        inference_data_loader: Dataloader,
        train_dataloader_config: TrainDataloaderConfig,
        module: moduleABC,
        post_processor: PreprocessingPipeline,
        output_dir: Path | str,
    ):

        self.config = config
        self.module = module
        self.output_dir = Path(output_dir)
        self.InferenceLoader = inference_data_loader
        self.TrainLoaderConfig = train_dataloader_config
        self.post_processor = post_processor

        self._setup = False

    def setup_distributed(
        self,
        distributed: Distributed,
        logger: logging.Logger,
    ):

        self.logger = logger
        if self.logger is None:
            print("Logger is None. Print is used instead ... \n\n ")

        self.distributed = distributed
        self.device = distributed.device
        self.rank = distributed.rank
        self.local_rank = distributed.local_rank
        self.world_size = distributed.world_size
        self.is_distributed = distributed.distributed
        self.is_on_root = distributed.is_root()

        self.log_root(logging.INFO, "Setting up writer.")

        self.log_root(
            logging.INFO,
            f"rank={self.rank} | local_rank={self.local_rank} | world_size={self.world_size} | device={self.device}",
        )

        if self.raw_module._get_device() != self.device:
            raise RuntimeError(
                f"Module is on {self.raw_module._get_device()}, but trainer device is {self.device}"
            )

        self.temp_save_dir = Path(self.output_dir) / "_temp"
        if self.is_on_root:
            os.makedirs(self.temp_save_dir, exist_ok=True)

        if self.is_distributed:
            self.distributed.barrier()

        self.predictor = self.config.predictor.build(
            self.module,
            self.distributed,
            self.output_dir,
            self.config.num_output_sampling,
        )

        if self.predictor.extract_training_vars:
            self.log_root(
                logging.INFO,
                "Running the model to extract training statistics. \n"
                "This might take a few minutes...",
            )
            self._save_train_stats()

        self._setup = True
        self.log_root(logging.INFO, "Writer setup complete.")

    @property
    def train_stats_save_dir(self):
        return Path(self.output_dir) / "training_variable_stats.pt"

    def predict(self):

        if not self._setup:
            raise RuntimeError("Call setup_distributed() before predict().")
        self.log_root(logging.INFO, "Starting Inference Loop...")
        self.start_time = time.time()
        clear_memory()

        self._predict()

        time_elapsed = time.time() - self.start_time

        self.log_root(logging.INFO, f"Inference finished in {time_elapsed:.2f}s")

    def _predict(self):

        self.module.eval()
        loader = self.InferenceLoader
        do_post_process = True

        if getattr(self.predictor, "save_latent", False):
            loader = self.build_train_loader(return_metadata=True, shuffle=False)
            do_post_process = False
        with torch.inference_mode():
            for batch in tqdm(
                loader,
                disable=not self.is_on_root,
                desc="Inference",
            ):
                batch = batch.to_device(self.device)

                self.predictor._infer_on_batch(batch)

            del loader
            clear_memory()
            self.aggregate_predictions_to_netcdf(do_post_process)

    def _save_train_stats(self):

        if not self.train_stats_save_dir.exists():
            loader = self.build_train_loader(
                from_validation=self.config.get_trained_model_stats_from_validation
            )

            stats = self.predictor.stats
            for batch in tqdm(
                loader,
                disable=not self.is_on_root,
                desc="Extracting training statistics",
            ):
                batch = batch.to_device(self.device)

                stats = self.predictor._infer_on_batch(
                    batch,
                    _getting_train_stats=True,
                )

            del loader
            clear_memory()
            self.aggregate_train_stats(stats)

        if self.is_distributed:
            self.distributed.barrier()

    def build_train_loader(
        self,
        from_validation: bool = False,
        return_metadata: bool = False,
        shuffle: bool | None = None,
    ):

        self.TrainLoaderConfig.setup_distributed(
            self.distributed,
            load_path=Path(RuntimeContext.GLOBAL_EXP_DIR) / "preprocessing_pipeline",
        )

        if from_validation:
            return self.TrainLoaderConfig.build_validation_loader(
                supress_error=False, return_metadata=return_metadata, shuffle=shuffle
            )
        else:
            return self.TrainLoaderConfig.build_train_loader(
                return_metadata=return_metadata, shuffle=shuffle
            )

    def aggregate_train_stats(self, stats: dict[str, RunningCovariance]):

        for stat in stats.values():
            if stat.sum_x is not None:
                stat.distributed_reduce()

        if self.is_on_root:
            save_dict = {}

            for name, stat in stats.items():
                if stat.sum_x is None:
                    continue

                mean, cov = stat.finalize()
                save_dict[f"{name}_mean"] = mean
                save_dict[f"{name}_cov"] = cov

            torch.save(
                save_dict,
                self.train_stats_save_dir,
            )

        if self.is_distributed:
            self.distributed.barrier()

    def log_root(self, level: int, msg: str, *args):
        """
        Log message from root process.

        Parameters
        ----------
        level : int
            Logging level.
        msg : str
            Message.
        *args
            Formatting arguments.

        Returns
        -------
        None
        """

        if self.is_on_root:
            if self.logger is not None:
                self.logger.log(level, msg, *args)
            else:
                print(msg)

    @property
    def raw_module(self):
        if isinstance(self.module, torch.nn.parallel.DistributedDataParallel):
            return self.module.module
        return self.module

    def aggregate_predictions_to_netcdf(self, do_post_process: bool = True):
        if self.is_distributed:
            self.distributed.barrier()

        post_processor = self.post_processor
        if not do_post_process:
            post_processor = None

        load_naming_convention = "prediction"
        if getattr(self.predictor, "save_latent", False):
            load_naming_convention = "latent"

        save_naming_convention = load_naming_convention

        if hasattr(self.predictor, "nstds"):
            save_naming_convention += f"_{self.predictor.nstds}stds"

        if self.config.num_output_sampling > 0:
            save_naming_convention += "_output_ensemble"

        if self.is_on_root:
            aggregate_predictions(
                post_processor=post_processor, 
                output_dir=self.output_dir, 
                load_naming_convention=load_naming_convention, 
                save_naming_convention=save_naming_convention,
                logger_function=self.log_root
            )

        if self.is_distributed:
            self.distributed.barrier()



def aggregate_predictions(
    post_processor: PreprocessingPipeline | None,
    output_dir: Path,
    load_naming_convention: str = "prediction",
    save_naming_convention: str = None,
    logger_function: callable = None,
    cleanup_temp: bool = True,
    init_time_dim: str = init_time_dim,
    lead_time_dim: str = lead_time_dim,
    optional_sample_dimensions: tuple[str, ...] = (realization_dim,),
):
    """
    Aggregate temporary inference batches into one file per
    initialization year.
    """
    if save_naming_convention is None:
        save_naming_convention = load_naming_convention

    output_dir = Path(output_dir)
    temp_save_dir = output_dir / "_temp"
    
    temp_files = sorted(
        temp_save_dir.glob(f"{load_naming_convention}_rank*_*.nc")
    )

    if not temp_files:
        raise RuntimeError(
            f"No temporary prediction files found in {temp_save_dir}."
        )

    # Collect initialization times while preserving DatetimeIndex or
    # CFTimeIndex behavior.
    all_times = None

    for path in temp_files:
        with xr.open_dataarray(path) as data:
            if init_time_dim not in data.coords:
                raise RuntimeError(
                    f"{path} does not contain the coordinate "
                    f"{init_time_dim!r}."
                )

            time_index = data.coords[init_time_dim].to_index()

            all_times = (
                time_index
                if all_times is None
                else all_times.union(time_index)
            )

    if all_times is None or len(all_times) == 0:
        raise RuntimeError(
            "Temporary prediction files contain no initialization times."
        )

    all_times = all_times.sort_values()

    # Works for both datetime64-backed DatetimeIndex and CFTimeIndex.
    all_times_da = xr.DataArray(
        all_times,
        dims=(init_time_dim,),
        coords={init_time_dim: all_times},
    )

    initialization_years = np.asarray(
        all_times_da.dt.year.values
    )

    years = np.unique(initialization_years)

    if logger_function is not None:
        logger_function(
            logging.INFO,
            "Aggregating temporary prediction files year-by-year.",
        )

    sample_coords = (
        init_time_dim,
        lead_time_dim,
        *optional_sample_dimensions,
    )

    def _sort_sample_coords(
        data: xr.DataArray | xr.Dataset,
    ) -> xr.DataArray | xr.Dataset:
        for coord in sample_coords:
            if coord in data.coords:
                data = data.sortby(coord)

        return data

    for year in tqdm(years, desc="Saving years ..."):
        year_times = all_times[
            initialization_years == year
        ]

        year_datasets = []

        for time in year_times:
            time_datasets = []

            for path in temp_files:
                with xr.open_dataarray(path) as data:
                    if init_time_dim not in data.coords:
                        raise RuntimeError(
                            f"{path} does not contain the coordinate "
                            f"{init_time_dim!r}."
                        )

                    available_times = data.coords[
                        init_time_dim
                    ].to_index()

                    if time not in available_times:
                        continue

                    data_time_part = data.sel(
                        {init_time_dim: slice(time, time)}
                    )

                    for dim in sample_coords:
                        if (
                            dim != init_time_dim
                            and dim in data_time_part.dims
                        ):
                            data_time_part = data_time_part.dropna(
                                dim=dim,
                                how="all",
                            )

                    data_time_part = _sort_sample_coords(
                        data_time_part.load()
                    )

                    time_datasets.append(data_time_part)

            if not time_datasets:
                continue

            combined_time = xr.concat(
                time_datasets,
                dim=lead_time_dim,
                coords="minimal",
                compat="equals",
                join="exact",
            )

            lead_times = np.asarray(
                combined_time[lead_time_dim].values
            )

            _, unique_indices = np.unique(
                lead_times,
                return_index=True,
            )

            if len(unique_indices) != len(lead_times):
                combined_time = combined_time.isel(
                    {
                        lead_time_dim: np.sort(
                            unique_indices
                        )
                    }
                )

            combined_time = combined_time.sortby(
                lead_time_dim
            )

            if not combined_time.indexes[
                lead_time_dim
            ].is_monotonic_increasing:
                raise RuntimeError(
                    "Lead times remain non-monotonic for "
                    f"initialization time {time}: "
                    f"{combined_time[lead_time_dim].values}"
                )

            data_time = _sort_sample_coords(combined_time)

            if post_processor is not None:
                data_time = post_processor.to_dataset(
                    data_time
                )
                data_time = post_processor.inverse_transform(
                    data_time
                )
            else:
                data_time = data_time.to_dataset(
                    dim="channels"
                )

            year_datasets.append(data_time)

        if not year_datasets:
            continue

        data_year = xr.concat(
            year_datasets,
            dim=init_time_dim,
            coords="minimal",
            compat="equals",
            join="exact",
        )

        data_year = _sort_sample_coords(data_year)

        if not data_year.indexes[
            init_time_dim
        ].is_monotonic_increasing:
            raise RuntimeError(
                "Initialization times remain non-monotonic for "
                f"year {year}: "
                f"{data_year[init_time_dim].values}"
            )

        output_path = (
            output_dir
            / f"{save_naming_convention}_{int(year)}.nc"
        )

        if post_processor is not None:
            data_year = post_processor.inverse_rename(data_year)
            
        data_year.to_netcdf(output_path)
        data_year.close()

        for data in year_datasets:
            data.close()

        if logger_function is not None:
            logger_function(
                logging.INFO,
                "Saved aggregated predictions for "
                f"initialization year {year}: {output_path}",
            )

    if cleanup_temp:
        for path in temp_files:
            path.unlink()

        temp_save_dir.rmdir()