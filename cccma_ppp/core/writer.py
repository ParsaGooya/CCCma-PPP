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
from cccma_ppp.generic import Distributed, RuntimeContext, RunningCovariance
from cccma_ppp.train import TrainDataloaderConfig
from cccma_ppp.core import moduleABC
from cccma_ppp.core.selectors import PredictorSelector
from cccma_ppp.preprocessing import PreprocessingPipeline

from cccma_ppp.configs import (required_sample_dimensions,
                               optional_sample_dimensions)

@dataclasses.dataclass
class WriterConfig:

    predictor: PredictorSelector
    num_output_covariance_sampling: int = 0
    saved_model_training_vars_from_validation: bool = False

    def __post_init__(self):

        if self.num_output_covariance_sampling < 0:
            raise ValueError(
                "num_output_covariance_sampling cannot be negative."
            )

    def build(
        self,
        inference_data_loader: Dataloader,
        train_dataloader_config: TrainDataloaderConfig,
        module: moduleABC,
        post_processor: PreprocessingPipeline,
        output_dir: Path | str,
    ):


        return Writer(
            config=self,
            inference_data_loader=inference_data_loader,
            train_dataloader_config=train_dataloader_config,
            module=module,
            post_processor=post_processor,
            output_dir = output_dir
        )


class Writer:

    def __init__(
        self,
        config: WriterConfig,
        inference_data_loader: Dataloader,
        train_dataloader_config: TrainDataloaderConfig,
        module: moduleABC,
        post_processor: PreprocessingPipeline,
        output_dir: Path | str
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

        self.predictor = self.config.predictor.build_predictor(self.module,                                              
                                                               self.distributed,
                                                               self.output_dir,
                                                               self.config.num_output_covariance_sampling)

        if self.predictor.extract_training_vars:
            self.log_root(logging.INFO, "Running the model to extract training statistics. \n" \
                            "This might take a few minutes...")
            self._save_train_stats()

        self._setup = True
        self.log_root(logging.INFO, "Writer setup complete.")

    @property
    def train_stats_save_dir(self):
        return Path(self.output_dir) / "training_variable_stats.pt"

    def predict(self):
  
        if not self._setup:
            raise RuntimeError(
                "Call setup_distributed() before predict()."
            )
        self.log_root(logging.INFO, "Starting Inference Loop...")
        self.start_time = time.time()
        clear_memory()

        self._predict()
 
        time_elapsed = time.time() - self.start_time

        self.log_root(logging.INFO, f"Inference finished in {time_elapsed:.2f}s")

    def _predict(self):

        self.module.eval()

        with torch.inference_mode():
            
            for batch in tqdm(
                self.InferenceLoader,
                disable=not self.is_on_root,
                desc="Inference",
            ):

                batch = batch.to_device(self.device)

                self.predictor._infer_on_batch(batch)
        
            self.aggregate_predictions_to_netcdf()

    def _save_train_stats(self):

        if not self.train_stats_save_dir.exists():

            self.TrainLoaderConfig.setup_distributed(self.distributed,
                    load_path= Path(RuntimeContext.GLOBAL_EXP_DIR) / "preprocessing_pipeline")
            
            if self.config.saved_model_training_vars_from_validation:
                TrainLoader =  self.TrainLoaderConfig.build_validation_loader(supress_error = False) 
            else:
                TrainLoader =  self.TrainLoaderConfig.build_train_loader()  

            stats = self.predictor.stats    
            for batch in tqdm(TrainLoader,
                            disable=not self.is_on_root,
                            desc="Extracting training statistics"):

                batch = batch.to_device(self.device)

                stats = self.predictor._infer_on_batch(
                    batch,
                    _getting_train_stats=True,
                )
            
            self.aggregate_train_stats(stats)
            del TrainLoader
            gc.collect()

        if self.is_distributed:
            self.distributed.barrier()


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
    
    def aggregate_predictions_to_netcdf(self):
        if self.is_distributed:
            self.distributed.barrier()


        naming_convention = "prediction"
        if self.predictor.save_latent:
            naming_convention = "latent"

        if self.is_on_root:
            aggregate_predictions(self.post_processor, 
                                self.output_dir, 
                                naming_convention,
                                self.log_root)
        
        if self.is_distributed:
            self.distributed.barrier()

            


def aggregate_predictions(post_processor: PreprocessingPipeline,
                          output_dir: Path, 
                          naming_convention: str = "prediction",
                          logger_function: callable = None,
                          cleanup_temp: bool = True):

    temp_save_dir = Path(output_dir) / "_temp"
    output_dir = Path(output_dir)

    temp_files = sorted(
        temp_save_dir.glob(f"{naming_convention}_rank*_*.nc")
    )

    if not temp_files:
        raise RuntimeError(
            f"No temporary prediction files found in {temp_save_dir}"
        )

    years = set()

    for path in temp_files:
        with xr.open_dataset(path) as ds:
            if "year" not in ds.coords:
                raise RuntimeError(
                    f"{path} does not contain a 'year' coordinate."
                )

            years.update(
                np.asarray(ds["year"].values).reshape(-1).tolist()
            )

    years = sorted(years)

    if logger_function is not None:
        logger_function(
            logging.INFO,
            f"Aggregating temporary prediction files year-by-year: ",
        )

    for year in tqdm(years,
                    desc="Saving years ..."):
        year_datasets = []


        for path in temp_files:
            with xr.open_dataarray(path) as ds:
                if "year" not in ds.coords:
                    raise RuntimeError(
                        f"{path} does not contain a 'year' coordinate."
                    )

                available_years = np.asarray(
                    ds["year"].values
                ).reshape(-1)

                if year not in available_years:
                    continue

                if "year" in ds.dims:
                    ds_year_part = ds.sel(year=slice(year, year))
                else:
                    # Handles cases where year is an auxiliary coordinate.
                    ds_year_part = ds.where(
                        ds["year"] == year,
                        drop=True,
                    )

                ds_year_part = ds_year_part.load()
                year_datasets.append(ds_year_part)

        if not year_datasets:
            continue

        ds_year = xr.combine_by_coords(
            year_datasets,
            combine_attrs="override",
        )

        ds_year = next(iter(ds_year.data_vars.values()))

        sort_coords = [
            coord
            for coord in required_sample_dimensions.union(
                optional_sample_dimensions
            )
            if coord in ds_year.coords
        ]

        for coord in sort_coords:
            ds_year = ds_year.sortby(coord)

        ds_year = post_processor.to_dataset(ds_year)
        ds_year = post_processor.inverse_transform(ds_year)

        output_path = output_dir / f"{naming_convention}_{year}.nc"
        ds_year.to_netcdf(output_path)
        ds_year.close()

        for ds in year_datasets:
            ds.close()

        if logger_function is not None:
            logger_function(
                logging.INFO,
                f"Saved aggregated predictions for year {year}: {output_path}",
            )

    if cleanup_temp:
        for path in temp_files:
            path.unlink()