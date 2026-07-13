import torch
from pathlib import Path
import logging
import dataclasses
import gc
import os
import time
from tqdm import tqdm
import pandas as pd
import numpy as np
import xarray as xr

from cccma_ppp.data_modules.dataloader import Dataloader
from cccma_ppp.generic import Distributed, RuntimeContext, RunningCovariance
from cccma_ppp.train import TrainDataloaderConfig
from cccma_ppp.core import moduleABC
from cccma_ppp.core.selectors import PredictorSelector
from cccma_ppp.preprocessing import PreprocessingPipeline

@dataclasses.dataclass
class WriterConfig:

    predictor: PredictorSelector
    num_output_covariance_sampling: int = 0
    saved_model_training_vars_from_validation: bool = False

    def __post_init__(self):

        pass

    def build(
        self,
        inference_data_loader: Dataloader,
        train_dataloader_config: TrainDataloaderConfig,
        module: moduleABC,
        output_dir: Path | str,
    ):


        return Writer(
            config=self,
            inference_data_loader=inference_data_loader,
            train_dataloader_config=train_dataloader_config,
            module=module,
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
                            "This will take a few minutes...")
            self._save_train_stats()

        self._setup = True
        self.log_root(logging.INFO, "Writer setup complete.")

    @property
    def train_stats_save_dir(self):
        return Path(self.output_dir) / "training_variable_stats.pt"

    def predict(self):
  
        assert self._setup, "make sure to setup the trainer first."
        self.log_root(logging.INFO, "Starting Inference Loop...")
        self.start_time = time.time()
        self._clear_memory()

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


    def _clear_memory(self):
        """
        Clear CPU and GPU memory.

        Returns
        -------
        None
        """

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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

        if not self.is_on_root:
            return
        
        naming_convention = "prediction"
        if self.predictor.save_latent:
            naming_convention = "latent"

        aggregate_predictions(self.post_processor, 
                              self.output_dir, 
                              naming_convention,
                              self.log_root)
        
        if self.is_distributed:
            self.distributed.barrier()

            


def aggregate_predictions(post_processor: PreprocessingPipeline,
                          output_dir: Path, 
                          naming_convention: str = "prediction",
                          logger_function: callable = None):

    temp_save_dir = Path(output_dir) / "_temp"
    temp_files = sorted(temp_save_dir.glob(f"{naming_convention}_rank*_*.nc"))

    if not temp_files:
        raise RuntimeError(f"No temporary prediction files found in {temp_save_dir}")

    output_dir = Path(output_dir)

    # Read only coordinates first to discover years.
    years = set()
    for path in temp_files:
        ds = xr.open_dataset(path)
        try:
            if "year" not in ds.coords:
                ds = ds.set_index(batch=[name for name in ds["batch"].to_index().names])
                ds = ds.unstack("batch")

            if "year" in ds.coords:
                years.update(np.asarray(ds["year"].values).ravel().tolist())
            elif "year" in ds.dims:
                years.update(np.asarray(ds["year"].values).ravel().tolist())
            else:
                batch_index = ds.indexes["batch"]
                years.update(batch_index.get_level_values("year").unique().tolist())
        finally:
            ds.close()

    years = sorted(int(y) for y in years)

    if logger_function is not None:
        logger_function(
        logging.INFO,
        f"Aggregating temporary prediction files year-by-year: {years}",
    )

    for year in years:
        year_datasets = []

        for path in temp_files:
            ds = xr.open_dataset(path)

            batch_index = ds.indexes.get("batch", None)

            if batch_index is None or not isinstance(batch_index, pd.MultiIndex):
                ds.close()
                raise RuntimeError(
                    f"{path} does not contain a MultiIndex batch dimension."
                )

            if "year" not in batch_index.names:
                ds.close()
                raise RuntimeError(
                    f"{path} batch MultiIndex does not contain a 'year' level."
                )

            mask = batch_index.get_level_values("year") == year

            if not np.any(mask):
                ds.close()
                continue

            ds_year = ds.isel(batch=np.where(mask)[0])

            # Load the selected subset, then close the file handle.
            ds_year = ds_year.load()
            ds.close()

            ds_year = ds_year.unstack("batch")
            year_datasets.append(ds_year)

        if not year_datasets:
            continue

        ds_year = xr.combine_by_coords(
            year_datasets,
            combine_attrs="override",
        )

        # Sort dimensions if present.
        sort_dims = [
            dim
            for dim in ("year", "lead_time", "ensembles")
            if dim in ds_year.dims or dim in ds_year.coords
        ]

        if sort_dims:
            ds_year = ds_year.sortby(sort_dims)

        output_path = output_dir / f"{naming_convention}_{year}.nc"

        ds_year = post_processor.to_dataset(ds_year)
        ds_year = post_processor.inverse_transform(ds_year)

        ds_year.to_netcdf(output_path)

        for ds in year_datasets:
            ds.close()

        ds_year.close()

        if logger_function is not None:
            logger_function(
                logging.INFO,
                f"Saved aggregated predictions for year {year}: {output_path}",
            )
