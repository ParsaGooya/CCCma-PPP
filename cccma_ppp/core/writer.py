import torch
from torch.cuda.amp import GradScaler
from pathlib import Path
import logging
import dataclasses
import gc
import os
import time

from cccma_ppp.core import moduleABC, OptimizerWrapper
from cccma_ppp.core.cVAE_module import cVAE
from cccma_ppp.data_modules.dataloader import Dataloader
from cccma_ppp.generic import Distributed, RuntimeContext
from cccma_ppp.train import TrainDataloaderConfig


@dataclasses.dataclass
class WriterConfig:

    output_ensemble_size: int = 1
    output_sampler: OutputsamplerConfig | None = None

    def __post_init__(self):

        pass #setup output_sampler configs

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
            output_sampler = self.output_sampler,
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
        output_sampler: OutputsamplerConfig,
        output_dir: Path | str
    ):


        self.config = config
        self.module = module
        self.output_dir = output_dir
        self.InferenceLoader = inference_data_loader
        self.TrainLoaderConfig = train_dataloader_config
        self.output_sampler = output_sampler

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

        self.output_sampler = self.output_sampler.build() #####

        self.temp_save_dir = Path(self.output_dir) / "_temp"
        if self.is_on_root:
            os.makedirs(self.temp_save_dir, exist_ok=True)

        if self.is_distributed:
            self.distributed.barrier()

        self._setup = True
        self.log_root(logging.INFO, "Writer setup complete.")

    def predict(self):
  
        assert self._setup, "make sure to setup the trainer first."
        self.log_root(logging.INFO, "Starting Inference Loop...")
        self.start_time_train = time.time()
        self._clear_memory()

        self._infer()
 
        time_elapsed = time.time() - self.start_time_train

        self.log_root(logging.INFO, f"Training finished in {time_elapsed:.2f}s")

    def _infer(self):

        self.module.eval()

        for batch in self.InferenceLoader:
            output = self._infer_on_batch(batch)
            output = self.output_sampler.add_decoder_noise(output, batch.metadata)
            self._batch_to_netcdf(output, batch.metadata)

        self.aggregate_predictions_to_netcdf(self.output_dir)

    def _get_train_stats(self):

        self.module.eval()
        if couldnot_load:
            self.TrainLoaderConfig.setup_distributed(self.distributed,
                    load_path= Path(RuntimeContext.GLOBAL_EXP_DIR) / "preprocessing_pipeline")
            TrainLoader =  self.TrainLoaderConfig.build_train_loader()                              
        
            for batch in TrainLoader:
                output = self._infer_on_batch(batch)
                mu.append(output.mu)
                log_var.append(output.logvar)
                ...
                residuals = batch.target - output
        else:
            stats_dir = Path(RuntimeContext.GLOBAL_EXP_DIR) / "train_stats"
            residuals, mu, log_var = self._load_train_stats(stats_dir)



    @torch.no_grad()
    def _infer_on_batch(self, batch):

        self._clear_memory()
        batch.to_device(self.device)
        kwargs = {}

        if self.raw_module.config._type.lower() not in ["deterministic", "default"]:
            kwargs["sample_size"] = self.config.output_ensemble_size

        with torch.cuda.amp.autocast(
            enabled=self.scaler.is_enabled() and self.device.type == "cuda"
        ):
            output = self.raw_module.predict(data=batch, **kwargs)

        return output.output


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

    def aggregate_predictions_to_netcdf(self, output_dir: Path | str):

        pass


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
        """
        Access underlying module (unwrap DDP if needed).

        Returns
        -------
        moduleABC
            Raw model instance.
        """

        if isinstance(self.module, torch.nn.parallel.DistributedDataParallel):
            return self.module.module
        return self.module