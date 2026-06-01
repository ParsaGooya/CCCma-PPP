import numpy as np
import torch
from torch.cuda.amp import GradScaler
from pathlib import Path
import logging
import dataclasses
import gc
import os
import time

from cccma_ppp.core.core_abc import moduleABC
from cccma_ppp.core.optimization import *
from cccma_ppp.core.cVAE_module import cVAE
from cccma_ppp.data.dataloader import Dataloader
from cccma_ppp.generic.distributed import Distributed
from cccma_ppp.generic.aggregator import MetricsAggregator
from loss.kld import BetaAnnealing




@dataclasses.dataclass
class TrainerConfig:
    beta_finder : BetaAnnealing | None = None
    earlystoppingbuffer : float = float("inf")
    minimum_validation_improvement_percentage : float = 0.02
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    grad_clip: float | None = None


    def __post_init__(self):

        if self.grad_clip is not None:
            assert self.grad_clip > 0


    def build(self,
              train_data_loader : Dataloader,
              validation_data_loader : Dataloader,
              optimization : OptimizerWrapper,
              module : moduleABC,
              epochs : int):


        self.num_train_batches = len(train_data_loader)
        if validation_data_loader is not None:
            self.num_validation_batches = len(validation_data_loader)

        assert module.built, 'make sure the module is built.'

        if isinstance(module, cVAE):
            assert self.beta_finder is not None, 'specify beta annealing config for cVAE module.'
            self.beta_finder.build(self.num_train_batches)

        return Trainer(config = self,
                        train_data_loader = train_data_loader,
                        validation_data_loader  = validation_data_loader,
                        module = module,
                        optimizer = optimization,
                        epochs = epochs)







class Trainer:
        def __init__(
            self,
            config : TrainerConfig,
            train_data_loader : Dataloader,
            module : moduleABC,
            optimizer : OptimizerWrapper,
            epochs : int,
            validation_data_loader : Dataloader | None = None):


            self.config = config
            self.optimizer = optimizer
            self.module = module
            self.TrainLoader = train_data_loader
            self.ValidationLoader = validation_data_loader

            self.epochs = epochs
            self.global_step = 0
            self.batch_step = 0
            self._start_epoch = 0
            self._epochs_trained = self._start_epoch
            # self._current_epoch_num_batches_seen = 0   ###you can work with this if you need batch level checkpoining and restarting
            self._best_validation_loss = float("inf")
            self.earlystopping_counter = 0

            if self.config.beta_finder is not None:
                self.beta_finder =  self.config.beta_finder

            self._setup = False

        def setup_distributed(
                    self,
                    distributed  : Distributed,
                    logger : logging.Logger,
                    log_every_n_epochs : int = 1,
                    save_checkpoint :  bool = True ):

            self.save_checkpoint = save_checkpoint
            self.log_every_n_epochs = log_every_n_epochs

            self.logger = logger
            if self.logger is None:
                print('Logger is None. Print is used instead ... \n\n ')

            self.distributed = distributed
            self.device = distributed.device
            self.rank = distributed.rank
            self.local_rank = distributed.local_rank
            self.world_size = distributed.world_size
            self.is_distributed = distributed.distributed
            self.is_on_root = distributed.is_root()

            self.log_root(logging.INFO, "Setting up trainer.")

            self.log_root(logging.INFO,
                f"rank={self.rank} | local_rank={self.local_rank} | world_size={self.world_size} | device={self.device}")


            if self.raw_module._get_device() != self.device:
                raise RuntimeError(f"Module is on {self.raw_module._get_device()}, but trainer device is {self.device}")

            self.scaler = GradScaler("cuda", enabled=self.config.mixed_precision)

            self.train_aggregator = MetricsAggregator(distributed, name = 'Train')
            self.validation_aggregator = MetricsAggregator(distributed, name = 'Validation')  if self.ValidationLoader is not None else None ##to do: add flexible validation loss


            if not self.save_checkpoint:
                self.log_root( logging.warning,
                    "Configured value of save_checkpoint is false, no checkpoints whatsoever will be saved! ")


            self.checkpoint_dir = Path(os.environ["GLOBAL_CHECKPOINT_DIR"])
            resuming = os.path.isfile(self.checkpoint_dir / '*latest*.pt')
            if resuming:
                self.log_root(logging.INFO, f"Resuming training from {self.checkpoint_dir / '*latest*.pt' }")
                self._load_checkpoint(self.checkpoint_dir / '*latest*.pt')
            else:
                if self.is_on_root and self.save_checkpoint:
                    os.makedirs(self.checkpoint_dir, exist_ok=True)

            self.plot_dir = Path(os.environ["GLOBAL_FIGURES_DIR"])
            if self.is_on_root:
                    os.makedirs(self.plot_dir, exist_ok=True)

            if self.is_distributed:
                self.distributed.barrier()

            self._setup = True
            self.log_root(logging.INFO, "Trainer setup complete.")

        def train(self):
            """
            Main training loop.

            Responsibilities:
            - loop over epochs
            - run training epoch
            - optionally run validation epoch
            - handle early stopping
            - checkpoint from root rank only
            """
            assert self._setup, 'make sure to setup the trainer first.'
            self.log_root(logging.INFO, "Starting Training Loop...")
            self.start_time_train = time.time()
            self._clear_memory()
            self.optimizer.zero_grad(set_to_none=True)

            while self._epochs_trained < self.epochs:

                time_elapsed = self._train_on_epoch()
                train_logs = self.train_aggregator._dist_compute()
                self.train_aggregator.record_epoch(train_logs, time_elapsed = time_elapsed)

                if self.ValidationLoader is not None:

                    self._validate_on_epoch()
                    validation_logs = self.validation_aggregator._dist_compute()
                    self.validation_aggregator.record_epoch(validation_logs)

                    validation_loss = validation_logs['total_loss'] ## can later be an input atgument that customizes validation loss
                    improved = self._is_improved(validation_loss)

                    if improved:
                        self._best_validation_loss = validation_loss
                        self.earlystopping_counter = 0

                        if self.is_on_root and self.save_checkpoint:
                            self._save_checkpoint(
                                name="best",
                                train_logs=train_logs,
                                validation_logs=validation_logs,
                            )
                    else:
                        self.earlystopping_counter += 1

                else:
                    validation_logs = None
                    validation_loss = None

                    if self.is_on_root and self.save_checkpoint:
                        self._save_checkpoint(
                            name="latest",
                            train_logs=train_logs,
                            validation_logs=None,
                        )

                if self.is_on_root:
                    if (self._epochs_trained ) % self.log_every_n_epochs == 0:
                        self._log_epoch(
                            train_logs=train_logs,
                            validation_logs=validation_logs,
                        )
                        MetricsAggregator.plot([self.train_aggregator, self.validation_aggregator], plot_dir = self.plot_dir)


                should_stop = self._should_stop_early()
                if self.is_distributed:
                    stop_tensor = torch.tensor(
                            int(self._should_stop_early()),
                            device=self.device)
                    self.distributed.broadcast(stop_tensor, src=0)
                    should_stop = bool(stop_tensor.item())

                if should_stop:


                    self.log_root(logging.INFO,
                            f"Early stopping triggered.  Best validation loss: {self._best_validation_loss:.6f}",
                            )

                    break

            if self.batch_step % self.config.gradient_accumulation_steps != 0:
                # The model changed after the last validation/checkpoint,
                # so  validate/checkpoint again.

                if not self._should_stop_early():

                    self._optimizer_step()

                    if self.ValidationLoader is not None:
                        self._validate_on_epoch()
                        validation_logs = self.validation_aggregator._dist_compute()
                        self.validation_aggregator.record_epoch(validation_logs, index = -1)

                        validation_loss = validation_logs['total_loss'] ## can later be an input atgument that customizes validation loss
                        improved = self._is_improved(validation_loss)

                        if improved:
                            self._best_validation_loss = validation_loss

                            self.validation_aggregator.remove_second_last_epoch()

                            if self.is_on_root and self.save_checkpoint:
                                self._save_checkpoint(
                                    name="best",
                                    train_logs=train_logs,
                                    validation_logs=validation_logs,
                                )
                    elif self.is_on_root and self.save_checkpoint:

                        self._save_checkpoint(
                                name="latest",
                                train_logs=train_logs,
                                validation_logs=None,
                            )


            time_elapsed = time.time() - self.start_time_train

            if self.is_on_root:
                MetricsAggregator.plot([self.train_aggregator, self.validation_aggregator], plot_dir = self.plot_dir)
            self.log_root(logging.INFO, f"Training finished in {time_elapsed:.2f}s")



        def _train_on_epoch(self):

            """
            Train for one epoch.

            Responsibilities:
            - set the DistributedSampler epoch
            - put module in train mode
            - iterate over train batches
            - call _train_on_batch()
            - record batch metrics
            """
            # self.log_root( logging.INFO, f'epoch {self._epochs_trained + 1}/{self.epochs}')
            self.TrainLoader.set_epoch(self._epochs_trained)
            self.module.train()

            # epoch_data = self.TrainLoader.subset_loader(start_batch=self._current_epoch_num_batches_seen)  ###you can work with this if you need batch level checkpoining and restarting

            start_time = time.time()
            for batch_id, batch in enumerate(self.TrainLoader):

                batch_loss_dict = self._train_on_batch(batch)

                # self._current_epoch_num_batches_seen += 1      ###you can work with this if you need batch level checkpoining and restarting
                self.train_aggregator.record(batch_loss_dict)


            self._epochs_trained += 1
            # self._current_epoch_num_batches_seen = 0 ###you can work with this if you need batch level checkpoining and restarting
            time_elapsed = time.time() - start_time

            return time_elapsed

        def _train_on_batch(self, batch) :

            batch.to_device(self.device)
            kwargs = {}

            if hasattr(self, "beta_finder"):
                beta = self.beta_finder(self.global_step)
                kwargs = dict(beta = beta)

            with torch.cuda.amp.autocast( #device_type=self.device.type,
                                    enabled=self.scaler.is_enabled() and self.device.type == "cuda"):

                    loss, loss_dict = self.raw_module._compute_loss(data=batch, **kwargs)
                    loss = loss / self.config.gradient_accumulation_steps


            self.scaler.scale(loss).backward()

            self.batch_step += 1

            if self.batch_step  % self.config.gradient_accumulation_steps == 0:

                self._optimizer_step()


            return loss_dict

        @torch.no_grad()
        def _validate_on_epoch(self):
            """
            Validate for one epoch.

            Responsibilities:
            - put module in eval mode
            - iterate over validation batches
            - record batch metrics
            """
            if self.ValidationLoader is None:
                raise RuntimeError("ValidationLoader is None, but validation was requested.")

            self.module.eval()

            for batch_id, batch in enumerate(self.ValidationLoader):

                batch_loss_dict = self._validate_on_batch(batch)

                # self._current_epoch_num_batches_seen += 1      ###you can work with this if you need batch level checkpoining and restarting
                self.validation_aggregator.record(batch_loss_dict)

        @torch.no_grad()
        def _validate_on_batch(self, batch) :

            batch.to_device(self.device)
            kwargs = {}

            if hasattr(self, "beta_finder"):
                beta = self.beta_finder(self.global_step)
                kwargs = dict(beta = beta)

            with torch.cuda.amp.autocast(
                #device_type=self.device.type,
                enabled=self.scaler.is_enabled() and self.device.type == "cuda"):


                _, loss_dict = self.raw_module._compute_loss(data=batch, **kwargs)

            return loss_dict

        def _optimizer_step(self):

                if self.config.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.module.parameters(),
                        self.config.grad_clip)

                old_scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer.optimizer)
                self.scaler.update()
                new_scale = self.scaler.get_scale()

                step_was_skipped = new_scale < old_scale  ##sometimes
                if not step_was_skipped:   ## With AMP, GradScaler.step() may skip the optimizer step if gradients contain inf/nan. If that happens, you probably should not step the LR scheduler or increment global_step

                    self.optimizer.scheduler_step()
                    self.global_step += 1

                self.optimizer.zero_grad(set_to_none=True)

        def _clear_memory(self):
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


        def _is_improved(self, validation_loss: float | torch.Tensor) -> bool:
            if isinstance(validation_loss, torch.Tensor):
                validation_loss = validation_loss.item()

            if self._best_validation_loss == float("inf"):
                return True

            required_improvement = (self.config.minimum_validation_improvement_percentage * abs(self._best_validation_loss))

            return validation_loss < self._best_validation_loss - required_improvement

        def _should_stop_early(self) -> bool:
            if self.ValidationLoader is None:
                return False

            if self.config.earlystoppingbuffer is None:
                return False

            if self.config.earlystoppingbuffer == float("inf"):
                return False

            return self.earlystopping_counter >= self.config.earlystoppingbuffer

        def _log_epoch(self, train_logs, validation_logs=None):
            elapsed_time = time.time() - self.start_time_train
            msg = (
                f"Epoch {self._epochs_trained}/{self.epochs} | "
                f"train loss: {train_logs['total_loss']:.6f}")


            if validation_logs is not None:
                msg += f" | validation loss: {validation_logs['total_loss']:.6f}"

            msg += (
                f" | global step: {self.global_step} | "
                f"| early stopping counter: {self.earlystopping_counter} |"
                f" elapsed time: {elapsed_time:.2f}")


            self.log_root(logging.INFO, msg)

        def _save_checkpoint(
            self,
            name: str,
            train_logs: dict,
            validation_logs: dict | None = None):

            """
            Save checkpoint.
            For DDP, save the underlying module, not the DDP wrapper.
            """

            checkpoint = {
                "epoch": self._epochs_trained,
                "global_step": self.global_step,
                "batch_step": self.batch_step,
                "best_validation_loss": self._best_validation_loss,
                "earlystopping_counter": self.earlystopping_counter,
                "module": self.raw_module.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "train_logs": train_logs,
                "validation_logs": validation_logs,
                "train_history": self.train_aggregator.state_dict(),
                "validation_history": self.validation_aggregator.state_dict() if self.validation_aggregator is not None else None,
            }

            path = Path(self.checkpoint_dir) / f"{name}.pt"
            torch.save(checkpoint, path)

            if self.is_distributed:
                self.distributed.barrier()

        def _load_checkpoint(
            self,
            path: str | Path | None = None,
            strict: bool = True):

            """
            Load trainer checkpoint.

            Assumes:
            - setup() has already been called
            - module has already been moved to device
            - DDP wrapping has already happened if distributed
            - optimizer has already been built
            - scaler has already been created
            """

            if path is None:
                path = Path(self.checkpoint_dir) / f"latest.pt"
            else:
                path = Path(path)

            if not path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {path}")

            checkpoint = torch.load( path,map_location=self.device, weights_only=False)

            self.raw_module.load_state_dict( checkpoint["module"],strict=strict)

            self.optimizer.load_state_dict(checkpoint["optimizer"])

            if "scaler" in checkpoint and checkpoint["scaler"] is not None:
                self.scaler.load_state_dict(checkpoint["scaler"])

            self._epochs_trained = checkpoint.get("epoch", 0)
            self._start_epoch = self._epochs_trained
            self.global_step = checkpoint.get("global_step", 0)
            self.batch_step = checkpoint.get("batch_step", 0)

            self._best_validation_loss = checkpoint.get("best_validation_loss",torch.inf)
            self.earlystopping_counter = checkpoint.get("earlystopping_counter",0)

            if "train_history" in checkpoint:
                self.train_aggregator.load_state_dict(checkpoint["train_history"])

            if ("validation_history" in checkpoint
                and checkpoint["validation_history"] is not None
                and self.validation_aggregator is not None):

                self.validation_aggregator.load_state_dict(checkpoint["validation_history"])

            if self.is_distributed:
                self.distributed.barrier()

            return checkpoint


        @property
        def raw_module(self):
            if isinstance(self.module, torch.nn.parallel.DistributedDataParallel):
                return self.module.module
            return self.module

        def log_root(self, level: int, msg: str, *args):
            if self.is_on_root:
                if self.logger is not None:
                    self.logger.log(level, msg, *args)
                else:
                    print(msg)



