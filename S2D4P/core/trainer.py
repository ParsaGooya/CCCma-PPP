import torch
from torch.cuda.amp import GradScaler
from pathlib import Path
import logging
import dataclasses
import gc
import os
import time
from tqdm import tqdm

from S2D4P.core.core_abc import moduleABC
from S2D4P.core.optimization import OptimizerWrapper
from S2D4P.core.models.cvae import cVAE
from S2D4P.data.dataloader import Dataloader
from S2D4P.generic.distributed import Distributed
from S2D4P.generic.aggregator import MetricsAggregator
from S2D4P.generic.runtime import RuntimeContext

from S2D4P.loss.kld import BetaAnnealing


@dataclasses.dataclass
class TrainerConfig:
    """
    Configuration for model training.

    Parameters
    ----------
    beta_finder : BetaAnnealing | None
        Beta-annealing schedule used to weight the Kullback-Leibler divergence
        during cVAE training.
    earlystoppingbuffer : float
        Number of consecutive validation epochs without sufficient improvement
        allowed before early stopping.
    minimum_validation_improvement_percentage : float
        Minimum fractional improvement in validation loss required for an epoch
        to be considered improved.
    gradient_accumulation_steps : int
        Number of batches over which gradients are accumulated before each
        optimizer update.
    mixed_precision : bool
        Whether automatic mixed-precision training is enabled.
    grad_clip : float | None
        Maximum gradient norm applied before optimizer updates. If ``None``,
        gradient clipping is disabled.
    """

    beta_finder: BetaAnnealing | None = None
    earlystoppingbuffer: float = float("inf")
    minimum_validation_improvement_percentage: float = 0.02
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    grad_clip: float | None = None

    def __post_init__(self):
        """
        Validate the trainer configuration.

        Raises
        ------
        AssertionError
            If ``grad_clip`` is configured with a nonpositive value.
        """
        if self.grad_clip is not None:
            assert self.grad_clip > 0

    def build(
        self,
        train_data_loader: Dataloader,
        validation_data_loader: Dataloader,
        optimization: OptimizerWrapper,
        module: moduleABC,
        max_epochs: int,
    ):
        """
        Build a trainer from this configuration.

        The number of training and validation batches is recorded. For cVAE modules,
        the configured beta-annealing schedule is initialized from the number of
        training batches.

        Parameters
        ----------
        train_data_loader : Dataloader
            Data loader providing training batches.
        validation_data_loader : Dataloader | None
            Optional data loader providing validation batches.
        optimization : OptimizerWrapper
            Optimizer and learning-rate scheduler wrapper.
        module : moduleABC
            Model module to train.
        max_epochs : int
            Maximum number of training epochs.

        Returns
        -------
        Trainer
            Configured trainer.

        Raises
        ------
        ValueError
            If a cVAE module is used without a beta-annealing configuration.
        """
        self.num_train_batches = len(train_data_loader)
        if validation_data_loader is not None:
            self.num_validation_batches = len(validation_data_loader)

        if isinstance(module, cVAE):
            if self.beta_finder is None:
                raise ValueError("specify beta annealing config for cVAE module.")
            self.beta_finder.build(self.num_train_batches)
        else:
            self.beta_finder = None

        return Trainer(
            config=self,
            train_data_loader=train_data_loader,
            validation_data_loader=validation_data_loader,
            module=module,
            optimizer=optimization,
            max_epochs=max_epochs,
        )


class Trainer:
    """
    Manage model training, validation, optimization, and checkpointing.

    The trainer supports distributed training, mixed precision, gradient
    accumulation, gradient clipping, validation-based early stopping, metric
    aggregation, checkpoint resumption, and learning-rate scheduling.

    Parameters
    ----------
    config : TrainerConfig
        Configuration controlling training behavior.
    train_data_loader : Dataloader
        Data loader providing training batches.
    module : moduleABC
        Model module to train.
    optimizer : OptimizerWrapper
        Optimizer and optional learning-rate scheduler.
    max_epochs : int
        Maximum number of training epochs.
    validation_data_loader : Dataloader | None
        Optional data loader providing validation batches.
    """

    def __init__(
        self,
        config: TrainerConfig,
        train_data_loader: Dataloader,
        module: moduleABC,
        optimizer: OptimizerWrapper,
        max_epochs: int,
        validation_data_loader: Dataloader | None = None,
    ):
        """
        Initialize the trainer.

        Parameters
        ----------
        config : TrainerConfig
            Configuration controlling training behavior.
        train_data_loader : Dataloader
            Data loader providing training batches.
        module : moduleABC
            Model module to train.
        optimizer : OptimizerWrapper
            Optimizer and optional learning-rate scheduler.
        max_epochs : int
            Maximum number of training epochs.
        validation_data_loader : Dataloader | None
            Optional data loader providing validation batches.
        """
        self.config = config
        self.optimizer = optimizer
        self.module = module
        self.TrainLoader = train_data_loader
        self.ValidationLoader = validation_data_loader

        self.max_epochs = max_epochs
        self.global_step = 0
        self.batch_step = 0
        self._start_epoch = 0
        self._epochs_trained = self._start_epoch

        self._best_validation_loss = float("inf")
        self.earlystopping_counter = 0

        if self.config.beta_finder is not None:
            self.beta_finder = self.config.beta_finder

        self.num_batches = len(self.TrainLoader)
        self.remainder_batch_size = (
            self.num_batches % self.config.gradient_accumulation_steps
        )

        self._setup = False
        self._skip_training = False

    def setup_distributed(
        self,
        distributed: Distributed,
        logger: logging.Logger,
        log_every_n_epochs: int = 1,
        save_checkpoint: bool = True,
    ):
        """
        Configure the trainer for distributed execution.

        The model is wrapped with distributed data parallel execution when needed,
        mixed-precision scaling and metric aggregators are initialized, and an
        existing best checkpoint is restored when available.

        Parameters
        ----------
        distributed : Distributed
            Distributed runtime containing device, rank, local-rank, and world-size
            information.
        logger : logging.Logger
            Logger used for training status messages.
        log_every_n_epochs : int
            Number of epochs between metric logging and plot updates.
        save_checkpoint : bool
            Whether training checkpoints are saved.

        Raises
        ------
        RuntimeError
            If the model is not located on the device assigned to the trainer.
        """
        self.save_checkpoint = save_checkpoint
        self.log_every_n_epochs = log_every_n_epochs

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

        if distributed.distributed:
            self.module = torch.nn.parallel.DistributedDataParallel(
                self.module,
                device_ids=[distributed.local_rank],
                output_device=distributed.local_rank,
                find_unused_parameters=False,
            )

        self.log_root(logging.INFO, "Setting up trainer.")

        self.log_root(
            logging.INFO,
            f"rank={self.rank} | local_rank={self.local_rank} | world_size={self.world_size} | device={self.device}",
        )

        if self.raw_module._get_device() != self.device:
            raise RuntimeError(
                f"Module is on {self.raw_module._get_device()}, but trainer device is {self.device}"
            )

        self.scaler = GradScaler(enabled=self.config.mixed_precision)

        self.train_aggregator = MetricsAggregator(distributed, name="Train")
        self.validation_aggregator = (
            MetricsAggregator(distributed, name="Validation")
            if self.ValidationLoader is not None
            else None
        )

        if not self.save_checkpoint:
            self.log_root(
                logging.warning,
                "Configured value of save_checkpoint is false, no checkpoints whatsoever will be saved!",
            )

        self.checkpoint_dir = Path(RuntimeContext.GLOBAL_CHECKPOINT_DIR)
        resuming = os.path.isfile(self.checkpoint_dir / "best.pt")

        if resuming:
            self.log_root(
                logging.INFO,
                f"Resuming training from \n {self.checkpoint_dir / 'best.pt'}.\n"
                "Warning: If all configurations don't match you will get RuntimeError!",
            )
            self._load_checkpoint(self.checkpoint_dir / "best.pt")
            if self._epochs_trained == self.max_epochs:
                self._skip_training = True
                self.log_root(
                    logging.INFO,
                    "maximum epochs already reached in the resumed model. No training will be done.",
                )
        else:
            if self.is_on_root and self.save_checkpoint:
                os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.plot_dir = Path(RuntimeContext.GLOBAL_FIGURES_DIR)
        if self.is_on_root:
            os.makedirs(self.plot_dir, exist_ok=True)

        if self.is_distributed:
            self.distributed.barrier()

        self._setup = True
        self.log_root(logging.INFO, "Trainer setup complete.")

    def train(self):
        """
        Run the model training loop.

        Each epoch performs training, distributed metric aggregation, optional
        validation, checkpointing, metric logging, and early-stopping evaluation.
        Outstanding accumulated gradients are handled after the final epoch when
        necessary.

        Raises
        ------
        RuntimeError
            If distributed trainer setup has not been completed.
        """
        if not self._setup:
            raise RuntimeError("Call setup_distributed() before predict().")

        self.log_root(logging.INFO, "Starting Training Loop...")
        self.start_time_train = time.time()
        clear_memory()
        self.optimizer.zero_grad(set_to_none=True)

        while self._epochs_trained < self.max_epochs:
            time_elapsed = self._train_on_epoch()
            train_logs = self.train_aggregator._dist_compute()
            self.train_aggregator.record_epoch(train_logs, time_elapsed=time_elapsed)

            if self.ValidationLoader is not None:
                self._validate_on_epoch()
                validation_logs = self.validation_aggregator._dist_compute()
                self.validation_aggregator.record_epoch(validation_logs)

                validation_loss = validation_logs["total_loss"]
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
                        name="best",
                        train_logs=train_logs,
                        validation_logs=None,
                    )

            if self.is_distributed:
                self.distributed.barrier()

            if self.is_on_root:
                if (self._epochs_trained) % self.log_every_n_epochs == 0:
                    self._log_epoch(
                        train_logs=train_logs,
                        validation_logs=validation_logs,
                    )
                    MetricsAggregator.plot(
                        [self.train_aggregator, self.validation_aggregator],
                        plot_dir=self.plot_dir,
                    )

            should_stop = self._should_stop_early()
            if self.is_distributed:
                stop_tensor = torch.tensor(int(should_stop), device=self.device)
                self.distributed.broadcast(stop_tensor, src=0)
                should_stop = bool(stop_tensor.item())

            if should_stop:
                self.log_root(
                    logging.INFO,
                    f"Early stopping triggered.  Best validation loss: {self._best_validation_loss:.6f}",
                )

                break

        if self.batch_step % self.config.gradient_accumulation_steps != 0:
            if not self._should_stop_early():
                self._optimizer_step()

                if self.ValidationLoader is not None:
                    self._validate_on_epoch()
                    validation_logs = self.validation_aggregator._dist_compute()
                    self.validation_aggregator.record_epoch(validation_logs, index=-1)

                    validation_loss = validation_logs["total_loss"]
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
                        name="best",
                        train_logs=train_logs,
                        validation_logs=None,
                    )

        time_elapsed = time.time() - self.start_time_train

        if self.is_on_root:
            if not self._skip_training:
                self._log_epoch(
                    train_logs=train_logs,
                    validation_logs=validation_logs,
                )
            MetricsAggregator.plot(
                [self.train_aggregator, self.validation_aggregator],
                plot_dir=self.plot_dir,
            )
        self.log_root(logging.INFO, f"Training finished in {time_elapsed:.2f}s")

    def _train_on_epoch(self):
        """
        Train the model for one epoch.

        The distributed sampler is updated for the current epoch and each training
        batch is processed using the appropriate gradient accumulation size.

        Returns
        -------
        float
            Elapsed training time for the epoch in seconds.
        """
        self.TrainLoader.set_epoch(self._epochs_trained)
        self.module.train()

        start_time = time.time()
        for batch_id, batch in tqdm(
            enumerate(self.TrainLoader),
            disable=not self.is_on_root,
            desc="Train",
        ):
            current_lr = self.optimizer.learning_rate
            current_accumulation_size = (
                self.remainder_batch_size
                if (
                    self.remainder_batch_size != 0
                    and batch_id >= self.num_batches - self.remainder_batch_size
                )
                else self.config.gradient_accumulation_steps
            )

            batch_loss_dict, kwargs = self._train_on_batch(
                batch, current_accumulation_size
            )

            self.train_aggregator.record(batch_loss_dict, current_lr, kwargs)

        self._epochs_trained += 1

        time_elapsed = time.time() - start_time

        return time_elapsed

    def _train_on_batch(self, batch, accumulation_size: int):
        """
        Train the model on one batch.

        The batch is moved to the trainer device, an optional beta value is resolved,
        the loss is evaluated under automatic mixed precision, and the scaled loss is
        backpropagated. An optimizer update is performed when the configured gradient
        accumulation interval is reached.

        Parameters
        ----------
        batch : Any
            Training batch to process.
        accumulation_size : int
            Number of batches over which the current gradients are accumulated.

        Returns
        -------
        tuple[dict, dict]
            Individual loss values and additional training values, such as the
            current beta coefficient, used for metric aggregation.
        """
        batch.to_device(self.device)
        kwargs = {}

        if hasattr(self, "beta_finder"):
            beta = self.beta_finder(self.global_step)
            kwargs = dict(beta=beta)

        with torch.cuda.amp.autocast(
            enabled=self.scaler.is_enabled() and self.device.type == "cuda"
        ):
            loss, loss_dict = self.raw_module._compute_loss(data=batch, **kwargs)
            loss = loss / accumulation_size

        self.scaler.scale(loss).backward()

        self.batch_step += 1

        if self.batch_step % self.config.gradient_accumulation_steps == 0:
            self._optimizer_step()

        return loss_dict, kwargs

    @torch.no_grad()
    def _validate_on_epoch(self):
        """
        Evaluate the model over one validation epoch.

        Validation losses are calculated without gradient tracking and recorded in
        the validation metric aggregator.

        Raises
        ------
        RuntimeError
            If validation is requested without a validation data loader.
        """
        if self.ValidationLoader is None:
            raise RuntimeError(
                "ValidationLoader is None, but validation was requested."
            )

        self.module.eval()

        for batch_id, batch in tqdm(
            enumerate(self.ValidationLoader),
            disable=not self.is_on_root,
            desc="Validate",
        ):
            batch_loss_dict = self._validate_on_batch(batch)

            self.validation_aggregator.record(batch_loss_dict)

    @torch.no_grad()
    def _validate_on_batch(self, batch):
        """
        Evaluate the model on one validation batch.

        Parameters
        ----------
        batch : Any
            Validation batch to process.

        Returns
        -------
        dict
            Individual validation loss values used for metric aggregation.
        """
        batch.to_device(self.device)
        kwargs = {}

        if hasattr(self, "beta_finder"):
            beta = self.beta_finder(self.global_step)
            kwargs = dict(beta=beta)

        with torch.cuda.amp.autocast(
            enabled=self.scaler.is_enabled() and self.device.type == "cuda"
        ):
            _, loss_dict = self.raw_module._compute_loss(data=batch, **kwargs)

        return loss_dict

    def _optimizer_step(self):
        """
        Perform a mixed-precision optimizer update.

        Gradients are optionally unscaled and clipped before the optimizer step. The
        learning-rate scheduler and global step advance only when the gradient-scaler
        update indicates that the optimizer step was not skipped. Gradients are reset
        afterward.
        """
        if self.config.grad_clip is not None:
            self.scaler.unscale_(self.optimizer.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.module.parameters(), self.config.grad_clip
            )

        old_scale = self.scaler.get_scale()
        self.scaler.step(self.optimizer.optimizer)
        self.scaler.update()
        new_scale = self.scaler.get_scale()

        step_was_skipped = new_scale < old_scale
        if not step_was_skipped:
            self.optimizer.scheduler_step()
            self.global_step += 1

        self.optimizer.zero_grad(set_to_none=True)

    def _is_improved(self, validation_loss: float | torch.Tensor) -> bool:
        """
        Determine whether validation loss has improved sufficiently.

        The required decrease is calculated as a fraction of the current best
        validation loss.

        Parameters
        ----------
        validation_loss : float | torch.Tensor
            Validation loss to compare with the current best value.

        Returns
        -------
        bool
            ``True`` if the validation loss satisfies the configured minimum
            improvement requirement.
        """
        if isinstance(validation_loss, torch.Tensor):
            validation_loss = validation_loss.item()

        if self._best_validation_loss == float("inf"):
            return True

        required_improvement = (
            self.config.minimum_validation_improvement_percentage
            * abs(self._best_validation_loss)
        )

        return validation_loss < self._best_validation_loss - required_improvement

    def _should_stop_early(self) -> bool:
        """
        Determine whether validation-based early stopping should occur.

        Returns
        -------
        bool
            ``True`` when the number of consecutive epochs without sufficient
            validation improvement reaches the configured early-stopping buffer.
        """
        if self.ValidationLoader is None:
            return False

        if self.config.earlystoppingbuffer is None:
            return False

        if self.config.earlystoppingbuffer == float("inf"):
            return False

        return self.earlystopping_counter >= self.config.earlystoppingbuffer

    def _log_epoch(self, train_logs, validation_logs=None):
        """
        Log a summary of the current training epoch.

        Parameters
        ----------
        train_logs : dict
            Aggregated training metrics for the epoch.
        validation_logs : dict | None
            Aggregated validation metrics for the epoch, if validation is enabled.
        """
        elapsed_time = time.time() - self.start_time_train
        msg = (
            f"Epoch {self._epochs_trained}/{self.max_epochs} | "
            f"train loss: {train_logs['total_loss']:.6f}"
        )

        if validation_logs is not None:
            msg += f" | validation loss: {validation_logs['total_loss']:.6f}"

        msg += (
            f" | global step: {self.global_step} | "
            f" early stopping counter: {self.earlystopping_counter} |"
            f" elapsed time: {elapsed_time:.2f}"
        )

        self.log_root(logging.INFO, msg)

    def _save_checkpoint(
        self, name: str, train_logs: dict, validation_logs: dict | None = None
    ):
        """
        Save the current training state to a checkpoint.

        The checkpoint includes model and optimizer state, mixed-precision scaler
        state, training progress, data shapes and metadata, module configuration, and
        training and validation metric histories.

        Parameters
        ----------
        name : str
            Base filename used for the checkpoint.
        train_logs : dict
            Aggregated training metrics associated with the checkpoint.
        validation_logs : dict | None
            Aggregated validation metrics associated with the checkpoint.
        """
        checkpoint = {
            "epoch": self._epochs_trained,
            "global_step": self.global_step,
            "batch_step": self.batch_step,
            "best_validation_loss": self._best_validation_loss,
            "earlystopping_counter": self.earlystopping_counter,
            "module": self.raw_module.state_dict(),
            "module_config": dataclasses.asdict(self.raw_module.config),
            "input_shape": self.TrainLoader.input_shape,
            "output_shape": self.TrainLoader.target_shape,
            "added_features_dim": self.TrainLoader.added_features_dim,
            "input_var_metadata": RuntimeContext.INPUT_VAR_METADATA,
            "output_var_metadata": RuntimeContext.TARGET_VAR_METADATA,
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "train_logs": train_logs,
            "validation_logs": validation_logs,
            "train_history": self.train_aggregator.state_dict(),
            "validation_history": self.validation_aggregator.state_dict()
            if self.validation_aggregator is not None
            else None,
        }

        path = Path(self.checkpoint_dir) / f"{name}.pt"
        torch.save(checkpoint, path)

    def _load_checkpoint(self, path: str | Path | None = None, strict: bool = True):
        """
        Restore training state from a checkpoint.

        Model, optimizer, scheduler, gradient-scaler, training progress,
        early-stopping state, and available metric histories are restored.

        Parameters
        ----------
        path : str | Path | None
            Path to the checkpoint file. If ``None``, ``best.pt`` in the configured
            checkpoint directory is used.
        strict : bool
            Whether checkpoint model parameters must exactly match the current module
            state dictionary.

        Returns
        -------
        dict
            Loaded checkpoint contents.

        Raises
        ------
        FileNotFoundError
            If the requested checkpoint file does not exist.
        """
        if path is None:
            path = Path(self.checkpoint_dir) / "best.pt"
        else:
            path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.raw_module.load_state_dict(checkpoint["module"], strict=strict)

        self.optimizer.load_state_dict(checkpoint["optimizer"])

        if "scaler" in checkpoint and checkpoint["scaler"] is not None:
            self.scaler.load_state_dict(checkpoint["scaler"])

        self._epochs_trained = checkpoint.get("epoch", 0)
        self._start_epoch = self._epochs_trained
        self.global_step = checkpoint.get("global_step", 0)
        self.batch_step = checkpoint.get("batch_step", 0)

        self._best_validation_loss = checkpoint.get("best_validation_loss", torch.inf)
        self.earlystopping_counter = checkpoint.get("earlystopping_counter", 0)

        if "train_history" in checkpoint:
            self.train_aggregator.load_state_dict(checkpoint["train_history"])

        if (
            "validation_history" in checkpoint
            and checkpoint["validation_history"] is not None
            and self.validation_aggregator is not None
        ):
            self.validation_aggregator.load_state_dict(checkpoint["validation_history"])

        self.log_root(
            logging.INFO,
            (
                f"Resumed at epoch={self._epochs_trained}, "
                f"global_step={self.global_step}, "
                f"scheduler_step={self.optimizer.lr_scheduler.num_steps}, "
                f"lr={self.optimizer.learning_rate}"
            ),
        )

        if self.is_distributed:
            self.distributed.barrier()

        return checkpoint

    @property
    def raw_module(self):
        """
        Restore training state from a checkpoint.

        Model, optimizer, scheduler, gradient-scaler, training progress,
        early-stopping state, and available metric histories are restored.

        Parameters
        ----------
        path : str | Path | None
            Path to the checkpoint file. If ``None``, ``best.pt`` in the configured
            checkpoint directory is used.
        strict : bool
            Whether checkpoint model parameters must exactly match the current module
            state dictionary.

        Returns
        -------
        dict
            Loaded checkpoint contents.

        Raises
        ------
        FileNotFoundError
            If the requested checkpoint file does not exist.
        """
        if isinstance(self.module, torch.nn.parallel.DistributedDataParallel):
            return self.module.module
        return self.module

    def log_root(self, level: int, msg: str, *args):
        """
        Log a message from the root process.

        Parameters
        ----------
        level : int
            Logging level used for the message.
        msg : str
            Message to log or print.
        args : Any
            Positional arguments passed to the configured logger.
        """
        if self.is_on_root:
            if self.logger is not None:
                self.logger.log(level, msg, *args)
            else:
                print(msg)


def clear_memory():
    """
    Release unused Python and CUDA memory.

    Runs Python garbage collection and, when CUDA is available, clears cached CUDA
    memory and collects unused inter-process CUDA allocations.
    """
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
