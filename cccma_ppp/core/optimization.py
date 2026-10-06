import torch
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import dataclasses
from typing import ClassVar
import math
from cccma_ppp.core.core_abc import moduleABC


@dataclasses.dataclass
class LRSchedulerConfig:
    """
    Configuration for a cosine annealing learning-rate scheduler.

    Parameters
    ----------
    min_lr : float
        Minimum learning rate reached by cosine annealing.
    warmup_epochs : int
        Number of initial epochs used for linear learning-rate warmup.
    total_epochs : int | None
        Total number of training epochs over which the learning-rate schedule is
        applied.
    hold_min_lr : bool
        Whether to keep the learning rate at ``min_lr`` after the configured
        schedule has completed.
    """

    min_lr: float = 0.0
    warmup_epochs: int = 0
    total_epochs: int = None
    hold_min_lr: bool = True

    def __post_init__(self):
        """
        Validate the learning-rate scheduler configuration.

        Raises
        ------
        AssertionError
            If ``min_lr`` or ``warmup_epochs`` is negative.
        """
        assert self.min_lr >= 0
        assert self.warmup_epochs >= 0

    def build(
        self,
        optimizer: torch.optim.Optimizer,
        num_batches: int,
        gradient_accumulation_steps: int = 1,
    ):
        """
        Build a cosine annealing learning-rate scheduler.

        The number of scheduler steps is computed from the number of batches,
        gradient accumulation factor, and configured number of training epochs.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer whose learning rate is controlled by the scheduler.
        num_batches : int
            Number of batches processed during each training epoch.
        gradient_accumulation_steps : int
            Number of batches accumulated before each optimizer update.

        Returns
        -------
        CosineAnnealingLRScheduler
            Configured learning-rate scheduler.

        Raises
        ------
        ValueError
            If ``total_epochs``, ``num_batches``, or
            ``gradient_accumulation_steps`` is nonpositive, or if
            ``warmup_epochs`` is greater than or equal to ``total_epochs``.
        """
        if self.total_epochs is None or self.total_epochs <= 0:
            raise ValueError("total_epochs must be positive.")

        if num_batches <= 0:
            raise ValueError("num_batches must be positive.")

        if gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive.")

        if self.warmup_epochs >= self.total_epochs:
            raise ValueError("warmup_epochs must be smaller than total_epochs.")

        self.total_steps = (
            math.ceil(num_batches / gradient_accumulation_steps) * self.total_epochs
        )
        self.warmup_steps = (
            math.ceil(num_batches / gradient_accumulation_steps) * self.warmup_epochs
        )

        return CosineAnnealingLRScheduler(self, optimizer)


@dataclasses.dataclass
class OptimizerConfig:
    """
    Configuration for an optimizer and optional learning-rate scheduler.

    Parameters
    ----------
    lr : float
        Initial learning rate used by the optimizer.
    weight_decay : float
        Weight-decay coefficient applied by the optimizer.
    optimizer_type : str
        Name of the optimizer implementation to use.
    lr_scheduler_config : LRSchedulerConfig | None
        Optional learning-rate scheduler configuration.
    """

    lr: float = 0.0001
    weight_decay: float = 0
    optimizer_type: str = "adam"
    lr_scheduler_config: LRSchedulerConfig | None = None

    OPTIMIZER_REGISTERY: ClassVar[dict] = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }

    def __post_init__(self):
        """
        Validate the optimizer configuration and initialize its runtime state.

        Raises
        ------
        ValueError
            If ``weight_decay`` is negative.
        """
        if self.weight_decay < 0:
            raise ValueError("weight_decay has to be positive")
        self.optimizer = None

    def build(
        self,
        module: moduleABC,
        num_batches: int = None,
        max_epochs: int = None,
        gradient_accumulation_steps: int = 1,
    ):
        """
        Build an optimizer wrapper from this configuration.

        Parameters
        ----------
        module : moduleABC
            Module whose trainable parameters are optimized.
        num_batches : int | None
            Number of batches processed during each training epoch. Required when a
            learning-rate scheduler is configured.
        max_epochs : int | None
            Maximum number of training epochs. Used as the scheduler duration when
            ``total_epochs`` is not explicitly configured.
        gradient_accumulation_steps : int
            Number of batches accumulated before each optimizer update.

        Returns
        -------
        OptimizerWrapper
            Wrapper containing the configured optimizer and optional learning-rate
            scheduler.

        Raises
        ------
        ValueError
            If a learning-rate scheduler requires ``max_epochs`` or ``num_batches``
            and the required value is not provided.
        """
        if self.lr_scheduler_config is not None:
            if self.lr_scheduler_config.total_epochs is None:
                if max_epochs is None:
                    raise ValueError(
                        "either max_epochs must be specified or total_epochs in the learning rate scheduler configuration."
                    )

                self.lr_scheduler_config.total_epochs = max_epochs
            if num_batches is None:
                raise ValueError(
                    "num_batches must be specified to set up learning rate scheduler."
                )

        return OptimizerWrapper(
            self, module, num_batches, max_epochs, gradient_accumulation_steps
        )


class OptimizerWrapper:
    """
    Wrap an optimizer and optional learning-rate scheduler.

    Only parameters requiring gradients are passed to the optimizer.

    Parameters
    ----------
    config : OptimizerConfig
        Configuration defining the optimizer and optional scheduler.
    module : moduleABC
        Module whose trainable parameters are optimized.
    num_batches : int | None
        Number of batches processed during each training epoch.
    max_epochs : int | None
        Maximum number of training epochs.
    gradient_accumulation_steps : int
        Number of batches accumulated before each optimizer update.
    """

    def __init__(
        self,
        config: OptimizerConfig,
        module: moduleABC,
        num_batches: int = None,
        max_epochs: int = None,
        gradient_accumulation_steps: int = 1,
    ):
        """
        Initialize an optimizer and optional learning-rate scheduler.

        Parameters
        ----------
        config : OptimizerConfig
            Configuration defining the optimizer and optional scheduler.
        module : moduleABC
            Module whose trainable parameters are optimized.
        num_batches : int | None
            Number of batches processed during each training epoch.
        max_epochs : int | None
            Maximum number of training epochs.
        gradient_accumulation_steps : int
            Number of batches accumulated before each optimizer update.

        Raises
        ------
        ValueError
            If scheduler configuration requires ``num_batches`` or an epoch count and
            the required value is not provided.
        """
        self.lr_scheduler = None
        params = [p for p in module.parameters() if p.requires_grad]

        self.optimizer = config.OPTIMIZER_REGISTERY.get(config.optimizer_type.lower())
        self.optimizer = self.optimizer(
            params, lr=config.lr, weight_decay=config.weight_decay
        )

        if config.lr_scheduler_config is not None:
            if num_batches is None:
                raise ValueError(
                    "num_batches must be specified to set up learning rate scheduler."
                )

            if all(
                [max_epochs is None, config.lr_scheduler_config.total_epochs is None]
            ):
                raise ValueError(
                    "either max_epochs must be specified or total_epochs in the learning rate scheduler configuration."
                )

            self.lr_scheduler = config.lr_scheduler_config.build(
                self.optimizer, num_batches, gradient_accumulation_steps
            )

    @property
    def learning_rate(self):
        """
        Return the current learning rate.

        Returns
        -------
        float
            Learning rate of the first optimizer parameter group.
        """
        return self.optimizer.param_groups[0]["lr"]

    def step(self):
        """
        Perform an optimizer update.

        Raises
        ------
        RuntimeError
            If the optimizer has not been built.
        """
        if self.optimizer is None:
            raise RuntimeError("Optimizer must be built before step().")

        self.optimizer.step()

    def scheduler_step(self):
        """
        Advance the learning-rate scheduler by one step.

        If no learning-rate scheduler is configured, this method has no effect.
        """
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def zero_grad(self, set_to_none=True, **kwargs):
        """
        Reset gradients for optimized parameters.

        Parameters
        ----------
        set_to_none : bool
            Whether gradients are set to ``None`` instead of being reset to zero.
        **kwargs : Any
            Additional keyword arguments passed to the optimizer's ``zero_grad``
            method.

        Raises
        ------
        RuntimeError
            If the optimizer has not been built.
        """
        if self.optimizer is None:
            raise RuntimeError("Optimizer must be built before zero_grad().")
        self.optimizer.zero_grad(set_to_none=set_to_none, **kwargs)

    def state_dict(self):
        """
        Return the optimizer and scheduler state.

        Returns
        -------
        dict
            State dictionaries for the optimizer and, when configured, the
            learning-rate scheduler.

        Raises
        ------
        RuntimeError
            If the optimizer has not been built.
        """
        if self.optimizer is None:
            raise RuntimeError("Optimizer must be built before state_dict().")

        return {
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": (
                self.lr_scheduler.state_dict()
                if self.lr_scheduler is not None
                else None
            ),
        }

    def load_state_dict(self, state_dict):
        """
        Restore optimizer and scheduler state.

        Parameters
        ----------
        state_dict : dict
            State containing the optimizer state and optional learning-rate scheduler
            state.

        Raises
        ------
        RuntimeError
            If the optimizer has not been built.
        """
        if self.optimizer is None:
            raise RuntimeError("Optimizer must be built before load_state_dict().")

        self.optimizer.load_state_dict(state_dict["optimizer"])

        if self.lr_scheduler is not None and state_dict["lr_scheduler"] is not None:
            self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])


class CosineAnnealingLRScheduler:
    """
    Manage warmup and cosine annealing of an optimizer learning rate.

    An optional linear warmup is followed by cosine annealing toward the
    configured minimum learning rate.

    Parameters
    ----------
    config : LRSchedulerConfig
        Configuration defining warmup, annealing duration, minimum learning rate,
        and post-schedule behavior.
    optimizer : torch.optim.Optimizer
        Optimizer whose learning rate is controlled.
    """

    def __init__(self, config: LRSchedulerConfig, optimizer: torch.optim.Optimizer):
        """
        Initialize the warmup and cosine annealing schedule.

        Parameters
        ----------
        config : LRSchedulerConfig
            Configuration containing the computed warmup and total step counts.
        optimizer : torch.optim.Optimizer
            Optimizer whose learning rate is controlled.
        """
        self.config = config
        self.optimizer = optimizer
        self.num_steps = 0

        if config.warmup_steps > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=1e-4,
                end_factor=1.0,
                total_iters=config.warmup_steps,
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=config.total_steps - config.warmup_steps,
                eta_min=config.min_lr,
            )

            self.scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[config.warmup_steps],
            )

        else:
            self.scheduler = CosineAnnealingLR(
                optimizer, T_max=config.total_steps, eta_min=config.min_lr
            )

    def step(self):
        """
        Advance the learning-rate schedule by one step.

        After the configured schedule is complete, the learning rate is held at
        ``min_lr`` when ``hold_min_lr`` is enabled. Otherwise, the underlying
        scheduler continues to advance.

        Raises
        ------
        RuntimeError
            If the underlying scheduler has not been built.
        """
        if not hasattr(self, "scheduler"):
            raise RuntimeError("Scheduler must be built before stepping.")

        if self.num_steps < self.config.total_steps:
            self.scheduler.step()
            self.num_steps += 1
            return

        if self.config.hold_min_lr:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.config.min_lr
        else:
            self.scheduler.step()
            self.num_steps += 1

    def state_dict(self):
        """
        Return the current learning-rate scheduler state.

        Returns
        -------
        dict
            State of the underlying scheduler together with the number of scheduler
            steps performed.

        Raises
        ------
        RuntimeError
            If the underlying scheduler has not been built.
        """
        if not hasattr(self, "scheduler"):
            raise RuntimeError("Scheduler must be built before stepping.")

        return {
            "scheduler": self.scheduler.state_dict(),
            "num_steps": self.num_steps,
        }

    def load_state_dict(self, state_dict):
        """
        Restore the learning-rate scheduler state.

        Supports both the wrapper state containing ``scheduler`` and ``num_steps``
        and a state dictionary belonging directly to the underlying scheduler.

        Parameters
        ----------
        state_dict : dict
            Scheduler state to restore.

        Raises
        ------
        RuntimeError
            If the underlying scheduler has not been built.
        """
        if not hasattr(self, "scheduler"):
            raise RuntimeError("Scheduler must be built before stepping.")

        if "scheduler" in state_dict:
            self.scheduler.load_state_dict(state_dict["scheduler"])
            self.num_steps = state_dict["num_steps"]

        else:
            self.scheduler.load_state_dict(state_dict)
            self.num_steps = self.scheduler.last_epoch
