import torch
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import dataclasses
from typing import ClassVar
import math
from cccma_ppp.core import moduleABC


@dataclasses.dataclass
class LRSchedulerConfig:
    """
    Configuration for learning rate scheduler.

    Parameters
    ----------
    min_lr : float, optional
        Minimum learning rate for cosine annealing.
    warmup_epochs : int, optional
        Number of warmup epochs with linearly increasing learning rate.
    total_epochs : int or None, optional
        Total number of training epochs.
    """

    min_lr: float = 0.0
    warmup_epochs: int = 0
    total_epochs: int = None

    def __post_init__(self):
        """
        Validate scheduler configuration parameters.

        Raises
        ------
        AssertionError
            If `min_lr` or `warmup_epochs` are negative.
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
        Construct learning rate scheduler.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer instance.
        num_batches : int
            Number of batches per epoch.

        Returns
        -------
        CosineAnnealingLRScheduler
            Configured scheduler.

        Raises
        ------
        AssertionError
            If required configuration values are invalid.
        ValueError
            If warmup exceeds total epochs.
        """

        assert self.total_epochs is not None
        assert self.total_epochs > 0
        assert num_batches > 0
        if self.warmup_epochs >= self.total_epochs:
            raise ValueError(
                "number of warmup epochs must be smaller than total epochs."
            )

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
    Configuration for optimizer and optional learning rate scheduler.

    Parameters
    ----------
    lr : float, optional
        Learning rate.
    weight_decay : float, optional
        Weight decay coefficient.
    optimizer_type : str, optional
        Optimizer type ("adam", "adamw").
    lr_scheduler_config : LRSchedulerConfig or None, optional
        Learning rate scheduler configuration.
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
        Validate optimizer configuration.

        Raises
        ------
        ValueError
            If weight_decay is negative.
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
        Construct optimizer wrapper.

        Parameters
        ----------
        module : ModuleABC
            Model whose parameters are optimized.
        num_batches : int, optional
            Number of batches per epoch.
        max_epochs : int, optional
            Total number of training epochs.

        Returns
        -------
        OptimizerWrapper
            Wrapped optimizer and scheduler.

        Raises
        ------
        ValueError
            If scheduler configuration is incomplete.
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
    Wrapper around optimizer and optional learning rate scheduler.

    Parameters
    ----------
    config : OptimizerConfig
        Optimizer configuration.
    module : moduleABC
        Model instance.
    num_batches : int or None
        Number of batches per epoch.
    max_epochs : int or None
        Total number of training epochs.
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
        Initialize optimizer and optional scheduler.

        Parameters
        ----------
        config : OptimizerConfig
        module : ModuleABC
        num_batches : int or None
        max_epochs : int or None

        Raises
        ------
        ValueError
            If scheduler configuration requirements are not met.
        """

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
        return self.optimizer.param_groups[0]["lr"]

    def step(self):
        """
        Perform optimizer step.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If optimizer has not been initialized.
        """

        if self.optimizer is None:
            raise RuntimeError("Optimizer must be built before step().")

        self.optimizer.step()

    def scheduler_step(self):
        """
        Perform scheduler step.

        Returns
        -------
        None
        """

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def zero_grad(self, set_to_none=True, **kwargs):
        """
        Reset gradients of model parameters.

        Parameters
        ----------
        set_to_none : bool, optional
            Whether to set gradients to None instead of zero.
        **kwargs
            Additional arguments passed to optimizer.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If optimizer has not been initialized.
        """

        if self.optimizer is None:
            raise RuntimeError("Optimizer must be built before zero_grad().")
        self.optimizer.zero_grad(set_to_none=set_to_none, **kwargs)

    def state_dict(self):
        """
        Return optimizer and scheduler state.

        Returns
        -------
        dict
            State dictionary containing optimizer and scheduler states.

        Raises
        ------
        RuntimeError
            If optimizer has not been initialized.
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
        Load optimizer and scheduler state.

        Parameters
        ----------
        state_dict : dict
            State dictionary.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If optimizer is not initialized.
        """

        if self.optimizer is None:
            raise RuntimeError("Optimizer must be built before load_state_dict().")

        self.optimizer.load_state_dict(state_dict["optimizer"])

        if self.lr_scheduler is not None and state_dict["lr_scheduler"] is not None:
            self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])


class CosineAnnealingLRScheduler:
    """
    Learning rate scheduler with optional warmup and cosine annealing.

    Parameters
    ----------
    config : LRSchedulerConfig
        Scheduler configuration.
    optimizer : torch.optim.Optimizer
        Optimizer instance.
    """

    def __init__(self, config: LRSchedulerConfig, optimizer: torch.optim.Optimizer):
        """
        Initialize scheduler with optional warmup phase.

        Parameters
        ----------
        config : LRSchedulerConfig
        optimizer : torch.optim.Optimizer

        Returns
        -------
        None
        """

        self.config = config

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
        Advance scheduler by one step.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If scheduler has not been initialized.
        """

        if not hasattr(self, "scheduler"):
            raise RuntimeError("Scheduler must be built before stepping.")
        self.scheduler.step()

    def state_dict(self):
        """
        Return scheduler state.

        Returns
        -------
        dict
            Scheduler state dictionary.

        Raises
        ------
        RuntimeError
            If scheduler has not been initialized.
        """

        if not hasattr(self, "scheduler"):
            raise RuntimeError("Scheduler must be built before stepping.")

        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        """
        Load scheduler state.

        Parameters
        ----------
        state_dict : dict
            Scheduler state dictionary.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If scheduler has not been initialized.
        """

        if not hasattr(self, "scheduler"):
            raise RuntimeError("Scheduler must be built before stepping.")

        self.scheduler.load_state_dict(state_dict)
