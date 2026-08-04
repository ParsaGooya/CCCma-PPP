import torch
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import dataclasses
from typing import ClassVar
import math
from cccma_ppp.core.core_abc import moduleABC


@dataclasses.dataclass
class LRSchedulerConfig:
    """
    Document this class.

    Parameters
    ----------
    min_lr : float
        Description not yet provided.
    warmup_epochs : int
        Description not yet provided.
    total_epochs : int
        Description not yet provided.
    """

    min_lr: float = 0.0
    warmup_epochs: int = 0
    total_epochs: int = None

    def __post_init__(self):
        """
        Document this function.

        Raises
        ------
        AssertionError
            Description not yet provided.
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
        Document this function.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Description not yet provided.
        num_batches : int
            Description not yet provided.
        gradient_accumulation_steps : int
            Description not yet provided.

        Returns
        -------
        Any
            Description not yet provided.

        Raises
        ------
        AssertionError
            Description not yet provided.
        ValueError
            Description not yet provided.
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
    Document this class.

    Parameters
    ----------
    lr : float
        Description not yet provided.
    weight_decay : float
        Description not yet provided.
    optimizer_type : str
        Description not yet provided.
    lr_scheduler_config : LRSchedulerConfig | None
        Description not yet provided.
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
        Document this function.

        Raises
        ------
        ValueError
            Description not yet provided.
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
        Document this function.

        Parameters
        ----------
        module : moduleABC
            Description not yet provided.
        num_batches : int
            Description not yet provided.
        max_epochs : int
            Description not yet provided.
        gradient_accumulation_steps : int
            Description not yet provided.

        Returns
        -------
        Any
            Description not yet provided.

        Raises
        ------
        ValueError
            Description not yet provided.
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
    Document this class.

    Parameters
    ----------
    config : OptimizerConfig
        Description not yet provided.
    module : moduleABC
        Description not yet provided.
    num_batches : int
        Description not yet provided.
    max_epochs : int
        Description not yet provided.
    gradient_accumulation_steps : int
        Description not yet provided.
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
        Document this function.

        Parameters
        ----------
        config : OptimizerConfig
            Description not yet provided.
        module : moduleABC
            Description not yet provided.
        num_batches : int
            Description not yet provided.
        max_epochs : int
            Description not yet provided.
        gradient_accumulation_steps : int
            Description not yet provided.

        Raises
        ------
        ValueError
            Description not yet provided.
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
        """
        Document this function.

        Returns
        -------
        Any
            Description not yet provided.
        """
        return self.optimizer.param_groups[0]["lr"]

    def step(self):
        """
        Document this function.

        Raises
        ------
        RuntimeError
            Description not yet provided.
        """
        if self.optimizer is None:
            raise RuntimeError("Optimizer must be built before step().")

        self.optimizer.step()

    def scheduler_step(self):
        """
        Document this function.
        """
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def zero_grad(self, set_to_none=True, **kwargs):
        """
        Document this function.

        Parameters
        ----------
        set_to_none : Any
            Description not yet provided.
        **kwargs : Any
            Description not yet provided.

        Raises
        ------
        RuntimeError
            Description not yet provided.
        """
        if self.optimizer is None:
            raise RuntimeError("Optimizer must be built before zero_grad().")
        self.optimizer.zero_grad(set_to_none=set_to_none, **kwargs)

    def state_dict(self):
        """
        Document this function.

        Returns
        -------
        Any
            Description not yet provided.

        Raises
        ------
        RuntimeError
            Description not yet provided.
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
        Document this function.

        Parameters
        ----------
        state_dict : Any
            Description not yet provided.

        Raises
        ------
        RuntimeError
            Description not yet provided.
        """
        if self.optimizer is None:
            raise RuntimeError("Optimizer must be built before load_state_dict().")

        self.optimizer.load_state_dict(state_dict["optimizer"])

        if self.lr_scheduler is not None and state_dict["lr_scheduler"] is not None:
            self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])


class CosineAnnealingLRScheduler:
    """
    Document this class.

    Parameters
    ----------
    config : LRSchedulerConfig
        Description not yet provided.
    optimizer : torch.optim.Optimizer
        Description not yet provided.
    """

    def __init__(self, config: LRSchedulerConfig, optimizer: torch.optim.Optimizer):
        """
        Document this function.

        Parameters
        ----------
        config : LRSchedulerConfig
            Description not yet provided.
        optimizer : torch.optim.Optimizer
            Description not yet provided.
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
        Document this function.

        Raises
        ------
        RuntimeError
            Description not yet provided.
        """
        if not hasattr(self, "scheduler"):
            raise RuntimeError("Scheduler must be built before stepping.")
        self.scheduler.step()

    def state_dict(self):
        """
        Document this function.

        Returns
        -------
        Any
            Description not yet provided.

        Raises
        ------
        RuntimeError
            Description not yet provided.
        """
        if not hasattr(self, "scheduler"):
            raise RuntimeError("Scheduler must be built before stepping.")

        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        """
        Document this function.

        Parameters
        ----------
        state_dict : Any
            Description not yet provided.

        Raises
        ------
        RuntimeError
            Description not yet provided.
        """
        if not hasattr(self, "scheduler"):
            raise RuntimeError("Scheduler must be built before stepping.")

        self.scheduler.load_state_dict(state_dict)
