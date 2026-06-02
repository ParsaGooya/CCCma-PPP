import torch
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import dataclasses
from typing import ClassVar

from cccma_ppp.core.core_abc import moduleABC


@dataclasses.dataclass
class LRSchedulerConfig:
    """
    Configuration for constructing learning rate schedulers.
    """

    min_lr: float = 0.0
    warmup_epochs: int = 0

    def __post_init__(self):
        """
        Validate scheduler configuration parameters.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If configuration values are invalid.
        """

        assert self.min_lr >= 0
        assert self.warmup_epochs >= 0

    def build(self, optimizer, num_batches, total_epochs):
        """
        Build a cosine annealing scheduler with optional warmup.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer to attach scheduler to.
        num_batches : int
            Number of batches per epoch.
        total_epochs : int
            Total number of training epochs.

        Returns
        -------
        CosineAnnealingLRScheduler
            Configured learning rate scheduler.

        Raises
        ------
        AssertionError
            If input arguments are invalid.
        """

        assert total_epochs > 0
        assert num_batches > 0
        assert self.warmup_epochs < total_epochs, (
            "number of warmup epochs must be smaller than total epochs."
        )

        self.total_steps = num_batches * total_epochs
        self.warmup_steps = num_batches * self.warmup_epochs

        return CosineAnnealingLRScheduler(self, optimizer)


@dataclasses.dataclass
class OptimizerConfig:
    """
    Configuration for optimizer and optional learning rate scheduler.
    """

    lr: float = 0.0001
    weight_decay: float = 0
    optimizer_type: str = "adam"  # type[torch.optim.Optimizer] = torch.optim.Adam
    lr_scheduler_config: LRSchedulerConfig | None = (
        None  # = dataclasses.field(default_factory=CosineAnnealingLRScheduler)
    )

    OPTIMIZER_REGISTERY: ClassVar[dict] = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }

    def __post_init__(self):
        """
        Validate optimizer configuration parameters.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If configuration values are invalid.
        """

        assert self.weight_decay >= 0, "weight_decay has to be postive"
        self.optimizer = None

    def build(self, module, num_batches=None, total_epochs=None):
        """
        Build optimizer wrapper for a module.

        Parameters
        ----------
        module : moduleABC
            Module containing trainable parameters.
        num_batches : int, optional
            Number of batches per epoch.
        total_epochs : int, optional
            Total number of training epochs.

        Returns
        -------
        OptimizerWrapper
            Wrapped optimizer instance.

        Raises
        ------
        AssertionError
            If module is not built.
        ValueError
            If scheduler configuration is incomplete.
        """

        assert module.built, "make sure module is built before optmizer."

        if self.lr_scheduler_config is not None:
            if any([num_batches is None, total_epochs is None]):
                raise ValueError(
                    "total_epochs and num_batches must be specified to set up learning rate scheduler."
                )

        return OptimizerWrapper(self, module, num_batches, total_epochs)


class OptimizerWrapper:
    """
    Wrapper for optimizer and optional learning rate scheduler.
    """

    def __init__(self, config, module, num_batches=None, total_epochs=None):
        """
        Initialize optimizer and optional scheduler.

        Parameters
        ----------
        config : OptimizerConfig
            Optimizer configuration.
        module : moduleABC
            Module providing parameters.
        num_batches : int, optional
            Number of batches per epoch.
        total_epochs : int, optional
            Total number of epochs.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If scheduler configuration is incomplete.
        """

        params = [p for p in module.parameters() if p.requires_grad]

        self.optimizer = config.OPTIMIZER_REGISTERY.get(config.optimizer_type.lower())
        self.optimizer = self.optimizer(
            params, lr=config.lr, weight_decay=config.weight_decay
        )

        if config.lr_scheduler_config is not None:
            if any([num_batches is None, total_epochs is None]):
                raise ValueError(
                    "total_epochs and num_batches must be specified to set up learning rate scheduler."
                )

            self.lr_scheduler = config.lr_scheduler_config.build(
                self.optimizer, num_batches, total_epochs
            )

    def step(self):
        """
        Perform an optimization step.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If optimizer is not initialized.
        """

        if self.optimizer is None:
            raise RuntimeError("Optimizer must be built before step().")

        self.optimizer.step()

    def scheduler_step(self):
        """
        Step the learning rate scheduler.

        Returns
        -------
        None
        """

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def zero_grad(self, set_to_none=True, **kwargs):
        """
        Reset gradients of all parameters.

        Parameters
        ----------
        set_to_none : bool, optional
            Whether to set gradients to None.
        **kwargs
            Additional arguments.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If optimizer is not initialized.
        """

        if self.optimizer is None:
            raise RuntimeError("Optimizer must be built before zero_grad().")
        self.optimizer.zero_grad(set_to_none=set_to_none, **kwargs)

    def state_dict(self):
        """
        Retrieve optimizer and scheduler state.

        Returns
        -------
        dict
            State dictionary.

        Raises
        ------
        RuntimeError
            If optimizer is not initialized.
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
    Cosine annealing learning rate scheduler with optional warmup phase.
    """

    def __init__(self, config, optimizer):
        """
        Initialize scheduler configuration.

        Parameters
        ----------
        config : LRSchedulerConfig
            Scheduler configuration.
        optimizer : torch.optim.Optimizer
            Optimizer instance.

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
            If scheduler is not initialized.
        """

        if not hasattr(self, "scheduler"):
            raise RuntimeError("Scheduler must be built before stepping.")
        self.scheduler.step()

    def state_dict(self):
        """
        Retrieve scheduler state.

        Returns
        -------
        dict
            Scheduler state dictionary.

        Raises
        ------
        RuntimeError
            If scheduler is not initialized.
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
            If scheduler is not initialized.
        """

        if not hasattr(self, "scheduler"):
            raise RuntimeError("Scheduler must be built before stepping.")

        self.scheduler.load_state_dict(state_dict)
