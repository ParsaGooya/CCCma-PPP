import torch
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import dataclasses
from typing import ClassVar

from cccma_ppp.core.core_abc import moduleABC

@dataclasses.dataclass
class LRSchedulerConfig:
    min_lr: float = 0.0
    warmup_epochs: int = 0
    total_epochs: int = None

    def __post_init__(self):
        assert self.min_lr >= 0
        assert self.warmup_epochs >= 0

    def build(
        self,
        optimizer: torch.optim.Optimizer,
        num_batches: int
    ):
        assert self.total_epochs is not None
        assert self.total_epochs > 0
        assert num_batches > 0
        assert self.warmup_epochs < self.total_epochs, 'number of warmup epochs must be smaller than total epochs.'

        self.total_steps = num_batches * self.total_epochs
        self.warmup_steps = num_batches * self.warmup_epochs


        return CosineAnnealingLRScheduler(self, optimizer )





@dataclasses.dataclass
class OptimizerConfig:

    lr : float = 0.0001
    weight_decay : float = 0
    optimizer_type : str  = 'adam'    # type[torch.optim.Optimizer] = torch.optim.Adam
    lr_scheduler_config : LRSchedulerConfig | None = None # = dataclasses.field(default_factory=CosineAnnealingLRScheduler)

    OPTIMIZER_REGISTERY : ClassVar[dict] = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW}


    def __post_init__(self):
        assert self.weight_decay >= 0, 'weight_decay has to be postive'
        self.optimizer = None

    def build(self,
            module: moduleABC,
            num_batches: int = None,
            max_epochs: int = None):

        assert module.built, 'make sure module is built before optmizer.'

        if self.lr_scheduler_config is not None:

            if self.lr_scheduler_config.total_epochs is None:
                if max_epochs is None:
                    raise ValueError('either max_epochs must be specified or total_epochs in the learning rate scheduler configuration.')

                self.lr_scheduler_config.total_epochs = max_epochs
            if num_batches is None:
                raise ValueError('num_batches must be specified to set up learning rate scheduler.')

        return OptimizerWrapper(self, module, num_batches, max_epochs)  ## max_epochs will be used if total_epochs is not specified.




class OptimizerWrapper:

    def __init__(self,
                config : OptimizerConfig,
                module: moduleABC,
                num_batches : int = None,
                max_epochs: int = None):


        params = [p for p in module.parameters() if p.requires_grad]

        self.optimizer = config.OPTIMIZER_REGISTERY.get(config.optimizer_type.lower())
        self.optimizer = self.optimizer(params, lr = config.lr, weight_decay = config.weight_decay)

        if config.lr_scheduler_config is not None:
            if num_batches is None: 
                raise ValueError('num_batches must be specified to set up learning rate scheduler.')
            
            if all([max_epochs is None, config.lr_scheduler_config.total_epochs is None]):
                raise ValueError('either max_epochs must be specified or total_epochs in the learning rate scheduler configuration.')

            self.lr_scheduler = config.lr_scheduler_config.build(self.optimizer,
                                                 num_batches)

    def step(self):
        if self.optimizer is None:
            raise RuntimeError("Optimizer must be built before step().")

        self.optimizer.step()

    def scheduler_step(self):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def zero_grad(self, set_to_none = True, **kwargs):
        if self.optimizer is None:
            raise RuntimeError("Optimizer must be built before zero_grad().")
        self.optimizer.zero_grad(set_to_none=set_to_none, **kwargs)

    def state_dict(self):
        if self.optimizer is None:
            raise RuntimeError("Optimizer must be built before state_dict().")

        return {
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": (
                self.lr_scheduler.state_dict()
                if self.lr_scheduler is not None
                else None )}



    def load_state_dict(self, state_dict):

        if self.optimizer is None:
            raise RuntimeError("Optimizer must be built before load_state_dict().")

        self.optimizer.load_state_dict(state_dict["optimizer"])

        if (self.lr_scheduler is not None and state_dict["lr_scheduler"] is not None):

            self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])





class CosineAnnealingLRScheduler:

    def __init__(self,
        config : LRSchedulerConfig,
        optimizer: torch.optim.Optimizer):

        self.config = config

        if config.warmup_steps > 0:

            warmup_scheduler = LinearLR(optimizer, start_factor=1e-4,  end_factor=1.0,  total_iters= config.warmup_steps )
            cosine_scheduler = CosineAnnealingLR( optimizer,  T_max= config.total_steps - config.warmup_steps,  eta_min= config.min_lr)

            self.scheduler = SequentialLR( optimizer, schedulers=[warmup_scheduler, cosine_scheduler],milestones=[config.warmup_steps],  )

        else:

            self.scheduler = CosineAnnealingLR( optimizer, T_max= config.total_steps,  eta_min= config.min_lr )

    def step(self):

        if not hasattr(self, "scheduler"):
            raise RuntimeError("Scheduler must be built before stepping.")
        self.scheduler.step()

    def state_dict(self):
        if not hasattr(self, "scheduler"):
            raise RuntimeError("Scheduler must be built before stepping.")

        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        if not hasattr(self, "scheduler"):
            raise RuntimeError("Scheduler must be built before stepping.")

        self.scheduler.load_state_dict(state_dict)