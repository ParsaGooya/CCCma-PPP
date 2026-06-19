import torch.nn as nn
import xarray as xr
from typing import ClassVar
import dataclasses

from cccma_ppp.loss import Registery, Reduction

@dataclasses.dataclass
class LossStepConfig:
    name: str
    args: dict[str, object] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class LosspipelineConfig:
    loss_pipeline: list[LossStepConfig]
    loss_weights: list[float] = None
    reduction: Reduction = "mean"

    def __post_init__(self):
        if not len(self.loss_pipeline) >= 1:
            raise ValueError("provide at least one loss term.")

        self.loss_types: set[str] = set()

        for loss in self.loss_pipeline:
            if len(loss.args) > 0:
                if {"reduction", "weights", "num_output_dimensions"}.intersection(
                    list(loss.args.keys())
                ):
                    raise ValueError(
                        "do not specify reduction, weights, or num_output_dimensions for specific loss terms manually. Set them for the parent LosspipelineConfig."
                    )

            self.loss_types.add(loss.name)

        if isinstance(self.loss_weights, list):
            if not len(self.loss_weights) == len(self.loss_pipeline):
                raise ValueError("Provide a weight for each loss term.")
            if not sum(self.loss_weights) == 1:
                raise ValueError("Sum of loss term weights should be 1.")
        else:
            self.loss_weights = [
                1 / len(self.loss_pipeline) for _ in self.loss_pipeline
            ]

    def build(self, weights: xr.DataArray, num_output_dimensions: int = 2):

        return Losspipeline(self, weights, num_output_dimensions)


class Losspipeline(nn.Module):
    registery: ClassVar[Registery] = Registery()

    def __init__(
        self,
        config: LosspipelineConfig,
        weights: xr.DataArray,
        num_output_dimensions: int = 2,
    ):

        super().__init__()
        self._checked_dimensionality = False
        self.config = config
        self.reduction = config.reduction
        self.weights = weights
        self.num_output_dimensions = num_output_dimensions
        self.pipeline = []
        self.steps = []

        for step in self.config.loss_pipeline:
            name = step.name
            args = step.args
            args["num_output_dimensions"] = self.num_output_dimensions
            args["weights"] = self.weights
            args["reduction"] = self.reduction

            self.pipeline.append(self.registery.get(name.lower(), args))

            if "low_ress_kernel_size" in args.keys():
                name = f"{name}_low_ress_{args.get('low_ress_kernel_size')}"

            if name in self.steps:
                name = f"{name}_{self.steps.count(name) + 1}"

            self.steps.append(name)

    @classmethod
    def register(cls, name: str):
        return cls.registery.register(name.lower())

    def forward(
        self,
        data,
        target,
        target_mask=None,
        print_loss=False,
        step_arguments: dict = None,
    ):
        total_loss = None
        indiv_loses = {}

        if step_arguments is None:
            step_arguments = dict()

        if not self._checked_dimensionality:
            expected_ndim = self.num_output_dimensions + 2  # N, C, spatial dims...
            if "generative_modeling" in step_arguments:
                expected_ndim += 1

            assert target.ndim == expected_ndim, (
                f"Expected target to have {expected_ndim} dims for "
                f"num_output_dimensions={self.num_output_dimensions}, "
                f"but got target.shape={target.shape}. "
                f"If target is flattened as B x C x F, use num_output_dimensions=1."
            )

            self._checked_dimensionality = True

        for ind, (name, criterion) in enumerate(zip(self.steps, self.pipeline)):
            if print_loss:
                step_arguments["print_loss"] = True

            loss = criterion(data, target, target_mask, **step_arguments)
            indiv_loses[name] = loss.item()

            if total_loss is None:
                total_loss = loss * self.config.loss_weights[ind]
            else:
                total_loss += loss * self.config.loss_weights[ind]

        return loss, indiv_loses
