import torch.nn as nn
import xarray as xr
from typing import ClassVar
import dataclasses
import torch

from cccma_ppp.loss.loss_abc import Reduction
from cccma_ppp.loss.registery import Registery

from cccma_ppp.core.core_abc import GenerativeContext


@dataclasses.dataclass
class LossStepConfig:
    """
    Document this class.

    Parameters
    ----------
    name : str
        Description not yet provided.
    args : dict[str, object]
        Description not yet provided.
    """

    name: str
    args: dict[str, object] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class LosspipelineConfig:
    """
    Document this class.

    Parameters
    ----------
    loss_pipeline : list[LossStepConfig]
        Description not yet provided.
    loss_weights : list[float]
        Description not yet provided.
    reduction : Reduction
        Description not yet provided.
    """

    loss_pipeline: list[LossStepConfig]
    loss_weights: list[float] = None
    reduction: Reduction = "mean"

    def __post_init__(self):
        """
        Document this function.

        Raises
        ------
        ValueError
            Description not yet provided.
        """
        if not len(self.loss_pipeline) >= 1:
            raise ValueError("provide at least one loss term.")

        self.loss_types: set[str] = set()

        for loss in self.loss_pipeline:
            if len(loss.args) > 0:
                if {
                    "reduction",
                    "weights",
                    "num_output_dimensions",
                    "generative_context",
                }.intersection(list(loss.args.keys())):
                    raise ValueError(
                        "Do not specify reduction, weights, num_output_dimensions or generative_context for individual loss terms manually. "
                        "Set them for the LosspipelineConfig."
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

    def build(
        self,
        weights: xr.DataArray,
        num_output_dimensions: int = 2,
        generative_context: GenerativeContext | None = None,
    ):
        """
        Document this function.

        Parameters
        ----------
        weights : xr.DataArray
            Description not yet provided.
        num_output_dimensions : int
            Description not yet provided.
        generative_context : GenerativeContext | None
            Description not yet provided.

        Returns
        -------
        Any
            Description not yet provided.
        """
        return Losspipeline(
            self, weights, num_output_dimensions, generative_context=generative_context
        )


class Losspipeline(nn.Module):
    """
    Document this class.

    Parameters
    ----------
    config : LosspipelineConfig
        Description not yet provided.
    weights : xr.DataArray
        Description not yet provided.
    num_output_dimensions : int
        Description not yet provided.
    generative_context : GenerativeContext | None
        Description not yet provided.
    """

    registery: ClassVar[Registery] = Registery()

    def __init__(
        self,
        config: LosspipelineConfig,
        weights: xr.DataArray,
        num_output_dimensions: int = 2,
        generative_context: GenerativeContext | None = None,
    ):
        """
        Document this function.

        Parameters
        ----------
        config : LosspipelineConfig
            Description not yet provided.
        weights : xr.DataArray
            Description not yet provided.
        num_output_dimensions : int
            Description not yet provided.
        generative_context : GenerativeContext | None
            Description not yet provided.
        """
        super().__init__()
        self._checked_dimensionality = False
        self.config = config
        self.reduction = config.reduction
        self.weights = weights
        self.num_output_dimensions = num_output_dimensions
        self.generative_context = (
            generative_context
            if generative_context is not None
            else GenerativeContext()
        )
        self.pipeline = []
        self.steps = []

        for step in self.config.loss_pipeline:
            name = step.name
            args = dict(step.args)
            args.update(
                num_output_dimensions=self.num_output_dimensions,
                weights=self.weights,
                reduction=self.reduction,
                generative_context=self.generative_context,
            )

            self.pipeline.append(self.registery.get(name.lower(), args))

            if "low_ress_kernel_size" in args.keys():
                name = f"{name}_low_ress_{args.get('low_ress_kernel_size')}"

            if name in self.steps:
                name = f"{name}_{self.steps.count(name) + 1}"

            self.steps.append(name)

        self.pipeline = torch.nn.ModuleList(self.pipeline)

    @classmethod
    def register(cls, name: str):
        """
        Document this function.

        Parameters
        ----------
        name : str
            Description not yet provided.

        Returns
        -------
        Any
            Description not yet provided.
        """
        return cls.registery.register(name.lower())

    def forward(
        self,
        data,
        target,
        target_mask=None,
        print_loss=False,
        step_arguments: dict = None,
    ):
        """
        Document this function.

        Parameters
        ----------
        data : Any
            Description not yet provided.
        target : Any
            Description not yet provided.
        target_mask : Any
            Description not yet provided.
        print_loss : Any
            Description not yet provided.
        step_arguments : dict
            Description not yet provided.

        Returns
        -------
        Any
            Description not yet provided.

        Raises
        ------
        AssertionError
            Description not yet provided.
        """
        total_loss = None
        indiv_loses = {}

        if step_arguments is None:
            step_arguments = dict()

        if not self._checked_dimensionality:
            expected_ndim = self.num_output_dimensions + 1
            if self.generative_context.generative_modeling:
                expected_ndim += 1

            assert target.ndim == expected_ndim, (
                f"Expected target to have {expected_ndim} dims for "
                f"num_output_dimensions={self.num_output_dimensions}, "
                f"but got target.shape={target.shape}. "
                f"If target is flattened as B x C x F, use num_output_dimensions=2."
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

        return total_loss, indiv_loses
