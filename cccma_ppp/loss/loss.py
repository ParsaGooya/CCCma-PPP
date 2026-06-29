from __future__ import annotations
import torch.nn as nn
import xarray as xr
from typing import ClassVar
import dataclasses

from cccma_ppp.loss.loss_abc import Reduction
from cccma_ppp.loss.registery import Registery


@dataclasses.dataclass
class LossStepConfig:
    """
    Configuration for a single loss step in a loss pipeline.

    Parameters
    ----------
    name : str
        Name of the registered loss function.
    args : dict[str, object], optional
        Arguments used to initialize the loss function.
    """

    name: str
    args: dict[str, object] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class LosspipelineConfig:
    """
    Configuration for combining multiple loss functions.

    Parameters
    ----------
    loss_pipeline : list of LossStepConfig
        Sequence of loss steps.
    loss_weights : list of float or None, optional
        Weights applied to each loss term.
    reduction : {"mean", "sum"}, optional
        Reduction method applied to individual losses.
    """

    loss_pipeline: list[LossStepConfig]
    loss_weights: list[float] = None
    reduction: Reduction = "mean"

    def __post_init__(self):
        """
        Validate loss pipeline configuration.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If no loss terms are provided.
        ValueError
            If loss weights are inconsistent or invalid.
        """

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
        """
        Construct loss pipeline.

        Parameters
        ----------
        weights : xr.DataArray
            Spatial or variable weights applied to loss computation.
        num_output_dimensions : int, optional
            Dimensionality of model outputs.

        Returns
        -------
        Losspipeline
            Initialized loss pipeline.
        """

        return Losspipeline(self, weights, num_output_dimensions)


class Losspipeline(nn.Module):
    """
    Container for sequentially applying multiple loss functions.

    Combines multiple loss terms using configurable weights,
    supports spatial weighting, and enforces dimensionality checks.

    Attributes
    ----------
    config : LosspipelineConfig
        Configuration object.
    weights : xr.DataArray
        Spatial or variable weights.
    num_output_dimensions : int
        Output dimensionality used for validation.
    pipeline : list
        List of instantiated loss functions.
    steps : list of str
        Names of loss steps.
    """

    registery: ClassVar[Registery] = Registery()

    def __init__(
        self,
        config: LosspipelineConfig,
        weights: xr.DataArray,
        num_output_dimensions: int = 2,
    ):
        """
        Initialize loss pipeline.

        Parameters
        ----------
        config : LossPipelineConfig
            Configuration for the loss pipeline.
        weights : xr.DataArray
            Precomputed spatial/variable weights.
        num_output_dimensions : int, optional
            Number of output dimensions handled by the loss.
        """

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
        """
        Register a loss function.

        Parameters
        ----------
        name : str
            Name used for registration.

        Returns
        -------
        Callable
            Decorator for registering loss functions.
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
        Compute combined loss from all pipeline steps.

        Parameters
        ----------
        data : torch.Tensor
            Model predictions.
        target : torch.Tensor
            Ground truth targets.
        target_mask : torch.Tensor or None, optional
            Optional mask applied to targets.
        print_loss : bool, optional
            Whether to print individual loss values.
        step_arguments : dict or None, optional
            Additional arguments passed to each loss step.

        Returns
        -------
        tuple
            (total_loss, individual_losses)

            total_loss : torch.Tensor
                Weighted combined loss.
            individual_losses : dict of str to float
                Individual loss contributions.

        Raises
        ------
        AssertionError
            If input dimensionality does not match expectations.
        """

        total_loss = None
        indiv_loses = {}

        if step_arguments is None:
            step_arguments = dict()

        if not self._checked_dimensionality:
            expected_ndim = self.num_output_dimensions + 2
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
