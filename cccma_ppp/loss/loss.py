import torch.nn as nn
import xarray as xr
from typing import ClassVar
import dataclasses
import torch
import math
from pathlib import Path
from cccma_ppp.loss.loss_abc import Reduction
from cccma_ppp.loss.registery import Registery

from cccma_ppp.core.core_abc import GenerativeContext
from cccma_ppp.data_modules.utils import _unwrap_data_variables


@dataclasses.dataclass
class LossStepConfig:
    """
    Configuration for an individual loss term.

    Parameters
    ----------
    name : str
        Name of the registered loss function to include in the pipeline.
    args : dict[str, object]
        Additional arguments used to construct the loss function.
    """

    name: str
    args: dict[str, object] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class LosspipelineConfig:
    """
    Configuration for a weighted pipeline of loss functions.

    Parameters
    ----------
    loss_pipeline : list[LossStepConfig]
        Ordered sequence of loss terms included in the pipeline.
    loss_weights : list[float] | None
        Weight assigned to each loss term. If ``None``, all loss terms receive
        equal weight. Explicit weights must sum to one.
    reduction : Reduction
        Reduction method applied by individual loss functions.
    masked_loss_calculation : bool
        Whether target and fixed output masks are applied during loss
        calculation.
    saved_output_mask_dir : str | None
        Path to a fixed output mask stored as a NetCDF or PyTorch file. The mask
        may include the channel dimension or omit it, and its spatial dimensions
        must match the model output.
    """

    loss_pipeline: list[LossStepConfig]
    loss_weights: list[float] = None
    reduction: Reduction = "mean"
    masked_loss_calculation: bool = True
    saved_output_mask_dir: str | None = None

    def __post_init__(self):
        """
        Initialize and validate the loss pipeline configuration.

        Loads an optional fixed output mask, validates individual loss arguments, and
        assigns equal loss weights when explicit weights are not provided.

        Raises
        ------
        ValueError
            If no loss terms are configured, an individual loss specifies an
            argument controlled by the pipeline, the number of loss weights does not
            match the number of loss terms, or the loss weights do not sum to one.
        """
        self.output_mask = self._load_output_mask()

        if not len(self.loss_pipeline) >= 1:
            raise ValueError("provide at least one loss term.")

        self.loss_types: set[str] = set()

        reserved_args = {
            "reduction",
            "weights",
            "num_output_dimensions",
            "generative_context",
        }

        for loss in self.loss_pipeline:
            if reserved_args & loss.args.keys():
                invalid_args = reserved_args & loss.args.keys()
                if invalid_args:
                    raise ValueError(
                        f"Do not specify {sorted(invalid_args)} for individual loss terms. "
                        "These are controlled by LosspipelineConfig."
                    )

            self.loss_types.add(loss.name)

        if self.loss_weights is None:
            self.loss_weights = [
                1.0 / len(self.loss_pipeline) for _ in self.loss_pipeline
            ]
        else:
            if len(self.loss_weights) != len(self.loss_pipeline):
                raise ValueError("Provide a weight for each loss term.")

            if not math.isclose(
                sum(self.loss_weights), 1.0, rel_tol=1e-6, abs_tol=1e-8
            ):
                raise ValueError("Sum of loss term weights should be 1.")

    def build(
        self,
        weights: xr.DataArray,
        num_output_dimensions: int = 2,
        generative_context: GenerativeContext | None = None,
        output_shape: tuple[int, ...] | None = None,
    ):
        """
        Build the configured loss pipeline.

        Parameters
        ----------
        weights : xr.DataArray
            Spatial and variable weights applied by the configured loss functions.
        num_output_dimensions : int
            Number of dimensions in each model output, including the channel
            dimension.
        generative_context : GenerativeContext | None
            Context describing the model's generative behavior. If ``None``, a
            default non-generative context is used.
        output_shape : tuple[int, ...] | None
            Model output shape excluding the batch dimension. Required when a saved
            output mask is configured.

        Returns
        -------
        Losspipeline
            Constructed weighted loss pipeline.
        """
        self._resove_output_mask(output_shape)

        return Losspipeline(
            self,
            weights,
            num_output_dimensions,
            generative_context=generative_context,
        )

    def _resove_output_mask(self, output_shape: tuple[int, ...]):
        """
        Validate and resolve the saved output mask against the model output shape.

        A mask without a channel dimension is expanded to include one. The resulting
        mask must have either one shared channel or one channel per model output
        variable.

        Parameters
        ----------
        output_shape : tuple[int, ...] | None
            Model output shape excluding the batch dimension.

        Raises
        ------
        RuntimeError
            If a saved mask is configured without an output shape, the mask has an
            incompatible number of dimensions, its non-channel dimensions do not
            match the model output, or its channel dimension is incompatible with the
            number of output channels.
        """
        if self.output_mask is not None:
            if output_shape is None:
                raise RuntimeError(
                    "When a saved output mask is specified, output_shape "
                    "must be provided when building Losspipeline."
                )

            mask = self.output_mask

            output_channels = output_shape[0]
            output_dims = tuple(output_shape[1:])

            if mask.ndim == len(output_dims):
                mask = mask.unsqueeze(0)

            if mask.ndim != len(output_shape):
                raise RuntimeError(
                    "The saved output mask must either have the same number of "
                    "dimensions as output_shape, including a channel dimension, "
                    "or omit only the channel dimension. "
                    f"Got mask shape {tuple(mask.shape)} for "
                    f"output shape {tuple(output_shape)}."
                )

            if tuple(mask.shape[1:]) != output_dims:
                raise RuntimeError(
                    "The non-channel dimensions of the saved output mask must "
                    "match the model output. "
                    f"Expected {output_dims}, got {tuple(mask.shape[1:])}."
                )

            if mask.shape[0] not in (1, output_channels):
                raise RuntimeError(
                    "The channel dimension of the saved output mask must either "
                    "be 1, indicating a mask shared across all output channels, "
                    "or match the number of output channels. "
                    f"Expected 1 or {output_channels}, got {mask.shape[0]}."
                )

            self.output_mask = mask

    def _load_output_mask(self) -> torch.Tensor | None:
        """
        Load a fixed output mask from disk.

        PyTorch files must contain a tensor. NetCDF files are converted to a single
        DataArray with variables represented along the channel dimension.

        Returns
        -------
        torch.Tensor | None
            Loaded floating-point output mask, or ``None`` if no mask path is
            configured.

        Raises
        ------
        FileNotFoundError
            If the configured mask file does not exist.
        TypeError
            If a PyTorch mask file does not contain a tensor or the mask file
        """
        if self.saved_output_mask_dir is None:
            return None

        mask_path = Path(self.saved_output_mask_dir)

        if not mask_path.is_file():
            raise FileNotFoundError(f"Saved mask file does not exist: {mask_path}")

        if mask_path.suffix == ".pt":
            mask = torch.load(
                mask_path,
                map_location="cpu",
                weights_only=True,
            )

            if not isinstance(mask, torch.Tensor):
                raise TypeError(
                    "The saved .pt mask must contain a torch.Tensor, "
                    f"got {type(mask).__name__}."
                )

            return mask.float()

        if mask_path.suffix == ".nc":
            with xr.open_dataset(mask_path) as ds:
                mask = _unwrap_data_variables(ds).values

            return torch.as_tensor(mask, dtype=torch.float32)

        raise TypeError(
            "The saved mask must be in NetCDF (.nc) or PyTorch (.pt) format."
        )


class Losspipeline(nn.Module):
    """
    Combine multiple weighted loss functions into a single objective.

    Each configured loss is constructed from the loss registry with shared
    weights, reduction, output dimensionality, and generative context. Optional
    target masks and a fixed output mask can restrict the regions contributing
    to the loss.

    Parameters
    ----------
    config : LosspipelineConfig
        Configuration defining the loss terms, weights, reduction, and masking
        behavior.
    weights : xr.DataArray
        Spatial and variable weights supplied to each loss term.
    num_output_dimensions : int
        Number of dimensions in each model output, including the channel
        dimension.
    generative_context : GenerativeContext | None
        Context describing whether generative modeling or generator-based
        sampling is enabled.
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
        Initialize a weighted loss pipeline.

        Parameters
        ----------
        config : LosspipelineConfig
            Configuration defining the loss terms, weights, reduction, and masking
            behavior.
        weights : xr.DataArray
            Spatial and variable weights supplied to each loss term.
        num_output_dimensions : int
            Number of dimensions in each model output, including the channel
            dimension.
        generative_context : GenerativeContext | None
            Context describing the model's generative behavior. If ``None``, a
            default context is used.
        """
        super().__init__()
        self._checked_dimensionality = False
        self.config = config
        self.reduction = config.reduction
        self.masked_loss_calculation = config.masked_loss_calculation
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

        output_mask = config.output_mask if self.masked_loss_calculation else None
        self.register_buffer("output_mask", output_mask)

    @classmethod
    def register(cls, name: str):
        """
        Create a decorator that registers a loss function.

        Parameters
        ----------
        name : str
            Name under which the loss function is registered.

        Returns
        -------
        Callable
            Decorator used to register the loss class.
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
        Compute the weighted combination of configured loss terms.

        The fixed output mask and target mask are combined when masked loss
        calculation is enabled. Each loss term is evaluated with shared and
        step-specific arguments, and the resulting losses are combined according to
        their configured weights.

        Parameters
        ----------
        data : torch.Tensor
            Model output supplied to each loss function.
        target : torch.Tensor
            Target tensor against which losses are calculated.
        target_mask : torch.Tensor | None
            Optional validity mask restricting loss calculation.
        print_loss : bool
            Whether individual loss functions print their computed values.
        step_arguments : dict | None
            Additional arguments passed to each loss function during evaluation.

        Returns
        -------
        tuple[torch.Tensor, dict[str, torch.Tensor]]
            Weighted total loss and detached individual loss values keyed by loss
            step name.

        Raises
        ------
        AssertionError
            If the target dimensionality does not match the configured output
            dimensionality and generative context.
        """
        if self.output_mask is not None:
            target_mask = (
                self.output_mask
                if target_mask is None
                else target_mask * self.output_mask
            )

        if not self.masked_loss_calculation:
            target_mask = None

        total_loss = 0.0
        indiv_loses = {}

        if step_arguments is None:
            step_arguments_ = dict()
        else:
            step_arguments_ = step_arguments.copy()

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
                step_arguments_["print_loss"] = True

            loss = criterion(data, target, target_mask, **step_arguments_)
            indiv_loses[name] = loss.detach()

            total_loss = total_loss + loss * self.config.loss_weights[ind]

        return total_loss, indiv_loses
