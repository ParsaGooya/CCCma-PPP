import torch.nn as nn
import xarray as xr
from typing import ClassVar
import dataclasses
import torch
import math
from pathlib import Path
from S2D4P.loss.loss_abc import Reduction
from S2D4P.loss.registery import Registery

from S2D4P.core.core_abc import GenerativeContext
from S2D4P.data.utils import _unwrap_data_variables


@dataclasses.dataclass
class LossStepConfig:
    """
    Configuration for a single loss term.

    Parameters
    ----------
    name : str
        Registered name of the loss function to construct.
    args : dict[str, object]
        Loss-specific arguments passed when constructing the loss function.
    """

    name: str
    args: dict[str, object] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class LosspipelineConfig:
    """
    Document this class.

    Parameters
    ----------
    loss_pipeline : list of LossStepConfig
        Sequence of loss steps.

    loss_weights : list of float or None, optional
        Weights applied to each loss term.

    reduction : {"mean", "sum"}, optional
        Reduction method applied to individual losses.

    saved_output_mask_dir: str or None

        Location of a saved fixed output maks for loss
        calculation. The mask could have any of the
        following shapes:

        1- C, D1, D2, ..
        2- 1, D1, D2, ...
        3- D1, D2, ...

        where C is channel size and D1, D2, ... are output sizes.
        The sizes must match that of the input and target data.
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
        resolves loss weights. Equal weights are assigned when explicit weights are
        not provided.

        Raises
        ------
        ValueError
            If no loss terms are configured, a loss term specifies an argument
            controlled by the pipeline, the number of weights differs from the number
            of loss terms, or the loss weights do not sum to one.
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
            Spatial and variable weights applied by individual loss functions.
        num_output_dimensions : int
            Number of model output dimensions expected by the loss functions.
        generative_context : GenerativeContext | None
            Context describing generative-model and generator behavior. If ``None``,
            a non-generative context is used.
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
        Validate and normalize the configured output mask.

        A mask without a channel dimension is expanded with a singleton channel
        dimension. Spatial dimensions must exactly match the model output, while the
        channel dimension must either be one or match the number of output channels.

        Parameters
        ----------
        output_shape : tuple[int, ...] | None
            Model output shape in channel-first order.

        Raises
        ------
        RuntimeError
            If a saved mask is configured without an output shape, has an incompatible
            number of dimensions, has mismatched non-channel dimensions, or has an
            invalid channel size.
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

        PyTorch files must contain a tensor. NetCDF datasets are converted to a
        channel-first DataArray before being converted to a tensor.

        Returns
        -------
        torch.Tensor | None
            Floating-point output mask, or ``None`` when no saved mask is configured.

        Raises
        ------
        FileNotFoundError
            If the configured mask file does not exist.
        TypeError
            If the mask file is not a supported ``.pt`` or ``.nc`` file, or a
            ``.pt`` file does not contain a tensor.
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
    Combine multiple weighted loss functions into one training objective.

    Each configured loss term is constructed from the loss registry with shared
    reduction, spatial and variable weights, output dimensionality, and generative
    context. Optional target masks and a fixed output mask can be applied before
    individual losses are evaluated.

    Parameters
    ----------
    config : LosspipelineConfig
        Configuration defining loss terms, weights, masking, and reduction.
    weights : xr.DataArray
        Spatial and variable weights supplied to each loss term.
    num_output_dimensions : int
        Number of model output dimensions expected by each loss function.
    generative_context : GenerativeContext | None
        Context describing generative-model and generator behavior.
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
        Initialize the weighted loss pipeline.

        Constructs each registered loss function with shared pipeline settings,
        assigns unique names to repeated loss terms, and registers the optional fixed
        output mask as a module buffer.

        Parameters
        ----------
        config : LosspipelineConfig
            Configuration defining the loss pipeline.
        weights : xr.DataArray
            Spatial and variable weights supplied to individual loss functions.
        num_output_dimensions : int
            Number of model output dimensions expected by each loss function.
        generative_context : GenerativeContext | None
            Context describing generative-model and generator behavior. If ``None``,
            a non-generative context is used.
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
        Evaluate and combine all configured loss terms.

        An optional fixed output mask is combined with the target mask before loss
        calculation. Each loss term is evaluated independently and multiplied by its
        configured weight before contributing to the total loss.

        Parameters
        ----------
        data : torch.Tensor
            Model predictions supplied to each loss function.
        target : torch.Tensor
            Target values used to calculate the losses.
        target_mask : torch.Tensor | None
            Optional validity mask applied during loss calculation.
        print_loss : bool
            Whether individual loss functions print their calculated values.
        step_arguments : dict | None
            Additional keyword arguments passed to each loss function.

        Returns
        -------
        tuple[torch.Tensor, dict[str, torch.Tensor]]
            Weighted total loss and detached individual loss values keyed by loss
            step name.

        Raises
        ------
        AssertionError
            If the target dimensionality is incompatible with the configured number
            of output dimensions and generative-model context.
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
