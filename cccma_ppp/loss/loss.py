import torch.nn as nn
from cccma_ppp.loss.registery import Registery
from typing import ClassVar
import dataclasses


@dataclasses.dataclass
class LossStepConfig:
    """
    Configuration for a single loss step in a loss pipeline.
    """

    name: str
    args: dict[str, object] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class LosspipelineConfig:
    """
    Configuration for constructing a composite loss pipeline.
    """

    loss_pipeline: list[LossStepConfig]
    loss_weights: list[float] = None
    reduction: str = "mean"

    def __post_init__(self):
        """
        Validate loss pipeline configuration.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If reduction or weights are invalid.
        ValueError
            If invalid loss arguments are provided.
        """

        assert self.reduction.lower() in ["mean", "sum"]
        assert len(self.loss_pipeline) >= 1, "provide at least one loss term."

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
            assert len(self.loss_weights) == len(self.loss_pipeline), (
                "Provide a weight for each loss term."
            )
            assert sum(self.loss_weights) == 1, "Sum of loss term weights should be 1."
        else:
            self.loss_weights = [
                1 / len(self.loss_pipeline) for _ in self.loss_pipeline
            ]

    def build(self, weights, num_output_dimensions=2):
        """
        Build a Losspipeline instance.

        Parameters
        ----------
        weights : xr.DataArray
            Spatial or variable weighting array.
        num_output_dimensions : int, optional
            Number of output spatial dimensions.

        Returns
        -------
        Losspipeline
            Constructed loss pipeline.
        """

        return Losspipeline(self, weights, num_output_dimensions)


class Losspipeline(nn.Module):
    """
    Pipeline for combining multiple loss functions with weights.
    """

    registery: ClassVar[Registery] = Registery()

    def __init__(self, config, weights, num_output_dimensions=2):
        """
        Initialize loss pipeline.

        Parameters
        ----------
        config : LosspipelineConfig
            Loss pipeline configuration.
        weights : xr.DataArray
            Spatial or variable weighting array.
        num_output_dimensions : int, optional
            Number of output spatial dimensions.

        Returns
        -------
        None
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
        Register a loss function in the pipeline registry.

        Parameters
        ----------
        name : str
            Name used for registration.

        Returns
        -------
        callable
            Decorator for class registration.
        """

        return cls.registery.register(name.lower())

    def forward(
        self, data, target, target_mask=None, print_loss=False, step_arguments=None
    ):
        """
        Compute total and individual losses.

        Parameters
        ----------
        data : torch.Tensor
            Predicted output.
        target : torch.Tensor
            Ground truth target.
        target_mask : torch.Tensor, optional
            Mask for valid target values.
        print_loss : bool, optional
            Whether to print individual loss values.
        step_arguments : dict, optional
            Additional arguments passed to loss steps.

        Returns
        -------
        tuple
            Total loss tensor and dictionary of individual losses.

        Raises
        ------
        AssertionError
            If input dimensionality is inconsistent.
        """

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


# class Losspipeline(nn.Module):
#     registery : ClassVar[registery] = registery()

#     def __init__(self,
#                  loss_pipeline : list[tuple[str, dict[str, object]] | tuple[str] | str],
#                  loss_weights : list[float] = None,
#                  reduction : str = 'mean'):

#         super().__init__()
#         assert reduction.lower() in ['mean', 'sum']
#         assert len(loss_pipeline) >= 1, 'provide at least one loss term.'
#         self.initialized = False
#         self.loss_weights = loss_weights
#         self.loss_pipeline = loss_pipeline
#         self.reduction = reduction


#         if isinstance(self.loss_weights, list):
#             assert len(self.loss_weights) == len(loss_pipeline), 'Provide a weight for each loss term.'
#             assert sum(self.loss_weights) == 1, 'Sum of loss term weights should be 1.'
#         else:
#             self.loss_weights = [1/len(loss_pipeline) for _ in loss_pipeline]


#     def init(self, weights : xr.DataArray, num_output_dimensions : int = 2):

#         self.weights = weights
#         self.num_output_dimensions = num_output_dimensions
#         self.pipeline = []
#         self.steps = []

#         for step in self.loss_pipeline:
#             if all([len(step) == 2, isinstance(step, tuple)]):
#                 name, args = step
#             elif all([len(step) == 1, isinstance(step, tuple)]):
#                 name , args = step[0], {}
#             elif isinstance(step, str):
#                 name , args = step, {}
#             else:
#                 raise ValueError(f"Invalid step format: {step}")

#             args = dict(args)
#             args['num_output_dimensions'] = self.num_output_dimensions
#             args['weights'] = self.weights
#             args['reduction'] = self.reduction

#             self.pipeline.append(self.registery.get(name, args))
#             if name in self.steps:
#                 name = f'{name}_{self.steps.count(name) + 1}'
#             if 'low_ress_kernel_size' in args.keys():
#                 name = f'{name}_low_ress{args.get("low_ress_kernel_size")}'

#             self.steps.append(name)
#             self.initialized = True
#     @classmethod
#     def register(cls, name : str):
#         return cls.registery.register(name.lower())

#     def forward(self, data, target, print_loss = False, step_arguments : dict = None):
#         total_loss = None
#         indiv_loses = ()

#         if step_arguments is None:
#             step_arguments = dict()


#         for ind, (name, criterion) in enumerate(zip(self.steps, self.pipeline)):

#             if print_loss:
#                 step_arguments['print_loss'] = True

#             loss = criterion(data, target , **step_arguments)
#             indiv_loses.append((name,  loss.item()))

#             if total_loss is None:
#                 total_loss = loss * self.loss_weights[ind]
#             else:
#                 total_loss += loss* self.loss_weights[ind]

#         return loss, indiv_loses
