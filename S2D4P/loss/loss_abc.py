import abc
import torch
import torch.nn as nn
from typing import Literal

from S2D4P.core.core_abc import GenerativeContext

Reduction = Literal["mean", "sum"]


class lossABC(nn.Module, abc.ABC):
    """
    Abstract base class for loss functions.

    Defines the common interface for computing, aggregating, and reporting model
    losses while retaining information about the model's generative behavior.

    Attributes
    ----------
    generative_context : GenerativeContext
        Context describing whether the associated model uses generative modeling
        or generator-based stochastic output sampling.
    """

    generative_context: GenerativeContext

    generative_context: GenerativeContext

    @abc.abstractmethod
    def forward(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        print_loss=False,
    ) -> torch.Tensor:
        """
        Compute the loss between model output and target data.

        Parameters
        ----------
        data : torch.Tensor
            Model predictions used to calculate the loss.
        target : torch.Tensor
            Target values against which predictions are compared.
        print_loss : bool
            Whether to print the calculated loss.

        Returns
        -------
        torch.Tensor
            Calculated loss.
        """
        pass

    @abc.abstractmethod
    def _print_loss(self, loss):
        """
        Print the calculated loss.

        Parameters
        ----------
        loss : torch.Tensor
            Loss value to print.
        """
        pass

    @abc.abstractmethod
    def _aggregate(self, loss) -> torch.Tensor:
        """
        Aggregate unreduced loss values.

        Parameters
        ----------
        loss : torch.Tensor
            Unreduced loss values.

        Returns
        -------
        torch.Tensor
            Loss reduced according to the implementation-specific aggregation method.
        """
        pass
