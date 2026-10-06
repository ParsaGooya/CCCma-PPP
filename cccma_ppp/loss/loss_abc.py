import abc
import torch
import torch.nn as nn
from typing import Literal

from cccma_ppp.core.core_abc import GenerativeContext

Reduction = Literal["mean", "sum"]


class lossABC(nn.Module, abc.ABC):
    """
    Abstract base class for loss functions.

    Defines the common interface for computing, aggregating, and reporting loss
    values while retaining information about the model's generative context.

    Attributes
    ----------
    generative_context : GenerativeContext
        Context describing whether generative modeling or generator-based
        stochastic sampling is enabled.
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
        Compute the loss between model output and target values.

        Parameters
        ----------
        data : torch.Tensor
            Model output for which the loss is computed.
        target : torch.Tensor
            Target values against which the model output is compared.
        print_loss : bool
            Whether to print the computed loss.

        Returns
        -------
        torch.Tensor
            Computed loss value.
        """
        pass

    @abc.abstractmethod
    def _print_loss(self, loss):
        """
        Print a computed loss value.

        Parameters
        ----------
        loss : torch.Tensor
            Loss value to print.
        """
        pass

    @abc.abstractmethod
    def _aggregate(self, loss) -> torch.Tensor:
        """
        Aggregate element-wise loss values.

        Parameters
        ----------
        loss : torch.Tensor
            Unaggregated loss values.

        Returns
        -------
        torch.Tensor
            Aggregated loss value.
        """
        pass
