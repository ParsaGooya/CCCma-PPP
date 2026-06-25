from __future__ import annotations
import abc
import torch
import torch.nn as nn
from typing import Literal


Reduction = Literal["mean", "sum"]


class lossABC(nn.Module, abc.ABC):
    """
    Abstract base class for loss functions.

    Defines the interface for computing loss, applying reductions,
    and optionally printing loss values.

    Methods
    -------
    forward(data, target, generative_modeling, generator, print_loss)
        Compute loss.
    _aggregate(loss)
        Apply reduction to loss values.
    _print_loss(loss)
        Print formatted loss value.
    """

    @abc.abstractmethod
    def forward(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        generative_modeling: bool = False,
        generator: bool = False,
        print_loss=False,
    ) -> torch.Tensor:
        """
        Compute loss between predictions and targets.

        Parameters
        ----------
        data : torch.Tensor
            Model predictions.
        target : torch.Tensor
            Ground truth targets.
        generative_modeling : bool, optional
            Whether loss is used in a generative modeling context.
        generator : bool, optional
            Indicates if generator-specific behavior is applied.
        print_loss : bool, optional
            Whether to print the loss value.

        Returns
        -------
        torch.Tensor
            Computed loss value.
        """

        pass

    @abc.abstractmethod
    def _print_loss(self, loss):
        """
        Print loss value.

        Parameters
        ----------
        loss : torch.Tensor
        """

        pass

    @abc.abstractmethod
    def _aggregate(self, loss) -> torch.Tensor:
        """
        Apply reduction to loss values.

        Parameters
        ----------
        loss : torch.Tensor

        Returns
        -------
        torch.Tensor
            Reduced loss.
        """

        pass
