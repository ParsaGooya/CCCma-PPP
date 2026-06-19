import abc
import torch
import torch.nn as nn
from typing import Literal


Reduction = Literal['mean', 'sum']


class lossABC(nn.Module, abc.ABC):

    """
    Abstract base class defining the interface for loss functions.

    Methods
    -------
    forward(data, target, generative_modeling=False, generator=False, print_loss=False)
        Compute loss between predictions and targets.
    _print_loss(loss)
        Print formatted loss values.
    _aggregate(loss)
        Reduce raw loss values into a final scalar.
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
        Compute loss between model outputs and targets.

        Parameters
        ----------
        data : torch.Tensor
            Model predictions or outputs.
        target : torch.Tensor
            Ground truth targets.
        generative_modeling : bool, optional
            Whether the loss is computed in a generative modeling setting
            (e.g., multiple samples/ensembles).
        generator : bool, optional
            Whether the loss is computed for a generator component.
        print_loss : bool, optional
            Whether to print loss values.

        Returns
        -------
        torch.Tensor
            Computed loss value.
        """

        pass

    @abc.abstractmethod
    def _print_loss(self, loss):

        """
        Print formatted loss value.

        Parameters
        ----------
        loss : torch.Tensor
            Loss value to display.

        Returns
        -------
        None
        """
        pass

    @abc.abstractmethod
    def _aggregate(self, loss) -> torch.Tensor:

        """
        Aggregate raw loss values.

        Parameters
        ----------
        loss : torch.Tensor
            Element-wise loss values.

        Returns
        -------
        torch.Tensor
            Aggregated loss according to reduction method.
        """
        
        pass
