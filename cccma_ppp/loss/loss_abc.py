import abc
import torch
import torch.nn as nn


class lossABC(nn.Module, abc.ABC):
    """
    Abstract base class for defining loss functions.
    """

    @abc.abstractmethod
    def forward(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        generative_modeling: bool = False,
        generator: bool = False,
        print_loss=False,
    ):
        """
        Compute loss between predictions and targets.

        Parameters
        ----------
        data : torch.Tensor
            Predicted output.
        target : torch.Tensor
            Ground truth target.
        generative_modeling : bool, optional
            Whether loss is used for generative modeling.
        generator : bool, optional
            Whether called in generator context.
        print_loss : bool, optional
            Whether to print loss value.

        Returns
        -------
        torch.Tensor
            Computed loss.

        Raises
        ------
        NotImplementedError
            If not implemented in subclass.
        """
        ...

    @abc.abstractmethod
    def _print_loss(self, loss):
        """
        Print loss value in a formatted way.

        Parameters
        ----------
        loss : torch.Tensor
            Loss value.

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            If not implemented in subclass.
        """
        ...

    @abc.abstractmethod
    def _aggregate(self, loss):
        """
        Apply reduction to loss values.

        Parameters
        ----------
        loss : torch.Tensor
            Element-wise loss values.

        Returns
        -------
        torch.Tensor
            Aggregated loss.

        Raises
        ------
        NotImplementedError
            If not implemented in subclass.
        """
        ...
