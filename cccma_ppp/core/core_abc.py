import abc
import numpy as np
from cccma_ppp.loss.loss import Losspipeline
import torch.nn as nn


class moduleABC(nn.Module, abc.ABC):
    """
    Abstract base class for neural network modules.
    """

    def __init__(self):
        """
        Initialize the base module.

        Returns
        -------
        None
        """

        super().__init__()

    @abc.abstractmethod
    def build(
        self,
        input_shape: np.ndarray,
        output_shape: np.ndarray | None = None,
        added_features_dim: int = None,
    ):
        """
        Build the module architecture based on input and output shapes.

        Parameters
        ----------
        input_shape : np.ndarray
            Shape of the input data.
        output_shape : np.ndarray or None, optional
            Shape of the output data.
        added_features_dim : int, optional
            Dimension of additional features.

        Returns
        -------
        object
            Built module instance.

        Raises
        ------
        NotImplementedError
            If not implemented in subclass.
        """
        ...

    @abc.abstractmethod
    def init_loss_function(self, reconstruction_loss: Losspipeline, **kwargs):
        """
        Initialize the loss function for training.

        Parameters
        ----------
        reconstruction_loss : Losspipeline
            Loss pipeline for reconstruction.
        **kwargs
            Additional configuration arguments.

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
    def _load_from_state(self, load_dir):
        """
        Load module state from a checkpoint.

        Parameters
        ----------
        load_dir : str or path-like
            Directory containing saved state.

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
    def _compute_loss(self):
        """
        Compute loss for the module.

        Returns
        -------
        object
            Computed loss value.

        Raises
        ------
        NotImplementedError
            If not implemented in subclass.
        """
        pass

    @abc.abstractmethod
    def forward(self):
        """
        Perform forward pass through the module.

        Returns
        -------
        object
            Forward pass output.

        Raises
        ------
        NotImplementedError
            If not implemented in subclass.
        """
        pass

    @abc.abstractmethod
    def preidct(self):
        """
        Generate predictions from the module.

        Returns
        -------
        object
            Predicted outputs.

        Raises
        ------
        NotImplementedError
            If not implemented in subclass.
        """
        pass
