import abc
import numpy as np
import torch
import torch.nn as nn


class flowABC(nn.Module, abc.ABC):
    """
    Abstract base class for normalizing flow models.
    """

    @abc.abstractmethod
    def forward(self, x, condition=None):
        """
        Apply forward transformation from input space to latent space.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        condition : torch.Tensor, optional
            Conditioning input.

        Returns
        -------
        object
            Transformed output.
        """
        ...

    @abc.abstractmethod
    def inverse(self, z, condition=None):
        """
        Apply inverse transformation from latent space to input space.

        Parameters
        ----------
        z : torch.Tensor
            Latent tensor.
        condition : torch.Tensor, optional
            Conditioning input.

        Returns
        -------
        object
            Inverse transformed output.
        """
        ...


class deterministicmodelsABC(nn.Module, abc.ABC):
    """
    Abstract base class for deterministic models.
    """

    def __init__(self):
        """
        Initialize base deterministic model.

        Returns
        -------
        None
        """

        super().__init__()

    @abc.abstractmethod
    def build(self, input_shape, output_shape=None, added_features_dim=None, **kwargs):
        """
        Build model architecture.

        Parameters
        ----------
        input_shape : np.ndarray
            Shape of input data.
        output_shape : np.ndarray or None, optional
            Shape of output data.
        added_features_dim : int, optional
            Dimension of additional features.
        **kwargs
            Additional arguments.

        Returns
        -------
        object
            Built model instance.
        """
        ...

    @abc.abstractmethod
    def _initialize_weights(self):
        """
        Initialize model weights.

        Returns
        -------
        None
        """
        ...

    @abc.abstractmethod
    def forward(self, x):
        """
        Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        object
            Model output.
        """
        ...


class cVAEmodelsABC(nn.Module, abc.ABC):
    """
    Abstract base class for conditional variational autoencoder models.
    """

    def __init__(self):
        """
        Initialize base cVAE model.

        Returns
        -------
        None
        """

        super().__init__()

    @abc.abstractmethod
    def build(self, input_shape, output_shape=None, added_features_dim=None, **kwargs):
        """
        Build model architecture.

        Parameters
        ----------
        input_shape : np.ndarray
            Shape of input data.
        output_shape : np.ndarray or None, optional
            Shape of output data.
        added_features_dim : int, optional
            Dimension of additional features.
        **kwargs
            Additional arguments.

        Returns
        -------
        object
            Built model instance.
        """
        ...

    @abc.abstractmethod
    def _initialize_weights(self):
        """
        Initialize model weights.

        Returns
        -------
        None
        """
        ...

    @abc.abstractmethod
    def forward(self, x):
        """
        Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        object
            Model output.
        """
        ...

    @abc.abstractmethod
    def predict(self, x):
        """
        Generate predictions from the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        object
            Predicted output.
        """
        ...

    @abc.abstractmethod
    def _recognition(self):
        """
        Encode inputs into latent distribution parameters.

        Returns
        -------
        tuple of torch.Tensor
            Latent distribution parameters.
        """
        ...

    @abc.abstractmethod
    def _condition(self):
        """
        Compute conditioning representations.

        Returns
        -------
        tuple of torch.Tensor
            Conditioning parameters.
        """
        ...

    @abc.abstractmethod
    def _generate(self):
        """
        Decode latent variables into output space.

        Returns
        -------
        torch.Tensor
            Generated output.
        """
        ...
