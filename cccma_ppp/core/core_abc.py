import abc
import numpy as np
import torch
import torch.nn as nn
from typing import final
import gc
from pathlib import Path
from typing import ClassVar
import dataclasses

from cccma_ppp.loss import Losspipeline


class moduleABC(nn.Module, abc.ABC):
    """
    Abstract base class for trainable modules.

    Methods
    -------
    init_loss_function(reconstruction_loss, **kwargs)
        Initialize loss function.
    _compute_loss()
        Compute training loss.
    forward()
        Perform forward pass.
    predict()
        Perform inference.
    """

    def __init__(self):
        """
        Initialize base module.

        Returns
        -------
        None
        """

        super().__init__()

    @abc.abstractmethod
    def init_loss_function(self, reconstruction_loss: Losspipeline, **kwargs):
        """
        Initialize loss function for training.

        Parameters
        ----------
        reconstruction_loss : Losspipeline
            Loss pipeline used for computing reconstruction loss.
        **kwargs
            Additional arguments for loss initialization.

        Returns
        -------
        None
        """

        pass

    @abc.abstractmethod
    def _compute_loss(self):
        """
        Compute total training loss.

        Returns
        -------
        tuple
            Loss tensor and dictionary of loss components.
        """

        pass

    @abc.abstractmethod
    def forward(self):
        """
        Perform forward pass.

        Returns
        -------
        object
            Model output (implementation-dependent).
        """

        pass

    @abc.abstractmethod
    def predict(self):
        """
        Perform inference using the model.

        Returns
        -------
        object
            Model predictions.
        """

        pass

    @final
    def _get_device(self):
        """
        Determine device on which the module resides.

        Returns
        -------
        torch.device
            Device of parameters or buffers. Defaults to CPU if none exist.
        """

        param = next(self.parameters(), None)

        if param is not None:
            return param.device

        buffer = next(self.buffers(), None)

        if buffer is not None:
            return buffer.device

        return torch.device("cpu")

    @final
    def _load_state_dict(self, load_path: Path | str, strict: bool = True):
        """
        Load model weights from checkpoint.

        Parameters
        ----------
        load_path : pathlib.Path or str
            Path to checkpoint file.
        strict : bool, optional
            Whether to enforce strict parameter matching.

        Returns
        -------
        None

        Raises
        ------
        FileNotFoundError
            If the checkpoint file does not exist.
        """

        if not Path(load_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")

        checkpoint = torch.load(
            Path(load_path), map_location=self._get_device(), weights_only=True
        )
        self.load_state_dict(checkpoint["module"], strict=strict)
        del checkpoint
        gc.collect()


class moduleConfigABC(abc.ABC):
    """
    Abstract base class for module configuration objects.

    Methods
    -------
    build(input_shape, output_shape=None, added_features_dim=None)
        Construct module instance.
    _load_from_checkpoint()
        Load configuration from checkpoint.
    """
    _type: ClassVar[str | None] = None


    @classmethod
    def check_registered(cls):
        '''
        Class attribute _type will only be set if the class is registered.
        '''
        if cls._type is None:
            raise RuntimeError(
                f"{cls.__name__} has not been registered."
            )

    @abc.abstractmethod
    def build(
        self,
        input_shape: np.ndarray,
        output_shape: np.ndarray | None = None,
        added_features_dim: int = None,
    ):
        """
        Construct and return a module instance.

        Parameters
        ----------
        input_shape : np.ndarray
            Shape of the input data expected by the module.
        output_shape : np.ndarray or None, optional
            Shape of the target/output data. If None, the module may assume
            the same shape as the input or infer it internally.
        added_features_dim : int, optional
            Number of additional feature dimensions provided alongside the input.

        Returns
        -------
        moduleABC
            Instantiated and optionally initialized module ready for training
            or inference.
        """

        pass

    @abc.abstractmethod
    def _load_from_checkpoint(self):
        """
        Load configuration from a saved checkpoint.

        Returns
        -------
        None
        """

        pass


@dataclasses.dataclass
class OutputABC:
    """
    Base container for model outputs.
    """
    output: torch.Tensor