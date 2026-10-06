import abc
import numpy as np
import torch
import torch.nn as nn
from typing import final
import gc
from pathlib import Path
from typing import ClassVar
import dataclasses


class moduleConfigABC(abc.ABC):
    """
    Abstract base class for module configurations.

    Defines the interface for constructing modules and restoring their
    configuration from checkpoints.
    """

    _type: ClassVar[str | None] = None

    @classmethod
    def check_registered(cls):
        """
        Verify that the module configuration has been registered.

        Raises
        ------
        RuntimeError
            If the configuration class has not been assigned a registered module
            type.
        """
        if cls._type is None:
            raise RuntimeError(f"{cls.__name__} has not been registered.")

    @abc.abstractmethod
    def build(
        self,
        input_shape: np.ndarray,
        output_shape: np.ndarray | None = None,
        added_features_dim: int = None,
    ):
        """
        Build a module from this configuration.

        Parameters
        ----------
        input_shape : np.ndarray
            Shape of the module input excluding the batch dimension.
        output_shape : np.ndarray | None
            Shape of the module output excluding the batch dimension.
        added_features_dim : int | None
            Number of additional features supplied to the module.

        Returns
        -------
        moduleABC
            Module constructed from this configuration.
        """
        pass

    @abc.abstractmethod
    def _load_from_checkpoint(self):
        """
        Restore module configuration from a checkpoint.

        Returns
        -------
        moduleConfigABC
            Configuration populated with settings restored from the checkpoint.
        """
        pass


class moduleABC(nn.Module, abc.ABC):
    """
    Abstract base class for trainable model modules.

    Defines the common interface for loss initialization, loss computation,
    forward evaluation, prediction, and checkpoint loading.
    """

    def __init__(self):
        """
        Initialize the base model module.
        """
        super().__init__()

    @abc.abstractmethod
    def init_loss_function(self, reconstruction_loss: nn.Module, **kwargs):
        """
        Initialize the reconstruction loss used by the module.

        Parameters
        ----------
        reconstruction_loss : nn.Module
            Reconstruction loss module used during training.
        **kwargs : Any
            Additional module-specific loss configuration arguments.
        """
        pass

    @abc.abstractmethod
    def _compute_loss(self):
        """
        Compute the training loss for the module.

        Returns
        -------
        Any
            Module-specific loss values and associated reporting information.
        """
        pass

    @abc.abstractmethod
    def forward(self):
        """
        Perform a forward pass through the module.

        Returns
        -------
        OutputABC
            Module-specific output produced by the forward pass.
        """

        pass

    @abc.abstractmethod
    def predict(self):
        """
        Generate predictions with the module.

        Returns
        -------
        OutputABC
            Module-specific prediction output.
        """
        pass

    @final
    def _get_device(self):
        """
        Determine the device on which the module is stored.

        Returns
        -------
        torch.device
            Device of the first module parameter, or the first module buffer when no
            parameters are present. Returns the CPU device if neither exists.
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
        Load module parameters from a checkpoint.

        Parameters
        ----------
        load_path : Path | str
            Path to the checkpoint file.
        strict : bool
            Whether the checkpoint state dictionary must exactly match the module
            state dictionary.

        Raises
        ------
        FileNotFoundError
            If the configured checkpoint file does not exist.
        """
        if not Path(load_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")

        checkpoint = torch.load(
            Path(load_path), map_location=self._get_device(), weights_only=True
        )
        self.load_state_dict(checkpoint["module"], strict=strict)
        del checkpoint
        gc.collect()


class GenerativeContext:
    """
    Describe the generative behavior of a model module.

    The context indicates separately whether generator-based stochastic sampling
    is configured and whether the underlying model performs generative modeling.

    Parameters
    ----------
    module : moduleABC | None
        Module from which to determine the generative settings. If ``None``,
        both settings are disabled.
    """

    def __init__(self, module: moduleABC | None = None):
        """
        Initialize the generative context for a module.

        Parameters
        ----------
        module : moduleABC | None
            Module from which to determine generator and generative-modeling
            settings. If ``None``, both settings are set to ``False``.
        """
        if module is not None:
            self.generator = getattr(module.model_config, "GENERATOR", None) is not None
            self.generative_modeling = getattr(
                module.model, "generative_modeling", False
            )

        else:
            self.generator = False
            self.generative_modeling = False


@dataclasses.dataclass
class OutputABC:
    """
    Base container for model module outputs.

    Parameters
    ----------
    output : torch.Tensor
        Tensor containing the primary model output.
    """

    output: torch.Tensor
