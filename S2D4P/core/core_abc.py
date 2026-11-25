import abc
import numpy as np
import torch
import torch.nn as nn
from typing import final
import gc
from pathlib import Path
from typing import ClassVar
import dataclasses
from S2D4P.generic.runtime import RuntimeContext

class moduleConfigABC(abc.ABC):
    """
    Abstract base class for model module configurations.

    Defines the interface for constructing model modules and restoring
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
            If the configuration class does not have a registered module type.
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
        Build a model module from this configuration.

        Parameters
        ----------
        input_shape : np.ndarray
            Shape of each model input excluding the batch dimension.
        output_shape : np.ndarray | None
            Shape of each model output excluding the batch dimension. If ``None``,
            the implementation may use the input shape.
        added_features_dim : int | None
            Number of additional features supplied with each sample.

        Returns
        -------
        Any
            Constructed model module.
        """
        pass

    @abc.abstractmethod
    def _load_from_checkpoint(self):
        """
        Restore module configuration from a checkpoint.

        Returns
        -------
        Any
            Implementation-specific restored configuration.
        """
        pass


class moduleABC(nn.Module, abc.ABC):
    """
    Abstract base class for trainable model modules.

    Defines the common interface for loss initialization, loss computation,
    forward evaluation, prediction, device resolution, and checkpoint loading.
    """

    def __init__(self):
        """
        Initialize the model module.
        """
        super().__init__()

    @final
    def model_setup(self, 
                config: moduleConfigABC,
                input_shape: np.ndarray,
                output_shape: np.ndarray | None = None,
                added_features_dim: int = None,
        ):

        self.config = config
        self.architecture_config = config.architecture_config

        self.criterion = None

        if output_shape is None:
            output_shape = input_shape.copy()

        if self.config.load_dir is not None:
            if (
                RuntimeContext.INPUT_VAR_METADATA
                != self.config.checkpoint_config.checkpoint_input_var_metadata
            ):
                raise RuntimeError(
                    "the loaded module was not trained for the consistent input variables or preprocessing steps"
                )
            if (
                RuntimeContext.TARGET_VAR_METADATA
                != self.config.checkpoint_config.checkpoint_output_var_metadata
            ):
                raise RuntimeError(
                    "the loaded module was not trained for the consistent output variables or preprocessing steps"
                )

            if input_shape != self.config.checkpoint_config.checkpoint_input_shape:
                raise RuntimeError(
                    f"the requested input shape ({input_shape}) does not match the loaded module : {self.config.checkpoint_config.checkpoint_input_shape}"
                )
            if output_shape != self.config.checkpoint_config.checkpoint_output_shape:
                raise RuntimeError(
                    f"the requested output shape ({output_shape}) does not match the loaded module : {self.config.checkpoint_config.checkpoint_output_shape}"
                )
            
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.added_features_dim = added_features_dim

        return self

    @abc.abstractmethod
    def init_loss_function(self, reconstruction_loss: nn.Module, **kwargs):
        """
        Initialize the loss function used by the model module.

        Parameters
        ----------
        reconstruction_loss : nn.Module
            Reconstruction loss module used during training.
        kwargs : Any
            Additional module-specific loss configuration arguments.
        """
        pass

    @abc.abstractmethod
    def _compute_loss(self):
        """
        Compute the training loss.

        Returns
        -------
        Any
            Module-specific loss value and associated loss information.
        """
        pass

    @abc.abstractmethod
    def forward(self):
        """
        Run the model forward pass.

        Returns
        -------
        Any
            Module-specific model output.
        """
        pass

    @abc.abstractmethod
    def predict(self):
        """
        Generate model predictions.

        Returns
        -------
        Any
            Module-specific prediction output.
        """
        pass

    @final
    def _get_device(self):
        """
        Return the device currently used by the module.

        The device of the first parameter is preferred, followed by the first
        registered buffer. CPU is returned when the module contains neither.

        Returns
        -------
        torch.device
            Device associated with the module.
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
            Whether checkpoint parameters must exactly match the module state
            dictionary.

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


class GenerativeContext:
    """
    Describe the generative behavior of a model module.

    The context distinguishes generator-based stochastic output sampling from
    generative modeling architectures such as conditional variational
    autoencoders.

    Parameters
    ----------
    module : moduleABC | None
        Model module from which generative settings are determined. If ``None``,
        both generative flags are disabled.
    """

    def __init__(self, module: moduleABC | None = None):
        """
        Initialize generative context from a model module.

        Parameters
        ----------
        module : moduleABC | None
            Model module whose architecture configuration and architecture are
            inspected for generator and generative-modeling behavior.
        """
        if module is not None:
            self.generator = (
                getattr(module.architecture_config, "GENERATOR", None) is not None
            )
            self.generative_modeling = getattr(
                module.arch, "generative_modeling", False
            )

        else:
            self.generator = False
            self.generative_modeling = False


@dataclasses.dataclass
class OutputABC:
    """
    Base container for model outputs.

    Parameters
    ----------
    output : torch.Tensor
        Tensor containing model predictions.
    """

    output: torch.Tensor
