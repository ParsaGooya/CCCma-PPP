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
    Document this class.
    """

    _type: ClassVar[str | None] = None

    @classmethod
    def check_registered(cls):
        """
        Document this function.

        Raises
        ------
        RuntimeError
            Description not yet provided.
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
        Document this function.

        Parameters
        ----------
        input_shape : np.ndarray
            Description not yet provided.
        output_shape : np.ndarray | None
            Description not yet provided.
        added_features_dim : int
            Description not yet provided.
        """
        pass

    @abc.abstractmethod
    def _load_from_checkpoint(self):
        """
        Document this function.
        """
        pass


class moduleABC(nn.Module, abc.ABC):
    """
    Document this class.
    """

    def __init__(self):
        """
        Document this function.
        """
        super().__init__()

    @abc.abstractmethod
    def init_loss_function(self, reconstruction_loss: nn.Module, **kwargs):
        """
        Document this function.

        Parameters
        ----------
        reconstruction_loss : nn.Module
            Description not yet provided.
        **kwargs : Any
            Description not yet provided.
        """
        pass

    @abc.abstractmethod
    def _compute_loss(self):
        """
        Document this function.
        """
        pass

    @abc.abstractmethod
    def forward(self):
        """
        Document this function.
        """
        pass

    @abc.abstractmethod
    def predict(self):
        """
        Document this function.
        """
        pass

    @final
    def _get_device(self):
        """
        Document this function.

        Returns
        -------
        Any
            Description not yet provided.
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
        Document this function.

        Parameters
        ----------
        load_path : Path | str
            Description not yet provided.
        strict : bool
            Description not yet provided.

        Raises
        ------
        FileNotFoundError
            Description not yet provided.
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
    Document this class.

    Parameters
    ----------
    module : moduleABC | None
        Description not yet provided.
    """

    def __init__(self, module: moduleABC | None = None):
        """
        Document this function.

        Parameters
        ----------
        module : moduleABC | None
            Description not yet provided.
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
    Document this class.

    Parameters
    ----------
    output : torch.Tensor
        Description not yet provided.
    """

    output: torch.Tensor
