import abc
import numpy as np
from cccma_ppp.loss.loss import Losspipeline
import torch
import torch.nn as nn
from typing import final
import gc
from pathlib import Path


class moduleABC(nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def build(
        self,
        input_shape: np.ndarray,
        output_shape: np.ndarray | None = None,
        added_features_dim: int = None,
    ): ...
    @abc.abstractmethod
    def init_loss_function(self, reconstruction_loss: Losspipeline, **kwargs): ...

    @abc.abstractmethod
    def _compute_loss(self):
        pass

    @abc.abstractmethod
    def forward(self):
        pass

    @abc.abstractmethod
    def preidct(self):
        pass

    @final
    def _get_device(self):
        param = next(self.parameters(), None)

        if param is not None:
            return param.device

        buffer = next(self.buffers(), None)

        if buffer is not None:
            return buffer.device

        return torch.device("cpu")

    @final
    def _load_state_dict(self, load_path: Path | str, strict: bool = True):

        if not Path(load_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")

        checkpoint = torch.load(
            Path(load_path), map_location=self._get_device(), weights_only=True
        )
        self.load_state_dict(checkpoint["module"], strict=strict)
        del checkpoint
        gc.collect()


class moduleConfigABC(abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def build(
        self,
        input_shape: np.ndarray,
        output_shape: np.ndarray | None = None,
        added_features_dim: int = None,
    ): ...
    @abc.abstractmethod
    def _load_from_checkpoint(self): ...
