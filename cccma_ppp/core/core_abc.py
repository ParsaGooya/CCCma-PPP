import abc
import numpy as np
from cccma_ppp.loss.loss import Losspipeline
import torch.nn as nn


class moduleABC( nn.Module,abc.ABC):

    def __init__(self):
        super().__init__()


    @abc.abstractmethod
    def build(self,
              input_shape : np.ndarray,
              output_shape: np.ndarray|None = None,
              added_features_dim : int = None):

        ...
    @abc.abstractmethod
    def init_loss_function(self,
                           reconstruction_loss : Losspipeline,
                           **kwargs):
        ...

    @abc.abstractmethod
    def _compute_loss(self):
        pass

    @abc.abstractmethod
    def forward(self):
        pass

    @abc.abstractmethod
    def preidct(self):
        pass



class moduleConfigABC(abc.ABC):

    def __init__(self):
        super().__init__()


    @abc.abstractmethod
    def build(self,
              input_shape : np.ndarray,
              output_shape: np.ndarray|None = None,
              added_features_dim : int = None):

        ...
    @abc.abstractmethod
    def _load_from_checkpoint(self):
        ...
