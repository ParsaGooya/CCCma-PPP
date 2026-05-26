import abc
import numpy as np
import torch
import torch.nn as nn

class flowABC(nn.Module,abc.ABC):

    @abc.abstractmethod
    def forward(self, x, condition = None):
        ...
    
    @abc.abstractmethod
    def inverse(self, z, condition = None):
        ...


class deterministicmodelsABC(nn.Module,abc.ABC):

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def build(self,
                input_shape : np.ndarray, 
                output_shape: np.ndarray|None = None,
                added_features_dim : int = None , **kwargs):
            
            ...
    
    @abc.abstractmethod
    def _initialize_weights(self):
        ...
    
    @abc.abstractmethod
    def forward(self,
                x):
        ...


class cVAEmodelsABC( nn.Module,abc.ABC):

    def __init__(self):
        super().__init__()


    @abc.abstractmethod
    def build(self,
                input_shape : np.ndarray, 
                output_shape: np.ndarray|None = None,
                added_features_dim : int = None , **kwargs):
            
            ...
    
    @abc.abstractmethod
    def _initialize_weights(self):
        ...
    
    @abc.abstractmethod
    def forward(self,
                x):
        ...
    @abc.abstractmethod
    def predict(self,
                x):
        ...

    @abc.abstractmethod
    def _recognition(self )-> tuple[torch.Tensor]:
        ...


    @abc.abstractmethod
    def _condition(self)-> tuple[torch.Tensor]:
        ...

    @abc.abstractmethod
    def _generate(self) -> torch.Tensor:

        ...