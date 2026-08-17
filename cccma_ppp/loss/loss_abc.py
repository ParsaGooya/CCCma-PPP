import abc
import torch
import torch.nn as nn
from typing import Literal

from cccma_ppp.core.core_abc import GenerativeContext

Reduction = Literal["mean", "sum"]



class lossABC(nn.Module, abc.ABC):
    """
    Document this class.
    
    Attributes
    ----------
    generative_context : GenerativeContext
        Description not yet provided.
    generative_context : GenerativeContext
        Description not yet provided.
    """
    generative_context: GenerativeContext


    generative_context: GenerativeContext

    @abc.abstractmethod
    def forward(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        print_loss=False,
    ) -> torch.Tensor:
        """
        Document this function.
        
        Parameters
        ----------
        data : torch.Tensor
            Description not yet provided.
        target : torch.Tensor
            Description not yet provided.
        print_loss : Any
            Description not yet provided.
        """
        pass

    @abc.abstractmethod
    def _print_loss(self, loss):
        """
        Document this function.
        
        Parameters
        ----------
        loss : Any
            Description not yet provided.
        """
        pass

    @abc.abstractmethod
    def _aggregate(self, loss) -> torch.Tensor:
        """
        Document this function.
        
        Parameters
        ----------
        loss : Any
            Description not yet provided.
        """
        pass
