import abc
import torch
import torch.nn as nn


class lossABC(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        generative_modeling: bool = False,
        generator: bool = False,
        print_loss=False,
    ) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def _print_loss(self, loss):
        pass

    @abc.abstractmethod
    def _aggregate(self, loss) -> torch.Tensor:
        pass
