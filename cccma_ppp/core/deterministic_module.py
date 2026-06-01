import torch
import numpy as np
import pathlib as Path
from pathlib import Path
import dataclasses

from cccma_ppp.loss.loss_abc import lossABC
from cccma_ppp.loss.loss import Losspipeline
from cccma_ppp.core.registery import *
from cccma_ppp.core.core_abc import moduleABC
from cccma_ppp.core.selectors import *
from cccma_ppp.data.dataloader import BatchData


@ModuleSelector.register('deterministic')
@ModuleSelector.register('default')
@dataclasses.dataclass
class deterministicConfig:
    ModelConfig  : deterministicModelSelector | None = None
    load_dir : str| None = None

    def __post_init__(self):
        if self.load_dir is None:
            assert self.ModelConfig is not None, 'provide loading dir or model configurations'
        else:
            RuntimeWarning(f'all model config overwritten by the loaded model: \n {self.load_dir}')


    def build(self,
              input_shape : np.ndarray,
              output_shape: np.ndarray|None = None,
              added_features_dim : int = None):

        return deterministic(self).build(input_shape= input_shape,
                                 output_shape = output_shape,
                                 added_features_dim = added_features_dim)






@dataclasses.dataclass
class deterministicOutput:
    output : torch.Tensor





class deterministic( moduleABC):
    def __init__(self,
        config : deterministicConfig| None = None):

        super().__init__()
        self.config = config
        self.model = self.config.ModelConfig.get_model()


        self.built = False
        self.criterion = None



    def build(self,
              input_shape : np.ndarray,
              output_shape: np.ndarray|None = None,
              added_features_dim : int = None):


        self.model.build(input_shape = input_shape,
                                            output_shape = output_shape,
                                            added_features_dim = added_features_dim)

        self.build = True
        if self.config.load_dir is not None:
            self._load_from_state(self.config.load_dir)

        return self

    def init_loss_function(self, reconstruction_loss : Losspipeline):

        self.criterion = reconstruction_loss

    def _compute_loss(self,
                        data : BatchData):

        assert self.criterion is not None, 'crieterion should be specified before training is possible. Hint: call .init_loss_function() method in your module first.'

        output = self.forward(data)

        if isinstance(data.target, (tuple, list)):
            target, target_mask = data.target
        else:
            target, target_mask = data.target , None

        total_loss, indiv_losses = self.criterion(
            output.output,
            target,
            target_mask = target_mask,
            print_loss = False)

        losses_dict = {
            "total_loss": total_loss.item()}

        for key, value in indiv_losses.items():
            losses_dict[key] = value

        return total_loss, losses_dict

    def forward(self, data: BatchData) -> deterministicOutput:

        return deterministicOutput(output = self.model(x = data.input,  added_features = data.added_features))

    def predict(self, data: BatchData) -> deterministicOutput:

        return deterministicOutput(output = self.model(x = data.input,  added_features = data.added_features))


    def _get_device(self):
        param = next(self.parameters(), None)

        if param is not None:
            return param.device

        buffer = next(self.buffers(), None)

        if buffer is not None:
            return buffer.device

        return torch.device("cpu")

    def _save_state_dict(self, save_path : Path | str):

        path = Path(save_path) / f"deterministic_module.pt"
        torch.save(self.state_dict(), path)


    def _load_from_state(self, load_path : Path | str, strict : bool = True):

        assert self.built, 'module stgate should be built for torch to load the weights into. Hint: call .build() method first.'

        if not Path(load_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")

        checkpoint = torch.load( Path(load_path), map_location=self.device, weights_only=False)
        self.load_state_dict( checkpoint ,strict=strict)

        return checkpoint

