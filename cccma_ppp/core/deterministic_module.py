import torch
import numpy as np
import pathlib as Path
from pathlib import Path
import dataclasses
import gc
from cccma_ppp.loss.loss import Losspipeline
from cccma_ppp.core.registery import *
from cccma_ppp.core.core_abc import moduleABC, moduleConfigABC
from cccma_ppp.core.selectors import *
from cccma_ppp.data.dataloader import BatchData


@ModuleSelector.register('deterministic')
@ModuleSelector.register('default')
@dataclasses.dataclass
class deterministicConfig(moduleConfigABC):
    ModelConfig  : deterministicModelSelector | None = None
    load_dir : str| None = None

    def __post_init__(self):
        if self.load_dir is None:
            assert self.ModelConfig is not None, 'provide loading dir or model configurations'
        else:
            self._load_from_checkpoint(self.load_dir)
            RuntimeWarning(f'all module config overwritten by the saved module: \n {self.load_dir}')

        self.model = self.ModelConfig.get_model()
    def build(self,
              input_shape : np.ndarray,
              output_shape: np.ndarray|None = None,
              added_features_dim : int = None):

        return deterministic(self).build(input_shape= input_shape,
                                 output_shape = output_shape,
                                 added_features_dim = added_features_dim)

    def _load_from_checkpoint(self, load_path : Path | str):

        if not Path(load_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")

        module_checkpoint = torch.load( Path(load_path), map_location=self.device, weights_only=False).get('module_config')
        
        self.ModelConfig  =  dacite.from_dict(
                                        data_class=deterministicModelSelector,
                                        data=module_checkpoint.ModelConfig,
                                        config=dacite.Config(strict=True))

        del module_checkpoint
        gc.collect()
        return self




@dataclasses.dataclass
class deterministicOutput:
    output : torch.Tensor





class deterministic( moduleABC):
    def __init__(self,
        config : deterministicConfig| None = None):

        super().__init__()
        self.config = config
        self.model = self.config.model


        self.built = False
        self.criterion = None



    def build(self,
              input_shape : np.ndarray,
              output_shape: np.ndarray|None = None,
              added_features_dim : int = None):


        self.model.build(input_shape = input_shape,
                                            output_shape = output_shape,
                                            added_features_dim = added_features_dim)

        self.built = True

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

    # def _save_state_dict(self, save_path : Path | str):

    #     path = Path(save_path) / f"deterministic_module.pt"
    #     torch.save(self.state_dict(), path)


    # def _load_from_state(self, load_path : Path | str, strict : bool = True):

    #     assert self.built, 'module stgate should be built for torch to load the weights into. Hint: call .build() method first.'

    #     if not Path(load_path).exists():
    #         raise FileNotFoundError(f"Checkpoint not found: {load_path}")

    #     checkpoint = torch.load( Path(load_path), map_location=self.device, weights_only=False)
    #     self.load_state_dict( checkpoint ,strict=strict)

    #     return checkpoint

