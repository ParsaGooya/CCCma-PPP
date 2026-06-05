import torch
import numpy as np
import pathlib as Path
from pathlib import Path
import dataclasses
import gc
import warnings
from cccma_ppp.loss.loss import Losspipeline
from cccma_ppp.core.registery import *
from cccma_ppp.core.core_abc import moduleABC, moduleConfigABC
from cccma_ppp.core.selectors import *
from cccma_ppp.data.dataloader import BatchData





@dataclasses.dataclass
class deterministicOutput:
    output : torch.Tensor




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
            warnings.warn(f'all module config overwritten by the saved module: \n {self.load_dir}')

        self.model_config = self.ModelConfig.get_model_config()
        self.model = self.model_config.build()
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

        checkpoint = torch.load( Path(load_path), weights_only=False)
        _checkpoint_config = checkpoint.get('module_config')

        self._checkpoint_input_shape = checkpoint.get('model_input_shape')
        self._checkpoint_output_shape = checkpoint.get('model_output_shape')
        
        self.ModelConfig  =  dacite.from_dict(
                                        data_class=deterministicModelSelector,
                                        data= _checkpoint_config.get("ModelConfig"),
                                        config=dacite.Config(strict=True))

        del checkpoint, _checkpoint_config
        gc.collect()
        return self








class deterministic( moduleABC):
    def __init__(self,
        config : deterministicConfig| None = None):

        super().__init__()
        self.config = config
        self.model = self.config.model


        self.built = False
        self.criterion = None

    def build(
        self,
        input_shape: np.ndarray,
        output_shape: np.ndarray | None = None,
        added_features_dim: int = None,
    ):

        if output_shape is None:
            output_shape = input_shape.copy()

    def build(self,
              input_shape : np.ndarray,
              output_shape: np.ndarray|None = None,
              added_features_dim : int = None):
        
        if output_shape is None:
            output_shape = input_shape.copy()
            
        self.input_shape = input_shape
        self.output_shape = output_shape

        if self.config.load_dir is not None:
            assert self.input_shape == self.config._checkpoint_input_shape, f'the requested input shape ({self.input_shape}) does not match the loaded module : {self.config._checkpoint_input_shape}'
            assert self.output_shape == self.config._checkpoint_output_shape, f'the requested output shape ({self.output_shape}) does not match the loaded module : {self.config._checkpoint_output_shape}'


        self.model.build(input_shape = input_shape,
                                            output_shape = output_shape,
                                            added_features_dim = added_features_dim)
        
        if self.config.load_dir is not None:
            self._load_state_dict(self.config.load_dir)

        self.built = True

        return self

    def init_loss_function(self, reconstruction_loss: Losspipeline):

        self.criterion = reconstruction_loss

    def _compute_loss(self, data: BatchData):

        assert self.criterion is not None, (
            "crieterion should be specified before training is possible. Hint: call .init_loss_function() method in your module first."
        )

        output = self.forward(data)

        if isinstance(data.target, (tuple, list)):
            target, target_mask = data.target
        else:
            target, target_mask = data.target, None

        total_loss, indiv_losses = self.criterion(
            output.output, target, target_mask=target_mask, print_loss=False
        )

        losses_dict = {"total_loss": total_loss.item()}

        for key, value in indiv_losses.items():
            losses_dict[key] = value

        return total_loss, losses_dict

    def forward(self, data: BatchData) -> deterministicOutput:

        return self.model(x = data.input,  added_features = data.added_features)

    def predict(self, data: BatchData) -> deterministicOutput:

        return self.forward(data)

    # def _save_state_dict(self, save_path : Path | str):





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

    #     return checkpoint
