import numpy as np
import dataclasses
from cccma_ppp.core.registery import *
from cccma_ppp.core.core_abc import  moduleABC
from cccma_ppp.models.models_abc import *
from typing import Any, ClassVar
from collections.abc import Callable,Mapping
from pathlib import Path
import gc
import warnings

@dataclasses.dataclass
class ModuleSelector:
    type: str
    config: Mapping[str, Any ]
    registery: ClassVar[Registery] = Registery()

    def __post_init__(self):
        self._module_config = self.registery.get(self.type.lower(), self.config)

    @classmethod
    def register(cls, name: str) -> Callable[..., moduleABC]:  # noqa: UP006
        return cls.registery.register(name.lower())   ##attentin: the return is on cls.registery. This means even when register is called on an instance of StepSelector, it will still register the type on the class-level registery, which is what we want.

    @classmethod
    def available(cls):
        return cls.registery.available()

    def build_module(self,
              input_shape : np.ndarray,
              output_shape: np.ndarray|None = None,
              added_features_dim : int = None):

        return self._module_config.build(input_shape = input_shape,
                                        output_shape = output_shape,
                                        added_features_dim  = added_features_dim)


@dataclasses.dataclass
class ModelSelector:

    type: str
    args: Mapping[str, Any ] | None = None
    load_dir: Path | str | None = None
    freeze_weights : bool = False

    registery: ClassVar[Registery] 

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.registery = Registery()

    def __post_init__(self):
        self.checkpoint_config = None

        if all([self.args is None, self.load_dir is None]):
             raise RuntimeError('Either specify model configuration with args dict or specify a path for loading.')

        
        if self.load_dir is not None:
            
            checkpoint_model, self.checkpoint_config = _load_config_from_checkpoint(self.load_dir)
            assert self.type == checkpoint_model.get('type'), f'the specified model does not have the correct type {self.type}'
            self.args = checkpoint_model.get('args')
            warnings.warn(f'all model config overwritten by the saved model from {self.load_dir}')
            if self.freeze_weights:
                warnings.warn(f'Froze model weights ...')

    @classmethod
    def register(cls, name: str) -> Callable[..., modelABC]:  # noqa: UP006
        return cls.registery.register(name.lower())   ##attentin: the return is on cls.registery. This means even when register is called on an instance of StepSelector, it will still register the type on the class-level registery, which is what we want.

    @classmethod
    def available(cls):
        return cls.registery.available()


    def get_model(self):

        model = self.registery.get(self.type.lower(), self.args)
        if self.checkpoint_config is not None:
            model._add_checkpoint_config(self.checkpoint_config)

        return model



class cVAEModelSelector(ModelSelector):
    pass

class deterministicModelSelector(ModelSelector):
    pass

    

##### can deleter this and add the registery machinery
#  to the NormalizedFlowConfig instead. This is because
# similar to preprocessing, we need to be able to pass
# a list of flows on top of each other #######

@dataclasses.dataclass
class FlowSelector:
    type  : str
    args : dict[str, object]
    registery: ClassVar[Registery] = Registery()

    def __post_init__(self):
        pass

    @classmethod
    def register(cls, name: str) -> Callable[..., flowABC]:  # noqa: UP006
        return cls.registery.register(name.lower())   ##attentin: the return is on cls.registery. This means even when register is called on an instance of StepSelector, it will still register the type on the class-level registery, which is what we want.

    @classmethod
    def available(cls):
        return cls.registery.available()


    def get_model(self):
            return self.registery.get(self.type.lower(), self.args)





def _load_config_from_checkpoint(load_path : Path | str, strict : bool = True):

    if not Path(load_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {load_path}")

    checkpoint = torch.load( Path(load_path), weights_only=False)
    checkpoint_model = checkpoint.get('module_config').get('ModelConfig')
    checkpoint_input_shape =  checkpoint.get('model_input_shape')
    checkpoint_output_shape =  checkpoint.get('model_output_shape')

    checkpoint_config = CheckpointConfig(load_path,
                                            checkpoint_input_shape,
                                            checkpoint_output_shape,
                                            strict = strict)

    del checkpoint
    gc.collect()

    return checkpoint_model, checkpoint_config


