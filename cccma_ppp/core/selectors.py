import numpy as np
import dataclasses
from cccma_ppp.core.registery import *
from cccma_ppp.core.core_abc import  moduleABC
from cccma_ppp.models.models_abc import *
from typing import Any, ClassVar
from collections.abc import Callable,Mapping
from pathlib import Path

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
class cVAEModelSelector:

    type: str
    args: Mapping[str, Any ] | None = None
    load_dir: Path | str | None = None
    registery: ClassVar[Registery] = Registery()

    def __post_init__(self):

        if all([self.args is None, self.load_dir is None]):
             raise RuntimeError('Either specify model configuration as args dict or specify a path for loading.')
        
        if self.load_dir is not None:
            RuntimeWarning(f'all model config overwritten by the saved model: \n {self.load_dir}')

    @classmethod
    def register(cls, name: str) -> Callable[..., cVAEmodelsABC]:  # noqa: UP006
        return cls.registery.register(name.lower())   ##attentin: the return is on cls.registery. This means even when register is called on an instance of StepSelector, it will still register the type on the class-level registery, which is what we want.

    def available(cls):
        return cls.registery.available()


    def get_model(self):
        
        if self.load_dir is not None:
            args = self._load_from_checkpoint(self.load_dir)
        else:
            args = self.args

        return self.registery.get(self.type.lower(), args)
   

    def _load_from_checkpoint(self, load_path : Path | str):

        if not Path(load_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")

        model_checkpoint = torch.load( Path(load_path), map_location=self.device, weights_only=False).get('module_config').get('ModelConfig')

        return model_checkpoint

@dataclasses.dataclass
class deterministicModelSelector:

    type: str
    args: Mapping[str, Any ] | None = None
    load_dir: Path | str | None = None
    registery: ClassVar[Registery] = Registery()

    def __post_init__(self):

        if all([self.args is None, self.load_dir is None]):
             raise RuntimeError('Either specify model configuration as args dict or specify a path for loading.')
        
        if self.load_dir is not None:
            RuntimeWarning(f'all model config overwritten by the saved model: \n {self.load_dir}')

    @classmethod
    def register(cls, name: str) -> Callable[..., deterministicmodelsABC]:  # noqa: UP006
        return cls.registery.register(name.lower())   ##attentin: the return is on cls.registery. This means even when register is called on an instance of StepSelector, it will still register the type on the class-level registery, which is what we want.

    def available(cls):
        return cls.registery.available()


    def get_model(self):
        
        if self.load_dir is not None:
            args = self._load_from_checkpoint(self.load_dir)
        else:
            args = self.args

        return self.registery.get(self.type.lower(), args)

    def _load_from_checkpoint(self, load_path : Path | str):

        if not Path(load_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")

        checkpoint_model_config = torch.load( Path(load_path), map_location=self.device, weights_only=False).get('module_config').get('ModelConfig')

        return checkpoint_model_config


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
    def register(cls, name: str) -> Callable[..., cVAEmodelsABC]:  # noqa: UP006
        return cls.registery.register(name.lower())   ##attentin: the return is on cls.registery. This means even when register is called on an instance of StepSelector, it will still register the type on the class-level registery, which is what we want.

    @classmethod
    def available(cls):
        return cls.registery.available()


    def get_model(self):
            return self.registery.get(self.type.lower(), self.args)
