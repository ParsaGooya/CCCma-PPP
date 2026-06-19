import numpy as np
import dataclasses
import torch
from typing import Any, ClassVar
from collections.abc import Callable, Mapping
from pathlib import Path
import gc
import warnings

from cccma_ppp.core.registery import Registery
from cccma_ppp.core import moduleABC
from cccma_ppp.models.models_abc import modelABC, flowABC, CheckpointConfig

@dataclasses.dataclass
class ModuleSelector:
    type: str
    config: Mapping[str, Any]
    registery: ClassVar[Registery] = Registery()

    def __post_init__(self):
        self._module_config = self.registery.get(self.type.lower(), self.config)

    @classmethod
    def register(cls, name: str) -> Callable[..., moduleABC]:
        return cls.registery.register(name.lower())

    @classmethod
    def available(cls):
        return cls.registery.available()

    def build_module(
        self,
        input_shape: np.ndarray,
        output_shape: np.ndarray | None = None,
        added_features_dim: int = None,
    ):

        return self._module_config.build(
            input_shape=input_shape,
            output_shape=output_shape,
            added_features_dim=added_features_dim,
        )


@dataclasses.dataclass
class ModelSelector:
    type: str
    config: Mapping[str, Any] | None = None
    load_dir: Path | str | None = None
    freeze_weights: bool = False

    registery: ClassVar[Registery]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.registery = Registery()

    def __post_init__(self):
        self.checkpoint_config = None

        if all([self.config is None, self.load_dir is None]):
            raise RuntimeError(
                "Either specify model configuration with config or specify a path for loading."
            )

        if self.load_dir is not None:
            checkpoint_module, self.checkpoint_config = _load_config_from_checkpoint(
                self.load_dir
            )
            checkpoint_model = checkpoint_module.get("ModelConfig")
            assert self.type == checkpoint_model.get("type"), (
                f"the specified model does not have the correct type {self.type}"
            )
            self.config = checkpoint_model.get("config")
            warnings.warn(
                f"all model config overwritten by the saved model from {self.load_dir}"
            )
            if self.freeze_weights:
                warnings.warn("Model weights will be frozen.")

    @classmethod
    def register(cls, name: str) -> Callable[..., modelABC]:
        return cls.registery.register(name.lower())

    @classmethod
    def available(cls):
        return cls.registery.available()

    def get_model_config(self):

        model_config = self.registery.get(self.type.lower(), self.config)
        if self.checkpoint_config is not None:
            model_config._add_checkpoint_config(self.checkpoint_config)

        return model_config


class cVAEModelSelector(ModelSelector):
    pass


class deterministicModelSelector(ModelSelector):
    pass


@dataclasses.dataclass
class FlowSelector:
    type: str
    args: dict[str, object]
    registery: ClassVar[Registery] = Registery()

    def __post_init__(self):
        pass

    @classmethod
    def register(cls, name: str) -> Callable[..., flowABC]:
        return cls.registery.register(name.lower())

    @classmethod
    def available(cls):
        return cls.registery.available()

    def get_model(self):
        return self.registery.get(self.type.lower(), self.args)


def _load_config_from_checkpoint(load_path: Path | str, strict: bool = True):

    if not Path(load_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {load_path}")

    checkpoint = torch.load(Path(load_path), weights_only=False)

    checkpoint_module = checkpoint.get("module_config")
    checkpoint_input_shape = checkpoint.get("input_shape")
    checkpoint_output_shape = checkpoint.get("output_shape")
    checkpoint_input_var_metadata = checkpoint.get("input_var_metadata")
    checkpoint_output_var_metadata = checkpoint.get("output_var_metadata")

    checkpoint_config = CheckpointConfig(
        load_path,
        checkpoint_input_shape=checkpoint_input_shape,
        checkpoint_output_shape=checkpoint_output_shape,
        checkpoint_input_var_metadata=checkpoint_input_var_metadata,
        checkpoint_output_var_metadata=checkpoint_output_var_metadata,
        strict=strict,
    )

    del checkpoint
    gc.collect()

    return checkpoint_module, checkpoint_config
