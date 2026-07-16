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
from cccma_ppp.generic import Distributed

@dataclasses.dataclass
class ModuleSelector:
    """
    Selector for constructing training modules from a registry.

    Parameters
    ----------
    type : str
        Name of the registered module.
    config : Mapping[str, Any]
        Configuration dictionary used to instantiate the module config.
    """

    type: str
    config: Mapping[str, Any]
    registery: ClassVar[Registery] = Registery()

    def __post_init__(self):
        """
        Retrieve module configuration from registry.

        Returns
        -------
        None
        """

        self._module_config = self.registery.get(self.type.lower(), self.config)

    @classmethod
    def register(cls, name: str) -> Callable[..., moduleABC]:
        """
        Register a module configuration class.

        Parameters
        ----------
        name : str
            Name used for registration.

        Returns
        -------
        Callable
            Decorator for registering module configuration classes.
        """

        return cls.registery.register(name.lower())

    @classmethod
    def available(cls):
        """
        Return available module types.

        Returns
        -------
        list of str
            Registered module names.
        """

        return cls.registery.available()
    
    @property
    def NUM_INPUT_DIMS(self) -> int:
        """
        Return number of input dims in
        the selected architecture.

        Return
        -------
        int
        """
        
        return self._module_config.model_config.NUM_INPUT_DIMS

    @property
    def NUM_OUTPUT_DIMS(self) -> int:
        """
        Return number of output dims in
        the selected architecture.

        Return
        -------
        int
        """

        return self._module_config.model_config.NUM_OUTPUT_DIMS 

    @property
    def GENERATOR(self) -> bool:
        """
        Check if the selected architecture 
        has a GENERATOR.

        Return
        -------
        bool
        """
        
        return getattr(self._module_config.model_config, "GENERATOR", False)

    def build_module(
        self,
        input_shape: np.ndarray,
        output_shape: np.ndarray | None = None,
        added_features_dim: int = None,
    ):
        """
        Build module instance.

        Parameters
        ----------
        input_shape : np.ndarray
            Input data shape.
        output_shape : np.ndarray or None, optional
            Output data shape.
        added_features_dim : int, optional
            Additional feature dimension.

        Returns
        -------
        moduleABC
            Built module instance.
        """

        return self._module_config.build(
            input_shape=input_shape,
            output_shape=output_shape,
            added_features_dim=added_features_dim,
        )




@dataclasses.dataclass
class PredictorSelector:
    """
    Selector for constructing Predictor objects from a registry.

    Parameters
    ----------
    config : Mapping[str, Any]
        Configuration dictionary used to instantiate the Predictor config.
    """

    config: Mapping[str, Any]
    registery: ClassVar[Registery] = Registery()

    @classmethod
    def register(cls, name: str) -> Callable[..., moduleABC]:
        """
        Register a module configuration class.

        Parameters
        ----------
        name : str
            Name used for registration.

        Returns
        -------
        Callable
            Decorator for registering module configuration classes.
        """

        return cls.registery.register(name.lower())

    @classmethod
    def available(cls):
        """
        Return available module types.

        Returns
        -------
        list of str
            Registered module names.
        """

        return cls.registery.available()

    def build_predictor(
        self,
        module: moduleABC,
        distributed: Distributed,
        output_dir: Path | str,
        num_output_covariance_sampling: int,
    ):
        """
        Build predictor instance.

        Parameters
        ----------
        module : moduleABC
            Trained module 

        Returns
        -------
        PreictorABC
            Built predictor instance.
        """
        Predictor_Config = self.registery.get(module.config._type.lower(), self.config)
        return Predictor_Config.build(module, 
                                      distributed,
                                      output_dir,
                                      num_output_covariance_sampling)



@dataclasses.dataclass
class ModelSelector:
    """
    Selector for constructing model configurations, optionally from checkpoint.

    Parameters
    ----------
    type : str
        Model type identifier.
    config : Mapping[str, Any] or None, optional
        Configuration dictionary for model.
    load_dir : pathlib.Path or str or None, optional
        Path to checkpoint for loading model configuration.
    freeze_weights : bool, optional
        Whether to freeze model weights after loading.
    """

    type: str
    config: Mapping[str, Any] | None = None
    load_dir: Path | str | None = None
    freeze_weights: bool = False

    registery: ClassVar[Registery]

    def __init_subclass__(cls, **kwargs):
        """
        Initialize subclass with its own registry.

        Returns
        -------
        None
        """

        super().__init_subclass__(**kwargs)
        cls.registery = Registery()

    def __post_init__(self):
        """
        Validate configuration and optionally load from checkpoint.

        Raises
        ------
        RuntimeError
            If neither configuration nor checkpoint path is provided.
        AssertionError
            If model type does not match checkpoint.
        """

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
        """
        Decorator for registering model classes.

        Parameters
        ----------
        name : str
            Name used for registration.

        Returns
        -------
        Callable
            Decorator for registering model classes.
        """

        return cls.registery.register(name.lower())

    @classmethod
    def available(cls):
        """
        Return available model types.

        Returns
        -------
        list of str
            Registered model names.
        """

        return cls.registery.available()

    def get_model_config(self):
        """
        Get instantiated model configuration.

        Returns
        -------
        modelABC
            Model configuration instance.
        """

        model_config = self.registery.get(self.type.lower(), self.config)
        if self.checkpoint_config is not None:
            model_config._add_checkpoint_config(self.checkpoint_config)

        return model_config


class cVAEModelSelector(ModelSelector):
    """
    Model selector for cVAE models.
    """

    pass


class deterministicModelSelector(ModelSelector):
    """
    Model selector for deterministic models.
    """

    pass


@dataclasses.dataclass
class FlowSelector:
    """
    Selector for constructing flow models.

    Parameters
    ----------
    type : str
        Flow model type.
    args : dict of str to object
        Arguments used to instantiate the flow model.
    """

    type: str
    args: dict[str, object]
    registery: ClassVar[Registery] = Registery()

    def __post_init__(self):
        """
        Post-initialization hook.

        Returns
        -------
        None
        """

        pass

    @classmethod
    def register(cls, name: str) -> Callable[..., flowABC]:
        """
        Register a flow model class.

        Parameters
        ----------
        name : str
            Name used for registration.

        Returns
        -------
        Callable
            Decorator for registering flow classes.
        """

        return cls.registery.register(name.lower())

    @classmethod
    def available(cls):
        """
        Return available flow types.

        Returns
        -------
        list of str
            Registered flow names.
        """

        return cls.registery.available()

    def get_model(self):
        """
        Instantiate flow model.

        Returns
        -------
        flowABC
            Flow model instance.
        """

        return self.registery.get(self.type.lower(), self.args)


def _load_config_from_checkpoint(load_path: Path | str, strict: bool = True):
    """
    Load configuration metadata from checkpoint.

    Parameters
    ----------
    load_path : pathlib.Path or str
        Path to checkpoint file.
    strict : bool, optional
        Whether to enforce strict loading.

    Returns
    -------
    tuple
        (checkpoint_module, checkpoint_config)

        checkpoint_module : dict
            Stored module configuration dictionary.
        checkpoint_config : CheckpointConfig
            Object containing metadata and shape information.

    Raises
    ------
    FileNotFoundError
        If checkpoint file does not exist.
    """

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
