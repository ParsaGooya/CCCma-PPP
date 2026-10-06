import numpy as np
import dataclasses
import torch
from typing import Any, ClassVar
from collections.abc import Callable, Mapping
from pathlib import Path
import gc
import warnings

from cccma_ppp.core.registery import Registery
from cccma_ppp.core.core_abc import moduleABC
from cccma_ppp.architectures.models_abc import (
    modelABC,
    flowABC,
    CheckpointConfig,
    modelConfigABC,
)


@dataclasses.dataclass
class ModuleSelector:
    """
    Select and configure a registered module type.

    Parameters
    ----------
    type : str
        Name of the registered module type.
    config : Mapping[str, Any]
        Configuration values used to construct the selected module configuration.
    """

    type: str
    config: Mapping[str, Any]
    registery: ClassVar[Registery] = Registery()

    def __post_init__(self):
        """
        Resolve the registered module configuration.
        """
        self._module_config = self.registery.get(self.type.lower(), self.config)

    @classmethod
    def register(cls, name: str) -> Callable[..., moduleABC]:
        """
        Create a decorator that registers a module configuration.

        Parameters
        ----------
        name : str
            Name under which the module configuration is registered.

        Returns
        -------
        Callable[..., moduleABC]
            Decorator used to register the module configuration class.
        """
        return cls.registery.register(name.lower())

    @classmethod
    def available(cls):
        """
        Return the names of registered module types.

        Returns
        -------
        list
            Names of all module types available through this selector.
        """
        return cls.registery.available()

    @property
    def NUM_INPUT_DIMS(self) -> int:
        """
        Return the number of dimensions expected by the selected model.

        Returns
        -------
        int
            Required number of input dimensions excluding the batch dimension.
        """
        return self.model_config.NUM_INPUT_DIMS

    @property
    def NUM_OUTPUT_DIMS(self) -> int:
        """
        Return the number of dimensions produced by the selected model.

        Returns
        -------
        int
            Required number of output dimensions excluding the batch dimension.
        """
        return self.model_config.NUM_OUTPUT_DIMS

    @property
    def GENERATOR(self) -> bool:
        """
        Return the generator configuration of the selected model.

        Returns
        -------
        GENERATORConfig | None
            Generator configuration, or ``None`` when generator behavior is disabled.
        """
        return self.model_config.GENERATOR

    @property
    def EXPECTS_MASK(self) -> bool:
        """
        Indicate whether the selected model uses validity masks.

        Returns
        -------
        bool
            Whether the selected model expects validity-mask information.
        """
        return self.model_config.EXPECTS_MASK

    @property
    def model_type(self) -> str:
        """
        Return the type of the selected model architecture.

        Returns
        -------
        str
            Registered model type.
        """
        return self._module_config.ModelConfig.type

    @property
    def model_config(self) -> modelConfigABC:
        """
        Return the resolved model configuration.

        Returns
        -------
        modelConfigABC
            Configuration of the selected model
        """
        return self._module_config.model_config

    def build_module(
        self,
        input_shape: np.ndarray,
        output_shape: np.ndarray | None = None,
        added_features_dim: int = None,
    ):
        """
        Build the selected module from its resolved configuration.

        Parameters
        ----------
        input_shape : np.ndarray
            Shape of the module input excluding the batch dimension.
        output_shape : np.ndarray | None
            Shape of the module output excluding the batch dimension. If ``None``,
            the selected module determines its default output shape.
        added_features_dim : int | None
            Number of additional features supplied to the module.

        Returns
        -------
        moduleABC
            Module constructed from the resolved module configuration.
        """
        return self._module_config.build(
            input_shape=input_shape,
            output_shape=output_shape,
            added_features_dim=added_features_dim,
        )


@dataclasses.dataclass
class ModelSelector:
    """
    Select and configure a registered model architecture.

    The model can be configured directly or restored from a checkpoint. When a
    checkpoint is provided, the saved model configuration replaces the explicitly
    provided configuration.

    Parameters
    ----------
    type : str
        Name of the registered model type.
    config : Mapping[str, Any] | None
        Configuration values used to construct the selected model configuration.
    load_dir : Path | str | None
        Path to a checkpoint from which the model configuration and weights are
        restored.
    freeze_weights : bool | None
        Whether model weights loaded from the checkpoint should be frozen.
    """

    type: str
    config: Mapping[str, Any] | None = None
    load_dir: Path | str | None = None
    freeze_weights: bool | None = None

    registery: ClassVar[Registery]

    def __post_init__(self):
        """
        Validate the selector and resolve optional checkpoint configuration.

        When ``load_dir`` is provided, the saved model configuration is loaded from
        the checkpoint and replaces ``config``. Checkpoint metadata is retained for
        subsequent model construction.

        Raises
        ------
        AssertionError
            If the model type stored in the checkpoint does not match ``type``.
        RuntimeError
            If neither ``config`` nor ``load_dir`` is provided.
        ValueError
            If ``freeze_weights`` is enabled without providing ``load_dir``.

        Warns
        -----
        UserWarning
            If the provided model configuration is replaced by checkpoint settings,
            or if checkpoint-loaded model weights are configured to be frozen.
        """
        self.checkpoint_config = None

        if all([self.config is None, self.load_dir is None]):
            raise RuntimeError(
                "Either specify model configuration with config or specify a path for loading."
            )

        if all([self.load_dir is None, self.freeze_weights]):
            raise ValueError(
                "When ''load_dir'' is not provided for a checkpoint model, ''freeze_weights'' must be False or None."
            )

        if self.load_dir is not None:
            self._freeze_weights = self.freeze_weights or False

            checkpoint_module, self.checkpoint_config = _load_config_from_checkpoint(
                self.load_dir, self._freeze_weights
            )
            checkpoint_model = checkpoint_module.get("ModelConfig")
            assert self.type.lower() == checkpoint_model.get("type").lower(), (
                f"The checkpoint {checkpoint_model.get('type')} model does not have the correct type {self.type}"
            )
            self.config = checkpoint_model.get("config")
            warnings.warn(
                f"all model config overwritten by the saved model from {self.load_dir}"
            )
            if self._freeze_weights:
                warnings.warn(f"Checkpoint {self.type} model weights will be frozen.")

    @classmethod
    def register(cls, name: str) -> Callable[..., modelABC]:
        """
        Create a decorator that registers a model configuration.

        Parameters
        ----------
        name : str
            Name under which the model configuration is registered.

        Returns
        -------
        Callable[..., modelABC]
            Decorator used to register the model configuration class.
        """
        return cls.registery.register(name.lower())

    @classmethod
    def available(cls):
        """
        Return the names of all registered model types.

        Returns
        -------
        list
            Names of the models available through this selector.
        """
        return cls.registery.available()

    def get_model_config(self):
        """
        Construct the selected model configuration.

        The registered configuration class is instantiated from ``config``. When
        checkpoint metadata is available, it is attached to the resulting model
        configuration for subsequent checkpoint loading.

        Returns
        -------
        modelConfigABC
            Resolved model configuration, including checkpoint information when
            applicable.
        """
        model_config = self.registery.get(self.type.lower(), self.config)
        if self.checkpoint_config is not None:
            model_config._add_checkpoint_config(self.checkpoint_config)

        return model_config


class cVAEModelSelector(ModelSelector):
    """
    Selector for registered conditional variational autoencoder architectures.

    Maintains an independent registry of model configurations available to cVAE
    modules.
    """

    registery: ClassVar[Registery] = Registery()


class deterministicModelSelector(ModelSelector):
    """
    Selector for registered deterministic model architectures.

    Maintains an independent registry of model configurations available to
    deterministic modules.
    """

    registery: ClassVar[Registery] = Registery()


@dataclasses.dataclass
class FlowSelector:
    """
    Select and configure a registered normalizing flow transformation.

    Parameters
    ----------
    type : str
        Name of the registered flow type.
    args : dict[str, object]
        Arguments used to construct the selected flow.
    """

    type: str
    args: dict[str, object]
    registery: ClassVar[Registery] = Registery()

    def __post_init__(self):
        """
        Initialize the flow selector after dataclass construction.
        """
        pass

    @classmethod
    def register(cls, name: str) -> Callable[..., flowABC]:
        """
        Create a decorator that registers a flow implementation.

        Parameters
        ----------
        name : str
            Name under which the flow implementation is registered.

        Returns
        -------
        Callable[..., flowABC]
            Decorator used to register the flow class.
        """
        return cls.registery.register(name.lower())

    @classmethod
    def available(cls):
        """
        Return the names of all registered flow types.

        Returns
        -------
        list
            Names of the flows available through this selector.
        """
        return cls.registery.available()

    def get_model(self):
        """
        Construct the selected flow implementation.

        The registered flow class is instantiated using the arguments stored in
        ``args``.

        Returns
        -------
        flowABC
            Constructed flow instance.
        """
        return self.registery.get(self.type.lower(), self.args)


def _load_config_from_checkpoint(
    load_path: Path | str, freeze_weights: bool = False, strict: bool = True
):
    """
    Load module configuration and compatibility metadata from a checkpoint.

    The saved module configuration is extracted together with the input and
    output shapes and variable metadata required to validate subsequent model
    construction.

    Parameters
    ----------
    load_path : Path | str
        Path to the checkpoint file.
    freeze_weights : bool
        Whether model weights loaded from the checkpoint should be frozen.
    strict : bool
        Whether checkpoint parameters must exactly match the model state
        dictionary when they are loaded.

    Returns
    -------
    tuple[dict, CheckpointConfig]
        Saved module configuration and a checkpoint configuration containing the
        checkpoint path, expected input and output shapes, variable metadata,
        strict-loading setting, and weight-freezing setting.

    Raises
    ------
    FileNotFoundError
        If ``load_path`` does not exist.
    """
    if not Path(load_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {load_path}")

    checkpoint = torch.load(
        Path(load_path), weights_only=False, map_location=torch.device("cpu")
    )

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
        freeze_weights=freeze_weights,
        strict=strict,
    )

    del checkpoint
    gc.collect()

    return checkpoint_module, checkpoint_config
