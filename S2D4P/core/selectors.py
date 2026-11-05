import numpy as np
import dataclasses
import torch
from typing import Any, ClassVar
from collections.abc import Callable, Mapping
from pathlib import Path
import gc
import warnings

from S2D4P.core.registery import Registery
from S2D4P.core.core_abc import moduleABC
from S2D4P.architectures.architecture_abc import (
    ArchABC,
    flowABC,
    CheckpointConfig,
    ArchConfigABC,
)


@dataclasses.dataclass
class ModuleSelector:
    """
    Select and configure a registered model module.

    Parameters
    ----------
    type : str
        Name of the registered module type.
    config : Mapping[str, Any]
        Configuration used to construct the selected module configuration.
    """

    type: str
    config: Mapping[str, Any]
    registery: ClassVar[Registery] = Registery()

    def __post_init__(self):
        """
        Resolve the selected module configuration from the registry.
        """
        self._module_config = self.registery.get(self.type.lower(), self.config)

    @classmethod
    def register(cls, name: str) -> Callable[..., moduleABC]:
        """
        Create a decorator that registers a model module configuration.

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
        Return the names of all registered module types.

        Returns
        -------
        list[str]
            Names currently available in the module registry.
        """
        return cls.registery.available()

    @property
    def NUM_INPUT_DIMS(self) -> int:
        """
        Return the number of input dimensions expected by the architecture.

        Returns
        -------
        int
            Number of input dimensions expected by the selected architecture.
        """
        return self.architecture_config.NUM_INPUT_DIMS

    @property
    def NUM_OUTPUT_DIMS(self) -> int:
        """
        Return the number of input dimensions expected by the architecture.

        Returns
        -------
        int
            Number of input dimensions expected by the selected architecture.
        """
        return self.architecture_config.NUM_OUTPUT_DIMS

    @property
    def GENERATOR(self) -> bool:
        """
        Return the generator configuration of the selected architecture.

        Returns
        -------
        GENERATORConfig | None
            Generator configuration, or ``None`` when generator-based sampling is
            disabled.
        """
        return self.architecture_config.GENERATOR

    @property
    def EXPECTS_MASK(self) -> bool:
        """
        Indicate whether the selected architecture expects spatial masks.

        Returns
        -------
        bool
            Whether spatial validity masks are required by the architecture.
        """
        return self.architecture_config.EXPECTS_MASK

    @property
    def arch_type(self) -> str:
        """
        Return the selected architecture type.

        Returns
        -------
        str
            Registered type name of the selected architecture.
        """

        return self._module_config.Architecture.type

    @property
    def architecture_config(self) -> ArchConfigABC:
        """
        Return the resolved architecture configuration.

        Returns
        -------
        ArchConfigABC
            Architecture configuration associated with the selected module.
        """
        return self._module_config.architecture_config

    def build_module(
        self,
        input_shape: np.ndarray,
        output_shape: np.ndarray | None = None,
        added_features_dim: int = None,
    ):
        """
        Build the selected model module.

        Parameters
        ----------
        input_shape : np.ndarray
            Shape of each model input excluding the batch dimension.
        output_shape : np.ndarray | None
            Shape of each model output excluding the batch dimension.
        added_features_dim : int | None
            Number of additional features supplied with each sample.

        Returns
        -------
        moduleABC
            Constructed model module.
        """
        return self._module_config.build(
            input_shape=input_shape,
            output_shape=output_shape,
            added_features_dim=added_features_dim,
        )


@dataclasses.dataclass
class ArchSelector:
    """
    Select and configure a registered model architecture.

    An architecture can be configured directly or reconstructed from a saved
    checkpoint. Checkpoint settings replace the explicitly supplied configuration
    when ``load_dir`` is provided.

    Parameters
    ----------
    type : str
        Name of the registered architecture type.
    config : Mapping[str, Any] | None
        Configuration used to construct the architecture.
    load_dir : Path | str | None
        Path to a checkpoint from which architecture configuration and metadata
        are loaded.
    freeze_weights : bool | None
        Whether weights restored from the checkpoint are frozen.
    """

    type: str
    config: Mapping[str, Any] | None = None
    load_dir: Path | str | None = None
    freeze_weights: bool | None = None

    registery: ClassVar[Registery]

    def __post_init__(self):
        """
        Resolve and validate the architecture selection.

        When a checkpoint is supplied, its saved architecture configuration replaces
        the explicitly configured values and checkpoint compatibility information is
        retained for architecture construction.

        Raises
        ------
        AssertionError
            If the architecture type stored in the checkpoint differs from the
            requested architecture type.
        RuntimeError
            If neither an architecture configuration nor a checkpoint path is
            provided.
        ValueError
            If weight freezing is requested without a checkpoint.

        Warns
        -----
        UserWarning
            If checkpoint configuration replaces the supplied architecture settings,
            or checkpoint weights are configured to be frozen.
        """
        self.checkpoint_config = None

        if all([self.config is None, self.load_dir is None]):
            raise RuntimeError(
                "Either specify architecture configuration with config or specify a path for loading."
            )

        if all([self.load_dir is None, self.freeze_weights]):
            raise ValueError(
                "When ''load_dir'' is not provided for a checkpoint architecture, ''freeze_weights'' must be False or None."
            )

        if self.load_dir is not None:
            self._freeze_weights = self.freeze_weights or False

            checkpoint_module, self.checkpoint_config = _load_config_from_checkpoint(
                self.load_dir, self._freeze_weights
            )
            checkpoint_model = checkpoint_module.get("Architecture")
            assert self.type.lower() == checkpoint_model.get("type").lower(), (
                f"The checkpoint {checkpoint_model.get('type')} architecture does not have the correct type {self.type}"
            )
            self.config = checkpoint_model.get("config")
            warnings.warn(
                f"all architecture config overwritten by the saved model from {self.load_dir}"
            )
            if self._freeze_weights:
                warnings.warn(
                    f"Checkpoint {self.type} architecture weights will be frozen."
                )

    @classmethod
    def register(cls, name: str) -> Callable[..., ArchABC]:
        """
        Create a decorator that registers an architecture configuration.

        Parameters
        ----------
        name : str
            Name under which the architecture configuration is registered.

        Returns
        -------
        Callable[..., ArchABC]
            Decorator used to register the architecture configuration class.
        """
        return cls.registery.register(name.lower())

    @classmethod
    def available(cls):
        """
        Return the names of all registered architectures.

        Returns
        -------
        list[str]
            Names currently available in the architecture registry.
        """
        return cls.registery.available()

    def get_architecture_config(self):
        """
        Construct the selected architecture configuration.

        Checkpoint compatibility information is attached to the resulting
        configuration when the selector was initialized from a checkpoint.

        Returns
        -------
        ArchConfigABC
            Resolved architecture configuration.
        """
        architecture_config = self.registery.get(self.type.lower(), self.config)
        if self.checkpoint_config is not None:
            architecture_config._add_checkpoint_config(self.checkpoint_config)

        return architecture_config


class cVAEArchSelector(ArchSelector):
    """
    Selector for registered conditional variational autoencoder architectures.

    Uses an architecture registry independent from deterministic architectures.
    """

    registery: ClassVar[Registery] = Registery()


class deterministicArchSelector(ArchSelector):
    """
    Selector for registered deterministic architectures.

    Uses an architecture registry independent from cVAE architectures.
    """

    registery: ClassVar[Registery] = Registery()


@dataclasses.dataclass
class FlowSelector:
    """
    Select and configure a registered normalizing-flow transformation.

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
        Initialize the flow selector.
        """
        pass

    @classmethod
    def register(cls, name: str) -> Callable[..., flowABC]:
        """
        Create a decorator that registers a normalizing-flow transformation.

        Parameters
        ----------
        name : str
            Name under which the flow is registered.

        Returns
        -------
        Callable[..., flowABC]
            Decorator used to register the flow class.
        """
        return cls.registery.register(name.lower())

    @classmethod
    def available(cls):
        """
        Return the names of all registered flow transformations.

        Returns
        -------
        list[str]
            Names currently available in the flow registry.
        """
        return cls.registery.available()

    def get_flow(self):
        """
        Construct the selected normalizing-flow transformation.

        Returns
        -------
        flowABC
            Flow instance constructed from the configured arguments.
        """
        return self.registery.get(self.type.lower(), self.args)


def _load_config_from_checkpoint(
    load_path: Path | str, freeze_weights: bool = False, strict: bool = True
):
    """
    Load module configuration and compatibility metadata from a checkpoint.

    The saved module configuration is returned together with a checkpoint
    configuration containing the input and output shapes, variable metadata,
    strict-loading setting, and weight-freezing behavior.

    Parameters
    ----------
    load_path : Path | str
        Path to the checkpoint file.
    freeze_weights : bool
        Whether architecture weights loaded from the checkpoint should be frozen.
    strict : bool
        Whether checkpoint parameters must exactly match the architecture state
        dictionary when loaded.

    Returns
    -------
    tuple[dict, CheckpointConfig]
        Saved module configuration and checkpoint compatibility configuration.

    Raises
    ------
    FileNotFoundError
        If the checkpoint file does not exist.
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
