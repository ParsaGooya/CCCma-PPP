import numpy as np
import dataclasses
from cccma_ppp.core.registery import *
from cccma_ppp.core.core_abc import moduleABC
from cccma_ppp.models.models_abc import *
from typing import Any, ClassVar
from collections.abc import Callable, Mapping


@dataclasses.dataclass
class ModuleSelector:
    """
    Selector for constructing registered module configurations.
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

        Raises
        ------
        ValueError
            If the specified module type is not registered.
        """

        self._module_config = self.registery.get(self.type.lower(), self.config)

    @classmethod
    def register(cls, name: str):
        """
        Register a module under a given name.

        Parameters
        ----------
        name : str
            Name used for registration.

        Returns
        -------
        callable
            Decorator for class registration.
        """
        # noqa: UP006
        return cls.registery.register(
            name.lower()
        )  ##attentin: the return is on cls.registery. This means even when register is called on an instance of StepSelector, it will still register the type on the class-level registery, which is what we want.

    @classmethod
    def available(cls):
        """
        List available registered module types.

        Returns
        -------
        list of str
            Registered module names.
        """

        return cls.registery.available()

    def build_module(self, input_shape, output_shape=None, added_features_dim=None):
        """
        Build selected module instance.

        Parameters
        ----------
        input_shape : np.ndarray
            Shape of input data.
        output_shape : np.ndarray or None, optional
            Shape of output data.
        added_features_dim : int, optional
            Dimension of additional features.

        Returns
        -------
        moduleABC
            Constructed module instance.

        Raises
        ------
        AttributeError
            If module configuration does not implement build().
        """

        return self._module_config.build(
            input_shape=input_shape,
            output_shape=output_shape,
            added_features_dim=added_features_dim,
        )


@dataclasses.dataclass
class cVAEModelSelector:
    """
    Selector for constructing registered cVAE models.
    """

    type: str
    args: Mapping[str, Any]
    registery: ClassVar[Registery] = Registery()

    def __post_init__(self):
        """
        Initialize selector instance.

        Returns
        -------
        None
        """

        pass

    @classmethod
    def register(cls, name: str):
        """
        Register a cVAE model under a given name.

        Parameters
        ----------
        name : str
            Name used for registration.

        Returns
        -------
        callable
            Decorator for class registration.
        """
        # noqa: UP006
        return cls.registery.register(
            name.lower()
        )  ##attentin: the return is on cls.registery. This means even when register is called on an instance of StepSelector, it will still register the type on the class-level registery, which is what we want.

    def available(cls):
        """
        List available registered model types.

        Returns
        -------
        list of str
            Registered model names.
        """

        return cls.registery.available()

    def get_model(self):
        """
        Retrieve and instantiate selected model.

        Returns
        -------
        cVAEmodelsABC
            Instantiated model.

        Raises
        ------
        ValueError
            If model type is not registered.
        """

        return self.registery.get(self.type.lower(), self.args)


@dataclasses.dataclass
class deterministicModelSelector:
    """
    Selector for constructing registered deterministic models.
    """

    type: str
    args: Mapping[str, Any]
    registery: ClassVar[Registery] = Registery()

    def __post_init__(self):
        """
        Initialize selector instance.

        Returns
        -------
        None
        """

        pass

    @classmethod
    def register(cls, name: str):
        """
        Register a deterministic model under a given name.

        Parameters
        ----------
        name : str
            Name used for registration.

        Returns
        -------
        callable
            Decorator for class registration.
        """
        # noqa: UP006
        return cls.registery.register(
            name.lower()
        )  ##attentin: the return is on cls.registery. This means even when register is called on an instance of StepSelector, it will still register the type on the class-level registery, which is what we want.

    def available(cls):
        """
        List available registered deterministic model types.

        Returns
        -------
        list of str
            Registered model names.
        """

        return cls.registery.available()

    def get_model(self):
        """
        Retrieve and instantiate selected deterministic model.

        Returns
        -------
        deterministicmodelsABC
            Instantiated model.

        Raises
        ------
        ValueError
            If model type is not registered.
        """

        return self.registery.get(self.type.lower(), self.args)


##### can deleter this and add the registery machinery
#  to the NormalizedFlowConfig instead. This is because
# similar to preprocessing, we need to be able to pass
# a list of flows on top of each other #######


@dataclasses.dataclass
class FlowSelector:
    """
    Selector for constructing registered flow components.
    """

    type: str
    args: dict[str, object]
    registery: ClassVar[Registery] = Registery()

    def __post_init__(self):
        """
        Initialize selector instance.

        Returns
        -------
        None
        """

        pass

    @classmethod
    def register(cls, name: str):
        """
        Register a flow component under a given name.

        Parameters
        ----------
        name : str
            Name used for registration.

        Returns
        -------
        callable
            Decorator for class registration.
        """
        # noqa: UP006
        return cls.registery.register(
            name.lower()
        )  ##attentin: the return is on cls.registery. This means even when register is called on an instance of StepSelector, it will still register the type on the class-level registery, which is what we want.

    @classmethod
    def available(cls):
        """
        List available registered flow types.

        Returns
        -------
        list of str
            Registered flow names.
        """

        return cls.registery.available()

    def get_model(self):
        """
        Retrieve and instantiate selected flow component.

        Returns
        -------
        object
            Instantiated flow component.

        Raises
        ------
        ValueError
            If flow type is not registered.
        """

        return self.registery.get(self.type.lower(), self.args)
