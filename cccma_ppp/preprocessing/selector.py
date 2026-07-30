import dataclasses
from typing import Callable, ClassVar

from cccma_ppp.preprocessing.preprocessing_ABC import PreprocessModuleABC
from cccma_ppp.preprocessing.registery import Registery


@dataclasses.dataclass
class PreprocessingStepSelector:
    """
    Selector for preprocessing modules.

    Parameters
    ----------
    name : str
        Name of the registered preprocessing module.
    args : dict of str to object, optional
        Arguments used to initialize the module.
    """

    name: str
    args: dict[str, object] = dataclasses.field(default_factory=dict)
    registery: ClassVar[Registery] = Registery()

    def get_preprocessor(self):
        """
        Instantiate preprocessing module.

        Returns
        -------
        PreprocessModuleABC
            Initialized preprocessing module.
        """

        return self.registery.get(self.name.lower(), self.args)

    @classmethod
    def register(cls, name: str) -> Callable[..., PreprocessModuleABC]:
        """
        Register preprocessing module.

        Parameters
        ----------
        name : str
            Name used for registration.

        Returns
        -------
        Callable
            Decorator for registering preprocessing modules.
        """

        return cls.registery.register(name.lower())

    @classmethod
    def available(cls):
        """
        List available preprocessing modules.

        Returns
        -------
        list of str
        """

        return cls.registery.available()
