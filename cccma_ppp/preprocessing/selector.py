import dataclasses
from typing import Callable, ClassVar

from cccma_ppp.preprocessing.preprocessing_ABC import PreprocessModuleABC
from cccma_ppp.preprocessing.registery import Registery


@dataclasses.dataclass
class PreprocessingStepSelector:
    """
    Document this class.

    Parameters
    ----------
    name : str
        Description not yet provided.
    args : dict[str, object]
        Description not yet provided.
    """

    name: str
    args: dict[str, object] = dataclasses.field(default_factory=dict)
    registery: ClassVar[Registery] = Registery()

    def get_preprocessor(self):
        """
        Document this function.

        Returns
        -------
        Any
            Description not yet provided.
        """
        return self.registery.get(self.name.lower(), self.args)

    @classmethod
    def register(cls, name: str) -> Callable[..., PreprocessModuleABC]:
        """
        Document this function.

        Parameters
        ----------
        name : str
            Description not yet provided.

        Returns
        -------
        Callable[..., PreprocessModuleABC]
            Description not yet provided.
        """
        return cls.registery.register(name.lower())

    @classmethod
    def available(cls):
        """
        Document this function.

        Returns
        -------
        Any
            Description not yet provided.
        """
        return cls.registery.available()
