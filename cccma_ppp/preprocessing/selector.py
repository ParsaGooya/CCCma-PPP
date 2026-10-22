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
    args : dict[str, object] | None
        Description not yet provided.
    """

    name: str
    args: dict[str, object] | None = None
    registery: ClassVar[Registery] = Registery()

    def __post_init__(self):
        if self.args is not None and not isinstance(self.args, dict):
            raise TypeError(
                f"Args must be dict or None, got {type(self.args).__name__}. "
                f"Provided args: {self.args}"
            )

        if not isinstance(self.name, str):
            raise TypeError(
                f"Step name must be string, got {type(self.name)}."
            )

        if not self.name:
            raise ValueError("Step name cannot be empty.")

        self.name = self.name.lower()

        available = self.registery.available()
        if self.name not in available:
            raise ValueError(
                f"Preprocessing step '{self.name}' is not registered. "
                f"Available: {available}"
            )


    def get_preprocessor(self):
        """
        Document this function.

        Returns
        -------
        Any
            Description not yet provided.
        """
        return self.registery.get(self.name, self.args)

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
