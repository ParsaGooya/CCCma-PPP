import dataclasses
from typing import Callable, ClassVar

from cccma_ppp.preprocessing.preprocessing_ABC import PreprocessModuleABC
from cccma_ppp.preprocessing.registery import Registery


@dataclasses.dataclass
class PreprocessingStepSelector:
    """
    Select and configure a registered preprocessing step.

    Parameters
    ----------
    name : str
        Name of the registered preprocessing step.
    args : dict[str, object] | None
        Arguments used to construct the preprocessing module. If ``None``, the
        module is constructed without arguments.
    """

    name: str
    args: dict[str, object] | None = None
    registery: ClassVar[Registery] = Registery()

    def __post_init__(self):
        """
        Validate and normalize the preprocessing step configuration.

        The step name is converted to lowercase and checked against the preprocessing
        registry.

        Raises
        ------
        TypeError
            If ``name`` is not a string or ``args`` is neither a dictionary nor
            ``None``.
        ValueError
            If ``name`` is empty or does not correspond to a registered preprocessing
            step.
        """
        if self.args is not None and not isinstance(self.args, dict):
            raise TypeError(
                f"Args must be dict or None, got {type(self.args).__name__}. "
                f"Provided args: {self.args}"
            )

        if not isinstance(self.name, str):
            raise TypeError(f"Step name must be string, got {type(self.name)}.")

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
        Construct the selected preprocessing module.

        Returns
        -------
        PreprocessModuleABC
            Preprocessing module constructed from the registered class and configured
            arguments.
        """
        return self.registery.get(self.name, self.args)

    @classmethod
    def register(cls, name: str) -> Callable[..., PreprocessModuleABC]:
        """
        Create a decorator that registers a preprocessing module.

        Parameters
        ----------
        name : str
            Name under which the preprocessing module is registered.

        Returns
        -------
        Callable[..., PreprocessModuleABC]
            Decorator used to register the preprocessing module class.
        """
        return cls.registery.register(name.lower())

    @classmethod
    def available(cls):
        """
        Return the names of all registered preprocessing steps.

        Returns
        -------
        list[str]
            Names of preprocessing steps available through the selector.
        """
        return cls.registery.available()
