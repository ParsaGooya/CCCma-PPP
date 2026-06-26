import numpy as np
import dataclasses
from typing import Callable, ClassVar
import joblib
from pathlib import Path
import os
import xarray as xr

from cccma_ppp.generic import RuntimeContext
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


@dataclasses.dataclass
class PreprocessingPipeline:
    """
    Sequential preprocessing pipeline, supports fitting,
    transformation, inverse transformation, and persistence.

    Parameters
    ----------
    preprocessors_list : list of PreprocessingStepSelector, optional
        List of preprocessing steps.
    load_dir : str or pathlib.Path or None, optional
        Path to load a previously fitted pipeline.
    """

    preprocessors_list: list[PreprocessingStepSelector] = dataclasses.field(
        default_factory=list
    )
    load_dir: str | Path = None
    num_instances: ClassVar[int] = 0

    def __post_init__(self):
        """
        Initialize preprocessing pipeline.

        Constructs pipeline steps or prepares for loading from disk.

        Returns
        -------
        None
        """

        self.fitted = False
        self.num_instances += 1
        if self.load_dir is None:
            self.name = f"instance_{self.num_instances}"
            self.pipeline = []
            for step in self.preprocessors_list:
                self.pipeline.append((step.name.lower(), step.get_preprocessor()))

    def set_name(self, name: str):
        """
        Set pipeline name.

        Parameters
        ----------
        name : str

        Returns
        -------
        None
        """

        self.name = name

    def fit(
        self,
        base_data: xr.DataArray = None,
        mask: xr.DataArray = None,
        save: bool = True,
        save_name: str | None = None,
        save_path: Path | str | None = None,
    ):
        """
        Fit preprocessing pipeline.

        Parameters
        ----------
        base_data : xr.DataArray, optional
            Input data used for fitting.
        mask : xr.DataArray or None, optional
            Optional mask applied during fitting.
        save : bool, optional
            Whether to save fitted pipeline.
        save_name : str or None, optional
            Filename for saving pipeline.
        save_path : pathlib.Path or str or None, optional
            Directory for saving pipeline.

        Returns
        -------
        PreprocessingPipeline
            Fitted pipeline.
        """

        if self.load_dir is None:
            data_processed = base_data
            self.fitted_based_year = base_data["year"].values
            self.steps = []
            self.fitted_preprocessors = []
            for step_name, preprocessor in self.pipeline:
                preprocessor.fit(data_processed, mask=mask)
                data_processed = preprocessor.transform(data_processed)
                self.steps.append(step_name)
                self.fitted_preprocessors.append(preprocessor)

            self.fitted = True

            if save:
                save_path = (
                    Path(save_path)
                    if save_path is not None
                    else Path(RuntimeContext.GLOBAL_EXP_DIR) / "preprocessing_pipeline"
                )
                save_name = save_name or f"{self.name}_preprocessing_pipeline.joblib"

                if not os.path.isdir(save_path):
                    os.makedirs(save_path)

                joblib.dump(self, save_path / save_name)

        else:
            self._load_from_memory(Path(self.load_dir))

        return self

    def transform(self, data: xr.DataArray, step_arguments=None):
        """
        Apply preprocessing pipeline.

        Parameters
        ----------
        data : xarray.DataArray
            Input data.
        step_arguments : dict or None, optional
            Per-step arguments for transformation.

        Returns
        -------
        xarray.DataArray
            Processed data.

        Raises
        ------
        ValueError
            If step arguments refer to unknown steps.
        """

        if step_arguments is None:
            step_arguments = dict()
        for a in step_arguments.keys():
            if a not in self.steps:
                raise ValueError(f"{a} not in preprocessing steps!")

        data_processed = data
        for step, preprocessor in zip(self.steps, self.fitted_preprocessors):
            args = dict(step_arguments.get(step, {}))
            data_processed = preprocessor.transform(data_processed, **args)

        return data_processed

    def inverse_transform(self, data: xr.DataArray, step_arguments=None):
        """
        Reverse preprocessing pipeline.

        Parameters
        ----------
        data : xarray.DataArray
            Transformed data.
        step_arguments : dict or None, optional
            Per-step arguments for inverse transformation.

        Returns
        -------
        xarray.DataArray
            Original representation.

        Raises
        ------
        ValueError
            If step arguments refer to unknown steps.
        """

        if step_arguments is None:
            step_arguments = dict()
        for a in step_arguments.keys():
            if a not in self.steps:
                raise ValueError(f"{a} not in preprocessing steps!")

        data_processed = data
        for step, preprocessor in zip(
            reversed(self.steps), reversed(self.fitted_preprocessors)
        ):
            args = dict(step_arguments.get(step, {}))
            data_processed = preprocessor.inverse_transform(data_processed, **args)
        return data_processed

    def get_preprocessors(self, name=None):
        """
        Retrieve fitted preprocessors.

        Parameters
        ----------
        name : str or None, optional
            Specific step name.

        Returns
        -------
        list or PreprocessModuleABC
            All preprocessors or a specific one.

        Raises
        ------
        RuntimeError
            If pipeline is not fitted.
        ValueError
            If step is not found.
        """

        if not self.fitted:
            raise RuntimeError("Pipeline needs to be fitted first")

        if name is None:
            return self.fitted_preprocessors
        else:
            idx = np.argwhere(np.array(self.steps) == name).flatten()
            if idx.size == 0:
                raise ValueError(f"{name} not in preprocessing steps!")
            return self.fitted_preprocessors[int(idx)]

    def add_fitted_preprocessor(self, preprocessor, name, index=None):
        """
        Add fitted preprocessor to pipeline.

        Parameters
        ----------
        preprocessor : PreprocessModuleABC
            Fitted preprocessor instance.
        name : str
            Name of the step.
        index : int or None, optional
            Position to insert step.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If preprocessor is not fitted.
        """

        assert preprocessor.fitted, "The preprocessor must be fitted"
        if index is None:
            self.fitted_preprocessors.append(preprocessor)
            self.steps.append(name)
        else:
            self.fitted_preprocessors.insert(index, preprocessor)
            self.steps.insert(index, name)

    def _load_from_memory(self, load_dir: str | Path):
        """
        Load fitted pipeline from disk.

        Parameters
        ----------
        load_dir : str or pathlib.Path
            Path to saved pipeline.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If loaded pipeline is not fitted.
        """

        loaded = joblib.load(Path(load_dir))

        if not loaded.fitted:
            raise ValueError("the preprocessor to be loaded has to be fitted first.")

        self.preprocessors_list = loaded.preprocessors_list
        self.pipeline = loaded.pipeline
        self.steps = loaded.steps
        self.fitted_preprocessors = loaded.fitted_preprocessors
        self.fitted = loaded.fitted
        self.name = loaded.name
        del loaded
