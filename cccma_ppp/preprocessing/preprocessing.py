import numpy as np
import dataclasses
from typing import ClassVar
import joblib
from pathlib import Path
import os


from cccma_ppp.preprocessing.registery import Registery


@dataclasses.dataclass
class PreprocessingStepSelector:
    """
    Selector for retrieving preprocessing modules from the registry.
    """

    name: str
    args: dict[str, object] = dataclasses.field(default_factory=dict)
    registery: ClassVar[Registery] = Registery()

    def get_preprocessor(self):
        """
        Retrieve preprocessing module instance.

        Returns
        -------
        PreprocessModuleABC
            Instantiated preprocessing module.

        Raises
        ------
        ValueError
            If the specified preprocessing step is not registered.
        """

        return self.registery.get(self.name.lower(), self.args)

    @classmethod
    def register(cls, name: str):
        """
        Register a preprocessing module under a given name.

        Parameters
        ----------
        name : str
            Name used for registration.

        Returns
        -------
        callable
            Decorator for registering preprocessing classes.
        """

        return cls.registery.register(name.lower())

    @classmethod
    def available(cls):
        """
        List available preprocessing modules.

        Returns
        -------
        list of str
            Registered preprocessing step names.
        """

        return cls.registery.available()


@dataclasses.dataclass
class PreprocessingPipeline:
    """
    Pipeline for sequential application of preprocessing modules.
    """

    preprocessors_list: list[PreprocessingStepSelector] = dataclasses.field(
        default_factory=list
    )
    load_dir: str | Path = None
    load_name: str = None
    num_instances: ClassVar[int] = 0

    def __post_init__(self):
        """
        Initialize pipeline structure and internal state.

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

    def fit(self, base_data=None, mask=None, save=True, save_name=None, save_path=None):
        """
        Fit all preprocessing steps sequentially and optionally save pipeline.

        Parameters
        ----------
        base_data : xr.DataArray, optional
            Input data used for fitting preprocessing steps.
        mask : xr.DataArray, optional
            Mask applied during preprocessing.
        save : bool, optional
            Whether to save fitted pipeline.
        save_name : str, optional
            Filename for saving.
        save_path : Path or str, optional
            Directory for saving.

        Returns
        -------
        PreprocessingPipeline
            Fitted pipeline instance.

        Raises
        ------
        AssertionError
            If loading pipeline is not properly fitted.
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
                    else Path(os.environ["GLOBAL_EXP_DIR"]) / "preprocessing_pipeline"
                )
                save_name = save_name or f"{self.name}_preprocessing_pipeline.joblib"

                if not os.path.isdir(save_path):
                    os.makedirs(save_path)

                joblib.dump(self, save_path / save_name)
        else:
            self._load_from_memory(Path(self.load_dir), self.load_name)

        return self

    def transform(self, data, step_arguments=None):
        """
        Apply preprocessing steps to data.

        Parameters
        ----------
        data : object
            Input data.
        step_arguments : dict, optional
            Step-specific arguments.

        Returns
        -------
        object
            Transformed data.

        Raises
        ------
        ValueError
            If step arguments contain invalid step names.
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

    def inverse_transform(self, data, step_arguments=None):
        """
        Apply inverse preprocessing transformations.

        Parameters
        ----------
        data : object
            Transformed data.
        step_arguments : dict, optional
            Step-specific arguments.

        Returns
        -------
        object
            Reconstructed data.

        Raises
        ------
        ValueError
            If step arguments contain invalid step names.
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
        Retrieve fitted preprocessing modules.

        Parameters
        ----------
        name : str, optional
            Name of a specific preprocessing step.

        Returns
        -------
        list or PreprocessModuleABC
            All preprocessors or selected one.

        Raises
        ------
        AssertionError
            If pipeline is not fitted.
        ValueError
            If specified step is not found.
        """

        assert self.fitted, "Pipeline needs to be fitted first"

        if name is None:
            return self.fitted_preprocessors
        else:
            idx = np.argwhere(np.array(self.steps) == name).flatten()
            if idx.size == 0:
                raise ValueError(f"{name} not in preprocessing steps!")
            return self.fitted_preprocessors[int(idx)]

    def add_fitted_preprocessor(self, preprocessor, name, index=None):
        """
        Add a fitted preprocessor to the pipeline.

        Parameters
        ----------
        preprocessor : PreprocessModuleABC
            Fitted preprocessor instance.
        name : str
            Name of the preprocessing step.
        index : int, optional
            Position to insert the step.

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

    def _load_from_memory(self, load_dir, load_name=None):
        """
        Load preprocessing pipeline from disk.

        Parameters
        ----------
        load_dir : str or Path
            Directory containing saved pipeline.
        load_name : str, optional
            Filename of saved pipeline.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If loaded pipeline is not fitted.
        """

        load_name = load_name or f"{self.name}_preprocessing_pipeline.joblib"
        loaded = joblib.load(Path(load_dir) / load_name)

        assert loaded.fitted, "the preprocessor to be loaded has to be fitted first."

        self.preprocessors_list = loaded.preprocessors_list
        self.pipeline = loaded.pipeline
        self.steps = loaded.steps
        self.fitted_preprocessors = loaded.fitted_preprocessors
        self.fitted = loaded.fitted
        self.name = loaded.name
        del loaded
