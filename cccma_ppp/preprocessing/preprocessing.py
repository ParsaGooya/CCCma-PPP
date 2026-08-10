import numpy as np
import dataclasses
from typing import ClassVar
import joblib
from pathlib import Path
import os
import xarray as xr

from cccma_ppp.generic.runtime import RuntimeContext
from cccma_ppp.preprocessing.selector import PreprocessingStepSelector
from cccma_ppp.configs import (supported_NN_dimensions_sorted,
                                required_sample_dimensions)

init_time_dim, lead_time_dim = required_sample_dimensions
@dataclasses.dataclass
class PreprocessingPipeline:
    """
    Document this class.

    Parameters
    ----------
    preprocessors_list : list[PreprocessingStepSelector]
        Description not yet provided.
    load_dir : str | Path
        Description not yet provided.
    """

    preprocessors_list: list[PreprocessingStepSelector] = dataclasses.field(
        default_factory=list
    )
    load_dir: str | Path = None
    num_instances: ClassVar[int] = 0


    init_time_time: str = dataclasses.field(
        init=False, default=init_time_dim
    )
    lead_time_time: str = dataclasses.field(
        init=False, default=lead_time_dim
    )
    supported_NN_dimensions: tuple[str] = dataclasses.field(
        init=False, default=supported_NN_dimensions_sorted
    )

    def __post_init__(self):
        """
        Document this function.
        """
        self.fitted = False
        self.reference_coords = None
        self.reference_var = None
        self.num_instances += 1
        if self.load_dir is None:
            self.name = f"instance_{self.num_instances}"
            self.pipeline = []
            for step in self.preprocessors_list:
                self.pipeline.append((step.name.lower(), step.get_preprocessor()))

    def set_name(self, name: str):
        """
        Document this function.

        Parameters
        ----------
        name : str
            Description not yet provided.
        """
        self.name = name

    def fit(
        self,
        base_data: xr.Dataset | xr.DataArray = None,
        mask: xr.DataArray = None,
        save: bool = True,
        save_name: str | None = None,
        save_path: Path | str | None = None,
    ):
        """
        Document this function.

        Parameters
        ----------
        base_data : xr.Dataset | xr.DataArray
            Description not yet provided.
        mask : xr.DataArray
            Description not yet provided.
        save : bool
            Description not yet provided.
        save_name : str | None
            Description not yet provided.
        save_path : Path | str | None
            Description not yet provided.

        Returns
        -------
        Any
            Description not yet provided.
        """
        if self.load_dir is None:
            data_processed = base_data
            self.fitted_based_time = base_data[self.init_time_time].values
            self.steps = []
            self.fitted_preprocessors = []

            for step_name, preprocessor in self.pipeline:
                preprocessor.fit(data_processed, mask=mask)
                data_processed = preprocessor.transform(data_processed)
                self.steps.append(step_name)
                self.fitted_preprocessors.append(preprocessor)

            self.fitted = True
            self.extract_output_coords_vars(base_data)

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
        Document this function.

        Parameters
        ----------
        data : xr.DataArray
            Description not yet provided.
        step_arguments : Any
            Description not yet provided.

        Returns
        -------
        Any
            Description not yet provided.

        Raises
        ------
        ValueError
            Description not yet provided.
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
        Document this function.

        Parameters
        ----------
        data : xr.DataArray
            Description not yet provided.
        step_arguments : Any
            Description not yet provided.

        Returns
        -------
        Any
            Description not yet provided.

        Raises
        ------
        ValueError
            Description not yet provided.
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

    def to_dataset(self, data: xr.DataArray) -> xr.Dataset:
        """
        Document this function.

        Parameters
        ----------
        data : xr.DataArray
            Description not yet provided.

        Returns
        -------
        xr.Dataset
            Description not yet provided.

        Raises
        ------
        ValueError
            Description not yet provided.
        """
        if len(data.channels) != len(self.reference_var):
            raise ValueError(
                "The dataset does not match the preprocessing pipeline."
                "make sure to use the same pipeline that was used during training."
            )

        output_dims = [dim for dim in data.dims if "output_dim_" in dim]

        from cccma_ppp.preprocessing.utils_preprocessing import Flattennanremove

        checklist = [
            isinstance(item, Flattennanremove) for item in self.fitted_preprocessors
        ]

        if any(checklist):
            data = data.rename({"output_dim_0": "ref"})
            data = data.assign_coords(
                ref=self.get_preprocessors("flattener").final_locations
            )
        else:
            data = data.rename(
                {
                    dim: list(self.reference_coords)[ind]
                    for ind, dim in enumerate(output_dims)
                }
            )
            data = data.assign_coords(self.reference_coords)

        return data.assign_coords(channels=self.reference_var).to_dataset(
            dim="channels"
        )

    def get_preprocessors(self, name=None):
        """
        Document this function.

        Parameters
        ----------
        name : Any
            Description not yet provided.

        Returns
        -------
        Any
            Description not yet provided.

        Raises
        ------
        RuntimeError
            Description not yet provided.
        ValueError
            Description not yet provided.
        """
        if not self.fitted:
            raise RuntimeError("Pipeline needs to be fitted first")

        if name is None:
            return self.fitted_preprocessors

        idx = np.argwhere(np.asarray(self.steps) == name).flatten()

        if idx.size == 0:
            raise ValueError(f"{name!r} not in preprocessing steps!")

        if idx.size > 1:
            raise ValueError(
                f"Expected exactly one preprocessor named {name!r}, "
                f"but found {idx.size} matches at indexes {idx.tolist()}."
            )

        return self.fitted_preprocessors[idx.item()]

    def add_fitted_preprocessor(self, preprocessor, name, index=None):
        """
        Document this function.

        Parameters
        ----------
        preprocessor : Any
            Description not yet provided.
        name : Any
            Description not yet provided.
        index : Any
            Description not yet provided.

        Raises
        ------
        AssertionError
            Description not yet provided.
        """
        assert preprocessor.fitted, "The preprocessor must be fitted"
        if index is None:
            self.fitted_preprocessors.append(preprocessor)
            self.steps.append(name)
        else:
            self.fitted_preprocessors.insert(index, preprocessor)
            self.steps.insert(index, name)

    def extract_output_coords_vars(self, base_data: xr.Dataset | xr.DataArray = None):
        """
        Document this function.

        Parameters
        ----------
        base_data : xr.Dataset | xr.DataArray
            Description not yet provided.

        Raises
        ------
        ValueError
            Description not yet provided.
        """
        if not self.fitted:
            raise ValueError(
                "Spatial coords can only be extracted for a fitted pipeline."
            )

        self.reference_coords = {
            dim: base_data[dim]
            for dim in self.supported_NN_dimensions
            if dim in base_data.dims
        }

        self.reference_var = list(base_data.data_vars)

    def load_from_memory(self, load_dir: str | Path):
        """
        Document this function.

        Parameters
        ----------
        load_dir : str | Path
            Description not yet provided.

        Returns
        -------
        Any
            Description not yet provided.

        Raises
        ------
        ValueError
            Description not yet provided.
        """
        loaded = joblib.load(Path(load_dir))

        if not loaded.fitted:
            raise ValueError("the preprocessor to be loaded has to be fitted first.")

        self.preprocessors_list = loaded.preprocessors_list
        self.pipeline = loaded.pipeline
        self.steps = loaded.steps
        self.fitted_preprocessors = loaded.fitted_preprocessors
        self.fitted = loaded.fitted
        self.fitted_based_time = loaded.fitted_based_time
        self.reference_coords = loaded.reference_coords
        self.reference_var = loaded.reference_var
        self.name = loaded.name
        del loaded
        return self
