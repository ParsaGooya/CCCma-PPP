import numpy as np
import dataclasses
from typing import ClassVar
import joblib
from pathlib import Path
import os
import xarray as xr

from cccma_ppp.generic.runtime import RuntimeContext
from cccma_ppp.preprocessing.selector import PreprocessingStepSelector
from cccma_ppp.configs import supported_NN_dimensions_sorted


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
        base_data: xr.Dataset | xr.DataArray = None,
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

    def to_dataset(self, data: xr.DataArray) -> xr.Dataset:
        """
        Write the transformed data array based to dataset
        on the base dataset used for fitting the pipeline.

        Parameters
        ----------
        data : xarray.DataArray
            Transformed data.

        Returns
        -------
        xarray.DataArray
            cooridnates corrected.
        Raises
        ------
        ValueError
            If the dara array does not have at least the same number
            of dimensions as the base dataset.

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
            data.assign_coords(self.reference_coords)

        return data.assign_coords(channels=self.reference_var).to_dataset(
            dim="channels"
        )

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

    def extract_output_coords_vars(self, base_data: xr.Dataset | xr.DataArray = None):
        """
        Save the reference coordinates and variable names
        for the writer to use.

        Parameters
        ----------
        base_data : xr.DataArray
            Data on which the pipeline is fit.

        Returns
        -------
        None
        """
        if not self.fitted:
            raise ValueError(
                "Spatial coords can only be extracted for a fitted pipeline."
            )

        self.reference_coords = {
            dim: base_data[dim]
            for dim in supported_NN_dimensions_sorted
            if dim in base_data.dims
        }

        self.reference_var = list(base_data.data_vars)

    def load_from_memory(self, load_dir: str | Path):
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
        self.fitted_based_year = loaded.fitted_based_year
        self.reference_coords = loaded.reference_coords
        self.reference_var = loaded.reference_var
        self.name = loaded.name
        del loaded
        return self
