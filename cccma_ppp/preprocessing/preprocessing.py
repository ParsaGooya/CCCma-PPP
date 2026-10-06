import numpy as np
import dataclasses
import joblib
from pathlib import Path
import os
import xarray as xr
import uuid


from cccma_ppp.generic.runtime import RuntimeContext
from cccma_ppp.preprocessing.selector import PreprocessingStepSelector
from cccma_ppp.preprocessing.utils_preprocessing import Flattennanremove
from cccma_ppp.configs import supported_NN_dimensions_sorted, required_sample_dimensions

init_time_dim, lead_time_dim = required_sample_dimensions


@dataclasses.dataclass
class PreprocessingPipeline:
    """
    Manage an ordered sequence of preprocessing transformations.

    The pipeline fits preprocessing steps sequentially, applies fitted
    transformations in order, applies inverse transformations in reverse order,
    and stores reference coordinates and variables for reconstructing model
    outputs.

    Parameters
    ----------
    preprocessors_list : list[PreprocessingStepSelector]
        Ordered preprocessing steps used to construct the pipeline.
    load_dir : str | Path | None
        Path to a saved preprocessing pipeline to load. If ``None``, a new
        unfitted pipeline is initialized.
    init_time_time : str
        Name of the initialization-time dimension.
    lead_time_time : str
        Name of the lead-time dimension.
    supported_NN_dimensions : tuple[str]
        Ordered dimensions supported as neural-network output dimensions.
    """

    preprocessors_list: list[PreprocessingStepSelector] = dataclasses.field(
        default_factory=list
    )
    load_dir: str | Path = None

    init_time_time: str = dataclasses.field(init=False, default=init_time_dim)
    lead_time_time: str = dataclasses.field(init=False, default=lead_time_dim)
    supported_NN_dimensions: tuple[str] = dataclasses.field(
        init=False, default=supported_NN_dimensions_sorted
    )

    def __post_init__(self):
        """
        Initialize or load the preprocessing pipeline.

        When no saved pipeline is provided, initializes empty fitted state and
        constructs the configured preprocessing steps. Otherwise, restores the
        pipeline from ``load_dir``.
        """

        if self.load_dir is None:
            self.fitted = False
            self.mask = None
            self.fitted_based_time = None
            self.reference_coords = None
            self.reference_var = None
            self.steps = []
            self.fitted_preprocessors = []
            self.pipeline = []

            self.name = f"pipeline_{uuid.uuid4().hex[:8]}"
            self.rename_dict = None

            for step in self.preprocessors_list:
                self.pipeline.append((step.name.lower(), step.get_preprocessor()))

        else:
            self._load_from_memory(Path(self.load_dir))

    def set_name(self, name: str, rename_dict: dict[str, str] | None):
        """
        Set the pipeline name and associated renaming configuration.

        Parameters
        ----------
        name : str
            Name assigned to the preprocessing pipeline.
        rename_dict : dict[str, str] | None
            Mapping used to rename dataset dimensions, coordinates, or variables.
        """
        self.name = name
        self.rename_dict = rename_dict

    def fit(
        self,
        base_data: xr.Dataset | xr.DataArray = None,
        mask: xr.DataArray = None,
        save: bool = True,
        save_name: str | None = None,
        save_path: Path | str | None = None,
    ):
        """
        Fit each preprocessing step sequentially.

        Each preprocessor is fitted on the output of the preceding transformation.
        After fitting, reference coordinates and variables are extracted from the
        original data and the pipeline can optionally be saved to disk.

        Parameters
        ----------
        base_data : xr.Dataset | xr.DataArray
            Data used to fit the preprocessing pipeline.
        mask : xr.DataArray | None
            Optional mask supplied while fitting preprocessing steps.
        save : bool
            Whether to save the fitted pipeline to disk.
        save_name : str | None
            Filename used when saving the pipeline. If ``None``, a name is generated
            from the pipeline name.
        save_path : Path | str | None
            Directory in which the fitted pipeline is saved. If ``None``, the
            experiment preprocessing directory is used.

        Returns
        -------
        PreprocessingPipeline
            This fitted preprocessing pipeline.
        """
        if self.load_dir is None:
            data_processed = base_data
            self.mask = mask
            self.fitted_based_time = base_data[self.init_time_time].values

            for step_name, preprocessor in self.pipeline:
                preprocessor.fit(data=data_processed, mask=mask)
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

        return self

    def transform(self, data: xr.DataArray, step_arguments=None):
        """
        Apply the fitted preprocessing steps in order.

        Parameters
        ----------
        data : xr.DataArray
            Data to transform.
        step_arguments : dict | None
            Optional mapping from preprocessing step names to keyword arguments passed
            to the corresponding transformation.

        Returns
        -------
        Any
            Data after all preprocessing transformations have been applied.

        Raises
        ------
        ValueError
            If ``step_arguments`` contains a step name that is not present in the
            fitted pipeline.
        """
        self._check_fitted()

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
        Apply the fitted preprocessing steps in reverse order.

        Parameters
        ----------
        data : xr.DataArray
            Data to inverse transform.
        step_arguments : dict | None
            Optional mapping from preprocessing step names to keyword arguments passed
            to the corresponding inverse transformation.

        Returns
        -------
        Any
            Data after all inverse preprocessing transformations have been applied.

        Raises
        ------
        ValueError
            If ``step_arguments`` contains a step name that is not present in the
            fitted pipeline.
        """
        self._check_fitted()

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

    @property
    def has_flattener(self) -> bool:
        """
        Indicate whether the fitted pipeline contains a flattening preprocessor.

        Returns
        -------
        bool
            ``True`` if a ``Flattennanremove`` preprocessor is present, otherwise
            ``False``.
        """
        return any(
            isinstance(item, Flattennanremove) for item in self.fitted_preprocessors
        )

    def get_flattener(self) -> Flattennanremove:
        """
        Return the fitted flattening preprocessor.

        Returns
        -------
        Flattennanremove
            First ``Flattennanremove`` instance found in the fitted pipeline.

        Raises
        ------
        ValueError
            If the fitted pipeline does not contain a ``Flattennanremove`` step.
        """
        flattener = None
        for preprocessor in self.fitted_preprocessors:
            if isinstance(preprocessor, Flattennanremove):
                flattener = preprocessor
                break

        if flattener is None:
            raise ValueError(
                "Pipeline does not contain a Flattennanremove step. "
                "Cannot reconstruct spatial coordinates."
            )
        return flattener

    def to_dataset(self, data: xr.DataArray) -> xr.Dataset:
        """
        Reconstruct an xarray dataset from model output data.

        Output dimensions and channel coordinates are replaced with the spatial
        coordinates and variable names recorded when the pipeline was fitted. For a
        flattened representation, the stored locations from the flattening
        preprocessor are assigned instead.

        Parameters
        ----------
        data : xr.DataArray
            Model output containing a ``channels`` dimension and output dimensions.

        Returns
        -------
        xr.Dataset
            Dataset with reconstructed output coordinates and variables.

        Raises
        ------
        RuntimeError
            If the preprocessing pipeline has not been fitted.
        ValueError
            If the input does not contain a ``channels`` dimension or the number of
            output channels does not match the variables recorded during fitting.
        """

        if not self.fitted:
            raise RuntimeError("Pipeline must be fitted before calling to_dataset().")

        if "channels" not in data.dims:
            raise ValueError(
                f"Input data must have 'channels' dimension. "
                f"Got dimensions: {list(data.dims)}"
            )

        if len(data.channels) != len(self.reference_var):
            raise ValueError(
                "The dataset does not match the preprocessing pipeline."
                "make sure to use the same pipeline that was used during training."
            )

        output_dims = [dim for dim in data.dims if "output_dim_" in dim]

        if self.has_flattener:
            data = data.rename({"output_dim_0": "ref"})
            flattener = self.get_flattener()
            data = data.assign_coords(ref=flattener.final_locations)

        else:
            data = data.rename(
                {
                    dim: list(self.reference_coords)[ind]
                    for ind, dim in enumerate(output_dims)
                }
            )
            data = data.assign_coords(self.reference_coords)

        data = data.assign_coords(channels=self.reference_var).to_dataset(
            dim="channels"
        )

        return data

    def inverse_rename(self, data: xr.Dataset):
        """
        Restore names that were changed before preprocessing.

        Parameters
        ----------
        data : xr.Dataset
            Dataset whose renamed fields are to be restored.

        Returns
        -------
        xr.Dataset
            Dataset with the configured renaming mapping reversed, or the original
            dataset if no renaming mapping is configured.
        """
        if self.rename_dict is not None:
            inverse_rename = {value: key for key, value in self.rename_dict.items()}
            data = data.rename(inverse_rename)
        return data

    def get_preprocessors(self, name=None):
        """
        Return fitted preprocessing modules.

        When ``name`` is omitted, all fitted preprocessors are returned. When a name
        is provided, exactly one matching preprocessing step must exist.

        Parameters
        ----------
        name : str | None
            Name of a preprocessing step to retrieve.

        Returns
        -------
        Any
            List of all fitted preprocessors when ``name`` is ``None``, or the single
            fitted preprocessor associated with ``name``.

        Raises
        ------
        RuntimeError
            If the pipeline has not been fitted.
        ValueError
            If the requested preprocessing step does not exist or more than one step
            has the requested name.
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
        Add an already fitted preprocessor to the pipeline.

        The preprocessor is appended by default or inserted at the requested
        position.

        Parameters
        ----------
        preprocessor : Any
            Fitted preprocessing module to add.
        name : str
            Name associated with the preprocessing step.
        index : int | None
            Position at which to insert the preprocessing step. If ``None``, the step
            is appended.

        Raises
        ------
        AssertionError
            If the supplied preprocessor has not been fitted.
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
        Extract reference output coordinates and variable names.

        Coordinates corresponding to supported neural-network dimensions and the
        names of all data variables are stored for later reconstruction of model
        outputs.

        Parameters
        ----------
        base_data : xr.Dataset | xr.DataArray
            Data from which reference coordinates and variables are extracted.

        Raises
        ------
        ValueError
            If the pipeline has not been fitted, no supported neural-network
            dimensions are present, or no data variables are available.
        """
        if not self.fitted:
            raise ValueError(
                "Spatial coords can only be extracted for a fitted pipeline."
            )

        if set(base_data.dims).isdisjoint(self.supported_NN_dimensions):
            raise ValueError(
                "No supported NN dimensions found in base_data. "
                "Cannot create reference coords."
            )

        self.reference_coords = {
            dim: base_data[dim]
            for dim in self.supported_NN_dimensions
            if dim in base_data.dims
        }

        self.reference_var = list(base_data.data_vars)

        if not self.reference_var:
            raise ValueError(
                "No data variables found in base_data. "
                "Cannot create reference variables."
            )

    def load_from_memory(self, load_dir: str | Path):
        """
        Load a fitted preprocessing pipeline from disk.

        The preprocessing steps, fitted modules, fitting metadata, mask, reference
        coordinates, reference variables, and pipeline name are restored from the
        serialized pipeline.

        Parameters
        ----------
        load_dir : str | Path
            Path to the serialized preprocessing pipeline.

        Returns
        -------
        PreprocessingPipeline
            This pipeline populated with the loaded fitted state.

        Raises
        ------
        ValueError
            If the loaded preprocessing pipeline is not fitted.
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
        self.mask = loaded.mask
        self.reference_coords = loaded.reference_coords
        self.reference_var = loaded.reference_var
        self.name = loaded.name
        del loaded
        return self

    def _check_fitted(self) -> None:
        """
        Verify that the preprocessing pipeline has been fitted.

        Raises
        ------
        RuntimeError
            If the pipeline has not been fitted before transformation or inverse
            transformation is requested.
        """
        if not self.fitted:
            raise RuntimeError(
                "Pipeline must be fitted before calling transform() "
                "or inverse_transform(). Call fit() first."
            )
