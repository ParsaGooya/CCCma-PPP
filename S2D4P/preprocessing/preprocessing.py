import numpy as np
import dataclasses
import joblib
from pathlib import Path
import os
import xarray as xr
import uuid


from S2D4P.generic.runtime import RuntimeContext
from S2D4P.preprocessing.selector import PreprocessingStepSelector
from S2D4P.preprocessing.utils_preprocessing import Flattennanremove
from S2D4P.configs import supported_NN_dimensions_sorted, required_sample_dimensions

init_time_dim, lead_time_dim = required_sample_dimensions


@dataclasses.dataclass
class PreprocessingPipeline:
    """
    Manage an ordered sequence of preprocessing transformations.

    The pipeline constructs preprocessing modules from configured selectors,
    fits them sequentially, applies forward and inverse transformations, stores
    reference output coordinates and variables, and supports serialization of
    fitted preprocessing state.

    Parameters
    ----------
    preprocessors_list : list[PreprocessingStepSelector]
        Ordered preprocessing steps used to construct the pipeline.
    load_dir : str | Path | None
        Path to a previously saved preprocessing pipeline. If provided, fitted
        pipeline state is restored during initialization.

    Notes
    ---------
    Below arguments are configured to be read from configs and not specified
    by user. This can be changed if adopting the code for other pipelines.

    init_time_dim : str
        Name of the initialization-time dimension used by the pipeline.
    lead_time_dim : str
        Name of the lead-time dimension used by the pipeline.
    supported_NN_dimensions : tuple[str, ...]
        Ordered dimensions supported as neural-network output dimensions.
    """

    preprocessors_list: list[PreprocessingStepSelector] = dataclasses.field(
        default_factory=list
    )
    load_dir: str | Path = None

    init_time_dim: str = dataclasses.field(init=False, default=init_time_dim)
    lead_time_dim: str = dataclasses.field(init=False, default=lead_time_dim)
    supported_NN_dimensions: tuple[str] = dataclasses.field(
        init=False, default=supported_NN_dimensions_sorted
    )

    def __post_init__(self):
        """
        Initialize or restore the preprocessing pipeline.

        When no saved pipeline is provided, preprocessing modules are constructed
        from ``preprocessors_list`` and an unfitted pipeline state is initialized.
        Otherwise, fitted state is restored from ``load_dir``.
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
        Set the pipeline name and coordinate-renaming information.

        Parameters
        ----------
        name : str
            Name used to identify the pipeline when saving and loading fitted state.
        rename_dict : dict[str, str] | None
            Mapping from original data names to standardized pipeline names.
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
        Fit the preprocessing pipeline sequentially.

        Each preprocessing module is fitted on the output of the preceding
        transformation. Reference output coordinates and variables are recorded after
        all steps are fitted, and the fitted pipeline can optionally be serialized.

        Parameters
        ----------
        base_data : xr.Dataset | xr.DataArray
            Data used to fit the preprocessing pipeline.
        mask : xr.DataArray | None
            Optional mask applied while fitting preprocessing statistics.
        save : bool
            Whether to save the fitted pipeline to disk.
        save_name : str | None
            Filename used when saving the pipeline. If ``None``, a name is generated
            from the pipeline identifier.
        save_path : Path | str | None
            Directory in which the fitted pipeline is saved. If ``None``, the
            experiment preprocessing directory is used.

        Returns
        -------
        PreprocessingPipeline
            This fitted preprocessing pipeline.
        """
        if self.fitted:
            self._reset_fit()

        if self.load_dir is None:
            data_processed = base_data
            self.mask = mask
            self.fitted_based_time = (
                base_data[self.init_time_dim].values
                if self.init_time_dim in base_data.dims
                else None
            )

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
        Apply all fitted preprocessing steps in order.

        Parameters
        ----------
        data : xr.DataArray
            Data to transform.
        step_arguments : dict | None
            Mapping from preprocessing step names to keyword arguments supplied to
            that step's ``transform`` method.

        Returns
        -------
        xr.DataArray
            Data after applying all preprocessing transformations.

        Raises
        ------
        ValueError
            If ``step_arguments`` contains a preprocessing step that is not part of
            the fitted pipeline.
        RuntimeError
            If the pipeline has not been fitted.
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
        Apply inverse preprocessing transformations in reverse order.

        Parameters
        ----------
        data : xr.DataArray
            Preprocessed data to restore.
        step_arguments : dict | None
            Mapping from preprocessing step names to keyword arguments supplied to
            that step's ``inverse_transform`` method.

        Returns
        -------
        xr.DataArray
            Data after reversing all preprocessing transformations.

        Raises
        ------
        ValueError
            If ``step_arguments`` contains a preprocessing step that is not part of
            the fitted pipeline.
        RuntimeError
            If the pipeline has not been fitted.
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
            ``True`` when a ``Flattennanremove`` step is present.
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
            First ``Flattennanremove`` instance in the fitted pipeline.

        Raises
        ------
        ValueError
            If the pipeline does not contain a ``Flattennanremove`` step.
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
        Reconstruct a dataset from channel-first model output.

        Output dimensions are restored to their reference spatial coordinates, or to
        the flattened reference locations when a flattening preprocessor is present.
        The channel dimension is then converted back to the original data variables.

        Parameters
        ----------
        data : xr.DataArray
            Model output containing a ``channels`` dimension and output dimensions.

        Returns
        -------
        xr.Dataset
            Dataset with the reference variables, dimensions, and coordinates restored.

        Raises
        ------
        RuntimeError
            If the preprocessing pipeline has not been fitted.
        ValueError
            If the input lacks a ``channels`` dimension or its channel count does not
            match the variables associated with the fitted pipeline.
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
        Restore original names from the pipeline renaming configuration.

        Parameters
        ----------
        data : xr.Dataset
            Dataset using standardized pipeline names.

        Returns
        -------
        xr.Dataset
            Dataset with original names restored when a renaming configuration is
            available.
        """
        if self.rename_dict is not None:
            inverse_rename = {value: key for key, value in self.rename_dict.items()}
            data = data.rename(inverse_rename)
        return data

    def get_preprocessors(self, name=None):
        """
        Return fitted preprocessing modules.

        When ``name`` is omitted, all fitted preprocessors are returned. Otherwise,
        exactly one fitted preprocessor with the requested step name is returned.

        Parameters
        ----------
        name : str | None
            Name of a preprocessing step to retrieve.

        Returns
        -------
        list | Any
            All fitted preprocessing modules when ``name`` is ``None``, otherwise the
            single matching preprocessor.

        Raises
        ------
        RuntimeError
            If the pipeline has not been fitted.
        ValueError
            If the requested step does not exist or occurs more than once.
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
        Add a fitted preprocessing module to the pipeline.

        Parameters
        ----------
        preprocessor : Any
            Fitted preprocessing module to add.
        name : str
            Name associated with the preprocessing step.
        index : int | None
            Position at which to insert the preprocessor. If ``None``, the
            preprocessor is appended.

        Raises
        ------
        AssertionError
            If ``preprocessor`` has not been fitted.
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
        Record reference output coordinates and variables.

        Supported neural-network dimensions are extracted from the fitting data and
        stored for later reconstruction of model outputs.

        Parameters
        ----------
        base_data : xr.Dataset | xr.DataArray
            Data from which reference coordinates and variable names are extracted.

        Raises
        ------
        ValueError
            If the pipeline is not fitted, no supported neural-network dimensions are
            present, or the fitting data contain no data variables.
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
        Restore a fitted preprocessing pipeline from disk.

        Fitted preprocessing modules, step names, masks, fitting times, reference
        coordinates, reference variables, and pipeline metadata are copied from the
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
            If the serialized preprocessing pipeline has not been fitted.
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
        Validate pipeline is fitted.

        Raises
        ------
        RuntimeError
            If pipeline not fitted.
        """
        if not self.fitted:
            raise RuntimeError(
                "Pipeline must be fitted before calling transform() "
                "or inverse_transform(). Call fit() first."
            )

    def _reset_fit(self):
        """Reset fitted state before fitting the pipeline again."""
        self.fitted = False
        self.steps = []
        self.fitted_preprocessors = []
        self.mask = None
        self.fitted_based_time = None
