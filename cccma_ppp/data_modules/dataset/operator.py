import numpy as np
import datetime
import cftime
from pathlib import Path
import xarray as xr
from collections.abc import Sequence

from cccma_ppp.data_modules.data.data_abc import DataConfigABC
from cccma_ppp.data_modules.dataset.dataset_abc import DatasetConfigABC
from cccma_ppp.data_modules.utils import  _validate_time_sequence
from cccma_ppp.data_modules.weights import WeightsConfig
from cccma_ppp.preprocessing.preprocessing_ABC import PreprocessModuleABC


class DatasetOperator:
    """
    Class for managing dataset-level operations.

    Parameters
    ----------
    config : DatasetConfigABC
        Dataset configuration.
    """

    def __init__(self, config: DatasetConfigABC):
        """
        Initialize dataset operator.

        Parameters
        ----------
        config : DatasetConfigABC

        Returns
        -------
        None
        """
        self.config = config

    @property
    def config_observation(self):
        """
        Observation dataset configuration if available.

        Returns
        -------
        ObsDataConfig or None
            Observation configuration if present, otherwise None.
        """
        if hasattr(self.config, "observation"):
            return self.config.observation

    def fit_preprocessors(
        self,
        train_times: (Sequence[np.datetime64 | datetime.datetime | cftime.datetime]
            | np.ndarray
            | xr.DataArray
            | slice
        ),
        save=False,
        save_path: Path | str | None = None,
        save_name: str | None = None,
    ):
        """
        Fit preprocessing pipelines for all datasets.

        Parameters
        ----------
        train_times : array-like
            Training times used for fitting. Values must be either NumPy
            datetime64 or cftime datetime objects.
        save : bool, optional
            Whether to persist fitted pipelines.
        save_path : pathlib.Path or str or None, optional
        save_name : str or None, optional

        Returns
        -------
        None

        Notes
        -----
        Applies fitting to model, observation, and condition datasets.
        """

        if not isinstance(train_times, slice):
            _validate_time_sequence(train_times)

        if self.config.model is not None:
            selection = {
                self.config.init_time_dim: train_times,
                self.config.lead_time_dim: self.config.model.info.coords[self.config.lead_time_dim],
            }
            if self.config.model.info.coords.get(self.config.realization_dim) is not None:
                selection[self.config.realization_dim] = self.config.model.info.coords[self.config.realization_dim]

            self.config.model.fit_preprocessor_pipeline(
                selection=selection,
                mask=True,
                save=save,
                save_path=save_path,
                save_name=save_name,
            )

        if self.config_observation is not None:
            selection = {self.config.init_time_dim: train_times}
            if self.config_observation.info.coords.get(self.config.realization_dim) is not None:
                selection[self.config.realization_dim] = self.config_observation.info.coords[
                    self.config.realization_dim
                ]

            self.config_observation.fit_preprocessor_pipeline(
                selection=selection, save=save, save_path=save_path, save_name=save_name
            )

        if self.config.effective_condition is not None:
            if self.config.condition_method.lower() == "static":
                selection = {}
            else:
                selection = {
                    self.config.init_time_dim: train_times,
                    self.config.lead_time_dim: self.config.effective_condition.info.coords[
                        self.config.lead_time_dim
                    ],
                }
                if (
                    self.config.effective_condition.info.coords.get(self.config.realization_dim)
                    is not None
                ):
                    selection[self.config.realization_dim] = (
                        self.config.effective_condition.info.coords[self.config.realization_dim]
                    )

            self.config.effective_condition.fit_preprocessor_pipeline(
                selection=selection,
                mask=True,
                save=save,
                save_path=save_path,
                save_name=save_name,
            )

        self.config._fitted_preprocessors = True

    def load_fitted_preprocessors(self, load_dir: Path | str | None = None):
        """
        Load fitted preprocessing pipelines.

        Parameters
        ----------
        load_dir : pathlib.Path or str or None

        Returns
        -------
        None
        """

        if self.config.model is not None:
            self.config.model.load_preprocessor_pipeline(load_dir)

        if self.config_observation is not None:
            self.config_observation.load_preprocessor_pipeline(load_dir)

        if self.config.effective_condition is not None:
            self.config.effective_condition.load_preprocessor_pipeline(load_dir)

        self.config._fitted_preprocessors = True

    def add_fitted_preprocessor(self, preprocessor: PreprocessModuleABC, index=0):
        """
        Add a fitted preprocessor to all relevant pipelines.

        Parameters
        ----------
        preprocessor : PreprocessModuleABC
        index : int, optional

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If preprocessor type is invalid.
        AssertionError
            If preprocessor is not fitted.
        """

        if not isinstance(preprocessor, PreprocessModuleABC):
            raise TypeError(
                f"preprocessor must be an instance of ProcessorConfig, "
                f"got {type(preprocessor)}"
            )
        assert preprocessor.fitted, "The preprocessor must be fitted"

        if self.config.model is not None:
            self.config.model.preprocessing_pipeline.add_fitted_preprocessor(
                preprocessor, index=index
            )
        if self.config_observation is not None:
            self.config_observation.preprocessing_pipeline.add_fitted_preprocessor(
                preprocessor, index=index
            )
        if self.config.effective_condition is not None:
            self.config.effective_condition.preprocessing_pipeline.add_fitted_preprocessor(
                preprocessor, index=index
            )

    def get_weights(
        self,
        config: WeightsConfig | None = None,
        save: bool = True,
        save_path: Path | str | None = None,
        save_name: str | None = None,
    ) -> xr.DataArray:
        """
        Compute spatial and variable weights.

        Parameters
        ----------
        config : WeightsConfig or None, optional
            Configuration controlling weight computation.
        save : bool, optional
            Whether to save computed weights.
        save_path : pathlib.Path or str or None, optional
            Directory where weights should be saved.
        save_name : str or None, optional
            Filename for saved weights.

        Returns
        -------
        xr.DataArray
            Computed spatial and variable weights.

        Raises
        ------
        ValueError
            If no valid dataset is available.
        RuntimeError
            If variable weights do not match expected variables.
        """
        if config is None:
            config = WeightsConfig()

        if self.config_observation is not None:
            ref = self.config_observation
        elif self.config.model is not None:
            ref = self.config.model
        else:
            raise ValueError(
                "No model or observation data is availablle. "
                "Weights could not be generated"
            )

        target_coords = {}
        for dim in [
            dim for dim in self.config.supported_NN_dimensions if dim in ref.info.coords
        ]:
            target_coords[dim] = ref.info.coords[dim]

        from cccma_ppp.preprocessing.utils_preprocessing import Flattennanremove

        checklist = [
            isinstance(item, Flattennanremove)
            for item in ref.preprocessing_pipeline.fitted_preprocessors
        ]

        weights = config.build_weights(
            target_coords,
            Flattennanremover=ref.preprocessing_pipeline.get_preprocessors("flattener")
            if any(checklist)
            else None,
            save=save,
            save_path=save_path,
            save_name=save_name,
        )

        if "channels" in weights.dims:
            error_msg = f"inconsistent variable weights {weights.channels.values} for output variables {ref.names}"
            if not np.array_equal(weights.channels.values, self.ref.names):
                raise RuntimeError(error_msg)

        return weights

    def get_input_var_metadata(self) -> dict:
        """
        Retrieve metadata for input variables.

        Returns
        -------
        dict
            Variable names and preprocessing steps.
        """

        metadata = dict(variables=list(), preprocessors=list())
        NN_dims = []

        if self.config.effective_condition is None:
            metadata = self._update_metadata_with_dataconfig_metadata(
                metadata, self.config.model
            )

            for dim in [
                dim
                for dim in self.config.supported_NN_dimensions
                if dim in self.config.model.info.coords
            ]:
                NN_dims.append(dim)

        else:
            if not self.config._using_model_data_as_condition:
                metadata = self._update_metadata_with_dataconfig_metadata(
                    metadata, self.config.model
                )
                metadata = self._update_metadata_with_dataconfig_metadata(
                    metadata, self.config.effective_condition
                )
            else:
                metadata = self._update_metadata_with_dataconfig_metadata(
                    metadata, self.config.effective_condition
                )

            for dim in [
                dim
                for dim in self.config.supported_NN_dimensions
                if dim in self.config.effective_condition.info.coords
            ]:
                NN_dims.append(dim)

        metadata["NN_dims"] = ["channels"] + NN_dims

        return metadata

    def get_target_var_metadata(self):
        """
        Retrieve metadata for target variables.

        Returns
        -------
        dict

        Raises
        ------
        ValueError
            If no target data is available.
        """

        metadata = dict(variables=list(), preprocessors=list())
        NN_dims = []

        if self.config_observation is None:
            if self.config.model is None:
                raise ValueError(
                    "No model or observation data is availablle. "
                    "target variable metadata could not be generated"
                )

            metadata = self._update_metadata_with_dataconfig_metadata(
                metadata, self.config.model
            )

            for dim in [
                dim
                for dim in self.config.supported_NN_dimensions
                if dim in self.config.model.info.coords
            ]:
                NN_dims.append(dim)

        else:
            metadata = self._update_metadata_with_dataconfig_metadata(
                metadata, self.config_observation
            )

            for dim in [
                dim
                for dim in self.config.supported_NN_dimensions
                if dim in self.config_observation.info.coords
            ]:
                NN_dims.append(dim)

        metadata["NN_dims"] = ["channels"] + NN_dims

        return metadata

    def _update_metadata_with_dataconfig_metadata(
        self, metadata: dict, dataconfig: DataConfigABC
    ):
        """
        Update metadata with dataset configuration information.

        Parameters
        ----------
        metadata : dict
        dataconfig : DataConfigABC

        Returns
        -------
        dict
        """
        preprocessor_names = [
            processor[0].lower()
            for processor in dataconfig.preprocessing_pipeline.pipeline
        ]
        for var in dataconfig.names:
            metadata["variables"].append(var)
            metadata["preprocessors"].append(preprocessor_names)

        return metadata


def _build_chunks(config: DataConfigABC | None = None):

    if config is None:
        return
    required_sample_dimensions = (config.init_time_dim, config.lead_time_dim)
    sample_dims = (*required_sample_dimensions, config.realization_dim)

    chunks = {dim: 1 for dim in sample_dims if dim in config.info.coords}

    return chunks
