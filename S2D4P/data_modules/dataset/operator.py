import numpy as np
import datetime
import cftime
from pathlib import Path
import xarray as xr
from collections.abc import Sequence

from S2D4P.data_modules.data.data_abc import DataConfigABC
from S2D4P.data_modules.dataset.dataset_abc import DatasetConfigABC
from S2D4P.data_modules.utils import _validate_time_sequence
from S2D4P.data_modules.weights import WeightsConfig
from S2D4P.preprocessing.preprocessing_ABC import PreprocessModuleABC


class DatasetOperator:
    """
    Document this class.

    Parameters
    ----------
    config : DatasetConfigABC
        Description not yet provided.
    """

    def __init__(self, config: DatasetConfigABC):
        """
        Document this function.

        Parameters
        ----------
        config : DatasetConfigABC
            Description not yet provided.
        """
        self.config = config

    @property
    def config_reference(self):
        """
        Document this function.

        Returns
        -------
        Any
            Description not yet provided.
        """
        if hasattr(self.config, "reference"):
            return self.config.reference

    def fit_preprocessors(
        self,
        train_times: (
            Sequence[np.datetime64 | datetime.datetime | cftime.datetime]
            | np.ndarray
            | xr.DataArray
        ),
        save=False,
        save_path: Path | str | None = None,
        save_name: str | None = None,
    ):
        """
        Document this function.

        Parameters
        ----------
        train_times : Sequence[np.datetime64 | datetime.datetime | cftime.datetime] | np.ndarray | xr.DataArray
            Description not yet provided.
        save : Any
            Description not yet provided.
        save_path : Path | str | None
            Description not yet provided.
        save_name : str | None
            Description not yet provided.

        Raises
        ------
        ValueError
            Description not yet provided.
        """

        train_times_array = np.asarray(train_times)

        _validate_time_sequence(train_times)

        missing = [t for t in train_times_array if t not in self.config.available_times]

        if missing:
            raise ValueError(f"The following train_times are unavailable: {missing}")

        if self.config.forecast is not None:
            selection = {
                self.config.init_time_dim: self.config.get_input_times(train_times),
                self.config.lead_time_dim: self.config.forecast.coords[
                    self.config.lead_time_dim
                ],
            }
            if self.config.forecast.coords.get(self.config.realization_dim) is not None:
                selection[self.config.realization_dim] = self.config.forecast.coords[
                    self.config.realization_dim
                ]

            self.config.forecast.fit_preprocessor_pipeline(
                selection=selection,
                mask=True,
                save=save,
                save_path=save_path,
                save_name=save_name,
            )

        if self.config_reference is not None:
            selection = {self.config.init_time_dim: train_times}
            if (
                self.config_reference.coords.get(self.config.realization_dim)
                is not None
            ):
                selection[self.config.realization_dim] = self.config_reference.coords[
                    self.config.realization_dim
                ]

            self.config_reference.fit_preprocessor_pipeline(
                selection=selection, save=save, save_path=save_path, save_name=save_name
            )

        if self.config.effective_condition is not None:
            if self.config.effective_condition_method == "static":
                selection = {}
            else:
                selection = {
                    self.config.init_time_dim: self.config.get_input_times(train_times),
                    self.config.lead_time_dim: self.config.effective_condition.coords[
                        self.config.lead_time_dim
                    ],
                }
                if (
                    self.config.effective_condition.coords.get(
                        self.config.realization_dim
                    )
                    is not None
                ):
                    selection[self.config.realization_dim] = (
                        self.config.effective_condition.coords[
                            self.config.realization_dim
                        ]
                    )

            self.config.effective_condition.fit_preprocessor_pipeline(
                selection=selection,
                mask=True,
                save=save,
                save_path=save_path,
                save_name=save_name,
            )

        self.config._fitted_preprocessors = self._fitted_preprocessors

    def load_fitted_preprocessors(self, load_dir: Path | str | None = None):
        """
        Document this function.

        Parameters
        ----------
        load_dir : Path | str | None
            Description not yet provided.
        """
        if self.config.forecast is not None:
            self.config.forecast.load_preprocessor_pipeline(load_dir)

        if self.config_reference is not None:
            self.config_reference.load_preprocessor_pipeline(load_dir)

        if self.config.effective_condition is not None:
            self.config.effective_condition.load_preprocessor_pipeline(load_dir)

        self.config._fitted_preprocessors = self._fitted_preprocessors

    def add_fitted_preprocessor(self, preprocessor: PreprocessModuleABC, index=0):
        """
        Document this function.

        Parameters
        ----------
        preprocessor : PreprocessModuleABC
            Description not yet provided.
        index : Any
            Description not yet provided.

        Raises
        ------
        AssertionError
            Description not yet provided.
        TypeError
            Description not yet provided.
        """
        if not isinstance(preprocessor, PreprocessModuleABC):
            raise TypeError(
                f"preprocessor must be an instance of PreprocessModuleABC, "
                f"got {type(preprocessor)}"
            )

        if not preprocessor.fitted:
            raise ValueError("The preprocessor must be fitted before adding.")

        if self.config.forecast is not None:
            self.config.forecast.preprocessing_pipeline.add_fitted_preprocessor(
                preprocessor, index=index
            )
        if self.config_reference is not None:
            self.config_reference.preprocessing_pipeline.add_fitted_preprocessor(
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
        Document this function.

        Parameters
        ----------
        config : WeightsConfig | None
            Description not yet provided.
        save : bool
            Description not yet provided.
        save_path : Path | str | None
            Description not yet provided.
        save_name : str | None
            Description not yet provided.

        Returns
        -------
        xr.DataArray
            Description not yet provided.

        Raises
        ------
        RuntimeError
            Description not yet provided.
        ValueError
            Description not yet provided.
        """
        if config is None:
            config = WeightsConfig()

        if self.config_reference is not None:
            ref = self.config_reference
        elif self.config.forecast is not None:
            ref = self.config.forecast
        else:
            raise ValueError(
                "No forecast or reference data is availablle. "
                "Weights could not be generated"
            )

        target_coords = {}
        for dim in [
            dim for dim in self.config.supported_NN_dimensions if dim in ref.coords
        ]:
            target_coords[dim] = ref.coords[dim]

        if ref.preprocessing_pipeline.has_flattener:
            flattener = ref.preprocessing_pipeline.get_flattener()
        else:
            flattener = None

        weights = config.build_weights(
            target_coords,
            Flattennanremover=flattener,
            save=save,
            save_path=save_path,
            save_name=save_name,
        )

        if "channels" in weights.dims:
            error_msg = f"inconsistent variable weights {weights.channels.values} for output variables {ref.names}"
            if not np.array_equal(weights.channels.values, ref.names):
                raise RuntimeError(error_msg)

        return weights

    def get_input_var_metadata(self) -> dict:
        """
        Document this function.

        Returns
        -------
        dict
            Description not yet provided.
        """
        metadata = dict(variables=list(), preprocessors=list())
        NN_dims = []

        if self.config.effective_condition is None:
            metadata = self._update_metadata_with_dataconfig_metadata(
                metadata, self.config.forecast
            )

            for dim in [
                dim
                for dim in self.config.supported_NN_dimensions
                if dim in self.config.forecast.coords
            ]:
                NN_dims.append(dim)

        else:
            
            if not self.config._using_forecast_data_as_condition:
                
                metadata = self._update_metadata_with_dataconfig_metadata(
                    metadata, self.config.forecast
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
                if dim in self.config.effective_condition.coords
            ]:
                NN_dims.append(dim)

        metadata["NN_dims"] = ["channels"] + NN_dims

        return metadata

    def get_target_var_metadata(self):
        """
        Document this function.

        Returns
        -------
        Any
            Description not yet provided.

        Raises
        ------
        ValueError
            Description not yet provided.
        """
        metadata = dict(variables=list(), preprocessors=list())
        NN_dims = []

        if self.config_reference is None:
            if self.config.forecast is None:
                raise ValueError(
                    "No forecast or reference data is availablle. "
                    "target variable metadata could not be generated"
                )

            metadata = self._update_metadata_with_dataconfig_metadata(
                metadata, self.config.forecast
            )

            for dim in [
                dim
                for dim in self.config.supported_NN_dimensions
                if dim in self.config.forecast.coords
            ]:
                NN_dims.append(dim)

        else:
            metadata = self._update_metadata_with_dataconfig_metadata(
                metadata, self.config_reference
            )

            for dim in [
                dim
                for dim in self.config.supported_NN_dimensions
                if dim in self.config_reference.coords
            ]:
                NN_dims.append(dim)

        metadata["NN_dims"] = ["channels"] + NN_dims

        return metadata

    def _update_metadata_with_dataconfig_metadata(
        self, metadata: dict, dataconfig: DataConfigABC
    ):
        """
        Document this function.

        Parameters
        ----------
        metadata : dict
            Description not yet provided.
        dataconfig : DataConfigABC
            Description not yet provided.

        Returns
        -------
        Any
            Description not yet provided.
        """
        preprocessor_names = [
            processor[0].lower()
            for processor in dataconfig.preprocessing_pipeline.pipeline
        ]
        for var in dataconfig.names:
            metadata["variables"].append(var)
            metadata["preprocessors"].append(preprocessor_names)

        return metadata

    @property
    def _fitted_preprocessors(self) -> bool:
        """Check if all active pipelines are fitted."""
        pipelines = []
        if self.config.forecast is not None:
            pipelines.append(self.config.forecast.preprocessing_pipeline.fitted)
        if self.config_reference is not None:
            pipelines.append(self.config_reference.preprocessing_pipeline.fitted)
        if self.config.effective_condition is not None:
            pipelines.append(
                self.config.effective_condition.preprocessing_pipeline.fitted
            )

        return all(pipelines) if pipelines else False


def _build_chunks(config: DataConfigABC | None = None):
    """
    Document this function.

    Parameters
    ----------
    config : DataConfigABC | None
        Description not yet provided.

    Returns
    -------
    Any
        Description not yet provided.
    """
    if config is None:
        return
    required_sample_dimensions = (config.init_time_dim, config.lead_time_dim)
    sample_dims = (*required_sample_dimensions, config.realization_dim)

    chunks = {dim: 1 for dim in sample_dims if dim in config.coords}

    return chunks
