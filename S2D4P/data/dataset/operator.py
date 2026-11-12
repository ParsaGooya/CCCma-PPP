import numpy as np
import datetime
import cftime
from pathlib import Path
import xarray as xr
from collections.abc import Sequence

from S2D4P.data.data_configs.data_configs_abc import DataConfigABC
from S2D4P.data.dataset.dataset_abc import DatasetConfigABC
from S2D4P.data.utils import _validate_time_sequence
from S2D4P.data.weights import WeightsConfig
from S2D4P.preprocessing.preprocessing_ABC import PreprocessModuleABC


class DatasetOperator:
    """
    Coordinate dataset-level preprocessing, weighting, and metadata operations.

    Provides shared operations across forecast, reference, and conditioning data
    configured by a dataset configuration.

    Parameters
    ----------
    config : DatasetConfigABC
        Dataset configuration defining the active data sources and preprocessing
        pipelines.
    """

    def __init__(self, config: DatasetConfigABC):
        """
        Initialize the dataset operator.

        Parameters
        ----------
        config : DatasetConfigABC
            Dataset configuration on which operations are performed.
        """
        self.config = config

    @property
    def config_reference(self):
        """
        Return the configured reference dataset when available.

        Returns
        -------
        DataConfigABC | None
            Reference data configuration, or ``None`` when the dataset configuration
            does not define reference data.
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
        Fit active preprocessing pipelines using training data.

        Training times are validated against the available dataset times before
        forecast, reference, and conditioning preprocessing pipelines are fitted.
        Temporal training masks are applied to forecast and non-static conditioning
        data.

        Parameters
        ----------
        train_times : Sequence[np.datetime64 | datetime.datetime | cftime.datetime] | np.ndarray | xr.DataArray
            Initialization times used to fit preprocessing statistics.
        save : bool
            Whether fitted preprocessing pipelines are saved.
        save_path : Path | str | None
            Directory in which fitted preprocessing pipelines are saved.
        save_name : str | None
            Optional filename used when saving preprocessing pipelines.

        Raises
        ------
        ValueError
            If one or more requested training times are unavailable.
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
        Load fitted preprocessing pipelines for all active data sources.

        Parameters
        ----------
        load_dir : Path | str | None
            Directory containing saved preprocessing pipelines. If ``None``, each
            data configuration uses its default preprocessing location.
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
        Add a fitted preprocessing module to all active pipelines.

        Parameters
        ----------
        preprocessor : PreprocessModuleABC
            Fitted preprocessing module to add.
        index : int
            Position at which the preprocessing module is inserted.

        Raises
        ------
        TypeError
            If ``preprocessor`` is not an instance of ``PreprocessModuleABC``.
        ValueError
            If ``preprocessor`` has not been fitted.
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
        Build loss weights from the configured target data.

        Reference data are used when available; otherwise, forecast data define the
        target coordinates and variables. If the target preprocessing pipeline
        contains a flattener, it is supplied when constructing the weights.

        Parameters
        ----------
        config : WeightsConfig | None
            Configuration used to construct loss weights. If ``None``, default weight
            settings are used.
        save : bool
            Whether generated weights are saved.
        save_path : Path | str | None
            Directory in which generated weights are saved.
        save_name : str | None
            Optional filename used when saving generated weights.

        Returns
        -------
        xr.DataArray
            Generated loss weights aligned with target dimensions and variables.

        Raises
        ------
        RuntimeError
            If channel-specific weights do not match the configured output variables.
        ValueError
            If neither reference nor forecast data are available to define the target
            coordinates.
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
        Return metadata describing model input variables.

        Metadata include input variable names, preprocessing steps applied to each
        variable, and neural-network dimensions. Conditioning metadata are included
        according to the configured conditioning behavior.

        Returns
        -------
        dict
            Input metadata containing ``variables``, ``preprocessors``, and
            ``NN_dims``.
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
        Return metadata describing model target variables.

        Reference data define the target when available. Otherwise, forecast data are
        used.

        Returns
        -------
        dict
            Target metadata containing ``variables``, ``preprocessors``, and
            ``NN_dims``.

        Raises
        ------
        ValueError
            If neither reference nor forecast data are available to define target
            metadata.
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
        Append data-source metadata to a metadata dictionary.

        Each configured variable is added together with the ordered preprocessing
        steps associated with its data source.

        Parameters
        ----------
        metadata : dict
            Metadata dictionary containing variable and preprocessor lists.
        dataconfig : DataConfigABC
            Data configuration whose variable and preprocessing information is added.

        Returns
        -------
        dict
            Updated metadata dictionary.
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
        """
        Indicate whether all active preprocessing pipelines are fitted.

        Returns
        -------
        bool
            ``True`` when every active forecast, reference, and conditioning
            preprocessing pipeline is fitted. Returns ``False`` when no active
            pipelines are available.
        """
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
    Build sample-dimension chunk settings for a data configuration.

    Initialization time, lead time, and realization dimensions that are present
    in the configured data are assigned a chunk size of one.

    Parameters
    ----------
    config : DataConfigABC | None
        Data configuration whose available sample dimensions determine the chunk
        settings.

    Returns
    -------
    dict[str, int] | None
        Mapping of available sample dimensions to a chunk size of one, or
        ``None`` when no configuration is provided.
    """
    if config is None:
        return
    required_sample_dimensions = (config.init_time_dim, config.lead_time_dim)
    sample_dims = (*required_sample_dimensions, config.realization_dim)

    chunks = {dim: 1 for dim in sample_dims if dim in config.coords}

    return chunks
