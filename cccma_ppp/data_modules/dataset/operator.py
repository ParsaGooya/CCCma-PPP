import numpy as np
import datetime
import cftime
from pathlib import Path
import xarray as xr
from collections.abc import Sequence

from cccma_ppp.data_modules.data.data_abc import DataConfigABC
from cccma_ppp.data_modules.dataset.dataset_abc import DatasetConfigABC
from cccma_ppp.data_modules.utils import _validate_time_sequence
from cccma_ppp.data_modules.weights import WeightsConfig
from cccma_ppp.preprocessing.preprocessing_ABC import PreprocessModuleABC


class DatasetOperator:
    """
    Provide shared operations for a configured dataset.

    The operator manages preprocessing pipelines, variable metadata, and spatial
    weights for the data sources associated with a dataset configuration.

    Parameters
    ----------
    config : DatasetConfigABC
        Dataset configuration containing the model, conditioning, and optional
        observation data sources.
    """

    def __init__(self, config: DatasetConfigABC):
        """
        Initialize a dataset operator.

        Parameters
        ----------
        config : DatasetConfigABC
            Dataset configuration on which operations are performed.
        """
        self.config = config

    @property
    def config_observation(self):
        """
        Return the configured observation data when available.

        Returns
        -------
        DataConfigABC | None
            Observation data configuration, or ``None`` when the dataset
            configuration does not define observation data.
        """
        if hasattr(self.config, "observation"):
            return self.config.observation

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
        Fit preprocessing pipelines using the requested training times.

        Model, observation, and effective conditioning preprocessors are fitted when
        their corresponding data sources are available. Temporal, lead-time, and
        realization selections are constructed as required by each data source.

        Parameters
        ----------
        train_times : Sequence[np.datetime64 | datetime.datetime | cftime.datetime] | np.ndarray | xr.DataArray
            Initialization times used to fit the preprocessing pipelines.
        save : bool
            Whether to save fitted preprocessing pipelines.
        save_path : Path | str | None
            Directory in which fitted preprocessing pipelines are saved.
        save_name : str | None
            Name used when saving fitted preprocessing pipelines.

        Raises
        ------
        ValueError
            If the training-time sequence is invalid or contains times unavailable
            from the dataset configuration.
        """

        train_times_array = np.asarray(train_times)

        _validate_time_sequence(train_times)

        missing = [t for t in train_times_array if t not in self.config.available_times]

        if missing:
            raise ValueError(f"The following train_times are unavailable: {missing}")

        if self.config.model is not None:
            selection = {
                self.config.init_time_dim: self.config.get_input_times(train_times),
                self.config.lead_time_dim: self.config.model.coords[
                    self.config.lead_time_dim
                ],
            }
            if self.config.model.coords.get(self.config.realization_dim) is not None:
                selection[self.config.realization_dim] = self.config.model.coords[
                    self.config.realization_dim
                ]

            self.config.model.fit_preprocessor_pipeline(
                selection=selection,
                mask=True,
                save=save,
                save_path=save_path,
                save_name=save_name,
            )

        if self.config_observation is not None:
            selection = {self.config.init_time_dim: train_times}
            if (
                self.config_observation.coords.get(self.config.realization_dim)
                is not None
            ):
                selection[self.config.realization_dim] = self.config_observation.coords[
                    self.config.realization_dim
                ]

            self.config_observation.fit_preprocessor_pipeline(
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
            data configuration uses its default preprocessing-pipeline location.
        """
        if self.config.model is not None:
            self.config.model.load_preprocessor_pipeline(load_dir)

        if self.config_observation is not None:
            self.config_observation.load_preprocessor_pipeline(load_dir)

        if self.config.effective_condition is not None:
            self.config.effective_condition.load_preprocessor_pipeline(load_dir)

        self.config._fitted_preprocessors = self._fitted_preprocessors

    def add_fitted_preprocessor(self, preprocessor: PreprocessModuleABC, index=0):
        """
        Add a fitted preprocessing module to all active preprocessing pipelines.

        Parameters
        ----------
        preprocessor : PreprocessModuleABC
            Fitted preprocessing module to add.
        index : int
            Position at which the preprocessing module is inserted into each
            pipeline.

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
        Build spatial or variable weights for the dataset target.

        Observation data are used as the reference when available; otherwise model
        data are used. If the reference preprocessing pipeline contains a flattener,
        it is supplied when constructing the weights.

        Parameters
        ----------
        config : WeightsConfig | None
            Configuration used to construct the weights. If ``None``, a default
            configuration is created.
        save : bool
            Whether to save the generated weights.
        save_path : Path | str | None
            Directory in which generated weights are saved.
        save_name : str | None
            Name used when saving generated weights.

        Returns
        -------
        xr.DataArray
            Generated weights aligned with the target dimensions and variables.

        Raises
        ------
        RuntimeError
            If channel-specific weights do not correspond to the target variable
            names.
        ValueError
            If neither model nor observation data are available as a weighting
            reference.
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
        Build metadata describing model input variables and preprocessing.

        Metadata are collected from the data sources contributing to the effective
        model input and include variable names, preprocessing operations, and
        neural-network dimensions.

        Returns
        -------
        dict
            Input metadata containing variable names, preprocessing operations, and
            neural-network dimension names.
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
                if dim in self.config.model.coords
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
                if dim in self.config.effective_condition.coords
            ]:
                NN_dims.append(dim)

        metadata["NN_dims"] = ["channels"] + NN_dims

        return metadata

    def get_target_var_metadata(self):
        """
        Build metadata describing target variables and preprocessing.

        Observation data are used when available; otherwise model data are used as
        the target reference.

        Returns
        -------
        dict
            Target metadata containing variable names, preprocessing operations, and
            neural-network dimension names.

        Raises
        ------
        ValueError
            If neither observation nor model data are available for constructing
            target metadata.
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
                if dim in self.config.model.coords
            ]:
                NN_dims.append(dim)

        else:
            metadata = self._update_metadata_with_dataconfig_metadata(
                metadata, self.config_observation
            )

            for dim in [
                dim
                for dim in self.config.supported_NN_dimensions
                if dim in self.config_observation.coords
            ]:
                NN_dims.append(dim)

        metadata["NN_dims"] = ["channels"] + NN_dims

        return metadata

    def _update_metadata_with_dataconfig_metadata(
        self, metadata: dict, dataconfig: DataConfigABC
    ):
        """
        Append metadata from a data configuration.

        Each configured variable is added together with the names of the preprocessing
        operations applied by its preprocessing pipeline.

        Parameters
        ----------
        metadata : dict
            Metadata dictionary to update.
        dataconfig : DataConfigABC
            Data configuration whose variable and preprocessing metadata are added.

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
        Check whether all active preprocessing pipelines are fitted.

        Returns
        -------
        bool
            ``True`` if every active model, observation, and conditioning
            preprocessing pipeline is fitted. Returns ``False`` when no pipelines are
            active or at least one active pipeline is not fitted.
        """
        pipelines = []
        if self.config.model is not None:
            pipelines.append(self.config.model.preprocessing_pipeline.fitted)
        if self.config_observation is not None:
            pipelines.append(self.config_observation.preprocessing_pipeline.fitted)
        if self.config.effective_condition is not None:
            pipelines.append(
                self.config.effective_condition.preprocessing_pipeline.fitted
            )

        return all(pipelines) if pipelines else False


def _build_chunks(config: DataConfigABC | None = None):
    """
    Build chunk sizes for dataset sampling dimensions.

    Initialization time, lead time, and realization dimensions present in the
    configured data are assigned a chunk size of one.

    Parameters
    ----------
    config : DataConfigABC | None
        Data configuration whose sampling dimensions are used to construct chunk
        sizes.

    Returns
    -------
    dict[str, int] | None
        Mapping from available sampling dimensions to a chunk size of one, or
        ``None`` if no data configuration is provided.
    """
    if config is None:
        return
    required_sample_dimensions = (config.init_time_dim, config.lead_time_dim)
    sample_dims = (*required_sample_dimensions, config.realization_dim)

    chunks = {dim: 1 for dim in sample_dims if dim in config.coords}

    return chunks
