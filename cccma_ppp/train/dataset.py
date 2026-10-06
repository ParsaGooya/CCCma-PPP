import numpy as np
import xarray as xr
import torch
import dataclasses
import warnings
import cftime
import datetime
from pathlib import Path
from collections.abc import Sequence

from cccma_ppp.data_modules.dataset.dataset_abc import (
    DatasetConfigABC,
    lead_time_config,
)
from cccma_ppp.data_modules.dataset.dataset_abc import (
    DatasetABC,
    AddedTimeFeatures,
)
from cccma_ppp.data_modules.dataset.operator import DatasetOperator
from cccma_ppp.data_modules.data.data_configs import (
    ModelDataConfig,
    ObsDataConfig,
    ConditionDataConfig,
)

from cccma_ppp.data_modules.utils import add_lead_times, _unwrap_data_variables


from cccma_ppp.configs import lead_time_resolution


@dataclasses.dataclass
class TrainDatasetConfig(DatasetConfigABC):
    """
    Configuration for a training dataset.

    Defines the model input data, optional observation targets, optional
    conditioning data, conditioning strategy, and lead times used to construct
    training and validation samples.

    Parameters
    ----------
    model : ModelDataConfig
        Model data configuration used to construct model inputs.
    observation : ObsDataConfig | None
        Observation data configuration used as the prediction target. If ``None``,
        model data are used as the target.
    condition : ConditionDataConfig | None
        Optional conditioning data configuration.
    condition_method : str | None
        Method used to construct conditioning samples.
    lead_times : lead_time_config | None
        Lead times included in dataset samples. If ``None``, all available input
        lead times are used.
    """

    model: ModelDataConfig
    observation: ObsDataConfig | None = None
    condition: ConditionDataConfig | None = None
    condition_method: str = None
    lead_times: lead_time_config | None = None

    def __post_init__(self):
        """
        Initialize and validate the training dataset configuration.

        Resolves the common dataset configuration and then validates compatibility
        between model and observation data.
        """

        super().__init__()

        self._check_observation()

    def _check_observation(self):
        """
        Validate observation data against the model configuration.

        Checks temporal representation compatibility and reports differences between
        model and observation neural-network coordinates. When no observation target
        is configured, a conditioning method is required.

        Returns
        -------
        TrainDatasetConfig
            This configuration instance after validation.

        Raises
        ------
        ValueError
            If model and observation data use different datetime representations, or
            if no observation target and no conditioning method are configured.

        Warns
        -----
        UserWarning
            If model and observation data have inconsistent neural-network dimensions
            or coordinates.
        """
        if self.observation is not None:
            for dim in [
                dim
                for dim in self.supported_NN_dimensions
                if dim in self.observation.coords
            ]:
                if dim in self.model.coords:
                    if not self.observation.coords[dim].equals(self.model.coords[dim]):
                        warnings.warn(
                            "\n=====================================================================\n"
                            f"model and observation data do not have the same {dim} cooridnates.\n"
                            "=====================================================================\n"
                        )

                else:
                    warnings.warn(
                        "\n======================================================================\n"
                        f"observation data has NN dim {dim} which is not present in model data.\n"
                        "======================================================================\n"
                    )

                if (
                    self.model.info.time_coords_type
                    != self.observation.info.time_coords_type
                ):
                    raise ValueError(
                        "Observation data and model data must have the same"
                        " cftime/datetime type time coordinates."
                    )

        else:
            if self.condition_method is None:
                raise ValueError(
                    "No target observation is specified. Specify condition_method!"
                )

        return self

    @property
    def effective_input(self):
        """
        Return the data configuration used as the primary training input.

        Returns
        -------
        ModelDataConfig
            Model data configuration.
        """
        return self.model

    @property
    def ds_operator(self):
        """
        Return an operator for this dataset configuration.

        Returns
        -------
        DatasetOperator
            Dataset operator providing preprocessing, weighting, and metadata operations
        """
        return DatasetOperator(self)

    @property
    def get_common_time(self):
        """
        Return the temporal range common to model and target data.

        When observations are unavailable, the complete model temporal range is
        returned. Otherwise, the intersection of model and observation temporal
        ranges is used.

        Returns
        -------
        Any
            Temporal range available to both model inputs and training targets.
        """
        if self.observation is None:
            return self.model.time_range

        else:
            return self.model.time_range.intersection(self.observation.time_range)

    @property
    def available_times(self):
        """
        Return initialization times available for training samples.

        Available times are restricted to the temporal coverage shared by model and
        target data. The comparison accounts for the initialization-time frequency
        and lead-time resolution.

        Returns
        -------
        Any
            Initialization times that can be used to construct valid training
            samples.
        """

        model_times = self.model.coords[self.init_time_dim].to_index()
        common_times = self.get_common_time
        time_freq = self.model.init_time_frequency

        if time_freq == "year":
            min_year = model_times.year.min()
            max_year = model_times.year.max()

            return common_times[
                (common_times.year >= min_year) & (common_times.year <= max_year)
            ]

        elif time_freq == "month" and self.lead_time_resolution == "day":
            min_time = model_times.min()
            max_time = model_times.max()

            min_month = min_time.year * 12 + min_time.month
            max_month = max_time.year * 12 + max_time.month

            common_months = common_times.year * 12 + common_times.month

            return common_times[
                (common_months >= min_month) & (common_months <= max_month)
            ]

        return common_times[
            (common_times >= model_times.min()) & (common_times <= model_times.max())
        ]

    def fit_preprocessors(
        self,
        train_times: (
            Sequence[np.datetime64 | datetime.datetime | cftime.datetime]
            | np.ndarray
            | xr.DataArray
            | slice
        ),
        save: bool = False,
        save_path: str | Path | None = None,
        save_name: str | None = None,
    ):
        """
        Fit preprocessing pipelines using the selected training times.

        Parameters
        ----------
        train_times : Sequence[np.datetime64 | datetime.datetime | cftime.datetime] | np.ndarray | xr.DataArray | slice
            Initialization times used to fit the preprocessing pipelines.
        save : bool
            Whether to save the fitted preprocessing pipelines.
        save_path : str | Path | None
            Directory in which fitted preprocessing pipelines are saved.
        save_name : str | None
            Name used when saving fitted preprocessing pipelines.
        """
        self.ds_operator.fit_preprocessors(
            train_times=train_times,
            save=save,
            save_path=save_path,
            save_name=save_name,
        )

    def load_fitted_preprocessors(self, load_dir: Path | str | None = None):
        """
        Load fitted preprocessing pipelines for the training data sources.

        Parameters
        ----------
        load_dir : Path | str | None
            Directory containing the saved preprocessing pipelines. If ``None``, the
            default preprocessing directory is used.
        """
        self.ds_operator.load_fitted_preprocessors(load_dir)

    def add_fitted_preprocessor(self, preprocessor, index=0):
        """
        Add a fitted preprocessing module to the active data pipelines.

        Parameters
        ----------
        preprocessor : Any
            Fitted preprocessing module to add.
        index : int
            Position at which the preprocessing module is inserted.
        """
        self.ds_operator.add_fitted_preprocessor(preprocessor, index)

    def build_dataset(
        self,
        times: (
            Sequence[np.datetime64 | datetime.datetime | cftime.datetime]
            | np.ndarray
            | xr.DataArray
        ),
        time_features: AddedTimeFeatures,
        mask: xr.DataArray | None = None,
        return_metadata: bool = False,
        load: bool = False,
    ):
        """
        Build a training dataset from this configuration.

        Parameters
        ----------
        times : Sequence[np.datetime64 | datetime.datetime | cftime.datetime] | np.ndarray | xr.DataArray
            Initialization times included in the dataset.
        time_features : AddedTimeFeatures
            Temporal feature generator applied to dataset samples.
        mask : xr.DataArray | None
            Optional mask identifying unavailable or excluded sample combinations.
        return_metadata : bool
            Whether each sample is returned with its sampling-coordinate metadata.
        load : bool
            Whether underlying xarray data are loaded into memory when opened.

        Returns
        -------
        TrainDataset
            Constructed training dataset.
        """
        return TrainDataset(
            config=self,
            requested_times=times,
            time_features=time_features,
            mask=mask,
            return_metadata=return_metadata,
            load=load,
        )


@dataclasses.dataclass
class TrainDataset(DatasetABC):
    """
    Dataset for constructing training samples and targets.

    Samples are assembled from model, observation, and conditioning data according
    to the configured conditioning strategy. Temporal features and optional
    sampling metadata can also be returned.

    Parameters
    ----------
    config : TrainDatasetConfig
        Configuration defining training data sources and sampling behavior.
    requested_times : Sequence[np.datetime64 | datetime.datetime | cftime.datetime] | np.ndarray | xr.DataArray
        Initialization times requested for dataset sampling.
    time_features : AddedTimeFeatures
        Temporal feature generator applied to samples.
    mask : xr.DataArray | None
        Optional mask identifying unavailable or excluded sample combinations.
    return_metadata : bool
        Whether each sample is returned with its sampling-coordinate metadata.
    load : bool
        Whether underlying xarray data are loaded into memory when opened.
    """

    config: TrainDatasetConfig
    requested_times: (
        Sequence[np.datetime64 | datetime.datetime | cftime.datetime]
        | np.ndarray
        | xr.DataArray
    )
    time_features: AddedTimeFeatures
    mask: xr.DataArray | None = None
    return_metadata: bool = False
    load: bool = False

    def __post_init__(self):
        """
        Initialize the training dataset.

        Initializes the common sampling state, opens observation data when configured,
        and maps sample coordinates to positional indexes in the observation dataset.
        """
        super().__init__()

        if self.config.observation is not None:
            self.config.observation.open_xarray_data(
                load=self.load, add_time_auxiliary_coords=True
            )

        self.obs_indexes = self.get_obs_indexes(self.sample_coords)

    @property
    def _autoencoding_model_data(self):
        """
        Indicate whether model data are used as the training target.

        Returns
        -------
        bool
            ``True`` when no observation dataset is configured, otherwise ``False``.
        """
        return self.config.observation is None

    @property
    def _load_model(self):
        """
        Indicate whether model data must be loaded separately.

        Returns
        -------
        bool
            ``True`` when model data are needed as an autoencoding target or are not
            being reused as the conditioning source.
        """
        return any(
            [
                self._autoencoding_model_data,
                not self.config._using_model_data_as_condition,
            ]
        )

    @property
    def _write_condition_to_input(self):
        """
        Indicate whether conditioning data are used directly as the model input.

        Returns
        -------
        bool
            ``True`` when model data are used as conditioning data or when the
            dataset is training on model data without observation targets.
        """
        if self.config._using_model_data_as_condition:
            return True
        else:
            if self._autoencoding_model_data:
                return True

        return False

    @property
    def _concat_condition_to_input(self):
        """
        Indicate whether conditioning data are concatenated with model inputs.

        Returns
        -------
        bool
            ``True`` when model data remain the primary input and a separate effective
            conditioning dataset is configured.
        """
        return (
            self._write_condition_to_input is False
            and self.config.effective_condition is not None
        )

    def get_obs_indexes(
        self,
        sample_coords: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray] | None:
        """
        Map sample coordinates to positional indexes in the observation dataset.

        Observation times are calculated by applying each sample's lead time to its
        model initialization time.

        Parameters
        ----------
        sample_coords : dict[str, np.ndarray]
            Sampling coordinates containing initialization and lead times.

        Returns
        -------
        dict[str, np.ndarray] | None
            Positional indexes into the observation dataset, or ``None`` when no
            observation data are configured.

        Raises
        ------
        ValueError
            If one or more calculated observation times cannot be found in the
            observation dataset.
        """
        if self.config.observation is None:
            return None

        model_times = np.asarray(sample_coords[self.config.init_time_dim])
        lead_times = np.asarray(sample_coords[self.config.lead_time_dim])

        observation_times = add_lead_times(
            init_times=model_times,
            lead_times=lead_times,
            lead_time_resolution=lead_time_resolution,
        )

        observation_coords = {
            self.config.init_time_dim: observation_times,
        }

        indexes = {
            dim: self.config.observation.indexes[dim].get_indexer(values)
            for dim, values in observation_coords.items()
        }

        missing_values = {
            dim: observation_coords[dim][positions == -1]
            for dim, positions in indexes.items()
            if np.any(positions == -1)
        }

        if missing_values:
            raise ValueError(
                "Some observation coordinates were not found in the observation "
                f"dataset: {missing_values}"
            )

        return indexes

    def get_target_shape(self):
        """
        Determine the shape of a training target sample.

        Observation dimensions are used when observation targets are configured. If
        the observation preprocessing pipeline contains a flattener, its retained
        locations define the non-channel target size. Otherwise, supported
        neural-network dimensions determine the target shape. For autoencoding,
        the input shape is returned.

        Returns
        -------
        tuple
            Target shape excluding the batch dimension.
        """
        if self.config.observation is not None:
            len_names = len(self.config.observation.names)

            if self.config.observation.preprocessing_pipeline.has_flattener:
                flattener = (
                    self.config.observation.preprocessing_pipeline.get_flattener()
                )
                out_shape = flattener.final_locations.shape

            else:
                out_shape = tuple(
                    self.config.observation.coords[dim].size
                    for dim in self.config.supported_NN_dimensions
                    if dim in self.config.observation.coords
                )

            return tuple([len_names, *out_shape])

        else:
            return self.get_input_shape()

    def _index_observation_dataset(self, ind: int) -> xr.DataArray | None:
        """
        Retrieve observation data for a sample.

        When observations retain individual ensemble realizations, one realization is
        selected randomly for the sample.

        Parameters
        ----------
        ind : int
            Sample index to retrieve.

        Returns
        -------
        xr.DataArray | None
            Selected and preprocessed observation variables, or ``None`` when no
            observation data are configured.
        """
        if self.config.observation is None:
            return None

        selection = {dim: [indexes[ind]] for dim, indexes in self.obs_indexes.items()}

        if (
            not self.config.observation.ensemble_mean
            and self.config.realization_dim in self.config.observation.dims
        ):
            selection[self.config.realization_dim] = [
                np.random.randint(
                    self.config.observation.data.sizes[self.config.realization_dim]
                )
            ]

        obs = self.config.observation.isel(**selection)

        return _unwrap_data_variables(obs)

    def __getitem__(self, ind):
        """
        Return a training sample.

        Model, observation, and conditioning data are indexed for the requested
        sample. The target and input are then resolved according to autoencoding and
        conditioning behavior, and configured temporal features are added.

        Parameters
        ----------
        ind : int
            Index of the training sample to retrieve.

        Returns
        -------
        dict | tuple[dict, dict]
            Dictionary containing input, target, and optional additional feature
            tensors. When metadata are enabled, the sample dictionary is returned
            together with its sampling coordinates.
        """
        selection = {dim: value[ind] for dim, value in self.sample_coords.items()}

        condition = self._index_condition_dataset(ind)
        target = self._index_observation_dataset(ind)
        input = self._index_model_dataset(ind)

        if self._autoencoding_model_data:
            target = input

        if self._write_condition_to_input:
            input = condition

        elif self._concat_condition_to_input:
            input = xr.concat([input, condition], dim="channels")

        added_features_array = self.time_features(ind, input)

        input_array, target_array = self._compute(
            input.data,
            target.data,
        )

        datadict = dict(
            input=torch.as_tensor(input_array, dtype=torch.float32),
            target=torch.as_tensor(target_array, dtype=torch.float32),
            added_features=torch.tensor(added_features_array, dtype=torch.float32)
            if added_features_array is not None
            else None,
        )

        if self.return_metadata:
            return datadict, selection
        else:
            return datadict
