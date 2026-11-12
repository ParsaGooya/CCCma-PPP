import numpy as np
import xarray as xr
import torch
import dataclasses
import warnings
import cftime
import datetime
from pathlib import Path
from collections.abc import Sequence

from S2D4P.data.dataset.dataset_abc import (
    DatasetConfigABC,
    lead_time_config,
)
from S2D4P.data.dataset.dataset_abc import (
    DatasetABC,
    AddedTimeFeatures,
)
from S2D4P.data.dataset.operator import DatasetOperator
from S2D4P.data.data_configs.data_configs import (
    ForecastDataConfig,
    RefDataConfig,
    ConditionDataConfig,
)

from S2D4P.data.utils import add_lead_times, _unwrap_data_variables


from S2D4P.configs import lead_time_resolution


@dataclasses.dataclass
class TrainDatasetConfig(DatasetConfigABC):
    """
    Configuration for a training dataset.

    Defines forecast inputs, optional reference targets, optional conditioning
    data, the conditioning strategy, and forecast lead times used to construct
    training samples.

    Parameters
    ----------
    forecast : ForecastDataConfig
        Forecast data configuration used to construct model inputs and, when no
        reference data are provided, training targets.
    reference : RefDataConfig | None
        Optional reference data configuration used to construct training targets.
    condition : ConditionDataConfig | None
        Optional auxiliary conditioning data configuration.
    condition_method : str | None
        Method used to associate conditioning data with forecast inputs.
    lead_times : lead_time_config | None
        Forecast lead times included in training samples. If ``None``, all
        available input lead times are used.
    """

    forecast: ForecastDataConfig
    reference: RefDataConfig | None = None
    condition: ConditionDataConfig | None = None
    condition_method: str = None
    lead_times: lead_time_config | None = None

    def __post_init__(self):
        """
        Initialize and validate the training dataset configuration.

        Performs common dataset configuration validation and then checks compatibility
        between forecast and optional reference data.
        """
        super().__init__()

        self._check_reference()

    def _check_reference(self):
        """
        Validate compatibility between forecast and reference data.

        Spatial neural-network coordinates are compared when reference data are
        provided, and both sources must use the same datetime representation. When no
        reference data are configured, a conditioning method is required.

        Returns
        -------
        TrainDatasetConfig
            This validated training dataset configuration.

        Raises
        ------
        ValueError
            If forecast and reference data use different datetime representations, or
            no reference data and no conditioning method are configured.

        Warns
        -----
        UserWarning
            If forecast and reference data have different neural-network coordinates
            or a reference neural-network dimension is absent from the forecast data.
        """
        if self.reference is not None:
            for dim in [
                dim
                for dim in self.supported_NN_dimensions
                if dim in self.reference.coords
            ]:
                if dim in self.forecast.coords:
                    if not self.reference.coords[dim].equals(self.forecast.coords[dim]):
                        warnings.warn(
                            "\n=====================================================================\n"
                            f"Forecast and reference data do not have the same {dim} cooridnates.\n"
                            "=====================================================================\n"
                        )

                else:
                    warnings.warn(
                        "\n======================================================================\n"
                        f"reference data has NN dim {dim} which is not present in forecast data.\n"
                        "======================================================================\n"
                    )

                if (
                    self.forecast.info.time_coords_type
                    != self.reference.info.time_coords_type
                ):
                    raise ValueError(
                        "reference data and forecast data must have the same"
                        " cftime/datetime type time coordinates."
                    )

        else:
            if self.condition_method is None:
                raise ValueError(
                    "No target reference is specified. Specify condition_method!"
                )

        return self

    @property
    def effective_input(self):
        """
        Return the primary data source used for model input.

        Returns
        -------
        ForecastDataConfig
            Forecast data configuration.
        """
        return self.forecast

    @property
    def ds_operator(self):
        """
        Return an operator for this training dataset configuration.

        Returns
        -------
        DatasetOperator
            Dataset operator providing preprocessing, weighting, and metadata
            operations.
        """
        return DatasetOperator(self)

    @property
    def get_common_time(self):
        """
        Return the temporal coverage shared by forecast and target data.

        When no reference data are configured, the forecast temporal range is used
        directly.

        Returns
        -------
        Any
            Intersection of forecast and reference temporal coverage, or forecast
            temporal coverage when no reference data are configured.
        """
        if self.reference is None:
            return self.forecast.time_range

        else:
            return self.forecast.time_range.intersection(self.reference.time_range)

    @property
    def available_times(self):
        """
        Return initialization times available for training.

        The common forecast and reference temporal coverage is restricted to the
        initialization period represented by the forecast data. Yearly and
        monthly-initialized forecasts are handled using their corresponding calendar
        resolution.

        Returns
        -------
        Any
            Times within the common temporal coverage that are valid forecast
            initialization times.
        """
        forecast_times = self.forecast.coords[self.init_time_dim].to_index()
        common_times = self.get_common_time
        time_freq = self.forecast.init_time_frequency

        if time_freq == "year":
            min_year = forecast_times.year.min()
            max_year = forecast_times.year.max()

            return common_times[
                (common_times.year >= min_year) & (common_times.year <= max_year)
            ]

        elif time_freq == "month" and self.lead_time_resolution == "day":
            min_time = forecast_times.min()
            max_time = forecast_times.max()

            min_month = min_time.year * 12 + min_time.month
            max_month = max_time.year * 12 + max_time.month

            common_months = common_times.year * 12 + common_times.month

            return common_times[
                (common_months >= min_month) & (common_months <= max_month)
            ]

        return common_times[
            (common_times >= forecast_times.min())
            & (common_times <= forecast_times.max())
        ]

    def fit_preprocessors(
        self,
        train_times: (
            Sequence[np.datetime64 | datetime.datetime | cftime.datetime]
            | np.ndarray
            | xr.DataArray
        ),
        save: bool = False,
        save_path: str | Path | None = None,
        save_name: str | None = None,
    ):
        """
        Fit preprocessing pipelines using selected training times.

        Parameters
        ----------
        train_times : Sequence[np.datetime64 | datetime.datetime | cftime.datetime] | np.ndarray | xr.DataArray
            Initialization times used to estimate preprocessing statistics.
        save : bool
            Whether fitted preprocessing pipelines are saved.
        save_path : str | Path | None
            Directory in which fitted preprocessing pipelines are saved.
        save_name : str | None
            Optional filename used when saving fitted preprocessing pipelines.
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
            Directory containing saved preprocessing pipelines. If ``None``, the
            default preprocessing location is used.
        """
        self.ds_operator.load_fitted_preprocessors(load_dir)

    def add_fitted_preprocessor(self, preprocessor, index=0):
        """
        Add a fitted preprocessing module to the active data pipelines.

        Parameters
        ----------
        preprocessor : PreprocessModuleABC
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
            Initialization times included in the training dataset.
        time_features : AddedTimeFeatures
            Temporal feature generator applied to training samples.
        mask : xr.DataArray | None
            Optional mask identifying initialization-time and lead-time combinations
            excluded from sampling.
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
    Dataset for constructing model inputs and targets during training.

    Training samples are assembled from forecast, reference, and conditioning
    data according to the configured conditioning strategy. When reference data
    are absent, forecast data are used as reconstruction targets.

    Parameters
    ----------
    config : TrainDatasetConfig
        Configuration defining training data sources and sampling behavior.
    requested_times : Sequence[np.datetime64 | datetime.datetime | cftime.datetime] | np.ndarray | xr.DataArray
        Initialization times requested for training.
    time_features : AddedTimeFeatures
        Temporal feature generator applied to samples.
    mask : xr.DataArray | None
        Optional mask identifying sample combinations excluded from training.
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

        Initializes common sampling state, opens reference data when configured, and
        maps forecast initialization and lead times to corresponding reference
        indexes.
        """
        super().__init__()

        if self.config.reference is not None:
            self.config.reference.open_xarray_data(
                load=self.load, add_time_auxiliary_coords=True
            )

        self.obs_indexes = self.get_obs_indexes(self.sample_coords)

    @property
    def _autoencoding_forecast_data(self):
        """
        Indicate whether forecast data are used as reconstruction targets.

        Returns
        -------
        bool
            ``True`` when no separate reference dataset is configured.
        """
        return self.config.reference is None

    @property
    def _load_forecast(self):
        """
        Indicate whether forecast data must be loaded separately.

        Returns
        -------
        bool
            ``True`` when forecast data are required as reconstruction targets or are
            not being reused through the conditioning configuration.
        """
        return any(
            [
                self._autoencoding_forecast_data,
                not self.config._using_forecast_data_as_condition,
            ]
        )

    @property
    def _write_condition_to_input(self):
        """
        Indicate whether conditioning data are used directly as model input.

        Returns
        -------
        bool
            ``True`` when forecast data are reused as conditioning data or when
            forecast autoencoding requires the resolved conditioning input to replace
            the separately indexed forecast input.
        """
        if self.config._using_forecast_data_as_condition:
            return True
        else:
            if self._autoencoding_forecast_data:
                return True

        return False

    @property
    def _concat_condition_to_input(self):
        """
        Indicate whether conditioning data are concatenated with forecast input.

        Returns
        -------
        bool
            ``True`` when forecast data remain the primary input and separate
            conditioning data are configured.
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
        Map forecast samples to positional reference-data indexes.

        Reference valid times are calculated from each forecast initialization time
        and lead time before being mapped to the reference dataset's time index.

        Parameters
        ----------
        sample_coords : dict[str, np.ndarray]
            Sampling coordinates for all valid training samples.

        Returns
        -------
        dict[str, np.ndarray] | None
            Positional reference-data indexes for each sample, or ``None`` when no
            reference dataset is configured.

        Raises
        ------
        ValueError
            If one or more calculated reference times cannot be found in the reference
            dataset.
        """
        if self.config.reference is None:
            return None

        forecast_times = np.asarray(sample_coords[self.config.init_time_dim])
        lead_times = np.asarray(sample_coords[self.config.lead_time_dim])

        reference_times = add_lead_times(
            init_times=forecast_times,
            lead_times=lead_times,
            lead_time_resolution=lead_time_resolution,
        )

        reference_coords = {
            self.config.init_time_dim: reference_times,
        }

        indexes = {
            dim: self.config.reference.indexes[dim].get_indexer(values)
            for dim, values in reference_coords.items()
        }

        missing_values = {
            dim: reference_coords[dim][positions == -1]
            for dim, positions in indexes.items()
            if np.any(positions == -1)
        }

        if missing_values:
            raise ValueError(
                "Some reference coordinates were not found in the reference "
                f"dataset: {missing_values}"
            )

        return indexes

    def get_target_shape(self):
        """
        Determine the shape of a target sample.

        Reference variables and spatial dimensions define the target shape when
        reference data are configured. A fitted flattener determines the flattened
        spatial size when applicable. Without reference data, the input shape is used.

        Returns
        -------
        tuple
            Target shape excluding the batch dimension.
        """
        if self.config.reference is not None:
            len_names = len(self.config.reference.names)

            if self.config.reference.preprocessing_pipeline.has_flattener:
                flattener = self.config.reference.preprocessing_pipeline.get_flattener()
                out_shape = flattener.final_locations.shape

            else:
                out_shape = tuple(
                    self.config.reference.coords[dim].size
                    for dim in self.config.supported_NN_dimensions
                    if dim in self.config.reference.coords
                )

            return tuple([len_names, *out_shape])

        else:
            return self.get_input_shape()

    def _index_reference_dataset(self, ind: int) -> xr.DataArray | None:
        """
        Retrieve reference target data for a sample.

        The precomputed reference time index is used to select the target. When the
        reference retains individual ensemble members, one realization is selected
        randomly.

        Parameters
        ----------
        ind : int
            Sample index to retrieve.

        Returns
        -------
        xr.DataArray | None
            Selected and preprocessed reference variables, or ``None`` when no
            reference dataset is configured.
        """
        if self.config.reference is None:
            return None

        selection = {dim: [indexes[ind]] for dim, indexes in self.obs_indexes.items()}

        if (
            not self.config.reference.ensemble_mean
            and self.config.realization_dim in self.config.reference.dims
        ):
            selection[self.config.realization_dim] = [
                np.random.randint(
                    self.config.reference.data.sizes[self.config.realization_dim]
                )
            ]

        obs = self.config.reference.isel(**selection)

        return _unwrap_data_variables(obs)

    def __getitem__(self, ind):
        """
        Return a training sample.

        Forecast, conditioning, and reference data are indexed and combined according
        to the configured training and conditioning strategy. Temporal features are
        generated and the resulting input and target data are converted to tensors.

        Parameters
        ----------
        ind : int
            Index of the training sample to retrieve.

        Returns
        -------
        dict | tuple[dict, dict]
            Dictionary containing input, target, and optional additional-feature
            tensors. When metadata are enabled, the sample dictionary is returned
            together with its sampling coordinates.
        """
        selection = {dim: value[ind] for dim, value in self.sample_coords.items()}

        condition = self._index_condition_dataset(ind)
        target = self._index_reference_dataset(ind)
        input = self._index_forecast_dataset(ind)

        if self._autoencoding_forecast_data:
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
