import dataclasses
from pathlib import Path
from collections.abc import Sequence
import numpy as np
import xarray as xr
import torch
import cftime
import datetime

from S2D4P.data.dataset.dataset_abc import (
    DatasetConfigABC,
    DatasetABC,
    lead_time_config,
    AddedTimeFeatures,
)

from S2D4P.data.dataset.operator import DatasetOperator

from S2D4P.data.data_configs.data_configs import (
    ForecastDataConfig,
    ConditionDataConfig,
)

from S2D4P.train.dataset import TrainDatasetConfig
from S2D4P.preprocessing.preprocessing_ABC import PreprocessModuleABC


@dataclasses.dataclass
class InferenceDatasetConfig(DatasetConfigABC):
    """
    Configuration for an inference dataset.

    Defines the forecast and conditioning data used to construct model inputs,
    the conditioning strategy, and the forecast lead times included during
    inference.

    Parameters
    ----------
    forecast : ForecastDataConfig | None
        Forecast data configuration used as an inference input.
    condition : ConditionDataConfig | None
        Optional conditioning data configuration.
    condition_method : str | None
        Method used to associate conditioning data with forecast inputs.
    lead_times : lead_time_config | None
        Lead times included in inference samples. If ``None``, all available
        input lead times are used.
    """

    forecast: ForecastDataConfig | None = None
    condition: ConditionDataConfig | None = None
    condition_method: str = None
    lead_times: lead_time_config | None = None

    def __post_init__(self):
        """
        Initialize and validate the inference dataset configuration.

        Performs common dataset configuration setup, including input-source,
        conditioning, compatibility, and lead-time validation.
        """
        super().__init__()

    @property
    def effective_input(self):
        """
        Return the primary data source used for inference input.

        Forecast data are preferred when configured. Otherwise, conditioning data are
        used as the primary input.

        Returns
        -------
        ForecastDataConfig | ConditionDataConfig
            Effective inference input configuration.
        """

        if self.forecast is not None:
            return self.forecast
        else:
            return self.condition

    @property
    def ds_operator(self):
        """
        Return an operator for this dataset configuration.

        Returns
        -------
        DatasetOperator
            Dataset operator providing preprocessing and metadata operations.
        """
        return DatasetOperator(self)

    @property
    def available_times(self):
        """
        Return initialization times available for inference.

        When both forecast and conditioning data are configured, only initialization
        times present in both data sources are returned.

        Returns
        -------
        Any
            Initialization times shared by all configured time-dependent input
            sources.
        """
        time_coords = list()
        if self.condition is not None:
            time_coords.append(
                self.condition.coords[self.init_time_dim].to_index(),
            )
        if self.forecast is not None:
            time_coords.append(
                self.forecast.coords[self.init_time_dim].to_index(),
            )

        common = time_coords[0]
        for tr in time_coords[1:]:
            common = common.intersection(tr)

        return common

    def load_fitted_preprocessors(self, load_dir: Path | str | None = None):
        """
        Load fitted preprocessing pipelines for inference data sources.

        Parameters
        ----------
        load_dir : Path | str | None
            Directory containing saved preprocessing pipelines. If ``None``, the
            default preprocessing directory is used.
        """
        self.ds_operator.load_fitted_preprocessors(load_dir)

    def add_fitted_preprocessor(self, preprocessor: PreprocessModuleABC, index=0):
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
        return_metadata: bool = False,
        load: bool = False,
    ):
        """
        Build an inference dataset from this configuration.

        Parameters
        ----------
        times : Sequence[np.datetime64 | datetime.datetime | cftime.datetime] | np.ndarray | xr.DataArray
            Initialization times included in the inference dataset.
        time_features : AddedTimeFeatures
            Temporal feature generator applied to inference samples.
        return_metadata : bool
            Whether each sample is returned with its sampling-coordinate metadata.
        load : bool
            Whether underlying xarray data are loaded into memory when opened.

        Returns
        -------
        InferenceDataset
            Constructed inference dataset.
        """
        return InferenceDataset(
            config=self,
            requested_times=times,
            time_features=time_features,
            return_metadata=return_metadata,
            load=load,
        )


@dataclasses.dataclass
class InferenceDataset(DatasetABC):
    """
    Dataset for constructing model inputs during inference.

    Samples are assembled from forecast and conditioning data according to the
    configured conditioning strategy. Optional temporal features and sampling
    metadata are included with each sample.

    Parameters
    ----------
    config : InferenceDatasetConfig
        Configuration defining inference data sources and sampling behavior.
    requested_times : Sequence[np.datetime64 | datetime.datetime | cftime.datetime] | np.ndarray | xr.DataArray
        Initialization times requested for inference.
    time_features : AddedTimeFeatures
        Temporal feature generator applied to samples.
    return_metadata : bool
        Whether each sample is returned with its sampling-coordinate metadata.
    load : bool
        Whether underlying xarray data are loaded into memory when opened.
    """

    config: InferenceDatasetConfig
    requested_times: (
        Sequence[np.datetime64 | datetime.datetime | cftime.datetime]
        | np.ndarray
        | xr.DataArray
    )
    time_features: AddedTimeFeatures
    return_metadata: bool = False
    load: bool = False

    mask: xr.DataArray | None = dataclasses.field(
        init=False,
        default=None,
    )

    def __post_init__(self):
        """
        Initialize the inference dataset.

        Initializes common dataset sampling state, opens required data sources,
        constructs source indexes, and prepares temporal features.
        """
        super().__init__()

    @property
    def _load_forecast(self):
        """
        Indicate whether forecast data must be loaded separately.

        Returns
        -------
        bool
            ``True`` when forecast data are configured and are not being reused as
            the conditioning source.
        """
        return all(
            [
                not self.config._using_forecast_data_as_condition,
                self.config.forecast is not None,
            ]
        )

    @property
    def _write_condition_to_input(self):
        """
        Indicate whether conditioning data are used directly as model input.

        Returns
        -------
        bool
            ``True`` when forecast data are reused as conditioning data or when no
            forecast dataset is configured.
        """
        return any(
            [
                self.config._using_forecast_data_as_condition,
                self.config.forecast is None,
            ]
        )

    @property
    def _concat_condition_to_input(self):
        """
        Indicate whether conditioning data are concatenated with forecast input.

        Returns
        -------
        bool
            ``True`` when forecast data remain the primary input and a separate
            conditioning dataset is configured.
        """
        return (
            self._write_condition_to_input is False
            and self.config.condition is not None
        )

    def __getitem__(self, ind):
        """
        Return an inference sample.

        Forecast and conditioning data are indexed for the requested sample and
        combined according to the configured conditioning strategy. Temporal features
        are then generated and the resulting data are converted to tensors.

        Parameters
        ----------
        ind : int
            Index of the inference sample to retrieve.

        Returns
        -------
        dict | tuple[dict, dict]
            Dictionary containing the input and optional additional-feature tensors.
            When metadata are enabled, the sample dictionary is returned together
            with its sampling coordinates.
        """
        selection = {dim: value[ind] for dim, value in self.sample_coords.items()}

        condition = self._index_condition_dataset(ind)
        input = self._index_forecast_dataset(ind)

        if self._write_condition_to_input:
            input = condition

        elif self._concat_condition_to_input:
            input = xr.concat([input, condition], dim="channels")

        added_features_array = self.time_features(ind, input)

        (input_array,) = self._compute(input.data)

        datadict = dict(
            input=torch.as_tensor(input_array, dtype=torch.float32),
            added_features=torch.tensor(added_features_array, dtype=torch.float32)
            if added_features_array is not None
            else None,
        )

        if self.return_metadata:
            return datadict, selection
        else:
            return datadict


def _from_train(
    train_dataset_config: TrainDatasetConfig,
) -> "InferenceDatasetConfig":
    """
    Derive an inference dataset configuration from a training configuration.

    Forecast, conditioning, conditioning-method, and lead-time settings required
    at inference are copied from the training configuration. Reference target data
    are excluded because they are not required for prediction.

    Parameters
    ----------
    train_dataset_config : TrainDatasetConfig
        Training dataset configuration from which inference settings are derived.

    Returns
    -------
    InferenceDatasetConfig
        Inference configuration compatible with the training dataset setup.

    Raises
    ------
    ValueError
        If an inference input configuration cannot be derived from the available
        training forecast and conditioning data.
    """
    import copy

    kwargs = {
        "condition_method": train_dataset_config.condition_method,
        "lead_times": copy.deepcopy(train_dataset_config.lead_times),
    }

    train_has_reference = train_dataset_config.reference is not None
    train_has_condition = train_dataset_config.effective_condition is not None
    train_condition_from_forecast = (
        train_dataset_config._using_forecast_data_as_condition
    )

    if train_has_reference and not train_has_condition:
        kwargs["forecast"] = copy.deepcopy(train_dataset_config.forecast)

    elif train_condition_from_forecast:
        kwargs["forecast"] = copy.deepcopy(train_dataset_config.forecast)

    elif train_has_condition:
        if train_has_reference:
            kwargs["forecast"] = copy.deepcopy(train_dataset_config.forecast)
        kwargs["condition"] = copy.deepcopy(train_dataset_config.condition)

    else:
        raise ValueError(
            "Could not infer inference dataset config from training dataset config."
        )

    return InferenceDatasetConfig(**kwargs)
