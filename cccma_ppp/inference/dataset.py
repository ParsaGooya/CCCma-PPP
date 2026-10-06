import dataclasses
from pathlib import Path
from collections.abc import Sequence
import numpy as np
import xarray as xr
import torch
import cftime
import datetime

from cccma_ppp.data_modules.dataset.dataset_abc import (
    DatasetConfigABC,
    DatasetABC,
    lead_time_config,
    AddedTimeFeatures,
)

from cccma_ppp.data_modules.dataset.operator import DatasetOperator

from cccma_ppp.data_modules.data.data_configs import (
    ModelDataConfig,
    ConditionDataConfig,
)

from cccma_ppp.train.dataset import TrainDatasetConfig
from cccma_ppp.preprocessing.preprocessing_ABC import PreprocessModuleABC


@dataclasses.dataclass
class InferenceDatasetConfig(DatasetConfigABC):
    """
    Configuration for an inference dataset.

    Parameters
    ----------
    model : ModelDataConfig | None
        Model data configuration used as an inference input when available.
    condition : ConditionDataConfig | None
        Conditioning data configuration used during inference.
    condition_method : str | None
        Method used to construct conditioning samples.
    lead_times : lead_time_config | None
        Lead times included in inference samples. If ``None``, all available
        input lead times are used.
    """

    model: ModelDataConfig | None = None
    condition: ConditionDataConfig | None = None
    condition_method: str = None
    lead_times: lead_time_config | None = None

    def __post_init__(self):
        """
        Initialize and validate the inference dataset configuration.

        Resolves the effective input and conditioning data, validates conditioning
        settings, and resolves the requested lead times.
        """
        super().__init__()

    @property
    def effective_input(self):
        """
        Return the data configuration used as the inference input.

        Model data are preferred when configured; otherwise, conditioning data are
        used directly as the input.

        Returns
        -------
        ModelDataConfig | ConditionDataConfig | None
            Data configuration used to construct inference inputs.
        """
        if self.model is not None:
            return self.model
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

        When both model and conditioning data are configured, only initialization
        times available in both data sources are returned.

        Returns
        -------
        Any
            Index containing initialization times common to all configured inference
            data sources.
        """
        time_coords = list()
        if self.condition is not None:
            time_coords.append(
                self.condition.coords[self.init_time_dim].to_index(),
            )
        if self.model is not None:
            time_coords.append(
                self.model.coords[self.init_time_dim].to_index(),
            )

        common = time_coords[0]
        for tr in time_coords[1:]:
            common = common.intersection(tr)

        return common

    def load_fitted_preprocessors(self, load_dir: Path | str | None = None):
        """
        Load fitted preprocessing pipelines for the inference data sources.

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
            Position at which the preprocessing module is inserted into each
            applicable pipeline.
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
            Initialization times to include in the inference dataset.
        time_features : AddedTimeFeatures
            Temporal feature generator applied to inference samples.
        return_metadata : bool
            Whether each sample is returned with its sampling-coordinate metadata.
        load : bool
            Whether the underlying xarray data are loaded into memory when opened.

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

    Samples are constructed from model data, conditioning data, or a concatenation
    of both according to the inference configuration. Optional temporal features
    and sampling metadata can also be returned.

    Parameters
    ----------
    config : InferenceDatasetConfig
        Configuration defining inference data sources and sampling behavior.
    requested_times : Sequence[np.datetime64 | datetime.datetime | cftime.datetime] | np.ndarray | xr.DataArray
        Initialization times requested for inference.
    time_features : AddedTimeFeatures
        Temporal feature generator applied to inference samples.
    return_metadata : bool
        Whether each sample is returned with its sampling-coordinate metadata.
    load : bool
        Whether the underlying xarray data are loaded into memory when opened.
    mask : xr.DataArray | None
        Sampling mask resolved during dataset initialization.
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

        Validates the requested sampling configuration, opens the required data
        sources, constructs valid sampling coordinates and indexes, and builds
        temporal features.
        """
        super().__init__()

    @property
    def _load_model(self):
        """
        Indicate whether model data must be loaded separately.

        Returns
        -------
        bool
            ``True`` when model data are configured and are not being reused as the
            conditioning source, otherwise ``False``.
        """
        return all(
            [
                not self.config._using_model_data_as_condition,
                self.config.model is not None,
            ]
        )

    @property
    def _write_condition_to_input(self):
        """
        Indicate whether conditioning data are used directly as the model input.

        Returns
        -------
        bool
            ``True`` when model data are being used as conditioning data or when no
            separate model data are configured.
        """
        return any(
            [self.config._using_model_data_as_condition, self.config.model is None]
        )

    @property
    def _concat_condition_to_input(self):
        """
        Indicate whether conditioning data are concatenated with model inputs.

        Returns
        -------
        bool
            ``True`` when model data form the primary input and a separate
            conditioning dataset is configured.
        """
        return (
            self._write_condition_to_input is False
            and self.config.condition is not None
        )

    def __getitem__(self, ind):
        """
        Return an inference sample.

        Model and conditioning values are indexed for the requested sample and
        combined according to the configured conditioning behavior. Temporal features
        are added separately from the model input.

        Parameters
        ----------
        ind : int
            Index of the inference sample to retrieve.

        Returns
        -------
        dict | tuple[dict, dict]
            Dictionary containing the input tensor and optional additional temporal
            features. When metadata are enabled, the sample dictionary is returned
            together with its sampling coordinates.
        """
        selection = {dim: value[ind] for dim, value in self.sample_coords.items()}

        condition = self._index_condition_dataset(ind)
        input = self._index_model_dataset(ind)

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
    Construct an inference dataset configuration from a training configuration.

    The inference data sources are selected according to whether training used
    observations, explicit conditioning data, or model data as the conditioning
    source. The conditioning method and lead-time selection are preserved.

    Parameters
    ----------
    train_dataset_config : TrainDatasetConfig
        Training dataset configuration from which inference settings are derived.

    Returns
    -------
    InferenceDatasetConfig
        Inference configuration containing deep copies of the required data
        sources and lead-time settings.

    Raises
    ------
    ValueError
        If an inference data configuration cannot be determined from the training
        dataset configuration.
    """
    import copy

    kwargs = {
        "condition_method": train_dataset_config.condition_method,
        "lead_times": copy.deepcopy(train_dataset_config.lead_times),
    }

    train_has_observation = train_dataset_config.observation is not None
    train_has_condition = train_dataset_config.effective_condition is not None
    train_condition_from_model = train_dataset_config._using_model_data_as_condition

    if train_has_observation and not train_has_condition:
        kwargs["model"] = copy.deepcopy(train_dataset_config.model)

    elif train_condition_from_model:
        kwargs["model"] = copy.deepcopy(train_dataset_config.model)

    elif train_has_condition:
        if train_has_observation:
            kwargs["model"] = copy.deepcopy(train_dataset_config.model)
        kwargs["condition"] = copy.deepcopy(train_dataset_config.condition)

    else:
        raise ValueError(
            "Could not infer inference dataset config from training dataset config."
        )

    return InferenceDatasetConfig(**kwargs)
