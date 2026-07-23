import dataclasses
from pathlib import Path
import numpy as np
import xarray as xr
import torch
import dask

from cccma_ppp.data_modules.dataset.config_abc import (
    DatasetConfigABC,
    lead_months_config,
)
from cccma_ppp.data_modules.dataset.dataset_abc import (
    DatasetABC,
    AddedTimeFeatures,
)
from cccma_ppp.data_modules.dataset.operator import DatasetOperator

from cccma_ppp.data_modules.data.data_configs import (
    ModelDataConfig,
    ConditionDataConfig,
)

from cccma_ppp.data_modules.utils import suppress_stderr

from cccma_ppp.train.dataset import TrainDatasetConfig
from cccma_ppp.preprocessing.preprocessing_ABC import PreprocessModuleABC


@dataclasses.dataclass
class InferenceDatasetConfig(DatasetConfigABC):
    """
    Configuration for an inference dataset.

    Parameters
    ----------
    model : ModelDataConfig or None, optional
        Configuration of the model dataset used as inference input.
    condition : ConditionDataConfig or None, optional
        Configuration of the conditioning dataset.
    condition_method : str or None, optional
        Method used to incorporate conditioning data.
    lead_months : lead_months_config or None, optional
        Lead months selected for inference.

    """

    model: ModelDataConfig | None = None
    condition: ConditionDataConfig | None = None
    condition_method: str = None
    lead_months: lead_months_config | None = None

    def __post_init__(self):
        """
        Initialize and validate the inference dataset configuration.

        """

        super().__init__()

    @property
    def effective_input(self):
        """
        Return the effective input-data configuration.

        Returns
        -------
        ModelDataConfig or ConditionDataConfig
            Model configuration when model data are available, otherwise the
            conditioning-data configuration.

        """
        if self.model is not None:
            return self.model
        else:
            return self.condition

    @property
    def ds_operator(self):
        """
        Return the dataset operator.

        Returns
        -------
        DatasetOperator
            Operator configured for this inference dataset.

        """
        return DatasetOperator(self)

    @property
    def available_times(self):
        """
        Return the times shared by the configured data sources.

        Returns
        -------
        numpy.ndarray
            Intersection of the available year coordinates across the model and
            conditioning datasets.

        """

        year_ranges = list()
        if self.condition is not None:
            year_ranges.append(
                self.condition.info.coords["year"].values,
            )
        if self.model is not None:
            year_ranges.append(
                self.model.info.coords["year"].values,
            )

        common = year_ranges[0]
        for yr in year_ranges[1:]:
            common = np.intersect1d(common, yr)

        return common

    def _check_model(self):
        """
        Validate model dataset configuration.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If configuration is inconsistent.
        """

        if self.condition_method == "same_member":
            if self.model.ensemble_mean:
                raise ValueError(
                    "for same member coniditioning the model data should not be ensemble mean."
                )

        return self

    def _check_condition(self):
        """
        Validate conditioning dataset configuration.

        Returns
        -------
        self
            The validated instance.

        Raises
        ------
        ValueError
            If the conditioning dataset configuration is invalid.
        """
        if self.effective_condition is not None:
            if self.condition_method is None:
                raise ValueError(
                    "You must specify condition_method for conditioning dataset!"
                )

            if self.condition_method in ["cross_ensemble", "same_member"]:
                if self.effective_condition.ensemble_mean:
                    raise ValueError(
                        "condition ensemble_mean cannot be True for cross_ensemble or same_member conditioning."
                    )
                if self.effective_condition.info.coords["ensembles"] is None:
                    raise ValueError(
                        "For cross_ensemble or same_member conditioning an ensembles dim must exist in the condition."
                    )
            elif self.condition_method == "ensemble_mean":
                if self.effective_condition.ensemble_mean is not True:
                    raise ValueError(
                        "Ensemble mean must be True for ensemble_mean conditioning."
                    )
            else:
                if self.effective_condition.ensemble_list is not None:
                    raise ValueError(
                        'For "static" conditioning fields cannot specify ensemble list.'
                    )
                if self._using_model_data_as_condition:
                    raise ValueError(
                        "'static' conditioning method cannot point to the same model data!"
                    )

        else:
            if self.condition_method == "static":
                raise ValueError(
                    "For static conditioning method condition dataset must be specified!"
                )

        return self

    @property
    def num_input_lead_months(self) -> int:
        """
        Number of lead months in model dataset.

        Returns
        -------
        int
        """

        return self.model.info.sizes["lead_time"]

    def load_fitted_preprocessors(
        self,
        load_dir: Path | str | None = None,
    ):
        """
        Load fitted preprocessing pipelines.

        Parameters
        ----------
        load_dir : pathlib.Path, str, or None, optional
            Directory containing the fitted preprocessing pipelines. If ``None``,
            the default preprocessing directory is used.

        """
        self.ds_operator.load_fitted_preprocessors(load_dir)

    def add_fitted_preprocessor(
        self,
        preprocessor: PreprocessModuleABC,
        index=0,
    ):
        """
        Add a fitted preprocessor to the dataset pipelines.

        Parameters
        ----------
        preprocessor : PreprocessModuleABC
            Fitted preprocessing module to add.
        index : int, optional
            Position at which the preprocessor is inserted.

        """

        self.ds_operator.add_fitted_preprocessor(preprocessor, index)

    def build_dataset(
        self,
        years: np.ndarray,
        time_features: AddedTimeFeatures,
        return_metadata: bool = False,
        load: bool = False,
    ):
        """
        Construct an inference dataset.

        Parameters
        ----------
        years : numpy.ndarray
            Years selected for inference.
        time_features : AddedTimeFeatures
            Generator for additional temporal features.
        return_metadata : bool, optional
            Whether each sample includes its coordinate metadata.
        load : bool, optional
            Whether the underlying data are loaded eagerly into memory.

        Returns
        -------
        InferenceDataset
            Configured inference dataset.

        """
        return InferenceDataset(
            config=self,
            requested_years=years,
            time_features=time_features,
            return_metadata=return_metadata,
            load=load,
        )


@dataclasses.dataclass
class InferenceDataset(DatasetABC):
    """
    Dataset used to prepare samples for model inference.

    Parameters
    ----------
    config : InferenceDatasetConfig
        Inference dataset configuration.
    requested_years : list of int, tuple of int, or numpy.ndarray
        Years selected for inference.
    time_features : AddedTimeFeatures
        Generator for additional temporal features.
    return_metadata : bool, optional
        Whether each sample includes its coordinate metadata.
    load : bool, optional
        Whether the underlying data are loaded eagerly into memory.
    mask : xarray.DataArray or None, optional
        Sampling mask initialized automatically by the dataset.

    """

    config: InferenceDatasetConfig
    requested_years: list[int] | tuple[int, ...] | np.ndarray
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

        """
        super().__init__()

    @property
    def _load_model(self):
        """
        Indicate whether model data should be loaded separately.

        Returns
        -------
        bool
            ``True`` when model data are available and are not reused as
            conditioning data.

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
        Indicate whether conditioning data should replace the model input.

        Returns
        -------
        bool
            ``True`` when model data are reused as conditioning data or no model
            dataset is configured.

        """

        return any(
            [self.config._using_model_data_as_condition, self.config.model is None]
        )

    @property
    def _concat_condition_to_input(self):
        """
        Indicate whether conditioning data should be concatenated with model input.

        Returns
        -------
        bool
            ``True`` when model and independent conditioning data are both used as
            input.

        """

        return (
            self._write_condition_to_input is False
            and self.config.condition is not None
        )

    def __getitem__(self, ind):
        """
        Return one inference sample.

        Parameters
        ----------
        ind : int
            Positional index of the requested sample.

        Returns
        -------
        dict or tuple of dict and dict
            Dictionary containing the input tensor and optional temporal features.
            If metadata are requested, the dictionary is returned with its sample
            coordinate selection.

        """

        selection = {dim: value[ind] for dim, value in self.sample_coords.items()}

        condition = self._index_condition_dataset(ind)
        input = self._index_model_dataset(ind)

        if self._write_condition_to_input:
            input = condition

        elif self._concat_condition_to_input:
            input = xr.concat([input, condition], dim="channels")

        time_features = self.time_features(selection, input)

        with suppress_stderr():
            input = dask.compute(
                input.data,
            )

        datadict = dict(
            input=torch.as_tensor(input, dtype=torch.float32),
            added_features=torch.as_tensor(time_features, dtype=torch.float32)
            if time_features is not None
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

    Parameters
    ----------
    train_dataset_config : TrainDatasetConfig
        Training dataset configuration from which inference settings are
        derived.

    Returns
    -------
    InferenceDatasetConfig
        Inference configuration containing copied model, condition, lead-month,
        and conditioning-method settings.

    Raises
    ------
    ValueError
        If an inference configuration cannot be inferred from the training
        dataset configuration.

    """

    import copy

    kwargs = {
        "condition_method": train_dataset_config.condition_method,
        "lead_months": copy.deepcopy(train_dataset_config.lead_months),
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
