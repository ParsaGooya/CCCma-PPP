import dataclasses
from pathlib import Path
import numpy as np
import xarray as xr
from torch.utils.data import Dataset
import torch


from cccma_ppp.data_modules.dataset.operator import DatasetOperator, _get_time_features
from cccma_ppp.data_modules.dataset.config_abc import (
    lead_months_config,
    DatasetConfigABC,
)

from cccma_ppp.data_modules.data import (
    DataConfigABC,
    ModelDataConfig,
    ConditionDataConfig,
)

from cccma_ppp.data_modules import (
    _unwrap_data_variables,
    _load_xarray_data,
    _create_train_mask,
)

from cccma_ppp.train.datasets import TrainDatasetConfig
from cccma_ppp.preprocessing.preprocessing_ABC import PreprocessModuleABC


@dataclasses.dataclass
class InferenceDatasetConfig(DatasetConfigABC):
    """
    Configuration for inference datasets.

    Parameters
    ----------
    model : ModelDataConfig or None, optional
        Model input dataset configuration.
    condition : ConditionDataConfig or None, optional
        Conditioning dataset configuration.
    condition_method : str or None, optional
        Method used to construct conditioning inputs.
    time_features : list of str or None, optional
        Additional temporal features.
    lead_months : lead_months_config or None, optional
        Lead months used during inference.
    """

    model: ModelDataConfig | None = None
    condition: ConditionDataConfig | None = None
    condition_method: str = None
    time_features: list[str] | None = None
    lead_months: lead_months_config | None = None

    def __post_init__(self):
        """
        Initialize and validate inference dataset configuration.

        Returns
        -------
        None
        """
        self._fitted_preprocessors: bool = False
        self._effective_condition: ConditionDataConfig | ModelDataConfig | None = None

        super().__init__()

        self._check_model()
        self._check_condition()

    def _check_model(self):
        """
        Validate model dataset configuration.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If model configuration is incompatible with the
            selected conditioning strategy.
        """
        if self.model is not None:
            if self.condition_method == "same_member":
                if self.model.ensemble_mean:
                    raise ValueError(
                        "for same member coniditioning the model data should not be ensemble mean."
                    )

        return self

    def _check_condition(self):
        """
        Validate conditioning configuration.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If conditioning configuration is inconsistent.
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
    def ds_operator(self):
        """
        Dataset operator associated with the configuration.

        Returns
        -------
        DatasetOperator
        """
        return DatasetOperator(self)

    @property
    def num_input_lead_months(self) -> int:
        """
        Number of available input lead months.

        Returns
        -------
        int
            Total number of lead months available from the
            effective input dataset.
        """
        if self.model is not None:
            return self.model.info.sizes["lead_time"]

        return self.condition.info.sizes["lead_time"]

    @property
    def get_common_time(self):
        """
        Time period shared by all required datasets.

        Returns
        -------
        np.ndarray
            Common years available across configured datasets.
        """
        year_ranges = list()
        if self.condition is not None:
            year_ranges.append(self.condition.year_range)
        if self.model is not None:
            year_ranges.append(self.model.year_range)

        common = year_ranges[0]
        for yr in year_ranges[1:]:
            common = np.intersect1d(common, yr)

        return common

    @property
    def available_inference_time(self):
        """
        Years available for inference.

        Returns
        -------
        np.ndarray
            Years with sufficient lead-time coverage.
        """
        num_lead_years = max(self.lead_months) // 12
        return np.arange(
            np.min(self.get_common_time),
            np.max(self.get_common_time) + 1 - num_lead_years + 1,
        )

    def _load_fitted_preprocessors(
        self,
        load_dir: Path | str | None = None,
    ):
        """
        Load fitted preprocessing pipelines.

        Parameters
        ----------
        load_dir : pathlib.Path or str or None, optional
            Directory containing saved preprocessing pipelines.

        Returns
        -------
        None
        """
        self.ds_operator._load_fitted_preprocessors(load_dir)

    def _add_fitted_preprocessor(
        self,
        preprocessor: PreprocessModuleABC,
        index=0,
    ):
        """
        Add a fitted preprocessor.

        Parameters
        ----------
        preprocessor : PreprocessModuleABC
            Fitted preprocessor instance.
        index : int, optional
            Insertion position.

        Returns
        -------
        None
        """

        self.ds_operator._add_fitted_preprocessor(preprocessor, index)

    @classmethod
    def read_from_train(
        cls,
        train_dataset_config: TrainDatasetConfig,
    ):
        """
        Create an inference configuration from training configuration.

        Parameters
        ----------
        train_dataset_config : TrainDatasetConfig
            Training dataset configuration.

        Returns
        -------
        InferenceDatasetConfig
        """
        return _from_train(train_dataset_config)

    def build_dataset(
        self,
        years: np.ndarray,
        return_metadata: bool = False,
    ):
        """
        Construct inference dataset.

        Parameters
        ----------
        years : np.ndarray
            Years to include.
        return_metadata : bool, optional
            Whether metadata should be returned with samples.

        Returns
        -------
        InferenceDataset
        """
        return InferenceDataset(
            config=self,
            requested_years=years,
            return_metadata=return_metadata,
        )


@dataclasses.dataclass
class InferenceDataset(Dataset):
    """
    Dataset used during inference.

    Parameters
    ----------
    config : InferenceDatasetConfig
        Dataset configuration.
    requested_years : array-like
        Years to evaluate.
    return_metadata : bool, optional
        Whether to return metadata for each sample.
    """

    config: InferenceDatasetConfig
    requested_years: list[int] | tuple[int] | np.ndarray
    return_metadata: bool = False

    def __post_init__(self):
        """
        Initialize inference dataset.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If preprocessors have not been fitted.
        ValueError
            If requested years are unavailable.
        """
        if not self.config._fitted_preprocessors:
            raise RuntimeError(
                "Make sure to fit preprocessors first!. Hint:  TrainDatasetConfig._fit_preprocessors()"
            )
        if not set(self.requested_years).issubset(
            set(self.config.available_train_time)
        ):
            raise ValueError(
                "the requested years are not common to model and condition data."
            )

        self.model_dataset = self.condition_dataset = None

        if self.config.model is not None:
            self.model_dataset = self._load_xarray_data(self.config.model)

        if self.config.effective_condition is not None:
            self.condition_dataset = self._load_xarray_data(
                self.config.effective_condition
            )

        self.mask = self._prepare_mask()
        self.model_indexes = self.get_model_indexes()
        self.cond_indexes = self.get_cond_indexes(self.model_indexes)

    @property
    def effective_input(self):
        """
        Effective input dataset configuration.

        Returns
        -------
        ModelDataConfig or ConditionDataConfig
        """
        if self.config.model is not None:
            return self.config.model
        else:
            return self.config.condition

    @property
    def _load_model(self):
        """
        Determine whether model data must be loaded.

        Returns
        -------
        bool
        """

        return not self.config._using_model_data_as_condition

    @property
    def _write_condition_to_input(self):
        """
        Determine whether condition data replaces model input.

        Returns
        -------
        bool
        """

        return self.config._using_model_data_as_condition

    @property
    def _concat_condition_to_input(self):
        """
        Determine whether condition data should be concatenated
        with model input.

        Returns
        -------
        bool
        """

        return (
            self._write_condition_to_input is False
            and self.config.effective_condition is not None
        )

    def _prepare_mask(self):
        """
        Construct inference mask.

        Returns
        -------
        xr.DataArray
            Mask defining valid inference samples.
        """
        mask = _create_train_mask(
            years=self.effective_input.year_range,
            lead_times=np.arange(1, self.effective_input.info.sizes["lead_time"] + 1),
        )
        mask = xr.full_like(mask, fill_value=False)

        mask = mask.sel(year=self.requested_years).sel(
            lead_time=self.config.lead_months
        )
        if all(
            [
                not self.effective_input.ensemble_mean,
                self.effective_input.info.coords["ensembles"] is not None,
            ]
        ):
            mask = mask.expand_dims(
                ensembles=len(self.effective_input.info.coords["ensembles"]), axis=0
            )
            mask = mask.assign_coords(
                ensembles=self.effective_input.info.coords["ensembles"]
            )

        mask = mask.where(~mask)

        return mask

    def _load_xarray_data(self, config: DataConfigABC):
        """
        Load xarray dataset.

        Parameters
        ----------
        config : DataConfigABC

        Returns
        -------
        xr.Dataset or xr.DataArray
        """

        return _load_xarray_data(
            config.list_paths,
            names=config.names,
            ensemble_mean=config.ensemble_mean,
            selection={"ensembles": config.info.coords["ensembles"]}
            if config.info.coords["ensembles"] is not None
            else None,
            concat_dim=config.concat_dim,
            rename_dict=config.rename_dict,
        )

    def get_model_indexes(self):
        """
        Generate sample indexes.

        Returns
        -------
        dict
            Mapping of dimension names to index arrays.
        """

        mask = (
            self.mask.stack(batch=dict(self.mask.sizes).keys())
            .transpose("batch", ...)
            .dropna(dim="batch")
            .batch.values
        )
        mask = tuple(map(np.array, zip(*mask)))
        indexes = {
            key: mask[ind]
            for ind, key in enumerate(tuple(dict(self.mask.sizes).keys()))
        }

        return indexes

    def get_cond_indexes(self, model_indexes: dict):
        """
        Generate condition dataset indexes.

        Parameters
        ----------
        model_indexes : dict

        Returns
        -------
        dict or None
        """

        if self.condition_dataset is not None:
            if self.config.condition_method != "static":
                indexes = {}
                indexes["year"] = model_indexes["year"]
                indexes["lead_time"] = model_indexes["lead_time"]
                if self.config.condition_method == "cross_ensemble":
                    ens_inds = [
                        np.random.choice(
                            self.config.effective_condition.info.coords["ensembles"]
                        )
                        for _ in range(len(model_indexes["year"]))
                    ]
                    indexes["ensembles"] = np.array(ens_inds)

                elif self.config.condition_method == "same_member":
                    indexes["ensembles"] = model_indexes["ensembles"]

                return indexes

    def get_input_shape(self):
        """
        Determine model input shape.

        Returns
        -------
        tuple
            Expected input tensor shape.
        """

        from cccma_ppp.preprocessing.utils_preprocessing import Flattennanremove

        checklist = [
            isinstance(item, Flattennanremove)
            for item in self.effective_input.preprocessing_pipeline.fitted_preprocessors
        ]

        len_names = len(self.effective_input.names)
        if self._concat_condition_to_input:
            len_names += len(self.config.effective_condition.names)

        if any(checklist):
            return (
                self.effective_input.preprocessing_pipeline.get_preprocessors(
                    "flattener"
                ).final_locations.size
                * len_names,
            )
        else:
            return (
                self.effective_input.info.coords["lat"].size,
                self.effective_input.info.coords["lon"].size,
            )

    def get_added_features_dim(self):
        """
        Number of additional temporal features.

        Returns
        -------
        int
        """

        return (
            0 if self.config.time_features is None else len(self.config.time_features)
        )

    def _index_condition_dataset(self, ind):
        """
        Retrieve a conditioned sample.

        Parameters
        ----------
        ind : int

        Returns
        -------
        xr.DataArray or None
        """

        if self.condition_dataset is not None:
            if self.config.condition_method != "static":
                year = float(self.cond_indexes["year"][ind])
                lead_time = float(self.cond_indexes["lead_time"][ind])
                selection = dict(year=year, lead_time=lead_time)
                if "ensembles" in self.cond_indexes:
                    selection["ensembles"] = self.cond_indexes["ensembles"][ind]

            else:
                selection = {}

            condition = (
                self.config.effective_condition.preprocessing_pipeline.transform(
                    self.condition_dataset.sel(**selection)
                )
            )
            condition = _unwrap_data_variables(condition)

            return condition

    def _index_model_dataset(self, ind):
        """
        Retrieve a model input sample.

        Parameters
        ----------
        ind : int

        Returns
        -------
        xr.DataArray or None
        """

        if self._load_model:
            year = float(self.model_indexes["year"][ind])
            lead_time = float(self.model_indexes["lead_time"][ind])
            selection = dict(year=year, lead_time=lead_time)

            if "ensembles" in self.model_indexes:
                selection["ensembles"] = self.model_indexes["ensembles"][ind]

            model = self.config.model.preprocessing_pipeline.transform(
                self.model_dataset.sel(**selection)
            )
            model = _unwrap_data_variables(model)

            return model

    def __getitem__(self, ind):
        """
        Retrieve a dataset sample.

        Parameters
        ----------
        ind : int
            Sample index.

        Returns
        -------
        dict or tuple
            Sample dictionary, optionally paired with metadata.
        """
        year = float(self.model_indexes["year"][ind])
        lead_time = float(self.model_indexes["lead_time"][ind])
        selection = dict(year=year, lead_time=lead_time)

        condition = self._index_condition_dataset(ind)
        input = self._index_model_dataset(ind)

        if self._write_condition_to_input:
            input = condition

        elif self._concat_condition_to_input:
            input = xr.concat([input, condition], dim="channels")

        time_features = _get_time_features(self.config, year, lead_time, input)

        datadict = dict(
            input=torch.as_tensor(input.to_numpy(), dtype=torch.float32),
            added_features=torch.as_tensor(time_features, dtype=torch.float32)
            if time_features is not None
            else None,
        )

        if self.return_metadata:
            return datadict, selection
        else:
            return datadict

    def __len__(self):
        """
        Dataset length.

        Returns
        -------
        int
        """
        return len(self.model_indexes.get(list(self.model_indexes.keys())[0]))


def _from_train(
    train_dataset_config: TrainDatasetConfig,
) -> "InferenceDatasetConfig":
    """
    Create an inference dataset configuration from a training
    dataset configuration.

    Parameters
    ----------
    train_dataset_config : TrainDatasetConfig
        Source training configuration.

    Returns
    -------
    InferenceDatasetConfig

    Raises
    ------
    ValueError
        If an inference configuration cannot be inferred from
        the training configuration.
    """

    import copy

    kwargs = {
        "condition_method": train_dataset_config.condition_method,
        "time_features": copy.deepcopy(train_dataset_config.time_features),
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
