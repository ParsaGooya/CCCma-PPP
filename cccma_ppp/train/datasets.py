from __future__ import annotations
import numpy as np
import xarray as xr
from torch.utils.data import Dataset
import torch
import dataclasses
import warnings
from pathlib import Path

from cccma_ppp.data_modules.dataset.config_abc import (
    DatasetConfigABC,
    lead_months_config,
)
from cccma_ppp.data_modules.dataset.operator import DatasetOperator, _get_time_features
from cccma_ppp.data_modules.data.data_configs import (
    ModelDataConfig,
    ObsDataConfig,
    ConditionDataConfig,
)

from cccma_ppp.data_modules.utils import (
    _unwrap_data_variables,
    _load_xarray_data,
    _create_train_mask,
)


@dataclasses.dataclass
class TrainDatasetConfig(DatasetConfigABC):
    """
    Configuration for training dataset construction.

    Parameters
    ----------
    model : ModelDataConfig
        Model dataset configuration.
    observation : ObsDataConfig or None, optional
        Observation dataset configuration.
    condition : ConditionDataConfig or None, optional
        Conditioning dataset configuration.
    condition_method : str or None, optional
        Method for conditioning (e.g., "cross_ensemble", "same_member", "static").
    time_features : list of str or None, optional
        Time-based features to include.
    lead_months : array-like or None, optional
        Lead months to use.
    """

    model: ModelDataConfig
    observation: ObsDataConfig | None = None
    condition: ConditionDataConfig | None = None
    condition_method: str = None
    time_features: list[str] | None = None
    lead_months: lead_months_config | None = None

    def __post_init__(self):
        """
        Initialize and validate dataset configuration.

        Returns
        -------
        self
        """
        self._fitted_preprocessors: bool = False
        self._effective_condition: ConditionDataConfig | ModelDataConfig | None = None

        super().__init__()

        self._check_model()
        self._check_observation()

        self._check_condition()
        return self

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

    def _check_observation(self):
        """
        Validate observation dataset configuration.

        Returns
        -------
        self

        Raises
        ------
        AssertionError
            If required observation data is missing.
        """
        if self.observation is not None:
            if not self.observation.info.coords["lat"].equals(
                self.model.info.coords["lat"]
            ):
                warnings.warn(
                    "model and observation data do not have the same latitudes cooridnates."
                )
            if not self.observation.info.coords["lon"].equals(
                self.model.info.coords["lon"]
            ):
                warnings.warn(
                    "model and observation data do not have the same longitudes cooridnates."
                )

        else:
            assert self.condition_method is not None, (
                "No target observation is specified. Specify condition_method!"
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
    def ds_operator(self):
        """
        Access dataset operator.

        Returns
        -------
        DatasetOperator
        """

        return DatasetOperator(self)

    @property
    def num_input_lead_months(self) -> int:
        """
        Number of lead months in model dataset.

        Returns
        -------
        int
        """

        return self.model.info.sizes["lead_time"]

    @property
    def get_common_time(self):
        """
        Compute common time range.

        Returns
        -------
        np.ndarray
        """

        if self.observation is not None:
            return np.intersect1d(self.model.year_range, self.observation.year_range)
        else:
            return self.model.year_range

    @property
    def available_train_time(self):
        """
        Available training years.

        Returns
        -------
        np.ndarray
        """

        num_lead_years = max(self.lead_months) // 12
        if self.observation is None:
            return np.arange(
                np.min(self.get_common_time),
                np.max(self.get_common_time) + 1 - num_lead_years + 1,
            )
        else:
            return self.get_common_time

    def _fit_preprocessors(
        self,
        train_years,
        save=False,
        save_path=None,
        save_name=None,
    ):
        """
        Fit preprocessing pipeline.

        Returns
        -------
        None
        """
        self.ds_operator._fit_preprocessors(
            train_years=train_years,
            save=save,
            save_path=save_path,
            save_name=save_name,
        )

    def _load_fitted_preprocessors(self, load_dir: Path | str | None = None):
        """
        Load fitted preprocessors.

        Returns
        -------
        None
        """
        self.ds_operator._load_fitted_preprocessors(load_dir)

    def _add_fitted_preprocessor(self, preprocessor, index=0):
        """
        Add fitted preprocessor.

        Parameters
        ----------
        preprocessor : PreprocessModuleABC
        index : int, optional

        Returns
        -------
        None
        """

        self.ds_operator._add_fitted_preprocessor(preprocessor, index)

    def build_dataset(
        self,
        years,
        mask=None,
        return_metadata=False,
    ):
        """
        Construct training dataset.

        Returns
        -------
        TrainDataset
        """
        return TrainDataset(
            config=self,
            requested_years=years,
            mask=mask,
            return_metadata=return_metadata,
        )


@dataclasses.dataclass
class TrainDataset(Dataset):
    """
    Training dataset for model learning.

    Parameters
    ----------
    config : TrainDatasetConfig
    requested_years : array-like
    mask : xr.DataArray or None, optional
    return_metadata : bool, optional
    """

    config: TrainDatasetConfig
    requested_years: list[int] | tuple[int] | np.ndarray
    mask: xr.DataArray = None
    return_metadata: bool = False

    def __post_init__(self):
        """
        Initialize dataset and load required data.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If preprocessors are not fitted.
        ValueError
            If requested years are invalid.
        """
        if not self.config._fitted_preprocessors:
            raise RuntimeError(
                "Make sure to fit preprocessors first!. Hint:  TrainDatasetConfig._fit_preprocessors()"
            )
        if not set(self.requested_years).issubset(
            set(self.config.available_train_time)
        ):
            raise ValueError(
                "the requested years are not common to input and target data."
            )

        self.observation_dataset = self.condition_dataset = None

        self.model_dataset = self._load_xarray_data(self.config.model)

        if self.config.observation is not None:
            self.observation_dataset = self._load_xarray_data(self.config.observation)

        if self.config.effective_condition is not None:
            self.condition_dataset = self._load_xarray_data(
                self.config.effective_condition
            )

        self.mask = self._prepare_mask()
        self.model_indexes = self.get_model_indexes()
        self.obs_indexes = self.get_obs_indexes(self.model_indexes)
        self.cond_indexes = self.get_cond_indexes(self.model_indexes)

    @property
    def _autoencoding_model_data(self):
        """
        Whether model is used for autoencoding.

        Returns
        -------
        bool
        """
        return self.config.observation is None

    @property
    def _load_model(self):
        """
        Determine whether the model dataset should be loaded.

        Returns
        -------
        bool
            True if the model dataset needs to be loaded.

        Notes
        -----
        This returns ``True`` in either of the following cases:

        - A condition dataset different from the model dataset is provided
        (i.e., ``_using_model_data_as_condition`` is ``False``), regardless
        of whether observations are provided.
        - No observation dataset is provided, meaning the model data is being
        autoencoded (the condition method is already validated in the
        configuration), regardless of whether a standalone condition dataset
        is provided.
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
        Determine whether the condition data replaces the model input.

        Returns
        -------
        bool
            True if the condition data should be used as the sole input to the
            machine learning model.

        Notes
        -----
        This returns ``True`` in either of the following cases:

        - No standalone condition dataset is provided, but a condition method is specified. In this case, ``_using_model_data_as_condition`` is ``True`` and the condition is derived from the model data. The model dataset will only be loaded if required.
        - A standalone condition dataset is provided, but no observation dataset is available. In this case, the model data is being autoencoded, so both the model and condition datasets must be loaded.
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
        Determine whether the condition data should be concatenated to the input.

        Returns
        -------
        bool
            True if the condition data should be concatenated to the model input.

        Notes
        -----
        This returns ``True`` when all of the following datasets are available:

        - A standalone condition dataset
        - A model dataset
        - An observation dataset

        In this case, ``_write_condition_to_input`` is ``False`` and ``effective_condition`` is available separately from the model input.
        """

        return (
            self._write_condition_to_input is False
            and self.config.effective_condition is not None
        )

    def _prepare_mask(self):
        """
        Prepare dataset mask.

        Returns
        -------
        xr.DataArray
        """
        mask = self.mask
        if mask is None:
            mask = _create_train_mask(
                years=self.config.model.year_range,
                lead_times=np.arange(1, self.config.model.info.sizes["lead_time"] + 1),
            )
            mask = xr.full_like(mask, fill_value=False)

        mask = mask.sel(year=self.requested_years).sel(
            lead_time=self.config.lead_months
        )
        if all(
            [
                not self.config.model.ensemble_mean,
                self.config.model.info.coords["ensembles"] is not None,
            ]
        ):
            mask = mask.expand_dims(
                ensembles=len(self.config.model.info.coords["ensembles"]), axis=0
            )
            mask = mask.assign_coords(
                ensembles=self.config.model.info.coords["ensembles"]
            )

        mask = mask.where(~mask)

        return mask

    def _load_xarray_data(self, config):
        """
        Load dataset from xarray sources.

        Returns
        -------
        xr.DataArray
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
        Compute indexes for model dataset.

        Returns
        -------
        dict
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

    def get_obs_indexes(self, model_indexes):
        """
        Compute indexes for observation dataset.

        Returns
        -------
        dict or None
        """

        if self.observation_dataset is not None:
            indexes = {}
            offset_year, month = np.divmod(model_indexes["lead_time"] - 0.5, 12)
            indexes["year"] = model_indexes["year"] + offset_year
            indexes["month"] = month + 0.5
            if all(
                [
                    not self.config.observation.ensemble_mean,
                    self.config.observation.info.coords["ensembles"] is not None,
                ]
            ):
                ens_inds = [
                    np.random.choice(self.config.observation.info.coords["ensembles"])
                    for _ in range(len(model_indexes["year"]))
                ]
                indexes["ensembles"] = np.array(ens_inds)

            return indexes

    def get_cond_indexes(self, model_indexes):
        """
        Compute indexes for conditioning dataset.

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

        return None

    def get_input_shape(self):
        """
        Determine input shape.

        Returns
        -------
        tuple
        """

        from cccma_ppp.preprocessing.utils_preprocessing import Flattennanremove

        checklist = [
            isinstance(item, Flattennanremove)
            for item in self.config.model.preprocessing_pipeline.fitted_preprocessors
        ]

        len_names = len(self.config.model.names)
        if self._concat_condition_to_input:
            len_names += len(self.config.effective_condition.names)

        if any(checklist):
            return (
                self.config.model.preprocessing_pipeline.get_preprocessors(
                    "flattener"
                ).final_locations.size
                * len_names,
            )

        else:
            return (
                self.config.model.info.coords["lat"].size,
                self.config.model.info.coords["lon"].size,
            )

    def get_target_shape(self):
        """
        Determine target shape.

        Returns
        -------
        tuple
        """

        from cccma_ppp.preprocessing.utils_preprocessing import Flattennanremove

        if self.observation_dataset is not None:
            checklist = [
                isinstance(item, Flattennanremove)
                for item in self.config.observation.preprocessing_pipeline.fitted_preprocessors
            ]

            if any(checklist):
                return self.config.observation.preprocessing_pipeline.get_preprocessors(
                    "flattener"
                ).final_locations.shape * len(self.config.observation.names)
            else:
                return (
                    self.config.observation.info.coords["lat"].size,
                    self.config.observation.info.coords["lon"].size,
                )
        else:
            return self.input_shape

    def get_added_features_dim(self):
        """
        Number of additional features.

        Returns
        -------
        int
        """

        return (
            0 if self.config.time_features is None else len(self.config.time_features)
        )

    def _index_condition_dataset(self, ind):
        """
        Index condition dataset.

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

    def _index_observation_dataset(self, ind):
        """
        Index observation dataset.

        Returns
        -------
        xr.DataArray or None
        """

        if self.observation_dataset is not None:
            year = float(self.obs_indexes["year"][ind])
            month = float(self.obs_indexes["month"][ind])
            selection = dict(year=year, month=month)

            if "ensembles" in self.obs_indexes:
                selection["ensembles"] = self.obs_indexes["ensembles"][ind]

            obs = self.config.observation.preprocessing_pipeline.transform(
                self.observation_dataset.sel(**selection)
            )
            obs = _unwrap_data_variables(obs)
            return obs

    def _index_model_dataset(self, ind):
        """
        Index model dataset.

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
        Retrieve dataset sample.

        Parameters
        ----------
        ind : int

        Returns
        -------
        dict or tuple
            Sample dictionary, optionally with metadata.
        """
        year = float(self.model_indexes["year"][ind])
        lead_time = float(self.model_indexes["lead_time"][ind])
        selection = dict(year=year, lead_time=lead_time)

        condition = self._index_condition_dataset(ind)
        target = self._index_observation_dataset(ind)
        input = self._index_model_dataset(ind)

        if self._autoencoding_model_data:
            target = input

        if self._write_condition_to_input:
            input = condition

        elif self._concat_condition_to_input:
            input = xr.concat([input, condition], dim="channels")

        time_features = _get_time_features(self.config, year, lead_time, input)

        datadict = dict(
            input=torch.as_tensor(input.to_numpy(), dtype=torch.float32),
            target=torch.as_tensor(target.to_numpy(), dtype=torch.float32),
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
