import numpy as np
import xarray as xr
from torch.utils.data import Dataset
import torch
import dataclasses
import warnings
from pathlib import Path

from cccma_ppp.data_modules.dataset import DatasetConfigABC, DatasetOperator
from cccma_ppp.data_modules.data import (
    DataConfigABC,
    ModelDataConfig,
    ObsDataConfig,
    ConditionDataConfig)

from cccma_ppp.data_modules import (
    _unwrap_data_variables,
    _load_xarray_data,
    _create_train_mask,
)

from cccma_ppp.preprocessing.preprocessing_ABC import PreprocessModuleABC


@dataclasses.dataclass
class TrainDatasetConfig(DatasetConfigABC):
    model: ModelDataConfig
    observation: ObsDataConfig | None = None
    condition: ConditionDataConfig | None = None
    condition_method: str = None
    time_features: list[str] | None = None
    num_lead_months: int | None = None

    def __post_init__(self):
        self._fitted_preprocessors: bool = False
        self._effective_condition: ConditionDataConfig | ModelDataConfig | None = None

        super().__init__()

        self._check_model()
        self._check_observation()

        self._resolve_condition()
        self._check_condition()

    def _check_model(self):

        if self.condition_method == "same_member":
            if self.model.ensemble_mean:
                raise ValueError(
                "for same member coniditioning the model data should not be ensemble mean."
            )
        
        if self.num_lead_months is None:
            self.num_lead_months = self.num_model_lead_months
        if not self.num_lead_months <= self.num_model_lead_months:
            raise ValueError(
            f"Maximum available lead months is {self.num_model_lead_months}"
        )
            
        return self
    
    def _check_observation(self):
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
                if not self.effective_condition.ensemble_mean is True:
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

        else: ##comeback
            if self.condition_method == "static":

                raise ValueError(
                "For static conditioning method condition dataset must be specified!"
            )
            
        return self

    @property
    def ds_operator(self):
        return DatasetOperator(self)

    @property
    def num_model_lead_months(self) -> int:
        return self.model.info.sizes["lead_time"]
    
    @property
    def get_common_time(self):
        if self.observation is not None:
            return np.intersect1d(self.model.year_range, self.observation.year_range)
        else:
            return self.model.year_range

    @property
    def available_train_time(self):
        num_lead_years = self.num_lead_months // 12
        if self.observation is None:
            return np.arange(
                np.min(self.get_common_time),
                np.max(self.get_common_time) + 1 - num_lead_years + 1,
            )
        else:
            return self.get_common_time

    def _fit_preprocessors(
        self,
        train_years: np.ndarray | list | tuple,
        save=False,
        save_path: Path | str | None = None,
        save_name: str | None = None,
    ):
        self.ds_operator._fit_preprocessors(train_years = train_years,
                                            save = save,
                                            save_path = save_path,
                                            save_name = save_name,
        )

    def _load_fitted_preprocessors(
        self, load_dir: Path | str
    ):
        self.ds_operator._load_fitted_preprocessors(load_dir)

    def _add_fitted_preprocessor(self, preprocessor: PreprocessModuleABC, index=0):

        self.ds_operator._add_fitted_preprocessor(preprocessor, index)

    def build_dataset(
        self,
        years: np.ndarray,
        mask: xr.DataArray = None,
        return_metadata: bool = False,
    ):
        return TrainDataset(
            config=self,
            requested_years=years,
            mask=mask,
            return_metadata=return_metadata,

        )
    

@dataclasses.dataclass
class TrainDataset(Dataset):
    config: TrainDatasetConfig
    requested_years: list[int] | tuple[int] | np.ndarray
    mask: xr.DataArray = None
    return_metadata: bool = False

    def __post_init__(self):
        if not self.config._fitted_preprocessors:
            raise RuntimeError(
                "Make sure to fit preprocessors first!. Hint:  DatasetOperators._fit_preprocessors()"
            )
        if not set(self.requested_years).issubset(set(self.config.get_common_time)):
            raise ValueError(
                "the requested years are not common to input and target data."
            )

        self.time_features = self.config.time_features
        self.observation_dataset = self.condition_dataset = None

        self.model_dataset = self._load_xarray_data(self.config.model)

        if self.config.observation is not None:
            self.observation_dataset = self._load_xarray_data(self.config.observation)

        if self.config.effective_condition is not None:
            self.condition_dataset = self._load_xarray_data(self.config.effective_condition)

        self.mask = self._prepare_mask()
        self.model_indexes = self.get_model_indexes()
        self.obs_indexes = self.get_obs_indexes(self.model_indexes)
        self.cond_indexes = self.get_cond_indexes(self.model_indexes)

    @property
    def _autoencoding_model_data(self):
        return self.config.observation is None

    @property
    def _load_model(self):
        ''' 
        Check if model data needs to be loaded. Is true if:

            1- A different codition data than model is provided, which means
            _using_model_data_as_condition is False, regardless of obs.
        
            1- No obervation is provided, hence, we are autoencoding 
            the model data (condition_method is already asserted in config),
            regardless of if stand alone condition is provided.

        '''
        return any( [
                self._autoencoding_model_data,
                not self.config._using_model_data_as_condition])
    
    @property
    def _write_condition_to_input(self):
        '''
        Check if we need to use the condition data as the only input of the ML model.
        Will be true if:

            1- No stand alone condition is provided but condition_method is. Hence, _using_model_data_as_condition
            is true and condition is read from model. Model will not be loaded if not necessary.

            2- If stand alone condition is provided, but we are autoencoding model
             as no obervation is available. Model will be loaded too.
        
        '''
        if self.config._using_model_data_as_condition:
                return True    
        else:   
            if self._autoencoding_model_data:
                return True

        return False    
    
    @property
    def _concat_condition_to_input(self):
        '''
        Check if all stand alone condition, model and observation exist.
        If so, _write_condition_to_input is False and effective_condition exists.
        '''

        return (self._write_condition_to_input is False and
                self.config.effective_condition is not None)


    def _prepare_mask(self):
        mask = self.mask
        if mask is None:
            mask = _create_train_mask(
                years=self.config.model.year_range,
                lead_times=np.arange(1, self.config.model.info.sizes["lead_time"] + 1),
            )
            mask = xr.full_like(mask, fill_value=False)

        mask = mask.sel(year=self.requested_years).sel(
            lead_time=np.arange(1, self.config.num_lead_months + 1)
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

    def _load_xarray_data(self, config: DataConfigABC):

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

    def get_obs_indexes(self, model_indexes: dict):

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

    def get_cond_indexes(self, model_indexes: dict):

        if self.condition_dataset is not None:
            if self.config.condition_method != "static":
                indexes = {}
                indexes["year"] = model_indexes["year"]
                indexes["lead_time"] = model_indexes["lead_time"]
                if self.config.condition_method == "cross_ensemble":
                    ens_inds = [
                        np.random.choice(self.config.effective_condition.info.coords["ensembles"])
                        for _ in range(len(model_indexes["year"]))
                    ]
                    indexes["ensembles"] = np.array(ens_inds)

                elif self.config.condition_method == "same_member":
                    indexes["ensembles"] = model_indexes["ensembles"]

            return indexes

    def get_input_shape(self):

        from cccma_ppp.preprocessing.utils_preprocessing import Flattennanremove

        checklist = [
            isinstance(item, Flattennanremove)
            for item in self.config.model.preprocessing_pipeline.fitted_preprocessors
        ]

        if any(checklist):
            return self.config.model.preprocessing_pipeline.get_preprocessors(
                "flattener"
            ).final_locations.shape * len(self.config.model.names)
        else:
            return (
                self.config.model.info.coords["lat"].size,
                self.config.model.info.coords["lon"].size,
            )

    def get_target_shape(self):

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

        return len(self.time_features)

    def _index_condition_dataset(self, ind):

        if self.condition_dataset is not None:
            if self.config.condition_method != "static":
                year = float(self.cond_indexes["year"][ind])
                lead_time = float(self.cond_indexes["lead_time"][ind])
                selection = dict(year=year, lead_time=lead_time)
                if "ensembles" in self.cond_indexes:
                    selection["ensembles"] = self.cond_indexes["ensembles"][ind]

            else:
                selection = {}

            condition = self.config.effective_condition.preprocessing_pipeline.transform(
                self.condition_dataset.sel(**selection)
            )
            condition = _unwrap_data_variables(condition)  

            return condition    

    def _index_observation_dataset(self, ind):

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

    def _get_time_features(self, year, lead_time, input : xr.DataArray):

        if self.time_features is not None:
            time_features_list = np.array([self.time_features]).flatten()
            feature_indices = {
                "year": 0,
                "lead_time": 1,
                "month_sin": 2,
                "month_cos": 3,
            }

            target_time = year + lead_time // 12
            target_month = lead_time

            y = (target_time - np.min(self.config.get_common_time)) / (
                np.max(self.config.get_common_time)
                - np.min(self.config.get_common_time)
            )
            lt = lead_time / self.config.num_lead_months
            msin = np.sin(2 * np.pi * target_month / 12.0)
            mcos = np.cos(2 * np.pi * target_month / 12.0)

            time_features = np.stack([y, lt, msin, mcos])
            time_features = time_features[
                ..., [feature_indices[k] for k in time_features_list]
            ]

            if input.ndim > 2:
                time_features = np.broadcast_to(
                    time_features[(...,) + (None,) * (input.ndim - 1)],
                    (time_features.shape[0],) + input.shape[1:],
                )

            return time_features


    def __getitem__(self, ind):
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

        time_features = self._get_time_features(year, lead_time, input)
 
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
        return len(self.model_indexes.get(list(self.model_indexes.keys())[0]))
