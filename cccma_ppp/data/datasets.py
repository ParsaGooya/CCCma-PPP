import numpy as np
import xarray as xr
from torch.utils.data import Dataset
import torch
import dataclasses
from pathlib import Path
import pydantic
import warnings
import gc
from typing import Literal, ClassVar

from cccma_ppp.data.data_abc import (
    DatasetConfigABC,
    DatasetOperatorABC,
    DataConfigABC,
)
from cccma_ppp.data.utils_data import (
    ModelDataConfig,
    ObsDataConfig,
    ConditionDataConfig,
    WeightsConfig,
    _unwrap_data_variables,
    _load_xarray_data,
    _create_train_mask,
)

from cccma_ppp.preprocessing.preprocessing_ABC import PreprocessModuleABC
from cccma_ppp.preprocessing.preprocessing import PreprocessingPipeline
from cccma_ppp.generic.runtime import RuntimeContext


@dataclasses.dataclass
class TrainDatasetConfig(DatasetConfigABC):
    model: ModelDataConfig
    observation: ObsDataConfig | None = None
    condition: ConditionDataConfig | None = None
    condition_method: str = None
    time_features: list[str] | None = None
    num_lead_months: int | None = None

    _VALID_CONDITION_METHODS: ClassVar[frozenset[str]] = frozenset({'ensemble_mean' , 'cross_ensemble' , 'same_member',  'static'})
    _VALID_TIME_FEATURES: ClassVar[frozenset[str]] = frozenset({'year', 'lead_time', 'month_sin', 'month_cos'}) 

    def __post_init__(self):
        self._fitted_preprocessors: bool = False
        self._effective_condition: ConditionDataConfig | ModelDataConfig | None = None

        self._check_condition_method()
        self._check_time_features()
        self._check_model()
        self._check_observation()
        self._check_condition()
        self._resolve_condition()

    def _check_condition_method(self):
        if self.condition_method is not None:
  
            if self.condition_method not in self._VALID_CONDITION_METHODS:
                raise ValueError(
                    f'Invalid condition_method: {self.condition_method}. '
                    f'Must be a in {sorted(self._VALID_CONDITION_METHODS)}.'
            )
        return self
    
    def _check_time_features(self):
        if self.time_features is not None:
  
            invalid = set(self.time_features) - self._VALID_TIME_FEATURES
            if invalid:
                raise ValueError(
                    f'Invalid time features: {sorted(invalid)}. '
                    f'Must be a subset of {sorted(self._VALID_TIME_FEATURES)}.'
            )
        return self

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

    def _resolve_condition(self):
        """Resolve the conditioning dataset, falling back to model data if no condition is provided but condition_method is."""
        if self.condition is not None:
            self._effective_condition = self.condition
        elif self._using_model_data_as_condition:
            self._effective_condition = self._model_as_condition()
        else:
            self._effective_condition = None

        return self

    @property
    def num_model_lead_months(self) -> int:
        return self.model.info.sizes["lead_time"]
    
    @property
    def _using_model_data_as_condition(self) -> bool:
        '''
        Check if the model data is going to be used as a condition.If so, we can avoid loading both model and condition data
        if not necessary. This should be true if:

           1- No condition data is provided but condition_method is (subject to condition_method not being "static")

           2- The provided condition data points to the same files and variables and ensemble members.
        
        '''
        if self.condition is None:
            return self.condition_method in {'ensemble_mean', 'cross_ensemble', 'same_member'}
        return (
            self.condition.paths == self.model.paths
            and self.condition.names == self.model.names
            and self.condition.ensemble_list == self.model.ensemble_list
        )

    @property
    def effective_condition(self) -> ConditionDataConfig | ModelDataConfig | None:
        return self._effective_condition
    
    def _model_as_condition(self) -> ModelDataConfig:
        ensemble_mean = self.condition_method == 'ensemble_mean'
        return  ModelDataConfig(
                paths=self.model.paths,
                names=self.model.names,
                preprocessing_pipeline=self.model.preprocessing_pipeline,
                ensemble_list=self.model.ensemble_list,
                concat_dim=self.model.concat_dim,
                file_type=self.model.file_type,
                ensemble_mean=ensemble_mean,
                rename_dict=self.model.rename_dict,
            )

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

    def build_operator(self):
        return TrainDatasetOperator(self)



class TrainDatasetOperator(DatasetOperatorABC): 

    def __init__(self, config: TrainDatasetConfig):
        self.config = config

    def _fit_preprocessors(
        self,
        train_years: np.ndarray | list | tuple,
        save=False,
        save_path: Path | str | None = None,
        save_name: str | None = None,
    ):
        
        def _fit_processor(
            dataconfig: ModelDataConfig | ObsDataConfig | ConditionDataConfig,
            selection: dict,
            mask: bool = False,
            save_path: Path | str | None = None,
            save_name: str | None = None,
        ):
            
            _base = _load_xarray_data(
                dataconfig.list_paths,
                names=dataconfig.names,
                concat_dim=dataconfig.concat_dim,
                selection=selection,
                ensemble_mean=dataconfig.ensemble_mean,
                rename_dict=dataconfig.rename_dict,
            )
            
            _mask = _create_train_mask(_base.year, _base.lead_time) if mask else None

            dataconfig.preprocessing_pipeline.fit(
                base_data=_base.load(),
                mask=_mask,
                save=save,
                save_path=save_path,
                save_name=save_name,
            )

            _base.close()
            del _base, _mask
            gc.collect()

        if self.config.model.preprocessing_pipeline.load_dir is None:
            selection = {
                "year": train_years,
                "lead_time": np.arange(1, self.config.num_lead_months + 1),
            }
            if self.config.model.info.coords["ensembles"] is not None:
                selection["ensembles"] = self.config.model.info.coords["ensembles"]

            _fit_processor(self.config.model, 
                              selection = selection, 
                              mask = True, 
                              save_path = save_path, 
                              save_name = save_name)
 
        else:
            self.config.model.preprocessing_pipeline._load_from_memory(
                Path(self.config.model.preprocessing_pipeline.load_dir),
                load_name=self.config.model.preprocessing_pipeline.load_name,
            )

        if self.config.observation is not None:
            if self.config.observation.preprocessing_pipeline.load_dir is None:
                selection = {"year": train_years}
                if self.config.observation.info.coords["ensembles"] is not None:
                    selection["ensembles"] = self.config.observation.info.coords["ensembles"]

                _fit_processor(self.config.observation, 
                    selection = selection, 
                    save_path = save_path, 
                    save_name = save_name)

            else:
                self.config.observation.preprocessing_pipeline._load_from_memory(
                    Path(self.config.observation.preprocessing_pipeline.load_dir),
                    load_name=self.config.observation.preprocessing_pipeline.load_name,
                )

        if self.config.effective_condition is not None:
            if self.config.effective_condition.preprocessing_pipeline.load_dir is None:
                if self.config.condition_method == 'static':
                    selection = {}
                else:
                    selection = {
                        "year": train_years,
                        "lead_time": np.arange(1, self.config.num_lead_months + 1),
                    }
                    if self.config.effective_condition.info.coords["ensembles"] is not None:
                        selection["ensembles"] = self.config.effective_condition.info.coords["ensembles"]
                
                _fit_processor(self.config.effective_condition, 
                    selection = selection, 
                    mask = True,
                    save_path = save_path, 
                    save_name = save_name)

            else:
                self.config.effective_condition.preprocessing_pipeline._load_from_memory(
                    Path(self.config.effective_condition.preprocessing_pipeline.load_dir),
                    load_name=self.config.effective_condition.preprocessing_pipeline.load_name,
                )

        self.config._fitted_preprocessors = True

    def _load_fitted_preprocessors(
        self, load_dir: Path | str | None = None
    ):
        
        def _load_preprocessor(
            pipeline : PreprocessingPipeline,
            load_dir: Path | str | None = None):

                if load_dir is None:
                    load_dir = (Path(RuntimeContext.GLOBAL_EXP_DIR)
                            / "preprocessing_pipeline"
                            / f"{pipeline.name}_preprocessing_pipeline.joblib"
                        )
                else:
                    load_dir = Path(load_dir)

                pipeline._load_from_memory(
                    Path(load_dir),
                )

                if not pipeline.fitted:
                    raise RuntimeError(
                        f"the loaded preprocessor for {pipeline.name} is not fitted!"
                    )
                
        _load_preprocessor(self.config.model.preprocessing_pipeline, load_dir)

        if self.config.observation is not None:

            _load_preprocessor(self.config.observation.preprocessing_pipeline, load_dir)

        if self.config.effective_condition is not None:

            _load_preprocessor(self.config.effective_condition.preprocessing_pipeline, load_dir)

        self.config._fitted_preprocessors = True

    def _add_fitted_preprocessor(self, preprocessor: PreprocessModuleABC, index=0):
        if not isinstance(preprocessor, PreprocessModuleABC):
            raise TypeError(
                f"preprocessor must be an instance of ProcessorConfig, "
                f"got {type(preprocessor)}"
            )
        assert preprocessor.fitted, "The preprocessor must be fitted"

        self.config.model.preprocessing_pipeline.add_fitted_preprocessor(
            preprocessor, index=index
        )
        if self.config.observation is not None:
            self.config.observation.preprocessing_pipeline.add_fitted_preprocessor(
                preprocessor, index=index
            )
        if self.config.effective_condition is not None:
            self.config.effective_condition.preprocessing_pipeline.add_fitted_preprocessor(
                preprocessor, index=index
            )

    def get_weights(
        self,
        config: WeightsConfig | None = None,
        save=True,
        save_path: Path | str | None = None,
        save_name: str | None = None,
    ):
        if config is None:
            config = WeightsConfig()

        if self.config.observation is not None:
            target_coords = self.config.observation.info.coords.copy()
        else:
            target_coords = self.config.model.info.coords.copy()

        if "ensembles" in target_coords:
            del target_coords["ensembles"]

        from cccma_ppp.preprocessing.utils_preprocessing import Oceannanremove

        if self.config.observation is not None:
            pipeline = self.config.observation.preprocessing_pipeline
        else:
            pipeline = self.config.model.preprocessing_pipeline

        checklist = [
            isinstance(item, Oceannanremove) for item in pipeline.fitted_preprocessors
        ]

        weights = config.build_weights(
            target_coords,
            oceannanremover=pipeline.get_preprocessors("oceannanremover")
            if any(checklist)
            else None,
            save=save,
            save_path=save_path,
            save_name=save_name,
        )

        if "channels" in weights.dims:
            if self.config.observation is not None:
                error_msg = f"inconsistent variable weights {weights.channels.values} for taget variables {self.config.observation.names}"
                if not weights.channels.values == self.config.observation.names:
                    raise RuntimeError(error_msg)
            else:
                error_msg = f"inconsistent variable weights {weights.channels.values} for taget variables {self.config.model.names}"
                if not weights.channels.values == self.config.model.names:
                    raise RuntimeError(error_msg)

        return weights

    def get_input_var_metadata(self):

        metadata = dict(variables=list(), preprocessors=list())

        if self.config.effective_condition is None:
            metadata = self._update_metadata_with_dataconfig_metadata(metadata, self.config.model)

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

        return metadata

    def get_target_var_metadata(self):

        metadata = dict(variables=list(), preprocessors=list())

        if self.config.observation is None:
            metadata = self._update_metadata_with_dataconfig_metadata(metadata, self.config.model)
        else:
            metadata = self._update_metadata_with_dataconfig_metadata(
                metadata, self.config.observation
            )

        return metadata

    def _update_metadata_with_dataconfig_metadata(
            self, metadata: dict, dataconfig: DataConfigABC
        ):
            preprocessor_names = [
                processor[0] for processor in dataconfig.preprocessing_pipeline.pipeline
            ]
            for var in dataconfig.names:
                metadata["variables"].append(var)
                metadata["preprocessors"].append(preprocessor_names)
                return metadata

    def build_dataset(
        self,
        years: np.ndarray,
        mask: xr.DataArray = None,
        return_metadata: bool = False,
    ):
        return TrainDataset(
            config=self.config,
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
                "Make sure to fit preprocessors first!. Hint:  TrainDatasetConfig._fit_preprocessors()"
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

        from cccma_ppp.preprocessing.utils_preprocessing import Oceannanremove

        checklist = [
            isinstance(item, Oceannanremove)
            for item in self.config.model.preprocessing_pipeline.fitted_preprocessors
        ]

        if any(checklist):
            return self.config.model.preprocessing_pipeline.get_preprocessors(
                "oceannanremover"
            ).final_locations.shape * len(self.config.model.names)
        else:
            return (
                self.config.model.info.coords["lat"].size,
                self.config.model.info.coords["lon"].size,
            )

    def get_target_shape(self):

        from cccma_ppp.preprocessing.utils_preprocessing import Oceannanremove

        if self.observation_dataset is not None:
            checklist = [
                isinstance(item, Oceannanremove)
                for item in self.config.observation.preprocessing_pipeline.fitted_preprocessors
            ]

            if any(checklist):
                return self.config.observation.preprocessing_pipeline.get_preprocessors(
                    "oceannanremover"
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
