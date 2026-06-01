import numpy as np
import xarray as xr
from torch.utils.data import Dataset
import torch
import dataclasses
from pathlib import Path
import os
import warnings

from cccma_ppp.data.data_abc import XarrayDatasetABC, XarrayDatasetConfigABC, DataConfig
from cccma_ppp.data.utils_data import ModelDataConfig, ObsDataConfig, ConditionDataConfig, WeightsConfig, _unwrape_data_variables, _load_xarray_data, infoclass, _create_train_mask

from cccma_ppp.preprocessing.preprocessing_ABC import PreprocessModuleABC



        
@dataclasses.dataclass
class XArrayDatasetConfig(XarrayDatasetConfigABC):

    model: ModelDataConfig
    observation: ObsDataConfig | None = None 
    condition : ConditionDataConfig | None = None 
    condition_method: str = None
    time_features : list[str] | None = None 
    num_lead_months  : int | None = None 
    
    def __post_init__(self):
        self._fitted_preprocessors = False
        self._using_model_data_as_condition = False
 
        self.num_model_lead_months = self.model.info.sizes['lead_time']

        if self.num_lead_months is None:
            self.num_lead_months = self.num_model_lead_months
        assert self.num_lead_months <= self.num_model_lead_months, f'Maximum available lead months is {self.num_model_lead_months}' 

        if self.time_features is not None:
            assert set(self.time_features).issubset(set(['year','lead_time','month_sin','month_cos']))

        self.model.preprocessing_pipeline.name = 'model'
        if self.condition_method == 'same_member':
            assert self.model.ensemble_mean is not None, 'for same member coniditioning the model data should not be ensemble mean.'
       
        if self.observation is not None:

            if not self.observation.info.coords["lat"].equals(self.model.info.coords["lat"]):
                warnings.warn(f'model and observation data do not have the same latitudes cooridnates.') 
            if not self.observation.info.coords["lon"].equals(self.model.info.coords["lon"]):
                warnings.warn(f'model and observation data do not have the same longitudes cooridnates.') 
            self.observation.preprocessing_pipeline.name = 'observation'
        else:
            assert self.condition_method is not None, f'No target observation is specifiec. Specify condition_method!' 

        if self.condition_method is not None:
            assert self.condition_method in  self._available_condiiton_methods(), f'condition_method must be from {self._available_condiiton_methods()} '

        if self.condition is not None:
            assert self.condition_method is not None, f'specify condition_method for conditioning dataset!'

            if self.condition.paths == self.model.paths:
                if self.condition.names == self.model.names:
                    self._using_model_data_as_condition = True       

            if self.condition_method in ['cross_ensemble', 'same_member']:
                assert self.condition.ensemble_mean is False, 'Ensemble mean cannot be True for cross_ensemble or same_member conditioning.'
                assert self.condition.info.coords['ensembles'] is not None, 'For cross_ensemble or same_member conditioning an ensembles dim must exist in condition.'
            elif  self.condition_method == 'ensemble_mean':
                assert self.condition.ensemble_mean is True, 'Ensemble mean must be True for ensemble_mean conditioning.'
            else:
                assert self.condition.ensemble_list is None, 'For "static" or "no_ensemble" conditioning fields cannot specify ensemble list.'
 

        elif self.condition_method is not None:
            assert self.condition_method in ['ensemble_mean', 'cross_ensemble', 'same_member'], 'for static and no-ensemble conditioning methods condition dataset must be specified!'
            
            if self.condition_method in ['cross_ensemble', 'same_member']:
                ensemble_mean = False
            elif  self.condition_method == 'ensemble_mean':
                ensemble_mean = True

            self._using_model_data_as_condition = True

            self.condition = ModelDataConfig(paths = self.model.paths, 
                                           names = self.model.names, 
                                           preprocessing_pipeline = self.model.preprocessing_pipeline, 
                                            ensemble_list = self.model.ensemble_list,
                                            concat_dim = self.model.concat_dim,
                                            file_type = self.model.file_type,
                                            ensemble_mean  = ensemble_mean,
                                            rename_dict = self.model.rename_dict)
            
        if self.condition is not None:

            if self.condition_method not in ['static']:
                    assert self.condition.info.sizes == self.model.info.sizes, 'non static coniditioning field must have the same times as the model data.'
            
            assert self.condition.info.coords['lat'].equals(self.model.info.coords['lat']), 'coniditioning field must have the same lat shape as the model data.'
            assert self.condition.info.coords['lon'].equals(self.model.info.coords['lon']), 'coniditioning field must have the same lon shape as the model data.'
            
            if self.condition_method in ['same_member']:
                    assert self.condition.info.coords['ensembles'].equals(self.model.info.coords['ensembles']), 'coniditioning field and model data must have the same ensemble dimension for same_member conditioning.'
        
            self.condition.preprocessing_pipeline.name = 'condition'
        
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
            return np.arange( np.min(self.get_common_time), np.max(self.get_common_time) + 1 - num_lead_years + 1) 
        else:
            return self.get_common_time
        
    
    def _fit_preprocessors(self, train_years : np.ndarray | list | tuple, save = False, save_path : Path | str | None = None, save_name : str| None = None):
 
        if self.model.preprocessing_pipeline.load_dir is None:

            selection={'year' : train_years, 'lead_time' :  np.arange(1,self.num_lead_months + 1) }
            if self.model.info.coords['ensembles'] is not None: 
                selection['ensembles'] = self.model.info.coords['ensembles']
            _base = _load_xarray_data(self.model.list_paths, names = self.model.names, concat_dim = self.model.concat_dim, selection=selection, ensemble_mean=self.model.ensemble_mean, rename_dict=self.model.rename_dict)
            _mask = _create_train_mask(_base.year, _base.lead_time)
            self.model.preprocessing_pipeline.fit(base_data = _base.load(), mask = _mask, save = save, save_path = save_path ,save_name = save_name)
            _base.close()
            del _base, _mask
        else:
            self.model.preprocessing_pipeline._load_from_memory(Path(self.model.preprocessing_pipeline.load_dir), load_name = self.model.preprocessing_pipeline.load_name)
        
        if self.observation is not None:
            if self.observation.preprocessing_pipeline.load_dir is None:
                selection={'year' : train_years }
                if self.observation.info.coords['ensembles'] is not None: 
                    selection['ensembles'] = self.observation.info.coords['ensembles']
                _base = _load_xarray_data(self.observation.list_paths,  names = self.observation.names ,concat_dim = self.observation.concat_dim, selection = selection , ensemble_mean=self.observation.ensemble_mean, rename_dict=self.observation.rename_dict)
                self.observation.preprocessing_pipeline.fit(base_data = _base.load(), save = save, save_path = save_path, save_name = save_name)
                _base.close()
                del _base
            else:
                self.observation.preprocessing_pipeline._load_from_memory(Path(self.observation.preprocessing_pipeline.load_dir), load_name = self.observation.preprocessing_pipeline.load_name)  

        if self.condition is not None:
            if self.condition.preprocessing_pipeline.load_dir is None:
                selection={'year' : train_years }
                if self.condition.info.coords['ensembles'] is not None:
                    selection['ensembles'] = self.condition.info.coords['ensembles']
                _base = _load_xarray_data(self.condition.list_paths, names = self.condition.names,  concat_dim = self.condition.concat_dim, selection= selection, ensemble_mean=self.condition.ensemble_mean, rename_dict=self.condition.rename_dict)
                _mask = _create_train_mask(_base.year, _base.lead_time)
                self.condition.preprocessing_pipeline.fit(base_data =  _base.load(), mask = _mask, save = save, save_path = save_path, save_name = save_name)
                _base.close()
                del _base, _mask
            else:
                self.condition.preprocessing_pipeline._load_from_memory(Path(self.condition.preprocessing_pipeline.load_dir ), load_name = self.condition.preprocessing_pipeline.load_name)   

        self._fitted_preprocessors = True
    def _load_fitted_preprocessors(self, load_dir : Path | str | None = None, load_name : str| None  = None):
        

        if load_dir is None:
            load_dir = Path(os.environ["GLOBAL_EXP_DIR"]) / 'preprocessing_pipeline' / f"{self.model.preprocessing_pipeline.name}_preprocessing_pipeline.joblib"
        else:
            load_dir = Path(load_dir)

        assert self.model.preprocessing_pipeline.fitted, 'the loaded preprocessor for model data is not fitted!'
        self.model.preprocessing_pipeline._load_from_memory(Path(load_dir), load_name = load_name)
        

        if self.observation is not None:
            if load_dir is None:
                load_dir = Path(os.environ["GLOBAL_EXP_DIR"]) / 'preprocessing_pipeline' /  f"{self.observation.preprocessing_pipeline.name}_preprocessing_pipeline.joblib"
            else:
                load_dir = Path(load_dir)

            assert self.observation.preprocessing_pipeline.fitted, 'the loaded preprocessor for observation data is not fitted!'
            self.observation.preprocessing_pipeline._load_from_memory(Path(load_dir), load_name = load_name)     
            

        if self.condition is not None:
            if self.condition.preprocessing_pipeline.load_dir is None:
                load_dir = Path(os.environ["GLOBAL_EXP_DIR"]) / 'preprocessing_pipeline' / f"{self.condition.preprocessing_pipeline.name}_preprocessing_pipeline.joblib"
            else:
                load_dir = Path(load_dir)
            
            assert self.condition.preprocessing_pipeline.fitted, 'the loaded preprocessor for condition is not fitted!' 
            self.condition.preprocessing_pipeline._load_from_memory(Path(load_dir), load_name = load_name)          
               
        self._fitted_preprocessors = True   

    def _add_fitted_preprocessor(self, preprocessor: PreprocessModuleABC , index = 0):
        if not isinstance(preprocessor, PreprocessModuleABC):
            raise TypeError(
                f"preprocessor must be an instance of ProcessorConfig, "
                f"got {type(preprocessor)}"
        )
        assert preprocessor.fitted, 'The preprocessor must be fitted'

        self.model.preprocessing_pipeline.add_fitted_preprocessor(preprocessor, index = index)
        if self.observation is not None:
            self.observation.preprocessing_pipeline.add_fitted_preprocessor(preprocessor, index = index)
        if self.condition is not None:
            self.condition.preprocessing_pipeline.add_fitted_preprocessor(preprocessor, index = index)

    def get_weights(self, config: WeightsConfig | None= None, save = True, save_path : Path | str | None = None, save_name : str| None = None):
        if config is None:
            config =  WeightsConfig()

        if self.observation is not None:
            target_coords = self.observation.info.coords.copy()
        else:
            target_coords = self.model.info.coords.copy()

        if 'ensembles' in target_coords:
            del target_coords['ensembles']

        from cccma_ppp.preprocessing.utils_preprocessing import Oceannanremove
        if self.observation is not None:
            pipeline = self.observation.preprocessing_pipeline
        else:
            pipeline = self.model.preprocessing_pipeline

        checklist = [isinstance(item, Oceannanremove) for item in pipeline.fitted_preprocessors]

        weights =  config.build_weights(target_coords,
                                        oceannanremover=pipeline.get_preprocessors('oceannanremover') if any(checklist) else None,
                                        save=save,
                                        save_path=save_path,
                                        save_name=save_name)

        if 'channels' in weights.dims:
            
            if self.observation is not None:
                error_msg = f'inconsistent variable weights {weights.channels.values} for taget variables {self.observation.names}'
                assert weights.channels.values == self.observation.names, error_msg
            else:
                error_msg = f'inconsistent variable weights {weights.channels.values} for taget variables {self.model.names}'
                assert weights.channels.values == self.model.names, error_msg
        
        return weights


    def build(self, years : np.ndarray, mask: xr. DataArray = None, return_metadata : bool = False):
        return  XArrayDataset(config = self, requested_years = years,mask =  mask, return_metadata = return_metadata)
        
    @classmethod
    def _available_condiiton_methods(cls):
        return ['ensemble_mean' , 'cross_ensemble' , 'same_member',  'static']

        

@dataclasses.dataclass
class XArrayDataset(Dataset, XarrayDatasetABC):

    
    config:XArrayDatasetConfig
    requested_years : list[int]  | tuple[int]  | np.ndarray
    mask : xr.DataArray = None
    return_metadata : bool = False
    
    def __post_init__(self):
        assert self.config._fitted_preprocessors, 'Make sure to fit preprocessors first!. Hint:  XArrayDatasetConfig._fit_preprocessors()'
        assert set(self.requested_years).issubset(set(self.config.get_common_time)), 'the requested years are not common to input and target data.'

        self._autoencoding_input = False
        self.observation_dataset = self.condition_dataset = None
        
        self.model_dataset = self._load_xarray_data(self.config.model)
        
        if self.config.observation is not None:
            self.observation_dataset = self._load_xarray_data(self.config.observation)
        else:
            self._autoencoding_input = True

        if self.config.condition is not None:
            self.condition_dataset = self._load_xarray_data(self.config.condition)          


        self._prepare_mask()

        self.mask = self.mask.where(~self.mask)
        self.model_indexes = self.get_model_indexes()
        self.obs_indexes = self.get_obs_indexes(self.model_indexes)
        self.cond_indexes = self.get_cond_indexes(self.model_indexes)

        self.time_features = self.config.time_features

    def _prepare_mask(self):

        if self.mask is None:
            self.mask = _create_train_mask(years = self.config.model.year_range, 
                                           lead_times= np.arange(1, self.config.model.info.sizes['lead_time']+ 1) )
            self.mask = xr.full_like(self.mask, fill_value = False)
                                                                          
        self.mask = self.mask.sel(year = self.requested_years).sel(lead_time = np.arange(1,self.config.num_lead_months + 1))
        if all([not self.config.model.ensemble_mean, self.config.model.info.coords['ensembles'] is not None]):
            self.mask = self.mask.expand_dims(ensembles = len(self.config.model.info.coords['ensembles']), axis = 0)
            self.mask = self.mask.assign_coords(ensembles = self.config.model.info.coords['ensembles'] )

        return self
       
    def _load_xarray_data(self, config : DataConfig):

        return _load_xarray_data(config.list_paths, 
                                               names = config.names, 
                                               ensemble_mean= config.ensemble_mean, 
                                               selection= {'ensembles' :  config.info.coords['ensembles']} if config.info.coords['ensembles'] is not None else None,
                                            #    preprocessor = self.model.preprocessing_pipeline,  ##checking if doing this once is faster
                                               concat_dim = config.concat_dim,
                                               rename_dict= config.rename_dict)

    def get_model_indexes(self):

        mask = self.mask.stack(batch = dict(self.mask.sizes).keys()).transpose('batch', ...).dropna(dim = 'batch').batch.values
        mask = tuple(map(np.array, zip(*mask)))
        indexes =  { key : mask[ind] for ind, key in enumerate(tuple(dict(self.mask.sizes).keys()))}

        return indexes   
    
    def get_obs_indexes(self, model_indexes : dict):

        if self.observation_dataset is not None:

            indexes = {}
            offset_year, month = np.divmod(model_indexes['lead_time'] - 0.5 ,12)
            indexes['year'] = model_indexes['year'] + offset_year
            indexes['month'] = month + 0.5
            if all([not self.config.observation.ensemble_mean, self.config.observation.info.coords['ensembles'] is not None]):
                ens_inds = [np.random.choice(self.config.observation.info.coords['ensembles']) for _ in range(len(model_indexes['year'])) ]    
                indexes['ensembles'] = np.array(ens_inds)

            return indexes

    def get_cond_indexes(self, model_indexes : dict):

        if self.condition_dataset is not None:
            if self.config.condition_method != 'static':
                indexes = {}
                indexes['year'] = model_indexes['year']
                indexes['lead_time'] = model_indexes['lead_time']
                if self.config.condition_method == 'cross_ensemble':
                    ens_inds = [np.random.choice(self.config.condition.info.coords['ensembles']) for _ in range(len(model_indexes['year'])) ]    
                    indexes['ensembles'] = np.array(ens_inds)

                elif self.config.condition_method == 'same_member':
                    indexes['ensembles'] = model_indexes['ensembles']
                    
            return indexes


    def get_input_shape(self):

        from cccma_ppp.preprocessing.utils_preprocessing import Oceannanremove
        checklist = [isinstance(item, Oceannanremove) for item in self.config.model.preprocessing_pipeline.fitted_preprocessors]

        if any(checklist):
            return self.config.model.preprocessing_pipeline.get_preprocessors('oceannanremover').final_locations.shape * len(self.config.model.names)
        else:
            return (self.config.model.info.coords['lat'].size, self.config.model.info.coords['lon'].size)
        
    def get_target_shape(self):

        from cccma_ppp.preprocessing.utils_preprocessing import Oceannanremove
        if self.observation_dataset is not None:
            checklist = [isinstance(item, Oceannanremove) for item in self.config.observation.preprocessing_pipeline.fitted_preprocessors]

            if any(checklist):
                return self.config.observation.preprocessing_pipeline.get_preprocessors('oceannanremover').final_locations.shape * len(self.config.observation.names)
            else:
                return (self.config.observation.info.coords['lat'].size, self.config.observation.info.coords['lon'].size) 
        else:
            return self.get_input_shape()
    
    @property
    def added_features_dim(self):

        return len(self.time_features)
        
        
    def __getitem__(self, ind):

        
        if self.condition_dataset is not None:
            if self.config.condition_method != 'static':
                year = float(self.cond_indexes['year'][ind])
                lead_time = float(self.cond_indexes['lead_time'][ind])
                selection = dict(year = year, lead_time = lead_time)
                if 'ensembles' in self.cond_indexes:
                    selection['ensembles'] =  self.cond_indexes['ensembles'][ind]  

            else:
                selection = {}

            condition = self.config.condition.preprocessing_pipeline.transform(self.condition_dataset.sel(**selection))  ##check if transforming once is faster
            condition = _unwrape_data_variables(condition)
        
        if self.observation_dataset is not None:
            year = float(self.obs_indexes['year'][ind])
            month = float(self.obs_indexes['month'][ind])
            selection = dict(year = year, month = month)

            if  'ensembles' in self.obs_indexes:
                selection['ensembles'] = self.obs_indexes['ensembles'][ind]

            target = self.config.observation.preprocessing_pipeline.transform(self.observation_dataset.sel(**selection))  ##check if transforming once is faster
            target = _unwrape_data_variables(target)

        year = float(self.model_indexes['year'][ind])
        lead_time = float(self.model_indexes['lead_time'][ind])
        selection = dict(year = year, lead_time = lead_time)

        if all([self.condition_dataset is not None,  self.config._using_model_data_as_condition,  not self._autoencoding_input]):
            input = condition
        else:
            if 'ensembles' in self.model_indexes:
                selection['ensembles'] = self.model_indexes['ensembles'][ind]
                
            input = self.config.model.preprocessing_pipeline.transform(self.model_dataset.sel(**selection))  ##check if transforming once is faster
            input = _unwrape_data_variables(input)

            if self._autoencoding_input:
                target = input

            if all([self.condition_dataset is not None,  self.config._using_model_data_as_condition]):
                input = condition

            elif all([self.condition_dataset is not None, not self.config._using_model_data_as_condition]):
                input = xr.concat([input, condition], dim = 'channels')

            
        time_features = self.get_time_features(year, lead_time)
        if (time_features is not None 
            and len(input.shape) > 2):
                time_features = np.broadcast_to(time_features[:, None, None], (len(time_features), input.shape[-2], input.shape[-1]))

        
        datadict = dict(input = torch.as_tensor(input.to_numpy(), dtype=torch.float32), 
                        target = torch.as_tensor(target.to_numpy(), dtype=torch.float32), 
                        added_features = torch.as_tensor(time_features, dtype=torch.float32) 
                          if time_features is not None else None)
                                                    

        if self.return_metadata:
            return datadict, selection
        else:
            return datadict


         
    def __len__(self):
        return len(self.model_indexes.get(list(self.model_indexes.keys())[0]))


    def get_time_features(self, year, lead_time):

        if self.time_features is not None:
            time_features_list = np.array([self.time_features]).flatten()
            feature_indices = {'year': 0, 'lead_time': 1, 'month_sin': 2, 'month_cos': 3}
            
            target_time = year + lead_time//12 
            target_month = lead_time

            y = (target_time - np.min(self.config.get_common_time)) /  (np.max(self.config.get_common_time) - np.min(self.config.get_common_time))
            lt = lead_time / self.config.num_lead_months
            msin = np.sin(2 * np.pi * target_month/12.0)
            mcos = np.cos(2 * np.pi * target_month/12.0)

            time_features = np.stack([y, lt, msin, mcos]) 
            time_features = time_features[..., [feature_indices[k] for k in time_features_list]]


            return time_features

    
        
    # def __getitem__(self, ind):
    #     year = float(self.indexes['year'][ind])
    #     lead_time = float(self.indexes['lead_time'][ind])
        
    #     if self.condition_dataset is not None:

    #         if self.config.condition_method != 'static':
    #             selection = dict(year = year, lead_time = lead_time)
    #         else:
    #             selection = {}

    #         if self.config.condition_method == 'cross_ensemble':
    #             selection['ensembles'] = np.random.choice(self.config.condition.info.coords['ensembles'])

    #         elif self.config.condition_method == 'same_member':
    #             selection['ensembles'] = self.indexes[ind].ensembles.values

    #         condition = self.config.condition.preprocessing_pipeline.transform(self.condition_dataset.sel(**selection))  ##check if transforming once is faster
    #         condition = _unwrape_data_variables(condition)
        
    #     if self.observation_dataset is not None:
    #         selection = dict(year = year, month = lead_time)
    #         lead_year, lead_month = np.divmod(lead_time - 0.5, 12)
    #         selection['year'] +=   lead_year
    #         selection['month'] =  lead_month + 0.5

    #         if all([not self.config.observation.ensemble_mean, self.config.observation.info.coords['ensembles'] is not None]):
    #             selection['ensembles'] = np.random.choice(self.config.observation.info.coords['ensembles']) 

    #         target = self.config.observation.preprocessing_pipeline.transform(self.observation_dataset.sel(**selection))  ##check if transforming once is faster
    #         target = _unwrape_data_variables(target)



    #     selection = dict(year = year, lead_time = lead_time)

    #     if all([self.condition_dataset is not None,  self.config._using_model_data_as_condition,  not self._autoencoding_input]):
    #         input = condition

    #     else:
    #         if all([not self.config.model.ensemble_mean, self.config.model.info.coords['ensembles'] is not None]):
    #             selection['ensembles'] = self.indexes[ind].ensembles.values
    #         input = self.config.model.preprocessing_pipeline.transform(self.model_dataset.sel(**selection))  ##check if transforming once is faster
    #         input = _unwrape_data_variables(input)

    #         if self._autoencoding_input:
    #             target = input

    #         if all([self.condition_dataset is not None,  self.config._using_model_data_as_condition]):
    #             input = condition

    #         elif all([self.condition_dataset is not None, not self.config._using_model_data_as_condition]):
    #             input = xr.concat([input, condition], dim = 'channels')

            

    #     if self.time_features is not None:
    #         time_features_list = np.array([self.time_features]).flatten()
    #         feature_indices = {'year': 0, 'lead_time': 1, 'month_sin': 2, 'month_cos': 3}
            
    #         target_time = year + lead_time//12 
    #         target_month = lead_time

    #         y = (target_time - np.min(self.config.get_common_time)) /  (np.max(self.config.get_common_time) - np.min(self.config.get_common_time))
    #         lt = lead_time / self.config.num_lead_months
    #         msin = np.sin(2 * np.pi * target_month/12.0)
    #         mcos = np.cos(2 * np.pi * target_month/12.0)

    #         time_features = np.stack([y, lt, msin, mcos]) 
    #         time_features = time_features[..., [feature_indices[k] for k in time_features_list]]
    #         if len(input.shape) > 2:
    #             time_features = np.broadcast_to(time_features[:, None, None], (len(time_features), input.shape[-2], input.shape[-1]))
    #     else:
    #         time_features = None

    
        
    #     datadict = dict(input = torch.as_tensor(input.to_numpy(), dtype=torch.float32), 
    #                     target = torch.as_tensor(target.to_numpy(), dtype=torch.float32), 
    #                     added_features = torch.as_tensor(time_features, dtype=torch.float32) 
    #                       if time_features is not None else None)
                                                    

    #     if self.return_metadata:
    #         return datadict, selection
    #     else:
    #         return datadict