import numpy as np
import xarray as xr
from torch.utils.data import Dataset
import torch
import dataclasses
from pathlib import Path
import os
import warnings

from data.data_abc import XarrayDatasetABC, XarrayDatasetConfigABC
from data.utils_data import DataConfig, WeightsConfig, _unwrape_data_variables, _load_xarray_data, infoclass, _create_train_mask

from preprocessing.preprocessing_ABC import PreprocessModuleABC



        
@dataclasses.dataclass
class XArrayDatasetConfig(Dataset, XarrayDatasetConfigABC):

    input: DataConfig
    target: DataConfig | None = None 
    condition : DataConfig | None = None 
    time_features : list[str] | None = None 
    num_lead_months  : int | None = None 
    
    def __post_init__(self):
        self._using_input_as_condition = False
        self.dataset = None

        assert self.input.condition_method is None, 'do not specify condition method for input'
        if all([self.target.ensemble_list is not None, not self.target.ensemble_mean]):
            assert len(self.target.ensemble_list) == len(self.input.ensemble_list), 'input and target must have same number of ensemble members'
        self.input_info = self._get_ds_info(self.input)
        assert 'lead_time' in self.input_info.sizes.keys(), f'lead_time is not an input dimention : {self.input_info.sizes.keys()}'
        
        self.num_input_lead_months = self.input_info.sizes['lead_time']
        self.input_range = np.arange(self.input_info.start_year, self.input_info.final_year  + self.num_input_lead_months//12 )
        self.input.preprocessing_pipeline.name = 'input'
       
        if self.target is not None:
            assert self.target.condition_method is None, 'do not specify condition method for target'
            self.target_info = self._get_ds_info(self.target)
            self.target_range = np.arange(self.target_info.start_year, self.target_info.final_year + 1 )
            if not self.target_info.coords["lat"].equals(self.input_info.coords["lat"]):
                warnings.warn(f'input and taget do not have the same latitudes cooridnates.') 
            if not self.target_info.coords["lon"].equals(self.input_info.coords["lon"]):
                warnings.warn(f'input and taget do not have the same longitudes cooridnates.') 
            self.target.preprocessing_pipeline.name = 'target'


        if self.condition is not None:
            assert self.condition.condition_method is not None, f'specify condition_method from {DataConfig._available_condiiton_methods()} '
            
            if self.condition.paths is None:              
               self._using_input_as_condition = True
            elif self.condition.paths == self.input.paths:
                if self.condition.names == self.input.names:
                    self._using_input_as_condition = True
            else: 
                self._using_input_as_condition = False

            if self._using_input_as_condition:
               self.condition = DataConfig(paths = self.input.path, 
                                           names = self.input.names, 
                                           preprocessing_pipeline = self.input.preprocessing_pipeline, 
                                            ensemble_list = self.input.ensemble_list,
                                            concat_dim = self.input.concat_dim,
                                            file_type = self.input.file_type,
                                            ensemble_mean  = None,
                                           condition_method = self.condition.condition_method)

            self.condition_info = self._get_ds_info(self.condition)
            if self.condition.condition_method not in ['static']:
                assert self.condition_info.sizes == self.input_info.sizes, 'non static coniditioning field must have the same times as the input.'
            assert self.condition_info.coords['lat'] == self.input_info.coords['lat'], 'coniditioning field must have the same lat shape as the input.'
            assert self.condition_info.coords['lon'] == self.input_info.coords['lon'], 'coniditioning field must have the same lon shape as the input.'
            if self.condition.condition_method in ['same_member']:
                   assert self.condition_info.coords['ensembles'] == self.input_info.coords['ensembles'], 'coniditioning field and input must have the same ensemble dimension for same_member conditioning.'

            self.condition_range = np.arange(self.condition_info.start_year, self.condition_info.final_year + self.num_input_lead_months//12)
            self.condition.preprocessing_pipeline.name = 'condition'

        if self.num_lead_months is None:
            self.num_lead_months = self.num_input_lead_months
        assert self.num_lead_months <= self.num_input_lead_months, f'Maximum available lead months is {self.num_input_lead_months}' 

        if self.time_features is not None:
            assert set(self.time_features).issubset(set(['year','lead_time','month_sin','month_cos']))


        
    @property
    def get_common_time(self):
        if self.target is not None:
            return np.intersect1d(self.input_range, self.target_range)
        else:
            return self.input_range
    
    @property
    def available_train_time(self):
        num_lead_years = self.num_lead_months // 12
        if self.target is None:
            return np.arange( np.min(self.get_common_time), np.max(self.get_common_time) + 1 - num_lead_years + 1) 
        else:
            return self.get_common_time
        

    def _get_ds_info(self, data_dict) -> infoclass:

        ds = _load_xarray_data(data_dict.paths, names = data_dict.names, concat_dim = data_dict.concat_dim)
        if data_dict.ensemble_list is not None:
                ds = ds.sel(ensembles = data_dict.ensemble_list)
        start_year, final_year = ds.year.min().values, ds.year.max().values
        sizes =  {dim : dict(ds.sizes).get(dim) for dim in dict(ds.sizes).keys() if dim not in ['ensembles', 'lat', 'lon']}
        coords = {dim : dict(ds.coords).get(dim, None) for dim in ['ensembles', 'lat', 'lon']}
        
        ds.close()
        del ds

        return infoclass(start_year = start_year, final_year = final_year, sizes = sizes, coords = coords)
    
    def _fit_preprocessors(self, train_years : np.ndarray | list | tuple, save = False, save_path : Path | str | None = None, save_name : str| None = None):
 
        if self.input.preprocessing_pipeline.load_dir is None:

            selection={'year' : train_years, 'lead_time' :  np.arange(1,self.num_lead_months + 1) }
            if self.input_info.coords['ensembles'] is not None: 
                selection['ensembles'] = self.input_info.coords['ensembles']
            input_base = _load_xarray_data(self.input.paths, names = self.input.names, concat_dim = self.input.concat_dim, selection=selection, ensemble_mean=self.input.ensemble_mean)
            mask_input = _create_train_mask(input_base.year, input_base.lead_time)
            self.input.preprocessing_pipeline.fit(base_data = input_base.load(), mask = mask_input, save = save, save_path = save_path ,save_name = save_name)
            input_base.close()
            del input_base
        else:
            self.input.preprocessing_pipeline._load_from_memory(Path(self.input.preprocessing_pipeline.load_dir), load_name = self.input.preprocessing_pipeline.load_name)
        
        if self.target is not None:
            if self.target.preprocessing_pipeline.load_dir is None:
                selection={'year' : train_years }
                if self.target_info.coords['ensembles'] is not None: 
                    selection['ensembles'] = self.target_info.coords['ensembles']
                target_base = _load_xarray_data(self.target.paths,  names = self.target.names ,concat_dim = self.target.concat_dim, selection = selection , ensemble_mean=self.target.ensemble_mean)
                self.target.preprocessing_pipeline.fit(base_data = target_base.load(), save = save, save_path = save_path, save_name = save_name)
                target_base.close()
                del target_base
            else:
                self.target.preprocessing_pipeline._load_from_memory(Path(self.target.preprocessing_pipeline.load_dir), load_name = self.target.preprocessing_pipeline.load_name)  

        if self.condition is not None:
            if self.condition.preprocessing_pipeline.load_dir is None:
                selection={'year' : train_years }
                if self.condition_info.coords['ensembles']is not None:
                    selection['ensembles'] = self.condition_info.coords['ensembles']
                condition_base = _load_xarray_data(self.condition.paths, names = self.condition.names,  concat_dim = self.condition.concat_dim, selection= selection, ensemble_mean=self.condition.ensemble_mean)
                self.condition.preprocessing_pipeline.fit(base_data =  condition_base.load(), mask = mask_input, save = save, save_path = save_path, save_name = save_name)
                condition_base.close()
                del condition_base
            else:
                self.condition.preprocessing_pipeline._load_from_memory(Path(self.condition.preprocessing_pipeline.load_dir ), load_name = self.condition.preprocessing_pipeline.load_name)   

    def _load_fitted_preprocessors(self, load_dir : Path | str | None = None, load_name : str| None  = None):
        

        if load_dir is None:
            load_dir = Path(os.environ["GLOBAL_EXP_DIR"]) / 'preprocessing_pipeline' / f"{self.input.preprocessing_pipeline.name}_preprocessing_pipeline.joblib"
        else:
            load_dir = Path(load_dir)
    
        self.input.preprocessing_pipeline._load_from_memory(Path(load_dir), load_name = load_name)
        assert self.input.preprocessing_pipeline.fitted, 'the loaded preprocessor for input is not fitted!'

        if self.target is not None:
            if load_dir is None:
                load_dir = Path(os.environ["GLOBAL_EXP_DIR"]) / 'preprocessing_pipeline' /  f"{self.target.preprocessing_pipeline.name}_preprocessing_pipeline.joblib"
            else:
                load_dir = Path(load_dir)

            self.target.preprocessing_pipeline._load_from_memory(Path(load_dir), load_name = load_name)     
            assert self.target.preprocessing_pipeline.fitted, 'the loaded preprocessor for target is not fitted!'

        if self.condition is not None:
            if self.condition.preprocessing_pipeline.load_dir is None:
                load_dir = Path(os.environ["GLOBAL_EXP_DIR"]) / 'preprocessing_pipeline' / f"{self.condition.preprocessing_pipeline.name}_preprocessing_pipeline.joblib"
            else:
                load_dir = Path(load_dir)
            
            self.condition.preprocessing_pipeline._load_from_memory(Path(load_dir), load_name = load_name)          
            assert self.condition.preprocessing_pipeline.fitted, 'the loaded preprocessor for condition is not fitted!'    
               

    def _add_fitted_preprocessor(self, preprocessor: PreprocessModuleABC , index = 0):
        if not isinstance(preprocessor, PreprocessModuleABC):
            raise TypeError(
                f"preprocessor must be an instance of ProcessorConfig, "
                f"got {type(preprocessor)}"
        )
        assert preprocessor.fitted, 'The preprocessor must be fitted'

        self.input.preprocessing_pipeline.add_fitted_preprocessor(preprocessor, index = index)
        if self.target is not None:
            self.target.preprocessing_pipeline.add_fitted_preprocessor(preprocessor, index = index)
        if self.condition is not None:
            self.condition.preprocessing_pipeline.add_fitted_preprocessor(preprocessor, index = index)

    def get_weights(self, config: WeightsConfig | None= None, save = True, save_path : Path | str | None = None, save_name : str| None = None):
        if config is None:
            config =  WeightsConfig()

        if self.target is not None:
            target_coords = self.target_info.coords.copy()
        else:
            target_coords = self.input_info.coords.copy()

        if 'ensembles' in target_coords:
            del target_coords['ensembles']

        from preprocessing.utils_preprocessing import Oceannanremove
        if self.target is not None:
            pipeline = self.target.preprocessing_pipeline
        else:
            pipeline = self.input.preprocessing_pipeline

        checklist = [isinstance(item, Oceannanremove) for item in pipeline.fitted_preprocessors]

        weights =  config.build_weights(target_coords,
                                        oceannanremover=pipeline.get_preprocessors('oceannanremover') if any(checklist) else None,
                                        save=save,
                                        save_path=save_path,
                                        save_name=save_name)

        if 'channels' in weights.dims:
            error_msg = f'inconsistent variable weights {weights.channels.values} for taget variables {self.target.names}'
            if self.target is not None:
                assert weights.channels.values == self.target.names, error_msg
            else:
                assert weights.channels.values == self.input.names, error_msg
        
        return weights


    def build(self, years : np.ndarray, mask: xr. DataArray = None, return_metadata : bool = False):
        return  XArrayDataset(config = self, requested_years = years,mask =  mask, return_metadata = return_metadata)
        


        

@dataclasses.dataclass
class XArrayDataset(Dataset, XarrayDatasetABC):

    
    config:XArrayDatasetConfig
    requested_years : list[int]  | tuple[int]  | np.ndarray
    mask : xr.DataArray = None
    return_metadata : bool = False
    
    def __post_init__(self):
        assert set(self.requested_years).issubset(set(self.config.get_common_time)), 'the requested years are not common to input and target data.'

        self.input = self.config.input
        self.target = self.config.target
        self.condition = self.config.condition
        self._autoencoding_input = False
        
        # if  self.config._using_input_as_condition is False:
        self.input.dataset = _load_xarray_data(self.input.paths, 
                                               names = self.input.names, 
                                               ensemble_mean=self.input.ensemble_mean, 
                                            #    preprocessor = self.input.preprocessing_pipeline,  ##checking if doing this once is faster
                                               concat_dim = self.input.concat_dim)
        if self.target is not None:
            self.target.dataset = _load_xarray_data(self.target.paths, 
                                               names = self.target.names, 
                                               ensemble_mean=self.target.ensemble_mean, 
                                            #    preprocessor = self.target.preprocessing_pipeline,  ##checking if doing this once is faster
                                               concat_dim = self.target.concat_dim)
        else:
            self._autoencoding_input = True

        if self.condition is not None:
            self.condition.dataset = _load_xarray_data(self.condition.paths, 
                                               names = self.condition.names, 
                                               ensemble_mean=self.condition.ensemble_mean, 
                                            #    preprocessor = self.condition.preprocessing_pipeline,  ##checking if doing this once is faster
                                               concat_dim = self.condition.concat_dim)            

        if self.mask is None:
            self.mask = _create_train_mask(years = self.config.input_range, 
                                           lead_times= np.arange(1, self.config.input_info.sizes['lead_time']+ 1) )
            self.mask = xr.full_like(self.mask, fill_value = False)
                                                                          
        self.mask = self.mask.sel(year = self.requested_years).sel(lead_time = np.arange(1,self.config.num_lead_months + 1))
        if all([not self.input.ensemble_mean, self.config.input_info.coords['ensembles'] is not None]):
            self.mask = self.mask.expand_dims(ensembles = len(self.config.input_info.coords['ensembles']), axis = 0)
            self.mask = self.mask.assign_coords(ensembles = self.config.input_info.coords['ensembles'] )

        self.mask = self.mask.where(~self.mask)
        mask = self.mask.stack(batch = dict(self.mask.sizes).keys()).transpose('batch', ...).dropna(dim = 'batch').batch.values
        mask = tuple(map(np.array, zip(*mask)))
        self.indexes =  { key : mask[ind] for ind, key in enumerate(tuple(dict(self.mask.sizes).keys()))}
        del mask           

        self.time_features = self.config.time_features

    def get_input_shape(self):
        from preprocessing.utils_preprocessing import Oceannanremove
        checklist = [isinstance(item, Oceannanremove) for item in self.input.preprocessing_pipeline.fitted_preprocessors]

        if any(checklist):
            return self.input.preprocessing_pipeline.get_preprocessors('oceannanremover').final_locations.shape * len(self.input.names)
        else:
            return (self.config.input_info.coords['lat'].size, self.config.input_info.coords['lon'].size)
        
    def get_target_shape(self):
        from preprocessing.utils_preprocessing import Oceannanremove
        checklist = [isinstance(item, Oceannanremove) for item in self.target.preprocessing_pipeline.fitted_preprocessors]

        if any(checklist):
            return self.target.preprocessing_pipeline.get_preprocessors('oceannanremover').final_locations.shape * len(self.target.names)
        else:
            return (self.config.target_info.coords['lat'].size, self.config.target_info.coords['lon'].size) 
    
    @property
    def added_features_dim(self):

        return len(self.time_features)
        
    def __getitem__(self, ind):
        year = float(self.indexes['year'][ind])
        lead_time = float(self.indexes['lead_time'][ind])
        
        if self.condition is not None:
            if self.condition.condition_method != 'static':
                selection = dict(year = year, lead_time = lead_time)
            else:
                selection = {}
            if self.condition.condition_method == 'cross_ensmeble':
                selection['ensembles'] = np.random.choice(self.config.condition_info.coords['ensembles'])
            elif self.condition.condition_method == 'same_member':
                selection['ensembles'] = self.indexes[ind].ensembles.values
            condition = self.condition.preprocessing_pipeline.transform(self.condition.dataset.sel(**selection))  ##check if transforming once is faster
            condition = _unwrape_data_variables(condition)
        
        if self.target is not None:
            selection = dict(year = year, lead_time = lead_time)
            lead_year, lead_month = np.divmod(lead_time - 0.5, 12)
            selection['year'] +=   lead_year
            selection['lead_time'] =  lead_month + 0.5

            if all([not self.target.ensemble_mean, self.config.target_info.coords['ensembles'] is not None]):
                selection['ensembles'] = np.random.choice(self.config.target_info.coords['ensembles']) 
            target = self.target.preprocessing_pipeline.transform(self.target.dataset.sel(**selection))  ##check if transforming once is faster
            target = _unwrape_data_variables(target)



        selection = dict(year = year, lead_time = lead_time)

        if all([self.condition is not None,  self.config._using_input_as_condition,  not self._autoencoding_input]):
            input = condition

        else:
            if all([not self.input.ensemble_mean, self.config.input_info.coords['ensembles'] is not None]):
                selection['ensembles'] = self.indexes[ind].ensembles.values
            input = self.input.preprocessing_pipeline.transform(self.input.dataset.sel(**selection))  ##check if transforming once is faster
            input = _unwrape_data_variables(input)

            if self._autoencoding_input:
                target = input

            if all([self.condition is not None,  self.config._using_input_as_condition]):
                input = condition

            elif all([self.condition is not None, not self.config._using_input_as_condition]):
                input = xr.concat([input, condition], dim = 'channels')

            

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
            if len(input.shape) > 2:
                time_features = np.broadcast_to(time_features[:, None, None], (len(time_features), input.shape[-2], input.shape[-1]))
        else:
            time_features = None

    
        
        datadict = dict(input = torch.as_tensor(input.to_numpy(), dtype=torch.float32), 
                        target = torch.as_tensor(target.to_numpy(), dtype=torch.float32), 
                        added_features = torch.as_tensor(time_features, dtype=torch.float32) 
                          if time_features is not None else None)
                                                    

        if self.return_metadata:
            return datadict, selection
        else:
            return datadict


         
    def __len__(self):
        return len(self.indexes.get(list(self.indexes.keys())[0]))

    
