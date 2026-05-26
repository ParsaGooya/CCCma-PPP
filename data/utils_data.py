import numpy as np
import pandas as pd
import xarray as xr
import dataclasses
from preprocessing.preprocessing import PreprocessingPipeline
from preprocessing.utils_preprocessing import Oceannanremove
import os
from pathlib import Path
import glob



@dataclasses.dataclass
class infoclass:
    sizes : dict
    start_year : xr.DataArray | np.ndarray | str | int
    final_year : xr.DataArray | np.ndarray | str | int
    coords : dict
    spatial_mask :  xr.Dataset = None





@dataclasses.dataclass
class DataConfig:
    paths: str | None = None
    names: list[str]  | None = None
    preprocessing_pipeline :  PreprocessingPipeline = dataclasses.field(default_factory= PreprocessingPipeline())
    ensemble_list: list | None = None
    ensemble_mean : bool | None = True
    condition_method: str = None
    concat_dim : str = 'year'
    file_type = '*.nc'

    
    def __post_init__(self):
        self.check_ensemble = False
        
        if self.condition_method is not None:
            assert self.condition_method in self._available_condiiton_methods(), f'{self.condition_method} is not a valid conditioning method. Available methods: {self._available_condiiton_methods()}'
         
            if self.condition_method in ['cross_ensemble', 'same_member']:
                assert self.ensemble_mean is not True, 'Ensemble mean cannot be True for cross_ensemble or same_member conditioning.'
                self.ensemble_mean = False
                self.check_ensemble = True
            elif  self.condition_method == 'ensemble_mean':
                self.ensemble_mean = True
                self.check_ensemble = True
            else:
                assert self.ensemble_list is None, 'For "static" or "no_ensemble" conditioning fields do not specify ensemble list.'
                self.check_ensemble = False


            if self.paths is not None:
                assert self.names is not None, 'specify the names of variables to load or set paths to None to use input as condition.'
                self._check_data()

        else:
            self._check_data()

    def _check_data(self) -> None:
        if not Path(self.paths).exists():
            raise FileNotFoundError(
                f"The following file does not exist:\n" + "\n".join(self.paths))
        
        list_paths = glob.glob(str(Path(self.paths).joinpath(self.file_type)))

        if len(list_paths) == 0:
            raise FileNotFoundError(
                f"The following file does is empty for {self.file_type} file type :\n" + "\n".join(self.paths))
        
        for p in list_paths:
            with xr.open_dataset(Path(p)) as ds:
                if self.check_ensemble:
                    assert 'ensembles' in ds.dims, f'"ensembles" not a valid data dimension: {ds.dims}'
                if self.condition_method == 'static':
                    assert all(['time','year','lead_time', 'month', 'ensembles']) not in ds.dims, 'static fields connot have time or ensemble dimensions.'

                missing = [name for name in self.names if name not in ds.data_vars]
                if missing:
                    raise ValueError(
                        f"{p} is missing variables: {missing}"
                    )
        self.paths = list_paths

    @classmethod
    def _available_condiiton_methods(cls):
        return ['ensemble_mean' , 'cross_ensemble' , 'same_member', 'no_ensemble', 'static']



@dataclasses.dataclass
class WeightsConfig:
    spatial_method: str = 'uniform'
    variable_weights: dict[str, float] | None = None
    load_dir : str | None = None

    def __post_init__(self):
        if self.load_dir is None:
            assert self.spatial_method.lower() in ['cosine_lat', 'uniform']

    def build_weights(self, 
                      target_coords = None, 
                      oceannanremover : Oceannanremove| None = None,
                      save = True, 
                      save_path : Path | str | None = None, 
                      save_name : str| None = None):

        if self.load_dir is not None:

            weights =  xr.open_dataset(self.load_dir)
            if isinstance(weights, xr.Dataset):
                weights = _unwrape_data_variables(weights)

            if "ref" in weights.dims and {"lat", "lon"}.issubset(weights.coords):
                weights = weights.set_index(ref=["lat", "lon"])

            if oceannanremover is not None:
                assert weights.coords['ref'].equals(oceannanremover.final_locations)
            else:
                assert weights.coords['lat'].equals(target_coords['lat'])
                assert weights.coords['lon'].equals(target_coords['lon'])

        else:
            weights = np.cos(np.ones_like(target_coords['lon']) * (np.deg2rad(target_coords['lat'].to_numpy()))[..., None]) 
            weights = xr.DataArray(weights, coords = target_coords, name = 'weights')

            if self.spatial_method == 'uniform':
                weights = xr.ones_like(weights)

            if self.variable_weights is not None:
                weights = xr.DataArray(list(self.variable_weights.values()), coords = {'channels': list(self.variable_weights.keys())}) * weights


            if oceannanremover is not None:
                weights = oceannanremover.transform(weights)

            if save:
                                
                save_path = Path(save_path) if save_path is not None else Path(os.environ["GLOBAL_EXP_DIR"])
                save_name = save_name or f"spatial_weights.nc"

                if not os.path.isdir(save_path):
                    os.makedirs(save_path)

                weights.reset_index("ref").to_netcdf(save_path / save_name )

        return weights



def _unwrape_data_variables(dataset : xr.Dataset):

        return xr.concat([dataset[v].expand_dims('channels', axis = 0 ) for v in list(dataset.data_vars)], dim='channels')


def _load_xarray_data(paths : list[str], 
                      selection: dict|None = None, 
                      names: list|None = None,
                      ensemble_mean = True, 
                      preprocessor : PreprocessingPipeline |None = None,
                      concat_dim = 'year'):
        
        ds =  xr.open_mfdataset(paths, combine = 'nested', concat_dim = concat_dim)
        ds =  ds.sel(selection) if selection is not None else ds

        if 'ensembles' in ds.coords:
            ds = ds.mean('ensembles') if ensemble_mean else ds

        if names is not None:
            ds = ds[names]
        if preprocessor is not None:
            ds = preprocessor.transform(ds)

        return ds



def _create_train_mask(years : list | xr.DataArray , lead_times: list | xr.DataArray |int , exclude_idx=0):

    if isinstance(lead_times , int):
        lead_times = np.arange(1, lead_times+1)

    mask = np.full((len(years), len(lead_times)), False, dtype=bool)
    x = np.arange(0, 12*mask.shape[0], 12)   
    y = np.arange(1, mask.shape[1] + 1)
    idx_array = x[..., None] + y
    # mask[idx_array >= idx_array[-1, exclude_idx + 12]] = True
    mask[idx_array > idx_array[-1, exclude_idx + 11]] = True

    return xr.DataArray(
        mask,
        dims=('year', 'lead_time'),
        coords={'year': years, 'lead_time': lead_times},
        name="mask"
    )
