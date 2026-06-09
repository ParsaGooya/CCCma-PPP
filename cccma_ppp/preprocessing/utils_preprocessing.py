import numpy as np
import xarray as xr
from pathlib import Path
import joblib
import os

from cccma_ppp.preprocessing.preprocessing import (
    PreprocessingStepSelector,
)                         
from cccma_ppp.preprocessing.preprocessing_ABC import PreprocessModuleABC


                                               
@PreprocessingStepSelector.register("normalizer")
class Normalizer(PreprocessModuleABC):
    def __init__(self, dims: list | None = None, **kwargs) -> None:
        self.min = None
        self.max = None
        self.dims = dims
        self.fitted = False

        if self.dims is not None:
            self.dims = tuple(self.dims)

    def fit(self, data: xr.DataArray, mask: xr.DataArray = None):

        if all(
            ["ensembles" in data.dims, self.dims is not None]
        ):                                                                                                                                                               
            if "ensembles" not in self.dims:
                self.large_ensemble = True
                self.dims = (
                    "ensembles",
                    *self.dims,
                )                                                                                            

        if mask is not None:
            data_masked = data.where(~np.isnan(mask))
        else:
            data_masked = data

        self.min = data_masked.min(self.dims).load()
        self.max = data_masked.max(self.dims).load()
        self.fitted = True
        return self

    def transform(self, data: xr.DataArray):
        data_normalized = (data - self.min) / (self.max - self.min)
        return data_normalized

    def inverse_transform(self, data: xr.DataArray):
        data_raw = data * (self.max - self.min) + self.min
        return data_raw


                                                 
@PreprocessingStepSelector.register("standardizer")
class Standardizer(PreprocessModuleABC):
    def __init__(self, dims: list | None = None, **kwargs) -> None:
        self.mean = None
        self.std = None
        self.dims = dims
        self.fitted = False

        if self.dims is not None:
            self.dims = tuple(self.dims)

    def fit(self, data: xr.DataArray, mask: xr.DataArray = None):

        if all(
            ["ensembles" in data.dims, self.dims is not None]
        ):                                                                                                                                                               
            if "ensembles" not in self.dims:
                self.large_ensemble = True
                self.dims = (
                    "ensembles",
                    *self.dims,
                )                                                                                            

        if mask is not None:
            data_masked = data.where(~np.isnan(mask))
        else:
            data_masked = data

        self.mean = data_masked.mean(self.dims).load()
        std = data_masked.std(self.dims).load()
        self.std = std.where(std > 0)
        self.fitted = True
        return self

    def transform(self, data: xr.DataArray):

        data_standardized = (data - self.mean) / self.std

        return data_standardized

    def inverse_transform(self, data: xr.DataArray):

        data_raw = data * self.std + self.mean

        return data_raw


                                              
@PreprocessingStepSelector.register("anomalies")
class AnomaliesScaler(PreprocessModuleABC):
    def __init__(self, dims: list | None = None, **kwargs) -> None:
        self.mean = None
        self.dims = dims
        self.fitted = False

        if self.dims is not None:
            self.dims = tuple(self.dims)

    def fit(self, data: xr.DataArray, mask: xr.DataArray = None):

        if all(
            ["ensembles" in data.dims, self.dims is not None]
        ):                                                                                                                                                               
            if "ensembles" not in self.dims:
                self.large_ensemble = True
                self.dims = (
                    "ensembles",
                    *self.dims,
                )                                                                                            

        if mask is not None:
            data_masked = data.where(~np.isnan(mask))
        else:
            data_masked = data      

        self.mean = data_masked.mean(self.dims).load()
        self.fitted = True
        return self

    def transform(self, data: xr.DataArray):
        data_anomalies = data - self.mean
        return data_anomalies

    def inverse_transform(self, data: xr.DataArray):

        if data.shape[-3] > 12 and self.mean.shape[-3] <= 12:
            lead_years = int(data.shape[-3] / 12)
            mean = xr.concat(
                [self.mean for _ in range(lead_years)], dim=self.mean.dims()[-3]
            )
            data_raw = data + mean
        else:
            data_raw = data + self.mean
        return data_raw


                                                    
@PreprocessingStepSelector.register("oceannanremover")
class Oceannanremove(PreprocessModuleABC):       
    def __init__(self, load_dir: Path | str = None, **kwargs):

        self.load_dir = load_dir
        self.fitted = False

    def fit(
        self,
        data: xr.DataArray,
        target: xr.DataArray | None = None,
        mask=None,
        save: bool = False,
        save_name: str | None = None,
        save_path: Path | str = None,
    ):                                                                    

        if self.load_dir is None:
            if target is not None:
                self.reference_shape = xr.Dataset(
                    coords={"lat": target["lat"], "lon": target["lon"]}
                )

                temp = target.stack(
                    ref=["lat", "lon"]
                ).sel(
                    ref=data.stack(ref=["lat", "lon"]).dropna(dim="ref").ref
                )                                                                               
                self.final_locations = (
                    temp.dropna("ref").load().ref
                )                                                                                                  
                self.common_to_input_and_target = True
            else:
                self.reference_shape = xr.Dataset(
                    coords={"lat": data["lat"], "lon": data["lon"]}
                )

                self.final_locations = (
                    data.stack(ref=["lat", "lon"]).dropna(dim="ref").ref.load()
                )                                                                               
                self.common_to_input_and_target = False

            self.fitted = True
            if save:
                save_name = save_name or "oceannanremover"
                save_path = Path(save_path) or Path(os.get["GLOBAL_EXP_DIR"])

                joblib.dump(self, save_path.joinpath(f"{save_name}.joblib"))
        else:
            self._load_from_memory(self.load_dir)

        return self

    def transform(
        self, data: xr.DataArray
    ):                                                               

        conditions = ["lat" in data.dims, "lon" in data.dims]

        if all(conditions):                             
            sampled = data.stack(ref=["lat", "lon"]).sel(ref=self.final_locations)
        else:                                                    
            sampled = data.sel(ref=self.final_locations)

        return sampled

    def inverse_transform(
        self, data: xr.DataArray
    ):                                              

        return data.unstack().combine_first(
            self.reference_shape
        )                                                                                                                                    

    def _load_from_memory(self, load_dir: Path | str):

        loaded = joblib.load(Path(load_dir))
        assert loaded.fitted, "the preprocessor to be loaded has to be fitted first."

        self.reference_shape = loaded.reference_shape
        self.final_locations = loaded.final_locations
        self.fitted = loaded.fitted
        del loaded
