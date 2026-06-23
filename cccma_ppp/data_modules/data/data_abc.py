import abc
from typing import final
from pathlib import Path
import gc
import glob
import xarray as xr
import numpy as np
import dataclasses


from cccma_ppp.data_modules.utils import (
    _load_xarray_data,
    _create_train_mask,
)
from cccma_ppp.generic import RuntimeContext


@dataclasses.dataclass
class infoclass:
    sizes: dict | None
    start_year: xr.DataArray | np.ndarray | str | int | None
    final_year: xr.DataArray | np.ndarray | str | int | None
    coords: dict
    spatial_mask: xr.Dataset = None



class DataConfigABC(abc.ABC):

    def __init__(self):
        if not hasattr(self, "preprocessing_pipeline"):
            raise AttributeError(
                f"{type(self).__name__} must define preprocessing_pipeline"
            )
        
        self.preprocessing_pipeline.set_name(self.TYPE)

    @property
    @abc.abstractmethod
    def TYPE(self) -> str:
        pass
    
    @classmethod
    @abc.abstractmethod
    def _allowed_dims(cls) -> frozenset[str]:
        pass
    
    @classmethod
    @abc.abstractmethod
    def _required_dims(cls) -> frozenset[str]:
        pass

    @final
    def _resolve_data(self):
        _resolve_data(self)

    @final
    def _get_ds_info(self):
        return  _get_ds_info(self)

    @final
    def _fit_preprocessor_pipeline(
        self,
        selection: dict,
        mask: bool = False,
        save: bool =True,
        save_path: Path | str | None = None,
        save_name: str | None = None,
        ):
        
        _base = _load_xarray_data(
            self.list_paths,
            names=self.names,
            concat_dim=self.concat_dim,
            selection=selection,
            ensemble_mean=self.ensemble_mean,
            rename_dict=self.rename_dict,
        )
        
        _mask = _create_train_mask(_base.year, _base.lead_time) if mask else None

        self.preprocessing_pipeline.fit(
            base_data=_base.load(),
            mask=_mask,
            save=save,
            save_path=save_path,
            save_name=save_name,
        )

        _base.close()
        del _base, _mask
        gc.collect()

    @final
    def _load_preprocessor_pipeline(
        self,
        load_dir: Path | str | None = None
        ):

        if load_dir is None:
            load_dir = Path(RuntimeContext.GLOBAL_EXP_DIR) / "preprocessing_pipeline"

        load_dir = (Path(load_dir)
                / f"{self.preprocessing_pipeline.name}_preprocessing_pipeline.joblib"
            )

        self.preprocessing_pipeline._load_from_memory(
            Path(load_dir),
        )

        if not self.preprocessing_pipeline.fitted:
            raise RuntimeError(
                f"the loaded preprocessor for {self.preprocessing_pipeline.name} is not fitted!"
            )
        


def _resolve_data(dataconfig: DataConfigABC) -> None:
    if not Path(dataconfig.paths).exists():
        raise FileNotFoundError(
            "The following file does not exist:\n" + str(dataconfig.paths)
        )

    list_paths = glob.glob(str(Path(dataconfig.paths).joinpath(dataconfig.file_type)))

    if len(list_paths) == 0:
        raise FileNotFoundError(
            f"The following file does is empty for {dataconfig.file_type} file type :\n"
            + str(dataconfig.paths)
        )

    for p in list_paths:
        with xr.open_dataset(Path(p)) as ds:
            if dataconfig.rename_dict is not None:
                ds = ds.rename(dataconfig.rename_dict)

            ds_dims = set(ds.dims)

            invalid = dataconfig._required_dims() - ds_dims
            if invalid:
                raise ValueError(
                    f"{dataconfig.TYPE} data must have {sorted(dataconfig._required_dims())} dimensions. Current dims : {sorted(ds_dims)} for {p}"
                )

            if dataconfig._check_ensemble:
                if not "ensembles" in ds.dims:
                    raise ValueError(
                        f"Cannot select ensemble_list as ensembles dim does not exist in {p}"
                    )

            invalid = ds_dims - dataconfig._allowed_dims() 
            if invalid:
                raise ValueError(
                    f'invalid data dimensions {sorted(ds_dims)} for {dataconfig.TYPE} data. Must be a subset ot {sorted(dataconfig._allowed_dims())} for {p}'
                )
            invalid = ds_dims - set(ds.coords.keys())
            if invalid:
                raise ValueError(
                    f'"coordinates for {sorted(ds_dims)} does not exist. Available coords: {sorted(set(ds.coords.keys()))} for {p}'
                    )

            missing = [name for name in dataconfig.names if name not in ds.data_vars]
            if missing:
                raise ValueError(f"{p} is missing variables: {missing}")
            
    dataconfig.list_paths = list_paths



def _get_ds_info(dataconfig: DataConfigABC) -> infoclass:
    if getattr(dataconfig, "list_paths", None) is None:
        list_paths = glob.glob(
            str(Path(dataconfig.paths).joinpath(dataconfig.file_type))
        )
    else:
        list_paths = dataconfig.list_paths

    ds = _load_xarray_data(
        list_paths,
        names=dataconfig.names,
        concat_dim=dataconfig.concat_dim,
        rename_dict=dataconfig.rename_dict,
    )

    if dataconfig.ensemble_list is not None:
        ds = ds.sel(ensembles=dataconfig.ensemble_list)

    if "year" in ds.dims:
        start_year, final_year = ds.year.min().values, ds.year.max().values
    else:
        start_year = final_year = None

    sizes = {
        dim: dict(ds.sizes).get(dim)
        for dim in dict(ds.sizes).keys()
        if dim not in ["ensembles", "lat", "lon"]
    }
    if not sizes:
        sizes = None

    coords = {
        dim: dict(ds.coords).get(dim, None) for dim in ["ensembles", "lat", "lon"]
    }

    ds.close()
    del ds

    return infoclass(
        start_year=start_year, final_year=final_year, sizes=sizes, coords=coords
    )

