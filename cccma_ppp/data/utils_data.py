import numpy as np
import xarray as xr
import dataclasses

from pathlib import Path
import glob
from typing import final, ClassVar
import os
from typing import Literal


from cccma_ppp.data.data_abc import DataConfigABC
from cccma_ppp.preprocessing.preprocessing import PreprocessingPipeline
from cccma_ppp.preprocessing.utils_preprocessing import Oceannanremove

from cccma_ppp.generic.runtime import RuntimeContext


spatialmethod = Literal["uniform", "cosine_lat"]


@dataclasses.dataclass
class infoclass:
    sizes: dict | None
    start_year: xr.DataArray | np.ndarray | str | int | None
    final_year: xr.DataArray | np.ndarray | str | int | None
    coords: dict
    spatial_mask: xr.Dataset = None


@dataclasses.dataclass
class ModelDataConfig(DataConfigABC):
    paths: str
    names: list[str]
    preprocessing_pipeline: PreprocessingPipeline = dataclasses.field(
        default_factory=PreprocessingPipeline
    )
    ensemble_list: list | None = None
    ensemble_mean: bool | None = True
    concat_dim: str = "year"
    file_type: str = "*.nc"
    rename_dict: dict = None
    TYPE: ClassVar[str] = "model"

    def __post_init__(self):
        self._check_ensemble = False
        if self.ensemble_list is not None:
            self._check_ensemble = True

        _check_data(self)
        self.info = _get_ds_info(self)
        self.year_range = np.arange(
            self.info.start_year,
            self.info.final_year + self.info.sizes["lead_time"] // 12,
        )

    @final
    @classmethod
    def _allowed_dims(cls):
        return ["year", "lead_time", "ensembles", "lat", "lon"]

    @final
    @classmethod
    def _required_dims(cls):
        return ["lead_time", "ensembles", "lat", "lon"]


@dataclasses.dataclass
class ObsDataConfig(DataConfigABC):
    paths: str
    names: list[str]
    preprocessing_pipeline: PreprocessingPipeline = dataclasses.field(
        default_factory=PreprocessingPipeline
    )
    ensemble_list: list | None = None
    ensemble_mean: bool | None = True
    concat_dim: str = "year"
    file_type: str = "*.nc"
    rename_dict: dict = None
    TYPE: ClassVar[str] = "observation"

    def __post_init__(self):
        self._check_ensemble = False
        if self.ensemble_list is not None:
            self._check_ensemble = True

        _check_data(self)
        self.info = _get_ds_info(self)
        self.year_range = np.arange(self.info.start_year, self.info.final_year + 1)

    @final
    @classmethod
    def _allowed_dims(cls):
        return ["year", "month", "ensembles", "lat", "lon"]

    @final
    @classmethod
    def _required_dims(cls):
        return ["month", "lat", "lon"]


@dataclasses.dataclass
class ConditionDataConfig(DataConfigABC):
    paths: str
    names: list[str]
    preprocessing_pipeline: PreprocessingPipeline = dataclasses.field(
        default_factory=PreprocessingPipeline
    )
    ensemble_list: list | None = None
    ensemble_mean: bool | None = True
    concat_dim: str = "year"
    file_type: str = "*.nc"
    rename_dict: dict = None
    TYPE: ClassVar[str] = "model"

    def __post_init__(self):
        self._check_ensemble = False
        if self.ensemble_list is not None:
            self._check_ensemble = True

        _check_data(self)
        self.info = _get_ds_info(self)
        if self.info.start_year is not None and self.info.final_year is not None:
            self.year_range = np.arange(
                self.info.start_year,
                self.info.final_year + self.info.sizes["lead_time"] // 12,
            )

    @final
    @classmethod
    def _allowed_dims(cls):
        return ["year", "lead_time", "ensembles", "lat", "lon"]

    @final
    @classmethod
    def _required_dims(cls):
        return ["lat", "lon"]


def _check_data(dataconfig: ModelDataConfig | ObsDataConfig) -> None:
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

            for dim in dataconfig._required_dims():
                if not dim in ds.dims:
                    raise ValueError(
                        f"{dataconfig.TYPE} data must have {dataconfig._required_dims()} dimensions. Current dims : {ds.dims}"
                    )

            if dataconfig._check_ensemble:
                if not "ensembles" in ds.dims:
                    raise ValueError(
                        "Cannot select ensemble_list as ensembles dim does not exist"
                    )

            for dim in ds.dims:
                if not dim in dataconfig._allowed_dims():
                    raise ValueError(
                        f'"{dim}" not a valid dimension for {dataconfig.TYPE} data: {dataconfig._allowed_dims()}'
                    )
                if not dim in ds.coords:
                    raise ValueError(
                        f'"coordinates for {dim} dimension does not exist. Available coords: {dict(ds.coords).keys()}'
                    )

            missing = [name for name in dataconfig.names if name not in ds.data_vars]
            if missing:
                raise ValueError(f"{p} is missing variables: {missing}")
    dataconfig.list_paths = list_paths


def _get_ds_info(dataconfig: ModelDataConfig | ObsDataConfig) -> infoclass:
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


@dataclasses.dataclass
class WeightsConfig:
    spatial_method: spatialmethod = "uniform"
    variable_weights: dict[str, float] | None = None
    load_dir: Path | str | None = None

    def __post_init__(self):
        if self.load_dir is not None:
            if not Path(self.load_dir).exists():
                raise FileNotFoundError(f"weights file not found at {self.load_dir}")

    def build_weights(
        self,
        target_coords=None,
        oceannanremover: Oceannanremove | None = None,
        save=True,
        save_path: Path | str | None = None,
        save_name: str | None = None,
    ):

        if self.load_dir is not None:
            weights = xr.open_dataset(Path(self.load_dir))
            if isinstance(weights, xr.Dataset):
                weights = _unwrap_data_variables(weights)

            if "ref" in weights.dims and {"lat", "lon"}.issubset(weights.coords):
                weights = weights.set_index(ref=["lat", "lon"])

            msg = f"the loaded weights from {self.load_dir} must have lat and lon coordinates that match the target coordinates"

            if oceannanremover is not None:
                if not weights.coords["ref"].equals(oceannanremover.final_locations):
                    raise ValueError(msg)
            else:
                if not weights.coords["lat"].equals(target_coords["lat"]):
                    raise ValueError(msg)
                if not weights.coords["lon"].equals(target_coords["lon"]):
                    raise ValueError(msg)

        else:
            weights = np.cos(
                np.ones_like(target_coords["lon"])
                * (np.deg2rad(target_coords["lat"].to_numpy()))[..., None]
            )
            weights = xr.DataArray(weights, coords=target_coords, name="weights")

            if self.spatial_method == "uniform":
                weights = xr.ones_like(weights)

            if self.variable_weights is not None:
                weights = (
                    xr.DataArray(
                        list(self.variable_weights.values()),
                        coords={"channels": list(self.variable_weights.keys())},
                    )
                    * weights
                )

            if oceannanremover is not None:
                weights = oceannanremover.transform(weights)

            if save:
                save_path = (
                    Path(save_path)
                    if save_path is not None
                    else Path(RuntimeContext.GLOBAL_EXP_DIR)
                )
                save_name = save_name or "spatial_weights.nc"

                if not os.path.isdir(save_path):
                    os.makedirs(save_path)

                weights.reset_index("ref").to_netcdf(save_path / save_name)

        return weights


def _unwrap_data_variables(dataset: xr.Dataset):

    return xr.concat(
        [dataset[v].expand_dims("channels", axis=0) for v in list(dataset.data_vars)],
        dim="channels",
    )


def _load_xarray_data(
    paths: list[str],
    selection: dict | None = None,
    names: list | None = None,
    ensemble_mean=False,
    preprocessor: PreprocessingPipeline | None = None,
    concat_dim="year",
    rename_dict: dict = None,
):

    ds = xr.open_mfdataset(paths, combine="nested", concat_dim=concat_dim)

    if rename_dict is not None:
        ds = ds.rename(rename_dict)

    ds = ds.sel(selection) if selection is not None else ds

    if "ensembles" in ds.coords:
        ds = ds.mean("ensembles") if ensemble_mean else ds

    if names is not None:
        ds = ds[names]
    if preprocessor is not None:
        ds = preprocessor.transform(ds)

    return ds


def _create_train_mask(
    years: list | xr.DataArray, lead_times: list | xr.DataArray | int, exclude_idx=0
):

    if isinstance(lead_times, int):
        lead_times = np.arange(1, lead_times + 1)

    mask = np.full((len(years), len(lead_times)), False, dtype=bool)
    x = np.arange(0, 12 * mask.shape[0], 12)
    y = np.arange(1, mask.shape[1] + 1)
    idx_array = x[..., None] + y

    mask[idx_array > idx_array[-1, exclude_idx + 11]] = True

    return xr.DataArray(
        mask,
        dims=("year", "lead_time"),
        coords={"year": years, "lead_time": lead_times},
        name="mask",
    )
