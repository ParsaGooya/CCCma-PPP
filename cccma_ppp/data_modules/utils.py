import numpy as np
import xarray as xr
import dataclasses
from pathlib import Path
import os
from typing import Literal

from cccma_ppp.preprocessing import PreprocessingPipeline, Flattennanremove
from cccma_ppp.generic import RuntimeContext


spatialmethod = Literal["uniform", "cosine_lat"]


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
        Flattennanremover: Flattennanremove | None = None,
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

            if Flattennanremover is not None:
                if not weights.coords["ref"].equals(Flattennanremover.final_locations):
                    raise ValueError(msg)
            elif target_coords is not None:
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

            if Flattennanremover is not None:
                weights = Flattennanremover.transform(weights)

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
