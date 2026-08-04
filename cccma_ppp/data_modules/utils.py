import numpy as np
import xarray as xr
import dataclasses
from pathlib import Path
import os
from typing import Literal
import contextlib
from collections.abc import Iterator

from cccma_ppp.preprocessing.preprocessing import PreprocessingPipeline
from cccma_ppp.preprocessing.utils_preprocessing import Flattennanremove
from cccma_ppp.generic.runtime import RuntimeContext
from cccma_ppp.configs import supported_NN_dimensions_sorted, required_sample_dimensions

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
        target_coords: dict,
        Flattennanremover: Flattennanremove | None = None,
        save=True,
        save_path: Path | str | None = None,
        save_name: str | None = None,
    ):

        if self.load_dir is not None:
            weights = xr.open_dataset(Path(self.load_dir))
            if isinstance(weights, xr.Dataset):
                weights = _unwrap_data_variables(weights)

            msg = f"the loaded weights from {self.load_dir} must have coordinates that match the target coordinates"

            for coord in target_coords:
                if coord not in weights.coords:
                    raise ValueError(msg)
                if not weights.coords[coord].equals(target_coords[coord]):
                    raise ValueError(msg)

        else:
            coords = xr.DataArray(dims=tuple(target_coords), coords=target_coords)

            dims = tuple(coords.sizes)
            shape = tuple(coords.sizes[dim] for dim in dims)

            weights = xr.DataArray(
                np.ones(shape, dtype=np.float32),
                dims=dims,
                coords=coords.coords,
                name="weights",
            )

            if self.spatial_method == "cosine_lat" and "lat" in weights.coords:
                latitude_weights = np.cos(np.deg2rad(weights.coords["lat"]))
                weights = weights * latitude_weights

            if self.variable_weights is not None:
                variable_weights = xr.DataArray(
                    list(self.variable_weights.values()),
                    dims=("channels",),
                    coords={"channels": list(self.variable_weights)},
                    name="variable_weights",
                )

                weights = variable_weights * weights

        if self.load_dir is None and save:
            save_path = (
                Path(save_path)
                if save_path is not None
                else Path(RuntimeContext.GLOBAL_EXP_DIR)
            )
            save_name = save_name or "spatial_weights.nc"

            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            weights.to_netcdf(save_path / save_name)

        if Flattennanremover is not None:
            weights = Flattennanremover.transform(weights)

        return weights


def _unwrap_data_variables(dataset: xr.Dataset) -> xr.DataArray:

    return xr.concat(
        [
            dataset[v].squeeze().expand_dims("channels", axis=0)
            for v in list(dataset.data_vars)
        ],
        dim="channels",
    )


def _load_xarray_data(
    paths: list[str],
    selection: dict | None = None,
    names: list | None = None,
    ensemble_mean: bool = False,
    preprocessor: PreprocessingPipeline | None = None,
    concat_dim: str = "year",
    rename_dict: dict | None = None,
    chunks: dict | None = None,
    load: bool = False,
):

    ds = xr.open_mfdataset(
        paths, combine="nested", concat_dim=concat_dim, chunks=chunks
    )

    if rename_dict is not None:
        ds = ds.rename(rename_dict)

    ds = ds.sel(selection) if selection is not None else ds

    if "ensembles" in ds.coords:
        ds = ds.mean("ensembles") if ensemble_mean else ds

    if names is not None:
        ds = ds[names]
    if preprocessor is not None:
        ds = preprocessor.transform(ds)

    nn_dims = [dim for dim in supported_NN_dimensions_sorted if dim in ds.dims]

    ds = ds.transpose(..., *nn_dims)
    if load:
        ds = ds.load()

    return ds


def _create_train_mask(
    time: list | xr.DataArray,
    lead_times: list | xr.DataArray | np.ndarray | int,
    exclude_idx=0,
):

    if not isinstance(lead_times, int):
        lead_times = max(lead_times)

    lead_times = np.arange(1, max(lead_times, 12) + 1)

    mask = np.full((len(time), len(lead_times)), False, dtype=bool)
    x = np.arange(0, 12 * mask.shape[0], 12)
    y = np.arange(1, mask.shape[1] + 1)
    idx_array = x[..., None] + y

    mask[idx_array > idx_array[-1, exclude_idx + 11]] = True

    time_dim, lead_time_dim = required_sample_dimensions

    return xr.DataArray(
        mask,
        dims=required_sample_dimensions,
        coords={time_dim: time, lead_time_dim: lead_times},
        name="mask",
    )


@contextlib.contextmanager
def suppress_stderr() -> Iterator[None]:
    stderr_fd = 2
    saved_stderr_fd = os.dup(stderr_fd)

    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stderr_fd)
