import numpy as np
import xarray as xr
import dataclasses
from pathlib import Path
import os
from typing import Literal


from cccma_ppp.preprocessing.utils_preprocessing import Flattennanremove
from cccma_ppp.generic.runtime import RuntimeContext
from cccma_ppp.data_modules.utils import _unwrap_data_variables
                    

spatialmethod = Literal["uniform", "cosine_lat"]

@dataclasses.dataclass
class WeightsConfig:
    """
    Configuration for computing spatial and variable weights.

    Parameters
    ----------
    spatial_method : {"uniform", "cosine_lat"}, optional
        Method used to compute spatial weights.
    variable_weights : dict[str, float] or None, optional
        Per-variable weighting factors.
    load_dir : pathlib.Path or str or None, optional
        Path to load precomputed weights.
    """

    spatial_method: spatialmethod = "uniform"
    variable_weights: dict[str, float] | None = None
    load_dir: Path | str | None = None

    def __post_init__(self):
        """
        Validate weight configuration.

        Returns
        -------
        None

        Raises
        ------
        FileNotFoundError
            If specified load path does not exist.
        """

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
        """
        Generate or load spatial weights.

        Parameters
        ----------
        target_coords : dict
            Spatial coordinates of target data.
        Flattennanremover : Flattennanremove or None, optional
            Preprocessor for flattened spatial representation.
        save : bool, optional
            Whether to save computed weights.
        save_path : pathlib.Path or str or None, optional
        save_name : str or None, optional

        Returns
        -------
        xr.DataArray
            Spatial (and optional variable) weights.

        Raises
        ------
        ValueError
            If loaded weights are incompatible with target coordinates.
        """

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
