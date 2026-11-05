import numpy as np
import xarray as xr
import dataclasses
from pathlib import Path
import os
from typing import Literal


from S2D4P.preprocessing.utils_preprocessing import Flattennanremove
from S2D4P.generic.runtime import RuntimeContext
from S2D4P.data.utils import _unwrap_data_variables


Spatialmethod = Literal["uniform", "cosine_lat"]


@dataclasses.dataclass
class WeightsConfig:
    """
    Configuration for spatial and variable loss weights.

    Weights can be generated from target coordinates or loaded from an existing
    NetCDF file. Spatial weighting can be uniform or scaled by the cosine of
    latitude, and optional variable-specific weights can be applied along the
    channel dimension.

    Parameters
    ----------
    spatial_method : Spatialmethod
        Spatial weighting method. Supported values are ``"uniform"`` and
        ``"cosine_lat"``.
    variable_weights : dict[str, float] | None
        Mapping from target variable names to multiplicative weights. If
        provided, entries must match the target channels exactly.
    load_dir : Path | str | None
        Path to a previously saved weights file. If ``None``, weights are
        generated from the target coordinates.
    """

    spatial_method: Spatialmethod = "uniform"
    variable_weights: dict[str, float] | None = None
    load_dir: Path | str | None = None

    def __post_init__(self):
        """
        Validate the weights configuration.

        Raises
        ------
        FileNotFoundError
            If ``load_dir`` is provided but the weights file does not exist.
        ValueError
            If ``spatial_method`` is not ``"uniform"`` or ``"cosine_lat"``.
        """
        if self.load_dir is not None:
            if not Path(self.load_dir).exists():
                raise FileNotFoundError(f"weights file not found at {self.load_dir}")

        valid_methods = {"uniform", "cosine_lat"}
        if self.spatial_method not in valid_methods:
            raise ValueError(
                f"Invalid spatial_method: {self.spatial_method!r}. "
                f"Must be one of {sorted(valid_methods)}."
            )

    def build_weights(
        self,
        target_coords: dict,
        Flattennanremover: Flattennanremove | None = None,
        save=True,
        save_path: Path | str | None = None,
        save_name: str | None = None,
    ):
        """
        Build or load weights for target variables and spatial coordinates.

        When ``load_dir`` is configured, stored weights are loaded and validated
        against the target coordinates. Otherwise, weights are initialized uniformly
        and optional cosine-latitude and variable-specific weighting are applied.
        Generated weights can be saved to disk and optionally transformed to a fitted
        flattened representation.

        Parameters
        ----------
        target_coords : dict
            Mapping from target dimension names to their coordinate values.
        Flattennanremover : Flattennanremove | None
            Optional fitted flattening preprocessor applied to the final weights.
        save : bool
            Whether newly generated weights are saved to disk.
        save_path : Path | str | None
            Directory in which generated weights are saved. If ``None``, the current
            experiment directory is used.
        save_name : str | None
            Filename used when saving generated weights. If ``None``,
            ``"spatial_weights.nc"`` is used.

        Returns
        -------
        xr.DataArray
            Weight array aligned with the target coordinates and optionally
            transformed to the flattened target representation.

        Raises
        ------
        ValueError
            If loaded weights do not match the target coordinates, cosine-latitude
            weighting is requested without a latitude coordinate, variable weights do
            not exactly match the target channels, or the resulting weights contain
            negative, missing, or infinite values.
        """
        if self.load_dir is not None:
            weights = xr.open_dataset(Path(self.load_dir))
            if isinstance(weights, xr.Dataset):
                weights = _unwrap_data_variables(weights)

            weights.load()
            weights.close()

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

            if self.spatial_method == "cosine_lat":
                if "lat" not in weights.coords:
                    raise ValueError(
                        "Cosine-latitude weighting requires a 'lat' coordinate. "
                        f"Available coordinates: {list(weights.coords.keys())}"
                    )

                latitude_weights = np.cos(np.deg2rad(weights.coords["lat"]))
                weights = weights * latitude_weights

            if self.variable_weights is not None:
                if "channels" in target_coords:
                    expected_channels = set(target_coords["channels"].values)
                    provided_channels = set(self.variable_weights.keys())

                    if expected_channels != provided_channels:
                        missing = sorted(expected_channels - provided_channels)
                        unexpected = sorted(provided_channels - expected_channels)
                        msg_parts = []
                        if missing:
                            msg_parts.append(f"Missing weights for: {missing}")
                        if unexpected:
                            msg_parts.append(f"Unexpected weights for: {unexpected}")
                        raise ValueError(
                            "Variable weights must match target channels exactly. "
                            + "; ".join(msg_parts)
                        )

                variable_weights = xr.DataArray(
                    list(self.variable_weights.values()),
                    dims=("channels",),
                    coords={"channels": list(self.variable_weights)},
                    name="variable_weights",
                )

                weights = variable_weights * weights

        if np.any(weights < 0):
            raise ValueError(
                f"Weights must be non-negative, found minimum {weights.min()}"
            )
        if np.any(np.isnan(weights)):
            raise ValueError("Weights contain NaN values")
        if np.any(np.isinf(weights)):
            raise ValueError("Weights contain infinite values")

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
