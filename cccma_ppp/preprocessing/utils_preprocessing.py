import numpy as np
import xarray as xr
from pathlib import Path
import joblib
import os

from cccma_ppp.preprocessing.preprocessing import (
    PreprocessingStepSelector,
)  # PreprocessingPipeline
from cccma_ppp.preprocessing.preprocessing_ABC import PreprocessModuleABC


@PreprocessingStepSelector.register("normalizer")
class Normalizer(PreprocessModuleABC):
    """
    Normalize data using min-max scaling along specified dimensions.
    """

    def __init__(self, dims=None, **kwargs):
        """
        Initialize Normalizer.

        Parameters
        ----------
        dims : list, optional
            Dimensions along which to compute min and max.

        Returns
        -------
        None
        """

        self.min = None
        self.max = None
        self.dims = dims
        self.fitted = False

        if self.dims is not None:
            self.dims = tuple(self.dims)

    def fit(self, data, mask=None):
        """
        Compute min and max values from data.

        Parameters
        ----------
        data : xr.DataArray
            Input dataset.
        mask : xr.DataArray, optional
            Mask indicating valid values.

        Returns
        -------
        Normalizer
            Fitted instance.
        """

        if all(
            ["ensembles" in data.dims, self.dims is not None]
        ):  ## PG: if ensemble exists in the dimentions. Note that we always pass a map like data to this function. Even if it is flattened, we first write back to maps.
            if "ensembles" not in self.dims:
                self.large_ensemble = True
                self.dims = (
                    "ensembles",
                    *self.dims,
                )  ## PG: Tell the object to average over both years and ensembles for calculating anomalies.

        if mask is not None:
            data_masked = data.where(~np.isnan(mask))
        else:
            data_masked = data

        self.min = data_masked.min(self.dims).load()
        self.max = data_masked.max(self.dims).load()
        self.fitted = True
        return self

    def transform(self, data):
        """
        Apply min-max normalization.

        Parameters
        ----------
        data : xr.DataArray
            Input data.

        Returns
        -------
        xr.DataArray
            Normalized data.
        """

        data_normalized = (data - self.min) / (self.max - self.min)
        return data_normalized

    def inverse_transform(self, data):
        """
        Reverse min-max normalization.

        Parameters
        ----------
        data : xr.DataArray
            Normalized data.

        Returns
        -------
        xr.DataArray
            Original-scale data.
        """

        data_raw = data * (self.max - self.min) + self.min
        return data_raw


@PreprocessingStepSelector.register("standardizer")
class Standardizer(PreprocessModuleABC):
    """
    Standardize data using mean and standard deviation scaling.
    """

    def __init__(self, dims=None, **kwargs):
        """
        Initialize Standardizer.

        Parameters
        ----------
        dims : list, optional
            Dimensions along which to compute statistics.

        Returns
        -------
        None
        """

        self.mean = None
        self.std = None
        self.dims = dims
        self.fitted = False

        if self.dims is not None:
            self.dims = tuple(self.dims)

    def fit(self, data, mask=None):
        """
        Compute mean and standard deviation.

        Parameters
        ----------
        data : xr.DataArray
            Input dataset.
        mask : xr.DataArray, optional
            Mask indicating valid values.

        Returns
        -------
        Standardizer
            Fitted instance.
        """

        if all(
            ["ensembles" in data.dims, self.dims is not None]
        ):  ## PG: if ensemble exists in the dimentions. Note that we always pass a map like data to this function. Even if it is flattened, we first write back to maps.
            if "ensembles" not in self.dims:
                self.large_ensemble = True
                self.dims = (
                    "ensembles",
                    *self.dims,
                )  ## PG: Tell the object to average over both years and ensembles for calculating anomalies.

        if mask is not None:
            data_masked = data.where(~np.isnan(mask))
        else:
            data_masked = data

        self.mean = data_masked.mean(self.dims).load()
        std = data_masked.std(self.dims).load()
        self.std = std.where(std > 0)
        self.fitted = True
        return self

    def transform(self, data):
        """
        Apply standardization.

        Parameters
        ----------
        data : xr.DataArray
            Input data.

        Returns
        -------
        xr.DataArray
            Standardized data.
        """

        data_standardized = (data - self.mean) / self.std

        return data_standardized

    def inverse_transform(self, data):
        """
        Reverse standardization.

        Parameters
        ----------
        data : xr.DataArray
            Standardized data.

        Returns
        -------
        xr.DataArray
            Original-scale data.
        """

        data_raw = data * self.std + self.mean

        return data_raw


@PreprocessingStepSelector.register("anomalies")
class AnomaliesScaler(PreprocessModuleABC):
    """
    Compute anomalies by removing mean along specified dimensions.
    """

    def __init__(self, dims=None, **kwargs):
        """
        Initialize AnomaliesScaler.

        Parameters
        ----------
        dims : list, optional
            Dimensions along which to compute mean.

        Returns
        -------
        None
        """

        self.mean = None
        self.dims = dims
        self.fitted = False

        if self.dims is not None:
            self.dims = tuple(self.dims)

    def fit(self, data, mask=None):
        """
        Compute mean for anomaly calculation.

        Parameters
        ----------
        data : xr.DataArray
            Input dataset.
        mask : xr.DataArray, optional
            Mask indicating valid values.

        Returns
        -------
        AnomaliesScaler
            Fitted instance.
        """

        if all(
            ["ensembles" in data.dims, self.dims is not None]
        ):  ## PG: if ensemble exists in the dimentions. Note that we always pass a map like data to this function. Even if it is flattened, we first write back to maps.
            if "ensembles" not in self.dims:
                self.large_ensemble = True
                self.dims = (
                    "ensembles",
                    *self.dims,
                )  ## PG: Tell the object to average over both years and ensembles for calculating anomalies.

        if mask is not None:
            data_masked = data.where(~np.isnan(mask))
        else:
            data_masked = data  # PG

        self.mean = data_masked.mean(self.dims).load()
        self.fitted = True
        return self

    def transform(self, data):
        """
        Compute anomalies.

        Parameters
        ----------
        data : xr.DataArray
            Input data.

        Returns
        -------
        xr.DataArray
            Anomaly data.
        """

        data_anomalies = data - self.mean
        return data_anomalies

    def inverse_transform(self, data):
        """
        Reconstruct original data by adding mean back.

        Parameters
        ----------
        data : xr.DataArray
            Anomaly data.

        Returns
        -------
        xr.DataArray
            Original data.
        """

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
    """
    Remove spatial locations corresponding to NaN (e.g., ocean points) and flatten spatial dimensions.
    """

    ## PG

    def __init__(self, load_dir=None, **kwargs):
        """
        Initialize Oceannanremove.

        Parameters
        ----------
        load_dir : Path or str, optional
            Directory to load fitted preprocessor from.

        Returns
        -------
        None
        """

        self.load_dir = load_dir
        self.fitted = False

    def fit(
        self, data, target=None, mask=None, save=False, save_name=None, save_path=None
    ):
        """
        Identify valid spatial locations and optionally save configuration.

        Parameters
        ----------
        data : xr.DataArray
            Input dataset.
        target : xr.DataArray, optional
            Target dataset for intersection of valid points.
        mask : xr.DataArray, optional
            Mask for valid values.
        save : bool, optional
            Whether to save fitted preprocessor.
        save_name : str, optional
            Filename for saving.
        save_path : Path or str, optional
            Directory to save file.

        Returns
        -------
        Oceannanremove
            Fitted instance.
        """
        ## PG: extract common grid points based on trainig and target data

        if self.load_dir is None:
            if target is not None:
                self.reference_shape = xr.Dataset(
                    coords={"lat": target["lat"], "lon": target["lon"]}
                )

                temp = target.stack(
                    ref=["lat", "lon"]
                ).sel(
                    ref=data.stack(ref=["lat", "lon"]).dropna(dim="ref").ref
                )  ## PG: flatten target in space and choose space points where data is not NaN.
                self.final_locations = (
                    temp.dropna("ref").load().ref
                )  ## PG: Extract locations common to target and training data by dropping the remaining NaN values
                self.common_to_input_and_target = True
            else:
                self.reference_shape = xr.Dataset(
                    coords={"lat": data["lat"], "lon": data["lon"]}
                )

                self.final_locations = (
                    data.stack(ref=["lat", "lon"]).dropna(dim="ref").ref.load()
                )  ## PG: flatten target in space and choose space points where data is not NaN.
                self.common_to_input_and_target = False

            self.fitted = True
            if save:
                save_name = save_name or "oceannanremover"
                save_path = Path(save_path) or Path(os.get["GLOBAL_EXP_DIR"])

                joblib.dump(self, save_path.joinpath(f"{save_name}.joblib"))
        else:
            self._load_from_memory(self.load_dir)

        return self

    def transform(self, data):
        """
        Extract valid spatial locations.

        Parameters
        ----------
        data : xr.DataArray
            Input dataset.

        Returns
        -------
        xr.DataArray
            Flattened data with selected locations.
        """
        ## PG: Pass a DataArray and sample at the extracted locations

        conditions = ["lat" in data.dims, "lon" in data.dims]

        if all(conditions):  ## PG: if a map get passeed
            sampled = data.stack(ref=["lat", "lon"]).sel(ref=self.final_locations)
        else:  ## PG: If a flattened dataset is passed (in space)
            sampled = data.sel(ref=self.final_locations)

        return sampled

    def inverse_transform(self, data):
        """
        Reconstruct spatial map from flattened representation.

        Parameters
        ----------
        data : xr.DataArray
            Flattened data.

        Returns
        -------
        xr.DataArray
            Reconstructed spatial dataset.
        """
        ## PG: Write back the flattened data to maps

        return data.unstack().combine_first(
            self.reference_shape
        )  ## Unstack the flattened spatial dim and write back to the initial format as saved in self.reference_shape using NaN as fill value

    def _load_from_memory(self, load_dir):
        """
        Load preprocessor state from disk.

        Parameters
        ----------
        load_dir : Path or str
            Path to saved preprocessor.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If loaded preprocessor is not fitted.
        """

        loaded = joblib.load(Path(load_dir))
        assert loaded.fitted, "the preprocessor to be loaded has to be fitted first."

        self.reference_shape = loaded.reference_shape
        self.final_locations = loaded.final_locations
        self.fitted = loaded.fitted
        del loaded
