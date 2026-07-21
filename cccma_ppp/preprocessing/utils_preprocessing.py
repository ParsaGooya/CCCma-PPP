import numpy as np
import xarray as xr
from pathlib import Path
import joblib
import os

from cccma_ppp.preprocessing.selector import PreprocessingStepSelector
from cccma_ppp.preprocessing.preprocessing_ABC import PreprocessModuleABC
from cccma_ppp.generic.runtime import RuntimeContext
from cccma_ppp.configs import supported_NN_dimensions_sorted


@PreprocessingStepSelector.register("normalizer")
class Normalizer(PreprocessModuleABC):
    """
    Min-max normalization preprocessor.

    Parameters
    ----------
    dims : list of str or None, optional
        Dimensions along which normalization statistics are computed.
    """

    def __init__(self, dims: list | None = None, **kwargs) -> None:
        """
        Initialize normalizer.

        Parameters
        ----------
        dims : list of str or None, optional
            Dimensions over which min and max are computed.

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

    def fit(self, data: xr.Dataset | xr.DataArray, mask: xr.DataArray = None):
        """
        Fit normalization parameters.

        Parameters
        ----------
        data : xr.DataArray
            Input data.
        mask : xr.DataArray or None, optional
            Mask specifying valid data.

        Returns
        -------
        self
        """

        if all(["ensembles" in data.dims, self.dims is not None]):
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
        """
        Apply min-max normalization.

        Parameters
        ----------
        data : xr.DataArray

        Returns
        -------
        xr.DataArray
            Normalized data.
        """

        data_normalized = (data - self.min) / (self.max - self.min)
        return data_normalized

    def inverse_transform(self, data: xr.DataArray):
        """
        Reverse normalization.

        Parameters
        ----------
        data : xr.DataArray
            Input data in normalized space.

        Returns
        -------
        xr.DataArray
            Data in original scale.
        """

        data_raw = data * (self.max - self.min) + self.min
        return data_raw


@PreprocessingStepSelector.register("standardizer")
class Standardizer(PreprocessModuleABC):
    """
    Standardization preprocessor.

    Parameters
    ----------
    dims : list of str or None, optional
        Dimensions along which mean and std are computed.
    """

    def __init__(self, dims: list | None = None, **kwargs) -> None:
        """
        Initialize standardizer.

        Parameters
        ----------
        dims : list of str or None, optional

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

    def fit(self, data: xr.Dataset | xr.DataArray, mask: xr.DataArray = None):
        """
        Fit standardization parameters.

        Parameters
        ----------
        data : xr.DataArray
        mask : xr.DataArray or None, optional

        Returns
        -------
        self
        """

        if all(["ensembles" in data.dims, self.dims is not None]):
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
        """
        Apply standardization.

        Parameters
        ----------
        data : xr.DataArray

        Returns
        -------
        xr.DataArray
            Standardized data.
        """

        data_standardized = (data - self.mean) / self.std

        return data_standardized

    def inverse_transform(self, data: xr.DataArray):
        """
        Reverse standardization.

        Parameters
        ----------
        data : xr.DataArray

        Returns
        -------
        xr.DataArray
            Original scale data.
        """

        data_raw = data * self.std + self.mean

        return data_raw


@PreprocessingStepSelector.register("anomalies")
class AnomaliesScaler(PreprocessModuleABC):
    """
    Anomaly scaling preprocessor.

    Computes anomalies relative to a mean climatology.

    Parameters
    ----------
    dims : list of str or None, optional
        Dimensions used to compute mean.
    """

    def __init__(self, dims: list | None = None, **kwargs) -> None:
        """
        Initialize anomaly scaler.

        Parameters
        ----------
        dims : list of str or None, optional

        Returns
        -------
        None
        """

        self.mean = None
        self.dims = dims
        self.fitted = False

        if self.dims is not None:
            self.dims = tuple(self.dims)

    def fit(self, data: xr.Dataset | xr.DataArray, mask: xr.DataArray = None):
        """
        Fit anomaly baseline.

        Parameters
        ----------
        data : xr.DataArray
        mask : xr.DataArray or None, optional

        Returns
        -------
        self
        """

        if all(["ensembles" in data.dims, self.dims is not None]):
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
        """
        Compute anomalies.

        Parameters
        ----------
        data : xr.DataArray

        Returns
        -------
        xr.DataArray
            Anomaly values.
        """

        data_anomalies = data - self.mean
        return data_anomalies

    def inverse_transform(self, data: xr.DataArray):
        """
        Reconstruct original values from anomalies.

        Parameters
        ----------
        data : xr.DataArray

        Returns
        -------
        xr.DataArray
            Reconstructed data.
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


@PreprocessingStepSelector.register("flattener")
class Flattennanremove(PreprocessModuleABC):
    """
    Flatten NN dimensions while removing NaN locations.

    Parameters
    ----------
    load_dir : pathlib.Path or str or None, optional
        Path to a previously fitted preprocessor.
    """

    def __init__(self, load_dir: Path | str = None, **kwargs):
        """
        Initialize flattener.

        Parameters
        ----------
        load_dir : pathlib.Path or str or None, optional
            Path to load a pre-fitted instance.

        Returns
        -------
        None
        """

        self.load_dir = load_dir
        self.fitted = False
        self.common_to_input_and_target = False
        self.NN_dims: list[str] = []

    def fit(
        self,
        data: xr.Dataset | xr.DataArray,
        target: xr.DataArray | None = None,
        mask=None,
        save: bool = False,
        save_name: str | None = None,
        save_path: Path | str = None,
    ):
        """
        Fit spatial flattening transformation.

        Determines valid spatial locations and optionally aligns them
        between input and target datasets.

        Parameters
        ----------
        data : xr.DataArray
            Input data.
        target : xr.DataArray or None, optional
            Target data for alignment.
        mask : xr.DataArray or None, optional
            Optional mask (unused in current implementation).
        save : bool, optional
            Whether to save fitted preprocessor.
        save_name : str or None, optional
            Name of saved file.
        save_path : pathlib.Path or str or None, optional
            Directory to save object.

        Returns
        -------
        self
        """

        if self.load_dir is not None:
            self._load_from_memory(self.load_dir)

            self._check_nn_dims(data)
            self._check_nn_dims(target)
            return self

        reference = target if target is not None else data

        self.NN_dims = [
            dim for dim in supported_NN_dimensions_sorted if dim in reference.dims
        ]

        missing_from_data = [dim for dim in self.NN_dims if dim not in data.dims]

        if missing_from_data:
            raise RuntimeError(
                "The input and reference data do not share all required NN "
                f"dimensions. Missing from input data: {missing_from_data}."
            )

        self.reference_shape = xr.Dataset(
            coords={dim: reference[dim] for dim in self.NN_dims}
        )

        data_stacked = data.stack(ref=self.NN_dims).dropna(
            dim="ref",
            how="any",
        )

        if target is not None:
            target_stacked = target.stack(ref=self.NN_dims).dropna(
                dim="ref",
                how="any",
            )

            self.final_locations = (
                target_stacked.sel(ref=data_stacked["ref"])
                .dropna(dim="ref", how="any")
                .load()["ref"]
            )

            self.common_to_input_and_target = True

        else:
            self.final_locations = data_stacked["ref"].load()
            self.common_to_input_and_target = False

        self.fitted = True

        if save:
            save_name = save_name or "flattener"

            resolved_save_path = (
                Path(save_path)
                if save_path is not None
                else Path(RuntimeContext.GLOBAL_EXP_DIR)
            )
            resolved_save_path.mkdir(parents=True, exist_ok=True)

            joblib.dump(
                self,
                resolved_save_path / f"{save_name}.joblib",
            )

        return self

    def _check_nn_dims(
        self,
        data: xr.Dataset | xr.DataArray,
    ):
        if data is not None:
            missing_dims = [dim for dim in self.NN_dims if dim not in data.dims]

            if missing_dims:
                raise ValueError(
                    "The saved preprocessor and data pipelines are not compatable. "
                    f"Missing dimensions: {missing_dims}."
                )

    def transform(self, data: xr.DataArray) -> xr.Dataset | xr.DataArray:
        """
        Apply flattening and spatial filtering.

        Parameters
        ----------
        data : xr.DataArray

        Returns
        -------
        xr.DataArray
            Flattened data with only valid spatial locations.

        Raises
        ------
        ValueError
            The data to be transformed does not have the correct NN dims.
        """

        if "ref" in data.dims:
            return data.sel(ref=self.final_locations)

        return (data.stack(ref=self.NN_dims).sel(ref=self.final_locations)).transpose(
            ..., "ref"
        )

    def inverse_transform(self, data: xr.DataArray) -> xr.DataArray:
        """
        Restore original spatial layout.

        Parameters
        ----------
        data : xr.DataArray
            Transformed data.

        Returns
        -------
        xr.DataArray
            Reconstructed data in the original spatial grid.

        Raises
        ------
        ValueError
            The data to be inverse transformed does not have the ref dims.
        """

        if "ref" not in data.dims:
            raise ValueError("The input must contain the flattened 'ref' dimension.")

        return data.unstack().combine_first(self.reference_shape)

    def _load_from_memory(self, load_dir: Path | str) -> None:
        """
        Load fitted preprocessor from disk.

        Parameters
        ----------
        load_dir : pathlib.Path or str
            Directory containing the saved preprocessor.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If the loaded preprocessor is not fitted.
        """

        loaded = joblib.load(Path(load_dir))
        if not loaded.fitted:
            raise RuntimeError("the preprocessor to be loaded has to be fitted first.")

        self.reference_shape = loaded.reference_shape
        self.final_locations = loaded.final_locations
        self.common_to_input_and_target = loaded.common_to_input_and_target
        self.fitted = loaded.fitted
        del loaded
