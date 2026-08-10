import numpy as np
import xarray as xr
from pathlib import Path
import joblib
from typing import ClassVar, Literal

from cccma_ppp.preprocessing.selector import PreprocessingStepSelector
from cccma_ppp.preprocessing.preprocessing_ABC import PreprocessModuleABC
from cccma_ppp.generic.runtime import RuntimeContext
from cccma_ppp.data_modules.utils import add_lead_times
from cccma_ppp.configs import (supported_NN_dimensions_sorted, 
                               required_sample_dimensions,
                               lead_time_unit)

init_time_dim, lead_time_dim = required_sample_dimensions
TemporalFrequency = Literal["year", "month", "day"]

@PreprocessingStepSelector.register("normalizer")
class Normalizer(PreprocessModuleABC):
    """
    Min-max normalization preprocessor.

    Parameters
    ----------
    dims : list[str] or None, optional
        Dimensions over which normalization statistics are computed.

    frequency : {"year", "month", "day"} or None, optional
        Valid-time frequency used to group statistics.

        - ``None``: no temporal grouping.
        - ``"year"``: group by valid year.
        - ``"month"``: group by valid month.
        - ``"day"``: group by valid day of year.

    """

    def __init__(
        self,
        dims: list[str] | None = None,
        frequency: TemporalFrequency | None = None,
        **kwargs,
    ) -> None:

        self.min: xr.DataArray | xr.Dataset | None = None
        self.max: xr.DataArray | xr.Dataset | None = None

        self.dims = tuple(dims) if dims is not None else None
        self.frequency = frequency

        self.large_ensemble = False
        self.fitted = False

        if self.frequency not in self.supported_frequencies:
            raise ValueError(
                f"Unsupported frequency {self.frequency!r}. "
                f"Expected one of {self.supported_frequencies}."
            )

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

        reduction_dims = self._get_reduction_dims(data)

        if self.frequency is None:
            self.min = data.min(reduction_dims).load()
            self.max = data.max(reduction_dims).load()

        else:
            grouped_data = self._add_grouping_coordinate(data)

            self.min = (
                grouped_data
                .groupby(self.frequency)
                .min(dim=reduction_dims)
                .load()
            )

            self.max = (
                grouped_data
                .groupby(self.frequency)
                .max(dim=reduction_dims)
                .load()
            )

        self.fitted = True
        return self

    def transform(self, data: xr.DataArray):
        """
        Apply min-max normalization.

        If the temporal auxiliary coordinate already exists on ``data``,
        it is used directly. Otherwise, the coordinate is derived from
        ``init_time``.

        Parameters
        ----------
        data : xr.DataArray

        Returns
        -------
        xr.DataArray
            Normalized data.
        """
        self._check_fitted()

        minimum = self._align_stat_for_transform(
            data=data,
            stat=self.min,
        )

        maximum = self._align_stat_for_transform(
            data=data,
            stat=self.max,
        )

        return (data - minimum) / (maximum - minimum)

    def inverse_transform(self, data: xr.DataArray):
        """
        Reverse normalization.

        Temporal statistics will be aligned using valid forecast times, 
        if the data has lead_time and stats has init_time but no lead_time so
        lead time is taken into account.

        Parameters
        ----------
        data : xr.DataArray
            Input data in normalized space.

        Returns
        -------
        xr.DataArray
            Data in original scale.
        """
        self._check_fitted()

        minimum = align_stat_data_lead_time_inverse_transform(
            ds=data,
            stat=self.min,
            lead_time_resolution=self.lead_time_resolution,
        )

        maximum = align_stat_data_lead_time_inverse_transform(
            ds=data,
            stat=self.max,
            lead_time_resolution=self.lead_time_resolution,
        )

        return data * (maximum - minimum) + minimum


@PreprocessingStepSelector.register("standardizer")
class Standardizer(PreprocessModuleABC):
    """
    Standardization preprocessor.

    Parameters
    ----------
    dims : list of str or None, optional
        Dimensions along which mean and std are computed.

    frequency : {"year", "month", "day"} or None, optional
        Temporal grouping derived from the initialization-time coordinate.

        - ``None``: no temporal grouping.
        - ``"year"``: group by initialization year.
        - ``"month"``: group by initialization month.
        - ``"day"``: group by initialization day of year.

    """

    def __init__(
        self,
        dims: list[str] | None = None,
        frequency: TemporalFrequency | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize standardizer.

        Parameters
        ----------
        dims : list of str or None, optional

        Returns
        -------
        None
        """
        self.mean: xr.Dataset | xr.DataArray | None = None
        self.std: xr.Dataset | xr.DataArray | None = None

        self.dims = tuple(dims) if dims is not None else None
        self.frequency = frequency

        self.large_ensemble = False
        self.fitted = False

        if self.frequency not in self.supported_frequencies:
            raise ValueError(
                f"Unsupported frequency {self.frequency!r}. "
                f"Expected one of {self.supported_frequencies}."
            )

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

        reduction_dims = self._get_reduction_dims(data)

        if mask is not None:
            data = data.where(~np.isnan(mask))

        if self.frequency is None:
            self.mean = data.mean(reduction_dims).load()
            std = data.std(reduction_dims).load()

        else:
            grouped_data = self._add_grouping_coordinate(data)

            self.mean = (
                grouped_data
                .groupby(self.frequency)
                .mean(dim=reduction_dims)
                .load()
            )

            std = (
                grouped_data
                .groupby(self.frequency)
                .std(dim=reduction_dims)
                .load()
            )

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

        self._check_fitted()

        mean = self._align_stat_for_transform(
            data=data,
            stat=self.mean,
        )

        std = self._align_stat_for_transform(
            data=data,
            stat=self.std,
        )

        return (data - mean) / std

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
        self._check_fitted()

        mean = align_stat_data_lead_time_inverse_transform(
            ds=data,
            stat=self.mean,
            lead_time_resolution=self.lead_time_resolution,
        )

        std = align_stat_data_lead_time_inverse_transform(
            ds=data,
            stat=self.std,
            lead_time_resolution=self.lead_time_resolution,
        )

        return data * std + mean


@PreprocessingStepSelector.register("anomalies")
class AnomaliesScaler(PreprocessModuleABC):
    """
    Anomaly scaling preprocessor.

    Computes anomalies relative to a mean climatology.

    Parameters
    ----------
    dims : list of str or None, optional
        Dimensions used to compute mean.

    frequency : {"year", "month", "day"} or None, optional
        Valid-time frequency used to group statistics.

        - ``None``: no temporal grouping.
        - ``"year"``: group by valid year.
        - ``"month"``: group by valid month.
        - ``"day"``: group by valid day of year.

    """

    def __init__(
        self,
        dims: list[str] | None = None,
        frequency: TemporalFrequency | None = None,
        **kwargs,
    ) -> None:

        self.mean: xr.DataArray | xr.Dataset | None = None

        self.dims = tuple(dims) if dims is not None else None
        self.frequency = frequency

        self.large_ensemble = False
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
        reduction_dims = self._get_reduction_dims(data)

        if mask is not None:
            data = data.where(~np.isnan(mask))

        if self.frequency is None:
            self.mean = data.mean(reduction_dims).load()

        else:
            grouped_data = self._add_grouping_coordinate(data)
            
            self.mean = (
                grouped_data
                .groupby(self.frequency)
                .mean(dim=reduction_dims)
                .load()
            )

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

        self._check_fitted()

        mean = self._align_stat_for_transform(
            data=data,
            stat=self.mean,
        )
       
        return data - mean

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

        self._check_fitted()

        mean = align_stat_data_lead_time_inverse_transform(
            ds=data,
            stat=self.mean,
            lead_time_resolution=self.lead_time_resolution,
        )

        return data + mean


@PreprocessingStepSelector.register("flattener")
class Flattennanremove(PreprocessModuleABC):
    """
    Flatten NN dimensions while removing NaN locations.

    Parameters
    ----------
    load_dir : pathlib.Path or str or None, optional
        Path to a previously fitted preprocessor.
    """
    supported_NN_dimensions_sorted: ClassVar[tuple] = supported_NN_dimensions_sorted

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
            dim for dim in self.supported_NN_dimensions_sorted if dim in reference.dims
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


def align_stat_data_lead_time_inverse_transform(
    ds: xr.DataArray,
    stat: xr.DataArray,
    lead_time_resolution: lead_time_unit = "month",
    init_time_dim: str = init_time_dim,
    lead_time_dim: str = lead_time_dim,
) -> xr.DataArray:
    """
    Align fitted temporal statistics to forecast target times.

    This is primarily needed when a preprocessing pipeline fitted to
    observations is inverse-applied to forecast data.

    Lead times are assumed to be one-based:

    - lead_time=1 corresponds to init_time
    - lead_time=2 corresponds to one period after init_time

    Parameters
    ----------
    ds : xr.DataArray
        Data containing initialization-time and lead-time coordinates.

    stat : xr.DataArray
        Fitted statistic. It may contain:

        - init_time
        - lead_time
        - year
        - month
        - day
        - no temporal dimensions

    lead_time_resolution : {"month", "day"}
        Temporal unit represented by one lead-time increment.

    Returns
    -------
    xr.DataArray
        Statistic aligned to the temporal dimensions of ``ds``.
    """

    # If the statistic already depends on lead time, its temporal
    # structure matches the forecast structure directly.
    if lead_time_dim in stat.dims:
        return stat

    temporal_stat_dims = {
        init_time_dim,
        "year",
        "month",
        "day",
    }

    # Nothing temporal needs to be aligned.
    if temporal_stat_dims.isdisjoint(stat.dims):
        return stat

    if init_time_dim not in ds.coords:
        raise ValueError(
            f"Data must contain the initialization-time coordinate "
            f"{init_time_dim!r}."
        )

    
    init_times = np.asarray(ds[init_time_dim].values)

    if init_times.ndim != 1:
        raise ValueError(
            f"Coordinate {init_time_dim!r} must be one-dimensional."
        )


    if lead_time_dim in ds.dims:
        lead_times = np.asarray(ds[lead_time_dim].values)

        if lead_times.ndim != 1:
            raise ValueError(
                f"Coordinate {lead_time_dim!r} must be one-dimensional."
            )

        init_time_grid, lead_time_grid = np.meshgrid(
            init_times,
            lead_times,
            indexing="ij",
        )

        target_times = add_lead_times(
            init_times=init_time_grid.reshape(-1),
            lead_times=lead_time_grid.reshape(-1),
            lead_time_resolution=lead_time_resolution,
        ).reshape(init_time_grid.shape)

        temporal_dims = (init_time_dim, lead_time_dim)
        temporal_coords = {
            init_time_dim: ds[init_time_dim],
            lead_time_dim: ds[lead_time_dim],
        }

    else:
        # For observations, init_time is already the valid/target time.
        target_times = init_times

        temporal_dims = (init_time_dim,)
        temporal_coords = {
            init_time_dim: ds[init_time_dim],
        }


    target_time = xr.DataArray(
        target_times,
        dims=temporal_dims,
        coords=temporal_coords,
        name="target_time",
    )

    rename_dims = {}
    indexers = {}

    if init_time_dim in stat.dims:
        rename_dims[init_time_dim] = "__stat_time"
        indexers["__stat_time"] = target_time

    if "year" in stat.dims:
        rename_dims["year"] = "__stat_year"
        indexers["__stat_year"] = target_time.dt.year

    if "month" in stat.dims:
        rename_dims["month"] = "__stat_month"
        indexers["__stat_month"] = target_time.dt.month

    if "day" in stat.dims:
        rename_dims["day"] = "__stat_day"
        indexers["__stat_day"] = target_time.dt.dayofyear

    aligned_stat = stat.rename(rename_dims).sel(indexers)

    temporary_coords = [
        coord
        for coord in (
            "__stat_time",
            "__stat_year",
            "__stat_month",
            "__stat_day",
        )
        if coord in aligned_stat.coords
    ]

    if temporary_coords:
        aligned_stat = aligned_stat.drop_vars(temporary_coords)

    return aligned_stat.assign_coords(temporal_coords)