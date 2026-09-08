import warnings
warnings.filterwarnings('ignore')

import numpy as np
import xarray as xr
from pathlib import Path
import joblib
import datetime
import cftime
from typing import ClassVar, Literal

from cccma_ppp.preprocessing.selector import PreprocessingStepSelector
from cccma_ppp.preprocessing.preprocessing_ABC import PreprocessModuleABC
from cccma_ppp.generic.runtime import RuntimeContext
from cccma_ppp.data_modules.utils import add_lead_times
from cccma_ppp.configs import (
    supported_NN_dimensions_sorted,
    required_sample_dimensions,
    lead_time_unit,
)

init_time_dim, lead_time_dim = required_sample_dimensions
TemporalFrequency = Literal["year", "month", "day"]


@PreprocessingStepSelector.register("normalizer")
class Normalizer(PreprocessModuleABC):
    """
    Document this class.

    Parameters
    ----------
    dims : list[str] | None
        Description not yet provided.
    frequency : TemporalFrequency | None
        Description not yet provided.
    **kwargs : Any
        Description not yet provided.
    """

    def __init__(
        self,
        dims: list[str] | None = None,
        frequency: TemporalFrequency | None = None,
        **kwargs,
    ) -> None:
        """
        Document this function.

        Parameters
        ----------
        dims : list[str] | None
            Description not yet provided.
        frequency : TemporalFrequency | None
            Description not yet provided.
        **kwargs : Any
            Description not yet provided.

        Raises
        ------
        ValueError
            Description not yet provided.
        """
        self.min: xr.DataArray | xr.Dataset | None = None
        self.max: xr.DataArray | xr.Dataset | None = None

        self.dims = tuple(dims) if dims is not None else tuple()
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
        Document this function.

        Parameters
        ----------
        data : xr.Dataset | xr.DataArray
            Description not yet provided.
        mask : xr.DataArray
            Description not yet provided.

        Returns
        -------
        Any
            Description not yet provided.
        """
        reduction_dims = self._get_reduction_dims(data)

        if mask is not None:
            data = data.where(~mask)

        if self.frequency is None or self.init_time_dim not in self.dims:
            self.min = data.min(reduction_dims).load()
            self.max = data.max(reduction_dims).load()

        else:
            grouped_data = self._add_grouping_coordinate(data)

            self.min = (
                grouped_data.groupby(self.frequency).min(dim=reduction_dims).load()
            )

            self.max = (
                grouped_data.groupby(self.frequency).max(dim=reduction_dims).load()
            )

        self.fitted = True
        return self

    def transform(self, data: xr.Dataset):
        """
        Document this function.

        Parameters
        ----------
        data : xr.Dataset
            Description not yet provided.

        Returns
        -------
        Any
            Description not yet provided.
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

    def inverse_transform(self, data: xr.Dataset):
        """
        Document this function.

        Parameters
        ----------
        data : xr.Dataset
            Description not yet provided.

        Returns
        -------
        Any
            Description not yet provided.
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
    Document this class.

    Parameters
    ----------
    dims : list[str] | None
        Description not yet provided.
    frequency : TemporalFrequency | None
        Description not yet provided.
    **kwargs : Any
        Description not yet provided.
    """

    def __init__(
        self,
        dims: list[str] | None = None,
        frequency: TemporalFrequency | None = None,
        **kwargs,
    ) -> None:
        """
        Document this function.

        Parameters
        ----------
        dims : list[str] | None
            Description not yet provided.
        frequency : TemporalFrequency | None
            Description not yet provided.
        **kwargs : Any
            Description not yet provided.

        Raises
        ------
        ValueError
            Description not yet provided.
        """
        self.mean: xr.Dataset | xr.DataArray | None = None
        self.std: xr.Dataset | xr.DataArray | None = None

        self.dims = tuple(dims) if dims is not None else tuple()
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
        Document this function.

        Parameters
        ----------
        data : xr.Dataset | xr.DataArray
            Description not yet provided.
        mask : xr.DataArray
            Description not yet provided.

        Returns
        -------
        Any
            Description not yet provided.
        """
        reduction_dims = self._get_reduction_dims(data)

        if mask is not None:
            data = data.where(~mask)

        if self.frequency is None or self.init_time_dim not in self.dims:
            self.mean = data.mean(reduction_dims).load()
            std = data.std(reduction_dims).load()

        else:
            grouped_data = self._add_grouping_coordinate(data)

            self.mean = (
                grouped_data.groupby(self.frequency).mean(dim=reduction_dims).load()
            )

            std = grouped_data.groupby(self.frequency).std(dim=reduction_dims, ddof = 1).load()

        self.std = std.where(std > 0)

        self.fitted = True
        return self

    def transform(self, data: xr.Dataset):
        """
        Document this function.

        Parameters
        ----------
        data : xr.Dataset
            Description not yet provided.

        Returns
        -------
        Any
            Description not yet provided.
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

    def inverse_transform(self, data: xr.Dataset):
        """
        Document this function.

        Parameters
        ----------
        data : xr.Dataset
            Description not yet provided.

        Returns
        -------
        Any
            Description not yet provided.
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
    Document this class.

    Parameters
    ----------
    dims : list[str] | None
        Description not yet provided.
    frequency : TemporalFrequency | None
        Description not yet provided.
    **kwargs : Any
        Description not yet provided.
    """

    def __init__(
        self,
        dims: list[str] | None = None,
        frequency: TemporalFrequency | None = None,
        **kwargs,
    ) -> None:
        """
        Document this function.

        Parameters
        ----------
        dims : list[str] | None
            Description not yet provided.
        frequency : TemporalFrequency | None
            Description not yet provided.
        **kwargs : Any
            Description not yet provided.
        """
        self.mean: xr.DataArray | xr.Dataset | None = None

        self.dims = tuple(dims) if dims is not None else tuple()
        self.frequency = frequency

        self.large_ensemble = False
        self.fitted = False

    def fit(self, data: xr.Dataset | xr.DataArray, mask: xr.DataArray = None):
        """
        Document this function.

        Parameters
        ----------
        data : xr.Dataset | xr.DataArray
            Description not yet provided.
        mask : xr.DataArray
            Description not yet provided.

        Returns
        -------
        Any
            Description not yet provided.
        """
        reduction_dims = self._get_reduction_dims(data)

        if mask is not None:
            data = data.where(~mask)

        if self.frequency is None or self.init_time_dim not in self.dims:
            self.mean = data.mean(reduction_dims).load()

        else:
            grouped_data = self._add_grouping_coordinate(data)

            self.mean = (
                grouped_data.groupby(self.frequency).mean(dim=reduction_dims).load()
            )

        self.fitted = True
        return self

    def transform(self, data: xr.Dataset):
        """
        Document this function.

        Parameters
        ----------
        data : xr.Dataset
            Description not yet provided.

        Returns
        -------
        Any
            Description not yet provided.
        """
        self._check_fitted()

        mean = self._align_stat_for_transform(
            data=data,
            stat=self.mean,
        )

        return data - mean

    def inverse_transform(self, data: xr.Dataset):
        """
        Document this function.

        Parameters
        ----------
        data : xr.Dataset
            Description not yet provided.

        Returns
        -------
        Any
            Description not yet provided.
        """
        self._check_fitted()

        mean = align_stat_data_lead_time_inverse_transform(
            ds=data,
            stat=self.mean,
            lead_time_resolution=self.lead_time_resolution,
        )

        return data + mean


@PreprocessingStepSelector.register("trend_remover")
class TrendRemover(PreprocessModuleABC):
    """
    Document this class.

    Parameters
    ----------
    frequency : Literal['month', 'day'] | None
        Description not yet provided.
    **kwargs : Any
        Description not yet provided.
    """

    def __init__(
        self,
        frequency: Literal["month", "day"] | None = None,
        **kwargs,
    ) -> None:
        """
        Document this function.

        Parameters
        ----------
        frequency : Literal['month', 'day'] | None
            Description not yet provided.
        **kwargs : Any
            Description not yet provided.

        Raises
        ------
        ValueError
            Description not yet provided.
        """
        if frequency not in {None, "month", "day"}:
            raise ValueError(
                f"Unsupported frequency {frequency!r}. "
                "Expected None, 'month', or 'day'."
            )

        self.frequency = frequency
        self.dims = [self.init_time_dim]

        self.slope: xr.Dataset | None = None
        self.intercept: xr.Dataset | None = None
        self.time_origin = None

        self.fitted = False

    @staticmethod
    def _time_to_numeric(
        times: xr.DataArray,
        origin: (np.datetime64 | datetime.datetime | cftime.datetime),
    ) -> xr.DataArray:
        """
        Document this function.

        Parameters
        ----------
        times : xr.DataArray
            Description not yet provided.
        origin : np.datetime64 | datetime.datetime | cftime.datetime
            Description not yet provided.

        Returns
        -------
        xr.DataArray
            Description not yet provided.

        Raises
        ------
        ValueError
            Description not yet provided.
        """
        values = np.asarray(times.values)

        if isinstance(origin, xr.DataArray):
            origin = origin.values

        if isinstance(origin, np.ndarray):
            if origin.ndim != 0:
                raise ValueError("'origin' must be a scalar time value.")
            origin = origin[()]

        if values.size == 0:
            raise ValueError("Time coordinate cannot be empty.")

        if isinstance(values.reshape(-1)[0], cftime.datetime):
            numeric = np.asarray(
                [
                    (time - origin).total_seconds() / 86400.0
                    for time in values.reshape(-1)
                ],
                dtype=np.float64,
            ).reshape(values.shape)

        else:
            numeric = ((values - origin) / np.timedelta64(1, "D")).astype(np.float64)

        return xr.DataArray(
            numeric,
            dims=times.dims,
            coords=times.coords,
        )

    def _fit_linear_trend(
        self,
        data: xr.Dataset,
    ) -> tuple[xr.Dataset, xr.Dataset]:
        """
        Document this function.

        Parameters
        ----------
        data : xr.Dataset
            Description not yet provided.

        Returns
        -------
        tuple[xr.Dataset, xr.Dataset]
            Description not yet provided.

        Raises
        ------
        RuntimeError
            Description not yet provided.
        """
        numeric_time = self._time_to_numeric(
            data[self.init_time_dim],
            origin=self.time_origin,
        )

        data_numeric = data.assign_coords(
            {
                self.init_time_dim: numeric_time.values,
            }
        )

        coefficients = data_numeric.polyfit(
            dim=self.init_time_dim,
            deg=1,
            skipna=True,
        )

        slope = xr.Dataset()
        intercept = xr.Dataset()

        for name in data.data_vars:
            coeff_name = f"{name}_polyfit_coefficients"

            if coeff_name not in coefficients:
                raise RuntimeError(
                    f"Expected coefficient variable {coeff_name!r} "
                    "was not produced by xarray.polyfit."
                )

            coeff = coefficients[coeff_name]

            slope[name] = coeff.sel(
                degree=1,
                drop=True,
            )

            intercept[name] = coeff.sel(
                degree=0,
                drop=True,
            )

        return slope, intercept

    def fit(
        self,
        data: xr.Dataset,
        mask: xr.DataArray | None = None,
    ):
        """
        Document this function.

        Parameters
        ----------
        data : xr.Dataset
            Description not yet provided.
        mask : xr.DataArray | None
            Description not yet provided.

        Returns
        -------
        Any
            Description not yet provided.

        Raises
        ------
        ValueError
            Description not yet provided.
        """
        if self.init_time_dim not in data.dims:
            raise ValueError(
                f"Data must contain the initialization-time dimension "
                f"{self.init_time_dim!r}."
            )

        if mask is not None:
            data = data.where(~mask)

        self.time_origin = data[self.init_time_dim].min()

        if self.frequency is None:
            self.slope, self.intercept = self._fit_linear_trend(data)

        else:
            grouped_data = self._add_grouping_coordinate(data)

            slopes = []
            intercepts = []

            for group_value, group in grouped_data.groupby(self.frequency):
                if group.sizes[self.init_time_dim] < 2:
                    raise ValueError(
                        "At least two initialization times are required "
                        "to fit each trend. "
                        f"{self.frequency!r} group {group_value!r} "
                        "contains fewer than two samples."
                    )

                slope, intercept = self._fit_linear_trend(group)

                slopes.append(slope.expand_dims({self.frequency: [group_value]}))

                intercepts.append(
                    intercept.expand_dims({self.frequency: [group_value]})
                )

            self.slope = xr.concat(
                slopes,
                dim=self.frequency,
            )

            self.intercept = xr.concat(
                intercepts,
                dim=self.frequency,
            )

        self.slope = self.slope.load()
        self.intercept = self.intercept.load()

        self.fitted = True
        return self

    def transform(
        self,
        data: xr.Dataset,
    ) -> xr.Dataset:
        """
        Document this function.

        Parameters
        ----------
        data : xr.Dataset
            Description not yet provided.

        Returns
        -------
        xr.Dataset
            Description not yet provided.
        """
        self._check_fitted()

        slope = self._align_stat_for_transform(
            data=data,
            stat=self.slope,
        )

        intercept = self._align_stat_for_transform(
            data=data,
            stat=self.intercept,
        )

        numeric_time = self._time_to_numeric(
            data[self.init_time_dim],
            origin=self.time_origin,
        )

        trend = slope * numeric_time + intercept

        return data - trend

    def inverse_transform(
        self,
        data: xr.Dataset,
    ) -> xr.Dataset:
        """
        Document this function.

        Parameters
        ----------
        data : xr.Dataset
            Description not yet provided.

        Returns
        -------
        xr.Dataset
            Description not yet provided.
        """
        self._check_fitted()

        slope = align_stat_data_lead_time_inverse_transform(
            ds=data,
            stat=self.slope,
            lead_time_resolution=self.lead_time_resolution,
        )

        intercept = align_stat_data_lead_time_inverse_transform(
            ds=data,
            stat=self.intercept,
            lead_time_resolution=self.lead_time_resolution,
        )

        target_time = self._get_inverse_target_time(data)

        numeric_time = self._time_to_numeric(
            target_time,
            origin=self.time_origin,
        )

        trend = slope * numeric_time + intercept

        return data + trend


@PreprocessingStepSelector.register("flattener")
class Flattennanremove(PreprocessModuleABC):
    """
    Document this class.

    Parameters
    ----------
    load_dir : Path | str
        Description not yet provided.
    **kwargs : Any
        Description not yet provided.
    """

    supported_NN_dimensions_sorted: ClassVar[tuple] = supported_NN_dimensions_sorted

    def __init__(self, load_dir: Path | str = None, **kwargs):
        """
        Document this function.

        Parameters
        ----------
        load_dir : Path | str
            Description not yet provided.
        **kwargs : Any
            Description not yet provided.
        """
        self.load_dir = load_dir
        self.fitted = False
        self.common_to_input_and_target = False
        self.NN_dims: list[str] = []

    def fit(
        self,
        data: xr.Dataset | xr.DataArray,
        target: xr.Dataset | xr.DataArray | None = None,
        mask=None,
        save: bool = False,
        save_name: str | None = None,
        save_path: Path | str = None,
    ):
        """
        Document this function.

        Parameters
        ----------
        data : xr.Dataset | xr.DataArray
            Description not yet provided.
        target : xr.Dataset | xr.DataArray | None
            Description not yet provided.
        mask : Any
            Description not yet provided.
        save : bool
            Description not yet provided.
        save_name : str | None
            Description not yet provided.
        save_path : Path | str
            Description not yet provided.

        Returns
        -------
        Any
            Description not yet provided.

        Raises
        ------
        RuntimeError
            Description not yet provided.
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
        """
        Document this function.

        Parameters
        ----------
        data : xr.Dataset | xr.DataArray
            Description not yet provided.

        Raises
        ------
        ValueError
            Description not yet provided.
        """
        if data is not None:
            missing_dims = [dim for dim in self.NN_dims if dim not in data.dims]

            if missing_dims:
                raise ValueError(
                    "The saved preprocessor and data pipelines are not compatable. "
                    f"Missing dimensions: {missing_dims}."
                )

    def transform(self, data: xr.Dataset) -> xr.Dataset:
        """
        Document this function.

        Parameters
        ----------
        data : xr.Dataset
            Description not yet provided.

        Returns
        -------
        xr.Dataset
            Description not yet provided.
        """
        if "ref" in data.dims:
            return data.sel(ref=self.final_locations)

        return (data.stack(ref=self.NN_dims).sel(ref=self.final_locations)).transpose(
            ..., "ref"
        )

    def inverse_transform(self, data: xr.Dataset) -> xr.Dataset:
        """
        Document this function.

        Parameters
        ----------
        data : xr.Dataset
            Description not yet provided.

        Returns
        -------
        xr.Dataset
            Description not yet provided.

        Raises
        ------
        ValueError
            Description not yet provided.
        """
        if "ref" not in data.dims:
            raise ValueError("The input must contain the flattened 'ref' dimension.")

        return data.unstack().combine_first(self.reference_shape)

    def _load_from_memory(self, load_dir: Path | str) -> None:
        """
        Document this function.

        Parameters
        ----------
        load_dir : Path | str
            Description not yet provided.

        Raises
        ------
        RuntimeError
            Description not yet provided.
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
    Document this function.

    Parameters
    ----------
    ds : xr.DataArray
        Description not yet provided.
    stat : xr.DataArray
        Description not yet provided.
    lead_time_resolution : lead_time_unit
        Description not yet provided.
    init_time_dim : str
        Description not yet provided.
    lead_time_dim : str
        Description not yet provided.

    Returns
    -------
    xr.DataArray
        Description not yet provided.

    Raises
    ------
    ValueError
        Description not yet provided.
    """
    if lead_time_dim in stat.dims:
        return stat

    temporal_stat_dims = {
        init_time_dim,
        "year",
        "month",
        "day",
    }

    if temporal_stat_dims.isdisjoint(stat.dims):
        return stat

    if init_time_dim not in ds.coords:
        raise ValueError(
            f"Data must contain the initialization-time coordinate {init_time_dim!r}."
        )

    init_times = np.asarray(ds[init_time_dim].values)

    if init_times.ndim != 1:
        raise ValueError(f"Coordinate {init_time_dim!r} must be one-dimensional.")

    if lead_time_dim in ds.dims:
        lead_times = np.asarray(ds[lead_time_dim].values)

        if lead_times.ndim != 1:
            raise ValueError(f"Coordinate {lead_time_dim!r} must be one-dimensional.")

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
