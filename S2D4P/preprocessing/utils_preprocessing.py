import warnings

warnings.filterwarnings("ignore")

import numpy as np
import xarray as xr
from pathlib import Path
import joblib
import datetime
import cftime
from typing import ClassVar, Literal

from S2D4P.preprocessing.selector import PreprocessingStepSelector
from S2D4P.preprocessing.preprocessing_ABC import PreprocessModuleABC
from S2D4P.generic.runtime import RuntimeContext
from S2D4P.data.utils import add_lead_times, infer_time_resolution
from S2D4P.configs import (
    supported_NN_dimensions_sorted,
    required_sample_dimensions,
    lead_time_unit,
)

init_time_dim, lead_time_dim = required_sample_dimensions
TemporalFrequency = Literal["year", "month", "day"]


@PreprocessingStepSelector.register("normalizer")
class Normalizer(PreprocessModuleABC):
    """
    Scale data to a normalized range using fitted minimum and maximum values.

    Statistics are calculated over the configured dimensions and can optionally
    be grouped by year, month, or day.

    Parameters
    ----------
    dims : list[str] | None
        Dimensions over which minimum and maximum statistics are calculated.
    frequency : TemporalFrequency | None
        Optional temporal frequency used to group fitted statistics.
    kwargs : Any
        Additional unused keyword arguments accepted for pipeline compatibility.
    """

    def __init__(
        self,
        dims: list[str] | None = None,
        frequency: TemporalFrequency | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the normalizer.

        Parameters
        ----------
        dims : list[str] | None
            Dimensions over which minimum and maximum statistics are calculated.
        frequency : TemporalFrequency | None
            Optional temporal grouping frequency. Supported values are ``"year"``,
            ``"month"``, ``"day"``, and ``None``.
        kwargs : Any
            Additional unused keyword arguments accepted for pipeline compatibility.

        Raises
        ------
        ValueError
            If ``frequency`` is unsupported.
        """
        self.min: xr.DataArray | xr.Dataset | None = None
        self.max: xr.DataArray | xr.Dataset | None = None

        self.dims = tuple(dims) if dims is not None else tuple()
        self.frequency = frequency

        self.fitted = False

        if self.frequency not in self.supported_frequencies:
            raise ValueError(
                f"Unsupported frequency {self.frequency!r}. "
                f"Expected one of {self.supported_frequencies}."
            )

    def fit(self, data: xr.Dataset | xr.DataArray, mask: xr.DataArray = None):
        """
        Fit minimum and maximum normalization statistics.

        An optional mask excludes values from fitting. When temporal grouping is
        enabled and initialization time is reduced, separate statistics are fitted
        for each temporal group.

        Parameters
        ----------
        data : xr.Dataset | xr.DataArray
            Data used to estimate normalization statistics.
        mask : xr.DataArray | None
            Optional mask identifying values excluded from fitting.

        Returns
        -------
        Normalizer
            This fitted normalizer.

        Raises
        ------
        ValueError
            If any fitted normalization range is zero.
        """
        self._reset_fit()

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

        self._check_zeros(self.max - self.min)
        self.fitted = True
        return self

    def transform(self, data: xr.Dataset):
        """
        Normalize data using fitted minimum and maximum values.

        Parameters
        ----------
        data : xr.Dataset
            Data to normalize.

        Returns
        -------
        xr.Dataset
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

    def inverse_transform(self, data: xr.Dataset):
        """
        Reverse normalization using fitted statistics.

        Statistics are temporally aligned with initialization or forecast target
        times before the original data scale is restored.

        Parameters
        ----------
        data : xr.Dataset
            Normalized data to restore.

        Returns
        -------
        xr.Dataset
            Data restored to its original scale.
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

    def _reset_fit(self):
        """Reset fitted statistics while preserving configuration."""
        self.min = None
        self.max = None
        self.fitted = False


@PreprocessingStepSelector.register("standardizer")
class Standardizer(PreprocessModuleABC):
    """
    Standardize data using fitted mean and standard deviation.

    Statistics are calculated over the configured dimensions and can optionally
    be grouped by year, month, or day. Locations with zero standard deviation are
    masked because they cannot be standardized.

    Parameters
    ----------
    dims : list[str] | None
        Dimensions over which mean and standard deviation are calculated.
    frequency : TemporalFrequency | None
        Optional temporal frequency used to group fitted statistics.
    kwargs : Any
        Additional unused keyword arguments accepted for pipeline compatibility.

    Notes
    -----
    Fitted standard deviations that are not greater than zero are replaced with
    missing values and therefore remain masked during transformation.
    """

    def __init__(
        self,
        dims: list[str] | None = None,
        frequency: TemporalFrequency | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the standardizer.

        Parameters
        ----------
        dims : list[str] | None
            Dimensions over which mean and standard deviation are calculated.
        frequency : TemporalFrequency | None
            Optional temporal grouping frequency. Supported values are ``"year"``,
            ``"month"``, ``"day"``, and ``None``.
        kwargs : Any
            Additional unused keyword arguments accepted for pipeline compatibility.

        Raises
        ------
        ValueError
            If ``frequency`` is unsupported.
        """
        self.mean: xr.Dataset | xr.DataArray | None = None
        self.std: xr.Dataset | xr.DataArray | None = None

        self.dims = tuple(dims) if dims is not None else tuple()
        self.frequency = frequency

        self.fitted = False

        if self.frequency not in self.supported_frequencies:
            raise ValueError(
                f"Unsupported frequency {self.frequency!r}. "
                f"Expected one of {self.supported_frequencies}."
            )

    def fit(self, data: xr.Dataset | xr.DataArray, mask: xr.DataArray = None):
        """
        Fit mean and sample standard-deviation statistics.

        An optional mask excludes values from fitting. When temporal grouping is
        enabled and initialization time is reduced, separate statistics are fitted
        for each temporal group. Standard deviations that are not positive are
        masked.

        Parameters
        ----------
        data : xr.Dataset | xr.DataArray
            Data used to estimate standardization statistics.
        mask : xr.DataArray | None
            Optional mask identifying values excluded from fitting.

        Returns
        -------
        Standardizer
            This fitted standardizer.
        """

        self._reset_fit()

        reduction_dims = self._get_reduction_dims(data)

        if mask is not None:
            data = data.where(~mask)

        if self.frequency is None or self.init_time_dim not in self.dims:
            self.mean = data.mean(reduction_dims).load()
            std = data.std(reduction_dims, ddof=1).load()

        else:
            grouped_data = self._add_grouping_coordinate(data)

            self.mean = (
                grouped_data.groupby(self.frequency).mean(dim=reduction_dims).load()
            )

            std = (
                grouped_data.groupby(self.frequency)
                .std(dim=reduction_dims, ddof=1)
                .load()
            )

        self.std = std.where(std > 0)

        self.fitted = True
        return self

    def transform(self, data: xr.Dataset):
        """
        Standardize data using fitted mean and standard deviation.

        Parameters
        ----------
        data : xr.Dataset
            Data to standardize.

        Returns
        -------
        xr.Dataset
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

    def inverse_transform(self, data: xr.Dataset):
        """
        Reverse standardization using fitted statistics.

        Statistics are temporally aligned with initialization or forecast target
        times before the original data scale is restored.

        Parameters
        ----------
        data : xr.Dataset
            Standardized data to restore.

        Returns
        -------
        xr.Dataset
            Data restored to its original scale.
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

    def _reset_fit(self):
        """Reset fitted statistics while preserving configuration."""
        self.mean = None
        self.std = None
        self.fitted = False


@PreprocessingStepSelector.register("anomalies")
class AnomaliesScaler(PreprocessModuleABC):
    """
    Convert data to anomalies relative to a fitted mean.

    The mean is calculated over configurable dimensions and can optionally be
    estimated separately by year, month, or day.

    Parameters
    ----------
    dims : list[str] | None
        Dimensions over which the mean is calculated.
    frequency : TemporalFrequency | None
        Optional temporal frequency used to group fitted means.
    kwargs : Any
        Additional unused keyword arguments accepted for pipeline compatibility.
    """

    def __init__(
        self,
        dims: list[str] | None = None,
        frequency: TemporalFrequency | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the anomaly scaler.

        Parameters
        ----------
        dims : list[str] | None
            Dimensions over which the mean is calculated.
        frequency : TemporalFrequency | None
            Optional temporal grouping frequency.
        kwargs : Any
            Additional unused keyword arguments accepted for pipeline compatibility.
        """
        self.mean: xr.DataArray | xr.Dataset | None = None

        self.dims = tuple(dims) if dims is not None else tuple()
        self.frequency = frequency

        self.fitted = False

    def fit(self, data: xr.Dataset | xr.DataArray, mask: xr.DataArray = None):
        """
        Fit mean values used to calculate anomalies.

        An optional mask excludes values from fitting. When temporal grouping is
        enabled and initialization time is reduced, a separate mean is fitted for
        each temporal group.

        Parameters
        ----------
        data : xr.Dataset | xr.DataArray
            Data used to estimate mean values.
        mask : xr.DataArray | None
            Optional mask identifying values excluded from fitting.

        Returns
        -------
        AnomaliesScaler
            This fitted anomaly scaler.
        """
        self._reset_fit()

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
        Convert data to anomalies by subtracting the fitted mean.

        Parameters
        ----------
        data : xr.Dataset
            Data to transform.

        Returns
        -------
        xr.Dataset
            Anomalies relative to the fitted mean.
        """
        self._check_fitted()

        mean = self._align_stat_for_transform(
            data=data,
            stat=self.mean,
        )

        return data - mean

    def inverse_transform(self, data: xr.Dataset):
        """
        Restore anomalies by adding the fitted mean.

        The fitted mean is temporally aligned with initialization or forecast target
        times before being added.

        Parameters
        ----------
        data : xr.Dataset
            Anomaly data to restore.

        Returns
        -------
        xr.Dataset
            Data with the fitted mean restored.
        """
        self._check_fitted()

        mean = align_stat_data_lead_time_inverse_transform(
            ds=data,
            stat=self.mean,
            lead_time_resolution=self.lead_time_resolution,
        )

        return data + mean

    def _reset_fit(self):
        """Reset fitted statistics while preserving configuration."""
        self.mean = None
        self.fitted = False


@PreprocessingStepSelector.register("trend_remover")
class TrendRemover(PreprocessModuleABC):
    """
    Remove a fitted linear temporal trend from data.

    A linear trend is fitted along initialization time. Trends can be fitted
    globally or separately by month or day-of-year.

    Parameters
    ----------
    frequency : Literal["month", "day"] | None
        Optional temporal grouping frequency. If ``None``, one trend is fitted
        across all initialization times.
    kwargs : Any
        Additional unused keyword arguments accepted for pipeline compatibility.
    """

    TEMPORAL_ORDER: ClassVar[dict] = {
        "day": 0,
        "month": 1,
        "year": 2,
    }

    def __init__(
        self,
        frequency: Literal["month", "day"] | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the linear trend remover.

        Parameters
        ----------
        frequency : Literal["month", "day"] | None
            Optional temporal grouping frequency. ``None`` fits one global trend.
        kwargs : Any
            Additional unused keyword arguments accepted for pipeline compatibility.

        Raises
        ------
        ValueError
            If ``frequency`` is not ``None``, ``"month"``, or ``"day"``.

        Notes
        -----
        Yearly grouping is not supported because a trend requires multiple temporal
        samples within each fitted group.
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
        Convert datetime values to elapsed days from an origin.

        Supports NumPy datetime values, Python datetimes, and CFTime-compatible
        datetime arithmetic.

        Parameters
        ----------
        times : xr.DataArray
            Time values to convert.
        origin : np.datetime64 | datetime.datetime | cftime.datetime
            Scalar reference time from which elapsed days are measured.

        Returns
        -------
        xr.DataArray
            Floating-point elapsed days with the same dimensions and coordinates as
            ``times``.

        Raises
        ------
        ValueError
            If ``origin`` is a nonscalar array or ``times`` is empty.
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

        SECONDS_PER_DAY = 24 * 60 * 60
        if isinstance(values.reshape(-1)[0], cftime.datetime):
            numeric = np.asarray(
                [
                    (time - origin).total_seconds() / SECONDS_PER_DAY
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
        Fit a first-degree polynomial along initialization time.

        Initialization times are converted to elapsed days before fitting. Separate
        slope and intercept datasets are extracted for every data variable.

        Parameters
        ----------
        data : xr.Dataset
            Data used to fit the linear trend.

        Returns
        -------
        tuple[xr.Dataset, xr.Dataset]
            Fitted slopes and intercepts for all variables.

        Raises
        ------
        RuntimeError
            If xarray does not produce the expected polynomial coefficient variable
            for a data variable.
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
        Fit linear temporal trends.

        Data can be fitted globally or independently within temporal groups. An
        optional mask excludes values from trend estimation.

        Parameters
        ----------
        data : xr.Dataset
            Data used to estimate linear trends.
        mask : xr.DataArray | None
            Optional mask identifying values excluded from fitting.

        Returns
        -------
        TrendRemover
            This fitted trend remover.

        Raises
        ------
        ValueError
            If initialization time is absent, the requested grouping frequency is
            coarser than the source temporal resolution, or a temporal group contains
            fewer than two initialization times.
        """
        self._reset_fit()

        if self.init_time_dim not in data.dims:
            raise ValueError(
                f"Data must contain the initialization-time dimension "
                f"{self.init_time_dim!r}."
            )

        if mask is not None:
            data = data.where(~mask)

        self.time_origin = data[self.init_time_dim].min()
        time_freq = infer_time_resolution(data[self.init_time_dim].to_index())

        if self.frequency is None:
            self.slope, self.intercept = self._fit_linear_trend(data)

        else:
            if self.TEMPORAL_ORDER.get(self.frequency) > self.TEMPORAL_ORDER.get(
                time_freq
            ):
                raise ValueError(
                    f"The requeted frequency of initial time: {self.frequency}"
                    "to fit the trend must be None or finer than the initial time "
                    f"frequency of the base data to fit on: {time_freq}"
                )

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
        Remove the fitted linear trend from data.

        Parameters
        ----------
        data : xr.Dataset
            Data from which the trend is removed.

        Returns
        -------
        xr.Dataset
            Detrended data.
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
        Restore the fitted linear trend.

        Trend statistics are aligned with the appropriate initialization or forecast
        target times before the trend is reconstructed and added to the data.

        Parameters
        ----------
        data : xr.Dataset
            Detrended data to restore.

        Returns
        -------
        xr.Dataset
            Data with the fitted linear trend restored.
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

        if self.lead_time_dim in slope.dims:
            target_time = data[self.init_time_dim]
        else:
            target_time = self._get_inverse_target_time(data)

        numeric_time = self._time_to_numeric(
            target_time,
            origin=self.time_origin,
        )

        trend = slope * numeric_time + intercept

        return data + trend

    def _reset_fit(self):
        """Reset fitted trend parameters while preserving configuration."""
        self.slope = None
        self.intercept = None
        self.time_origin = None
        self.fitted = False


@PreprocessingStepSelector.register("flattener")
class Flattennanremove(PreprocessModuleABC):
    """
    Flatten neural-network dimensions while removing invalid locations.

    Supported neural-network dimensions are stacked into a single ``ref``
    dimension. Only locations containing valid data are retained. When both input
    and target data are supplied, only locations valid in both are preserved.

    Parameters
    ----------
    load_dir : Path | str | None
        Path to a previously fitted flattening preprocessor.
    kwargs : Any
        Additional unused keyword arguments accepted for pipeline compatibility.
    """

    supported_NN_dimensions_sorted: ClassVar[tuple] = supported_NN_dimensions_sorted

    def __init__(self, load_dir: Path | str = None, **kwargs):
        """
        Initialize the flattening preprocessor.

        Parameters
        ----------
        load_dir : Path | str | None
            Path to a previously fitted flattening preprocessor. If provided, fitted
            state is restored during ``fit``.
        kwargs : Any
            Additional unused keyword arguments accepted for pipeline compatibility.
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
        Fit the valid flattened locations.

        Neural-network dimensions are stacked into a ``ref`` dimension and locations
        containing missing values are removed. When a target is provided, retained
        locations are restricted to those valid in both input and target data.

        Parameters
        ----------
        data : xr.Dataset | xr.DataArray
            Input data used to determine valid flattened locations.
        target : xr.Dataset | xr.DataArray | None
            Optional target data used to restrict locations to those valid in both
            input and target datasets.
        mask : Any
            Optional mask argument accepted for preprocessing pipeline compatibility.
            It is not used by this preprocessor.
        save : bool
            Whether to serialize the fitted preprocessor.
        save_name : str | None
            Base filename used when saving the fitted preprocessor.
        save_path : Path | str | None
            Directory in which the fitted preprocessor is saved.

        Returns
        -------
        Flattennanremove
            This fitted flattening preprocessor.

        Raises
        ------
        RuntimeError
            If input data do not contain all neural-network dimensions required by
            the reference data.
        """
        self._reset_fit()

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
        Validate neural-network dimensions against fitted state.

        Parameters
        ----------
        data : xr.Dataset | xr.DataArray
            Data whose dimensions are validated.

        Raises
        ------
        ValueError
            If one or more fitted neural-network dimensions are absent from ``data``.
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
        Flatten data and retain fitted valid locations.

        Parameters
        ----------
        data : xr.Dataset
            Data to flatten.

        Returns
        -------
        xr.Dataset
            Data containing the fitted valid locations along the trailing ``ref``
            dimension.
        """
        self._check_fitted()

        if "ref" in data.dims:
            return data.sel(ref=self.final_locations)

        return (data.stack(ref=self.NN_dims).sel(ref=self.final_locations)).transpose(
            ..., "ref"
        )

    def inverse_transform(self, data: xr.Dataset) -> xr.Dataset:
        """
        Restore flattened data to its original neural-network dimensions.

        Missing locations removed during flattening are reintroduced using the stored
        reference coordinate structure.

        Parameters
        ----------
        data : xr.Dataset
            Flattened data containing a ``ref`` dimension.

        Returns
        -------
        xr.Dataset
            Data restored to the original neural-network dimensions.

        Raises
        ------
        ValueError
            If ``data`` does not contain the flattened ``ref`` dimension.
        """
        self._check_fitted()

        if "ref" not in data.dims:
            raise ValueError("The input must contain the flattened 'ref' dimension.")

        return data.unstack().combine_first(self.reference_shape)

    def _load_from_memory(self, load_dir: Path | str) -> None:
        """
        Restore fitted flattening state from disk.

        Parameters
        ----------
        load_dir : Path | str
            Path to the serialized flattening preprocessor.

        Raises
        ------
        RuntimeError
            If the loaded preprocessor has not been fitted.
        """
        loaded = joblib.load(Path(load_dir))
        if not loaded.fitted:
            raise RuntimeError("the preprocessor to be loaded has to be fitted first.")

        self.reference_shape = loaded.reference_shape
        self.final_locations = loaded.final_locations
        self.common_to_input_and_target = loaded.common_to_input_and_target
        self.fitted = loaded.fitted
        self.NN_dims = loaded.NN_dims.copy()
        del loaded

    def _reset_fit(self):
        """Reset fitted attributes while preserving configuration."""
        self.fitted = False
        self.common_to_input_and_target = False
        self.NN_dims = []
        self.reference_shape = None
        self.final_locations = None


def align_stat_data_lead_time_inverse_transform(
    ds: xr.DataArray | xr.Dataset,
    stat: xr.DataArray | xr.Dataset,
    lead_time_resolution: lead_time_unit = "month",
    init_time_dim: str = init_time_dim,
    lead_time_dim: str = lead_time_dim,
) -> xr.DataArray | xr.Dataset:
    """
    Align preprocessing statistics with data for inverse transformation.

    Statistics containing a lead-time dimension are aligned using
    initialization times. Otherwise, forecast valid times are used.

    Temporal grouping dimensions ('year', 'month', 'day') are aligned
    to the appropriate temporal coordinates.

    Parameters
    ----------
    ds : xr.DataArray | xr.Dataset
        Data to be inverse-transformed.
    stat : xr.DataArray | xr.Dataset
        Fitted preprocessing statistics.
    lead_time_resolution : lead_time_unit
        Resolution of forecast lead times.
    init_time_dim : str
        Initialization-time dimension name.
    lead_time_dim : str
        Lead-time dimension name.

    Returns
    -------
    xr.DataArray | xr.Dataset
        Temporally aligned statistics.
    """
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

    # Select the temporal reference.
    if lead_time_dim in stat.dims or lead_time_dim not in ds.dims:
        # Statistics already have a lead-time dimension.
        # Align temporal groups using initialization times.
        target_time = ds[init_time_dim]

    else:
        # Statistics do not have lead_time.
        # Align temporal groups using forecast valid times.
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

        target_time = xr.DataArray(
            target_times,
            dims=(init_time_dim, lead_time_dim),
            coords={
                init_time_dim: ds[init_time_dim],
                lead_time_dim: ds[lead_time_dim],
            },
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

    return aligned_stat.assign_coords(
        {
            dim: ds[dim]
            for dim in (init_time_dim, lead_time_dim)
            if dim in aligned_stat.dims and dim in ds.coords
        }
    )
