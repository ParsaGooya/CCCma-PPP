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

import warnings

warnings.filterwarnings("ignore")

init_time_dim, lead_time_dim = required_sample_dimensions
TemporalFrequency = Literal["year", "month", "day"]


@PreprocessingStepSelector.register("normalizer")
class Normalizer(PreprocessModuleABC):
    """
    Normalize data to a range determined by fitted minimum and maximum values.

    Statistics are computed over the configured dimensions and can optionally be
    grouped by a temporal frequency.

    Parameters
    ----------
    dims : list[str] | None
        Dimensions over which minimum and maximum values are calculated. If
        ``None``, no explicit reduction dimensions are configured.
    frequency : TemporalFrequency | None
        Temporal frequency used to group normalization statistics. Supported
        values are ``"year"``, ``"month"``, ``"day"``, and ``None``.
    kwargs : Any
        Additional keyword arguments accepted for preprocessing compatibility.
    """

    def __init__(
        self,
        dims: list[str] | None = None,
        frequency: TemporalFrequency | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize a normalization preprocessor.

        Parameters
        ----------
        dims : list[str] | None
            Dimensions over which minimum and maximum values are calculated. If
            ``None``, no explicit reduction dimensions are configured.
        frequency : TemporalFrequency | None
            Temporal frequency used to group normalization statistics.
        kwargs : Any
            Additional keyword arguments accepted for preprocessing compatibility.

        Raises
        ------
        ValueError
            If ``frequency`` is not supported.
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

        An optional mask is applied before statistics are calculated. When temporal
        grouping is enabled and initialization time is a reduction dimension,
        statistics are calculated separately for each temporal group.

        Parameters
        ----------
        data : xr.Dataset | xr.DataArray
            Data used to calculate normalization statistics.
        mask : xr.DataArray | None
            Optional mask identifying values excluded from fitting.

        Returns
        -------
        Normalizer
            This fitted normalization preprocessor.

        Raises
        ------
        ValueError
            If the fitted normalization range contains zero values.
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

        self._check_zeros(self.max - self.min)
        self.fitted = True
        return self

    def transform(self, data: xr.Dataset):
        """
        Normalize data using fitted minimum and maximum statistics.

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
        Restore normalized data to its original scale.

        Fitted statistics are aligned with target times derived from initialization
        and lead times before the inverse transformation is applied.

        Parameters
        ----------
        data : xr.Dataset
            Normalized data to inverse transform.

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


@PreprocessingStepSelector.register("standardizer")
class Standardizer(PreprocessModuleABC):
    """
    Standardize data using fitted mean and standard deviation statistics.

    Statistics are computed over the configured dimensions and can optionally be
    grouped by temporal frequency. Standard deviation values that are zero or
    negative are masked and therefore become missing values during
    standardization.

    Parameters
    ----------
    dims : list[str] | None
        Dimensions over which mean and standard deviation are calculated. If
        ``None``, no explicit reduction dimensions are configured.
    frequency : TemporalFrequency | None
        Temporal frequency used to group standardization statistics. Supported
        values are ``"year"``, ``"month"``, ``"day"``, and ``None``.
    kwargs : Any
        Additional keyword arguments accepted for preprocessing compatibility.
    """

    def __init__(
        self,
        dims: list[str] | None = None,
        frequency: TemporalFrequency | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize a standardization preprocessor.

        Parameters
        ----------
        dims : list[str] | None
            Dimensions over which mean and standard deviation are calculated. If
            ``None``, no explicit reduction dimensions are configured.
        frequency : TemporalFrequency | None
            Temporal frequency used to group standardization statistics.
        kwargs : Any
            Additional keyword arguments accepted for preprocessing compatibility.

        Raises
        ------
        ValueError
            If ``frequency`` is not supported.
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
        Fit mean and standard deviation statistics.

        An optional mask is applied before statistics are calculated. When temporal
        grouping is enabled and initialization time is a reduction dimension,
        statistics are calculated separately for each temporal group. Nonpositive
        standard deviations are masked.

        Parameters
        ----------
        data : xr.Dataset | xr.DataArray
            Data used to calculate standardization statistics.
        mask : xr.DataArray | None
            Optional mask identifying values excluded from fitting.

        Returns
        -------
        Standardizer
            This fitted standardization preprocessor.
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
        Standardize data using fitted mean and standard deviation statistics.

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
        Restore standardized data to its original scale.

        Fitted statistics are aligned with target times derived from initialization
        and lead times before the inverse transformation is applied.

        Parameters
        ----------
        data : xr.Dataset
            Standardized data to inverse transform.

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


@PreprocessingStepSelector.register("anomalies")
class AnomaliesScaler(PreprocessModuleABC):
    """
    Convert data to anomalies relative to a fitted mean.

    The mean is computed over configured dimensions and can optionally be grouped
    by temporal frequency.

    Parameters
    ----------
    dims : list[str] | None
        Dimensions over which the mean is calculated. If ``None``, no explicit
        reduction dimensions are configured.
    frequency : TemporalFrequency | None
        Temporal frequency used to group fitted means.
    kwargs : Any
        Additional keyword arguments accepted for preprocessing compatibility.
    """

    def __init__(
        self,
        dims: list[str] | None = None,
        frequency: TemporalFrequency | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize an anomaly-scaling preprocessor.

        Parameters
        ----------
        dims : list[str] | None
            Dimensions over which the mean is calculated. If ``None``, no explicit
            reduction dimensions are configured.
        frequency : TemporalFrequency | None
            Temporal frequency used to group fitted means.
        kwargs : Any
            Additional keyword arguments accepted for preprocessing compatibility.
        """
        self.mean: xr.DataArray | xr.Dataset | None = None

        self.dims = tuple(dims) if dims is not None else tuple()
        self.frequency = frequency

        self.fitted = False

    def fit(self, data: xr.Dataset | xr.DataArray, mask: xr.DataArray = None):
        """
        Fit the mean used to calculate anomalies.

        An optional mask is applied before the mean is calculated. When temporal
        grouping is enabled and initialization time is a reduction dimension,
        separate means are calculated for each temporal group.

        Parameters
        ----------
        data : xr.Dataset | xr.DataArray
            Data used to calculate the fitted mean.
        mask : xr.DataArray | None
            Optional mask identifying values excluded from fitting.

        Returns
        -------
        AnomaliesScaler
            This fitted anomaly-scaling preprocessor.
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

        The fitted mean is aligned with target times derived from initialization and
        lead times before the inverse transformation is applied.

        Parameters
        ----------
        data : xr.Dataset
            Anomaly data to inverse transform.

        Returns
        -------
        xr.Dataset
            Data restored by adding the fitted mean.
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
    Remove fitted linear temporal trends from data.

    A single trend can be fitted across all initialization times, or separate
    trends can be fitted by month or day of year.

    Parameters
    ----------
    frequency : Literal["month", "day"] | None
        Temporal grouping used when fitting trends. If ``None``, one trend is
        fitted across all initialization times.
    kwargs : Any
        Additional keyword arguments accepted for preprocessing compatibility.
    """

    def __init__(
        self,
        frequency: Literal["month", "day"] | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize a linear trend-removal preprocessor.

        Parameters
        ----------
        frequency : Literal["month", "day"] | None
            Temporal grouping used when fitting trends. If ``None``, one trend is
            fitted across all initialization times.
        kwargs : Any
            Additional keyword arguments accepted for preprocessing compatibility.

        Raises
        ------
        ValueError
            If ``frequency`` is not ``None``, ``"month"``, or ``"day"``.

        Notes
        -----
        Yearly grouping is not supported because a trend requires variation across
        years within each temporal group.
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
        Convert datetime values to elapsed days relative to an origin.

        Supports NumPy datetime values and CF datetime values.

        Parameters
        ----------
        times : xr.DataArray
            Datetime values to convert.
        origin : np.datetime64 | datetime.datetime | cftime.datetime
            Reference time corresponding to zero elapsed days.

        Returns
        -------
        xr.DataArray
            Floating-point elapsed days with the same dimensions and coordinates as
            ``times``.

        Raises
        ------
        ValueError
            If ``origin`` is a non-scalar array or the time coordinate is empty.
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
        Fit a linear temporal trend to each data variable.

        Initialization times are converted to elapsed days relative to the fitted
        time origin before first-degree polynomial fitting is performed.

        Parameters
        ----------
        data : xr.Dataset
            Dataset for which linear trends are fitted.

        Returns
        -------
        tuple[xr.Dataset, xr.Dataset]
            Fitted slope and intercept values for each data variable.

        Raises
        ------
        RuntimeError
            If the expected polynomial coefficient variable is not produced for a
            data variable.
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
        Fit linear temporal trends to a dataset.

        An optional mask is applied before fitting. When a temporal frequency is
        configured, independent trends are fitted to each month or day-of-year group.

        Parameters
        ----------
        data : xr.Dataset
            Dataset used to fit temporal trends.
        mask : xr.DataArray | None
            Optional mask identifying values excluded from trend fitting.

        Returns
        -------
        TrendRemover
            This fitted trend-removal preprocessor.

        Raises
        ------
        ValueError
            If the initialization-time dimension is absent or a temporal group
            contains fewer than two initialization times.
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
        Remove the fitted linear trend from data.

        Fitted slope and intercept values are aligned with the input initialization
        times before the trend is calculated and subtracted.

        Parameters
        ----------
        data : xr.Dataset
            Data from which the fitted trend is removed.

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
        Restore the fitted linear trend to data.

        Trend statistics are aligned with target times derived from initialization
        and lead times before the trend is reconstructed and added.

        Parameters
        ----------
        data : xr.Dataset
            Detrended data to inverse transform.

        Returns
        -------
        xr.Dataset
            Data with the fitted temporal trend restored.
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
    Flatten supported neural-network dimensions and remove invalid locations.

    The preprocessor identifies spatial locations that contain valid values and
    represents the supported neural-network dimensions with a single ``ref``
    dimension. When a target is supplied, only locations valid in both input and
    target data are retained.

    Parameters
    ----------
    load_dir : Path | str | None
        Path to a previously fitted flattening preprocessor to load during
        fitting.
    kwargs : Any
        Additional keyword arguments accepted for preprocessing compatibility.
    """

    supported_NN_dimensions_sorted: ClassVar[tuple] = supported_NN_dimensions_sorted

    def __init__(self, load_dir: Path | str = None, **kwargs):
        """
        Initialize a flattening preprocessor.

        Parameters
        ----------
        load_dir : Path | str | None
            Path to a previously fitted flattening preprocessor to load during
            fitting.
        kwargs : Any
            Additional keyword arguments accepted for preprocessing compatibility.
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
        Fit the flattened spatial representation.

        Supported neural-network dimensions are stacked into a ``ref`` dimension and
        locations containing missing values are removed. When target data are
        provided, the retained locations are restricted to those valid in both input
        and target data.

        Parameters
        ----------
        data : xr.Dataset | xr.DataArray
            Input data used to determine valid flattened locations.
        target : xr.Dataset | xr.DataArray | None
            Optional target data used to restrict locations to those valid in both
            input and target data.
        mask : Any
            Optional mask accepted for preprocessing-pipeline compatibility but not
            used by this preprocessor.
        save : bool
            Whether to save the fitted preprocessor.
        save_name : str | None
            Base name used when saving the fitted preprocessor.
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
        Validate data against the fitted neural-network dimensions.

        Parameters
        ----------
        data : xr.Dataset | xr.DataArray | None
            Data whose dimensions are checked against the fitted configuration.

        Raises
        ------
        ValueError
            If one or more neural-network dimensions required by the saved
            preprocessor are absent from the data.
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
        Transform data to the fitted flattened representation.

        If the data already contain a ``ref`` dimension, they are restricted to the
        fitted locations. Otherwise, the neural-network dimensions are stacked before
        the fitted locations are selected.

        Parameters
        ----------
        data : xr.Dataset
            Data to flatten.

        Returns
        -------
        xr.Dataset
            Data restricted to the fitted ``ref`` locations.
        """
        if "ref" in data.dims:
            return data.sel(ref=self.final_locations)

        return (data.stack(ref=self.NN_dims).sel(ref=self.final_locations)).transpose(
            ..., "ref"
        )

    def inverse_transform(self, data: xr.Dataset) -> xr.Dataset:
        """
        Restore flattened data to the fitted spatial representation.

        Parameters
        ----------
        data : xr.Dataset
            Flattened data containing the fitted ``ref`` dimension.

        Returns
        -------
        xr.Dataset
            Data unstacked to the original neural-network dimensions and combined
            with the stored reference coordinates.

        Raises
        ------
        ValueError
            If the input data do not contain a ``ref`` dimension.
        """
        if "ref" not in data.dims:
            raise ValueError("The input must contain the flattened 'ref' dimension.")

        return data.unstack().combine_first(self.reference_shape)

    def _load_from_memory(self, load_dir: Path | str) -> None:
        """
        Load a fitted flattening preprocessor from disk.

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
        del loaded


def align_stat_data_lead_time_inverse_transform(
    ds: xr.DataArray,
    stat: xr.DataArray,
    lead_time_resolution: lead_time_unit = "month",
    init_time_dim: str = init_time_dim,
    lead_time_dim: str = lead_time_dim,
) -> xr.DataArray:
    """
    Align fitted temporal statistics with times used for inverse transformation.

    Statistics indexed directly by lead time are returned unchanged. Otherwise,
    target times are derived from initialization and lead times and used to
    select statistics indexed by initialization time, year, month, or day of
    year.

    Parameters
    ----------
    ds : xr.DataArray
        Data whose temporal coordinates define the target times required for
        alignment.
    stat : xr.DataArray
        Fitted statistic to align with the data.
    lead_time_resolution : lead_time_unit
        Temporal unit used to interpret lead-time values.
    init_time_dim : str
        Name of the initialization-time dimension.
    lead_time_dim : str
        Name of the lead-time dimension.

    Returns
    -------
    xr.DataArray
        Statistic aligned with the temporal coordinates of ``ds``.

    Raises
    ------
    ValueError
        If a required initialization-time coordinate is absent or initialization
        or lead-time coordinates are not one-dimensional.
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
