from __future__ import annotations
import numpy as np
import xarray as xr
import os
from typing import Literal
from collections.abc import Sequence
import contextlib
import datetime
import cftime
import calendar
import pandas as pd
from collections.abc import Iterator

from cccma_ppp.configs import (
    supported_NN_dimensions_sorted,
    required_sample_dimensions,
    realization_dim,
    lead_time_unit,
    lead_time_resolution,
)


TimeTypes = Literal["datetime", "cftime"]
TimeFrequency = Literal["day", "month", "year"]
spatialmethod = Literal["uniform", "cosine_lat"]
TEMPORAL_ORDER = {
    "day": 0,
    "month": 1,
    "year": 2,
}
init_time_dim, lead_time_dim = required_sample_dimensions


def _unwrap_data_variables(dataset: xr.Dataset) -> xr.DataArray:
    """
    Combine dataset variables into a channel dimension.

    Each data variable is squeezed, assigned a singleton ``channels`` dimension,
    and concatenated with the remaining variables along that dimension.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset containing variables to combine.

    Returns
    -------
    xr.DataArray
        Data variables combined along a leading ``channels`` dimension.
    """
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
    concat_dim: str = "year",
    rename_dict: dict | None = None,
    chunks: dict | None = None,
    load: bool = False,
    add_time_auxiliary_coords: bool = False,
    init_time_dim: str = init_time_dim,
    resolved_init_time_frequency: TemporalFrequency | None = None,
    realization_dim: str = realization_dim,
    supported_NN_dimensions_sorted: tuple = supported_NN_dimensions_sorted,
):
    """
    Open and prepare data from multiple xarray-compatible files.

    Files are combined along the configured concatenation dimension and can be
    renamed, subset, temporally standardized, ensemble averaged, restricted to
    selected variables, and loaded into memory. Neural-network dimensions are
    placed at the end of the resulting dimension order.

    Parameters
    ----------
    paths : list[str]
        Paths to the data files to open.
    selection : dict | None
        Coordinate selections applied after opening and renaming the data.
    names : list | None
        Names of data variables to retain. If ``None``, all variables are kept.
    ensemble_mean : bool
        Whether to average over the realization dimension when it is present.
    concat_dim : str
        Dimension along which the input files are concatenated.
    rename_dict : dict | None
        Mapping from existing dimension names to replacement names.
    chunks : dict | None
        Dask chunk sizes used when opening the files.
    load : bool
        Whether to load the resulting dataset into memory.
    add_time_auxiliary_coords : bool
        Whether to derive year, month, and day-of-year coordinates from the
        initialization-time coordinate.
    init_time_dim : str
        Name of the initialization-time coordinate.
    resolved_init_time_frequency : TimeFrequency | None
        Temporal frequency to which initialization timestamps are standardized.
        If ``None``, timestamps are left unchanged.
    realization_dim : str
        Name of the ensemble realization dimension.
    supported_NN_dimensions_sorted : tuple
        Ordered neural-network dimensions placed at the end of the dataset
        dimension order when present.

    Returns
    -------
    xr.Dataset
        Opened and prepared dataset.

    Raises
    ------
    TypeError
        If temporal auxiliary coordinates are requested but the initialization
        coordinate does not contain supported datetime values.
    ValueError
        If ``rename_dict`` contains dimensions absent from the dataset, temporal
        auxiliary coordinates are requested without a valid one-dimensional
        initialization-time coordinate, or the time sequence is invalid.
    """

    ds = xr.open_mfdataset(
        paths, combine="nested", concat_dim=concat_dim, chunks=chunks
    )

    if rename_dict is not None:
        invalid_keys = set(rename_dict) - set(ds.dims)
        if invalid_keys:
            raise ValueError(f"Invalid keys in rename_dict: {sorted(invalid_keys)}")
        ds = ds.rename(rename_dict)

    ds = ds.sel(selection) if selection is not None else ds

    if resolved_init_time_frequency is not None:
        ds = _standardize_time(
            ds,
            frequency=resolved_init_time_frequency,
            time_dim=init_time_dim,
        )

    if realization_dim in ds.coords and ensemble_mean:
        ds = ds.mean(realization_dim)

    if names is not None:
        ds = ds[names]

    if add_time_auxiliary_coords:
        if init_time_dim not in ds.coords:
            raise ValueError(
                "Cannot add temporal auxiliary coordinates because "
                f"{init_time_dim!r} is not present in the data coordinates."
            )

        init_time = ds.coords[init_time_dim]

        _validate_time_sequence(init_time)

        if init_time.ndim != 1 or init_time.dims != (init_time_dim,):
            raise ValueError(
                f"Coordinate {init_time_dim!r} must be one-dimensional "
                f"over dimension {init_time_dim!r}, but found dimensions "
                f"{init_time.dims}."
            )

        try:
            ds = ds.assign_coords(
                year=init_time.dt.year,
                month=init_time.dt.month,
                day=init_time.dt.dayofyear,
            )
        except (AttributeError, TypeError) as exc:
            raise TypeError(
                f"Coordinate {init_time_dim!r} must contain datetime64 "
                "or cftime datetime values to derive year, month, and day."
            ) from exc

    nn_dims = [dim for dim in supported_NN_dimensions_sorted if dim in ds.dims]

    ds = ds.transpose(..., *nn_dims)
    if load:
        ds.load()
        ds.close()

    return ds


def _standardize_time(
    ds: xr.Dataset | xr.DataArray,
    frequency: TimeFrequency,
    time_dim: str = init_time_dim,
) -> xr.Dataset | xr.DataArray:
    """
    Standardize temporal coordinates to representative timestamps.

    Only the time coordinate is modified; the underlying data are neither
    resampled nor aggregated. Daily timestamps are moved to midnight, monthly
    timestamps to the 15th day of each month, and yearly timestamps to January 1.

    Parameters
    ----------
    ds : xr.Dataset | xr.DataArray
        Data containing the temporal coordinate to standardize.
    frequency : TimeFrequency
        Temporal frequency used to standardize timestamps. Supported values are
        ``"day"``, ``"month"``, and ``"year"``.
    time_dim : str
        Name of the temporal coordinate to standardize.

    Returns
    -------
    xr.Dataset | xr.DataArray
        Data with standardized temporal coordinates.

    Raises
    ------
    ValueError
        If ``frequency`` is unsupported, the temporal coordinate is missing or
        is not a one-dimensional dimension coordinate, or standardization
        produces duplicate timestamps.
    TypeError
        If the temporal coordinate does not use a supported datetime index type.
    """
    if frequency not in ("day", "month", "year"):
        raise ValueError(f"Unsupported temporal frequency: {frequency}")

    if time_dim not in ds.coords:
        raise ValueError(f"Coordinate {time_dim!r} not found.")

    times = ds[time_dim]

    if times.ndim != 1 or times.dims != (time_dim,):
        raise ValueError(
            f"{time_dim!r} must be a one-dimensional dimension coordinate."
        )

    index = ds.indexes[time_dim]

    def standardize(t):
        """
        Convert a timestamp to the representative time for the configured frequency.

        Parameters
        ----------
        t : Any
            Timestamp to standardize.

        Returns
        -------
        Any
            Timestamp of the same type with its temporal components standardized.
        """
        year, month, day = t.year, t.month, t.day
        hour = t.hour

        if frequency == "year":
            return type(t)(year, 1, 1)

        if frequency == "month":
            return type(t)(year, month, 15)

        if frequency == "day":
            return type(t)(year, month, day)

        return type(t)(year, month, day, hour)

    if isinstance(index, pd.DatetimeIndex):
        standardized = pd.DatetimeIndex([standardize(t) for t in index])
    elif isinstance(index, xr.CFTimeIndex):
        standardized = xr.CFTimeIndex([standardize(t) for t in index])
    else:
        raise TypeError(f"Unsupported time index type: {type(index)}")

    if standardized.has_duplicates:
        raise ValueError(
            "Time standardization produced duplicate "
            "timestamps. Check the input frequency."
        )

    return ds.assign_coords({time_dim: standardized})


def _create_train_mask(
    init_times: (
        Sequence[np.datetime64 | datetime.datetime | cftime.datetime]
        | np.ndarray
        | xr.DataArray
    ),
    lead_times: Sequence | xr.DataArray | np.ndarray | int,
    max_allowed_time: np.datetime64 | datetime.datetime | cftime.datetime | None = None,
    lead_time_resolution: lead_time_unit = lead_time_resolution,
    init_time_dim: str = init_time_dim,
    lead_time_dim: str = lead_time_dim,
) -> xr.DataArray:
    """
    Create a mask identifying sample times beyond the available temporal range.

    Valid times are calculated from every initialization-time and lead-time
    combination. Samples extending beyond the temporal coverage of the
    initialization data are masked, with comparison behavior adjusted for yearly
    and monthly initialization frequencies.

    Parameters
    ----------
    init_times : Sequence[np.datetime64 | datetime.datetime | cftime.datetime] | np.ndarray | xr.DataArray
        Initialization times defining the available temporal coverage.
    lead_times : Sequence | xr.DataArray | np.ndarray | int
        One-based lead times. An integer creates all lead times from one through
        the specified value.
    max_allowed_time : np.datetime64 | datetime.datetime | cftime.datetime | None
        Optional additional upper bound on valid target times.
    lead_time_resolution : lead_time_unit
        Temporal unit used to interpret lead-time offsets.
    init_time_dim : str
        Name of the initialization-time dimension.
    lead_time_dim : str
        Name of the lead-time dimension.

    Returns
    -------
    xr.DataArray
        Boolean mask indexed by initialization time and lead time, where ``True``
        identifies target times outside the allowed temporal range.

    Raises
    ------
    ValueError
        If the initialization-time sequence is invalid, lead times are empty or
        not one-dimensional, an integer lead-time count is less than one, or any
        lead time is less than one.
    """
    _validate_time_sequence(init_times)

    init_times = xr.DataArray(
        init_times,
        dims=(init_time_dim,),
        coords={init_time_dim: init_times},
    )

    if isinstance(lead_times, int):
        if lead_times < 1:
            raise ValueError("'lead_times' must be at least 1.")

        lead_times = np.arange(1, lead_times + 1)
    else:
        lead_times = np.asarray(lead_times)

        if lead_times.ndim != 1 or lead_times.size == 0:
            raise ValueError("'lead_times' must be a non-empty 1D array.")

        if np.any(lead_times < 1):
            raise ValueError("Lead times must be one-based and at least 1.")

    target_times = _calculate_target_times(
        init_times=init_times,
        lead_times=lead_times,
        lead_time_resolution=lead_time_resolution,
    )

    target_times = xr.DataArray(
        target_times,
        dims=(init_time_dim, lead_time_dim),
        coords={
            init_time_dim: init_times[init_time_dim],
            lead_time_dim: lead_times,
        },
    )

    time_resolution = infer_time_resolution(init_times.coords[init_time_dim].to_index())

    if time_resolution == "year":
        cutoff_year = int(init_times.dt.year.max().item())
        mask = target_times.dt.year > cutoff_year

    elif time_resolution == "month" and lead_time_resolution == "day":
        latest_init_time = init_times.max()

        cutoff_month = latest_init_time.dt.year * 12 + latest_init_time.dt.month

        target_month = target_times.dt.year * 12 + target_times.dt.month

        mask = target_month > cutoff_month

    else:
        cutoff_time = init_times.max().values
        mask = target_times > cutoff_time

    if max_allowed_time is not None:
        mask = mask | (target_times > max_allowed_time)

    mask.name = "mask"
    return mask


def _calculate_target_times(
    init_times: xr.DataArray,
    lead_times: np.ndarray,
    lead_time_resolution: lead_time_unit,
) -> np.ndarray:
    """
    Calculate target times from initialization times and lead times.

    Lead times are one-based, such that a lead time of one corresponds to the
    initialization time itself.

    Parameters
    ----------
    init_times : xr.DataArray
        Initialization times from which target times are calculated.
    lead_times : np.ndarray
        One-based lead-time values.
    lead_time_resolution : lead_time_unit
        Temporal unit used to interpret lead-time offsets.

    Returns
    -------
    np.ndarray
        Target times with shape ``(len(init_times), len(lead_times))``.

    Raises
    ------
    ValueError
        If any lead time is less than one or ``lead_time_resolution`` is not
        supported.
    """

    if np.any(lead_times < 1):
        raise ValueError("Lead times must be one-based (minimum value 1).")

    offsets = lead_times.astype(int) - 1

    first_time = init_times.values[0]
    is_cftime = isinstance(first_time, cftime.datetime)

    if lead_time_resolution == "day":
        valid_times = [
            [
                init_time + datetime.timedelta(days=int(offset))
                if is_cftime
                else init_time + np.timedelta64(int(offset), "D")
                for offset in offsets
            ]
            for init_time in init_times.values
        ]

        return np.asarray(
            valid_times,
            dtype=object if is_cftime else None,
        )

    if lead_time_resolution == "month":
        valid_times = [
            [_add_months(init_time, int(offset)) for offset in offsets]
            for init_time in init_times.values
        ]

        return np.asarray(
            valid_times,
            dtype=object if is_cftime else "datetime64[ns]",
        )

    else:
        raise ValueError(
            f"Invalid lead_time_resolution {lead_time_resolution}. Must be in "
            "['day', 'month']. "
        )


def _add_months(
    time: np.datetime64 | cftime.datetime,
    n_months: int,
) -> np.datetime64 | cftime.datetime:
    """
    Advance a datetime by a specified number of months.

    Parameters
    ----------
    time : np.datetime64 | cftime.datetime
        Starting datetime.
    n_months : int
        Number of months to advance.

    Returns
    -------
    np.datetime64 | cftime.datetime
        Datetime obtained after advancing by ``n_months`` months.
    """
    is_cftime = isinstance(time, cftime.datetime)

    calendar = time.calendar if is_cftime else "proleptic_gregorian"

    return xr.date_range(
        start=time,
        periods=n_months + 1,
        freq="MS",
        calendar=calendar,
        use_cftime=is_cftime,
    )[-1]


def _validate_time_sequence(
    times_sequence: Sequence | np.ndarray | xr.DataArray,
) -> None:
    """
    Validate a sequence of datetime values.

    The sequence must be non-empty, one-dimensional, and contain a consistent
    datetime representation. CF datetime sequences must use a single calendar,
    while Python datetime values must be naive or UTC.

    Parameters
    ----------
    times_sequence : Sequence | np.ndarray | xr.DataArray
        Sequence of datetime values to validate.

    Raises
    ------
    TypeError
        If values are not supported datetime objects or different datetime
        representations are mixed within the sequence.
    ValueError
        If the sequence is empty or not one-dimensional, CF datetime values use
        multiple calendars, or Python datetime values contain non-UTC timezone
        offsets.
    """
    values = (
        times_sequence.values
        if isinstance(times_sequence, xr.DataArray)
        else np.asarray(times_sequence)
    )

    if values.ndim != 1:
        raise ValueError("'times_sequence' must be one-dimensional.")

    if values.size == 0:
        raise ValueError("'times_sequence' cannot be empty.")

    first = values[0]

    is_cftime = isinstance(first, cftime.datetime)
    is_numpy_datetime = isinstance(first, np.datetime64)
    is_python_datetime = isinstance(first, datetime.datetime)

    if not (is_cftime or is_numpy_datetime or is_python_datetime):
        raise TypeError(
            "'time_sequence' must contain cftime.datetime, "
            "numpy.datetime64, or datetime.datetime objects."
        )

    if is_cftime:
        if not all(isinstance(value, cftime.datetime) for value in values):
            raise TypeError(
                "'time_sequence' cannot mix cftime objects with other datetime types."
            )

        calendars = {value.calendar for value in values}

        if len(calendars) > 1:
            raise ValueError(
                f"'time_sequence' contains multiple CF calendars: {calendars}."
            )

    elif is_numpy_datetime:
        if not np.issubdtype(values.dtype, np.datetime64):
            raise TypeError(
                "'time_sequence' cannot mix numpy.datetime64 values "
                "with other datetime types."
            )

    else:
        if not all(isinstance(value, datetime.datetime) for value in values):
            raise TypeError(
                "'time_sequence' cannot mix datetime.datetime values "
                "with other datetime types."
            )

        offsets = {value.utcoffset() for value in values}

        if any(offset not in {None, datetime.timedelta(0)} for offset in offsets):
            raise ValueError("'time_sequence' contains non-UTC datetime values.")


@contextlib.contextmanager
def suppress_stderr() -> Iterator[None]:
    """
    Temporarily suppress output written to standard error.

    Standard error is redirected to the operating system null device for the
    duration of the context and restored afterward.

    Yields
    ------
    None
        Control to the code executed while standard error is suppressed.
    """
    stderr_fd = 2
    saved_stderr_fd = os.dup(stderr_fd)

    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stderr_fd)


def add_lead_times(
    init_times: np.ndarray | xr.DataArray,
    lead_times: np.ndarray | xr.DataArray,
    lead_time_resolution: lead_time_unit = "month",
) -> np.ndarray:
    """
    Calculate paired target times from initialization times and lead times.

    Each initialization time is paired with the lead time at the same array
    position. Lead times are one-based, so a lead time of one corresponds to the
    initialization time itself.

    Parameters
    ----------
    init_times : np.ndarray | xr.DataArray
        One-dimensional array of initialization times.
    lead_times : np.ndarray | xr.DataArray
        One-dimensional array of one-based lead times corresponding element-wise
        to ``init_times``.
    lead_time_resolution : lead_time_unit
        Temporal unit used to interpret lead-time offsets.

    Returns
    -------
    np.ndarray
        Target times obtained by applying each lead-time offset to its
        corresponding initialization time.

    Raises
    ------
    TypeError
        If initialization times are neither ``numpy.datetime64`` nor
        ``cftime.datetime`` values.
    ValueError
        If the input arrays are not one-dimensional or do not have matching
        shapes, lead times are not integer-valued or contain values less than one,
        or the lead-time resolution is unsupported.
    """
    init_times = np.asarray(init_times)
    lead_times = np.asarray(lead_times)

    if init_times.ndim != 1 or lead_times.ndim != 1:
        raise ValueError("'init_times' and 'lead_times' must be one-dimensional.")

    if init_times.shape != lead_times.shape:
        raise ValueError("'init_times' and 'lead_times' must have the same shape.")

    if not np.issubdtype(lead_times.dtype, np.integer):
        if not np.all(lead_times == lead_times.astype(int)):
            raise ValueError("Lead times must contain integer values.")

        lead_times = lead_times.astype(int)

    if np.any(lead_times < 1):
        raise ValueError("Lead times must be one-based and at least 1.")

    offsets = lead_times.astype(int) - 1

    first_time = init_times[0]

    if isinstance(first_time, cftime.datetime):
        valid_times = [
            _add_cftime_offset(
                init_time=init_time,
                offset=int(offset),
                resolution=lead_time_resolution,
            )
            for init_time, offset in zip(init_times, offsets)
        ]

        return np.asarray(valid_times, dtype=object)

    if np.issubdtype(init_times.dtype, np.datetime64):
        if lead_time_resolution == "day":
            return init_times + offsets.astype("timedelta64[D]")

        if lead_time_resolution == "month":
            month_times = init_times.astype("datetime64[M]")
            return month_times + offsets.astype("timedelta64[M]")

        raise ValueError(f"Unsupported lead-time resolution: {lead_time_resolution!r}.")

    raise TypeError(
        "'init_times' must contain numpy.datetime64 or cftime.datetime values."
    )


def _add_cftime_offset(
    init_time: cftime.datetime,
    offset: int,
    resolution: lead_time_unit,
) -> cftime.datetime:
    """
    Add a temporal offset to a CF datetime.

    Daily offsets are applied directly. Monthly offsets preserve the original
    day when possible and otherwise use the final valid day of the target month.

    Parameters
    ----------
    init_time : cftime.datetime
        Starting CF datetime.
    offset : int
        Number of temporal units to add.
    resolution : lead_time_unit
        Temporal unit of the offset.

    Returns
    -------
    cftime.datetime
        Shifted datetime using the same CF datetime type as ``init_time``.

    Raises
    ------
    ValueError
        If ``resolution`` is not supported.
    """
    if resolution == "day":
        return init_time + datetime.timedelta(days=offset)

    if resolution != "month":
        raise ValueError(f"Unsupported lead-time resolution: {resolution!r}.")

    total_months = init_time.year * 12 + init_time.month - 1 + offset

    target_year, zero_based_month = divmod(total_months, 12)
    target_month = zero_based_month + 1

    target_day = min(
        init_time.day,
        _days_in_cftime_month(
            year=target_year,
            month=target_month,
            calendar_name=init_time.calendar,
        ),
    )

    return type(init_time)(
        target_year,
        target_month,
        target_day,
        init_time.hour,
        init_time.minute,
        init_time.second,
        init_time.microsecond,
    )


def _days_in_cftime_month(
    year: int,
    month: int,
    calendar_name: str,
) -> int:
    """
    Return the number of days in a month for a CF calendar.

    Parameters
    ----------
    year : int
        Calendar year.
    month : int
        Calendar month from 1 through 12.
    calendar_name : str
        Name of the CF calendar.

    Returns
    -------
    int
        Number of days in the requested month.
    """
    if calendar_name == "360_day":
        return 30

    if calendar_name in {"noleap", "365_day"}:
        month_lengths = (
            31,
            28,
            31,
            30,
            31,
            30,
            31,
            31,
            30,
            31,
            30,
            31,
        )
        return month_lengths[month - 1]

    if calendar_name in {"all_leap", "366_day"}:
        month_lengths = (
            31,
            29,
            31,
            30,
            31,
            30,
            31,
            31,
            30,
            31,
            30,
            31,
        )
        return month_lengths[month - 1]

    return calendar.monthrange(year, month)[1]


def assign_datetime_init_time(
    ds: xr.Dataset | xr.DataArray,
    init_time_dim: str = init_time_dim,
    calendar: str | None = None,
) -> xr.Dataset | xr.DataArray:
    """
    Convert integer initialization years to datetime coordinates.

    Integer year coordinates are replaced with January 1 timestamps. NumPy
    datetimes are used when no calendar is specified; otherwise CF datetime
    values are generated using the requested calendar.

    Parameters
    ----------
    ds : xr.Dataset | xr.DataArray
        Data containing an integer initialization-year coordinate.
    init_time_dim : str
        Name of the initialization-time coordinate.
    calendar : str | None
        CF calendar used to construct timestamps. If ``None``, NumPy datetime
        values are created.

    Returns
    -------
    xr.Dataset | xr.DataArray
        Data with the initialization coordinate replaced by datetime values.

    Raises
    ------
    TypeError
        If the initialization-time coordinate does not contain integer years.
    ValueError
        If ``init_time_dim`` is not a coordinate of the input data.
    """
    if init_time_dim not in ds.coords:
        raise ValueError(f"{init_time_dim!r} is not a coordinate.")

    years = np.asarray(ds.coords[init_time_dim].values)
    if years.ndim == 0:
        years = np.array([years])

    if not np.issubdtype(years.dtype, np.integer):
        raise TypeError(f"{init_time_dim!r} must contain integer years.")

    if calendar is None:
        times = np.array([np.datetime64(f"{year:04d}-01-01") for year in years])
    else:
        times = xr.date_range(
            start=f"{years.min():04d}-01-01",
            periods=len(years),
            freq="YS",
            calendar=calendar,
            use_cftime=True,
        )

    return ds.assign_coords({init_time_dim: (init_time_dim, times)})


def get_time_representation(
    time: xr.DataArray | pd.DatetimeIndex | xr.CFTimeIndex,
) -> TimeTypes:
    """
    Determine the datetime representation used by a temporal coordinate.

    Parameters
    ----------
    time : xr.DataArray | pd.DatetimeIndex | xr.CFTimeIndex
        Temporal coordinate or index to inspect.

    Returns
    -------
    TimeTypes
        ``"datetime"`` for NumPy, Python, or pandas datetime values, or
        ``"cftime"`` for CF datetime values.

    Raises
    ------
    TypeError
        If the input does not use a supported datetime representation.
    """
    if isinstance(time, xr.DataArray):
        values = time.values

        first = values[0]

        if isinstance(first, cftime.datetime):
            return "cftime"

        if isinstance(first, np.datetime64) or isinstance(first, datetime.datetime):
            return "datetime"

    elif isinstance(time, xr.CFTimeIndex):
        return "cftime"

    elif isinstance(time, pd.DatetimeIndex):
        return "datetime"

    raise TypeError("Time coordinate must use either datetime or cftime objects.")


def infer_time_resolution(
    times: pd.DatetimeIndex | xr.CFTimeIndex,
) -> TimeFrequency:
    """
    Infer the temporal resolution of a datetime index.

    Resolution is determined from the smallest interval between consecutive
    timestamps. Intervals shorter than 28 days are classified as daily, intervals
    shorter than 365 days as monthly, and longer intervals as yearly.

    Parameters
    ----------
    times : pd.DatetimeIndex | xr.CFTimeIndex
        Ordered datetime index from which to infer temporal resolution.

    Returns
    -------
    TimeFrequency
        Inferred temporal resolution: ``"day"``, ``"month"``, or ``"year"``.

    Raises
    ------
    ValueError
        If fewer than two timestamps are provided.
    """
    if len(times) < 2:
        raise ValueError(
            "At least two timestamps are required to infer the resolution."
        )

    deltas = [(t2 - t1).days for t1, t2 in zip(times[:-1], times[1:])]

    min_delta = min(deltas)

    if min_delta < 28:
        return "day"

    if min_delta < 365:
        return "month"

    return "year"
