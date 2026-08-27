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

init_time_dim, lead_time_dim = required_sample_dimensions


def _unwrap_data_variables(dataset: xr.Dataset) -> xr.DataArray:
    """
    Document this function.

    Parameters
    ----------
    dataset : xr.Dataset
        Description not yet provided.

    Returns
    -------
    xr.DataArray
        Description not yet provided.
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
    realization_dim: str = realization_dim,
    supported_NN_dimensions_sorted: tuple = supported_NN_dimensions_sorted,
):
    """
    Document this function.

    Parameters
    ----------
    paths : list[str]
        Description not yet provided.
    selection : dict | None
        Description not yet provided.
    names : list | None
        Description not yet provided.
    ensemble_mean : bool
        Description not yet provided.
    concat_dim : str
        Description not yet provided.
    rename_dict : dict | None
        Description not yet provided.
    chunks : dict | None
        Description not yet provided.
    load : bool
        Description not yet provided.
    add_time_auxiliary_coords : bool
        Description not yet provided.
    init_time_dim : str
        Description not yet provided.
    realization_dim : str
        Description not yet provided.
    supported_NN_dimensions_sorted : tuple
        Description not yet provided.

    Returns
    -------
    Any
        Description not yet provided.

    Raises
    ------
    TypeError
        Description not yet provided.
    ValueError
        Description not yet provided.
    """
    ds = xr.open_mfdataset(
        paths, combine="nested", concat_dim=concat_dim, chunks=chunks
    )

    if rename_dict is not None:
        ds = ds.rename(rename_dict)

    ds = ds.sel(selection) if selection is not None else ds

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
        ds = ds.load()

    return ds


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
    Document this function.

    Parameters
    ----------
    init_times : Sequence[np.datetime64 | datetime.datetime | cftime.datetime] | np.ndarray | xr.DataArray
        Description not yet provided.
    lead_times : Sequence | xr.DataArray | np.ndarray | int
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

        cutoff_month = (
            latest_init_time.dt.year * 12
            + latest_init_time.dt.month
        )

        target_month = (
            target_times.dt.year * 12
            + target_times.dt.month
        )

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
    Document this function.

    Parameters
    ----------
    init_times : xr.DataArray
        Description not yet provided.
    lead_times : np.ndarray
        Description not yet provided.
    lead_time_resolution : lead_time_unit
        Description not yet provided.

    Returns
    -------
    np.ndarray
        Description not yet provided.

    Raises
    ------
    ValueError
        Description not yet provided.
    """
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
    Document this function.

    Parameters
    ----------
    time : np.datetime64 | cftime.datetime
        Description not yet provided.
    n_months : int
        Description not yet provided.

    Returns
    -------
    np.datetime64 | cftime.datetime
        Description not yet provided.
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
    Document this function.

    Parameters
    ----------
    times_sequence : Sequence | np.ndarray | xr.DataArray
        Description not yet provided.

    Raises
    ------
    TypeError
        Description not yet provided.
    ValueError
        Description not yet provided.
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


@contextlib.contextmanager
def suppress_stderr() -> Iterator[None]:
    """
    Document this function.

    Yields
    ------
    None
        Description not yet provided.
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
    Document this function.

    Parameters
    ----------
    init_times : np.ndarray | xr.DataArray
        Description not yet provided.
    lead_times : np.ndarray | xr.DataArray
        Description not yet provided.
    lead_time_resolution : lead_time_unit
        Description not yet provided.

    Returns
    -------
    np.ndarray
        Description not yet provided.

    Raises
    ------
    TypeError
        Description not yet provided.
    ValueError
        Description not yet provided.
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
    Document this function.

    Parameters
    ----------
    init_time : cftime.datetime
        Description not yet provided.
    offset : int
        Description not yet provided.
    resolution : lead_time_unit
        Description not yet provided.

    Returns
    -------
    cftime.datetime
        Description not yet provided.

    Raises
    ------
    ValueError
        Description not yet provided.
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
    Document this function.

    Parameters
    ----------
    year : int
        Description not yet provided.
    month : int
        Description not yet provided.
    calendar_name : str
        Description not yet provided.

    Returns
    -------
    int
        Description not yet provided.
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
    Document this function.

    Parameters
    ----------
    ds : xr.Dataset | xr.DataArray
        Description not yet provided.
    init_time_dim : str
        Description not yet provided.
    calendar : str | None
        Description not yet provided.

    Returns
    -------
    xr.Dataset | xr.DataArray
        Description not yet provided.

    Raises
    ------
    TypeError
        Description not yet provided.
    ValueError
        Description not yet provided.
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
    Document this function.

    Parameters
    ----------
    time : xr.DataArray | pd.DatetimeIndex | xr.CFTimeIndex
        Description not yet provided.

    Returns
    -------
    TimeTypes
        Description not yet provided.

    Raises
    ------
    TypeError
        Description not yet provided.
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
    Document this function.

    Parameters
    ----------
    times : pd.DatetimeIndex | xr.CFTimeIndex
        Description not yet provided.

    Returns
    -------
    TimeFrequency
        Description not yet provided.

    Raises
    ------
    ValueError
        Description not yet provided.
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
