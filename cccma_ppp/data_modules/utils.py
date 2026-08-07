import numpy as np
import xarray as xr
import os
from typing import Literal
from collections.abc import Sequence
import contextlib
import datetime
import cftime
import calendar
from collections.abc import Iterator

from cccma_ppp.preprocessing.preprocessing_ABC import PreprocessModuleABC
from cccma_ppp.configs import (supported_NN_dimensions_sorted, 
                               required_sample_dimensions, 
                               realization_dim,
                               lead_time_unit,
                               lead_time_resolution)

spatialmethod = Literal["uniform", "cosine_lat"]
init_time_dim, lead_time_dim = required_sample_dimensions



def _unwrap_data_variables(dataset: xr.Dataset) -> xr.DataArray:
    """
    Convert dataset variables into a channel dimension.

    Parameters
    ----------
    dataset : xr.Dataset

    Returns
    -------
    xr.DataArray
        Concatenated variables along "channels" dimension.
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
    preprocessor: PreprocessModuleABC | None = None,
    concat_dim: str = "year",
    rename_dict: dict | None = None,
    chunks: dict | None = None,
    load: bool = False,
    add_time_auxiliary_coords: bool = False,
    init_time_dim: str = init_time_dim,
    realization_dim: str = realization_dim,
    supported_NN_dimensions_sorted: tuple = supported_NN_dimensions_sorted
):
    """
    Load and optionally preprocess an xarray dataset.

    Parameters
    ----------
    paths : list[str]
        File paths to load.
    selection : dict or None, optional
        Subset selection applied after loading.
    names : list[str] or None, optional
        Variables to extract.
    ensemble_mean : bool, optional
        Whether to average across ensembles.
    preprocessor : PreprocessModuleABC or None, optional
        Preprocessing pipeline to apply.
    concat_dim : str, optional
        Dimension used for concatenation.
    rename_dict : dict or None, optional
        Variable or dimension renaming mapping.
    chunks : dict or None, optional
        Chunking passed to ``xr.open_mfdataset``.
    load : bool, optional
        Whether to load the result into memory.
    add_time_auxiliary_coords : bool, optional
        Whether to add ``year``, ``month``, and ``day`` coordinates
        derived from ``init_time``. Here, ``day`` represents day of year.
    init_time_dim : str, optional
        Name of the initialization-time dimension.

    Returns
    -------
    xr.Dataset or xr.DataArray
        Loaded and optionally preprocessed data.
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

    if preprocessor is not None:
        ds = preprocessor.transform(ds)

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
    lead_time_resolution: lead_time_unit = lead_time_resolution,
    init_time_dim: str = init_time_dim,
    lead_time_dim: str = lead_time_dim
) -> xr.DataArray:
    """
    Create a mask for forecast samples whose valid time extends beyond
    the final initialization time.

    Parameters
    ----------
    init_times : array-like
        Forecast initialization times. Values must be either NumPy
        datetime64 or cftime datetime objects.

    lead_times : array-like or int
        One-based lead-time values. If an integer is provided, it is
        interpreted as the number of lead times, producing
        ``1, ..., lead_times``.

    lead_time_resolution : {"month", "day"}
        Temporal resolution represented by one lead-time increment.

    Returns
    -------
    xr.DataArray
        Boolean mask with dimensions ``(init_time, lead_time)``.
        True indicates a sample whose valid time exceeds the final
        initialization time.

    Notes
    -----
    Lead times are assumed to be one-based:

    - lead_time=1 corresponds to init_time
    - lead_time=2 corresponds to one period after init_time
    """

    _validate_time_sequence(init_times)

    init_times = xr.DataArray(
        init_times,
        dims=(init_time_dim,),
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

    cutoff_time = init_times.max().values

    mask = target_times > cutoff_time

    return xr.DataArray(
        mask,
        dims=(init_time_dim, lead_time_dim),
        coords={
            init_time_dim: init_times.values,
            lead_time_dim: lead_times,
        },
        name="mask",
    )


def _calculate_target_times(
    init_times: xr.DataArray,
    lead_times: np.ndarray,
    lead_time_resolution: lead_time_unit,
) -> np.ndarray:
    """
    Calculate valid times for every init-time and lead-time combination.
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
            [
                _add_months(init_time, int(offset))
                for offset in offsets
            ]
            for init_time in init_times.values
        ]

        return np.asarray(
            valid_times,
            dtype=object if is_cftime else "datetime64[ns]",
        )

    else:
        raise ValueError(
            f"Invalid lead_time_resolution {lead_time_resolution}. Must be in " \
            "['day', 'month']. "
        )


def _add_months(
    time: np.datetime64 | cftime.datetime,
    n_months: int,
) -> np.datetime64 | cftime.datetime:
    """
    Add calendar months while preserving the time representation.
    """
    is_cftime = isinstance(time, cftime.datetime)

    calendar = (
        time.calendar
        if is_cftime
        else "proleptic_gregorian"
    )

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
                "'time_sequence' cannot mix cftime objects with "
                "other datetime types."
            )

        calendars = {value.calendar for value in values}

        if len(calendars) > 1:
            raise ValueError(
                "'time_sequence' contains multiple CF calendars: "
                f"{calendars}."
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
    """Temporarily suppress Python and C-library stderr output."""
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
    lead_time_resolution: lead_time_unit = 'month',
) -> np.ndarray:
    """
    Compute target times for paired initialization and lead-time values.

    Lead times are one-based:
        lead_time=1 -> initialization time
        lead_time=2 -> one period after initialization
    """
    init_times = np.asarray(init_times)
    lead_times = np.asarray(lead_times)

    if init_times.ndim != 1 or lead_times.ndim != 1:
        raise ValueError("'init_times' and 'lead_times' must be one-dimensional.")

    if init_times.shape != lead_times.shape:
        raise ValueError(
            "'init_times' and 'lead_times' must have the same shape."
        )

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
            # This assumes monthly coordinates represent month starts.
            month_times = init_times.astype("datetime64[M]")
            return month_times + offsets.astype("timedelta64[M]")

        raise ValueError(
            f"Unsupported lead-time resolution: {lead_time_resolution!r}."
        )

    raise TypeError(
        "'init_times' must contain numpy.datetime64 or cftime.datetime values."
    )


def _add_cftime_offset(
    init_time: cftime.datetime,
    offset: int,
    resolution: lead_time_unit,
) -> cftime.datetime:
    if resolution == "day":
        return init_time + datetime.timedelta(days=offset)

    if resolution != "month":
        raise ValueError(
            f"Unsupported lead-time resolution: {resolution!r}."
        )

    total_months = init_time.year * 12 + init_time.month - 1 + offset

    target_year, zero_based_month = divmod(total_months, 12)
    target_month = zero_based_month + 1

    # Preserve the original day when possible, but clamp it for shorter months.
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
    if calendar_name == "360_day":
        return 30

    if calendar_name in {"noleap", "365_day"}:
        month_lengths = (
            31, 28, 31, 30, 31, 30,
            31, 31, 30, 31, 30, 31,
        )
        return month_lengths[month - 1]

    if calendar_name in {"all_leap", "366_day"}:
        month_lengths = (
            31, 29, 31, 30, 31, 30,
            31, 31, 30, 31, 30, 31,
        )
        return month_lengths[month - 1]

    return calendar.monthrange(year, month)[1]


def assign_datetime_init_time(
    ds: xr.Dataset | xr.DataArray,
    init_time_dim: str = init_time_dim,
    calendar: str | None = None,
) -> xr.Dataset | xr.DataArray:
    """
    Replace an integer year coordinate with a datetime coordinate.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Input object containing an integer year coordinate.
    init_time_dim : str, optional
        Name of the initialization-time coordinate.
    calendar : str or None, optional
        CF calendar to use. If None, numpy.datetime64 is used.
        Otherwise a CFTimeIndex is created.

    Returns
    -------
    xr.Dataset or xr.DataArray
        Object with the integer year coordinate replaced by datetime
        values corresponding to January 1st of each year.
    """
    if init_time_dim not in ds.coords:
        raise ValueError(
            f"{init_time_dim!r} is not a coordinate."
        )

    years = np.asarray(ds.coords[init_time_dim].values)
    if years.ndim == 0:  
        years = np.array([years])

    if not np.issubdtype(years.dtype, np.integer):
        raise TypeError(
            f"{init_time_dim!r} must contain integer years."
        )

    if calendar is None:
        times = np.array(
            [np.datetime64(f"{year:04d}-01-01") for year in years ]
        )
    else:
        times = xr.date_range(
            start=f"{years.min():04d}-01-01",
            periods=len(years),
            freq="YS",
            calendar=calendar,
            use_cftime=True,
        )

    return ds.assign_coords(
        {init_time_dim: (init_time_dim, times)}
    )