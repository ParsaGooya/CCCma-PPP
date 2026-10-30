import numpy as np
import pandas as pd
import cftime
from pathlib import Path
from typing import Literal

lead_time_unit = Literal["day", "month"]

from S2D4P.data.utils import add_lead_times


def create_observation_validation_csv(
    init_times: np.ndarray,
    lead_times: np.ndarray,
    lead_time_resolution: lead_time_unit = "month",
    output_path: str = "/fs/site7/eccc/crd/cccma/users/rna002/output/observation_validation/observation_validation.csv",
) -> pd.DataFrame:
    """
    Create a CSV file for observation validation using cccma_ppp add_lead_times.

    Parameters
    ----------
    init_times : np.ndarray
        Array of initialization times
    lead_times : np.ndarray
        Array of lead times (1-based)
    lead_time_resolution : lead_time_unit
        Resolution of lead time: "day" or "month"
    output_path : str
        Path to save the CSV file

    Returns
    -------
    pd.DataFrame
        DataFrame containing the validation data
    """
    # Calculate observation times using cccma_ppp function
    observation_times = add_lead_times(init_times, lead_times, lead_time_resolution)

    # Create DataFrame - handle both numpy.datetime64 and cftime.datetime
    # Convert init_times
    if init_times.dtype == object and isinstance(init_times[0], cftime.datetime):
        init_times_col = [str(t) for t in init_times]
    else:
        init_times_col = pd.to_datetime(init_times)

    # Convert observation_times
    if observation_times.dtype == object and isinstance(
        observation_times[0], cftime.datetime
    ):
        observation_times_col = [str(t) for t in observation_times]
    else:
        observation_times_col = pd.to_datetime(observation_times)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "init_time": init_times_col,
            "lead_time": lead_times,
            "lead_time_resolution": lead_time_resolution,
            "observation_time": observation_times_col,
        }
    )

    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Observation validation CSV saved to: {output_path}")

    return df


# Example 1: Monthly lead times with numpy datetime64
def example_monthly_numpy():
    """Example with monthly lead times using numpy datetime64."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Monthly Lead Times (numpy.datetime64)")
    print("=" * 60)

    # Create initialization times (every month for a year)
    init_times = np.arange("2023-01-01", "2024-01-01", dtype="datetime64[M]")

    # Create lead times (1 to 12 months)
    lead_times = np.arange(1, 13)

    df = create_observation_validation_csv(
        init_times=init_times,
        lead_times=lead_times,
        lead_time_resolution="month",
        output_path="/fs/site7/eccc/crd/cccma/users/rna002/output/observation_validation/obs_val_monthly_numpy.csv",
    )

    print("\nFirst 5 rows:")
    print(df.head())
    print(f"\nTotal rows: {len(df)}")
    return df


# Example 2: Daily lead times
def example_daily_numpy():
    """Example with daily lead times using numpy datetime64."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Daily Lead Times (numpy.datetime64)")
    print("=" * 60)

    # Create initialization times (daily for 30 days)
    init_times = np.arange("2023-01-01", "2023-01-31", dtype="datetime64[D]")

    # Create lead times (1 to 10 days, repeated)
    lead_times = np.tile(np.arange(1, 11), len(init_times) // 10 + 1)[: len(init_times)]

    df = create_observation_validation_csv(
        init_times=init_times,
        lead_times=lead_times,
        lead_time_resolution="day",
        output_path="/fs/site7/eccc/crd/cccma/users/rna002/output/observation_validation/obs_val_daily_numpy.csv",
    )

    print("\nFirst 10 rows:")
    print(df.head(10))
    print(f"\nTotal rows: {len(df)}")
    return df


# Example 3: Custom with cftime.datetime
def example_cftime():
    """Example with cftime.datetime objects (climate model calendars)."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Custom Initialization with cftime.datetime")
    print("=" * 60)

    # Create cftime datetime objects (standard calendar)
    init_times = np.array(
        [
            cftime.DatetimeGregorian(2023, 1, 15, 0, 0, 0),
            cftime.DatetimeGregorian(2023, 2, 20, 0, 0, 0),
            cftime.DatetimeGregorian(2023, 3, 10, 0, 0, 0),
            cftime.DatetimeGregorian(2023, 4, 5, 0, 0, 0),
            cftime.DatetimeGregorian(2023, 5, 12, 0, 0, 0),
        ],
        dtype=object,
    )

    # Custom lead times
    lead_times = np.array([1, 3, 5, 2, 4])

    df = create_observation_validation_csv(
        init_times=init_times,
        lead_times=lead_times,
        lead_time_resolution="day",
        output_path="/fs/site7/eccc/crd/cccma/users/rna002/output/observation_validation/obs_val_cftime.csv",
    )

    print("\nGenerated CSV:")
    print(df)
    return df


# Example 4: Monthly with cftime.datetime (non-standard calendar)
def example_cftime_noleap():
    """Example with cftime.datetime using noleap (365-day) calendar."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: cftime.datetime with noleap Calendar")
    print("=" * 60)

    # Create cftime datetime objects (noleap calendar - no February 29)
    init_times = np.array(
        [
            cftime.DatetimeNoLeap(2023, 1, 1, 0, 0, 0),
            cftime.DatetimeNoLeap(2023, 2, 1, 0, 0, 0),
            cftime.DatetimeNoLeap(2023, 3, 1, 0, 0, 0),
            cftime.DatetimeNoLeap(2023, 4, 1, 0, 0, 0),
            cftime.DatetimeNoLeap(2023, 5, 1, 0, 0, 0),
            cftime.DatetimeNoLeap(2023, 6, 1, 0, 0, 0),
        ],
        dtype=object,
    )

    # Lead times: 1 to 6 months
    lead_times = np.arange(1, 7)

    df = create_observation_validation_csv(
        init_times=init_times,
        lead_times=lead_times,
        lead_time_resolution="month",
        output_path="/fs/site7/eccc/crd/cccma/users/rna002/output/observation_validation/obs_val_cftime_noleap.csv",
    )

    print("\nGenerated CSV:")
    print(df)
    return df


if __name__ == "__main__":
    example_monthly_numpy()
    example_daily_numpy()
    example_cftime()
    example_cftime_noleap()

    print("\n" + "=" * 60)
    print("All CSV files created successfully!")
    print("=" * 60)
