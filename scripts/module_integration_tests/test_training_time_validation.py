import numpy as np
import xarray as xr
import cftime
import pandas as pd
from pathlib import Path

from cccma_ppp.data_modules.data.data_configs import (
    ModelDataConfig,
    ObsDataConfig,
)
from cccma_ppp.train.dataset import TrainDatasetConfig
from cccma_ppp.data_modules.utils import add_lead_times


def create_synthetic_model_data(output_path: str, years: list = None) -> xr.Dataset:
    """Create synthetic model data from years 1960-2000."""
    if years is None:
        years = list(range(1960, 2001))

    print("\n" + "=" * 80)
    print("CREATING SYNTHETIC MODEL DATA (1960-2000)")
    print("=" * 80)

    # Create initialization times (monthly)
    dates = []
    for year in years:
        for month in range(1, 13):
            dates.append(cftime.DatetimeGregorian(year, month, 1, 0, 0, 0))

    init_times = np.array(dates, dtype=object)

    # Create lead times (1-6 months)
    lead_times = np.arange(1, 7)

    # Create spatial grid
    lats = np.linspace(-90, 90, 10)
    lons = np.linspace(-180, 180, 20)

    # Create random temperature data
    np.random.seed(42)
    temp_data = np.random.normal(
        280, 10, (len(init_times), len(lead_times), len(lats), len(lons))
    )

    # Create dataset with time dimension (as expected by cccma_ppp)
    ds = xr.Dataset(
        {
            "tas": (["time", "lead_time", "lat", "lon"], temp_data),
        },
        coords={
            "time": init_times,
            "lead_time": lead_times,
            "lat": lats,
            "lon": lons,
        },
    )

    # Add attributes
    ds["tas"].attrs["long_name"] = "Surface Air Temperature"
    ds["tas"].attrs["units"] = "K"
    ds["time"].attrs["long_name"] = "Initialization Time"
    ds["lead_time"].attrs["long_name"] = "Lead Time"

    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save
    ds.to_netcdf(output_path)
    print("\nModel data saved")
    print(f"  Path: {output_path}")
    print(f"  Initialization times: {len(init_times)} (1960-01 to 2000-12)")
    print(f"  Lead times: {len(lead_times)} months")
    print(f"  Spatial grid: {len(lats)} x {len(lons)}")

    return ds


def create_synthetic_observations(output_path: str, dates: list = None) -> xr.Dataset:
    """Create synthetic observation data at specific dates."""
    print("\n" + "=" * 80)
    print("CREATING SYNTHETIC OBSERVATIONS")
    print("=" * 80)

    if dates is None:
        dates = []

        # Before 2009: 1960-2008
        for year in range(1960, 2009):
            for month in range(1, 13):
                dates.append(cftime.DatetimeGregorian(year, month, 1, 0, 0, 0))

        # 2009 June
        dates.append(cftime.DatetimeGregorian(2009, 6, 1, 0, 0, 0))

        # 2015: all months
        for month in range(1, 13):
            dates.append(cftime.DatetimeGregorian(2015, month, 1, 0, 0, 0))

        # 2023: all months
        for month in range(1, 13):
            dates.append(cftime.DatetimeGregorian(2023, month, 1, 0, 0, 0))

    obs_times = np.array(dates, dtype=object)

    # Create spatial grid (same as model)
    lats = np.linspace(-90, 90, 10)
    lons = np.linspace(-180, 180, 20)

    # Create random observation data
    np.random.seed(43)
    obs_data = np.random.normal(280, 10, (len(obs_times), len(lats), len(lons)))

    # Create dataset with time dimension (as expected by cccma_ppp)
    ds_obs = xr.Dataset(
        {
            "tas": (["time", "lat", "lon"], obs_data),
        },
        coords={
            "time": obs_times,
            "lat": lats,
            "lon": lons,
        },
    )

    # Add attributes
    ds_obs["tas"].attrs["long_name"] = "Observed Surface Air Temperature"
    ds_obs["tas"].attrs["units"] = "K"
    ds_obs["time"].attrs["long_name"] = "Observation Time"

    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save
    ds_obs.to_netcdf(output_path)

    print("\nObservation data saved")
    print(f"  Path: {output_path}")
    print(f"  Observation times: {len(obs_times)}")
    print("  Coverage:")
    print(f"    - Before 2009: 1960-2008 ({49 * 12} months)")
    print("    - June 2009: 1 month")
    print("    - Year 2015: 12 months")
    print("    - Year 2023: 12 months")

    return ds_obs


def show_available_times(config: TrainDatasetConfig):
    """Show available training times from TrainDatasetConfig."""
    print("\n" + "=" * 80)
    print("AVAILABLE TIMES FROM TrainDatasetConfig")
    print("=" * 80)

    if hasattr(config, "available_times"):
        available_times = config.available_times

        print(f"\nAvailable times found: {len(available_times)}")
        print(f"  Type: {type(available_times)}")
        print(f"  Range: {available_times[0]} to {available_times[-1]}")

        print("\n  First 5:")
        for i, t in enumerate(available_times[:5]):
            print(f"    {i}: {t}")

        if len(available_times) > 10:
            print("    ...")
            print("\n  Last 5:")
            for i, t in enumerate(available_times[-5:]):
                print(f"    {len(available_times) - 5 + i}: {t}")

        return available_times
    else:
        print("No available_times attribute")
        return None


def show_lead_time_calculation(init_times, lead_times_list):
    """Show how add_lead_times calculates observation times."""
    print("\n" + "=" * 80)
    print("LEAD TIME CALCULATION (using add_lead_times)")
    print("=" * 80)

    try:
        print("\nFormula: observation_time = init_time + (lead_time - 1) months")
        print(f"\n{'Init Time':<20} {'Lead':<6} {'Observation Time':<20}")
        print("-" * 80)

        # Show a few examples with each lead time
        for init_t in init_times[:5]:
            for lead_t in lead_times_list[:6]:
                # Convert pandas Timestamp to numpy datetime64
                init_t_np = np.datetime64(pd.Timestamp(init_t))

                # Use the library's add_lead_times function
                obs_time = add_lead_times(
                    np.array([init_t_np]), np.array([lead_t]), "month"
                )[0]

                print(f"{str(init_t):<20} {lead_t:<6} {str(obs_time):<20}")

    except Exception as e:
        print(f"Error: {e}")


def show_training_dataset_info(config: TrainDatasetConfig):
    """Show information about the TrainDataset."""
    print("\n" + "=" * 80)
    print("TRAINING DATASET INFORMATION")
    print("=" * 80)

    try:
        # Don't create TrainDataset - just show info from config
        available_times = config.available_times
        lead_times_arr = np.array(config.lead_times)

        print("\nTrainDatasetConfig information:")
        print(f"  Available init times: {len(available_times)}")
        print(f"  Lead times: {config.lead_times}")
        print(
            f"  Expected samples: {len(available_times)} init_times × {len(config.lead_times)} lead_times"
        )
        print(
            f"  = {len(available_times) * len(config.lead_times)} total training combinations"
        )

        print("\n  Sample training pairs (first 12):")
        print(f"  {'Pair':<8} {'Init Time':<20} {'Lead':<6} {'Obs Time':<20}")
        print("-" * 80)

        # Show first 12 combinations
        pair_count = 0
        for init_idx, init_time in enumerate(available_times[:3]):
            for lead_idx, lead_time in enumerate(lead_times_arr[:4]):
                # Convert pandas Timestamp to numpy datetime64
                init_time_np = np.datetime64(pd.Timestamp(init_time))

                # Calculate observation time
                obs_time = add_lead_times(
                    np.array([init_time_np]), np.array([lead_time]), "month"
                )[0]

                print(
                    f"  {pair_count:<8} {str(init_time):<20} {lead_time:<6} {str(obs_time):<20}"
                )
                pair_count += 1

        print(
            f"\n  ... and {len(available_times) * len(config.lead_times) - 12} more training pairs"
        )

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


def main(
    output_dir: str = "/fs/site7/eccc/crd/cccma/users/rna002/output/training_time_validation",
    model_filename: str = "model_data_1960_2000.nc",
    obs_filename: str = "observations.nc",
):
    """Main function - creates data then shows training times using library."""

    print("\n" + "=" * 80)
    print("TRAINING TIMES VALIDATION - COMPLETE")
    print("=" * 80)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build full paths
    model_path = str(output_dir / model_filename)
    obs_path = str(output_dir / obs_filename)

    create_synthetic_model_data(model_path)
    create_synthetic_observations(obs_path)

    print("\n" + "=" * 80)
    print("LOADING WITH cccma_ppp LIBRARY")
    print("=" * 80)

    try:
        # Create model data config
        print("\nCreating ModelDataConfig...")
        print(f"  Directory: {output_dir}")

        model_config = ModelDataConfig(
            paths=str(output_dir),
            file_type="model_data_*.nc",
            names=["tas"],
        )
        print("ModelDataConfig created")

        # Create observation data config
        print("\nCreating ObsDataConfig...")

        obs_config = ObsDataConfig(
            paths=str(output_dir),
            file_type="observations*.nc",
            names=["tas"],
        )
        print("ObsDataConfig created")

        # Create TrainDatasetConfig
        print("\nCreating TrainDatasetConfig...")
        lead_times = [1, 2, 3, 4, 5, 6]

        config = TrainDatasetConfig(
            model=model_config,
            observation=obs_config,
            condition_method=None,
            lead_times=lead_times,
        )
        print("TrainDatasetConfig created")
        print(f"  Lead times: {lead_times}")

        available_times = show_available_times(config)

        if available_times is not None and len(available_times) > 0:
            # Show lead time calculation
            show_lead_time_calculation(available_times, lead_times)

        # Show training dataset info
        show_training_dataset_info(config)

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print("\nSynthetic data created:")
        print(f"  Model: {model_path}")
        print(f"  Observations: {obs_path}")
        print("\ncccma_ppp Library used:")
        print("  ModelDataConfig: loads model data from directory")
        print("  ObsDataConfig: loads observation data from directory")
        print("  TrainDatasetConfig: combines them and finds common times")
        print("  TrainDataset: creates (forecast, observation) training pairs")
        print(
            "  add_lead_times(): calculates observation_time from init_time + lead_time"
        )
        print("\nTraining time formula:")
        print("  observation_time = init_time + (lead_time - 1) months")
        print("\n" + "=" * 80 + "\n")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = (
            "/fs/site7/eccc/crd/cccma/users/rna002/output/training_time_validation"
        )

    main(output_dir)
