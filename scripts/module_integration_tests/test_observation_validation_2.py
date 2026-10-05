from __future__ import annotations

import json
import logging
import numpy as np
import pandas as pd
import cftime
from pathlib import Path
from typing import Literal, Any, Optional
from datetime import datetime
from dataclasses import dataclass

# Import actual TrainDatasetConfig from cccma_ppp (for production use)
try:
    from cccma_ppp.train.dataset import TrainDatasetConfig
    from cccma_ppp.data_modules.utils import add_lead_times

    CCCMA_AVAILABLE = True
except ImportError:
    # For testing without full cccma_ppp installation, provide mock add_lead_times
    TrainDatasetConfig = None
    CCCMA_AVAILABLE = False

    def add_lead_times(init_times, lead_times, resolution="month"):
        """Mock implementation for testing when cccma_ppp unavailable."""
        result = []

        for init_t, lead_t in zip(init_times, lead_times):
            lead_t = int(lead_t)

            # Handle cftime datetime objects
            if isinstance(init_t, cftime.datetime):
                if resolution == "month":
                    # Add months to cftime datetime
                    new_month = init_t.month + lead_t
                    new_year = init_t.year + (new_month - 1) // 12
                    new_month = ((new_month - 1) % 12) + 1
                    result.append(
                        type(init_t)(
                            new_year,
                            new_month,
                            init_t.day,
                            init_t.hour,
                            init_t.minute,
                            init_t.second,
                        )
                    )
                else:  # day
                    # Add days to cftime datetime
                    ts = pd.Timestamp(str(init_t))
                    new_ts = ts + pd.Timedelta(days=lead_t)
                    result.append(
                        cftime.DatetimeGregorian(
                            new_ts.year,
                            new_ts.month,
                            new_ts.day,
                            new_ts.hour,
                            new_ts.minute,
                            new_ts.second,
                        )
                    )
            else:
                # Handle numpy datetime64
                if resolution == "month":
                    result.append(pd.Timestamp(init_t) + pd.DateOffset(months=lead_t))
                else:  # day
                    result.append(pd.Timestamp(init_t) + pd.Timedelta(days=lead_t))

        return np.array(result)


lead_time_unit = Literal["day", "month"]


@dataclass
class ValidationRange:
    """Configuration for observation time validation range."""

    min_lead_time: int
    max_lead_time: int
    resolution: lead_time_unit
    description: str = ""

    def __post_init__(self):
        if not self.description:
            self.description = (
                f"{self.min_lead_time} to {self.max_lead_time} {self.resolution}s"
            )


@dataclass
class ValidationConfig:
    """Configuration for observation validation."""

    output_dir: Path
    ranges: list[ValidationRange]
    sample_init_times: Optional[np.ndarray] = None
    sampling_dims: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict:
        """Convert config to JSON-serializable dict."""
        return {
            "output_dir": str(self.output_dir),
            "sampling_dims": self.sampling_dims,
            "ranges": [
                {
                    "min_lead_time": r.min_lead_time,
                    "max_lead_time": r.max_lead_time,
                    "resolution": r.resolution,
                    "description": r.description,
                }
                for r in self.ranges
            ],
            "num_sample_init_times": len(self.sample_init_times)
            if self.sample_init_times is not None
            else None,
        }


class ObservationValidator:
    """Validate observation times for dataset configurations."""

    def __init__(self, config: ValidationConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()
        self.results = {}

    def _setup_logger(self) -> logging.Logger:
        """Setup logging to file and console."""
        logger = logging.getLogger("ObservationValidator")
        logger.setLevel(logging.INFO)

        # File handler
        log_path = self.config.output_dir / "validation.log"
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger

    def validate_range(
        self,
        validation_range: ValidationRange,
        init_times: Optional[np.ndarray] = None,
    ) -> dict[str, Any]:
        """Validate observation times for a specific range."""
        self.logger.info(f"Validating range: {validation_range.description}")

        # Use provided init_times or config sample
        if init_times is None:
            if self.config.sample_init_times is None:
                self.logger.warning("No init_times provided and none in config")
                return {}
            init_times = self.config.sample_init_times

        # Create lead times for this range
        lead_times_range = np.arange(
            validation_range.min_lead_time, validation_range.max_lead_time + 1
        )

        # Tile/repeat lead_times to match length of init_times
        num_repeats = len(init_times) // len(lead_times_range)
        remainder = len(init_times) % len(lead_times_range)
        lead_times = np.tile(lead_times_range, num_repeats)
        if remainder > 0:
            lead_times = np.concatenate([lead_times, lead_times_range[:remainder]])

        try:
            # Calculate observation times using cccma_ppp function
            obs_times = add_lead_times(
                init_times, lead_times, validation_range.resolution
            )

            # Create validation dataframe
            data = []
            for init_t, obs_t, lead_t in zip(init_times, obs_times, lead_times):
                data.append(
                    {
                        "init_time": str(init_t),
                        "lead_time": int(lead_t),
                        "lead_time_resolution": validation_range.resolution,
                        "observation_time": str(obs_t),
                    }
                )

            df = pd.DataFrame(data)

            # Save CSV
            csv_path = (
                self.config.output_dir
                / f"validation_{validation_range.resolution}_{validation_range.min_lead_time}to{validation_range.max_lead_time}.csv"
            )
            df.to_csv(csv_path, index=False)

            result = {
                "range": validation_range.description,
                "resolution": validation_range.resolution,
                "min_lead_time": validation_range.min_lead_time,
                "max_lead_time": validation_range.max_lead_time,
                "num_samples": len(df),
                "unique_init_times": df["init_time"].nunique(),
                "unique_lead_times": df["lead_time"].nunique(),
                "output_file": str(csv_path),
                "first_row": df.iloc[0].to_dict() if len(df) > 0 else None,
                "last_row": df.iloc[-1].to_dict() if len(df) > 0 else None,
                "status": "success",
            }

            self.logger.info(f"Range validated: {len(df)} samples")
            return result

        except Exception as e:
            result = {
                "range": validation_range.description,
                "status": "failed",
                "error": str(e),
            }
            self.logger.error(f"Range validation failed: {e}")
            return result

    def validate_all_ranges(
        self,
        init_times: Optional[np.ndarray] = None,
    ) -> list[dict[str, Any]]:
        """Validate all configured ranges."""
        self.logger.info(f"Starting validation of {len(self.config.ranges)} ranges")
        self.logger.info(f"Sampling dimensions: {self.config.sampling_dims}")

        results = []
        for validation_range in self.config.ranges:
            result = self.validate_range(validation_range, init_times)
            results.append(result)
            self.results[validation_range.description] = result

        return results

    def generate_summary_report(self, results: list[dict]) -> dict[str, Any]:
        """Generate comprehensive summary report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config.to_dict(),
            "validation_results": results,
            "summary": {
                "total_ranges": len(results),
                "successful": sum(1 for r in results if r.get("status") == "success"),
                "failed": sum(1 for r in results if r.get("status") == "failed"),
            },
        }

        # Save report
        report_path = self.config.output_dir / "validation_report.json"
        with report_path.open("w") as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info(f"Summary: {report['summary']}")
        self.logger.info(f"Report saved to {report_path}")

        return report


class DatasetDimensionExtractor:
    """Extract sampling dimensions from actual TrainDatasetConfig."""

    @staticmethod
    def extract_from_traindatasetconfig(config: TrainDatasetConfig) -> dict[str, Any]:
        """
        Extract dimensions from actual TrainDatasetConfig object.

        Parameters
        ----------
        config : TrainDatasetConfig
            Actual TrainDatasetConfig instance from cccma_ppp

        Returns
        -------
        dict[str, Any]
            Dictionary containing model_config, observation_config,
            condition_config, lead_times_config, and their shapes
        """
        dims = {
            "model": config.model,
            "observation": config.observation,
            "condition": config.condition,
            "lead_times": config.lead_times,
            "condition_method": config.condition_method,
        }

        # Extract shapes/dimensions from each config
        if hasattr(config.model, "data_shape"):
            dims["model_shape"] = config.model.data_shape
        if hasattr(config.model, "coords"):
            dims["model_coords"] = config.model.coords

        if config.observation is not None:
            if hasattr(config.observation, "data_shape"):
                dims["observation_shape"] = config.observation.data_shape
            if hasattr(config.observation, "coords"):
                dims["observation_coords"] = config.observation.coords

        if config.condition is not None:
            if hasattr(config.condition, "data_shape"):
                dims["condition_shape"] = config.condition.data_shape
            if hasattr(config.condition, "coords"):
                dims["condition_coords"] = config.condition.coords

        return dims

    @staticmethod
    def infer_init_times(
        num_samples: int,
        start_date: str = "2020-01-01",
        frequency: str = "monthly",
    ) -> np.ndarray:
        """Infer initialization times from number of samples."""
        start = pd.Timestamp(start_date)

        if frequency == "monthly":
            return np.array(
                [start + pd.DateOffset(months=i) for i in range(num_samples)]
            )
        elif frequency == "daily":
            return np.array([start + pd.Timedelta(days=i) for i in range(num_samples)])
        else:
            raise ValueError(f"Unknown frequency: {frequency}")


def example_1_monthly_numpy():
    """Example 1: Monthly lead times with numpy datetime64."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Monthly Lead Times (numpy.datetime64)")
    print("=" * 60)

    # Create initialization times (every month for a year)
    init_times = np.arange("2023-01-01", "2024-01-01", dtype="datetime64[M]")

    # Create lead times (1 to 12 months)
    lead_times = np.arange(1, 13)

    # Calculate observation times using cccma_ppp function
    observation_times = add_lead_times(init_times, lead_times, "month")

    # Create DataFrame
    df = pd.DataFrame(
        {
            "init_time": pd.to_datetime(init_times),
            "lead_time": lead_times,
            "lead_time_resolution": "month",
            "observation_time": pd.to_datetime(observation_times),
        }
    )

    # Create output directory and save
    output_dir = Path("/home/rna002/hall7/output/observation_validation_2")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "obs_val_monthly_numpy.csv"

    df.to_csv(output_path, index=False)
    print(f"Observation validation CSV saved to: {output_path}")

    print("\nFirst 5 rows:")
    print(df.head())
    print(f"\nTotal rows: {len(df)}")
    return df


def example_2_daily_numpy():
    """Example 2: Daily lead times with numpy datetime64."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Daily Lead Times (numpy.datetime64)")
    print("=" * 60)

    # Create initialization times (daily for 30 days)
    init_times = np.arange("2023-01-01", "2023-01-31", dtype="datetime64[D]")

    # Create lead times (1 to 10 days, repeated)
    lead_times = np.tile(np.arange(1, 11), len(init_times) // 10 + 1)[: len(init_times)]

    # Calculate observation times using cccma_ppp function
    observation_times = add_lead_times(init_times, lead_times, "day")

    # Create DataFrame
    df = pd.DataFrame(
        {
            "init_time": pd.to_datetime(init_times),
            "lead_time": lead_times,
            "lead_time_resolution": "day",
            "observation_time": pd.to_datetime(observation_times),
        }
    )

    # Create output directory and save
    output_dir = Path("/home/rna002/hall7/output/observation_validation_2")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "obs_val_daily_numpy.csv"

    df.to_csv(output_path, index=False)
    print(f"Observation validation CSV saved to: {output_path}")

    print("\nFirst 10 rows:")
    print(df.head(10))
    print(f"\nTotal rows: {len(df)}")
    return df


def example_3_cftime():
    """Example 3: Custom with cftime.datetime objects."""
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

    # Calculate observation times using cccma_ppp function
    observation_times = add_lead_times(init_times, lead_times, "day")

    # Create DataFrame - handle cftime datetime objects
    init_times_col = [str(t) for t in init_times]
    observation_times_col = [str(t) for t in observation_times]

    df = pd.DataFrame(
        {
            "init_time": init_times_col,
            "lead_time": lead_times,
            "lead_time_resolution": "day",
            "observation_time": observation_times_col,
        }
    )

    # Create output directory and save
    output_dir = Path("/home/rna002/hall7/output/observation_validation_2")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "obs_val_cftime.csv"

    df.to_csv(output_path, index=False)
    print(f"Observation validation CSV saved to: {output_path}")

    print("\nGenerated CSV:")
    print(df)
    return df


def example_4_cftime_noleap():
    """Example 4: cftime.datetime with noleap (365-day) calendar."""
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

    # Calculate observation times using cccma_ppp function
    observation_times = add_lead_times(init_times, lead_times, "month")

    # Create DataFrame - handle cftime datetime objects
    init_times_col = [str(t) for t in init_times]
    observation_times_col = [str(t) for t in observation_times]

    df = pd.DataFrame(
        {
            "init_time": init_times_col,
            "lead_time": lead_times,
            "lead_time_resolution": "month",
            "observation_time": observation_times_col,
        }
    )

    # Create output directory and save
    output_dir = Path("/home/rna002/hall7/output/observation_validation_2")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "obs_val_cftime_noleap.csv"

    df.to_csv(output_path, index=False)
    print(f"Observation validation CSV saved to: {output_path}")

    print("\nGenerated CSV:")
    print(df)
    return df


def example_5_validation_ranges_40month():
    """Example 5: Comprehensive 40-month validation ranges."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: 40-Month Validation Ranges")
    print("=" * 70)

    # Create initialization times (40 months)
    init_times = np.arange("2020-01-01", "2023-05-01", dtype="datetime64[M]")

    # Define multiple validation ranges for 40-month period
    ranges = [
        ValidationRange(1, 1, "month", "Lead time 1 month"),
        ValidationRange(1, 12, "month", "Full Year (12 months)"),
        ValidationRange(1, 24, "month", "2 Years (24 months)"),
        ValidationRange(1, 40, "month", "Full 40-month range"),
    ]

    sampling_dims = {
        "temporal_steps": 40,
        "model": "MonthlyForecast",
        "resolution": "monthly",
    }

    config = ValidationConfig(
        output_dir=Path("/home/rna002/hall7/output/observation_validation_2"),
        ranges=ranges,
        sample_init_times=init_times,
        sampling_dims=sampling_dims,
    )

    validator = ObservationValidator(config)
    results = validator.validate_all_ranges()
    report = validator.generate_summary_report(results)

    print("\nValidation Summary:")
    print(f"  Total ranges: {report['summary']['total_ranges']}")
    print(f"  Successful: {report['summary']['successful']}")
    print(f"  Failed: {report['summary']['failed']}")

    return report


def example_6_validation_ranges_60day():
    """Example 6: Comprehensive 60-day validation ranges."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: 60-Day Validation Ranges")
    print("=" * 70)

    # Create initialization times (60 days)
    init_times = np.arange("2023-01-01", "2023-03-01", dtype="datetime64[D]")

    # Define multiple validation ranges for 60-day period
    ranges = [
        ValidationRange(1, 7, "day", "Weekly (7 days)"),
        ValidationRange(1, 14, "day", "Bi-weekly (14 days)"),
        ValidationRange(1, 30, "day", "Monthly (30 days)"),
        ValidationRange(1, 60, "day", "Full 60-day range"),
    ]

    sampling_dims = {
        "temporal_steps": 60,
        "model": "DailyForecast",
        "resolution": "daily",
    }

    config = ValidationConfig(
        output_dir=Path("/home/rna002/hall7/output/observation_validation_2"),
        ranges=ranges,
        sample_init_times=init_times,
        sampling_dims=sampling_dims,
    )

    validator = ObservationValidator(config)
    results = validator.validate_all_ranges()
    report = validator.generate_summary_report(results)

    print("\nValidation Summary:")
    print(f"  Total ranges: {report['summary']['total_ranges']}")
    print(f"  Successful: {report['summary']['successful']}")
    print(f"  Failed: {report['summary']['failed']}")

    return report


def example_7_validation_ranges_mixed():
    """Example 7: Mixed resolution validation ranges."""
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Mixed Resolution Validation Ranges")
    print("=" * 70)

    # Create initialization times (40 months)
    init_times = np.arange("2020-01-01", "2023-05-01", dtype="datetime64[M]")

    # Define validation ranges with different time steps
    ranges = [
        ValidationRange(1, 3, "month", "Seasonal (3 months)"),
        ValidationRange(1, 6, "month", "Half-year (6 months)"),
        ValidationRange(1, 12, "month", "Annual (12 months)"),
        ValidationRange(1, 24, "month", "2-Year (24 months)"),
        ValidationRange(1, 40, "month", "Full 40-month range"),
    ]

    sampling_dims = {
        "temporal_steps": 40,
        "model": "MixedResolution",
        "resolutions": ["monthly"],
    }

    config = ValidationConfig(
        output_dir=Path("/home/rna002/hall7/output/observation_validation_2"),
        ranges=ranges,
        sample_init_times=init_times,
        sampling_dims=sampling_dims,
    )

    validator = ObservationValidator(config)
    results = validator.validate_all_ranges()
    report = validator.generate_summary_report(results)

    print("\nValidation Summary:")
    print(f"  Total ranges: {report['summary']['total_ranges']}")
    print(f"  Successful: {report['summary']['successful']}")
    print(f"  Failed: {report['summary']['failed']}")

    return report


def example_8_validation_ranges_advanced():
    """Example 8: Advanced multi-range validation with complex lead times."""
    print("\n" + "=" * 70)
    print("EXAMPLE 8: Advanced Multi-Range Validation")
    print("=" * 70)

    # Create initialization times
    init_times = np.arange("2020-01-01", "2023-05-01", dtype="datetime64[M]")

    # Define comprehensive validation ranges
    ranges = [
        ValidationRange(1, 1, "month", "Immediate (LT=1)"),
        ValidationRange(1, 3, "month", "Short-term (LT=1-3)"),
        ValidationRange(1, 6, "month", "Medium-term (LT=1-6)"),
        ValidationRange(1, 12, "month", "Annual (LT=1-12)"),
        ValidationRange(1, 24, "month", "2-Year (LT=1-24)"),
        ValidationRange(13, 24, "month", "Year 2 only (LT=13-24)"),
        ValidationRange(25, 40, "month", "Years 3-4 (LT=25-40)"),
        ValidationRange(1, 40, "month", "Full 40-month range"),
    ]

    sampling_dims = {
        "temporal_steps": 40,
        "model": "AdvancedValidation",
        "resolutions": ["monthly"],
        "analysis_type": "comprehensive",
    }

    config = ValidationConfig(
        output_dir=Path("/home/rna002/hall7/output/observation_validation_2"),
        ranges=ranges,
        sample_init_times=init_times,
        sampling_dims=sampling_dims,
    )

    validator = ObservationValidator(config)
    results = validator.validate_all_ranges()
    report = validator.generate_summary_report(results)

    print("\nValidation Summary:")
    print(f"  Total ranges: {report['summary']['total_ranges']}")
    print(f"  Successful: {report['summary']['successful']}")
    print(f"  Failed: {report['summary']['failed']}")

    # Print detailed results
    print("\nDetailed Range Results:")
    for result in report["validation_results"]:
        if result.get("status") == "success":
            print(
                f"  {result['range']:30s} | Samples: {result['num_samples']:4d} | File: {Path(result['output_file']).name}"
            )

    return report


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("OBSERVATION VALIDATION WITH COMPREHENSIVE RANGES")
    print("=" * 70)

    # Run original 4 examples
    example_1_monthly_numpy()
    example_2_daily_numpy()
    example_3_cftime()
    example_4_cftime_noleap()

    # Run new comprehensive range examples
    example_5_validation_ranges_40month()
    example_6_validation_ranges_60day()
    example_7_validation_ranges_mixed()
    example_8_validation_ranges_advanced()

    print("\n" + "=" * 70)
    print("All validation examples completed successfully!")
    print("=" * 70)
