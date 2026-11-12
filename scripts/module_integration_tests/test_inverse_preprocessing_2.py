import sys
import traceback
from pathlib import Path

import numpy as np
import xarray as xr

from S2D4P.preprocessing.selector import PreprocessingStepSelector
from S2D4P.preprocessing.utils_preprocessing import TrendRemover

DATA_PATH = Path(
    "/space/hall7/sitestore/eccc/crd/cccma/users/rpg002/data/tas/forecast_data/datetime"
)
MAX_SAMPLES = 100


def section(title: str) -> None:
    print(f"\n{'-' * 70}\n{title}\n{'-' * 70}")


def as_dataarray(obj) -> xr.DataArray:
    return obj["data"] if isinstance(obj, xr.Dataset) else obj


def max_abs_diff(a: xr.DataArray, b: xr.DataArray) -> float:
    """Largest absolute difference between a and b, with b aligned to a's dims."""
    av = np.asarray(a.values, dtype=np.float64)
    bv = np.asarray(b.transpose(*a.dims).values, dtype=np.float64)
    return float(np.max(np.abs(av - bv)))


def load_data(data_path: Path, max_samples: int) -> xr.Dataset:
    section("Loading data")
    files = sorted(data_path.glob("*.nc"))[:max_samples]
    print(f"Files read: {len(files)} from {data_path}")

    pieces = []
    for path in files:
        with xr.open_dataset(path) as ds:
            init = ds["year"].values[0].astype("datetime64[ns]")
            da = ds["tas"].load()
            pieces.append(da.expand_dims(time=[init]))

    combined = xr.concat(pieces, dim="time")
    print(f"Array dims: {dict(combined.sizes)}")
    print(f"Init times: {combined['time'].values[0]} to {combined['time'].values[-1]}")
    return xr.Dataset({"data": combined})


def run_one(name: str, data: xr.DataArray) -> None:
    section(
        f"{name.upper()} (preprocessor fitted on all {data.sizes['time']} init times)"
    )
    prep = PreprocessingStepSelector(name=name, args=None).get_preprocessor()
    prep.fit(data)

    print("Lead-time 0, one init time: transform then inverse_transform")
    sample = data.isel(time=[0], lead_time=[0])
    fwd = as_dataarray(prep.transform(sample))
    inv = as_dataarray(prep.inverse_transform(fwd))
    print(
        f"  Round-trip error, max|x - inverse(transform(x))|: "
        f"{max_abs_diff(sample, inv):.3e}"
    )
    print(
        f"  Forward change, max|transform(x) - x|:           "
        f"{max_abs_diff(sample, fwd):.3e}"
    )

    print("Fit data, all init times and lead times: transform only")
    full_out = as_dataarray(prep.transform(data))
    print(f"  Output mean over fit data: {float(full_out.mean()):+.6e}")
    print(f"  Output std over fit data:  {float(full_out.std()):.6e}")

    print("Late half of init times (not in the first half): transform only")
    n = data.sizes["time"]
    late_out = as_dataarray(prep.transform(data.isel(time=slice(n // 2, n))))
    print(f"  Output mean over late half: {float(late_out.mean()):+.6e}")
    print(f"  Output std over late half:  {float(late_out.std()):.6e}")


def run_trend(prep_ds: xr.Dataset) -> None:
    section("TREND_REMOVER (fitted on all init times)")
    trend = TrendRemover()
    trend.fit(prep_ds)
    data = prep_ds["data"]

    print("First init time, all lead times: transform then inverse_transform")
    sample = data.isel(time=[0])
    fwd = as_dataarray(trend.transform(sample))
    inv = as_dataarray(trend.inverse_transform(fwd))
    print(
        f"  Round-trip error, max|x - inverse(transform(x))|: "
        f"{max_abs_diff(sample, inv):.3e}"
    )

    lt = data.sizes["lead_time"]
    print(f"First init time, lead time {lt - 1} only: transform then inverse_transform")
    last = data.isel(time=[0], lead_time=[lt - 1])
    fwd_last = as_dataarray(trend.transform(last))
    inv_last = as_dataarray(trend.inverse_transform(fwd_last))
    print(
        f"  Round-trip error, max|x - inverse(transform(x))|: "
        f"{max_abs_diff(last, inv_last):.3e}"
    )


def main() -> int:
    prep_ds = load_data(DATA_PATH, MAX_SAMPLES)

    for name in ["standardizer", "anomalies", "normalizer"]:
        try:
            run_one(name, prep_ds["data"])
        except Exception:
            traceback.print_exc()

    try:
        run_trend(prep_ds)
    except Exception:
        traceback.print_exc()

    return 0


if __name__ == "__main__":
    sys.exit(main())
