import copy
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import yaml

from cccma_ppp.data_modules.data.data_configs import (
    ModelDataConfig,
    ObsDataConfig,
)
from cccma_ppp.train.dataset import TrainDatasetConfig
from cccma_ppp.preprocessing.selector import PreprocessingStepSelector
from cccma_ppp.preprocessing.preprocessing import PreprocessingPipeline
from cccma_ppp.data_modules.utils import _create_train_mask

BASE_CONFIG = (
    "scripts/integration_suite_train_config.yaml"  # template YAML for all cases
)
BASE_DIR = Path(
    "/fs/site7/eccc/crd/cccma/users/rna002/output/training_time_validation_2_2_annual_init_120"
)  # where dummy data is written
OUTPUT_ROOT = BASE_DIR  # where validation outputs are written, one folder per case

MODEL_VAR = "tas"  # model variable name; must match the YAML "names" entry
OBS_VAR = "tas"  # obs variable name; must match the YAML "names" entry
N_LAT, N_LON = 4, 4  # small grid to keep files tiny

# Each case: (name, model_start, model_end, obs_start, obs_end).
# Model inits are decadal (10YS) between the model dates; obs are monthly (MS).
CASES = [
    ("case_obs_pre1995", "1980-01-01", "2000-01-01", "1980-01-01", "1995-12-01"),
    ("case_obs_2000jun", "1980-01-01", "2000-01-01", "1980-01-01", "2000-06-01"),
    (
        "case_obs_2005mar",
        "1980-01-01",
        "2000-01-01",
        "1980-01-01",
        "2005-03-01",
    ),
    ("case_obs_post2015", "1980-01-01", "2000-01-01", "1980-01-01", "2015-12-01"),
]


def _coverage(valid: int, total: int) -> float:
    """Percent of valid pairs. Returns 0.0 for an empty mask instead of dividing by zero."""
    return 100.0 * valid / total if total else 0.0


def make_grid():
    """Return a small regular lat/lon grid for the dummy data."""
    lat = np.linspace(-45, 45, N_LAT)
    lon = np.linspace(0, 90, N_LON)
    return lat, lon


def make_model(start: str, end: str, lat, lon) -> xr.Dataset:
    """Build dummy decadal model output with dims (time, lead_time, lat, lon)."""
    init_times = pd.date_range(start, end, freq="YS")  # one init every 1 year
    lead_times = np.arange(1, 121)  # leads 1-120 months
    data = np.random.default_rng(0).normal(  # seeded so runs are reproducible
        size=(len(init_times), len(lead_times), N_LAT, N_LON)
    )
    return xr.Dataset(
        {MODEL_VAR: (("time", "lead_time", "lat", "lon"), data)},
        coords={
            "time": init_times,
            "lead_time": lead_times,
            "lat": lat,
            "lon": lon,
        },
    )


def make_obs(start: str, end: str, lat, lon) -> xr.Dataset:
    """Build dummy monthly observations with dims (time, lat, lon)."""
    times = pd.date_range(start, end, freq="MS")  # first day of each month
    data = np.random.default_rng(1).normal(
        size=(len(times), N_LAT, N_LON)
    )  # separate seed from model
    return xr.Dataset(
        {OBS_VAR: (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lat, "lon": lon},
    )


def generate_cases() -> dict:
    """Write the model and obs netCDF files for every case."""
    lat, lon = make_grid()
    case_dirs = {}
    for name, m_start, m_end, o_start, o_end in CASES:
        case_dir = BASE_DIR / name
        model_dir, obs_dir = case_dir / "model", case_dir / "obs"
        model_dir.mkdir(parents=True, exist_ok=True)
        obs_dir.mkdir(parents=True, exist_ok=True)

        model_ds = make_model(m_start, m_end, lat, lon)
        obs_ds = make_obs(o_start, o_end, lat, lon)
        model_ds.to_netcdf(model_dir / f"{name}_model.nc")
        obs_ds.to_netcdf(obs_dir / f"{name}_obs.nc")

        print(
            f"[{name}] model: {len(model_ds.time)} init steps, "
            f"obs: {len(obs_ds.time)} steps"
        )
        case_dirs[name] = case_dir
    return case_dirs


def load_config(config_path: str) -> dict:
    """Load a YAML config file into a dict."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(output_dir: Path) -> logging.Logger:
    """Send logs to a per-case file and to the console."""
    log_file = output_dir / "training_validation.log"
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def _build_pipeline(section: dict) -> PreprocessingPipeline:
    """Build a preprocessing pipeline from a model or obs YAML section."""
    preprocessors_list = section.get("preprocessing_pipeline", {}).get(
        "preprocessors_list", []
    )
    steps = [
        PreprocessingStepSelector(name=p["name"], args=p.get("args"))
        for p in preprocessors_list
    ]
    return PreprocessingPipeline(steps)


def create_config_from_yaml(config_dict: dict) -> TrainDatasetConfig:
    """Build the model, obs, and training dataset configs from the YAML dict."""
    logger = logging.getLogger(__name__)
    dataset_dict = config_dict["train_loader"]["dataset_config"]

    model_dict = dataset_dict["model"]
    model_config = ModelDataConfig(
        paths=model_dict["paths"],
        file_type="*.nc",
        names=model_dict["names"],
        preprocessing_pipeline=_build_pipeline(model_dict),
        ensemble_mean=model_dict.get("ensemble_mean", False),
        rename_dict=dict(
            model_dict.get("rename_dict", {})
        ),  # copy so the YAML dict is not mutated
    )
    logger.info("ModelDataConfig created")

    obs_dict = dataset_dict["observation"]
    obs_config = ObsDataConfig(
        paths=obs_dict["paths"],
        file_type="*.nc",
        names=obs_dict["names"],
        preprocessing_pipeline=_build_pipeline(obs_dict),
    )
    logger.info("ObsDataConfig created")

    lt = dataset_dict["lead_times"]
    lead_times = list(range(lt["start"], lt["end"] + 1))  # end is inclusive
    train_config = TrainDatasetConfig(
        model=model_config,
        observation=obs_config,
        condition_method=dataset_dict.get("condition_method", "ensemble_mean"),
        lead_times=lead_times,
    )
    logger.info("TrainDatasetConfig created")
    return train_config


def verify_and_print_times(config: TrainDatasetConfig):
    """Log the common and available time sets, and return both."""
    logger = logging.getLogger(__name__)
    common_times = config.get_common_time
    available_times = config.available_times

    logger.info("COMMON TIMES (intersection of model and obs):")
    logger.info(f"   Count: {len(common_times)}")
    if len(common_times):
        logger.info(f"   Range: {common_times[0]} to {common_times[-1]}")

    logger.info("AVAILABLE TIMES (for training):")
    logger.info(f"   Count: {len(available_times)}")
    if len(available_times):
        logger.info(f"   Range: {available_times[0]} to {available_times[-1]}")

    logger.info("COMMON TIMES (first 5 and last 5):")
    for i, t in enumerate(common_times):
        if i < 5 or i >= len(common_times) - 5:
            logger.info(f"   [{i}] {t}")
        elif i == 5:
            logger.info(f"   ... ({len(common_times) - 10} more) ...")

    logger.info("AVAILABLE TIMES (first 5 and last 5):")
    for i, t in enumerate(available_times):
        if i < 5 or i >= len(available_times) - 5:
            logger.info(f"   [{i}] {t}")
        elif i == 5:
            logger.info(f"   ... ({len(available_times) - 10} more) ...")

    return common_times, available_times


def create_training_mask(config, available_times, output_dir: Path) -> np.ndarray:
    """Build the boolean training mask and save it as training_mask.npy."""
    logger = logging.getLogger(__name__)
    model_init_times = config.get_common_time.to_series().index.unique()
    lead_times = list(config.lead_times)

    logger.info(f"Model init times: {len(model_init_times)}")
    logger.info(f"Lead times: {len(lead_times)}")
    logger.info(f"Total pairs: {len(model_init_times) * len(lead_times)}")

    # _create_train_mask returns True where a target is invalid, so invert it.
    model_valid = ~_create_train_mask(
        init_times=model_init_times.values,
        lead_times=lead_times,
        lead_time_resolution="month",  # check the allowed values of lead_time_unit
    ).values

    # The function never sees available_times, so keep the obs check separately.
    available = set(
        pd.DatetimeIndex(available_times)
    )  # set gives fast membership tests
    obs_valid = np.zeros_like(model_valid)
    for i, init_time in enumerate(model_init_times):
        for j, lead in enumerate(lead_times):
            obs_valid[i, j] = init_time + pd.DateOffset(months=int(lead)) in available

    mask = model_valid & obs_valid

    valid = int(mask.sum())
    logger.info(f"Valid pairs: {valid}")
    logger.info(f"Invalid pairs: {mask.size - valid}")
    logger.info(f"Coverage: {_coverage(valid, mask.size):.1f}%")

    np.save(output_dir / "training_mask.npy", mask)
    return mask


def plot_training_mask(mask: np.ndarray, output_dir: Path, lead_times=None) -> None:
    """Save the mask as training_mask.png. Errors are logged, not raised."""
    logger = logging.getLogger(__name__)
    try:
        fig, ax = plt.subplots(figsize=(16, 10))
        im = ax.imshow(
            mask,
            cmap="binary",
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )  # white = valid, black = masked
        ax.set_title(
            "Training Time Mask\n(white=valid sample, black=masked)",
            fontsize=16,
            fontweight="bold",
        )
        ax.set_ylabel("Initialization Time Index", fontsize=12)
        ax.set_xlabel("Lead Time Index", fontsize=12)
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Valid (1) / Masked (0)", fontsize=11)

        if lead_times is not None and len(lead_times) == mask.shape[1]:
            step = max(1, len(lead_times) // 20)  # label about 20 leads, not all 120
            ticks = np.arange(0, len(lead_times), step)
            ax.set_xticks(ticks)
            ax.set_xticklabels(
                [str(lead_times[i]) for i in ticks], rotation=45, ha="right"
            )

        valid = int(mask.sum())
        ax.text(
            0.02,
            0.98,
            f"Valid pairs: {valid:,}\nTotal pairs: {mask.size:,}\n"
            f"Coverage: {_coverage(valid, mask.size):.1f}%",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round", facecolor="wheat", alpha=0.8, edgecolor="black"
            ),
        )
        fig.tight_layout()
        fig.savefig(output_dir / "training_mask.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved plot: {output_dir / 'training_mask.png'}")
    except Exception as e:
        logger.error(f"Error plotting mask: {e}")


def validate_case(config_path: Path, output_dir: Path) -> None:
    """Run the full training-time validation for one case and write its outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    logger.info("TRAINING TIME VALIDATION")
    logger.info(f"Config: {config_path}")

    config_dict = load_config(str(config_path))
    train_config = create_config_from_yaml(config_dict)
    common_times, available_times = verify_and_print_times(train_config)

    lead_times = list(train_config.lead_times)
    mask = create_training_mask(train_config, available_times, output_dir)
    plot_training_mask(mask, output_dir, lead_times=lead_times)

    report = {  # summary written to training_validation_report.json
        "timestamp": datetime.now().isoformat(),
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "times": {
            "common_times_count": len(common_times),
            "common_times_range": [str(common_times[0]), str(common_times[-1])]
            if len(common_times)
            else [],  # empty when model and obs don't overlap
            "available_times_count": len(available_times),
            "available_times_range": [str(available_times[0]), str(available_times[-1])]
            if len(available_times)
            else [],
        },
        "mask": {
            "shape": list(mask.shape),
            "valid_pairs": int(mask.sum()),
            "invalid_pairs": int(mask.size - mask.sum()),
            "coverage_percent": _coverage(int(mask.sum()), int(mask.size)),
            "lead_times_count": len(lead_times),
            "lead_times_range": [int(lead_times[0]), int(lead_times[-1])],
        },
    }
    with (output_dir / "training_validation_report.json").open("w") as f:
        json.dump(report, f, indent=2)
    logger.info("SUCCESS")


def run_validation(case_dirs: dict) -> None:
    """Run validate_case on every generated case."""
    with open(BASE_CONFIG) as f:
        base = yaml.safe_load(f)

    for name, case_dir in case_dirs.items():
        cfg = copy.deepcopy(base)  # deep copy so each case starts from the same base
        dataset = cfg["train_loader"]["dataset_config"]
        dataset["model"]["paths"] = str(case_dir / "model")
        dataset["observation"]["paths"] = str(case_dir / "obs")
        dataset["model"][
            "rename_dict"
        ] = {}  # dummy model already uses 'time'; no rename needed

        out_dir = OUTPUT_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = out_dir / "config.yaml"
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        print(f"\n=== {name} ===")
        try:
            validate_case(cfg_path, out_dir)
        except Exception as e:
            print(f"[{name}] failed: {type(e).__name__}: {e}")
            traceback.print_exc()


def main():
    """Generate all dummy cases, then validate each one."""
    case_dirs = generate_cases()
    run_validation(case_dirs)


if __name__ == "__main__":
    main()
