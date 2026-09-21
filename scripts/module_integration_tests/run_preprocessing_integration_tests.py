from __future__ import annotations

import csv
import inspect
import json
import logging
import re
import shutil
import warnings
from copy import deepcopy
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterator

import dacite
import joblib
import numpy as np
import torch
import xarray as xr
import yaml
from tqdm import tqdm

from cccma_ppp.generic.distributed import Distributed
from cccma_ppp.generic.runtime import RuntimeContext
from cccma_ppp.train.train_configs import TrainConfig, prepare_config
from integrationtestslib import LoggingUtils

warnings.filterwarnings("ignore")
logging.raiseExceptions = False


ROOT = Path("/fs/site7/eccc/crd/cccma/users/rna002/CCCma-PPP/scripts")
BASE_TRAIN_CONFIG = ROOT / "integration_suite_train_config.yaml"
OUTPUT_DIR = ROOT / "../../output" / "preprocessing_integration_test_results"
CASES_DIR = OUTPUT_DIR / "cases"
RESULTS_CSV = OUTPUT_DIR / "results.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary.txt"

RUN_INDIVIDUAL_CASES = True
RUN_DIMENSION_CASES = True
RUN_FREQUENCY_CASES = True
RUN_STACKED_CASES = True
RUN_COMBINED_SOURCE_CASES = True
RUN_MODEL_CASES = True
RUN_OBSERVATION_CASES = True

MAX_CASES: int | None = None
MAX_BATCHES_PER_CASE = 4

DEFAULT_YEAR_SLICE = [1980, 2007]
DEFAULT_TIME_FEATURES = ["time", "lead_time", "month_sin", "month_cos"]

TIME_DIM = "time"
LEAD_TIME_DIM = "lead_time"
LAT_DIM = "lat"
LON_DIM = "lon"

MODEL_TIME_DIMS = [TIME_DIM]
MODEL_SPATIAL_DIMS = [LAT_DIM, LON_DIM]
MODEL_GLOBAL_DIMS = [TIME_DIM, LAT_DIM, LON_DIM]

OBSERVATION_TIME_DIMS = [TIME_DIM]
OBSERVATION_SPATIAL_DIMS = [LAT_DIM, LON_DIM]
OBSERVATION_GLOBAL_DIMS = [TIME_DIM, LAT_DIM, LON_DIM]

VALID_FREQUENCIES = {"month", "day", "year"}

EXPECTED_PREPROCESSOR_STATISTICS = {
    "anomalies": (("mean", "mean_", "means", "means_"),),
    "standardizer": (
        ("mean", "mean_", "means", "means_"),
        ("std", "std_", "standard_deviation", "standard_deviation_"),
    ),
    "normalizer": (),
    "trend_remover": (),
}

CSV_COLUMNS = [
    "timestamp",
    "suite",
    "case_name",
    "description",
    "preprocessing_sources",
    "preprocessing_steps",
    "year_slice",
    "num_data_workers",
    "case_dir",
    "config_path",
    "log_path",
    "statistics_dir",
    "joblib_paths",
    "result",
    "error",
]


def preprocessing_step(
    name: str,
    *,
    dims: list[str] | None = None,
    frequency: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Create one preprocessing configuration entry."""
    args: dict[str, Any] = {}
    if dims is not None:
        args["dims"] = deepcopy(dims)
    if frequency is not None:
        args["frequency"] = frequency
    args.update(deepcopy(kwargs))
    result: dict[str, Any] = {"name": name}
    if args:
        result["args"] = args
    return result


class PreprocessingScenarios:
    """All preprocessing test scenarios."""

    MODEL_ANOMALIES = {
        "name": "model_anomalies",
        "description": ("Fit and apply anomalies to model data over time."),
        "pipelines": {"model": [preprocessing_step("anomalies", dims=MODEL_TIME_DIMS)]},
    }

    MODEL_STANDARDIZER = {
        "name": "model_standardizer",
        "description": (
            "Fit and apply standardization to model data over time and space."
        ),
        "pipelines": {
            "model": [preprocessing_step("standardizer", dims=MODEL_GLOBAL_DIMS)]
        },
    }

    MODEL_NORMALIZER = {
        "name": "model_normalizer",
        "description": (
            "Fit and apply normalization to model data over time and space."
        ),
        "pipelines": {
            "model": [preprocessing_step("normalizer", dims=MODEL_GLOBAL_DIMS)]
        },
    }

    MODEL_TREND_REMOVER = {
        "name": "model_trend_remover",
        "description": ("Fit and apply trend removal to model data."),
        "pipelines": {
            "model": [preprocessing_step("trend_remover", dims=MODEL_SPATIAL_DIMS)]
        },
    }

    OBSERVATION_ANOMALIES = {
        "name": "observation_anomalies",
        "description": ("Fit and apply anomalies to observation data over time."),
        "pipelines": {
            "observation": [preprocessing_step("anomalies", dims=OBSERVATION_TIME_DIMS)]
        },
    }

    OBSERVATION_STANDARDIZER = {
        "name": "observation_standardizer",
        "description": (
            "Fit and apply standardization to observation data over time and space."
        ),
        "pipelines": {
            "observation": [
                preprocessing_step("standardizer", dims=OBSERVATION_GLOBAL_DIMS)
            ]
        },
    }

    OBSERVATION_NORMALIZER = {
        "name": "observation_normalizer",
        "description": (
            "Fit and apply normalization to observation data over time and space."
        ),
        "pipelines": {
            "observation": [
                preprocessing_step("normalizer", dims=OBSERVATION_GLOBAL_DIMS)
            ]
        },
    }

    OBSERVATION_TREND_REMOVER = {
        "name": "observation_trend_remover",
        "description": ("Fit and apply trend removal to observation data."),
        "pipelines": {
            "observation": [
                preprocessing_step("trend_remover", dims=OBSERVATION_SPATIAL_DIMS)
            ]
        },
    }

    MODEL_ANOMALIES_TIME = {
        "name": "model_anomalies_dims_time",
        "description": ("Calculate model anomaly means over time only."),
        "pipelines": {"model": [preprocessing_step("anomalies", dims=[TIME_DIM])]},
    }

    MODEL_ANOMALIES_TIME_LAT_LON = {
        "name": "model_anomalies_dims_time_lat_lon",
        "description": (
            "Calculate model anomaly means over time, latitude, and longitude."
        ),
        "pipelines": {
            "model": [preprocessing_step("anomalies", dims=MODEL_GLOBAL_DIMS)]
        },
    }

    MODEL_STANDARDIZER_LAT_LON = {
        "name": "model_standardizer_dims_lat_lon",
        "description": (
            "Calculate model standardization statistics over latitude and longitude."
        ),
        "pipelines": {
            "model": [preprocessing_step("standardizer", dims=MODEL_SPATIAL_DIMS)]
        },
    }

    MODEL_STANDARDIZER_TIME_LAT_LON = {
        "name": "model_standardizer_dims_time_lat_lon",
        "description": (
            "Calculate model standardization statistics over time, latitude, and longitude."
        ),
        "pipelines": {
            "model": [preprocessing_step("standardizer", dims=MODEL_GLOBAL_DIMS)]
        },
    }

    OBSERVATION_ANOMALIES_TIME = {
        "name": "observation_anomalies_dims_time",
        "description": ("Calculate observation anomaly means over time only."),
        "pipelines": {
            "observation": [preprocessing_step("anomalies", dims=[TIME_DIM])]
        },
    }

    OBSERVATION_ANOMALIES_TIME_LAT_LON = {
        "name": "observation_anomalies_dims_time_lat_lon",
        "description": (
            "Calculate observation anomaly means over time, latitude, and longitude."
        ),
        "pipelines": {
            "observation": [
                preprocessing_step("anomalies", dims=OBSERVATION_GLOBAL_DIMS)
            ]
        },
    }

    OBSERVATION_STANDARDIZER_LAT_LON = {
        "name": "observation_standardizer_dims_lat_lon",
        "description": (
            "Calculate observation standardization statistics over latitude and longitude."
        ),
        "pipelines": {
            "observation": [
                preprocessing_step("standardizer", dims=OBSERVATION_SPATIAL_DIMS)
            ]
        },
    }

    OBSERVATION_STANDARDIZER_TIME_LAT_LON = {
        "name": "observation_standardizer_dims_time_lat_lon",
        "description": (
            "Calculate observation standardization statistics over time, latitude, and longitude."
        ),
        "pipelines": {
            "observation": [
                preprocessing_step("standardizer", dims=OBSERVATION_GLOBAL_DIMS)
            ]
        },
    }

    MODEL_ANOMALIES_MONTH = {
        "name": "model_anomalies_frequency_month",
        "description": ("Calculate monthly model anomaly statistics."),
        "pipelines": {
            "model": [
                preprocessing_step(
                    "anomalies", dims=MODEL_GLOBAL_DIMS, frequency="month"
                )
            ]
        },
    }

    MODEL_ANOMALIES_DAY = {
        "name": "model_anomalies_frequency_day",
        "description": ("Calculate daily model anomaly statistics."),
        "pipelines": {
            "model": [
                preprocessing_step("anomalies", dims=MODEL_GLOBAL_DIMS, frequency="day")
            ]
        },
    }

    MODEL_ANOMALIES_YEAR = {
        "name": "model_anomalies_frequency_year",
        "description": ("Calculate annual model anomaly statistics."),
        "pipelines": {
            "model": [
                preprocessing_step(
                    "anomalies", dims=MODEL_GLOBAL_DIMS, frequency="year"
                )
            ]
        },
    }

    MODEL_STANDARDIZER_MONTH = {
        "name": "model_standardizer_frequency_month",
        "description": ("Calculate monthly model standardization statistics."),
        "pipelines": {
            "model": [
                preprocessing_step(
                    "standardizer", dims=MODEL_GLOBAL_DIMS, frequency="month"
                )
            ]
        },
    }

    MODEL_STANDARDIZER_DAY = {
        "name": "model_standardizer_frequency_day",
        "description": ("Calculate daily model standardization statistics."),
        "pipelines": {
            "model": [
                preprocessing_step(
                    "standardizer", dims=MODEL_GLOBAL_DIMS, frequency="day"
                )
            ]
        },
    }

    MODEL_STANDARDIZER_YEAR = {
        "name": "model_standardizer_frequency_year",
        "description": ("Calculate annual model standardization statistics."),
        "pipelines": {
            "model": [
                preprocessing_step(
                    "standardizer", dims=MODEL_GLOBAL_DIMS, frequency="year"
                )
            ]
        },
    }

    OBSERVATION_ANOMALIES_MONTH = {
        "name": "observation_anomalies_frequency_month",
        "description": ("Calculate monthly observation anomaly statistics."),
        "pipelines": {
            "observation": [
                preprocessing_step(
                    "anomalies", dims=OBSERVATION_GLOBAL_DIMS, frequency="month"
                )
            ]
        },
    }

    OBSERVATION_ANOMALIES_DAY = {
        "name": "observation_anomalies_frequency_day",
        "description": ("Calculate daily observation anomaly statistics."),
        "pipelines": {
            "observation": [
                preprocessing_step(
                    "anomalies", dims=OBSERVATION_GLOBAL_DIMS, frequency="day"
                )
            ]
        },
    }

    OBSERVATION_ANOMALIES_YEAR = {
        "name": "observation_anomalies_frequency_year",
        "description": ("Calculate annual observation anomaly statistics."),
        "pipelines": {
            "observation": [
                preprocessing_step(
                    "anomalies", dims=OBSERVATION_GLOBAL_DIMS, frequency="year"
                )
            ]
        },
    }

    OBSERVATION_STANDARDIZER_MONTH = {
        "name": "observation_standardizer_frequency_month",
        "description": ("Calculate monthly observation standardization statistics."),
        "pipelines": {
            "observation": [
                preprocessing_step(
                    "standardizer", dims=OBSERVATION_GLOBAL_DIMS, frequency="month"
                )
            ]
        },
    }

    OBSERVATION_STANDARDIZER_DAY = {
        "name": "observation_standardizer_frequency_day",
        "description": ("Calculate daily observation standardization statistics."),
        "pipelines": {
            "observation": [
                preprocessing_step(
                    "standardizer", dims=OBSERVATION_GLOBAL_DIMS, frequency="day"
                )
            ]
        },
    }

    OBSERVATION_STANDARDIZER_YEAR = {
        "name": "observation_standardizer_frequency_year",
        "description": ("Calculate annual observation standardization statistics."),
        "pipelines": {
            "observation": [
                preprocessing_step(
                    "standardizer", dims=OBSERVATION_GLOBAL_DIMS, frequency="year"
                )
            ]
        },
    }

    MODEL_ANOMALIES_STANDARDIZER = {
        "name": "model_anomalies_then_standardizer",
        "description": (
            "Apply monthly anomalies followed by spatial standardization to model data."
        ),
        "pipelines": {
            "model": [
                preprocessing_step(
                    "anomalies", dims=MODEL_GLOBAL_DIMS, frequency="month"
                ),
                preprocessing_step("standardizer", dims=MODEL_SPATIAL_DIMS),
            ],
        },
    }

    MODEL_TREND_ANOMALIES = {
        "name": "model_trend_then_anomalies",
        "description": (
            "Apply trend removal followed by monthly anomalies to model data."
        ),
        "pipelines": {
            "model": [
                preprocessing_step("trend_remover", dims=MODEL_SPATIAL_DIMS),
                preprocessing_step(
                    "anomalies", dims=MODEL_GLOBAL_DIMS, frequency="month"
                ),
            ],
        },
    }

    MODEL_TREND_ANOMALIES_STANDARDIZER = {
        "name": "model_trend_anomalies_standardizer",
        "description": (
            "Apply trend removal, monthly anomalies, and standardization to model data."
        ),
        "pipelines": {
            "model": [
                preprocessing_step("trend_remover", dims=MODEL_SPATIAL_DIMS),
                preprocessing_step(
                    "anomalies", dims=MODEL_GLOBAL_DIMS, frequency="month"
                ),
                preprocessing_step("standardizer", dims=MODEL_SPATIAL_DIMS),
            ],
        },
    }

    MODEL_DIFFERENT_DIMS_FREQUENCIES = {
        "name": "model_different_dims_frequencies",
        "description": (
            "Stack model preprocessors using different dimensions and frequencies."
        ),
        "pipelines": {
            "model": [
                preprocessing_step(
                    "anomalies", dims=MODEL_GLOBAL_DIMS, frequency="month"
                ),
                preprocessing_step(
                    "standardizer", dims=MODEL_TIME_DIMS, frequency="day"
                ),
            ],
        },
    }

    OBSERVATION_ANOMALIES_STANDARDIZER = {
        "name": "observation_anomalies_then_standardizer",
        "description": (
            "Apply monthly anomalies followed by spatial standardization to observation data."
        ),
        "pipelines": {
            "observation": [
                preprocessing_step(
                    "anomalies", dims=OBSERVATION_GLOBAL_DIMS, frequency="month"
                ),
                preprocessing_step("standardizer", dims=OBSERVATION_SPATIAL_DIMS),
            ],
        },
    }

    OBSERVATION_TREND_ANOMALIES_STANDARDIZER = {
        "name": "observation_trend_anomalies_standardizer",
        "description": (
            "Apply trend removal, monthly anomalies, and standardization to observation data."
        ),
        "pipelines": {
            "observation": [
                preprocessing_step("trend_remover", dims=OBSERVATION_SPATIAL_DIMS),
                preprocessing_step(
                    "anomalies", dims=OBSERVATION_GLOBAL_DIMS, frequency="month"
                ),
                preprocessing_step("standardizer", dims=OBSERVATION_SPATIAL_DIMS),
            ],
        },
    }

    OBSERVATION_DIFFERENT_DIMS_FREQUENCIES = {
        "name": "observation_different_dims_frequencies",
        "description": (
            "Stack observation preprocessors using different dimensions and frequencies."
        ),
        "pipelines": {
            "observation": [
                preprocessing_step(
                    "anomalies", dims=OBSERVATION_GLOBAL_DIMS, frequency="month"
                ),
                preprocessing_step(
                    "standardizer", dims=OBSERVATION_TIME_DIMS, frequency="day"
                ),
            ],
        },
    }

    MODEL_OBSERVATION_ANOMALIES = {
        "name": "model_observation_anomalies",
        "description": (
            "Fit monthly anomaly preprocessors for model and observation data in the same loader."
        ),
        "pipelines": {
            "model": [
                preprocessing_step(
                    "anomalies", dims=MODEL_GLOBAL_DIMS, frequency="month"
                )
            ],
            "observation": [
                preprocessing_step(
                    "anomalies", dims=OBSERVATION_GLOBAL_DIMS, frequency="month"
                )
            ],
        },
    }

    MODEL_OBSERVATION_STANDARDIZERS = {
        "name": "model_observation_standardizers",
        "description": (
            "Fit monthly standardizers for model and observation data in the same loader."
        ),
        "pipelines": {
            "model": [
                preprocessing_step(
                    "standardizer", dims=MODEL_GLOBAL_DIMS, frequency="month"
                )
            ],
            "observation": [
                preprocessing_step(
                    "standardizer", dims=OBSERVATION_GLOBAL_DIMS, frequency="month"
                )
            ],
        },
    }

    MODEL_OBSERVATION_STACKED = {
        "name": "model_observation_stacked",
        "description": (
            "Fit distinct stacked pipelines for model and observation data."
        ),
        "pipelines": {
            "model": [
                preprocessing_step("trend_remover", dims=MODEL_SPATIAL_DIMS),
                preprocessing_step(
                    "anomalies", dims=MODEL_GLOBAL_DIMS, frequency="month"
                ),
                preprocessing_step("standardizer", dims=MODEL_SPATIAL_DIMS),
            ],
            "observation": [
                preprocessing_step(
                    "anomalies", dims=OBSERVATION_GLOBAL_DIMS, frequency="day"
                ),
                preprocessing_step("normalizer", dims=OBSERVATION_SPATIAL_DIMS),
            ],
        },
    }

    @classmethod
    def individual(cls) -> list[dict[str, Any]]:
        """Return individual preprocessing cases."""
        cases: list[dict[str, Any]] = []
        if RUN_MODEL_CASES:
            cases.extend(
                [
                    cls.MODEL_ANOMALIES,
                    cls.MODEL_STANDARDIZER,
                    cls.MODEL_NORMALIZER,
                    cls.MODEL_TREND_REMOVER,
                ]
            )
        if RUN_OBSERVATION_CASES:
            cases.extend(
                [
                    cls.OBSERVATION_ANOMALIES,
                    cls.OBSERVATION_STANDARDIZER,
                    cls.OBSERVATION_NORMALIZER,
                    cls.OBSERVATION_TREND_REMOVER,
                ]
            )
        return cases

    @classmethod
    def dimensions(cls) -> list[dict[str, Any]]:
        """Return dimension-specific cases."""
        cases: list[dict[str, Any]] = []
        if RUN_MODEL_CASES:
            cases.extend(
                [
                    cls.MODEL_ANOMALIES_TIME,
                    cls.MODEL_ANOMALIES_TIME_LAT_LON,
                    cls.MODEL_STANDARDIZER_LAT_LON,
                    cls.MODEL_STANDARDIZER_TIME_LAT_LON,
                ]
            )
        if RUN_OBSERVATION_CASES:
            cases.extend(
                [
                    cls.OBSERVATION_ANOMALIES_TIME,
                    cls.OBSERVATION_ANOMALIES_TIME_LAT_LON,
                    cls.OBSERVATION_STANDARDIZER_LAT_LON,
                    cls.OBSERVATION_STANDARDIZER_TIME_LAT_LON,
                ]
            )
        return cases

    @classmethod
    def frequencies(cls) -> list[dict[str, Any]]:
        """Return frequency-specific cases."""
        cases: list[dict[str, Any]] = []
        if RUN_MODEL_CASES:
            cases.extend(
                [
                    cls.MODEL_ANOMALIES_MONTH,
                    cls.MODEL_ANOMALIES_DAY,
                    cls.MODEL_ANOMALIES_YEAR,
                    cls.MODEL_STANDARDIZER_MONTH,
                    cls.MODEL_STANDARDIZER_DAY,
                    cls.MODEL_STANDARDIZER_YEAR,
                ]
            )
        if RUN_OBSERVATION_CASES:
            cases.extend(
                [
                    cls.OBSERVATION_ANOMALIES_MONTH,
                    cls.OBSERVATION_ANOMALIES_DAY,
                    cls.OBSERVATION_ANOMALIES_YEAR,
                    cls.OBSERVATION_STANDARDIZER_MONTH,
                    cls.OBSERVATION_STANDARDIZER_DAY,
                    cls.OBSERVATION_STANDARDIZER_YEAR,
                ]
            )
        return cases

    @classmethod
    def stacked(cls) -> list[dict[str, Any]]:
        """Return stacked preprocessing cases."""
        cases: list[dict[str, Any]] = []
        if RUN_MODEL_CASES:
            cases.extend(
                [
                    cls.MODEL_ANOMALIES_STANDARDIZER,
                    cls.MODEL_TREND_ANOMALIES,
                    cls.MODEL_TREND_ANOMALIES_STANDARDIZER,
                    cls.MODEL_DIFFERENT_DIMS_FREQUENCIES,
                ]
            )
        if RUN_OBSERVATION_CASES:
            cases.extend(
                [
                    cls.OBSERVATION_ANOMALIES_STANDARDIZER,
                    cls.OBSERVATION_TREND_ANOMALIES_STANDARDIZER,
                    cls.OBSERVATION_DIFFERENT_DIMS_FREQUENCIES,
                ]
            )
        return cases

    @classmethod
    def combined_sources(cls) -> list[dict[str, Any]]:
        """Return combined model and observation cases."""
        return [
            cls.MODEL_OBSERVATION_ANOMALIES,
            cls.MODEL_OBSERVATION_STANDARDIZERS,
            cls.MODEL_OBSERVATION_STACKED,
        ]


def clean_name(value: Any) -> str:
    """Convert a value into a filesystem-safe name."""
    value = str(value).replace("None", "none")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def serialize_json(value: Any) -> str:
    """Serialize a value as JSON."""
    return json.dumps(value, sort_keys=True, default=str)


def case_paths(case_name: str) -> dict[str, Path]:
    """Return paths for one case."""
    case_dir = CASES_DIR / clean_name(case_name)
    return {
        "case_dir": case_dir,
        "config_path": case_dir / "config.yaml",
        "log_path": case_dir / "case.log",
        "statistics_dir": case_dir / "fitted_statistics",
        "joblib_dir": case_dir / "fitted_joblib",
    }


def prepare_output_directories() -> None:
    """Create output directories."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CASES_DIR.mkdir(parents=True, exist_ok=True)


def init_csv() -> None:
    """Initialize the CSV results file."""
    prepare_output_directories()
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=CSV_COLUMNS, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()


def log_result(
    *,
    task: dict[str, Any],
    paths: dict[str, Path],
    passed: bool,
    error: str,
    joblib_paths: list[Path],
) -> None:
    """Append one result to the CSV file."""
    params = task["params"]
    row = {
        "timestamp": datetime.now().isoformat(),
        "suite": task["suite"],
        "case_name": task["case_name"],
        "description": params.get("description", ""),
        "preprocessing_sources": serialize_json(sorted(params["pipelines"])),
        "preprocessing_steps": serialize_json(params["pipelines"]),
        "year_slice": serialize_json(params.get("year_slice", DEFAULT_YEAR_SLICE)),
        "num_data_workers": params.get("num_data_workers", 0),
        "case_dir": str(paths["case_dir"]),
        "config_path": str(paths["config_path"]),
        "log_path": str(paths["log_path"]),
        "statistics_dir": str(paths["statistics_dir"]),
        "joblib_paths": serialize_json([str(path) for path in joblib_paths]),
        "result": "PASS" if passed else "FAIL",
        "error": error,
    }
    with RESULTS_CSV.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=CSV_COLUMNS, extrasaction="ignore", lineterminator="\n"
        )
        writer.writerow(row)
        file.flush()


def normalize_nullable_fields(cfg: dict[str, Any]) -> dict[str, Any]:
    """Remove nullable fields."""
    loader_cfg = cfg.setdefault("train_loader", {})
    dataset_cfg = loader_cfg.setdefault("dataset_config", {})
    if loader_cfg.get("time_features") is None:
        loader_cfg.pop("time_features", None)
    if dataset_cfg.get("condition_method") is None:
        dataset_cfg.pop("condition_method", None)
    return cfg


def validate_generated_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate configuration structure."""
    train_loader_cfg = cfg.get("train_loader")
    if not isinstance(train_loader_cfg, dict):
        raise ValueError("train_loader must be a mapping")
    dataset_cfg = train_loader_cfg.get("dataset_config")
    if not isinstance(dataset_cfg, dict):
        raise ValueError("train_loader.dataset_config must be a mapping")
    if not isinstance(dataset_cfg.get("model"), dict):
        raise ValueError("train_loader.dataset_config.model must be a mapping")
    if not isinstance(dataset_cfg.get("observation"), dict):
        raise ValueError("train_loader.dataset_config.observation must be a mapping")
    return cfg


def base_config() -> dict[str, Any]:
    """Load and normalize base configuration."""
    if not BASE_TRAIN_CONFIG.exists():
        raise FileNotFoundError(f"Base config not found: {BASE_TRAIN_CONFIG}")
    cfg = prepare_config(BASE_TRAIN_CONFIG)
    if cfg is None:
        raise ValueError(f"Base config is empty: {BASE_TRAIN_CONFIG}")
    cfg = deepcopy(cfg)
    train_loader_cfg = cfg.get("train_loader")
    if not isinstance(train_loader_cfg, dict):
        raise ValueError("Base config must contain train_loader mapping")
    train_loader_cfg["batch_size"] = 2
    train_loader_cfg["num_data_workers"] = 0
    train_loader_cfg["num_validation_years"] = 0
    train_loader_cfg["train_years_slice"] = deepcopy(DEFAULT_YEAR_SLICE)
    train_loader_cfg["time_features"] = deepcopy(DEFAULT_TIME_FEATURES)
    train_loader_cfg["load"] = False
    for key in ("shuffle", "return_metadata", "return_spatial_mask", "prefetch_factor"):
        train_loader_cfg.pop(key, None)
    normalize_nullable_fields(cfg)
    validate_generated_config(cfg)
    return cfg


def get_source_config(dataset_cfg: dict[str, Any], source_name: str) -> dict[str, Any]:
    """Get data-source configuration."""
    source_keys = {
        "model": "model",
        "observation": "observation",
        "condition": "condition",
    }
    if source_name not in source_keys:
        raise ValueError(f"Unsupported source: {source_name!r}")
    source_key = source_keys[source_name]
    source_cfg = dataset_cfg.get(source_key)
    if not isinstance(source_cfg, dict):
        raise ValueError(f"dataset_config.{source_key} must be a mapping")
    return source_cfg


def remove_existing_preprocessing(dataset_cfg: dict[str, Any]) -> None:
    """Remove inherited preprocessing."""
    for source_name in ("model", "observation", "condition"):
        source_cfg = dataset_cfg.get(source_name)
        if isinstance(source_cfg, dict):
            source_cfg.pop("preprocessing_pipeline", None)


def set_source_preprocessing(
    dataset_cfg: dict[str, Any],
    *,
    source_name: str,
    preprocessing_steps: list[dict[str, Any]],
) -> None:
    """Set preprocessing for one source."""
    source_cfg = get_source_config(dataset_cfg, source_name)
    source_cfg["preprocessing_pipeline"] = {
        "preprocessors_list": deepcopy(preprocessing_steps)
    }


def apply_preprocessing_case(
    cfg: dict[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
    """Apply preprocessing scenario to config."""
    cfg = deepcopy(cfg)
    train_loader_cfg = cfg["train_loader"]
    dataset_cfg = train_loader_cfg["dataset_config"]
    remove_existing_preprocessing(dataset_cfg)
    pipelines = params.get("pipelines")
    if not isinstance(pipelines, dict) or not pipelines:
        raise ValueError("Case must define at least one pipeline")
    for source_name, preprocessing_steps in pipelines.items():
        if source_name not in {"model", "observation", "condition"}:
            raise ValueError(f"Unsupported source: {source_name!r}")
        if not isinstance(preprocessing_steps, list) or not preprocessing_steps:
            raise ValueError(f"{source_name} pipeline must be non-empty list")
        set_source_preprocessing(
            dataset_cfg,
            source_name=source_name,
            preprocessing_steps=preprocessing_steps,
        )
    for source_name, expected_steps in pipelines.items():
        source_cfg = get_source_config(dataset_cfg, source_name)
        pipeline_cfg = source_cfg.get("preprocessing_pipeline")
        if not isinstance(pipeline_cfg, dict):
            raise AssertionError(f"{source_name} preprocessing_pipeline not written")
        actual_steps = pipeline_cfg.get("preprocessors_list")
        if actual_steps != expected_steps:
            raise AssertionError(f"{source_name} preprocessing written incorrectly")
    train_loader_cfg["train_years_slice"] = deepcopy(
        params.get("year_slice", DEFAULT_YEAR_SLICE)
    )
    train_loader_cfg["num_data_workers"] = params.get("num_data_workers", 0)
    train_loader_cfg["load"] = False
    num_data_workers = train_loader_cfg["num_data_workers"]
    if (
        isinstance(num_data_workers, bool)
        or not isinstance(num_data_workers, int)
        or num_data_workers < 0
    ):
        raise ValueError("num_data_workers must be non-negative integer")
    if num_data_workers == 0:
        train_loader_cfg.pop("prefetch_factor", None)
    else:
        train_loader_cfg.setdefault("prefetch_factor", 2)
    for key in ("shuffle", "return_metadata", "return_spatial_mask"):
        train_loader_cfg.pop(key, None)
    normalize_nullable_fields(cfg)
    validate_generated_config(cfg)
    return cfg


def validate_preprocessing_written(cfg: dict[str, Any], params: dict[str, Any]) -> None:
    """Validate preprocessing in config."""
    dataset_cfg = cfg["train_loader"]["dataset_config"]
    expected_pipelines = params["pipelines"]
    for source_name, expected_steps in expected_pipelines.items():
        source_cfg = get_source_config(dataset_cfg, source_name)
        pipeline_cfg = source_cfg.get("preprocessing_pipeline")
        if not isinstance(pipeline_cfg, dict):
            raise AssertionError(f"{source_name} preprocessing_pipeline not written")
        actual_steps = pipeline_cfg.get("preprocessors_list")
        if actual_steps != expected_steps:
            raise AssertionError(f"{source_name} preprocessing differs")
    for source_name in ("model", "observation", "condition"):
        if source_name in expected_pipelines:
            continue
        source_cfg = dataset_cfg.get(source_name)
        if isinstance(source_cfg, dict) and "preprocessing_pipeline" in source_cfg:
            raise AssertionError(f"{source_name} retained preprocessing incorrectly")


def write_yaml(cfg: dict[str, Any], output_path: Path) -> Path:
    """Write config to YAML."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(deepcopy(cfg), file, sort_keys=False)
    return output_path


def call_with_supported_arguments(function: Callable[..., Any], **kwargs: Any) -> Any:
    """Call function with only supported arguments."""
    signature = inspect.signature(function)
    supported = {
        name: value for name, value in kwargs.items() if name in signature.parameters
    }
    return function(**supported)


def load_train_config(config_path: Path) -> TrainConfig:
    """Load config into TrainConfig."""
    data = prepare_config(config_path)
    if data is None:
        raise ValueError(f"Config is empty: {config_path}")
    normalize_nullable_fields(data)
    validate_generated_config(data)
    return dacite.from_dict(
        data_class=TrainConfig, data=data, config=dacite.Config(strict=True)
    )


def setup_loader(config: TrainConfig, config_path: Path) -> Any:
    """Build training loader."""
    RuntimeContext.GLOBAL_EXP_DIR = str(config_path.resolve().parent)
    distributed = Distributed.get_instance()
    call_with_supported_arguments(
        config.train_loader.setup_distributed, distributed=distributed
    )
    return call_with_supported_arguments(
        config.train_loader.build_train_loader,
        return_metadata=True,
        return_spatial_mask=True,
        shuffle=False,
    )


class LimitedLoader:
    """Limit iteration to max batches."""

    def __init__(self, loader: Any, max_batches: int | None) -> None:
        self._loader = loader
        self._max_batches = max_batches

    def __iter__(self) -> Iterator:
        for index, batch in enumerate(self._loader):
            if self._max_batches is not None and index >= self._max_batches:
                break
            yield batch

    def __len__(self) -> int:
        length = len(self._loader)
        return length if self._max_batches is None else min(length, self._max_batches)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader, name)


def find_joblib_files(case_dir: Path) -> list:
    """Find joblib files under case directory."""
    return sorted(
        path.resolve()
        for path in case_dir.rglob("*.joblib")
        if path.is_file() and "fitted_joblib" not in path.parts
    )


def classify_joblib_source(path: Path) -> str | None:
    """Classify joblib by filename."""
    name = path.name.lower()
    if "observation" in name or "obs" in name:
        return "observation"
    if "condition" in name:
        return "condition"
    if "model" in name:
        return "model"
    return None


def find_source_joblib(case_dir: Path, *, source_name: str) -> Path | None:
    """Find joblib for one source."""
    joblib_files = find_joblib_files(case_dir)
    source_matches = [
        path for path in joblib_files if classify_joblib_source(path) == source_name
    ]
    if len(source_matches) == 1:
        return source_matches[0]
    if len(source_matches) > 1:
        raise AssertionError(f"Multiple joblib files for {source_name!r}")
    if len(joblib_files) == 1:
        return joblib_files[0]
    return None


def copy_joblib(joblib_path: Path, *, output_dir: Path) -> Path:
    """Copy joblib to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / joblib_path.name
    if joblib_path.resolve() != destination.resolve():
        shutil.copy2(joblib_path, destination)
    return destination


def get_named_preprocessors(pipeline: Any, preprocessor_name: str) -> list:
    """Get fitted preprocessors by name."""
    if hasattr(pipeline, "get_preprocessors"):
        result = pipeline.get_preprocessors(preprocessor_name)
        if result is None:
            return []
        return list(result) if isinstance(result, (list, tuple)) else [result]
    if hasattr(pipeline, "get_preprocessor"):
        result = pipeline.get_preprocessor(preprocessor_name)
        if result is None:
            return []
        return list(result) if isinstance(result, (list, tuple)) else [result]
    normalized_name = re.sub(r"[^a-z0-9]+", "", preprocessor_name.lower())
    for collection in (
        getattr(pipeline, "preprocessors", None),
        getattr(pipeline, "preprocessors_list", None),
        getattr(pipeline, "steps", None),
    ):
        if collection is None:
            continue
        values = (
            list(collection.items())
            if isinstance(collection, dict)
            else list(collection)
        )
        matches: list[Any] = []
        for item in values:
            configured_name, candidate = ("", item)
            if isinstance(item, tuple) and len(item) == 2:
                configured_name, candidate = item
            possible_names = (
                str(configured_name),
                str(getattr(candidate, "name", "")),
                type(candidate).__name__,
            )
            normalized_candidates = [
                re.sub(r"[^a-z0-9]+", "", value.lower()) for value in possible_names
            ]
            if any(
                normalized_name == cn or normalized_name in cn
                for cn in normalized_candidates
            ):
                matches.append(candidate)
        if matches:
            return matches
    return []


def find_statistic_attribute(
    preprocessor: Any, candidates: tuple[str, ...]
) -> tuple[str, Any] | None:
    """Find fitted statistic."""
    for attribute_name in candidates:
        if hasattr(preprocessor, attribute_name):
            value = getattr(preprocessor, attribute_name)
            if value is not None:
                return attribute_name, value
    return None


def get_array_summary(array: Any) -> dict[str, Any]:
    """Summarize array shape/dims/coords."""
    result: dict[str, Any] = {}

    if isinstance(array, xr.DataArray):
        result["type"] = "xarray.DataArray"
        result["dimensions"] = list(array.dims)
        result["shape"] = list(array.shape)
        result["coordinates"] = {
            str(name): array.coords[name].values.tolist() for name in array.coords
        }
    elif isinstance(array, xr.Dataset):
        result["type"] = "xarray.Dataset"
        result["dimensions"] = {
            str(name): int(size) for name, size in array.sizes.items()
        }
        result["variables"] = list(array.data_vars)
        result["coordinates"] = {
            str(name): array.coords[name].values.tolist() for name in array.coords
        }
    elif isinstance(array, (np.ndarray, torch.Tensor)):
        if isinstance(array, torch.Tensor):
            array = array.detach().cpu().numpy()
        result["type"] = (
            "numpy.ndarray" if isinstance(array, np.ndarray) else "torch.Tensor"
        )
        result["shape"] = list(array.shape)
        result["dtype"] = str(array.dtype)

    return result


def inspect_fitted_pipeline(
    pipeline: Any,
    *,
    source_name: str,
    configured_steps: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    """Inspect fitted pipeline and save report."""
    source_dir = output_dir / source_name
    source_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "source": source_name,
        "pipeline_type": type(pipeline).__name__,
        "configured_steps": deepcopy(configured_steps),
        "fitted_statistics": {},
    }

    occurrence_counts: dict[str, int] = {}

    for configured_step in configured_steps:
        preprocessor_name = str(configured_step["name"])
        occurrence_index = occurrence_counts.get(preprocessor_name, 0)
        occurrence_counts[preprocessor_name] = occurrence_index + 1

        fitted_preprocessors = get_named_preprocessors(pipeline, preprocessor_name)
        if not fitted_preprocessors:
            raise AssertionError(
                f"No {preprocessor_name} in fitted {source_name} pipeline"
            )
        if occurrence_index >= len(fitted_preprocessors):
            raise AssertionError(
                f"Not enough {preprocessor_name} instances in fitted {source_name} pipeline"
            )

        fitted_preprocessor = fitted_preprocessors[occurrence_index]

        preprocessor_key = f"{preprocessor_name}[{occurrence_index}]"
        report["fitted_statistics"][preprocessor_key] = {
            "class": type(fitted_preprocessor).__name__,
            "configured_args": configured_step.get("args", {}),
            "statistics": {},
        }

        expected_statistics = EXPECTED_PREPROCESSOR_STATISTICS.get(
            preprocessor_name, ()
        )

        for statistic_candidates in expected_statistics:
            statistic_name = statistic_candidates[0]
            statistic_result = find_statistic_attribute(
                fitted_preprocessor, statistic_candidates
            )

            if statistic_result is None:
                raise AssertionError(
                    f"{source_name}.{preprocessor_name} missing {statistic_name}"
                )

            attribute_name, statistic_value = statistic_result

            summary = get_array_summary(statistic_value)
            summary["attribute"] = attribute_name
            report["fitted_statistics"][preprocessor_key]["statistics"][
                statistic_name
            ] = summary

            print(f"  {source_name}.{preprocessor_key}.{attribute_name}: {summary}")

    # Save report
    report_path = source_dir / "fitted_pipeline_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    return report


def read_yaml(config_path: Path) -> dict[str, Any]:
    """Read YAML config."""
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Config is not mapping: {config_path}")
    return data


def run_preprocessing_case(config_path: Path, params: dict[str, Any]) -> dict[str, Any]:
    """Run preprocessing case."""
    generated_config = read_yaml(config_path)
    config = load_train_config(config_path)
    loader = setup_loader(config, config_path)
    paths = case_paths(params["name"])

    print("Iterating loader to fit preprocessing...")
    batch_count = 0
    for batch in LimitedLoader(loader, MAX_BATCHES_PER_CASE):
        batch_count += 1
    print(f"Iterated {batch_count} batches - pipelines fitted")

    all_joblib_files = find_joblib_files(paths["case_dir"])
    if not all_joblib_files:
        raise AssertionError(f"No joblib files created under {paths['case_dir']}")

    loaded_joblib_paths: dict[str, Path] = {}
    for source_name, configured_steps in params["pipelines"].items():
        joblib_path = find_source_joblib(paths["case_dir"], source_name=source_name)
        if joblib_path is None:
            raise AssertionError(f"No joblib found for {source_name}")

        copied_joblib_path = copy_joblib(joblib_path, output_dir=paths["joblib_dir"])
        pipeline = joblib.load(copied_joblib_path)
        if pipeline is None:
            raise AssertionError("joblib.load returned None")

        loaded_joblib_paths[source_name] = copied_joblib_path
        print(f"Loaded {source_name} pipeline from {copied_joblib_path}")

        inspect_fitted_pipeline(
            pipeline,
            source_name=source_name,
            configured_steps=configured_steps,
            output_dir=paths["statistics_dir"],
        )

    print(f"Fitted sources: {sorted(loaded_joblib_paths.keys())}")
    return {"joblib_paths": list(loaded_joblib_paths.values())}


def add_task(
    tasks: list[dict[str, Any]], *, suite: str, params: dict[str, Any]
) -> None:
    """Add task."""
    task_params = deepcopy(params)
    tasks.append(
        {
            "suite": suite,
            "case_name": task_params["name"],
            "params": task_params,
            "cfg_builder": partial(
                apply_preprocessing_case, base_config(), task_params
            ),
            "runner": run_preprocessing_case,
        }
    )


def build_test_suite() -> list[dict[str, Any]]:
    """Build test suite."""
    tasks: list[dict[str, Any]] = []
    if RUN_INDIVIDUAL_CASES:
        for case in PreprocessingScenarios.individual():
            add_task(tasks, suite="individual_preprocessor", params=case)
    if RUN_DIMENSION_CASES:
        for case in PreprocessingScenarios.dimensions():
            add_task(tasks, suite="preprocessing_dimensions", params=case)
    if RUN_FREQUENCY_CASES:
        for case in PreprocessingScenarios.frequencies():
            add_task(tasks, suite="preprocessing_frequencies", params=case)
    if RUN_STACKED_CASES:
        for case in PreprocessingScenarios.stacked():
            add_task(tasks, suite="stacked_preprocessors", params=case)
    if RUN_COMBINED_SOURCE_CASES:
        for case in PreprocessingScenarios.combined_sources():
            add_task(tasks, suite="combined_source_preprocessing", params=case)
    return tasks


def get_step_arguments(configured_step: dict[str, Any]) -> dict[str, Any]:
    """Get preprocessor args."""
    args = configured_step.get("args", {})
    if not isinstance(args, dict):
        raise ValueError("Args must be mapping")
    return args


def validate_tasks(tasks: list[dict[str, Any]]) -> None:
    """Validate tasks."""
    required_fields = {"suite", "case_name", "params", "cfg_builder", "runner"}
    names: set[str] = set()
    for task_index, task in enumerate(tasks):
        missing = required_fields - set(task)
        if missing:
            raise ValueError(f"Task {task_index} missing: {sorted(missing)}")
        case_name = str(task["case_name"])
        if case_name in names:
            raise ValueError(f"Duplicate case: {case_name}")
        names.add(case_name)
        pipelines = task["params"].get("pipelines")
        if not isinstance(pipelines, dict) or not pipelines:
            raise ValueError(f"Task {case_name} has no pipelines")
        for source_name, configured_steps in pipelines.items():
            if source_name not in {"model", "observation", "condition"}:
                raise ValueError(f"Task {case_name} unsupported source {source_name}")
            if not isinstance(configured_steps, list) or not configured_steps:
                raise ValueError(f"Task {case_name} empty {source_name} pipeline")
            for step_index, configured_step in enumerate(configured_steps):
                if not isinstance(configured_step, dict):
                    raise ValueError(f"Task {case_name} step {step_index} not mapping")
                if "name" not in configured_step:
                    raise ValueError(f"Task {case_name} step {step_index} no name")
                unexpected_fields = set(configured_step) - {"name", "args"}
                if unexpected_fields:
                    raise ValueError(
                        f"Task {case_name} step {step_index} unexpected fields"
                    )
                arguments = get_step_arguments(configured_step)
                dims = arguments.get("dims")
                frequency = arguments.get("frequency")
                if dims is not None and not isinstance(dims, list):
                    raise ValueError(
                        f"Task {case_name} step {step_index} dims not list"
                    )
                if frequency is not None and frequency not in VALID_FREQUENCIES:
                    raise ValueError(
                        f"Task {case_name} step {step_index} invalid frequency"
                    )


def run_task(task: dict[str, Any]) -> tuple[bool, str]:
    """Run task."""
    paths = case_paths(task["case_name"])
    case_dir = paths["case_dir"]
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    paths["statistics_dir"].mkdir(parents=True, exist_ok=True)
    paths["joblib_dir"].mkdir(parents=True, exist_ok=True)

    passed = False
    error = ""
    joblib_paths: list[Path] = []

    try:
        params = task["params"]
        cfg_builder = task["cfg_builder"]
        cfg = cfg_builder()
        if not isinstance(cfg, dict):
            raise TypeError(f"cfg_builder returned {type(cfg).__name__}")
        cfg["experiment_dir"] = str(paths["case_dir"].resolve())
        validate_preprocessing_written(cfg, params)
        write_yaml(cfg, paths["config_path"])
        print(f"\nRunning {task['suite']}:{task['case_name']}", flush=True)
        result = LoggingUtils.run_silenced(
            partial(task["runner"], paths["config_path"], params), paths["log_path"]
        )
        joblib_paths = result.get("joblib_paths", [])
        passed = True
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        LoggingUtils.append_exception(paths["log_path"])
        LoggingUtils.cleanup_handlers()
    finally:
        RuntimeContext.GLOBAL_EXP_DIR = None
        try:
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    log_result(
        task=task, paths=paths, passed=passed, error=error, joblib_paths=joblib_paths
    )
    return passed, error


def write_summary(
    tasks: list[dict[str, Any]], passed_count: int, failed_count: int
) -> None:
    """Write summary."""
    suite_counts: dict[str, int] = {}
    for task in tasks:
        suite = task["suite"]
        suite_counts[suite] = suite_counts.get(suite, 0) + 1
    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        file.write(f"timestamp: {datetime.now().isoformat()}\n")
        file.write(f"total_cases: {len(tasks)}\n")
        file.write(f"passed: {passed_count}\n")
        file.write(f"failed: {failed_count}\n")
        file.write(f"cases_dir: {CASES_DIR}\n")
        file.write(f"results_csv: {RESULTS_CSV}\n")
        file.write("suites:\n")
        for suite, count in sorted(suite_counts.items()):
            file.write(f"  {suite}: {count}\n")


def main() -> None:
    """Main entry point."""
    prepare_output_directories()
    LoggingUtils.cleanup_handlers()
    init_csv()
    tasks = build_test_suite()
    validate_tasks(tasks)
    if MAX_CASES is not None:
        tasks = tasks[:MAX_CASES]
    passed_count = 0
    failed_count = 0
    try:
        with tqdm(
            total=len(tasks),
            desc="preprocessing_integration_tests",
            unit="case",
            dynamic_ncols=True,
            leave=True,
        ) as progress:
            for task in tasks:
                progress.set_postfix_str(f"{task['suite']}:{task['case_name'][:40]}")
                passed, error = run_task(task)
                if passed:
                    passed_count += 1
                else:
                    failed_count += 1
                    tqdm.write(f"FAILED {task['case_name']}: {error}")
                progress.update(1)
    finally:
        RuntimeContext.GLOBAL_EXP_DIR = None
        LoggingUtils.cleanup_handlers()
    write_summary(tasks, passed_count, failed_count)
    if failed_count > 0:
        raise RuntimeError(
            f"Preprocessing suite failed: {failed_count} failures. See {RESULTS_CSV} and {CASES_DIR}."
        )
    print(f"Suite completed: {passed_count} passed")
    print(f"Cases: {CASES_DIR}")
    print(f"Results: {RESULTS_CSV}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
