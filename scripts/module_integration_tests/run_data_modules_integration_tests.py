from __future__ import annotations

import csv
import inspect
import json
import logging
import re
import shutil
import traceback
import warnings
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterator

import dacite
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

from cccma_ppp.architectures.mlp import cvae as _mlp_cvae  # noqa: F401
from cccma_ppp.architectures.mlp import (
    deterministic as _mlp_deterministic,  # noqa: F401
)
from cccma_ppp.architectures.unet import cvae as _unet_cvae  # noqa: F401
from cccma_ppp.architectures.unet import (
    deterministic as _unet_deterministic,  # noqa: F401
)
from cccma_ppp.configs import required_sample_dimensions
from cccma_ppp.generic import registry_imports  # noqa: F401
from cccma_ppp.generic.distributed import Distributed
from cccma_ppp.generic.runtime import RuntimeContext
from cccma_ppp.loss import kld as _loss_kld  # noqa: F401
from cccma_ppp.loss import loss as _loss_pipeline  # noqa: F401
from cccma_ppp.loss import utils_loss as _loss_utils  # noqa: F401
from cccma_ppp.train.train_configs import TrainConfig, prepare_config

warnings.filterwarnings("ignore")
logging.raiseExceptions = False


ROOT = Path("/fs/site7/eccc/crd/cccma/users/rna002/CCCma-PPP/scripts")
BASE_TRAIN_CONFIG = ROOT / "integration_suite_train_config.yaml"
BASE_INFERENCE_CONFIG = ROOT / "scripts" / "integration_suite_inference_config.yaml"
OUTPUT_DIR = ROOT / "../../output" / "data_modules_integration_test_results"

CASES_DIR = OUTPUT_DIR / "cases"
RESULTS_CSV = OUTPUT_DIR / "integration_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary.txt"

INIT_TIME_DIM, LEAD_TIME_DIM = required_sample_dimensions


TAS_STDR_PATH = Path(
    "/space/hall7/sitestore/eccc/crd/cccma/users/rpg002/data/tas_std/forecast_data"
)
DUMMY_MASK_PATH = Path(
    "/space/hall7/sitestore/eccc/crd/cccma/users/rpg002/data/tas/dummy_mask"
)


RUN_DATA_CONFIG_COMBINATIONS = True
RUN_DATASET_GETITEM_CASES = True
RUN_TRAIN_DATALOADER_CASES = True
RUN_INFERENCE_DATALOADER_CASES = True
RUN_MULTIWORKER_CASES = True
RUN_FUTURE_MASK_CASES = True
RUN_OBSERVATION_CUTOFF_CASES = True
RUN_DISTRIBUTED_PREPROCESSOR_CASES = True
RUN_EXTERNAL_CONDITION_CASES = True
RUN_CHANNEL_PLOTS = True

MAX_CASES: int | None = None
MAX_BATCHES_PER_CASE = 4
MAX_PLOT_CHANNELS = 16
SHUFFLE_SEED = 1847


class DataCombinationScenarios:
    VALID_TIME_FEATURES = {
        INIT_TIME_DIM,
        LEAD_TIME_DIM,
        "month_sin",
        "month_cos",
        "day_sin",
        "day_cos",
    }
    DEFAULT_TIME_FEATURES = [LEAD_TIME_DIM, "month_sin", "month_cos"]
    FULL_LEAD_MONTHS = {"start": 1, "end": 12}

    AUTOENCODE_MODEL_ONLY = {
        "name": "autoencode_model_conditioned_on_itself",
        "description": (
            "Autoencode model data: reconstruct model data conditioned on itself. "
            "Condition is sampled according to condition_method."
        ),
        "model_present": True,
        "condition_present": False,
        "observation_present": False,
        "condition_method_requirement": "MUST",
        "condition_method": "ensemble_mean",
        "model_ensemble_mean": True,
        "expected_input": "model data selected according to condition_method",
        "expected_target": "model data at sampled init_time, lead_time, and ensemble coords",
        "expected_result": "PASS",
    }

    AUTOENCODE_MODEL_WITH_CONDITION = {
        "name": "autoencode_model_with_explicit_condition",
        "description": (
            "Autoencode model data using an explicit condition. "
            "Condition is sampled according to condition_method."
        ),
        "model_present": True,
        "condition_present": True,
        "observation_present": False,
        "condition_method_requirement": "MUST",
        "condition_method": "ensemble_mean",
        "model_ensemble_mean": True,
        "condition_ensemble_mean": True,
        "expected_input": "explicit condition data selected according to condition_method",
        "expected_target": "model data at sampled init_time, lead_time, and ensemble coords",
        "expected_result": "PASS",
    }

    BIAS_CORRECT_MODEL_AND_CONDITION = {
        "name": "bias_correct_model_and_condition_to_observation",
        "description": (
            "Bias-correct to observations using model and condition as conditioning fields. "
            "Condition sampling is determined by condition_method."
        ),
        "model_present": True,
        "condition_present": True,
        "observation_present": True,
        "condition_method_requirement": "MUST",
        "condition_method": "same_member",
        "model_ensemble_mean": False,
        "condition_ensemble_mean": False,
        "expected_input": "model + condition data at compatible sample coordinates",
        "expected_target": "observation data at calculated valid time",
        "expected_result": "PASS",
    }

    BIAS_CORRECT_MODEL_WITH_METHOD = {
        "name": "bias_correct_model_to_observation_with_method",
        "description": (
            "Bias-correct to observations using model data. "
            "Model sampling is determined by condition_method."
        ),
        "model_present": True,
        "condition_present": False,
        "observation_present": True,
        "condition_method_requirement": "OPTIONAL",
        "condition_method": "ensemble_mean",
        "model_ensemble_mean": True,
        "expected_input": "model data selected according to condition_method",
        "expected_target": "observation data at calculated valid time",
        "expected_result": "PASS",
    }

    BIAS_CORRECT_MODEL_DIRECT = {
        "name": "bias_correct_model_to_observation_without_method",
        "description": (
            "Bias-correct to observations using directly indexed model samples. "
            "No condition_method is applied."
        ),
        "model_present": True,
        "condition_present": False,
        "observation_present": True,
        "condition_method_requirement": "OPTIONAL",
        "condition_method": None,
        "model_ensemble_mean": False,
        "expected_input": "model data at sampled init_time, lead_time, and ensemble coords",
        "expected_target": "observation data at calculated valid time",
        "expected_result": "PASS",
    }

    @classmethod
    def all(cls) -> list[dict[str, Any]]:
        return [
            cls.AUTOENCODE_MODEL_ONLY,
            cls.AUTOENCODE_MODEL_WITH_CONDITION,
            cls.BIAS_CORRECT_MODEL_AND_CONDITION,
            cls.BIAS_CORRECT_MODEL_WITH_METHOD,
            cls.BIAS_CORRECT_MODEL_DIRECT,
        ]


class ExternalConditionScenarios:
    TAS_STDR_CONDITION = {
        "name": "bias_correct_with_tas_stdr_condition",
        "description": "Use tas_stdr as a real-data condition without ensemble dimension.",
        "model_present": True,
        "condition_present": True,
        "observation_present": True,
        "condition_kind": "tas_stdr",
        "condition_path": str(TAS_STDR_PATH),
        "condition_method_requirement": "MUST",
        "condition_method": "ensemble_mean",
        "model_ensemble_mean": False,
        "condition_ensemble_mean": None,
        "condition_has_ensemble": False,
        "expected_input": "model data concatenated with ensemble-free tas_stdr condition",
        "expected_target": "observation data at calculated valid time",
        "expected_result": "PASS",
    }

    STATIC_MASK_CONDITION = {
        "name": "bias_correct_with_static_dummy_mask",
        "description": "Use static dummy mask as an additional static condition.",
        "model_present": True,
        "condition_present": True,
        "observation_present": True,
        "condition_kind": "static",
        "condition_path": str(DUMMY_MASK_PATH),
        "condition_method_requirement": "MUST",
        "condition_method": "static",
        "model_ensemble_mean": False,
        "condition_ensemble_mean": None,
        "condition_has_ensemble": False,
        "expected_input": "model data concatenated with static dummy-mask channel",
        "expected_target": "observation data at calculated valid time",
        "expected_result": "PASS",
    }

    @classmethod
    def all(cls) -> list[dict[str, Any]]:
        return [cls.TAS_STDR_CONDITION, cls.STATIC_MASK_CONDITION]


class DataloaderScenarios:
    YEAR_SLICE_ONLY = {
        "name": "train_year_slice",
        "description": "Select exactly the configured training-year range and iterate fully.",
        "year_slice": [1980, 1981],
        "num_data_workers": 0,
        "shuffle": False,
        "time_features": deepcopy(DataCombinationScenarios.DEFAULT_TIME_FEATURES),
        "check": "year_slice",
        "expected_result": "PASS",
    }

    NO_TIME_FEATURES = {
        "name": "train_time_features_none",
        "description": "Build batches without any additional time features.",
        "year_slice": [1980, 1981],
        "num_data_workers": 0,
        "shuffle": False,
        "time_features": None,
        "check": "time_features",
        "expected_result": "PASS",
    }

    ALL_TIME_FEATURES = {
        "name": "train_time_features_all",
        "description": "Build batches with all supported time features.",
        "year_slice": [1980, 1981],
        "num_data_workers": 0,
        "shuffle": False,
        "time_features": sorted(DataCombinationScenarios.VALID_TIME_FEATURES),
        "check": "time_features",
        "expected_result": "PASS",
    }

    MULTIWORKER_NO_SHUFFLE = {
        "name": "train_multiworker_no_shuffle",
        "description": (
            "Multiple data workers must preserve sample coverage and deterministic order "
            "when shuffling is disabled."
        ),
        "year_slice": [1980, 1981],
        "num_data_workers": 2,
        "shuffle": False,
        "time_features": deepcopy(DataCombinationScenarios.DEFAULT_TIME_FEATURES),
        "check": "multiworker",
        "expected_result": "PASS",
    }

    MULTIWORKER_WITH_SHUFFLE = {
        "name": "train_multiworker_shuffle",
        "description": (
            "Multiple data workers with shuffling must preserve sample coverage "
            "while randomizing order."
        ),
        "year_slice": [1980, 1981],
        "num_data_workers": 2,
        "shuffle": True,
        "time_features": deepcopy(DataCombinationScenarios.DEFAULT_TIME_FEATURES),
        "check": "multiworker",
        "expected_result": "PASS",
    }

    @classmethod
    def all(cls) -> list[dict[str, Any]]:
        return [
            cls.YEAR_SLICE_ONLY,
            cls.NO_TIME_FEATURES,
            cls.ALL_TIME_FEATURES,
            cls.MULTIWORKER_NO_SHUFFLE,
            cls.MULTIWORKER_WITH_SHUFFLE,
        ]


class TemporalMaskScenarios:
    BEFORE_REQUESTED_TIMES = {
        "name": "observation_cutoff_before_requested_times",
        "description": (
            "Observation cutoff is before all requested valid times. "
            "Should return no valid samples."
        ),
        "observation_cutoff_position": "before",
        "expected_available": 0,
        "expected_result": "PASS",
    }

    BETWEEN_REQUESTED_TIMES = {
        "name": "observation_cutoff_between_requested_times",
        "description": (
            "Observation cutoff falls between first and last requested valid times. "
            "Should partially filter samples."
        ),
        "observation_cutoff_position": "between",
        "expected_result": "PASS",
    }

    AFTER_REQUESTED_TIMES = {
        "name": "observation_cutoff_after_requested_times",
        "description": (
            "Observation cutoff is after all requested valid times. "
            "Should return all samples."
        ),
        "observation_cutoff_position": "after",
        "expected_result": "PASS",
    }

    @classmethod
    def all(cls) -> list[dict[str, Any]]:
        return [
            cls.BEFORE_REQUESTED_TIMES,
            cls.BETWEEN_REQUESTED_TIMES,
            cls.AFTER_REQUESTED_TIMES,
        ]


CSV_COLUMNS = [
    "timestamp",
    "suite",
    "case_name",
    "description",
    "model_present",
    "condition_present",
    "observation_present",
    "condition_kind",
    "condition_method_requirement",
    "condition_method",
    "model_ensemble_mean",
    "condition_ensemble_mean",
    "time_features",
    "year_slice",
    "num_data_workers",
    "expected_input",
    "expected_target",
    "expected_result",
    "expected_failure_reason",
    "expected_error_pattern",
    "case_dir",
    "config_path",
    "log_path",
    "preprocessor_dir",
    "plot_dir",
    "result",
    "error",
]

PATH_FIELD_NAMES = {
    "path",
    "paths",
    "data_path",
    "data_paths",
    "directory",
    "data_directory",
    "root",
    "store",
    "url",
    "filename",
    "filenames",
    "file_pattern",
    "glob_pattern",
}

ENSEMBLE_FIELD_NAMES = {
    "ensemble",
    "ensembles",
    "ensemble_dim",
    "ensemble_dimension",
    "ensemble_members",
    "ensemble_mean",
    "member",
    "members",
    "member_dim",
    "member_dimension",
}


def clean_name(value: Any) -> str:
    value = str(value).replace("None", "none")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def case_paths(case_name: str) -> dict[str, Path]:
    case_dir = CASES_DIR / clean_name(case_name)
    return {
        "case_dir": case_dir,
        "config_path": case_dir / "config.yaml",
        "log_path": case_dir / "case.log",
        "preprocessor_dir": case_dir / "preprocessing_pipeline",
        "plot_dir": case_dir / "plots",
        "metadata_path": case_dir / "batch_metadata.json",
    }


def prepare_output_directories() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CASES_DIR.mkdir(parents=True, exist_ok=True)

    obsolete = [
        OUTPUT_DIR / "_generated_configs",
        OUTPUT_DIR / "_logs",
        OUTPUT_DIR / "_preprocessing_pipeline",
        OUTPUT_DIR / "_parallel_monitoring",
    ]
    for directory in obsolete:
        if directory.exists():
            shutil.rmtree(directory)


def serialize_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def init_csv() -> None:
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
) -> None:
    params = task["params"]
    row = {
        "timestamp": datetime.now().isoformat(),
        "suite": task["suite"],
        "case_name": task["case_name"],
        "description": params.get("description", ""),
        "model_present": params.get("model_present"),
        "condition_present": params.get("condition_present"),
        "observation_present": params.get("observation_present"),
        "condition_kind": params.get("condition_kind", "dynamic"),
        "condition_method_requirement": params.get("condition_method_requirement"),
        "condition_method": params.get("condition_method"),
        "model_ensemble_mean": params.get("model_ensemble_mean"),
        "condition_ensemble_mean": params.get("condition_ensemble_mean"),
        "time_features": serialize_json(params.get("time_features")),
        "year_slice": serialize_json(params.get("year_slice")),
        "num_data_workers": params.get("num_data_workers"),
        "expected_input": params.get("expected_input"),
        "expected_target": params.get("expected_target"),
        "expected_result": params.get("expected_result", "PASS"),
        "expected_failure_reason": params.get("expected_failure_reason", ""),
        "expected_error_pattern": params.get("expected_error_pattern", ""),
        "case_dir": str(paths["case_dir"]),
        "config_path": str(paths["config_path"]),
        "log_path": str(paths["log_path"]),
        "preprocessor_dir": str(paths["preprocessor_dir"]),
        "plot_dir": str(paths["plot_dir"]),
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
    loader_cfg = cfg.setdefault("train_loader", {})
    dataset_cfg = loader_cfg.setdefault("dataset_config", {})

    if "time_features" in dataset_cfg:
        raise ValueError(
            "time_features moved to train_loader level, not dataset_config"
        )

    loader_cfg.pop("time_features", None) if loader_cfg.get(
        "time_features"
    ) is None else None
    dataset_cfg.pop("condition_method", None) if dataset_cfg.get(
        "condition_method"
    ) is None else None
    dataset_cfg.pop("lead_times", None) if dataset_cfg.get(
        "lead_times"
    ) is None else None

    return cfg


def validate_generated_config(cfg: dict[str, Any]) -> dict[str, Any]:
    train_loader_cfg = cfg.get("train_loader")
    if not isinstance(train_loader_cfg, dict):
        raise ValueError("train_loader must be a mapping")

    dataset_cfg = train_loader_cfg.get("dataset_config")
    if not isinstance(dataset_cfg, dict):
        raise ValueError("train_loader.dataset_config must be a mapping")

    runtime_loader_arguments = {"shuffle", "return_metadata", "return_spatial_mask"}
    misplaced = runtime_loader_arguments & set(train_loader_cfg)
    if misplaced:
        raise ValueError(f"Runtime arguments in config: {sorted(misplaced)}")

    if "time_features" in dataset_cfg:
        raise ValueError(
            "time_features must be in train_loader, not train_loader.dataset_config"
        )

    time_features = train_loader_cfg.get("time_features")
    if time_features is not None:
        if not isinstance(time_features, list):
            raise TypeError("time_features must be a list or null")
        unsupported = set(time_features) - DataCombinationScenarios.VALID_TIME_FEATURES
        if unsupported:
            raise ValueError(f"Unsupported time_features: {sorted(unsupported)}")

    if "model" not in dataset_cfg:
        raise ValueError("dataset_config.model is required")

    num_data_workers = train_loader_cfg.get("num_data_workers", 0)
    if (
        isinstance(num_data_workers, bool)
        or not isinstance(num_data_workers, int)
        or num_data_workers < 0
    ):
        raise ValueError("num_data_workers must be a non-negative integer")

    if num_data_workers == 0 and train_loader_cfg.get("prefetch_factor") is not None:
        raise ValueError("prefetch_factor must be null when num_data_workers=0")

    return cfg


def write_yaml(cfg: dict[str, Any], output_path: Path) -> Path:
    cfg = deepcopy(cfg)
    normalize_nullable_fields(cfg)
    validate_generated_config(cfg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(cfg, file, sort_keys=False)
    return output_path


def base_config() -> dict[str, Any]:
    cfg = prepare_config(BASE_TRAIN_CONFIG)
    if cfg is None:
        raise ValueError(f"Base config is empty: {BASE_TRAIN_CONFIG}")

    cfg = deepcopy(cfg)
    train_loader_cfg = cfg.get("train_loader")
    if not isinstance(train_loader_cfg, dict):
        raise ValueError("Base config must have train_loader mapping")

    dataset_cfg = train_loader_cfg.get("dataset_config")
    if not isinstance(dataset_cfg, dict):
        raise ValueError("Base config must have train_loader.dataset_config")

    if "time_features" in dataset_cfg:
        raise ValueError("Move time_features to train_loader level")

    train_loader_cfg["batch_size"] = 2
    train_loader_cfg["num_data_workers"] = 0
    train_loader_cfg["num_validation_years"] = 0
    train_loader_cfg["train_years_slice"] = [1980, 1981]
    train_loader_cfg["load"] = False

    for key in ["shuffle", "return_metadata", "return_spatial_mask", "prefetch_factor"]:
        train_loader_cfg.pop(key, None)

    normalize_nullable_fields(cfg)
    validate_generated_config(cfg)
    return cfg


def load_train_config(config_path: Path) -> TrainConfig:
    data = prepare_config(config_path)
    if data is None:
        raise ValueError(f"Config is empty: {config_path}")

    normalize_nullable_fields(data)
    validate_generated_config(data)

    return dacite.from_dict(
        data_class=TrainConfig,
        data=data,
        config=dacite.Config(strict=True),
    )


def replace_data_path(value: Any, new_path: str) -> tuple[Any, int]:
    replaced = 0

    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            if str(key).lower() in PATH_FIELD_NAMES or str(key).lower().endswith(
                ("_path", "_paths", "_directory", "_file", "_files")
            ):
                result[key] = [new_path] if isinstance(child, list) else new_path
                replaced += 1
            else:
                result[key], child_replaced = replace_data_path(child, new_path)
                replaced += child_replaced
        return result, replaced

    if isinstance(value, list):
        result = []
        for child in value:
            updated, child_replaced = replace_data_path(child, new_path)
            result.append(updated)
            replaced += child_replaced
        return result, replaced

    return deepcopy(value), 0


def remove_ensemble_configuration(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: remove_ensemble_configuration(child)
            for key, child in value.items()
            if str(key).lower() not in ENSEMBLE_FIELD_NAMES
        }
    if isinstance(value, list):
        return [remove_ensemble_configuration(child) for child in value]
    return deepcopy(value)


def apply_data_combination(
    cfg: dict[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
    train_loader_cfg = cfg["train_loader"]
    dataset_cfg = train_loader_cfg["dataset_config"]

    if not params.get("model_present", True):
        raise ValueError("Model data is required")

    if "model_config_override" in params:
        dataset_cfg["model"] = deepcopy(params["model_config_override"])

    model_cfg = dataset_cfg["model"]
    if "model_ensemble_mean" in params:
        model_cfg["ensemble_mean"] = params["model_ensemble_mean"]

    if params.get("condition_present", False):
        if "condition_config_override" in params:
            condition_cfg = deepcopy(params["condition_config_override"])
        elif params.get("condition_path"):
            source_cfg = dataset_cfg.get("condition", model_cfg)
            condition_cfg = build_external_condition_config(
                source_cfg,
                path=params["condition_path"],
                has_ensemble=params.get("condition_has_ensemble", True),
                condition_kind=params.get("condition_kind", "dynamic"),
            )
        else:
            condition_cfg = deepcopy(dataset_cfg.get("condition", model_cfg))

        if params.get("condition_ensemble_mean") is not None:
            condition_cfg["ensemble_mean"] = params["condition_ensemble_mean"]

        dataset_cfg["condition"] = condition_cfg
    else:
        dataset_cfg.pop("condition", None)

    if params.get("observation_present", False):
        if "observation_config_override" in params:
            dataset_cfg["observation"] = deepcopy(params["observation_config_override"])
        elif "observation" not in dataset_cfg:
            raise ValueError(
                "observation_present=True but no observation in base config"
            )
    else:
        dataset_cfg.pop("observation", None)

    condition_method = params.get("condition_method")
    if condition_method is None:
        dataset_cfg.pop("condition_method", None)
    else:
        dataset_cfg["condition_method"] = condition_method

    if "time_features" in params:
        if params["time_features"] is None:
            train_loader_cfg.pop("time_features", None)
        else:
            train_loader_cfg["time_features"] = deepcopy(params["time_features"])

    if "year_slice" in params:
        train_loader_cfg["train_years_slice"] = deepcopy(params["year_slice"])

    if "lead_months" in params:
        if params["lead_months"] is None:
            dataset_cfg.pop("lead_times", None)
        else:
            dataset_cfg["lead_times"] = deepcopy(params["lead_months"])

    if "num_data_workers" in params:
        num_data_workers = params["num_data_workers"]
        if (
            isinstance(num_data_workers, bool)
            or not isinstance(num_data_workers, int)
            or num_data_workers < 0
        ):
            raise ValueError("num_data_workers must be non-negative integer")
        train_loader_cfg["num_data_workers"] = num_data_workers

    num_data_workers = train_loader_cfg.get("num_data_workers", 0)
    if num_data_workers == 0:
        train_loader_cfg.pop("prefetch_factor", None)
    else:
        train_loader_cfg.setdefault("prefetch_factor", 2)

    for key in ["shuffle", "return_metadata", "return_spatial_mask"]:
        train_loader_cfg.pop(key, None)

    normalize_nullable_fields(cfg)
    validate_generated_config(cfg)
    return cfg


def build_external_condition_config(
    source_cfg: dict[str, Any],
    *,
    path: str,
    has_ensemble: bool,
    condition_kind: str,
) -> dict[str, Any]:
    condition_cfg = deepcopy(source_cfg)
    condition_cfg, replacements = replace_data_path(condition_cfg, path)

    if replacements == 0:
        raise ValueError("No path-like fields found in condition config")

    if not has_ensemble:
        condition_cfg = remove_ensemble_configuration(condition_cfg)

    if condition_kind == "static":
        for key in [
            "lead_times",
            "init_times",
            "initialization_times",
            "time_slice",
            "year_slice",
            "train_years_slice",
        ]:
            condition_cfg.pop(key, None)

    return condition_cfg


def cleanup_logging_handlers() -> None:
    logger_names = ["", "training", "inference"]
    logger_names.extend(
        name
        for name in logging.Logger.manager.loggerDict
        if name.startswith("cccma_ppp")
    )

    for name in dict.fromkeys(logger_names):
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.flush()
            except Exception:
                pass
            try:
                handler.close()
            except Exception:
                pass
        logger.propagate = False


def run_silenced(function: Callable[[], Any], log_path: Path) -> Any:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cleanup_logging_handlers()

    with log_path.open("w", encoding="utf-8") as log_file:
        try:
            with redirect_stdout(log_file), redirect_stderr(log_file):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return function()
        finally:
            cleanup_logging_handlers()


def append_exception_to_log(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as file:
        traceback.print_exc(file=file)


def configure_runtime_context(config_path: Path) -> Path:
    case_dir = Path(config_path).resolve().parent
    case_dir.mkdir(parents=True, exist_ok=True)
    RuntimeContext.GLOBAL_EXP_DIR = str(case_dir)
    return case_dir


def reset_runtime_context() -> None:
    RuntimeContext.GLOBAL_EXP_DIR = None


def call_with_supported_arguments(function: Callable[..., Any], **kwargs: Any) -> Any:
    signature = inspect.signature(function)
    supported = {
        name: value for name, value in kwargs.items() if name in signature.parameters
    }
    return function(**supported)


def setup_loader(
    config: TrainConfig,
    config_path: Path,
    *,
    return_metadata: bool = False,
    return_spatial_mask: bool = True,
    shuffle: bool = False,
) -> Any:
    configure_runtime_context(config_path)
    distributed = Distributed.get_instance()

    call_with_supported_arguments(
        config.train_loader.setup_distributed,
        distributed=distributed,
    )

    return call_with_supported_arguments(
        config.train_loader.build_train_loader,
        return_metadata=return_metadata,
        return_spatial_mask=return_spatial_mask,
        shuffle=shuffle,
    )


class LimitedLoader:
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


def sample_get(sample: Any, name: str) -> Any:
    if isinstance(sample, dict):
        return sample.get(name)
    return getattr(sample, name, None)


def extract_metadata(batch: Any) -> Any:
    return sample_get(batch, "metadata")


def value_is_batched(value: Any) -> bool:
    if isinstance(value, str):
        return False
    if isinstance(value, torch.Tensor):
        return value.ndim > 0
    if isinstance(value, np.ndarray):
        return value.ndim > 0
    return isinstance(value, (list, tuple))


def value_length(value: Any) -> int | None:
    if not value_is_batched(value):
        return None
    try:
        return len(value)
    except TypeError:
        return None


def index_value(value: Any, index: int) -> Any:
    if not value_is_batched(value):
        return value

    indexed = value[index]
    if isinstance(indexed, torch.Tensor) and indexed.ndim == 0:
        return indexed.item()
    if isinstance(indexed, np.ndarray) and indexed.ndim == 0:
        return indexed.item()
    return indexed


def flatten_metadata(metadata: Any) -> list:
    if metadata is None:
        return []
    if isinstance(metadata, list):
        return metadata
    if isinstance(metadata, tuple):
        return list(metadata)
    if isinstance(metadata, dict):
        lengths = [
            length
            for value in metadata.values()
            if (length := value_length(value)) is not None
        ]

        if not lengths:
            return [metadata]

        if len(set(lengths)) != 1:
            raise AssertionError(f"Inconsistent metadata field lengths: {lengths}")

        count = lengths[0]
        return [
            {key: index_value(value, i) for key, value in metadata.items()}
            for i in range(count)
        ]

    return [metadata]


def scalar_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return value.detach().cpu().tolist()
        return value.detach().cpu().item()
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return value.tolist()
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def canonical_value(value: Any) -> Any:
    value = scalar_value(value)

    if isinstance(value, dict):
        return {
            str(key): canonical_value(child)
            for key, child in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [canonical_value(child) for child in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return str(value.astype("datetime64[ns]"))
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass

    return value


def canonical_sample_key(item: Any) -> str:
    if isinstance(item, dict):
        value = item
    elif hasattr(item, "__dict__"):
        value = vars(item)
    else:
        value = item

    return serialize_json(canonical_value(value))


def assert_tensor(name: str, value: Any) -> None:
    if not isinstance(value, torch.Tensor):
        raise AssertionError(f"{name} is not a tensor: {type(value).__name__}")
    if value.numel() == 0:
        raise AssertionError(f"{name} is empty")
    if torch.is_floating_point(value) and not torch.isfinite(value).all():
        raise AssertionError(f"{name} contains non-finite values")


def collect_loader_metadata(loader: Any, max_batches: int | None = None) -> list:
    items = []
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        batch_metadata = flatten_metadata(extract_metadata(batch))
        items.extend(batch_metadata)
    return items


def save_metadata(metadata: list[Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            canonical_value(metadata), file, indent=2, sort_keys=True, default=str
        )


def channel_axis(tensor: torch.Tensor) -> int:
    return 0 if tensor.ndim < 2 else 1


def first_channel_image(tensor: torch.Tensor, channel_index: int) -> np.ndarray:
    value = tensor.detach().float().cpu()

    if value.ndim >= 4:
        value = value[0, channel_index]
    elif value.ndim == 3:
        value = value[channel_index]
    elif value.ndim == 2:
        value = value[channel_index].unsqueeze(0)
    else:
        value = value.reshape(1, -1)

    while value.ndim > 2:
        value = value[0]

    return value.numpy()


def plot_tensor_channels(
    tensor: torch.Tensor,
    *,
    name: str,
    output_dir: Path,
    metadata: Any,
) -> None:

    if not RUN_CHANNEL_PLOTS:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    axis = channel_axis(tensor)

    if tensor.ndim <= axis:
        return

    channels = min(
        int(tensor.shape[axis]),
        MAX_PLOT_CHANNELS,
    )

    if channels == 0:
        return

    for channel_index in range(channels):
        try:
            image = first_channel_image(
                tensor,
                channel_index,
            )
        except (IndexError, ValueError):
            continue

        figure, axes = plt.subplots(
            1,
            1,
            figsize=(6, 5),
        )

        visual = axes.imshow(
            image,
            origin="lower",
            aspect="auto",
        )

        axes.set_title(f"{name}, channel {channel_index}")

        axes.set_xlabel(clean_name(serialize_json(canonical_value(metadata)))[:80])

        figure.colorbar(
            visual,
            ax=axes,
            shrink=0.8,
        )

        figure.tight_layout()

        figure.savefig(
            output_dir / f"{clean_name(name)}_channel_{channel_index:03d}.png",
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(figure)


def plot_first_batch(batch: Any, *, output_dir: Path) -> None:
    metadata = flatten_metadata(extract_metadata(batch))
    first_metadata = metadata[0] if metadata else {}

    for name in ("input", "target", "added_features", "input_mask", "target_mask"):
        tensor = sample_get(batch, name)
        if isinstance(tensor, torch.Tensor):
            plot_tensor_channels(
                tensor, name=name, output_dir=output_dir, metadata=first_metadata
            )


def metadata_value(item: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        if isinstance(item, dict) and name in item:
            return item[name]
        if hasattr(item, name):
            return getattr(item, name)
    return None


def metadata_init_time(item: Any) -> Any:
    return metadata_value(item, (INIT_TIME_DIM, "init_time", "initialization_time"))


def metadata_lead_time(item: Any) -> Any:
    return metadata_value(item, (LEAD_TIME_DIM, "lead_time"))


def to_datetime64_month(value: Any) -> np.datetime64:
    value = scalar_value(value)
    return np.datetime64(value, "M")


def to_integer_lead(value: Any) -> int:
    value = scalar_value(value)
    if isinstance(value, np.timedelta64):
        return int(value.astype("timedelta64[M]").astype(int))
    return int(value)


def valid_time_from_metadata(item: Any) -> np.datetime64:
    init_time = metadata_init_time(item)
    lead_time = metadata_lead_time(item)

    if init_time is None or lead_time is None:
        raise AssertionError("Metadata missing init_time or lead_time")

    lead = to_integer_lead(lead_time)
    if lead < 1:
        raise AssertionError(f"Lead time must be 1-based, got {lead}")

    return to_datetime64_month(init_time) + np.timedelta64(lead - 1, "M")


def assert_metadata_indexing(metadata: list[Any]) -> None:
    if not metadata:
        raise AssertionError("No metadata available")

    keys = []
    for item in metadata:
        init_time = metadata_init_time(item)
        lead_time = metadata_lead_time(item)

        if init_time is None:
            raise AssertionError(f"Missing {INIT_TIME_DIM} in {item}")
        if lead_time is None:
            raise AssertionError(f"Missing {LEAD_TIME_DIM} in {item}")

        lead = to_integer_lead(lead_time)
        if lead < 1:
            raise AssertionError(f"Lead time not 1-based: {lead}")

        valid_time = valid_time_from_metadata(item)
        if valid_time < to_datetime64_month(init_time):
            raise AssertionError(
                f"Valid time precedes init time: "
                f"init={init_time}, lead={lead}, valid={valid_time}"
            )

        keys.append(canonical_sample_key(item))

    if len(keys) != len(set(keys)):
        counts = Counter(keys)
        duplicates = {key: count for key, count in counts.items() if count > 1}
        raise AssertionError(f"Duplicate samples in metadata: {duplicates}")


def metadata_year(item: Any) -> int:
    value = metadata_init_time(item)
    if value is None:
        raise AssertionError("Could not extract init_time")

    value = scalar_value(value)
    if hasattr(value, "year"):
        return int(value.year)

    return int(np.datetime64(value, "Y").astype(int) + 1970)


def assert_dataset_getitem_contract(dataset: Any, params: dict[str, Any]) -> None:
    if len(dataset) <= 0:
        raise AssertionError("Dataset is empty")

    indices = sorted({0, len(dataset) // 2, len(dataset) - 1})
    metadata_keys = []

    for index in indices:
        sample = dataset[index]

        if not isinstance(sample, tuple):
            raise AssertionError(
                f"__getitem__ must return tuple, got {type(sample).__name__}"
            )
        if len(sample) != 2:
            raise AssertionError(f"__getitem__ must return 2 values, got {len(sample)}")

        sample_data, metadata = sample

        if not isinstance(sample_data, dict):
            raise AssertionError(
                f"Sample data must be dict, got {type(sample_data).__name__}"
            )
        if not isinstance(metadata, dict):
            raise AssertionError(
                f"Metadata must be dict, got {type(metadata).__name__}"
            )

        required_fields = {"input", "target"}
        missing = required_fields - set(sample_data)
        if missing:
            raise AssertionError(f"Missing sample fields: {sorted(missing)}")

        assert_tensor(f"sample[{index}].input", sample_data["input"])
        assert_tensor(f"sample[{index}].target", sample_data["target"])

        features = params.get(
            "time_features", DataCombinationScenarios.DEFAULT_TIME_FEATURES
        )
        added_features = sample_data.get("added_features")

        if features:
            assert_tensor(f"sample[{index}].added_features", added_features)
        elif isinstance(added_features, torch.Tensor) and added_features.numel() > 0:
            raise AssertionError("added_features populated but time_features is empty")

        required_meta = {INIT_TIME_DIM, LEAD_TIME_DIM}
        missing_meta = required_meta - set(metadata)
        if missing_meta:
            raise AssertionError(f"Missing metadata fields: {sorted(missing_meta)}")

        metadata_keys.append(canonical_sample_key(metadata))

    if len(metadata_keys) != len(set(metadata_keys)):
        raise AssertionError(f"Duplicate metadata at indices {indices}")

    dataset_config = getattr(dataset, "config", None)
    if params.get("observation_present"):
        if getattr(dataset_config, "observation", None) is None:
            raise AssertionError("Observation requested but not in config")
    if params.get("condition_present"):
        if getattr(dataset_config, "condition", None) is None:
            raise AssertionError("Condition requested but not in config")

    print(f"dataset_length={len(dataset)}")
    print(f"checked_indices={indices}")
    print(f"metadata_keys={metadata_keys}")


def assert_batch_contract(
    loader: Any,
    time_features: list[str] | None,
    *,
    plot_dir: Path | None = None,
    metadata_path: Path | None = None,
) -> list:
    batches = 0
    all_metadata = []

    for batch in LimitedLoader(loader, MAX_BATCHES_PER_CASE):
        input_tensor = sample_get(batch, "input")
        target_tensor = sample_get(batch, "target")
        added_features = sample_get(batch, "added_features")

        assert_tensor("batch.input", input_tensor)

        if target_tensor is not None:
            assert_tensor("batch.target", target_tensor)

        if time_features:
            assert_tensor("batch.added_features", added_features)
        elif isinstance(added_features, torch.Tensor) and added_features.numel() > 0:
            raise AssertionError("added_features populated but time_features is empty")

        metadata = flatten_metadata(extract_metadata(batch))
        if not metadata:
            raise AssertionError("No metadata returned")

        batch_size = int(input_tensor.shape[0])
        if len(metadata) != batch_size:
            raise AssertionError(
                f"Metadata count ({len(metadata)}) != batch size ({batch_size})"
            )

        all_metadata.extend(metadata)

        if batches == 0 and plot_dir is not None:
            plot_first_batch(batch, output_dir=plot_dir)

        batches += 1

    if batches == 0:
        raise AssertionError("Dataloader returned no batches")

    if metadata_path is not None:
        save_metadata(all_metadata, metadata_path)

    return all_metadata


def assert_year_slice(loader: Any, year_slice: list[int]) -> None:
    metadata = collect_loader_metadata(loader, max_batches=None)

    if not metadata:
        raise AssertionError("No metadata for year-slice validation")

    assert_metadata_indexing(metadata)

    lower, upper = year_slice
    years = [metadata_year(item) for item in metadata]
    observed_years = set(years)

    if min(years) < lower or max(years) > upper:
        raise AssertionError(
            f"Years outside [{lower}, {upper}]: {sorted(observed_years)}"
        )

    expected_years = set(range(lower, upper + 1))
    missing_years = expected_years - observed_years

    if missing_years:
        raise AssertionError(f"Missing years: {sorted(missing_years)}")


def assert_no_future_leakage(loader: Any, cutoff: Any) -> None:
    cutoff_month = to_datetime64_month(cutoff)
    metadata = collect_loader_metadata(loader, max_batches=None)

    if not metadata:
        raise AssertionError("No metadata for leakage validation")

    inspected = 0
    for item in metadata:
        valid_time = valid_time_from_metadata(item)
        if valid_time > cutoff_month:
            raise AssertionError(
                f"Future leakage: valid_time={valid_time} > cutoff={cutoff_month}"
            )
        inspected += 1

    if inspected == 0:
        raise AssertionError("No samples inspected")


def requested_valid_times(loader: Any) -> list[np.datetime64]:
    metadata = collect_loader_metadata(loader, max_batches=None)
    return sorted({valid_time_from_metadata(item) for item in metadata})


def cutoff_for_position(
    valid_times: list[np.datetime64], position: str
) -> np.datetime64:
    if not valid_times:
        raise AssertionError("No valid times")

    if position == "before":
        return valid_times[0] - np.timedelta64(1, "M")
    elif position == "between":
        return valid_times[len(valid_times) // 2]
    elif position == "after":
        return valid_times[-1] + np.timedelta64(1, "M")
    else:
        raise ValueError(f"Unknown position: {position}")


def assert_observation_cutoff_indexing(loader: Any, position: str) -> None:
    valid_times = requested_valid_times(loader)
    cutoff = cutoff_for_position(valid_times, position)

    available = [v for v in valid_times if v <= cutoff]
    unavailable = [v for v in valid_times if v > cutoff]

    if position == "before":
        if available:
            raise AssertionError(
                f"Before cutoff should exclude all, got {len(available)}"
            )
        if not unavailable:
            raise AssertionError("Before cutoff should exclude something")
    elif position == "between":
        if not available or not unavailable:
            raise AssertionError("Between cutoff must split times")
    elif position == "after":
        if unavailable:
            raise AssertionError(
                f"After cutoff should include all, excluded {len(unavailable)}"
            )

    print(f"observation_cutoff_position={position}")
    print(f"observation_cutoff={cutoff}")
    print(f"requested_valid_times={len(valid_times)}")
    print(f"available_valid_times={len(available)}")
    print(f"unavailable_valid_times={len(unavailable)}")


def set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def assert_multiworker_consistency(
    config: TrainConfig, shuffle: bool, config_path: Path
) -> None:
    configure_runtime_context(config_path)
    distributed = Distributed.get_instance()

    single_cfg = deepcopy(config.train_loader)
    single_cfg.num_data_workers = 0
    if hasattr(single_cfg, "prefetch_factor"):
        single_cfg.prefetch_factor = None

    call_with_supported_arguments(single_cfg.setup_distributed, distributed=distributed)
    set_torch_seed(SHUFFLE_SEED)

    single_loader = call_with_supported_arguments(
        single_cfg.build_train_loader,
        return_metadata=True,
        return_spatial_mask=True,
        shuffle=False,
    )

    single_items = collect_loader_metadata(single_loader, max_batches=None)

    multi_cfg = deepcopy(config.train_loader)
    call_with_supported_arguments(multi_cfg.setup_distributed, distributed=distributed)
    set_torch_seed(SHUFFLE_SEED)

    multi_loader = call_with_supported_arguments(
        multi_cfg.build_train_loader,
        return_metadata=True,
        return_spatial_mask=True,
        shuffle=shuffle,
    )

    multi_items = collect_loader_metadata(multi_loader, max_batches=None)

    single_keys = [canonical_sample_key(item) for item in single_items]
    multi_keys = [canonical_sample_key(item) for item in multi_items]

    if Counter(multi_keys) != Counter(single_keys):
        raise AssertionError(
            f"Multi-worker changed coverage: single={len(single_keys)}, multi={len(multi_keys)}"
        )

    if len(multi_keys) != len(set(multi_keys)):
        raise AssertionError("Multi-worker duplicated samples")

    if not shuffle and multi_keys != single_keys:
        raise AssertionError("No-shuffle multi-worker changed order")

    if shuffle and len(multi_keys) > 1 and multi_keys == single_keys:
        raise AssertionError("Shuffle had no effect")


def preprocessor_files(directory: Path) -> list:
    return sorted(directory.glob("*_preprocessing_pipeline.joblib"))


def assert_rank_zero_preprocessor_setup(config: TrainConfig, config_path: Path) -> None:
    case_dir = configure_runtime_context(config_path)
    preprocessor_dir = case_dir / "preprocessing_pipeline"

    if preprocessor_dir.exists():
        shutil.rmtree(preprocessor_dir)

    distributed = Distributed.get_instance()

    if distributed.rank != 0:
        loader_cfg = deepcopy(config.train_loader)
        call_with_supported_arguments(
            loader_cfg.setup_distributed,
            distributed=distributed,
            load_path=preprocessor_dir,
        )
        call_with_supported_arguments(
            loader_cfg.build_train_loader,
            return_metadata=True,
            return_spatial_mask=True,
            shuffle=False,
        )

        if not preprocessor_files(preprocessor_dir):
            raise AssertionError(f"Non-rank-0 did not load from {preprocessor_dir}")
        return

    call_with_supported_arguments(
        config.train_loader.setup_distributed, distributed=distributed
    )
    call_with_supported_arguments(
        config.train_loader.build_train_loader,
        return_metadata=True,
        return_spatial_mask=True,
        shuffle=False,
    )

    written = preprocessor_files(preprocessor_dir)
    if not written:
        raise AssertionError(f"Rank-0 did not save to {preprocessor_dir}")

    before_hashes = {
        path.name: (path.stat().st_size, path.stat().st_mtime_ns) for path in written
    }

    loader_cfg = deepcopy(config.train_loader)
    call_with_supported_arguments(
        loader_cfg.setup_distributed,
        distributed=distributed,
        load_path=preprocessor_dir,
    )
    call_with_supported_arguments(
        loader_cfg.build_train_loader,
        return_metadata=True,
        return_spatial_mask=True,
        shuffle=False,
    )

    after = preprocessor_files(preprocessor_dir)
    after_hashes = {
        path.name: (path.stat().st_size, path.stat().st_mtime_ns) for path in after
    }

    if after_hashes != before_hashes:
        raise AssertionError("Reload phase rewrote artifacts")

    if getattr(distributed, "world_size", 1) == 1:
        print("Note: Run with torchrun --nproc_per_node=2 for multi-rank validation")


def run_dataset_case(config_path: Path, params: dict[str, Any]) -> None:
    config = load_train_config(config_path)
    loader = setup_loader(
        config,
        config_path,
        return_metadata=True,
        return_spatial_mask=True,
        shuffle=False,
    )

    assert_dataset_getitem_contract(loader.dataset, params)

    paths = case_paths(params["name"])
    metadata = assert_batch_contract(
        loader,
        params.get("time_features", DataCombinationScenarios.DEFAULT_TIME_FEATURES),
        plot_dir=paths["plot_dir"],
        metadata_path=paths["metadata_path"],
    )

    assert_metadata_indexing(metadata)


def run_dataloader_case(config_path: Path, params: dict[str, Any]) -> None:
    config = load_train_config(config_path)
    check = params["check"]

    if check == "multiworker":
        assert_multiworker_consistency(config, params["shuffle"], config_path)
        return

    loader = setup_loader(
        config,
        config_path,
        return_metadata=True,
        return_spatial_mask=True,
        shuffle=params.get("shuffle", False),
    )

    paths = case_paths(params["name"])
    metadata = assert_batch_contract(
        loader,
        params.get("time_features"),
        plot_dir=paths["plot_dir"],
        metadata_path=paths["metadata_path"],
    )

    assert_metadata_indexing(metadata)

    if check == "year_slice":
        loader = setup_loader(
            config,
            config_path,
            return_metadata=True,
            return_spatial_mask=True,
            shuffle=False,
        )
        assert_year_slice(loader, params["year_slice"])


def run_future_mask_case(config_path: Path, params: dict[str, Any]) -> None:
    config = load_train_config(config_path)
    loader = setup_loader(
        config,
        config_path,
        return_metadata=True,
        return_spatial_mask=True,
        shuffle=False,
    )
    assert_no_future_leakage(loader, cutoff=params["cutoff"])


def run_observation_cutoff_case(config_path: Path, params: dict[str, Any]) -> None:
    config = load_train_config(config_path)
    loader = setup_loader(
        config,
        config_path,
        return_metadata=True,
        return_spatial_mask=True,
        shuffle=False,
    )
    assert_observation_cutoff_indexing(
        loader, position=params["observation_cutoff_position"]
    )


def run_preprocessor_case(config_path: Path, params: dict[str, Any]) -> None:
    config = load_train_config(config_path)
    assert_rank_zero_preprocessor_setup(config, config_path)


def run_inference_case(config_path: Path, params: dict[str, Any]) -> None:
    from cccma_ppp.inference.inference_configs import InferenceConfig

    configure_runtime_context(config_path)
    train_config = load_train_config(config_path)

    inference_data = prepare_config(params["inference_config_path"])
    if inference_data is None:
        raise ValueError("Inference config is empty")

    inference_config = dacite.from_dict(
        data_class=InferenceConfig,
        data=inference_data,
        config=dacite.Config(strict=True),
    )

    inference_loader_cfg = inference_config.inference_loader
    inference_loader_cfg.read_configs_from_train(train_config.train_loader)

    if inference_loader_cfg.time_features != train_config.train_loader.time_features:
        raise AssertionError("Inference did not preserve training time_features")

    distributed = Distributed.get_instance()

    setup_func = inference_loader_cfg.setup_distributed
    sig = inspect.signature(setup_func)
    kwargs = {}

    if "train_loader_config" in sig.parameters:
        kwargs["train_loader_config"] = train_config.train_loader
    elif "train_loader" in sig.parameters:
        kwargs["train_loader"] = train_config.train_loader
    elif "train_config" in sig.parameters:
        kwargs["train_config"] = train_config.train_loader

    if "distributed" in sig.parameters:
        kwargs["distributed"] = distributed

    if kwargs:
        setup_func(**kwargs)
    else:
        setup_func()

    loader = call_with_supported_arguments(
        inference_loader_cfg.build_inference_loader,
        return_metadata=True,
        return_spatial_mask=True,
    )

    paths = case_paths(params["name"])
    metadata = assert_batch_contract(
        loader,
        train_config.train_loader.time_features,
        plot_dir=paths["plot_dir"],
        metadata_path=paths["metadata_path"],
    )

    assert_metadata_indexing(metadata)

    if "cutoff" in params:
        assert_no_future_leakage(loader, cutoff=params["cutoff"])


def build_case_config(params: dict[str, Any]) -> dict[str, Any]:
    cfg = base_config()
    apply_data_combination(cfg, params)
    return cfg


def task_defaults(case: dict[str, Any]) -> dict[str, Any]:
    return {
        **deepcopy(case),
        "time_features": deepcopy(
            case.get("time_features", DataCombinationScenarios.DEFAULT_TIME_FEATURES)
        ),
        "year_slice": deepcopy(case.get("year_slice", [1980, 1981])),
        "num_data_workers": case.get("num_data_workers", 0),
    }


def add_task(
    tasks: list[dict[str, Any]],
    *,
    suite: str,
    params: dict[str, Any],
    runner: Callable[..., Any],
) -> None:
    tasks.append(
        {
            "suite": suite,
            "case_name": params["name"],
            "params": params,
            "cfg_builder": partial(build_case_config, params),
            "runner": runner,
        }
    )


def run_task(task: dict[str, Any]) -> tuple[bool, str]:
    paths = case_paths(task["case_name"])
    case_dir = paths["case_dir"]

    if case_dir.exists():
        shutil.rmtree(case_dir)

    case_dir.mkdir(parents=True, exist_ok=True)

    passed = False
    error = ""

    try:
        params = task["params"]
        cfg_builder = task["cfg_builder"]

        if not callable(cfg_builder):
            raise TypeError(f"Invalid cfg_builder for {task['case_name']}")

        cfg = cfg_builder()
        if not isinstance(cfg, dict):
            raise TypeError(f"cfg_builder returned {type(cfg).__name__}, expected dict")

        write_yaml(cfg, paths["config_path"])
        configure_runtime_context(paths["config_path"])

        print(f"\nStarting {task['suite']}:{task['case_name']}", flush=True)
        print(f"Case directory: {case_dir}", flush=True)
        print(f"Log: {paths['log_path']}", flush=True)

        run_silenced(
            partial(task["runner"], paths["config_path"], params),
            paths["log_path"],
        )

        passed = True

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        append_exception_to_log(paths["log_path"])
        cleanup_logging_handlers()

    finally:
        reset_runtime_context()

        try:
            import gc

            import xarray as xr

            if hasattr(xr.backends, "file_manager"):
                fm = xr.backends.file_manager
                if hasattr(fm, "FILE_MANAGER"):
                    if hasattr(fm.FILE_MANAGER, "_cache"):
                        fm.FILE_MANAGER._cache.clear()
                    if hasattr(fm.FILE_MANAGER, "_lock"):
                        fm.FILE_MANAGER._lock = None

            if hasattr(xr.backends, "netCDF4_"):
                xr.backends.netCDF4_ = None

        except Exception:
            pass

        gc.collect()
        gc.collect()
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log_result(task=task, paths=paths, passed=passed, error=error)
    return passed, error


def validate_tasks(tasks: list[dict[str, Any]]) -> None:
    names = set()
    required_fields = {"suite", "case_name", "params", "cfg_builder", "runner"}

    for index, task in enumerate(tasks):
        missing = required_fields - set(task)
        if missing:
            raise ValueError(f"Task {index} missing: {sorted(missing)}")

        case_name = str(task["case_name"])
        if case_name in names:
            raise ValueError(f"Duplicate case name: {case_name}")
        names.add(case_name)

        if not callable(task["cfg_builder"]):
            raise TypeError(f"Invalid cfg_builder: {case_name}")
        if not callable(task["runner"]):
            raise TypeError(f"Invalid runner: {case_name}")


def classify_result(
    *,
    task: dict[str, Any],
    passed: bool,
    error: str,
    unexpected_failures: list[str],
    expected_failures: list[str],
    unexpected_passes: list[str],
) -> None:
    params = task["params"]
    expected = params.get("expected_result", "PASS")

    if expected == "FAIL":
        if passed:
            unexpected_passes.append(task["case_name"])
            return

        pattern = params.get("expected_error_pattern")
        if pattern and not re.search(pattern, error, re.IGNORECASE | re.DOTALL):
            unexpected_failures.append(f"{task['case_name']}: wrong error: {error}")
        else:
            expected_failures.append(task["case_name"])
        return

    if not passed:
        unexpected_failures.append(f"{task['case_name']}: {error}")


def write_summary(
    tasks: list[dict[str, Any]],
    unexpected_failures: list[str],
    expected_failures: list[str],
    unexpected_passes: list[str],
) -> None:
    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        file.write(f"total_cases: {len(tasks)}\n")
        file.write(f"unexpected_failures: {len(unexpected_failures)}\n")
        file.write(f"expected_failures: {len(expected_failures)}\n")
        file.write(f"unexpected_passes: {len(unexpected_passes)}\n")
        file.write(f"cases_dir: {CASES_DIR}\n")
        file.write(f"results_csv: {RESULTS_CSV}\n")

        for title, values in [
            ("unexpected_failures", unexpected_failures),
            ("expected_failures", expected_failures),
            ("unexpected_passes", unexpected_passes),
        ]:
            if values:
                file.write(f"\n{title}:\n")
                for value in values:
                    file.write(f"- {value}\n")


def build_test_suite() -> list[dict[str, Any]]:
    tasks = []

    if RUN_DATA_CONFIG_COMBINATIONS:
        for case in DataCombinationScenarios.all():
            params = task_defaults(case)
            add_task(
                tasks,
                suite="data_config_combination",
                params=params,
                runner=run_dataset_case,
            )

    if RUN_EXTERNAL_CONDITION_CASES:
        for case in ExternalConditionScenarios.all():
            condition_path = Path(case.get("condition_path", ""))
            params = task_defaults(case)

            if not condition_path.exists():
                params["expected_result"] = "FAIL"
                params["expected_failure_reason"] = (
                    f"Path doesn't exist: {condition_path}"
                )
                params["expected_error_pattern"] = (
                    r"(does not exist|not found|no such file)"
                )

            add_task(
                tasks,
                suite="external_condition",
                params=params,
                runner=run_dataset_case,
            )

    if RUN_DATASET_GETITEM_CASES:
        for case in DataCombinationScenarios.all():
            params = task_defaults(case)
            params["name"] = f"getitem_{case['name']}"
            add_task(
                tasks, suite="dataset_getitem", params=params, runner=run_dataset_case
            )

    if RUN_TRAIN_DATALOADER_CASES:
        base = deepcopy(DataCombinationScenarios.all()[-2])
        for case in DataloaderScenarios.all():
            if case["check"] == "multiworker" and not RUN_MULTIWORKER_CASES:
                continue
            params = task_defaults({**base, **deepcopy(case)})
            add_task(
                tasks,
                suite="train_dataloader",
                params=params,
                runner=run_dataloader_case,
            )

    if RUN_FUTURE_MASK_CASES:
        params = task_defaults(
            {
                **deepcopy(DataCombinationScenarios.all()[-2]),
                "name": "future_leakage_valid_time_cutoff",
                "description": "Validate no samples look beyond December 2020",
                "year_slice": [2019, 2020],
                "cutoff": "2020-12",
                "num_data_workers": 0,
                "shuffle": False,
                "time_features": deepcopy(
                    DataCombinationScenarios.DEFAULT_TIME_FEATURES
                ),
                "lead_months": deepcopy(DataCombinationScenarios.FULL_LEAD_MONTHS),
                "expected_result": "PASS",
            }
        )
        add_task(tasks, suite="future_mask", params=params, runner=run_future_mask_case)

    if RUN_OBSERVATION_CUTOFF_CASES:
        base = deepcopy(DataCombinationScenarios.all()[-2])
        for case in TemporalMaskScenarios.all():
            params = task_defaults(
                {
                    **base,
                    **deepcopy(case),
                    "year_slice": [1980, 1981],
                    "lead_months": deepcopy(DataCombinationScenarios.FULL_LEAD_MONTHS),
                }
            )
            add_task(
                tasks,
                suite="observation_cutoff",
                params=params,
                runner=run_observation_cutoff_case,
            )

    if RUN_DISTRIBUTED_PREPROCESSOR_CASES:
        params = task_defaults(
            {
                **deepcopy(DataCombinationScenarios.all()[-2]),
                "name": "rank_zero_fits_other_ranks_load_preprocessors",
                "description": "Rank-0 fits/saves, other ranks load",
                "year_slice": [1980, 1981],
                "num_data_workers": 0,
                "shuffle": False,
                "time_features": deepcopy(
                    DataCombinationScenarios.DEFAULT_TIME_FEATURES
                ),
                "expected_result": "PASS",
            }
        )
        add_task(
            tasks,
            suite="distributed_preprocessing",
            params=params,
            runner=run_preprocessor_case,
        )

    if RUN_INFERENCE_DATALOADER_CASES:
        if BASE_INFERENCE_CONFIG.is_file():
            params = task_defaults(
                {
                    **deepcopy(DataCombinationScenarios.all()[-2]),
                    "name": "inference_reuses_training_data_contract",
                    "description": "Inference copies training contract and time_features",
                    "year_slice": [1980, 1981],
                    "num_data_workers": 0,
                    "shuffle": False,
                    "time_features": deepcopy(
                        DataCombinationScenarios.DEFAULT_TIME_FEATURES
                    ),
                    "inference_config_path": BASE_INFERENCE_CONFIG,
                    "expected_result": "PASS",
                }
            )
            add_task(
                tasks,
                suite="inference_dataloader",
                params=params,
                runner=run_inference_case,
            )

    return tasks


def main() -> None:
    prepare_output_directories()
    cleanup_logging_handlers()
    init_csv()

    tasks = build_test_suite()
    validate_tasks(tasks)

    if MAX_CASES is not None:
        tasks = tasks[:MAX_CASES]

    unexpected_failures = []
    expected_failures = []
    unexpected_passes = []

    try:
        with tqdm(
            total=len(tasks),
            desc="data_modules",
            unit="case",
            dynamic_ncols=True,
            leave=True,
        ) as progress:
            for task in tasks:
                progress.set_postfix_str(f"{task['suite']}:{task['case_name'][:60]}")

                passed, error = run_task(task)

                classify_result(
                    task=task,
                    passed=passed,
                    error=error,
                    unexpected_failures=unexpected_failures,
                    expected_failures=expected_failures,
                    unexpected_passes=unexpected_passes,
                )

                progress.update(1)

    finally:
        reset_runtime_context()
        cleanup_logging_handlers()

    write_summary(tasks, unexpected_failures, expected_failures, unexpected_passes)

    if unexpected_failures or unexpected_passes:
        problems = []
        if unexpected_failures:
            problems.append(f"{len(unexpected_failures)} unexpected failure(s)")
        if unexpected_passes:
            problems.append(f"{len(unexpected_passes)} unexpected pass(es)")

        raise RuntimeError(
            f"Integration suite failed: {' and '.join(problems)}. "
            f"See {RESULTS_CSV} and {CASES_DIR}."
        )

    print("✓ Integration test suite completed successfully")
    print(f"  Cases: {CASES_DIR}")
    print(f"  Results: {RESULTS_CSV}")
    print(f"  Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
