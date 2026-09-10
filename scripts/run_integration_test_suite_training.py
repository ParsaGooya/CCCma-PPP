from __future__ import annotations

import csv
import json
import logging
import re
import shutil
import traceback
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import datetime
from functools import partial
from itertools import islice, product
from pathlib import Path
from typing import Any, Callable, Iterator

import dacite
import yaml
from tqdm import tqdm

# Import architecture modules (registers them in the factory)
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

# Import loss modules (registers them in the factory)
from cccma_ppp.loss import kld as _loss_kld  # noqa: F401
from cccma_ppp.loss import loss as _loss_pipeline  # noqa: F401
from cccma_ppp.loss import utils_loss as _loss_utils  # noqa: F401

# Import the training entry point
from cccma_ppp.train.train import main as train_main
from cccma_ppp.train.train_configs import TrainConfig, prepare_config

warnings.filterwarnings("ignore")
logging.raiseExceptions = False


# Find the project root (two levels up from this script)
ROOT = Path(__file__).resolve().parents[1]

# Base training configuration template
BASE_TRAIN_CONFIG = ROOT / "scripts" / "integration_suite_train_config.yaml"

# Output directories for test results
OUTPUT_DIR = ROOT / "../output" / "train_integration_test_results"
CASES_DIR = OUTPUT_DIR / "cases"
GENERATED_CONFIG_DIR = OUTPUT_DIR / "_generated_configs"
LOG_DIR = OUTPUT_DIR / "_logs"
RESULTS_CSV = OUTPUT_DIR / "integration_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary.txt"


# Extract time dimension names from config (e.g., "init_time", "lead_time")
INIT_TIME_DIM, LEAD_TIME_DIM = required_sample_dimensions


# Control which test suites to run
RUN_BASELINE_TESTS = False  # Special tests like resume and dataset pipeline
RUN_TRAIN_LOADER_CONFIG_CASES = True  # Test different data loader configurations
RUN_CVAE_MLP_GRID = True  # Test cVAE with MLP architecture
RUN_CVAE_UNET_GRID = True  # Test cVAE with UNet architecture
RUN_DETERMINISTIC_MLP_GRID = True  # Test deterministic regression with MLP
RUN_DETERMINISTIC_UNET_GRID = True  # Test deterministic regression with UNet


# Limit number of test cases per suite (for faster testing, set to None for all)
MAX_TRAIN_LOADER_CONFIG_CASES = None
MAX_CVAE_MLP_CASES = None
MAX_CVAE_UNET_CASES = None
MAX_DETERMINISTIC_MLP_CASES = None
MAX_DETERMINISTIC_UNET_CASES = None


# Batch limits for faster testing (data loader batches to run per test case)
MAX_TRAIN_BATCHES_PER_CASE = 1  # Only run 1 training batch per case
MAX_VALIDATION_BATCHES_PER_CASE = 1  # Only run 1 validation batch per case

# Batch limits for baseline tests (resume, dataset pipeline)
MAX_BASELINE_TRAIN_BATCHES = 1
MAX_BASELINE_VALIDATION_BATCHES = 1


# Small neural network architectures for fast testing
TEST_ENCODER_HIDDEN_DIMS = [16]
TEST_DECODER_HIDDEN_DIMS = [16]
TEST_CONDITION_EMBEDDING_DIMS = [16]

# Small UNet architecture for fast testing
TEST_UNET_CHANNELS = [4, 8]
TEST_UNET_CONDITION_CHANNELS = [4, 8]
TEST_UNET_OUTPUT_BLOCK_HIDDEN_CHANNELS = 8

TEST_UNET_BLOCK_CONFIG = {
    "name": "convnext",
    "num_blocks": 2,
}


# Time features that are valid/supported
VALID_TIME_FEATURES = {
    INIT_TIME_DIM,
    LEAD_TIME_DIM,
    "month_sin",
    "month_cos",
    "day_sin",
    "day_cos",
}


# Standard lead time (forecast lead distance) for most tests
FULL_LEAD_MONTHS = {
    "start": 1,
    "end": 1,
}


# Test cases for data loader and dataset configuration
# These test different combinations of:
# - Conditioning methods (None, ensemble_mean, same_member, cross_ensemble, static)
# - Data presence (observation present/missing, condition present/missing)
# - Time features (none, init_time, production features, all)
TRAIN_LOADER_CONFIG_CASES = [
    {
        "name": "tlc_condition_none_obs_present",
        "condition_method": None,
        "model_ensemble_mean": True,
        "remove_observation": False,
        "add_condition": False,
        "time_features": [INIT_TIME_DIM],
        "lead_months": deepcopy(FULL_LEAD_MONTHS),
        "expected_result": "PASS",
        "expected_failure_reason": "",
    },
    {
        "name": "tlc_ensemble_mean_obs_present",
        "condition_method": "ensemble_mean",
        "model_ensemble_mean": True,
        "remove_observation": False,
        "add_condition": False,
        "time_features": [INIT_TIME_DIM],
        "lead_months": deepcopy(FULL_LEAD_MONTHS),
        "expected_result": "PASS",
        "expected_failure_reason": "",
    },
    {
        "name": "tlc_same_member_model_not_mean",
        "condition_method": "same_member",
        "model_ensemble_mean": False,
        "remove_observation": False,
        "add_condition": False,
        "time_features": [INIT_TIME_DIM],
        "lead_months": deepcopy(FULL_LEAD_MONTHS),
        "expected_result": "PASS",
        "expected_failure_reason": "",
    },
    {
        "name": "tlc_cross_ensemble_model_not_mean",
        "condition_method": "cross_ensemble",
        "model_ensemble_mean": False,
        "remove_observation": False,
        "add_condition": False,
        "time_features": [INIT_TIME_DIM],
        "lead_months": deepcopy(FULL_LEAD_MONTHS),
        "expected_result": "PASS",
        "expected_failure_reason": "",
    },
    {
        "name": "tlc_same_member_invalid_model_mean",
        "condition_method": "same_member",
        "model_ensemble_mean": True,  # Invalid combo: ensemble_mean contradicts same_member
        "remove_observation": False,
        "add_condition": False,
        "time_features": [INIT_TIME_DIM],
        "lead_months": deepcopy(FULL_LEAD_MONTHS),
        "expected_result": "FAIL",
        "expected_failure_reason": (
            "same_member conditioning should fail when model.ensemble_mean=True."
        ),
        "expected_error_pattern": (
            r"same_member.*ensemble_mean|"
            r"ensemble_mean.*same_member"
        ),
    },
    {
        "name": "tlc_observation_missing_condition_none",
        "condition_method": None,
        "model_ensemble_mean": True,
        "remove_observation": True,  # Missing observation
        "add_condition": False,
        "time_features": [INIT_TIME_DIM],
        "lead_months": deepcopy(FULL_LEAD_MONTHS),
        "expected_result": "FAIL",
        "expected_failure_reason": (
            "A training target is unavailable when observation is "
            "missing and condition_method is None."
        ),
        "expected_error_pattern": (r"observation|target|condition_method"),
    },
    {
        "name": "tlc_observation_missing_ensemble_mean",
        "condition_method": "ensemble_mean",
        "model_ensemble_mean": True,
        "remove_observation": True,  # Missing observation is OK with condition
        "add_condition": False,
        "time_features": [INIT_TIME_DIM],
        "lead_months": deepcopy(FULL_LEAD_MONTHS),
        "expected_result": "PASS",
        "expected_failure_reason": "",
    },
    {
        "name": "tlc_time_features_none",
        "condition_method": "ensemble_mean",
        "model_ensemble_mean": True,
        "remove_observation": False,
        "add_condition": False,
        "time_features": None,  # No time features
        "lead_months": deepcopy(FULL_LEAD_MONTHS),
        "expected_result": "PASS",
        "expected_failure_reason": "",
    },
    {
        "name": "tlc_time_features_init_time",
        "condition_method": "ensemble_mean",
        "model_ensemble_mean": True,
        "remove_observation": False,
        "add_condition": False,
        "time_features": [INIT_TIME_DIM],  # Only initialization time
        "lead_months": deepcopy(FULL_LEAD_MONTHS),
        "expected_result": "PASS",
        "expected_failure_reason": "",
    },
    {
        "name": "tlc_time_features_production",
        "condition_method": "ensemble_mean",
        "model_ensemble_mean": True,
        "remove_observation": False,
        "add_condition": False,
        "time_features": [  # Production-ready time features
            LEAD_TIME_DIM,
            "month_sin",
            "month_cos",
        ],
        "lead_months": deepcopy(FULL_LEAD_MONTHS),
        "expected_result": "PASS",
        "expected_failure_reason": "",
    },
    {
        "name": "tlc_time_features_all",
        "condition_method": "ensemble_mean",
        "model_ensemble_mean": True,
        "remove_observation": False,
        "add_condition": False,
        "time_features": [  # All possible time features
            INIT_TIME_DIM,
            LEAD_TIME_DIM,
            "month_sin",
            "month_cos",
            "day_sin",
            "day_cos",
        ],
        "lead_months": deepcopy(FULL_LEAD_MONTHS),
        "expected_result": "PASS",
        "expected_failure_reason": "",
    },
    {
        "name": "tlc_lead_months_single",
        "condition_method": "ensemble_mean",
        "model_ensemble_mean": True,
        "remove_observation": False,
        "add_condition": False,
        "time_features": [INIT_TIME_DIM],
        "lead_months": {
            "start": 1,
            "end": 1,
        },
        "expected_result": "PASS",
        "expected_failure_reason": "",
    },
    {
        "name": "tlc_lead_months_missing",
        "condition_method": "ensemble_mean",
        "model_ensemble_mean": True,
        "remove_observation": False,
        "add_condition": False,
        "time_features": [INIT_TIME_DIM],
        "lead_months": None,  # No lead time specification
        "expected_result": "PASS",
        "expected_failure_reason": "",
    },
    {
        "name": "tlc_explicit_condition_ensemble_mean",
        "condition_method": "ensemble_mean",
        "model_ensemble_mean": True,
        "condition_ensemble_mean": True,  # Explicit condition dataset
        "remove_observation": False,
        "add_condition": True,  # Include condition data
        "time_features": [INIT_TIME_DIM],
        "lead_months": deepcopy(FULL_LEAD_MONTHS),
        "expected_result": "PASS",
        "expected_failure_reason": "",
    },
    {
        "name": "tlc_explicit_condition_cross_ensemble",
        "condition_method": "cross_ensemble",
        "model_ensemble_mean": False,
        "condition_ensemble_mean": False,
        "remove_observation": False,
        "add_condition": True,
        "time_features": [INIT_TIME_DIM],
        "lead_months": deepcopy(FULL_LEAD_MONTHS),
        "expected_result": "PASS",
        "expected_failure_reason": "",
    },
    {
        "name": "tlc_static_without_condition",
        "condition_method": "static",  # Requires explicit condition dataset
        "model_ensemble_mean": True,
        "remove_observation": False,
        "add_condition": False,  # But we didn't provide one
        "time_features": [INIT_TIME_DIM],
        "lead_months": deepcopy(FULL_LEAD_MONTHS),
        "expected_result": "FAIL",
        "expected_failure_reason": (
            "Static conditioning requires an explicit condition dataset."
        ),
        "expected_error_pattern": (r"static.*condition|condition.*static"),
    },
    {
        "name": "tlc_static_with_model_copied_condition",
        "condition_method": "static",
        "model_ensemble_mean": True,
        "condition_ensemble_mean": True,
        "remove_observation": False,
        "add_condition": True,  # Condition is same as model data
        "time_features": [INIT_TIME_DIM],
        "lead_months": deepcopy(FULL_LEAD_MONTHS),
        "expected_result": "FAIL",
        "expected_failure_reason": (
            "Static conditioning cannot use the same data configuration "
            "as the model input."
        ),
        "expected_error_pattern": (r"static|same.*model|condition"),
    },
]


# Hyperparameter grids for cVAE model architecture
# Each combination will be tested with both MLP and UNet
CVAE_GRID = {
    "posterior_variance_limits": [
        None,  # No explicit variance limits
        [0.5, 2.0],  # Constrained variance range
    ],
    "use_prior_flow": [
        False,  # Standard prior
    ],
    "combined_CGCN_weight": [
        0.0,  # No combined CGCN loss
        0.1,  # Some combined CGCN loss
    ],
    "latent_size": [
        4,  # Small latent space
        10,  # Medium latent space
    ],
    "condition_dependant_latent": [
        True,  # Condition affects latent distribution
        False,  # Condition-independent latent
    ],
    "condemb_to_decoder": [
        True,  # Pass condition embedding to decoder
        False,  # Don't pass condition embedding to decoder
    ],
    "init_method": [
        "default",
        "xavier",
        "trunc_normal",
    ],
}


# MLP-specific hyperparameters (additional to CVAE_GRID)
MLP_GRID = {
    "batch_normalization": [
        False,
        True,
    ],
    "dropout_rate": [
        None,  # No dropout
        0.5,  # 50% dropout
    ],
}


# Deterministic model hyperparameters
DETERMINISTIC_GRID = {
    "init_method": [
        "default",
        "xavier",
        "trunc_normal",
    ],
}


# MLP-specific deterministic hyperparameters
DETERMINISTIC_MLP_GRID = {
    "append_mode": [
        1,  # Different concatenation modes
        2,  # for deterministic regression
        3,
    ],
    "batch_normalization": [
        False,
        True,
    ],
    "dropout_rate": [
        None,
        0.5,
    ],
}


# Column names for the results CSV file
CSV_COLUMNS = [
    "timestamp",
    "suite",
    "case_name",
    "module_type",
    "architecture",
    "model_type",
    "condition_method",
    "model_ensemble_mean",
    "condition_ensemble_mean",
    "remove_observation",
    "add_condition",
    "time_features",
    "lead_months",
    "posterior_variance_limits",
    "use_prior_flow",
    "combined_CGCN_weight",
    "latent_size",
    "condition_dependant_latent",
    "condemb_to_decoder",
    "batch_normalization",
    "dropout_rate",
    "init_method",
    "append_mode",
    "expected_result",
    "expected_failure_reason",
    "expected_error_pattern",
    "config_path",
    "experiment_dir",
    "log_path",
    "result",
    "error",
]


class LimitedLoader:
    """
    Wrapper around a data loader that limits the number of batches yielded.
    Used to speed up tests by limiting data loading to a small number of batches.
    """

    def __init__(
        self,
        loader: Any,
        max_batches: int | None,
    ) -> None:
        self._loader = loader
        self._max_batches = max_batches

    def __iter__(self) -> Iterator:
        """Yield batches from the underlying loader, limited to max_batches."""
        iterator = iter(self._loader)

        if self._max_batches is None:
            # No limit, yield everything
            yield from iterator
            return

        # Limit to max_batches
        yield from islice(
            iterator,
            self._max_batches,
        )

    def __len__(self) -> int:
        """Return the number of batches this loader will yield."""
        loader_length = len(self._loader)

        if self._max_batches is None:
            return loader_length

        return min(
            loader_length,
            self._max_batches,
        )

    def __getattr__(
        self,
        name: str,
    ) -> Any:
        """Forward any other attribute access to the underlying loader."""
        return getattr(
            self._loader,
            name,
        )


def prepare_output_directories() -> None:
    """Create all necessary output directories for test results."""
    for path in (
        OUTPUT_DIR,
        CASES_DIR,
        GENERATED_CONFIG_DIR,
        LOG_DIR,
    ):
        path.mkdir(
            parents=True,
            exist_ok=True,
        )


def cleanup_logging_handlers() -> None:
    """
    Close and remove all logging handlers.
    Prevents resource leaks and file handle conflicts between test cases.
    """
    logger_names = [
        "",
        "training",
    ]

    # Collect all loggers that start with the package name
    logger_names.extend(
        logger_name
        for logger_name in logging.Logger.manager.loggerDict
        if logger_name.startswith("cccma_ppp")
    )

    # Remove all handlers from each logger
    for logger_name in dict.fromkeys(logger_names):
        logger = logging.getLogger(logger_name)

        for handler in list(logger.handlers):
            logger.removeHandler(handler)

            # Flush and close the handler
            try:
                handler.flush()
            except Exception:
                pass

            try:
                handler.close()
            except Exception:
                pass

        logger.propagate = False


def init_csv() -> None:
    """Create the results CSV file with headers."""
    prepare_output_directories()

    with RESULTS_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_COLUMNS,
        )
        writer.writeheader()


def clean_name(
    value: Any,
) -> str:
    """
    Convert a value to a filesystem-safe name.
    Removes special characters and collapses multiple underscores.
    """
    value = str(value).replace(
        "None",
        "none",
    )
    # Replace special characters with underscores
    value = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        value,
    )
    # Collapse multiple underscores
    value = re.sub(
        r"_+",
        "_",
        value,
    )

    return value.strip("_")


def serialize_json(
    value: Any,
) -> str:
    """Convert a value to JSON for storing in CSV."""
    return json.dumps(
        value,
        sort_keys=True,
        default=str,
    )


def case_paths(
    case_name: str,
) -> dict[str, Path]:
    """
    Generate filesystem paths for a test case.
    Returns dict with config_path, experiment_dir, and log_path.
    """
    safe_name = clean_name(case_name)

    return {
        "config_path": (GENERATED_CONFIG_DIR / f"{safe_name}.yaml"),
        "experiment_dir": (CASES_DIR / safe_name),
        "log_path": (LOG_DIR / f"{safe_name}.log"),
    }


def log_result(
    *,
    task: dict[str, Any],
    paths: dict[str, Path],
    passed: bool,
    error: str,
) -> None:
    """Log a test result to the CSV file."""
    params = task["params"]

    row = {
        "timestamp": datetime.now().isoformat(),
        "suite": task["suite"],
        "case_name": task["case_name"],
        "module_type": task["module_type"],
        "architecture": task["architecture"],
        "model_type": params.get("model_type"),
        "condition_method": params.get("condition_method"),
        "model_ensemble_mean": params.get("model_ensemble_mean"),
        "condition_ensemble_mean": params.get("condition_ensemble_mean"),
        "remove_observation": params.get("remove_observation"),
        "add_condition": params.get("add_condition"),
        "time_features": serialize_json(params.get("time_features")),
        "lead_months": serialize_json(params.get("lead_months")),
        "posterior_variance_limits": serialize_json(
            params.get("posterior_variance_limits")
        ),
        "use_prior_flow": params.get("use_prior_flow"),
        "combined_CGCN_weight": params.get("combined_CGCN_weight"),
        "latent_size": params.get("latent_size"),
        "condition_dependant_latent": params.get("condition_dependant_latent"),
        "condemb_to_decoder": params.get("condemb_to_decoder"),
        "batch_normalization": params.get("batch_normalization"),
        "dropout_rate": params.get("dropout_rate"),
        "init_method": params.get("init_method"),
        "append_mode": params.get("append_mode"),
        "expected_result": params.get(
            "expected_result",
            "PASS",
        ),
        "expected_failure_reason": params.get(
            "expected_failure_reason",
            "",
        ),
        "expected_error_pattern": params.get(
            "expected_error_pattern",
            "",
        ),
        "config_path": str(paths["config_path"]),
        "experiment_dir": str(paths["experiment_dir"]),
        "log_path": str(paths["log_path"]),
        "result": ("PASS" if passed else "FAIL"),
        "error": error,
    }

    with RESULTS_CSV.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_COLUMNS,
        )
        writer.writerow(row)


def normalize_numeric_fields(
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    Normalize numeric fields in the configuration.
    Converts string numbers to floats where needed (e.g., "0.001" -> 0.001).
    """
    optimization_cfg = cfg.get("optimization")

    if not isinstance(
        optimization_cfg,
        dict,
    ):
        return cfg

    # Learning rate and weight decay might be strings
    numeric_fields = [
        (
            optimization_cfg,
            "lr",
        ),
        (
            optimization_cfg,
            "weight_decay",
        ),
    ]

    scheduler_cfg = optimization_cfg.get("lr_scheduler_config")

    if isinstance(
        scheduler_cfg,
        dict,
    ):
        numeric_fields.append(
            (
                scheduler_cfg,
                "min_lr",
            )
        )

    for owner, field_name in numeric_fields:
        value = owner.get(field_name)

        if value is None:
            continue

        if isinstance(
            value,
            bool,
        ):
            raise TypeError(f"{field_name} must be numeric, not boolean.")

        if isinstance(
            value,
            str,
        ):
            try:
                value = float(value)
            except ValueError as exc:
                raise ValueError(
                    f"{field_name} must be numeric, got {value!r}."
                ) from exc

        if not isinstance(
            value,
            (
                int,
                float,
            ),
        ):
            raise TypeError(
                f"{field_name} must be numeric, got {type(value).__name__}."
            )

        owner[field_name] = float(value)

    return cfg


def normalize_nullable_fields(
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    Remove None-valued fields from configuration.
    None values in YAML should be removed rather than kept.
    """
    normalize_numeric_fields(cfg)

    train_loader_cfg = cfg.setdefault(
        "train_loader",
        {},
    )
    dataset_cfg = train_loader_cfg.setdefault(
        "dataset_config",
        {},
    )

    # Validate field locations
    if "time_features" in dataset_cfg:
        raise ValueError(
            "time_features is no longer a DatasetConfig field. "
            "Configure it under train_loader.time_features."
        )

    # Remove None values
    if train_loader_cfg.get("time_features") is None:
        train_loader_cfg.pop(
            "time_features",
            None,
        )

    if dataset_cfg.get("condition_method") is None:
        dataset_cfg.pop(
            "condition_method",
            None,
        )

    if dataset_cfg.get("lead_times") is None:
        dataset_cfg.pop(
            "lead_times",
            None,
        )

    module_cfg = cfg.get(
        "module",
        {},
    ).get(
        "config",
        {},
    )

    if module_cfg.get("posterior_variance_limits") is None:
        module_cfg.pop(
            "posterior_variance_limits",
            None,
        )

    if module_cfg.get("prior_flow_config") is None:
        module_cfg.pop(
            "prior_flow_config",
            None,
        )

    model_cfg = module_cfg.get(
        "ModelConfig",
        {},
    ).get(
        "config",
        {},
    )

    if model_cfg.get("dropout_rate") is None:
        model_cfg.pop(
            "dropout_rate",
            None,
        )

    if model_cfg.get("deterministic_guess_config") is None:
        model_cfg.pop(
            "deterministic_guess_config",
            None,
        )

    return cfg


def validate_generated_config(
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    Validate that all required fields are present and types are correct.
    Checks for deprecated field names and invalid configurations.
    """
    train_loader_cfg = cfg.get("train_loader")

    if not isinstance(
        train_loader_cfg,
        dict,
    ):
        raise ValueError("train_loader must be configured as a mapping.")

    dataset_cfg = train_loader_cfg.get("dataset_config")

    if not isinstance(
        dataset_cfg,
        dict,
    ):
        raise ValueError("train_loader.dataset_config must be configured as a mapping.")

    # Check for deprecated field location
    if "time_features" in dataset_cfg:
        raise ValueError(
            "time_features must be configured under "
            "train_loader, not train_loader.dataset_config."
        )

    # Validate time features
    time_features = train_loader_cfg.get("time_features")

    if time_features is not None:
        if not isinstance(
            time_features,
            list,
        ):
            raise TypeError("train_loader.time_features must be a list or null.")

        unsupported = set(time_features) - VALID_TIME_FEATURES

        if unsupported:
            raise ValueError(
                "Unsupported train_loader.time_features: "
                f"{sorted(unsupported)}. Supported values are "
                f"{sorted(VALID_TIME_FEATURES)}."
            )

    # Validate module configuration
    module_cfg = cfg.get("module")

    if not isinstance(
        module_cfg,
        dict,
    ):
        raise ValueError("module must be configured as a mapping.")

    module_type = str(
        module_cfg.get(
            "type",
            "",
        )
    ).lower()

    raw_module_config = module_cfg.get("config")

    if not isinstance(
        raw_module_config,
        dict,
    ):
        raise ValueError("module.config must be configured as a mapping.")

    # Check for deprecated fields
    if "min_posterior_variance" in raw_module_config:
        raise ValueError(
            "min_posterior_variance is obsolete. Use posterior_variance_limits."
        )

    # Validate model selector
    model_selector = raw_module_config.get("ModelConfig")

    if not isinstance(
        model_selector,
        dict,
    ):
        raise ValueError("module.config.ModelConfig must be configured as a mapping.")

    if not model_selector.get("type"):
        raise ValueError("module.config.ModelConfig.type must be configured.")

    model_config = model_selector.get("config")

    if not isinstance(
        model_config,
        dict,
    ):
        raise ValueError("module.config.ModelConfig.config must be a mapping.")

    model_type = str(model_selector["type"]).lower()

    # UNet-specific validation
    if model_type in {
        "unet",
        "unetsic",
    }:
        block_config = model_config.get("block_config")

        if not isinstance(
            block_config,
            dict,
        ):
            raise ValueError("UNet ModelConfig requires block_config to be a mapping.")

        if not block_config.get("name"):
            raise ValueError("UNet ModelConfig.block_config requires a nonempty name.")

        # Check for deprecated UNet fields
        if "process_skip" in model_config:
            raise ValueError("process_skip is obsolete. Use process_skip_connections.")

        # Deterministic and cVAE use different skip alignment fields
        if (
            module_type == "deterministic"
            and "upsampling_alignment_method" in model_config
        ):
            raise ValueError(
                "Deterministic UNet uses "
                "skip_alignment_method, not "
                "upsampling_alignment_method."
            )

        if module_type == "cvae" and "skip_alignment_method" in model_config:
            raise ValueError(
                "cVAE UNet uses upsampling_alignment_method, not skip_alignment_method."
            )

    return cfg


def write_yaml(
    cfg: dict[str, Any],
    output_path: Path,
) -> Path:
    """
    Normalize, validate, and write a configuration to a YAML file.
    Ensures the configuration is valid before writing.
    """
    cfg = deepcopy(cfg)

    normalize_numeric_fields(cfg)
    normalize_nullable_fields(cfg)
    validate_generated_config(cfg)

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            cfg,
            file,
            sort_keys=False,
        )

    return output_path


def load_train_config(
    config_path: Path,
) -> TrainConfig:
    """
    Load a training configuration from a YAML file and convert to TrainConfig dataclass.
    """
    cfg_data = prepare_config(config_path)

    if cfg_data is None:
        raise ValueError(f"Generated training configuration is empty: {config_path}")

    normalize_numeric_fields(cfg_data)
    normalize_nullable_fields(cfg_data)
    validate_generated_config(cfg_data)

    # Convert dict to TrainConfig dataclass with strict validation
    return dacite.from_dict(
        data_class=TrainConfig,
        data=cfg_data,
        config=dacite.Config(
            strict=True,
        ),
    )


def base_config() -> dict[str, Any]:
    """
    Load and prepare the base training configuration template.
    Sets common parameters and validates the result.
    """
    cfg = prepare_config(BASE_TRAIN_CONFIG)

    if cfg is None:
        raise ValueError(
            f"The base training configuration is empty: {BASE_TRAIN_CONFIG}"
        )

    normalize_numeric_fields(cfg)

    # Override key parameters for faster testing
    cfg["max_epochs"] = 1
    cfg["log_every_n_epochs"] = 1
    cfg["save_checkpoint"] = True

    train_loader_cfg = cfg.get("train_loader")

    if not isinstance(
        train_loader_cfg,
        dict,
    ):
        raise ValueError("The base configuration must contain a train_loader mapping.")

    dataset_cfg = train_loader_cfg.get("dataset_config")

    if not isinstance(
        dataset_cfg,
        dict,
    ):
        raise ValueError(
            "The base configuration must contain train_loader.dataset_config."
        )

    if "time_features" in dataset_cfg:
        raise ValueError(
            "Move dataset_config.time_features to train_loader.time_features."
        )

    # Configure minimal data loading for fast tests
    train_loader_cfg["batch_size"] = 1
    train_loader_cfg["num_data_workers"] = 0
    train_loader_cfg["num_validation_years"] = 0
    train_loader_cfg["train_years_slice"] = [
        1980,
        1981,
    ]

    train_loader_cfg["load"] = False  # Don't cache preprocessed data

    # Remove optional fields that slow things down
    train_loader_cfg.pop(
        "prefetch_factor",
        None,
    )

    normalize_nullable_fields(cfg)
    validate_generated_config(cfg)

    return cfg


def run_silenced(
    function: Callable[[], Any],
    log_path: Path,
) -> Any:
    """
    Run a function and capture all stdout/stderr to a log file.
    Useful for capturing training output.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cleanup_logging_handlers()

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as log_file:
        try:
            # Redirect all output to the log file
            with (
                redirect_stdout(log_file),
                redirect_stderr(log_file),
            ):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return function()
        finally:
            cleanup_logging_handlers()


def assert_training_completed(
    *,
    experiment_dir: Path,
    log_path: Path,
) -> None:
    """
    Verify that training completed successfully.
    Checks for checkpoint files and completion message in log.
    """
    experiment_dir = Path(experiment_dir)
    log_path = Path(log_path)

    checkpoint_dir = experiment_dir / "checkpoints"

    if not checkpoint_dir.is_dir():
        raise AssertionError(
            f"Expected checkpoint directory does not exist: {checkpoint_dir}"
        )

    checkpoint_files = sorted(checkpoint_dir.glob("*.pt"))

    if not checkpoint_files:
        raise AssertionError(f"No checkpoint files were written in {checkpoint_dir}")

    if not log_path.is_file():
        raise AssertionError(f"Log file does not exist: {log_path}")

    log_text = log_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    if "Training finished" not in log_text:
        raise AssertionError(
            "Training did not reach completion. "
            "The log does not contain 'Training finished': "
            f"{log_path}"
        )


def ensure_cvae_beta_finder(
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    Add default beta finder configuration for cVAE models.
    Beta finder controls the KL divergence weight during training.
    """
    cfg.setdefault(
        "trainer",
        {},
    ).setdefault(
        "beta_finder",
        {
            "beta": 0.01,
            "num_epoch_to_warmup": 10,
        },
    )

    return cfg


def remove_deterministic_beta_finder(
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Remove beta finder configuration for deterministic models (not needed)."""
    cfg.setdefault(
        "trainer",
        {},
    ).pop(
        "beta_finder",
        None,
    )

    return cfg


def apply_train_loader_config_params(
    cfg: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """
    Apply test case parameters to the base configuration.
    Modifies data loading, conditioning, and time feature settings.
    """
    train_loader_cfg = cfg["train_loader"]
    dataset_cfg = train_loader_cfg["dataset_config"]

    if "time_features" in dataset_cfg:
        raise ValueError("time_features must not be configured inside dataset_config.")

    # Apply model ensemble mean setting
    if "model_ensemble_mean" in params:
        dataset_cfg["model"]["ensemble_mean"] = params["model_ensemble_mean"]

    # Remove observation data if requested
    if params.get(
        "remove_observation",
        False,
    ):
        dataset_cfg.pop(
            "observation",
            None,
        )

    # Add or remove condition data
    if params.get(
        "add_condition",
        False,
    ):
        dataset_cfg["condition"] = deepcopy(dataset_cfg["model"])

        condition_ensemble_mean = params.get("condition_ensemble_mean")

        if condition_ensemble_mean is not None:
            dataset_cfg["condition"]["ensemble_mean"] = condition_ensemble_mean
    else:
        dataset_cfg.pop(
            "condition",
            None,
        )

    # Set conditioning method
    condition_method = params.get("condition_method")

    if condition_method is None:
        dataset_cfg.pop(
            "condition_method",
            None,
        )
    else:
        dataset_cfg["condition_method"] = condition_method

    # Set time features
    if "time_features" in params:
        time_features = params["time_features"]

        if time_features is None:
            train_loader_cfg.pop(
                "time_features",
                None,
            )
        else:
            train_loader_cfg["time_features"] = deepcopy(time_features)

    # Set lead time configuration
    if "lead_months" in params:
        lead_months = params["lead_months"]

        if lead_months is None:
            dataset_cfg.pop(
                "lead_times",
                None,
            )
        else:
            dataset_cfg["lead_times"] = deepcopy(lead_months)

    normalize_nullable_fields(cfg)

    return cfg


def build_mlp_model_config(
    *,
    latent_size: int | None = None,
    condition_dependant_latent: bool | None = None,
    condemb_to_decoder: bool = True,
    batch_normalization: bool = False,
    dropout_rate: float | None = None,
    init_method: str = "default",
    append_mode: int | None = None,
    cvae: bool = False,
) -> dict[str, Any]:
    """
    Build an MLP model configuration.
    Used for both cVAE and deterministic regression models.
    """
    model_cfg = {
        "encoder_hidden_dims": deepcopy(TEST_ENCODER_HIDDEN_DIMS),
        "decoder_hidden_dims": deepcopy(TEST_DECODER_HIDDEN_DIMS),
        "batch_normalization": (batch_normalization),
        "init_method": init_method,
    }

    if dropout_rate is not None:
        model_cfg["dropout_rate"] = dropout_rate

    if cvae:
        # cVAE-specific parameters
        model_cfg["condition_embedding_dims"] = deepcopy(TEST_CONDITION_EMBEDDING_DIMS)
        model_cfg["latent_size"] = latent_size
        model_cfg["condition_embedding_size"] = latent_size
        model_cfg["condition_dependant_latent"] = condition_dependant_latent
        model_cfg["condemb_to_decoder"] = condemb_to_decoder
    else:
        # Deterministic-specific parameters
        model_cfg["append_mode"] = append_mode

    return {
        "type": "MLP",
        "config": model_cfg,
    }


def build_unet_model_config(
    *,
    latent_size: int | None = None,
    condition_dependant_latent: bool | None = None,
    condemb_to_decoder: bool = True,
    init_method: str = "default",
    model_type: str = "UNetSIC",
    deterministic_guess_checkpoint: (str | Path | None) = None,
    cvae: bool = False,
) -> dict[str, Any]:
    """
    Build a UNet model configuration.
    Used for both cVAE and deterministic regression models.
    """
    model_cfg = {
        "channels": deepcopy(TEST_UNET_CHANNELS),
        "block_config": deepcopy(TEST_UNET_BLOCK_CONFIG),
        "upsampling_method": "bilinear",
        "transpose_kernel_sizes": 3,
        "mask_pooling": "any",
        "mask_fraction_threshold": 0.5,
        "output_activation": "identity",
        "output_block_hidden_channels": (TEST_UNET_OUTPUT_BLOCK_HIDDEN_CHANNELS),
        "init_method": init_method,
        "GENERATOR": None,
        "clip_output": False,
    }

    if cvae:
        # cVAE-specific UNet parameters
        if latent_size is None:
            raise ValueError("latent_size is required for cVAE UNetSIC.")

        model_cfg.update(
            {
                "latent_size": latent_size,
                "latent_normalization": "layer",
                "condition_embedding_channels": deepcopy(TEST_UNET_CONDITION_CHANNELS),
                "condition_embedding_size": latent_size,
                "condition_dependant_latent": (condition_dependant_latent),
                "condemb_to_decoder": (condemb_to_decoder),
                "upsampling_alignment_method": "padd",
                "add_skip_latent": False,
            }
        )

        # Optional deterministic guess configuration
        if deterministic_guess_checkpoint is None:
            model_cfg["deterministic_guess_config"] = None
        else:
            checkpoint_path = Path(deterministic_guess_checkpoint)

            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    "The deterministic-guess checkpoint "
                    "does not exist: "
                    f"{checkpoint_path}"
                )

            model_cfg["deterministic_guess_config"] = {
                "type": "UNetSIC",
                "share_output_block": True,
                "load_dir": str(checkpoint_path),
                "freeze_weights": True,
            }
    else:
        # Deterministic-specific UNet parameters
        model_cfg.update(
            {
                "skip_alignment_method": "padd",
                "add_skip_connections": True,
                "process_skip_connections": False,
            }
        )

    return {
        "type": model_type,
        "config": model_cfg,
    }


def modify_preprocessors(
    cfg: dict[str, Any],
    *,
    add_flattener: bool,
) -> dict[str, Any]:
    """
    Add or remove the flattener preprocessing step.
    MLPs need flattened inputs, but UNets work with spatial data.
    """
    dataset_cfg = cfg["train_loader"]["dataset_config"]

    for data_name in (
        "model",
        "observation",
        "condition",
    ):
        data_cfg = dataset_cfg.get(data_name)

        if not isinstance(
            data_cfg,
            dict,
        ):
            continue

        pipeline = data_cfg.setdefault(
            "preprocessing_pipeline",
            {},
        )
        preprocessors = pipeline.setdefault(
            "preprocessors_list",
            [],
        )

        # Remove any existing flattener
        preprocessors[:] = [
            step
            for step in preprocessors
            if not (
                isinstance(
                    step,
                    dict,
                )
                and step.get("name") == "flattener"
            )
        ]

        # Add flattener if requested
        if add_flattener:
            preprocessors.append(
                {
                    "name": "flattener",
                }
            )

    return cfg


def build_cvae_config_from_params(
    params: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a complete cVAE training configuration from test parameters.
    Applies parameter overrides to the base configuration.
    """
    cfg = base_config()

    cfg["module"]["type"] = "cVAE"

    apply_train_loader_config_params(
        cfg,
        params,
    )

    module_cfg = cfg["module"]["config"]

    original_flow_config = deepcopy(module_cfg.get("prior_flow_config"))

    # Set combined CGCN weight
    module_cfg["combined_CGCN_weight"] = params.get(
        "combined_CGCN_weight",
        0.0,
    )

    # Set posterior variance limits
    posterior_variance_limits = params.get("posterior_variance_limits")

    if posterior_variance_limits is None:
        module_cfg.pop(
            "posterior_variance_limits",
            None,
        )
    else:
        module_cfg["posterior_variance_limits"] = deepcopy(posterior_variance_limits)

    # Configure prior flow if requested
    if params.get(
        "use_prior_flow",
        False,
    ):
        if original_flow_config is None:
            raise ValueError(
                "The base cVAE configuration does not contain prior_flow_config."
            )

        module_cfg["prior_flow_config"] = original_flow_config

        cfg.setdefault(
            "losspipeline",
            {},
        )["reduction"] = "sum"
    else:
        module_cfg.pop(
            "prior_flow_config",
            None,
        )

    # Configure architecture-specific model
    architecture = params["architecture"]

    modify_preprocessors(
        cfg,
        add_flattener=(architecture == "MLP"),
    )

    if architecture == "MLP":
        module_cfg["ModelConfig"] = build_mlp_model_config(
            latent_size=params["latent_size"],
            condition_dependant_latent=params["condition_dependant_latent"],
            condemb_to_decoder=params.get(
                "condemb_to_decoder",
                True,
            ),
            batch_normalization=params.get(
                "batch_normalization",
                False,
            ),
            dropout_rate=params.get("dropout_rate"),
            init_method=params["init_method"],
            cvae=True,
        )
    elif architecture == "UNet":
        module_cfg["ModelConfig"] = build_unet_model_config(
            latent_size=params["latent_size"],
            condition_dependant_latent=params["condition_dependant_latent"],
            condemb_to_decoder=params.get(
                "condemb_to_decoder",
                True,
            ),
            init_method=params["init_method"],
            model_type="UNetSIC",
            deterministic_guess_checkpoint=params.get("deterministic_guess_checkpoint"),
            cvae=True,
        )
    else:
        raise ValueError(f"Unsupported cVAE architecture: {architecture}")

    ensure_cvae_beta_finder(cfg)
    normalize_nullable_fields(cfg)
    validate_generated_config(cfg)

    return cfg


def build_deterministic_config_from_params(
    params: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a complete deterministic regression training configuration from test parameters.
    """
    cfg = base_config()

    apply_train_loader_config_params(
        cfg,
        params,
    )

    architecture = params["architecture"]

    modify_preprocessors(
        cfg,
        add_flattener=(architecture == "MLP"),
    )

    # Build model selector based on architecture
    if architecture == "MLP":
        model_selector = build_mlp_model_config(
            batch_normalization=params["batch_normalization"],
            dropout_rate=params.get("dropout_rate"),
            init_method=params["init_method"],
            append_mode=params["append_mode"],
            cvae=False,
        )
    elif architecture == "UNet":
        model_selector = build_unet_model_config(
            init_method=params["init_method"],
            model_type="UNetSIC",
            cvae=False,
        )
    else:
        raise ValueError(f"Unsupported deterministic architecture: {architecture}")

    cfg["module"] = {
        "type": "deterministic",
        "config": {
            "ModelConfig": model_selector,
        },
    }

    remove_deterministic_beta_finder(cfg)
    normalize_nullable_fields(cfg)
    validate_generated_config(cfg)

    return cfg


def common_training_params() -> dict[str, Any]:
    """
    Return common parameters shared by all architecture tests.
    Used as base for architecture-specific grids.
    """
    return {
        "condition_method": "ensemble_mean",
        "model_ensemble_mean": True,
        "remove_observation": False,
        "add_condition": False,
        "time_features": [
            LEAD_TIME_DIM,
            "month_sin",
            "month_cos",
        ],
        "lead_months": deepcopy(FULL_LEAD_MONTHS),
    }


def grid_cases(
    grid: dict[str, list[Any]],
) -> Iterator[dict[str, Any]]:
    """
    Generate all combinations of parameters from a hyperparameter grid.
    Uses Cartesian product of all grid values.
    """
    keys = list(grid)
    values = [grid[key] for key in keys]

    for combination in product(*values):
        yield dict(
            zip(
                keys,
                combination,
            )
        )


def mark_cvae_expected_result(
    params: dict[str, Any],
) -> dict[str, Any]:
    """
    Mark cVAE test cases as expected pass/fail based on parameter validity.
    Some parameter combinations are invalid and expected to fail.
    """
    # Invalid: prior flow with condition-independent latent
    invalid_prior_flow = (
        params.get("use_prior_flow") is True
        and params.get("condition_dependant_latent") is False
    )

    # Invalid: condition-independent latent without decoder path
    invalid_condition_path = (
        params.get("condition_dependant_latent") is False
        and params.get(
            "condemb_to_decoder",
            True,
        )
        is False
        and params.get("deterministic_guess_checkpoint") is None
    )

    if invalid_prior_flow:
        params["expected_result"] = "FAIL"
        params["expected_failure_reason"] = (
            "A prior flow cannot be used with this "
            "condition-independent latent configuration."
        )
        params["expected_error_pattern"] = (
            r"prior.flow|condition.*latent|"
            r"flow.*condition"
        )
    elif invalid_condition_path:
        params["expected_result"] = "FAIL"
        params["expected_failure_reason"] = (
            "When condition_dependant_latent=False, "
            "the condition embedding must be passed to "
            "the decoder unless a deterministic guess exists."
        )
        params["expected_error_pattern"] = (
            r"condition embedding.*decoder|"
            r"condemb_to_decoder|"
            r"deterministic guess"
        )
    else:
        params["expected_result"] = "PASS"
        params["expected_failure_reason"] = ""
        params.pop(
            "expected_error_pattern",
            None,
        )

    return params


def generate_train_loader_config_cases() -> list[dict[str, Any]]:
    """Generate the data loader configuration test cases."""
    cases = deepcopy(TRAIN_LOADER_CONFIG_CASES)

    if MAX_TRAIN_LOADER_CONFIG_CASES is not None:
        cases = cases[:MAX_TRAIN_LOADER_CONFIG_CASES]

    return cases


def generate_cvae_cases(
    architecture: str,
) -> list[dict[str, Any]]:
    """
    Generate all cVAE test cases for a given architecture (MLP or UNet).
    Creates Cartesian product of CVAE_GRID and architecture-specific parameters.
    """
    grid = deepcopy(CVAE_GRID)

    if architecture == "MLP":
        grid.update(deepcopy(MLP_GRID))

    cases = []

    for params in grid_cases(grid):
        params.update(common_training_params())
        params["architecture"] = architecture
        params["model_type"] = "MLP" if architecture == "MLP" else "UNetSIC"

        # Mark expected pass/fail based on parameter validity
        mark_cvae_expected_result(params)

        params["name"] = clean_name(
            f"cvae_{architecture.lower()}"
            f"_pvl_{params['posterior_variance_limits']}"
            f"_flow_{params['use_prior_flow']}"
            f"_cgcn_{params['combined_CGCN_weight']}"
            f"_latent_{params['latent_size']}"
            f"_condlat_{params['condition_dependant_latent']}"
            f"_conddec_{params['condemb_to_decoder']}"
            f"_bn_{params.get('batch_normalization')}"
            f"_drop_{params.get('dropout_rate')}"
            f"_init_{params['init_method']}"
        )

        cases.append(params)

    maximum = MAX_CVAE_MLP_CASES if architecture == "MLP" else MAX_CVAE_UNET_CASES

    if maximum is not None:
        cases = cases[:maximum]

    return cases


def generate_deterministic_cases(
    architecture: str,
) -> list[dict[str, Any]]:
    """
    Generate all deterministic regression test cases for a given architecture.
    Creates Cartesian product of DETERMINISTIC_GRID and architecture-specific parameters.
    """
    grid = deepcopy(DETERMINISTIC_GRID)

    if architecture == "MLP":
        grid.update(deepcopy(DETERMINISTIC_MLP_GRID))

    cases = []

    for params in grid_cases(grid):
        params.update(common_training_params())
        params["architecture"] = architecture
        params["model_type"] = "MLP" if architecture == "MLP" else "UNetSIC"
        params["expected_result"] = "PASS"
        params["expected_failure_reason"] = ""

        params["name"] = clean_name(
            f"det_{architecture.lower()}"
            f"_append_{params.get('append_mode')}"
            f"_bn_{params.get('batch_normalization')}"
            f"_drop_{params.get('dropout_rate')}"
            f"_init_{params['init_method']}"
        )

        cases.append(params)

    maximum = (
        MAX_DETERMINISTIC_MLP_CASES
        if architecture == "MLP"
        else MAX_DETERMINISTIC_UNET_CASES
    )

    if maximum is not None:
        cases = cases[:maximum]

    return cases


def append_exception_to_log(
    log_path: Path,
) -> None:
    """Append the current exception traceback to a log file."""
    log_path = Path(log_path)
    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with log_path.open(
        "a",
        encoding="utf-8",
    ) as log_file:
        traceback.print_exc(file=log_file)


def remove_existing_case_outputs(
    *paths: Path,
) -> None:
    """Remove any existing output directories/files from a prior test run."""
    for path in paths:
        path = Path(path)

        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


@contextmanager
def limit_config_loader_batches(
    config_path: Path,
    *,
    max_train_batches: int | None,
    max_validation_batches: int | None,
):
    """
    Context manager that patches data loaders to limit batch count.
    Allows testing with just a few batches for speed without modifying the config file.
    """

    config = load_train_config(config_path)

    train_loader_config = config.train_loader

    if train_loader_config is None:
        raise ValueError("The training configuration does not contain a train_loader.")

    config_class = type(train_loader_config)

    # Get the original builder methods
    original_train_builder = getattr(
        config_class,
        "build_train_loader",
        None,
    )

    if not callable(original_train_builder):
        raise TypeError(f"{config_class.__name__}.build_train_loader is not callable.")

    original_validation_builder = getattr(
        config_class,
        "build_validation_loader",
        None,
    )

    # Track patches to restore later
    class_attribute_patches: list[tuple[type, str, Any]] = []

    def patch_attribute(
        owner: type,
        name: str,
        value: Any,
    ) -> None:
        """Patch a class attribute and record the original value."""
        original_value = getattr(
            owner,
            name,
        )
        class_attribute_patches.append(
            (
                owner,
                name,
                original_value,
            )
        )
        setattr(
            owner,
            name,
            value,
        )

    # Create wrapper that limits train batches
    def limited_train_builder(
        self,
        *args,
        **kwargs,
    ):
        loader = original_train_builder(
            self,
            *args,
            **kwargs,
        )

        return LimitedLoader(
            loader,
            max_train_batches,
        )

    patch_attribute(
        config_class,
        "build_train_loader",
        limited_train_builder,
    )

    # Create wrapper that limits validation batches if builder exists
    if callable(original_validation_builder):

        def limited_validation_builder(
            self,
            *args,
            **kwargs,
        ):
            loader = original_validation_builder(
                self,
                *args,
                **kwargs,
            )

            if loader is None:
                return None

            return LimitedLoader(
                loader,
                max_validation_batches,
            )

        patch_attribute(
            config_class,
            "build_validation_loader",
            limited_validation_builder,
        )

    try:
        yield
    finally:
        # Restore original methods
        for (
            owner,
            name,
            original_value,
        ) in reversed(class_attribute_patches):
            setattr(
                owner,
                name,
                original_value,
            )


def run_limited_training(
    config_path: Path,
    *,
    max_train_batches: int | None,
    max_validation_batches: int | None,
) -> None:
    """
    Run training with limited batch counts for faster testing.
    Patches data loaders to limit iterations without modifying the config.
    """
    config_path = Path(config_path)

    with limit_config_loader_batches(
        config_path,
        max_train_batches=max_train_batches,
        max_validation_batches=max_validation_batches,
    ):
        train_main(str(config_path))


def run_training_task(
    task: dict[str, Any],
) -> tuple[bool, str]:
    """
    Run a single training test task.
    Returns (passed, error) tuple.
    """
    paths = case_paths(task["case_name"])

    # Clean up old outputs from any prior run
    remove_existing_case_outputs(
        paths["experiment_dir"],
        paths["config_path"],
        paths["log_path"],
    )

    try:
        # Get the configuration builder function
        cfg_builder = task.get("cfg_builder")

        if cfg_builder is None:
            raise KeyError(f"Training task is missing cfg_builder: {task['case_name']}")

        if not callable(cfg_builder):
            raise TypeError(
                "Training task cfg_builder must be callable, "
                f"got {type(cfg_builder).__name__} for "
                f"{task['case_name']}."
            )

        # Build the test case configuration
        case_cfg = cfg_builder()

        if not isinstance(
            case_cfg,
            dict,
        ):
            raise TypeError(
                "cfg_builder must return a dictionary, "
                f"got {type(case_cfg).__name__} for "
                f"{task['case_name']}."
            )

        case_cfg = deepcopy(case_cfg)
        case_cfg["experiment_dir"] = str(paths["experiment_dir"])

        # Write configuration to YAML file
        write_yaml(
            case_cfg,
            paths["config_path"],
        )

        print(
            f"\nStarting {task['suite']}:{task['case_name']}",
            flush=True,
        )
        print(
            f"Log: {paths['log_path']}",
            flush=True,
        )

        # Run training with limited batches
        run_silenced(
            lambda: run_limited_training(
                paths["config_path"],
                max_train_batches=(MAX_TRAIN_BATCHES_PER_CASE),
                max_validation_batches=(MAX_VALIDATION_BATCHES_PER_CASE),
            ),
            paths["log_path"],
        )

        # Verify training completed successfully
        assert_training_completed(
            experiment_dir=(paths["experiment_dir"]),
            log_path=(paths["log_path"]),
        )

        passed = True
        error = ""

    except Exception as exc:
        passed = False
        error = f"{type(exc).__name__}: {exc}"

        append_exception_to_log(paths["log_path"])
        cleanup_logging_handlers()

    log_result(
        task=task,
        paths=paths,
        passed=passed,
        error=error,
    )

    return passed, error


def make_train_config(
    output_path: Path,
    experiment_dir: Path,
) -> Path:
    """Create and save a training configuration for baseline tests."""
    params = {
        "architecture": "MLP",
        "model_type": "MLP",
        "condition_method": "ensemble_mean",
        "model_ensemble_mean": True,
        "remove_observation": False,
        "add_condition": False,
        "time_features": [
            LEAD_TIME_DIM,
            "month_sin",
            "month_cos",
        ],
        "lead_months": deepcopy(FULL_LEAD_MONTHS),
        "posterior_variance_limits": None,
        "use_prior_flow": False,
        "combined_CGCN_weight": 0.0,
        "latent_size": 4,
        "condition_dependant_latent": True,
        "condemb_to_decoder": True,
        "batch_normalization": False,
        "dropout_rate": None,
        "init_method": "default",
    }

    cfg = build_cvae_config_from_params(params)
    cfg["experiment_dir"] = str(experiment_dir)

    ensure_cvae_beta_finder(cfg)

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return write_yaml(
        cfg,
        output_path,
    )


def run_resume_training(
    root_dir: Path,
) -> None:
    """Test training resumption from checkpoints."""
    experiment_dir = root_dir / "resume"

    if experiment_dir.exists():
        shutil.rmtree(experiment_dir)

    # First training run
    first_config = make_train_config(
        GENERATED_CONFIG_DIR / "resume_train.yaml",
        experiment_dir,
    )
    first_log = LOG_DIR / "resume_first.log"

    remove_existing_case_outputs(first_log)

    run_silenced(
        lambda: run_limited_training(
            first_config,
            max_train_batches=(MAX_BASELINE_TRAIN_BATCHES),
            max_validation_batches=(MAX_BASELINE_VALIDATION_BATCHES),
        ),
        first_log,
    )

    assert_training_completed(
        experiment_dir=experiment_dir,
        log_path=first_log,
    )

    # Resume training from checkpoint
    resumed_cfg = prepare_config(first_config)

    if resumed_cfg is None:
        raise ValueError(f"The resume configuration is empty: {first_config}")

    normalize_numeric_fields(resumed_cfg)

    resumed_cfg["resume_dir"] = str(experiment_dir)

    resumed_cfg["max_epochs"] = 2

    ensure_cvae_beta_finder(resumed_cfg)

    second_config = write_yaml(
        resumed_cfg,
        GENERATED_CONFIG_DIR / "resume_second.yaml",
    )
    second_log = LOG_DIR / "resume_second.log"

    remove_existing_case_outputs(second_log)

    run_silenced(
        lambda: run_limited_training(
            second_config,
            max_train_batches=(MAX_BASELINE_TRAIN_BATCHES),
            max_validation_batches=(MAX_BASELINE_VALIDATION_BATCHES),
        ),
        second_log,
    )

    assert_training_completed(
        experiment_dir=experiment_dir,
        log_path=second_log,
    )


def run_dataset_pipeline(
    root_dir: Path,
) -> None:
    """Test the data loading and preprocessing pipeline."""
    experiment_dir = root_dir / "dataset_pipeline"

    if experiment_dir.exists():
        shutil.rmtree(experiment_dir)

    config_path = make_train_config(
        GENERATED_CONFIG_DIR / "dataset_pipeline.yaml",
        experiment_dir,
    )
    log_path = LOG_DIR / "dataset_pipeline.log"

    remove_existing_case_outputs(log_path)

    def run_pipeline() -> None:
        """Test that data loads and contains expected keys."""
        config = load_train_config(config_path)
        distributed = Distributed.get_instance()

        config.train_loader.setup_distributed(distributed)

        train_loader = config.train_loader.build_train_loader()

        if len(train_loader.dataset) <= 0:
            raise AssertionError("The training dataset is empty.")

        # Check first sample contains required keys
        sample = train_loader.dataset[0]

        required_keys = {
            "input",
            "target",
            "added_features",
        }
        missing_keys = required_keys - set(sample)

        if missing_keys:
            raise AssertionError(
                f"Dataset sample is missing required keys: {sorted(missing_keys)}"
            )

        print(f"dataset_samples={len(train_loader.dataset)}")
        print(f"loader_batches={len(train_loader)}")
        print(f"sample_keys={sorted(sample)}")

    run_silenced(
        run_pipeline,
        log_path,
    )


def run_baseline_task(
    task: dict[str, Any],
) -> tuple[bool, str]:
    """
    Run a baseline test task (resume, dataset pipeline, etc).
    Returns (passed, error) tuple.
    """
    paths = {
        "config_path": Path(task["config_path"]),
        "experiment_dir": Path(task["experiment_dir"]),
        "log_path": Path(task["log_path"]),
    }

    try:
        # Get the runner function
        runner = task.get("runner")

        if runner is None:
            raise KeyError(f"Baseline task is missing runner: {task['case_name']}")

        if not callable(runner):
            raise TypeError(
                "Baseline task runner must be callable, "
                f"got {type(runner).__name__} for "
                f"{task['case_name']}."
            )

        print(
            f"\nStarting baseline:{task['case_name']}",
            flush=True,
        )
        print(
            f"Log: {paths['log_path']}",
            flush=True,
        )

        runner()

        passed = True
        error = ""

    except Exception as exc:
        passed = False
        error = f"{type(exc).__name__}: {exc}"

        append_exception_to_log(paths["log_path"])
        cleanup_logging_handlers()

    log_result(
        task=task,
        paths=paths,
        passed=passed,
        error=error,
    )

    return passed, error


def add_baseline_tasks(
    tasks: list[dict[str, Any]],
) -> None:
    """Add baseline test tasks if enabled."""
    if not RUN_BASELINE_TESTS:
        return

    tasks.extend(
        [
            {
                "suite": "baseline",
                "case_name": "resume_training",
                "module_type": "cVAE",
                "architecture": "MLP",
                "params": {
                    "model_type": "MLP",
                    "architecture": "MLP",
                    "expected_result": "PASS",
                    "expected_failure_reason": "",
                },
                "runner": partial(
                    run_resume_training,
                    OUTPUT_DIR,
                ),
                "config_path": (GENERATED_CONFIG_DIR / "resume_second.yaml"),
                "experiment_dir": (OUTPUT_DIR / "resume"),
                "log_path": (LOG_DIR / "resume_second.log"),
            },
            {
                "suite": "baseline",
                "case_name": "dataset_pipeline",
                "module_type": "cVAE",
                "architecture": "MLP",
                "params": {
                    "model_type": "MLP",
                    "architecture": "MLP",
                    "expected_result": "PASS",
                    "expected_failure_reason": "",
                },
                "runner": partial(
                    run_dataset_pipeline,
                    OUTPUT_DIR,
                ),
                "config_path": (GENERATED_CONFIG_DIR / "dataset_pipeline.yaml"),
                "experiment_dir": (OUTPUT_DIR / "dataset_pipeline"),
                "log_path": (LOG_DIR / "dataset_pipeline.log"),
            },
        ]
    )


def add_train_loader_config_tasks(
    tasks: list[dict[str, Any]],
) -> None:
    """Add train loader configuration test tasks if enabled."""
    if not RUN_TRAIN_LOADER_CONFIG_CASES:
        return

    defaults = {
        "architecture": "MLP",
        "model_type": "MLP",
        "posterior_variance_limits": None,
        "use_prior_flow": False,
        "combined_CGCN_weight": 0.0,
        "latent_size": 4,
        "condition_dependant_latent": True,
        "condemb_to_decoder": True,
        "batch_normalization": False,
        "dropout_rate": None,
        "init_method": "default",
    }

    for case in generate_train_loader_config_cases():
        params = {
            **case,
            "architecture": "MLP",
            "model_type": "MLP",
        }

        build_params = {
            **defaults,
            **params,
        }

        tasks.append(
            {
                "suite": "train_loader_config",
                "case_name": params["name"],
                "module_type": "cVAE",
                "architecture": "MLP",
                "params": params,
                "cfg_builder": partial(
                    build_cvae_config_from_params,
                    build_params,
                ),
            }
        )


def add_cvae_tasks(
    tasks: list[dict[str, Any]],
    architecture: str,
) -> None:
    """Add cVAE test tasks for a given architecture if enabled."""
    enabled = RUN_CVAE_MLP_GRID if architecture == "MLP" else RUN_CVAE_UNET_GRID

    if not enabled:
        return

    for case in generate_cvae_cases(architecture):
        params = deepcopy(case)

        tasks.append(
            {
                "suite": (f"cvae_{architecture.lower()}"),
                "case_name": params["name"],
                "module_type": "cVAE",
                "architecture": architecture,
                "params": params,
                "cfg_builder": partial(
                    build_cvae_config_from_params,
                    params,
                ),
            }
        )


def add_deterministic_tasks(
    tasks: list[dict[str, Any]],
    architecture: str,
) -> None:
    """Add deterministic regression test tasks for a given architecture if enabled."""
    enabled = (
        RUN_DETERMINISTIC_MLP_GRID
        if architecture == "MLP"
        else RUN_DETERMINISTIC_UNET_GRID
    )

    if not enabled:
        return

    for case in generate_deterministic_cases(architecture):
        params = deepcopy(case)

        tasks.append(
            {
                "suite": (f"deterministic_{architecture.lower()}"),
                "case_name": params["name"],
                "module_type": "deterministic",
                "architecture": architecture,
                "params": params,
                "cfg_builder": partial(
                    build_deterministic_config_from_params,
                    params,
                ),
            }
        )


def validate_tasks(
    tasks: list[dict[str, Any]],
) -> None:
    """Validate that all tasks are properly configured."""
    case_names = set()

    for index, task in enumerate(tasks):
        required_keys = {
            "suite",
            "case_name",
            "module_type",
            "architecture",
            "params",
        }

        missing_keys = required_keys - set(task)

        if missing_keys:
            raise ValueError(
                f"Task {index} is missing required keys: {sorted(missing_keys)}"
            )

        case_name = task["case_name"]

        if case_name in case_names:
            raise ValueError(f"Duplicate integration case name: {case_name}")

        case_names.add(case_name)

        # Each task must have exactly one of runner or cfg_builder
        has_runner = "runner" in task
        has_builder = "cfg_builder" in task

        if has_runner == has_builder:
            raise ValueError(
                f"Task {case_name!r} must define exactly one of runner or cfg_builder."
            )

        callable_name = "runner" if has_runner else "cfg_builder"

        if not callable(task[callable_name]):
            raise TypeError(
                f"Task {case_name!r} field {callable_name!r} must be callable."
            )


def classify_result(
    *,
    task: dict[str, Any],
    passed: bool,
    error: str,
    unexpected_failures: list[str],
    expected_failures: list[str],
    unexpected_passes: list[str],
) -> None:
    """
    Classify a test result as expected or unexpected.
    Tracks unexpected failures/passes for final reporting.
    """
    case_name = task["case_name"]
    params = task["params"]

    expected_result = params.get(
        "expected_result",
        "PASS",
    )

    if expected_result == "FAIL":
        # Test was expected to fail
        if passed:
            # But it passed - this is unexpected!
            unexpected_passes.append(case_name)
            return

        # Verify it failed for the right reason
        expected_error_pattern = params.get("expected_error_pattern")

        if expected_error_pattern and not re.search(
            expected_error_pattern,
            error,
            flags=(re.IGNORECASE | re.DOTALL),
        ):
            unexpected_failures.append(
                f"{case_name}: failed for the wrong reason: {error}"
            )
            return

        # Failed as expected
        expected_failures.append(case_name)
        return

    # Test was expected to pass
    if not passed:
        unexpected_failures.append(f"{case_name}: {error}")


def write_summary(
    tasks: list[dict[str, Any]],
    unexpected_failures: list[str],
    expected_failures: list[str],
    unexpected_passes: list[str],
) -> None:
    """Write a summary report of the test run."""
    actual_failures = len(unexpected_failures) + len(expected_failures)
    actual_passes = len(tasks) - actual_failures

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(f"total_cases: {len(tasks)}\n")
        file.write(f"actual_passes: {actual_passes}\n")
        file.write(f"unexpected_failures: {len(unexpected_failures)}\n")
        file.write(f"expected_failures: {len(expected_failures)}\n")
        file.write(f"unexpected_passes: {len(unexpected_passes)}\n")
        file.write(f"max_train_batches_per_case: {MAX_TRAIN_BATCHES_PER_CASE}\n")
        file.write(
            f"max_validation_batches_per_case: {MAX_VALIDATION_BATCHES_PER_CASE}\n"
        )
        file.write(f"results_csv: {RESULTS_CSV}\n")
        file.write(f"log_dir: {LOG_DIR}\n")

        groups = [
            (
                "unexpected_failures",
                unexpected_failures,
            ),
            (
                "expected_failures",
                expected_failures,
            ),
            (
                "unexpected_passes",
                unexpected_passes,
            ),
        ]

        for title, values in groups:
            if not values:
                continue

            file.write(f"\n{title}:\n")

            for value in values:
                file.write(f"- {value}\n")


def main() -> None:
    """Main entry point for the integration test suite."""
    prepare_output_directories()
    cleanup_logging_handlers()
    init_csv()

    tasks: list[dict[str, Any]] = []

    # Add all test tasks based on enabled suites
    add_baseline_tasks(tasks)
    add_train_loader_config_tasks(tasks)
    add_cvae_tasks(
        tasks,
        "MLP",
    )
    add_cvae_tasks(
        tasks,
        "UNet",
    )
    add_deterministic_tasks(
        tasks,
        "MLP",
    )
    add_deterministic_tasks(
        tasks,
        "UNet",
    )

    validate_tasks(tasks)

    unexpected_failures: list[str] = []
    expected_failures: list[str] = []
    unexpected_passes: list[str] = []

    # Run all tasks with progress bar
    with tqdm(
        total=len(tasks),
        desc="integration",
        unit="case",
        dynamic_ncols=True,
        leave=True,
        bar_format=(
            "{l_bar}{bar}| "
            "{n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, "
            "{rate_fmt}] "
            "{postfix}"
        ),
    ) as progress:
        for task in tasks:
            suite = task["suite"]
            case_name = task["case_name"]

            progress.set_postfix_str(f"{suite}:{case_name[:60]}")

            # Run baseline or training task
            if "runner" in task:
                passed, error = run_baseline_task(task)
            else:
                passed, error = run_training_task(task)

            # Classify the result
            classify_result(
                task=task,
                passed=passed,
                error=error,
                unexpected_failures=(unexpected_failures),
                expected_failures=(expected_failures),
                unexpected_passes=(unexpected_passes),
            )

            progress.set_postfix_str(
                "unexpected_failures="
                f"{len(unexpected_failures)} "
                "expected_failures="
                f"{len(expected_failures)} "
                "unexpected_passes="
                f"{len(unexpected_passes)}"
            )

            progress.update(1)

    cleanup_logging_handlers()

    write_summary(
        tasks,
        unexpected_failures,
        expected_failures,
        unexpected_passes,
    )

    # Check for failures
    if unexpected_failures or unexpected_passes:
        problems = []

        if unexpected_failures:
            problems.append(f"{len(unexpected_failures)} unexpected failure(s)")

        if unexpected_passes:
            problems.append(f"{len(unexpected_passes)} unexpected pass(es)")

        raise RuntimeError(
            "Integration suite completed with "
            + " and ".join(problems)
            + f". See {RESULTS_CSV} and {LOG_DIR}."
        )

    print("Integration suite completed successfully.")
    print(f"Results: {RESULTS_CSV}")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Logs: {LOG_DIR}")


if __name__ == "__main__":
    main()
