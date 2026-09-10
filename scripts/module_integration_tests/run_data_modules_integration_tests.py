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

# Import different model architectures we support
# MLPs (simple dense networks) and UNets (encoder-decoder with skip connections)
# Each has two variants: CVAE (Conditional Variational AutoEncoder - probabilistic)
# and deterministic (direct prediction)
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

# Suppress warnings during testing - we're checking structure, not debugging warnings
warnings.filterwarnings("ignore")
logging.raiseExceptions = False


# Find the project root (documentation to audit)
ROOT = Path("/fs/site7/eccc/crd/cccma/users/rna002/CCCma-PPP/scripts")
# These are the "template" configs we use as starting points for each test
BASE_TRAIN_CONFIG = ROOT / "integration_suite_train_config.yaml"
BASE_INFERENCE_CONFIG = ROOT / "scripts" / "integration_suite_inference_config.yaml"
# All results go here - one subdirectory per test case
OUTPUT_DIR = ROOT / "../../output" / "data_modules_integration_test_results"

# Make subdirectories for organizing test results
# Each test case gets its own directory under CASES_DIR
CASES_DIR = OUTPUT_DIR / "cases"
# Summary CSV file listing all test results (one row per case)
RESULTS_CSV = OUTPUT_DIR / "integration_results.csv"
# Overall summary statistics from entire test run
SUMMARY_PATH = OUTPUT_DIR / "summary.txt"

# These are the required time dimensions for samples
# init_time: when the forecast starts (e.g., 2020-01)
# lead_time: how far ahead we're predicting (e.g., 3 months)
INIT_TIME_DIM, LEAD_TIME_DIM = required_sample_dimensions

# Paths to actual data files we'll use in external condition tests
# tas_stdr is temperature standard deviation (a real external data source)
TAS_STDR_PATH = Path(
    "/space/hall7/sitestore/eccc/crd/cccma/users/rpg002/data/tas_std/forecast_data"
)
# dummy_mask is a static spatial mask (same for all time steps)
DUMMY_MASK_PATH = Path(
    "/space/hall7/sitestore/eccc/crd/cccma/users/rpg002/data/tas/dummy_mask"
)

# Toggle flags to run different test suites
# Set to False to skip a test category (useful for quick debugging!)
RUN_DATA_CONFIG_COMBINATIONS = (
    True  # Test 5 ways to combine model/condition/observation
)
RUN_DATASET_GETITEM_CASES = True  # Test dataset[index] contract
RUN_TRAIN_DATALOADER_CASES = True  # Test different dataloader configs
RUN_INFERENCE_DATALOADER_CASES = True  # Test that inference uses same data as training
RUN_MULTIWORKER_CASES = True  # Test parallel data loading
RUN_FUTURE_MASK_CASES = True  # Test that no future data leaks in
RUN_OBSERVATION_CUTOFF_CASES = True  # Test observation availability filtering
RUN_DISTRIBUTED_PREPROCESSOR_CASES = True  # Test rank-0 preprocessing setup
RUN_EXTERNAL_CONDITION_CASES = True  # Test external data files (tas_stdr, mask)
RUN_CHANNEL_PLOTS = True  # Create PNG visualizations of data
RUN_RAW_DATA_EXPORT = True  # Export raw data to NetCDF

# Limits on how many cases/batches to test (prevents tests from taking forever)
# None means "test everything"
MAX_CASES: int | None = None
# Only iterate through first N batches per test case (saves time)
MAX_BATCHES_PER_CASE = 4
# Only plot first N channels (too many channels = too many PNGs)
MAX_PLOT_CHANNELS = 16
# Random seed for reproducible shuffling
SHUFFLE_SEED = 1847


class DataCombinationScenarios:
    """
    Different ways we can combine model, condition, and observation data.
    Each scenario is a dict describing what data should be present and how it should behave.

    Think of these as "data recipes":
    - Some use model data as input to reconstruct model data (autoencoding)
    - Some use model+condition to bias-correct observations
    - Some use just model to bias-correct observations
    """

    # Valid time features we can add to samples
    # These are derived from the temporal metadata (init_time, lead_time)
    VALID_TIME_FEATURES = {
        INIT_TIME_DIM,  # Absolute time when forecast starts
        LEAD_TIME_DIM,  # Months ahead being predicted
        "month_sin",  # Sinusoidal encoding of month (captures seasonality)
        "month_cos",  # Cosine version for orthogonality
        "day_sin",  # Day of year encoding
        "day_cos",  # Cosine version
    }
    # These are the default time features we use in most tests
    DEFAULT_TIME_FEATURES = [LEAD_TIME_DIM, "month_sin", "month_cos"]
    # Default lead time range for testing (1-12 months = seasonal forecast)
    FULL_LEAD_MONTHS = {"start": 1, "end": 12}

    # Scenario 1: Just take model data and try to reconstruct it (no condition needed)
    # This tests: can the model learn to copy its own output?
    AUTOENCODE_MODEL_ONLY = {
        "name": "autoencode_model_conditioned_on_itself",
        "description": (
            "Autoencode model data: reconstruct model data conditioned on itself. "
            "Condition is sampled according to condition_method."
        ),
        "model_present": True,  # We have model forecast data
        "condition_present": False,  # No external condition
        "observation_present": False,  # No observation to validate against
        "condition_method_requirement": "MUST",  # We need a way to sample model data
        "condition_method": "ensemble_mean",  # Use mean over ensemble members
        "model_ensemble_mean": True,  # Average ensemble members for input
        "expected_input": "model data selected according to condition_method",
        "expected_target": "model data at sampled init_time, lead_time, and ensemble coords",
        "expected_result": "PASS",
    }

    # Scenario 2: Use explicit condition data along with model data
    # This tests: can we learn with an external condition?
    AUTOENCODE_MODEL_WITH_CONDITION = {
        "name": "autoencode_model_with_explicit_condition",
        "description": (
            "Autoencode model data using an explicit condition. "
            "Condition is sampled according to condition_method."
        ),
        "model_present": True,
        "condition_present": True,  # We have a separate condition data source
        "observation_present": False,
        "condition_method_requirement": "MUST",
        "condition_method": "ensemble_mean",
        "model_ensemble_mean": True,
        "condition_ensemble_mean": True,  # Average condition ensemble too
        "expected_input": "explicit condition data selected according to condition_method",
        "expected_target": "model data at sampled init_time, lead_time, and ensemble coords",
        "expected_result": "PASS",
    }

    # Scenario 3: Bias correction - correct model+condition using observations
    # This tests: can we learn a correction from model/condition to observations?
    BIAS_CORRECT_MODEL_AND_CONDITION = {
        "name": "bias_correct_model_and_condition_to_observation",
        "description": (
            "Bias-correct to observations using model and condition as conditioning fields. "
            "Condition sampling is determined by condition_method."
        ),
        "model_present": True,
        "condition_present": True,
        "observation_present": True,  # We have observations to correct toward
        "condition_method_requirement": "MUST",
        "condition_method": "same_member",  # Match condition ensemble member to model
        "model_ensemble_mean": False,  # Keep individual ensemble members
        "condition_ensemble_mean": False,  # Keep individual ensemble members
        "expected_input": "model + condition data at compatible sample coordinates",
        "expected_target": "observation data at calculated valid time",
        "expected_result": "PASS",
    }

    # Scenario 4: Bias correction - correct model using observations (with method)
    # This tests: do we need a condition, or is model alone enough for bias correction?
    BIAS_CORRECT_MODEL_WITH_METHOD = {
        "name": "bias_correct_model_to_observation_with_method",
        "description": (
            "Bias-correct to observations using model data. "
            "Model sampling is determined by condition_method."
        ),
        "model_present": True,
        "condition_present": False,  # No explicit condition data
        "observation_present": True,
        "condition_method_requirement": "OPTIONAL",  # May or may not use sampling method
        "condition_method": "ensemble_mean",  # Average model ensemble
        "model_ensemble_mean": True,
        "expected_input": "model data selected according to condition_method",
        "expected_target": "observation data at calculated valid time",
        "expected_result": "PASS",
    }

    # Scenario 5: Direct indexing - just grab model at specific indices and correct it
    # This tests: what if we don't use any ensemble averaging?
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
        "condition_method": None,  # No sampling method
        "model_ensemble_mean": False,  # Direct indexing, no averaging
        "expected_input": "model data at sampled init_time, lead_time, and ensemble coords",
        "expected_target": "observation data at calculated valid time",
        "expected_result": "PASS",
    }

    @classmethod
    def all(cls) -> list[dict[str, Any]]:
        """Return all scenarios as a list"""
        return [
            cls.AUTOENCODE_MODEL_ONLY,
            cls.AUTOENCODE_MODEL_WITH_CONDITION,
            cls.BIAS_CORRECT_MODEL_AND_CONDITION,
            cls.BIAS_CORRECT_MODEL_WITH_METHOD,
            cls.BIAS_CORRECT_MODEL_DIRECT,
        ]


class ExternalConditionScenarios:
    """
    Test cases for using external data as conditions (like temperature stddev, masks, etc)

    These test whether we can load data from different file paths and combine it with
    model/observation data. Important for real-world use cases where you might have:
    - Static masks (same for all times)
    - Auxiliary fields (temperature stddev, elevation, etc)
    - Different ensemble structures in different files
    """

    # Using tas_stdr (temperature standard deviation) as condition data
    # This is a real external data source without ensemble members
    TAS_STDR_CONDITION = {
        "name": "bias_correct_with_tas_stdr_condition",
        "description": "Use tas_stdr as a real-data condition without ensemble dimension.",
        "model_present": True,
        "condition_present": True,
        "observation_present": True,
        "condition_kind": "tas_stdr",  # Type of external data
        "condition_path": str(TAS_STDR_PATH),  # Where to find it
        "condition_method_requirement": "MUST",
        "condition_method": "ensemble_mean",
        "model_ensemble_mean": False,
        "condition_ensemble_mean": None,  # tas_stdr has no ensemble dimension
        "condition_has_ensemble": False,  # Important: this data has no ensemble axis
        "expected_input": "model data concatenated with ensemble-free tas_stdr condition",
        "expected_target": "observation data at calculated valid time",
        "expected_result": "PASS",
    }

    # Using a static mask (same for all time steps) as additional condition
    STATIC_MASK_CONDITION = {
        "name": "bias_correct_with_static_dummy_mask",
        "description": "Use static dummy mask as an additional static condition.",
        "model_present": True,
        "condition_present": True,
        "observation_present": True,
        "condition_kind": "static",
        "condition_path": str(DUMMY_MASK_PATH),
        "condition_method_requirement": "MUST",
        "condition_method": "static",  # Static data, not time-varying
        "model_ensemble_mean": False,
        "condition_ensemble_mean": None,
        "condition_has_ensemble": False,
        "expected_input": "model data concatenated with static dummy-mask channel",
        "expected_target": "observation data at calculated valid time",
        "expected_result": "PASS",
    }

    # tas_stdr but without any sampling method (direct time indexing)
    TAS_STDR_WITH_DIRECT_INDEXING = {
        "name": "bias_correct_with_tas_stdr_direct_indexing",
        "description": "Use tas_stdr without sampling method (direct time indexing).",
        "model_present": True,
        "condition_present": True,
        "observation_present": True,
        "condition_kind": "tas_stdr",
        "condition_path": str(TAS_STDR_PATH),
        "condition_method_requirement": "OPTIONAL",
        "condition_method": None,  # No sampling method
        "model_ensemble_mean": False,
        "condition_ensemble_mean": None,
        "condition_has_ensemble": False,
        "expected_input": "model and tas_stdr data at sampled init_time, lead_time",
        "expected_target": "observation data at calculated valid time",
        "expected_result": "PASS",
    }

    # Static mask with ensemble model members
    DUMMY_MASK_WITH_ENSEMBLE_MODEL = {
        "name": "bias_correct_with_dummy_mask_ensemble_model",
        "description": "Use static mask with model ensemble mean for feature combination.",
        "model_present": True,
        "condition_present": True,
        "observation_present": True,
        "condition_kind": "static",
        "condition_path": str(DUMMY_MASK_PATH),
        "condition_method_requirement": "MUST",
        "condition_method": "static",
        "model_ensemble_mean": True,  # Average model ensemble
        "condition_ensemble_mean": None,
        "condition_has_ensemble": False,
        "expected_input": "model ensemble mean concatenated with static mask channel",
        "expected_target": "observation data at calculated valid time",
        "expected_result": "PASS",
    }

    # Try to match ensemble members between tas_stdr and model
    TAS_STDR_SAME_MEMBER_METHOD = {
        "name": "bias_correct_with_tas_stdr_same_member",
        "description": "Use tas_stdr with member matching (if ensemble exists in data).",
        "model_present": True,
        "condition_present": True,
        "observation_present": True,
        "condition_kind": "tas_stdr",
        "condition_path": str(TAS_STDR_PATH),
        "condition_method_requirement": "MUST",
        "condition_method": "same_member",  # Try to match ensemble members
        "model_ensemble_mean": False,
        "condition_ensemble_mean": False,
        "condition_has_ensemble": False,
        "expected_input": "model and tas_stdr at matched ensemble indices",
        "expected_target": "observation data at calculated valid time",
        "expected_result": "PASS",
    }

    @classmethod
    def all(cls) -> list[dict[str, Any]]:
        """Return all external condition scenarios"""
        return [
            cls.TAS_STDR_CONDITION,
            cls.STATIC_MASK_CONDITION,
            cls.TAS_STDR_WITH_DIRECT_INDEXING,
            cls.DUMMY_MASK_WITH_ENSEMBLE_MODEL,
            cls.TAS_STDR_SAME_MEMBER_METHOD,
        ]


class DataloaderScenarios:
    """
    Different ways to configure the dataloader for training
    (batch size, shuffling, time features, workers, etc)

    These test various dataloader configurations to make sure:
    - Different year ranges work
    - Time features are included correctly
    - Multiple workers don't break anything
    - Shuffling still gets all samples
    """

    # Simple case: just load a specific year range
    YEAR_SLICE_ONLY = {
        "name": "train_year_slice",
        "description": "Select exactly the configured training-year range and iterate fully.",
        "year_slice": [1980, 1981],  # Test years 1980-1981
        "num_data_workers": 0,  # Single worker
        "shuffle": False,  # Deterministic order
        "time_features": deepcopy(DataCombinationScenarios.DEFAULT_TIME_FEATURES),
        "check": "year_slice",  # What to validate
        "expected_result": "PASS",
    }

    # Test with no time features (just spatial/ensemble data)
    NO_TIME_FEATURES = {
        "name": "train_time_features_none",
        "description": "Build batches without any additional time features.",
        "year_slice": [1980, 1981],
        "num_data_workers": 0,
        "shuffle": False,
        "time_features": None,  # Empty list = no time features
        "check": "time_features",
        "expected_result": "PASS",
    }

    # Test with ALL possible time features
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

    # Multiple workers loading data in parallel (no shuffling)
    MULTIWORKER_NO_SHUFFLE = {
        "name": "train_multiworker_no_shuffle",
        "description": (
            "Multiple data workers must preserve sample coverage and deterministic order "
            "when shuffling is disabled."
        ),
        "year_slice": [1980, 1981],
        "num_data_workers": 2,  # Two parallel workers
        "shuffle": False,  # Deterministic = must get same samples in same order as single worker
        "time_features": deepcopy(DataCombinationScenarios.DEFAULT_TIME_FEATURES),
        "check": "multiworker",
        "expected_result": "PASS",
    }

    # Multiple workers with shuffling enabled
    MULTIWORKER_WITH_SHUFFLE = {
        "name": "train_multiworker_shuffle",
        "description": (
            "Multiple data workers with shuffling must preserve sample coverage "
            "while randomizing order."
        ),
        "year_slice": [1980, 1981],
        "num_data_workers": 2,
        "shuffle": True,  # Shuffled = must get same samples but different order
        "time_features": deepcopy(DataCombinationScenarios.DEFAULT_TIME_FEATURES),
        "check": "multiworker",
        "expected_result": "PASS",
    }

    @classmethod
    def all(cls) -> list[dict[str, Any]]:
        """Return all dataloader scenarios"""
        return [
            cls.YEAR_SLICE_ONLY,
            cls.NO_TIME_FEATURES,
            cls.ALL_TIME_FEATURES,
            cls.MULTIWORKER_NO_SHUFFLE,
            cls.MULTIWORKER_WITH_SHUFFLE,
        ]


class TemporalMaskScenarios:
    """
    Test what happens when we have an observation cutoff date
    (i.e., observations only available up to a certain time)

    In real forecasting, observations are delayed. You might have model forecasts
    for 2025 but observations only up to 2024. These tests check that we correctly
    exclude samples where the target (observation) isn't available yet.
    """

    # Cutoff is before all the times we're trying to predict
    # Expected: should return no valid samples (nothing to train on)
    BEFORE_REQUESTED_TIMES = {
        "name": "observation_cutoff_before_requested_times",
        "description": (
            "Observation cutoff is before all requested valid times. "
            "Should return no valid samples."
        ),
        "observation_cutoff_position": "before",
        "expected_available": 0,  # Expect zero samples
        "expected_result": "PASS",
    }

    # Cutoff falls somewhere in the middle of our time range
    # Expected: should partially filter samples (some valid, some not)
    BETWEEN_REQUESTED_TIMES = {
        "name": "observation_cutoff_between_requested_times",
        "description": (
            "Observation cutoff falls between first and last requested valid times. "
            "Should partially filter samples."
        ),
        "observation_cutoff_position": "between",
        "expected_result": "PASS",
    }

    # Cutoff is after all times we're predicting (no filtering needed)
    # Expected: should return all samples (observations available for everything)
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
        """Return all temporal mask scenarios"""
        return [
            cls.BEFORE_REQUESTED_TIMES,
            cls.BETWEEN_REQUESTED_TIMES,
            cls.AFTER_REQUESTED_TIMES,
        ]


# Columns we'll write to the CSV results file
# This is a comprehensive record of each test's configuration and outcome
CSV_COLUMNS = [
    "timestamp",  # When did this test run
    "suite",  # Which test suite (data_config_combination, dataloader, etc)
    "case_name",  # Name of this specific test case
    "description",  # Human description of what we're testing
    "model_present",  # Did this test use model data?
    "condition_present",  # Did this test use condition data?
    "observation_present",  # Did this test use observation data?
    "condition_kind",  # Type of condition (dynamic, static, tas_stdr, etc)
    "condition_method_requirement",  # Was condition_method required/optional?
    "condition_method",  # How did we sample condition (ensemble_mean, same_member, etc)
    "model_ensemble_mean",  # Did we average model ensemble members?
    "condition_ensemble_mean",  # Did we average condition ensemble members?
    "time_features",  # Which time features did we include?
    "year_slice",  # Which years did we test?
    "num_data_workers",  # How many parallel workers?
    "expected_input",  # What kind of input data was expected?
    "expected_target",  # What kind of target data was expected?
    "expected_result",  # Did we expect this test to PASS or FAIL?
    "expected_failure_reason",  # If failure expected, why?
    "expected_error_pattern",  # If failure expected, what error message pattern?
    "case_dir",  # Output directory for this test
    "config_path",  # Path to config.yaml used
    "log_path",  # Path to case.log
    "preprocessor_dir",  # Path to preprocessor artifacts
    "plot_dir",  # Path to PNG plots
    "result",  # Actual result (PASS or FAIL)
    "error",  # If failed, the error message
]

# Field names that typically contain file/directory paths
# Used when replacing paths in configs for external data
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

# Field names related to ensemble members
# Used when removing ensemble config if data doesn't have ensemble dimension
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
    """
    Convert any value to a filesystem-safe name.
    Removes special characters that might cause issues in file paths.
    Example: "autoencode_model_conditioned_on_itself" -> "autoencode_model_conditioned_on_itself"
             "condition=ensemble_mean" -> "condition_ensemble_mean"
    """
    value = str(value).replace("None", "none")
    # Replace special chars with underscore (keep alphanumeric, dash, dot only)
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    # Remove duplicate underscores
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def case_paths(case_name: str) -> dict[str, Path]:
    """
    Generate all the paths we need for a single test case.
    Makes it easy to keep all files organized without hardcoding paths everywhere.
    Returns a dict like:
    {
        'case_dir': Path/to/case_dir,
        'config_path': Path/to/config.yaml,
        'log_path': Path/to/case.log,
        'plot_dir': Path/to/plots,
        'raw_data_dir': Path/to/raw_data,
        ...
    }
    """
    case_dir = CASES_DIR / clean_name(case_name)
    return {
        "case_dir": case_dir,
        "config_path": case_dir / "config.yaml",  # The config used for this test
        "log_path": case_dir / "case.log",  # Stdout/stderr from test execution
        "preprocessor_dir": case_dir / "preprocessing_pipeline",  # Fitted preprocessors
        "plot_dir": case_dir / "plots",  # PNG visualizations
        "metadata_path": case_dir / "batch_metadata.json",  # Temporal info
        "raw_data_dir": case_dir / "raw_data",  # NetCDF + summary stats
    }


def prepare_output_directories() -> None:
    """
    Create output directories and clean up old ones.
    Ensures we start fresh and don't have stale files from previous runs.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CASES_DIR.mkdir(parents=True, exist_ok=True)

    # Remove old directories that are no longer used (legacy cleanup)
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
    """
    Convert anything to a JSON string.
    Used for storing complex config dicts as CSV cell values.
    Example: [1980, 1981] -> "[1980, 1981]"
    """
    return json.dumps(value, sort_keys=True, default=str)


def init_csv() -> None:
    """
    Create the CSV file with headers.
    Called once at the start of the test run to set up the results file.
    """
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
    """
    Write one test result row to the CSV file.
    Called after each test to record whether it passed/failed and why.
    """
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
    """
    Clean up config by removing None values where they shouldn't be.
    Some fields should be explicit about being None, others should just be absent.
    This enforces that convention.
    """
    loader_cfg = cfg.setdefault("train_loader", {})
    dataset_cfg = loader_cfg.setdefault("dataset_config", {})

    if "time_features" in dataset_cfg:
        raise ValueError(
            "time_features moved to train_loader level, not dataset_config"
        )

    # Remove None time_features if they exist (just pop it, don't set it)
    loader_cfg.pop("time_features", None) if loader_cfg.get(
        "time_features"
    ) is None else None
    # Remove None condition_method (we'll omit it from the config entirely)
    dataset_cfg.pop("condition_method", None) if dataset_cfg.get(
        "condition_method"
    ) is None else None
    # Remove None lead_times
    dataset_cfg.pop("lead_times", None) if dataset_cfg.get(
        "lead_times"
    ) is None else None

    return cfg


def validate_generated_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Check that the config is valid before using it.
    Catches common mistakes early rather than failing mysteriously during test execution.
    Examples:
    - time_features at wrong nesting level
    - Missing required fields (like model config)
    - Invalid num_data_workers
    """
    train_loader_cfg = cfg.get("train_loader")
    if not isinstance(train_loader_cfg, dict):
        raise ValueError("train_loader must be a mapping")

    dataset_cfg = train_loader_cfg.get("dataset_config")
    if not isinstance(dataset_cfg, dict):
        raise ValueError("train_loader.dataset_config must be a mapping")

    # Runtime-only arguments shouldn't be in saved config
    # (they're added at test time, not in config file)
    runtime_loader_arguments = {"shuffle", "return_metadata", "return_spatial_mask"}
    misplaced = runtime_loader_arguments & set(train_loader_cfg)
    if misplaced:
        raise ValueError(f"Runtime arguments in config: {sorted(misplaced)}")

    # time_features must be at train_loader level, not dataset level
    if "time_features" in dataset_cfg:
        raise ValueError(
            "time_features must be in train_loader, not train_loader.dataset_config"
        )

    # Validate time_features if present
    time_features = train_loader_cfg.get("time_features")
    if time_features is not None:
        if not isinstance(time_features, list):
            raise TypeError("time_features must be a list or null")
        unsupported = set(time_features) - DataCombinationScenarios.VALID_TIME_FEATURES
        if unsupported:
            raise ValueError(f"Unsupported time_features: {sorted(unsupported)}")

    # Model config is always required
    if "model" not in dataset_cfg:
        raise ValueError("dataset_config.model is required")

    # Validate num_data_workers setting
    num_data_workers = train_loader_cfg.get("num_data_workers", 0)
    if (
        isinstance(num_data_workers, bool)
        or not isinstance(num_data_workers, int)
        or num_data_workers < 0
    ):
        raise ValueError("num_data_workers must be a non-negative integer")

    # Can't prefetch if we're using single worker (doesn't make sense)
    if num_data_workers == 0 and train_loader_cfg.get("prefetch_factor") is not None:
        raise ValueError("prefetch_factor must be null when num_data_workers=0")

    return cfg


def write_yaml(cfg: dict[str, Any], output_path: Path) -> Path:
    """
    Write a config dict to a YAML file.
    Saves the generated config so we can review exactly what was tested.
    """
    cfg = deepcopy(cfg)
    normalize_nullable_fields(cfg)
    validate_generated_config(cfg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(cfg, file, sort_keys=False)
    return output_path


def base_config() -> dict[str, Any]:
    """
    Load and prepare the base training config.
    This is a template config we modify for each test case.
    """
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

    # Set up default values for testing (small batch, single worker, quick run)
    train_loader_cfg["batch_size"] = 2  # Small batch for quick testing
    train_loader_cfg["num_data_workers"] = 0  # Single worker by default
    train_loader_cfg["num_validation_years"] = (
        0  # No validation set for integration test
    )
    train_loader_cfg["train_years_slice"] = [1980, 1981]  # Just 2 years for speed
    train_loader_cfg["load"] = False  # Don't load old checkpoints

    # Remove runtime-only arguments (added at test time, not in config)
    for key in ["shuffle", "return_metadata", "return_spatial_mask", "prefetch_factor"]:
        train_loader_cfg.pop(key, None)

    normalize_nullable_fields(cfg)
    validate_generated_config(cfg)
    return cfg


def load_train_config(config_path: Path) -> TrainConfig:
    """
    Load and parse a training config file.
    Returns a TrainConfig dataclass that we can use to build loaders.
    """
    data = prepare_config(config_path)
    if data is None:
        raise ValueError(f"Config is empty: {config_path}")

    normalize_nullable_fields(data)
    validate_generated_config(data)

    # Convert dict to TrainConfig dataclass
    return dacite.from_dict(
        data_class=TrainConfig,
        data=data,
        config=dacite.Config(strict=True),
    )


def replace_data_path(value: Any, new_path: str) -> tuple[Any, int]:
    """
    Recursively find all path-like fields in a dict/list and replace their values.
    Used when we want to point config at a different data source.
    Returns (modified_value, number_of_replacements_made).
    Example: if config has {"data_path": "/old/path"}, returns ({"data_path": "/new/path"}, 1)
    """
    replaced = 0

    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            # Check if key looks like a path field
            if str(key).lower() in PATH_FIELD_NAMES or str(key).lower().endswith(
                ("_path", "_paths", "_directory", "_file", "_files")
            ):
                # Replace with new path (keep as list if it was a list)
                result[key] = [new_path] if isinstance(child, list) else new_path
                replaced += 1
            else:
                # Recurse into child value
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

    # Base case: not a dict or list, return as-is
    return deepcopy(value), 0


def remove_ensemble_configuration(value: Any) -> Any:
    """
    Recursively remove any ensemble-related config fields.
    Useful when testing with data that doesn't have ensemble members.
    Example: if config has {"ensemble_mean": True, "members": 10},
    this removes both fields so the dataloader knows to not expect ensemble dimension.
    """
    if isinstance(value, dict):
        return {
            key: remove_ensemble_configuration(child)
            for key, child in value.items()
            # Skip ensemble-related keys
            if str(key).lower() not in ENSEMBLE_FIELD_NAMES
        }
    if isinstance(value, list):
        return [remove_ensemble_configuration(child) for child in value]
    return deepcopy(value)


def apply_data_combination(
    cfg: dict[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
    """
    Modify the base config to set up a specific data combination scenario.
    This includes setting up model, condition, observation data configs.
    We take a generic base config and customize it for this specific test.
    """
    train_loader_cfg = cfg["train_loader"]
    dataset_cfg = train_loader_cfg["dataset_config"]

    # Model is always required
    if not params.get("model_present", True):
        raise ValueError("Model data is required")

    # Set up model config
    if "model_config_override" in params:
        dataset_cfg["model"] = deepcopy(params["model_config_override"])

    model_cfg = dataset_cfg["model"]
    if "model_ensemble_mean" in params:
        model_cfg["ensemble_mean"] = params["model_ensemble_mean"]

    # Set up condition config (if needed)
    if params.get("condition_present", False):
        if "condition_config_override" in params:
            condition_cfg = deepcopy(params["condition_config_override"])
        elif params.get("condition_path"):
            # Build condition config from external data path
            source_cfg = dataset_cfg.get("condition", model_cfg)
            condition_cfg = build_external_condition_config(
                source_cfg,
                path=params["condition_path"],
                has_ensemble=params.get("condition_has_ensemble", True),
                condition_kind=params.get("condition_kind", "dynamic"),
            )
        else:
            # Use default condition config (usually a copy of model config)
            condition_cfg = deepcopy(dataset_cfg.get("condition", model_cfg))

        if params.get("condition_ensemble_mean") is not None:
            condition_cfg["ensemble_mean"] = params["condition_ensemble_mean"]

        dataset_cfg["condition"] = condition_cfg
    else:
        # Remove condition config if not needed for this test
        dataset_cfg.pop("condition", None)

    # Set up observation config (if needed)
    if params.get("observation_present", False):
        if "observation_config_override" in params:
            dataset_cfg["observation"] = deepcopy(params["observation_config_override"])
        elif "observation" not in dataset_cfg:
            raise ValueError(
                "observation_present=True but no observation in base config"
            )
    else:
        # Remove observation config if not needed
        dataset_cfg.pop("observation", None)

    # Set condition_method (how to sample ensemble members, etc)
    condition_method = params.get("condition_method")
    if condition_method is None:
        dataset_cfg.pop("condition_method", None)  # Remove if None
    else:
        dataset_cfg["condition_method"] = condition_method

    # Set time features (month_sin, month_cos, lead_time, etc)
    if "time_features" in params:
        if params["time_features"] is None:
            train_loader_cfg.pop("time_features", None)
        else:
            train_loader_cfg["time_features"] = deepcopy(params["time_features"])

    # Set year slice for training data
    if "year_slice" in params:
        train_loader_cfg["train_years_slice"] = deepcopy(params["year_slice"])

    # Set lead time range (1-12 months, etc)
    if "lead_months" in params:
        if params["lead_months"] is None:
            dataset_cfg.pop("lead_times", None)
        else:
            dataset_cfg["lead_times"] = deepcopy(params["lead_months"])

    # Set number of parallel data loading workers
    if "num_data_workers" in params:
        num_data_workers = params["num_data_workers"]
        if (
            isinstance(num_data_workers, bool)
            or not isinstance(num_data_workers, int)
            or num_data_workers < 0
        ):
            raise ValueError("num_data_workers must be non-negative integer")
        train_loader_cfg["num_data_workers"] = num_data_workers

    # Handle prefetch_factor (only makes sense with multiple workers)
    num_data_workers = train_loader_cfg.get("num_data_workers", 0)
    if num_data_workers == 0:
        train_loader_cfg.pop("prefetch_factor", None)
    else:
        train_loader_cfg.setdefault("prefetch_factor", 2)

    # Remove runtime-only arguments
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
    """
    Build a condition config for external data (e.g., from a different file path).
    Takes a source config and modifies it to point to external data.
    """
    condition_cfg = deepcopy(source_cfg)
    # Replace data paths with the external path
    condition_cfg, replacements = replace_data_path(condition_cfg, path)

    if replacements == 0:
        raise ValueError("No path-like fields found in condition config")

    # Remove ensemble config if the external data doesn't have ensemble members
    if not has_ensemble:
        condition_cfg = remove_ensemble_configuration(condition_cfg)

    # For static data, remove time-related config (same mask for all times)
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
    """
    Close all active logging handlers to prevent resource leaks.
    Important in multi-task scenarios where we create many loggers.
    """
    logger_names = ["", "training", "inference"]
    # Get all loggers that start with our package name
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
    """
    Run a function and capture all stdout/stderr to a log file.
    This keeps test output from cluttering the console while still preserving it.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cleanup_logging_handlers()

    with log_path.open("w", encoding="utf-8") as log_file:
        try:
            # Redirect stdout and stderr to log file
            with redirect_stdout(log_file), redirect_stderr(log_file):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return function()
        finally:
            cleanup_logging_handlers()


def append_exception_to_log(log_path: Path) -> None:
    """Append the current exception traceback to a log file"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as file:
        traceback.print_exc(file=file)


def configure_runtime_context(config_path: Path) -> Path:
    """
    Set up runtime environment for running a test (sets global experiment directory).
    The RuntimeContext is used by the training code to know where to save artifacts.
    """
    case_dir = Path(config_path).resolve().parent
    case_dir.mkdir(parents=True, exist_ok=True)
    RuntimeContext.GLOBAL_EXP_DIR = str(case_dir)
    return case_dir


def reset_runtime_context() -> None:
    """Clear runtime context after test completes"""
    RuntimeContext.GLOBAL_EXP_DIR = None


def call_with_supported_arguments(function: Callable[..., Any], **kwargs: Any) -> Any:
    """
    Call a function but only pass kwargs that it actually accepts.
    Useful for calling functions with different signatures
    (e.g., some versions have 'distributed', others don't).
    """
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
    """
    Set up and return a data loader for the given config.
    This initializes distributed setup and builds the actual dataloader.
    """
    configure_runtime_context(config_path)
    distributed = Distributed.get_instance()

    # Initialize distributed training setup (if multi-GPU)
    call_with_supported_arguments(
        config.train_loader.setup_distributed,
        distributed=distributed,
    )

    # Build and return the actual loader
    return call_with_supported_arguments(
        config.train_loader.build_train_loader,
        return_metadata=return_metadata,
        return_spatial_mask=return_spatial_mask,
        shuffle=shuffle,
    )


class LimitedLoader:
    """
    Wrapper that limits how many batches we iterate through.
    Saves time during testing by not iterating through entire dataset.
    """

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
        # Delegate unknown attributes to wrapped loader
        return getattr(self._loader, name)


def sample_get(sample: Any, name: str) -> Any:
    """
    Get a field from a sample (works with both dict and object with attributes).
    Some dataloaders return dicts, some return objects with attributes.
    This works with both to be flexible.
    """
    if isinstance(sample, dict):
        return sample.get(name)
    return getattr(sample, name, None)


def extract_metadata(batch: Any) -> Any:
    """Get metadata from a batch"""
    return sample_get(batch, "metadata")


def value_is_batched(value: Any) -> bool:
    """Check if a value is batched (multiple samples) vs scalar"""
    if isinstance(value, str):
        return False
    if isinstance(value, torch.Tensor):
        return value.ndim > 0
    if isinstance(value, np.ndarray):
        return value.ndim > 0
    return isinstance(value, (list, tuple))


def value_length(value: Any) -> int | None:
    """Get the batch size of a value (None if not batched)"""
    if not value_is_batched(value):
        return None
    try:
        return len(value)
    except TypeError:
        return None


def index_value(value: Any, index: int) -> Any:
    """Extract one sample from a batched value"""
    if not value_is_batched(value):
        return value

    indexed = value[index]
    # Convert 0-d tensors/arrays to scalars
    if isinstance(indexed, torch.Tensor) and indexed.ndim == 0:
        return indexed.item()
    if isinstance(indexed, np.ndarray) and indexed.ndim == 0:
        return indexed.item()
    return indexed


def flatten_metadata(metadata: Any) -> list:
    """
    Convert metadata from various formats into a list of dicts.
    If metadata is batched (dict with array values), separate it into per-sample dicts.
    This normalizes different ways the dataloader might return metadata.
    """
    if metadata is None:
        return []
    if isinstance(metadata, list):
        return metadata
    if isinstance(metadata, tuple):
        return list(metadata)
    if isinstance(metadata, dict):
        # Check if any values are batched
        lengths = [
            length
            for value in metadata.values()
            if (length := value_length(value)) is not None
        ]

        if not lengths:
            # No batched values, return as single-item list
            return [metadata]

        # All batched values should have same length
        if len(set(lengths)) != 1:
            raise AssertionError(f"Inconsistent metadata field lengths: {lengths}")

        # Separate batched metadata into per-sample dicts
        count = lengths[0]
        return [
            {key: index_value(value, i) for key, value in metadata.items()}
            for i in range(count)
        ]

    return [metadata]


def scalar_value(value: Any) -> Any:
    """Convert tensors/arrays to Python scalars"""
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
    """
    Recursively convert values to canonical form for comparison.
    Converts tensors to lists, dicts to sorted dicts, etc.
    Useful for comparing whether two samples are the same.
    """
    value = scalar_value(value)

    if isinstance(value, dict):
        # Sort keys for consistent comparison
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
    """
    Generate a canonical key for a sample (used for detecting duplicates).
    Converts to JSON so different representations of same data produce same key.
    """
    if isinstance(item, dict):
        value = item
    elif hasattr(item, "__dict__"):
        value = vars(item)
    else:
        value = item

    return serialize_json(canonical_value(value))


def assert_tensor(name: str, value: Any) -> None:
    """Check that a value is a valid tensor (not empty, all finite values)"""
    if not isinstance(value, torch.Tensor):
        raise AssertionError(f"{name} is not a tensor: {type(value).__name__}")
    if value.numel() == 0:
        raise AssertionError(f"{name} is empty")
    if torch.is_floating_point(value) and not torch.isfinite(value).all():
        raise AssertionError(f"{name} contains non-finite values")


def collect_loader_metadata(loader: Any, max_batches: int | None = None) -> list:
    """Iterate through loader and collect all metadata from all batches"""
    items = []
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        batch_metadata = flatten_metadata(extract_metadata(batch))
        items.extend(batch_metadata)
    return items


def save_metadata(metadata: list[Any], output_path: Path) -> None:
    """Save metadata list to a JSON file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            canonical_value(metadata), file, indent=2, sort_keys=True, default=str
        )


def export_batch_to_netcdf(
    batch: Any,
    *,
    output_path: Path,
    metadata: list[Any],
    batch_index: int,
) -> None:
    """
    Export batch data to NetCDF format.
    """
    try:
        import xarray as xr
    except ImportError:
        print("Warning: xarray not installed, skipping NetCDF export")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract tensors from batch
    input_data = sample_get(batch, "input")
    target_data = sample_get(batch, "target")
    added_features = sample_get(batch, "added_features")

    if input_data is None or target_data is None:
        raise ValueError("Batch missing input or target data")

    # Convert to numpy (remove from GPU if needed)
    input_np = (
        input_data.detach().cpu().numpy()
        if isinstance(input_data, torch.Tensor)
        else np.array(input_data)
    )
    target_np = (
        target_data.detach().cpu().numpy()
        if isinstance(target_data, torch.Tensor)
        else np.array(target_data)
    )

    features_np = None
    if added_features is not None:
        features_np = (
            added_features.detach().cpu().numpy()
            if isinstance(added_features, torch.Tensor)
            else np.array(added_features)
        )

    # Build data variables (what actually gets saved)
    data_vars = {}

    # Input shape: (batch, channels, height, width) or similar
    # We need to figure out what dimensions it has
    if input_np.ndim == 4:
        data_vars["input"] = (["sample", "channel", "x", "y"], input_np)
    elif input_np.ndim == 3:
        data_vars["input"] = (["sample", "channel", "space"], input_np)
    elif input_np.ndim == 2:
        data_vars["input"] = (["sample", "feature"], input_np)
    else:
        data_vars["input"] = (["sample"], input_np)

    # Target shape
    if target_np.ndim == 4:
        data_vars["target"] = (["sample", "channel", "x", "y"], target_np)
    elif target_np.ndim == 3:
        data_vars["target"] = (["sample", "channel", "space"], target_np)
    elif target_np.ndim == 2:
        data_vars["target"] = (["sample", "feature"], target_np)
    else:
        data_vars["target"] = (["sample"], target_np)

    # Added features (time features, etc)
    if features_np is not None:
        if features_np.ndim == 2:
            data_vars["added_features"] = (["sample", "feature"], features_np)
        else:
            data_vars["added_features"] = (["sample"], features_np)

    # Create coordinates from metadata (init_time, lead_time, etc)
    # These become the "index" dimensions that label each sample
    coords = {}
    if metadata:
        # Extract all metadata keys that should become coordinates
        for key in ["init_time", INIT_TIME_DIM, "lead_time", LEAD_TIME_DIM]:
            values = [
                (item.get(key) if isinstance(item, dict) else getattr(item, key, None))
                for item in metadata
            ]
            if values and any(v is not None for v in values):
                # Convert to string for NetCDF compatibility
                coords[key] = ("sample", [str(v) for v in values])

    # Create xarray Dataset (this is the main data container)
    ds = xr.Dataset(data_vars=data_vars, coords=coords)

    # Add attributes (metadata about the dataset - not dimension-specific)
    ds.attrs["batch_index"] = batch_index
    ds.attrs["batch_size"] = len(metadata) if metadata else input_np.shape[0]
    ds.attrs["description"] = (
        "Batch export from data_modules integration testing. "
        "Input: model/condition data. Target: observation data. "
        "Open with: import xarray as xr; ds = xr.open_dataset(filename)"
    )

    # Save to NetCDF (try netcdf4 engine first, fallback to scipy)
    try:
        ds.to_netcdf(output_path, engine="netcdf4", unlimited_dims=["sample"])
    except Exception:
        try:
            ds.to_netcdf(output_path, engine="scipy")
        except Exception as exc:
            print(f"Failed to save NetCDF: {exc}")


def create_data_summary_report(
    batch: Any,
    *,
    output_path: Path,
    metadata: list[Any],
) -> None:
    """
    Create human-readable summary of batch statistics.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        file.write("=" * 88 + "\n")
        file.write("DATA BATCH SUMMARY STATISTICS\n")
        file.write("=" * 88 + "\n\n")

        # INPUT STATISTICS
        input_data = sample_get(batch, "input")
        if input_data is not None:
            inp = (
                input_data.detach().cpu().numpy()
                if isinstance(input_data, torch.Tensor)
                else np.array(input_data)
            )
            file.write("INPUT DATA\n")
            file.write("-" * 88 + "\n")
            file.write(f"  Shape: {inp.shape}\n")
            file.write(f"  Min:   {float(np.nanmin(inp)):.6f}\n")
            file.write(f"  Max:   {float(np.nanmax(inp)):.6f}\n")
            file.write(f"  Mean:  {float(np.nanmean(inp)):.6f}\n")
            file.write(f"  Std:   {float(np.nanstd(inp)):.6f}\n")
            file.write(f"  NaN count: {int(np.isnan(inp).sum())} / {inp.size}\n")
            file.write(f"  Inf count: {int(np.isinf(inp).sum())} / {inp.size}\n")
            file.write("\n")

        # TARGET STATISTICS
        target_data = sample_get(batch, "target")
        if target_data is not None:
            tgt = (
                target_data.detach().cpu().numpy()
                if isinstance(target_data, torch.Tensor)
                else np.array(target_data)
            )
            file.write("TARGET DATA\n")
            file.write("-" * 88 + "\n")
            file.write(f"  Shape: {tgt.shape}\n")
            file.write(f"  Min:   {float(np.nanmin(tgt)):.6f}\n")
            file.write(f"  Max:   {float(np.nanmax(tgt)):.6f}\n")
            file.write(f"  Mean:  {float(np.nanmean(tgt)):.6f}\n")
            file.write(f"  Std:   {float(np.nanstd(tgt)):.6f}\n")
            file.write(f"  NaN count: {int(np.isnan(tgt).sum())} / {tgt.size}\n")
            file.write(f"  Inf count: {int(np.isinf(tgt).sum())} / {tgt.size}\n")
            file.write("\n")

        # INPUT vs TARGET COMPARISON
        # (useful for bias correction checks - is model consistently too high/low?)
        if input_data is not None and target_data is not None:
            file.write("INPUT vs TARGET COMPARISON\n")
            file.write("-" * 88 + "\n")
            input_mean = float(np.nanmean(inp))
            target_mean = float(np.nanmean(tgt))
            file.write(f"  Input mean:  {input_mean:.6f}\n")
            file.write(f"  Target mean: {target_mean:.6f}\n")
            file.write(f"  Difference:  {abs(input_mean - target_mean):.6f}\n")
            file.write(f"  Bias (input - target): {(input_mean - target_mean):.6f}\n")
            file.write("\n")

        # METADATA SUMMARY
        if metadata:
            file.write("METADATA SUMMARY\n")
            file.write("-" * 88 + "\n")
            file.write(f"  Samples: {len(metadata)}\n")

            # Extract init_times and lead_times if present
            init_times = [
                (
                    item.get(INIT_TIME_DIM)
                    if isinstance(item, dict)
                    else getattr(item, INIT_TIME_DIM, None)
                )
                for item in metadata
            ]
            lead_times = [
                (
                    item.get(LEAD_TIME_DIM)
                    if isinstance(item, dict)
                    else getattr(item, LEAD_TIME_DIM, None)
                )
                for item in metadata
            ]

            if any(t is not None for t in init_times):
                init_times_clean = [t for t in init_times if t is not None]
                file.write(f"  Init times (first 5): {init_times_clean[:5]}\n")

            if any(t is not None for t in lead_times):
                lead_times_clean = [t for t in lead_times if t is not None]
                file.write(f"  Lead times (first 5): {lead_times_clean[:5]}\n")

            file.write("\n")

        file.write("CLIMATE SANITY CHECKS\n")
        file.write("-" * 88 + "\n")

        if input_data is not None:
            inp_min = float(np.nanmin(inp))
            inp_max = float(np.nanmax(inp))

            # Temperature range check (assuming Celsius)
            if -60 <= inp_min <= 60 and -60 <= inp_max <= 60:
                file.write("  Input range looks reasonable for temperature\n")
            else:
                file.write(
                    f"  ⚠ WARNING: Input range [{inp_min:.1f}, {inp_max:.1f}] "
                    "may be unrealistic for temperature (-60°C to 60°C expected)\n"
                )

            # Variance check (constant data is suspicious)
            inp_std = float(np.nanstd(inp))
            if inp_std < 0.001:
                file.write(
                    f"  ⚠ WARNING: Input has very low variance (std={inp_std:.6f}), "
                    "almost constant value\n"
                )
            else:
                file.write(f"  Input has reasonable variance (std={inp_std:.6f})\n")

            # NaN check (missing data)
            nan_count = np.isnan(inp).sum()
            nan_pct = 100.0 * nan_count / inp.size
            if nan_pct > 0:
                file.write(
                    f"  ⚠ WARNING: {nan_pct:.2f}% NaN values in input "
                    f"({int(nan_count)} / {inp.size})\n"
                )
            else:
                file.write("  No NaN values in input\n")

            # Inf check (exploding values)
            inf_count = np.isinf(inp).sum()
            if inf_count > 0:
                file.write(
                    f"  ⚠ ERROR: {int(inf_count)} Inf values in input! "
                    "This is a serious data quality issue.\n"
                )
            else:
                file.write("  No Inf values in input\n")

        file.write("\n" + "=" * 88 + "\n")


def channel_axis(tensor: torch.Tensor) -> int:
    """Figure out which axis is the channel axis (0 or 1)"""
    return 0 if tensor.ndim < 2 else 1


def first_channel_image(tensor: torch.Tensor, channel_index: int) -> np.ndarray:
    """
    Extract a 2D image from a tensor for a specific channel.
    Handles various tensor shapes (batches, spatial dims, etc).
    """
    value = tensor.detach().float().cpu()

    # Remove batch dimension if present (4D+ tensor)
    if value.ndim >= 4:
        value = value[0, channel_index]
    elif value.ndim == 3:
        value = value[channel_index]
    elif value.ndim == 2:
        value = value[channel_index].unsqueeze(0)
    else:
        value = value.reshape(1, -1)

    # Keep squeezing until we have a 2D image
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
    """
    Create plots of the first few channels of a tensor.
    Useful for visually inspecting what data the model is seeing.
    """

    if not RUN_CHANNEL_PLOTS:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    axis = channel_axis(tensor)

    # Skip if tensor is too low dimensional
    if tensor.ndim <= axis:
        return

    # Plot up to MAX_PLOT_CHANNELS channels
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

        # Use metadata as x-axis label (truncated for readability)
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
    """
    Create plots of input, target, and other fields from the first batch.
    Good for sanity-checking what data looks like.
    """
    metadata = flatten_metadata(extract_metadata(batch))
    first_metadata = metadata[0] if metadata else {}

    # Plot these fields if they exist in the batch
    for name in ("input", "target", "added_features", "input_mask", "target_mask"):
        tensor = sample_get(batch, name)
        if isinstance(tensor, torch.Tensor):
            plot_tensor_channels(
                tensor, name=name, output_dir=output_dir, metadata=first_metadata
            )


def metadata_value(item: Any, names: tuple[str, ...]) -> Any:
    """
    Try to get a value from metadata using multiple possible key names.
    Works with both dicts and objects with attributes.
    Useful because different dataloaders might use different field names.
    """
    for name in names:
        if isinstance(item, dict) and name in item:
            return item[name]
        if hasattr(item, name):
            return getattr(item, name)
    return None


def metadata_init_time(item: Any) -> Any:
    """Get the initialization time from metadata"""
    return metadata_value(item, (INIT_TIME_DIM, "init_time", "initialization_time"))


def metadata_lead_time(item: Any) -> Any:
    """Get the lead time from metadata"""
    return metadata_value(item, (LEAD_TIME_DIM, "lead_time"))


def to_datetime64_month(value: Any) -> np.datetime64:
    """Convert a value to numpy datetime64 (month precision)"""
    value = scalar_value(value)
    return np.datetime64(value, "M")


def to_integer_lead(value: Any) -> int:
    """Convert a lead time value to an integer (in months)"""
    value = scalar_value(value)
    if isinstance(value, np.timedelta64):
        return int(value.astype("timedelta64[M]").astype(int))
    return int(value)


def valid_time_from_metadata(item: Any) -> np.datetime64:
    """
    Calculate the valid time (when prediction is valid) from init_time and lead_time.
    valid_time = init_time + lead_time (in months)
    Example: if init_time is 2020-01 and lead_time is 3 months,
    valid_time is 2020-04 (April 2020)
    """
    init_time = metadata_init_time(item)
    lead_time = metadata_lead_time(item)

    if init_time is None or lead_time is None:
        raise AssertionError("Metadata missing init_time or lead_time")

    lead = to_integer_lead(lead_time)
    if lead < 1:
        raise AssertionError(f"Lead time must be 1-based, got {lead}")

    return to_datetime64_month(init_time) + np.timedelta64(lead - 1, "M")


def assert_metadata_indexing(metadata: list[Any]) -> None:
    """
    Check that metadata is valid:
    - Has init_time and lead_time
    - No duplicate samples
    - Lead times are 1-based (start at 1, not 0)
    - Valid times are >= init times
    """
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

    # Check for duplicates
    if len(keys) != len(set(keys)):
        counts = Counter(keys)
        duplicates = {key: count for key, count in counts.items() if count > 1}
        raise AssertionError(f"Duplicate samples in metadata: {duplicates}")


def metadata_year(item: Any) -> int:
    """Extract the year from an item's init_time"""
    value = metadata_init_time(item)
    if value is None:
        raise AssertionError("Could not extract init_time")

    value = scalar_value(value)
    if hasattr(value, "year"):
        return int(value.year)

    # Handle numpy datetime64
    return int(np.datetime64(value, "Y").astype(int) + 1970)


def assert_dataset_getitem_contract(dataset: Any, params: dict[str, Any]) -> None:
    """
    Test that dataset.__getitem__ returns proper format:
    - Returns tuple of (sample_dict, metadata_dict)
    - sample_dict has "input" and "target" tensors
    - metadata_dict has required fields

    We test a few indices (start, middle, end) to catch edge cases.
    """
    if len(dataset) <= 0:
        raise AssertionError("Dataset is empty")

    # Test a few indices (start, middle, end)
    indices = sorted({0, len(dataset) // 2, len(dataset) - 1})
    metadata_keys = []

    for index in indices:
        sample = dataset[index]

        # Should return 2-tuple
        if not isinstance(sample, tuple):
            raise AssertionError(
                f"__getitem__ must return tuple, got {type(sample).__name__}"
            )
        if len(sample) != 2:
            raise AssertionError(f"__getitem__ must return 2 values, got {len(sample)}")

        sample_data, metadata = sample

        # sample_data should be a dict
        if not isinstance(sample_data, dict):
            raise AssertionError(
                f"Sample data must be dict, got {type(sample_data).__name__}"
            )
        if not isinstance(metadata, dict):
            raise AssertionError(
                f"Metadata must be dict, got {type(metadata).__name__}"
            )

        # Must have input and target tensors
        required_fields = {"input", "target"}
        missing = required_fields - set(sample_data)
        if missing:
            raise AssertionError(f"Missing sample fields: {sorted(missing)}")

        assert_tensor(f"sample[{index}].input", sample_data["input"])
        assert_tensor(f"sample[{index}].target", sample_data["target"])

        # If time_features are configured, added_features should be populated
        features = params.get(
            "time_features", DataCombinationScenarios.DEFAULT_TIME_FEATURES
        )
        added_features = sample_data.get("added_features")

        if features:
            assert_tensor(f"sample[{index}].added_features", added_features)
        elif isinstance(added_features, torch.Tensor) and added_features.numel() > 0:
            raise AssertionError("added_features populated but time_features is empty")

        # Must have required metadata
        required_meta = {INIT_TIME_DIM, LEAD_TIME_DIM}
        missing_meta = required_meta - set(metadata)
        if missing_meta:
            raise AssertionError(f"Missing metadata fields: {sorted(missing_meta)}")

        metadata_keys.append(canonical_sample_key(metadata))

    # Check for duplicate samples at test indices
    if len(metadata_keys) != len(set(metadata_keys)):
        raise AssertionError(f"Duplicate metadata at indices {indices}")

    # Check that config has expected fields
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
    raw_data_dir: Path | None = None,
) -> list:
    """
    Test that dataloader returns batches with proper format:
    - Tensors are non-empty and finite
    - Batch size matches metadata count
    - Time features match config
    """
    batches = 0
    all_metadata = []

    for batch in LimitedLoader(loader, MAX_BATCHES_PER_CASE):
        input_tensor = sample_get(batch, "input")
        target_tensor = sample_get(batch, "target")
        added_features = sample_get(batch, "added_features")

        # Check input tensor
        assert_tensor("batch.input", input_tensor)

        if target_tensor is not None:
            assert_tensor("batch.target", target_tensor)

        # Check added_features based on time_features config
        if time_features:
            assert_tensor("batch.added_features", added_features)
        elif isinstance(added_features, torch.Tensor) and added_features.numel() > 0:
            raise AssertionError("added_features populated but time_features is empty")

        # Extract per-sample metadata from batch
        metadata = flatten_metadata(extract_metadata(batch))
        if not metadata:
            raise AssertionError("No metadata returned")

        # Metadata count should match batch size
        batch_size = int(input_tensor.shape[0])
        if len(metadata) != batch_size:
            raise AssertionError(
                f"Metadata count ({len(metadata)}) != batch size ({batch_size})"
            )

        all_metadata.extend(metadata)

        # Plot first batch for visual inspection
        if batches == 0 and plot_dir is not None:
            plot_first_batch(batch, output_dir=plot_dir)

        if batches == 0 and raw_data_dir is not None and RUN_RAW_DATA_EXPORT:
            raw_data_dir.mkdir(parents=True, exist_ok=True)

            # Export to NetCDF
            netcdf_path = raw_data_dir / f"batch_{batches:04d}.nc"
            try:
                export_batch_to_netcdf(
                    batch,
                    output_path=netcdf_path,
                    metadata=metadata,
                    batch_index=batches,
                )
            except Exception as exc:
                print(f"Warning: NetCDF export failed: {exc}")

            # Create summary report for manual inspection
            summary_path = raw_data_dir / f"batch_{batches:04d}_summary.txt"
            try:
                create_data_summary_report(
                    batch,
                    output_path=summary_path,
                    metadata=metadata,
                )
            except Exception as exc:
                print(f"Warning: Summary report failed: {exc}")

        batches += 1

    # Should have iterated through at least one batch
    if batches == 0:
        raise AssertionError("Dataloader returned no batches")

    # Save metadata to file for analysis
    if metadata_path is not None:
        save_metadata(all_metadata, metadata_path)

    return all_metadata


def assert_year_slice(loader: Any, year_slice: list[int]) -> None:
    """
    Check that all years in the loader are within the specified range.
    Also check that all years within the range are present (no gaps).
    """
    metadata = collect_loader_metadata(loader, max_batches=None)

    if not metadata:
        raise AssertionError("No metadata for year-slice validation")

    assert_metadata_indexing(metadata)

    lower, upper = year_slice
    years = [metadata_year(item) for item in metadata]
    observed_years = set(years)

    # Check bounds
    if min(years) < lower or max(years) > upper:
        raise AssertionError(
            f"Years outside [{lower}, {upper}]: {sorted(observed_years)}"
        )

    # Check for gaps
    expected_years = set(range(lower, upper + 1))
    missing_years = expected_years - observed_years

    if missing_years:
        raise AssertionError(f"Missing years: {sorted(missing_years)}")


def assert_no_future_leakage(loader: Any, cutoff: Any) -> None:
    """
    Check that no samples look beyond a cutoff date.
    Important to prevent the model from accidentally seeing future data during training.
    """
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
    """Collect all unique valid times from the loader (sorted)"""
    metadata = collect_loader_metadata(loader, max_batches=None)
    return sorted({valid_time_from_metadata(item) for item in metadata})


def cutoff_for_position(
    valid_times: list[np.datetime64], position: str
) -> np.datetime64:
    """
    Calculate a cutoff date based on position in the valid times.
    "before": one month before first valid time
    "between": halfway through valid times
    "after": one month after last valid time
    """
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
    """
    Check that observation cutoff filtering works correctly.
    Samples with valid_time > cutoff should be excluded.
    """
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
    """Set random seed for PyTorch (for reproducibility)"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def assert_multiworker_consistency(
    config: TrainConfig, shuffle: bool, config_path: Path
) -> None:
    """
    Check that multi-worker dataloading gives same coverage as single-worker.
    If shuffle=False, order should be identical.
    If shuffle=True, items should be the same but order should differ.
    """
    configure_runtime_context(config_path)
    distributed = Distributed.get_instance()

    # Create single-worker loader
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

    # Create multi-worker loader
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

    # Convert to canonical keys for comparison
    single_keys = [canonical_sample_key(item) for item in single_items]
    multi_keys = [canonical_sample_key(item) for item in multi_items]

    # Same items should be present
    if Counter(multi_keys) != Counter(single_keys):
        raise AssertionError(
            f"Multi-worker changed coverage: single={len(single_keys)}, multi={len(multi_keys)}"
        )

    # No duplicates
    if len(multi_keys) != len(set(multi_keys)):
        raise AssertionError("Multi-worker duplicated samples")

    # If no shuffle, order should be identical
    if not shuffle and multi_keys != single_keys:
        raise AssertionError("No-shuffle multi-worker changed order")

    # If shuffling, order should be different (very high probability)
    if shuffle and len(multi_keys) > 1 and multi_keys == single_keys:
        raise AssertionError("Shuffle had no effect")


def preprocessor_files(directory: Path) -> list:
    """Find all preprocessor pipeline files in a directory"""
    return sorted(directory.glob("*_preprocessing_pipeline.joblib"))


def assert_rank_zero_preprocessor_setup(config: TrainConfig, config_path: Path) -> None:
    """
    Test that rank 0 creates preprocessors and other ranks load them.
    This validates distributed preprocessing (important for multi-GPU training).
    """
    case_dir = configure_runtime_context(config_path)
    preprocessor_dir = case_dir / "preprocessing_pipeline"

    if preprocessor_dir.exists():
        shutil.rmtree(preprocessor_dir)

    distributed = Distributed.get_instance()

    # Non-rank-0 processes should load from preprocessor_dir (after rank 0 creates it)
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

    # Rank 0: first run to create preprocessors
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

    # Record file hashes before reload
    before_hashes = {
        path.name: (path.stat().st_size, path.stat().st_mtime_ns) for path in written
    }

    # Second run: load existing preprocessors (shouldn't rewrite them)
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

    # Files shouldn't have changed
    if after_hashes != before_hashes:
        raise AssertionError("Reload phase rewrote artifacts")

    if getattr(distributed, "world_size", 1) == 1:
        print("Note: Run with torchrun --nproc_per_node=2 for multi-rank validation")


def run_dataset_case(config_path: Path, params: dict[str, Any]) -> None:
    """Test dataset.__getitem__ and batch contract"""
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
        raw_data_dir=paths["raw_data_dir"],
    )

    assert_metadata_indexing(metadata)


def run_dataloader_case(config_path: Path, params: dict[str, Any]) -> None:
    """Test dataloader with different configurations"""
    config = load_train_config(config_path)
    check = params["check"]

    # Special case: test multi-worker consistency
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
        raw_data_dir=paths["raw_data_dir"],
    )

    assert_metadata_indexing(metadata)

    # If testing year slice, create fresh loader and validate
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
    """Test that no samples look beyond a cutoff date"""
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
    """Test observation cutoff filtering"""
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
    """Test distributed preprocessor setup"""
    config = load_train_config(config_path)
    assert_rank_zero_preprocessor_setup(config, config_path)


def run_inference_case(config_path: Path, params: dict[str, Any]) -> None:
    """Test inference loader compatibility with training config"""
    from cccma_ppp.inference.inference_configs import InferenceConfig

    configure_runtime_context(config_path)
    train_config = load_train_config(config_path)

    # Load inference config
    inference_data = prepare_config(params["inference_config_path"])
    if inference_data is None:
        raise ValueError("Inference config is empty")

    inference_config = dacite.from_dict(
        data_class=InferenceConfig,
        data=inference_data,
        config=dacite.Config(strict=True),
    )

    # Copy training config into inference loader config
    inference_loader_cfg = inference_config.inference_loader
    inference_loader_cfg.read_configs_from_train(train_config.train_loader)

    # Check that time_features are preserved
    if inference_loader_cfg.time_features != train_config.train_loader.time_features:
        raise AssertionError("Inference did not preserve training time_features")

    distributed = Distributed.get_instance()

    # Set up inference loader (with proper arguments)
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

    # Build and test inference loader
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
        raw_data_dir=paths["raw_data_dir"],
    )

    assert_metadata_indexing(metadata)

    # Check for future leakage if cutoff specified
    if "cutoff" in params:
        assert_no_future_leakage(loader, cutoff=params["cutoff"])


def build_case_config(params: dict[str, Any]) -> dict[str, Any]:
    """Build a full config from base config + params"""
    cfg = base_config()
    apply_data_combination(cfg, params)
    return cfg


def task_defaults(case: dict[str, Any]) -> dict[str, Any]:
    """Add default values to a case dict"""
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
    """Add a task to the test suite"""
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
    """
    Run a single test task.
    Returns (passed: bool, error_message: str).
    """
    paths = case_paths(task["case_name"])
    case_dir = paths["case_dir"]

    # Clean up old case directory
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

        # Build config
        cfg = cfg_builder()
        if not isinstance(cfg, dict):
            raise TypeError(f"cfg_builder returned {type(cfg).__name__}, expected dict")

        # Write config to file
        write_yaml(cfg, paths["config_path"])
        configure_runtime_context(paths["config_path"])

        print(f"\nStarting {task['suite']}:{task['case_name']}", flush=True)
        print(f"Case directory: {case_dir}", flush=True)
        print(f"Log: {paths['log_path']}", flush=True)

        # Run the actual test (capture output to log file)
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

        # Clean up resources
        try:
            import gc

            import xarray as xr

            # Clear xarray's file manager cache
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

        # Force garbage collection
        gc.collect()
        gc.collect()
        gc.collect()

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Log this result to CSV
    log_result(task=task, paths=paths, passed=passed, error=error)
    return passed, error


def validate_tasks(tasks: list[dict[str, Any]]) -> None:
    """Check that all tasks are valid"""
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
    """
    Classify a test result as:
    - unexpected_failure: test failed but was expected to pass
    - expected_failure: test failed and was expected to fail
    - unexpected_pass: test passed but was expected to fail
    """
    params = task["params"]
    expected = params.get("expected_result", "PASS")

    if expected == "FAIL":
        # Test was expected to fail
        if passed:
            # But it passed (unexpected!)
            unexpected_passes.append(task["case_name"])
            return

        # It failed as expected, but check if error matches pattern
        pattern = params.get("expected_error_pattern")
        if pattern and not re.search(pattern, error, re.IGNORECASE | re.DOTALL):
            unexpected_failures.append(f"{task['case_name']}: wrong error: {error}")
        else:
            expected_failures.append(task["case_name"])
        return

    # Test was expected to pass
    if not passed:
        unexpected_failures.append(f"{task['case_name']}: {error}")


def write_summary(
    tasks: list[dict[str, Any]],
    unexpected_failures: list[str],
    expected_failures: list[str],
    unexpected_passes: list[str],
) -> None:
    """Write summary report to file"""
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
    """
    Build the entire test suite by creating all test cases.
    Returns a list of task dicts that can be executed.
    """
    tasks = []

    # Test 1: Data combination scenarios
    if RUN_DATA_CONFIG_COMBINATIONS:
        for case in DataCombinationScenarios.all():
            params = task_defaults(case)
            add_task(
                tasks,
                suite="data_config_combination",
                params=params,
                runner=run_dataset_case,
            )

    # Test 2: External condition scenarios
    if RUN_EXTERNAL_CONDITION_CASES:
        for case in ExternalConditionScenarios.all():
            condition_path = Path(case.get("condition_path", ""))
            params = task_defaults(case)

            # Mark as failure if path doesn't exist
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

    # Test 3: Dataset __getitem__ contract
    if RUN_DATASET_GETITEM_CASES:
        for case in DataCombinationScenarios.all():
            params = task_defaults(case)
            params["name"] = f"getitem_{case['name']}"
            add_task(
                tasks, suite="dataset_getitem", params=params, runner=run_dataset_case
            )

    # Test 4: Dataloader scenarios
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

    # Test 5: Future leakage (no data after cutoff)
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

    # Test 6: Observation cutoff (filter by available obs)
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

    # Test 7: Distributed preprocessing
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

    # Test 8: Inference dataloader
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
    """Main entry point - run the entire integration test suite"""
    prepare_output_directories()
    cleanup_logging_handlers()
    init_csv()

    # Build all test cases
    tasks = build_test_suite()
    validate_tasks(tasks)

    # Optionally limit number of cases
    if MAX_CASES is not None:
        tasks = tasks[:MAX_CASES]

    unexpected_failures = []
    expected_failures = []
    unexpected_passes = []

    try:
        # Run tasks with progress bar
        with tqdm(
            total=len(tasks),
            desc="data_modules",
            unit="case",
            dynamic_ncols=True,
            leave=True,
        ) as progress:
            for task in tasks:
                progress.set_postfix_str(f"{task['suite']}:{task['case_name'][:60]}")

                # Run the task
                passed, error = run_task(task)

                # Classify result
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

    # Write results summary
    write_summary(tasks, unexpected_failures, expected_failures, unexpected_passes)

    # Check for problems and exit
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

    print("Integration test suite completed successfully")
    print(f"  Cases: {CASES_DIR}")
    print(f"  Results: {RESULTS_CSV}")
    print(f"  Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
