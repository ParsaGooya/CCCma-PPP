"""
Data Modules Integration Test Suite (Hardcoded Scenarios)

Improvements over original:
1. Rich metadata logging: Per-sample metadata + temporal statistics
2. All 4 condition methods: ensemble_mean, cross_ensemble, same_member, static
3. Better __getitem__ testing: Tuple format validation, mask analysis
4. Separate mask testing: Binary validation separate from channel visualization
"""

from __future__ import annotations

import csv
import inspect
import json
import logging
import re
import shutil
import traceback
import warnings
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


from cccma_ppp.configs import required_sample_dimensions
from cccma_ppp.generic.distributed import Distributed
from cccma_ppp.generic.runtime import RuntimeContext
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
RUN_RAW_DATA_EXPORT = True

MAX_CASES: int | None = None
MAX_BATCHES_PER_CASE = 4
MAX_PLOT_CHANNELS = 16
SHUFFLE_SEED = 1847

CSV_COLUMNS = [
    "timestamp",
    "suite",
    "case_name",
    "description",
    "model_present",
    "condition_present",
    "observation_present",
    "condition_kind",
    "condition_method",
    "time_features",
    "year_slice",
    "num_data_workers",
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


class DataCombinationScenarios:
    """Hardcoded scenarios for data combinations with ALL 4 condition methods"""

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

    # Scenario 1: Autoencode model only - all 4 methods
    AUTOENCODE_MODEL_ONLY_ENSEMBLE_MEAN = {
        "name": "autoencode_model_only_ensemble_mean",
        "model_present": True,
        "condition_present": False,
        "observation_present": False,
        "condition_method": "ensemble_mean",
        "model_ensemble_mean": True,
        "expected_result": "PASS",
    }

    AUTOENCODE_MODEL_ONLY_CROSS_ENSEMBLE = {
        "name": "autoencode_model_only_cross_ensemble",
        "model_present": True,
        "condition_present": False,
        "observation_present": False,
        "condition_method": "cross_ensemble",
        "model_ensemble_mean": False,
        "expected_result": "PASS",
    }

    AUTOENCODE_MODEL_ONLY_SAME_MEMBER = {
        "name": "autoencode_model_only_same_member",
        "model_present": True,
        "condition_present": False,
        "observation_present": False,
        "condition_method": "same_member",
        "model_ensemble_mean": False,
        "expected_result": "PASS",
    }

    # Scenario 2: Model with condition - all 4 methods
    AUTOENCODE_WITH_CONDITION_ENSEMBLE_MEAN = {
        "name": "autoencode_with_condition_ensemble_mean",
        "model_present": True,
        "condition_present": True,
        "observation_present": False,
        "condition_method": "ensemble_mean",
        "model_ensemble_mean": True,
        "condition_ensemble_mean": True,
        "expected_result": "PASS",
    }

    AUTOENCODE_WITH_CONDITION_CROSS_ENSEMBLE = {
        "name": "autoencode_with_condition_cross_ensemble",
        "model_present": True,
        "condition_present": True,
        "observation_present": False,
        "condition_method": "cross_ensemble",
        "model_ensemble_mean": False,
        "condition_ensemble_mean": False,
        "expected_result": "PASS",
    }

    AUTOENCODE_WITH_CONDITION_SAME_MEMBER = {
        "name": "autoencode_with_condition_same_member",
        "model_present": True,
        "condition_present": True,
        "observation_present": False,
        "condition_method": "same_member",
        "model_ensemble_mean": False,
        "condition_ensemble_mean": False,
        "expected_result": "PASS",
    }

    # Scenario 3: Bias correct model + condition - all 4 methods
    BIAS_CORRECT_MODEL_CONDITION_ENSEMBLE_MEAN = {
        "name": "bias_correct_model_condition_ensemble_mean",
        "model_present": True,
        "condition_present": True,
        "observation_present": True,
        "condition_method": "ensemble_mean",
        "model_ensemble_mean": False,
        "condition_ensemble_mean": False,
        "expected_result": "PASS",
    }

    BIAS_CORRECT_MODEL_CONDITION_CROSS_ENSEMBLE = {
        "name": "bias_correct_model_condition_cross_ensemble",
        "model_present": True,
        "condition_present": True,
        "observation_present": True,
        "condition_method": "cross_ensemble",
        "model_ensemble_mean": False,
        "condition_ensemble_mean": False,
        "expected_result": "PASS",
    }

    BIAS_CORRECT_MODEL_CONDITION_SAME_MEMBER = {
        "name": "bias_correct_model_condition_same_member",
        "model_present": True,
        "condition_present": True,
        "observation_present": True,
        "condition_method": "same_member",
        "model_ensemble_mean": False,
        "condition_ensemble_mean": False,
        "expected_result": "PASS",
    }

    BIAS_CORRECT_MODEL_CONDITION_STATIC = {
        "name": "bias_correct_model_condition_static",
        "model_present": True,
        "condition_present": True,
        "observation_present": True,
        "condition_method": "static",
        "model_ensemble_mean": False,
        "condition_ensemble_mean": None,
        "expected_result": "PASS",
    }

    # Scenario 4: Bias correct model only
    BIAS_CORRECT_MODEL_WITH_METHOD = {
        "name": "bias_correct_model_with_method",
        "model_present": True,
        "condition_present": False,
        "observation_present": True,
        "condition_method": "ensemble_mean",
        "model_ensemble_mean": True,
        "expected_result": "PASS",
    }

    BIAS_CORRECT_MODEL_DIRECT = {
        "name": "bias_correct_model_direct",
        "model_present": True,
        "condition_present": False,
        "observation_present": True,
        "condition_method": None,
        "model_ensemble_mean": False,
        "expected_result": "PASS",
    }

    @classmethod
    def all(cls) -> list[dict[str, Any]]:
        """Return all scenarios"""
        return [
            cls.AUTOENCODE_MODEL_ONLY_ENSEMBLE_MEAN,
            cls.AUTOENCODE_MODEL_ONLY_CROSS_ENSEMBLE,
            cls.AUTOENCODE_MODEL_ONLY_SAME_MEMBER,
            cls.AUTOENCODE_WITH_CONDITION_ENSEMBLE_MEAN,
            cls.AUTOENCODE_WITH_CONDITION_CROSS_ENSEMBLE,
            cls.AUTOENCODE_WITH_CONDITION_SAME_MEMBER,
            cls.BIAS_CORRECT_MODEL_CONDITION_ENSEMBLE_MEAN,
            cls.BIAS_CORRECT_MODEL_CONDITION_CROSS_ENSEMBLE,
            cls.BIAS_CORRECT_MODEL_CONDITION_SAME_MEMBER,
            cls.BIAS_CORRECT_MODEL_CONDITION_STATIC,
            cls.BIAS_CORRECT_MODEL_WITH_METHOD,
            cls.BIAS_CORRECT_MODEL_DIRECT,
        ]


class ExternalConditionScenarios:
    """Hardcoded external condition scenarios with all 4 methods"""

    TAS_STDR_ENSEMBLE_MEAN = {
        "name": "bias_correct_with_tas_stdr_ensemble_mean",
        "model_present": True,
        "condition_present": True,
        "observation_present": True,
        "condition_kind": "tas_stdr",
        "condition_path": str(TAS_STDR_PATH),
        "condition_method": "ensemble_mean",
        "model_ensemble_mean": False,
        "condition_ensemble_mean": None,
        "condition_has_ensemble": False,
        "expected_result": "PASS",
    }

    TAS_STDR_CROSS_ENSEMBLE = {
        "name": "bias_correct_with_tas_stdr_cross_ensemble",
        "model_present": True,
        "condition_present": True,
        "observation_present": True,
        "condition_kind": "tas_stdr",
        "condition_path": str(TAS_STDR_PATH),
        "condition_method": "cross_ensemble",
        "model_ensemble_mean": False,
        "condition_ensemble_mean": None,
        "condition_has_ensemble": False,
        "expected_result": "PASS",
    }

    TAS_STDR_SAME_MEMBER = {
        "name": "bias_correct_with_tas_stdr_same_member",
        "model_present": True,
        "condition_present": True,
        "observation_present": True,
        "condition_kind": "tas_stdr",
        "condition_path": str(TAS_STDR_PATH),
        "condition_method": "same_member",
        "model_ensemble_mean": False,
        "condition_ensemble_mean": None,
        "condition_has_ensemble": False,
        "expected_result": "PASS",
    }

    STATIC_MASK_CONDITION = {
        "name": "bias_correct_with_static_dummy_mask",
        "model_present": True,
        "condition_present": True,
        "observation_present": True,
        "condition_kind": "static",
        "condition_path": str(DUMMY_MASK_PATH),
        "condition_method": "static",
        "model_ensemble_mean": False,
        "condition_ensemble_mean": None,
        "condition_has_ensemble": False,
        "expected_result": "PASS",
    }

    @classmethod
    def all(cls) -> list[dict[str, Any]]:
        """Return all external condition scenarios"""
        return [
            cls.TAS_STDR_ENSEMBLE_MEAN,
            cls.TAS_STDR_CROSS_ENSEMBLE,
            cls.TAS_STDR_SAME_MEMBER,
            cls.STATIC_MASK_CONDITION,
        ]


class DataloaderScenarios:
    """Hardcoded dataloader scenarios"""

    YEAR_SLICE_ONLY = {
        "name": "train_year_slice",
        "year_slice": [1980, 1981],
        "num_data_workers": 0,
        "shuffle": False,
        "time_features": deepcopy(DataCombinationScenarios.DEFAULT_TIME_FEATURES),
        "expected_result": "PASS",
    }

    NO_TIME_FEATURES = {
        "name": "train_time_features_none",
        "year_slice": [1980, 1981],
        "num_data_workers": 0,
        "shuffle": False,
        "time_features": None,
        "expected_result": "PASS",
    }

    ALL_TIME_FEATURES = {
        "name": "train_time_features_all",
        "year_slice": [1980, 1981],
        "num_data_workers": 0,
        "shuffle": False,
        "time_features": sorted(DataCombinationScenarios.VALID_TIME_FEATURES),
        "expected_result": "PASS",
    }

    MULTIWORKER_NO_SHUFFLE = {
        "name": "train_multiworker_no_shuffle",
        "year_slice": [1980, 1981],
        "num_data_workers": 2,
        "shuffle": False,
        "time_features": deepcopy(DataCombinationScenarios.DEFAULT_TIME_FEATURES),
        "expected_result": "PASS",
    }

    MULTIWORKER_WITH_SHUFFLE = {
        "name": "train_multiworker_shuffle",
        "year_slice": [1980, 1981],
        "num_data_workers": 2,
        "shuffle": True,
        "time_features": deepcopy(DataCombinationScenarios.DEFAULT_TIME_FEATURES),
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


def clean_name(value: Any) -> str:
    """Convert value to filesystem-safe name"""
    value = str(value).replace("None", "none")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def case_paths(case_name: str) -> dict[str, Path]:
    """Generate paths for a test case"""
    case_dir = CASES_DIR / clean_name(case_name)
    return {
        "case_dir": case_dir,
        "config_path": case_dir / "config.yaml",
        "log_path": case_dir / "case.log",
        "preprocessor_dir": case_dir / "preprocessing_pipeline",
        "plot_dir": case_dir / "plots",
        "metadata_path": case_dir / "batch_metadata.json",
        "raw_data_dir": case_dir / "raw_data",
        "metadata_statistics_path": case_dir / "metadata_statistics.json",
    }


def prepare_output_directories() -> None:
    """Create output directories"""
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
    """Convert value to JSON string"""
    return json.dumps(value, sort_keys=True, default=str)


def init_csv() -> None:
    """Initialize CSV results file"""
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
    """Write test result to CSV"""
    params = task["params"]
    row = {
        "timestamp": datetime.now().isoformat(),
        "suite": task["suite"],
        "case_name": task["case_name"],
        "model_present": params.get("model_present"),
        "condition_present": params.get("condition_present"),
        "observation_present": params.get("observation_present"),
        "condition_kind": params.get("condition_kind", "dynamic"),
        "condition_method": params.get("condition_method"),
        "time_features": serialize_json(params.get("time_features")),
        "year_slice": serialize_json(params.get("year_slice")),
        "num_data_workers": params.get("num_data_workers"),
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


def extract_metadata(batch: Any) -> Any:
    """Extract metadata from batch"""
    if isinstance(batch, dict):
        return batch.get("metadata")
    return getattr(batch, "metadata", None)


def flatten_metadata(metadata: Any) -> list:
    """Convert metadata to list of dicts"""
    if metadata is None:
        return []
    if isinstance(metadata, list):
        return metadata
    if isinstance(metadata, tuple):
        return list(metadata)
    if isinstance(metadata, dict):
        lengths = [
            len(value)
            for value in metadata.values()
            if isinstance(value, (list, tuple, torch.Tensor, np.ndarray))
        ]
        if not lengths:
            return [metadata]

        count = lengths[0]
        return [
            {
                key: value[i] if isinstance(value, (list, tuple)) else value
                for key, value in metadata.items()
            }
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
    """Convert values to canonical form"""
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


def save_batch_metadata(
    metadata_list: list[Any],
    *,
    output_path: Path,
    batch_sizes: list[int],
    batch_index: int,
) -> None:
    """Save metadata from batches with rich statistics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export = {
        "timestamp": datetime.now().isoformat(),
        "batch_index": batch_index,
        "num_batches": len(batch_sizes),
        "batch_sizes": batch_sizes,
        "total_samples": sum(batch_sizes),
        "samples": [],
        "statistics": {},
    }

    for idx, item in enumerate(metadata_list):
        sample_data = {
            "sample_index": idx,
            "metadata": canonical_value(item),
        }
        export["samples"].append(sample_data)

    init_times = []
    lead_times = []

    for item in metadata_list:
        if isinstance(item, dict):
            init_times.append(item.get(INIT_TIME_DIM) or item.get("init_time"))
            lead_times.append(item.get(LEAD_TIME_DIM) or item.get("lead_time"))
        else:
            init_times.append(
                getattr(item, INIT_TIME_DIM, None) or getattr(item, "init_time", None)
            )
            lead_times.append(
                getattr(item, LEAD_TIME_DIM, None) or getattr(item, "lead_time", None)
            )

    export["statistics"]["init_times"] = {
        "unique": sorted(set(str(t) for t in init_times if t is not None)),
        "count": len([t for t in init_times if t is not None]),
    }

    export["statistics"]["lead_times"] = {
        "values": sorted(set(str(t) for t in lead_times if t is not None)),
        "count": len([t for t in lead_times if t is not None]),
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(export, file, indent=2, sort_keys=True, default=str)


def compute_metadata_statistics(metadata_list: list[Any]) -> dict[str, Any]:
    """Compute summary statistics from metadata"""
    if not metadata_list:
        return {}

    return {
        "total_samples": len(metadata_list),
        "unique_init_times": sorted(
            set(
                str(item.get(INIT_TIME_DIM) or item.get("init_time"))
                for item in metadata_list
                if isinstance(item, dict)
            )
        ),
        "unique_lead_times": sorted(
            set(
                str(item.get(LEAD_TIME_DIM) or item.get("lead_time"))
                for item in metadata_list
                if isinstance(item, dict)
            )
        ),
    }


def assert_tensor(name: str, value: Any) -> None:
    """Check that value is a valid tensor"""
    if not isinstance(value, torch.Tensor):
        raise AssertionError(f"{name} is not a tensor: {type(value).__name__}")
    if value.numel() == 0:
        raise AssertionError(f"{name} is empty")
    if torch.is_floating_point(value) and not torch.isfinite(value).all():
        raise AssertionError(f"{name} contains non-finite values")


def analyze_mask(mask: torch.Tensor, name: str = "mask") -> dict[str, Any]:
    """Analyze a spatial mask tensor (separate from channel visualization)"""
    if mask is None:
        return {"present": False}

    if not isinstance(mask, torch.Tensor):
        return {
            "present": False,
            "error": f"Expected tensor, got {type(mask).__name__}",
        }

    mask_np = mask.detach().cpu().numpy()

    is_binary = bool(np.all(np.isin(mask_np, [0, 1])))

    return {
        "present": True,
        "shape": tuple(mask.shape),
        "dtype": str(mask.dtype),
        "is_binary": is_binary,
        "valid_count": int(np.sum(mask_np == 1)) if is_binary else None,
        "valid_percentage": round(100.0 * np.sum(mask_np == 1) / mask_np.size, 2)
        if is_binary
        else None,
    }


def validate_masks_in_batch(batch: Any) -> dict[str, Any]:
    """Validate all masks in batch"""
    mask_fields = ["input_mask", "target_mask", "spatial_mask"]
    results = {}

    for field_name in mask_fields:
        if isinstance(batch, dict):
            mask = batch.get(field_name)
        else:
            mask = getattr(batch, field_name, None)

        if mask is not None:
            results[field_name] = analyze_mask(mask, name=field_name)

    return results


def save_batch_masks(
    batch: Any,
    *,
    output_dir: Path,
    batch_index: int,
) -> None:
    """
    Save masks from batch as NumPy files and JSON metadata.

    Saves:
    - input_mask.npy / input_mask.pt
    - target_mask.npy / target_mask.pt
    - spatial_mask.npy / spatial_mask.pt
    - masks_metadata.json (statistics about masks)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_fields = ["input_mask", "target_mask", "spatial_mask"]
    masks_metadata = {
        "timestamp": datetime.now().isoformat(),
        "batch_index": batch_index,
        "masks": {},
    }

    for field_name in mask_fields:
        if isinstance(batch, dict):
            mask = batch.get(field_name)
        else:
            mask = getattr(batch, field_name, None)

        if mask is None:
            continue

        # Convert to numpy if it's a tensor
        if isinstance(mask, torch.Tensor):
            mask_np = mask.detach().cpu().numpy()
        elif isinstance(mask, np.ndarray):
            mask_np = mask
        else:
            continue

        # Save as numpy file
        np_output = output_dir / f"{field_name}.npy"
        np.save(np_output, mask_np)

        # Also save as PyTorch file if it was a tensor
        if isinstance(mask, torch.Tensor):
            pt_output = output_dir / f"{field_name}.pt"
            torch.save(mask, pt_output)

        # Record statistics
        masks_metadata["masks"][field_name] = {
            "shape": tuple(mask_np.shape),
            "dtype": str(mask_np.dtype),
            "saved_as_npy": str(np_output),
            "saved_as_pt": str(output_dir / f"{field_name}.pt")
            if isinstance(mask, torch.Tensor)
            else None,
            "analysis": analyze_mask(mask, name=field_name),
        }

    # Save metadata JSON
    metadata_output = output_dir / "masks_metadata.json"
    with metadata_output.open("w") as f:
        json.dump(masks_metadata, f, indent=2, default=str)


def export_masks_and_data(
    batch: Any,
    *,
    output_dir: Path,
    metadata: list[Any],
    batch_index: int,
) -> None:
    """
    Export masks, data, and metadata together.

    Saves:
    - input_data.npy / input_data.pt
    - target_data.npy / target_data.pt
    - input_mask.npy / input_mask.pt
    - target_mask.npy / target_mask.pt
    - spatial_mask.npy / spatial_mask.pt
    - export_metadata.json (everything together)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    export_data = {
        "timestamp": datetime.now().isoformat(),
        "batch_index": batch_index,
        "num_samples": len(metadata),
        "samples_metadata": metadata,
        "data_fields": {},
        "mask_fields": {},
    }

    # Save data tensors
    data_fields = ["input", "target", "added_features"]
    for field_name in data_fields:
        if isinstance(batch, dict):
            data = batch.get(field_name)
        else:
            data = getattr(batch, field_name, None)

        if data is None:
            continue

        # Convert to numpy if tensor
        if isinstance(data, torch.Tensor):
            data_np = data.detach().cpu().numpy()
        elif isinstance(data, np.ndarray):
            data_np = data
        else:
            continue

        # Save as numpy file
        np_output = output_dir / f"{field_name}_data.npy"
        np.save(np_output, data_np)

        # Also save as PyTorch file if it was a tensor
        if isinstance(data, torch.Tensor):
            pt_output = output_dir / f"{field_name}_data.pt"
            torch.save(data, pt_output)

        # Record in metadata
        export_data["data_fields"][field_name] = {
            "shape": tuple(data_np.shape),
            "dtype": str(data_np.dtype),
            "min": float(np.nanmin(data_np)) if data_np.size > 0 else None,
            "max": float(np.nanmax(data_np)) if data_np.size > 0 else None,
            "mean": float(np.nanmean(data_np)) if data_np.size > 0 else None,
            "nan_count": int(np.isnan(data_np).sum()),
            "saved_as_npy": str(np_output),
            "saved_as_pt": str(output_dir / f"{field_name}_data.pt")
            if isinstance(data, torch.Tensor)
            else None,
        }

    # Save mask tensors
    mask_fields = ["input_mask", "target_mask", "spatial_mask"]
    for field_name in mask_fields:
        if isinstance(batch, dict):
            mask = batch.get(field_name)
        else:
            mask = getattr(batch, field_name, None)

        if mask is None:
            continue

        # Convert to numpy if tensor
        if isinstance(mask, torch.Tensor):
            mask_np = mask.detach().cpu().numpy()
        elif isinstance(mask, np.ndarray):
            mask_np = mask
        else:
            continue

        # Save as numpy file
        np_output = output_dir / f"{field_name}.npy"
        np.save(np_output, mask_np)

        # Also save as PyTorch file if it was a tensor
        if isinstance(mask, torch.Tensor):
            pt_output = output_dir / f"{field_name}.pt"
            torch.save(mask, pt_output)

        # Record in metadata
        export_data["mask_fields"][field_name] = {
            "shape": tuple(mask_np.shape),
            "dtype": str(mask_np.dtype),
            "is_binary": bool(np.all(np.isin(mask_np, [0, 1]))),
            "valid_count": int(np.sum(mask_np == 1)),
            "valid_percentage": round(100.0 * np.sum(mask_np == 1) / mask_np.size, 2)
            if mask_np.size > 0
            else 0,
            "saved_as_npy": str(np_output),
            "saved_as_pt": str(output_dir / f"{field_name}.pt")
            if isinstance(mask, torch.Tensor)
            else None,
        }

    # Save comprehensive metadata JSON
    metadata_output = output_dir / "export_metadata.json"
    with metadata_output.open("w") as f:
        json.dump(export_data, f, indent=2, default=str)


def test_dataset_getitem_format(dataset: Any) -> None:
    """Test that dataset.__getitem__ returns proper format"""
    if len(dataset) <= 0:
        raise AssertionError("Dataset is empty")

    indices = sorted({0, len(dataset) // 2, len(dataset) - 1})

    for index in indices:
        item = dataset[index]

        if not isinstance(item, tuple) or len(item) != 2:
            raise AssertionError(f"__getitem__ must return 2-tuple, got {type(item)}")

        sample_data, metadata = item

        if not isinstance(sample_data, dict):
            raise AssertionError(
                f"Sample must be dict, got {type(sample_data).__name__}"
            )
        if not isinstance(metadata, dict):
            raise AssertionError(
                f"Metadata must be dict, got {type(metadata).__name__}"
            )

        if "input" not in sample_data or "target" not in sample_data:
            raise AssertionError(f"Missing input or target at index {index}")

        assert_tensor(f"sample[{index}].input", sample_data["input"])
        assert_tensor(f"sample[{index}].target", sample_data["target"])

        # NEW: Validate masks separately
        if "input_mask" in sample_data:
            mask_analysis = analyze_mask(sample_data["input_mask"])
            if not mask_analysis["present"]:
                raise AssertionError(f"Invalid input_mask at {index}")

        required_meta = {INIT_TIME_DIM, LEAD_TIME_DIM}
        missing = required_meta - set(metadata.keys())
        if missing:
            raise AssertionError(f"Missing metadata: {sorted(missing)}")

    print("✓ Dataset.__getitem__ validated")


def normalize_nullable_fields(cfg: dict[str, Any]) -> dict[str, Any]:
    """Clean up config"""
    loader_cfg = cfg.setdefault("train_loader", {})
    dataset_cfg = loader_cfg.setdefault("dataset_config", {})

    if "time_features" in dataset_cfg:
        raise ValueError("time_features must be in train_loader")

    loader_cfg.pop("time_features", None) if loader_cfg.get(
        "time_features"
    ) is None else None
    dataset_cfg.pop("condition_method", None) if dataset_cfg.get(
        "condition_method"
    ) is None else None

    return cfg


def validate_generated_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate config"""
    train_loader_cfg = cfg.get("train_loader")
    if not isinstance(train_loader_cfg, dict):
        raise ValueError("train_loader must be a mapping")

    dataset_cfg = train_loader_cfg.get("dataset_config")
    if not isinstance(dataset_cfg, dict):
        raise ValueError("train_loader.dataset_config must be a mapping")

    if "model" not in dataset_cfg:
        raise ValueError("dataset_config.model is required")

    return cfg


def base_config() -> dict[str, Any]:
    """Load base training config"""
    cfg = prepare_config(BASE_TRAIN_CONFIG)
    if cfg is None:
        raise ValueError(f"Base config is empty: {BASE_TRAIN_CONFIG}")

    cfg = deepcopy(cfg)
    train_loader_cfg = cfg.get("train_loader")
    if not isinstance(train_loader_cfg, dict):
        raise ValueError("Base config must have train_loader mapping")

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


def apply_case_config(cfg: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """Apply test case parameters to config (runtime args NOT written to YAML)"""
    train_loader_cfg = cfg["train_loader"]
    dataset_cfg = train_loader_cfg["dataset_config"]

    if "condition_method" in params:
        condition_method = params["condition_method"]
        if condition_method is None:
            dataset_cfg.pop("condition_method", None)
        else:
            dataset_cfg["condition_method"] = condition_method

    if "time_features" in params:
        time_features = params["time_features"]
        if time_features is None:
            train_loader_cfg.pop("time_features", None)
        else:
            train_loader_cfg["time_features"] = deepcopy(time_features)

    if "year_slice" in params:
        train_loader_cfg["train_years_slice"] = deepcopy(params["year_slice"])

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

    # IMPORTANT: Do NOT write shuffle, return_metadata, return_spatial_mask to config
    # These are runtime arguments passed to setup_loader(), not config file fields
    for key in ["shuffle", "return_metadata", "return_spatial_mask"]:
        train_loader_cfg.pop(key, None)

    normalize_nullable_fields(cfg)
    validate_generated_config(cfg)
    return cfg


def run_dataset_case(config_path: Path, params: dict[str, Any]) -> None:
    """Test dataset.__getitem__ contract"""
    config = load_train_config(config_path)

    # Pass shuffle as runtime argument, NOT from config
    loader = setup_loader(
        config,
        config_path,
        return_metadata=True,
        return_spatial_mask=True,
        shuffle=False,  # Always deterministic for dataset testing
    )

    # Test __getitem__ format
    test_dataset_getitem_format(loader.dataset)

    # Collect metadata and create plots from first batch
    paths = case_paths(params["name"])
    all_metadata = []
    batch_sizes = []
    plot_created = False
    masks_saved = False

    for batch_idx, batch in enumerate(LimitedLoader(loader, MAX_BATCHES_PER_CASE)):
        batch_metadata = flatten_metadata(extract_metadata(batch))
        if batch_metadata:
            all_metadata.extend(batch_metadata)
            batch_sizes.append(len(batch_metadata))

        # Plot first batch only
        if batch_idx == 0 and not plot_created and RUN_CHANNEL_PLOTS:
            plot_first_batch(batch, output_dir=paths["plot_dir"])
            plot_created = True

        # Save masks and data from first batch
        if batch_idx == 0 and not masks_saved and RUN_RAW_DATA_EXPORT:
            export_masks_and_data(
                batch,
                output_dir=paths["raw_data_dir"],
                metadata=batch_metadata,
                batch_index=0,
            )
            masks_saved = True

    # Test masks
    mask_results = {"batches_checked": 0}
    for batch_idx, batch in enumerate(LimitedLoader(loader, MAX_BATCHES_PER_CASE)):
        masks = validate_masks_in_batch(batch)
        if masks:
            mask_results["batches_checked"] += 1

    # Save metadata
    save_batch_metadata(
        all_metadata,
        output_path=paths["metadata_path"],
        batch_sizes=batch_sizes,
        batch_index=0,
    )

    # Save statistics
    stats = compute_metadata_statistics(all_metadata)
    with paths["metadata_statistics_path"].open("w") as f:
        json.dump(stats, f, indent=2, default=str)

    print(
        f"✓ Collected {len(all_metadata)} samples | plots: {plot_created} | masks_exported: {masks_saved} | masks_validated: {mask_results['batches_checked']}"
    )


def run_dataloader_case(config_path: Path, params: dict[str, Any]) -> None:
    """Test dataloader with different configurations"""
    config = load_train_config(config_path)

    # Pass shuffle as runtime argument from params, NOT from config
    loader = setup_loader(
        config,
        config_path,
        return_metadata=True,
        return_spatial_mask=True,
        shuffle=params.get("shuffle", False),  # ← Runtime argument
    )

    paths = case_paths(params["name"])
    all_metadata = []
    batch_sizes = []
    plot_created = False
    masks_saved = False

    for batch_idx, batch in enumerate(LimitedLoader(loader, MAX_BATCHES_PER_CASE)):
        batch_metadata = flatten_metadata(extract_metadata(batch))
        if batch_metadata:
            all_metadata.extend(batch_metadata)
            batch_sizes.append(len(batch_metadata))

        # Plot first batch only
        if batch_idx == 0 and not plot_created and RUN_CHANNEL_PLOTS:
            plot_first_batch(batch, output_dir=paths["plot_dir"])
            plot_created = True

        # Save masks and data from first batch
        if batch_idx == 0 and not masks_saved and RUN_RAW_DATA_EXPORT:
            export_masks_and_data(
                batch,
                output_dir=paths["raw_data_dir"],
                metadata=batch_metadata,
                batch_index=0,
            )
            masks_saved = True

    save_batch_metadata(
        all_metadata,
        output_path=paths["metadata_path"],
        batch_sizes=batch_sizes,
        batch_index=0,
    )

    stats = compute_metadata_statistics(all_metadata)
    with paths["metadata_statistics_path"].open("w") as f:
        json.dump(stats, f, indent=2, default=str)

    print(
        f"✓ Dataloader: {len(all_metadata)} samples | plots: {plot_created} | masks_exported: {masks_saved}"
    )


def write_yaml(cfg: dict[str, Any], output_path: Path) -> Path:
    """Write config to YAML file"""
    cfg = deepcopy(cfg)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(cfg, file, sort_keys=False)
    return output_path


def cleanup_logging_handlers() -> None:
    """Close all logging handlers"""
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


def run_silenced(function: Callable[[], Any], log_path: Path) -> Any:
    """Run function and capture output to log file"""
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
    """Append exception traceback to log file"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as file:
        traceback.print_exc(file=file)


def configure_runtime_context(config_path: Path) -> Path:
    """Set up runtime environment"""
    case_dir = Path(config_path).resolve().parent
    case_dir.mkdir(parents=True, exist_ok=True)
    RuntimeContext.GLOBAL_EXP_DIR = str(case_dir)
    return case_dir


def reset_runtime_context() -> None:
    """Clear runtime context"""
    RuntimeContext.GLOBAL_EXP_DIR = None


def call_with_supported_arguments(function: Callable[..., Any], **kwargs: Any) -> Any:
    """Call function with only supported kwargs"""
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
    """Set up data loader"""
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


def load_train_config(config_path: Path) -> TrainConfig:
    """Load training config from file"""
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


class LimitedLoader:
    """Wrapper that limits batch iteration"""

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


def collect_batch_data(
    loader: Any,
    max_batches: int | None = None,
) -> tuple[list[Any], list[int]]:
    """Iterate through loader and collect metadata and batch sizes"""
    all_metadata = []
    batch_sizes = []

    for batch_idx, batch in enumerate(LimitedLoader(loader, max_batches)):
        batch_metadata = flatten_metadata(extract_metadata(batch))
        if batch_metadata:
            all_metadata.extend(batch_metadata)
            batch_sizes.append(len(batch_metadata))

    return all_metadata, batch_sizes


def plot_tensor_channels(
    tensor: torch.Tensor,
    *,
    name: str,
    output_dir: Path,
    metadata: Any = None,
) -> None:
    """Create plots of tensor channels for visual inspection"""
    if not RUN_CHANNEL_PLOTS:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    if tensor is None or not isinstance(tensor, torch.Tensor):
        return

    # Determine channel axis
    if tensor.ndim < 2:
        return

    channel_axis = 1 if tensor.ndim >= 2 else 0

    # Get number of channels
    if tensor.ndim >= 2:
        num_channels = tensor.shape[channel_axis]
    else:
        return

    if num_channels == 0:
        return

    # Plot up to MAX_PLOT_CHANNELS
    channels_to_plot = min(num_channels, MAX_PLOT_CHANNELS)

    for ch_idx in range(channels_to_plot):
        try:
            # Extract 2D image for this channel
            value = tensor.detach().float().cpu()

            # Remove batch dimension if present
            if value.ndim >= 4:
                value = value[0, ch_idx]
            elif value.ndim == 3:
                value = value[ch_idx]
            elif value.ndim == 2:
                value = value[ch_idx].unsqueeze(0) if ch_idx == 0 else value
            else:
                continue

            # Squeeze to 2D
            while value.ndim > 2:
                value = value[0]

            if value.ndim != 2:
                continue

            image = value.numpy()

            # Create plot
            figure, axes = plt.subplots(1, 1, figsize=(6, 5))
            im = axes.imshow(image, origin="lower", aspect="auto", cmap="viridis")
            axes.set_title(f"{name} - Channel {ch_idx}")
            figure.colorbar(im, ax=axes, shrink=0.8)
            figure.tight_layout()

            # Save
            output_file = output_dir / f"{clean_name(name)}_channel_{ch_idx:03d}.png"
            figure.savefig(output_file, dpi=150, bbox_inches="tight")
            plt.close(figure)

        except (IndexError, ValueError, RuntimeError):
            continue


def plot_first_batch(batch: Any, *, output_dir: Path) -> None:
    """Create plots of input, target, and masks from first batch"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot data
    if isinstance(batch, dict):
        input_tensor = batch.get("input")
        target_tensor = batch.get("target")
        added_features = batch.get("added_features")
    else:
        input_tensor = getattr(batch, "input", None)
        target_tensor = getattr(batch, "target", None)
        added_features = getattr(batch, "added_features", None)

    metadata = flatten_metadata(extract_metadata(batch))
    first_meta = metadata[0] if metadata else {}

    # Plot data channels
    if isinstance(input_tensor, torch.Tensor):
        plot_tensor_channels(
            input_tensor, name="input", output_dir=output_dir, metadata=first_meta
        )

    if isinstance(target_tensor, torch.Tensor):
        plot_tensor_channels(
            target_tensor, name="target", output_dir=output_dir, metadata=first_meta
        )

    if isinstance(added_features, torch.Tensor):
        plot_tensor_channels(
            added_features,
            name="added_features",
            output_dir=output_dir,
            metadata=first_meta,
        )

    # Plot masks
    plot_masks_from_batch(batch, output_dir=output_dir)


def plot_mask(
    mask: torch.Tensor,
    *,
    name: str,
    output_dir: Path,
) -> None:
    """
    Plot a binary mask as a visual heatmap.
    1s are white/bright, 0s are black/dark.
    """
    if not RUN_CHANNEL_PLOTS:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    if mask is None or not isinstance(mask, torch.Tensor):
        return

    mask_np = mask.detach().cpu().numpy()

    # Flatten to 2D if needed (squeeze out batch/channel dimensions for visualization)
    while mask_np.ndim > 2:
        mask_np = mask_np[0]

    if mask_np.ndim == 1:
        return

    # Create figure
    figure, axes = plt.subplots(1, 1, figsize=(8, 6))

    # Plot with binary colormap (0=black, 1=white)
    im = axes.imshow(mask_np, cmap="binary", origin="lower", aspect="auto")

    axes.set_title(f"{name} - Binary Mask\n(white=valid, black=masked)")
    axes.set_xlabel("X")
    axes.set_ylabel("Y")

    # Add colorbar
    cbar = figure.colorbar(im, ax=axes, shrink=0.8)
    cbar.set_label("Valid (1) / Masked (0)")

    # Add statistics as text
    valid_pct = 100.0 * np.sum(mask_np == 1) / mask_np.size if mask_np.size > 0 else 0
    axes.text(
        0.02,
        0.98,
        f"Valid: {valid_pct:.1f}%\nMasked: {100 - valid_pct:.1f}%",
        transform=axes.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    figure.tight_layout()

    # Save
    output_file = output_dir / f"{clean_name(name)}_mask.png"
    figure.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_masks_from_batch(batch: Any, *, output_dir: Path) -> None:
    """Plot all masks from batch"""
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_fields = ["input_mask", "target_mask", "spatial_mask", "temporal_mask"]

    for field_name in mask_fields:
        if isinstance(batch, dict):
            mask = batch.get(field_name)
        else:
            mask = getattr(batch, field_name, None)

        if mask is not None and isinstance(mask, torch.Tensor):
            plot_mask(mask, name=field_name, output_dir=output_dir)


def add_task(
    tasks: list[dict[str, Any]],
    *,
    suite: str,
    params: dict[str, Any],
    runner: Callable[..., Any],
) -> None:
    """Add task to suite"""
    tasks.append(
        {
            "suite": suite,
            "case_name": params["name"],
            "params": params,
            "cfg_builder": partial(apply_case_config, base_config(), params),
            "runner": runner,
        }
    )


def run_task(task: dict[str, Any]) -> tuple[bool, str]:
    """Run single test task"""
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

        cfg = cfg_builder()
        if not isinstance(cfg, dict):
            raise TypeError(f"cfg_builder returned {type(cfg).__name__}")

        write_yaml(cfg, paths["config_path"])

        print(f"\nRunning {task['suite']}:{task['case_name']}", flush=True)

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

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    log_result(task=task, paths=paths, passed=passed, error=error)
    return passed, error


def validate_tasks(tasks: list[dict[str, Any]]) -> None:
    """Validate all tasks are properly formed"""
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


def build_test_suite() -> list[dict[str, Any]]:
    """Build entire test suite using hardcoded scenarios"""
    tasks = []

    # Data combination scenarios
    if RUN_DATA_CONFIG_COMBINATIONS:
        for case in DataCombinationScenarios.all():
            add_task(
                tasks,
                suite="data_config_combination",
                params=case,
                runner=run_dataset_case,
            )

    # External condition scenarios
    if RUN_EXTERNAL_CONDITION_CASES:
        for case in ExternalConditionScenarios.all():
            add_task(
                tasks,
                suite="external_condition",
                params=case,
                runner=run_dataset_case,
            )

    # Dataloader scenarios
    if RUN_TRAIN_DATALOADER_CASES:
        for case in DataloaderScenarios.all():
            add_task(
                tasks,
                suite="train_dataloader",
                params=case,
                runner=run_dataloader_case,
            )

    return tasks


def write_summary(
    tasks: list[dict[str, Any]],
    passed_count: int,
    failed_count: int,
) -> None:
    """Write summary report"""
    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        file.write(f"total_cases: {len(tasks)}\n")
        file.write(f"passed: {passed_count}\n")
        file.write(f"failed: {failed_count}\n")
        file.write(f"cases_dir: {CASES_DIR}\n")
        file.write(f"results_csv: {RESULTS_CSV}\n")


def main() -> None:
    """Main entry point"""
    prepare_output_directories()
    cleanup_logging_handlers()
    init_csv()

    # Build test suite
    tasks = build_test_suite()
    validate_tasks(tasks)

    if MAX_CASES is not None:
        tasks = tasks[:MAX_CASES]

    passed_count = 0
    failed_count = 0

    try:
        with tqdm(
            total=len(tasks),
            desc="integration_tests",
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

                progress.update(1)

    finally:
        reset_runtime_context()
        cleanup_logging_handlers()

    write_summary(tasks, passed_count, failed_count)

    if failed_count > 0:
        raise RuntimeError(
            f"Integration suite failed: {failed_count} failure(s). "
            f"See {RESULTS_CSV} and {CASES_DIR}."
        )

    print(f"✓ Integration test suite completed: {passed_count} passed")
    print(f"  Cases: {CASES_DIR}")
    print(f"  Results: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
