from __future__ import annotations

import csv
import json
import logging
import re
import shutil
import traceback
import warnings
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

from cccma_ppp.configs import required_sample_dimensions
from cccma_ppp.generic.runtime import RuntimeContext
from cccma_ppp.train.train_configs import prepare_config

warnings.filterwarnings("ignore")
logging.raiseExceptions = False


ROOT = Path("/fs/site7/eccc/crd/cccma/users/rna002/CCCma-PPP/scripts")
BASE_TRAIN_CONFIG = "scripts/module_integration_tests/deterministic_config.yaml"

OUTPUT_DIR = (
    ROOT / "../../output" / "data_modules_integration_test_results_deterministic"
)

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
RUN_EXTERNAL_CONDITION_CASES = True
RUN_TRAIN_DATALOADER_CASES = True

MAX_CASES: int | None = None
MAX_BATCHES_PER_CASE = 4
MAX_PLOT_CHANNELS = 16

CSV_COLUMNS = [
    "timestamp",
    "suite",
    "case_name",
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
    "plot_dir",
    "result",
    "error",
]


@dataclass
class CaseParams:
    """Parameters for a single test case"""

    name: str
    model_present: bool
    condition_present: bool
    observation_present: bool
    condition_method: Optional[str] = None
    condition_kind: Optional[str] = None
    condition_path: Optional[str] = None
    model_ensemble_mean: Optional[bool] = None
    condition_ensemble_mean: Optional[bool] = None
    condition_has_ensemble: Optional[bool] = None
    time_features: Optional[List[str]] = None
    year_slice: Optional[List[int]] = None
    num_data_workers: Optional[int] = None
    shuffle: Optional[bool] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict, excluding None values"""
        return {k: v for k, v in asdict(self).items() if v is not None}


class TestScenarios:
    """All test scenarios"""

    VALID_TIME_FEATURES = {
        INIT_TIME_DIM,
        LEAD_TIME_DIM,
        "month_sin",
        "month_cos",
        "day_sin",
        "day_cos",
    }
    DEFAULT_TIME_FEATURES = [LEAD_TIME_DIM, "month_sin", "month_cos"]

    AUTOENCODE_MODEL_ONLY_ENSEMBLE_MEAN = CaseParams(
        name="autoencode_model_only_ensemble_mean",
        model_present=True,
        condition_present=False,
        observation_present=False,
        condition_method="ensemble_mean",
        model_ensemble_mean=True,
    )

    AUTOENCODE_MODEL_ONLY_CROSS_ENSEMBLE = CaseParams(
        name="autoencode_model_only_cross_ensemble",
        model_present=True,
        condition_present=False,
        observation_present=False,
        condition_method="cross_ensemble",
        model_ensemble_mean=False,
    )

    AUTOENCODE_MODEL_ONLY_SAME_MEMBER = CaseParams(
        name="autoencode_model_only_same_member",
        model_present=True,
        condition_present=False,
        observation_present=False,
        condition_method="same_member",
        model_ensemble_mean=False,
    )

    AUTOENCODE_WITH_CONDITION_ENSEMBLE_MEAN = CaseParams(
        name="autoencode_with_condition_ensemble_mean",
        model_present=True,
        condition_present=True,
        observation_present=False,
        condition_method="ensemble_mean",
        model_ensemble_mean=True,
        condition_ensemble_mean=True,
    )

    AUTOENCODE_WITH_CONDITION_CROSS_ENSEMBLE = CaseParams(
        name="autoencode_with_condition_cross_ensemble",
        model_present=True,
        condition_present=True,
        observation_present=False,
        condition_method="cross_ensemble",
        model_ensemble_mean=False,
        condition_ensemble_mean=False,
    )

    AUTOENCODE_WITH_CONDITION_SAME_MEMBER = CaseParams(
        name="autoencode_with_condition_same_member",
        model_present=True,
        condition_present=True,
        observation_present=False,
        condition_method="same_member",
        model_ensemble_mean=False,
        condition_ensemble_mean=False,
    )

    BIAS_CORRECT_MODEL_CONDITION_ENSEMBLE_MEAN = CaseParams(
        name="bias_correct_model_condition_ensemble_mean",
        model_present=True,
        condition_present=True,
        observation_present=True,
        condition_method="ensemble_mean",
        model_ensemble_mean=False,
        condition_ensemble_mean=False,
    )

    BIAS_CORRECT_MODEL_CONDITION_CROSS_ENSEMBLE = CaseParams(
        name="bias_correct_model_condition_cross_ensemble",
        model_present=True,
        condition_present=True,
        observation_present=True,
        condition_method="cross_ensemble",
        model_ensemble_mean=False,
        condition_ensemble_mean=False,
    )

    BIAS_CORRECT_MODEL_CONDITION_SAME_MEMBER = CaseParams(
        name="bias_correct_model_condition_same_member",
        model_present=True,
        condition_present=True,
        observation_present=True,
        condition_method="same_member",
        model_ensemble_mean=False,
        condition_ensemble_mean=False,
    )

    BIAS_CORRECT_MODEL_CONDITION_STATIC = CaseParams(
        name="bias_correct_model_condition_static",
        model_present=True,
        condition_present=True,
        observation_present=True,
        condition_method="static",
        model_ensemble_mean=False,
        condition_ensemble_mean=None,
    )

    BIAS_CORRECT_MODEL_WITH_METHOD = CaseParams(
        name="bias_correct_model_with_method",
        model_present=True,
        condition_present=False,
        observation_present=True,
        condition_method="ensemble_mean",
        model_ensemble_mean=True,
    )

    BIAS_CORRECT_MODEL_DIRECT = CaseParams(
        name="bias_correct_model_direct",
        model_present=True,
        condition_present=False,
        observation_present=True,
        condition_method=None,
        model_ensemble_mean=False,
    )

    TAS_STDR_ENSEMBLE_MEAN = CaseParams(
        name="bias_correct_with_tas_stdr_ensemble_mean",
        model_present=True,
        condition_present=True,
        observation_present=True,
        condition_kind="tas_stdr",
        condition_path=str(TAS_STDR_PATH),
        condition_method="ensemble_mean",
        model_ensemble_mean=False,
        condition_ensemble_mean=None,
        condition_has_ensemble=False,
    )

    TAS_STDR_CROSS_ENSEMBLE = CaseParams(
        name="bias_correct_with_tas_stdr_cross_ensemble",
        model_present=True,
        condition_present=True,
        observation_present=True,
        condition_kind="tas_stdr",
        condition_path=str(TAS_STDR_PATH),
        condition_method="cross_ensemble",
        model_ensemble_mean=False,
        condition_ensemble_mean=None,
        condition_has_ensemble=False,
    )

    TAS_STDR_SAME_MEMBER = CaseParams(
        name="bias_correct_with_tas_stdr_same_member",
        model_present=True,
        condition_present=True,
        observation_present=True,
        condition_kind="tas_stdr",
        condition_path=str(TAS_STDR_PATH),
        condition_method="same_member",
        model_ensemble_mean=False,
        condition_ensemble_mean=None,
        condition_has_ensemble=False,
    )

    STATIC_MASK_CONDITION = CaseParams(
        name="bias_correct_with_static_dummy_mask",
        model_present=True,
        condition_present=True,
        observation_present=True,
        condition_kind="static",
        condition_path=str(DUMMY_MASK_PATH),
        condition_method="static",
        model_ensemble_mean=False,
        condition_ensemble_mean=None,
        condition_has_ensemble=False,
    )

    YEAR_SLICE_ONLY = CaseParams(
        name="train_year_slice",
        model_present=True,
        condition_present=True,
        observation_present=True,
        year_slice=[1980, 2007],
        num_data_workers=0,
        shuffle=False,
        time_features=deepcopy(DEFAULT_TIME_FEATURES),
    )

    NO_TIME_FEATURES = CaseParams(
        name="train_time_features_none",
        model_present=True,
        condition_present=True,
        observation_present=True,
        year_slice=[1980, 2007],
        num_data_workers=0,
        shuffle=False,
        time_features=None,
    )

    ALL_TIME_FEATURES = CaseParams(
        name="train_time_features_all",
        model_present=True,
        condition_present=False,
        observation_present=True,
        year_slice=[1980, 2007],
        num_data_workers=0,
        shuffle=False,
        time_features=sorted(VALID_TIME_FEATURES),
    )

    @classmethod
    def data_combinations(cls) -> list[CaseParams]:
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

    @classmethod
    def external_conditions(cls) -> list[CaseParams]:
        return [
            cls.TAS_STDR_ENSEMBLE_MEAN,
            cls.TAS_STDR_CROSS_ENSEMBLE,
            cls.TAS_STDR_SAME_MEMBER,
            cls.STATIC_MASK_CONDITION,
        ]

    @classmethod
    def dataloaders(cls) -> list[CaseParams]:
        return [
            cls.YEAR_SLICE_ONLY,
            cls.NO_TIME_FEATURES,
            cls.ALL_TIME_FEATURES,
        ]


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
        "plot_dir": case_dir / "plots",
        "preprocessing_dir": case_dir / "preprocessing_pipeline",
        "metadata_path": case_dir / "batch_metadata.json",
    }


def prepare_output_directories() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CASES_DIR.mkdir(parents=True, exist_ok=True)


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
    case_name: str,
    suite: str,
    params: dict[str, Any],
    paths: dict[str, Path],
    passed: bool,
    error: str,
) -> None:
    row = {
        "timestamp": datetime.now().isoformat(),
        "suite": suite,
        "case_name": case_name,
        "model_present": params.get("model_present"),
        "condition_present": params.get("condition_present"),
        "observation_present": params.get("observation_present"),
        "condition_kind": params.get("condition_kind"),
        "condition_method": params.get("condition_method"),
        "time_features": serialize_json(params.get("time_features")),
        "year_slice": serialize_json(params.get("year_slice")),
        "num_data_workers": params.get("num_data_workers"),
        "case_dir": str(paths["case_dir"]),
        "config_path": str(paths["config_path"]),
        "log_path": str(paths["log_path"]),
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


def append_exception_to_log(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as file:
        traceback.print_exc(file=file)


def reset_runtime_context() -> None:
    RuntimeContext.GLOBAL_EXP_DIR = None


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


def plot_tensor_channels(
    tensor: torch.Tensor,
    *,
    name: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if tensor is None or not isinstance(tensor, torch.Tensor) or tensor.ndim < 2:
        return

    num_channels = tensor.shape[1] if tensor.ndim >= 2 else 0
    if num_channels == 0:
        return

    channels_to_plot = min(num_channels, MAX_PLOT_CHANNELS)

    for ch_idx in range(channels_to_plot):
        try:
            value = tensor.detach().float().cpu()

            if value.ndim >= 4:
                value = value[0, ch_idx]
            elif value.ndim == 3:
                value = value[ch_idx]
            else:
                continue

            while value.ndim > 2:
                value = value[0]

            if value.ndim != 2:
                continue

            image = value.numpy()

            figure, axes = plt.subplots(1, 1, figsize=(6, 5))
            im = axes.imshow(image, origin="lower", aspect="auto", cmap="viridis")
            axes.set_title(f"{name} - Channel {ch_idx}")
            figure.colorbar(im, ax=axes, shrink=0.8)
            figure.tight_layout()

            output_file = output_dir / f"{clean_name(name)}_channel_{ch_idx:03d}.png"
            figure.savefig(output_file, dpi=150, bbox_inches="tight")
            plt.close(figure)

        except (IndexError, ValueError, RuntimeError):
            continue


def plot_mask(
    mask: torch.Tensor,
    *,
    name: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if mask is None or not isinstance(mask, torch.Tensor):
        return

    mask_np = mask.detach().cpu().numpy()

    while mask_np.ndim > 2:
        mask_np = mask_np[0]

    if mask_np.ndim != 2:
        return

    figure, axes = plt.subplots(1, 1, figsize=(8, 6))
    im = axes.imshow(mask_np, cmap="binary", origin="lower", aspect="auto")

    axes.set_title(f"{name} - Mask")
    cbar = figure.colorbar(im, ax=axes, shrink=0.8)
    cbar.set_label("Valid (1) / Masked (0)")

    figure.tight_layout()

    output_file = output_dir / f"{clean_name(name)}_mask.png"
    figure.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_first_batch(batch: Any, *, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(batch, dict):
        input_tensor = batch.get("input")
        target_tensor = batch.get("target")
        added_features = batch.get("added_features")
        temporal_mask = batch.get("temporal_mask")
    else:
        input_tensor = getattr(batch, "input", None)
        target_tensor = getattr(batch, "target", None)
        added_features = getattr(batch, "added_features", None)
        temporal_mask = getattr(batch, "temporal_mask", None)

    if isinstance(input_tensor, torch.Tensor):
        plot_tensor_channels(input_tensor, name="input", output_dir=output_dir)

    if isinstance(target_tensor, torch.Tensor):
        plot_tensor_channels(target_tensor, name="target", output_dir=output_dir)

    if isinstance(added_features, torch.Tensor):
        plot_tensor_channels(
            added_features, name="added_features", output_dir=output_dir
        )

    if isinstance(temporal_mask, torch.Tensor):
        plot_mask(temporal_mask, name="temporal_mask", output_dir=output_dir)


def extract_metadata(batch: Any) -> Any:
    if isinstance(batch, dict):
        return batch.get("metadata")
    return getattr(batch, "metadata", None)


def flatten_metadata(metadata: Any) -> list:
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
                key: scalar_value(value[i])
                if isinstance(value, (list, tuple))
                else scalar_value(value)
                for key, value in metadata.items()
            }
            for i in range(count)
        ]
    return [metadata]


def scalar_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_batch_metadata(
    metadata_list: list[Any],
    *,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export = {
        "timestamp": datetime.now().isoformat(),
        "num_samples": len(metadata_list),
        "samples": [],
    }

    for idx, item in enumerate(metadata_list):
        export["samples"].append(
            {
                "sample_index": idx,
                "metadata": item,
            }
        )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(export, file, indent=2, sort_keys=True, default=str)


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
        raise ValueError("Base config must have dataset_config")

    train_loader_cfg["batch_size"] = 2
    train_loader_cfg["num_data_workers"] = 0
    train_loader_cfg["num_validation_years"] = 0
    train_loader_cfg["train_years_slice"] = [1980, 2007]
    train_loader_cfg["load"] = False

    for key in ["shuffle", "return_metadata", "return_spatial_mask", "prefetch_factor"]:
        train_loader_cfg.pop(key, None)

    return cfg


def apply_case_config(cfg: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    cfg = deepcopy(cfg)
    train_loader_cfg = cfg["train_loader"]
    dataset_cfg = train_loader_cfg["dataset_config"]

    if not params.get("model_present", True):
        raise ValueError("Model is required")

    if "model" not in dataset_cfg:
        raise ValueError("Base config missing model config")

    model_cfg = dataset_cfg["model"]
    if "model_ensemble_mean" in params and params["model_ensemble_mean"] is not None:
        model_cfg["ensemble_mean"] = params["model_ensemble_mean"]

    if params.get("condition_present", False):
        if "condition" not in dataset_cfg:
            condition_cfg = deepcopy(model_cfg)
        else:
            condition_cfg = dataset_cfg["condition"]

        if "condition_path" in params and params["condition_path"]:
            condition_cfg = _apply_external_condition_path(
                condition_cfg,
                path=params["condition_path"],
                has_ensemble=params.get("condition_has_ensemble", True),
                condition_kind=params.get("condition_kind", "dynamic"),
            )

        if "condition_ensemble_mean" in params:
            if params["condition_ensemble_mean"] is not None:
                condition_cfg["ensemble_mean"] = params["condition_ensemble_mean"]

        dataset_cfg["condition"] = condition_cfg
    else:
        dataset_cfg.pop("condition", None)

    if params.get("observation_present", False):
        if "observation" not in dataset_cfg:
            raise ValueError("Base config missing observation config")
    else:
        dataset_cfg.pop("observation", None)

    if "condition_method" in params:
        if params["condition_method"] is None:
            dataset_cfg.pop("condition_method", None)
        else:
            dataset_cfg["condition_method"] = params["condition_method"]

    if "time_features" in params:
        if params["time_features"] is None:
            train_loader_cfg.pop("time_features", None)
        else:
            train_loader_cfg["time_features"] = deepcopy(params["time_features"])

    if "year_slice" in params and params["year_slice"]:
        train_loader_cfg["train_years_slice"] = deepcopy(params["year_slice"])

    if "num_data_workers" in params and params["num_data_workers"] is not None:
        num_data_workers = params["num_data_workers"]
        if not isinstance(num_data_workers, int) or num_data_workers < 0:
            raise ValueError("num_data_workers must be non-negative integer")
        train_loader_cfg["num_data_workers"] = num_data_workers

    num_data_workers = train_loader_cfg.get("num_data_workers", 0)
    if num_data_workers == 0:
        train_loader_cfg.pop("prefetch_factor", None)
    else:
        train_loader_cfg.setdefault("prefetch_factor", 2)

    for key in ["shuffle", "return_metadata", "return_spatial_mask"]:
        train_loader_cfg.pop(key, None)

    return cfg


def _apply_external_condition_path(
    condition_cfg: dict[str, Any],
    *,
    path: str,
    has_ensemble: bool,
    condition_kind: str,
) -> dict[str, Any]:
    cfg = deepcopy(condition_cfg)
    cfg = _replace_paths_in_dict(cfg, path)

    if not has_ensemble:
        cfg = _remove_ensemble_fields(cfg)

    if condition_kind == "static":
        for key in [
            "lead_times",
            "init_times",
            "initialization_times",
            "time_slice",
            "year_slice",
            "train_years_slice",
        ]:
            cfg.pop(key, None)

    return cfg


def _replace_paths_in_dict(cfg: dict[str, Any], new_path: str) -> dict[str, Any]:
    path_keywords = {
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

    def _recurse(obj: Any) -> Any:
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                key_lower = str(key).lower()
                if key_lower in path_keywords or key_lower.endswith(
                    ("_path", "_paths", "_directory", "_file", "_files")
                ):
                    result[key] = [new_path] if isinstance(value, list) else new_path
                else:
                    result[key] = _recurse(value)
            return result
        elif isinstance(obj, list):
            return [_recurse(item) for item in obj]
        return obj

    return _recurse(cfg)


def _remove_ensemble_fields(cfg: dict[str, Any]) -> dict[str, Any]:
    ensemble_keywords = {
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

    def _recurse(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                key: _recurse(value)
                for key, value in obj.items()
                if str(key).lower() not in ensemble_keywords
            }
        elif isinstance(obj, list):
            return [_recurse(item) for item in obj]
        return obj

    return _recurse(cfg)


def write_yaml(cfg: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    return output_path


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
        suite = task["suite"]

        cfg = apply_case_config(base_config(), params)
        write_yaml(cfg, paths["config_path"])

        def test_fn():
            from cccma_ppp.generic.distributed import Distributed
            import dacite
            from cccma_ppp.train.train_configs import TrainConfig

            RuntimeContext.GLOBAL_EXP_DIR = str(case_dir)
            data = prepare_config(paths["config_path"])
            config = dacite.from_dict(
                data_class=TrainConfig,
                data=data,
                config=dacite.Config(strict=True),
            )

            distributed = Distributed.get_instance()
            config.train_loader.setup_distributed(distributed=distributed)

            loader = config.train_loader.build_train_loader(
                return_metadata=True,
                return_spatial_mask=True,
                shuffle=False,
            )

            dataset = loader.dataset
            length = len(dataset)
            print(f"dataset length: {length}")

            all_metadata = []

            for i, batch in enumerate(loader):
                if i >= MAX_BATCHES_PER_CASE:
                    break

                if i == 0:
                    plot_first_batch(batch, output_dir=paths["plot_dir"])

                metadata = extract_metadata(batch)
                if metadata is not None:
                    batch_metadata = flatten_metadata(metadata)
                    if batch_metadata:
                        all_metadata.extend(batch_metadata)

                input_ = (
                    batch.get("input")
                    if isinstance(batch, dict)
                    else getattr(batch, "input", None)
                )
                shape = tuple(input_.shape) if input_ is not None else None
                print(f"batch {i}: input shape = {shape}")

            if all_metadata:
                save_batch_metadata(all_metadata, output_path=paths["metadata_path"])

            print("Run completed without error.")

        run_silenced(test_fn, paths["log_path"])
        passed = True

    except Exception as exc:
        error = f"{type(exc).__name__}: {str(exc)[:200]}"
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

    log_result(
        case_name=task["case_name"],
        suite=suite,
        params=task["params"],
        paths=paths,
        passed=passed,
        error=error,
    )

    return passed, error


def build_test_suite() -> list[dict[str, Any]]:
    tasks = []

    if RUN_DATA_CONFIG_COMBINATIONS:
        for case in TestScenarios.data_combinations():
            tasks.append(
                {
                    "suite": "data_config_combination",
                    "case_name": case.name,
                    "params": case.to_dict(),
                }
            )

    if RUN_EXTERNAL_CONDITION_CASES:
        for case in TestScenarios.external_conditions():
            tasks.append(
                {
                    "suite": "external_condition",
                    "case_name": case.name,
                    "params": case.to_dict(),
                }
            )

    if RUN_TRAIN_DATALOADER_CASES:
        for case in TestScenarios.dataloaders():
            tasks.append(
                {
                    "suite": "train_dataloader",
                    "case_name": case.name,
                    "params": case.to_dict(),
                }
            )

    return tasks


def write_summary(
    tasks: list[dict[str, Any]],
    passed_count: int,
    failed_count: int,
) -> None:
    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        f.write(f"total_cases: {len(tasks)}\n")
        f.write(f"passed: {passed_count}\n")
        f.write(f"failed: {failed_count}\n")
        f.write(f"cases_dir: {CASES_DIR}\n")
        f.write(f"results_csv: {RESULTS_CSV}\n")


def main() -> None:
    prepare_output_directories()
    cleanup_logging_handlers()
    init_csv()

    tasks = build_test_suite()

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
        ) as pbar:
            for task in tasks:
                pbar.set_postfix_str(f"{task['suite']}:{task['case_name'][:40]}")

                passed, error = run_task(task)
                if passed:
                    passed_count += 1
                else:
                    failed_count += 1
                    print(f"  {task['case_name']}: {error[:80]}")

                pbar.update(1)

    finally:
        reset_runtime_context()
        cleanup_logging_handlers()

    write_summary(tasks, passed_count, failed_count)

    print(f"\n{'=' * 80}")
    print(f"Integration test suite: {passed_count} passed, {failed_count} failed")
    print(f"Results: {RESULTS_CSV}")
    print(f"Cases: {CASES_DIR}")
    print(f"{'=' * 80}")

    if failed_count > 0:
        raise RuntimeError(f"{failed_count} test(s) failed")


if __name__ == "__main__":
    main()
