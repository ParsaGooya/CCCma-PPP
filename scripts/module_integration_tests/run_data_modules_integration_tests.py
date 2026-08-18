from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterator
import csv
import gc
import json
import logging
import re
import shutil
import traceback
import warnings

import dacite
import numpy as np
import torch
import yaml
from tqdm import tqdm

from cccma_ppp.configs import required_sample_dimensions
from cccma_ppp.generic import registry_imports  # noqa: F401
from cccma_ppp.generic.distributed import Distributed
from cccma_ppp.train.train_configs import TrainConfig, prepare_config


warnings.filterwarnings("ignore")
logging.raiseExceptions = False


ROOT = Path(__file__).resolve().parents[1]
BASE_TRAIN_CONFIG = ROOT / "scripts" / "integration_suite_train_config.yaml"

OUTPUT_DIR = ROOT / "../output" / "data_modules_integration_test_results"
GENERATED_CONFIG_DIR = OUTPUT_DIR / "_generated_configs"
LOG_DIR = OUTPUT_DIR / "_logs"
PREPROCESSOR_DIR = OUTPUT_DIR / "_preprocessing_pipeline"
RESULTS_CSV = OUTPUT_DIR / "integration_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary.txt"

INIT_TIME_DIM, LEAD_TIME_DIM = required_sample_dimensions

RUN_DATA_CONFIG_COMBINATIONS = True
RUN_DATASET_GETITEM_CASES = True
RUN_TRAIN_DATALOADER_CASES = True
RUN_INFERENCE_DATALOADER_CASES = True
RUN_MULTIWORKER_CASES = True
RUN_FUTURE_MASK_CASES = True
RUN_DISTRIBUTED_PREPROCESSOR_CASES = True

MAX_CASES: int | None = None
MAX_BATCHES_PER_CASE = 4

VALID_TIME_FEATURES = {
    INIT_TIME_DIM,
    LEAD_TIME_DIM,
    "month_sin",
    "month_cos",
    "day_sin",
    "day_cos",
}

DEFAULT_TIME_FEATURES = [
    LEAD_TIME_DIM,
    "month_sin",
    "month_cos",
]

FULL_LEAD_MONTHS = {
    "start": 1,
    "end": 12,
}

CSV_COLUMNS = [
    "timestamp",
    "suite",
    "case_name",
    "description",
    "model_present",
    "condition_present",
    "observation_present",
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
    "config_path",
    "log_path",
    "result",
    "error",
]


# Required Dataset.__getitem__ combinations.
#
# model condition observation condition_method  INPUT                    TARGET
# Y     N         N           MUST              model by method          model
# Y     Y         N           MUST              condition by method      model
# Y     Y         Y           MUST              model + condition        observation
# Y     N         Y           OPTIONAL          model                    observation
DATA_CONFIG_COMBINATIONS = [
    {
        "name": "autoencode_model_conditioned_on_itself",
        "description": (
            "Autoencode model data conditioned on model data sampled according "
            "to condition_method."
        ),
        "model_present": True,
        "condition_present": False,
        "observation_present": False,
        "condition_method_requirement": "MUST",
        "condition_method": "ensemble_mean",
        "model_ensemble_mean": True,
        "expected_input": "model data according to condition_method",
        "expected_target": (
            "model data according to available initialization time, lead time, "
            "and ensemble"
        ),
        "expected_result": "PASS",
    },
    {
        "name": "autoencode_model_with_explicit_condition",
        "description": (
            "Autoencode model data conditioned on the provided condition data, "
            "sampled according to condition_method."
        ),
        "model_present": True,
        "condition_present": True,
        "observation_present": False,
        "condition_method_requirement": "MUST",
        "condition_method": "ensemble_mean",
        "model_ensemble_mean": True,
        "condition_ensemble_mean": True,
        "expected_input": "condition data according to condition_method",
        "expected_target": (
            "model data according to available initialization time, lead time, "
            "and ensemble"
        ),
        "expected_result": "PASS",
    },
    {
        "name": "bias_correct_model_and_condition_to_observation",
        "description": (
            "Bias-correct to observations using model and condition data as the "
            "conditioning field."
        ),
        "model_present": True,
        "condition_present": True,
        "observation_present": True,
        "condition_method_requirement": "MUST",
        "condition_method": "same_member",
        "model_ensemble_mean": False,
        "condition_ensemble_mean": False,
        "expected_input": (
            "model data at the sampled coordinates concatenated with condition "
            "data selected according to condition_method"
        ),
        "expected_target": "observation data",
        "expected_result": "PASS",
    },
    {
        "name": "bias_correct_model_to_observation_with_method",
        "description": (
            "Bias-correct to observations using model data as the conditioning "
            "field, selected according to condition_method."
        ),
        "model_present": True,
        "condition_present": False,
        "observation_present": True,
        "condition_method_requirement": "OPTIONAL",
        "condition_method": "ensemble_mean",
        "model_ensemble_mean": True,
        "expected_input": "model data according to condition_method",
        "expected_target": "observation data",
        "expected_result": "PASS",
    },
    {
        "name": "bias_correct_model_to_observation_without_method",
        "description": (
            "Bias-correct to observations using all available model times, lead "
            "times, and members when condition_method is omitted."
        ),
        "model_present": True,
        "condition_present": False,
        "observation_present": True,
        "condition_method_requirement": "OPTIONAL",
        "condition_method": None,
        "model_ensemble_mean": False,
        "expected_input": (
            "model data over all available years, lead times, and ensembles"
        ),
        "expected_target": "observation data",
        "expected_result": "PASS",
    },
]


DATALOADER_CASES = [
    {
        "name": "train_year_slice",
        "description": "Select the configured training years and iterate batches.",
        "year_slice": [1980, 1981],
        "num_data_workers": 0,
        "shuffle": False,
        "time_features": deepcopy(DEFAULT_TIME_FEATURES),
        "check": "year_slice",
        "expected_result": "PASS",
    },
    {
        "name": "train_time_features_none",
        "description": "Build batches without added time features.",
        "year_slice": [1980, 1981],
        "num_data_workers": 0,
        "shuffle": False,
        "time_features": None,
        "check": "time_features",
        "expected_result": "PASS",
    },
    {
        "name": "train_time_features_all",
        "description": "Build batches with all supported time features.",
        "year_slice": [1980, 1981],
        "num_data_workers": 0,
        "shuffle": False,
        "time_features": sorted(VALID_TIME_FEATURES),
        "check": "time_features",
        "expected_result": "PASS",
    },
    {
        "name": "train_multiworker_no_shuffle",
        "description": (
            "Multiple workers must preserve sample coverage and deterministic "
            "ordering when shuffle is disabled."
        ),
        "year_slice": [1980, 1981],
        "num_data_workers": 2,
        "shuffle": False,
        "time_features": deepcopy(DEFAULT_TIME_FEATURES),
        "check": "multiworker",
        "expected_result": "PASS",
    },
    {
        "name": "train_multiworker_shuffle",
        "description": (
            "Multiple workers with shuffling must neither duplicate nor lose "
            "samples within an epoch."
        ),
        "year_slice": [1980, 1981],
        "num_data_workers": 2,
        "shuffle": True,
        "time_features": deepcopy(DEFAULT_TIME_FEATURES),
        "check": "multiworker",
        "expected_result": "PASS",
    },
]


class LimitedLoader:
    def __init__(self, loader: Any, max_batches: int | None) -> None:
        self._loader = loader
        self._max_batches = max_batches

    def __iter__(self) -> Iterator[Any]:
        for index, batch in enumerate(self._loader):
            if self._max_batches is not None and index >= self._max_batches:
                break
            yield batch

    def __len__(self) -> int:
        length = len(self._loader)
        return length if self._max_batches is None else min(length, self._max_batches)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader, name)


def prepare_output_directories() -> None:
    for path in (OUTPUT_DIR, GENERATED_CONFIG_DIR, LOG_DIR, PREPROCESSOR_DIR):
        path.mkdir(parents=True, exist_ok=True)


def cleanup_logging_handlers() -> None:
    logger_names = ["", "training"]
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
                handler.close()
            except Exception:
                pass
        logger.propagate = False


def clean_name(value: Any) -> str:
    value = str(value).replace("None", "none")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def serialize_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def init_csv() -> None:
    prepare_output_directories()
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=CSV_COLUMNS).writeheader()


def case_paths(case_name: str) -> dict[str, Path]:
    safe_name = clean_name(case_name)
    return {
        "config_path": GENERATED_CONFIG_DIR / f"{safe_name}.yaml",
        "log_path": LOG_DIR / f"{safe_name}.log",
    }


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
        "config_path": str(paths["config_path"]),
        "log_path": str(paths["log_path"]),
        "result": "PASS" if passed else "FAIL",
        "error": error,
    }
    with RESULTS_CSV.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writerow(row)


def normalize_nullable_fields(cfg: dict[str, Any]) -> dict[str, Any]:
    loader_cfg = cfg.setdefault("train_loader", {})
    dataset_cfg = loader_cfg.setdefault("dataset_config", {})

    if "time_features" in dataset_cfg:
        raise ValueError(
            "time_features is no longer a DatasetConfig field. Configure it "
            "under train_loader.time_features."
        )
    if loader_cfg.get("time_features") is None:
        loader_cfg.pop("time_features", None)
    if dataset_cfg.get("condition_method") is None:
        dataset_cfg.pop("condition_method", None)
    if dataset_cfg.get("lead_times") is None:
        dataset_cfg.pop("lead_times", None)
    return cfg


def validate_generated_config(
    cfg: dict[str, Any],
) -> dict[str, Any]:

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

    runtime_loader_arguments = {
        "shuffle",
        "return_metadata",
    }
    misplaced_arguments = runtime_loader_arguments & set(train_loader_cfg)

    if misplaced_arguments:
        raise ValueError(
            "The following values are runtime arguments to "
            "build_train_loader(), not TrainDataloaderConfig "
            "fields: "
            f"{sorted(misplaced_arguments)}"
        )

    if "time_features" in dataset_cfg:
        raise ValueError(
            "time_features must be configured under "
            "train_loader, not train_loader.dataset_config."
        )

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

    if "model" not in dataset_cfg:
        raise ValueError(
            "Every data-module integration case requires "
            "train_loader.dataset_config.model."
        )

    num_data_workers = train_loader_cfg.get(
        "num_data_workers",
        0,
    )

    if (
        isinstance(num_data_workers, bool)
        or not isinstance(num_data_workers, int)
        or num_data_workers < 0
    ):
        raise ValueError(
            "train_loader.num_data_workers must be a non-negative integer."
        )

    if num_data_workers == 0 and train_loader_cfg.get("prefetch_factor") is not None:
        raise ValueError(
            "train_loader.prefetch_factor must be omitted or "
            "null when num_data_workers=0."
        )

    return cfg


def write_yaml(cfg: dict[str, Any], output_path: Path) -> Path:
    cfg = validate_generated_config(normalize_nullable_fields(deepcopy(cfg)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(cfg, file, sort_keys=False)
    return output_path


def base_config() -> dict[str, Any]:

    cfg = prepare_config(BASE_TRAIN_CONFIG)

    if cfg is None:
        raise ValueError(
            f"The base training configuration is empty: {BASE_TRAIN_CONFIG}"
        )

    cfg = deepcopy(cfg)

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
            "Move train_loader.dataset_config.time_features "
            "to train_loader.time_features."
        )

    train_loader_cfg["batch_size"] = 2
    train_loader_cfg["num_data_workers"] = 0
    train_loader_cfg["num_validation_years"] = 0
    train_loader_cfg["train_years_slice"] = [
        1980,
        1981,
    ]
    train_loader_cfg["load"] = False

    train_loader_cfg.pop(
        "shuffle",
        None,
    )
    train_loader_cfg.pop(
        "return_metadata",
        None,
    )

    train_loader_cfg.pop(
        "prefetch_factor",
        None,
    )

    normalize_nullable_fields(cfg)
    validate_generated_config(cfg)

    return cfg


def load_train_config(config_path: Path) -> TrainConfig:
    data = prepare_config(config_path)
    if data is None:
        raise ValueError(f"Generated config is empty: {config_path}")
    data = validate_generated_config(normalize_nullable_fields(data))
    return dacite.from_dict(
        data_class=TrainConfig,
        data=data,
        config=dacite.Config(strict=True),
    )


def copy_data_config(data_cfg: dict[str, Any]) -> dict[str, Any]:
    copied = deepcopy(data_cfg)

    copied["preprocessing_pipeline"] = deepcopy(data_cfg["preprocessing_pipeline"])
    return copied


def apply_data_combination(
    cfg: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:

    train_loader_cfg = cfg["train_loader"]
    dataset_cfg = train_loader_cfg["dataset_config"]
    model_cfg = dataset_cfg["model"]

    if "time_features" in dataset_cfg:
        raise ValueError(
            "time_features must be configured under train_loader, "
            "not train_loader.dataset_config."
        )

    if "model_ensemble_mean" in params:
        model_cfg["ensemble_mean"] = params["model_ensemble_mean"]

    if params.get(
        "condition_present",
        False,
    ):
        condition_source = dataset_cfg.get(
            "condition",
            model_cfg,
        )
        dataset_cfg["condition"] = copy_data_config(condition_source)

        condition_ensemble_mean = params.get("condition_ensemble_mean")

        if condition_ensemble_mean is not None:
            dataset_cfg["condition"]["ensemble_mean"] = condition_ensemble_mean
    else:
        dataset_cfg.pop(
            "condition",
            None,
        )

    if not params.get(
        "observation_present",
        False,
    ):
        dataset_cfg.pop(
            "observation",
            None,
        )

    condition_method = params.get("condition_method")

    if condition_method is None:
        dataset_cfg.pop(
            "condition_method",
            None,
        )
    else:
        dataset_cfg["condition_method"] = condition_method

    if "time_features" in params:
        time_features = params["time_features"]

        if time_features is None:
            train_loader_cfg.pop(
                "time_features",
                None,
            )
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
            raise ValueError("num_data_workers must be a non-negative integer.")

        train_loader_cfg["num_data_workers"] = num_data_workers

    num_data_workers = train_loader_cfg.get(
        "num_data_workers",
        0,
    )

    if num_data_workers == 0:
        train_loader_cfg.pop(
            "prefetch_factor",
            None,
        )
    else:
        train_loader_cfg.setdefault(
            "prefetch_factor",
            2,
        )

    train_loader_cfg.pop(
        "shuffle",
        None,
    )
    train_loader_cfg.pop(
        "return_metadata",
        None,
    )

    normalize_nullable_fields(cfg)
    validate_generated_config(cfg)

    return cfg


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


def setup_loader(config: TrainConfig) -> Any:
    distributed = Distributed.get_instance()
    config.train_loader.setup_distributed(distributed)
    return config.train_loader.build_train_loader(shuffle=False)


def sample_get(sample: Any, name: str) -> Any:
    if isinstance(sample, dict):
        return sample.get(name)
    return getattr(sample, name, None)


def assert_tensor(name: str, value: Any) -> None:
    if not isinstance(value, torch.Tensor):
        raise AssertionError(f"{name} is not a torch.Tensor: {type(value).__name__}")
    if value.numel() == 0:
        raise AssertionError(f"{name} is empty.")
    if torch.is_floating_point(value) and not torch.isfinite(value).all():
        raise AssertionError(f"{name} contains non-finite values.")


def assert_dataset_getitem_contract(dataset: Any, params: dict[str, Any]) -> None:
    if len(dataset) <= 0:
        raise AssertionError("Dataset is empty.")
    sample = dataset[0]
    input_tensor = sample_get(sample, "input")
    target_tensor = sample_get(sample, "target")
    added_features = sample_get(sample, "added_features")

    assert_tensor("sample.input", input_tensor)
    assert_tensor("sample.target", target_tensor)

    features = params.get("time_features", DEFAULT_TIME_FEATURES)
    if features:
        assert_tensor("sample.added_features", added_features)
    elif (
        added_features is not None and getattr(added_features, "numel", lambda: 0)() > 0
    ):
        raise AssertionError(
            "added_features is populated although time_features is empty."
        )

    if params.get("observation_present"):
        observation_cfg = getattr(getattr(dataset, "config", None), "observation", None)
        if observation_cfg is None:
            raise AssertionError(
                "Observation target was requested but dataset has none."
            )

    if params.get("condition_present"):
        condition_cfg = getattr(getattr(dataset, "config", None), "condition", None)
        if condition_cfg is None:
            raise AssertionError(
                "Explicit condition was requested but dataset has none."
            )

    print(f"dataset_length={len(dataset)}")
    print(f"input_shape={tuple(input_tensor.shape)}")
    print(f"target_shape={tuple(target_tensor.shape)}")
    print(f"expected_input={params.get('expected_input')}")
    print(f"expected_target={params.get('expected_target')}")


def extract_metadata(batch: Any) -> Any:
    return getattr(batch, "metadata", None)


def flatten_metadata(metadata: Any) -> list[Any]:
    if metadata is None:
        return []
    if isinstance(metadata, list):
        return metadata
    if isinstance(metadata, tuple):
        return list(metadata)
    if isinstance(metadata, dict):
        lengths = [
            len(value) for value in metadata.values() if hasattr(value, "__len__")
        ]
        if not lengths:
            return [metadata]
        count = min(lengths)
        return [
            {key: value[index] for key, value in metadata.items()}
            for index in range(count)
        ]
    return [metadata]


def metadata_value(item: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        if isinstance(item, dict) and name in item:
            return item[name]
        if hasattr(item, name):
            return getattr(item, name)
    return None


def canonical_sample_key(item: Any) -> str:
    if isinstance(item, dict):
        return serialize_json(item)
    if hasattr(item, "__dict__"):
        return serialize_json(vars(item))
    return str(item)


def collect_loader_metadata(loader: Any, max_batches: int | None = None) -> list[Any]:
    items: list[Any] = []
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        items.extend(flatten_metadata(extract_metadata(batch)))
    return items


def assert_batch_contract(loader: Any, time_features: list[str] | None) -> None:
    batches = 0
    for batch in LimitedLoader(loader, MAX_BATCHES_PER_CASE):
        assert_tensor("batch.input", getattr(batch, "input", None))
        target = getattr(batch, "target", None)
        if target is not None:
            assert_tensor("batch.target", target)
        features = getattr(batch, "added_features", None)
        if time_features:
            assert_tensor("batch.added_features", features)
        batches += 1
    if batches == 0:
        raise AssertionError("Dataloader yielded no batches.")


def assert_year_slice(loader: Any, year_slice: list[int]) -> None:
    metadata = collect_loader_metadata(loader, MAX_BATCHES_PER_CASE)
    if not metadata:
        raise AssertionError("return_metadata=True is required for year-slice checks.")
    lower, upper = year_slice
    years = []
    for item in metadata:
        value = metadata_value(
            item, (INIT_TIME_DIM, "init_time", "initialization_time")
        )
        if value is None:
            continue
        if hasattr(value, "year"):
            years.append(int(value.year))
        else:
            years.append(
                int(np.asarray(value).astype("datetime64[Y]").astype(int) + 1970)
            )
    if not years:
        raise AssertionError("Could not extract initialization years from metadata.")
    if min(years) < lower or max(years) > upper:
        raise AssertionError(
            f"Loader returned years outside [{lower}, {upper}]: {sorted(set(years))}"
        )


def assert_multiworker_consistency(config: TrainConfig, shuffle: bool) -> None:

    distributed = Distributed.get_instance()
    multi_cfg = config.train_loader
    multi_cfg.setup_distributed(distributed)
    multi_loader = multi_cfg.build_train_loader(return_metadata=True, shuffle=shuffle)
    multi_items = collect_loader_metadata(multi_loader)

    single_cfg = deepcopy(config.train_loader)
    single_cfg.num_data_workers = 0
    if hasattr(single_cfg, "prefetch_factor"):
        single_cfg.prefetch_factor = None
    single_cfg.setup_distributed(distributed)
    single_loader = single_cfg.build_train_loader(return_metadata=True, shuffle=False)
    single_items = collect_loader_metadata(single_loader)

    multi_keys = [canonical_sample_key(item) for item in multi_items]
    single_keys = [canonical_sample_key(item) for item in single_items]
    if sorted(multi_keys) != sorted(single_keys):
        raise AssertionError(
            "Multi-worker loading changed sample coverage. "
            f"single={len(single_keys)}, multi={len(multi_keys)}"
        )
    if len(multi_keys) != len(set(multi_keys)):
        raise AssertionError("Multi-worker loading duplicated one or more samples.")
    if not shuffle and multi_keys != single_keys:
        raise AssertionError("shuffle=False did not preserve deterministic order.")


def add_months_numpy(value: Any, months: int) -> np.datetime64:
    month_value = np.datetime64(value, "M") + np.timedelta64(int(months), "M")
    return month_value.astype("datetime64[D]")


def assert_no_future_leakage(loader: Any, cutoff_year: int) -> None:
    cutoff = np.datetime64(f"{cutoff_year + 1}-01-01", "D")
    inspected = 0
    for batch in loader:
        metadata = flatten_metadata(extract_metadata(batch))
        target_mask = getattr(batch, "target_mask", None)
        input_mask = getattr(batch, "input_mask", None)
        mask = target_mask if target_mask is not None else input_mask
        for index, item in enumerate(metadata):
            init_time = metadata_value(
                item, (INIT_TIME_DIM, "init_time", "initialization_time")
            )
            lead_time = metadata_value(item, (LEAD_TIME_DIM, "lead_time"))
            if init_time is None or lead_time is None:
                continue
            valid_time = add_months_numpy(init_time, int(lead_time))
            should_be_valid = valid_time < cutoff
            if mask is not None:
                sample_mask = (
                    mask[index] if mask.ndim > 0 and len(mask) > index else mask
                )
                is_valid = bool(torch.as_tensor(sample_mask).bool().any().item())
                if is_valid != should_be_valid:
                    raise AssertionError(
                        "Future-leakage mask mismatch for "
                        f"init_time={init_time}, lead_time={lead_time}, "
                        f"valid_time={valid_time}, cutoff={cutoff}, mask_valid={is_valid}."
                    )
            elif not should_be_valid:
                raise AssertionError(
                    "A sample whose valid time exceeds the training cutoff was "
                    "returned without a mask."
                )
            inspected += 1
    if inspected == 0:
        raise AssertionError("No metadata samples were available for mask validation.")


def preprocessor_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*_preprocessing_pipeline.joblib"))


def assert_rank_zero_preprocessor_setup(config: TrainConfig) -> None:
    if PREPROCESSOR_DIR.exists():
        shutil.rmtree(PREPROCESSOR_DIR)
    PREPROCESSOR_DIR.mkdir(parents=True)
    distributed = Distributed.get_instance()
    rank = getattr(distributed, "rank", 0)
    if rank != 0:
        raise RuntimeError("Run the preprocessor writer phase on rank 0.")

    config.train_loader.setup_distributed(distributed, load_path=PREPROCESSOR_DIR)
    _ = config.train_loader.build_train_loader(return_metadata=True, shuffle=False)
    written = preprocessor_files(PREPROCESSOR_DIR)
    if not written:
        raise AssertionError(
            "Rank 0 did not save preprocessing-pipeline artifacts to disk."
        )

    mtimes = {path: path.stat().st_mtime_ns for path in written}

    loader_cfg = deepcopy(config.train_loader)
    loader_cfg.setup_distributed(distributed, load_path=PREPROCESSOR_DIR)
    _ = loader_cfg.build_train_loader(return_metadata=True, shuffle=False)

    after = preprocessor_files(PREPROCESSOR_DIR)
    if [path.name for path in after] != [path.name for path in written]:
        raise AssertionError("Preprocessor load phase changed artifact membership.")
    if {path: path.stat().st_mtime_ns for path in after} != mtimes:
        raise AssertionError(
            "Preprocessor load phase rewrote artifacts instead of loading them."
        )


def build_case_config(params: dict[str, Any]) -> dict[str, Any]:
    cfg = base_config()
    apply_data_combination(cfg, params)
    return cfg


def run_dataset_case(config_path: Path, params: dict[str, Any]) -> None:
    config = load_train_config(config_path)
    distributed = Distributed.get_instance()
    config.train_loader.setup_distributed(distributed)
    loader = config.train_loader.build_train_loader(
        return_metadata=True,
        shuffle=False,
    )
    assert_dataset_getitem_contract(loader.dataset, params)
    assert_batch_contract(loader, params.get("time_features", DEFAULT_TIME_FEATURES))


def run_dataloader_case(config_path: Path, params: dict[str, Any]) -> None:
    config = load_train_config(config_path)
    distributed = Distributed.get_instance()
    config.train_loader.setup_distributed(distributed)

    check = params["check"]
    if check == "multiworker":
        assert_multiworker_consistency(config, params["shuffle"])
        return

    loader = config.train_loader.build_train_loader(
        return_metadata=True,
        shuffle=params.get("shuffle", False),
    )
    assert_batch_contract(loader, params.get("time_features"))
    if check == "year_slice":
        assert_year_slice(loader, params["year_slice"])


def run_future_mask_case(config_path: Path, params: dict[str, Any]) -> None:
    config = load_train_config(config_path)
    distributed = Distributed.get_instance()
    config.train_loader.setup_distributed(distributed)
    loader = config.train_loader.build_train_loader(
        return_metadata=True,
        shuffle=False,
    )
    assert_no_future_leakage(loader, cutoff_year=params["cutoff_year"])


def run_preprocessor_case(config_path: Path, params: dict[str, Any]) -> None:
    del params
    config = load_train_config(config_path)
    assert_rank_zero_preprocessor_setup(config)


def run_inference_case(config_path: Path, params: dict[str, Any]) -> None:
    from cccma_ppp.inference.inference_configs import InferenceConfig

    train_config = load_train_config(config_path)
    inference_data = prepare_config(params["inference_config_path"])
    if inference_data is None:
        raise ValueError("Inference configuration is empty.")
    inference_config = dacite.from_dict(
        data_class=InferenceConfig,
        data=inference_data,
        config=dacite.Config(strict=True),
    )
    inference_config.inference_loader.read_configs_from_train(train_config.train_loader)
    if (
        inference_config.inference_loader.time_features
        != train_config.train_loader.time_features
    ):
        raise AssertionError("Inference did not preserve training time_features.")
    distributed = Distributed.get_instance()
    inference_config.inference_loader.setup_distributed(distributed)
    loader = inference_config.inference_loader.build_inference_loader(
        return_metadata=True
    )
    assert_batch_contract(loader, train_config.train_loader.time_features)


def run_task(task: dict[str, Any]) -> tuple[bool, str]:
    paths = case_paths(task["case_name"])
    for path in paths.values():
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()

    try:
        params = task["params"]
        cfg = task["cfg_builder"]()
        write_yaml(cfg, paths["config_path"])
        print(f"\nStarting {task['suite']}:{task['case_name']}", flush=True)
        print(f"Log: {paths['log_path']}", flush=True)
        run_silenced(
            partial(task["runner"], paths["config_path"], params),
            paths["log_path"],
        )
        passed, error = True, ""
    except Exception as exc:
        passed = False
        error = f"{type(exc).__name__}: {exc}"
        append_exception_to_log(paths["log_path"])
        cleanup_logging_handlers()

    log_result(task=task, paths=paths, passed=passed, error=error)
    gc.collect()
    return passed, error


def add_data_combination_tasks(tasks: list[dict[str, Any]]) -> None:
    if not RUN_DATA_CONFIG_COMBINATIONS:
        return
    for case in DATA_CONFIG_COMBINATIONS:
        params = {
            **deepcopy(case),
            "time_features": deepcopy(DEFAULT_TIME_FEATURES),
            "year_slice": [1980, 1981],
            "num_data_workers": 0,
        }
        tasks.append(
            {
                "suite": "data_config_combination",
                "case_name": params["name"],
                "params": params,
                "cfg_builder": partial(build_case_config, params),
                "runner": run_dataset_case,
            }
        )


def add_getitem_tasks(tasks: list[dict[str, Any]]) -> None:
    if not RUN_DATASET_GETITEM_CASES:
        return
    for case in DATA_CONFIG_COMBINATIONS:
        params = {
            **deepcopy(case),
            "name": f"getitem_{case['name']}",
            "time_features": deepcopy(DEFAULT_TIME_FEATURES),
            "year_slice": [1980, 1981],
            "num_data_workers": 0,
        }
        tasks.append(
            {
                "suite": "dataset_getitem",
                "case_name": params["name"],
                "params": params,
                "cfg_builder": partial(build_case_config, params),
                "runner": run_dataset_case,
            }
        )


def add_dataloader_tasks(tasks: list[dict[str, Any]]) -> None:
    if not RUN_TRAIN_DATALOADER_CASES:
        return
    base_combination = deepcopy(DATA_CONFIG_COMBINATIONS[-2])
    for case in DATALOADER_CASES:
        if case["check"] == "multiworker" and not RUN_MULTIWORKER_CASES:
            continue
        params = {**base_combination, **deepcopy(case)}
        tasks.append(
            {
                "suite": "train_dataloader",
                "case_name": params["name"],
                "params": params,
                "cfg_builder": partial(build_case_config, params),
                "runner": run_dataloader_case,
            }
        )


def add_future_mask_tasks(tasks: list[dict[str, Any]]) -> None:
    if not RUN_FUTURE_MASK_CASES:
        return
    params = {
        **deepcopy(DATA_CONFIG_COMBINATIONS[-2]),
        "name": "future_leakage_mask_december_cutoff",
        "description": (
            "For a final training year of 2020, samples initialized in December "
            "2019 must mask lead months whose valid times reach 2021 or later."
        ),
        "year_slice": [2019, 2020],
        "cutoff_year": 2020,
        "num_data_workers": 0,
        "shuffle": False,
        "time_features": deepcopy(DEFAULT_TIME_FEATURES),
        "lead_months": deepcopy(FULL_LEAD_MONTHS),
        "expected_result": "PASS",
    }
    tasks.append(
        {
            "suite": "future_mask",
            "case_name": params["name"],
            "params": params,
            "cfg_builder": partial(build_case_config, params),
            "runner": run_future_mask_case,
        }
    )


def add_preprocessor_tasks(tasks: list[dict[str, Any]]) -> None:
    if not RUN_DISTRIBUTED_PREPROCESSOR_CASES:
        return
    params = {
        **deepcopy(DATA_CONFIG_COMBINATIONS[-2]),
        "name": "rank_zero_fits_other_ranks_load_preprocessors",
        "description": (
            "Rank 0 fits and saves preprocessors. Subsequent setup loads the "
            "same artifacts without rewriting them."
        ),
        "year_slice": [1980, 1981],
        "num_data_workers": 0,
        "shuffle": False,
        "time_features": deepcopy(DEFAULT_TIME_FEATURES),
        "expected_result": "PASS",
    }
    tasks.append(
        {
            "suite": "distributed_preprocessing",
            "case_name": params["name"],
            "params": params,
            "cfg_builder": partial(build_case_config, params),
            "runner": run_preprocessor_case,
        }
    )


def add_inference_tasks(tasks: list[dict[str, Any]]) -> None:
    if not RUN_INFERENCE_DATALOADER_CASES:
        return
    inference_path = ROOT / "scripts" / "integration_suite_inference_config.yaml"
    if not inference_path.is_file():
        print(
            "Skipping inference loader cases because the base inference config "
            f"does not exist: {inference_path}",
            flush=True,
        )
        return
    params = {
        **deepcopy(DATA_CONFIG_COMBINATIONS[-2]),
        "name": "inference_reuses_training_data_contract",
        "description": (
            "Inference copies the training dataset contract and time_features, "
            "then produces iterable masked batches."
        ),
        "year_slice": [1980, 1981],
        "num_data_workers": 0,
        "shuffle": False,
        "time_features": deepcopy(DEFAULT_TIME_FEATURES),
        "inference_config_path": inference_path,
        "expected_result": "PASS",
    }
    tasks.append(
        {
            "suite": "inference_dataloader",
            "case_name": params["name"],
            "params": params,
            "cfg_builder": partial(build_case_config, params),
            "runner": run_inference_case,
        }
    )


def validate_tasks(tasks: list[dict[str, Any]]) -> None:
    names: set[str] = set()
    for index, task in enumerate(tasks):
        missing = {"suite", "case_name", "params", "cfg_builder", "runner"} - set(task)
        if missing:
            raise ValueError(f"Task {index} is missing fields: {sorted(missing)}")
        if task["case_name"] in names:
            raise ValueError(f"Duplicate case name: {task['case_name']}")
        names.add(task["case_name"])
        if not callable(task["cfg_builder"]) or not callable(task["runner"]):
            raise TypeError(f"Task callables are invalid: {task['case_name']}")


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
            unexpected_failures.append(
                f"{task['case_name']}: failed for the wrong reason: {error}"
            )
        else:
            expected_failures.append(task["case_name"])
    elif not passed:
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
        file.write(f"results_csv: {RESULTS_CSV}\n")
        file.write(f"log_dir: {LOG_DIR}\n")
        for title, values in (
            ("unexpected_failures", unexpected_failures),
            ("expected_failures", expected_failures),
            ("unexpected_passes", unexpected_passes),
        ):
            if values:
                file.write(f"\n{title}:\n")
                for value in values:
                    file.write(f"- {value}\n")


def main() -> None:
    prepare_output_directories()
    cleanup_logging_handlers()
    init_csv()

    tasks: list[dict[str, Any]] = []
    add_data_combination_tasks(tasks)
    add_getitem_tasks(tasks)
    add_dataloader_tasks(tasks)
    add_future_mask_tasks(tasks)
    add_preprocessor_tasks(tasks)
    add_inference_tasks(tasks)
    validate_tasks(tasks)

    if MAX_CASES is not None:
        tasks = tasks[:MAX_CASES]

    unexpected_failures: list[str] = []
    expected_failures: list[str] = []
    unexpected_passes: list[str] = []

    with tqdm(total=len(tasks), desc="data_modules", unit="case") as progress:
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

    cleanup_logging_handlers()
    write_summary(
        tasks,
        unexpected_failures,
        expected_failures,
        unexpected_passes,
    )

    if unexpected_failures or unexpected_passes:
        problems = []
        if unexpected_failures:
            problems.append(f"{len(unexpected_failures)} unexpected failure(s)")
        if unexpected_passes:
            problems.append(f"{len(unexpected_passes)} unexpected pass(es)")
        raise RuntimeError(
            "Data-module integration suite completed with "
            + " and ".join(problems)
            + f". See {RESULTS_CSV} and {LOG_DIR}."
        )

    print("Data-module integration suite completed successfully.")
    print(f"Results: {RESULTS_CSV}")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Logs: {LOG_DIR}")


if __name__ == "__main__":
    main()
