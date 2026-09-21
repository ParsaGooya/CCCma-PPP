from __future__ import annotations

import shutil
import warnings
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, List, Optional

import torch
from tqdm import tqdm

from integrationtestslib import (
    ConfigManager,
    DataModulesUtils,
    LimitedLoader,
    LoggingUtils,
    OutputFetcher,
    PathManager,
    ResultsManager,
    StringUtils,
    TensorUtils,
)
from cccma_ppp.configs import required_sample_dimensions
from cccma_ppp.generic.runtime import RuntimeContext
from cccma_ppp.train.train_configs import prepare_config

warnings.filterwarnings("ignore")

MODE = "deterministic"
ROOT = Path("/fs/site7/eccc/crd/cccma/users/rna002/CCCma-PPP/scripts")
BASE_TRAIN_CONFIG = f"scripts/{MODE}_config.yaml"

OUTPUT_DIR = ROOT / "../../output" / f"data_modules_integration_test_results_{MODE}"
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


@dataclass
class CaseParams:
    """Parameters for a single data_modules test case"""

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
        return {k: v for k, v in asdict(self).items() if v is not None}


class DataModulesConfigManager(ConfigManager):
    """Config manager specialized for data_modules"""

    @staticmethod
    def base_config() -> dict[str, Any]:
        """Load base training config"""
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

        for key in [
            "shuffle",
            "return_metadata",
            "return_spatial_mask",
            "prefetch_factor",
        ]:
            train_loader_cfg.pop(key, None)

        return cfg

    @staticmethod
    def apply_case_config(
        cfg: dict[str, Any], params: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply test case parameters to config"""
        cfg = deepcopy(cfg)
        train_loader_cfg = cfg["train_loader"]
        dataset_cfg = train_loader_cfg["dataset_config"]

        if not params.get("model_present", True):
            raise ValueError("Model is required")

        if "model" not in dataset_cfg:
            raise ValueError("Base config missing model config")

        model_cfg = dataset_cfg["model"]
        if (
            "model_ensemble_mean" in params
            and params["model_ensemble_mean"] is not None
        ):
            model_cfg["ensemble_mean"] = params["model_ensemble_mean"]

        if params.get("condition_present", False):
            if "condition" not in dataset_cfg:
                condition_cfg = deepcopy(model_cfg)
            else:
                condition_cfg = dataset_cfg["condition"]

            if "condition_path" in params and params["condition_path"]:
                condition_cfg = DataModulesConfigManager._apply_external_condition_path(
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

    @staticmethod
    def _apply_external_condition_path(
        condition_cfg: dict[str, Any],
        *,
        path: str,
        has_ensemble: bool,
        condition_kind: str,
    ) -> dict[str, Any]:
        cfg = deepcopy(condition_cfg)
        cfg = DataModulesConfigManager._replace_paths_in_dict(cfg, path)

        if not has_ensemble:
            cfg = DataModulesConfigManager._remove_ensemble_fields(cfg)

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

    @staticmethod
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
                        result[key] = (
                            [new_path] if isinstance(value, list) else new_path
                        )
                    else:
                        result[key] = _recurse(value)
                return result
            elif isinstance(obj, list):
                return [_recurse(item) for item in obj]
            return obj

        return _recurse(cfg)

    @staticmethod
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


class TestScenarios:
    """All test scenarios for data_modules"""

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

    MULTIWORKER_NO_SHUFFLE = CaseParams(
        name="train_multiworker_no_shuffle",
        model_present=True,
        condition_present=True,
        observation_present=True,
        year_slice=[1980, 2007],
        num_data_workers=2,
        shuffle=False,
        time_features=deepcopy(DEFAULT_TIME_FEATURES),
    )

    MULTIWORKER_WITH_SHUFFLE = CaseParams(
        name="train_multiworker_shuffle",
        model_present=True,
        condition_present=True,
        observation_present=True,
        year_slice=[1980, 2007],
        num_data_workers=2,
        shuffle=True,
        time_features=deepcopy(DEFAULT_TIME_FEATURES),
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
            cls.MULTIWORKER_NO_SHUFFLE,
            cls.MULTIWORKER_WITH_SHUFFLE,
        ]


def reset_runtime_context() -> None:
    RuntimeContext.GLOBAL_EXP_DIR = None


def call_with_supported_arguments(function: Callable[..., Any], **kwargs: Any) -> Any:
    """Call function with only supported kwargs"""
    import inspect

    signature = inspect.signature(function)
    supported = {
        name: value for name, value in kwargs.items() if name in signature.parameters
    }
    return function(**supported)


def setup_loader(
    config: Any,
    config_path: Path,
    *,
    return_metadata: bool = False,
    return_spatial_mask: bool = True,
    shuffle: bool = False,
) -> Any:
    """Set up data loader"""
    from cccma_ppp.generic.distributed import Distributed

    case_dir = Path(config_path).resolve().parent
    case_dir.mkdir(parents=True, exist_ok=True)
    RuntimeContext.GLOBAL_EXP_DIR = str(case_dir)

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


def load_train_config(config_path: Path) -> Any:
    """Load training config from file"""
    import dacite
    from cccma_ppp.train.train_configs import TrainConfig

    data = prepare_config(config_path)
    if data is None:
        raise ValueError(f"Config is empty: {config_path}")

    return dacite.from_dict(
        data_class=TrainConfig,
        data=data,
        config=dacite.Config(strict=True),
    )


def build_result_row(
    *,
    case_name: str,
    suite: str,
    params: dict[str, Any],
    paths: dict[str, Path],
    passed: bool,
    error: str,
) -> dict[str, Any]:
    """Build a result row for CSV"""
    from datetime import datetime

    return {
        "timestamp": datetime.now().isoformat(),
        "suite": suite,
        "case_name": case_name,
        "description": None,
        "model_present": params.get("model_present"),
        "condition_present": params.get("condition_present"),
        "observation_present": params.get("observation_present"),
        "condition_kind": params.get("condition_kind"),
        "condition_method": params.get("condition_method"),
        "time_features": StringUtils.serialize_json(params.get("time_features")),
        "year_slice": StringUtils.serialize_json(params.get("year_slice")),
        "num_data_workers": params.get("num_data_workers"),
        "case_dir": str(paths["case_dir"]),
        "config_path": str(paths["config_path"]),
        "log_path": str(paths["log_path"]),
        "preprocessor_dir": str(paths.get("preprocessor_dir")),
        "plot_dir": str(paths["plot_dir"]),
        "result": "PASS" if passed else "FAIL",
        "error": error,
    }


def run_dataset_case(config_path: Path, params: dict[str, Any]) -> None:
    """Test dataset.__getitem__ contract"""
    config = load_train_config(config_path)
    loader = setup_loader(
        config,
        config_path,
        return_metadata=True,
        return_spatial_mask=True,
        shuffle=False,
    )

    # Validate dataset format
    DataModulesUtils.test_dataset_getitem_format(
        loader.dataset, init_time_dim=INIT_TIME_DIM, lead_time_dim=LEAD_TIME_DIM
    )

    paths = PathManager(CASES_DIR, OUTPUT_DIR).case_paths(params["name"])
    all_metadata = []
    batch_sizes = []

    fetcher = OutputFetcher(paths["case_dir"], max_plot_channels=MAX_PLOT_CHANNELS)

    # Check premade mask
    mask_check = DataModulesUtils.check_premade_mask_in_dataset(loader.dataset)

    if mask_check["has_mask_attribute"]:
        fetcher.extract_and_save_premade_mask(loader.dataset)
        fetcher.plot_premade_mask(loader.dataset)

    for i, batch in enumerate(LimitedLoader(loader, MAX_BATCHES_PER_CASE)):
        # Plot first batch
        if i == 0:
            fetcher.plot_batch(batch)

            # Compare masks if premade exists
            if mask_check["has_mask_attribute"]:
                fetcher.compare_premade_vs_nan_masks(loader.dataset, batch)

            # Export first batch
            metadata = TensorUtils.extract_metadata(batch)
            batch_metadata = TensorUtils.flatten_metadata(metadata)
            if batch_metadata:
                fetcher.export_masks_and_data(
                    batch,
                    metadata=batch_metadata,
                    batch_index=0,
                )

        # Collect metadata from all batches
        metadata = TensorUtils.extract_metadata(batch)
        batch_metadata = TensorUtils.flatten_metadata(metadata)
        if batch_metadata:
            all_metadata.extend(batch_metadata)
            batch_sizes.append(len(batch_metadata))

    # Save metadata
    if all_metadata:
        fetcher.save_batch_metadata(
            all_metadata,
            output_path=paths["metadata_path"],
            batch_sizes=batch_sizes,
            batch_index=0,
        )

        # Save statistics
        results_manager = ResultsManager(RESULTS_CSV, SUMMARY_PATH, CSV_COLUMNS)
        results_manager.compute_and_save_metadata_statistics(
            all_metadata,
            paths["metadata_statistics_path"],
            init_time_dim=INIT_TIME_DIM,
            lead_time_dim=LEAD_TIME_DIM,
        )

    print(f"Dataset case completed: {len(all_metadata)} samples")


def run_dataloader_case(config_path: Path, params: dict[str, Any]) -> None:
    """Test dataloader with different configurations"""
    config = load_train_config(config_path)
    loader = setup_loader(
        config,
        config_path,
        return_metadata=True,
        return_spatial_mask=True,
        shuffle=params.get("shuffle", False),
    )

    paths = PathManager(CASES_DIR, OUTPUT_DIR).case_paths(params["name"])
    all_metadata = []
    batch_sizes = []

    fetcher = OutputFetcher(paths["case_dir"], max_plot_channels=MAX_PLOT_CHANNELS)

    # Check premade mask
    mask_check = DataModulesUtils.check_premade_mask_in_dataset(loader.dataset)
    if mask_check["has_mask_attribute"]:
        fetcher.extract_and_save_premade_mask(loader.dataset)
        fetcher.plot_premade_mask(loader.dataset)

    for i, batch in enumerate(LimitedLoader(loader, MAX_BATCHES_PER_CASE)):
        # Plot first batch
        if i == 0:
            fetcher.plot_batch(batch)

            if mask_check["has_mask_attribute"]:
                fetcher.compare_premade_vs_nan_masks(loader.dataset, batch)

            metadata = TensorUtils.extract_metadata(batch)
            batch_metadata = TensorUtils.flatten_metadata(metadata)
            if batch_metadata:
                fetcher.export_masks_and_data(
                    batch,
                    metadata=batch_metadata,
                    batch_index=0,
                )

        # Collect metadata from all batches
        metadata = TensorUtils.extract_metadata(batch)
        batch_metadata = TensorUtils.flatten_metadata(metadata)
        if batch_metadata:
            all_metadata.extend(batch_metadata)
            batch_sizes.append(len(batch_metadata))

    # Save metadata
    if all_metadata:
        fetcher.save_batch_metadata(
            all_metadata,
            output_path=paths["metadata_path"],
            batch_sizes=batch_sizes,
            batch_index=0,
        )

        results_manager = ResultsManager(RESULTS_CSV, SUMMARY_PATH, CSV_COLUMNS)
        results_manager.compute_and_save_metadata_statistics(
            all_metadata,
            paths["metadata_statistics_path"],
            init_time_dim=INIT_TIME_DIM,
            lead_time_dim=LEAD_TIME_DIM,
        )

    print(f"Dataloader case completed: {len(all_metadata)} samples")


def run_task(task: dict[str, Any]) -> tuple[bool, str]:
    """Run a single test task"""
    path_manager = PathManager(CASES_DIR, OUTPUT_DIR)
    paths = path_manager.case_paths(task["case_name"])
    case_dir = paths["case_dir"]

    if case_dir.exists():
        shutil.rmtree(case_dir)

    case_dir.mkdir(parents=True, exist_ok=True)

    passed = False
    error = ""

    try:
        params = task["params"]
        suite = task["suite"]

        cfg = DataModulesConfigManager.apply_case_config(
            DataModulesConfigManager.base_config(), params
        )
        DataModulesConfigManager.write_yaml(cfg, paths["config_path"])

        runner = task["runner"]
        LoggingUtils.run_silenced(
            partial(runner, paths["config_path"], params),
            paths["log_path"],
        )

        passed = True

    except Exception as exc:
        error = f"{type(exc).__name__}: {str(exc)[:200]}"
        LoggingUtils.append_exception(paths["log_path"])
        LoggingUtils.cleanup_handlers()

    finally:
        reset_runtime_context()
        try:
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # Log result
    results_manager = ResultsManager(RESULTS_CSV, SUMMARY_PATH, CSV_COLUMNS)
    row = build_result_row(
        case_name=task["case_name"],
        suite=suite,
        params=task["params"],
        paths=paths,
        passed=passed,
        error=error,
    )
    results_manager.log_result(row)

    return passed, error


def build_test_suite() -> list[dict[str, Any]]:
    """Build entire test suite"""
    tasks = []

    if RUN_DATA_CONFIG_COMBINATIONS:
        for case in TestScenarios.data_combinations():
            tasks.append(
                {
                    "suite": "data_config_combination",
                    "case_name": case.name,
                    "params": case.to_dict(),
                    "runner": run_dataset_case,
                }
            )

    if RUN_EXTERNAL_CONDITION_CASES:
        for case in TestScenarios.external_conditions():
            tasks.append(
                {
                    "suite": "external_condition",
                    "case_name": case.name,
                    "params": case.to_dict(),
                    "runner": run_dataset_case,
                }
            )

    if RUN_TRAIN_DATALOADER_CASES:
        for case in TestScenarios.dataloaders():
            tasks.append(
                {
                    "suite": "train_dataloader",
                    "case_name": case.name,
                    "params": case.to_dict(),
                    "runner": run_dataloader_case,
                }
            )

    return tasks


def main() -> None:
    """Main entry point"""
    path_manager = PathManager(CASES_DIR, OUTPUT_DIR)
    path_manager.prepare_output_directories()

    LoggingUtils.cleanup_handlers()

    results_manager = ResultsManager(RESULTS_CSV, SUMMARY_PATH, CSV_COLUMNS)
    results_manager.init_csv()

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
                    print(f"  ✗ {task['case_name']}: {error[:80]}")

                pbar.update(1)

    finally:
        reset_runtime_context()
        LoggingUtils.cleanup_handlers()

    results_manager.write_summary(len(tasks), passed_count, failed_count)

    print(f"\n{'=' * 80}")
    print(f"Integration test suite: {passed_count} passed, {failed_count} failed")
    print(f"Results: {RESULTS_CSV}")
    print(f"Cases: {CASES_DIR}")
    print(f"{'=' * 80}")

    if failed_count > 0:
        raise RuntimeError(f"{failed_count} test(s) failed")


if __name__ == "__main__":
    main()
