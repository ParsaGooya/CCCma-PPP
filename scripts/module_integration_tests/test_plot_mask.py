from __future__ import annotations

import shutil
import warnings
from functools import partial
from pathlib import Path
from typing import Any
import copy

import numpy as np
import torch
import yaml
from tqdm import tqdm

from integrationtestslib import (
    DataModulesUtils,
    LoggingUtils,
    OutputFetcher,
    ResultsManager,
)
from S2D4P.configs import required_sample_dimensions
from S2D4P.generic.runtime import RuntimeContext
from S2D4P.train.train_configs import prepare_config
from S2D4P.generic.distributed import Distributed

warnings.filterwarnings("ignore")

ROOT = Path("/fs/site7/eccc/crd/cccma/users/rna002/CCCma-PPP/scripts")
BASE_CONFIG_FILE = ROOT / "module_integration_tests/deterministic_config.yaml"
OUTPUT_DIR = ROOT / "../../output" / "plot_mask_test_results"
CONFIG_DIR = OUTPUT_DIR / "plot_test_configs"
CASES_DIR = OUTPUT_DIR / "cases"
RESULTS_CSV = OUTPUT_DIR / "integration_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary.txt"

INIT_TIME_DIM, LEAD_TIME_DIM = required_sample_dimensions

MAX_PLOT_CHANNELS = 16

CSV_COLUMNS = [
    "timestamp",
    "suite",
    "case_name",
    "config_file",
    "case_dir",
    "log_path",
    "result",
    "error",
]


def load_base_config(config_file: Path) -> dict:
    """Load base configuration from YAML file"""
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def create_lead_times_list(values: list[int]) -> dict:
    """Create lead_times configuration with list_lead_times"""
    return {"lead_times": {"list_lead_times": values}}


def create_lead_times_range(start: int, end: int) -> dict:
    """Create lead_times configuration with start and end"""
    return {"lead_times": {"start": start, "end": end}}


def generate_configs(
    base_config: dict,
    lead_time_configs: list[dict],
    output_dir: Path,
) -> list[Path]:
    """
    Generate YAML configuration files from lead time configurations.

    Returns:
        List of paths to generated config files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    for config in lead_time_configs:
        config_name = config["name"]
        lead_times = config["lead_times"]

        # Deep copy base config to avoid modifying original
        full_config = copy.deepcopy(base_config)

        # Insert lead_times into dataset_config
        full_config["train_loader"]["dataset_config"]["lead_times"] = lead_times

        # Write to file
        config_path = output_dir / f"{config_name}.yaml"
        with open(config_path, "w") as f:
            yaml.dump(full_config, f, default_flow_style=False, sort_keys=False)

        generated_files.append(config_path)
        print(f"Generated: {config_path}")

    return generated_files


def reset_runtime_context() -> None:
    RuntimeContext.GLOBAL_EXP_DIR = None


def call_with_supported_arguments(function: Any, **kwargs: Any) -> Any:
    """Call function with only supported kwargs"""
    import inspect

    signature = inspect.signature(function)
    supported = {
        name: value for name, value in kwargs.items() if name in signature.parameters
    }
    return function(**supported)


def setup_train_loader(
    config: Any,
    config_path: Path,
) -> Any:
    """Set up training data loader"""
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
        return_metadata=True,
        return_spatial_mask=True,
        shuffle=False,
    )


def load_train_config(config_path: Path) -> Any:
    """Load training config from file"""
    import dacite
    from S2D4P.train.train_configs import TrainConfig

    data = prepare_config(config_path)
    if data is None:
        raise ValueError(f"Config is empty: {config_path}")

    return dacite.from_dict(
        data_class=TrainConfig,
        data=data,
        config=dacite.Config(strict=False),
    )


def run_plot_mask_case(config_file: Path, case_dir: Path) -> None:
    """Load config and plot premade mask"""
    # Copy config to case dir
    config_path = case_dir / "config.yaml"
    shutil.copy(config_file, config_path)

    config = load_train_config(config_path)
    loader = setup_train_loader(config, config_path)

    # Extract lead_times from config for plotting
    lead_times = None
    lead_times_cfg = config.train_loader.dataset_config.lead_times

    print(f"DEBUG: lead_times_cfg type = {type(lead_times_cfg)}")
    print(f"DEBUG: lead_times_cfg = {lead_times_cfg}")

    # Handle different types of lead_times_cfg
    if isinstance(lead_times_cfg, np.ndarray):
        # It's a numpy array (from list_lead_times)
        lead_times = lead_times_cfg.tolist()
        print(f"DEBUG: Extracted from numpy.ndarray: {lead_times}")
    elif isinstance(lead_times_cfg, (list, tuple)):
        # It's already a list or tuple (from list_lead_times)
        lead_times = list(lead_times_cfg)
        print(f"DEBUG: Extracted from list/tuple: {lead_times}")
    elif isinstance(lead_times_cfg, dict):
        # It's a dict with start/end
        if "list_lead_times" in lead_times_cfg:
            lt = lead_times_cfg["list_lead_times"]
            if isinstance(lt, np.ndarray):
                lead_times = lt.tolist()
            else:
                lead_times = list(lt)
            print(f"DEBUG: Extracted from dict['list_lead_times']: {lead_times}")
        elif "start" in lead_times_cfg and "end" in lead_times_cfg:
            lead_times = list(range(lead_times_cfg["start"], lead_times_cfg["end"] + 1))
            print(f"DEBUG: Extracted from dict start/end: {lead_times}")
    else:
        # It's an object with attributes
        if hasattr(lead_times_cfg, "list_lead_times"):
            lt = lead_times_cfg.list_lead_times
            if isinstance(lt, np.ndarray):
                lead_times = lt.tolist()
            else:
                lead_times = list(lt) if hasattr(lt, "__iter__") else lt
            print(f"DEBUG: Extracted from .list_lead_times: {lead_times}")
        elif hasattr(lead_times_cfg, "start") and hasattr(lead_times_cfg, "end"):
            lead_times = list(range(lead_times_cfg.start, lead_times_cfg.end + 1))
            print(f"DEBUG: Extracted from .start/.end: {lead_times}")

    print(f"DEBUG: Final lead_times = {lead_times}")

    # Validate dataset format
    DataModulesUtils.test_dataset_getitem_format(
        loader.dataset, init_time_dim=INIT_TIME_DIM, lead_time_dim=LEAD_TIME_DIM
    )

    # Check and plot premade mask
    mask_check = DataModulesUtils.check_premade_mask_in_dataset(loader.dataset)
    if mask_check["has_mask_attribute"]:
        fetcher = OutputFetcher(case_dir, max_plot_channels=MAX_PLOT_CHANNELS)
        print(f"DEBUG: Calling plot_premade_mask with lead_times={lead_times}")
        fetcher.plot_premade_mask(loader.dataset, lead_times=lead_times)
        print(f"  Mask plotted: {case_dir / 'raw_data' / 'masks' / 'premade_mask.png'}")
    else:
        print("  No premade mask in dataset")

    print("  Case completed")


def run_task(config_file: Path) -> tuple[bool, str]:
    """Run a single test case with config from file"""
    case_name = config_file.stem
    case_dir = CASES_DIR / case_name
    log_path = case_dir / "run.log"

    if case_dir.exists():
        shutil.rmtree(case_dir)

    case_dir.mkdir(parents=True, exist_ok=True)

    passed = False
    error = ""

    try:
        LoggingUtils.run_silenced(
            partial(
                run_plot_mask_case,
                config_file=config_file,
                case_dir=case_dir,
            ),
            log_path,
        )

        passed = True

    except Exception as exc:
        error = f"{type(exc).__name__}: {str(exc)[:200]}"
        LoggingUtils.append_exception(log_path)
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

    results_manager = ResultsManager(RESULTS_CSV, SUMMARY_PATH, CSV_COLUMNS)
    row = {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "suite": "plot_mask",
        "case_name": case_name,
        "config_file": str(config_file),
        "case_dir": str(case_dir),
        "log_path": str(log_path),
        "result": "PASS" if passed else "FAIL",
        "error": error,
    }
    results_manager.log_result(row)

    return passed, error


def main() -> None:
    """Main entry point"""
    # Load base configuration
    if not BASE_CONFIG_FILE.exists():
        raise ValueError(f"Base config file not found: {BASE_CONFIG_FILE}")

    print(f"Loading base config from: {BASE_CONFIG_FILE}")
    base_config = load_base_config(BASE_CONFIG_FILE)

    # Define lead time configurations as a grid
    lead_time_configs = [
        {"name": "case1", "lead_times": {"list_lead_times": [3, 15, 45, 90]}},
        {"name": "case2", "lead_times": {"start": 7, "end": 84}},
        {"name": "case3", "lead_times": {"list_lead_times": [2, 21, 60, 120]}},
        {"name": "case4", "lead_times": {"start": 1, "end": 30}},
        {"name": "case5", "lead_times": {"list_lead_times": [5, 10, 20, 75, 150]}},
        {"name": "case6", "lead_times": {"start": 14, "end": 180}},
        {"name": "case7", "lead_times": {"list_lead_times": [1, 14, 45]}},
        {"name": "case8", "lead_times": {"start": 1, "end": 365}},
        {"name": "case9", "lead_times": {"list_lead_times": [6, 12, 30, 60, 90, 180]}},
    ]

    # Generate configs
    print("Generating configurations...")
    config_files = generate_configs(base_config, lead_time_configs, CONFIG_DIR)

    if not config_files:
        raise ValueError("No configs were generated")

    CASES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    LoggingUtils.cleanup_handlers()

    results_manager = ResultsManager(RESULTS_CSV, SUMMARY_PATH, CSV_COLUMNS)
    results_manager.init_csv()

    passed_count = 0
    failed_count = 0

    try:
        with tqdm(
            total=len(config_files),
            desc="plot_mask_tests",
            unit="case",
            dynamic_ncols=True,
            leave=True,
        ) as pbar:
            for config_file in config_files:
                case_name = config_file.stem
                pbar.set_postfix_str(f"{case_name}")

                passed, error = run_task(config_file)
                if passed:
                    passed_count += 1
                else:
                    failed_count += 1
                    print(f"  {case_name}: {error[:80]}")

                pbar.update(1)

    finally:
        reset_runtime_context()
        LoggingUtils.cleanup_handlers()

    results_manager.write_summary(len(config_files), passed_count, failed_count)

    print(f"\n{'=' * 80}")
    print(f"Plot mask test suite: {passed_count} passed, {failed_count} failed")
    print(f"Results: {RESULTS_CSV}")
    print(f"Cases: {CASES_DIR}")
    print(f"{'=' * 80}")

    if failed_count > 0:
        raise RuntimeError(f"{failed_count} test(s) failed")


if __name__ == "__main__":
    main()
