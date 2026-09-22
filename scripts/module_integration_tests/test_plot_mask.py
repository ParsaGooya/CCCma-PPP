from __future__ import annotations

import shutil
import warnings
from functools import partial
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from integrationtestslib import (
    DataModulesUtils,
    LoggingUtils,
    OutputFetcher,
    ResultsManager,
)
from cccma_ppp.configs import required_sample_dimensions
from cccma_ppp.generic.runtime import RuntimeContext
from cccma_ppp.train.train_configs import prepare_config
from cccma_ppp.generic.distributed import Distributed

warnings.filterwarnings("ignore")

ROOT = Path("/fs/site7/eccc/crd/cccma/users/rna002/CCCma-PPP/scripts")
CONFIG_DIR = ROOT / "module_integration_tests/plot_test_configs"
OUTPUT_DIR = ROOT / "../../output" / "plot_mask_test_results"
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
    from cccma_ppp.train.train_configs import TrainConfig

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

    # Validate dataset format
    DataModulesUtils.test_dataset_getitem_format(
        loader.dataset, init_time_dim=INIT_TIME_DIM, lead_time_dim=LEAD_TIME_DIM
    )

    # Check and plot premade mask
    mask_check = DataModulesUtils.check_premade_mask_in_dataset(loader.dataset)
    if mask_check["has_mask_attribute"]:
        fetcher = OutputFetcher(case_dir, max_plot_channels=MAX_PLOT_CHANNELS)
        fetcher.plot_premade_mask(loader.dataset)
        print(f"  Mask plotted: {case_dir / 'premade_mask.png'}")
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
    if not CONFIG_DIR.exists():
        raise ValueError(f"Config directory does not exist: {CONFIG_DIR}")

    # Find all YAML configs in the directory
    config_files = sorted(CONFIG_DIR.glob("*.yaml"))
    if not config_files:
        raise ValueError(f"No YAML configs found in {CONFIG_DIR}")

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
