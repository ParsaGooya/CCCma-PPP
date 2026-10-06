from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import torch

from integrationtestslib import (
    DataModulesUtils,
    LimitedLoader,
    LoggingUtils,
    OutputFetcher,
    PathManager,
    ResultsManager,
    StringUtils,
    TensorUtils,
    ConfigManager,
)
from cccma_ppp.configs import required_sample_dimensions
from cccma_ppp.generic.runtime import RuntimeContext
from cccma_ppp.train.train_configs import prepare_config
from cccma_ppp.generic.distributed import Distributed
import dacite
from cccma_ppp.train.train_configs import TrainConfig

warnings.filterwarnings("ignore")

INIT_TIME_DIM, LEAD_TIME_DIM = required_sample_dimensions

CSV_COLUMNS = [
    "timestamp",
    "suite",
    "case_name",
    "description",
    "loader_type",
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


def call_with_supported_arguments(function, **kwargs):
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
    *,
    return_metadata: bool = False,
    return_spatial_mask: bool = True,
    shuffle: bool = False,
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
        return_metadata=return_metadata,
        return_spatial_mask=return_spatial_mask,
        shuffle=shuffle,
    )


def load_train_config(config_path: Path) -> Any:
    """Load training config from file"""
    data = prepare_config(config_path)
    if data is None:
        raise ValueError(f"Config is empty: {config_path}")

    return dacite.from_dict(
        data_class=TrainConfig,
        data=data,
        config=dacite.Config(strict=True),
    )


def reset_runtime_context() -> None:
    RuntimeContext.GLOBAL_EXP_DIR = None


def run_test_case(
    config_file: str,
    mode: str,
    output_dir: Path,
) -> bool:
    """
    Run a test case using a pre-existing config file.

    Args:
        config_file: Path to config YAML file (relative to scripts dir)
        mode: 'cvae' or 'deterministic'
        output_dir: Shared output directory for results

    Returns:
        True if passed, False if failed
    """
    case_name = f"model_condition_static_{mode}"
    cases_dir = output_dir / "cases"
    case_dir = cases_dir / StringUtils.clean_name(case_name)
    config_path = case_dir / "config.yaml"
    log_path = case_dir / "case.log"

    # Setup
    case_dir.mkdir(parents=True, exist_ok=True)
    LoggingUtils.cleanup_handlers()

    try:
        # Load config from file
        cfg = prepare_config(config_file)
        if cfg is None:
            raise ValueError(f"Config is empty: {config_file}")

        # Write config to case directory
        ConfigManager.write_yaml(cfg, config_path)

        # Load and setup loader
        config = load_train_config(config_path)
        loader = setup_train_loader(
            config,
            config_path,
            return_metadata=True,
            return_spatial_mask=True,
            shuffle=False,
        )

        # Validate dataset
        DataModulesUtils.test_dataset_getitem_format(
            loader.dataset, init_time_dim=INIT_TIME_DIM, lead_time_dim=LEAD_TIME_DIM
        )

        # Setup plotting
        paths = PathManager(cases_dir, output_dir).case_paths(case_name)
        fetcher = OutputFetcher(paths["case_dir"], max_plot_channels=16)

        all_metadata = []
        batch_sizes = []

        # Check premade mask
        mask_check = DataModulesUtils.check_premade_mask_in_dataset(loader.dataset)

        if mask_check["has_mask_attribute"]:
            fetcher.extract_and_save_premade_mask(loader.dataset)
            fetcher.plot_premade_mask(loader.dataset)

        # Process batches
        for i, batch in enumerate(LimitedLoader(loader, 4)):
            if i == 0:
                fetcher.plot_batch(batch, loader=loader)

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

            results_manager = ResultsManager(
                output_dir / "integration_results.csv",
                output_dir / "summary.txt",
                CSV_COLUMNS,
            )
            results_manager.compute_and_save_metadata_statistics(
                all_metadata,
                paths["metadata_statistics_path"],
                init_time_dim=INIT_TIME_DIM,
                lead_time_dim=LEAD_TIME_DIM,
            )

        print(f"{mode}: {case_name} - {len(all_metadata)} samples")
        return True

    except Exception as e:
        print(f"{mode}: {case_name} - {type(e).__name__}: {str(e)[:80]}")
        LoggingUtils.append_exception(log_path)
        import traceback

        traceback.print_exc()
        return False

    finally:
        reset_runtime_context()
        LoggingUtils.cleanup_handlers()
        try:
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def main() -> None:
    """Run static condition cases from existing config files"""
    root = Path("/fs/site7/eccc/crd/cccma/users/rna002/CCCma-PPP/scripts")

    # Single shared output directory for both modes
    output_dir = root / "../../output" / "data_modules_model_condition_static"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Config files to run
    configs = {
        "cvae": "scripts/module_integration_tests/model_condition_static_cvae_config.yaml",
        "deterministic": "scripts/module_integration_tests/model_condition_static_deterministic_config.yaml",
    }

    results = {}

    print("\n" + "=" * 80)
    print("Running model_condition_static configs")
    print(f"Output: {output_dir}")
    print("=" * 80 + "\n")

    for mode, config_file in configs.items():
        success = run_test_case(config_file, mode, output_dir)
        results[mode] = "PASS" if success else "FAIL"

    print("\n" + "=" * 80)
    print("Summary:")
    for mode, result in results.items():
        print(f"  {mode}: {result}")
    print(f"\nAll results in: {output_dir}")
    print("=" * 80 + "\n")

    if all(v == "PASS" for v in results.values()):
        print("All tests passed!")
    else:
        print(
            f"Some tests failed: {sum(1 for v in results.values() if v == 'FAIL')} failure(s)"
        )


if __name__ == "__main__":
    main()
