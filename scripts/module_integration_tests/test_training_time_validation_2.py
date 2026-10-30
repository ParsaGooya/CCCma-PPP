import numpy as np
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json
import logging
import pandas as pd

# cccma_ppp imports
from S2D4P.data.data_configs.data_configs import (
    ModelDataConfig,
    ObsDataConfig,
)
from S2D4P.train.dataset import TrainDatasetConfig
from S2D4P.preprocessing.selector import PreprocessingStepSelector
from S2D4P.preprocessing.preprocessing import PreprocessingPipeline

# Configuration
CONFIG_PATH = "scripts/integration_suite_train_config.yaml"
OUTPUT_DIR = "/fs/site7/eccc/crd/cccma/users/rna002/output/training_time_validation_2"


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to file and console."""
    log_file = output_dir / "training_validation.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def create_config_from_yaml(config_dict: dict) -> TrainDatasetConfig:
    """Create ModelDataConfig, ObsDataConfig, and TrainDatasetConfig from YAML."""
    logger = logging.getLogger(__name__)
    logger.info("\n" + "=" * 80)
    logger.info("CREATING CONFIGS FROM YAML")
    logger.info("=" * 80)

    # Model config
    model_dict = config_dict["train_loader"]["dataset_config"]["model"]
    preprocessors_list = model_dict.get("preprocessing_pipeline", {}).get(
        "preprocessors_list", []
    )
    if preprocessors_list:
        preprocessor_steps = [
            PreprocessingStepSelector(name=proc["name"], args=proc.get("args"))
            for proc in preprocessors_list
        ]
        pipeline = PreprocessingPipeline(preprocessor_steps)
    else:
        pipeline = PreprocessingPipeline([])

    # The model data uses 'year' as init time, not 'time'
    # Rename it so the library recognizes it
    rename_dict = model_dict.get("rename_dict", {})
    if "year" not in rename_dict:
        rename_dict["year"] = "time"

    model_config = ModelDataConfig(
        paths=model_dict["paths"],
        file_type="*.nc",
        names=model_dict["names"],
        preprocessing_pipeline=pipeline,
        ensemble_mean=model_dict.get("ensemble_mean", False),
        rename_dict=rename_dict,
    )
    logger.info(" ModelDataConfig created")

    # Obs config
    obs_dict = config_dict["train_loader"]["dataset_config"]["observation"]
    preprocessors_list = obs_dict.get("preprocessing_pipeline", {}).get(
        "preprocessors_list", []
    )
    if preprocessors_list:
        preprocessor_steps = [
            PreprocessingStepSelector(name=proc["name"], args=proc.get("args"))
            for proc in preprocessors_list
        ]
        pipeline = PreprocessingPipeline(preprocessor_steps)
    else:
        pipeline = PreprocessingPipeline([])

    obs_config = ObsDataConfig(
        paths=obs_dict["paths"],
        file_type="*.nc",
        names=obs_dict["names"],
        preprocessing_pipeline=pipeline,
    )
    logger.info(" ObsDataConfig created")

    # Train config
    dataset_dict = config_dict["train_loader"]["dataset_config"]
    lead_times_config = dataset_dict["lead_times"]
    lead_times = list(range(lead_times_config["start"], lead_times_config["end"] + 1))

    train_config = TrainDatasetConfig(
        model=model_config,
        observation=obs_config,
        condition_method=dataset_dict.get("condition_method", "ensemble_mean"),
        lead_times=lead_times,
    )
    logger.info(" TrainDatasetConfig created")

    return train_config


def verify_and_print_times(config: TrainDatasetConfig) -> tuple:
    """Verify times and print them explicitly."""
    logger = logging.getLogger(__name__)
    logger.info("\n" + "=" * 80)
    logger.info("VERIFYING TIMES")
    logger.info("=" * 80)

    # Get common times
    common_times = config.get_common_time
    logger.info("\nCOMMON TIMES (intersection of model and obs):")
    logger.info(f"   Count: {len(common_times)}")
    logger.info(f"   Range: {common_times[0]} to {common_times[-1]}")

    # Get available times
    available_times = config.available_times
    logger.info("\nAVAILABLE TIMES (for training):")
    logger.info(f"   Count: {len(available_times)}")
    logger.info(f"   Range: {available_times[0]} to {available_times[-1]}")

    # Print first and last 5
    logger.info("\nCOMMON TIMES (first 5 and last 5):")
    for i, t in enumerate(common_times):
        if i < 5 or i >= len(common_times) - 5:
            logger.info(f"   [{i}] {t}")
        elif i == 5:
            logger.info(f"   ... ({len(common_times) - 10} more times) ...")

    logger.info("\nAVAILABLE TIMES (first 5 and last 5):")
    for i, t in enumerate(available_times):
        if i < 5 or i >= len(available_times) - 5:
            logger.info(f"   [{i}] {t}")
        elif i == 5:
            logger.info(f"   ... ({len(available_times) - 10} more times) ...")

    return common_times, available_times


def create_training_mask(
    config: TrainDatasetConfig, available_times, output_dir: Path
) -> np.ndarray:
    """Create training mask."""
    logger = logging.getLogger(__name__)
    logger.info("\n" + "=" * 80)
    logger.info("CREATING TRAINING MASK")
    logger.info("=" * 80)

    # Get model initialization times
    model_init_times = config.get_common_time.to_series().index.unique()
    lead_times = config.lead_times

    logger.info("\nMask dimensions:")
    logger.info(f"  Model init times: {len(model_init_times)}")
    logger.info(f"  Lead times: {len(lead_times)}")
    logger.info(f"  Total pairs: {len(model_init_times) * len(lead_times)}")

    # Create mask
    mask = np.zeros((len(model_init_times), len(lead_times)), dtype=bool)
    valid_count = 0

    for init_idx, init_time in enumerate(model_init_times):
        for lead_idx, lead_time in enumerate(lead_times):
            forecast_time = init_time + pd.DateOffset(months=int(lead_time))
            if forecast_time in available_times:
                mask[init_idx, lead_idx] = True
                valid_count += 1

    coverage = 100.0 * valid_count / mask.size
    logger.info("\nMask statistics:")
    logger.info(f"   Valid pairs: {valid_count}")
    logger.info(f"   Invalid pairs: {mask.size - valid_count}")
    logger.info(f"   Coverage: {coverage:.1f}%")

    # Save mask
    mask_file = output_dir / "training_mask.npy"
    np.save(mask_file, mask)
    logger.info(f"\n Saved mask: {mask_file}")

    return mask


def plot_training_mask(
    mask: np.ndarray, output_dir: Path, lead_times: list = None
) -> None:
    """Plot training mask using matplotlib."""
    logger = logging.getLogger(__name__)
    logger.info("\n" + "=" * 80)
    logger.info("PLOTTING TRAINING MASK")
    logger.info("=" * 80)

    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 10))

        # Plot mask (white=valid, black=invalid)
        im = ax.imshow(
            mask,
            cmap="binary",
            origin="lower",
            aspect="auto",
            interpolation="nearest",
        )

        ax.set_title(
            "Training Time Mask\n(white=valid sample, black=masked)",
            fontsize=16,
            fontweight="bold",
        )
        ax.set_ylabel("Initialization Time Index", fontsize=12)
        ax.set_xlabel("Lead Time Index", fontsize=12)

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Valid (1) / Masked (0)", fontsize=11)

        # Set x-axis labels to lead times if provided
        if lead_times is not None and len(lead_times) == mask.shape[1]:
            # Show every Nth lead time to avoid crowding
            step = max(1, len(lead_times) // 20)
            tick_positions = np.arange(0, len(lead_times), step)
            ax.set_xticks(tick_positions)
            tick_labels = [str(lead_times[i]) for i in tick_positions]
            ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        else:
            # Just show index
            step = max(1, mask.shape[1] // 20)
            tick_positions = np.arange(0, mask.shape[1], step)
            ax.set_xticks(tick_positions)

        # Add statistics text
        valid_count = np.sum(mask)
        total_count = mask.size
        coverage = 100.0 * valid_count / total_count

        stats_text = (
            f"Valid pairs: {valid_count:,}\n"
            f"Total pairs: {total_count:,}\n"
            f"Coverage: {coverage:.1f}%"
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round",
                facecolor="wheat",
                alpha=0.8,
                edgecolor="black",
                linewidth=1.5,
            ),
        )

        fig.tight_layout()

        # Save figure
        output_file = output_dir / "training_mask.png"
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f" Saved plot: {output_file}")

    except Exception as e:
        logger.error(f"Error plotting mask: {e}")


def main(config_path: str = None, output_dir: str = None) -> None:
    """Main workflow."""
    if config_path is None:
        config_path = CONFIG_PATH
    if output_dir is None:
        output_dir = OUTPUT_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING TIME VALIDATION")
    logger.info("=" * 80)
    logger.info("Validates which (init_time, lead_time) pairs are usable for training")

    try:
        # Load config
        logger.info("\nLoading configuration...")
        config_dict = load_config(config_path)
        logger.info(f" Config loaded from: {config_path}")

        # Create configs from YAML
        train_config = create_config_from_yaml(config_dict)

        # Verify and print times
        common_times, available_times = verify_and_print_times(train_config)

        # Create training mask
        lead_times = train_config.lead_times
        mask = create_training_mask(train_config, available_times, output_dir)

        # Plot mask
        plot_training_mask(mask, output_dir, lead_times=list(lead_times))

        # Save summary report
        report = {
            "timestamp": datetime.now().isoformat(),
            "config_path": str(config_path),
            "output_dir": str(output_dir),
            "times": {
                "common_times_count": len(common_times),
                "common_times_range": [str(common_times[0]), str(common_times[-1])],
                "available_times_count": len(available_times),
                "available_times_range": [
                    str(available_times[0]),
                    str(available_times[-1]),
                ],
            },
            "mask": {
                "shape": tuple(mask.shape),
                "valid_pairs": int(np.sum(mask)),
                "invalid_pairs": int(mask.size - np.sum(mask)),
                "coverage_percent": float(100.0 * np.sum(mask) / mask.size),
                "lead_times_count": len(lead_times),
                "lead_times_range": [int(lead_times[0]), int(lead_times[-1])],
            },
            "files": {
                "mask_npy": "training_mask.npy",
                "mask_plot": "training_mask.png",
                "report_json": "training_validation_report.json",
                "log_file": "training_validation.log",
            },
        }

        report_file = output_dir / "training_validation_report.json"
        with report_file.open("w") as f:
            json.dump(report, f, indent=2)

        logger.info("\n" + "=" * 80)
        logger.info(" SUCCESS")
        logger.info("=" * 80)
        logger.info(f"\nResults saved to: {output_dir}")
        logger.info("  - training_mask.npy (binary mask)")
        logger.info("  - training_mask.png (visualization)")
        logger.info("  - training_validation_report.json (summary)")
        logger.info("  - training_validation.log (detailed log)")

    except Exception as e:
        logger.error(f"\n[!] Error: {type(e).__name__}: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else None
        main(config_path, output_dir)
    else:
        main()
