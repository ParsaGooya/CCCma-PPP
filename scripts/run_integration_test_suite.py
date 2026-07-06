from pathlib import Path
import yaml
import traceback

import dacite

import cccma_ppp.models.mlp_models

from cccma_ppp.train.train import main as train_main
from cccma_ppp.inference.train import main as inference_main

from cccma_ppp.train.train_configs import (
    prepare_config,
    TrainConfig,
)

from cccma_ppp.generic.distributed import Distributed


print("Imported cVAE models module")
print("Module:", cccma_ppp.models.mlp_models.__file__)

BASE_TRAIN_CONFIG = Path(
    "/fs/site7/eccc/crd/cccma/users/rna002/CCCma-PPP/scripts/integration_suite_config.yaml"
)

BASE_INFERENCE_CONFIG = Path(
    "/fs/site7/eccc/crd/cccma/users/rna002/CCCma-PPP/scripts/inference_integration_config.yaml"
)


def load_train_config(config_path):

    cfg_data = prepare_config(config_path)

    return dacite.from_dict(
        data_class=TrainConfig,
        data=cfg_data,
        config=dacite.Config(strict=True),
    )


def make_train_config(
    output_yaml: Path,
    experiment_dir: Path,
):

    cfg = prepare_config(BASE_TRAIN_CONFIG)

    cfg["experiment_dir"] = str(experiment_dir)

    output_yaml.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_yaml, "w") as f:
        yaml.safe_dump(cfg, f)

    return output_yaml


def make_inference_config(
    output_yaml: Path,
    experiment_dir: Path,
):

    cfg = prepare_config(BASE_INFERENCE_CONFIG)

    cfg["experiment_dir"] = str(experiment_dir)

    with open(output_yaml, "w") as f:
        yaml.safe_dump(cfg, f)

    return output_yaml


def test_training_checkpoint_inference(root_dir):

    print("\n=== Training -> Checkpoint -> Inference ===")

    exp_dir = root_dir / "train_to_inference"

    train_cfg = make_train_config(
        root_dir / "train.yaml",
        exp_dir,
    )

    train_main(str(train_cfg))

    checkpoint_dir = exp_dir / "checkpoints"

    assert checkpoint_dir.exists()

    inference_cfg = make_inference_config(
        root_dir / "inference.yaml",
        exp_dir,
    )

    inference_main(str(inference_cfg))

    print("✓ Training -> Inference succeeded")


def test_preprocessor_roundtrip(root_dir):

    print("\n=== Preprocessor Roundtrip ===")

    exp_dir = root_dir / "preprocessors"

    train_cfg = make_train_config(
        root_dir / "preprocess.yaml",
        exp_dir,
    )

    train_main(str(train_cfg))

    preprocess_dir = exp_dir / "preprocessing_pipeline"

    assert preprocess_dir.exists()

    preprocessing_files = list(preprocess_dir.glob("*"))

    assert preprocessing_files

    inference_cfg = make_inference_config(
        root_dir / "preprocess_inference.yaml",
        exp_dir,
    )

    inference_main(str(inference_cfg))

    print("✓ Preprocessor save/load succeeded")


def test_resume_training(root_dir):

    print("\n=== Resume Training ===")

    exp_dir = root_dir / "resume"

    cfg = make_train_config(
        root_dir / "resume_train.yaml",
        exp_dir,
    )

    train_main(str(cfg))

    resumed_cfg = prepare_config(cfg)

    resumed_cfg["resume_dir"] = str(exp_dir)
    resumed_cfg["max_epochs"] = 2

    resume_yaml = root_dir / "resume_second.yaml"

    with open(resume_yaml, "w") as f:
        yaml.safe_dump(resumed_cfg, f)

    train_main(str(resume_yaml))

    print("✓ Resume training succeeded")


def test_cvae_ensemble_inference(root_dir):

    print("\n=== cVAE Ensemble Inference ===")

    exp_dir = root_dir / "cvae"

    train_cfg = make_train_config(
        root_dir / "cvae_train.yaml",
        exp_dir,
    )

    train_main(str(train_cfg))

    inference_cfg = make_inference_config(
        root_dir / "ensemble_inference.yaml",
        exp_dir,
    )

    cfg = prepare_config(inference_cfg)

    cfg["output_ensemble_size"] = 5

    with open(inference_cfg, "w") as f:
        yaml.safe_dump(cfg, f)

    inference_main(str(inference_cfg))

    print("✓ Ensemble inference succeeded")


def test_dataset_pipeline(root_dir):

    print("\n=== Dataset Pipeline ===")

    exp_dir = root_dir / "dataset_pipeline"

    train_cfg = make_train_config(
        root_dir / "dataset_pipeline.yaml",
        exp_dir,
    )

    config = load_train_config(train_cfg)

    distributed = Distributed.get_instance()

    config.train_loader.setup_distributed(distributed)

    dataset = config.train_loader.dataset_config.build_dataset(
        years=config.train_loader.train_years,
        return_metadata=False,
    )

    assert len(dataset) > 0

    sample = dataset[0]

    assert "input" in sample
    assert "target" in sample

    print("✓ Dataset pipeline succeeded")


def test_metadata_failure(root_dir):

    print("\n=== Metadata Failure Test ===")

    exp_dir = root_dir / "metadata_failure"

    train_cfg = make_train_config(
        root_dir / "metadata_train.yaml",
        exp_dir,
    )

    train_main(str(train_cfg))

    inference_cfg = make_inference_config(
        root_dir / "bad_inference.yaml",
        exp_dir,
    )

    bad_cfg = prepare_config(inference_cfg)

    bad_cfg["model"]["names"] = ["totally_invalid_variable"]

    with open(inference_cfg, "w") as f:
        yaml.safe_dump(
            bad_cfg,
            f,
        )

    try:
        inference_main(str(inference_cfg))

    except Exception:
        print("✓ Metadata validation correctly failed")
        return

    raise AssertionError("Expected metadata validation to fail")


def main():

    output_dir = Path("output/integration_test_results")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("CCCma-PPP Integration Test Suite")
    print("=" * 60)

    tests = [
        (
            "Training -> Checkpoint -> Inference",
            lambda: test_training_checkpoint_inference(output_dir),
        ),
        (
            "Preprocessor Roundtrip",
            lambda: test_preprocessor_roundtrip(output_dir),
        ),
        (
            "Resume Training",
            lambda: test_resume_training(output_dir),
        ),
        (
            "cVAE Ensemble Inference",
            lambda: test_cvae_ensemble_inference(output_dir),
        ),
        (
            "Dataset Pipeline",
            lambda: test_dataset_pipeline(output_dir),
        ),
        (
            "Metadata Failure",
            lambda: test_metadata_failure(output_dir),
        ),
    ]

    failures = []

    for name, test_func in tests:
        print(f"\nRunning: {name}")

        try:
            test_func()

            print(f"✓ PASSED: {name}")

        except Exception as exc:
            print(f"✗ FAILED: {name}")
            print(exc)

            traceback.print_exc()

            failures.append(name)

    print("\n" + "=" * 60)

    if failures:
        print("FAILED TESTS:")

        for failure in failures:
            print(f" - {failure}")

    else:
        print("ALL INTEGRATION TESTS PASSED")

    print("=" * 60)


if __name__ == "__main__":
    main()
