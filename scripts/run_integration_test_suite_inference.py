from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import datetime
from importlib import import_module
from itertools import product
from pathlib import Path
import csv
import json
import logging
import re
import shutil
import traceback
import warnings

import yaml
from tqdm import tqdm


import cccma_ppp.core.cVAE_module  # noqa: F401
import cccma_ppp.core.deterministic_module  # noqa: F401
import cccma_ppp.inference.predictors  # noqa: F401
import cccma_ppp.loss.utils_loss  # noqa: F401
import cccma_ppp.models.mlp_models.cvae  # noqa: F401
import cccma_ppp.models.mlp_models.deterministic  # noqa: F401
import cccma_ppp.models.unet_models.cvae  # noqa: F401
import cccma_ppp.models.unet_models.deterministic  # noqa: F401
import cccma_ppp.preprocessing.utils_preprocessing  # noqa: F401

from cccma_ppp.train.train import main as train_main


warnings.filterwarnings("ignore")
logging.raiseExceptions = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_TRAIN_CONFIG = PROJECT_ROOT / "scripts" / "integration_suite_train_config.yaml"

BASE_INFERENCE_CONFIG = (
    PROJECT_ROOT / "scripts" / "integration_suite_inference_config.yaml"
)

OUTPUT_DIR = PROJECT_ROOT / "output" / "inference_integration_test_results"

TRAINING_EXPERIMENT_DIR = OUTPUT_DIR / "_trained_model"
GENERATED_CONFIG_DIR = OUTPUT_DIR / "_generated_configs"
LOG_DIR = OUTPUT_DIR / "_logs"

TRAINING_CONFIG_PATH = GENERATED_CONFIG_DIR / "inference_model_train.yaml"

TRAINING_LOG_PATH = LOG_DIR / "inference_model_train.log"

RESULTS_CSV = OUTPUT_DIR / "inference_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary.txt"


TRAIN_MODEL = True
REUSE_EXISTING_MODEL = False

RUN_BASELINE_CASE = True
RUN_INFERENCE_GRID = True

MAX_INFERENCE_CASES = None

TRAINING_MAX_EPOCHS = 3

TEST_ENCODER_HIDDEN_DIMS = [16]
TEST_DECODER_HIDDEN_DIMS = [16]
TEST_CONDITION_EMBEDDING_DIMS = [16]

VALID_TIME_FEATURES = {
    "year",
    "lead_time",
    "month_sin",
    "month_cos",
}

REQUIRE_OUTPUT_ARTIFACT = False


INFERENCE_GRID = {
    "num_latent_samples": [
        1,
        10,
    ],
    "num_output_covariance_sampling": [
        1,
        2,
    ],
    "batch_size": [
        1,
        12,
    ],
    "inference_years": [
        [2018, 2018],
        [2018, 2020],
    ],
    "num_data_workers": [
        0,
    ],
    "load": [
        False,
        True,
    ],
}


CSV_COLUMNS = [
    "timestamp",
    "suite",
    "case_name",
    "shared_training_experiment_dir",
    "experiment_dir",
    "num_latent_samples",
    "num_output_covariance_sampling",
    "batch_size",
    "inference_years",
    "num_data_workers",
    "load",
    "expected_result",
    "expected_failure_reason",
    "config_path",
    "log_path",
    "result",
    "error",
]


def get_inference_main():
    candidates = [
        "cccma_ppp.inference.inference",
        "cccma_ppp.inference.main",
        "cccma_ppp.inference.predict",
    ]

    errors = []

    for module_name in candidates:
        try:
            module = import_module(module_name)
        except ImportError as exc:
            errors.append(f"{module_name}: {exc}")
            continue

        function = getattr(module, "main", None)

        if callable(function):
            return function

        errors.append(f"{module_name}: no callable main")

    details = "\n- ".join(errors)

    raise ImportError(
        f"Could not locate the inference main function. Tried:\n- {details}"
    )


def prepare_output_directories():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    GENERATED_CONFIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def cleanup_logging_handlers():
    logger_names = [
        "",
        "training",
        "inference",
        "prediction",
        "writer",
    ]

    for logger_name in list(logging.Logger.manager.loggerDict):
        if logger_name.startswith("cccma_ppp"):
            logger_names.append(logger_name)

    for logger_name in dict.fromkeys(logger_names):
        logger = logging.getLogger(logger_name)

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


def init_csv():
    prepare_output_directories()

    with RESULTS_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()


def clean_name(value):
    value = str(value).replace(
        "None",
        "none",
    )
    value = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        value,
    )
    value = re.sub(
        r"_+",
        "_",
        value,
    )

    return value.strip("_")


def read_yaml(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"Configuration file is empty: {path}")

    if not isinstance(config, dict):
        raise TypeError(f"Configuration must be a mapping: {path}")

    return config


def write_raw_yaml(
    config,
    output_path,
):
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            deepcopy(config),
            file,
            sort_keys=False,
        )

    return output_path


def run_silenced(
    function,
    log_path,
):
    log_path = Path(log_path)

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cleanup_logging_handlers()

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as log_file:
        try:
            with (
                redirect_stdout(log_file),
                redirect_stderr(log_file),
            ):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return function()
        finally:
            cleanup_logging_handlers()


def append_exception_to_log(log_path):
    log_path = Path(log_path)

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with log_path.open(
        "a",
        encoding="utf-8",
    ) as log_file:
        traceback.print_exc(file=log_file)


def append_message_to_log(
    log_path,
    message,
):
    log_path = Path(log_path)

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with log_path.open(
        "a",
        encoding="utf-8",
    ) as log_file:
        log_file.write(f"\n{message}\n")


def remove_existing_case_outputs(
    experiment_dir,
    config_path,
    log_path,
):
    experiment_dir = Path(experiment_dir)
    config_path = Path(config_path)
    log_path = Path(log_path)

    if experiment_dir.exists():
        shutil.rmtree(experiment_dir)

    if config_path.exists():
        config_path.unlink()

    if log_path.exists():
        log_path.unlink()


def validate_time_features(
    loader_config,
    *,
    loader_name,
):
    dataset_config = loader_config.get("dataset_config")

    if not isinstance(dataset_config, dict):
        raise ValueError(
            f"{loader_name}.dataset_config must be configured as a mapping."
        )

    if "time_features" in dataset_config:
        raise ValueError(
            "time_features is no longer a DatasetConfig field. "
            f"Move {loader_name}.dataset_config.time_features to "
            f"{loader_name}.time_features."
        )

    time_features = loader_config.get("time_features")

    if time_features is None:
        return

    if not isinstance(time_features, list):
        raise TypeError(f"{loader_name}.time_features must be a list or null.")

    if not all(isinstance(feature, str) for feature in time_features):
        raise TypeError(f"{loader_name}.time_features must contain strings.")

    unsupported = set(time_features) - VALID_TIME_FEATURES

    if unsupported:
        raise ValueError(
            f"Unsupported {loader_name}.time_features: "
            f"{sorted(unsupported)}. Supported values are "
            f"{sorted(VALID_TIME_FEATURES)}."
        )


def normalize_training_config(config):
    train_loader = config.get("train_loader")

    if not isinstance(train_loader, dict):
        raise ValueError("train_loader must be configured as a mapping.")

    validate_time_features(
        train_loader,
        loader_name="train_loader",
    )

    dataset_config = train_loader["dataset_config"]

    if dataset_config.get("condition_method") is None:
        dataset_config.pop(
            "condition_method",
            None,
        )

    if dataset_config.get("lead_months") is None:
        dataset_config.pop(
            "lead_months",
            None,
        )

    if train_loader.get("time_features") is None:
        train_loader.pop(
            "time_features",
            None,
        )

    module_config = config.get("module", {}).get("config", {})

    if module_config.get("min_posterior_variance") is None:
        module_config.pop(
            "min_posterior_variance",
            None,
        )

    if module_config.get("prior_flow_config") is None:
        module_config.pop(
            "prior_flow_config",
            None,
        )

    model_config = module_config.get("ModelConfig", {}).get("config", {})

    if model_config.get("dropout_rate") is None:
        model_config.pop(
            "dropout_rate",
            None,
        )

    return config


def validate_training_config(config):
    experiment_dir = config.get("experiment_dir")

    if not experiment_dir:
        raise ValueError("experiment_dir must be configured.")

    max_epochs = config.get("max_epochs")

    if not isinstance(max_epochs, int) or max_epochs <= 0:
        raise ValueError("max_epochs must be a positive integer.")

    train_loader = config.get("train_loader")

    if not isinstance(train_loader, dict):
        raise ValueError("train_loader must be configured as a mapping.")

    validate_time_features(
        train_loader,
        loader_name="train_loader",
    )

    num_data_workers = train_loader.get("num_data_workers")

    if not isinstance(num_data_workers, int) or num_data_workers < 0:
        raise ValueError(
            "train_loader.num_data_workers must be a non-negative integer."
        )

    module = config.get("module")

    if not isinstance(module, dict):
        raise ValueError("module must be configured as a mapping.")

    module_type = module.get("type")

    if not isinstance(module_type, str) or not module_type:
        raise ValueError("module.type must be a non-empty string.")

    module_config = module.get("config")

    if not isinstance(module_config, dict):
        raise ValueError("module.config must be configured as a mapping.")

    model_selector = module_config.get("ModelConfig")

    if not isinstance(model_selector, dict):
        raise ValueError("module.config.ModelConfig must be configured as a mapping.")

    model_type = model_selector.get("type")

    if not isinstance(model_type, str) or not model_type:
        raise ValueError("module.config.ModelConfig.type must be a non-empty string.")

    model_config = model_selector.get("config")

    if not isinstance(model_config, dict):
        raise ValueError(
            "module.config.ModelConfig.config must be configured as a mapping."
        )

    return config


def build_training_fixture_config():
    config = read_yaml(BASE_TRAIN_CONFIG)

    config.pop(
        "resume_dir",
        None,
    )

    config["experiment_dir"] = str(TRAINING_EXPERIMENT_DIR.resolve())
    config["max_epochs"] = TRAINING_MAX_EPOCHS
    config["save_checkpoint"] = True

    train_loader = config.get("train_loader")

    if not isinstance(train_loader, dict):
        raise ValueError(
            "The base training configuration must contain a train_loader mapping."
        )

    dataset_config = train_loader.get("dataset_config")

    if not isinstance(dataset_config, dict):
        raise ValueError(
            "The base training configuration must contain train_loader.dataset_config."
        )

    if "time_features" in dataset_config:
        raise ValueError(
            "The base training configuration still defines "
            "train_loader.dataset_config.time_features. Move it "
            "to train_loader.time_features."
        )

    train_loader["time_features"] = [
        "year",
    ]
    train_loader["num_data_workers"] = 0
    train_loader["num_validation_years"] = 0

    dataset_config["condition_method"] = "ensemble_mean"
    dataset_config["lead_months"] = {
        "start": 1,
        "end": 12,
    }

    model_data_config = dataset_config.get("model")

    if not isinstance(model_data_config, dict):
        raise ValueError(
            "train_loader.dataset_config.model must be configured as a mapping."
        )

    model_data_config["ensemble_mean"] = True

    dataset_config.pop(
        "condition",
        None,
    )

    module = config.setdefault(
        "module",
        {},
    )
    module["type"] = "cVAE"

    module_config = module.setdefault(
        "config",
        {},
    )

    model_selector = module_config.setdefault(
        "ModelConfig",
        {},
    )
    model_selector["type"] = "mlp"

    model_config = model_selector.setdefault(
        "config",
        {},
    )

    model_config["encoder_hidden_dims"] = deepcopy(TEST_ENCODER_HIDDEN_DIMS)
    model_config["decoder_hidden_dims"] = deepcopy(TEST_DECODER_HIDDEN_DIMS)
    model_config["condition_embedding_dims"] = deepcopy(TEST_CONDITION_EMBEDDING_DIMS)

    latent_size = model_config.get(
        "latent_size",
        10,
    )

    model_config["latent_size"] = latent_size
    model_config["condition_embedding_size"] = latent_size
    model_config["condition_dependant_latent"] = True
    model_config["condemb_to_decoder"] = True
    model_config["batch_normalization"] = False
    model_config["init_method"] = "xavier"

    model_config.pop(
        "dropout_rate",
        None,
    )

    module_config.pop(
        "prior_flow_config",
        None,
    )
    module_config.pop(
        "min_posterior_variance",
        None,
    )

    module_config["combined_CGCN_weight"] = 0.0

    config.setdefault(
        "trainer",
        {},
    ).setdefault(
        "beta_finder",
        {
            "beta": 10,
            "num_epoch_to_warmup": 10,
        },
    )

    normalize_training_config(config)
    validate_training_config(config)

    return config


def validate_training_experiment(experiment_dir):
    experiment_dir = Path(experiment_dir)

    if not experiment_dir.is_dir():
        raise FileNotFoundError(
            f"Training experiment directory does not exist: {experiment_dir}"
        )

    checkpoint_dir = experiment_dir / "checkpoints"

    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"Training checkpoint directory does not exist: {checkpoint_dir}"
        )

    checkpoint_files = sorted(checkpoint_dir.glob("*.pt"))

    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files were found in: {checkpoint_dir}")

    return checkpoint_files


def assert_training_completed(
    *,
    require_log=True,
):
    validate_training_experiment(TRAINING_EXPERIMENT_DIR)

    if not require_log:
        return

    if not TRAINING_LOG_PATH.is_file():
        raise AssertionError(f"Training log does not exist: {TRAINING_LOG_PATH}")

    log_text = TRAINING_LOG_PATH.read_text(
        encoding="utf-8",
        errors="replace",
    )

    if "Training finished" not in log_text:
        raise AssertionError(
            "Training did not reach completion. "
            "The log does not contain 'Training finished': "
            f"{TRAINING_LOG_PATH}"
        )


def train_model_once():
    if REUSE_EXISTING_MODEL and TRAINING_EXPERIMENT_DIR.exists():
        assert_training_completed(
            require_log=False,
        )
        return TRAINING_EXPERIMENT_DIR

    if not TRAIN_MODEL:
        validate_training_experiment(TRAINING_EXPERIMENT_DIR)
        return TRAINING_EXPERIMENT_DIR

    if TRAINING_EXPERIMENT_DIR.exists():
        shutil.rmtree(TRAINING_EXPERIMENT_DIR)

    if TRAINING_CONFIG_PATH.exists():
        TRAINING_CONFIG_PATH.unlink()

    if TRAINING_LOG_PATH.exists():
        TRAINING_LOG_PATH.unlink()

    training_config = build_training_fixture_config()

    write_raw_yaml(
        training_config,
        TRAINING_CONFIG_PATH,
    )

    try:
        run_silenced(
            lambda: train_main(str(TRAINING_CONFIG_PATH)),
            TRAINING_LOG_PATH,
        )
    except KeyboardInterrupt:
        append_message_to_log(
            TRAINING_LOG_PATH,
            "Training fixture interrupted by the user.",
        )
        cleanup_logging_handlers()
        raise

    assert_training_completed(
        require_log=True,
    )

    return TRAINING_EXPERIMENT_DIR


def link_or_copy_file(
    source,
    destination,
):
    source = Path(source)
    destination = Path(destination)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(
            source,
            destination,
        )

    return str(destination)


def clone_training_experiment(destination):
    destination = Path(destination)

    if destination.exists():
        shutil.rmtree(destination)

    shutil.copytree(
        TRAINING_EXPERIMENT_DIR,
        destination,
        copy_function=link_or_copy_file,
    )

    validate_training_experiment(destination)

    return destination


def validate_inference_config(config):
    experiment_dir = config.get("experiment_dir")

    if not experiment_dir:
        raise ValueError("experiment_dir must be configured.")

    inference_loader = config.get("inference_loader")

    if not isinstance(inference_loader, dict):
        raise ValueError("inference_loader must be configured as a mapping.")

    if "dataset_config" in inference_loader:
        validate_time_features(
            inference_loader,
            loader_name="inference_loader",
        )

    batch_size = inference_loader.get("batch_size")

    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("inference_loader.batch_size must be a positive integer.")

    inference_years = inference_loader.get("inference_years")

    if not isinstance(inference_years, list) or len(inference_years) != 2:
        raise ValueError("inference_loader.inference_years must be a two-element list.")

    if not all(isinstance(year, int) for year in inference_years):
        raise TypeError("inference_loader.inference_years must contain integers.")

    if inference_years[0] > inference_years:
        raise ValueError(
            "The first inference year cannot be greater than the final inference year."
        )

    num_data_workers = inference_loader.get("num_data_workers")

    if not isinstance(num_data_workers, int) or num_data_workers < 0:
        raise ValueError(
            "inference_loader.num_data_workers must be a non-negative integer."
        )

    load = inference_loader.get("load")

    if not isinstance(load, bool):
        raise TypeError("inference_loader.load must be a boolean.")

    writer = config.get("writer")

    if not isinstance(writer, dict):
        raise ValueError("writer must be configured as a mapping.")

    predictor = writer.get("predictor")

    if not isinstance(predictor, dict):
        raise ValueError("writer.predictor must be configured as a mapping.")

    predictor_type = predictor.get("type")

    if predictor_type is not None and (
        not isinstance(predictor_type, str) or not predictor_type
    ):
        raise ValueError("writer.predictor.type must be a non-empty string.")

    predictor_config = predictor.get("config")

    if not isinstance(predictor_config, dict):
        raise ValueError("writer.predictor.config must be configured as a mapping.")

    num_latent_samples = predictor_config.get("num_latent_samples")

    if not isinstance(num_latent_samples, int) or num_latent_samples <= 0:
        raise ValueError(
            "writer.predictor.config.num_latent_samples must be a positive integer."
        )

    covariance_samples = writer.get("num_output_covariance_sampling")

    if not isinstance(covariance_samples, int) or covariance_samples <= 0:
        raise ValueError(
            "writer.num_output_covariance_sampling must be a positive integer."
        )

    return config


def load_base_inference_config():
    config = read_yaml(BASE_INFERENCE_CONFIG)

    config["experiment_dir"] = str(TRAINING_EXPERIMENT_DIR.resolve())

    inference_loader = config.get("inference_loader")

    if isinstance(inference_loader, dict):
        dataset_config = inference_loader.get("dataset_config")

        if isinstance(dataset_config, dict) and "time_features" in dataset_config:
            raise ValueError(
                "The base inference configuration still defines "
                "inference_loader.dataset_config.time_features. "
                "Move it to inference_loader.time_features."
            )

    validate_inference_config(config)

    return config


def apply_inference_case(
    base_config,
    case,
):
    config = deepcopy(base_config)

    writer = config.setdefault(
        "writer",
        {},
    )
    predictor = writer.setdefault(
        "predictor",
        {},
    )
    predictor_config = predictor.setdefault(
        "config",
        {},
    )
    inference_loader = config.setdefault(
        "inference_loader",
        {},
    )

    predictor_config["num_latent_samples"] = case["num_latent_samples"]

    writer["num_output_covariance_sampling"] = case["num_output_covariance_sampling"]

    inference_loader["batch_size"] = case["batch_size"]
    inference_loader["inference_years"] = deepcopy(case["inference_years"])
    inference_loader["num_data_workers"] = case["num_data_workers"]
    inference_loader["load"] = case["load"]

    validate_inference_config(config)

    return config


def make_case_name(case):
    years = case["inference_years"]

    return clean_name(
        "inference"
        f"_latent_{case['num_latent_samples']}"
        "_covariance_"
        f"{case['num_output_covariance_sampling']}"
        f"_batch_{case['batch_size']}"
        f"_years_{years[0]}_{years[1]}"
        f"_workers_{case['num_data_workers']}"
        f"_load_{case['load']}"
    )


def generate_inference_cases():
    keys = list(INFERENCE_GRID)
    values = [INFERENCE_GRID[key] for key in keys]

    cases = []

    for combination in product(*values):
        case = dict(
            zip(
                keys,
                combination,
            )
        )

        case["name"] = make_case_name(case)
        case["expected_result"] = "PASS"
        case["expected_failure_reason"] = ""

        cases.append(case)

    if MAX_INFERENCE_CASES is not None:
        cases = cases[:MAX_INFERENCE_CASES]

    return cases


def baseline_case_from_config(config):
    inference_loader = config["inference_loader"]
    writer = config["writer"]
    predictor_config = writer["predictor"]["config"]

    return {
        "name": "inference_baseline",
        "num_latent_samples": (predictor_config["num_latent_samples"]),
        "num_output_covariance_sampling": (writer["num_output_covariance_sampling"]),
        "batch_size": inference_loader["batch_size"],
        "inference_years": deepcopy(inference_loader["inference_years"]),
        "num_data_workers": (inference_loader["num_data_workers"]),
        "load": inference_loader["load"],
        "expected_result": "PASS",
        "expected_failure_reason": "",
    }


def build_inference_tasks(base_config):
    tasks = []

    if RUN_BASELINE_CASE:
        baseline = baseline_case_from_config(base_config)

        tasks.append(
            {
                "case": baseline,
                "config": deepcopy(base_config),
            }
        )

    if RUN_INFERENCE_GRID:
        for case in generate_inference_cases():
            tasks.append(
                {
                    "case": case,
                    "config": apply_inference_case(
                        base_config,
                        case,
                    ),
                }
            )

    return tasks


def snapshot_files(root_dir):
    root_dir = Path(root_dir)

    snapshot = {}

    if not root_dir.exists():
        return snapshot

    ignored_directories = {
        "checkpoints",
        "_generated_configs",
        "_logs",
    }

    for path in root_dir.rglob("*"):
        if not path.is_file():
            continue

        relative_parts = path.relative_to(root_dir).parts

        if any(part in ignored_directories for part in relative_parts):
            continue

        try:
            stat = path.stat()
        except OSError:
            continue

        snapshot[str(path)] = (
            stat.st_mtime_ns,
            stat.st_size,
        )

    return snapshot


def find_changed_files(
    before,
    after,
):
    return [
        Path(path)
        for path, state in after.items()
        if path not in before or before[path] != state
    ]


def assert_inference_completed(
    *,
    log_path,
    changed_files,
):
    log_path = Path(log_path)

    if not log_path.is_file():
        raise AssertionError(f"Inference log does not exist: {log_path}")

    log_text = log_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    if "Traceback (most recent call last)" in log_text:
        raise AssertionError(f"The inference log contains a traceback: {log_path}")

    if REQUIRE_OUTPUT_ARTIFACT and not changed_files:
        raise AssertionError(
            "Inference completed without creating or modifying an output artifact."
        )


def log_result(
    *,
    case,
    experiment_dir,
    config_path,
    log_path,
    passed,
    error="",
):
    row = {
        "timestamp": datetime.now().isoformat(),
        "suite": "inference",
        "case_name": case["name"],
        "shared_training_experiment_dir": str(TRAINING_EXPERIMENT_DIR),
        "experiment_dir": str(experiment_dir),
        "num_latent_samples": (case["num_latent_samples"]),
        "num_output_covariance_sampling": (case["num_output_covariance_sampling"]),
        "batch_size": case["batch_size"],
        "inference_years": json.dumps(case["inference_years"]),
        "num_data_workers": (case["num_data_workers"]),
        "load": case["load"],
        "expected_result": case.get(
            "expected_result",
            "PASS",
        ),
        "expected_failure_reason": case.get(
            "expected_failure_reason",
            "",
        ),
        "config_path": str(config_path),
        "log_path": str(log_path),
        "result": "PASS" if passed else "FAIL",
        "error": error,
    }

    with RESULTS_CSV.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_COLUMNS,
            lineterminator="\n",
        )
        writer.writerow(row)
        file.flush()


def run_inference_case(
    inference_main,
    task,
):
    case = task["case"]
    config = deepcopy(task["config"])

    safe_name = clean_name(case["name"])

    experiment_dir = OUTPUT_DIR / safe_name
    config_path = GENERATED_CONFIG_DIR / f"{safe_name}.yaml"
    log_path = LOG_DIR / f"{safe_name}.log"

    remove_existing_case_outputs(
        experiment_dir,
        config_path,
        log_path,
    )

    clone_training_experiment(experiment_dir)

    config["experiment_dir"] = str(experiment_dir.resolve())

    validate_inference_config(config)
    validate_training_experiment(experiment_dir)

    write_raw_yaml(
        config,
        config_path,
    )

    before = snapshot_files(experiment_dir)

    try:
        run_silenced(
            lambda: inference_main(str(config_path)),
            log_path,
        )

        after = snapshot_files(experiment_dir)

        changed_files = find_changed_files(
            before,
            after,
        )

        assert_inference_completed(
            log_path=log_path,
            changed_files=changed_files,
        )

        passed = True
        error = ""

    except KeyboardInterrupt:
        passed = False
        error = "KeyboardInterrupt: inference case was interrupted by the user."

        append_message_to_log(
            log_path,
            error,
        )
        cleanup_logging_handlers()

        log_result(
            case=case,
            experiment_dir=experiment_dir,
            config_path=config_path,
            log_path=log_path,
            passed=passed,
            error=error,
        )

        raise

    except Exception as exc:
        passed = False
        error = f"{type(exc).__name__}: {exc}"

        append_exception_to_log(log_path)
        cleanup_logging_handlers()

    log_result(
        case=case,
        experiment_dir=experiment_dir,
        config_path=config_path,
        log_path=log_path,
        passed=passed,
        error=error,
    )

    return passed


def classify_result(
    *,
    case,
    passed,
    unexpected_failures,
    expected_failures,
    unexpected_passes,
):
    expected_result = case.get(
        "expected_result",
        "PASS",
    )

    if expected_result == "FAIL":
        if passed:
            unexpected_passes.append(case["name"])
        else:
            expected_failures.append(case["name"])
    elif not passed:
        unexpected_failures.append(case["name"])


def write_summary(
    tasks,
    completed_cases,
    actual_passes,
    unexpected_failures,
    expected_failures,
    unexpected_passes,
    interrupted_case=None,
):
    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(f"base_train_config: {BASE_TRAIN_CONFIG}\n")
        file.write(f"base_inference_config: {BASE_INFERENCE_CONFIG}\n")
        file.write(f"shared_training_experiment_dir: {TRAINING_EXPERIMENT_DIR}\n")
        file.write(f"training_config: {TRAINING_CONFIG_PATH}\n")
        file.write(f"training_log: {TRAINING_LOG_PATH}\n")
        file.write(f"total_inference_cases: {len(tasks)}\n")
        file.write(f"completed_cases: {completed_cases}\n")
        file.write(f"actual_passes: {actual_passes}\n")
        file.write(f"unexpected_failures: {len(unexpected_failures)}\n")
        file.write(f"expected_failures: {len(expected_failures)}\n")
        file.write(f"unexpected_passes: {len(unexpected_passes)}\n")
        file.write(f"interrupted_case: {interrupted_case or ''}\n")
        file.write(f"results_csv: {RESULTS_CSV}\n")
        file.write(f"generated_config_dir: {GENERATED_CONFIG_DIR}\n")
        file.write(f"log_dir: {LOG_DIR}\n")

        groups = [
            (
                "unexpected_failures",
                unexpected_failures,
            ),
            (
                "expected_failures",
                expected_failures,
            ),
            (
                "unexpected_passes",
                unexpected_passes,
            ),
        ]

        for title, values in groups:
            if not values:
                continue

            file.write(f"\n{title}:\n")

            for value in values:
                file.write(f"- {value}\n")


def main():
    prepare_output_directories()
    cleanup_logging_handlers()

    inference_main = get_inference_main()

    print("Training one shared cVAE model for inference...")

    train_model_once()

    print(f"Training fixture completed: {TRAINING_EXPERIMENT_DIR}")

    init_csv()

    base_inference_config = load_base_inference_config()

    tasks = build_inference_tasks(base_inference_config)

    unexpected_failures = []
    expected_failures = []
    unexpected_passes = []

    completed_cases = 0
    actual_passes = 0
    interrupted_case = None

    try:
        with tqdm(
            total=len(tasks),
            desc="inference integration",
            unit="case",
            dynamic_ncols=True,
            leave=True,
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}] "
                "{postfix}"
            ),
        ) as progress:
            for task in tasks:
                case = task["case"]

                progress.set_postfix_str(case["name"][:80])

                try:
                    passed = run_inference_case(
                        inference_main,
                        task,
                    )
                except KeyboardInterrupt:
                    interrupted_case = case["name"]
                    completed_cases += 1
                    raise

                completed_cases += 1

                if passed:
                    actual_passes += 1

                classify_result(
                    case=case,
                    passed=passed,
                    unexpected_failures=(unexpected_failures),
                    expected_failures=(expected_failures),
                    unexpected_passes=(unexpected_passes),
                )

                progress.set_postfix_str(
                    "unexpected_failures="
                    f"{len(unexpected_failures)} "
                    "expected_failures="
                    f"{len(expected_failures)} "
                    "unexpected_passes="
                    f"{len(unexpected_passes)}"
                )

                progress.update(1)

    except KeyboardInterrupt:
        cleanup_logging_handlers()

        write_summary(
            tasks,
            completed_cases,
            actual_passes,
            unexpected_failures,
            expected_failures,
            unexpected_passes,
            interrupted_case=interrupted_case,
        )

        print("\nInference integration suite interrupted.")
        print(f"Interrupted case: {interrupted_case}")
        print(f"Partial results: {RESULTS_CSV}")
        print(f"Summary: {SUMMARY_PATH}")

        raise

    cleanup_logging_handlers()

    write_summary(
        tasks,
        completed_cases,
        actual_passes,
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
            "Inference integration suite completed with "
            + " and ".join(problems)
            + f". See {RESULTS_CSV}, "
            f"{GENERATED_CONFIG_DIR}, and {LOG_DIR}."
        )

    print("Training and inference integration suite completed successfully.")
    print(f"Shared training model: {TRAINING_EXPERIMENT_DIR}")
    print(f"Case experiments: {OUTPUT_DIR}")
    print(f"Generated configs: {GENERATED_CONFIG_DIR}")
    print(f"Results: {RESULTS_CSV}")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Logs: {LOG_DIR}")


if __name__ == "__main__":
    main()
