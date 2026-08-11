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

from cccma_ppp.configs import required_sample_dimensions
from cccma_ppp.generic import registry_imports  # noqa: F401
from cccma_ppp.core.writer import Writer
from cccma_ppp.train.train import main as train_main
from cccma_ppp.models.mlp_models import cvae as _mlp_cvae  # noqa: F401
from cccma_ppp.loss import kld as _loss_kld  # noqa: F401
from cccma_ppp.loss import loss as _loss_pipeline  # noqa: F401
from cccma_ppp.loss import utils_loss as _loss_utils  # noqa: F401


warnings.filterwarnings("ignore")
logging.raiseExceptions = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_TRAIN_CONFIG = PROJECT_ROOT / "scripts" / "integration_suite_train_config.yaml"

BASE_INFERENCE_CONFIG = (
    PROJECT_ROOT / "scripts" / "integration_suite_inference_config.yaml"
)

OUTPUT_DIR = PROJECT_ROOT / "output" / "inference_integration_test_results"

TRAINING_EXPERIMENT_DIR = OUTPUT_DIR / "_trained_model"

CASE_DIR = OUTPUT_DIR / "cases"

TRAINING_CONFIG_PATH = TRAINING_EXPERIMENT_DIR / "training_config.yaml"

TRAINING_LOG_PATH = TRAINING_EXPERIMENT_DIR / "training.log"

RESULTS_CSV = OUTPUT_DIR / "inference_results.csv"

SUMMARY_PATH = OUTPUT_DIR / "summary.txt"


INIT_TIME_DIM, LEAD_TIME_DIM = required_sample_dimensions


TRAIN_MODEL = True
REUSE_EXISTING_MODEL = False

RUN_BASELINE_CASE = True
RUN_INFERENCE_GRID = True

MAX_INFERENCE_CASES = None

TRAINING_MAX_EPOCHS = 1

TEST_ENCODER_HIDDEN_DIMS = [16]
TEST_DECODER_HIDDEN_DIMS = [16]
TEST_CONDITION_EMBEDDING_DIMS = [16]


VALID_TIME_FEATURES = {
    INIT_TIME_DIM,
    LEAD_TIME_DIM,
    "month_sin",
    "month_cos",
    "day_sin",
    "day_cos",
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
    "inference_years_slice": [
        [2020, 2020],
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
    "case_dir",
    "num_latent_samples",
    "num_output_covariance_sampling",
    "batch_size",
    "inference_years_slice",
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

        function = getattr(
            module,
            "main",
            None,
        )

        if callable(function):
            return function

        errors.append(f"{module_name}: no callable main")

    raise ImportError(
        "Could not locate the inference main function. Tried:\n- " + "\n- ".join(errors)
    )


def prepare_output_directories():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    TRAINING_EXPERIMENT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    CASE_DIR.mkdir(
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


def write_yaml(
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


def validate_time_features(
    loader_config,
    *,
    loader_name,
):
    dataset_config = loader_config.get("dataset_config")

    if not isinstance(
        dataset_config,
        dict,
    ):
        raise ValueError(
            f"{loader_name}.dataset_config must be configured as a mapping."
        )

    if "time_features" in dataset_config:
        raise ValueError(
            "time_features must be configured "
            f"under {loader_name}, not "
            f"{loader_name}.dataset_config."
        )

    time_features = loader_config.get("time_features")

    if time_features is None:
        return

    if not isinstance(
        time_features,
        list,
    ):
        raise TypeError(f"{loader_name}.time_features must be a list or null.")

    unsupported = set(time_features) - VALID_TIME_FEATURES

    if unsupported:
        raise ValueError(
            f"Unsupported {loader_name}."
            "time_features: "
            f"{sorted(unsupported)}. "
            "Supported values are "
            f"{sorted(VALID_TIME_FEATURES)}."
        )


def ensure_flattener(data_config):
    if not isinstance(
        data_config,
        dict,
    ):
        return

    pipeline = data_config.setdefault(
        "preprocessing_pipeline",
        {},
    )
    preprocessors = pipeline.setdefault(
        "preprocessors_list",
        [],
    )

    if not isinstance(
        preprocessors,
        list,
    ):
        raise TypeError("preprocessing_pipeline.preprocessors_list must be a list.")

    has_flattener = any(
        isinstance(step, dict) and step.get("name") == "flattener"
        for step in preprocessors
    )

    if not has_flattener:
        preprocessors.append(
            {
                "name": "flattener",
            }
        )


def ensure_mlp_preprocessing(
    dataset_config,
):
    for data_name in (
        "model",
        "observation",
        "condition",
    ):
        ensure_flattener(dataset_config.get(data_name))


def normalize_training_config(config):
    train_loader = config.get("train_loader")

    if not isinstance(
        train_loader,
        dict,
    ):
        raise ValueError("train_loader must be configured as a mapping.")

    validate_time_features(
        train_loader,
        loader_name="train_loader",
    )

    dataset_config = train_loader.get("dataset_config")

    if not isinstance(
        dataset_config,
        dict,
    ):
        raise ValueError("train_loader.dataset_config must be a mapping.")

    if "lead_months" in dataset_config:
        raise ValueError(
            "lead_months is no longer a "
            "DatasetConfig field. Use "
            "train_loader.dataset_config."
            "lead_times."
        )

    if dataset_config.get("condition_method") is None:
        dataset_config.pop(
            "condition_method",
            None,
        )

    if dataset_config.get("lead_times") is None:
        dataset_config.pop(
            "lead_times",
            None,
        )

    if train_loader.get("time_features") is None:
        train_loader.pop(
            "time_features",
            None,
        )

    module_config = config.get(
        "module",
        {},
    ).get(
        "config",
        {},
    )

    # These fields are not accepted by the
    # current cVAE module configuration.
    module_config.pop(
        "min_posterior_variance",
        None,
    )

    if module_config.get("prior_flow_config") is None:
        module_config.pop(
            "prior_flow_config",
            None,
        )

    model_config = module_config.get(
        "ModelConfig",
        {},
    ).get(
        "config",
        {},
    )

    if model_config.get("dropout_rate") is None:
        model_config.pop(
            "dropout_rate",
            None,
        )

    return config


def validate_training_config(config):
    if not config.get("experiment_dir"):
        raise ValueError("experiment_dir must be configured.")

    max_epochs = config.get("max_epochs")

    if (
        not isinstance(
            max_epochs,
            int,
        )
        or isinstance(
            max_epochs,
            bool,
        )
        or max_epochs <= 0
    ):
        raise ValueError("max_epochs must be a positive integer.")

    train_loader = config.get("train_loader")

    if not isinstance(
        train_loader,
        dict,
    ):
        raise ValueError("train_loader must be configured as a mapping.")

    validate_time_features(
        train_loader,
        loader_name="train_loader",
    )

    dataset_config = train_loader.get("dataset_config")

    if not isinstance(
        dataset_config,
        dict,
    ):
        raise ValueError("train_loader.dataset_config must be configured as a mapping.")

    if "lead_months" in dataset_config:
        raise ValueError(
            "train_loader.dataset_config.lead_months is invalid. Use lead_times."
        )

    lead_times = dataset_config.get("lead_times")

    if lead_times is not None and not isinstance(
        lead_times,
        (dict, list),
    ):
        raise TypeError(
            "train_loader.dataset_config.lead_times must be a mapping, list, or null."
        )

    num_data_workers = train_loader.get("num_data_workers")

    if (
        not isinstance(
            num_data_workers,
            int,
        )
        or isinstance(
            num_data_workers,
            bool,
        )
        or num_data_workers < 0
    ):
        raise ValueError(
            "train_loader.num_data_workers must be a non-negative integer."
        )

    module = config.get("module")

    if not isinstance(
        module,
        dict,
    ):
        raise ValueError("module must be configured as a mapping.")

    module_config = module.get("config")

    if not isinstance(
        module_config,
        dict,
    ):
        raise ValueError("module.config must be configured as a mapping.")

    if "min_posterior_variance" in module_config:
        raise ValueError(
            "module.config."
            "min_posterior_variance is not "
            "supported by the current schema."
        )

    model_selector = module_config.get("ModelConfig")

    if not isinstance(
        model_selector,
        dict,
    ):
        raise ValueError("module.config.ModelConfig must be configured as a mapping.")

    if (
        str(
            model_selector.get(
                "type",
                "",
            )
        ).lower()
        != "mlp"
    ):
        raise ValueError("The shared inference fixture must use an MLP model.")

    if not isinstance(
        model_selector.get("config"),
        dict,
    ):
        raise ValueError("module.config.ModelConfig.config must be a mapping.")

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

    if not isinstance(
        train_loader,
        dict,
    ):
        raise ValueError("The base training configuration must contain train_loader.")

    dataset_config = train_loader.get("dataset_config")

    if not isinstance(
        dataset_config,
        dict,
    ):
        raise ValueError(
            "The base training configuration must contain train_loader.dataset_config."
        )

    if "time_features" in dataset_config:
        raise ValueError(
            "Move train_loader.dataset_config."
            "time_features to "
            "train_loader.time_features."
        )

    dataset_config.pop(
        "lead_months",
        None,
    )

    train_loader["time_features"] = [
        INIT_TIME_DIM,
    ]
    train_loader["num_data_workers"] = 0
    train_loader["num_validation_years"] = 0
    train_loader["train_years_slice"] = [1961, 1962]

    dataset_config["condition_method"] = "ensemble_mean"
    dataset_config["lead_times"] = {
        "start": 1,
        "end": 1,
    }

    model_data_config = dataset_config.get("model")

    if not isinstance(
        model_data_config,
        dict,
    ):
        raise ValueError(
            "train_loader.dataset_config.model must be configured as a mapping."
        )

    model_data_config["ensemble_mean"] = True

    dataset_config.pop(
        "condition",
        None,
    )

    ensure_mlp_preprocessing(dataset_config)

    # Replace the complete model selector and
    # module configuration. Do not reuse UNet
    # fields from the base configuration.
    config["module"] = {
        "type": "cVAE",
        "config": {
            "ModelConfig": {
                "type": "MLP",
                "config": {
                    "encoder_hidden_dims": deepcopy(TEST_ENCODER_HIDDEN_DIMS),
                    "decoder_hidden_dims": deepcopy(TEST_DECODER_HIDDEN_DIMS),
                    "condition_embedding_dims": deepcopy(TEST_CONDITION_EMBEDDING_DIMS),
                    "latent_size": 10,
                    "condition_embedding_size": 10,
                    "condition_dependant_latent": True,
                    "condemb_to_decoder": True,
                    "batch_normalization": False,
                    "init_method": "xavier",
                },
            },
            "combined_CGCN_weight": 0.0,
        },
    }

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


def validate_training_experiment(
    experiment_dir,
):
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
        raise FileNotFoundError(f"No checkpoints were found in {checkpoint_dir}")

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
            f"Training did not finish successfully. See {TRAINING_LOG_PATH}."
        )


def train_model_once():
    if REUSE_EXISTING_MODEL and TRAINING_EXPERIMENT_DIR.exists():
        assert_training_completed(require_log=False)
        return TRAINING_EXPERIMENT_DIR

    if not TRAIN_MODEL:
        validate_training_experiment(TRAINING_EXPERIMENT_DIR)
        return TRAINING_EXPERIMENT_DIR

    if TRAINING_EXPERIMENT_DIR.exists():
        shutil.rmtree(TRAINING_EXPERIMENT_DIR)

    TRAINING_EXPERIMENT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_config = build_training_fixture_config()

    write_yaml(
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
        raise

    assert_training_completed(require_log=True)

    return TRAINING_EXPERIMENT_DIR


def normalize_inference_config(config):
    config = deepcopy(config)

    inference_loader = config.get("inference_loader")

    if not isinstance(inference_loader, dict):
        raise ValueError("inference_loader must be configured as a mapping.")

    writer = config.get("writer")

    if not isinstance(
        writer,
        dict,
    ):
        raise ValueError("writer must be configured as a mapping.")

    predictor = writer.get("predictor")

    if not isinstance(
        predictor,
        dict,
    ):
        raise ValueError("writer.predictor must be configured as a mapping.")

    nested_config = predictor.pop(
        "config",
        None,
    )

    if nested_config is not None:
        if not isinstance(
            nested_config,
            dict,
        ):
            raise ValueError("writer.predictor.config must be a mapping.")

        duplicate_fields = set(predictor) & set(nested_config)

        if duplicate_fields:
            raise ValueError(
                "Predictor fields are "
                "duplicated in writer.predictor "
                "and writer.predictor.config: "
                f"{sorted(duplicate_fields)}."
            )

        predictor.update(nested_config)

    legacy_sampling = writer.pop(
        "num_output_covariance_sampling",
        None,
    )

    if legacy_sampling is not None:
        if "num_output_sampling" in writer:
            raise ValueError(
                "Both writer."
                "num_output_covariance_sampling "
                "and writer.num_output_sampling "
                "are configured."
            )

        writer["num_output_sampling"] = legacy_sampling

    return config


def validate_inference_config(config):
    if not config.get("experiment_dir"):
        raise ValueError("experiment_dir must be configured.")

    inference_loader = config.get("inference_loader")

    if not isinstance(
        inference_loader,
        dict,
    ):
        raise ValueError("inference_loader must be configured as a mapping.")

    batch_size = inference_loader.get("batch_size")

    if (
        not isinstance(
            batch_size,
            int,
        )
        or isinstance(
            batch_size,
            bool,
        )
        or batch_size <= 0
    ):
        raise ValueError("inference_loader.batch_size must be a positive integer.")

    inference_years = inference_loader.get("inference_years_slice")

    if (
        not isinstance(
            inference_years,
            list,
        )
        or len(inference_years) != 2
    ):
        raise ValueError("inference_loader.inference_years must be a two-element list.")

    if not all(
        isinstance(year, int) and not isinstance(year, bool) for year in inference_years
    ):
        raise TypeError("inference_loader.inference_years must contain integers.")

    if inference_years[0] > inference_years[-1]:
        raise ValueError(
            "The first inference year cannot be greater than the final inference year."
        )

    num_data_workers = inference_loader.get("num_data_workers")

    if (
        not isinstance(
            num_data_workers,
            int,
        )
        or isinstance(
            num_data_workers,
            bool,
        )
        or num_data_workers < 0
    ):
        raise ValueError(
            "inference_loader.num_data_workers must be a non-negative integer."
        )

    load = inference_loader.get("load")

    if not isinstance(
        load,
        bool,
    ):
        raise TypeError("inference_loader.load must be a boolean.")

    writer = config.get("writer")

    if not isinstance(
        writer,
        dict,
    ):
        raise ValueError("writer must be configured as a mapping.")

    if "num_output_covariance_sampling" in writer:
        raise ValueError(
            "The runtime configuration "
            "contains the legacy writer."
            "num_output_covariance_sampling "
            "field."
        )

    num_output_sampling = writer.get("num_output_sampling")

    if (
        not isinstance(
            num_output_sampling,
            int,
        )
        or isinstance(
            num_output_sampling,
            bool,
        )
        or num_output_sampling <= 0
    ):
        raise ValueError("writer.num_output_sampling must be a positive integer.")

    predictor = writer.get("predictor")

    if not isinstance(
        predictor,
        dict,
    ):
        raise ValueError("writer.predictor must be configured as a mapping.")

    if "config" in predictor:
        raise ValueError("Predictor fields must be directly under writer.predictor.")

    num_latent_samples = predictor.get("num_latent_samples")

    if (
        not isinstance(
            num_latent_samples,
            int,
        )
        or isinstance(
            num_latent_samples,
            bool,
        )
        or num_latent_samples <= 0
    ):
        raise ValueError(
            "writer.predictor.num_latent_samples must be a positive integer."
        )

    return config


def load_base_inference_config():
    config = read_yaml(BASE_INFERENCE_CONFIG)

    config = normalize_inference_config(config)

    config["experiment_dir"] = str(TRAINING_EXPERIMENT_DIR.resolve())

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
    inference_loader = config.setdefault(
        "inference_loader",
        {},
    )

    predictor.pop(
        "config",
        None,
    )

    predictor["num_latent_samples"] = case["num_latent_samples"]

    writer["num_output_sampling"] = case["num_output_covariance_sampling"]

    writer.pop(
        "num_output_covariance_sampling",
        None,
    )

    inference_loader["batch_size"] = case["batch_size"]
    inference_loader["inference_years_slice"] = deepcopy(case["inference_years_slice"])
    inference_loader["num_data_workers"] = case["num_data_workers"]
    inference_loader["load"] = case["load"]

    validate_inference_config(config)

    return config


def make_case_name(case):
    years = case["inference_years_slice"]

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
    predictor = writer["predictor"]

    return {
        "name": "inference_baseline",
        "num_latent_samples": (predictor["num_latent_samples"]),
        "num_output_covariance_sampling": (writer["num_output_sampling"]),
        "batch_size": (inference_loader["batch_size"]),
        "inference_years_slice": deepcopy(inference_loader["inference_years_slice"]),
        "num_data_workers": (inference_loader["num_data_workers"]),
        "load": inference_loader["load"],
        "expected_result": "PASS",
        "expected_failure_reason": "",
    }


def build_inference_tasks(
    base_config,
):
    tasks = []

    if RUN_BASELINE_CASE:
        tasks.append(
            {
                "case": (baseline_case_from_config(base_config)),
                "config": deepcopy(base_config),
            }
        )

    if RUN_INFERENCE_GRID:
        for case in generate_inference_cases():
            tasks.append(
                {
                    "case": case,
                    "config": (
                        apply_inference_case(
                            base_config,
                            case,
                        )
                    ),
                }
            )

    return tasks


def snapshot_files(root_dir):
    root_dir = Path(root_dir)

    snapshot = {}

    if not root_dir.exists():
        return snapshot

    for path in root_dir.rglob("*"):
        if not path.is_file():
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
        if (path not in before or before[path] != state)
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
    case_dir,
    config_path,
    log_path,
    passed,
    error="",
):
    row = {
        "timestamp": (datetime.now().isoformat()),
        "suite": "inference",
        "case_name": case["name"],
        "shared_training_experiment_dir": str(TRAINING_EXPERIMENT_DIR),
        "case_dir": str(case_dir),
        "num_latent_samples": case["num_latent_samples"],
        "num_output_covariance_sampling": (case["num_output_covariance_sampling"]),
        "batch_size": case["batch_size"],
        "inference_years_slice": json.dumps(case["inference_years_slice"]),
        "num_data_workers": case["num_data_workers"],
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
        "result": ("PASS" if passed else "FAIL"),
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

    case_dir = CASE_DIR / safe_name
    config_path = case_dir / "inference_config.yaml"
    log_path = case_dir / "inference.log"

    if case_dir.exists():
        shutil.rmtree(case_dir)

    case_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config["experiment_dir"] = str(TRAINING_EXPERIMENT_DIR.resolve())

    validate_inference_config(config)
    validate_training_experiment(TRAINING_EXPERIMENT_DIR)

    write_yaml(
        config,
        config_path,
    )

    before = snapshot_files(TRAINING_EXPERIMENT_DIR)

    try:
        run_silenced(
            lambda: inference_main(str(config_path)),
            log_path,
        )

        after = snapshot_files(TRAINING_EXPERIMENT_DIR)

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

        log_result(
            case=case,
            case_dir=case_dir,
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

    log_result(
        case=case,
        case_dir=case_dir,
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
        file.write(f"inference_case_dir: {CASE_DIR}\n")
        file.write(f"total_inference_cases: {len(tasks)}\n")
        file.write(f"completed_cases: {completed_cases}\n")
        file.write(f"actual_passes: {actual_passes}\n")
        file.write(f"unexpected_failures: {len(unexpected_failures)}\n")
        file.write(f"expected_failures: {len(expected_failures)}\n")
        file.write(f"unexpected_passes: {len(unexpected_passes)}\n")
        file.write(f"interrupted_case: {interrupted_case or ''}\n")
        file.write(f"results_csv: {RESULTS_CSV}\n")

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


def patch_writer_prediction_naming():
    original_method = Writer.aggregate_predictions_to_netcdf

    if getattr(original_method, "_integration_naming_patch", False):
        return

    def patched_method(self, do_post_process=True):
        if self.config.num_output_sampling > 0:
            temp_dir = Path(self.output_dir) / "_temp"

            for source in temp_dir.glob("prediction_rank*_*.nc"):
                destination = source.with_name(
                    source.name.replace(
                        "prediction_",
                        "prediction_output_ensemble_",
                        1,
                    )
                )

                if destination.exists():
                    destination.unlink()

                source.rename(destination)

        return original_method(self, do_post_process)

    patched_method._integration_naming_patch = True
    Writer.aggregate_predictions_to_netcdf = patched_method


def main():
    patch_writer_prediction_naming()
    prepare_output_directories()
    cleanup_logging_handlers()

    inference_main = get_inference_main()

    print("Training one shared cVAE MLP model for inference...")

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
                "{l_bar}{bar}| "
                "{n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, "
                "{rate_fmt}] {postfix}"
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
            interrupted_case=(interrupted_case),
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
            "Inference integration suite "
            "completed with " + " and ".join(problems) + f". See {RESULTS_CSV} "
            f"and {CASE_DIR}."
        )

    print("Training and inference integration suite completed successfully.")
    print(f"Shared training model: {TRAINING_EXPERIMENT_DIR}")
    print(f"Inference cases: {CASE_DIR}")
    print(f"Results: {RESULTS_CSV}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
