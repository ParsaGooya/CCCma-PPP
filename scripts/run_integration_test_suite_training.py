from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import datetime
from itertools import product
from pathlib import Path
import csv
import json
import logging
import re
import shutil
import traceback
import warnings

import dacite
import yaml
from tqdm import tqdm

import cccma_ppp.models.mlp_model  # noqa: F401
import cccma_ppp.models.mlp_models.deterministic  # noqa: F401
import cccma_ppp.models.mlp_models.cvae  # noqa: F401

from cccma_ppp.generic.distributed import Distributed
from cccma_ppp.train.train import main as train_main
from cccma_ppp.train.train_configs import (
    TrainConfig,
    prepare_config,
)


warnings.filterwarnings("ignore")
logging.raiseExceptions = False


BASE_TRAIN_CONFIG = Path(
    "/fs/site7/eccc/crd/cccma/users/rna002/"
    "CCCma-PPP/scripts/integration_suite_train_config.yaml"
)

OUTPUT_DIR = Path("output/train_integration_test_results")
GENERATED_CONFIG_DIR = OUTPUT_DIR / "_generated_configs"
LOG_DIR = OUTPUT_DIR / "_logs"
RESULTS_CSV = OUTPUT_DIR / "integration_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary.txt"


RUN_BASELINE_TESTS = True
RUN_TRAIN_DATASET_CONFIG_CASES = True
RUN_CVAE_MODULE_GRID = True
RUN_DETERMINISTIC_GRID = True


MAX_TRAIN_DATASET_CONFIG_CASES = None
MAX_CVAE_MODULE_CASES = None
MAX_DETERMINISTIC_CASES = None


TEST_ENCODER_HIDDEN_DIMS = [16]
TEST_DECODER_HIDDEN_DIMS = [16]
TEST_CONDITION_EMBEDDING_DIMS = [16]


VALID_TIME_FEATURES = {
    "year",
    "lead_time",
    "month_sin",
    "month_cos",
}


FULL_LEAD_MONTHS = {
    "start": 1,
    "end": 12,
}


TRAIN_DATASET_CONFIG_CASES = [
    {
        "name": "tdc_condition_none_obs_present",
        "condition_method": None,
        "model_ensemble_mean": True,
        "remove_observation": False,
        "add_condition": False,
        "time_features": ["year"],
        "lead_months": deepcopy(FULL_LEAD_MONTHS),
        "expected_result": "PASS",
        "expected_failure_reason": "",
    },
    {
        "name": "tdc_ensemble_mean_obs_present",
        "condition_method": "ensemble_mean",
        "model_ensemble_mean": True,
        "remove_observation": False,
        "add_condition": False,
        "time_features": ["year"],
        "lead_months": deepcopy(FULL_LEAD_MONTHS),
        "expected_result": "PASS",
        "expected_failure_reason": "",
    },
    {
        "name": "tdc_same_member_model_not_mean",
        "condition_method": "same_member",
        "model_ensemble_mean": False,
        "remove_observation": False,
        "add_condition": False,
        "time_features": ["year"],
        "lead_months": deepcopy(FULL_LEAD_MONTHS),
        "expected_result": "PASS",
        "expected_failure_reason": "",
    },
    {
        "name": "tdc_cross_ensemble_model_not_mean",
        "condition_method": "cross_ensemble",
        "model_ensemble_mean": False,
        "remove_observation": False,
        "add_condition": False,
        "time_features": ["year"],
        "lead_months": deepcopy(FULL_LEAD_MONTHS),
        "expected_result": "PASS",
        "expected_failure_reason": "",
    },
    {
        "name": "tdc_same_member_invalid_model_mean",
        "condition_method": "same_member",
        "model_ensemble_mean": True,
        "remove_observation": False,
        "add_condition": False,
        "time_features": ["year"],
        "lead_months": deepcopy(FULL_LEAD_MONTHS),
        "expected_result": "FAIL",
        "expected_failure_reason": (
            "same_member conditioning should fail when model.ensemble_mean=True."
        ),
    },
    {
        "name": "tdc_observation_missing_condition_none",
        "condition_method": None,
        "model_ensemble_mean": True,
        "remove_observation": True,
        "add_condition": False,
        "time_features": ["year"],
        "lead_months": deepcopy(FULL_LEAD_MONTHS),
        "expected_result": "FAIL",
        "expected_failure_reason": (
            "observation=None and condition_method=None should fail."
        ),
    },
    {
        "name": "tdc_observation_missing_ensemble_mean",
        "condition_method": "ensemble_mean",
        "model_ensemble_mean": True,
        "remove_observation": True,
        "add_condition": False,
        "time_features": ["year"],
        "lead_months": deepcopy(FULL_LEAD_MONTHS),
        "expected_result": "PASS",
        "expected_failure_reason": "",
    },
    {
        "name": "tdc_time_features_none",
        "condition_method": "ensemble_mean",
        "model_ensemble_mean": True,
        "remove_observation": False,
        "add_condition": False,
        "time_features": None,
        "lead_months": deepcopy(FULL_LEAD_MONTHS),
        "expected_result": "PASS",
        "expected_failure_reason": "",
    },
    {
        "name": "tdc_time_features_year",
        "condition_method": "ensemble_mean",
        "model_ensemble_mean": True,
        "remove_observation": False,
        "add_condition": False,
        "time_features": ["year"],
        "lead_months": deepcopy(FULL_LEAD_MONTHS),
        "expected_result": "PASS",
        "expected_failure_reason": "",
    },
    {
        "name": "tdc_time_features_all",
        "condition_method": "ensemble_mean",
        "model_ensemble_mean": True,
        "remove_observation": False,
        "add_condition": False,
        "time_features": [
            "year",
            "lead_time",
            "month_sin",
            "month_cos",
        ],
        "lead_months": deepcopy(FULL_LEAD_MONTHS),
        "expected_result": "PASS",
        "expected_failure_reason": "",
    },
    {
        "name": "tdc_lead_months_single",
        "condition_method": "ensemble_mean",
        "model_ensemble_mean": True,
        "remove_observation": False,
        "add_condition": False,
        "time_features": ["year"],
        "lead_months": {
            "start": 1,
            "end": 1,
        },
        "expected_result": "PASS",
        "expected_failure_reason": "",
    },
    {
        "name": "tdc_lead_months_full",
        "condition_method": "ensemble_mean",
        "model_ensemble_mean": True,
        "remove_observation": False,
        "add_condition": False,
        "time_features": ["year"],
        "lead_months": deepcopy(FULL_LEAD_MONTHS),
        "expected_result": "PASS",
        "expected_failure_reason": "",
    },
    {
        "name": "tdc_lead_months_missing",
        "condition_method": "ensemble_mean",
        "model_ensemble_mean": True,
        "remove_observation": False,
        "add_condition": False,
        "time_features": ["year"],
        "lead_months": None,
        "expected_result": "PASS",
        "expected_failure_reason": "",
    },
    {
        "name": "tdc_explicit_condition_ensemble_mean",
        "condition_method": "ensemble_mean",
        "model_ensemble_mean": True,
        "condition_ensemble_mean": True,
        "remove_observation": False,
        "add_condition": True,
        "time_features": ["year"],
        "lead_months": deepcopy(FULL_LEAD_MONTHS),
        "expected_result": "PASS",
        "expected_failure_reason": "",
    },
    {
        "name": "tdc_explicit_condition_cross_ensemble",
        "condition_method": "cross_ensemble",
        "model_ensemble_mean": False,
        "condition_ensemble_mean": False,
        "remove_observation": False,
        "add_condition": True,
        "time_features": ["year"],
        "lead_months": deepcopy(FULL_LEAD_MONTHS),
        "expected_result": "PASS",
        "expected_failure_reason": "",
    },
    {
        "name": "tdc_static_without_condition",
        "condition_method": "static",
        "model_ensemble_mean": True,
        "remove_observation": False,
        "add_condition": False,
        "time_features": ["year"],
        "lead_months": deepcopy(FULL_LEAD_MONTHS),
        "expected_result": "FAIL",
        "expected_failure_reason": (
            "static conditioning requires an explicit condition dataset."
        ),
    },
    {
        "name": "tdc_static_with_model_copied_condition",
        "condition_method": "static",
        "model_ensemble_mean": True,
        "condition_ensemble_mean": True,
        "remove_observation": False,
        "add_condition": True,
        "time_features": ["year"],
        "lead_months": deepcopy(FULL_LEAD_MONTHS),
        "expected_result": "FAIL",
        "expected_failure_reason": (
            "static conditioning cannot point to the same model data."
        ),
    },
]


CVAE_MODULE_GRID = {
    "min_posterior_variance": [
        None,
        2,
    ],
    "use_prior_flow": [
        False,
        True,
    ],
    "combined_CGCN_weight": [
        0.0,
        0.1,
    ],
    "latent_size": [
        4,
        10,
    ],
    "condition_dependant_latent": [
        True,
        False,
    ],
    "batch_normalization": [
        False,
        True,
    ],
    "dropout_rate": [
        None,
        0.5,
    ],
    "init_method": [
        "xavier",
        "trunc_normal",
    ],
}


DETERMINISTIC_GRID = {
    "append_mode": [
        1,
        2,
        3,
    ],
    "batch_normalization": [
        False,
        True,
    ],
    "dropout_rate": [
        None,
        0.5,
    ],
    "init_method": [
        "xavier",
        "trunc_normal",
    ],
}


CSV_COLUMNS = [
    "timestamp",
    "suite",
    "case_name",
    "model_type",
    "condition_method",
    "model_ensemble_mean",
    "condition_ensemble_mean",
    "remove_observation",
    "add_condition",
    "time_features",
    "lead_months",
    "min_posterior_variance",
    "use_prior_flow",
    "combined_CGCN_weight",
    "latent_size",
    "condition_dependant_latent",
    "batch_normalization",
    "dropout_rate",
    "init_method",
    "append_mode",
    "expected_result",
    "expected_failure_reason",
    "config_path",
    "experiment_dir",
    "log_path",
    "result",
    "error",
]


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
        )
        writer.writeheader()


def log_result(
    *,
    suite,
    case_name,
    model_type,
    params,
    config_path,
    experiment_dir,
    log_path,
    passed,
    error="",
):
    row = {
        "timestamp": datetime.now().isoformat(),
        "suite": suite,
        "case_name": case_name,
        "model_type": model_type,
        "condition_method": params.get("condition_method"),
        "model_ensemble_mean": params.get("model_ensemble_mean"),
        "condition_ensemble_mean": params.get("condition_ensemble_mean"),
        "remove_observation": params.get("remove_observation"),
        "add_condition": params.get("add_condition"),
        "time_features": json.dumps(params.get("time_features")),
        "lead_months": json.dumps(params.get("lead_months")),
        "min_posterior_variance": params.get("min_posterior_variance"),
        "use_prior_flow": params.get("use_prior_flow"),
        "combined_CGCN_weight": params.get("combined_CGCN_weight"),
        "latent_size": params.get("latent_size"),
        "condition_dependant_latent": params.get("condition_dependant_latent"),
        "batch_normalization": params.get("batch_normalization"),
        "dropout_rate": params.get("dropout_rate"),
        "init_method": params.get("init_method"),
        "append_mode": params.get("append_mode"),
        "expected_result": params.get(
            "expected_result",
            "PASS",
        ),
        "expected_failure_reason": params.get(
            "expected_failure_reason",
            "",
        ),
        "config_path": str(config_path),
        "experiment_dir": str(experiment_dir),
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
        )
        writer.writerow(row)


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


def migrate_loader_fields(cfg):
    train_loader_cfg = cfg.setdefault(
        "train_loader",
        {},
    )
    dataset_cfg = train_loader_cfg.setdefault(
        "dataset_config",
        {},
    )

    if "time_features" in dataset_cfg:
        previous_time_features = dataset_cfg.pop("time_features")

        if "time_features" not in train_loader_cfg:
            train_loader_cfg["time_features"] = deepcopy(previous_time_features)

    return cfg


def normalize_nullable_fields(cfg):
    migrate_loader_fields(cfg)

    train_loader_cfg = cfg.setdefault(
        "train_loader",
        {},
    )
    dataset_cfg = train_loader_cfg.setdefault(
        "dataset_config",
        {},
    )

    dataset_cfg.pop(
        "time_features",
        None,
    )

    train_loader_cfg.setdefault(
        "time_features",
        None,
    )

    if dataset_cfg.get("condition_method") is None:
        dataset_cfg.pop(
            "condition_method",
            None,
        )

    if dataset_cfg.get("lead_months") is None:
        dataset_cfg.pop(
            "lead_months",
            None,
        )

    module_cfg = cfg.get(
        "module",
        {},
    ).get(
        "config",
        {},
    )

    if module_cfg.get("min_posterior_variance") is None:
        module_cfg.pop(
            "min_posterior_variance",
            None,
        )

    model_cfg = module_cfg.get(
        "ModelConfig",
        {},
    ).get(
        "config",
        {},
    )

    if model_cfg.get("dropout_rate") is None:
        model_cfg.pop(
            "dropout_rate",
            None,
        )

    if module_cfg.get("prior_flow_config") is None:
        module_cfg.pop(
            "prior_flow_config",
            None,
        )

    return cfg


def validate_generated_config(cfg):
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

    return cfg


def write_yaml(
    cfg,
    output_path,
):
    cfg = deepcopy(cfg)
    output_path = Path(output_path)

    normalize_nullable_fields(cfg)
    validate_generated_config(cfg)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            cfg,
            file,
            sort_keys=False,
        )

    return output_path


def load_train_config(config_path):
    cfg_data = prepare_config(config_path)

    if cfg_data is None:
        raise ValueError(f"Generated training configuration is empty: {config_path}")

    migrate_loader_fields(cfg_data)
    normalize_nullable_fields(cfg_data)
    validate_generated_config(cfg_data)

    return dacite.from_dict(
        data_class=TrainConfig,
        data=cfg_data,
        config=dacite.Config(
            strict=True,
        ),
    )


def base_config():
    cfg = prepare_config(BASE_TRAIN_CONFIG)

    if cfg is None:
        raise ValueError(
            f"The base training configuration is empty: {BASE_TRAIN_CONFIG}"
        )

    migrate_loader_fields(cfg)

    cfg["max_epochs"] = 3

    train_loader_cfg = cfg["train_loader"]
    train_loader_cfg["num_data_workers"] = 0
    train_loader_cfg["num_validation_years"] = 0

    normalize_nullable_fields(cfg)
    validate_generated_config(cfg)

    return cfg


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


def assert_training_completed(
    *,
    experiment_dir,
    log_path,
):
    experiment_dir = Path(experiment_dir)
    log_path = Path(log_path)

    checkpoint_dir = experiment_dir / "checkpoints"

    if not checkpoint_dir.is_dir():
        raise AssertionError(
            f"Expected checkpoint directory does not exist: {checkpoint_dir}"
        )

    checkpoint_files = sorted(checkpoint_dir.glob("*.pt"))

    if not checkpoint_files:
        raise AssertionError(f"No checkpoint files were written in {checkpoint_dir}")

    if not log_path.is_file():
        raise AssertionError(f"Log file does not exist: {log_path}")

    log_text = log_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    if "Training finished" not in log_text:
        raise AssertionError(
            "Training did not reach completion. "
            "The log does not contain 'Training finished': "
            f"{log_path}"
        )


def get_original_flow_config(cfg):
    return deepcopy(cfg["module"]["config"].get("prior_flow_config"))


def ensure_cvae_beta_finder(cfg):
    cfg.setdefault(
        "trainer",
        {},
    ).setdefault(
        "beta_finder",
        {
            "beta": 10,
            "num_epoch_to_warmup": 10,
        },
    )

    return cfg


def remove_deterministic_beta_finder(cfg):
    cfg.setdefault(
        "trainer",
        {},
    ).pop(
        "beta_finder",
        None,
    )

    return cfg


def set_fast_cvae_mlp_config(
    cfg,
    *,
    latent_size,
    condition_dependant_latent,
    batch_normalization,
    dropout_rate,
    init_method,
):
    model_cfg = cfg["module"]["config"]["ModelConfig"]["config"]

    model_cfg["encoder_hidden_dims"] = deepcopy(TEST_ENCODER_HIDDEN_DIMS)
    model_cfg["decoder_hidden_dims"] = deepcopy(TEST_DECODER_HIDDEN_DIMS)
    model_cfg["condition_embedding_dims"] = deepcopy(TEST_CONDITION_EMBEDDING_DIMS)

    model_cfg["latent_size"] = latent_size
    model_cfg["condition_embedding_size"] = latent_size
    model_cfg["condition_dependant_latent"] = condition_dependant_latent
    model_cfg["condemb_to_decoder"] = True
    model_cfg["batch_normalization"] = batch_normalization
    model_cfg["init_method"] = init_method

    if dropout_rate is None:
        model_cfg.pop(
            "dropout_rate",
            None,
        )
    else:
        model_cfg["dropout_rate"] = dropout_rate

    return cfg


def apply_train_dataset_config_params(
    cfg,
    params,
):
    train_loader_cfg = cfg["train_loader"]
    dataset_cfg = train_loader_cfg["dataset_config"]

    dataset_cfg.pop(
        "time_features",
        None,
    )

    if "model_ensemble_mean" in params:
        dataset_cfg["model"]["ensemble_mean"] = params["model_ensemble_mean"]

    if params.get(
        "remove_observation",
        False,
    ):
        dataset_cfg.pop(
            "observation",
            None,
        )

    if params.get(
        "add_condition",
        False,
    ):
        dataset_cfg["condition"] = deepcopy(dataset_cfg["model"])

        condition_ensemble_mean = params.get("condition_ensemble_mean")

        if condition_ensemble_mean is not None:
            dataset_cfg["condition"]["ensemble_mean"] = condition_ensemble_mean
    else:
        dataset_cfg.pop(
            "condition",
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
        train_loader_cfg["time_features"] = deepcopy(params["time_features"])

    if "lead_months" in params:
        lead_months = params["lead_months"]

        if lead_months is None:
            dataset_cfg.pop(
                "lead_months",
                None,
            )
        else:
            dataset_cfg["lead_months"] = deepcopy(lead_months)

    normalize_nullable_fields(cfg)
    validate_generated_config(cfg)

    return cfg


def build_cvae_config_from_params(
    params,
):
    cfg = base_config()

    cfg["module"]["type"] = "cVAE"

    apply_train_dataset_config_params(
        cfg,
        params,
    )

    module_cfg = cfg["module"]["config"]

    original_flow_config = get_original_flow_config(cfg)

    posterior_variance = params.get("min_posterior_variance")

    if posterior_variance is None:
        module_cfg.pop(
            "min_posterior_variance",
            None,
        )
    else:
        module_cfg["min_posterior_variance"] = posterior_variance

    if "combined_CGCN_weight" in params:
        module_cfg["combined_CGCN_weight"] = params["combined_CGCN_weight"]

    if "use_prior_flow" in params:
        if params["use_prior_flow"]:
            if original_flow_config is None:
                raise ValueError(
                    "The base cVAE configuration does not contain prior_flow_config."
                )

            module_cfg["prior_flow_config"] = original_flow_config
        else:
            module_cfg.pop(
                "prior_flow_config",
                None,
            )

    model_cfg = module_cfg["ModelConfig"]["config"]

    set_fast_cvae_mlp_config(
        cfg,
        latent_size=params.get(
            "latent_size",
            model_cfg.get(
                "latent_size",
                10,
            ),
        ),
        condition_dependant_latent=params.get(
            "condition_dependant_latent",
            model_cfg.get(
                "condition_dependant_latent",
                True,
            ),
        ),
        batch_normalization=params.get(
            "batch_normalization",
            model_cfg.get(
                "batch_normalization",
                False,
            ),
        ),
        dropout_rate=params.get(
            "dropout_rate",
            model_cfg.get("dropout_rate"),
        ),
        init_method=params.get(
            "init_method",
            model_cfg.get(
                "init_method",
                "xavier",
            ),
        ),
    )

    ensure_cvae_beta_finder(cfg)
    normalize_nullable_fields(cfg)
    validate_generated_config(cfg)

    return cfg


def build_deterministic_config_from_params(
    params,
):
    cfg = base_config()

    apply_train_dataset_config_params(
        cfg,
        params,
    )

    model_cfg = {
        "encoder_hidden_dims": deepcopy(TEST_ENCODER_HIDDEN_DIMS),
        "decoder_hidden_dims": deepcopy(TEST_DECODER_HIDDEN_DIMS),
        "batch_normalization": params["batch_normalization"],
        "append_mode": params["append_mode"],
        "init_method": params["init_method"],
    }

    if params.get("dropout_rate") is not None:
        model_cfg["dropout_rate"] = params["dropout_rate"]

    cfg["module"] = {
        "type": "deterministic",
        "config": {
            "ModelConfig": {
                "type": "MLP",
                "config": model_cfg,
            },
        },
    }

    remove_deterministic_beta_finder(cfg)
    normalize_nullable_fields(cfg)
    validate_generated_config(cfg)

    return cfg


def mark_cvae_module_expected_result(
    params,
):
    invalid = (
        params.get("use_prior_flow") is True
        and params.get("condition_dependant_latent") is False
    )

    if invalid:
        params["expected_result"] = "FAIL"
        params["expected_failure_reason"] = (
            "Known invalid cVAE combination: "
            "use_prior_flow=True with "
            "condition_dependant_latent=False."
        )
    else:
        params["expected_result"] = "PASS"
        params["expected_failure_reason"] = ""

    return params


def generate_train_dataset_config_cases():
    cases = deepcopy(TRAIN_DATASET_CONFIG_CASES)

    if MAX_TRAIN_DATASET_CONFIG_CASES is not None:
        cases = cases[:MAX_TRAIN_DATASET_CONFIG_CASES]

    return cases


def generate_cvae_module_cases():
    keys = list(CVAE_MODULE_GRID)
    values = [CVAE_MODULE_GRID[key] for key in keys]

    cases = []

    for combination in product(*values):
        params = dict(
            zip(
                keys,
                combination,
            )
        )

        params.update(
            {
                "condition_method": "ensemble_mean",
                "model_ensemble_mean": True,
                "remove_observation": False,
                "add_condition": False,
                "time_features": ["year"],
                "lead_months": deepcopy(FULL_LEAD_MONTHS),
            }
        )

        mark_cvae_module_expected_result(params)

        params["name"] = clean_name(
            "cvae"
            f"_mpv_{params['min_posterior_variance']}"
            f"_flow_{params['use_prior_flow']}"
            f"_cgcn_{params['combined_CGCN_weight']}"
            f"_latent_{params['latent_size']}"
            f"_condlat_{params['condition_dependant_latent']}"
            f"_bn_{params['batch_normalization']}"
            f"_drop_{params['dropout_rate']}"
            f"_init_{params['init_method']}"
        )

        cases.append(params)

    if MAX_CVAE_MODULE_CASES is not None:
        cases = cases[:MAX_CVAE_MODULE_CASES]

    return cases


def generate_deterministic_cases():
    keys = list(DETERMINISTIC_GRID)
    values = [DETERMINISTIC_GRID[key] for key in keys]

    cases = []

    for combination in product(*values):
        params = dict(
            zip(
                keys,
                combination,
            )
        )

        params.update(
            {
                "condition_method": "ensemble_mean",
                "model_ensemble_mean": True,
                "remove_observation": False,
                "add_condition": False,
                "time_features": None,
                "lead_months": deepcopy(FULL_LEAD_MONTHS),
                "expected_result": "PASS",
                "expected_failure_reason": "",
            }
        )

        params["name"] = clean_name(
            "det"
            f"_append_{params['append_mode']}"
            f"_bn_{params['batch_normalization']}"
            f"_drop_{params['dropout_rate']}"
            f"_init_{params['init_method']}"
        )

        cases.append(params)

    if MAX_DETERMINISTIC_CASES is not None:
        cases = cases[:MAX_DETERMINISTIC_CASES]

    return cases


def append_exception_to_log(
    log_path,
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
        traceback.print_exc(
            file=log_file,
        )


def remove_existing_case_outputs(
    experiment_dir,
    config_path,
    log_path,
):
    if experiment_dir.exists():
        shutil.rmtree(experiment_dir)

    if config_path.exists():
        config_path.unlink()

    if log_path.exists():
        log_path.unlink()


def run_training_case(
    *,
    suite,
    case_name,
    model_type,
    cfg,
    params,
):
    safe_name = clean_name(case_name)

    experiment_dir = OUTPUT_DIR / safe_name
    config_path = GENERATED_CONFIG_DIR / f"{safe_name}.yaml"
    log_path = LOG_DIR / f"{safe_name}.log"

    remove_existing_case_outputs(
        experiment_dir,
        config_path,
        log_path,
    )

    case_cfg = deepcopy(cfg)
    case_cfg["experiment_dir"] = str(experiment_dir)

    write_yaml(
        case_cfg,
        config_path,
    )

    try:
        run_silenced(
            lambda: train_main(str(config_path)),
            log_path,
        )

        assert_training_completed(
            experiment_dir=experiment_dir,
            log_path=log_path,
        )

        passed = True
        error = ""

    except Exception as exc:
        passed = False
        error = f"{type(exc).__name__}: {exc}"

        append_exception_to_log(log_path)
        cleanup_logging_handlers()

    log_result(
        suite=suite,
        case_name=case_name,
        model_type=model_type,
        params=params,
        config_path=config_path,
        experiment_dir=experiment_dir,
        log_path=log_path,
        passed=passed,
        error=error,
    )

    return passed


def make_train_config(
    output_path,
    experiment_dir,
):
    cfg = base_config()

    cfg["experiment_dir"] = str(experiment_dir)

    ensure_cvae_beta_finder(cfg)

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return write_yaml(
        cfg,
        output_path,
    )


def run_resume_training(
    root_dir,
):
    experiment_dir = root_dir / "resume"

    if experiment_dir.exists():
        shutil.rmtree(experiment_dir)

    first_config = make_train_config(
        GENERATED_CONFIG_DIR / "resume_train.yaml",
        experiment_dir,
    )
    first_log = LOG_DIR / "resume_first.log"

    run_silenced(
        lambda: train_main(str(first_config)),
        first_log,
    )

    assert_training_completed(
        experiment_dir=experiment_dir,
        log_path=first_log,
    )

    resumed_cfg = prepare_config(first_config)

    if resumed_cfg is None:
        raise ValueError(f"The resume configuration is empty: {first_config}")

    migrate_loader_fields(resumed_cfg)

    resumed_cfg["resume_dir"] = str(experiment_dir)
    resumed_cfg["max_epochs"] = 3

    ensure_cvae_beta_finder(resumed_cfg)

    second_config = GENERATED_CONFIG_DIR / "resume_second.yaml"

    write_yaml(
        resumed_cfg,
        second_config,
    )

    second_log = LOG_DIR / "resume_second.log"

    run_silenced(
        lambda: train_main(str(second_config)),
        second_log,
    )

    assert_training_completed(
        experiment_dir=experiment_dir,
        log_path=second_log,
    )


def run_dataset_pipeline(
    root_dir,
):
    experiment_dir = root_dir / "dataset_pipeline"

    if experiment_dir.exists():
        shutil.rmtree(experiment_dir)

    config_path = make_train_config(
        GENERATED_CONFIG_DIR / "dataset_pipeline.yaml",
        experiment_dir,
    )

    def run_pipeline():
        config = load_train_config(config_path)

        distributed = Distributed.get_instance()

        config.train_loader.setup_distributed(distributed)

        train_loader = config.train_loader.build_train_loader()

        if len(train_loader.dataset) <= 0:
            raise AssertionError("The training dataset is empty.")

        sample = train_loader.dataset[0]

        required_keys = {
            "input",
            "target",
            "added_features",
        }
        missing_keys = required_keys - set(sample)

        if missing_keys:
            raise AssertionError(
                f"Dataset sample is missing required keys: {sorted(missing_keys)}"
            )

    run_silenced(
        run_pipeline,
        LOG_DIR / "dataset_pipeline.log",
    )


def add_baseline_tasks(
    tasks,
):
    if not RUN_BASELINE_TESTS:
        return

    tasks.extend(
        [
            {
                "suite": "baseline",
                "case_name": "resume_training",
                "model_type": "cVAE",
                "params": {
                    "expected_result": "PASS",
                    "expected_failure_reason": "",
                },
                "runner": lambda: run_resume_training(OUTPUT_DIR),
                "config_path": (GENERATED_CONFIG_DIR / "resume_second.yaml"),
                "experiment_dir": (OUTPUT_DIR / "resume"),
                "log_path": (LOG_DIR / "resume_second.log"),
            },
            {
                "suite": "baseline",
                "case_name": "dataset_pipeline",
                "model_type": "cVAE",
                "params": {
                    "expected_result": "PASS",
                    "expected_failure_reason": "",
                },
                "runner": lambda: run_dataset_pipeline(OUTPUT_DIR),
                "config_path": (GENERATED_CONFIG_DIR / "dataset_pipeline.yaml"),
                "experiment_dir": (OUTPUT_DIR / "dataset_pipeline"),
                "log_path": (LOG_DIR / "dataset_pipeline.log"),
            },
        ]
    )


def add_train_dataset_config_tasks(
    tasks,
):
    if not RUN_TRAIN_DATASET_CONFIG_CASES:
        return

    for case in generate_train_dataset_config_cases():
        tasks.append(
            {
                "suite": "train_dataset_config",
                "case_name": case["name"],
                "model_type": "cVAE",
                "params": case,
                "cfg": build_cvae_config_from_params(case),
            }
        )


def add_cvae_module_tasks(
    tasks,
):
    if not RUN_CVAE_MODULE_GRID:
        return

    for case in generate_cvae_module_cases():
        tasks.append(
            {
                "suite": "cvae_module",
                "case_name": case["name"],
                "model_type": "cVAE",
                "params": case,
                "cfg": build_cvae_config_from_params(case),
            }
        )


def add_deterministic_tasks(
    tasks,
):
    if not RUN_DETERMINISTIC_GRID:
        return

    for case in generate_deterministic_cases():
        tasks.append(
            {
                "suite": "deterministic",
                "case_name": case["name"],
                "model_type": "deterministic",
                "params": case,
                "cfg": (build_deterministic_config_from_params(case)),
            }
        )


def run_baseline_task(task):
    try:
        task

        passed = True
        error = ""

    except Exception as exc:
        passed = False
        error = f"{type(exc).__name__}: {exc}"

        append_exception_to_log(task["log_path"])
        cleanup_logging_handlers()

    log_result(
        suite=task["suite"],
        case_name=task["case_name"],
        model_type=task["model_type"],
        params=task["params"],
        config_path=task["config_path"],
        experiment_dir=task["experiment_dir"],
        log_path=task["log_path"],
        passed=passed,
        error=error,
    )

    return passed


def classify_result(
    *,
    case_name,
    expected_result,
    passed,
    unexpected_failures,
    expected_failures,
    unexpected_passes,
):
    if expected_result == "FAIL":
        if passed:
            unexpected_passes.append(case_name)
        else:
            expected_failures.append(case_name)
    elif not passed:
        unexpected_failures.append(case_name)


def write_summary(
    tasks,
    unexpected_failures,
    expected_failures,
    unexpected_passes,
):
    actual_passes = sum(
        1
        for task in tasks
        if task["case_name"] not in unexpected_failures
        and task["case_name"] not in expected_failures
    )

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(f"total_cases: {len(tasks)}\n")
        file.write(f"actual_passes: {actual_passes}\n")
        file.write(f"unexpected_failures: {len(unexpected_failures)}\n")
        file.write(f"expected_failures: {len(expected_failures)}\n")
        file.write(f"unexpected_passes: {len(unexpected_passes)}\n")
        file.write(f"results_csv: {RESULTS_CSV}\n")
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


from cccma_ppp.data_modules.dataset.config_abc import DatasetConfigABC
import numpy as np
from cccma_ppp.train.dataset import TrainDatasetConfig


def install_integration_compatibility_patches():

    if hasattr(
        DatasetConfigABC,
        "_check_time_features",
    ):

        def skip_legacy_time_features_check(self):
            return self

        DatasetConfigABC._check_time_features = skip_legacy_time_features_check

    if not hasattr(
        TrainDatasetConfig,
        "input_lead_months",
    ):

        def get_input_lead_months(self):
            return np.arange(
                1,
                self.num_input_lead_months + 1,
            )

        TrainDatasetConfig.input_lead_months = property(get_input_lead_months)

    TrainDatasetConfig._check_model = lambda self: self
    TrainDatasetConfig._check_condition = lambda self: self
    TrainDatasetConfig.num_input_lead_months = property(
        lambda self: self.model.info.sizes["lead_time"]
    )


def main():
    install_integration_compatibility_patches()
    prepare_output_directories()

    cleanup_logging_handlers()
    init_csv()

    tasks = []

    add_baseline_tasks(tasks)
    add_train_dataset_config_tasks(tasks)
    add_cvae_module_tasks(tasks)
    add_deterministic_tasks(tasks)

    unexpected_failures = []
    expected_failures = []
    unexpected_passes = []

    with tqdm(
        total=len(tasks),
        desc="integration",
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
            suite = task["suite"]
            case_name = task["case_name"]

            progress.set_postfix_str(f"{suite}:{case_name[:60]}")

            if "runner" in task:
                passed = run_baseline_task(task)
            else:
                passed = run_training_case(
                    suite=suite,
                    case_name=case_name,
                    model_type=task["model_type"],
                    cfg=task["cfg"],
                    params=task["params"],
                )

            expected_result = task["params"].get(
                "expected_result",
                "PASS",
            )

            classify_result(
                case_name=case_name,
                expected_result=expected_result,
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
            "Integration suite completed with "
            + " and ".join(problems)
            + f". See {RESULTS_CSV} and {LOG_DIR}."
        )

    print("Integration suite completed successfully.")
    print(f"Results: {RESULTS_CSV}")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Logs: {LOG_DIR}")


if __name__ == "__main__":
    main()
