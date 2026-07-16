from pathlib import Path
from copy import deepcopy
from itertools import product
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr
import csv
import json
import logging
import re
import traceback
import warnings
import yaml

warnings.filterwarnings("ignore")
logging.raiseExceptions = False

from tqdm import tqdm
import dacite


import cccma_ppp.models.mlp_models  # noqa: F401
import cccma_ppp.core.cVAE_module  # noqa: F401
import cccma_ppp.core.deterministic_module  # noqa: F401
import cccma_ppp.loss.utils_loss  # noqa: F401
import cccma_ppp.preprocessing.utils_preprocessing  # noqa: F401
from cccma_ppp.models.normalized_flows import MAF  # noqa: F401
from cccma_ppp.models.normalized_flows import RealNVP  # noqa: F401

from cccma_ppp.train.train import main as train_main

from cccma_ppp.train.train_configs import (
    prepare_config,
    TrainConfig,
)

from cccma_ppp.generic.distributed import Distributed


BASE_TRAIN_CONFIG = Path(
    "/fs/site7/eccc/crd/cccma/users/rna002/CCCma-PPP/scripts/integration_suite_config.yaml"
)


OUTPUT_DIR = Path("output/integration_test_results")
GENERATED_CONFIG_DIR = OUTPUT_DIR / "_generated_configs"
LOG_DIR = OUTPUT_DIR / "_logs"
RESULTS_CSV = OUTPUT_DIR / "integration_results.csv"


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


TRAIN_DATASET_CONFIG_CASES = [
    {
        "name": "tdc_condition_none_obs_present",
        "condition_method": None,
        "model_ensemble_mean": True,
        "remove_observation": False,
        "add_condition": False,
        "time_features": ["year"],
        "lead_months": {"start": 1, "end": 12},
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
        "lead_months": {"start": 1, "end": 12},
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
        "lead_months": {"start": 1, "end": 12},
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
        "lead_months": {"start": 1, "end": 12},
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
        "lead_months": {"start": 1, "end": 12},
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
        "lead_months": {"start": 1, "end": 12},
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
        "lead_months": {"start": 1, "end": 12},
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
        "lead_months": {"start": 1, "end": 12},
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
        "lead_months": {"start": 1, "end": 12},
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
        "lead_months": {"start": 1, "end": 1},
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
        "lead_months": {"start": 1, "end": 12},
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
        "expected_result": "FAIL",
        "expected_failure_reason": (
            "lead_months=None may be invalid because training dataset logic "
            "expects lead_months to be iterable/resolved."
        ),
    },
    {
        "name": "tdc_explicit_condition_ensemble_mean",
        "condition_method": "ensemble_mean",
        "model_ensemble_mean": True,
        "condition_ensemble_mean": True,
        "remove_observation": False,
        "add_condition": True,
        "time_features": ["year"],
        "lead_months": {"start": 1, "end": 12},
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
        "lead_months": {"start": 1, "end": 12},
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
        "lead_months": {"start": 1, "end": 12},
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
        "lead_months": {"start": 1, "end": 12},
        "expected_result": "FAIL",
        "expected_failure_reason": (
            "static conditioning cannot point to the same model data. "
            "This case intentionally copies model config into condition."
        ),
    },
]


CVAE_MODULE_GRID = {
    "min_posterior_variance": [None, 2],
    "use_prior_flow": [False, True],
    "combined_CGCN_weight": [0.0, 0.1],
    "latent_size": [4, 10],
    "condition_dependant_latent": [True, False],
    "batch_normalization": [False, True],
    "dropout_rate": [None, 0.5],
    "init_method": ["xavier", "trunc_normal"],
}


DETERMINISTIC_GRID = {
    "append_mode": [1, 2, 3],
    "batch_normalization": [False, True],
    "dropout_rate": [None, 0.5],
    "init_method": ["xavier", "trunc_normal"],
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


def cleanup_logging_handlers():
    logger_names = [
        "",
        "training",
    ]

    logger_dict = logging.Logger.manager.loggerDict

    for logger_name in list(logger_dict):
        if logger_name.startswith("cccma_ppp"):
            logger_names.append(logger_name)

    seen = set()

    for logger_name in logger_names:
        if logger_name in seen:
            continue

        seen.add(logger_name)

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
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RESULTS_CSV,
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
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
        "expected_result": params.get("expected_result", "PASS"),
        "expected_failure_reason": params.get("expected_failure_reason", ""),
        "config_path": str(config_path),
        "experiment_dir": str(experiment_dir),
        "log_path": str(log_path),
        "result": "PASS" if passed else "FAIL",
        "error": error,
    }

    with open(
        RESULTS_CSV,
        "a",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=CSV_COLUMNS,
        )
        writer.writerow(row)


def clean_name(name):
    name = str(name)
    name = name.replace("None", "none")
    name = name.replace("none", "none")
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def write_yaml(
    cfg,
    output_yaml,
):
    output_yaml.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_yaml,
        "w",
    ) as f:
        yaml.safe_dump(
            cfg,
            f,
            sort_keys=False,
        )

    return output_yaml


def load_train_config(config_path):
    cfg_data = prepare_config(config_path)

    return dacite.from_dict(
        data_class=TrainConfig,
        data=cfg_data,
        config=dacite.Config(strict=True),
    )


def base_config():
    cfg = prepare_config(BASE_TRAIN_CONFIG)

    cfg["max_epochs"] = 3

    cfg["train_loader"]["num_data_workers"] = 0
    cfg["train_loader"]["num_validation_years"] = 0

    return cfg


def normalize_nullable_fields(cfg):
    ds_cfg = cfg.get(
        "train_loader",
        {},
    ).get(
        "dataset_config",
        {},
    )

    if ds_cfg.get("condition_method") is None:
        ds_cfg.pop(
            "condition_method",
            None,
        )

    if ds_cfg.get("time_features") is None:
        ds_cfg.pop(
            "time_features",
            None,
        )

    if ds_cfg.get("lead_months") is None:
        ds_cfg.pop(
            "lead_months",
            None,
        )

    module = cfg.get(
        "module",
        {},
    )

    module_cfg = module.get(
        "config",
        {},
    )

    if module_cfg.get("min_posterior_variance") is None:
        module_cfg.pop(
            "min_posterior_variance",
            None,
        )

    model_cfg = module_cfg.get("ModelConfig", {}).get("config", {})

    if model_cfg.get("dropout_rate") is None:
        model_cfg.pop(
            "dropout_rate",
            None,
        )

    return cfg


def run_silenced(
    func,
    log_path,
):
    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cleanup_logging_handlers()

    with open(
        log_path,
        "w",
    ) as log_file:
        try:
            with redirect_stdout(log_file), redirect_stderr(log_file):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return func()

        finally:
            cleanup_logging_handlers()


def assert_training_completed(
    *,
    experiment_dir,
    log_path,
):
    checkpoint_dir = experiment_dir / "checkpoints"

    if not checkpoint_dir.exists():
        raise AssertionError(
            f"Expected checkpoint directory does not exist: {checkpoint_dir}"
        )

    checkpoint_files = list(checkpoint_dir.glob("*.pt"))

    if not checkpoint_files:
        raise AssertionError(f"No checkpoint files were written in {checkpoint_dir}")

    if not log_path.exists():
        raise AssertionError(f"Log file does not exist: {log_path}")

    log_text = log_path.read_text(
        errors="replace",
    )

    if "Training finished" not in log_text:
        raise AssertionError(
            "Training did not reach completion. "
            f"Log does not contain 'Training finished': {log_path}"
        )


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

    cfg["experiment_dir"] = str(experiment_dir)

    normalize_nullable_fields(cfg)

    write_yaml(
        cfg,
        config_path,
    )

    try:

        def _run():
            train_main(str(config_path))

        run_silenced(
            _run,
            log_path,
        )

        assert_training_completed(
            experiment_dir=experiment_dir,
            log_path=log_path,
        )

        log_result(
            suite=suite,
            case_name=case_name,
            model_type=model_type,
            params=params,
            config_path=config_path,
            experiment_dir=experiment_dir,
            log_path=log_path,
            passed=True,
        )

        return True

    except Exception as exc:
        error = str(exc)

        log_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            log_path,
            "a",
        ) as log_file:
            traceback.print_exc(
                file=log_file,
            )

        cleanup_logging_handlers()

        log_result(
            suite=suite,
            case_name=case_name,
            model_type=model_type,
            params=params,
            config_path=config_path,
            experiment_dir=experiment_dir,
            log_path=log_path,
            passed=False,
            error=error,
        )

        return False


def get_original_flow_config(cfg):
    return deepcopy(cfg["module"]["config"].get("prior_flow_config"))


def ensure_cvae_beta_finder(cfg):
    cfg.setdefault(
        "trainer",
        {},
    )

    cfg["trainer"].setdefault(
        "beta_finder",
        {
            "beta": 10,
            "num_epoch_to_warmup": 10,
        },
    )

    return cfg


def remove_deterministic_beta_finder(cfg):
    if "trainer" in cfg:
        cfg["trainer"].pop(
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

    if dropout_rate is None:
        model_cfg.pop(
            "dropout_rate",
            None,
        )
    else:
        model_cfg["dropout_rate"] = dropout_rate

    model_cfg["init_method"] = init_method


def apply_train_dataset_config_params(
    cfg,
    params,
):
    ds_cfg = cfg["train_loader"]["dataset_config"]

    if "model_ensemble_mean" in params:
        ds_cfg["model"]["ensemble_mean"] = params["model_ensemble_mean"]

    if params.get("remove_observation") is True:
        ds_cfg.pop(
            "observation",
            None,
        )

    if params.get("add_condition") is True:
        ds_cfg["condition"] = deepcopy(ds_cfg["model"])

        if "condition_ensemble_mean" in params:
            ds_cfg["condition"]["ensemble_mean"] = params["condition_ensemble_mean"]

    else:
        ds_cfg.pop(
            "condition",
            None,
        )

    if "condition_method" in params:
        if params["condition_method"] is None:
            ds_cfg.pop(
                "condition_method",
                None,
            )
        else:
            ds_cfg["condition_method"] = params["condition_method"]

    if "time_features" in params:
        if params["time_features"] is None:
            ds_cfg.pop(
                "time_features",
                None,
            )
        else:
            ds_cfg["time_features"] = params["time_features"]

    if "lead_months" in params:
        if params["lead_months"] is None:
            ds_cfg.pop(
                "lead_months",
                None,
            )
        else:
            ds_cfg["lead_months"] = params["lead_months"]

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

    if "min_posterior_variance" in params:
        if params["min_posterior_variance"] is None:
            module_cfg.pop(
                "min_posterior_variance",
                None,
            )
        else:
            module_cfg["min_posterior_variance"] = params["min_posterior_variance"]

    if "combined_CGCN_weight" in params:
        module_cfg["combined_CGCN_weight"] = params["combined_CGCN_weight"]

    if "use_prior_flow" in params:
        if params["use_prior_flow"]:
            module_cfg["prior_flow_config"] = original_flow_config
        else:
            module_cfg["prior_flow_config"] = None

    model_cfg = cfg["module"]["config"]["ModelConfig"]["config"]

    set_fast_cvae_mlp_config(
        cfg,
        latent_size=params.get(
            "latent_size",
            model_cfg.get("latent_size", 10),
        ),
        condition_dependant_latent=params.get(
            "condition_dependant_latent",
            model_cfg.get("condition_dependant_latent", True),
        ),
        batch_normalization=params.get(
            "batch_normalization",
            model_cfg.get("batch_normalization", False),
        ),
        dropout_rate=params.get(
            "dropout_rate",
            model_cfg.get("dropout_rate"),
        ),
        init_method=params.get(
            "init_method",
            model_cfg.get("init_method", "xavier"),
        ),
    )

    ensure_cvae_beta_finder(cfg)

    normalize_nullable_fields(cfg)

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

    return cfg


def mark_cvae_module_expected_result(params):
    if (
        params.get("use_prior_flow") is True
        and params.get("condition_dependant_latent") is False
    ):
        params["expected_result"] = "FAIL"
        params["expected_failure_reason"] = (
            "Known invalid cVAE combo: use_prior_flow=True with "
            "condition_dependant_latent=False causes normalized-flow "
            "condition/input shape mismatch."
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
    keys = list(CVAE_MODULE_GRID.keys())
    values = [CVAE_MODULE_GRID[key] for key in keys]

    cases = []

    for combo in product(*values):
        params = dict(
            zip(
                keys,
                combo,
            )
        )

        params["condition_method"] = "ensemble_mean"
        params["model_ensemble_mean"] = True
        params["remove_observation"] = False
        params["add_condition"] = False
        params["time_features"] = ["year"]
        params["lead_months"] = {"start": 1, "end": 12}

        params = mark_cvae_module_expected_result(params)

        name = (
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

        params["name"] = clean_name(name)

        cases.append(params)

    if MAX_CVAE_MODULE_CASES is not None:
        cases = cases[:MAX_CVAE_MODULE_CASES]

    return cases


def generate_deterministic_cases():
    keys = list(DETERMINISTIC_GRID.keys())
    values = [DETERMINISTIC_GRID[key] for key in keys]

    cases = []

    for combo in product(*values):
        params = dict(
            zip(
                keys,
                combo,
            )
        )

        params["condition_method"] = "ensemble_mean"
        params["model_ensemble_mean"] = True
        params["remove_observation"] = False
        params["add_condition"] = False
        params["time_features"] = None
        params["lead_months"] = {"start": 1, "end": 12}
        params["expected_result"] = "PASS"
        params["expected_failure_reason"] = ""

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


def make_train_config(
    output_yaml: Path,
    experiment_dir: Path,
):
    cfg = base_config()

    cfg["experiment_dir"] = str(experiment_dir)

    ensure_cvae_beta_finder(cfg)

    normalize_nullable_fields(cfg)

    output_yaml.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_yaml,
        "w",
    ) as f:
        yaml.safe_dump(
            cfg,
            f,
            sort_keys=False,
        )

    return output_yaml


def test_resume_training(root_dir):
    exp_dir = root_dir / "resume"

    train_cfg = make_train_config(
        GENERATED_CONFIG_DIR / "resume_train.yaml",
        exp_dir,
    )

    log_path_1 = LOG_DIR / "resume_first.log"

    run_silenced(
        lambda: train_main(str(train_cfg)),
        log_path_1,
    )

    assert_training_completed(
        experiment_dir=exp_dir,
        log_path=log_path_1,
    )

    resumed_cfg = prepare_config(train_cfg)

    resumed_cfg["resume_dir"] = str(exp_dir)
    resumed_cfg["max_epochs"] = 3

    ensure_cvae_beta_finder(resumed_cfg)

    normalize_nullable_fields(resumed_cfg)

    resume_yaml = GENERATED_CONFIG_DIR / "resume_second.yaml"

    with open(
        resume_yaml,
        "w",
    ) as f:
        yaml.safe_dump(
            resumed_cfg,
            f,
            sort_keys=False,
        )

    log_path_2 = LOG_DIR / "resume_second.log"

    run_silenced(
        lambda: train_main(str(resume_yaml)),
        log_path_2,
    )

    assert_training_completed(
        experiment_dir=exp_dir,
        log_path=log_path_2,
    )


def test_dataset_pipeline(root_dir):
    exp_dir = root_dir / "dataset_pipeline"

    train_cfg = make_train_config(
        GENERATED_CONFIG_DIR / "dataset_pipeline.yaml",
        exp_dir,
    )

    def _run():
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

    run_silenced(
        _run,
        LOG_DIR / "dataset_pipeline.log",
    )


def add_baseline_tasks(tasks):
    if not RUN_BASELINE_TESTS:
        return

    tasks.append(
        {
            "suite": "baseline",
            "case_name": "resume_training",
            "model_type": "cVAE",
            "params": {
                "expected_result": "PASS",
                "expected_failure_reason": "",
            },
            "runner": lambda: test_resume_training(OUTPUT_DIR),
            "config_path": GENERATED_CONFIG_DIR / "resume_second.yaml",
            "experiment_dir": OUTPUT_DIR / "resume",
            "log_path": LOG_DIR / "resume_second.log",
        }
    )

    tasks.append(
        {
            "suite": "baseline",
            "case_name": "dataset_pipeline",
            "model_type": "cVAE",
            "params": {
                "expected_result": "PASS",
                "expected_failure_reason": "",
            },
            "runner": lambda: test_dataset_pipeline(OUTPUT_DIR),
            "config_path": GENERATED_CONFIG_DIR / "dataset_pipeline.yaml",
            "experiment_dir": OUTPUT_DIR / "dataset_pipeline",
            "log_path": LOG_DIR / "dataset_pipeline.log",
        }
    )


def add_train_dataset_config_tasks(tasks):
    if not RUN_TRAIN_DATASET_CONFIG_CASES:
        return

    for case in generate_train_dataset_config_cases():
        cfg = build_cvae_config_from_params(case)

        tasks.append(
            {
                "suite": "train_dataset_config",
                "case_name": case["name"],
                "model_type": "cVAE",
                "params": case,
                "cfg": cfg,
            }
        )


def add_cvae_module_tasks(tasks):
    if not RUN_CVAE_MODULE_GRID:
        return

    for case in generate_cvae_module_cases():
        cfg = build_cvae_config_from_params(case)

        tasks.append(
            {
                "suite": "cvae_module",
                "case_name": case["name"],
                "model_type": "cVAE",
                "params": case,
                "cfg": cfg,
            }
        )


def add_deterministic_tasks(tasks):
    if not RUN_DETERMINISTIC_GRID:
        return

    for case in generate_deterministic_cases():
        cfg = build_deterministic_config_from_params(case)

        tasks.append(
            {
                "suite": "deterministic",
                "case_name": case["name"],
                "model_type": "deterministic",
                "params": case,
                "cfg": cfg,
            }
        )


def log_baseline_result(
    *,
    task,
    passed,
    error="",
):
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


def main():
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
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    ) as pbar:
        for task in tasks:
            suite = task["suite"]
            case_name = task["case_name"]

            pbar.set_postfix_str(f"{suite}:{case_name[:60]}")

            if "runner" in task:
                try:
                    task
                    passed = True
                    error = ""

                except Exception as exc:
                    passed = False
                    error = str(exc)

                    task["log_path"].parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    with open(
                        task["log_path"],
                        "a",
                    ) as log_file:
                        traceback.print_exc(
                            file=log_file,
                        )

                    cleanup_logging_handlers()

                log_baseline_result(
                    task=task,
                    passed=passed,
                    error=error,
                )

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

            if expected_result == "FAIL" and not passed:
                expected_failures.append(case_name)

            elif expected_result == "FAIL" and passed:
                unexpected_passes.append(case_name)

            elif expected_result == "PASS" and not passed:
                unexpected_failures.append(case_name)

            pbar.set_postfix_str(
                (
                    f"unexpected_failures={len(unexpected_failures)} "
                    f"expected_failures={len(expected_failures)} "
                    f"unexpected_passes={len(unexpected_passes)}"
                )
            )

            pbar.update(1)

    cleanup_logging_handlers()

    summary_path = OUTPUT_DIR / "summary.txt"

    with open(
        summary_path,
        "w",
    ) as f:
        f.write(f"total_cases: {len(tasks)}\n")
        f.write(f"unexpected_failures: {len(unexpected_failures)}\n")
        f.write(f"expected_failures: {len(expected_failures)}\n")
        f.write(f"unexpected_passes: {len(unexpected_passes)}\n")
        f.write(f"results_csv: {RESULTS_CSV}\n")
        f.write(f"log_dir: {LOG_DIR}\n")

        if unexpected_failures:
            f.write("\nunexpected_failures:\n")

            for failure in unexpected_failures:
                f.write(f"- {failure}\n")

        if expected_failures:
            f.write("\nexpected_failures:\n")

            for failure in expected_failures:
                f.write(f"- {failure}\n")

        if unexpected_passes:
            f.write("\nunexpected_passes:\n")

            for failure in unexpected_passes:
                f.write(f"- {failure}\n")


if __name__ == "__main__":
    main()
