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

from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.raiseExceptions = False

import dacite

import cccma_ppp.models.mlp_models
import cccma_ppp.train.registry_imports

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
RUN_CVAE_DATASET_CASES = True
RUN_CVAE_MODULE_GRID = True
RUN_DETERMINISTIC_GRID = True


MAX_CVAE_DATASET_CASES = None
MAX_CVAE_MODULE_CASES = None
MAX_DETERMINISTIC_CASES = None


TEST_ENCODER_HIDDEN_DIMS = [64]
TEST_DECODER_HIDDEN_DIMS = [64]
TEST_CONDITION_EMBEDDING_DIMS = [64]


CVAE_DATASET_CASES = [
    {
        "name": "cvae_dataset_condition_none",
        "condition_method": None,
        "model_ensemble_mean": True,
    },
    {
        "name": "cvae_dataset_ensemble_mean",
        "condition_method": "ensemble_mean",
        "model_ensemble_mean": True,
    },
    {
        "name": "cvae_dataset_same_member",
        "condition_method": "same_member",
        "model_ensemble_mean": False,
    },
    {
        "name": "cvae_dataset_cross_ensemble",
        "condition_method": "cross_ensemble",
        "model_ensemble_mean": False,
    },
]


CVAE_MODULE_GRID = {
    "min_posterior_variance": [None, 2],
    "use_prior_flow": [False, True],
    "combined_CGCN_weight": [0.0, 0.1],
    "latent_size": [4, 10],
    # Keep this True unless/until the shape issue is fixed.
    # False can trigger:
    # mat1 and mat2 shapes cannot be multiplied.
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
    "time_features",
    "min_posterior_variance",
    "use_prior_flow",
    "combined_CGCN_weight",
    "latent_size",
    "condition_dependant_latent",
    "batch_normalization",
    "dropout_rate",
    "init_method",
    "append_mode",
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

    for logger_name in logger_dict:
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
        "time_features": json.dumps(params.get("time_features")),
        "min_posterior_variance": params.get("min_posterior_variance"),
        "use_prior_flow": params.get("use_prior_flow"),
        "combined_CGCN_weight": params.get("combined_CGCN_weight"),
        "latent_size": params.get("latent_size"),
        "condition_dependant_latent": params.get("condition_dependant_latent"),
        "batch_normalization": params.get("batch_normalization"),
        "dropout_rate": params.get("dropout_rate"),
        "init_method": params.get("init_method"),
        "append_mode": params.get("append_mode"),
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

    cfg["max_epochs"] = 1

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


def apply_cvae_dataset_params(
    cfg,
    params,
):
    ds_cfg = cfg["train_loader"]["dataset_config"]

    if "condition_method" in params:
        if params["condition_method"] is None:
            ds_cfg.pop(
                "condition_method",
                None,
            )
        else:
            ds_cfg["condition_method"] = params["condition_method"]

    if "model_ensemble_mean" in params:
        ds_cfg["model"]["ensemble_mean"] = params["model_ensemble_mean"]


def build_cvae_config_from_params(
    params,
):
    cfg = base_config()

    cfg["module"]["type"] = "cVAE"

    apply_cvae_dataset_params(
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

    normalize_nullable_fields(cfg)

    return cfg


def build_deterministic_config_from_params(
    params,
):
    cfg = base_config()

    ds_cfg = cfg["train_loader"]["dataset_config"]

    ds_cfg["time_features"] = params.get(
        "time_features",
        None,
    )

    if ds_cfg.get("time_features") is None:
        ds_cfg.pop(
            "time_features",
            None,
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

    normalize_nullable_fields(cfg)

    return cfg


def generate_cvae_dataset_cases():
    cases = deepcopy(CVAE_DATASET_CASES)

    if MAX_CVAE_DATASET_CASES is not None:
        cases = cases[:MAX_CVAE_DATASET_CASES]

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
    resumed_cfg["max_epochs"] = 2

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
            "params": {},
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
            "params": {},
            "runner": lambda: test_dataset_pipeline(OUTPUT_DIR),
            "config_path": GENERATED_CONFIG_DIR / "dataset_pipeline.yaml",
            "experiment_dir": OUTPUT_DIR / "dataset_pipeline",
            "log_path": LOG_DIR / "dataset_pipeline.log",
        }
    )


def add_cvae_dataset_tasks(tasks):
    if not RUN_CVAE_DATASET_CASES:
        return

    for case in generate_cvae_dataset_cases():
        cfg = build_cvae_config_from_params(case)

        tasks.append(
            {
                "suite": "cvae_dataset",
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
    add_cvae_dataset_tasks(tasks)
    add_cvae_module_tasks(tasks)
    add_deterministic_tasks(tasks)

    failures = []

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

                    log_baseline_result(
                        task=task,
                        passed=True,
                    )

                except Exception as exc:
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
                        passed=False,
                        error=error,
                    )

                    failures.append(case_name)

            else:
                passed = run_training_case(
                    suite=suite,
                    case_name=case_name,
                    model_type=task["model_type"],
                    cfg=task["cfg"],
                    params=task["params"],
                )

                if not passed:
                    failures.append(case_name)

            pbar.set_postfix_str(f"failures={len(failures)}")

            pbar.update(1)

    cleanup_logging_handlers()

    summary_path = OUTPUT_DIR / "summary.txt"

    with open(
        summary_path,
        "w",
    ) as f:
        f.write(f"total_cases: {len(tasks)}\n")
        f.write(f"failures: {len(failures)}\n")
        f.write(f"results_csv: {RESULTS_CSV}\n")
        f.write(f"log_dir: {LOG_DIR}\n")

        if failures:
            f.write("\nfailed_cases:\n")

            for failure in failures:
                f.write(f"- {failure}\n")


if __name__ == "__main__":
    main()
