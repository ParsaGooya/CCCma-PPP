from __future__ import annotations

from tqdm import tqdm


from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    wait,
)
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any
import csv
import multiprocessing as mp
import os
import signal
import sys
import traceback


PARALLEL_WORKERS = 16
THREADS_PER_WORKER = 1
MAX_PENDING_TASKS = PARALLEL_WORKERS * 2
CPU_ONLY = True
ARCHIVE_INCOMPATIBLE_CHECKPOINT = True
SUITE_MODULE_NAME = "run_integration_test_suite_training"


_WORKER_MODULE: Any | None = None
_WORKER_TASKS: list[dict[str, Any]] | None = None


def configure_process_environment() -> None:

    thread_count = str(THREADS_PER_WORKER)

    os.environ["OMP_NUM_THREADS"] = thread_count
    os.environ["MKL_NUM_THREADS"] = thread_count
    os.environ["OPENBLAS_NUM_THREADS"] = thread_count
    os.environ["NUMEXPR_NUM_THREADS"] = thread_count
    os.environ["VECLIB_MAXIMUM_THREADS"] = thread_count
    os.environ["BLIS_NUM_THREADS"] = thread_count

    if CPU_ONLY:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


def load_suite_module() -> Any:

    configure_process_environment()

    module = import_module(SUITE_MODULE_NAME)

    return module


def build_tasks(
    suite: Any,
) -> list[dict[str, Any]]:

    tasks: list[dict[str, Any]] = []

    suite.add_baseline_tasks(tasks)
    suite.add_train_loader_config_tasks(tasks)
    suite.add_cvae_tasks(
        tasks,
        "MLP",
    )
    suite.add_cvae_tasks(
        tasks,
        "UNet",
    )
    suite.add_deterministic_tasks(
        tasks,
        "MLP",
    )
    suite.add_deterministic_tasks(
        tasks,
        "UNet",
    )

    suite.validate_tasks(tasks)

    return tasks


def task_key(
    task: dict[str, Any],
) -> tuple[str, str]:

    return (
        str(task["suite"]),
        str(task["case_name"]),
    )


def row_key(
    row: dict[str, Any],
) -> tuple[str, str]:

    return (
        str(
            row.get(
                "suite",
                "",
            )
        ),
        str(
            row.get(
                "case_name",
                "",
            )
        ),
    )


def task_paths(
    suite: Any,
    task: dict[str, Any],
) -> dict[str, Path]:

    if "runner" in task:
        return {
            "config_path": Path(task["config_path"]),
            "experiment_dir": Path(task["experiment_dir"]),
            "log_path": Path(task["log_path"]),
        }

    return suite.case_paths(task["case_name"])


def serialize_json(
    suite: Any,
    value: Any,
) -> str:

    return suite.serialize_json(value)


def build_result_row(
    suite: Any,
    *,
    task: dict[str, Any],
    passed: bool,
    error: str,
    paths: dict[str, Path],
) -> dict[str, Any]:

    params = task["params"]

    return {
        "timestamp": datetime.now().isoformat(),
        "suite": task["suite"],
        "case_name": task["case_name"],
        "module_type": task["module_type"],
        "architecture": task["architecture"],
        "model_type": params.get("model_type"),
        "condition_method": params.get("condition_method"),
        "model_ensemble_mean": params.get("model_ensemble_mean"),
        "condition_ensemble_mean": params.get("condition_ensemble_mean"),
        "remove_observation": params.get("remove_observation"),
        "add_condition": params.get("add_condition"),
        "time_features": serialize_json(
            suite,
            params.get("time_features"),
        ),
        "lead_months": serialize_json(
            suite,
            params.get("lead_months"),
        ),
        "posterior_variance_limits": serialize_json(
            suite,
            params.get("posterior_variance_limits"),
        ),
        "use_prior_flow": params.get("use_prior_flow"),
        "combined_CGCN_weight": params.get("combined_CGCN_weight"),
        "latent_size": params.get("latent_size"),
        "condition_dependant_latent": params.get("condition_dependant_latent"),
        "condemb_to_decoder": params.get("condemb_to_decoder"),
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
        "expected_error_pattern": params.get(
            "expected_error_pattern",
            "",
        ),
        "config_path": str(paths["config_path"]),
        "experiment_dir": str(paths["experiment_dir"]),
        "log_path": str(paths["log_path"]),
        "result": ("PASS" if passed else "FAIL"),
        "error": error,
    }


def archive_checkpoint(
    results_csv: Path,
    reason: str,
) -> Path:

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    archive_path = results_csv.with_name(f"{results_csv.name}.{reason}.{timestamp}")

    results_csv.replace(archive_path)

    return archive_path


def write_results_csv(
    suite: Any,
    rows: list[dict[str, Any]],
) -> None:

    suite.prepare_output_directories()

    results_csv = Path(suite.RESULTS_CSV)

    temporary_path = results_csv.with_suffix(results_csv.suffix + ".tmp")

    with temporary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=suite.CSV_COLUMNS,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    column: row.get(
                        column,
                        "",
                    )
                    for column in suite.CSV_COLUMNS
                }
            )

        file.flush()

        try:
            os.fsync(file.fileno())
        except OSError:
            pass

    temporary_path.replace(results_csv)


def append_result_row(
    suite: Any,
    row: dict[str, Any],
) -> None:

    results_csv = Path(suite.RESULTS_CSV)

    with results_csv.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=suite.CSV_COLUMNS,
            extrasaction="ignore",
        )

        writer.writerow(row)

        file.flush()

        try:
            os.fsync(file.fileno())
        except OSError:
            pass


def read_checkpoint_rows(
    suite: Any,
) -> list[dict[str, str]]:

    results_csv = Path(suite.RESULTS_CSV)

    if not results_csv.is_file():
        return []

    if results_csv.stat().st_size == 0:
        archive_path = archive_checkpoint(
            results_csv,
            "empty",
        )

        print(
            f"The empty checkpoint was archived as: {archive_path}",
            flush=True,
        )

        return []

    with results_csv.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        fieldnames = reader.fieldnames

        if not fieldnames:
            archive_path = archive_checkpoint(
                results_csv,
                "headerless",
            )

            print(
                f"The headerless checkpoint was archived as: {archive_path}",
                flush=True,
            )

            return []

        normalized_columns = {
            str(column).strip() for column in fieldnames if column is not None
        }

        missing_columns = set(suite.CSV_COLUMNS) - normalized_columns

        rows = [
            dict(row) for row in reader if row.get("suite") and row.get("case_name")
        ]

    if missing_columns:
        if ARCHIVE_INCOMPATIBLE_CHECKPOINT:
            archive_path = archive_checkpoint(
                results_csv,
                "incompatible_schema",
            )

            print(
                f"The incompatible checkpoint was archived as: {archive_path}",
                flush=True,
            )

            return []

        raise ValueError(
            f"The checkpoint is missing required columns: {sorted(missing_columns)}"
        )

    return rows


def validate_checkpoint_rows(
    tasks: list[dict[str, Any]],
    rows: list[dict[str, str]],
) -> bool:

    valid_keys = {task_key(task) for task in tasks}

    observed_keys: set[tuple[str, str]] = set()

    for row in rows:
        key = row_key(row)

        if key not in valid_keys:
            return False

        if key in observed_keys:
            return False

        observed_keys.add(key)

    return True


def prepare_checkpoint(
    suite: Any,
    tasks: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, str]],
]:

    suite.prepare_output_directories()

    results_csv = Path(suite.RESULTS_CSV)

    rows = read_checkpoint_rows(suite)

    if rows and not validate_checkpoint_rows(
        tasks,
        rows,
    ):
        if not ARCHIVE_INCOMPATIBLE_CHECKPOINT:
            raise ValueError("The checkpoint does not match the current task matrix.")

        archive_path = archive_checkpoint(
            results_csv,
            "task_matrix_mismatch",
        )

        print(
            "The checkpoint did not match the current task matrix "
            f"and was archived as: {archive_path}",
            flush=True,
        )

        rows = []

    if not rows:
        write_results_csv(
            suite,
            [],
        )

        print(
            "No usable checkpoint found. Starting all tasks.",
            flush=True,
        )

        return (
            list(tasks),
            [],
        )

    last_completed_row = rows[-1]
    retained_rows = rows[:-1]

    retained_keys = {row_key(row) for row in retained_rows}

    pending_tasks = [task for task in tasks if task_key(task) not in retained_keys]

    write_results_csv(
        suite,
        retained_rows,
    )

    print(
        "Checkpoint found.",
        flush=True,
    )
    print(
        f"Retained completed cases: {len(retained_rows)}",
        flush=True,
    )
    print(
        "Rerunning the last completed case: "
        f"{last_completed_row['suite']}:"
        f"{last_completed_row['case_name']}",
        flush=True,
    )
    print(
        f"Pending cases: {len(pending_tasks)}/{len(tasks)}",
        flush=True,
    )

    return (
        pending_tasks,
        retained_rows,
    )


def parse_passed(
    row: dict[str, str],
) -> bool:

    value = (
        str(
            row.get(
                "result",
                "",
            )
        )
        .strip()
        .upper()
    )

    if value == "PASS":
        return True

    if value == "FAIL":
        return False

    raise ValueError(f"Invalid checkpoint result: {row.get('result')!r}")


def restore_classification(
    suite: Any,
    *,
    tasks: list[dict[str, Any]],
    rows: list[dict[str, str]],
    unexpected_failures: list[str],
    expected_failures: list[str],
    unexpected_passes: list[str],
) -> None:

    tasks_by_key = {task_key(task): task for task in tasks}

    for row in rows:
        task = tasks_by_key[row_key(row)]

        suite.classify_result(
            task=task,
            passed=parse_passed(row),
            error=row.get(
                "error",
                "",
            ),
            unexpected_failures=unexpected_failures,
            expected_failures=expected_failures,
            unexpected_passes=unexpected_passes,
        )


def write_parallel_summary(
    suite: Any,
    *,
    total_tasks: int,
    completed_rows: list[dict[str, Any]],
    unexpected_failures: list[str],
    expected_failures: list[str],
    unexpected_passes: list[str],
) -> None:

    completed_count = len(completed_rows)
    remaining_count = total_tasks - completed_count
    failure_count = len(unexpected_failures) + len(expected_failures)
    pass_count = completed_count - failure_count

    summary_path = Path(suite.SUMMARY_PATH)

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(f"parallel_workers: {PARALLEL_WORKERS}\n")
        file.write(f"threads_per_worker: {THREADS_PER_WORKER}\n")
        file.write(f"cpu_only: {CPU_ONLY}\n")
        file.write(f"total_cases: {total_tasks}\n")
        file.write(f"completed_cases: {completed_count}\n")
        file.write(f"remaining_cases: {remaining_count}\n")
        file.write(f"actual_passes: {pass_count}\n")
        file.write(f"unexpected_failures: {len(unexpected_failures)}\n")
        file.write(f"expected_failures: {len(expected_failures)}\n")
        file.write(f"unexpected_passes: {len(unexpected_passes)}\n")
        file.write(f"results_csv: {suite.RESULTS_CSV}\n")
        file.write(f"log_dir: {suite.LOG_DIR}\n")

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

        for heading, values in groups:
            if not values:
                continue

            file.write(f"\n{heading}:\n")

            for value in values:
                file.write(f"- {value}\n")


def worker_initializer() -> None:

    global _WORKER_MODULE
    global _WORKER_TASKS

    configure_process_environment()

    suite = load_suite_module()

    suite.log_result = lambda **kwargs: None

    suite.logging.raiseExceptions = False

    if CPU_ONLY:
        try:
            import torch

            torch.set_num_threads(THREADS_PER_WORKER)

            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
        except Exception:
            pass

        original_base_config = suite.base_config

        def cpu_base_config() -> dict[str, Any]:
            config = original_base_config()

            config.setdefault(
                "trainer",
                {},
            )["mixed_precision"] = False

            return config

        suite.base_config = cpu_base_config

    _WORKER_MODULE = suite
    _WORKER_TASKS = build_tasks(suite)


def execute_task_in_worker(
    task_index: int,
) -> dict[str, Any]:

    suite = _WORKER_MODULE
    tasks = _WORKER_TASKS

    if suite is None or tasks is None:
        raise RuntimeError("The worker was not initialized.")

    task = tasks[task_index]
    paths = task_paths(
        suite,
        task,
    )

    try:
        if "runner" in task:
            passed, error = suite.run_baseline_task(task)
        else:
            passed, error = suite.run_training_task(task)

        return {
            "task_index": task_index,
            "passed": bool(passed),
            "error": str(error),
            "worker_pid": os.getpid(),
            "paths": {name: str(path) for name, path in paths.items()},
        }

    except BaseException as exc:
        return {
            "task_index": task_index,
            "passed": False,
            "error": (f"{type(exc).__name__}: {exc}\n" + traceback.format_exc()),
            "worker_pid": os.getpid(),
            "paths": {name: str(path) for name, path in paths.items()},
        }


def submit_available_tasks(
    executor: ProcessPoolExecutor,
    pending_futures: dict[
        Future,
        int,
    ],
    task_indices: list[int],
    next_position: int,
) -> int:

    while (
        next_position < len(task_indices) and len(pending_futures) < MAX_PENDING_TASKS
    ):
        task_index = task_indices[next_position]

        future = executor.submit(
            execute_task_in_worker,
            task_index,
        )

        pending_futures[future] = task_index

        next_position += 1

    return next_position


def main() -> None:
    configure_process_environment()

    script_directory = Path(__file__).resolve().parent

    if str(script_directory) not in sys.path:
        sys.path.insert(
            0,
            str(script_directory),
        )

    suite = load_suite_module()

    suite.prepare_output_directories()
    suite.cleanup_logging_handlers()

    tasks = build_tasks(suite)

    (
        pending_tasks,
        retained_rows,
    ) = prepare_checkpoint(
        suite,
        tasks,
    )

    tasks_by_key = {task_key(task): task for task in tasks}

    task_indices_by_key = {task_key(task): index for index, task in enumerate(tasks)}

    pending_indices = [task_indices_by_key[task_key(task)] for task in pending_tasks]

    unexpected_failures: list[str] = []
    expected_failures: list[str] = []
    unexpected_passes: list[str] = []

    restore_classification(
        suite,
        tasks=tasks,
        rows=retained_rows,
        unexpected_failures=unexpected_failures,
        expected_failures=expected_failures,
        unexpected_passes=unexpected_passes,
    )

    completed_rows: list[dict[str, Any]] = list(retained_rows)

    write_parallel_summary(
        suite,
        total_tasks=len(tasks),
        completed_rows=completed_rows,
        unexpected_failures=unexpected_failures,
        expected_failures=expected_failures,
        unexpected_passes=unexpected_passes,
    )

    if not pending_indices:
        print(
            "No pending cases remain.",
            flush=True,
        )
        return

    print(
        f"Starting {len(pending_indices)} pending case(s) "
        f"with {PARALLEL_WORKERS} worker processes.",
        flush=True,
    )
    print(
        f"Threads per worker: {THREADS_PER_WORKER}",
        flush=True,
    )
    print(
        f"CPU only: {CPU_ONLY}",
        flush=True,
    )

    context = mp.get_context("fork")

    pending_futures: dict[
        Future,
        int,
    ] = {}

    next_position = 0

    try:
        with ProcessPoolExecutor(
            max_workers=PARALLEL_WORKERS,
            mp_context=context,
            initializer=worker_initializer,
        ) as executor:
            next_position = submit_available_tasks(
                executor,
                pending_futures,
                pending_indices,
                next_position,
            )

            with tqdm(
                total=len(tasks),
                initial=len(retained_rows),
                desc="integration",
                unit="case",
                dynamic_ncols=True,
                leave=True,
                bar_format=(
                    "{l_bar}{bar}| "
                    "{n_fmt}/{total_fmt} "
                    "[{elapsed}<{remaining}, "
                    "{rate_fmt}] "
                    "{postfix}"
                ),
            ) as progress:
                while pending_futures:
                    completed_futures, _ = wait(
                        pending_futures,
                        return_when=FIRST_COMPLETED,
                    )

                    for future in completed_futures:
                        expected_index = pending_futures.pop(future)

                        try:
                            result = future.result()
                        except BaseException as exc:
                            result = {
                                "task_index": expected_index,
                                "passed": False,
                                "error": (
                                    f"{type(exc).__name__}: {exc}\n"
                                    + traceback.format_exc()
                                ),
                                "worker_pid": None,
                                "paths": {
                                    name: str(path)
                                    for name, path in task_paths(
                                        suite,
                                        tasks[expected_index],
                                    ).items()
                                },
                            }

                        task_index = int(result["task_index"])
                        task = tasks[task_index]
                        passed = bool(result["passed"])
                        error = str(
                            result.get(
                                "error",
                                "",
                            )
                        )

                        paths = {
                            name: Path(path) for name, path in result["paths"].items()
                        }

                        result_row = build_result_row(
                            suite,
                            task=task,
                            passed=passed,
                            error=error,
                            paths=paths,
                        )

                        append_result_row(
                            suite,
                            result_row,
                        )

                        completed_rows.append(result_row)

                        suite.classify_result(
                            task=task,
                            passed=passed,
                            error=error,
                            unexpected_failures=unexpected_failures,
                            expected_failures=expected_failures,
                            unexpected_passes=unexpected_passes,
                        )

                        worker_pid = result.get("worker_pid")

                        progress.update(1)

                        progress.set_postfix_str(
                            f"pid={worker_pid} "
                            f"running={len(pending_futures)} "
                            "unexpected_failures="
                            f"{len(unexpected_failures)} "
                            "expected_failures="
                            f"{len(expected_failures)} "
                            "unexpected_passes="
                            f"{len(unexpected_passes)}"
                        )

                        write_parallel_summary(
                            suite,
                            total_tasks=len(tasks),
                            completed_rows=completed_rows,
                            unexpected_failures=unexpected_failures,
                            expected_failures=expected_failures,
                            unexpected_passes=unexpected_passes,
                        )

                    next_position = submit_available_tasks(
                        executor,
                        pending_futures,
                        pending_indices,
                        next_position,
                    )

    except KeyboardInterrupt:
        print(
            "\nInterrupt received. Cancelling pending cases...",
            flush=True,
        )

        for future in pending_futures:
            future.cancel()

        write_parallel_summary(
            suite,
            total_tasks=len(tasks),
            completed_rows=completed_rows,
            unexpected_failures=unexpected_failures,
            expected_failures=expected_failures,
            unexpected_passes=unexpected_passes,
        )

        print(
            f"Checkpoint retained at: {suite.RESULTS_CSV}",
            flush=True,
        )
        print(
            "The next run will rerun the final completed case.",
            flush=True,
        )

        raise

    except BaseException:
        write_parallel_summary(
            suite,
            total_tasks=len(tasks),
            completed_rows=completed_rows,
            unexpected_failures=unexpected_failures,
            expected_failures=expected_failures,
            unexpected_passes=unexpected_passes,
        )

        print(
            f"\nCheckpoint retained at: {suite.RESULTS_CSV}",
            flush=True,
        )

        raise

    finally:
        suite.cleanup_logging_handlers()

    write_parallel_summary(
        suite,
        total_tasks=len(tasks),
        completed_rows=completed_rows,
        unexpected_failures=unexpected_failures,
        expected_failures=expected_failures,
        unexpected_passes=unexpected_passes,
    )

    if unexpected_failures or unexpected_passes:
        problems = []

        if unexpected_failures:
            problems.append(f"{len(unexpected_failures)} unexpected failure(s)")

        if unexpected_passes:
            problems.append(f"{len(unexpected_passes)} unexpected pass(es)")

        raise RuntimeError(
            "Parallel integration suite completed with "
            + " and ".join(problems)
            + f". See {suite.RESULTS_CSV} "
            f"and {suite.LOG_DIR}."
        )

    print("Parallel integration suite completed successfully.")
    print(f"Results: {suite.RESULTS_CSV}")
    print(f"Summary: {suite.SUMMARY_PATH}")
    print(f"Logs: {suite.LOG_DIR}")


if __name__ == "__main__":
    original_sigint_handler = signal.getsignal(signal.SIGINT)

    try:
        main()
    finally:
        signal.signal(
            signal.SIGINT,
            original_sigint_handler,
        )
