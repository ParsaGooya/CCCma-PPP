import json
import subprocess
from multiprocessing import Pool
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

print("WORKING DIRECTORY:", os.getcwd())

OUTPUT_DIR = Path("output/test_suite_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT = OUTPUT_DIR / "test_map.json"
CHECKPOINT = OUTPUT_DIR / "test_map_checkpoint.json"


if CHECKPOINT.exists():
    with open(CHECKPOINT) as f:
        test_map = json.load(f)

    completed_tests = set(test_map.keys())

    print("Loaded checkpoint")
    print(f"Completed tests: {len(completed_tests)}")

else:
    test_map = {}
    completed_tests = set()


result = subprocess.run(
    ["pytest", "--collect-only", "-q"],
    capture_output=True,
    text=True,
)

tests = [
    line.strip()
    for line in result.stdout.splitlines()
    if line.strip().startswith("tests/")
]

print(f"Found {len(tests)} total tests")

remaining = [t for t in tests if t not in completed_tests]

print(f"Remaining tests: {len(remaining)}")


def run_test(test):
    print(f"[PID {os.getpid()}] Running: {test}", flush=True)
    TMP_COV = OUTPUT_DIR / f"tmp_cov_{os.getpid()}.json"

    env = os.environ.copy()
    env["COVERAGE_FILE"] = str(OUTPUT_DIR / f".coverage.{os.getpid()}")
    env["PYTEST_ADDOPTS"] = f"--basetemp={OUTPUT_DIR / f'tmp_{os.getpid()}'}"

    subprocess.run(["coverage", "erase"], capture_output=True, env=env)

    run_result = subprocess.run(
        [
            "coverage",
            "run",
            "--branch",
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            test,
        ],
        capture_output=True,
        text=True,
        env=env,
    )

    if run_result.returncode != 0:
        return {
            "test": test,
            "status": "failed",
            "stdout": run_result.stdout[-1000:],
            "stderr": run_result.stderr[-1000:],
        }

    json_result = subprocess.run(
        ["coverage", "json", "-o", str(TMP_COV)],
        capture_output=True,
        text=True,
        env=env,
    )

    if json_result.returncode != 0:
        return {"test": test, "status": "cov_failed"}

    try:
        with open(TMP_COV) as f:
            cov = json.load(f)
    except Exception:
        return {"test": test, "status": "cov_parse_failed"}

    test_data = {}

    for module, data in cov.get("files", {}).items():
        if "site-packages" in module:
            continue

        branches = data.get("executed_branches", [])

        if branches:
            test_data[module] = branches

    return {"test": test, "status": "ok", "data": test_data}


NUM_WORKERS = 16

print(f"Using {NUM_WORKERS} workers")
completed = 0
total = len(remaining)
with Pool(NUM_WORKERS) as pool:
    for result in pool.imap_unordered(run_test, remaining):
        completed += 1

        test = result["test"]

        print(f"\n[{completed}/{total}] Finished: {test}", flush=True)

        if result["status"] != "ok":
            print(f"❌ Failed: {result['status']}", flush=True)

            print(result["stdout"])
            print(result["stderr"])

            continue

        test_map[test] = result["data"]

        total_branches = sum(len(v) for v in result["data"].values())

        print(f"✅ Modules: {len(result['data'])}", flush=True)
        print(f"✅ Branches: {total_branches}", flush=True)

        with open(CHECKPOINT, "w") as f:
            json.dump(test_map, f, indent=2)

        with open(OUTPUT, "w") as f:
            json.dump(test_map, f, indent=2)


with open(OUTPUT, "w") as f:
    json.dump(test_map, f, indent=2)

print("\n=================================================")
print("DONE")
print("=================================================")

print(f"Saved final map to: {OUTPUT}")
print(f"Total tests stored: {len(test_map)}")

total_modules = set()

for data in test_map.values():
    total_modules.update(data.keys())

print(f"Modules with branch data: {len(total_modules)}")
