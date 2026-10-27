import json
import os
import subprocess
from multiprocessing import Pool
from pathlib import Path

# Find project root (one level up from this script's directory)
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

print("WORKING DIRECTORY:", os.getcwd())

# Create output directory for all our results
OUTPUT_DIR = Path("../output/test_suite_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

# Final output file - maps tests to the code branches they execute
OUTPUT = OUTPUT_DIR / "test_map.json"
# Checkpoint file - periodically saved as we process tests (resume if interrupted)
CHECKPOINT = OUTPUT_DIR / "test_map_checkpoint.json"


# If we crashed before, load the checkpoint so we don't re-run completed tests
if CHECKPOINT.exists():
    with open(CHECKPOINT) as f:
        test_map = json.load(f)

    completed_tests = set(test_map.keys())

    print("Loaded checkpoint")
    print(f"Completed tests: {len(completed_tests)}")

else:
    # Starting fresh - empty map
    test_map = {}
    completed_tests = set()


# Ask pytest to list all tests without actually running them
result = subprocess.run(
    ["pytest", "--collect-only", "-q"],
    capture_output=True,
    text=True,
)

# Extract test paths from pytest output (lines starting with "tests/")
tests = [
    line.strip()
    for line in result.stdout.splitlines()
    if line.strip().startswith("tests/")
]

print(f"Found {len(tests)} total tests")

# Filter out tests we've already processed
remaining = [t for t in tests if t not in completed_tests]

print(f"Remaining tests: {len(remaining)}")


def run_test(test):
    """
    Run a single test with code coverage tracking.
    Returns coverage data showing which code branches the test executed.

    This function runs in a worker process, so each test is isolated.
    """
    print(f"[PID {os.getpid()}] Running: {test}", flush=True)

    # Temporary file to store coverage data for this test
    TMP_COV = OUTPUT_DIR / f"tmp_cov_{os.getpid()}.json"

    # Set up environment variables for this worker process
    env = os.environ.copy()
    # Each worker gets its own coverage file (prevents conflicts)
    env["COVERAGE_FILE"] = str(OUTPUT_DIR / f".coverage.{os.getpid()}")
    # Each worker gets its own temp directory for pytest
    env["PYTEST_ADDOPTS"] = f"--basetemp={OUTPUT_DIR / f'tmp_{os.getpid()}'}"

    # Clear any previous coverage data from this worker
    subprocess.run(["coverage", "erase"], capture_output=True, env=env)

    # Run the test and collect code coverage data
    # --branch: track not just lines, but individual code branches (if/else paths)
    # -p no:cacheprovider: disable pytest's cache plugin
    run_result = subprocess.run(
        [
            "coverage",
            "run",
            "--branch",  # Track branches (more detailed than line coverage)
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",  # Disable cache to avoid conflicts
            test,
        ],
        capture_output=True,
        text=True,
        env=env,
    )

    # If test failed, return error info instead of trying to get coverage
    if run_result.returncode != 0:
        return {
            "test": test,
            "status": "failed",
            "stdout": run_result.stdout[-1000:],  # Last 1000 chars (avoid huge output)
            "stderr": run_result.stderr[-1000:],
        }

    # Convert coverage data to JSON format
    json_result = subprocess.run(
        ["coverage", "json", "-o", str(TMP_COV)],
        capture_output=True,
        text=True,
        env=env,
    )

    if json_result.returncode != 0:
        return {"test": test, "status": "cov_failed"}

    # Load the coverage JSON file
    try:
        with open(TMP_COV) as f:
            cov = json.load(f)
    except Exception:
        return {"test": test, "status": "cov_parse_failed"}

    # Extract branch coverage data, filtering out third-party libraries
    test_data = {}

    for module, data in cov.get("files", {}).items():
        # Skip coverage of installed packages (we only care about our code)
        if "site-packages" in module:
            continue

        # Get the branches that were executed in this module
        branches = data.get("executed_branches", [])

        # Only store modules that actually have branch data
        if branches:
            test_data[module] = branches

    return {"test": test, "status": "ok", "data": test_data}


# Use 16 parallel workers to run tests faster
# (each test runs in its own process with isolated coverage data)
NUM_WORKERS = 16

print(f"Using {NUM_WORKERS} workers")
completed = 0
total = len(remaining)

# Pool processes tests in parallel
# imap_unordered: returns results as they complete (not in order, but faster feedback)
with Pool(NUM_WORKERS) as pool:
    for result in pool.imap_unordered(run_test, remaining):
        completed += 1

        test = result["test"]

        print(f"\n[{completed}/{total}] Finished: {test}", flush=True)

        # Check if test execution failed
        if result["status"] != "ok":
            print(f" Failed: {result['status']}", flush=True)

            # Print last part of output so we can see what went wrong
            print(result["stdout"])
            print(result["stderr"])

            # Skip to next test (don't save coverage data for failed tests)
            continue

        # Test passed - extract coverage data
        test_map[test] = result["data"]

        # Print summary of what this test covered
        total_branches = sum(len(v) for v in result["data"].values())

        print(f"Modules: {len(result['data'])}", flush=True)
        print(f"Branches: {total_branches}", flush=True)

        # Save progress to checkpoint (so we can resume if script crashes)
        with open(CHECKPOINT, "w") as f:
            json.dump(test_map, f, indent=2)

        # Also save to final output file (constantly updated)
        with open(OUTPUT, "w") as f:
            json.dump(test_map, f, indent=2)


# Write final output (in case last batch of results hasn't been written yet)
with open(OUTPUT, "w") as f:
    json.dump(test_map, f, indent=2)

print("\n=================================================")
print("DONE")
print("=================================================")

print(f"Saved final map to: {OUTPUT}")
print(f"Total tests stored: {len(test_map)}")

# Count how many unique modules have branch coverage data
total_modules = set()

for data in test_map.values():
    total_modules.update(data.keys())

print(f"Modules with branch data: {len(total_modules)}")
