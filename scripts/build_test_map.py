import json
import os
import subprocess
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
# Checkpoint file - saved after each test (resume if interrupted)
CHECKPOINT = OUTPUT_DIR / "test_map_checkpoint.json"
# Temporary file - coverage data gets written here by the coverage tool
TMP_COV = OUTPUT_DIR / "tmp_cov.json"

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

# Run tests ONE BY ONE (sequential, not parallel like the other script)
# This is slower but easier to debug since output isn't mixed between processes
for i, test in enumerate(remaining, 1):
    print("\n=================================================")
    print(f"[{i}/{len(remaining)}]")
    print(test)
    print("=================================================")

    # Clear any previous coverage data from the last test
    subprocess.run(
        ["coverage", "erase"],
        capture_output=True,
        text=True,
    )

    # Run the test and collect code coverage data
    # --branch: track not just lines, but individual code branches (if/else paths)
    run_result = subprocess.run(
        ["coverage", "run", "--branch", "-m", "pytest", test],
        capture_output=True,
        text=True,
    )

    print("RETURN CODE:", run_result.returncode)

    # Check if test execution failed
    if run_result.returncode != 0:
        print("\nTEST FAILED")

        # Print test output so we can debug why it failed
        print("\nSTDOUT:")
        print(run_result.stdout[-3000:])  # Last 3000 chars (avoid huge output)

        print("\nSTDERR:")
        print(run_result.stderr[-3000:])

        # Skip to next test (don't save coverage data for failed tests)
        continue

    # Convert coverage data to JSON format
    # This gives us structured data we can parse
    json_result = subprocess.run(
        ["coverage", "json", "-o", str(TMP_COV)],
        capture_output=True,
        text=True,
    )

    if json_result.returncode != 0:
        print("\nFAILED TO GENERATE COVERAGE JSON")

        print(json_result.stdout)
        print(json_result.stderr)

        # Skip to next test if we can't get coverage data
        continue

    # Load the coverage JSON file
    try:
        with open(TMP_COV) as f:
            cov = json.load(f)

    except Exception as e:
        print("\nFAILED TO LOAD COVERAGE JSON")
        print(e)

        # Skip to next test if we can't parse the JSON
        continue

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

    # Store the coverage data for this test
    test_map[test] = test_data

    # Print summary of what this test covered
    print(f"Modules with branch coverage: {len(test_data)}")

    total_branches = sum(len(v) for v in test_data.values())

    print(f"Total executed branches: {total_branches}")

    # Save progress to checkpoint (so we can resume if script crashes)
    with open(CHECKPOINT, "w") as f:
        json.dump(test_map, f, indent=2)

    # Also save to final output file
    with open(OUTPUT, "w") as f:
        json.dump(test_map, f, indent=2)

    print("Checkpoint saved")

# Write final output (make sure everything is written)
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
