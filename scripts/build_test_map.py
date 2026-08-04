import json
import subprocess
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

print("WORKING DIRECTORY:", os.getcwd())

OUTPUT_DIR = Path("output/test_suite_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT = OUTPUT_DIR / "test_map.json"
CHECKPOINT = OUTPUT_DIR / "test_map_checkpoint.json"
TMP_COV = OUTPUT_DIR / "tmp_cov.json"

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

for i, test in enumerate(remaining, 1):
    print("\n=================================================")
    print(f"[{i}/{len(remaining)}]")
    print(test)
    print("=================================================")

    subprocess.run(
        ["coverage", "erase"],
        capture_output=True,
        text=True,
    )

    run_result = subprocess.run(
        ["coverage", "run", "--branch", "-m", "pytest", test],
        capture_output=True,
        text=True,
    )

    print("RETURN CODE:", run_result.returncode)

    if run_result.returncode != 0:
        print("\nTEST FAILED")

        print("\nSTDOUT:")
        print(run_result.stdout[-3000:])

        print("\nSTDERR:")
        print(run_result.stderr[-3000:])

        continue

    json_result = subprocess.run(
        ["coverage", "json", "-o", str(TMP_COV)],
        capture_output=True,
        text=True,
    )

    if json_result.returncode != 0:
        print("\nFAILED TO GENERATE COVERAGE JSON")

        print(json_result.stdout)
        print(json_result.stderr)

        continue

    try:
        with open(TMP_COV) as f:
            cov = json.load(f)

    except Exception as e:
        print("\nFAILED TO LOAD COVERAGE JSON")
        print(e)

        continue

    test_data = {}

    for module, data in cov.get("files", {}).items():
        if "site-packages" in module:
            continue

        branches = data.get("executed_branches", [])

        if branches:
            test_data[module] = branches

    test_map[test] = test_data

    print(f"Modules with branch coverage: {len(test_data)}")

    total_branches = sum(len(v) for v in test_data.values())

    print(f"Total executed branches: {total_branches}")

    with open(CHECKPOINT, "w") as f:
        json.dump(test_map, f, indent=2)

    with open(OUTPUT, "w") as f:
        json.dump(test_map, f, indent=2)

    print("Checkpoint saved")

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
