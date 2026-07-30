import json
import os
from collections import defaultdict
from pathlib import Path

import pulp


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

OUTPUT_DIR = ROOT / "test_suite_analysis"

TEST_MAP_FILE = OUTPUT_DIR / "test_map.json"
BASELINE_COVERAGE_FILE = OUTPUT_DIR / "baseline_cov.json"

OUTPUT_SELECTED = OUTPUT_DIR / "ilp_selected_tests.json"
OUTPUT_COVERAGE = OUTPUT_DIR / "ilp_final_coverage.json"

TARGET_BRANCH_COVERAGE = 0.90


MIN_RELAXATION = 0


SOLVER_THREADS = 8
TIME_LIMIT_SECONDS = 600


def normalize_branch(branch):
    if not isinstance(branch, list):
        return None

    if len(branch) != 2:
        return None

    return (branch[0], branch[1])


def compute_coverage(selected_tests, test_branches, module_totals):
    covered = defaultdict(set)

    for test in selected_tests:
        for module, branches in test_branches[test].items():
            covered[module].update(branches)

    coverage = {}

    for module, total in module_totals.items():
        if total == 0:
            coverage[module] = 1.0
            continue

        coverage[module] = len(covered[module]) / total

    return coverage


print("Loading data...")

with open(TEST_MAP_FILE) as f:
    raw_map = json.load(f)

with open(BASELINE_COVERAGE_FILE) as f:
    baseline = json.load(f)


print("Extracting module branch totals...")

module_branch_totals = {}

for module, data in baseline["files"].items():
    if "site-packages" in module:
        continue

    if module.startswith("tests/"):
        continue

    total = data["summary"]["num_branches"]

    if total > 0:
        module_branch_totals[module] = total


print("Processing test coverage map...")


test_branches = {}


branch_to_tests = defaultdict(set)


module_to_branches = defaultdict(set)

for test, modules in raw_map.items():
    normalized_modules = {}

    for module, branches in modules.items():
        if "site-packages" in module:
            continue

        if module.startswith("tests/"):
            continue

        cleaned = set()

        for branch in branches:
            normalized = normalize_branch(branch)

            if normalized is not None:
                cleaned.add(normalized)

        if cleaned:
            normalized_modules[module] = cleaned

            for branch in cleaned:
                branch_to_tests[(module, branch)].add(test)
                module_to_branches[module].add(branch)

    if normalized_modules:
        test_branches[test] = normalized_modules


print("\nCoverage targets:\n")

required_branch_counts = {}
for module, observed_branches in sorted(module_to_branches.items()):
    observed = len(observed_branches)

    baseline_total = module_branch_totals[module]

    import math

    required = math.ceil(baseline_total * TARGET_BRANCH_COVERAGE)

    required = min(required, observed)

    required_branch_counts[module] = required

    print(module)
    print(f"  baseline={baseline_total}")
    print(f"  observed={observed}")
    print(f"  required={required}")
    print()

    required = int(baseline_total * TARGET_BRANCH_COVERAGE)


print("Deduplicating equivalent tests...")

signature_to_tests = defaultdict(list)

for test, modules in test_branches.items():
    signature = frozenset(
        (module, branch) for module, branches in modules.items() for branch in branches
    )

    signature_to_tests[signature].append(test)

deduplicated_tests = []

for group in signature_to_tests.values():
    deduplicated_tests.append(group[0])

print(f"Original tests:     {len(test_branches)}")
print(f"Deduplicated tests: {len(deduplicated_tests)}")


print("\nBuilding ILP model...")

problem = pulp.LpProblem(
    "Minimal_Test_Suite",
    pulp.LpMinimize,
)


x = {}

for i, test in enumerate(deduplicated_tests):
    x[test] = pulp.LpVariable(
        f"x_{i}",
        cat="Binary",
    )


y = {}

for i, key in enumerate(branch_to_tests.keys()):
    y[key] = pulp.LpVariable(
        f"y_{i}",
        cat="Binary",
    )


print("Adding objective...")

problem += pulp.lpSum(x[test] for test in deduplicated_tests)


print("Adding branch constraints...")

for (module, branch), tests_covering_branch in branch_to_tests.items():
    valid_tests = [test for test in tests_covering_branch if test in x]

    if not valid_tests:
        continue

    problem += y[(module, branch)] <= pulp.lpSum(x[test] for test in valid_tests)


print("Adding module coverage constraints...")

for module, branches in module_to_branches.items():
    required = required_branch_counts[module]

    problem += pulp.lpSum(y[(module, branch)] for branch in branches) >= required


print("\nSolving ILP...\n")

solver = pulp.PULP_CBC_CMD(
    msg=True,
    threads=SOLVER_THREADS,
    timeLimit=TIME_LIMIT_SECONDS,
)

status = problem.solve(solver)

print("\nSolver status:", pulp.LpStatus[status])


selected_tests = []

for test in deduplicated_tests:
    value = pulp.value(x[test])

    if value is not None and value > 0.5:
        selected_tests.append(test)


final_coverage = compute_coverage(
    selected_tests,
    test_branches,
    module_branch_totals,
)

removed_tests = sorted(set(test_branches.keys()) - set(selected_tests))

zero_branch_tests = sorted([test for test in raw_map if test not in test_branches])


print("\n=== FINAL SUMMARY ===")

print(f"Original tests:  {len(raw_map)}")
print(f"Branch tests:    {len(test_branches)}")
print(f"Zero branch:     {len(zero_branch_tests)}")
print(f"Remaining tests: {len(selected_tests)}")
print(f"Removed tests:   {len(removed_tests)}")

if raw_map:
    reduction = 100 * (len(raw_map) - len(selected_tests)) / len(raw_map)
else:
    reduction = 0.0

print(f"Reduction:       {reduction:.2f}%")


print("\n=== FINAL MODULE BRANCH COVERAGE ===")

for module, cov in sorted(final_coverage.items()):
    status = "OK"

    if cov < TARGET_BRANCH_COVERAGE:
        status = "FAIL"

    print(f"{module}: {cov:.4f} [{status}]")


with open(OUTPUT_DIR / "pruned_tests.json", "w") as f:
    json.dump(
        sorted(selected_tests),
        f,
        indent=2,
    )

with open(OUTPUT_DIR / "removed_tests.json", "w") as f:
    json.dump(
        sorted(removed_tests),
        f,
        indent=2,
    )

with open(OUTPUT_DIR / "zero_branch_tests.json", "w") as f:
    json.dump(
        sorted(zero_branch_tests),
        f,
        indent=2,
    )

with open(OUTPUT_DIR / "final_module_coverage.json", "w") as f:
    json.dump(
        {module: round(cov, 4) for module, cov in sorted(final_coverage.items())},
        f,
        indent=2,
    )

print("\nSaved:")
print("  test_suite_analysis/pruned_tests.json")
print("  test_suite_analysis/removed_tests.json")
print("  test_suite_analysis/zero_branch_tests.json")
print("  test_suite_analysis/final_module_coverage.json")
