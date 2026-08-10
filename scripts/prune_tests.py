import json
import math
import os
from collections import defaultdict
from pathlib import Path

import pulp


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

OUTPUT_DIR = ROOT / "output" / "test_suite_analysis"

TEST_MAP_FILE = OUTPUT_DIR / "test_map.json"
BASELINE_COVERAGE_FILE = OUTPUT_DIR / "baseline_cov.json"

OUTPUT_SELECTED = OUTPUT_DIR / "pruned_tests.json"
OUTPUT_REMOVED = OUTPUT_DIR / "removed_tests.json"
OUTPUT_ZERO_BRANCH = OUTPUT_DIR / "zero_branch_tests.json"
OUTPUT_COVERAGE = OUTPUT_DIR / "final_module_coverage.json"
OUTPUT_TARGETS = OUTPUT_DIR / "module_coverage_targets.json"

TARGET_BRANCH_COVERAGE = 0.90

SOLVER_THREADS = 8
TIME_LIMIT_SECONDS = 600


def normalize_branch(branch):
    if not isinstance(branch, list):
        return None

    if len(branch) != 2:
        return None

    return branch[0], branch[1]


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist: {path}")

    with path.open(
        encoding="utf-8",
    ) as file:
        return json.load(file)


def compute_covered_branches(
    selected_tests,
    test_branches,
):
    covered = defaultdict(set)

    for test in selected_tests:
        for module, branches in test_branches.get(
            test,
            {},
        ).items():
            covered[module].update(branches)

    return covered


def compute_coverage(
    selected_tests,
    test_branches,
    module_totals,
):
    covered = compute_covered_branches(
        selected_tests,
        test_branches,
    )

    coverage = {}

    for module, total in module_totals.items():
        if total == 0:
            coverage[module] = 1.0
        else:
            coverage[module] = len(covered[module]) / total

    return coverage


print("Loading data...")

raw_map = load_json(TEST_MAP_FILE)
baseline = load_json(BASELINE_COVERAGE_FILE)


print("Extracting baseline module coverage...")

module_branch_totals = {}
module_baseline_covered = {}

for module, data in baseline["files"].items():
    if "site-packages" in module:
        continue

    if module.startswith("tests/"):
        continue

    summary = data.get(
        "summary",
        {},
    )

    total = summary.get(
        "num_branches",
        0,
    )
    covered = summary.get(
        "covered_branches",
        0,
    )

    if total > 0:
        module_branch_totals[module] = total
        module_baseline_covered[module] = covered


if not module_branch_totals:
    raise SystemExit(
        "No production modules with branches were found in baseline_cov.json."
    )


print("Processing per-test coverage map...")

test_branches = {}
branch_to_tests = defaultdict(set)
module_to_branches = defaultdict(set)

for test, modules in raw_map.items():
    normalized_modules = {}

    for module, branches in modules.items():
        if module not in module_branch_totals:
            continue

        cleaned = set()

        for branch in branches:
            normalized = normalize_branch(branch)

            if normalized is not None:
                cleaned.add(normalized)

        if not cleaned:
            continue

        normalized_modules[module] = cleaned

        module_to_branches[module].update(cleaned)

        for branch in cleaned:
            branch_to_tests[
                module,
                branch,
            ].add(test)

    if normalized_modules:
        test_branches[test] = normalized_modules


print("\nCoverage targets:\n")

required_branch_counts = {}
module_target_metadata = {}
incomplete_modules = {}

for module, baseline_total in sorted(module_branch_totals.items()):
    baseline_covered = module_baseline_covered[module]
    baseline_coverage = baseline_covered / baseline_total

    observed = len(
        module_to_branches.get(
            module,
            set(),
        )
    )

    if baseline_coverage >= TARGET_BRANCH_COVERAGE:
        required = math.ceil(baseline_total * TARGET_BRANCH_COVERAGE)
        target_policy = "target"
    else:
        required = baseline_covered
        target_policy = "preserve_baseline"

    required_branch_counts[module] = required

    effective_target = required / baseline_total

    module_target_metadata[module] = {
        "total_branches": baseline_total,
        "baseline_covered_branches": (baseline_covered),
        "baseline_coverage": round(
            baseline_coverage,
            6,
        ),
        "observed_branches_in_test_map": (observed),
        "required_branches": required,
        "effective_target": round(
            effective_target,
            6,
        ),
        "policy": target_policy,
    }

    print(module)
    print(f"  total={baseline_total}")
    print(f"  baseline covered={baseline_covered}")
    print(f"  baseline coverage={baseline_coverage:.4f}")
    print(f"  observed in test map={observed}")
    print(f"  required={required}")
    print(f"  effective target={effective_target:.4f}")
    print(f"  policy={target_policy}")

    if observed < required:
        incomplete_modules[module] = {
            "total": baseline_total,
            "baseline_covered": (baseline_covered),
            "baseline_coverage": (baseline_coverage),
            "observed": observed,
            "required": required,
        }

        print("  status=INCOMPLETE TEST MAP")
    else:
        print("  status=OK")

    print()


if incomplete_modules:
    print("The per-test coverage map cannot reproduce the required module coverage.")
    print("No optimization or pruning output will be generated.")
    print()

    for module, values in sorted(incomplete_modules.items()):
        print(
            f"  {module}: "
            f"observed={values['observed']}, "
            f"required={values['required']}, "
            f"baseline_covered="
            f"{values['baseline_covered']}, "
            f"total={values['total']}"
        )

    raise SystemExit("Rebuild test_map.json with complete per-test branch coverage.")


print("Deduplicating equivalent tests...")

signature_to_tests = defaultdict(list)

for test, modules in test_branches.items():
    signature = frozenset(
        (
            module,
            branch,
        )
        for module, branches in modules.items()
        for branch in branches
    )

    signature_to_tests[signature].append(test)


deduplicated_tests = []
equivalent_test_groups = {}

for signature, equivalent_tests in signature_to_tests.items():
    sorted_tests = sorted(equivalent_tests)
    representative = sorted_tests[0]

    deduplicated_tests.append(representative)

    equivalent_test_groups[representative] = sorted_tests


deduplicated_tests.sort()

print(f"Original branch tests: {len(test_branches)}")
print(f"Deduplicated tests:    {len(deduplicated_tests)}")
print(
    f"Equivalent tests removed before ILP: "
    f"{len(test_branches) - len(deduplicated_tests)}"
)


print("Rebuilding branch map after deduplication...")

deduplicated_branch_to_tests = defaultdict(set)

for test in deduplicated_tests:
    for module, branches in test_branches[test].items():
        for branch in branches:
            deduplicated_branch_to_tests[
                module,
                branch,
            ].add(test)


print("\nBuilding ILP model...")

problem = pulp.LpProblem(
    "Minimal_Test_Suite",
    pulp.LpMinimize,
)


x = {}

for index, test in enumerate(deduplicated_tests):
    x[test] = pulp.LpVariable(
        f"x_{index}",
        cat="Binary",
    )


y = {}

for index, branch_key in enumerate(sorted(deduplicated_branch_to_tests)):
    y[branch_key] = pulp.LpVariable(
        f"y_{index}",
        cat="Binary",
    )


print("Adding objective...")

problem += pulp.lpSum(x[test] for test in deduplicated_tests)


print("Adding branch constraints...")

for (
    module,
    branch,
), covering_tests in deduplicated_branch_to_tests.items():
    selected_covering_tests = pulp.lpSum(x[test] for test in covering_tests)

    problem += y[module, branch] <= selected_covering_tests

    for test in covering_tests:
        problem += y[module, branch] >= x[test]


print("Adding module coverage constraints...")

for module, required in required_branch_counts.items():
    observed_branches = module_to_branches.get(
        module,
        set(),
    )

    problem += pulp.lpSum(y[module, branch] for branch in observed_branches) >= required


print("\nSolving ILP...\n")

solver = pulp.PULP_CBC_CMD(
    msg=True,
    threads=SOLVER_THREADS,
    timeLimit=TIME_LIMIT_SECONDS,
)

solver_status_code = problem.solve(solver)
solver_status = pulp.LpStatus[solver_status_code]

print(
    "\nSolver status:",
    solver_status,
)


if solver_status != "Optimal":
    raise SystemExit(
        "The optimizer did not find a proven "
        "optimal solution. "
        f"Solver status: {solver_status}. "
        "No pruning files were written."
    )


selected_representatives = sorted(
    test
    for test in deduplicated_tests
    if (pulp.value(x[test]) is not None and pulp.value(x[test]) > 0.5)
)


selected_tests = selected_representatives


covered_branches = compute_covered_branches(
    selected_tests,
    test_branches,
)

final_coverage = compute_coverage(
    selected_tests,
    test_branches,
    module_branch_totals,
)


print("\n=== FINAL MODULE BRANCH COVERAGE ===")

failed_modules = {}
final_module_results = {}

for module, total in sorted(module_branch_totals.items()):
    covered_count = len(covered_branches[module])
    required_count = required_branch_counts[module]
    baseline_count = module_baseline_covered[module]
    coverage = covered_count / total

    if covered_count >= required_count:
        module_status = "OK"
    else:
        module_status = "FAIL"

        failed_modules[module] = {
            "covered": covered_count,
            "required": required_count,
            "baseline_covered": (baseline_count),
            "total": total,
            "coverage": coverage,
        }

    final_module_results[module] = {
        "covered_branches": covered_count,
        "total_branches": total,
        "coverage": round(
            coverage,
            6,
        ),
        "required_branches": (required_count),
        "baseline_covered_branches": (baseline_count),
        "status": module_status,
        "policy": (module_target_metadata[module]["policy"]),
    }

    print(
        f"{module}: "
        f"{covered_count}/{total} "
        f"({coverage:.4f}) "
        f"required={required_count} "
        f"baseline={baseline_count} "
        f"[{module_status}]"
    )


if failed_modules:
    print("\nThe optimized suite failed post-solver validation.")
    print("No pruning files will be written.")
    print()

    for module, values in sorted(failed_modules.items()):
        print(
            f"  {module}: "
            f"covered={values['covered']}, "
            f"required={values['required']}, "
            f"baseline_covered="
            f"{values['baseline_covered']}, "
            f"total={values['total']}, "
            f"coverage="
            f"{values['coverage']:.4f}"
        )

    raise SystemExit("Coverage validation failed.")


all_branch_tests = set(test_branches)
selected_test_set = set(selected_tests)

removed_tests = sorted(all_branch_tests - selected_test_set)

zero_branch_tests = sorted(set(raw_map) - all_branch_tests)


print("\n=== FINAL SUMMARY ===")

print(f"Original tests:       {len(raw_map)}")
print(f"Branch tests:         {len(test_branches)}")
print(f"Zero-branch tests:    {len(zero_branch_tests)}")
print(f"Selected tests:       {len(selected_tests)}")
print(f"Removed branch tests: {len(removed_tests)}")

total_removed = len(removed_tests) + len(zero_branch_tests)

print(f"Total removable:      {total_removed}")

if raw_map:
    reduction = 100 * total_removed / len(raw_map)
else:
    reduction = 0.0

print(f"Reduction:            {reduction:.2f}%")


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

with OUTPUT_SELECTED.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        selected_tests,
        file,
        indent=2,
    )

with OUTPUT_REMOVED.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        removed_tests,
        file,
        indent=2,
    )

with OUTPUT_ZERO_BRANCH.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        zero_branch_tests,
        file,
        indent=2,
    )

with OUTPUT_COVERAGE.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        final_module_results,
        file,
        indent=2,
    )

with OUTPUT_TARGETS.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        module_target_metadata,
        file,
        indent=2,
    )


print("\nSaved:")
print(f"  {OUTPUT_SELECTED}")
print(f"  {OUTPUT_REMOVED}")
print(f"  {OUTPUT_ZERO_BRANCH}")
print(f"  {OUTPUT_COVERAGE}")
print(f"  {OUTPUT_TARGETS}")
