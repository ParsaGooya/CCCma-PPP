import json
import math
from collections import defaultdict
from pathlib import Path

import pulp

# Output directory where results will be saved
OUTPUT_DIR = Path("../output/test_suite_analysis")

# Input files
TEST_MAP_FILE = OUTPUT_DIR / "test_map.json"  # Maps tests to branches they cover
BASELINE_COVERAGE_FILE = (
    OUTPUT_DIR / "baseline_cov.json"
)  # Full coverage from running all tests

# Output files
OUTPUT_SELECTED = OUTPUT_DIR / "pruned_tests.json"  # Tests to keep
OUTPUT_REMOVED = OUTPUT_DIR / "removed_tests.json"  # Tests we can remove
OUTPUT_ZERO_BRANCH = (
    OUTPUT_DIR / "zero_branch_tests.json"
)  # Tests that don't test any branches
OUTPUT_COVERAGE = (
    OUTPUT_DIR / "final_module_coverage.json"
)  # Coverage achieved by selected tests
OUTPUT_TARGETS = (
    OUTPUT_DIR / "module_coverage_targets.json"
)  # Target coverage per module

# We want each module to have at least 90% branch coverage (or preserve baseline if it's lower)
TARGET_BRANCH_COVERAGE = 0.90

# ILP solver settings
SOLVER_THREADS = 8  # Use 8 CPU cores for optimization
TIME_LIMIT_SECONDS = 600  # Give solver up to 10 minutes to find optimal solution


def normalize_branch(branch):
    """
    Validate and normalize a branch representation.
    Branches should be [line_number, branch_id] tuples.
    Returns (line, branch_id) or None if invalid.
    """
    if not isinstance(branch, list):
        return None

    if len(branch) != 2:
        return None

    return branch[0], branch[1]


def load_json(path):
    """Load and parse a JSON file. Raise error if file doesn't exist."""
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
    """
    Given a set of selected tests, compute which branches they cover.
    Returns dict mapping module -> set of (line, branch_id) tuples covered by the selected tests.
    """
    covered = defaultdict(set)

    for test in selected_tests:
        # For each test, add all branches it covers to the covered set
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
    """
    Compute branch coverage percentage for each module.
    Returns dict mapping module -> coverage_fraction (0.0 to 1.0).
    """
    covered = compute_covered_branches(
        selected_tests,
        test_branches,
    )

    coverage = {}

    for module, total in module_totals.items():
        if total == 0:
            # No branches to cover in this module - 100% coverage by default
            coverage[module] = 1.0
        else:
            # Coverage = (branches covered) / (total branches)
            coverage[module] = len(covered[module]) / total

    return coverage


print("Loading data...")

# Load the test-to-branches mapping from previous analysis
raw_map = load_json(TEST_MAP_FILE)
# Load baseline coverage from running ALL tests
baseline = load_json(BASELINE_COVERAGE_FILE)


print("Extracting baseline module coverage...")

# Extract the total branches and baseline coverage for each production module
module_branch_totals = {}  # module -> total number of branches
module_baseline_covered = {}  # module -> number of branches covered in baseline

for module, data in baseline["files"].items():
    # Skip third-party packages
    if "site-packages" in module:
        continue

    # Skip test files (we're only interested in production code)
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

    # Only track modules that actually have branches
    if total > 0:
        module_branch_totals[module] = total
        module_baseline_covered[module] = covered


if not module_branch_totals:
    raise SystemExit(
        "No production modules with branches were found in baseline_cov.json."
    )


print("Processing per-test coverage map...")

# Build three data structures:
# 1. test_branches: test -> module -> set of branches
# 2. branch_to_tests: (module, branch) -> set of tests covering it
# 3. module_to_branches: module -> set of all branches observed in any test

test_branches = {}
branch_to_tests = defaultdict(set)
module_to_branches = defaultdict(set)

for test, modules in raw_map.items():
    normalized_modules = {}

    for module, branches in modules.items():
        # Skip modules not in baseline (probably third-party or tests)
        if module not in module_branch_totals:
            continue

        cleaned = set()

        # Validate and normalize each branch
        for branch in branches:
            normalized = normalize_branch(branch)

            if normalized is not None:
                cleaned.add(normalized)

        # Skip tests that don't cover any branches in production modules
        if not cleaned:
            continue

        normalized_modules[module] = cleaned

        # Track all branches seen in test coverage map
        module_to_branches[module].update(cleaned)

        # Track which tests cover each branch
        for branch in cleaned:
            branch_to_tests[
                module,
                branch,
            ].add(test)

    # Only keep tests that cover at least one production module branch
    if normalized_modules:
        test_branches[test] = normalized_modules


print("\nCoverage targets:\n")

# For each module, determine the target coverage and required number of branches
required_branch_counts = {}  # module -> number of branches to cover
module_target_metadata = {}  # detailed info about each module's targets
incomplete_modules = {}  # modules where we don't have full coverage info

for module, baseline_total in sorted(module_branch_totals.items()):
    baseline_covered = module_baseline_covered[module]
    baseline_coverage = baseline_covered / baseline_total

    # How many branches do we observe in the test map for this module?
    observed = len(
        module_to_branches.get(
            module,
            set(),
        )
    )

    # Policy decision: if baseline already meets target, aim for TARGET_BRANCH_COVERAGE
    # Otherwise, just preserve the baseline coverage
    if baseline_coverage >= TARGET_BRANCH_COVERAGE:
        # Baseline is good - shoot for the target
        required = math.ceil(baseline_total * TARGET_BRANCH_COVERAGE)
        target_policy = "target"
    else:
        # Baseline is low - just preserve what we have
        required = baseline_covered
        target_policy = "preserve_baseline"

    required_branch_counts[module] = required

    effective_target = required / baseline_total

    # Store metadata about this module's coverage target
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

    # Print details for this module
    print(module)
    print(f"  total={baseline_total}")
    print(f"  baseline covered={baseline_covered}")
    print(f"  baseline coverage={baseline_coverage:.4f}")
    print(f"  observed in test map={observed}")
    print(f"  required={required}")
    print(f"  effective target={effective_target:.4f}")
    print(f"  policy={target_policy}")

    # Check if the test map is incomplete (doesn't cover all required branches)
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


# If test map is incomplete, we can't optimize (can't achieve the targets)
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

# Two tests are equivalent if they cover exactly the same branches
# We only need to keep one representative from each group

signature_to_tests = defaultdict(list)

# Create a "signature" for each test (the set of branches it covers)
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

# For each group of equivalent tests, pick the first one alphabetically as representative
for signature, equivalent_tests in signature_to_tests.items():
    sorted_tests = sorted(equivalent_tests)
    representative = sorted_tests[0]

    deduplicated_tests.append(representative)

    # Remember which tests are equivalent to this representative
    equivalent_test_groups[representative] = sorted_tests


deduplicated_tests.sort()

print(f"Original branch tests: {len(test_branches)}")
print(f"Deduplicated tests:    {len(deduplicated_tests)}")
print(
    f"Equivalent tests removed before ILP: "
    f"{len(test_branches) - len(deduplicated_tests)}"
)


print("Rebuilding branch map after deduplication...")

# Rebuild the branch-to-tests map using only deduplicated tests
deduplicated_branch_to_tests = defaultdict(set)

for test in deduplicated_tests:
    for module, branches in test_branches[test].items():
        for branch in branches:
            deduplicated_branch_to_tests[
                module,
                branch,
            ].add(test)


print("\nBuilding ILP model...")

# Integer Linear Programming (ILP) problem to find the minimal set of tests
# that still achieves the coverage targets.
# Think of it as: "Pick the smallest subset of tests that still covers all required branches"

problem = pulp.LpProblem(
    "Minimal_Test_Suite",
    pulp.LpMinimize,  # Minimize: we want the fewest tests
)


# Binary variables: x_test = 1 if we select this test, 0 if we remove it
x = {}

for index, test in enumerate(deduplicated_tests):
    x[test] = pulp.LpVariable(
        f"x_{index}",
        cat="Binary",
    )


# Binary variables: y_branch = 1 if this branch is covered by selected tests
y = {}

for index, branch_key in enumerate(sorted(deduplicated_branch_to_tests)):
    y[branch_key] = pulp.LpVariable(
        f"y_{index}",
        cat="Binary",
    )


print("Adding objective...")

# Objective: minimize the total number of selected tests
problem += pulp.lpSum(x[test] for test in deduplicated_tests)


print("Adding branch constraints...")

# For each branch, y_branch should be 1 if ANY covering test is selected
# This is expressed as: if any test that covers this branch is selected, mark the branch as covered
for (
    module,
    branch,
), covering_tests in deduplicated_branch_to_tests.items():
    # y_branch <= (sum of tests covering this branch)
    # This means: if y_branch = 1, at least one covering test must be selected
    selected_covering_tests = pulp.lpSum(x[test] for test in covering_tests)

    problem += y[module, branch] <= selected_covering_tests

    # Also: y_branch >= x_test for each covering test
    # This means: if we select a test, mark all branches it covers
    for test in covering_tests:
        problem += y[module, branch] >= x[test]


print("Adding module coverage constraints...")

# For each module, we must cover at least the required number of branches
for module, required in required_branch_counts.items():
    observed_branches = module_to_branches.get(
        module,
        set(),
    )

    # Sum of covered branches for this module >= required
    problem += pulp.lpSum(y[module, branch] for branch in observed_branches) >= required


print("\nSolving ILP...\n")

# Solve the optimization problem using the CBC solver (parallel with multiple threads)
solver = pulp.PULP_CBC_CMD(
    msg=True,  # Print solver progress
    threads=SOLVER_THREADS,
    timeLimit=TIME_LIMIT_SECONDS,
)

solver_status_code = problem.solve(solver)
solver_status = pulp.LpStatus[solver_status_code]

print(
    "\nSolver status:",
    solver_status,
)


# If solver didn't find optimal solution, bail out (results wouldn't be reliable)
if solver_status != "Optimal":
    raise SystemExit(
        "The optimizer did not find a proven "
        "optimal solution. "
        f"Solver status: {solver_status}. "
        "No pruning files were written."
    )


# Extract the selected tests from the optimal solution
# A test is selected if x_test value > 0.5 (since it's binary, it's either 0 or 1)
selected_representatives = sorted(
    test
    for test in deduplicated_tests
    if (pulp.value(x[test]) is not None and pulp.value(x[test]) > 0.5)
)


selected_tests = selected_representatives


# Now verify the selected tests actually meet all coverage requirements
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

failed_modules = {}  # Modules that didn't meet the coverage requirement
final_module_results = {}  # Results for all modules

for module, total in sorted(module_branch_totals.items()):
    covered_count = len(covered_branches[module])
    required_count = required_branch_counts[module]
    baseline_count = module_baseline_covered[module]
    coverage = covered_count / total

    # Check if this module meets the requirement
    if covered_count >= required_count:
        module_status = "OK"
    else:
        module_status = "FAIL"

        # Record what went wrong
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


# If any module failed, don't write output (results are invalid)
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


# Calculate which tests were removed
all_branch_tests = set(test_branches)
selected_test_set = set(selected_tests)

removed_tests = sorted(all_branch_tests - selected_test_set)

# Tests that don't cover any production code branches (not included in optimization)
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


# Create output directory and write results
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# Save the selected tests (the minimal set we need to keep)
with OUTPUT_SELECTED.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        selected_tests,
        file,
        indent=2,
    )

# Save the removed tests (tests we can safely delete)
with OUTPUT_REMOVED.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        removed_tests,
        file,
        indent=2,
    )

# Save tests that don't test any production branches
with OUTPUT_ZERO_BRANCH.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        zero_branch_tests,
        file,
        indent=2,
    )

# Save the coverage achieved by the selected tests
with OUTPUT_COVERAGE.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        final_module_results,
        file,
        indent=2,
    )

# Save the coverage targets we aimed for
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
