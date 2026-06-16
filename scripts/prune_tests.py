import json
from collections import defaultdict


TARGET_BRANCH_COVERAGE = 0.90

TEST_MAP_FILE = "test_map.json"

with open(TEST_MAP_FILE) as f:
    raw_map = json.load(f)

test_branches = {}

for test, modules in raw_map.items():
    normalized_modules = {}

    for module, branches in modules.items():
        if "site-packages" in module:
            continue

        if "cccma_ppp" not in module:
            continue

        cleaned = set()

        for branch in branches:
            if not isinstance(branch, list):
                continue

            if len(branch) != 2:
                continue

            src, dst = branch

            cleaned.add((src, dst))

        if cleaned:
            normalized_modules[module] = cleaned

    if normalized_modules:
        test_branches[test] = normalized_modules

all_module_branches = defaultdict(set)

for modules in test_branches.values():
    for module, branches in modules.items():
        all_module_branches[module] |= branches

print("\n=== TOTAL BRANCHES PER MODULE ===")

for module, branches in sorted(all_module_branches.items()):
    print(f"{module}: {len(branches)}")

branch_count = defaultdict(int)

for modules in test_branches.values():
    for module, branches in modules.items():
        for branch in branches:
            branch_count[(module, branch)] += 1


def uniqueness(test_name):
    """
    Number of branches covered ONLY by this test.
    """

    score = 0

    for module, branches in test_branches[test_name].items():
        for branch in branches:
            if branch_count[(module, branch)] == 1:
                score += 1

    return score


def compute_module_coverage(selected_tests):
    """
    Returns:
        {
            module: coverage_ratio
        }
    """

    covered = defaultdict(set)

    for test in selected_tests:
        for module, branches in test_branches[test].items():
            covered[module] |= branches

    coverage = {}

    for module, total in all_module_branches.items():
        if not total:
            coverage[module] = 1.0
            continue

        cov = len(covered[module]) / len(total)

        coverage[module] = cov

    return coverage


def meets_threshold(selected_tests):

    coverage = compute_module_coverage(selected_tests)

    for module, cov in coverage.items():
        if cov < TARGET_BRANCH_COVERAGE:
            return False

    return True


current_selected = set(test_branches.keys())

initial_coverage = compute_module_coverage(current_selected)

print("\n=== INITIAL MODULE COVERAGE ===")

for module, cov in sorted(initial_coverage.items()):
    print(f"{module}: {cov:.4f}")

print("\n=== STARTING PRUNING ===")

sorted_tests = sorted(
    current_selected, key=lambda t: (uniqueness(t), len(test_branches[t]))
)

removed = []

for i, test in enumerate(sorted_tests, 1):
    if test not in current_selected:
        continue

    print(f"\n[{i}/{len(sorted_tests)}] Trying removal:")
    print(f"  {test}")

    candidate = current_selected - {test}

    if meets_threshold(candidate):
        current_selected.remove(test)
        removed.append(test)

        print("  REMOVED")

        coverage = compute_module_coverage(current_selected)

        minimum = min(coverage.values())

        print(f"  Min module coverage: {minimum:.4f}")

    else:
        print("  KEPT (required for coverage)")

final_coverage = compute_module_coverage(current_selected)

print("\n=== FINAL SUMMARY ===")

print(f"Original tests:  {len(test_branches)}")
print(f"Remaining tests: {len(current_selected)}")
print(f"Removed tests:   {len(removed)}")

reduction = 100 * len(removed) / len(test_branches)

print(f"Reduction:       {reduction:.2f}%")

print("\n=== FINAL MODULE BRANCH COVERAGE ===")

for module, cov in sorted(final_coverage.items()):
    status = "OK"

    if cov < TARGET_BRANCH_COVERAGE:
        status = "FAIL"

    print(f"{module}: {cov:.4f} [{status}]")

with open("pruned_tests.json", "w") as f:
    json.dump(sorted(list(current_selected)), f, indent=2)

with open("removed_tests.json", "w") as f:
    json.dump(sorted(removed), f, indent=2)

with open("final_module_coverage.json", "w") as f:
    json.dump(
        {module: round(cov, 4) for module, cov in sorted(final_coverage.items())},
        f,
        indent=2,
    )

print("\nSaved:")
print("  pruned_tests.json")
print("  removed_tests.json")
print("  final_module_coverage.json")
