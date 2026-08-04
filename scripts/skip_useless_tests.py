import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "output/test_suite_analysis"

ZERO_BRANCH_FILE = OUTPUT_DIR / "zero_branch_tests.json"
REMOVED_FILE = OUTPUT_DIR / "removed_tests.json"

with open(ZERO_BRANCH_FILE) as f:
    zero_branch_tests = json.load(f)

with open(REMOVED_FILE) as f:
    removed_tests = json.load(f)

zero_branch_set = set(zero_branch_tests)
removed_set = set(removed_tests)

grouped = {}

all_tests = sorted(zero_branch_set | removed_set)

for test in all_tests:
    file_path, test_name = test.split("::")

    grouped.setdefault(file_path, []).append(test_name)

for file_path, test_names in grouped.items():
    path = ROOT / file_path

    if not path.exists():
        print(f"Missing: {file_path}")
        continue

    lines = path.read_text().splitlines()

    modified = []
    inserted_pytest_import = False

    has_pytest_import = any("import pytest" in line for line in lines)

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("def "):
            fn_name = stripped.split("(")[0].replace("def ", "")
            matching_test = f"{file_path}::{fn_name}"

            if fn_name in test_names:
                start = len(modified)

                while start > 0 and modified[start - 1].strip().startswith("@"):
                    start -= 1

                if not has_pytest_import and not inserted_pytest_import:
                    modified.insert(0, "import pytest")
                    inserted_pytest_import = True

                    start += 1

                if matching_test in zero_branch_set:
                    modified.insert(start, "# Remove test due to no coverage")

                modified.insert(start, "@pytest.mark.pruned")

        modified.append(line)
        i += 1

    path.write_text("\n".join(modified))

    print(f"Updated: {file_path}")

print("\nDone")
