import ast
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "output" / "test_suite_analysis"

TEST_MAP_FILE = OUTPUT_DIR / "test_map.json"
ZERO_BRANCH_FILE = OUTPUT_DIR / "zero_branch_tests.json"
REMOVED_FILE = OUTPUT_DIR / "removed_tests.json"

PRUNED_MARKER = "@pytest.mark.pruned"
ZERO_BRANCH_COMMENT = "# Remove test due to no coverage"


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist: {path}")

    with path.open(
        encoding="utf-8",
    ) as file:
        return json.load(file)


def normalize_file_path(file_path):
    path = Path(file_path)

    if path.is_absolute():
        try:
            path = path.relative_to(ROOT)
        except ValueError:
            return path.as_posix()

    return path.as_posix()


def parse_node_id(node_id):
    """
    Parse a pytest node ID into:

    - project-relative file path
    - class-qualified function path without parameters

    Examples
    --------
    tests/a_test.py::test_one[param]
        -> tests/a_test.py, test_one

    tests/a_test.py::TestGroup::test_one[param]
        -> tests/a_test.py, TestGroup::test_one
    """
    file_path, separator, test_path = node_id.partition("::")

    if not separator:
        return None

    components = [component for component in test_path.split("::") if component]

    if not components:
        return None

    components[-1] = components[-1].split(
        "[",
        1,
    )[0]

    return (
        normalize_file_path(file_path),
        "::".join(components),
    )


def collect_function_locations(source, path):
    """
    Return class-qualified function locations from a Python test file.
    """
    tree = ast.parse(
        source,
        filename=str(path),
    )

    locations = {}

    def visit(
        nodes,
        class_path=(),
    ):
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                visit(
                    node.body,
                    (
                        *class_path,
                        node.name,
                    ),
                )
                continue

            if not isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            ):
                continue

            qualified_name = "::".join(
                (
                    *class_path,
                    node.name,
                )
            )

            if node.decorator_list:
                insertion_line = min(
                    decorator.lineno for decorator in node.decorator_list
                )
            else:
                insertion_line = node.lineno

            locations[qualified_name] = {
                "definition_line": node.lineno,
                "insertion_line": insertion_line,
            }

    visit(tree.body)

    return tree, locations


def imports_pytest(tree):
    for node in tree.body:
        if isinstance(node, ast.Import):
            if any(alias.name == "pytest" for alias in node.names):
                return True

        if isinstance(node, ast.ImportFrom) and node.module == "pytest":
            return True

    return False


def pytest_import_index(tree):
    """
    Return a zero-based safe insertion index for import pytest.
    """
    insertion_line = 1
    body = tree.body
    index = 0

    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(
            body[0].value,
            ast.Constant,
        )
        and isinstance(
            body[0].value.value,
            str,
        )
    ):
        insertion_line = (body[0].end_lineno or body[0].lineno) + 1
        index = 1

    while index < len(body):
        node = body[index]

        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            insertion_line = (node.end_lineno or node.lineno) + 1
            index += 1
            continue

        break

    return insertion_line - 1


def marker_exists_above(
    lines,
    insertion_index,
    marker,
):
    index = insertion_index - 1

    while index >= 0:
        stripped = lines[index].strip()

        if not stripped:
            index -= 1
            continue

        if stripped == marker:
            return True

        if stripped.startswith("@") or stripped.startswith("#"):
            index -= 1
            continue

        break

    return False


raw_map = load_json(TEST_MAP_FILE)
zero_branch_tests = load_json(ZERO_BRANCH_FILE)
removed_tests = load_json(REMOVED_FILE)

all_node_ids = set(raw_map)
zero_branch_set = set(zero_branch_tests)
removed_set = set(removed_tests)

removable_set = zero_branch_set | removed_set


all_cases_by_function = defaultdict(set)

for node_id in all_node_ids:
    parsed = parse_node_id(node_id)

    if parsed is None:
        print(f"Invalid test node ID in test_map.json: {node_id}")
        continue

    key = parsed
    all_cases_by_function[key].add(node_id)


removable_cases_by_function = defaultdict(set)

for node_id in removable_set:
    parsed = parse_node_id(node_id)

    if parsed is None:
        print(f"Invalid removable node ID: {node_id}")
        continue

    removable_cases_by_function[parsed].add(node_id)


functions_to_prune = {}
partial_functions = {}

for key, removable_cases in removable_cases_by_function.items():
    all_cases = all_cases_by_function.get(
        key,
        set(),
    )

    if not all_cases:
        print(f"Removable test was not found in test_map.json: {key}")
        continue

    selected_cases = all_cases - removable_cases

    if selected_cases:
        partial_functions[key] = {
            "all": all_cases,
            "removable": removable_cases,
            "selected": selected_cases,
        }
        continue

    functions_to_prune[key] = {
        "zero_branch": all(case in zero_branch_set for case in all_cases),
        "cases": all_cases,
    }


print(f"Collected test cases: {len(all_node_ids)}")
print(f"Removable test cases: {len(removable_set)}")
print(f"Functions safe to mark: {len(functions_to_prune)}")
print(
    "Partially removable parameterized functions "
    f"left unchanged: {len(partial_functions)}"
)


if partial_functions:
    print(
        "\nParameterized functions that cannot be "
        "safely pruned with a function decorator:"
    )

    for (
        file_path,
        test_path,
    ), details in sorted(partial_functions.items()):
        print(
            f"  {file_path}::{test_path}: "
            f"{len(details['removable'])} removable, "
            f"{len(details['selected'])} selected"
        )


targets_by_file = defaultdict(dict)

for (
    file_path,
    test_path,
), information in functions_to_prune.items():
    targets_by_file[file_path][test_path] = information


total_marked = 0
total_unmatched = 0


for file_path, targets in sorted(targets_by_file.items()):
    path = ROOT / file_path

    if not path.exists():
        print(f"Missing: {file_path}")
        total_unmatched += len(targets)
        continue

    source = path.read_text(
        encoding="utf-8",
    )
    lines = source.splitlines()

    try:
        tree, locations = collect_function_locations(
            source,
            path,
        )
    except SyntaxError as error:
        print(f"Could not parse {file_path}: {error}")
        total_unmatched += len(targets)
        continue

    insertions = defaultdict(list)
    marked_count = 0

    for test_path, information in sorted(targets.items()):
        location = locations.get(test_path)

        if location is None:
            print(f"Unmatched: {file_path}::{test_path}")
            total_unmatched += 1
            continue

        insertion_index = location["insertion_line"] - 1

        definition_index = location["definition_line"] - 1

        definition_line = lines[definition_index]

        indentation = definition_line[
            : len(definition_line) - len(definition_line.lstrip())
        ]

        additions = []

        if information["zero_branch"] and not marker_exists_above(
            lines,
            insertion_index,
            ZERO_BRANCH_COMMENT,
        ):
            additions.append(indentation + ZERO_BRANCH_COMMENT)

        if not marker_exists_above(
            lines,
            insertion_index,
            PRUNED_MARKER,
        ):
            additions.append(indentation + PRUNED_MARKER)

        if additions:
            insertions[insertion_index].extend(additions)
            marked_count += 1

    needs_pytest_import = marked_count > 0 and not imports_pytest(tree)

    if needs_pytest_import:
        import_index = pytest_import_index(tree)
        import_additions = ["import pytest"]

        if import_index < len(lines) and lines[import_index].strip():
            import_additions.append("")

        insertions[import_index][0:0] = import_additions

    modified = []

    for index, line in enumerate(lines):
        if index in insertions:
            modified.extend(insertions[index])

        modified.append(line)

    if len(lines) in insertions:
        modified.extend(insertions[len(lines)])

    updated_source = "\n".join(modified) + "\n"

    try:
        ast.parse(
            updated_source,
            filename=str(path),
        )
    except SyntaxError as error:
        print(f"Refusing to write invalid result for {file_path}: {error}")
        total_unmatched += len(targets)
        continue

    path.write_text(
        updated_source,
        encoding="utf-8",
    )

    total_marked += marked_count

    print(f"Updated: {file_path} ({marked_count} functions marked)")


print()
print("Done")
print(f"Functions marked: {total_marked}")
print(f"Functions unmatched: {total_unmatched}")
print(f"Parameterized functions left unchanged: {len(partial_functions)}")
