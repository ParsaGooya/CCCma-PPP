import ast
import json
from collections import defaultdict
from pathlib import Path

# Find the project root (two levels up from this script)
ROOT = Path(__file__).resolve().parents[1]

# Directory where test optimization results are stored
OUTPUT_DIR = Path("../output/test_suite_analysis")

# Input files from the test optimization process
TEST_MAP_FILE = OUTPUT_DIR / "test_map.json"  # All tests and their coverage
ZERO_BRANCH_FILE = (
    OUTPUT_DIR / "zero_branch_tests.json"
)  # Tests with no branch coverage
REMOVED_FILE = OUTPUT_DIR / "removed_tests.json"  # Tests to remove from the suite

# Decorator to mark tests as pruned (safe to remove)
PRUNED_MARKER = "@pytest.mark.pruned"
# Comment to mark tests that don't test any branches
ZERO_BRANCH_COMMENT = "# Remove test due to no coverage"


def load_json(path):
    """Load and parse a JSON file. Raise error if file doesn't exist."""
    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist: {path}")

    with path.open(
        encoding="utf-8",
    ) as file:
        return json.load(file)


def normalize_file_path(file_path):
    """
    Convert a file path to a normalized posix-style relative path.
    Handles both absolute and relative paths.
    """
    path = Path(file_path)

    # If it's an absolute path, convert to relative to ROOT
    if path.is_absolute():
        try:
            path = path.relative_to(ROOT)
        except ValueError:
            # If not under ROOT, just return as-is
            return path.as_posix()

    # Return using forward slashes (posix) for consistency
    return path.as_posix()


def parse_node_id(node_id):
    """
    Parse a pytest node ID into (file_path, test_path).

    Example: "tests/test_foo.py::TestBar::test_baz[param1]" ->
    ("tests/test_foo.py", "TestBar::test_baz")
    """
    file_path, separator, test_path = node_id.partition("::")

    # Invalid node ID if no "::" separator
    if not separator:
        return None

    # Split the test path by "::" to handle nested classes
    components = [component for component in test_path.split("::") if component]

    if not components:
        return None

    # Remove parametrization from the last component (e.g., "test_foo[param1]" -> "test_foo")
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
    Parse Python source and find line numbers for all test functions.
    Returns (ast_tree, locations_dict).

    locations_dict maps qualified_name -> {"definition_line": int, "insertion_line": int}
    insertion_line is the line before decorators (where we add new decorators)
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
        """Recursively walk AST nodes, tracking nested class paths."""
        for node in nodes:
            # Handle nested classes by updating the class path
            if isinstance(node, ast.ClassDef):
                visit(
                    node.body,
                    (
                        *class_path,
                        node.name,
                    ),
                )
                continue

            # Only interested in functions and async functions
            if not isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            ):
                continue

            # Build the qualified name (e.g., "TestClass::test_method")
            qualified_name = "::".join(
                (
                    *class_path,
                    node.name,
                )
            )

            # If function has decorators, insert marker before the first decorator
            # Otherwise, insert before the "def" line
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
    """Check if the module already imports pytest."""
    for node in tree.body:
        if isinstance(node, ast.Import):
            if any(alias.name == "pytest" for alias in node.names):
                return True

        if isinstance(node, ast.ImportFrom) and node.module == "pytest":
            return True

    return False


def pytest_import_index(tree):
    """
    Find the line index where we should add "import pytest".
    Returns the index right after module docstring and __future__ imports.
    """
    insertion_line = 1
    body = tree.body
    index = 0

    # Skip module docstring (first statement if it's a string constant)
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

    # Skip __future__ imports (must come first after docstring)
    while index < len(body):
        node = body[index]

        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            insertion_line = (node.end_lineno or node.lineno) + 1
            index += 1
            continue

        break

    # Convert to 0-based index
    return insertion_line - 1


def marker_exists_above(
    lines,
    insertion_index,
    marker,
):
    """
    Check if a marker (decorator or comment) already exists above this line.
    Looks upward through decorators and blank lines.
    """
    index = insertion_index - 1

    while index >= 0:
        stripped = lines[index].strip()

        # Skip blank lines
        if not stripped:
            index -= 1
            continue

        # Found the marker we're looking for
        if stripped == marker:
            return True

        # Skip over other decorators and comments
        if stripped.startswith("@") or stripped.startswith("#"):
            index -= 1
            continue

        # Hit a line that's not a decorator/comment - stop searching
        break

    return False


# Load the test optimization results
raw_map = load_json(TEST_MAP_FILE)
zero_branch_tests = load_json(ZERO_BRANCH_FILE)
removed_tests = load_json(REMOVED_FILE)

# Build sets for fast lookup
all_node_ids = set(raw_map)
zero_branch_set = set(zero_branch_tests)
removed_set = set(removed_tests)

# Tests that can be safely removed
removable_set = zero_branch_set | removed_set


# Map (file_path, test_path) -> set of node_ids for all tests
all_cases_by_function = defaultdict(set)

for node_id in all_node_ids:
    parsed = parse_node_id(node_id)

    if parsed is None:
        print(f"Invalid test node ID in test_map.json: {node_id}")
        continue

    key = parsed
    all_cases_by_function[key].add(node_id)


# Map (file_path, test_path) -> set of removable node_ids
removable_cases_by_function = defaultdict(set)

for node_id in removable_set:
    parsed = parse_node_id(node_id)

    if parsed is None:
        print(f"Invalid removable node ID: {node_id}")
        continue

    removable_cases_by_function[parsed].add(node_id)


# Separate functions into two categories:
# 1. functions_to_prune: all test cases are removable (can mark with decorator)
# 2. partial_functions: some test cases are removable, some are kept (can't use decorator)

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

    # Tests that will be kept in the suite
    selected_cases = all_cases - removable_cases

    if selected_cases:
        # Some tests are kept, some are removed - can't use a single decorator
        # (parametrized tests would need individual markers)
        partial_functions[key] = {
            "all": all_cases,
            "removable": removable_cases,
            "selected": selected_cases,
        }
        continue

    # All test cases are removable - safe to mark the entire function
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


# Report on parametrized functions that can't be safely marked
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


# Group functions by file for efficient processing
targets_by_file = defaultdict(dict)

for (
    file_path,
    test_path,
), information in functions_to_prune.items():
    targets_by_file[file_path][test_path] = information


total_marked = 0
total_unmatched = 0


# Process each file
for file_path, targets in sorted(targets_by_file.items()):
    path = ROOT / file_path

    # Check if file exists
    if not path.exists():
        print(f"Missing: {file_path}")
        total_unmatched += len(targets)
        continue

    source = path.read_text(
        encoding="utf-8",
    )
    lines = source.splitlines()

    # Parse the file to find function locations
    try:
        tree, locations = collect_function_locations(
            source,
            path,
        )
    except SyntaxError as error:
        print(f"Could not parse {file_path}: {error}")
        total_unmatched += len(targets)
        continue

    # Dictionary mapping line_index -> list of lines to insert
    insertions = defaultdict(list)
    marked_count = 0

    for test_path, information in sorted(targets.items()):
        location = locations.get(test_path)

        # Function not found in the file
        if location is None:
            print(f"Unmatched: {file_path}::{test_path}")
            total_unmatched += 1
            continue

        # Get the line where we should insert the decorator
        insertion_index = location["insertion_line"] - 1

        # Get the function definition line (to extract indentation)
        definition_index = location["definition_line"] - 1

        definition_line = lines[definition_index]

        # Extract indentation from the definition line
        indentation = definition_line[
            : len(definition_line) - len(definition_line.lstrip())
        ]

        additions = []

        # Add zero-branch comment if applicable and not already present
        if information["zero_branch"] and not marker_exists_above(
            lines,
            insertion_index,
            ZERO_BRANCH_COMMENT,
        ):
            additions.append(indentation + ZERO_BRANCH_COMMENT)

        # Add pruned marker if not already present
        if not marker_exists_above(
            lines,
            insertion_index,
            PRUNED_MARKER,
        ):
            additions.append(indentation + PRUNED_MARKER)

        # Record the lines to insert before this function
        if additions:
            insertions[insertion_index].extend(additions)
            marked_count += 1

    # Check if we need to add "import pytest" at the top of the file
    needs_pytest_import = marked_count > 0 and not imports_pytest(tree)

    if needs_pytest_import:
        import_index = pytest_import_index(tree)
        import_additions = ["import pytest"]

        # Add blank line after import if there's code after it
        if import_index < len(lines) and lines[import_index].strip():
            import_additions.append("")

        insertions[import_index][0:0] = import_additions

    # Build the modified source by inserting lines at the correct positions
    modified = []

    for index, line in enumerate(lines):
        if index in insertions:
            # Add any lines that should be inserted before this line
            modified.extend(insertions[index])

        # Add the original line
        modified.append(line)

    # Add any lines that should be inserted at the end of the file
    if len(lines) in insertions:
        modified.extend(insertions[len(lines)])

    updated_source = "\n".join(modified) + "\n"

    # Verify the modified source is valid Python before writing
    try:
        ast.parse(
            updated_source,
            filename=str(path),
        )
    except SyntaxError as error:
        print(f"Refusing to write invalid result for {file_path}: {error}")
        total_unmatched += len(targets)
        continue

    # Write the modified source back to the file
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
