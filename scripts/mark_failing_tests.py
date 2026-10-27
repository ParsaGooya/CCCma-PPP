import ast
import json
import subprocess
import sys
from pathlib import Path

# Find the project root directory
ROOT = Path.cwd().resolve()
# Pytest stores a cache of failed tests here (after a run)
PYTEST_CACHE = ROOT / ".pytest_cache" / "v" / "cache" / "lastfailed"

# The comment we'll add above failed test functions
FAILED_COMMENT = "# FAILED during full pytest run"


def clear_pytest_cache():
    """
    Delete pytest's cache directory completely.
    This forces pytest to run all tests (not just failed ones) on the next run.
    Useful if you want a fresh start or if the cache is corrupted.
    """
    cache_directory = ROOT / ".pytest_cache"

    if not cache_directory.exists():
        return

    import shutil

    shutil.rmtree(cache_directory)

    print(f"Removed pytest cache: {cache_directory}")


def run_full_test_suite():
    """
    Execute the full pytest test suite.
    Pytest will automatically record which tests fail in its cache.

    Returns the exit code from pytest (0 = all passed, non-zero = some failed).
    """
    print("\nRunning the complete pytest suite from scratch...\n")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",  # Quiet mode (less output)
            "--tb=no",  # No tracebacks (we just want pass/fail status)
        ],
        cwd=ROOT,
        check=False,  # Don't raise if pytest exits with non-zero (some tests failed)
    )

    print(f"\nPytest finished with exit code {result.returncode}.")

    return result.returncode


def load_failed_node_ids():
    """
    Read pytest's cache to find which tests failed.
    Returns a list of pytest node IDs (e.g., "tests/test_foo.py::test_bar").

    The cache is a JSON file where each key is a node ID and value is True if it failed.
    """
    if not PYTEST_CACHE.exists():
        print("\nNo pytest lastfailed cache was created.")
        return []

    try:
        cache_contents = json.loads(PYTEST_CACHE.read_text())
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Could not parse {PYTEST_CACHE}") from error

    # Filter to only the failed tests (value is True)
    return sorted(node_id for node_id, failed in cache_contents.items() if failed)


def normalize_failed_tests(node_ids):
    """
    Group failed tests by their source file.
    Converts pytest node IDs into a dict mapping file -> set of function names.

    Example:
    Input: ["tests/test_foo.py::test_bar", "tests/test_foo.py::test_baz"]
    Output: {"tests/test_foo.py": {"test_bar", "test_baz"}}
    """
    grouped = {}

    for node_id in node_ids:
        parts = node_id.split("::")

        # Node IDs should have at least "file.py::function_name"
        if len(parts) < 2:
            print(
                f"Cannot annotate collection failure without a test function: {node_id}"
            )
            continue

        file_path = parts[0]
        # Handle parametrized tests like "test_foo[param1]" -> extract "test_foo"
        function_name = parts[-1].split(
            "[",
            maxsplit=1,
        )[0]

        grouped.setdefault(
            file_path,
            set(),
        ).add(function_name)

    return grouped


def get_function_start(node):
    """
    Get the line number where a function definition starts.
    This accounts for decorators, which appear before the def line.
    Returns the line number of the first decorator (or def if no decorators).
    """
    lines = [node.lineno]

    # Decorators have their own line numbers, and come before the function
    lines.extend(decorator.lineno for decorator in node.decorator_list)

    return min(lines)


def find_failed_functions(source, failed_names):
    """
    Parse Python source code and find functions that match the failed test names.
    Returns list of dicts with: name, function_line (def line), start_line (first decorator or def).
    """
    tree = ast.parse(source)
    matches = []

    for node in ast.walk(tree):
        # Only interested in function definitions
        if not isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            continue

        # Only interested in functions whose name matches a failed test
        if node.name not in failed_names:
            continue

        matches.append(
            {
                "name": node.name,
                "function_line": node.lineno,  # Line with "def test_foo():"
                "start_line": get_function_start(node),  # Line with @decorator or def
            }
        )

    return matches


def comment_already_present(lines, start_index):
    """
    Check if the FAILED_COMMENT already exists above this function.
    Looks upward from start_index through decorators and blank lines.
    Returns True if the comment is already there (to avoid duplicates).
    """
    index = start_index - 1

    while index >= 0:
        stripped = lines[index].strip()

        # Found the comment we're looking for!
        if stripped == FAILED_COMMENT:
            return True

        # Skip over decorators (they come before the comment)
        if stripped.startswith("@"):
            index -= 1
            continue

        # Skip over blank lines
        if not stripped:
            index -= 1
            continue

        # Hit a line of actual code - no comment found
        break

    return False


def annotate_file(file_path, failed_names):
    """
    Add FAILED_COMMENT above all failed test functions in a single file.
    Returns (number_of_comments_added, set_of_function_names_found).
    """
    path = ROOT / file_path

    # Make sure the file exists
    if not path.is_file():
        print(f"Missing source file: {file_path}")
        return 0, set()

    # Only process Python files
    if path.suffix != ".py":
        print(f"Skipping non-Python file: {file_path}")
        return 0, set()

    source = path.read_text()

    # Parse the source and find failed test functions
    try:
        matches = find_failed_functions(
            source,
            failed_names,
        )
    except SyntaxError as error:
        print(f"Could not parse {file_path}: {error}")
        return 0, set()

    lines = source.splitlines()
    insertions = []  # List of (line_index, comment_text) to insert
    found_names = set()

    # Process each failed function we found
    for match in matches:
        found_names.add(match["name"])

        start_index = match["start_line"] - 1  # Convert to 0-based indexing
        function_index = match["function_line"] - 1

        # Check if we already added this comment (avoid duplicates)
        if comment_already_present(
            lines,
            start_index,
        ):
            continue

        # Get the indentation from the function definition line
        function_line = lines[function_index]

        indentation = function_line[: len(function_line) - len(function_line.lstrip())]

        # Record that we want to insert a comment here
        insertions.append(
            (
                start_index,
                f"{indentation}{FAILED_COMMENT}",
            )
        )

    # Insert the comments in reverse order so line numbers don't shift
    for index, comment in sorted(
        insertions,
        reverse=True,
    ):
        lines.insert(index, comment)

    # If we made changes, verify they're valid Python and write the file
    if insertions:
        updated_source = "\n".join(lines).rstrip() + "\n"

        try:
            # Parse the modified source to verify it's still valid Python
            ast.parse(
                updated_source,
                filename=str(path),
            )
        except SyntaxError as error:
            raise RuntimeError(
                f"Adding comments made {file_path} invalid Python."
            ) from error

        # Write the modified source back to the file
        path.write_text(updated_source)

        print(f"Updated: {file_path} ({len(insertions)} comment(s))")

    return len(insertions), found_names


def annotate_failed_tests(node_ids):
    """
    Main logic: add FAILED_COMMENT above all failed test functions.
    Handles grouping by file, finding functions, and reporting results.
    """
    # Group failed tests by file
    grouped = normalize_failed_tests(node_ids)

    total_added = 0
    total_failed_functions = sum(len(names) for names in grouped.values())

    # Process each file
    for file_path, failed_names in sorted(grouped.items()):
        added, found_names = annotate_file(
            file_path,
            failed_names,
        )

        total_added += added

        # Report any test functions we couldn't find in the source
        missing_names = failed_names - found_names

        for name in sorted(missing_names):
            print(f"Could not locate test function: {file_path}::{name}")

    # Print summary
    print()
    print(f"Detected {total_failed_functions} failed test function(s).")
    print(f"Added {total_added} new failure comment(s).")


def main():
    """
    Main entry point:
    1. Clear pytest cache (forces full run)
    2. Run pytest and collect failures
    3. Annotate source code with FAILED_COMMENT above each failed test
    """
    # Start fresh - delete old cache
    clear_pytest_cache()

    # Run all tests, pytest will record which ones fail
    pytest_exit_code = run_full_test_suite()

    # Read the list of failed test node IDs from pytest's cache
    failed_node_ids = load_failed_node_ids()

    if not failed_node_ids:
        print("\nNo failed test functions were found.")
        return pytest_exit_code

    print(f"\nFound {len(failed_node_ids)} failed pytest node ID(s).")

    # Add comments above the failed test functions in the source code
    annotate_failed_tests(failed_node_ids)

    return pytest_exit_code


if __name__ == "__main__":
    raise SystemExit(main())
