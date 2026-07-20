import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd().resolve()
PYTEST_CACHE = ROOT / ".pytest_cache" / "v" / "cache" / "lastfailed"

FAILED_COMMENT = "# FAILED during full pytest run"


def clear_pytest_cache():
    cache_directory = ROOT / ".pytest_cache"

    if not cache_directory.exists():
        return

    import shutil

    shutil.rmtree(cache_directory)

    print(f"Removed pytest cache: {cache_directory}")


def run_full_test_suite():
    print("\nRunning the complete pytest suite from scratch...\n")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--tb=no",
        ],
        cwd=ROOT,
        check=False,
    )

    print(f"\nPytest finished with exit code {result.returncode}.")

    return result.returncode


def load_failed_node_ids():
    if not PYTEST_CACHE.exists():
        print("\nNo pytest lastfailed cache was created.")
        return []

    try:
        cache_contents = json.loads(PYTEST_CACHE.read_text())
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Could not parse {PYTEST_CACHE}") from error

    return sorted(node_id for node_id, failed in cache_contents.items() if failed)


def normalize_failed_tests(node_ids):
    grouped = {}

    for node_id in node_ids:
        parts = node_id.split("::")

        if len(parts) < 2:
            print(
                f"Cannot annotate collection failure without a test function: {node_id}"
            )
            continue

        file_path = parts[0]
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
    lines = [node.lineno]

    lines.extend(decorator.lineno for decorator in node.decorator_list)

    return min(lines)


def find_failed_functions(source, failed_names):
    tree = ast.parse(source)
    matches = []

    for node in ast.walk(tree):
        if not isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            continue

        if node.name not in failed_names:
            continue

        matches.append(
            {
                "name": node.name,
                "function_line": node.lineno,
                "start_line": get_function_start(node),
            }
        )

    return matches


def comment_already_present(lines, start_index):
    index = start_index - 1

    while index >= 0:
        stripped = lines[index].strip()

        if stripped == FAILED_COMMENT:
            return True

        if stripped.startswith("@"):
            index -= 1
            continue

        if not stripped:
            index -= 1
            continue

        break

    return False


def annotate_file(file_path, failed_names):
    path = ROOT / file_path

    if not path.is_file():
        print(f"Missing source file: {file_path}")
        return 0, set()

    if path.suffix != ".py":
        print(f"Skipping non-Python file: {file_path}")
        return 0, set()

    source = path.read_text()

    try:
        matches = find_failed_functions(
            source,
            failed_names,
        )
    except SyntaxError as error:
        print(f"Could not parse {file_path}: {error}")
        return 0, set()

    lines = source.splitlines()
    insertions = []
    found_names = set()

    for match in matches:
        found_names.add(match["name"])

        start_index = match["start_line"] - 1
        function_index = match["function_line"] - 1

        if comment_already_present(
            lines,
            start_index,
        ):
            continue

        function_line = lines[function_index]

        indentation = function_line[: len(function_line) - len(function_line.lstrip())]

        insertions.append(
            (
                start_index,
                f"{indentation}{FAILED_COMMENT}",
            )
        )

    for index, comment in sorted(
        insertions,
        reverse=True,
    ):
        lines.insert(index, comment)

    if insertions:
        updated_source = "\n".join(lines).rstrip() + "\n"

        try:
            ast.parse(
                updated_source,
                filename=str(path),
            )
        except SyntaxError as error:
            raise RuntimeError(
                f"Adding comments made {file_path} invalid Python."
            ) from error

        path.write_text(updated_source)

        print(f"Updated: {file_path} ({len(insertions)} comment(s))")

    return len(insertions), found_names


def annotate_failed_tests(node_ids):
    grouped = normalize_failed_tests(node_ids)

    total_added = 0
    total_failed_functions = sum(len(names) for names in grouped.values())

    for file_path, failed_names in sorted(grouped.items()):
        added, found_names = annotate_file(
            file_path,
            failed_names,
        )

        total_added += added

        missing_names = failed_names - found_names

        for name in sorted(missing_names):
            print(f"Could not locate test function: {file_path}::{name}")

    print()
    print(f"Detected {total_failed_functions} failed test function(s).")
    print(f"Added {total_added} new failure comment(s).")


def main():
    clear_pytest_cache()

    pytest_exit_code = run_full_test_suite()

    failed_node_ids = load_failed_node_ids()

    if not failed_node_ids:
        print("\nNo failed test functions were found.")
        return pytest_exit_code

    print(f"\nFound {len(failed_node_ids)} failed pytest node ID(s).")

    annotate_failed_tests(failed_node_ids)

    return pytest_exit_code


if __name__ == "__main__":
    raise SystemExit(main())
