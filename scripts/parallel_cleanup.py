import shutil
from pathlib import Path

# Find the project root (two levels up from this script)
ROOT = Path(__file__).resolve().parents[1]

# Directory where test artifacts accumulate (coverage files, temp JSON, etc.)
OUTPUT_DIR = ROOT / "../output/test_suite_analysis"


# Patterns for temporary files to clean up
# These are created during test coverage analysis runs
PATTERNS = [
    ".coverage*",  # Coverage data files (pytest-cov creates these)
    "tmp_cov_*.json",  # Temporary coverage JSON files
    "tmp_*",  # Temporary directories used by pytest
]


def remove_path(path: Path):
    """
    Delete a file or directory.
    If it's a directory, recursively delete everything inside.
    If it's a file, just unlink (delete) it.
    """
    if path.is_dir():
        # Recursively delete directory and all contents
        shutil.rmtree(path)

    elif path.exists():
        # Delete single file
        path.unlink()


def main():
    """
    Main cleanup function:
    1. Find all files/dirs matching PATTERNS
    2. Try to delete each one
    3. Report what was removed and what failed
    """

    print("=================================================")
    print("CLEANING TEMPORARY TEST ARTIFACTS")
    print("=================================================\n")

    removed = []  # Successfully deleted paths
    failed = []  # Paths we couldn't delete (and why)

    # Process each pattern (e.g., ".coverage*", "tmp_*")
    for pattern in PATTERNS:
        print(f"Searching for: {pattern}")

        # Find all paths matching this pattern
        for path in OUTPUT_DIR.glob(pattern):
            try:
                # Try to delete it
                remove_path(path)

                # If successful, record it
                removed.append(path)

                print(f"  REMOVED: {path}")

            except Exception as e:
                # If it fails, record the error
                failed.append((path, str(e)))

                print(f"  FAILED:  {path}")
                print(f"           {e}")

        print()

    # Print summary statistics
    print("=================================================")
    print("CLEANUP SUMMARY")
    print("=================================================\n")

    print(f"Removed artifacts: {len(removed)}")
    print(f"Failures:          {len(failed)}")

    # If there were failures, print details
    if failed:
        print("\nFailed removals:\n")

        for path, err in failed:
            print(f"  {path}")
            print(f"    {err}")

    print("\nDone.")


if __name__ == "__main__":
    main()
