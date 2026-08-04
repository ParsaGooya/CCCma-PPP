import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "output/test_suite_analysis"


PATTERNS = [
    ".coverage*",
    "tmp_cov_*.json",
    "tmp_*",
]


def remove_path(path: Path):
    if path.is_dir():
        shutil.rmtree(path)

    elif path.exists():
        path.unlink()


def main():

    print("=================================================")
    print("CLEANING TEMPORARY TEST ARTIFACTS")
    print("=================================================\n")

    removed = []
    failed = []

    for pattern in PATTERNS:
        print(f"Searching for: {pattern}")

        for path in OUTPUT_DIR.glob(pattern):
            try:
                remove_path(path)

                removed.append(path)

                print(f"  REMOVED: {path}")

            except Exception as e:
                failed.append((path, str(e)))

                print(f"  FAILED:  {path}")
                print(f"           {e}")

        print()

    print("=================================================")
    print("CLEANUP SUMMARY")
    print("=================================================\n")

    print(f"Removed artifacts: {len(removed)}")
    print(f"Failures:           {len(failed)}")

    if failed:
        print("\nFailed removals:\n")

        for path, err in failed:
            print(f"  {path}")
            print(f"    {err}")

    print("\nDone.")


if __name__ == "__main__":
    main()
