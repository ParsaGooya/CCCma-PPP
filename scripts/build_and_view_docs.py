import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCS_DIR = PROJECT_ROOT / "docs"
SOURCE_DIR = DOCS_DIR / "source"

BUILD_DIR = DOCS_DIR / "_build"
HTML_DIR = BUILD_DIR / "html"

WARNINGS_FILE = BUILD_DIR / "warnings.txt"


def run(cmd, cwd=None):
    result = subprocess.run(
        [str(x) for x in cmd],
        cwd=cwd,
    )
    return result.returncode


def clean():
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)


def build(strict=False):
    cmd = [
        sys.executable,
        "-m",
        "sphinx",
        "-b",
        "html",
        "--keep-going",
        str(SOURCE_DIR),
        str(HTML_DIR),
        "-w",
        str(WARNINGS_FILE),
    ]

    if strict:
        cmd.insert(5, "-W")

    return run(cmd, cwd=PROJECT_ROOT)


def summarize_warnings():

    if not WARNINGS_FILE.exists():
        return

    lines = WARNINGS_FILE.read_text(errors="ignore").splitlines()

    issues = [line for line in lines if "WARNING" in line or "ERROR" in line]

    print()
    print(f"Warnings: {len(issues)}")

    preview = issues[:20]

    for line in preview:
        print(line)

    if len(issues) > len(preview):
        print(f"... and {len(issues) - len(preview)} more")


def serve(port):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "http.server",
            str(port),
            "-d",
            str(HTML_DIR),
        ]
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors",
    )

    parser.add_argument(
        "--serve",
        action="store_true",
        help="Serve generated HTML",
    )

    args = parser.parse_args()

    clean()

    rc = build(strict=args.strict)

    print()
    print("HTML:")
    print(HTML_DIR / "index.html")

    summarize_warnings()

    if rc == 0:
        serve(18000)

    sys.exit(rc)


if __name__ == "__main__":
    main()
