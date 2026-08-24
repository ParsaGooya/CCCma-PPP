import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCS_DIR = PROJECT_ROOT / "docs"
SOURCE_DIR = DOCS_DIR / "source"
API_DIR = SOURCE_DIR / "api"

PACKAGE_DIR = PROJECT_ROOT / "cccma_ppp"

BUILD_DIR = DOCS_DIR / "_build"
HTML_DIR = BUILD_DIR / "html"
WARNINGS_FILE = BUILD_DIR / "warnings.txt"


def run(cmd, cwd=None):
    print("$", " ".join(str(x) for x in cmd))

    result = subprocess.run(
        [str(x) for x in cmd],
        cwd=cwd,
        check=False,
    )

    return result.returncode


def clean():
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)

    if API_DIR.exists():
        shutil.rmtree(API_DIR)


def generate_api():

    if not PACKAGE_DIR.exists():
        print(f"ERROR: Package directory does not exist: {PACKAGE_DIR}")
        return 1

    if API_DIR.exists():
        shutil.rmtree(API_DIR)

    API_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "sphinx.ext.apidoc",
        "-f",
        "-o",
        str(API_DIR),
        str(PACKAGE_DIR),
    ]

    return run(cmd, cwd=PROJECT_ROOT)


def build(strict=False):
    cmd = [
        sys.executable,
        "-m",
        "sphinx",
        "-b",
        "html",
        "--keep-going",
    ]

    if strict:
        cmd.append("-W")

    cmd.extend(
        [
            str(SOURCE_DIR),
            str(HTML_DIR),
            "-w",
            str(WARNINGS_FILE),
        ]
    )

    return run(cmd, cwd=PROJECT_ROOT)


def summarize_warnings():
    if not WARNINGS_FILE.exists():
        return

    lines = WARNINGS_FILE.read_text(errors="ignore").splitlines()

    issues = [line for line in lines if "WARNING" in line or "ERROR" in line]

    print()
    print(f"Warnings: {len(issues)}")

    for line in issues[:20]:
        print(line)

    if len(issues) > 20:
        print(f"... and {len(issues) - 20} more")


def serve(port):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "http.server",
            str(port),
            "-d",
            str(HTML_DIR),
        ],
        check=False,
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Serve generated HTML after a successful build",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=18000,
        help="Port used by the documentation server (default: 18000)",
    )

    args = parser.parse_args()

    clean()

    print("Generating API documentation...")
    rc = generate_api()

    if rc != 0:
        sys.exit(rc)

    print()
    print("Building documentation...")
    rc = build(strict=args.strict)

    print()
    print("HTML:")
    print(HTML_DIR / "index.html")

    summarize_warnings()

    if rc == 0 and args.serve:
        print()
        print(f"Serving documentation at http://localhost:{args.port}")
        print("Press Ctrl+C to stop the server.")
        serve(args.port)

    elif rc == 0:
        print()
        print("Documentation build completed. Server disabled.")

    sys.exit(rc)


if __name__ == "__main__":
    main()
