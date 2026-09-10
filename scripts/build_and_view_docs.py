import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Find the root of the project (two levels up from this script)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Set up all the directory paths we'll need
DOCS_DIR = PROJECT_ROOT / "docs"
SOURCE_DIR = DOCS_DIR / "source"  # Where the .rst documentation files live
API_DIR = SOURCE_DIR / "api"  # Where auto-generated API docs go

# The actual package we're documenting
PACKAGE_DIR = PROJECT_ROOT / "cccma_ppp"

# Output directories for the build
BUILD_DIR = DOCS_DIR / "_build"
HTML_DIR = BUILD_DIR / "html"  # Final HTML output
WARNINGS_FILE = BUILD_DIR / "warnings.txt"  # Log of any build warnings/errors


def run(cmd, cwd=None):
    """
    Execute a shell command and print it before running.
    Returns the exit code (0 = success, non-zero = failure).
    """
    # Print the command so we can see what's happening
    print("$", " ".join(str(x) for x in cmd))

    result = subprocess.run(
        [str(x) for x in cmd],
        cwd=cwd,
        check=False,  # Don't crash if command fails (we handle the return code)
    )

    return result.returncode


def clean():
    """
    Delete old build outputs and generated API docs.
    Useful before doing a fresh build to avoid stale files.
    """
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)

    if API_DIR.exists():
        shutil.rmtree(API_DIR)


def generate_api():
    """
    Auto-generate API documentation from the source code.
    This creates .rst files by reading docstrings from the package.
    Uses sphinx.ext.apidoc to do the heavy lifting.
    """

    # Make sure the package actually exists (sanity check)
    if not PACKAGE_DIR.exists():
        print(f"ERROR: Package directory does not exist: {PACKAGE_DIR}")
        return 1

    # Clean up any old API docs before regenerating
    if API_DIR.exists():
        shutil.rmtree(API_DIR)

    # Create the API docs output directory
    API_DIR.mkdir(parents=True, exist_ok=True)

    # Build the command to run sphinx's apidoc tool
    # -f: force overwrite existing files
    # -o: output directory
    cmd = [
        sys.executable,  # Use same Python interpreter as this script
        "-m",
        "sphinx.ext.apidoc",  # Run the apidoc module
        "-f",  # Force overwrite
        "-o",
        str(API_DIR),  # Output directory
        str(PACKAGE_DIR),  # Source package to document
    ]

    return run(cmd, cwd=PROJECT_ROOT)


def build(strict=False):
    """
    Run Sphinx to build the HTML documentation.

    Args:
        strict: If True, treat warnings as errors (build fails if any warnings)
    """
    cmd = [
        sys.executable,
        "-m",
        "sphinx",  # Run Sphinx
        "-b",
        "html",  # Build HTML (not PDF, man pages, etc.)
        "--keep-going",  # Don't stop at first error (keep building everything)
    ]

    # In strict mode, warnings become errors (useful for CI/CD)
    if strict:
        cmd.append("-W")

    # Add the source and output directories, plus warnings log file
    cmd.extend(
        [
            str(SOURCE_DIR),  # Where the .rst files are
            str(HTML_DIR),  # Where to put the HTML output
            "-w",  # Write warnings to file
            str(WARNINGS_FILE),
        ]
    )

    return run(cmd, cwd=PROJECT_ROOT)


def summarize_warnings():
    """
    Read the warnings file and print a summary.
    Shows the first 20 warnings, then says how many more there are.
    Helps catch documentation issues without scrolling through everything.
    """
    if not WARNINGS_FILE.exists():
        return

    # Read the warnings file (ignore encoding errors in case there are weird characters)
    lines = WARNINGS_FILE.read_text(errors="ignore").splitlines()

    # Filter for lines that mention warnings or errors
    issues = [line for line in lines if "WARNING" in line or "ERROR" in line]

    print()
    print(f"Warnings: {len(issues)}")

    # Print the first 20 issues so you can see what needs fixing
    for line in issues[:20]:
        print(line)

    # If there are more, let user know there's more to look at
    if len(issues) > 20:
        print(f"... and {len(issues) - 20} more")


def serve(port):
    """
    Start a simple HTTP server to view the documentation locally.
    This is way easier than trying to open file:// URLs in a browser.
    """
    subprocess.run(
        [
            sys.executable,
            "-m",
            "http.server",  # Python's built-in web server
            str(port),
            "-d",  # Serve from this directory
            str(HTML_DIR),
        ],
        check=False,
    )


def main():
    """Main entry point - handles CLI arguments and orchestrates the build process"""

    parser = argparse.ArgumentParser()

    # Option to treat warnings as errors (good for CI/CD to catch doc issues early)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors",
    )

    # Option to serve docs after building (defaults to True)
    parser.add_argument(
        "--serve",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Serve generated HTML after a successful build",
    )

    # Custom port for the server (in case 18000 is already in use)
    parser.add_argument(
        "--port",
        type=int,
        default=18000,
        help="Port used by the documentation server (default: 18000)",
    )

    args = parser.parse_args()

    # Start fresh - delete old build artifacts
    clean()

    # Step 1: Generate API docs from docstrings
    print("Generating API documentation...")
    rc = generate_api()

    if rc != 0:
        # If API generation failed, stop here
        sys.exit(rc)

    # Step 2: Run Sphinx to build the actual HTML
    print()
    print("Building documentation...")
    rc = build(strict=args.strict)

    # Tell user where the docs are
    print()
    print("HTML:")
    print(HTML_DIR / "index.html")

    # Show any warnings that came up during the build
    summarize_warnings()

    # Step 3: Optionally serve the docs locally
    if rc == 0 and args.serve:
        print()
        print(f"Serving documentation at http://localhost:{args.port}")
        print("Press Ctrl+C to stop the server.")
        serve(args.port)

    elif rc == 0:
        # Build succeeded but user disabled server
        print()
        print("Documentation build completed. Server disabled.")

    # Exit with the same code as the build (0 = success, non-zero = failure)
    sys.exit(rc)


if __name__ == "__main__":
    main()
