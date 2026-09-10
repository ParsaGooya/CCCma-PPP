import shutil
import subprocess
import sys
from pathlib import Path

# Find project root (two levels up from this script)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Documentation directory structure
DOCS_DIR = PROJECT_ROOT / "docs"
SOURCE_DIR = DOCS_DIR / "source" / "api"  # Where auto-generated API docs go
BUILD_DIR = DOCS_DIR / "_build"  # Where Sphinx outputs built HTML

# The package we're documenting
PACKAGE_NAME = "cccma_ppp"
PACKAGE_DIR = PROJECT_ROOT / PACKAGE_NAME

# Metadata for the documentation
PROJECT_NAME = "CCCma PPP"
AUTHOR = "Parsa Gooya"


def run(cmd, cwd=None):
    """
    Execute a shell command and raise an error if it fails.
    Converts all arguments to strings (makes Path objects work).
    """
    result = subprocess.run(
        [str(x) for x in cmd],
        cwd=cwd,
    )

    # If command failed (non-zero return code), exit with same code
    if result.returncode:
        raise SystemExit(result.returncode)


def ensure_package():
    """
    Make sure every directory in the package has an __init__.py file.
    This is required for Python to recognize directories as packages.
    Useful if someone added a new directory but forgot the __init__.py.
    """
    for directory in PACKAGE_DIR.rglob("*"):
        if directory.is_dir():
            # touch() creates the file if it doesn't exist, does nothing if it does
            (directory / "__init__.py").touch(exist_ok=True)


def clean():
    """
    Clean up old documentation source files and build artifacts.
    Start fresh so old files don't interfere with new generation.
    """
    # Make sure source directory exists
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    # Remove old build artifacts
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)

    # Delete all .rst files in source (will be regenerated)
    for rst in SOURCE_DIR.glob("*.rst"):
        rst.unlink()


def write_conf():
    """
    Generate conf.py - the Sphinx configuration file.
    This tells Sphinx how to build the documentation and which extensions to use.
    """
    conf = DOCS_DIR / "source" / "conf.py"

    conf.write_text(
        f"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Make sure we can import the package we're documenting
sys.path.insert(0, str(PROJECT_ROOT))

project = "{PROJECT_NAME}"
author = "{AUTHOR}"

# Extensions we're using:
# autodoc: automatically pull docstrings from source code
# autosummary: generate summary pages for modules
# napoleon: support Google-style docstrings (nicer than reStructuredText format)
# viewcode: add links to source code in the documentation
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# Use the Read the Docs theme (looks professional)
html_theme = "sphinx_rtd_theme"

# Auto-generate summary pages from module docstrings
autosummary_generate = True
autosummary_imported_members = False  # Don't document things just imported from elsewhere

# How to handle type hints in docstrings
autodoc_typehints = "description"
# Inherit docstrings from parent classes if child doesn't have them
autodoc_inherit_docstrings = True

# Default options for autodoc (what to include, what to skip)
autodoc_default_options = {{
    "members": True,  # Include class members
    "undoc-members": False,  # Skip members without docstrings
    "show-inheritance": False,  # Don't show parent classes
    "imported-members": False,  # Skip imported members
}}

# Patterns to exclude from documentation
exclude_patterns = [
    "_build",
]

# Don't be super strict about missing references (less error spam)
nitpicky = False

# Suppress specific warnings that are noisy but not important
suppress_warnings = [
    "ref.python",  # Python reference warnings
    "ref.ref",  # Generic reference warnings
    "image.not_readable",  # Image loading warnings
]

# List of external modules to skip (we don't want to document third-party code)
exclude_external_modules = {{
    "pathlib",
    "collections",
    "typing",
    "torch",
    "numpy",
    "timm",
    "contextlib"
}}

def skip_member(app, what, name, obj, skip, options):
    \"\"\"Hook that tells autodoc to skip third-party modules\"\"\"
    module = getattr(obj, "__module__", "")

    if module:
        # Get the root module name (e.g., "torch" from "torch.nn.Module")
        root = module.split(".")[0]

        # Skip if it's an external module
        if root in exclude_external_modules:
            return True

    return skip

def setup(app):
    \"\"\"Register our skip_member hook with Sphinx\"\"\"
    app.connect("autodoc-skip-member", skip_member)
""".strip()
        + "\n"
    )


def generate_api():
    """
    Run sphinx.ext.apidoc to auto-generate .rst files from docstrings.
    This scans the package and creates documentation source files.
    """
    run(
        [
            sys.executable,
            "-m",
            "sphinx.ext.apidoc",
            "--force",  # Overwrite existing files
            "--module-first",  # Put module name first in file (looks better)
            "--separate",  # Create separate .rst file for each module
            "-o",
            str(SOURCE_DIR),  # Output directory
            str(PACKAGE_DIR),  # Package to scan
        ],
        cwd=PROJECT_ROOT,
    )


def remove_unwanted_files():
    """
    Delete .rst files we don't want in the final documentation.
    modules.rst is an index of all modules, but we provide our own index.
    """
    for fname in ("modules.rst",):
        p = SOURCE_DIR / fname
        if p.exists():
            p.unlink()


def patch_rst():
    """
    Modify the auto-generated .rst files to add extra Sphinx directives.
    Specifically, adds :no-index: to automodule directives
    (prevents duplicate entries in the index).
    """
    for rst in SOURCE_DIR.glob("*.rst"):
        if rst.name == "index.rst":
            continue  # Don't modify the index file

        lines = rst.read_text().splitlines()

        new_lines = []

        for line in lines:
            new_lines.append(line)

            # If this line starts an automodule directive, add :no-index: after it
            if line.strip().startswith(".. automodule::"):
                new_lines.append("   :no-index:")

        rst.write_text("\n".join(new_lines) + "\n")


def write_index():
    """
    Create the main index.rst file that ties everything together.
    This is the landing page for the documentation.
    """
    (DOCS_DIR / "source" / "index.rst").write_text(
        f"""
{PROJECT_NAME}
{"=" * len(PROJECT_NAME)}

.. toctree::
   :maxdepth: 3

   {PACKAGE_NAME}
""".strip()
        + "\n"
    )


def main():
    """Main entry point - orchestrates the documentation generation pipeline"""
    # Step 1: Make sure all directories have __init__.py
    ensure_package()

    # Step 2: Clean up old files
    clean()

    # Step 3: Write Sphinx configuration
    write_conf()

    # Step 4: Auto-generate .rst files from docstrings
    generate_api()

    # Step 5: Remove files we don't want
    remove_unwanted_files()

    # Step 6: Patch .rst files to add extra directives
    patch_rst()

    # Step 7: Create the main index
    write_index()

    print("DOC SOURCES UPDATED")


if __name__ == "__main__":
    main()
