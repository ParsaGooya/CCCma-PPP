import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCS_DIR = PROJECT_ROOT / "docs"
SOURCE_DIR = DOCS_DIR / "source"
BUILD_DIR = DOCS_DIR / "_build"

PACKAGE_NAME = "cccma_ppp"
PACKAGE_DIR = PROJECT_ROOT / PACKAGE_NAME

PROJECT_NAME = "CCCma PPP"
AUTHOR = "Parsa Gooya"


def run(cmd, cwd=None):
    result = subprocess.run(
        [str(x) for x in cmd],
        cwd=cwd,
    )

    if result.returncode:
        raise SystemExit(result.returncode)


def ensure_package():
    for directory in PACKAGE_DIR.rglob("*"):
        if directory.is_dir():
            (directory / "__init__.py").touch(exist_ok=True)


def clean():
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)

    for rst in SOURCE_DIR.glob("*.rst"):
        rst.unlink()


def write_conf():
    conf = SOURCE_DIR / "conf.py"

    conf.write_text(
        f"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))

project = "{PROJECT_NAME}"
author = "{AUTHOR}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

html_theme = "sphinx_rtd_theme"

autosummary_generate = True
autosummary_imported_members = False

autodoc_typehints = "description"
autodoc_inherit_docstrings = True

autodoc_default_options = {{
    "members": True,
    "undoc-members": False,
    "show-inheritance": False,
    "imported-members": False,
}}

exclude_patterns = [
    "_build",
]

nitpicky = False

suppress_warnings = [
    "ref.python",
    "ref.ref",
    "image.not_readable",
]

exclude_external_modules = {{
    "pathlib",
    "collections",
    "typing",
    "torch",
    "numpy",
    "timm",
}}

def skip_member(app, what, name, obj, skip, options):
    module = getattr(obj, "__module__", "")

    if module:
        root = module.split(".")[0]

        if root in exclude_external_modules:
            return True

    return skip

def setup(app):
    app.connect("autodoc-skip-member", skip_member)
""".strip()
        + "\n"
    )


def generate_api():
    run(
        [
            sys.executable,
            "-m",
            "sphinx.ext.apidoc",
            "--force",
            "--module-first",
            "--separate",
            "-o",
            str(SOURCE_DIR),
            str(PACKAGE_DIR),
        ],
        cwd=PROJECT_ROOT,
    )


def remove_unwanted_files():
    for fname in ("modules.rst",):
        p = SOURCE_DIR / fname
        if p.exists():
            p.unlink()


def patch_rst():
    for rst in SOURCE_DIR.glob("*.rst"):
        if rst.name == "index.rst":
            continue

        lines = rst.read_text().splitlines()

        new_lines = []

        for line in lines:
            new_lines.append(line)

            if line.strip().startswith(".. automodule::"):
                new_lines.append("   :no-index:")

        rst.write_text("\n".join(new_lines) + "\n")


def write_index():
    (SOURCE_DIR / "index.rst").write_text(
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
    ensure_package()
    clean()
    write_conf()
    generate_api()
    remove_unwanted_files()
    patch_rst()
    write_index()

    print("DOC SOURCES UPDATED")


if __name__ == "__main__":
    main()
