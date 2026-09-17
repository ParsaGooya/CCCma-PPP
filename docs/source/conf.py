import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))

project = "CCCma PPP"
author = "Parsa Gooya"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

html_theme = "sphinx_rtd_theme"

# Better theme styling
html_theme_options = {
    "collapse_navigation": False,  # Keep navigation expanded
    "sticky_navigation": True,
    "navigation_depth": 4,  # Show more levels
    "includehidden": True,
    "titles_only": False,  # Show actual content in sidebar
}

autosummary_generate = True
autosummary_imported_members = False

autodoc_typehints = "description"
autodoc_inherit_docstrings = True

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,  # More useful with hierarchy
    "imported-members": False,
}

exclude_patterns = [
    "_build",
]

nitpicky = False

suppress_warnings = [
    "ref.python",
    "ref.ref",
    "image.not_readable",
]

exclude_external_modules = {
    "pathlib",
    "collections",
    "typing",
    "torch",
    "numpy",
    "timm",
    "contextlib",
}


def skip_member(app, what, name, obj, skip, options):
    module = getattr(obj, "__module__", "")

    if module:
        root = module.split(".")[0]

        if root in exclude_external_modules:
            return True

    return skip


def setup(app):
    app.connect("autodoc-skip-member", skip_member)
