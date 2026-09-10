from __future__ import annotations

import argparse
import ast
import difflib
import fnmatch
import hashlib
import json
import shutil
import sys
import tokenize
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

# Default directory to scan for undocumented code
DEFAULT_ROOT = Path(".")

# Where to save the report of what was inserted
DEFAULT_REPORT = Path("../output/documentation_audit_results/docstring_insertions.json")

# Configuration file for specifying which files/symbols to skip
DEFAULT_CONFIG = Path("docstring_config.json")


# Directories that usually don't contain package code (safe to skip)
IGNORED_DIRECTORIES = {
    ".git",
    ".github",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "docs",
    "site-packages",
    "venv",
}


# Built-in files to ignore (if any added by users to config)
BUILTIN_IGNORED_FILES: set[str] = set()


# These special methods are often documented manually, so we skip them by default
OPTIONAL_UNDOCUMENTED_METHODS = {
    "__repr__",
    "__str__",
}


# Class attributes that are auto-generated (safe to skip)
IGNORED_CLASS_ATTRIBUTES = {
    "__annotations__",
    "__dataclass_fields__",
    "__match_args__",
    "__slots__",
}


# Placeholder text for template docstrings
FUNCTION_SUMMARY = "Document this function."
CLASS_SUMMARY = "Document this class."
PLACEHOLDER_DESCRIPTION = "Description not yet provided."

# Number of spaces for indentation
INDENT_WIDTH = 4


# Type alias: a function is either sync (FunctionDef) or async (AsyncFunctionDef)
FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef
# Type alias: anything we might want to document (class or function)
DocumentableNode = ast.ClassDef | FunctionNode


@dataclass(frozen=True)
class SourceEdit:
    """Represents a single text replacement in the source code"""

    start: int  # Character offset where edit starts
    end: int  # Character offset where edit ends
    text: str  # Text to insert (replacement)
    qualified_name: (
        str  # Full name of the symbol being edited (e.g., "MyClass.my_method")
    )
    line: int  # Line number where symbol is defined
    symbol_type: str  # "class", "function", or "async function"


@dataclass(frozen=True)
class PlannedInsertion:
    """Records a docstring that we're planning to insert"""

    file: str  # Path to the file
    qualified_name: str  # Full name of the symbol (e.g., "MyClass.my_method")
    line: int  # Line number where symbol is defined
    symbol_type: str  # "class", "function", or "async function"
    parameters: list[str]  # Parameter names we documented
    attributes: list[str]  # Attribute names we documented
    has_returns: bool  # Whether this function/method returns something
    has_yields: bool  # Whether this function/method is a generator
    raises: list[str]  # Exception types that might be raised
    warnings: list[str]  # Warning types that might be raised


@dataclass(frozen=True)
class SkippedSymbol:
    """Records a symbol we skipped and why"""

    file: str  # Path to the file
    qualified_name: str  # Full name of the symbol
    line: int  # Line number where symbol is defined
    symbol_type: str  # "class", "function", or "async function"
    reason: str  # Why we skipped it


@dataclass
class RunReport:
    """Summary of a full run of the docstring inserter"""

    root: str  # Root directory/file we scanned
    mode: str  # "apply" or "dry-run"
    configuration: str  # Path to config file used
    files_scanned: int = 0  # How many Python files we looked at
    files_changed: int = 0  # How many we actually modified
    inserted: list[PlannedInsertion] = field(
        default_factory=list
    )  # All insertions made
    skipped: list[SkippedSymbol] = field(default_factory=list)  # All skipped symbols
    errors: list[dict[str, Any]] = field(
        default_factory=list
    )  # Any errors that occurred


@dataclass
class DocstringConfig:
    """Configuration for controlling which symbols to skip"""

    ignored_files: set[str] = field(default_factory=set)  # Exact filenames to skip
    ignored_file_patterns: list[str] = field(
        default_factory=list
    )  # Glob patterns for files (e.g., "*.pb2.py")
    ignored_symbols: dict[str, set[str]] = field(
        default_factory=dict
    )  # Exact symbol names to skip per file
    ignored_symbol_patterns: dict[str, list[str]] = field(
        default_factory=dict
    )  # Glob patterns for symbols per file


class ScopedFlowVisitor(ast.NodeVisitor):
    """
    Walks through a function's AST to find:
    - Return statements (for detecting "has_returns")
    - Yield statements (for detecting generators)
    - Raise statements (for detecting exceptions)
    - Assert statements (exceptions from assertions)
    - Warning calls (for detecting warnings)

    Only looks at the function body, not nested functions or classes.
    """

    def __init__(self, root: FunctionNode) -> None:
        self.root = root  # The function we're analyzing
        self.returns: list[ast.Return] = []
        self.yields: list[ast.Yield | ast.YieldFrom] = []
        self.raises: list[ast.Raise] = []
        self.asserts: list[ast.Assert] = []
        self.warning_calls: list[ast.Call] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Only descend into the root function, skip nested functions"""
        if node is self.root:
            self.generic_visit(node)

    def visit_AsyncFunctionDef(
        self,
        node: ast.AsyncFunctionDef,
    ) -> None:
        """Only descend into the root function, skip nested async functions"""
        if node is self.root:
            self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Skip nested classes"""
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """Skip lambda expressions"""
        return

    def visit_Return(self, node: ast.Return) -> None:
        self.returns.append(node)
        self.generic_visit(node)

    def visit_Yield(self, node: ast.Yield) -> None:
        self.yields.append(node)
        self.generic_visit(node)

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        self.yields.append(node)
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        self.raises.append(node)
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        self.asserts.append(node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Collect calls to warnings.warn"""
        if dotted_name(node.func) in {
            "warnings.warn",
            "warn",
        }:
            self.warning_calls.append(node)

        self.generic_visit(node)


def normalize_path(path: str | Path) -> str:
    """Convert paths to forward slashes (for consistent config file matching)"""
    return Path(path).as_posix()


def read_string_list(
    payload: dict[str, Any],
    key: str,
    config_path: Path,
) -> list:
    """
    Extract a list of strings from JSON config.
    Validates that the key exists and contains only strings.
    """
    values = payload.get(key, [])

    if not isinstance(values, list):
        raise ValueError(f"{key!r} must be a JSON list in {config_path}.")

    if not all(isinstance(value, str) for value in values):
        raise ValueError(f"Every value in {key!r} must be a string.")

    return values


def read_symbol_mapping(
    payload: dict[str, Any],
    key: str,
    config_path: Path,
) -> dict[str, list[str]]:
    """
    Extract a mapping of filename -> list of symbols from JSON config.
    Example: {"ignored_symbols": {"myfile.py": ["MyClass", "my_function"]}}
    """
    mapping = payload.get(key, {})

    if not isinstance(mapping, dict):
        raise ValueError(f"{key!r} must be a JSON object in {config_path}.")

    result: dict[str, list[str]] = {}

    for file_name, symbols in mapping.items():
        if not isinstance(file_name, str):
            raise ValueError(f"Every key in {key!r} must be a string.")

        if not isinstance(symbols, list):
            raise ValueError(f"{key!r}[{file_name!r}] must be a JSON list.")

        if not all(isinstance(symbol, str) for symbol in symbols):
            raise ValueError(f"Every value in {key!r}[{file_name!r}] must be a string.")

        # Normalize paths for consistent matching
        normalized_file = "*" if file_name == "*" else normalize_path(file_name)

        result[normalized_file] = symbols

    return result


def load_docstring_config(path: Path) -> DocstringConfig:
    """
    Load configuration from a JSON file.
    Returns default config if file doesn't exist.
    """
    if not path.exists():
        return DocstringConfig()

    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)

    if not isinstance(payload, dict):
        raise ValueError(f"Configuration must contain a JSON object: {path}")

    ignored_symbols = read_symbol_mapping(
        payload,
        "ignored_symbols",
        path,
    )

    ignored_symbol_patterns = read_symbol_mapping(
        payload,
        "ignored_symbol_patterns",
        path,
    )

    return DocstringConfig(
        ignored_files={
            normalize_path(value)
            for value in read_string_list(
                payload,
                "ignored_files",
                path,
            )
        },
        ignored_file_patterns=[
            normalize_path(value)
            for value in read_string_list(
                payload,
                "ignored_file_patterns",
                path,
            )
        ],
        ignored_symbols={
            file_name: set(symbols) for file_name, symbols in ignored_symbols.items()
        },
        ignored_symbol_patterns={
            file_name: list(patterns)
            for file_name, patterns in ignored_symbol_patterns.items()
        },
    )


def ignored_file_reason(
    path: Path,
    config: DocstringConfig,
) -> str | None:
    """
    Check if a file should be skipped based on config.
    Returns the reason string if ignored, None if we should process it.
    """
    normalized = normalize_path(path)

    # Exact match against ignored_files list
    if normalized in config.ignored_files:
        return "listed in ignored_files"

    # Pattern match against ignored_file_patterns
    for pattern in config.ignored_file_patterns:
        if fnmatch.fnmatchcase(normalized, pattern):
            return f"matches ignored file pattern {pattern!r}"

    return None


def ignored_symbol_reason(
    path: Path,
    qualified_symbol_name: str,
    config: DocstringConfig,
) -> str | None:
    """
    Check if a symbol should be skipped based on config.
    Returns the reason string if ignored, None if we should document it.
    """
    normalized_file = normalize_path(path)

    # Get all explicitly ignored symbols for this file and global ("*")
    explicitly_ignored = {
        *config.ignored_symbols.get("*", set()),
        *config.ignored_symbols.get(
            normalized_file,
            set(),
        ),
    }

    # Check if this symbol is explicitly listed
    if qualified_symbol_name in explicitly_ignored:
        return "listed in ignored_symbols"

    # Get all patterns for this file and global ("*")
    patterns = [
        *config.ignored_symbol_patterns.get(
            "*",
            [],
        ),
        *config.ignored_symbol_patterns.get(
            normalized_file,
            [],
        ),
    ]

    # Check if this symbol matches any pattern
    for pattern in patterns:
        if fnmatch.fnmatchcase(
            qualified_symbol_name,
            pattern,
        ):
            return f"matches ignored symbol pattern {pattern!r}"

    return None


def is_builtin_ignored(path: Path) -> bool:
    """Check if a path should be ignored based on built-in rules"""
    # Exact filename match
    if path.name in BUILTIN_IGNORED_FILES:
        return True

    # Check if any path component is in ignored directories
    return any(part in IGNORED_DIRECTORIES for part in path.parts)


def python_files(root: Path) -> list:
    """
    Find all Python files under a root directory or return the file if it's a .py file.
    Filters out files in ignored directories.
    """
    if root.is_file():
        if root.suffix == ".py" and not is_builtin_ignored(root):
            return [root]

        return []

    return sorted(
        path
        for path in root.rglob("*.py")
        if (path.is_file() and not is_builtin_ignored(path))
    )


def qualified_name(
    stack: Sequence[str],
    name: str,
) -> str:
    """
    Build a qualified name from a stack of class/function names.
    Example: stack=["MyClass"], name="my_method" -> "MyClass.my_method"
    """
    if stack:
        return ".".join([*stack, name])

    return name


def dotted_name(node: ast.AST) -> str:
    """
    Extract the dotted name from an AST node.
    Example: ast.Attribute(value=Name("warnings"), attr="warn") -> "warnings.warn"
    """
    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Attribute):
        left = dotted_name(node.value)

        if left:
            return f"{left}.{node.attr}"

        return node.attr

    if isinstance(node, ast.Call):
        return dotted_name(node.func)

    if isinstance(node, ast.Subscript):
        return dotted_name(node.value)

    return ""


def annotation_text(
    node: ast.expr | None,
    fallback: str = "Any",
) -> str:
    """
    Convert a type annotation AST node to readable text.
    Example: ast.Name("int") -> "int"
    If annotation is missing, use fallback ("Any" by default).
    """
    if node is None:
        return fallback

    try:
        return ast.unparse(node)
    except (AttributeError, ValueError):
        return fallback


def has_overload_decorator(
    node: FunctionNode,
) -> bool:
    """Check if a function has @overload decorator (type stub)"""
    return any(
        dotted_name(decorator).endswith("overload") for decorator in node.decorator_list
    )


def is_dataclass(node: ast.ClassDef) -> bool:
    """Check if a class is decorated with @dataclass"""
    return any(
        dotted_name(decorator).split(".")[-1] == "dataclass"
        for decorator in node.decorator_list
    )


def exception_name(
    expression: ast.expr | None,
) -> str | None:
    """
    Extract the exception class name from an expression.
    Example: ast.Call(func=Name("ValueError"), args=[...]) -> "ValueError"
    """
    if expression is None:
        return None

    if isinstance(expression, ast.Call):
        return exception_name(expression.func)

    if isinstance(expression, ast.Name):
        return expression.id

    if isinstance(expression, ast.Attribute):
        return dotted_name(expression)

    return None


def warning_name(call: ast.Call) -> str:
    """
    Extract the warning category from a warnings.warn() call.
    Example: warnings.warn("msg", DeprecationWarning) -> "DeprecationWarning"
    Defaults to "UserWarning" if not specified.
    """
    # warnings.warn(msg, category) - positional argument
    if len(call.args) >= 2:
        category = exception_name(call.args[1])

        if category:
            return category

    # warnings.warn(msg, category=SomeWarning) - keyword argument
    for keyword in call.keywords:
        if keyword.arg != "category":
            continue

        category = exception_name(keyword.value)

        if category:
            return category

    return "UserWarning"


def function_parameters(
    node: FunctionNode,
) -> list[tuple[str, ast.arg]]:
    """
    Extract all parameters from a function, excluding 'self' and 'cls'.
    Returns list of (param_name, ast.arg) tuples.
    Includes *args, keyword-only args, and **kwargs.
    """
    parameters: list[tuple[str, ast.arg]] = []

    # Positional-only and regular positional arguments
    for argument in [
        *node.args.posonlyargs,
        *node.args.args,
    ]:
        if argument.arg not in {
            "self",
            "cls",
        }:
            parameters.append((argument.arg, argument))

    # *args (vararg)
    if node.args.vararg is not None:
        parameters.append(
            (
                f"*{node.args.vararg.arg}",
                node.args.vararg,
            )
        )

    # Keyword-only arguments (after * in signature)
    for argument in node.args.kwonlyargs:
        parameters.append((argument.arg, argument))

    # **kwargs (kwarg)
    if node.args.kwarg is not None:
        parameters.append(
            (
                f"**{node.args.kwarg.arg}",
                node.args.kwarg,
            )
        )

    return parameters


def class_initializer(
    node: ast.ClassDef,
) -> FunctionNode | None:
    """Find the __init__ method of a class, if it exists"""
    for statement in node.body:
        if (
            isinstance(
                statement,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            )
            and statement.name == "__init__"
        ):
            return statement

    return None


def is_classvar(
    annotation: ast.expr | None,
) -> bool:
    """Check if a type annotation is typing.ClassVar (class variable, not instance attribute)"""
    if annotation is None:
        return False

    return "ClassVar" in annotation_text(
        annotation,
        fallback="",
    )


def dataclass_parameters(
    node: ast.ClassDef,
) -> list[tuple[str, str]]:
    """
    Extract parameter names and types from a dataclass.
    For dataclasses, parameters are the annotated fields (not in __init__).
    """
    parameters: list[tuple[str, str]] = []

    for statement in node.body:
        if not isinstance(
            statement,
            ast.AnnAssign,
        ):
            continue

        if not isinstance(
            statement.target,
            ast.Name,
        ):
            continue

        # Skip class variables
        if is_classvar(statement.annotation):
            continue

        name = statement.target.id

        # Skip private attributes
        if name.startswith("_"):
            continue

        parameters.append(
            (
                name,
                annotation_text(statement.annotation),
            )
        )

    return parameters


def initializer_parameters(
    node: ast.ClassDef,
) -> list[tuple[str, str]]:
    """
    Extract parameter names and types from a class's __init__ method.
    """
    initializer = class_initializer(node)

    if initializer is None:
        return []

    return [
        (
            name,
            annotation_text(argument.annotation),
        )
        for name, argument in function_parameters(initializer)
    ]


def class_parameters(
    node: ast.ClassDef,
) -> list[tuple[str, str]]:
    """
    Get the parameters that should be documented for a class.
    For dataclasses, use field annotations.
    For regular classes, use __init__ parameters.
    """
    if is_dataclass(node):
        return dataclass_parameters(node)

    return initializer_parameters(node)


def class_attributes(
    node: ast.ClassDef,
) -> list[tuple[str, str]]:
    """
    Extract instance attributes (annotated fields) from a class.
    Only for non-dataclasses (dataclass fields are documented as parameters).
    """
    if is_dataclass(node):
        return []

    attributes: list[tuple[str, str]] = []

    for statement in node.body:
        if not isinstance(
            statement,
            ast.AnnAssign,
        ):
            continue

        if not isinstance(
            statement.target,
            ast.Name,
        ):
            continue

        # Skip class variables
        if is_classvar(statement.annotation):
            continue

        name = statement.target.id

        # Skip special/internal attributes
        if name in IGNORED_CLASS_ATTRIBUTES:
            continue

        attributes.append(
            (
                name,
                annotation_text(statement.annotation),
            )
        )

    return attributes


def meaningful_returns(
    visitor: ScopedFlowVisitor,
) -> list[ast.Return]:
    """
    Filter return statements to meaningful ones (not "return None").
    We only document a function as having a return if it returns something useful.
    """
    result: list[ast.Return] = []

    for statement in visitor.returns:
        value = statement.value

        # "return" with no value is just "return None" - skip it
        if value is None:
            continue

        # "return None" explicitly - skip it
        if isinstance(value, ast.Constant) and value.value is None:
            continue

        result.append(statement)

    return result


def yielded_type(
    annotation: ast.expr | None,
) -> str:
    """
    Extract the yielded type from a generator's return annotation.
    Example: Generator[int, None, None] -> "int"
    Example: Iterator[str] -> "str"
    """
    if annotation is None:
        return "Any"

    # Check if it's a subscripted type like Iterator[int]
    if isinstance(annotation, ast.Subscript):
        container = dotted_name(annotation.value).split(".")[-1]

        # For Iterator/Iterable, the yielded type is the subscript
        if container in {
            "Iterator",
            "Iterable",
            "AsyncIterator",
            "AsyncIterable",
        }:
            return annotation_text(annotation.slice)

        # For Generator[YieldType, SendType, ReturnType], take first element
        if container == "Generator":
            slice_node = annotation.slice

            if (
                isinstance(
                    slice_node,
                    ast.Tuple,
                )
                and slice_node.elts
            ):
                return annotation_text(slice_node.elts[0])

    return annotation_text(annotation)


def analyze_flow(
    node: FunctionNode,
) -> ScopedFlowVisitor:
    """
    Analyze a function's control flow to find returns, yields, raises, warnings, etc.
    """
    visitor = ScopedFlowVisitor(node)
    visitor.visit(node)

    return visitor


def build_function_docstring(
    node: FunctionNode,
) -> tuple[str, dict[str, Any]]:
    """
    Generate a NumPy-style docstring template for a function.
    Analyzes the function to determine what sections are needed.
    """
    flow = analyze_flow(node)
    parameters = function_parameters(node)
    returns = meaningful_returns(flow)
    yields = flow.yields

    # Collect exception types that might be raised
    raised = {
        name
        for statement in flow.raises
        if (name := exception_name(statement.exc)) is not None
    }

    # Asserts raise AssertionError
    if flow.asserts:
        raised.add("AssertionError")

    # Collect warning types that might be issued
    warnings = {warning_name(call) for call in flow.warning_calls}

    lines = [FUNCTION_SUMMARY]

    # Add Parameters section if function has parameters
    if parameters:
        lines.extend(
            [
                "",
                "Parameters",
                "----------",
            ]
        )

        for name, argument in parameters:
            lines.append(f"{name} : {annotation_text(argument.annotation)}")
            lines.append(f"    {PLACEHOLDER_DESCRIPTION}")

    has_yields = bool(yields)

    # Determine if this function has meaningful return values
    has_returns = (
        bool(returns)
        and not has_yields
        and node.name
        not in {
            "__init__",
            "__post_init__",
        }
    )

    # Add Yields section if it's a generator
    if has_yields:
        lines.extend(
            [
                "",
                "Yields",
                "------",
                yielded_type(node.returns),
                f"    {PLACEHOLDER_DESCRIPTION}",
            ]
        )

    # Add Returns section if it returns something
    elif has_returns:
        lines.extend(
            [
                "",
                "Returns",
                "-------",
                annotation_text(node.returns),
                f"    {PLACEHOLDER_DESCRIPTION}",
            ]
        )

    # Add Raises section if it might raise exceptions
    if raised:
        lines.extend(
            [
                "",
                "Raises",
                "------",
            ]
        )

        for name in sorted(raised):
            lines.append(name)
            lines.append(f"    {PLACEHOLDER_DESCRIPTION}")

    # Add Warns section if it might issue warnings
    if warnings:
        lines.extend(
            [
                "",
                "Warns",
                "-----",
            ]
        )

        for name in sorted(warnings):
            lines.append(name)
            lines.append(f"    {PLACEHOLDER_DESCRIPTION}")

    # Collect metadata about what we documented
    metadata = {
        "parameters": [name for name, _ in parameters],
        "attributes": [],
        "has_returns": has_returns,
        "has_yields": has_yields,
        "raises": sorted(raised),
        "warnings": sorted(warnings),
    }

    return "\n".join(lines), metadata


def build_class_docstring(
    node: ast.ClassDef,
) -> tuple[str, dict[str, Any]]:
    """
    Generate a NumPy-style docstring template for a class.
    Documents parameters (__init__ args or dataclass fields) and attributes.
    """
    parameters = class_parameters(node)
    attributes = class_attributes(node)

    lines = [CLASS_SUMMARY]

    # Add Parameters section
    if parameters:
        lines.extend(
            [
                "",
                "Parameters",
                "----------",
            ]
        )

        for name, data_type in parameters:
            lines.append(f"{name} : {data_type}")
            lines.append(f"    {PLACEHOLDER_DESCRIPTION}")

    # Add Attributes section
    if attributes:
        lines.extend(
            [
                "",
                "Attributes",
                "----------",
            ]
        )

        for name, data_type in attributes:
            lines.append(f"{name} : {data_type}")
            lines.append(f"    {PLACEHOLDER_DESCRIPTION}")

    # Collect metadata about what we documented
    metadata = {
        "parameters": [name for name, _ in parameters],
        "attributes": [name for name, _ in attributes],
        "has_returns": False,
        "has_yields": False,
        "raises": [],
        "warnings": [],
    }

    return "\n".join(lines), metadata


def build_docstring(
    node: DocumentableNode,
) -> tuple[str, dict[str, Any]]:
    """Generate a docstring template for either a class or function"""
    if isinstance(node, ast.ClassDef):
        return build_class_docstring(node)

    return build_function_docstring(node)


def symbol_type(
    node: DocumentableNode,
) -> str:
    """Determine what kind of symbol this is (class, function, or async function)"""
    if isinstance(node, ast.ClassDef):
        return "class"

    if isinstance(
        node,
        ast.AsyncFunctionDef,
    ):
        return "async function"

    return "function"


def leading_whitespace(line: str) -> str:
    """Extract the leading whitespace from a line"""
    return line[: len(line) - len(line.lstrip())]


def node_start_lineno(
    node: ast.stmt,
) -> int:
    """
    Get the actual start line number of a node, accounting for decorators.
    (Decorators come before the definition, so we need the decorator's line)
    """
    decorators = getattr(
        node,
        "decorator_list",
        [],
    )

    if decorators:
        return min(
            node.lineno,
            *(decorator.lineno for decorator in decorators),
        )

    return node.lineno


def body_indent(
    node: DocumentableNode,
    source_lines: Sequence[str],
) -> str:
    """
    Determine the correct indentation for a docstring inside this symbol.
    Looks at the indentation of the first statement in the body.
    """
    if node.body:
        first_statement = node.body[0]
        first_line_number = node_start_lineno(first_statement)

        line = source_lines[first_line_number - 1]

        detected = leading_whitespace(line)

        # If detected indent is deeper than the definition, use it
        if len(detected.expandtabs()) > node.col_offset:
            return detected

    # Fall back to definition line indent + one indentation level
    definition_line = source_lines[node.lineno - 1]

    return leading_whitespace(definition_line) + " " * INDENT_WIDTH


def render_docstring(
    doc: str,
    indent: str,
    newline: str,
) -> str:
    """
    Format a docstring with proper indentation and triple quotes.
    Returns the complete docstring including the surrounding quotes and newlines.
    """
    if not doc:
        raise ValueError("Cannot render an empty docstring.")

    if '"""' in doc:
        raise ValueError("Generated documentation unexpectedly contains triple quotes.")

    lines = doc.splitlines()

    rendered = [
        f'{indent}"""',
    ]

    for line in lines:
        if line:
            rendered.append(f"{indent}{line}")
        else:
            # Blank lines in docstring (with just indentation)
            rendered.append(indent)

    rendered.append(f'{indent}"""')

    return newline.join(rendered) + newline


def line_start_offsets(
    source: str,
) -> list:
    """
    Calculate byte offsets for the start of each line.
    Useful for converting line numbers to character offsets for edits.
    """
    offsets = [0]

    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))

    return offsets


def docstring_insertion_range(
    node: DocumentableNode,
    source_lines: Sequence[str],
    offsets: Sequence[int],
) -> tuple[int, int]:
    """
    Find the byte range where we should insert a docstring.
    Returns (start_offset, end_offset) for the insertion point.
    The range is from after the function/class definition to before the first statement.
    """
    if not node.body:
        raise ValueError(f"Symbol {node.name!r} has no body.")

    first_statement = node.body[0]

    first_statement_line = node_start_lineno(first_statement)

    first_statement_index = first_statement_line - 1

    # Find where to insert (skip blank lines after the definition)
    insertion_index = first_statement_index

    while (
        insertion_index > node.lineno and not source_lines[insertion_index - 1].strip()
    ):
        insertion_index -= 1

    # Get byte offsets
    start = offsets[insertion_index]
    end = offsets[first_statement_index]

    return start, end


def iter_symbols(
    body: Sequence[ast.stmt],
    stack: Sequence[str] = (),
) -> Iterable[tuple[DocumentableNode, str]]:
    """
    Walk through all classes and functions in a body, yielding them with their qualified names.
    Also recurses into nested classes and functions.
    """
    for node in body:
        if isinstance(node, ast.ClassDef):
            qname = qualified_name(
                stack,
                node.name,
            )

            yield node, qname

            # Recurse into class body
            yield from iter_symbols(
                node.body,
                [*stack, node.name],
            )

        elif isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            qname = qualified_name(
                stack,
                node.name,
            )

            yield node, qname

            # Recurse into function body (for nested functions)
            yield from iter_symbols(
                node.body,
                [*stack, node.name],
            )


def apply_edits(
    source: str,
    edits: Sequence[SourceEdit],
) -> str:
    """
    Apply multiple edits to source code.
    Processes edits in reverse order to avoid messing up offsets.
    """
    updated = source
    previous_start = len(source) + 1

    # Process edits in reverse order (from end of file backwards)
    for edit in sorted(
        edits,
        key=lambda item: (
            item.start,
            item.end,
        ),
        reverse=True,
    ):
        # Check for overlapping edits (would corrupt the source)
        if edit.end > previous_start:
            raise ValueError(
                f"Overlapping source edits detected for {edit.qualified_name!r}."
            )

        # Apply this edit
        updated = updated[: edit.start] + edit.text + updated[edit.end :]

        previous_start = edit.start

    return updated


def existing_docstring_digest(
    tree: ast.Module,
) -> dict[str, str]:
    """
    Create a hash of every existing docstring in the module.
    Used later to verify we didn't accidentally modify existing docstrings.
    """
    result: dict[str, str] = {}

    for node, qname in iter_symbols(tree.body):
        doc = ast.get_docstring(
            node,
            clean=False,
        )

        if doc is None:
            continue

        key = f"{symbol_type(node)}::{qname}"

        # Hash the docstring so we can detect changes
        result[key] = hashlib.sha256(doc.encode("utf-8")).hexdigest()

    return result


def verify_existing_docstrings(
    before: dict[str, str],
    after_tree: ast.Module,
) -> None:
    """
    Verify that existing docstrings weren't modified by our edits.
    Raises RuntimeError if any docstring changed or disappeared.
    """
    after = existing_docstring_digest(after_tree)

    for key, expected_digest in before.items():
        observed_digest = after.get(key)

        if observed_digest != expected_digest:
            raise RuntimeError(
                f"Existing docstring changed or disappeared for {key!r}."
            )


def detect_newline(source: str) -> str:
    """
    Detect whether the file uses CRLF (Windows) or LF (Unix) line endings.
    Returns either "\r\n" or "\n".
    """
    if "\r\n" in source:
        return "\r\n"

    return "\n"


def make_diff(
    path: Path,
    before: str,
    after: str,
) -> str:
    """
    Generate a unified diff showing the changes.
    Useful for reviewing what will be inserted.
    """
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=str(path),
            tofile=str(path),
        )
    )


def atomic_write(
    path: Path,
    text: str,
    encoding: str,
) -> None:
    """
    Write text to a file atomically (write to temp file, then rename).
    Prevents data corruption if the script crashes mid-write.
    """
    temporary = path.with_name(f".{path.name}.docstrings.tmp")

    try:
        temporary.write_text(
            text,
            encoding=encoding,
            newline="",  # Preserve original line endings
        )

        # Atomic rename
        temporary.replace(path)

    finally:
        # Clean up temp file if anything went wrong
        if temporary.exists():
            temporary.unlink()


def parse_source(
    path: Path,
) -> tuple[str, str, ast.Module]:
    """
    Read and parse a Python source file.
    Returns (source_text, encoding, ast_tree).
    Uses tokenize.open to detect encoding from the file.
    """
    with tokenize.open(path) as stream:
        encoding = stream.encoding
        source = stream.read()

    tree = ast.parse(
        source,
        filename=str(path),
        type_comments=True,
    )

    return source, encoding, tree


def should_skip(
    node: DocumentableNode,
    *,
    include_optional_methods: bool,
) -> str | None:
    """
    Determine if a symbol should be skipped based on its properties.
    Returns a reason string if it should be skipped, None otherwise.
    """
    if isinstance(
        node,
        (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
        ),
    ):
        # Skip @overload declarations (just type stubs)
        if has_overload_decorator(node):
            return "overload declaration"

        # Skip special methods like __repr__ unless requested
        if not include_optional_methods and node.name in OPTIONAL_UNDOCUMENTED_METHODS:
            return "optional special method"

    return None


def plan_file(
    path: Path,
    source: str,
    tree: ast.Module,
    *,
    include_optional_methods: bool,
    config: DocstringConfig,
) -> tuple[
    str,
    list[PlannedInsertion],
    list[SkippedSymbol],
]:
    """
    Plan all docstring insertions for a file (don't actually modify yet).
    Returns (modified_source, list_of_insertions, list_of_skipped_symbols).
    """
    source_lines = source.splitlines()
    offsets = line_start_offsets(source)
    newline = detect_newline(source)

    # Capture hashes of existing docstrings to verify we don't modify them
    existing_digests = existing_docstring_digest(tree)

    edits: list[SourceEdit] = []
    insertions: list[PlannedInsertion] = []
    skipped: list[SkippedSymbol] = []

    # Process each symbol in the module
    for node, qname in iter_symbols(tree.body):
        existing_docstring = ast.get_docstring(
            node,
            clean=False,
        )

        # Skip symbols that already have docstrings
        if existing_docstring is not None:
            continue

        # Check if symbol is explicitly ignored by config
        ignore_reason = ignored_symbol_reason(
            path,
            qname,
            config,
        )

        if ignore_reason is not None:
            skipped.append(
                SkippedSymbol(
                    file=str(path),
                    qualified_name=qname,
                    line=node.lineno,
                    symbol_type=symbol_type(node),
                    reason=ignore_reason,
                )
            )
            continue

        # Check if symbol should be skipped for other reasons
        skip_reason = should_skip(
            node,
            include_optional_methods=(include_optional_methods),
        )

        if skip_reason is not None:
            skipped.append(
                SkippedSymbol(
                    file=str(path),
                    qualified_name=qname,
                    line=node.lineno,
                    symbol_type=symbol_type(node),
                    reason=skip_reason,
                )
            )
            continue

        # Generate the docstring template
        doc, metadata = build_docstring(node)

        # Determine correct indentation
        indent = body_indent(
            node,
            source_lines,
        )

        # Find where to insert the docstring
        edit_start, edit_end = docstring_insertion_range(
            node,
            source_lines,
            offsets,
        )

        # Create the edit
        edits.append(
            SourceEdit(
                start=edit_start,
                end=edit_end,
                text=render_docstring(
                    doc,
                    indent,
                    newline,
                ),
                qualified_name=qname,
                line=node.lineno,
                symbol_type=symbol_type(node),
            )
        )

        # Record what we're inserting
        insertions.append(
            PlannedInsertion(
                file=str(path),
                qualified_name=qname,
                line=node.lineno,
                symbol_type=symbol_type(node),
                parameters=metadata["parameters"],
                attributes=metadata["attributes"],
                has_returns=metadata["has_returns"],
                has_yields=metadata["has_yields"],
                raises=metadata["raises"],
                warnings=metadata["warnings"],
            )
        )

    # Apply all edits to get the modified source
    updated = apply_edits(
        source,
        edits,
    )

    # If no changes, return early
    if not edits:
        return updated, insertions, skipped

    # Verify the modified source is still valid Python
    updated_tree = ast.parse(
        updated,
        filename=str(path),
        type_comments=True,
    )

    compile(
        updated,
        str(path),
        "exec",
        dont_inherit=True,
    )

    # Verify we didn't accidentally modify existing docstrings
    verify_existing_docstrings(
        existing_digests,
        updated_tree,
    )

    # Verify that our generated docstrings are actually present
    updated_symbols = {
        (
            symbol_type(node),
            qname,
        ): node
        for node, qname in iter_symbols(updated_tree.body)
    }

    for insertion in insertions:
        key = (
            insertion.symbol_type,
            insertion.qualified_name,
        )

        updated_node = updated_symbols.get(key)

        if updated_node is None:
            raise RuntimeError(
                f"Generated symbol disappeared: {insertion.qualified_name}"
            )

        generated_docstring = ast.get_docstring(
            updated_node,
            clean=False,
        )

        if generated_docstring is None:
            raise RuntimeError(
                f"Generated docstring was not recognized for {insertion.qualified_name}"
            )

    return updated, insertions, skipped


def process_file(
    path: Path,
    *,
    apply: bool,
    backup: bool,
    show_diff: bool,
    include_optional_methods: bool,
    config: DocstringConfig,
) -> tuple[
    list[PlannedInsertion],
    list[SkippedSymbol],
    bool,
]:
    """
    Process a single file: parse it, plan changes, optionally apply them.
    Returns (insertions, skipped_symbols, was_changed).
    """
    source, encoding, tree = parse_source(path)

    updated, inserted, skipped = plan_file(
        path,
        source,
        tree,
        include_optional_methods=(include_optional_methods),
        config=config,
    )

    changed = updated != source

    # If no changes needed, we're done
    if not changed:
        return inserted, skipped, False

    # Show diff if requested
    if show_diff:
        print(
            make_diff(
                path,
                source,
                updated,
            ),
            end="",
        )

    # Apply changes if requested
    if apply:
        if backup:
            backup_path = path.with_suffix(path.suffix + ".bak")

            if backup_path.exists():
                print(f"USING EXISTING BACKUP: {backup_path}")
            else:
                # Create backup before modifying
                shutil.copy2(
                    path,
                    backup_path,
                )

        # Write the modified file atomically
        atomic_write(
            path,
            updated,
            encoding,
        )

    return inserted, skipped, True


def write_report(
    report: RunReport,
    path: Path,
) -> None:
    """Save the run report as a JSON file"""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            asdict(report),
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser"""
    parser = argparse.ArgumentParser(
        description=(
            "Insert NumPy-style docstrings into "
            "undocumented classes, functions, and "
            "methods. Existing docstrings are never "
            "modified."
        )
    )

    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=DEFAULT_ROOT,
        help=("Python file or package directory to scan. Default: cccma_ppp"),
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help=("Write validated changes. Without this option, perform a dry run."),
    )

    parser.add_argument(
        "--no-backup",
        action="store_true",
        help=("Do not create .py.bak files when applying changes."),
    )

    parser.add_argument(
        "--no-diff",
        action="store_true",
        help="Do not print unified diffs.",
    )

    parser.add_argument(
        "--include-optional-methods",
        action="store_true",
        help=("Also document __repr__ and __str__ when their docstrings are missing."),
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=("JSON configuration containing ignored files and symbols."),
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="Path for the JSON run report.",
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """
    Main entry point. Orchestrates the entire docstring insertion process.
    """
    args = build_parser().parse_args(argv)
    root: Path = args.root

    # Validate the root path exists
    if not root.exists():
        print(
            f"ERROR: Path does not exist: {root}",
            file=sys.stderr,
        )
        return 2

    # Load configuration
    try:
        config = load_docstring_config(args.config)
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        print(
            f"ERROR: Could not load docstring configuration: {exc}",
            file=sys.stderr,
        )
        return 2

    # Find all Python files to process
    files = python_files(root)

    if not files:
        print(
            f"ERROR: No Python files found under {root}",
            file=sys.stderr,
        )
        return 2

    # Initialize the report
    report = RunReport(
        root=str(root),
        mode=("apply" if args.apply else "dry-run"),
        configuration=str(args.config),
        files_scanned=len(files),
    )

    # Process each file
    for path in files:
        file_ignore_reason = ignored_file_reason(
            path,
            config,
        )

        # Skip ignored files
        if file_ignore_reason is not None:
            report.skipped.append(
                SkippedSymbol(
                    file=str(path),
                    qualified_name="<file>",
                    line=1,
                    symbol_type="file",
                    reason=file_ignore_reason,
                )
            )

            print(f"SKIPPED FILE: {path}: {file_ignore_reason}")
            continue

        try:
            inserted, skipped, changed = process_file(
                path,
                apply=args.apply,
                backup=not args.no_backup,
                show_diff=not args.no_diff,
                include_optional_methods=(args.include_optional_methods),
                config=config,
            )

            report.inserted.extend(inserted)
            report.skipped.extend(skipped)

            if changed:
                report.files_changed += 1

        except (
            OSError,
            SyntaxError,
            UnicodeError,
            ValueError,
            RuntimeError,
        ) as exc:
            # Record error in report
            report.errors.append(
                {
                    "file": str(path),
                    "error_type": (type(exc).__name__),
                    "message": str(exc),
                }
            )

            print(
                f"ERROR: {path}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    # Save the report
    write_report(
        report,
        args.report,
    )

    # Calculate summary statistics
    class_count = sum(item.symbol_type == "class" for item in report.inserted)

    function_count = sum(
        item.symbol_type
        in {
            "function",
            "async function",
        }
        for item in report.inserted
    )

    action = "Inserted" if args.apply else "Would insert"

    # Print summary
    print()
    print("=" * 72)
    print("MISSING DOCSTRING INSERTION SUMMARY")
    print("=" * 72)
    print(f"Mode:                 {report.mode}")
    print(f"Root:                 {report.root}")
    print(f"Configuration:        {report.configuration}")
    print(f"Files scanned:        {report.files_scanned}")
    print(f"Files changed:        {report.files_changed}")
    print(f"{action + ':':<22}{len(report.inserted)}")
    print(f"Classes:              {class_count}")
    print(f"Functions/methods:    {function_count}")
    print(f"Skipped:              {len(report.skipped)}")
    print(f"Errors:               {len(report.errors)}")
    print(f"JSON report:          {args.report}")

    # Remind user to use --apply if this was a dry run
    if not args.apply and report.inserted:
        print()
        print("Dry run only. Review the diff, then re-run with --apply.")

    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
