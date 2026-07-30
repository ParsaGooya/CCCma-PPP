from __future__ import annotations

import ast
import inspect
import csv
import json
import re
import sys
import tokenize
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path("cccma_ppp")
OUTPUT_DIR = Path("output/documentation_audit_results")
OUTPUT_TEXT = OUTPUT_DIR / Path("audit.txt")
OUTPUT_ISSUES_CSV = OUTPUT_DIR / Path("doc_audit_issues.csv")
OUTPUT_SUMMARY_CSV = OUTPUT_DIR / Path("doc_audit_summary.csv")
OUTPUT_SYMBOLS_CSV = OUTPUT_DIR / Path("doc_audit_symbols.csv")
OUTPUT_JSON = OUTPUT_DIR / Path("doc_audit.json")
OUTPUT_HEATMAP = OUTPUT_DIR / Path("doc_audit_heatmap.png")
OUTPUT_TOTALS = OUTPUT_DIR / Path("doc_audit_totals.png")

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

IGNORED_FILES: set[str] = set()


OPTIONAL_DOCSTRING_METHODS = {
    "__repr__",
    "__str__",
}


OPTIONAL_RETURNS_METHODS: set[str] = set()


IGNORED_CLASS_ATTRIBUTES = {
    "__annotations__",
    "__dataclass_fields__",
    "__match_args__",
    "__slots__",
}


CHECK_PRIVATE_ATTRIBUTES = True


CHECK_PRIVATE_SYMBOLS = True


CHECK_MODULE_DOCSTRINGS = False


CHECK_TYPE_HINTS = True


REQUIRE_EXAMPLES = False


REQUIRE_ABSTRACT_CLASS_NOTES = False


REQUIRE_RETURNS_FOR_EXPLICIT_NONE = False


INCLUDE_ASSERTION_ERRORS = True


CHECK_WARNINGS = True


STRICT_ABSTRACT_METHOD_DOCS = True

SUMMARY_LINE_MAX_LENGTH = 88

STANDARD_SECTIONS = (
    "Parameters",
    "Other Parameters",
    "Returns",
    "Yields",
    "Receives",
    "Raises",
    "Warns",
    "Attributes",
    "Methods",
    "See Also",
    "Notes",
    "References",
    "Examples",
)

ENTRY_SECTIONS = {
    "Parameters",
    "Other Parameters",
    "Returns",
    "Yields",
    "Receives",
    "Raises",
    "Warns",
    "Attributes",
    "Methods",
}

MULTI_NAME_SECTIONS = {
    "Parameters",
    "Other Parameters",
    "Attributes",
}

ISSUE_CHECKS = [
    "syntax_error",
    "missing_module_docstring",
    "missing_docstring",
    "empty_docstring",
    "missing_summary",
    "summary_too_long",
    "summary_not_imperative",
    "summary_punctuation",
    "todo_docs",
    "malformed_section",
    "duplicate_section",
    "section_order",
    "empty_sections",
    "unknown_section",
    "missing_param",
    "extra_param",
    "duplicate_params",
    "missing_varargs",
    "missing_kwargs",
    "missing_param_type",
    "missing_param_description",
    "missing_returns",
    "extra_returns",
    "missing_return_type",
    "missing_return_description",
    "return_arity_mismatch",
    "missing_yields",
    "extra_yields",
    "missing_yield_type",
    "missing_yield_description",
    "returns_and_yields",
    "missing_raise",
    "extra_raise",
    "duplicate_raises",
    "missing_raise_description",
    "bare_raise",
    "dynamic_raise",
    "missing_warns",
    "extra_warns",
    "missing_warn_description",
    "missing_attributes",
    "extra_attributes",
    "duplicate_attributes",
    "missing_attribute_type",
    "missing_attribute_description",
    "redundant_methods_section",
    "undocumented_decorator_behavior",
    "missing_examples",
    "missing_notes",
    "missing_parameter_annotation",
    "missing_return_annotation",
    "documented_type_mismatch",
    "property_returns_missing",
    "constructor_returns_section",
    "private_name_in_public_docs",
    "invalid_backticks",
    "unbalanced_code_fence",
    "trailing_whitespace",
]


@dataclass
class Issue:
    file: str
    line: int
    column: int
    symbol: str
    qualified_name: str
    symbol_type: str
    check: str
    severity: str
    message: str
    details: dict[str, Any]


@dataclass
class SymbolRecord:
    file: str
    line: int
    qualified_name: str
    symbol_type: str
    public: bool
    documented: bool
    issue_count: int = 0
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0


@dataclass
class Section:
    name: str
    body: str
    start_line: int
    end_line: int


@dataclass
class DocEntry:
    names: tuple[str, ...]
    type_text: str | None
    description: str
    line: int


def is_ignored(path: Path) -> bool:
    if path.name in IGNORED_FILES:
        return True
    return any(part in IGNORED_DIRECTORIES for part in path.parts)


def python_files(root: Path) -> list:
    return sorted(
        path for path in root.rglob("*.py") if path.is_file() and not is_ignored(path)
    )


def is_public_name(name: str) -> bool:
    return not name.startswith("_") or (name.startswith("__") and name.endswith("__"))


def clean_docstring(doc: str) -> str:
    return inspect.cleandoc(doc).rstrip()


def first_nonempty_line(doc: str) -> str:
    for line in doc.splitlines():
        if line.strip():
            return line.strip()
    return ""


def node_end_lineno(node: ast.AST) -> int:
    return getattr(node, "end_lineno", getattr(node, "lineno", 0))


def qualified_name(stack: list[str], name: str) -> str:
    return ".".join([*stack, name]) if stack else name


def decorator_name(decorator: ast.expr) -> str:
    if isinstance(decorator, ast.Name):
        return decorator.id
    if isinstance(decorator, ast.Attribute):
        left = decorator_name(decorator.value)
        return f"{left}.{decorator.attr}" if left else decorator.attr
    if isinstance(decorator, ast.Call):
        return decorator_name(decorator.func)
    if isinstance(decorator, ast.Subscript):
        return decorator_name(decorator.value)
    return ""


def annotation_text(node: ast.expr | None) -> str | None:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None


def normalized_type(text: str | None) -> str:
    if not text:
        return ""

    value = text.lower()

    replacements = {
        "typing.": "",
        "numpy.": "np.",
        "pathlib.": "",
        "builtins.": "",
        "collections.abc.": "",
        "nonetype": "none",
    }

    for old, new in replacements.items():
        value = value.replace(old, new)

    value = re.sub(r"\s+or\s+", "|", value)

    optional_match = re.fullmatch(
        r"\s*optional\s*\[(.*)]\s*",
        value,
    )
    if optional_match:
        value = f"{optional_match.group(1)}|none"

    value = value.replace(" ", "")
    value = value.replace("`", "")
    value = value.replace("~", "")

    if "|" in value:
        value = "|".join(sorted(value.split("|")))

    return value


def likely_type_match(annotation: str, documented: str) -> bool:
    ann = normalized_type(annotation)
    doc = normalized_type(documented)

    if not ann or not doc:
        return True

    aliases = {
        "ndarray": {"ndarray", "np.ndarray", "array-like", "arraylike"},
        "np.ndarray": {"ndarray", "np.ndarray", "array-like", "arraylike"},
        "path": {"path", "str", "path|str"},
        "callable": {"callable", "collections.abc.callable"},
        "mapping": {"mapping", "dict"},
        "sequence": {"sequence", "list", "tuple"},
        "iterable": {"iterable", "list", "tuple", "sequence"},
        "tensor": {"tensor", "torch.tensor"},
    }

    if ann == doc or ann in doc or doc in ann:
        return True

    for key, values in aliases.items():
        if key in ann and any(value in doc for value in values):
            return True

    return False


def severity_for(check: str) -> str:
    errors = {
        "syntax_error",
        "missing_docstring",
        "missing_module_docstring",
        "missing_param",
        "extra_param",
        "missing_returns",
        "extra_returns",
        "missing_yields",
        "extra_yields",
        "missing_raise",
        "missing_attributes",
        "malformed_section",
        "empty_sections",
    }
    warnings = {
        "empty_docstring",
        "missing_summary",
        "todo_docs",
        "duplicate_section",
        "section_order",
        "duplicate_params",
        "duplicate_raises",
        "duplicate_attributes",
        "missing_param_type",
        "missing_param_description",
        "missing_return_type",
        "missing_return_description",
        "missing_raise_description",
        "missing_warns",
        "missing_warn_description",
        "missing_attribute_type",
        "missing_attribute_description",
        "missing_parameter_annotation",
        "missing_return_annotation",
        "documented_type_mismatch",
        "property_returns_missing",
        "constructor_returns_section",
    }
    if check in errors:
        return "error"
    if check in warnings:
        return "warning"
    return "info"


SECTION_UNDERLINE_RE = re.compile(r"^\s*-{3,}\s*$")
ENTRY_RE = re.compile(
    r"^(?P<indent>\s*)"
    r"(?P<names>\*{0,2}[A-Za-z_]\w*"
    r"(?:\s*,\s*\*{0,2}[A-Za-z_]\w*)*)"
    r"\s*:\s*(?P<type>.+?)\s*$"
)
RETURN_ENTRY_RE = re.compile(
    r"^(?P<indent>\s*)"
    r"(?:(?P<name>[A-Za-z_]\w*)\s*:\s*)?"
    r"(?P<type>.+?)\s*$"
)
EXCEPTION_ENTRY_RE = re.compile(
    r"^(?P<indent>\s*)"
    r"(?P<name>[A-Za-z_][\w.]*(?:Error|Exception|Warning))\s*$"
)


def parse_sections(doc: str) -> tuple[str, list[Section], list[tuple[int, str]]]:
    lines = doc.splitlines()
    headings: list[tuple[int, str]] = []
    malformed: list[tuple[int, str]] = []

    for index, line in enumerate(lines):
        stripped = line.strip()

        if index + 1 < len(lines) and SECTION_UNDERLINE_RE.match(lines[index + 1]):
            if stripped:
                headings.append((index, stripped))
            else:
                malformed.append((index + 1, "Section heading is empty."))

        elif stripped in STANDARD_SECTIONS:
            next_line = lines[index + 1] if index + 1 < len(lines) else ""
            if not SECTION_UNDERLINE_RE.match(next_line):
                malformed.append(
                    (index + 1, f"Section {stripped!r} has no valid underline.")
                )

    if not headings:
        return doc, [], malformed

    summary = "\n".join(lines[: headings[0][0]]).rstrip()
    sections: list[Section] = []

    for position, (index, name) in enumerate(headings):
        body_start = index + 2
        body_end = (
            headings[position + 1][0] if position + 1 < len(headings) else len(lines)
        )
        sections.append(
            Section(
                name=name,
                body="\n".join(lines[body_start:body_end]).rstrip(),
                start_line=index + 1,
                end_line=body_end,
            )
        )

    return summary, sections, malformed


def section_map(sections: Iterable[Section]) -> dict[str, list[Section]]:
    result: dict[str, list[Section]] = defaultdict(list)
    for section in sections:
        result[section.name].append(section)
    return result


def parse_typed_entries(section: Section) -> list[DocEntry]:
    lines = section.body.splitlines()
    entries: list[DocEntry] = []
    index = 0

    while index < len(lines):
        line = lines[index]

        if not line.strip():
            index += 1
            continue

        match = ENTRY_RE.match(line)
        if not match:
            index += 1
            continue

        base_indent = len(match.group("indent"))
        names = tuple(name.strip() for name in match.group("names").split(","))
        type_text = match.group("type").strip()

        description_lines: list[str] = []
        start_index = index
        index += 1

        while index < len(lines):
            candidate = lines[index]

            if not candidate.strip():
                description_lines.append("")
                index += 1
                continue

            candidate_match = ENTRY_RE.match(candidate)
            candidate_indent = len(candidate) - len(candidate.lstrip())

            if candidate_match and candidate_indent <= base_indent:
                break

            if candidate_indent <= base_indent:
                break

            description_lines.append(candidate.strip())
            index += 1

        entries.append(
            DocEntry(
                names=names,
                type_text=type_text,
                description="\n".join(description_lines).strip(),
                line=section.start_line + 2 + start_index,
            )
        )

    return entries


def parse_return_entries(section: Section) -> list[DocEntry]:
    lines = section.body.splitlines()
    entries: list[DocEntry] = []
    index = 0

    while index < len(lines):
        line = lines[index]

        if not line.strip():
            index += 1
            continue

        indent = len(line) - len(line.lstrip())
        if indent != 0:
            index += 1
            continue

        match = RETURN_ENTRY_RE.match(line)
        if not match:
            index += 1
            continue

        name = match.group("name")
        type_text = match.group("type").strip()
        description_lines: list[str] = []
        start_index = index
        index += 1

        while index < len(lines):
            candidate = lines[index]

            if not candidate.strip():
                description_lines.append("")
                index += 1
                continue

            candidate_indent = len(candidate) - len(candidate.lstrip())

            if candidate_indent == 0:
                break

            description_lines.append(candidate.strip())
            index += 1

        entries.append(
            DocEntry(
                names=(name,) if name else tuple(),
                type_text=type_text,
                description="\n".join(description_lines).strip(),
                line=section.start_line + 2 + start_index,
            )
        )

    return entries


def parse_exception_entries(section: Section) -> list[DocEntry]:
    lines = section.body.splitlines()
    entries: list[DocEntry] = []
    index = 0

    while index < len(lines):
        line = lines[index]

        if not line.strip():
            index += 1
            continue

        indent = len(line) - len(line.lstrip())
        match = EXCEPTION_ENTRY_RE.match(line)

        if not match or indent != 0:
            index += 1
            continue

        name = match.group("name")
        description_lines: list[str] = []
        start_index = index
        index += 1

        while index < len(lines):
            candidate = lines[index]

            if not candidate.strip():
                description_lines.append("")
                index += 1
                continue

            candidate_indent = len(candidate) - len(candidate.lstrip())

            if candidate_indent == 0:
                break

            description_lines.append(candidate.strip())
            index += 1

        entries.append(
            DocEntry(
                names=(name,),
                type_text=None,
                description="\n".join(description_lines).strip(),
                line=section.start_line + 2 + start_index,
            )
        )

    return entries


class ScopedFlowVisitor(ast.NodeVisitor):
    def __init__(self, root: ast.AST):
        self.root = root
        self.returns: list[ast.Return] = []
        self.yields: list[ast.Yield | ast.YieldFrom] = []
        self.raises: list[ast.Raise] = []
        self.asserts: list[ast.Assert] = []
        self.warning_calls: list[ast.Call] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node is self.root:
            self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if node is self.root:
            self.generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        if node is self.root:
            self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if node is self.root:
            self.generic_visit(node)

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
        name = decorator_name(node.func)
        if name in {"warnings.warn", "warn"}:
            self.warning_calls.append(node)
        self.generic_visit(node)


def analyze_flow(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> ScopedFlowVisitor:
    visitor = ScopedFlowVisitor(node)
    visitor.visit(node)
    return visitor


def exception_name(expr: ast.expr | None) -> str | None:
    if expr is None:
        return None

    if isinstance(expr, ast.Call):
        return exception_name(expr.func)

    if isinstance(expr, ast.Name):
        return expr.id

    if isinstance(expr, ast.Attribute):
        return expr.attr

    return None


def warning_name(call: ast.Call) -> str:
    if len(call.args) >= 2:
        category = exception_name(call.args[1])

        if category:
            return category

    for keyword in call.keywords:
        if keyword.arg == "category":
            category = exception_name(keyword.value)

            if category:
                return category

    return "UserWarning"


def function_parameters(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[tuple[str, ast.arg, str]]:
    values: list[tuple[str, ast.arg, str]] = []

    positional = [*node.args.posonlyargs, *node.args.args]

    for argument in positional:
        if argument.arg not in {"self", "cls"}:
            values.append((argument.arg, argument, "parameter"))

    if node.args.vararg is not None:
        values.append((f"*{node.args.vararg.arg}", node.args.vararg, "varargs"))

    for argument in node.args.kwonlyargs:
        values.append((argument.arg, argument, "keyword-only"))

    if node.args.kwarg is not None:
        values.append((f"**{node.args.kwarg.arg}", node.args.kwarg, "kwargs"))

    return values


def return_arity(expr: ast.expr | None) -> int:
    if expr is None:
        return 0
    if isinstance(expr, (ast.Tuple, ast.List)):
        return len(expr.elts)
    return 1


def meaningful_returns(
    flow: ScopedFlowVisitor,
) -> list[ast.Return]:
    result = []

    for statement in flow.returns:
        if statement.value is None:
            if REQUIRE_RETURNS_FOR_EXPLICIT_NONE:
                result.append(statement)
            continue

        if isinstance(statement.value, ast.Constant) and statement.value.value is None:
            if REQUIRE_RETURNS_FOR_EXPLICIT_NONE:
                result.append(statement)
            continue

        result.append(statement)

    return result


def assigned_names(target: ast.expr) -> set[str]:
    result: set[str] = set()

    if isinstance(target, ast.Name):
        result.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            result.update(assigned_names(element))

    return result


def self_attribute_names(target: ast.expr) -> set[str]:
    result: set[str] = set()

    if (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    ):
        result.add(target.attr)

    elif isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            result.update(self_attribute_names(element))

    return result


def class_attributes(
    node: ast.ClassDef,
) -> dict[str, str | None]:
    attrs: dict[str, str | None] = {}

    for statement in node.body:
        if not isinstance(statement, ast.AnnAssign):
            continue

        if not isinstance(statement.target, ast.Name):
            continue

        annotation = annotation_text(statement.annotation)

        if annotation and "ClassVar" in annotation:
            continue

        name = statement.target.id

        if name in IGNORED_CLASS_ATTRIBUTES:
            continue

        if not CHECK_PRIVATE_ATTRIBUTES and name.startswith("_"):
            continue

        attrs[name] = annotation

    return attrs


def class_init_parameters(
    node: ast.ClassDef,
) -> list[tuple[str, ast.arg, str]]:
    for statement in node.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if statement.name == "__init__":
                return function_parameters(statement)
    return []


def dataclass_fields(
    node: ast.ClassDef,
) -> dict[str, str | None]:
    fields: dict[str, str | None] = {}

    for statement in node.body:
        if not isinstance(statement, ast.AnnAssign):
            continue

        if not isinstance(statement.target, ast.Name):
            continue

        annotation = annotation_text(statement.annotation)

        if annotation and "ClassVar" in annotation:
            continue

        fields[statement.target.id] = annotation

    return fields


def is_dataclass(node: ast.ClassDef) -> bool:
    return any(
        decorator_name(decorator).endswith("dataclass")
        for decorator in node.decorator_list
    )


def is_property(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(
        decorator_name(decorator) in {"property", "cached_property"}
        or decorator_name(decorator).endswith(".setter")
        for decorator in node.decorator_list
    )


def is_abstract(node: ast.AST) -> bool:
    decorators = getattr(node, "decorator_list", [])
    return any(
        "abstractmethod" in decorator_name(decorator) for decorator in decorators
    )


def has_overload_decorator(node: ast.AST) -> bool:
    decorators = getattr(node, "decorator_list", [])
    return any(
        decorator_name(decorator).endswith("overload") for decorator in decorators
    )


def source_docstring_location(
    node: ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[int, int]:
    if not node.body:
        return getattr(node, "lineno", 1), getattr(node, "col_offset", 0)

    first = node.body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return first.lineno, first.col_offset

    return getattr(node, "lineno", 1), getattr(node, "col_offset", 0)


class Auditor:
    def __init__(self, root: Path):
        self.root = root
        self.issues: list[Issue] = []
        self.symbols: list[SymbolRecord] = []
        self.results: defaultdict[str, Counter[str]] = defaultdict(Counter)
        self.current_file = Path()
        self.source = ""
        self.source_lines: list[str] = []

    def add_issue(
        self,
        node: ast.AST,
        symbol: str,
        qname: str,
        symbol_type: str,
        check: str,
        message: str,
        *,
        severity: str | None = None,
        details: dict[str, Any] | None = None,
        line: int | None = None,
        column: int | None = None,
    ) -> None:
        relative = str(self.current_file)
        issue = Issue(
            file=relative,
            line=line if line is not None else getattr(node, "lineno", 1),
            column=(column if column is not None else getattr(node, "col_offset", 0)),
            symbol=symbol,
            qualified_name=qname,
            symbol_type=symbol_type,
            check=check,
            severity=severity or severity_for(check),
            message=message,
            details=details or {},
        )
        self.issues.append(issue)
        self.results[relative][check] += 1

        detail_text = ""
        if issue.details:
            pairs = ", ".join(
                f"{key}={value!r}" for key, value in issue.details.items()
            )
            detail_text = f" [{pairs}]"

        print(
            f"{issue.file}:{issue.line}:{issue.column + 1}: "
            f"{issue.severity.upper()}: {issue.check}: "
            f"{issue.qualified_name}: {issue.message}{detail_text}"
        )

    def audit(self) -> None:
        if not self.root.exists():
            raise FileNotFoundError(f"Audit root does not exist: {self.root}")

        files = python_files(self.root)

        if not files:
            raise RuntimeError(f"No Python files found under {self.root}")

        for path in files:
            self.audit_file(path)

    def audit_file(self, path: Path) -> None:
        self.current_file = path

        try:
            with tokenize.open(path) as stream:
                self.source = stream.read()
        except (OSError, UnicodeError, SyntaxError) as exc:
            self.source = path.read_text(encoding="utf-8", errors="replace")
            self.source_lines = self.source.splitlines()

            fake = ast.Module(body=[], type_ignores=[])
            self.add_issue(
                fake,
                "<module>",
                str(path),
                "module",
                "syntax_error",
                f"Could not decode source: {exc}",
                line=1,
            )
            return

        self.source_lines = self.source.splitlines()

        try:
            tree = ast.parse(
                self.source,
                filename=str(path),
                type_comments=True,
            )
        except SyntaxError as exc:
            fake = ast.Module(body=[], type_ignores=[])
            self.add_issue(
                fake,
                "<module>",
                str(path),
                "module",
                "syntax_error",
                exc.msg,
                line=exc.lineno or 1,
                column=(exc.offset or 1) - 1,
                details={"text": (exc.text or "").strip()},
            )
            return

        module_doc = ast.get_docstring(tree, clean=False)

        if CHECK_MODULE_DOCSTRINGS and module_doc is None:
            self.add_issue(
                tree,
                "<module>",
                str(path),
                "module",
                "missing_module_docstring",
                "Module has no docstring.",
                line=1,
            )
        elif module_doc is not None:
            self.audit_docstring(
                tree,
                "<module>",
                str(path),
                "module",
                clean_docstring(module_doc),
                public=True,
            )

        self.walk_scope(tree.body, [])

    def walk_scope(self, body: list[ast.stmt], stack: list[str]) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                qname = qualified_name(stack, node.name)
                self.audit_class(node, qname)
                self.walk_scope(node.body, [*stack, node.name])

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qname = qualified_name(stack, node.name)
                self.audit_function(node, qname)
                self.walk_scope(node.body, [*stack, node.name])

    def should_check_symbol(self, name: str) -> bool:
        if CHECK_PRIVATE_SYMBOLS:
            return True
        return is_public_name(name)

    def audit_class(self, node: ast.ClassDef, qname: str) -> None:
        public = is_public_name(node.name)

        if not self.should_check_symbol(node.name):
            return

        doc = ast.get_docstring(node, clean=False)

        record = SymbolRecord(
            file=str(self.current_file),
            line=node.lineno,
            qualified_name=qname,
            symbol_type="class",
            public=public,
            documented=doc is not None,
        )
        self.symbols.append(record)

        issue_start = len(self.issues)

        if doc is None:
            self.add_issue(
                node,
                node.name,
                qname,
                "class",
                "missing_docstring",
                "Class has no docstring.",
            )
            self.finalize_symbol_record(record, issue_start)
            return

        doc = clean_docstring(doc)

        section_index = self.audit_docstring(
            node,
            node.name,
            qname,
            "class",
            doc,
            public=public,
        )

        if "Methods" in section_index:
            self.add_issue(
                node,
                node.name,
                qname,
                "class",
                "redundant_methods_section",
                "Class docstring contains a Methods section.",
            )

        class_is_dataclass = is_dataclass(node)

        if class_is_dataclass:
                                                                                  
                                                                         
            dataclass_field_map = dataclass_fields(node)

            expected_params = {
                name for name in dataclass_field_map if not name.startswith("_")
            }

            expected_attrs: dict[str, str | None] = {}
        else:
                                                                              
            expected_params = {
                name.lstrip("*") for name, _, _ in class_init_parameters(node)
            }

                                                                          
                                                                                
                                 
            expected_attrs = class_attributes(node)

        parameter_entries = [
            entry
            for section_name in ("Parameters", "Other Parameters")
            for section in section_index.get(section_name, [])
            for entry in parse_typed_entries(section)
        ]

        documented_param_list = [
            name.lstrip("*") for entry in parameter_entries for name in entry.names
        ]

        documented_params = set(documented_param_list)

        parameter_entry_map = {
            name.lstrip("*"): entry
            for entry in parameter_entries
            for name in entry.names
        }

        for name in sorted(expected_params - documented_params):
            self.add_issue(
                node,
                node.name,
                qname,
                "class",
                "missing_param",
                f"Constructor parameter {name!r} is not documented in the class docstring.",
                details={"parameter": name},
            )

        for name in sorted(documented_params - expected_params):
            self.add_issue(
                node,
                node.name,
                qname,
                "class",
                "extra_param",
                f"Documented class parameter {name!r} does not match "
                "a detected constructor parameter.",
                details={"parameter": name},
            )

        duplicate_params = [
            name for name, count in Counter(documented_param_list).items() if count > 1
        ]

        for name in sorted(duplicate_params):
            self.add_issue(
                node,
                node.name,
                qname,
                "class",
                "duplicate_params",
                f"Constructor parameter {name!r} is documented more than once.",
                details={
                    "parameter": name,
                    "count": documented_param_list.count(name),
                },
            )

        if class_is_dataclass:
            expected_param_types = dataclass_field_map
        else:
            expected_param_types = {
                name.lstrip("*"): annotation_text(argument.annotation)
                for name, argument, _ in class_init_parameters(node)
            }

        for name in sorted(expected_params & documented_params):
            entry = parameter_entry_map[name]

            if not entry.type_text:
                self.add_issue(
                    node,
                    node.name,
                    qname,
                    "class",
                    "missing_param_type",
                    f"Constructor parameter {name!r} has no documented type.",
                    details={"parameter": name},
                )

            if not entry.description:
                self.add_issue(
                    node,
                    node.name,
                    qname,
                    "class",
                    "missing_param_description",
                    f"Constructor parameter {name!r} has no description.",
                    details={"parameter": name},
                )

            annotation = expected_param_types.get(name)

            if (
                annotation
                and entry.type_text
                and not likely_type_match(annotation, entry.type_text)
            ):
                self.add_issue(
                    node,
                    node.name,
                    qname,
                    "class",
                    "documented_type_mismatch",
                    f"Constructor parameter {name!r} has inconsistent "
                    "annotated and documented types.",
                    details={
                        "parameter": name,
                        "annotation": annotation,
                        "documented": entry.type_text,
                    },
                )

        attribute_entries = [
            entry
            for section in section_index.get("Attributes", [])
            for entry in parse_typed_entries(section)
        ]

        documented_attr_list = [
            name for entry in attribute_entries for name in entry.names
        ]

        documented_attrs = set(documented_attr_list)

        attribute_entry_map = {
            name: entry for entry in attribute_entries for name in entry.names
        }

        expected_attr_names = set(expected_attrs)

        for name in sorted(expected_attr_names - documented_attrs):
            self.add_issue(
                node,
                node.name,
                qname,
                "class",
                "missing_attributes",
                f"Attribute {name!r} is not documented.",
                details={"attribute": name},
            )

        for name in sorted(documented_attrs - expected_attr_names):
            if class_is_dataclass and name in expected_params:
                message = (
                    f"Dataclass field {name!r} is documented under Attributes. "
                    "Document dataclass fields under Parameters instead."
                )
            else:
                message = (
                    f"Documented attribute {name!r} was not detected "
                    "as an explicitly declared class attribute."
                )

            self.add_issue(
                node,
                node.name,
                qname,
                "class",
                "extra_attributes",
                message,
                severity="info",
                details={"attribute": name},
            )

        duplicate_attrs = [
            name for name, count in Counter(documented_attr_list).items() if count > 1
        ]

        for name in sorted(duplicate_attrs):
            self.add_issue(
                node,
                node.name,
                qname,
                "class",
                "duplicate_attributes",
                f"Attribute {name!r} is documented more than once.",
                details={
                    "attribute": name,
                    "count": documented_attr_list.count(name),
                },
            )

        for name in sorted(expected_attr_names & documented_attrs):
            entry = attribute_entry_map[name]

            if not entry.type_text:
                self.add_issue(
                    node,
                    node.name,
                    qname,
                    "class",
                    "missing_attribute_type",
                    f"Attribute {name!r} has no documented type.",
                    details={"attribute": name},
                )

            if not entry.description:
                self.add_issue(
                    node,
                    node.name,
                    qname,
                    "class",
                    "missing_attribute_description",
                    f"Attribute {name!r} has no description.",
                    details={"attribute": name},
                )

            annotation = expected_attrs.get(name)

            if (
                annotation
                and entry.type_text
                and not likely_type_match(annotation, entry.type_text)
            ):
                self.add_issue(
                    node,
                    node.name,
                    qname,
                    "class",
                    "documented_type_mismatch",
                    f"Attribute {name!r} has inconsistent annotated and documented types.",
                    details={
                        "attribute": name,
                        "annotation": annotation,
                        "documented": entry.type_text,
                    },
                )

        if (
            REQUIRE_ABSTRACT_CLASS_NOTES
            and any(is_abstract(child) for child in node.body)
            and "Notes" not in section_index
        ):
            self.add_issue(
                node,
                node.name,
                qname,
                "class",
                "missing_notes",
                "Abstract class has no Notes section.",
            )

        self.finalize_symbol_record(record, issue_start)

    def audit_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        qname: str,
    ) -> None:
        public = is_public_name(node.name)

        if not self.should_check_symbol(node.name):
            return

        if node.name in OPTIONAL_DOCSTRING_METHODS and ast.get_docstring(node) is None:
            return

        doc = ast.get_docstring(node, clean=False)
        symbol_type = (
            "async function" if isinstance(node, ast.AsyncFunctionDef) else "function"
        )

        record = SymbolRecord(
            file=str(self.current_file),
            line=node.lineno,
            qualified_name=qname,
            symbol_type=symbol_type,
            public=public,
            documented=doc is not None,
        )
        self.symbols.append(record)
        issue_start = len(self.issues)

        if doc is None:
            if not has_overload_decorator(node):
                self.add_issue(
                    node,
                    node.name,
                    qname,
                    symbol_type,
                    "missing_docstring",
                    "Function has no docstring.",
                )
            self.finalize_symbol_record(record, issue_start)
            return

        doc = clean_docstring(doc)
        section_index = self.audit_docstring(
            node,
            node.name,
            qname,
            symbol_type,
            doc,
            public=public,
        )

        parameters = function_parameters(node)
        expected_parameter_names = {name for name, _, _ in parameters}

        parameter_entries = [
            entry
            for section_name in ("Parameters", "Other Parameters")
            for section in section_index.get(section_name, [])
            for entry in parse_typed_entries(section)
        ]

        documented_parameter_list = [
            name for entry in parameter_entries for name in entry.names
        ]
        documented_parameter_names = set(documented_parameter_list)

        for name in sorted(expected_parameter_names - documented_parameter_names):
            check = "missing_param"
            if name.startswith("**"):
                check = "missing_kwargs"
            elif name.startswith("*"):
                check = "missing_varargs"

            self.add_issue(
                node,
                node.name,
                qname,
                symbol_type,
                check,
                f"Parameter {name!r} is not documented.",
                details={"parameter": name},
            )

        for name in sorted(documented_parameter_names - expected_parameter_names):
            self.add_issue(
                node,
                node.name,
                qname,
                symbol_type,
                "extra_param",
                f"Documented parameter {name!r} is not in the signature.",
                details={"parameter": name},
            )

        duplicates = [
            name
            for name, count in Counter(documented_parameter_list).items()
            if count > 1
        ]
        for name in sorted(duplicates):
            self.add_issue(
                node,
                node.name,
                qname,
                symbol_type,
                "duplicate_params",
                f"Parameter {name!r} is documented more than once.",
                details={"parameter": name},
            )

        entry_by_parameter = {
            name: entry for entry in parameter_entries for name in entry.names
        }

        arg_by_name = {name: arg for name, arg, _ in parameters}

        for name in sorted(expected_parameter_names & documented_parameter_names):
            entry = entry_by_parameter[name]
            argument = arg_by_name[name]

            if not entry.type_text:
                self.add_issue(
                    node,
                    node.name,
                    qname,
                    symbol_type,
                    "missing_param_type",
                    f"Parameter {name!r} has no documented type.",
                    details={"parameter": name},
                )

            if not entry.description:
                self.add_issue(
                    node,
                    node.name,
                    qname,
                    symbol_type,
                    "missing_param_description",
                    f"Parameter {name!r} has no description.",
                    details={"parameter": name},
                )

            annotation = annotation_text(argument.annotation)

            if (
                CHECK_TYPE_HINTS
                and public
                and argument.annotation is None
                and not name.startswith("*")
            ):
                self.add_issue(
                    node,
                    node.name,
                    qname,
                    symbol_type,
                    "missing_parameter_annotation",
                    f"Public parameter {name!r} has no type annotation.",
                    details={"parameter": name},
                )

            if (
                annotation
                and entry.type_text
                and not likely_type_match(annotation, entry.type_text)
            ):
                self.add_issue(
                    node,
                    node.name,
                    qname,
                    symbol_type,
                    "documented_type_mismatch",
                    f"Parameter {name!r} has inconsistent annotated and "
                    "documented types.",
                    details={
                        "parameter": name,
                        "annotation": annotation,
                        "documented": entry.type_text,
                    },
                )

        flow = analyze_flow(node)
        returns = meaningful_returns(flow)
        yields = flow.yields

        return_sections = section_index.get("Returns", [])
        yield_sections = section_index.get("Yields", [])

        has_returns_section = bool(return_sections)
        has_yields_section = bool(yield_sections)

        if (
            returns
            and not has_returns_section
            and node.name not in OPTIONAL_RETURNS_METHODS
        ):
            self.add_issue(
                node,
                node.name,
                qname,
                symbol_type,
                "missing_returns",
                "Function returns a value but has no Returns section.",
            )

        if (
            not returns
            and has_returns_section
            and node.name not in {"__init__", "__post_init__"}
        ):
            self.add_issue(
                node,
                node.name,
                qname,
                symbol_type,
                "extra_returns",
                "Docstring has a Returns section, but no value is returned.",
            )

        if node.name in {"__init__", "__post_init__"} and has_returns_section:
            self.add_issue(
                node,
                node.name,
                qname,
                symbol_type,
                "constructor_returns_section",
                "Constructors and post-initialization hooks should not "
                "document a Returns section.",
            )

        if yields and not has_yields_section:
            self.add_issue(
                node,
                node.name,
                qname,
                symbol_type,
                "missing_yields",
                "Generator yields values but has no Yields section.",
            )

        if not yields and has_yields_section:
            self.add_issue(
                node,
                node.name,
                qname,
                symbol_type,
                "extra_yields",
                "Docstring has a Yields section, but the function does not yield.",
            )

        if yields and has_returns_section:
            self.add_issue(
                node,
                node.name,
                qname,
                symbol_type,
                "returns_and_yields",
                "Generator contains a Returns section. Use Yields unless a "
                "generator return value is intentionally documented.",
            )

        if return_sections:
            return_entries = [
                entry
                for section in return_sections
                for entry in parse_return_entries(section)
            ]

            for entry in return_entries:
                if not entry.type_text:
                    self.add_issue(
                        node,
                        node.name,
                        qname,
                        symbol_type,
                        "missing_return_type",
                        "A return value has no documented type.",
                    )

                if not entry.description:
                    self.add_issue(
                        node,
                        node.name,
                        qname,
                        symbol_type,
                        "missing_return_description",
                        "A return value has no description.",
                    )

            arities = {
                return_arity(statement.value)
                for statement in returns
                if statement.value is not None
            }
            if (
                len(arities) == 1
                and return_entries
                and next(iter(arities)) > 1
                and len(return_entries) not in {1, next(iter(arities))}
            ):
                self.add_issue(
                    node,
                    node.name,
                    qname,
                    symbol_type,
                    "return_arity_mismatch",
                    "Documented return count does not match the detected "
                    "tuple return size.",
                    details={
                        "detected": next(iter(arities)),
                        "documented": len(return_entries),
                    },
                )

        if yield_sections:
            yield_entries = [
                entry
                for section in yield_sections
                for entry in parse_return_entries(section)
            ]

            for entry in yield_entries:
                if not entry.type_text:
                    self.add_issue(
                        node,
                        node.name,
                        qname,
                        symbol_type,
                        "missing_yield_type",
                        "A yielded value has no documented type.",
                    )

                if not entry.description:
                    self.add_issue(
                        node,
                        node.name,
                        qname,
                        symbol_type,
                        "missing_yield_description",
                        "A yielded value has no description.",
                    )

        if (
            CHECK_TYPE_HINTS
            and public
            and node.name not in {"__init__", "__post_init__"}
            and node.returns is None
        ):
            self.add_issue(
                node,
                node.name,
                qname,
                symbol_type,
                "missing_return_annotation",
                "Public function has no return annotation.",
            )

        if is_property(node) and returns and not has_returns_section:
            self.add_issue(
                node,
                node.name,
                qname,
                symbol_type,
                "property_returns_missing",
                "Property returns a value but does not document it.",
            )

        self.audit_raises_and_warnings(
            node,
            qname,
            symbol_type,
            flow,
            section_index,
        )

        decorators = {decorator_name(decorator) for decorator in node.decorator_list}
        important_decorators = {
            name
            for name in decorators
            if any(
                marker in name
                for marker in (
                    "deprecated",
                    "contextmanager",
                    "lru_cache",
                    "cache",
                )
            )
        }

        if important_decorators and "Notes" not in section_index:
            self.add_issue(
                node,
                node.name,
                qname,
                symbol_type,
                "undocumented_decorator_behavior",
                "Behavior-affecting decorators are not discussed in Notes.",
                details={"decorators": sorted(important_decorators)},
            )

        if (
            STRICT_ABSTRACT_METHOD_DOCS
            and is_abstract(node)
            and len(first_nonempty_line(doc).split()) < 3
        ):
            self.add_issue(
                node,
                node.name,
                qname,
                symbol_type,
                "missing_summary",
                "Abstract method summary is too short to describe its contract.",
            )

        self.finalize_symbol_record(record, issue_start)

    def audit_raises_and_warnings(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        qname: str,
        symbol_type: str,
        flow: ScopedFlowVisitor,
        section_index: dict[str, list[Section]],
    ) -> None:
        raised: list[str] = []
        dynamic_raise_count = 0
        bare_raise_count = 0

        for statement in flow.raises:
            if statement.exc is None:
                bare_raise_count += 1
                continue

            name = exception_name(statement.exc)
            if name:
                raised.append(name)
            else:
                dynamic_raise_count += 1

        if INCLUDE_ASSERTION_ERRORS and flow.asserts:
            raised.append("AssertionError")

        raise_entries = [
            entry
            for section in section_index.get("Raises", [])
            for entry in parse_exception_entries(section)
        ]
        documented_raises = [entry.names[0] for entry in raise_entries if entry.names]

        for name in sorted(set(raised) - set(documented_raises)):
            self.add_issue(
                node,
                node.name,
                qname,
                symbol_type,
                "missing_raise",
                f"Raised exception {name!r} is not documented.",
                details={"exception": name},
            )

        for name in sorted(set(documented_raises) - set(raised)):
            self.add_issue(
                node,
                node.name,
                qname,
                symbol_type,
                "extra_raise",
                f"Documented exception {name!r} is not raised directly.",
                severity="info",
                details={
                    "exception": name,
                    "note": "It may be raised by a called function.",
                },
            )

        duplicates = [
            name for name, count in Counter(documented_raises).items() if count > 1
        ]
        for name in sorted(duplicates):
            self.add_issue(
                node,
                node.name,
                qname,
                symbol_type,
                "duplicate_raises",
                f"Exception {name!r} is documented more than once.",
                details={"exception": name},
            )

        for entry in raise_entries:
            if not entry.description:
                self.add_issue(
                    node,
                    node.name,
                    qname,
                    symbol_type,
                    "missing_raise_description",
                    f"Exception {entry.names[0]!r} has no description.",
                    details={"exception": entry.names[0]},
                )

        if bare_raise_count:
            self.add_issue(
                node,
                node.name,
                qname,
                symbol_type,
                "bare_raise",
                "Function contains one or more bare re-raise statements.",
                severity="info",
                details={"count": bare_raise_count},
            )

        if dynamic_raise_count:
            self.add_issue(
                node,
                node.name,
                qname,
                symbol_type,
                "dynamic_raise",
                "One or more raised exception types could not be resolved.",
                severity="info",
                details={"count": dynamic_raise_count},
            )

        if not CHECK_WARNINGS:
            return

        emitted_warnings = [warning_name(call) for call in flow.warning_calls]
        warn_entries = [
            entry
            for section in section_index.get("Warns", [])
            for entry in parse_exception_entries(section)
        ]
        documented_warnings = [entry.names[0] for entry in warn_entries if entry.names]

        for name in sorted(set(emitted_warnings) - set(documented_warnings)):
            self.add_issue(
                node,
                node.name,
                qname,
                symbol_type,
                "missing_warns",
                f"Emitted warning {name!r} is not documented.",
                details={"warning": name},
            )

        for name in sorted(set(documented_warnings) - set(emitted_warnings)):
            self.add_issue(
                node,
                node.name,
                qname,
                symbol_type,
                "extra_warns",
                f"Documented warning {name!r} is not emitted directly.",
                severity="info",
                details={"warning": name},
            )

        for entry in warn_entries:
            if not entry.description:
                self.add_issue(
                    node,
                    node.name,
                    qname,
                    symbol_type,
                    "missing_warn_description",
                    f"Warning {entry.names[0]!r} has no description.",
                    details={"warning": entry.names[0]},
                )

    def audit_docstring(
        self,
        node: ast.AST,
        symbol: str,
        qname: str,
        symbol_type: str,
        doc: str,
        *,
        public: bool,
    ) -> dict[str, list[Section]]:
        if not doc.strip():
            self.add_issue(
                node,
                symbol,
                qname,
                symbol_type,
                "empty_docstring",
                "Docstring is empty.",
            )
            return {}

        summary, sections, malformed = parse_sections(doc)
        index = section_map(sections)
        summary_line = first_nonempty_line(summary)

        if not summary_line:
            self.add_issue(
                node,
                symbol,
                qname,
                symbol_type,
                "missing_summary",
                "Docstring has no summary line.",
            )
        else:
            if len(summary_line) > SUMMARY_LINE_MAX_LENGTH:
                self.add_issue(
                    node,
                    symbol,
                    qname,
                    symbol_type,
                    "summary_too_long",
                    "Summary line exceeds the configured maximum length.",
                    details={
                        "length": len(summary_line),
                        "maximum": SUMMARY_LINE_MAX_LENGTH,
                    },
                )

            if summary_line[-1:] not in {".", "!", "?", ":"}:
                self.add_issue(
                    node,
                    symbol,
                    qname,
                    symbol_type,
                    "summary_punctuation",
                    "Summary line does not end with punctuation.",
                )

            if symbol_type in {"function", "async function"} and re.match(
                r"^(This|The|A|An|It|Function|Method)\b",
                summary_line,
                re.IGNORECASE,
            ):
                self.add_issue(
                    node,
                    symbol,
                    qname,
                    symbol_type,
                    "summary_not_imperative",
                    "Function summary may not use imperative mood.",
                    severity="info",
                    details={"summary": summary_line},
                )

        if re.search(r"(?mi)^\s*(TODO|FIXME|TBD|XXX)\b", doc):
            self.add_issue(
                node,
                symbol,
                qname,
                symbol_type,
                "todo_docs",
                "Docstring contains an unfinished documentation marker.",
            )

        for relative_line, message in malformed:
            doc_line, doc_column = source_docstring_location(node)
            self.add_issue(
                node,
                symbol,
                qname,
                symbol_type,
                "malformed_section",
                message,
                line=doc_line + relative_line,
                column=doc_column,
            )

        for name, instances in index.items():
            if len(instances) > 1:
                self.add_issue(
                    node,
                    symbol,
                    qname,
                    symbol_type,
                    "duplicate_section",
                    f"Section {name!r} appears more than once.",
                    details={"section": name, "count": len(instances)},
                )

            for section in instances:
                if not section.body.strip():
                    self.add_issue(
                        node,
                        symbol,
                        qname,
                        symbol_type,
                        "empty_sections",
                        f"Section {name!r} is empty.",
                        details={"section": name},
                    )

        unknown_sections = sorted(set(index) - set(STANDARD_SECTIONS))
        for name in unknown_sections:
            self.add_issue(
                node,
                symbol,
                qname,
                symbol_type,
                "unknown_section",
                f"Unknown NumPy docstring section {name!r}.",
                severity="info",
                details={"section": name},
            )

        observed_standard = [
            section.name for section in sections if section.name in STANDARD_SECTIONS
        ]
        expected_positions = {
            name: position for position, name in enumerate(STANDARD_SECTIONS)
        }
        positions = [expected_positions[name] for name in observed_standard]

        if positions != sorted(positions):
            self.add_issue(
                node,
                symbol,
                qname,
                symbol_type,
                "section_order",
                "NumPy docstring sections are not in the recommended order.",
                details={"observed": observed_standard},
            )

        if REQUIRE_EXAMPLES and public and "Examples" not in index:
            self.add_issue(
                node,
                symbol,
                qname,
                symbol_type,
                "missing_examples",
                "Public symbol has no Examples section.",
            )

        if doc.count("```") % 2:
            self.add_issue(
                node,
                symbol,
                qname,
                symbol_type,
                "unbalanced_code_fence",
                "Docstring contains an unbalanced Markdown code fence.",
            )

        if doc.count("``") % 2:
            self.add_issue(
                node,
                symbol,
                qname,
                symbol_type,
                "invalid_backticks",
                "Docstring may contain unbalanced double backticks.",
            )

        trailing_lines = [
            number
            for number, line in enumerate(doc.splitlines(), start=1)
            if line.rstrip() != line
        ]
        if trailing_lines:
            self.add_issue(
                node,
                symbol,
                qname,
                symbol_type,
                "trailing_whitespace",
                "Docstring contains trailing whitespace.",
                severity="info",
                details={"docstring_lines": trailing_lines},
            )

        parameter_entries = [
            entry
            for section_name in ("Parameters", "Other Parameters")
            for section in index.get(section_name, [])
            for entry in parse_typed_entries(section)
        ]
        private_documented = sorted(
            {
                name
                for entry in parameter_entries
                for name in entry.names
                if name.lstrip("*").startswith("_")
            }
        )

        if public and private_documented:
            self.add_issue(
                node,
                symbol,
                qname,
                symbol_type,
                "private_name_in_public_docs",
                "Public API documentation includes private parameter names.",
                severity="info",
                details={"parameters": private_documented},
            )

        return index

    def finalize_symbol_record(
        self,
        record: SymbolRecord,
        issue_start: int,
    ) -> None:
        relevant = self.issues[issue_start:]
        counts = Counter(issue.severity for issue in relevant)
        record.issue_count = len(relevant)
        record.error_count = counts["error"]
        record.warning_count = counts["warning"]
        record.info_count = counts["info"]

    def write_text_report(self, summary: pd.DataFrame) -> None:
        lines = []

        lines.append("=" * 88)
        lines.append("NUMPY DOCSTRING AUDIT")
        lines.append("=" * 88)
        lines.append("")

        issues_by_file = defaultdict(list)

        for issue in self.issues:
            issues_by_file[issue.file].append(issue)

        for file in sorted(issues_by_file):
            file_issues = issues_by_file[file]

            lines.append(file)
            lines.append("-" * len(file))

            for issue in sorted(
                file_issues,
                key=lambda item: (
                    item.line,
                    item.column,
                    item.check,
                    item.qualified_name,
                ),
            ):
                details = ""

                if issue.details:
                    details = (
                        " ["
                        + ", ".join(
                            f"{key}={value!r}" for key, value in issue.details.items()
                        )
                        + "]"
                    )

                lines.append(
                    f"{issue.line}:{issue.column + 1}: "
                    f"{issue.severity.upper()}: "
                    f"{issue.check}: "
                    f"{issue.qualified_name}: "
                    f"{issue.message}"
                    f"{details}"
                )

            lines.append("")

        errors = sum(issue.severity == "error" for issue in self.issues)
        warnings = sum(issue.severity == "warning" for issue in self.issues)
        information = sum(issue.severity == "info" for issue in self.issues)

        documented = sum(record.documented for record in self.symbols)

        coverage = 100.0 * documented / len(self.symbols) if self.symbols else 100.0

        lines.append("=" * 88)
        lines.append("AUDIT SUMMARY")
        lines.append("=" * 88)
        lines.append(f"Root:                 {self.root}")
        lines.append(f"Python files:         {len(python_files(self.root))}")
        lines.append(f"Audited symbols:      {len(self.symbols)}")
        lines.append(f"Total issues:         {len(self.issues)}")
        lines.append(f"Errors:               {errors}")
        lines.append(f"Warnings:             {warnings}")
        lines.append(f"Informational:        {information}")
        lines.append(f"Docstring coverage:   {coverage:.2f}%")
        lines.append("")

        if self.issues:
            lines.append("Most frequent checks:")
            lines.append("")

            for check, count in Counter(
                issue.check for issue in self.issues
            ).most_common():
                lines.append(f"  {check:<42} {count:>6}")

            lines.append("")

        if not summary.empty:
            lines.append("Files with issues:")
            lines.append("")

            issue_summary = summary.loc[summary["total"] > 0]

            for file, row in issue_summary.iterrows():
                lines.append(
                    f"  {file:<65} "
                    f"errors={int(row['errors']):>4} "
                    f"warnings={int(row['warnings']):>4} "
                    f"info={int(row['info']):>4} "
                    f"total={int(row['total']):>4}"
                )

            lines.append("")

        lines.append("Generated reports:")
        lines.append(f"  Issue report:       {OUTPUT_ISSUES_CSV}")
        lines.append(f"  File summary:       {OUTPUT_SUMMARY_CSV}")
        lines.append(f"  Symbol report:      {OUTPUT_SYMBOLS_CSV}")
        lines.append(f"  JSON report:        {OUTPUT_JSON}")
        lines.append(f"  Heatmap:            {OUTPUT_HEATMAP}")
        lines.append(f"  Totals chart:       {OUTPUT_TOTALS}")
        lines.append("")

        OUTPUT_TEXT.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

    def write_outputs(self) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        issue_rows = []
        for issue in self.issues:
            row = asdict(issue)
            row["details"] = json.dumps(
                issue.details,
                sort_keys=True,
                default=str,
            )
            issue_rows.append(row)

        issue_columns = [
            "file",
            "line",
            "column",
            "symbol",
            "qualified_name",
            "symbol_type",
            "check",
            "severity",
            "message",
            "details",
        ]

        with OUTPUT_ISSUES_CSV.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=issue_columns)
            writer.writeheader()
            writer.writerows(issue_rows)

        all_checks = sorted(
            set(ISSUE_CHECKS)
            | {check for counter in self.results.values() for check in counter}
        )

        summary = pd.DataFrame.from_dict(
            {
                file: {check: counter.get(check, 0) for check in all_checks}
                for file, counter in self.results.items()
            },
            orient="index",
        )

        all_files = [str(path) for path in python_files(self.root)]
        summary = summary.reindex(index=all_files, fill_value=0)
        summary = summary.reindex(columns=all_checks, fill_value=0)
        summary = summary.fillna(0).astype(int)
        summary.index.name = "file"

        severity_by_file = pd.DataFrame(
            [
                {
                    "file": file,
                    "errors": sum(
                        1
                        for issue in self.issues
                        if issue.file == file and issue.severity == "error"
                    ),
                    "warnings": sum(
                        1
                        for issue in self.issues
                        if issue.file == file and issue.severity == "warning"
                    ),
                    "info": sum(
                        1
                        for issue in self.issues
                        if issue.file == file and issue.severity == "info"
                    ),
                }
                for file in all_files
            ]
        ).set_index("file")

        summary["total"] = summary.sum(axis=1)
        summary["errors"] = severity_by_file["errors"]
        summary["warnings"] = severity_by_file["warnings"]
        summary["info"] = severity_by_file["info"]

        summary = summary.sort_values(
            ["errors", "warnings", "total"],
            ascending=False,
        )
        summary.to_csv(OUTPUT_SUMMARY_CSV)

        self.write_text_report(summary)

        symbol_frame = pd.DataFrame([asdict(record) for record in self.symbols])
        if not symbol_frame.empty:
            symbol_frame["quality_score"] = (
                100
                - symbol_frame["error_count"] * 20
                - symbol_frame["warning_count"] * 8
                - symbol_frame["info_count"] * 2
            ).clip(lower=0)
            symbol_frame = symbol_frame.sort_values(
                ["quality_score", "file", "line"],
                ascending=[True, True, True],
            )
        symbol_frame.to_csv(OUTPUT_SYMBOLS_CSV, index=False)

        payload = {
            "root": str(self.root),
            "configuration": {
                "check_private_attributes": CHECK_PRIVATE_ATTRIBUTES,
                "check_private_symbols": CHECK_PRIVATE_SYMBOLS,
                "check_module_docstrings": CHECK_MODULE_DOCSTRINGS,
                "check_type_hints": CHECK_TYPE_HINTS,
                "require_examples": REQUIRE_EXAMPLES,
                "include_assertion_errors": INCLUDE_ASSERTION_ERRORS,
                "check_warnings": CHECK_WARNINGS,
            },
            "totals": {
                "files": len(all_files),
                "symbols": len(self.symbols),
                "issues": len(self.issues),
                "errors": sum(issue.severity == "error" for issue in self.issues),
                "warnings": sum(issue.severity == "warning" for issue in self.issues),
                "info": sum(issue.severity == "info" for issue in self.issues),
            },
            "issues": [asdict(issue) for issue in self.issues],
            "symbols": [asdict(record) for record in self.symbols],
        }

        OUTPUT_JSON.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

        self.create_visualizations(summary)

        print("\n" + "=" * 88)
        print("AUDIT SUMMARY")
        print("=" * 88)
        print(f"Root:                 {self.root}")
        print(f"Python files:         {len(all_files)}")
        print(f"Audited symbols:      {len(self.symbols)}")
        print(f"Total issues:         {len(self.issues)}")
        print(
            "Errors:               "
            f"{sum(issue.severity == 'error' for issue in self.issues)}"
        )
        print(
            "Warnings:             "
            f"{sum(issue.severity == 'warning' for issue in self.issues)}"
        )
        print(
            "Informational:        "
            f"{sum(issue.severity == 'info' for issue in self.issues)}"
        )

        documented = sum(record.documented for record in self.symbols)
        coverage = 100.0 * documented / len(self.symbols) if self.symbols else 100.0
        print(f"Docstring coverage:   {coverage:.2f}%")
        print()
        print(f"Issue report:         {OUTPUT_ISSUES_CSV}")
        print(f"File summary:         {OUTPUT_SUMMARY_CSV}")
        print(f"Symbol report:        {OUTPUT_SYMBOLS_CSV}")
        print(f"JSON report:          {OUTPUT_JSON}")
        print(f"Heatmap:              {OUTPUT_HEATMAP}")
        print(f"Totals chart:         {OUTPUT_TOTALS}")

        if self.issues:
            print("\nMost frequent checks:")
            for check, count in Counter(
                issue.check for issue in self.issues
            ).most_common(20):
                print(f"  {check:<40} {count:>6}")

            print("\nFiles with the most errors:")
            for file, row in summary.head(20).iterrows():
                print(
                    f"  {file:<65} "
                    f"errors={row['errors']:>4} "
                    f"warnings={row['warnings']:>4} "
                    f"total={row['total']:>4}"
                )

    def create_visualizations(self, summary: pd.DataFrame) -> None:
        check_columns = [
            column
            for column in summary.columns
            if column not in {"total", "errors", "warnings", "info"}
            and summary[column].sum() > 0
        ]

        if check_columns and not summary.empty:
            heatmap_data = summary.loc[
                summary[check_columns].sum(axis=1) > 0,
                check_columns,
            ]

            width = max(14, len(check_columns) * 0.75)
            height = max(6, len(heatmap_data) * 0.35)

            plt.figure(figsize=(width, height))
            sns.heatmap(
                heatmap_data,
                annot=True,
                fmt="d",
                cmap="Reds",
                linewidths=0.5,
                linecolor="white",
                cbar_kws={"label": "Issue count"},
            )
            plt.title("NumPy docstring audit by file")
            plt.xlabel("Audit check")
            plt.ylabel("Python file")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(OUTPUT_HEATMAP, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.figure(figsize=(10, 4))
            plt.text(
                0.5,
                0.5,
                "No audit issues detected",
                ha="center",
                va="center",
                fontsize=18,
            )
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(OUTPUT_HEATMAP, dpi=300, bbox_inches="tight")
            plt.close()

        totals = Counter(issue.check for issue in self.issues)

        if totals:
            total_frame = (
                pd.DataFrame(
                    totals.items(),
                    columns=["check", "count"],
                )
                .sort_values("count", ascending=True)
                .tail(30)
            )

            plt.figure(figsize=(12, max(6, len(total_frame) * 0.35)))
            sns.barplot(
                data=total_frame,
                x="count",
                y="check",
                color="firebrick",
            )
            plt.title("Most frequent NumPy docstring issues")
            plt.xlabel("Issue count")
            plt.ylabel("Audit check")
            plt.tight_layout()
            plt.savefig(OUTPUT_TOTALS, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.figure(figsize=(10, 4))
            plt.text(
                0.5,
                0.5,
                "No audit issues detected",
                ha="center",
                va="center",
                fontsize=18,
            )
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(OUTPUT_TOTALS, dpi=300, bbox_inches="tight")
            plt.close()


auditor = Auditor(ROOT)

try:
    auditor.audit()
    auditor.write_outputs()
except Exception as exc:
    print(
        f"\nAUDIT FAILED: {type(exc).__name__}: {exc}",
        file=sys.stderr,
    )
    raise
