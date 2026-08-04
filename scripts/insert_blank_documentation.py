from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import shutil
import sys
import tokenize
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_ROOT = Path("cccma_ppp")
DEFAULT_REPORT = Path("output/documentation_audit_results/docstring_insertions.json")

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

OPTIONAL_UNDOCUMENTED_METHODS = {
    "__repr__",
    "__str__",
}

IGNORED_CLASS_ATTRIBUTES = {
    "__annotations__",
    "__dataclass_fields__",
    "__match_args__",
    "__slots__",
}

FUNCTION_SUMMARY = "Document this function."
CLASS_SUMMARY = "Document this class."
PLACEHOLDER_DESCRIPTION = "Description not yet provided."

INDENT_WIDTH = 4

FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef
DocumentableNode = ast.ClassDef | FunctionNode


@dataclass(frozen=True)
class SourceEdit:
    offset: int
    text: str
    qualified_name: str
    line: int
    symbol_type: str


@dataclass(frozen=True)
class PlannedInsertion:
    file: str
    qualified_name: str
    line: int
    symbol_type: str
    parameters: list[str]
    attributes: list[str]
    has_returns: bool
    has_yields: bool
    raises: list[str]
    warnings: list[str]


@dataclass(frozen=True)
class SkippedSymbol:
    file: str
    qualified_name: str
    line: int
    symbol_type: str
    reason: str


@dataclass
class RunReport:
    root: str
    mode: str
    files_scanned: int = 0
    files_changed: int = 0
    inserted: list[PlannedInsertion] = field(default_factory=list)
    skipped: list[SkippedSymbol] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)


class ScopedFlowVisitor(ast.NodeVisitor):
    def __init__(self, root: FunctionNode) -> None:
        self.root = root
        self.returns: list[ast.Return] = []
        self.yields: list[ast.Yield | ast.YieldFrom] = []
        self.raises: list[ast.Raise] = []
        self.asserts: list[ast.Assert] = []
        self.warning_calls: list[ast.Call] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node is self.root:
            self.generic_visit(node)

    def visit_AsyncFunctionDef(
        self,
        node: ast.AsyncFunctionDef,
    ) -> None:
        if node is self.root:
            self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
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
        if dotted_name(node.func) in {"warnings.warn", "warn"}:
            self.warning_calls.append(node)

        self.generic_visit(node)


def is_ignored(path: Path) -> bool:

    if path.name in IGNORED_FILES:
        return True

    return any(part in IGNORED_DIRECTORIES for part in path.parts)


def python_files(root: Path) -> list:

    if root.is_file():
        if root.suffix == ".py" and not is_ignored(root):
            return [root]

        return []

    return sorted(
        path for path in root.rglob("*.py") if path.is_file() and not is_ignored(path)
    )


def qualified_name(
    stack: Sequence[str],
    name: str,
) -> str:

    if stack:
        return ".".join([*stack, name])

    return name


def dotted_name(node: ast.AST) -> str:

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

    if node is None:
        return fallback

    try:
        return ast.unparse(node)
    except (AttributeError, ValueError):
        return fallback


def has_overload_decorator(node: FunctionNode) -> bool:

    return any(
        dotted_name(decorator).endswith("overload") for decorator in node.decorator_list
    )


def is_dataclass(node: ast.ClassDef) -> bool:

    return any(
        dotted_name(decorator).split(".")[-1] == "dataclass"
        for decorator in node.decorator_list
    )


def exception_name(
    expression: ast.expr | None,
) -> str | None:

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

    if len(call.args) >= 2:
        category = exception_name(call.args[1])

        if category:
            return category

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

    parameters: list[tuple[str, ast.arg]] = []

    positional = [
        *node.args.posonlyargs,
        *node.args.args,
    ]

    for argument in positional:
        if argument.arg not in {"self", "cls"}:
            parameters.append((argument.arg, argument))

    if node.args.vararg is not None:
        parameters.append(
            (
                f"*{node.args.vararg.arg}",
                node.args.vararg,
            )
        )

    for argument in node.args.kwonlyargs:
        parameters.append((argument.arg, argument))

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


def is_classvar(annotation: ast.expr | None) -> bool:

    if annotation is None:
        return False

    return "ClassVar" in annotation_text(
        annotation,
        fallback="",
    )


def dataclass_parameters(
    node: ast.ClassDef,
) -> list[tuple[str, str]]:

    parameters: list[tuple[str, str]] = []

    for statement in node.body:
        if not isinstance(statement, ast.AnnAssign):
            continue

        if not isinstance(statement.target, ast.Name):
            continue

        if is_classvar(statement.annotation):
            continue

        name = statement.target.id

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

    if is_dataclass(node):
        return dataclass_parameters(node)

    return initializer_parameters(node)


def class_attributes(
    node: ast.ClassDef,
) -> list[tuple[str, str]]:

    if is_dataclass(node):
        return []

    attributes: list[tuple[str, str]] = []

    for statement in node.body:
        if not isinstance(statement, ast.AnnAssign):
            continue

        if not isinstance(statement.target, ast.Name):
            continue

        if is_classvar(statement.annotation):
            continue

        name = statement.target.id

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

    result: list[ast.Return] = []

    for statement in visitor.returns:
        value = statement.value

        if value is None:
            continue

        if isinstance(value, ast.Constant) and value.value is None:
            continue

        result.append(statement)

    return result


def yielded_type(
    annotation: ast.expr | None,
) -> str:

    if annotation is None:
        return "Any"

    if isinstance(annotation, ast.Subscript):
        container = dotted_name(annotation.value).split(".")[-1]

        if container in {
            "Iterator",
            "Iterable",
            "AsyncIterator",
            "AsyncIterable",
        }:
            return annotation_text(annotation.slice)

        if container == "Generator":
            slice_node = annotation.slice

            if isinstance(slice_node, ast.Tuple) and slice_node.elts:
                return annotation_text(slice_node.elts[0])

    return annotation_text(annotation)


def analyze_flow(
    node: FunctionNode,
) -> ScopedFlowVisitor:

    visitor = ScopedFlowVisitor(node)
    visitor.visit(node)

    return visitor


def build_function_docstring(
    node: FunctionNode,
) -> tuple[str, dict[str, Any]]:

    flow = analyze_flow(node)
    parameters = function_parameters(node)
    returns = meaningful_returns(flow)
    yields = flow.yields

    raised = {
        name
        for statement in flow.raises
        if (name := exception_name(statement.exc)) is not None
    }

    if flow.asserts:
        raised.add("AssertionError")

    warnings = {warning_name(call) for call in flow.warning_calls}

    lines = [FUNCTION_SUMMARY]

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

    has_returns = (
        bool(returns)
        and not has_yields
        and node.name
        not in {
            "__init__",
            "__post_init__",
        }
    )

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

    parameters = class_parameters(node)
    attributes = class_attributes(node)

    lines = [CLASS_SUMMARY]

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

    if isinstance(node, ast.ClassDef):
        return build_class_docstring(node)

    return build_function_docstring(node)


def symbol_type(node: DocumentableNode) -> str:

    if isinstance(node, ast.ClassDef):
        return "class"

    if isinstance(node, ast.AsyncFunctionDef):
        return "async function"

    return "function"


def leading_whitespace(line: str) -> str:

    return line[: len(line) - len(line.lstrip())]


def body_indent(
    node: DocumentableNode,
    source_lines: Sequence[str],
) -> str:

    if node.body:
        first_statement = node.body[0]
        line = source_lines[first_statement.lineno - 1]
        detected = leading_whitespace(line)

        if len(detected.expandtabs()) > node.col_offset:
            return detected

    definition_line = source_lines[node.lineno - 1]

    return leading_whitespace(definition_line) + " " * INDENT_WIDTH


def render_docstring(
    doc: str,
    indent: str,
    newline: str,
) -> str:

    if '"""' in doc:
        raise ValueError("Generated documentation unexpectedly contains triple quotes.")

    lines = doc.splitlines()

    rendered = [f'{indent}"""{lines[0]}']

    for line in lines[1:]:
        if line:
            rendered.append(f"{indent}{line}")
        else:
            rendered.append(indent)

    rendered[-1] += '"""'

    return newline.join(rendered) + newline


def line_start_offsets(
    source: str,
) -> list:

    offsets = [0]

    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))

    return offsets


def insertion_offset(
    node: DocumentableNode,
    offsets: Sequence[int],
) -> int:

    if not node.body:
        raise ValueError(f"Symbol {node.name!r} has no body.")

    first_statement = node.body[0]

    return offsets[first_statement.lineno - 1]


def iter_symbols(
    body: Sequence[ast.stmt],
    stack: Sequence[str] = (),
) -> Iterable[tuple[DocumentableNode, str]]:

    for node in body:
        if isinstance(node, ast.ClassDef):
            qname = qualified_name(
                stack,
                node.name,
            )

            yield node, qname

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

            yield from iter_symbols(
                node.body,
                [*stack, node.name],
            )


def apply_edits(
    source: str,
    edits: Sequence[SourceEdit],
) -> str:

    updated = source

    for edit in sorted(
        edits,
        key=lambda item: item.offset,
        reverse=True,
    ):
        updated = updated[: edit.offset] + edit.text + updated[edit.offset :]

    return updated


def existing_docstring_digest(
    tree: ast.Module,
) -> dict[str, str]:

    result: dict[str, str] = {}

    for node, qname in iter_symbols(tree.body):
        doc = ast.get_docstring(
            node,
            clean=False,
        )

        if doc is None:
            continue

        key = f"{symbol_type(node)}::{qname}"

        result[key] = hashlib.sha256(doc.encode("utf-8")).hexdigest()

    return result


def verify_existing_docstrings(
    before: dict[str, str],
    after_tree: ast.Module,
) -> None:

    after = existing_docstring_digest(after_tree)

    for key, expected_digest in before.items():
        observed_digest = after.get(key)

        if observed_digest != expected_digest:
            raise RuntimeError(
                f"Existing docstring changed or disappeared for {key!r}."
            )


def detect_newline(source: str) -> str:

    if "\r\n" in source:
        return "\r\n"

    return "\n"


def make_diff(
    path: Path,
    before: str,
    after: str,
) -> str:

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

    temporary = path.with_name(f".{path.name}.docstrings.tmp")

    try:
        temporary.write_text(
            text,
            encoding=encoding,
            newline="",
        )

        temporary.replace(path)

    finally:
        if temporary.exists():
            temporary.unlink()


def parse_source(
    path: Path,
) -> tuple[str, str, ast.Module]:

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

    if isinstance(
        node,
        (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
        ),
    ):
        if has_overload_decorator(node):
            return "overload declaration"

        if not include_optional_methods and node.name in OPTIONAL_UNDOCUMENTED_METHODS:
            return "optional special method"

    return None


def plan_file(
    path: Path,
    source: str,
    tree: ast.Module,
    *,
    include_optional_methods: bool,
) -> tuple[
    str,
    list[PlannedInsertion],
    list[SkippedSymbol],
]:

    source_lines = source.splitlines()
    offsets = line_start_offsets(source)
    newline = detect_newline(source)

    existing_digests = existing_docstring_digest(tree)

    edits: list[SourceEdit] = []
    insertions: list[PlannedInsertion] = []
    skipped: list[SkippedSymbol] = []

    for node, qname in iter_symbols(tree.body):
        existing_docstring = ast.get_docstring(
            node,
            clean=False,
        )

        if existing_docstring is not None:
            continue

        reason = should_skip(
            node,
            include_optional_methods=(include_optional_methods),
        )

        if reason is not None:
            skipped.append(
                SkippedSymbol(
                    file=str(path),
                    qualified_name=qname,
                    line=node.lineno,
                    symbol_type=symbol_type(node),
                    reason=reason,
                )
            )
            continue

        doc, metadata = build_docstring(node)

        indent = body_indent(
            node,
            source_lines,
        )

        edits.append(
            SourceEdit(
                offset=insertion_offset(
                    node,
                    offsets,
                ),
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

    updated = apply_edits(
        source,
        edits,
    )

    if not edits:
        return updated, insertions, skipped

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

    verify_existing_docstrings(
        existing_digests,
        updated_tree,
    )

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

        node = updated_symbols.get(key)

        if node is None:
            raise RuntimeError(
                f"Generated symbol disappeared: {insertion.qualified_name}"
            )

        generated_docstring = ast.get_docstring(
            node,
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
) -> tuple[
    list[PlannedInsertion],
    list[SkippedSymbol],
    bool,
]:

    source, encoding, tree = parse_source(path)

    updated, inserted, skipped = plan_file(
        path,
        source,
        tree,
        include_optional_methods=(include_optional_methods),
    )

    changed = updated != source

    if not changed:
        return inserted, skipped, False

    if show_diff:
        print(
            make_diff(
                path,
                source,
                updated,
            ),
            end="",
        )

    if apply:
        if backup:
            backup_path = path.with_suffix(path.suffix + ".bak")

            if backup_path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite existing backup: {backup_path}"
                )

            shutil.copy2(
                path,
                backup_path,
            )

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

    parser = argparse.ArgumentParser(
        description=(
            "Insert NumPy-style docstring scaffolds "
            "into undocumented classes, functions, "
            "and methods. Existing docstrings are "
            "never modified."
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
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="Path for the JSON run report.",
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:

    args = build_parser().parse_args(argv)
    root: Path = args.root

    if not root.exists():
        print(
            f"ERROR: Path does not exist: {root}",
            file=sys.stderr,
        )
        return 2

    files = python_files(root)

    if not files:
        print(
            f"ERROR: No Python files found under {root}",
            file=sys.stderr,
        )
        return 2

    report = RunReport(
        root=str(root),
        mode=("apply" if args.apply else "dry-run"),
        files_scanned=len(files),
    )

    for path in files:
        try:
            inserted, skipped, changed = process_file(
                path,
                apply=args.apply,
                backup=not args.no_backup,
                show_diff=not args.no_diff,
                include_optional_methods=(args.include_optional_methods),
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

    write_report(
        report,
        args.report,
    )

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

    print()
    print("=" * 72)
    print("MISSING DOCSTRING INSERTION SUMMARY")
    print("=" * 72)
    print(f"Mode:                 {report.mode}")
    print(f"Root:                 {report.root}")
    print(f"Files scanned:        {report.files_scanned}")
    print(f"Files changed:        {report.files_changed}")
    print(f"{action + ':':<22}{len(report.inserted)}")
    print(f"Classes:              {class_count}")
    print(f"Functions/methods:    {function_count}")
    print(f"Skipped:              {len(report.skipped)}")
    print(f"Errors:               {len(report.errors)}")
    print(f"JSON report:          {args.report}")

    if not args.apply and report.inserted:
        print()
        print("Dry run only. Review the diff, then re-run with --apply.")

    if report.errors:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
