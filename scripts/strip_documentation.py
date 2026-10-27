import ast
import io
import pathlib
import tokenize


# Comments containing these tags should NEVER be removed
# These are directives for linters, type checkers, and formatters
PRESERVE_TAGS = {
    "noqa",  # Disable linting rules for this line
    "type:",  # Type hint directive
    "pylint:",  # Pylint configuration
    "pyright:",  # Pyright type checker directive
    "mypy:",  # Mypy type checker directive
    "pragma:",  # Generic pragma directive
    "fmt:",  # Formatting directive
    "isort:",  # Import sorting directive
    "flake8:",  # Flake8 linter directive
    "yapf:",  # YAPF formatter directive
    "autopep8:",  # Autopep8 formatter directive
}


def should_preserve_comment(comment_text):
    """
    Check if a comment should be kept.
    Returns True if the comment contains any preservation tag (linter directives, etc).
    These comments must be kept because they control tool behavior.
    """
    # Remove the '#' and surrounding whitespace to get the actual comment text
    text = comment_text.lstrip("#").strip()

    # Check if any preservation tag is in this comment
    return any(tag in text for tag in PRESERVE_TAGS)


# Process all Python files in the current directory and subdirectories
for path in pathlib.Path(".").rglob("*.py"):
    try:
        # Read the source code
        src = path.read_text(encoding="utf-8")

        # Parse the source to find docstrings (which we'll remove separately)
        tree = ast.parse(src)
        remove_lines = set()

        # Find all docstrings in the module, classes, and functions
        # Docstrings are the first statement in a body and are string constants
        for node in ast.walk(tree):
            if (
                isinstance(
                    node,
                    (
                        ast.Module,  # Module-level docstring
                        ast.FunctionDef,  # Function docstring
                        ast.AsyncFunctionDef,  # Async function docstring
                        ast.ClassDef,  # Class docstring
                    ),
                )
                and node.body  # Has at least one statement
                and isinstance(
                    node.body[0], ast.Expr
                )  # First statement is an expression
                and isinstance(
                    getattr(node.body[0], "value", None), ast.Constant
                )  # Expression is a constant
                and isinstance(
                    node.body[0].value.value, str
                )  # Constant is a string (docstring)
            ):
                # Mark all lines that make up this docstring for removal
                remove_lines.update(
                    range(node.body[0].lineno, node.body[0].end_lineno + 1)
                )

        # Remove the docstring lines from the source
        src = "".join(
            line
            for i, line in enumerate(src.splitlines(True), 1)
            if i not in remove_lines
        )

        # Now tokenize the source to process comments
        out = []  # Tokens to keep
        tokgen = tokenize.generate_tokens(io.StringIO(src).readline)

        for tok in tokgen:
            # If this token is a comment, decide whether to keep it
            if tok.type == tokenize.COMMENT:
                # Skip this comment if it doesn't contain a preservation tag
                # (should_preserve_comment returns True if we SHOULD keep it)
                if not should_preserve_comment(tok.string):
                    # Skip this comment - don't add it to output
                    continue

            # Add this token to the output (either non-comment or preserved comment)
            out.append(tok)

        # Convert tokens back to source code and write it back to the file
        path.write_text(tokenize.untokenize(out), encoding="utf-8")
        print("Processed", path)

    except Exception as e:
        # If anything goes wrong with this file, print error and continue to next file
        print("FAILED", path, e)
