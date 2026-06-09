import ast
import io
import pathlib
import tokenize

for path in pathlib.Path(".").rglob("*.py"):
    try:
        src = path.read_text(encoding="utf-8")

                         
        tree = ast.parse(src)
        remove_lines = set()

        for node in ast.walk(tree):
            if (
                isinstance(
                    node,
                    (
                        ast.Module,
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                        ast.ClassDef,
                    ),
                )
                and node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(getattr(node.body[0], "value", None), ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                remove_lines.update(
                    range(node.body[0].lineno, node.body[0].end_lineno + 1)
                )

                                
        src = "".join(
            line
            for i, line in enumerate(src.splitlines(True), 1)
            if i not in remove_lines
        )

                         
        out = []
        tokgen = tokenize.generate_tokens(io.StringIO(src).readline)

        for tok in tokgen:
            if tok.type == tokenize.COMMENT:
                continue
            out.append(tok)

        path.write_text(tokenize.untokenize(out), encoding="utf-8")
        print("Processed", path)

    except Exception as e:
        print("FAILED", path, e)
