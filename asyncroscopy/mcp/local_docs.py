"""Describe a device command from the local source when the device sends no description.

A device server running older code or a plain ``tango.server.command`` sends no ``doc_in`` or ``doc_out``.
The tool would then be known by name only, and a model could not tell what arguments it takes.

The MCP server then looks up the same class name in the local source and reads the docstring. The source
is parsed and never imported, so no vendor SDK is needed. A description sent by the device always wins.
"""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path

SOURCE_ROOTS = (Path(__file__).resolve().parents[1] / "instruments", Path(__file__).resolve().parents[1] / "jobs")


def _is_command(decorator: ast.expr) -> bool:
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    return (isinstance(target, ast.Name) and target.id == "command") or (isinstance(target, ast.Attribute) and target.attr == "command")


def _doc_in(decorators: list[ast.expr]) -> str:
    for decorator in decorators:
        if isinstance(decorator, ast.Call) and _is_command(decorator):
            for keyword in decorator.keywords:
                if keyword.arg == "doc_in" and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                    return keyword.value.value
    return ""


def _summary(function: ast.FunctionDef) -> str:
    """The docstring's first paragraph on one line, as ``asyncroscopy.utils.commands.summary`` gives it."""
    return " ".join((ast.get_docstring(function) or "").split("\n\n", 1)[0].split())


@lru_cache(maxsize=1)
def _classes() -> dict[str, tuple[list[str], dict[str, tuple[str, str]]]]:
    """Every class in the instrument source, as name -> (base names, {command: (summary, doc_in)})."""
    found: dict[str, tuple[list[str], dict[str, tuple[str, str]]]] = {}
    for root in SOURCE_ROOTS:
        for path in sorted(root.rglob("*.py")) if root.is_dir() else []:
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                bases = [b.id if isinstance(b, ast.Name) else b.attr for b in node.bases if isinstance(b, (ast.Name, ast.Attribute))]
                commands = {f.name: (_summary(f), _doc_in(f.decorator_list)) for f in node.body
                            if isinstance(f, ast.FunctionDef) and any(_is_command(d) for d in f.decorator_list)}
                found.setdefault(node.name, (bases, commands))
    return found


def describe(dev_class: str, command: str) -> tuple[str, str]:
    """Return the command summary and its ``doc_in`` from the local source. Each is "" when not found.

    A command the class does not declare is looked for in its base classes, nearest first."""
    classes, seen, queue = _classes(), set(), [dev_class]
    while queue:
        name = queue.pop(0)
        if name in seen or name not in classes:
            continue
        seen.add(name)
        bases, commands = classes[name]
        if command in commands and any(commands[command]):
            return commands[command]
        queue.extend(bases)
    return "", ""
