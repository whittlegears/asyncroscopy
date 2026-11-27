"""The values a node is given and the conditions an edge is decided by.

A value written exactly ``"{inputs.name}"`` or ``"{node.output}"`` is a reference. When the node runs it is replaced
by what it names and keeps its type. A condition is a small expression over the same names, such as
``tile.remaining > 0``. It may use comparisons, and, or, not, numbers and text.
"""

from __future__ import annotations

import ast
import math
import re
from typing import Any

from .blocks import TYPES

REFERENCE = re.compile(r"^\{([A-Za-z_][\w.]*)\}$")
_ALLOWED = (
    ast.Expression, ast.BoolOp, ast.UnaryOp, ast.Compare, ast.BinOp, ast.Name, ast.Attribute, ast.Load, ast.Constant,
    ast.And, ast.Or, ast.Not, ast.USub, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq, ast.Add, ast.Sub, ast.Mult, ast.Div,
)


def references(value: Any) -> list[str]:
    """Every reference found in a value, without its braces."""
    if isinstance(value, str):
        match = REFERENCE.match(value)
        return [match.group(1)] if match else []
    if isinstance(value, list):
        return [r for v in value for r in references(v)]
    if isinstance(value, dict):
        return [r for v in value.values() for r in references(v)]
    return []


def lookup(path: str, inputs: dict, outputs: dict) -> Any:
    head, *rest = path.split(".")
    value: Any = inputs if head == "inputs" else outputs.get(head)
    if head != "inputs" and head not in outputs:
        raise KeyError(f"{{{path}}}: node {head} has not run yet")
    for part in rest:
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            raise KeyError(f"{{{path}}}: there is no {part} there")
    return value


def fill(value: Any, inputs: dict, outputs: dict) -> Any:
    """Return ``value`` with every reference replaced by what it names."""
    if isinstance(value, str):
        match = REFERENCE.match(value)
        return lookup(match.group(1), inputs, outputs) if match else value
    if isinstance(value, list):
        return [fill(v, inputs, outputs) for v in value]
    if isinstance(value, dict):
        return {k: fill(v, inputs, outputs) for k, v in value.items()}
    return value


def condition_names(text: str) -> list[str]:
    """The dotted names a condition reads. Raises ValueError if it uses anything we do not allow."""
    tree = ast.parse(str(text), mode="eval")
    names = []
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise ValueError(f"{type(node).__name__} is not allowed in a condition")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Attribute, ast.Name)):
            dotted = _dotted(node)
            if dotted and not any(n.startswith(dotted + ".") for n in names):
                names.append(dotted)
    whole = {n for n in names if not any(o != n and o.startswith(n + ".") for o in names)}
    return sorted(whole)


def _dotted(node: ast.AST) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return ".".join([node.id, *reversed(parts)])
    return None


def evaluate(text: str, inputs: dict, outputs: dict) -> Any:
    """Work out a condition. It is checked by ``condition_names`` first."""

    def value(node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.Name, ast.Attribute)):
            return lookup(_dotted(node) or "", inputs, outputs)
        if isinstance(node, ast.UnaryOp):
            operand = value(node.operand)
            return (not operand) if isinstance(node.op, ast.Not) else -operand
        if isinstance(node, ast.BoolOp):
            values = [value(v) for v in node.values]
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.BinOp):
            a, b = value(node.left), value(node.right)
            return {ast.Add: lambda: a + b, ast.Sub: lambda: a - b, ast.Mult: lambda: a * b, ast.Div: lambda: a / b}[type(node.op)]()
        if isinstance(node, ast.Compare):
            left = value(node.left)
            for op, right_node in zip(node.ops, node.comparators):
                right = value(right_node)
                ok = {ast.Lt: lambda: left < right, ast.LtE: lambda: left <= right, ast.Gt: lambda: left > right,
                      ast.GtE: lambda: left >= right, ast.Eq: lambda: left == right, ast.NotEq: lambda: left != right}[type(op)]()
                if not ok:
                    return False
                left = right
            return True
        raise ValueError(f"{type(node).__name__} is not allowed in a condition")

    condition_names(text)
    return value(ast.parse(str(text), mode="eval").body)


def check_inputs(workflow: str, declared: dict, given: dict | None) -> dict:
    """Return a run's inputs, which are the declared defaults with ``given`` on top.

    Raises ValueError naming what is wrong. An input outside its range is refused and never clipped, because the
    range is what a reviewer approved.
    """
    given = dict(given or {})
    unknown = sorted(set(given) - set(declared))
    if unknown:
        raise ValueError(f"{workflow} takes inputs {sorted(declared)}, not {unknown}")
    values = {}
    for name, port in declared.items():
        port = port or {}
        if name not in given and "default" not in port:
            raise ValueError(f"{workflow} needs input {name}")
        value = given.get(name, port.get("default"))
        kind = port.get("type", "any")
        if kind == "number" and isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                pass
        if kind == "integer" and isinstance(value, (str, float)):
            try:
                as_float = float(value)
            except ValueError:
                as_float = math.nan
            if math.isfinite(as_float) and as_float == int(as_float):
                value = int(as_float)
        if (isinstance(value, bool) and kind in ("number", "integer")) or not isinstance(value, TYPES.get(kind, (object,))):
            raise ValueError(f"input {name} must be a {kind}, not {value!r}")
        if kind in ("number", "integer") and not math.isfinite(value):
            # NaN and infinity can pass the range checks below
            raise ValueError(f"input {name} must be a finite {kind}, not {value!r}")
        for bound, bad in (("min", lambda v, b: v < b), ("max", lambda v, b: v > b)):
            if bound in port and bad(value, port[bound]):
                raise ValueError(f"input {name} = {value} is outside its range ({bound} {port[bound]})")
        values[name] = value
    return values
