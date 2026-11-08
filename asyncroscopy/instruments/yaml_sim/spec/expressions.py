"""The small expression language of a YAML instrument file: names and parsing checked at load, safe evaluation."""

from __future__ import annotations

import ast
import math
import re
from typing import Any

NAME = re.compile(r"^[a-z][a-z0-9_]*$")
FUNCTIONS = {
    "abs": abs, "min": min, "max": max, "sqrt": math.sqrt, "exp": math.exp, "log": math.log,
    "sin": math.sin, "cos": math.cos, "radians": math.radians, "degrees": math.degrees,
}


OPERATORS = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Call, ast.Name, ast.Load, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or,
    ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq, ast.IfExp,
)


class SpecError(ValueError):
    """The YAML file does not describe a valid instrument; the message names the place."""


class Refused(ValueError):
    """A command or acquisition was refused: an argument out of range or a failed requirement."""


def _check_keys(where: str, mapping: Any, allowed: set[str]) -> dict:
    if not isinstance(mapping, dict):
        raise SpecError(f"{where}: expected a mapping, got {type(mapping).__name__}")
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise SpecError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(allowed)}")
    return mapping


def _check_name(where: str, name: Any) -> str:
    if not isinstance(name, str) or not NAME.match(name):
        raise SpecError(f"{where}: {name!r} is not a valid name (lowercase letters, digits and _, starting with a letter)")
    return name


def _compile(where: str, text: Any, names: set[str]) -> ast.Expression | None:
    """An expression's tree, or None for a value that is not a string (a literal)."""
    if not isinstance(text, str):
        return None
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise SpecError(f"{where}: {text!r} is not an expression ({exc.msg})") from None
    for node in ast.walk(tree):
        if not isinstance(node, OPERATORS):
            raise SpecError(f"{where}: {text!r} uses {type(node).__name__}, which expressions do not allow")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS):
            raise SpecError(f"{where}: {text!r} calls something other than {sorted(FUNCTIONS)}")
        if isinstance(node, ast.Name) and node.id not in names and node.id not in FUNCTIONS:
            raise SpecError(f"{where}: {text!r} names {node.id!r}, which is not a setting or the command's input; known: {sorted(names)}")
    return tree


def evaluate(tree: ast.Expression, values: dict[str, Any]) -> Any:
    """Evaluate a tree accepted by ``_compile``."""

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return values[node.id] if node.id in values else FUNCTIONS[node.id]
        if isinstance(node, ast.UnaryOp):
            operand = ev(node.operand)
            return {ast.USub: lambda v: -v, ast.UAdd: lambda v: +v, ast.Not: lambda v: not v}[type(node.op)](operand)
        if isinstance(node, ast.BinOp):
            left, right = ev(node.left), ev(node.right)
            ops = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
                   ast.Div: lambda a, b: a / b, ast.Pow: lambda a, b: a ** b, ast.Mod: lambda a, b: a % b}
            return ops[type(node.op)](left, right)
        if isinstance(node, ast.BoolOp):
            results = [ev(v) for v in node.values]
            return all(results) if isinstance(node.op, ast.And) else any(results)
        if isinstance(node, ast.Compare):
            left = ev(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                right = ev(comparator)
                ok = {ast.Lt: left < right, ast.LtE: left <= right, ast.Gt: left > right, ast.GtE: left >= right,
                      ast.Eq: left == right, ast.NotEq: left != right}[type(op)]
                if not ok:
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            return ev(node.body) if ev(node.test) else ev(node.orelse)
        if isinstance(node, ast.Call):
            return FUNCTIONS[node.func.id](*[ev(a) for a in node.args])
        raise TypeError(type(node).__name__)

    return ev(tree)
