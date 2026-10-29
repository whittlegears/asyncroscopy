"""An instrument described in one YAML file: settings, commands and simulated acquisitions.

``load_spec(path)`` reads and checks the file and returns an ``InstrumentSpec``;
``SimInstrument(spec)`` holds the settings' current values and runs commands and
acquisitions. Nothing here imports Tango, so the behaviour is tested without a
device server; ``asyncroscopy.instruments.yaml_sim.device`` turns it into one.

The file format, with every key (see examples/xrd.yaml for a working instrument)::

    description: One line an agent reads about the instrument.
    seed: 0                              # random seed for the simulated noise
    settings:
      exposure_s:                        # -> get_exposure_s, set_exposure_s and an attribute
        default: 1.0
        min: 0.01                        # optional bounds, checked on every set
        max: 10
        units: s
        type: float                      # float (default), int, bool or str
        settable: true                   # false: get_ only
        doc: Counting time per point.
    commands:
      xray_off:
        doc: Switch the X-ray source off.
        input: {name: power_w, type: float, min: 0, max: 1600, units: W}   # optional, one argument
        requires: ["tube_power_w > 0"]   # optional checks, refused with the expression
        sets: {tube_power_w: 0}          # settings to change; values are expressions
        returns: "X-rays off"            # a string, a number, or a mapping (sent as JSON)
        duration_s: 0.5                  # optional simulated time
    acquisitions:
      acquire_scan:                      # names start with acquire; returns a data key
        doc: Measure a diffractogram.
        kind: spectrum                   # spectrum (1-D), image (2-D) or value
        ...

Strings in ``sets``, ``returns``, ``requires``, ``duration_s`` and the numbers of an
acquisition are expressions over the settings (and the command's input): numbers,
names, + - * / ** %, comparisons, and/or/not, and abs, min, max, sqrt, exp, log,
sin, cos, radians, degrees. A plain string that is not an expression is returned as text.
"""

from __future__ import annotations

import ast
import json
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

TYPES = {"float": float, "int": int, "bool": bool, "str": str}
NAME = re.compile(r"^[a-z][a-z0-9_]*$")
MAX_DURATION_S = 25.0  # the MCP server stops waiting for a command after 30 s
MAX_POINTS = 1_000_000
FUNCTIONS = {
    "abs": abs, "min": min, "max": max, "sqrt": math.sqrt, "exp": math.exp, "log": math.log,
    "sin": math.sin, "cos": math.cos, "radians": math.radians, "degrees": math.degrees,
}
OPERATORS = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Call, ast.Name, ast.Load, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or,
    ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq, ast.IfExp,
)

SETTING_KEYS = {"default", "min", "max", "units", "type", "settable", "doc"}
INPUT_KEYS = {"name", "type", "min", "max", "units", "doc"}
COMMAND_KEYS = {"doc", "input", "requires", "sets", "returns", "duration_s"}
ACQUISITION_KEYS = {"doc", "kind", "requires", "duration_s", "detector", "axis", "peaks", "background", "noise", "shape", "pixel_size_m", "spots", "values"}
AXIS_KEYS = {"start", "stop", "points", "label", "units"}
PEAK_KEYS = {"position", "height", "width"}
SPOT_KEYS = {"count", "radius_px", "height"}
TOP_KEYS = {"description", "seed", "settings", "commands", "acquisitions"}
KINDS = {"spectrum", "image", "value"}
NOISES = {"poisson", "gaussian", "none"}


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


@dataclass
class Quantity:
    """A setting, or a command's input: a typed value with optional bounds."""

    name: str
    type: type = float
    default: Any = None
    min: float | None = None
    max: float | None = None
    units: str = ""
    doc: str = ""
    settable: bool = True

    def coerce(self, value: Any) -> Any:
        try:
            if self.type is bool and isinstance(value, str):
                lowered = value.strip().lower()
                if lowered not in {"true", "false", "1", "0"}:
                    raise ValueError(value)
                value = lowered in {"true", "1"}
            value = self.type(value)
        except (TypeError, ValueError):
            raise Refused(f"{self.name} must be a {self.type.__name__}, not {value!r}") from None
        if self.type in (int, float):
            if isinstance(value, float) and not math.isfinite(value):
                raise Refused(f"{self.name} must be a finite number, not {value}")
            if self.min is not None and value < self.min or self.max is not None and value > self.max:
                low = "-inf" if self.min is None else f"{self.min:g}"
                high = "inf" if self.max is None else f"{self.max:g}"
                raise Refused(f"{self.name} = {value:g} {self.units}".rstrip() + f" is outside [{low}, {high}]")
        return value

    def describe(self) -> str:
        """The doc with units and bounds, e.g. "Counting time per point, in s, from 0.01 to 10."."""
        text = (self.doc or self.name.replace("_", " ")).rstrip(".")
        if self.units:
            text += f", in {self.units}"
        if self.min is not None and self.max is not None:
            text += f", from {self.min:g} to {self.max:g}"
        elif self.min is not None:
            text += f", at least {self.min:g}"
        elif self.max is not None:
            text += f", at most {self.max:g}"
        return text + "."


def _quantity(where: str, name: str, raw: Any, allowed: set[str]) -> Quantity:
    raw = _check_keys(where, raw if raw is not None else {}, allowed)
    type_name = raw.get("type", "float")
    if type_name not in TYPES:
        raise SpecError(f"{where}.type: {type_name!r} is not one of {sorted(TYPES)}")
    quantity = Quantity(name=name, type=TYPES[type_name], units=str(raw.get("units", "")), doc=str(raw.get("doc", "")).strip())
    for bound in ("min", "max"):
        if raw.get(bound) is not None:
            if quantity.type not in (int, float) or not isinstance(raw[bound], (int, float)) or isinstance(raw[bound], bool):
                raise SpecError(f"{where}.{bound}: bounds need a number on a float or int {'setting' if 'settable' in allowed else 'input'}")
            setattr(quantity, bound, float(raw[bound]))
    if quantity.min is not None and quantity.max is not None and quantity.min > quantity.max:
        raise SpecError(f"{where}: min {quantity.min} is above max {quantity.max}")
    if "settable" in allowed:
        quantity.settable = bool(raw.get("settable", True))
        if "default" not in raw:
            raise SpecError(f"{where}: a setting needs a default")
        try:
            quantity.default = quantity.coerce(raw["default"])
        except Refused as exc:
            raise SpecError(f"{where}.default: {exc}") from None
    return quantity


@dataclass
class Command:
    name: str
    doc: str
    input: Quantity | None = None
    requires: list[tuple[str, ast.Expression]] = field(default_factory=list)
    sets: dict[str, tuple[Any, ast.Expression | None]] = field(default_factory=dict)
    returns: Any = None
    returns_trees: Any = None
    duration: tuple[Any, ast.Expression | None] = (0.0, None)


@dataclass
class Acquisition:
    name: str
    doc: str
    kind: str
    raw: dict
    trees: dict[str, Any]
    requires: list[tuple[str, ast.Expression]] = field(default_factory=list)
    duration: tuple[Any, ast.Expression | None] = (0.0, None)


@dataclass
class InstrumentSpec:
    path: Path
    description: str
    seed: int
    settings: dict[str, Quantity]
    commands: dict[str, Command]
    acquisitions: dict[str, Acquisition]

    def command_names(self) -> list[str]:
        names = [f"get_{n}" for n in self.settings] + [f"set_{n}" for n, q in self.settings.items() if q.settable]
        return names + list(self.commands) + list(self.acquisitions)


def _requires(where: str, raw: Any, names: set[str]) -> list[tuple[str, ast.Expression]]:
    if raw is None:
        return []
    if not isinstance(raw, list) or not all(isinstance(r, str) for r in raw):
        raise SpecError(f"{where}: requires is a list of expressions, e.g. [\"tube_power_w > 0\"]")
    return [(text, _compile(f"{where}.requires", text, names)) for text in raw]


def _duration(where: str, raw: Any, names: set[str]) -> tuple[Any, ast.Expression | None]:
    raw = 0.0 if raw is None else raw
    tree = _compile(f"{where}.duration_s", raw, names)
    if tree is None and (not isinstance(raw, (int, float)) or isinstance(raw, bool) or not 0 <= raw <= MAX_DURATION_S):
        raise SpecError(f"{where}.duration_s: a number of seconds from 0 to {MAX_DURATION_S}, or an expression")
    return raw, tree


def _returns_trees(where: str, raw: Any, names: set[str]) -> Any:
    if isinstance(raw, dict):
        return {k: _returns_trees(f"{where}.{k}", v, names) for k, v in raw.items()}
    if isinstance(raw, list):
        return [_returns_trees(f"{where}[{i}]", v, names) for i, v in enumerate(raw)]
    if isinstance(raw, str):
        try:
            return _compile(where, raw, names)
        except SpecError:
            return None  # plain text, returned as written
    return None


def _command(name: str, raw: Any, settings: dict[str, Quantity]) -> Command:
    where = f"commands.{name}"
    raw = _check_keys(where, raw, COMMAND_KEYS)
    if "doc" not in raw:
        raise SpecError(f"{where}: needs a doc (the agent reads it as the tool's description)")
    quantity = _quantity(f"{where}.input", "", raw["input"], INPUT_KEYS) if raw.get("input") is not None else None
    names = set(settings)
    if quantity is not None:
        quantity.name = _check_name(f"{where}.input.name", raw["input"].get("name"))
        if quantity.name in settings:
            raise SpecError(f"{where}.input.name: {quantity.name!r} is also a setting; pick another name")
        names.add(quantity.name)
    sets = {}
    for target, value in _check_keys(f"{where}.sets", raw.get("sets") or {}, set(settings)).items():
        sets[target] = (value, _compile(f"{where}.sets.{target}", value, names))
    if (name.startswith(("get_", "list_"))) and (sets or raw.get("duration_s")):
        raise SpecError(f"{where}: get_ and list_ commands must not change anything (the safety policy never holds them); rename it or drop sets/duration_s")
    return Command(
        name=name,
        doc=str(raw["doc"]).strip(),
        input=quantity,
        requires=_requires(where, raw.get("requires"), names),
        sets=sets,
        returns=raw.get("returns"),
        returns_trees=_returns_trees(f"{where}.returns", raw.get("returns"), names),
        duration=_duration(where, raw.get("duration_s"), names),
    )


def _acquisition(name: str, raw: Any, settings: dict[str, Quantity]) -> Acquisition:
    where = f"acquisitions.{name}"
    if not name.startswith("acquire"):
        raise SpecError(f"{where}: acquisition names start with 'acquire' (the MCP server previews and traces those)")
    raw = _check_keys(where, raw, ACQUISITION_KEYS)
    if "doc" not in raw:
        raise SpecError(f"{where}: needs a doc (the agent reads it as the tool's description)")
    kind = raw.get("kind")
    if kind not in KINDS:
        raise SpecError(f"{where}.kind: {kind!r} is not one of {sorted(KINDS)}")
    names = set(settings)
    trees: dict[str, Any] = {}

    def number(key: str, value: Any, required: bool = True) -> None:
        if value is None:
            if required:
                raise SpecError(f"{where}.{key}: required for kind {kind}")
            return
        trees[key] = _compile(f"{where}.{key}", value, names)
        if trees[key] is None and (not isinstance(value, (int, float)) or isinstance(value, bool)):
            raise SpecError(f"{where}.{key}: a number or an expression")

    if raw.get("noise", "poisson") not in NOISES:
        raise SpecError(f"{where}.noise: {raw.get('noise')!r} is not one of {sorted(NOISES)}")
    if kind == "spectrum":
        axis = _check_keys(f"{where}.axis", raw.get("axis"), AXIS_KEYS)
        for key in ("start", "stop", "points"):
            number(f"axis.{key}", axis.get(key))
        points = axis.get("points")
        if trees["axis.points"] is None and (not isinstance(points, int) or not 2 <= points <= MAX_POINTS):
            raise SpecError(f"{where}.axis.points: a whole number from 2 to {MAX_POINTS}, or an expression")
        for i, peak in enumerate(raw.get("peaks") or []):
            peak = _check_keys(f"{where}.peaks[{i}]", peak, PEAK_KEYS)
            for key in ("position", "height", "width"):
                number(f"peaks[{i}].{key}", peak.get(key))
        number("background", raw.get("background", 0.0))
    elif kind == "image":
        shape = raw.get("shape")
        if not (isinstance(shape, list) and len(shape) == 2 and all(isinstance(v, int) and not isinstance(v, bool) and 2 <= v <= 4096 for v in shape)):
            raise SpecError(f"{where}.shape: [rows, columns], each a whole number from 2 to 4096")
        number("pixel_size_m", raw.get("pixel_size_m"))
        number("background", raw.get("background", 0.0))
        spots = _check_keys(f"{where}.spots", raw.get("spots") or {}, SPOT_KEYS)
        for key in SPOT_KEYS & set(spots):
            number(f"spots.{key}", spots[key])
    else:
        values = _check_keys(f"{where}.values", raw.get("values"), set(raw.get("values") or {}))
        if not values:
            raise SpecError(f"{where}.values: name each value and give an expression, e.g. {{temperature_k: \"setpoint_k + 0.1\"}}")
        for key, value in values.items():
            _check_name(f"{where}.values", key)
            number(f"values.{key}", value)
    return Acquisition(
        name=name, doc=str(raw["doc"]).strip(), kind=kind, raw=raw, trees=trees,
        requires=_requires(where, raw.get("requires"), names), duration=_duration(where, raw.get("duration_s"), names),
    )


def load_spec(path: str | Path) -> InstrumentSpec:
    """Read and check an instrument YAML file; SpecError names the first problem."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SpecError(f"cannot read the instrument file {path}: {exc}") from None
    except yaml.YAMLError as exc:
        raise SpecError(f"{path} is not valid YAML: {exc}") from None
    raw = _check_keys(str(path), raw, TOP_KEYS)
    settings = {}
    for name, item in _check_keys("settings", raw.get("settings") or {}, set(raw.get("settings") or {})).items():
        settings[_check_name("settings", name)] = _quantity(f"settings.{name}", name, item, SETTING_KEYS)
    commands = {}
    for name, item in (raw.get("commands") or {}).items():
        commands[_check_name("commands", name)] = _command(name, item, settings)
    acquisitions = {}
    for name, item in (raw.get("acquisitions") or {}).items():
        acquisitions[_check_name("acquisitions", name)] = _acquisition(name, item, settings)
    spec = InstrumentSpec(path, str(raw.get("description", "")).strip(), int(raw.get("seed", 0)), settings, commands, acquisitions)
    names = spec.command_names()
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise SpecError(f"{path}: {duplicates} defined twice (settings make get_<name> and set_<name> commands)")
    if not names:
        raise SpecError(f"{path}: no settings, commands or acquisitions; nothing for an agent to call")
    return spec


class SimInstrument:
    """The instrument's state: setting values, and running commands and acquisitions against them."""

    def __init__(self, spec: InstrumentSpec, sleep=time.sleep):
        self.spec = spec
        self.values = {name: q.default for name, q in spec.settings.items()}
        self.rng = np.random.default_rng(spec.seed)
        self._sleep = sleep

    def get(self, name: str) -> Any:
        return self.values[name]

    def set(self, name: str, value: Any) -> None:
        quantity = self.spec.settings[name]
        if not quantity.settable:
            raise Refused(f"{name} is read-only")
        self.values[name] = quantity.coerce(value)

    def _check(self, name: str, requires, scope) -> None:
        for text, tree in requires:
            if not evaluate(tree, scope):
                state = {n: scope[n] for n in sorted({node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id in scope})}
                raise Refused(f"{name} refused: it needs {text} (now {json.dumps(state)})")

    def _wait(self, duration, scope) -> None:
        raw, tree = duration
        seconds = float(evaluate(tree, scope) if tree is not None else raw)
        if not 0 <= seconds <= MAX_DURATION_S:
            raise Refused(f"this would take {seconds:.3g} s; the simulator allows 0 to {MAX_DURATION_S} s per call")
        if seconds:
            self._sleep(seconds)

    def _value(self, raw, tree, scope) -> Any:
        if isinstance(raw, dict):
            return {k: self._value(v, (tree or {}).get(k), scope) for k, v in raw.items()}
        if isinstance(raw, list):
            return [self._value(v, t, scope) for v, t in zip(raw, tree or [None] * len(raw))]
        return evaluate(tree, scope) if tree is not None else raw

    def run_command(self, name: str, argument: Any = None) -> str:
        command = self.spec.commands[name]
        scope = dict(self.values)
        if command.input is not None:
            scope[command.input.name] = command.input.coerce(argument)
        self._check(name, command.requires, scope)
        updates = {target: self.spec.settings[target].coerce(evaluate(tree, scope) if tree is not None else value) for target, (value, tree) in command.sets.items()}
        self._wait(command.duration, scope)
        self.values.update(updates)
        result = self._value(command.returns, command.returns_trees, {**scope, **updates})
        if result is None:
            return ""
        return result if isinstance(result, str) else json.dumps(result)

    def acquire(self, name: str) -> tuple[str, str, np.ndarray, dict]:
        """Run an acquisition: (acquisition_type, detector, data, dataset attrs). ``value`` kinds return data=None."""
        acquisition = self.spec.acquisitions[name]
        scope = dict(self.values)
        self._check(name, acquisition.requires, scope)
        self._wait(acquisition.duration, scope)
        raw, trees = acquisition.raw, acquisition.trees

        def num(key: str, value: Any) -> float:
            return float(evaluate(trees[key], scope) if trees.get(key) is not None else value)

        detector = str(raw.get("detector", acquisition.kind))
        kind_type = name.removeprefix("acquire").strip("_") or acquisition.kind
        attrs: dict[str, Any] = {"units": "counts", "settings": json.dumps(self.values)}
        if acquisition.kind == "spectrum":
            axis = raw["axis"]
            start, stop, points = num("axis.start", axis["start"]), num("axis.stop", axis["stop"]), round(num("axis.points", axis["points"]))
            if not 2 <= points <= MAX_POINTS:
                raise Refused(f"{name} would have {points} points; 2 to {MAX_POINTS} are allowed")
            x = np.linspace(start, stop, points)
            signal = np.full(points, num("background", raw.get("background", 0.0)))
            for i, peak in enumerate(raw.get("peaks") or []):
                width = max(abs(num(f"peaks[{i}].width", peak["width"])), 1e-30)
                signal += num(f"peaks[{i}].height", peak["height"]) * np.exp(-0.5 * ((x - num(f"peaks[{i}].position", peak["position"])) / width) ** 2)
            data = self._noisy(signal, raw.get("noise", "poisson"))
            attrs.update({"axis_offset": start, "axis_scale": (stop - start) / (points - 1), "axis_label": str(axis.get("label", "x")), "axis_units": str(axis.get("units", ""))})
        elif acquisition.kind == "image":
            rows, cols = raw["shape"]
            image = np.full((rows, cols), num("background", raw.get("background", 0.0)))
            spots = raw.get("spots") or {}
            if spots:
                yy, xx = np.mgrid[0:rows, 0:cols]
                radius = max(num("spots.radius_px", spots.get("radius_px", 3)), 0.5)
                for _ in range(int(num("spots.count", spots.get("count", 5)))):
                    cy, cx = self.rng.uniform(0, rows), self.rng.uniform(0, cols)
                    image += num("spots.height", spots.get("height", 100)) * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * radius**2))
            data = self._noisy(image, raw.get("noise", "poisson"))
            attrs["pixel_size_m"] = num("pixel_size_m", raw["pixel_size_m"])
        else:
            values = {key: num(f"values.{key}", value) for key, value in raw["values"].items()}
            return kind_type, detector, None, values
        return kind_type, detector, data, attrs

    def _noisy(self, signal: np.ndarray, noise: str) -> np.ndarray:
        signal = np.clip(signal, 0, None)
        if noise == "poisson":
            return self.rng.poisson(signal).astype(np.float64)
        if noise == "gaussian":
            return signal + self.rng.normal(0, np.sqrt(signal + 1.0))
        return signal.astype(np.float64)


if __name__ == "__main__":
    # Check a file before starting it:  uv run python -m asyncroscopy.instruments.yaml_sim.spec my_instrument.yaml
    if len(sys.argv) != 2:
        sys.exit("usage: python -m asyncroscopy.instruments.yaml_sim.spec <instrument.yaml>")
    try:
        checked = load_spec(sys.argv[1])
    except SpecError as exc:
        sys.exit(f"not valid: {exc}")
    print(f"{checked.path}: valid. Commands (each one an agent tool, named <class_name>_<command>):")
    for command_name in checked.command_names():
        print(f"  {command_name}")
