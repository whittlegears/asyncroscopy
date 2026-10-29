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
        requires: ["exposure_s < 5 or not shutter_open"]   # optional checks on every change (the new value is in scope)
        doc: Counting time per point.
      grating:
        default: "1800"
        type: str
        choices: ["600", "1800"]         # optional: the only values allowed
    commands:
      xray_on:
        doc: Switch the X-ray source on.
        input: {name: power_w, type: float, min: 0, max: 1600, units: W}   # one argument
        requires: ["power_w > 0"]        # optional checks, refused with the expression
        sets: {tube_power_w: power_w}    # settings to change; values are expressions
        returns: "X-rays on"             # a string, a number, or a mapping (sent as JSON)
        duration_s: 0.5                  # optional simulated time (at most 25 s per call)
      move_xy:
        doc: Move the stage.
        inputs:                          # several arguments: the tool takes them by name
          - {name: x_m, min: -0.005, max: 0.005, units: m}
          - {name: y_m, min: -0.005, max: 0.005, units: m}
        sets: {stage_x_m: x_m, stage_y_m: y_m}
    acquisitions:
      acquire_scan:                      # names start with acquire; returns a data key
        doc: Measure a diffractogram.
        kind: spectrum                   # spectrum (1-D), image (2-D) or value
        units: counts                    # what the values are (default counts)
        inputs: [...]                    # optional arguments, as for commands
        peaks:
          - {position: 28.44, height: 900, fwhm: 0.14, shape: gaussian}   # gaussian, lorentzian or voigt
        noise: poisson                   # poisson (counts only), gaussian or none
        noise_sigma: 2.0                 # gaussian's sigma in the data's units (needed unless units are counts)
        ...
    jobs:                                # optional: a job device for long or many-point runs
      line_scan:
        doc: Spectra along a line or a grid of stage positions.
        axes: [stage_x_m, stage_y_m]     # settings a plan sweeps (start/stop/points) or lists
        settings: [exposure_s]           # settings a plan may set for the whole job
        acquire: [acquire_scan]          # run at every point, in order
        settle_s: 0.2                    # simulated time to reach each point
        requires: ["tube_power_w > 0"]   # checked at reserve
        before_move: []                  # commands run at each point before its axes move (e.g. retract a probe)
        after_move: []                   # commands run at each point after it is reached, before acquiring (e.g. approach again)
        on_stop: [xray_off]              # commands run when the job is aborted or fails

A single command or acquisition call is limited to 25 s (the MCP server stops
waiting at 30 s); a job is not, because the agent waits for it with
wait_for_job. Jobs run on a second device (see ``device.py``) with the standard
job envelope, so run_job, wait_for_job, abort, approval and limits all apply.

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
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

TYPES = {"float": float, "int": int, "bool": bool, "str": str}
NAME = re.compile(r"^[a-z][a-z0-9_]*$")
MAX_DURATION_S = 25.0  # the MCP server stops waiting for a command after 30 s
MAX_STEP_DURATION_S = 3600.0  # one acquisition inside a job (the agent waits for jobs, not calls)
MAX_POINTS = 1_000_000
MAX_JOB_POINTS = 10_000_000
FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))
FUNCTIONS = {
    "abs": abs, "min": min, "max": max, "sqrt": math.sqrt, "exp": math.exp, "log": math.log,
    "sin": math.sin, "cos": math.cos, "radians": math.radians, "degrees": math.degrees,
}
OPERATORS = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Call, ast.Name, ast.Load, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or,
    ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq, ast.IfExp,
)

SETTING_KEYS = {"default", "min", "max", "units", "type", "settable", "doc", "choices", "requires"}
INPUT_KEYS = {"name", "type", "min", "max", "units", "doc", "choices"}
COMMAND_KEYS = {"doc", "input", "inputs", "requires", "sets", "returns", "duration_s"}
ACQUISITION_KEYS = {"doc", "kind", "requires", "duration_s", "detector", "axis", "peaks", "background", "noise", "shape", "pixel_size_m", "spots", "values", "inputs", "units", "noise_sigma"}
AXIS_KEYS = {"start", "stop", "points", "label", "units"}
PEAK_KEYS = {"position", "height", "fwhm", "width", "shape", "eta"}
PEAK_SHAPES = {"gaussian", "lorentzian", "voigt"}
SPOT_KEYS = {"count", "radius_px", "height"}
JOB_KEYS = {"doc", "axes", "settings", "acquire", "settle_s", "max_points", "requires", "before_move", "after_move", "on_stop", "on_finish"}
TOP_KEYS = {"description", "seed", "settings", "commands", "acquisitions", "jobs"}
KINDS = {"spectrum", "image", "value"}
NOISES = {"poisson", "gaussian", "none"}
# Commands and attributes a job device (an instrument with jobs) has already.
JOB_DEVICE_NAMES = {"reserve", "start", "job_status", "abort", "results", "release", "get_reservation", "job_kind", "plan_example"}


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
    choices: list | None = None

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
        if self.choices is not None and value not in self.choices:
            raise Refused(f"{self.name} must be one of {self.choices}, not {value!r}")
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
        if self.choices is not None:
            text += f", one of {', '.join(str(c) for c in self.choices)}"
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
    if raw.get("choices") is not None:
        choices = raw["choices"]
        if not isinstance(choices, list) or not choices:
            raise SpecError(f"{where}.choices: a non-empty list of the allowed values")
        try:
            quantity.choices = [quantity.coerce(c) for c in choices]
        except Refused as exc:
            raise SpecError(f"{where}.choices: {exc}") from None
        if len(set(quantity.choices)) != len(quantity.choices):
            raise SpecError(f"{where}.choices: a value is listed twice")
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
    inputs: list[Quantity] = field(default_factory=list)
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
    inputs: list[Quantity] = field(default_factory=list)
    units: str = "counts"


@dataclass
class Job:
    """A run of acquisitions over many points (a line, a grid or a list), done by the job device."""

    name: str
    doc: str
    axes: list[str]
    settings: list[str]
    acquire: list[str]
    settle_s: float = 0.0
    max_points: int = 100_000
    requires: list[tuple[str, ast.Expression]] = field(default_factory=list)
    on_stop: list[str] = field(default_factory=list)
    on_finish: list[str] = field(default_factory=list)
    before_move: list[str] = field(default_factory=list)
    after_move: list[str] = field(default_factory=list)


@dataclass
class InstrumentSpec:
    path: Path
    description: str
    seed: int
    settings: dict[str, Quantity]
    commands: dict[str, Command]
    acquisitions: dict[str, Acquisition]
    jobs: dict[str, Job] = field(default_factory=dict)
    setting_requires: dict[str, list[tuple[str, ast.Expression]]] = field(default_factory=dict)

    def command_names(self) -> list[str]:
        names = [f"get_{n}" for n in self.settings] + [f"set_{n}" for n, q in self.settings.items() if q.settable]
        return names + list(self.commands) + list(self.acquisitions)


def _requires(where: str, raw: Any, names: set[str]) -> list[tuple[str, ast.Expression]]:
    if raw is None:
        return []
    if not isinstance(raw, list) or not all(isinstance(r, str) for r in raw):
        raise SpecError(f"{where}: requires is a list of expressions, e.g. [\"tube_power_w > 0\"]")
    return [(text, _compile(f"{where}.requires", text, names)) for text in raw]


def _duration(where: str, raw: Any, names: set[str], limit: float = MAX_DURATION_S) -> tuple[Any, ast.Expression | None]:
    raw = 0.0 if raw is None else raw
    tree = _compile(f"{where}.duration_s", raw, names)
    if tree is None and (not isinstance(raw, (int, float)) or isinstance(raw, bool) or not 0 <= raw <= limit):
        raise SpecError(f"{where}.duration_s: a number of seconds from 0 to {limit:g}, or an expression")
    return raw, tree


def _inputs(where: str, raw: dict, settings: dict[str, Quantity]) -> list[Quantity]:
    """A command's or acquisition's arguments: ``input`` (one) or ``inputs`` (a list)."""
    if raw.get("input") is not None and raw.get("inputs") is not None:
        raise SpecError(f"{where}: give input (one argument) or inputs (a list), not both")
    items = [raw["input"]] if raw.get("input") is not None else raw.get("inputs") or []
    if not isinstance(items, list):
        raise SpecError(f"{where}.inputs: a list of arguments, e.g. [{{name: x_m, min: -1, max: 1}}]")
    inputs: list[Quantity] = []
    for i, item in enumerate(items):
        here = f"{where}.inputs[{i}]" if raw.get("inputs") is not None else f"{where}.input"
        quantity = _quantity(here, "", item, INPUT_KEYS)
        quantity.name = _check_name(f"{here}.name", (item or {}).get("name"))
        if quantity.name in settings:
            raise SpecError(f"{here}.name: {quantity.name!r} is also a setting; pick another name")
        if quantity.name in {q.name for q in inputs}:
            raise SpecError(f"{here}.name: {quantity.name!r} is used twice")
        inputs.append(quantity)
    return inputs


def parse_arguments(name: str, inputs: list[Quantity], argument: Any) -> dict[str, Any]:
    """The checked arguments of a call: one value for a single input, or a mapping (or its JSON) for several."""
    if not inputs:
        return {}
    if len(inputs) == 1:
        if isinstance(argument, dict) and set(argument) == {inputs[0].name}:
            argument = argument[inputs[0].name]
        return {inputs[0].name: inputs[0].coerce(argument)}
    example = json.dumps({q.name: (q.choices[0] if q.choices else q.min if q.min is not None else 0) for q in inputs})
    if isinstance(argument, str):
        try:
            argument = json.loads(argument)
        except json.JSONDecodeError:
            raise Refused(f"{name} takes {', '.join(q.name for q in inputs)} as an object, e.g. {example}") from None
    if not isinstance(argument, dict):
        raise Refused(f"{name} takes {', '.join(q.name for q in inputs)} as an object, e.g. {example}; not {argument!r}")
    unknown = sorted(set(argument) - {q.name for q in inputs})
    missing = [q.name for q in inputs if q.name not in argument]
    if unknown or missing:
        raise Refused(f"{name} takes exactly {[q.name for q in inputs]}" + (f"; unknown {unknown}" if unknown else "") + (f"; missing {missing}" if missing else ""))
    return {q.name: q.coerce(argument[q.name]) for q in inputs}


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
    inputs = _inputs(where, raw, settings)
    names = set(settings) | {q.name for q in inputs}
    sets = {}
    for target, value in _check_keys(f"{where}.sets", raw.get("sets") or {}, set(settings)).items():
        sets[target] = (value, _compile(f"{where}.sets.{target}", value, names))
    if (name.startswith(("get_", "list_"))) and (sets or raw.get("duration_s")):
        raise SpecError(f"{where}: get_ and list_ commands must not change anything (the safety policy never holds them); rename it or drop sets/duration_s")
    return Command(
        name=name,
        doc=str(raw["doc"]).strip(),
        inputs=inputs,
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
    inputs = _inputs(where, raw, settings)
    names = set(settings) | {q.name for q in inputs}
    trees: dict[str, Any] = {}
    if not isinstance(raw.get("units", "counts"), str):
        raise SpecError(f"{where}.units: a string, e.g. counts or V")

    def number(key: str, value: Any, required: bool = True) -> None:
        if value is None:
            if required:
                raise SpecError(f"{where}.{key}: required for kind {kind}")
            return
        trees[key] = _compile(f"{where}.{key}", value, names)
        if trees[key] is None and (not isinstance(value, (int, float)) or isinstance(value, bool)):
            raise SpecError(f"{where}.{key}: a number or an expression")

    noise, units = raw.get("noise", "poisson"), raw.get("units", "counts")
    if noise not in NOISES:
        raise SpecError(f"{where}.noise: {noise!r} is not one of {sorted(NOISES)}")
    if kind != "value" and units != "counts":
        # Poisson noise, or Gaussian noise of sigma sqrt(signal), only makes sense for counts: on metres or newtons it buries the signal.
        if noise == "poisson":
            raise SpecError(f"{where}.noise: Poisson noise is for counts; for {units} give noise: gaussian with noise_sigma (in {units}), or noise: none")
        if noise == "gaussian" and "noise_sigma" not in raw:
            raise SpecError(f"{where}.noise_sigma: Gaussian noise on {units} needs its sigma, in {units}")
    if "noise_sigma" in raw:
        if noise != "gaussian":
            raise SpecError(f"{where}.noise_sigma: only Gaussian noise has a sigma")
        number("noise_sigma", raw["noise_sigma"])
    if kind == "spectrum":
        axis = _check_keys(f"{where}.axis", raw.get("axis"), AXIS_KEYS)
        for key in ("start", "stop", "points"):
            number(f"axis.{key}", axis.get(key))
        points = axis.get("points")
        if trees["axis.points"] is None and (not isinstance(points, int) or not 2 <= points <= MAX_POINTS):
            raise SpecError(f"{where}.axis.points: a whole number from 2 to {MAX_POINTS}, or an expression")
        for i, peak in enumerate(raw.get("peaks") or []):
            here = f"{where}.peaks[{i}]"
            peak = _check_keys(here, peak, PEAK_KEYS)
            shape = peak.get("shape", "gaussian")
            if shape not in PEAK_SHAPES:
                raise SpecError(f"{here}.shape: {shape!r} is not one of {sorted(PEAK_SHAPES)}")
            if ("fwhm" in peak) == ("width" in peak):
                raise SpecError(f"{here}: give fwhm (the full width at half maximum, in axis units)" + ("; not also width" if "fwhm" in peak else ""))
            if "width" in peak and shape != "gaussian":
                raise SpecError(f"{here}.width: width is a Gaussian's sigma; a {shape} peak takes fwhm")
            if "eta" in peak and shape != "voigt":
                raise SpecError(f"{here}.eta: only a voigt peak has eta (its Lorentzian fraction)")
            for key in ("position", "height", "fwhm" if "fwhm" in peak else "width"):
                number(f"peaks[{i}].{key}", peak.get(key))
            if shape == "voigt":
                number(f"peaks[{i}].eta", peak.get("eta", 0.5))
                eta = peak.get("eta", 0.5)
                if trees[f"peaks[{i}].eta"] is None and not 0 <= eta <= 1:
                    raise SpecError(f"{here}.eta: the Lorentzian fraction, from 0 to 1")
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
        requires=_requires(where, raw.get("requires"), names),
        duration=_duration(where, raw.get("duration_s"), names, limit=MAX_STEP_DURATION_S),
        inputs=inputs, units=str(raw.get("units", "counts")),
    )


def _names_list(where: str, raw: Any, allowed: dict, what: str) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list) or not all(isinstance(n, str) for n in raw):
        raise SpecError(f"{where}: a list of {what} names")
    unknown = [n for n in raw if n not in allowed]
    if unknown:
        raise SpecError(f"{where}: {unknown} are not {what}s; known: {sorted(allowed)}")
    return list(raw)


def _job(name: str, raw: Any, spec: "InstrumentSpec") -> Job:
    where = f"jobs.{name}"
    raw = _check_keys(where, raw, JOB_KEYS)
    if "doc" not in raw:
        raise SpecError(f"{where}: needs a doc (the agent reads it with the job's tools)")
    axes = _names_list(f"{where}.axes", raw.get("axes"), spec.settings, "setting")
    if not axes:
        raise SpecError(f"{where}.axes: list the settings the job steps over (e.g. stage positions)")
    for axis in axes:
        quantity = spec.settings[axis]
        if not quantity.settable or quantity.type not in (int, float) or quantity.choices is not None:
            raise SpecError(f"{where}.axes: {axis!r} must be a settable number to be stepped over")
    settings = _names_list(f"{where}.settings", raw.get("settings"), spec.settings, "setting")
    for setting in settings:
        if not spec.settings[setting].settable:
            raise SpecError(f"{where}.settings: {setting!r} is read-only")
    if set(axes) & set(settings):
        raise SpecError(f"{where}: {sorted(set(axes) & set(settings))} listed in both axes and settings")
    acquire = _names_list(f"{where}.acquire", raw.get("acquire"), spec.acquisitions, "acquisition")
    if not acquire:
        raise SpecError(f"{where}.acquire: list the acquisitions to run at every point")
    for acquisition in acquire:
        if spec.acquisitions[acquisition].inputs:
            raise SpecError(f"{where}.acquire: {acquisition!r} takes arguments; a job runs acquisitions without arguments")
    for key in ("before_move", "after_move", "on_stop", "on_finish"):
        for command_name in _names_list(f"{where}.{key}", raw.get(key), spec.commands, "command"):
            if spec.commands[command_name].inputs:
                raise SpecError(f"{where}.{key}: {command_name!r} takes arguments; list commands without arguments")
    settle = raw.get("settle_s", 0.0)
    if not isinstance(settle, (int, float)) or isinstance(settle, bool) or not 0 <= settle <= MAX_STEP_DURATION_S:
        raise SpecError(f"{where}.settle_s: seconds from 0 to {MAX_STEP_DURATION_S:g}")
    max_points = raw.get("max_points", 100_000)
    if not isinstance(max_points, int) or isinstance(max_points, bool) or not 1 <= max_points <= MAX_JOB_POINTS:
        raise SpecError(f"{where}.max_points: a whole number from 1 to {MAX_JOB_POINTS}")
    return Job(
        name=name, doc=str(raw["doc"]).strip(), axes=axes, settings=settings, acquire=acquire,
        settle_s=float(settle), max_points=max_points, requires=_requires(where, raw.get("requires"), set(spec.settings)),
        on_stop=list(raw.get("on_stop") or []), on_finish=list(raw.get("on_finish") or []),
        before_move=list(raw.get("before_move") or []), after_move=list(raw.get("after_move") or []),
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
    for name, item in (raw.get("settings") or {}).items():
        if (item or {}).get("requires"):
            if not settings[name].settable:
                raise SpecError(f"settings.{name}.requires: a read-only setting is never set")
            spec.setting_requires[name] = _requires(f"settings.{name}", item["requires"], set(settings))
    for name, item in _check_keys("jobs", raw.get("jobs") or {}, set(raw.get("jobs") or {})).items():
        spec.jobs[_check_name("jobs", name)] = _job(name, item, spec)
    names = spec.command_names()
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise SpecError(f"{path}: {duplicates} defined twice (settings make get_<name> and set_<name> commands)")
    if not names:
        raise SpecError(f"{path}: no settings, commands or acquisitions; nothing for an agent to call")
    taken = sorted(set(names) & JOB_DEVICE_NAMES) if spec.jobs else []
    if taken:
        raise SpecError(f"{path}: {taken} are the job device's own commands (an instrument with jobs has them); rename yours")
    return spec


class SimInstrument:
    """The instrument's state: setting values, and running commands and acquisitions against them."""

    def __init__(self, spec: InstrumentSpec, sleep=time.sleep):
        self.spec = spec
        self.values = {name: q.default for name, q in spec.settings.items()}
        self.rng = np.random.default_rng(spec.seed)
        self._sleep = sleep
        # Tango command threads and a job's worker share the settings.
        self.lock = threading.RLock()

    def get(self, name: str) -> Any:
        with self.lock:
            return self.values[name]

    def set(self, name: str, value: Any) -> None:
        quantity = self.spec.settings[name]
        if not quantity.settable:
            raise Refused(f"{name} is read-only")
        with self.lock:
            coerced = quantity.coerce(value)
            # A setting's requires guard every way of changing it (set_<name>, a job's axes), like a command's.
            self._check(f"set_{name}", self.spec.setting_requires.get(name, []), {**self.values, name: coerced})
            self.values[name] = coerced

    def _check(self, name: str, requires, scope) -> None:
        for text, tree in requires:
            if not evaluate(tree, scope):
                state = {n: scope[n] for n in sorted({node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id in scope})}
                raise Refused(f"{name} refused: it needs {text} (now {json.dumps(state)})")

    def seconds(self, duration, scope) -> float:
        raw, tree = duration
        return float(evaluate(tree, scope) if tree is not None else raw)

    def _wait(self, name: str, duration, scope, limit: float | None = MAX_DURATION_S, sleep=None) -> None:
        seconds = self.seconds(duration, scope)
        if seconds < 0:
            raise Refused(f"{name} would take {seconds:.3g} s; a duration cannot be negative")
        if limit is not None and seconds > limit:
            jobs = [job for job in self.spec.jobs.values() if name in job.acquire]
            hint = (
                f" Run it as part of the {jobs[0].name} job instead (run_job), which has no such limit."
                if jobs else " Longer runs need a job (a jobs: section in the instrument file)."
            )
            raise Refused(f"{name} would take {seconds:.3g} s; one call is limited to {limit:g} s.{hint}")
        if seconds:
            (sleep or self._sleep)(seconds)

    def _value(self, raw, tree, scope) -> Any:
        if isinstance(raw, dict):
            return {k: self._value(v, (tree or {}).get(k), scope) for k, v in raw.items()}
        if isinstance(raw, list):
            return [self._value(v, t, scope) for v, t in zip(raw, tree or [None] * len(raw))]
        return evaluate(tree, scope) if tree is not None else raw

    def run_command(self, name: str, argument: Any = None) -> str:
        command = self.spec.commands[name]
        with self.lock:
            scope = {**self.values, **parse_arguments(name, command.inputs, argument)}
            self._check(name, command.requires, scope)
            updates = {target: self.spec.settings[target].coerce(evaluate(tree, scope) if tree is not None else value) for target, (value, tree) in command.sets.items()}
        self._wait(name, command.duration, scope)
        with self.lock:
            self.values.update(updates)
        result = self._value(command.returns, command.returns_trees, {**scope, **updates})
        if result is None:
            return ""
        return result if isinstance(result, str) else json.dumps(result)

    def acquire(self, name: str, argument: Any = None, limit: float | None = MAX_DURATION_S, sleep=None) -> tuple[str, str, np.ndarray, dict]:
        """Run an acquisition: (acquisition_type, detector, data, dataset attrs). ``value`` kinds return data=None.

        ``limit`` is the longest one call may take (None inside a job); ``sleep`` replaces the
        simulated wait, so a job can stop it early for an abort.
        """
        acquisition = self.spec.acquisitions[name]
        with self.lock:
            scope = {**self.values, **parse_arguments(name, acquisition.inputs, argument)}
            self._check(name, acquisition.requires, scope)
        self._wait(name, acquisition.duration, scope, limit=limit, sleep=sleep)
        raw, trees = acquisition.raw, acquisition.trees

        def num(key: str, value: Any) -> float:
            return float(evaluate(trees[key], scope) if trees.get(key) is not None else value)

        detector = str(raw.get("detector", acquisition.kind))
        kind_type = name.removeprefix("acquire").strip("_") or acquisition.kind
        attrs: dict[str, Any] = {"units": acquisition.units, "settings": json.dumps({k: v for k, v in scope.items() if k in self.spec.settings})}
        arguments = {k: v for k, v in scope.items() if k not in self.spec.settings}
        if arguments:
            attrs["arguments"] = json.dumps(arguments)
        if acquisition.kind == "spectrum":
            axis = raw["axis"]
            start, stop, points = num("axis.start", axis["start"]), num("axis.stop", axis["stop"]), round(num("axis.points", axis["points"]))
            if not 2 <= points <= MAX_POINTS:
                raise Refused(f"{name} would have {points} points; 2 to {MAX_POINTS} are allowed")
            x = np.linspace(start, stop, points)
            signal = np.full(points, num("background", raw.get("background", 0.0)))
            for i, peak in enumerate(raw.get("peaks") or []):
                shape = peak.get("shape", "gaussian")
                if "fwhm" in peak:
                    fwhm = max(abs(num(f"peaks[{i}].fwhm", peak["fwhm"])), 1e-30)
                else:  # width: a Gaussian's sigma, as files written before fwhm existed give it
                    fwhm = max(abs(num(f"peaks[{i}].width", peak["width"])), 1e-30) * FWHM_PER_SIGMA
                eta = {"gaussian": 0.0, "lorentzian": 1.0}.get(shape)
                if eta is None:
                    eta = num(f"peaks[{i}].eta", peak.get("eta", 0.5))
                signal += num(f"peaks[{i}].height", peak["height"]) * peak_profile(x, num(f"peaks[{i}].position", peak["position"]), fwhm, eta)
            data = self._noisy(signal, raw.get("noise", "poisson"), num("noise_sigma", raw["noise_sigma"]) if "noise_sigma" in raw else None)
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
            data = self._noisy(image, raw.get("noise", "poisson"), num("noise_sigma", raw["noise_sigma"]) if "noise_sigma" in raw else None)
            attrs["pixel_size_m"] = num("pixel_size_m", raw["pixel_size_m"])
        else:
            values = {key: num(f"values.{key}", value) for key, value in raw["values"].items()}
            return kind_type, detector, None, values
        return kind_type, detector, data, attrs

    def _noisy(self, signal: np.ndarray, noise: str, sigma: float | None = None) -> np.ndarray:
        """Poisson counts (clipped at 0), or Gaussian noise of ``sigma`` (default sqrt(counts + 1)); otherwise signs are kept."""
        with self.lock:  # numpy Generators are not thread-safe
            if noise == "poisson":
                return self.rng.poisson(np.clip(signal, 0, None)).astype(np.float64)
            if noise == "gaussian":
                scale = abs(sigma) if sigma is not None else np.sqrt(np.clip(signal, 0, None) + 1.0)
                return signal + self.rng.normal(0, scale, signal.shape)
        return signal.astype(np.float64)


def peak_profile(x: np.ndarray, position: float, fwhm: float, eta: float) -> np.ndarray:
    """A peak of height 1 at ``position``: Gaussian (eta 0), Lorentzian (eta 1) or the
    pseudo-Voigt mix of the two (0 < eta < 1), all with the same full width at half maximum."""
    offset = (x - position) / fwhm
    gaussian = np.exp(-4.0 * math.log(2.0) * offset**2)
    lorentzian = 1.0 / (1.0 + 4.0 * offset**2)
    return (1.0 - eta) * gaussian + eta * lorentzian


@dataclass
class JobPlan:
    """A job plan checked against the instrument file: which points, in what order, with which settings."""

    job: Job
    axes: dict[str, tuple[float, float, int]]  # setting -> (start, stop, points), the last one changing fastest
    points: list[dict[str, float]] | None      # or an explicit list of points
    settings: dict[str, Any]
    count: int

    def point(self, index: int) -> dict[str, float]:
        """The axis values of point ``index`` (0-based)."""
        if self.points is not None:
            return self.points[index]
        values = {}
        for name, (start, stop, n) in reversed(list(self.axes.items())):
            index, k = divmod(index, n)
            values[name] = start if n == 1 else start + (stop - start) * k / (n - 1)
        return dict(reversed(list(values.items())))


def plan_example(spec: InstrumentSpec, job: Job) -> dict:
    """A valid plan for ``job``: a short line over its first axis, inside the axis's limits."""
    first = spec.settings[job.axes[0]]
    low = first.min if first.min is not None else float(first.default) - 1.0
    high = first.max if first.max is not None else float(first.default) + 1.0
    start, stop = low + 0.45 * (high - low), low + 0.55 * (high - low)
    plan: dict[str, Any] = {"axes": {job.axes[0]: {"start": round(start, 12), "stop": round(stop, 12), "points": 5}}}
    if len(spec.jobs) > 1:
        plan = {"job": job.name, **plan}
    return plan


def resolve_job_plan(spec: InstrumentSpec, plan: dict, values: dict[str, Any]) -> JobPlan:
    """Check a plan's shape, names and types (limits are checked separately, as violations); ValueError says what is wrong."""
    if not spec.jobs:
        raise ValueError("this instrument has no jobs")
    if not isinstance(plan, dict):
        raise ValueError("the plan must be an object")
    unknown = sorted(set(plan) - {"job", "axes", "points", "settings"})
    if unknown:
        raise ValueError(f"unknown plan field(s) {unknown}; a plan has job, axes or points, and settings")
    name = plan.get("job") or (next(iter(spec.jobs)) if len(spec.jobs) == 1 else None)
    if name not in spec.jobs:
        raise ValueError(f"name the job: one of {sorted(spec.jobs)}")
    job = spec.jobs[name]
    if ("axes" in plan) == ("points" in plan):
        raise ValueError(f"give axes (start, stop and points for each of {job.axes}) or points (a list of positions), not both")
    axes: dict[str, tuple[float, float, int]] = {}
    points = None
    if "axes" in plan:
        if not isinstance(plan["axes"], dict) or not plan["axes"]:
            raise ValueError("axes maps each stepped setting to start, stop and points")
        for axis, item in plan["axes"].items():
            if axis not in job.axes:
                raise ValueError(f"{axis!r} is not an axis of {name}; its axes are {job.axes}")
            if not isinstance(item, dict) or set(item) - {"start", "stop", "points"} or "start" not in item:
                raise ValueError(f"axes.{axis} needs start, stop and points")
            n = item.get("points", 1)
            if not isinstance(n, int) or isinstance(n, bool) or n < 1:
                raise ValueError(f"axes.{axis}.points must be a whole number of at least 1")
            start = float(item["start"])
            stop = float(item.get("stop", start))
            if not (math.isfinite(start) and math.isfinite(stop)):
                raise ValueError(f"axes.{axis}: start and stop must be finite numbers")
            axes[axis] = (start, stop, n)
        count = math.prod(n for _, _, n in axes.values())
    else:
        if not isinstance(plan["points"], list) or not plan["points"]:
            raise ValueError(f"points is a non-empty list of positions, each with {job.axes}")
        if len(plan["points"]) > job.max_points:
            raise ValueError(f"{len(plan['points'])} points; {name} allows at most {job.max_points}")
        points = []
        for i, item in enumerate(plan["points"]):
            if not isinstance(item, dict) or not item or set(item) - set(job.axes):
                raise ValueError(f"points[{i}] gives values for some of {job.axes}, e.g. {json.dumps({job.axes[0]: 0.0})}")
            points.append({k: float(v) for k, v in item.items()})
        count = len(points)
    if count > job.max_points:
        raise ValueError(f"{count} points; {name} allows at most {job.max_points}")
    settings = plan.get("settings") or {}
    if not isinstance(settings, dict):
        raise ValueError(f"settings maps some of {job.settings} to values")
    checked = {}
    for setting, value in settings.items():
        if setting not in job.settings:
            raise ValueError(f"{setting!r} cannot be set by {name}; it sets {job.settings}")
        quantity = spec.settings[setting]
        bounded = Quantity(setting, quantity.type, units=quantity.units, choices=quantity.choices)  # limits are checked as violations
        try:
            checked[setting] = bounded.coerce(value)
        except Refused as exc:
            raise ValueError(str(exc)) from None
    scope = {**values, **checked}
    for text, tree in job.requires:
        if not evaluate(tree, scope):
            state = {n: scope[n] for n in sorted({node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id in scope})}
            raise ValueError(f"{name} needs {text} (now {json.dumps(state)})")
    return JobPlan(job=job, axes=axes, points=points, settings=checked, count=count)


def job_estimate(spec: InstrumentSpec, sim: "SimInstrument", resolved: JobPlan, values: dict[str, Any]) -> dict:
    """Points and seconds: the move time plus every acquisition's duration, at each point (sampled for big plans)."""
    sample = range(resolved.count) if resolved.count <= 2000 else sorted({0, resolved.count - 1, *range(0, resolved.count, max(1, resolved.count // 1000))})
    total = 0.0
    for index in sample:
        scope = {**values, **resolved.settings, **resolved.point(index)}
        total += resolved.job.settle_s + sum(sim.seconds(spec.acquisitions[a].duration, scope) for a in resolved.job.acquire)
    per_point = total / len(sample)
    return {"points": resolved.count, "per_point_s": round(per_point, 6), "duration_s": round(per_point * resolved.count, 3), "acquisitions_per_point": resolved.job.acquire}


def job_targets(spec: InstrumentSpec, resolved: JobPlan) -> list[tuple[int, str, float, float | None, float | None]]:
    """(step, setting, value, min, max) for every bounded value the job will set. A swept axis is
    linear, so its first and last values bound every point; a list is checked point by point."""
    targets = []
    for setting, value in resolved.settings.items():
        quantity = spec.settings[setting]
        if quantity.type in (int, float) and (quantity.min is not None or quantity.max is not None):
            targets.append((0, setting, float(value), quantity.min, quantity.max))
    if resolved.points is not None:
        for index, point in enumerate(resolved.points):
            for setting, value in point.items():
                quantity = spec.settings[setting]
                targets.append((index, setting, value, quantity.min, quantity.max))
    else:
        last = resolved.count - 1
        for setting, (start, stop, n) in resolved.axes.items():
            quantity = spec.settings[setting]
            for step, value in ((0, start), (last, stop)) if n > 1 else ((0, start),):
                targets.append((step, setting, value, quantity.min, quantity.max))
    return targets


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
    for job_name, job_spec in checked.jobs.items():
        print(f"Job {job_name} (on the job device: run_job with plan {json.dumps(plan_example(checked, job_spec))})")
