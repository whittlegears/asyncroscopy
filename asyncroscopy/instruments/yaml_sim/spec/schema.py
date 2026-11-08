"""The YAML instrument file's schema: its dataclasses, the checks made at load, and what the file declares to the MCP server."""

from __future__ import annotations

import ast
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from asyncroscopy.core.names import STOP_ROLES, check_role
from .expressions import Refused, SpecError, _check_keys, _check_name, _compile, evaluate

if TYPE_CHECKING:
    from .physics import SimInstrument

PROJECT_DIR = Path(__file__).resolve().parents[4]
TYPES = {"float": float, "int": int, "bool": bool, "str": str}
MAX_DURATION_S = 25.0  # the MCP server stops waiting for a command after 30 s
MAX_STEP_DURATION_S = 3600.0  # one acquisition inside a job (the agent waits for jobs, not calls)
MAX_POINTS = 1_000_000
MAX_JOB_POINTS = 10_000_000
SETTING_KEYS = {"default", "min", "max", "units", "type", "settable", "doc", "choices", "requires"}
INPUT_KEYS = {"name", "type", "min", "max", "units", "doc", "choices"}
COMMAND_KEYS = {"doc", "input", "inputs", "requires", "sets", "returns", "duration_s", "role", "risk"}
ACQUISITION_KEYS = {"doc", "kind", "requires", "duration_s", "detector", "axis", "peaks", "background", "noise", "shape", "pixel_size_m", "spots", "values", "inputs", "units", "noise_sigma", "role"}
KIND_ROLES = {"image": "acquire.image", "spectrum": "acquire.spectrum", "value": "acquire.other"}
AXIS_KEYS = {"start", "stop", "points", "label", "units"}
PEAK_KEYS = {"position", "height", "fwhm", "width", "shape", "eta"}
PEAK_SHAPES = {"gaussian", "lorentzian", "voigt"}
SPOT_KEYS = {"count", "radius_px", "height", "x_m", "y_m", "per_um2", "radius_m"}
SAMPLE_CELL_M = 1e-6  # spots fixed on the sample are drawn per square of this side, each from its own seed
SAMPLE_CELLS_MAX = 40_000  # an image wider than this many cells is refused (a 200 um square)
JOB_KEYS = {"doc", "axes", "settings", "acquire", "settle_s", "max_points", "requires", "before_move", "after_move", "on_stop", "on_finish"}
MAP_KEYS = {"image", "field_of_view", "stage", "before_move", "after_move", "safe", "settings", "budgets"}
MAP_BUDGET_KEYS = {"per_tile", "per_revisit", "combine", "units", "doc"}
MAP_COMMAND = "get_map_profile"  # the command a YAML device offers when its file can be mapped
TOP_KEYS = {"description", "seed", "settings", "commands", "acquisitions", "jobs", "map"}
KINDS = {"spectrum", "image", "value"}
NOISES = {"poisson", "gaussian", "none"}
# Commands and attributes a job device (an instrument with jobs) has already.
# Names the instrument device has already (Tango names ignore case): its step protocol, and the Instrument base's commands and attributes.
RESERVED_NAMES = {"hold", "release_hold", "start_step", "step_status", "stop_step", "connect", "disconnect", "instrument_type", "clock_s"}
SETTLE_STEP = "_settle"  # the step a job runs for its settle_s; "_" keeps it apart from every name a file may give
# Verbs that make a get_/list_ command's doc read as an action: the safety policy never holds a get_
# or list_ command, so one that indents, moves or dispenses would run unasked in approve mode.
ACTION_VERBS = frozenset({
    "make", "move", "indent", "pick", "place", "aspirate", "dispense", "start", "run", "apply", "scan", "drive",
    "approach", "engage", "heat", "fire", "inject", "pump", "mill", "cut", "write", "expose", "trigger", "load", "push",
})


def _reads_as_action(doc: str) -> str | None:
    """The action verb a doc starts with ("Make one indent ...", "Moves the stage ..."), or None."""
    words = re.findall(r"[a-z]+", doc.lower())
    if not words:
        return None
    first = words[0]
    for verb in (first, first[:-1] if first.endswith("s") else None, first[:-2] if first.endswith("es") else None):
        if verb in ACTION_VERBS:
            return verb
    return None


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
    role: str | None = None  # a canonical role (asyncroscopy.core.names.ROLES); None: the rules decide
    risk: str | None = None  # "low" or "high" as the file says; None: from the role (``declaration``)

    def declaration(self) -> dict:
        """What the command is, for the MCP server: {"risk", "role", "stop"} as far as the file says (see the
        module docstring); {} for a command with neither risk nor role (the server's name rules decide)."""
        found: dict = {}
        role = self.role or ("setting.read" if self.name.startswith(("get_", "list_")) else None)
        if role:
            found["role"] = role
        if self.risk:
            found["risk"] = self.risk
        elif role in STOP_ROLES or role == "setting.read":
            found["risk"] = "low"
        elif role:
            found["risk"] = "high"
        if role in STOP_ROLES and not self.inputs and found.get("risk") == "low":
            found["stop"] = True
        return found


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
    role: str | None = None  # None: from the kind (KIND_ROLES)


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
    map: "MapSection | None" = None  # the file's map: section, if it has one (map_section() infers one otherwise)
    warnings: list[str] = field(default_factory=list)  # loads, but probably not what was meant (printed at load)

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
        role=_role(where, raw.get("role")),
        risk=_risk(where, raw.get("risk")),
    )


def _risk(where: str, raw: Any) -> str | None:
    """An optional ``risk:``: low or high."""
    if raw is None:
        return None
    if str(raw).strip().lower() not in ("low", "high"):
        raise SpecError(f"{where}.risk: {raw!r} is not low or high")
    return str(raw).strip().lower()


def _role(where: str, raw: Any) -> str | None:
    """An optional ``role:``, checked against the canonical roles."""
    if raw is None:
        return None
    try:
        return check_role(raw, f"{where}.role")
    except ValueError as exc:
        raise SpecError(str(exc)) from None


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
        on_sample = {"x_m", "y_m", "per_um2", "radius_m"} & set(spots)
        if on_sample and not {"x_m", "y_m", "per_um2"} <= set(spots):
            raise SpecError(f"{where}.spots: spots fixed on the sample need x_m and y_m (where the image centre is, e.g. stage_x_m) and per_um2")
        if on_sample and "count" in spots:
            raise SpecError(f"{where}.spots: count is for spots drawn afresh in every image; spots fixed on the sample take per_um2")
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
        role=_role(where, raw.get("role")),
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


@dataclass
class MapBudget:
    """One budget a map plan can give (``budgets: {name: limit}``): what a tile and a revisit image cost of it."""

    name: str
    per_tile: tuple[Any, ast.Expression | None]
    per_revisit: tuple[Any, ast.Expression | None]
    combine: str  # "max": the worst single image counts (a peak force); "sum": every image adds (wear, exposure)
    units: str
    doc: str


@dataclass
class MapSection:
    """How the Mapper tiles this instrument (instrument_profile yaml): which image, stage, field of view and safe sequence."""

    image: str
    stage: list[str]
    field_of_view: str | None = None
    before_move: list[str] = field(default_factory=list)
    after_move: list[str] = field(default_factory=list)
    safe: list[str] = field(default_factory=list)
    settings: list[str] = field(default_factory=list)
    budgets: dict[str, MapBudget] = field(default_factory=dict)


def _number_or_expression(where: str, raw: Any, names: set[str]) -> tuple[Any, ast.Expression | None]:
    tree = _compile(where, raw, names)
    if tree is None and (not isinstance(raw, (int, float)) or isinstance(raw, bool)):
        raise SpecError(f"{where}: a number or an expression")
    return raw, tree


def _map(raw: Any, spec: InstrumentSpec) -> MapSection:
    where = "map"
    raw = _check_keys(where, raw, MAP_KEYS)
    inferred = map_section(spec)
    images = [n for n, a in spec.acquisitions.items() if a.kind == "image"]
    image = raw.get("image", inferred.image if inferred else None)
    if image not in images:
        raise SpecError(f"{where}.image: {image!r} is not an image acquisition; image acquisitions: {images}")
    if spec.acquisitions[image].inputs:
        raise SpecError(f"{where}.image: {image!r} takes arguments; a map acquires its tiles without arguments")
    stage = _names_list(f"{where}.stage", raw.get("stage", inferred.stage if inferred else None), spec.settings, "setting")
    if len(stage) != 2:
        raise SpecError(f"{where}.stage: the stage's x and y settings, e.g. [stage_x_m, stage_y_m]")
    fov = raw.get("field_of_view", inferred.field_of_view if inferred else None)
    for name in stage + ([fov] if fov else []):
        if name not in spec.settings:
            raise SpecError(f"{where}: {name!r} is not a setting")
        quantity = spec.settings[name]
        if not quantity.settable or quantity.type not in (int, float):
            raise SpecError(f"{where}: {name!r} must be a settable number")
    lists = {}
    for key in ("before_move", "after_move", "safe"):
        default = getattr(inferred, key) if inferred else []
        lists[key] = _names_list(f"{where}.{key}", raw.get(key, default), spec.commands, "command")
        for command_name in lists[key]:
            if spec.commands[command_name].inputs:
                raise SpecError(f"{where}.{key}: {command_name!r} takes arguments; list commands without arguments")
    settings = _names_list(f"{where}.settings", raw.get("settings", inferred.settings if inferred else []), spec.settings, "setting")
    for name in settings:
        if not spec.settings[name].settable or name in stage or name == fov:
            raise SpecError(f"{where}.settings: {name!r} must be settable and not the stage or field of view")
    budgets = {}
    names = set(spec.settings)
    for name, item in _check_keys(f"{where}.budgets", raw.get("budgets") or {}, set(raw.get("budgets") or {})).items():
        here = f"{where}.budgets.{_check_name(f'{where}.budgets', name)}"
        item = _check_keys(here, item, MAP_BUDGET_KEYS)
        if "per_tile" not in item:
            raise SpecError(f"{here}.per_tile: what one tile costs of this budget (a number or an expression over the settings)")
        combine = item.get("combine", "sum")
        if combine not in ("max", "sum"):
            raise SpecError(f"{here}.combine: max (the worst single image counts) or sum (every image adds)")
        budgets[name] = MapBudget(
            name=name,
            per_tile=_number_or_expression(f"{here}.per_tile", item["per_tile"], names),
            per_revisit=_number_or_expression(f"{here}.per_revisit", item.get("per_revisit", item["per_tile"]), names),
            combine=combine, units=str(item.get("units", "")), doc=str(item.get("doc", "")).strip(),
        )
    return MapSection(image=image, stage=stage, field_of_view=fov, settings=settings, budgets=budgets, **lists)


def map_section(spec: InstrumentSpec) -> MapSection | None:
    """The file's map: section, or one inferred from it; None when the file has no image acquisition or no stage.

    Inferred: the first image acquisition; the settings stage_x_m and stage_y_m, else the first job's two
    axes; as field of view the one setting in metres that the image's pixel_size_m uses; the first
    job's before_move, after_move, on_stop (as safe) and settings.
    """
    if spec.map is not None:
        return spec.map
    images = [n for n, a in spec.acquisitions.items() if a.kind == "image" and not a.inputs]
    if not images:
        return None
    job = next(iter(spec.jobs.values()), None)
    settable = {n for n, q in spec.settings.items() if q.settable and q.type in (int, float)}
    if {"stage_x_m", "stage_y_m"} <= settable:
        stage = ["stage_x_m", "stage_y_m"]
    elif job is not None and len(job.axes) == 2:
        stage = list(job.axes)
    else:
        return None
    tree = spec.acquisitions[images[0]].trees.get("pixel_size_m")
    used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} if tree is not None else set()
    metres = [n for n in sorted(used & settable) if spec.settings[n].units == "m" and n not in stage]
    return MapSection(
        image=images[0], stage=stage, field_of_view=metres[0] if len(metres) == 1 else None,
        before_move=list(job.before_move) if job else [], after_move=list(job.after_move) if job else [],
        safe=list(job.on_stop) if job else [], settings=[s for s in (job.settings if job else []) if s not in stage],
    )


JSON_TYPES = {float: "number", int: "integer", bool: "boolean", str: "string"}


def _argument(quantity: Quantity, axis: str | None = None) -> dict:
    """One argument (or value) as the action space shows it: type, units, range, choices, stage axis."""
    found = {"name": quantity.name, "type": JSON_TYPES[quantity.type]}
    for key, value in (("units", quantity.units), ("min", quantity.min), ("max", quantity.max), ("enum", quantity.choices), ("doc", quantity.doc), ("axis", axis)):
        if value not in (None, ""):
            found[key] = value
    return found


def action_hints(spec: InstrumentSpec) -> dict:
    """What the file says about each command beyond its doc, for the MCP server (its descriptors and action
    space): {command: {"risk", "role", "stop", "args", "returns"}}.

    Risk and role as the module docstring lists them. A role is the file's own ``role:``, an acquisition's
    kind (KIND_ROLES), or, for the stage settings and the field of view of the map section (given or
    inferred), stage.move / stage.read / optics.fov; any other setting is setting.read / setting.set.
    Arguments carry type, units, range and choices,
    and ``axis`` (x, y, z) when they set a stage setting, so a controller can fill them from a position.
    The map section's image acquisition is ``primary``: the instrument's main observable.
    """
    section = map_section(spec)
    stage = section.stage if section else [n for n in spec.settings if re.match(r"stage_[xyz](_|$)", n)]
    axes = {}
    for i, name in enumerate(stage):
        letter = re.match(r"stage_([xyz])(_|$)", name)
        axes[name] = letter.group(1) if letter else "xyz"[i]
    fov = section.field_of_view if section else None
    hints: dict[str, dict] = {}
    for name, quantity in spec.settings.items():
        returns = {k: v for k, v in _argument(quantity, axes.get(name)).items() if k not in ("name", "doc")}
        hints[f"get_{name}"] = {"risk": "low", "role": "stage.read" if name in axes else "setting.read", "returns": returns}
        if quantity.settable:
            role = "stage.move" if name in axes else "optics.fov" if name == fov else "setting.set"
            hints[f"set_{name}"] = {"risk": "high", "role": role, "args": [_argument(quantity, axes.get(name))]}
    for name, cmd in spec.commands.items():
        sets = {value: target for target, (value, _tree) in cmd.sets.items() if isinstance(value, str)}  # input -> the setting it sets
        hints[name] = {**cmd.declaration(), "args": [_argument(q, axes.get(sets.get(q.name, ""))) for q in cmd.inputs]}
    if section is not None:
        hints[MAP_COMMAND] = {"risk": "low", "role": "setting.read"}
    for name, acquisition in spec.acquisitions.items():
        hints[name] = {
            **({"primary": True} if section and name == section.image else {}),  # what a map tiles with: the main observable
            "risk": "high",
            "role": acquisition.role or KIND_ROLES[acquisition.kind],
            "args": [_argument(q) for q in acquisition.inputs],
            "returns": {"type": "object" if acquisition.kind == "value" else "data key", "kind": acquisition.kind, "units": acquisition.units},
        }
    return hints


def map_description(spec: InstrumentSpec, sim: "SimInstrument", overrides: dict | None = None) -> dict:
    """What the Mapper needs to tile this instrument, evaluated at the current settings with ``overrides``.

    ``overrides`` are settings a map would set (its field of view and plan settings), checked as a
    set_<name> would check them; Refused names the first one out of range.
    """
    section = map_section(spec)
    if section is None:
        raise Refused("this instrument cannot be mapped: it needs an image acquisition and stage x and y settings (or a map: section)")
    overrides = dict(overrides or {})
    allowed = set(section.settings) | ({section.field_of_view} if section.field_of_view else set())
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise Refused(f"a map cannot set {unknown}; it may set {sorted(allowed)}")
    with sim.lock:
        scope = dict(sim.values)
    for name, value in overrides.items():
        try:
            scope[name] = spec.settings[name].coerce(value)
        except Refused:
            raise
        except (TypeError, ValueError) as exc:
            raise Refused(f"{name}: {exc}") from None
    acquisition = spec.acquisitions[section.image]
    rows, cols = acquisition.raw["shape"]
    trees = acquisition.trees
    pixel = float(evaluate(trees["pixel_size_m"], scope) if trees.get("pixel_size_m") is not None else acquisition.raw["pixel_size_m"])

    def seconds(names: list[str]) -> float:
        return sum(duration_seconds(spec.commands[n].duration, scope) for n in names)

    tile_s = duration_seconds(acquisition.duration, scope) + seconds(section.before_move + section.after_move + section.safe)

    def limits(name: str) -> dict:
        q = spec.settings[name]
        return {"setting": name, "min": q.min, "max": q.max, "units": q.units}

    budgets = {}
    for name, budget in section.budgets.items():
        per_tile = float(evaluate(budget.per_tile[1], scope) if budget.per_tile[1] is not None else budget.per_tile[0])
        per_revisit = float(evaluate(budget.per_revisit[1], scope) if budget.per_revisit[1] is not None else budget.per_revisit[0])
        budgets[name] = {"per_tile": per_tile, "per_revisit": per_revisit, "combine": budget.combine, "units": budget.units, "doc": budget.doc}
    commands = [f"get_{n}" for n in section.stage] + [f"set_{n}" for n in section.stage] + [section.image]
    commands += [f"set_{n}" for n in sorted(allowed)] + section.before_move + section.after_move + section.safe + [MAP_COMMAND]
    return {
        "image": section.image,
        "detector": str(acquisition.raw.get("detector", "image")),
        "units": acquisition.units,
        "shape": [int(rows), int(cols)],
        "pixel_size_m": pixel,
        "tile_m": [cols * pixel, rows * pixel],
        "field_of_view": limits(section.field_of_view) if section.field_of_view else None,
        "stage": {"x": limits(section.stage[0]), "y": limits(section.stage[1])},
        "stage_xy_m": [float(scope[section.stage[0]]), float(scope[section.stage[1]])],
        "before_move": section.before_move,
        "after_move": section.after_move,
        "safe": section.safe,
        "settings": {n: {**limits(n), "value": scope[n]} for n in section.settings},
        "plan_settings": {n: scope[n] for n in overrides},
        "tile_s": tile_s,
        "budgets": budgets,
        "commands": list(dict.fromkeys(commands)),
    }


def resolve_path(path: str) -> Path:
    """The instrument file a ``yaml_path`` property names (absolute, or relative to the working directory or the repository)."""
    if not path:
        raise SpecError("the yaml_path device property is empty; set it to the instrument YAML file")
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() and not candidate.exists() and (PROJECT_DIR / candidate).exists():
        return PROJECT_DIR / candidate
    return candidate


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
    if raw.get("map") is not None:
        spec.map = _map(raw["map"], spec)
    names = spec.command_names()
    if map_section(spec) is not None:
        names.append(MAP_COMMAND)
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise SpecError(f"{path}: {duplicates} defined twice (settings make get_<name> and set_<name> commands)")
    if not names:
        raise SpecError(f"{path}: no settings, commands or acquisitions; nothing for an agent to call")
    if "action_hints" in spec.settings:
        raise SpecError(f"{path}: settings.action_hints: the device's own attribute has that name (its commands' roles); rename the setting")
    taken = sorted({name.lower() for name in (*names, *spec.settings)} & RESERVED_NAMES)
    if taken:
        raise SpecError(f"{path}: {taken} are the instrument device's own names; rename yours")
    for name, command in spec.commands.items():
        verb = _reads_as_action(command.doc) if name.startswith(("get_", "list_")) else None
        if verb:
            spec.warnings.append(
                f"{path}: commands.{name}: its doc reads as an action ({verb!r}: {command.doc[:60]!r}), but the safety policy "
                f"treats every get_ and list_ command as a read and never holds it for approval. If it acts, rename it "
                f"(e.g. {verb}_{name[4:] if name.startswith('get_') else name[5:]}), or mark it high in the MCP config: "
                f"mcp.safety.risk: {{\"<class_name>.{name}\": \"high\"}}"
            )
    for message in spec.warnings:  # stderr, so the device server's console and the check below both show it
        print(f"[yaml_sim] WARNING {message}", file=sys.stderr)
    return spec


def duration_seconds(duration: tuple[Any, ast.Expression | None], scope: dict[str, Any]) -> float:
    """A command's or acquisition's ``duration_s`` at the settings in ``scope``."""
    raw, tree = duration
    return float(evaluate(tree, scope) if tree is not None else raw)
