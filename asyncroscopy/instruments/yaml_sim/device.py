"""A Tango device built from an instrument YAML file (see ``spec.py`` for the format).

Every setting becomes an attribute plus ``get_<name>`` (and ``set_<name>`` when
settable) commands; every command and acquisition in the file becomes a Tango
command, so the MCP server offers each one as an agent tool and the safety
policy classes it like any other command (``get_*`` read-only, ``set_*`` and
``acquire*`` held in approve mode, ``xray_off``-style stops always allowed).
Acquisitions are saved through the DATA device and return its key, so agents
look at them with ``view_data``.

A file with a ``jobs:`` section gives a job device instead: the same commands
plus the job envelope (reserve, start, job_status, abort, results, release), so
an agent runs a line or a grid of acquisitions with ``run_job`` and waits for it
with ``wait_for_job``. A job is not limited to 25 s like a single call; while
it runs, ``set_*`` and ``acquire*`` calls are refused (commands such as a
laser or X-ray switch-off still work), and abort stops it within one wait.

In a launcher config, give the class name the tools should carry and the file::

    instrument:
      class_name: XRDTwin
      module_name: asyncroscopy.instruments.yaml_sim.device
      properties:
        yaml_path: asyncroscopy/instruments/yaml_sim/examples/xrd.yaml
        data_device_address: asyncroscopy/data/default

The server takes its class name from the Tango database entry the launcher
writes (``<class_name>/<instance>``), and reads the file then to see whether it
has jobs, so the tools are ``XRDTwin_get_two_theta_deg`` and so on, with no
Python written for the instrument.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import tango
from tango import AttrWriteType, CmdArgType, DevState
from tango.server import Device, device_property

from asyncroscopy.data.data_writer import save_acquisition
from asyncroscopy.instruments.yaml_sim.spec import (
    MAP_COMMAND,
    MAX_DURATION_S,
    JobPlan,
    Refused,
    SimInstrument,
    SpecError,
    action_hints,
    job_estimate,
    job_targets,
    load_spec,
    map_description,
    map_section,
    plan_example,
    resolve_job_plan,
)
from asyncroscopy.jobs.job_device import RUNNING_STATES, JobDevice
from asyncroscopy.core.names import ACTION_HINTS
from asyncroscopy.utils.commands import command

PROJECT_DIR = Path(__file__).resolve().parents[3]
TANGO_TYPES = {float: CmdArgType.DevDouble, int: CmdArgType.DevLong64, bool: CmdArgType.DevBoolean, str: CmdArgType.DevString}
DATA_TIMEOUT_MS = 15_000  # DATA registers a file in Tiled within about 10 s
KEYS_IN_SUMMARY = 200  # data keys per acquisition a job's results list (the newest); the rest are in its step records
ABORT_POLL_S = 0.2


def _yaml_path_property():
    return device_property(dtype=str, default_value="", doc="The instrument YAML file (absolute, or relative to the working directory or the repository).")


def _data_property():
    return device_property(dtype=str, default_value="", doc="DATA device that saves and registers acquisitions; empty saves files under output_directory.")


def _output_property():
    return device_property(dtype=str, default_value="outputs/yaml_sim", doc="Where acquisitions go when there is no DATA device.")


def _time_scale_property():
    return device_property(dtype=float, default_value=1.0, doc="Real seconds per simulated second of every duration_s (0: no waiting; tests and fast rehearsals).")


class _Stopped(Exception):
    """A job's wait was cut short by an abort."""


def _resolve(path: str) -> Path:
    if not path:
        raise SpecError("the yaml_path device property is empty; set it to the instrument YAML file")
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() and not candidate.exists() and (PROJECT_DIR / candidate).exists():
        return PROJECT_DIR / candidate
    return candidate


def _named(func, name: str, doc: str):
    func.__name__, func.__doc__ = name, doc
    return func


def _refusing(func, *args, **kwargs):
    """Run ``func``; a Refused becomes a ValueError the client sees with its message."""
    try:
        return func(*args, **kwargs)
    except Refused as exc:
        raise ValueError(str(exc)) from None


class _YamlInstrument:
    """What both device classes do: load the file and add its settings, commands and acquisitions."""

    def _load(self) -> bool:
        self.acquisition_save_directory = self.output_directory
        self.sim = None
        try:
            spec = load_spec(_resolve(self.yaml_path))
        except SpecError as exc:
            message = f"{type(self).__name__}: the instrument file is not valid: {exc}"
            print(message, file=sys.stderr, flush=True)
            self.set_state(DevState.FAULT)
            self.set_status(message)
            return False
        scale = max(0.0, float(getattr(self, "time_scale", 1.0)))
        self.sim = SimInstrument(spec, sleep=lambda seconds: time.sleep(seconds * scale) if scale else None)
        for name, quantity in spec.settings.items():
            self._build_setting(name, quantity)
        for name, cmd in spec.commands.items():
            self._build_command(name, cmd)
        for name, acquisition in spec.acquisitions.items():
            self._build_acquisition(name, acquisition)
        if map_section(spec) is not None:
            self._build_map_profile()
        self._build_action_hints(action_hints(spec))
        self.set_state(DevState.ON)
        self.set_status(spec.description or f"Simulated instrument from {spec.path.name}")
        return True

    def busy(self) -> str | None:
        """Why settings and acquisitions are refused now (a running job), or None."""
        return None

    def _unless_busy(self, func, *args, **kwargs):
        reason = self.busy()
        if reason:
            raise ValueError(reason)
        return _refusing(func, *args, **kwargs)

    def _build_setting(self, name, quantity) -> None:
        sim = self.sim

        def read(attr):
            attr.set_value(sim.get(name))

        def write(attr):
            self._unless_busy(sim.set, name, attr.get_write_value())

        access = AttrWriteType.READ_WRITE if quantity.settable else AttrWriteType.READ
        attr = tango.Attr(name, TANGO_TYPES[quantity.type], access)
        props = tango.UserDefaultAttrProp()
        props.set_description(quantity.describe())
        if quantity.units:
            props.set_unit(quantity.units)
        attr.set_default_properties(props)
        # Tango calls attribute methods with (device, attribute), unlike dynamic commands.
        self.add_attribute(attr, r_meth=lambda _device, a: read(a), w_meth=(lambda _device, a: write(a)) if quantity.settable else None)

        def get():
            return sim.get(name)

        get.__name__ = f"get_{name}"
        get.__doc__ = f"Current {quantity.describe()[0].lower()}{quantity.describe()[1:]}"
        self.add_command(command(f=get, dtype_out=quantity.type), device_level=True)
        if quantity.settable:

            def set_(value):
                self._unless_busy(sim.set, name, value)

            set_.__name__ = f"set_{name}"
            set_.__doc__ = f"Set the {quantity.describe()[0].lower()}{quantity.describe()[1:]}"
            self.add_command(command(f=set_, dtype_in=quantity.type, doc_in=f":param {name}: {quantity.describe()}"), device_level=True)

    def _add_call(self, name: str, doc: str, inputs, run, dtype_out=str) -> None:
        """A command taking no argument, one typed argument, or several (a JSON object the MCP server shows field by field)."""
        if not inputs:

            def call():
                return run(None)

            call.__name__, call.__doc__ = name, doc
            self.add_command(command(f=call, dtype_out=dtype_out), device_level=True)
        elif len(inputs) == 1:

            def call_with(value):
                return run(value)

            call_with.__name__, call_with.__doc__ = name, doc
            self.add_command(command(f=call_with, dtype_in=inputs[0].type, dtype_out=dtype_out, doc_in=f":param {inputs[0].name}: {inputs[0].describe()}"), device_level=True)
        else:

            def call_with_request(request):
                return run(request)

            fields = ", ".join(f'"{q.name}": {q.describe()}' for q in inputs)
            call_with_request.__name__, call_with_request.__doc__ = name, doc
            self.add_command(command(f=call_with_request, dtype_in=str, dtype_out=dtype_out, doc_in=f":param request: JSON {{{fields}}}"), device_level=True)

    def _build_map_profile(self) -> None:
        """``get_map_profile``: what the Mapper (instrument_profile yaml) needs to tile this instrument, as JSON."""

        def describe(request: str) -> str:
            try:
                overrides = (json.loads(request) if request.strip() else {}).get("settings") or {}
                return json.dumps(map_description(self.sim.spec, self.sim, overrides))
            except (Refused, ValueError, AttributeError) as exc:  # the caller's settings: say why, never fail the device
                return json.dumps({"error": str(exc)})

        doc = ("How a Mapper maps this instrument: its image acquisition, stage and field-of-view settings and limits, the commands run "
               "around moves and between tiles, the time per tile and the map budgets, at the current settings with the given ones.")
        self.add_command(command(f=_named(describe, MAP_COMMAND, doc), dtype_in=str, dtype_out=str, doc_in=':param request: JSON {"settings": {name: value}} (may be {})'), device_level=True)

    def _build_action_hints(self, hints: dict) -> None:
        """The ``action_hints`` attribute: each command's role, arguments (units, ranges, stage axis) and returns,
        as JSON. The MCP server's action space reads it (asyncroscopy.mcp.action_space)."""
        text = json.dumps(hints)
        attr = tango.Attr(ACTION_HINTS, CmdArgType.DevString, AttrWriteType.READ)
        props = tango.UserDefaultAttrProp()
        props.set_description("Each command's canonical role, arguments and returns, as JSON (for the MCP action space).")
        attr.set_default_properties(props)
        self.add_attribute(attr, r_meth=lambda _device, a: a.set_value(text))

    def _build_command(self, name, cmd) -> None:
        self._add_call(name, cmd.doc, cmd.inputs, lambda argument: _refusing(self.sim.run_command, name, argument))

    def _build_acquisition(self, name, acquisition) -> None:
        jobs = [job.name for job in self.sim.spec.jobs.values() if name in job.acquire]
        doc = acquisition.doc + (" Returns the data key; look at it with view_data." if acquisition.kind != "value" else " Returns JSON.")
        if jobs:
            doc += f" For many points, or more than 25 s, run the {jobs[0]} job (run_job)."
        self._add_call(name, doc, acquisition.inputs, lambda argument: self._unless_busy(self._acquire_and_save, name, argument))

    def _acquire_and_save(self, name: str, argument=None, limit: float | None = MAX_DURATION_S, sleep=None, file_attrs: dict | None = None) -> str:
        """Run one acquisition and save it through DATA; the data key (or JSON for a value acquisition)."""
        acquisition_type, detector, data, attrs = self.sim.acquire(name, argument, limit=limit, sleep=sleep)
        if data is None:
            return json.dumps(attrs)
        proxy = None
        if self.data_device_address:
            proxy = tango.DeviceProxy(self.data_device_address)
            proxy.set_timeout_millis(DATA_TIMEOUT_MS)
        return save_acquisition(
            self, proxy, acquisition_type, detector, data,
            dataset_name="spectrum" if data.ndim == 1 else "image",
            dataset_attrs=attrs,
            file_attrs={"instrument": type(self).__name__, "description": self.sim.spec.description, **(file_attrs or {})},
        )


class YamlSimDevice(_YamlInstrument, Device):
    yaml_path = _yaml_path_property()
    data_device_address = _data_property()
    output_directory = _output_property()
    time_scale = _time_scale_property()

    def init_device(self) -> None:
        Device.init_device(self)
        self.instrument_id = self.get_name()  # what every dataset it saves records as instrument_id
        self._load()


class YamlSimJobDevice(_YamlInstrument, JobDevice):
    """An instrument file with jobs: its commands plus the job envelope, so run_job runs a job."""

    # No dry runs: the steps drive the simulator directly, not through proxy(), so nothing would be recorded.
    SUPPORTS_DRY_RUN = False

    yaml_path = _yaml_path_property()
    data_device_address = _data_property()
    output_directory = _output_property()
    time_scale = _time_scale_property()

    def init_device(self) -> None:
        JobDevice.init_device(self)
        self._plans: dict[str, JobPlan] = {}
        if not self._load():
            return
        spec = self.sim.spec
        if not spec.jobs:
            self.set_state(DevState.FAULT)
            self.set_status(f"{type(self).__name__}: {spec.path.name} has no jobs: section; start it with the plain YamlSimDevice")
            return
        first = next(iter(spec.jobs.values()))
        kinds = "; ".join(f"{job.name}: {job.doc}" for job in spec.jobs.values())
        # The MCP server shows these with the job tools; set on the class, which the envelope reads.
        type(self).JOB_KIND = kinds
        type(self).PLAN_EXAMPLE = plan_example(spec, first)

    def busy(self) -> str | None:
        with self._lock:
            job = self._active()
            if job is not None and job["state"] in RUNNING_STATES:
                return f"busy: job {job.get('job_id', job['reservation_id'])} is {job['state']}; wait for it (wait_for_job) or abort it first"
        return None

    # -- job hooks --------------------------------------------------------
    def _plan(self, job: dict) -> JobPlan:
        plan = self._plans.get(job["reservation_id"])
        if plan is None:
            plan = self._plans[job["reservation_id"]] = resolve_job_plan(self.sim.spec, job["plan"], dict(self.sim.values))
        return plan

    def resolve_plan(self, plan: dict) -> dict:
        if self.sim is None:
            raise ValueError(f"{type(self).__name__} has no valid instrument file: {self.get_status()}")
        values = dict(self.sim.values)
        resolved = resolve_job_plan(self.sim.spec, plan, values)
        return {
            "job": resolved.job.name,
            "step_count": resolved.count,
            "settings": resolved.settings,
            "axes": {k: {"start": a, "stop": b, "points": n} for k, (a, b, n) in resolved.axes.items()},
            "estimate": {"steps": resolved.count, **job_estimate(self.sim.spec, self.sim, resolved, values)},
        }

    def step_targets(self, job: dict) -> list[dict]:
        resolved = resolve_job_plan(self.sim.spec, job["plan"], dict(self.sim.values))
        return [
            {"step": step, "device": self._name, "attribute": setting, "value": value, "min": low, "max": high}
            for step, setting, value, low, high in job_targets(self.sim.spec, resolved)
        ]

    def step_at(self, job: dict, index: int) -> dict:
        return self._plan(job).point(index)

    def phase(self, job: dict) -> str | None:
        return f"{self._plan(job).job.name}: acquiring" if job.get("state") == "running" else None

    def _sleeper(self, job: dict):
        def sleep(seconds: float) -> None:
            end = time.monotonic() + seconds
            while True:
                if self.aborting(job):
                    raise _Stopped()
                remaining = end - time.monotonic()
                if remaining <= 0:
                    return
                time.sleep(min(ABORT_POLL_S, remaining))

        return sleep

    def on_start(self, job: dict) -> None:
        for setting, value in self._plan(job).settings.items():
            self.sim.set(setting, value)

    def run_step(self, job: dict, index: int) -> dict | None:
        plan = self._plan(job)
        point = plan.point(index)
        sleep = self._sleeper(job)
        try:
            for name in plan.job.before_move:  # e.g. retract a probe before the stage moves; a refusal fails the job
                self.sim.run_command(name)
            for setting, value in point.items():
                self.sim.set(setting, value)
            if plan.job.settle_s:
                sleep(plan.job.settle_s)
            for name in plan.job.after_move:  # e.g. approach the probe again at the new point
                self.sim.run_command(name)
            keys = {}
            for name in plan.job.acquire:
                keys[name] = self._acquire_and_save(
                    name, limit=None, sleep=sleep,
                    file_attrs={"job": plan.job.name, "job_id": job.get("job_id", ""), "point_index": index, "point": json.dumps(point)},
                )
        except _Stopped:
            return None
        return {"index": index, "point": point, "keys": keys, "key": next(iter(keys.values()), None)}

    def _run_commands(self, names: list[str]) -> list[str]:
        problems = []
        for name in names:
            try:
                self.sim.run_command(name)
            except Exception as exc:  # report, but never mask why the job stopped
                problems.append(f"{name}: {exc}")
        return problems

    def on_stop(self, job: dict, reason: str) -> None:
        problems = self._run_commands(self._plan(job).job.on_stop)
        if problems:
            raise RuntimeError(f"on_stop commands failed: {'; '.join(problems)}")

    def on_finish(self, job: dict) -> dict:
        problems = self._run_commands(self._plan(job).job.on_finish)
        return {"on_finish_errors": "; ".join(problems)} if problems else {}

    def summary(self, job: dict) -> dict:
        plan = self._plan(job)
        records = job.get("step_results", {})
        keys: dict[str, list] = {name: [] for name in plan.job.acquire}
        for index in sorted(records, key=int):
            for name, key in (records[index].get("keys") or {}).items():
                keys.setdefault(name, []).append(key)
        shown = {name: values[-KEYS_IN_SUMMARY:] for name, values in keys.items()}
        omitted = {name: len(values) - len(shown[name]) for name, values in keys.items() if len(values) > KEYS_IN_SUMMARY}
        return {"job": plan.job.name, "points": plan.count, "keys": shown, **({"keys_omitted": omitted} if omitted else {})}


_classes: dict[tuple[str, bool], type] = {}


def device_class(name: str, jobs: bool = False) -> type:
    """A device class called ``name`` (the MCP server names tools after the Tango class): with the job envelope when ``jobs``."""
    base = YamlSimJobDevice if jobs else YamlSimDevice
    if name == base.__name__:
        return base
    if (name, jobs) not in _classes:
        _classes[(name, jobs)] = type(name, (base,), {"__module__": __name__})
    return _classes[(name, jobs)]


def _registration(instance: str) -> tuple[str, bool]:
    """The class the launcher registered for this server instance (``<class_name>/<instance>``), and whether its file has jobs."""
    try:
        database = tango.Database()
        # Listed whole and matched here: PyTango's own database (the one run_servers starts)
        # returns nothing for a pattern starting with a wildcard such as "*/<instance>".
        servers = [s for s in database.get_server_list("*").value_string if s.split("/", 1)[-1].lower() == instance.lower()]
        if not servers:
            return YamlSimDevice.__name__, False
        class_name = servers[0].split("/", 1)[0]
        listing = list(database.get_device_class_list(servers[0]).value_string)
        devices = [listing[i] for i in range(0, len(listing) - 1, 2) if listing[i + 1] == class_name]
        paths = database.get_device_property(devices[0], "yaml_path")["yaml_path"] if devices else []
    except tango.DevFailed:
        return YamlSimDevice.__name__, False
    try:
        return class_name, bool(paths) and bool(load_spec(_resolve(paths[0])).jobs)
    except SpecError:
        return class_name, False  # the device starts, finds the same problem and reports it as FAULT


if __name__ == "__main__":
    registered, with_jobs = _registration(sys.argv[1]) if len(sys.argv) > 1 else (YamlSimDevice.__name__, False)
    device_class(registered, jobs=with_jobs).run_server()
