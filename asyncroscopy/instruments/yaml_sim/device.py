"""A Tango instrument built from an instrument YAML file (see ``spec.py`` for the format).

Every setting becomes an attribute plus ``get_<name>`` (and ``set_<name>`` when
settable) commands; every command and acquisition in the file becomes a Tango
command, so the MCP server offers each one as an agent tool and the safety
policy classes it like any other command (``get_*`` read-only, ``set_*`` and
``acquire*`` held in approve mode, ``xray_off``-style stops always allowed).
Acquisitions are saved through the DATA device and return its key, so agents
look at them with ``view_data``.

A file with a ``jobs:`` section also gets a separate job device, ``YamlJob``
(job.py), named ``<Name>Job``: an agent runs a line or a grid of acquisitions
with ``run_job(device="<Name>Job", plan=...)`` and waits for it with
``wait_for_job``. A job is not limited to 25 s like a single call. While it
runs, ``set_*`` and ``acquire*`` calls are refused (commands such as a laser or
X-ray switch-off still work), and abort stops the running step within 0.2 s.

The job device drives this device through a small step protocol that agents
never see (the commands are ``operator_only``):

* ``hold(job_token)`` keeps ``set_*`` and ``acquire*`` calls from anyone else
  out. The hold is a lease: the job renews it with another ``hold``, and if it
  stops (the job device died), the hold lapses after ``hold_lease_s``, the
  running step stops and the file's ``on_stop`` commands make the instrument safe;
* ``start_step`` returns at once and runs the step on a worker thread, with no
  25 s limit; ``step_status`` reports its state, result and error;
* ``stop_step`` ends the running step at once (its wait is interrupted);
* ``release_hold`` gives the instrument back.

In a launcher config, give the class name the tools should carry and the file::

    instrument:
      class_name: XRDTwin
      module_name: asyncroscopy.instruments.yaml_sim.device
      properties:
        yaml_path: asyncroscopy/instruments/yaml_sim/examples/xrd.yaml
        data_device_address: asyncroscopy/data/default

The server takes its class name from the Tango database entry the launcher
writes (``<class_name>/<instance>``), so the tools are
``XRDTwin_get_two_theta_deg`` and so on, with no Python written for the
instrument. run_servers starts the job device beside it when the file has jobs.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass, field

import tango
from tango import AttrWriteType, CmdArgType, DevState
from tango.server import Device, device_property

from asyncroscopy.data.data_writer import save_acquisition
from asyncroscopy.instruments.instrument import Instrument
from asyncroscopy.instruments.yaml_sim.spec import (
    MAP_COMMAND,
    MAX_DURATION_S,
    SETTLE_STEP,
    Refused,
    SimInstrument,
    SpecError,
    action_hints,
    load_spec,
    map_description,
    map_section,
    resolve_path,
)
from asyncroscopy.utils.commands import action_hints_attribute, command

TANGO_TYPES = {float: CmdArgType.DevDouble, int: CmdArgType.DevLong64, bool: CmdArgType.DevBoolean, str: CmdArgType.DevString}
DATA_TIMEOUT_MS = 15_000  # DATA registers a file in Tiled within about 10 s
STOP_WAIT_S = 0.2  # stop_step waits this long for the step to end; only a step in the middle of saving takes longer
HOLD_WATCH_S = 0.05  # how often a hold's lease is checked


class _Stopped(Exception):
    """A step's wait was cut short by stop_step or a lapsed hold."""


@dataclass
class _Hold:
    """Who holds the instrument, until when (time.monotonic)."""

    token: str
    expires: float
    lapsed: bool = False  # true from the moment the lease ran out until the instrument is safe and free again
    ended: threading.Event = field(default_factory=threading.Event)


@dataclass
class _Step:
    """One job step running (or finished) on the instrument."""

    command: str
    args: object = None
    file_attrs: dict | None = None
    state: str = "running"  # running, done, failed or stopped
    result: str = ""
    error: str = ""
    stop_reason: str = ""
    stop: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None


def _named(func, name: str, doc: str):
    func.__name__, func.__doc__ = name, doc
    return func


def _refusing(func, *args, **kwargs):
    """Run ``func``; a Refused becomes a ValueError the client sees with its message."""
    try:
        return func(*args, **kwargs)
    except Refused as exc:
        raise ValueError(str(exc)) from None


class YamlInstrument(Instrument):
    """An instrument defined by a YAML file: its settings, commands and acquisitions, and the step protocol jobs drive it through."""

    action_hints = action_hints_attribute(extra=lambda device: device.spec_hints)

    yaml_path = device_property(dtype=str, default_value="", doc="The instrument YAML file (absolute, or relative to the working directory or the repository).")
    output_directory = device_property(dtype=str, default_value="outputs/yaml_sim", doc="Where acquisitions go when there is no DATA device.")
    time_scale = device_property(dtype=float, default_value=1.0, doc="Real seconds per simulated second of every duration_s (0: no waiting; tests and fast rehearsals).")
    hold_lease_s = device_property(dtype=float, default_value=10.0, doc="How long a job's hold lasts without being renewed; a job device that dies stops holding the instrument after this long.")

    # ------------------------------------------------------------------
    # Instrument
    # ------------------------------------------------------------------
    def read_instrument_type(self) -> str:
        return "YAML_SIM"

    def _init_device_attributes(self) -> None:
        self.acquisition_save_directory = self.output_directory
        self.sim = None
        self.spec_hints: dict = {}
        self._hold_lock = threading.RLock()  # guards _active_hold and _step, which Tango threads, the step worker and the watcher share
        self._active_hold: _Hold | None = None
        self._step: _Step | None = None
        self._scale = max(0.0, float(self.time_scale))
        try:
            spec = load_spec(resolve_path(self.yaml_path))
        except SpecError as exc:
            self._load_error = f"{type(self).__name__}: the instrument file is not valid: {exc}"
            print(self._load_error, file=sys.stderr, flush=True)
            return
        self._load_error = None
        self.sim = SimInstrument(spec, sleep=self._direct_sleep)
        for name, quantity in spec.settings.items():
            self._build_setting(name, quantity)
        for name, cmd in spec.commands.items():
            self._build_command(name, cmd)
        for name, acquisition in spec.acquisitions.items():
            self._build_acquisition(name, acquisition)
        if map_section(spec) is not None:
            self._build_map_profile()
        self.spec_hints = action_hints(spec)

    def _connect(self) -> None:
        if self.sim is None:
            self._connect_error = self._load_error
            self.set_state(DevState.FAULT)
            self.set_status(self._load_error)
            return
        self._connect_error = None
        spec = self.sim.spec
        self.set_state(DevState.ON)
        self.set_status(spec.description or f"Simulated instrument from {spec.path.name}")

    def _disconnect(self) -> None:
        self._drop_hold("the instrument was disconnected")

    def delete_device(self) -> None:
        self._drop_hold("the device was re-initialised")
        Device.delete_device(self)

    # ------------------------------------------------------------------
    # The file's settings, commands and acquisitions
    # ------------------------------------------------------------------
    def _scaled_sleep(self, seconds: float) -> None:
        if seconds and self._scale:
            time.sleep(seconds * self._scale)

    def _direct_sleep(self, seconds: float) -> None:
        """The simulated wait of a call made directly (not as a job step)."""
        if self._active_hold is not None:  # free Tango's per-device monitor, so a job's short calls never wait behind this one
            with tango.AutoTangoAllowThreads(self):
                self._scaled_sleep(seconds)
        else:
            self._scaled_sleep(seconds)

    def _unless_held(self, func, *args, **kwargs):
        with self._hold_lock:
            hold = self._active_hold
        if hold is not None:
            tango.Except.throw_exception(
                "InstrumentHeld", f"busy: job {hold.token} holds this instrument; wait for it (wait_for_job) or abort it first", "YamlInstrument",
            )
        return _refusing(func, *args, **kwargs)

    def _build_setting(self, name, quantity) -> None:
        sim = self.sim

        def read(attr):
            attr.set_value(sim.get(name))

        def write(attr):
            self._unless_held(sim.set, name, attr.get_write_value())

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
                self._unless_held(sim.set, name, value)

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

    def _build_command(self, name, cmd) -> None:
        self._add_call(name, cmd.doc, cmd.inputs, lambda argument: _refusing(self.sim.run_command, name, argument))

    def _build_acquisition(self, name, acquisition) -> None:
        jobs = [job.name for job in self.sim.spec.jobs.values() if name in job.acquire]
        doc = acquisition.doc + (" Returns the data key; look at it with view_data." if acquisition.kind != "value" else " Returns JSON.")
        if jobs:
            doc += f" For many points, or more than 25 s, run the {jobs[0]} job (run_job on {type(self).__name__}Job)."
        self._add_call(name, doc, acquisition.inputs, lambda argument: self._unless_held(self._acquire_and_save, name, argument))

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

    # ------------------------------------------------------------------
    # The step protocol: how a job device drives this instrument
    # ------------------------------------------------------------------
    @command(dtype_in=str, dtype_out=float, doc_in=":param job_token: Names the job (and is needed for every later step of it).", risk="high", operator_only=True, role="other")
    def hold(self, job_token: str) -> float:
        """Keep every other set_* and acquire* call out for this job, or renew its hold. Returns the lease in seconds; renew within it."""
        lease = float(self.hold_lease_s)
        with self._hold_lock:
            hold = self._active_hold
            if hold is not None and (hold.token != job_token or hold.lapsed):
                tango.Except.throw_exception("InstrumentHeld", f"job {hold.token} holds this instrument (its hold ends in at most {lease:g} s if nobody renews it)", "hold()")
            if hold is None:
                hold = self._active_hold = _Hold(token=job_token, expires=time.monotonic() + lease)
                threading.Thread(target=self._watch_hold, args=(hold,), name=f"hold-{job_token}", daemon=True).start()
            else:
                hold.expires = time.monotonic() + lease
        return lease

    @command(dtype_in=str, doc_in=":param job_token: The token the instrument was held with.", risk="low", operator_only=True, role="other")
    def release_hold(self, job_token: str) -> None:
        """Give the instrument back (stopping any step still running); a hold that has already lapsed is gone, so this then does nothing."""
        with self._hold_lock:
            hold = self._active_hold
            if hold is None or hold.lapsed:
                return
            if hold.token != job_token:
                tango.Except.throw_exception("InstrumentHeld", f"job {hold.token} holds this instrument, not {job_token}", "release_hold()")
        self._drop_hold(f"job {job_token} released the instrument")

    @command(dtype_in=str, doc_in=':param request: JSON {"job_token": "...", "command": "set_x, a command or an acquisition", "args": value, null or an object of the arguments, "file_attrs": {...}, optional, saved with the data}', risk="high", operator_only=True, role="other")
    def start_step(self, request: str) -> None:
        """Start one job step on a worker thread and return at once; poll step_status for the outcome."""
        body = json.loads(request)
        token, name = body["job_token"], body["command"]
        with self._hold_lock:
            self._require_token(token, "start_step()")
            if self._step is not None and self._step.state == "running":
                tango.Except.throw_exception("StepRunning", f"step {self._step.command} is still running; wait for it (step_status) or stop it", "start_step()")
            if not self._knows_step(name):
                tango.Except.throw_exception("UnknownStep", f"{name!r} is not a setting (set_<name>), a command or an acquisition of this instrument", "start_step()")
            step = _Step(command=name, args=body.get("args"), file_attrs=body.get("file_attrs"))
            step.thread = threading.Thread(target=self._run_step, args=(step,), name=f"step-{name}", daemon=True)
            self._step = step
            step.thread.start()

    @command(dtype_out=str, risk="low", operator_only=True, role="other")
    def step_status(self) -> str:
        """The latest step as JSON: state (idle, running, done, failed or stopped), command, result and error. Answers at once."""
        step = self._step
        if step is None:
            return json.dumps({"state": "idle", "command": None, "result": "", "error": ""})
        return json.dumps({"state": step.state, "command": step.command, "result": step.result, "error": step.error})

    @command(dtype_in=str, doc_in=":param job_token: The token the instrument was held with.", risk="low", operator_only=True, role="other")
    def stop_step(self, job_token: str) -> None:
        """End the running step now: its wait is interrupted, it is reported as stopped, and nothing it was going to set is set."""
        with self._hold_lock:
            self._require_token(job_token, "stop_step()")
            step = self._step
        self._stop(step, f"job {job_token} stopped it")

    def _require_token(self, token: str, origin: str) -> None:
        hold = self._active_hold
        if hold is None or hold.lapsed:
            tango.Except.throw_exception("NotHeld", "this instrument is not held (its hold lapsed, or none was taken); call hold first", origin)
        if hold.token != token:
            tango.Except.throw_exception("InstrumentHeld", f"job {hold.token} holds this instrument, not {token}", origin)

    def _knows_step(self, name: str) -> bool:
        spec = self.sim.spec
        return name == SETTLE_STEP or name in spec.commands or name in spec.acquisitions or (name.startswith("set_") and name[4:] in spec.settings)

    def _run_step(self, step: _Step) -> None:
        """The step worker: one command, setting, acquisition or wait, with no time limit and stoppable."""

        def sleep(seconds: float) -> None:
            if step.stop.wait(seconds * self._scale):
                raise _Stopped()

        with tango.EnsureOmniThread():  # a thread Tango did not create, about to call the DATA device
            try:
                step.result = self._step_outcome(step, sleep) or ""
                step.state = "done"
            except _Stopped:
                step.error, step.state = step.stop_reason, "stopped"
            except Refused as exc:
                step.error, step.state = str(exc), "failed"
            except Exception as exc:
                step.error, step.state = f"{type(exc).__name__}: {exc}", "failed"

    def _step_outcome(self, step: _Step, sleep) -> str:
        name, args, spec = step.command, step.args, self.sim.spec
        if name == SETTLE_STEP:
            sleep(float(args))
            return ""
        if name in spec.acquisitions:
            return self._acquire_and_save(name, args, limit=None, sleep=sleep, file_attrs=step.file_attrs)
        if name in spec.commands:
            return self.sim.run_command(name, args, sleep=sleep)
        self.sim.set(name[4:], args)
        return ""

    def _stop(self, step: _Step | None, reason: str) -> None:
        if step is None or step.state != "running":
            return
        step.stop_reason = reason
        step.stop.set()
        step.thread.join(STOP_WAIT_S)

    def _watch_hold(self, hold: _Hold) -> None:
        """End the hold when its lease runs out: the job device is gone, so nobody else will make the instrument safe."""
        with tango.EnsureOmniThread():
            while not hold.ended.wait(HOLD_WATCH_S):
                with self._hold_lock:
                    if self._active_hold is not hold or time.monotonic() < hold.expires:
                        continue
                    hold.lapsed = True
                    step = self._step
                self.warn_stream(f"The hold of job {hold.token} lapsed (not renewed for {float(self.hold_lease_s):g} s); stopping its step and making the instrument safe")
                self._stop(step, f"the hold of job {hold.token} lapsed: nobody renewed it")
                self._make_safe()
                self._drop_hold("")
                return

    def _make_safe(self) -> None:
        """Run every job's on_stop commands (which job held the instrument is not known here), each at most once."""
        spec = self.sim.spec
        for name in dict.fromkeys(name for job in spec.jobs.values() for name in job.on_stop):
            try:
                self.sim.run_command(name, sleep=self._scaled_sleep)
            except Exception as exc:  # the others must still run
                self.warn_stream(f"on_stop command {name} failed after the hold lapsed: {exc}")

    def _drop_hold(self, reason: str) -> None:
        with self._hold_lock:
            hold, step = self._active_hold, self._step
        if hold is None:
            return
        self._stop(step, reason)
        with self._hold_lock:
            if self._active_hold is hold:
                self._active_hold = None
                hold.ended.set()


_classes: dict[str, type] = {}


def device_class(name: str) -> type:
    """An instrument class called ``name`` (the MCP server names tools after the Tango class)."""
    if name == YamlInstrument.__name__:
        return YamlInstrument
    if name not in _classes:
        _classes[name] = type(name, (YamlInstrument,), {"__module__": __name__})
    return _classes[name]


def registered_class(instance: str, default: str) -> str:
    """The class the launcher registered for this server instance (``<class_name>/<instance>``), or ``default``."""
    try:
        # Listed whole and matched here: PyTango's own database (the one run_servers starts)
        # returns nothing for a pattern starting with a wildcard such as "*/<instance>".
        servers = [s for s in tango.Database().get_server_list("*").value_string if s.split("/", 1)[-1].lower() == instance.lower()]
    except tango.DevFailed:
        return default
    return servers[0].split("/", 1)[0] if servers else default


if __name__ == "__main__":
    device_class(registered_class(sys.argv[1], YamlInstrument.__name__) if len(sys.argv) > 1 else YamlInstrument.__name__).run_server()
