"""Log what the instrument reports, every minute, for hours or days.

The calibration scripts each take a few minutes, so they cannot see what changes over a night: the beam current
falling, focus and astigmatism being corrected again and again, the stage creeping. This logger sits beside whatever
the microscope is doing (a map, a user's session, nothing at all) and writes one line per reading to the Instrument
Trace (``outputs/trace/state.jsonl``; docs/Operation/trace.md), so the readings line up in time with every tool call
and job step.

It only reads. Every reading below is a status command or an attribute; the logger never moves the stage, changes a
setting or takes an image, so it is safe to leave running on a real instrument.

    uv run python -m asyncroscopy.calibration.trace_logger --device asyncroscopy/instrument/default --hours 48

``--summarise`` turns a finished log into numbers: how each reading moved, and the twin parameters that follow from
it. With ``--profile`` those go into an instrument profile (``profile.py``)::

    uv run python -m asyncroscopy.calibration.trace_logger --summarise outputs/trace/state.jsonl --profile configs/profiles/spectra300.yaml

What the summary can and cannot say: the defocus and stigmator are what the operator (or autofocus) set, so their
swing shows how much correction the column needed, which includes the specimen's height as well as the column's own
drift. Take the focus numbers from a log where the stage stayed in one place.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import tango

from asyncroscopy.calibration import profile as instrument_profile
from asyncroscopy.trace import read_trace
from asyncroscopy.trace.writer import TraceWriter

ACTION = "instrument.state"
DEFAULT_LOG = Path("outputs/trace/state.jsonl")

# Each reading: the name it is logged under, and the command (called with no argument) or attribute that gives it.
# All of them only read.
READINGS = {
    "stage": ("command", "get_stage"),  # [x, y, z (m), alpha, beta (deg)] as the stage reports it
    "defocus_m": ("command", "get_defocus"),
    "stigmator_m": ("command", "get_stigmator"),  # [a, b] in metres of A1
    "fov_m": ("command", "get_fov"),
    "beam_current_pa": ("attribute", "beam_current_pa"),
    "beam_current_source": ("attribute", "beam_current_source"),  # "measured" or "default"
    "stage_moving": ("attribute", "stage_moving"),
}


def read_state(microscope: tango.DeviceProxy) -> dict:
    """One reading of everything in READINGS. A reading the instrument cannot give is listed under ``unread``
    with the reason, and the rest are still logged."""
    state: dict = {}
    unread: dict[str, str] = {}
    for name, (kind, source) in READINGS.items():
        try:
            value = microscope.command_inout(source) if kind == "command" else microscope.read_attribute(source).value
        except tango.DevFailed as failed:
            unread[name] = str(failed.args[0].desc).strip()[:200]
            continue
        state[name] = value.tolist() if hasattr(value, "tolist") else value
    if unread:
        state["unread"] = unread
    return state


def instrument_clock(microscope: tango.DeviceProxy) -> float | None:
    """The instrument's own clock: simulated seconds on a twin, so a twin's log is in the twin's time."""
    try:
        return float(microscope.read_attribute("clock_s").value)
    except (tango.DevFailed, TypeError, ValueError):
        return None


def log(microscope: tango.DeviceProxy, path: Path, interval_s: float, readings: int) -> int:
    """Append ``readings`` lines to ``path``, one every ``interval_s`` seconds. Returns how many were written."""
    writer = TraceWriter(path, source="state", writer=microscope.dev_name(), clock=lambda: instrument_clock(microscope))
    for n in range(readings):
        started = time.time()
        state = read_state(microscope)
        writer.record(ACTION, ok="unread" not in state, started=started, **state)
        if n + 1 < readings:
            time.sleep(max(0.0, interval_s - (time.time() - started)))
    return readings


def _series(events: list[dict]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Each numeric reading as (hours since the first line, values); a list reading becomes one series per item."""
    t0 = events[0]["clock_s"]
    series: dict[str, tuple[list, list]] = {}
    for event in events:
        hours = (event["clock_s"] - t0) / 3600.0
        for name in READINGS:
            value = event.get(name)
            items = {name: value} if not isinstance(value, list) else {f"{name}[{i}]": v for i, v in enumerate(value)}
            for key, item in items.items():
                if isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item):
                    times, values = series.setdefault(key, ([], []))
                    times.append(hours)
                    values.append(float(item))
    return {key: (np.array(times), np.array(values)) for key, (times, values) in series.items()}


def summarise(path: str | Path) -> dict:
    """How each reading in a log moved: its mean, spread, total swing and steady change per hour."""
    events = [event for event in read_trace(path) if event.get("action") == ACTION]
    if len(events) < 2:
        raise ValueError(f"{path} holds {len(events)} state readings; a summary needs at least 2")
    readings = {}
    for name, (hours, values) in _series(events).items():
        readings[name] = {
            "n": len(values),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "swing": float(np.ptp(values)),
            "per_hour": float(np.polyfit(hours, values, 1)[0]) if np.ptp(hours) > 0 else 0.0,
        }
    measured_current = all(event.get("beam_current_source") == "measured" for event in events)
    return {
        "hours": float((events[-1]["clock_s"] - events[0]["clock_s"]) / 3600.0),
        "clock": events[0].get("clock", "wall"),
        "beam_current_measured": measured_current,
        "readings": readings,
    }


def twin_parameters(summary: dict) -> tuple[dict[str, float], dict[str, float]]:
    """(values, sigmas) of the twin parameters a log supports. The focus and astigmatism swings are halved because
    the twin's ``focus_drift_m`` is the size of one swing to either side."""
    readings = summary["readings"]
    values: dict[str, float] = {}
    sigmas: dict[str, float] = {}
    if summary["beam_current_measured"] and "beam_current_pa" in readings:
        values["probe_current_pa"] = readings["beam_current_pa"]["mean"]
        sigmas["probe_current_pa"] = readings["beam_current_pa"]["std"]
    if "defocus_m" in readings:
        values["focus_drift_m"] = readings["defocus_m"]["swing"] / 2.0
    stigmator = [readings[k]["swing"] for k in ("stigmator_m[0]", "stigmator_m[1]") if k in readings]
    if stigmator:
        values["astigmatism_drift_m"] = max(stigmator) / 2.0
    return values, sigmas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--device", default="asyncroscopy/instrument/default", help="Microscope Tango device.")
    parser.add_argument("--out", type=Path, default=DEFAULT_LOG, help="The log to append to.")
    parser.add_argument("--interval", type=float, default=60.0, help="Seconds between readings.")
    parser.add_argument("--hours", type=float, default=24.0, help="How long to log.")
    parser.add_argument("--summarise", type=Path, help="Summarise this log and stop; nothing is read from the instrument.")
    parser.add_argument("--profile", default="", help="With --summarise: the instrument profile to write the twin parameters into.")
    args = parser.parse_args()
    if args.summarise is None:
        readings = max(1, int(args.hours * 3600.0 / args.interval))
        print(f"logging {readings} readings of {args.device} to {args.out}, one every {args.interval:.0f} s (Ctrl+C stops)")
        try:
            log(tango.DeviceProxy(args.device), args.out, args.interval, readings)
        except KeyboardInterrupt:
            print("stopped")
        return
    summary = summarise(args.summarise)
    print(f"# {args.summarise}: {summary['hours']:.1f} h on the {summary['clock']} clock")
    for name, r in summary["readings"].items():
        print(f"    {name}: mean {r['mean']:.3e}, std {r['std']:.3e}, swing {r['swing']:.3e}, per hour {r['per_hour']:.3e} ({r['n']} readings)")
    values, sigmas = twin_parameters(summary)
    instrument_profile_note = ""
    if args.profile:
        instrument_profile.record(args.profile, values, source=f"trace_logger ({summary['hours']:.1f} h)", sigma=sigmas)
        instrument_profile_note = f" (written to {args.profile})"
    print(f"# twin parameters{instrument_profile_note}")
    for name, value in values.items():
        print(f"    {name}: {value:.3e}")


if __name__ == "__main__":
    main()
