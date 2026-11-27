"""Conformance check for an ``ElectronMicroscope`` backend (any vendor, or a twin).

Run it on every new backend, first against its simulator, then on the
instrument. Read-only unless you ask for more:

    uv run python -m asyncroscopy.conformance asyncroscopy/instrument/default
    uv run python -m asyncroscopy.conformance asyncroscopy/instrument/default --writes --acquire --json

What it checks:

* every read (``get_stage``, ``get_fov``, ``get_defocus``, ``get_parameters``, ...)
  either works and returns the documented shape and units, or fails with the
  base class's clean "does not implement" error, never a crash or a hang;
* units are plausible: stage x/y/z in metres (a value above 1 cm is almost
  certainly nanometres or micrometres), tilts in degrees, field of view in
  metres, ``get_parameters`` is a JSON object;
* ``--writes``: set/get round trips that write back the *current* value
  (field of view, defocus) and a zero-length stage move. Nothing should move,
  but these do talk to the hardware;
* ``--acquire``: one acquisition with the scanning detectors, checked for a
  DATA/Tiled key or file path. The beam is used.

It ends with **feature readiness**: which asyncroscopy features this backend
can run (imaging, the specimen atlas, recipes that move the stage, EDS,
autofocus, ...), so an adopter knows what works before the first session.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

PASS, FAIL, WARN, SKIP, UNSUPPORTED = "PASS", "FAIL", "WARN", "SKIP", "UNSUPPORTED"

FEATURES = {
    "imaging": ["acquire_scanned_image", "set_fov", "get_fov"],
    "stage navigation": ["get_stage", "move_stage"],
    "specimen atlas (verified return-to-site)": ["get_stage", "move_stage", "set_fov", "get_fov",
                                                 "acquire_scanned_image"],
    "focus control": ["get_defocus", "set_defocus"],
    "autofocus": ["auto_focus"],
    "beam blanking / supervision STOP": ["blank_beam", "unblank_beam"],
    "point spectroscopy (EDS/EELS)": ["place_beam", "acquire_spectrum"],
    "beam tilt / image shift": ["get_beam_tilt", "set_beam_tilt", "get_image_shift", "set_image_shift"],
    "status for agents (get_parameters)": ["get_parameters"],
}


@dataclass
class Result:
    level: str
    name: str
    detail: str = ""

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class Report:
    device: str
    dev_class: str = ""
    results: list[Result] = field(default_factory=list)
    supported: dict[str, bool] = field(default_factory=dict)      # command -> works
    features: dict[str, dict] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(r.level == FAIL for r in self.results)

    def add(self, level: str, name: str, detail: str = "") -> None:
        self.results.append(Result(level, name, detail))

    def to_json(self) -> dict:
        return {"device": self.device, "dev_class": self.dev_class, "ok": self.ok,
                "results": [r.to_json() for r in self.results], "supported": self.supported,
                "features": self.features}


def _err_text(exc: BaseException) -> str:
    args = getattr(exc, "args", ())
    if args and hasattr(args[0], "desc"):
        return " ".join(f"{getattr(e, 'reason', '')}: {' '.join(str(getattr(e, 'desc', '')).split())}"
                        for e in args[:1])
    return f"{type(exc).__name__}: {exc}"


def _not_implemented(exc: BaseException) -> bool:
    text = _err_text(exc)
    return "NotImplementedError" in text or "does not implement" in text or "not supported" in text.lower()


class Checker:
    def __init__(self, proxy, device: str, *, timeout_s: float = 30.0):
        self.proxy = proxy
        self.report = Report(device)
        self.timeout_s = timeout_s
        try:
            self.commands = {c.cmd_name for c in proxy.command_list_query()}
        except Exception:
            self.commands = set()

    def call(self, name: str, *args) -> tuple[str, Any]:
        """Run a command; returns (outcome, value_or_error) where outcome is ok / unsupported / error / missing."""
        if self.commands and name not in self.commands:
            self.report.supported[name] = False
            return "missing", f"the device has no {name} command"
        t0 = time.monotonic()
        try:
            value = self.proxy.command_inout(name, *args) if args else self.proxy.command_inout(name)
        except Exception as exc:
            self.report.supported[name] = False
            if _not_implemented(exc):
                return "unsupported", _err_text(exc)
            return "error", _err_text(exc)
        elapsed = time.monotonic() - t0
        self.report.supported[name] = True
        if elapsed > self.timeout_s:
            self.report.add(WARN, name, f"took {elapsed:.1f} s")
        return "ok", value

    def read(self, name: str, validate: Callable[[Any], str | None] | None = None) -> Any:
        outcome, value = self.call(name)
        if outcome == "ok":
            problem = validate(value) if validate else None
            if problem:
                self.report.add(FAIL, name, problem)
                self.report.supported[name] = False
            else:
                self.report.add(PASS, name, _short(value))
            return value
        if outcome in ("unsupported", "missing"):
            self.report.add(UNSUPPORTED, name, value)
        else:
            self.report.add(FAIL, name, f"crashed instead of returning or raising 'not implemented': {value}")
        return None


def _short(value: Any, limit: int = 80) -> str:
    try:
        if hasattr(value, "tolist"):
            value = value.tolist()
        text = json.dumps(value, default=str)
    except Exception:
        text = repr(value)
    return text if len(text) <= limit else text[:limit] + "…"


def _check_stage(value) -> str | None:
    try:
        v = [float(x) for x in value]
    except Exception:
        return f"expected a list of numbers, got {value!r}"
    if len(v) != 5:
        return f"expected [x, y, z, alpha, beta] (5 values), got {len(v)}"
    if any(not math.isfinite(x) for x in v):
        return "non-finite stage value"
    if any(abs(x) > 1e-2 for x in v[:3]):
        return f"x/y/z {v[:3]} exceed 1 cm: positions must be in metres (is this nm or µm?)"
    if any(abs(x) > 90 for x in v[3:]):
        return f"tilts {v[3:]} exceed 90: tilts must be in degrees"
    return None


def _check_fov(value) -> str | None:
    try:
        f = float(value)
    except Exception:
        return f"expected a number, got {value!r}"
    if not (0 < f < 1e-2):
        return f"field of view {f!r} should be positive metres below 1 cm"
    return None


def _check_metres(value) -> str | None:
    try:
        f = float(value)
    except Exception:
        return f"expected a number, got {value!r}"
    return None if abs(f) < 1e-3 else f"{f!r} m looks too large: expected metres"


def _check_parameters(value) -> str | None:
    try:
        data = json.loads(value)
    except Exception:
        return "get_parameters must return a JSON string"
    if not isinstance(data, dict):
        return "get_parameters must return a JSON object"
    return None


def _check_pair(value) -> str | None:
    try:
        v = [float(x) for x in value]
    except Exception:
        return f"expected two numbers, got {value!r}"
    return None if len(v) == 2 else f"expected 2 values, got {len(v)}"


def run(proxy, device: str, *, writes: bool = False, acquire: bool = False) -> Report:
    c = Checker(proxy, device)
    rep = c.report
    try:
        rep.dev_class = proxy.info().dev_class
    except Exception as exc:
        rep.add(FAIL, "device info", _err_text(exc))
        return rep
    try:
        state = str(proxy.state())
        rep.add(PASS if state in ("ON", "STANDBY", "RUNNING") else WARN, "state", state)
    except Exception as exc:
        rep.add(FAIL, "state", _err_text(exc))

    stage = c.read("get_stage", _check_stage)
    fov = c.read("get_fov", _check_fov)
    defocus = c.read("get_defocus", _check_metres)
    c.read("get_parameters", _check_parameters)
    c.read("get_image_shift", _check_pair)
    c.read("get_beam_tilt", _check_pair)
    c.read("get_diffraction_shift", _check_pair)

    if writes:
        if fov is not None:
            _roundtrip(c, "set_fov", "get_fov", float(fov), rel=1e-6)
        if defocus is not None:
            _roundtrip(c, "set_defocus", "get_defocus", float(defocus), abs_tol=1e-10)
        if stage is not None:
            outcome, value = c.call("move_stage", [float(v) for v in stage])
            if outcome == "ok":
                after = proxy.command_inout("get_stage")
                moved = max(abs(float(a) - float(b)) for a, b in zip(stage[:3], after[:3]))
                rep.add(PASS if moved < 1e-7 else WARN, "move_stage (zero-length)",
                        f"moved {moved * 1e9:.1f} nm" if moved >= 1e-7 else "stayed put")
            else:
                rep.add(UNSUPPORTED if outcome != "error" else FAIL, "move_stage (zero-length)", value)
        for cmd in ("blank_beam", "unblank_beam"):
            outcome, value = c.call(cmd)
            rep.add(PASS if outcome == "ok" else UNSUPPORTED if outcome != "error" else FAIL, cmd,
                    "" if outcome == "ok" else value)
    else:
        rep.add(SKIP, "writes", "rerun with --writes for set/get round trips and a zero-length stage move")

    if acquire:
        outcome, value = c.call("acquire_scanned_image", ["haadf"])
        if outcome == "ok":
            ok = isinstance(value, str) and value.strip() != ""
            rep.add(PASS if ok else FAIL, "acquire_scanned_image",
                    f"returned {value!r}" + ("" if ok else ": expected a DATA/Tiled key or file path"))
        else:
            rep.add(UNSUPPORTED if outcome != "error" else FAIL, "acquire_scanned_image", value)
    else:
        rep.add(SKIP, "acquire", "rerun with --acquire for one acquisition (uses the beam)")
        for cmd in ("acquire_scanned_image", "acquire_spectrum", "auto_focus", "place_beam", "set_fov",
                    "set_defocus", "move_stage", "blank_beam", "unblank_beam", "set_beam_tilt", "set_image_shift"):
            if cmd not in rep.supported and cmd in c.commands:
                rep.supported.setdefault(cmd, None)      # present, not exercised

    for feature, needs in FEATURES.items():
        missing = [n for n in needs if rep.supported.get(n) is False or (c.commands and n not in c.commands)]
        untested = [n for n in needs if rep.supported.get(n) is None and n not in missing]
        status = "no" if missing else ("untested" if untested else "yes")
        rep.features[feature] = {"status": status, "missing": missing, "untested": untested}
    return rep


def _roundtrip(c: Checker, setter: str, getter: str, value: float, rel: float = 0.0, abs_tol: float = 0.0) -> None:
    outcome, err = c.call(setter, value)
    if outcome != "ok":
        c.report.add(UNSUPPORTED if outcome != "error" else FAIL, f"{setter} round trip", err)
        return
    outcome, back = c.call(getter)
    if outcome != "ok":
        c.report.add(FAIL, f"{setter} round trip", f"{getter} failed after {setter}: {back}")
        return
    if math.isclose(float(back), value, rel_tol=rel, abs_tol=abs_tol):
        c.report.add(PASS, f"{setter} round trip", f"{value:g}")
    else:
        c.report.add(FAIL, f"{setter} round trip", f"wrote {value:g}, read back {float(back):g}")


def format_report(rep: Report) -> str:
    lines = [f"Conformance: {rep.device} ({rep.dev_class})", ""]
    for r in rep.results:
        lines.append(f"  {r.level:11s} {r.name:28s} {r.detail}")
    lines += ["", "Feature readiness:"]
    for feature, info in rep.features.items():
        mark = {"yes": "✓", "no": "✗", "untested": "?"}[info["status"]]
        extra = (f"  missing: {', '.join(info['missing'])}" if info["missing"] else
                 f"  not exercised: {', '.join(info['untested'])} (use --writes/--acquire)" if info["untested"] else "")
        lines.append(f"  {mark} {feature}{extra}")
    lines += ["", "RESULT: " + ("conformant" if rep.ok else "NOT conformant (fix the FAIL lines)")]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("device", nargs="?", default="asyncroscopy/instrument/default")
    ap.add_argument("--writes", action="store_true")
    ap.add_argument("--acquire", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--tango-host", default=None, help="host:port; defaults to $TANGO_HOST")
    a = ap.parse_args(argv)
    if a.tango_host:
        os.environ["TANGO_HOST"] = a.tango_host
    import tango

    proxy = tango.DeviceProxy(a.device)
    proxy.set_timeout_millis(60_000)
    rep = run(proxy, a.device, writes=a.writes, acquire=a.acquire)
    print(json.dumps(rep.to_json(), indent=2, default=str) if a.json else format_report(rep))
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
