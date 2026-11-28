"""The instrument profile: what is known about one microscope, as numbers the digital twin can use.

The twin is three things kept apart. Its *physics* is the code (how a stage misses, how a detector counts), the same
for every microscope. The *specimen* is a plug-in. The *instrument profile* is this file: one YAML per microscope
holding each twin parameter's value, how well it is known, and where it came from. Fitting the twin to an
instrument only ever changes its profile, so the twin never becomes one laboratory's microscope.

A profile looks like this (``configs/profiles/spectra300.yaml`` is a full one)::

    name: Spectra 300
    kind: probe-corrected S/TEM, 300 kV
    parameters:
      stage_move_noise_std:
        device: stage              # the twin device the property belongs to: instrument or stage
        value: 1.5e-7              # the best estimate
        sigma: 5.0e-8              # how well it is known (1 sigma); leave out when unknown
        range: [2.0e-8, 5.0e-7]    # what microscopes of this kind span; leave out when unknown
        unit: m
        source: stage_repeatability   # a calibration script, a paper, a datasheet, or "placeholder"
        measured: 2026-11-21          # the day it was measured on this instrument; leave out when it was not

Three ways to use one:

* the twin of this instrument: every parameter at its ``value``;
* a twin that might be this instrument (``--draw SEED``): every parameter drawn from ``value`` +/- ``sigma``, so a
  rehearsal or a planner sees how much the instrument's uncertainty matters;
* a twin of some instrument of this kind (``--draw SEED --across``): every parameter drawn from its ``range``, for
  training a model that must not depend on one microscope.

Turn a profile into a stack config that starts from an existing one (its ``extends:``)::

    uv run python -m asyncroscopy.calibration.profile configs/profiles/spectra300.yaml
    uv run python -m asyncroscopy.calibration.profile configs/profiles/spectra300.yaml --out configs/local/twin_spectra300.yaml
    uv run python -m asyncroscopy.calibration.profile configs/profiles/spectra300.yaml --draw 7 --across --out configs/local/twin_draw7.yaml

The calibration scripts write what they measure into a profile with ``--profile``, and the trace logger's summary
does the same (``asyncroscopy.calibration.trace_logger``).
"""

from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path

import numpy as np
import yaml

DEVICES = ("instrument", "stage")  # where a twin property lives in a stack config
HEADER = "# An instrument profile: see asyncroscopy/calibration/profile.py for the format and how to use it.\n"


def load(path: str | Path) -> dict:
    """Read a profile and check it, naming every problem at once."""
    profile = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    problems = []
    for name, entry in (profile.get("parameters") or {}).items():
        if not isinstance(entry, dict) or not isinstance(entry.get("value"), (int, float)):
            problems.append(f"{name}: needs a numeric value")
            continue
        if entry.get("device", "instrument") not in DEVICES:
            problems.append(f"{name}: device must be one of {DEVICES}")
        if "range" in entry and not (isinstance(entry["range"], list) and len(entry["range"]) == 2 and entry["range"][0] <= entry["range"][1]):
            problems.append(f"{name}: range must be [low, high]")
        if entry.get("sigma", 0) < 0:
            problems.append(f"{name}: sigma must be >= 0")
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    profile.setdefault("parameters", {})
    return profile


def save(path: str | Path, profile: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(HEADER + yaml.safe_dump(profile, sort_keys=False, allow_unicode=True), encoding="utf-8")


def record(path: str | Path, measured: dict[str, float], source: str, sigma: dict[str, float] | None = None, device: str = "instrument") -> dict:
    """Write measured values into the profile at ``path`` (made if missing) and return it.

    Each value replaces the parameter's old one and is stamped with ``source`` and today's date. A parameter the
    profile already has keeps its ``device``, ``unit`` and ``range``; a new one goes to ``device``.
    """
    path = Path(path)
    profile = load(path) if path.exists() else {"name": path.stem, "parameters": {}}
    today = datetime.date.today().isoformat()
    for name, value in measured.items():
        entry = profile["parameters"].setdefault(name, {"device": device})
        entry.update({"value": float(value), "source": source, "measured": today})
        if sigma and name in sigma:
            entry["sigma"] = float(sigma[name])
    save(path, profile)
    return profile


def unmeasured(profile: dict) -> list[str]:
    """The parameters nobody has measured on this instrument."""
    return [name for name, entry in profile["parameters"].items() if not entry.get("measured")]


def _draw(entry: dict, rng: np.random.Generator, across: bool) -> float:
    """One parameter's value: from its range (``across``) or from value +/- sigma, kept inside the range."""
    low, high = entry.get("range", (-np.inf, np.inf))
    if across and "range" in entry:
        # Ranges that span more than a decade are drawn evenly in the logarithm, so small values are as likely as large ones.
        if low > 0 and high / low > 10:
            return float(np.exp(rng.uniform(np.log(low), np.log(high))))
        return float(rng.uniform(low, high))
    return float(np.clip(rng.normal(entry["value"], entry.get("sigma", 0.0)), low, high))


def twin_properties(profile: dict, rng: np.random.Generator | None = None, across: bool = False) -> dict[str, dict[str, float]]:
    """The profile as twin properties, per device: ``{"instrument": {...}, "stage": {...}}``.

    Without ``rng`` every parameter is its ``value``. With ``rng`` each is drawn: from value +/- sigma, or, with
    ``across``, from its range (a parameter with no range falls back to value +/- sigma).
    """
    properties: dict[str, dict[str, float]] = {device: {} for device in DEVICES}
    for name, entry in profile["parameters"].items():
        properties[entry.get("device", "instrument")][name] = float(entry["value"]) if rng is None else _draw(entry, rng, across)
    return properties


def stack_config(profile: dict, base: str | Path, out: str | Path, rng: np.random.Generator | None = None, across: bool = False) -> dict:
    """A stack config that is ``base`` with this profile's properties, as a mapping ready to dump at ``out``."""
    properties = twin_properties(profile, rng, across)
    config: dict = {"extends": Path(os.path.relpath(Path(base).resolve(), Path(out).resolve().parent)).as_posix()}
    if properties["instrument"]:
        config["instrument"] = {"properties": properties["instrument"]}
    if properties["stage"]:
        config["devices"] = {"stage": {"properties": properties["stage"]}}
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Show an instrument profile, or write it as a twin stack config.")
    parser.add_argument("profile", type=Path)
    parser.add_argument("--base", type=Path, default=Path("configs/digital_twin_mapping.yaml"), help="The stack config to start from.")
    parser.add_argument("--out", type=Path, help="Write the stack config here; without it, print it.")
    parser.add_argument("--draw", type=int, help="Seed: draw each parameter from its uncertainty, not its value.")
    parser.add_argument("--across", action="store_true", help="With --draw: draw from each parameter's range across instruments.")
    args = parser.parse_args()
    profile = load(args.profile)
    rng = np.random.default_rng(args.draw) if args.draw is not None else None
    out = args.out or args.base.parent / "profile_preview.yaml"
    text = yaml.safe_dump(stack_config(profile, args.base, out, rng, args.across), sort_keys=False)
    missing = unmeasured(profile)
    note = f"# {profile.get('name', args.profile.stem)}: {len(profile['parameters']) - len(missing)} of {len(profile['parameters'])} parameters measured on the instrument"
    how = "" if rng is None else f" (drawn with seed {args.draw} {'across instruments' if args.across else 'from its uncertainty'})"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(f"{note}{how}\n{text}", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(f"{note}{how}\n{text}", end="")
    if missing:
        print(f"# not measured yet: {', '.join(missing)}")


if __name__ == "__main__":
    main()
