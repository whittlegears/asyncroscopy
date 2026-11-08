"""The simulated instrument: settings, commands, and the signals of its acquisitions."""

from __future__ import annotations

import ast
import json
import math
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from .schema import InstrumentSpec, MAX_DURATION_S, MAX_POINTS, SAMPLE_CELLS_MAX, SAMPLE_CELL_M, duration_seconds, parse_arguments
from .expressions import Refused, evaluate

FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))


class SimInstrument:
    """The instrument's state: setting values, and running commands and acquisitions against them."""

    def __init__(self, spec: InstrumentSpec, sleep: Callable[[float], None] = time.sleep) -> None:
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

    def _wait(self, name: str, duration, scope, limit: float | None = MAX_DURATION_S, sleep=None) -> None:
        seconds = duration_seconds(duration, scope)
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

    def run_command(self, name: str, argument: Any = None, sleep: Callable[[float], None] | None = None) -> str:
        """Run a command; ``sleep`` replaces its simulated wait, so a job step can stop it early."""
        command = self.spec.commands[name]
        with self.lock:
            scope = {**self.values, **parse_arguments(name, command.inputs, argument)}
            self._check(name, command.requires, scope)
            updates = {target: self.spec.settings[target].coerce(evaluate(tree, scope) if tree is not None else value) for target, (value, tree) in command.sets.items()}
            after = {**self.values, **updates}
            for target in updates:  # a command that changes a guarded setting answers to its requires too
                self._check(name, self.spec.setting_requires.get(target, []), {**scope, **after})
        self._wait(name, command.duration, scope, sleep=sleep)
        with self.lock:
            self.values.update(updates)
        result = self._value(command.returns, command.returns_trees, {**scope, **updates})
        if result is None:
            return ""
        return result if isinstance(result, str) else json.dumps(result)

    def acquire(self, name: str, argument: Any = None, limit: float | None = MAX_DURATION_S, sleep: Callable[[float], None] | None = None) -> tuple[str, str, np.ndarray, dict]:
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
            if "x_m" in spots:
                pixel = num("pixel_size_m", raw["pixel_size_m"])
                radius_m = num("spots.radius_m", spots["radius_m"]) if "radius_m" in spots else max(num("spots.radius_px", spots.get("radius_px", 3)), 0.5) * pixel
                image += sample_spots(
                    self.spec.seed, rows, cols, pixel, num("spots.x_m", spots["x_m"]), num("spots.y_m", spots["y_m"]),
                    num("spots.per_um2", spots["per_um2"]), radius_m, num("spots.height", spots.get("height", 100)),
                )
            elif spots:
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


def sample_spots(seed: int, rows: int, cols: int, pixel: float, cx: float, cy: float, per_um2: float, radius_m: float, height: float) -> np.ndarray:
    """Gaussian spots fixed on the sample, seen by a ``rows`` x ``cols`` image of ``pixel`` metres centred on (cx, cy).

    The sample is cut into SAMPLE_CELL_M squares, each with its own seed, so every image of the same
    place shows the same spots whatever its size or position. Images use the stitcher's convention:
    rows are +y, columns +x, and pixel (r, c) sits at cx + (c - (cols - 1) / 2) * pixel.
    """
    image = np.zeros((rows, cols))
    if per_um2 <= 0 or pixel <= 0:
        return image
    radius = max(float(radius_m), 1e-12)
    reach = 4.0 * radius
    xs = cx + (np.arange(cols) - (cols - 1) / 2.0) * pixel
    ys = cy + (np.arange(rows) - (rows - 1) / 2.0) * pixel
    i0, i1 = math.floor((xs[0] - reach) / SAMPLE_CELL_M), math.floor((xs[-1] + reach) / SAMPLE_CELL_M)
    j0, j1 = math.floor((ys[0] - reach) / SAMPLE_CELL_M), math.floor((ys[-1] + reach) / SAMPLE_CELL_M)
    if (i1 - i0 + 1) * (j1 - j0 + 1) > SAMPLE_CELLS_MAX:
        raise Refused(f"an image {cols * pixel:.3g} m wide is too large for spots fixed on the sample (at most {SAMPLE_CELLS_MAX} cells of {SAMPLE_CELL_M:g} m)")
    expected = float(per_um2) * (SAMPLE_CELL_M / 1e-6) ** 2
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, (i + 2**31) & 0xFFFFFFFF, (j + 2**31) & 0xFFFFFFFF])
            for sx, sy in rng.uniform(0.0, SAMPLE_CELL_M, size=(rng.poisson(expected), 2)) + (i * SAMPLE_CELL_M, j * SAMPLE_CELL_M):
                c0, c1 = np.searchsorted(xs, sx - reach), np.searchsorted(xs, sx + reach)
                r0, r1 = np.searchsorted(ys, sy - reach), np.searchsorted(ys, sy + reach)
                if c0 < c1 and r0 < r1:
                    dx2 = (xs[c0:c1] - sx) ** 2
                    dy2 = (ys[r0:r1] - sy) ** 2
                    image[r0:r1, c0:c1] += height * np.exp(-(dy2[:, None] + dx2[None, :]) / (2.0 * radius**2))
    return image


def peak_profile(x: np.ndarray, position: float, fwhm: float, eta: float) -> np.ndarray:
    """A peak of height 1 at ``position``: Gaussian (eta 0), Lorentzian (eta 1) or the
    pseudo-Voigt mix of the two (0 < eta < 1), all with the same full width at half maximum."""
    offset = (x - position) / fwhm
    gaussian = np.exp(-4.0 * math.log(2.0) * offset**2)
    lorentzian = 1.0 / (1.0 + 4.0 * offset**2)
    return (1.0 - eta) * gaussian + eta * lorentzian
