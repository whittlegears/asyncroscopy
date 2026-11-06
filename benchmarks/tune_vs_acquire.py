"""How long to tune before acquiring? A bench on the twin, with no LLM and no live stack (principle 4).

Each trial has a fixed instrument-time budget T. The column starts with a known hidden defocus
and two-fold astigmatism, and optionally a focus that drifts. The trial first runs the Tuner
(the real job device, strategy image_metric, focus and stigmation) with a search budget from
none to large, on a sacrificial field next to the map; then it spends whatever time is left
acquiring a fixed map, tile after tile, until T is used up. The evaluators of
``asyncroscopy.evaluate`` then score the trial from its Instrument Trace and data:
``information`` (unique ground-truth particles found), ``resolution`` (the focus error behind
every frame), ``dose`` and ``time``. The curve is information against tuning budget; its
optimum moves with the starting error and with drift (with fast drift, tuning goes stale).

    uv run python benchmarks/tune_vs_acquire.py                 # the full study (a few minutes)
    uv run python benchmarks/tune_vs_acquire.py --tiny          # a check that it runs (~30 s)
    uv run python benchmarks/tune_vs_acquire.py --budget-s 60 --conditions large_error_fast_drift

Everything runs in this process: a Tango test context with a SCAN device, the twin and a Tuner,
on free ports. Results: benchmarks/results/tune_vs_acquire.json and .md (or --out).

What is modelled here and not in the twin itself (so it is stated, not hidden):

* focus drift: the twin has stage drift but no drift of the focus, so ``FocusDriftTwin`` below
  adds ``rate x (clock - t0)`` to C1;
* per-frame overhead: the twin's clock counts frame and stage-move time only, so each frame also
  costs ``--frame-overhead-s`` (the Tuner's own estimate assumes 0.5 s per tuning frame);
* the ground truth (particle positions, the hidden focus error) is the twin's own model, so
  "information" is what this twin lets a detector find, not what a specimen would give.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np
import tango

from asyncroscopy.evaluate import (
    RunData,
    get_evaluator,
    keep_tuning,
    score_run,
    stale_after_s,
)
from asyncroscopy.instruments.electron_microscope.digital_twin import DigitalTwin
from asyncroscopy.instruments.electron_microscope.hardware.scan import SCAN
from asyncroscopy.jobs.tuning.tuner import Tuner
from asyncroscopy.mcp.tool_log import TraceWriter
from asyncroscopy.trace.stream import read_trace

if TYPE_CHECKING:
    from tango.test_context import MultiDeviceTestContext

# ---------------------------------------------------------------------- the study
# Tuning budgets: the Tuner's search sizes (image_metric). "none" skips tuning.
BUDGETS = {
    "none": None,
    "focus_only": {"corrections": ["focus"], "coarse_points": 5, "fine_points": 3, "rounds": 1},
    "xs": {"corrections": ["focus", "stigmation"], "coarse_points": 5, "fine_points": 3, "rounds": 1, "stig_passes": 1},
    "s": {"corrections": ["focus", "stigmation"], "coarse_points": 7, "fine_points": 5, "rounds": 1, "stig_passes": 1},
    "m": {"corrections": ["focus", "stigmation"], "coarse_points": 9, "fine_points": 7, "rounds": 1, "stig_passes": 1},
    "l": {"corrections": ["focus", "stigmation"], "coarse_points": 13, "fine_points": 9, "rounds": 2, "stig_passes": 1},
}
# Starting column: hidden C1 and A1 (m), and focus drift (m/s).
CONDITIONS = {
    "small_error": {"c1_m": 30e-9, "a1_m": [10e-9, 0.0], "drift_m_per_s": 0.0},
    "large_error": {"c1_m": 250e-9, "a1_m": [40e-9, -30e-9], "drift_m_per_s": 0.0},
    "large_error_slow_drift": {"c1_m": 250e-9, "a1_m": [40e-9, -30e-9], "drift_m_per_s": 0.3e-9},
    "large_error_fast_drift": {"c1_m": 250e-9, "a1_m": [40e-9, -30e-9], "drift_m_per_s": 2e-9},
}
TUNE_PLAN = {
    "strategy": "image_metric",
    "dose_budget_e_per_a2": 1e7,  # the bench measures dose; it does not limit it
    "max_steps": 1000,
    "focus_range_m": 4e-7,
    "stig_range_m": 1.5e-7,
    "fov_m": 1e-7,
    "imsize": 128,
    "dwell_s": 1e-6,
}
ACQUIRE = {"fov_m": 1e-7, "imsize": 256, "dwell_s": 4e-6}  # one map tile
MAP_GRID = 9  # tiles per side (100 nm apart, no overlap)
FOCUS_FIELD_M = (0.52e-6, 0.0)  # sacrificial field off the map, where the tuning is done
TWIN = {
    "sample_seed": 77,
    "sample_size_xy": 1.2e-6,
    "sample_particle_count": 2900,
    "support_thickness": 3e-9,
    "stage_move_overhead": 1.0,  # move and settle (s)
    "dose_tracking": False,
    "time_scale": 0.0,
}
DEPTH_OF_FIELD_M = 20e-9  # the resolution evaluator's in-focus tolerance; also the policy's staleness bound
# Two objectives that disagree: particles found (blur is tolerated), and particles found in frames taken in focus.
OBJECTIVES = ("information", "information_in_focus")


class FocusDriftTwin(DigitalTwin):
    """The particle twin plus two things the bench needs: a focus that drifts, and a fixed overhead per frame."""

    def init_device(self) -> None:
        super().init_device()
        self._focus_drift = 0.0
        self._drift_t0 = 0.0
        self._frame_overhead_s = 0.0
        self.last_frame_focus: dict | None = None

    def set_focus_drift(self, rate_m_per_s: float) -> None:
        """Bench hook, not a Tango command: C1 grows by ``rate`` per simulated second from now on."""
        self._focus_drift, self._drift_t0 = float(rate_m_per_s), self._clock.now()

    def _aberrations(self) -> dict:
        ab = super()._aberrations()
        drift = getattr(self, "_focus_drift", 0.0)
        if drift:
            ab["C10"] = float(ab.get("C10", 0.0)) + drift * (self._clock.now() - self._drift_t0) * 1e9
        return ab

    def _scan_frame(self, imsize: int, dwell_time: float, scan_region: Sequence[float]) -> dict:
        self._clock.advance(getattr(self, "_frame_overhead_s", 0.0))  # set-up, settle and read-out of a frame
        self.last_frame_focus = self._focus_truth()  # the hidden focus error this frame is taken with
        return super()._scan_frame(imsize, dwell_time, scan_region)


def _prop_path(path: Path) -> str:
    from asyncroscopy.testing import prop_path

    return prop_path(path)


def devices_info(work: Path) -> list[dict]:
    return [
        {"class": SCAN, "devices": [{"name": "bench/scan/tva", "properties": {}}]},
        {"class": FocusDriftTwin, "devices": [{"name": "bench/twin/tva", "properties": {
            **TWIN, "scan_device_address": "bench/scan/tva", "acquisition_save_directory": _prop_path(work / "data")}}]},
        {"class": Tuner, "devices": [{"name": "bench/tuner/tva", "properties": {
            "microscope_device_address": "bench/twin/tva", "scan_device_address": "bench/scan/tva",
            "state_directory": _prop_path(work / "tuner_state"), "trace_path": _prop_path(work / "trace" / "jobs.jsonl"), "simulated": True}}]},
    ]


def _call(proxy: tango.DeviceProxy, verb: str, payload: dict) -> dict:
    return json.loads(getattr(proxy, verb)(json.dumps(payload)))


def map_tiles() -> list[tuple[float, float]]:
    """A serpentine grid of tile centres around the origin."""
    step = ACQUIRE["fov_m"]
    offsets = [(i - (MAP_GRID - 1) / 2) * step for i in range(MAP_GRID)]
    tiles = []
    for row, y in enumerate(offsets):
        for x in offsets if row % 2 == 0 else reversed(offsets):
            tiles.append((x, y))
    return tiles


class Bench:
    def __init__(self, ctx: MultiDeviceTestContext, work: Path, frame_overhead_s: float) -> None:
        self.work = work
        self.twin = tango.DeviceProxy(ctx.get_device_access("bench/twin/tva"))
        self.scan = tango.DeviceProxy(ctx.get_device_access("bench/scan/tva"))
        self.tuner = tango.DeviceProxy(ctx.get_device_access("bench/tuner/tva"))
        self.tuner.set_timeout_millis(120_000)
        self.device = tango.Util.instance().get_device_by_name("bench/twin/tva")
        self.device._frame_overhead_s = float(frame_overhead_s)
        self.trace_path = work / "trace" / "bench.jsonl"

    def clock(self) -> float:
        return float(self.device.instrument_clock_s())

    def reset(self, condition: dict) -> None:
        """A fresh column: no stigmator or defocus set, the condition's hidden error, its drift from now."""
        self.twin.blank_beam()
        self.twin.set_defocus(0.0)
        self.twin.set_stigmator([0.0, 0.0])
        self.device.set_focus_error(condition["c1_m"], condition["a1_m"])
        self.device.set_focus_drift(condition["drift_m_per_s"])

    def tune(self, budget: dict, trial: str) -> dict:
        plan = {**TUNE_PLAN, **budget}
        reserved = _call(self.tuner, "reserve", {"plan": plan, "run": trial, "caller": "bench"})
        if "error" in reserved:
            raise RuntimeError(f"Tuner refused the plan: {reserved['error']}")
        started = _call(self.tuner, "start", {"reservation_id": reserved["reservation_id"], "run": trial})
        while True:
            status = _call(self.tuner, "job_status", {"job_id": started["job_id"]})
            if status["state"] in ("done", "failed", "aborted"):
                break
            time.sleep(0.02)
        results = _call(self.tuner, "results", {"job_id": started["job_id"]})
        _call(self.tuner, "release", {"reservation_id": reserved["reservation_id"]})
        tune = (results.get("summary") or {}).get("tune") or {}
        return {"state": status["state"], "outcome": tune.get("outcome"), "reason": tune.get("reason"), "images": tune.get("images", 0),
                "dose_e_per_a2": tune.get("dose_e_per_a2", 0.0)}

    def trial(self, condition_name: str, budget_name: str, budget_s: float, index: int) -> dict:
        condition = CONDITIONS[condition_name]
        trial = f"tva-{condition_name}-{budget_name}-{index}-{int(time.time() * 1000) % 10**8}"
        writer = TraceWriter(self.trace_path, "script", writer="bench:tune_vs_acquire", defaults={"run": trial, "simulated": True}, clock=self.clock)
        self.reset(condition)
        self.twin.move_stage([*FOCUS_FIELD_M, 0.0, 0.0, 0.0])
        t0 = self.clock()
        writer.record("bench.start", True, time.time(), arguments={"condition": condition_name, "budget": budget_name, "budget_s": budget_s})
        tuned = {"state": "skipped", "images": 0, "dose_e_per_a2": 0.0}
        if BUDGETS[budget_name] is not None:
            tuned = self.tune(BUDGETS[budget_name], trial)
        tune_s = self.clock() - t0
        after_tune = self.device._focus_truth()
        # Acquire the map, tile after tile, while a whole tile still fits in the budget.
        self.twin.set_fov(ACQUIRE["fov_m"])
        self.scan.imsize = ACQUIRE["imsize"]
        self.scan.dwell_time = ACQUIRE["dwell_s"]
        self.scan.scan_region = [0.0, 0.0, 1.0, 1.0]
        frame_s = ACQUIRE["imsize"] ** 2 * ACQUIRE["dwell_s"] + self.device._frame_overhead_s
        focus_by_acquisition: dict[str, dict] = {}
        keys: list[str] = []
        for x, y in map_tiles():
            move_s = TWIN["stage_move_overhead"] + math.hypot(x - self.device._stage.commanded[0], y - self.device._stage.commanded[1]) / 5e-6
            if self.clock() - t0 + move_s + frame_s > budget_s:
                break
            self.twin.move_stage([x, y, 0.0, 0.0, 0.0])
            started = time.time()
            self.twin.unblank_beam()
            try:
                key = self.twin.acquire_scanned_image(["haadf"])
            finally:
                self.twin.blank_beam()
            truth = self.device.last_frame_focus or {}
            index_ = self.device._acq_index
            focus_by_acquisition[str(index_)] = {"c1_m": truth.get("c1_m"), "a1_abs_m": truth.get("a1_abs_m")}
            writer.record("bench.acquire", True, started, duration_s=time.time() - started, result_key=key, arguments={"xy_m": [x, y]})
            keys.append(key)
        writer.record("bench.end", True, time.time())
        events = read_trace([self.trace_path, self.work / "trace" / "jobs.jsonl"], run=trial)
        truth = json.loads(self.twin.export_ground_truth())
        truth["focus_by_acquisition"] = focus_by_acquisition
        data = RunData(trial, events, truth=truth)
        # Information per T: the whole budget, tuning included, is the time it cost.
        scores = score_run(data, [get_evaluator("information", duration_s=budget_s), "resolution", "dose", "time"])
        for key in keys:  # the frames are scored; keep the disk small
            try:
                os.remove(key)
            except OSError:
                pass
        value = lambda ev, obj: scores.get(ev, {}).get(obj, {}).get("value")
        return {
            "condition": condition_name,
            "budget": budget_name,
            "index": index,
            "run": trial,
            "budget_s": budget_s,
            "tune_s": round(tune_s, 3),
            "tune_state": tuned["state"],
            "tune_outcome": tuned.get("outcome"),
            "tune_reason": tuned.get("reason"),
            "tune_images": tuned["images"],
            "tune_dose_e_per_a2": round(float(tuned["dose_e_per_a2"]), 2),
            "residual_after_tune_m": math.hypot(after_tune["c1_m"], after_tune["a1_abs_m"]),
            "tiles": len(keys),
            "acquire_s": round(budget_s - tune_s, 3),
            "information": value("information", "features_found"),
            "information_in_focus": value("information", "features_in_focus"),
            "features_per_s": value("information", "features_per_s"),
            "recall": value("information", "recall"),
            "median_focus_error_m": value("resolution", "focus_error_m"),
            "in_focus_fraction": value("resolution", "in_focus_fraction"),
            "max_dose_e_per_a2": value("dose", "max_dose_e_per_a2"),
            "events": len(events),
            "scores": scores,
        }


def summarise(rows: list[dict], budget_s: float) -> dict:
    """Per condition: the mean curve over repeats, its optimum, and what the stopping rule would have said."""
    out = {}
    for condition in dict.fromkeys(r["condition"] for r in rows):
        curve = []
        for budget in dict.fromkeys(r["budget"] for r in rows if r["condition"] == condition):
            mine = [r for r in rows if r["condition"] == condition and r["budget"] == budget]

            def mean(key: str, mine: list[dict] = mine) -> float | None:
                values = [r[key] for r in mine if r[key] is not None]
                return float(np.mean(values)) if values else None

            curve.append({
                "budget": budget,
                "tune_s": mean("tune_s"),
                "tune_images": mean("tune_images"),
                "tiles": mean("tiles"),
                "information": mean("information"),
                "information_in_focus": mean("information_in_focus"),
                "features_per_s": mean("features_per_s"),
                "residual_after_tune_m": mean("residual_after_tune_m"),
                "median_focus_error_m": mean("median_focus_error_m"),
                "in_focus_fraction": mean("in_focus_fraction"),
                "tune_failed": sum(r["tune_state"] == "failed" for r in mine),
                "repeats": len(mine),
            })
        optima = {}
        for objective in OBJECTIVES:
            best = max(curve, key=lambda p: (p[objective] or 0.0, -(p["tune_s"] or 0.0)))
            optima[objective] = {"budget": best["budget"], "value": best[objective], "tune_s": best["tune_s"]}
        # The stopping rule (asyncroscopy.evaluate.policy.keep_tuning), fed the rates this bench measured:
        # would it have told an agent to tune with each budget instead of acquiring at once, and was that right?
        # The rule's rates come from these same trials, so this checks the rule's arithmetic and its staleness
        # term, not its ability to predict.
        base = next((p for p in curve if p["budget"] == "none"), None)
        drift = CONDITIONS[condition]["drift_m_per_s"]
        for point in curve:
            if base is None or point["budget"] == "none" or not point["tune_s"]:
                continue
            for objective in OBJECTIVES:
                rate_now = (base[objective] or 0.0) / budget_s
                rate_tuned = (point[objective] or 0.0) / max(1e-9, budget_s - point["tune_s"])
                stale = stale_after_s(DEPTH_OF_FIELD_M, drift, point["residual_after_tune_m"] or 0.0)
                # Unless the objective needs focus, a stale focus still finds particles: staleness applies to in-focus information only.
                window = stale if objective == "information_in_focus" else math.inf
                decision = keep_tuning(rate_now, rate_tuned, point["tune_s"], budget_s, window)
                point.setdefault("rule_says_tune", {})[objective] = decision.keep_tuning
                point.setdefault("tuning_paid_off", {})[objective] = (point[objective] or 0.0) > (base[objective] or 0.0)
        out[condition] = {"curve": curve, "optimum": optima, **CONDITIONS[condition]}
    return out


def markdown(results: dict) -> str:
    lines = [
        f"# Tune vs acquire: {results['git']}, T = {results['budget_s']:g} s of instrument time",
        "",
        f"Twin: {results['twin']['sample_particle_count']} particles on {results['twin']['sample_size_xy'] * 1e6:g} um, "
        f"map tiles {ACQUIRE['imsize']} px at {ACQUIRE['fov_m'] * 1e9:g} nm, dwell {ACQUIRE['dwell_s'] * 1e6:g} us; "
        f"{results['frame_overhead_s']:g} s overhead per frame, {TWIN['stage_move_overhead']:g} s per move. "
        f"Information: unique ground-truth particles found (twin only). {results['repeats']} repeat(s) per point.",
        "",
    ]
    for condition, summary in results["summary"].items():
        lines += [
            f"## {condition}: C1 {summary['c1_m'] * 1e9:g} nm, |A1| {math.hypot(*summary['a1_m']) * 1e9:.0f} nm, drift {summary['drift_m_per_s'] * 1e9:g} nm/s",
            "",
            "| budget | tune images | tune s | tiles | found | found in focus | residual after tune nm | median focus error nm | in-focus frames | rule: tune? (found / in focus) | paid off? |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        yn = lambda d: "-" if not d else " / ".join("yes" if d[o] else "no" for o in OBJECTIVES)
        nm = lambda v: "-" if v is None else f"{v * 1e9:.1f}"
        num = lambda v, f=".1f": "-" if v is None else format(v, f)
        for p in summary["curve"]:
            failed = f" ({p['tune_failed']} failed)" if p["tune_failed"] else ""
            lines.append(
                f"| {p['budget']}{failed} | {num(p['tune_images'], '.0f')} | {num(p['tune_s'])} | {num(p['tiles'], '.0f')} | {num(p['information'])} | "
                f"{num(p['information_in_focus'])} | {nm(p['residual_after_tune_m'])} | {nm(p['median_focus_error_m'])} | {num(p['in_focus_fraction'], '.2f')} | "
                f"{yn(p.get('rule_says_tune'))} | {yn(p.get('tuning_paid_off'))} |"
            )
        lines.append("")
        for objective, best in summary["optimum"].items():
            lines.append(f"Optimum for {objective}: **{best['budget']}** ({num(best['value'])} features, {num(best['tune_s'])} s tuning).")
        lines.append("")
    return "\n".join(lines) + "\n"


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=HERE.parent, check=False).stdout.strip() or "?"
    except OSError:
        return "?"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--budget-s", type=float, default=120.0, help="Instrument time per trial, tuning plus acquisition (s).")
    parser.add_argument("--conditions", nargs="*", default=list(CONDITIONS), choices=list(CONDITIONS))
    parser.add_argument("--budgets", nargs="*", default=list(BUDGETS), choices=list(BUDGETS))
    parser.add_argument("--repeats", type=int, default=2, help="Trials per (condition, budget); the twin's noise differs between them.")
    parser.add_argument("--frame-overhead-s", type=float, default=0.5, help="Set-up and read-out time charged per frame, tuning and map alike.")
    parser.add_argument("--tiny", action="store_true", help="A quick check: T = 25 s, budgets none and focus_only, one condition, one repeat.")
    parser.add_argument("--out", default=None, help="Results JSON (a .md is written next to it). Default benchmarks/results/tune_vs_acquire.json.")
    parser.add_argument("--work", default=None, help="Scratch directory for data and traces (default: a temporary one, removed after).")
    args = parser.parse_args(argv)
    if args.tiny:
        args.budget_s, args.budgets, args.conditions, args.repeats = 25.0, ["none", "focus_only"], ["large_error"], 1

    from asyncroscopy.testing import instrument_context

    work = Path(args.work) if args.work else Path(tempfile.mkdtemp(prefix="tva_", dir=os.getcwd()))
    keep = bool(args.work)
    work.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else HERE / "results" / "tune_vs_acquire.json"
    started = time.time()
    rows: list[dict] = []
    try:
        with instrument_context(devices_info(work)) as ctx:
            bench = Bench(ctx, work, args.frame_overhead_s)
            for condition in args.conditions:
                for budget in args.budgets:
                    for index in range(args.repeats):
                        row = bench.trial(condition, budget, args.budget_s, index)
                        rows.append(row)
                        print(
                            f"{condition:24s} {budget:10s} #{index} tune {row['tune_s']:6.1f} s ({row['tune_images']:3d} img, {row['tune_state']}) "
                            f"tiles {row['tiles']:3d} info {row['information']} residual {row['residual_after_tune_m'] * 1e9:6.1f} nm",
                            flush=True,
                        )
    finally:
        if not keep:
            shutil.rmtree(work, ignore_errors=True)
    results = {
        "git": git_head(),
        "budget_s": args.budget_s,
        "repeats": args.repeats,
        "frame_overhead_s": args.frame_overhead_s,
        "twin": TWIN,
        "tune_plan": TUNE_PLAN,
        "acquire": ACQUIRE,
        "budgets": {k: BUDGETS[k] for k in args.budgets},
        "seconds": round(time.time() - started, 1),
        "summary": summarise(rows, args.budget_s),
        "rows": [{k: v for k, v in r.items() if k != "scores"} | {"scores": r["scores"]} for r in rows],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    out.with_suffix(".md").write_text(markdown(results), encoding="utf-8")
    print(markdown(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
