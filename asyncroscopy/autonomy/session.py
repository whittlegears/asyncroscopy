"""One instruction, a whole imaging session: health, region, focus, aberrations, image, record.

``run_session`` chains the routines into the sequence an operator would do
by hand, and returns one report a teammate can read later:

1. **health**: every device checked; faults get the automatic recovery rungs.
   A required device that stays down stops the session before anything moves.
2. **region** (optional): a low-magnification survey, the most structured
   regions ranked, the best one centred with the calibrated stage mapping.
3. **aberrations**: corrector measure-and-correct (this also sets C1, i.e.
   defocus, from a direct measurement), then optional image-based polishing
   of named aberrations.
4. **focus**: image-based trust-region polish on defocus (or stage z). It
   keeps a change only if a verified re-measurement shows a gain.
5. **image**: a final acquisition, scored against the first one.
6. **record**: the final image is saved as an atlas site with the session
   report as provenance, so the result can be revisited and verified later.

Every stage records its status and duration. A failed optional stage is
reported and skipped; a failed required stage ends the session in a known
state (each routine restores its start setting on failure).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from asyncroscopy.autonomy.metrics import all_metrics
from asyncroscopy.autonomy.optimize import Step, direct_step
from asyncroscopy.autonomy.routines import Imager, Resilient, find_regions, focus, measure_and_correct, tune_aberration
from asyncroscopy.health.monitor import HealthMonitor, recover


@dataclass
class SessionPlan:
    name: str = "autonomous session"
    find_region: bool = True
    survey_factor: float = 4.0
    focus_mode: str = "defocus"              # defocus | stage_z | none
    focus_step_m: float = 10e-9
    focus_max_offset_m: float | None = None
    correct: bool = True                     # corrector measure-and-correct
    thresholds_m: dict[str, float] = field(default_factory=lambda: {"C1": 2e-9, "A1": 2e-9, "B2": 30e-9,
                                                                    "A2": 30e-9})
    max_correction_m: dict[str, float] = field(default_factory=lambda: {"C1": 200e-9, "A1": 100e-9,
                                                                        "B2": 1e-6, "A2": 1e-6})
    polish: dict[str, float] = field(default_factory=dict)   # aberration -> step (m) for image-based tuning
    score: str = "edge_sharpness"
    save_site: bool = True
    required_devices: list[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, data: dict | None) -> "SessionPlan":
        data = dict(data or {})
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SessionContext:
    imager: Imager
    mic: Any = None                  # TangoMicroscope-like (get_stage/move_stage/acquire/current_fov/settings)
    corrector: Any = None
    monitor: HealthMonitor | None = None
    atlas: Any = None                # asyncroscopy.atlas.store.Atlas


def _stage(report: dict, name: str):
    entry = {"stage": name, "status": "running", "started": time.time()}
    report["stages"].append(entry)
    return entry


def _done(entry: dict, status: str, **data) -> None:
    entry.update(status=status, duration_s=round(time.time() - entry["started"], 3), **data)


def run_session(ctx: SessionContext, plan: SessionPlan | None = None, step: Step = direct_step,
                provenance: dict | None = None) -> dict:
    plan = plan or SessionPlan()
    t0 = time.time()
    report: dict[str, Any] = {"name": plan.name, "plan": asdict(plan), "stages": [], "status": "running"}
    resilient = ctx.imager.resilient

    # 1. health ---------------------------------------------------------------
    entry = _stage(report, "health")
    if ctx.monitor is not None:
        problems = []
        for h in ctx.monitor.check_all():
            if h.healthy:
                continue
            outcome = recover(ctx.monitor, h.device, allowed=resilient.allowed)
            problems.append({"device": h.device, "problem": h.problem, "recovery": outcome["status"]})
        still_down = [p for p in problems if p["recovery"] not in ("recovered", "healthy")]
        blocking = [p for p in still_down if not plan.required_devices or p["device"] in plan.required_devices]
        _done(entry, "failed" if blocking else "ok", problems=problems)
        if blocking:
            report.update(status="stopped", reason=f"required devices down: {[p['device'] for p in blocking]}",
                          duration_s=time.time() - t0)
            return report
    else:
        _done(entry, "skipped", reason="no health monitor")

    first = step("acquire", {"purpose": "initial image"}, ctx.imager.acquire)
    report["initial"] = {"key": first[0], "metrics": all_metrics(first[1])}

    # 2. region -----------------------------------------------------------------
    entry = _stage(report, "region")
    if plan.find_region and ctx.mic is not None:
        try:
            fov = ctx.mic.current_fov()
            size = ctx.mic.current_imsize() or first[1].shape[0]
            survey_fov = fov * plan.survey_factor
            survey = step("acquire_survey", {"fov_m": survey_fov, "imsize": size},
                          lambda: ctx.mic.acquire(survey_fov, size))
            regions = find_regions(survey, n=3)
            here = ctx.mic.get_stage()
            best = regions[0]
            mapping = ctx.atlas.mapping if ctx.atlas is not None else None
            px = survey_fov / survey.shape[1]
            if mapping is not None:
                x, y = mapping.pixel_to_stage(here[:2], best["col"], best["row"], survey.shape, px)
            else:      # identity axes; warn
                x = here[0] + (best["col"] - (survey.shape[1] - 1) / 2) * px
                y = here[1] + (best["row"] - (survey.shape[0] - 1) / 2) * px
            target = [x, y, *here[2:]]
            step("move_stage", {"position": target, "purpose": "centre region"}, lambda: ctx.mic.move_stage(target))
            step("set_fov", {"fov_m": fov}, lambda: ctx.imager.resilient.call(ctx.imager.instrument, "set_fov", fov))
            _done(entry, "ok", regions=regions, target=target,
                  mapping_calibrated=bool(mapping is not None and mapping.calibrated))
        except Exception as exc:
            _done(entry, "failed", error=f"{type(exc).__name__}: {exc}")
    else:
        _done(entry, "skipped")

    # 3. aberrations -------------------------------------------------------------
    entry = _stage(report, "aberrations")
    if plan.correct and ctx.corrector is not None:
        try:
            res = measure_and_correct(ctx.corrector, thresholds_m=plan.thresholds_m, resilient=resilient,
                                      max_correction_m=plan.max_correction_m, step=step)
            polished = {}
            for name, s in plan.polish.items():
                polished[name] = {k: v for k, v in tune_aberration(ctx.imager, ctx.corrector, name, step_m=s,
                                                                   score=plan.score, step=step).items()
                                  if k != "trials"}
            _done(entry, res["status"], result=res, polish=polished)
        except Exception as exc:
            _done(entry, "failed", error=f"{type(exc).__name__}: {exc}")
    else:
        _done(entry, "skipped", reason="no corrector" if ctx.corrector is None else "disabled")

    # 4. focus ------------------------------------------------------------------
    entry = _stage(report, "focus")
    if plan.focus_mode != "none":
        try:
            res = focus(ctx.imager, mode=plan.focus_mode, step_m=plan.focus_step_m,
                        max_offset_m=plan.focus_max_offset_m, score=plan.score, step=step)
            _done(entry, res["status"], result={k: v for k, v in res.items() if k != "trials"},
                  trials=len(res["trials"]))
        except Exception as exc:
            _done(entry, "failed", error=f"{type(exc).__name__}: {exc}")
    else:
        _done(entry, "skipped")

    # 5. final image -------------------------------------------------------------
    entry = _stage(report, "image")
    key, image = step("acquire", {"purpose": "final image"}, ctx.imager.acquire)
    final = all_metrics(image)
    gain = {k: (final[k] / report["initial"]["metrics"][k] - 1) if report["initial"]["metrics"][k] else None
            for k in final}
    report["final"] = {"key": key, "metrics": final, "relative_change": gain}
    _done(entry, "ok", key=key)

    # 6. record ------------------------------------------------------------------
    entry = _stage(report, "record")
    if plan.save_site and ctx.atlas is not None and ctx.mic is not None:
        try:
            fov = ctx.mic.current_fov()
            site = ctx.atlas.add_site(plan.name, image, ctx.mic.get_stage(), fov,
                                      notes=f"{plan.name}: {plan.score} {gain.get(plan.score, 0):+.1%}",
                                      settings=ctx.mic.settings(),
                                      provenance={**(provenance or {}), "session": _summary(report)})
            report["site_id"] = site.id
            _done(entry, "ok", site_id=site.id)
        except Exception as exc:
            _done(entry, "failed", error=f"{type(exc).__name__}: {exc}")
    else:
        _done(entry, "skipped")

    report["recoveries"] = list(resilient.events)
    failed = [s["stage"] for s in report["stages"] if s["status"] == "failed"]
    report["status"] = "completed" if not failed else "completed_with_failures"
    report["failed_stages"] = failed
    report["duration_s"] = round(time.time() - t0, 3)
    return report


def _summary(report: dict) -> dict:
    return {"name": report["name"], "stages": [{k: s.get(k) for k in ("stage", "status", "duration_s")}
                                               for s in report["stages"]]}


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
