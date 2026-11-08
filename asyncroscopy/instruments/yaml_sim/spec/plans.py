"""Job plans of a YAML instrument: resolving a plan into points, its estimate and what it sets."""

from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass
from typing import Any

from .schema import InstrumentSpec, Job, Quantity, duration_seconds
from .expressions import Refused, evaluate

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


def job_estimate(spec: InstrumentSpec, resolved: JobPlan, values: dict[str, Any]) -> dict:
    """Points and seconds: the move time plus every acquisition's duration, at each point (sampled for big plans)."""
    sample = range(resolved.count) if resolved.count <= 2000 else sorted({0, resolved.count - 1, *range(0, resolved.count, max(1, resolved.count // 1000))})
    total = 0.0
    for index in sample:
        scope = {**values, **resolved.settings, **resolved.point(index)}
        total += resolved.job.settle_s + sum(duration_seconds(spec.acquisitions[a].duration, scope) for a in resolved.job.acquire)
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
