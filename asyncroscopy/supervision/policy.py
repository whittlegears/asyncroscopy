"""Hard limits and approval gates, enforced in the control layer.

These rules live here, where every action passes on its way to the instrument,
and not in the agent's prompt: a prompt can be argued with, ignored, or
truncated, a check in the queue cannot. The agent has no tool that changes
the policy or approves its own actions; only the operator HTTP API can approve.

Three kinds of rule:

``forbid``
    Tool name patterns (``fnmatch``) that are refused outright.
``require_approval``
    Patterns whose actions wait in the queue, ``awaiting_approval``, until an
    operator approves or denies them. Approval is per action.
``stage_limits`` / ``arg_limits``
    Numeric bounds. A move outside ``stage_limits`` (or a jump larger than
    ``max_step_xy_m``) is refused and nothing moves; ``arg_limits`` bounds named
    arguments of matching tools, e.g. the dwell time of an acquisition.

Patterns match the queued tool name, which is ``<DeviceClass>_<command>`` for
Tango tools (``DigitalTwin_move_stage``) and ``atlas.*`` / ``recipe.*`` for the
supervision tools, and also the role form ``<role>.<command>``
(``instrument.move_stage``), so one policy covers a twin and a real scope.
"""

from __future__ import annotations

import fnmatch
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any


class PolicyViolation(RuntimeError):
    """A hard limit refused the action. Nothing was sent to the instrument."""


@dataclass
class Decision:
    allowed: bool
    needs_approval: bool = False
    reason: str = ""


STAGE_AXES = ("x", "y", "z", "alpha", "beta")


@dataclass
class ControlPolicy:
    forbid: list[str] = field(default_factory=list)
    require_approval: list[str] = field(default_factory=list)
    # axis -> [min, max]; x/y/z in metres, alpha/beta in degrees
    stage_limits: dict[str, list[float]] = field(default_factory=dict)
    max_step_xy_m: float | None = None
    # tool pattern -> {argument: [min, max]}
    arg_limits: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    # Recipes may run on a real instrument only after a successful run of the
    # same recipe version on a digital twin.
    require_twin_validation: bool = True

    @classmethod
    def from_config(cls, data: dict | None) -> ControlPolicy:
        data = dict(data or {})
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def _matches(patterns: list[str], names: list[str]) -> str | None:
        for pattern in patterns:
            for name in names:
                if fnmatch.fnmatchcase(name, pattern):
                    return pattern
        return None

    def check(self, names: list[str], args: dict[str, Any],
              current_stage: Callable[[], list[float]] | None = None) -> Decision:
        """Decide on one action. *names* are the tool's aliases (class form, role form)."""
        hit = self._matches(self.forbid, names)
        if hit:
            return Decision(False, reason=f"{names[0]} is forbidden by policy ({hit!r})")

        if any(n.endswith("move_stage") for n in names):
            target = _stage_target(args)
            if target is not None:
                problem = self._check_stage(target, current_stage)
                if problem:
                    return Decision(False, reason=problem)

        for pattern, limits in self.arg_limits.items():
            if self._matches([pattern], names):
                for arg, (lo, hi) in limits.items():
                    if arg in args and args[arg] is not None:
                        try:
                            value = float(args[arg])
                        except (TypeError, ValueError):
                            continue
                        if not (lo <= value <= hi):
                            return Decision(False, reason=f"{arg}={value:g} is outside the policy "
                                                          f"limit [{lo:g}, {hi:g}] for {pattern!r}")

        hit = self._matches(self.require_approval, names)
        if hit:
            return Decision(True, needs_approval=True, reason=f"policy requires operator approval ({hit!r})")
        return Decision(True)

    def _check_stage(self, target: list[float], current_stage) -> str | None:
        for axis, value in zip(STAGE_AXES, target):
            bounds = self.stage_limits.get(axis)
            if bounds is not None and not (bounds[0] <= value <= bounds[1]):
                unit = "deg" if axis in ("alpha", "beta") else "m"
                return (f"stage {axis}={value:g} {unit} is outside the policy limit "
                        f"[{bounds[0]:g}, {bounds[1]:g}] {unit}; nothing was moved")
        if self.max_step_xy_m is not None and current_stage is not None:
            try:
                now = current_stage()
            except Exception as exc:  # cannot verify the step size: refuse rather than guess
                return f"cannot read the stage to check max_step_xy_m ({exc}); nothing was moved"
            step = math.hypot(target[0] - now[0], target[1] - now[1])
            if step > self.max_step_xy_m:
                return (f"stage move of {step:.3g} m exceeds max_step_xy_m {self.max_step_xy_m:g} m; "
                        f"nothing was moved")
        return None


def _stage_target(args: dict[str, Any]) -> list[float] | None:
    for key in ("position", "arg", "target"):
        value = args.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                return [float(v) for v in value]
            except (TypeError, ValueError):
                return None
    return None
