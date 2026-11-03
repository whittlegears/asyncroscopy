"""The Evaluator interface and its registry.

An evaluator scores what a run did, never how the run was driven: it reads the run's
Instrument Trace events and the data they reference (``RunData``) and returns
``{objective: Score}``. Two levels:

* ``score_event(event, data)``: one operation (an acquisition, a job step, a tool call);
  ``{}`` when the event says nothing about this evaluator's objectives;
* ``score_run(data)``: the whole sequence.

Evaluators are kept apart from the twin, the agents and each other (organisers' principle 3),
so resolution, damage, information and time can be explored as separate objectives and
combined only where a team chooses to (``asyncroscopy.evaluate.composite``).

A new evaluator is a small class registered with ``@register_evaluator`` or installed with the
``asyncroscopy.evaluators`` entry point::

    [project.entry-points."asyncroscopy.evaluators"]
    contamination = "my_package.evaluators:Contamination"
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

ENTRY_POINT_GROUP = "asyncroscopy.evaluators"

log = logging.getLogger(__name__)


@dataclass
class Score:
    """One objective's value, its units and a short reason a person can check."""

    value: float | None
    units: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        value = self.value
        if isinstance(value, float) and not math.isfinite(value):
            value = None  # JSON has no NaN or inf: an objective that could not be computed is null
        return {"value": value, "units": self.units, "reason": self.reason}


class Evaluator:
    """Scores a run's operations and the run as a whole. Subclass, set ``name``, implement ``score_run``.

    ``objectives``: {objective: (units, "max" | "min")}, what ``score_run`` returns and which way is
    better, so a composite or a learner can read them without running anything. ``version``: bump it
    when a score changes meaning, so scores from different versions are not mixed. Options are
    keyword arguments of the constructor (``get_evaluator("dose", threshold_e_per_a2=500)``).
    """

    name = ""
    version = "1.0.0"
    description = ""
    objectives: dict[str, tuple[str, str]] = {}

    def __init__(self, **options: Any) -> None:
        unknown = set(options) - set(self.options())
        if unknown:
            raise ValueError(f"evaluator {self.name} has no option(s) {sorted(unknown)}; options: {self.options()}")
        for key, value in options.items():
            setattr(self, key, value)

    @classmethod
    def options(cls) -> dict[str, Any]:
        """Options and their defaults: the class attributes listed in ``option_names``."""
        return {key: getattr(cls, key) for key in getattr(cls, "option_names", ())}

    def score_event(self, event: dict, data: RunData) -> dict[str, Score]:  # noqa: F821 (RunData: asyncroscopy.evaluate.run)
        return {}

    def score_run(self, data: RunData) -> dict[str, Score]:  # noqa: F821
        raise NotImplementedError


EVALUATORS: dict[str, type[Evaluator]] = {}
_entry_points_loaded = False


def register_evaluator(cls: type[Evaluator]) -> type[Evaluator]:
    """Make ``cls`` available by its ``name`` (``--evaluators name`` on the command line); usable as a class decorator."""
    if not getattr(cls, "name", ""):
        raise ValueError(f"{cls.__name__} needs a name")
    EVALUATORS[cls.name] = cls
    return cls


def evaluators() -> dict[str, type[Evaluator]]:
    """Every evaluator: built in, registered, and installed with the ``asyncroscopy.evaluators`` entry point."""
    global _entry_points_loaded
    if not _entry_points_loaded:
        _entry_points_loaded = True
        from importlib.metadata import entry_points

        for point in entry_points(group=ENTRY_POINT_GROUP):
            try:
                register_evaluator(point.load())
            except Exception:  # a broken plug-in must not take the others down
                log.warning("evaluator entry point %s could not be loaded", point.name, exc_info=True)
    return EVALUATORS


def get_evaluator(name: str, **options: Any) -> Evaluator:
    """An instance of the evaluator called ``name``; ValueError listing the known ones otherwise."""
    known = evaluators()
    if name not in known:
        raise ValueError(f"unknown evaluator {name!r}; registered: {sorted(known)}")
    return known[name](**options)


def _instances(names) -> list[Evaluator]:
    if names is None:
        names = DEFAULT_EVALUATORS
    if isinstance(names, str):
        names = [n.strip() for n in names.split(",") if n.strip()]
    return [n if isinstance(n, Evaluator) else get_evaluator(n) for n in names]


DEFAULT_EVALUATORS = ("information", "resolution", "dose", "time")


def _safe(call, name: str) -> dict[str, dict]:
    """One evaluator's scores as plain dicts; a failure is reported as a score, never raised into the caller's run."""
    try:
        return {objective: score.to_dict() for objective, score in call().items()}
    except Exception as exc:
        return {"error": Score(None, "", f"{type(exc).__name__}: {exc}"[:300]).to_dict()}


def score_run(data: RunData | str | list, names: str | Iterable[str | Evaluator] | None = None) -> dict[str, dict[str, dict]]:  # noqa: F821
    """{evaluator: {objective: {value, units, reason}}} for a whole run (``data``: a RunData, or a run id)."""
    from asyncroscopy.evaluate.run import as_run_data

    data = as_run_data(data)
    return {e.name: _safe(lambda e=e: e.score_run(data), e.name) for e in _instances(names)}


def score_event(event: dict, data: RunData | str | list, names: str | Iterable[str | Evaluator] | None = None) -> dict[str, dict[str, dict]]:  # noqa: F821
    """{evaluator: {objective: ...}} for one event of ``data``; evaluators with nothing to say are left out."""
    from asyncroscopy.evaluate.run import as_run_data

    data = as_run_data(data)
    out = {}
    for e in _instances(names):
        scores = _safe(lambda e=e: e.score_event(event, data), e.name)
        if scores:
            out[e.name] = scores
    return out
