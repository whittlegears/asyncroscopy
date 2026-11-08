"""A team's own reward: a weighted sum of other evaluators' objectives.

    from benchmarks.evaluate import composite, register_evaluator

    register_evaluator(composite("careful", {
        "information.features_found": 1.0,
        "dose.over_threshold_fraction": -200.0,   # each 1 % of the area over the dose budget costs 2 features
        "time.idle_s": -0.1,
    }))

Weights carry the trade-off between objectives that conflict (more information costs dose
and time); choosing them is the team's decision, and learning them from operator traces is a
research problem in itself (``benchmarks.evaluate.features``). An objective that could not be
computed for a run (no images, no clock) is left out of the sum and named in the reason, so a
missing objective is never read as zero by accident; ``strict=True`` makes the reward null instead.
"""

from __future__ import annotations

from typing import Any

from benchmarks.evaluate.core import Evaluator, Score, get_evaluator
from benchmarks.evaluate.run import RunData


class WeightedObjective(Evaluator):
    """reward = sum(weight x value) over ``weights`` ({"evaluator.objective": weight})."""

    name = "weighted"
    description = "Weighted sum of other evaluators' objectives"
    objectives = {"reward": ("weighted sum", "max")}
    option_names = ("weights", "strict", "evaluator_options")
    weights: dict[str, float] = {}
    strict = False
    evaluator_options: dict[str, dict[str, Any]] = {}

    def _parts(self) -> dict[str, list[tuple[str, float]]]:
        parts: dict[str, list[tuple[str, float]]] = {}
        for key, weight in dict(self.weights).items():
            evaluator, _, objective = key.partition(".")
            if not objective:
                raise ValueError(f"weight key {key!r} must be 'evaluator.objective'")
            parts.setdefault(evaluator, []).append((objective, float(weight)))
        return parts

    def _combine(self, scored: dict[str, dict[str, Score]]) -> dict[str, Score]:
        total, terms, missing = 0.0, [], []
        for evaluator, objectives in self._parts().items():
            for objective, weight in objectives:
                score = scored.get(evaluator, {}).get(objective)
                if score is None or score.value is None:
                    missing.append(f"{evaluator}.{objective}")
                    continue
                total += weight * float(score.value)
                terms.append(f"{weight:g} x {evaluator}.{objective} ({float(score.value):.4g})")
        if missing and self.strict:
            return {"reward": Score(None, "weighted sum", f"missing {', '.join(missing)} (strict)")}
        reason = " + ".join(terms) or "no terms"
        if missing:
            reason += f"; left out (not computed): {', '.join(missing)}"
        return {"reward": Score(total, "weighted sum", reason)}

    def _evaluator(self, name: str) -> Evaluator:
        return get_evaluator(name, **dict(self.evaluator_options).get(name, {}))

    def score_run(self, data: RunData) -> dict[str, Score]:
        return self._combine({name: self._evaluator(name).score_run(data) for name in self._parts()})

    def score_event(self, event: dict, data: RunData) -> dict[str, Score]:
        scored = {name: self._evaluator(name).score_event(event, data) for name in self._parts()}
        if not any(scored.values()):
            return {}
        return self._combine(scored)


def composite(name: str, weights: dict[str, float], strict: bool = False, evaluator_options: dict | None = None, description: str = "") -> type[WeightedObjective]:
    """A named WeightedObjective class with these weights, ready for ``register_evaluator``."""
    attrs = {
        "name": name,
        "weights": dict(weights),
        "strict": bool(strict),
        "evaluator_options": dict(evaluator_options or {}),
        "description": description or "weighted: " + ", ".join(f"{w:g} x {k}" for k, w in weights.items()),
    }
    return type(f"Composite_{name}", (WeightedObjective,), attrs)
