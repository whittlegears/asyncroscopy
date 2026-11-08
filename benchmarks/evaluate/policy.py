"""When to stop tuning and start acquiring (organisers' principle 4), as pure functions.

Tuning raises the information rate of the acquisition that follows, but the seconds it takes
produce no data, and with focus drift the gain only lasts until the focus has walked out of the
depth of field again. The rule here: tune one more step while the information that step is
expected to add, per second of tuning, beats the information a second of acquiring now gives::

    window   = min(remaining - step, stale_after)        # how long the better focus is worth having
    marginal = (rate_tuned - rate_now) * window / step   # features per second spent tuning
    keep tuning  <=>  marginal > rate_now

The rates are the agent's (or the team's) estimates, e.g. from the ``information`` evaluator on the
last frames and from a model of how the rate depends on the focus error; benchmarks/tune_vs_acquire.py
measures them on the twin. Nothing here talks to an instrument, and nothing calls it on its own:
an agent or a job decides to.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass
class TuneDecision:
    keep_tuning: bool
    marginal_gain_per_s: float
    acquisition_rate_per_s: float
    window_s: float
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def stale_after_s(depth_of_field_m: float, focus_drift_m_per_s: float, residual_m: float = 0.0) -> float:
    """Seconds until drift takes a freshly tuned focus out of the depth of field (inf without drift)."""
    drift = abs(float(focus_drift_m_per_s))
    room = max(0.0, float(depth_of_field_m) - abs(float(residual_m)))
    return math.inf if drift == 0 else room / drift


def keep_tuning(rate_now_per_s: float, rate_tuned_per_s: float, step_s: float, remaining_s: float, stale_after: float = math.inf) -> TuneDecision:
    """Whether one more tuning step of ``step_s`` seconds is expected to pay for itself before the time runs out (or goes stale)."""
    if step_s <= 0:
        raise ValueError("step_s must be > 0")
    window = max(0.0, min(float(remaining_s) - float(step_s), float(stale_after)))
    marginal = (float(rate_tuned_per_s) - float(rate_now_per_s)) * window / float(step_s)
    keep = marginal > float(rate_now_per_s)
    reason = (
        f"a {step_s:g} s tuning step would add {marginal:.3g} features per tuning second over a {window:.3g} s window "
        f"({'remaining time' if window < stale_after else 'until drift makes it stale'}), "
        f"{'more' if keep else 'not more'} than the {rate_now_per_s:.3g} features/s acquiring now gives"
    )
    return TuneDecision(keep, marginal, float(rate_now_per_s), window, reason)


def best_budget(curve: list[dict], value: str = "information", budget: str = "tune_s") -> dict:
    """The point of a measured tune-vs-acquire curve with the most ``value``; ties go to the least ``budget``."""
    if not curve:
        raise ValueError("empty curve")
    return max(curve, key=lambda point: (point[value], -point[budget]))
