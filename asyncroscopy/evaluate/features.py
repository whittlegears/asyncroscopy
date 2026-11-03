"""A run as one flat row: trace -> features -> scores, the table a team could fit a reward to.

``features(run)`` returns a flat dict of plain numbers and strings:

* ``run.*``: what the trace says the run did (events, tool calls, job steps, failures,
  approvals, dry-run, simulated), counted without reading any data;
* ``<evaluator>.<objective>``: every registered evaluator's run scores (values only;
  ``feature_units()`` gives the units).

Many rows (``features_table(runs)``) with an outcome column added by the team (an operator's
rating, a later analysis result) are the input for learning a reward: regress the outcome on
the columns, or fit the weights of a ``composite``. Nothing here learns; the point is that the
inputs and outputs are a clean table.
"""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from asyncroscopy.evaluate.core import Evaluator, _instances, _safe
from asyncroscopy.evaluate.run import RunData, as_run_data


def trace_features(data: RunData) -> dict[str, Any]:
    """Counts from the events alone (no data read)."""
    events = data.events
    sources = Counter(str(e.get("source", "")) for e in events)
    actions = [str(e.get("action", "")) for e in events]
    steps = [e for e, a in zip(events, actions) if a.endswith(".step")]
    simulated = {e.get("simulated") for e in events if "simulated" in e}
    return {
        "run.id": data.run or "",
        "run.events": len(events),
        "run.tool_calls": sources.get("mcp", 0),
        "run.job_steps": len(steps),
        "run.jobs": len({e.get("job_id") for e in events if e.get("source") == "job" and e.get("job_id")}),
        "run.failed": sum(1 for e in events if e.get("ok") is False),
        "run.approvals": sources.get("approvals", 0),
        "run.dry_run_events": sum(1 for e in events if e.get("dry_run")),
        "run.acquire_calls": sum(1 for a in actions if "acquire" in a.lower()),
        "run.distinct_actions": len(set(actions)),
        "run.simulated": (simulated == {True}) if simulated else None,
    }


def features(run: RunData | str | list, evaluators: str | Iterable[str | Evaluator] | None = None) -> dict[str, Any]:
    """One flat row for ``run`` (a RunData, a run id read from outputs/trace, or a list of events)."""
    data = as_run_data(run)
    row = trace_features(data)
    for evaluator in _instances(evaluators):
        for objective, score in _safe(lambda e=evaluator: e.score_run(data), evaluator.name).items():
            row[f"{evaluator.name}.{objective}"] = score["value"]
    return row


def feature_units(evaluators: str | Iterable[str | Evaluator] | None = None) -> dict[str, str]:
    """Units of the evaluator columns of ``features`` ({"dose.max_dose_e_per_a2": "e/A^2", ...})."""
    return {f"{e.name}.{objective}": units for e in _instances(evaluators) for objective, (units, _better) in e.objectives.items()}


def features_table(runs: Iterable[RunData | str | list], evaluators: str | Iterable[str | Evaluator] | None = None) -> list[dict[str, Any]]:
    return [features(run, evaluators) for run in runs]


def write_table(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Rows as CSV (columns: the union of every row's keys, in first-seen order)."""
    path = Path(path)
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path
