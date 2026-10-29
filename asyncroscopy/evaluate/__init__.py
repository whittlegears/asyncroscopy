"""Evaluators: score a run's operations and sequences, separate from the twin and the agents.

    from asyncroscopy.evaluate import load_run, score_run, features

    data = load_run("669e81c5855d")              # events from outputs/trace, data read lazily
    score_run(data)                               # {"information": {...}, "resolution": ..., "dose": ..., "time": ...}
    features(data)                                # one flat row for learning a reward

Command line: ``python -m asyncroscopy.evaluate --run <id>``. See docs/Operation/evaluators.md.
"""

from asyncroscopy.evaluate import builtin as _builtin  # noqa: F401  (registers the built-in evaluators)
from asyncroscopy.evaluate.composite import WeightedObjective, composite
from asyncroscopy.evaluate.core import (
    DEFAULT_EVALUATORS,
    ENTRY_POINT_GROUP,
    EVALUATORS,
    Evaluator,
    Score,
    evaluators,
    get_evaluator,
    register_evaluator,
    score_event,
    score_run,
)
from asyncroscopy.evaluate.features import (
    feature_units,
    features,
    features_table,
    write_table,
)
from asyncroscopy.evaluate.policy import (
    TuneDecision,
    best_budget,
    keep_tuning,
    stale_after_s,
)
from asyncroscopy.evaluate.run import (
    Image,
    RunData,
    data_keys,
    detect_blobs,
    load_run,
    match_particles,
)

__all__ = [
    "DEFAULT_EVALUATORS",
    "ENTRY_POINT_GROUP",
    "EVALUATORS",
    "Evaluator",
    "Image",
    "RunData",
    "Score",
    "TuneDecision",
    "WeightedObjective",
    "best_budget",
    "composite",
    "data_keys",
    "detect_blobs",
    "evaluators",
    "feature_units",
    "features",
    "features_table",
    "get_evaluator",
    "keep_tuning",
    "load_run",
    "match_particles",
    "register_evaluator",
    "score_event",
    "score_run",
    "stale_after_s",
    "write_table",
]
