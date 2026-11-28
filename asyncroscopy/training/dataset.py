"""Grade recorded runs and export the good ones as a training set.

The recorder (recorder.py) writes raw records and never changes them. Grades
go in a separate file, ``grades.jsonl``, keyed by run id; the newest grade for a
run wins, so a run can be re-graded.

    python -m asyncroscopy.training.dataset grade outputs/training_data <run id> --passed
    python -m asyncroscopy.training.dataset export outputs/training_data train.jsonl --holdout holdout.txt

The run id is the "id" field of a line in runs.jsonl. ``grade`` without
``--passed`` records a fail.

The export writes every model call of every run that passed, as
{"tools": [...], "messages": [...]} lines. Later calls in a run repeat the
earlier calls' messages, so the steps early in a run appear more than once.
Prompts in the holdout file (one per line) never go in, so they stay fair test prompts.
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

GRADES_FILE = "grades.jsonl"
RUNS_FILE = "runs.jsonl"
CALLS_FILE = "llm_calls.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSON-lines file, skipping (with a warning) any line that is damaged."""
    if not path.exists():
        return []
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            warnings.warn(f"{path.name} line {number} is not valid JSON; skipped", RuntimeWarning, stacklevel=2)
    return records


def load_runs(directory: str | Path) -> list[dict]:
    return _read_jsonl(Path(directory) / RUNS_FILE)


def load_grades(directory: str | Path) -> dict[str, dict]:
    """The newest grade for each run id."""
    return {grade["run_id"]: grade for grade in _read_jsonl(Path(directory) / GRADES_FILE)}


def add_grade(directory: str | Path, run_id: str, passed: bool, score: float | None = None,
              grader: str = "", details: dict | None = None) -> dict:
    """Append a grade for a recorded run. ``details`` can hold the scores it was based on."""
    if run_id not in {run["id"] for run in load_runs(directory)}:
        raise ValueError(f"No recorded run has id {run_id!r}")
    grade = {
        "run_id": run_id,
        "passed": bool(passed),
        "score": score,
        "grader": grader,
        "details": details or {},
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with (Path(directory) / GRADES_FILE).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(grade) + "\n")
    return grade


def _same_prompt(text: str) -> str:
    """Prompts match ignoring case and spacing."""
    return " ".join(text.lower().split())


def export_training_set(directory: str | Path, out_path: str | Path, holdout_prompts: list[str] = ()) -> dict:
    """Write the model calls of passed runs, leaving out held-out prompts. Returns counts."""
    runs = {run["id"]: run for run in load_runs(directory)}
    passed = {run_id for run_id, grade in load_grades(directory).items() if grade["passed"] and run_id in runs}
    holdout = {_same_prompt(prompt) for prompt in holdout_prompts if prompt.strip()}
    held_out = {run_id for run_id in passed if _same_prompt(runs[run_id]["prompt"]) in holdout}
    keep = passed - held_out

    calls = [call for call in _read_jsonl(Path(directory) / CALLS_FILE) if call.get("run_id") in keep]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for call in calls:
            handle.write(json.dumps({"tools": call.get("tools", []), "messages": call["messages"]}, ensure_ascii=False) + "\n")
    return {
        "runs_recorded": len(runs),
        "runs_passed": len(passed),
        "runs_held_out": len(held_out),
        "examples_written": len(calls),
        "path": str(out),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grade recorded agent runs and export a training set.")
    actions = parser.add_subparsers(dest="action", required=True)

    grade = actions.add_parser("grade", help="Append a grade for one run.")
    grade.add_argument("directory")
    grade.add_argument("run_id")
    grade.add_argument("--passed", action="store_true", help="Mark the run as good (default: failed).")
    grade.add_argument("--score", type=float)
    grade.add_argument("--grader", default="")

    export = actions.add_parser("export", help="Write the model calls of passed runs.")
    export.add_argument("directory")
    export.add_argument("out_path")
    export.add_argument("--holdout", help="Text file with one held-out prompt per line.")

    args = parser.parse_args(argv)
    if args.action == "grade":
        result = add_grade(args.directory, args.run_id, args.passed, args.score, grader=args.grader)
    else:
        holdout = Path(args.holdout).read_text(encoding="utf-8").splitlines() if args.holdout else []
        result = export_training_set(args.directory, args.out_path, holdout)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
