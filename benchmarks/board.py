"""Verdicts, the capability board and run-to-run comparison for the workflow benchmark's results.

Pure functions over the results JSON that benchmarks/run_workflows.py writes: no stack, no model.

* A benchmark's ``bar`` is ``{pass: p, of: n}``: it passes when at least p/n of its counted trials passed, and
  it is ``unproven`` until n trials were counted (a 3 of 3 is weak evidence for a 4 of 5 bar).
* ``pass_hat_k`` is tau-bench's pass^k (arXiv 2406.12045): the chance that k trials of a task all pass.
* The suite's pass rate is the mean of the tasks' own rates, with a standard error clustered on the task
  (trials of one task are not independent), and two runs are compared as a paired difference over the tasks
  they share (Miller, "Adding Error Bars to Evals", arXiv 2411.00640).
"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

DEFAULT_BAR = {"pass": 4, "of": 5}
HISTORY = Path(__file__).resolve().parents[1] / "outputs" / "benchmarks" / "history"
KINDS = ("capability", "regression")
PASS, FAIL, UNPROVEN, SKIPPED = "pass", "fail", "unproven", "skipped"


def parse_bar(bar: object) -> dict:
    """A bar as ``{pass: p, of: n}`` from that mapping or from "p/n"; ValueError says what is wrong with it."""
    if bar is None:
        return dict(DEFAULT_BAR)
    if isinstance(bar, str) and re.fullmatch(r"\s*\d+\s*/\s*\d+\s*", bar):
        passed, of = (int(part) for part in bar.split("/"))
    elif isinstance(bar, dict) and set(bar) == {"pass", "of"}:
        passed, of = bar["pass"], bar["of"]
    else:
        raise ValueError(f"bar must be {{pass: p, of: n}} or \"p/n\", not {bar!r}")
    if not (isinstance(passed, int) and isinstance(of, int) and 1 <= passed <= of):
        raise ValueError(f"bar needs whole numbers with 1 <= pass <= of, not {bar!r}")
    return {"pass": passed, "of": of}


def verdict(passed: int, of: int, bar: dict | None = None) -> str:
    """pass, fail or unproven for ``passed`` of ``of`` counted trials against the bar.

    With fewer trials than the bar asks for, the verdict is fail only when the missing trials could not
    reach the bar even if all of them passed; otherwise it is unproven, whatever the rate so far."""
    bar = bar or DEFAULT_BAR
    if of <= 0:
        return SKIPPED
    if of < bar["of"]:
        return FAIL if passed + (bar["of"] - of) < bar["pass"] else UNPROVEN
    return PASS if passed * bar["of"] >= bar["pass"] * of else FAIL


def pass_hat_k(passed: int, of: int, k: int) -> float | None:
    """The unbiased estimate of pass^k from ``passed`` of ``of`` trials: C(passed, k) / C(of, k); None if of < k."""
    if k < 1 or of < k:
        return None
    return math.comb(passed, k) / math.comb(of, k)


def mean_and_error(values: list[float]) -> tuple[float, float] | None:
    """The mean and its standard error (0 for one value); None for no values."""
    if not values:
        return None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(variance / len(values))


def rows(results: dict) -> list[dict]:
    """One row per benchmark and runner that ran: its counts, bar, verdict and which checks failed."""
    out = []
    for key, value in (results.get("workflows") or {}).items():
        name, _, runner = key.rpartition("/")
        bar = parse_bar(value.get("bar"))
        passed, of = int(value.get("passed", 0)), int(value.get("of", 0))
        trials = value.get("rows") or []
        failed = sorted({label for r in trials for label, c in (r.get("checks") or {}).items() if not c.get("ok")}
                        | {"error" for r in trials if r.get("error") and not r.get("infra_error")})
        seconds = sorted(r.get("seconds", 0) for r in trials if not r.get("infra_error"))
        out.append({
            "key": key, "name": name, "runner": runner, "task_hash": value.get("task_hash"), "passed": passed, "of": of, "bar": bar,
            "verdict": verdict(passed, of, bar), "capability": value.get("capability") or "other",
            "kind": value.get("kind") or KINDS[0], "held_out": bool(value.get("held_out")),
            "failed_checks": failed, "median_s": seconds[len(seconds) // 2] if seconds else None,
            "infra_errors": int(value.get("infra_errors", 0)),
            "transcripts": [r["caller"] for r in trials if not r.get("ok") and not r.get("infra_error") and r.get("caller")],
        })
    return out


def summary(results: dict) -> dict:
    """The suite's pass rate with its clustered error, pass^k, and one line per capability."""
    counted = [r for r in rows(results) if r["of"] > 0]
    rate = mean_and_error([r["passed"] / r["of"] for r in counted])
    k = min((r["of"] for r in counted), default=0)
    hats = [pass_hat_k(r["passed"], r["of"], k) for r in counted] if k else []
    capabilities: dict[str, dict] = {}
    for r in counted:
        entry = capabilities.setdefault(r["capability"], {"passing": 0, "of": 0, "unproven": 0, "failing": []})
        entry["of"] += 1
        entry["passing"] += r["verdict"] == PASS
        entry["unproven"] += r["verdict"] == UNPROVEN
        if r["verdict"] == FAIL:
            entry["failing"].append(r["key"])
    for entry in capabilities.values():
        if entry["failing"]:
            entry["status"] = "partial" if entry["passing"] else "failing"
        else:
            entry["status"] = "unproven" if entry["unproven"] else "validated"
    return {
        "benchmarks": len(counted),
        "pass_rate": rate[0] if rate else None,
        "pass_rate_se": rate[1] if rate else None,
        "k": k,
        "pass_hat_k": sum(hats) / len(hats) if hats else None,
        "capabilities": dict(sorted(capabilities.items())),
        "regressions_failing": sorted(r["key"] for r in counted if r["kind"] == "regression" and r["verdict"] == FAIL),
    }


def compare(new: dict, old: dict) -> dict:
    """The paired difference in pass rate from ``old`` to ``new`` over benchmarks with the same ``task_hash``, and the
    verdicts that got worse or better."""
    before = {r["key"]: r for r in rows(old) if r["of"] > 0}
    after = {r["key"]: r for r in rows(new) if r["of"] > 0}
    both = sorted(set(before) & set(after))
    task_unknown = [k for k in both if not (before[k]["task_hash"] and after[k]["task_hash"])]
    task_changed = [k for k in both if k not in task_unknown and before[k]["task_hash"] != after[k]["task_hash"]]
    shared = [k for k in both if k not in task_unknown and k not in task_changed]
    difference = mean_and_error([after[k]["passed"] / after[k]["of"] - before[k]["passed"] / before[k]["of"] for k in shared])
    rank = {FAIL: 0, UNPROVEN: 1, PASS: 2}
    worse = [k for k in shared if rank[after[k]["verdict"]] < rank[before[k]["verdict"]]]
    better = [k for k in shared if rank[after[k]["verdict"]] > rank[before[k]["verdict"]]]
    return {
        "shared": len(shared), "only_new": sorted(set(after) - set(before)), "only_old": sorted(set(before) - set(after)),
        "difference": difference[0] if difference else None, "difference_se": difference[1] if difference else None,
        "worse": worse, "better": better, "task_changed": task_changed, "task_unknown": task_unknown,
        "old": {"model": old.get("model"), "git": old.get("git"), "started": old.get("started")},
    }


def history_path(results: dict, directory: Path) -> Path:
    """Where a finished run is kept: <directory>/<stack>/<time>_<model>_<git>.json."""
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(float(results.get("started") or time.time())))
    model = re.sub(r"[^A-Za-z0-9.]+", "-", str(results.get("model") or "model")).strip("-")
    return Path(directory) / str(results.get("stack") or "twin") / f"{stamp}_{model}_{results.get('git') or 'nogit'}.json"


def save_history(results: dict, directory: Path) -> Path:
    path = history_path(results, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return path


def load_history(directory: Path, stack: str | None = None, model: str | None = None) -> list[tuple[Path, dict]]:
    """Kept runs, oldest first, of one stack and model when given. A file that is not a results file is skipped."""
    found = []
    for path in sorted(Path(directory).glob("*/*.json")):
        try:
            results = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(results, dict) or "workflows" not in results:
            continue
        if (stack is None or results.get("stack") == stack) and (model is None or results.get("model") == model):
            found.append((path, results))
    return sorted(found, key=lambda item: float(item[1].get("started") or 0))


def _tasks(results: dict) -> set[tuple[str, str]]:
    return {(key, value["task_hash"]) for key, value in (results.get("workflows") or {}).items() if value.get("task_hash")}


def previous_run(results: dict, directory: Path) -> dict | None:
    """The newest earlier run of the same stack and model sharing a benchmark (name, runner and task hash)."""
    keys = _tasks(results)
    started = float(results.get("started") or math.inf)
    earlier = [r for _, r in load_history(directory, results.get("stack"), results.get("model"))
               if float(r.get("started") or 0) < started and keys & _tasks(r)]
    return earlier[-1] if earlier else None


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.0f} %"


def report(results: dict, old: dict | None = None) -> str:
    """The board as Markdown: capabilities, then every benchmark, then what changed since ``old``."""
    info = summary(results)
    table = rows(results)
    lines = [f"# Benchmark board: {results.get('model')}, stack {results.get('stack')}, {results.get('git', '?')}", ""]
    if info["pass_rate"] is None:
        return "\n".join(lines + ["No benchmark has a counted trial."]) + "\n"
    interval = 1.96 * (info["pass_rate_se"] or 0.0)
    lines.append(f"Pass rate {_percent(info['pass_rate'])} (+/- {100 * interval:.0f} points, 95 %) over {info['benchmarks']} benchmarks; "
                 f"pass^{info['k']} {_percent(info['pass_hat_k'])}.")
    if info["regressions_failing"]:
        lines.append(f"Regression benchmarks failing: {', '.join(info['regressions_failing'])}.")
    lines += ["", "| capability | passing | status |", "|---|---|---|"]
    for name, entry in info["capabilities"].items():
        lines.append(f"| {name} | {entry['passing']}/{entry['of']} | {entry['status']} |")
    lines += ["", "| benchmark | runner | kind | trials | bar | verdict | failed checks |", "|---|---|---|---|---|---|---|"]
    for r in table:
        if r["of"] <= 0:
            continue
        kind = r["kind"] + (", held out" if r["held_out"] else "")
        lines.append(f"| {r['name']} | {r['runner']} | {kind} | {r['passed']}/{r['of']} | {r['bar']['pass']}/{r['bar']['of']} | "
                     f"{r['verdict']} | {', '.join(r['failed_checks']) or '-'} |")
    transcripts = [(r["key"], r["transcripts"]) for r in table if r["transcripts"]]
    if transcripts:
        lines += ["", "Failed trials (outputs/live/transcripts/<name>.json):"]
        lines += [f"- {key}: {', '.join(names)}" for key, names in transcripts]
    if old is not None:
        change = compare(results, old)
        lines += ["", f"## Since {change['old']['git']} ({change['old']['model']})", ""]
        if change["task_changed"]:
            lines.append(f"Not paired, the task changed: {', '.join(change['task_changed'])}.")
        if change["task_unknown"]:
            lines.append(f"Not paired, a run records no task hash: {', '.join(change['task_unknown'])}.")
        if not change["shared"]:
            lines.append("No benchmark in common.")
        else:
            spread = 1.96 * (change["difference_se"] or 0.0)
            lines.append(f"Pass rate {100 * change['difference']:+.0f} points (+/- {100 * spread:.0f}) over {change['shared']} shared benchmarks.")
            lines.append(f"Worse: {', '.join(change['worse']) or 'none'}.")
            lines.append(f"Better: {', '.join(change['better']) or 'none'}.")
    return "\n".join(lines) + "\n"
