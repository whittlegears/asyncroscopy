"""Your own benchmarks for the agent: list them, prove they can tell a good run from a bad one, read the board.

None of this needs a stack or a model. The trials themselves are run by benchmarks/run_workflows.py.

    uv run python benchmarks/benchmark.py list                       # every benchmark by capability, with its bar
    uv run python benchmarks/benchmark.py validate                   # can each benchmark fail? does its reference pass?
    uv run python benchmarks/benchmark.py validate --only my_name --require-reference
    uv run python benchmarks/benchmark.py report                     # the board of the newest kept run, against the one before
    uv run python benchmarks/benchmark.py report --model gemma4:31b --stack twin

Validity (after the Agentic Benchmark Checklist, arXiv 2507.02825). A benchmark is valid when

* a run that did nothing and answered "Done." fails at least one check, and
* its ``reference`` (a made-up good run written beside the checks) passes every check.

A reference is not a run on the twin: it shows that the checks accept the outcome you mean, and reject
doing nothing. Whether the twin can do the task is shown the first time a trial passes.

    reference:
      before: {fov: 1.0e-6}                      # readings before the run (stage, fov, defocus); odd defaults otherwise
      calls:                                     # the calls of a good run, in order
        - {tool: Microscope_set_fov, args: {fov: 3.0e-7}}
        - {tool: Microscope_acquire_scanned_image, result: image_0001}
        - {tool: view_data, args: {key: image_0001}}
        - {tool: Microscope_move_stage, ok: false, result: "outside the limits"}    # a refused call
      after: {fov: 3.0e-7}                       # readings that differ after the run
      answer: The image is image_0001.           # the final answer
      facts: {truth: {...}}                      # optional: facts the runner would have measured
      events: [...]                              # optional: extra Instrument Trace events (job steps)
      risks: {Microscope_move_stage: high}       # optional: the endpoint's risk table, for the sleepwalking checks

``{result_2}`` in a later call's args or in the answer stands for what call 2 returned (an acquisition's key).

The same reference can be replayed on a running simulator, with no model, to show that the stack can do the task
and that the checks pass on what it really produced:

    uv run python benchmarks/run_workflows.py --workflows benchmarks/mine --runners reference

There the calls are made for real; ``result``, ``ok``, ``after``, ``facts`` and ``events`` are ignored (they are
measured), and the answer is the reference's with its {result_N} filled in.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add benchmarks/ so the runner's modules import the same way they do when it is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import board
from run_workflows import WORKFLOW_DIR, InfraError, WorkflowError, load_workflows, reference_calls, render
from workflow_checks import Call, Run, evaluate

HERE = Path(__file__).resolve().parent
MINE = HERE / "mine"  # your own benchmarks, one file per capability
HISTORY = board.HISTORY
# Readings no request would ask for, so a check on a target value cannot pass by accident.
BEFORE = {"stage": [1.13e-7, -7.7e-8, 0.0, 0.0, 0.0], "fov": 1.234e-6, "defocus": 1.1e-8}
NULL_ANSWER = "Done."


def default_paths() -> list[Path]:
    return [WORKFLOW_DIR] + ([MINE] if MINE.is_dir() else [])


# validity


def _before(wf: dict) -> dict:
    return {**BEFORE, **((wf.get("reference") or {}).get("before") or {})}


def null_run(wf: dict) -> Run:
    """A run that called nothing, changed nothing and answered "Done."."""
    before = _before(wf)
    return Run(facts={"before": before, "after": dict(before)}, final=NULL_ANSWER, has_calls=wf.get("runners") != ["mapping"],
               risks=(wf.get("reference") or {}).get("risks"))


def reference_run(wf: dict) -> Run:
    """The benchmark's ``reference`` as the Run the checks would see. A tool named <Class>_<command> is a device
    tool, so it is in the Instrument Trace too; a lowercase one (view_data, run_job) is the agent's own call."""
    reference = wf["reference"]
    before = _before(wf)
    calls, events, results = [], [], {}
    for i, call in enumerate(reference_calls(wf), 1):
        ok, result = bool(call.get("ok", True)), str(call.get("result", ""))
        args = render(dict(call.get("args") or {}), results)  # {result_1}: what call 1 returned, as in a replay
        calls.append(Call(str(call["tool"]), args, result, not ok))
        results[f"result_{i}"] = result
        if not str(call["tool"])[:1].islower():
            events.append({"source": "mcp", "action": str(call["tool"]), "ok": ok, "ts": float(i), "caller": "reference",
                           "arguments": args, **({"result": result} if ok else {"error": result})})
    events += [dict(e) for e in reference.get("events") or []]
    facts = {**(reference.get("facts") or {}), "before": before, "after": {**before, **(reference.get("after") or {})}}
    return Run(calls=calls, events=events, facts=facts, final=render(str(reference.get("answer", "")), results), has_calls=wf.get("runners") != ["mapping"],
               skills=list(reference.get("skills") or []), risks=reference.get("risks"))


def _failed(wf: dict, run: Run) -> dict[str, str]:
    """The checks ``run`` fails, with their details. An empty answer fails as it does in a live run."""
    facts = run.facts
    checks = evaluate(run, render(wf["checks"], {**facts, **(facts.get("truth") or {})}))
    if run.has_calls and not (run.final or "").strip():
        checks["answered"] = {"ok": False, "detail": "no final answer"}
    return {label: str(c["detail"]) for label, c in checks.items() if not c["ok"]}


def validity(wf: dict, require_reference: bool = False) -> dict:
    """Whether a benchmark can tell a good run from doing nothing: {valid, null, reference, why}."""
    out = {"name": wf["name"], "file": wf.get("file"), "valid": True, "why": []}
    try:
        null_failed = _failed(wf, null_run(wf))
    except InfraError as exc:  # a {placeholder} only a live setup fills
        return {**out, "valid": None, "null": "not checked", "reference": "not checked", "why": [f"needs a live setup: {exc}"]}
    out["null"] = "fails" if null_failed else "passes"
    if not null_failed:
        out["valid"] = False
        out["why"].append("doing nothing passes every check: add a check on the outcome or on a fact the answer must state")
    if wf.get("reference") is None:
        out["reference"] = "none"
        if require_reference:
            out["valid"] = False
            out["why"].append("no reference: write the calls, readings and answer of a good run")
        return out
    reference_failed = _failed(wf, reference_run(wf))
    out["reference"] = "fails" if reference_failed else "passes"
    if reference_failed:
        out["valid"] = False
        out["why"] += [f"the reference fails {label}: {detail}" for label, detail in reference_failed.items()]
    return out


# commands


def _select(args: argparse.Namespace) -> list[dict]:
    workflows = load_workflows(args.workflows)
    unknown = sorted(set(args.only or ()) - {wf["name"] for wf in workflows})
    if unknown:
        raise WorkflowError(f"no benchmark named {unknown}; `list` shows them")
    return [wf for wf in workflows if not args.only or wf["name"] in args.only]


def command_list(args: argparse.Namespace) -> int:
    workflows = _select(args)
    for capability in sorted({wf["capability"] for wf in workflows}):
        print(capability)
        for wf in (w for w in workflows if w["capability"] == capability):
            bar = board.parse_bar(wf.get("bar"))
            tags = [wf.get("kind", board.KINDS[0])] + (["held out"] if wf.get("held_out") else []) + (["reference"] if wf.get("reference") else [])
            print(f"  {wf['name']:34s} bar {bar['pass']}/{bar['of']}  stack {wf['stack']:6s} {', '.join(tags)}")
    print(f"{len(workflows)} benchmarks")
    return 0


def command_validate(args: argparse.Namespace) -> int:
    found = [validity(wf, args.require_reference) for wf in _select(args)]
    if args.json:
        print(json.dumps(found, indent=2))
    else:
        for item in found:
            state = {True: "VALID", False: "INVALID", None: "SKIPPED"}[item["valid"]]
            print(f"{item['name']:34s} {state:8s} do-nothing {item['null']:12s} reference {item['reference']}")
            for why in item["why"]:
                print(f"    {why}")
        invalid = sum(1 for item in found if item["valid"] is False)
        print(f"{len(found)} benchmarks: {invalid} invalid, {sum(1 for item in found if item['valid'] is None)} not checked offline")
    return 1 if any(item["valid"] is False for item in found) else 0


def command_report(args: argparse.Namespace) -> int:
    if args.results:
        results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    else:
        kept = board.load_history(Path(args.history), args.stack, args.model)
        if not kept:
            print(f"no kept run in {args.history}" + (f" for stack {args.stack}" if args.stack else "") + (f" and model {args.model}" if args.model else ""))
            return 1
        results = kept[-1][1]
    old = json.loads(Path(args.against).read_text(encoding="utf-8")) if args.against else board.previous_run(results, Path(args.history))
    print(board.report(results, old))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("list", "validate"):
        sub = commands.add_parser(name)
        sub.add_argument("--workflows", nargs="+", type=Path, default=default_paths(), help="YAML files or directories of them.")
        sub.add_argument("--only", nargs="*", default=None, help="Benchmark names.")
    validate = commands.choices["validate"]
    validate.add_argument("--require-reference", action="store_true", help="A benchmark without a reference is invalid (use it for new ones).")
    validate.add_argument("--json", action="store_true")
    report = commands.add_parser("report")
    report.add_argument("--results", default=None, help="A results JSON (default: the newest kept run).")
    report.add_argument("--against", default=None, help="A results JSON to compare with (default: the kept run before it).")
    report.add_argument("--history", default=str(HISTORY))
    report.add_argument("--stack", default=None)
    report.add_argument("--model", default=None)
    args = parser.parse_args(argv)
    try:
        return {"list": command_list, "validate": command_validate, "report": command_report}[args.command](args)
    except WorkflowError as exc:
        print(exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
