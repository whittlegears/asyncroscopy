"""Score one run from the Instrument Trace.

    python -m benchmarks.evaluate --run 669e81c5855d
    python -m benchmarks.evaluate --run 669e81c5855d --evaluators dose,time --json scores.json
    python -m benchmarks.evaluate --run wf-map-general-1 --trace-dir outputs/trace --tiled-uri http://127.0.0.1:9091
    python -m benchmarks.evaluate --run 669e81c5855d --weights information.features_found=1,dose.over_threshold_fraction=-100
    python -m benchmarks.evaluate --run 669e81c5855d --features            # the flat row for learning a reward
    python -m benchmarks.evaluate --list

``--run`` matches the agent run first, then the caller, then a job or reservation id. Prints a
table; ``--json -`` prints JSON instead, ``--json FILE`` writes it.
"""

from __future__ import annotations

import argparse
import json
import sys

from benchmarks.evaluate import (
    composite,
    evaluators,
    features,
    load_run,
    register_evaluator,
    score_run,
)
from benchmarks.evaluate.run import DEFAULT_TRACE_DIR


def _weights(text: str) -> dict[str, float]:
    out = {}
    for part in text.split(","):
        key, _, value = part.partition("=")
        if not value:
            raise SystemExit(f"--weights wants evaluator.objective=weight pairs, got {part!r}")
        out[key.strip()] = float(value)
    return out


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.evaluate", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", help="Agent run id (X-Asyncroscopy-Run), caller, or job/reservation id.")
    parser.add_argument("--evaluators", default=None, help="Comma-separated evaluator names (default: information,resolution,dose,time).")
    parser.add_argument("--trace-dir", action="append", default=None, help=f"Trace directory, file or glob (repeatable; default {DEFAULT_TRACE_DIR}).")
    parser.add_argument("--truth", default=None, help="The twin's export_ground_truth() JSON, for ground-truth scores (twin only).")
    parser.add_argument("--tiled-uri", default=None, help="Tiled server to read data keys from (default: keys are local files).")
    parser.add_argument("--weights", default=None, help="Add a 'reward' evaluator: evaluator.objective=weight,... (a weighted sum).")
    parser.add_argument("--features", action="store_true", help="Print the run's flat feature row instead of the scores.")
    parser.add_argument("--json", default=None, help="Write JSON to this file ('-': print JSON instead of the table).")
    parser.add_argument("--list", action="store_true", help="List the registered evaluators and exit.")
    args = parser.parse_args(argv)

    if args.list:
        for name, cls in sorted(evaluators().items()):
            print(f"{name:14s} {cls.version:7s} {cls.description}")
            for objective, (units, better) in cls.objectives.items():
                print(f"{'':14s}   {objective:26s} {units:22s} ({better})")
        return 0
    if not args.run:
        parser.error("--run is required (or --list)")
    names = [n.strip() for n in args.evaluators.split(",")] if args.evaluators else None
    if args.weights:
        register_evaluator(composite("reward", _weights(args.weights)))
        names = (names or ["information", "resolution", "dose", "time"]) + ["reward"]
    data = load_run(args.run, args.trace_dir, truth=args.truth, tiled_uri=args.tiled_uri)
    if not data.events:
        print(f"no trace events for {args.run!r} in {args.trace_dir or [DEFAULT_TRACE_DIR]}", file=sys.stderr)
        return 1
    out = {"run": args.run, "events": len(data.events)}
    if args.features:
        out["features"] = features(data, names)
    else:
        out["scores"] = score_run(data, names)
    if data.missing:
        out["unreadable_keys"] = data.missing
    text = json.dumps(out, indent=2, default=str)
    if args.json and args.json != "-":
        with open(args.json, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    if args.json == "-":
        print(text)
        return 0
    print(f"run {args.run}: {len(data.events)} trace events")
    if args.features:
        for key, value in out["features"].items():
            print(f"  {key:40s} {_fmt(value)}")
    else:
        for evaluator, scores in out["scores"].items():
            for objective, score in scores.items():
                print(f"  {evaluator:12s} {objective:26s} {_fmt(score['value']):>12s} {score['units']:20s} {score['reason']}")
    if data.missing:
        print(f"  ({len(data.missing)} data key(s) could not be read; give --tiled-uri for Tiled keys)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
