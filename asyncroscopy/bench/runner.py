"""Run and score agent benchmarks on a running benchmark twin stack.

Baseline (our routines) over several seeds, as a leaderboard row::

    uv run python -m asyncroscopy.bench baseline --seeds 3

Score an agent (hackathon flow)::

    uv run python -m asyncroscopy.bench setup --task focus --seed 7 --state run.json
    # ... the agent works through MCP (configs/mcp_benchmark.yaml) ...
    uv run python -m asyncroscopy.bench score --state run.json --team "my team" --board leaderboard.jsonl

``setup`` prints the task instructions to hand to the agent. ``score``
reads ground truth from the twin and appends a row to the leaderboard file.
If supervision is on, the run's action history (who did what, how many
instrument calls) is attached from the queue's ``history.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from asyncroscopy.bench.tasks import TASKS, Env


def connect(instrument: str, corrector: str, data: str) -> Env:
    import h5py
    import tango

    inst = tango.DeviceProxy(instrument)
    inst.set_timeout_millis(120_000)
    corr = None
    try:
        corr = tango.DeviceProxy(corrector)
        corr.ping()
    except Exception:
        corr = None
    uri = None
    try:
        uri = json.loads(tango.DeviceProxy(data).get_config()).get("uri")
    except Exception:
        pass

    def read(key: str) -> np.ndarray:
        if os.path.exists(key):
            with h5py.File(key, "r") as h5:
                g = h5["image"]
                return (g if isinstance(g, h5py.Dataset) else g[next(iter(g))])[()]
        from tiled.client import from_uri

        from asyncroscopy.mcp.mcp_server import MCPServer
        return MCPServer._find_first_2d_array(from_uri(uri)[key], prefer_key="image")

    return Env(instrument=inst, corrector=corr, fetch_image=read, fetch_pattern=read, corrector_name=corrector)


def run_baseline(env: Env, tasks: list[str], seeds: list[int], log=print) -> list[dict]:
    rows = []
    for name in tasks:
        task = TASKS[name]
        if "corrector" in task.needs and env.corrector is None:
            log(f"skip {name}: no corrector device")
            continue
        for seed in seeds:
            state = task.setup(env, np.random.default_rng(seed))
            t0 = time.time()
            try:
                detail = task.baseline(env)
                status = detail.get("status")
            except Exception as exc:
                status = f"error: {exc}"
            result = task.score(env, state)
            row = {"task": name, "seed": seed, "team": "asyncroscopy baseline", "duration_s": time.time() - t0,
                   "routine_status": status, **result}
            rows.append(row)
            log(f"{name:12s} seed {seed:3d}  {'PASS' if result['pass'] else 'fail'}  error {result['error']:.4g} "
                f"{result['unit']}  score {result['score']:.2f}  {row['duration_s']:.1f}s")
    return rows


def leaderboard_markdown(rows: list[dict]) -> str:
    teams = sorted({r["team"] for r in rows})
    tasks = sorted({r["task"] for r in rows})
    lines = ["| team | " + " | ".join(tasks) + " | mean score |", "|---|" + "---:|" * (len(tasks) + 1)]
    for team in teams:
        cells, scores = [], []
        for t in tasks:
            rs = [r for r in rows if r["team"] == team and r["task"] == t]
            if not rs:
                cells.append("—")
                continue
            s = float(np.mean([r["score"] for r in rs]))
            scores.append(s)
            cells.append(f"{s:.2f} ({sum(r['pass'] for r in rs)}/{len(rs)} pass)")
        lines.append(f"| {team} | " + " | ".join(cells) + f" | {np.mean(scores):.2f} |" if scores else "")
    return "\n".join(lines) + "\n"


def _history_summary(path: Path, since: float) -> dict:
    from asyncroscopy.supervision.queue import load_history

    actions = [a for a in load_history(path).values() if a.get("created", 0) >= since]
    top = [a for a in actions if not a.get("parent_id")]
    return {"actions": len(top), "instrument_steps": len(actions),
            "by_origin": {o: sum(a["origin"] == o for a in top) for o in ("agent", "operator", "recipe")},
            "refused": sum(a["status"] in ("refused", "rejected", "denied") for a in actions)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("baseline", "setup", "score"):
        p = sub.add_parser(name)
        p.add_argument("--tango-host", default=os.environ.get("TANGO_HOST", "127.0.0.1:9094"))
        p.add_argument("--instrument", default="asyncroscopy/instrument/default")
        p.add_argument("--corrector", default="asyncroscopy/corrector/default")
        p.add_argument("--data", default="asyncroscopy/data/default")
    b = sub.choices["baseline"]
    b.add_argument("--tasks", nargs="*", default=list(TASKS))
    b.add_argument("--seeds", type=int, default=3)
    b.add_argument("--board", type=Path, default=None)
    s = sub.choices["setup"]
    s.add_argument("--task", choices=list(TASKS), required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--state", type=Path, default=Path("bench_state.json"))
    c = sub.choices["score"]
    c.add_argument("--state", type=Path, default=Path("bench_state.json"))
    c.add_argument("--team", required=True)
    c.add_argument("--board", type=Path, default=Path("leaderboard.jsonl"))
    c.add_argument("--history", type=Path, default=Path("outputs/supervision/history.jsonl"))
    a = ap.parse_args(argv)
    os.environ["TANGO_HOST"] = a.tango_host
    env = connect(a.instrument, a.corrector, a.data)

    if a.cmd == "baseline":
        rows = run_baseline(env, a.tasks, list(range(a.seeds)))
        if a.board:
            with a.board.open("a", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, default=str) + "\n")
        print()
        print(leaderboard_markdown(rows))
        return 0
    if a.cmd == "setup":
        task = TASKS[a.task]
        state = task.setup(env, np.random.default_rng(a.seed))
        a.state.write_text(json.dumps({"task": a.task, "seed": a.seed, "state": state, "started": time.time()}))
        print(task.instructions)
        return 0
    saved = json.loads(a.state.read_text())
    result = TASKS[saved["task"]].score(env, saved["state"])
    row = {"task": saved["task"], "seed": saved["seed"], "team": a.team,
           "duration_s": time.time() - saved["started"], **result}
    if a.history.exists():
        row["history"] = _history_summary(a.history, saved["started"])
    with a.board.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str) + "\n")
    print(json.dumps(row, indent=2, default=str))
    rows = [json.loads(line) for line in a.board.read_text().splitlines() if line.strip()]
    print()
    print(leaderboard_markdown(rows))
    return 0
