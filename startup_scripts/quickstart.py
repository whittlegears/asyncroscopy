#!/usr/bin/env python
"""One command from a fresh checkout to an agent-ready digital microscope.

    uv run startup_scripts/quickstart.py                 # twin stack + MCP + checks
    uv run startup_scripts/quickstart.py --demo          # ...and run one autonomous session through MCP
    uv run startup_scripts/quickstart.py --stack benchmark --demo   # tilt/diffraction twin (needs --extra diffraction)

It starts the device stack with run_servers.py and the MCP bridge with
run_mcp.py (the same launchers you use by hand, so nothing behaves
differently), waits until each is ready, runs the doctor, optionally drives
one autonomous session as an agent would, and prints where to connect:

* agents (Claude Code, SciAgentGUI, notebooks): http://localhost:<port>/mcp

Ctrl+C (or SIGTERM) stops everything it started.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from asyncroscopy.utils.process_manager import ProcessManager  # noqa: E402

STACKS = {
    "gold": ("configs/polycrystalline_gold.yaml", "configs/mcp_dt.yaml",
             "atomistic polycrystalline gold + corrector twin (aberrations change the image)"),
    "benchmark": ("configs/benchmark_twin.yaml", "configs/mcp_benchmark.yaml",
                  "silicon-lamella tilt/diffraction twin + corrector twin (focus, zone axis, benchmark)"),
    "basic": ("configs/DigitalTwin.yaml", "configs/mcp_dt.yaml", "nanoparticle twin (fastest start)"),
}


def wait_for(proc: subprocess.Popen, log: Path, markers: tuple[str, ...], failures: tuple[str, ...],
             timeout: float) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        for f in failures:
            if f in text:
                raise RuntimeError(f"{f!r} in {log}:\n" + "\n".join(text.splitlines()[-15:]))
        for m in markers:
            if m in text:
                return m
        if proc.poll() is not None:
            raise RuntimeError(f"process exited with code {proc.returncode}; last lines of {log}:\n"
                               + "\n".join(text.splitlines()[-15:]))
        time.sleep(0.5)
    raise TimeoutError(f"not ready after {timeout:.0f} s; see {log}")


def mcp_config(base: Path, port: int, tool_log: Path, tango: tuple[str, int]) -> Path:
    data = yaml.safe_load(base.read_text(encoding="utf-8"))
    data["tango"] = {"host": tango[0], "port": tango[1]}
    data["mcp"]["http_port"] = port
    data["mcp"]["http_host"] = "127.0.0.1"
    data["mcp"]["tool_log"] = str(tool_log)
    out = Path(tempfile.mkdtemp(prefix="asyncroscopy-quickstart-")) / "mcp.yaml"
    out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return out


def demo(url: str) -> dict:
    """Drive one autonomous session through MCP, exactly as an agent would."""
    import asyncio

    from fastmcp import Client

    async def run():
        async with Client(url, timeout=900) as c:
            tools = {t.name for t in await c.list_tools()}
            health = (await c.call_tool("system_health", {})).data
            plan = {"name": "quickstart session", "find_region": False}
            if any(n.endswith("set_aberrations_coeff_sim") for n in tools):
                setter = next(n for n in tools if n.endswith("set_aberrations_coeff_sim"))
                await c.call_tool(setter, {"json_aberrations_string": json.dumps({"C1": [20e-9], "A1": [5e-9, -3e-9]})})
            result = await c.call_tool("autonomy_run_session", {"plan": plan}, raise_on_error=False)
            return {"health": health, "session": result.data if result.data is not None else
                    [getattr(x, "text", str(x)) for x in result.content]}

    return asyncio.run(run())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stack", choices=list(STACKS), default="gold")
    ap.add_argument("--port", type=int, default=8000, help="MCP HTTP port")
    ap.add_argument("--demo", action="store_true", help="run one autonomous session through MCP")
    ap.add_argument("--exit-after-demo", action="store_true", help="stop everything after the demo (CI)")
    ap.add_argument("--logs", type=Path, default=PROJECT_DIR / "outputs" / "quickstart")
    a = ap.parse_args(argv)

    servers_yaml, mcp_yaml, description = STACKS[a.stack]
    servers_cfg = yaml.safe_load((PROJECT_DIR / servers_yaml).read_text(encoding="utf-8"))
    tango = (servers_cfg["tango"]["host"], int(servers_cfg["tango"]["port"]))
    a.logs.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "TANGO_HOST": f"{tango[0]}:{tango[1]}"}

    def stop(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    print(f"Quickstart: {a.stack} stack — {description}")
    with ProcessManager(name=f"quickstart_{a.port}") as manager:
        try:
            servers_log = a.logs / "servers.log"
            with servers_log.open("w") as fh:
                servers = manager.start_process("servers", "run_servers", [sys.executable, "-u",
                                                str(PROJECT_DIR / "startup_scripts" / "run_servers.py"),
                                                "--yaml", str(PROJECT_DIR / servers_yaml)],
                                                env=env, stdout=fh, stderr=subprocess.STDOUT)
            print(f"  device stack starting (log: {servers_log}) ...", flush=True)
            wait_for(servers.process, servers_log, ("All asyncroscopy servers are ready.",),
                     ("Startup failed",), timeout=600)
            print("  device stack ready")

            mcp_log = a.logs / "mcp.log"
            cfg = mcp_config(PROJECT_DIR / mcp_yaml, a.port, a.logs / "tool_calls.jsonl", tango)
            with mcp_log.open("w") as fh:
                mcp = manager.start_process("mcp", "run_mcp", [sys.executable, "-u",
                                            str(PROJECT_DIR / "startup_scripts" / "run_mcp.py"), "--yaml", str(cfg)],
                                            env=env, stdout=fh, stderr=subprocess.STDOUT)
            print(f"  MCP bridge starting (log: {mcp_log}) ...", flush=True)
            wait_for(mcp.process, mcp_log, ("Uvicorn running",), ("MCP ERROR", "Traceback"), timeout=300)
            base = f"http://127.0.0.1:{a.port}"
            print("  MCP bridge ready\n")

            from asyncroscopy.health.doctor import OK, INFO, run_checks
            checks = run_checks(tango[0], tango[1], mcp_url=base)
            bad = [c for c in checks if c.level not in (OK, INFO)]
            print(f"Doctor: {len(checks) - len(bad)}/{len(checks)} checks OK")
            for c in bad:
                print(f"  {c.level} {c.name}: {c.detail}\n      → {c.fix}")

            if a.demo:
                print("\nDemo: one autonomous session through MCP (as an agent would call it) ...", flush=True)
                out = demo(base + "/mcp")
                session = out["session"]
                if isinstance(session, dict):
                    print(f"  status {session['status']} in {session.get('duration_s')} s")
                    for s in session["stages"]:
                        print(f"    {s['stage']:12s} {s['status']}")
                    final = session.get("final") or {}
                    print(f"  final image {final.get('key')} at stage {final.get('stage')}")
                else:
                    print("  session failed:", session)
                (a.logs / "demo.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

            print(f"""
Ready.
  Agents (Claude Code, SciAgentGUI, notebooks):  {base}/mcp
  Tool-call log:  {a.logs / 'tool_calls.jsonl'}
  Health check:   uv run python -m asyncroscopy.health --mcp-url {base}
  Benchmarks:     uv run python -m asyncroscopy.bench baseline   (benchmark stack)
Press Ctrl+C to stop everything.""", flush=True)
            if a.exit_after_demo:
                return 0 if not bad else 1
            while servers.running and mcp.running:
                time.sleep(0.5)
            print("A component exited; see the logs in", a.logs)
            return 1
        except KeyboardInterrupt:
            print("\nStopping ...")
            return 0
        except Exception as exc:
            print(f"\nQuickstart failed: {exc}")
            return 1
        finally:
            # Ask the launchers to shut down their own children cleanly before the manager reaps them.
            for p in list(manager.active_processes):
                if p.running:
                    p.process.send_signal(signal.SIGTERM)
            deadline = time.monotonic() + 60
            while any(p.running for p in manager.active_processes) and time.monotonic() < deadline:
                time.sleep(0.5)


if __name__ == "__main__":
    raise SystemExit(main())
