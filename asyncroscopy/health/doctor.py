"""Doctor: is the stack healthy, and if not, what exactly to do.

Turns the symptom list in ``docs/Operation/troubleshooting.md`` into checks:

    uv run python -m asyncroscopy.health                          # read-only checklist
    uv run python -m asyncroscopy.health --fix                    # also reconnect/Init faulted devices
    uv run python -m asyncroscopy.health --fix --allow-restart    # ...and restart their device servers
    uv run python -m asyncroscopy.health --mcp-url http://localhost:8000 --json

Exit code 0 when everything is OK, 1 when something needs attention.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import urllib.request
from dataclasses import asdict, dataclass

OK, WARN, FAIL, INFO = "OK", "WARN", "FAIL", "INFO"


@dataclass
class Check:
    level: str
    name: str
    detail: str = ""
    fix: str = ""


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _http_ok(url: str, timeout: float = 3.0) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:   # noqa: S310 (operator-supplied local URL)
            return r.status < 500, f"HTTP {r.status}"
    except urllib.error.HTTPError as exc:
        return exc.code < 500, f"HTTP {exc.code}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def run_checks(tango_host: str, tango_port: int, *, fix: bool = False, allow_restart: bool = False,
               mcp_url: str | None = None) -> list[Check]:
    os.environ["TANGO_HOST"] = f"{tango_host}:{tango_port}"
    checks: list[Check] = []

    if not _port_open(tango_host, tango_port):
        checks.append(Check(FAIL, "Tango database", f"nothing listening on {tango_host}:{tango_port}",
                            "Start the stack: uv run startup_scripts/run_servers.py --yaml <config>. If it fails "
                            "to start, answer 'yes' to clearing old processes, or delete stale .db files."))
        return checks

    import tango

    from asyncroscopy.health.monitor import DEFAULT_ALLOWED, HealthMonitor, recover

    try:
        db = tango.Database(tango_host, tango_port)
        checks.append(Check(OK, "Tango database", f"{tango_host}:{tango_port}"))
    except Exception as exc:
        checks.append(Check(FAIL, "Tango database", str(exc).split("\n")[0],
                            "The port is open but it is not a Tango database: another program (or a stale "
                            "server) holds it. Clear old processes and restart."))
        return checks

    monitor = HealthMonitor(database=db)
    try:
        devices = monitor.devices()
    except Exception as exc:
        checks.append(Check(FAIL, "device list", str(exc)))
        return checks
    if not devices:
        checks.append(Check(WARN, "devices", "the database lists no exported devices",
                            "Start the device servers with run_servers.py."))
    allowed = DEFAULT_ALLOWED + (("restart_server",) if allow_restart else ())
    data_device = None
    for name in devices:
        h = monitor.check(name)
        if h.dev_class == "DATA":
            data_device = name
        if h.healthy:
            checks.append(Check(OK, name, f"{h.state} · {h.latency_ms} ms · {h.dev_class}"))
            continue
        if fix:
            result = recover(monitor, name, allowed=allowed)
            if result["status"] == "recovered":
                checks.append(Check(OK, name, f"was {h.problem}; recovered by {result['by']}"))
                continue
            checks.append(Check(FAIL, name, f"{h.problem}: {h.detail}",
                                result.get("message", "") or "recovery did not bring it back"))
            continue
        hint = ("rerun with --fix to try reconnect/Init" if h.reachable
                else f"its server ({h.server or 'unknown'}) is not running; restart the stack or that server")
        checks.append(Check(FAIL if h.problem in ("unreachable", "fault") else WARN, name,
                            f"{h.problem}: {h.detail}", hint))

    if data_device:
        try:
            config = json.loads(tango.DeviceProxy(data_device).get_config())
            uri = config.get("uri")
            ok, detail = _http_ok(uri.rstrip("/") + "/api/v1/") if uri else (False, "no uri in DATA config")
            checks.append(Check(OK if ok else FAIL, "Tiled server", f"{uri} · {detail}",
                                "" if ok else "Acquisitions will not be readable. A stale Tiled may be holding "
                                              "the port: stop it and let run_servers.py start Tiled on the "
                                              "current save path."))
            save = config.get("save_path")
            if save:
                exists = os.path.isdir(os.path.expanduser(save))
                checks.append(Check(OK if exists else WARN, "save path", save,
                                    "" if exists else "The DATA save path does not exist on this machine "
                                                      "(fine if the DATA device runs elsewhere)."))
        except Exception as exc:
            checks.append(Check(WARN, "Tiled server", f"could not read DATA config: {exc}"))
    else:
        checks.append(Check(INFO, "Tiled server", "no DATA device found; skipped"))

    if mcp_url:
        base = mcp_url.rstrip("/")
        port_ok, detail = _http_ok(base + "/mcp")
        checks.append(Check(OK if port_ok else FAIL, "MCP server", f"{base}/mcp · {detail}",
                            "" if port_ok else "Start it with uv run startup_scripts/run_mcp.py --yaml "
                                               "configs/mcp_dt.yaml."))
    return checks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    host, _, port = os.environ.get("TANGO_HOST", "localhost:9094").partition(":")
    ap.add_argument("--tango-host", default=host or "localhost")
    ap.add_argument("--tango-port", type=int, default=int(port or 9094))
    ap.add_argument("--fix", action="store_true", help="reconnect/Init devices that are faulted")
    ap.add_argument("--allow-restart", action="store_true", help="with --fix, also restart device servers")
    ap.add_argument("--mcp-url", default=None, help="also check an MCP server, e.g. http://localhost:8000")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    checks = run_checks(a.tango_host, a.tango_port, fix=a.fix, allow_restart=a.allow_restart, mcp_url=a.mcp_url)
    if a.json:
        print(json.dumps([asdict(c) for c in checks], indent=2))
    else:
        for c in checks:
            print(f"{c.level:4s}  {c.name:38s} {c.detail}")
            if c.fix:
                print(f"      → {c.fix}")
    return 0 if all(c.level in (OK, INFO) for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
