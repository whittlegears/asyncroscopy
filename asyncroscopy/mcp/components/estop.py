"""stop_all: the emergency stop tool. The stop itself is ``asyncroscopy.safety.estop``; this picks each
device's stops from the tool descriptors (action role or declared stop) and reports what is left."""

from __future__ import annotations

import json
import time
import warnings
from typing import TYPE_CHECKING, Any

import tango
from fastmcp.tools import tool

from asyncroscopy.mcp.interceptors import describe_error
from asyncroscopy.safety import estop

if TYPE_CHECKING:
    from asyncroscopy.mcp.mcp_server import MCPServer, ServerConfig


def _failed(exc: Exception) -> str:
    return describe_error(exc) if isinstance(exc, tango.DevFailed) else str(exc)[:500]


class Estop:
    def __init__(self, server: MCPServer, cfg: ServerConfig) -> None:
        self.server = server

    def natives(self) -> list:
        return [] if self.server.endpoint_role == "monitor" else [self.stop_all]

    def targets(self) -> tuple[list[estop.StopTarget], list[dict]]:
        """(each device's stops, the steps already failed: a beam, job or probe device with no stop for it here)."""
        server = self.server
        targets: list[estop.StopTarget] = []
        beams: list[dict] = []
        jobs: list[dict] = []
        probes: list[dict] = []
        for prefix in sorted(server.devices):
            device = server.devices[prefix]
            if device.proxy is None:
                continue
            target = estop.StopTarget(prefix, device.proxy)
            for command in sorted(server.tools.get(prefix, {})):
                tool = server.descriptors.get(f"{prefix}_{command}")
                if tool is None or not tool.void:
                    continue
                if tool.action_role == "beam.blank":
                    target.energy.append(command)
                elif command == estop.ABORT_ALL and device.envelope:
                    target.other.append(command)
                elif tool.action_role == "probe.retract" and device.probe:
                    target.other.append(command)
                    target.retracts.add(command)
                elif tool.stop:
                    target.other.append(command)
            targets.append(target)
            if device.stem and not target.energy:
                beams.append({"tool": f"{prefix}_blank_beam", "ok": False, "error": f"{prefix} offers no beam blank here (blocked or missing); blank it by hand"})
            if device.envelope and estop.ABORT_ALL not in target.other:
                jobs.append({"tool": f"{prefix}_abort_all", "ok": False, "error": f"{prefix} offers no abort_all here, so its jobs were not stopped; abort them by job_id"})
            if device.probe and not target.retracts:
                probes.append({"tool": f"{prefix}_retract", "ok": False, "error": f"{prefix} offers no retract here; lift the probe by hand"})
        missing = beams + jobs + probes
        return targets, missing

    @tool()
    def stop_all(self) -> dict[str, Any]:
        """Emergency stop. First puts the approval mode in approve and voids every unused approval, so nothing starts
        again without a person; then, on every device at once: blanks beams, closes shutters, switches lasers, lamps
        and X-rays off, aborts running jobs and lifts probes; then blanks and closes once more. Never waits for
        approval, and never switches off high tension, a gun or a filament (blanking is enough to stop, and those take
        hours to recover). A device that cannot be stopped is reported, not skipped. A stage move already under way
        is not stopped (there is no stage stop to call) and is listed in "not_stopped". Returns what each step did."""
        server = self.server
        latch = None
        if server.gate.address is not None:
            def latch(command: str, body: dict) -> None:
                getattr(server.gate.approvals(), command)(json.dumps(body))
        targets, missing = self.targets()
        report = estop.stop_all(targets, latch=latch, who=f"stop_all ({server.caller() or 'unknown caller'})", missing=missing,
                                describe=_failed, wait_s=estop.STOP_ALL_WAIT_S)
        # No stage stop exists to call: AutoScript 1.18's stage (CompuStage) has none, only stop_jogging, and its
        # absolute moves run to the end; a twin's move does too. Say so, so "stopped" is not read as the stage too.
        not_stopped = [
            f"{prefix}: a stage move already under way is not stopped (no stage stop exists; AutoScript 1.18 has none); "
            "stop it at the microscope's stage controls if it must not finish"
            for prefix, device in sorted(server.devices.items())
            if device.stem and any(t.action_role == "stage.move" and t.device == prefix for t in server.descriptors.values())
        ]
        if server.trace is not None:
            try:
                server.trace.record("stop_all", all(s["ok"] for s in report), time.time(), caller=server.caller(), run=server.run_id() or None,
                                    arguments={"steps": [s["tool"] for s in report]})
            except Exception as exc:
                warnings.warn(f"stop_all: trace event not written: {type(exc).__name__}: {exc}", RuntimeWarning, stacklevel=2)
        reached = any(t.energy or t.other for t in targets)
        return {"stopped": bool(report) and all(s["ok"] for s in report), "steps": report,
                **({"not_stopped": not_stopped} if not_stopped else {}),
                **({} if reached else {"note": "this endpoint reaches no beam, job or probe to stop"})}


def register(server: MCPServer, cfg: ServerConfig) -> Estop:
    return Estop(server, cfg)
