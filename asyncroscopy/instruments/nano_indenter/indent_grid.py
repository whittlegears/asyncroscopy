"""IndentGrid: a grid of indents as one job, built on JobDevice.

Everything a long experiment needs comes from JobDevice: reserve (check the
plan, return an estimate), start (returns at once), job_status, abort,
results, release, a state file per job so a restarted device resumes, and an
Instrument Trace line per step. A team writes only the four hooks below.

The MCP server sees reserve ... release and treats this device as a job
device: reserve, job_status, results and release are low risk, start is the
one high-risk call (one approval per grid in approve mode), and abort is
always allowed.
"""

from __future__ import annotations

import json

import tango
from tango.server import device_property

from asyncroscopy.jobs import JobDevice

MAX_INDENTS = 400


class IndentGrid(JobDevice):
    JOB_KIND = "a grid of nanoindents with hardness at each point"
    PLAN_EXAMPLE = {"origin_m": [0.0, 0.0], "nx": 5, "ny": 5, "pitch_m": 2e-5, "load_n": 0.01}

    indenter_device_address = device_property(dtype=str, default_value="asyncroscopy/instrument/default", doc="Tango address of the indenter (the launcher names the instrument asyncroscopy/instrument/default).")

    def _indenter(self) -> tango.DeviceProxy:
        # proxy(), not tango.DeviceProxy: an indent takes longer than Tango's 3 s default timeout.
        return self.proxy(self.indenter_device_address)

    def resolve_plan(self, plan: dict) -> dict:
        """plan: {"origin_m": [x, y], "nx": int, "ny": int, "pitch_m": float, "load_n": float}."""
        nx, ny = int(plan.get("nx", 0)), int(plan.get("ny", 0))
        pitch, load = float(plan.get("pitch_m", 0)), float(plan.get("load_n", 0))
        x0, y0 = (float(v) for v in plan.get("origin_m", [0.0, 0.0]))
        if nx < 1 or ny < 1 or nx * ny > MAX_INDENTS:
            raise ValueError(f"nx and ny must be at least 1 and nx * ny at most {MAX_INDENTS}")
        if pitch <= 0 or load <= 0:
            raise ValueError("pitch_m and load_n must be positive")
        steps = [{"x": x0 + i * pitch, "y": y0 + j * pitch} for j in range(ny) for i in range(nx)]
        # Check against the instrument's own limits now, not halfway through the grid.
        limits = json.loads(self._indenter().get_limits())
        if load > limits["max_load_n"]:
            raise ValueError(f"load_n {load} is above the indenter's {limits['max_load_n']} N")
        far = max(max(abs(s["x"]), abs(s["y"])) for s in steps)
        if far > limits["travel_m"]:
            raise ValueError(f"the grid reaches {far:.3g} m from the origin; the stage travels {limits['travel_m']} m")
        # Everything except "steps" is kept as job["resolved_plan"] and returned by reserve.
        return {
            "steps": steps,
            "origin_m": [x0, y0], "nx": nx, "ny": ny, "pitch_m": pitch, "load_n": load,
            "estimate": {"indents": len(steps), "duration_s": 20.0 * len(steps)},
        }

    def run_step(self, job: dict, index: int) -> dict:
        step = job["steps"][index]
        indenter = self._indenter()
        indenter.move_xy([step["x"], step["y"]])
        result = json.loads(indenter.indent(job["resolved_plan"]["load_n"]))
        return {"index": index, "x_m": step["x"], "y_m": step["y"], **result}

    def on_stop(self, job: dict, reason: str) -> None:
        # Abort or failure: leave the tip off the sample.
        self._indenter().retract()

    def on_finish(self, job: dict) -> dict:
        self._indenter().retract()
        return {}  # keys of anything saved to Tiled would go here

    def summary(self, job: dict) -> dict:
        # Small enough to return whole; a real device would save to Tiled and return keys.
        indents = [job["step_results"][k] for k in sorted(job["step_results"], key=int)]
        return {"indents": [{k: r[k] for k in ("x_m", "y_m", "hardness_gpa")} for r in indents]}


if __name__ == "__main__":
    IndentGrid.run_server()
