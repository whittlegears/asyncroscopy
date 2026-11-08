"""The digital twin's STAGE device: it runs the stage model.

Commanded, true and reported positions, bias, Gaussian landing error, backlash on a reversal, settling, thermal
drift and whether a long move is still under way all live here (``asyncroscopy.sim.stage_model``), as the
corrector twin holds its aberrations. The twin microscope orchestrates through its DeviceProxy:

* ``move_on_clock`` moves the stage at the microscope's simulated time and returns how long the move takes, so the
  microscope advances its clock, the single time source;
* ``frame_trajectory`` hands over the positions and a compact drift description, once per frame, so rendering each
  line from the true position does not take a Tango call per line.

A position written directly to ``position`` or an axis attribute counts as a stage command: the microscope notices it
on its next ``frame_trajectory`` and moves the model at that time, as a real controller would have moved the stage.
Both commands reveal where the stage really is, so they are operator only. ``position_sigma_m`` is worked out from the
stage model: a Mapper whose ``stage_sigma_m`` is not set reads it, so a twin configuration never has to derive the
stitching bound by hand.

Twin microscopes that do not model the stage (the atomistic twins) just write ``position``.
"""

from __future__ import annotations

import json
import math
import time

import numpy as np
from tango import AttrWriteType
from tango.server import attribute, device_property

from asyncroscopy.instruments.electron_microscope.hardware.stage import STAGE
from asyncroscopy.sim.rng import EventRng
from asyncroscopy.sim.stage_model import StageModel, StageParams
from asyncroscopy.utils.commands import command

MOVE_BLOCK_S = 2.0  # longest real time one command may wait for a stage move


def _prop(dtype, default, doc):
    return device_property(dtype=dtype, default_value=default, doc=doc)


class DigitalTwinSTAGE(STAGE):
    """The STAGE device of a simulated stack: limits and holders as on any STAGE, plus the stage's own errors and dynamics."""

    stage_move_noise_std = _prop(float, 0.0, "Gaussian landing error (m) on x, y and z.")
    stage_bias_x = _prop(float, 0.0, "Systematic x landing error (m).")
    stage_bias_y = _prop(float, 0.0, "Systematic y landing error (m).")
    stage_reported_noise_std = _prop(float, 0.0, "Stage readback error (m).")
    stage_backlash_x = _prop(float, 0.0, "x shortfall when the move direction reverses (m).")
    stage_backlash_y = _prop(float, 0.0, "y shortfall when the move direction reverses (m).")
    stage_speed = _prop(float, 5e-6, "Stage speed (m/s), sets move duration.")
    stage_move_overhead = _prop(float, 0.0, "Fixed time per stage move (s).")
    stage_settle_drift = _prop(float, 0.0, "Drift speed right after a move (m/s), decays with stage_settle_tau.")
    stage_settle_tau = _prop(float, 30.0, "Settling time constant (s).")
    drift_rate_per_min = _prop(float, 0.0, "Thermal drift (m/min).")
    drift_random_walk = _prop(float, 0.0, "Drift random-walk step per minute (m).")

    # Declared again because Tango binds an inherited attribute to the base class's read method.
    position_sigma_m = attribute(dtype=float, unit="m", access=AttrWriteType.READ, doc="1-sigma distance between where the stage lands and where it reports (m), from the stage model (landing noise, backlash, readback noise).")
    stage_moving = attribute(dtype=bool, access=AttrWriteType.READ, doc="True while a long stage move is under way.")
    moving_seconds_left = attribute(dtype=float, unit="s", access=AttrWriteType.READ, doc="Real seconds until the current long move ends; 0 when none is under way.")

    def init_device(self) -> None:
        super().init_device()
        self._model: StageModel | None = None
        self._session: str | None = None
        self._pending: list[float] | None = None  # a position written directly, not yet run through the model
        self._moving_until = 0.0  # monotonic time a long move ends

    def _params(self) -> StageParams:
        return StageParams(
            bias_m=(float(self.stage_bias_x), float(self.stage_bias_y)),
            noise_std_m=float(self.stage_move_noise_std),
            reported_noise_std_m=float(self.stage_reported_noise_std),
            backlash_m=(float(self.stage_backlash_x), float(self.stage_backlash_y)),
            speed_m_s=float(self.stage_speed),
            move_overhead_s=float(self.stage_move_overhead),
            settle_drift_m_s=float(self.stage_settle_drift),
            settle_tau_s=float(self.stage_settle_tau),
            drift_m_per_min=float(self.drift_rate_per_min),
            drift_random_walk_m=float(self.drift_random_walk),
        )

    # ------------------------------------------------------------------
    # The position every STAGE client sees
    # ------------------------------------------------------------------
    def _read_position(self) -> list[float]:
        return list(self._position)

    def _write_position(self, value) -> None:
        position = [float(component) for component in value]
        if len(position) != 5:
            raise ValueError("Stage position must be [x, y, z, alpha, beta]")
        self._position = position
        settled = self._model.reported if self._model is not None else np.zeros(5)
        self._pending = None if np.allclose(position, settled, rtol=0.0, atol=1e-15) else position

    def _read_beta_tilt_enabled(self) -> bool:
        return True

    def read_stage_moving(self) -> bool:
        return time.monotonic() < self._moving_until

    def read_moving_seconds_left(self) -> float:
        return max(0.0, self._moving_until - time.monotonic())

    def read_position_sigma_m(self) -> float:
        """How far the stage lands from where it reports, 1 sigma: the landing noise, the backlash on a reversal and the
        readback noise together (the bias is the same for every tile, so it does not separate neighbours)."""
        backlash = max(abs(float(self.stage_backlash_x)), abs(float(self.stage_backlash_y)))
        modelled = math.sqrt(float(self.stage_move_noise_std) ** 2 + backlash**2 + float(self.stage_reported_noise_std) ** 2)
        return modelled if modelled > 0 else float(self.position_error_m)

    # ------------------------------------------------------------------
    # The stage model, driven by the microscope's simulated clock
    # ------------------------------------------------------------------
    def _model_for(self, request: dict) -> StageModel:
        """The model of the microscope's session: a microscope that restarted (a new session) starts a fresh model."""
        if self._model is None or self._session != request["session"]:
            self._session = str(request["session"])
            self._model = StageModel(self._params(), EventRng(int(request["seed"])))
            self._pending = None if np.allclose(self._position, 0.0, rtol=0.0, atol=1e-15) else list(self._position)
        return self._model

    def _apply_pending(self, model: StageModel, clock_s: float) -> float:
        """Run a position written directly through the model at ``clock_s``; returns the move's simulated seconds."""
        if self._pending is None:
            return 0.0
        target, self._pending = self._pending, None
        duration = model.move(target, clock_s)
        self._position = model.reported.tolist()
        return duration

    @command(dtype_in=str, dtype_out=str, doc_in=':param request: JSON {"session", "seed", "clock_s", "position": [x, y, z, alpha, beta], "time_scale"}.', risk="high", operator_only=True, role="stage.move")
    def move_on_clock(self, request: str) -> str:
        """Move the stage at the microscope's simulated time. Returns JSON {"duration_s", "wait", "trajectory"}; ``wait`` is
        false when the move would hold the caller too long, which then reads stage_moving until it would have arrived."""
        body = json.loads(request)
        target = [float(component) for component in body["position"]]
        if getattr(self, "_frozen", None):
            raise ValueError(f"the stage did not move: {self._frozen}")
        self._check_limits(target)
        left = self.read_moving_seconds_left()
        if left > 0:
            raise ValueError(f"the stage is still moving (about {left:.0f} s left); wait until stage_moving is false")
        model = self._model_for(body)
        clock_s = float(body["clock_s"])
        pending_s = self._apply_pending(model, clock_s)  # a direct write the microscope has not picked up yet
        start_s = clock_s + pending_s
        real = model.move_seconds(target) * max(0.0, float(body.get("time_scale", 0.0)))
        duration = model.move(target, start_s)
        wait = real <= MOVE_BLOCK_S
        if not wait:
            self._moving_until = time.monotonic() + real
        self._position = model.reported.tolist()
        return json.dumps({"duration_s": pending_s + duration, "wait": wait, "trajectory": model.trajectory(start_s + duration).to_dict()})

    @command(dtype_in=str, dtype_out=str, doc_in=':param request: JSON {"session", "seed", "clock_s", "window_s"}.', risk="high", operator_only=True, role="other")
    def frame_trajectory(self, request: str) -> str:
        """Take up any position written directly, at the microscope's simulated time. Returns JSON {"moved_s", "trajectory"}:
        the seconds that move took (the microscope advances its clock by them) and the stage's positions and drift for the next ``window_s`` seconds."""
        body = json.loads(request)
        model = self._model_for(body)
        clock_s = float(body["clock_s"])
        moved_s = self._apply_pending(model, clock_s)
        start_s = clock_s + moved_s
        return json.dumps({"moved_s": moved_s, "trajectory": model.trajectory(start_s, start_s + float(body.get("window_s", 0.0))).to_dict()})

    @command(dtype_out=str, risk="low", operator_only=True, role="data.read")
    def stage_moves(self) -> str:
        """Every move since the microscope's session began, for export_ground_truth: JSON [{"index", "time_s", "commanded", "true", "reported"}]."""
        return json.dumps([vars(move) for move in (self._model.history if self._model is not None else [])])


if __name__ == "__main__":
    DigitalTwinSTAGE.run_server()
