"""What a simulated scanning-beam instrument shares, whatever it images: ``ScanningBeamTwin``.

A mixin for a twin device, ahead of its Tango base class in the bases, holding no Tango declarations of its own: the
simulated clock, the stage (whose model runs on the DigitalTwinSTAGE device, on this clock), the blanker, the image
shift, the scan of one frame with its dose, and the ground-truth file. ``_start_beam_twin`` makes that state; the twin
then sets ``_scan_params``, ``_detector_params``, ``_detector_drift``, ``_dose`` and ``_specimen``, and reads the
``sample_seed``, ``time_scale`` and ``acquisition_save_directory`` properties it declares.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np

from asyncroscopy.instruments.electron_microscope.twin import pixel_sample_coordinates
from asyncroscopy.instruments.electron_microscope.twin.detector import electrons_per_pixel
from asyncroscopy.instruments.electron_microscope.twin.scan_distortion import frame_time
from asyncroscopy.instruments.scanning_beam import check_scan
from asyncroscopy.sim.clock import SimClock
from asyncroscopy.sim.rng import EventRng
from asyncroscopy.sim.stage_model import StageTrajectory

ANGSTROM = 1e-10


class ScanningBeamTwin:
    """The clock, stage, blanker, image shift and frame scan of a simulated scanning-beam instrument."""

    def _start_beam_twin(self, seed: int) -> None:
        """The simulation's state at the start of a session."""
        self._rng = EventRng(int(seed))
        self._clock = SimClock(float(self.time_scale))
        self._acq_index = 0
        # Every acquisition's truth goes to a JSONL file as it is taken (export_ground_truth reads it back), so a
        # long session does not hold them all in memory; only the count is kept here.
        self._truth_path = Path(self.acquisition_save_directory) / "ground_truth" / f"{self.get_name().replace('/', '_')}_acquisitions.jsonl"
        self._truth_path.parent.mkdir(parents=True, exist_ok=True)
        self._truth_path.write_text("", encoding="utf-8")
        self._acquisition_count = 0
        # The stage model runs on the STAGE device (the session lets it start afresh when this device restarts);
        # this is what the device last said, at rest at the origin until it first answers.
        self._stage_session = uuid.uuid4().hex
        self._stage = StageTrajectory()
        self._blanked = False
        self._image_shift = np.zeros(2, dtype=np.float64)

    def _set_image_shift(self, shift) -> None:
        self._image_shift = np.asarray(shift, dtype=np.float64)

    def _blank_beam(self) -> None:
        """A blanked beam gives no signal and no dose, as on the instrument."""
        self._blanked = True

    def _unblank_beam(self) -> None:
        self._blanked = False

    def _beam_current_pa(self) -> float:
        return 0.0 if self._blanked else self._detector_params.beam_current_pa

    def _stage_request(self, **fields) -> str:
        """What the STAGE device needs to run its model on this microscope's clock."""
        return json.dumps({"session": self._stage_session, "seed": int(self.sample_seed), "clock_s": self._clock.now(), **fields})

    def _fetch_stage(self, window_s: float = 0.0) -> None:
        """Ask the STAGE device where the stage is (one call, however many pixels the frame has) and advance the clock
        by any move written to it directly; the drift it returns holds for the next ``window_s`` simulated seconds."""
        stage = self._detector_proxies.get("stage")
        if stage is None:
            return
        reply = json.loads(stage.frame_trajectory(self._stage_request(window_s=float(window_s))))
        if reply["moved_s"] > 0:
            self._clock.advance(reply["moved_s"])
        self._stage = StageTrajectory.from_dict(reply["trajectory"])
        self._stage_position = self._stage.true.copy()

    def _sync_stage_from_proxy(self) -> None:
        """A position written to the STAGE device directly counts as a stage command."""
        self._fetch_stage()

    def _stage_moving(self) -> bool:
        """True while a long stage move is under way; moves and acquisitions are refused until it is false."""
        stage = self._detector_proxies.get("stage")
        return False if stage is None else bool(stage.stage_moving)

    def instrument_clock_s(self) -> float:
        """clock_s is the simulated clock that drift, moves and frames run on, whatever time_scale is."""
        clock = getattr(self, "_clock", None)
        return clock.now() if clock is not None else 0.0

    def _refuse_while_moving(self) -> None:
        stage = self._detector_proxies.get("stage")
        left = 0.0 if stage is None else float(stage.moving_seconds_left)
        if left > 0:
            raise ValueError(f"the stage is still moving (about {left:.0f} s left); wait until stage_moving is false")

    def _move_stage(self, position):
        if len(position) != 5:
            raise ValueError("Stage position must have 5 values: [x, y, z, alpha, beta]")
        target = np.asarray(position, dtype=np.float64)
        if not np.all(np.isfinite(target)):
            raise ValueError(f"Stage position must be finite numbers, got {list(position)}")
        stage = self._detector_proxies.get("stage")
        if stage is None:
            raise RuntimeError(f"{type(self).__name__} cannot move a stage: set stage_device_address to a DigitalTwinSTAGE")
        # The stage checks the limits and refuses while a long move is under way. A move whose real wait would hold
        # the microscope longer than it may hold a command returns at once, with stage_moving true until it would have arrived.
        reply = json.loads(stage.move_on_clock(self._stage_request(position=target.tolist(), time_scale=self._clock.time_scale)))
        self._clock.advance(reply["duration_s"], wait=reply["wait"])
        self._stage = StageTrajectory.from_dict(reply["trajectory"])
        self._stage_position = self._stage.true.copy()

    def _get_stage(self):
        self._sync_stage_from_proxy()
        return self._stage.reported.copy()

    def _get_image_shift(self):
        return self._image_shift.copy()

    def _beam_current(self) -> dict:
        # Always current: nothing on the twin drifts it between readings.
        return {"pa": float(self._detector_params.beam_current_pa), "source": "measured", "age_s": 0.0}

    def _scan_frame(self, imsize: int, dwell_time: float, scan_region) -> dict:
        """Lay out one scan: pixel coordinates on the sample, timing, and the ground-truth record."""
        self._acq_index += 1
        index = self._acq_index
        # SCAN, AutoScript and the tilt twin all use [left, top, width, height] in fractions of the field.
        region = check_scan(imsize, dwell_time, scan_region)
        # One call for the whole frame; its drift holds for as long as the largest frame takes.
        self._fetch_stage(frame_time(int(imsize), int(imsize), float(dwell_time), self._scan_params))
        left, top, width, height = region
        t0 = self._clock.now()
        scan_truth: dict = {}
        xs, ys = pixel_sample_coordinates(
            int(imsize),
            float(self._fov),
            (left, top, min(1.0, left + width), min(1.0, top + height)),
            float(dwell_time),
            self._scan_params,
            self._beam_xy_at,
            t0,
            self._rng.for_event("scan", index),
            truth=scan_truth,
        )
        rows, cols = xs.shape
        duration = frame_time(rows, cols, float(dwell_time), self._scan_params)
        pixel = float(self._fov) / int(imsize)
        self._dose.add_frame(xs, ys, electrons_per_pixel(self._beam_current_pa(), dwell_time), pixel)
        self._clock.advance(duration)
        record = {
            "index": index,
            "kind": "image",
            "t_start_s": t0,
            "t_end_s": self._clock.now(),
            "commanded_m": self._stage.commanded.tolist(),
            "reported_m": self._stage.reported.tolist(),
            "true_m": self._stage.true.tolist(),
            "drift_start_m": self._stage.drift_offset_at(t0).tolist(),
            "drift_end_m": self._stage.drift_offset_at(self._clock.now()).tolist(),
            "fov_m": float(self._fov),
            "imsize": int(imsize),
            "scan_region": list(region),
            "dwell_s": float(dwell_time),
            "center_true_m": [float(xs.mean()), float(ys.mean())],
            # What bent and brightened this frame, for scoring stitching: the scan distortion, each line's jitter
            # and the scanning detectors' gain and offset when the frame started.
            "scan": {**vars(self._scan_params), **scan_truth},
            "detector": {k: v for k, v in vars(self._detector_drift.at(self._detector_params, t0)).items() if k in ("gain", "offset")},
        }
        self._record_acquisition(record)
        return {"xs": xs, "ys": ys, "index": index, "pixel_m": pixel, "dwell_s": float(dwell_time), "record": record}

    def _record_acquisition(self, record: dict) -> None:
        """Append one acquisition's truth to the ground-truth file."""
        with self._truth_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        self._acquisition_count += 1

    def _acquisition_records(self) -> list[dict]:
        """Every acquisition's truth since the sample was made, from the ground-truth file."""
        with self._truth_path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def _beam_xy_at(self, t: float) -> np.ndarray:
        """Where the scan is centred at sim time ``t``: the true stage position plus the image shift."""
        return np.asarray(self._stage.true_xy_at(t), dtype=np.float64) + self._image_shift

    def _support_angstrom(self, x: float, y: float) -> float:
        """Support film (the specimen's) plus contamination (the beam's doing) at a sample point, in angstrom."""
        return float(self._specimen.support_thickness_m) / ANGSTROM + self._dose.contamination_angstrom(x, y)
