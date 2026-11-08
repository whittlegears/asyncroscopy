"""The SPM map profile: the Mapper tiles any ``SPMMicroscope`` by scanning at each stage position.

Around every stage move the tip is lifted and brought back: ``retract`` before the move (``SPMMicroscope.move_stage``
refuses while it is engaged), ``approach`` after it, and each frame is taken through the SCAN device
(``acquire_scan``), whose frame settings the profile writes. The tip is retracted again as soon as the frame is done,
and whenever the job stops. A move is only as fast as the instrument's own ``move_stage``, so the job can stop between
tiles but not in the middle of one.

Its Mapper setting is ``scan_device_address``; the stage is read from the Mapper's ``stage_device_address``. The only
budget is the Mapper's ``time_s``: a setpoint's meaning depends on the SPM mode, so no force budget is invented here.
A plan cannot ask for live drift tracking, because an SPM has no image shift.

The Mapper reads each frame back as an acquisition, so the backend's scan must return one a reader understands (HDF5,
or a key DATA serves); anything else ends the tile with an error that says so. The simulated backend (simulated.py)
saves HDF5. The Jupiter backend does not yet: its scan returns the path of Asylum's .ibw file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import tango

from asyncroscopy.core.names import REGISTRABLE_SUFFIXES
from asyncroscopy.jobs.mapping.profile import InstrumentProfile, Setting, register_profile

# One frame takes about lines / rate seconds; twice that and a minute more for a slow start (as the Jupiter backend allows).
SCAN_TIMEOUT_MARGIN_S = 60.0


@register_profile
class SPMProfile(InstrumentProfile):
    name = "spm"
    doc = "SPM: a scan per tile through the SCAN device (imsize, scan_rate_hz), the tip retracted for every stage move and approached after it; budget time_s."
    job_kind = "map acquisition: a grid of probe scans over a stage box, the tip lifted for every move, stitched into a zoomable map when it finishes"
    plan_example = {"size_m": [3e-6, 3e-6], "fov_m": 1e-6, "overlap": 0.25}
    plan_fields = frozenset({"imsize", "scan_rate_hz"})
    plan_defaults = {"imsize": 256, "scan_rate_hz": 1.0}
    image_fields = frozenset({"imsize", "scan_rate_hz"})
    settings = {"scan_device_address": Setting(str, "asyncroscopy/scan/default", "SPM SCAN device (frame settings and acquire_scan).")}
    needs = {"microscope_device_address": ("retract", "approach", "move_stage"), "scan_device_address": ("acquire_scan",)}

    # -- the devices -----------------------------------------------------
    def _scan(self, lines: int | None = None, rate_hz: float | None = None) -> tango.DeviceProxy:
        """The SCAN device, with a timeout that outlasts a frame of ``lines`` at ``rate_hz`` (the default one to set it up)."""
        if lines is None or rate_hz is None:
            return self.ctx.proxy(self.setting("scan_device_address"))
        return self.ctx.proxy(self.setting("scan_device_address"), timeout_s=2.0 * int(lines) / float(rate_hz) + SCAN_TIMEOUT_MARGIN_S)

    def _stage(self) -> tango.DeviceProxy:
        address = self.ctx.settings.stage_device_address
        if not address:
            raise ValueError("the 'spm' instrument profile reads the stage from the Mapper's stage_device_address, which is blank")
        return self.ctx.proxy(address)

    # -- plan ------------------------------------------------------------
    def resolve(self, p: dict) -> None:
        found = str(self.instrument().read_attribute("instrument_type").value)
        if found != "SPM":
            raise ValueError(f"the 'spm' instrument profile drives SPM instruments; {self.ctx.settings.microscope_device_address} is a {found}")
        if not (int(p["imsize"]) >= 2 and float(p["scan_rate_hz"]) > 0):
            raise ValueError("imsize must be >= 2 and scan_rate_hz > 0")

    def current_xy(self) -> tuple[float, float]:
        stage = self._stage()
        return float(stage.stage_x_m), float(stage.stage_y_m)

    def resolve_revisit(self, p: dict, revisit: dict) -> dict:
        return {"fov_m": float(revisit["fov_m"]), "imsize": int(revisit.get("imsize", int(p["imsize"]))), "scan_rate_hz": float(revisit.get("scan_rate_hz", float(p["scan_rate_hz"])))}

    def tile_seconds(self, p: dict) -> float:
        return int(p["imsize"]) / float(p["scan_rate_hz"])

    def revisit_seconds(self, revisit: dict) -> float:
        return int(revisit["imsize"]) / float(revisit["scan_rate_hz"])

    def move_target(self, plan: dict, name: str, arguments: Any) -> dict | None:
        if name != "move_stage":
            return None
        x, y = (float(v) for v in list(arguments[0])[:2])
        return {"x": x, "y": y}

    # -- run -------------------------------------------------------------
    def _set_frame(self, size_m: float, lines: int, rate_hz: float) -> None:
        scan = self._scan()
        scan.scan_size_m = float(size_m)
        scan.scan_size_px = int(lines)
        scan.scan_rate_hz = float(rate_hz)

    def setup(self, job: dict) -> None:
        plan = job["resolved_plan"]
        self._set_frame(float(plan["fov_m"]), int(plan["imsize"]), float(plan["scan_rate_hz"]))
        scan = self._scan()
        scan.x_scan_center_m = 0.0  # the frame sits on the stage position the tile records
        scan.y_scan_center_m = 0.0
        self.make_safe(job)

    def make_safe(self, job: dict) -> None:
        self.instrument().retract()

    def before_move(self, job: dict) -> None:
        self.instrument().retract()

    def move(self, job: dict, x: float, y: float) -> bool:
        self.instrument().move_stage([float(x), float(y)])
        return not self.ctx.aborting(job)

    def after_move(self, job: dict) -> None:
        if self.instrument().approach() is False:
            raise RuntimeError("the tip did not engage after the stage move; the tile was not taken")

    def stage_reported(self, job: dict) -> list[float]:
        stage = self._stage()
        return [float(stage.stage_x_m), float(stage.stage_y_m)]

    def acquire_tile(self, job: dict, tile: dict) -> dict | None:
        plan = job["resolved_plan"]
        started = self.ctx.clock_s()
        try:
            key = self._scan(int(plan["imsize"]), float(plan["scan_rate_hz"])).acquire_scan()
            if not job.get("dry_run") and Path(key).suffix.lower() not in REGISTRABLE_SUFFIXES:
                raise RuntimeError(f"the scan returned {key!r}, a file the Mapper cannot read back: the SCAN device must save its frames as an acquisition (HDF5 or a DATA key)")
            return {"image_key": key, "acquired": (started + self.ctx.clock_s()) / 2.0}
        finally:
            self.instrument().retract()

    def acquire_revisit(self, job: dict, revisit: dict) -> str:
        plan = job["resolved_plan"]
        self._set_frame(float(revisit["fov_m"]), int(revisit["imsize"]), float(revisit["scan_rate_hz"]))
        try:
            return self._scan(int(revisit["imsize"]), float(revisit["scan_rate_hz"])).acquire_scan()
        finally:
            self.instrument().retract()
            self._set_frame(float(plan["fov_m"]), int(plan["imsize"]), float(plan["scan_rate_hz"]))
