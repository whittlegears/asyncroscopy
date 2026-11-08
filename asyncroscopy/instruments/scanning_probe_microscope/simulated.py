"""A simulated scanning probe microscope: the ``_hw_*`` hooks of the SPM devices over a seeded height field.

The devices talk to each other over Tango as the real ones do: the SCAN device asks the STAGE where it is and the
APPROACH device whether the tip is down, and refuses to scan with the tip up. Frames are saved as HDF5 acquisitions
(``height`` in metres, rows along +y, columns along +x), which the Mapper reads back. No instrument API is involved.
"""

from __future__ import annotations

import numpy as np
import tango
from tango.server import device_property

from asyncroscopy.data.data_writer import DEFAULT_ACQUISITION_DIR, save_acquisition
from asyncroscopy.instruments.scanning_probe_microscope.hardware.spm_approach import SPM_APPROACH
from asyncroscopy.instruments.scanning_probe_microscope.hardware.spm_feedback import SPM_FEEDBACK
from asyncroscopy.instruments.scanning_probe_microscope.hardware.spm_scan import SPM_SCAN
from asyncroscopy.instruments.scanning_probe_microscope.hardware.spm_stage import SPM_STAGE
from asyncroscopy.instruments.scanning_probe_microscope.scanning_probe_microscope import SPMMicroscope, SPMMode

BUMPS = 1500  # features on the simulated surface
SURFACE_HALF_M = 1e-5
BUMP_RADIUS_M = (5e-8, 2.5e-7)
BUMP_HEIGHT_M = (5e-9, 4e-8)


def height_field(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bump centres [n, 2], radii and heights of the seeded surface."""
    rng = np.random.default_rng(int(seed))
    return rng.uniform(-SURFACE_HALF_M, SURFACE_HALF_M, size=(BUMPS, 2)), rng.uniform(*BUMP_RADIUS_M, size=BUMPS), rng.uniform(*BUMP_HEIGHT_M, size=BUMPS)


class SimulatedSPM(SPMMicroscope):
    """The top-level device: everything but its mode and meters is the sub-devices'."""

    def _connect_hardware(self) -> None:
        """Nothing to reach."""

    def _hw_get_spm_mode(self) -> SPMMode:
        return SPMMode.CONTACT_AFM

    def _hw_get_meter_values(self) -> dict:
        return {"sum": 0.0, "deflection": 0.0, "lateral": 0.0, "z": 0.0}


class SimulatedSPMApproach(SPM_APPROACH):
    """The tip: engaged or not, and how often it was moved."""

    def init_device(self) -> None:
        self._approached = False
        self.approaches = 0
        self.retracts = 0
        super().init_device()

    def _hw_approach(self) -> None:
        self._approached = True
        self.approaches += 1

    def _hw_retract(self) -> None:
        self._approached = False
        self.retracts += 1

    def _hw_stop(self) -> None:
        """Nothing is ever running."""

    def _hw_is_approached(self) -> bool:
        return self._approached


class SimulatedSPMFeedback(SPM_FEEDBACK):
    """The Z loop's settings, kept."""

    def init_device(self) -> None:
        self._params = {"setpoint": 0.5, "i_gain": 10.0}
        self._loop_on = False
        super().init_device()

    def _hw_read_feedback_params(self) -> dict:
        return dict(self._params)

    def _hw_write_feedback_param(self, name: str, value) -> None:
        self._params[name] = float(value)

    def _hw_feedback_on(self) -> None:
        self._loop_on = True

    def _hw_feedback_off(self) -> None:
        self._loop_on = False

    def _hw_is_feedback_on(self) -> bool:
        return self._loop_on


class SimulatedSPMStage(SPM_STAGE):
    """The coarse XY stage, exact and immediate."""

    def init_device(self) -> None:
        self._xy = [0.0, 0.0]
        super().init_device()

    def _hw_read_stage_position(self) -> list[float]:
        return list(self._xy)

    def _hw_move_stage_relative(self, dx: float, dy: float) -> None:
        self._xy = [self._xy[0] + float(dx), self._xy[1] + float(dy)]

    def _hw_stop(self) -> None:
        """Nothing is ever running."""


class SimulatedSPMScan(SPM_SCAN):
    """Frames of the height field under the stage position, one HDF5 acquisition each."""

    stage_device_address = device_property(dtype=str, doc="SPM STAGE device: where the frame is.")
    approach_device_address = device_property(dtype=str, doc="SPM APPROACH device: a frame needs the tip down.")
    sample_device_address = device_property(dtype=str, default_value="", doc="SAMPLE device each frame records.")
    acquisition_save_directory = device_property(dtype=str, default_value=DEFAULT_ACQUISITION_DIR, doc="Directory where frames are saved when there is no DATA device.")
    sample_seed = device_property(dtype=int, default_value=7, doc="Seed of the simulated surface.")

    def init_device(self) -> None:
        self._params = {"x_scan_center_m": 0.0, "y_scan_center_m": 0.0, "scan_size_m": 1e-6, "scan_size_px": 64, "scan_angle_deg": 0.0, "scan_rate_hz": 1.0}
        self._probe = [0.0, 0.0]
        super().init_device()
        self._surface = height_field(int(self.sample_seed))  # device properties are read in the base's init_device
        self._detector_proxies = {"sample": tango.DeviceProxy(self.sample_device_address)} if self.sample_device_address else {}

    def _hw_read_scan_params(self) -> dict:
        return dict(self._params)

    def _hw_write_scan_param(self, name: str, value) -> None:
        self._params[name] = value

    def _height(self) -> np.ndarray:
        centres, radii, heights = self._surface
        stage = tango.DeviceProxy(self.stage_device_address)
        size, lines = float(self._params["scan_size_m"]), int(self._params["scan_size_px"])
        cx, cy = float(stage.stage_x_m) + float(self._params["x_scan_center_m"]), float(stage.stage_y_m) + float(self._params["y_scan_center_m"])
        axis = (np.arange(lines) + 0.5) / lines - 0.5
        xs, ys = np.meshgrid(cx + axis * size, cy + axis * size)
        near = np.max(np.abs(centres - [cx, cy]), axis=1) < size / 2 + 3 * radii
        field = np.zeros((lines, lines))
        for (bx, by), radius, height in zip(centres[near], radii[near], heights[near]):
            field += height * np.exp(-((xs - bx) ** 2 + (ys - by) ** 2) / (2 * radius**2))
        return field

    def _hw_acquire_scan(self) -> str:
        if not bool(tango.DeviceProxy(self.approach_device_address).approached):
            raise RuntimeError("a frame needs the tip on the surface: approach first")
        pixel = float(self._params["scan_size_m"]) / int(self._params["scan_size_px"])
        attrs = {"units": "m", "pixel_size_m": pixel, "fov_m": float(self._params["scan_size_m"]), "scan_rate_hz": float(self._params["scan_rate_hz"]), "axes": "rows=+y, columns=+x"}
        return save_acquisition(self, self._data_proxy, "spm_image", "height", self._height(), dataset_name="image", dataset_attrs=attrs)

    def _hw_stop_scan(self) -> None:
        """Frames are immediate."""

    def _hw_read_probe_position(self) -> list[float]:
        return list(self._probe)

    def _hw_move_probe(self, x: float, y: float) -> None:
        self._probe = [float(x), float(y)]

