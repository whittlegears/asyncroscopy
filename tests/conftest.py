"""
Shared pytest fixtures for Tango device tests.

Starts BOTH the detector device(s) and the STEMMicroscope device in ONE Tango
test device server using MultiDeviceTestContext, so the STEMMicroscope can
create DeviceProxy connections to detectors by device name.

This avoids:
- "No proxy found for detector 'scan'. Available: []"
- Needing a real Tango DB
- Flaky multi-context issues from spinning up multiple separate servers
"""

import os

import pytest

import numpy as np
import tango

# The DATA device must never reach a developer's running Tiled (its default is
# 127.0.0.1:9091): point it at a closed local port for the whole session.
os.environ["ASYNCROSCOPY_TILED_URI"] = "http://127.0.0.1:9"
from tango.test_context import MultiDeviceTestContext

from asyncroscopy.testing import prop_path

# Import device classes to test
from asyncroscopy.instruments.electron_microscope.detectors.camera import CAMERA
from asyncroscopy.instruments.electron_microscope.detectors.eds import EDS
from asyncroscopy.instruments.electron_microscope.hardware.scan import SCAN
from asyncroscopy.instruments.electron_microscope.hardware.TestStage import TestStage
from asyncroscopy.instruments.electron_microscope.hardware.corrector_digital_twin import DigitalTwinCorrector
from asyncroscopy.instruments.electron_microscope.digital_twin import DigitalTwin
from asyncroscopy.instruments.electron_microscope.digital_twin_particles import DigitalTwinParticles
from asyncroscopy.instruments.electron_microscope.digital_twin_tilt import DigitalTwinTilt
from asyncroscopy.instruments.electron_microscope.auto_script import AutoScriptMicroscope
from asyncroscopy.data.data import DATA
from asyncroscopy.instruments.electron_microscope.mapper import Mapper
from asyncroscopy.safety.approvals import Approvals
from asyncroscopy.instruments.nano_indenter.indent_grid import IndentGrid
from asyncroscopy.instruments.nano_indenter.sim_indenter import SimIndenter


from tests.test_agent import setup_llm_stubs
# Stub out heavy LangChain/Ollama dependencies so the agent swarm imports without them
setup_llm_stubs()

class FakeAdornedImage:
    def __init__(self, data: np.ndarray):
        self.data = data


# We use DigitalTwin as our simulated microscope for all tests.

MAPPING_TWIN_PROPERTIES = {
    "sample_seed": 2024,
    "sample_size_xy": 4e-7,
    "sample_particle_count": 40,
    "particle_radius_mean": 8e-9,
    "particle_radius_std": 3e-9,
    "particle_aspect_ratio": 0.6,
    "support_thickness": 1e-8,
    "stage_move_noise_std": 5e-9,
    "stage_bias_x": 3e-9,
    "stage_bias_y": -2e-9,
    "stage_reported_noise_std": 1e-9,
    "stage_backlash_x": 4e-9,
    "stage_backlash_y": 6e-9,
    "drift_rate_per_min": 0.5e-9,
    "scan_line_jitter_px": 0.3,
    "scan_shear": 0.005,
    "detector_read_noise": 0.5,
    "dose_tracking": True,
    "contamination_max": 2e-9,
    "time_scale": 0.0,
}

@pytest.fixture(scope="session")
def data_save_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("data-acquisitions")


@pytest.fixture(scope="session")
def job_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("jobs")


@pytest.fixture(scope="session")
def tango_ctx(data_save_dir, job_dir):
    """
    One Tango device server hosting SCAN + STEMMicroscope together.

    Device names here MUST match what you put into STEMMicroscope properties.
    """
    devices_info = [
        {
            "class": SCAN,
            "devices": [
                {
                    "name": "asyncroscopy/scan/default",
                    "properties": {
                        # put SCAN defaults here if you want
                        # e.g. "dwell_time": 2e-6  (only if it's a device_property)
                    },
                },
                {"name": "asyncroscopy/scan/mapping", "properties": {}},
            ],
        },
        {
            "class": EDS,
            "devices": [
                {
                    "name": "asyncroscopy/eds/default",
                    "properties": {},
                }
            ],
        },
        {
            "class": CAMERA,
            "devices": [
                {
                    "name": "asyncroscopy/camera/default",
                    "properties": {},
                }
            ],
        },
        {
            "class": TestStage,
            "devices": [
                {
                    "name": "asyncroscopy/stage/default",
                    "properties": {},
                },
                {"name": "asyncroscopy/stage/mapping", "properties": {}},
            ],
        },
        {
            "class": DigitalTwinCorrector,
            "devices": [
                {
                    "name": "asyncroscopy/corrector/default",
                    "properties": {"measurement_noise_fraction": 0.0},
                }
            ],
        },
        {
            "class": DATA,
            "devices": [
                {
                    "name": "asyncroscopy/data/default",
                    "properties": {},
                }
            ],
        },
        {
            "class": DigitalTwin,
            "devices": [
                {
                    "name": "asyncroscopy/digitaltwin/default",
                    "properties": {
                        "scan_device_address": "asyncroscopy/scan/default",
                        "eds_device_address": "asyncroscopy/eds/default",
                        "stage_device_address": "asyncroscopy/stage/default",
                        "corrector_device_address": "asyncroscopy/corrector/default",
                        "camera_device_address": "asyncroscopy/camera/default",
                        "acquisition_save_directory": prop_path(data_save_dir),
                    },
                },
                {
                    # A twin with every realistic effect on, for mapping and Mapper tests.
                    "name": "asyncroscopy/digitaltwin/mapping",
                    "properties": {
                        "scan_device_address": "asyncroscopy/scan/mapping",
                        "eds_device_address": "asyncroscopy/eds/default",
                        "stage_device_address": "asyncroscopy/stage/mapping",
                        "acquisition_save_directory": prop_path(data_save_dir),
                        **MAPPING_TWIN_PROPERTIES,
                    },
                },
            ],
        },
        {
            "class": Mapper,
            "devices": [
                {
                    "name": "asyncroscopy/mapper/default",
                    "properties": {
                        "microscope_device_address": "asyncroscopy/digitaltwin/mapping",
                        "scan_device_address": "asyncroscopy/scan/mapping",
                        "eds_device_address": "asyncroscopy/eds/default",
                        "output_directory": prop_path(job_dir / "maps"),
                        "state_directory": prop_path(job_dir / "state"),
                        "trace_path": prop_path(job_dir / "trace" / "jobs.jsonl"),
                        "estimate_move_s": 1.0,
                    },
                }
            ],
        },
        {
            "class": Approvals,
            "devices": [
                {
                    "name": "asyncroscopy/approvals/default",
                    "properties": {"trace_path": prop_path(job_dir / "trace" / "approvals.jsonl"), "mode_file": prop_path(job_dir / "approvals_mode.json"), "request_timeout_s": 600},
                }
            ],
        },
        # The worked example from docs/Adding_New_Hardware/new_instrument.md.
        {"class": SimIndenter, "devices": [{"name": "asyncroscopy/indenter/default", "properties": {"seed": 3}}]},
        {
            "class": IndentGrid,
            "devices": [
                {
                    "name": "asyncroscopy/indentgrid/default",
                    "properties": {
                        "indenter_device_address": "asyncroscopy/indenter/default",
                        "state_directory": prop_path(job_dir / "indent" / "state"),
                        "trace_path": prop_path(job_dir / "indent" / "trace.jsonl"),
                    },
                }
            ],
        },
        {
            "class": DigitalTwinParticles,
            "devices": [
                {
                    "name": "asyncroscopy/digitaltwinparticles/default",
                    "properties": {
                        "scan_device_address": "asyncroscopy/scan/default",
                        "eds_device_address": "asyncroscopy/eds/default",
                        "stage_device_address": "asyncroscopy/stage/default",
                        "acquisition_save_directory": prop_path(data_save_dir),
                        "sample_particle_count": 8,
                        "world_imsize": 48,
                    },
                }
            ],
        },
        {
            "class": DigitalTwinTilt,
            "devices": [
                {
                    "name": "asyncroscopy/digitaltwintilt/default",
                    "properties": {
                        "scan_device_address": "asyncroscopy/scan/default",
                        "stage_device_address": "asyncroscopy/stage/default",
                        "camera_device_address": "asyncroscopy/camera/default",
                        "acquisition_save_directory": prop_path(data_save_dir),
                        "diffraction_image_size": 16,
                        "randomness_scale": 0.0,
                    },
                }
            ],
        },

        {
            "class": AutoScriptMicroscope,
            "devices": [
                {
                    "name": "asyncroscopy/autoscriptmicroscope/default",
                    "properties": {
                        "testing_mode_bool": True,
                        "scan_device_address": "asyncroscopy/scan/default",
                        "camera_device_address": "asyncroscopy/camera/default",
                        "eds_device_address": "asyncroscopy/eds/default",
                        "stage_device_address": "asyncroscopy/stage/default",
                        "data_device_address": "asyncroscopy/data/default",
                    },
                }
            ],
        },
    ]

    # Keep one in-process context for the whole session. Starting multiple
    # in-process Tango contexts in one interpreter can segfault in PyTango.
    ctx = MultiDeviceTestContext(devices_info, process=False)
    with ctx:
        data = tango.DeviceProxy(ctx.get_device_access("asyncroscopy/data/default"))
        data.save_path = str(data_save_dir)
        yield ctx



@pytest.fixture(scope="session")
def scan_proxy(tango_ctx):
    return tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/scan/default"))


@pytest.fixture(scope="session")
def twin_proxy(tango_ctx):
    return tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/digitaltwin/default"))


@pytest.fixture(scope="session")
def particles_twin_proxy(tango_ctx):
    return tango.DeviceProxy(
        tango_ctx.get_device_access("asyncroscopy/digitaltwinparticles/default")
    )


@pytest.fixture(scope="session")
def tilt_twin_proxy(tango_ctx):
    return tango.DeviceProxy(
        tango_ctx.get_device_access("asyncroscopy/digitaltwintilt/default")
    )


@pytest.fixture(scope="session")
def eds_proxy(tango_ctx):
    return tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/eds/default"))


@pytest.fixture(scope="session")
def camera_proxy(tango_ctx):
    return tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/camera/default"))


@pytest.fixture(scope="session")
def stage_proxy(tango_ctx):
    return tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/stage/default"))


@pytest.fixture(scope="session")
def corrector_proxy(tango_ctx):
    return tango.DeviceProxy(
        tango_ctx.get_device_access("asyncroscopy/corrector/default")
    )


@pytest.fixture(scope="session")
def data_proxy(tango_ctx):
    return tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/data/default"))


@pytest.fixture(scope="session")
def auto_script_proxy(tango_ctx):
    return tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/autoscriptmicroscope/default"))


@pytest.fixture
def patched_single_image(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Patch AutoScriptMicroscope._acquire_scanned_image so acquire_scanned_image() works
    without AutoScript/hardware.
    """
    def fake_acquire(self, imsize: int, dwell_time: float, detector_list: list = ["haadf"], scan_region: list[float] = [0.0, 0.0, 1.0, 1.0], output_format: str = ".h5"):
        # Deterministic image makes tests stable
        arr = np.arange(imsize * imsize, dtype=np.uint16)
        return FakeAdornedImage(arr.reshape(imsize, imsize))

    monkeypatch.setattr(
        AutoScriptMicroscope,
        "_acquire_scanned_image",
        fake_acquire,
    )
    monkeypatch.setattr(
        DigitalTwin,
        "_acquire_scanned_image",
        fake_acquire,
    )


@pytest.fixture
def patched_path_acquisition(monkeypatch: pytest.MonkeyPatch, tmp_path):
    calls = []

    def fake_acquire(self, imsize: int, dwell_time: float, detector_list: list = ["haadf"], scan_region: list[float] = [0.0, 0.0, 1.0, 1.0], output_format: str = ".h5"):
        calls.append(
            {
                "imsize": imsize,
                "dwell_time": dwell_time,
                "detector_list": list(detector_list),
                "scan_region": list(scan_region),
            }
        )
        path = tmp_path / f"stem_{imsize}.h5"
        path.write_bytes(b"fake-h5")
        return str(path)

    monkeypatch.setattr(AutoScriptMicroscope, "_acquire_scanned_image", fake_acquire)
    return calls


@pytest.fixture
def patched_scanned_path_acquisition(monkeypatch: pytest.MonkeyPatch, tmp_path):
    calls = []

    def fake_acquire(self, imsize: int, dwell_time: float, detector_list: list = ["haadf"], scan_region: list[float] = [0.0, 0.0, 1.0, 1.0], output_format: str = ".h5"):
        calls.append(
            {
                "imsize": imsize,
                "dwell_time": dwell_time,
                "detector_list": list(detector_list),
                "scan_region": list(scan_region),
            }
        )
        path = tmp_path / f"stem_{imsize}.h5"
        path.write_bytes(b"fake-stem-h5")
        return str(path)

    monkeypatch.setattr(AutoScriptMicroscope, "_acquire_scanned_image", fake_acquire)
    return calls


@pytest.fixture
def patched_scanned_data_acquisition(monkeypatch: pytest.MonkeyPatch):
    calls = []

    def fake_acquire(
        self,
        imsize: int,
        dwell_time: float,
        detector: str,
        scan_region: list[float],
    ):
        calls.append(
            {
                "imsize": imsize,
                "dwell_time": dwell_time,
                "detector": detector,
                "scan_region": list(scan_region),
            }
        )
        return "fake-stem-data-key"

    monkeypatch.setattr(AutoScriptMicroscope, "_acquire_scanned_data_advanced", fake_acquire)
    return calls


@pytest.fixture
def patched_camera_path_acquisition(monkeypatch: pytest.MonkeyPatch, tmp_path):
    calls = []

    def fake_acquire(
        self,
        imsize: int,
        exposure_time: float,
        detector: str,
        readout_area: str,
        frame_combining: int = 1,
        output_format: str = ".h5",
    ):
        calls.append(
            {
                "imsize": imsize,
                "exposure_time": exposure_time,
                "detector": detector,
                "readout_area": readout_area,
                "frame_combining": frame_combining,
                "output_format": output_format,
            }
        )
        path = tmp_path / f"camera_{imsize}.h5"
        path.write_bytes(b"fake-camera-h5")
        return str(path)

    monkeypatch.setattr(AutoScriptMicroscope, "_acquire_camera_image", fake_acquire)
    return calls


@pytest.fixture
def patched_spectrum_path_acquisition(monkeypatch: pytest.MonkeyPatch, tmp_path):
    calls = []

    def fake_acquire(self, detector_name: str, exposure_time: float):
        calls.append({"detector_name": detector_name, "exposure_time": exposure_time})
        path = tmp_path / f"spectrum_{detector_name}.h5"
        path.write_bytes(b"fake-spectrum-h5")
        return str(path)

    monkeypatch.setattr(AutoScriptMicroscope, "_acquire_spectrum", fake_acquire)
    return calls


@pytest.fixture(scope="session")
def mapping_twin_proxy(tango_ctx):
    return tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/digitaltwin/mapping"))


@pytest.fixture(scope="session")
def mapping_scan_proxy(tango_ctx):
    return tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/scan/mapping"))


@pytest.fixture(scope="session")
def mapping_stage_proxy(tango_ctx):
    return tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/stage/mapping"))


@pytest.fixture(scope="session")
def mapper_proxy(tango_ctx):
    proxy = tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/mapper/default"))
    proxy.set_timeout_millis(60_000)
    return proxy


@pytest.fixture(scope="session")
def approvals_proxy(tango_ctx):
    return tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/approvals/default"))


@pytest.fixture(scope="session")
def indent_grid_proxy(tango_ctx):
    return tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/indentgrid/default"))
