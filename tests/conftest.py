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

import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import numpy as np
import tango

# The DATA device must never reach a developer's running Tiled (its default is
# 127.0.0.1:9091): point it at a closed local port for the whole session.
os.environ["ASYNCROSCOPY_TILED_URI"] = "http://127.0.0.1:9"
# Likewise the dose ledger: the session's frames go to a fresh file, not a developer's outputs/dose.
os.environ["ASYNCROSCOPY_DOSE_LEDGER"] = os.path.join(tempfile.mkdtemp(prefix="asyncroscopy-dose-"), "ledger.jsonl")
# Seals of test data go to a throwaway manifest and key, never outputs/integrity.
_integrity_dir = tempfile.mkdtemp(prefix="asyncroscopy-integrity-")
os.environ["ASYNCROSCOPY_INTEGRITY_LOG"] = os.path.join(_integrity_dir, "manifest.jsonl")
os.environ["ASYNCROSCOPY_INTEGRITY_KEY_FILE"] = os.path.join(_integrity_dir, "key")
from tango.test_context import MultiDeviceTestContext

from asyncroscopy.testing import prop_path

# Import device classes to test
from asyncroscopy.instruments.electron_microscope.detectors.camera import CAMERA
from asyncroscopy.instruments.electron_microscope.detectors.eds import EDS
from asyncroscopy.instruments.electron_microscope.hardware.scan import SCAN
from asyncroscopy.instruments.electron_microscope.hardware.TestStage import TestStage
from asyncroscopy.instruments.electron_microscope.hardware.stage_digital_twin import DigitalTwinSTAGE
from asyncroscopy.instruments.electron_microscope.hardware.corrector_digital_twin import DigitalTwinCorrector
from asyncroscopy.instruments.electron_microscope.digital_twin import DigitalTwin
from asyncroscopy.instruments.electron_microscope.digital_twin_particles import DigitalTwinParticles
from asyncroscopy.instruments.electron_microscope.digital_twin_tilt import DigitalTwinTilt
from asyncroscopy.instruments.electron_microscope.auto_script import AutoScriptMicroscope
from asyncroscopy.data.data import DATA
from asyncroscopy.jobs.mapping.mapper import Mapper
from asyncroscopy.jobs.tuning.tuner import Tuner
from asyncroscopy.safety.approvals import Approvals
from asyncroscopy.instruments.nano_indenter.indent_grid import IndentGrid
from asyncroscopy.instruments.nano_indenter.sim_indenter import SimIndenter
from asyncroscopy.instruments.yaml_sim.device import device_class as yaml_device_class
from asyncroscopy.instruments.yaml_sim.job import job_class as yaml_job_class

AFM_YAML = Path(__file__).resolve().parents[1] / "asyncroscopy" / "instruments" / "yaml_sim" / "examples" / "afm.yaml"
STEP_YAML = Path(__file__).resolve().parent / "data" / "step_twin.yaml"


from tests.test_agent import setup_llm_stubs
# Stub out heavy LangChain/Ollama dependencies so the agent swarm imports without them
setup_llm_stubs()

class FakeAdornedImage:
    def __init__(self, data: np.ndarray) -> None:
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

DRIFT_TWIN_PROPERTIES = {
    **MAPPING_TWIN_PROPERTIES,
    "sample_seed": 7,
    "stage_move_noise_std": 5e-10,
    "stage_bias_x": 0.0,
    "stage_bias_y": 0.0,
    "stage_reported_noise_std": 2e-10,
    "stage_backlash_x": 0.0,
    "stage_backlash_y": 0.0,
    "stage_move_overhead": 2.0,  # long moves, short frames: little drift within a frame
    "drift_rate_per_min": 6e-8,  # 1 nm/s, steady (no random walk, no settling)
    "scan_line_jitter_px": 0.0,
    "scan_shear": 0.0,
    "dose_tracking": False,
    "time_scale": 0.0,
}

# Live drift tracking (tests/test_live_drift.py): the brief's 5e-11 m/s steady drift, a dense sample so every
# overlap between 50 nm tiles has particles to register on, a 1 nm stage, and a 10 s move-and-settle per tile.
LIVE_DRIFT_TWIN_PROPERTIES = {
    **DRIFT_TWIN_PROPERTIES,
    "sample_seed": 11,
    "sample_particle_count": 1200,
    "particle_radius_mean": 3e-9,
    "particle_radius_std": 1e-9,
    "stage_move_noise_std": 1e-9,
    "stage_move_overhead": 10.0,
    "drift_rate_per_min": 5e-11 * 60,
}


# Every sample feature the procedural specimen has (clusters, height gradient, support) and every
# instrument effect that reads the sample (dose damage, contamination), for the golden regression.
SPECIMEN_GOLDEN_PROPERTIES = {
    **MAPPING_TWIN_PROPERTIES,
    "sample_seed": 4242,
    "sample_particle_count": 60,
    "sample_cluster_count": 3,
    "sample_cluster_radius": 5e-8,
    "sample_cluster_fraction": 0.5,
    "height_gradient_x": 0.01,
    "height_gradient_y": -0.005,
    "damage_critical_dose": 0.5,  # e/A^2: the second survey frame below passes it
}


# One instrument (MAPPING_TWIN_PROPERTIES), three specimens: tests/test_specimens.py runs the same Mapper
# plan on each. Flat properties, because this test database cuts property values at ":".
SPECIMENS_DIR = Path(__file__).resolve().parents[1] / "configs" / "specimens"
SPECIMEN_TWINS = {
    "procedural": {"specimen_source": "procedural"},
    "crystal": {"specimen_source": "crystal"},
    "recorded": {
        "specimen_source": "recorded",
        "specimen_path": prop_path(SPECIMENS_DIR / "recorded_example_haadf.png"),
        "specimen_pixel_size_m": 1.5625e-9,
        "specimen_labels_path": prop_path(SPECIMENS_DIR / "recorded_example_labels.json"),
    },
}


@pytest.fixture(scope="session")
def data_save_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("data-acquisitions")


@pytest.fixture(scope="session")
def job_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("jobs")


@pytest.fixture(scope="session")
def tango_ctx(data_save_dir: Path, job_dir: Path) -> Iterator[MultiDeviceTestContext]:
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
                {"name": "asyncroscopy/scan/drift", "properties": {}},
                {"name": "asyncroscopy/scan/livedrift", "properties": {}},
                {"name": "asyncroscopy/scan/tuner", "properties": {}},
                *({"name": f"asyncroscopy/scan/specimen_{source}", "properties": {}} for source in SPECIMEN_TWINS),
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
                {"name": "asyncroscopy/stage/holders", "properties": {
                    "holders": json.dumps({"low_tilt": {"alpha_deg": [-15.0, 15.0], "beta_deg": [-5.0, 5.0], "z_m": [-1e-4, 1e-4]}, "single_tilt": {"beta_deg": [0.0, 1e-9]}}),
                    "holder": "low_tilt",
                }},
                {"name": "asyncroscopy/stage/parity", "properties": {}},
                {"name": "asyncroscopy/stage/livedrift", "properties": {}},
                *({"name": f"asyncroscopy/stage/specimen_{source}", "properties": {}} for source in SPECIMEN_TWINS),
            ],
        },
        {
            # The twin's own STAGE device: the drift twin reports its stage model's error to it (test_twin_configs.py).
            "class": DigitalTwinSTAGE,
            "devices": [{"name": "asyncroscopy/stage/drift", "properties": {}}],
        },
        {
            "class": DigitalTwinCorrector,
            "devices": [
                {
                    "name": "asyncroscopy/corrector/default",
                    "properties": {"measurement_noise_fraction": 0.0, "measurement_noise_m": 0.0},
                },
                # tests/test_tuner.py: the tuner twin's corrector (noisy measurements, as by default),
                # and a wrong-sign one whose corrections add instead of subtract.
                {"name": "asyncroscopy/corrector/tuner", "properties": {}},
                {"name": "asyncroscopy/corrector/wrongsign", "properties": {"correction_sign": -1.0}},
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
                {
                    # Strong steady drift and a precise stage, run with time_scale 0: the drift fit and
                    # revisits must use the twin's clock (clock_s), not this computer's.
                    "name": "asyncroscopy/digitaltwin/drift",
                    "properties": {
                        "scan_device_address": "asyncroscopy/scan/drift",
                        "stage_device_address": "asyncroscopy/stage/drift",
                        "acquisition_save_directory": prop_path(data_save_dir),
                        **DRIFT_TWIN_PROPERTIES,
                    },
                },
                {
                    # The brief's 5e-11 m/s drift, for live drift tracking (tests/test_live_drift.py).
                    "name": "asyncroscopy/digitaltwin/livedrift",
                    "properties": {
                        "scan_device_address": "asyncroscopy/scan/livedrift",
                        "stage_device_address": "asyncroscopy/stage/livedrift",
                        "acquisition_save_directory": prop_path(data_save_dir),
                        **LIVE_DRIFT_TWIN_PROPERTIES,
                    },
                },
                {
                    # Real-time stage moves (1 um takes 1 s) and no STAGE device: its own travel limits apply.
                    "name": "asyncroscopy/digitaltwin/slowstage",
                    "properties": {
                        "scan_device_address": "asyncroscopy/scan/default",
                        "acquisition_save_directory": prop_path(data_save_dir),
                        "time_scale": 0.1,
                        "stage_speed": 1e-7,
                        "stage_move_overhead": 0.0,
                        "stage_travel_m": 1e-5,
                    },
                },
                {
                    # An ideal twin (every effect off) that tests/test_parity.py compares with AutoScript on the fake client.
                    "name": "asyncroscopy/digitaltwin/parity",
                    "properties": {
                        "scan_device_address": "asyncroscopy/scan/default",
                        "eds_device_address": "asyncroscopy/eds/default",
                        "stage_device_address": "asyncroscopy/stage/parity",
                        "acquisition_save_directory": prop_path(data_save_dir),
                        "sample_particle_count": 10,
                    },
                },
                {
                    # Ideal stage and scan with dose tracking, for tests/test_dose_aware.py: its DoseMap cell
                    # (2 um / 200 = 10 nm) matches the dose Mapper's eds_dose_spread_m.
                    "name": "asyncroscopy/digitaltwin/dose",
                    "properties": {
                        "scan_device_address": "asyncroscopy/scan/default",
                        "eds_device_address": "asyncroscopy/eds/default",
                        "acquisition_save_directory": prop_path(data_save_dir),
                        "sample_size_xy": 2e-6,
                        "sample_particle_count": 20,
                        "dose_tracking": True,
                        "time_scale": 0.0,
                    },
                },
                {
                    # tests/test_specimens.py: the procedural specimen's output must stay bit-identical to the
                    # golden file; only that test touches this twin, so its event indices are reproducible.
                    "name": "asyncroscopy/digitaltwin/specimen_golden",
                    "properties": {"acquisition_save_directory": prop_path(data_save_dir), **SPECIMEN_GOLDEN_PROPERTIES},
                },
                # tests/test_specimens.py: one instrument, three specimens, the same Mapper plan on each.
                *(
                    {
                        "name": f"asyncroscopy/digitaltwin/specimen_{source}",
                        "properties": {
                            "scan_device_address": f"asyncroscopy/scan/specimen_{source}",
                            "stage_device_address": f"asyncroscopy/stage/specimen_{source}",
                            "eds_device_address": "asyncroscopy/eds/default",
                            "acquisition_save_directory": prop_path(data_save_dir),
                            **MAPPING_TWIN_PROPERTIES,
                            **specimen,
                        },
                    }
                    for source, specimen in SPECIMEN_TWINS.items()
                ),
                {
                    # tests/test_tuner.py: a dense particle sample on a thin film, with a height gradient
                    # (focus changes across the stage) and dose tracking; each test injects its own hidden
                    # focus error with set_focus_error.
                    "name": "asyncroscopy/digitaltwin/tuner",
                    "properties": {
                        "scan_device_address": "asyncroscopy/scan/tuner",
                        "corrector_device_address": "asyncroscopy/corrector/tuner",
                        "acquisition_save_directory": prop_path(data_save_dir),
                        "sample_seed": 77,
                        "sample_size_xy": 1e-6,
                        "sample_particle_count": 2000,
                        "support_thickness": 3e-9,
                        "height_gradient_x": 0.02,
                        "height_gradient_y": -0.01,
                        "dose_tracking": True,
                        "time_scale": 0.0,
                    },
                },
            ],
        },
        {
            "class": Tuner,
            "devices": [
                {
                    "name": "asyncroscopy/tuner/default",
                    "properties": {
                        "microscope_device_address": "asyncroscopy/digitaltwin/tuner",
                        "scan_device_address": "asyncroscopy/scan/tuner",
                        "corrector_device_address": "asyncroscopy/corrector/tuner",
                        "state_directory": prop_path(job_dir / "tuner" / "state"),
                        "trace_path": prop_path(job_dir / "tuner" / "trace.jsonl"),
                        "simulated": True,
                    },
                },
                {
                    "name": "asyncroscopy/tuner/wrongsign",
                    "properties": {
                        "microscope_device_address": "asyncroscopy/digitaltwin/tuner",
                        "scan_device_address": "asyncroscopy/scan/tuner",
                        "corrector_device_address": "asyncroscopy/corrector/wrongsign",
                        "state_directory": prop_path(job_dir / "tuner_wrongsign" / "state"),
                        "trace_path": prop_path(job_dir / "tuner_wrongsign" / "trace.jsonl"),
                        "simulated": True,
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
                        "stage_device_address": "asyncroscopy/stage/mapping",
                        "output_directory": prop_path(job_dir / "maps"),
                        "state_directory": prop_path(job_dir / "state"),
                        "trace_path": prop_path(job_dir / "trace" / "jobs.jsonl"),
                        "estimate_move_s": 1.0,
                        "simulated": True,
                    },
                },
                {
                    "name": "asyncroscopy/mapper/lease",
                    "properties": {
                        "microscope_device_address": "asyncroscopy/digitaltwin/mapping",
                        "scan_device_address": "asyncroscopy/scan/mapping",
                        "eds_device_address": "asyncroscopy/eds/default",
                        "stage_device_address": "asyncroscopy/stage/mapping",
                        "output_directory": prop_path(job_dir / "lease" / "maps"),
                        "state_directory": prop_path(job_dir / "lease" / "state"),
                        "trace_path": prop_path(job_dir / "lease" / "trace.jsonl"),
                        "estimate_move_s": 1.0,
                        "simulated": True,
                        "lease_s": 1.5,
                        "pause_limit_s": 3.0,
                    },
                },
                {
                    "name": "asyncroscopy/mapper/dose",
                    "properties": {
                        "microscope_device_address": "asyncroscopy/digitaltwin/dose",
                        "scan_device_address": "asyncroscopy/scan/default",
                        "eds_device_address": "asyncroscopy/eds/default",
                        "output_directory": prop_path(job_dir / "dose" / "maps"),
                        "state_directory": prop_path(job_dir / "dose" / "state"),
                        "trace_path": prop_path(job_dir / "dose" / "trace" / "jobs.jsonl"),
                        "estimate_move_s": 1.0,
                        "eds_dose_spread_m": 1e-8,
                    },
                },
                *(
                    {
                        "name": f"asyncroscopy/mapper/specimen_{source}",
                        "properties": {
                            "microscope_device_address": f"asyncroscopy/digitaltwin/specimen_{source}",
                            "scan_device_address": f"asyncroscopy/scan/specimen_{source}",
                            "eds_device_address": "asyncroscopy/eds/default",
                            "stage_device_address": f"asyncroscopy/stage/specimen_{source}",
                            "output_directory": prop_path(job_dir / "specimen" / source / "maps"),
                            "state_directory": prop_path(job_dir / "specimen" / source / "state"),
                            "trace_path": prop_path(job_dir / "specimen" / source / "trace.jsonl"),
                            "estimate_move_s": 1.0,
                            "simulated": True,
                        },
                    }
                    for source in SPECIMEN_TWINS
                ),
                {
                    # Maps of the drifting twin below, for the one-time-source tests (test_dry_run_and_clock.py).
                    "name": "asyncroscopy/mapper/drift",
                    "properties": {
                        "microscope_device_address": "asyncroscopy/digitaltwin/drift",
                        "scan_device_address": "asyncroscopy/scan/drift",
                        "stage_device_address": "asyncroscopy/stage/drift",
                        "output_directory": prop_path(job_dir / "drift" / "maps"),
                        "state_directory": prop_path(job_dir / "drift" / "state"),
                        "trace_path": prop_path(job_dir / "drift" / "trace.jsonl"),
                        "stage_sigma_m": 3e-9,
                        "max_drift_m_per_s": 2e-8,
                        "simulated": True,
                    },
                },
                {
                    # Maps of the 5e-11 m/s twin above, with and without live drift tracking (test_live_drift.py).
                    "name": "asyncroscopy/mapper/livedrift",
                    "properties": {
                        "microscope_device_address": "asyncroscopy/digitaltwin/livedrift",
                        "scan_device_address": "asyncroscopy/scan/livedrift",
                        "stage_device_address": "asyncroscopy/stage/livedrift",
                        "output_directory": prop_path(job_dir / "livedrift" / "maps"),
                        "state_directory": prop_path(job_dir / "livedrift" / "state"),
                        "trace_path": prop_path(job_dir / "livedrift" / "trace.jsonl"),
                        "estimate_move_s": 10.0,
                        "stage_sigma_m": 2e-9,
                        "max_drift_m_per_s": 1e-10,
                        "simulated": True,
                    },
                },
                {
                    # The mapping twin driven as an SEM: the beam commands only, no EDS (test_map_profiles.py).
                    "name": "asyncroscopy/mapper/sem",
                    "properties": {
                        "instrument_profile": "sem",
                        "microscope_device_address": "asyncroscopy/digitaltwin/mapping",
                        "scan_device_address": "asyncroscopy/scan/mapping",
                        "stage_device_address": "asyncroscopy/stage/mapping",
                        "output_directory": prop_path(job_dir / "sem" / "maps"),
                        "state_directory": prop_path(job_dir / "sem" / "state"),
                        "trace_path": prop_path(job_dir / "sem" / "trace.jsonl"),
                        "estimate_move_s": 1.0,
                        "simulated": True,
                    },
                },
                {
                    # The YAML AFM twin below, mapped through its map: section (test_map_profiles.py).
                    "name": "asyncroscopy/mapper/afm",
                    "properties": {
                        "instrument_profile": "yaml",
                        "microscope_device_address": "asyncroscopy/afm/default",
                        "stage_device_address": "",
                        "output_directory": prop_path(job_dir / "afm" / "maps"),
                        "state_directory": prop_path(job_dir / "afm" / "state"),
                        "trace_path": prop_path(job_dir / "afm" / "trace.jsonl"),
                        "estimate_move_s": 1.0,
                        "stage_sigma_m": 5e-9,
                        "max_drift_m_per_s": 0.0,
                        "simulated": True,
                    },
                },
            ],
        },
        {
            # The AFM example as a YAML instrument (its job device is left out here), with no simulated waits.
            "class": yaml_device_class("AFMTwin"),
            "devices": [
                {
                    "name": "asyncroscopy/afm/default",
                    "properties": {
                        "yaml_path": prop_path(AFM_YAML),
                        "output_directory": prop_path(job_dir / "afm" / "acquisitions"),
                        "time_scale": 0.0,
                    },
                },
            ],
        },
        {
            # The step protocol between a YAML instrument and its job device (tests/test_yaml_step_protocol.py):
            # one simulated second is 20 ms, so a 40 s step takes 0.8 s; the second instrument's hold lapses fast.
            "class": yaml_device_class("StepTwin"),
            "devices": [
                {
                    "name": "asyncroscopy/steptwin/default",
                    "properties": {"yaml_path": prop_path(STEP_YAML), "output_directory": prop_path(job_dir / "step" / "acquisitions"), "time_scale": 0.02},
                },
                {
                    "name": "asyncroscopy/steptwin/lapsing",
                    "properties": {"yaml_path": prop_path(STEP_YAML), "output_directory": prop_path(job_dir / "step" / "acquisitions"), "time_scale": 0.02, "hold_lease_s": 0.4},
                },
            ],
        },
        {
            "class": yaml_job_class("StepTwinJob"),
            "devices": [
                {
                    "name": "asyncroscopy/steptwinjob/default",
                    "properties": {
                        "yaml_path": prop_path(STEP_YAML),
                        "instrument_device_address": "asyncroscopy/steptwin/default",
                        "state_directory": prop_path(job_dir / "step" / "state"),
                        "trace_path": prop_path(job_dir / "step" / "trace.jsonl"),
                        "simulated": True,
                    },
                },
            ],
        },
        {
            "class": Approvals,
            "devices": [
                {
                    "name": "asyncroscopy/approvals/default",
                    "properties": {"trace_path": prop_path(job_dir / "trace" / "approvals.jsonl"), "mode_file": prop_path(job_dir / "approvals_mode.json"), "request_timeout_s": 600, "simulated": True},
                },
                {
                    # The gate of a stack that is not simulated (simulated: false): auto mode is refused.
                    "name": "asyncroscopy/approvals/hardware",
                    "properties": {"trace_path": prop_path(job_dir / "trace" / "approvals_hw.jsonl"), "mode_file": prop_path(job_dir / "approvals_hw_mode.json"), "simulated": False},
                },
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
def scan_proxy(tango_ctx: MultiDeviceTestContext) -> tango.DeviceProxy:
    return tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/scan/default"))


@pytest.fixture(scope="session")
def twin_proxy(tango_ctx: MultiDeviceTestContext) -> tango.DeviceProxy:
    return tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/digitaltwin/default"))


@pytest.fixture(scope="session")
def particles_twin_proxy(tango_ctx: MultiDeviceTestContext) -> tango.DeviceProxy:
    return tango.DeviceProxy(
        tango_ctx.get_device_access("asyncroscopy/digitaltwinparticles/default")
    )


@pytest.fixture(scope="session")
def tilt_twin_proxy(tango_ctx: MultiDeviceTestContext) -> tango.DeviceProxy:
    return tango.DeviceProxy(
        tango_ctx.get_device_access("asyncroscopy/digitaltwintilt/default")
    )


@pytest.fixture(scope="session")
def eds_proxy(tango_ctx: MultiDeviceTestContext) -> tango.DeviceProxy:
    return tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/eds/default"))


@pytest.fixture(scope="session")
def camera_proxy(tango_ctx: MultiDeviceTestContext) -> tango.DeviceProxy:
    return tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/camera/default"))


@pytest.fixture(scope="session")
def stage_proxy(tango_ctx: MultiDeviceTestContext) -> tango.DeviceProxy:
    return tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/stage/default"))


@pytest.fixture(scope="session")
def corrector_proxy(tango_ctx: MultiDeviceTestContext) -> tango.DeviceProxy:
    return tango.DeviceProxy(
        tango_ctx.get_device_access("asyncroscopy/corrector/default")
    )


@pytest.fixture(scope="session")
def data_proxy(tango_ctx: MultiDeviceTestContext) -> tango.DeviceProxy:
    return tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/data/default"))


@pytest.fixture(scope="session")
def auto_script_proxy(tango_ctx: MultiDeviceTestContext) -> tango.DeviceProxy:
    return tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/autoscriptmicroscope/default"))


@pytest.fixture
def patched_single_image(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Patch AutoScriptMicroscope._acquire_scanned_image so acquire_scanned_image() works
    without AutoScript/hardware.
    """
    def fake_acquire(self, imsize: int, dwell_time: float, detector_list: list = ["haadf"], scan_region: list[float] = [0.0, 0.0, 1.0, 1.0], output_format: str = ".h5") -> FakeAdornedImage:
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
def patched_path_acquisition(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[dict]:
    calls = []

    def fake_acquire(self, imsize: int, dwell_time: float, detector_list: list = ["haadf"], scan_region: list[float] = [0.0, 0.0, 1.0, 1.0], output_format: str = ".h5") -> str:
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
def patched_scanned_path_acquisition(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[dict]:
    calls = []

    def fake_acquire(self, imsize: int, dwell_time: float, detector_list: list = ["haadf"], scan_region: list[float] = [0.0, 0.0, 1.0, 1.0], output_format: str = ".h5") -> str:
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
def patched_scanned_data_acquisition(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls = []

    def fake_acquire(
        self,
        imsize: int,
        dwell_time: float,
        detector: str,
        scan_region: list[float],
    ) -> str:
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
def patched_camera_path_acquisition(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[dict]:
    calls = []

    def fake_acquire(
        self,
        imsize: int,
        exposure_time: float,
        detector: str,
        readout_area: str,
        frame_combining: int = 1,
        output_format: str = ".h5",
    ) -> str:
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
def patched_spectrum_path_acquisition(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[dict]:
    calls = []

    def fake_acquire(self, detector_name: str, exposure_time: float) -> str:
        calls.append({"detector_name": detector_name, "exposure_time": exposure_time})
        path = tmp_path / f"spectrum_{detector_name}.h5"
        path.write_bytes(b"fake-spectrum-h5")
        return str(path)

    monkeypatch.setattr(AutoScriptMicroscope, "_acquire_spectrum", fake_acquire)
    return calls


@pytest.fixture(scope="session")
def mapping_twin_proxy(tango_ctx: MultiDeviceTestContext) -> tango.DeviceProxy:
    return tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/digitaltwin/mapping"))


@pytest.fixture(scope="session")
def mapping_scan_proxy(tango_ctx: MultiDeviceTestContext) -> tango.DeviceProxy:
    return tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/scan/mapping"))


@pytest.fixture(scope="session")
def mapping_stage_proxy(tango_ctx: MultiDeviceTestContext) -> tango.DeviceProxy:
    return tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/stage/mapping"))


@pytest.fixture(scope="session")
def slow_stage_twin_proxy(tango_ctx: MultiDeviceTestContext) -> tango.DeviceProxy:
    return tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/digitaltwin/slowstage"))


@pytest.fixture(scope="session")
def mapper_proxy(tango_ctx: MultiDeviceTestContext) -> tango.DeviceProxy:
    proxy = tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/mapper/default"))
    proxy.set_timeout_millis(60_000)
    return proxy


@pytest.fixture(scope="session")
def sem_mapper_proxy(tango_ctx: MultiDeviceTestContext) -> tango.DeviceProxy:
    proxy = tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/mapper/sem"))
    proxy.set_timeout_millis(60_000)
    return proxy


@pytest.fixture(scope="session")
def afm_mapper_proxy(tango_ctx: MultiDeviceTestContext) -> tango.DeviceProxy:
    proxy = tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/mapper/afm"))
    proxy.set_timeout_millis(60_000)
    return proxy


@pytest.fixture(scope="session")
def afm_twin_proxy(tango_ctx: MultiDeviceTestContext) -> tango.DeviceProxy:
    return tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/afm/default"))


@pytest.fixture(scope="session")
def dose_twin_proxy(tango_ctx: MultiDeviceTestContext) -> tango.DeviceProxy:
    return tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/digitaltwin/dose"))


@pytest.fixture(scope="session")
def dose_mapper_proxy(tango_ctx: MultiDeviceTestContext) -> tango.DeviceProxy:
    proxy = tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/mapper/dose"))
    proxy.set_timeout_millis(60_000)
    return proxy


@pytest.fixture(scope="session")
def parity_twin_device(tango_ctx: MultiDeviceTestContext) -> Any:
    """The parity twin's device object itself (the context runs in this process), so tests/test_parity.py
    runs each command's body on it exactly as it does on AutoScriptMicroscope over the fake client."""
    tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/digitaltwin/parity")).ping()
    return tango.Util.instance().get_device_by_name("asyncroscopy/digitaltwin/parity")


@pytest.fixture(scope="session")
def approvals_proxy(tango_ctx: MultiDeviceTestContext) -> tango.DeviceProxy:
    return tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/approvals/default"))


@pytest.fixture(scope="session")
def indent_grid_proxy(tango_ctx: MultiDeviceTestContext) -> tango.DeviceProxy:
    return tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/indentgrid/default"))
