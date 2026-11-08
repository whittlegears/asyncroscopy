"""
The shell every simulated electron microscope shares: ``ElectronMicroscopeTwin``.

It holds the Tango plumbing a twin needs and nothing about what it images: the
detector proxies, the beam position, the field of view and defocus, the stage
position (read from and written to the STAGE device), the sample's extent, and
``simulated`` true. A twin fills in ``_generate_sample`` (called on connect) and
its acquisitions. The general-purpose twin is ``DigitalTwin`` in digital_twin.py
(samples are ``Specimen`` plug-ins, twin/specimen.py); the atomistic twins
(tilt, diffraction, polycrystalline gold) add ``twin.atomistic.ParticleSpectrumMixin``.
"""

import json
from abc import abstractmethod

import numpy as np
import tango
from tango import AttrWriteType, DevState
from tango.server import Device, attribute, device_property

from asyncroscopy.instruments.electron_microscope.electron_microscope import ElectronMicroscope
from asyncroscopy.instruments.electron_microscope.detectors.camera import CAMERA

DEFAULT_ACQUISITION_DIR = "outputs/tiled_acquisitions"


class ElectronMicroscopeTwin(ElectronMicroscope):
    """A simulated electron microscope: beam, field of view, defocus and stage state over the support devices."""

    # ------------------------------------------------------------------
    # Device properties — configure in Tango DB per deployment
    # ------------------------------------------------------------------

    # A twin is a simulator unless its stack config says otherwise (run_servers' simulated:).
    simulated = device_property(dtype=bool, default_value=True, doc="True when this stack drives a simulator; set from run_servers' top-level simulated:.")

    sample_seed = device_property(
        dtype=int,
        default_value=12345,
        doc="Seed used to generate deterministic sample geometry.",
    )
    sample_particle_count = device_property(
        dtype=int,
        default_value=40,
        doc="Number of particles in the generated sample.",
    )
    sample_size_xy = device_property(
        dtype=float,
        default_value=6e-9,
        doc="Absolute sample XY size in meters (must be > 0).",
    )
    sample_size_z = device_property(
        dtype=float,
        default_value=6e-9,
        doc="Absolute sample thickness (Z) in meters (must be > 0).",
    )
    acquisition_save_directory = device_property(
        dtype=str,
        default_value=DEFAULT_ACQUISITION_DIR,
        doc="Directory where simulated acquisitions are saved before the Tiled server serves them.",
    )
    acquisition_file_format = device_property(
        dtype=str,
        default_value="h5",
        doc="Acquisition file format. HDF5 stores simulated acquisition data and metadata.",
    )
    data_device_address = device_property(
        dtype=str,
        default_value="",
        doc="Optional Tango device address for the DATA device, e.g. 'asyncroscopy/data/default'.",
    )

    manufacturer = attribute(
        label="DigitalTwin",
        dtype=str,
        doc="Simulation backend",
    )

    beam_pos = attribute(
        label="Beam Position",
        dtype=(float,),
        max_dim_x=2,
        access=AttrWriteType.READ_WRITE,
        unit="fractional",
        min_value=0.0,
        max_value=1.0,
        doc="Beam position as [x, y] fractional coordinates, each in range [0.0, 1.0]",
    )

    def init_device(self) -> None:
        """Initialize the Tango device and simulation state variables."""
        Device.init_device(self)
        self.instrument_id = self.get_name()  # what every dataset it saves records as instrument_id
        self.set_state(DevState.INIT)

        self._init_device_attributes()  # the base's detector proxies and command lock
        self._stem_mode = True
        self._manufacturer = "UTKTeam"
        self._beam_pos_x = 0.5
        self._beam_pos_y = 0.5
        self._defocus = 0.0
        self._imsize = 512
        self._fov = 200e-10  # meters
        self._stage_position = np.zeros(5, dtype=np.float64)  # x, y, z, alpha, beta
        # The sample's extent in angstrom (status reports); _generate_sample sets it.
        self._world_bounds_ang = {
            "x_min": -1.0,
            "x_max": 1.0,
            "y_min": -1.0,
            "y_max": 1.0,
            "z_min": -1.0,
            "z_max": 1.0,
        }

        self._connect()

    def _connect(self):
        """Connect the support devices and build the sample."""
        self._connect_detector_proxies()
        self._generate_sample(seed=int(self.sample_seed))
        self.set_state(DevState.ON)

    def _sync_stage_from_proxy(self) -> None:
        """Fetch the current stage position from the stage device proxy."""
        stage = self._detector_proxies.get("stage")
        if stage is None:
            return
        try:
            self._stage_position = np.array(
                [stage.x, stage.y, stage.z, stage.alpha, stage.beta],
                dtype=np.float64,
            )
        except tango.DevFailed:
            self.error_stream("Failed to read stage proxy position; using internal stage state.")

    @abstractmethod
    def _generate_sample(self, seed: int) -> None:
        """Build the sample from ``seed`` and set ``_world_bounds_ang``; called on connect."""
        raise NotImplementedError(f"{type(self).__name__} does not implement _generate_sample")

    # ------------------------------------------------------------------
    # Attribute read methods
    # ------------------------------------------------------------------
    def read_manufacturer(self) -> str:
        """Read method for the manufacturer attribute."""
        return self._manufacturer

    def read_beam_pos(self) -> list[float]:
        """Read method for the beam position attribute."""
        return [self._beam_pos_x, self._beam_pos_y]

    def write_beam_pos(self, value: list[float]) -> None:
        """Write method for the beam position attribute."""
        x, y = value[0], value[1]
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"beam_pos values must be in [0.0, 1.0], got x={x}, y={y}")
        self._beam_pos_x = float(x)
        self._beam_pos_y = float(y)

    def _viewport_metadata(self) -> dict:
        fov_ang = self._fov * 1e10
        stage_xyz_ang = self._stage_position[:3] * 1e10
        return {
            "stage_position": [float(v) for v in self._stage_position],
            "beam_position": [float(self._beam_pos_x), float(self._beam_pos_y)],
            "fov_m": float(self._fov),
            "fov_angstrom": float(fov_ang),
            "imsize": int(self._imsize),
            "sample_seed": int(self.sample_seed),
            "sample_size_xy": float(self.sample_size_xy),
            "sample_size_z": float(self.sample_size_z),
            "viewport_world_angstrom": {
                "x_min": float(stage_xyz_ang[0] - fov_ang * 0.5),
                "x_max": float(stage_xyz_ang[0] + fov_ang * 0.5),
                "y_min": float(stage_xyz_ang[1] - fov_ang * 0.5),
                "y_max": float(stage_xyz_ang[1] + fov_ang * 0.5),
                "z_center": float(stage_xyz_ang[2]),
            },
            "world_bounds_angstrom": self._world_bounds_ang,
        }

    def _acquire_camera_image(
        self,
        imsize: int,
        exposure_time: float,
        detector: str,
        readout_area: str,
        frame_combining: int = 1,
        electron_counting: bool = True,
        output_format: str = ".h5",
    ) -> str:
        """A single-shot camera image: a twin without a camera model images with its scan (a HAADF image), so
        acquire_camera_image returns a usable image instead of the base ElectronMicroscope's "unsupported"."""
        return self._acquire_scanned_image(int(imsize), float(exposure_time), ["haadf"], [0.0, 0.0, 1.0, 1.0], output_format)

    def _place_beam(self, position) -> None:
        """Place the electron beam at the specified [x, y] coordinates."""
        x, y = position
        self.write_beam_pos([x, y])

    def _set_fov(self, fov) -> None:
        """Set the field of view in meters."""
        self._fov = float(fov)

    def _get_fov(self) -> float:
        """Get the field of view in meters."""
        return float(self._fov)

    def _set_defocus(self, defocus) -> None:
        """Set defocus in meters."""
        self._defocus = float(defocus)

    def _get_defocus(self) -> float:
        """Get defocus in meters."""
        return self._defocus

    def _get_stage(self):
        """Return the current 5-axis stage position."""
        self._sync_stage_from_proxy()
        return self._stage_position

    def _move_stage(self, position):
        """Move the stage to the requested 5-axis vector."""
        if len(position) != 5:
            raise ValueError("Stage position must have 5 values: [x, y, z, alpha, beta]")
        target = np.array(position, dtype=np.float64)

        stage = self._detector_proxies.get("stage")
        if stage is not None:
            # One write of all five axes: the STAGE device checks them together and refuses the whole move
            # (axis by axis, a refused z would leave x and y already moved).
            stage.position = [float(v) for v in target]

        self._stage_position = target

    def _get_parameters(self) -> str:
        """Return all simulated status parameters as a JSON string.

        The base class exposes this through the get_parameters Tango command,
        which is typed DevString — returning a dict (or the inherited abstract
        stub's None) makes the command fail with a Tango translation error.

        The detector keys are deliberately split by what they mean: device_proxies
        are the twin's connected settings devices (scan, stage, data, ... — not
        detectors), scan_detectors is what the twin's STEM renderer actually
        produces (a simulated HAADF signal, whatever label is requested),
        spectrum_detectors is what acquire_spectrum accepts, and camera_detectors
        are the names the CAMERA device validates writes against. An earlier
        draft published the proxy keys under a "detectors" key, which told an
        agent that "stage" and "data" were detectors it could scan with.
        """
        self._sync_stage_from_proxy()
        parameters = {
            "manufacturer": self._manufacturer,
            "stem_mode": bool(self._stem_mode),
            "defocus_m": float(self._defocus),
            "device_proxies": sorted(self._detector_proxies.keys()),
            "scan_detectors": ["haadf"],
            "spectrum_detectors": ["eds"],
            "camera_detectors": sorted(CAMERA._CAMERA_DETECTORS),
            **self._viewport_metadata(),
        }
        return json.dumps(parameters)
