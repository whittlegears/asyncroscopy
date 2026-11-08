"""A thin simulated SEM: the scanning-beam twin's clock, stage, scan and dose over a ``Specimen``.

It reuses what the STEM twin is made of: ``ScanningBeamTwin`` (the clock, the DigitalTwinSTAGE stage model, the blanker, the
frame scan with its distortions and dose), the ``Specimen`` plug-ins, the renderer and the counting detector. Nothing here
models secondary-electron physics: ``se`` and ``bse`` are the renderer's mass-thickness signals (HAADF and ADF), which give
particles on a support film the contrast that stitching, scoring and revisits need. No vendor API is involved.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import tango
from tango import DevState
from tango.server import device_property

from asyncroscopy.data.data_writer import DEFAULT_ACQUISITION_DIR, save_acquisition
from asyncroscopy.instruments.electron_microscope.twin import SampleParams, ScanParams
from asyncroscopy.instruments.electron_microscope.twin.detector import DetectorDrift, DetectorParams, count
from asyncroscopy.instruments.electron_microscope.twin.render import Renderer, RenderParams
from asyncroscopy.instruments.electron_microscope.twin.specimen import SpecimenContext, build_specimen, parse_specimen
from asyncroscopy.instruments.scanning_beam_twin import ScanningBeamTwin
from asyncroscopy.instruments.scanning_electron_microscope.scanning_electron_microscope import ScanningElectronMicroscope
from asyncroscopy.routines.beam import DEFAULT_BEAM_CURRENT_PA
from asyncroscopy.sim.dose import DoseMap, DoseParams

# The SEM detector -> the renderer's signal standing in for it.
SIGNALS = {"SE": "HAADF", "BSE": "ADF"}


def _prop(dtype, default, doc):
    return device_property(dtype=dtype, default_value=default, doc=doc)


class DigitalTwinSEM(ScanningBeamTwin, ScanningElectronMicroscope):
    """A seeded particle (or any projected ``Specimen``) sample imaged by a simulated SEM."""

    simulated = _prop(bool, True, "True when this stack drives a simulator; set from run_servers' top-level simulated:.")
    specimen = _prop(str, "", "Specimen as a mapping (JSON or YAML), overriding specimen_source; see docs/Operation/specimens.md.")
    specimen_source = _prop(str, "procedural", "Specimen source: procedural (seeded particles), recorded, crystal, or a plug-in that renders in 'projected' mode.")
    sample_seed = _prop(int, 12345, "Seed used to generate deterministic sample geometry.")
    sample_size_xy = _prop(float, 2e-6, "Sample XY size (m).")
    sample_particle_count = _prop(int, 200, "Number of particles in the generated sample.")
    particle_radius_mean = _prop(float, 2e-8, "Mean particle radius (m).")
    particle_radius_std = _prop(float, 5e-9, "Particle radius standard deviation (m).")
    particle_aspect_ratio = _prop(float, 0.6, "Particle thickness / radius.")
    particle_elements = _prop(str, "Au,Pt", "Comma-separated particle elements.")
    support_thickness = _prop(float, 1e-8, "Support film thickness (m).")
    acquisition_save_directory = _prop(str, DEFAULT_ACQUISITION_DIR, "Directory where simulated acquisitions are saved before the Tiled server serves them.")
    acceleration_voltage = _prop(float, 15e3, "Acceleration voltage (V).")
    probe_current_pa = _prop(float, DEFAULT_BEAM_CURRENT_PA, "Probe current (pA); reported as the measured beam_current_pa.")
    detector_gain = _prop(float, 1.0, "Counts per detected electron.")
    detector_offset = _prop(float, 0.0, "Detector dark offset (counts).")
    detector_read_noise = _prop(float, 0.0, "Gaussian read noise (counts).")
    detector_gain_drift = _prop(float, 0.0, "Largest slow swing of the detectors' gain, as a fraction (0: steady).")
    detector_offset_drift = _prop(float, 0.0, "Largest slow swing of the detectors' offset (counts; 0: steady).")
    scan_flyback_time = _prop(float, 0.0, "Line flyback time (s).")
    scan_line_jitter_px = _prop(float, 0.0, "Per-line x jitter standard deviation (pixels).")
    scan_rotation_deg = _prop(float, 0.0, "Scan rotation relative to the stage (deg).")
    scan_shear = _prop(float, 0.0, "Scan shear (x shift per unit y).")
    scan_flyback_compression = _prop(float, 0.0, "Left-edge compression after flyback (fraction).")
    fov_scale_error = _prop(float, 0.0, "Field-of-view calibration error (fraction).")
    dose_tracking = _prop(bool, False, "Record where the beam has been.")
    time_scale = _prop(float, 0.0, "Real seconds slept per simulated second (0 = never wait).")

    # ------------------------------------------------------------------
    # The session
    # ------------------------------------------------------------------
    def _connect(self) -> None:
        self._connect_detector_proxies()
        self._start_beam_twin(int(self.sample_seed))
        elements = tuple(e.strip() for e in str(self.particle_elements).split(",") if e.strip())
        mapping = parse_specimen(str(self.specimen or ""))
        self._specimen = build_specimen(
            {"source": str(mapping.get("source") or self.specimen_source).strip(), **mapping},
            SpecimenContext(rng=self._rng, sample=SampleParams(
                size_xy_m=float(self.sample_size_xy),
                particle_count=int(self.sample_particle_count),
                radius_mean_m=float(self.particle_radius_mean),
                radius_std_m=float(self.particle_radius_std),
                aspect_ratio=float(self.particle_aspect_ratio),
                elements=elements,
                support_thickness_m=float(self.support_thickness),
            )),
        )
        if "projected" not in self._specimen.render_modes:
            raise ValueError(f"specimen source {self._specimen.source!r} renders in {list(self._specimen.render_modes)}, not the 'projected' mode this twin uses")
        # atomic_pixel_m 0: an SEM never resolves atoms, so every field is rendered at survey level.
        self._renderer = Renderer(self._specimen, RenderParams(voltage_v=float(self.acceleration_voltage), atomic_pixel_m=0.0))
        self._scan_params = ScanParams(
            flyback_s=float(self.scan_flyback_time),
            line_jitter_px=float(self.scan_line_jitter_px),
            rotation_deg=float(self.scan_rotation_deg),
            shear=float(self.scan_shear),
            flyback_compression=float(self.scan_flyback_compression),
            fov_scale_error=float(self.fov_scale_error),
        )
        self._detector_params = DetectorParams(
            beam_current_pa=float(self.probe_current_pa),
            gain=float(self.detector_gain),
            offset=float(self.detector_offset),
            read_noise=float(self.detector_read_noise),
        )
        self._detector_drift = DetectorDrift(float(self.detector_gain_drift), float(self.detector_offset_drift), self._rng.for_event("detector_drift"))
        x_min, x_max, _y_min, _y_max = self._specimen.bounds
        self._dose = DoseMap(DoseParams(enabled=bool(self.dose_tracking), cell_m=max((x_max - x_min) / 200.0, 2e-9)), self._specimen.bounds, margin_m=(x_max - x_min) * 0.25)
        self._fov = 2e-6
        self._beam_position = [0.5, 0.5]
        self.set_state(DevState.ON)

    def _set_fov(self, fov) -> None:
        self._fov = float(fov)

    def _get_fov(self) -> float:
        return float(self._fov)

    def _place_beam(self, position) -> None:
        self._beam_position = [float(position[0]), float(position[1])]

    def _get_parameters(self) -> str:
        self._sync_stage_from_proxy()
        return json.dumps({
            "fov_m": float(self._fov),
            "stage_position": self._stage.reported.tolist(),
            "beam_position": list(self._beam_position),
            "acceleration_voltage_v": float(self.acceleration_voltage),
            "scan_detectors": sorted(d.lower() for d in SIGNALS),
            "device_proxies": sorted(self._detector_proxies),
            "beam_blanked": self._blanked,
            "stage_moving": self._stage_moving(),
            "simulated_time_s": self._clock.now(),
            "time_scale": self._clock.time_scale,
            "simulated": True,
            "specimen_source": self._specimen.source,
            **self.beam_current_fields(),
        })

    # ------------------------------------------------------------------
    # Imaging
    # ------------------------------------------------------------------
    def _render(self, detector: str, frame: dict) -> np.ndarray:
        xs, ys = frame["xs"], frame["ys"]
        cx, cy = float(xs.mean()), float(ys.mean())
        visible = self._specimen.in_box(float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max()))
        fraction = self._renderer.render(
            xs, ys, frame["pixel_m"], SIGNALS[detector], {"C10": 0.0},
            support_angstrom=self._support_angstrom(cx, cy),
            damage=self._dose.damage_factors(self._specimen.feature_centers, visible),
        )
        params = self._detector_drift.at(replace(self._detector_params, beam_current_pa=self._beam_current_pa()), frame["record"]["t_start_s"])
        return count(fraction, frame["dwell_s"], params, self._rng.for_event(f"detector_{detector}", frame["index"]))

    def _acquire_scanned_image(self, imsize: int, dwell_time: float, detector_list: list[str], scan_region: list[float], output_format: str = ".h5") -> str:
        self._refuse_while_moving()
        detectors = list(dict.fromkeys(d.upper() for d in detector_list))
        unknown = [d for d in detectors if d not in SIGNALS]
        if unknown:
            tango.Except.throw_exception("UnknownDetector", f"The SEM twin simulates scanning detectors {sorted(d.lower() for d in SIGNALS)}; got {unknown}.", "acquire_scanned_image()")
        frame = self._scan_frame(int(imsize), float(dwell_time), scan_region)
        images = [self._render(d, frame) for d in detectors]
        record = frame["record"]
        attrs = {
            "units": "counts",
            "acquisition_index": record["index"],
            "stage_commanded_m": record["commanded_m"],
            "stage_reported_m": record["reported_m"],
            "image_shift_m": self._image_shift.tolist(),
            "fov_m": record["fov_m"],
            "pixel_size_m": frame["pixel_m"],
            "dwell_time_s": record["dwell_s"],
            "scan_region": record["scan_region"],
            "beam_current_pa": self._beam_current_pa(),
            "beam_current_source": "measured",
            "sim_time_start_s": record["t_start_s"],
            "axes": "rows=+y, columns=+x",
        }
        return save_acquisition(self, self._detector_proxies.get("data"), "sem_image", detectors, images, dataset_attrs=[dict(attrs) for _ in images], output_format=output_format)


if __name__ == "__main__":
    DigitalTwinSEM.run_server()
