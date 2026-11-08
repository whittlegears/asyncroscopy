"""
General-purpose STEM digital twin.

Runs the same commands as the real microscope against a seeded, simulated
sample, so workflows can be rehearsed and benchmarked before they touch
hardware. Every realistic effect (stage error, backlash, drift, scan
distortion, focus, detector noise, EDS spectra, beam dose) is a device
property that defaults to off, so a bare config behaves like an ideal
instrument; ``configs/digital_twin_mapping.yaml`` turns them on with
literature magnitudes. See docs/digital_twin.md.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np
import tango
from tango.server import device_property

from asyncroscopy.utils.commands import command
from asyncroscopy.data.data_writer import save_acquisition
from asyncroscopy.routines.beam import DEFAULT_BEAM_CURRENT_PA
from asyncroscopy.routines.dose import ELECTRON_CHARGE
from asyncroscopy.instruments.electron_microscope.digital_twin_base import (
    DEFAULT_ACQUISITION_DIR,
    ElectronMicroscopeTwin,
)
from asyncroscopy.instruments.electron_microscope.electron_microscope import check_defocus, check_scan
from asyncroscopy.instruments.electron_microscope.twin import (
    EdsParams,
    SampleParams,
    ScanParams,
    pixel_sample_coordinates,
    point_spectrum,
)
from asyncroscopy.sim.dose import DoseMap
from asyncroscopy.sim.rng import EventRng
from asyncroscopy.sim.clock import SimClock
from asyncroscopy.sim.stage_model import StageTrajectory
from asyncroscopy.instruments.electron_microscope.twin.detector import (
    DetectorDrift,
    DetectorParams,
    count,
    electrons_per_pixel,
)
from asyncroscopy.sim.dose import DoseParams
from asyncroscopy.instruments.electron_microscope.twin.optics import aberrations_nm
from asyncroscopy.instruments.electron_microscope.twin.render import (
    DETECTOR_SCALE,
    Renderer,
    RenderParams,
)
from asyncroscopy.instruments.electron_microscope.twin.elements import ELEMENTS
from asyncroscopy.instruments.electron_microscope.twin.specimen import SpecimenContext
from asyncroscopy.instruments.electron_microscope.twin.scan_distortion import frame_time

__all__ = ["DEFAULT_ACQUISITION_DIR", "DigitalTwin"]

ANGSTROM = 1e-10


def _prop(dtype, default, doc):
    return device_property(dtype=dtype, default_value=default, doc=doc)


class DigitalTwin(ElectronMicroscopeTwin):
    """Seeded particle-sample STEM twin with realistic, switchable instrument effects."""

    # Specimen: where the sample comes from (docs/Operation/specimens.md). The flat sample properties below
    # are the procedural source's settings and the other sources' defaults (seed, size, support, height).
    # specimen and specimen_source are the base class's; specimen_source: procedural (seeded particles), recorded (an image file), crystal (a polycrystalline film), or a plug-in.
    specimen_path = _prop(str, "", "recorded: the image file (.npy, .npz, .png, .tif, .h5).")
    specimen_pixel_size_m = _prop(float, 0.0, "recorded: pixel size of the image (m); 0 reads it from the file.")
    specimen_labels_path = _prop(str, "", "recorded: optional JSON of labelled particles (ground truth and EDS composition).")
    # Sample (sample_seed, sample_particle_count, sample_size_xy come from the base class)
    particle_radius_mean = _prop(float, 1.6e-9, "Mean particle radius (m).")
    particle_radius_std = _prop(float, 0.2e-9, "Particle radius standard deviation (m).")
    particle_aspect_ratio = _prop(float, 0.4, "Particle thickness / radius.")
    particle_elements = _prop(str, "Au,Pt,Fe", "Comma-separated particle elements.")
    support_thickness = _prop(float, 0.0, "Amorphous carbon support film thickness (m).")
    sample_cluster_count = _prop(int, 0, "Number of particle clusters (0: particles spread uniformly).")
    sample_cluster_radius = _prop(float, 0.0, "Typical cluster radius (m).")
    sample_cluster_fraction = _prop(float, 0.0, "Fraction of particles placed in clusters (0-1).")
    height_gradient_x = _prop(float, 0.0, "Specimen height change per metre along x (focus varies across the map).")
    height_gradient_y = _prop(float, 0.0, "Specimen height change per metre along y.")
    # Limits a real column enforces; the twin refuses rather than rendering nonsense (or running out of memory).
    # min_fov, max_fov, max_defocus (render cost grows with defocus squared) and max_stigmator are the base's.
    max_image_shift = _prop(float, 1e-6, "Largest |image shift| per axis accepted (m).")
    # Focus truth the operator does not see: get_defocus and get_stigmator never include it. A twin
    # started with these set is defocused and astigmatic, and its images are blurred accordingly, so a
    # focus routine has to find the correction from the images (export_ground_truth reports the residual).
    focus_offset_m = _prop(float, 0.0, "Hidden focus error (m) added to C1; not reported by get_defocus.")
    astigmatism_offset_a_m = _prop(float, 0.0, "Hidden two-fold astigmatism A1, first component (m); not reported by get_stigmator.")
    astigmatism_offset_b_m = _prop(float, 0.0, "Hidden two-fold astigmatism A1, second component (m).")
    autofocus_residual_m = _prop(float, 1e-9, "Standard deviation (m) of the C1 and of each A1 component the simulated OptiSTEM leaves behind.")
    autofocus_frames = _prop(int, 8, "Frames the simulated OptiSTEM scans (at the SCAN settings); they count in the dose map and the clock.")
    # Optics and detector
    acceleration_voltage = _prop(float, 300e3, "Acceleration voltage (V).")
    convergence_angle_mrad = _prop(float, 30.0, "Probe convergence semi-angle (mrad).")
    probe_current_pa = _prop(float, DEFAULT_BEAM_CURRENT_PA, "Probe current (pA); reported as the measured beam_current_pa.")
    haadf_z_exponent = _prop(float, 1.7, "HAADF contrast exponent n in Z^n.")
    haadf_scattering_scale = _prop(float, 2.25e-5, "k in 1-exp(-k * sum Z^n rho t[A]).")
    atomic_pixel_threshold = _prop(float, 0.2e-9, "Below this pixel size (m) atoms are rendered.")
    detector_gain = _prop(float, 1.0, "Counts per detected electron.")
    detector_offset = _prop(float, 0.0, "Detector dark offset (counts).")
    detector_read_noise = _prop(float, 0.0, "Gaussian read noise (counts).")
    detector_gain_drift = _prop(float, 0.0, "Largest slow swing of the scanning detectors' gain, as a fraction (0: steady).")
    detector_offset_drift = _prop(float, 0.0, "Largest slow swing of the scanning detectors' offset (counts; 0: steady).")
    # Scan
    scan_flyback_time = _prop(float, 0.0, "Line flyback time (s).")
    scan_line_jitter_px = _prop(float, 0.0, "Per-line x jitter standard deviation (pixels).")
    scan_rotation_deg = _prop(float, 0.0, "Scan rotation relative to the stage (deg).")
    scan_shear = _prop(float, 0.0, "Scan shear (x shift per unit y).")
    scan_flyback_compression = _prop(float, 0.0, "Left-edge compression after flyback (fraction).")
    fov_scale_error = _prop(float, 0.0, "Field-of-view calibration error (fraction).")
    # The stage's errors, dynamics and drift are the STAGE device's (DigitalTwinSTAGE); the clock is the microscope's.
    time_scale = _prop(float, 0.0, "Real seconds slept per simulated second (0 = never wait).")
    # EDS
    eds_fwhm_mn_ka = _prop(float, 130.0, "EDS resolution at Mn K-alpha (eV).")
    eds_solid_angle = _prop(float, 0.7, "EDS collection solid angle (sr).")
    eds_dead_time = _prop(float, 1e-6, "EDS paralyzable dead time (s).")
    eds_channels = _prop(int, 2048, "EDS channels.")
    eds_energy_max = _prop(float, 20000.0, "EDS energy range upper limit (eV).")
    # Dose
    dose_tracking = _prop(bool, False, "Record dose and apply contamination/damage.")
    contamination_max = _prop(float, 0.0, "Carbon contamination thickness at saturation (m).")
    contamination_dose = _prop(float, 1e4, "Dose (e/A^2) for contamination to reach 63% of saturation.")
    damage_critical_dose = _prop(float, 0.0, "Dose (e/A^2) beyond which particles start to fade (0 = off).")
    dose_spot_spread = _prop(float, 1e-8, "A parked beam's dose (EDS) is counted over a square this wide (m); keep it equal to the Mapper's eds_dose_spread_m.")

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------
    def _check_properties(self) -> None:
        """Refuse impossible configuration at start-up, not with a cryptic error at the first image."""
        rules = [
            ("probe_current_pa", float(self.probe_current_pa) > 0, "must be > 0"),
            ("particle_radius_mean", float(self.particle_radius_mean) > 0, "must be > 0"),
            ("sample_size_xy", float(self.sample_size_xy) > 0, "must be > 0"),
            ("particle_elements", any(e.strip() for e in str(self.particle_elements).split(",")), "must name at least one element"),
            ("fov_scale_error", float(self.fov_scale_error) > -1, "must be > -1 (a fraction)"),
            ("eds_channels", int(self.eds_channels) >= 16, "must be >= 16"),
            ("eds_energy_max", float(self.eds_energy_max) > 0, "must be > 0"),
            ("contamination_dose", float(self.contamination_dose) > 0, "must be > 0"),
            ("sample_cluster_count", int(self.sample_cluster_count) >= 0, "must be >= 0"),
            ("sample_cluster_radius", float(self.sample_cluster_radius) >= 0, "must be >= 0"),
            ("sample_cluster_fraction", 0.0 <= float(self.sample_cluster_fraction) <= 1.0, "must be in [0, 1]"),
        ]
        bad = [f"{name} {why} (got {getattr(self, name)!r})" for name, ok, why in rules if not ok]
        if bad:
            raise ValueError("digital twin configuration: " + "; ".join(bad))

    def _specimen_options(self) -> dict:
        """The specimen_* flat properties, overridden by the ``specimen`` mapping."""
        options = super()._specimen_options()
        if options["source"] == "recorded":  # the flat shorthands are the recorded source's options
            path, pixel, labels = str(self.specimen_path or "").strip(), float(self.specimen_pixel_size_m or 0.0), str(self.specimen_labels_path or "").strip()
            options = {**{k: v for k, v in (("path", path), ("pixel_size_m", pixel), ("labels_path", labels)) if v}, **options}
        return options

    def _generate_sample(self, seed: int) -> None:
        self._check_properties()
        self._rng = EventRng(int(seed))
        self._clock = SimClock(float(self.time_scale))
        self._acq_index = 0
        # Every acquisition's truth goes to a JSONL file as it is taken (export_ground_truth reads it back), so a
        # long session does not hold them all in memory; only the count is kept here.
        self._truth_path = Path(self.acquisition_save_directory) / "ground_truth" / f"{self.get_name().replace('/', '_')}_acquisitions.jsonl"
        self._truth_path.parent.mkdir(parents=True, exist_ok=True)
        self._truth_path.write_text("", encoding="utf-8")
        self._acquisition_count = 0
        elements = tuple(e.strip() for e in str(self.particle_elements).split(",") if e.strip())
        # The specimen is a plug-in (twin/specimen.py, docs/Operation/specimens.md); everything below it
        # is the instrument, which only asks the specimen for weights, densities, heights and features.
        self._load_specimen(
            seed,
            SpecimenContext(
                rng=self._rng,
                sample=SampleParams(
                    size_xy_m=float(self.sample_size_xy),
                    particle_count=int(self.sample_particle_count),
                    radius_mean_m=float(self.particle_radius_mean),
                    radius_std_m=float(self.particle_radius_std),
                    aspect_ratio=float(self.particle_aspect_ratio),
                    elements=elements,
                    support_thickness_m=float(self.support_thickness),
                    height_gradient=(float(self.height_gradient_x), float(self.height_gradient_y)),
                    cluster_count=int(self.sample_cluster_count),
                    cluster_radius_m=float(self.sample_cluster_radius),
                    cluster_fraction=float(self.sample_cluster_fraction),
                ),
            ),
        )
        for warning in self._specimen.warnings():
            self.warn_stream(warning)
        # The stage model runs on the STAGE device (the session lets it start afresh when this device restarts);
        # this is what the device last said, at rest at the origin until it first answers.
        self._stage_session = uuid.uuid4().hex
        self._stage = StageTrajectory()
        self._renderer = Renderer(
            self._specimen,
            RenderParams(
                z_exponent=float(self.haadf_z_exponent),
                haadf_scale=float(self.haadf_scattering_scale),
                atomic_pixel_m=float(self.atomic_pixel_threshold),
                voltage_v=float(self.acceleration_voltage),
                convergence_mrad=float(self.convergence_angle_mrad),
            ),
        )
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
        self._eds_params = EdsParams(
            energy_max_ev=float(self.eds_energy_max),
            channels=int(self.eds_channels),
            fwhm_mn_ka_ev=float(self.eds_fwhm_mn_ka),
            solid_angle_sr=float(self.eds_solid_angle),
            dead_time_s=float(self.eds_dead_time),
            voltage_v=float(self.acceleration_voltage),
        )
        x_min, x_max, y_min, y_max = self._specimen.bounds
        self._dose = DoseMap(
            DoseParams(
                enabled=bool(self.dose_tracking),
                cell_m=max((x_max - x_min) / 200.0, 2e-9),
                contamination_angstrom_max=float(self.contamination_max) / ANGSTROM,
                contamination_dose_e_per_a2=float(self.contamination_dose),
                damage_critical_dose_e_per_a2=float(self.damage_critical_dose),
                spot_spread_m=float(self.dose_spot_spread),
            ),
            self._specimen.bounds,
            margin_m=(x_max - x_min) * 0.25,
        )
        self._blanked = False
        self._image_shift = np.zeros(2, dtype=np.float64)
        self._stigmator = np.zeros(2, dtype=np.float64)
        self._autofocus_runs = 0
        self._focus_error = {
            "c1_m": float(self.focus_offset_m),
            "a1_m": [float(self.astigmatism_offset_a_m), float(self.astigmatism_offset_b_m)],
        }
        self._stage_position = self._stage.true.copy()
        self._world_bounds_ang = {
            "x_min": x_min / ANGSTROM,
            "x_max": x_max / ANGSTROM,
            "y_min": y_min / ANGSTROM,
            "y_max": y_max / ANGSTROM,
            "z_min": -float(self.sample_size_z) / ANGSTROM / 2,
            "z_max": float(self.sample_size_z) / ANGSTROM / 2,
        }

    def _blank_beam(self) -> None:
        """A blanked beam gives no signal and no dose, as on the instrument."""
        self._blanked = True

    def _unblank_beam(self) -> None:
        self._blanked = False

    def _beam_current_pa(self) -> float:
        return 0.0 if self._blanked else self._detector_params.beam_current_pa

    # ------------------------------------------------------------------
    # Stage: commanded / true / reported
    # ------------------------------------------------------------------
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

    def _set_fov(self, fov) -> None:
        self._fov = float(fov)

    def _set_defocus(self, defocus) -> None:
        self._defocus = float(defocus)

    def _get_stage(self):
        self._sync_stage_from_proxy()
        return self._stage.reported.copy()

    # ------------------------------------------------------------------
    # Image shift, stigmator and the simulated vendor autofocus
    # ------------------------------------------------------------------
    def _set_image_shift(self, shift) -> None:
        value = np.asarray(shift, dtype=np.float64)
        limit = float(self.max_image_shift)
        if value.shape != (2,) or not np.all(np.isfinite(value)):
            raise ValueError(f"image shift must be [x, y] in metres, got {np.atleast_1d(value).tolist()}")
        if np.any(np.abs(value) > limit):
            raise ValueError(f"image shift must be within +/-{limit} m per axis (did you pass nanometres?), got {value.tolist()}")
        self._image_shift = value

    def _get_image_shift(self):
        return self._image_shift.copy()

    def _set_stigmator(self, stigmator: list[float]) -> None:
        self._stigmator = np.asarray(stigmator, dtype=np.float64)

    def _get_stigmator(self) -> list[float]:
        return self._stigmator.tolist()

    def _beam_current(self) -> dict:
        # Always current: nothing on the twin drifts it between readings.
        return {"pa": float(self._detector_params.beam_current_pa), "source": "measured", "age_s": 0.0}

    def set_focus_error(self, c1_m: float, a1_m: tuple[float, float] = (0.0, 0.0)) -> None:
        """Test and bench hook, not a Tango command: replace the hidden focus error, as a new specimen would."""
        self._focus_error = {"c1_m": float(c1_m), "a1_m": [float(a1_m[0]), float(a1_m[1])]}

    def _focus_truth(self) -> dict:
        """The column's effective C1 at the beam position (specimen height included) and its A1, in metres."""
        ab = self._aberrations()
        x, y = (float(v) for v in self._stage.true[:2] + self._image_shift)
        c1 = ab.get("C10", 0.0) * 1e-9 - float(self._specimen.height_at(x, y))
        a1 = [ab.get("C12a", 0.0) * 1e-9, ab.get("C12b", 0.0) * 1e-9]
        return {
            "c1_m": float(c1), "a1_m": a1, "a1_abs_m": float(np.hypot(*a1)),
            "defocus_m": float(self._defocus), "stigmator_m": self._stigmator.tolist(),
        }

    def _auto_focus(self) -> None:
        """Stand-in for OptiSTEM C1A1: scans autofocus_frames frames, then zeroes C1 and A1 to within autofocus_residual_m."""
        self._refuse_while_moving()
        scan = self._detector_proxies.get("scan")
        imsize, dwell = (int(scan.imsize), float(scan.dwell_time)) if scan is not None else (256, 1e-6)
        blanked = self._blanked
        self._blanked = False  # the vendor routine unblanks for its own scans
        try:
            for _ in range(max(0, int(self.autofocus_frames))):
                self._scan_frame(imsize, dwell, [0.0, 0.0, 1.0, 1.0])
        finally:
            self._blanked = blanked
        self._autofocus_runs += 1
        rng = self._rng.for_event("autofocus", self._autofocus_runs)
        residual = max(0.0, float(self.autofocus_residual_m))
        truth = self._focus_truth()
        self._defocus = check_defocus(self._defocus - truth["c1_m"] + rng.normal(0.0, residual), self.max_defocus)
        self._stigmator = self._stigmator - np.asarray(truth["a1_m"]) + rng.normal(0.0, residual, size=2)

    # ------------------------------------------------------------------
    # Imaging
    # ------------------------------------------------------------------
    def _aberrations(self) -> dict:
        coefficients = None
        corrector = self._detector_proxies.get("corrector")
        if corrector is not None:
            try:
                coefficients = json.loads(corrector.get_aberrations_coeff_sim())
            except (tango.DevFailed, AttributeError, ValueError):
                coefficients = None
        ab = aberrations_nm(coefficients, float(self.acceleration_voltage), float(self._defocus))
        # The hidden focus error and the stigmator, neither of which the corrector sees.
        error = getattr(self, "_focus_error", {"c1_m": 0.0, "a1_m": [0.0, 0.0]})
        stigmator = getattr(self, "_stigmator", np.zeros(2))
        ab["C10"] = float(ab.get("C10", 0.0)) + error["c1_m"] * 1e9
        ab["C12a"] = float(ab.get("C12a", 0.0)) + (error["a1_m"][0] + float(stigmator[0])) * 1e9
        ab["C12b"] = float(ab.get("C12b", 0.0)) + (error["a1_m"][1] + float(stigmator[1])) * 1e9
        return ab

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

    def _render_stem_image(self, imsize: int, dwell_time: float, detector_list: list, scan_region=None, frame=None) -> np.ndarray:
        """Render one detector channel; pass ``frame`` to share one scan between detectors."""
        if frame is None:
            frame = self._scan_frame(int(imsize), float(dwell_time), scan_region)
        detector = str(detector_list[0]).upper()
        xs, ys = frame["xs"], frame["ys"]
        cx, cy = float(xs.mean()), float(ys.mean())
        visible = self._specimen.in_box(float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max()))
        fraction = self._renderer.render(
            xs,
            ys,
            frame["pixel_m"],
            detector,
            self._aberrations(),
            support_angstrom=self._support_angstrom(cx, cy),
            damage=self._dose.damage_factors(self._specimen.feature_centers, visible),
        )
        rng = self._rng.for_event(f"detector_{detector}", frame["index"])
        params = self._detector_drift.at(replace(self._detector_params, beam_current_pa=self._beam_current_pa()), frame["record"]["t_start_s"])
        return count(fraction, frame["dwell_s"], params, rng)

    def _acquire_scanned_image(
        self,
        imsize: int,
        dwell_time: float,
        detector_list: list[str] = ["haadf"],
        scan_region: list[float] = [0.0, 0.0, 1.0, 1.0],
        output_format: str = ".h5",
    ) -> str:
        self._refuse_while_moving()
        detectors = list(dict.fromkeys(d.upper() for d in detector_list))  # "HAADF" and "haadf" are one detector
        unknown = [d for d in detectors if d not in DETECTOR_SCALE]
        if unknown:
            tango.Except.throw_exception(
                "UnknownDetector",
                f"The twin simulates scanning detectors {sorted(d.lower() for d in DETECTOR_SCALE)}; got {unknown}.",
                "acquire_scanned_image()",
            )
        frame = self._scan_frame(int(imsize), float(dwell_time), scan_region)
        images = [self._render_stem_image(int(imsize), float(dwell_time), [d], scan_region, frame=frame) for d in detectors]
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
            "sim_time_start_s": record["t_start_s"],
            "axes": "rows=+y, columns=+x",
        }
        return save_acquisition(
            self,
            self._detector_proxies.get("data"),
            "stem_image",
            detectors,
            images,
            dataset_attrs=[dict(attrs) for _ in images],
            output_format=output_format,
        )

    # ------------------------------------------------------------------
    # EDS
    # ------------------------------------------------------------------
    def _beam_sample_xy(self, t: float) -> np.ndarray:
        """Sample point under the beam at time t, from the fractional beam position."""
        fov = float(self._fov) * (1.0 + self._scan_params.fov_scale_error)
        x = (self._beam_pos_x - 0.5) * fov
        y = (self._beam_pos_y - 0.5) * fov
        x = x + self._scan_params.shear * y
        if self._scan_params.rotation_deg:
            a = np.deg2rad(self._scan_params.rotation_deg)
            x, y = np.cos(a) * x - np.sin(a) * y, np.sin(a) * x + np.cos(a) * y
        # The beam is placed in the scan's frame, which the image shift moves with it (as on the instrument).
        return self._stage.true_xy_at(t) + self._image_shift + np.array([x, y])

    def _support_angstrom(self, x: float, y: float) -> float:
        """Support film (the specimen's) plus contamination (the beam's doing) at a sample point, in angstrom."""
        return float(self._specimen.support_thickness_m) / ANGSTROM + self._dose.contamination_angstrom(x, y)

    def _areal_density(self, xy: np.ndarray) -> dict[str, float]:
        """Atoms per square angstrom of each element along the beam at a sample point."""
        x, y = float(xy[0]), float(xy[1])
        density = dict(self._specimen.areal_density(x, y))
        support = self._support_angstrom(x, y)
        if support > 0:
            element = self._specimen.support_element
            density[element] = density.get(element, 0.0) + ELEMENTS[element][1] * support
        return density

    def _acquire_camera_image(self, *args, **kwargs):
        self._refuse_while_moving()
        return super()._acquire_camera_image(*args, **kwargs)

    def _acquire_spectrum(self, detector_name: str, exposure_time: float) -> str:
        self._refuse_while_moving()
        real_time = float(exposure_time)
        self._fetch_stage(real_time)
        self._acq_index += 1
        index = self._acq_index
        t0 = self._clock.now()
        xy = self._beam_sample_xy(t0 + real_time / 2.0)
        energies, counts, live_time = point_spectrum(
            self._areal_density(xy),
            self._beam_current_pa(),
            real_time,
            self._eds_params,
            self._rng.for_event("eds", index),
        )
        electrons = self._beam_current_pa() * 1e-12 * real_time / ELECTRON_CHARGE
        self._dose.add_point(float(xy[0]), float(xy[1]), electrons, 1e-20)
        self._clock.advance(real_time)
        self._record_acquisition(
            {
                "index": index,
                "kind": "spectrum",
                "t_start_s": t0,
                "t_end_s": self._clock.now(),
                "beam_fraction": [float(self._beam_pos_x), float(self._beam_pos_y)],
                "beam_true_m": [float(v) for v in xy],
                "reported_m": self._stage.reported.tolist(),
                "true_m": self._stage.true.tolist(),
            }
        )
        dispersion = float(energies[1] - energies[0])
        return save_acquisition(
            self,
            self._detector_proxies.get("data"),
            "spectrum",
            detector_name,
            counts,
            dataset_name="spectrum",
            dataset_attrs={
                "units": "counts",
                "energy_offset_ev": float(energies[0]),
                "energy_dispersion_ev": dispersion,
                "energy_axis": "channel i is at energy_offset_ev + i * energy_dispersion_ev",
                "real_time_s": real_time,
                "live_time_s": float(live_time),
                "beam_position": [float(self._beam_pos_x), float(self._beam_pos_y)],
                "stage_reported_m": self._stage.reported.tolist(),
                "fov_m": float(self._fov),
                "beam_current_pa": self._beam_current_pa(),
                "axes": "rows=+y, columns=+x",
                "acquisition_index": index,
            },
        )

    # ------------------------------------------------------------------
    # Status and ground truth
    # ------------------------------------------------------------------
    def _viewport_metadata(self) -> dict:
        # The base class reads _stage_position, which here is the true pose: report what the stage reports.
        metadata = super()._viewport_metadata()
        x, y, z = (float(v) / ANGSTROM for v in self._stage.reported[:3])
        half = self._fov / ANGSTROM / 2
        metadata["stage_position"] = self._stage.reported.tolist()
        metadata["viewport_world_angstrom"] = {"x_min": x - half, "x_max": x + half, "y_min": y - half, "y_max": y + half, "z_center": z}
        # The seed regenerates the sample: export_ground_truth (operator only) has it; an agent must find
        # particles by imaging.
        metadata.pop("sample_seed", None)
        return metadata

    def _get_parameters(self) -> str:
        parameters = json.loads(super()._get_parameters())
        parameters["stage_position"] = self._stage.reported.tolist()
        parameters["scan_detectors"] = sorted(d.lower() for d in DETECTOR_SCALE)
        parameters["simulated_time_s"] = self._clock.now()
        parameters["time_scale"] = self._clock.time_scale
        # The twin delivers exactly its configured current, so it counts as measured (AutoScript: the screen current).
        parameters.update(self.beam_current_fields())
        parameters["beam_blanked"] = self._blanked
        parameters["stage_moving"] = self._stage_moving()
        parameters["simulated"] = True  # for clients; job devices take the stack's `simulated` property, never this
        parameters["specimen_source"] = self._specimen.source  # what kind of sample, not what is on it
        return json.dumps(parameters)

    @command(dtype_out=str, risk="high", operator_only=True, role="data.read")
    def export_ground_truth(self) -> str:
        """Hidden truth for scoring workflows: particles, true stage pose and drift of every acquisition.

        Operator only (an agent must not read a twin's answers); tests and calibration scripts call it through Tango.
        """
        specimen_truth = self._specimen.ground_truth()
        stage = self._detector_proxies.get("stage")
        return json.dumps(
            {
                "sample_seed": int(self.sample_seed),
                "sample_bounds_m": list(self._specimen.bounds),
                # The particle list scorers compare against; empty when the source has none (a crystal
                # film) or does not know them (a recorded image without labels: specimen.known is false).
                "particles": specimen_truth.get("particles", []),
                "specimen": {**self._specimen.describe(), **{k: v for k, v in specimen_truth.items() if k != "particles"}},
                "moves": json.loads(stage.stage_moves()) if stage is not None else [],
                "acquisitions": self._acquisition_records(),
                "simulated_time_s": self._clock.now(),
                "dose": self._dose.summary(),
                "focus": {**self._focus_truth(), "hidden_error": self._focus_error},
            }
        )


if __name__ == "__main__":
    DigitalTwin.run_server()
