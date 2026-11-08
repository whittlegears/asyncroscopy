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
from dataclasses import replace

import numpy as np
import tango
from tango.server import device_property

from asyncroscopy.utils.commands import command
from asyncroscopy.data.data_writer import save_acquisition
from asyncroscopy.routines.beam import DEFAULT_BEAM_CURRENT_PA
from asyncroscopy.routines.dose import ELECTRON_CHARGE
from asyncroscopy.instruments.scanning_beam_twin import ANGSTROM, ScanningBeamTwin
from asyncroscopy.instruments.electron_microscope.digital_twin_base import (
    DEFAULT_ACQUISITION_DIR,
    ElectronMicroscopeTwin,
)
from asyncroscopy.instruments.electron_microscope.electron_microscope import check_defocus
from asyncroscopy.instruments.electron_microscope.twin import (
    EdsParams,
    SampleParams,
    ScanParams,
    point_spectrum,
)
from asyncroscopy.sim.dose import DoseMap
from asyncroscopy.instruments.electron_microscope.twin.detector import (
    DetectorDrift,
    DetectorParams,
    count,
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

__all__ = ["DEFAULT_ACQUISITION_DIR", "DigitalTwin"]


def _prop(dtype, default, doc):
    return device_property(dtype=dtype, default_value=default, doc=doc)


class DigitalTwin(ScanningBeamTwin, ElectronMicroscopeTwin):
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
    # min_fov, max_fov, max_image_shift, max_defocus (render cost grows with defocus squared) and max_stigmator are the bases'.
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
        self._start_beam_twin(int(seed))
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
            ),
            self._specimen.bounds,
            margin_m=(x_max - x_min) * 0.25,
        )
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

    # ------------------------------------------------------------------
    # Stage: commanded / true / reported
    # ------------------------------------------------------------------
    def _set_fov(self, fov) -> None:
        self._fov = float(fov)

    def _set_defocus(self, defocus) -> None:
        self._defocus = float(defocus)

    # ------------------------------------------------------------------
    # Image shift, stigmator and the simulated vendor autofocus
    # ------------------------------------------------------------------
    def _set_stigmator(self, stigmator: list[float]) -> None:
        self._stigmator = np.asarray(stigmator, dtype=np.float64)

    def _get_stigmator(self) -> list[float]:
        return self._stigmator.tolist()

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
