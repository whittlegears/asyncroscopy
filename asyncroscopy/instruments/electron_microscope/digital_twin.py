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
from tango.server import command, device_property

from asyncroscopy.data.data_writer import save_acquisition
from asyncroscopy.instruments.electron_microscope.digital_twin_base import (
    DEFAULT_ACQUISITION_DIR,
    AtomisticDigitalTwin,
)
from asyncroscopy.instruments.electron_microscope.twin import (
    DoseMap,
    EdsParams,
    EventRng,
    SampleModel,
    SampleParams,
    ScanParams,
    SimClock,
    StageModel,
    StageParams,
    pixel_sample_coordinates,
    point_spectrum,
)
from asyncroscopy.instruments.electron_microscope.twin.detector import (
    DetectorParams,
    count,
    electrons_per_pixel,
)
from asyncroscopy.instruments.electron_microscope.twin.dose import DoseParams
from asyncroscopy.instruments.electron_microscope.twin.optics import aberrations_nm
from asyncroscopy.instruments.electron_microscope.twin.render import (
    DETECTOR_SCALE,
    Renderer,
    RenderParams,
)
from asyncroscopy.instruments.electron_microscope.twin.sample_model import ELEMENTS
from asyncroscopy.instruments.electron_microscope.twin.scan_distortion import frame_time

__all__ = ["DEFAULT_ACQUISITION_DIR", "DigitalTwin"]

ANGSTROM = 1e-10


def _prop(dtype, default, doc):
    return device_property(dtype=dtype, default_value=default, doc=doc)


class DigitalTwin(AtomisticDigitalTwin):
    """Seeded particle-sample STEM twin with realistic, switchable instrument effects."""

    # Sample (sample_seed, sample_particle_count, sample_size_xy come from the base class)
    particle_radius_mean = _prop(float, 1.6e-9, "Mean particle radius (m).")
    particle_radius_std = _prop(float, 0.2e-9, "Particle radius standard deviation (m).")
    particle_aspect_ratio = _prop(float, 0.4, "Particle thickness / radius.")
    particle_elements = _prop(str, "Au,Pt,Fe", "Comma-separated particle elements.")
    support_thickness = _prop(float, 0.0, "Amorphous carbon support film thickness (m).")
    height_gradient_x = _prop(float, 0.0, "Specimen height change per metre along x (focus varies across the map).")
    height_gradient_y = _prop(float, 0.0, "Specimen height change per metre along y.")
    # Limits a real column enforces; the twin refuses rather than rendering nonsense (or running out of memory).
    min_fov = _prop(float, 1e-9, "Smallest field of view accepted (m).")
    max_fov = _prop(float, 1e-4, "Largest field of view accepted (m).")
    max_defocus = _prop(float, 1e-5, "Largest |defocus| accepted (m); render cost grows with defocus squared.")
    # Optics and detector
    acceleration_voltage = _prop(float, 300e3, "Acceleration voltage (V).")
    convergence_angle_mrad = _prop(float, 30.0, "Probe convergence semi-angle (mrad).")
    beam_current_pa = _prop(float, 50.0, "Probe current (pA).")
    haadf_z_exponent = _prop(float, 1.7, "HAADF contrast exponent n in Z^n.")
    haadf_scattering_scale = _prop(float, 2.25e-5, "k in 1-exp(-k * sum Z^n rho t[A]).")
    atomic_pixel_threshold = _prop(float, 0.2e-9, "Below this pixel size (m) atoms are rendered.")
    detector_gain = _prop(float, 1.0, "Counts per detected electron.")
    detector_offset = _prop(float, 0.0, "Detector dark offset (counts).")
    detector_read_noise = _prop(float, 0.0, "Gaussian read noise (counts).")
    # Scan
    scan_flyback_time = _prop(float, 0.0, "Line flyback time (s).")
    scan_line_jitter_px = _prop(float, 0.0, "Per-line x jitter standard deviation (pixels).")
    scan_rotation_deg = _prop(float, 0.0, "Scan rotation relative to the stage (deg).")
    scan_shear = _prop(float, 0.0, "Scan shear (x shift per unit y).")
    scan_flyback_compression = _prop(float, 0.0, "Left-edge compression after flyback (fraction).")
    fov_scale_error = _prop(float, 0.0, "Field-of-view calibration error (fraction).")
    # Stage and drift (stage_move_noise_std comes from the base class)
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
            ("beam_current_pa", float(self.beam_current_pa) > 0, "must be > 0"),
            ("particle_radius_mean", float(self.particle_radius_mean) > 0, "must be > 0"),
            ("sample_size_xy", float(self.sample_size_xy) > 0, "must be > 0"),
            ("particle_elements", any(e.strip() for e in str(self.particle_elements).split(",")), "must name at least one element"),
            ("fov_scale_error", float(self.fov_scale_error) > -1, "must be > -1 (a fraction)"),
            ("eds_channels", int(self.eds_channels) >= 16, "must be >= 16"),
            ("eds_energy_max", float(self.eds_energy_max) > 0, "must be > 0"),
            ("contamination_dose", float(self.contamination_dose) > 0, "must be > 0"),
        ]
        bad = [f"{name} {why} (got {getattr(self, name)!r})" for name, ok, why in rules if not ok]
        if bad:
            raise ValueError("digital twin configuration: " + "; ".join(bad))

    def _generate_sample(self, seed: int) -> None:
        self._check_properties()
        self._rng = EventRng(int(seed))
        self._clock = SimClock(float(self.time_scale))
        self._acq_index = 0
        self._acquisitions: list[dict] = []
        elements = tuple(e.strip() for e in str(self.particle_elements).split(",") if e.strip())
        self._sample = SampleModel(
            SampleParams(
                size_xy_m=float(self.sample_size_xy),
                particle_count=int(self.sample_particle_count),
                radius_mean_m=float(self.particle_radius_mean),
                radius_std_m=float(self.particle_radius_std),
                aspect_ratio=float(self.particle_aspect_ratio),
                elements=elements,
                support_thickness_m=float(self.support_thickness),
                height_gradient=(float(self.height_gradient_x), float(self.height_gradient_y)),
            ),
            self._rng,
        )
        if len(self._sample) < self._sample.requested_count:
            self.warn_stream(
                f"Placed {len(self._sample)} of {self._sample.requested_count} particles: the rest do not fit in "
                f"{float(self.sample_size_xy):.2e} m. Increase sample_size_xy or reduce the count or radius."
            )
        self._stage = StageModel(
            StageParams(
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
            ),
            self._rng,
            self._clock,
        )
        self._renderer = Renderer(
            self._sample,
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
            beam_current_pa=float(self.beam_current_pa),
            gain=float(self.detector_gain),
            offset=float(self.detector_offset),
            read_noise=float(self.detector_read_noise),
        )
        self._eds_params = EdsParams(
            energy_max_ev=float(self.eds_energy_max),
            channels=int(self.eds_channels),
            fwhm_mn_ka_ev=float(self.eds_fwhm_mn_ka),
            solid_angle_sr=float(self.eds_solid_angle),
            dead_time_s=float(self.eds_dead_time),
            voltage_v=float(self.acceleration_voltage),
        )
        x_min, x_max, y_min, y_max = self._sample.bounds
        self._dose = DoseMap(
            DoseParams(
                enabled=bool(self.dose_tracking),
                cell_m=max((x_max - x_min) / 200.0, 2e-9),
                contamination_angstrom_max=float(self.contamination_max) / ANGSTROM,
                contamination_dose_e_per_a2=float(self.contamination_dose),
                damage_critical_dose_e_per_a2=float(self.damage_critical_dose),
            ),
            self._sample.bounds,
            margin_m=(x_max - x_min) * 0.25,
        )
        self._blanked = False
        self._last_written_stage: np.ndarray | None = None
        self._stage_position = self._stage.true.copy()
        # The atom-renderer state of the base class is unused here.
        self._particle_records_base = []
        self._particle_records_view = []
        self._all_sample_elements = sorted(set(elements))
        self._world_bounds_ang = {
            "x_min": x_min / ANGSTROM,
            "x_max": x_max / ANGSTROM,
            "y_min": y_min / ANGSTROM,
            "y_max": y_max / ANGSTROM,
            "z_min": -float(self.sample_size_z) / ANGSTROM / 2,
            "z_max": float(self.sample_size_z) / ANGSTROM / 2,
        }

    def _update_view_cache(self, force: bool = False) -> None:
        """The particle renderer works in sample coordinates; there is no rotated atom copy to refresh."""

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
    def _write_stage_proxy(self, position: np.ndarray) -> None:
        stage = self._detector_proxies.get("stage")
        if stage is None:
            return
        stage.x, stage.y, stage.z, stage.alpha, stage.beta = (float(v) for v in position)
        self._last_written_stage = np.asarray(position, dtype=np.float64).copy()

    def _sync_stage_from_proxy(self) -> None:
        """A position written to the STAGE device directly counts as a stage command."""
        stage = self._detector_proxies.get("stage")
        if stage is None:
            return
        try:
            proxy_position = np.array([stage.x, stage.y, stage.z, stage.alpha, stage.beta], dtype=np.float64)
        except tango.DevFailed:
            self.error_stream("Failed to read stage proxy position; using internal stage state.")
            return
        known = self._last_written_stage if self._last_written_stage is not None else self._stage.reported
        if not np.allclose(proxy_position, known, rtol=0.0, atol=1e-15):
            self._apply_move(proxy_position)

    def _apply_move(self, position) -> None:
        self._stage.move(position)
        self._stage_position = self._stage.true.copy()
        self._write_stage_proxy(self._stage.reported)

    def _move_stage(self, position):
        if len(position) != 5:
            raise ValueError("Stage position must have 5 values: [x, y, z, alpha, beta]")
        target = np.asarray(position, dtype=np.float64)
        if not np.all(np.isfinite(target)):
            raise ValueError(f"Stage position must be finite numbers, got {list(position)}")
        self._check_stage_limits(target)
        self._apply_move(target)

    def _check_stage_limits(self, target: np.ndarray) -> None:
        """Refuse a move the STAGE device would refuse, before the twin's own model moves."""
        stage = self._detector_proxies.get("stage")
        if stage is None:
            return
        for axis, value in zip(("x", "y", "z", "alpha", "beta"), target):
            try:
                config = stage.get_attribute_config(axis)
            except (tango.DevFailed, AttributeError):
                continue
            for bound, outside in ((config.min_value, lambda b: value < b), (config.max_value, lambda b: value > b)):
                try:
                    limit = float(bound)
                except (TypeError, ValueError):  # "Not specified"
                    continue
                if outside(limit):
                    raise ValueError(f"stage {axis} = {value} is outside the stage limits [{config.min_value}, {config.max_value}]")

    def _set_fov(self, fov) -> None:
        fov = float(fov)
        if not (np.isfinite(fov) and float(self.min_fov) <= fov <= float(self.max_fov)):
            raise ValueError(f"field of view must be between {self.min_fov} and {self.max_fov} m, got {fov}")
        self._fov = fov

    def _set_defocus(self, defocus) -> None:
        defocus = float(defocus)
        if not (np.isfinite(defocus) and abs(defocus) <= float(self.max_defocus)):
            raise ValueError(f"defocus must be within +/-{self.max_defocus} m (did you pass nanometres?), got {defocus}")
        self._defocus = defocus

    def _get_stage(self):
        self._sync_stage_from_proxy()
        return self._stage.reported.copy()

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
        return aberrations_nm(coefficients, float(self.acceleration_voltage), float(self._defocus))

    def _scan_frame(self, imsize: int, dwell_time: float, scan_region) -> dict:
        """Lay out one scan: pixel coordinates on the sample, timing, and the ground-truth record."""
        self._sync_stage_from_proxy()
        self._acq_index += 1
        index = self._acq_index
        region = tuple(float(v) for v in scan_region) if scan_region is not None else (0.0, 0.0, 1.0, 1.0)
        # SCAN, AutoScript and the tilt twin all use [left, top, width, height] in fractions of the field.
        if len(region) != 4 or not all(np.isfinite(region)):
            raise ValueError(f"scan_region must be four finite numbers [left, top, width, height], got {list(region)}")
        left, top, width, height = region
        if not (left >= 0 and top >= 0 and width > 0 and height > 0 and left + width <= 1 + 1e-9 and top + height <= 1 + 1e-9):
            raise ValueError(f"scan_region [left, top, width, height] must lie inside the field (0..1), got {list(region)}")
        if not (np.isfinite(imsize) and int(imsize) >= 1 and np.isfinite(dwell_time) and float(dwell_time) > 0):
            raise ValueError(f"imsize must be >= 1 and dwell_time > 0, got {imsize}, {dwell_time}")
        t0 = self._clock.now()
        xs, ys = pixel_sample_coordinates(
            int(imsize),
            float(self._fov),
            (left, top, min(1.0, left + width), min(1.0, top + height)),
            float(dwell_time),
            self._scan_params,
            self._stage.true_xy_at,
            t0,
            self._rng.for_event("scan", index),
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
        }
        self._acquisitions.append(record)
        return {"xs": xs, "ys": ys, "index": index, "pixel_m": pixel, "dwell_s": float(dwell_time), "record": record}

    def _render_stem_image(self, imsize: int, dwell_time: float, detector_list: list, scan_region=None, frame=None) -> np.ndarray:
        """Render one detector channel; pass ``frame`` to share one scan between detectors."""
        if frame is None:
            frame = self._scan_frame(int(imsize), float(dwell_time), scan_region)
        detector = str(detector_list[0]).upper()
        xs, ys = frame["xs"], frame["ys"]
        cx, cy = float(xs.mean()), float(ys.mean())
        visible = self._sample.in_box(float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max()))
        fraction = self._renderer.render(
            xs,
            ys,
            frame["pixel_m"],
            detector,
            self._aberrations(),
            support_angstrom=float(self.support_thickness) / ANGSTROM + self._dose.contamination_angstrom(cx, cy),
            damage=self._dose.damage_factors(self._sample.centers, visible),
        )
        rng = self._rng.for_event(f"detector_{detector}", frame["index"])
        params = replace(self._detector_params, beam_current_pa=self._beam_current_pa())
        return count(fraction, frame["dwell_s"], params, rng)

    def _acquire_scanned_image(
        self,
        imsize: int,
        dwell_time: float,
        detector_list: list[str] = ["haadf"],
        scan_region: list[float] = [0.0, 0.0, 1.0, 1.0],
        output_format: str = ".h5",
    ) -> str:
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
        return self._stage.true_xy_at(t) + np.array([x, y])

    def _areal_density(self, xy: np.ndarray) -> dict[str, float]:
        """Atoms per square angstrom of each element along the beam at a sample point."""
        density: dict[str, float] = {}
        x, y = float(xy[0]), float(xy[1])
        for i in self._sample.in_box(x, x, y, y):
            t = float(self._sample.thickness_at(int(i), np.array(x), np.array(y)))
            if t > 0:
                element = self._sample.elements[int(i)]
                density[element] = density.get(element, 0.0) + ELEMENTS[element][1] * t / ANGSTROM
        support = float(self.support_thickness) / ANGSTROM + self._dose.contamination_angstrom(x, y)
        if support > 0:
            element = self._sample.params.support_element
            density[element] = density.get(element, 0.0) + ELEMENTS[element][1] * support
        return density

    def _acquire_spectrum(self, detector_name: str, exposure_time: float) -> str:
        self._sync_stage_from_proxy()
        self._acq_index += 1
        index = self._acq_index
        real_time = float(exposure_time)
        t0 = self._clock.now()
        xy = self._beam_sample_xy(t0 + real_time / 2.0)
        energies, counts, live_time = point_spectrum(
            self._areal_density(xy),
            self._beam_current_pa(),
            real_time,
            self._eds_params,
            self._rng.for_event("eds", index),
        )
        electrons = self._beam_current_pa() * 1e-12 * real_time / 1.602176634e-19
        self._dose.add_point(float(xy[0]), float(xy[1]), electrons, 1e-20)
        self._clock.advance(real_time)
        self._acquisitions.append(
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
        # The seed regenerates the sample and the count is part of the answer: export_ground_truth
        # (operator only) has both; an agent must find particles by imaging.
        metadata.pop("sample_seed", None)
        metadata.pop("particle_count", None)
        return metadata

    def _get_parameters(self) -> str:
        parameters = json.loads(super()._get_parameters())
        parameters["stage_position"] = self._stage.reported.tolist()
        parameters["scan_detectors"] = sorted(d.lower() for d in DETECTOR_SCALE)
        parameters["simulated_time_s"] = self._clock.now()
        parameters["time_scale"] = self._clock.time_scale
        parameters["beam_current_pa"] = self._detector_params.beam_current_pa
        parameters["beam_blanked"] = self._blanked
        return json.dumps(parameters)

    @command(dtype_out=str)
    def export_ground_truth(self) -> str:
        """Hidden truth for scoring workflows: particles, true stage pose and drift of every acquisition.

        Blocked from agents in the MCP configs; tests and calibration scripts call it through Tango.
        """
        return json.dumps(
            {
                "sample_seed": int(self.sample_seed),
                "sample_bounds_m": list(self._sample.bounds),
                "particles": self._sample.ground_truth(),
                "moves": [vars(m) for m in self._stage.history],
                "acquisitions": self._acquisitions,
                "simulated_time_s": self._clock.now(),
                "dose": self._dose.summary(),
            }
        )


if __name__ == "__main__":
    DigitalTwin.run_server()
