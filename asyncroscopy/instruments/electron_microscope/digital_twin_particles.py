"""
Particle digital twin for HAADF-EDX.

Useful for testing and development without requiring AutoScript hardware.
"""

from __future__ import annotations

import numpy as np
import pyTEMlib.image_tools as it
import pyTEMlib.probe_tools as pt

from tango import AttrWriteType
from tango.server import attribute, device_property

from asyncroscopy.data.data_writer import save_acquisition
from asyncroscopy.instruments.electron_microscope.digital_twin_base import ElectronMicroscopeTwin
from asyncroscopy.instruments.electron_microscope.twin.sample_model import SampleParams
from asyncroscopy.instruments.electron_microscope.twin.specimen import SpecimenContext, ViewRequest
from asyncroscopy.routines.dose import ELECTRON_CHARGE
from asyncroscopy.sim.rng import EventRng


class DigitalTwinParticles(ElectronMicroscopeTwin):
    """
    Detector-specific settings (dwell time, resolution) are stored in
    dedicated detector devices and read via DeviceProxy at acquisition time.
    """

    render_mode = "particle_potential"
    specimen_source = device_property(dtype=str, default_value="particles", doc="Specimen source (particles, or a plug-in in the particle_potential render mode).")

    # ------------------------------------------------------------------
    # Attributes
    # ------------------------------------------------------------------
    # not finishded
    manufacturer = attribute(
        label="DigitalTwinParticles",
        dtype=str,
        doc="Simulation backend",
    )

    beam_pos = attribute(
        label="Beam Position",
        dtype=(float,),        # 1D array of floats
        max_dim_x=2,           # exactly 2 elements: [x, y]
        access=AttrWriteType.READ_WRITE,
        unit="fractional",
        min_value=0.0,
        max_value=1.0,
        doc="Beam position as [x, y] fractional coordinates, each in range [0.0, 1.0]",
    )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_device(self) -> None:
        """Initialize the current digital-twin contract and particle world."""
        super().init_device()
        self._manufacturer = "UTKTeam Particle Digital Twin"
        self._particle_records = []

    def _generate_sample(self, seed: int) -> None:
        """Build the particle world (the shell connects the support devices first)."""
        self._load_specimen(seed, SpecimenContext(rng=EventRng(int(seed)), sample=SampleParams(particle_count=int(self.sample_particle_count)), fov_m=float(self._fov)))
        self._read_stage()

    def _read_stage(self) -> None:
        """Take the stage position from the STAGE device."""
        stage = self._detector_proxies.get("stage")
        if stage is not None:
            self._stage_position = np.asarray([stage.x, stage.y, stage.z, stage.alpha, stage.beta], dtype=np.float64)

    def _viewport_metadata(self) -> dict:
        return {**super()._viewport_metadata(), "particle_count": len(self._particle_records)}


    # ------------------------------------------------------------------
    # Attribute read methods
    # ------------------------------------------------------------------

    def read_manufacturer(self) -> str:
        return self._manufacturer


    def read_beam_pos(self) -> list[float]:
        """Return beam position as [x, y] fractional coordinates."""
        return [self._beam_pos_x, self._beam_pos_y]

    # --- Write Method ---

    def write_beam_pos(self, value: list[float]) -> None:
        """Set beam position from [x, y] fractional coordinates."""
        x, y = value[0], value[1]

        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(
                f"beam_pos values must be in [0.0, 1.0], got x={x}, y={y}"
            )

        self._beam_pos_x = x
        self._beam_pos_y = y


    # ------------------------------------------------------------------
    # Internal acquisition helpers
    # ------------------------------------------------------------------
    def _acquire_scanned_image(
        self,
        imsize: int,
        dwell_time: float,
        detector_list: list[str] = ["haadf"],
        scan_region: list[float] = [0.0, 0.0, 1.0, 1.0],
        output_format: str = ".h5",
    ) -> str:
        """
        Acquire a simulated STEM image using the pre-cooked sample state.
        """
        self._read_stage()
        size = imsize
        self._imsize = imsize
        fov      = self._fov * 1e10          # Å
        edge_crop = 20
        beam_current     = 1000              # pA
        blur_noise_level = 0.1
        pixel_size       = fov / size

        # ── Probe aberrations (unchanged) ─────────────────────────────────────────
        ab = pt.get_target_aberrations("Spectra300", 200000)
        ab['acceleration_voltage'] = 200e3
        ab['FOV']                  = fov / 10   # nm
        ab['convergence_angle']    = 30         # mrad
        ab['wavelength']           = it.get_wavelength(ab['acceleration_voltage'])

        random = self._specimen.random  # one stream with the sample, so a rebuild and what follows it repeat

        # ── Helper functions (unchanged from original) ────────────────────────────
        def sub_pix_gaussian(size=10, sigma=0.2, dx=0.0, dy=0.0):
            coords = np.arange(size) - (size - 1) / 2.0
            x, y   = np.meshgrid(coords, coords)
            g      = np.exp(-(((x + dx)**2 + (y + dy)**2) / (2 * sigma**2)))
            return g / g.max()

        def create_pseudo_potential(xtal, pixel_size, sigma, bounds, atom_frame=11):
            x_min, x_max = bounds[0], bounds[1]
            y_min, y_max = bounds[2], bounds[3]
            pixels_x = int((x_max - x_min) / pixel_size)
            pixels_y = int((y_max - y_min) / pixel_size)
            potential_map = np.zeros((pixels_x, pixels_y))
            padding       = atom_frame
            potential_map = np.pad(potential_map, padding, mode='constant', constant_values=0.0)

            atomic_numbers = xtal.get_atomic_numbers()
            positions      = xtal.get_positions()[:, :2]

            mask = (
                (positions[:, 0] >= x_min) & (positions[:, 0] < x_max) &
                (positions[:, 1] >= y_min) & (positions[:, 1] < y_max)
            )
            positions      = positions[mask]
            atomic_numbers = atomic_numbers[mask]

            for pos, Z in zip(positions, atomic_numbers):
                x, y   = np.round(pos / pixel_size)
                dx, dy = pos - np.round(pos)
                atom   = sub_pix_gaussian(size=atom_frame, sigma=sigma, dx=dx, dy=dy) * Z
                potential_map[
                    int(x + padding + dx - padding//2 - 1):int(x + padding + dx + padding//2),
                    int(y + padding + dy - padding//2 - 1):int(y + padding + dy + padding//2)
                ] += atom

            potential_map = potential_map[padding:-padding, padding:-padding]
            return potential_map / np.max(potential_map)

        def poisson_noise(image, counts=1e9):
            image = image - image.min()
            image = image / image.sum()
            noisy = random.poisson(image * counts).astype(float)
            noisy = noisy - noisy.min()
            return noisy / noisy.max()

        def lowfreq_noise(image, noise_level=0.1, freq_scale=0.1):
            size_x, size_y = image.shape
            noise      = random.normal(0, noise_level, (size_x, size_y))
            noise_fft  = np.fft.fft2(noise)
            x_freqs    = np.fft.fftfreq(size_x)
            y_freqs    = np.fft.fftfreq(size_y)
            freq_filter = np.outer(
                np.exp(-np.square(x_freqs) / (2 * freq_scale**2)),
                np.exp(-np.square(y_freqs) / (2 * freq_scale**2)),
            )
            noisy = np.fft.ifft2(noise_fft * freq_filter).real
            noisy = noisy - noisy.min()
            return noisy / noisy.max()

        # ── This view of the specimen: its atoms and the particles visible ──────────
        field = self._specimen.view(ViewRequest(stage=self._stage_position, fov_angstrom=fov, imsize=size, edge_crop=edge_crop))
        xtal = field.atoms
        self._particle_records = field.records

        # ── Pseudo-potential + PSF convolution (unchanged logic) ──────────────────
        edge     = 2 * edge_crop * pixel_size
        frame    = (0, fov + edge, 0, fov + edge)

        # create_pseudo_potential masks the atoms by position, to the padded frame.
        potential = create_pseudo_potential(
            xtal, pixel_size, sigma=1, bounds=frame, atom_frame=11
        )

        probe, A_k, chi = pt.get_probe(
            ab, size + 2*edge_crop, size + 2*edge_crop, verbose=True
        )

        psf_shifted = np.fft.ifftshift(probe)
        image = np.fft.ifft2(np.fft.fft2(potential) * np.fft.fft2(psf_shifted))
        image = np.absolute(image)
        image = image[edge_crop:-edge_crop, edge_crop:-edge_crop]

        # ── Noise chain (unchanged) ───────────────────────────────────────────────
        scan_time   = dwell_time * size * size
        counts      = scan_time * (beam_current * 1e-12) / ELECTRON_CHARGE
        noisy_image = poisson_noise(image, counts=counts)
        blur_noise  = lowfreq_noise(noisy_image, noise_level=0.1, freq_scale=0.1)
        noisy_image += blur_noise * blur_noise_level

        image = np.array(noisy_image, dtype=np.float32)
        detectors = [detector.upper() for detector in detector_list]
        images = [image.copy() for _detector in detectors]
        return save_acquisition(
            self,
            self._detector_proxies.get("data"),
            "stem_image",
            detectors,
            images,
            dataset_attrs={
                "sample_type": "oriented_particles",
                "particle_count": field.info["particle_count"],
                "world_shape": field.info["world_shape"],
                "voxel_size_angstrom": field.info["voxel_size_angstrom"],
                "stage_position": list(self._get_stage()),
            },
            output_format=output_format,
        )

    def _acquire_spectrum(self, detector_name: str, exposure_time: float) -> str:
        random = self._specimen.random
        px, py = self.read_beam_pos()   # fractional [0, 1]
        px_pix = px * self._imsize
        py_pix = py * self._imsize

        for rec in self._particle_records:
            cx, cy = rec['center']   # pixels
            r      = rec['radius']   # pixels
            if (px_pix - cx)**2 + (py_pix - cy)**2 <= r**2:
                raw   = {el: frac for el, frac in rec['composition'].items()}
                total = sum(raw.values())
                spectrum = {
                    el: max(0.0, v / total + random.normal(0.01, 0.1))
                    for el, v in raw.items()
                }
                break
        else:
            all_elements = {
                el for rec in self._particle_records for el in rec['composition']
            }
            spectrum = {
                el: float(np.abs(random.normal(0, 0.05)))
                for el in sorted(all_elements)
            }

        elements = sorted(spectrum)
        return save_acquisition(
            self,
            self._detector_proxies.get("data"),
            "spectrum",
            detector_name,
            np.asarray([spectrum[element] for element in elements], dtype=np.float64),
            dataset_name="spectrum",
            dataset_attrs={"elements": elements, "exposure_time": float(exposure_time)},
        )


    def _place_beam(self, position) -> None:
        """
        sets resting beam position, [0:1]
        """
        x, y = position
        self.write_beam_pos([x, y])


    def _set_fov(self, fov) -> None:
        """set field of view in meters"""
        self._fov = float(fov)
        self._specimen.set_field(self._fov)
        self._world_bounds_ang = dict(self._specimen.world_bounds_ang)
        self._read_stage()

    def _get_fov(self) -> float:
        return float(self._fov)

    def _set_defocus(self, defocus) -> None:
        """Set defocus in meters."""
        self._defocus = float(defocus)

    def _get_defocus(self) -> float:
        """Get defocus in meters."""
        return self._defocus


    def _get_stage(self):
        """Return current stage position as [x, y, z, alpha, beta], with tilts in degrees."""
        return self._stage_position
    
    def _move_stage(self, position):
        """Move stage to [x, y, z, alpha, beta], with x/y/z in meters and tilts in degrees."""
        super()._move_stage(position)
        self._read_stage()

# ----------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    DigitalTwinParticles.run_server()
