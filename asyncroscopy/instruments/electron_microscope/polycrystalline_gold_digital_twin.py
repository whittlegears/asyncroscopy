"""Atomistic polycrystalline gold digital twin."""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pyTEMlib.image_tools as image_tools
import pyTEMlib.probe_tools as probe_tools
from tango.server import device_property

from asyncroscopy.utils.commands import command
from asyncroscopy.data.data_writer import save_acquisition
from asyncroscopy.instruments.electron_microscope.digital_twin_base import ElectronMicroscopeTwin
from asyncroscopy.instruments.electron_microscope.twin.atomistic import ParticleSpectrumMixin
from asyncroscopy.instruments.electron_microscope.twin.specimen import ViewRequest
from asyncroscopy.routines.dose import ELECTRON_CHARGE

# The share of the probe's electrons the annular detector counts, which sets the shot noise of an image
# (more electrons, less noise). An HAADF detector collects a few per cent of the beam; 1 % is this twin's choice.
HAADF_COLLECTED_FRACTION = 0.01


class PolycrystallineGoldDigitalTwin(ParticleSpectrumMixin, ElectronMicroscopeTwin):
    """Atomistic grains (the ``polycrystal`` specimen) rendered through a digital-twin corrector probe."""

    render_mode = "grain_potential"
    specimen_source = device_property(dtype=str, default_value="polycrystal", doc="Specimen source (polycrystal, or a plug-in in the grain_potential render mode).")
    acceleration_voltage_ev = device_property(dtype=float, default_value=200_000.0)
    convergence_angle_mrad = device_property(dtype=float, default_value=30.0)
    probe_current_pa = device_property(dtype=float, default_value=100.0, doc="Probe current (pA).")
    blur_noise_level = device_property(dtype=float, default_value=0.5)

    def init_device(self) -> None:
        super().init_device()
        self._fov = 20e-9
        self._manufacturer = "UTKTeam Atomistic Polycrystalline Gold Digital Twin"

    def _generate_sample(self, seed: int) -> None:
        self._load_specimen(seed)
        self._last_rendered_atom_count = 0
        self._raw_potential_cache = None
        self._raw_potential_cache_key = None
        self._raw_potential_cache_atom_count = 0
        self._raw_potential_cache_hits = 0
        details = {k: v for k, v in self._specimen.describe().items() if k not in ("source", "bounds_m", "element")}
        self._sample_metadata = {"sample_type": "atomistic_polycrystalline_gold_slab", **details, "corrector_backend": "DigitalTwinCorrector", "probe_backend": "pyTEMlib.probe_tools.get_probe", "acceleration_voltage_ev": float(self.acceleration_voltage_ev), "convergence_angle_mrad": float(self.convergence_angle_mrad)}

    def _corrector_coefficients(self) -> dict:
        corrector = self._detector_proxies.get("corrector")
        if corrector is None:
            raise RuntimeError("PolycrystallineGoldDigitalTwin requires the corrector digital twin")
        info = json.loads(corrector.get_info()).get("result", {})
        if not info.get("simulation") or info.get("model") != "DigitalTwinCorrector":
            raise RuntimeError("PolycrystallineGoldDigitalTwin requires DigitalTwinCorrector")
        coefficients = json.loads(corrector.get_aberrations_coeff_sim())
        if not coefficients:
            raise RuntimeError("DigitalTwinCorrector returned no aberrations")
        return coefficients

    def _set_defocus(self, defocus) -> None:
        coefficients = self._corrector_coefficients()
        coefficients["C1"] = [float(defocus)]
        self._detector_proxies["corrector"].set_aberrations_coeff_sim(json.dumps(coefficients))
        self._defocus = float(defocus)

    def _get_defocus(self) -> float:
        coefficients = self._corrector_coefficients()
        return float(coefficients.get("C1", [self._defocus])[0])

    def _raw_sample_potential(self, imsize: int) -> np.ndarray:
        self._sync_stage_from_proxy()
        self._imsize = int(imsize)
        fov_angstrom = float(self._fov) * 1e10
        pixel_size_angstrom = fov_angstrom / imsize
        edge_crop = max(12, int(round(0.06 * imsize)))
        padded_size = imsize + 2 * edge_crop
        padded_fov_angstrom = pixel_size_angstrom * padded_size
        cache_key = (int(imsize), round(fov_angstrom, 9), tuple(np.round(self._stage_position, 12)))
        if cache_key == self._raw_potential_cache_key:
            self._raw_potential_cache_hits += 1
            self._last_rendered_atom_count = self._raw_potential_cache_atom_count
            return self._raw_potential_cache
        field = self._specimen.view(ViewRequest(stage=self._stage_position, fov_angstrom=padded_fov_angstrom))
        atom_positions = field.atoms.get_positions()
        x_min, y_min = field.info["window_angstrom"]
        self._last_rendered_atom_count = len(atom_positions)
        atom_frame = 11
        padding = atom_frame
        potential = np.zeros((padded_size + 2 * padding, padded_size + 2 * padding), dtype=np.float32)
        pixel_coordinates = (atom_positions[:, :2] - np.array([x_min, y_min])) / pixel_size_angstrom
        rounded_coordinates = np.round(pixel_coordinates)
        fractional_offsets = pixel_coordinates - rounded_coordinates
        stamp_coordinates = np.arange(atom_frame) - (atom_frame - 1) / 2.0
        stamp_x, stamp_y = np.meshgrid(stamp_coordinates, stamp_coordinates)
        for rounded, offset, atomic_number in zip(rounded_coordinates.astype(int), fractional_offsets, field.atoms.numbers):
            gaussian = np.exp(-((stamp_x + offset[0]) ** 2 + (stamp_y + offset[1]) ** 2) / 2.0)
            gaussian /= gaussian.max()
            start_x = rounded[0] + padding - atom_frame // 2
            start_y = rounded[1] + padding - atom_frame // 2
            potential[start_x:start_x + atom_frame, start_y:start_y + atom_frame] += gaussian * float(atomic_number)
        potential = potential[padding:-padding, padding:-padding]
        if potential.max() > 0:
            potential /= potential.max()
        self._raw_potential_cache = potential
        self._raw_potential_cache_key = cache_key
        self._raw_potential_cache_atom_count = self._last_rendered_atom_count
        return potential

    def _render_stem_image(self, imsize: int, dwell_time: float) -> np.ndarray:
        potential = self._raw_sample_potential(imsize)
        self._imsize = int(imsize)
        fov_angstrom = float(self._fov) * 1e10
        pixel_size_angstrom = fov_angstrom / imsize
        edge_crop = max(12, int(round(0.06 * imsize)))
        padded_size = imsize + 2 * edge_crop
        padded_fov_angstrom = pixel_size_angstrom * padded_size
        coefficients = self._corrector_coefficients()
        aberrations = probe_tools.get_target_aberrations("Spectra300", int(self.acceleration_voltage_ev))
        mappings = {"C1": ("C10",), "A1": ("C12a", "C12b"), "B2": ("C21a", "C21b"), "A2": ("C23a", "C23b"), "C3": ("C30",), "S3": ("C32a", "C32b"), "A3": ("C34a", "C34b"), "D4": ("C41a", "C41b"), "B4": ("C43a", "C43b"), "A4": ("C45a", "C45b")}
        for source, destinations in mappings.items():
            values = np.atleast_1d(coefficients[source]).astype(float)
            if len(values) != len(destinations):
                raise ValueError(f"Corrector coefficient {source} has {len(values)} value(s); expected {len(destinations)}")
            for destination, value_m in zip(destinations, values):
                aberrations[destination] = float(value_m * 1e9)
        aberrations["acceleration_voltage"] = float(self.acceleration_voltage_ev)
        aberrations["FOV"] = padded_fov_angstrom / 10.0
        aberrations["convergence_angle"] = float(self.convergence_angle_mrad)
        aberrations["wavelength"] = image_tools.get_wavelength(aberrations["acceleration_voltage"])
        probe, _aperture, _chi = probe_tools.get_probe(aberrations, padded_size, padded_size, verbose=False)
        image = np.fft.ifft2(np.fft.fft2(potential) * np.fft.fft2(np.fft.ifftshift(probe)))
        image = np.absolute(image)[edge_crop:-edge_crop, edge_crop:-edge_crop]

        scan_time = dwell_time * imsize * imsize
        counts = scan_time * (float(self.probe_current_pa) * 1e-12) / ELECTRON_CHARGE * HAADF_COLLECTED_FRACTION
        pose_seed = abs(hash((int(self.sample_seed), int(imsize), round(fov_angstrom, 6), tuple(np.round(self._stage_position, 12))))) % (2**32)
        rng = np.random.default_rng(pose_seed)
        image -= image.min()
        image /= image.sum() if image.sum() > 0 else 1.0
        noisy = rng.poisson(image * counts).astype(np.float32)
        noisy -= noisy.min()
        noisy /= noisy.max() if noisy.max() > 0 else 1.0
        noise = rng.normal(0.0, 0.1, noisy.shape)
        noise_fft = np.fft.fft2(noise)
        frequencies = np.fft.fftfreq(imsize)
        frequency_filter = np.outer(np.exp(-np.square(frequencies) / (2.0 * 0.5**2)), np.exp(-np.square(frequencies) / (2.0 * 0.5**2)))
        blur_noise = np.fft.ifft2(noise_fft * frequency_filter).real
        blur_noise -= blur_noise.min()
        blur_noise /= blur_noise.max() if blur_noise.max() > 0 else 1.0
        return np.asarray(noisy + blur_noise * float(self.blur_noise_level), dtype=np.float32)

    def _acquire_scanned_image(self, imsize: int, dwell_time: float, detector_list: list[str] = ["haadf"], scan_region: list[float] = [0.0, 0.0, 1.0, 1.0], output_format: str = ".h5") -> str:
        detector_list = [detector.upper() for detector in detector_list]
        images = [self._render_stem_image(int(imsize), float(dwell_time)) for _detector in detector_list]
        metadata = {**self._sample_metadata, "last_rendered_atom_count": self._last_rendered_atom_count, "raw_potential_cache_hits": self._raw_potential_cache_hits, "defocus_m": self._get_defocus(), "corrector_aberrations": self._corrector_coefficients()}
        attrs = [metadata.copy() for _image in images]
        return save_acquisition(self, self._detector_proxies.get("data"), "stem_image", detector_list, images, dataset_attrs=attrs, file_attrs=metadata, output_format=output_format)

    @command(dtype_out=str, risk="low", role="setting.read")
    def get_volume_metadata(self) -> str:
        """The rendered sample's metadata as JSON: grains, atom count, defocus and corrector aberrations."""
        metadata = {**self._sample_metadata, "last_rendered_atom_count": self._last_rendered_atom_count, "raw_potential_cache_hits": self._raw_potential_cache_hits, "defocus_m": self._get_defocus(), "corrector_aberrations": self._corrector_coefficients()}
        return json.dumps(metadata)

    @command(dtype_out=str, risk="low", role="setting.read")
    def get_region_parameters(self) -> str:
        """The sample's regions (grain orientation, thickness, ...) as a JSON list."""
        return json.dumps([asdict(parameter) for parameter in self._specimen.regions()])


if __name__ == "__main__":
    PolycrystallineGoldDigitalTwin.run_server()
