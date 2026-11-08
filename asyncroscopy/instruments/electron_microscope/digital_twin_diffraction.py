"""Digital twin for parked-beam nanoparticle diffraction."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from tango.server import device_property

from asyncroscopy.data.data_writer import save_acquisition
from asyncroscopy.instruments.electron_microscope.digital_twin_base import DEFAULT_ACQUISITION_DIR, ElectronMicroscopeTwin
from asyncroscopy.instruments.electron_microscope.twin.atomistic import ParticleSpectrumMixin
from asyncroscopy.instruments.electron_microscope.twin.specimen import ViewRequest
from asyncroscopy.instruments.electron_microscope.twin.specimen_nanoparticle_map import FOV_M, FOV_NM

DEFAULT_OVERVIEW_IMAGE_SIZE = 1024
DIFFRACTION_MAX_ANGLE_MRAD = 80.0
VACUUM_LATERAL_SIZE_ANGSTROM = 40.0


class DigitalTwinDiffraction(ParticleSpectrumMixin, ElectronMicroscopeTwin):
    """Digital twin that pairs a HAADF nanoparticle overview (the ``nanoparticle_map`` specimen) with local diffraction."""

    render_mode = "diffraction_map"
    specimen_source = device_property(dtype=str, default_value="nanoparticle_map", doc="Specimen source (nanoparticle_map, or a plug-in in the diffraction_map render mode).")
    acquisition_save_directory = device_property(
        dtype=str,
        default_value=DEFAULT_ACQUISITION_DIR,
        doc='Directory where simulated acquisitions are saved before the Tiled server serves them.',
    )
    diffraction_image_size = device_property(
        dtype=int,
        default_value=128,
        doc='Diffraction image size used when the camera device does not provide one.',
    )
    overview_image_size = device_property(
        dtype=int,
        default_value=DEFAULT_OVERVIEW_IMAGE_SIZE,
        doc='Fixed HAADF overview image size in pixels.',
    )
    convergence_angle_mrad = device_property(
        dtype=float,
        default_value=5.0,
        doc='Probe convergence semi-angle in mrad for abTEM diffraction simulations.',
    )
    haadf_poisson_counts = device_property(
        dtype=float,
        default_value=120.0,
        doc='Mean per-pixel count scale used for Poisson noise in the HAADF overview.',
    )

    def init_device(self) -> None:
        super().init_device()
        self._manufacturer = 'UTKTeam Diffraction Twin'
        self._fov = FOV_M

    def _generate_sample(self, seed: int) -> None:
        self._fov = FOV_M
        self._load_specimen(seed)

    def _render_stem_image(self, imsize: int, dwell_time: float, detector_list: list) -> np.ndarray:
        self._sync_stage_from_proxy()
        self._imsize = imsize
        image = self._specimen.view(ViewRequest(stage=self._stage_position, imsize=int(imsize))).maps["haadf"]
        seed = int(abs(hash((int(self.sample_seed), tuple(np.round(self._stage_position, 10)), int(imsize)))) % (2**32))
        rng = np.random.default_rng(seed)
        counts = max(1.0, float(self.haadf_poisson_counts))
        image = rng.poisson(np.clip(image, 0.0, 1.0) * counts).astype(np.float32) / counts
        image += rng.normal(0.0, 0.008, size=image.shape).astype(np.float32)
        return np.clip(image, 0.0, 1.0)

    def _beam_atoms(self):
        """The atoms under the parked beam, or None over vacuum; their info describes the particle."""
        self._sync_stage_from_proxy()
        px, py = self.read_beam_pos()
        stage_nm = self._stage_position[:2] * 1e9
        x_nm = (float(px) - 0.5) * FOV_NM + stage_nm[0]
        y_nm = (float(py) - 0.5) * FOV_NM + stage_nm[1]
        seed = self._particle_seed(int(self.sample_seed), self._stage_position, self.read_beam_pos())
        return self._specimen.atoms_at(x_nm, y_nm, ViewRequest(stage=self._stage_position), seed)

    @staticmethod
    def _particle_seed(sample_seed: int, stage_position: np.ndarray, beam_position: list[float]) -> int:
        return int(abs(hash((int(sample_seed), tuple(np.round(stage_position, 10)), tuple(np.round(beam_position, 6))))) % (2**32))

    def _vacuum_diffraction(self, imsize: int) -> np.ndarray:
        cell = [VACUUM_LATERAL_SIZE_ANGSTROM, VACUUM_LATERAL_SIZE_ANGSTROM, 10.0]
        atoms = Atoms(cell=cell, pbc=(False, False, False))
        return self._simulate_diffraction_from_atoms(atoms, imsize)

    def _simulate_diffraction_from_atoms(self, atoms, imsize: int) -> np.ndarray:
        try:
            import abtem
        except ImportError as exc:
            raise RuntimeError(
                "abTEM is required for diffraction simulation. "
                "Install it with `uv sync --extra diffraction`."
            ) from exc

        potential = abtem.Potential(atoms, sampling=0.08, slice_thickness=1.0)
        probe = abtem.Probe(energy=200e3, semiangle_cutoff=float(self.convergence_angle_mrad))
        probe.grid.match(potential)
        exit_wave = probe.multislice(potential)
        pattern = exit_wave.diffraction_patterns(max_angle=DIFFRACTION_MAX_ANGLE_MRAD).compute()
        array = np.asarray(getattr(pattern, 'array', pattern), dtype=np.float32)
        array = np.squeeze(array)
        if array.ndim > 2:
            array = array.reshape((-1,) + array.shape[-2:])[0]
        return self._resize_nearest(array, imsize)

    @staticmethod
    def _resize_nearest(array: np.ndarray, imsize: int) -> np.ndarray:
        if array.shape == (imsize, imsize):
            result = array.astype(np.float32)
        else:
            yi = np.rint(np.linspace(0, array.shape[0] - 1, imsize)).astype(int)
            xi = np.rint(np.linspace(0, array.shape[1] - 1, imsize)).astype(int)
            result = array[np.ix_(yi, xi)].astype(np.float32)
        result -= float(result.min())
        max_val = float(result.max())
        return result / max_val if max_val > 0.0 else result

    def _acquire_camera_image(
        self,
        imsize: int,
        exposure_time: float,
        detector: str,
        readout_area: str,
        frame_combining: int = 1,
        output_format: str = '.h5',
    ) -> str:
        atoms = self._beam_atoms()
        rattle_value = float(atoms.info["rattle_value"]) if atoms is not None else 0.0
        image_size = int(imsize or self.diffraction_image_size)
        diffraction = self._vacuum_diffraction(image_size) if atoms is None else self._simulate_diffraction_from_atoms(atoms, image_size)
        data_server = self._detector_proxies.get('data')
        attrs = {
            'pixel_size_mrad': float(2.0 * DIFFRACTION_MAX_ANGLE_MRAD / image_size),
            'pixel_size_rad': float(2.0 * DIFFRACTION_MAX_ANGLE_MRAD * 1e-3 / image_size),
            'max_angle_mrad': float(DIFFRACTION_MAX_ANGLE_MRAD),
            'rattle_value': float(rattle_value),
            'beam_position_x': float(self._beam_pos_x),
            'beam_position_y': float(self._beam_pos_y),
            'exposure_time': float(exposure_time),
            'readout_area': str(readout_area),
            'frame_combining': int(frame_combining),
        }
        if atoms is not None:
            particle = atoms.info
            attrs.update(
                {
                    'particle_id': int(particle['particle_id']),
                    'particle_kind': str(particle['particle_kind']),
                    'particle_lattice_parameter_angstrom': float(particle['lattice_parameter']),
                    'particle_center_lattice_parameter_angstrom': float(particle['center_lattice_parameter']),
                    'particle_lattice_strain_fraction': float(particle['lattice_strain_fraction']),
                    'particle_lattice_strain_percent': float(100.0 * particle['lattice_strain_fraction']),
                    'particle_rotation_degrees': float(particle['rotation_degrees']),
                    'particle_zone_axis': '<100>',
                }
            )
        return save_acquisition(
            self,
            data_server,
            'diffraction',
            str(detector),
            diffraction,
            dataset_name='image',
            dataset_attrs=attrs,
            output_format=output_format,
        )

    def _acquire_scanned_image(
        self,
        imsize: int,
        dwell_time: float,
        detector_list: list[str] = ['haadf'],
        scan_region: list[float] = [0.0, 0.0, 1.0, 1.0],
        output_format: str = '.h5',
    ) -> str:
        image_size = int(self.overview_image_size)
        detector_list = [detector.upper() for detector in detector_list]
        data_server = self._detector_proxies.get('data')
        attrs = {
            'pixel_size_nm': float(FOV_NM / image_size),
            'pixel_size_m': float(FOV_M / image_size),
            'fov_nm': float(FOV_NM),
            'fov_m': float(FOV_M),
            'map_size_px': self._specimen.map_size,
            'sample_pixel_size_nm': float(FOV_NM / self._specimen.map_size),
        }
        images = [self._render_stem_image(image_size, float(dwell_time), [detector]) for detector in detector_list]
        return save_acquisition(self, data_server, 'stem_image', detector_list, images, dataset_attrs=attrs, output_format=output_format)

    def _set_fov(self, fov) -> None:
        self.warn_stream('DigitalTwinDiffraction uses a fixed 500 nm field of view.')
        self._fov = FOV_M

    def _get_fov(self) -> float:
        return FOV_M


if __name__ == '__main__':
    DigitalTwinDiffraction.run_server()
