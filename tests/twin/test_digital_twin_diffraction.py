from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from asyncroscopy.instruments.electron_microscope.digital_twin_diffraction import DigitalTwinDiffraction
from asyncroscopy.instruments.electron_microscope.twin.specimen import ViewRequest
from asyncroscopy.instruments.electron_microscope.twin.specimen_nanoparticle_map import (
    EDGE_LATTICE_CHANGE_FRACTION,
    FOV_M,
    OVAL_CENTER_LATTICE_MEAN_ANGSTROM,
    ROUND_CENTER_LATTICE_MEAN_ANGSTROM,
    local_lattice_parameter,
    rattle_stdev,
)


def twin_for_path(tmp_path: Path) -> DigitalTwinDiffraction:
    twin = DigitalTwinDiffraction.__new__(DigitalTwinDiffraction)
    twin._tango_properties = {}
    twin._detector_proxies = {}
    twin._stage_position = np.zeros(5, dtype=np.float64)
    twin._beam_pos_x = 0.5
    twin._beam_pos_y = 0.5
    twin._fov = FOV_M
    twin.sample_seed = 12345
    twin.specimen = '{"source": "nanoparticle_map", "map_size": 256}'
    twin.specimen_source = "nanoparticle_map"
    twin.overview_image_size = 64
    twin.diffraction_image_size = 64
    twin.convergence_angle_mrad = 5.0
    twin.haadf_poisson_counts = 120.0
    twin.acquisition_save_directory = str(tmp_path)
    return twin


def test_diffraction_twin_generates_dense_nanoparticle_field(tmp_path: Path):
    twin = twin_for_path(tmp_path)

    twin._generate_sample(seed=12345)

    coverage = np.count_nonzero(twin._specimen.particle_id_map) / twin._specimen.particle_id_map.size
    kinds = {record['kind'] for record in twin._specimen.records}
    round_ratios = []
    for record in twin._specimen.records:
        if record['kind'] != 'round':
            continue
        axis_x, axis_y = record['axes_nm']
        round_ratios.append(max(axis_x, axis_y) / min(axis_x, axis_y))
    assert 0.20 <= coverage <= 0.26
    assert len(twin._specimen.records) >= 8
    assert kinds == {'round', 'oval'}
    assert max(round_ratios) <= 1.08
    assert all('rotation_degrees' in record for record in twin._specimen.records)
    assert sum(record['pixel_count'] for record in twin._specimen.records) == np.count_nonzero(twin._specimen.particle_id_map)
    assert twin._get_fov() == FOV_M


def test_lattice_parameter_varies_smoothly_from_particle_center_to_edge(tmp_path: Path):
    twin = twin_for_path(tmp_path)
    twin._generate_sample(seed=12345)

    records_by_kind = {record['kind']: record for record in twin._specimen.records}
    for kind, expected_mean in (
        ('round', ROUND_CENTER_LATTICE_MEAN_ANGSTROM),
        ('oval', OVAL_CENTER_LATTICE_MEAN_ANGSTROM),
    ):
        record = records_by_kind[kind]
        center_lattice = record['center_lattice_parameter']
        local_values = twin._specimen.lattice_parameter_map[twin._specimen.particle_id_map == record['id']]

        assert abs(center_lattice - expected_mean) < 5 * 0.015
        if kind == 'round':
            assert center_lattice * (1.0 - EDGE_LATTICE_CHANGE_FRACTION) <= local_values.min()
            assert local_values.min() < center_lattice * 0.78
            assert center_lattice * 0.99 < local_values.max() <= center_lattice
        else:
            assert center_lattice <= local_values.min() < center_lattice * 1.01
            assert center_lattice * 1.22 < local_values.max()
            assert local_values.max() <= center_lattice * (1.0 + EDGE_LATTICE_CHANGE_FRACTION)


def test_lattice_parameter_profile_has_smooth_monotonic_endpoints():
    radius = np.linspace(0.0, 1.0, 101)
    round_profile = local_lattice_parameter(4.078, -0.25, radius)
    oval_profile = local_lattice_parameter(4.678, 0.25, radius)

    assert round_profile[0] == 4.078
    assert round_profile[-1] == 4.078 * 0.75
    assert np.all(np.diff(round_profile) <= 0.0)
    assert oval_profile[0] == 4.678
    assert oval_profile[-1] == 4.678 * 1.25
    assert np.all(np.diff(oval_profile) >= 0.0)


def test_beam_particle_uses_local_lattice_parameter(tmp_path: Path):
    twin = twin_for_path(tmp_path)
    twin._generate_sample(seed=12345)
    record = next(record for record in twin._specimen.records if record['kind'] == 'round')
    particle_mask = twin._specimen.particle_id_map == record['id']
    edge_y, edge_x = np.unravel_index(
        np.argmin(np.where(particle_mask, twin._specimen.lattice_parameter_map, np.inf)),
        particle_mask.shape,
    )
    twin._beam_pos_x = edge_x / (particle_mask.shape[1] - 1)
    twin._beam_pos_y = edge_y / (particle_mask.shape[0] - 1)

    particle = twin._beam_atoms().info

    assert particle['center_lattice_parameter'] == record['center_lattice_parameter']
    assert particle['lattice_parameter'] == twin._specimen.lattice_parameter_map[edge_y, edge_x]
    assert particle['lattice_strain_fraction'] < -0.22
    assert particle['rattle_value'] > 0.0


def test_particle_camera_acquisition_saves_local_lattice_metadata(monkeypatch, tmp_path: Path):
    twin = twin_for_path(tmp_path)
    particle = {
        'particle_id': 7,
        'particle_kind': 'oval',
        'lattice_parameter': 5.80,
        'center_lattice_parameter': 4.678,
        'lattice_strain_fraction': 5.80 / 4.678 - 1.0,
        'rotation_degrees': 23.0,
        'rattle_value': 0.75,
    }
    monkeypatch.setattr(twin, '_beam_atoms', lambda: SimpleNamespace(info=particle))
    monkeypatch.setattr(
        twin,
        '_simulate_diffraction_from_atoms',
        lambda atoms, imsize: np.ones((imsize, imsize), dtype=np.float32),
    )

    saved_path = Path(
        twin._acquire_camera_image(
            32,
            0.1,
            'BM-Ceta',
            'Half',
            frame_combining=6,
        )
    )

    with h5py.File(saved_path, 'r') as h5:
        attrs = h5['image'].attrs
        assert attrs['particle_lattice_parameter_angstrom'] == particle['lattice_parameter']
        assert attrs['particle_center_lattice_parameter_angstrom'] == particle['center_lattice_parameter']
        assert attrs['particle_lattice_strain_fraction'] == particle['lattice_strain_fraction']
        assert attrs['particle_lattice_strain_percent'] == 100.0 * particle['lattice_strain_fraction']
        assert attrs['exposure_time'] == pytest.approx(0.1)
        assert attrs['readout_area'] == 'Half'
        assert attrs['frame_combining'] == 6


def test_off_particle_camera_acquisition_saves_vacuum_diffraction(tmp_path: Path):
    pytest.importorskip('abtem', exc_type=ImportError)

    twin = twin_for_path(tmp_path)
    twin._generate_sample(seed=12345)
    twin._specimen.particle_id_map = np.zeros((64, 64), dtype=np.int16)
    twin._specimen.rattle_map = np.zeros((64, 64), dtype=np.float32)
    twin._specimen.records = []

    saved_path = Path(twin._acquire_camera_image(32, 0.1, 'BM-Ceta', 'Full'))

    assert saved_path.suffix == '.h5'
    with h5py.File(saved_path, 'r') as h5:
        image = h5['image'][()]
        assert image.shape == (32, 32)
        assert h5['image'].attrs['acquisition_type'] == 'diffraction'
        assert h5['image'].attrs['detector'] == 'BM-Ceta'
        assert h5['image'].attrs['pixel_size_mrad'] == 5.0
        assert h5['image'].attrs['max_angle_mrad'] == 80.0
        assert h5['image'].attrs['rattle_value'] == 0.0


def test_particle_atoms_include_rotation_and_rattle(tmp_path: Path):
    twin = twin_for_path(tmp_path)
    twin._generate_sample(seed=12345)
    specimen = twin._specimen
    row, column = np.argwhere(specimen.rattle_map > 0.0)[0]
    size = specimen.particle_id_map.shape[0]
    x_nm, y_nm = (column / (size - 1) - 0.5) * 500.0, (row / (size - 1) - 0.5) * 500.0
    record = specimen.records[int(specimen.particle_id_map[row, column]) - 1]

    rattled = specimen.atoms_at(x_nm, y_nm, ViewRequest(stage=np.zeros(5)), seed=1)
    again = specimen.atoms_at(x_nm, y_nm, ViewRequest(stage=np.zeros(5)), seed=2)

    assert rattled.info['rotation_degrees'] == record['rotation_degrees']
    assert rattled.info['particle_id'] == record['id']
    assert rattle_stdev(float(specimen.rattle_map[row, column])) > 0.03
    assert not np.allclose(rattled.get_positions(), again.get_positions())  # the frozen-phonon seed moves the atoms


def test_scanned_image_stage_sync_uses_map_cache(tmp_path: Path):
    class FakeStage:
        x = 5e-9
        y = -5e-9
        z = 0.0
        alpha = 0.0
        beta = 0.0

    twin = twin_for_path(tmp_path)
    twin._generate_sample(seed=12345)
    twin._detector_proxies = {'stage': FakeStage()}

    image = twin._render_stem_image(32, 1e-6, ['HAADF'])

    assert image.shape == (32, 32)
    assert twin._stage_position.tolist() == [5e-9, -5e-9, 0.0, 0.0, 0.0]


def test_scanned_image_saves_pixel_size_metadata(tmp_path: Path):
    twin = twin_for_path(tmp_path)
    twin._generate_sample(seed=12345)

    saved_path = Path(twin._acquire_scanned_image(32, 1e-6, ['HAADF']))

    with h5py.File(saved_path, 'r') as h5:
        image = h5['image/HAADF']
        assert image.shape == (64, 64)
        assert image.attrs['pixel_size_nm'] == 500.0 / 64
        assert image.attrs['pixel_size_m'] == FOV_M / 64
        assert image.attrs['sample_pixel_size_nm'] == 500.0 / 256
