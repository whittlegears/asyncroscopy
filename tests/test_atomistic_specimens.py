"""The atomistic twins take their samples from Specimen plug-ins, and rendered output for a fixed seed does not change.

* The four sources are registered, chosen by ``specimen: {source: ...}``, and reject options they do not know.
* A twin refuses a specimen its renderer cannot image.
* The element table is shared.
* The rendered output of a fixed command sequence matches tests/data/atomistic_specimens_golden.json: array statistics
  (mean, spread, extremes and block means), so numerical differences between platforms and FFT libraries do not matter.
  The golden values were written after a command-for-command comparison with the twins as they were before the
  specimens became plug-ins (identical arrays, apart from abTEM's own round-off). ASYNCROSCOPY_WRITE_GOLDEN=1 rewrites it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import tango
import yaml

from asyncroscopy import settings
from asyncroscopy.instruments.electron_microscope import digital_twin_diffraction, digital_twin_particles, digital_twin_tilt, polycrystalline_gold_digital_twin
from asyncroscopy.instruments.electron_microscope.digital_twin_tilt import DigitalTwinTilt
from asyncroscopy.instruments.electron_microscope.twin import elements, sample_model
from asyncroscopy.instruments.electron_microscope.twin.specimen import SPECIMENS, SpecimenContext, ViewRequest, build_specimen
from asyncroscopy.sim.rng import EventRng

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).resolve().parent / "data" / "atomistic_specimens_golden.json"
WRITE_GOLDEN = settings.get().write_golden


def context(fov_m: float = 2e-8) -> SpecimenContext:
    return SpecimenContext(rng=EventRng(5), fov_m=fov_m)


def test_every_atomistic_source_is_registered_and_built_by_name():
    for source, options in (
        ("silicon_lamella", {"lamella_width_nm": 500.0}),
        ("polycrystal", {"volume_size_xy_nm": 4.0, "volume_thickness_nm": 2.0}),
        ("particles", {"world_imsize": 32, "particle_count": 4}),
        ("nanoparticle_map", {"map_size": 200}),
    ):
        assert source in SPECIMENS.all()
        specimen = build_specimen({"source": source, **options}, context())
        assert specimen.source == source and specimen.render_modes
        assert specimen.element_symbols


@pytest.mark.parametrize("name, source", [("digital_twin_tilt.yaml", "silicon_lamella"), ("digital_twin_particles.yaml", "particles"), ("polycrystalline_gold.yaml", "polycrystal"), ("diffraction.yaml", "nanoparticle_map")])
def test_each_twin_config_selects_its_specimen_in_yaml(name, source):
    mapping = dict(yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))["instrument"]["properties"]["specimen"])
    assert mapping["source"] == source
    assert build_specimen(mapping, context()).source == source


def test_a_source_refuses_an_option_it_does_not_know():
    with pytest.raises(ValueError, match="no option"):
        build_specimen({"source": "silicon_lamella", "width": 1.0}, context())


def test_a_twin_refuses_a_specimen_its_renderer_cannot_image():
    twin = DigitalTwinTilt.__new__(DigitalTwinTilt)
    twin._tango_properties = {}
    twin._fov = 1e-6
    twin.specimen, twin.specimen_source = "", "polycrystal"
    with pytest.raises(ValueError, match="renders in"):
        twin._load_specimen(1)


def test_the_element_table_is_shared():
    assert sample_model.ELEMENTS is elements.ELEMENTS
    au = elements.bulk_cell("Au", cubic=True)
    assert au.cell.lengths()[0] == pytest.approx(elements.lattice_constant("Au"))


def test_a_view_is_a_function_of_the_stage_and_field():
    world = build_specimen({"source": "particles", "world_imsize": 32, "particle_count": 3}, context())
    a = world.view(ViewRequest(stage=np.zeros(5), fov_angstrom=200.0, imsize=32, edge_crop=4))
    b = world.view(ViewRequest(stage=np.zeros(5), fov_angstrom=200.0, imsize=32, edge_crop=4))
    moved = world.view(ViewRequest(stage=np.array([1e-9, 0.0, 0.0, 0.0, 0.0]), fov_angstrom=200.0, imsize=32, edge_crop=4))
    assert np.array_equal(a.atoms.get_positions(), b.atoms.get_positions())
    assert a.records and not np.array_equal(a.atoms.get_positions(), moved.atoms.get_positions())
    grain = build_specimen({"source": "polycrystal", "volume_size_xy_nm": 4.0, "volume_thickness_nm": 2.0}, context())
    field = grain.view(ViewRequest(stage=np.zeros(5), fov_angstrom=20.0))
    assert len(field.atoms) > 0 and set(field.atoms.numbers) == {79}


# ----------------------------------------------------------------------
# Output against the golden statistics
# ----------------------------------------------------------------------
def _fingerprint(array) -> dict:
    a = np.asarray(array, dtype=np.float64)
    blocks = np.array_split(a.reshape(a.shape[0], -1).mean(axis=1), 8) if a.ndim == 2 else [a]
    return {"shape": list(a.shape), "mean": float(a.mean()), "std": float(a.std()), "min": float(a.min()), "max": float(a.max()), "blocks": [float(np.mean(b)) for b in blocks]}


def _atoms_digest(atoms) -> str:
    return hashlib.sha256(np.round(atoms.get_positions(), 4).tobytes()).hexdigest()


def _matches(run, golden) -> None:
    assert run["shape"] == golden["shape"]
    for key in ("mean", "std", "min", "max"):
        assert run[key] == pytest.approx(golden[key], rel=1e-6, abs=1e-9), key
    assert run["blocks"] == pytest.approx(golden["blocks"], rel=1e-6, abs=1e-9)


def _device(tango_ctx, name):
    return tango.Util.instance().get_device_by_name(name), tango.DeviceProxy(tango_ctx.get_device_access(name.replace("atomistic/", "stage/atomistic_")))


def _capture_saves(monkeypatch) -> list:
    saved: list = []

    def fake_save(device, data_server, kind, detectors, arrays, **kwargs):
        saved.append((kind, arrays))
        return f"key-{len(saved)}"

    for module in (digital_twin_tilt, digital_twin_particles, digital_twin_diffraction, polycrystalline_gold_digital_twin):
        monkeypatch.setattr(module, "save_acquisition", fake_save)
    return saved


def test_rendered_output_matches_the_golden_statistics(tango_ctx, monkeypatch):
    run: dict[str, dict] = {}
    saved = _capture_saves(monkeypatch)

    tilt, tilt_stage = _device(tango_ctx, "asyncroscopy/atomistic/tilt")
    tilt._move_stage([1e-8, -2e-8, 0.0, 10.0, 2.0])
    run["tilt_image"] = _fingerprint(tilt._render_stem_image(64, 1e-6, ["HAADF"], [0.0, 0.0, 1.0, 1.0]))
    tilt._set_beam_tilt([1e-4, -2e-4])
    tilt._set_defocus(3e-9)
    run["tilt_region"] = _fingerprint(tilt._render_stem_image(64, 1e-6, ["HAADF"], [0.1, 0.1, 0.5, 0.5]))
    tilt_stage.alpha = 3.0
    material, local_x, local_y = tilt._beam_material()
    assert material == "silicon"
    run["tilt_atoms"] = {"digest": _atoms_digest(tilt._silicon_atoms(local_x, local_y, 7))}
    stage = tilt._get_parameters()
    assert json.loads(stage)["lamella_width_nm"] == 620.0

    particles, _stage = _device(tango_ctx, "asyncroscopy/atomistic/particles")
    particles._acquire_scanned_image(64, 1e-5, ["haadf"], [0.0, 0.0, 1.0, 1.0])
    run["particles_image"] = _fingerprint(saved[-1][1][0])
    particles._place_beam([0.5, 0.5])
    particles._acquire_spectrum("eds", 1.0)
    run["particles_spectrum"] = {"values": [float(v) for v in saved[-1][1]]}
    particles._set_fov(3e-8)
    particles._move_stage([1e-9, -2e-9, 0.0, 0.0, 3.0])
    particles._acquire_scanned_image(64, 1e-5, ["haadf"], [0.0, 0.0, 1.0, 1.0])
    run["particles_tilted"] = _fingerprint(saved[-1][1][0])

    diffraction, _stage = _device(tango_ctx, "asyncroscopy/atomistic/diffraction")
    diffraction._acquire_scanned_image(128, 1e-6, ["haadf"], [0.0, 0.0, 1.0, 1.0])
    run["diffraction_overview"] = _fingerprint(saved[-1][1][0])
    diffraction._move_stage([2e-8, -1e-8, 0.0, 0.0, 0.0])
    diffraction._acquire_scanned_image(128, 1e-6, ["haadf"], [0.0, 0.0, 1.0, 1.0])
    run["diffraction_moved"] = _fingerprint(saved[-1][1][0])
    diffraction._place_beam([0.5, 0.5])
    atoms = next(a for a in (diffraction._specimen.atoms_at(x, y, ViewRequest(stage=np.zeros(5)), seed=11) for x, y in [(-50.0, -50.0), (0.0, 0.0), (60.0, -30.0), (-100.0, 80.0), (30.0, 100.0), (120.0, 40.0), (-20.0, 150.0)]) if a is not None)
    run["diffraction_atoms"] = {"digest": _atoms_digest(atoms), "lattice": float(atoms.info["lattice_parameter"])}

    polycrystal, _stage = _device(tango_ctx, "asyncroscopy/atomistic/polycrystal")
    polycrystal._set_fov(1e-8)
    polycrystal._move_stage([3e-9, -1e-9, 0.0, 0.0, 0.0])
    polycrystal._acquire_scanned_image(48, 1e-5, ["haadf"], [0.0, 0.0, 1.0, 1.0])
    run["polycrystal_image"] = _fingerprint(saved[-1][1][0])
    run["polycrystal_atoms"] = {"count": polycrystal._last_rendered_atom_count}
    polycrystal._acquire_scanned_image(48, 1e-5, ["haadf"], [0.0, 0.0, 1.0, 1.0])
    assert polycrystal._raw_potential_cache_hits >= 1

    if WRITE_GOLDEN:
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(run, indent=1, sort_keys=True), encoding="utf-8")
        pytest.skip(f"wrote {GOLDEN}")
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert set(run) == set(golden)
    for key, value in run.items():
        if "shape" in value:
            _matches(value, golden[key])
        elif "values" in value:
            assert value["values"] == pytest.approx(golden[key]["values"], rel=1e-6), key
        else:
            assert value == golden[key], key
