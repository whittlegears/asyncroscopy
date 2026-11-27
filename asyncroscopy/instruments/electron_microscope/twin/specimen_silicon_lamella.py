"""The ``silicon_lamella`` specimen: a layered silicon FIB lamella attached to a gridbar.

The tilt twin (``digital_twin_tilt.py``) works out, per view, which point of the lamella each pixel sees (tilt axis,
stage, beam tilt) and asks this specimen what is there. ``view`` gives the overview image and the lamella and gridbar
masks at those points; ``atoms_at`` gives the silicon supercell a parked beam sees, with the local lattice parameter
and crystal rotation, tilted by the stage.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from .elements import bulk_cell
from .specimen import FieldView, Specimen, SpecimenContext, ViewRequest, check_options, register_specimen


@dataclass(frozen=True)
class LamellaParams:
    lamella_width_nm: float = 620.0
    lamella_height_nm: float = 360.0
    lamella_thickness_nm: float = 80.0
    gridbar_width_nm: float = 220.0
    layer_period_nm: float = 45.0
    silicon_lattice_parameter_angstrom: float = 5.431  # unstrained
    lattice_parameter_gradient_x_percent: float = 2.0  # end-to-end change across the lamella width
    lattice_parameter_gradient_y_percent: float = 0.5
    crystal_rotation_deg: float = 1.5  # mean in-plane rotation
    crystal_rotation_gradient_x_deg: float = 4.0
    crystal_rotation_gradient_y_deg: float = -2.0
    crystal_tilt_x_mrad: float = 3.0
    crystal_tilt_y_mrad: float = -2.0
    crystal_tilt_gradient_x_mrad: float = 10.0
    crystal_tilt_gradient_y_mrad: float = -7.0
    multislice_supercell_xy: int = 4  # unit cells in x and y for every probe position
    multislice_supercell_z: int = 8  # unit cells along the beam
    multislice_vacuum_angstrom: float = 2.0
    frozen_phonon_sigma_angstrom: float = 0.03  # ASE rattle standard deviation before each multislice calculation


@register_specimen
class SiliconLamella(Specimen):
    """A layered, slightly strained silicon lamella on a gridbar; the probe sees an ASE supercell per position."""

    source = "silicon_lamella"
    render_modes = ("lamella",)

    def __init__(self, params: LamellaParams) -> None:
        self.params = params
        p = params
        x_min, x_max = -0.5 * p.lamella_width_nm - p.gridbar_width_nm, 0.5 * p.lamella_width_nm
        y_half = 0.7 * p.lamella_height_nm
        self.bounds = (x_min * 1e-9, x_max * 1e-9, -y_half * 1e-9, y_half * 1e-9)
        self.world_bounds_ang = {
            "x_min": x_min * 10.0,
            "x_max": x_max * 10.0,
            "y_min": -y_half * 10.0,
            "y_max": y_half * 10.0,
            "z_min": -0.5 * p.lamella_thickness_nm * 10.0,
            "z_max": 0.5 * p.lamella_thickness_nm * 10.0,
        }

    @classmethod
    def from_options(cls, options: dict, context: SpecimenContext) -> "SiliconLamella":
        names = [f.name for f in fields(LamellaParams)]
        check_options(cls.source, options, names)
        defaults = LamellaParams()
        kwargs = {name: type(getattr(defaults, name))(options[name]) for name in names if name in options}
        return cls(LamellaParams(**kwargs))

    @property
    def element_symbols(self) -> tuple[str, ...]:
        return ("Si",)

    @property
    def vacuum_lateral_angstrom(self) -> float:
        p = self.params
        return p.silicon_lattice_parameter_angstrom * p.multislice_supercell_xy + 2.0 * p.multislice_vacuum_angstrom

    def _lattice_fields(self, local_x_nm, local_y_nm):
        p = self.params
        x_fraction = np.clip(np.asarray(local_x_nm) / p.lamella_width_nm, -0.5, 0.5)
        y_fraction = np.clip(np.asarray(local_y_nm) / p.lamella_height_nm, -0.5, 0.5)
        strain = (p.lattice_parameter_gradient_x_percent * x_fraction + p.lattice_parameter_gradient_y_percent * y_fraction) / 100.0
        lattice_parameter = p.silicon_lattice_parameter_angstrom * (1.0 + strain)
        rotation_deg = p.crystal_rotation_deg + p.crystal_rotation_gradient_x_deg * x_fraction + p.crystal_rotation_gradient_y_deg * y_fraction
        tilt_x_mrad = p.crystal_tilt_x_mrad + p.crystal_tilt_gradient_x_mrad * x_fraction
        tilt_y_mrad = p.crystal_tilt_y_mrad + p.crystal_tilt_gradient_y_mrad * y_fraction
        return lattice_parameter, rotation_deg, tilt_x_mrad, tilt_y_mrad

    def view(self, request: ViewRequest) -> FieldView:
        """maps: {"image": the overview signal, "lamella", "gridbar": masks} at the request's local_nm points."""
        p = self.params
        local_x, local_y = request.local_nm
        size = local_x.shape[0]
        lamella = (np.abs(local_x) <= 0.5 * p.lamella_width_nm) & (np.abs(local_y) <= 0.5 * p.lamella_height_nm)
        gridbar = (
            (local_x >= -0.5 * p.lamella_width_nm - p.gridbar_width_nm)
            & (local_x < -0.5 * p.lamella_width_nm)
            & (np.abs(local_y) <= 0.68 * p.lamella_height_nm)
        )
        layer_phase = np.floor((local_y + 0.5 * p.lamella_height_nm) / p.layer_period_nm)
        layers = 0.5 * (1.0 + np.where(layer_phase % 2 == 0, 1.0, -1.0))
        wedge = np.clip(0.68 + 0.30 * (local_x / p.lamella_width_nm + 0.5), 0.55, 1.0)
        curtains = 0.025 * np.sin(2.0 * np.pi * local_x / 38.0) ** 2
        texture = 0.025 * np.sin(local_x / 21.0 + local_y / 33.0)
        image = np.full((size, size), 0.008, dtype=np.float64)
        image[lamella] = 0.28 + 0.11 * layers[lamella] + 0.22 * wedge[lamella] + texture[lamella]
        image[lamella] += curtains[lamella]
        image[gridbar] = 0.97

        projected_thickness = p.lamella_thickness_nm / max(abs(float(request.rotation[2, 2])), 0.25)
        image[lamella] *= np.clip(projected_thickness / p.lamella_thickness_nm, 1.0, 2.3)
        return FieldView(maps={"image": np.clip(image, 0.0, 1.0), "lamella": lamella, "gridbar": gridbar})

    def atoms_at(self, x_nm: float, y_nm: float, request: ViewRequest, seed: int, thermal_scale: float = 1.0):
        """The silicon supercell under the probe at lamella point (x_nm, y_nm), tilted by the stage; info holds the local crystal."""
        p = self.params
        lattice_parameter, rotation_deg, tilt_x_mrad, tilt_y_mrad = (float(v) for v in self._lattice_fields(x_nm, y_nm))
        repetitions = (p.multislice_supercell_xy, p.multislice_supercell_xy, p.multislice_supercell_z)
        atoms = bulk_cell("Si", lattice_parameter, cubic=True).repeat(repetitions)
        atoms.rotate(rotation_deg, "z", center="COP")
        atoms.rotate(float(request.stage[3]) + np.degrees(tilt_x_mrad * 1e-3), "x", center="COP")
        atoms.rotate(float(request.stage[4]) + np.degrees(tilt_y_mrad * 1e-3), "y", center="COP")
        sigma = p.frozen_phonon_sigma_angstrom * thermal_scale
        if sigma > 0.0:
            atoms.rattle(stdev=sigma, seed=seed)
        atoms.center(vacuum=p.multislice_vacuum_angstrom)
        atoms.info.update(
            {
                "local_x_nm": float(x_nm),
                "local_y_nm": float(y_nm),
                "lattice_parameter_angstrom": lattice_parameter,
                "nearest_neighbor_distance_angstrom": lattice_parameter * np.sqrt(3.0) / 4.0,
                "crystal_rotation_deg": rotation_deg,
                "crystal_tilt_x_mrad": tilt_x_mrad,
                "crystal_tilt_y_mrad": tilt_y_mrad,
            }
        )
        return atoms

    def describe(self) -> dict:
        return {**super().describe(), **{f.name: getattr(self.params, f.name) for f in fields(LamellaParams)}}
