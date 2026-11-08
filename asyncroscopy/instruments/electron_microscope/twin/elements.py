"""The element table every specimen source shares: atomic number, number density and crystal structure.

Keys are element symbols; each value is (Z, atoms per cubic angstrom, ASE bulk structure, lattice constant in
angstrom). ``bulk_cell`` builds a source's unit cell from it, so an element has one lattice constant everywhere
unless a source is told otherwise.
"""

from __future__ import annotations

from ase.build import bulk

ELEMENTS: dict[str, tuple[int, float, str, float]] = {
    "C": (6, 0.100, "diamond", 3.567),
    "Si": (14, 0.050, "diamond", 5.431),
    "Fe": (26, 0.085, "bcc", 2.87),
    "Ni": (28, 0.091, "fcc", 3.52),
    "Cu": (29, 0.085, "fcc", 3.615),
    "Pd": (46, 0.068, "fcc", 3.89),
    "Ag": (47, 0.059, "fcc", 4.09),
    "Pt": (78, 0.066, "fcc", 3.92),
    "Au": (79, 0.059, "fcc", 4.08),
}


def atomic_number(element: str) -> int:
    return ELEMENTS[element][0]


def lattice_constant(element: str) -> float:
    return ELEMENTS[element][3]


def bulk_cell(element: str, lattice_angstrom: float | None = None, cubic: bool = False):
    """The ASE unit cell of ``element`` from the table; ``lattice_angstrom`` overrides its lattice constant."""
    _z, _rho, structure, tabulated = ELEMENTS[element]
    return bulk(element, structure, a=tabulated if lattice_angstrom is None else float(lattice_angstrom), cubic=cubic)
