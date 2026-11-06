"""What the atomistic twins share beyond the ``ElectronMicroscopeTwin`` shell: a table of particles and the
EDS spectrum it gives under the parked beam.

The tilt, diffraction and polycrystalline-gold twins (``digital_twin_tilt.py``, ``digital_twin_diffraction.py``,
``polycrystalline_gold_digital_twin.py``) build their own samples and renderers; what they share is kept here,
in a mixin put before the shell in their bases::

    class DigitalTwinTilt(ParticleSpectrumMixin, ElectronMicroscopeTwin): ...

A twin's ``_generate_sample`` fills ``_particle_records_base`` ({"center": [x, y, z] in angstrom, "radius" in
angstrom, "btype", "composition": {element: fraction}}) and ``_all_sample_elements``; the mixin keeps
``_particle_records_view``, the same particles where the stage has put them (tilted, then shifted), and turns
them into a spectrum. A twin whose records hold something else overrides ``_update_view_cache``. The
``DigitalTwin`` (specimens, ``twin/specimen.py``) does not use it.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from asyncroscopy.data.data_writer import save_acquisition


class ParticleSpectrumMixin:
    """Particle records that follow the stage, and the EDS spectrum they give at the beam position."""

    def init_device(self) -> None:
        self._particle_records_base: list[dict] = []
        self._particle_records_view: list[dict] = []
        self._all_sample_elements: list[str] = []
        self._cached_pose_key: tuple | None = None
        super().init_device()

    def _sync_stage_from_proxy(self) -> None:
        super()._sync_stage_from_proxy()
        self._update_view_cache(force=False)

    def _move_stage(self, position) -> None:
        super()._move_stage(position)
        self._update_view_cache(force=False)

    def _update_view_cache(self, force: bool = False) -> None:
        """Place the particle records where the current stage pose puts them (tilt about x then y, then shift)."""
        pose_key = tuple(np.round(self._stage_position, 12))
        if not force and pose_key == self._cached_pose_key:
            return
        alpha_deg, beta_deg = self._stage_position[3], self._stage_position[4]
        stage_xyz_ang = self._stage_position[:3] * 1e10
        self._particle_records_view = []
        for rec in self._particle_records_base:
            dummy = Atoms("H", positions=[rec["center"]])
            if alpha_deg != 0.0:
                dummy.rotate(alpha_deg, "x", center=(0, 0, 0))
            if beta_deg != 0.0:
                dummy.rotate(beta_deg, "y", center=(0, 0, 0))
            dummy.translate(-stage_xyz_ang)
            self._particle_records_view.append(
                {"center": dummy.positions[0], "radius": rec["radius"], "btype": rec["btype"], "composition": rec["composition"]}
            )
        self._cached_pose_key = pose_key

    def _simulate_spectrum(self, detector_name: str, exposure_time: float) -> dict[str, float]:
        """Element fractions at the current beam position, weighted by the particles under it."""
        self._sync_stage_from_proxy()
        self._update_view_cache(force=False)

        px, py = self.read_beam_pos()
        fov_ang = self._fov * 1e10
        beam_x = (px - 0.5) * fov_ang
        beam_y = (py - 0.5) * fov_ang

        weighted: dict[str, float] = {}
        weight_sum = 0.0
        for rec in self._particle_records_view:
            cx, cy = rec["center"][:2]
            radius = rec["radius"]
            dist = float(np.hypot(beam_x - cx, beam_y - cy))
            if dist <= radius:
                w = max(1e-6, 1.0 - (dist / radius))
                weight_sum += w
                for element, frac in rec["composition"].items():
                    weighted[element] = weighted.get(element, 0.0) + w * frac

        spectrum_seed = int(
            abs(hash((tuple(np.round(self._stage_position, 10)), round(px, 6), round(py, 6), round(exposure_time, 6), int(self.sample_seed))))
            % (2**32)
        )
        rng = np.random.default_rng(spectrum_seed)

        if weight_sum <= 0.0:
            return {element: float(np.abs(rng.normal(0.0, 0.02))) for element in (self._all_sample_elements or ["Au", "Pt", "Fe"])}

        normalized = {el: val / weight_sum for el, val in weighted.items()}
        noisy = {element: float(max(0.0, value + rng.normal(0.0, 0.01))) for element, value in normalized.items()}
        total = sum(noisy.values())
        if total <= 0.0:
            return noisy
        return {el: val / total for el, val in noisy.items()}

    def _acquire_spectrum(self, detector_name: str, exposure_time: float) -> str:
        """Simulate a spectrum, save it, and return its DATA/Tiled key."""
        spectrum = self._simulate_spectrum(detector_name, exposure_time)
        return save_acquisition(
            self,
            self._detector_proxies.get("data"),
            "spectrum",
            detector_name,
            np.array(list(spectrum.values()), dtype=np.float64),
            dataset_name="spectrum",
            dataset_attrs={"elements": list(spectrum.keys())},
        )

    def _viewport_metadata(self) -> dict:
        return {**super()._viewport_metadata(), "particle_count": len(self._particle_records_base)}
