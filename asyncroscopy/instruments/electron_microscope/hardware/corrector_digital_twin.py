"""Stateful digital twin of the CEOS aberration-corrector API."""

from __future__ import annotations

import copy
import json

from tango import DevState
from tango.server import device_property

from asyncroscopy.instruments.electron_microscope.hardware.corrector import CORRECTOR, ROUND_ABERRATIONS


DEFAULT_ABERRATIONS = {
    "WD": [0.0, 0.0],
    "C1": [0.0],
    "A1": [0.0, 0.38448128113770325e-9],
    "B2": [-68.45251255685642e-9, 64.85359774641199e-9],
    "A2": [11.667578600494137e-9, -29.775627778458194e-9],
    "C3": [123.0e-9],
    "S3": [95.3047364258614e-9, -189.72105710231244e-9],
    "A3": [-47.45099594807912e-9, -94.67424667529909e-9],
    "D4": [-905.31842572806e-9, 981.316128853203e-9],
    "B4": [4021.8433526960034e-9, 131.72716642732158e-9],
    "A4": [-4702.390968272048e-9, -208.25028574642903e-9],
}

# Fit order used by the CEOS-RPC protocol. A tableau returns every aberration
# up to and including its type's default "maxFit" entry.
FIT_ORDER = ["A2", "B2", "C3", "A3", "S3", "A4", "D4", "B4", "C5", "A5", "R5", "S5"]
TABLEAU_MAX_FIT = {"Fast": "B2", "Standard": "A4", "Enhanced": "A5"}
# Returned by every tableau regardless of type.
TABLEAU_BASE = ["WD", "C1", "A1"]


def tableau_aberration_names(tab_type: str) -> list[str]:
    """Aberrations a CEOS tableau of ``tab_type`` reports, in fit order."""
    if tab_type not in TABLEAU_MAX_FIT:
        raise ValueError(
            f"Unknown tableau type {tab_type!r}; expected one of {sorted(TABLEAU_MAX_FIT)}"
        )
    last = FIT_ORDER.index(TABLEAU_MAX_FIT[tab_type])
    return TABLEAU_BASE + FIT_ORDER[: last + 1]


class DigitalTwinCorrector(CORRECTOR):
    """Simulate CEOS commands while retaining the same inputs and JSON shape."""

    random_seed = device_property(dtype=int, default_value=1729)
    measurement_noise_fraction = device_property(dtype=float, default_value=0.0)

    def init_device(self) -> None:
        self._response_id = 1
        super().init_device()

    def _connect_backend(self) -> None:
        if not getattr(self, "_simulation_aberrations", None):
            self._simulation_aberrations = copy.deepcopy(DEFAULT_ABERRATIONS)
        self._last_status = "Digital twin ready"
        self.set_state(DevState.ON)

    def _response(self, result) -> str:
        payload = {"jsonrpc": "2.0", "id": self._response_id, "result": result}
        self._response_id += 1
        self._last_status = "OK"
        return json.dumps(payload, separators=(",", ":"))

    def _get_info(self) -> str:
        return self._response(
            {
                "manufacturer": "CEOS",
                "model": "DigitalTwinCorrector",
                "connected": True,
                "simulation": True,
                "correctorType": "CESCOR",
                "mode": "STEM",
            }
        )

    def _measured_aberrations(self) -> dict[str, list[float]]:
        coefficients = copy.deepcopy(self._simulation_aberrations)
        noise_fraction = max(0.0, float(self.measurement_noise_fraction))
        if noise_fraction == 0.0:
            return coefficients
        import numpy as np

        rng = np.random.default_rng(int(self.random_seed) + self._response_id)
        for name, values in coefficients.items():
            coefficients[name] = [
                float(value + rng.normal(0.0, max(abs(value), 1e-12) * noise_fraction))
                for value in values
            ]
        return coefficients

    @staticmethod
    def _wire(names, measured) -> dict[str, list[float]]:
        """Format aberrations as the CEOS server sends them: [x, y] pairs."""
        out = {}
        for name in names:
            if name not in measured:
                continue  # not simulated (for example C5, A5)
            values = list(measured[name])
            if name in ROUND_ABERRATIONS:
                values = [values[0], 0.0]
            out[name] = values
        return out

    def _acquire_tableau(self, tab_type: str, angle: float) -> str:
        names = tableau_aberration_names(tab_type)
        if not float(angle) > 0.0:
            raise ValueError(f"Tableau angle must be > 0 mrad, got {angle}")
        return self._response(
            {
                "tabType": tab_type,
                "angle": float(angle),
                "aberrations": self._wire(names, self._measured_aberrations()),
            }
        )

    def _measure_c1a1(self) -> str:
        return self._response(
            {"aberrations": self._wire(("C1", "A1"), self._measured_aberrations())}
        )

    def _correct_aberration(self, name: str, values: list[float]) -> str:
        if name not in self._simulation_aberrations:
            raise ValueError(f"Unknown aberration {name!r}")
        current = self._simulation_aberrations[name]
        if name in ROUND_ABERRATIONS and len(values) == 2:
            values = values[:1]  # the y component of a scalar is ignored
        if len(values) != len(current):
            raise ValueError(
                f"Aberration {name} expects {len(current)} value(s), got {len(values)}"
            )
        self._simulation_aberrations[name] = [
            float(existing - correction)
            for existing, correction in zip(current, values)
        ]
        # The CEOS server returns a null result for a successful correction.
        return self._response(None)

    def _set_simulation_aberrations(self, coefficients: dict) -> None:
        unknown = set(coefficients) - set(DEFAULT_ABERRATIONS)
        if unknown:
            raise ValueError(f"Unknown aberration keys: {sorted(unknown)}")
        merged = copy.deepcopy(DEFAULT_ABERRATIONS)
        for name, values in coefficients.items():
            expected = len(merged[name])
            normalized = [float(value) for value in values]
            if len(normalized) != expected:
                raise ValueError(f"Aberration {name} expects {expected} value(s)")
            merged[name] = normalized
        self._simulation_aberrations = merged


if __name__ == "__main__":
    DigitalTwinCorrector.run_server()
