"""Stateful digital twin of the CEOS aberration-corrector API."""

from __future__ import annotations

import copy
import json

from tango import DevState
from tango.server import device_property

from asyncroscopy.instruments.electron_microscope.hardware.corrector import CORRECTOR, FIRST_ORDER


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


class DigitalTwinCorrector(CORRECTOR):
    """Simulate CEOS commands while retaining the same inputs and JSON shape.

    A measurement is the hidden truth plus noise (both on by default), so a
    measure-then-correct loop does not converge by construction. ``correction_sign``
    -1 makes a correction add instead of subtract: the wrong-sign corrector a
    first-light sign test guards against.
    """

    random_seed = device_property(dtype=int, default_value=1729)
    measurement_noise_fraction = device_property(dtype=float, default_value=0.02, doc="Measurement noise, as a fraction of each coefficient.")
    measurement_noise_m = device_property(dtype=float, default_value=0.5e-9, doc="Measurement noise floor (m), added to every coefficient.")
    correction_sign = device_property(dtype=float, default_value=1.0, doc="1: a correction subtracts the value sent (CEOS as assumed); -1: it adds (a wrong-sign backend).")

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
            }
        )

    def _measured_aberrations(self) -> dict[str, list[float]]:
        coefficients = copy.deepcopy(self._simulation_aberrations)
        noise_fraction = max(0.0, float(self.measurement_noise_fraction))
        noise_floor = max(0.0, float(self.measurement_noise_m))
        if noise_fraction == 0.0 and noise_floor == 0.0:
            return coefficients
        import numpy as np

        rng = np.random.default_rng(int(self.random_seed) + self._response_id)
        for name, values in coefficients.items():
            coefficients[name] = [
                float(value + rng.normal(0.0, float(np.hypot(abs(value) * noise_fraction, noise_floor)) or 1e-30))
                for value in values
            ]
        return coefficients

    def _acquire_tableau(self, tab_type: str, angle: float) -> str:
        return self._response(
            {
                "tabType": tab_type,
                "angle": float(angle),
                "aberrations": self._measured_aberrations(),
            }
        )

    def _measure_aberrations(self) -> dict[str, list[float]]:
        measured = self._measured_aberrations()
        return {name: measured[name] for name in FIRST_ORDER}

    def _measure_c1a1(self) -> str:
        return self._response({"aberrations": self._measure_aberrations()})

    def _correct(self, name: str, values: list[float]) -> None:
        if name not in self._simulation_aberrations:
            raise ValueError(f"Unknown aberration {name!r}")
        current = self._simulation_aberrations[name]
        if len(values) != len(current):
            raise ValueError(
                f"Aberration {name} expects {len(current)} value(s), got {len(values)}"
            )
        sign = 1.0 if float(self.correction_sign) >= 0 else -1.0
        self._simulation_aberrations[name] = [
            float(existing - sign * correction)
            for existing, correction in zip(current, values)
        ]

    def _correct_aberration(self, name: str, values: list[float]) -> str:
        self._correct(name, values)
        return self._response(
            {"name": name, "value": self._simulation_aberrations[name], "corrected": True}
        )

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
