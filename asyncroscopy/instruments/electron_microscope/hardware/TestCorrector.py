"""
Concrete CORRECTOR implementation for tests and local simulated configs.

Simulates the CEOS aberration corrector server so digital twin deployments
expose the same corrector device (and MCP tools) as the real instrument
config. All Tango commands are inherited from CORRECTOR unchanged; only the
hardware boundary (_connect and _call) is replaced, so responses keep the
exact JSON-RPC 2.0 formatting the real CEOS server returns.
"""

import json

from tango import DevState

from asyncroscopy.instruments.electron_microscope.hardware.corrector import CORRECTOR

# Plausible resting aberration coefficients for a corrected Spectra 300 column.
# Values are [x] or [x, y] in meters, keyed by CEOS aberration names.
_ABERRATION_DEFAULTS: dict[str, list[float]] = {
    "C1": [50e-9],
    "A1": [12e-9, -8e-9],
    "B2": [40e-9, 25e-9],
    "A2": [60e-9, -30e-9],
    "C3": [1.2e-6],
    "A3": [0.4e-6, 0.2e-6],
    "S3": [0.5e-6, -0.3e-6],
}


class TestCorrector(CORRECTOR):
    """Cached corrector implementation used when no CEOS server exists."""

    def init_device(self) -> None:
        # CORRECTOR.init_device probes the CEOS TCP server via _connect; the
        # override below turns that into an unconditional simulated connect.
        self._aberrations = {name: list(values) for name, values in _ABERRATION_DEFAULTS.items()}
        super().init_device()

    def _connect(self) -> None:
        self._last_status = "Connected (simulated)"
        self.set_state(DevState.ON)

    def _call(self, method: str, params: dict | None = None) -> str:
        """Answer JSON-RPC methods from cached state, formatted like the real server."""
        params = params or {}
        message_id = self._message_id
        self._message_id += 1

        if method == "getInfo":
            result = {
                "type": "CETCOR (simulated)",
                "version": "digital-twin",
                "aberrations": sorted(self._aberrations),
            }
        elif method == "measureC1A1":
            result = {"C1": self._aberrations["C1"], "A1": self._aberrations["A1"]}
        elif method == "acquireTableau":
            result = {
                "tabType": params.get("tabType"),
                "angle": params.get("angle"),
                "coefficients": {name: list(values) for name, values in self._aberrations.items()},
            }
        elif method == "correctAberration":
            name = params.get("name")
            if name not in self._aberrations:
                raise ValueError(
                    f"Unknown aberration {name!r}. Available: {sorted(self._aberrations)}"
                )
            values = [float(value) for value in params.get("value", [])]
            if len(values) != len(self._aberrations[name]):
                raise ValueError(
                    f"Aberration {name!r} takes {len(self._aberrations[name])} value(s), "
                    f"got {len(values)}"
                )
            self._aberrations[name] = values
            result = {"name": name, "value": values, "status": "corrected"}
        else:
            raise ValueError(f"Unknown CEOS method {method!r}")

        self._last_status = "OK"
        return json.dumps({"jsonrpc": "2.0", "id": message_id, "result": result})


if __name__ == "__main__":
    TestCorrector.run_server()
