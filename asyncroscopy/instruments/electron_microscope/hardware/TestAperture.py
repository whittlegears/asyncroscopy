"""
Concrete APERTURE implementation for tests and local simulated configs.

Simulates the motorized aperture mechanisms of a Spectra 300 so digital twin
deployments expose the same aperture device (and MCP tools) as the real
instrument config.
"""

import math

from asyncroscopy.instruments.electron_microscope.hardware.aperture import APERTURE

# Mechanism layout loosely modeled on a Spectra 300 column. Diameters in meters.
_MECHANISM_DEFAULTS: dict[str, dict] = {
    "C1": {
        "apertures": {"2000 um": 2000e-6, "150 um": 150e-6, "100 um": 100e-6, "50 um": 50e-6},
        "selected": "150 um",
        "retractable": False,
    },
    "C2": {
        "apertures": {"150 um": 150e-6, "70 um": 70e-6, "50 um": 50e-6, "30 um": 30e-6},
        "selected": "70 um",
        "retractable": False,
    },
    "C3": {
        "apertures": {"2000 um": 2000e-6, "1000 um": 1000e-6, "100 um": 100e-6, "40 um": 40e-6},
        "selected": "1000 um",
        "retractable": False,
    },
    "OBJ": {
        "apertures": {"100 um": 100e-6, "70 um": 70e-6, "40 um": 40e-6, "20 um": 20e-6},
        "selected": "70 um",
        "retractable": True,
    },
    "SA": {
        "apertures": {"800 um": 800e-6, "200 um": 200e-6, "40 um": 40e-6, "10 um": 10e-6},
        "selected": "200 um",
        "retractable": True,
    },
}


class TestAperture(APERTURE):
    """Cached aperture implementation used when no hardware-backed aperture exists."""

    def init_device(self) -> None:
        super().init_device()
        self._mechanisms = {
            name: {
                "apertures": dict(spec["apertures"]),
                "selected": spec["selected"],
                "retractable": spec["retractable"],
                "inserted": True,
                "enabled": True,
                "position": [0.0, 0.0],
            }
            for name, spec in _MECHANISM_DEFAULTS.items()
        }
        self._mechanism = next(iter(self._mechanisms))

    def _state(self) -> dict:
        try:
            return self._mechanisms[self._mechanism]
        except KeyError:
            raise ValueError(
                f"Unknown mechanism {self._mechanism!r}. "
                f"Available: {sorted(self._mechanisms)}"
            ) from None

    def _read_available_mechanisms(self) -> list[str]:
        return list(self._mechanisms)

    def _read_available_apertures(self) -> list[str]:
        return list(self._state()["apertures"])

    def _read_selected_aperture(self) -> str:
        return self._state()["selected"]

    def _write_selected_aperture(self, value: str) -> None:
        state = self._state()
        if value not in state["apertures"]:
            raise ValueError(
                f"Unknown aperture {value!r} on mechanism {self._mechanism!r}. "
                f"Available: {list(state['apertures'])}"
            )
        state["selected"] = value

    def _read_aperture_type(self) -> str:
        return "Circular"

    def _read_aperture_diameter(self) -> float:
        state = self._state()
        diameter = state["apertures"].get(state["selected"])
        return math.nan if diameter is None else float(diameter)

    def _read_insertion_state(self) -> str:
        return "Inserted" if self._state()["inserted"] else "Retracted"

    def _read_enabled(self) -> bool:
        return self._state()["enabled"]

    def _read_retractable(self) -> bool:
        return self._state()["retractable"]

    def _read_position(self) -> list[float]:
        return list(self._state()["position"])

    def _write_position(self, value) -> None:
        position = [float(component) for component in value]
        if len(position) != 2:
            raise ValueError("Aperture position must be [x, y]")
        self._state()["position"] = position

    def _insert(self) -> None:
        self._state()["inserted"] = True

    def _retract(self) -> None:
        state = self._state()
        if not state["retractable"]:
            raise ValueError(f"Mechanism {self._mechanism!r} is not retractable")
        state["inserted"] = False

    def _enable(self) -> None:
        self._state()["enabled"] = True

    def _disable(self) -> None:
        self._state()["enabled"] = False

    def _reset_positions(self) -> None:
        for state in self._mechanisms.values():
            state["position"] = [0.0, 0.0]


if __name__ == "__main__":
    TestAperture.run_server()
