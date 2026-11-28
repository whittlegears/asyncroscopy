"""Backend-neutral aberration-corrector Tango interface."""

from __future__ import annotations

import json
import math
from abc import abstractmethod
from typing import Any

from tango import AttrWriteType, DevState, DevString
from tango.server import Device, attribute, command, device_property


# Aberration names in the CEOS-RPC protocol (version 4).
ABERRATION_NAMES = {"WG", "We", "WD", "C1", "A1", "A2", "B2", "C3", "A3", "S3", "A4", "D4", "B4", "C5", "A5", "R5", "S5"}
# Round aberrations have no direction: CEOS sends them as [x, 0].
ROUND_ABERRATIONS = {"C1", "C3", "C5"}


def _parse_limits(text: str, property_name: str) -> dict[str, float]:
    """Parse a JSON object of aberration name -> positive limit."""
    limits = json.loads(text or "{}")
    if not isinstance(limits, dict):
        raise ValueError(f"{property_name} must be a JSON object of aberration name -> limit")
    unknown = sorted(set(limits) - ABERRATION_NAMES)
    if unknown:
        raise ValueError(f"{property_name} names unknown aberrations: {unknown}")
    parsed = {name: float(value) for name, value in limits.items()}
    if any(not value > 0 for value in parsed.values()):
        raise ValueError(f"{property_name} limits must be > 0")
    return parsed


def correction_size(values: list[float]) -> float:
    """Size of one correction: |x|, or the vector length of [x, y]."""
    return math.hypot(*values)


class CORRECTOR(Device):
    """Common Tango API implemented by real and simulated correctors.

    The public commands deliberately retain the CEOS-compatible JSON response
    structure used by the existing notebooks. Concrete implementations only
    supply backend operations.
    """

    max_step = device_property(
        dtype=str,
        default_value="{}",
        doc=(
            "JSON object: largest single correction allowed per aberration, in the "
            "aberration's own unit (m; rad for WD), e.g. {\"C1\": 50e-9}. "
            "Aberrations not listed are not limited. Set by the instrument's operators."
        ),
    )
    max_total_since_tableau = device_property(
        dtype=str,
        default_value="{}",
        doc=(
            "JSON object: largest sum of correction sizes per aberration between two "
            "tableaus. Stops many small steps from adding up past max_step."
        ),
    )

    status_message = attribute(
        label="Status message",
        dtype=str,
        access=AttrWriteType.READ,
        doc="Last backend status string",
    )

    def init_device(self) -> None:
        Device.init_device(self)
        self.set_state(DevState.INIT)
        self._last_status = "Initialising"
        self._simulation_aberrations: dict[str, Any] = {}
        self._max_step = _parse_limits(self.max_step, "max_step")
        self._max_total = _parse_limits(self.max_total_since_tableau, "max_total_since_tableau")
        self._applied_since_tableau: dict[str, float] = {}
        self._connect_backend()

    def read_status_message(self) -> str:
        return self._last_status

    @command(dtype_out=DevString)
    def get_info(self) -> str:
        return self._get_info()

    @command(dtype_in=str, dtype_out=DevString)
    def acquire_tableau(self, args: str) -> str:
        parts = args.strip().split()
        if len(parts) != 2:
            raise ValueError("acquire_tableau expects '<type> <angle>'")
        tab_type, angle = parts
        result = self._acquire_tableau(tab_type, float(angle))
        # A successful measurement resets the running total of corrections.
        if "error" not in json.loads(result):
            self._applied_since_tableau = {}
        return result

    @command(dtype_out=DevString)
    def measure_c1a1(self) -> str:
        return self._measure_c1a1()

    @command(dtype_in=str, dtype_out=DevString)
    def correct_aberration(self, args: str) -> str:
        parts = args.strip().split()
        if len(parts) < 2:
            raise ValueError("correct_aberration expects '<name> <value> [value]' ")
        name = parts[0]
        values = [float(value) for value in parts[1:]]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Refused {name} correction: values must be finite numbers, got {values}")
        if name in ROUND_ABERRATIONS and len(values) == 2 and values[1] != 0.0:
            raise ValueError(f"Refused {name} correction: {name} has no direction, so its y value must be 0")
        self._check_limits(name, values)
        result = self._correct_aberration(name, values)
        self._applied_since_tableau[name] = (
            self._applied_since_tableau.get(name, 0.0) + correction_size(values)
        )
        return result

    @command(dtype_out=DevString)
    def get_correction_limits(self) -> str:
        """Return the correction limits and how much of each total is used."""
        return json.dumps(
            {
                "max_step": self._max_step,
                "max_total_since_tableau": self._max_total,
                "applied_since_tableau": self._applied_since_tableau,
            }
        )

    def _check_limits(self, name: str, values: list[float]) -> None:
        """Refuse a correction that is larger than the configured limits."""
        size = correction_size(values)
        step_limit = self._max_step.get(name)
        if step_limit is not None and size > step_limit:
            raise ValueError(
                f"Refused {name} correction of size {size:.3g} (m; rad for WD): larger than "
                f"the max_step limit of {step_limit:.3g}. Correct in smaller steps or "
                "ask the operator to change the limit."
            )
        total_limit = self._max_total.get(name)
        if total_limit is not None:
            total = self._applied_since_tableau.get(name, 0.0) + size
            if total > total_limit:
                raise ValueError(
                    f"Refused {name} correction: corrections since the last tableau "
                    f"would reach {total:.3g} (m; rad for WD), above the max_total_since_tableau "
                    f"limit of {total_limit:.3g}. Acquire a tableau to re-measure first."
                )

    @command
    def reconnect(self) -> None:
        self._connect_backend()

    @command(dtype_in=str)
    def set_aberrations_coeff_sim(self, json_aberrations_string: str) -> None:
        coefficients = json.loads(json_aberrations_string)
        if not isinstance(coefficients, dict):
            raise ValueError("Simulation aberrations must be a JSON object")
        self._set_simulation_aberrations(coefficients)

    @command(dtype_out=str)
    def get_aberrations_coeff_sim(self) -> str:
        return json.dumps(self._get_simulation_aberrations())

    def _set_simulation_aberrations(self, coefficients: dict[str, Any]) -> None:
        self._simulation_aberrations = coefficients

    def _get_simulation_aberrations(self) -> dict[str, Any]:
        return self._simulation_aberrations

    @abstractmethod
    def _connect_backend(self) -> None:
        """Connect or initialize the concrete backend."""

    @abstractmethod
    def _get_info(self) -> str:
        """Return a CEOS-compatible JSON response string."""

    @abstractmethod
    def _acquire_tableau(self, tab_type: str, angle: float) -> str:
        """Acquire aberrations and return a CEOS-compatible JSON response."""

    @abstractmethod
    def _measure_c1a1(self) -> str:
        """Measure first-order aberrations."""

    @abstractmethod
    def _correct_aberration(self, name: str, values: list[float]) -> str:
        """Apply a correction using backend-native semantics."""


if __name__ == "__main__":
    CORRECTOR.run_server()
