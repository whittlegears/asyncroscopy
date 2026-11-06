"""Backend-neutral aberration-corrector Tango interface.

Two typed commands are what jobs (the Tuner) use, whatever the backend:

* ``measure_aberrations``: the first-order aberrations as JSON of one fixed schema,
  ``{"C1": [c1], "A1": [a1_x, a1_y]}``, every value in metres;
* ``correct``: one correction, ``([values in metres], [name])``, for example
  ``([2e-9], ["C1"])``. A correction subtracts the values sent from the aberration.

Each backend implements ``_measure_aberrations`` and ``_correct`` and keeps its wire
format to itself. The CEOS-compatible commands (``measure_c1a1``,
``correct_aberration``, ``acquire_tableau``, ``get_info``) stay for notebooks that use
the CEOS JSON-RPC replies directly.
"""

from __future__ import annotations

import json
import math
from abc import abstractmethod
from typing import Any

import tango
from tango import AttrWriteType, DevState, DevString, DevVarDoubleStringArray
from tango.server import Device, attribute

from asyncroscopy.utils.commands import action_hints_attribute, command

# Components of each aberration a corrector reports (CEOS naming): one for round ones, two for the rest.
ABERRATION_COMPONENTS = {"WD": 2, "C1": 1, "A1": 2, "B2": 2, "A2": 2, "C3": 1, "S3": 2, "A3": 2, "D4": 2, "B4": 2, "A4": 2}
FIRST_ORDER = ("C1", "A1")


def check_values(name: str, values: list[float]) -> list[float]:
    """``values`` for aberration ``name`` as finite floats of the right length; ValueError otherwise."""
    if name not in ABERRATION_COMPONENTS:
        raise ValueError(f"unknown aberration {name!r}; known: {sorted(ABERRATION_COMPONENTS)}")
    numbers = [float(v) for v in values]
    if len(numbers) != ABERRATION_COMPONENTS[name] or not all(math.isfinite(v) for v in numbers):
        raise ValueError(f"{name} takes {ABERRATION_COMPONENTS[name]} finite value(s) in metres, got {values!r}")
    return numbers


class CORRECTOR(Device):
    """Common Tango API implemented by real and simulated correctors.

    The public commands deliberately retain the CEOS-compatible JSON response
    structure used by the existing notebooks. Concrete implementations only
    supply backend operations.
    """

    # What each command is (risk, role, stop), as its @command declares it, for the MCP server.
    action_hints = action_hints_attribute()

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
        self._connect_backend()

    def read_status_message(self) -> str:
        return self._last_status

    @command(dtype_out=DevString, risk="low", role="setting.read")
    def get_info(self) -> str:
        return self._get_info()

    @command(dtype_in=str, dtype_out=DevString, risk="high", role="acquire.other")
    def acquire_tableau(self, args: str) -> str:
        parts = args.strip().split()
        if len(parts) != 2:
            raise ValueError("acquire_tableau expects '<type> <angle>'")
        tab_type, angle = parts
        return self._acquire_tableau(tab_type, float(angle))

    @command(dtype_out=DevString, risk="high", role="other")
    def measure_c1a1(self) -> str:
        return self._measure_c1a1()

    @command(dtype_in=str, dtype_out=DevString, risk="high", role="other")
    def correct_aberration(self, args: str) -> str:
        parts = args.strip().split()
        if len(parts) < 2:
            raise ValueError("correct_aberration expects '<name> <value> [value]' ")
        name = parts[0]
        values = [float(value) for value in parts[1:]]
        return self._correct_aberration(name, values)

    @command(dtype_out=DevString, doc_out='JSON {"C1": [c1], "A1": [a1_x, a1_y]}, in metres.', risk="high", role="other")
    def measure_aberrations(self) -> str:
        """Measure the first-order aberrations (C1 and A1), in metres, in one fixed schema whatever the backend."""
        measured = self._measure_aberrations()
        try:
            return json.dumps({name: check_values(name, measured[name]) for name in FIRST_ORDER})
        except (KeyError, TypeError, ValueError) as exc:
            tango.Except.throw_exception("CorrectorMeasurement", f"the corrector's measurement has no usable C1/A1: {exc}", "measure_aberrations()")
            raise AssertionError("unreachable") from exc

    @command(dtype_in=DevVarDoubleStringArray, doc_in="([values in metres], [aberration name]), e.g. ([2e-9], ['C1']).", risk="high", role="other")
    def correct(self, request: tuple[list[float], list[str]]) -> None:
        """Correct one aberration by the values given: the corrector subtracts them from the aberration."""
        values, names = request
        if len(names) != 1:
            raise ValueError(f"correct takes one aberration name, got {list(names)!r}")
        self._correct(str(names[0]), check_values(str(names[0]), list(values)))

    @command(risk="high", role="other")
    def reconnect(self) -> None:
        self._connect_backend()

    @command(dtype_in=str, risk="high", role="setting.set")
    def set_aberrations_coeff_sim(self, json_aberrations_string: str) -> None:
        coefficients = json.loads(json_aberrations_string)
        if not isinstance(coefficients, dict):
            raise ValueError("Simulation aberrations must be a JSON object")
        self._set_simulation_aberrations(coefficients)

    @command(dtype_out=str, risk="low", role="setting.read")
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
        """Measure first-order aberrations (CEOS-compatible JSON)."""

    @abstractmethod
    def _measure_aberrations(self) -> dict[str, list[float]]:
        """Measure first-order aberrations: at least {"C1": [m], "A1": [m, m]}."""

    @abstractmethod
    def _correct(self, name: str, values: list[float]) -> None:
        """Subtract ``values`` (metres, already checked) from aberration ``name``; raise on a refusal."""

    @abstractmethod
    def _correct_aberration(self, name: str, values: list[float]) -> str:
        """Apply a correction using backend-native semantics."""


if __name__ == "__main__":
    CORRECTOR.run_server()
