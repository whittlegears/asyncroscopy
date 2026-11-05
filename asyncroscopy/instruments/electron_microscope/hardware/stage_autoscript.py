"""
AutoScript-backed STAGE Tango device.

This device exposes the same public stage API as the generic STAGE device, but
reads and writes the physical stage through AutoScript.
"""

import math
from tango import DevState
from tango.server import device_property

from asyncroscopy.instruments.electron_microscope.hardware.stage import STAGE
# AutoScript imports — only available on the microscope PC.
# Wrapped in try/except so the device can still be imported and tested
# on a development machine without AutoScript installed.
try:
    from autoscript_tem_microscope_client import TemMicroscopeClient
    _AUTOSCRIPT_AVAILABLE = True
except ImportError:
    _AUTOSCRIPT_AVAILABLE = False


def changes_z_or_tilt(current, target: tuple) -> bool:
    """True when ``target`` (x, y, z, a[, b]; m and rad, None for an axis left alone) moves z or a tilt away
    from ``current`` (an AutoScript StagePosition): such a move goes through absolute_move_safe."""
    if not math.isclose(float(target[2]), float(current.z), rel_tol=0.0, abs_tol=1e-9):
        return True
    for wanted, now in zip(target[3:], (current.a, current.b)):
        if wanted is not None and (now is None or not math.isclose(float(wanted), float(now), rel_tol=0.0, abs_tol=1e-5)):
            return True
    return False


class AutoScriptSTAGE(STAGE):
    """AutoScript-backed STAGE device.

    We use meters for x/y/z and degrees for alpha/beta.
    AutoScript expects alpha/beta in radians, so this device converts at the
    hardware boundary.
    """

    # ------------------------------------------------------------------
    # Device properties — set per-deployment in the Tango DB
    # ------------------------------------------------------------------
    hardware_host = device_property(
        dtype=str,
        default_value="10.46.217.241",
        doc="Hostname or IP of the AutoScript microscope server",
    )
    hardware_port = device_property(
        dtype=int,
        default_value=9095,
        doc="Port of the AutoScript microscope server",
    )
    simulated = device_property(
        dtype=bool,
        default_value=False,
        doc="True when this stack drives a simulator, set from run_servers' top-level simulated:. "
        "False (real hardware): a stage without AutoScript is in FAULT.",
    )

    # ------------------------------------------------------------------
    # Attributes
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_device(self) -> None:
        super().init_device()
        self._connect_autoscript()

    def _connect_autoscript(self) -> None:
        """Connect the AutoScript client; without it (no package, or a refused connection) the device is in FAULT
        with the reason as its Status, except on a simulated stack where a missing package is only a warning."""
        self._microscope = None
        if not _AUTOSCRIPT_AVAILABLE:
            reason = "autoscript_tem_microscope_client is not installed in this environment"
            if self.simulated:
                self.warn_stream(f"AutoScript not available ({reason}); the stack is simulated")
                return
            # On hardware a stage that cannot reach AutoScript must not look ready.
            self.set_state(DevState.FAULT)
            self.set_status(f"Not connected to AutoScript: {reason}")
            self.error_stream(f"AutoScript STAGE not available: {reason}")
            return

        try:
            self._microscope = TemMicroscopeClient()
            self._microscope.connect(self.hardware_host, self.hardware_port)
            self.info_stream(f"Connected AutoScript STAGE at {self.hardware_host}:{self.hardware_port}")
        except Exception as exc:
            self._microscope = None
            self.set_state(DevState.FAULT)
            self.set_status(f"Not connected to AutoScript at {self.hardware_host}:{self.hardware_port}: {exc}")
            self.error_stream(f"AutoScript STAGE connection failed: {exc}")

    # ------------------------------------------------------------------
    # Attribute read / write
    # ------------------------------------------------------------------

    def _read_position(self) -> list[float]:
        """Read [x, y, z, alpha, beta] from AutoScript, exposing tilts in degrees."""
        pos = self._microscope.specimen.stage.position
        beta_tilt_enabled = self._read_beta_tilt_enabled()
        position = [
            float(pos.x),
            float(pos.y),
            float(pos.z),
            float("nan") if pos.a is None else math.degrees(float(pos.a)),
            0.0 if not beta_tilt_enabled or pos.b is None else math.degrees(float(pos.b)),
        ]

        return position

    def _write_position(self, value) -> None:
        """Move AutoScript stage to [x, y, z, alpha, beta], with tilts supplied in degrees.

        A move that changes z or a tilt goes through absolute_move_safe, which AutoScript runs with the tilt
        set to zero first, so the holder does not swing into the pole piece on the way; an x/y move at the
        same height and tilt uses absolute_move.
        """
        position = [float(component) for component in value]
        beta_tilt_enabled = self._read_beta_tilt_enabled()

        if beta_tilt_enabled:
            if len(position) != 5:
                raise ValueError("Stage position must be [x, y, z, alpha, beta]")

            x, y, z, a, b = position
            a = None if math.isnan(a) else math.radians(a)
            b = None if math.isnan(b) else math.radians(b)
            position = (x, y, z, a, b)
        else:
            if len(position) not in (4, 5):
                raise ValueError("Stage position must be [x, y, z, alpha] or [x, y, z, alpha, beta]")
            if len(position) == 5 and not math.isnan(position[4]) and position[4] != 0.0:
                raise ValueError(f"stage beta = {position[4]:.6g}, but the fitted holder has no beta tilt: give beta 0; the stage did not move")

            x, y, z, a = position[:4]
            a = None if math.isnan(a) else math.radians(a)
            position = (x, y, z, a)

        stage = self._microscope.specimen.stage
        if changes_z_or_tilt(stage.position, position):
            stage.absolute_move_safe(position)
        else:
            stage.absolute_move(position)

    def _read_beta_tilt_enabled(self) -> bool:
        """Return True if the stage supports beta tilt, False otherwise."""
        stage_type = self._microscope.specimen.stage.get_holder_type()
        if stage_type == "DoubleTilt":
            return True
        else:
            return False

# ----------------------------------------------------------------------
# Server entry point
# ----------------------------------------------------------------------

AutoscriptSTAGE = AutoScriptSTAGE


if __name__ == "__main__":
    AutoScriptSTAGE.run_server()
