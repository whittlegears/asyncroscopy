"""
AutoScript-backed STAGE Tango device.

This device exposes the same public stage API as the generic STAGE device, but
reads and writes the physical stage through AutoScript.
"""

import math
from tango import DevState
from tango.server import device_property

from asyncroscopy.core.names import autoscript_port
from asyncroscopy.instruments.electron_microscope.hardware.stage import STAGE
from asyncroscopy.utils.commands import command
# AutoScript is installed only on the microscope PC; elsewhere the device still imports for tests.
try:
    from autoscript_tem_microscope_client import TemMicroscopeClient
    from autoscript_tem_microscope_client.enumerations import StageAxis
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


NO_HOST = ("the device property hardware_host is not set: give it in the stack YAML (devices.stage.properties.hardware_host, as configs/Spectra300.yaml does); "
           "there is no default address")


class AutoScriptSTAGE(STAGE):
    """AutoScript-backed STAGE device.

    We use meters for x/y/z and degrees for alpha/beta.
    AutoScript expects alpha/beta in radians, so this device converts at the
    hardware boundary.
    """

    hardware_host = device_property(
        dtype=str,
        default_value="",
        doc="Hostname or IP of the AutoScript microscope server. Required: there is no default, so a device never "
        "reaches an instrument its stack YAML does not name.",
    )
    hardware_port = device_property(
        dtype=int,
        default_value=autoscript_port,
        doc="Port of the AutoScript microscope server",
    )
    simulated = device_property(
        dtype=bool,
        default_value=False,
        doc="True when this stack drives a simulator, set from run_servers' top-level simulated:. "
        "False (real hardware): a stage without AutoScript is in FAULT.",
    )


    def _connect(self) -> None:
        self._connect_autoscript()

    def _read_hardware_limits(self) -> dict[str, tuple[float, float]]:
        """Each axis's limits from AutoScript (get_axis_limits: m, and rad for the tilts, given here in degrees). Beta is
        read only on a double-tilt holder; set_holder reads them again, so a double tilt fitted later brings its beta
        limits. An axis AutoScript cannot answer for is left to the configured limits and logged."""
        if getattr(self, "_microscope", None) is None:
            return {}
        stage = self._microscope.specimen.stage
        axes = [("x", StageAxis.X), ("y", StageAxis.Y), ("z", StageAxis.Z), ("alpha", StageAxis.A)]
        if self._read_beta_tilt_enabled():
            axes.append(("beta", StageAxis.B))
        found = {}
        for axis, name in axes:
            try:
                limits = stage.get_axis_limits(name)
                low, high = float(limits.min), float(limits.max)
            except Exception as exc:
                self.warn_stream(f"AutoScript gave no {axis} limits ({exc}); the configured ones apply")
                continue
            found[axis] = (math.degrees(low), math.degrees(high)) if axis in ("alpha", "beta") else (low, high)
        return found

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
            if not self.hardware_host:
                raise ValueError(NO_HOST)
            self._microscope = TemMicroscopeClient()
            self._microscope.connect(self.hardware_host, self.hardware_port)
            self.info_stream(f"Connected AutoScript STAGE at {self.hardware_host}:{self.hardware_port}")
        except Exception as exc:
            self._microscope = None
            self.set_state(DevState.FAULT)
            self.set_status(f"Not connected to AutoScript at {self.hardware_host}:{self.hardware_port}: {exc}")
            self.error_stream(f"AutoScript STAGE connection failed: {exc}")


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

    def _is_moving(self) -> bool:
        if self._microscope is None:
            raise RuntimeError("not connected to AutoScript, so whether the stage is moving is not known; the stage did not move")
        return bool(self._microscope.specimen.stage.is_moving)

    @command(risk="low", stop=True, role="safe_stop")
    def stop_jogging(self) -> None:
        """Stop jogging on all stage axes (AutoScript stop_jogging). A move started with move_stage runs to its end."""
        if self._microscope is None:
            raise RuntimeError("not connected to AutoScript, so jogging could not be stopped; stop it at the microscope's stage controls")
        self._microscope.specimen.stage.stop_jogging()

    def _read_beta_tilt_enabled(self) -> bool:
        """Return True if the stage supports beta tilt, False otherwise."""
        stage_type = self._microscope.specimen.stage.get_holder_type()
        if stage_type == "DoubleTilt":
            return True
        else:
            return False


AutoscriptSTAGE = AutoScriptSTAGE


if __name__ == "__main__":
    AutoScriptSTAGE.run_server()
