"""
STAGE Tango device.

This device holds params for the scan.
It does NOT talk to AutoScript directly — the STEMMicroscope device
reads these attributes via DeviceProxy before acquiring.
"""

import json
import math
import time
from pathlib import Path

from tango import AttrWriteType, DevState, DevVarFloatArray
from tango.server import Device, attribute, device_property

from asyncroscopy.utils.commands import action_hints_attribute, command
from abc import abstractmethod

TILT_LIMITS_DEG = {"alpha": (-35.0, 35.0), "beta": (-20.0, 20.0)}
HOLDER_KEYS = {"z_m": "z", "alpha_deg": "alpha", "beta_deg": "beta"}


def holder_profiles(text: str) -> dict[str, dict[str, tuple[float, float]]]:
    """The ``holders`` property: {"name": {"z_m": [min, max], "alpha_deg": [...], "beta_deg": [...]}}, every key optional."""
    try:
        raw = json.loads(text or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"holders must be JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("holders must be a JSON object of holder name -> limits")
    profiles = {}
    for name, limits in raw.items():
        if not isinstance(limits, dict) or set(limits) - set(HOLDER_KEYS):
            raise ValueError(f"holder {name!r}: limits are an object with keys from {sorted(HOLDER_KEYS)}")
        profiles[str(name)] = {HOLDER_KEYS[key]: STAGE._pair(value, f"holder {name!r} {key}") for key, value in limits.items()}
    return profiles

def narrow_travel(travel: dict[str, tuple[float, float]], hardware: dict[str, tuple[float, float]]) -> tuple[dict, list[str]]:
    """The configured ``travel`` narrowed to the instrument's ``hardware`` limits, axis by axis, and the axes where the
    two do not overlap (those keep the configured travel, and the device refuses every move)."""
    narrowed, apart = dict(travel), []
    for axis, (low, high) in hardware.items():
        both = (max(low, travel[axis][0]), min(high, travel[axis][1]))
        if both[0] < both[1]:
            narrowed[axis] = both
        else:
            apart.append(f"{axis} (configured {list(travel[axis])}, the instrument's {[low, high]})")
    return narrowed, apart


class STAGE(Device):
    """Stage settings device: travel, holder limits and position. Which sample is loaded is the SAMPLE device's.

    Public stage vectors are always [x, y, z, alpha, beta], with x/y/z in
    meters and alpha/beta in degrees.
    """

    action_hints = action_hints_attribute()

    # Conservative defaults for a TEM goniometer; set them to the travel of your stage.
    x_limits_m = device_property(dtype=(float,), default_value=[-1e-3, 1e-3], doc="[min, max] stage x travel (m).")
    y_limits_m = device_property(dtype=(float,), default_value=[-1e-3, 1e-3], doc="[min, max] stage y travel (m).")
    z_limits_m = device_property(dtype=(float,), default_value=[-5e-4, 5e-4], doc="[min, max] stage z travel (m).")
    # No holder values are built in; fill these in from your holders' specifications.
    holders = device_property(dtype=str, default_value="{}", doc='JSON {"holder name": {"z_m": [min, max], "alpha_deg": [min, max], "beta_deg": [min, max]}}; every key optional.')
    holder = device_property(dtype=str, default_value="", doc="The holder installed when no holder_file says otherwise (a name in holders); blank: none.")
    # Conservative placeholders; set both from the fitted holder's pole piece clearance before tilting far.
    high_tilt_deg = device_property(dtype=float, default_value=15.0, doc="Tilt (deg, alpha or beta) above which |z| is held to high_tilt_max_abs_z_m. Set from the fitted holder.")
    high_tilt_max_abs_z_m = device_property(dtype=float, default_value=1e-4, doc="Largest |z| (m) allowed while tilted beyond high_tilt_deg. Set from the fitted holder.")
    holder_file = device_property(dtype=str, default_value="", doc="Where set_holder records the installed holder, so a restart keeps it; blank: outputs/stage/<device name>_holder.json.")
    position_error_m = device_property(
        dtype=float, default_value=0.0,
        doc="1-sigma distance between where the stage lands and where it reports (m), from its specification; reported as "
        "position_sigma_m, which a Mapper without its own stage_sigma_m uses to bound the stitching search. 0: unknown.",
    )

    beta_tilt_enabled = attribute(
        label="Beta tilt enabled",
        dtype=bool,
        access=AttrWriteType.READ,
        doc="Whether the beta tilt is enabled on this stage",
    )

    position = attribute(
        label="Position",
        dtype=(float,),
        max_dim_x=5,
        access=AttrWriteType.READ_WRITE,
        unit="m, m, m, deg, deg",
        doc="Stage position [x, y, z, alpha, beta], with tilts in degrees",
    )

    x = attribute(
        label="Position",
        dtype=float,
        access=AttrWriteType.READ_WRITE,
        unit="m",
        doc="Stage X position in meters (limits from the x_limits_m property)",
    )

    y = attribute(
        label="Position",
        dtype=float,
        access=AttrWriteType.READ_WRITE,
        unit="m",
        doc="Stage Y position in meters (limits from the y_limits_m property)",
    )

    z = attribute(
        label="Position",
        dtype=float,
        access=AttrWriteType.READ_WRITE,
        unit="m",
        doc="Stage Z position in meters (limits from the z_limits_m property)",
    )

    alpha = attribute(
        label="Alpha tilt",
        dtype=float,
        access=AttrWriteType.READ_WRITE,
        unit="degrees",
        min_value = -35,
        max_value = 35,
        doc="Stage alpha tilt in degrees",
    )

    beta = attribute(
        label="Beta tilt",
        dtype=float,
        access=AttrWriteType.READ_WRITE,
        unit="degrees",
        min_value = -20,
        max_value = 20,
        doc="Stage beta tilt in degrees",
    )

    position_sigma_m = attribute(
        dtype=float,
        unit="m",
        access=AttrWriteType.READ,
        doc="1-sigma distance between where the stage lands and where it reports (m); 0 when unknown (position_error_m).",
    )


    def init_device(self) -> None:
        Device.init_device(self)
        self.set_state(DevState.ON)
        self._position = [0.0, 0.0, 0.0, 0.0, 0.0]
        self._configured_travel = {
            "x": self._pair(self.x_limits_m, "x_limits_m"),
            "y": self._pair(self.y_limits_m, "y_limits_m"),
            "z": self._pair(self.z_limits_m, "z_limits_m"),
            **TILT_LIMITS_DEG,
        }
        self._connect()
        self._hardware_limits = self._read_hardware_limits()
        self._travel, apart = narrow_travel(self._configured_travel, self._hardware_limits)
        self._holders = holder_profiles(self.holders)
        self._holder = ""
        self._frozen = None
        self._limits_conflict: str | None = None
        self._holder_path = Path(self.holder_file or f"outputs/stage/{self.get_name().replace('/', '_')}_holder.json")
        installed, problem = self._restore_holder()
        self._apply_holder(installed)
        if problem:
            # Unknown holder: hold the stage to what every configured holder allows.
            strictest = dict(self._limits)
            for name in self._holders:
                for axis, (low, high) in self._limits_for(name).items():
                    strictest[axis] = (max(low, strictest[axis][0]), min(high, strictest[axis][1]))
            empty = [axis for axis, (low, high) in strictest.items() if not low < high]
            if empty:
                self._frozen = f"{problem}; the configured holders share no {', '.join(empty)} range, so no move is allowed until set_holder says which is fitted"
            else:
                self._limits = strictest
                self._apply_limits()
            self._alarm(self._frozen or f"{problem}; until set_holder says which is fitted, moves are held to the limits every configured holder allows.")
        if apart:
            self._limits_conflict = self._conflict(apart)
            self._alarm(self._limits_conflict)

        self.info_stream("STAGE device initialised")


    def read_position(self) -> list[float]:
        return self._read_position()

    def read_position_sigma_m(self) -> float:
        return float(self.position_error_m)

    def write_position(self, value: list[float]) -> None:
        self._refuse_if_held()
        self._check_limits(value)
        self._write_position(value)

    def _refuse_if_held(self) -> None:
        for reason in (getattr(self, "_limits_conflict", None), getattr(self, "_frozen", None)):
            if reason:
                raise ValueError(f"the stage did not move: {reason}")

    @staticmethod
    def _conflict(apart: list[str]) -> str:
        return f"the configured stage limits do not overlap the instrument's for {', '.join(apart)}; no move is allowed until the configuration is fixed and the device restarted"

    def _alarm(self, status: str) -> None:
        """ALARM with ``status``, unless already in FAULT, whose reason stays first."""
        if self.get_state() == DevState.FAULT:
            fault = str(self.get_status()).strip()
            self.set_status(status if status in fault else f"{fault}\n{status}")
            return
        self.set_state(DevState.ALARM)
        self.set_status(status)

    def _reread_hardware_limits(self) -> None:
        """Re-read the axis limits (a holder change can bring new ones) and narrow the travel to them."""
        fresh = self._read_hardware_limits()
        self._hardware_limits = {**self._hardware_limits, **fresh}
        self._travel, apart = narrow_travel(self._configured_travel, self._hardware_limits)
        if apart and not self._limits_conflict:
            self._limits_conflict = self._conflict(apart)

    def _check_limits(self, value: list[float]) -> None:
        """Refuse a position outside the limits (with the installed holder's name), or a high tilt away from eucentric height."""
        for axis, v in zip(("x", "y", "z", "alpha", "beta"), value):
            low, high = self._limits[axis]
            if not low <= float(v) <= high:
                holder = getattr(self, "_holder", "")
                where = f"holder {holder!r}'s limits" if holder else "the stage limits"
                raise ValueError(f"stage {axis} = {float(v):.6g} is outside {where} [{low}, {high}]")
        self._check_z_and_tilt(value)

    def _check_z_and_tilt(self, value: list[float]) -> None:
        """Refuse a beta tilt on a holder without one (never drop it), and a high tilt far from eucentric height."""
        z, alpha = float(value[2]), float(value[3])
        beta = float(value[4]) if len(value) > 4 else 0.0
        if beta != 0.0 and not math.isnan(beta) and not self.read_beta_tilt_enabled():
            raise ValueError(f"stage beta = {beta:.6g}, but the fitted holder has no beta tilt: give beta 0; the stage did not move")
        tilt = max((abs(t) for t in (alpha, beta) if not math.isnan(t)), default=0.0)
        if tilt > float(self.high_tilt_deg) and abs(z) > float(self.high_tilt_max_abs_z_m):
            raise ValueError(
                f"stage z = {z:.6g} m with a tilt of {tilt:.6g} deg is refused: beyond {float(self.high_tilt_deg):g} deg, |z| must be "
                f"at most {float(self.high_tilt_max_abs_z_m):g} m (high_tilt_deg, high_tilt_max_abs_z_m; the holder could touch the pole piece)"
            )

    def _limits_for(self, holder: str) -> dict[str, tuple[float, float]]:
        """The travel, narrowed by ``holder``'s profile (blank: none)."""
        limits = dict(self._travel)
        for axis, (low, high) in self._holders.get(holder, {}).items():
            limits[axis] = (max(low, limits[axis][0]), min(high, limits[axis][1]))
            if not limits[axis][0] < limits[axis][1]:
                raise ValueError(f"holder {holder!r} {axis} limits [{low}, {high}] do not overlap the stage travel {self._travel[axis]}")
        return limits

    def _restore_holder(self) -> tuple[str, str | None]:
        """The holder set_holder last recorded, or the holder property when none was recorded; and why the record was not used."""
        configured = str(self.holder).strip()
        if not self._holder_path.exists():
            return configured, None
        try:
            saved = json.loads(self._holder_path.read_text(encoding="utf-8"))["holder"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return configured, f"the recorded holder in {self._holder_path} could not be read ({exc})"
        if saved and saved not in self._holders:
            return configured, f"the recorded holder {saved!r} is not in the holders property"
        return str(saved), None

    def _apply_holder(self, holder: str) -> None:
        if holder and holder not in self._holders:
            raise ValueError(f"unknown holder {holder!r}; holders configured: {sorted(self._holders) or 'none (set the holders property)'}")
        self._limits = self._limits_for(holder)
        self._holder = holder
        self._apply_limits()

    def _apply_limits(self) -> None:
        attributes = self.get_device_attr()
        for axis in ("x", "y", "z", "alpha", "beta"):
            low, high = self._limits[axis]
            attr = attributes.get_attr_by_name(axis)
            attr.set_min_value(low)
            attr.set_max_value(high)

    @command(dtype_out=str, risk="low", role="setting.read")
    def get_holder(self) -> str:
        """The installed holder and the stage limits it leaves: JSON {"holder", "holders", "limits": {axis: [min, max]}} (m, deg),
        and "hardware_limits", the axis limits the instrument itself reports (empty when the backend reads none)."""
        return json.dumps({"holder": self._holder or None, "holders": sorted(self._holders), "frozen": self._frozen,
                           "limits_conflict": self._limits_conflict,
                           "limits": {axis: list(pair) for axis, pair in self._limits.items()},
                           "hardware_limits": {axis: list(pair) for axis, pair in self._hardware_limits.items()}})

    @command(dtype_in=str, dtype_out=str, doc_in="Name of the holder now installed (one of holders), or blank for none.", risk="high", operator_only=True, role="setting.set")
    def set_holder(self, name: str) -> str:
        """Say which holder is installed, so moves are checked against its Z and tilt limits. Refused while the stage
        is outside them: move it inside first."""
        name = name.strip()
        if name and name not in self._holders:
            raise ValueError(f"unknown holder {name!r}; holders configured: {sorted(self._holders) or 'none (set the holders property)'}")
        self._reread_hardware_limits()
        limits = self._limits_for(name)  # raises for a holder whose limits miss the stage travel, before anything changes
        for axis, value in zip(("x", "y", "z", "alpha", "beta"), self.read_position()):
            low, high = limits[axis]
            if not low <= float(value) <= high:
                raise ValueError(f"stage {axis} = {float(value):.6g} is outside holder {name!r}'s limits [{low}, {high}]; move it inside first")
        self._holder_path.parent.mkdir(parents=True, exist_ok=True)
        self._holder_path.write_text(json.dumps({"holder": name, "at": time.time()}), encoding="utf-8")
        self._apply_holder(name)
        self._frozen = None
        if self._limits_conflict:
            self._alarm(self._limits_conflict)
        elif self.get_state() == DevState.ALARM:
            self.set_state(DevState.ON)
            self.set_status(f"holder {name or 'none'}")
        elif self.get_state() != DevState.FAULT:
            self.set_status(f"holder {name or 'none'}")
        return self.get_holder()

    @command(dtype_in=DevVarFloatArray, doc_in=":param delta: [dx, dy, dz, dalpha, dbeta]; m for x, y, z and degrees for the tilts.",
             risk="high", operator_only=True, role="stage.move")
    def move_relative(self, delta: list[float]) -> None:
        """Move the stage by delta from where it is: the target (here + delta) is checked against the stage and holder
        limits and sent as an absolute move, as move_stage is. Refused while the stage is moving, since "here" is not yet known."""
        self._refuse_if_held()
        step = [float(v) for v in delta]
        if len(step) != 5 or not all(math.isfinite(v) for v in step):
            raise ValueError(f"delta must be five finite numbers [dx, dy, dz, dalpha, dbeta], got {list(delta)}; the stage did not move")
        if self._is_moving():
            raise ValueError("the stage is moving, so where a relative move would end is not known; wait for it to stop. The stage did not move")
        here = [float(v) for v in self.read_position()]
        target = [h + d for h, d in zip(here, step)]
        self._check_limits(target)
        self._write_position(target)

    @staticmethod
    def _pair(value, name: str) -> tuple[float, float]:
        low, high = (float(v) for v in value)
        if not low < high:
            raise ValueError(f"{name} must be [min, max] with min < max, got {list(value)}")
        return low, high

    def read_beta_tilt_enabled(self) -> bool:
        return self._read_beta_tilt_enabled()

    def _connect(self) -> None:
        """Connect the backend, once, at init; a backend that cannot connect sets FAULT with the reason as its Status."""

    def _read_hardware_limits(self) -> dict[str, tuple[float, float]]:
        """{axis: (min, max)} in m and deg as the instrument reports them; {} for a backend that reads none."""
        return {}

    @abstractmethod
    def _is_moving(self) -> bool:
        pass

    @abstractmethod
    def _read_position(self):
        pass

    @abstractmethod
    def _write_position(self, value):
        pass

    @abstractmethod
    def _read_beta_tilt_enabled(self):
        pass

    def read_x(self) -> float:
        return self.read_position()[0]

    def write_x(self, value: float) -> None:
        position = self.read_position()
        position[0] = float(value)
        self.write_position(position)

    def read_y(self) -> float:
        return self.read_position()[1]

    def write_y(self, value: float) -> None:
        position = self.read_position()
        position[1] = float(value)
        self.write_position(position)

    def read_z(self) -> float:
        return self.read_position()[2]

    def write_z(self, value: float) -> None:
        position = self.read_position()
        position[2] = float(value)
        self.write_position(position)

    def read_alpha(self) -> float:
        return self.read_position()[3]

    def write_alpha(self, value: float) -> None:
        position = self.read_position()
        position[3] = float(value)
        self.write_position(position)

    def read_beta(self) -> float:
        position = self.read_position()
        if len(position) < 5:
            return 0.0
        return position[4]

    def write_beta(self, value: float) -> None:
        if not self.read_beta_tilt_enabled():
            if float(value) == 0.0:
                return
            raise ValueError("This stage does not support beta tilt")
        position = self.read_position()
        position[4] = float(value)
        self.write_position(position)


if __name__ == "__main__":
    STAGE.run_server()
