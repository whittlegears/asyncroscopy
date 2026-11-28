"""ANALYSIS Tango device: measured scores an operator or LLM agent can act on.

Every command takes data keys and returns JSON. A key is a Tiled key, or a path
to a local .h5 file (as the twins return when no Tiled server runs).
The MCP server turns each command into an agent tool, so the agent passes keys,
not images, and gets numbers with units, the target they are judged against,
and pass or fail.

The targets are this device's properties. The operator sets them in the YAML;
the agent can read them with get_targets but cannot change them. A property
left at 0 means "no target".
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import h5py
import numpy as np
import tango
from tango import DevState
from tango.server import Device, command, device_property

from asyncroscopy.analysis.correction import correction_score
from asyncroscopy.analysis.drift import DEFAULT_MAX_DRIFT_PER_FRAME_PX, measure_drift
from asyncroscopy.analysis.resolution import fft_resolution

# Names the data writer stores with every scanned image (see data_writer.scan_attrs).
PIXEL_SIZE = "pixel_size_m"
FRAME_TIME = "frame_time_s"
START_TIME = "acquisition_start_utc"
TILED_TIMEOUT_MS = 120_000


def _first_image(group, read):
    """The first 2D image in a group: under "image" if present, then by name.

    ``group`` is an h5py file or a Tiled container; both have keys() and [].
    ``read(node)`` returns (array, attrs) for an array node, or None for a group.
    """
    names = sorted(group.keys(), key=lambda name: (name != "image", name))
    for name in names:
        node = group[name]
        found = read(node)
        if found is None and hasattr(node, "keys"):
            found = _first_image(node, read)
        if found is not None:
            return found
    return None


def _h5_read(node):
    if isinstance(node, h5py.Dataset):
        return (node[()], dict(node.attrs)) if node.ndim == 2 else None
    return None


def _tiled_read(node):
    shape = getattr(node, "shape", None)
    if shape is None:
        return None  # a container
    return (np.asarray(node.read()), dict(node.metadata or {})) if len(shape) == 2 else None


def _is_local_file(key: str) -> bool:
    """A key names a local file if it exists and carries a directory or a drive letter.

    A bare name such as "stem_image.h5" is always a Tiled key, so a stray file
    with that name in the working directory is never read by mistake.
    """
    directory, _ = os.path.split(key)
    drive, _ = os.path.splitdrive(key)
    return bool(directory or drive) and os.path.isfile(key)


def _target(value: float) -> float | None:
    """Device properties use 0 for "no target"."""
    return float(value) or None


class ANALYSIS(Device):
    """Scores for drift, image resolution and aberration correction."""

    data_device_address = device_property(dtype=str, default_value="asyncroscopy/data/default")
    accelerating_voltage_v = device_property(
        dtype=float, default_value=0.0, doc="Accelerating voltage, for the electron wavelength. Must be set."
    )
    convergence_semi_angle_mrad = device_property(
        dtype=float, default_value=0.0, doc="Probe convergence semi-angle in use: the correction target."
    )
    max_drift_per_frame_px = device_property(
        dtype=float, default_value=DEFAULT_MAX_DRIFT_PER_FRAME_PX, doc="Largest acceptable drift during one frame."
    )
    target_resolution_m = device_property(dtype=float, default_value=0.0, doc="Resolution images should reach.")
    expected_spacings_m = device_property(
        dtype=str, default_value="{}", doc='JSON of lattice spacings to look for, e.g. {"Au(111)": 2.355e-10}.'
    )

    def init_device(self) -> None:
        Device.init_device(self)
        self._spacings = {str(k): float(v) for k, v in json.loads(self.expected_spacings_m or "{}").items()}
        self._tiled = None
        self.set_state(DevState.ON)

    def _read(self, key: str) -> tuple[np.ndarray, dict]:
        """The first 2D image under a data key, with its stored metadata."""
        if _is_local_file(key):
            with h5py.File(key, "r") as h5:
                found = _first_image(h5, _h5_read)
        else:
            if self._tiled is None:
                from tiled.client import from_uri

                data = tango.DeviceProxy(self.data_device_address)
                data.set_timeout_millis(TILED_TIMEOUT_MS)
                self._tiled = from_uri(json.loads(data.get_config())["uri"],
                                       api_key=os.environ.get("ASYNCROSCOPY_TILED_API_KEY", "secret"))
            found = _first_image(self._tiled[key], _tiled_read)
        if found is None:
            raise ValueError(f"No 2D image found under {key!r}")
        return found

    @staticmethod
    def _attr(attrs: dict, name: str, key: str):
        if name not in attrs:
            raise ValueError(f"{key!r} has no {name!r} metadata")
        value = attrs[name]
        return value.decode() if isinstance(value, bytes) else value

    @command(dtype_out=str)
    def get_targets(self) -> str:
        """Return the targets every score is judged against."""
        return json.dumps(
            {
                "convergence_semi_angle_mrad": _target(self.convergence_semi_angle_mrad),
                "max_drift_per_frame_px": float(self.max_drift_per_frame_px),
                "target_resolution_m": _target(self.target_resolution_m),
                "expected_spacings_m": self._spacings,
            }
        )

    @command(dtype_in=str, dtype_out=str)
    def drift(self, keys_json: str) -> str:
        """Drift between two frames of the same area. Input: JSON list of two data keys, oldest first."""
        keys = json.loads(keys_json)
        if not isinstance(keys, list) or len(keys) != 2:
            raise ValueError('drift expects a JSON list of two keys, e.g. ["first.h5", "second.h5"]')
        (first, first_attrs), (second, second_attrs) = self._read(keys[0]), self._read(keys[1])
        pixel_size = float(self._attr(first_attrs, PIXEL_SIZE, keys[0]))
        if not np.isclose(pixel_size, float(self._attr(second_attrs, PIXEL_SIZE, keys[1]))) or first.shape != second.shape:
            raise ValueError("The two frames have different pixel sizes or image sizes")
        started = [datetime.fromisoformat(self._attr(a, START_TIME, k)) for a, k in ((first_attrs, keys[0]), (second_attrs, keys[1]))]
        result = measure_drift(
            first,
            second,
            pixel_size_m=pixel_size,
            seconds_between=(started[1] - started[0]).total_seconds(),
            frame_time_s=float(self._attr(second_attrs, FRAME_TIME, keys[1])),
            max_drift_per_frame_px=float(self.max_drift_per_frame_px),
        )
        return json.dumps({"keys": keys, **result.as_dict()})

    @command(dtype_in=str, dtype_out=str)
    def resolution(self, key: str) -> str:
        """Finest spacing one image shows (FFT resolution), and which expected spacings show."""
        image, attrs = self._read(key)
        result = fft_resolution(
            image,
            pixel_size_m=float(self._attr(attrs, PIXEL_SIZE, key)),
            expected_spacings_m=self._spacings,
            target_m=_target(self.target_resolution_m),
        )
        return json.dumps({"key": key, **result.as_dict()})

    @command(dtype_in=str, dtype_out=str)
    def correction(self, tableau_json: str) -> str:
        """Score a Standard or Enhanced tableau. Input: the JSON text acquire_tableau returned."""
        if not self.accelerating_voltage_v:
            raise ValueError("Set this device's accelerating_voltage_v property before scoring tableaus")
        payload = json.loads(tableau_json)
        aberrations = payload.get("result", payload).get("aberrations") or {}
        score = correction_score(
            aberrations,
            float(self.accelerating_voltage_v),
            target_mrad=_target(self.convergence_semi_angle_mrad),
        )
        return json.dumps(score.as_dict())


if __name__ == "__main__":
    ANALYSIS.run_server()
