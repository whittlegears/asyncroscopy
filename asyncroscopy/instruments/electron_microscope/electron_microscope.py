"""
Electron microscope Tango device.

Detector settings are read from the corresponding detector DeviceProxy
so that each detector device is the single source of truth for its own params.

What every scanning-beam instrument shares (the beam, the field of view, image shift, stage access, the
beam current, limits and the command lock) is in ``asyncroscopy.instruments.scanning_beam``. This device adds
what is particular to a TEM/STEM, and backends implement only ``_x`` methods:

* defocus and the stigmator (``set_stigmator``/``get_stigmator``, metres of A1), checked against
  ``max_defocus`` and ``max_stigmator`` before the backend sees the value;
* beam tilt, diffraction shift, the fluorescent screen and its current, the column valves;
* the camera, 4D data, spectra and spectrum images, and the corrector and detector devices.

Return convention for image commands
-------------------------------------
Image commands return a string supplied by the concrete microscope
implementation, typically a DATA/Tiled unique id.
"""

import json
import math
from abc import abstractmethod

import tango
from tango import AttrWriteType, DevDouble, DevEncoded, DevFloat, DevString, DevVarFloatArray
from tango.server import attribute, device_property

from asyncroscopy.instruments.scanning_beam import BEAM_ADDRESSES, ScanningBeamInstrument, check_scan, hardware_call, unsupported
from asyncroscopy.utils.commands import command

# Detector and sub-device proxies the base connects: {name: the device property holding its address}.
DETECTOR_ADDRESSES = {
    **BEAM_ADDRESSES,
    "eds": "eds_device_address",
    "camera": "camera_device_address",
    "corrector": "corrector_device_address",
}
# The limit properties' defaults: what the base enforces when a deployment sets none.
DEFAULT_MAX_DEFOCUS = 1e-5
DEFAULT_MAX_STIGMATOR = 1e-6
DEFAULT_MAX_SPECTRUM_IMAGE_POINTS = 64


def check_defocus(defocus: float, max_defocus: float) -> float:
    """Defocus in metres, refused beyond +/-max_defocus."""
    defocus = float(defocus)
    if not (math.isfinite(defocus) and abs(defocus) <= float(max_defocus)):
        raise ValueError(f"defocus must be within +/-{max_defocus} m (did you pass nanometres?), got {defocus}")
    return defocus


def check_stigmator(stigmator: list[float], max_stigmator: float) -> list[float]:
    """Stigmator [a, b] in metres of A1, refused beyond +/-max_stigmator per component."""
    value = [float(v) for v in stigmator]
    if len(value) != 2 or not all(math.isfinite(v) for v in value):
        raise ValueError(f"stigmator must be [a, b] in metres, got {list(stigmator)}")
    if any(abs(v) > float(max_stigmator) for v in value):
        raise ValueError(f"stigmator must be within +/-{float(max_stigmator)} m per component, got {value}")
    return value


class ElectronMicroscope(ScanningBeamInstrument):
    """
    Top-level electron microscope device (TEM/STEM).

    Detector-specific settings such as dwell time and resolution are stored in
    dedicated detector devices and read via DeviceProxy at acquisition time.
    """

    proxy_addresses = DETECTOR_ADDRESSES

    corrector_device_address = device_property(
        dtype=str,
        doc="Tango device address for the aberration corrector settings device. "
        "DB mode: 'test/hardware/corrector' "
        "No-DB mode: 'tango://127.0.0.1:8888/test/nodb/corrector#dbase=no'",
    )

    eds_device_address = device_property(
        dtype=str,
        doc="Tango device address for the EDS settings device. "
        "DB mode: 'asyncroscopy/eds/default' "
        "No-DB mode: 'tango://127.0.0.1:8887/asyncroscopy/haadf/default#dbase=no'",
    )

    camera_device_address = device_property(
        dtype=str,
        doc="Tango device address for the CAMERA settings. "
        "DB mode: 'asyncroscopy/camera/default' "
        "No-DB mode: 'tango://127.0.0.1:8888/asyncroscopy/camera/default#dbase=no'",
    )

    default_scan_detector = device_property(
        dtype=str,
        default_value="haadf",
        doc="Scanning detector used when a caller names none: acquire_scanned_image with an empty list, a Tuner plan without a detector.",
    )

    # Limits checked before any backend sees a value; action_hints tells clients the same ranges.
    max_defocus = device_property(dtype=float, default_value=DEFAULT_MAX_DEFOCUS, doc="Largest |defocus| accepted (m).")
    max_stigmator = device_property(dtype=float, default_value=DEFAULT_MAX_STIGMATOR, doc="Largest |stigmator| component accepted (m of A1).")
    max_spectrum_image_points = device_property(dtype=int, default_value=DEFAULT_MAX_SPECTRUM_IMAGE_POINTS, doc="Largest points per side acquire_spectrum_image accepts.")

    stem_mode = attribute(
        label="STEM Mode",
        dtype=bool,
        access=AttrWriteType.READ,
        doc="True when the microscope is in STEM mode",
    )

    def _init_device_attributes(self) -> None:
        super()._init_device_attributes()
        self._stem_mode: bool = False

    def read_instrument_type(self) -> str:
        return 'TEM'

    @abstractmethod
    def _connect(self):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _connect; "
            "this vendor backend is missing the override"
        )

    @abstractmethod
    def _connect_hardware(self) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _connect_hardware; "
            "this vendor backend is missing the override"
        )

    def read_stem_mode(self) -> bool:
        return self._stem_mode

    def argument_hints(self) -> dict:
        """The base's hints, plus defocus and the screen current, at this device's limits."""
        max_defocus = float(self.max_defocus)
        return {
            **super().argument_hints(),
            "set_defocus": {"args": [{"name": "defocus", "type": "number", "units": "m", "min": -max_defocus, "max": max_defocus}]},
            "get_defocus": {"returns": {"type": "number", "units": "m"}},
            "get_screen_current": {"returns": {"type": "number", "units": "pA"}},
            "set_screen_current": {"args": [{"name": "current", "type": "number", "units": "pA"}]},
        }

    @command(
        dtype_in=str,
        dtype_out=str,
        doc_in=":param detector_name: Spectrum detector name, e.g. 'eds'.",
        risk="high", role="acquire.spectrum", routine=True,
    )
    @hardware_call()
    def acquire_spectrum(self, detector_name: str) -> str:
        """Acquire a single spectrum and return its DATA/Tiled unique id."""
        detector_name = detector_name.lower().strip()
        proxy = self._detector_proxies.get(detector_name)
        if proxy is None:
            tango.Except.throw_exception(
                'UnknownDetector',
                f"No spectrum detector named '{detector_name}'. "
                f"Configured detector devices: {sorted(self._detector_proxies.keys())}.",
                'acquire_spectrum()',
            )
        return self._acquire_spectrum(detector_name, proxy.exposure_time)

    @command(dtype_in=int, dtype_out=str, doc_in=":param points: Beam positions per side of the field (2 to the max_spectrum_image_points property).", risk="high", role="acquire.other", routine=True)
    @hardware_call()
    def acquire_spectrum_image(self, points: int) -> str:
        """Acquire an EDS spectrum image of points x points positions, at the EDS device's exposure per position, and
        return its DATA/Tiled key. A backend that cannot raises UnsupportedCommand."""
        points = int(points)
        if not 2 <= points <= int(self.max_spectrum_image_points):
            raise ValueError(f"points must be between 2 and {int(self.max_spectrum_image_points)}, got {points}")
        proxy = self._detector_proxies.get("eds")
        if proxy is None:
            tango.Except.throw_exception('UnknownDetector', f"No spectrum detector named 'eds'. Configured detector devices: {sorted(self._detector_proxies.keys())}.", 'acquire_spectrum_image()')
        return self._acquire_spectrum_image(points, proxy.exposure_time)

    @command(dtype_out=str, risk="high", role="acquire.other", routine=True)
    @hardware_call()
    def acquire_scanned_data_advanced(self) -> str:
        """Trigger an advanced 4D scanned data acquisition with the Ceta camera."""
        scan = self._detector_proxies.get('scan')
        region = list(check_scan(scan.imsize, scan.dwell_time, list(scan.scan_region)))  # refused here, before the instrument
        return self._acquire_scanned_data_advanced(scan.imsize, scan.dwell_time, 'BM-Ceta', region)

    @command(dtype_out=str, risk="high", role="acquire.image")
    @hardware_call()
    def acquire_camera_image(self) -> str:
        """Acquire a camera image using settings from the camera device."""
        camera = self._detector_proxies.get('camera')
        return self._acquire_camera_image(
            camera.imsize,
            camera.exposure_time,
            camera.camera_detector,
            camera.readout_area,
            camera.frame_combining,
            camera.output_format,
        )

    @command(dtype_in=int, dtype_out=DevEncoded, doc_in=':param index: Index of a cached image.', risk="low", role="data.read")
    def get_image_data_cached(self, index: int) -> tuple[str, bytes]:
        """Retrieve cached image by index."""
        if not hasattr(self, '_cached_images'):
            tango.Except.throw_exception('NoCache', 'Call acquire_scanned_image() first', 'get_image_data()')
        if index >= len(self._cached_images):
            tango.Except.throw_exception('InvalidIndex', f'Index {index} out of range', 'get_image_data()')

        cached_image = self._cached_images[index]
        img_data = cached_image.data if hasattr(cached_image, 'data') else cached_image

        meta = {'shape': list(img_data.shape), 'dtype': str(img_data.dtype)}
        return json.dumps(meta), img_data.tobytes()

    @command(dtype_in=str, doc_in=":param state: 'open' or 'close'.", risk="high", role="setting.set")
    @hardware_call()
    def set_column_valves(self, state: str) -> None:
        """Open or close the column valves."""
        self._set_column_valves(state)

    @command(dtype_in=DevDouble, doc_in=':param defocus: Defocus in metres (e.g. 2e-8 for 20 nm).', risk="low", role="optics.focus")
    @hardware_call()
    def set_defocus(self, defocus: float) -> None:
        """Set the defocus in meters."""
        self._set_defocus(check_defocus(defocus, self.max_defocus))

    @command(dtype_out=DevDouble, risk="low", role="setting.read")
    @hardware_call()
    def get_defocus(self) -> float:
        """Read the defocus in meters."""
        return self._get_defocus()

    @command(dtype_in=DevVarFloatArray, doc_in=":param stigmator: [a, b] objective stigmator in metres of A1.", risk="high", role="setting.set", routine=True)
    @hardware_call()
    def set_stigmator(self, stigmator: list[float]) -> None:
        """Set the objective stigmator: the two A1 components it adds to the column, in metres."""
        self._set_stigmator(check_stigmator(stigmator, self.max_stigmator))

    @command(dtype_out=DevVarFloatArray, risk="low", role="setting.read")
    @hardware_call()
    def get_stigmator(self) -> list[float]:
        """Read the objective stigmator [a, b] in metres of A1; the column's own astigmatism is not included."""
        return [float(v) for v in self._get_stigmator()]

    @command(dtype_out=DevVarFloatArray, risk="low", role="setting.read")
    @hardware_call()
    def get_beam_tilt(self) -> list[float]:
        """Get the current beam tilt as [alpha, beta] in radian."""
        return self._get_beam_tilt()

    @command(dtype_in=DevVarFloatArray, doc_in=':param tilt: [x, y] beam tilt in radians.', risk="high", role="setting.set", routine=True)
    @hardware_call()
    def set_beam_tilt(self, tilt: list[float]) -> None:
        """Set the beam tilt to [x_tilt, y_tilt] in radian."""
        self._set_beam_tilt(tilt)

    @command(dtype_out=DevVarFloatArray, risk="low", role="setting.read")
    @hardware_call()
    def get_diffraction_shift(self) -> list[float]:
        """Get the current diffraction shift as [alpha, beta] in radian."""
        return self._get_diffraction_shift()
    
    @command(dtype_in=DevVarFloatArray, doc_in=':param shift: [x, y] diffraction shift.', risk="high", role="setting.set")
    @hardware_call()
    def set_diffraction_shift(self, shift: list[float]) -> None:
        """Set the diffraction shift to [x_shift, y_shift] in radian."""
        self._set_diffraction_shift(shift)

    @command(dtype_in=DevString, doc_in=":param position: 'inserted' or 'retracted'.", risk="high", role="setting.set")
    @hardware_call()
    def set_screen(self, position: str) -> None:
        """Insert or retract the fluorescent screen."""
        self._set_screen(position)

    @command(risk="high", role="other")
    @hardware_call()
    def calibrate_screen_current(self) -> None:
        """Measure the beam current on the screen, so dose estimates use a measured current."""
        self._calibrate_screen_current()


    @command(dtype_in=DevFloat, doc_in=':param current: Screen current in picoamperes (pA).', risk="high", role="setting.set")
    @hardware_call()
    def set_screen_current(self, current: float) -> None:
        """Set the screen current in pA."""
        self._set_screen_current(current)

    @command(dtype_out=DevFloat, risk="low", role="setting.read")
    @hardware_call()
    def get_screen_current(self) -> float:
        """Get the screen current in pA."""
        return self._get_screen_current()

    @command(risk="high", role="optics.focus", routine=True)
    @hardware_call()
    def auto_focus(self) -> None:
        """Run the microscope's autofocus routine."""
        self._auto_focus()

    
    def _acquire_camera_image(
        self,
        imsize: int,
        exposure_time: float,
        detector: str,
        readout_area: str,
        frame_combining: int = 1,
        output_format: str = '.h5',
    ) -> str:
        """Vendor-specific camera acquisition implementation."""
        tango.Except.throw_exception(
            'UnsupportedCommand',
            'This microscope does not support camera image acquisition.',
            '_acquire_camera_image()',
        )

    def _acquire_spectrum_image(self, points: int, exposure_time: float) -> str:
        """Vendor-specific EDS spectrum image; the base backend has none."""
        unsupported(self, "acquire_spectrum_image")

    def _acquire_scanned_data_advanced(self, imsize: int, dwell_time: float, detector: str, scan_region: list[float]) -> str:
        """Vendor-specific advanced 4D scanned data acquisition trigger."""
        tango.Except.throw_exception(
            'UnsupportedCommand',
            'This microscope does not support advanced scanned data acquisition.',
            '_acquire_scanned_data_advanced()',
        )

    def _set_stigmator(self, stigmator: list[float]) -> None:
        """Set the objective stigmator [a, b] (metres of A1, already checked against max_stigmator)."""
        unsupported(self, "set_stigmator")

    def _get_stigmator(self) -> list[float]:
        """The objective stigmator [a, b] in metres of A1."""
        unsupported(self, "get_stigmator")

    def _set_defocus(self, defocus):
        pass

    @abstractmethod
    def _get_defocus(self):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _get_defocus; "
            "this vendor backend is missing the override"
        )

    @abstractmethod
    def _set_screen(self, position):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _set_screen; "
            "this vendor backend is missing the override"
        )

    @abstractmethod
    def _set_screen_current(self, current):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _set_screen_current; "
            "this vendor backend is missing the override"
        )
    
    @abstractmethod
    def _calibrate_screen_current(self):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _calibrate_screen_current; "
            "this vendor backend is missing the override"
        )

    @abstractmethod
    def _get_screen_current(self):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _get_screen_current; "
            "this vendor backend is missing the override"
        )

    @abstractmethod
    def _get_beam_tilt(self):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _get_beam_tilt; "
            "this vendor backend is missing the override"
        )

    @abstractmethod
    def _set_beam_tilt(self,tilt):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _set_beam_tilt; "
            "this vendor backend is missing the override"
        )

    @abstractmethod
    def _get_diffraction_shift(self):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _get_diffraction_shift; "
            "this vendor backend is missing the override"
        )

    @abstractmethod
    def _set_diffraction_shift(self, tilt):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _set_diffraction_shift; "
            "this vendor backend is missing the override"
        )

    @abstractmethod
    def _auto_focus(self):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _auto_focus; "
            "this vendor backend is missing the override"
        )


if __name__ == '__main__':
    ElectronMicroscope.run_server()
