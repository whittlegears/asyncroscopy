"""
Electron microscope Tango device.

Detector settings are read from the corresponding detector DeviceProxy
so that each detector device is the single source of truth for its own params.

Return convention for image commands
-------------------------------------
Image commands return a string supplied by the concrete microscope
implementation, typically a DATA/Tiled unique id.
"""

import json
from abc import abstractmethod
from typing import Optional

import numpy as np
import tango
from tiled.client import from_uri
from tango import AttrWriteType, DevEncoded, DevFloat, DevString, DevState, DevVarFloatArray, DevVarStringArray
from tango.server import attribute, command, device_property

from asyncroscopy.data.data_reader import describe_tiled_node
from asyncroscopy.instruments.instrument import Instrument

MAX_CACHED_IMAGE_KEYS = 64


class ElectronMicroscope(Instrument):
    """
    Top-level electron microscope device.

    Detector-specific settings such as dwell time and resolution are stored in
    dedicated detector devices and read via DeviceProxy at acquisition time.
    """

    scan_device_address = device_property(
        dtype=str,
        doc="Tango device address for the SCAN settings device. "
        "DB mode: 'test/detector/scan' "
        "No-DB mode: 'tango://127.0.0.1:8888/test/nodb/scan#dbase=no'",
    )

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

    stage_device_address = device_property(
        dtype=str,
        doc="Tango device address for the STAGE settings device. "
        "DB mode: 'asyncroscopy/stage/default' "
        "No-DB mode: 'tango://127.0.0.1:8888/asyncroscopy/stage/default#dbase=no'",
    )

    camera_device_address = device_property(
        dtype=str,
        doc="Tango device address for the CAMERA settings. "
        "DB mode: 'asyncroscopy/camera/default' "
        "No-DB mode: 'tango://127.0.0.1:8888/asyncroscopy/camera/default#dbase=no'",
    )

    stem_mode = attribute(
        label="STEM Mode",
        dtype=bool,
        access=AttrWriteType.READ,
        doc="True when the microscope is in STEM mode",
    )

    def _init_device_attributes(self) -> None:
        self._microscope: Optional[object] = None
        self._stem_mode: bool = False
        self._detector_proxies: dict[str, tango.DeviceProxy] = {}

    def read_instrument_type(self) -> str:
        return 'TEM'

    @abstractmethod
    def _connect(self):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _connect; "
            "this vendor backend is missing the override"
        )

    def _disconnect(self):
        self._microscope = None
        self.info_stream('Disconnected from microscope hardware')

    @abstractmethod
    def _connect_hardware(self) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _connect_hardware; "
            "this vendor backend is missing the override"
        )

    @abstractmethod
    def _connect_detector_proxies(self) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _connect_detector_proxies; "
            "this vendor backend is missing the override"
        )

    def read_stem_mode(self) -> bool:
        return self._stem_mode

    @command
    def Disconnect(self) -> None:
        """Disconnect from microscope hardware gracefully."""
        self.set_state(DevState.OFF)
        self._disconnect()

    @command(
        dtype_in=str,
        dtype_out=str,
        doc_in=":param detector_name: Spectrum detector name, e.g. 'eds'.",
    )
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
        return self._remember_acquired_key(self._acquire_spectrum(detector_name, proxy.exposure_time))

    @command(
        dtype_in=DevVarStringArray,
        dtype_out=str,
        doc_in=":param detector_list: Scanning detector names, e.g. ['haadf']. "
               "An empty list uses ['haadf'].",
    )
    def acquire_scanned_image(self, detector_list: list[str] = ['haadf']) -> str:
        """
        Acquire an image with scanning detectors and return its DATA/Tiled key.
        The default detector list is ['haadf'].
        """
        # Tango's wire protocol has no concept of "omitted", so remote callers (e.g. the
        # MCP/LLM bridge) can and do send an empty list instead of relying on the Python
        # default above; treat that the same as not specifying detectors.
        if not detector_list:
            detector_list = ['haadf']
        scan = self._detector_proxies.get('scan')
        return self._remember_acquired_key(
            self._acquire_scanned_image(scan.imsize, scan.dwell_time, detector_list, list(scan.scan_region), scan.output_format)
        )

    @command(dtype_out=str)
    def acquire_scanned_data_advanced(self) -> str:
        """Trigger an advanced 4D scanned data acquisition with the Ceta camera."""
        scan = self._detector_proxies.get('scan')
        return self._remember_acquired_key(
            self._acquire_scanned_data_advanced(scan.imsize, scan.dwell_time, 'BM-Ceta', list(scan.scan_region))
        )

    @command(dtype_out=str)
    def acquire_camera_image(self) -> str:
        """Acquire a camera image using settings from the camera device."""
        camera = self._detector_proxies.get('camera')
        return self._remember_acquired_key(self._acquire_camera_image(
            camera.imsize,
            camera.exposure_time,
            camera.camera_detector,
            camera.readout_area,
            camera.frame_combining,
            camera.electron_counting,
            camera.output_format,
        ))

    def _remember_acquired_key(self, key: str) -> str:
        """Remember a DATA/Tiled key so get_image_data_cached() can look it up by index."""
        if not hasattr(self, '_cached_images'):
            self._cached_images: list[str] = []
        self._cached_images.append(key)
        del self._cached_images[:-MAX_CACHED_IMAGE_KEYS]
        return key

    @command(dtype_in=int, dtype_out=DevEncoded)
    def get_image_data_cached(self, index: int) -> tuple[str, bytes]:
        """
        Retrieve dataset metadata and a small preview for a previously acquired
        image, by index into the keys returned by the acquire_* commands (most
        recent first up to MAX_CACHED_IMAGE_KEYS). Reads from the DATA/Tiled
        server, the same source get_data_from_key (MCP) reads from.
        """
        if not hasattr(self, '_cached_images') or not self._cached_images:
            tango.Except.throw_exception('NoCache', 'Call an acquire_* command first', 'get_image_data_cached()')
        if index < 0 or index >= len(self._cached_images):
            tango.Except.throw_exception(
                'InvalidIndex',
                f'Index {index} out of range for {len(self._cached_images)} cached key(s)',
                'get_image_data_cached()',
            )

        # _cached_images appends in acquisition order; index 0 is documented as
        # the MOST RECENT acquisition, so index from the end.
        key = self._cached_images[-(index + 1)]
        data_proxy = self._detector_proxies.get('data')
        if data_proxy is None:
            tango.Except.throw_exception(
                'NoDataDevice',
                'No DATA device is configured for this instrument; cannot resolve cached keys.',
                'get_image_data_cached()',
            )

        config = json.loads(data_proxy.get_config())
        uri = config.get('uri')
        if not uri:
            tango.Except.throw_exception(
                'NoTiledUri', 'The DATA device did not provide a Tiled URI.', 'get_image_data_cached()',
            )

        client = from_uri(uri)
        try:
            node = client[key]
        except KeyError:
            tango.Except.throw_exception(
                'KeyNotFound', f"Cached key '{key}' could not be resolved from Tiled server '{uri}'.", 'get_image_data_cached()',
            )

        description = describe_tiled_node(key, uri, node)
        preview = np.asarray(description['datasets'][0]['preview']) if description['datasets'] else np.asarray([])
        return json.dumps(description), preview.tobytes()

    @command(dtype_in=DevVarFloatArray, dtype_out=None)
    def place_beam(self, position) -> None:
        """Set resting beam position, [0:1]."""
        self._place_beam(position)

    @command(dtype_in=DevVarFloatArray, dtype_out=None)
    def place_beam_list(self, positions) -> None:
        """Place beam at multiple positions sequentially."""
        if len(positions) % 2 != 0:
            raise ValueError('Input must contain pairs of (x, y) values.')

        for i in range(0, len(positions), 2):
            x = float(positions[i])
            y = float(positions[i + 1])
            self._place_beam([x, y])

    @command(dtype_in=str)
    def set_column_valves(self, state: str) -> None:
        """Open or close the column valves."""
        self._set_column_valves(state)

    @command()
    def blank_beam(self) -> None:
        """Blank beam."""
        self._blank_beam()

    @command()
    def unblank_beam(self) -> None:
        """Unblank beam."""
        self._unblank_beam()

    @command(dtype_in=DevFloat)
    def set_defocus(self, defocus):
        """Set the defocus in meters."""
        self._set_defocus(defocus)

    @command(dtype_out=DevFloat)
    def get_defocus(self):
        """Read the defocus in meters."""
        return self._get_defocus()

    @command(dtype_in=DevFloat)
    def set_fov(self, fov):
        """Set the field of view for the next acquisition, in meters."""
        self._set_fov(fov)

    @command(dtype_out=DevFloat)
    def get_fov(self):
        """Read the field of view for the next acquisition, in meters."""
        return self._get_fov()
    
    @command(dtype_in=DevVarFloatArray)
    def set_image_shift(self, shift):
        """Set the image shift to [x_shift, y_shift] in meters."""
        self._set_image_shift(shift)

    @command(dtype_out=DevVarFloatArray)
    def get_image_shift(self):
        """Get the image shiftas [x, y] in m."""
        return self._get_image_shift()
    
    @command(dtype_out=DevVarFloatArray)
    def get_beam_tilt(self):
        """Get the current beam tilt as [alpha, beta] in radian."""
        return self._get_beam_tilt()

    @command(dtype_in=DevVarFloatArray)
    def set_beam_tilt(self, tilt):
        """Set the beam tilt to [x_tilt, y_tilt] in radian."""
        self._set_beam_tilt(tilt)

    @command(dtype_out=DevVarFloatArray)
    def get_diffraction_shift(self):
        """Get the current  diffraction shift as [alpha, beta] in radian."""
        return self._get_diffraction_shift()
    
    @command(dtype_in=DevVarFloatArray)
    def set_diffraction_shift(self, shift):
        """Set the diffraction shift to [x_shift, y_shift] in radian."""
        self._set_diffraction_shift(shift)

    @command(dtype_out=DevString)
    def get_parameters(self) -> str:
        """ Get all status parameters"""
        return self._get_parameters()

    @command(dtype_in=DevString)
    def set_screen(self, position: str)->None:
        self._set_screen(position)

    @command()
    def calibrate_screen_current(self):
        """Set the screen current in pA."""
        self._calibrate_screen_current()


    @command(dtype_in=DevFloat)
    def set_screen_current(self, current):
        """Set the screen current in pA."""
        self._set_screen_current(current)

    @command(dtype_out=DevFloat)
    def get_screen_current(self):
        """Get the screen current in pA."""
        return self._get_screen_current()

    @command(dtype_out=DevVarFloatArray)
    def get_stage(self):
        """Get the current stage position as [x, y, z, alpha, beta], with x/y/z in meters and tilts in degrees."""
        return self._get_stage()
   
    @command(dtype_in=DevVarFloatArray)
    def move_stage(self, position):
        """Move the stage to [x, y, z, alpha, beta], with x/y/z in meters and tilts in degrees."""
        self._move_stage(position)

    @command()
    def auto_focus(self):
        """Run the microscope's autofocus routine."""
        self._auto_focus()

    
    @abstractmethod
    def _acquire_spectrum(self, detector_name: str, exposure_time: float) -> str:
        """Vendor-specific spectrum acquisition implementation."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _acquire_spectrum; "
            "this vendor backend is missing the override"
        )

    @abstractmethod
    def _acquire_scanned_image(
        self,
        imsize: int,
        dwell_time: float,
        detector_list: list[str] = ['haadf'],
        scan_region: list[float] = [0.0, 0.0, 1.0, 1.0],
        output_format: str = '.h5',
    ) -> str:
        """Vendor-specific scanned image acquisition implementation."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _acquire_scanned_image; "
            "this vendor backend is missing the override"
        )

    def _acquire_camera_image(
        self,
        imsize: int,
        exposure_time: float,
        detector: str,
        readout_area: str,
        frame_combining: int = 1,
        electron_counting: bool = True,
        output_format: str = '.h5',
    ) -> str:
        """Vendor-specific camera acquisition implementation."""
        tango.Except.throw_exception(
            'UnsupportedCommand',
            'This microscope does not support camera image acquisition.',
            '_acquire_camera_image()',
        )

    def _acquire_scanned_data_advanced(self, imsize: int, dwell_time: float, detector: str, scan_region: list[float]) -> str:
        """Vendor-specific advanced 4D scanned data acquisition trigger."""
        tango.Except.throw_exception(
            'UnsupportedCommand',
            'This microscope does not support advanced scanned data acquisition.',
            '_acquire_scanned_data_advanced()',
        )

    def _place_beam(self, position):
        pass

    @abstractmethod
    def _set_column_valves(self, state: str):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _set_column_valves; "
            "this vendor backend is missing the override"
        )

    def _blank_beam(self):
        pass

    def _unblank_beam(self):
        pass

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
    def _move_stage(self, position):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _move_stage; "
            "this vendor backend is missing the override"
        )

    @abstractmethod
    def _get_stage(self):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _get_stage; "
            "this vendor backend is missing the override"
        )

    @abstractmethod
    def _get_image_shift(self):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _get_image_shift; "
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
    def _get_parameters(self):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _get_parameters; "
            "this vendor backend is missing the override"
        )

    @abstractmethod
    def _set_fov(self, fov):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _set_fov; "
            "this vendor backend is missing the override"
        )

    @abstractmethod
    def _get_fov(self):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _get_fov; "
            "this vendor backend is missing the override"
        )

    @abstractmethod
    def _auto_focus(self):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _auto_focus; "
            "this vendor backend is missing the override"
        )

    @abstractmethod
    def _set_image_shift(self, shift):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _set_image_shift; "
            "this vendor backend is missing the override"
        )


if __name__ == '__main__':
    ElectronMicroscope.run_server()
