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
import time
from abc import abstractmethod
from typing import Optional

import tango
from tango import AttrWriteType, DevEncoded, DevFloat, DevString, DevState, DevVarFloatArray, DevVarStringArray
from tango.server import attribute, command, device_property

from asyncroscopy.instruments.instrument import Instrument


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
        return self._acquire_spectrum(detector_name, proxy.exposure_time)

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
        return self._acquire_scanned_image(scan.imsize, scan.dwell_time, detector_list, list(scan.scan_region), scan.output_format)

    @command(dtype_out=str)
    def acquire_scanned_data_advanced(self) -> str:
        """Trigger an advanced 4D scanned data acquisition with the Ceta camera."""
        scan = self._detector_proxies.get('scan')
        return self._acquire_scanned_data_advanced(scan.imsize, scan.dwell_time, 'BM-Ceta', list(scan.scan_region))

    @command(dtype_out=str)
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

    @command(dtype_in=int, dtype_out=DevEncoded)
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

    @command(
        dtype_in=str,
        doc_in=':param spec: JSON object with keys: rows (int), cols (int), '
        'overlap (float 0-0.5, default 0.15), detectors (list, default ["haadf"]), '
        'eds (bool, acquire an EDS point spectrum at each tile centre, default false), '
        'settle_s (float, pause after each stage move, default 0), '
        'snake (bool, serpentine traversal, default true), '
        'calibrate (bool, default true: first measure which way each stage axis moves the image), '
        'calibration_step_fraction (float, test move as a fraction of the field of view, default 0.2), '
        'calibration_fov_m (float, field of view for the test moves; use a lower magnification when the '
        'mapping field of view shows only a lattice. Default: the current field of view), '
        'stage_to_image_px_per_m (2 x 2 list, a known calibration from an earlier manifest; skips measuring), '
        'flip_x / flip_y (bool, only with calibrate false: invert the assumed stage direction), '
        'return_to_start (bool, default true). Example: {"rows": 3, "cols": 3}',
        dtype_out=str,
        doc_out='JSON grid manifest: {"grid": [rows, cols], "overlap", "fov_m", "imsize_px", '
        '"pixel_size_nm", "stage_origin_m", "calibration", "tiles": [{"row", "col", "key", '
        '"stage_xy_m" (read back after the move), "commanded_xy_m", "eds_key"}]}. Pass the tiles '
        'to the mapping MCP server (a separate MCP connection - see start_map/add_tiles) to stitch '
        'a zoomable sample map; this command only drives the stage and camera.',
    )
    def acquire_map_grid(self, spec: str) -> str:
        """Acquire an N x M grid of overlapping images by stepping the stage, for building a zoomable sample map.

        Before the grid it measures which way each stage axis moves the image (one
        small move per axis), so tile (r, c + 1) is always to the right of tile (r, c)
        in the image, whatever the scan rotation or axis convention. The scan stops
        with an error if that measurement fails its checks. Returns a JSON manifest of
        DATA keys; stitch it on the mapping MCP server (start_map/add_tiles/finalize_map).
        """
        import numpy as np

        from asyncroscopy.mapping.calibration import (
            DEFAULT_STEP_FRACTION,
            StageCalibration,
            measure_stage_to_image,
            plan_calibrated_grid,
        )

        cfg = json.loads(spec) if spec and spec.strip() else {}
        rows = int(cfg.get('rows', 3))
        cols = int(cfg.get('cols', 3))
        overlap = float(cfg.get('overlap', 0.15))
        detectors = list(cfg.get('detectors', ['haadf']))
        want_eds = bool(cfg.get('eds', False))
        settle_s = float(cfg.get('settle_s', 0.0))
        if rows < 1 or cols < 1 or rows * cols > 400:
            raise ValueError('rows*cols must be between 1 and 400')
        if not 0.0 < overlap < 0.6:
            raise ValueError('overlap must be between 0 and 0.6')

        scan = self._detector_proxies.get('scan')
        if scan is None:
            raise ValueError('acquire_map_grid needs a SCAN device')
        eds_proxy = self._detector_proxies.get('eds') if want_eds else None
        if want_eds and eds_proxy is None:
            raise ValueError('EDS requested but no EDS device is configured')

        fov_raw = self._get_fov()
        if fov_raw is None or float(fov_raw) <= 0:
            raise ValueError('Field of view is not set; call set_fov before acquire_map_grid')
        fov = float(fov_raw)
        imsize = int(scan.imsize)
        pixel_size_m = fov / imsize

        def acquire_key() -> str:
            return self._acquire_scanned_image(imsize, scan.dwell_time, detectors, list(scan.scan_region), scan.output_format)

        def settle_move(position) -> None:
            self._move_stage([float(v) for v in position])
            if settle_s > 0:
                time.sleep(settle_s)

        start = [float(v) for v in self._get_stage()]
        if cfg.get('stage_to_image_px_per_m') is not None:
            calibration = StageCalibration.from_matrix(cfg['stage_to_image_px_per_m'], pixel_size_m)
        elif bool(cfg.get('calibrate', True)):
            step_fraction = float(cfg.get('calibration_step_fraction', DEFAULT_STEP_FRACTION))
            # At atomic resolution a lattice can fool the correlation, so the test moves
            # may run at a lower magnification. The matrix scales with 1 / pixel size;
            # this assumes the scan rotation is the same at both magnifications.
            cal_fov = float(cfg.get('calibration_fov_m') or fov)
            if cal_fov != fov:
                self._set_fov(cal_fov)
            try:
                coarse = measure_stage_to_image(
                    acquire=lambda: self._read_scanned_image(acquire_key()),
                    move=settle_move,
                    start=start,
                    pixel_size_m=cal_fov / imsize,
                    step_m=step_fraction * cal_fov,
                )
            finally:
                if cal_fov != fov:
                    self._set_fov(fov)
            calibration = coarse
            if cal_fov != fov:
                calibration = StageCalibration(
                    (np.asarray(coarse.px_per_m) * (cal_fov / fov)).tolist(),
                    pixel_size_m,
                    coarse.step_m,
                    coarse.confidence,
                    coarse.length_error,
                    coarse.axis_cosine,
                    calibration_fov_m=cal_fov,
                )
        else:
            calibration = StageCalibration.nominal(
                pixel_size_m, flip_x=bool(cfg.get('flip_x', False)), flip_y=bool(cfg.get('flip_y', False))
            )
        order = plan_calibrated_grid(rows, cols, imsize * (1.0 - overlap), calibration, snake=bool(cfg.get('snake', True)))

        tiles = []
        try:
            for cell in order:
                target = list(start)
                target[0] = start[0] + cell['offset_m'][0]
                target[1] = start[1] + cell['offset_m'][1]
                settle_move(target)
                actual = [float(v) for v in self._get_stage()]
                entry = {
                    'row': cell['row'],
                    'col': cell['col'],
                    'key': acquire_key(),
                    'stage_xy_m': actual[:2],
                    'commanded_xy_m': target[:2],
                }
                if eds_proxy is not None:
                    self._place_beam([0.5, 0.5])
                    entry['eds_key'] = self._acquire_spectrum('eds', eds_proxy.exposure_time)
                tiles.append(entry)
        finally:
            if bool(cfg.get('return_to_start', True)):
                self._move_stage(start)

        return json.dumps(
            {
                'grid': [rows, cols],
                'overlap': overlap,
                'fov_m': fov,
                'imsize_px': imsize,
                'pixel_size_nm': pixel_size_m * 1e9,
                'stage_origin_m': start[:2],
                'calibration': calibration.as_dict(),
                'tiles': tiles,
            }
        )

    def _read_scanned_image(self, key: str):
        """The first 2D image saved under a DATA key: a local HDF5 path, or a Tiled key."""
        import os

        import h5py
        import numpy as np

        from asyncroscopy.analysis.analysis_device import _first_image, _h5_read, _is_local_file, _tiled_read

        if _is_local_file(key):
            with h5py.File(key, 'r') as h5:
                found = _first_image(h5, _h5_read)
        else:
            from tiled.client import from_uri

            data = self._detector_proxies.get('data')
            if data is None:
                raise ValueError(f'{key!r} is not a local file and no DATA device is configured to resolve it')
            client = from_uri(json.loads(data.get_config())['uri'], api_key=os.environ.get('ASYNCROSCOPY_TILED_API_KEY', 'secret'))
            found = _first_image(client[key], _tiled_read)
        if found is None:
            raise ValueError(f'No 2D image found under {key!r}')
        return np.asarray(found[0], dtype=float)

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
