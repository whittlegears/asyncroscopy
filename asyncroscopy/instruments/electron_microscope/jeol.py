"""
JEOL electron microscope Tango device.

This module starts from the JEOL implementation that exists on upstream/main
and adapts it to the instrument-centered package layout.
"""

import numpy as np
import tango
from tango import DevState
from tango.server import device_property

from asyncroscopy.data.data_writer import DEFAULT_ACQUISITION_DIR, save_acquisition
from asyncroscopy.instruments.electron_microscope.electron_microscope import ElectronMicroscope

try:
    from PyJEM import detector

    _PYJEM_AVAILABLE = True
except ImportError:
    _PYJEM_AVAILABLE = False


class JeolMicroscope(ElectronMicroscope):
    """
    JEOL microscope adapter.

    Detector-specific settings such as dwell time and resolution are stored in
    dedicated detector devices and read via DeviceProxy at acquisition time.
    """

    hardware_host = device_property(
        dtype=str,
        default_value='localhost',
        doc='Hostname or IP of the JEOL microscope control server. PyJEM REST '
        'clients default to localhost (the scope PC); set this to point at the '
        'scope from another machine.',
    )
    hardware_port = device_property(
        dtype=int,
        default_value=0,
        doc='Not used: the PyJEM detector REST service has its own default port.',
    )
    hardware_timeout_seconds = device_property(
        dtype=int,
        default_value=120,
        doc='Hardware connection timeout in seconds.',
    )
    acquisition_save_directory = device_property(
        dtype=str,
        default_value=DEFAULT_ACQUISITION_DIR,
        doc='Directory where JEOL acquisitions are saved before the Tiled server serves them.',
    )
    acquisition_file_format = device_property(
        dtype=str,
        default_value='h5',
        doc='Acquisition file format. HDF5 stores acquisition data and parsed metadata attributes.',
    )
    data_device_address = device_property(
        dtype=str,
        default_value='',
        doc="Optional Tango device address for the DATA device, e.g. 'asyncroscopy/data/default'.",
    )

    def _connect(self):
        self._connect_hardware()
        self._connect_detector_proxies()
        self.set_state(DevState.ON)

    def _connect_hardware(self) -> None:
        """Point the PyJEM ``detector`` module, which is itself the REST handle, at the JEOL server.

        ``set_port`` is not called: the detector REST service has its own default port, and
        ``hardware_port`` (AutoScript's) is unused here.
        """
        if not _PYJEM_AVAILABLE or self.testing_mode_bool:
            self.warn_stream('PyJEM not available; running JEOL device in testing mode.')
            return
        try:
            detector.set_ip(self.hardware_host)
            # PyJEM 1.3.0.3564 has no check_connection(); this REST round trip raises when TEMCenter is
            # unreachable. An empty list means reachable but unconfigured.
            attached = detector.get_attached_detector()
            self._microscope = detector
            self.info_stream(
                f'Connected to JEOL/PyJEM detector server at {self.hardware_host}; '
                f'attached detectors: {attached}'
            )
        except Exception as exc:
            self.error_stream(f'JEOL/PyJEM connection failed: {exc}')
            self.set_state(DevState.FAULT)
            self._microscope = None


    @staticmethod
    def _decode_rawdata(raw: bytes, width: int, height: int) -> np.ndarray:
        """
        Decode a PyJEM ``snapshot_rawdata()`` byte buffer into a 2D image.

        PyJEM returns raw little-endian int16 bytes; the shape comes from the
        detector's imaging area. Reshape order follows the PyJEM docs
        (``(width, height)``) and is unverified for non-square scans.
        """
        return np.frombuffer(raw, dtype=np.dtype('<i2')).reshape((width, height))

    def _acquire_scanned_image(
        self,
        imsize: int,
        dwell_time: float,
        detector_list: list[str] = ['haadf'],
        scan_region: list[float] = [0.0, 0.0, 1.0, 1.0],
        output_format: str = '.h5',
    ) -> str:
        """
        Acquire a STEM scan over the requested detectors via the PyJEM
        ``detector`` module and return the saved acquisition's DATA/Tiled key.

        PyJEM acquisition is synchronous, so each detector is scanned in turn.
        ``dwell_time`` is given in seconds; the PyJEM API takes microseconds.
        """
        if not _PYJEM_AVAILABLE:
            tango.Except.throw_exception(
                'UnsupportedCommand',
                'PyJEM is not installed; cannot acquire on JEOL hardware.',
                '_acquire_scanned_image()',
            )

        detector_list = [d.upper() for d in detector_list]
        full_frame = [0.0, 0.0, 1.0, 1.0]
        images = []
        image_settings = []
        for name in detector_list:
            det = detector.Detector(name)
            if scan_region == full_frame:
                det.set_scanmode(0)                          # full-frame Scan
                det.set_imaging_area(imsize, imsize)
            else:
                extent = det.get_detectorsetting()['ImagingArea']['Width']
                left, top, right, bottom = scan_region
                det.set_scanmode(3)                          # Area (sub-region)
                det.set_areamode_imagingarea(
                    Width=int((right - left) * extent),
                    Height=int((bottom - top) * extent),
                    X=int(left * extent),
                    Y=int(top * extent),
                )
            det.set_exposuretime_value(dwell_time * 1e6)     # PyJEM API takes microseconds while AutoScript takes seconds -- Lets keep default to seconds
            settings = det.get_detectorsetting()
            area = settings['AreaModeImagingArea'] if scan_region != full_frame else settings['ImagingArea']
            pixels = self._decode_rawdata(det.snapshot_rawdata(), area['Width'], area['Height'])
            images.append(pixels)
            image_settings.append(settings)

        data_server = self._detector_proxies.get('data')
        return save_acquisition(
            self, data_server, 'stem_image', detector_list, images,
            dataset_attrs=image_settings, output_format=output_format,
        )

    def _acquire_spectrum(self, detector_name: str, exposure_time: float) -> str:
        tango.Except.throw_exception('UnsupportedCommand', 'JEOL spectrum acquisition is not implemented yet.', '_acquire_spectrum()')

    def _set_screen_current(self, current):
        tango.Except.throw_exception('UnsupportedCommand', 'JEOL screen current control is not implemented yet.', '_set_screen_current()')

    def _get_screen_current(self):
        tango.Except.throw_exception('UnsupportedCommand', 'JEOL screen current readback is not implemented yet.', '_get_screen_current()')

    def _move_stage(self, position):
        tango.Except.throw_exception('UnsupportedCommand', 'JEOL stage motion is not implemented yet.', '_move_stage()')

    def _get_stage(self):
        tango.Except.throw_exception('UnsupportedCommand', 'JEOL stage readback is not implemented yet.', '_get_stage()')

    def _set_fov(self, fov):
        tango.Except.throw_exception('UnsupportedCommand', 'JEOL field-of-view control is not implemented yet.', '_set_fov()')

    def _get_fov(self):
        tango.Except.throw_exception('UnsupportedCommand', 'JEOL field-of-view readback is not implemented yet.', '_get_fov()')

    def _auto_focus(self):
        tango.Except.throw_exception('UnsupportedCommand', 'JEOL autofocus is not implemented yet.', '_auto_focus()')

    def _set_image_shift(self, shift):
        tango.Except.throw_exception('UnsupportedCommand', 'JEOL image shift control is not implemented yet.', '_set_image_shift()')


if __name__ == '__main__':
    JeolMicroscope.run_server()
