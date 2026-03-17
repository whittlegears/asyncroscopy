"""
Microscope Tango device.

Detector settings are read from the corresponding detector DeviceProxy
so that each detector device is the single source of truth for its own params.

Return convention for image commands
-------------------------------------
All image commands return DevEncoded = (str, bytes) where:
  - str  : JSON string containing metadata (shape, dtype, dwell_time, …)
  - bytes: raw numpy array bytes (reconstruct with np.frombuffer + reshape)

Client-side reconstruction example::

    import json, numpy as np
    encoded = proxy.get_haadf_image()   # returns (json_str, raw_bytes)
    meta    = json.loads(encoded[0])
    image   = np.frombuffer(encoded[1], dtype=meta["dtype"]).reshape(meta["shape"])
"""

import json
import time
from typing import Optional

from abc import abstractmethod, ABC, ABCMeta

import numpy as np
import tango
from tango import AttrWriteType, DevEncoded, DevState, DevVarFloatArray, DevVarCharArray
from tango.server import Device, DeviceMeta, attribute, command, device_property

class CombinedMeta(DeviceMeta, ABCMeta):
    """Combines Tango DeviceMeta and ABCMeta to allow abstract methods in Devices."""
    pass

class Microscope(Device, metaclass=CombinedMeta):
    """
    Top-level TEM microscope device.
    Detector-specific settings (dwell time, resolution) are stored in
    dedicated detector devices and read via DeviceProxy at acquisition time.
    """

    # ------------------------------------------------------------------
    # Device properties — configure in Tango DB per deployment
    # ------------------------------------------------------------------

    haadf_device_address = device_property(
        dtype=str,
        doc="Tango device address for the HAADF settings device. "
            "DB mode: 'test/detector/haadf' "
            "No-DB mode: 'tango://127.0.0.1:8888/test/nodb/haadf#dbase=no'",
    )

    eds_device_address = device_property(
        dtype=str,
        doc="Tango device address for the EDS settings device. "
            "DB mode: 'test/detector/eds' "
            "No-DB mode: 'tango://127.0.0.1:8887/test/nodb/haadf#dbase=no'",
    )

    advanced_acquisition_device_address = device_property(
        dtype=str,
        doc="Tango device address for the HAADF settings device. "
            "DB mode: 'test/detector/advancedacquisition' "
            "No-DB mode: 'tango://127.0.0.1:8888/test/nodb/advancedacquisition#dbase=no'",
    )

    stage_device_address = device_property(
        dtype=str,
        doc="Tango device address for the STAGE settings device. "
            "DB mode: 'test/hardware/stage' "
            "No-DB mode: 'tango://127.0.0.1:8888/test/nodb/stage#dbase=no'",
    )

    # Add further detector device_property entries here as detectors are added
    # eels_device_address  = device_property(dtype=str, default_value="test/detector/eels")

    # ------------------------------------------------------------------
    # Attributes
    # ------------------------------------------------------------------

    stem_mode = attribute(
        label="STEM Mode",
        dtype=bool,
        access=AttrWriteType.READ,
        doc="True when the microscope is in STEM mode",
    )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def init_device(self) -> None:
        Device.init_device(self)
        self.set_state(DevState.INIT)

        self._microscope: Optional[object] = None  # TemMicroscopeClient instance
        self._stem_mode: bool = False

        # Dict mapping detector name string → DeviceProxy
        # Populated in _connect_detector_proxies
        self._detector_proxies: dict[str, tango.DeviceProxy] = {}

        self._connect()

    @abstractmethod
    def _connect(self):
        print(f"Must define a class-specific _connect() method")
    
    @abstractmethod
    def _connect_hardware(self) -> None:
        print(f"Must define a class-specific _connect_hardware() method")

    @abstractmethod
    def _connect_detector_proxies(self) -> None:
        print(f"Must define a class-specific _connect_detector_proxies() method")  

    # ------------------------------------------------------------------
    # Attribute read methods
    # ------------------------------------------------------------------

    def read_stem_mode(self) -> bool:
        # TODO: query self._microscope.optics.mode when AutoScript available
        return self._stem_mode

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    @command
    def Connect(self) -> None:
        """Explicitly (re)connect to microscope hardware. Useful after a fault.
        Also, sets the timeout fofr Tango device for 2 minutes (for larger things)
        """
        self._connect()

    @command
    def Disconnect(self) -> None:
        """Disconnect from microscope hardware gracefully."""
        # TODO: self._microscope.disconnect() when AutoScript available
        self._microscope = None
        self.set_state(DevState.OFF)
        self.info_stream("Disconnected from microscope hardware")


    @command(dtype_in=str, dtype_out=DevEncoded)
    def get_spectrum(self, detector_name: str) -> tuple[str, bytes]:
        """
        Acquire a single spectrum from the named detector with the specified exposure time.

        Parameters
        ----------
        detector_name:
            Name of the detector, e.g. "eds".

        Returns
        -------
        DevEncoded = (json_metadata, raw_bytes)
            json_metadata includes: shape, dtype, dwell_time, detector,
            timestamp, and any other relevant metadata.
            raw_bytes is the flat numpy array bytes; reshape using shape from metadata.
        """

        detector_name = detector_name.lower().strip()
        proxy = self._detector_proxies.get(detector_name)
        
        # Read acquisition settings from the detector device
        exposure_time = proxy.exposure_time # float

        adorned_spectrum = self._acquire_spectrum(detector_name, exposure_time)

        metadata = {
            "detector": detector_name,
            "dtype": str(adorned_spectrum.dtype),
            "dwell_time": exposure_time,
            "timestamp": time.time(),
            # TODO: add metadata from adorned_spectrum.metadata when using real AutoScript
        }

        return json.dumps(metadata), adorned_spectrum.tobytes()


    @command(dtype_in=str, dtype_out=DevEncoded)#In PyTango, DevEncoded is a special Tango data type designed to send binary data + a small description string together as a single return value.
    def get_image(self, detector_name: str) -> tuple[str, bytes]:
        """
        Acquire a single STEM image from the named detector.

        Parameters
        ----------
        detector_name:
            Name of the detector, e.g. "haadf". Must match a key in
            self._detector_proxies.

        Returns
        -------
        DevEncoded = (json_metadata, raw_bytes)
            json_metadata includes: shape, dtype, dwell_time, detector,
            timestamp, and any other relevant metadata.
            raw_bytes is the flat numpy array bytes; reshape using shape from metadata.
        """
        detector_name = detector_name.lower().strip()

        proxy = self._detector_proxies.get(detector_name)

        # Read acquisition settings from the detector device
        dwell_time: float = proxy.dwell_time
        width: int  = proxy.image_width
        height: int = proxy.image_height

        # TODO: map (width, height) → AutoScript ImageSize enum
        # e.g. ImageSize.PRESET_1024 when width == height == 1024

        adorned_image = self._acquire_stem_image(detector_name, width, height, dwell_time)

        metadata = {
            "detector": detector_name,
            "shape": [height, width],
            "dtype": str(adorned_image.dtype),
            "dwell_time": dwell_time,
            "timestamp": time.time(),
            # TODO: add metadata from adorned_image.metadata when using real AutoScript
        }

        return json.dumps(metadata), adorned_image.tobytes()

    @command(dtype_in=('str',), dtype_out=str)
    def get_images(self, detector_names: list[str]) -> str:
        """
        Acquire multiple STEM images simultaneously.

        Parameters
        ----------
        detector_names: list of detector names, e.g. ["HAADF", "BF"]

        Returns
        -------
        JSON string with metadata for all images and retrieval instructions
        
        Usage: Call get_image_data(index) to retrieve each image's bytes
        """
        # Normalize and validate
        detector_names = [name.lower().strip() for name in detector_names]
        
        # Get settings from AdvancedAcquisition device
        adv_acq_proxy = self._detector_proxies.get("AdvancedAcquistion")
        dwell_time = adv_acq_proxy.dwell_time
        base_resolution = adv_acq_proxy.base_resolution
        scan_region = adv_acq_proxy.scan_region
        auto_beam_blank = adv_acq_proxy.auto_beam_blank
        
        # Acquire all images
        adorned_images = self._acquire_stem_image_advanced(
            detector_names,
            base_resolution,
            scan_region,
            dwell_time,
            auto_beam_blank
        )
        
        # Package results
        # Cache and build metadata
        self._cached_images = adorned_images
        timestamp = time.time()
        
        metadata_list = []
        for i, (name, adorned_img) in enumerate(zip(detector_names, adorned_images)):
            # Access the numpy array from AdornedImage
            img_data = adorned_img.data if hasattr(adorned_img, 'data') else adorned_img
            
            metadata_list.append({
                "index": i,
                "detector": name,
                "shape": list(img_data.shape),
                "dtype": str(img_data.dtype),
                "timestamp": timestamp,
            })
        
        return json.dumps({"images": metadata_list, "count": len(adorned_images)})

    @command(dtype_in=int, dtype_out=DevEncoded)
    def get_image_data_cached(self, index: int) -> tuple[str, bytes]:
        """Retrieve cached image by index."""
        if not hasattr(self, '_cached_images'):
            tango.Except.throw_exception("NoCache", "Call get_images() first", "get_image_data()")
        if index >= len(self._cached_images):
            tango.Except.throw_exception("InvalidIndex", f"Index {index} out of range", "get_image_data()")
        
        adorned_img = self._cached_images[index]
        # Extract numpy array from AdornedImage
        img_data = adorned_img.data if hasattr(adorned_img, 'data') else adorned_img
        
        meta = {"shape": list(img_data.shape), "dtype": str(img_data.dtype)}
        return json.dumps(meta), img_data.tobytes()
    
    @command(dtype_in=DevVarFloatArray, dtype_out=None)
    def place_beam(self, position) -> None:
        """
        sets resting beam position, [0:1]
        """
        self._place_beam(position)

    @command()
    def blank_beam(self) -> None:
        """blank beam"""
        self._blank_beam()

    @command()
    def unblank_beam(self) -> None:
        """
        unblank beam
        """
        self._unblank_beam()


    @command(dtype_out=DevVarFloatArray)
    def get_stage(self):
        """
        Get the current stage position as a list of floats [x, y, z, alpha, beta].

        Returns
        -------
        DevVarFloatArray = [x, y, z, alpha, beta]

        """
        position = self._get_stage()

        return position

    @command(dtype_in=DevVarFloatArray)
    def move_stage(self, position):
        """
        Move the the stage
        to an absolute position  [x, y, z, alpha, beta]

        Parameters
        position: an absolute reference frame move position (not relative)

        """
        self._move_stage(position)

    # ------------------------------------------------------------------
    # Internal acquisition helpers
    # ------------------------------------------------------------------
    @abstractmethod
    def _acquire_stem_image():
        print("Get image")
        pass

    @abstractmethod
    def _acquire_stem_image_advanced():
        print("Get image with more flexible settings")
        pass

    def _place_beam():
        # define in the inherit class
        pass

    def _blank_beam():
        # define in the inherit class
        pass

    def _unblank_beam():
        # define in the inherit class
        pass

    @abstractmethod
    def _move_stage():
        # define in the inherit class
        pass

    @abstractmethod
    def _get_stage():
        pass
# ----------------------------------------------------------------------
# Server entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    Microscope.run_server()