# AutoScript microscope

`AutoScriptMicroscope` (`asyncroscopy/instruments/electron_microscope/auto_script.py`) is the Thermo Fisher AutoScript vendor class of the [electron microscope family](modify_base_electron_microscope.md). It owns the AutoScript connection and implements the base's `_helper` methods. Use it as the reference when you write a new vendor class.

## Connection

`_connect` opens AutoScript and the support-device proxies. If AutoScript does not connect, the state is `FAULT` and the status gives the reason. Each helper that needs AutoScript is wrapped by `_requires_microscope`. A call without a connection raises `NotConnected` and tells the caller to run `Connect` once.

In testing mode, `self._microscope` is `None`. Live reads (`fov`, `defocus`, `camera_length`, `acceleration_voltage`, `beam_state`) must then return `nan` or `False`. Guard every such read.

## Saving data

Every acquisition helper ends with `save_acquisition` (`asyncroscopy/data/data_writer.py`). The public command reads `output_format` from the SCAN or CAMERA device and passes it on:

- `.h5` writes one HDF5 file for all detectors and returns its Tiled key.
- `.tiff` writes one file per detector and returns the shared stem.

Pick the `dataset_name` that fits: `"image"`, `"spectrum"` or `"stem_data"`. A multi-detector image is stored under `image/<DETECTOR>`. The `data` proxy comes from `self._detector_proxies.get("data")`.

The helpers that exist: `_acquire_scanned_image`, `_acquire_camera_image`, `_acquire_scanned_data_advanced`, `_acquire_spectrum`.

## Common changes

Add an attribute when clients need to read new device state. Guard the read for testing mode.

Change acquisition settings by reading more values from the support devices (dwell time, resolution, scan region, exposure). Pass them into the AutoScript `*Settings` object inside the helper.

`register_stage` copies the live stage position from AutoScript to the STAGE device.

Convert the tilts alpha and beta to radians only when you send them to AutoScript. The public API uses degrees.

4. **Adding a new detector**

Declare a device property for its address on the base. Add it to the dict in `_connect_detector_proxies`:

   ```python
   "newdet": self.newdet_device_address,
   ```

Then read it in the helper that uses it. See [adding a detector](../Adding_New_Hardware/add_detector.md).
