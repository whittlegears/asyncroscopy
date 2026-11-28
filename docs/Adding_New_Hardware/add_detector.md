
## Adding a new detector

This page is for the electron microscope family. For other instruments, see [adding your instrument](../new_instrument.md).

1. Copy `asyncroscopy/detectors/HAADF.py` to `asyncroscopy/detectors/NEWDET.py` and adjust the attributes for that detector's settings.

   The detector devices are now in `asyncroscopy/instruments/electron_microscope/detectors/`. Start from the closest one, for example `camera.py` or `eds.py`.

2. Add a `device_property` for its address to `ElectronMicroscope`:
   ```python
   newdet_device_address = device_property(dtype=str, default_value="asyncroscopy/newdet/default")
   ```
3. Add it to `_connect_detector_proxies()` in each vendor class, for example `AutoScriptMicroscope`. Do not add it to the base `ElectronMicroscope`, because the base does not open proxies.
   ```python
   "newdet": self.newdet_device_address,
   ```

   Add the device to the YAML config under `devices.newdet`. The runner then sets `newdet_device_address` on the instrument.

4. Add acquisition logic:

   Add or change the public command on the [base](../Microscopy/modify_base_electron_microscope.md). Then implement its helper in each [vendor class](../Microscopy/modify_auto_script_microscope.md).

5. Add `tests/detectors/test_NEWDET.py` following `test_HAADF.py` as a template.

   The tests are now flat in `tests/`. Use `tests/test_CAMERA.py` as the template.
