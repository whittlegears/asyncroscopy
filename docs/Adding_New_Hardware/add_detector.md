## Adding a new detector

Detectors are Tango devices in `asyncroscopy/instruments/electron_microscope/detectors/` (`camera.py`, `eds.py`, `eels.py`). To add one:

1. Copy `asyncroscopy/instruments/electron_microscope/detectors/camera.py` to `detectors/newdet.py` and adjust the attributes for that detector's settings.
2. Add a `device_property` to the microscope class that will use it (for example `auto_script.py` or `digital_twin_base.py`):
   ```python
   newdet_device_address = device_property(dtype=str, default_value="asyncroscopy/newdet/default")
   ```
   The launcher fills `<key>_device_address` automatically for every device in the config, so a config entry `newdet:` gives this property its value.
3. Register it in that class's `_connect_detector_proxies()`. See step 4 in [modify_auto_script_microscope.md](../Microscopy/modify_auto_script_microscope.md).
   ```python
   "newdet": self.newdet_device_address,
   ```
   The base class `ElectronMicroscope` (`asyncroscopy/instruments/electron_microscope/electron_microscope.py`) is not the right place for this.
4. Add acquisition logic:
   - see step 3 in [modify_base_electron_microscope](../Microscopy/modify_base_electron_microscope.md)
   - see step 5 in [modify_auto_script_microscope](../Microscopy/modify_auto_script_microscope.md)
5. Add the device to your config under `devices:`:
   ```yaml
   newdet:
     module_name: asyncroscopy.instruments.electron_microscope.detectors.newdet
   ```
6. Add `tests/test_NEWDET.py`, using `tests/test_CAMERA.py` as a template.

For an instrument that is not an electron microscope, see [new_instrument.md](new_instrument.md).
