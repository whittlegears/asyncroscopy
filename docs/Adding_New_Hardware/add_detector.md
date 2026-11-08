## Adding a new detector

Detectors are Tango devices in `asyncroscopy/instruments/electron_microscope/detectors/` (`camera.py`, `eds.py`, `eels.py`). To add one:

1. Copy `asyncroscopy/instruments/electron_microscope/detectors/camera.py` to `detectors/newdet.py` and adjust the attributes for that detector's settings. Rename the class and the `run_server()` call at the bottom.
2. Add a `device_property` for its address to the `ElectronMicroscope` base (`asyncroscopy/instruments/electron_microscope/electron_microscope.py`):
   ```python
   newdet_device_address = device_property(dtype=str, doc="Tango device address for the NEWDET settings device.")
   ```
   The launcher fills `<key>_device_address` automatically for every device in the config, so a config entry `newdet:` gives this property its value.
3. Add it to the modality's `DETECTOR_ADDRESSES` (`ScanningBeamInstrument.proxy_addresses` is what connects them), which `_connect_detector_proxies()` connects for every backend. See step 4 in [modify_auto_script_microscope.md](../Microscopy/modify_auto_script_microscope.md).
   ```python
   DETECTOR_ADDRESSES = {..., "newdet": "newdet_device_address"}
   ```
   A spectrum detector also goes in `SPECTRUM_DETECTORS` (same file): `acquire_spectrum` looks only there.
4. Add acquisition logic:
   - see step 3 in [modify_base_electron_microscope](../Microscopy/modify_base_electron_microscope.md)
   - see step 5 in [modify_auto_script_microscope](../Microscopy/modify_auto_script_microscope.md)
   - implement or check the helper on each backend you run, including the twin (`digital_twin.py`) and JEOL (`jeol.py`). The twin's `_acquire_spectrum` only models EDS.
5. Add the device to your config under `devices:`:
   ```yaml
   newdet:
     module_name: asyncroscopy.instruments.electron_microscope.detectors.newdet
   ```
6. Add `tests/test_NEWDET.py`, using `tests/test_CAMERA.py` as a template. Add the device to `devices_info` in `tests/conftest.py`, and add a `newdet_proxy` fixture.

For an instrument that is not an electron microscope, see [new_instrument.md](new_instrument.md).
