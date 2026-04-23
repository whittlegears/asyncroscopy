
## Adding a new detector

1. Copy `asyncroscopy/detectors/HAADF.py` to `asyncroscopy/detectors/NEWDET.py` and adjust the attributes for that detector's settings.
2. Add a `device_property` in Microscope.py:
   ```python
   newdet_device_address = device_property(dtype=str, default_value="test/detector/newdet")
   ```
3. Register it in `_connect_detector_proxies()` - see step 4 in  [modify_thermo_microscope.md](../modify_thermo_microscope.md)
   ```python
   "newdet": self.newdet_device_address,
   ```
- note : base class `Microscope` at asyncroscopy/Microscope.py is not the right place for this:

4. Add acquisition logic:
- see step 3 in [modify_base_microscope](../modify_base_microscope.md) 
- see step 5 in [modify_thermo_microscope](../modify_thermo_microscope.md)

5. Add `tests/detectors/test_NEWDET.py` following `test_HAADF.py` as a template.
