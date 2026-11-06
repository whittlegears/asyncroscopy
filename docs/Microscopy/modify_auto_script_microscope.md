# Modifying `AutoScriptMicroscope`

`AutoScriptMicroscope` (`asyncroscopy/instruments/electron_microscope/auto_script.py`) is the AutoScript vendor
subclass of [`ElectronMicroscope`](modify_base_electron_microscope.md). It owns the AutoScript
connection and implements the `_helper` methods the base declares abstract.

**Image helpers end via `_persist`; spectrum and STEM-data helpers via
`save_acquisition` directly.** `_persist` reads `scan.output_format` and
dispatches: `.h5` → `save_acquisition` (one HDF5 file, nested per detector),
`.tiff` → AutoScript `image.save()` (one file per detector). Either path
registers via the `data` proxy (`asyncroscopy/data/data.py`) and returns the
**Tiled key (`.h5`) or shared stem (`.tiff`)** the command sends to the client.
`save_acquisition` lives in `asyncroscopy/data/data_writer.py`; `data_server`
comes from `self._detector_proxies.get("data")`. See
[data_integration.md](../Tiled_server/data_integration.md).

If you're editing this class, you're usually doing one of these:

1. **Adding or modifying an attribute**
   Expose device state clients can read. The live-state reads
   (`fov`, `defocus`, `camera_length`, `acceleration_voltage`, `beam_state`)
   pull from `self._microscope` and must return `nan`/`False` when it is `None`
   (testing mode) — guard every such read. `register_stage` is the matching
   command: it pushes the live stage position onto the STAGE child device.
   The public stage vector is `[x, y, z, alpha, beta]`, with x/y/z in meters
   and alpha/beta in degrees. Convert tilts to radians only when sending values
   to AutoScript.

2. **Updating an attribute read/write method**
   (Control how a value is validated, stored, or synced with AutoScript.)

3. **Implementing or changing a `_helper`**
   Implement the base's abstract `_helper` (or override an optional one).
   Image helpers must finish via `_persist` (honors `output_format`); spectrum
   and STEM-data helpers via `save_acquisition`. Examples already present:
   `_acquire_scanned_image`, `_acquire_camera_image`,
   `_acquire_scanned_data_advanced`, `_acquire_spectrum`.

4. **Adding a new detector**
   Add a device property for its address on the base, add it to the base's
   `DETECTOR_ADDRESSES`, and read its proxy in the relevant helper.
   ```python
   DETECTOR_ADDRESSES = {..., "newdet": "newdet_device_address"}
   ```

6. **What the base already does for AutoScript**
   Limits are checked by the base before a helper runs; helpers send the value
   as given. The command lock is the base's `@hardware_call()`; AutoScript adds
   `_check_connected` (refuse with `NotConnected` when there is no client) and runs
   the server without Tango's own per-device lock (`serve`), so `blank_beam` never
   queues behind a long call. The beam current reported until the screen has
   measured it is the base's `default_beam_current_pa`. The stigmator is
   AutoScript's objective stigmator, in its optical units, converted with
   `objective_stigmator_m_per_unit`; until first light sets that, `get_stigmator`
   and `set_stigmator` are refused (and the Tuner treats the microscope as having
   no stigmator). `stage_moving` reads `specimen.stage.is_moving`.

5. **Adding or changing acquisition settings**
   Extend what is read from the detector devices — dwell time, resolution,
   scan region, exposure — and pass it into the AutoScript `*Settings` object
   inside the helper. Pick the right `dataset_name` for `save_acquisition`
   (e.g. `"image"`, `"spectrum"`, `"stem_data"`); multi-detector results are
   stored under `image/<DETECTOR>`.
