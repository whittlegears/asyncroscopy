# Modifying the base `ElectronMicroscope`

`ElectronMicroscope` (`asyncroscopy/instruments/electron_microscope/electron_microscope.py`) is the **vendor-agnostic** Tango
device. It owns the public `@command` API and the abstract `_helper` methods
each vendor subclass (e.g. `AutoScriptMicroscope`) must fill in.

**The pattern:** a public `@command` validates input and reads settings from the
detector `DeviceProxy` objects, then delegates to a vendor `_helper`. Acquisition
commands return a **Tiled key (a string)**; the client reads the data back from
the Tiled server with that key.

The base owns everything every backend must do the same way, so a vendor class
implements only `_helper` methods:

- **Limits.** `set_fov`, `set_defocus`, `set_stigmator` and the scans are checked
  against `min_fov`, `max_fov`, `max_defocus`, `max_stigmator` and a scan region
  inside the field (`check_fov`, `check_defocus`, `check_stigmator`, `check_scan`)
  before the helper is called. A helper receives a value already checked. The
  properties' defaults (`DEFAULT_MIN_FOV`, ...) are at the top of
  `electron_microscope.py`, and `argument_hints()` describes the same ranges to
  clients through the device's `action_hints` attribute.
- **What each command is.** Every public command is declared on the base with
  `asyncroscopy.utils.commands.command(risk=, role=, stop=, routine=)`: `blank_beam`
  is a low-risk stop, `set_fov` and `set_defocus` are low, the imaging, spectra,
  stage moves and beam placement are high and routine in auto mode, and the column
  valves, screen and diffraction shift are high and held. A backend inherits all of
  it and declares nothing; a new public command on the base needs its declaration
  (the MCP server lists any undeclared command at start-up).
- **Beam current.** The attributes `beam_current_pa`, `beam_current_source`
  (`measured` or `default`) and `beam_current_age_s`, and the same fields in
  `get_parameters` (`beam_current_fields()`), come from `_beam_current()`. Its
  default reports the `default_beam_current_pa` property as assumed; a backend
  that measures the current overrides it (AutoScript from the screen, the twin
  from its probe current).
- **Stigmator and `stage_moving`.** `set_stigmator`/`get_stigmator` (metres of
  A1) and the `stage_moving` attribute call `_set_stigmator`, `_get_stigmator`
  and `_stage_moving`. A backend without them inherits a clear
  `UnsupportedCommand` error (JEOL, the specialised twins).
- **Detector proxies.** `_connect_detector_proxies` (on `ScanningBeamInstrument`,
  `asyncroscopy/instruments/scanning_beam.py`, with the beam, field of view, image shift, stage,
  beam current and limits) builds a proxy for every address in `DETECTOR_ADDRESSES`; `_proxy_timeout_ms(name)` sets each one's
  timeout (AutoScript lengthens the stage's).
- **The command lock.** Every public command that reaches the instrument is
  decorated with `@hardware_call()`: it runs under the device's command lock, after
  `_check_connected` (AutoScript refuses with `NotConnected` when it has no
  client). `blank_beam` uses `@hardware_call(serialized=False)`, so it never waits
  for a running acquisition or move. Put `@hardware_call()` under `@command` on any
  new public command that calls a helper.
- **The default detector.** `acquire_scanned_image([])` uses the
  `default_scan_detector` property, reported as the `default_detector` attribute
  (the Tuner reads it for a plan without a detector).

If you're editing this class, you're usually doing one of these:

1. **Adding or modifying an attribute**
   (Expose new device state clients can read, e.g. `stem_mode`.)

2. **Updating an attribute read/write method**
   (Control how a value is validated, stored, or synced with the vendor API.)

3. **Adding or modifying a public command**
   Add a thin `@command` that validates input, reads any settings from a
   detector proxy, then calls a vendor `_helper`. Existing groups:
   - acquisition — `acquire_scanned_image`, `acquire_spectrum`,
     `acquire_camera_image`,
     `acquire_scanned_data_advanced`
   - beam / optics — `place_beam`, `place_beam_list`, `blank_beam`,
     `unblank_beam`, `set_defocus` / `get_defocus`, `set_image_shift`,
     `set_column_valves`
   - stage — `get_stage`, `move_stage`
   - imaging conditions — `set_fov` / `get_fov`, `set_screen_current` /
     `get_screen_current`, `auto_focus`

4. **Adding or changing a vendor `_helper` contract**
   The `_helper` methods are the vendor extension points. Add the public
   `@command` here, and the `_helper` it delegates to.
   - **Required** (declared `@abstractmethod`; every subclass must implement):
     `_connect`, `_connect_hardware`, `_acquire_scanned_image`, `_get_stage`,
     `_move_stage`, `_set_fov` / `_get_fov`, `_set_screen_current` /
     `_get_screen_current`, `_auto_focus`, `_set_image_shift`.
   - **Optional** (default no-op or `UnsupportedCommand`; override only if the
     vendor supports it): `_acquire_camera_image`, `_acquire_scanned_data_advanced`,
     `_place_beam`, `_blank_beam`, `_unblank_beam`, `_set_defocus` /
     `_get_defocus`, `_set_stigmator` / `_get_stigmator`, `_stage_moving`,
     `_beam_current`, `_check_connected`.
   - **Note:** `acquire_spectrum` delegates to `_acquire_spectrum`, which is
     *not* declared on the base — it's defined only in the vendor subclass.
     If you add a new vendor, you must provide `_acquire_spectrum` yourself.

   Stage positions use `[x, y, z, alpha, beta]`; x/y/z are meters and alpha/beta
   are degrees in the public Tango API. Vendor helpers should convert only at
   the hardware boundary if the vendor API expects another unit.

5. **Changing the return / transport convention**
   Acquisition commands return a Tiled key string; the actual save happens in
   the vendor helper via `save_acquisition` (`asyncroscopy/data/data_writer.py`)
   and registration via the DATA device (`asyncroscopy/data/data.py`). See
   [data_integration.md](../Tiled_server/data_integration.md). The legacy
   `get_image_data_cached` (returns `DevEncoded`) is the only remaining
   byte-over-Tango path.

6. **Improving robustness**
   (Connection failures, missing proxies, vendor-API errors, simulation
   fallback, or state transitions like `FAULT` / `ON` / `OFF`.)
