# Modifying the base `ElectronMicroscope`

`ElectronMicroscope` (`asyncroscopy/instruments/electron_microscope/electron_microscope.py`) is the **vendor-agnostic** Tango
device. It owns the public `@command` API and the abstract `_helper` methods
each vendor subclass (e.g. `AutoScriptMicroscope`) must fill in.

**The pattern:** a public `@command` validates input and reads settings from the
detector `DeviceProxy` objects, then delegates to a vendor `_helper`. Acquisition
commands return a **Tiled key (a string)**; the client reads the data back from
the Tiled server with that key.

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
   - **Required** (declared `@abstractmethod` — every subclass must implement):
     `_connect`, `_connect_hardware`, `_connect_detector_proxies`,
     `_acquire_spectrum`, `_acquire_scanned_image`, `_set_column_valves`,
     `_get_defocus`, `_get_stage`, `_move_stage`, `_get_image_shift`,
     `_get_beam_tilt` / `_set_beam_tilt`, `_get_diffraction_shift` /
     `_set_diffraction_shift`, `_get_parameters`, `_set_fov` / `_get_fov`,
     `_auto_focus`, `_set_image_shift`, `_set_screen`, `_set_screen_current`,
     `_calibrate_screen_current`, `_get_screen_current`.
   - **Optional** (default no-op or "unsupported" — override only if the vendor
     supports it): `_acquire_camera_image`, `_acquire_scanned_data_advanced`,
     `_place_beam`, `_blank_beam`, `_unblank_beam`, `_set_defocus`.
   - **Note:** `acquire_spectrum` delegates to `_acquire_spectrum`, declared
     `@abstractmethod` on the base (2026-08-10) so a vendor that forgets it
     fails with a clear message instead of `AttributeError`. Every current
     vendor subclass implements it; if you add a new one, you must too.

   Stage positions use `[x, y, z, alpha, beta]`; x/y/z are meters and alpha/beta
   are degrees in the public Tango API. Vendor helpers should convert only at
   the hardware boundary if the vendor API expects another unit.

5. **Changing the return / transport convention**
   Acquisition commands return a Tiled key string; the actual save happens in
   the vendor helper via `save_acquisition` (`asyncroscopy/data/data_writer.py`)
   and registration via the DATA device (`asyncroscopy/data/data.py`). See
   [data_integration.md](../Tiled_server/data_integration.md).
   `get_image_data_cached` (returns `DevEncoded`) is a byte-over-Tango path
   for clients without MCP: the four `acquire_*` commands remember each
   returned key (most recent `MAX_CACHED_IMAGE_KEYS`), and
   `get_image_data_cached(index)` reads that key back from the DATA/Tiled
   server — the same source `get_data_from_key` (MCP) reads from, via the
   shared `asyncroscopy/data/data_reader.py::describe_tiled_node`. Index 0 is
   the most recent acquisition (the cache list appends oldest-first, so the
   lookup indexes from the end). Before 2026-08-10 this read from an
   in-process `_cached_images` list that no caller ever populated, so it
   always failed; before 2026-08-25 index 0 wrongly returned the oldest
   cached key.

6. **Improving robustness**
   (Connection failures, missing proxies, vendor-API errors, simulation
   fallback, or state transitions like `FAULT` / `ON` / `OFF`.)
