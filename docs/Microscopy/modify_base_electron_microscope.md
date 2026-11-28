# Electron microscope family

`ElectronMicroscope` (`asyncroscopy/instruments/electron_microscope/electron_microscope.py`) is the vendor-agnostic family base. It holds the public `@command` API and the `_helper` methods that each vendor class fills in: `AutoScriptMicroscope`, `JeolMicroscope` and the digital twins.

A public command validates the input, reads settings from the support-device proxies, and calls a `_helper`. An acquisition command returns a Tiled key. The client reads the data from Tiled with that key.

## Public commands

| Group | Commands |
|---|---|
| Acquisition | `acquire_scanned_image`, `acquire_spectrum`, `acquire_camera_image`, `acquire_scanned_data_advanced` |
| Beam and optics | `place_beam`, `place_beam_list`, `blank_beam`, `unblank_beam`, `set_defocus` / `get_defocus`, `set_image_shift` / `get_image_shift`, `set_beam_tilt` / `get_beam_tilt`, `set_diffraction_shift` / `get_diffraction_shift`, `set_column_valves` |
| Stage | `get_stage`, `move_stage` |
| Imaging conditions | `set_fov` / `get_fov`, `set_screen`, `set_screen_current` / `get_screen_current`, `calibrate_screen_current`, `auto_focus`, `get_parameters` |

To add a command, add a thin `@command` here and the `_helper` it calls. Then add the `_helper` to each vendor class.

## Vendor helpers

Required. Each is an `@abstractmethod`, so every vendor class must implement it:

`_connect`, `_connect_hardware`, `_connect_detector_proxies`, `_acquire_scanned_image`, `_get_defocus`, `_set_screen`, `_set_screen_current`, `_get_screen_current`, `_calibrate_screen_current`, `_move_stage`, `_get_stage`, `_set_fov`, `_get_fov`, `_auto_focus`, `_set_image_shift`, `_get_image_shift`, `_get_beam_tilt`, `_set_beam_tilt`, `_get_diffraction_shift`, `_set_diffraction_shift`, `_get_parameters`.

Optional. The base gives a default. Override it only if the vendor supports it:

`_acquire_camera_image`, `_acquire_scanned_data_advanced`, `_place_beam`, `_blank_beam`, `_unblank_beam`, `_set_defocus`.

Not on the base: `acquire_spectrum` calls `_acquire_spectrum`, and `set_column_valves` calls `_set_column_valves`. The base does not declare either one. A new vendor must supply them.

## Units

The stage vector is `[x, y, z, alpha, beta]`. x, y and z are in meters. alpha and beta are in degrees. Convert only at the hardware boundary, if the vendor API needs other units.

## Data

The vendor helper saves the data with `save_acquisition` (`asyncroscopy/data/data_writer.py`) and registers it through the DATA device. See [data and Tiled](../Tiled_server/data_integration.md). `get_image_data_cached` is the only command left that sends bytes (`DevEncoded`) over Tango.

## Robustness

6. **Improving robustness**

Handle connection failures, missing proxies and vendor API errors. Set the state to `FAULT`, `ON` or `OFF` to match what is true.
