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
   (`camera_length`, `acceleration_voltage`) pull from `self._microscope` and
   must return `nan` when it is `None` (testing mode) — guard every such read.
   An action never passes silently without a client: reach it through `self._client("<command>")`,
   which raises NotConnected (so `blank_beam`, and with it `stop_all`, report the beam not blanked in testing mode).
   Field of view, defocus and the blanker are also read-only attributes (`fov`, `defocus`, `beam_state`), declared on
   the `ElectronMicroscope` base for Tango polling, events and monitors such as Taurus or Jive. Each reads through the
   reader its command uses (`_get_fov`, `_get_defocus`, `_beam_blanked`, which `get_parameters`' `beam_blanked` reads
   too), so there is no second implementation, and a read without a client raises NotConnected. `manufacturer` is a
   string, "Thermo Fisher Scientific": AutoScript TEM drives only that maker's microscopes, and the 1.18 client has
   no manufacturer member outside an image's metadata. The stage is read with `get_stage`; there is no
   `register_stage` (the AutoScript STAGE device reads the hardware itself).
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

## What the device reads from AutoScript

The client is the 1.18 wheel that `pyproject.toml` installs, so the code uses 1.18 members only: the high
tension comes from `source.acceleration_voltage` and `source.is_accelerator_on` (the `optics.*` ones are
deprecated and go in 1.21), and defocus from `optics.focusing.stem.objective.defocus` or
`optics.focusing.tem.defocus` (1.18 has no `optics.defocus`).

- `get_parameters` adds, read from the instrument each time: `camera_length_m` (calibrated, nominal and label of
  `optics.camera_length.value`) and `camera_lengths` (the labels of `available_values`), `magnification`,
  `convergence_angle_rad`, `beam_diameter_m`, `auto_beam_blank_mode` (`optics.blanker.auto_beam_blank.mode.value`),
  and `versions` (`service.autoscript.server.version`, `server.is_offline`, `client.version`,
  `service.system.serial_number` and `version`). `versions_at_connect` is what Connect read; it is also logged.
  A reading this instrument cannot give (the convergence angle and beam diameter need three condenser lenses) is
  `{"unavailable": the error}`, never a made-up value.
- `get_detectors` returns `detectors.scanning_detectors`, `camera_detectors`, `eds_detectors` and `eels_detectors`,
  and for each camera `exposure_time_limits` (AutoScript gives them for the Flucam only) and
  `get_supported_image_sizes()`. A scanned image with a detector the microscope does not list (such as the twin's
  "ADF") is refused before anything is acquired. The twins answer `get_detectors` with what they render.
- The AutoScript STAGE device reads `specimen.stage.get_axis_limits(StageAxis.X/Y/Z/A, and B on a double-tilt
  holder)` when it connects. They narrow the configured `x_limits_m`, `y_limits_m`, `z_limits_m` and tilt limits,
  so a move is checked against the instrument's own limits as well; `get_holder` shows them as `hardware_limits`.
  When the two do not overlap on an axis, every move is refused until the configuration is fixed.

## Operator-only actions waiting for review

These commands were added from the AutoScript 1.18 stub (settings classes and enumeration values exactly as the
stub names them). Each is declared `operator_only=True`, so only the operator endpoint offers it and no agent is
given it, until a person has reviewed it on the instrument and decides to lift that. All are risk `high`. On the
twins and JEOL each one raises UnsupportedCommand; no twin pretends to run them.

| Command | Device | AutoScript call | Role |
| :-- | :-- | :-- | :-- |
| `run_stem_auto_focus(request)` | microscope | `auto_functions.run_stem_auto_focus(RunStemAutoFocusSettings)`; STEM only; returns `defocus_m`, `astigmatism_m` and `beam_blanked` (the blanker read after the routine) | `optics.focus` |
| `run_beam_tilt_auto_focus(request)` | microscope | `auto_functions.run_beam_tilt_auto_focus(RunBeamTiltAutoFocusSettings)`; TEM only; returns `defocus_m`, `astigmatism_m`, `drift_m`, `beam_blanked` | `optics.focus` |
| `acquire_camera_series(length)` | microscope | `acquisition.acquire_camera_series_advanced(CameraAcquisitionSettings, length)`, with the CAMERA device's settings; at most `max_camera_series_frames` frames and `max_camera_series_bytes` bytes (frames x size² x `camera_bytes_per_pixel`, which must be set from the camera: the API does not report it) | `acquire.image` |
| `acquire_eels_spectra(request)` | microscope | `analysis.eels.acquire_spectra(EelsAcquisitionSettings)`; the detector must be in `eels_detectors` | `acquire.spectrum` |
| `acquire_eels_spectrum_image(request)` | microscope | `analysis.eels.acquire_spectrum_image(EelsSpectrumImageSettings)`; saved as the vendor file, like the EDS one | `acquire.other` |
| `set_projector_mode(mode)` | microscope | `optics.projector_mode` = `ProjectorMode.IMAGING` or `DIFFRACTION` | `setting.set` |
| `set_camera_length(label)` | microscope | `optics.camera_length.value` = the `available_values` entry with that label | `setting.set` |
| `set_beam_stopper(state)` | microscope | `optics.beam_stopper.insert()`, `half_insert()` or `retract()` | `setting.set` |
| `set_scanning_detector([detector, state])` | microscope | `detectors.get_scanning_detector(name).insert()` or `retract()` | `setting.set` |
| `move_relative(delta)` | STAGE | refused while `specimen.stage.is_moving`; the target (here + delta) is checked against the stage and holder limits and sent as the same absolute move `move_stage` makes (`absolute_move_safe` when z or a tilt changes, else `absolute_move`) | `stage.move` |

The focus routines take only the settings fields the stub lists for them (others are refused);
`reset_beam_blanker_afterwards` is not offered, so AutoScript's default, a blanked beam afterwards, holds.

Two changes are not operator only:

- `stop_jogging()` on the AutoScript STAGE device calls `specimen.stage.stop_jogging()`. It is a stop (risk low,
  role `safe_stop`), so agents keep it and `stop_all` calls it. It stops jogging only; a move under way runs to its
  end, as `stop_all`'s `not_stopped` says.
- `auto_focus` now returns JSON: on AutoScript in STEM, `{"routine": "run_opti_stem C1A1", "quality": ...}` from
  `RunOptiStemResults.quality` (Bad, Improved or Good); `{}` where the routine reports nothing (TEM, the twins).
