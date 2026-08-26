# Digital twin realism layer

The `DigitalTwin` device separates the **commanded** instrument state (what the
operator asked for) from the **observed** state (what reads and acquisitions
report). A configurable environment model sits between the two, so the twin can
behave like the physical Spectra 300 instead of an ideal simulator: moves land
*near* the target, positions drift over time, and acquisitions only produce
signal when electrons actually reach the sample.

Everything is driven by Tango device properties, set per deployment under
`instrument.properties` in the config YAML (see `configs/DigitalTwin.yaml`).
Setting a value to `0` (or `false`) disables that effect, which is what the
test fixtures and any subclass that doesn't opt in get by default.

## Stage environment model

| Property | Meaning |
| --- | --- |
| `stage_move_noise_std` | Gaussian landing error (m) applied to x/y/z on every move. |
| `stage_backlash` | Undershoot (m) applied to an axis that reverses travel direction. |
| `stage_settle_amplitude` / `stage_settle_time_s` | Post-move relaxation: right after a move the stage sits `amplitude` short of its landing point along the travel direction, decaying exponentially with the given time constant. Reading the position (or acquiring) immediately after a move therefore differs from reading it a few seconds later, like on the real column. |
| `stage_drift_rate` | Continuous thermal drift speed (m/s) along a fixed, seed-derived direction (mostly lateral). Applied to both `get_stage` reads and image rendering, so features crawl across repeated acquisitions. |
| `stage_limit_xy` / `stage_limit_z` | Hard travel limits (m); out-of-range moves raise, mirroring the real stage. Alpha/beta limits are enforced by the `STAGE` device attributes (±35° / ±20°). |

The commanded pose lives in `_commanded_stage` (and in the stage proxy); every
read recomputes `_stage_position = commanded + drift + settling`, and the
sample viewport is rendered from that observed pose.

**Scale note:** the default sample world is only ~6 nm across with a 20 nm
FOV, so the shipped values in `DigitalTwin.yaml` are 10–100× smaller than real
stage errors. If you grow `sample_size_xy`, scale the stage errors up with it.

## Beam-state gating (`enforce_beam_state`)

When `enforce_beam_state` is true, `acquire_scanned_image`,
`acquire_camera_image`, and `acquire_spectrum` return detector noise only
(dark frames / noise-floor spectra) unless the column valves are open **and**
the beam is unblanked — so an agent driving the twin has to run the same
bring-up sequence as on the real instrument (`set_column_valves("open")`,
`unblank_beam`). `get_parameters` reports `column_valves_state` and
`beam_blanked` so the state is discoverable.

## Optics coupling

- `set_defocus` now feeds into the simulated probe (`C10`, in nm), so defocus
  visibly blurs the image and `auto_focus` visibly restores it.
- `set_image_shift` translates the rendered viewport without moving the
  mechanical stage, matching how image shift is used on the real instrument at
  small FOVs.

## Tool parity with the real instrument

The twin exposes the same Tango surface as `AutoScriptMicroscope`
(`configs/Spectra300.yaml`), so notebooks, GUIs, and MCP agents written
against the real instrument run on the twin unchanged:

- **Read attributes**: `fov`, `defocus`, `camera_length`, `beam_state`,
  `acceleration_voltage` (the twin's simulated 200 kV also feeds the probe
  renderer).
- **`acquire_scanned_data_advanced`**: previously threw `UnsupportedCommand`
  on the twin even though the MCP tool existed; it now renders the scan
  signal and saves it under the `stem_data` acquisition type like the real
  command. Per-pixel 4D diffraction patterns are not simulated.
- **`register_stage`**: publishes the commanded pose onto the STAGE child
  device, matching the real microscope's notebook workflow.
- **Aperture device**: `TestAperture`
  (`hardware/TestAperture.py`) simulates the Spectra 300's motorized
  mechanisms (C1/C2/C3/OBJ/SA) with named apertures, insert/retract,
  enable/disable, and per-mechanism positions. It is wired into
  `configs/DigitalTwin.yaml` the same way `AutoScriptAPERTURE` is wired into
  the Spectra config, so the MCP server exposes the same aperture tools.
  `get_parameters` reports each mechanism as `Disabled`/`Retracted`/aperture
  name, exactly like the real `_get_parameters`.

Still real-instrument-only: the CEOS `corrector` device (a JSON-RPC bridge to
the corrector PC — a simulated backend would need a fake CEOS server) and the
EELS detector from `Spectra300eels.yaml`.

## Adding more environment effects

The pattern for new effects is:

1. Add a `device_property` with a **zero/off default** so tests and existing
   deployments are unaffected.
2. Apply the effect in the observed-state path (`_stage_environment_offset`,
   `_sync_stage_from_proxy`, `_render_stem_image`, `_simulate_spectrum`), never
   to the commanded state.
3. Turn it on with a realistic value in the deployment YAML.
4. Cover it in `tests/test_digital_twin_realism.py` against the
   `realistic_twin_proxy` fixture; the ideal `twin_proxy` stays deterministic.

Good candidates for future effects: scan distortion / probe jitter during long
dwell times, contamination growth under the beam, tilt-dependent focus change,
sample charging, energy-dependent EDS peak broadening, and a finite stage move
duration (device `MOVING` state while a move is in flight).
