# Digital twin vs real microscope, and AutoScript coverage (audit, 2026-11-27)

This was a read-only audit of branch `claude/audit-hardcoded-values-3wt5b7`. The AutoScript API surface was listed
from the client wheels in `stubs/AutoScript_v_1.15`, `_1.17` and `_1.18`. Nothing below is fixed yet; each item is a
proposal.

## How tools arise
- Only Tango **commands** become MCP tools, named `{prefix}_{command}` (`mcp/mcp_server.py` `_find_tools`).
  Attributes never do; `action_space.py` lists them under "state".
- An agent is kept away from a command in only three ways:
  - `operator_only=True` on the command;
  - Tango's own Init, Kill and RestartServer (`safety/policy.py`);
  - the per-config `blocked_functions`.
  `action_roles` and `descriptors` set role and risk; they hide nothing.
- `configs/mcp_dt.yaml` blocks less than `configs/mcp.yaml`. On hardware, corrector writes (`correct_aberration`,
  `measure_c1a1`, `acquire_tableau`, `set_aberrations_coeff_sim`) and `Disconnect` are blocked. On the twin the same
  tools are open, so an agent rehearsing on the twin uses tools it will not have on hardware.

## Twin-only items
| Item | Agent sees it | Proposal |
| :-- | :-- | :-- |
| `export_ground_truth` | no (operator only) | keep: it is a scoring hook |
| DigitalTwinSTAGE `move_on_clock`, `frame_trajectory`, `stage_moves` | no (operator only) | keep; optionally block on the operator endpoint too |
| `beam_pos` attribute, read-write and declared twice | writable over Tango, bypassing `place_beam`'s checks | make it read-only; drop the duplicate in DigitalTwinParticles |
| PolycrystallineGold `get_volume_metadata`, `get_region_parameters` | **yes** | remove: no callers, and they leak ground truth |
| CORRECTOR `get_aberrations_coeff_sim` / `set_aberrations_coeff_sim` (on the shared base) | **yes**, on both endpoints | move to DigitalTwinCorrector and make operator only |
| `sample_seed`, `particle_count`, DigitalTwinTilt specimen truth in `get_parameters` | yes | strip them (only DigitalTwin already pops the seed) |
| DigitalTwinCorrector `get_info` reporting `"manufacturer": "CEOS"` | yes | fix the label |

## Dead or duplicate commands on both twin and hardware
- `get_image_data_cached`: `_cached_images` is never assigned, so it always raises.
- `place_beam_list`: no callers.
- `register_stage`: duplicates `get_stage`.
- AutoScript attributes `fov`, `defocus` and `beam_state` duplicate commands. `manufacturer` is dtype bool and
  raises when read.

## Behaviour that differs between twin and hardware
1. **Stop commands that do nothing (safety-critical).** The base `_blank_beam`, `_unblank_beam`, `_place_beam` and
   `_set_defocus` are `pass` (`scanning_beam.py:431-438`, `electron_microscope.py:343`). On JeolMicroscope (real
   hardware, `configs/covalent-F200.yaml`) and the atomistic twins (DigitalTwinParticles, DigitalTwinTilt,
   DigitalTwinDiffraction, PolycrystallineGold), `blank_beam` and the e-stop report success without blanking.
   They should raise UnsupportedCommand.
2. **Diffraction shift.** AutoScript uses `optics.deflectors.image_shift` (metres). The 1.18 stub documents
   `deflectors.image_tilt` (radians) as the diffraction shift. Outside diffraction mode `get_diffraction_shift`
   returns a made-up `[0, 0]`.
3. **Beam tilt units.** The base says radians, AutoScript is unitless −1..1, and DigitalTwinTilt uses radians.
4. **Bad input accepted silently.** On AutoScript, a bad `set_column_valves` state and an unknown `set_screen`
   value succeed silently; the twin raises.
5. **Camera images.** On DigitalTwin, DigitalTwinParticles and PolycrystallineGold, `acquire_camera_image` returns
   a HAADF scan, using the exposure as dwell. The base also misbinds `output_format` into `electron_counting`.
6. **`auto_focus`.** On AutoScript in TEM mode it only stigmates; in STEM mode it discards `RunOptiStemResults.quality`.
7. **Detector names.** The twin has HAADF, ADF and BF; AutoScript has BF, DF2, DF4, HAADF, BF-S and DF-S. "ADF"
   works only on the twin.
8. **`get_parameters` schema.** DigitalTwinTilt uses a different schema, and the parity test covers only DigitalTwin.
9. **Screen current.** DigitalTwinTilt `get_screen_current` echoes the stored value, and calibrate does nothing.
10. **Stage position length.** AutoScriptSTAGE accepts 4 axes on single-tilt holders; the twin requires 5.
11. **Stigmation.** AutoScript refuses until `objective_stigmator_m_per_unit` is set; the twin does not.
12. **No twin devices.** APERTURE and EELS have no twin, so they cannot be rehearsed.
13. **`set_column_valves`** raises AttributeError on the atomistic twins.

## AutoScript (1.18) surface not yet exposed, in priority order
1. **Stage axis limits** (read): `specimen.stage.get_axis_limits(StageAxis.*)`, replacing the placeholder limits.
2. **Detector lists** (read): `detectors.scanning_detectors`, `camera_detectors`, `eds_detectors`, `eels_detectors`;
   per camera `exposure_time_limits` and `get_supported_image_sizes()`. These would replace the hard-coded detector
   lists.
3. **Optics readout** (read): `camera_length.value` and `available_values`, `magnification.value`,
   `convergence_angle`, `beam_diameter`.
4. **Versions** (read): `service.autoscript.server.version`, `client.version`, `server.is_offline`,
   `service.system.serial_number`/`version`.
5. **Auto-blank state** (read): `optics.blanker.auto_beam_blank.mode.value`.
6. **Stops:** `specimen.stage.stop_jogging()`; check `service.autoscript.client.is_stop_requested` in long loops.
7. **Focus routines:** `auto_functions.run_beam_tilt_auto_focus` (TEM) and `run_stem_auto_focus` (STEM, returns
   defocus and astigmatism); return `run_opti_stem(...).quality`.
8. **Imaging set-up** (high risk): projector mode, `camera_length.value`, `reset_defocus`, beam stopper, scanning
   detector insert/retract.
9. **Acquisition:**
   - `relative_move_safe`;
   - `acquire_camera_series_advanced`;
   - `analysis.eels.acquire_spectra` and `acquire_spectrum_image`;
   - an EDS spectrum image over a region.
10. **Operator only or never for agents:**
    - dose setters: spot size, convergence angle, intensity;
    - FEG: flash, extractor voltage, gun lens;
    - gun deflectors;
    - raw deflectors and stigmators;
    - alignment writes;
    - corrector auto-tuning;
    - simulator and internal services.

## Deprecated or removed AutoScript members still in use
- `optics.is_accelerator_on` and `optics.acceleration_voltage` (`auto_script.py:698,778-779`) are removed in 1.21;
  use `source.*`.
- The `hasattr(optics, "defocus")` branch fires only on ≤1.17; 1.18 removed `optics.defocus`.
- 1.18 adds `client.authenticate`. If the facility turns on key-based access, `connect` will need it.

## Test gaps
`tests/fake_autoscript.py` mirrors only the subset in use:
- `aperture_mechanisms.get_available()` returns `[]`, so AutoScriptAPERTURE is untested.
- It has no `camera_length`, `convergence_angle`, `detectors.*` or `image_tilt`.
