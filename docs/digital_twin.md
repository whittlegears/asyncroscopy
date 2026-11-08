# DigitalTwin

`DigitalTwin` is a simulated STEM that answers the same Tango commands as the real microscope. Use it to write and debug a workflow, to benchmark an agent, or to rehearse an experiment before running it on hardware. Everyone can run it: it needs no vendor software and no GPU.

It is not a physics simulator of the specimen. Its job is to give workflows and agents the same problems a real instrument gives them: a stage that doesn't land exactly where it was told, drift, a probe that blurs when out of focus, shot noise, scan distortion, real-looking EDS spectra and beam damage. Every effect is seeded, so the same seed and the same commands always give the same run.

## Quick start

```bash
uv sync --extra agent --extra ollama
# An ideal instrument (all effects off), small 6 nm sample:
uv run startup_scripts/run_servers.py --yaml configs/DigitalTwin.yaml
# A realistic instrument and a 2 um, 400-particle sample for mapping:
uv run startup_scripts/run_servers.py --yaml configs/digital_twin_mapping.yaml
```

## How it works

| Model | File | What it does |
| :-- | :-- | :-- |
| Random streams | `asyncroscopy/sim/rng.py` | One seeded stream per kind of event (stage move n, acquisition n, ...). Repeating an acquisition gets fresh noise; rerunning the same script reproduces the run. |
| Clock | `asyncroscopy/sim/clock.py` | Simulated time. Moves, frames and spectra take as long as they would on the instrument. `time_scale` sets how much real time passes (1 = real waiting, 0 = none). |
| Stage | `asyncroscopy/sim/stage_model.py`, run by the `DigitalTwinSTAGE` device | Keeps **commanded**, **true** and **reported** positions. True = commanded + bias + Gaussian error + backlash on reversal. Reported = commanded + small readback error. Adds settling drift after each move and thermal drift over time. |
| Sample | `twin/specimen.py`, `twin/sample_model.py` | A swappable specimen ([specimens.md](Operation/specimens.md)). The default `procedural` one is a table of ellipsoidal particles (element, centre, semi-axes, orientation) on a carbon support film; a 2 um, 400-particle sample is a few kilobytes. With `support_texture` the film's thickness varies at every length from `support_texture_scale_m` down to 0.5 nm, so tiles can be registered on the film itself at any pixel size (a frame sees only the waves at least two of its pixels long, so overlapping frames see the same film however their grids fall); with `subpixel_atoms` each atom is drawn at its sub-pixel position. `recorded` images an image file, `crystal` a polycrystalline film. |
| Scan | `twin/scan_distortion.py` | Where each pixel really lands: drift during the frame, line jitter, flyback compression, scan rotation and shear, field-of-view calibration error. |
| Rendering | `twin/render.py` | Pixel >= 0.2 nm: projected thickness x Z^1.7 per particle, Gaussian probe blur. Pixel < 0.2 nm: atoms built for the particles in view only, convolved with the pyTEMlib probe from the current aberrations. HAADF, ADF and BF have their own contrast. |
| Optics | `twin/optics.py` | Aberrations from the corrector twin (if connected) plus `set_defocus`. Defocus and astigmatism blur the image, so autofocus has a real objective. |
| Detector | `twin/detector.py` | Raw counts: electrons per pixel from beam current and dwell, Poisson noise, gain, offset, read noise. No per-tile normalisation, so overlapping tiles agree in brightness, unless `detector_gain_drift` / `detector_offset_drift` make the gain and offset wander slowly with the clock (as a warming amplifier does), so tiles taken minutes apart differ. |
| EDS | `twin/eds_model.py` | 2048-channel spectrum with an energy axis: Kramers continuum, characteristic lines from pyTEMlib's X-ray tables with energy-dependent resolution, Si escape and sum peaks, paralyzable dead time; reports real and live time. |
| Dose | `asyncroscopy/sim/dose.py` | Where the beam has been, in cells kept only where it went and sized to each frame (about an eighth of its field), so a 50 nm frame on a large sample keeps its dose on the 50 nm it hit. Builds a carbon contamination layer that saturates, and fades particles past a critical dose. |

Coordinates are metres in the sample frame. Images are indexed `[row, column]`; rows run along +y and columns along +x. The stage position is the sample point under the centre of the field of view.

### What it does not model

Before you trust a workflow that passes on the twin, know where the twin is simpler than the instrument:

- **Stage z, alpha and beta don't change the image.** They are tracked, recorded in ground truth and checked against the STAGE limits, but focus responds only to `set_defocus` and the specimen height gradient (`height_gradient_x/y`). A z-based autofocus will see a flat focus metric on the twin.
- **The support film never ends.** Far from the particles the twin shows clean carbon. There are no grid bars, holes or sample edges, so nothing tells an agent it has left the sample.
- **Particles are analytic ellipsoids**, not real specimen structure. Contrast is a mass-thickness model, not multislice.
- **Inputs are limited to plausible values.** `set_fov` outside `min_fov`..`max_fov` (1 nm to 100 um), `|defocus|` above `max_defocus` (10 um), a stigmator component above `max_stigmator` (1 um), non-finite numbers and stage moves outside the STAGE limits are refused with an error. The `ElectronMicroscope` base checks these for every backend, so AutoScript refuses the same values. Your instrument's limits may differ, so set these properties to match it.
- **`scan_region` is `[left, top, width, height]`** in fractions of the field, as on the SCAN device and in AutoScript.

Rendering time on a laptop (GTX-class CPU, no GPU needed): about 30 ms for a 512x512 survey tile, about 0.7 s for the first 512x512 atomic-resolution tile of a particle and 0.1 s after that. `tests/test_twin_render_perf.py` checks this.

## The stage device

Every twin config runs `DigitalTwinSTAGE` (`hardware/stage_digital_twin.py`) as its STAGE device, and it owns the stage's behaviour: commanded, true and reported positions, bias, landing error, backlash, settling, thermal drift, `stage_moving` and the travel limits. The twin microscope orchestrates it through its DeviceProxy, as it does the corrector twin: `move_on_clock` moves the stage at the microscope's simulated time and says how long the move takes, and the microscope advances its clock (the single time source) by that; `frame_trajectory`, called once per frame, returns the true position and a compact description of the drift, so rendering each line from the true position does not cost a Tango call per line. A position written directly to `position` or an axis counts as a stage command; the microscope picks it up on its next `frame_trajectory`. `position_sigma_m` is worked out from the stage model, so a Mapper without its own `stage_sigma_m` reads it. A `DigitalTwin` with no STAGE device images at the origin and refuses `move_stage`.

## Ground truth

`export_ground_truth()` returns JSON with the particle table (with each particle's lattice orientation, `euler_deg`), every stage move (commanded, true, reported), and the true pose and drift of every acquisition, with each frame's scan distortion (`scan`: rotation, shear, field-of-view error, flyback, and `line_jitter`, the line count and spread from which `twin.scan_distortion.line_jitter_m` gives every line's x offset again) and the scanning detectors' gain and offset when it started (`detector`). Use it to score a workflow, for example how far a stitched map is from where the tiles really were.

Agents must not see it: `configs/mcp_dt.yaml` blocks it. Call it from test scripts through Tango.

## Properties

All new effects default to off, so `configs/DigitalTwin.yaml` behaves like an ideal instrument. `configs/digital_twin_mapping.yaml` turns them on.

| Group | Properties |
| :-- | :-- |
| Specimen | `specimen` (a mapping such as `{source: recorded, path: ..., pixel_size_m: ...}`), or `specimen_source`, `specimen_path`, `specimen_pixel_size_m`, `specimen_labels_path`; see [specimens.md](Operation/specimens.md) |
| Sample | `sample_seed`, `sample_size_xy`, `sample_particle_count`, `particle_radius_mean`, `particle_radius_std`, `particle_aspect_ratio`, `particle_elements` (comma list of Au, Pt, Pd, Ag, Cu, Ni, Fe, Si, C), `support_thickness`, `height_gradient_x/y` |
| Optics | `acceleration_voltage`, `convergence_angle_mrad`, `probe_current_pa` (reported as the measured `beam_current_pa`), `haadf_z_exponent`, `haadf_scattering_scale`, `atomic_pixel_threshold` |
| Detector | `detector_gain`, `detector_offset`, `detector_read_noise`, `detector_gain_drift` (fraction), `detector_offset_drift` (counts) |
| Scan | `scan_flyback_time`, `scan_line_jitter_px`, `scan_rotation_deg`, `scan_shear`, `scan_flyback_compression`, `fov_scale_error` |
| Stage (properties of the `DigitalTwinSTAGE` device, under `devices: stage:`) | `stage_move_noise_std`, `stage_bias_x/y`, `stage_reported_noise_std`, `stage_backlash_x/y`, `stage_speed`, `stage_move_overhead`, `stage_settle_drift`, `stage_settle_tau`, `drift_rate_per_min`, `drift_random_walk`, and the travel limits `x_limits_m`, `y_limits_m`, `z_limits_m` |
| Time | `time_scale` |
| EDS | `eds_fwhm_mn_ka`, `eds_solid_angle`, `eds_dead_time`, `eds_channels`, `eds_energy_max` |
| Dose | `dose_tracking`, `contamination_max`, `contamination_dose`, `damage_critical_dose` |

Where the mapping config's values come from is noted in the file. `digital_twin_crystal.yaml`, `digital_twin_recorded.yaml` and `digital_twin_large_map.yaml` start from it with `extends: digital_twin_mapping.yaml` and set only what differs; `null` removes a property or device the base sets. Values marked unverified are placeholders: run the scripts in `asyncroscopy/calibration/` on your instrument and paste the YAML they print.

Each acquisition's record is appended to `<acquisition_save_directory>/ground_truth/<device>_acquisitions.jsonl` as it is taken, so a long session holds none of them in memory; `export_ground_truth` reads the file back.

## Key commands

- `move_stage([x, y, z, alpha, beta])` (x/y/z metres, alpha/beta degrees), `get_stage()` (reported position)
- `set_fov(fov)`, `get_fov()`, `set_defocus(defocus)`, `get_defocus()`, `set_stigmator([a, b])`, `get_stigmator()` (metres of A1)
- `acquire_scanned_image(["haadf", "bf"])`: one scan, one image per detector; honours the SCAN device's `scan_region`
- `place_beam([x, y])` (fractions of the field of view), `acquire_spectrum("eds")`
- `get_parameters()`: status JSON including simulated time; the attributes `beam_current_pa`, `beam_current_source` and `stage_moving` say the same for a client that reads attributes
- `export_ground_truth()`: see above

## Specialised twins

`ElectronMicroscopeTwin` (`digital_twin_base.py`) is the thin shell every twin shares: its properties, the detector proxies, the beam position, field of view and defocus, the stage basics, and `_generate_sample(seed)`, where a twin loads its sample. Every twin's sample is a `Specimen` plug-in chosen by `specimen: {source: ...}` ([specimens.md](Operation/specimens.md)); a twin class differs only in its render mode. The tilt, diffraction and gold twins also share `ParticleSpectrumMixin` (`twin/atomistic.py`): particle records that follow the stage, and the EDS spectrum they give under the parked beam. The specialised twins:

- `DigitalTwinDiffraction`: parked-beam nanoparticle diffraction with abTEM (`uv sync --extra diffraction`). Fixes the field of view at 500 nm and saves a simulated diffraction pattern from `acquire_camera_image()`.
- `DigitalTwinTilt`: `silicon_lamella` tilt series with ASE/abTEM multislice.
- `DigitalTwinParticles`: `particles`, voxel-backed oriented particles.
- `PolycrystallineGoldDigitalTwin`: `polycrystal` grains, driven by the corrector twin.

To build a twin of another electron microscope, start from `DigitalTwin` if it images a sample by scanning, or from `ElectronMicroscopeTwin` if you want the shared plumbing and your own renderer (implement `_generate_sample`). Every command, and what it is (risk, role, stop, routine), comes from the `ElectronMicroscope` base: a twin declares nothing. For a different kind of instrument, start from the nanoindenter in [new_instrument.md](Adding_New_Hardware/new_instrument.md) (a `NanoIndenter` base with `SimIndenter` as its simulated backend) and reuse the modality-free models in `asyncroscopy/sim/`: `stage_model.py` (five axes `[x, y, z, alpha, beta]`; pass zeros for the axes you do not have), `clock.py`, `rng.py` and `dose.py`.
