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
| Random streams | `twin/rng.py` | One seeded stream per kind of event (stage move n, acquisition n, ...). Repeating an acquisition gets fresh noise; rerunning the same script reproduces the run. |
| Clock | `twin/clock.py` | Simulated time. Moves, frames and spectra take as long as they would on the instrument. `time_scale` sets how much real time passes (1 = real waiting, 0 = none). |
| Stage | `twin/stage_model.py` | Keeps **commanded**, **true** and **reported** positions. True = commanded + bias + Gaussian error + backlash on reversal. Reported = commanded + small readback error. Adds settling drift after each move and thermal drift over time. |
| Sample | `twin/specimen.py`, `twin/sample_model.py` | A swappable specimen ([specimens.md](Operation/specimens.md)). The default `procedural` one is a table of ellipsoidal particles (element, centre, semi-axes, orientation) on a carbon support film; a 2 um, 400-particle sample is a few kilobytes. With `support_texture` the film's thickness varies at every length from `support_texture_scale_m` down to 0.5 nm, so tiles can be registered on the film itself at any pixel size (a frame sees only the waves at least two of its pixels long, so overlapping frames see the same film however their grids fall); with `subpixel_atoms` each atom is drawn at its sub-pixel position. `recorded` images an image file, `crystal` a polycrystalline film. |
| Scan | `twin/scan_distortion.py` | Where each pixel really lands: drift during the frame, line jitter, flyback compression, scan rotation and shear, field-of-view calibration error. |
| Rendering | `twin/render.py` | Pixel >= 0.2 nm: projected thickness x Z^1.7 per particle, Gaussian probe blur. Pixel < 0.2 nm: atoms built for the particles in view only, convolved with the pyTEMlib probe from the current aberrations. HAADF, ADF and BF have their own contrast. |
| Optics | `twin/optics.py` | Aberrations from the corrector twin (if connected) plus `set_defocus`. Defocus and astigmatism blur the image, so autofocus has a real objective. |
| Detector | `twin/detector.py` | Raw counts: electrons per pixel from beam current and dwell, Poisson noise, gain, offset, read noise. No per-tile normalisation, so overlapping tiles agree in brightness, unless `detector_gain_drift` / `detector_offset_drift` make the gain and offset wander slowly with the clock (as a warming amplifier does), so tiles taken minutes apart differ. |
| EDS | `twin/eds_model.py` | 2048-channel spectrum with an energy axis: Kramers continuum, characteristic lines from pyTEMlib's X-ray tables with energy-dependent resolution, Si escape and sum peaks, paralyzable dead time; reports real and live time. |
| Dose | `twin/dose.py` | Where the beam has been, in cells kept only where it went and sized to each frame (about an eighth of its field), so a 50 nm frame on a large sample keeps its dose on the 50 nm it hit. Builds a carbon contamination layer that saturates, and fades particles past a critical dose. |

Coordinates are metres in the sample frame. Images are indexed `[row, column]`; rows run along +y and columns along +x. The stage position is the sample point under the centre of the field of view.

### What it does not model

Before you trust a workflow that passes on the twin, know where the twin is simpler than the instrument:

- **Stage z, alpha and beta don't change the image.** They are tracked, recorded in ground truth and checked against the STAGE limits, but focus responds only to `set_defocus` and the specimen height gradient (`height_gradient_x/y`). A z-based autofocus will see a flat focus metric on the twin.
- **The support film never ends.** Far from the particles the twin shows clean carbon. There are no grid bars, holes or sample edges, so nothing tells an agent it has left the sample.
- **Particles are analytic ellipsoids**, not real specimen structure. Contrast is a mass-thickness model, not multislice.
- **Inputs are limited to plausible values.** `set_fov` outside `min_fov`..`max_fov` (1 nm to 100 um), `|defocus|` above `max_defocus` (10 um), non-finite numbers and stage moves outside the STAGE limits are refused with an error. Your instrument's limits may differ, so set these properties to match it.
- **`scan_region` is `[left, top, width, height]`** in fractions of the field, as on the SCAN device and in AutoScript.

Rendering time on a laptop (GTX-class CPU, no GPU needed): about 30 ms for a 512x512 survey tile, about 0.7 s for the first 512x512 atomic-resolution tile of a particle and 0.1 s after that. `tests/test_twin_render_perf.py` checks this.

## Ground truth

`export_ground_truth()` returns JSON with the particle table (with each particle's lattice orientation, `euler_deg`), every stage move (commanded, true, reported), and the true pose and drift of every acquisition, with each frame's scan distortion (`scan`: rotation, shear, field-of-view error, flyback, and `line_jitter`, the line count and spread from which `twin.scan_distortion.line_jitter_m` gives every line's x offset again) and the scanning detectors' gain and offset when it started (`detector`). Use it to score a workflow, for example how far a stitched map is from where the tiles really were.

Agents must not see it: `configs/mcp_dt.yaml` blocks it. Call it from test scripts through Tango.

## Properties

All new effects default to off, so `configs/DigitalTwin.yaml` behaves like an ideal instrument. `configs/digital_twin_mapping.yaml` turns them on.

| Group | Properties |
| :-- | :-- |
| Specimen | `specimen` (a mapping such as `{source: recorded, path: ..., pixel_size_m: ...}`), or `specimen_source`, `specimen_path`, `specimen_pixel_size_m`, `specimen_labels_path`; see [specimens.md](Operation/specimens.md) |
| Sample | `sample_seed`, `sample_size_xy`, `sample_particle_count`, `particle_radius_mean`, `particle_radius_std`, `particle_aspect_ratio`, `particle_elements` (comma list of Au, Pt, Pd, Ag, Cu, Ni, Fe, Si, C), `support_thickness`, `height_gradient_x/y` |
| Optics | `acceleration_voltage`, `convergence_angle_mrad`, `beam_current_pa`, `haadf_z_exponent`, `haadf_scattering_scale`, `atomic_pixel_threshold` |
| Detector | `detector_gain`, `detector_offset`, `detector_read_noise`, `detector_gain_drift` (fraction), `detector_offset_drift` (counts) |
| Scan | `scan_flyback_time`, `scan_line_jitter_px`, `scan_rotation_deg`, `scan_shear`, `scan_flyback_compression`, `fov_scale_error` |
| Stage | `stage_move_noise_std`, `stage_bias_x/y`, `stage_reported_noise_std`, `stage_backlash_x/y`, `stage_speed`, `stage_move_overhead`, `stage_settle_drift`, `stage_settle_tau` |
| Drift and time | `drift_rate_per_min`, `drift_random_walk`, `time_scale` |
| EDS | `eds_fwhm_mn_ka`, `eds_solid_angle`, `eds_dead_time`, `eds_channels`, `eds_energy_max` |
| Dose | `dose_tracking`, `contamination_max`, `contamination_dose`, `damage_critical_dose` |

Where the mapping config's values come from is noted in the file. Values marked unverified are placeholders: run the scripts in `asyncroscopy/calibration/` on your instrument and paste the YAML they print.

## Key commands

- `move_stage([x, y, z, alpha, beta])` (x/y/z metres, alpha/beta degrees), `get_stage()` (reported position)
- `set_fov(fov)`, `get_fov()`, `set_defocus(defocus)`, `get_defocus()`
- `acquire_scanned_image(["haadf", "bf"])`: one scan, one image per detector; honours the SCAN device's `scan_region`
- `place_beam([x, y])` (fractions of the field of view), `acquire_spectrum("eds")`
- `get_parameters()`: status JSON including simulated time
- `export_ground_truth()`: see above

## Specialised twins

`AtomisticDigitalTwin` (`digital_twin_base.py`) holds the original atom-by-atom renderer and the Tango plumbing shared by the specialised twins:

- `DigitalTwinDiffraction`: parked-beam nanoparticle diffraction with abTEM (`uv sync --extra diffraction`). Fixes the field of view at 500 nm and saves a simulated diffraction pattern from `acquire_camera_image()`.
- `DigitalTwinTilt`: silicon lamella tilt series with ASE/abTEM multislice.
- `DigitalTwinParticles`: voxel-backed oriented particles.
- `PolycrystallineGoldDigitalTwin`: polycrystalline gold driven by the corrector twin.

To build a twin of another electron microscope, start from `DigitalTwin` if it images a sample by scanning, or from `AtomisticDigitalTwin` if you want the shared plumbing and your own renderer. For a different kind of instrument, start from the simulated nanoindenter in [new_instrument.md](Adding_New_Hardware/new_instrument.md) and reuse the instrument-neutral models: `stage_model.py` (five axes; pass zeros for the ones you lack), `clock.py` and `rng.py` in `asyncroscopy/instruments/electron_microscope/twin/` have no Tango or electron-optics dependency and import in a fraction of a second.
