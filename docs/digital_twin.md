# Digital twins

`DigitalTwin` (`asyncroscopy/instruments/electron_microscope/digital_twin.py`) is a simulated `ElectronMicroscope`.
It provides realistic-enough image and spectrum behavior for development, testing, and demos without requiring AutoScript or hardware.

## How it works

1. On startup, the twin generates a **persistent synthetic sample** (deterministic from seed).
2. The stage pose `[x, y, z, alpha, beta]` sets the current view into that sample. x, y and z are in meters. alpha and beta are in degrees.
3. `acquire_scanned_image(["haadf"])` calls `_acquire_scanned_image()`. That helper renders the current pose and field of view, writes an HDF5 file with metadata, and returns the Tiled key if a DATA device is configured.
4. `acquire_spectrum("eds")` calls `_acquire_spectrum()`. That helper estimates the composition at the beam position and saves an HDF5 spectrum.

This means moving the stage navigates the sample, and revisiting the same pose can reproduce the same view when stage noise is disabled.

## Available features

- Persistent sample per device session
- Deterministic sample generation via seed
- Stage-coupled navigation in **XY + Z + alpha/beta tilt**
- Beam-position-dependent spectrum simulation
- Acquisitions saved to files with the same HDF5 writer as the hardware path
- Configurable stage move noise
- Viewport metadata reporting
- Manual sample regeneration with a new seed

## Twins that extend it

Each twin is a subclass of `DigitalTwin` with its own config:

| Class | Config | Simulates |
|---|---|---|
| `DigitalTwin` | `configs/DigitalTwin.yaml` | Particles for HAADF and EDS |
| `DigitalTwinParticles` | `configs/digital_twin_particles.yaml` | Voxel-backed oriented particles for HAADF and EDS |
| `DigitalTwinDiffraction` | `configs/diffraction.yaml` | Parked-beam nanoparticle diffraction |
| `DigitalTwinTilt` | `configs/digital_twin_tilt.yaml` | Silicon FIB-lamella tilt and diffraction |
| `PolycrystallineGoldDigitalTwin` | `configs/polycrystalline_gold.yaml` | Atomistic FCC gold grains through a twin corrector probe |

`DigitalTwinDiffraction` keeps the HAADF overview, fixes the field of view at 500 nm, and saves a diffraction pattern at the beam position with `acquire_camera_image()`. If the beam is not on a nanoparticle, it waits 3 seconds and saves detector noise. It uses `abTEM` with a small Au FCC unit cell, and `ase` for the structure.

The diffraction and tilt twins need the optional simulation packages:

```bash
uv sync --extra diffraction
```

## Corrector twin

`DigitalTwinCorrector` (`hardware/corrector_digital_twin.py`) stands in for a CEOS corrector. It is meant for testing code, not for simulating corrector physics. Corrections are exact and the readout has no noise unless you set `measurement_noise_fraction`. What it does copy is the shape of the real CEOS replies, so code that works on the twin also works on the hardware:

- `acquire_tableau` accepts only `Fast`, `Standard` or `Enhanced` (case-sensitive) and an angle above 0 mrad. The CEOS defaults are 9, 18 and 34 mrad.
- Each tableau type reports only the aberrations it fits. `Fast` gives C1, A1, A2 and B2. `Standard` adds C3, A3, S3 and A4. `Enhanced` adds D4 and B4. WD is included in all three.
- Every aberration is sent as an `[x, y]` pair. For the scalar aberrations C1 and C3, `y` is 0.
- `correct_aberration` removes the value you send. For C1 and C3 it accepts one value, or a pair whose y value is 0. A successful correction returns a `null` result, the same as the real server.

`get_aberrations_coeff_sim` and `set_aberrations_coeff_sim` read and write the simulated state directly. They keep one value for C1 and C3.

## Key properties

- `sample_seed`: controls deterministic sample generation
- `sample_particle_count`: controls synthetic particle count
- `sample_extent_scale`: controls sample XY size relative to FoV
- `stage_move_noise_std`: adds Gaussian perturbation to stage moves
- `frame_noise`: off by default, so the same pose always gives the same image. Turn it on to give each frame new shot noise, as a real detector does. The particle twin always draws new noise and ignores this setting.

Set them under `instrument.properties` in the config.

## Key commands

- `move_stage([x, y, z, alpha, beta])`
- `get_stage()`
- `set_fov(fov)`
- `set_defocus(defocus)` and `get_defocus()`
- `acquire_scanned_image(["haadf"])`
- `place_beam([x, y])`
- `acquire_spectrum("eds")`
- `get_viewport_metadata()`
- `get_ground_truth()`: the true particles in the current view, in image pixels (row, column, radius, type). Use it to grade an agent's results. The MCP configs block it, so the agent never sees the answer. Only the base twin has it; the particle and gold twins refuse it.

Every image the twins save carries `pixel_size_m`, `dwell_time_s`, `frame_time_s`, `imsize_px` and `acquisition_start_utc`, under the same names as images from the real microscope.

Note: the code no longer has `sample_extent_scale` or a sample regeneration command. `DigitalTwin` now sets the sample size with `sample_size_xy` and `sample_size_z`, in meters.
