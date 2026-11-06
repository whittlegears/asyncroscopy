# Specimens: swapping the sample under the twin

The mapping twin (`DigitalTwin`, see [digital_twin.md](../digital_twin.md)) is two models: an **instrument** and a **specimen**. The specimen is a plug-in. You can image seeded particles, a recorded image, or a crystal film on the same simulated microscope, and run the same controller on each.

This answers three questions from the hackathon design principles:

- **Can we replace the specimen?** Yes. Set one device property.
- **Where does it come from?** From a physical model (`procedural` particles, `crystal` film), from recorded data (`recorded`), or from any source you register.
- **Can we test the same controller across different samples?** Yes. `tests/test_specimens.py` runs one Mapper plan on three specimens. The stage error, drift, timing, dose and focus are identical in all three runs. The images and spectra are different.

## The coupling

The instrument never looks inside the sample. It asks the specimen four questions, all in sample coordinates: metres, image rows along +y, columns along +x.

| The instrument part | Asks the specimen | Method |
| :-- | :-- | :-- |
| Renderer (`twin/render.py`) | the projected Z^n-weighted mass thickness, `sum(Z^n * atoms/A^3 * thickness/A)`, on a regular grid at the frame's pixel size, at `level` "survey" or "atomic" | `projected_weight(x_min, y_min, n, pixel_m, level, z_exponent, damage)` |
| EDS (`twin/eds_model.py`) | atoms per square angstrom of each element under the beam | `areal_density(x, y)` |
| Focus (renderer, autofocus, Tuner truth) | the specimen height, which adds to the defocus | `height_at(x, y)` |
| Dose map (`asyncroscopy/sim/dose.py`) | which features are in the frame and where they are, so it can fade them past the critical dose | `in_box(...)`, `feature_centers` |
| Renderer and EDS | the support film's element and thickness | `support_element`, `support_thickness_m` |
| Dose map and status | the sample's extent | `bounds` |

Everything else belongs to the instrument and is the same whatever the specimen:

- the stage (commanded, true and reported positions, backlash, settling, drift);
- the clock;
- scan distortion;
- the probe: aberrations, defocus blur and the atomic-resolution probe;
- the detector response for HAADF, ADF and BF, and its counting noise;
- the EDS detector;
- the dose bookkeeping and carbon contamination, which the beam deposits.

The coupling is one-way. The instrument decides *where* each pixel lands on the sample and *how* the signal is detected. The specimen only says *what is there*. Its random draws come from its own stream (`EventRng.for_event("sample")` or `"crystal"`), so changing the specimen does not change any instrument random number.

Ground truth is the specimen's fifth method, `ground_truth()`. Only evaluators read it, through `export_ground_truth`, which the MCP configs block for agents. The instrument never reads it.

## Choosing a specimen

Put a `specimen` mapping under the twin's properties in the stack YAML. `run_servers` passes it to the twin as one string property, and the twin parses it as JSON or YAML:

```yaml
instrument:
  class_name: DigitalTwin
  properties:
    specimen:
      source: recorded
      path: configs/specimens/recorded_example_haadf.png
      pixel_size_m: 1.5625e-9
      labels_path: configs/specimens/recorded_example_labels.json
```

Flat properties do the same job: `specimen_source`, plus `specimen_path`, `specimen_pixel_size_m` and `specimen_labels_path` for `recorded`. Use them where a property value cannot contain `:`, as in the Tango test context. Keys in the `specimen` mapping override the flat properties.

Without either, the twin uses `procedural`, exactly as before. The flat `sample_*`, `particle_*`, `support_thickness` and `height_gradient_*` properties stay the procedural source's settings. The other sources use them as defaults for size, support film and height. Each source also takes a `seed` option, which gives a different sample on an identical instrument. Mistyped options are refused, and the error lists the options the source takes.

| Example config | Source |
| :-- | :-- |
| `configs/digital_twin_mapping.yaml` | `procedural`: 400 particles over 2 um |
| `configs/digital_twin_recorded.yaml` | `recorded`: a 400 nm HAADF image with labels |
| `configs/digital_twin_crystal.yaml` | `crystal`: a 2 um polycrystalline Au film |

All three share the same instrument settings. `get_parameters` reports `specimen_source`, which says what kind of sample is loaded but not what is on it.

## The sources

### `procedural` (default)

This source is the seeded particle table that the twin has always used (`twin/sample_model.py`, `SampleModel`). It places ellipsoids of the given elements, optionally in clusters, on a support film:

- **survey:** analytic projected thickness;
- **atomic resolution:** the particles in view are built atom by atom with ASE.

Options are any `SampleParams` field (`particle_count`, `elements`, `cluster_count`, ...) and `seed`. `support_texture` (a fraction, default 0) and `support_texture_scale_m` make the film's thickness vary at every length from that scale down to 0.5 nm, from its own seeded stream so no particle moves; `subpixel_atoms` (default false) draws each atom at its sub-pixel position. With no options it is bit-identical to the twin before specimens became plug-ins. `tests/data/specimen_procedural_golden.npz` holds the check: two survey frames with dose damage, one atomic-resolution frame, one EDS spectrum and the ground truth.

### `recorded`

This source is an image that someone has already acquired, laid on the sample plane (`twin/specimen_recorded.py`).

| Option | Meaning |
| :-- | :-- |
| `path` | `.npy`, `.npz`, `.png`, `.tif` or `.h5`/`.hdf5`. A relative path is resolved against the working directory, then against the repository. |
| `pixel_size_m` | The image's pixel size. It can be left out when the file has it: `pixel_size_m` in an `.npz`, or a `pixel_size_m` attribute in HDF5, as twin acquisitions and Mapper run files have. |
| `dataset`, `index` | Which array of an `.npz` or `.h5` file to use, and which frame of a stack. A Mapper run's `tiles` is a stack. |
| `center_m` | Where the image centre sits in sample coordinates. The default is `[0, 0]`. |
| `invert` | Set it for bright-field data, where material is dark. |
| `contrast_weight` | The projected weight of the brightest pixel. The default, 1.6e4, scatters about 30 % to HAADF, like a 16 nm gold particle. |
| `labels_path` | Labelled particles. See below. |
| `composition`, `eds_areal_density` | Without labels: EDS composition such as `{Au: 3, Pd: 1}`, spread in proportion to the image intensity, and the atoms/A^2 at the brightest pixel. |

The intensity is scaled to 0..1 between the 0.5th and 99.9th percentiles and becomes the projected weight. The instrument then images it like any other specimen, so noise, dose, drift, scan distortion, defocus blur and the HAADF/ADF/BF response all still apply. When the field's pixels are larger than the recorded ones, the image is low-passed first so it does not alias. Off the image there is vacuum and the support film.

The twin cannot invent detail finer than the recorded pixel. An atomic-resolution field of a 1.5 nm-pixel image is only smooth interpolation, blurred by the probe.

The labels file is JSON in the same format as `export_ground_truth`'s particles, so a twin's own export is a valid labels file:

```json
{"particles": [{"center_m": [1.2e-8, -3.0e-8], "semi_axes_m": [8e-9, 8e-9, 4e-9], "theta_rad": 0.0, "element": "Au"}]}
```

`radius_m` can stand in for `semi_axes_m`, as `[r, r, r/2]`. Labels give EDS its composition and the dose map particles to fade.

`configs/specimens/recorded_example_haadf.png` is a stand-in made by the twin: seed 99 rendered at 1.5625 nm pixels with shot noise. Its particles are in `recorded_example_labels.json`. Point `path` at your own data.

### `crystal`

This source is a polycrystalline film of one element, built from its ASE unit cell (`twin/specimen_crystal.py`). The film is cut into Voronoi grains. Each grain has its own seeded in-plane rotation, thickness and small tilt off the [001] zone axis.

- **survey:** the film's mass thickness. Grains near the zone axis are brighter by up to `channeling`, so grains show in an overview.
- **atomic resolution:** the projected [001] columns of the cubic cell, rotated per grain and cut at grain boundaries. Each column is weighted by the atoms it holds through the grain's thickness. For fcc gold this is a square lattice of columns a/2 = 2.04 A apart, and the test finds that spacing in the image's FFT.

| Option | Default | Meaning |
| :-- | :-- | :-- |
| `element` | `Au` | Any element in `ELEMENTS`. The structure and lattice constant come from there. |
| `lattice_constant_m` | from `ELEMENTS` | Overrides the lattice constant, for strained films. |
| `thickness_m` | `5e-9` | Film thickness. |
| `thickness_variation` | `0.15` | Fractional spread of the thickness from grain to grain. |
| `grain_count` | `12` | Number of grains. |
| `mosaic_spread_mrad` | `20` | Spread of the grains' tilt off the zone axis. |
| `channeling` | `0.3` | Extra HAADF signal for grains close to the zone axis. |
| `size_xy_m` | `sample_size_xy` | Size of the film. |
| `support_thickness_m` | `support_thickness` | Thickness of the support film under it. |
| `height_gradient` | `height_gradient_x/y` | Specimen height change per metre. |
| `seed` | `sample_seed` | Seed for the grains. |

There is no multislice: contrast is the same mass-thickness model as the particles, which keeps the source cheap at any field of view. An abTEM-backed source would plug in the same way. Its `projected_weight` would return the projected potential.

## What ground truth each source gives

| Source | `export_ground_truth()["particles"]` | `export_ground_truth()["specimen"]` |
| :-- | :-- | :-- |
| `procedural` | every particle: id, centre, semi-axes, orientation, element, lattice orientation (`euler_deg`) | `known: true`, count, elements, mean radius, clusters |
| `recorded` without labels | `[]` | `known: false`, path, shape, pixel size |
| `recorded` with labels | the labels | `known: true`, `labelled: true` |
| `crystal` | `[]`: it has no particles | `known: true`, `grains` (seed point, rotation, tilt, thickness) and `lattice` (element, structure, lattice constant, zone axis) |
| your source | what its `ground_truth()` returns | its `describe()` and the rest of `ground_truth()` |

The instrument's own truth does not depend on the specimen. Every source gives it in full:

- every stage move (`moves`: commanded, true and reported);
- the true centre and drift of every acquisition (`acquisitions[*].center_true_m`, `drift_*`);
- each frame's scan distortion, every line's jitter, and the detectors' gain and offset (`acquisitions[*].scan`, `acquisitions[*].detector`);
- the dose (`dose`);
- the focus (`focus`).

A scorer that checks *where the instrument went*, such as stitching accuracy against `center_true_m`, works on every source. A scorer that checks *what was found*, such as particles detected against particles placed, needs `specimen.known` to be true. Check it before scoring.

## Adding a source in about 40 lines

A source subclasses `Specimen` and is registered by name. It implements `from_options`, `projected_weight` and `areal_density`. The other methods have defaults:

- `height_at`: flat, or `height_gradient`;
- `support_texture`: a uniform support film (`procedural` varies its film with its `support_texture` option);
- `in_box` and `feature_centers`: no damageable features;
- `ground_truth`: `known: false`.

Here is a gold cross-grating replica, the usual magnification standard:

```python
import numpy as np

from asyncroscopy.instruments.electron_microscope.twin.specimen import Specimen, check_options, register_specimen

GOLD_Z, GOLD_ATOMS_PER_A3, ANGSTROM = 79, 0.059, 1e-10


@register_specimen
class CrossGrating(Specimen):
    """Gold lines in two directions with square holes between them; pitch 463 nm for 2160 lines/mm."""

    source = "cross_grating"

    def __init__(self, pitch_m: float, line_fraction: float, thickness_m: float, size_m: float):
        self.pitch_m, self.line_fraction, self.thickness_m = pitch_m, line_fraction, thickness_m
        self.bounds = (-size_m / 2, size_m / 2, -size_m / 2, size_m / 2)

    @classmethod
    def from_options(cls, options, context):
        check_options(cls.source, options, {"pitch_m", "line_fraction", "thickness_m"})
        size = context.sample.size_xy_m if context.sample else 1e-5
        return cls(float(options.get("pitch_m", 4.63e-7)), float(options.get("line_fraction", 0.4)), float(options.get("thickness_m", 2e-8)), size)

    def _gold(self, x, y):
        x, y = np.asarray(x), np.asarray(y)
        on_line = (np.mod(x / self.pitch_m, 1.0) < self.line_fraction) | (np.mod(y / self.pitch_m, 1.0) < self.line_fraction)
        x_min, x_max, y_min, y_max = self.bounds
        return on_line & (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)

    def projected_weight(self, x_min, y_min, n, pixel_m, level, z_exponent, damage):
        centres = (np.arange(n) + 0.5) * pixel_m
        x, y = np.meshgrid(x_min + centres, y_min + centres)
        film = GOLD_Z**z_exponent * GOLD_ATOMS_PER_A3 * self.thickness_m / ANGSTROM
        return np.where(self._gold(x, y), film, 0.0)

    def areal_density(self, x, y):
        return {"Au": GOLD_ATOMS_PER_A3 * self.thickness_m / ANGSTROM} if self._gold(x, y) else {}

    @property
    def element_symbols(self):
        return ("Au",)

    def ground_truth(self):
        return {"known": True, "particles": [], "pitch_m": self.pitch_m, "line_fraction": self.line_fraction}
```

`tests/test_specimens.py` runs this block from this page, so the example stays correct. The twin finds the source in one of two ways:

- the module defining it is imported before the twin starts;
- a package installs it with the `asyncroscopy.specimens` entry point:

```toml
[project.entry-points."asyncroscopy.specimens"]
cross_grating = "my_package.grating:CrossGrating"
```

Then select it with `specimen: {source: cross_grating, pitch_m: 4.63e-7}`.

Rules for a source:

- **Coordinates:** work in metres in the sample frame. Never read the stage or the clock: the instrument has already worked out where each pixel landed.
- **Signal:** return mass thickness, not detector counts. The instrument applies the detector response, the probe, the noise and the support film. If you add noise or blur yourself, it will be applied twice.
- **Random numbers:** draw them at construction from `context.rng_for(options).for_event("<your source>")`. A seed then gives the same sample, and the instrument's random streams are untouched.
- **Size:** keep `projected_weight` cheap. It is called once per frame and detector on the frame's padded grid, which is up to imsize x imsize.
- **Atomic resolution:** at `level == "atomic"` the pixels are under 0.2 nm. Place atoms or columns if you have them (see `thermal_smear` in `sample_model.py`), or return the smooth weight.

## Same controller, different samples

`test_the_same_mapper_plan_on_three_specimens` in `tests/test_specimens.py` starts three twins from `MAPPING_TWIN_PROPERTIES`, the realistic mapping instrument in `tests/conftest.py`. They differ only in the specimen: `procedural`, `crystal`, and `recorded` with labels. The test runs the same Mapper plan on each, with `time_scale: 0`: a 3 x 3 grid of 32-pixel tiles and an EDS point per tile. It takes about 2 s. It checks:

- `moves`, `acquisitions` (true centres, drift), `simulated_time_s`, `dose` and `focus` from `export_ground_truth` are identical across the three runs.
- The run files' reported stage positions, commanded positions and acquisition times are identical.
- The tiles differ for every pair of specimens.
- Two runs' spectra at a point differ exactly where either specimen has material under the beam. Where neither has, both show the same support film with the same detector noise.

To try a controller of your own across samples, use the same pattern. Start one twin per specimen with identical instrument properties, run the controller on each, and compare `export_ground_truth()` and its results.

## What is not swappable yet

- The atomistic twins built on `ElectronMicroscopeTwin` (`digital_twin_particles.py`, `digital_twin_tilt.py`, `digital_twin_diffraction.py` and `polycrystalline_gold_digital_twin.py`) still generate their own samples in `_generate_sample`. Their renderers read per-view state this interface does not carry: the tilt twin's lattice fields follow the stage tilt, the gold twin's grain tree works with the corrector probe, the particles twin rebuilds a voxel recipe whenever the field of view changes, and the diffraction twin's particle maps feed abTEM. A `Specimen` gives projected weight, areal density and height at a point, so turning them into plug-ins would change what they render. The `crystal` source already gives `DigitalTwin` a polycrystalline film.
- The specimen cannot be changed while the twin runs. Restart the twin with a new `specimen` property. This is deliberate: a Tango command to swap it would be an agent tool.
- Stage z, alpha and beta still do not change what any specimen looks like.
