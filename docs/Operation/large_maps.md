# Large maps

How big a map the stack handles today, what each part costs as maps grow,
and how to rehearse a large map on the digital twin. Every number below is
from a file in pyTEMlib's `benchmarks/results/` (named with each table),
measured on a Windows laptop on 2026-10-29; rerun the command under a
table to reproduce it.

## One call: acquire, stitch, explore

A map is one Mapper job (`run_job` with device `Mapper`, then
`wait_for_job` until it is done). Its phases are steps of the same job, so
abort, approval and progress (`phase` in the status) cover all of them:

1. **tiles**: one step per tile. Steps are computed from the plan's grid
   when needed, so a plan of millions of tiles stores a few numbers.
2. **stitch** (`stitch`, default on; needs the Mapper's `analysis_url`):
   the run is stitched on the pyTEMlib analysis server. Every EDS point
   spectrum becomes a point of interest at its tile's stitched centre,
   linked to its spectrum key.
3. **score** (`find_regions: N`): the best N regions become points of
   interest with score, reason and scorer version.
4. **revisit** (`revisit: {"fov_m": ...}`): the stage goes to each region,
   only inside the plan box, and takes a higher-magnification image. The
   image is linked to a `high_mag` point of interest that also names its
   region (`poi:<id>`).

If the analysis server is down or stitching fails, the job still ends done
with every tile safe; `summary.stitch_error` says why and gives the
`assemble_map` call that retries it.

### Revisiting without mapping again

A plan with no area and a `revisit` that names what to image takes no tiles:

* `{"revisit": {"fov_m": 3e-8, "points": [[x, y], ...]}}` images those stage
  points (metres). Its area is the points' extent plus half an image, or
  `box_m` if given; a point outside it is refused.
* `{"revisit": {"fov_m": 3e-8, "map": "<map key>", "top_k": 3}}` images the
  map's best scored regions (score it first with `score_map`). Its area is
  the map's stage extent.

Both count against `dose_budget_e_per_a2`, and both are checked against the
stage limits at reserve like any plan.

### Resuming

An aborted or failed map (not yet released) resumes with
`Mapper_start {"reservation_id": "<id>"}` (what `run_job(device, resume=<id>)`
calls). It returns a new `job_id` and skips finished steps; a stitch or
scoring job already started on the analysis server is waited for, not
started again, and no region is imaged twice.

### What the reply holds

`results.summary` stays short however large the map: counts, the map key,
stitching quality and drift totals, and at most three regions, revisits and
spectrum keys. Everything else is on the map: every spectrum is an
`eds_point` point of interest (`list_pois`), and the run file has all keys
(`view_data`).

## Stage safety

* The STAGE device refuses positions outside its `x_limits_m`, `y_limits_m`
  and `z_limits_m` properties (default +/-1 mm, +/-0.5 mm in z; set your
  stage's), naming the limit. The digital twin checks moves against them, or
  against its own `stage_travel_m` when it has no STAGE device.
* The Mapper checks every plan at reserve: the outermost tile centres and
  every revisit point must lie inside the STAGE limits
  (`stage_device_address`), or the plan is refused before anything moves.
* No command holds the microscope for long. A twin move that would take
  more than 2 s returns at once, `stage_moving` is true until the stage
  arrives, and a second move or an acquisition is refused until then;
  `blank_beam` always answers. The Mapper waits on `stage_moving` (and stops
  waiting on abort).

## Stitching accuracy: bounded search and drift

The stage never lands exactly where it was told, and the sample drifts
over the hours a large map takes. Stitching (pyTEMlib `mosaic`) therefore:

* searches each overlap only within the stage's stated uncertainty
  (`stage_sigma_m` on the Mapper) plus the most the sample can drift between
  the two tiles (`max_drift_m_per_s`); a correlation peak outside that bound,
  or a weak one, is not a match, so a tile is never appended at a wrong
  place;
* fits the tile positions and a piecewise-linear drift over time in one
  sparse least-squares solve (Huber-weighted, outliers dropped), then
  matches the pairs again with each window centred on the stage position
  plus the fitted drift, and solves again;
* keeps tiles with no usable match at their drift-corrected stage position
  and counts them (`fallback_tiles`).

`map_info` shows `tiles`, `matches_used`, the median and 95th-percentile
match residual, `fallback_tiles` and the fitted drift.

Median placement error against ground truth, image-based synthetic runs
(64 px tiles, 30 % overlap, reported positions off by 5 % of the field of
view, drift 0.02 px per tile). From `benchmarks/results/mosaic_scale_2026-10-29.jsonl`,
`python benchmarks/mosaic_scale.py --sides 10,32,100 [--density 2e-3]`:

| Tiles | Sample (particles/px^2) | Stage only | Appending each tile | Bounded + drift solve: median / p95 / max | Tiles with no match |
|---|---|---|---|---|---|
| 100 | 8e-4 (sparse) | 3.9 px | 18 px | 0.55 / 5.6 / 10.3 px | 5 |
| 1 024 | 8e-4 | 6.8 px | 117 px | 1.26 / 3.9 / 17.8 px | 48 |
| 10 000 | 8e-4 | 59 px | 215 px | 1.09 / 3.7 / 13.2 px | 485 |
| 99 856 | 8e-4 | 583 px | not run | 0.99 / 3.5 / 24.5 px | 4 441 |
| 100 | 2e-3 (textured) | 3.9 px | 21 px | 0.38 / 4.0 / 8.6 px | 5 |
| 1 024 | 2e-3 | 6.8 px | 38 px | 0.36 / 1.4 / 17.1 px | 4 |
| 10 000 | 2e-3 | 59 px | 200 px | 0.46 / 1.1 / 12.0 px | 38 |

At 1 024 sparse tiles without drift the median is 0.84 px; with neither
stage error nor drift, 0.12 px. The fitted drift at 99 856 tiles ends within
1.2 px of the true 2 000 px.

What this does and does not show:

* On textured samples the median is under half a pixel and does not grow
  from 100 to 10 000 tiles. On the sparse sample it is about 1 px; about
  5 % of its tiles have no confident match (no particle in the overlap) and
  sit at their drift-corrected stage position, which sets the p95 and max.
* The solve alone, fed Gaussian match errors instead of images
  (`benchmarks/results/solver_scale_2026-10-29.jsonl`), does grow with size:
  median 0.36 / 0.66 / 2.3 px at 1 024 / 10 000 / 102 400 tiles with
  0.1 px match noise, 0.79 / 1.6 / 4.8 px with 0.5 px. The error is a
  stretch along the slow scan axis that the free drift can absorb (a linear
  drift and a slow-axis scale error look alike in a raster). Image-based runs
  do not show it to 100 000 tiles, but nothing here proves it cannot appear
  on a real map of that size: check large maps against a known feature
  spacing.

## How cost grows

| Part | Growth | Measured |
|---|---|---|
| Plan, reserve, job state | O(1) per step | a 2000 x 2000 plan stores the grid axes; progress is stored as done ranges |
| Pair finding | O(n) (spatial grid) | was O(n^2) |
| Registration (two matching passes, images) | O(n) | 11 s at 1 024 tiles, 114 s at 10 000, 1 027 s and 144 MB extra at 99 856 (64 px tiles; mosaic_scale) |
| The solve alone (direct, up to 400 000 unknowns) | about O(n^1.5) | 1 s at 10 000 tiles, 32 s and 143 MB at 102 400 (solver_scale) |
| The solve alone (iterative, beyond 400 000 unknowns) | | 422 500 tiles did not finish within 25 minutes (solver_scale): maps beyond 400 000 tiles are not supported yet |
| Tile reads while matching | O(n) chunk reads | 4.7 reads per chunk with strip ordering vs 46 without (10 000 tiles, 1 MB cache; tile_reads) |
| Assembly of the pyramid | O(pixels), bounded memory | 79 s and 11 MB extra for a 14 198 x 14 199 map (99 856 tiles) |
| Region read at any zoom | O(region) | one region read touches only the chunks it covers |
| Points of interest | O(points in view) | 64 x 64 spatial cells; a box query reads only the cells it touches |

## Rehearse a large map on the twin

```bash
uv run startup_scripts/run_servers.py --yaml configs/digital_twin_large_map.yaml
```

That twin has an 8 um sample with particle clusters (structure at every
zoom) and does not wait in real time, so a 40 x 40 map takes minutes. Start
the MCP servers and the pyTEMlib analysis server as in the quickstart.

## Limits for hackathon users

* Up to about 10 000 tiles: acquisition, stitching, scoring and revisits
  end to end, in under an hour on a laptop with the fast twin.
* 10 000 to 400 000 tiles: stitching and the pyramid are linear with bounded
  memory; registration takes about 17 minutes per 100 000 tiles of 64 px
  (longer for larger tiles); acquisition time is the instrument's. Check the
  map's scale against a known feature spacing (see above).
* Beyond 400 000 tiles the solve switches to an iterative method that does
  not finish within 25 minutes: split such a map into several plans.
* Use at least 25 % overlap for samples with little texture, and set
  `stage_sigma_m` and `max_drift_m_per_s` on the Mapper to your stage.
