# Large maps

How big a map the stack handles today, what each part costs as maps grow,
and how to rehearse a large map on the digital twin. Numbers are from a
Windows laptop (6 GB GPU, which mapping does not use) on 2026-10-28.

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

## Stitching accuracy: bounded search and drift

The stage never lands exactly where it was told, and the sample drifts
over the hours a large map takes. Stitching (pyTEMlib `mosaic`) therefore:

* searches each overlap only within the stage's stated uncertainty
  (`stage_sigma_m` on the Mapper) plus the most the sample can drift between
  the two tiles (`max_drift_m_per_s`); a correlation peak outside that bound,
  or a weak one, is not a match, so a tile is never appended at a wrong
  place;
* fits the tile positions and a piecewise-linear drift over time in one
  sparse least-squares solve, then matches the pairs again with each window
  centred on the stage position plus the fitted drift, and solves again;
* keeps tiles with no usable match at their drift-corrected stage position
  and counts them (`fallback_tiles` in the map's metadata).

The map records `matches_used`, the median and 95th-percentile residual of
the matches, `fallback_tiles` and the fitted `drift` (knots, total in nm,
mean rate in nm/s); `map_info` shows them.

Measured against ground truth on synthetic runs (pyTEMlib
`benchmarks/mosaic_scale.py`: 64 px tiles, 30 % overlap, a sparse particle
sample, reported positions off by 5 % of the field of view plus a drift of
0.02 px per tile): median placement error

| Tiles | Stage only | Appending each tile to the last | Bounded + drift solve |
|---|---|---|---|
| 100 | 3.9 px | 18 px | 0.55 px |
| 1 024 | 6.8 px | 117 px | 1.26 px |
| 10 000 | 59 px | not run | 1.09 px |
| 99 856 | 583 px | not run | 0.99 px |

The error of the bounded, drift-corrected solve does not grow with the
map (the fitted drift at 100 000 tiles is within 1.2 px of the true 2 000 px);
appending grows without bound. With stage error only (no drift) the
median is 0.84 px at 1 024 tiles; with neither, 0.12 px. About 5 % of tiles
in this sparse sample have no confident match and fall back to their
drift-corrected stage position, which sets most of the remaining error.
Sparse samples need more overlap: with the 15 % default on the large-map
twin most overlaps hold no particle, and the map relies on the stage (good
to about 10 nm on the twin).

## How cost grows

| Part | Growth | Measured |
|---|---|---|
| Plan, reserve, job state | O(1) per step | a 2000 x 2000 plan stores the grid axes; progress is stored as done ranges |
| Acquisition (twin, `time_scale` 0) | O(n) | 0.28 s per 128 px tile |
| Run file | O(n), one chunked tiles array | 64 MB for 1 600 tiles of 128 px |
| Pair finding | O(n) (spatial grid) | was O(n^2): 1 600 tiles took 917 s to stitch before, of which most was pair search |
| Registration (two matching passes) | O(n) | 11 s per 1 000 tiles of 64 px; 1 382 s and 146 MB extra memory at 100 000 tiles |
| Assembly of the pyramid | O(pixels), bounded memory | 111 s and 8 MB extra for a 14 198 x 14 199 map (100 000 tiles) |
| Live twin map, end to end | O(n) | 1 600 tiles with EDS, stitch, score and 3 revisits in 983 s (the old stitcher alone took 917 s for 1 600 tiles) |
| Region read at any zoom | O(region) | 0.1 to 0.2 s for a 512 px view at every level |
| Points of interest | O(batch) per add | one batch per add; `list_pois` filters by kind and box |

## Rehearse a large map on the twin

```bash
uv run startup_scripts/run_servers.py --yaml configs/digital_twin_large_map.yaml
```

That twin has an 8 um sample with particle clusters (structure at every
zoom) and does not wait in real time, so a 40 x 40 map takes minutes. Start
the MCP servers and the pyTEMlib analysis server as in the quickstart.

## Limits for hackathon users

* Up to about 10 000 tiles: everything above, end to end, in under an hour
  on a laptop with the fast twin.
* 10 000 to 1 000 000 tiles: stitching and the pyramid are linear and use
  bounded memory; registration takes about 25 minutes per 100 000 tiles of
  64 px (longer for larger tiles); acquisition time is the instrument's.
* Use at least 25 % overlap for samples with little texture, and set
  `stage_sigma_m` and `max_drift_m_per_s` on the Mapper to your stage.
