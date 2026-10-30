# Multiscale sample maps on the twin, transferable to the Spectra 300

*   **Author:** whittlegears
*   **Status:** Draft
*   **Date:** 2026-10-30
*   **Predecessor:** `concepts_brief.md` (three rounds of six-lens review)

--------------------------------------------------------------------------------

## Overview

A microscopy group wants to map a whole TEM grid and then zoom in one viewer from the whole grid down to atomic columns, with no visible seams and features that stay continuous across tile edges. They label regions on these maps, share them on the lab's data server (Tiled), and cite them in papers. Hackathon teams build on the same code for their own instruments.

This design covers the backend that makes such maps: the digital twin that the maps are first built and judged on, stitching, cross-scale placement of finer images on coarser ones, and the storage rules that keep maps and their dependents intact. The GUI viewer, transfer to the real instrument and tamper-evident sealing follow, in that order, and are scoped here only as far as the backend has to prepare for them.

## The Problem

Today a person can take an image, a spectrum or a stage move by hand about as fast as an agent can. What a person cannot do is map thousands of tiles and stitch them. The pieces exist (the Mapper job device, pyTEMlib's stitcher, a twin that draws one sample from survey to atomic scale), but four things stop them from producing a map anyone could cite:

1.  **Maps are deleted under their dependents.** A re-stitch deletes the previous map in Tiled, and a local re-stitch overwrites the file. Points of interest live inside the map node and are replaced by deleting the old layer. Once finer images are placed on a map, any re-stitch would orphan all of them.
2.  **Stitching has never been measured on the twin,** and it is known to lose accuracy: tiles are rounded to whole pixels when the map is drawn (`mosaic.py:595`), brightness is never matched between tiles, and tiles with no match are blended in from the stage's guess. The benchmark scores positions before rounding, so its numbers hide this.
3.  **There is no cross-scale placement.** Finer images are positioned by stage reading, which is 150 to 350 nm off, against fields of 10 to 50 nm. A child image cannot be placed on its parent at all.
4.  **The twin is too kind in some ways and too thin in others.** Its support film is uniform (nothing to register on at grid scale), gain never drifts (brightness matching cannot be tested), atoms snap to whole pixels (overlapping atomic tiles disagree by up to a pixel), its ground truth lacks the scan distortion of each frame, and its dose map is too coarse for fine tiers. The Mapper's stated stage uncertainty (30 nm) is a fifth of the twin's real error (about 150 nm), so many tile pairs are searched in the wrong window.

There is also one security gap found during review: the Spectra 300's agent endpoint configuration (`configs/mcp.yaml`) offers agents data tools the twin's configuration hides, and keeps no trace.

## Goals

The first deliverable is phases P0 to P3. Each goal has a measured check; a phase is not done until its checks pass.

*   **G0. Nothing a map depends on is deleted.** Re-stitching a run keeps every previous map. No agent tool deletes a map or a point-of-interest layer. Every agent endpoint configuration blocks at least what the twin's does and writes a trace.
*   **G1. A twin that is hard enough to be worth passing,** at the 8 to 20 µm scale: faint support texture visible at every pixel size, slow gain and offset drift, atoms drawn at sub-pixel positions, coarse and fine drawings that agree at the 0.2 nm switch, ground truth that includes each frame's distortion, a dose map that keeps a 50 nm frame's dose local, and a Mapper whose stage uncertainty matches the twin's.
*   **G2. Stitching measured against truth on the twin, then fixed until** (20 × 20 tiles of 512 pixels, 30 % overlap, distortion on, ±10 % gain drift): median placement error ≤ 0.3 px, 95th percentile ≤ 0.7 px, worst ≤ 2 px; no tile drawn from the stage's guess where its overlap holds enough features; brightness step across tile edges ≤ 1.2 × the step inside tiles; mean brightness difference between neighbours ≤ 1 %.
*   **G3. Three tiers on the twin** (about 16× per step, adaptive): each child image placed on its parent with a recorded placement, 95th-percentile error ≤ 0.5 parent pixel, at least 90 % placed, zero wrong placements (more than 2 parent pixels off) among those marked placed, and an atomic image placed in the survey through two placements within the two steps' allowances added up. (The first draft asked for atomic columns within 1 atomic pixel after two placements; that cannot be met by any registration from a survey: an image's absolute position is only as good as the coarsest placement in its chain, and 0.2 survey pixels of 4.7 nm are about 26 atomic pixels.)

## Non-Goals

*   An atomic tiling of the whole grid. At 20 pm pixels it would take about 3 × 10^8 years. Atomic columns are imaged only on small, dose-capped regions.
*   Imitating a specific real specimen. The twin is synthetic; `specimen_recorded.py` can already replay a real image.
*   Auto mode on hardware. Real runs use approve mode.
*   Signatures for readers of papers, and DOIs.
*   In this deliverable: the 2 mm grid (P5), the deep-zoom viewer (P4), the real-instrument run (P6), sealing (P7). Their requirements on P0 to P3 are listed where they constrain a decision.
*   The parked streams: labels as outlines, labels in the data server, per-user identity, removing the GUI's named hard-codings, calibration fitting from real runs, a source for every number an agent reports.

## Decisions taken from the brief's open questions

The brief left six questions open. The user asked to proceed to design and implementation, so each is decided here with its reason; any of them can be reopened.

| # | Question | Decision | Reason |
| :-- | :-- | :-- | :-- |
| 1 | Order and first deliverable | P0 to P3, in that order. | P0 is needed whatever else is decided; P1 to P3 are the smallest set that proves a multiscale map on the twin. |
| 2 | Fixed or adaptive tier ratio | Adaptive, target 16×, bounded 4× to 20×. A **bridge image** is inserted when the parent patch under a child holds too little detail. Bridges are images in the child's tier marked `bridge`, not a tier of their own. | A fixed ratio fails exactly where the parent patch is blank, and the brief measured that at 16 to 20× the child covers only 32 to 51 parent pixels. |
| 3 | Survey pixel: point or area | Point sample, as a small probe gives. | An area average would make registration easier on the twin than on the instrument and flatter the stitching numbers. The consistency check compares the two drawings where a pixel is no larger than the 0.2 nm switch, where point and area coincide within the probe. |
| 4 | Thresholds | The brief's proposed numbers, as in Goals. | They come from the existing benchmark's measured spread; they are revisited only with a measurement that shows one is unreachable for a physical reason. |
| 5 | Placement transform and storage | Similarity (shift, rotation, scale) after rectification, anchored to the parent **tile** and its version. Placements are JSON records shaped like OME-NGFF coordinate transforms (scale, rotation, translation). No OME-Zarr writing yet. | Anchoring to a tile lets a re-stitch carry its children unchanged. Writing OME-Zarr now would be a second storage format before a viewer reads it. |
| 6 | Viewer transport | Deferred to P4. P0 to P3 only promise that every tier is readable as stored 512-pixel chunks through Tiled's block endpoint. | The viewer is not in this deliverable. |

## Critical User Journeys (CUJs)

### CUJ 1: Re-stitch without losing anything

*   **Given**: a run with a stitched map `M1`, a point of interest scored on it, and one child image placed on one of its tiles.
*   **When**: an agent calls `assemble_map` on the same run with different settings.
*   **Then**: a new map `M2` exists beside `M1`; `M1` is still readable with every level; the point of interest is still listed for the run; the child's placement still names its parent tile and resolves on `M2` because tiles keep their identity; the run's "latest map" names `M2`.

### CUJ 2: Judge stitching on the twin

*   **Given**: the twin started from `configs/digital_twin_mapping.yaml` with distortion and gain drift on.
*   **When**: `benchmarks/mosaic_twin.py` maps a 20 × 20 grid, stitches it, and reads the twin's ground truth through the operator endpoint.
*   **Then**: it writes `benchmarks/results/mosaic_twin.md` with placement error, seam steps, brightness differences and the unplaced rate, and exits non-zero if any G2 threshold fails.

### CUJ 3: Zoom from survey to atomic columns on the twin

*   **Given**: a survey tier stitched and a region scored on it.
*   **When**: the multiscale skill acquires a T1 image over the region, places it on its parent tile, re-centres on the strongest feature by image, acquires T2 and T3 the same way, and records each placement.
*   **Then**: every image is listed with its placement record and status; reading the chain survey → T1 → T2 → T3 gives each atomic column's position in survey coordinates within 1 T3 pixel of the twin's truth; an image that could not be placed is listed as `unplaced` and is never drawn into its parent.

### CUJ 4: An agent on the Spectra endpoint

*   **Given**: `configs/mcp.yaml` started against the Spectra 300.
*   **When**: an agent lists its tools.
*   **Then**: it sees no tool the twin's agent endpoint hides (Tiled configuration, data registration, approval internals, ground truth), and every call it makes is written to the trace.

## Technical Plan (Architecture)

### Components and where they live

```text
           asyncroscopy (instrument stack)                     pyTEMlib (analysis server :8003)
 ┌───────────────────────────────────────────────┐   ┌──────────────────────────────────────────────┐
 │ twin/  sample_model, render, dose, detector,   │   │ image_tools/mosaic.py   solve + assemble      │
 │        scan_distortion, stage_model            │   │ image_tools/register.py register_child  (new) │
 │   ground truth (operator only)                 │   │ mcp_server/map_store.py versioned maps, run   │
 │ mapper.py  tiles → run file (HDF5 in Tiled)    │   │                         point layer           │
 │ jobs/      plans, dose, pause/continue         │   │ mcp_server/map_tools.py assemble_map,         │
 │ mcp/       agent/operator/monitor endpoints    │   │                         place_image (new),    │
 │ data/      record fields, dose ledger          │   │                         list_placements (new) │
 └──────────────────┬────────────────────────────┘   └──────────────────┬───────────────────────────┘
                    │ run files, frames                                  │ maps, placements, points
                    ▼                                                    ▼
                       Tiled (:9091): one key per run, one key per map version,
                       one point layer per run, one placement log per run
```

asyncroscopy owns acquisition and the twin. pyTEMlib owns stitching, placement and map storage, and never imports asyncroscopy (this rule already holds in `map_store.py`). Tiled is the one store both read.

### Storage model after P0

*   **Map versions.** Each `assemble_map` writes a new key, `<run stem>_map_<UTC yyyymmddThhmmss>_<4 hex>`. The first map of a run made before this change keeps its old key `<run stem>_map`. No code path deletes a map version. Map metadata gains `version` (the key), `inputs_sha256` (a hash of the run key, the run's tile count and written mask, the stitch settings and the pyTEMlib version) and keeps the existing `id`, `parents` and sample fields. `current_map(run)` already returns the newest key in sort order; it keeps doing so, and it is the only "latest pointer", because a stored pointer would be a second source of truth.
*   **Local maps** follow the same rule: `<run stem>_map_<stamp>_<hex>.h5` beside the run, never overwritten. `current_map` for a local run globs these and the legacy `<run stem>_map.h5` and returns the newest.
*   **Points of interest leave map nodes.** They move to one container per run, `<run stem>_points`, with the existing layer layout (spatial cells for EDS points, a highlights group for the rest). Points are in stage metres already, so they do not depend on a map version. Replacing points (`score_map` re-scoring a tag) builds a new layer and switches the container's `poi_layer` metadata to it, as today, but **keeps** the old layer; the container's metadata lists every layer with the time it was replaced. Reading a map's points reads its run's point container. Maps written before this change keep their in-node points readable; the first write of points for such a run copies them into the run container once and records that it did.
*   **Agent tools.** None of the analysis server's tools deletes a node. `assemble_map` aborting a half-written map still removes its own unfinished key; that is the only deletion left, and it applies only to a key nothing can yet refer to (the map is not listed until its metadata says `kind: stitched_map`).

### Twin changes (P1)

*   **Support texture**: a seeded thickness field on the support film, a sum of plane waves with lengths from 2 µm down to 0.5 nm, defined at every point by a deterministic function of position. A frame draws only the waves at least two of its pixels long (finer ones would alias into a pattern that depends on where the frame's grid starts), so overlapping frames of one pixel size see the same film and a coarse frame sees a smoothed version of a fine one, as on the instrument. Its standard deviation is 30 % of the film thickness: at 1 % (the brief's first guess) a 10 nm carbon film's variation is about 10^-4 counts per pixel, far below shot noise, and nothing could register on it.
*   **Gain and offset drift**: the scanning detectors' gain and offset follow a seeded sum of slow sine waves in the twin's clock (periods 10 min to 2 h; the mapping configs allow ±10 % gain and ±2 counts offset). The values for every frame go into ground truth. EDS counts do not drift.
*   **Sub-pixel atoms**: atoms are drawn by evaluating their Gaussian at true sub-pixel positions, not placed on the nearest pixel.
*   **Consistency at the switch**: a test draws the same box with the survey branch and the atomic branch at 0.2 nm, blurs both to the probe size, and requires the mean projected weight to agree within 2 % and the correlation to exceed 0.9.
*   **Ground truth**: `export_ground_truth` adds each acquisition's scan distortion parameters, its line-jitter draw (line count and spread; `scan_distortion.line_jitter_m` regenerates every line's offset from the seed and acquisition index, so a 10,000-tile map's truth stays small), its gain and offset, and each particle's lattice orientation. It stays operator-only.
*   **Sparse dose map**: cells are created only where dose lands, sized to the frame that put them there (about an eighth of its field, in power-of-two steps from the largest cell down to 0.5 nm), so a 50 nm frame on a 20 µm sample keeps its dose in cells of about 6 nm. A parked beam's dose is counted over the Mapper's 10 nm spot spread.
*   **Stage uncertainty**: the twin reports the commanded position plus 10 nm of readback noise, while the stage really lands with 150 nm of noise plus bias and backlash, so the Mapper's `stage_sigma_m` in every twin configuration becomes 150 nm (it was 30 nm); a test fails if a twin configuration states less than its `stage_move_noise_std`.

### Stitching changes (P2)

The benchmark is written first and decides which of these are needed; each is added only when the benchmark shows its error.

*   **Sub-pixel assembly**: tiles are resampled at their solved sub-pixel position (windowed-sinc, Lanczos-3) instead of rounded.
*   **Gain and offset**: one linear solve over all overlap pairs for a per-tile gain and offset (anchored so the mean gain is 1 and the mean offset 0), applied at assembly and recorded in map metadata so the raw tiles plus the record rebuild the map.
*   **Blending**: overlap regions use a feathered weight that falls to zero at each tile's edge (already the shape of `_blend_weight`), applied after gain matching; multi-band blending is added only if the seam check still fails.
*   **Rectification**: a global distortion (rotation, shear, scale per scan setting) is estimated from the overlap residuals and removed from each tile before solving; per-tile drift-shear only at fine tiers.
*   **Unplaced tiles**: a tile with no accepted match is not drawn. Its position, match count and residual are stored, and `map_info` lists it as unplaced.

#### What the benchmark found, and what was built (2026-10-30)

`benchmarks/mosaic_twin.py` (results in `benchmarks/results/mosaic_twin.md`), 20 x 20 tiles of 512 pixels, 30 % overlap, on the large-map twin with every effect on:

| Check | Before P2 | After P2 | Threshold |
| :-- | :-- | :-- | :-- |
| Tiles against each other (after removing the shared affine map, fitted to the tiles within 2 px of it): median / 95th / worst | not measured | 0.10 / 0.24 / 0.50 px | 0.3 / 0.7 / 2 px |
| Tiles placed by a match but more than 2 px from where they belong | not measured | 0 | 0 |
| Against truth after only the common offset (the shared distortion is a calibration) | not measured | 3.5 / 8.9 px | — |
| Seam step across edges / inside tiles | not measured | 0.97 | 1.2 |
| Brightness difference between neighbours | 1.2 to 35 % (gain drift, misplacement) | 0.27 % | 1 % |
| Unplaced tiles (listed, drawn only where nothing else is) | 3 of 16 at 560 nm fields | 3 of 400 | 0 |
| Stitching time, 400 tiles | 956 s | about 300 s | — |

Findings that changed the plan:

*   **The stage error must be smaller than the overlap.** At 560 nm fields the twin's stage (150 nm noise, up to 250 nm backlash) moves tiles further than a 30 % overlap, so neighbours often do not overlap at all. A map needs fields of about 2.4 µm or more with this stage, or a larger overlap. The Mapper should refuse or warn on a plan whose overlap is under about 3 stage sigmas plus backlash (follow-up).
*   **The Mapper's stage_sigma_m must include backlash.** At 150 nm the stitcher rejected true matches of tiles the stage put 470 nm off; the twin configurations now state 290 nm, and a test checks them.
*   **Scale needs a calibration on small maps.** The only metric reference is the stage, and its error hides a 1 % field-of-view error until a map has hundreds of tiles. Large maps measure it (0.990 against the twin's 0.990); small maps take it as a calibration (`scan_linear`), measured once per scan setting.

Built in pyTEMlib (`mosaic.py`, commit 828f787): sub-pixel drawing by cubic spline (not Lanczos: same accuracy here, simpler); gain and offset modelled as smooth in acquisition time with priors and caps (per-tile values ran away across large maps in review); the scan's shared linear map measured with the stage regressed on tile centres and the solve repeated with the stage prior in the tiles' scale; a calibration input; robust solving that drops worst-first and takes back correct matches, including links that rejoin groups of tiles whose wrong links outnumbered their right ones; overlap-only correlation (2.8x faster). Multi-band blending and per-tile rectification were not needed: seams already pass, and switching the twin's flyback distortion off changed nothing.

**G2 is met at the overlap the Mapper's warning recommends** (`benchmarks/results/mosaic_twin_overlap57.md`, the same 20 x 20 map at 0.57 overlap): placement 0.07 / 0.17 / 0.30 px, no wrong or unplaced tile, seams 1.01, brightness 0.34 %. Stitching then took 950 s (more overlap, larger correlations); with wide searches made coarse to fine (half resolution, then the full-resolution correlation at each whole pixel within 3 px; pyTEMlib 612a94e) it takes 422 s with the same placement.

At the 30 % overlap (the result above):

*   **Three of 400 tiles are unplaced.** At the turn from the first row to the second the stage landed 600 nm from where it reported, outside the search window a 30 % overlap allows (0.6 of the overlap, 432 nm). They are listed and drawn only where no placed tile is, never guessed. The Mapper's estimate now warns when a plan's overlap is smaller than the stage can put neighbours apart (`Mapper._overlap_warning`) and says what overlap would do; re-imaging such a tile is the other remedy (follow-up).

How the placement result was reached (for whoever changes the stitcher next): an earlier 0.75 px "median" was the measurement, not the stitcher: three misplaced tiles tilted the affine fit and every other tile's error with it. The misplaced tiles came from wrong links outvoting right ones (`_tile_vote`, `_consensus_links`). Sub-pixel refinement by upsampled correlation, a peak-ambiguity test, and cubic instead of linear resampling in the twin were each measured and dropped: none improved placement, two broke accuracy tests. The twin's flyback compression is the main source of single-match error (95th percentile 0.48 px with it, 0.27 without) but does not move the placement result.

### Cross-scale placement (P3)

`register_child(child, parent_tiles, predicted, ...)` in pyTEMlib returns a placement record or `unplaced`:

```text
placement = {
  "child": <key>, "child_id": <record id>,
  "parent_tile": {"run": <key>, "index": <int>, "map_version": <key>},
  "transform": {"scale": s, "rotation_rad": r, "translation_px": [dy, dx]},   # child px → parent tile px
  "uncertainty_px": [sy, sx], "score": <NCC>, "ambiguity": <second peak / first peak>,
  "method": "simulated-parent-ncc", "method_version": <pyTEMlib version>,
  "inputs_sha256": <hash of child and parent tile bytes>,
  "status": "placed" | "unplaced" | "stale"
}
```

The method: simulate what the parent would have seen of the child (downsample the child with the parent's pixel size and a probe-size blur, not a plain average), then search a window of the parent tile set by the parent's own placement uncertainty plus the stage error, by normalised cross-correlation with a sub-pixel peak. A placement is accepted only if the score passes a threshold and the second-best peak is well below the best (ambiguity ≤ 0.8); otherwise it is `unplaced`. Atomic tiers are always guided by their parent, never by stage position, which removes the wrong-lattice-repeat failure. Placements go into an append-only log per run (`<run stem>_placements`), and `stale` is written when a newer map version's tile differs from the one a placement was made against.

Navigation: the skill moves the stage to the parent-predicted position, acquires, places, and if the target is more than a quarter field off centre, shifts the beam or stage by the measured offset and acquires again.

#### What the benchmark found, and what was built (2026-10-30)

`benchmarks/multiscale_twin.py` (results in `benchmarks/results/multiscale_twin.md`): one 2.4 µm survey tile (4.7 nm pixels), 50 children at 16x (0.29 nm pixels) aimed by image shift, and in each a grandchild at 8x more (0.037 nm pixels, atoms drawn); each placed by `pyTEMlib.image_tools.register.register_child` from the commanded shift alone and scored against where the twin's frames really landed.

| Step | Placed | Error, parent pixels (median / 95th) | Wrong |
| :-- | :-- | :-- | :-- |
| survey -> tier 2 | 48 of 50 (96 %) | 0.23 / 0.49 | 0 |
| tier 2 -> tier 3 (atomic) | 45 of 50 (90 %) | 0.14 / 0.27 | 0 |
| tier 3 in the survey through both | 43 | 0.23 / 0.50 survey pixels (allowance 0.53) | — |

Every G3 check passes. What made it pass, and what was tried:

*   Survey images are faint (a few counts a pixel): matching raw images left most children with correlations of 0.2 to 0.4 and unplaced. Both images are smoothed by about a parent pixel to decide whether there is a clear match, and the sub-pixel position is taken from lightly smoothed ones near that peak (smoothing enough to judge a match blunts the peak, 95th percentile 1.3 px at 1.5 px smoothing).
*   A "significance" test (peak against the spread of the window's scores) was tried and dropped: inside a small, smooth window it rejects right matches.
*   The unplaced children are ones another place matches almost as well (film with too little in it at that scale), refused rather than guessed.
*   Children were aimed by image shift, not the stage: the stage's error (hundreds of nanometres) is larger than a tier-2 field. Re-centring through tiers by image (design 4.3) is how the real instrument would do it; the twin's image shift is exact, the instrument's will need its own calibration.
*   Version 1 measures translation only; scale and rotation between tiers are given by the pixel sizes and scan rotation. Fitting them is a follow-up when a calibration says they differ.

Built for agents: the analysis server's `place_image` (places a finer image on a stitched map or a coarser image, from the stage position it records or the point it was aimed at by image shift, within `window_m`) and `list_placements`, which keep every result, placed or unplaced with the reason, in an append-only JSON-lines log (`--placements`, pyTEMlib 958a6c6). `place_image` is in the agent tool set, and the `multiscale_region` skill walks an agent through the tiers: aim by image shift, place, use the placed position (never the aimed one) for the next step, stop at an unplaced image or the dose cap.

### P4 so far (2026-10-30)

The first step is built in SciAgentGUI (`MapView.tsx`, `tiledClient.ts`, commit a50d98a on claude_dev): the Maps page fetches the stored chunks a view touches through Tiled's block endpoint, nearest the centre first and six at a time, keeps 160 of them by map, level and chunk, draws every cached coarser chunk under the finer ones, and paints every chunk with the map's one display range (from its coarsest level), so chunks meet without brightness steps and a zoom never shows a stretched stale picture. Checked in the running app on a 2,000-pixel, 4-level twin map. The Maps page also draws the finer images placed on a map (`list_placements`): outlined once 6 px on screen, read and drawn at their placed position once 48 px (SciAgentGUI 230813d; checked in the running app with an image `place_image` placed on a twin map; of 15 older images whose stage readings fell inside a map, 14 were refused as ambiguous or too faint and none was guessed). Not built: the timing harness the "done when" numbers need, a finer image's own placed children (one tier deep only), and OpenSeadragon.

### What P4 to P7 require of this

*   P4 (viewer) needs every tier stored as chunked arrays readable by block, placements as similarity transforms in metres, and unplaced images listed: all provided above.
*   P5 (2 mm) needs the twin's sample to be generated lazily from (seed, cell); P1's support texture and sparse dose map are written as functions of position so they need no change.
*   P6 (transfer) needs the hardware driver to record what the twin records; the record fields already exist, and the placement chain never relies on the twin's ground truth.
*   P7 (sealing) needs stable, never-overwritten content: P0's versioned maps and append-only layers and logs provide it, so sealing needs no migration.

## Alternatives Considered

### Alternative A: Keep deleting old maps, re-point dependents on re-stitch

*   **Pros**: less storage; one map per run is simpler to list.
*   **Cons**: every re-stitch must find and rewrite every placement, label and point; a crash halfway leaves dependents pointing at nothing; nothing cited can be reproduced.
*   **Reason for Rejection**: maps are the thing users cite. Storage is cheap next to a lost map; a level pyramid of a 10,000-tile map is about the size of the run itself.

### Alternative B: Build a new twin for multiscale work

*   **Pros**: a free hand on the sample model.
*   **Cons**: the existing `procedural` and `CrystalSpecimen` samples already draw one seeded sample at both scales, with distortion, stage error, dose and ground truth; a second twin would duplicate all of it and split the tests.
*   **Reason for Rejection**: review round three found the existing twin already spans scales; its gaps are specific and smaller than a rewrite.

### Alternative C: Free affine transform per tile

*   **Pros**: absorbs any distortion.
*   **Cons**: six unknowns per tile instead of two, poorly constrained by thin overlaps; OpenSeadragon and OME-Zarr cannot draw it; it hides distortion instead of measuring it.
*   **Reason for Rejection**: distortion shared by all tiles is calibrated once and removed; only distortion inside a frame needs a per-tile term, and only at fine tiers.

### Alternative D: Place fine images by stage reading

*   **Pros**: no registration step; works on blank areas.
*   **Cons**: the stage is 150 to 350 nm off against 10 to 50 nm fields: the child would land several fields away.
*   **Reason for Rejection**: physically impossible to meet G3 this way.

### Alternative E: Area-averaged survey pixels in the twin

*   **Pros**: registration on the twin is easier; small particles are visible at survey scale.
*   **Cons**: the real instrument point-samples; stitching numbers measured on such a twin would not transfer.
*   **Reason for Rejection**: the twin exists to predict the instrument; one kinder than the instrument defeats its purpose.

### Alternative F: Store maps and tiers in OME-Zarr now

*   **Pros**: a community format with coming affine transforms.
*   **Cons**: a second storage path beside Tiled's HDF5 and arrays; no reader in this deliverable; the transforms specification is not final.
*   **Reason for Rejection**: placements are shaped like OME-NGFF transforms so writing OME-Zarr later is a conversion, not a redesign.

--------------------------------------------------------------------------------

## Non-Functional Requirements

*   **Security**: ground truth stays operator-only. Agent endpoints must block at least what the twin's agent endpoint blocks, checked by a test over every agent configuration in `configs/`. No analysis tool deletes data.
*   **Performance**: placement of one child (512² against a parent window) under 1 s; gain and offset solve linear in the number of overlap pairs; the dose map's memory grows with the number of frames, not the sample's area.
*   **Storage**: each re-stitch adds a full map version. The run file is the large part; a map version is at most the run's size plus a third for its pyramid. A cleanup tool for map versions is out of scope; it would be operator-only and refuse a version that any placement names.
*   **Rollout and compatibility**: maps written before P0 keep their keys and their in-node points; readers handle both layouts. Twin changes are behind specimen options with defaults that leave `procedural` seed 12345 pixel-identical, except the stage uncertainty, which is a configuration value.

--------------------------------------------------------------------------------

### Additional Resources

*   `concepts_brief.md` (this repository's root): the reviewed brief.
*   `docs/designs/agentic_sample_mapping.md`: the mapping design this extends.
*   `docs/Operation/large_maps.md`: the existing stitching benchmark.
*   `../pyTEMlib-mcp/pyTEMlib/image_tools/mosaic.py`, `../pyTEMlib-mcp/pyTEMlib/mcp_server/map_store.py`.

### Future Improvements

*   **2 mm grid with lazy generation (P5)**: grid bars and holes, film, clusters, particles, lattice, each generated from (seed, cell).
*   **Deep-zoom viewer (P4)**: tiles and cache on today's Maps page, then tiers layered in one view.
*   **Transfer (P6)**: hardware records matching the twin's; the point-measurement skill unchanged on the Spectra; a real survey as stitching test data.
*   **Sealing (P7)**: started. `save_acquisition` seals every file it writes; a sealed file whose bytes changed is refused a new seal; `startup_scripts/verify_integrity.py` reports changed or missing files and breaks in the manifest's chain. Still to do: seal stitched map versions and placement logs on the analysis server, chain the dose ledger, and check seals where data is read.

--------------------------------------------------------------------------------

## Detailed Implementation Plan

Paths are relative to the asyncroscopy repository; `pyTEMlib/` paths are in the `../pyTEMlib-mcp` worktree (branch `feature/mcp-mapping`).

### P0. Guardrails

#### 1. `configs/mcp.yaml` (Modify)

*   **Type / Scope**: agent endpoint configuration for the Spectra 300.
*   **Proposed Changes**: add `endpoint_role: agent`; copy the twin's `blocked_functions` entries for `State`, `Status`, `DATA`, `Approvals`, `Mapper` and `Tuner`; add `trace: outputs/trace/mcp_agent.jsonl` and `clock_device: ""` (real time on hardware); add `tool_log: ""` and `include_only_functions: []` so the keys match the twin's.

#### 2. `tests/test_agent_configs.py` (Create)

*   **Type / Scope**: unit test.
*   **Proposed Changes**: load every `configs/mcp*.yaml` whose role is `agent` (missing role means agent). For each, assert its `blocked_classes` and every `blocked_functions` entry include the corresponding entries of `configs/mcp_dt.yaml`, and that `trace` is non-empty. A config may block more, never less.

#### 3. `pyTEMlib/mcp_server/map_store.py` (Modify)

*   **Type / Scope**: map storage.
*   **Proposed Changes**:
    *   `map_writer(run)`: always a new versioned key or local file; `_TiledMapWriter.swap` and `_LocalMapWriter.swap` no longer delete or replace the previous version (local: rename the temporary file to the new versioned name).
    *   `current_map(run)`: local runs glob `<stem>_map*.h5` (excluding `.tmp`) and return the newest by name.
    *   `list_maps(run)` (new): every version of a run's map, newest first, with id, created time and `inputs_sha256`.
    *   `write_map`: adds `version` and `inputs_sha256` to the metadata; takes no points.
    *   Points: `points_key(run)` (new) names `<stem>_points` (Tiled) or `<stem>_points.h5` (local). `add_pois`, `list_pois`, `update_pois`, `poi_counts`, `poi_kinds` take the run's point container. `update_pois` keeps the old layer and appends `{layer, replaced_at}` to the container's `layers` metadata. `_points_for_map(map_key)` reads `run` from map metadata, and for a pre-P0 map with in-node points and no run container yet, returns the in-node points (read-only).
    *   `_delete_node` stays only for `abandon`.

#### 4. `pyTEMlib/mcp_server/map_tools.py` (Modify)

*   **Type / Scope**: MCP tools.
*   **Proposed Changes**: `_assemble` no longer carries old points into the new map (they stay in the run's container) and still adds EDS points from the run, deduplicated by id. `score_map` and `add_poi` write through the run's point container. `map_info` reports `version`, `inputs_sha256`, and the other versions of the run (count and newest three). New read-only tool `list_maps(run)`.

#### 5. `pyTEMlib/tests/test_map_versions.py` (Create)

*   **Type / Scope**: unit tests against local HDF5 (and Tiled when `TILED_TEST_URI` is set, otherwise that half skips and says so).
*   **Proposed Changes**: re-stitch keeps the first map readable; `current_map` returns the newer; points added before the re-stitch are listed on the new map; re-scoring a tag keeps the old layer and lists it in `layers`; a pre-P0 map with in-node points still lists them; an aborted assemble leaves no half map and the previous map intact; no registered MCP tool name contains `delete` or `remove`.

#### 6. `asyncroscopy/data/integrity.py` (Modify) and `asyncroscopy/mcp/tool_log.py` (Modify)

*   **Type / Scope**: correctness of an unwired module.
*   **Proposed Changes**: rename `_file_lock` to `file_lock` in `tool_log.py` (its callers updated) so `integrity.py` imports; correct the module description to say sealing is not yet called by `save_acquisition` or the map writer (it is wired in P7). Add `tests/test_integrity.py` covering seal, check of a changed file, a removed line and a moved line in the chain, so the module is correct before it is wired.

#### 7. `pyproject.toml` (Modify)

*   **Type / Scope**: dependency pin.
*   **Proposed Changes**: bump the pytemlib git rev to the P0 commit.

#### 8. `docs/Operation/large_maps.md` (Modify)

*   **Proposed Changes**: a "Map versions and points" section: versioned keys, `list_maps`, points per run, nothing deleted.

### P1. Twin gaps at the µm scale

#### 9. `asyncroscopy/instruments/electron_microscope/twin/sample_model.py` (Modify)

*   **Proposed Changes**: `SampleParams.support_texture` (relative amplitude, default 0 so seed 12345 stays pixel-identical; the mapping configs set 0.01) and `support_texture_scale_m`. The texture is a sum of seeded cosines over a fixed set of wave vectors, evaluated at each pixel's centre, added to the support thickness in both `_weight_survey` and `_weight_atomic`. `ground_truth` adds each particle's lattice orientation.

#### 10. `asyncroscopy/instruments/electron_microscope/twin/render.py` (Modify)

*   **Proposed Changes**: atoms drawn at sub-pixel positions (Gaussian evaluated at the true offset within a local window) behind `subpixel_atoms` (default false for pixel-identity; mapping configs set true).

#### 11. `asyncroscopy/instruments/electron_microscope/twin/detector.py` (Modify)

*   **Proposed Changes**: `gain_drift` (fraction, default 0) and `offset_drift` (fraction of full scale, default 0), a seeded random walk in twin time, applied to each frame and to each spectrum's counts; the frame's values recorded for ground truth.

#### 12. `asyncroscopy/instruments/electron_microscope/twin/dose.py` (Modify)

*   **Proposed Changes**: `DoseMap` becomes sparse: a dict of cells keyed by (level, i, j), cell size chosen per frame as the largest power-of-two fraction of a base cell not larger than the frame's field / 8. `dose_at` sums over levels. `summary` unchanged in shape.

#### 13. `asyncroscopy/instruments/electron_microscope/digital_twin.py` (Modify)

*   **Proposed Changes**: ground truth records per acquisition: distortion parameters, line-jitter seed and offsets, gain and offset. `export_ground_truth` stays operator-only.

#### 14. `configs/digital_twin_mapping.yaml`, `configs/digital_twin_large_map.yaml` (Modify)

*   **Proposed Changes**: turn on support texture, sub-pixel atoms, gain drift and scan rotation; set the Mapper's `stage_sigma_m` to the stage model's real error.

#### 15. Tests (Create/Modify)

*   `tests/test_twin_consistency.py`: the 0.2 nm switch agreement; same seed same pixels with the new options on; seed 12345 unchanged with them off; texture visible (non-zero gradient) at 200 nm and 0.2 nm pixels; sub-pixel atoms: the same column in two overlapping frames lands within 0.1 px after the known shift.
*   `tests/test_twin_dose.py`: a 50 nm frame's dose stays within its field to within 10 %; dose map memory grows with frames.
*   `tests/test_mapper_config.py`: every twin configuration's `stage_sigma_m` is at least the stage model's error.

### P2. Stitching measured on the twin

#### 16. `benchmarks/mosaic_twin.py` (Create)

*   **Proposed Changes**: drives a running twin stack (agent and operator endpoints, analysis server): a 20 × 20 map, `assemble_map`, then reads ground truth through the operator endpoint and the map through Tiled. Scores: placement error on the drawn canvas after removing the common offset (median, 95th, worst), seam step ratio (mean absolute gradient across tile edges ÷ inside tiles), neighbour brightness difference, unplaced rate. Writes `benchmarks/results/mosaic_twin.md`; non-zero exit on a failed threshold.

#### 17. `pyTEMlib/image_tools/mosaic.py` (Modify, as the benchmark requires)

*   **Proposed Changes**: `assemble` resamples each tile at its sub-pixel position (Lanczos-3); `solve_gain_offset(pairs, tiles)` (new) and its use in `assemble`; unplaced tiles skipped in assembly and returned in `Placement.unplaced`; `rectify(tile, distortion)` (new) and a global distortion estimate from overlap residuals. Each lands only after the benchmark shows its error, with its own test in `pyTEMlib/tests/test_mosaic.py`.

#### 18. `pyTEMlib/mcp_server/map_tools.py` (Modify)

*   **Proposed Changes**: `map_info` lists unplaced tiles (index, match count, residual) paged; map metadata records the gain and offset solve.

### P3. Three tiers on the twin

#### 19. `pyTEMlib/image_tools/register.py` (Create)

*   **Proposed Changes**: `simulate_parent_view(child, child_px, parent_px, probe_px)`, `register_child(child, child_px, parent_tile, parent_px, predicted_px, window_px)` returning the placement record or `unplaced`, with the ambiguity test.

#### 20. `pyTEMlib/mcp_server/map_tools.py` (Modify)

*   **Proposed Changes**: tools `place_image(image, run, map, predicted_m)` (an analysis job, writes the placement to the run's placement log) and `list_placements(run, status, limit)`.

#### 21. `asyncroscopy/skills/multiscale_region/SKILL.md` (Create)

*   **Proposed Changes**: the procedure of CUJ 3 in plain text: pick a scored region, acquire, `place_image`, re-centre, step down a tier, stop at the dose cap; uses only tools the agent endpoint lists.

#### 22. `benchmarks/multiscale_twin.py` (Create)

*   **Proposed Changes**: 50 children per step over three tiers; scores G3 against ground truth; writes `benchmarks/results/multiscale_twin.md`.

### Verification Plan

*   **Unit tests**: asyncroscopy `uv run --no-sync pytest -q` (one process at a time); pyTEMlib `.venv/Scripts/python.exe -m pytest pyTEMlib/tests -q` from the worktree.
*   **Live**: the twin stack (`configs/digital_twin_mapping.yaml`, endpoints :8000/:8001, analysis :8003, Tiled :9091); CUJ 1 by hand through the agent endpoint; `benchmarks/mosaic_twin.py` (P2) and `benchmarks/multiscale_twin.py` (P3) as the done-when checks.
*   **Review**: each phase's diff goes to adversarial review agents before commit; findings are fixed or answered.

--------------------------------------------------------------------------------

*Drafted with EGM*
