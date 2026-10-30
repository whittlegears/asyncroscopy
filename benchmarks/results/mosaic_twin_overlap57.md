# Stitching on the twin

`benchmarks/mosaic_twin.py`, asyncroscopy 4d473c7, 2026-10-30.

400 tiles (20 x 20, serpentine) of 512 pixels, field 2400 nm
(4.69 nm pixels), 57 % overlap, dwell 1.0 us, from
`configs\digital_twin_large_map.yaml` (a 36 um sample of 20000 particles; its film texture
carries registration between particles), twin settings changed: none, scan calibration
from: this map; stitched with stage_sigma_m 290 nm. Detector gain over the run:
[1.0000207822019005, 1.0474225843029834]. Taking the tiles took 29.0 s, stitching 950.5 s;
2431 overlap matches were used.

## Placement against the twin's truth (pixels: median / 95th percentile / worst)

| Positions | After removing the common offset (all tiles) | After removing the best affine map (tiles within 2 px of it) |
| :-- | :-- | :-- |
| stage only | 39.53 / 78.45 / 127.66 | 39.48 / 78.08 / 122.13 |
| stitched | 3.37 / 8.21 / 9.88 | 0.07 / 0.17 / 0.30 |

"Stitched" is where the tiles are drawn on the map: pyTEMlib resamples each tile at its sub-pixel position. The
scan distortion the stitcher removed: [[0.9907828825390923, -0.00582484119135543], [-0.0013846358194101467, 0.9938124817757984]]
(measured [[0.9907828825390923, -0.00582484119135543], [-0.0013846358194101467, 0.9938124817757984]]); tile gains it applied range over [0.9809353047356452, 1.01978824266129].
The difference between the two columns is the shared scan distortion (rotation, shear, field-of-view scale)
bending the whole map; what is left after the affine fit is tiles disagreeing with each other. Placement is judged
on the second column (the shared distortion is a calibration, measured on a large map or given): tiles further than
2 px from the affine fit are misplaced, and counted, not averaged in. Misplaced or unplaced: [].

## Seams and brightness

* Step across drawn tile edges / step inside tiles: 1.012 (6662092 edge pixels).
* Brightness difference between neighbouring tiles over their shared area: mean 0.34 %,
  95th percentile 1.19 % (3832 pairs).
* Tiles with no usable match (drawn from the stage's guess): 0.
* Tiles placed by a match but more than 2 px from where they belong (wrong placements): [].

## Against the thresholds (design goal G2)

| Check | Threshold | Result |
| :-- | :-- | :-- |
| placement_median_px | 0.3 | pass |
| placement_p95_px | 0.7 | pass |
| placement_max_px | 2.0 | pass |
| misplaced_tiles | 0 | pass |
| seam_ratio | 1.2 | pass |
| neighbour_brightness_pct | 1.0 | pass |
| unplaced_tiles | 0 | pass |
