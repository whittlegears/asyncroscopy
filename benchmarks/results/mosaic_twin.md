# Stitching on the twin

`benchmarks/mosaic_twin.py`, asyncroscopy 9f98558, 2026-10-30.

400 tiles (20 x 20, serpentine) of 512 pixels, field 2400 nm
(4.69 nm pixels), 30 % overlap, dwell 1.0 us, from
`configs\digital_twin_large_map.yaml` (a 36 um sample of 20000 particles; its film texture
carries registration between particles), twin settings changed: none, scan calibration
from: this map; stitched with stage_sigma_m 290 nm. Detector gain over the run:
[1.0001079499085654, 1.0490517690261871]. Taking the tiles took 31.6 s, stitching 304.1 s;
1271 overlap matches were used.

## Placement against the twin's truth (pixels: median / 95th percentile / worst)

| Positions | After removing the common offset | After removing the best affine map |
| :-- | :-- | :-- |
| stage only | 39.48 / 78.38 / 127.64 | 39.66 / 77.96 / 122.74 |
| stitched | 3.53 / 8.93 / 131.91 | 0.75 / 1.94 / 133.81 |

"Stitched" is where the tiles are drawn on the map: pyTEMlib resamples each tile at its sub-pixel position. The
scan distortion the stitcher removed: [[0.9900996588193678, -0.00539800577685154], [-0.0010257501495202284, 0.9925582756144286]]
(measured [[0.9900996588193678, -0.00539800577685154], [-0.0010257501495202284, 0.9925582756144286]]); tile gains it applied range over [0.982824691660336, 1.0168495418164176].
The difference between the two columns is the shared scan distortion (rotation, shear, field-of-view scale)
bending the whole map; what is left after the affine fit is tiles disagreeing with each other.

## Seams and brightness

* Step across drawn tile edges / step inside tiles: 0.965 (10617182 edge pixels).
* Brightness difference between neighbouring tiles over their shared area: mean 0.28 %,
  95th percentile 0.91 % (1478 pairs).
* Tiles with no usable match (drawn from the stage's guess): 1.

## Against the thresholds (design goal G2)

| Check | Threshold | Result |
| :-- | :-- | :-- |
| placement_median_px | 0.3 | FAIL |
| placement_p95_px | 0.7 | FAIL |
| placement_max_px | 2.0 | FAIL |
| seam_ratio | 1.2 | pass |
| neighbour_brightness_pct | 1.0 | pass |
| unplaced_tiles | 0 | FAIL |
