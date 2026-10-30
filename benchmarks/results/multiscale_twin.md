# Multiscale placement on the twin

`benchmarks/multiscale_twin.py`, asyncroscopy bcef20d, 2026-10-30, from `C:\Users\leafe\Documents\Github\asyncroscopy\configs\digital_twin_large_map.yaml`.

Tiers: 2400.00 nm field (4.6875 nm pixels), 150.00 nm field (0.2930 nm pixels), 18.75 nm field (0.0366 nm pixels);
steps 16.0x then 8.0x; 512-pixel images; each child predicted from the commanded image
shift and searched within 8.0 parent pixels. 31.9 s in all.

| Step | Children | Placed | Error, parent pixels: median / 95th | Wrong (> 2.0 px) | Why some were not placed |
| :-- | :-- | :-- | :-- | :-- | :-- |
| survey -> tier 2 | 50 | 48 (96 %) | 0.232 / 0.492 | 0 | another place matches almost as well (0.91 of the best); the best match scores 0.49, below 0.5 |
| tier 2 -> tier 3 (atomic) | 50 | 45 (90 %) | 0.144 / 0.273 | 0 | another place matches almost as well (0.81 of the best); another place matches almost as well (0.93 of the best); another place matches almost as well (0.98 of the best); another place matches almost as well (0.99 of the best) |

A grandchild's centre placed in the survey through both placements, against the truth, in survey pixels:
median 0.2343229967170586, 95th percentile 0.5020225638185846 (43 chained). An image's absolute position is only as good as
the coarsest placement in its chain: 0.2 survey pixels here is about 1 nm, many atomic pixels.

| Check (design goal G3) | Result |
| :-- | :-- |
| tier2_p95_parent_px | pass |
| tier3_p95_parent_px | pass |
| tier2_placed_share | pass |
| tier3_placed_share | pass |
| wrong_placements | pass |
| atomic_image_in_survey_after_two_placements | pass |
