---
name: multiscale_region
description: Zoom into one region of a map tier by tier, down to atomic columns, placing each finer image on the coarser one by matching it (never by the stage), and stop at the dose cap.
version: 1.2.0
min_asyncroscopy: 0.1.0
keywords: [zoom in, magnification steps, tiers, atomic resolution, atomic columns, multiscale, finer image, closer and closer]
requires: [place_image, "*_acquire_scanned_image", "*_aim_image_shift"]
acts: true
tools: [place_image, list_placements, find_features, view_data, to_stage, "*_unblank_beam", "*_blank_beam", "*_get_stage", "*_move_stage", "*_set_fov", "*_get_fov", "*_aim_image_shift", "*_set_image_shift", "*_get_image_shift", "*_acquire_scanned_image", "*_get_sample"]
---

# Zoom into a region, tier by tier

Each step is a finer image inside the one before. The stage cannot land a 75 nm or 10 nm field where it was asked (its
error is hundreds of nanometres), so a finer image is aimed by image shift and then placed on the coarser image by
`place_image`, which matches the two. Do not move the stage or take a new survey: start from the image you were given.

1. `<Microscope>_unblank_beam` (a blanked beam gives images of noise). The parent is the image you were given.
2. Find what to aim at: `find_features` with `{"image": <parent>, "min_size_m": <next fov / 4>, "max_size_m":
   <next fov x 2>}` (next fov from the table). Take the first feature's `x_m` and `y_m`. No features: the parent shows
   nothing distinct; say so and stop.
3. Aim there: `<Microscope>_aim_image_shift` with `{"position": [x_m, y_m]}` (the stage does not move; it works out the
   shift itself). Refused beyond about 1 um from the stage: say so and stop.
4. `<Microscope>_set_fov` with the next field of view, then `<Microscope>_acquire_scanned_image`
   `{"detector_list": ["haadf"]}`: the new image's key.

   | parent fov_m | next fov_m | find_features min_size_m, max_size_m | place_image window_m |
   | :-- | :-- | :-- | :-- |
   | 6e-7 (600 nm) | 7.5e-8 | 1.9e-8, 1.5e-7 | 9.4e-9 |
   | 7.5e-8 | 9.4e-9 | 2.3e-9, 1.9e-8 | 1.2e-9 |
   | other | parent / 8 | next / 4, next x 2 | parent / 64 |

5. `place_image` with `{"image": <new key>, "parent": <parent>, "centre_xy_m": [x_m, y_m], "window_m": <from the
   table>}`.
   - `"status": "placed"`: the new image is the next parent; go back to step 2 with it.
   - `"status": "unplaced"`: report its `reason` and stop going finer (an image inside a crystal often matches its
     parent in more than one place, one lattice spacing apart: that is not a failure of the run, say it).
6. Stop when the fov is about 1e-8 m (atomic columns show at pixels under 1e-10 m). Then
   `<Microscope>_set_image_shift` `{"shift": [0, 0]}` and `<Microscope>_blank_beam`.

Dose: each finer image puts far more dose on its small area (about 7,800 e/A^2 per frame at 20 pm pixels on a
typical setting). Stop at the request's or the instrument's dose cap; never re-image the coarser tiers after the finer
ones (the finer ones damage and contaminate them).

Report every image's key, its field of view, and its placement (`centre_xy_m`, `score`) or why it was unplaced, from
the coarsest image to the finest. `list_placements` with `{"parent": <key>}` shows what has been placed on an image.
