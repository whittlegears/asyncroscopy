---
name: find_feature_then_measure
description: Find a feature (a particle, a grain, a bright spot) in an image, then take a spectrum or a closer image on it and report where it is.
version: 1.3.0
min_asyncroscopy: 0.1.0
keywords: [find a particle, find a feature, find a grain, on it, on the particle, at the particle, locate, spot, cluster]
requires: ["*_acquire_scanned_image", view_data]
acts: true
tools: [view_data, find_features, to_stage, "*_get_fov", "*_unblank_beam", "*_blank_beam", "*_get_sample", "*_acquire_scanned_image", "*_acquire_spectrum", "*_place_beam", "*_move_stage", "*_get_stage", "*_set_fov"]
---

# Find a feature, then measure it

1. Image the area: `<Microscope>_acquire_scanned_image`. It returns a data key. The field of view must be several times the
   feature: for particles, `<Microscope>_set_fov` to 3e-7 (300 nm) first unless the request gives one or `<Microscope>_get_fov`
   is already 2e-7 to 6e-7. A particle cannot be found in a field narrower than about three times its size.
2. Look at it: `view_data` with `{"key": <key>, "format": "text"}` gives the field of view and pixel size; with `"format": "png"` you see it. Pick the feature the request means (for example the brightest particle) and note its position in pixels (row, column).
3. Turn pixels into a position with `to_stage` (`{"dataset": <key>, "row": ..., "col": ...}`): it uses the stage position recorded with that image, so do not compute it yourself. When its reply has a `sample_load_id`, check `<Stage>_get_sample` gives the same one before moving: a different one means the sample was reloaded since that image, so the position is stale; take a new image instead. If `to_stage` is not offered: the image's centre is where the stage was when it was taken; x = centre_x + (column - (width - 1)/2) x pixel_size, y = centre_y + (row - (height - 1)/2) x pixel_size (metres).
4. Measure there:
   - a spectrum, in this order: `<Microscope>_unblank_beam`; `<Microscope>_place_beam` with the feature's `place_beam` arguments from
     `find_features` copied as it is (not [0.5, 0.5], which is the image centre); `<Microscope>_acquire_spectrum`;
     `<Microscope>_blank_beam`. Without the beam unblanked first the spectrum has 0 counts;
   - a closer image: `<Microscope>_move_stage` to the position, a smaller field of view with `<Microscope>_set_fov`, then `<Microscope>_acquire_scanned_image`.
5. Read the result with `view_data` (`"format": "text"`) before saying what it shows.

If `find_features` or the image shows nothing, widen the field of view once (as in step 1) and look again; if it
still shows nothing, say so. Do not take spectra at random spots.

Report the feature's position (metres, and pixels in which image), the key of the measurement and what it shows. If nothing in the image matches the request, say so instead of measuring a random spot.
