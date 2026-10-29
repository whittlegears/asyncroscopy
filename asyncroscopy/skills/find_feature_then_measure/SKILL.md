---
name: find_feature_then_measure
description: Find a feature (a particle, a grain, a bright spot) in an image, then take a spectrum or a closer image on it and report where it is.
version: 1.0.0
min_asyncroscopy: 0.1.0
keywords: [find a particle, find a feature, find a grain, on it, on the particle, at the particle, locate, spot, cluster]
requires: ["*_acquire_scanned_image", view_data]
acts: true
tools: [view_data, "*_acquire_scanned_image", "*_acquire_spectrum", "*_place_beam", "*_move_stage", "*_get_stage", "*_set_fov"]
---

# Find a feature, then measure it

1. Image the area: `<Microscope>_acquire_scanned_image` (the current field of view unless the request gives one). It returns a data key.
2. Look at it: `view_data` with `{"key": <key>, "format": "text"}` gives the field of view and pixel size; with `"format": "png"` you see it. Pick the feature the request means (for example the brightest particle) and note its position in pixels (row, column).
3. Turn pixels into a position: the image's centre is the stage position (`<Microscope>_get_stage`); x = centre_x + (column - width/2) x pixel_size, y = centre_y + (row - height/2) x pixel_size (metres).
4. Measure there:
   - a spectrum: `<Microscope>_place_beam` at the feature (in the image's fractional coordinates, column/width and row/height), then `<Microscope>_acquire_spectrum`;
   - a closer image: `<Microscope>_move_stage` to the position, a smaller field of view with `<Microscope>_set_fov`, then `<Microscope>_acquire_scanned_image`.
5. Read the result with `view_data` (`"format": "text"`) before saying what it shows.

Report the feature's position (metres, and pixels in which image), the key of the measurement and what it shows. If nothing in the image matches the request, say so instead of measuring a random spot.
