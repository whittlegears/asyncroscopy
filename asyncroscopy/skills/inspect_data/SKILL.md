---
name: inspect_data
description: Find stored images, spectra and maps and read them as text or pictures; judge peaks, noise and what the data do and do not show.
version: 1.0.0
min_asyncroscopy: 0.1.0
keywords: [look at the data, look at the spectrum, look at the image, inspect, show me, what is in, analyse, analyze, peaks, elements, composition, latest image, latest spectrum, data]
requires: [view_data]
tools: [list_data, view_data, map_info, get_region, list_pois]
---

# Looking at data

Data are keys (file names in Tiled such as `stem_image_HAADF_<time>.h5`, `map_<id>.h5`, `map_<id>_map`). Never ask for raw arrays.

1. No key yet: `list_data` with `{"kind": "spectrum"}` (or `image`, `spectrum_image`, `stack`, `map`), newest first. Narrow with `search` (part of the key). If `more` is true, call again with `offset` = `next_offset`.
2. Read it: `view_data` with `{"key": ..., "format": "text"}`. A file with several arrays lists them; pick one with `dataset` (for a map run: `tiles/tile_00000`, `spectra/spectrum_00000`).
3. A picture: `{"format": "png"}`. The text part still gives the statistics and peaks.
4. Part of a big image: `region` `[row0, row1, col0, col1]` in pixels. Part of a spectrum: `region` `[min, max]` in eV. One position of a spectrum image: `index` `[row, col]`.
5. Stitched maps: `map_info` for size and stage box, `get_region` with a stage box `[x_min, x_max, y_min, y_max]` in metres to zoom in or out, `list_pois` for marked points.

## Judging spectra

- Peaks listed with candidate elements are significant (5 sigma over the background, summed over the line width).
- "Possible escape peak" or "possible sum peak" is a detector artefact, not an element.
- "No peaks detected above noise at this exposure" means the exposure was too short to tell. Say that; never conclude an element is absent from it.

## Report

Give the key you looked at, what it is (kind, shape, field of view or energy range) and what it shows, in a few lines.
