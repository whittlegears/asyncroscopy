---
name: tiled-data-access
description: Inspect saved acquisitions in Tiled with get_data_from_key and understand the returned metadata.
version: 1.0.0
tags: [data, tiled, hdf5, metadata]
triggers: ["what was saved", "show me the data", "data key", "tiled"]
tools: ["get_data_from_key"]
---

# Reading acquired data

Every acquisition tool returns a key (a file name such as
`stem_image_HAADF_<timestamp>.h5`). Data lives on the Tiled server that the
DATA device points at; you never read files directly.

1. Call `get_data_from_key` with `{"key": "<key>", "max_values": 64}`.
   `max_values` caps how many numbers are returned per dataset (default 64).
2. The result is a dict:
   - `key`, `uri`, `format`
   - `attrs`: top-level HDF5 attributes (acquisition metadata such as detector,
     dwell time, field of view; non-scalar values are JSON strings)
   - `datasets`: a list of `{name, shape, dtype, attrs, preview}` entries.
     Images are named `image/<DETECTOR>` (for example `image/HAADF`);
     spectra are named `spectrum`.
3. Use `shape` to describe the data (a 2D shape is an image, 1D is a spectrum)
   and `preview` for a quick sanity check (all zeros usually means a blanked
   beam or an empty acquisition).
4. Raise `max_values` (for example 4096) only when the user needs the numbers;
   for plots, tell the user the key so they can load it with the Tiled client:

   ```python
   from tiled.client import from_uri
   client = from_uri("http://localhost:9091")
   image = client[key]["image"]["HAADF"].read()
   ```

## Pitfalls

- A `FileNotFoundError` means the key is not registered yet or is misspelled;
  re-check the exact key returned by the acquisition tool.
- Do not guess a key from a pattern; only use keys returned by tools.
