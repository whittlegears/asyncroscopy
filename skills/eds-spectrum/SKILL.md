---
name: eds-spectrum
description: Acquire an EDS spectrum and report the elemental composition from the saved data.
version: 1.0.0
tags: [spectroscopy, eds, composition, acquisition]
triggers: ["eds", "spectrum", "composition", "which elements"]
tools: ["*_acquire_spectrum", "get_data_from_key"]
---

# EDS spectrum

1. Call `<DeviceClass>_acquire_spectrum` with `{"detector_name": "eds"}`.
   The exposure time comes from the EDS detector device (`exposure_time`).
2. The tool returns a Tiled key such as `spectrum_eds_20260918T101530.h5`
   (plus a plot preview on clients that render images).
3. Read it back with `get_data_from_key` (`{"key": "<key>", "max_values": 64}`).
   The result has one dataset named `spectrum`:
   - `shape`: `[N]` channels or elements
   - `attrs.elements`: a JSON list of element symbols when the instrument
     provides per-element values (the DigitalTwin does)
   - `preview`: the first values of the spectrum
4. When `attrs.elements` is present and has the same length as `preview`, pair
   them up: the DigitalTwin returns normalised fractions (they sum to about 1),
   so "Si 0.61, O 0.39" is a composition, not counts.
5. When there are no element labels, describe the spectrum by channel count and
   the strongest channels; do not invent element assignments.

## Pitfalls

- Acquire the image first if the user wants to know where the spectrum came
  from; the beam position is whatever the last scan or `place_beam` left it at.
- The EDS detector must exist on the instrument. If the tool reports "No
  spectrum detector named 'eds'", list the devices and stop.
