---
name: haadf-imaging
description: Acquire a scanned HAADF (STEM) image and confirm it was saved to Tiled.
version: 1.0.0
tags: [imaging, stem, haadf, acquisition]
triggers: ["take an image", "acquire an image", "haadf", "scanned image", "stem image"]
tools: ["*_acquire_scanned_image", "get_data_from_key", "list_devices"]
---

# HAADF imaging

1. Find the instrument tool with `list_devices` if you do not know the device
   class. The acquisition tool is named `<DeviceClass>_acquire_scanned_image`
   (for example `DigitalTwin_acquire_scanned_image`).
2. Call it with `{"detector_list": ["haadf"]}`. Detector names are
   case-insensitive; the image is stored under `image/HAADF`.
   Other scanning detectors can be added to the list (`["haadf", "bf"]`).
3. The tool returns a Tiled data key such as `stem_image_HAADF_20260918T101500.h5`.
   Some clients also receive a PNG preview; the key is the important part.
4. Verify with `get_data_from_key` (`{"key": "<key>", "max_values": 16}`):
   expect one dataset named `image/HAADF` with a 2D `shape` (for example
   `[512, 512]`) and a numeric `dtype`.
5. Report the key and the image shape to the user. Never claim an image was
   acquired if the tool returned an error.

## Scan settings

Image size, dwell time and scan region come from the SCAN device
(`imsize`, `dwell_time`, `scan_region` attributes). Change them there before
acquiring if the user asks for a different resolution. Field of view is set on
the instrument with `<DeviceClass>_set_fov` (a float in the instrument's units)
and read back with `<DeviceClass>_get_fov`.

## Pitfalls

- A cold first acquisition can take several seconds; do not retry immediately
  on a timeout, check `get_data_from_key` first.
- An empty `detector_list` falls back to `["haadf"]`.
- `Init`, `Kill` and `RestartServer` are blocked; never try to call them.
