---
name: drift-tracked-tilt-series
description: Use when the operator wants a tilt series (images or diffraction patterns at several stage alpha angles) of the same area. Keeps the area centred by measuring the drift after each tilt and correcting it with the image shift.
requires_tools: [measure_image_shift]
---
# Drift-tracked tilt series

No single procedure does this yet, so it is built from instrument tools plus the
pyTEMlib `measure_image_shift` tool. Instrument tool names start with the instrument's
class name, for example `DigitalTwinTilt_move_stage` or `SCAN_get_imsize`. Use the
ones the connected instrument actually offers.

Unlike the calibration procedures, nothing here undoes a failure automatically. You
must put the stage and image shift back yourself, as step 7 says.

## 1. Record the starting state

Call `get_stage` (gives [x, y, z, alpha, beta]; alpha in degrees), `get_image_shift`,
`get_fov` and `SCAN_get_imsize`. Pixel size = field of view / image size.
Ask the operator for the alpha range and step. Do not assume them.

## 2. Reference image

Acquire one image at the starting alpha and keep its data key as the reference.

## 3. Calibrate the correction direction (once per session)

Image-shift axes may point opposite to image axes, so measure the direction rather
than guess it:

1. Apply a known shift `d` (about 5 % of the field of view) in x only with
   `set_image_shift(start + [d, 0])`. Acquire an image.
2. Call `measure_image_shift(reference=<reference key>, moving=<new key>, pixel_size=<pixel size>)`.
   Its `shift_physical` is (row, column).
3. `sign_x = -d / column_shift`. It should be close to +1 or -1. Repeat for y with
   `row_shift` to get `sign_y`.
4. If either value is far from ±1 (say below 0.7 or above 1.3 in size), the scan is
   rotated relative to the image shift. Stop and tell the operator; this skill does
   not handle rotation.
5. Return to the starting image shift.

## 4. For each alpha

1. `move_stage` with only alpha changed.
2. Acquire a tracking image; call `measure_image_shift` against the reference.
3. New image shift = current + [sign_x × column_shift, sign_y × row_shift]; call
   `set_image_shift`.
4. Acquire the data the operator asked for (image, or camera image for diffraction).
   Record alpha, the data key and the measured shift.
5. If a measured shift exceeds a quarter of the field of view, the area has moved
   too far to track. Stop and report.

## 5. Large tilts

Drift grows with tilt. Step outwards from 0 degrees and correct at every step, rather
than jumping straight to the largest angle.

## 6. Report

Give a table of alpha, data key and correction applied, and whether any step was
stopped. The keys can be passed to `align_image_series` or other analysis tools.

## 7. Always finish by restoring

Whether the series finished or failed, call `move_stage` with the starting position
and `set_image_shift` with the starting shift. Confirm both with `get_stage` and
`get_image_shift`.
