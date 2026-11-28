---
name: spectra_in_image
description: Take EDS spectra at points chosen in the current image (a bright area, a dark area, an interface, a particle) by parking the beam with place_beam, and say which elements each point shows.
version: 1.0.0
enabled: true
min_asyncroscopy: 0.1.0
keywords: [eds, spectrum, spectra, interface, phase, bright, dark, area, point, place beam, element, elements, composition, where is]
requires: ["*_acquire_spectrum", "*_place_beam", "*_acquire_scanned_image", view_data]
acts: true
tools: ["*_place_beam", "*_acquire_spectrum", "*_acquire_scanned_image", view_data]
---

# Spectra at points of the image

## 1. Where the beam goes

`<Microscope>_place_beam` takes `{"position": [x, y]}` as **fractions of the current image, each from 0 to 1**. It is not metres and not pixels.

- `x` runs from the left edge (0) to the right edge (1); `y` from the top (0) to the bottom (1). The centre is `[0.5, 0.5]`.
- A feature at column `c`, row `r` of an image `W` wide and `H` high is `[c / W, r / H]`. In a 1024 x 1024 image, row 550 at the middle column is `[0.5, 0.537]`.
- "Top right" is about `[0.8, 0.2]`, "bottom left" about `[0.2, 0.8]`.
- Never send stage coordinates (numbers like -0.00027) or a shift in nanometres: the instrument refuses anything outside 0 to 1.

The fractions refer to the image as it is now. Take an image first (`<Microscope>_acquire_scanned_image`), choose the points on that image, and do not move the stage or change the field of view before the spectra. If either changed, take a new image and choose again.

## 2. Choosing points

Look at the image itself. Pick each point well inside the feature it stands for, away from its edge, unless the edge is what was asked for. For "the interface", put the point on the boundary line between the two areas. For "different areas of the bright phase", pick separate bright features, not one feature three times.

The darkest areas of a HAADF image may be vacuum (a hole, or beyond the sample's edge). A spectrum there has very few counts and only the zero-energy peak: say "vacuum", not a phase.

Say in your answer where each spectrum was taken: the fractions and what is there ("[0.8, 0.2], the bright band at the top right").

## 3. Taking them

For each point: `place_beam`, then `<Microscope>_acquire_spectrum` with `{"detector_name": "eds"}`. The reply has the spectrum's key and its peaks.

The time a spectrum counts for is the EDS detector's exposure time, 1 s unless changed. When the user asks for another time, look for `EDS_set_eds_exposure_time`; if the endpoint does not offer it, say that the exposure time cannot be set from here and stop. Do not search for it again, and do not take the spectra at the old time as if it were the new one.

## 4. Reading a spectrum

Read energies, never channel numbers. The peaks table of the reply (or `view_data` with `{"key": ..., "format": "text"}`) lists each peak's energy in eV with the X-ray lines that fit it, and above it "elements most consistent with these peaks". Report elements from that table.

- The scale is in the data. Use channels only when a reply shows no energies. For example, on a detector with 5 eV channels that start at -250 eV, channel `n` is at `-250 + 5 * n` eV, so channel 347 is 1485 eV, the aluminium K line.
- The peak at about 0 to 20 eV is the detector's zero-energy noise peak. It is in every spectrum, the vacuum's too. It is not an element and not the support film.
- An element is present when its main line is in the table. Lines to expect (eV): C K 277, O K 525, Al K 1487, Si K 1740, Mn K 5899, Fe K 6404, Ni K 7478, Cu K 8048 (and Cu L 930), Ce L 4840 (with 5262), Zn K 8639.
- The "most consistent" line is a suggestion that prefers few elements. It can offer one rare heavy element (Yb, Eu, Re) in place of two common ones whose lines sit at the same energies (Yb for Al K at 1487 with Ni K at 7478; Eu for Mn K at 5899). Check the table: when a peak lists a common element's K line first, report that element, and mention the other reading only if the user asks.
- Copper in every spectrum, also in vacuum, usually comes from the grid or the holder, not the sample. Say so when you report it.
- "Not detected" is about this spectrum only: say the counting time and the total counts. With a 1 s spectrum of a few hundred counts a minor element cannot be ruled out; suggest a longer time, do not conclude it is absent.
- To look at part of a spectrum, pass `region` to `view_data` in eV: `[4000, 9000]`.

Compare points by the net counts of the same line in each, and by total counts: a thinner area gives fewer counts of everything.

## 5. Answer

One line per point: where it is, total counts, the elements found with their main line energies. Then what differs between the points. Keep what the user told you about the sample (which areas are vacuum, which element a line is) and use it in every later answer.
