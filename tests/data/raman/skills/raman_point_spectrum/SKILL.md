---
name: raman_point_spectrum
description: Record a Raman spectrum at a stage position and read its peaks (Si 520 cm-1, graphene G and 2D bands).
version: 1.0.0
keywords: [raman, raman spectrum, raman shift, graphene, 2d band, g band, wavenumber]
requires: [RamanTwin*_acquire_raman_spectrum]
tools: [RamanTwin_get_status, RamanTwin_laser_on, RamanTwin_laser_off, RamanTwin_set_stage_x_m, RamanTwin_set_stage_y_m, RamanTwin_set_exposure_s, RamanTwin_acquire_raman_spectrum, view_data]
---

# Raman point spectrum

1. `RamanTwin_get_status`: note the laser power, exposure and stage position.
2. If the request names a position, move with `RamanTwin_set_stage_x_m` and `RamanTwin_set_stage_y_m` (metres; 1 um = 1e-6).
3. If the laser is off, `RamanTwin_laser_on` with 2 mW unless the request says otherwise. Never more than 10 mW on graphene.
4. Keep `exposure_s * accumulations` under 20 s per spectrum; one call may not take longer.
5. `RamanTwin_acquire_raman_spectrum`; it returns a data key.
6. `view_data` with `{"key": <key>, "format": "text"}` lists the peaks in cm-1.
7. `RamanTwin_laser_off` when done.

Report the key and the peak positions in cm-1: Si near 520, graphene G near 1580 and 2D near 2690 (2D taller than G means a monolayer). If a call is held for approval, wait for the operator; never try another way around it.
