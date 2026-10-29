---
name: quick_diffractogram
description: Take a diffractogram over a 2theta range and read its peaks.
version: 1.0.0
keywords: [diffractogram, xrd, 2theta, powder pattern, peaks]
requires: [XRDTwin_acquire_diffractogram]
tools: [XRDTwin_get_status, XRDTwin_xray_on, XRDTwin_set_start_deg, XRDTwin_set_stop_deg, XRDTwin_acquire_diffractogram, view_data]
acts: true   # following it moves or exposes the instrument; false for a skill that only reads data
---

<!-- Template: copy this file to asyncroscopy/skills/<name>/SKILL.md (or to <dir>/<name>/SKILL.md
     with <dir> listed in ASYNCROSCOPY_SKILLS_PATH), rename it, and replace the steps with your
     instrument's. Check it with: uv run python -m asyncroscopy.agents.skills lint <path>
     Agents get it automatically when a request contains its keywords. Agents never see
     comments like this one. -->

# Quick diffractogram

1. Turn the tube on with `XRDTwin_xray_on` if `XRDTwin_get_status` says it is off.
2. Set the range the request asks for with `XRDTwin_set_start_deg` and `XRDTwin_set_stop_deg` (default 20 to 80 degrees).
3. `XRDTwin_acquire_diffractogram`; it returns a data key.
4. `view_data` with `{"key": <key>, "format": "text"}` lists the peaks with their 2theta.

Report the key and the strongest peaks. If a call is held for approval, wait for the operator; never try another way around it.
