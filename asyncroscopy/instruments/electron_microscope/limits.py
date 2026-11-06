"""An electron microscope's limit defaults, and its commands' arguments with their units and ranges.

The ElectronMicroscope base checks every value against the device's own ``min_fov``, ``max_fov``,
``max_defocus`` and ``max_stigmator`` properties before a backend sees it (electron_microscope.py);
these are those properties' defaults. ``action_hints`` describes the same ranges for clients
(asyncroscopy.mcp.action_space).
"""

from __future__ import annotations


# The ElectronMicroscope properties' defaults (min_fov, max_fov, max_defocus, max_stigmator): what the base
# enforces when a deployment sets none.
DEFAULT_MIN_FOV = 1e-9
DEFAULT_MAX_FOV = 1e-4
DEFAULT_MAX_DEFOCUS = 1e-5
DEFAULT_MAX_STIGMATOR = 1e-6
STAGE_ELEMENTS = ("x", "y", "z", "alpha", "beta")
STAGE_UNITS = ("m", "m", "m", "deg", "deg")


def action_hints(min_fov: float = DEFAULT_MIN_FOV, max_fov: float = DEFAULT_MAX_FOV, max_defocus: float = DEFAULT_MAX_DEFOCUS) -> dict:
    """An electron microscope's command arguments with the units and ranges the checks above enforce,
    in the shape of a device's ``action_hints`` (see asyncroscopy.mcp.action_space)."""
    stage = {"type": "array", "items": "number", "elements": list(STAGE_ELEMENTS), "units": list(STAGE_UNITS)}
    return {
        "set_fov": {"args": [{"name": "fov", "type": "number", "units": "m", "min": float(min_fov), "max": float(max_fov)}]},
        "set_defocus": {"args": [{"name": "defocus", "type": "number", "units": "m", "min": -float(max_defocus), "max": float(max_defocus)}]},
        "get_fov": {"returns": {"type": "number", "units": "m"}},
        "get_defocus": {"returns": {"type": "number", "units": "m"}},
        "place_beam": {"args": [{"name": "position", "type": "array", "items": "number", "elements": ["x", "y"],
                                 "units": "fraction of the field of view", "min": 0.0, "max": 1.0}]},
        "move_stage": {"args": [{"name": "position", **stage}]},
        "get_stage": {"returns": stage},
        "get_screen_current": {"returns": {"type": "number", "units": "pA"}},
        "set_screen_current": {"args": [{"name": "current", "type": "number", "units": "pA"}]},
        "acquire_scanned_image": {"primary": True, "args": [{"name": "detector_list", "type": "array", "items": "string", "doc": "scanning detectors; [] means the microscope's default_detector"}]},
    }
