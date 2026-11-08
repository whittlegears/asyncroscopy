"""An instrument described in one YAML file: settings, commands and simulated acquisitions.

``load_spec(path)`` reads and checks the file and returns an ``InstrumentSpec``;
``SimInstrument(spec)`` holds the settings' current values and runs commands and
acquisitions. Nothing here imports Tango, so the behaviour is tested without a
device server; ``asyncroscopy.instruments.yaml_sim.device`` turns it into one (and
``asyncroscopy.instruments.yaml_sim.job`` serves a file's ``jobs:`` section).

The file format, with every key (see examples/xrd.yaml for a working instrument)::

    description: One line an agent reads about the instrument.
    seed: 0                              # random seed for the simulated noise
    settings:
      exposure_s:                        # -> get_exposure_s, set_exposure_s and an attribute
        default: 1.0
        min: 0.01                        # optional bounds, checked on every set
        max: 10
        units: s
        type: float                      # float (default), int, bool or str
        settable: true                   # false: get_ only
        requires: ["exposure_s < 5 or not shutter_open"]   # optional checks on every change (the new value is in scope)
        doc: Counting time per point.
      grating:
        default: "1800"
        type: str
        choices: ["600", "1800"]         # optional: the only values allowed
    commands:
      xray_on:
        doc: Switch the X-ray source on.
        input: {name: power_w, type: float, min: 0, max: 1600, units: W}   # one argument
        requires: ["power_w > 0"]        # optional checks, refused with the expression
        sets: {tube_power_w: power_w}    # settings to change; values are expressions
        returns: "X-rays on"             # a string, a number, or a mapping (sent as JSON)
        duration_s: 0.5                  # optional simulated time (at most 25 s per call)
        role: beam.unblank               # optional canonical role (asyncroscopy.core.names.ROLES), for role-based controllers
        risk: high                       # optional: low or high as the approval gate sees it (see below)
      move_xy:
        doc: Move the stage.
        inputs:                          # several arguments: the tool takes them by name
          - {name: x_m, min: -0.005, max: 0.005, units: m}
          - {name: y_m, min: -0.005, max: 0.005, units: m}
        sets: {stage_x_m: x_m, stage_y_m: y_m}
    acquisitions:
      acquire_scan:                      # names start with acquire; returns a data key
        doc: Measure a diffractogram.
        kind: spectrum                   # spectrum (1-D), image (2-D) or value
        units: counts                    # what the values are (default counts)
        inputs: [...]                    # optional arguments, as for commands
        role: acquire.spectrum           # optional; by default from the kind (acquire.image, acquire.spectrum, acquire.other)
        peaks:
          - {position: 28.44, height: 900, fwhm: 0.14, shape: gaussian}   # gaussian, lorentzian or voigt
        noise: poisson                   # poisson (counts only), gaussian or none
        noise_sigma: 2.0                 # gaussian's sigma in the data's units (needed unless units are counts)
        ...
    jobs:                                # optional: a job device for long or many-point runs
      line_scan:
        doc: Spectra along a line or a grid of stage positions.
        axes: [stage_x_m, stage_y_m]     # settings a plan sweeps (start/stop/points) or lists
        settings: [exposure_s]           # settings a plan may set for the whole job
        acquire: [acquire_scan]          # run at every point, in order
        settle_s: 0.2                    # simulated time to reach each point
        requires: ["tube_power_w > 0"]   # checked at reserve
        before_move: []                  # commands run at each point before its axes move (e.g. retract a probe)
        after_move: []                   # commands run at each point after it is reached, before acquiring (e.g. approach again)
        on_stop: [xray_off]              # commands run when the job is aborted or fails
    map:                                 # optional: how the Mapper (instrument_profile yaml) tiles this instrument
      image: acquire_topography          # the image acquisition per tile (default: the first image acquisition)
      field_of_view: scan_size_m         # the setting that is a tile's width in metres (default: the one pixel_size_m uses, if in m)
      stage: [stage_x_m, stage_y_m]      # the stage x and y settings (default: stage_x_m and stage_y_m, or a job's two axes)
      before_move: [retract_tip]         # default: a job's before_move, after_move and on_stop (as safe)
      after_move: [approach_tip]
      safe: [retract_tip]                # run between tiles and whenever the map stops
      settings: [setpoint_v]             # settings a map plan may set ("settings": {...}); default: a job's settings
      budgets:                           # what a plan can be refused for besides time_s
        tip_force_nn: {per_tile: "setpoint_v * 20", combine: max, units: nN, doc: peak tip force}

The file declares what each command is, so the MCP server need not guess from its name
(``asyncroscopy.mcp.descriptors``): ``get_<setting>`` reads (low); ``set_<setting>`` and every
acquisition are high; ``get_map_profile`` reads. A command is what its ``risk:`` says; without one, a
command whose ``role:`` stops something (beam.blank, probe.retract, safe_stop) is low and, taking no
input, one the emergency stop calls; one with any other role is high, and a ``get_``/``list_`` command
reads. A command with neither ``risk:`` nor ``role:`` is left to the MCP server's name rules, which list
it at startup: give it one.

An image's ``spots`` are drawn afresh for every image (``count`` of them), or,
with ``x_m`` and ``y_m`` (expressions such as ``stage_x_m``: where the image
centre is on the sample), fixed on the sample at ``per_um2`` per square
micrometre, so overlapping images of the same place show the same spots and a
map of them stitches.

A single command or acquisition call is limited to 25 s (the MCP server stops
waiting at 30 s); a job is not, because the agent waits for it with
wait_for_job. Jobs run on a second device (see ``device.py``) with the standard
job envelope, so run_job, wait_for_job, abort, approval and limits all apply.

Strings in ``sets``, ``returns``, ``requires``, ``duration_s`` and the numbers of an
acquisition are expressions over the settings (and the command's input): numbers,
names, + - * / ** %, comparisons, and/or/not, and abs, min, max, sqrt, exp, log,
sin, cos, radians, degrees. A plain string that is not an expression is returned as text.
"""

from __future__ import annotations

from .expressions import FUNCTIONS, NAME, OPERATORS, Refused, SpecError, evaluate
from .schema import ACQUISITION_KEYS, ACTION_VERBS, AXIS_KEYS, Acquisition, COMMAND_KEYS, Command, INPUT_KEYS, InstrumentSpec, JOB_KEYS, JSON_TYPES, Job, KINDS, KIND_ROLES, MAP_BUDGET_KEYS, MAP_COMMAND, MAP_KEYS, MAX_DURATION_S, MAX_JOB_POINTS, MAX_POINTS, MAX_STEP_DURATION_S, MapBudget, MapSection, NOISES, PEAK_KEYS, PEAK_SHAPES, PROJECT_DIR, Quantity, RESERVED_NAMES, SAMPLE_CELLS_MAX, SAMPLE_CELL_M, SETTING_KEYS, SETTLE_STEP, SPOT_KEYS, TOP_KEYS, TYPES, action_hints, duration_seconds, load_spec, map_description, map_section, parse_arguments, resolve_path
from .physics import FWHM_PER_SIGMA, SimInstrument, peak_profile, sample_spots
from .plans import JobPlan, job_estimate, job_targets, plan_example, resolve_job_plan

__all__ = ["ACQUISITION_KEYS", "ACTION_VERBS", "AXIS_KEYS", "Acquisition", "COMMAND_KEYS", "Command", "FUNCTIONS", "FWHM_PER_SIGMA", "INPUT_KEYS", "InstrumentSpec", "JOB_KEYS", "JSON_TYPES", "Job", "JobPlan", "KINDS", "KIND_ROLES", "MAP_BUDGET_KEYS", "MAP_COMMAND", "MAP_KEYS", "MAX_DURATION_S", "MAX_JOB_POINTS", "MAX_POINTS", "MAX_STEP_DURATION_S", "MapBudget", "MapSection", "NAME", "NOISES", "OPERATORS", "PEAK_KEYS", "PEAK_SHAPES", "PROJECT_DIR", "Quantity", "RESERVED_NAMES", "Refused", "SAMPLE_CELLS_MAX", "SAMPLE_CELL_M", "SETTING_KEYS", "SETTLE_STEP", "SPOT_KEYS", "SimInstrument", "SpecError", "TOP_KEYS", "TYPES", "action_hints", "duration_seconds", "evaluate", "job_estimate", "job_targets", "load_spec", "map_description", "map_section", "parse_arguments", "peak_profile", "plan_example", "resolve_job_plan", "resolve_path", "sample_spots"]
