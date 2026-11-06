"""One home for the names every layer shares: tool names, HTTP headers, job envelope verbs and states,
data-key patterns, the canonical action roles, and the default endpoint addresses.

Nothing here imports anything outside the standard library.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------- endpoints and headers
DEFAULT_MCP_URL = "http://127.0.0.1:8000/mcp"
DEFAULT_ANALYSIS_URL = "http://127.0.0.1:8003/mcp"
# Who is calling (self-reported; with mcp.auth the token's subject replaces it) and which agent run
# the call belongs to: agents send both with every call, and run_job finds a held start again by them.
# HTTP header names ignore case; servers read them lower-cased.
CALLER_HEADER = "X-Asyncroscopy-Caller"
RUN_HEADER = "X-Asyncroscopy-Run"


def tool_name(prefix: str, command: str) -> str:
    """The MCP tool name of a device command: ``<prefix>_<command>`` with "/" and "-" made "_"."""
    return f"{prefix}_{command}".replace("/", "_").replace("-", "_")


# ---------------------------------------------------------------------- the job envelope
# A device implementing every one of these runs jobs (asyncroscopy.jobs.job_device).
ENVELOPE_VERBS = frozenset({"reserve", "start", "job_status", "abort", "results", "release"})
# Job states. A worker owns a job in RUNNING_JOB_STATES; ACTIVE_JOB_STATES adds a reservation not yet
# started; FINISHED_JOB_STATES never change again.
RUNNING_JOB_STATES = frozenset({"queued", "running", "aborting", "paused_awaiting_approval", "paused_suspended", "paused"})
ACTIVE_JOB_STATES = RUNNING_JOB_STATES | {"reserved"}
FINISHED_JOB_STATES = frozenset({"done", "failed", "aborted"})

# ---------------------------------------------------------------------- data keys
# File types the DATA device registers with Tiled; an acquisition's result is a key ending in one.
REGISTRABLE_SUFFIXES = (".h5", ".hdf5", ".hdf", ".nxs", ".tif", ".tiff")
_SUFFIXES = "|".join(sorted((re.escape(s[1:]) for s in REGISTRABLE_SUFFIXES), key=len, reverse=True))
# A data key inside free text (a tool reply, a model's answer).
DATA_KEY_IN_TEXT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-.]*\.(?:" + _SUFFIXES + r")" + r"(?![A-Za-z0-9_])")
# A whole string that is a data key (it may name a folder: "run_1/tile_0003.h5").
_DATA_KEY = re.compile(r"^[\w.\-/]+\.(?:" + _SUFFIXES + r")$", re.IGNORECASE)


def is_data_key(text: str) -> bool:
    """True when ``text`` (stripped) is one data key and nothing else."""
    return bool(_DATA_KEY.match(text.strip()))


# ---------------------------------------------------------------------- canonical action roles
# The Tango attribute through which a device annotates its commands, as JSON
# {command: {"role", "args", "returns", "risk", "operator_only", "stop", "auto"}}; see asyncroscopy.utils.commands.
ACTION_HINTS = "action_hints"

# role -> what a tool with that role does (shown in the action space and the docs). Kept short on
# purpose: add a role only when a controller needs to tell two tools apart that these lump together.
ROLES: dict[str, str] = {
    "stage.move": "move the sample stage (all axes, or one)",
    "stage.read": "read the stage position",
    "beam.blank": "stop the probe beam (electrons, laser, X-rays) reaching the sample",
    "beam.unblank": "let the probe beam reach the sample",
    "beam.place": "park the beam on one point of the field of view",
    "optics.focus": "set the focus or defocus, or run an autofocus",
    "optics.fov": "set the field of view (scan size, magnification)",
    "acquire.image": "acquire a 2-D image; returns a data key",
    "acquire.spectrum": "acquire a 1-D spectrum or curve; returns a data key",
    "acquire.other": "acquire anything else (a value, a 4-D dataset)",
    "probe.approach": "bring a probe (tip, indenter) onto the sample",
    "probe.retract": "lift a probe off the sample",
    "job.run": "run a whole job in one call (reserve, start, wait)",
    "job.reserve": "check a job plan and hold the device",
    "job.start": "start a reserved job",
    "job.status": "read a job's state, or wait for it",
    "job.abort": "stop a running job, leaving the instrument safe",
    "job.results": "read a finished job's results",
    "job.release": "free a reserved job device",
    "safe_stop": "stop or switch off something (always low risk)",
    "setting.read": "read a setting or the instrument's status",
    "setting.set": "change a setting",
    "data.read": "read stored data (by key)",
    "other": "anything the rules do not place",
}
# Roles whose commands stop something: stop_all calls the no-argument ones, and they are never held.
STOP_ROLES = frozenset({"beam.blank", "probe.retract", "safe_stop"})


def check_role(role: str, where: str = "role") -> str:
    """``role`` if it is in the vocabulary; otherwise ValueError naming the roles there are."""
    if role not in ROLES:
        raise ValueError(f"{where}: {role!r} is not a role; roles: {', '.join(ROLES)}")
    return role
