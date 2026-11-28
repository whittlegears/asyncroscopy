"""One home for the names every layer shares: tool names, HTTP headers, job envelope verbs and states,
data-key patterns, the canonical action roles, and the default endpoint addresses.

Nothing here imports anything outside the standard library.
"""

from __future__ import annotations

import json
import re

DEFAULT_MCP_URL = "http://127.0.0.1:8000/mcp"
DEFAULT_ANALYSIS_URL = "http://127.0.0.1:8003/mcp"
DEFAULT_TILED_URI = "http://127.0.0.1:9091"
default_operator_url = "http://127.0.0.1:8001/mcp"
autoscript_port = 9095
# Who is calling and which agent run the call belongs to. With mcp.auth the token's subject replaces the caller.
# Agents send both with every call, and run_job uses them to find a held start again.
# HTTP header names ignore case, and servers read them lower-cased.
CALLER_HEADER = "X-Asyncroscopy-Caller"
RUN_HEADER = "X-Asyncroscopy-Run"
RUN_KEY_HEADER = "X-Asyncroscopy-Run-Key"


endpoint_url_names = ("operator", "agent_server", "workflow_operator", "tiled", "analysis")


def endpoint_urls(raw: dict | None = None, where: str = "urls") -> dict[str, str | None]:
    """Every name in ``endpoint_url_names`` with its URL from ``raw`` (empty is null); any other name is refused."""
    raw = dict(raw or {})
    unknown = sorted(set(raw) - set(endpoint_url_names))
    if unknown:
        raise ValueError(f"{where}: unknown {unknown}; the names are {', '.join(endpoint_url_names)}")
    return {name: raw.get(name) or None for name in endpoint_url_names}


def devices_of_kind(endpoint_info: dict, kind: str) -> list[str]:
    """The tool prefixes endpoint_info's device_kinds gives ``kind`` (stem, probe, envelope, map, stage)."""
    return sorted(prefix for prefix, kinds in (endpoint_info.get("device_kinds") or {}).items() if kind in (kinds or []))


# The agent graphs one request can name, besides the swarm the agent MCP server runs by default.
AGENT_GRAPHS = ("mapping", "general", "team")


def tool_name(prefix: str, command: str) -> str:
    """The MCP tool name of a device command: ``<prefix>_<command>`` with "/" and "-" made "_"."""
    return f"{prefix}_{command}".replace("/", "_").replace("-", "_")


# A device that implements every one of these runs jobs.
ENVELOPE_VERBS = frozenset({"reserve", "start", "job_status", "abort", "results", "release"})
# Job states. A worker owns a job in RUNNING_JOB_STATES. ACTIVE_JOB_STATES adds a reservation
# not yet started. FINISHED_JOB_STATES never change again.
RUNNING_JOB_STATES = frozenset({"queued", "running", "aborting", "paused_suspended", "paused"})
ACTIVE_JOB_STATES = RUNNING_JOB_STATES | {"reserved"}
FINISHED_JOB_STATES = frozenset({"done", "failed", "aborted"})

# File types the DATA device registers with Tiled. An acquisition's result is a key ending in one.
REGISTRABLE_SUFFIXES = (".h5", ".hdf5", ".hdf", ".nxs", ".tif", ".tiff")
_SUFFIXES = "|".join(sorted((re.escape(s[1:]) for s in REGISTRABLE_SUFFIXES), key=len, reverse=True))
# A data key inside free text, such as a tool reply or a model's answer.
DATA_KEY_IN_TEXT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-.]*\.(?:" + _SUFFIXES + r")" + r"(?![A-Za-z0-9_])")
# A whole string that is a data key. It may name a folder, as in "run_1/tile_0003.h5".
_DATA_KEY = re.compile(r"^[\w.\-/]+\.(?:" + _SUFFIXES + r")$", re.IGNORECASE)


def is_data_key(text: str) -> bool:
    """True when ``text`` (stripped) is one data key and nothing else."""
    return bool(_DATA_KEY.match(text.strip()))


# The fields of a JSON reply that name data: an acquisition's key, a job's result keys.
KEY_FIELDS = ("key", "image_key", "spectrum_key", "result_key")


def reply_keys(reply: object, derived: bool = False) -> list[str]:
    """The data keys a tool's reply names: the key on the first line of an acquisition's text, and in a JSON reply
    the ``KEY_FIELDS`` and the ``keys`` list. ``derived`` adds the values of ``result_keys``: what an analysis job
    made, such as a stitched map."""
    if isinstance(reply, str):
        try:
            reply = json.loads(reply)
        except ValueError:
            first = reply.strip().split("\n", 1)[0].strip()
            return [first] if first and is_data_key(first) else []
    keys: list[str] = []
    if isinstance(reply, dict):
        keys += [str(reply[name]) for name in KEY_FIELDS if isinstance(reply.get(name), str) and reply[name]]
        keys += [str(k) for k in reply.get("keys") or [] if isinstance(k, str) and k] if isinstance(reply.get("keys"), list) else []
        found = reply.get("result_keys") if derived else None
        keys += [str(v) for v in found.values() if isinstance(v, str) and v] if isinstance(found, dict) else []
    elif isinstance(reply, str) and is_data_key(reply):
        keys.append(reply)
    return list(dict.fromkeys(keys))


# The Tango attribute where a device describes its commands as JSON. See asyncroscopy.utils.commands.
ACTION_HINTS = "action_hints"

# What a tool with each role does, as shown in the action space and the docs.
# Add a role only when a controller needs to tell two tools apart that these lump together.
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
# Roles whose commands stop something. stop_all calls the ones that take no argument, and they are never held.
STOP_ROLES = frozenset({"beam.blank", "probe.retract", "safe_stop"})
# Roles whose commands only observe. A command declared with one of these changes nothing.
READ_ROLES = frozenset({"stage.read", "setting.read", "data.read", "job.status", "job.results"})

# Commands every Tango device has. Nobody declares them, so the MCP server knows them by name.
# State and Status read. Init restarts the device, which loses its state.
TANGO_READS = frozenset({"State", "Status"})
TANGO_ADMIN = frozenset({"Init"})


def check_role(role: str, where: str = "role") -> str:
    """``role`` if it is in the vocabulary; otherwise ValueError naming the roles there are."""
    if role not in ROLES:
        raise ValueError(f"{where}: {role!r} is not a role; roles: {', '.join(ROLES)}")
    return role
