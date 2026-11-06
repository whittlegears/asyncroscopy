"""The one approval decision: may this call run now, wait for a person, or run as a job?

Every place that asks it uses ``decide``: the MCP server's approval gate, ``find_tools`` ("when"),
``endpoint_info`` ("auto"), and the Approvals device's ``check`` in auto mode. ``required_role`` is
the token role a tool needs on an endpoint with ``mcp.auth``, from the same facts.

A tool is described by what the MCP server declared for it at discovery
(``asyncroscopy.mcp.descriptors.ToolDescriptor``); anything with the attributes of ``Decidable`` will do.
"""

from __future__ import annotations

from fnmatch import fnmatchcase
from typing import Iterable, Literal, Protocol

Outcome = Literal["runs", "held", "job", "refused"]


class Decidable(Protocol):
    name: str
    risk: str  # "low" | "high"
    operator_only: bool
    routine: bool  # auto mode runs it without a person
    job_start: bool  # it starts a job: auto mode runs it when every command the job drives runs


def switched_on(name: str, patterns: Iterable[str]) -> bool:
    """True when an operator switched ``name`` on for auto mode (tool-name globs, case ignored)."""
    return any(fnmatchcase(name.lower(), str(p).lower()) for p in patterns)


def decide(tool: Decidable, *, endpoint_role: str, mode: str | None, switched: Iterable[str] = ()) -> Outcome:
    """What the approval mode does with one call of ``tool`` on an endpoint of ``endpoint_role``.

    ``mode``: the Approvals device's mode ("auto" or "approve"); None when the endpoint has no
    Approvals device (nothing is held); "unknown" when it does not answer (high-risk calls are refused).
    ``switched``: the tool-name globs an operator switched on for auto mode.

    "runs": no person needed. "held": a person must approve this exact call (a grant may already
    exist; the Approvals device checks). "job": auto mode runs this job start when every command the
    reserved plan drives would run. "refused": the approval service is not answering.
    """
    if endpoint_role == "operator" or tool.risk == "low" or mode is None:
        return "runs"
    if mode == "unknown":
        return "refused"
    if mode == "auto":
        if tool.routine or switched_on(tool.name, switched):
            return "runs"
        if tool.job_start:
            return "job"
    return "held"


def required_role(tool: Decidable | None) -> str:
    """The token role a tool needs: "approve" for operator-only commands, "operate" for high-risk ones,
    "read" for the rest. A tool the endpoint does not know (None) needs "operate": unknown is high."""
    if tool is None:
        return "operate"
    if tool.operator_only:
        return "approve"
    return "operate" if tool.risk != "low" else "read"
