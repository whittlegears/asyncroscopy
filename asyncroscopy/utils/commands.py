"""``command``: tango.server.command that also tells agents what the command does, and declares what it is.

Tango does not carry a command's Python docstring to clients. The MCP server
builds each tool's description from the command's ``doc_in`` and ``doc_out``,
so a command declared with only a docstring reaches an agent as
":return: (not documented)". This decorator copies the docstring's first
paragraph into ``doc_out`` when ``doc_out`` is not given:

    from asyncroscopy.utils.commands import action_hints_attribute, command

    class Indenter(Device):
        action_hints = action_hints_attribute()

        @command(dtype_in=float, dtype_out=str, doc_in=":param load_n: Peak load in newtons.", risk="high")
        def indent(self, load_n):
            \"\"\"Make one indent at the current position. Returns JSON with hardness (GPa).\"\"\"

        @command(risk="low", stop=True, role="probe.retract")
        def retract(self):
            \"\"\"Lift the indenter off the sample.\"\"\"

When ``doc_in`` does not start with ":param <name>:", the Python parameter's
name is added, so the agent's tool argument is called ``load_n`` rather than
``arg``. It accepts everything tango.server.command accepts, with or without
arguments, and these declarations, which the MCP server reads through the
device's ``action_hints`` attribute (``action_hints_attribute``) instead of
guessing from the command's name:

- ``risk``: "low" or "high" (as the approval gate sees it);
- ``operator_only``: only the operator endpoint offers it;
- ``stop``: a no-argument command the emergency stop calls;
- ``routine``: auto mode runs it without a person;
- ``role``: its canonical action role (``asyncroscopy.core.names.ROLES``).
"""

from __future__ import annotations

import inspect
import json
from typing import Any, Callable

from tango.server import attribute
from tango.server import command as tango_command

from asyncroscopy.core.names import ACTION_HINTS, check_role

DECLARED = "__asyncroscopy_declares__"


def summary(func: Callable[..., Any]) -> str:
    """First paragraph of a function's docstring, on one line."""
    doc = inspect.getdoc(func) or ""
    return " ".join(doc.split("\n\n", 1)[0].split())


def _declaration(risk: str | None, operator_only: bool | None, stop: bool | None, routine: bool | None, role: str | None) -> dict[str, Any]:
    found: dict[str, Any] = {}
    if risk is not None:
        if risk not in ("low", "high"):
            raise ValueError(f"risk must be 'low' or 'high', got {risk!r}")
        found["risk"] = risk
    for key, value in (("operator_only", operator_only), ("stop", stop), ("routine", routine)):
        if value is not None:
            found[key] = bool(value)
    if role is not None:
        found["role"] = check_role(role)
    return found


def command(f: Callable[..., Any] | None = None, *, risk: str | None = None, operator_only: bool | None = None, stop: bool | None = None,
            routine: bool | None = None, role: str | None = None, **kwargs: Any) -> Any:
    declared = _declaration(risk, operator_only, stop, routine, role)

    def decorate(func: Callable[..., Any]) -> Any:
        options = dict(kwargs)  # a decorator reused on several functions must not share their docs
        if not options.get("doc_out"):
            options["doc_out"] = summary(func)
        # The MCP server names the tool's argument after ":param <name>:" (otherwise "arg");
        # take the name from the Python signature when doc_in does not give one.
        params = [p for p in inspect.signature(func).parameters.values() if p.name != "self"]
        doc_in = options.get("doc_in") or ""
        if params and ":param " not in doc_in:
            options["doc_in"] = f":param {params[0].name}: {doc_in}".rstrip()
        wrapped = tango_command(func, **options)
        if declared:
            setattr(wrapped, DECLARED, (options.get("name") or func.__name__, dict(declared)))
        return wrapped

    return decorate(f) if f is not None else decorate


def declared_hints(device_class: type) -> dict[str, dict[str, Any]]:
    """{command: declarations} of every command of ``device_class`` (and its bases) declared with ``command``."""
    found: dict[str, dict[str, Any]] = {}
    for klass in reversed(device_class.__mro__):
        for value in vars(klass).values():
            entry = getattr(value, DECLARED, None)
            if entry is not None:
                name, declared = entry
                found[name] = {**found.get(name, {}), **declared}
    return found


def action_hints_attribute(extra: Callable[[Any], dict[str, dict]] | None = None) -> Any:
    """A read-only Tango attribute named ``action_hints`` serving what the device's commands declare, as JSON
    {command: {...}}. ``extra(device)`` adds hints the device works out itself (units, ranges, roles)."""

    def read(device: Any) -> str:
        hints = declared_hints(type(device))
        for command_name, hint in (extra(device) if extra is not None else {}).items():
            hints[command_name] = {**hints.get(command_name, {}), **hint}
        return json.dumps(hints)

    return attribute(fget=read, name=ACTION_HINTS, dtype=str, doc="What this device's commands are (risk, role, stop), for the MCP server.")
