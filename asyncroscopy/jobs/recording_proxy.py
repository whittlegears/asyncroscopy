"""RecordingProxy: what ``JobDevice.proxy`` hands a dry-run job instead of a DeviceProxy.

A dry run rehearses a job on the real stack without moving anything (decision D3
of docs/designs/concepts_brief.md):

* every read passes through: attribute reads, ``read_attribute``,
  ``get_attribute_config``, commands named ``get_*``/``list_*``, ``state``,
  ``status``, ``ping`` and the job envelope's reads (``job_status``, ``results``,
  ``get_reservation``);
* every other command and every attribute write is recorded, not performed, and
  returns ``DRY_RUN_PLACEHOLDER``;
* blanking (``blank_beam``) is the one write that really runs: making the
  instrument safer is never rehearsed.

Each call that is not a plain read goes to ``recorder(address, name, kind, args,
performed)``; the job device writes it to the Instrument Trace with
``dry_run: true`` and checks it against the plan (``JobDevice.check_dry_call``).
"""

from __future__ import annotations

from typing import Any, Callable

DRY_RUN_PLACEHOLDER = "dry_run:not_performed"
# Commands that really run in a dry run: blanking only makes the instrument safer.
PASS_THROUGH_WRITES = frozenset({"blank_beam"})
# Commands that only read (the code base's naming convention, safety/policy.py, plus the envelope's reads).
READ_COMMANDS = frozenset({"state", "status", "ping", "job_status", "results", "get_reservation", "request_status"})
READ_PREFIXES = ("get_", "list_")
# DeviceProxy methods that only read or set up the local proxy.
PROXY_READS = frozenset({
    "read_attribute", "read_attributes", "get_attribute_config", "get_attribute_config_ex", "attribute_query",
    "attribute_list_query", "attribute_list_query_ex", "get_attribute_list", "command_query", "command_list_query",
    "get_command_list", "info", "name", "dev_name", "adm_name", "description", "ping", "state", "status",
    "set_timeout_millis", "get_timeout_millis", "get_property", "get_device_db", "import_info",
})
# DeviceProxy methods that write: recorded, never performed.
PROXY_WRITES = frozenset({
    "write_attribute", "write_attributes", "write_read_attribute", "write_read_attributes", "write_attribute_asynch",
    "put_property", "delete_property",
})
PROXY_COMMANDS = frozenset({"command_inout", "command_inout_asynch", "command_inout_raw"})


def is_read_command(name: str) -> bool:
    lowered = name.lower()
    return lowered in READ_COMMANDS or lowered.startswith(READ_PREFIXES)


class RecordingProxy:
    """Wraps a DeviceProxy: reads pass through, writes and commands are recorded (see the module doc)."""

    def __init__(self, proxy, address: str, recorder: Callable[[str, str, str, Any, bool], None]):
        object.__setattr__(self, "_proxy", proxy)
        object.__setattr__(self, "_address", address)
        object.__setattr__(self, "_recorder", recorder)
        object.__setattr__(self, "_commands", None)

    def _command_names(self) -> set[str]:
        if self._commands is None:
            try:
                names = {str(n).lower() for n in self._proxy.get_command_list()}
            except Exception:  # cannot list them: treat every unknown name as a command (recorded, the safe side)
                names = None
            object.__setattr__(self, "_commands", names)
        return self._commands

    def _attribute_names(self) -> set[str]:
        try:
            return {str(n).lower() for n in self._proxy.get_attribute_list()}
        except Exception:
            return set()

    def _call(self, name: str, args: tuple) -> Any:
        """One command: read -> pass through; blank_beam -> run and record; anything else -> record only."""
        if is_read_command(name):
            return getattr(self._proxy, name)(*args)
        performed = name.lower() in PASS_THROUGH_WRITES
        self._recorder(self._address, name, "command", list(args), performed)
        if performed:
            return getattr(self._proxy, name)(*args)
        return DRY_RUN_PLACEHOLDER

    def __getattr__(self, name: str) -> Any:
        if name in PROXY_READS:
            return getattr(self._proxy, name)
        if name in PROXY_WRITES:
            def write(*args, **kwargs):
                attribute = args[0] if args else kwargs.get("attr_name")
                value = args[1] if len(args) > 1 else kwargs.get("value")
                self._recorder(self._address, "write_attribute", "write", {"attribute": attribute, "value": value} if isinstance(attribute, str) else list(args), False)
                return None
            return write
        if name in PROXY_COMMANDS:
            def command_inout(command: str, *args, **kwargs):
                return self._call(command, args)
            return command_inout
        commands = self._command_names()
        lowered = name.lower()
        if (commands is not None and lowered in commands) or (commands is None and lowered not in self._attribute_names() and not name.startswith("_")):
            return lambda *args: self._call(name, args)
        return getattr(self._proxy, name)  # an attribute read, or another DeviceProxy method

    def __setattr__(self, name: str, value: Any) -> None:
        # ``proxy.dwell_time = 1e-6`` writes a Tango attribute: recorded, not performed.
        self._recorder(self._address, "write_attribute", "write", {"attribute": name, "value": value}, False)
