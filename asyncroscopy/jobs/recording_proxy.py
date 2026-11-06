"""RecordingProxy: what ``JobDevice.proxy`` hands a dry-run job instead of a DeviceProxy.

A dry run rehearses a job on the real stack without moving anything (decision D3
of docs/designs/concepts_brief.md). It works from allow-lists, so a DeviceProxy
method nobody listed can never reach the instrument by accident:

* reads pass through: reading a Tango attribute (``proxy.imsize``), the DeviceProxy
  methods in ``PROXY_READS`` (``read_attribute``, ``get_attribute_config``, ...),
  commands the safety policy counts as read-only (``safety.policy.is_read_only``:
  ``get_*``, ``list_*``, ``State``, ``Status``, ``ping``, unless the deployment's
  ``risk_overrides`` mark one high) and the job envelope's reads (``job_status``,
  ``results``, ``request_status``);
* every other command, every attribute write and every DeviceProxy method in
  ``PROXY_WRITES`` (``write_attribute*``, ``set_attribute_config``,
  ``poll_attribute``, ``write_pipe``, ...) is recorded, not performed, and returns
  ``DRY_RUN_PLACEHOLDER`` (``None`` for the DeviceProxy writes);
* any other DeviceProxy method (one on neither list) is refused with
  ``DryRunRefused``: an unknown method is assumed to write;
* the job device's safe commands (``JobDevice.safe_commands``: blanking the beam for a
  Mapper on an electron microscope, retracting the tip for a probe) really run:
  making the instrument safer is never rehearsed.

Each call that is not a plain read goes to ``recorder(address, name, kind, args,
performed)``; the job device writes it to the Instrument Trace with
``dry_run: true`` and checks it against the plan (``JobDevice.check_dry_call``).
"""

from __future__ import annotations

from fnmatch import fnmatchcase
from typing import TYPE_CHECKING, Any, Callable

from asyncroscopy.safety.policy import check_overrides, is_read_only

if TYPE_CHECKING:
    import tango

DRY_RUN_PLACEHOLDER = "dry_run:not_performed"
# The job envelope's and approval device's reads, beside the policy's read-only names.
READ_COMMANDS = frozenset({"job_status", "results", "request_status"})
# DeviceProxy methods that only read or set up the local proxy: passed through.
PROXY_READS = frozenset({
    "read_attribute", "read_attributes", "get_attribute_config", "get_attribute_config_ex", "attribute_query",
    "attribute_list_query", "attribute_list_query_ex", "get_attribute_list", "command_query", "command_list_query",
    "get_command_list", "get_command_config", "info", "name", "dev_name", "adm_name", "description", "ping", "state", "status",
    "set_timeout_millis", "get_timeout_millis", "get_property", "get_device_db", "import_info", "get_pipe_list",
    "is_attribute_polled", "get_attribute_poll_period", "get_command_poll_period", "polling_status", "black_box",
})
# DeviceProxy methods that write to the device (values, configuration, polling, pipes, properties): recorded, never performed.
PROXY_WRITES = frozenset({
    "write_attribute", "write_attributes", "write_read_attribute", "write_read_attributes", "write_attribute_asynch",
    "write_attributes_asynch", "write_attribute_reply", "write_attributes_reply", "set_attribute_config",
    "poll_attribute", "stop_poll_attribute", "poll_command", "stop_poll_command", "write_pipe", "write_read_pipe",
    "put_property", "delete_property",
})
ATTRIBUTE_WRITES = frozenset({
    "write_attribute", "write_attributes", "write_read_attribute", "write_read_attributes", "write_attribute_asynch",
    "put_property", "delete_property",
})
PROXY_COMMANDS = frozenset({"command_inout", "command_inout_asynch", "command_inout_raw"})


class DryRunRefused(AttributeError):
    """A DeviceProxy method that is on neither the read nor the write list: a dry run does not offer it."""


def is_read_command(name: str, dev_class: str = "", risk_overrides: dict[str, str] | None = None) -> bool:
    """A command a dry run passes through: read-only by the safety policy, or one of the envelope's reads.

    An override marking it high wins, written as "Class.command" or as the MCP tool name "Class_command"
    (mcp.safety.risk takes both). When the device's class is unknown ("") and any override is high, a
    read is recorded rather than risk passing through one the deployment marked high.
    """
    if name.lower() in READ_COMMANDS:
        return True
    high = [p.lower() for p, level in check_overrides(risk_overrides).items() if level == "high"]
    if high and not dev_class:
        return False
    if any(fnmatchcase(f"{dev_class}_{name}".lower(), pattern) for pattern in high):
        return False
    return is_read_only(dev_class, name, risk_overrides)


class RecordingProxy:
    """Wraps a DeviceProxy: reads pass through, writes and commands are recorded (see the module doc)."""

    def __init__(
        self,
        proxy: tango.DeviceProxy,
        address: str,
        recorder: Callable[[str, str, str, Any, bool], None],
        risk_overrides: dict[str, str] | None = None,
        safe_commands: frozenset[str] = frozenset(),
    ) -> None:
        object.__setattr__(self, "_proxy", proxy)
        object.__setattr__(self, "_safe", frozenset(c.lower() for c in safe_commands))
        object.__setattr__(self, "_address", address)
        object.__setattr__(self, "_recorder", recorder)
        object.__setattr__(self, "_risk_overrides", risk_overrides)
        object.__setattr__(self, "_commands", None)
        object.__setattr__(self, "_dev_class", None)

    def _command_names(self) -> set[str] | None:
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

    def _class(self) -> str:
        """The device's Tango class, for the deployment's risk patterns ("" when it cannot be read, or none are set)."""
        if not self._risk_overrides:
            return ""
        if self._dev_class is None:
            try:
                dev_class = getattr(self._proxy.info(), "dev_class", "")
            except Exception:
                dev_class = ""
            object.__setattr__(self, "_dev_class", dev_class if isinstance(dev_class, str) else "")
        return self._dev_class

    def _call(self, name: str, args: tuple) -> Any:
        """One command: read -> pass through; a safe command -> run and record; anything else -> record only."""
        if is_read_command(name, self._class(), self._risk_overrides):
            return getattr(self._proxy, name)(*args)
        performed = name.lower() in self._safe
        self._recorder(self._address, name, "command", list(args), performed)
        if performed:
            return getattr(self._proxy, name)(*args)
        return DRY_RUN_PLACEHOLDER

    def _record_write(self, method: str):
        def write(*args, **kwargs):
            attribute = args[0] if args else kwargs.get("attr_name")
            value = args[1] if len(args) > 1 else kwargs.get("value")
            if method in ATTRIBUTE_WRITES:  # traced as write_attribute, as they always were
                arguments = {"attribute": attribute, "value": value} if isinstance(attribute, str) else list(args)
                self._recorder(self._address, "write_attribute", "write", arguments, False)
            else:
                plain = lambda v: v if isinstance(v, (str, int, float, bool, type(None), list, tuple, dict)) else repr(v)  # noqa: E731
                self._recorder(self._address, method, "write", {"args": [plain(a) for a in args], "kwargs": {k: plain(v) for k, v in kwargs.items()}}, False)
            return None
        return write

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in PROXY_READS:
            return getattr(self._proxy, name)
        if name in PROXY_WRITES:
            return self._record_write(name)
        if name in PROXY_COMMANDS:
            def command_inout(command: str, *args, **kwargs):
                return self._call(command, args)
            return command_inout
        commands = self._command_names()
        lowered = name.lower()
        if commands is not None and lowered in commands:
            return lambda *args: self._call(name, args)
        attributes = self._attribute_names()
        if lowered in attributes:
            return getattr(self._proxy, name)  # reading a Tango attribute
        if commands is None:
            return lambda *args: self._call(name, args)  # cannot tell: a command, recorded (the safe side)
        raise DryRunRefused(
            f"a dry run does not offer DeviceProxy.{name} on {self._address}: it is not an attribute, a command, "
            "or a DeviceProxy method known to only read (recording_proxy.PROXY_READS) or to write (PROXY_WRITES)"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        # ``proxy.dwell_time = 1e-6`` writes a Tango attribute: recorded, not performed.
        self._recorder(self._address, "write_attribute", "write", {"attribute": name, "value": value}, False)
