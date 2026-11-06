"""What each tool of the instrument endpoint is, as the MCP server decided it at discovery.

Agent programs read it once per run from ``endpoint_info`` (its ``tools``: device, command, canonical
role, risk, whether it acts), through the MCP client (``AgentConfig.catalog``), and never guess from a
tool's name. A tool the catalog does not list is not one of the instrument endpoint's (an analysis
server tool, or a name a model made up): it is no instrument action and carries no risk here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

# What kind of instrument action a canonical role is (a goal's no_unrequested categories; claim checks).
_CATEGORIES = {
    "acquire.image": "acquire", "acquire.spectrum": "acquire", "acquire.other": "acquire",
    "stage.move": "move",
    "probe.approach": "probe", "probe.retract": "probe",
    "beam.blank": "beam", "beam.unblank": "beam", "beam.place": "beam",
    "optics.focus": "settings", "optics.fov": "settings", "setting.set": "settings",
    "job.run": "job", "job.reserve": "job", "job.start": "job", "job.status": "job", "job.abort": "job",
    "job.results": "job", "job.release": "job",
}


@dataclass(frozen=True)
class ToolInfo:
    """One tool: its device prefix and command (None for a native tool), canonical role, risk, and whether it acts."""

    name: str
    device: str | None
    command: str | None
    role: str
    risk: str
    acts: bool


class ToolCatalog:
    """The instrument endpoint's tools by name. ``known``: it came from the endpoint (an empty catalog knows nothing)."""

    def __init__(self, tools: Iterable[ToolInfo] = (), known: bool = True) -> None:
        self.tools = {t.name: t for t in tools}
        self.known = known

    @classmethod
    def unknown(cls) -> ToolCatalog:
        """A catalog for a run without an endpoint to ask: nothing is an instrument action, no risk is known."""
        return cls((), known=False)

    @classmethod
    def from_info(cls, info: Mapping) -> ToolCatalog:
        """From an ``endpoint_info`` reply ({} or one without ``tools``: an unknown catalog)."""
        raw = info.get("tools") if isinstance(info, Mapping) else None
        if not isinstance(raw, Mapping) or not raw:
            return cls.unknown()
        return cls(ToolInfo(name=str(name), device=t.get("device"), command=t.get("command"), role=str(t.get("role") or "other"),
                            risk=str(t.get("risk") or "high"), acts=bool(t.get("acts", True)))
                   for name, t in raw.items() if isinstance(t, Mapping))

    def get(self, name: str) -> ToolInfo | None:
        return self.tools.get(name)

    def risk(self, name: str) -> str | None:
        """The tool's risk as the approval gate sees it; None for a tool the endpoint does not offer."""
        tool = self.tools.get(name)
        return tool.risk if tool is not None else None

    def device(self, name: str) -> str | None:
        """The device prefix of a device command; None for a native tool or one the endpoint does not offer."""
        tool = self.tools.get(name)
        return tool.device if tool is not None else None

    def role(self, name: str) -> str | None:
        tool = self.tools.get(name)
        return tool.role if tool is not None else None

    def is_device_tool(self, name: str) -> bool:
        return self.device(name) is not None

    def category(self, name: str) -> str:
        """What kind of instrument action a tool is: job, acquire, move, probe, beam, settings, or other.
        Only device commands and run_job are instrument actions; reads, stops and anything unknown are other."""
        tool = self.tools.get(name)
        if tool is None or (tool.device is None and tool.role != "job.run"):
            return "other"
        return _CATEGORIES.get(tool.role, "other")

    def device_names(self) -> set[str]:
        return {t.name for t in self.tools.values() if t.device is not None}
