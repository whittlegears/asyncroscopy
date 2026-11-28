"""The agent memory this endpoint keeps (asyncroscopy.agents.memory), for agent programs: notes_for.

An app that runs its own agent against this endpoint reads the same instrument notes, lessons and
experiment summaries as asyncroscopy's own agents, so what the instrument has learned reaches every agent.
Read only: nothing here writes the memory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastmcp.tools import tool

from asyncroscopy.core.names import devices_of_kind

if TYPE_CHECKING:
    from asyncroscopy.mcp.mcp_server import MCPServer, ServerConfig

MAX_BUDGET_CHARS = 4000


class MemoryTools:
    def __init__(self, server: MCPServer, cfg: ServerConfig) -> None:
        self.server = server

    def natives(self) -> list:
        return [self.notes_for]

    @tool(annotations={"readOnlyHint": True})
    def notes_for(self, request: str, budget: int = 1500) -> dict[str, Any]:
        """For agent programs, not models: this instrument's notes most relevant to a request, chosen as
        asyncroscopy's own agents choose them. Sources: INSTRUMENT.md by section, LESSONS.md one lesson each,
        experiments/<page> summaries. Data keys and ids are replaced by placeholders. budget: characters, at
        most 4000. Returns notes ([{source, text}], best first) and text (them written out, "" if none)."""
        from asyncroscopy.agents.catalog import ToolCatalog
        from asyncroscopy.agents.memory import Memory

        memory = Memory.from_env()
        if not memory.instrument_file.exists() and memory.shared_instrument_file() is None:
            try:
                info = self.server.discovery.endpoint_info()
                memory.seed_instrument(ToolCatalog.from_info(info), stem=bool(devices_of_kind(info, "stem")))
            except Exception:  # noqa: BLE001 - notes are a help; a seeding fault must not fail the call
                pass
        found = memory.notes(request, max(0, min(int(budget), MAX_BUDGET_CHARS)))
        return {"notes": [{"source": source, "text": text} for source, text in found],
                "text": "\n".join(f"[{source}] {text}" for source, text in found)}


def register(server: MCPServer, cfg: ServerConfig) -> MemoryTools:
    return MemoryTools(server, cfg)
