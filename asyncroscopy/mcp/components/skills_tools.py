"""Skills this endpoint offers: list_skills and load_skill (asyncroscopy.skills.registry)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastmcp.exceptions import ToolError
from fastmcp.tools import tool

if TYPE_CHECKING:
    from asyncroscopy.mcp.mcp_server import MCPServer, ServerConfig


class SkillsTools:
    def __init__(self, server: MCPServer, cfg: ServerConfig) -> None:
        self.server = server

    def natives(self) -> list:
        return [self.list_skills, self.load_skill]

    def description_of(self, name: str) -> str | None:
        """load_skill's description names the skills offered here, so a model need not call list_skills first."""
        return self.load_skill_description() if name == "load_skill" else None

    def skills(self) -> list:
        """Skills this endpoint offers: a learned skill only when every tool it lists is offered here."""
        from asyncroscopy.skills.registry import list_skills, usable_by

        names = self.server.tool_names()
        return [s for s in list_skills(names) if not s.learned or usable_by(s, names)]

    @tool()
    def list_skills(self) -> list[dict[str, Any]]:
        """Skills: procedures for kinds of experiment (e.g. mapping). Load one with load_skill before doing that kind of task."""
        return [{"name": s.name, "description": s.description, "version": s.version, "available": s.available, **({"problem": s.problem} if s.problem else {})}
                for s in self.skills()]

    def load_skill_description(self) -> str:
        """load_skill's description with the skill names this endpoint offers."""
        names = [s.name for s in self.skills() if s.available]
        listed = f" Skills here at startup: {', '.join(names)} (list_skills has descriptions and any learned since)." if names else ""
        return (self.load_skill.__doc__ or "").strip() + listed

    @tool()
    def load_skill(self, name: str) -> str:
        """Full instructions of one skill from list_skills. Follow them for that kind of task."""
        from asyncroscopy.skills.registry import read_front_matter

        skill = next((s for s in self.skills() if s.name == name), None)
        if skill is None:
            raise ToolError(f"no skill named {name!r}; available: {[s.name for s in self.skills() if s.available]}")
        if not skill.available:
            raise ToolError(f"skill {name!r} cannot be used: {skill.problem}")
        return read_front_matter(skill.path)[1]


def register(server: MCPServer, cfg: ServerConfig) -> SkillsTools:
    return SkillsTools(server, cfg)
