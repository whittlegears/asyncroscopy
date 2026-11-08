"""Skills this endpoint offers: list_skills, load_skill and skills_for (asyncroscopy.skills.registry)."""

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
        return [self.list_skills, self.load_skill, self.skills_for]

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

    @tool()
    def skills_for(self, request: str, agent_tools: list[str] | None = None) -> dict[str, Any]:
        """For agent programs, not models: the skills matching a request, written out for a system prompt,
        chosen as asyncroscopy's own agents choose them (keywords, name and description; a read-only request
        gets no skill that acts). agent_tools: the tools the caller gives its model (default: this endpoint's
        agent_tools); a skill none of whose tools the model holds is not chosen. Returns names (best first),
        text (their instructions, within the prompt budget) and cut (names too long to fit: load_skill them)."""
        from asyncroscopy.mcp.agent_tools import for_model, tool_lists
        from asyncroscopy.skills.registry import select_skills, skills_text

        discovery = self.server.discovery
        own = self.server.tool_names()
        if agent_tools:
            held = set(agent_tools)
        else:
            globs, lists = discovery.agent_tool_globs(), tool_lists(discovery.agent_profile)
            held = {name for name in own if for_model(name, globs, lists)}
        names = [s.name for s in select_skills(request, held, own | held)]
        text, cut = skills_text(names, own | held)
        return {"names": names, "text": text, "cut": cut}


def register(server: MCPServer, cfg: ServerConfig) -> SkillsTools:
    return SkillsTools(server, cfg)
