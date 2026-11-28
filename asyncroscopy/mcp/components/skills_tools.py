"""Skills this endpoint offers: list_skills, load_skill and skills_for (asyncroscopy.skills.registry), and for a
client's Skills page (never a model) skill_text, save_skill, delete_skill and switch_skill (asyncroscopy.skills.lab)."""

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
        from asyncroscopy import settings

        # Hide the tools when skills are off because an empty load_skill still attracts models.
        if not settings.get().skills:
            return []
        return [self.list_skills, self.load_skill, self.skills_for, self.skill_text, self.save_skill, self.delete_skill, self.switch_skill]

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
        """Skills: multi-step procedures (a map of an area, a survey, a zoom series). One action (one image, one
        spectrum, a move, a setting) needs none: call that instrument tool directly."""
        return [{"name": s.name, "description": s.description, "version": s.version, "available": s.available,
                 "source": s.source, "enabled": s.enabled, "off": s.off, "acts": s.acts, **({"problem": s.problem} if s.problem else {})}
                for s in self.skills()]

    def _named(self, name: str):
        skill = next((s for s in self.skills() if s.name == name), None)
        if skill is None:
            raise ToolError(f"no skill named {name!r}; there are: {[s.name for s in self.skills()]}")
        return skill

    @tool()
    def skill_text(self, name: str) -> dict[str, Any]:
        """For a client's Skills page, not models: one skill as written, whether or not it can be used here.
        Returns name, source (shipped, path, lab or learned), enabled (its front matter), off, problem, body (the manual) and text (the whole
        SKILL.md, front matter included, to edit and save_skill again)."""
        from asyncroscopy.skills.registry import read_front_matter

        skill = self._named(name)
        return {"name": skill.name, "source": skill.source, "enabled": skill.enabled, "off": skill.off, "problem": skill.problem,
                "body": read_front_matter(skill.path)[1], "text": skill.path.read_text(encoding="utf-8")}

    @tool()
    def save_skill(self, name: str, text: str, description: str | None = None) -> dict[str, Any]:
        """For a client's Skills page, not models: write the lab skill ``name`` from ``text`` (a SKILL.md, or a
        Markdown manual that gets front matter made for it), replacing a lab skill of that name. A lab skill with a
        shipped skill's name replaces the shipped one here. Refused with the linter's reasons when it is not
        acceptable. Returns its name and source."""
        from asyncroscopy.skills import lab

        try:
            lab.save(name, text, description)
        except ValueError as exc:
            raise ToolError(f"skill not saved: {exc}") from exc
        return {"name": lab.slug(name), "source": "lab", "saved": True}

    @tool()
    def delete_skill(self, name: str) -> dict[str, Any]:
        """For a client's Skills page, not models: delete the lab skill ``name``. A shipped skill is not deleted:
        switch it off with switch_skill."""
        from asyncroscopy.skills import lab

        try:
            lab.delete(name)
        except KeyError as exc:
            raise ToolError(str(exc.args[0])) from exc
        return {"name": name, "deleted": True}

    @tool()
    def switch_skill(self, name: str, on: bool) -> dict[str, Any]:
        """For a client's Skills page, not models: switch the skill ``name`` (any source) on or off on this
        endpoint. A skill switched off is listed but given to no agent. Returns the names now switched off."""
        from asyncroscopy.skills import lab

        self._named(name)
        return {"name": name, "on": on, "off": sorted(lab.switch(name, on))}

    def load_skill_description(self) -> str:
        """load_skill's description with the skill names this endpoint offers."""
        skills = [s for s in self.skills() if s.available]
        # Say what each skill is for so a model does not pick one by name alone.
        listed = f" Skills here at startup: {'; '.join(f'{s.name} ({_brief(s.description)})' for s in skills)}. list_skills has any learned since." if skills else ""
        return (self.load_skill.__doc__ or "").strip() + listed

    @tool()
    def load_skill(self, name: str) -> str:
        """Full instructions of one skill: a multi-step procedure. Load one only when the request asks for what
        that skill does. One action (one image, one spectrum, a move, a setting) needs no skill: call that
        instrument tool directly."""
        from asyncroscopy.skills.registry import read_front_matter

        skill = next((s for s in self.skills() if s.name == name), None)
        if skill is None:
            raise ToolError(f"no skill named {name!r}; available: {[s.name for s in self.skills() if s.available]}")
        if not skill.available:
            raise ToolError(f"skill {name!r} cannot be used: {skill.problem}")
        return read_front_matter(skill.path)[1]

    @tool()
    def skills_for(self, request: str, agent_tools: list[str] | None = None, context: list[str] | None = None) -> dict[str, Any]:
        """For agent programs, not models: what skills say for a request, written out for a system prompt, chosen
        as asyncroscopy's own agents choose it. The procedures the request asks for come whole (keywords, name and
        description; a read-only request gets no skill that acts), then passages of any skill that answer this
        step (asyncroscopy.skills.index). agent_tools: the tools the caller gives its model (default: this
        endpoint's agent_tools); a skill none of whose tools the model holds is not chosen. context: the latest
        tool replies, newest last, so the passages follow the task; call again after each tool round. Returns
        names (procedures, best first), text (procedures, then the advice), cut (procedures too long to fit:
        load_skill them), advice (the passages alone, to add after the latest tool reply each step so the system
        prompt stays the same) and passages (each passage's label)."""
        from asyncroscopy.agents.profile import SkillLimits
        from asyncroscopy.mcp.agent_tools import for_model, tool_lists
        from asyncroscopy.skills.index import passages_text, search
        from asyncroscopy.skills.registry import read_only_request, select_skills, skills_text, usable_by

        discovery = self.server.discovery
        own = self.server.tool_names()
        if agent_tools:
            held = set(agent_tools)
        else:
            globs, lists = discovery.agent_tool_globs(), tool_lists(discovery.agent_profile)
            held = {name for name in own if for_model(name, globs, lists)}
        limits = SkillLimits()
        names = [s.name for s in select_skills(request, held, own | held)]
        text, cut = skills_text(names, own | held)
        read_only = read_only_request(request)
        usable = [s for s in self.skills() if usable_by(s, held) and not (read_only and s.acts)]
        parts = [(request, 1.0), *[(reply[:800], 0.6) for reply in (context or [])[-2:]]]
        found = search(parts, usable, k=limits.passages, skip=set(names))
        advice, fitted = passages_text([p for p, _score in found], min(limits.passage_chars, max(0, limits.prompt_chars - len(text))))
        if advice:
            advice = "Advice that may help with this step, from this instrument's skills (use what applies):\n" + advice
            text = (text + "\n\n" if text else "") + advice
        return {"names": names, "text": text, "cut": cut, "advice": advice, "passages": [p.label for p in fitted]}


def _brief(description: str, limit: int = 110) -> str:
    """The first sentence of a description, cut to ``limit`` characters."""
    first = description.split(". ")[0].rstrip(".")
    return first if len(first) <= limit else first[:limit].rsplit(" ", 1)[0].rstrip(",;") + " ..."


def register(server: MCPServer, cfg: ServerConfig) -> SkillsTools:
    return SkillsTools(server, cfg)
