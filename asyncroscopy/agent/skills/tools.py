"""LangChain tool wrappers around the skill registry."""

from __future__ import annotations

from typing import Any

from asyncroscopy.agent.skills.registry import SkillRegistry, skill_functions

SKILL_TOOL_DESCRIPTIONS = {
    "skills_list": "List every installed skill with its name, version, and description.",
    "skill_search": "Search the skills library by keyword. Returns the best-matching skills with descriptions.",
    "skill_view": "Load the full step-by-step instructions of a skill. Call this before performing a task a skill covers.",
    "skill_save": (
        "Save a new reusable skill (procedure) to the skills library as Markdown. "
        "Use a short lowercase-hyphenated name, a one-line description, and a body with numbered steps "
        "naming the exact tools and arguments that worked."
    ),
}


def make_skill_tools(registry: SkillRegistry) -> list[Any]:
    """Return ``skills_list``, ``skill_search``, ``skill_view``, ``skill_save`` as StructuredTools."""
    from langchain_core.tools import StructuredTool

    tools = []
    for name, fn in skill_functions(registry).items():
        tools.append(StructuredTool.from_function(fn, name=name, description=SKILL_TOOL_DESCRIPTIONS[name]))
    return tools
