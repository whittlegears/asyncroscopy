"""Skills-aware ReAct agent (single agent with MCP tools)."""

from __future__ import annotations

from typing import Any, Sequence

DEFAULT_SYSTEM_PROMPT = (
    "You are an operator assistant for an electron microscope controlled through MCP tools.\n"
    "Tool names are '<DeviceClass>_<command>' (e.g. DigitalTwin_acquire_scanned_image). Acquisition tools "
    "return a Tiled data key; use get_data_from_key to inspect what was saved. Work step by step: call one "
    "tool, read its result, then decide the next call. Report tool results faithfully, including failures. "
    "Never call Init, Kill or RestartServer."
)


def build_react_graph(
    model: Any,
    tools: Sequence[Any],
    registry: Any | None = None,
    system_prompt: str | None = None,
    name: str = "react_agent",
    **create_agent_kwargs: Any,
) -> Any:
    """Compile a ReAct agent over ``tools``.

    When a ``SkillRegistry`` is given, the four skill tools are added and the
    skills index (names + descriptions) is appended to the system prompt so the
    model can load a skill's full instructions on demand.
    """
    from langchain.agents import create_agent

    all_tools = list(tools)
    prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    if registry is not None:
        from asyncroscopy.agent.skills.registry import render_skills_index
        from asyncroscopy.agent.skills.tools import make_skill_tools

        all_tools.extend(make_skill_tools(registry))
        prompt = f"{prompt}\n\n{render_skills_index(registry)}"

    return create_agent(model=model, tools=all_tools, system_prompt=prompt, name=name, **create_agent_kwargs)
