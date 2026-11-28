"""The middleware an agent graph wraps its model in, built from the profile's list (``AgentConfig.middleware``)."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from asyncroscopy.agents.catalog import ToolCatalog
from asyncroscopy.agents.claims import ClaimCheckMiddleware
from asyncroscopy.agents.guards import EmptyReplyMiddleware, JobWaitMiddleware, UserWordsMiddleware
from asyncroscopy.agents.profile import AgentProfile
from asyncroscopy.agents.skill_middleware import SkillMiddleware


def build_middleware(names: Iterable[str], *, prompt: str, tool_names: set[str], offered: set[str], tools: list,
                     memory: Callable | None = None, num_ctx: int = 8192, deadline: float | None = None,
                     claim_check: bool | None = None, catalog: ToolCatalog | None = None,
                     profile: AgentProfile | None = None) -> list[Any]:
    """One middleware per name, in order (the first wraps the rest). ``tool_names``: the model's tools;
    ``offered``: every tool of the endpoint; ``tools``: the LangChain tools (job_wait calls wait_for_job);
    ``catalog``: what the endpoint's tools are (the claim check knows from it which calls acquire or move);
    ``memory``: ``Memory.notes_for``, given the profile's notes budget; ``profile``: the agent profile, whose
    skill limits and tool gates the skills middleware applies (none of either when None)."""
    limits = profile.skills if profile is not None else None
    gates = profile.tool_gates if profile is not None else ()
    notes = memory
    if memory is not None and limits is not None:
        def notes(request: str) -> str:
            return memory(request, limits.notes_chars)
    makers: dict[str, Callable[[], Any]] = {
        "skills": lambda: SkillMiddleware(prompt, tool_names, offered, memory=notes, num_ctx=num_ctx, limits=limits, gates=gates),
        "user_words": lambda: UserWordsMiddleware(tools),
        "job_wait": lambda: JobWaitMiddleware(tools, deadline=deadline),
        "empty_reply": EmptyReplyMiddleware,
        # Last, so it wraps the model call itself.
        "claim_check": lambda: ClaimCheckMiddleware(tool_names, enabled=claim_check, catalog=catalog),
    }
    unknown = [n for n in names if n not in makers]
    if unknown:
        raise ValueError(f"unknown middleware {unknown}; known: {list(makers)}")
    return [makers[name]() for name in names]
