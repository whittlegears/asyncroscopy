"""The middleware an agent graph wraps its model in, built from the profile's list (``AgentConfig.middleware``)."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from asyncroscopy.agents.claims import ClaimCheckMiddleware
from asyncroscopy.agents.guards import EmptyReplyMiddleware, JobWaitMiddleware
from asyncroscopy.agents.skill_middleware import SkillMiddleware


def build_middleware(names: Iterable[str], *, prompt: str, tool_names: set[str], offered: set[str], tools: list,
                     memory: Callable | None = None, num_ctx: int = 8192, deadline: float | None = None,
                     claim_check: bool | None = None) -> list[Any]:
    """One middleware per name, in order (the first wraps the rest). ``tool_names``: the model's tools;
    ``offered``: every tool of the endpoint; ``tools``: the LangChain tools (job_wait calls wait_for_job)."""
    makers: dict[str, Callable[[], Any]] = {
        # Skills that match the request and the progress, and memory notes, in the system prompt.
        "skills": lambda: SkillMiddleware(prompt, tool_names, offered, memory=memory, num_ctx=num_ctx),
        # A job of any length is one tool round.
        "job_wait": lambda: JobWaitMiddleware(tools, deadline=deadline),
        "empty_reply": EmptyReplyMiddleware,
        # Last by default, so it wraps the model call itself: a final answer is checked against the run's tool calls.
        "claim_check": lambda: ClaimCheckMiddleware(tool_names, enabled=claim_check),
    }
    unknown = [n for n in names if n not in makers]
    if unknown:
        raise ValueError(f"unknown middleware {unknown}; known: {list(makers)}")
    return [makers[name]() for name in names]
