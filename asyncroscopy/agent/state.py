"""Graph state schemas shared by the agent graphs."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Sequence, TypedDict

from langchain_core.messages import BaseMessage


class SupervisorState(TypedDict):
    """State of the supervisor/worker swarm graph."""

    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_agent: str
    current_task: str


class SurveyState(TypedDict, total=False):
    """State of the deterministic image-then-EDS survey workflow."""

    detector: str
    spectrum_detector: str
    image_key: str
    spectrum_key: str
    image_info: dict[str, Any]
    spectrum_info: dict[str, Any]
    summary: str
    errors: list[str]
