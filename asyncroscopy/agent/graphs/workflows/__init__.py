"""Deterministic LangGraph workflows.

Each module exposes a ``build_*_graph(tools, ...)`` factory returning a compiled
graph. Register new workflows in ``WORKFLOWS`` so the LLM device's
``RunWorkflow`` command and the Studio factories can find them by name.
"""

from __future__ import annotations

from typing import Any, Callable

WORKFLOWS: dict[str, Callable[..., Any]] = {}


def register_workflow(name: str, builder: Callable[..., Any]) -> None:
    WORKFLOWS[name] = builder


def get_workflow_builder(name: str) -> Callable[..., Any]:
    _ensure_builtin_workflows()
    try:
        return WORKFLOWS[name]
    except KeyError as exc:
        available = ", ".join(sorted(WORKFLOWS)) or "<none>"
        raise KeyError(f"Unknown workflow {name!r}. Available workflows: {available}") from exc


def list_workflows() -> list[str]:
    _ensure_builtin_workflows()
    return sorted(WORKFLOWS)


def _ensure_builtin_workflows() -> None:
    if "image_eds_survey" not in WORKFLOWS:
        from asyncroscopy.agent.graphs.workflows import image_eds_survey  # noqa: F401  (registers itself)
