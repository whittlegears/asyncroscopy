"""Deterministic workflow: acquire a HAADF image, then an EDS spectrum, then summarise.

Every step is a fixed graph node calling a fixed MCP tool. The model (optional)
only rephrases the final summary; the acquisition order and arguments never
depend on it, which is what makes this workflow reproducible.

Nodes:
    acquire_image -> acquire_spectrum -> read_back -> summarize -> END
Any node error is recorded in ``state["errors"]`` and routes straight to
``summarize`` so the run always ends with a report.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from asyncroscopy.agent.graphs.workflows import register_workflow
from asyncroscopy.agent.state import SurveyState
from asyncroscopy.agent.tools import find_tool, tool_result_dict, tool_result_text

WORKFLOW_NAME = "image_eds_survey"
DEFAULT_IMAGE_TOOL = "*_acquire_scanned_image"
DEFAULT_SPECTRUM_TOOL = "*_acquire_spectrum"
DEFAULT_DATA_TOOL = "get_data_from_key"

SUMMARY_SYSTEM_PROMPT = (
    "You summarise microscope acquisitions for an operator. Rewrite the report below in two or three "
    "clear sentences. Keep every number and data key exactly as given; do not invent values."
)


def _first_dataset(info: dict[str, Any], prefer: str | None = None) -> dict[str, Any] | None:
    datasets = info.get("datasets")
    if isinstance(datasets, list) and datasets:
        if prefer:
            for ds in datasets:
                if isinstance(ds, dict) and str(ds.get("name", "")).lower().endswith(prefer.lower()):
                    return ds
        return datasets[0] if isinstance(datasets[0], dict) else None
    return info if "shape" in info or "preview" in info else None


def _elements_from_attrs(attrs: Any) -> list[str] | None:
    if not isinstance(attrs, dict):
        return None
    raw = attrs.get("elements")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return None
    if isinstance(raw, (list, tuple)) and raw and all(isinstance(x, str) for x in raw):
        return list(raw)
    return None


def deterministic_summary(state: SurveyState) -> str:
    """Plain-text report built from the state alone (no model involved)."""
    lines: list[str] = []
    if state.get("image_key"):
        line = f"HAADF image saved as {state['image_key']}"
        ds = _first_dataset(state.get("image_info") or {}, prefer="HAADF")
        if ds and ds.get("shape") is not None:
            line += f" (shape {tuple(ds['shape'])}, dtype {ds.get('dtype', '?')})"
        lines.append(line + ".")
    if state.get("spectrum_key"):
        line = f"EDS spectrum saved as {state['spectrum_key']}"
        ds = _first_dataset(state.get("spectrum_info") or {}, prefer="spectrum")
        if ds:
            elements = _elements_from_attrs(ds.get("attrs"))
            preview = ds.get("preview")
            if elements and isinstance(preview, list) and len(preview) == len(elements):
                composition = ", ".join(f"{el} {float(val):.2f}" for el, val in zip(elements, preview))
                line += f". Composition: {composition}"
            elif ds.get("shape") is not None:
                line += f" (shape {tuple(ds['shape'])})"
        lines.append(line + ".")
    for error in state.get("errors") or []:
        lines.append(f"ERROR: {error}")
    if not lines:
        lines.append("Nothing was acquired.")
    return "\n".join(lines)


def build_image_eds_survey_graph(
    tools: Sequence[Any],
    model: Any | None = None,
    *,
    image_tool: str = DEFAULT_IMAGE_TOOL,
    spectrum_tool: str = DEFAULT_SPECTRUM_TOOL,
    data_tool: str = DEFAULT_DATA_TOOL,
    preview_values: int = 32,
) -> Any:
    """Compile the survey workflow over the given MCP tools.

    Args:
        tools: LangChain tools loaded from the MCP server.
        model: optional chat model used only to rephrase the final summary.
        image_tool/spectrum_tool/data_tool: names or globs of the tools to use.
        preview_values: ``max_values`` passed to ``get_data_from_key``.
    """
    acquire_image_tool = find_tool(tools, image_tool)
    acquire_spectrum_tool = find_tool(tools, spectrum_tool)
    read_tool = find_tool(tools, data_tool)

    def _with_error(state: SurveyState, node: str, exc: BaseException) -> dict:
        return {"errors": list(state.get("errors") or []) + [f"{node}: {type(exc).__name__}: {exc}"]}

    async def acquire_image(state: SurveyState) -> dict:
        try:
            detector = state.get("detector") or "haadf"
            result = await acquire_image_tool.ainvoke({"detector_list": [detector]})
            key = tool_result_text(result)
            if not key:
                raise RuntimeError("acquisition returned no data key")
            return {"image_key": key}
        except Exception as exc:  # noqa: BLE001 - reported in state, run continues to summary
            return _with_error(state, "acquire_image", exc)

    async def acquire_spectrum(state: SurveyState) -> dict:
        try:
            detector = state.get("spectrum_detector") or "eds"
            result = await acquire_spectrum_tool.ainvoke({"detector_name": detector})
            key = tool_result_text(result)
            if not key:
                raise RuntimeError("acquisition returned no data key")
            return {"spectrum_key": key}
        except Exception as exc:  # noqa: BLE001
            return _with_error(state, "acquire_spectrum", exc)

    async def read_back(state: SurveyState) -> dict:
        update: dict = {}
        errors = list(state.get("errors") or [])
        for field, key in (("image_info", state.get("image_key")), ("spectrum_info", state.get("spectrum_key"))):
            if not key:
                continue
            try:
                result = await read_tool.ainvoke({"key": key, "max_values": preview_values})
                update[field] = tool_result_dict(result)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"read_back[{field}]: {type(exc).__name__}: {exc}")
        if errors:
            update["errors"] = errors
        return update

    async def summarize(state: SurveyState) -> dict:
        report = deterministic_summary(state)
        if model is None:
            return {"summary": report}
        try:
            response = await model.ainvoke([SystemMessage(content=SUMMARY_SYSTEM_PROMPT), HumanMessage(content=report)])
            text = response.content if isinstance(response.content, str) else str(response.content)
            text = text.strip()
        except Exception as exc:  # noqa: BLE001 - the deterministic report is always available
            text = f"{report}\n(model summary failed: {exc})"
        return {"summary": text or report}

    def continue_or_summarize(next_node: str):
        def route(state: SurveyState) -> str:
            return "summarize" if state.get("errors") else next_node

        return route

    builder = StateGraph(SurveyState)
    builder.add_node("acquire_image", acquire_image)
    builder.add_node("acquire_spectrum", acquire_spectrum)
    builder.add_node("read_back", read_back)
    builder.add_node("summarize", summarize)
    builder.add_edge(START, "acquire_image")
    builder.add_conditional_edges(
        "acquire_image", continue_or_summarize("acquire_spectrum"), {"acquire_spectrum": "acquire_spectrum", "summarize": "summarize"}
    )
    builder.add_conditional_edges(
        "acquire_spectrum", continue_or_summarize("read_back"), {"read_back": "read_back", "summarize": "summarize"}
    )
    builder.add_edge("read_back", "summarize")
    builder.add_edge("summarize", END)
    return builder.compile(name=WORKFLOW_NAME)


register_workflow(WORKFLOW_NAME, build_image_eds_survey_graph)
