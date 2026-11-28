"""An EDS spectrum at a stage position already known, and the elements it shows.

    START -> move -> spectrum -> peaks -> END

The stage moves to the point (keeping its height and tilts), the beam is parked on it, the spectrum is taken and
the beam blanked, then the spectrum's peaks are read. The position comes from somewhere exact: a region of a map, a
point of interest, coordinates the user gave.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from asyncroscopy.workflows.graph import Steps, Workflow, WorkflowState

INPUTS = {
    "x_m": {"type": "number", "min": -3.0e-3, "max": 3.0e-3, "doc": "Stage x of the point, metres, copied exactly from where it came from."},
    "y_m": {"type": "number", "min": -3.0e-3, "max": 3.0e-3, "doc": "Stage y of the point, metres."},
}


def build(use: Steps) -> StateGraph:
    graph = StateGraph(WorkflowState)

    # Nodes. "{node.output}" is an earlier node's output, "{inputs.name}" a run input.
    graph.add_node("move", use.block("move_to", x_m="{inputs.x_m}", y_m="{inputs.y_m}"))
    graph.add_node("spectrum", use.block(
        "spectrum_at",
        x_m="{inputs.x_m}",
        y_m="{inputs.y_m}",
        tool="*_acquire_spectrum",
        arguments={"detector_name": "eds"},
    ))
    graph.add_node("peaks", use.block("read_spectrum", key="{spectrum.key}"))

    # Edges
    graph.add_edge(START, "move")
    graph.add_edge("move", "spectrum")
    graph.add_edge("spectrum", "peaks")
    graph.add_edge("peaks", END)
    return graph


WORKFLOW = Workflow(
    name="spectrum_at_point",
    summary="Take an EDS spectrum at a known stage position (a map region, a point of interest, given coordinates): move there, park "
            "the beam, take the spectrum, blank the beam, and read which elements it shows.",
    inputs=INPUTS,
    budget={"max_seconds": 600, "max_tool_calls": 20},
    build=build,
)
