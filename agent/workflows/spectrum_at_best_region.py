"""An EDS spectrum at the best region of a scored map, on the most distinct feature there.

    START -> regions -> move -> zoom -> survey -> feature -> spectrum -> peaks -> fov_back -> END

The map's best regions are listed, the stage moves to the first, and from there it is ``spectrum_on_feature``: a
survey image, its most distinct feature (a region's centre can be bare support), the spectrum on that feature, and
its peaks. A map with no scored regions stops the run at ``regions``, before the stage moves. The field of view is put back at the end, and also when the run fails or is
stopped part way. The stage stays at the region, as it does when the run succeeds.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from asyncroscopy.workflows.graph import Steps, Workflow, WorkflowState

INPUTS = {
    "map": {"type": "string", "doc": "The stitched map's key (scored: made with find_regions, or scored with score_map)."},
    "top_k": {"type": "integer", "default": 3, "min": 1, "max": 20, "doc": "How many of the map's best regions to list."},
    "fov_m": {"type": "number", "default": 3.0e-7, "min": 2.0e-8, "max": 5.0e-6,
              "doc": "Field of view of the survey image at the region, metres."},
}


def build(use: Steps) -> StateGraph:
    graph = StateGraph(WorkflowState)

    # Nodes. "{node.output}" is an earlier node's output, "{inputs.name}" a run input.
    graph.add_node("regions", use.block("best_regions", map="{inputs.map}", top_k="{inputs.top_k}"))
    graph.add_node("move", use.block("move_to", x_m="{regions.x_m}", y_m="{regions.y_m}"))
    graph.add_node("zoom", use.block("zoom", fov_m="{inputs.fov_m}"))
    graph.add_node("survey", use.block("image", detectors=["haadf"]))
    graph.add_node("feature", use.block("find_feature", image="{survey.key}"))
    graph.add_node("spectrum", use.block(
        "spectrum_at",
        x_m="{feature.x_m}",
        y_m="{feature.y_m}",
        tool="*_acquire_spectrum",
        arguments={"detector_name": "eds"},
    ))
    graph.add_node("peaks", use.block("read_spectrum", key="{spectrum.key}"))
    graph.add_node("fov_back", use.block("zoom_back", fov_m="{zoom.was}"))

    graph.add_edge(START, "regions")
    graph.add_edge("regions", "move")
    graph.add_edge("move", "zoom")
    graph.add_edge("zoom", "survey")
    graph.add_edge("survey", "feature")
    graph.add_edge("feature", "spectrum")
    graph.add_edge("spectrum", "peaks")
    graph.add_edge("peaks", "fov_back")
    graph.add_edge("fov_back", END)
    return graph


WORKFLOW = Workflow(
    name="spectrum_at_best_region",
    summary="Take an EDS spectrum at the best region of a scored map: list its best regions, move to the first, find the most "
            "distinct feature there in a survey image, take the spectrum on it, blank, and read which elements it shows.",
    inputs=INPUTS,
    budget={"max_seconds": 900, "max_tool_calls": 40},
    build=build,
)
