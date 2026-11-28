"""Find the most distinct feature (a particle, a grain, a bright spot) near here and take an EDS spectrum on it.

    START -> zoom -> survey -> feature -> spectrum -> peaks -> fov_back -> END

A survey image is taken at ``fov_m`` (several times the size of a particle), its most distinct feature is found,
the beam is parked on it for the spectrum and blanked, and the spectrum's peaks are read. The field of view is put
back at the end, and also when the run fails or is stopped part way. If the survey shows nothing distinct the run stops at ``feature``, before the beam is parked; run
it again with a larger ``fov_m``.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from asyncroscopy.workflows.graph import Steps, Workflow, WorkflowState

INPUTS = {
    "fov_m": {"type": "number", "default": 3.0e-7, "min": 2.0e-8, "max": 5.0e-6,
              "doc": "Field of view of the survey image, metres: several times the feature's size (300 nm for nanoparticles)."},
}


def build(use: Steps) -> StateGraph:
    graph = StateGraph(WorkflowState)

    # Nodes. "{node.output}" is an earlier node's output, "{inputs.name}" a run input.
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

    # Edges
    graph.add_edge(START, "zoom")
    graph.add_edge("zoom", "survey")
    graph.add_edge("survey", "feature")
    graph.add_edge("feature", "spectrum")
    graph.add_edge("spectrum", "peaks")
    graph.add_edge("peaks", "fov_back")
    graph.add_edge("fov_back", END)
    return graph


WORKFLOW = Workflow(
    name="spectrum_on_feature",
    summary="Find the most distinct feature (a particle, a grain, a bright spot) in a survey image at the current position and take an "
            "EDS spectrum on it: survey, find, park the beam, take the spectrum, blank, and read which elements it shows.",
    inputs=INPUTS,
    budget={"max_seconds": 900, "max_tool_calls": 30},
    build=build,
)
