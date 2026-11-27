"""A 4 by 4 map at the current magnification. It takes no settings.

    START -> fov -> plan -> tile --(tile.remaining > 0)--> tile
                             |
                             +--(otherwise)--> run -> stitch -> home -> END

The field of view is read from the microscope and sixteen tiles are laid around the current stage position.
Tiles are reached by image shift and each is checked against the tile before it. A tile that does not fit
gets a bridging image closer to the last one. If the field of view is too large for image shift alone, the
stage moves between groups of tiles.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from ..graph import Steps, Workflow, WorkflowState

INPUTS = {
    "overlap": {"type": "number", "default": 0.3, "min": 0.05, "max": 0.5, "doc": "Fraction of a tile shared with its neighbour."},
}


def build(use: Steps) -> StateGraph:
    graph = StateGraph(WorkflowState)

    # Nodes. "{node.output}" is an earlier node's output, "{inputs.name}" a run input.
    graph.add_node("fov", use.block("run_tool", tool="*_get_fov"))
    graph.add_node("plan", use.block("plan_survey", size_m="{fov.result}", fov_m="{fov.result}", per_side=4, overlap="{inputs.overlap}"))
    graph.add_node("tile", use.block(
        "survey_tiles",
        levels="{plan.levels}",
        centre_xy_m="{plan.centre_xy_m}",
        stage_rest="{plan.stage_rest}",
        start_shift_m="{plan.start_shift_m}",
        start_fov_m="{plan.start_fov_m}",
    ))
    graph.add_node("run", use.block(
        "collect_run",
        keys="{tile.keys}",
        stage_xy_m="{tile.stage_xy_m}",
        fov_m="{tile.fov_m}",
        plan={"workflow": "map_4x4", "fov_m": "{tile.fov_m}", "overlap": "{inputs.overlap}"},
    ))
    graph.add_node("stitch", use.block("stitch_map", run="{run.run}"))
    graph.add_node("home", use.block(
        "return_home",
        xy_m="{plan.centre_xy_m}",
        stage_rest="{plan.stage_rest}",
        shift_m="{plan.start_shift_m}",
        fov_m="{plan.start_fov_m}",
        move_stage="{plan.moves_stage}",
    ))

    graph.add_edge(START, "fov")
    graph.add_edge("fov", "plan")
    graph.add_edge("plan", "tile")
    graph.add_conditional_edges("tile", use.when("tile.remaining > 0", then="tile", otherwise="run", max_visits=40))
    graph.add_edge("run", "stitch")
    graph.add_edge("stitch", "home")
    graph.add_edge("home", END)
    return graph


WORKFLOW = Workflow(
    name="map_4x4",
    summary="Map a 4 by 4 grid of scanned images around the current stage position at the current field of view, checking each tile "
            "against the one before it as it is taken, then stitch it into one map. Takes no settings.",
    inputs=INPUTS,
    budget={"max_seconds": 3600, "max_tool_calls": 800},
    build=build,
)
