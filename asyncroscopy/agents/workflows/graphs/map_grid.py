"""Map a square as a grid of tiles and stitch them into one map.

    START -> plan -> tile --(tile.remaining > 0)--> tile
                      |
                      +--(otherwise)--> home -> run -> stitch -> END

Tiles are reached by image shift by default, so the stage does not move. Use ``move_by: stage`` for a square
larger than the image shift can reach. The run puts the instrument back as it found it and blanks the beam.
Each tile is a checkpoint, so a stopped or failed run continues at its next tile.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from ..graph import Steps, Workflow, WorkflowState

INPUTS = {
    "size_m": {"type": "number", "default": 6.0e-7, "min": 5.0e-8, "max": 2.0e-5,
               "doc": "Side of the square to map, metres. Not used when tiles_per_side is given."},
    "tiles_per_side": {"type": "integer", "default": 0, "min": 0, "max": 44,
                       "doc": "Tiles along each side: 4 for 'a 4 by 4 grid'. Give this, not size_m, when the request names a count. "
                              "0: as many as size_m needs."},
    "fov_m": {"type": "number", "default": 3.0e-7, "min": 2.0e-8, "max": 5.0e-6,
              "doc": "Field of view of one tile, metres. For a map at the current magnification pass what get_fov returns."},
    "overlap": {"type": "number", "default": 0.3, "min": 0.05, "max": 0.5,
                "doc": "Fraction of a tile shared with its neighbour. Lower values leave less to match on."},
    "move_by": {"type": "string", "default": "image_shift",
                "doc": "image_shift: set the image shift for each tile, the stage stays still (the square may reach max_shift_m from the "
                       "stage position). stage: move the stage."},
    "max_shift_m": {"type": "number", "default": 0, "min": 0, "max": 1.0e-4,
                    "doc": "With image_shift: the largest shift per axis, metres. A square that needs more is refused before anything moves. "
                           "0: the instrument's own limit; a value above that limit still gets only the limit."},
    "stage_sigma_m": {"type": "number", "default": 0, "min": 0, "max": 5.0e-6,
                      "doc": "1-sigma error of the stage's reported position, metres; bounds the stitch's overlap search. 0: 5 % of fov_m. "
                             "A real stage may need a larger value."},
}


def build(use: Steps) -> StateGraph:
    graph = StateGraph(WorkflowState)

    # Nodes. "{node.output}" is an earlier node's output, "{inputs.name}" a run input.
    graph.add_node("plan", use.block(
        "plan_grid",
        size_m="{inputs.size_m}",
        per_side="{inputs.tiles_per_side}",
        fov_m="{inputs.fov_m}",
        overlap="{inputs.overlap}",
        move_by="{inputs.move_by}",
        max_shift_m="{inputs.max_shift_m}",
    ))
    graph.add_node("tile", use.block(
        "acquire_tile",
        tiles="{plan.tiles}",
        fov_m="{inputs.fov_m}",
        stage_rest="{plan.stage_rest}",
        move_by="{inputs.move_by}",
        origin_xy_m="{plan.start_xy_m}",
        start_shift_m="{plan.start_shift_m}",
        start_fov_m="{plan.start_fov_m}",
    ))
    graph.add_node("home", use.block(
        "return_start",
        xy_m="{plan.start_xy_m}",
        stage_rest="{plan.stage_rest}",
        move_by="{inputs.move_by}",
        shift_m="{plan.start_shift_m}",
        fov_m="{plan.start_fov_m}",
    ))
    graph.add_node("run", use.block(
        "collect_run",
        keys="{tile.keys}",
        stage_xy_m="{tile.stage_xy_m}",
        fov_m="{inputs.fov_m}",
        plan={"workflow": "map_grid", "size_m": "{inputs.size_m}", "fov_m": "{inputs.fov_m}", "overlap": "{inputs.overlap}"},
    ))
    graph.add_node("stitch", use.block("stitch_map", run="{run.run}", stage_sigma_m="{inputs.stage_sigma_m}"))

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "tile")
    graph.add_conditional_edges("tile", use.when("tile.remaining > 0", then="tile", otherwise="home", max_visits=2000))
    graph.add_edge("home", "run")
    graph.add_edge("run", "stitch")
    graph.add_edge("stitch", END)
    return graph


WORKFLOW = Workflow(
    name="map_grid",
    summary="Map a square around the current stage position as a grid of scanned-image tiles, reached by image shift (the stage stays "
            "still) or by moving the stage, then stitch it into one map. For 'an N by N grid' give tiles_per_side=N.",
    inputs=INPUTS,
    budget={"max_seconds": 7200, "max_tool_calls": 9000},
    build=build,
)
