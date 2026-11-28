"""A whole area of the sample mapped at one or several magnifications, with spectra at points along the way.

    START -> plan --(plan.count > ask_over)--> ok --(answer "yes")--> tile
               |                               +--(otherwise)--> END
               +--(otherwise)--> tile

    tile -> eds -> eels --(tile.remaining > 0)--> tile                     one level, a batch of tiles at a time
                     +--(otherwise)--> run -> stitch -> maps
    maps --(tile.left > 0)--> tile                                         the next, finer level
       +--(otherwise)--> home -> END

Level 0 covers ``size_m`` at ``fov_m``. Each deeper level covers the same square at a field of view ``zoom``
times smaller, and every level is stitched into its own map. Within a level the stage moves between groups of
tiles and the tiles in a group are reached by image shift. Each tile is checked against the one before it,
and a tile that does not fit gets a bridging image.

The inner loop is ``tile -> eds -> eels``. ``tile`` takes up to ``batch`` tiles a visit and stops early at a
tile due for spectra, where ``eds`` and ``eels`` each take a spectrum if switched on. To do more at those
points, add a node between ``tile`` and the edge back.

A plan of more than ``ask_over`` images is put to a person before the first move. The run puts the instrument
back as it found it and blanks the beam.

VISITS counts every tile, spectra stop and level as a visit, so any plan plan_survey accepts fits under it.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from asyncroscopy.workflows.graph import Steps, Workflow, WorkflowState
from asyncroscopy.workflows.survey_blocks import MAX_SPECTRA_POINTS, MAX_SURVEY_TILES

VISITS = MAX_SURVEY_TILES + MAX_SPECTRA_POINTS + 10

INPUTS = {
    "size_m": {"type": "number", "min": 1.0e-7, "max": 3.0e-3, "doc": "Side of the square to map around the current stage position, metres."},
    "fov_m": {"type": "number", "min": 2.0e-8, "max": 5.0e-4,
              "doc": "Field of view of one tile of the first, coarsest map, metres. For the current magnification pass what get_fov returns."},
    "depth": {"type": "integer", "default": 0, "min": 0, "max": 3,
              "doc": "How many finer maps to make of the same square after the first. 0: one map. Each multiplies the images by about zoom squared."},
    "zoom": {"type": "number", "default": 4.0, "min": 2.0, "max": 10.0, "doc": "How many times smaller the field of view is at each deeper level."},
    "overlap": {"type": "number", "default": 0.3, "min": 0.05, "max": 0.5, "doc": "Fraction of a tile shared with its neighbour."},
    "max_tiles": {"type": "integer", "default": 2000, "min": 1, "max": 20000,
                  "doc": "The most images the run may take. A plan over it is refused before anything moves, with the count per level."},
    "ask_over": {"type": "integer", "default": 200, "min": 0, "max": 20000,
                 "doc": "A plan of more images than this is put to a person, with its count, before the first move."},
    "check": {"type": "boolean", "default": True, "doc": "Check each tile against the tile before it as it is taken."},
    "retries": {"type": "integer", "default": 1, "min": 0, "max": 3,
                "doc": "Bridging images to try for a tile that does not fit, each closer to the last. On a blank area every tile costs these."},
    "batch": {"type": "integer", "default": 50, "min": 1, "max": 500, "doc": "Tiles taken between checkpoints; a failed batch is retaken from its first tile."},
    "spectra_every": {"type": "integer", "default": 0, "min": 0, "max": 20000,
                      "doc": "Take spectra at every this many tiles (at most 500 points a run). 0: none."},
    "eds": {"type": "boolean", "default": False, "doc": "At those tiles take an EDS spectrum."},
    "eels": {"type": "boolean", "default": False, "doc": "At those tiles take an EELS spectrum (after the EDS one)."},
    "spectra_at": {"type": "list", "default": [0.5, 0.5], "doc": "Where on the tile the beam is parked: [x, y] as fractions of the image, x from the left, y from the top."},
    "max_shift_m": {"type": "number", "default": 1.0e-6, "min": 1.0e-8, "max": 1.0e-6,
                    "doc": "The largest image shift per axis, metres: how far from a stage position a tile may be."},
    "stage_limit_m": {"type": "number", "default": 9.0e-4, "min": 1.0e-6, "max": 1.5e-3,
                      "doc": "The farthest from its zero the stage may be sent on x or y, metres. A plan that needs more is refused before anything moves."},
    "stage_sigma_m": {"type": "number", "default": 0, "min": 0, "max": 5.0e-6,
                      "doc": "1-sigma error of the stage's reported position, metres; bounds the stitch's overlap search. 0: 5 % of the field of view."},
}


def build(use: Steps) -> StateGraph:
    graph = StateGraph(WorkflowState)

    graph.add_node("plan", use.block(
        "plan_survey",
        size_m="{inputs.size_m}",
        fov_m="{inputs.fov_m}",
        overlap="{inputs.overlap}",
        depth="{inputs.depth}",
        zoom="{inputs.zoom}",
        max_shift_m="{inputs.max_shift_m}",
        max_tiles="{inputs.max_tiles}",
        stage_limit_m="{inputs.stage_limit_m}",
        spectra_every="{inputs.spectra_every}",
    ))
    graph.add_node("ok", use.block("ask_person", question="{plan.summary}", options=["yes", "no"]))
    graph.add_node("tile", use.block(
        "survey_tiles",
        levels="{plan.levels}",
        centre_xy_m="{plan.centre_xy_m}",
        stage_rest="{plan.stage_rest}",
        start_shift_m="{plan.start_shift_m}",
        start_fov_m="{plan.start_fov_m}",
        batch="{inputs.batch}",
        check="{inputs.check}",
        retries="{inputs.retries}",
        spectra_every="{inputs.spectra_every}",
        max_shift_m="{inputs.max_shift_m}",
    ))
    graph.add_node("eds", use.block(
        "spectrum_here",
        tool="*_acquire_spectrum",
        arguments={"detector_name": "eds"},
        on="{inputs.eds}",
        position="{inputs.spectra_at}",
        due="{tile.measure}",
        at="{tile.last_key}",
        shift_m="{tile.shift_m}",
        home_shift_m="{plan.start_shift_m}",
    ))
    graph.add_node("eels", use.block(
        "spectrum_here",
        tool="*_acquire_eels_spectrum",
        on="{inputs.eels}",
        position="{inputs.spectra_at}",
        due="{tile.measure}",
        at="{tile.last_key}",
        shift_m="{tile.shift_m}",
        home_shift_m="{plan.start_shift_m}",
    ))
    graph.add_node("run", use.block(
        "collect_run",
        keys="{tile.keys}",
        stage_xy_m="{tile.stage_xy_m}",
        fov_m="{tile.fov_m}",
        spectrum_keys="{eds.keys}",
        spectrum_tiles="{eds.at}",
        spectra_at="{inputs.spectra_at}",
        plan={"workflow": "map_sample", "level": "{tile.level}", "size_m": "{inputs.size_m}", "fov_m": "{tile.fov_m}", "overlap": "{inputs.overlap}"},
    ))
    graph.add_node("stitch", use.block("stitch_map", run="{run.run}", stage_sigma_m="{inputs.stage_sigma_m}"))
    graph.add_node("maps", use.block("keep", value="{stitch.map}"))
    graph.add_node("home", use.block(
        "return_home",
        xy_m="{plan.centre_xy_m}",
        stage_rest="{plan.stage_rest}",
        shift_m="{plan.start_shift_m}",
        fov_m="{plan.start_fov_m}",
        move_stage="{plan.moves_stage}",
    ))

    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", use.when("plan.count > inputs.ask_over", then="ok", otherwise="tile"))
    graph.add_conditional_edges("ok", use.when('ok.answer == "yes"', then="tile", otherwise=END))
    graph.add_edge("tile", "eds")
    graph.add_edge("eds", "eels")
    graph.add_conditional_edges("eels", use.when("tile.remaining > 0", then="tile", otherwise="run", max_visits=VISITS))
    graph.add_edge("run", "stitch")
    graph.add_edge("stitch", "maps")
    graph.add_conditional_edges("maps", use.when("tile.left > 0", then="tile", otherwise="home", max_visits=VISITS))
    graph.add_edge("home", END)
    return graph


WORKFLOW = Workflow(
    name="map_sample",
    summary="Map a large square of the sample, up to thousands of images: one stitched map at fov_m, and with depth a finer map of the "
            "same square for each level. The stage moves between groups of tiles, tiles are reached by image shift and checked as they "
            "are taken, and EDS or EELS spectra can be taken at every so many tiles. A large plan is put to a person before it starts.",
    inputs=INPUTS,
    budget={"max_seconds": 172800, "max_tool_calls": 200000},
    build=build,
)
