"""The shipped blocks. Each does one thing through MCP tools and workflows connect them.

Tool names are glob patterns like ``*_move_stage`` because each instrument's tools carry its own prefix.
"""

from __future__ import annotations

import math

from asyncroscopy.core.map_analysis import AnalysisFailed, stitch_map
from asyncroscopy.core.names import DATA_KEY_IN_TEXT

from .blocks import Block, BlockContext, BlockFailed, register_block

MAX_TILES = 2000
MOVES = ("image_shift", "stage")  # the two ways to reach a tile
DEFAULT_MAX_SHIFT_M = 1.0e-6  # per axis, same as scanning_beam.DEFAULT_MAX_IMAGE_SHIFT


def _move_by(args: dict) -> str:
    by = str(args.get("move_by") or "stage")
    if by not in MOVES:
        raise BlockFailed(f"move_by must be one of {', '.join(MOVES)}, not {by!r}")
    return by


def _numbers(value, count: int, what: str) -> list[float]:
    if isinstance(value, dict):
        value = value.get("position") or value.get("stage") or list(value.values())
    if not isinstance(value, (list, tuple)) or len(value) < count or not all(isinstance(v, (int, float)) for v in value[:count]):
        raise BlockFailed(f"{what} was not {count} numbers: {str(value)[:200]}")
    return [float(v) for v in value]


async def _plan_grid(ctx: BlockContext, args: dict, previous: dict | None) -> dict:
    size, fov, overlap = float(args["size_m"]), float(args["fov_m"]), float(args["overlap"])
    if not 0.0 <= overlap < 0.9:
        raise BlockFailed(f"overlap {overlap} must be from 0 to below 0.9")
    if fov <= 0 or size <= 0:
        raise BlockFailed("size_m and fov_m must be above 0")
    stage = _numbers(await ctx.call("*_get_stage"), 5, "the stage position")
    centre = args.get("centre_xy_m") or stage[:2]
    step = fov * (1.0 - overlap)
    asked = int(args.get("per_side") or 0)  # when given, the tile count sets the size
    if asked < 0:
        raise BlockFailed(f"per_side {asked} must be 0 (work it out from size_m) or above")
    per_side = asked or (max(1, math.ceil((size - fov) / step - 1e-9) + 1) if size > fov else 1)
    if per_side * per_side > MAX_TILES:
        raise BlockFailed(f"a {size:g} m square at a {fov:g} m field of view is {per_side * per_side} tiles, over the limit of {MAX_TILES}; use a larger fov_m")
    start = -(per_side - 1) * step / 2.0
    tiles = []
    for row in range(per_side):
        cols = range(per_side) if row % 2 == 0 else range(per_side - 1, -1, -1)  # snake order keeps moves short
        for col in cols:
            tiles.append([centre[0] + start + col * step, centre[1] + start + row * step])
    start_shift = [0.0, 0.0]
    start_fov = await ctx.call("*_get_fov")  # the run puts this back when it ends
    if isinstance(start_fov, bool) or not isinstance(start_fov, (int, float)) or not start_fov > 0:
        raise BlockFailed(f"the field of view was not a number above 0: {str(start_fov)[:200]}")
    if _move_by(args) == "image_shift":
        # Refuse a grid the image shift cannot reach now, before anything moves.
        reach = await ctx.call("*_get_image_shift_limit")
        if isinstance(reach, bool) or not isinstance(reach, (int, float)) or not (math.isfinite(reach) and reach > 0):
            raise BlockFailed(f"the instrument's image shift limit was not a number above 0: {str(reach)[:200]}")
        asked_limit = float(args.get("max_shift_m") or 0.0)
        limit = min(float(reach), asked_limit) if asked_limit > 0 else float(reach)
        furthest = max(max(abs(x - stage[0]), abs(y - stage[1])) for x, y in tiles)
        if furthest > limit * (1 + 1e-9):
            raise BlockFailed(f"this grid needs an image shift of {furthest:.3g} m, over the limit of {limit:.3g} m per axis: "
                              f"map a square of at most {2 * limit + fov:.3g} m around the stage position, or use move_by stage")
        start_shift = _numbers(await ctx.call("*_get_image_shift"), 2, "the image shift")[:2]
    return {"tiles": tiles, "count": len(tiles), "rows": per_side, "cols": per_side, "stage_rest": stage[2:5], "start_xy_m": stage[:2],
            "start_shift_m": start_shift, "start_fov_m": float(start_fov)}


register_block(Block(
    name="plan_grid",
    summary="Plan a square grid of overlapping tiles around the current stage position (or a given centre). Moves nothing.",
    run=_plan_grid,
    inputs={
        "size_m": {"type": "number", "doc": "Side of the square to cover, metres."},
        "fov_m": {"type": "number", "doc": "Field of view of one tile, metres."},
        "overlap": {"type": "number", "default": 0.3, "doc": "Fraction of a tile shared with its neighbour."},
        "per_side": {"type": "integer", "default": 0, "doc": "Tiles along each side (4 for a 4 by 4 grid); then size_m is not used. 0: as many as size_m needs."},
        "centre_xy_m": {"type": "list", "default": None, "doc": "[x, y] in metres; leave out for the current position."},
        "move_by": {"type": "string", "default": "stage", "doc": "How tiles will be reached: image_shift (the image shift is set, the stage stays still) or stage."},
        "max_shift_m": {"type": "number", "default": 0.0, "doc": "With image_shift: the largest shift per axis, metres; a grid that needs more is "
                                                                 "refused. 0: the instrument's own limit (get_image_shift_limit); never more than it."},
    },
    outputs={"tiles": "list", "count": "integer", "rows": "integer", "cols": "integer", "stage_rest": "list", "start_xy_m": "list",
             "start_shift_m": "list", "start_fov_m": "number"},
    tools=("*_get_stage", "*_get_image_shift", "*_get_image_shift_limit", "*_get_fov"),
))


SCAN_BUSY = ("scantask is already active", "scan task is already active")


async def _put_back(ctx: BlockContext, pattern: str, arguments: dict) -> None:
    """Restore a setting even on stop or failure; a failed restore goes to ``ctx.unrestored`` instead of raising."""
    try:
        await ctx.call_always(pattern, arguments)
    except Exception as exc:
        ctx.unrestored.append({"tool": pattern, "arguments": dict(arguments), "error": str(exc)[:300]})


async def _expose(ctx: BlockContext, args: dict, index: int) -> str:
    """Unblank the beam, take one scanned image and blank again. Returns the image's data key."""
    key, failure = None, None
    try:
        await ctx.call("*_unblank_beam")
        key = await ctx.call("*_acquire_scanned_image", {"detector_list": list(args["detectors"])})
    except BlockFailed as exc:
        if any(text in str(exc).lower() for text in SCAN_BUSY):
            failure = BlockFailed(f"tile {index} was not taken: a scan is already running on the microscope (its own software's live view or an "
                                  "acquisition). Stop that scan at the microscope, then resume this run; nothing is repeated. "
                                  f"The instrument said: {str(exc)[:300]}")
            failure.__cause__ = exc
        else:
            failure = exc
    except BaseException as exc:
        failure = exc
    # Blank even after a failure; the error still says why the tile failed.
    try:
        await ctx.call_always("*_blank_beam")
    except Exception as blank_error:
        if failure is not None:
            raise BlockFailed(f"{failure} (and the beam could not be blanked after it: {str(blank_error)[:300]})") from failure
        taken = f"; tile {index} was taken as {key}, and is taken again when the run is resumed" if isinstance(key, str) else ""
        raise BlockFailed(f"the beam could not be blanked after tile {index}{taken}: {str(blank_error)[:300]}") from blank_error
    if failure is not None:
        raise failure
    if not isinstance(key, str):
        raise BlockFailed(f"the acquisition at tile {index} gave no data key: {str(key)[:200]}")
    return key


async def _acquire_tile(ctx: BlockContext, args: dict, previous: dict | None) -> dict:
    tiles = args["tiles"]
    index = int(previous["index"]) + 1 if previous else 0
    keys = list(previous["keys"]) if previous else []
    taken_at = list(previous.get("stage_xy_m") or []) if previous else []
    if index >= len(tiles):
        return {"index": index - 1, "remaining": 0, "keys": keys, "last_key": keys[-1] if keys else None, "stage_xy_m": taken_at}
    x, y = _numbers(tiles[index], 2, f"tile {index}")
    rest = _numbers(args["stage_rest"], 3, "stage_rest")
    by_shift = _move_by(args) == "image_shift"
    home_shift = _numbers(args.get("start_shift_m") or [0.0, 0.0], 2, "start_shift_m")[:2]
    home_fov = args.get("start_fov_m")
    try:
        # Set on every tile because a run that failed put the old field of view back before it was resumed.
        await ctx.call("*_set_fov", {"fov": float(args["fov_m"])})
        if by_shift:
            origin = _numbers(args.get("origin_xy_m"), 2, "origin_xy_m (plan_grid's start_xy_m)")
            try:
                await ctx.call("*_set_image_shift", {"shift": [x - origin[0], y - origin[1]]})
                shift = _numbers(await ctx.call("*_get_image_shift"), 2, f"the image shift at tile {index}")[:2]
                reported = [origin[0] + shift[0], origin[1] + shift[1]]
                key = await _expose(ctx, args, index)
            finally:
                # Put the image shift back after every tile so a stopped run never leaves it shifted.
                await _put_back(ctx, "*_set_image_shift", {"shift": home_shift})
        else:
            await ctx.call("*_move_stage", {"position": [x, y, *rest]})
            reported = _numbers(await ctx.call("*_get_stage"), 2, f"the stage position at tile {index}")[:2]
            key = await _expose(ctx, args, index)
    except BaseException:  # on any failure or stop, put back the field of view and the stage
        if not by_shift and args.get("origin_xy_m"):
            home_xy = _numbers(args["origin_xy_m"], 2, "origin_xy_m (plan_grid's start_xy_m)")[:2]
            await _put_back(ctx, "*_move_stage", {"position": [*home_xy, *rest]})
        if home_fov:
            await _put_back(ctx, "*_set_fov", {"fov": float(home_fov)})
        raise
    keys.append(key)
    taken_at.append(reported)
    return {"index": index, "remaining": len(tiles) - index - 1, "keys": keys, "last_key": key, "stage_xy_m": taken_at}


register_block(Block(
    name="acquire_tile",
    summary="Take the next tile of a planned grid: go there (by image shift, or by moving the stage), read where it is, unblank, acquire one scanned image, blank; "
            "an image shift is put back after every tile, and the field of view if the tile fails. "
            "Visit it in a loop while its 'remaining' output is above 0.",
    run=_acquire_tile,
    inputs={
        "tiles": {"type": "list", "doc": "plan_grid's tiles."},
        "fov_m": {"type": "number", "doc": "Field of view of one tile, metres."},
        "stage_rest": {"type": "list", "doc": "[z, alpha, beta] kept for every move (plan_grid's stage_rest)."},
        "detectors": {"type": "list", "default": ["haadf"], "doc": "Detectors of the scanned image."},
        "move_by": {"type": "string", "default": "stage", "doc": "image_shift: set the image shift for each tile and leave the stage still; stage: move the stage."},
        "origin_xy_m": {"type": "list", "default": None, "doc": "With image_shift: the stage [x, y] the shifts are measured from (plan_grid's start_xy_m)."},
        "start_shift_m": {"type": "list", "default": None, "doc": "With image_shift: the image shift to go back to after every tile (plan_grid's start_shift_m); left out: [0, 0]."},
        "start_fov_m": {"type": "number", "default": None, "doc": "The field of view to put back if the tile fails or the run is stopped (plan_grid's start_fov_m)."},
    },
    outputs={"index": "integer", "remaining": "integer", "keys": "list", "last_key": "string", "stage_xy_m": "list"},
    tools=("*_set_fov", "*_move_stage", "*_get_stage", "*_set_image_shift", "*_get_image_shift", "*_unblank_beam", "*_acquire_scanned_image",
           "*_blank_beam"),
    always=("*_blank_beam",),
    acts=True,
))


async def _return_stage(ctx: BlockContext, args: dict, previous: dict | None) -> dict:
    x, y = _numbers(args["xy_m"], 2, "xy_m")
    await ctx.call("*_move_stage", {"position": [x, y, *_numbers(args["stage_rest"], 3, "stage_rest")]})
    return {"xy_m": [x, y]}


register_block(Block(
    name="return_stage",
    summary="Move the stage to a position, usually back to where the workflow started.",
    run=_return_stage,
    inputs={"xy_m": {"type": "list", "doc": "[x, y] in metres."}, "stage_rest": {"type": "list", "doc": "[z, alpha, beta]."}},
    outputs={"xy_m": "list"},
    tools=("*_move_stage",),
    acts=True,
))


async def _return_start(ctx: BlockContext, args: dict, previous: dict | None) -> dict:
    if args.get("fov_m"):
        await ctx.call("*_set_fov", {"fov": float(args["fov_m"])})
    if _move_by(args) == "image_shift":
        shift = _numbers(args.get("shift_m"), 2, "shift_m (plan_grid's start_shift_m)")[:2]
        await ctx.call("*_set_image_shift", {"shift": shift})
        return {"xy_m": _numbers(args["xy_m"], 2, "xy_m")[:2], "shift_m": shift}
    out = await _return_stage(ctx, args, previous)
    return {**out, "shift_m": []}


register_block(Block(
    name="return_start",
    summary="Put the instrument back where a tile loop started: the image shift it had (move_by image_shift) or the stage position (move_by stage), and the field of view.",
    run=_return_start,
    inputs={"xy_m": {"type": "list", "doc": "The stage [x, y] at the start, metres (plan_grid's start_xy_m)."},
            "stage_rest": {"type": "list", "doc": "[z, alpha, beta]."},
            "move_by": {"type": "string", "default": "stage", "doc": "image_shift or stage: what the tile loop moved."},
            "shift_m": {"type": "list", "default": None, "doc": "With image_shift: the image shift at the start (plan_grid's start_shift_m)."},
            "fov_m": {"type": "number", "default": None, "doc": "The field of view to put back (plan_grid's start_fov_m); left out: it stays as the tiles had it."}},
    outputs={"xy_m": "list", "shift_m": "list"},
    tools=("*_move_stage", "*_set_image_shift", "*_set_fov"),
    acts=True,
))


async def _collect_run(ctx: BlockContext, args: dict, previous: dict | None) -> dict:
    request = {"image_keys": list(args["keys"]), "plan": dict(args.get("plan") or {})}
    if args.get("stage_xy_m") and args.get("fov_m"):  # used to place images that record no position
        request.update(stage_xy_m=[list(xy) for xy in args["stage_xy_m"]], fov_m=float(args["fov_m"]))
    spectra = _spectra_by_tile(request["image_keys"], args.get("spectrum_keys") or [], args.get("spectrum_tiles") or [])
    if any(spectra):
        if not args.get("spectra_at"):
            raise BlockFailed("spectrum_keys needs spectra_at, where on each tile the spectra were taken")
        request.update(spectrum_keys=spectra, spectra_at=_numbers(args["spectra_at"], 2, "spectra_at")[:2])
    reply = await ctx.analyse("collect_map_run", request)
    if not isinstance(reply, dict) or "run" not in reply:
        raise BlockFailed(f"collect_map_run gave no run: {str(reply)[:300]}")
    return {"run": reply["run"], "tiles": int(reply.get("tiles", len(args["keys"]))), "spectra": int(reply.get("spectra") or 0),
            "note": str(reply.get("note") or "")}


def _spectra_by_tile(image_keys: list, spectrum_keys: list, spectrum_tiles: list) -> list[str]:
    """One entry per image: the key of the spectrum taken on it, "" for none. Spectra on tiles of another run are left out."""
    if len(spectrum_keys) != len(spectrum_tiles):
        raise BlockFailed(f"spectrum_keys has {len(spectrum_keys)} keys for {len(spectrum_tiles)} tiles in spectrum_tiles; give the tile of each")
    position = {str(key): k for k, key in enumerate(image_keys)}
    out = [""] * len(image_keys)
    for spectrum, tile in zip(spectrum_keys, spectrum_tiles):
        k = position.get(str(tile))
        if not spectrum or k is None:
            continue
        if out[k]:
            raise BlockFailed(f"tile {tile} has two spectra, {out[k]} and {spectrum}; a run keeps one per tile")
        out[k] = str(spectrum)
    return out


register_block(Block(
    name="collect_run",
    summary="Gather acquired tile images into one map run that can be stitched.",
    run=_collect_run,
    inputs={"keys": {"type": "list", "doc": "The tiles' data keys (acquire_tile's keys)."},
            "plan": {"type": "object", "default": None, "doc": "Facts to keep with the run (size, field of view)."},
            "stage_xy_m": {"type": "list", "default": None, "doc": "Where the stage was for each tile (acquire_tile's stage_xy_m): places tiles that record no position."},
            "fov_m": {"type": "number", "default": None, "doc": "Field of view of every tile, metres; needed with stage_xy_m."},
            "spectrum_keys": {"type": "list", "default": None, "doc": "Keys of EDS point spectra taken on the tiles (spectrum_here's keys); the map marks each where it was taken."},
            "spectrum_tiles": {"type": "list", "default": None, "doc": "The tile each of spectrum_keys was taken on (spectrum_here's at)."},
            "spectra_at": {"type": "list", "default": None, "doc": "Where on each tile the spectra were taken: [x, y] fractions, x from the left, y from the top."}},
    outputs={"run": "string", "tiles": "integer", "spectra": "integer", "note": "string"},
    analysis_tools=("collect_map_run",),
))


async def _stitch_map(ctx: BlockContext, args: dict, previous: dict | None) -> dict:
    try:
        stitched = await stitch_map(ctx.analyse, str(args["run"]), refine=bool(args["refine"]), stage_sigma_m=float(args["stage_sigma_m"]) if args.get("stage_sigma_m") else None)
    except AnalysisFailed as exc:
        raise BlockFailed(f"stitching {args['run']} failed: {exc}") from exc
    return {"map": stitched["map"], "summary": stitched["summary"]}


register_block(Block(
    name="stitch_map",
    summary="Stitch a map run into one zoomable map and wait for it.",
    run=_stitch_map,
    inputs={"run": {"type": "string", "doc": "The run's key (collect_run's run)."},
            "refine": {"type": "boolean", "default": True, "doc": "Refine tile positions by matching overlaps; turn off for blank samples."},
            "stage_sigma_m": {"type": "number", "default": None, "doc": "1-sigma error of the stage positions, metres; bounds the overlap search. 0 or left out: 5 % of the field of view."}},
    outputs={"map": "string", "summary": "object"},
    analysis_tools=("assemble_map", "analysis_job_status", "analysis_job_results"),
))


SERVERS = ("instrument", "analysis")


async def _run_tool(ctx: BlockContext, args: dict, previous: dict | None) -> dict:
    arguments = args.get("arguments") or {}
    if not isinstance(arguments, dict):
        raise BlockFailed(f"arguments must be a mapping of the tool's arguments, not {str(arguments)[:200]}")
    server = str(args.get("server") or "instrument")
    if server not in SERVERS:
        raise BlockFailed(f"server must be one of {', '.join(SERVERS)}, not {server!r}")
    result = await (ctx.analyse if server == "analysis" else ctx.call)(str(args["tool"]), arguments)
    return {"result": result, "key": result if isinstance(result, str) and DATA_KEY_IN_TEXT.fullmatch(result) else None}


register_block(Block(
    name="run_tool",
    summary="Call one tool of the instrument endpoint (or of the analysis server) with fixed arguments: the general step, for anything "
            "no other block does. The tool is named in the workflow itself, so whoever approves it sees every tool it can call. "
            "Two of these on parallel branches are called at the same time.",
    run=_run_tool,
    inputs={
        "tool": {"type": "string", "doc": "The tool's name, written out (not taken from an input or another node). A leading *_ stands for "
                                          "the instrument's own prefix: *_acquire_spectrum."},
        "arguments": {"type": "object", "default": None, "doc": 'The tool\'s arguments as {name: value}; a value may be "{inputs.name}" or "{node.output}".'},
        "server": {"type": "string", "default": "instrument", "doc": "instrument or analysis: which server has the tool."},
    },
    outputs={"result": "any", "key": "string"},
    acts=True,
    tool_input="tool",
))


async def _ask_person(ctx: BlockContext, args: dict, previous: dict | None) -> dict:
    question = str(args.get("question") or "").strip()
    if not question:
        raise BlockFailed("question is empty: say what the person is to decide")
    options = [str(o) for o in (args.get("options") or [])]
    answer = ctx.ask({"question": question, "options": options, "show": args.get("show")})
    if options:
        chosen = next((o for o in options if o.strip().lower() == str(answer).strip().lower()), None)
        if chosen is None:
            raise BlockFailed(f"the answer {str(answer)[:200]!r} is not one of {options}")
        answer = chosen
    return {"answer": answer, "question": question}


register_block(Block(
    name="ask_person",
    summary="Stop and ask a person a question (a go-ahead, a choice, a recommendation), and wait for the answer. Nothing runs while "
            'it waits. Branch on the answer with a condition: ok.answer == "yes".',
    run=_ask_person,
    inputs={
        "question": {"type": "string", "doc": "What the person is to decide, in a sentence they can answer without reading the workflow."},
        "options": {"type": "list", "default": None, "doc": 'The answers allowed, as text (["yes", "no"]); left out: any answer.'},
        "show": {"type": "any", "default": None, "doc": 'What the person should look at first, usually a data key: "{image.key}".'},
    },
    outputs={"answer": "any", "question": "string"},
))
