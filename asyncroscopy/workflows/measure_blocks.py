"""Blocks for measuring at one place: go there, image, find a feature, take a spectrum on a stage position, read it.

The measurement workflows (``agent/workflows/spectrum_at_point.py``, ``spectrum_on_feature.py``,
``spectrum_at_best_region.py``) are built from these. Every block that unblanks the beam blanks it again
before it returns, even when the acquisition fails, and declares ``*_blank_beam`` as an ``always`` tool so the
runner blanks it at the end of any run.
"""

from __future__ import annotations

from asyncroscopy.core.names import DATA_KEY_IN_TEXT

from .blocks import Block, BlockContext, BlockFailed, register_block
from .shipped_blocks import _numbers

PEAKS_CUT = "data (sums"  # cuts the channel table that crowds out the peaks
PEAKS_CHARS = 3000
ELEMENTS_LINE = "elements most consistent with these peaks"


def _data_key(result, what: str) -> str:
    if not (isinstance(result, str) and DATA_KEY_IN_TEXT.fullmatch(result)):
        raise BlockFailed(f"{what} gave no data key: {str(result)[:300]}")
    return result


def _checked(reply, tool: str) -> dict:
    """An analysis reply as a dict, failing on a structured error ({"error": {...}})."""
    if not isinstance(reply, dict):
        raise BlockFailed(f"{tool} replied {str(reply)[:300]}")
    if reply.get("error"):
        error = reply["error"]
        raise BlockFailed(f"{tool} refused: {error.get('message', error) if isinstance(error, dict) else error}")
    return reply


async def _move_to(ctx: BlockContext, args: dict, previous: dict | None) -> dict:
    stage = _numbers(await ctx.call("*_get_stage"), 5, "the stage position")
    x, y = float(args["x_m"]), float(args["y_m"])
    await ctx.call("*_move_stage", {"position": [x, y, *stage[2:5]]})
    return {"xy_m": [x, y], "from_xy_m": stage[:2], "stage_rest": stage[2:5]}


register_block(Block(
    name="move_to",
    summary="Move the stage to a stage position [x_m, y_m], keeping its height and tilts.",
    run=_move_to,
    inputs={"x_m": {"type": "number", "doc": "Stage x, metres."},
            "y_m": {"type": "number", "doc": "Stage y, metres."}},
    outputs={"xy_m": "list", "from_xy_m": "list", "stage_rest": "list"},
    tools=("*_get_stage", "*_move_stage"),
    acts=True,
))


async def _zoom(ctx: BlockContext, args: dict, previous: dict | None) -> dict:
    was = previous["was"] if previous else await ctx.call("*_get_fov")
    if isinstance(was, bool) or not isinstance(was, (int, float)) or not was > 0:
        raise BlockFailed(f"the field of view was not a number above 0: {str(was)[:200]}")
    fov = float(args["fov_m"])
    ctx.changed("fov", {"tool": "*_set_fov", "arguments": {"fov": fov}}, {"tool": "*_set_fov", "arguments": {"fov": float(was)}})
    await ctx.call("*_set_fov", {"fov": fov})
    return {"was": float(was), "fov_m": fov}


register_block(Block(
    name="zoom",
    summary="Set the field of view, remembering the one before. If the run fails or is stopped before zoom_back, the runner puts the "
            "old one back; a resumed run sets this one again.",
    run=_zoom,
    inputs={"fov_m": {"type": "number", "doc": "The field of view to set, metres."}},
    outputs={"was": "number", "fov_m": "number"},
    tools=("*_get_fov", "*_set_fov"),
    acts=True,
))


async def _zoom_back(ctx: BlockContext, args: dict, previous: dict | None) -> dict:
    fov = float(args["fov_m"])
    await ctx.call("*_set_fov", {"fov": fov})
    ctx.put_back_done("fov")
    return {"fov_m": fov}


register_block(Block(
    name="zoom_back",
    summary="Put back the field of view zoom found: zoom_back with fov_m \"{zoom.was}\".",
    run=_zoom_back,
    inputs={"fov_m": {"type": "number", "doc": "The field of view to put back: \"{zoom.was}\"."}},
    outputs={"fov_m": "number"},
    tools=("*_set_fov",),
    acts=True,
))


async def _image(ctx: BlockContext, args: dict, previous: dict | None) -> dict:
    try:
        await ctx.call("*_unblank_beam")
        result = await ctx.call("*_acquire_scanned_image", {"detector_list": list(args.get("detectors") or ["haadf"])})
    finally:
        await ctx.call_always("*_blank_beam")
    return {"key": _data_key(result, "the image")}


register_block(Block(
    name="image",
    summary="Take one scanned image at the current field of view: unblank, acquire, blank (also when the acquisition fails).",
    run=_image,
    inputs={"detectors": {"type": "list", "default": ["haadf"], "doc": "The detectors to read: [\"haadf\"]."}},
    outputs={"key": "string"},
    tools=("*_unblank_beam", "*_acquire_scanned_image", "*_blank_beam"),
    always=("*_blank_beam",),
    acts=True,
))


async def _find_feature(ctx: BlockContext, args: dict, previous: dict | None) -> dict:
    reply = _checked(await ctx.analyse("find_features", {"image": args["image"], "n": int(args.get("n") or 1)}), "find_features")
    features = [f for f in reply.get("features") or [] if isinstance(f, dict) and "x_m" in f and "y_m" in f]
    if not features:
        why = reply.get("hint") or "the image shows nothing distinct (bare support or noise)"
        raise BlockFailed(f"no feature found in {args['image']}: {why}")
    best = features[0]
    return {"x_m": float(best["x_m"]), "y_m": float(best["y_m"]), "feature": best, "features": features}


register_block(Block(
    name="find_feature",
    summary="Find the most distinct features (particles, clusters) of an image, strongest first, with their stage positions. "
            "Fails, before anything else moves, when the image shows nothing distinct.",
    run=_find_feature,
    inputs={"image": {"type": "string", "doc": "The image's key: \"{survey.key}\"."},
            "n": {"type": "integer", "default": 1, "doc": "How many features to list."}},
    outputs={"x_m": "number", "y_m": "number", "feature": "object", "features": "list"},
    analysis_tools=("find_features",),
))


async def _spectrum_at(ctx: BlockContext, args: dict, previous: dict | None) -> dict:
    x, y = float(args["x_m"]), float(args["y_m"])
    arguments = args.get("arguments") or {}
    if not isinstance(arguments, dict):
        raise BlockFailed(f"arguments must be a mapping of the tool's arguments, not {str(arguments)[:200]}")
    try:
        await ctx.call("*_unblank_beam")
        await ctx.call("*_place_beam_at", {"position": [x, y]})
        result = await ctx.call(str(args["tool"]), arguments)
    finally:
        await ctx.call_always("*_blank_beam")  # never leave the beam parked on the sample
    return {"key": _data_key(result, str(args["tool"])), "xy_m": [x, y]}


register_block(Block(
    name="spectrum_at",
    summary="Park the beam on a stage position [x_m, y_m] in the current field of view, unblank, call one acquisition tool (an EDS "
            "or EELS spectrum), and blank. The tool is named in the workflow itself.",
    run=_spectrum_at,
    inputs={"x_m": {"type": "number", "doc": "Stage x of the point, metres."},
            "y_m": {"type": "number", "doc": "Stage y of the point, metres."},
            "tool": {"type": "string", "doc": "The acquisition tool's name, written out: *_acquire_spectrum."},
            "arguments": {"type": "object", "default": None, "doc": 'The tool\'s arguments: {"detector_name": "eds"}.'}},
    outputs={"key": "string", "xy_m": "list"},
    tools=("*_unblank_beam", "*_place_beam_at", "*_blank_beam"),
    always=("*_blank_beam",),
    acts=True,
    tool_input="tool",
))


async def _read_spectrum(ctx: BlockContext, args: dict, previous: dict | None) -> dict:
    text = str(await ctx.call("view_data", {"key": args["key"], "format": "text"}))
    peaks = text.split(PEAKS_CUT, 1)[0].rstrip()[:PEAKS_CHARS]
    elements = next((line.strip() for line in peaks.splitlines() if ELEMENTS_LINE in line), None)
    return {"peaks": peaks, "elements": elements}


register_block(Block(
    name="read_spectrum",
    summary="Read a spectrum's peaks and the elements most consistent with them (view_data as text, without the channel table).",
    run=_read_spectrum,
    inputs={"key": {"type": "string", "doc": "The spectrum's key: \"{spectrum.key}\"."}},
    outputs={"peaks": "string", "elements": "string"},
    tools=("view_data",),
))


async def _best_regions(ctx: BlockContext, args: dict, previous: dict | None) -> dict:
    top_k = int(args.get("top_k") or 3)
    reply = _checked(await ctx.analyse("list_pois", {"map": args["map"], "kind": "candidate", "limit": top_k}), "list_pois")
    regions = [p for p in reply.get("pois") or [] if isinstance(p, dict) and "x_m" in p and "y_m" in p]
    if not regions:
        raise BlockFailed(f"the map {args['map']} has no scored regions: score it first (score_map), or map it with find_regions")
    return {"regions": regions, "x_m": float(regions[0]["x_m"]), "y_m": float(regions[0]["y_m"]), "best": regions[0]}


register_block(Block(
    name="best_regions",
    summary="The best scored regions of a stitched map (its candidate points of interest), best first, with the best one's stage "
            "position. Fails when the map has none.",
    run=_best_regions,
    inputs={"map": {"type": "string", "doc": "The stitched map's key."},
            "top_k": {"type": "integer", "default": 3, "doc": "How many regions to list."}},
    outputs={"regions": "list", "x_m": "number", "y_m": "number", "best": "object"},
    analysis_tools=("list_pois",),
))
