"""Blocks for mapping an area too large for one image-shift grid, at one or several magnifications.

The area is covered level by level. Level 0 is the area at ``fov_m``. Each deeper level is the same area at a field
of view ``zoom`` times smaller. Within a level the stage moves from field to field. Inside a field the tiles are
reached by image shift, which is much more exact than the stage.

Each tile is checked against the tile before it as soon as it is taken, using the analysis server's ``match_tiles``.
A tile that does not fit gets a bridging tile between the two, closer to the last one. This way the overlaps the
stitch needs are known to be good while the instrument is still at that spot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

from asyncroscopy.core.names import DATA_KEY_IN_TEXT

from .blocks import Block, BlockContext, BlockFailed, register_block
from .shipped_blocks import DEFAULT_MAX_SHIFT_M, _expose, _numbers, _put_back

MAX_SURVEY_TILES = 20000
MAX_SPECTRA_POINTS = 500  # each point adds several checkpoints of the whole run
NEIGHBOUR_REACH = 0.9  # share of a field on both axes below which tiles overlap enough to compare
COUNTS = ("taken", "matched", "unmatched", "unchecked", "extra")


def _level(size: float, fov: float, overlap: float, max_shift: float, per_side: int = 0) -> dict:
    step = fov * (1.0 - overlap)
    tiles = per_side or (max(1, math.ceil((size - fov) / step - 1e-9) + 1) if size > fov else 1)
    reach = max(1, math.floor(2.0 * max_shift / step + 1e-9) + 1)  # tiles per side image shift can reach
    fields = math.ceil(tiles / reach)
    in_field = math.ceil(tiles / fields)
    return {"fov_m": fov, "step_m": step, "fields_per_side": fields, "tiles_per_field_side": in_field, "count": (fields * in_field) ** 2}


def _tile(level: dict, centre: list[float], index: int) -> tuple[int, list[float], list[float]]:
    """(field number, field stage [x, y], tile offset) of tile ``index``; fields and tiles go in snake order."""
    fields, in_field, step = int(level["fields_per_side"]), int(level["tiles_per_field_side"]), float(level["step_m"])
    field, tile = divmod(index, in_field * in_field)
    field_row, field_col = divmod(field, fields)
    if field_row % 2:
        field_col = fields - 1 - field_col
    row, col = divmod(tile, in_field)
    if row % 2:
        col = in_field - 1 - col
    pitch = in_field * step
    at = [centre[0] + (field_col - (fields - 1) / 2.0) * pitch, centre[1] + (field_row - (fields - 1) / 2.0) * pitch]
    return field, at, [(col - (in_field - 1) / 2.0) * step, (row - (in_field - 1) / 2.0) * step]


async def _plan_survey(ctx: BlockContext, args: dict, previous: dict | None) -> dict:
    size, fov, overlap = float(args["size_m"]), float(args["fov_m"]), float(args["overlap"])
    depth, zoom, per_side = int(args["depth"]), float(args["zoom"]), int(args.get("per_side") or 0)
    max_shift, limit, every = float(args["max_shift_m"]), float(args["stage_limit_m"]), int(args.get("spectra_every") or 0)
    if not 0.0 <= overlap < 0.9:
        raise BlockFailed(f"overlap {overlap} must be from 0 to below 0.9")
    if fov <= 0 or size <= 0 or max_shift <= 0:
        raise BlockFailed("size_m, fov_m and max_shift_m must be above 0")
    if depth < 0 or zoom <= 1 and depth > 0 or per_side < 0 or every < 0:
        raise BlockFailed("depth, per_side and spectra_every must be 0 or above, and zoom above 1")
    stage = _numbers(await ctx.call("*_get_stage"), 5, "the stage position")
    centre = stage[:2]
    if per_side:
        size = fov + (per_side - 1) * fov * (1.0 - overlap)
    levels = [_level(size, fov / zoom ** d, overlap, max_shift, per_side if d == 0 else 0) for d in range(depth + 1)]
    count = sum(level["count"] for level in levels)
    shape = "; ".join(f"level {d}: {level['count']} images at a {level['fov_m']:.3g} m field of view, "
                      f"{level['fields_per_side'] ** 2} stage position(s) of {level['tiles_per_field_side'] ** 2}" for d, level in enumerate(levels))
    most = min(int(args["max_tiles"]), MAX_SURVEY_TILES)
    if count > most:
        raise BlockFailed(f"this is {count} images, over the limit of {most} ({shape}); use a smaller size_m or depth, a larger fov_m, "
                          "a smaller zoom, or raise max_tiles")
    stops = count // every if every else 0
    if stops > MAX_SPECTRA_POINTS:
        raise BlockFailed(f"spectra at every {every} tiles is {stops} points, over the limit of {MAX_SPECTRA_POINTS}; "
                          f"use spectra_every of at least {math.ceil(count / MAX_SPECTRA_POINTS)}")
    moves = any(level["fields_per_side"] > 1 for level in levels)
    half_spans = [(level["fields_per_side"] - 1) / 2.0 * level["tiles_per_field_side"] * level["step_m"] for level in levels]
    furthest = max(abs(c) + half_span for half_span in half_spans for c in centre)
    if moves and furthest > limit * (1 + 1e-9):
        raise BlockFailed(f"this area needs the stage at {furthest:.3g} m from its zero, beyond stage_limit_m {limit:.3g} m; "
                          "map a smaller size_m, start nearer the centre of the sample, or raise stage_limit_m if the holder allows it")
    start_fov = await ctx.call("*_get_fov")
    if isinstance(start_fov, bool) or not isinstance(start_fov, (int, float)) or not start_fov > 0:
        raise BlockFailed(f"the field of view was not a number above 0: {str(start_fov)[:200]}")
    start_shift = _numbers(await ctx.call("*_get_image_shift"), 2, "the image shift")[:2]
    summary = (f"{count} images over a {size:.3g} m square around the stage position, in {len(levels)} level(s). {shape}. "
               + (f"Spectra at {stops} point(s). " if stops else "")
               + ("The stage moves between stage positions; tiles within one are reached by image shift." if moves
                  else "The stage does not move: every tile is reached by image shift."))
    return {"levels": levels, "count": count, "stops": stops, "centre_xy_m": centre, "stage_rest": stage[2:5], "moves_stage": moves,
            "start_shift_m": start_shift, "start_fov_m": float(start_fov), "summary": summary}


register_block(Block(
    name="plan_survey",
    summary="Plan a square area around the current stage position as one map per level: level 0 at fov_m, each deeper level the same "
            "area at a field of view zoom times smaller. Tiles are grouped into stage positions, each as many tiles as image shift "
            "reaches. Refuses a plan over max_tiles or beyond stage_limit_m. Moves nothing.",
    run=_plan_survey,
    inputs={
        "size_m": {"type": "number", "doc": "Side of the square to cover, metres. Not used when per_side is given."},
        "fov_m": {"type": "number", "doc": "Field of view of one level-0 tile, metres."},
        "overlap": {"type": "number", "default": 0.3, "doc": "Fraction of a tile shared with its neighbour."},
        "depth": {"type": "integer", "default": 0, "doc": "Levels below level 0. 0: one map at fov_m. Each level multiplies the images by about zoom squared."},
        "zoom": {"type": "number", "default": 4.0, "doc": "How many times smaller the field of view is at each deeper level."},
        "per_side": {"type": "integer", "default": 0, "doc": "Level-0 tiles along each side (4 for a 4 by 4 grid); then size_m follows from it. 0: as many as size_m needs."},
        "max_shift_m": {"type": "number", "default": DEFAULT_MAX_SHIFT_M, "doc": "The largest image shift per axis, metres: how far from a stage position a tile may be."},
        "max_tiles": {"type": "integer", "default": 2000, "doc": f"The most images the whole plan may be (at most {MAX_SURVEY_TILES}); a larger plan is refused with its count."},
        "stage_limit_m": {"type": "number", "default": 9.0e-4, "doc": "The farthest from its zero the stage may be sent on x or y, metres; a plan that needs more is refused."},
        "spectra_every": {"type": "integer", "default": 0, "doc": "Stop for spectra after every this many tiles (survey_tiles' measure output). 0: never."},
    },
    outputs={"levels": "list", "count": "integer", "stops": "integer", "centre_xy_m": "list", "stage_rest": "list", "moves_stage": "boolean",
             "start_shift_m": "list", "start_fov_m": "number", "summary": "string"},
    tools=("*_get_stage", "*_get_image_shift", "*_get_fov"),
))


class _Shot(NamedTuple):
    """One image and the stage position its centre was taken at."""

    key: str
    at: list[float]


@dataclass
class _Visit:
    """One visit of survey_tiles with where the instrument is and what it has taken."""

    ctx: BlockContext
    args: dict
    fov: float
    home_shift: list[float]
    rest: list[float]
    moves: bool
    max_shift: float
    retries: int
    keys: list
    taken_at: list
    counts: dict
    shift: list[float]
    field: int | None = None
    origin: list[float] | None = None

    async def go_to(self, field: int, at: list[float]) -> None:
        if self.moves:
            await self.ctx.call("*_move_stage", {"position": [at[0], at[1], *self.rest]})
        self.origin = _numbers(await self.ctx.call("*_get_stage"), 2, "the stage position")[:2]
        self.field = field

    async def take(self, offset: list[float], what: int) -> _Shot:
        home = self.home_shift
        await self.ctx.call("*_set_image_shift", {"shift": [home[0] + offset[0], home[1] + offset[1]]})
        self.shift = _numbers(await self.ctx.call("*_get_image_shift"), 2, f"the image shift at tile {what}")[:2]
        key = await _expose(self.ctx, self.args, what)
        return _Shot(key, [self.origin[0] + self.shift[0] - home[0], self.origin[1] + self.shift[1] - home[1]])

    async def fits(self, shot: _Shot, other: _Shot) -> bool:
        offset = [shot.at[0] - other.at[0], shot.at[1] - other.at[1]]
        reply = await self.ctx.analyse("match_tiles", {"image": shot.key, "neighbour": other.key, "offset_xy_m": offset, "fov_m": self.fov})
        if not isinstance(reply, dict) or "matched" not in reply:
            raise BlockFailed(f"match_tiles gave no answer for {shot.key}: {str(reply)[:300]}")
        return bool(reply["matched"])

    async def bridge(self, at: list[float], what: int) -> _Shot:
        """Take one more image centred on ``at``. Use image shift if it reaches, or move the stage there."""
        offset = [at[0] - self.origin[0], at[1] - self.origin[1]]
        if max(abs(offset[0]), abs(offset[1])) > self.max_shift * (1 + 1e-9):
            await self.go_to(-1, at)  # -1 is no real field: the next tile moves the stage
            offset = [0.0, 0.0]
        return await self.take(offset, what)

    async def fits_with_bridges(self, tile: _Shot, before: _Shot, index: int) -> bool:
        """Check the newest tile against the one before it, taking bridging images while it does not fit."""
        ok = await self.fits(tile, before)
        fitted, fraction = before, 0.5
        for _ in range(self.retries):
            if ok:
                break
            aim = [fitted.at[axis] + (tile.at[axis] - fitted.at[axis]) * fraction for axis in (0, 1)]
            extra = await self.bridge(aim, index)
            self.keys.insert(len(self.keys) - 1, extra.key)
            self.taken_at.insert(len(self.taken_at) - 1, extra.at)
            self.counts["extra"] += 1
            if await self.fits(extra, fitted):
                ok = await self.fits(tile, extra)
                fitted = extra
            else:
                fraction /= 2.0
        return ok

    async def take_tile(self, plan: dict, centre: list[float], index: int, check: bool) -> None:
        """Take tile ``index`` of a level, check it against the tile before it and count how that went.

        It goes back to this tile's field, so spectra taken after the visit land on the tile they are filed under.
        """
        field, field_at, offset = _tile(plan, centre, index)
        if field != self.field:
            await self.go_to(field, field_at)
        tile = await self.take(offset, index)
        tile_shift = list(self.shift)
        before = _Shot(self.keys[-1], self.taken_at[-1]) if self.keys else None
        self.keys.append(tile.key)
        self.taken_at.append(tile.at)
        self.counts["taken"] += 1
        near = before is not None and all(abs(tile.at[axis] - before.at[axis]) <= NEIGHBOUR_REACH * self.fov for axis in (0, 1))
        if not (check and near):
            self.counts["unchecked"] += 1
        else:
            self.counts["matched" if await self.fits_with_bridges(tile, before, index) else "unmatched"] += 1
            if self.field != field:
                await self.go_to(field, field_at)
            self.shift = tile_shift


async def _survey_tiles(ctx: BlockContext, args: dict, previous: dict | None) -> dict:
    levels = args["levels"]
    centre = _numbers(args["centre_xy_m"], 2, "centre_xy_m")[:2]
    rest = _numbers(args["stage_rest"], 3, "stage_rest")
    home_shift = _numbers(args.get("start_shift_m") or [0.0, 0.0], 2, "start_shift_m")[:2]
    home_fov, max_shift = args.get("start_fov_m"), float(args["max_shift_m"])
    batch = max(1, int(args["batch"]))
    every = max(0, int(args.get("spectra_every") or 0))
    retries = max(0, int(args["retries"]))
    check = bool(args["check"])
    moves = any(int(level["fields_per_side"]) > 1 for level in levels)

    last = previous or {}
    fresh = not previous or int(last["remaining"]) == 0  # first visit, or a level just finished
    level = (int(last["level"]) + 1 if previous else 0) if fresh else int(last["level"])
    if level >= len(levels):
        return {**last, "measure": 0}
    index = 0 if fresh else int(last["index"]) + 1
    keys, taken_at = ([], []) if fresh else (list(last["keys"]), list(last["stage_xy_m"]))
    counts = {name: int(last.get(name, 0)) for name in COUNTS}
    plan = levels[level]
    fov, count = float(plan["fov_m"]), int(plan["count"])
    visit = _Visit(ctx, args, fov, home_shift, rest, moves, max_shift, retries, keys, taken_at, counts, shift=list(home_shift))

    measure, done = 0, 0
    try:
        await ctx.call("*_set_fov", {"fov": fov})  # every visit: a failed run restored the old one
        try:
            while True:
                await visit.take_tile(plan, centre, index, check)
                done += 1
                if every and counts["taken"] % every == 0:
                    measure = 1
                if measure or index == count - 1 or done >= batch:
                    break
                index += 1
        finally:
            await _put_back(ctx, "*_set_image_shift", {"shift": home_shift})
    except BaseException:  # on any failure or stop, restore the field of view
        if home_fov:
            await _put_back(ctx, "*_set_fov", {"fov": float(home_fov)})
        raise
    remaining = count - index - 1
    in_later_levels = sum(int(later["count"]) for later in levels[level + 1:])
    return {"level": level, "index": index, "remaining": remaining, "left": remaining + in_later_levels,
            "measure": measure, "keys": keys, "stage_xy_m": taken_at, "fov_m": fov, "last_key": keys[-1], "shift_m": visit.shift, **counts}


register_block(Block(
    name="survey_tiles",
    summary="Take the next tiles of a plan_survey plan, up to 'batch' a visit: the stage goes to each stage position, its tiles are "
            "reached by image shift, and each tile is checked against the one before it (match_tiles); one that does not fit gets a "
            "bridging image closer to the last. Visit it in a loop while 'remaining' is above 0 (one level), and again while "
            "'left' is above 0 (the next level). 'measure' is 1 when the visit stopped at a tile due for spectra.",
    run=_survey_tiles,
    inputs={
        "levels": {"type": "list", "doc": "plan_survey's levels."},
        "centre_xy_m": {"type": "list", "doc": "plan_survey's centre_xy_m."},
        "stage_rest": {"type": "list", "doc": "[z, alpha, beta] kept for every stage move (plan_survey's stage_rest)."},
        "start_shift_m": {"type": "list", "default": None, "doc": "The image shift to measure from and go back to (plan_survey's start_shift_m)."},
        "start_fov_m": {"type": "number", "default": None, "doc": "The field of view to put back if a visit fails or the run is stopped (plan_survey's start_fov_m)."},
        "detectors": {"type": "list", "default": ["haadf"], "doc": "Detectors of the scanned image."},
        "batch": {"type": "integer", "default": 50, "doc": "Tiles a visit takes before the run is checkpointed; a visit that fails is repeated from its first tile."},
        "check": {"type": "boolean", "default": True, "doc": "Check each tile against the tile before it as it is taken."},
        "retries": {"type": "integer", "default": 1, "doc": "Bridging images to try for a tile that does not fit; each is closer to the last tile than the one before."},
        "spectra_every": {"type": "integer", "default": 0, "doc": "End the visit after every this many tiles with measure 1, so the next nodes take spectra there. 0: never."},
        "max_shift_m": {"type": "number", "default": DEFAULT_MAX_SHIFT_M, "doc": "The largest image shift per axis, metres (plan_survey's)."},
    },
    outputs={"level": "integer", "index": "integer", "remaining": "integer", "left": "integer", "measure": "integer", "keys": "list",
             "stage_xy_m": "list", "fov_m": "number", "last_key": "string", "shift_m": "list", "taken": "integer", "matched": "integer",
             "unmatched": "integer", "unchecked": "integer", "extra": "integer"},
    tools=("*_set_fov", "*_move_stage", "*_get_stage", "*_set_image_shift", "*_get_image_shift", "*_unblank_beam", "*_acquire_scanned_image",
           "*_blank_beam"),
    analysis_tools=("match_tiles",),
    always=("*_blank_beam",),
    acts=True,
))


async def _spectrum_here(ctx: BlockContext, args: dict, previous: dict | None) -> dict:
    keys, places = (list(previous["keys"]), list(previous["at"])) if previous else ([], [])
    if not args.get("on") or not args.get("due"):
        return {"key": None, "keys": keys, "at": places, "result": None}
    x, y = _numbers(args["position"], 2, "position")[:2]
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise BlockFailed(f"position [{x:g}, {y:g}] must be fractions of the image from 0 to 1")
    arguments = args.get("arguments") or {}
    if not isinstance(arguments, dict):
        raise BlockFailed(f"arguments must be a mapping of the tool's arguments, not {str(arguments)[:200]}")
    shift, home = args.get("shift_m"), args.get("home_shift_m")
    try:
        if shift:
            await ctx.call("*_set_image_shift", {"shift": _numbers(shift, 2, "shift_m")[:2]})
        await ctx.call("*_place_beam", {"position": [x, y]})
        try:
            await ctx.call("*_unblank_beam")
            result = await ctx.call(str(args["tool"]), arguments)
        finally:
            await ctx.call_always("*_blank_beam")  # never leave the beam parked on the sample
    finally:
        if shift and home:
            await _put_back(ctx, "*_set_image_shift", {"shift": _numbers(home, 2, "home_shift_m")[:2]})
    key = result if isinstance(result, str) and DATA_KEY_IN_TEXT.fullmatch(result) else None
    keys.append(key)
    places.append(args.get("at"))
    return {"key": key, "keys": keys, "at": places, "result": result}


register_block(Block(
    name="spectrum_here",
    summary="Park the beam at a point of the image just taken, unblank, call one acquisition tool (an EDS or EELS spectrum), and "
            "blank. The tool is named in the workflow itself. In a loop it keeps every key with where it was taken; it does nothing on "
            "a visit where 'on' or 'due' is false.",
    run=_spectrum_here,
    inputs={
        "tool": {"type": "string", "doc": "The acquisition tool's name, written out: *_acquire_spectrum, *_acquire_eels_spectrum."},
        "arguments": {"type": "object", "default": None, "doc": 'The tool\'s arguments: {"detector_name": "eds"}.'},
        "position": {"type": "list", "default": [0.5, 0.5], "doc": "[x, y] as fractions of the image, 0 to 1: x from the left, y from the top."},
        "on": {"type": "boolean", "default": True, "doc": "False: this node does nothing. Usually a run input."},
        "due": {"type": "any", "default": True, "doc": "Falsy: this visit does nothing. In a tile loop: survey_tiles' measure."},
        "at": {"type": "any", "default": None, "doc": "What to keep with the key as its place, usually the tile's key."},
        "shift_m": {"type": "list", "default": None, "doc": "The image shift the image was taken at (survey_tiles' shift_m); set before the beam is parked."},
        "home_shift_m": {"type": "list", "default": None, "doc": "The image shift to go back to afterwards (plan_survey's start_shift_m)."},
    },
    outputs={"key": "string", "keys": "list", "at": "list", "result": "any"},
    tools=("*_place_beam", "*_unblank_beam", "*_blank_beam", "*_set_image_shift"),
    always=("*_blank_beam",),
    acts=True,
    tool_input="tool",
))


async def _keep(ctx: BlockContext, args: dict, previous: dict | None) -> dict:
    items = list(previous["items"]) if previous else []
    items.append(args.get("value"))
    return {"items": items, "count": len(items)}


register_block(Block(
    name="keep",
    summary="Keep a value from every visit of a loop (each level's map), so the run ends with all of them and not only the last.",
    run=_keep,
    inputs={"value": {"type": "any", "doc": 'What to keep this visit: "{stitch.map}".'}},
    outputs={"items": "list", "count": "integer"},
))


async def _return_home(ctx: BlockContext, args: dict, previous: dict | None) -> dict:
    if args.get("fov_m"):
        await ctx.call("*_set_fov", {"fov": float(args["fov_m"])})
    shift = _numbers(args["shift_m"], 2, "shift_m")[:2]
    await ctx.call("*_set_image_shift", {"shift": shift})
    xy = _numbers(args["xy_m"], 2, "xy_m")[:2]
    if args.get("move_stage"):
        await ctx.call("*_move_stage", {"position": [*xy, *_numbers(args["stage_rest"], 3, "stage_rest")]})
    return {"xy_m": xy, "shift_m": shift}


register_block(Block(
    name="return_home",
    summary="Put the instrument back where a survey started: the field of view, the image shift, and the stage position if the survey moved the stage.",
    run=_return_home,
    inputs={"xy_m": {"type": "list", "doc": "The stage [x, y] at the start (plan_survey's centre_xy_m)."},
            "stage_rest": {"type": "list", "doc": "[z, alpha, beta]."},
            "shift_m": {"type": "list", "doc": "The image shift at the start (plan_survey's start_shift_m)."},
            "fov_m": {"type": "number", "default": None, "doc": "The field of view to put back (plan_survey's start_fov_m)."},
            "move_stage": {"type": "any", "default": True, "doc": "Falsy: the stage is left alone (plan_survey's moves_stage)."}},
    outputs={"xy_m": "list", "shift_m": "list"},
    tools=("*_move_stage", "*_set_image_shift", "*_set_fov"),
    acts=True,
))
