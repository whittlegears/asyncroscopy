"""Map plans: field checks, synonyms, budgets, and resolving a plan into the steps the Mapper runs.

A plan (asyncroscopy/contracts/map_plan.schema.json) is one of three kinds:

* a tiled map: an area (``box_m``, or ``size_m`` around ``centre_m`` or the stage), a tile field of view
  and overlap, then optionally stitch, score (``find_regions``) and revisit phases;
* a revisit-only plan (``revisit`` with ``points`` or an existing ``map``): images at given places, no tiles;
* a zoom plan (``zoom``): one image per tier, each aimed by image shift at the parent image's most
  distinct feature, no stage moves.

``resolve`` checks a plan and returns it resolved: defaults merged, the box computed, the grid stored as
a few numbers (``compact_grid``), the budgets checked and an ``estimate``. Steps are computed from the
resolved plan when needed (``step_at``), so a plan of millions of tiles stores a few numbers. Nothing
here talks to a device except through the profile and the ``analysis`` callable it is given.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass

import numpy as np

from asyncroscopy.contracts import load, validate
from asyncroscopy.routines.grid import EDGE_TOLERANCE_M, compact_grid, grid_axes, tile_at

from . import live_drift
from .profile import GENERIC_PLAN_DEFAULTS, GENERIC_PLAN_FIELDS, TIME_BUDGET, Cost, InstrumentProfile
from .stitch_service import StitchService

# pyTEMlib's stitcher searches each overlap only up to this share of its width; neighbours the stage puts further
# apart than that cannot be matched and are left unplaced (docs/designs/multiscale_maps.md, P2 findings).
STITCH_SEARCH_SHARE = 0.6
REVISIT_FOV_MIN_FRACTION = 1 / 20  # a revisit image may zoom in at most 20x on a tile
ZOOM_MIN_FOV_M = 5e-9  # no zoom tier narrower than this: atomic columns show well before it, and the dose climbs as 1/fov^2
ZOOM_STEP = 8  # without fovs_m, each tier's field of view is the one before's over this, down to ZOOM_MIN_FOV_M
ZOOM_EXAMPLE = {"zoom": {"parent": "<the image key to start from>.h5"}}
EMPTY_GRID = {"rows": 0, "cols": 0, "x0": 0.0, "dx": 0.0, "y0": 0.0, "dy": 0.0}

FIELD_SYNONYMS = {
    "max_dose_e_per_a2": "dose_budget_e_per_a2", "max_dose": "dose_budget_e_per_a2", "dose_limit": "dose_budget_e_per_a2",
    "dose_budget": "dose_budget_e_per_a2", "dose": "dose_budget_e_per_a2", "fov": "fov_m", "field_of_view_m": "fov_m",
    "size": "size_m", "box": "box_m", "centre": "centre_m", "center_m": "centre_m", "center": "centre_m",
    "dwell": "dwell_s", "dwell_time": "dwell_s", "image_size": "imsize", "pixels": "imsize", "regions": "find_regions",
}
REVISIT_ONLY_KEYS = ("map", "top_k", "points")  # fields of a revisit plan ({"revisit": {...}}) that do not exist at the top level
ZOOM_PLAN_FIELDS = frozenset({"zoom", "dry_run", "label", "budgets"})  # plus the profile's budget and image fields


@dataclass(frozen=True)
class PlanEnvironment:
    """What resolving a plan needs from the Mapper: its limits, the stage's 1-sigma position error (kept in the
    resolved plan: the stitch and live drift tracking bound their searches with it), and the analysis server
    (None: it has none)."""

    max_tiles: int
    estimate_move_s: float
    stage_sigma_m: float
    analysis: StitchService | None = None


# ----------------------------------------------------------------------
# Field checks and budgets
# ----------------------------------------------------------------------
def check_plan_fields(plan: dict) -> None:
    """Refuse unknown plan fields by name, with the closest allowed field and the full list, before anything else."""
    allowed = sorted(load("map_plan")["properties"])
    unknown = [k for k in plan if k not in allowed]
    if not unknown:
        return
    revisit_keys = [k for k in unknown if k in REVISIT_ONLY_KEYS]
    if revisit_keys:
        # Fields of a revisit plan given at the top level: show the plan they belong in, with the caller's own values.
        nested = {**(plan.get("revisit") if isinstance(plan.get("revisit"), dict) else {}), **{k: plan[k] for k in revisit_keys}}
        nested.setdefault("fov_m", plan.get("fov_m", "<a few times smaller than the map's tiles>"))
        raise ValueError(f"{', '.join(map(repr, revisit_keys))} belong inside revisit. Nothing was run; send the plan as {json.dumps({'revisit': nested}, default=str)}")
    hints = []
    for key in unknown:
        guess = FIELD_SYNONYMS.get(key) or next(iter(difflib.get_close_matches(key, allowed, n=1, cutoff=0.6)), None)
        hints.append(f"{key!r} (did you mean {guess!r}?)" if guess else repr(key))
    raise ValueError(f"unknown plan field(s): {', '.join(hints)}. Nothing was dropped or changed: resend the plan with the right names. Allowed fields: {', '.join(allowed)}")


def check_profile_fields(plan: dict, profile: InstrumentProfile) -> None:
    """Refuse plan fields this instrument's profile does not use (a tip has no dwell time), by name, before anything else."""
    allowed = GENERIC_PLAN_FIELDS | profile.plan_fields
    unused = [k for k in plan if k not in allowed]
    if unused:
        raise ValueError(
            f"{', '.join(map(repr, unused))} not used when mapping with the {profile.name!r} instrument profile ({profile.doc}). "
            f"Nothing was run; resend the plan without them. Fields it takes: {', '.join(sorted(allowed))}"
        )


def plan_budgets(p: dict, profile: InstrumentProfile) -> dict:
    """The plan's budgets by name: ``budgets`` plus the profile's fields that are a budget under their own name."""
    budgets = dict(p.get("budgets") or {})
    for field, name in profile.budget_fields.items():
        if p.get(field) is not None:
            budgets.setdefault(name, p[field])
    return budgets


def check_budgets(budgets: dict, costs: dict[str, Cost], duration_s: float, profile: InstrumentProfile) -> dict | None:
    """Refuse a plan over any budget it gives (every profile can budget time_s); what each budgeted cost comes to."""
    if not budgets:
        return None
    costs = {**costs, TIME_BUDGET: Cost(duration_s, "s", why="the estimated duration of the whole job", advice="map a smaller area, find fewer regions, or shorten each tile")}
    unknown = sorted(set(budgets) - set(costs))
    if unknown:
        raise ValueError(f"unknown budget(s) {unknown} for the {profile.name!r} instrument profile; it can budget {sorted(costs)}")
    for name, limit in budgets.items():
        if costs[name].value > float(limit):
            raise ValueError(costs[name].refusal(limit))
    return {name: {"value": round(costs[name].value, 3), "budget": limit, "units": costs[name].units} for name, limit in budgets.items()}


def overlap_warning(overlap: float, tile_m: float, stage_sigma_m: float) -> str | None:
    """Why a plan's overlap may be too small for this stage to stitch, or None.

    Two neighbours land up to about 2 x sqrt(2) stage sigmas further apart than planned; the stitcher looks for
    them only within STITCH_SEARCH_SHARE of the overlap, so a pair further off is not matched."""
    sigma = float(stage_sigma_m)
    reach = 2.0 * np.sqrt(2.0) * sigma
    window = STITCH_SEARCH_SHARE * overlap * tile_m
    if window >= reach or sigma <= 0:
        return None
    needed = min(0.9, reach / (STITCH_SEARCH_SHARE * tile_m))
    return (
        f"overlap {overlap:.2f} of {tile_m * 1e9:.0f} nm tiles lets the stitcher search {window * 1e9:.0f} nm, but this stage "
        f"(stage_sigma_m {sigma * 1e9:.0f} nm) can put neighbours {reach * 1e9:.0f} nm from where they were planned: tiles it "
        f"moves that far are left unplaced. An overlap of {needed:.2f}, or tiles about {reach / (STITCH_SEARCH_SHARE * overlap) * 1e6:.1f} um "
        "across at this overlap, would let every pair be found."
    )


# ----------------------------------------------------------------------
# Resolving
# ----------------------------------------------------------------------
def resolve(plan: dict, profile: InstrumentProfile, env: PlanEnvironment) -> dict:
    """The plan checked and resolved (see the module docstring); ValueError says what to change."""
    check_plan_fields(plan)
    validate(plan, "map_plan")
    check_profile_fields(plan, profile)
    if "zoom" in plan:
        return _resolve_zoom_only(plan, profile, env)
    if isinstance(plan.get("revisit"), dict) and ("points" in plan["revisit"] or "map" in plan["revisit"]):
        return _resolve_revisit_only(plan, profile, env)
    return _resolve_tiles(plan, profile, env)


def _resolve_tiles(plan: dict, profile: InstrumentProfile, env: PlanEnvironment) -> dict:
    if ("box_m" in plan) == ("size_m" in plan):
        raise ValueError("give the area as box_m [x_min, x_max, y_min, y_max] or as size_m [width, height], not both or neither")
    p = {**GENERIC_PLAN_DEFAULTS, **profile.plan_defaults, **plan}
    profile.resolve(p)
    if "fov_m" not in p:
        raise ValueError("fov_m (the field of view of each tile, metres) is required: add \"fov_m\": 2e-7 (200 nm tiles) unless the request gives a tile size")
    if "size_m" in p:
        # The device does the arithmetic: callers (often small models) get centre +/- half wrong.
        width, height = (float(v) for v in p["size_m"])
        cx, cy = (float(v) for v in (p.get("centre_m") or profile.current_xy()))
        p["box_m"] = [cx - width / 2, cx + width / 2, cy - height / 2, cy + height / 2]
    x_min, x_max, y_min, y_max = (float(v) for v in p["box_m"])
    if not (x_min < x_max and y_min < y_max):
        raise ValueError("box_m must be [x_min, x_max, y_min, y_max] with min < max")
    if (p["find_regions"] or p.get("revisit")) and not (p["stitch"] and env.analysis):
        raise ValueError("find_regions and revisit need the map stitched: keep stitch on, on a Mapper with an analysis_url")
    if p.get("revisit") and not p["find_regions"]:
        raise ValueError("revisit images the regions find_regions finds (give find_regions > 0), or give revisit points [[x, y], ...] or an existing map (revisit map, top_k)")
    fov = float(p["fov_m"])
    revisit = None
    if p.get("revisit"):
        if "fov_m" not in p["revisit"]:
            raise ValueError(f"revisit fov_m is required with a map: a few times smaller than the tiles' {fov:.3g} m (between {fov * REVISIT_FOV_MIN_FRACTION:.3g} m and {fov:.3g} m)")
        revisit = profile.resolve_revisit(p, p["revisit"])
        if not fov * REVISIT_FOV_MIN_FRACTION <= revisit["fov_m"] <= fov:
            raise ValueError(f"revisit fov_m must be between {fov * REVISIT_FOV_MIN_FRACTION:.3g} m (a twentieth of the tile) and the tile's {fov:.3g} m")
        p["revisit"] = revisit
    tile_w, tile_h = profile.tile_size_m(p)
    if tile_h != tile_w:
        p["tile_m"] = [tile_w, tile_h]
    grid = compact_grid(*grid_axes([x_min, x_max, y_min, y_max], tile_w, float(p["overlap"]), limit=int(env.max_tiles), fov_y=tile_h if tile_h != tile_w else None))
    tiles = grid["rows"] * grid["cols"]
    live_drift.resolve(p, profile, (tile_w, tile_h), tiles)
    if tiles == 1:
        # One tile is an image, not a map: almost always the area's size put in fov_m. A strip one tile high is a map.
        raise ValueError(f"fov_m {fov:.3g} m covers the whole area, so the map would be one tile; fov_m is the field of view of each tile (2e-7 when the request gives no tile size). To image given places, use a revisit plan: {{\"revisit\": {{\"fov_m\": ..., \"points\": [[x, y], ...]}}}}")

    stitch = bool(p["stitch"] and env.analysis)
    regions = int(p["find_regions"]) if stitch else 0
    revisits = regions if revisit else 0
    budgets = plan_budgets(p, profile)
    costs, extra = profile.costs(p, grid, revisit, revisits, budgets)
    revisit_s = revisits * (float(env.estimate_move_s) + (profile.revisit_seconds(revisit) if revisit else 0.0))
    duration = tiles * (float(env.estimate_move_s) + profile.tile_seconds(p)) + revisit_s
    budgeted = check_budgets(budgets, costs, duration, profile)
    steps = tiles + int(stitch) + int(regions > 0) + revisits
    estimate = {
        "steps": steps,
        "tiles": tiles,
        "duration_s": round(duration, 1),
        **extra,
        "phases": ["tiles"] + (["stitch"] if stitch else []) + (["score"] if regions else []) + (["revisit"] if revisits else []),
    }
    if budgeted:
        estimate["budgets"] = budgeted
    if stitch:
        warning = overlap_warning(float(p["overlap"]), min(tile_w, tile_h), env.stage_sigma_m)
        if warning:
            estimate["warnings"] = [warning]
    else:
        estimate["stitch"] = "off" if not p["stitch"] else "this Mapper has no analysis_url; stitch the run with assemble_map"
    return {
        **p, "box_m": [x_min, x_max, y_min, y_max], "stitch": stitch, "find_regions": regions, "step_count": steps, "estimate": estimate, "grid": grid,
        "stage_sigma_m": float(env.stage_sigma_m),
    }


def _resolve_revisit_only(plan: dict, profile: InstrumentProfile, env: PlanEnvironment) -> dict:
    """A plan that only images given points, or an existing map's best regions: no tiles, no stitch."""
    p = {**GENERIC_PLAN_DEFAULTS, **profile.plan_defaults, **plan}
    revisit = dict(p["revisit"])
    if ("points" in revisit) == ("map" in revisit):
        raise ValueError("give revisit points [[x, y], ...] or an existing map (revisit map, top_k), not both")
    tile_only = sorted(k for k in ("find_regions", *profile.tile_fields) if p.get(k))
    if tile_only:
        raise ValueError(f"a revisit-only plan takes no tiles: leave out {' and '.join(tile_only)}, or give an area to map")
    if live_drift.tracking(p):
        raise ValueError("live drift tracking (drift.track) corrects tiles of a map, and a revisit-only plan has none: leave out drift")
    profile.resolve(p)
    if "fov_m" not in revisit:
        # A small agent wrote "fov_m": <Microscope>_get_fov into its plan (invalid JSON, so no call at all): the job reads it.
        revisit["fov_m"] = profile.current_fov_m()
        if revisit["fov_m"] is None:
            raise ValueError(f"give revisit fov_m: the {profile.name!r} instrument profile has no field of view to read")
    revisit.update(profile.resolve_revisit(p, revisit))
    fov = float(revisit["fov_m"])
    half = fov / 2.0
    regions: list[dict] = []
    if "points" in revisit:
        regions = [{"id": f"point:{k}", "x_m": float(x), "y_m": float(y)} for k, (x, y) in enumerate(revisit["points"])]
        xs = [r["x_m"] for r in regions]
        ys = [r["y_m"] for r in regions]
        box = p.get("box_m") or [min(xs) - half, max(xs) + half, min(ys) - half, max(ys) + half]
        count = len(regions)
    else:
        if env.analysis is None:
            raise ValueError("revisiting an existing map's regions needs a Mapper with an analysis_url")
        info = env.analysis.map_info(map=revisit["map"])
        box = p.get("box_m") or list(info["stage_box_m"])
        count = int(revisit.get("top_k", 3))
    x_min, x_max, y_min, y_max = (float(v) for v in box)
    budgets = plan_budgets(p, profile)
    costs, extra = profile.revisit_only_costs(p, revisit, count, budgets)
    duration = count * (float(env.estimate_move_s) + profile.revisit_seconds(revisit))
    budgeted = check_budgets(budgets, costs, duration, profile)
    for r in regions:
        # With the tolerance: a box built as point +/- half the fov rounds the point a hair outside its own box.
        tol = EDGE_TOLERANCE_M
        if not (x_min + half - tol <= r["x_m"] <= x_max - half + tol and y_min + half - tol <= r["y_m"] <= y_max - half + tol):
            raise ValueError(f"revisit point ({r['x_m']:.3e}, {r['y_m']:.3e}) m is outside the approved area {[x_min, x_max, y_min, y_max]} for a {fov:.2e} m image")
    steps = count + (1 if "map" in revisit else 0)
    estimate = {
        "steps": steps, "tiles": 0, "revisits": count,
        "duration_s": round(duration, 1),
        **extra,
        "phases": (["select"] if "map" in revisit else []) + ["revisit"],
    }
    if budgeted:
        estimate["budgets"] = budgeted
    return {
        **p, "fov_m": fov, "revisit": revisit, "box_m": [x_min, x_max, y_min, y_max], "stitch": False, "find_regions": 0,
        "step_count": steps, "estimate": estimate, "grid": dict(EMPTY_GRID), "regions": regions, "revisit_only": True,
    }


def _resolve_zoom_only(plan: dict, profile: InstrumentProfile, env: PlanEnvironment) -> dict:
    """A plan that zooms in from an existing image by image shift: no tiles, no stage moves."""
    allowed = ZOOM_PLAN_FIELDS | set(profile.budget_fields) | profile.image_fields
    others = sorted(k for k in plan if k not in allowed)
    if others:
        raise ValueError(f"a zoom plan takes no area, tiles or revisit: leave out {', '.join(others)}")
    if env.analysis is None:
        raise ValueError("zooming finds and places images on the analysis server: it needs a Mapper with an analysis_url")
    if not profile.image_shift_limit_m:
        raise ValueError(f"zooming aims each image by image shift, which the {profile.name!r} instrument profile cannot set")
    zoom = dict(plan["zoom"])
    frame = env.analysis.find_features(image=zoom["parent"], n=0)
    if "error" in frame:
        raise ValueError(f"cannot zoom from {zoom['parent']!r}: {frame['error'].get('message', frame['error'])}")
    parent_fov = float(frame["pixel_size_m"]) * float(max(frame["shape"]))
    if "fovs_m" in zoom:
        fovs = [float(f) for f in zoom["fovs_m"]]
    else:
        fovs = []
        while parent_fov / ZOOM_STEP ** (len(fovs) + 1) >= ZOOM_MIN_FOV_M and len(fovs) < 4:
            fovs.append(parent_fov / ZOOM_STEP ** (len(fovs) + 1))
        if not fovs:
            raise ValueError(f"{zoom['parent']} is {parent_fov:.3g} m across: a tier {ZOOM_STEP}x finer would be under {ZOOM_MIN_FOV_M:g} m, so there is nothing finer to zoom to")
    if fovs[0] >= parent_fov:
        raise ValueError(f"zoom fovs_m must start below the parent's {parent_fov:.3g} m field of view; got {fovs}")
    if any(b >= a for a, b in zip(fovs, fovs[1:])):
        raise ValueError(f"zoom fovs_m must get smaller at every tier, coarsest first; got {fovs}")
    if fovs[-1] < ZOOM_MIN_FOV_M:
        raise ValueError(f"zoom fovs_m {fovs[-1]:.3g} m is narrower than {ZOOM_MIN_FOV_M:g} m: atomic columns show at about 1e-8 m, and the dose rises as 1/fov^2")
    p = {**GENERIC_PLAN_DEFAULTS, **profile.plan_defaults, **plan, "fov_m": fovs[0], "stitch": False, "find_regions": 0}
    profile.resolve(p)
    tiers = [profile.resolve_revisit(p, {"fov_m": f, **{k: v for k, v in zoom.items() if k in profile.image_fields}}) for f in fovs]
    budgets = plan_budgets(p, profile)
    # The finest image lies inside every coarser one, so its spot gets every tier's share of each cost: the sum is the most any spot gets.
    tier_costs = [profile.revisit_only_costs(p, tier, 1, budgets) for tier in tiers]
    costs = {
        name: Cost(sum(c[name].value for c, _ in tier_costs), cost.units, why=f"{len(tiers)} zoom images, all on the finest image's spot", advice="zoom less far (a larger last fovs_m)")
        for name, cost in tier_costs[0][0].items()
    }
    duration = sum(profile.revisit_seconds(t) for t in tiers)
    budgeted = check_budgets(budgets, costs, duration, profile)
    x, y = profile.current_xy()
    reach = float(profile.image_shift_limit_m)
    estimate = {
        "steps": len(tiers), "tiles": 0, "zoom_tiers": len(tiers), "duration_s": round(duration, 1),
        **tier_costs[-1][1], **{f"max_{name}": round(cost.value, 3) for name, cost in costs.items()}, "phases": ["zoom"],
    }
    if budgeted:
        estimate["budgets"] = budgeted
    return {
        **p, "zoom": {**zoom, "fovs_m": fovs, "parent_fov_m": parent_fov, "tiers": tiers}, "box_m": [x - reach, x + reach, y - reach, y + reach],
        "step_count": len(tiers), "estimate": estimate, "grid": dict(EMPTY_GRID), "zoom_only": True,
    }


# ----------------------------------------------------------------------
# Steps of a resolved plan
# ----------------------------------------------------------------------
def tile_count(plan: dict) -> int:
    grid = plan["grid"]
    return grid["rows"] * grid["cols"]


def step_at(plan: dict, index: int) -> dict:
    """Step ``index`` of a resolved plan: {"phase": ..., and the tile, region or tier}."""
    if plan.get("zoom_only"):
        return {"phase": "zoom", "tier": index}
    if plan.get("revisit_only"):
        if "map" in plan["revisit"]:
            return {"phase": "select"} if index == 0 else {"phase": "revisit", "region": index - 1}
        return {"phase": "revisit", "region": index}
    tiles = tile_count(plan)
    if index < tiles:
        return {"phase": "tiles", **tile_at(plan["grid"], plan["order"], index)}
    k = index - tiles
    if plan["stitch"] and k == 0:
        return {"phase": "stitch"}
    k -= int(plan["stitch"])
    if plan["find_regions"] and k == 0:
        return {"phase": "score"}
    return {"phase": "revisit", "region": k - int(bool(plan["find_regions"]))}


def stage_extremes(plan: dict) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """The stage x and y values the plan goes to at its extremes (outermost tile centres, given revisit points), named."""
    grid = plan["grid"]
    xs: list[tuple[str, float]] = []
    ys: list[tuple[str, float]] = []
    if grid["rows"] and grid["cols"]:
        xs += [("tiles (left column)", grid["x0"]), ("tiles (right column)", grid["x0"] + (grid["cols"] - 1) * grid["dx"])]
        ys += [("tiles (first row)", grid["y0"]), ("tiles (last row)", grid["y0"] + (grid["rows"] - 1) * grid["dy"])]
    for region in plan.get("regions") or []:
        xs.append((f"revisit {region['id']}", region["x_m"]))
        ys.append((f"revisit {region['id']}", region["y_m"]))
    if plan.get("revisit_only") and "map" in plan["revisit"]:  # the map's regions are chosen later, inside its box
        x_min, x_max, y_min, y_max = plan["box_m"]
        xs += [("revisit area", x_min), ("revisit area", x_max)]
        ys += [("revisit area", y_min), ("revisit area", y_max)]
    return xs, ys


def tile_half(plan: dict, fov: float | None = None) -> tuple[float, float]:
    """Half the width and height of an image of ``fov`` (default a tile); tiles may be rectangular (plan tile_m)."""
    if fov is not None:
        return fov / 2.0, fov / 2.0
    width, height = plan.get("tile_m") or (plan["fov_m"], plan["fov_m"])
    return float(width) / 2.0, float(height) / 2.0
