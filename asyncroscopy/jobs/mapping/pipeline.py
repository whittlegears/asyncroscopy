"""The map pipeline after the tiles: stitch, score, select, revisit and zoom, on the analysis server.

1. stitch (``stitch``, default on, needs ``analysis_url``): the finished run is stitched on the analysis
   server (``assemble_map``) with search bounds from the Mapper's ``stage_sigma_m`` and
   ``max_drift_m_per_s``; every EDS point spectrum becomes a point of interest linked to its key.
2. score (``find_regions``: N > 0): the best N regions become points of interest.
3. select (a revisit-only plan with a ``map``): that map's best scored regions, each distinct.
4. revisit (``revisit``: {"fov_m": ...}): the stage goes to each region, corrected for the drift the
   stitch fitted (plus live tracking's estimate), and only inside the plan box, and takes a
   higher-magnification image, added as a point of interest linked to the image and its region.
5. zoom (a ``zoom`` plan): each tier finds the parent image's most distinct feature (``find_features``),
   aims the image at it by image shift, images it and places it on its parent (``place_image``).

Every phase calls the analysis server through a ``StitchService`` (stitch_service.py): by default the analysis server
over MCP, built the first time it is needed. Stitch and score go through asyncroscopy.core.map_analysis, the one stitch
and score path every caller shares: it waits for the analysis job with backoff and a stall limit, checking for an abort
between waits. The analysis job's id is saved in the job before waiting, so a resumed job waits for that same job (or, if
another client already started one on the same run, adopts it) instead of starting a second. If the analysis server is
down, restarted or too slow, the job still ends done with every tile safe; ``stitch_error`` says why and how to retry.
Scoring and revisits are idempotent: a resumed job never images a region twice or adds its points twice.
"""

from __future__ import annotations

import asyncio
import math
import threading
from collections.abc import Callable

from asyncroscopy.core.map_analysis import AnalysisFailed, AnalysisUnreachable, score_map, start_job, stitch_arguments, stitch_map
from asyncroscopy.routines.drift import drift_at
from asyncroscopy.routines.grid import EDGE_TOLERANCE_M

from . import live_drift
from .profile import InstrumentProfile
from .stitch_service import STITCH_SERVICES, StitchService

REVISIT_EDGE_SHIFT = 0.8  # a revisit image may sit this fraction of its half-width off a region near the box edge
ZOOM_FEATURE_SIZE = (0.25, 2.0)  # a tier aims at a feature between these multiples of its own field of view across
ZOOM_WINDOW_SHARE = 1 / 16  # place_image searches this share of the parent's field of view around the aimed point
ZOOM_CANDIDATES = 8  # features asked for at each tier. The first whose finer image fits inside the parent is used.
REGION_FIELDS = ("id", "x_m", "y_m", "score", "reason", "scorer_version", "window_m")
# What a dry run reports in place of every image and spectrum.
DRY_RUN_SKIPPED = "skipped (dry run): nothing was acquired and the sample was not exposed; the commands are in the trace with dry_run: true"


class Pipeline:
    """The analysis phases of one Mapper's jobs, over the StitchService ``backend`` names. ``save(job)`` persists the job under ``lock``."""

    def __init__(
        self,
        url: str,
        max_drift_m_per_s: float,
        aborting: Callable[[dict], bool],
        lock: threading.RLock,
        save: Callable[[dict], None],
        warn: Callable[[str], None],
        backend: str = "mcp",
    ) -> None:
        self.url = url
        self.backend = backend
        self.max_drift_m_per_s = float(max_drift_m_per_s)
        self._aborting = aborting
        self._lock = lock
        self._save = save
        self._warn = warn
        self._service: StitchService | None = None

    # calls
    def service(self) -> StitchService:
        """The stitching backend, built the first time it is needed."""
        if self._service is None:
            self._service = STITCH_SERVICES.get(self.backend)(self.url)
        return self._service

    def close(self) -> None:
        if self._service is not None:
            self._service.close()
            self._service = None

    @staticmethod
    def state(job: dict) -> dict:
        """The job's pipeline record (analysis job ids, map, regions, revisits, zoom tiers, errors)."""
        return job.setdefault("pipeline", {})

    def stitch_arguments(self, job: dict, run: str) -> dict:
        """assemble_map's arguments: the run, bounded by the stage error the plan was resolved with."""
        return stitch_arguments(run, stage_sigma_m=float(job["resolved_plan"]["stage_sigma_m"]), max_drift_m_per_s=self.max_drift_m_per_s, eds_pois=True)

    async def _analyse(self, tool: str, arguments: dict) -> dict:
        """One call to the analysis server, off this thread; any failure to get a reply is AnalysisUnreachable."""
        try:
            return await asyncio.to_thread(self.service().call, tool, arguments)
        except Exception as exc:
            raise AnalysisUnreachable(f"the analysis server at {self.url}: {type(exc).__name__}: {exc}") from exc

    def _wait_options(self, job: dict, slot: str) -> dict:
        """How a shared stitch or score runs for this job: it resumes the analysis job saved in ``slot``, saves the id of a
        new one there before waiting, and stops when this job aborts."""
        def keep(job_id: str) -> None:
            with self._lock:
                self.state(job)[slot] = job_id
                self._save(job)
        return {"job_id": self.state(job).get(slot), "on_started": keep, "stopping": lambda: self._aborting(job), "warn": self._warn}

    # phases
    def stitch(self, job: dict, run: str, tiles: int) -> dict | None:
        pipeline = self.state(job)
        arguments = self.stitch_arguments(job, run)
        try:
            stitched = asyncio.run(stitch_map(self._analyse, **arguments, **self._wait_options(job, "stitch_job")))
        except AnalysisFailed as exc:  # the tiles are safe, the job goes on
            pipeline["stitch_error"] = f"{exc}. The tiles are in {run}; stitch them later with assemble_map(run={run!r})."
            pipeline.pop("stitch_job", None)
            return {"phase": "stitch", "ok": False}
        if stitched is None:
            return None  # aborted while stitching, and a resume waits for the same stitch
        pipeline["map"] = stitched["map"]
        summary = stitched["summary"]
        pipeline["drift_fit"] = summary.get("drift")  # kept for revisits; the reply shows only its totals
        pipeline["stitching"] = {k: v for k, v in summary.items() if k in ("tiles", "levels", "canvas_shape", "matches_used", "median_residual_px", "p95_residual_px", "fallback_tiles", "eds_pois", "solver", "drift_unmeasured")}
        if summary.get("drift"):
            pipeline["stitching"]["drift"] = {k: summary["drift"][k] for k in ("total_nm", "mean_rate_nm_per_s") if k in summary["drift"]}
        with self._lock:
            job["result_keys"]["map"] = pipeline["map"]
        return {"phase": "stitch", "ok": True, "map": pipeline["map"]}

    def stitch_released(self, job: dict, run: str) -> None:
        """Start a stitch of a released job's tiles; nobody waits for it (results name its analysis job)."""
        try:
            job_id = asyncio.run(start_job(self._analyse, "assemble_map", self.stitch_arguments(job, run)))
            with self._lock:
                job["result_keys"]["map_job"] = job_id
        except Exception as exc:
            with self._lock:
                self.state(job)["stitch_error"] = f"{type(exc).__name__}: {exc}. Stitch the tiles with assemble_map(run={run!r})."
        with self._lock:
            self._save(job)

    def score(self, job: dict) -> dict | None:
        pipeline = self.state(job)
        if "map" not in pipeline:
            return {"phase": "score", "skipped": "no map"}
        if "regions" in pipeline:
            return {"phase": "score", "ok": True, "regions": len(pipeline["regions"])}  # already scored before a resume
        n = int(job["resolved_plan"]["find_regions"])
        # The job's tag lets a retry replace the regions it added before.
        map_key, tag = pipeline["map"], job["reservation_id"]
        try:
            scored = asyncio.run(score_map(self._analyse, map_key, n, tag, **self._wait_options(job, "score_job")))
        except AnalysisFailed as exc:
            pipeline["score_error"] = f"{exc}. Score again with score_map(map={pipeline['map']!r}, top_k={n})."
            pipeline.pop("score_job", None)
            return {"phase": "score", "ok": False}
        if scored is None:
            return None
        pipeline["regions"] = [{k: p.get(k) for k in REGION_FIELDS} for p in scored["pois"]]
        return {"phase": "score", "ok": True, "regions": len(pipeline["regions"])}

    def select(self, job: dict) -> dict:
        """Take an existing map's best scored regions (revisit-only plans with a map)."""
        pipeline = self.state(job)
        if "regions" in pipeline:
            return {"phase": "select", "regions": len(pipeline["regions"])}  # already chosen before a resume
        revisit = job["resolved_plan"]["revisit"]
        reply = self.service().call("list_pois", {"map": revisit["map"], "kind": "candidate", "limit": 500})
        candidates = sorted((p for p in reply.get("pois", []) if p.get("score") is not None), key=lambda p: p["score"], reverse=True)
        # Scoring a map twice stores the same regions again under new ids.
        # Regions closer than half a field of view share one image, so only the best of them is kept.
        apart = float(revisit["fov_m"]) / 2.0
        top: list[dict] = []
        for c in candidates:
            if len(top) < int(revisit.get("top_k", 3)) and all(math.hypot(c["x_m"] - t["x_m"], c["y_m"] - t["y_m"]) >= apart for t in top):
                top.append(c)
        with self._lock:
            pipeline["map"] = revisit["map"]
            pipeline["regions"] = [{k: c.get(k) for k in REGION_FIELDS} for c in top]
            if not top:
                pipeline["select_error"] = f"{revisit['map']} has no scored regions; score it first with score_map(map={revisit['map']!r})"
            self._save(job)
        return {"phase": "select", "regions": len(top)}

    def link_pending(self, job: dict) -> None:
        """Add the point of interest of every revisit image that has none yet (a link lost to a failure or an abort)."""
        pipeline = self.state(job)
        if "map" not in pipeline:
            return
        regions = {r["id"]: r for r in pipeline.get("regions") or []}
        for visit in pipeline.get("revisits", []):
            if "image" not in visit or "poi" in visit or visit["region"] not in regions:
                continue
            region = regions[visit["region"]]
            try:
                poi = self.service().call("add_poi", {
                    "map": pipeline["map"], "x_m": float(region["x_m"]), "y_m": float(region["y_m"]), "kind": "high_mag",
                    "reason": f"higher-magnification image of region {region['id']}", "links": [visit["image"], f"poi:{region['id']}"],
                    "id": f"{job['reservation_id']}:revisit:{region['id']}",
                })
                with self._lock:
                    visit["poi"] = poi.get("id")
                    visit.pop("poi_error", None)
                    self._save(job)
            except Exception as exc:  # the image is saved either way and a later step tries the link again
                visit["poi_error"] = f"{type(exc).__name__}: {exc}"

    def revisit(self, job: dict, k: int, profile: InstrumentProfile, move: Callable[[dict, float, float], bool]) -> dict | None:
        pipeline = self.state(job)
        if "regions" not in pipeline and job["resolved_plan"].get("regions"):
            pipeline["regions"] = job["resolved_plan"]["regions"]  # a plan that only revisits given points
        self.link_pending(job)
        regions = pipeline.get("regions") or []
        if k >= len(regions):
            return {"phase": "revisit", "skipped": "no such region"}
        region = regions[k]
        visits = pipeline.setdefault("revisits", [])
        visit = next((v for v in visits if v.get("region") == region["id"]), None)
        plan = job["resolved_plan"]
        revisit = plan["revisit"]
        fov = float(revisit["fov_m"])
        if visit is None:
            # The map places content at the stage reading plus the drift.
            # To put the region under the probe the stage goes to its map position minus the drift so far.
            now = profile.clock_s()
            dx, dy = drift_at(pipeline.get("drift_fit"), now - float(job.get("first_tile_s", now)))
            # With live tracking the stitch fitted only the drift that tracking left.
            # The tracker's own estimate is the rest.
            tx, ty = live_drift.estimate_at(job, now)
            dx, dy = dx + tx, dy + ty
            x, y = float(region["x_m"]) - dx, float(region["y_m"]) - dy
            x_min, x_max, y_min, y_max = plan["box_m"]
            half = fov / 2.0
            # The image must stay inside the reserved plan's box. A region near the edge is still imaged
            # with the image moved inward, as long as the region keeps a margin inside it.
            # A plan of points only has a box as wide as the image, give or take rounding.
            cx = min(max(x, x_min + half), x_max - half) if x_max - x_min >= fov - EDGE_TOLERANCE_M else None
            cy = min(max(y, y_min + half), y_max - half) if y_max - y_min >= fov - EDGE_TOLERANCE_M else None
            if cx is None or cy is None or abs(cx - x) > REVISIT_EDGE_SHIFT * half or abs(cy - y) > REVISIT_EDGE_SHIFT * half:
                note = f"region {region['id']} at ({x:.3e}, {y:.3e}) m (drift-corrected) is too close to the edge of the plan box for a {fov:.2e} m image; not visited"
                visits.append({"region": region["id"], "skipped": note})
                return {"phase": "revisit", "skipped": note}
            x, y = cx, cy
            if not move(job, x, y):
                return None
            image_key = profile.acquire_revisit(job, revisit)
            # clock_s is the instrument time the drift correction was worked out for.
            visit = {"region": region["id"], "image": image_key, "fov_m": fov, "stage_xy_m": [x, y], "clock_s": now, "drift_xy_m": [dx, dy]}
            if job.get("dry_run"):
                visit = {"region": region["id"], "acquisition": DRY_RUN_SKIPPED, "fov_m": fov, "stage_xy_m": [x, y], "clock_s": now, "drift_xy_m": [dx, dy]}
            with self._lock:  # saved before the point is added, so a resume links this image instead of taking another
                visits.append(visit)
                self._save(job)
        self.link_pending(job)
        return {"phase": "revisit", **visit}

    def zoom(self, job: dict, k: int, profile: InstrumentProfile) -> dict | None:
        """One zoom tier: find a feature on the parent, aim at it, image it at this tier's fov, place it on the parent."""
        pipeline = self.state(job)
        tiers = pipeline.setdefault("zoom", [])
        done = next((t for t in tiers if t.get("tier") == k), None)
        if done is not None:
            return self._zoom_step_result(done)  # a resumed job never images a tier twice
        zoom = job["resolved_plan"]["zoom"]
        tier = zoom["tiers"][k]
        before = tiers[-1] if tiers else None
        if before is not None and (before.get("stopped") or before.get("placement", {}).get("status") != "placed"):
            record = {"tier": k, "fov_m": float(tier["fov_m"]), "stopped": f"tier {before['tier']} ended the zoom"}
        else:
            record = self._zoom_tier(job, k, before["image"] if before else zoom["parent"], tier, profile)
            if record is None:
                return None  # aborted before imaging, and a resume does this tier again
        with self._lock:
            tiers.append(record)
            self._save(job)
        return self._zoom_step_result(record)

    @staticmethod
    def _zoom_step_result(record: dict) -> dict:
        """A tier as its step result: short enough for the trace's result field (the whole record is in the summary)."""
        placement = record.get("placement") or {}
        short = {key: placement[key] for key in ("status", "score", "ambiguity") if key in placement}
        out = {"phase": "zoom", "tier": record["tier"], "image": record.get("image"), "parent": record.get("parent"), "pixel_m": record.get("pixel_m")}
        if short:
            out["placement"] = short
        if record.get("stopped"):
            out["stopped"] = record["stopped"][:160]
        return out

    def _zoom_tier(self, job: dict, k: int, parent: str, tier: dict, profile: InstrumentProfile) -> dict | None:
        fov = float(tier["fov_m"])
        record: dict = {"tier": k, "parent": parent, "fov_m": fov, "pixel_m": fov / int(tier["imsize"]) if tier.get("imsize") else None}
        low, high = ZOOM_FEATURE_SIZE
        found = self.service().call("find_features", {"image": parent, "n": ZOOM_CANDIDATES, "min_size_m": low * fov, "max_size_m": high * fov})
        if "error" in found:
            return {**record, "stopped": f"find_features refused {parent}: {found['error'].get('message', found['error'])}"}
        if not found.get("features"):
            return {**record, "stopped": f"{parent} shows no distinct feature {low * fov:.2g} to {high * fov:.2g} m across to zoom in on"}
        # The finer image must lie inside its parent to be placed on it, so skip features near the parent's edge.
        half = fov / 2.0 / float(found["pixel_size_m"])
        rows, cols = (int(v) for v in found["shape"][:2])
        inside = [f for f in found["features"] if half <= f["row"] <= rows - 1 - half and half <= f["col"] <= cols - 1 - half]
        if not inside:
            return {**record, "stopped": f"every distinct feature on {parent} is too near its edge for a {fov:.2g} m image to lie inside it"}
        feature = inside[0]
        target = [float(feature["x_m"]), float(feature["y_m"])]
        parent_fov = float(found["pixel_size_m"]) * float(max(found["shape"]))
        record.update({"feature": feature, "aimed_xy_m": target})
        x, y = profile.current_xy()
        shift = [target[0] - x, target[1] - y]
        limit = float(profile.image_shift_limit_m)
        if max(abs(v) for v in shift) > limit:
            return {**record, "stopped": f"the feature is {max(abs(v) for v in shift):.3g} m from the stage, beyond the {limit:.3g} m the image shift reaches"}
        if self._aborting(job):
            return None
        profile.set_image_shift(job, shift)
        record["image_shift_m"] = shift
        if job.get("dry_run"):
            return {**record, "acquisition": DRY_RUN_SKIPPED}
        record["image"] = profile.acquire_revisit(job, tier)
        window = ZOOM_WINDOW_SHARE * parent_fov
        placed = self.service().call("place_image", {"image": record["image"], "parent": parent, "centre_xy_m": target, "window_m": window})
        if "error" in placed:
            placed = {"status": "unplaced", "reason": placed["error"].get("message", str(placed["error"]))}
        record["placement"] = {key: placed[key] for key in ("status", "reason", "score", "ambiguity", "centre_xy_m", "shift_from_prediction_m", "id") if key in placed}
        record["placement"]["window_m"] = window
        return record
