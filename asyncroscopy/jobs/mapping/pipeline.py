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

Analysis calls go through one session to the analysis server. Each phase waits for its analysis job with
backoff and an overall deadline that grows with the map, checking for an abort between waits. The analysis
job's id is saved in the job before waiting, so a resumed job waits for that same job (or, if another
client already started one on the same run, adopts it) instead of starting a second. If the analysis
server is down, restarted or too slow, the job still ends done with every tile safe; ``stitch_error`` says
why and how to retry. Scoring and revisits are idempotent: a resumed job never images a region twice or
adds its points twice.
"""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from collections.abc import Callable

from asyncroscopy.routines.drift import drift_at
from asyncroscopy.routines.grid import EDGE_TOLERANCE_M

from . import live_drift
from .profile import InstrumentProfile

ANALYSIS_CALL_S = 60.0  # longest one call to the analysis server may take
ANALYSIS_WAIT_S = 5.0  # longest one wait on an analysis job; abort is checked between waits
ANALYSIS_BACKOFF_S = (1.0, 2.0, 5.0, 10.0, 20.0)  # after a failed call to the analysis server
STITCH_DEADLINE_S = (600.0, 0.05)  # base, plus per tile: the longest a stitch may take before it counts as failed
SCORE_DEADLINE_S = 900.0
REVISIT_EDGE_SHIFT = 0.8  # a revisit image may sit this fraction of its half-width off a region near the box edge
ZOOM_FEATURE_SIZE = (0.25, 2.0)  # a tier aims at a feature between these multiples of its own field of view across
ZOOM_WINDOW_SHARE = 1 / 16  # place_image searches this share of the parent's field of view around the aimed point
ZOOM_CANDIDATES = 8  # features asked for at each tier; the first whose finer image fits inside the parent is aimed at
REGION_FIELDS = ("id", "x_m", "y_m", "score", "reason", "scorer_version", "window_m")
# What a dry run (plan "dry_run": true, see JobDevice) reports in place of every image and spectrum.
DRY_RUN_SKIPPED = "skipped (dry run): nothing was acquired and the sample was not exposed; the commands are in the trace with dry_run: true"


class AnalysisSession:
    """One MCP session to the analysis server, used from any thread; reopened after a failure."""

    def __init__(self, url: str) -> None:
        self.url = url
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client = None
        self._lock = threading.Lock()

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None:
                loop = asyncio.new_event_loop()
                threading.Thread(target=loop.run_forever, name="mapper-analysis", daemon=True).start()
                self._loop = loop
            return self._loop

    async def _call(self, tool: str, arguments: dict, timeout: float) -> dict:
        from fastmcp import Client

        if self._client is None:
            client = Client(self.url, timeout=timeout)
            await client.__aenter__()
            self._client = client
        try:
            result = await asyncio.wait_for(self._client.call_tool(tool, arguments, raise_on_error=False), timeout)
        except BaseException:
            await self._drop()
            raise
        if result.is_error:
            raise RuntimeError(" ".join(getattr(b, "text", "") for b in result.content) or f"{tool} failed")
        if result.structured_content is not None:
            return result.structured_content
        return json.loads("".join(getattr(b, "text", "") for b in result.content))

    async def _drop(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            try:
                await client.__aexit__(None, None, None)
            except BaseException:
                pass

    def call(self, tool: str, arguments: dict, timeout: float = ANALYSIS_CALL_S) -> dict:
        future = asyncio.run_coroutine_threadsafe(self._call(tool, arguments, timeout), self._ensure_loop())
        return future.result(timeout + 10.0)

    def close(self) -> None:
        if self._loop is not None:
            asyncio.run_coroutine_threadsafe(self._drop(), self._loop).result(10.0)
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop = None


class AnalysisFailed(RuntimeError):
    """The analysis server could not do (or finish) the job; the message says why and how to retry."""


class Pipeline:
    """The analysis phases of one Mapper's jobs. ``save(job)`` persists the job under ``lock``."""

    def __init__(
        self,
        url: str,
        stage_sigma_m: float,
        max_drift_m_per_s: float,
        aborting: Callable[[dict], bool],
        lock: threading.RLock,
        save: Callable[[dict], None],
        warn: Callable[[str], None],
    ) -> None:
        self.url = url
        self.stage_sigma_m = float(stage_sigma_m)
        self.max_drift_m_per_s = float(max_drift_m_per_s)
        self._aborting = aborting
        self._lock = lock
        self._save = save
        self._warn = warn
        self._session: AnalysisSession | None = None

    # -- calls -----------------------------------------------------------
    def call(self, tool: str, arguments: dict) -> dict:
        """One call to the analysis server over this pipeline's session."""
        if self._session is None:
            self._session = AnalysisSession(self.url)
        return self._session.call(tool, arguments)

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

    @staticmethod
    def state(job: dict) -> dict:
        """The job's pipeline record (analysis job ids, map, regions, revisits, zoom tiers, errors)."""
        return job.setdefault("pipeline", {})

    def stitch_arguments(self, run: str) -> dict:
        return {"run": run, "stage_sigma_m": self.stage_sigma_m, "max_drift_m_per_s": self.max_drift_m_per_s, "eds_pois": True}

    def _submit(self, job: dict, slot: str, tool: str, arguments: dict) -> str:
        """Start (or adopt) an analysis job and save its id in the job before waiting on it."""
        pipeline = self.state(job)
        if pipeline.get(slot):
            return pipeline[slot]  # a resumed job waits for the analysis job it already started
        started = None
        last: Exception | None = None
        for delay in (0.0, *ANALYSIS_BACKOFF_S):
            time.sleep(delay)
            if self._aborting(job):
                raise AnalysisFailed("aborted")
            try:
                started = self.call(tool, arguments)
                break
            except Exception as exc:  # down, restarting, or the call timed out: try again; a busy reply below adopts
                last = exc
        if started is None:
            raise AnalysisFailed(f"the analysis server at {self.url} did not answer ({type(last).__name__}: {last})")
        if "error" in started:
            if started["error"].get("code") == "busy" and started.get("job_id"):
                started = {"job_id": started["job_id"]}  # someone (or this job, before a timeout) is already on it
            else:
                raise AnalysisFailed(started["error"].get("message", str(started["error"])))
        with self._lock:
            pipeline[slot] = started["job_id"]
            self._save(job)
        return started["job_id"]

    def _await(self, job: dict, job_id: str, deadline_s: float) -> dict | None:
        """Results of an analysis job; None if this job was aborted; AnalysisFailed if it failed, vanished or took too long."""
        deadline = time.monotonic() + deadline_s
        failures = 0
        while True:
            if self._aborting(job):
                try:
                    self.call("analysis_job_abort", {"job_id": job_id})
                except Exception as exc:
                    self._warn(f"could not abort analysis job {job_id}: {exc}")
                return None
            if time.monotonic() > deadline:
                raise AnalysisFailed(f"analysis job {job_id} did not finish within {deadline_s:.0f} s")
            try:
                status = self.call("analysis_job_status", {"job_id": job_id, "wait_s": ANALYSIS_WAIT_S})
                failures = 0
            except Exception as exc:
                failures += 1
                if failures > len(ANALYSIS_BACKOFF_S):
                    raise AnalysisFailed(f"lost the analysis server at {self.url} ({type(exc).__name__}: {exc})") from exc
                time.sleep(ANALYSIS_BACKOFF_S[failures - 1])
                continue
            if "error" in status or "state" not in status:
                # unknown_job: the server restarted and forgot the job.
                raise AnalysisFailed(f"the analysis server no longer knows job {job_id} ({(status.get('error') or {}).get('message', status)}); it may have restarted")
            if status["state"] in {"done", "failed", "aborted", "released"}:
                break
        if status["state"] != "done":
            raise AnalysisFailed(f"the analysis job ended {status['state']}: {(status.get('last_error') or {}).get('message', '')}")
        return self.call("analysis_job_results", {"job_id": job_id})

    # -- phases ----------------------------------------------------------
    def stitch(self, job: dict, run: str, tiles: int) -> dict | None:
        pipeline = self.state(job)
        deadline = STITCH_DEADLINE_S[0] + STITCH_DEADLINE_S[1] * tiles
        try:
            results = self._await(job, self._submit(job, "stitch_job", "assemble_map", self.stitch_arguments(run)), deadline)
        except AnalysisFailed as exc:  # the tiles are safe, the job goes on
            if str(exc) == "aborted":
                return None
            pipeline["stitch_error"] = f"{exc}. The tiles are in {run}; stitch them later with assemble_map(run={run!r})."
            pipeline.pop("stitch_job", None)
            return {"phase": "stitch", "ok": False}
        if results is None:
            return None  # aborted while stitching; a resume waits for (or restarts) the same stitch
        pipeline["map"] = results["result_keys"]["map"]
        summary = results.get("summary", {})
        pipeline["drift_fit"] = summary.get("drift")  # kept for revisits; the reply shows only its totals
        pipeline["stitching"] = {k: v for k, v in summary.items() if k in ("tiles", "levels", "canvas_shape", "matches_used", "median_residual_px", "p95_residual_px", "fallback_tiles", "eds_pois", "solver")}
        if summary.get("drift"):
            pipeline["stitching"]["drift"] = {k: summary["drift"][k] for k in ("total_nm", "mean_rate_nm_per_s") if k in summary["drift"]}
        with self._lock:
            job["result_keys"]["map"] = pipeline["map"]
        return {"phase": "stitch", "ok": True, "map": pipeline["map"]}

    def stitch_released(self, job: dict, run: str) -> None:
        """Start a stitch of a released job's tiles; nobody waits for it (results name its analysis job)."""
        try:
            started = self.call("assemble_map", self.stitch_arguments(run))
            with self._lock:
                job["result_keys"]["map_job"] = started.get("job_id")
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
        # The job's tag makes scoring idempotent: a retry replaces the regions it added before.
        arguments = {"map": pipeline["map"], "top_k": n, "tag": job["reservation_id"]}
        try:
            results = self._await(job, self._submit(job, "score_job", "score_map", arguments), SCORE_DEADLINE_S)
        except AnalysisFailed as exc:
            if str(exc) == "aborted":
                return None
            pipeline["score_error"] = f"{exc}. Score again with score_map(map={pipeline['map']!r}, top_k={n})."
            pipeline.pop("score_job", None)
            return {"phase": "score", "ok": False}
        if results is None:
            return None
        pipeline["regions"] = [{k: p.get(k) for k in REGION_FIELDS} for p in results.get("summary", {}).get("added_pois", [])]
        return {"phase": "score", "ok": True, "regions": len(pipeline["regions"])}

    def select(self, job: dict) -> dict:
        """Take an existing map's best scored regions (revisit-only plans with a map)."""
        pipeline = self.state(job)
        if "regions" in pipeline:
            return {"phase": "select", "regions": len(pipeline["regions"])}  # already chosen before a resume
        revisit = job["resolved_plan"]["revisit"]
        reply = self.call("list_pois", {"map": revisit["map"], "kind": "candidate", "limit": 500})
        candidates = sorted((p for p in reply.get("pois", []) if p.get("score") is not None), key=lambda p: p["score"], reverse=True)
        # Scoring a map twice stores the same regions again under new ids, and two regions closer than half a
        # field of view share one image: each is imaged once, and the next best distinct region takes its place.
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
                poi = self.call(
                    "add_poi",
                    {"map": pipeline["map"], "x_m": float(region["x_m"]), "y_m": float(region["y_m"]), "kind": "high_mag", "reason": f"higher-magnification image of region {region['id']}", "links": [visit["image"], f"poi:{region['id']}"], "id": f"{job['reservation_id']}:revisit:{region['id']}"},
                )
                with self._lock:
                    visit["poi"] = poi.get("id")
                    visit.pop("poi_error", None)
                    self._save(job)
            except Exception as exc:  # the image is saved either way; the next revisit step or the finish tries again
                visit["poi_error"] = f"{type(exc).__name__}: {exc}"

    def revisit(self, job: dict, k: int, profile: InstrumentProfile, move: Callable[[dict, float, float], bool]) -> dict | None:
        pipeline = self.state(job)
        if "regions" not in pipeline and job["resolved_plan"].get("regions"):
            pipeline["regions"] = job["resolved_plan"]["regions"]  # revisit-only plan with given points
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
            # The map places content at stage reading + drift(t) (pyTEMlib's fit), so to put the region under
            # the probe now the stage goes to its map position minus the drift, extrapolated to now on the
            # instrument's clock (the one the tiles were timed with).
            now = profile.clock_s()
            dx, dy = drift_at(pipeline.get("drift_fit"), now - float(job.get("first_tile_s", now)))
            # With live tracking the tiles were corrected, so the stitch fitted only what tracking left; the
            # tracker's own estimate (on the same clock, from the same first tile) is the rest.
            tx, ty = live_drift.estimate_at(job, now)
            dx, dy = dx + tx, dy + ty
            x, y = float(region["x_m"]) - dx, float(region["y_m"]) - dy
            x_min, x_max, y_min, y_max = plan["box_m"]
            half = fov / 2.0
            # The image must stay inside the approved box. A region near its edge is still imaged, off centre,
            # with the image moved inward, as long as the region keeps a margin inside that image.
            # A points-only plan's box is the point +/- half the image, so its width equals fov but for rounding.
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
            # clock_s: the instrument time the drift correction (drift_xy_m) was extrapolated to.
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
                return None  # aborted before imaging; a resume does this tier again
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
        found = self.call("find_features", {"image": parent, "n": ZOOM_CANDIDATES, "min_size_m": low * fov, "max_size_m": high * fov})
        if "error" in found:
            return {**record, "stopped": f"find_features refused {parent}: {found['error'].get('message', found['error'])}"}
        if not found.get("features"):
            return {**record, "stopped": f"{parent} shows no distinct feature {low * fov:.2g} to {high * fov:.2g} m across to zoom in on"}
        # The finer image must lie inside its parent to be placed on it: a feature near the parent's edge is passed over.
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
        placed = self.call("place_image", {"image": record["image"], "parent": parent, "centre_xy_m": target, "window_m": window})
        if "error" in placed:
            placed = {"status": "unplaced", "reason": placed["error"].get("message", str(placed["error"]))}
        record["placement"] = {key: placed[key] for key in ("status", "reason", "score", "ambiguity", "centre_xy_m", "shift_from_prediction_m", "id") if key in placed}
        record["placement"]["window_m"] = window
        return record
