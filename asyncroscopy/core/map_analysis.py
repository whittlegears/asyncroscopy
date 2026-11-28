"""Stitching and scoring a map on the analysis server: the one path the Mapper, the ``stitch_map`` block and the mapping
graph share. Each passes its own ``analyse``, an async ``(tool, arguments) -> reply`` that raises ``AnalysisUnreachable``
when the server did not answer (retried with backoff). A wait fails only when the job's progress stops for the stall limit.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

analysis_wait_s = 5.0
analysis_backoff_s = (1.0, 2.0, 5.0, 10.0, 20.0)
stitch_stall_s = 600.0
score_stall_s = 900.0
ended_states = frozenset({"done", "failed", "aborted", "released"})

Analyse = Callable[[str, dict], Awaitable[Any]]


class AnalysisUnreachable(RuntimeError):
    pass


class AnalysisFailed(RuntimeError):
    """The analysis server could not do, or finish, the job; the message says why."""


def _never() -> bool:
    return False


def _ignore(message: str) -> None:
    return None


def stitch_arguments(run: str, refine: bool | None = None, stage_sigma_m: float | None = None, max_drift_m_per_s: float | None = None,
                     eds_pois: bool | None = None) -> dict:
    """assemble_map's arguments; None ones are left out so the server's defaults apply."""
    given = {"refine": refine, "stage_sigma_m": stage_sigma_m, "max_drift_m_per_s": max_drift_m_per_s, "eds_pois": eds_pois}
    return {"run": run, **{name: value for name, value in given.items() if value is not None}}


async def start_job(analyse: Analyse, tool: str, arguments: dict, stopping: Callable[[], bool] = _never) -> str | None:
    """Start an analysis job, or adopt the one already running on the same input; its id, or None when stopping."""
    last: AnalysisUnreachable | None = None
    started: Any = None
    for delay in (0.0, *analysis_backoff_s):
        await asyncio.sleep(delay)
        if stopping():
            return None
        try:
            started = await analyse(tool, arguments)
            break
        except AnalysisUnreachable as exc:
            last = exc
    else:
        raise AnalysisFailed(f"the analysis server did not answer {tool} ({last})")
    if not isinstance(started, dict):
        raise AnalysisFailed(f"{tool} did not start: {str(started)[:300]}")
    error = started.get("error")
    if error:
        if isinstance(error, dict) and error.get("code") == "busy" and started.get("job_id"):
            return str(started["job_id"])
        raise AnalysisFailed(error.get("message", str(error)) if isinstance(error, dict) else str(error))
    if not started.get("job_id"):
        raise AnalysisFailed(f"{tool} did not start: {str(started)[:300]}")
    return str(started["job_id"])


async def wait_for_job(analyse: Analyse, job_id: str, stall_s: float, stopping: Callable[[], bool] = _never,
                       warn: Callable[[str], None] = _ignore) -> dict | None:
    """A finished analysis job's results; None when stopping (the job is aborted); AnalysisFailed if it failed, vanished or stalled."""
    deadline = time.monotonic() + stall_s
    last_progress = None
    failures = 0
    while True:
        if stopping():
            try:
                await analyse("analysis_job_abort", {"job_id": job_id})
            except Exception as exc:
                warn(f"could not abort analysis job {job_id}: {exc}")
            return None
        if time.monotonic() > deadline:
            raise AnalysisFailed(f"analysis job {job_id} made no progress for {stall_s:.0f} s")
        try:
            status = await analyse("analysis_job_status", {"job_id": job_id, "wait_s": analysis_wait_s})
            failures = 0
        except AnalysisUnreachable as exc:
            failures += 1
            if failures > len(analysis_backoff_s):
                raise AnalysisFailed(f"lost the analysis server ({exc})") from exc
            await asyncio.sleep(analysis_backoff_s[failures - 1])
            continue
        if not isinstance(status, dict) or "error" in status or "state" not in status:
            said = (status.get("error") or {}).get("message", status) if isinstance(status, dict) else status
            raise AnalysisFailed(f"the analysis server no longer knows job {job_id} ({said}); it may have restarted")
        if status["state"] in ended_states:
            break
        progress = (status.get("progress") or {}).get("done")
        if progress != last_progress:
            last_progress, deadline = progress, time.monotonic() + stall_s
    if status["state"] != "done":
        raise AnalysisFailed(f"the analysis job ended {status['state']}: {(status.get('last_error') or {}).get('message', '')}")
    results = await analyse("analysis_job_results", {"job_id": job_id})
    if not isinstance(results, dict):
        raise AnalysisFailed(f"analysis job {job_id} gave no results: {str(results)[:300]}")
    return results


async def _run_job(analyse: Analyse, tool: str, arguments: dict, stall_s: float, job_id: str | None,
                   on_started: Callable[[str], None] | None, stopping: Callable[[], bool], warn: Callable[[str], None]) -> dict | None:
    if job_id is None:
        job_id = await start_job(analyse, tool, arguments, stopping)
        if job_id is None:
            return None
        if on_started is not None:
            on_started(job_id)
    return await wait_for_job(analyse, job_id, stall_s, stopping, warn)


async def stitch_map(analyse: Analyse, run: str, refine: bool | None = None, stage_sigma_m: float | None = None,
                     max_drift_m_per_s: float | None = None, eds_pois: bool | None = None, job_id: str | None = None,
                     on_started: Callable[[str], None] | None = None, stopping: Callable[[], bool] = _never,
                     warn: Callable[[str], None] = _ignore) -> dict | None:
    """Stitch ``run`` into a map and wait for it: {"map": its key, "summary": the stitch's summary}; None when stopping."""
    arguments = stitch_arguments(run, refine, stage_sigma_m, max_drift_m_per_s, eds_pois)
    results = await _run_job(analyse, "assemble_map", arguments, stitch_stall_s, job_id, on_started, stopping, warn)
    if results is None:
        return None
    key = (results.get("result_keys") or {}).get("map")
    if not key:
        raise AnalysisFailed(f"the stitching job gave no map: {str(results)[:300]}")
    return {"map": key, "summary": results.get("summary") or {}}


async def score_map(analyse: Analyse, map: str, top_k: int, tag: str | None = None, job_id: str | None = None,
                    on_started: Callable[[str], None] | None = None, stopping: Callable[[], bool] = _never,
                    warn: Callable[[str], None] = _ignore) -> dict | None:
    """Score ``map`` for its ``top_k`` best regions and wait; ``tag`` makes a retry replace the points it added before."""
    arguments = {"map": map, "top_k": int(top_k), **({"tag": tag} if tag else {})}
    results = await _run_job(analyse, "score_map", arguments, score_stall_s, job_id, on_started, stopping, warn)
    if results is None:
        return None
    summary = results.get("summary") or {}
    return {"pois": [p for p in summary.get("added_pois") or [] if isinstance(p, dict)], "summary": summary}
