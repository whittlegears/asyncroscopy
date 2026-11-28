"""From a rehearsal on a digital twin to a signed-off run on the instrument (``docs/MCP/twin_to_instrument.md``).

A workflow server does three things here, each through the operator endpoint of the stack it acts on
(``endpoint_info.urls.operator``), never through a tool an agent is given:

- **Before and after a rehearsal** it reads the twin's truth (the operator-only ``export_ground_truth``), so the
  rehearsal record keeps the twin's time, dose and twin score (``asyncroscopy.sim.truth_score``). The score is kept in
  the library's rehearsal record only, which only ``review_workflow`` on the workflow operator endpoint returns.
- **When an approved workflow starts on a real instrument** it pre-approves, for that run only, every instrument call
  the rehearsal the person signed off made (``preapprove_run``). The run's calls carry the run id and a secret run key
  (``X-Asyncroscopy-Run``, ``X-Asyncroscopy-Run-Key``); the key is in no reply, so no other caller can use the
  pre-approvals even when it knows the run id.
- **When that run ends or is stopped** it removes them (``end_run_preapproval``). They also expire after the stack's
  ``mcp.approval_ttl_s``; a resumed run is granted them again, for the same run id.
"""

from __future__ import annotations

import fnmatch
import json
import secrets
from pathlib import Path
from typing import Any

from asyncroscopy.agents.mcp_tools import McpServer, result_text
from asyncroscopy.safety.permissions import run_key_sha256
from asyncroscopy.sim.truth_score import (
    score_label,
    score_version,
    truth_marker,
    twin_score,
    twin_time_and_dose,
)

from .library import Library


def new_run_key() -> str:
    return secrets.token_hex(16)


def signoff_name(workflow: str, hash_: str) -> str:
    return f"workflow_signoff:{workflow}@{hash_}"


def _marker_path(state_dir: Path, run_id: str) -> Path:
    return Path(state_dir) / "runs" / f"{run_id}.twin_marker.json"


async def twin_truth(operator: McpServer | None) -> tuple[dict | None, str | None]:
    """(the twin's ``export_ground_truth`` document, None) or (None, why it could not be read)."""
    if operator is None:
        return None, "the stack reports no operator endpoint (endpoint_info.urls.operator), where export_ground_truth is"
    try:
        names = sorted(t.name for t in await operator.list_tools())
    except Exception as exc:
        return None, f"the operator endpoint could not be reached: {type(exc).__name__}: {exc}"
    found = [n for n in names if fnmatch.fnmatchcase(n, "*_export_ground_truth")]
    if not found:
        return None, "the operator endpoint offers no export_ground_truth"
    problems = []
    for name in found:
        try:
            result = await operator.call(name, {})
        except Exception as exc:
            problems.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        text = result_text(result)
        try:
            truth = json.loads(text)
        except ValueError:
            problems.append(f"{name}: {text[:200]}")
            continue
        if isinstance(truth, dict) and not result.is_error:
            return truth, None
        problems.append(f"{name}: {text[:200]}")
    return None, "; ".join(problems)


async def mark_rehearsal(state_dir: Path, run_id: str, operator: McpServer | None) -> None:
    """Keep the twin's truth marker from before a rehearsal (``truth_marker``), beside the run's record. A resumed
    rehearsal keeps the one from its start."""
    path = _marker_path(state_dir, run_id)
    if path.exists():
        return
    truth, why = await twin_truth(operator)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"marker": truth_marker(truth)} if truth is not None else {"why": why}), encoding="utf-8")


async def rehearsal_results(state_dir: Path, record: dict, operator: McpServer | None) -> dict:
    """A finished rehearsal's results: the runner's (data keys, map, calls, wall time) with the twin's time, dose and
    twin score added. What the twin could not give is null, with the reason in ``twin_score.why``."""
    results = dict(record.get("results") or {})
    path = _marker_path(state_dir, str(record.get("run_id")))
    try:
        kept = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        kept = {"why": "no truth marker was taken before this run"}
    truth, why = await twin_truth(operator) if "marker" in kept else (None, kept.get("why"))
    if truth is None:
        results.update(instrument_s=None, instrument_s_from=None, dose=None,
                       twin_score={"label": score_label, "version": score_version, "known": False, "why": why})
        return results
    clock, dose = twin_time_and_dose(kept["marker"], truth)
    results.update(instrument_s=clock, instrument_s_from="the twin's simulated clock" if clock is not None else None,
                   dose={**dose, "from": "the twin's dose map"} if dose is not None else None,
                   twin_score=twin_score(kept["marker"], truth, record))
    if record.get("state") != "waiting":
        path.unlink(missing_ok=True)
    return results


def signed_off_rehearsal(library: Library, name: str) -> tuple[dict | None, dict | None, str | None]:
    """(the approval, the rehearsal it was given on, None) for an approved workflow, or why there is no such rehearsal."""
    approval = library.approval(name)
    if not approval:
        return None, None, f"{name} has no approval"
    wanted = (approval.get("rehearsed") or {}).get("run_id")
    if not wanted:
        return approval, None, f"{name} was approved without a rehearsal (on the command line), so no call is pre-approved"
    found = next((r for r in library.rehearsals(name, approval.get("hash")) if r.get("run_id") == wanted), None)
    if found is None:
        return approval, None, f"the rehearsal {wanted} the approval was given on is no longer kept"
    return approval, found, None


async def grant(operator: McpServer | None, library: Library, name: str, run_id: str, run_key: str) -> dict:
    """Pre-approve, for the hardware run ``run_id``, the calls of the rehearsal ``name`` was signed off on. Returns
    the run record's ``preapproval``: {"granted": true, "approved_by", "approver", "rehearsal_run_id", "calls",
    "expires_at"} or {"granted": false, "why"}. Without it every call is permissioned as usual."""
    approval, rehearsal, why = signed_off_rehearsal(library, name)
    if rehearsal is None:
        return {"granted": False, "why": why}
    if operator is None:
        return {"granted": False, "why": "the instrument reports no operator endpoint (endpoint_info.urls.operator) to pre-approve its calls on"}
    calls = [{"tool": c["tool"], "arguments": c["arguments"]} for c in rehearsal.get("calls_made") or [] if c.get("server") == "instrument"]
    arguments = {"run_id": run_id, "run_key_sha256": run_key_sha256(run_key), "calls": calls,
                 "approved_by": signoff_name(name, str(approval["hash"])), "approver": str(approval.get("by") or ""),
                 "workflow": name, "hash": str(approval["hash"]), "rehearsal_run_id": rehearsal.get("run_id")}
    try:
        result = await operator.call("preapprove_run", arguments)
    except Exception as exc:
        return {"granted": False, "why": f"the operator endpoint could not be reached: {type(exc).__name__}: {exc}"}
    granted = getattr(result, "structured_content", None)
    if result.is_error or not isinstance(granted, dict):
        return {"granted": False, "why": f"the operator endpoint refused the pre-approval: {result_text(result)[:300]}"}
    return {"granted": True, **{k: granted.get(k) for k in ("approved_by", "approver", "rehearsal_run_id", "calls", "granted_at", "expires_at")},
            "rehearsal_inputs": rehearsal.get("inputs") or {}}


async def end(operator: McpServer | None, run_id: str) -> dict[str, Any] | None:
    """Remove a hardware run's pre-approval; None when it could not be reached (the pre-approval then expires)."""
    if operator is None:
        return None
    try:
        result = await operator.call("end_run_preapproval", {"run_id": run_id})
    except Exception:
        return None
    return getattr(result, "structured_content", None)
