"""From a rehearsal on a digital twin to a signed-off run on the instrument, always through the operator endpoint of the
stack acted on (``endpoint_info.urls.operator``); ``docs/MCP/twin_to_instrument.md`` is the contract."""

from __future__ import annotations

import fnmatch
import json
import secrets
from pathlib import Path
from typing import Any

from asyncroscopy.agents.mcp_tools import McpServer, result_text
from asyncroscopy.safety.permissions import canonical_json, run_key_sha256
from asyncroscopy.sim.truth_score import (
    score_label,
    score_version,
    truth_marker,
    twin_score,
    twin_time_and_dose,
)

from .library import Library, matched_calls


def new_run_key() -> str:
    return secrets.token_hex(16)


def signoff_name(workflow: str, hash_: str) -> str:
    return f"workflow_signoff:{workflow}@{hash_}"


def _marker_path(state_dir: Path, run_id: str) -> Path:
    return Path(state_dir) / "runs" / f"{run_id}.twin_marker.json"


async def twin_truth(operator: McpServer | str) -> tuple[dict | None, str | None]:
    """(the twin's ``export_ground_truth`` document, None) or (None, why it could not be read)."""
    if isinstance(operator, str):
        return None, operator
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


async def mark_rehearsal(state_dir: Path, run_id: str, operator: McpServer | str) -> None:
    """Keep the twin's truth marker from before a rehearsal; a resumed rehearsal keeps the one from its start."""
    path = _marker_path(state_dir, run_id)
    if path.exists():
        return
    truth, why = await twin_truth(operator)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"marker": truth_marker(truth)} if truth is not None else {"why": why}), encoding="utf-8")


def forget_rehearsal(state_dir: Path, run_id: str) -> None:
    _marker_path(state_dir, run_id).unlink(missing_ok=True)


async def rehearsal_results(state_dir: Path, record: dict, operator: McpServer | str) -> dict:
    """The runner's results with the twin's time, dose and twin score added."""
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
    """(the approval, the rehearsal it was given on, None), or why there is none."""
    approval = library.approval(name)
    if not approval:
        return None, None, f"{name} has no approval"
    rehearsed = approval.get("rehearsed")
    if not rehearsed:
        return approval, None, f"{name} was approved without a rehearsal (on the command line), so no call is pre-approved"
    if "calls_made" not in rehearsed:
        return approval, None, f"{name} was approved before an approval kept its rehearsal's calls; approve it again to pre-approve them"
    return approval, rehearsed, None


def input_problem(library: Library, name: str, inputs: dict) -> str | None:
    """Why a hardware run with ``inputs`` (defaults filled in) is refused: they are not the signed-off rehearsal's."""
    rehearsed = (library.approval(name) or {}).get("rehearsed")
    if not rehearsed or canonical_json(inputs) == canonical_json(rehearsed.get("inputs") or {}):
        return None
    return (f"{name} was signed off for the inputs its rehearsal ran with ({canonical_json(rehearsed.get('inputs') or {})}, rehearsal "
            f"{rehearsed.get('run_id')}); these are {canonical_json(inputs)}. Other inputs need a new rehearsal on the twin and a new sign-off.")


async def grant(operator: McpServer | str, library: Library, name: str, run_id: str, run_key: str, info: dict) -> dict:
    """Pre-approve for ``run_id`` the signed-off rehearsal's instrument calls whose tool still matches the recorded one;
    returns the run record's ``preapproval``."""
    approval, rehearsal, why = signed_off_rehearsal(library, name)
    if rehearsal is None:
        return {"granted": False, "why": why}
    if isinstance(operator, str):
        return {"granted": False, "why": operator}
    matched = matched_calls(rehearsal, info)
    calls = [{"tool": match["instrument_tool"], "arguments": match["arguments"], "count": match.get("count") or 1}
             for match, signed in zip(matched, rehearsal["calls_made"])
             if match.get("server") == "instrument" and match.get("instrument_tool") and match["instrument_tool"] == signed.get("instrument_tool")]
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


async def end(operator: McpServer | str, run_id: str) -> dict[str, Any] | None:
    """Remove a hardware run's pre-approval; None when unreachable, so it expires instead."""
    if isinstance(operator, str):
        return None
    try:
        result = await operator.call("end_run_preapproval", {"run_id": run_id})
    except Exception:
        return None
    return getattr(result, "structured_content", None)
