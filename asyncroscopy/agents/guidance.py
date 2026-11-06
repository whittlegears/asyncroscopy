"""Prose for a model from the instrument and analysis servers' structured replies.

The MCP servers answer with facts: a job's ``state``, the ``next_actions`` to take ({"tool", "args"}; an
action on another server also names it, {"server": "instrument", "tool" or "command", "args"}, where
"command" is a device command whose tool name carries the device's prefix),
``resume_with`` (the call that continues an aborted job) and ``warnings`` ({"code", ...}). Agent
programs can act on those directly. A small model does better with a sentence telling it what to do
and what not to, so the agent graphs turn the fields into the notes below before the model sees a
reply (``asyncroscopy.agents.mcp_tools`` applies ``render`` to every tool reply and error).

The wording was chosen from benchmark runs with qwen3:8b; change it here, not in the server.
"""

from __future__ import annotations

import json

# What the model is told when a wait ends before the job does, by job state.
WAIT_NOTES = {
    "queued": "Still working, which is normal for a long job: call wait_for_job again. Abort only if the user asked to stop.",
    "running": "Still working, which is normal for a long job: call wait_for_job again. Abort only if the user asked to stop.",
    "aborting": "Abort in progress: call wait_for_job again to see the job stop.",
    "paused_suspended": "Paused by a safety condition ({reason}); it resumes by itself when the condition clears. Call wait_for_job again, or abort if the user asked to stop.",
    "paused_awaiting_approval": "Waiting for an operator to approve the next step: call wait_for_job again. Do not retry or start another job.",
    "paused": "Paused by a person, with the instrument waiting safely; it goes on when they continue it. Call wait_for_job again; do not start it yourself or start another job.",
}
# A finished job's reply: a small model handed a finished map stitched it again and then ran the whole map again.
DONE_NOTE = ("Done: every phase of the plan ran (a map is already stitched, and scored when the plan asked). Answer from this "
             "reply and its result_keys; nothing needs running or stitching again.")
STRUCTURED = ("next_actions", "resume_with", "warnings")
# Analysis server replies (asyncroscopy.analysis_server) whose next step is on the instrument endpoint.
UNKNOWN_ANALYSIS_JOB_NOTE = ("not a stitching or scoring job: a job id from Mapper_start belongs to the instrument, "
                             "so use wait_for_job or Mapper_job_status there")
BUSY_ANALYSIS_JOB_NOTE = "wait for job {job_id} (analysis_job_status) instead of starting another"
SAMPLE_LOAD_NOTE = "before moving there, check the instrument's get_sample gives this sample_load_id: another load puts the sample elsewhere"
BEAM_BLANKED_HINT = "this image was taken with the beam blanked (beam_current_pa 0), so it shows only noise: unblank_beam and take it again"


def _codes(reply: dict) -> dict[str, dict]:
    return {w["code"]: w for w in reply.get("warnings") or [] if isinstance(w, dict) and w.get("code")}


def _call_text(action: dict) -> str:
    return json.dumps({"device": action["args"].get("device"), "resume": action["args"].get("resume")})


def render_reply(tool: str, args: dict, reply: dict) -> dict:
    """``reply`` with its structured fields turned into the notes a model reads (and the fields removed)."""
    if not any(k in reply for k in STRUCTURED):
        return reply
    out = {k: v for k, v in reply.items() if k not in STRUCTURED}
    codes = _codes(reply)
    actions = [a for a in reply.get("next_actions") or [] if isinstance(a, dict)]
    state = reply.get("state")
    resume = reply.get("resume_with")
    if isinstance(resume, dict):
        text = f"run_job with {_call_text(resume)} and no plan: it goes on from the next step"
        out["to_resume"] = text if state == "aborted" else f"once wait_for_job says aborted, {text}"
    if "finished" in codes:
        out["note"] = DONE_NOTE
    elif "results_unreadable" in codes:
        out["note"] = f"done, but its results could not be read: {codes['results_unreadable'].get('detail', '')}"
    elif any(a.get("tool") == "wait_for_job" for a in actions) and state in WAIT_NOTES:
        job_id = next(a["args"].get("job_id") for a in actions if a.get("tool") == "wait_for_job")
        abort = next((a["tool"] for a in actions if str(a.get("tool", "")).endswith("_abort")), None)
        if tool == "run_job" and abort is not None:
            out["note"] = f"The job runs in the background: give the user its job_id. {abort} stops it; wait_for_job waits for it."
        elif tool == "run_job" and state in ("queued", "running"):
            phases = (reply.get("estimate") or {}).get("phases")
            out["note"] = (f"Still running, which is normal: call wait_for_job with {{\"job_id\": \"{job_id}\"}} now. The job does every phase "
                           f"of its plan itself{' (' + ', '.join(phases) + ')' if phases else ''}; other calls for it meanwhile repeat or spoil its work.")
        else:
            out["note"] = WAIT_NOTES[state].format(reason=(reply.get("suspended") or {}).get("reason", ""))
    _render_analysis(reply, actions, codes, out)
    return out


def _render_analysis(reply: dict, actions: list[dict], codes: dict[str, dict], out: dict) -> None:
    """The notes for the analysis server's structured fields (``out`` is changed in place)."""
    error = reply.get("error") if isinstance(reply.get("error"), dict) else {}
    on_instrument = [a for a in actions if a.get("server") == "instrument"]
    if error.get("code") == "unknown_job" and any(a.get("tool") == "wait_for_job" for a in on_instrument):
        out["note"] = UNKNOWN_ANALYSIS_JOB_NOTE
    elif error.get("code") == "busy":
        waits = [a for a in actions if a.get("tool") == "analysis_job_status"]
        if waits:
            out["note"] = BUSY_ANALYSIS_JOB_NOTE.format(job_id=waits[0].get("args", {}).get("job_id"))
    if any(a.get("command") == "get_sample" for a in on_instrument):
        out["note"] = "; ".join(n for n in (out.get("note"), SAMPLE_LOAD_NOTE) if n)
    if "beam_blanked" in codes and any(a.get("command") == "unblank_beam" for a in on_instrument):
        out["hint"] = BEAM_BLANKED_HINT


def render_error(text: str) -> str:
    """An error's text with a structured error the server sent (beam_blanked) turned into the sentence a model reads."""
    start = text.find("{")
    if start < 0:
        return text
    try:
        body, end = json.JSONDecoder().raw_decode(text[start:])
    except ValueError:
        return text
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict) or error.get("code") != "beam_blanked":
        return text
    actions = [a.get("tool") for a in body.get("next_actions") or [] if isinstance(a, dict)]
    unblank = actions[0] if actions else "the beam unblank command"
    retake = actions[1] if len(actions) > 1 else "the acquisition"
    key = next((w.get("key") for w in body.get("warnings") or [] if isinstance(w, dict) and w.get("code") == "noise_only_file"), "")
    # The retake comes first and the key last: a model given the key first viewed and reported the noise.
    prose = (f"Not acquired: the beam is blanked. Next call {unblank}, then {retake} again with the same arguments, and use the key "
             f"that call returns. (The noise-only file {key} was saved; do not view or report it.)")
    return text[:start] + prose + text[start + end:]


def render(tool: str, args: dict | None, text: str, error: bool = False) -> str:
    """A tool reply's text (or an error's) as the model should read it; anything unstructured is returned unchanged."""
    if error:
        return render_error(text)
    try:
        reply = json.loads(text)
    except (TypeError, ValueError):
        return text
    if not isinstance(reply, dict):
        return text
    rendered = render_reply(tool, args or {}, reply)
    return text if rendered is reply else json.dumps(rendered)


__all__ = ["BEAM_BLANKED_HINT", "BUSY_ANALYSIS_JOB_NOTE", "DONE_NOTE", "SAMPLE_LOAD_NOTE", "UNKNOWN_ANALYSIS_JOB_NOTE", "WAIT_NOTES",
           "render", "render_error", "render_reply"]
