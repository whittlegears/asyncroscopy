"""Prose for a model from the instrument server's structured replies.

The MCP server answers with facts: a job's ``state``, the ``next_actions`` to take ({"tool", "args"}),
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
    return out


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


__all__ = ["DONE_NOTE", "WAIT_NOTES", "render", "render_error", "render_reply"]
