"""asyncroscopy.agents.guidance: the server's structured replies as the sentences a small model reads."""

import json

from asyncroscopy.agents.guidance import DONE_NOTE, WAIT_NOTES, render


def test_a_reply_without_structured_fields_is_left_as_it_came():
    text = '{"job_id": "j1", "state": "running"}'
    assert render("wait_for_job", {}, text) == text
    assert render("DigitalTwin_get_stage", {}, "[0.0, 0.0]") == "[0.0, 0.0]"
    assert render("x", {}, "not json") == "not json"
    assert render("x", {}, "Error: plain failure", error=True) == "Error: plain failure"


def test_a_still_running_wait_becomes_the_state_note_and_loses_the_fields():
    reply = {"job_id": "j1", "state": "paused_awaiting_approval", "next_actions": [{"tool": "wait_for_job", "args": {"job_id": "j1"}}]}
    shown = json.loads(render("wait_for_job", {"job_id": "j1"}, json.dumps(reply)))
    assert shown == {"job_id": "j1", "state": "paused_awaiting_approval", "note": WAIT_NOTES["paused_awaiting_approval"]}


def test_a_finished_job_says_nothing_needs_running_again():
    reply = {"job_id": "j1", "state": "done", "result_keys": {"map": "m"}, "next_actions": [], "warnings": [{"code": "finished"}]}
    assert json.loads(render("run_job", {}, json.dumps(reply)))["note"] == DONE_NOTE


def test_a_blanked_acquisition_error_reads_as_the_retake():
    body = {"error": {"code": "beam_blanked", "message": "Not acquired: the beam is blanked."}, "state": "not_acquired",
            "next_actions": [{"tool": "Twin_unblank_beam", "args": {}}, {"tool": "Twin_acquire_scanned_image", "args": {}}],
            "warnings": [{"code": "noise_only_file", "key": "noise.h5"}]}
    text = f"Error calling tool 'Twin_acquire_scanned_image': ToolError: {json.dumps(body)}\nThis call has failed 2 times in a row."
    shown = render("Twin_acquire_scanned_image", {}, text, error=True)
    assert "ToolError: Not acquired: the beam is blanked. Next call Twin_unblank_beam, then Twin_acquire_scanned_image again" in shown
    assert shown.endswith("(The noise-only file noise.h5 was saved; do not view or report it.)\nThis call has failed 2 times in a row.")
