"""Claim check: final answers checked against the run's tool calls (recorded-style messages, no model, no stack)."""

import asyncio
import json

from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from asyncroscopy.agents.claims import CLAIM_CHECK, ClaimCheckMiddleware, check_answer, corrections_used
from asyncroscopy.agents.guards import repeats
from asyncroscopy.agents.skill_middleware import _request

TOOLS = {"DigitalTwin_set_defocus", "DigitalTwin_get_defocus", "DigitalTwin_set_fov", "DigitalTwin_acquire_scanned_image",
         "DigitalTwin_acquire_spectrum", "DigitalTwin_move_stage", "DigitalTwin_get_parameters", "run_job", "view_data", "list_pois"}
IMAGES = ["stem_image_HAADF_20261029T094818640835.h5", "stem_image_HAADF_20261029T094826511245.h5", "stem_image_HAADF_20261029T094835162444.h5"]
SPECTRUM = "spectrum_eds_20261029T104411375142.h5"
MAP = "map_16ee61243764_map"


class Calls:
    """Builds a recorded message list: each call() adds the model's tool call and the server's reply."""

    def __init__(self, request: str):
        self.messages = [HumanMessage(request)]
        self.n = 0

    def call(self, name, args=None, reply="null", error=False):
        self.n += 1
        cid = f"c{self.n}"
        self.messages += [AIMessage("", tool_calls=[{"name": name, "args": args or {}, "id": cid}]),
                          ToolMessage(reply if isinstance(reply, str) else json.dumps(reply), tool_call_id=cid, name=name, status="error" if error else "success")]
        return self


def focus_series(last_defocus=2e-8):
    run = Calls("Take HAADF images at defocus -20 nm, 0 nm and +20 nm, then set the defocus back to 0.")
    for value, key in zip((-2e-8, 0.0, last_defocus), IMAGES):
        run.call("DigitalTwin_set_defocus", {"defocus": value})
        run.call("DigitalTwin_acquire_scanned_image", {"detector": "HAADF"}, {"key": key})
    return run.messages


FOCUS_ANSWER = (f"I took three HAADF images at defocus values of -20 nm, 0 nm, and +20 nm. The keys of the acquired images are: "
                f"`{IMAGES[0]}`, `{IMAGES[1]}`, and `{IMAGES[2]}`. The defocus was successfully set back to 0 nm.")


def survey(read=False):
    run = Calls("Make a 600 nm overview map around here, list the three most interesting regions, then go to the best one, "
                "take an EDS spectrum there and tell me what the sample contains.")
    run.call("run_job", {"device": "Mapper", "plan": "{}"}, {"job_id": "a1b2c3d4e5f6", "state": "done", "result_keys": {"run": "map_16ee61243764.h5", "map": MAP}})
    run.call("list_pois", {"map": MAP}, [{"x_m": 1e-7, "y_m": 2e-7, "score": 0.9}])
    run.call("DigitalTwin_move_stage", {"position": [1e-7, 2e-7, 0, 0, 0]})
    run.call("DigitalTwin_acquire_spectrum", {"exposure_time": 1.0}, {"key": SPECTRUM})
    if read:
        run.call("view_data", {"key": SPECTRUM, "format": "text"}, "spectrum: peaks Ca K-alpha 3690 eV, Ag L-alpha 2984 eV")
    return run.messages


def kinds(claims):
    return [c.kind for c in claims]


# ---------------------------------------------------------------------- the three benchmark failures
def test_focus_series_setting_claimed_back_to_zero_but_last_set_was_plus_20():
    claims = check_answer(focus_series(), FOCUS_ANSWER, TOOLS)
    assert kinds(claims) == ["setting"]
    assert "set back to 0 nm" in claims[0].text and "20 nm" in claims[0].why and "DigitalTwin_set_defocus" in claims[0].why


def test_three_regions_answer_lists_image_keys_no_tool_returned():
    run = Calls("Find the three most interesting regions on the map and image them at high magnification.")
    run.call("run_job", {"device": "Mapper"}, {"job_id": "0123456789ab", "state": "done", "result_keys": {"map": MAP, "images": [IMAGES[0]]}})
    invented = ["stem_image_HAADF_20261029T031724111111.h5", "stem_image_HAADF_20261029T031725222222.h5"]
    answer = f"I imaged the three most interesting regions on `{MAP}` at high magnification. The keys are: {IMAGES[0]}, {invented[0]}, {invented[1]}."
    claims = check_answer(run.messages, answer, TOOLS)
    assert kinds(claims) == ["key", "key"]
    assert [c.text for c in claims] == invented


def test_survey_leaves_viewing_the_spectrum_to_the_user_without_reading_it():
    answer = f"I made the map {MAP}, moved to the best region and took an EDS spectrum, saved as {SPECTRUM}. You can view the spectrum for further analysis."
    claims = check_answer(survey(), answer, TOOLS)
    assert kinds(claims) == ["deferred_read"]
    assert "view_data" in claims[0].why


# ---------------------------------------------------------------------- no false alarms
def test_correct_answers_are_not_flagged():
    assert check_answer(focus_series(last_defocus=0.0), FOCUS_ANSWER.replace("+20 nm", "0 nm"), TOOLS) == []
    answer = f"I set the defocus to -20 nm, took the images {IMAGES[0]} and {IMAGES[1]}, and the defocus is now 0 nm."
    assert check_answer(focus_series(last_defocus=0.0), answer, TOOLS) == []
    answer = f"I made {MAP}, moved to the best region and took an EDS spectrum ({SPECTRUM}). It shows calcium and silver."
    assert check_answer(survey(read=True), answer, TOOLS) == []
    # Read but deferred, or a request that does not ask what the data contain: not a deferred-read claim.
    assert check_answer(survey(read=True), f"Saved as {SPECTRUM}; you can view the spectrum.", TOOLS) == []


def test_a_value_read_back_counts_and_a_key_from_the_request_may_be_named():
    run = Calls(f"Set the defocus to 0 and look at {IMAGES[2]}.")
    run.call("DigitalTwin_set_defocus", {"defocus": 2e-8}).call("DigitalTwin_get_parameters", {}, {"defocus_m": 0.0, "fov_m": 2e-7, "fov_angstrom": 2000.0})
    assert check_answer(run.messages, f"The defocus is now 0 nm. {IMAGES[2]} is a HAADF image.", TOOLS) == []
    run.call("DigitalTwin_get_defocus", {}, "3e-08")
    claims = check_answer(run.messages, "The defocus is now 0 nm.", TOOLS)
    assert kinds(claims) == ["setting"] and "30 nm" in claims[0].why


def test_negated_hedged_and_uncheckable_statements_are_not_claims():
    run = Calls("Take an image.").call("DigitalTwin_acquire_scanned_image", {}, "Error: beam is blanked", error=True)
    answer = ("I could not take an image: the beam is blanked. No image was acquired. The stage was not moved. "
              "The previously acquired images are unchanged. The defocus was not set back to 0.")
    assert check_answer(run.messages, answer, TOOLS) == []
    # A unitless non-zero value cannot be compared exactly (its unit is unknown): not flagged.
    assert check_answer(focus_series(), "The defocus is now 5.", TOOLS) == []


def test_actions_with_no_call_of_their_kind_are_flagged():
    run = Calls("Go to the best region and take an image.").call("list_pois", {"map": MAP}, [])
    claims = check_answer(run.messages, "I moved to the best region and took an image there.", TOOLS)
    assert kinds(claims) == ["action", "action"]
    assert check_answer(run.messages, "I set the defocus to 0 nm.", TOOLS)[0].why == "no successful set_defocus call in this run"
    # A job moves and acquires: its answer is not flagged.
    run.call("run_job", {"device": "Mapper"}, {"job_id": "0123456789ab", "state": "done"})
    assert check_answer(run.messages, "I moved to the best region and took an image there.", TOOLS) == []


# ---------------------------------------------------------------------- the middleware
class Request:
    def __init__(self, messages, tools=(), system_message=None):
        self.messages, self.state, self.tools, self.system_message = messages, {"messages": messages}, list(tools), system_message

    def override(self, **changes):
        new = Request(self.messages, self.tools, self.system_message)
        new.state = self.state
        for key, value in changes.items():
            setattr(new, key, value)
        return new


class Model:
    """A scripted model: returns its answers in order and records what each call saw."""

    def __init__(self, *answers):
        self.answers, self.seen = list(answers), []

    def __call__(self, request):
        self.seen.append(list(request.messages))
        answer = self.answers.pop(0)
        return ModelResponse(result=[answer if isinstance(answer, AIMessage) else AIMessage(answer)])


def test_one_correction_turn_lists_exactly_what_is_unsupported():
    messages = focus_series()
    fixed = AIMessage("", tool_calls=[{"name": "DigitalTwin_set_defocus", "args": {"defocus": 0.0}, "id": "fix"}])
    model = Model(FOCUS_ANSWER, fixed)
    response = ClaimCheckMiddleware(TOOLS, enabled=True).wrap_model_call(Request(messages), model)
    first, correction, second = response.result
    assert first.content == FOCUS_ANSWER and second is fixed  # the model makes the missing call
    assert correction.type == "human" and correction.name == CLAIM_CHECK
    assert '"defocus was successfully set back to 0 nm"' in correction.content and "20 nm" in correction.content
    assert IMAGES[0] not in correction.content  # only the unsupported claim
    assert model.seen[1][-2:] == [first, correction]  # the model saw its answer and the correction


def test_a_corrected_answer_passes_without_a_note():
    model = Model(FOCUS_ANSWER, FOCUS_ANSWER.replace("successfully set back to 0 nm", "left at +20 nm"))
    response = ClaimCheckMiddleware(TOOLS, enabled=True).wrap_model_call(Request(focus_series()), model)
    assert len(response.result) == 3 and "[Claim check" not in response.result[-1].content


def test_after_one_retry_a_visible_note_names_what_could_not_be_verified():
    model = Model(FOCUS_ANSWER, FOCUS_ANSWER)
    response = ClaimCheckMiddleware(TOOLS, enabled=True).wrap_model_call(Request(focus_series()), model)
    final = response.result[-1]
    assert final.content.startswith(FOCUS_ANSWER)  # nothing removed
    assert "[Claim check: these statements could not be verified" in final.content and "set back to 0 nm" in final.content
    assert len(model.seen) == 2  # one retry, no more


def test_the_retry_cap_holds_across_tool_rounds():
    # The correction was already used earlier in this request: the next bad answer gets the note, not another turn.
    messages = [*focus_series(), AIMessage("x"), HumanMessage("check", name=CLAIM_CHECK)]
    assert corrections_used(messages) == 1
    model = Model(FOCUS_ANSWER)
    response = ClaimCheckMiddleware(TOOLS, enabled=True).wrap_model_call(Request(messages), model)
    assert len(model.seen) == 1 and len(response.result) == 1 and "[Claim check" in response.result[0].content
    # A new request from the user resets the cap.
    assert corrections_used([*messages, AIMessage("ok"), HumanMessage("next")]) == 0


def test_supported_answers_tool_calls_and_a_disabled_check_pass_through(monkeypatch):
    good = focus_series(last_defocus=0.0)
    model = Model(FOCUS_ANSWER)
    response = ClaimCheckMiddleware(TOOLS, enabled=True).wrap_model_call(Request(good), model)
    assert [m.content for m in response.result] == [FOCUS_ANSWER] and len(model.seen) == 1
    call = AIMessage("", tool_calls=[{"name": "view_data", "args": {"key": "x.h5"}, "id": "v"}])
    assert ClaimCheckMiddleware(TOOLS, enabled=True).wrap_model_call(Request(focus_series()), Model(call)).result == [call]
    monkeypatch.setenv("ASYNCROSCOPY_CLAIM_CHECK", "0")
    assert ClaimCheckMiddleware(TOOLS).wrap_model_call(Request(focus_series()), Model(FOCUS_ANSWER)).result[0].content == FOCUS_ANSWER
    monkeypatch.setenv("ASYNCROSCOPY_CLAIM_CHECK", "1")
    assert ClaimCheckMiddleware(TOOLS).enabled


def test_the_async_path_corrects_too():
    async def handler(request, model=Model(FOCUS_ANSWER, FOCUS_ANSWER)):
        return model(request)

    response = asyncio.run(ClaimCheckMiddleware(TOOLS, enabled=True).awrap_model_call(Request(focus_series()), handler))
    assert len(response.result) == 3 and "[Claim check" in response.result[-1].content


def test_keys_in_the_system_prompt_are_known():
    old = "map_aaaaaaaaaaaa_map"
    model = Model(f"The earlier map {old} covers this area.")
    response = ClaimCheckMiddleware(TOOLS, enabled=True).wrap_model_call(Request(focus_series(), system_message=AIMessage(f"notes: {old}")), model)
    assert len(response.result) == 1


def test_the_correction_is_not_taken_for_the_users_request():
    messages = [*focus_series(), AIMessage(FOCUS_ANSWER), HumanMessage("Check before you answer ...", name=CLAIM_CHECK)]
    assert _request(messages).startswith("Take HAADF images")
    # Loop detection still sees the rounds before the correction as this request's.
    looped = Calls("Read it.").call("view_data", {"key": "a.h5"}, "same").call("view_data", {"key": "a.h5"}, "same").messages
    assert repeats([*looped, HumanMessage("check", name=CLAIM_CHECK)]) == 1



def test_describing_existing_data_as_acquired_is_not_a_claim_of_acquiring():
    from asyncroscopy.agents.claims import ACQUIRED
    assert not ACQUIRED.search("The most recently acquired image shows bright particles on a dark support.")
    assert not ACQUIRED.search("The latest acquired spectrum has peaks at 2.98 and 3.69 keV.")
    assert ACQUIRED.search("I acquired a HAADF image at the best region.")
    assert ACQUIRED.search("I then took three images at different defocus values.")


def test_an_answer_that_announces_its_next_step_instead_of_taking_it_is_flagged():
    # Live run find_particle_then_eds (qwen3:8b): it viewed the image, then ended its turn announcing the spectrum.
    run = Calls("Find the brightest particle in a HAADF image and take an EDS spectrum on it.")
    run.call("DigitalTwin_acquire_scanned_image", {"detector": "HAADF"}, {"key": IMAGES[0]})
    run.call("view_data", {"key": IMAGES[0]}, "image")
    answer = ("The HAADF image shows a sparse distribution of bright particles. The brightest particle is likely the target. "
              "Let's locate it and take an EDS spectrum.\n\nI'll now find the brightest particle in the image and prepare to take an EDS spectrum on it.")
    claims = check_answer(run.messages, answer, TOOLS)
    assert [c.kind for c in claims] == ["unfinished"]

    # Not flagged: a question to the user, an offer, an earlier sentence describing the plan, or a finished report.
    for fine in ("I took the spectrum and it shows Au. Would you like me to map the area next?",
                 "Let me know if you want another image.",
                 "I'll now take an image. " + "It shows three particles. The brightest is at the centre.",
                 "The map is done; its key is in the reply above."):
        assert [c for c in check_answer(run.messages, fine, TOOLS) if c.kind == "unfinished"] == [], fine


def test_a_placement_claimed_when_place_image_refused_it_is_flagged():
    run = Calls("Zoom into a particle down to atomic columns, placing each image.")
    run.call("place_image", {"image": IMAGES[1], "parent": IMAGES[0]}, {"child": IMAGES[1], "status": "placed", "score": 0.99})
    run.call("place_image", {"image": IMAGES[2], "parent": IMAGES[1]}, {"child": IMAGES[2], "status": "unplaced", "reason": "another place matches almost as well"})
    claims = check_answer(run.messages, f"`{IMAGES[1]}` was placed on the survey and `{IMAGES[2]}` was successfully placed on it.")
    assert kinds(claims) == ["placement"] and IMAGES[2] in claims[0].why
    assert check_answer(run.messages, f"`{IMAGES[1]}` was placed; `{IMAGES[2]}` could not be placed: another place matches almost as well.") == []
    refused = Calls("Place the image.").call("place_image", {"image": IMAGES[2]}, {"child": IMAGES[2], "status": "unplaced", "reason": "x"})
    assert kinds(check_answer(refused.messages, "The image was placed on its parent.")) == ["placement"]
    assert check_answer(refused.messages, "The image is unplaced: x.") == []
