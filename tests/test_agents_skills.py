"""Skills found for the agent: from the request, then from what happens during the task (no model)."""

import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from asyncroscopy.agents.skill_middleware import MAX_APPLIED, SkillMiddleware, skills_for
from asyncroscopy.skills import list_skills, load_skill, select_skills, skills_text
from asyncroscopy.skills.registry import PROMPT_BUDGET_CHARS

STEM = {
    "run_job", "wait_for_job", "Mapper_abort", "approval_status", "load_skill", "view_data", "list_data", "score_map",
    "assemble_map", "map_info", "get_region", "list_pois", "add_poi", "Mapper_reserve",
}


@pytest.mark.parametrize("request_text, expected", [
    ("Map a 600 nm square around the current stage position.", "mapping"),
    ("Map a 600 nm square and take an EDS spectrum with each image.", "eds_survey"),
    ("Map a 600 nm square and find the 3 most interesting regions.", "explore_regions"),
    ("What is in the latest spectrum?", "inspect_data"),
    ("My map start was held for approval, what now?", "approval_etiquette"),
    ("The reserve keeps failing", "recover_from_errors"),
])
def test_the_request_picks_the_right_skill(request_text, expected):
    assert expected in [s.name for s in select_skills(request_text, STEM)]


def test_selection_respects_the_agents_tools_and_the_endpoint():
    indenter = {"IndentGrid_reserve", "run_job", "view_data", "list_data"}
    assert "mapping" not in [s.name for s in select_skills("map the sample", indenter)]
    analyst = {"view_data", "list_data", "score_map", "map_info", "get_region", "list_pois", "add_poi"}
    assert select_skills("find interesting regions", analyst) == []  # explore_regions needs a Mapper on the endpoint
    assert [s.name for s in select_skills("find interesting regions", analyst, endpoint_tools=STEM)][:1] == ["explore_regions"]
    assert select_skills("hello there", STEM) == []


def tool_reply(name, text, call_id, status="success"):
    return [AIMessage("", tool_calls=[{"name": name, "args": {}, "id": call_id}]), ToolMessage(text, tool_call_id=call_id, name=name, status=status)]


def applied(messages, tools=STEM):
    return [(a["skill"], a["reason"]) for a in skills_for(messages, tools, STEM)]


def test_a_failed_call_brings_recover_from_errors_once():
    start = [HumanMessage("acquire an image here")]
    failing = start + tool_reply("run_job", "Error: plan_rejected: fov_m out of range", "c1", status="error")
    twice = failing + tool_reply("run_job", "Error: plan_rejected: fov_m out of range", "c2", status="error")
    skills = applied(twice)
    assert [s for s, _ in skills].count("recover_from_errors") == 1
    assert ("recover_from_errors", "run_job failed") in skills
    assert "recover_from_errors" not in [s for s, _ in applied(start)]


def test_a_held_or_denied_call_brings_approval_etiquette():
    held = [HumanMessage("map a 600 nm square here")] + tool_reply("run_job", "APPROVAL REQUIRED: run_job is held (request_id=ab12)", "c1", status="error")
    skills = [s for s, _ in applied(held)]
    assert "approval_etiquette" in skills and "recover_from_errors" not in skills  # a hold is not a mistake to fix
    denied = [HumanMessage("map here")] + tool_reply("approval_status", json.dumps({"request_id": "ab12", "state": "denied"}), "c1")
    assert "approval_etiquette" in [s for s, _ in applied(denied)]


def test_data_arriving_brings_inspect_data_and_a_map_with_a_regions_request_brings_explore_regions():
    done = json.dumps({"job_id": "j1", "state": "done", "result_keys": {"run": "map_r1.h5"}})
    run = [HumanMessage("map a 600 nm square here and tell me where to look")] + tool_reply("wait_for_job", done, "c1")
    stitched = run + tool_reply("wait_for_job", json.dumps({"state": "done", "result_keys": {"map": "map_r1_map"}}), "c2")
    skills = dict(applied(stitched))
    assert "data key" in skills["inspect_data"] and "stitched map" in skills["explore_regions"]
    plain = [HumanMessage("map a 600 nm square here")] + tool_reply("wait_for_job", json.dumps({"result_keys": {"map": "map_r1_map"}}), "c1")
    assert "explore_regions" not in dict(applied(plain))  # no regions were asked for


def test_a_skill_the_agent_cannot_use_is_never_added():
    operator = {"run_job", "wait_for_job", "approval_status"}
    run = [HumanMessage("acquire it")] + tool_reply("wait_for_job", json.dumps({"result_keys": {"run": "map_r1.h5"}}), "c1")
    assert "inspect_data" not in dict(applied(run, operator))  # the operator has no view_data


def test_the_number_and_size_of_skills_are_capped():
    messages = [HumanMessage("map a 600 nm square with EDS and find interesting regions in the latest spectrum")]
    messages += tool_reply("run_job", "APPROVAL REQUIRED: held (request_id=x)", "c1", status="error")
    messages += tool_reply("run_job", "Error: busy", "c2", status="error")
    messages += tool_reply("wait_for_job", json.dumps({"result_keys": {"map": "m1_map", "run": "r.h5"}}), "c3")
    skills = skills_for(messages, STEM, STEM)
    assert len(skills) == MAX_APPLIED
    text, cut = skills_text([s["skill"] for s in skills], STEM)
    assert len(text) <= PROMPT_BUDGET_CHARS + 2000  # bodies within the budget, the rest listed by description
    assert all(f"=== skill: {name} (call load_skill" in text for name in cut)


class FakeRequest:
    def __init__(self, messages, tools, applied=None):
        self.state = {"messages": messages, **({"skills_applied": applied} if applied is not None else {})}
        self.messages = messages
        self.tools = tools

    def override(self, **changes):
        return SimpleNamespace(**changes)


def test_the_middleware_writes_skills_into_the_prompt_and_hides_load_skill_unless_needed():
    middleware = SkillMiddleware("BASE", STEM, STEM, memory=lambda request: "INSTRUMENT: twin" if "map" in request else "")
    messages = [HumanMessage("map a 600 nm square here")]
    update = middleware.before_model({"messages": messages}, None)
    assert update["skills_applied"][0] == {"skill": "mapping", "reason": "the request asks for a new map", "in_prompt": True}
    assert middleware.before_model({"messages": messages, **update}, None) is None  # nothing new: no state write
    tools = [SimpleNamespace(name="run_job"), SimpleNamespace(name="load_skill")]
    out = middleware._prompted(FakeRequest(messages, tools, update["skills_applied"]))
    prompt = out.system_message.content
    assert prompt.startswith("BASE") and "=== instructions (not a tool) ===\n# Mapping" in prompt and "INSTRUMENT: twin" in prompt
    assert [t.name for t in out.tools] == ["run_job", "load_skill"]  # an acting request may need a skill the rules did not pick
    reading = [HumanMessage("Show me the latest spectrum")]
    out = middleware._prompted(FakeRequest(reading, tools, middleware.before_model({"messages": reading}, None)["skills_applied"]))
    assert [t.name for t in out.tools] == ["run_job"]  # a read-only request with every skill fitted: no load_skill


def test_tools_that_write_notes_are_offered_only_when_the_request_asks_to_write_one():
    middleware = SkillMiddleware("BASE", STEM, STEM)
    tools = [SimpleNamespace(name=n) for n in ("list_pois", "annotate", "add_poi", "edit_annotation", "view_data")]
    for request, offered in (
        ("List the points of interest on the stitched map abc_map with their stage coordinates.", ["list_pois", "view_data"]),
        ("Mark the brightest particle on abc_map with a note.", ["list_pois", "annotate", "add_poi", "edit_annotation", "view_data"]),
        ("Add a point of interest at x = 10 nm on abc_map.", ["list_pois", "annotate", "add_poi", "edit_annotation", "view_data"]),
    ):
        messages = [HumanMessage(request)]
        out = middleware._prompted(FakeRequest(messages, tools, middleware.before_model({"messages": messages}, None)["skills_applied"]))
        assert [t.name for t in out.tools] == offered, request


def test_every_packaged_skill_loads_and_points_at_run_job_and_wait_for_job():
    skills = {s.name: s for s in list_skills(STEM)}
    for name in ("inspect_data", "explore_regions", "approval_etiquette", "recover_from_errors", "eds_survey", "mapping"):
        assert skills[name].available, skills[name].problem
        assert load_skill(name, STEM).startswith("#")
    for name in ("eds_survey", "explore_regions"):
        body = load_skill(name, STEM)
        assert "run_job" in body and "wait_for_job" in body and "Mapper_reserve" not in body


# Requests that only ask about data never get a skill that acts (the review's examples and the
# data-only workflows): a map key coming back must not lead a small model to start a new map.
READ_ONLY = [
    "What features are visible on the map abc123_map?",
    "Which regions of map abc_map look different? Do not acquire anything.",
    "Is there anything interesting in the latest image?",
    "Show me the latest spectrum",
    "What was the most recently acquired image? Describe it briefly (size, field of view and intensity range).",
    "How many image tiles are in the map run map_r1.h5?",
    "Look at the EDS spectrum spectrum_eds_1.h5. Which elements does it show?",
    "List the points of interest on the stitched map map_r1_map with their stage coordinates.",
    "The stitched map map_r1_map covers a large area. Show me an overview of the whole map, then zoom into its "
    "highest-scoring point of interest and show that area at full resolution.",
]


@pytest.mark.parametrize("request_text", READ_ONLY)
def test_a_read_only_request_never_gets_a_skill_that_acts(request_text):
    acting = {s.name for s in list_skills(STEM) if s.acts}
    assert acting >= {"mapping", "eds_survey", "explore_regions"}
    messages = [HumanMessage(request_text), *tool_reply("view_data", "map map_r9_map, regions, spectrum_eds_2.h5", "c1")]
    assert not {name for name, _ in applied(messages)} & acting
    assert not {s.name for s in select_skills(request_text, STEM)} & acting


def test_acting_requests_still_get_acting_skills():
    assert "mapping" in [s.name for s in select_skills("Map a 300 nm square around the current stage position.", STEM)]
    assert "explore_regions" in [s.name for s in select_skills("On map_r1_map, find the three most interesting regions and image them at high magnification.", STEM)]


def test_memory_notes_are_read_once_per_run_and_share_the_skill_budget():
    reads = []

    def memory(request):
        reads.append(request)
        return "N" * 3000

    middleware = SkillMiddleware("BASE", STEM, STEM, memory=memory)
    messages = [HumanMessage("map a 600 nm square here")]
    update = middleware.before_model({"messages": messages}, None)
    for _ in range(3):
        middleware._prompted(FakeRequest(messages, [SimpleNamespace(name="run_job")], update["skills_applied"]))
    assert len(reads) == 1
    prompt = middleware._prompted(FakeRequest(messages, [], update["skills_applied"])).system_message.content
    assert len(prompt) <= len("BASE") + PROMPT_BUDGET_CHARS + 400  # skills and notes together stay inside the budget


def test_every_shipped_skill_passes_the_skill_linter():
    from asyncroscopy.agents.learning import lint_skill

    for skill in list_skills(STEM):
        if not skill.learned:
            assert lint_skill(skill.path) == [], skill.name


# Acting unless clearly read-only: wrongly treating these as read-only would keep the skills they need away.
ACTING_PHRASINGS = [
    "Show me a map of a new area",
    "show me a new image",
    "Can you get an image at higher magnification of region 2?",
    "Measure an EDS spectrum here and tell me the elements",
    "Do a 1 um map please",
    "Perform a survey of the sample?",
    "Can you check whether there is copper by taking a spectrum?",
    "Tell me what is at x=1um by imaging it",
    "Is the beam blanked? If not, blank it.",
    "What does the sample look like?",  # no reference to stored data: it may need a new image
]


@pytest.mark.parametrize("request_text", ACTING_PHRASINGS)
def test_requests_act_unless_clearly_read_only(request_text):
    from asyncroscopy.skills.registry import read_only_request

    assert not read_only_request(request_text)


@pytest.mark.parametrize("request_text", READ_ONLY)
def test_questions_about_existing_data_and_do_not_acquire_stay_read_only(request_text):
    from asyncroscopy.skills.registry import read_only_request

    assert read_only_request(request_text)


def test_each_turn_is_judged_by_its_own_request():
    first = [HumanMessage("Show me the latest spectrum"), AIMessage("It shows Cu.")]
    assert not {name for name, _ in applied(first)} & {"mapping", "eds_survey", "explore_regions"}
    later = first + [HumanMessage("Now map a 600 nm square around here.")]
    assert "mapping" in {name for name, _ in applied(later)}


def test_the_benchmark_counts_only_skills_whose_text_reached_the_prompt():
    from benchmarks.run_workflows import skills_of

    assert skills_of([{"skill": "mapping", "in_prompt": True}, {"skill": "eds_survey", "in_prompt": False}, {"skill": "inspect_data"}]) == ["mapping", "inspect_data"]


MICROSCOPE = STEM | {"DigitalTwin_acquire_scanned_image", "DigitalTwin_acquire_spectrum", "DigitalTwin_place_beam",
                     "DigitalTwin_move_stage", "DigitalTwin_get_stage", "DigitalTwin_set_fov"}


@pytest.mark.parametrize("request_text", [
    "Survey a 2 um area and give me an overview of the sample.",
    "Make a quick map of the area around the stage, then find the 3 most interesting regions.",
    "Acquire a 3x3 tiled map at 400 nm field of view.",
])
def test_a_request_for_a_new_map_always_gets_the_mapping_procedure_first(request_text):
    assert [a["skill"] for a in skills_for([HumanMessage(request_text)], MICROSCOPE, MICROSCOPE)][:1] == ["mapping"]


def test_a_map_without_eds_does_not_get_the_eds_survey():
    names = [a["skill"] for a in skills_for([HumanMessage("Map a 600 nm square without EDS spectra.")], MICROSCOPE, MICROSCOPE)]
    assert "mapping" in names and "eds_survey" not in names


def test_finding_a_feature_then_measuring_it_gets_its_own_skill():
    names = [s.name for s in select_skills("Find the brightest particle in an image and take a spectrum on it.", MICROSCOPE)]
    assert names[:1] == ["find_feature_then_measure"]


def test_a_skill_acts_unless_it_says_it_only_reads(tmp_path):
    from asyncroscopy.skills.registry import _parse

    for extra, acts in (("", True), ("acts: false\n", False), ("acts: true\n", True)):
        path = tmp_path / "SKILL.md"
        path.write_text(f"---\nname: grid\ndescription: Indent a grid.\nversion: 1.0.0\ntools: [view_data, '*_get_*']\n{extra}---\n\n# Grid\n", encoding="utf-8")
        assert _parse(path).acts is acts, extra
