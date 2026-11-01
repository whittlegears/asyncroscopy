"""Skills that run themselves: run_skill sends a skill's listed calls in order in the background."""

import asyncio
import json
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool

from asyncroscopy.agents.goals import trace_from_messages
from asyncroscopy.agents.skill_middleware import SkillMiddleware
from asyncroscopy.agents.skill_runs import SkillRuns
from asyncroscopy.skills import list_skills

TWIN = {
    "DigitalTwin_set_fov", "DigitalTwin_unblank_beam", "DigitalTwin_blank_beam", "DigitalTwin_acquire_scanned_image",
    "DigitalTwin_place_beam_at", "DigitalTwin_acquire_spectrum", "DigitalTwin_get_fov", "DigitalTwin_move_stage",
    "DigitalTwin_get_stage", "find_features", "view_data", "run_job", "wait_for_job",
}


def instrument(features, more=None, names=TWIN):
    """Stand-ins for the twin's tools that record every call; find_features finds ``features``; ``more`` adds or replaces replies."""
    sent = []
    replies = {
        "DigitalTwin_acquire_scanned_image": lambda a: "survey.h5",
        "find_features": lambda a: json.dumps({"features": features, "hint": "nothing distinct"} if not features else {"features": features}),
        "DigitalTwin_acquire_spectrum": lambda a: "eds.h5",
        "view_data": lambda a: "peaks: Au M 2.12 keV\ndata (sums of 11 channels):\n1,2",
        "DigitalTwin_place_beam_at": lambda a: json.dumps([0.6, 0.4]),
        "DigitalTwin_get_stage": lambda a: json.dumps([0.0, 0.0, 1e-9, 0.1, 0.0]),
        **(more or {}),
    }

    def tool(name):
        async def run(**kwargs):
            sent.append((name, kwargs))
            return replies.get(name, lambda a: json.dumps({"result": None}))(kwargs)

        return StructuredTool(name=name, description=name, args_schema={"type": "object", "properties": {}}, coroutine=run)

    return [tool(n) for n in sorted(names)], sent


def run(features, inputs=None):
    async def go():
        tools, sent = instrument(features)
        runs = SkillRuns(tools)
        started = json.loads(await runs.start("find_feature_then_measure", inputs))
        assert started["state"] == "running" and "wait_for_skill" in started["next"]
        return json.loads(await runs.wait(started["run_id"])), sent

    return asyncio.run(go())


def test_a_run_sends_the_skills_calls_in_order_with_each_reply_filled_into_the_next():
    reply, sent = run([{"x_m": 1.5e-8, "y_m": -2e-8}], {"fov_m": "4e-7"})
    assert reply["state"] == "done"
    assert [name for name, _ in sent] == [
        "DigitalTwin_set_fov", "DigitalTwin_unblank_beam", "DigitalTwin_acquire_scanned_image", "DigitalTwin_blank_beam", "find_features",
        "DigitalTwin_unblank_beam", "DigitalTwin_place_beam_at", "DigitalTwin_acquire_spectrum", "DigitalTwin_blank_beam", "view_data"]
    assert sent[0][1] == {"fov": 4e-7} and sent[4][1] == {"image": "survey.h5", "n": 1}
    assert sent[6][1] == {"position": [1.5e-8, -2e-8]} and sent[9][1] == {"key": "eds.h5", "format": "text"}
    assert reply["outputs"]["spectrum"] == "eds.h5" and reply["outputs"]["peaks"] == "peaks: Au M 2.12 keV"  # without the channel table
    assert list(reply)[:4] == ["run_id", "skill", "state", "next"] and "elements most consistent" in reply["next"]


def test_a_run_that_finds_nothing_stops_before_the_spectrum_and_still_blanks_the_beam():
    reply, sent = run([])
    assert reply["state"] == "stopped" and "the feature step's reply has no features.0.x_m" in reply["stopped"] and "nothing distinct" in reply["stopped"]
    assert [name for name, _ in sent][-2:] == ["DigitalTwin_unblank_beam", "DigitalTwin_blank_beam"]
    assert "DigitalTwin_acquire_spectrum" not in [name for name, _ in sent]


def test_a_run_refuses_unknown_inputs_and_a_second_run_while_one_is_going():
    async def go():
        tools, _sent = instrument([{"x_m": 0.0, "y_m": 0.0}])
        runs = SkillRuns(tools)
        refused = json.loads(await runs.start("find_feature_then_measure", {"size": 3}))
        assert "takes inputs ['fov_m'], not ['size']" in refused["error"]
        assert "mapping is instructions to follow with its tools" in json.loads(await runs.start("mapping"))["error"]
        assert "view_data is a tool: call it directly" in json.loads(await runs.start("view_data"))["error"]
        first = json.loads(await runs.start("find_feature_then_measure"))
        second = json.loads(await runs.start("find_feature_then_measure"))
        assert first["run_id"] in second["error"]
        await runs.wait(first["run_id"])

    asyncio.run(go())


def test_a_runs_calls_count_as_the_agents_own_in_the_trace():
    reply, _sent = run([{"x_m": 1e-8, "y_m": 1e-8}])
    messages = [
        HumanMessage("find a particle and take a spectrum on it"),
        AIMessage("", tool_calls=[{"id": "w", "name": "wait_for_skill", "args": {"run_id": reply["run_id"]}}]),
        ToolMessage(json.dumps(reply), tool_call_id="w"),
    ]
    names = [c.name for c in trace_from_messages(messages)]
    assert names[0] == "wait_for_skill" and "DigitalTwin_acquire_spectrum" in names and len(names) == 11


def test_run_skill_is_offered_only_with_a_skill_that_runs_by_itself():
    middleware = SkillMiddleware("BASE", TWIN | {"run_skill", "wait_for_skill"}, TWIN | {"run_skill", "wait_for_skill"})
    tools = [SimpleNamespace(name=n) for n in ("run_skill", "wait_for_skill", "view_data")]
    for request, offered in (
        ("Find a particle near here and take an EDS spectrum on it.", ["run_skill", "wait_for_skill", "view_data"]),
        ("Map a 600 nm square around the current stage position.", ["view_data"]),
    ):
        messages = [HumanMessage(request)]
        applied = middleware.before_model({"messages": messages}, None)["skills_applied"]
        request_ = SimpleNamespace(state={"messages": messages, "skills_applied": applied}, messages=messages, tools=tools,
                                   override=lambda **changes: SimpleNamespace(**changes))
        assert [t.name for t in middleware._prompted(request_).tools] == offered, request


def test_the_packaged_run_list_names_only_tools_a_stem_twin_offers():
    skill = next(s for s in list_skills(TWIN) if s.name == "find_feature_then_measure")
    assert skill.available and skill.run["inputs"] == {"fov_m": 3e-7}
    assert SkillRuns([SimpleNamespace(name=n) for n in TWIN]).offered()


def test_a_skill_is_not_runnable_when_a_step_names_a_tool_that_is_not_offered():
    without_analysis = TWIN - {"find_features"}
    assert sorted(SkillRuns([SimpleNamespace(name=n) for n in without_analysis]).skills) == ["spectrum_at_point"]


def test_a_spectrum_at_a_point_needs_the_point_and_moves_there_keeping_z_and_tilts():
    async def go():
        tools, sent = instrument([])
        runs = SkillRuns(tools)
        wrong = json.loads(await runs.start("spectrum_at_point", {"map": "m_map", "top_k": 1}))
        assert "not ['map', 'top_k']" in wrong["error"] and '"inputs": {"x_m": "<number>", "y_m": "<number>"}' in wrong["error"]
        refused = json.loads(await runs.start("spectrum_at_point"))
        assert "needs inputs ['x_m', 'y_m']" in refused["error"] and '"inputs": {"x_m": "<number>"' in refused["error"]
        started = json.loads(await runs.start("spectrum_at_point", {"x_m": "-5.4e-8", "y_m": -2e-7}))
        return json.loads(await runs.wait(started["run_id"])), sent

    reply, sent = asyncio.run(go())
    assert reply["state"] == "done" and reply["outputs"]["peaks"] == "peaks: Au M 2.12 keV"
    assert ("DigitalTwin_move_stage", {"position": [-5.4e-8, -2e-7, 1e-9, 0.1, 0.0]}) in sent
    assert ("DigitalTwin_place_beam_at", {"position": [-5.4e-8, -2e-7]}) in sent


def test_run_skill_is_offered_with_a_skill_that_sends_the_agent_to_a_run():
    tools = TWIN | {"run_skill", "wait_for_skill", "score_map", "Mapper_reserve", "list_pois"}
    middleware = SkillMiddleware("BASE", tools, tools)
    messages = [HumanMessage("Make a 600 nm map here, find the three most interesting regions, then take an EDS spectrum at the best one.")]
    applied = middleware.before_model({"messages": messages}, None)["skills_applied"]
    assert "explore_regions" in [a["skill"] for a in applied]
    request = SimpleNamespace(state={"messages": messages, "skills_applied": applied}, messages=messages,
                              tools=[SimpleNamespace(name="run_skill"), SimpleNamespace(name="wait_for_skill")],
                              override=lambda **changes: SimpleNamespace(**changes))
    assert [t.name for t in middleware._prompted(request).tools] == ["run_skill", "wait_for_skill"]


def test_a_skill_only_other_skills_send_the_agent_to_is_never_chosen_by_the_requests_words():
    from asyncroscopy.skills import select_skills

    assert "spectrum_at_point" not in [s.name for s in select_skills("Take an EDS spectrum at the point x = 10 nm, y = 5 nm.", TWIN)]


def test_tools_that_park_the_beam_are_offered_only_when_the_request_is_about_a_spectrum():
    middleware = SkillMiddleware("BASE", TWIN, TWIN)
    tools = [SimpleNamespace(name=n) for n in ("DigitalTwin_place_beam", "DigitalTwin_place_beam_at", "DigitalTwin_acquire_scanned_image")]
    for request, offered in (
        ("Map a 600 nm square, find the most interesting region, then take a higher-magnification image of it.", ["DigitalTwin_acquire_scanned_image"]),
        ("Take an EDS spectrum on the brightest particle.", ["DigitalTwin_place_beam", "DigitalTwin_place_beam_at", "DigitalTwin_acquire_scanned_image"]),
    ):
        messages = [HumanMessage(request)]
        applied = middleware.before_model({"messages": messages}, None)["skills_applied"]
        request_ = SimpleNamespace(state={"messages": messages, "skills_applied": applied}, messages=messages, tools=tools,
                                   override=lambda **changes: SimpleNamespace(**changes))
        assert [t.name for t in middleware._prompted(request_).tools] == offered, request


def test_tools_no_chosen_skill_uses_are_dropped_when_the_prompt_would_not_leave_room_for_the_conversation():
    tools = TWIN | {"run_skill", "wait_for_skill"}
    middleware = SkillMiddleware("BASE", tools, tools, num_ctx=4096)
    offered = [StructuredTool(name=n, description="x" * 900, args_schema={"type": "object", "properties": {}}, coroutine=None)
               for n in sorted(TWIN - {"find_features"}) + ["run_skill", "wait_for_skill", "list_annotations", "list_devices"]]
    messages = [HumanMessage("Find a particle near here and take an EDS spectrum on it.")]
    applied = middleware.before_model({"messages": messages}, None)["skills_applied"]
    request = SimpleNamespace(state={"messages": messages, "skills_applied": applied}, messages=messages, tools=offered,
                              override=lambda **changes: SimpleNamespace(**changes))
    kept = [t.name for t in middleware._prompted(request).tools]
    assert "list_annotations" not in kept and "list_devices" not in kept and "run_job" not in kept
    assert {"run_skill", "wait_for_skill", "view_data"} <= set(kept)


def test_pixel_conversions_are_not_offered_when_the_request_gives_a_stage_position():
    middleware = SkillMiddleware("BASE", TWIN, TWIN)
    tools = [SimpleNamespace(name=n) for n in ("to_stage", "to_pixel", "run_job")]
    for request, offered in (
        ("The map m_map has a point of interest at x = -325.4 nm, y = 389.0 nm. Go back to it and take a HAADF image there.", ["run_job"]),
        ("Find the brightest particle in image a.h5 and give its stage position.", ["to_stage", "to_pixel", "run_job"]),
    ):
        messages = [HumanMessage(request)]
        applied = middleware.before_model({"messages": messages}, None)["skills_applied"]
        request_ = SimpleNamespace(state={"messages": messages, "skills_applied": applied}, messages=messages, tools=tools,
                                   override=lambda **changes: SimpleNamespace(**changes))
        assert [t.name for t in middleware._prompted(request_).tools] == offered, request


def best_region(regions, inputs):
    async def go():
        pois = {"list_pois": lambda a: json.dumps({"total": len(regions), "pois": regions})}
        tools, sent = instrument([{"x_m": 1.2e-7, "y_m": -1.9e-7}], pois, TWIN | {"Mapper_reserve", "list_pois"})
        runs = SkillRuns(tools)
        started = json.loads(await runs.start("best_region_spectrum", inputs))
        return json.loads(await runs.wait(started["run_id"])), sent

    return asyncio.run(go())


def test_a_spectrum_at_the_best_region_moves_to_the_maps_first_region_and_measures_on_the_feature_there():
    reply, sent = best_region([{"id": "a", "x_m": 1e-7, "y_m": -2e-7, "score": 0.9}, {"id": "b", "x_m": 0.0, "y_m": 0.0, "score": 0.5}],
                              {"map": "m_map", "top_k": "5"})
    assert reply["state"] == "done" and reply["outputs"]["peaks"] == "peaks: Au M 2.12 keV"
    assert sent[0] == ("list_pois", {"map": "m_map", "kind": "candidate", "limit": 5})
    assert ("DigitalTwin_move_stage", {"position": [1e-7, -2e-7, 1e-9, 0.1, 0.0]}) in sent
    assert ("DigitalTwin_set_fov", {"fov": 3e-7}) in sent  # the nested skill's own default
    assert ("DigitalTwin_place_beam_at", {"position": [1.2e-7, -1.9e-7]}) in sent  # on the feature found at the region
    assert reply["outputs"]["regions"]["pois"][0]["id"] == "a" and "regions are regions pois" in reply["next"]


def test_a_map_without_regions_stops_the_run_before_the_stage_moves():
    reply, sent = best_region([], {"map": "m_map"})
    assert reply["state"] == "stopped" and "the regions step's reply has no pois.0.x_m" in reply["stopped"]
    assert [name for name, _ in sent if name.startswith("DigitalTwin_")] == ["DigitalTwin_get_stage"]

def test_a_skill_that_runs_another_is_runnable_only_when_that_one_is():
    with_mapper = TWIN | {"Mapper_reserve", "list_pois"}
    assert "best_region_spectrum" in SkillRuns([SimpleNamespace(name=n) for n in with_mapper]).skills
    assert "best_region_spectrum" not in SkillRuns([SimpleNamespace(name=n) for n in with_mapper - {"find_features"}]).skills


def test_an_acquisitions_summary_reaches_the_agent_and_its_key_alone_goes_into_the_next_step():
    from mcp.types import TextContent

    from asyncroscopy.agents.mcp_tools import result_text

    summary = "taken with the beam blanked (beam_current_pa 0): it shows only detector noise, not the sample; unblank_beam and take it again"
    reply = SimpleNamespace(structured_content={"result": "survey.h5"},
                            content=[TextContent(type="text", text="survey.h5"), TextContent(type="text", text=summary)])
    assert result_text(reply) == f"survey.h5\n{summary}"
    assert result_text(SimpleNamespace(structured_content={"result": "survey.h5"}, content=[TextContent(type="text", text="survey.h5")])) == "survey.h5"

    async def go():
        tools, sent = instrument([{"x_m": 1e-8, "y_m": 0.0}], {"DigitalTwin_acquire_scanned_image": lambda a: f"survey.h5\n{summary}",
                                                               "DigitalTwin_acquire_spectrum": lambda a: "eds.h5\npeaks: Au"})
        runs = SkillRuns(tools)
        started = json.loads(await runs.start("find_feature_then_measure"))
        return json.loads(await runs.wait(started["run_id"])), sent

    done, sent = asyncio.run(go())
    assert done["state"] == "done"
    assert dict(sent)["find_features"] == {"image": "survey.h5", "n": 1} and dict(sent)["view_data"]["key"] == "eds.h5"
