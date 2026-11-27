"""Agent profiles: the model-specific settings live in agents/profiles/*.yaml, read in one place."""

import pytest

from asyncroscopy.agents.config import AgentConfig
from asyncroscopy.agents.profile import MIDDLEWARE, load_profile


def test_the_small_model_profile_keeps_its_model_and_middleware():
    profile = load_profile("default")
    assert profile.model == "qwen3:8b" and profile.num_ctx == 8192 and profile.middleware == MIDDLEWARE


def test_without_a_choice_agents_use_the_capable_profile_on_gemma4(monkeypatch):
    assert load_profile().name == "capable"
    config = AgentConfig()
    assert (config.profile, config.model, config.model_provider, config.num_ctx) == ("capable", "gemma4:31b", "ollama", 32768)
    assert config.middleware == list(MIDDLEWARE)
    assert AgentConfig(model="other:1b").model == "other:1b"
    monkeypatch.delenv("ASYNCROSCOPY_AGENT_PROFILE", raising=False)
    monkeypatch.delenv("ASYNCROSCOPY_MODEL", raising=False)
    monkeypatch.delenv("ASYNCROSCOPY_NUM_CTX", raising=False)
    assert (AgentConfig.from_env().profile, AgentConfig.from_env().model) == ("capable", "gemma4:31b")


def test_from_env_picks_a_profile_and_a_model(monkeypatch, tmp_path):
    lean = tmp_path / "lean.yaml"
    lean.write_text("model: tiny:1b\nnum_ctx: 4096\nmiddleware: [job_wait]\ntools: {stem: ['*_move_stage']}\n", encoding="utf-8")
    monkeypatch.setenv("ASYNCROSCOPY_AGENT_PROFILE", str(lean))
    monkeypatch.delenv("ASYNCROSCOPY_MODEL", raising=False)
    config = AgentConfig.from_env()
    assert (config.model, config.num_ctx, config.middleware) == ("tiny:1b", 4096, ["job_wait"])
    assert config.tool_lists().stem == ("*_move_stage",)
    monkeypatch.setenv("ASYNCROSCOPY_MODEL", "chosen:7b")
    assert AgentConfig.from_env().model == "chosen:7b"


def test_a_profile_with_unknown_middleware_is_refused(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("model: x\nmiddleware: [skils]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown middleware"):
        load_profile(str(bad))


def test_the_middleware_is_built_in_the_profiles_order():
    from asyncroscopy.agents.middleware import build_middleware

    made = build_middleware(["job_wait", "empty_reply"], prompt="p", tool_names={"run_job"}, offered={"run_job"}, tools=[])
    assert [type(m).__name__ for m in made] == ["JobWaitMiddleware", "EmptyReplyMiddleware"]
    with pytest.raises(ValueError):
        build_middleware(["nope"], prompt="p", tool_names=set(), offered=set(), tools=[])


# What depends on the model is data in the profile. A capable model's profile turns off the aids for a small model.
def test_the_capable_profile_extends_the_default_and_turns_the_small_model_aids_off():
    default, capable = load_profile("default"), load_profile("capable")
    assert default.tool_gates and default.replies == "notes" and default.skills.from_request == 2
    assert capable.tool_gates == () and capable.replies == "facts_and_notes" and capable.skills.prompt_chars > default.skills.prompt_chars
    # What it does not change it takes from the default. That is the tool selection, the team split and the middleware.
    assert capable.tools == default.tools and capable.team == default.team and capable.middleware == default.middleware
    assert capable.num_ctx > default.num_ctx


def test_a_profile_extends_another_and_changes_only_what_it_names(tmp_path):
    (tmp_path / "base.yaml").write_text("model: a:1b\nnum_ctx: 4096\nskills: {from_request: 1, at_most: 2}\nreplies: notes\n", encoding="utf-8")
    child = tmp_path / "child.yaml"
    child.write_text("extends: base.yaml\nmodel: b:2b\nskills: {at_most: 5}\n", encoding="utf-8")
    profile = load_profile(str(child))
    assert (profile.model, profile.num_ctx, profile.skills.from_request, profile.skills.at_most) == ("b:2b", 4096, 1, 5)
    loop = tmp_path / "loop.yaml"
    loop.write_text("extends: loop.yaml\nmodel: c:1b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="circle"):
        load_profile(str(loop))


@pytest.mark.parametrize("text, refused", [
    ("skills: {from_request: 0}", "whole numbers above 0"),
    ("skills: {matched: 2}", "whole numbers above 0"),
    ("tool_gates: [{tools: [annotate]}]", "offer_if or hide_if"),
    ("tool_gates: [{tools: [annotate], offer_if: ['re:(']}]", "not a valid pattern"),
    ("replies: prose", "replies is one of"),
])
def test_a_profile_with_a_bad_setting_is_refused(tmp_path, text, refused):
    bad = tmp_path / "bad.yaml"
    bad.write_text(f"model: a:1b\n{text}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=refused):
        load_profile(str(bad))


@pytest.mark.parametrize("request_text, held", [
    # The note writers wait for a request to write. A cue matches whole words and word starts in any case.
    ("Describe the stitched map abc_map.", {"annotate", "add_poi"}),
    ("Label the grain boundary on abc_map.", set()),
    ("ANNOTATION please", set()),
    ("Image the landmark again", {"annotate", "add_poi", "Scope_place_beam", "Scope_place_beam_at"}),  # "mark*" starts a word, so not inside "landmark"
    # The beam parkers wait for a spectrum. A phrase cue needs its words in order.
    ("Place the beam on the particle", {"annotate", "add_poi"}),
    ("Take an X-ray spectrum", {"annotate", "add_poi"}),
    ("Take a closer image of the particle.", {"annotate", "add_poi", "Scope_place_beam", "Scope_place_beam_at"}),
])
def test_the_default_profiles_tool_gates_read_the_request(request_text, held):
    from asyncroscopy.agents.profile import held_back

    tools = ["annotate", "add_poi", "Scope_place_beam", "Scope_place_beam_at", "view_data"]
    beam = {"Scope_place_beam", "Scope_place_beam_at"}
    found = held_back(load_profile("default").tool_gates, request_text, tools)
    assert found - beam == held - beam
    if beam & held:
        assert beam <= found
    assert held_back(load_profile("capable").tool_gates, request_text, tools) == set()


def test_a_stage_position_in_the_request_holds_back_the_pixel_conversions():
    from asyncroscopy.agents.profile import held_back

    gates = load_profile("default").tool_gates
    assert held_back(gates, "Go to x = 1.5e-7 m and image", ["to_stage", "to_pixel"]) == {"to_stage", "to_pixel"}
    assert held_back(gates, "Where is the particle in a.h5 on the stage?", ["to_stage", "to_pixel"]) == set()


def test_structured_replies_are_replaced_by_notes_or_kept_beside_them():
    import json

    from asyncroscopy.agents.guidance import render

    reply = json.dumps({"state": "running", "job_id": "j1", "next_actions": [{"tool": "wait_for_job", "args": {"job_id": "j1"}}]})
    small = json.loads(render("wait_for_job", {"job_id": "j1"}, reply))
    capable = json.loads(render("wait_for_job", {"job_id": "j1"}, reply, replies="facts_and_notes"))
    assert "next_actions" not in small and "wait_for_job again" in small["note"]
    assert capable["next_actions"][0]["tool"] == "wait_for_job" and capable["note"] == small["note"]
    blanked = 'Error: {"error": {"code": "beam_blanked"}, "next_actions": [{"tool": "S_unblank_beam"}, {"tool": "S_acquire"}]}'
    assert render("S_acquire", {}, blanked, error=True).startswith("Error: Not acquired: the beam is blanked")
    kept = render("S_acquire", {}, blanked, error=True, replies="facts_and_notes")
    assert kept.startswith(blanked) and "Next call S_unblank_beam" in kept
