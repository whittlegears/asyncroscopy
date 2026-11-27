"""Skills as passages retrieved for each step (asyncroscopy.skills.index), no model."""

import time
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from asyncroscopy.agents.skill_middleware import SkillMiddleware, step_query
from asyncroscopy.skills import list_skills
from asyncroscopy.skills.index import MAX_PASSAGE_CHARS, passages_of, passages_text, rank_texts, search, split_manual

STEM = {
    "run_job", "wait_for_job", "Mapper_abort", "load_skill", "view_data", "list_data", "score_map",
    "assemble_map", "map_info", "get_region", "list_pois", "add_poi", "Mapper_reserve", "place_beam", "move_stage",
}


def usable():
    return [s for s in list_skills(STEM) if s.available]


def test_a_manual_splits_into_sections_and_a_long_section_into_its_items():
    items = "\n".join(f"{i}. " + "word " * 60 for i in range(1, 9))
    body = f"# Doing a thing\n\nWhy it matters.\n\n## 1. Plan\n\nPick a size.\n\n## 2. Steps\n\n{items}\n"
    passages = split_manual("thing", body)
    assert [p.heading for p in passages][:2] == ["", "1. Plan"]
    assert all(p.title == "Doing a thing" for p in passages)
    steps = [p for p in passages if p.heading == "2. Steps"]
    assert len(steps) > 1 and all(len(p.text) <= MAX_PASSAGE_CHARS for p in steps)
    assert steps[0].text.startswith("1. ") and "8. " in steps[-1].text  # every item kept, in order


def test_every_shipped_skill_splits_into_passages():
    for skill in list_skills():
        assert passages_of(skill), skill.name


def test_an_error_finds_the_advice_for_that_error():
    found = search([("look closer at the particle", 1.0), ("Error: coordinates must be in 0..1 range", 1.0)], usable(), k=2)
    assert found and found[0][0].skill == "recover_from_errors"
    assert "0..1 range" in found[0][0].text


def test_small_talk_and_negated_words_find_nothing():
    assert search([("hello there, how are you", 1.0)], usable()) == []
    assert all(p.skill != "eds_survey" for p, _ in search([("map a 600 nm square without EDS spectra", 1.0)], usable(), k=5))


def test_a_boost_brings_a_skill_only_with_words_in_common():
    boosted = search([("Error: stage refused", 1.0)], usable(), k=3, boost={"recover_from_errors": 3.0})
    assert boosted[0][0].skill == "recover_from_errors"
    assert search([("hello", 1.0)], usable(), boost={"recover_from_errors": 3.0}) == []


def test_passages_fit_the_budget_and_keep_each_skills_order():
    found = [p for p, _ in search([("take spectra at points of the image", 1.0)], usable(), k=3)]
    text, fitted = passages_text(found, budget=100_000)
    assert [p.order for p in fitted if p.skill == fitted[0].skill] == sorted(p.order for p in fitted if p.skill == fitted[0].skill)
    assert all(f"--- {p.label} ---" in text for p in fitted)
    small, kept = passages_text(found, budget=300)
    assert len(small) <= 300 + 50 * len(kept)


def test_retrieval_is_fast():
    start = time.perf_counter()
    for _ in range(20):
        search([("map a square and take spectra", 1.0), ("Error: busy", 0.6)], usable())
    assert (time.perf_counter() - start) / 20 < 0.05  # well under a model call; about 3 ms here


def test_rank_texts_puts_the_rarer_shared_word_first():
    texts = ["- the stage drifts after a move", "- the stage is slow", "- always blank the beam"]
    assert rank_texts(texts, [("why does the stage drift", 1.0)])[0] == 0
    assert rank_texts(texts, [("nothing in common", 1.0)]) == []


def reply(name, text, call_id, status="success"):
    return [AIMessage("", tool_calls=[{"name": name, "args": {}, "id": call_id}]), ToolMessage(text, tool_call_id=call_id, name=name, status=status)]


def test_the_step_query_is_the_request_and_the_latest_replies():
    messages = [HumanMessage("image the particle")] + reply("a", "first", "1") + reply("b", "second", "2") + reply("c", "Error: third", "3", "error")
    assert step_query(messages) == [("image the particle", 1.0), ("Error: third", 1.0), ("second", 0.6)]


def test_a_failed_call_brings_the_matching_advice_each_step_not_the_whole_manual():
    middleware = SkillMiddleware("BASE", STEM, STEM)
    start = [HumanMessage("look closer at the bright particle")]
    failed = start + reply("place_beam", "Error: coordinates must be in 0..1 range", "c1", status="error")
    update = middleware.before_model({"messages": failed}, None)
    assert any(label.startswith("When a call fails") for label in update["guidance"])
    assert {"skill": "recover_from_errors", "reason": "place_beam failed", "in_prompt": True} in update["skills_applied"]
    tools = [SimpleNamespace(name="place_beam")]
    request = SimpleNamespace(state={"messages": failed, **update}, messages=failed, tools=tools, override=lambda **c: SimpleNamespace(**c))
    out = middleware._prompted(request)
    prompt, last = out.system_message.content, out.messages[-1]
    # The advice rides on the latest tool reply, so the system prompt (the start a server caches) does not change per step.
    assert "Advice that may help" not in prompt and last.type == "tool"
    original, advice = last.content.split("\n\nAdvice that may help with this step")
    assert original == failed[-1].content and "0..1 range" in advice
    assert out.messages[:-1] == failed[:-1] and "Advice" not in failed[-1].content  # the state keeps the reply as it was
    assert advice.count("--- When a call fails") <= 2 and len(advice) < len("".join(p.text for p in passages_of(next(s for s in usable() if s.name == "recover_from_errors"))))
