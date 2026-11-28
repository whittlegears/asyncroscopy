"""Skill discovery, version gating and the MCP skill tools."""

import pytest

from asyncroscopy.skills import list_skills, load_skill, skills_text


def test_mapping_skill_is_listed_and_loads():
    names = {s.name: s for s in list_skills()}
    assert "mapping" in names and names["mapping"].available
    text = load_skill("mapping")
    assert "run_job" in text and "wait_for_job" in text and not text.startswith("---")


def test_unknown_skill_names_the_available_ones():
    with pytest.raises(KeyError, match="mapping"):
        load_skill("does-not-exist")


def test_extra_skill_folder_and_version_gate(tmp_path, monkeypatch):
    (tmp_path / "future").mkdir()
    (tmp_path / "future" / "SKILL.md").write_text("---\nname: future\ndescription: needs a newer install\nmin_asyncroscopy: 99.0.0\n---\nbody", encoding="utf-8")
    (tmp_path / "local").mkdir()
    (tmp_path / "local" / "SKILL.md").write_text("---\nname: local\ndescription: a lab's own skill\n---\nDo the thing.", encoding="utf-8")
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "SKILL.md").write_text("no front matter", encoding="utf-8")
    monkeypatch.setenv("ASYNCROSCOPY_SKILLS_PATH", str(tmp_path))
    skills = {s.name: s for s in list_skills()}
    assert skills["local"].available and load_skill("local") == "Do the thing."
    assert not skills["future"].available and "99.0.0" in skills["future"].problem
    assert not skills["broken"].available
    with pytest.raises(RuntimeError, match="cannot be used"):
        load_skill("future")


def test_a_labs_skill_replaces_the_packaged_one(tmp_path, monkeypatch):
    (tmp_path / "mapping").mkdir()
    (tmp_path / "mapping" / "SKILL.md").write_text("---\nname: mapping\ndescription: our lab's version\n---\nours", encoding="utf-8")
    monkeypatch.setenv("ASYNCROSCOPY_SKILLS_PATH", str(tmp_path))
    assert [s.description for s in list_skills() if s.name == "mapping"] == ["our lab's version"]


def test_skill_needing_absent_tools_is_not_offered():
    stem = {s.name: s for s in list_skills({"Mapper_reserve", "Mapper_start"})}
    indenter = {s.name: s for s in list_skills({"IndentGrid_reserve", "SimIndenter_indent"})}
    assert stem["mapping"].available
    assert not indenter["mapping"].available and "Mapper*_reserve" in indenter["mapping"].problem


def test_mcp_skill_tools(monkeypatch):
    from fastmcp.exceptions import ToolError

    from asyncroscopy.mcp.mcp_server import MCPServer

    monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)
    server = MCPServer("t", "h", 1, blocked_classes=[], needs_approval={}, data_device_address="d", verbose=False)
    with pytest.raises(ToolError, match=r"Mapper\*_reserve"):
        server.skills.load_skill("mapping")  # this endpoint has no Mapper
    server.tools = {"Mapper": {"reserve": None}}
    assert any(s["name"] == "mapping" and s["available"] for s in server.skills.list_skills())
    assert "run_job" in server.skills.load_skill("mapping")
    with pytest.raises(ToolError):
        server.skills.load_skill("nope")
    assert "Skills here at startup: " in server.skills.load_skill_description() and "mapping" in server.skills.load_skill_description()
    server.tools = {}
    assert "mapping" not in server.skills.load_skill_description()  # not offered where it cannot run


def test_chosen_skills_go_into_the_prompt_in_full_while_they_fit(tmp_path, monkeypatch):
    for name, body in (("alpha", "A" * 300), ("beta", "B" * 300)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "SKILL.md").write_text(f"---\nname: {name}\ndescription: the {name} task\n---\n{body}", encoding="utf-8")
    monkeypatch.setattr("asyncroscopy.skills.registry.skill_dirs", lambda: [tmp_path])
    text, cut = skills_text(["alpha", "beta"], set(), budget=1000)
    assert not cut and "A" * 300 in text and "B" * 300 in text
    text, cut = skills_text(["alpha", "beta"], set(), budget=400)
    assert cut == ["beta"] and "A" * 300 in text and "the beta task" in text and "B" * 300 not in text and "load_skill" in text
    assert skills_text(["nope"], set()) == ("", [])


def test_skills_named_in_skills_off_are_offered_to_no_one(monkeypatch):
    from asyncroscopy.skills.registry import list_skills, select_skills

    tools = {"Mapper_reserve", "run_job", "wait_for_job"}
    assert "mapping" in [s.name for s in select_skills("Map a 600 nm square here.", tools)]
    monkeypatch.setenv("ASYNCROSCOPY_SKILLS_OFF", "mapping, multiscale_*")
    names = {s.name for s in list_skills()}
    assert not names & {"mapping", "multiscale_region"} and "inspect_data" in names
    assert "mapping" not in [s.name for s in select_skills("Map a 600 nm square here.", tools)]


def test_with_skills_off_nothing_is_listed_matched_or_put_in_a_prompt(monkeypatch):
    from langchain_core.messages import HumanMessage

    from asyncroscopy.agents.skill_middleware import SkillMiddleware, skills_for
    from asyncroscopy.mcp.components.skills_tools import SkillsTools
    from asyncroscopy.skills.registry import list_skills, select_skills

    tools = {"Mapper_reserve", "run_job", "wait_for_job", "load_skill"}
    monkeypatch.setenv("ASYNCROSCOPY_SKILLS", "0")
    assert list_skills() == [] and select_skills("Map a 600 nm square here.", tools) == []
    messages = [HumanMessage("Map a 600 nm square here.")]
    assert skills_for(messages, tools) == []
    assert SkillMiddleware("BASE", tools).before_model({"messages": messages, "skills_applied": []}, None) is None
    assert SkillsTools(None, None).natives() == []
    monkeypatch.delenv("ASYNCROSCOPY_SKILLS")
    assert list_skills() == []  # skills are off by default
