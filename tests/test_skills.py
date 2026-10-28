"""Skill discovery, version gating and the MCP skill tools."""

import pytest

from asyncroscopy.skills import list_skills, load_skill


def test_mapping_skill_is_listed_and_loads():
    names = {s.name: s for s in list_skills()}
    assert "mapping" in names and names["mapping"].available
    text = load_skill("mapping")
    assert "Mapper_reserve" in text and not text.startswith("---")


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
    assert not indenter["mapping"].available and "Mapper_reserve" in indenter["mapping"].problem


def test_mcp_skill_tools(monkeypatch):
    from fastmcp.exceptions import ToolError

    from asyncroscopy.mcp.mcp_server import MCPServer

    monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)
    server = MCPServer("t", "h", 1, blocked_classes=[], blocked_functions={}, data_device_address="d", verbose=False)
    with pytest.raises(ToolError, match="Mapper_reserve"):
        server.load_skill("mapping")  # this endpoint has no Mapper
    server.tools = {"Mapper": {"reserve": None}}
    assert any(s["name"] == "mapping" and s["available"] for s in server.list_skills())
    assert "Mapper_reserve" in server.load_skill("mapping")
    with pytest.raises(ToolError):
        server.load_skill("nope")
