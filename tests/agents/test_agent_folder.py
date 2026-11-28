"""The agent/ folder: skills and workflows are read from settings.agent_dir, and each skill's enabled switch."""

from pathlib import Path

import pytest

from asyncroscopy import settings
from asyncroscopy.agents.learning import lint_skill
from asyncroscopy.skills.registry import list_skills, load_skill, select_skills
from asyncroscopy.workflows import all_workflows

ROOT = Path(__file__).resolve().parents[2]
AGENT = ROOT / "agent"


def write_skill(folder: Path, name: str, enabled: str | None = None) -> None:
    (folder / name).mkdir(parents=True)
    switch = f"enabled: {enabled}\n" if enabled is not None else ""
    (folder / name / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: the {name} task\n{switch}keywords: [{name} thing]\nacts: false\n---\nDo the {name} thing.",
        encoding="utf-8")


@pytest.fixture
def checkout(monkeypatch):
    monkeypatch.delenv("ASYNCROSCOPY_AGENT_DIR", raising=False)
    yield
    monkeypatch.delenv("ASYNCROSCOPY_AGENT_DIR", raising=False)
    all_workflows()


def test_skills_and_workflows_come_from_the_agent_folder_of_the_checkout(checkout):
    assert settings.get().agent_dir == AGENT
    shipped = {s.name: s for s in list_skills() if s.source == "shipped"}
    assert set(shipped) == {p.parent.name for p in (AGENT / "skills").glob("*/SKILL.md")}
    assert all(s.path.parent.parent == AGENT / "skills" for s in shipped.values())
    assert not (ROOT / "asyncroscopy" / "skills" / "mapping").exists() and not (ROOT / "asyncroscopy" / "workflows" / "graphs").exists()


def test_every_shipped_workflow_file_registers(checkout):
    files = sorted((AGENT / "workflows").glob("*.py"))
    registered = all_workflows()
    sources = {w.source(): w for w in registered.values()}
    assert len(files) == 6
    for path in files:
        workflow = sources[path.read_text(encoding="utf-8")]
        assert workflow.blueprint()["nodes"] and workflow.hash()
        assert "could not be built" not in " ".join(workflow.lint()["problems"])


def test_an_agent_dir_elsewhere_replaces_the_shipped_skills_and_workflows(checkout, monkeypatch, tmp_path):
    write_skill(tmp_path / "skills", "elsewhere")
    (tmp_path / "workflows").mkdir()
    grid = (AGENT / "workflows" / "map_grid.py").read_text(encoding="utf-8").replace('name="map_grid"', 'name="tiny_grid"')
    (tmp_path / "workflows" / "tiny.py").write_text(grid, encoding="utf-8")
    monkeypatch.setenv("ASYNCROSCOPY_AGENT_DIR", str(tmp_path))
    assert [(s.name, s.source) for s in list_skills() if s.source == "shipped"] == [("elsewhere", "shipped")]
    assert load_skill("elsewhere") == "Do the elsewhere thing."
    found = all_workflows()
    assert "tiny_grid" in found and "map_grid" not in found and found["tiny_grid"].source() == grid
    monkeypatch.delenv("ASYNCROSCOPY_AGENT_DIR")
    found = all_workflows()
    assert "map_grid" in found and "tiny_grid" not in found


def test_a_skill_with_enabled_false_is_listed_as_disabled_and_given_to_no_agent(monkeypatch, tmp_path):
    write_skill(tmp_path, "quiet", "false")
    write_skill(tmp_path, "loud", "true")
    write_skill(tmp_path, "plain")
    monkeypatch.setenv("ASYNCROSCOPY_SKILLS_PATH", str(tmp_path))
    skills = {s.name: s for s in list_skills()}
    assert not skills["quiet"].enabled and not skills["quiet"].available and "disabled" in skills["quiet"].problem
    assert skills["loud"].available and skills["plain"].available and skills["plain"].enabled
    assert [s.name for s in select_skills("do the quiet thing", set())] == []
    assert [s.name for s in select_skills("do the loud thing", set())] == ["loud"]
    with pytest.raises(RuntimeError, match="disabled"):
        load_skill("quiet")


def test_the_mcp_skill_list_shows_a_disabled_skill(monkeypatch, tmp_path):
    from fastmcp.exceptions import ToolError

    from asyncroscopy.mcp.mcp_server import MCPServer

    write_skill(tmp_path, "quiet", "false")
    monkeypatch.setenv("ASYNCROSCOPY_SKILLS_PATH", str(tmp_path))
    monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)
    server = MCPServer("t", "h", 1, blocked_classes=[], needs_approval={}, data_device_address="d", verbose=False)
    listed = {s["name"]: s for s in server.skills.list_skills()}
    assert listed["quiet"]["enabled"] is False and listed["quiet"]["available"] is False
    assert server.skills.skill_text("quiet")["enabled"] is False
    assert "quiet" not in server.skills.load_skill_description()
    with pytest.raises(ToolError, match="disabled"):
        server.skills.load_skill("quiet")


def test_the_environment_switches_come_before_each_skills_enabled(monkeypatch, tmp_path):
    write_skill(tmp_path, "loud", "true")
    write_skill(tmp_path, "quiet", "false")
    monkeypatch.setenv("ASYNCROSCOPY_SKILLS_PATH", str(tmp_path))
    monkeypatch.setenv("ASYNCROSCOPY_SKILLS_OFF", "lou*,quiet")
    names = {s.name for s in list_skills()}
    assert not names & {"loud", "quiet"} and "mapping" in names
    monkeypatch.delenv("ASYNCROSCOPY_SKILLS_OFF")
    monkeypatch.setenv("ASYNCROSCOPY_SKILLS", "0")
    assert list_skills() == [] and select_skills("do the loud thing", set()) == []


def test_enabled_must_be_true_or_false(tmp_path):
    write_skill(tmp_path, "unsure", '"no"')
    assert "'enabled' must be true or false" in lint_skill(tmp_path / "unsure" / "SKILL.md")
    assert lint_skill(AGENT / "skills" / "focus" / "SKILL.md") == []
