"""Tests for the skills registry (stdlib only: runs without the agent extra)."""

import sqlite3
from pathlib import Path

import pytest

from asyncroscopy.agent.skills import (
    SkillError,
    SkillRegistry,
    parse_skill_markdown,
    render_skill_markdown,
    render_skills_index,
    skill_functions,
)
from asyncroscopy.agent.skills.frontmatter import Skill

PROJECT_SKILLS = Path(__file__).resolve().parents[1] / "skills"


def _write_skill(base: Path, name: str, description: str, body: str, **extra) -> Path:
    skill_dir = base / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    lines = ["---", f"name: {name}", f"description: {description}", "version: 1.0.0"]
    for key, value in extra.items():
        lines.append(f"{key}: {value}")
    lines += ["---", "", body]
    path = skill_dir / "SKILL.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    base = tmp_path / "skills"
    _write_skill(base, "haadf-imaging", "Acquire a HAADF scanned image.", "# HAADF\nCall acquire_scanned_image with detector_list=['haadf'].", tags="[imaging, stem]")
    _write_skill(base, "eds-spectrum", "Acquire an EDS spectrum and read the composition.", "# EDS\nCall acquire_spectrum with detector_name='eds'. The zebra keyword lives only in this body.", tags="[spectroscopy]")
    _write_skill(base, "stage-moves", "Move the stage safely.", "# Stage\nUse small steps.", triggers='["move stage"]')
    return base


class TestFrontmatter:
    def test_parse_valid(self):
        text = "---\nname: demo\ndescription:  A demo   skill \nversion: 2.0.0\ntags: [a, b]\ntools: ['*_acquire_spectrum']\n---\n\n# Body\nhello\n"
        skill = parse_skill_markdown(text)
        assert skill.name == "demo"
        assert skill.description == "A demo skill"
        assert skill.version == "2.0.0"
        assert skill.tags == ("a", "b")
        assert skill.tools == ("*_acquire_spectrum",)
        assert skill.body == "# Body\nhello"

    def test_missing_name_raises(self):
        with pytest.raises(SkillError, match="name"):
            parse_skill_markdown("---\ndescription: x\n---\nbody")

    def test_missing_description_raises(self):
        with pytest.raises(SkillError, match="description"):
            parse_skill_markdown("---\nname: demo\n---\nbody")

    def test_no_frontmatter_raises(self):
        with pytest.raises(SkillError, match="frontmatter"):
            parse_skill_markdown("# Just markdown\n")

    def test_bad_name_raises(self):
        with pytest.raises(SkillError, match="Invalid skill name"):
            parse_skill_markdown("---\nname: ../evil\ndescription: x\n---\nbody")

    def test_folder_mismatch_raises(self, tmp_path):
        with pytest.raises(SkillError, match="folder"):
            parse_skill_markdown("---\nname: demo\ndescription: x\n---\nbody", path=tmp_path / "other" / "SKILL.md")

    def test_render_round_trip(self):
        skill = Skill(name="demo", description="A demo", version="1.2.3", tags=("a",), triggers=("go",), tools=("t_*",), body="# Hi\n\nSteps.")
        parsed = parse_skill_markdown(render_skill_markdown(skill))
        assert parsed.name == skill.name
        assert parsed.description == skill.description
        assert parsed.version == skill.version
        assert parsed.tags == skill.tags
        assert parsed.triggers == skill.triggers
        assert parsed.tools == skill.tools
        assert parsed.body == skill.body


class TestRegistry:
    def test_list_sorted_and_get(self, skills_dir):
        registry = SkillRegistry.from_dirs(skills_dir)
        assert [s.name for s in registry.list()] == ["eds-spectrum", "haadf-imaging", "stage-moves"]
        assert registry.get("HAADF-Imaging").description.startswith("Acquire a HAADF")
        assert registry.get("nope") is None
        assert len(registry) == 3
        assert registry.errors == []

    def test_missing_directory_is_empty(self, tmp_path):
        registry = SkillRegistry.from_dirs(tmp_path / "missing")
        assert registry.list() == []
        assert registry.search("anything") == []

    def test_search_ranks_by_relevance(self, skills_dir):
        registry = SkillRegistry.from_dirs(skills_dir)
        hits = registry.search("EDS spectrum composition")
        assert hits and hits[0][0].name == "eds-spectrum"

    def test_search_finds_body_only_terms(self, skills_dir):
        registry = SkillRegistry.from_dirs(skills_dir)
        assert [s.name for s, _ in registry.search("zebra")] == ["eds-spectrum"]

    def test_search_tolerates_punctuation_and_no_hits(self, skills_dir):
        registry = SkillRegistry.from_dirs(skills_dir)
        assert registry.search('acquire "haadf" (image)!')[0][0].name == "haadf-imaging"
        assert registry.search("quantum chromodynamics") == []
        assert registry.search("") == []

    def test_search_or_fallback_when_and_has_no_hits(self, skills_dir):
        registry = SkillRegistry.from_dirs(skills_dir)
        names = [s.name for s, _ in registry.search("stage nonexistentword")]
        assert names == ["stage-moves"]

    def test_malformed_skill_is_skipped_with_error(self, skills_dir):
        (skills_dir / "broken").mkdir()
        (skills_dir / "broken" / "SKILL.md").write_text("no frontmatter here", encoding="utf-8")
        with pytest.warns(UserWarning, match="Skipping skill"):
            registry = SkillRegistry.from_dirs(skills_dir)
        assert "broken" not in registry
        assert len(registry.errors) == 1

    def test_save_creates_and_reindexes(self, skills_dir):
        registry = SkillRegistry.from_dirs(skills_dir)
        path = registry.save("quick-survey", "Image then EDS in one go.", "1. image\n2. eds", tags=["survey"], tools=["*_acquire_spectrum"])
        assert path == skills_dir / "quick-survey" / "SKILL.md"
        assert registry.get("quick-survey").tools == ("*_acquire_spectrum",)
        assert registry.search("survey")[0][0].name == "quick-survey"
        assert parse_skill_markdown(path.read_text(encoding="utf-8"), path=path).name == "quick-survey"

    def test_save_refuses_overwrite_unless_asked(self, skills_dir):
        registry = SkillRegistry.from_dirs(skills_dir)
        with pytest.raises(SkillError, match="already exists"):
            registry.save("haadf-imaging", "dup", "body")
        registry.save("haadf-imaging", "Replaced description.", "new body", overwrite=True)
        assert registry.get("haadf-imaging").body == "new body"

    def test_save_rejects_traversal_and_empty_body(self, skills_dir):
        registry = SkillRegistry.from_dirs(skills_dir)
        with pytest.raises(SkillError):
            registry.save("../evil", "x", "body")
        with pytest.raises(SkillError):
            registry.save("Bad Name", "x", "body")
        with pytest.raises(SkillError):
            registry.save("empty", "x", "   ")
        assert not (skills_dir.parent / "evil").exists()

    def test_fts_unavailable_falls_back_to_substring(self, skills_dir, monkeypatch):
        def broken_connect(*args, **kwargs):
            raise sqlite3.OperationalError("no such module: fts5")

        monkeypatch.setattr(sqlite3, "connect", broken_connect)
        with pytest.warns(UserWarning, match="FTS5 unavailable"):
            registry = SkillRegistry.from_dirs(skills_dir)
        assert [s.name for s, _ in registry.search("zebra")] == ["eds-spectrum"]
        assert registry.search("EDS haadf")[0][1] >= 1
        assert registry.search("nothing-here") == []


class TestRenderingAndFunctions:
    def test_index_has_descriptions_not_bodies(self, skills_dir):
        registry = SkillRegistry.from_dirs(skills_dir)
        index = render_skills_index(registry)
        assert "haadf-imaging (v1.0.0): Acquire a HAADF scanned image." in index
        assert "zebra" not in index
        assert "skill_view" in index

    def test_index_for_empty_registry(self, tmp_path):
        assert "No skills" in render_skills_index(SkillRegistry.from_dirs(tmp_path))

    def test_skill_functions(self, skills_dir):
        import json

        registry = SkillRegistry.from_dirs(skills_dir)
        fns = skill_functions(registry)
        assert set(fns) == {"skills_list", "skill_search", "skill_view", "skill_save"}
        listed = json.loads(fns["skills_list"]())
        assert [s["name"] for s in listed] == ["eds-spectrum", "haadf-imaging", "stage-moves"]
        search = json.loads(fns["skill_search"]("zebra"))
        assert search["results"][0]["name"] == "eds-spectrum"
        assert json.loads(fns["skill_search"]("nothing-matches"))["results"] == []
        view = fns["skill_view"]("eds-spectrum")
        assert view.startswith("# eds-spectrum (v1.0.0)") and "zebra" in view
        assert "Unknown skill" in fns["skill_view"]("missing")
        saved = fns["skill_save"]("agent-made", "Made by the agent.", "1. do it", tags=["auto"])
        assert saved.startswith("Saved skill 'agent-made'")
        assert registry.get("agent-made") is not None
        assert "already exists" in fns["skill_save"]("agent-made", "again", "body")


class TestRepositorySkills:
    def test_bundled_skills_parse(self):
        registry = SkillRegistry.from_dirs(PROJECT_SKILLS)
        assert registry.errors == []
        names = registry.names()
        for expected in ("haadf-imaging", "eds-spectrum", "tiled-data-access", "digital-twin-operation", "image-eds-survey"):
            assert expected in names
        assert registry.search("EDS composition")[0][0].name in {"eds-spectrum", "image-eds-survey"}
