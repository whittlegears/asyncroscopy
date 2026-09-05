"""Tests for the store <-> Hermes skill bridges."""

import json

import pytest

from asyncroscopy.skills.hermes_bridge import (
    EXPORT_SUBDIR,
    export_skills,
    import_new_skills,
    resolve_hermes_home,
)
from asyncroscopy.skills.store import SkillRecord


def _record(skill_id, text=None, enabled=True):
    return SkillRecord(
        id=skill_id,
        name=skill_id.replace("-", " "),
        description=f"about {skill_id}",
        text=text if text is not None else f"---\nname: {skill_id}\ndescription: d\n---\n\nBody.",
        enabled=enabled,
    )


class TestResolveHermesHome:
    def test_explicit_path_wins(self, tmp_path):
        assert resolve_hermes_home(str(tmp_path)) == tmp_path

    def test_explicit_missing_path_never_falls_back(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert resolve_hermes_home(str(tmp_path / "missing")) is None

    def test_env_fallback(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert resolve_hermes_home() == tmp_path


class TestExportSkills:
    def test_writes_enabled_skills(self, tmp_path):
        report = export_skills([_record("focus-sweep")], tmp_path)
        target = tmp_path / EXPORT_SUBDIR / "focus-sweep" / "SKILL.md"
        assert report == {"exported": 1, "removed": 0}
        assert "Body." in target.read_text(encoding="utf-8")

    def test_skips_disabled_and_prunes_removed(self, tmp_path):
        export_skills([_record("old-skill")], tmp_path)
        report = export_skills([_record("old-skill", enabled=False), _record("new-skill")], tmp_path)
        assert report == {"exported": 1, "removed": 1}
        assert not (tmp_path / EXPORT_SUBDIR / "old-skill").exists()
        assert (tmp_path / EXPORT_SUBDIR / "new-skill" / "SKILL.md").is_file()

    def test_adds_frontmatter_when_missing(self, tmp_path):
        export_skills([_record("bare", text="Just a body, no frontmatter.")], tmp_path)
        text = (tmp_path / EXPORT_SUBDIR / "bare" / "SKILL.md").read_text(encoding="utf-8")
        assert text.startswith("---\nname: bare\n")
        assert "Just a body, no frontmatter." in text

    def test_keeps_existing_frontmatter(self, tmp_path):
        original = "---\nname: styled\ndescription: keep me\n---\n\nBody."
        export_skills([_record("styled", text=original)], tmp_path)
        assert (tmp_path / EXPORT_SUBDIR / "styled" / "SKILL.md").read_text(encoding="utf-8") == original


class _Service:
    def __init__(self):
        self.proposals = []

    def propose_skill(self, name, content):
        self.proposals.append((name, content))
        return "id"


def _write(skills_dir, category, name, body="Do it."):
    skill_dir = skills_dir / category / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: d\n---\n\n{body}", encoding="utf-8"
    )


class TestImportNewSkills:
    def test_first_scan_is_a_silent_baseline(self, tmp_path):
        skills, state = tmp_path / "skills", tmp_path / "state.json"
        _write(skills, "bundled", "pre-existing")
        service = _Service()
        assert import_new_skills(skills, service, state) == []
        assert service.proposals == []
        assert json.loads(state.read_text(encoding="utf-8"))

    def test_new_skill_is_proposed(self, tmp_path):
        skills, state = tmp_path / "skills", tmp_path / "state.json"
        _write(skills, "bundled", "pre-existing")
        service = _Service()
        import_new_skills(skills, service, state)
        _write(skills, "microscopy", "focus-sweep")
        assert import_new_skills(skills, service, state) == ["microscopy/focus-sweep"]
        assert service.proposals[0][0] == "focus-sweep"

    def test_changed_skill_is_reproposed_once(self, tmp_path):
        skills, state = tmp_path / "skills", tmp_path / "state.json"
        _write(skills, "bundled", "thing")
        service = _Service()
        import_new_skills(skills, service, state)
        _write(skills, "bundled", "thing", body="Updated.")
        assert import_new_skills(skills, service, state) == ["bundled/thing"]
        assert import_new_skills(skills, service, state) == []

    def test_export_subdir_is_ignored(self, tmp_path):
        skills, state = tmp_path / "skills", tmp_path / "state.json"
        skills.mkdir()
        service = _Service()
        import_new_skills(skills, service, state)
        _write(skills, EXPORT_SUBDIR, "from-the-store")
        assert import_new_skills(skills, service, state) == []

    def test_corrupt_state_recovers_as_baseline_rescan(self, tmp_path):
        skills, state = tmp_path / "skills", tmp_path / "state.json"
        _write(skills, "bundled", "thing")
        state.write_text("not json", encoding="utf-8")
        service = _Service()
        import_new_skills(skills, service, state)
        assert [name for name, _ in service.proposals] == ["thing"]
        assert import_new_skills(skills, service, state) == []
